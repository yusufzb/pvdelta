"""Small, inventory-independent analysis orchestration used by Stage 1."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from pvdelta.conditions import ConditionMatch, ConditionObservation, match_conditions
from pvdelta.config import DetectionConfig
from pvdelta.detect import assess_inter_module_anomalies, assess_intra_module_anomalies
from pvdelta.models import Finding, ModuleObservation, ThermalFrame
from pvdelta.normalize import Extent
from pvdelta.quality import Capability, QualityAssessment
from pvdelta.reference import erode
from pvdelta.regions import ModuleAnnotation, polygon_to_mask
from pvdelta.report import ModuleReport, ReportArtifacts, write_stage1_report
from pvdelta.store import (
    CaptureRecord,
    Stage1StoreSummary,
    connect_store,
    upsert_capture,
    upsert_run,
    write_stage1_results,
)


@dataclass(frozen=True)
class ModuleAnalysis:
    findings: tuple[Finding, ...]
    quality: QualityAssessment
    condition_match: ConditionMatch


@dataclass(frozen=True)
class Stage1Execution:
    module_results: tuple[ModuleReport, ...]
    report_artifacts: ReportArtifacts
    store_summary: Stage1StoreSummary | None = None


def analyze_module(
    frame: ThermalFrame,
    module: ModuleObservation,
    module_mask: np.ndarray,
    *,
    condition_observations: list[ConditionObservation] | None = None,
    condition_scope_id: str = "plant",
    max_condition_age: dt.timedelta = dt.timedelta(seconds=30),
    extent: Extent | None = None,
    config: DetectionConfig | None = None,
    has_camera_pose: bool = False,
    has_calibrated_geometry: bool = False,
) -> ModuleAnalysis:
    matched = match_conditions(
        frame.capture_time, condition_observations or [],
        scope_id=condition_scope_id, max_age=max_condition_age,
    )
    assessment = assess_intra_module_anomalies(
        frame, module, module_mask, config=config,
        conditions=list(matched.conditions), extent=extent,
    )
    quality = QualityAssessment(
        Capability.UNAVAILABLE if assessment.reasons else Capability.AVAILABLE,
        Capability.AVAILABLE if matched.conditions else Capability.UNAVAILABLE,
        Capability.AVAILABLE if has_camera_pose and has_calibrated_geometry else Capability.UNAVAILABLE,
        assessment.reasons,
    )
    findings = assessment.findings
    source_flags = [f"condition source: {source}" for source in matched.sources]
    enriched = tuple(replace(
        finding,
        normalization_sources=list(matched.sources),
        normalization_flags=[
            *finding.normalization_flags, *matched.flags, *source_flags,
        ],
    ) for finding in findings)
    return ModuleAnalysis(enriched, quality, matched)


def analyze_annotations(
    frame: ThermalFrame,
    annotations: list[ModuleAnnotation],
    *,
    region_margin_px: int = 0,
    config: DetectionConfig | None = None,
    direction: Literal["hot", "cold", "both"] = "both",
    module_masks: dict[str, np.ndarray] | None = None,
) -> tuple[ModuleReport, ...]:
    """Analyze explicit native-grid regions while preserving rejected module coverage."""
    from pvdelta.surface_constraints import apply_configured_surface

    frame=apply_configured_surface(frame)
    if region_margin_px < 0:
        raise ValueError("region_margin_px must be non-negative")
    if direction not in {"hot", "cold", "both"}:
        raise ValueError("direction must be hot, cold or both")
    cfg = config or DetectionConfig()
    if len({a.module_id for a in annotations}) != len(annotations):
        raise ValueError("duplicate module IDs in annotations")
    results = []
    accepted_masks = {}
    for annotation in annotations:
        polygon = list(annotation.polygon)
        base_module = ModuleObservation(
            annotation.module_id,
            frame.frame_id,
            polygon,
            visibility=annotation.visibility,
        )
        try:
            if module_masks is None:
                envelope = polygon_to_mask(polygon, frame.temperature_c.shape)
                measurement = erode(envelope, region_margin_px)
            else:
                if region_margin_px:
                    raise ValueError(
                        "region_margin_px must be zero when explicit measurement masks are supplied"
                    )
                measurement = module_masks[annotation.module_id]
                if (
                    measurement.shape != frame.temperature_c.shape
                    or measurement.dtype != np.bool_
                ):
                    raise ValueError("explicit module mask must be boolean and match the frame")
                envelope = polygon_to_mask(polygon, frame.temperature_c.shape)
                if np.any(measurement & ~envelope):
                    raise ValueError("explicit measurement mask extends outside module polygon")
        except (KeyError, ValueError) as exc:
            results.append(
                ModuleReport(base_module, assessment_status="not_assessed", reasons=(str(exc),))
            )
            continue
        valid = measurement & np.isfinite(frame.temperature_c)
        if frame.valid_mask is not None:
            valid &= frame.valid_mask
        if valid.sum() < cfg.min_reference_px + cfg.min_region_px:
            results.append(
                ModuleReport(
                    base_module,
                    assessment_status="not_assessed",
                    reasons=("too few valid native pixels after the region margin",),
                )
            )
            continue
        values = frame.temperature_c[valid]
        module = replace(
            base_module,
            temp_min=float(np.min(values)),
            temp_max=float(np.max(values)),
            temp_mean=float(np.mean(values)),
            temp_median=float(np.median(values)),
            quality_flags=(
                ["partial module observation"] if annotation.visibility == "partial" else []
            ),
        )
        accepted_masks[annotation.module_id] = measurement
        analysis = analyze_module(frame, module, measurement, config=cfg)
        if analysis.quality.image_analysis is Capability.UNAVAILABLE:
            results.append(
                ModuleReport(
                    module,
                    assessment_status="not_assessed",
                    reasons=tuple(analysis.quality.reasons),
                )
            )
            continue
        safe = erode(measurement, cfg.edge_erosion_px)
        safe_boundary = safe & ~erode(safe, 1)
        selected = []
        for finding in analysis.findings:
            finding_direction = "hot" if "hot" in finding.anomaly_type else "cold"
            if direction != "both" and direction != finding_direction:
                continue
            boundary_sensitive = False
            if finding.region_pixels_yx:
                points = np.asarray(finding.region_pixels_yx, dtype=int)
                boundary_sensitive = bool(
                    np.any(safe_boundary[points[:, 0], points[:, 1]])
                )
            if boundary_sensitive and cfg.reject_boundary_findings:
                continue
            selected.append(
                replace(
                    finding,
                    assessment_status=(
                        "review_required" if boundary_sensitive else finding.assessment_status
                    ),
                )
            )
        reasons = (
            ("one or more findings touch the effective measurement boundary",)
            if any(item.assessment_status == "review_required" for item in selected)
            else ("edge-touching candidates excluded by the measurement mask",)
            if any(
                finding.region_pixels_yx
                and np.any(
                    safe_boundary[
                        np.asarray(finding.region_pixels_yx, dtype=int)[:, 0],
                        np.asarray(finding.region_pixels_yx, dtype=int)[:, 1],
                    ]
                )
                for finding in analysis.findings
            ) and cfg.reject_boundary_findings
            else ()
        )
        results.append(ModuleReport(module, tuple(selected), reasons=reasons))
    # Shared pixels cannot be assigned to two physical modules without review.
    occupancy = np.zeros(frame.temperature_c.shape, dtype=np.int32)
    for mask in accepted_masks.values():
        occupancy += mask
    overlaps = {mid for mid, mask in accepted_masks.items() if np.any(mask & (occupancy > 1))}
    results = [ModuleReport(r.module, assessment_status='not_assessed',
                            reasons=('measurement mask overlaps another module',))
               if r.module.module_id in overlaps else r for r in results]
    groups = {a.module_id: a.peer_group for a in annotations if a.peer_group and a.visibility == 'full'}
    eligible = [r.module for r in results if r.assessment_status == 'analyzed'
                and r.module.module_id in groups]
    peer_assessment = assess_inter_module_anomalies(
        frame, eligible, accepted_masks, {m.module_id: groups[m.module_id] for m in eligible},
        config=cfg,
    )
    enriched = []
    for result in results:
        if result.assessment_status != 'analyzed':
            enriched.append(result)
            continue
        mid = result.module.module_id
        extra = tuple(f for f in peer_assessment.findings if f.module_id == mid
                      and (direction == 'both' or direction in f.anomaly_type))
        reasons = list(result.reasons)
        if mid not in groups:
            reasons.append('whole-module comparison unavailable: no full-module explicit peer group')
        elif groups[mid] in peer_assessment.insufficient_groups:
            reasons.append('whole-module comparison unavailable: insufficient or inconsistent peers')
        enriched.append(replace(result, findings=(*result.findings, *extra), reasons=tuple(reasons)))
    results = enriched
    return tuple(results)


def execute_stage1(
    frame: ThermalFrame,
    annotations: list[ModuleAnnotation],
    *,
    output_dir: str | Path,
    run_id: str,
    region_margin_px: int = 0,
    config: DetectionConfig | None = None,
    direction: Literal["hot", "cold", "both"] = "both",
    database_path: str | Path | None = None,
    site_label: str | None = None,
    overwrite: bool = False,
    module_masks: dict[str, np.ndarray] | None = None,
) -> Stage1Execution:
    """Analyze once, then write matching report and optional DuckDB records."""
    from pvdelta.measurement_profile import frame_measurement_profile
    from pvdelta.surface_constraints import apply_configured_surface

    frame=apply_configured_surface(frame)
    profile = frame_measurement_profile(frame)
    cfg = config or DetectionConfig()
    modules = analyze_annotations(
        frame,
        annotations,
        region_margin_px=region_margin_px,
        config=cfg,
        direction=direction,
        module_masks=module_masks,
    )
    settings = {
        "detection": asdict(cfg),
        "region_margin_px": region_margin_px,
        "direction": direction,
        "measurement_support": (
            "explicit_native_masks" if module_masks is not None else "annotation_polygons"
        ),
    }
    if frame.raw_metadata.get('pvdelta_surface_constraint'):
        settings['surface_constraint']=frame.raw_metadata['pvdelta_surface_constraint']
    artifacts = write_stage1_report(
        output_dir,
        run_id,
        frame,
        modules,
        settings=settings,
        overwrite=overwrite,
    )
    store_summary = None
    if database_path is not None:
        connection = connect_store(database_path)
        capture_id = f"sha256:{frame.source_hash}"
        try:
            upsert_run(
                connection,
                run_id=run_id,
                site_label=site_label,
                started_at_utc=frame.capture_time,
                camera_model=(
                    None if frame.camera_profile is None else frame.camera_profile.model
                ),
                source_root=str(Path(frame.source_path).parent),
            )
            exists = connection.execute(
                "SELECT count(*) FROM captures WHERE capture_id = ?", [capture_id]
            ).fetchone()[0]
            if not exists:
                finite = frame.temperature_c[np.isfinite(frame.temperature_c)]
                # Capture-level summaries describe the original measurement, never
                # whichever corrected run happened to be inserted first.
                source_values = finite if not profile['requested_overrides'] else np.array([])
                recorded = profile['recorded_parameters'] or {}
                upsert_capture(
                    connection,
                    CaptureRecord(
                        capture_id=capture_id,
                        source_path=frame.source_path,
                        source_hash=frame.source_hash,
                        image_kind="thermal",
                        run_id=run_id,
                        capture_time_utc=frame.capture_time,
                        camera_model=(
                            None if frame.camera_profile is None else frame.camera_profile.model
                        ),
                        width_px=frame.width,
                        height_px=frame.height,
                        temperature_min_c=float(np.min(source_values)) if source_values.size else None,
                        temperature_max_c=float(np.max(source_values)) if source_values.size else None,
                        temperature_mean_c=float(np.mean(source_values)) if source_values.size else None,
                        temperature_median_c=float(np.median(source_values)) if source_values.size else None,
                        object_distance_m=recorded.get('distance_m'),
                        relative_humidity_pct=recorded.get('humidity_pct'),
                        emissivity=recorded.get('emissivity'),
                        atmospheric_temperature_c=recorded.get('ambient_temp_c'),
                        reflected_apparent_temperature_c=recorded.get('reflected_temp_c'),
                        warnings=("capture inserted by Stage 1 analysis; metadata ingest not run",),
                    ),
                )
            store_summary = write_stage1_results(
                connection,
                analysis_run_id=run_id,
                capture_id=capture_id,
                module_results=modules,
                measurement_profile=profile,
            )
        finally:
            connection.close()
    return Stage1Execution(modules, artifacts, store_summary)
