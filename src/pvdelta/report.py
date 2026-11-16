"""Deterministic, inventory-independent Stage 1 report artifacts."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any

from pvdelta.contracts import dumps_strict
from pvdelta.measurement_profile import frame_measurement_profile
from pvdelta.models import Finding, ModuleObservation, ThermalFrame

SCHEMA_VERSION = "pvdelta.stage1-report/v1"
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ModuleReport:
    """One module's assessment outcome, including zero-candidate outcomes."""

    module: ModuleObservation
    findings: tuple[Finding, ...] = ()
    assessment_status: str = "analyzed"
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.assessment_status not in {"analyzed", "not_assessed"}:
            raise ValueError("assessment_status must be analyzed or not_assessed")
        if self.assessment_status == "not_assessed" and self.findings:
            raise ValueError("a not-assessed module cannot contain findings")
        for finding in self.findings:
            if finding.module_id != self.module.module_id:
                raise ValueError("finding and module IDs differ")
            if finding.frame_id != self.module.frame_id:
                raise ValueError("finding and module frame IDs differ")


@dataclass(frozen=True)
class ReportArtifacts:
    jsonl_path: Path
    csv_path: Path
    overlay_path: Path


def _run_record(
    run_id: str,
    frame: ThermalFrame,
    modules: tuple[ModuleReport, ...],
    settings: dict[str, Any],
) -> dict[str, Any]:
    finding_count = sum(len(item.findings) for item in modules)
    analyzed = sum(item.assessment_status == "analyzed" for item in modules)
    return {
        "record_type": "run",
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "frame_id": frame.frame_id,
        "source_name": Path(frame.source_path).name,
        "source_hash": frame.source_hash,
        "native_width": frame.width,
        "native_height": frame.height,
        "module_count": len(modules),
        "analyzed_module_count": analyzed,
        "not_assessed_module_count": len(modules) - analyzed,
        "finding_count": finding_count,
        "settings": settings,
        "radiometry": frame_measurement_profile(frame),
    }


def _jsonl_text(
    run_id: str,
    frame: ThermalFrame,
    modules: tuple[ModuleReport, ...],
    settings: dict[str, Any],
) -> str:
    records: list[dict[str, Any]] = [_run_record(run_id, frame, modules, settings)]
    for result in modules:
        module_status = (
            "not_assessed"
            if result.assessment_status == "not_assessed"
            else ("candidate" if result.findings else "no_candidate")
        )
        records.append(
            {
                "record_type": "module",
                "run_id": run_id,
                "frame_id": frame.frame_id,
                "module_id": result.module.module_id,
                "assessment_status": module_status,
                "reasons": list(result.reasons),
                "module": asdict(result.module),
            }
        )
        records.extend(
            {
                "record_type": "finding",
                "run_id": run_id,
                "frame_id": frame.frame_id,
                "module_id": result.module.module_id,
                "finding": asdict(finding),
            }
            for finding in result.findings
        )
    return "".join(f"{dumps_strict(record)}\n" for record in records)


_CSV_FIELDS = (
    "schema_version",
    "run_id",
    "frame_id",
    "source_name",
    "source_hash",
    "module_id",
    "module_status",
    "module_visibility",
    "reasons",
    "finding_id",
    "anomaly_type",
    "delta_t_c",
    "absolute_t_c",
    "reference_t_c",
    "reference_method",
    "n_region_px",
    "n_reference_px",
    "normalized_delta_t_c",
    "normalization_basis",
    "finding_assessment_status",
    "pixel_polygon",
    "measurement_profile_id",
    "native_temperature_sha256",
    "measurement_distance_m",
    "measurement_emissivity",
    "measurement_humidity_pct",
    "measurement_ambient_temp_c",
    "measurement_reflected_temp_c",
)


def _csv_rows(
    run_id: str,
    frame: ThermalFrame,
    modules: tuple[ModuleReport, ...],
) -> Iterable[dict[str, Any]]:
    profile = frame_measurement_profile(frame)
    params = profile['effective_parameters'] or {}
    common = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "frame_id": frame.frame_id,
        "source_name": Path(frame.source_path).name,
        "source_hash": frame.source_hash,
        "measurement_profile_id": profile['profile_id'],
        "native_temperature_sha256": profile['native_temperature_sha256'],
        "measurement_distance_m": params.get('distance_m'),
        "measurement_emissivity": params.get('emissivity'),
        "measurement_humidity_pct": params.get('humidity_pct'),
        "measurement_ambient_temp_c": params.get('ambient_temp_c'),
        "measurement_reflected_temp_c": params.get('reflected_temp_c'),
    }
    for result in modules:
        module_status = (
            "not_assessed"
            if result.assessment_status == "not_assessed"
            else ("candidate" if result.findings else "no_candidate")
        )
        base = {
            **common,
            "module_id": result.module.module_id,
            "module_status": module_status,
            "module_visibility": result.module.visibility,
            "reasons": " | ".join(result.reasons),
        }
        if not result.findings:
            yield base
            continue
        for finding in result.findings:
            yield {
                **base,
                "finding_id": finding.finding_id,
                "anomaly_type": finding.anomaly_type,
                "delta_t_c": finding.delta_t_c,
                "absolute_t_c": finding.absolute_t_c,
                "reference_t_c": finding.reference_t_c,
                "reference_method": finding.reference_method,
                "n_region_px": finding.n_region_px,
                "n_reference_px": finding.n_reference_px,
                "normalized_delta_t_c": finding.normalized_delta_t_c,
                "normalization_basis": finding.normalization_basis,
                "finding_assessment_status": finding.assessment_status,
                "pixel_polygon": json.dumps(finding.pixel_polygon, separators=(",", ":")),
            }


def _csv_text(
    run_id: str,
    frame: ThermalFrame,
    modules: tuple[ModuleReport, ...],
) -> str:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(_csv_rows(run_id, frame, modules))
    return output.getvalue()


def _points(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:g},{y:g}" for x, y in points)


def _support_path(finding: Finding) -> str | None:
    if not finding.region_pixels_yx:
        return None
    return " ".join(f"M{x} {y}h1v1h-1z" for y, x in finding.region_pixels_yx)


def _overlay_svg(frame: ThermalFrame, modules: tuple[ModuleReport, ...]) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {frame.width} '
            f'{frame.height}" width="{frame.width}" height="{frame.height}">'
        ),
        '<rect width="100%" height="100%" fill="#101827"/>',
        '<g fill="none" stroke="#4ade80" stroke-width="1">',
    ]
    for result in modules:
        if result.module.pixel_polygon:
            parts.append(f'<polygon points="{_points(result.module.pixel_polygon)}"/>')
    parts.append("</g>")
    for result in modules:
        for finding in result.findings:
            colour = "#ef4444" if "hot" in finding.anomaly_type else "#38bdf8"
            support = _support_path(finding)
            if support:
                parts.append(
                    f'<path d="{support}" fill="{colour}" fill-opacity="0.55" '
                    f'stroke="{colour}" stroke-width="0.25"/>'
                )
            elif finding.pixel_polygon:
                parts.append(
                    f'<polygon points="{_points(finding.pixel_polygon)}" fill="none" '
                    f'stroke="{colour}" stroke-width="2"/>'
                )
            if finding.pixel_polygon:
                x, y = finding.pixel_polygon[0]
                label = escape(f"{finding.module_id} {finding.delta_t_c:+.1f} K")
                parts.append(
                    f'<text x="{x:g}" y="{max(10.0, y - 3):g}" fill="{colour}" '
                    f'font-family="sans-serif" font-size="9">{label}</text>'
                )
    parts.append("</svg>\n")
    return "\n".join(parts)


def write_stage1_report(
    output_dir: str | Path,
    run_id: str,
    frame: ThermalFrame,
    module_results: Iterable[ModuleReport],
    *,
    settings: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> ReportArtifacts:
    """Write JSONL provenance, CSV summary and native-grid SVG overlay.

    The SVG is a coordinate overlay, not a temperature image. Consumers may place it over a
    correctly registered native-grid preview. Source paths are reduced to their filename; the
    source hash remains the traceability key.
    """
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must use letters, digits, dot, underscore or hyphen")
    modules = tuple(module_results)
    module_ids = [item.module.module_id for item in modules]
    if len(module_ids) != len(set(module_ids)):
        raise ValueError("module IDs must be unique within a frame report")
    if any(item.module.frame_id != frame.frame_id for item in modules):
        raise ValueError("every module must belong to the reported frame")

    destination = Path(output_dir)
    artifacts = ReportArtifacts(
        destination / f"{run_id}.jsonl",
        destination / f"{run_id}.csv",
        destination / f"{run_id}-overlay.svg",
    )
    targets = (artifacts.jsonl_path, artifacts.csv_path, artifacts.overlay_path)
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"report output exists: {existing[0]}")

    jsonl = _jsonl_text(run_id, frame, modules, settings or {})
    csv_text = _csv_text(run_id, frame, modules)
    svg = _overlay_svg(frame, modules)
    destination.mkdir(parents=True, exist_ok=True)
    artifacts.jsonl_path.write_text(jsonl, encoding="utf-8", newline="\n")
    artifacts.csv_path.write_text(csv_text, encoding="utf-8", newline="")
    artifacts.overlay_path.write_text(svg, encoding="utf-8", newline="\n")
    return artifacts
