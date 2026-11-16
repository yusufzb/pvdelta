"""Manifest-driven Stage 1 batch execution with frame-local failure isolation."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from pvdelta.config import DetectionConfig
from pvdelta.models import ThermalFrame
from pvdelta.pipeline import execute_stage1
from pvdelta.regions import load_annotations, load_mask_archive

BATCH_SCHEMA_VERSION = "pvdelta.batch/v1"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class BatchFrameResult:
    index: int
    source_name: str
    status: str
    modules: int = 0
    findings: int = 0
    not_assessed: int = 0
    output_directory: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class BatchResult:
    run_id: str
    status: str
    total_frames: int
    succeeded_frames: int
    failed_frames: int
    frames: tuple[BatchFrameResult, ...]
    summary_json: Path
    summary_csv: Path


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _safe_folder(index: int, source: Path) -> str:
    stem = _SAFE_NAME.sub("-", source.stem).strip("-._") or "frame"
    return f"{index:04d}-{stem}"


def _safe_error(exc: Exception, source_name: str, manifest_root: Path) -> str:
    """Keep actionable error type/detail without persisting local absolute paths."""
    if isinstance(exc, OSError) and exc.strerror:
        detail = exc.strerror
    else:
        detail = str(exc)
    detail = detail.replace(str(manifest_root), "<manifest-dir>")
    return f"{type(exc).__name__}: {source_name}: {detail}"


def _config(payload: dict[str, Any]) -> DetectionConfig:
    allowed = {
        "delta_threshold_k",
        "min_region_px",
        "edge_erosion_px",
        "min_reference_px",
        "detrend_gradient",
        "max_reference_iterations",
        "min_valid_fraction",
        "reject_boundary_findings",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown detection settings: {', '.join(sorted(unknown))}")
    return DetectionConfig(**payload)


def _summary_csv(frames: list[BatchFrameResult]) -> str:
    output = StringIO(newline="")
    fields = tuple(BatchFrameResult.__dataclass_fields__)
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(asdict(frame) for frame in frames)
    return output.getvalue()


def run_batch(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    decoder: Callable[..., ThermalFrame],
    database_path: str | Path | None = None,
    sdk_lib: str | None = None,
    overwrite: bool = False,
) -> BatchResult:
    """Run every manifest frame; expected frame errors do not abort later frames."""
    manifest_file = Path(manifest_path).resolve()
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if payload.get("schema_version") != BATCH_SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version must be {BATCH_SCHEMA_VERSION}")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("manifest run_id must not be empty")
    frames_payload = payload.get("frames")
    if not isinstance(frames_payload, list) or not frames_payload:
        raise ValueError("manifest frames must be a non-empty list")
    defaults = payload.get("defaults", {})
    if not isinstance(defaults, dict):
        raise TypeError("manifest defaults must be an object")
    config_payload = defaults.get("detection", {})
    if not isinstance(config_payload, dict):
        raise TypeError("defaults.detection must be an object")
    config = _config(config_payload)
    direction = defaults.get("direction", "both")
    if direction not in {"hot", "cold", "both"}:
        raise ValueError("defaults.direction must be hot, cold or both")
    margin = defaults.get("region_margin_px", 0)
    if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
        raise ValueError("defaults.region_margin_px must be a non-negative integer")

    destination = Path(output_dir)
    summary_json = destination / "batch-summary.json"
    summary_csv = destination / "batch-summary.csv"
    if not overwrite:
        existing = next((path for path in (summary_json, summary_csv) if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"batch output exists: {existing}")
    destination.mkdir(parents=True, exist_ok=True)

    root = manifest_file.parent
    results = []
    expected_errors = (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError)
    for index, item in enumerate(frames_payload, 1):
        source_name = f"frame-{index}"
        try:
            if not isinstance(item, dict):
                raise TypeError("frame entry must be an object")
            source_value = item.get("input")
            regions_value = item.get("regions")
            if not isinstance(source_value, str) or not isinstance(regions_value, str):
                raise TypeError("frame input and regions must be paths")
            source = _resolve(root, source_value)
            source_name = source.name
            regions_path = _resolve(root, regions_value)
            from pvdelta.measurement_profile import validate_overrides

            overrides = validate_overrides(item.get('measurement_overrides', defaults.get('measurement_overrides')))
            frame = decoder(source, sdk_lib, measurement_overrides=overrides) if overrides else decoder(source, sdk_lib)
            expected_hash = item.get('source_sha256')
            if expected_hash is not None and expected_hash != frame.source_hash:
                raise ValueError('Manifest source hash differs from decoded original')
            annotations = load_annotations(regions_path)
            archive_value = item.get("mask_archive")
            module_masks = None
            frame_margin = margin
            if archive_value is not None:
                if not isinstance(archive_value, str):
                    raise ValueError("mask_archive must be a path")
                if margin:
                    raise ValueError(
                        "defaults.region_margin_px must be zero when a mask archive is used"
                    )
                module_masks = load_mask_archive(
                    _resolve(root, archive_value),
                    annotations,
                    frame.temperature_c.shape,
                )
                frame_margin = 0
            frame_output = destination / "frames" / _safe_folder(index, source)
            execution = execute_stage1(
                frame,
                annotations,
                output_dir=frame_output,
                run_id=run_id,
                region_margin_px=frame_margin,
                config=config,
                direction=direction,
                database_path=database_path,
                site_label=payload.get("site_label"),
                overwrite=overwrite,
                module_masks=module_masks,
            )
            results.append(
                BatchFrameResult(
                    index=index,
                    source_name=source_name,
                    status="complete",
                    modules=len(execution.module_results),
                    findings=sum(len(result.findings) for result in execution.module_results),
                    not_assessed=sum(
                        result.assessment_status == "not_assessed"
                        for result in execution.module_results
                    ),
                    output_directory=str(frame_output.relative_to(destination)),
                )
            )
        except expected_errors as exc:
            results.append(
                BatchFrameResult(
                    index=index,
                    source_name=source_name,
                    status="failed",
                    error=_safe_error(exc, source_name, root),
                )
            )

    succeeded = sum(result.status == "complete" for result in results)
    failed = len(results) - succeeded
    status = "complete" if failed == 0 else ("failed" if succeeded == 0 else "partial")
    summary = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "total_frames": len(results),
        "succeeded_frames": succeeded,
        "failed_frames": failed,
        "frames": [asdict(result) for result in results],
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_csv.write_text(_summary_csv(results), encoding="utf-8", newline="")
    return BatchResult(
        run_id,
        status,
        len(results),
        succeeded,
        failed,
        tuple(results),
        summary_json,
        summary_csv,
    )
