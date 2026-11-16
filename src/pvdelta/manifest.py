"""Discover DJI thermal files and match explicitly named native annotations."""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from pvdelta.batch import BATCH_SCHEMA_VERSION
from pvdelta.config import DetectionConfig
from pvdelta.regions import load_annotations


def prepare_manifest(
    flight_dir: Path,
    regions_dir: Path,
    output: Path,
    *,
    run_id: str,
    mask_dir: Path | None = None,
    overwrite: bool = False,
) -> dict:
    """Match mirrored sidecars, never reuse a different frame's annotations.

    Discovery is filename-based, not proof of radiometric content or alignment.
    When mask_dir is supplied, every included frame must have an exact mask file.
    Native array validation and decoding remain the batch runner's responsibility.
    """
    flight_dir, regions_dir, output = (
        Path(path).resolve() for path in (flight_dir, regions_dir, output)
    )
    if mask_dir is not None:
        mask_dir = Path(mask_dir).resolve()
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    for label, directory in (
        ("flight", flight_dir), ("regions", regions_dir), ("mask", mask_dir)
    ):
        if directory is not None and not directory.is_dir():
            raise ValueError(f"{label} directory does not exist")
    if output.exists() and not overwrite:
        raise FileExistsError("manifest output exists; choose another path or use --overwrite")

    def relative(path: Path) -> str:
        return Path(os.path.relpath(path, output.parent)).as_posix()

    sources = sorted(
        (
            path for path in flight_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
            and path.stem.upper().endswith("_T")
        ),
        key=lambda path: path.relative_to(flight_dir).as_posix(),
    )
    frames, unresolved = [], []
    sidecar_keys = Counter(path.relative_to(flight_dir).with_suffix("") for path in sources)
    protected = set(sources)
    for source in sources:
        local = source.relative_to(flight_dir)
        regions = regions_dir / local.with_suffix(".regions.json")
        archive = None if mask_dir is None else mask_dir / local.with_suffix(".masks.npz")
        protected.add(regions)
        if archive is not None:
            protected.add(archive)
        reason = None
        if sidecar_keys[local.with_suffix("")] > 1:
            reason = "ambiguous_capture_stem"
        elif not regions.is_file():
            reason = "missing_regions"
        else:
            try:
                annotations = load_annotations(regions)
                if not annotations:
                    reason = "empty_regions"
                elif archive is not None:
                    if not archive.is_file():
                        reason = "missing_mask_archive"
                    elif any(item.mask_label is None for item in annotations):
                        reason = "missing_mask_labels"
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                reason = "invalid_regions"
        if reason:
            unresolved.append({"input": local.as_posix(), "reason": reason})
            continue
        entry = {"input": relative(source), "regions": relative(regions)}
        if archive is not None:
            entry["mask_archive"] = relative(archive)
        frames.append(entry)

    if output in protected:
        raise ValueError("manifest output must not replace a capture or annotation input")
    status = "ready" if frames and not unresolved else ("partial" if frames else "blocked")
    payload = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "run_id": run_id,
        "defaults": {
            "direction": "both",
            "region_margin_px": 0,
            "detection": asdict(DetectionConfig()),
        },
        "frames": frames,
        "preparation": {
            "status": status,
            "discovered_frames": len(sources),
            "matched_frames": len(frames),
            "unresolved_frames": len(unresolved),
            "unresolved": unresolved,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return payload
