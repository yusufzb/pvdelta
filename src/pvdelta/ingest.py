"""Index a local DJI flight folder into the private DuckDB inspection store."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pvdelta.metadata import parse_capture_metadata
from pvdelta.store import CaptureRecord, connect_store, upsert_capture, upsert_run

_CAPTURE_NAME = re.compile(
    r"^(?P<prefix>.+)_(?P<sequence>\d+)_(?P<kind>[TV])\.JPE?G$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IngestSummary:
    run_id: str
    discovered_files: int
    indexed_captures: int
    thermal_captures: int
    rgb_captures: int
    paired_captures: int
    geolocated_captures: int
    laser_targets: int
    files_with_warnings: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_xmp_float(raw_xmp: dict[str, str], name: str) -> float | None:
    raw = raw_xmp.get(name)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _image_dimensions(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except (ImportError, OSError, ValueError):
        return None, None


def discover_capture_files(folder: str | Path) -> list[Path]:
    """Return DJI thermal and visible JPEGs, excluding unrelated images."""
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"flight folder does not exist: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and _CAPTURE_NAME.match(path.name)
    )


def _pair_key(root: Path, path: Path, match: re.Match[str]) -> str:
    """Pair DJI thermal/RGB frames by folder and sequence, not capture timestamp."""
    parent = path.parent.relative_to(root).as_posix().casefold()
    return f"{parent}:{match.group('sequence')}"


def index_flight_folder(
    folder: str | Path,
    database_path: str | Path,
    *,
    run_id: str,
    site_label: str | None = None,
    progress: Callable[[int, int, Path], None] | None = None,
) -> IngestSummary:
    """Index capture metadata without decoding or copying private images."""
    root = Path(folder).resolve()
    files = discover_capture_files(root)
    if not files:
        raise ValueError("flight folder contains no *_T.JPG or *_V.JPG captures")

    hashes: dict[Path, str] = {}
    capture_ids: dict[Path, str] = {}
    pairs: dict[str, dict[str, Path]] = {}
    for path in files:
        digest = _sha256(path)
        hashes[path] = digest
        capture_ids[path] = f"sha256:{digest}"
        match = _CAPTURE_NAME.match(path.name)
        assert match is not None
        pairs.setdefault(_pair_key(root, path, match), {})[match.group("kind").upper()] = path

    connection = connect_store(database_path)
    indexed = thermal = rgb = paired = geolocated = laser_targets = warned = 0
    first_capture_time = None
    camera_model = None
    connection.execute("BEGIN TRANSACTION")
    try:
        for position, path in enumerate(files, start=1):
            match = _CAPTURE_NAME.match(path.name)
            assert match is not None
            kind_code = match.group("kind").upper()
            image_kind = "thermal" if kind_code == "T" else "rgb"
            other_code = "V" if kind_code == "T" else "T"
            pair_path = pairs[_pair_key(root, path, match)].get(other_code)
            warnings: list[str] = []
            try:
                metadata = parse_capture_metadata(path)
                warnings.extend(metadata.warnings)
            except (OSError, ValueError) as exc:
                metadata = None
                warnings.append(f"metadata parse failed: {exc}")

            width_px, height_px = _image_dimensions(path)
            if width_px is None or height_px is None:
                warnings.append("image dimensions unavailable")

            profile = None if metadata is None else metadata.camera_profile
            raw_xmp = {} if metadata is None else metadata.raw_xmp
            capture_time = None if metadata is None else metadata.exposure_time_utc
            if capture_time is not None and (
                first_capture_time is None or capture_time < first_capture_time
            ):
                first_capture_time = capture_time
            if camera_model is None and profile is not None:
                camera_model = profile.model

            target_latitude = _optional_xmp_float(raw_xmp, "drone-dji:LRFTargetLat")
            target_longitude = _optional_xmp_float(raw_xmp, "drone-dji:LRFTargetLon")
            if (target_latitude is None) != (target_longitude is None):
                warnings.append("laser target coordinate pair is incomplete")
                target_latitude = None
                target_longitude = None

            record = CaptureRecord(
                capture_id=capture_ids[path],
                source_path=path.relative_to(root).as_posix(),
                source_hash=hashes[path],
                image_kind=image_kind,
                run_id=run_id,
                capture_time_utc=capture_time,
                paired_capture_id=None if pair_path is None else capture_ids[pair_path],
                camera_model=None if profile is None else profile.model,
                width_px=width_px,
                height_px=height_px,
                gps_status=None if metadata is None else metadata.gps_status,
                rtk_flag=None if metadata is None else metadata.rtk_flag,
                camera_latitude=None if metadata is None else metadata.camera_latitude,
                camera_longitude=None if metadata is None else metadata.camera_longitude,
                absolute_altitude_m=None if metadata is None else metadata.absolute_altitude_m,
                relative_altitude_m=None if profile is None else profile.relative_altitude_m,
                gimbal_pitch_deg=None if profile is None else profile.gimbal_pitch,
                gimbal_yaw_deg=None if profile is None else profile.gimbal_yaw,
                gimbal_roll_deg=None if profile is None else profile.gimbal_roll,
                laser_target_distance_m=(
                    None if metadata is None else metadata.laser_target_distance_m
                ),
                laser_status=None if metadata is None else metadata.laser_status,
                laser_target_latitude=target_latitude,
                laser_target_longitude=target_longitude,
                laser_target_absolute_altitude_m=_optional_xmp_float(
                    raw_xmp, "drone-dji:LRFTargetAbsAlt"
                ),
                focal_length_mm=None if profile is None else profile.focal_length_mm,
                warnings=tuple(warnings),
            )
            upsert_capture(connection, record)
            indexed += 1
            thermal += image_kind == "thermal"
            rgb += image_kind == "rgb"
            paired += pair_path is not None
            geolocated += record.camera_latitude is not None
            laser_targets += record.laser_target_latitude is not None
            warned += bool(warnings)
            if progress is not None:
                progress(position, len(files), path)

        upsert_run(
            connection,
            run_id=run_id,
            site_label=site_label,
            started_at_utc=first_capture_time,
            camera_model=camera_model,
            source_root=str(root),
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()

    return IngestSummary(
        run_id=run_id,
        discovered_files=len(files),
        indexed_captures=indexed,
        thermal_captures=thermal,
        rgb_captures=rgb,
        paired_captures=paired,
        geolocated_captures=geolocated,
        laser_targets=laser_targets,
        files_with_warnings=warned,
    )
