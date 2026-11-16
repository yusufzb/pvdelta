"""Read DJI XMP from original image files without interpreting missing pose as zero."""

from __future__ import annotations

import datetime as dt
import io
import math
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from pvdelta import camera_db
from pvdelta.models import CameraProfile

_XMP_START = b"<x:xmpmeta"
_XMP_END = b"</x:xmpmeta>"
_MAX_XMP_BYTES = 1_000_000


@dataclass(frozen=True)
class CaptureMetadata:
    """Observed XMP values; GPS is the camera position, not a panel location."""

    exposure_time_utc: dt.datetime | None
    created_time_utc: dt.datetime | None
    camera_latitude: float | None
    camera_longitude: float | None
    gps_status: str | None
    rtk_flag: str | None
    absolute_altitude_m: float | None
    altitude_type: str | None
    camera_profile: CameraProfile
    raw_xmp: dict[str, str]
    warnings: tuple[str, ...]
    laser_target_distance_m: float | None = None
    laser_status: str | None = None


@dataclass(frozen=True)
class MetadataReadiness:
    usable_for_camera_pose: bool
    usable_for_panel_location: bool
    reasons: tuple[str, ...]


def extract_dji_xmp(image_path: str | Path) -> dict[str, str]:
    """Return namespaced XMP attributes and simple element text as strings.

    This reads a local original file. Callers must not log the returned GPS or
    device-serial values into public output.
    """
    data = Path(image_path).read_bytes()
    start = data.find(_XMP_START)
    if start < 0:
        return {}
    end = data.find(_XMP_END, start)
    if end < 0:
        raise ValueError("XMP packet has no closing tag")
    packet = data[start:end + len(_XMP_END)]
    if len(packet) > _MAX_XMP_BYTES or b"<!DOCTYPE" in packet or b"<!ENTITY" in packet:
        raise ValueError("XMP packet is unsafe or too large")
    try:
        namespaces = dict(payload for _, payload in ET.iterparse(io.BytesIO(packet), events=("start-ns",)))
        root = ET.fromstring(packet)
    except ET.ParseError as exc:
        raise ValueError("invalid XMP XML") from exc
    uri_to_prefix = {uri: prefix for prefix, uri in namespaces.items() if prefix}

    def key(name: str) -> str:
        if name.startswith("{"):
            uri, local = name[1:].split("}", 1)
            prefix = uri_to_prefix.get(uri)
            return f"{prefix}:{local}" if prefix else local
        return name

    result: dict[str, str] = {}
    for element in root.iter():
        for name, value in element.attrib.items():
            result[key(name)] = value
        if len(element) == 0 and element.text and element.text.strip():
            result[key(element.tag)] = element.text.strip()
    return result


def _exif_focal_length_mm(image_path: str | Path) -> float | None:
    """Read EXIF FocalLength (0x920A) from a JPEG without an image dependency."""
    data = Path(image_path).read_bytes()
    if not data.startswith(b"\xff\xd8"):
        return None

    cursor = 2
    exif_payload: bytes | None = None
    while cursor + 4 <= len(data):
        if data[cursor] != 0xFF:
            break
        while cursor < len(data) and data[cursor] == 0xFF:
            cursor += 1
        if cursor >= len(data):
            break
        marker = data[cursor]
        cursor += 1
        if marker in (0xD9, 0xDA):
            break
        if marker in range(0xD0, 0xD8) or marker == 0x01:
            continue
        if cursor + 2 > len(data):
            break
        segment_length = int.from_bytes(data[cursor:cursor + 2], "big")
        if segment_length < 2 or cursor + segment_length > len(data):
            break
        payload = data[cursor + 2:cursor + segment_length]
        if marker == 0xE1 and payload.startswith(b"Exif\x00\x00"):
            exif_payload = payload[6:]
            break
        cursor += segment_length
    if exif_payload is None or len(exif_payload) < 8:
        return None

    endian_mark = exif_payload[:2]
    if endian_mark == b"II":
        endian = "<"
    elif endian_mark == b"MM":
        endian = ">"
    else:
        return None

    def uint16(offset: int) -> int:
        if offset < 0 or offset + 2 > len(exif_payload):
            raise ValueError
        return struct.unpack_from(f"{endian}H", exif_payload, offset)[0]

    def uint32(offset: int) -> int:
        if offset < 0 or offset + 4 > len(exif_payload):
            raise ValueError
        return struct.unpack_from(f"{endian}I", exif_payload, offset)[0]

    def find_entry(ifd_offset: int, tag: int) -> tuple[int, int, int] | None:
        count = uint16(ifd_offset)
        if count > 4096:
            raise ValueError
        for index in range(count):
            offset = ifd_offset + 2 + index * 12
            if offset + 12 > len(exif_payload):
                raise ValueError
            if uint16(offset) == tag:
                return uint16(offset + 2), uint32(offset + 4), offset + 8
        return None

    try:
        if uint16(2) != 42:
            return None
        ifd0 = uint32(4)
        exif_pointer = find_entry(ifd0, 0x8769)
        if exif_pointer is None or exif_pointer[:2] != (4, 1):
            return None
        exif_ifd = uint32(exif_pointer[2])
        focal_entry = find_entry(exif_ifd, 0x920A)
        if focal_entry is None or focal_entry[:2] != (5, 1):
            return None
        rational_offset = uint32(focal_entry[2])
        numerator = uint32(rational_offset)
        denominator = uint32(rational_offset + 4)
    except (ValueError, struct.error):
        return None
    if denominator == 0:
        return None
    focal_length = numerator / denominator
    return focal_length if math.isfinite(focal_length) and focal_length > 0 else None


def _profile_from_xmp(
    xmp: dict[str, str],
    *,
    exif_focal_length_mm: float | None = None,
) -> CameraProfile:
    def optional_float(name: str) -> float | None:
        raw = xmp.get(name)
        if raw is None:
            return None
        try:
            value = float(raw)
        except ValueError:
            return None
        return value if math.isfinite(value) else None

    model = xmp.get("drone-dji:DroneModel", xmp.get("tiff:Model", "Unknown"))

    focal_length_mm = exif_focal_length_mm
    focal_length_source = "image_exif" if focal_length_mm is not None else None
    sensor_width_mm = None
    sensor_height_mm = None
    pixel_pitch_um = None

    platform = camera_db.lookup_by_xmp_model(model)
    if platform is not None and platform.thermal is not None:
        database_focal_length = platform.thermal.actual_focal_length_mm
        if focal_length_mm is None and isinstance(database_focal_length, (int, float)):
            focal_length_mm = float(database_focal_length)
            focal_length_source = "camera_database"
        sensor_width_mm = platform.thermal.sensor_width_mm
        sensor_height_mm = platform.thermal.sensor_height_mm
        pixel_pitch_um = platform.thermal.pixel_pitch_um

    return CameraProfile(
        make=xmp.get("tiff:Make", "Unknown"),
        model=model,
        gimbal_pitch=optional_float("drone-dji:GimbalPitchDegree"),
        gimbal_yaw=optional_float("drone-dji:GimbalYawDegree"),
        gimbal_roll=optional_float("drone-dji:GimbalRollDegree"),
        relative_altitude_m=optional_float("drone-dji:RelativeAltitude"),
        focal_length_mm=focal_length_mm,
        focal_length_source=focal_length_source,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm,
        pixel_pitch_um=pixel_pitch_um,
    )


def parse_camera_profile(image_path: str | Path) -> CameraProfile:
    """Prefer image EXIF values; use reviewed camera specifications as fallback."""
    return _profile_from_xmp(
        extract_dji_xmp(image_path),
        exif_focal_length_mm=_exif_focal_length_mm(image_path),
    )


def parse_capture_metadata(image_path: str | Path) -> CaptureMetadata:
    """Keep the two DJI time fields separate and flag disagreement explicitly."""
    xmp = extract_dji_xmp(image_path)
    warnings: list[str] = []

    def timestamp(key: str, *, utc_named: bool) -> dt.datetime | None:
        raw = xmp.get(key)
        if not raw:
            return None
        try:
            value = dt.datetime.fromisoformat(raw)
        except ValueError:
            warnings.append(f"{key}: invalid timestamp")
            return None
        if value.tzinfo is None:
            if not utc_named:
                warnings.append(f"{key}: timezone missing")
                return None
            value = value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc)

    def coordinate(key: str, limit: float) -> float | None:
        raw = xmp.get(key)
        if raw is None:
            return None
        try:
            value = float(raw)
        except ValueError:
            value = math.nan
        if not math.isfinite(value) or abs(value) > limit:
            warnings.append(f"{key}: invalid coordinate")
            return None
        return value

    exposure = timestamp("drone-dji:UTCAtExposure", utc_named=True)
    created = timestamp("xmp:CreateDate", utc_named=False)
    if exposure and created and abs((exposure - created).total_seconds()) > 1:
        warnings.append("XMP time fields differ by more than one second")
    lat = coordinate("drone-dji:GpsLatitude", 90)
    lon = coordinate("drone-dji:GpsLongitude", 180)
    if (lat is None) != (lon is None):
        warnings.append("camera GPS coordinate pair is incomplete")
    laser_distance = coordinate("drone-dji:LRFTargetDistance", float("inf"))
    if laser_distance is not None and laser_distance <= 0:
        warnings.append("drone-dji:LRFTargetDistance: distance must be positive")
        laser_distance = None
    return CaptureMetadata(
        exposure, created, lat, lon,
        xmp.get("drone-dji:GpsStatus"), xmp.get("drone-dji:RtkFlag"),
        coordinate("drone-dji:AbsoluteAltitude", float("inf")),
        xmp.get("drone-dji:AltitudeType"),
        _profile_from_xmp(xmp, exif_focal_length_mm=_exif_focal_length_mm(image_path)),
        dict(xmp), tuple(warnings),
        laser_target_distance_m=laser_distance,
        laser_status=xmp.get("drone-dji:LRFStatus"),
    )


def assess_metadata_readiness(capture: CaptureMetadata) -> MetadataReadiness:
    """Distinguish camera pose metadata from a calibrated panel-location solution."""
    reasons = []
    profile = capture.camera_profile
    if capture.camera_latitude is None or capture.camera_longitude is None:
        reasons.append("camera GPS coordinates are missing")
    if profile.relative_altitude_m is None and capture.absolute_altitude_m is None:
        reasons.append("camera altitude is missing")
    if any(value is None for value in (
        profile.gimbal_pitch, profile.gimbal_yaw, profile.gimbal_roll,
    )):
        reasons.append("gimbal orientation is incomplete")
    pose_ready = not reasons
    location_reasons = list(reasons)
    if profile.focal_length_mm is None:
        location_reasons.append("calibrated camera intrinsics are missing")
    location_reasons.append("camera-to-thermal boresight and surface model are not calibrated")
    return MetadataReadiness(pose_ready, not location_reasons, tuple(location_reasons))


def visible_pair_candidate(thermal_path: str | Path) -> Path | None:
    """Suggest one filename candidate; this never establishes sensor alignment.

    DJI pairs can have neighbouring timestamp seconds, while retaining the same
    capture counter. Require that counter and a one-second bound, and withhold
    ambiguous suggestions. Original-source controls must still verify the pair.
    """
    path = Path(thermal_path)
    lower_name = path.name.lower()
    if not lower_name.endswith("_t.jpg"):
        return None
    pattern = r'DJI_(\d{14})_(\d{4,})_([TV])\.JPG'
    source = re.fullmatch(pattern, path.name, re.IGNORECASE)
    if source:
        try:
            # Compare filename clocks only; their timezone is unknown and no
            # absolute timestamp is returned or attached to capture metadata.
            timestamp = dt.datetime.strptime(source[1], '%Y%m%d%H%M%S')  # noqa: DTZ007
            candidates = []
            for candidate in path.parent.iterdir():
                match = re.fullmatch(pattern, candidate.name, re.IGNORECASE)
                if not match or match[3].upper() != 'V' or match[2] != source[2]:
                    continue
                try:
                    other_time = dt.datetime.strptime(match[1], '%Y%m%d%H%M%S')  # noqa: DTZ007
                except ValueError:
                    continue
                if abs((other_time-timestamp).total_seconds()) <= 1 and candidate.is_file():
                    candidates.append(candidate)
            return candidates[0] if len(candidates) == 1 else None
        except (OSError, ValueError):
            return None
    # Retain the legacy exact-suffix suggestion for other filename conventions.
    candidate_name = path.name[:-6] + "_V.JPG"
    candidate = path.with_name(candidate_name)
    if candidate.is_file():
        return candidate
    candidate_name_lower = path.name[:-6] + "_v.jpg"
    candidate_lower = path.with_name(candidate_name_lower)
    return candidate_lower if candidate_lower.is_file() else None
