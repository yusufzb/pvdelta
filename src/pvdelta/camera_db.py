"""Drone kamera veritabanı yükleyici.

XMP'den okunan `tiff:Model` etiketini kullanarak ilgili platformun
termal, RGB ve LRF özelliklerini döndürür.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DB_PATH = Path(__file__).with_name("camera_db.json")

# ---------------------------------------------------------------------------
# Veri sınıfları
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraSpec:
    """Tek bir kamera sensörünün özellikleri."""
    role: str                               # "thermal", "wide", "zoom", ...
    sensor_type: str                        # "uncooled_vox", "cmos"
    sensor_width_px: int
    sensor_height_px: int
    equivalent_focal_length_mm: float | list[float]
    actual_focal_length_mm: float | list[float] | None = None
    pixel_pitch_um: float | None = None
    sensor_width_mm: float | None = None
    sensor_height_mm: float | None = None
    dfov_deg: float | None = None
    aperture: str | None = None
    netd_mk: int | None = None
    fps: int | None = None
    spectral_range_um: list[float] | None = None
    file_suffix: str | None = None
    image_source_tag: str | None = None


@dataclass(frozen=True)
class LaserRangeFinder:
    """Laser Range Finder özellikleri."""
    range_min_m: int
    range_max_m: int
    wavelength_nm: int = 905
    accuracy: str | None = None


@dataclass(frozen=True)
class PlatformSpec:
    """Bir drone platformunun tüm kamera ve LRF özellikleri."""
    platform_key: str                       # "DJI_M4T"
    manufacturer: str
    model: str                              # "Matrice 4T"
    xmp_model_id: str                       # XMP'deki tiff:Model değeri
    cameras: dict[str, CameraSpec]          # role → spec
    lrf: LaserRangeFinder | None = None
    source_urls: tuple[str, ...] = ()
    verification_status: str = "unverified"

    @property
    def thermal(self) -> CameraSpec | None:
        return self.cameras.get("thermal")

    @property
    def wide(self) -> CameraSpec | None:
        return self.cameras.get("wide")


# ---------------------------------------------------------------------------
# Yükleme
# ---------------------------------------------------------------------------

def _load_db(path: Path | None = None) -> dict[str, Any]:
    p = path or _DB_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("platforms"), dict):
        raise TypeError("camera database must contain a platforms object")
    return raw


def _positive_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{field_name} must be a positive number")
    return float(value)


def _positive_focal(value: Any, field_name: str) -> float | list[float] | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{field_name} must not be empty")
        return [_positive_number(item, field_name) for item in value]
    return _positive_number(value, field_name)


def _parse_camera(role: str, data: dict[str, Any]) -> CameraSpec:
    width = _positive_number(data["sensor_width_px"], f"{role}.sensor_width_px")
    height = _positive_number(data["sensor_height_px"], f"{role}.sensor_height_px")
    if not width.is_integer() or not height.is_integer():
        raise ValueError(f"{role} sensor dimensions must be integers")
    return CameraSpec(
        role=role,
        sensor_type=data["sensor_type"],
        sensor_width_px=int(width),
        sensor_height_px=int(height),
        equivalent_focal_length_mm=_positive_focal(
            data["equivalent_focal_length_mm"], f"{role}.equivalent_focal_length_mm",
        ),
        actual_focal_length_mm=_positive_focal(
            data.get("actual_focal_length_mm"), f"{role}.actual_focal_length_mm",
        ),
        pixel_pitch_um=data.get("pixel_pitch_um"),
        sensor_width_mm=data.get("sensor_width_mm"),
        sensor_height_mm=data.get("sensor_height_mm"),
        dfov_deg=data.get("dfov_deg"),
        aperture=data.get("aperture"),
        netd_mk=data.get("netd_mk"),
        fps=data.get("fps"),
        spectral_range_um=data.get("spectral_range_um"),
        file_suffix=data.get("file_suffix"),
        image_source_tag=data.get("image_source_tag"),
    )


def _parse_lrf(data: dict[str, Any] | None) -> LaserRangeFinder | None:
    if data is None:
        return None
    return LaserRangeFinder(
        range_min_m=data["range_min_m"],
        range_max_m=data["range_max_m"],
        wavelength_nm=data.get("wavelength_nm", 905),
        accuracy=data.get("accuracy"),
    )


def _parse_platform(key: str, data: dict[str, Any]) -> PlatformSpec:
    cameras = {
        role: _parse_camera(role, cam_data)
        for role, cam_data in data["cameras"].items()
    }
    return PlatformSpec(
        platform_key=key,
        manufacturer=data["manufacturer"],
        model=data["model"],
        xmp_model_id=data["xmp_model_id"],
        cameras=cameras,
        lrf=_parse_lrf(data.get("lrf")),
        source_urls=tuple(data.get("source_urls", ())),
        verification_status=data.get("verification_status", "unverified"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CACHE: dict[str, PlatformSpec] | None = None
_XMP_INDEX: dict[str, str] | None = None


def load_database(path: str | Path) -> dict[str, PlatformSpec]:
    """Load and validate a camera database without changing the process cache."""
    raw = _load_db(Path(path))
    result: dict[str, PlatformSpec] = {}
    seen_models: set[str] = set()
    for key, pdata in raw["platforms"].items():
        if not isinstance(pdata, dict):
            raise TypeError(f"platform {key} must be an object")
        spec = _parse_platform(key, pdata)
        if spec.xmp_model_id in seen_models:
            raise ValueError(f"duplicate xmp_model_id: {spec.xmp_model_id}")
        if not spec.source_urls:
            raise ValueError(f"platform {key} has no source_urls")
        seen_models.add(spec.xmp_model_id)
        result[key] = spec
    return result


def _ensure_loaded() -> None:
    global _CACHE, _XMP_INDEX
    if _CACHE is not None:
        return
    _CACHE = load_database(_DB_PATH)
    _XMP_INDEX = {}
    for key, spec in _CACHE.items():
        _XMP_INDEX[spec.xmp_model_id] = key


def get_platform(platform_key: str) -> PlatformSpec:
    """Platform key ile spec döndürür (örn: 'DJI_M4T')."""
    _ensure_loaded()
    assert _CACHE is not None
    if platform_key not in _CACHE:
        raise KeyError(
            f"Unknown platform '{platform_key}'. "
            f"Available: {sorted(_CACHE.keys())}"
        )
    return _CACHE[platform_key]


def lookup_by_xmp_model(xmp_model: str) -> PlatformSpec | None:
    """XMP'deki tiff:Model değerinden platform spec döndürür.

    Eşleşme bulamazsa None döner (crash etmez — bilinmeyen drone modeli
    olabilir, pipeline bunu quality flag olarak raporlamalı).
    """
    _ensure_loaded()
    assert _XMP_INDEX is not None
    key = _XMP_INDEX.get(xmp_model)
    if key is None:
        return None
    assert _CACHE is not None
    return _CACHE[key]


def list_platforms() -> list[str]:
    """Tüm kayıtlı platform key'lerini döndürür."""
    _ensure_loaded()
    assert _CACHE is not None
    return sorted(_CACHE.keys())
