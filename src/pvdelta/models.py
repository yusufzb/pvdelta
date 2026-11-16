"""Sürümlü veri sözleşmeleri (Data Contracts) ve temel veri yapıları.

Bu modül, Pipeline boyunca dolaşacak verinin şemasını belirler (T03).
Saf hesaplama hedefi doğrultusunda, bağımlılıkları minimumda tutmak için
Python'un yerleşik `dataclass` yapısı kullanılmıştır.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class CoordinateSpace(str, Enum):
    THERMAL_NATIVE = "thermal_native"


class MatchStatus(str, Enum):
    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    UNMATCHED = "unmatched"
    NOT_ATTEMPTED = "not_attempted"


class LocationStatus(str, Enum):
    ESTIMATED = "estimated"
    INSUFFICIENT_METADATA = "insufficient_metadata"
    NOT_ATTEMPTED = "not_attempted"


@dataclass(frozen=True)
class RunManifest:
    """Tüm pipeline koşusunun yapılandırma ve versiyon kaydı. Yinelenebilirliği (reproducibility) sağlar."""
    run_id: str
    utc_time: datetime.datetime
    schema_version: str = "1.0"
    code_version: str = "unknown"
    config_hash: str = ""
    inputs_hash: dict[str, str] = field(default_factory=dict)
    sdk_version: str = "unknown"

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if self.utc_time.tzinfo is None or self.utc_time.utcoffset() != datetime.timedelta(0):
            raise ValueError("utc_time must be timezone-aware UTC")


@dataclass(frozen=True)
class CameraProfile:
    """Observed camera metadata plus clearly sourced specification fallbacks."""
    make: str = "DJI"
    model: str = "Unknown"
    focal_length_mm: float | None = None
    focal_length_source: str | None = None
    sensor_width_mm: float | None = None
    sensor_height_mm: float | None = None
    pixel_pitch_um: float | None = None
    # Gimbal açıları Raycasting (T14) için kritik
    gimbal_pitch: float | None = None
    gimbal_yaw: float | None = None
    gimbal_roll: float | None = None
    relative_altitude_m: float | None = None


@dataclass(frozen=True)
class ThermalFrame:
    """Tek bir R-JPEG görüntüsünü, ham sıcaklık matrisini ve tüm bağlamını temsil eder."""
    frame_id: str
    source_path: str
    source_hash: str
    temperature_c: np.ndarray  # HxW float32 matris (T05'ten gelen)
    valid_mask: np.ndarray | None
    width: int
    height: int
    capture_time: datetime.datetime | None = None
    camera_profile: CameraProfile | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.frame_id.strip() or not self.source_hash.strip():
            raise ValueError("frame_id and source_hash must not be empty")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("frame dimensions must be positive")
        if not isinstance(self.temperature_c, np.ndarray) or self.temperature_c.ndim != 2:
            raise ValueError("temperature_c must be a two-dimensional NumPy array")
        if not np.issubdtype(self.temperature_c.dtype, np.floating):
            raise ValueError("temperature_c must use a floating-point dtype")
        if self.temperature_c.shape != (self.height, self.width):
            raise ValueError(
                f"temperature_c shape {self.temperature_c.shape} must match (height={self.height}, width={self.width})"
            )
        if self.valid_mask is not None and (
            not isinstance(self.valid_mask, np.ndarray)
            or self.valid_mask.dtype != np.bool_
            or self.valid_mask.shape != self.temperature_c.shape
        ):
            raise ValueError("valid_mask must be a boolean array matching temperature_c")
        if self.capture_time is not None and self.capture_time.tzinfo is None:
            raise ValueError("capture_time must include a timezone")


@dataclass(frozen=True)
class ModuleObservation:
    """ThermalFrame içinde tespit edilmiş (manuel veya otomatik) bir PV modülü."""
    module_id: str
    frame_id: str
    # Modülün piksel koordinatlarındaki köşe noktaları [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    pixel_polygon: list[tuple[float, float]]

    # İstatistikler (T09)
    temp_min: float | None = None
    temp_max: float | None = None
    temp_mean: float | None = None
    temp_median: float | None = None

    # Kalite kapısı (T07) durumu
    is_valid: bool = True
    quality_flags: list[str] = field(default_factory=list)
    coordinate_space: CoordinateSpace = CoordinateSpace.THERMAL_NATIVE
    visibility: str = "full"

    def __post_init__(self) -> None:
        if not self.module_id.strip() or not self.frame_id.strip():
            raise ValueError("module_id and frame_id must not be empty")
        if not isinstance(self.coordinate_space, CoordinateSpace):
            raise TypeError("coordinate_space must be a CoordinateSpace")
        if self.visibility not in {"full", "partial"}:
            raise ValueError("visibility must be full or partial")
        if self.pixel_polygon and len(self.pixel_polygon) < 3:
            raise ValueError("pixel_polygon needs at least three vertices")
        for point in self.pixel_polygon:
            if len(point) != 2 or not all(math.isfinite(value) for value in point):
                raise ValueError("pixel_polygon needs finite (x, y) coordinates")
        for value in (self.temp_min, self.temp_max, self.temp_mean, self.temp_median):
            if value is not None and not math.isfinite(value):
                raise ValueError("temperature statistics must be finite")
        if (self.temp_min is not None and self.temp_max is not None
                and self.temp_min > self.temp_max):
            raise ValueError("temp_min cannot exceed temp_max")


@dataclass(frozen=True)
class Finding:
    """Tespit edilen bir anomali (Stage 1 ciktisi)."""
    finding_id: str
    module_id: str
    frame_id: str
    anomaly_type: str

    delta_t_c: float
    absolute_t_c: float

    pixel_polygon: list[tuple[float, float]]
    confidence: float | None = None

    # T11: Normalizasyon
    is_normalized: bool = False
    normalized_delta_t_c: float | None = None
    normalization_basis: str | None = None
    normalization_sources: list[str] = field(default_factory=list)
    normalization_flags: list[str] = field(default_factory=list)
    reference_t_c: float | None = None
    reference_method: str | None = None
    n_region_px: int | None = None
    n_reference_px: int | None = None
    assessment_status: str = "candidate"
    # Exact component support; pixel_polygon is only its display bounding box.
    region_pixels_yx: list[tuple[int, int]] | None = None
    region_max_t_c: float | None = None
    region_p95_t_c: float | None = None
    region_max_residual_k: float | None = None
    delta_method: str | None = None

    def __post_init__(self) -> None:
        if not all((self.finding_id.strip(), self.module_id.strip(), self.frame_id.strip())):
            raise ValueError("finding, module and frame IDs must not be empty")
        if not math.isfinite(self.delta_t_c) or not math.isfinite(self.absolute_t_c):
            raise ValueError("finding temperatures must be finite")
        if self.pixel_polygon and len(self.pixel_polygon) < 3:
            raise ValueError("finding polygon needs at least three vertices")
        if any(len(point) != 2 or not all(math.isfinite(v) for v in point)
               for point in self.pixel_polygon):
            raise ValueError("finding polygon needs finite (x, y) coordinates")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be within [0, 1]")
        if self.reference_t_c is not None and not math.isfinite(self.reference_t_c):
            raise ValueError("reference temperature must be finite")
        if self.normalized_delta_t_c is not None and not math.isfinite(self.normalized_delta_t_c):
            raise ValueError("normalized delta must be finite")
        if self.is_normalized != (self.normalized_delta_t_c is not None):
            raise ValueError("normalization status and value disagree")
        if self.is_normalized != (self.normalization_basis is not None):
            raise ValueError("normalization basis and status disagree")
        if self.n_region_px is not None and self.n_region_px < 1:
            raise ValueError("region pixel count must be positive")
        if self.n_reference_px is not None and self.n_reference_px < 1:
            raise ValueError("reference pixel count must be positive")
        if self.assessment_status not in {
            "candidate",
            "review_required",
            "reviewed",
            "not_assessed",
        }:
            raise ValueError("invalid assessment_status")
        for value in (self.region_max_t_c, self.region_p95_t_c, self.region_max_residual_k):
            if value is not None and not math.isfinite(value):
                raise ValueError("region statistics must be finite")
        if self.region_pixels_yx is not None:
            if len(self.region_pixels_yx) != self.n_region_px:
                raise ValueError("exact component support must match n_region_px")
            if len(set(map(tuple, self.region_pixels_yx))) != len(self.region_pixels_yx):
                raise ValueError("component pixels must be unique")
            if any(len(p) != 2 or any(not isinstance(v, (int, np.integer)) or v < 0 for v in p)
                   for p in self.region_pixels_yx):
                raise ValueError("component pixels must be non-negative integer (y, x)")


@dataclass(frozen=True)
class LocationEstimate:
    """Optional Stage 1 geographic estimate, distinct from inventory coordinates."""

    observation_id: str
    status: LocationStatus = LocationStatus.NOT_ATTEMPTED
    latitude: float | None = None
    longitude: float | None = None
    horizontal_accuracy_m: float | None = None
    method: str | None = None

    def __post_init__(self) -> None:
        if not self.observation_id.strip() or not isinstance(self.status, LocationStatus):
            raise ValueError("valid observation_id and location status are required")
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be provided together")
        if self.latitude is not None and not -90 <= self.latitude <= 90:
            raise ValueError("latitude is outside [-90, 90]")
        if self.longitude is not None and not -180 <= self.longitude <= 180:
            raise ValueError("longitude is outside [-180, 180]")
        if self.horizontal_accuracy_m is not None and (
            not math.isfinite(self.horizontal_accuracy_m) or self.horizontal_accuracy_m < 0
        ):
            raise ValueError("horizontal accuracy must be finite and non-negative")
        if self.status is LocationStatus.ESTIMATED and (
            self.latitude is None or self.method is None or self.horizontal_accuracy_m is None
        ):
            raise ValueError("estimated locations require coordinates, method and accuracy")


@dataclass(frozen=True)
class AssetMatch:
    """Optional Stage 2 match. Score is an uncalibrated ranking score, not probability."""

    observation_id: str
    status: MatchStatus = MatchStatus.NOT_ATTEMPTED
    asset_id: str | None = None
    ranking_score: float | None = None
    candidate_asset_ids: tuple[str, ...] = ()
    evidence: str | None = None

    def __post_init__(self) -> None:
        if not self.observation_id.strip() or not isinstance(self.status, MatchStatus):
            raise ValueError("valid observation_id and match status are required")
        if self.ranking_score is not None and not math.isfinite(self.ranking_score):
            raise ValueError("ranking_score must be finite")
        if self.evidence is not None and not self.evidence.strip():
            raise ValueError("evidence must not be empty when supplied")
        if self.status is MatchStatus.MATCHED and not self.asset_id:
            raise ValueError("matched status requires an asset_id")
        if self.status is not MatchStatus.MATCHED and self.asset_id is not None:
            raise ValueError("only a matched record may carry an asset_id")
