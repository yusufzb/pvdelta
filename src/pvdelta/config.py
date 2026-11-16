"""Yapılandırma (Configuration) nesneleri (T03).

Uygulamanın çalışmasını ve algoritmaların parametrelerini yönetir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral


@dataclass(frozen=True)
class DetectionConfig:
    """Exploratory candidate settings; these are not IEC severity bands."""

    delta_threshold_k: float = 2.5
    min_region_px: int = 4
    edge_erosion_px: int = 1
    min_reference_px: int = 16
    detrend_gradient: bool = True
    max_reference_iterations: int = 4
    min_valid_fraction: float = 0.95
    # Keep edge candidates visible for review by default. A strict screening
    # profile may opt into rejection, but silently dropping edge faults is unsafe.
    reject_boundary_findings: bool = False

    def __post_init__(self) -> None:
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (
            self.min_region_px, self.min_reference_px, self.edge_erosion_px,
            self.max_reference_iterations,
        )):
            raise ValueError("pixel counts and iteration count must be integers")
        if not 0 < self.min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be in (0, 1]")
        if not isinstance(self.reject_boundary_findings, bool):
            raise TypeError("reject_boundary_findings must be boolean")
        if not math.isfinite(self.delta_threshold_k) or self.delta_threshold_k <= 0:
            raise ValueError("delta_threshold_k must be finite and positive")
        if self.min_region_px < 1 or self.min_reference_px < 1 or self.edge_erosion_px < 0:
            raise ValueError("pixel counts must be valid positive sizes")
        if self.max_reference_iterations < 1:
            raise ValueError("max_reference_iterations must be positive")


@dataclass(frozen=True)
class QualityConfig:
    min_valid_fraction: float = 0.95
    min_reference_px: int = 16
    max_condition_age_s: float = 30.0

    def __post_init__(self) -> None:
        if not 0 < self.min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be within (0, 1]")
        if self.min_reference_px < 1:
            raise ValueError("min_reference_px must be positive")
        if not math.isfinite(self.max_condition_age_s) or self.max_condition_age_s < 0:
            raise ValueError("max_condition_age_s must be finite and non-negative")


@dataclass(frozen=True)
class PipelineConfig:
    """Tüm analiz hattını yönetecek global yapılandırma."""
    # İşlenecek görüntülerin veya manifestonun yolu
    input_source: str = ""
    # Çıktıların yazılacağı ana dizin
    output_dir: str = "output"

    # Alt konfigürasyonlar
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)

    # Flag'ler
    run_stage2: bool = False
    debug_images: bool = False
