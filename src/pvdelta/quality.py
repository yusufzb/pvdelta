"""Data integrity checks; camera-specific physical limits require a validated profile."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class QualityError(ValueError):
    """Raised when a frame or region cannot be measured safely."""


class Capability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class QualityAssessment:
    image_analysis: Capability
    normalization: Capability
    location: Capability
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegionQuality:
    analysis: Capability
    valid_pixels: int
    region_pixels: int
    valid_fraction: float
    reference_pixels: int
    reasons: tuple[str, ...] = ()


def assess_region_quality(
    temp_c: np.ndarray,
    region_mask: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    reference_mask: np.ndarray | None = None,
    min_valid_fraction: float = 0.95,
    min_reference_pixels: int = 16,
    saturation_bounds_c: tuple[float, float] | None = None,
) -> RegionQuality:
    if temp_c.ndim != 2 or region_mask.shape != temp_c.shape or region_mask.dtype != np.bool_:
        raise QualityError("region mask must be boolean and match the temperature grid")
    if not 0 < min_valid_fraction <= 1 or min_reference_pixels < 1:
        raise QualityError("quality thresholds are invalid")
    region_pixels = int(region_mask.sum())
    if region_pixels == 0:
        raise QualityError("region contains no pixels")
    finite = np.isfinite(temp_c)
    valid = region_mask & finite
    if valid_mask is not None:
        if valid_mask.shape != temp_c.shape or valid_mask.dtype != np.bool_:
            raise QualityError("valid mask must be boolean and match the temperature grid")
        valid &= valid_mask
    if reference_mask is None:
        reference = valid.copy()
    else:
        if reference_mask.shape != temp_c.shape or reference_mask.dtype != np.bool_:
            raise QualityError("reference mask must be boolean and match the temperature grid")
        reference = reference_mask & finite
        if valid_mask is not None:
            reference &= valid_mask
    valid_pixels = int(valid.sum())
    reference_pixels = int(reference.sum())
    fraction = valid_pixels / region_pixels
    reasons = []
    if fraction < min_valid_fraction:
        reasons.append(f"valid pixel fraction {fraction:.1%} is below {min_valid_fraction:.1%}")
    if reference_pixels < min_reference_pixels:
        reasons.append("too few valid reference pixels")
    if saturation_bounds_c is not None and valid_pixels:
        low, high = saturation_bounds_c
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise QualityError("saturation bounds are invalid")
        saturated = valid & ((temp_c <= low) | (temp_c >= high))
        if saturated.any():
            reasons.append(f"{int(saturated.sum())} pixels reach configured saturation bounds")
    capability = Capability.AVAILABLE if not reasons else Capability.UNAVAILABLE
    return RegionQuality(
        capability, valid_pixels, region_pixels, fraction, reference_pixels, tuple(reasons),
    )


def assess_capabilities(
    temp_c: np.ndarray,
    *,
    has_conditions: bool,
    has_camera_pose: bool,
    has_calibrated_geometry: bool,
) -> QualityAssessment:
    """Evaluate capabilities independently so one missing input does not hide the others."""
    reasons = tuple(validate_thermal_frame(temp_c))
    analysis = Capability.AVAILABLE if not reasons else Capability.UNAVAILABLE
    normalization = Capability.AVAILABLE if has_conditions else Capability.UNAVAILABLE
    location = (
        Capability.AVAILABLE
        if has_camera_pose and has_calibrated_geometry
        else Capability.UNAVAILABLE
    )
    extra = list(reasons)
    if not has_conditions:
        extra.append("normalization unavailable: no usable matched condition")
    if not has_camera_pose:
        extra.append("location unavailable: camera pose is incomplete")
    elif not has_calibrated_geometry:
        extra.append("location unavailable: camera/surface geometry is not calibrated")
    return QualityAssessment(analysis, normalization, location, tuple(extra))


def validate_thermal_frame(
    temp_c: np.ndarray,
    *,
    min_c: float | None = None,
    max_c: float | None = None,
) -> list[str]:
    """Return quality flags while rejecting unusable frames.

    Camera range bounds are optional: no M4T gain mode is inferred from a raster.
    """
    if not isinstance(temp_c, np.ndarray) or temp_c.ndim != 2 or temp_c.size == 0:
        raise QualityError("temperature grid must be a non-empty two-dimensional array")
    if not np.issubdtype(temp_c.dtype, np.floating):
        raise QualityError("temperature grid must contain floating-point Celsius values")
    finite = np.isfinite(temp_c)
    if not finite.any():
        raise QualityError("temperature grid contains no finite values")
    if min_c is not None and (not math.isfinite(min_c) or np.min(temp_c[finite]) < min_c):
        raise QualityError("temperature below configured camera range")
    if max_c is not None and (not math.isfinite(max_c) or np.max(temp_c[finite]) > max_c):
        raise QualityError("temperature above configured camera range")
    flags = []
    if not finite.all():
        flags.append("Frame contains NaN or Inf values")
    if np.ptp(temp_c[finite]) == 0:
        flags.append("Zero variance in thermal frame (all valid pixels identical)")
    return flags


def validate_module_observation(
    temp_min: float,
    temp_max: float,
    pixel_polygon: list[tuple[float, float]],
    frame_width: int,
    frame_height: int,
) -> list[str]:
    """Validate simple polygon geometry and its temperature statistics."""
    if frame_width <= 0 or frame_height <= 0:
        raise QualityError("frame dimensions must be positive")
    if not all(math.isfinite(v) for v in (temp_min, temp_max)):
        raise QualityError("module temperatures must be finite")
    if temp_max < temp_min:
        raise QualityError("temp_max cannot be less than temp_min")
    if len(pixel_polygon) < 3:
        raise QualityError("Polygon must have at least 3 vertices")
    if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in pixel_polygon):
        raise QualityError("Polygon vertices must be finite (x, y) pairs")
    xs = [p[0] for p in pixel_polygon]
    ys = [p[1] for p in pixel_polygon]
    area2 = sum(
        xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i]
        for i in range(len(xs))
    )
    if abs(area2) < 1e-9:
        raise QualityError("Polygon has zero area")
    if min(xs) >= frame_width or max(xs) <= 0 or min(ys) >= frame_height or max(ys) <= 0:
        raise QualityError("Polygon is completely outside the frame boundaries")
    flags = []
    if min(xs) < 0 or max(xs) > frame_width or min(ys) < 0 or max(ys) > frame_height:
        flags.append("Polygon is partially clipped by the frame boundaries")
    return flags
