"""Deterministic flight-planning calculations for the optional dashboard and CLI."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


class FlightPlanningError(ValueError):
    """Raised when a flight-plan input cannot produce a meaningful result."""


@dataclass(frozen=True)
class FlightPlan:
    slant_distance_m: float
    view_angle_from_normal_deg: float
    thermal_horizontal_fov_deg: float
    thermal_vertical_fov_deg: float
    footprint_width_m: float
    footprint_height_m: float
    thermal_gsd_cm_px: float
    pixels_per_smallest_cell_feature: float
    module_width_px: float
    module_height_px: float
    maximum_height_for_resolution_m: float
    along_track_step_m: float
    cross_track_step_m: float
    overlap_limited_speed_m_s: float
    estimated_flight_lines: int | None
    estimated_images: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagonal_to_axis_fov(
    diagonal_fov_deg: float,
    width_px: int,
    height_px: int,
) -> tuple[float, float]:
    """Convert rectilinear diagonal FOV to horizontal and vertical FOV."""
    if not 0 < diagonal_fov_deg < 180:
        raise FlightPlanningError("diagonal FOV must be between 0 and 180 degrees")
    if width_px <= 0 or height_px <= 0:
        raise FlightPlanningError("camera dimensions must be positive")
    diagonal = math.hypot(width_px, height_px)
    diagonal_tangent = math.tan(math.radians(diagonal_fov_deg) / 2)
    horizontal = 2 * math.atan(diagonal_tangent * width_px / diagonal)
    vertical = 2 * math.atan(diagonal_tangent * height_px / diagonal)
    return math.degrees(horizontal), math.degrees(vertical)


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise FlightPlanningError(f"{name} must be a positive finite number")


def _capture_positions(span_m: float, footprint_m: float, step_m: float) -> int:
    if span_m <= footprint_m:
        return 1
    return math.ceil((span_m - footprint_m) / step_m) + 1


def plan_flight(
    *,
    thermal_width_px: int,
    thermal_height_px: int,
    thermal_diagonal_fov_deg: float,
    height_above_module_m: float,
    panel_tilt_deg: float,
    camera_aligned_to_panel: bool,
    forward_overlap_pct: float,
    side_overlap_pct: float,
    capture_interval_s: float,
    planned_speed_m_s: float,
    smallest_cell_feature_mm: float,
    module_width_m: float,
    module_height_m: float,
    site_length_m: float | None,
    site_width_m: float | None,
    required_pixels_per_cell: float = 5.0,
    long_image_axis_along_flight: bool = True,
    tilt_along_image_height: bool = True,
) -> FlightPlan:
    """Calculate approximate native-thermal coverage at a planar module surface.

    Height is vertical distance above the module plane at the ray intersection. When
    the camera is aligned with the panel normal, the slant distance grows with tilt.
    A nadir view uses a conservative one-axis projection for the tilted surface.
    """
    for value, name in (
        (height_above_module_m, "height"),
        (capture_interval_s, "capture interval"),
        (planned_speed_m_s, "planned speed"),
        (smallest_cell_feature_mm, "smallest cell feature"),
        (module_width_m, "module width"),
        (module_height_m, "module height"),
        (required_pixels_per_cell, "required pixels per cell"),
    ):
        _positive(float(value), name)
    if not 0 <= panel_tilt_deg < 90:
        raise FlightPlanningError("panel tilt must be in [0, 90) degrees")
    if not 0 <= forward_overlap_pct < 100 or not 0 <= side_overlap_pct < 100:
        raise FlightPlanningError("overlaps must be in [0, 100) percent")
    if (site_length_m is None) != (site_width_m is None):
        raise FlightPlanningError("site length and width must be provided together")
    if site_length_m is not None and site_width_m is not None:
        _positive(float(site_length_m), "site length")
        _positive(float(site_width_m), "site width")

    horizontal_fov, vertical_fov = diagonal_to_axis_fov(
        thermal_diagonal_fov_deg, thermal_width_px, thermal_height_px,
    )
    tilt_rad = math.radians(panel_tilt_deg)
    if camera_aligned_to_panel:
        slant_distance = height_above_module_m / math.cos(tilt_rad)
        view_angle = 0.0
        projection_scale = 1.0
    else:
        slant_distance = height_above_module_m
        view_angle = panel_tilt_deg
        projection_scale = 1 / math.cos(tilt_rad)

    footprint_width = 2 * slant_distance * math.tan(math.radians(horizontal_fov) / 2)
    footprint_height = 2 * slant_distance * math.tan(math.radians(vertical_fov) / 2)
    if tilt_along_image_height:
        footprint_height *= projection_scale
    else:
        footprint_width *= projection_scale
    gsd_width = footprint_width / thermal_width_px
    gsd_height = footprint_height / thermal_height_px
    limiting_gsd = max(gsd_width, gsd_height)
    pixels_per_cell = (smallest_cell_feature_mm / 1000) / limiting_gsd
    maximum_height = height_above_module_m * pixels_per_cell / required_pixels_per_cell

    if long_image_axis_along_flight:
        along_footprint = max(footprint_width, footprint_height)
        cross_footprint = min(footprint_width, footprint_height)
    else:
        along_footprint = min(footprint_width, footprint_height)
        cross_footprint = max(footprint_width, footprint_height)
    along_step = along_footprint * (1 - forward_overlap_pct / 100)
    cross_step = cross_footprint * (1 - side_overlap_pct / 100)
    speed_limit = along_step / capture_interval_s
    if site_length_m is None or site_width_m is None:
        lines = None
        estimated_images = None
    else:
        lines = _capture_positions(site_width_m, cross_footprint, cross_step)
        images_per_line = _capture_positions(site_length_m, along_footprint, along_step)
        estimated_images = lines * images_per_line

    return FlightPlan(
        slant_distance_m=slant_distance,
        view_angle_from_normal_deg=view_angle,
        thermal_horizontal_fov_deg=horizontal_fov,
        thermal_vertical_fov_deg=vertical_fov,
        footprint_width_m=footprint_width,
        footprint_height_m=footprint_height,
        thermal_gsd_cm_px=limiting_gsd * 100,
        pixels_per_smallest_cell_feature=pixels_per_cell,
        module_width_px=module_width_m / limiting_gsd,
        module_height_px=module_height_m / limiting_gsd,
        maximum_height_for_resolution_m=maximum_height,
        along_track_step_m=along_step,
        cross_track_step_m=cross_step,
        overlap_limited_speed_m_s=speed_limit,
        estimated_flight_lines=lines,
        estimated_images=estimated_images,
    )
