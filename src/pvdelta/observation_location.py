"""Project native module-observation centres onto a reviewed panel plane."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from pvdelta.geolocation import FootprintRaycastError, LrfPlanarProjector
from pvdelta.tracking import quadrilateral_center


def project_observation_center(
    polygon_json: str | list[list[float]] | list[tuple[float, float]],
    *,
    target_latitude: float,
    target_longitude: float,
    laser_distance_m: float,
    gimbal_pitch_deg: float,
    gimbal_yaw_deg: float,
    gimbal_roll_deg: float,
    image_width_px: int = 640,
    image_height_px: int = 512,
    diagonal_fov_deg: float = 45.0,
    surface_tilt_deg: float = 25.0,
    surface_azimuth_deg: float = 180.0,
) -> dict[str, Any]:
    """Return an estimated WGS84 centre from a native quadrilateral.

    The plane is through the recorded LRF target, with a documented panel tilt and
    azimuth. Camera GPS is intentionally not used as the module location anchor.
    The result is an estimate until independent boresight and surface controls pass.
    """
    polygon = json.loads(polygon_json) if isinstance(polygon_json, str) else polygon_json
    points = np.asarray(polygon, dtype=float)
    if points.shape != (4, 2):
        raise ValueError("module polygon must contain four native quadrilateral points")
    x_px, y_px = quadrilateral_center(points)
    projector = LrfPlanarProjector(
        target_latitude=target_latitude,
        target_longitude=target_longitude,
        laser_distance_m=laser_distance_m,
        gimbal_pitch_deg=gimbal_pitch_deg,
        gimbal_yaw_deg=gimbal_yaw_deg,
        gimbal_roll_deg=gimbal_roll_deg,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        diagonal_fov_deg=diagonal_fov_deg,
        surface_tilt_deg=surface_tilt_deg,
        surface_azimuth_deg=surface_azimuth_deg,
    )
    projected = projector.project_pixel(x_px, y_px)
    return {
        "latitude": projected.latitude,
        "longitude": projected.longitude,
        "east_m": projected.east_m,
        "north_m": projected.north_m,
        "up_m": projected.up_m,
        "ray_distance_m": projected.ray_distance_m,
        "pixel_center_xy": [x_px, y_px],
        "method": "lrf_center_planar_raycast_v1_provisional",
        "surface_tilt_deg": surface_tilt_deg,
        "surface_azimuth_deg": surface_azimuth_deg,
        "view_angle_from_normal_deg": projector.view_angle_from_normal_deg,
        "view_direction_enu": list(projector.pixel_view_direction_enu(x_px, y_px)),
    }


__all__ = ["FootprintRaycastError", "project_observation_center"]
