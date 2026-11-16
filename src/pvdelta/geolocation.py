"""Project thermal-image coordinates onto a reviewed planar surface.

The projector uses the recorded laser target as the image-centre anchor. Camera GPS is not
used as a panel coordinate, and the resulting geometry remains an estimate whose surface
assumption and accuracy must be recorded separately before persistence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pvdelta.flight import FlightPlanningError, diagonal_to_axis_fov

_WGS84_A_M = 6_378_137.0
_WGS84_E2 = 6.69437999014e-3
_EPSILON = 1e-9


class FootprintRaycastError(ValueError):
    """Raised when rays cannot produce a defensible planar intersection."""


@dataclass(frozen=True)
class BoresightValidation:
    """Result of an independent camera/LRF boresight consistency check.

    A camera-GPS versus LRF residual can diagnose the flight metadata, but it is not an
    independent boresight calibration target. Callers must set ``independent_control`` only
    when the observed coordinates come from surveyed/asymmetric control points.
    """

    status: str
    accepted: bool
    sample_count: int
    median_horizontal_error_m: float | None
    p95_horizontal_error_m: float | None
    max_horizontal_error_m: float | None
    reasons: tuple[str, ...]


def validate_boresight_checks(
    observed_enu_m: Sequence[tuple[float, float]],
    predicted_enu_m: Sequence[tuple[float, float]],
    *,
    max_p95_error_m: float,
    independent_control: bool,
    min_samples: int = 4,
) -> BoresightValidation:
    """Gate boresight residuals using held-out, independent ENU control points.

    ``observed_enu_m`` and ``predicted_enu_m`` are ``(east, north)`` pairs in the same local
    frame. The check reports residual statistics even when it cannot accept them. This keeps the
    useful camera-GPS/LRF diagnostic visible while preventing it from being labelled calibrated.
    """
    if not math.isfinite(max_p95_error_m) or max_p95_error_m <= 0:
        raise ValueError("max_p95_error_m must be positive and finite")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    if len(observed_enu_m) != len(predicted_enu_m):
        raise ValueError("observed and predicted control points must have equal length")
    errors: list[float] = []
    for observed, predicted in zip(observed_enu_m, predicted_enu_m, strict=True):
        if len(observed) != 2 or len(predicted) != 2:
            raise ValueError("control points must be (east, north) pairs")
        values = (*observed, *predicted)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("control points must be finite")
        errors.append(math.hypot(float(observed[0]) - float(predicted[0]), float(observed[1]) - float(predicted[1])))
    reasons: list[str] = []
    if len(errors) < min_samples:
        reasons.append(f"requires at least {min_samples} independent control points")
    if not independent_control:
        reasons.append("independent surveyed/asymmetric control points are required")
    if errors:
        ordered = sorted(errors)
        median = float(_percentile(ordered, 50.0))
        p95 = float(_percentile(ordered, 95.0))
        maximum = float(max(ordered))
        if p95 > max_p95_error_m:
            reasons.append(f"control-point P95 {p95:.3f} m exceeds {max_p95_error_m:.3f} m")
    else:
        median = p95 = maximum = None
    accepted = not reasons
    if accepted:
        status = "validated"
    elif independent_control and errors and any("P95" in reason for reason in reasons):
        status = "rejected"
    else:
        status = "not_validated"
    return BoresightValidation(status, accepted, len(errors), median, p95, maximum, tuple(reasons))


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


@dataclass(frozen=True)
class ProjectedPoint:
    """One image coordinate projected relative to the laser-target anchor."""

    latitude: float
    longitude: float
    east_m: float
    north_m: float
    up_m: float
    ray_distance_m: float


@dataclass(frozen=True)
class RaycastFootprint:
    """Image-boundary projection with method and geometry diagnostics."""

    geojson: dict[str, Any]
    corners: tuple[ProjectedPoint, ...]
    horizontal_fov_deg: float
    vertical_fov_deg: float
    view_angle_from_normal_deg: float
    surface_area_m2: float
    method: str = "lrf_center_planar_raycast_v1"


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise FootprintRaycastError(f"{name} must be finite")
    return result


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = _norm(vector)
    if length <= _EPSILON:
        raise FootprintRaycastError("camera orientation produced a zero-length axis")
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def _subtract(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _triangle_area(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    third: tuple[float, float, float],
) -> float:
    return 0.5 * _norm(_cross(_subtract(second, first), _subtract(third, first)))


@dataclass(frozen=True)
class LrfPlanarProjector:
    """Ray-cast image coordinates onto a plane through the recorded LRF target.

    Angles use DJI-style navigation conventions: yaw/azimuth are clockwise from north,
    gimbal pitch is 0° at the horizon and -90° at nadir, and positive surface tilt raises
    the plane opposite its facing azimuth. Pixel coordinates are continuous image coordinates
    from the top-left boundary; the image centre is ``(width/2, height/2)``.
    """

    target_latitude: float
    target_longitude: float
    laser_distance_m: float
    gimbal_pitch_deg: float
    gimbal_yaw_deg: float
    gimbal_roll_deg: float
    image_width_px: int
    image_height_px: int
    diagonal_fov_deg: float
    surface_tilt_deg: float
    surface_azimuth_deg: float

    def __post_init__(self) -> None:
        latitude = _finite(self.target_latitude, "target latitude")
        longitude = _finite(self.target_longitude, "target longitude")
        if not -90 <= latitude <= 90:
            raise FootprintRaycastError("target latitude is outside [-90, 90]")
        if not -180 <= longitude <= 180:
            raise FootprintRaycastError("target longitude is outside [-180, 180]")
        if _finite(self.laser_distance_m, "laser distance") <= 0:
            raise FootprintRaycastError("laser distance must be positive")
        for value, name in (
            (self.gimbal_pitch_deg, "gimbal pitch"),
            (self.gimbal_yaw_deg, "gimbal yaw"),
            (self.gimbal_roll_deg, "gimbal roll"),
            (self.diagonal_fov_deg, "diagonal FOV"),
            (self.surface_tilt_deg, "surface tilt"),
            (self.surface_azimuth_deg, "surface azimuth"),
        ):
            _finite(value, name)
        if self.image_width_px <= 0 or self.image_height_px <= 0:
            raise FootprintRaycastError("image dimensions must be positive")
        if not 0 < self.diagonal_fov_deg < 180:
            raise FootprintRaycastError("diagonal FOV must be between 0 and 180 degrees")
        if not 0 <= self.surface_tilt_deg < 90:
            raise FootprintRaycastError("surface tilt must be in [0, 90) degrees")

    @property
    def axis_fov_deg(self) -> tuple[float, float]:
        try:
            return diagonal_to_axis_fov(
                self.diagonal_fov_deg, self.image_width_px, self.image_height_px
            )
        except FlightPlanningError as exc:  # kept as one geolocation-facing error type
            raise FootprintRaycastError(str(exc)) from exc

    def _camera_axes(
        self,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        yaw = math.radians(self.gimbal_yaw_deg)
        pitch = math.radians(self.gimbal_pitch_deg)
        roll = math.radians(self.gimbal_roll_deg)
        forward = _unit(
            (math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch))
        )
        right = _unit((math.cos(yaw), -math.sin(yaw), 0.0))
        down = _unit(_cross(forward, right))
        rolled_right = _unit(
            tuple(
                math.cos(roll) * r + math.sin(roll) * d
                for r, d in zip(right, down, strict=True)
            )
        )
        rolled_down = _unit(
            tuple(
                -math.sin(roll) * r + math.cos(roll) * d
                for r, d in zip(right, down, strict=True)
            )
        )
        return forward, rolled_right, rolled_down

    def _surface_normal(self) -> tuple[float, float, float]:
        tilt = math.radians(self.surface_tilt_deg)
        azimuth = math.radians(self.surface_azimuth_deg)
        return _unit(
            (
                math.sin(azimuth) * math.sin(tilt),
                math.cos(azimuth) * math.sin(tilt),
                math.cos(tilt),
            )
        )

    @property
    def camera_offset_enu_m(self) -> tuple[float, float, float]:
        """Return the camera position relative to the LRF target in east/north/up metres."""
        forward, _, _ = self._camera_axes()
        return tuple(-self.laser_distance_m * value for value in forward)  # type: ignore[return-value]

    @property
    def view_angle_from_normal_deg(self) -> float:
        """Return centre-ray incidence angle against the outward surface normal."""
        forward, _, _ = self._camera_axes()
        toward_camera = tuple(-value for value in forward)
        cosine = max(-1.0, min(1.0, _dot(toward_camera, self._surface_normal())))
        return math.degrees(math.acos(cosine))

    def _to_geographic(self, east_m: float, north_m: float) -> tuple[float, float]:
        latitude_rad = math.radians(self.target_latitude)
        sin_latitude = math.sin(latitude_rad)
        denominator = math.sqrt(1 - _WGS84_E2 * sin_latitude**2)
        prime_vertical_radius = _WGS84_A_M / denominator
        meridional_radius = (
            _WGS84_A_M * (1 - _WGS84_E2) / denominator**3
        )
        latitude = self.target_latitude + math.degrees(north_m / meridional_radius)
        longitude_scale = prime_vertical_radius * math.cos(latitude_rad)
        if abs(longitude_scale) <= _EPSILON:
            raise FootprintRaycastError("longitude is undefined at the selected latitude")
        longitude = self.target_longitude + math.degrees(east_m / longitude_scale)
        return latitude, longitude

    def pixel_view_direction_enu(self, x_px: float, y_px: float) -> tuple[float, float, float]:
        """Unit direction from this pixel's surface point toward the camera (ENU)."""
        x = _finite(x_px, "pixel x")
        y = _finite(y_px, "pixel y")
        if not 0 <= x <= self.image_width_px or not 0 <= y <= self.image_height_px:
            raise FootprintRaycastError("pixel coordinate lies outside the image boundary")

        horizontal_fov, vertical_fov = self.axis_fov_deg
        forward, right, down = self._camera_axes()
        if _dot(tuple(-value for value in forward), self._surface_normal()) <= 0:
            raise FootprintRaycastError("camera is behind the selected surface face")
        x_scale = (2 * x / self.image_width_px - 1) * math.tan(
            math.radians(horizontal_fov) / 2
        )
        y_scale = (2 * y / self.image_height_px - 1) * math.tan(
            math.radians(vertical_fov) / 2
        )
        ray = _unit(
            tuple(
                f + x_scale * r + y_scale * d
                for f, r, d in zip(forward, right, down, strict=True)
            )
        )
        return tuple(-value for value in ray)

    def project_pixel(self, x_px: float, y_px: float) -> ProjectedPoint:
        """Project one continuous image coordinate onto the configured surface plane."""
        ray = tuple(-value for value in self.pixel_view_direction_enu(x_px, y_px))
        camera = self.camera_offset_enu_m
        normal = self._surface_normal()
        denominator = _dot(ray, normal)
        numerator = -_dot(camera, normal)
        if abs(denominator) <= _EPSILON:
            raise FootprintRaycastError("image ray is parallel to the selected surface")
        ray_distance = numerator / denominator
        if ray_distance <= 0:
            raise FootprintRaycastError("image ray intersects the selected surface behind camera")
        east, north, up = tuple(
            origin + ray_distance * direction
            for origin, direction in zip(camera, ray, strict=True)
        )
        latitude, longitude = self._to_geographic(east, north)
        return ProjectedPoint(latitude, longitude, east, north, up, ray_distance)

    def footprint(
        self, *, max_view_angle_from_normal_deg: float | None = None
    ) -> RaycastFootprint:
        """Project the four image boundaries and return a closed GeoJSON polygon."""
        view_angle = self.view_angle_from_normal_deg
        if max_view_angle_from_normal_deg is not None:
            maximum = _finite(max_view_angle_from_normal_deg, "maximum view angle")
            if not 0 < maximum < 90:
                raise FootprintRaycastError("maximum view angle must be between 0 and 90 degrees")
            if view_angle > maximum:
                raise FootprintRaycastError(
                    f"view angle {view_angle:.2f}° exceeds {maximum:.2f}° limit"
                )
        corners = tuple(
            self.project_pixel(x, y)
            for x, y in (
                (0.0, 0.0),
                (float(self.image_width_px), 0.0),
                (float(self.image_width_px), float(self.image_height_px)),
                (0.0, float(self.image_height_px)),
            )
        )
        coordinates = [[point.longitude, point.latitude] for point in corners]
        coordinates.append(coordinates[0])
        local = tuple((point.east_m, point.north_m, point.up_m) for point in corners)
        area = _triangle_area(local[0], local[1], local[2]) + _triangle_area(
            local[0], local[2], local[3]
        )
        horizontal_fov, vertical_fov = self.axis_fov_deg
        return RaycastFootprint(
            geojson={"type": "Polygon", "coordinates": [coordinates]},
            corners=corners,
            horizontal_fov_deg=horizontal_fov,
            vertical_fov_deg=vertical_fov,
            view_angle_from_normal_deg=view_angle,
            surface_area_m2=area,
        )
