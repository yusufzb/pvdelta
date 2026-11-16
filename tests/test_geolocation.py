import math

import pytest

from pvdelta.flight import diagonal_to_axis_fov
from pvdelta.geolocation import (
    FootprintRaycastError,
    LrfPlanarProjector,
    validate_boresight_checks,
)


def _projector(**overrides):
    values = {
        "target_latitude": 40.0,
        "target_longitude": 29.0,
        "laser_distance_m": 20.0,
        "gimbal_pitch_deg": -90.0,
        "gimbal_yaw_deg": 0.0,
        "gimbal_roll_deg": 0.0,
        "image_width_px": 640,
        "image_height_px": 512,
        "diagonal_fov_deg": 45.0,
        "surface_tilt_deg": 0.0,
        "surface_azimuth_deg": 180.0,
    }
    values.update(overrides)
    return LrfPlanarProjector(**values)


def test_image_centre_is_the_laser_target_anchor():
    projector = _projector(surface_tilt_deg=25.0)
    point = projector.project_pixel(320, 256)
    assert point.east_m == pytest.approx(0, abs=1e-9)
    assert point.north_m == pytest.approx(0, abs=1e-9)
    assert point.up_m == pytest.approx(0, abs=1e-9)
    assert point.latitude == pytest.approx(40.0)
    assert point.longitude == pytest.approx(29.0)
    assert projector.view_angle_from_normal_deg == pytest.approx(25.0)
    assert projector.camera_offset_enu_m == pytest.approx((0.0, 0.0, 20.0), abs=1e-9)


def test_nadir_horizontal_footprint_matches_rectilinear_fov():
    projector = _projector()
    footprint = projector.footprint()
    horizontal_fov, vertical_fov = diagonal_to_axis_fov(45, 640, 512)
    expected_width = 2 * 20 * math.tan(math.radians(horizontal_fov) / 2)
    expected_height = 2 * 20 * math.tan(math.radians(vertical_fov) / 2)
    east_span = max(point.east_m for point in footprint.corners) - min(
        point.east_m for point in footprint.corners
    )
    north_span = max(point.north_m for point in footprint.corners) - min(
        point.north_m for point in footprint.corners
    )
    assert east_span == pytest.approx(expected_width)
    assert north_span == pytest.approx(expected_height)
    assert footprint.surface_area_m2 == pytest.approx(expected_width * expected_height)
    assert footprint.geojson["coordinates"][0][0] == footprint.geojson["coordinates"][0][-1]


def test_yaw_rotates_the_image_axes_on_horizontal_surface():
    north_facing = _projector(gimbal_yaw_deg=0).footprint()
    east_facing = _projector(gimbal_yaw_deg=90).footprint()
    north_east_span = max(p.east_m for p in north_facing.corners) - min(
        p.east_m for p in north_facing.corners
    )
    east_north_span = max(p.north_m for p in east_facing.corners) - min(
        p.north_m for p in east_facing.corners
    )
    assert east_north_span == pytest.approx(north_east_span)


def test_roll_180_preserves_footprint_extent():
    normal = _projector(gimbal_roll_deg=0).footprint()
    inverted = _projector(gimbal_roll_deg=180).footprint()
    normal_coordinates = sorted((round(p.east_m, 8), round(p.north_m, 8)) for p in normal.corners)
    inverted_coordinates = sorted(
        (round(p.east_m, 8), round(p.north_m, 8)) for p in inverted.corners
    )
    assert inverted_coordinates == normal_coordinates


def test_tilted_surface_changes_corner_height_but_keeps_points_on_plane():
    projector = _projector(surface_tilt_deg=25, surface_azimuth_deg=180)
    footprint = projector.footprint()
    assert max(point.up_m for point in footprint.corners) > 0
    assert min(point.up_m for point in footprint.corners) < 0
    for point in footprint.corners:
        assert point.up_m == pytest.approx(point.north_m * math.tan(math.radians(25)))


def test_oblique_view_rejects_corner_ray_that_points_away_from_plane():
    projector = _projector(gimbal_pitch_deg=-5)
    with pytest.raises(FootprintRaycastError, match="behind camera"):
        projector.footprint()


def test_view_angle_gate_rejects_front_facing_but_over_oblique_capture():
    projector = _projector(gimbal_pitch_deg=-30, surface_tilt_deg=0)
    with pytest.raises(FootprintRaycastError, match="exceeds"):
        projector.footprint(max_view_angle_from_normal_deg=45)


def test_camera_behind_surface_face_is_rejected():
    projector = _projector(gimbal_pitch_deg=10, surface_tilt_deg=0)
    with pytest.raises(FootprintRaycastError, match="behind"):
        projector.project_pixel(320, 256)


@pytest.mark.parametrize(
    "field,value",
    [
        ("laser_distance_m", 0),
        ("surface_tilt_deg", 90),
        ("target_latitude", 91),
        ("diagonal_fov_deg", float("nan")),
    ],
)
def test_invalid_inputs_are_rejected(field, value):
    with pytest.raises(FootprintRaycastError):
        _projector(**{field: value})


def test_pixel_outside_image_is_rejected():
    with pytest.raises(FootprintRaycastError, match="outside"):
        _projector().project_pixel(641, 10)


def test_boresight_gate_accepts_independent_control_points():
    result = validate_boresight_checks(
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
        [(0.01, 0.0), (1.0, 0.01), (0.0, 1.01), (1.01, 1.0)],
        max_p95_error_m=0.05,
        independent_control=True,
    )
    assert result.status == "validated"
    assert result.accepted is True
    assert result.p95_horizontal_error_m == pytest.approx(0.01, abs=1e-6)


def test_boresight_gate_keeps_camera_gps_diagnostic_unvalidated():
    result = validate_boresight_checks(
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
        [(0.1, 0.0), (1.1, 0.0), (0.1, 1.0), (1.1, 1.0)],
        max_p95_error_m=0.5,
        independent_control=False,
    )
    assert result.status == "not_validated"
    assert result.accepted is False
    assert "independent surveyed/asymmetric control points" in result.reasons[0]


def test_boresight_gate_rejects_large_held_out_residual():
    result = validate_boresight_checks(
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 2.0)],
        max_p95_error_m=0.5,
        independent_control=True,
    )
    assert result.status == "rejected"
    assert result.accepted is False
    assert result.p95_horizontal_error_m > 0.5


def test_boresight_gate_requires_enough_controls():
    result = validate_boresight_checks(
        [(0.0, 0.0)],
        [(0.0, 0.0)],
        max_p95_error_m=0.5,
        independent_control=True,
    )
    assert result.status == "not_validated"
    assert result.accepted is False
