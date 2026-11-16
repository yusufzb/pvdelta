import numpy as np
import pytest

from pvdelta.alignment import (
    assess_boundary_transfer,
    fit_homography,
    pixel_grid_transform,
    planar_motion_to_image_shift,
    transform_points,
    validate_homography_checks,
)


def test_boundary_transfer_gate_requires_native_inside_and_bounded_edge_fit():
    polygon = np.array([[10, 10], [30, 10], [30, 25], [10, 25]], dtype=float)
    accepted = assess_boundary_transfer(
        polygon, frame_shape=(100, 120), edge_score_px=2.0, translation_xy=(4.0, -3.0)
    )
    assert accepted.accepted
    rejected = assess_boundary_transfer(
        polygon, frame_shape=(100, 120), edge_score_px=9.0, translation_xy=(4.0, -3.0)
    )
    assert not rejected.accepted
    assert "edge score" in rejected.reasons[0]


def test_boundary_transfer_gate_rejects_outside_polygon_and_large_shift():
    result = assess_boundary_transfer(
        np.array([[-1, 10], [30, 10], [30, 25]], dtype=float),
        frame_shape=(100, 120), edge_score_px=1.0, translation_xy=(13.0, 0.0)
    )
    assert not result.accepted
    assert any("outside" in reason for reason in result.reasons)
    assert any("translation" in reason for reason in result.reasons)


def test_homography_recovers_known_projective_mapping():
    source = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [4, 7]], dtype=float)
    expected_matrix = np.array([[2, .1, 5], [.2, 3, 7], [.001, -.002, 1]])
    target = transform_points(source, expected_matrix)
    fit = fit_homography(source, target)
    np.testing.assert_allclose(transform_points(source, fit.matrix), target, atol=1e-8)
    assert fit.rmse_px < 1e-8


def test_native_preview_pixel_centres_round_trip_with_half_pixel_offset():
    mapping = pixel_grid_transform((512, 640), (1024, 1280))
    native = np.array([[0, 0], [639, 0], [0, 511], [639, 511], [123.25, 91.75]])
    preview = mapping.forward(native)
    np.testing.assert_allclose(
        preview[:4], [[0.5, 0.5], [1278.5, 0.5], [0.5, 1022.5], [1278.5, 1022.5]]
    )
    np.testing.assert_allclose(mapping.inverse(preview), native, atol=1e-12)


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_orthogonal_grid_orientation_is_reversible(turns):
    target_shape = (8, 6) if turns % 2 else (6, 8)
    mapping = pixel_grid_transform((3, 4), target_shape, clockwise_quarter_turns=turns)
    asymmetric = np.array([[0, 0], [3, 0], [0, 2], [2, 1]], dtype=float)
    np.testing.assert_allclose(mapping.inverse(mapping.forward(asymmetric)), asymmetric, atol=1e-12)


def test_nonuniform_grid_scale_is_rejected():
    with pytest.raises(ValueError, match="aspect ratio"):
        pixel_grid_transform((512, 640), (1024, 1279))


def test_registration_gate_uses_held_out_points_and_rejects_panel_period_shift():
    source = np.array([[0, 0], [100, 0], [0, 80], [100, 80], [17, 43]], dtype=float)
    matrix = np.array([[1.4, 0.03, 200], [-0.02, 1.3, 90], [0.0001, -0.0002, 1]])
    target = transform_points(source, matrix)
    valid = validate_homography_checks(
        matrix,
        fit_point_count=6,
        check_source_xy=source,
        check_target_xy=target,
        max_p95_source_px=2,
    )
    assert valid.accepted
    shifted_matrix = matrix.copy()
    shifted_matrix[0, 2] += 30
    shifted = validate_homography_checks(
        shifted_matrix,
        fit_point_count=6,
        check_source_xy=source,
        check_target_xy=target,
        max_p95_source_px=2,
    )
    assert not shifted.accepted
    assert shifted.check_p95_source_px > 10


def test_planar_motion_uses_yaw_and_scene_moves_opposite_camera():
    assert planar_motion_to_image_shift(0.10, 0.0, gimbal_yaw_deg=0.0, ground_sample_distance_m_px=0.01) == pytest.approx((0.0, -10.0))
    assert planar_motion_to_image_shift(0.0, 0.10, gimbal_yaw_deg=0.0, ground_sample_distance_m_px=0.01) == pytest.approx((-10.0, 0.0))


def test_planar_motion_rejects_invalid_gsd():
    with pytest.raises(ValueError, match="GSD"):
        planar_motion_to_image_shift(0.0, 0.0, gimbal_yaw_deg=0.0, ground_sample_distance_m_px=0.0)
