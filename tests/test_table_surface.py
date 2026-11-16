import numpy as np
import pytest

pytest.importorskip("cv2")

from pvdelta.table_surface import (
    brown_project,
    fit_piecewise_table_mapping,
    fit_south_plane,
    fit_table_mapping,
    map_piecewise_table_pixels,
    map_table_pixels,
    propose_tables,
)


@pytest.fixture
def camera():
    return {"projection_type": "brown", "width": 1600, "height": 1200, "focal_x": .7,
            "focal_y": .71, "c_x": .001, "c_y": -.002, "k1": .02, "k2": -.01,
            "k3": .001, "p1": .0001, "p2": -.0002}


def test_north_edge_is_higher_and_wrong_slope_fails_holdout():
    rng = np.random.default_rng(42)
    xy = rng.uniform([0, 0], [20, 4], (200, 2))
    z = 100 + np.tan(np.deg2rad(25))*xy[:, 1]
    offset, quality = fit_south_plane(np.column_stack([xy, z]))
    assert offset == pytest.approx(100)
    assert quality['held_out_p95_abs_m'] < 1e-10
    _, wrong = fit_south_plane(np.column_stack([xy, 200-z]))
    assert wrong['held_out_p95_abs_m'] > 1


@pytest.mark.parametrize('points', [np.zeros((39, 3)), np.full((50, 3), np.nan)])
def test_invalid_plane_support_rejected(points):
    with pytest.raises(ValueError):
        fit_south_plane(points)


def test_projection_uses_opensfm_pixel_centre_and_accepts_noncontiguous(camera):
    points = np.zeros((4, 6))[:, ::2]
    points[:, 2] = 20
    uv, depth = brown_project(points, [0, 0, 0], [0, 0, 0], camera)
    np.testing.assert_allclose(uv, np.tile([801.1, 596.3], (4, 1)))
    np.testing.assert_allclose(depth, 20)


def test_local_mapping_reconstructs_unseen_distorted_points(camera):
    rng = np.random.default_rng(3)
    xy = rng.uniform([-5, -2], [5, 2], (240, 2))
    xyz = np.column_stack([xy, 20+.4663*xy[:, 1]])
    pixels, _ = brown_project(xyz, [.02, .01, -.04], [.2, -.3, 0], camera)
    matrix, evidence = fit_table_mapping(xy, pixels, camera)
    query = rng.uniform([-5, -2], [5, 2], (25, 2))
    expected, _ = brown_project(np.column_stack([query, 20+.4663*query[:, 1]]),
                                [.02, .01, -.04], [.2, -.3, 0], camera)
    np.testing.assert_allclose(map_table_pixels(query, matrix, camera), expected, atol=.001)
    assert evidence['p95_px'] < .001


def test_bad_holdout_cannot_be_hidden_by_ransac(camera):
    rng = np.random.default_rng(7)
    xy = rng.uniform(-3, 3, (200, 2))
    pixels, _ = brown_project(np.column_stack([xy, np.full(200, 20)]),
                              [0, 0, 0], [0, 0, 0], camera)
    pixels[::4] += 15
    with pytest.raises(ValueError, match='Held-out'):
        fit_table_mapping(xy, pixels, camera)


def test_table_candidates_preserve_variable_spacing_and_reject_partial():
    rgb = np.full((180, 800, 3), [80, 120, 30], dtype=np.uint8)
    rgb[20:60, 20:225] = [40, 60, 90]
    rgb[100:140, 400:605] = [40, 60, 90]
    rgb[0:40, 570:775] = [40, 60, 90]
    candidates, rejected = propose_tables(rgb, .1)
    assert len(candidates) == 2
    assert all(t['columns'] == 9 for t in candidates)
    assert any(t['reason'] == 'image_edge_partial' for t in rejected)
    partial = next(t for t in rejected if t['reason'] == 'image_edge_partial')
    assert partial['box'][2] > 570  # right coordinate, not component width


def test_merged_eighteen_columns_need_observed_seam():
    rgb = np.full((80, 460, 3), [80, 120, 30], dtype=np.uint8)
    rgb[20:60, 20:430] = [40, 60, 90]
    candidates, rejected = propose_tables(rgb, .1)
    assert not candidates
    assert rejected[0]['reason'] == 'no_supported_table_split'
    rgb[20:60, 225:227] = 5
    candidates, _ = propose_tables(rgb, .1)
    assert len(candidates) == 2


def test_piecewise_field_handles_local_deformation_without_relaxing_limits(camera):
    rng = np.random.default_rng(41)
    xy = rng.uniform([-10, -2], [10, 2], (1200, 2))
    pixels, _ = brown_project(np.column_stack([xy, np.full(len(xy), 25.)]),
                              [0, 0, 0], [0, 0, 0], camera)
    pixels[:, 1] += .01*xy[:, 0]**3
    with pytest.raises(ValueError):
        fit_table_mapping(xy, pixels, camera)
    model, evidence = fit_piecewise_table_mapping(xy, pixels, camera, [-10, 10])
    assert evidence['held_out_features'] == 300
    assert evidence['median_px'] < 2 and evidence['p95_px'] < 5
    prediction = map_piecewise_table_pixels(xy[::4], model, camera)
    assert np.percentile(np.linalg.norm(prediction-pixels[::4], axis=1), 95) < 5


def test_piecewise_fit_cannot_use_corrupted_holdout(camera):
    rng = np.random.default_rng(2)
    xy = rng.uniform([-10, -2], [10, 2], (800, 2))
    pixels, _ = brown_project(np.column_stack([xy, np.full(len(xy), 25.)]),
                              [0, 0, 0], [0, 0, 0], camera)
    pixels[::4] += 15
    with pytest.raises(ValueError, match='Held-out piecewise'):
        fit_piecewise_table_mapping(xy, pixels, camera, [-10, 10])


def test_piecewise_mapping_preserves_unsupported_regions(camera):
    rng = np.random.default_rng(3)
    xy = rng.uniform([-10, -2], [-3, 2], (500, 2))
    pixels, _ = brown_project(np.column_stack([xy, np.full(len(xy), 25.)]),
                              [0, 0, 0], [0, 0, 0], camera)
    model, _ = fit_piecewise_table_mapping(xy, pixels, camera, [-10, 10])
    mapped = map_piecewise_table_pixels(np.array([[-7., 0.], [7., 0.]]), model, camera)
    assert np.isfinite(mapped[0]).all()
    assert np.isnan(mapped[1]).all()


def test_piecewise_mapping_rejects_collinear_support(camera):
    xy = np.column_stack([np.linspace(-10, 10, 100), np.zeros(100)])
    pixels, _ = brown_project(np.column_stack([xy, np.full(len(xy), 25.)]),
                              [0, 0, 0], [0, 0, 0], camera)
    with pytest.raises(ValueError, match='support'):
        fit_piecewise_table_mapping(xy, pixels, camera, [-10, 10])
