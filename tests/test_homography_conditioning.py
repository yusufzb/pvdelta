"""Coordinate conditioning and held-out safety, independent of the PV fixture."""
import numpy as np
import pytest

from pvdelta.alignment import fit_homography, transform_points, validate_homography_checks


def test_noisy_fit_is_invariant_to_pixel_origin_and_uniform_unit_changes():
    rng = np.random.default_rng(319)
    source = rng.uniform([120, 80], [520, 410], (32, 2))
    expected = np.array([[2.8, .07, 1500], [-.03, 2.7, 650], [.00003, -.00002, 1]])
    target = transform_points(source, expected)+rng.normal(0, 1.2, source.shape)
    first = fit_homography(source, target)
    source_offset, target_offset = np.array([2e5, -3e5]), np.array([-4e5, 5e5])
    second = fit_homography(source*7+source_offset, target*4+target_offset)
    observed = (transform_points(source*7+source_offset, second.matrix)-target_offset)/4
    np.testing.assert_allclose(observed, transform_points(source, first.matrix), atol=1e-7, rtol=0)


def test_exact_minimal_four_point_fit_keeps_the_ninth_null_space_vector():
    source = np.array([[380, 220], [410, 220], [410, 330], [380, 330]], float)
    expected = np.array([[2.8, .07, 1500], [-.03, 2.7, 650], [.00003, -.00002, 1]])
    target = transform_points(source, expected)
    model = fit_homography(source, target)
    check = np.array([[390, 240], [395, 300], [400, 290], [405, 325]])
    np.testing.assert_allclose(transform_points(check, model.matrix),
                               transform_points(check, expected), atol=1e-8)


@pytest.mark.parametrize('kind', ['source_line', 'target_line', 'coincident', 'repeated'])
def test_rank_deficient_controls_are_withheld(kind):
    source = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    target = source*3+[1700, 900]
    if kind == 'source_line': source[:, 1] = 0
    if kind == 'target_line': target[:, 1] = 900
    if kind == 'coincident': source[:] = [100, 100]
    if kind == 'repeated': source[3] = source[2]; target[3] = target[2]
    with pytest.raises(ValueError, match='degenerate|spread|rank'):
        fit_homography(source, target)


def test_conditioning_cannot_fit_or_hide_an_unused_wrong_panel():
    source = np.array([[100, 50], [300, 50], [500, 50], [100, 400], [300, 400], [500, 400]], float)
    target = source*3+[1000, 700]
    fit = fit_homography(source, target)
    checks = np.array([[240, 160], [280, 160], [280, 260], [240, 260]], float)
    wrong = checks*3+[1000+120, 700]
    result = validate_homography_checks(fit.matrix, fit_point_count=len(source),
        check_source_xy=checks, check_target_xy=wrong, max_p95_source_px=2)
    assert not result.accepted and result.check_p95_source_px > 39


def test_nonfinite_controls_still_fail_before_fitting():
    source = np.array([[0, 0], [1, 0], [1, 1], [0, np.nan]])
    with pytest.raises(ValueError, match='finite'):
        fit_homography(source, source)
