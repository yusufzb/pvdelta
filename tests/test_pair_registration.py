import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from pvdelta.pair_registration import (
    PairRegistrationConfig,
    apply_correction,
    check_registration_controls,
    refine_panel_registration,
)


def scene(matrix=None, *, omit=()):
    polygons = []
    for col in range(3):
        for row in range(4):
            x, y = 100+col*125, 65+row*88
            polygons.append(np.float32([[x, y], [x+112, y], [x+112, y+74], [x, y+74]]))
    matrix = np.float64([[1, 0, 3], [0, 1, -2]]) if matrix is None else matrix
    image = np.full((512, 640), 25, np.uint8)
    actual = [apply_correction(p, matrix) for p in polygons]
    for i, p in enumerate(actual):
        if i not in omit:
            cv2.fillConvexPoly(image, np.rint(p).astype(np.int32), 150)
    return image, polygons, actual


@pytest.mark.parametrize('matrix', [
    np.float64([[1, 0, 3], [0, 1, -2]]),
    np.float64([[1.006, -.008, 3], [.005, .996, -2]]),
])
def test_known_bounded_correction_uses_excluded_module_checks(matrix):
    image, prior, truth = scene(matrix)
    before = image.copy()
    result = refine_panel_registration(image, prior)
    assert result.image_consistent, result.to_dict()
    assert set(result.train_module_indices).isdisjoint(result.check_module_indices)
    assert set(result.train_module_indices+result.check_module_indices) == set(range(len(prior)))
    errors = [np.linalg.norm(apply_correction(p, result.correction)-q, axis=1)
              for p, q in zip(prior, truth)]
    assert np.percentile(errors, 95) < 1.5
    assert result.check_p95_px <= 2
    assert np.array_equal(image, before)
    assert not result.correction.flags.writeable
    assert result.to_dict()['validation_scope'].endswith('_provisional')


def test_missing_check_modules_cannot_be_fitted_away():
    image, prior, _ = scene(omit=(1, 3, 5, 7, 9, 11))
    result = refine_panel_registration(image, prior)
    assert not result.image_consistent
    assert 'held-out' in result.reasons[0]
    assert result.correction is None


def test_uniform_surface_and_large_shift_are_rejected():
    image, prior, _ = scene()
    assert not refine_panel_registration(np.full_like(image, 90), prior).image_consistent
    image, prior, _ = scene(np.float64([[1, 0, 22], [0, 1, 0]]))
    assert not refine_panel_registration(image, prior).image_consistent


def test_repeated_panel_edges_cannot_establish_physical_correspondence():
    image, prior, _ = scene()
    # A one-column identity error can still align with identical neighbouring panels.
    # Keep both columns in frame to avoid a trivial margin rejection.
    wrong = [p+[125, 0] for p in prior[:8]]
    result = refine_panel_registration(image, wrong)
    assert result.image_consistent
    assert result.to_dict()['measurement_status'].startswith('withheld')
    truth = np.concatenate(prior[:8]) + [3, -2]
    check = check_registration_controls(result, np.concatenate(wrong), truth)
    assert not check['accepted_at_controls']
    assert check['p95_native_px'] > 100


def test_insufficient_modules_and_outside_frame_are_explicit():
    image, prior, _ = scene()
    assert not refine_panel_registration(image, prior[:3]).image_consistent
    assert not refine_panel_registration(image, [p-[150, 0] for p in prior]).image_consistent


@pytest.mark.parametrize('configuration', [
    {'max_displacement_px': True}, {'local_search_px': 0}, {'max_linear_change': float('nan')},
    {'max_check_p95_px': 0}, {'min_ambiguity_gap_px': -1},
])
def test_invalid_config(configuration):
    with pytest.raises(ValueError):
        PairRegistrationConfig(**configuration)


def test_invalid_inputs():
    image, prior, _ = scene()
    with pytest.raises(ValueError):
        refine_panel_registration(image.astype(float), prior)
    bad = prior[0].copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        refine_panel_registration(image, [bad]+prior[1:])
    with pytest.raises(ValueError):
        apply_correction([[1, 2]], np.ones((3, 3)))


def test_separate_controls_do_not_refit_a_wrong_mapping():
    image, prior, truth = scene()
    result = refine_panel_registration(image, prior)
    before = result.correction.copy()
    source, target = np.concatenate(prior), np.concatenate(truth)
    assert check_registration_controls(result, source, target)['accepted_at_controls']
    assert not check_registration_controls(result, source, target+[0, 5])['accepted_at_controls']
    np.testing.assert_array_equal(before, result.correction)
    assert not check_registration_controls(result, source[:3], target[:3])['accepted_at_controls']
    assert not check_registration_controls(result, [[0, 0]]*4, [[1, 1]]*4)['accepted_at_controls']
    with pytest.raises(ValueError):
        check_registration_controls(result, source, target, max_p95_px=float('nan'))
