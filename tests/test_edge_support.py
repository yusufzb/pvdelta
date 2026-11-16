import numpy as np
import pytest

pytest.importorskip('cv2')
from pvdelta.edge_support import combine_edge_evidence, native_edge_continuity


def test_weak_continuous_frame_is_observed_and_native_values_unchanged():
    frame = np.full((140, 100), 25.)
    frame[:, 49:52] = 24.7
    original = frame.copy()
    result = native_edge_continuity(frame, [50, 10], [50, 130])
    assert result.coverage == 1 and result.largest_gap_fraction == 0
    assert result.polarity == 'cold'
    assert not result.measurement_authorised
    np.testing.assert_array_equal(original, frame)


def test_missing_middle_is_not_filled_from_endpoints():
    frame = np.full((140, 100), 25.)
    frame[:, 49:52] = 24.7
    frame[50:90] = 25.
    result = native_edge_continuity(frame, [50, 10], [50, 130])
    assert .6 < result.coverage < .7
    assert result.largest_gap_fraction == pytest.approx(40/121)


def test_one_sided_step_is_not_a_frame_ridge():
    frame = np.full((140, 100), 25.)
    frame[:, 50:] = 35.
    assert native_edge_continuity(frame, [50, 10], [50, 130]).coverage == 0


def test_strong_adjacent_ridge_cannot_supply_missing_line():
    frame = np.full((140, 100), 25.)
    frame[:, 57:60] = 15.
    assert native_edge_continuity(frame, [50, 10], [50, 130]).coverage == 0


def test_unobserved_border_is_missing_not_padded_support():
    frame = np.full((140, 100), 25.)
    result = native_edge_continuity(frame, [0, 10], [0, 130])
    assert result.valid_samples == 0 and result.coverage == 0
    assert result.largest_gap_fraction == 1


def test_noise_alone_does_not_create_continuous_edge():
    frame = np.random.default_rng(17).normal(25, .3, (140, 100))
    result = native_edge_continuity(frame, [50, 10], [50, 130])
    assert result.required_contrast_k >= 3*result.noise_k
    assert result.coverage < .2


@pytest.mark.parametrize('kwargs', [{'min_contrast_k':0}, {'min_snr':float('nan')}, {'min_snr':True}])
def test_invalid_limits_fail(kwargs):
    with pytest.raises(ValueError):
        native_edge_continuity(np.ones((50, 50)), [10, 10], [10, 30], **kwargs)


@pytest.mark.parametrize('endpoints', [([10,10],[10,10]), ([10,10],[float('nan'),30])])
def test_invalid_endpoints_fail(endpoints):
    with pytest.raises(ValueError):
        native_edge_continuity(np.ones((50,50)), *endpoints)


def test_complementary_observations_join_without_filling_unobserved_gaps():
    frame = np.full((140, 100), 25.)
    frame[10:50, 49:52] = 24.7
    ridge = native_edge_continuity(frame, [50, 10], [50, 130])
    result = combine_edge_evidence(ridge, [[.5, .7]])
    assert .5 < result['coverage'] < .6
    assert result['largest_gap_fraction'] > .25
    assert result['line_only_samples'] > 0 and result['ridge_only_samples'] > 0
    assert not result['measurement_authorised']
    assert combine_edge_evidence(ridge, [])['coverage'] == ridge.coverage


@pytest.mark.parametrize('intervals', [[[-.1,.5]], [[.7,.2]], [[0,float('nan')]]])
def test_invalid_observed_intervals_fail(intervals):
    ridge = native_edge_continuity(np.ones((50,50)), [20,10], [20,40])
    with pytest.raises(ValueError):
        combine_edge_evidence(ridge, intervals)
