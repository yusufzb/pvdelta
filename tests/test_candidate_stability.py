import hashlib
import json

import numpy as np
import pytest

from pvdelta.candidate_stability import candidate_stability
from pvdelta.config import DetectionConfig


def scene():
    yy, xx = np.indices((40, 40))
    values = 25 + .03*xx + .02*yy
    measurement = np.zeros(values.shape, bool)
    measurement[2:38, 2:38] = True
    reference = measurement.copy()
    points = np.argwhere((yy >= 18) & (yy < 22) & (xx >= 18) & (xx < 22))
    reference[points[:, 0], points[:, 1]] = False
    return values, measurement, reference, points


def test_linear_background_with_hotspot_is_consistent_without_changing_inputs():
    values, mask, ref, points = scene()
    values[points[:, 0], points[:, 1]] += 8
    before = [hashlib.sha256(v.tobytes()).hexdigest() for v in (values, mask, ref)]
    result = candidate_stability(values, mask, ref, {'hot': points}, config=DetectionConfig())
    region = result['regions']['hot']
    assert region['reference_test'] == 'above_threshold_in_all_folds'
    assert region['delta_min_k'] == pytest.approx(8)
    assert region['delta_max_k'] == pytest.approx(8)
    assert region['extra_margin_retained_fraction'] == {'1': 1., '2': 1.}
    assert sum(f['heldout_pixels'] for f in result['folds']) == ref.sum()
    assert all(f['heldout_pixels']+f['training_pixels'] == ref.sum() for f in result['folds'])
    assert all(f['heldout_p95_abs_k'] < 1e-10 for f in result['folds'])
    assert before == [hashlib.sha256(v.tobytes()).hexdigest() for v in (values, mask, ref)]
    json.dumps(result, allow_nan=False)


def test_inconsistent_reference_can_cross_existing_threshold():
    values, mask, ref, points = scene()
    values[points[:, 0], points[:, 1]] += 2.6
    yy, xx = np.indices(values.shape)
    values[ref & (yy < 20) & (xx < 20)] += 1.2
    result = candidate_stability(values, mask, ref, {'hot': points}, config=DetectionConfig())
    r = result['regions']['hot']
    assert r['reference_test'] == 'crosses_threshold'
    assert r['delta_min_k'] < 2.5 <= r['delta_max_k']
    assert not r['fault_confirmed']


def test_edge_hotspot_is_retained_as_candidate_when_margin_stress_removes_support():
    values, mask, ref, _ = scene()
    points = np.array([[2, 15], [2, 16], [3, 15], [3, 16]])
    ref[points[:, 0], points[:, 1]] = False
    values[points[:, 0], points[:, 1]] += 10
    r = candidate_stability(values, mask, ref, {'edge': points}, config=DetectionConfig())['regions']['edge']
    assert r['reference_test'] == 'above_threshold_in_all_folds'
    assert r['extra_margin_retained_fraction'] == {'1': .5, '2': 0.}
    assert not r['fault_confirmed']


def test_missing_blocks_cannot_be_called_consistent():
    values, mask, ref, points = scene()
    ref[20:] = False
    points = np.array([[25, 25], [25, 26], [26, 25], [26, 26]])
    result = candidate_stability(values, mask, ref, {'hot': points}, config=DetectionConfig())
    assert result['regions']['hot']['reference_test'] == 'insufficient_reference_blocks'
    assert result['regions']['hot']['valid_folds'] == 2
    assert sum(f['status'] == 'unavailable' for f in result['folds']) == 2


def test_candidate_never_enters_reference_fit_and_median_mode_is_explicit():
    values, mask, ref, points = scene()
    config = DetectionConfig(detrend_gradient=False)
    first = candidate_stability(values, mask, ref, {'hot': points}, config=config)
    values[points[:, 0], points[:, 1]] += 15
    second = candidate_stability(values, mask, ref, {'hot': points}, config=config)
    assert first['folds'] == second['folds']
    assert second['regions']['hot']['delta_min_k']-first['regions']['hot']['delta_min_k'] == pytest.approx(15)


@pytest.mark.parametrize('bad', ['reference_contaminated', 'out_of_bounds', 'float_pixels', 'outside_mask', 'nan'])
def test_bad_support_is_not_silently_scored(bad):
    values, mask, ref, points = scene()
    if bad == 'reference_contaminated':
        ref[points[0, 0], points[0, 1]] = True
    elif bad == 'out_of_bounds':
        points[0] = [-1, 0]
    elif bad == 'float_pixels':
        points = points.astype(float)
    elif bad == 'outside_mask':
        points[0] = [0, 0]
    else:
        values[points[0, 0], points[0, 1]] = np.nan
    with pytest.raises(ValueError):
        candidate_stability(values, mask, ref, {'hot': points}, config=DetectionConfig())
