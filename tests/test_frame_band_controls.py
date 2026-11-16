from copy import deepcopy

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from pvdelta.frame_band_controls import build_frame_band_controls, observe_frame_band_corner


@pytest.mark.parametrize('polarity', [1, -1])
def test_band_midpoints_in_each_polarity_leave_original_pixels_unchanged(polarity):
    image = np.full((220, 240), 30, np.uint8)
    cv2.line(image, (60, 20), (60, 200), 220, 5)
    cv2.line(image, (20, 70), (220, 70), 220, 5)
    if polarity == -1: image = 255-image
    original = image.copy()
    r = observe_frame_band_corner(image, [62, 68], [[0, 100], [100, 0]], [100, 100], radius=8)
    assert r['status'] == 'observed_paired_band_corner', r
    np.testing.assert_allclose(r['observed_xy'], [60, 70], atol=.35)
    assert not r['single_edge_fallback'] and r['source_only']
    assert all(min(s['spatial_strip_support']) >= .6 for s in r['sides'])
    np.testing.assert_array_equal(image, original)


def test_two_steps_are_not_accepted_as_two_band_centres():
    image = np.full((220, 240), 30, np.uint8)
    image[:, 60:] = 110
    image[70:, :] += 80
    r = observe_frame_band_corner(image, [62, 68], [[0, 100], [100, 0]], [100, 100], radius=8)
    assert r['observed_xy'] is None and r['reason'] == 'paired_edges_unresolved'


@pytest.mark.parametrize('mode', ['flat', 'missing_arm', 'outside'])
def test_missing_or_unsupported_band_remains_unknown(mode):
    image = np.full((220, 240), 30, np.uint8)
    seed = [62, 68]
    if mode != 'flat': cv2.line(image, (60, 20), (60, 200), 220, 5)
    if mode == 'outside': seed = [232, 210]
    r = observe_frame_band_corner(image, seed, [[0, 100], [100, 0]], [100, 100], radius=8)
    assert r['observed_xy'] is None and not r['measurement_authorised']


def test_one_longitudinal_support_gap_cannot_be_hidden_by_global_line_fit():
    image = np.full((220, 240), 30, np.uint8)
    cv2.line(image, (60, 20), (60, 200), 220, 5)
    cv2.line(image, (20, 70), (220, 70), 220, 5)
    image[110:135, 53:68] = 30
    r = observe_frame_band_corner(image, [62, 68], [[0, 100], [100, 0]], [100, 100], radius=8)
    assert r['observed_xy'] is None


def test_equal_parallel_bands_cannot_be_chosen_by_first_peak():
    image = np.full((220, 240), 30, np.uint8)
    for x in (60, 76): cv2.line(image, (x, 20), (x, 200), 220, 3)
    cv2.line(image, (20, 70), (220, 70), 220, 3)
    r = observe_frame_band_corner(image, [68, 68], [[0, 100], [100, 0]], [100, 100], radius=12)
    assert r['observed_xy'] is None and r['reason'] == 'competing_parallel_bands'


@pytest.fixture
def controls(monkeypatch):
    import pvdelta.frame_band_controls as module

    native, rgb = np.zeros((400, 400), np.uint8), np.zeros((1200, 1200), np.uint8)
    matches = []
    for i, (x, y) in enumerate([(30, 30), (150, 30), (30, 200), (150, 200)]):
        q = np.array([[x, y], [x+60, y], [x+60, y+100], [x, y+100]], float)
        matches.append({'module_id': f'M{i}', 'native_polygon': q.tolist(), 'rgb_polygon': (q*2+100).tolist()})
    split = {'fit_module_ids': ['M0', 'M1'], 'check_module_ids': ['M2', 'M3']}
    calls = []
    def observed(gradients, seed, directions, lengths, **kwargs):
        calls.append((gradients[0].shape, np.asarray(seed).copy()))
        return {'observed_xy': np.asarray(seed).tolist(), 'reason': None}
    monkeypatch.setattr(module, '_observe_corner', observed)
    return native, rgb, matches, split, calls


def test_frozen_roles_and_every_raw_check_are_retained(controls):
    native, rgb, matches, split, calls = controls
    original = deepcopy((matches, split))
    result = build_frame_band_controls(native, rgb, matches, split)
    assert result['status'] == 'source_observed_controls_draft'
    assert len(result['observations']) == 16 and len(calls) == 32
    assert result['fit_point_count'] == result['distinct_check_point_count'] == 8
    assert len(result['all_observed_check_points']) == 8
    assert result['split'] == split and (matches, split) == original
    assert not result['physical_identity_verified']


def test_missing_check_is_mandatory_even_with_plenty_of_fit_points(controls, monkeypatch):
    import pvdelta.frame_band_controls as module

    native, rgb, matches, split, _ = controls
    original = module._observe_corner
    def missing(gradients, seed, *args, **kwargs):
        if gradients[0].shape == native.shape and list(seed) == matches[2]['native_polygon'][0]:
            return {'observed_xy': None, 'reason': 'paired_edges_unresolved'}
        return original(gradients, seed, *args, **kwargs)
    monkeypatch.setattr(module, '_observe_corner', missing)
    result = build_frame_band_controls(native, rgb, matches, split)
    assert result['status'] == 'unresolved'
    assert result['missing_check_control_ids'] == ['M2-0']
    assert len(result['observations']) == 16


def test_shared_fit_corner_cannot_leak_into_checking(controls):
    native, rgb, matches, split, _ = controls
    matches[0]['native_polygon'] = deepcopy(matches[2]['native_polygon'])
    matches[0]['rgb_polygon'] = deepcopy(matches[2]['rgb_polygon'])
    result = build_frame_band_controls(native, rgb, matches, split)
    assert result['fit_point_count'] == 4
    assert {p['control_id'] for p in result['points']}.isdisjoint({'M0-0', 'M0-1', 'M0-2', 'M0-3'})
    assert len(result['all_observed_check_points']) == 8
    assert result['status'] == 'unresolved'


def test_one_module_cannot_be_both_fit_and_check(controls):
    native, rgb, matches, split, _ = controls
    split['fit_module_ids'][0] = 'M2'
    with pytest.raises(ValueError, match='Frozen disjoint'):
        build_frame_band_controls(native, rgb, matches, split)
