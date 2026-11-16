from copy import deepcopy

import cv2
import numpy as np
import pytest

from pvdelta.native_repeat import native_motion_check, native_repeat_checks


def example():
    rng = np.random.default_rng(76)
    a = cv2.GaussianBlur(rng.integers(0, 256, (512, 640), dtype=np.uint8), (3, 3), .5)
    b = cv2.warpAffine(a, np.float32([[1, 0, 2], [0, 1, 3]]), (640, 512))
    def geometry(dx, dy):
        return {'proposals': [{'proposal_id': f'p{y}-{x}', 'source_method': 'closed_contour',
            'threshold_votes': 3, 'polygon_native_xy': [[dx+20+60*x, dy+20+40*y],
                [dx+70+60*x, dy+20+40*y], [dx+70+60*x, dy+50+40*y],
                [dx+20+60*x, dy+50+40*y]]} for y in range(4) for x in range(4)]}
    return a, b, geometry(0, 0), geometry(2, 3)


def test_native_motion_and_heldout_controls_pass_but_never_authorize_identity_or_measurement():
    a, b, ga, gb = example()
    originals = (a.copy(), b.copy(), deepcopy(ga), deepcopy(gb))
    report = native_motion_check(a, b, ga, gb)
    assert report['status'] == 'geometry_consistent_provisional'
    assert report['feature_check_count'] >= 6 and report['feature_check_p95_px'] < .1
    assert len(report['panel_controls']['matches']) == 16
    assert report['panel_controls']['heldout_p95_px'] < .001
    assert not any(report[k] for k in ('same_panel_verified', 'measurement_authorised', 'glare_clearance'))
    assert 'transform' not in report and 'polygon' not in report
    assert np.array_equal(a, originals[0]) and np.array_equal(b, originals[1])
    assert (ga, gb) == originals[2:]


def test_good_image_motion_cannot_override_bad_observed_panel_controls():
    a, b, ga, gb = example()
    gb['proposals'][5]['polygon_native_xy'] = (np.asarray(gb['proposals'][5]['polygon_native_xy'])+[8, 0]).tolist()
    report = native_motion_check(a, b, ga, gb)
    assert report['feature_check_p95_px'] < .1
    assert report['status'] == 'withheld' and not report['control_check_passed']
    assert report['reasons'] == ['held_out_panel_residual_too_large']
    assert len(report['panel_controls']['residuals_px']) == 16  # no difficult control trimmed


def test_return_track_rotation_is_compensated_on_display_only():
    a, _b, ga, _gb = example()
    original = a.copy()
    b = cv2.rotate(a, cv2.ROTATE_180)
    gb = deepcopy(ga)
    for proposal in gb['proposals']:
        quad = np.asarray(proposal['polygon_native_xy'])
        proposal['polygon_native_xy'] = (np.array([a.shape[1]-1, a.shape[0]-1])-quad).tolist()
    report = native_motion_check(a, b, ga, gb)
    assert report['status'] == 'geometry_consistent_provisional', report['reasons']
    assert report['feature_check_p95_px'] < .2
    assert len(report['panel_controls']['matches']) == 16
    assert report['same_panel_verified'] is False
    np.testing.assert_array_equal(a, original)
    np.testing.assert_array_equal(b, cv2.rotate(original, cv2.ROTATE_180))


def test_repeated_target_control_is_not_forced_into_a_selected_identity():
    a, b, ga, gb = example()
    gb['proposals'].append(dict(deepcopy(gb['proposals'][0]), proposal_id='duplicate'))
    report = native_motion_check(a, b, ga, gb)
    assert all(p['source_id'] != 'p0-0' for p in report['panel_controls']['matches'])
    assert report['same_panel_verified'] is False


def test_blank_or_unrelated_texture_does_not_publish_a_transform():
    a, b, ga, gb = example()
    for target in (np.zeros_like(b), np.random.default_rng(8).integers(0, 256, b.shape, dtype=np.uint8)):
        report = native_motion_check(a, target, ga, gb)
        assert report['status'] == 'withheld' and not report['control_check_passed']
        assert report['panel_controls'] is None and 'transform' not in report


@pytest.mark.parametrize('image', [np.zeros((64, 64, 3), np.uint8), np.zeros((64, 64), float)])
def test_invalid_native_coordinate_grid_is_rejected(image):
    with pytest.raises(ValueError, match='native-size'):
        native_motion_check(image, image, {}, {})


def test_bounded_runner_reuses_source_sdk_and_keeps_failed_target_visible(monkeypatch):
    from pvdelta import native_repeat

    a, b, ga, gb = example()
    rows = [{'capture_id': 'source', 'thermal_path': 'source_T.JPG', 'diagnostic': {'native_geometry': ga}},
        {'capture_id': 'target', 'thermal_path': 'target_T.JPG', 'time_gap_s': 1,
         'diagnostic': {'native_geometry': gb}},
        {'capture_id': 'missing', 'thermal_path': 'missing_T.JPG', 'time_gap_s': 2, 'diagnostic': {}}]
    monkeypatch.setattr(native_repeat, '_snapshot', lambda *args: (deepcopy(rows), 'frozen-binding', 'p0-0'))
    calls = []
    def decode(row, sdk):
        calls.append((row['capture_id'], sdk))
        return a if row['capture_id'] == 'source' else b
    monkeypatch.setattr(native_repeat, '_native_gray', decode)
    report = native_repeat_checks('database', 'finding', sdk_library='exact-sdk')
    assert calls == [('source', 'exact-sdk'), ('target', 'exact-sdk')]
    assert len(report['rows']) == 2
    assert report['rows'][0]['selected_control_target'] == 'p0-0'
    assert report['rows'][1]['status'] == 'unavailable' and report['rows'][1]['reasons']
    assert not any(report[k] for k in ('same_panel_verified', 'measurement_authorised', 'glare_clearance', 'database_written'))


def test_changed_sources_during_check_reject_the_entire_result(monkeypatch):
    from pvdelta import native_repeat

    a, _b, ga, _gb = example()
    rows = [{'capture_id': 'source', 'thermal_path': 'source_T.JPG', 'diagnostic': {'native_geometry': ga}}]
    bindings = iter(('before', 'after'))
    monkeypatch.setattr(native_repeat, '_snapshot', lambda *args: (deepcopy(rows), next(bindings), 'p0-0'))
    monkeypatch.setattr(native_repeat, '_native_gray', lambda *args: a)
    with pytest.raises(ValueError, match='changed during'):
        native_repeat_checks('database', 'finding')


def test_motion_input_is_palette_independent_monotonic_and_preserves_temperatures(monkeypatch):
    from pvdelta import native_repeat

    values = np.linspace(15, 55, 64*80, dtype=np.float32).reshape(64, 80)
    original = values.copy()
    palette_calls = []

    class Reader:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def processed_rgb(self):
            palette_calls.append(True)
            # Two different display palettes for exactly the same temperatures.
            return np.full((*values.shape, 3), 30 if len(palette_calls) == 1 else 210, np.uint8)

    profile = {'profile_id': 'synthetic-profile'}
    calls = []
    def restore(reader, path, saved):
        calls.append((path, saved))
        return values
    monkeypatch.setattr('pvdelta.decode.RJpeg', Reader)
    monkeypatch.setattr('pvdelta.measurement_profile.apply_saved_measurement_profile', restore)
    row = {'thermal_path': 'original_T.JPG', 'profile': profile}
    a = native_repeat._native_gray(row, 'test-sdk')
    b = native_repeat._native_gray(row, 'test-sdk')
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(values, original)
    assert a.shape == values.shape and a.dtype == np.uint8
    assert a.min() == 0 and a.max() == 255
    assert (np.diff(a.ravel().astype(int)) >= 0).all()
    assert not palette_calls
    assert calls == [('original_T.JPG', profile)]*2


def test_flat_native_values_cannot_borrow_fake_palette_texture(monkeypatch):
    from pvdelta import native_repeat

    class Reader:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def processed_rgb(self):
            return np.random.default_rng(0).integers(0, 256, (64, 80, 3), np.uint8)

    monkeypatch.setattr('pvdelta.decode.RJpeg', Reader)
    monkeypatch.setattr('pvdelta.measurement_profile.apply_saved_measurement_profile',
                        lambda *args: np.full((64, 80), 25., np.float32))
    with pytest.raises(ValueError, match='dynamic range'):
        native_repeat._native_gray({'thermal_path': 'flat_T.JPG', 'profile': {}}, 'test-sdk')
