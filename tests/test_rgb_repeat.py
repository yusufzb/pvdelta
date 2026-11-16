from copy import deepcopy

import pytest

from pvdelta import repeat_views, rgb_repeat


def inputs():
    quad = [[100, 100], [180, 100], [180, 220], [100, 220]]
    source = {'capture_id': 'source', 'rgb_path': 'source_V.JPG', 'thermal_path': 'source_T.JPG',
        'rgb_sha256': 'rgb-source', 'thermal_sha256': 'thermal-source', 'gimbal_pitch_deg': -90,
        'module_id': 'M004', 'module_polygon': quad,
        'diagnostic': {'modules': [{'module': 'M004', 'native_polygon': quad}]}}
    targets = [{'capture_id': name, 'rgb_path': name+'_V.JPG', 'thermal_path': name+'_T.JPG',
        'camera_model': 'M4T', 'rgb_size': [4032, 3024], 'rgb_sha256': 'rgb-'+name,
        'actual_rgb_sha256': 'rgb-'+name if name == 'good' else 'changed',
        'thermal_sha256': 'thermal-'+name, 'actual_thermal_sha256': 'thermal-'+name,
        'gimbal_pitch_deg': -90, 'time_gap_s': 40, 'camera_distance_m': 2.5} for name in ('good', 'bad')]
    registration = {'model': {'centre': [0, 0], 'scale': [1, 1],
        'coefficients': [[0, 0], [1, 0], [0, 1], [0, 0], [0, 0], [0, 0]]}}
    return [source, *targets], quad, registration


def setup(monkeypatch, *, binding_changes=False):
    rows, quad, registration = inputs()
    bindings = iter(('first', 'changed'))
    monkeypatch.setattr(rgb_repeat, '_snapshot',
        lambda *args: (deepcopy(rows), next(bindings) if binding_changes else 'frozen'))
    verified, extracted, matches = [], [], []
    monkeypatch.setattr(rgb_repeat, '_verify_source', lambda row, reg, sdk: verified.append(sdk))
    def observed(path, sha, pitch):
        extracted.append(path)
        return {'modules': [{'polygon_rgb_xy': deepcopy(quad), 'table_index': 0}]}
    monkeypatch.setattr(rgb_repeat, '_observed', observed)
    def match(*args):
        matches.append(args)
        return repeat_views.PanelMatch('matched_provisional', 'fixed checks passed', target_index=0,
            iou=.99, check_p95_px=.2, check_count=8, fit_inliers=24, target_corner_indices=(0, 1, 2, 3))
    monkeypatch.setattr(repeat_views, 'match_panel', match)
    return rows, quad, registration, verified, extracted, matches


def test_observed_rgb_matches_keep_cross_sensor_component_and_glare_unresolved(monkeypatch):
    rows, quad, registration, verified, extracted, matches = setup(monkeypatch)
    original = deepcopy(rows)
    report = rgb_repeat.observed_rgb_repeat('private.duckdb', 'finding', registration, sdk_library='exact-sdk')
    assert verified == ['exact-sdk'] and len(matches) == 2
    assert extracted == ['source_V.JPG', 'good_V.JPG']
    assert report['source_anchor_status'] == 'approximate_native_association'
    assert report['source_rgb_polygon'] == quad
    good, bad = report['rows']
    assert good['status'] == 'matched_provisional' and good['rgb_polygon'] == quad
    assert good['reciprocal_check_passed'] and good['reverse_result']['target_index'] == 0
    assert bad['status'] == 'unavailable' and bad['rgb_polygon'] is None
    assert 'indexed identity' in bad['reason']
    flags = ('panel_identity_verified', 'cross_sensor_verified', 'thermal_component_correspondence_verified',
             'measurement_authorised', 'glare_clearance')
    assert all(not r[k] for r in [report, *report['rows']] for k in flags)
    assert all(r['component_evidence'] == 'unknown' for r in [report, *report['rows']])
    assert report['database_written'] is False and 'delta_t_k' not in good
    assert 'native_polygon' not in good and 'temperature' not in good
    assert rows == original


def test_no_unique_source_anchor_is_not_replaced_by_another_panel(monkeypatch):
    _rows, _quad, registration, _v, _e, matches = setup(monkeypatch)
    monkeypatch.setattr(rgb_repeat, 'match_module_quads', lambda *args: [])
    with pytest.raises(ValueError, match='no unique bounded'):
        rgb_repeat.observed_rgb_repeat('db', 'finding', registration)
    assert not matches


def test_changed_binding_during_rgb_check_withholds_result(monkeypatch):
    _rows, _quad, registration, *_ = setup(monkeypatch, binding_changes=True)
    with pytest.raises(ValueError, match='changed during'):
        rgb_repeat.observed_rgb_repeat('db', 'finding', registration)


def test_absent_or_ambiguous_target_has_no_overlay(monkeypatch):
    _rows, _quad, registration, *_ = setup(monkeypatch)
    monkeypatch.setattr(repeat_views, 'match_panel', lambda *args:
        repeat_views.PanelMatch('unavailable', 'target panel absent or ambiguous'))
    report = rgb_repeat.observed_rgb_repeat('db', 'finding', registration)
    assert all(r['rgb_polygon'] is None for r in report['rows'])


def test_forward_match_to_neighbour_cannot_pass_without_reverse_panel_identity(monkeypatch):
    _rows, quad, registration, *_ = setup(monkeypatch)
    observed = rgb_repeat._observed
    neighbour = [[x + 100, y] for x, y in quad]
    monkeypatch.setattr(rgb_repeat, '_observed', lambda path, sha, pitch:
        {'modules': [{'polygon_rgb_xy': quad}, {'polygon_rgb_xy': neighbour}]}
        if path == 'source_V.JPG' else observed(path, sha, pitch))
    calls = []
    def match(source, target, polygon, targets):
        calls.append((str(source), str(target), len(targets)))
        return repeat_views.PanelMatch('matched_provisional', 'local fit passed',
            target_index=0 if str(source) == 'source_V.JPG' else 1,
            iou=.95, check_p95_px=.1, check_count=12, fit_inliers=30)
    monkeypatch.setattr(repeat_views, 'match_panel', match)
    report = rgb_repeat.observed_rgb_repeat('db', 'finding', registration)
    good = report['rows'][0]
    assert good['status'] == 'unavailable' and good['rgb_polygon'] is None
    assert not good['reciprocal_check_passed']
    assert good['forward_result']['status'] == 'matched_provisional'
    assert good['reverse_result']['target_index'] == 1
    assert 'different source panel' in good['reason']
    assert calls == [('source_V.JPG', 'good_V.JPG', 1), ('good_V.JPG', 'source_V.JPG', 2)]


def test_failed_reverse_tracking_keeps_the_forward_diagnostic_but_no_mark(monkeypatch):
    _rows, _quad, registration, *_ = setup(monkeypatch)
    forward = repeat_views.PanelMatch('matched_provisional', 'forward passed', target_index=0,
        iou=.99, check_p95_px=.2, check_count=8, fit_inliers=24)
    reverse = repeat_views.PanelMatch('unavailable', 'panel falls outside supported feature area')
    outcomes = iter((forward, reverse))
    monkeypatch.setattr(repeat_views, 'match_panel', lambda *args: next(outcomes))
    report = rgb_repeat.observed_rgb_repeat('db', 'finding', registration)
    good = report['rows'][0]
    assert good['status'] == 'unavailable' and good['rgb_polygon'] is None
    assert not good['reciprocal_check_passed']
    assert good['forward_result']['check_p95_px'] == .2
    assert good['reverse_result']['reason'] == reverse.reason
    assert 'Reverse RGB check failed' in good['reason']


def test_inconsistent_return_corner_order_does_not_transfer_a_panel_outline(monkeypatch):
    _rows, _quad, registration, *_ = setup(monkeypatch)
    outcomes = iter((
        repeat_views.PanelMatch('matched_provisional', 'forward', target_index=0,
            target_corner_indices=(0, 1, 2, 3)),
        repeat_views.PanelMatch('matched_provisional', 'reverse', target_index=0,
            target_corner_indices=(1, 2, 3, 0))))
    monkeypatch.setattr(repeat_views, 'match_panel', lambda *args: next(outcomes))
    report = rgb_repeat.observed_rgb_repeat('db', 'finding', registration)
    good = report['rows'][0]
    assert good['status'] == 'unavailable' and good['rgb_polygon'] is None
    assert not good['reciprocal_check_passed']
    assert 'corner cycle' in good['reason']


@pytest.mark.parametrize('order', ((1, 2, 3, 0), (3, 2, 1, 0)))
def test_source_vertex_origin_and_winding_are_not_different_panel_identities(monkeypatch, order):
    _rows, quad, registration, _v, _e, matches = setup(monkeypatch)
    observed = rgb_repeat._observed
    monkeypatch.setattr(rgb_repeat, '_observed', lambda path, sha, pitch:
        {'modules': [{'polygon_rgb_xy': [quad[i] for i in order]}]}
        if path == 'source_V.JPG' else observed(path, sha, pitch))
    report = rgb_repeat.observed_rgb_repeat('db', 'finding', registration)
    assert report['rows'][0]['reciprocal_check_passed']
    assert matches[1][3][0].tolist() == quad


def image_report(tmp_path):
    import hashlib

    from PIL import Image

    pairs = []
    for name in ('source', 'target', 'unmatched'):
        row = {'capture_id': name, 'status': 'matched_provisional' if name == 'target' else 'unavailable',
            'reason': 'provisional test', 'rgb_polygon': [[100, 100], [180, 100], [180, 220], [100, 220]]}
        for kind in ('thermal', 'rgb'):
            path = tmp_path/f'{name}-{kind}.png'
            Image.new('RGB', (300, 300), '#404040').save(path)
            row[kind+'_path'] = str(path)
            row[kind+'_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        pairs.append(row)
    pairs[2]['rgb_polygon'] = None
    report = {'schema': 'pvdelta.observed-rgb-repeat/v1', 'binding': 'f1-one', 'source': pairs[0],
        'source_rgb_polygon': pairs[0]['rgb_polygon'], 'rows': pairs[1:]}
    report['content_sha256'] = rgb_repeat.digest(report)
    return report


def test_preview_marks_only_observed_rgb_never_thermal_or_unmatched_photos(tmp_path):
    import hashlib
    from pathlib import Path

    import numpy as np

    report = image_report(tmp_path)
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob('*.png')}
    thermal, rgb = rgb_repeat.repeat_pair_images(report, 'target')
    assert rgb.getpixel((100, 100)) == (0, 255, 255)
    assert np.asarray(thermal).min() == np.asarray(thermal).max() == 64
    unmarked, unmatched = rgb_repeat.repeat_pair_images(report, 'unmatched')
    assert np.asarray(unmatched).min() == np.asarray(unmatched).max() == 64
    assert np.array_equal(unmarked, thermal)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in before.items())


def test_modified_report_or_changed_photo_cannot_reuse_a_rgb_outline(tmp_path):
    from pathlib import Path

    report = image_report(tmp_path)
    changed = deepcopy(report)
    changed['rows'][0]['rgb_polygon'][0][0] += 10
    with pytest.raises(ValueError, match='report content changed'):
        rgb_repeat.repeat_pair_images(changed, 'target')
    path = Path(report['rows'][0]['rgb_path'])
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='source changed'):
        rgb_repeat.repeat_pair_images(report, 'target')


def test_source_scope_requires_exact_flight_calibration_and_forwards_sdk_profile(tmp_path, monkeypatch):
    import hashlib
    import json

    import numpy as np

    from pvdelta import decode, measurement_profile

    _rows, _quad, registration = inputs()
    calibration = tmp_path/'calibration.json'
    (tmp_path/'anchor_T.JPG').write_bytes(b'local anchor')
    payload = {'quadratic_diagnostic': {'accepted': True, 'forward': registration['model']},
        'cross_frame_visual_audit': {'calibration_frame': 'anchor_T.JPG'}}
    calibration.write_text(json.dumps(payload), encoding='utf-8')
    scope = {'schema': 'pvdelta.rgb-search-scope/v1', 'source_root': str(tmp_path),
        'calibration_sha256': hashlib.sha256(calibration.read_bytes()).hexdigest(),
        'calibration_anchor_sha256': hashlib.sha256((tmp_path/'anchor_T.JPG').read_bytes()).hexdigest()}
    (tmp_path/'rgb-search-scope.json').write_text(json.dumps(scope), encoding='utf-8')
    registration.update(path=str(calibration), calibration_frame='anchor_T.JPG')
    source = {'camera_model': 'M4T', 'native_size': [640, 512], 'rgb_size': [4032, 3024],
        'actual_rgb_sha256': 'r', 'rgb_sha256': 'r', 'actual_thermal_sha256': 't', 'thermal_sha256': 't',
        'source_root': str(tmp_path), 'thermal_path': 'source_T.JPG', 'profile': {'saved': 'profile'}}
    calls = []
    class Reader:
        def __init__(self, path, sdk): calls.append((path, sdk))
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(decode, 'RJpeg', Reader)
    def apply(reader, path, profile):
        calls.append(profile)
        return np.zeros((512, 640), np.float32)
    monkeypatch.setattr(measurement_profile, 'apply_saved_measurement_profile', apply)
    rgb_repeat._verify_source(source, registration, 'exact-sdk')
    assert calls == [('source_T.JPG', 'exact-sdk'), {'saved': 'profile'}]
    source['camera_model'] = 'another-camera'
    with pytest.raises(ValueError, match='original M4T'):
        rgb_repeat._verify_source(source, registration, 'exact-sdk')
    source['camera_model'] = 'M4T'
    other = tmp_path/'another-flight'
    other.mkdir()
    (other/'anchor_T.JPG').write_bytes(b'local anchor')
    source['source_root'] = str(other)
    with pytest.raises(ValueError, match='cross-flight reuse withheld'):
        rgb_repeat._verify_source(source, registration, 'exact-sdk')
    source['source_root'] = str(tmp_path)
    registration['model'] = {'changed': True}
    with pytest.raises(ValueError, match='not bound'):
        rgb_repeat._verify_source(source, registration, 'exact-sdk')


def test_ui_is_explicit_navigates_and_clears_evidence_when_region_changes(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from pvdelta import rgb_repeat_ui

    report = image_report(tmp_path)
    calls = []
    def compute(database, finding, registration, **kwargs):
        calls.append((finding, kwargs['sdk_library']))
        result = deepcopy(report)
        result['binding'] = finding+'-one'
        result['content_sha256'] = rgb_repeat.digest({k: v for k, v in result.items() if k != 'content_sha256'})
        return result
    monkeypatch.setattr(rgb_repeat_ui, 'observed_rgb_repeat', compute)
    monkeypatch.setattr(rgb_repeat_ui, 'rgb_repeat_signature', lambda db, finding, reg, **kw: finding+'-one')
    script = ('import streamlit as st\nfrom pvdelta.rgb_repeat_ui import observed_rgb_review\n'
        'fid=st.selectbox("Finding", ["f1","f2"],key="finding")\n'
        'observed_rgb_review("db",fid,{"model":"prior"},"exact-sdk",key="test")\n')
    app = AppTest.from_string(script, default_timeout=20).run()
    assert not calls and not app.get('image')
    app.button(key='test-rgb-find').click().run()
    assert not app.exception and not app.error and len(app.get('image')) == 2
    assert calls == [('f1', 'exact-sdk')]
    app.button(key='test-rgb-frame-f1-next').click().run()
    assert not app.exception and app.selectbox(key='test-rgb-frame-f1').value == 'target'
    assert calls == [('f1', 'exact-sdk')]
    from pathlib import Path

    bad = Path(report['rows'][1]['rgb_path'])
    bad.write_bytes(bad.read_bytes()+b'changed')
    app.selectbox(key='test-rgb-frame-f1').select('unmatched').run()
    assert not app.exception and not app.get('image')
    assert any('Selected comparison photos unavailable' in w.value for w in app.warning)
    assert 'test-rgb-result' in app.session_state
    app.selectbox(key='test-rgb-frame-f1').select('target').run()
    assert not app.exception and len(app.get('image')) == 2
    assert calls == [('f1', 'exact-sdk')]
    app.selectbox(key='finding').select('f2').run()
    assert not app.exception and not app.get('image')
    assert 'test-rgb-result' not in app.session_state
    app.button(key='test-rgb-find').click().run()
    assert not app.exception and app.selectbox(key='test-rgb-frame-f2').value == 'source'
    assert calls[-1] == ('f2', 'exact-sdk')
