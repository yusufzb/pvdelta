import hashlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pvdelta.models import ThermalFrame
from pvdelta.rgb_pair_check import digest, fit_module_check, match_module_quads, pair_check_binding


@pytest.fixture
def scene(tmp_path):
    thermal, rgb = tmp_path/'thermal.JPG', tmp_path/'rgb.JPG'
    thermal.write_bytes(b'synthetic source, not decoded by the corner fitter')
    Image.new('RGB', (4032, 3024)).save(rgb)
    values = np.full((512, 640), 30., np.float32)
    values.setflags(write=False)
    frame = ThermalFrame('test', str(thermal), hashlib.sha256(thermal.read_bytes()).hexdigest(),
                         values, None, 640, 512)
    model = {'centre': [0, 0], 'scale': [1, 1],
        'coefficients': [[1000, 700], [3, 0], [0, 3], [0, 0], [0, 0], [0, 0]]}
    modules, observed = [], []
    for col in range(3):
        for row in range(4):
            x, y = 90+col*120, 50+row*95
            q = np.array([[x, y], [x+105, y], [x+105, y+80], [x, y+80]], float)
            modules.append({'module': f'M{col}{row}', 'native_polygon': q.tolist()})
            actual = q*3+[1009, 694]
            observed.append({'polygon_rgb_xy': np.roll(actual, 2, axis=0).tolist(), 'table_index': 0})
    return frame, rgb, model, modules, observed


def test_known_shift_and_unused_target_are_recovered_without_temperature_changes(scene):
    frame, rgb, model, modules, observed = scene
    before = deepcopy((modules, observed))
    matches = match_module_quads(modules, observed, model)
    assert len(matches) == len(modules)
    result = fit_module_check(frame, rgb, matches, 'M11')
    assert result['status'] == 'image_consistent_provisional', result['reasons']
    assert result['check_p95_native_px'] < 1e-6
    assert result['prior_check_p95_native_px'] > 3
    assert 'M11' in result['check_module_ids'] and 'M11' not in result['fit_module_ids']
    assert set(result['check_module_ids']).isdisjoint(result['fit_module_ids'])
    assert not result['correspondence_verified'] and not result['measurement_authorised']
    assert result['maximum_correction_native_px'] < 4
    assert (modules, observed) == before
    np.testing.assert_array_equal(frame.temperature_c, np.full((512, 640), 30., np.float32))


def test_corrupt_unused_target_is_not_optimised_away(scene):
    frame, rgb, model, modules, observed = scene
    matches = match_module_quads(modules, observed, model)
    target = next(m for m in matches if m['module_id'] == 'M11')
    target['rgb_polygon'] = (np.array(target['rgb_polygon'])+[18, 0]).tolist()
    result = fit_module_check(frame, rgb, matches, 'M11')
    assert result['status'] == 'unresolved'
    assert result['check_p95_native_px'] > 2
    assert not result['registration']['geometry_gate_passed']


def test_competing_source_or_duplicate_target_is_not_forced(scene):
    _, _, model, modules, observed = scene
    modules.append({'module': 'duplicate', 'native_polygon': modules[0]['native_polygon']})
    result = match_module_quads(modules, observed, model)
    assert not {'M00', 'duplicate'} & {m['module_id'] for m in result}
    observed.append(deepcopy(observed[1]))
    result = match_module_quads(modules[:-1], observed, model)
    assert 'M01' not in {m['module_id'] for m in result}


def test_wrong_table_and_missing_target_stay_unresolved(scene):
    frame, rgb, model, modules, observed = scene
    observed[5]['table_index'] = None
    matches = match_module_quads(modules, observed, model)
    assert fit_module_check(frame, rgb, matches, 'M11')['status'] == 'unresolved'
    assert fit_module_check(frame, rgb, matches, 'missing')['status'] == 'unresolved'


def test_large_initial_displacement_cannot_borrow_neighbour_identity(scene):
    _, _, model, modules, observed = scene
    for item in observed:
        item['polygon_rgb_xy'] = (np.array(item['polygon_rgb_xy'])+[60, 0]).tolist()
    assert not match_module_quads(modules, observed, model)


def test_shared_corners_are_not_both_fit_and_check_controls(scene):
    frame, rgb, model, modules, observed = scene
    for source, target in zip(modules, observed):
        q = np.array(source['native_polygon'])
        q[1:3, 0] += 15
        q[2:, 1] += 15
        source['native_polygon'] = q.tolist()
        target['polygon_rgb_xy'] = (q*3+[1009, 694]).tolist()
    matches = match_module_quads(modules, observed, model)
    result = fit_module_check(frame, rgb, matches, 'M11')
    assert 'all_raw_check_corners_still_scored' in result['shared_corner_policy']
    assert result['fit_controls_after_shared_corner_exclusion'] < 4*len(result['fit_module_ids'])
    if result['status'] == 'image_consistent_provisional':
        assert result['check_p95_native_px'] < 1e-6
    else:
        assert any('distinct controls' in r for r in result['reasons'])


@pytest.fixture
def checked_case(scene):
    from pvdelta.measurement_profile import frame_measurement_profile

    frame, rgb, model, modules, observed = scene
    profile = frame_measurement_profile(frame)
    prior = {'model': model, 'calibration_frame': Path(frame.source_path).name}
    diagnostic = {'geometry_method': 'native_observed_boundaries', 'modules': modules}
    result = fit_module_check(frame, rgb, match_module_quads(modules, observed, model), 'M11')
    result['binding'] = pair_check_binding(frame.source_path, rgb, profile, diagnostic, 'M11', prior)
    result['content_sha256'] = digest(result)
    return frame, rgb, profile, prior, diagnostic, result


def test_checked_projection_requires_current_source_profile_model_module_and_support(checked_case):
    from pvdelta.rgb_preview import rgb_finding_preview

    frame, rgb, profile, prior, _, result = checked_case
    settings = {'registration': prior, 'camera_model': 'M4T', 'native_size': (640, 512),
        'source_root': rgb.parent, 'expected_rgb_hash': hashlib.sha256(rgb.read_bytes()).hexdigest(),
        'native_verified': True, 'module_polygon': result['target_module']['native_polygon'],
        'pair_check': result, 'measurement_profile': profile}
    polygon = [[240, 170], [245, 170], [245, 175], [240, 175]]
    assert rgb_finding_preview(rgb, polygon, **settings).registration_status == 'image_checked_provisional'
    changed = deepcopy(result)
    changed['check_p95_native_px'] = 99
    assert rgb_finding_preview(rgb, polygon, **(settings | {'pair_check': changed})).registration_status == 'approximate'
    other_profile = profile | {'profile_id': 'different'}
    assert rgb_finding_preview(rgb, polygon, **(settings | {'measurement_profile': other_profile})).registration_status == 'approximate'
    outside = [[90, 50], [95, 50], [95, 55], [90, 55]]
    preview = rgb_finding_preview(rgb, outside, **settings)
    assert preview.registration_status == 'approximate' and 'outside local support' in preview.message
    assert frame.temperature_c[0, 0] == 30


def test_check_ui_is_explicit_discards_stale_sources_and_clears_failed_attempts(checked_case, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from pvdelta import rgb_pair_check

    frame, rgb, profile, prior, diagnostic, _ = checked_case
    calls = []

    def check(thermal, rgb, profile, diagnostic, module, registration, **kwargs):
        calls.append(module)
        if len(calls) == 2:
            raise RuntimeError('SDK unavailable')
        result = {'status': 'unresolved', 'reasons': ['test geometry rejected'],
            'binding': pair_check_binding(thermal, rgb, profile, diagnostic, module, registration)}
        result['content_sha256'] = digest(result)
        return result

    monkeypatch.setattr(rgb_pair_check, 'check_rgb_pair', check)
    script = ('from pvdelta.rgb_pair_check import rgb_pair_check_view\n'
        f'rgb_pair_check_view({frame.source_path!r}, {str(rgb)!r}, {profile!r}, {diagnostic!r}, '
        f'"M11", {prior!r}, sdk_library=None, key="check")\n')
    app = AppTest.from_string(script, default_timeout=20).run()
    assert not app.exception and not calls
    app.button(key='check-start').click().run()
    assert not app.exception and calls == ['M11']
    assert app.session_state['check-result']['status'] == 'unresolved'
    app.button(key='check-start').click().run()
    assert not app.exception and any('SDK unavailable' in w.value for w in app.warning)
    with pytest.raises(KeyError):
        app.session_state['check-result']
    app.button(key='check-start').click().run()
    assert len(calls) == 3
    rgb.write_bytes(b'changed source')
    app.run()
    assert not app.exception
    with pytest.raises(KeyError):
        app.session_state['check-result']


def test_match_explanations_keep_every_native_source_and_original_acceptance(scene):
    from pvdelta.rgb_pair_check import explain_module_matches

    _, _, model, modules, observed = scene
    explained = explain_module_matches(modules, observed, model)
    assert len(explained['decisions']) == len(modules)
    assert explained['matches'] == match_module_quads(modules, observed, model)
    assert {d['status'] for d in explained['decisions']} == {'bounded_mutual_quad_match'}
    assert explained['semantic_panel_membership'] == 'unverified'


def test_junction_review_button_is_explicit_isolated_and_discards_failed_results(checked_case, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from pvdelta import rgb_pair_check

    frame, rgb, profile, prior, diagnostic, _ = checked_case
    calls = []
    def check(thermal, rgb, profile, diagnostic, module, registration, **kwargs):
        calls.append(kwargs)
        assert kwargs['use_junctions'] is True
        if len(calls) == 2:
            raise RuntimeError('SDK unavailable')
        result = {'binding':pair_check_binding(thermal,rgb,profile,diagnostic,module,registration),
            'supported_proposals':0,'requested_proposals':1,'decisions':[{'proposal_id':'P1','status':'withheld'}]}
        result['content_sha256'] = digest(result)
        return result
    monkeypatch.setattr(rgb_pair_check,'check_rgb_pair',check)
    script = ('from pvdelta.rgb_pair_check import rgb_pair_check_view\n'
        f'rgb_pair_check_view({frame.source_path!r}, {str(rgb)!r}, {profile!r}, {diagnostic!r}, '
        f'"M11", {prior!r}, sdk_library=None, key="check")\n')
    app = AppTest.from_string(script,default_timeout=20).run()
    assert not app.exception and not calls
    app.button(key='check-junction-start').click().run()
    assert not app.exception and len(calls)==1 and len(app.dataframe)==1
    with pytest.raises(KeyError):
        app.session_state['check-result']
    app.run()
    assert len(calls)==1
    rgb.write_bytes(b'changed source')
    app.run()
    with pytest.raises(KeyError):
        app.session_state['check-junction-result']
    app.button(key='check-junction-start').click().run()
    assert not app.exception and any('SDK unavailable' in w.value for w in app.warning)
    with pytest.raises(KeyError):
        app.session_state['check-junction-result']


def test_automatic_geometry_reuses_frame_cache_and_does_not_loop_after_sdk_failure(checked_case, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from pvdelta import rgb_pair_check

    frame,rgb,profile,prior,diagnostic,_ = checked_case
    calls = []
    def check(thermal,rgb,profile,diagnostic,module,registration,**kwargs):
        assert module == '__frame_geometry__' and kwargs['use_junctions']
        calls.append(module)
        if len(calls)==2:
            raise RuntimeError('missing SDK')
        result={'binding':pair_check_binding(thermal,rgb,profile,diagnostic,module,registration),
            'requested_proposals':1,'supported_proposals':0,'decisions':[{'proposal_id':'P1','status':'withheld'}]}
        result['content_sha256']=digest(result)
        return result
    monkeypatch.setattr(rgb_pair_check,'check_rgb_pair',check)
    script=('import streamlit as st\nfrom pvdelta.rgb_pair_check import rgb_pair_check_view\n'
        'module=st.selectbox("Module",["M11","M12"])\n'
        f'rgb_pair_check_view({frame.source_path!r},{str(rgb)!r},{profile!r},{diagnostic!r},module,{prior!r},'
        'sdk_library=None,key="auto",automatic_junctions=True)\n')
    app=AppTest.from_string(script,default_timeout=20).run()
    assert not app.exception and len(calls)==1
    app.selectbox[0].select('M12').run()
    assert not app.exception and len(calls)==1
    app.button(key='auto-junction-start').click().run()
    assert not app.exception and len(calls)==2
    app.run()
    assert not app.exception and len(calls)==2
    assert any('missing SDK' in w.value for w in app.warning)


def test_absent_rgb_support_is_unknown_instead_of_dropping_the_source(scene):
    from pvdelta.rgb_pair_check import explain_module_matches

    _, _, model, modules, _ = scene
    result = explain_module_matches(modules, [], model)
    assert not result['matches'] and len(result['decisions']) == len(modules)
    assert {d['status'] for d in result['decisions']} == {'no_observed_rgb_counterpart'}
    assert result['semantic_panel_membership'] == 'unverified'


def test_duplicate_rgb_and_native_competition_have_distinct_reasons(scene):
    from pvdelta.rgb_pair_check import explain_module_matches

    _, _, model, modules, observed = scene
    result = explain_module_matches(modules, observed+[deepcopy(observed[0])], model)
    assert result['decisions'][0]['status'] == 'ambiguous_observed_rgb_targets'
    duplicate = deepcopy(modules[0]); duplicate['module'] = 'other_source'
    result = explain_module_matches(modules+[duplicate], observed, model)
    assert result['decisions'][0]['status'] == 'competing_native_sources'
    assert result['decisions'][-1]['status'] == 'competing_native_sources'


def test_corner_bound_stays_twelve_native_pixels_with_an_explanation(scene):
    from pvdelta.rgb_pair_check import explain_module_matches

    _, _, model, modules, observed = scene
    for target in observed:
        target['polygon_rgb_xy'] = (np.array(target['polygon_rgb_xy'])+[38, 0]).tolist()
    result = explain_module_matches(modules, observed, model)
    assert not result['matches']
    assert any(d['status'] == 'corner_correction_exceeds_native_bound' for d in result['decisions'])
    assert all(d['max_corner_correction_native_px'] > 12 for d in result['decisions']
               if d['status'] == 'corner_correction_exceeds_native_bound')
