import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from pvdelta import cli
from pvdelta.control_review import (
    export_reviewed_controls,
    latest_control_decisions,
    load_review_pack,
    prepare_control_review,
    save_control_decision,
)
from pvdelta.local_rgb_transfer import _context
from pvdelta.models import ThermalFrame


@pytest.fixture
def controls(tmp_path, monkeypatch):
    thermal, rgb = tmp_path/'original_T.JPG', tmp_path/'original_V.JPG'
    thermal.write_bytes(b'original thermal source')
    rgb.write_bytes(cv2.imencode('.png', np.full((128, 160, 3), 90, np.uint8))[1].tobytes())
    values = np.linspace(20, 40, 64*80, dtype=np.float32).reshape(64, 80)

    class Reader:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def temperature(self):
            return values.copy()

        def processed_rgb(self):
            return np.full((64, 80, 3), 80, np.uint8)

    monkeypatch.setattr('pvdelta.decode.RJpeg', Reader)
    frame = ThermalFrame('test', str(thermal), hashlib.sha256(thermal.read_bytes()).hexdigest(),
                         values, None, 80, 64)
    parent = {'frames': [{'schema': 'pvdelta.local-rgb-controls/v1', 'pair': _context(frame, rgb),
        'provenance': {'kind': 'synthetic', 'note': 'unit test only'},
        'points': [{'control_id': f'{x}-{y}', 'role': 'fit' if (i+j)%2 == 0 else 'check',
                    'native_xy': [x, y], 'rgb_xy': [2*x, 2*y]}
                   for i, x in enumerate((10, 25, 40, 55)) for j, y in enumerate((10, 25, 40))]}]}
    sources = [{'thermal': str(thermal), 'rgb': str(rgb)}]
    folder = tmp_path/'pack'
    return parent, sources, folder, tmp_path/'decisions'


def decide(pack, folder, point, *, status='OK'):
    return save_control_decision(pack, folder, point['control_id'],
        native_xy=[v+.25 for v in point['native_xy']], rgb_xy=point['rgb_xy'],
        status=status, reviewer='unit-test reviewer', note='source-only synthetic review')


@pytest.fixture
def two_pairs(controls):
    parent,sources,folder,output=controls
    second=deepcopy(parent['frames'][0])
    thermal=folder.parent/'second_T.JPG';thermal.write_bytes(b'second original thermal')
    rgb=folder.parent/'second_V.JPG'
    rgb.write_bytes(cv2.imencode('.png',np.full((128,160,3),110,np.uint8))[1].tobytes())
    second['pair']['source_sha256']=hashlib.sha256(thermal.read_bytes()).hexdigest()
    second['pair']['rgb_sha256']=hashlib.sha256(rgb.read_bytes()).hexdigest()
    for point in second['points']:point['control_id']='second::'+point['control_id']
    parent['frames'].append(second)
    sources.append({'thermal':str(thermal),'rgb':str(rgb)})
    pack=prepare_control_review(parent,sources,folder)
    return pack,folder,output


def test_complete_pair_exports_without_approving_or_dropping_other_pair_controls(two_pairs):
    pack,_folder,output=two_pairs
    old=json.dumps(pack,sort_keys=True)
    for point in pack['frames'][0]['controls']['points']:decide(pack,output,point)
    path=export_reviewed_controls(pack,output,frame_index=0)
    bundle=json.loads(path.read_bytes());original=pack['frames'][0]['controls']
    assert len(bundle['frames'])==1 and bundle['frames'][0]['pair']==original['pair']
    assert [(p['control_id'],p['role']) for p in bundle['frames'][0]['points']]==[
        (p['control_id'],p['role']) for p in original['points']]
    assert bundle['frames'][0]['provenance']['source_pair_index']==0
    assert not bundle['frames'][0]['provenance']['independent_review_verified']
    assert len(latest_control_decisions(pack,output))==len(original['points'])
    assert json.dumps(pack,sort_keys=True)==old
    for kwargs in ({},{'frame_index':1}):
        with pytest.raises(ValueError,match='Every original'):export_reviewed_controls(pack,output,**kwargs)
    decide(pack,output,original['points'][0],status='Wait')
    with pytest.raises(ValueError,match='Every original'):
        export_reviewed_controls(pack,output,frame_index=0)
    assert path.is_file()


@pytest.mark.parametrize('index',[-1,2,True,'0',.5])
def test_pair_export_rejects_invalid_or_ambiguous_selection(two_pairs,index):
    pack,_,output=two_pairs
    with pytest.raises(ValueError,match='pair index'):
        export_reviewed_controls(pack,output,frame_index=index)
    assert not output.exists()


def test_pair_export_ui_keeps_pending_pairs_and_status_filters_out_of_export_gate(two_pairs,tmp_path):
    from streamlit.testing.v1 import AppTest

    pack,folder,output=two_pairs
    for point in pack['frames'][0]['controls']['points']:decide(pack,output,point)
    script=tmp_path/'pair_export_app.py'
    script.write_text('from pvdelta.control_review_ui import control_review_view\n'
        f'control_review_view({str(folder/"review-pack.json")!r},output={str(output)!r})',encoding='utf-8')
    app=AppTest.from_file(str(script),default_timeout=30).run()
    element=lambda items,label:next(e for e in items if e.label==label)
    assert not app.exception
    element(app.selectbox,'Source photo pair').set_value(0).run()
    element(app.selectbox,'Control review status').set_value('Needs review').run()
    assert not app.exception and not element(app.button,'Export selected photo pair').disabled
    assert element(app.button,'Export revised controls').disabled
    element(app.button,'Export selected photo pair').click().run()
    assert not app.exception and app.success
    paths=list(output.glob('controls-*.json'))
    assert len(paths)==1 and len(json.loads(paths[0].read_bytes())['frames'])==1
    element(app.selectbox,'Source photo pair').set_value(1).run()
    assert not app.exception and element(app.button,'Export selected photo pair').disabled
    assert not app.success
    decide(pack,output,pack['frames'][0]['controls']['points'][0],status='Not OK')
    element(app.selectbox,'Source photo pair').set_value(0).run()
    assert not app.exception and element(app.button,'Export selected photo pair').disabled
    assert not app.success


def test_new_pack_is_unmarked_exact_source_bound_and_immutable(controls):
    parent, sources, folder, _ = controls
    original = json.dumps(parent)
    pack = prepare_control_review(parent, sources, folder)
    assert load_review_pack(folder/'review-pack.json') == pack
    assert pack['model_predictions_included'] is False
    assert pack['parent_controls'] == parent and json.dumps(parent) == original
    assert (folder/'000-original-rgb.jpg').read_bytes() == __import__('pathlib').Path(sources[0]['rgb']).read_bytes()
    with pytest.raises(FileExistsError):
        prepare_control_review(parent, sources, folder)


def test_preparation_rejects_other_original_or_native_grid(controls):
    parent, sources, folder, _ = controls
    parent['frames'][0]['pair']['native_temperature_sha256'] = '0'*64
    with pytest.raises(ValueError, match='exact original'):
        prepare_control_review(parent, sources, folder)
    assert not folder.exists()


def test_modified_review_image_invalidates_pack(controls):
    parent, sources, folder, _ = controls
    prepare_control_review(parent, sources, folder)
    (folder/'000-native.png').write_bytes(b'changed preview')
    with pytest.raises(ValueError, match='image identity'):
        load_review_pack(folder/'review-pack.json')


def test_all_controls_required_no_role_change_or_independent_validation_claim(controls):
    parent, sources, folder, output = controls
    pack = prepare_control_review(parent, sources, folder)
    with pytest.raises(ValueError, match='Every original'):
        export_reviewed_controls(pack, output)
    for p in parent['frames'][0]['points']:
        decide(pack, output, p)
    path = export_reviewed_controls(pack, output)
    revised = json.loads(path.read_bytes())
    old, new = parent['frames'][0], revised['frames'][0]
    assert old['pair'] == new['pair']
    assert [p['control_id'] for p in old['points']] == [p['control_id'] for p in new['points']]
    assert [p['role'] for p in old['points']] == [p['role'] for p in new['points']]
    assert new['points'][0]['native_xy'][0] == old['points'][0]['native_xy'][0]+.25
    assert not new['provenance']['independent_review_verified']
    assert export_reviewed_controls(pack, output) == path
    decide(pack, output, old['points'][0], status='Wait')
    assert latest_control_decisions(pack, output)[old['points'][0]['control_id']]['status'] == 'Wait'
    with pytest.raises(ValueError, match='Every original'):
        export_reviewed_controls(pack, output)
    assert path.is_file()  # Previous version remains in history.


def test_unknown_id_outside_coordinates_and_missing_reviewer_rejected(controls):
    parent, sources, folder, output = controls
    pack = prepare_control_review(parent, sources, folder)
    point = parent['frames'][0]['points'][0]
    kwargs = {'native_xy': [10, 10], 'rgb_xy': [20, 20], 'status': 'OK', 'reviewer': 'test', 'note': ''}
    for override in ({'control_id': 'unknown'}, {'native_xy': [float('nan'), 10]},
                     {'rgb_xy': [160, 10]}, {'reviewer': ''}, {'status': 'accepted'}):
        args = {'control_id': point['control_id'], **kwargs, **override}
        with pytest.raises(ValueError):
            save_control_decision(pack, output, **args)


def test_ai_review_is_explicit_and_does_not_become_a_human_or_independent_review(controls):
    parent,sources,folder,output=controls
    pack=prepare_control_review(parent,sources,folder)
    p=parent['frames'][0]['points'][0]
    args={'native_xy':p['native_xy'],'rgb_xy':p['rgb_xy'],'status':'OK',
          'reviewer':'Synthetic reviewer','note':'Source-only development observation'}
    saved=save_control_decision(pack,output,p['control_id'],**args,review_basis='AI_source_visual_review')
    assert saved['review_basis']=='AI_source_visual_review'
    assert not saved['independent_review_verified']
    assert latest_control_decisions(pack,output)[p['control_id']]==saved
    with pytest.raises(ValueError):
        save_control_decision(pack,output,p['control_id'],**args,review_basis='independent_certified')
    for point in parent['frames'][0]['points']:
        save_control_decision(pack,output,point['control_id'],native_xy=point['native_xy'],
            rgb_xy=point['rgb_xy'],status='OK',reviewer='Synthetic reviewer',note='Synthetic source review',
            review_basis='AI_source_visual_review')
    exported=json.loads(export_reviewed_controls(pack,output,frame_index=0).read_bytes())
    assert exported['frames'][0]['provenance']['kind']=='AI_assisted_source_visual_revision'
    assert not exported['frames'][0]['provenance']['independent_review_verified']


def test_changed_role_in_saved_decision_is_not_loaded(controls):
    parent, sources, folder, output = controls
    pack = prepare_control_review(parent, sources, folder)
    decision = decide(pack, output, parent['frames'][0]['points'][0])
    path = output/pack['pack_id']/f"{decision['decision_id']}.json"
    decision['role'] = 'check'
    path.write_text(json.dumps(decision))
    with pytest.raises(ValueError, match='identity'):
        latest_control_decisions(pack, output)


def test_later_review_wins_when_windows_clock_returns_the_same_timestamp(controls,monkeypatch):
    from datetime import datetime, timezone

    import pvdelta.control_review as review

    parent,sources,folder,output=controls
    pack=prepare_control_review(parent,sources,folder)
    fixed=datetime(2026,11,1,tzinfo=timezone.utc)
    class Clock:
        @staticmethod
        def now(tz):return fixed
    monkeypatch.setattr(review,'datetime',Clock)
    p=parent['frames'][0]['points'][0]
    first=decide(pack,output,p,status='OK');second=decide(pack,output,p,status='Wait')
    assert first['reviewed_at']==second['reviewed_at']
    assert first['review_sequence']==1 and second['review_sequence']==2
    assert latest_control_decisions(pack,output)[p['control_id']]==second


def test_prepare_cli_requires_exact_declared_originals(controls, capsys):
    from pathlib import Path

    parent, sources, folder, _ = controls
    controls_file = folder.parent/'controls.json'
    source_file = folder.parent/'sources.json'
    controls_file.write_text(json.dumps(parent))
    source_file.write_text(json.dumps({'frames': sources}))
    assert cli.main(['prepare-control-review', '--controls', str(controls_file),
        '--sources', str(source_file), '--source-root', str(Path(sources[0]['thermal']).parent),
        '--out', str(folder)]) == 0
    assert json.loads(capsys.readouterr().out)['frames'] == 1


def test_control_editor_navigation_save_and_export(controls):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest

    parent, sources, folder, output = controls
    pack = prepare_control_review(parent, sources, folder)
    app = AppTest.from_string('from pvdelta.control_review_ui import control_review_view\n'
        f'control_review_view({str(folder/"review-pack.json")!r},output={str(output)!r},key="test")',
        default_timeout=15).run()

    def element(items, label):
        return next(e for e in items if e.label == label)

    assert not app.exception and not app.error
    assert element(app.button, 'Export revised controls').disabled
    element(app.button, 'Next →').click().run()
    assert not app.exception and element(app.selectbox, 'Source control').value == parent['frames'][0]['points'][1]['control_id']
    element(app.button, '← Previous').click().run()
    element(app.text_input, 'Control reviewer').input('test user').run()
    element(app.radio, 'Source correspondence').set_value('OK')
    element(app.button, 'Save source control').click().run()
    assert not app.exception and not app.error
    assert len(latest_control_decisions(pack, output)) == 1
    for p in parent['frames'][0]['points']:
        decide(pack, output, p)
    app.run()
    assert not element(app.button, 'Export revised controls').disabled
    element(app.button, 'Export revised controls').click().run()
    assert not app.exception and list(output.glob('controls-*.json'))
    app.run()
    assert any('New control bundle saved' in m.value for m in app.success)
    decide(pack, output, parent['frames'][0]['points'][0], status='Wait')
    app.run()
    assert not app.exception and element(app.button, 'Export revised controls').disabled
    assert not app.success


def test_dataset_switch_keeps_path_navigation_and_review_identity_separate(controls):
    from copy import deepcopy

    from streamlit.testing.v1 import AppTest

    parent, sources, folder, output = controls
    first = prepare_control_review(parent, sources, folder)
    second_parent = deepcopy(parent)
    for p in second_parent['frames'][0]['points']:
        p['control_id'] = 'second::'+p['control_id']
    second_folder = folder.parent/'second-pack'
    second = prepare_control_review(second_parent, sources, second_folder)
    assert first['pack_id'] != second['pack_id']
    options = {'Original controls': str(folder/'review-pack.json'),
               'Frame-band controls': str(second_folder/'review-pack.json')}
    script = ('from pvdelta.control_review_ui import control_review_view\n'
        f'control_review_view(output={str(output)!r},key="test",pack_options={options!r})')
    app = AppTest.from_string(script, default_timeout=15).run()
    assert not app.exception and not app.error
    next(b for b in app.button if b.label == 'Next →').click().run()
    assert not app.exception
    first_selection = next(s.value for s in app.selectbox if s.label == 'Source control')
    next(s for s in app.selectbox if s.label == 'Source review dataset').set_value('Frame-band controls').run()
    assert not app.exception and not app.error
    assert next(t.value for t in app.text_input if t.label == 'Source control review pack') == options['Frame-band controls']
    assert next(s.value for s in app.selectbox if s.label == 'Source control').startswith('second::')
    assert not latest_control_decisions(second, output)
    next(s for s in app.selectbox if s.label == 'Source review dataset').set_value('Original controls').run()
    assert not app.exception and not app.error
    assert next(t.value for t in app.text_input if t.label == 'Source control review pack') == options['Original controls']
    # Streamlit may discard hidden widget state; a reset is safe, cross-pack reuse is not.
    assert next(s.value for s in app.selectbox if s.label == 'Source control') in {
        first_selection, parent['frames'][0]['points'][0]['control_id']}


def test_filtered_queue_save_next_does_not_approve_hidden_controls(controls):
    from streamlit.testing.v1 import AppTest

    parent, sources, folder, output = controls
    pack = prepare_control_review(parent, sources, folder)
    app = AppTest.from_string('from pvdelta.control_review_ui import control_review_view\n'
        f'control_review_view({str(folder/"review-pack.json")!r}, output={str(output)!r})',
        default_timeout=15).run()

    def element(items, label):
        return next(e for e in items if e.label == label)

    element(app.selectbox, 'Control review status').set_value('Unreviewed').run()
    element(app.checkbox, 'Show full source frames').check().run()
    assert not app.exception and not app.error
    element(app.radio, 'Source correspondence').set_value('OK')
    # A failed save must not advance or create a decision.
    element(app.button, 'Save and next').click().run()
    assert app.error and not latest_control_decisions(pack, output)
    assert element(app.selectbox, 'Source control').value == parent['frames'][0]['points'][0]['control_id']
    element(app.text_input, 'Control reviewer').input('test reviewer').run()
    element(app.radio, 'Source correspondence').set_value('OK')
    element(app.button, 'Save and next').click().run()
    assert not app.exception and not app.error
    assert element(app.selectbox, 'Source control').value == parent['frames'][0]['points'][1]['control_id']
    assert len(latest_control_decisions(pack, output)) == 1
    element(app.selectbox, 'Control review status').set_value('OK').run()
    assert len(element(app.selectbox, 'Source control').options) == 1
    assert element(app.button, 'Export revised controls').disabled
    element(app.selectbox, 'Control review status').set_value('Not OK').run()
    assert not app.exception and not app.error
    assert any('No source controls' in m.value for m in app.info)
    assert element(app.button, 'Export revised controls').disabled
    for point in parent['frames'][0]['points']:
        decide(pack, output, point)
    app.run()
    # An empty visible queue must not hide the completed whole-pack export.
    assert not element(app.button, 'Export revised controls').disabled


def test_frame_filter_navigation_and_status_keep_source_identity(controls):
    from copy import deepcopy

    from streamlit.testing.v1 import AppTest

    parent, sources, folder, output = controls
    second = deepcopy(parent['frames'][0])
    for point in second['points']:
        point['control_id'] = 'second::'+point['control_id']
    parent['frames'].append(second)
    pack = prepare_control_review(parent, sources*2, folder)
    app = AppTest.from_string('from pvdelta.control_review_ui import control_review_view\n'
        f'control_review_view({str(folder/"review-pack.json")!r}, output={str(output)!r})',
        default_timeout=15).run()
    frame = next(e for e in app.selectbox if e.label == 'Source photo pair')
    frame.set_value(1).run()
    assert not app.exception
    assert next(e.value for e in app.selectbox if e.label == 'Source control').startswith('second::')
    next(b for b in app.button if b.label == 'Next →').click().run()
    assert next(e.value for e in app.selectbox if e.label == 'Source control') == second['points'][1]['control_id']
    next(e for e in app.selectbox if e.label == 'Source photo pair').set_value(0).run()
    assert not app.exception and not app.error
    assert next(e.value for e in app.selectbox if e.label == 'Source control') == parent['frames'][0]['points'][0]['control_id']
    assert not latest_control_decisions(pack, output)


def test_context_and_edge_crop_preserve_geometry_and_source_bytes(tmp_path):
    from PIL import Image

    from pvdelta.control_review_ui import _crop, _overview

    path = tmp_path/'source.png'
    Image.new('RGB', (1200, 600), 'white').save(path)
    original = path.read_bytes()
    context = _overview(path, [600, 300])
    assert context.size == (1000, 500)
    assert context.getpixel((500, 250)) == (255, 0, 255)
    crop, box = _crop(path, [0, 30], 20)
    assert box == (0, 10, 21, 51)
    assert crop.size == (410, 410)
    assert crop.getpixel((0, 205)) == (32, 32, 32)  # padding, not stretched pixels
    assert path.read_bytes() == original
