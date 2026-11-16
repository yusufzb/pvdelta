import io
import json
from copy import deepcopy

import pytest

pytest.importorskip('PIL')
from PIL import Image

from pvdelta.region_annotation import (
    load_region_pack,
    region_editor_html,
    save_region_export,
    validate_region_export,
)


@pytest.fixture
def prepared(tmp_path):
    Image.new('RGB', (80, 60), 'gray').save(tmp_path/'native.png')
    Image.new('RGB', (160, 120), 'green').save(tmp_path/'rgb.jpg')
    frame = {'schema': 'pvdelta.native-panel-instances/v1', 'case_id': 'C001',
             'source_file': 'frame_T.JPG', 'source_sha256': 'a'*64,
             'native_temperature_sha256': 'b'*64, 'native_height': 60, 'native_width': 80,
             'coordinate_space': 'thermal_native', 'coverage': 'unresolved',
             'review_status': 'unreviewed', 'panels': None, 'image': 'native.png',
             'rgb_context_image': 'rgb.jpg', 'rgb_source_file': 'paired_V.JPG',
             'rgb_source_sha256': 'c'*64}
    path = tmp_path/'pack.json'
    path.write_text(json.dumps({'schema': 'pvdelta.pending-panel-reference-pack/v1',
                                'frames': [frame]}))
    pack = load_region_pack(path)
    f = pack['frames'][0]
    work = {'schema': 'pvdelta.thermal-region-work/v1', 'pack_token': pack['pack_token'],
            'frames': [{**{k: f[k] for k in ('source_sha256', 'native_temperature_sha256',
                         'native_width', 'native_height', 'display_png_sha256')},
                        'regions': [], 'ignore': [], 'exhaustive': False,
                        'reviewer': '', 'note': '', 'empty_reason': ''}]}
    return path, pack, work


def region(identifier='R1', x=5):
    return {'region_id': identifier, 'polygon_native_xy': [[x, 5], [x+4, 5], [x+4, 9], [x, 9]]}


def test_partial_and_multiple_regions_do_not_become_panels_or_truth(prepared):
    _, pack, work = prepared
    w = work['frames'][0]
    w['regions'] = [region(), region('R2', 15)]
    original = deepcopy(work)
    result = validate_region_export(pack, json.dumps(work))
    assert work == original
    f = result['frames'][0]
    assert f['regions'] is None and len(f['draft_regions']) == 2
    assert not f['review_complete'] and not f['ready_to_score']
    assert 'panels' not in f and 'draft_panels' not in f
    w.update(exhaustive=True, reviewer='operator', independent_review_verified=True,
             measurement_authorised=True, review_status='independently_reviewed')
    f = validate_region_export(pack, json.dumps(work))['frames'][0]
    assert f['review_complete'] and len(f['regions']) == 2
    assert f['review_status'] == 'developer_draft'
    assert not f['ready_to_score'] and not f['independent_review_verified']
    assert not f['measurement_authorised']
    assert f['rgb_source_file'] == 'paired_V.JPG'


def test_empty_completed_frame_requires_reviewer_and_reason(prepared):
    _, pack, work = prepared
    w = work['frames'][0]
    w['exhaustive'] = True
    with pytest.raises(ValueError, match='reviewer'):
        validate_region_export(pack, json.dumps(work))
    w['reviewer'] = 'operator'
    with pytest.raises(ValueError, match='reason'):
        validate_region_export(pack, json.dumps(work))
    w['empty_reason'] = 'No visible review pattern; uncertain area explicitly excluded.'
    w['ignore'] = [region('U1')]
    f = validate_region_export(pack, json.dumps(work))['frames'][0]
    assert f['regions'] == [] and len(f['ignore']) == 1
    assert f['review_complete'] and not f['ready_to_score']


@pytest.mark.parametrize('change', ['overlap', 'ignore_overlap', 'duplicate_id', 'bounds',
                                  'bowtie', 'nan', 'bool', 'subpixel', 'source', 'schema'])
def test_invalid_exports_fail_before_save(prepared, tmp_path, change):
    path, _, work = prepared
    w = work['frames'][0]
    w['regions'] = [region()]
    if change == 'overlap':
        w['regions'].append(region('R2', 6))
    elif change == 'ignore_overlap':
        w['ignore'] = [region('U1')]
    elif change == 'duplicate_id':
        w['regions'].append(region('R1', 15))
    elif change == 'bounds':
        w['regions'][0]['polygon_native_xy'][0][0] = -1
    elif change == 'bowtie':
        w['regions'][0]['polygon_native_xy'] = [[5, 5], [9, 9], [9, 5], [5, 9]]
    elif change in ('nan', 'bool'):
        w['regions'][0]['polygon_native_xy'][0][0] = float('nan') if change == 'nan' else True
    elif change == 'subpixel':
        w['regions'][0]['polygon_native_xy'] = [[5.1, 5.1], [5.2, 5.1], [5.2, 5.2], [5.1, 5.2]]
    elif change == 'source':
        w['source_sha256'] = 'd'*64
    else:
        work['schema'] = 'pvdelta.panel-annotation-work/v1'
    with pytest.raises(ValueError):
        save_region_export(path, json.dumps(work), tmp_path/'saved')
    assert not (tmp_path/'saved').exists()


@pytest.mark.parametrize('changed_image', ['rgb.jpg', 'native.png'])
def test_changed_display_rejects_stale_export(prepared, tmp_path, changed_image):
    path, pack, work = prepared
    output = tmp_path/'saved'
    first, _ = save_region_export(path, json.dumps(work), output)
    same, _ = save_region_export(path, json.dumps(work), output)
    assert first == same
    Image.new('RGB', (80, 60), 'red').save(tmp_path/changed_image)
    assert load_region_pack(path)['pack_token'] != pack['pack_token']
    with pytest.raises(ValueError, match='exact native pack'):
        save_region_export(path, json.dumps(work), output)
    assert len(list(output.iterdir())) == 1


def test_rgb_path_escape_rejected(prepared):
    path, _, _ = prepared
    d = json.loads(path.read_text())
    d['frames'][0]['rgb_context_image'] = '../outside.jpg'
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='inside'):
        load_region_pack(path)


def test_editor_has_region_schema_and_paired_context_no_predictions(prepared):
    _, pack, _ = prepared
    html = region_editor_html(pack)
    assert 'pvdelta.thermal-region-work/v1' in html
    assert 'pvdelta.panel-annotation-work/v1' not in html
    assert 'paired_V.JPG' in html and 'data:image/jpeg;base64,' in html
    assert 'regions:[]' in html and 'panels:[]' not in html
    assert 'No detector predictions are supplied' in html


def test_streamlit_open_validate_save_and_stale_upload(prepared, tmp_path, monkeypatch):
    st = pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest

    path, _, work = prepared
    script = tmp_path/'app.py'
    script.write_text('from pvdelta.region_annotation_ui import region_annotation_view\n'
                      f'region_annotation_view({str(path)!r},output={str(tmp_path/"saved")!r})\n',
                      encoding='utf-8')
    monkeypatch.setattr(st, 'file_uploader', lambda *a, **k: io.BytesIO(json.dumps(work).encode()))
    app = AppTest.from_file(str(script), default_timeout=30).run()
    assert not app.exception and len(app.get('iframe')) == 0
    app.checkbox(key='thermal-reference-load').check().run()
    assert not app.exception
    app.checkbox(key='thermal-reference-open').check().run()
    assert not app.exception and len(app.get('iframe')) == 1
    app.button(key='thermal-reference-save').click().run()
    assert not app.exception and any('Saved 0/1' in s.value for s in app.success)
    assert len(list((tmp_path/'saved').glob('*.json'))) == 1
    work['pack_token'] = 'stale'
    app.run()
    assert not app.exception and any('validation failed' in s.value for s in app.error)
    assert not app.success


def baseline(pack, findings):
    return {'schema': 'pvdelta.frozen-panel-predictions/v1', 'frames': [
        {'source_sha256': f['source_sha256'], 'native_shape': [60, 80],
         'findings': [{'finding_id': r['region_id'], 'polygon_native_xy': r['polygon_native_xy']}
                      for r in findings]} for f in pack['frames']]}


def test_region_protocol_pending_is_not_zero_errors(prepared):
    from pvdelta.region_reference_benchmark import evaluate_region_work, freeze_region_protocol

    _, pack, work = prepared
    protocol = freeze_region_protocol(pack, baseline(pack, [region()]))
    result = evaluate_region_work(pack, json.dumps(work), protocol)
    assert result['scored_frames'] == 0 and result['pending_frames'] == 1
    assert result['totals'] is None
    assert result['frames'][0]['predictions'] == 1


def test_region_protocol_counts_duplicates_misses_and_uncertain(prepared):
    from pvdelta.region_reference_benchmark import evaluate_region_work, freeze_region_protocol

    _, pack, work = prepared
    w = work['frames'][0]
    w.update(exhaustive=True, reviewer='synthetic', regions=[region('ref'), region('missed', 30)],
             ignore=[region('unknown', 50)])
    protocol = freeze_region_protocol(pack, baseline(pack, [region('match'), region('duplicate'),
                                                           region('uncertain', 50), region('wrong', 65)]))
    original = deepcopy(protocol)
    result = evaluate_region_work(pack, json.dumps(work), protocol)
    assert protocol == original
    t = result['totals']['primary']
    assert (t['true_positives'], t['false_positives'], t['false_negatives']) == (1, 2, 1)
    assert t['ignored_predictions'] == 1 and t['recall'] == .5
    assert result['frames'][0]['primary']['missed_reference_ids'] == ['missed']
    assert not result['independent_review_verified'] and not result['stage1_accepted']


def test_completed_empty_reference_counts_false_positives(prepared):
    from pvdelta.region_reference_benchmark import evaluate_region_work, freeze_region_protocol

    _, pack, work = prepared
    work['frames'][0].update(exhaustive=True, reviewer='synthetic', empty_reason='Synthetic empty case')
    protocol = freeze_region_protocol(pack, baseline(pack, [region()]))
    t = evaluate_region_work(pack, json.dumps(work), protocol)['totals']['primary']
    assert t['false_positives'] == 1 and t['recall'] is None and t['precision'] == 0


def test_empty_prediction_population_keeps_missed_reference(prepared):
    from pvdelta.region_reference_benchmark import evaluate_region_work, freeze_region_protocol

    _, pack, work = prepared
    work['frames'][0].update(exhaustive=True, reviewer='synthetic', regions=[region()])
    protocol = freeze_region_protocol(pack, baseline(pack, []))
    t = evaluate_region_work(pack, json.dumps(work), protocol)['totals']['primary']
    assert t['false_negatives'] == 1 and t['recall'] == 0 and t['precision'] is None


@pytest.mark.parametrize('change', ['threshold', 'pack', 'prediction', 'rehash_rules'])
def test_protocol_changes_rejected(prepared, change):
    from pvdelta.panel_annotation import _canonical, _sha
    from pvdelta.region_reference_benchmark import evaluate_region_work, freeze_region_protocol

    _, pack, work = prepared
    protocol = freeze_region_protocol(pack, baseline(pack, [region()]))
    if change in ('threshold', 'rehash_rules'):
        protocol['rules']['primary_iou'] = .01
    elif change == 'pack':
        protocol['pack_token'] = 'wrong'
    else:
        protocol['frames'][0]['predictions'] = []
    if change == 'rehash_rules':
        protocol['protocol_sha256'] = _sha(_canonical({k: v for k, v in protocol.items()
                                                      if k != 'protocol_sha256'}))
    with pytest.raises(ValueError):
        evaluate_region_work(pack, json.dumps(work), protocol)


def test_auxiliary_overlap_does_not_replace_primary_score(prepared):
    from pvdelta.region_reference_benchmark import evaluate_region_work, freeze_region_protocol

    _, pack, work = prepared
    work['frames'][0].update(exhaustive=True, reviewer='synthetic', regions=[region('ref')])
    protocol = freeze_region_protocol(pack, baseline(pack, [region('shifted', 7)]))
    scores = evaluate_region_work(pack, json.dumps(work), protocol)['totals']
    assert scores['primary']['true_positives'] == 0
    assert scores['primary']['false_negatives'] == 1
    assert scores['auxiliary']['true_positives'] == 1


def test_partial_second_frame_cannot_add_false_positives(prepared):
    from pvdelta.region_reference_benchmark import evaluate_region_work, freeze_region_protocol

    _, pack, work = prepared
    second = dict(pack['frames'][0], source_sha256='d'*64, case_id='C002')
    pack['frames'].append(second)
    work['frames'].append(dict(work['frames'][0], source_sha256='d'*64))
    work['frames'][0].update(exhaustive=True, reviewer='synthetic', regions=[region()])
    protocol = freeze_region_protocol(pack, baseline(pack, [region()]))
    result = evaluate_region_work(pack, json.dumps(work), protocol)
    assert result['scored_frames'] == result['pending_frames'] == 1
    assert result['totals']['primary']['true_positives'] == 1
    assert result['totals']['primary']['false_positives'] == 0
    assert 'primary' not in result['frames'][1]


def test_streamlit_baseline_comparison_keeps_pending_explicit(prepared, tmp_path, monkeypatch):
    st = pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest

    from pvdelta.region_reference_benchmark import freeze_region_protocol

    path, pack, work = prepared
    path.with_name('evaluation-protocol.json').write_text(json.dumps(
        freeze_region_protocol(pack, baseline(pack, [region()]))))
    path.with_name('evaluation-protocol-v2.json').write_text(json.dumps(
        freeze_region_protocol(pack, baseline(pack, [region()]), version=2)))
    script = tmp_path/'score_app.py'
    script.write_text('from pvdelta.region_annotation_ui import region_annotation_view\n'
                      f'region_annotation_view({str(path)!r},output={str(tmp_path/"saved")!r})\n',
                      encoding='utf-8')
    monkeypatch.setattr(st, 'file_uploader', lambda *a, **k: io.BytesIO(json.dumps(work).encode()))
    app = AppTest.from_file(str(script), default_timeout=30).run()
    app.checkbox(key='thermal-reference-load').check().run()
    app.checkbox(key='thermal-reference-score').check().run()
    assert not app.exception and any('Awaiting full-frame' in i.value for i in app.info)
    work['frames'][0].update(exhaustive=True, reviewer='synthetic', regions=[region()])
    app.run()
    assert not app.exception and not app.error
    assert len(app.dataframe) == 2
    assert app.dataframe[1].value.iloc[0]['true_positives'] == 1


@pytest.mark.parametrize('reverse', [False, True])
def test_concave_reference_keeps_notch_in_saved_geometry_and_raster_score(prepared, reverse):
    from pvdelta.region_polygon import simple_polygon_mask
    from pvdelta.region_reference_benchmark import evaluate_region_work, freeze_region_protocol

    _, pack, work = prepared
    polygon = [[5, 5], [15, 5], [15, 9], [9, 9], [9, 15], [5, 15]]
    if reverse:
        polygon.reverse()
    mask = simple_polygon_mask(polygon, (60, 80))
    assert mask.sum() == 64 and not mask[12, 12] and mask[6, 12]
    work['frames'][0].update(exhaustive=True, reviewer='synthetic',
                             regions=[{'region_id': 'L', 'polygon_native_xy': polygon}])
    checked = validate_region_export(pack, json.dumps(work))
    assert checked['frames'][0]['regions'][0]['polygon_native_xy'] == polygon
    # The detected rectangle lies entirely in the inward notch, so it must not match.
    notch = {'region_id': 'notch', 'polygon_native_xy': [[10,10],[14,10],[14,14],[10,14]]}
    protocol = freeze_region_protocol(pack, baseline(pack, [notch]), version=2)
    scores = evaluate_region_work(pack, json.dumps(work), protocol)
    assert scores['totals']['auxiliary']['true_positives'] == 0
    assert scores['totals']['primary']['false_negatives'] == 1


@pytest.mark.parametrize('polygon', [
    [[0,0],[5,0],[3,0],[3,5],[0,5]],  # adjacent edge backtrack
    [[0,0],[6,0],[6,6],[3,0],[0,6]],  # nonadjacent edge touches
    [[0,0],[6,0],[6,6],[0,6],[6,0]],  # repeated vertex
])
def test_non_simple_polygon_rejected(polygon):
    from pvdelta.region_polygon import simple_polygon_mask

    with pytest.raises(ValueError):
        simple_polygon_mask(polygon, (60, 80))


def test_triangle_and_collinear_forward_vertex_are_valid():
    from pvdelta.region_polygon import simple_polygon_mask

    assert simple_polygon_mask([[0,0],[5,0],[0,5]], (60,80)).any()
    assert simple_polygon_mask([[0,0],[2,0],[5,0],[5,5],[0,5]], (60,80)).sum() == 25
