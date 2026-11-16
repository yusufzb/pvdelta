import json
from copy import deepcopy

import pytest

pytest.importorskip('PIL')
from PIL import Image

from pvdelta.instance_benchmark import evaluate_panel_instances
from pvdelta.panel_annotation import (
    annotation_editor_html,
    load_annotation_pack,
    save_annotation_export,
    validate_annotation_export,
)


@pytest.fixture
def prepared(tmp_path):
    frames = []
    for i in range(2):
        Image.new('RGB', (80, 60), (i, 0, 0)).save(tmp_path / f'{i}.png')
        frames.append({'schema':'pvdelta.native-panel-instances/v1', 'coordinate_space':'thermal_native',
            'case_id':f'frame-{i}', 'source_file':f'{i}_T.JPG', 'source_sha256':str(i)*64,
            'native_temperature_sha256':str(i+2)*64, 'native_height':60, 'native_width':80,
            'coverage':'unresolved', 'review_status':'unreviewed', 'panels':None, 'image':f'{i}.png'})
    path = tmp_path / 'pending.json'
    path.write_text(json.dumps({'schema':'pvdelta.pending-panel-reference-pack/v1', 'frames':frames}))
    pack = load_annotation_pack(path)
    work = {'schema':'pvdelta.panel-annotation-work/v1', 'pack_token':pack['pack_token'],
        'frames':[{**{k:v for k,v in f.items() if k in ('source_sha256', 'native_temperature_sha256',
            'native_height', 'native_width', 'display_png_sha256')},
            'panels':[], 'ignore':[], 'exhaustive':False, 'reviewer':'', 'note':'', 'empty_reason':''}
            for f in pack['frames']]}
    return path, pack, work


def polygon(x=5):
    return [[x,5],[x+20,5],[x+20,40],[x,40]]


def content(work):
    return json.dumps(work).encode()


def test_partial_work_stays_unscored_even_with_valid_shapes(prepared):
    _, pack, work = prepared
    work['frames'][0]['panels'] = [{'instance_id':'P1', 'polygon_native_xy':polygon()}]
    original = deepcopy(work)
    bundle = validate_annotation_export(pack, content(work))
    assert work == original
    assert bundle['completed_frames'] == 0
    for frame in bundle['frames']:
        assert frame['panels'] is None and frame['coverage'] == 'unresolved'
        with pytest.raises(ValueError, match='exhaustive'):
            evaluate_panel_instances(frame, [], source_sha256=frame['source_sha256'], native_shape=(60,80))
    assert len(bundle['frames'][0]['draft_panels']) == 1
    assert not any(bundle[k] for k in ('independent_review_verified','database_written','measurement_authorised'))


def test_completion_is_only_a_developer_draft_and_works_with_existing_scorer(prepared):
    _, pack, work = prepared
    w = work['frames'][0]
    w.update(exhaustive=True, reviewer=' Actual reviewer ', review_status='independently_reviewed',
             measurement_authorised=True, panels=[{'instance_id':'P1', 'polygon_native_xy':polygon()}])
    bundle = validate_annotation_export(pack, content(work))
    ref = bundle['frames'][0]
    assert bundle['completed_frames'] == 1 and ref['reviewer'] == 'Actual reviewer'
    assert ref['review_status'] == 'developer_draft'
    assert not ref['independent_review_verified'] and not ref['measurement_authorised']
    score = evaluate_panel_instances(ref, [{'proposal_id':'p','polygon_native_xy':polygon()}],
                                    source_sha256=ref['source_sha256'], native_shape=(60,80))
    assert score['true_positives'] == 1 and score['false_negatives'] == 0


@pytest.mark.parametrize('field,value', [('pack_token','x'), ('source_sha256','f'*64),
    ('native_temperature_sha256','f'*64), ('native_width',81), ('native_height',61),
    ('display_png_sha256','f'*64), ('exhaustive',1), ('reviewer',None), ('source_sha256',[])])
def test_stale_or_malformed_identity_and_completion_rejected(prepared, field, value):
    _, pack, work = prepared
    (work if field == 'pack_token' else work['frames'][0])[field] = value
    with pytest.raises(ValueError):
        validate_annotation_export(pack, content(work))


@pytest.mark.parametrize('change', ['overlap','ignore_overlap','out_of_bounds','bowtie','nan',
                                   'bool','duplicate_id','three_corners','degenerate'])
def test_invalid_geometry_is_rejected_even_in_incomplete_drafts(prepared, change):
    _, pack, work = prepared
    w = work['frames'][0]
    w['panels'] = [{'instance_id':'P1','polygon_native_xy':polygon()}]
    if change == 'overlap':
        w['panels'].append({'instance_id':'P2','polygon_native_xy':polygon(10)})
    elif change == 'ignore_overlap':
        w['ignore'] = [{'region_id':'I1','polygon_native_xy':polygon()}]
    elif change == 'out_of_bounds':
        w['panels'][0]['polygon_native_xy'][0][0] = -1
    elif change == 'bowtie':
        w['panels'][0]['polygon_native_xy'] = [[5,5],[25,40],[25,5],[5,40]]
    elif change == 'nan':
        w['panels'][0]['polygon_native_xy'][0][0] = float('nan')
    elif change == 'bool':
        w['panels'][0]['polygon_native_xy'][0][0] = True
    elif change == 'duplicate_id':
        w['panels'].append({'instance_id':'P1','polygon_native_xy':polygon(40)})
    elif change == 'three_corners':
        w['panels'][0]['polygon_native_xy'].pop()
    else:
        w['panels'][0]['polygon_native_xy'] = [[5,5]]*4
    with pytest.raises(ValueError):
        validate_annotation_export(pack, content(work))


def test_zero_panel_completion_needs_name_and_explicit_reason(prepared):
    _, pack, work = prepared
    w = work['frames'][0]
    w['exhaustive'] = True
    with pytest.raises(ValueError, match='reviewer'):
        validate_annotation_export(pack, content(work))
    w['reviewer'] = 'reviewer'
    with pytest.raises(ValueError, match='zero-panel'):
        validate_annotation_export(pack, content(work))
    w['empty_reason'] = 'All visible modules are partial, explicitly ignored.'
    ref = validate_annotation_export(pack, content(work))['frames'][0]
    assert ref['panels'] == [] and ref['ready_to_score']


def test_duplicate_and_missing_export_sources_rejected(prepared):
    _, pack, work = prepared
    work['frames'][1] = deepcopy(work['frames'][0])
    with pytest.raises(ValueError, match='unique'):
        validate_annotation_export(pack, content(work))
    work['frames'].pop()
    with pytest.raises(ValueError, match='exact'):
        validate_annotation_export(pack, content(work))


@pytest.mark.parametrize('change', ['resize','outside_pack','resolved','duplicate','native_bool'])
def test_bad_native_pack_rejected(prepared, change):
    path, _, _ = prepared
    d = json.loads(path.read_text())
    if change == 'resize':
        Image.new('RGB',(40,30)).save(path.parent/'0.png')
    elif change == 'outside_pack':
        d['frames'][0]['image'] = '../outside.png'
    elif change == 'resolved':
        d['frames'][0]['panels'] = []
    elif change == 'duplicate':
        d['frames'][1]['source_sha256'] = d['frames'][0]['source_sha256']
    else:
        d['frames'][0]['native_width'] = True
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_annotation_pack(path)


def test_save_versions_idempotence_and_image_change_rejected(prepared, tmp_path):
    path, pack, work = prepared
    output = tmp_path/'private'
    first, saved = save_annotation_export(path, content(work), output)
    second, _ = save_annotation_export(path, content(work), output)
    assert first == second and saved['completed_frames'] == 0
    work['frames'][0]['note'] = 'Changed draft'
    third, _ = save_annotation_export(path, content(work), output)
    assert third != first and len(list(output.glob('*.json'))) == 2
    Image.new('RGB',(80,60),'red').save(path.parent/'0.png')
    with pytest.raises(ValueError, match='exact native pack'):
        save_annotation_export(path, content(work), output)
    assert pack['pack_token'] != load_annotation_pack(path)['pack_token']
    assert len(list(output.glob('*.json'))) == 2


def test_tampered_saved_version_never_overwritten(prepared, tmp_path):
    path, _, work = prepared
    saved, _ = save_annotation_export(path, content(work), tmp_path/'private')
    saved.write_text('tampered')
    with pytest.raises(ValueError, match='refusing to overwrite'):
        save_annotation_export(path, content(work), tmp_path/'private')
    assert saved.read_text() == 'tampered'


def test_editor_is_self_contained_and_script_identity_escaped(prepared):
    _, pack, _ = prepared
    pack['frames'][0]['source_file'] = '</script><script>alert(1)</script>'
    page = annotation_editor_html(pack)
    embedded, _ = json.JSONDecoder().raw_decode(page.split('const PACK=', 1)[1])
    assert embedded == pack
    assert '__PVDELTA_PACK__' not in page
    assert 'data:image/png;base64,' in page
    assert pack['frames'][0]['source_file'] not in page
    assert 'https://' not in page and 'http://' not in page
    assert 'Previous' in page and 'Next' in page
