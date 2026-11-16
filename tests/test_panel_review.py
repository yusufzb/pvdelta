import hashlib
import json

import pytest

from pvdelta.panel_review import filter_panel_groups, panel_groups, save_panel_review
from pvdelta.store import (
    CaptureRecord,
    connect_store,
    upsert_capture,
    upsert_finding_review,
    upsert_run,
)


@pytest.fixture
def panel_database(tmp_path):
    from PIL import Image

    database = tmp_path/'panels.duckdb'
    c = connect_store(database)
    for run in ('photos', 'analysis', 'other-analysis'):
        upsert_run(c, run_id=run, source_root=str(tmp_path), started_at_utc=None, site_label=None, camera_model=None)
    for capture in ('t1', 't2'):
        path = tmp_path/f'{capture}.JPG'
        Image.new('RGB', (64, 64), '#303030').save(path)
        upsert_capture(c, CaptureRecord(capture, path.name, hashlib.sha256(path.read_bytes()).hexdigest(),
            'thermal', run_id='photos', width_px=64, height_px=64, paired_capture_id='r'+capture[1:]))
        rgb = tmp_path/f'r{capture[1:]}.JPG'
        Image.new('RGB', (128, 128), '#404040').save(rgb)
        upsert_capture(c, CaptureRecord('r'+capture[1:], rgb.name, hashlib.sha256(rgb.read_bytes()).hexdigest(),
            'rgb', run_id='photos', width_px=128, height_px=128))
    module = json.dumps([[5, 5], [55, 5], [55, 55], [5, 55]])
    for observation, capture, run in (('obs1', 't1', 'analysis'), ('obs2', 't2', 'analysis'),
                                       ('other-obs', 't1', 'other-analysis')):
        c.execute('''INSERT INTO module_observations (observation_id,capture_id,frame_module_id,
            pixel_polygon_json,visibility,analysis_run_id,assessment_status)
            VALUES (?,?,'M004',?,'full',?,'analyzed')''', [observation, capture, module, run])
    for fid, observation, run, x in (('f1', 'obs1', 'analysis', 15), ('f2', 'obs1', 'analysis', 35),
                                    ('f3', 'obs2', 'analysis', 15), ('f4', 'other-obs', 'other-analysis', 15)):
        polygon = json.dumps([[x-1, x-1], [x+2, x-1], [x+2, x+2], [x-1, x+2]])
        c.execute('''INSERT INTO findings (finding_id,observation_id,anomaly_type,assessment_status,
            delta_t_k,absolute_temperature_c,reference_temperature_c,pixel_polygon_json,
            analysis_run_id,region_pixels_yx_json,n_region_px)
            VALUES (?,?,'hot_region','review_required',10,40,30,?,?,?,1)''',
            [fid, observation, polygon, run, json.dumps([[x, x]])])
    upsert_finding_review(c, finding_id='f1', review_status='ok', note='first note')
    upsert_finding_review(c, finding_id='f2', review_status='wait', note='second note')
    c.close()
    return database


@pytest.fixture
def pair_proposals(panel_database):
    from pvdelta.identity_store import observation_binding
    from pvdelta.panel_pair_review import load_pair_proposals

    c = connect_store(panel_database, read_only=True)
    groups = panel_groups(c, 'analysis')
    c.close()
    content = json.dumps([{'source': 'obs1', 'target': 'obs2',
        'status': 'overlap_only_review_required', 'camera_distance_m': 3., 'iou': .9, 'reverse_iou': .9}]).encode()
    path = panel_database.parent/'pairs.json'
    path.write_bytes(content)
    panel_database.with_suffix('.panel-pairs.json').write_text(json.dumps({
        'database': str(panel_database), 'run': 'analysis', 'pairs': str(path),
        'pairs_sha256': hashlib.sha256(content).hexdigest(),
        'bindings': {g['observation_id']: observation_binding(g) for g in groups}}), encoding='utf-8')
    return groups, load_pair_proposals(panel_database, groups)['pairs'][0]


def test_pair_review_revisions_preserve_sources_and_do_not_assign_identity(panel_database, pair_proposals):
    from pvdelta.panel_pair_review import pair_decisions, save_pair_decision

    _, pair = pair_proposals
    before = hashlib.sha256(panel_database.read_bytes()).hexdigest()
    paths = [save_pair_decision(panel_database, pair, status, 'operator')
             for status in ('wait', 'same_panel', 'wait')]
    assert len(set(paths)) == 3
    assert pair_decisions(panel_database, [pair])[pair['pair_id']]['status'] == 'wait'
    assert not panel_database.with_suffix('.identities.duckdb').exists()
    assert hashlib.sha256(panel_database.read_bytes()).hexdigest() == before
    paths[0].write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='content changed'):
        pair_decisions(panel_database, [pair])


@pytest.mark.parametrize('change', ['source', 'geometry', 'reviewer', 'status', 'binding'])
def test_pair_review_rejects_changed_source_or_invalid_decision(panel_database, pair_proposals, change):
    from pvdelta.panel_pair_review import save_pair_decision

    _, pair = pair_proposals
    status, reviewer = 'same_panel', 'operator'
    if change == 'source':
        (panel_database.parent/'r2.JPG').write_bytes(b'changed')
    elif change == 'geometry':
        c = connect_store(panel_database)
        c.execute("UPDATE module_observations SET pixel_polygon_json='[]' WHERE observation_id='obs2'")
        c.close()
    elif change == 'reviewer':
        reviewer = ' '
    elif change == 'status':
        status = 'automatic_pass'
    else:
        pair['pair_id'] = 'unbound'
    with pytest.raises(ValueError):
        save_pair_decision(panel_database, pair, status, reviewer)
    assert not panel_database.with_suffix('.panel-pair-reviews').exists()


def test_pair_proposal_loader_withholds_changed_geometry_and_tampering(panel_database, pair_proposals):
    from pvdelta.panel_pair_review import load_pair_proposals

    groups, _ = pair_proposals
    groups[0]['regions'][0]['module_polygon'] = '[]'
    result = load_pair_proposals(panel_database, groups)
    assert result['pairs'] == [] and result['stale'] == 1
    (panel_database.parent/'pairs.json').write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='proposals changed'):
        load_pair_proposals(panel_database, groups)


def test_pair_proposal_rgb_evidence_is_hash_bound(panel_database, pair_proposals):
    from pvdelta.panel_pair_review import load_pair_proposals

    groups, pair = pair_proposals
    pointer = panel_database.with_suffix('.panel-pairs.json')
    config = json.loads(pointer.read_bytes())
    evidence = panel_database.parent/'rgb-check.json'
    texture = {'pair_id': pair['pair_id'], 'fit_points': 8, 'check_points': 2,
               'neighbour_controls': 4, 'nominal_ray_span_deg': 6.1}
    content = json.dumps([{'source': 'obs1', 'target': 'obs2',
                           'rgb_status': 'overlap_only_review_required',
                           'original_rgb_texture': texture}]).encode()
    evidence.write_bytes(content)
    config.update(rgb_pairs=str(evidence), rgb_pairs_sha256=hashlib.sha256(content).hexdigest())
    pointer.write_text(json.dumps(config), encoding='utf-8')
    loaded, = load_pair_proposals(panel_database, groups)['pairs']
    assert loaded['rgb_status'] == 'overlap_only_review_required'
    assert loaded['original_rgb_texture'] == texture
    evidence.write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='RGB pair evidence changed'):
        load_pair_proposals(panel_database, groups)


def test_pair_review_ui_is_opt_in_and_saves_explicit_decision(panel_database, pair_proposals):
    from streamlit.testing.v1 import AppTest

    from pvdelta.panel_pair_review import pair_decisions

    _, pair = pair_proposals
    script = ('from pvdelta.panel_pair_review_ui import panel_pair_review\n'
        'from pvdelta.store import connect_store\nfrom pvdelta.panel_review import panel_groups\n'
        f'c=connect_store({str(panel_database)!r},read_only=True)\n'
        'groups=panel_groups(c,"analysis")\nc.close()\n'
        f'panel_pair_review({str(panel_database)!r},groups,key="pairs")\n')
    app = AppTest.from_string(script).run()
    assert not app.exception and not app.get('image')
    app.checkbox(key='pairs-pairs-open').check().run()
    assert not app.exception and len(app.get('image')) == 4
    same = next(b for b in app.button if b.label == 'Same panel')
    same.click().run()
    assert not app.exception and any('Reviewer name' in e.value for e in app.error)
    app.text_input(key='pairs-pair-reviewer').set_value('operator')
    next(b for b in app.button if b.label == 'Same panel').click().run()
    assert not app.exception and not app.error
    assert pair_decisions(panel_database, [pair])[pair['pair_id']]['status'] == 'same_panel'
    assert any('same_panel' in i.value for i in app.info)
    queue = 'pairs-pair-queue'
    app.selectbox(key=queue).select('Unreviewed').run()
    assert not app.exception and not app.get('image')
    assert any('No panel pairs' in i.value for i in app.info)
    app.selectbox(key=queue).select('All pairs').run()
    assert not app.exception and len(app.get('image')) == 4
    app.checkbox(key='pairs-pair-rgb-only').check().run()
    assert not app.exception and not app.get('image')
    app.checkbox(key='pairs-pair-rgb-only').uncheck().run()
    assert not app.exception and len(app.get('image')) == 4


@pytest.fixture
def identity_workbench(panel_database):
    from pvdelta.identity_store import observation_binding

    c = connect_store(panel_database, read_only=True)
    groups = panel_groups(c, 'analysis')
    c.close()
    records = []
    for group in groups:
        row = group['regions'][0]
        records.append({'observation_id': group['observation_id'],
            'binding': observation_binding(group), 'thermal_sha256': row['thermal_hash'],
            'rgb': {'sha256': row['rgb_hash'], 'original_size': [128, 128]}})
    pack = {'schema': 'pvdelta.table-identity-review-pack/v1', 'run': 'analysis',
            'records': records}
    pack['package_id'] = hashlib.sha256(json.dumps(pack, sort_keys=True).encode()).hexdigest()
    record = records[0]
    document = {'schema': 'pvdelta.table-identity-decisions/v1',
        'package_id': pack['package_id'], 'run': 'analysis', 'decisions': [{
            'observation_id': record['observation_id'], 'binding': record['binding'],
            'rgb_sha256': record['rgb']['sha256'], 'thermal_sha256': record['thermal_sha256'],
            'status': 'confirmed_table', 'reviewer': 'test operator', 'table_label': 'P01',
            'source_layout_checked': True,
            'polygon_rgb': [[10, 10], [110, 10], [110, 110], [10, 110]]}]}
    path = panel_database.parent/'pack.json'
    path.write_text(json.dumps(pack), encoding='utf-8')
    archive = panel_database.parent/'review.zip'
    archive.write_bytes(b'prepared package test placeholder')
    panel_database.with_suffix('.identity-workbench.json').write_text(json.dumps({
        'database': str(panel_database), 'run': 'analysis', 'pack': str(path),
        'archive': str(archive)}), encoding='utf-8')
    return pack, document


def test_identity_review_saves_idempotently_without_promoting_identity(panel_database, identity_workbench):
    from pvdelta.identity_review import save_decisions, workbench_package

    pack, document = identity_workbench
    before = hashlib.sha256(panel_database.read_bytes()).hexdigest()
    assert workbench_package(panel_database, 'analysis')[0] == pack
    first, report = save_decisions(panel_database, pack, document)
    second, _ = save_decisions(panel_database, pack, document)
    assert first == second and json.loads(first.read_bytes()) == document
    assert report['counts'] == {'confirmed_table': 1}
    assert report['pending_source_cards'] == 1
    assert report['identity_database_updated'] is False
    assert not panel_database.with_suffix('.identities.duckdb').exists()
    assert hashlib.sha256(panel_database.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('bad', ['package', 'duplicate', 'reviewer', 'polygon', 'source',
                                'binding', 'layout', 'empty', 'malformed'])
def test_identity_review_rejects_invalid_input(panel_database, identity_workbench, bad):
    from pvdelta.identity_review import validate_decisions

    pack, document = identity_workbench
    decision = document['decisions'][0]
    if bad == 'package':
        document['package_id'] = 'wrong'
    elif bad == 'duplicate':
        document['decisions'].append(dict(decision))
    elif bad == 'reviewer':
        decision['reviewer'] = ' '
    elif bad == 'polygon':
        decision['polygon_rgb'] = [[1, 1], [90, 90], [1, 90], [90, 1]]
    elif bad == 'source':
        (panel_database.parent/'t1.JPG').write_bytes(b'changed source bytes')
    elif bad == 'binding':
        c = connect_store(panel_database)
        c.execute("UPDATE module_observations SET pixel_polygon_json='[]' WHERE observation_id='obs1'")
        c.close()
    elif bad == 'layout':
        decision['source_layout_checked'] = False
    elif bad == 'empty':
        document['decisions'] = []
    else:
        document['decisions'] = [None]
    with pytest.raises(ValueError):
        validate_decisions(panel_database, pack, document)


def test_identity_workbench_rejects_stale_manifest_and_other_run(panel_database, identity_workbench):
    from pvdelta.identity_review import workbench_package

    with pytest.raises(ValueError, match='different database or run'):
        workbench_package(panel_database, 'other-analysis')
    path = panel_database.parent/'pack.json'
    pack = json.loads(path.read_bytes())
    pack['records'][0]['rgb']['original_size'] = [1, 1]
    path.write_text(json.dumps(pack), encoding='utf-8')
    with pytest.raises(ValueError, match='manifest changed'):
        workbench_package(panel_database, 'analysis')


def test_identity_workbench_ui_load_is_explicit(panel_database, identity_workbench):
    from streamlit.testing.v1 import AppTest

    script = ('from pvdelta.identity_ui import identity_workbench\n'
              f'identity_workbench({str(panel_database)!r}, "analysis", key="review")\n')
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert not app.get('download_button')
    app.checkbox(key='review-workbench-load').check().run()
    assert not app.exception and len(app.get('download_button')) == 1
    assert len(app.get('file_uploader')) == 1


def test_identity_workbench_upload_validation_and_save(panel_database, identity_workbench, monkeypatch):
    from types import SimpleNamespace

    import streamlit as st
    from streamlit.testing.v1 import AppTest

    _, document = identity_workbench
    data = json.dumps(document).encode()
    monkeypatch.setattr(st, 'file_uploader', lambda *a, **kw:
                        SimpleNamespace(size=len(data), getvalue=lambda: data))
    script = ('from pvdelta.identity_ui import identity_workbench\n'
              f'identity_workbench({str(panel_database)!r}, "analysis", key="review")\n')
    app = AppTest.from_string(script).run()
    assert not app.exception and not app.error
    app.button(key='review-workbench-save').click().run()
    assert not app.exception and len(app.success) == 1
    assert len(list(panel_database.with_suffix('.operator-reviews').glob('*.json'))) == 1
    document['decisions'][0]['reviewer'] = ''
    data = json.dumps(document).encode()
    app.run()
    assert not app.exception and any('Reviewer identity required' in e.value for e in app.error)
    assert not any(b.key == 'review-workbench-save' for b in app.button)


def test_grouping_uses_exact_observation_and_run_not_reused_module_id(panel_database):
    before = hashlib.sha256(panel_database.read_bytes()).hexdigest()
    c = connect_store(panel_database, read_only=True)
    groups = panel_groups(c, 'analysis')
    assert len(groups) == 2 and [g['region_count'] for g in groups] == [2, 1]
    assert groups[0]['module_id'] == groups[1]['module_id'] == 'M004'
    assert groups[0]['capture_id'] != groups[1]['capture_id']
    assert groups[0]['review_status'] == 'mixed'
    assert len(panel_groups(c, 'other-analysis')) == 1
    c.close()
    assert hashlib.sha256(panel_database.read_bytes()).hexdigest() == before


def test_identity_hierarchy_is_stable_source_bound_and_preserves_all_regions(panel_database):
    from copy import deepcopy

    from pvdelta.identity_store import (
        assign_identity,
        clear_identity,
        identity_groups,
        identity_snapshot,
    )

    c=connect_store(panel_database,read_only=True)
    groups=panel_groups(c,'analysis')
    c.close()
    before=hashlib.sha256(panel_database.read_bytes()).hexdigest()
    assert len(identity_groups(groups,identity_snapshot(panel_database,groups)))==2
    table,panel=assign_identity(panel_database,groups[0],'p01','B-01')
    assert assign_identity(panel_database,groups[0],'P01','b-01')==(table,panel)
    other,_=assign_identity(panel_database,groups[1],'P01')
    assert other==table
    snapshot=identity_snapshot(panel_database,groups)
    assert {r['level'] for r in identity_groups(groups,snapshot)}=={'table','panel'}
    together=identity_groups(groups,snapshot,prefer_table=True)
    assert len(together)==1 and len(together[0]['regions'])==3
    changed=deepcopy(groups[0])
    changed['regions'][0]['module_polygon']='[]'
    assert identity_snapshot(panel_database,[changed])['stale']==[changed['observation_id']]
    with pytest.raises(ValueError,match='changed'):
        assign_identity(panel_database,changed,'P02')
    clear_identity(panel_database,groups[0])
    assert groups[0]['observation_id'] not in identity_snapshot(panel_database,groups)['assignments']
    assert hashlib.sha256(panel_database.read_bytes()).hexdigest()==before


def test_imported_identity_conflict_rolls_back_whole_registry(panel_database):
    import uuid

    from pvdelta.identity_store import identity_snapshot, import_registry

    uid=str(uuid.uuid4())
    row={'uid':uid,'kind':'table','label':'P01','parent_uid':None,'provenance':'pilot'}
    import_registry(panel_database,[row])
    import_registry(panel_database,[row])
    new={'uid':str(uuid.uuid4()),'kind':'table','label':'P02','parent_uid':None,'provenance':'pilot'}
    with pytest.raises(ValueError,match='different identity'):
        import_registry(panel_database,[new,dict(row,label='WRONG')])
    assert [e['label'] for e in identity_snapshot(panel_database,[])['entities']]==['P01']


def test_same_frame_cannot_assign_two_observations_to_one_panel_identity(panel_database):
    from pvdelta.identity_store import assign_identity

    c=connect_store(panel_database)
    c.execute("UPDATE module_observations SET capture_id='t1' WHERE observation_id='obs2'")
    groups=panel_groups(c,'analysis')
    c.close()
    assign_identity(panel_database,groups[0],'P01','A-01')
    with pytest.raises(ValueError,match='already belongs'):
        assign_identity(panel_database,groups[1],'P01','A-01')
    assert assign_identity(panel_database,groups[1],'P01')[1] is None


def test_identity_browser_opens_grouped_source_and_operator_can_link_table(panel_database):
    from streamlit.testing.v1 import AppTest

    script=('from pvdelta.panel_review_ui import panel_review_view\n'
        f'panel_review_view({str(panel_database)!r},"analysis",None,None,key="identity-test")\n')
    app=AppTest.from_string(script,default_timeout=20).run()
    assert not app.exception
    next(t for t in app.text_input if t.label=='Table ID').set_value('T07')
    submit=next(b for b in app.button if b.label=='Save identity link')
    submit.click().run()
    assert not app.exception
    assert next(m.value for m in app.metric if m.label=='Table ID groups')=='1'
    assert len(app.selectbox(key='identity-test-identity-group').options)==1
    app.button(key='identity-test-identity-open').click().run()
    assert not app.exception and app.selectbox(key='identity-test-card').value=='obs1'


def test_bulk_table_assignment_rolls_back_and_preserves_existing_panel_links(panel_database):
    from pvdelta.identity_store import (
        assign_identity,
        assign_table_batch,
        clear_identity,
        identity_groups,
        identity_snapshot,
    )
    c=connect_store(panel_database,read_only=True)
    groups=panel_groups(c,'analysis')
    c.close()
    assign_identity(panel_database,groups[0],'P01','A-01')
    with pytest.raises(ValueError,match='existing panel'):
        assign_table_batch(panel_database,[groups[1],groups[0]],'P02')
    snapshot=identity_snapshot(panel_database,groups)
    assert len(snapshot['assignments'])==1
    assert 'P02' not in [e['label'] for e in snapshot['entities']]
    clear_identity(panel_database,groups[0])
    assign_table_batch(panel_database,groups,'P02')
    result=identity_groups(groups,identity_snapshot(panel_database,groups))
    assert len(result)==1 and result[0]['label']=='P02' and len(result[0]['regions'])==3


def test_filter_keeps_all_regions_of_a_matching_panel(panel_database):
    c = connect_store(panel_database, read_only=True)
    groups = panel_groups(c, 'analysis')
    filtered = filter_panel_groups(groups, 'Contains Wait')
    assert len(filtered) == 1 and {r['finding_id'] for r in filtered[0]['regions']} == {'f1', 'f2'}
    assert filtered[0]['regions'][0]['review_status'] == 'ok'
    assert filter_panel_groups(groups, 'Contains unreviewed')[0]['observation_id'] == 'obs2'
    assert filter_panel_groups(groups, 'Mixed decisions') == filtered
    assert not filter_panel_groups(groups, 'All OK')
    c.close()


def test_batch_and_individual_review_preserve_other_panels_and_notes(panel_database):
    c = connect_store(panel_database)
    save_panel_review(c, 'analysis', 'obs1', ['f1', 'f2'], 'ok')
    assert c.execute('SELECT finding_id,review_status,note FROM finding_reviews ORDER BY 1').fetchall() == [
        ('f1', 'ok', 'first note'), ('f2', 'ok', 'second note')]
    save_panel_review(c, 'analysis', 'obs1', ['f1', 'f2'], 'not_ok', target_finding_id='f2', note='corrected')
    groups = panel_groups(c, 'analysis')
    assert groups[0]['review_status'] == 'mixed'
    assert groups[1]['review_status'] == 'unreviewed'
    assert c.execute("SELECT review_status,note FROM finding_reviews WHERE finding_id='f1'").fetchone() == ('ok', 'first note')
    assert c.execute('SELECT count(*) FROM findings').fetchone()[0] == 4
    c.close()


def test_stale_or_cross_panel_batch_is_rejected_without_writes(panel_database):
    c = connect_store(panel_database)
    before = c.execute('SELECT finding_id,review_status,note FROM finding_reviews ORDER BY 1').fetchall()
    for observation, expected in [('obs1', ['f1']), ('obs2', ['f1', 'f2']), ('other-obs', ['f1', 'f2'])]:
        with pytest.raises(ValueError, match='changed'):
            save_panel_review(c, 'analysis', observation, expected, 'ok')
    with pytest.raises(ValueError, match='outside'):
        save_panel_review(c, 'analysis', 'obs1', ['f1', 'f2'], 'ok', target_finding_id='f3')
    assert c.execute('SELECT finding_id,review_status,note FROM finding_reviews ORDER BY 1').fetchall() == before
    c.close()


def test_batch_review_rolls_back_if_any_region_write_fails(panel_database, monkeypatch):
    from pvdelta import store

    c = connect_store(panel_database)
    before = c.execute('SELECT finding_id,review_status,note FROM finding_reviews ORDER BY 1').fetchall()
    real = store.upsert_finding_review
    calls = []

    def fail_second(connection, **kwargs):
        calls.append(kwargs['finding_id'])
        if len(calls) == 2:
            raise RuntimeError('simulated write failure')
        return real(connection, **kwargs)

    monkeypatch.setattr(store, 'upsert_finding_review', fail_second)
    with pytest.raises(RuntimeError):
        save_panel_review(c, 'analysis', 'obs1', ['f1', 'f2'], 'not_ok', note='must roll back')
    assert c.execute('SELECT finding_id,review_status,note FROM finding_reviews ORDER BY 1').fetchall() == before
    c.close()


def test_panel_ui_shows_all_regions_once_and_reviews_individually_or_together(panel_database):
    from streamlit.testing.v1 import AppTest

    script = ('from pvdelta.panel_review_ui import panel_review_view\n'
        f'panel_review_view({str(panel_database)!r}, "analysis", None, None, key="test")\n')
    app = AppTest.from_string(script, default_timeout=20).run()
    assert not app.exception and not app.error
    assert len(app.selectbox(key='test-card').options) == 2
    assert len(app.get('image')) == 3
    assert app.dataframe[0].value['Region'].tolist() == ['A1', 'A2']
    app.radio(key='test-filter').set_value('Contains Wait').run()
    assert app.dataframe[0].value['Region'].tolist() == ['A1', 'A2']  # OK region must stay visible too
    app.selectbox(key='test-scope-obs1').select('f2').run()
    app.button(key='test-mark-not_ok').click().run()
    assert not app.exception and not app.error
    c = connect_store(panel_database, read_only=True)
    assert c.execute("SELECT finding_id,review_status FROM finding_reviews ORDER BY 1").fetchall() == [
        ('f1', 'ok'), ('f2', 'not_ok')]
    c.close()
    app.radio(key='test-filter').set_value('All').run()
    app.selectbox(key='test-scope-obs1').select('all').run()
    app.button(key='test-mark-wait').click().run()
    assert not app.exception
    c = connect_store(panel_database, read_only=True)
    assert c.execute('SELECT DISTINCT review_status FROM finding_reviews').fetchall() == [('wait',)]
    c.close()
    app.button(key='test-card-next').click().run()
    assert not app.exception and app.selectbox(key='test-card').value == 'obs2'
    assert app.dataframe[0].value['Region'].tolist() == ['A1']


def test_evidence_queue_keeps_boundary_candidates_and_mixed_card_siblings(panel_database, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from pvdelta import candidate_triage_ui

    monkeypatch.setattr(candidate_triage_ui, 'support_audit', lambda *a: {
        'f1': {'support': 'interior', 'reason': 'Inside'},
        'f2': {'support': 'boundary', 'reason': 'Edge'},
        'f3': {'support': 'boundary', 'reason': 'Edge'}})
    before = hashlib.sha256(panel_database.read_bytes()).hexdigest()
    app = AppTest.from_string('from pvdelta.panel_review_ui import panel_review_view\n'
        f'panel_review_view({str(panel_database)!r}, "analysis", None, None, key="triage")\n',
        default_timeout=20).run()
    assert not app.exception
    assert app.radio(key='triage-evidence-queue').value == 'Interior candidates'
    assert len(app.selectbox(key='triage-card').options) == 1
    assert app.dataframe[0].value['Region'].tolist() == ['A1', 'A2']
    assert app.dataframe[0].value['Support'].tolist() == ['Interior candidate', 'Boundary review']
    app.radio(key='triage-evidence-queue').set_value('Boundary review').run()
    assert not app.exception and len(app.selectbox(key='triage-card').options) == 2
    app.button(key='triage-card-next').click().run()
    assert app.selectbox(key='triage-card').value == 'obs2'
    app.radio(key='triage-evidence-queue').set_value('Support unavailable').run()
    assert not app.exception and not app.selectbox
    app.radio(key='triage-evidence-queue').set_value('All candidates').run()
    assert not app.exception and len(app.selectbox(key='triage-card').options) == 2
    assert hashlib.sha256(panel_database.read_bytes()).hexdigest() == before


def test_temperature_evidence_is_explicit_and_failure_does_not_clear_candidates(panel_database, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from pvdelta import temperature_evidence_store

    calls = []

    def unavailable(database, run, observation, *, sdk_library):
        calls.append((run, observation, sdk_library))
        raise ValueError('Saved region population changed')

    monkeypatch.setattr(temperature_evidence_store, 'check_and_save_temperature_evidence', unavailable)
    before = hashlib.sha256(panel_database.read_bytes()).hexdigest()
    app = AppTest.from_string('from pvdelta.panel_review_ui import panel_review_view\n'
        f'panel_review_view({str(panel_database)!r}, "analysis", "sdk", None, key="ref")\n',
        default_timeout=20).run()
    assert not calls and not app.exception
    app.button(key='ref-temperature-evidence').click().run()
    assert calls == [('analysis', 'obs1', 'sdk')]
    assert any('Saved region population changed' in w.value for w in app.warning)
    assert app.dataframe[0].value['Region'].tolist() == ['A1', 'A2']
    app.button(key='ref-card-next').click().run()
    assert len(calls) == 1 and app.selectbox(key='ref-card').value == 'obs2'
    assert hashlib.sha256(panel_database.read_bytes()).hexdigest() == before


def test_related_candidate_navigation_opens_target_own_regions_and_resets_filters(panel_database, monkeypatch):
    from PIL import Image
    from streamlit.testing.v1 import AppTest

    from pvdelta import candidate_links_ui
    from pvdelta.rgb_pair_check import digest

    root=panel_database.parent
    report={'finding_id':'f1','source_observation_id':'obs1','binding':'fixed',
        'source':{'capture_id':'t1','rgb_path':str(root/'r1.JPG'),'thermal_path':str(root/'t1.JPG')},
        'rows':[{'capture_id':'t2','thermal_path':str(root/'t2.JPG'),'rgb_path':str(root/'r2.JPG'),
            'status':'provisional_candidate_link','reason':'Synthetic passing geometry',
            'target_observation_id':'obs2','target_finding_ids':['f3'],
            'forward':{'check_p95_px':.2},'reverse':{'check_p95_px':.3}}]}
    report['content_sha256']=digest(report)
    calls=[]
    def compute(*a,**k):
        calls.append(a[3]);return {'status':'available','report':report}
    monkeypatch.setattr(candidate_links_ui,'check_and_save_comparison',compute)
    monkeypatch.setattr(candidate_links_ui,'candidate_links_signature',lambda *a,**k:'fixed')
    monkeypatch.setattr(candidate_links_ui,'candidate_link_images',lambda *a:(Image.new('RGB',(64,64)),Image.new('RGB',(64,64))))
    before=hashlib.sha256(panel_database.read_bytes()).hexdigest()
    app=AppTest.from_string('from pvdelta.panel_review_ui import panel_review_view\n'+
        f'panel_review_view({str(panel_database)!r},"analysis",None,{{"model":{{}},"path":"missing"}},key="related")\n',
        default_timeout=30).run()
    assert not calls and not app.exception
    app.radio(key='related-filter').set_value('Contains Wait').run()
    app.button(key='related-find-related').click().run()
    assert calls==['f1'] and not app.exception
    app.button(key='related-open-related-card').click().run()
    assert not app.exception and app.selectbox(key='related-card').value=='obs2'
    assert app.radio(key='related-filter').value=='All'
    assert app.radio(key='related-evidence-queue').value=='All candidates'
    assert app.dataframe[0].value['Region'].tolist()==['A1']
    assert len(calls)==1 and hashlib.sha256(panel_database.read_bytes()).hexdigest()==before


def test_persistent_related_comparison_opens_without_recomputing_and_stays_card_scoped(panel_database,monkeypatch):
    from PIL import Image
    from streamlit.testing.v1 import AppTest

    from pvdelta import candidate_links_ui
    from pvdelta.rgb_pair_check import digest

    root=panel_database.parent
    report={'finding_id':'f1','source_observation_id':'obs1','binding':'current',
        'source':{'capture_id':'t1','rgb_path':str(root/'r1.JPG'),'thermal_path':str(root/'t1.JPG')},
        'rows':[{'capture_id':'t2','thermal_path':str(root/'t2.JPG'),'rgb_path':str(root/'r2.JPG'),
            'status':'provisional_candidate_link','reason':'Synthetic saved geometry',
            'target_observation_id':'obs2','target_finding_ids':['f3'],
            'forward':{'check_p95_px':.2},'reverse':{'check_p95_px':.3}}]}
    report['content_sha256']=digest(report)
    monkeypatch.setattr(candidate_links_ui,'load_candidate_comparison',lambda db,run,oid,*a,**k:
        {'status':'available','report':report,'checked_at_utc':'2026-10-27T00:00:00+00:00'} if oid=='obs1'
        else {'status':'unavailable','reason':'No bounded source anchor'})
    monkeypatch.setattr(candidate_links_ui,'check_and_save_comparison',lambda *a,**k:pytest.fail('Saved review must not recompute'))
    monkeypatch.setattr(candidate_links_ui,'candidate_link_images',lambda *a:(Image.new('RGB',(64,64)),Image.new('RGB',(64,64))))
    before=hashlib.sha256(panel_database.read_bytes()).hexdigest()
    app=AppTest.from_string('from pvdelta.panel_review_ui import panel_review_view\n'+
        f'panel_review_view({str(panel_database)!r},"analysis",None,{{"model":{{}},"path":"missing"}},key="saved")\n',
        default_timeout=30).run()
    assert not app.exception and any('Saved comparison' in c.value for c in app.caption)
    assert not app.button(key='saved-open-related-card').disabled
    app.button(key='saved-open-related-card').click().run()
    assert not app.exception and app.selectbox(key='saved-card').value=='obs2'
    assert any('No bounded source anchor' in i.value for i in app.info)
    assert app.dataframe[0].value['Region'].tolist()==['A1']
    assert hashlib.sha256(panel_database.read_bytes()).hexdigest()==before


def test_basic_v2_default_switches_back_to_unchanged_v1_individual_navigation(panel_database):
    from streamlit.testing.v1 import AppTest

    script = ('from pvdelta.basic_dashboard import basic_dashboard\n'
        f'basic_dashboard({str(panel_database)!r}, "missing-sdk")\n')
    app = AppTest.from_string(script, default_timeout=20).run()
    app.selectbox(key='basic-analysis').select('analysis').run()
    assert not app.exception and not app.error
    assert app.radio(key='basic-review-version').value == 'Panel cards (v2)'
    assert len(app.selectbox(key='basic-panels-v2-card').options) == 2
    app.radio(key='basic-review-version').set_value('Individual anomalies (v1)').run()
    assert not app.exception and not app.error
    assert len(app.selectbox(key='basic-candidate').options) == 3
    app.button(key='basic-candidate-next').click().run()
    assert not app.exception and app.selectbox(key='basic-candidate').value == 'f2'
    app.radio(key='basic-review-version').set_value('Panel cards (v2)').run()
    assert not app.exception and len(app.selectbox(key='basic-panels-v2-card').options) == 2


def test_panel_comparison_is_opt_in_region_scoped_navigable_and_read_only(panel_database, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from pvdelta import repeat_views

    before = hashlib.sha256(panel_database.read_bytes()).hexdigest()
    requests = []

    def signature(database, finding, profile):
        if profile is not None:
            raise FileNotFoundError('No site profile')
        return 'test-manual-'+finding

    def nearby(database, finding):
        requests.append(finding)
        return [{'capture_id': capture, 'status': 'unavailable', 'reason': 'panel identity unverified',
            'thermal_path': str(panel_database.parent/f'{capture}.JPG'),
            'rgb_path': str(panel_database.parent/f'r{capture[1:]}.JPG'), 'time_gap_s': index}
            for index, capture in enumerate(('t1', 't2'))]

    monkeypatch.setattr(repeat_views, 'repeat_evidence_signature', signature)
    monkeypatch.setattr(repeat_views, 'nearby_image_pairs', nearby)
    monkeypatch.setattr(repeat_views, 'build_repeat_evidence',
        lambda *a, **k: pytest.fail('Matching must not run without an explicit request'))
    script = ('from pvdelta.panel_review_ui import panel_review_view\n'
        f'panel_review_view({str(panel_database)!r}, "analysis", "saved-sdk-path", None, key="test")\n')
    app = AppTest.from_string(script, default_timeout=20).run()
    assert not app.exception and len(app.get('image')) == 3 and not requests
    app.checkbox(key='test-compare').check().run()
    assert not app.exception and not requests
    assert app.button(key='test-comparison-find-views').disabled
    app.button(key='test-comparison-nearby-pairs').click().run()
    assert not app.exception and requests == ['f1'] and len(app.get('image')) == 5
    manual = next(x for x in app.selectbox if x.label == 'Nearby thermal frame')
    app.button(key=manual.key+'-next').click().run()
    assert not app.exception and app.selectbox(key=manual.key).value == 't2'
    assert requests == ['f1']  # navigation reuses the same result
    app.selectbox(key='test-comparison-region-obs1').select('f2').run()
    assert not app.exception and len(app.get('image')) == 3  # A1 evidence never leaks into A2
    app.button(key='test-comparison-nearby-pairs').click().run()
    assert not app.exception and requests == ['f1', 'f2'] and len(app.get('image')) == 5
    app.button(key='test-card-next').click().run()
    assert not app.exception and app.selectbox(key='test-card').value == 'obs2'
    assert len(app.get('image')) == 3  # no previous panel evidence
    app.checkbox(key='test-compare').uncheck().run()
    assert not app.exception and len(app.get('image')) == 3
    assert hashlib.sha256(panel_database.read_bytes()).hexdigest() == before


def test_comparison_failure_keeps_primary_panel_regions_visible(panel_database, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from pvdelta import panel_review_ui

    def broken(*args, **kwargs):
        assert kwargs['sdk_library'] == 'saved-sdk-path'
        raise RuntimeError('Comparison source unavailable')

    monkeypatch.setattr(panel_review_ui, 'repeat_view_review', broken)
    script = ('from pvdelta.panel_review_ui import panel_review_view\n'
        f'panel_review_view({str(panel_database)!r}, "analysis", "saved-sdk-path", None, key="test")\n')
    app = AppTest.from_string(script, default_timeout=20).run()
    app.checkbox(key='test-compare').check().run()
    assert not app.exception and not app.error
    assert len(app.get('image')) == 3
    assert app.dataframe[0].value['Region'].tolist() == ['A1', 'A2']
    assert any('Comparison source unavailable' in item.value for item in app.warning)


def test_native_result_is_explicit_and_cleared_on_region_or_binding_change(panel_database, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from pvdelta import native_repeat

    version = ['one']
    calls = []
    def signature(db, finding, sdk):
        return finding+'-'+version[0]
    def check(db, finding, **kwargs):
        calls.append((finding, kwargs['sdk_library']))
        return {'binding': signature(db, finding, None), 'source_file': 'source_T.JPG', 'rows': []}
    monkeypatch.setattr(native_repeat, 'native_repeat_signature', signature)
    monkeypatch.setattr(native_repeat, 'native_repeat_checks', check)
    script = ('from pvdelta.panel_review_ui import panel_review_view\n'
        f'panel_review_view({str(panel_database)!r}, "analysis", "exact-sdk", None, key="test")\n')
    app = AppTest.from_string(script, default_timeout=20).run()
    app.checkbox(key='test-compare').check().run()
    assert not calls  # checkbox/expander alone cannot decode the flight
    app.button(key='test-native-check').click().run()
    assert not app.exception and calls == [('f1', 'exact-sdk')]
    assert 'test-native-result' in app.session_state
    app.selectbox(key='test-comparison-region-obs1').select('f2').run()
    assert not app.exception and 'test-native-result' not in app.session_state
    assert len(app.get('image')) == 3
    app.button(key='test-native-check').click().run()
    assert calls[-1] == ('f2', 'exact-sdk')
    version[0] = 'two'
    app.run()
    assert not app.exception and 'test-native-result' not in app.session_state
    assert len(calls) == 2


def test_native_signature_tracks_geometry_and_rejects_changed_source_even_if_stat_preserved(panel_database):
    import os

    from pvdelta.native_repeat import native_repeat_signature

    c = connect_store(panel_database)
    c.execute('CREATE TABLE flight_screening_frames (analysis_run_id VARCHAR,capture_id VARCHAR,diagnostics_json JSON)')
    c.execute("INSERT INTO flight_screening_frames VALUES ('analysis','t1',?)", [json.dumps({'native_geometry': {}})])
    c.close()
    first = native_repeat_signature(panel_database, 'f1', None)
    c = connect_store(panel_database)
    c.execute("UPDATE flight_screening_frames SET diagnostics_json=?", [json.dumps({'native_geometry': {'changed': True}})])
    c.close()
    assert native_repeat_signature(panel_database, 'f1', None) != first
    path = panel_database.parent/'t1.JPG'
    stat = path.stat()
    content = path.read_bytes()
    path.write_bytes(content[:-1]+bytes([content[-1]^1]))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match='differs from indexed'):
        native_repeat_signature(panel_database, 'f1', None)


def test_v2_gps_browser_shows_return_track_pair_without_sdk_or_site_profile(panel_database, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from pvdelta import repeat_candidates

    requests = []
    def pairs(database, finding):
        requests.append(finding)
        return [{'capture_id': capture, 'thermal_path': str(panel_database.parent/f'{capture}.JPG'),
            'rgb_path': str(panel_database.parent/f'r{capture[1:]}.JPG'),
            'search_method': 'camera_gps', 'camera_distance_m': index*2.5, 'time_gap_s': index*40.,
            'gimbal_yaw_deg': -90+180*index, 'gimbal_pitch_deg': -90}
            for index, capture in enumerate(('t1', 't2'))]
    monkeypatch.setattr(repeat_candidates, 'select_repeat_pairs', pairs)
    before = hashlib.sha256(panel_database.read_bytes()).hexdigest()
    script = ('from pvdelta.panel_review_ui import panel_review_view\n'
        f'panel_review_view({str(panel_database)!r}, "analysis", "missing-sdk", None, key="test")\n')
    app = AppTest.from_string(script, default_timeout=20).run()
    app.checkbox(key='test-compare').check().run()
    assert not requests  # the gallery is lazy
    app.button(key='test-gps-browse').click().run()
    assert not app.exception and not app.error and len(app.get('image')) == 5
    app.button(key='test-gps-frame-f1-next').click().run()
    assert not app.exception and app.selectbox(key='test-gps-frame-f1').value == 't2'
    assert any('Time gap: 40.0 s' in x.value for x in app.caption)
    assert any('Yaw change alone' in x.value for x in app.caption)
    app.selectbox(key='test-comparison-region-obs1').select('f2').run()
    assert not app.exception and app.selectbox(key='test-gps-frame-f2').value == 't1'
    assert requests[-1] == 'f2'
    assert hashlib.sha256(panel_database.read_bytes()).hexdigest() == before
