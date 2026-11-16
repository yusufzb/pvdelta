import hashlib
import json
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
from test_panel_review import panel_database  # noqa: F401

from pvdelta import thermal_view_scan as scan
from pvdelta.config import DetectionConfig
from pvdelta.repeat_views import PanelMatch
from pvdelta.store import connect_store


@pytest.mark.parametrize('failure', ['missing', 'other_run', 'incomplete', 'corrupt', 'not_object'])
def test_unavailable_native_settings_are_explicit_and_never_guessed(tmp_path, failure):
    summary = {'schema':'pvdelta.native-flight/v1', 'analysis_run_id':'analysis',
               'settings':asdict(DetectionConfig())}
    path = tmp_path/'summary.json'
    if failure == 'other_run':
        summary['analysis_run_id'] = 'different'
    elif failure == 'incomplete':
        del summary['settings']['min_region_px']
    if failure != 'missing':
        path.write_text('broken JSON' if failure == 'corrupt' else
                        json.dumps([] if failure == 'not_object' else summary))
    settings, reason = scan._analysis_settings(tmp_path/'inspection.duckdb', 'analysis')
    assert settings is None and reason
    assert 'No such file' not in reason
    assert path.exists() is (failure != 'missing')


def test_legacy_run_without_summary_retains_nearby_photos_and_real_candidate_counts(request, monkeypatch):
    db = request.getfixturevalue('panel_database')
    assert not db.with_name('summary.json').exists()
    rows = [{'capture_id':cid, 'thermal_path':str(db.parent/f'{cid}.JPG'),
             'rgb_path':str(db.parent/f'r{i}.JPG'), 'thermal_sha256':f'thermal-{i}',
             'rgb_sha256':f'rgb-{i}'} for i,cid in enumerate(('t1','t2'),1)]
    monkeypatch.setattr(scan.rgb_repeat, '_snapshot', lambda *a:(deepcopy(rows),'base'))
    monkeypatch.setattr(scan, '_runtime', lambda *a:{})
    def forbidden(*args):
        pytest.fail('Missing historical profile cannot enter matching or thermal replay')
    monkeypatch.setattr(scan.rgb_repeat, '_verify_source', forbidden)
    monkeypatch.setattr(scan, '_replay', forbidden)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    report = scan.scan_thermal_views(db,'f1',{})
    assert report['status'] == 'unavailable' and 'legacy analysis' in report['reason']
    assert report['retrieved_nearby_views'] == 1 and report['rows'][0]['status'] == 'not_attempted'
    assert report['rows'][0]['capture_candidate_count'] == 1
    assert report['rows'][0]['thermal_evidence'] is None
    assert report['source']['thermal_path'] == rows[0]['thermal_path']
    assert not report['glare_clearance'] and not report['physical_identity_verified']
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_retrieval_retains_candidate_free_capture_and_ignores_other_run(request,monkeypatch):
    db=request.getfixturevalue('panel_database')
    c=connect_store(db);c.execute("DELETE FROM findings WHERE finding_id='f3'");c.close()
    db.with_name('summary.json').write_text(json.dumps({'schema':'pvdelta.native-flight/v1',
        'analysis_run_id':'analysis','settings':asdict(DetectionConfig())}))
    rows=[{'capture_id':'t1'},{'capture_id':'t2'}]
    monkeypatch.setattr(scan.rgb_repeat,'_snapshot',lambda *a: (deepcopy(rows),'base'))
    monkeypatch.setattr(scan,'read_measurement_profile',lambda *a: {'profile_id':'test'})
    monkeypatch.setattr(scan,'read_native_diagnostic',lambda *a: {'diagnostic':{'modules':[]}})
    monkeypatch.setattr(scan,'_runtime',lambda *a: {})
    before=hashlib.sha256(db.read_bytes()).hexdigest()
    selected,run,oid,_settings,binding=scan._context(db,'f1',{},None,32)
    assert [r['capture_id'] for r in selected]==['t1','t2']
    assert selected[1]['observations']['M004']['findings']==[]
    assert selected[0]['observations']['M004']['observation_id']=='obs1'
    assert (run,oid)==('analysis','obs1') and binding
    assert hashlib.sha256(db.read_bytes()).hexdigest()==before


def prepare(monkeypatch,*,source_error=False,reverse_wrong=False,changed=False):
    quad=[[10,10],[30,10],[30,50],[10,50]]
    rows=[{'capture_id':n,'thermal_path':n+'_T.JPG','rgb_path':n+'_V.JPG','thermal_sha256':n,'rgb_sha256':n,
        'input_error':None,'module_id':'M','module_polygon':quad,'gimbal_pitch_deg':-90,
        'observations':{'M':{'observation_id':n+'-obs','polygon':quad,'findings':[]}}} for n in ('source','target')]
    count=[0]
    def context(*a):
        count[0]+=1
        return deepcopy(rows),'run','source-obs',{},'changed' if changed and count[0]>1 else 'binding'
    monkeypatch.setattr(scan,'_context',context)
    monkeypatch.setattr(scan.rgb_repeat,'_verify_source',lambda *a:None)
    monkeypatch.setattr(scan.rgb_repeat,'_observed',lambda *a:{'modules':[{'polygon_rgb_xy':quad}]})
    def anchor(*a,**k):
        if source_error:raise ValueError('Source has no anchor')
        return {'module_id':'M','native_polygon':quad,'rgb_polygon':quad}
    monkeypatch.setattr(scan,'_anchor',anchor)
    calls=[]
    def replay(db,run,row,*a):
        calls.append(row['capture_id'])
        return {'status':'analyzed_no_candidates','regions':[],'healthy_panel':None,'glare_clearance':False}
    monkeypatch.setattr(scan,'_replay',replay)
    from pvdelta import repeat_views
    directions=[]
    def match(*a):
        directions.append(1)
        return PanelMatch('matched_provisional','pass',target_index=1 if reverse_wrong and len(directions)>1 else 0)
    monkeypatch.setattr(repeat_views,'match_panel',match)
    return calls


def test_candidate_free_target_gets_two_way_match_and_own_replay(monkeypatch):
    calls=prepare(monkeypatch)
    report=scan.scan_thermal_views('db','f',{})
    row=report['rows'][0]
    assert calls==['source','target'] and row['capture_candidate_count']==0
    assert row['status']=='matched_provisional' and row['thermal_evidence']['status']=='analyzed_no_candidates'
    assert row['thermal_evidence']['healthy_panel'] is None and not row['glare_clearance']
    assert report['selection_uses_candidate_presence'] is False


def test_source_failure_keeps_unattempted_target_denominator(monkeypatch):
    calls=prepare(monkeypatch,source_error=True)
    report=scan.scan_thermal_views('db','f',{})
    assert not calls and report['status']=='unavailable'
    assert len(report['rows'])==report['retrieved_nearby_views']==1
    assert report['rows'][0]['status']=='not_attempted'


def test_reverse_mismatch_cannot_become_negative_thermal_evidence(monkeypatch):
    calls=prepare(monkeypatch,reverse_wrong=True)
    row=scan.scan_thermal_views('db','f',{})['rows'][0]
    assert calls==['source'] and row['status']=='unavailable' and row['thermal_evidence'] is None


def test_changed_inputs_withhold_completed_report(monkeypatch):
    prepare(monkeypatch,changed=True)
    with pytest.raises(ValueError,match='changed during'):
        scan.scan_thermal_views('db','f',{})


@pytest.mark.parametrize('has_new_hotspot',[False,True])
def test_empty_saved_population_must_reproduce_on_native_temperatures(monkeypatch,has_new_hotspot):
    values=np.full((64,64),25.,np.float32)
    if has_new_hotspot:values[25:30,25:30]=40
    class Reader:
        def __init__(self,*a):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
    class Connection:
        def close(self):pass
    monkeypatch.setattr(scan,'RJpeg',Reader)
    monkeypatch.setattr(scan,'apply_saved_measurement_profile',lambda *a:values)
    monkeypatch.setattr(scan,'connect_store',lambda *a,**k:Connection())
    monkeypatch.setattr(scan,'native_support_masks',lambda *a:{'M':np.ones((64,64),bool)})
    row={'thermal_path':'fake','thermal_sha256':'hash','capture_id':'t','profile':{},
        'observations':{'M':{'assessment_status':'analyzed','observation_id':'obs',
            'polygon':[[0,0],[64,0],[64,64],[0,64]],'findings':[]}}}
    if has_new_hotspot:
        with pytest.raises(ValueError,match='population does not reproduce'):
            scan._replay('db','run',row,'M',asdict(DetectionConfig()),None)
    else:
        result=scan._replay('db','run',row,'M',asdict(DetectionConfig()),None)
        assert result['status']=='analyzed_no_candidates' and result['healthy_panel'] is None
        assert result['minimum_c']==result['median_c']==result['maximum_c']==25.


def test_persistent_scans_reject_changes_and_never_overwrite_candidate_records(tmp_path,monkeypatch):
    prepare(monkeypatch)
    report=scan.scan_thermal_views('db','f',{},max_views=8)
    db=tmp_path/'inspection.duckdb';db.write_bytes(b'original')
    monkeypatch.setattr(scan,'thermal_views_signature',lambda *a,**k:'binding')
    scan.save_view_scan(db,report)
    assert scan.load_view_scan(db,'f',{},max_views=8)['status']=='available'
    assert scan.load_view_scan(db,'f',{},max_views=32)['status']=='missing'
    monkeypatch.setattr(scan,'thermal_views_signature',lambda *a,**k:'changed')
    assert scan.load_view_scan(db,'f',{},max_views=8)['status']=='stale'
    assert db.read_bytes()==b'original'
