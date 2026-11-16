import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from pvdelta import surface_recovery as recovery
from pvdelta.models import ThermalFrame


@pytest.fixture
def manifest(tmp_path):
    source = tmp_path/'thermal.JPG'
    source.write_bytes(b'original test source')
    values = (25+np.arange(160*340, dtype=np.float32).reshape(160,340)/10000)
    params = {'distance_m':5., 'humidity_pct':50., 'emissivity':.95,
              'ambient_temp_c':20., 'reflected_temp_c':20.}
    profile = {'schema':'pvdelta.measurement-profile/v1', 'source_sha256':recovery._sha(source.read_bytes()),
        'native_shape':[160,340], 'native_temperature_sha256':recovery._sha(values.tobytes()),
        'recorded_parameters':params, 'effective_parameters':params, 'requested_overrides':{}}
    canonical = json.dumps(profile,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    profile['profile_id'] = 'sha256:'+hashlib.sha256(canonical).hexdigest()
    ref = {'case_id':'C001','source_sha256':profile['source_sha256'],
        'native_temperature_sha256':profile['native_temperature_sha256'],
        'native_height':160,'native_width':340,'review_complete':True,
        'panel_polygons':[{'polygon_native_xy':[[10,10],[320,10],[320,140],[10,140]]}],
        'non_panel_polygons':[]}
    baseline = {'proposals':[]}
    item = {'case_id':'C001','source_path':str(source),'surface_reference':ref,
        'measurement_profile':profile,'baseline':baseline,'baseline_sha256':recovery._sha(recovery._bytes(baseline))}
    document = {'schema':'pvdelta.surface-recovery-input/v1',
        'surface_meaning':'panel_surface_and_non_panel','frames':[item]}
    path = tmp_path/'manifest.json'
    path.write_bytes(recovery._bytes(document))
    return path,document,values


@pytest.mark.parametrize('change',['duplicate','incomplete','baseline','source','meaning','unsafe_id'])
def test_rejects_ineligible_or_mismatched_surface_configuration(manifest,change):
    path,doc,_ = manifest
    item = doc['frames'][0]
    if change=='duplicate':doc['frames'].append(deepcopy(item))
    elif change=='incomplete':item['surface_reference']['review_complete']=False
    elif change=='baseline':item['baseline']['extra']='changed'
    elif change=='source':item['surface_reference']['source_sha256']='b'*64
    elif change=='meaning':doc['surface_meaning']='hotspots'
    else:item['case_id']='../other'
    path.write_bytes(recovery._bytes(doc))
    with pytest.raises(ValueError):recovery.load_surface_manifest(path)


def test_surface_conflict_and_unknown_pixels_are_excluded(manifest):
    _,doc,values = manifest
    ref = doc['frames'][0]['surface_reference']
    ref['non_panel_polygons']=[{'polygon_native_xy':[[30,30],[50,30],[50,50],[30,50]]}]
    mask,windows = recovery._surface_inputs(ref,values.shape)
    assert mask[20,20] and not mask[40,40] and not mask[0,0]
    assert windows==[[0,0,340,160]]


def test_explicit_surface_policy_preserves_geometry_and_marks_conflict_unassessed(manifest,monkeypatch):
    _,doc,values=manifest
    item=doc['frames'][0]
    item['baseline']['proposals']=[{'proposal_id':'P','polygon_native_xy':[[20,20],[100,20],[100,100],[20,100]]}]
    item['surface_reference']['non_panel_polygons']=[{'polygon_native_xy':[[20,20],[100,20],[100,100],[20,100]]}]
    monkeypatch.setattr(recovery,'decode_frame',lambda path,sdk,**kwargs:
        ThermalFrame('source',str(path),item['surface_reference']['source_sha256'],values,None,340,160))
    monkeypatch.setattr(recovery,'complete_observed_surface_gaps',lambda temperature,baseline,**kwargs:baseline)
    saved=deepcopy(item['baseline'])
    record,_,_=recovery.recover_surface_frame(item,'sdk',exclude_declared_non_panel=True)
    assert record['geometry']==saved and record['old_geometry_preserved']
    assert not record['old_measurements_preserved'] and not record['database_written']
    assert record['declared_surface_policy']['withheld_observations']==1
    assert record['module_reports'][0]['assessment_status']=='not_assessed'


def test_real_analysis_rejects_changed_source_before_decode(manifest,monkeypatch):
    _,doc,_ = manifest
    item = doc['frames'][0]
    from pathlib import Path
    Path(item['source_path']).write_bytes(b'changed')
    monkeypatch.setattr(recovery,'decode_frame',lambda *a,**k:pytest.fail('must not decode changed source'))
    with pytest.raises(ValueError,match='source changed'):recovery.recover_surface_frame(item,'sdk')


def test_actual_recovery_uses_saved_profile_and_checks_native_hash(manifest,monkeypatch):
    _,doc,values = manifest
    item = doc['frames'][0]
    def decode(path,sdk,*,measurement_overrides):
        assert measurement_overrides==item['measurement_profile']['effective_parameters']
        return ThermalFrame('source',str(path),item['surface_reference']['source_sha256'],values,None,340,160)
    monkeypatch.setattr(recovery,'decode_frame',decode)
    record,before,after = recovery.recover_surface_frame(item,'sdk')
    assert record['old_measurements_preserved'] and not record['database_written']
    assert before.size==after.size==(340,160)
    values[40,40]+=1
    with pytest.raises(ValueError,match='radiometry differs'):recovery.recover_surface_frame(item,'sdk')


def test_runs_are_persistent_separate_and_artifacts_verified(manifest,tmp_path,monkeypatch):
    path,_doc,_ = manifest
    def compute(item,sdk):
        return {'case_id':item['case_id'],'source_file':'thermal.JPG','baseline_count':2,
            'observation_count':3,'added_observations':1,'candidate_regions':0},Image.new('RGB',(8,8)),Image.new('RGB',(8,8))
    monkeypatch.setattr(recovery,'recover_surface_frame',compute)
    root=tmp_path/'runs';progress=[]
    result,first=recovery.run_surface_recovery(path,'sdk',root,progress=lambda d,n:progress.append((d,n)))
    assert progress==[(1,1)] and result['completed_frames']==1 and result['failed_frames']==0
    assert recovery.load_surface_run(root,path)==(result,first)
    _,second=recovery.run_surface_recovery(path,'sdk',root)
    assert first!=second and (first/'summary.json').is_file()
    (second/'C001-after.png').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='artifact hash'):recovery.load_surface_run(root,path)


def test_failed_frames_stay_failed_not_empty_success(manifest,tmp_path,monkeypatch):
    path,_,_=manifest
    def fail(*a):raise ValueError('native grid mismatch')
    monkeypatch.setattr(recovery,'recover_surface_frame',fail)
    result,_=recovery.run_surface_recovery(path,'sdk',tmp_path/'runs')
    assert result['failed_frames']==1 and result['completed_frames']==0
    assert 'observation_count' not in result['frames'][0]
    assert result['frames'][0]['error']=='native grid mismatch'


def test_configuration_change_during_run_prevents_latest_activation(manifest,tmp_path,monkeypatch):
    path,_,_=manifest
    def fail(*a):
        path.write_bytes(b'changed during scan')
        raise ValueError('changed')
    monkeypatch.setattr(recovery,'recover_surface_frame',fail)
    root=tmp_path/'runs'
    with pytest.raises(ValueError,match='changed during scan'):
        recovery.run_surface_recovery(path,'sdk',root)
    assert not (root/'latest.json').exists()


def test_failed_retry_preserves_successful_latest_and_can_be_opened(manifest,tmp_path,monkeypatch):
    path,_,_=manifest
    def compute(item,sdk):
        return {'case_id':item['case_id'],'source_file':'thermal.JPG','baseline_count':2,
            'observation_count':3,'added_observations':1,'candidate_regions':0},Image.new('RGB',(8,8)),Image.new('RGB',(8,8))
    monkeypatch.setattr(recovery,'recover_surface_frame',compute)
    root=tmp_path/'runs'
    success,first=recovery.run_surface_recovery(path,'sdk',root)
    original=(root/'latest.json').read_bytes()
    def fail(*a):raise ValueError('SDK unavailable')
    monkeypatch.setattr(recovery,'recover_surface_frame',fail)
    failed,second=recovery.run_surface_recovery(path,'sdk',root)
    assert (root/'latest.json').read_bytes()==original
    assert recovery.load_surface_run(root,path)==(success,first)
    assert recovery.load_surface_run(root,path,run_id=second.name)==(failed,second)
    assert set(recovery.available_surface_runs(root))=={first.name,second.name}


def test_first_failed_run_is_available_without_becoming_latest(manifest,tmp_path,monkeypatch):
    path,_,_=manifest
    def fail(*a):raise ValueError('SDK unavailable')
    monkeypatch.setattr(recovery,'recover_surface_frame',fail)
    root=tmp_path/'runs'
    result,folder=recovery.run_surface_recovery(path,'sdk',root)
    assert not (root/'latest.json').exists()
    assert recovery.available_surface_runs(root)==[folder.name]
    assert recovery.load_surface_run(root,path,run_id=folder.name)==(result,folder)


def test_legacy_latest_remains_accessible_after_successful_rerun(manifest,tmp_path,monkeypatch):
    path,_,_=manifest
    def compute(item,sdk):
        return {'case_id':item['case_id'],'source_file':'thermal.JPG','baseline_count':2,
            'observation_count':3,'added_observations':1,'candidate_regions':0},Image.new('RGB',(8,8)),Image.new('RGB',(8,8))
    monkeypatch.setattr(recovery,'recover_surface_frame',compute)
    root=tmp_path/'runs'
    original,first=recovery.run_surface_recovery(path,'sdk',root)
    (root/'history'/f'{first.name}.json').unlink()  # Pre-history format only has latest.json.
    _,second=recovery.run_surface_recovery(path,'sdk',root)
    assert recovery.available_surface_runs(root)[0]==second.name
    assert recovery.load_surface_run(root,path,run_id=first.name)==(original,first)
    (first/'summary.json').write_bytes(b'changed')
    with pytest.raises(ValueError,match='summary hash'):
        recovery.load_surface_run(root,path,run_id=first.name)


@pytest.mark.parametrize('run_id',['../other','run-../../outside','C001'])
def test_history_selection_rejects_unsafe_run_ids(manifest,tmp_path,run_id):
    with pytest.raises(ValueError,match='run ID'):
        recovery.load_surface_run(tmp_path,manifest[0],run_id=run_id)


def test_partial_retry_is_saved_but_does_not_replace_complete_run(manifest,tmp_path,monkeypatch):
    path,doc,_=manifest
    doc['frames'].append(dict(doc['frames'][0],case_id='C002'))
    digest=recovery._sha(path.read_bytes())
    monkeypatch.setattr(recovery,'load_surface_manifest',lambda p:(doc,digest))
    fail_second=False
    def compute(item,sdk):
        if fail_second and item['case_id']=='C002':raise ValueError('decode failed')
        return {'case_id':item['case_id'],'source_file':'thermal.JPG','baseline_count':2,
            'observation_count':3,'added_observations':1,'candidate_regions':0},Image.new('RGB',(8,8)),Image.new('RGB',(8,8))
    monkeypatch.setattr(recovery,'recover_surface_frame',compute)
    root=tmp_path/'runs'
    complete,first=recovery.run_surface_recovery(path,'sdk',root)
    fail_second=True
    partial,second=recovery.run_surface_recovery(path,'sdk',root)
    assert partial['completed_frames']==partial['failed_frames']==1
    assert recovery.load_surface_run(root,path)==(complete,first)
    assert recovery.load_surface_run(root,path,run_id=second.name)==(partial,second)


def test_requested_history_cannot_redirect_to_another_run(manifest,tmp_path):
    root=tmp_path/'runs';history=root/'history';history.mkdir(parents=True)
    requested='run-'+'a'*32
    (history/(requested+'.json')).write_text(json.dumps({'run_id':'run-'+'b'*32}))
    with pytest.raises(ValueError,match='no matching history'):
        recovery.load_surface_run(root,manifest[0],run_id=requested)


def test_isolated_mode_is_forwarded_and_bound_to_saved_run(manifest,tmp_path,monkeypatch):
    path,_,_=manifest
    def compute(item,sdk,*,allow_isolated=False):
        assert allow_isolated is True
        return {'case_id':item['case_id'],'source_file':'thermal.JPG','baseline_count':2,
            'observation_count':3,'added_observations':1,'candidate_regions':0},Image.new('RGB',(8,8)),Image.new('RGB',(8,8))
    monkeypatch.setattr(recovery,'recover_surface_frame',compute)
    root=tmp_path/'runs'
    result,directory=recovery.run_surface_recovery(path,'sdk',root,allow_isolated=True)
    assert result['allow_isolated'] is True and result['completed_frames']==1
    assert recovery.load_surface_run(root,path)==(result,directory)
    with pytest.raises(ValueError,match='allow_isolated'):
        recovery.run_surface_recovery(path,'sdk',tmp_path/'invalid',allow_isolated=1)
    assert not (tmp_path/'invalid').exists()
