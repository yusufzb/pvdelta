import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from pvdelta.decode import MeasurementParams
from pvdelta.detect import assess_inter_module_anomalies, assess_intra_module_anomalies
from pvdelta.geometry_revision import analyze_geometry_version, revision_engine_sources
from pvdelta.measurement_profile import frame_measurement_profile
from pvdelta.models import ModuleObservation, ThermalFrame
from pvdelta.native_flight import analyze_native_frame
from pvdelta.pipeline import analyze_annotations
from pvdelta.regions import ModuleAnnotation, polygon_to_mask
from pvdelta.surface_constraints import apply_configured_surface


@pytest.fixture
def configured(tmp_path,monkeypatch):
    monkeypatch.delenv('PVDELTA_SURFACE_MANIFEST',raising=False)
    source=tmp_path/'T.bin';source.write_bytes(b'original')
    values=np.full((96,192),25.,np.float32)
    for x,t in ((20,35.),(80,60.),(158,45.)):values[20:28,x:x+8]=t
    params=asdict(MeasurementParams(5.,50.,.95,20.,20.))
    frame=ThermalFrame('f',str(source),hashlib.sha256(source.read_bytes()).hexdigest(),values,None,192,96,
        raw_metadata={'measurement_params':params,'recorded_measurement_params':params,'measurement_overrides':{}})
    geometry={'proposals':[{'proposal_id':name,'polygon_native_xy':[[x,5],[x+50,5],[x+50,85],[x,85]]}
                          for name,x in [('panel',5),('background',65),('unknown',125)]]}
    ref={'case_id':'C001','source_file':source.name,'source_sha256':frame.source_hash,'review_complete':True,
        'native_temperature_sha256':hashlib.sha256(values.tobytes()).hexdigest(),'native_height':96,'native_width':192,
        'panel_polygons':[{'polygon_native_xy':[[0,0],[60,0],[60,95],[0,95]]}],
        'non_panel_polygons':[{'polygon_native_xy':[[60,0],[120,0],[120,95],[60,95]]}]}
    item={'case_id':'C001','surface_reference':ref,'source_path':str(source),
        'measurement_profile':frame_measurement_profile(frame),'baseline':geometry,
        'baseline_sha256':hashlib.sha256(json.dumps(geometry,sort_keys=True,allow_nan=False).encode()).hexdigest()}
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'schema':'pvdelta.surface-recovery-input/v1',
        'surface_meaning':'panel_surface_and_non_panel','frames':[item]}))
    return frame,geometry,manifest,params


def test_shared_wrapper_preserves_native_sensor_invalidity_and_unknown(configured,monkeypatch):
    frame,_,manifest,_=configured
    valid=np.ones(frame.temperature_c.shape,bool);valid[0,0]=False
    frame=replace(frame,valid_mask=valid)
    monkeypatch.setenv('PVDELTA_SURFACE_MANIFEST',str(manifest))
    protected=apply_configured_surface(frame)
    assert not protected.valid_mask[0,0] and not protected.valid_mask[24,84]
    assert protected.valid_mask[24,24] and protected.valid_mask[24,162]
    assert not protected.valid_mask.flags.writeable
    assert apply_configured_surface(protected) is protected
    assert protected.temperature_c is frame.temperature_c
    assert not frame.raw_metadata.get('pvdelta_surface_constraint')
    assert valid[24,84]


def test_pipeline_native_and_direct_detector_all_observe_configured_veto(configured,monkeypatch):
    frame,geometry,manifest,_=configured
    annotations=[ModuleAnnotation(p['proposal_id'],tuple(map(tuple,p['polygon_native_xy']))) for p in geometry['proposals']]
    before=analyze_annotations(frame,annotations,direction='hot')
    assert before[1].findings
    monkeypatch.setenv('PVDELTA_SURFACE_MANIFEST',str(manifest))
    protected=analyze_annotations(frame,annotations,direction='hot')
    assert protected[0].findings and protected[2].findings
    assert protected[1].assessment_status=='not_assessed' and not protected[1].findings
    native,detail=analyze_native_frame(frame,proposal_builder=lambda *a,**k:geometry)
    assert native[1].assessment_status=='not_assessed'
    assert detail['modules'][1]['interior_pixels']==0 and detail['surface_constraint']['manifest_sha256']
    p=geometry['proposals'][1]
    direct=assess_intra_module_anomalies(frame,ModuleObservation('B','f',p['polygon_native_xy']),
        polygon_to_mask(list(map(tuple,p['polygon_native_xy'])),frame.temperature_c.shape))
    assert direct.reasons and not direct.findings


def test_geometry_remeasurement_binds_policy_before_effective_mask_statistics(configured,monkeypatch):
    frame,geometry,manifest,params=configured
    monkeypatch.setenv('PVDELTA_SURFACE_MANIFEST',str(manifest))
    version=analyze_geometry_version(frame,geometry,measurement_parameters=params,
        provenance={'kind':'AI_developer_draft','actor':'unit test','reference_assisted':True,'note':'Synthetic surface test'})
    assert version['source']['surface_constraint']['manifest_sha256']
    assert version['module_reports'][1]['assessment_status']=='not_assessed'


def test_manifest_revision_invalidates_engine_binding_and_explicit_frame_rebind(configured,monkeypatch):
    frame,_,manifest,_=configured
    monkeypatch.setenv('PVDELTA_SURFACE_MANIFEST',str(manifest))
    protected=apply_configured_surface(frame)
    old=revision_engine_sources()['configured_surface_manifest']
    manifest.write_bytes(manifest.read_bytes()+b'\n')
    assert revision_engine_sources()['configured_surface_manifest']!=old
    with pytest.raises(ValueError,match='decode the original'):
        apply_configured_surface(protected,manifest)


def test_unreferenced_source_is_unchanged_and_native_mismatch_cannot_fall_back(configured,monkeypatch,tmp_path):
    frame,_,manifest,_=configured
    monkeypatch.setenv('PVDELTA_SURFACE_MANIFEST',str(manifest))
    other=tmp_path/'other';other.write_bytes(b'other')
    unknown=replace(frame,source_path=str(other),source_hash=hashlib.sha256(other.read_bytes()).hexdigest())
    assert apply_configured_surface(unknown) is unknown
    values=frame.temperature_c.copy();values[1,1]+=1
    with pytest.raises(ValueError,match='source-bound'):
        apply_configured_surface(replace(frame,temperature_c=values))


def test_direct_peer_comparison_cannot_use_explicit_non_panel_reference(configured,monkeypatch):
    frame,geometry,manifest,_=configured
    values=frame.temperature_c.copy();values[:,65:116]=60.
    frame=replace(frame,temperature_c=values)
    document=json.loads(manifest.read_bytes());item=document['frames'][0]
    item['surface_reference']['native_temperature_sha256']=hashlib.sha256(values.tobytes()).hexdigest()
    item['measurement_profile']=frame_measurement_profile(frame)
    manifest.write_text(json.dumps(document))
    modules=[ModuleObservation(p['proposal_id'],'f',p['polygon_native_xy']) for p in geometry['proposals']]
    masks={m.module_id:polygon_to_mask(list(map(tuple,m.pixel_polygon)),values.shape) for m in modules}
    peers={m.module_id:'group' for m in modules}
    before=assess_inter_module_anomalies(frame,modules,masks,peers)
    assert any(f.module_id=='background' for f in before.findings)
    monkeypatch.setenv('PVDELTA_SURFACE_MANIFEST',str(manifest))
    after=assess_inter_module_anomalies(frame,modules,masks,peers)
    assert not any(f.module_id=='background' for f in after.findings)


@pytest.mark.parametrize('change', ['polygon', 'completion', 'grid'])
def test_attached_surface_edit_cannot_reuse_stale_measurement_mask(configured, change):
    frame, _, manifest, _ = configured
    protected = apply_configured_surface(frame, manifest)
    reference = protected.raw_metadata['pvdelta_surface_reference']
    if change == 'polygon':
        reference['non_panel_polygons'][0]['polygon_native_xy'][0][0] = 65
    elif change == 'completion':
        reference['review_complete'] = False
    else:
        reference['native_width'] -= 1
    with pytest.raises(ValueError, match='decode the original'):
        apply_configured_surface(protected)
