import hashlib
import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from pvdelta.config import DetectionConfig
from pvdelta.geometry_revision import (
    analyze_geometry_version,
    apply_geometry_patch,
    geometry_version_impact,
    load_geometry_version,
    save_geometry_version,
    verify_geometry_version,
)
from pvdelta.models import ThermalFrame


def panel(identifier, x, right=None):
    return {'proposal_id':identifier,'polygon_native_xy':[[x,15],[right or x+35,15],[right or x+35,70],[x,70]],
        'source_method':'closed_contour','threshold_votes':2,'GPS':'stale','native_edge_evidence':'stale'}


@pytest.fixture
def prepared(tmp_path):
    path=tmp_path/'original.bin'
    path.write_bytes(b'original source bytes')
    y,x=np.mgrid[:90,:140]
    values=(25+x*.02+y*.005).astype(np.float32)
    values[35:40,39:44]+=8
    frame=ThermalFrame('frame',str(path),hashlib.sha256(path.read_bytes()).hexdigest(),values,None,140,90)
    geometry={'proposals':[panel('A',10,50),panel('B',45,80)]}
    parameters={'decoder':'synthetic','emissivity':.95}
    provenance={'kind':'automatic','actor':'test fixture'}
    root=analyze_geometry_version(frame,geometry,measurement_parameters=parameters,provenance=provenance)
    return frame,geometry,parameters,root


def patch(root,ops):
    return {'schema':'pvdelta.native-geometry-patch/v1','parent_version_id':root['version_id'],
        'source_sha256':root['source']['source_sha256'],
        'native_temperature_sha256':root['source']['native_temperature_sha256'],'operations':ops}


def child(prepared,ops,**kwargs):
    frame,_geometry,parameters,root=prepared
    plan=patch(root,ops)
    geometry=apply_geometry_patch(root,plan)
    version=analyze_geometry_version(frame,geometry,measurement_parameters=parameters,
        provenance={'kind':'user_edit','actor':'declared reviewer'},parent=root,patch=plan,**kwargs)
    return version,geometry_version_impact(root,version)


def test_removing_a_neighbour_reanalyzes_unchanged_geometry_and_frees_its_native_pixels(prepared):
    frame,geometry,_parameters,root=prepared
    old=deepcopy((geometry,root))
    native=frame.temperature_c.copy()
    revised,impact=child(prepared,[{'op':'remove','proposal_id':'B','reason':'geometry review'}])
    assert impact['removed_ids']==['B'] and impact['geometry_changed_ids']==[]
    assert impact['unchanged_geometry_with_changed_mask_ids']==['A']
    assert impact['reports_changed_ids']==['A']
    assert revised['mask_summaries'][0]['effective_pixels']>root['mask_summaries'][0]['effective_pixels']
    assert (geometry,root)==old
    np.testing.assert_array_equal(frame.temperature_c,native)
    assert not revised['main_scan_updated'] and not revised['measurement_authorised']


def test_replace_clears_old_edge_gps_and_other_derived_metadata(prepared):
    revised,impact=child(prepared,[{'op':'replace','proposal_id':'A','reason':'align outer frame',
        'polygon_native_xy':[[12,16],[48,16],[48,68],[12,68]]}])
    p=revised['geometry']['proposals'][0]
    assert p['proposal_id']=='A' and p['parent_proposal_sha256']
    assert not {'GPS','native_edge_evidence','source_method','threshold_votes'}&p.keys()
    assert impact['geometry_changed_ids']==['A'] and 'A' in impact['effective_masks_changed_ids']
    assert revised['source']==prepared[3]['source']


def test_addition_changes_shared_measurement_ownership_and_never_reuses_reports(prepared):
    revised,impact=child(prepared,[{'op':'add','proposal_id':'C','reason':'visible missing panel',
        'polygon_native_xy':[[80,15],[115,15],[115,70],[80,70]]}])
    assert impact['added_ids']==['C'] and len(revised['module_reports'])==3
    assert impact['all_child_ROIs_reanalyzed']
    assert not revised['independent_review_verified']


def test_empty_frame_version_retains_removed_ids_in_the_parent(prepared):
    revised,impact=child(prepared,[{'op':'remove','proposal_id':i,'reason':'counterfactual empty frame'} for i in ('A','B')])
    assert revised['geometry']['proposals']==revised['mask_summaries']==revised['module_reports']==[]
    assert impact['removed_ids']==['A','B'] and len(prepared[3]['module_reports'])==2


def test_noop_replacement_can_change_provenance_without_changing_masks_or_reports(prepared):
    revised,impact=child(prepared,[{'op':'replace','proposal_id':'A','reason':'same outline reviewed',
        'polygon_native_xy':prepared[3]['geometry']['proposals'][0]['polygon_native_xy']}])
    assert revised['version_id']!=prepared[3]['version_id']
    assert impact['geometry_changed_ids']==impact['effective_masks_changed_ids']==impact['reports_changed_ids']==[]


def test_visibility_and_explicit_peer_metadata_are_analysis_inputs(prepared):
    revised,impact=child(prepared,[{'op':'replace','proposal_id':'A','reason':'partial view',
        'polygon_native_xy':prepared[3]['geometry']['proposals'][0]['polygon_native_xy'],
        'visibility':'partial','peer_group':'table1'}])
    assert impact['annotation_changed_ids']==['A']
    assert revised['module_reports'][0]['module']['visibility']=='partial'
    assert impact['reports_changed_ids']==['A']


@pytest.mark.parametrize('key',['parent_version_id','source_sha256','native_temperature_sha256'])
def test_patch_cannot_apply_to_a_different_parent_or_source(prepared,key):
    plan=patch(prepared[3],[{'op':'remove','proposal_id':'A','reason':'test'}])
    plan[key]='wrong'
    with pytest.raises(ValueError,match='exact parent'):
        apply_geometry_patch(prepared[3],plan)


@pytest.mark.parametrize('ops',[
    [],[{'op':'remove','proposal_id':'missing','reason':'test'}],
    [{'op':'add','proposal_id':'A','reason':'test'}],
    [{'op':'remove','proposal_id':'A','reason':'test'},{'op':'add','proposal_id':'A','reason':'reuse'}],
    [{'op':'remove','proposal_id':'A','reason':''}],
    [{'op':'remove','proposal_id':'A','reason':'test','measurement_authorised':True}],
    [{'op':'remove','proposal_id':'A','reason':'test','polygon_native_xy':[]}],
    [{'op':'replace','proposal_id':'A','reason':'test','polygon_native_xy':[[0,0],[0,0],[0,0],[0,0]]}],
    [{'op':'replace','proposal_id':'A','reason':'test','polygon_native_xy':[[True,1],[20,1],[20,20],[1,20]]}],
    [{'op':'replace','proposal_id':'A','reason':'test','polygon_native_xy':[[-1,1],[20,1],[20,20],[-1,20]]}],
])
def test_bad_or_ambiguous_operations_fail_before_analysis(prepared,ops):
    with pytest.raises(ValueError):
        apply_geometry_patch(prepared[3],patch(prepared[3],ops))


@pytest.mark.parametrize('change',['native','valid_mask','source_file','settings','parameters','engine'])
def test_parent_cannot_be_reused_for_changed_analysis_inputs(prepared,change,monkeypatch):
    frame,_geometry,parameters,root=prepared
    plan=patch(root,[{'op':'remove','proposal_id':'B','reason':'test'}])
    geometry=apply_geometry_patch(root,plan)
    kwargs={}
    if change=='native':
        frame.temperature_c[20,20]+=1
    elif change=='valid_mask':
        frame=replace(frame,valid_mask=np.zeros(frame.temperature_c.shape,bool))
    elif change=='source_file':
        from pathlib import Path
        Path(frame.source_path).write_bytes(b'different source')
    elif change=='settings':
        kwargs['config']=DetectionConfig(delta_threshold_k=3)
    elif change=='parameters':
        parameters=dict(parameters,emissivity=.9)
    else:
        monkeypatch.setattr('pvdelta.geometry_revision.revision_engine_sources',lambda:{'changed':'a'*64})
    with pytest.raises(ValueError,match='mismatch|changed'):
        analyze_geometry_version(frame,geometry,measurement_parameters=parameters,
            provenance={'kind':'user_edit','actor':'test'},parent=root,patch=plan,**kwargs)


def test_mutation_during_reanalysis_is_detected(prepared,monkeypatch):
    from pvdelta.geometry_revision import analyze_annotations
    frame,geometry,parameters,_root=prepared
    def mutate(*args,**kwargs):
        result=analyze_annotations(*args,**kwargs)
        frame.temperature_c[0,0]+=1
        return result
    monkeypatch.setattr('pvdelta.geometry_revision.analyze_annotations',mutate)
    with pytest.raises(RuntimeError,match='changed during'):
        analyze_geometry_version(frame,geometry,measurement_parameters=parameters,
            provenance={'kind':'automatic','actor':'test'})


def test_version_identity_tamper_and_approval_claims_are_rejected(prepared):
    root=prepared[3]
    assert verify_geometry_version(root)==root['version_id']
    broken=deepcopy(root)
    broken['module_reports'][0]['module']['temp_max']=999
    with pytest.raises(ValueError,match='identity'):
        verify_geometry_version(broken)
    broken=deepcopy(root)
    broken['measurement_authorised']=True
    with pytest.raises(ValueError,match='unauthorised'):
        verify_geometry_version(broken)


def test_versions_save_and_reload_without_overwriting_the_parent(prepared,tmp_path):
    parent=prepared[3]
    original=json.dumps(parent,sort_keys=True)
    revised,_=child(prepared,[{'op':'remove','proposal_id':'B','reason':'test'}])
    p=save_geometry_version(parent,tmp_path/'versions')
    before=p.read_bytes()
    c=save_geometry_version(revised,tmp_path/'versions')
    assert p!=c and p.read_bytes()==before
    assert load_geometry_version(c)==revised and json.dumps(parent,sort_keys=True)==original
    assert save_geometry_version(parent,tmp_path/'versions')==p
    p.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='refusing overwrite'):
        save_geometry_version(parent,tmp_path/'versions')


def test_valid_pixel_mask_and_nonfinite_samples_remain_bound_and_unhealthy_is_not_inferred(prepared):
    frame,geometry,parameters,_root=prepared
    frame.temperature_c[20,20]=np.nan
    frame=replace(frame,valid_mask=np.zeros(frame.temperature_c.shape,bool))
    version=analyze_geometry_version(frame,geometry,measurement_parameters=parameters,
        provenance={'kind':'automatic','actor':'test'})
    assert all(r['assessment_status']=='not_assessed' for r in version['module_reports'])
    assert all(s['effective_pixels']==0 for s in version['mask_summaries'])
    assert not version['measurement_authorised']


@pytest.mark.parametrize('key,value',[('native_shape',[90,141]),('native_temperature_sha256','0'*64),
    ('source_sha256','0'*64)])
def test_declared_geometry_from_another_grid_is_not_silently_rebound(prepared,key,value):
    frame,geometry,parameters,_root=prepared
    with pytest.raises(ValueError,match='Declared geometry'):
        analyze_geometry_version(frame,dict(geometry,**{key:value}),measurement_parameters=parameters,
            provenance={'kind':'automatic','actor':'test'})
