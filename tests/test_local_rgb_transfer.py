import hashlib
from copy import deepcopy

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
from pvdelta.local_rgb_transfer import (
    _context,
    fit_local_rgb_registration,
    project_local_native_to_rgb,
    propose_observed_junction_transfer,
    transfer_local_rgb_boundaries,
)
from pvdelta.models import ThermalFrame


@pytest.fixture
def case(tmp_path):
    source=tmp_path/'original_T.bin';source.write_bytes(b'original thermal bytes')
    rgb=tmp_path/'original_V.png'
    rgb.write_bytes(cv2.imencode('.png',np.full((440,520,3),120,np.uint8))[1].tobytes())
    values=np.linspace(20,35,220*260,dtype=np.float32).reshape(220,260)
    frame=ThermalFrame('frame',str(source),hashlib.sha256(source.read_bytes()).hexdigest(),values,None,260,220)
    controls={'schema':'pvdelta.local-rgb-controls/v1','pair':_context(frame,rgb),
        'provenance':{'kind':'synthetic_known_transform','note':'Synthetic physical correspondence is known only in this unit test'},
        'points':[{'control_id':f'C{i}-{j}','native_xy':[x,y],'rgb_xy':[2*x+12,2*y+18],
                   'role':'fit' if (i+j)%2==0 else 'check'} for i,x in enumerate((30,70,110,150,190))
                  for j,y in enumerate((30,80,130,180))]}
    q=[[90,80],[130,80],[130,120],[90,120]]
    proposals=[{'proposal_id':'A','polygon_rgb_xy':(np.asarray(q)*2+[12,18]).tolist(),'GPS':'must not transfer'}]
    return frame,rgb,controls,proposals,q


def test_explicit_development_tolerance_keeps_default_and_all_controls(case):
    frame,rgb,controls,proposals,_=case
    for point in controls['points']:
        if point['role']=='check':
            point['rgb_xy'][0]+=4.2  # Known2.1 native-pixel check displacement.
    frozen=deepcopy(controls)
    values=frame.temperature_c.copy()
    default=fit_local_rgb_registration(frame,rgb,controls)
    assert not default['geometry_gate_passed'] and default['native_to_rgb'] is None
    declared=fit_local_rgb_registration(frame,rgb,controls,max_check_p95_px=2.20,
        tolerance_note='Explicit user-authorized development geometry tolerance')
    assert declared['geometry_gate_passed']
    assert 2. < declared['validation']['check_p95_source_px'] < 2.20
    assert default['validation']['check_p95_source_px']==declared['validation']['check_p95_source_px']
    assert default['fit_control_ids']==declared['fit_control_ids']
    assert default['check_control_ids']==declared['check_control_ids']
    assert declared['point_gate_policy']['max_check_p95_native_px']==2.20
    assert default['point_gate_policy']['max_check_p95_native_px']==2.
    transfer=transfer_local_rgb_boundaries(frame,rgb,declared,proposals)
    assert transfer['supported_proposals']==1
    assert transfer['registration_id']==declared['registration_id']
    assert not transfer['measurement_authorised'] and not transfer['main_scan_updated']
    assert controls==frozen
    np.testing.assert_array_equal(frame.temperature_c,values)
    # The reason/threshold is bound to the registration identity, not mutable metadata.
    altered=deepcopy(declared)
    altered['point_gate_policy']['max_check_p95_native_px']=2.
    with pytest.raises(ValueError,match='identity mismatch'):
        transfer_local_rgb_boundaries(frame,rgb,altered,proposals)


@pytest.mark.parametrize('tolerance',[True,0,-1,2.21,float('nan'),float('inf'),'2.2'])
def test_local_point_tolerance_cannot_be_unbounded_or_invalid(case,tolerance):
    frame,rgb,controls,_,_=case
    with pytest.raises(ValueError,match='tolerance'):
        fit_local_rgb_registration(frame,rgb,controls,max_check_p95_px=tolerance,
            tolerance_note='Development trial')


@pytest.mark.parametrize('note',[None,'', '   ',True])
def test_relaxed_development_gate_requires_explicit_reason(case,note):
    frame,rgb,controls,_,_=case
    with pytest.raises(ValueError,match='note'):
        fit_local_rgb_registration(frame,rgb,controls,max_check_p95_px=2.20,tolerance_note=note)


def test_stricter_tolerance_is_recorded_without_claiming_default_gate(case):
    frame,rgb,controls,_,_=case
    registration=fit_local_rgb_registration(frame,rgb,controls,max_check_p95_px=1.5)
    assert registration['geometry_gate_passed']
    policy=registration['point_gate_policy']
    assert policy['max_check_p95_native_px']==1.5
    assert policy['note']=='Explicit stricter local source geometry gate'


def test_observed_junction_pipeline_is_source_bound_and_isolates_tables(case, monkeypatch):
    import pvdelta.junction_registration as junction

    frame, rgb, controls, proposals, truth = case
    nodes = [{'native_xy': p['native_xy'], 'rgb_xy': p['rgb_xy']} for p in controls['points']]
    def observed(*args, config):
        assert config.module_side_only and config.measurement_control_split
        return {'image_consistent': True, 'nodes': nodes,
            'train_node_indices': [i for i,p in enumerate(controls['points']) if p['role']=='fit'],
            'check_node_indices': [i for i,p in enumerate(controls['points']) if p['role']=='check']}
    monkeypatch.setattr(junction, 'match_table_junctions', observed)
    requested = [dict(proposals[0], table_index=1),
        dict(proposals[0], proposal_id='other', table_index=2),
        dict(proposals[0], proposal_id='orphan', table_index=None)]
    matches = [{'rgb_table':1,'rgb_polygon':proposals[0]['polygon_rgb_xy'],'native_polygon':truth}]
    before = frame.temperature_c.copy()
    result = propose_observed_junction_transfer(frame, rgb, matches, requested)
    assert result['supported_proposals'] == 1
    assert result['proposals'][0]['proposal_id'] == 'A'
    decisions = {d['proposal_id']:d['status'] for d in result['decisions']}
    assert decisions['other'] == 'no_bounded_same_table_matches'
    assert decisions['orphan'] == 'explicit_table_unavailable'
    assert len(decisions) == 3
    assert not result['measurement_authorised'] and not result['database_written']
    np.testing.assert_array_equal(frame.temperature_c, before)


def test_failed_junction_pipeline_keeps_requested_regions_without_reusing_a_model(case, monkeypatch):
    import pvdelta.junction_registration as junction

    frame, rgb, _, proposals, truth = case
    monkeypatch.setattr(junction, 'match_table_junctions', lambda *a, **k: {'image_consistent':False})
    result = propose_observed_junction_transfer(frame, rgb,
        [{'rgb_table':1,'rgb_polygon':proposals[0]['polygon_rgb_xy'],'native_polygon':truth}],
        [dict(proposals[0],table_index=1)])
    assert result['proposals'] == []
    assert result['decisions'] == [{'proposal_id':'A','status':'junction_registration_failed'}]


def test_observed_pipeline_cannot_hide_duplicate_requests(case):
    frame, rgb, _, proposals, _ = case
    with pytest.raises(ValueError, match='Unique'):
        propose_observed_junction_transfer(frame,rgb,[],proposals*2)


def test_observed_pipeline_carries_same_explicit_policy_through_both_gates(case,monkeypatch):
    import pvdelta.junction_registration as junction

    frame,rgb,controls,proposals,truth=case
    nodes=[{'native_xy':p['native_xy'],
            'rgb_xy':[p['rgb_xy'][0]+(4.2 if p['role']=='check' else 0),p['rgb_xy'][1]]}
           for p in controls['points']]
    def observed(*args,config):
        assert config.full_module_edge_support and config.max_check_p95_px==2.20
        return {'image_consistent':True,'nodes':nodes,
            'train_node_indices':[i for i,p in enumerate(controls['points']) if p['role']=='fit'],
            'check_node_indices':[i for i,p in enumerate(controls['points']) if p['role']=='check']}
    monkeypatch.setattr(junction,'match_table_junctions',observed)
    result=propose_observed_junction_transfer(frame,rgb,
        [{'rgb_table':1,'rgb_polygon':proposals[0]['polygon_rgb_xy'],'native_polygon':truth}],
        [dict(proposals[0],table_index=1)],full_module_edge_support=True,max_check_p95_px=2.20,
        tolerance_note='User-selected development tolerance')
    assert result['supported_proposals']==1
    assert result['point_gate_policy']==result['groups'][0]['registration']['point_gate_policy']
    assert not result['measurement_authorised'] and not result['database_written']


def test_pair_specific_control_mapping_recovers_original_native_corners_without_warp(case):
    frame,rgb,controls,proposals,truth=case
    old=deepcopy((controls,proposals));native=frame.temperature_c.copy()
    model=fit_local_rgb_registration(frame,rgb,controls)
    assert model['geometry_gate_passed'] and model['validation']['check_p95_source_px']<1e-7
    result=transfer_local_rgb_boundaries(frame,rgb,model,proposals)
    assert result['supported_proposals']==1
    np.testing.assert_allclose(result['proposals'][0]['polygon_native_xy'],truth,atol=1e-7)
    assert 'GPS' not in result['proposals'][0] and not result['proposals'][0]['measurement_authorised']
    assert not any(result[k] for k in ('radiometric_resampling','measurement_authorised','correspondence_verified',
                                       'database_written','main_scan_updated'))
    assert old==(controls,proposals)
    np.testing.assert_array_equal(frame.temperature_c,native)


def test_bad_heldout_points_are_not_fitted_trimmed_or_given_a_usable_transform(case):
    frame,rgb,controls,proposals,_=case
    for p in controls['points']:
        if p['role']=='check':p['rgb_xy'][0]+=12
    model=fit_local_rgb_registration(frame,rgb,controls)
    assert not model['geometry_gate_passed'] and model['native_to_rgb'] is None
    assert model['validation']['check_p95_source_px']>5
    result=transfer_local_rgb_boundaries(frame,rgb,model,proposals)
    assert result['proposals']==[] and result['decisions'][0]['status']=='local_registration_check_failed'


def test_every_corner_must_stay_inside_both_control_hulls_no_extrapolation(case):
    frame,rgb,controls,proposals,_=case
    model=fit_local_rgb_registration(frame,rgb,controls)
    proposals.append({'proposal_id':'outside','polygon_rgb_xy':[[32,38],[92,38],[92,118],[32,118]]})
    result=transfer_local_rgb_boundaries(frame,rgb,model,proposals)
    assert result['requested_proposals']==2 and result['supported_proposals']==1
    assert result['decisions'][1]['status']=='outside_local_fit_or_check_support'


@pytest.mark.parametrize('change',['native','rgb','source','grid','spaces'])
def test_wrong_source_or_coordinate_contract_cannot_reuse_controls(case,change):
    frame,rgb,controls,_proposals,_=case
    if change=='native':frame.temperature_c[0,0]+=1
    elif change=='rgb':rgb.write_bytes(cv2.imencode('.png',np.zeros((440,520,3),np.uint8))[1].tobytes())
    elif change=='source':controls['pair']['source_sha256']='0'*64
    elif change=='grid':controls['pair']['rgb_shape']=[220,260]
    elif change=='spaces':controls['pair']['coordinate_spaces']=['preview','resized_rgb']
    with pytest.raises(ValueError,match='Source-bound'):
        fit_local_rgb_registration(frame,rgb,controls)


def test_model_reuse_on_a_different_native_grid_or_rgb_image_is_rejected(case):
    frame,rgb,controls,proposals,_=case
    model=fit_local_rgb_registration(frame,rgb,controls)
    frame.temperature_c[0,0]+=1
    with pytest.raises(ValueError,match='exact source'):
        transfer_local_rgb_boundaries(frame,rgb,model,proposals)


def test_registration_tamper_is_detected(case):
    frame,rgb,controls,proposals,_=case
    model=fit_local_rgb_registration(frame,rgb,controls)
    model['native_to_rgb'][0][2]+=4
    with pytest.raises(ValueError,match='content identity'):
        transfer_local_rgb_boundaries(frame,rgb,model,proposals)


@pytest.mark.parametrize('change',['duplicate_id','duplicate_xy','too_few_fit','collinear','nan','bool','bad_role'])
def test_invalid_or_shared_control_observations_are_rejected(case,change):
    frame,rgb,controls,_proposals,_=case
    points=controls['points']
    if change=='duplicate_id':points[1]['control_id']=points[0]['control_id']
    elif change=='duplicate_xy':points[1]['native_xy']=points[0]['native_xy'][:]
    elif change=='too_few_fit':
        for p in points:p['role']='check'
    elif change=='collinear':
        for p in points:p['native_xy'][1]=60
    elif change=='nan':points[0]['native_xy'][0]=float('nan')
    elif change=='bool':points[0]['native_xy'][0]=True
    elif change=='bad_role':points[0]['role']='automatic_best_split'
    with pytest.raises(ValueError):fit_local_rgb_registration(frame,rgb,controls)


def test_periodic_wrong_correspondences_can_pass_geometry_but_never_verify_identity(case):
    frame,rgb,controls,proposals,_=case
    for p in controls['points']:p['native_xy'][0]+=20
    controls['provenance']['kind']='independently_reviewed_claim'
    model=fit_local_rgb_registration(frame,rgb,controls)
    result=transfer_local_rgb_boundaries(frame,rgb,model,proposals)
    assert model['geometry_gate_passed'] and result['supported_proposals']==1
    assert not model['correspondence_verified'] and not model['independent_review_verified']
    assert not result['measurement_authorised']


def test_original_rgb_grid_ignores_exif_display_rotation(case):
    from PIL import Image
    frame,rgb,_controls,_proposals,_=case
    rotated=rgb.with_suffix('.jpg')
    image=Image.new('RGB',(520,440));exif=image.getexif();exif[274]=6
    image.save(rotated,exif=exif)
    assert _context(frame,rotated)['rgb_shape']==[440,520]


@pytest.mark.parametrize('change',['duplicate_id','out_of_grid','not_quad','nonconvex'])
def test_malformed_rgb_geometry_fails_explicitly(case,change):
    frame,rgb,controls,proposals,_=case
    model=fit_local_rgb_registration(frame,rgb,controls)
    if change=='duplicate_id':proposals.append(deepcopy(proposals[0]))
    elif change=='out_of_grid':proposals[0]['polygon_rgb_xy'][0][0]=-1
    elif change=='not_quad':proposals[0]['polygon_rgb_xy'].pop()
    elif change=='nonconvex':proposals[0]['polygon_rgb_xy'][2]=[220,190]
    with pytest.raises(ValueError):transfer_local_rgb_boundaries(frame,rgb,model,proposals)


def brown_case(case, *, k1=.18, focal=.65, native_quad=None):
    from pvdelta.table_surface import brown_intrinsics, brown_project
    frame,rgb,controls,proposals,truth=case
    camera={'projection_type':'brown','width':520,'height':440,'focal_x':focal,'focal_y':focal,
        'c_x':.01,'c_y':-.01,'k1':k1,'k2':-.04,'k3':.01,'p1':.0001,'p2':-.0002}
    intrinsic,_=brown_intrinsics(camera)
    def distort(points):
        q=np.asarray(points,float)
        rays=np.column_stack(((q[:,0]-intrinsic[0,2])/intrinsic[0,0],
            (q[:,1]-intrinsic[1,2])/intrinsic[1,1],np.ones(len(q))))
        return brown_project(rays,[0.,0.,0.],[0.,0.,0.],camera)[0].tolist()
    for p in controls['points']:p['rgb_xy']=distort([p['rgb_xy']])[0]
    if native_quad is not None:
        truth=native_quad
        proposals[0]['polygon_rgb_xy']=(np.asarray(truth)*2+[12,18]).tolist()
    proposals[0]['polygon_rgb_xy']=distort(proposals[0]['polygon_rgb_xy'])
    calibration={'schema':'pvdelta.rgb-brown-calibration/v1','rgb_source_sha256':controls['pair']['rgb_sha256'],
        'camera':camera,'provenance':{'note':'Fixed known synthetic camera; coefficients not fitted to check controls'}}
    return frame,rgb,controls,proposals,truth,calibration


def test_fixed_brown_model_recovers_native_geometry_and_projects_original_rgb(case):
    frame,rgb,controls,proposals,truth,calibration=brown_case(case)
    model=fit_local_rgb_registration(frame,rgb,controls,rgb_calibration=calibration)
    assert model['geometry_gate_passed'] and model['validation']['check_p95_source_px']<2e-4
    assert model['native_to_rgb'] is None and model['native_to_undistorted_rgb'] is not None
    transferred=transfer_local_rgb_boundaries(frame,rgb,model,proposals)
    assert transferred['supported_proposals']==1
    q=transferred['proposals'][0]['polygon_native_xy']
    np.testing.assert_allclose(q,truth,atol=2e-4)
    np.testing.assert_allclose(project_local_native_to_rgb(q,model),proposals[0]['polygon_rgb_xy'],atol=2e-4)
    assert not transferred['radiometric_resampling'] and not model['correspondence_verified']


@pytest.mark.parametrize('change',['source','shape','coefficients','provenance'])
def test_wrong_or_invalid_brown_calibration_is_not_silently_applied(case,change):
    frame,rgb,controls,_proposals,_truth,calibration=brown_case(case)
    if change=='source':calibration['rgb_source_sha256']='0'*64
    elif change=='shape':calibration['camera']['width']=521
    elif change=='coefficients':calibration['camera']['k1']=float('nan')
    elif change=='provenance':calibration['provenance']={}
    with pytest.raises(ValueError):fit_local_rgb_registration(frame,rgb,controls,rgb_calibration=calibration)


def test_bad_brown_check_controls_still_withhold_every_transform_and_mask(case):
    frame,rgb,controls,proposals,_truth,calibration=brown_case(case)
    for p in controls['points']:
        if p['role']=='check':p['rgb_xy'][0]+=12
    model=fit_local_rgb_registration(frame,rgb,controls,rgb_calibration=calibration)
    assert not model['geometry_gate_passed']
    assert model['native_to_rgb'] is None and model['native_to_undistorted_rgb'] is None
    assert transfer_local_rgb_boundaries(frame,rgb,model,proposals)['supported_proposals']==0
    assert all(r['role'] in ('fit','check') for r in model['control_residuals'])
    assert any(r['error_native_px']>2 for r in model['control_residuals'] if r['role']=='check')


def test_forward_rendering_rejects_points_outside_support_and_modified_models(case):
    frame,rgb,controls,_proposals,truth=case
    model=fit_local_rgb_registration(frame,rgb,controls)
    with pytest.raises(ValueError,match='outside local'):project_local_native_to_rgb([[0,0]],model)
    model['native_to_rgb'][0][2]+=1
    with pytest.raises(ValueError,match='content identity'):project_local_native_to_rgb(truth,model)


def test_residual_diagnostics_preserve_exact_observations_and_do_not_refit_checks(case):
    frame,rgb,controls,_proposals,_=case
    controls['points'][1]['rgb_xy'][0]+=6
    before=deepcopy(controls)
    model=fit_local_rgb_registration(frame,rgb,controls)
    rows={r['control_id']:r for r in model['control_residuals']}
    for p in controls['points']:
        assert rows[p['control_id']]['observed_rgb_xy']==p['rgb_xy']
        assert rows[p['control_id']]['observed_native_xy']==p['native_xy']
    assert controls==before


def test_nonlinear_edges_cannot_be_silently_replaced_by_straight_quad_masks(case):
    frame,rgb,controls,proposals,_truth,calibration=brown_case(case,k1=.8,focal=.9,
        native_quad=[[50,70],[170,70],[170,140],[50,140]])
    model=fit_local_rgb_registration(frame,rgb,controls,rgb_calibration=calibration)
    assert model['geometry_gate_passed']
    result=transfer_local_rgb_boundaries(frame,rgb,model,proposals)
    assert result['supported_proposals']==0
    assert result['decisions'][0]['status']=='nonlinear_edge_exceeds_quad_tolerance'


def test_opencv_4_iterative_signature_compatibility_on_current_runtime(case,monkeypatch):
    original=cv2.undistortPoints
    existing=getattr(cv2,'undistortPointsIter',None)
    calls=[]
    def legacy(points,intrinsic,distortion,rotation,projection,criteria):
        calls.append(True)
        if existing is not None:
            return existing(points,intrinsic,distortion,rotation,projection,criteria)
        return original(points,intrinsic,distortion,R=rotation,P=projection,criteria=criteria)
    monkeypatch.setattr(cv2,'undistortPointsIter',legacy,raising=False)
    frame,rgb,controls,_proposals,_truth,calibration=brown_case(case)
    assert fit_local_rgb_registration(frame,rgb,controls,rgb_calibration=calibration)['geometry_gate_passed']
    assert calls
