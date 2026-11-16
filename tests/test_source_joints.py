import hashlib
from copy import deepcopy

import numpy as np
import pytest

cv2=pytest.importorskip('cv2')
from pvdelta.control_review import prepare_control_review, save_control_decision
from pvdelta.local_rgb_transfer import _context
from pvdelta.models import ThermalFrame
from pvdelta.source_joints import evaluate_source_joints, source_joint_binding


@pytest.fixture
def source(tmp_path,monkeypatch):
    thermal=tmp_path/'thermal.bin';thermal.write_bytes(b'synthetic thermal source')
    rgb=tmp_path/'rgb.png';rgb.write_bytes(cv2.imencode('.png',np.full((220,260,3),100,np.uint8))[1].tobytes())
    values=np.linspace(20,30,100*120,dtype=np.float32).reshape(100,120)
    frame=ThermalFrame('source',str(thermal),hashlib.sha256(thermal.read_bytes()).hexdigest(),values,None,120,100)
    points=[{'control_id':f'C{i}-{j}','native_xy':[x,y],'rgb_xy':[2*x+1,2*y+1],
        'role':'fit' if (i+j)%2==0 else 'check'} for i,x in enumerate((10,30,50,70,90))
        for j,y in enumerate((10,30,50,70))]
    points.extend([dict(deepcopy(points[0]),control_id='shared-check',role='check'),
                   dict(deepcopy(points[1]),control_id='duplicate-check')])
    controls={'schema':'pvdelta.local-rgb-controls/v1','pair':_context(frame,rgb),
        'provenance':{'kind':'synthetic','note':'Known transform in unit test'},'points':points}
    class Reader:
        def __init__(self,*a):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def temperature(self):return values.copy()
        def processed_rgb(self):return np.zeros((100,120,3),np.uint8)
    monkeypatch.setattr('pvdelta.decode.RJpeg',Reader)
    pack=prepare_control_review(controls,[{'thermal':thermal,'rgb':rgb}],tmp_path/'pack')
    output=tmp_path/'reviews'
    for p in points:
        save_control_decision(pack,output,p['control_id'],native_xy=p['native_xy'],rgb_xy=p['rgb_xy'],
            status='OK',reviewer='synthetic AI',note='unit test',review_basis='AI_source_visual_review')
    groups=[['C0-0','shared-check'],['C0-1','duplicate-check']]
    aliases={'schema':'pvdelta.source-joint-aliases/v1','binding':source_joint_binding(pack,output,0),
        'groups':groups,'note':'Synthetic source identities declared before fitting'}
    return frame,rgb,pack,output,aliases


def run(source,**kwargs):
    frame,rgb,pack,output,aliases=source
    return evaluate_source_joints(frame,rgb,pack,output,aliases,frame_index=0,**kwargs)


def decide(source,cid,status):
    _,_,pack,output,aliases=source
    p=next(p for p in pack['frames'][0]['controls']['points'] if p['control_id']==cid)
    save_control_decision(pack,output,cid,native_xy=p['native_xy'],rgb_xy=p['rgb_xy'],status=status,
        reviewer='synthetic AI',note='unit test',review_basis='AI_source_visual_review')
    aliases['binding']=source_joint_binding(pack,output,0)


def test_check_joint_never_enters_fit_and_every_raw_check_is_scored(source,monkeypatch):
    from pvdelta import alignment
    original=alignment.validate_homography_checks;counts=[]
    def validate(*a,**k):
        counts.append(len(k['check_source_xy']));return original(*a,**k)
    monkeypatch.setattr(alignment,'validate_homography_checks',validate)
    frozen=deepcopy(source[4]);values=source[0].temperature_c.copy()
    result=run(source)
    assert result['status']=='reviewed_joint_point_gate_passed'
    assert result['fit_joint_count']==9 and result['check_joint_count']==11
    assert len(result['raw_observations'])==22 and counts==[12]
    assert result['registration']['geometry_gate_passed']
    assert next(j for j in result['joints'] if j['original_controls'][0]['control_id']=='C0-0')['role']=='check'
    assert source[4]==frozen and np.array_equal(values,source[0].temperature_c)
    assert not result['correspondence_verified'] and not result['database_written']


def test_wait_blocks_transform_even_when_explicit_diagnostic_numbers_pass(source):
    decide(source,'C0-1','Wait')
    blocked=run(source)
    assert blocked['status']=='source_review_pending' and blocked['canonical_validation'] is None
    diagnostic=run(source,diagnostic=True)
    assert diagnostic['status']=='pending_source_diagnostic_only'
    assert diagnostic['canonical_validation']['accepted']
    assert diagnostic['registration'] is None and not diagnostic['review_complete']


def test_not_ok_is_not_fit_in_diagnostic_mode(source):
    decide(source,'C0-1','Not OK')
    assert run(source,diagnostic=True)['canonical_validation'] is None


def test_failed_numeric_gate_still_reports_every_original_check_residual(source):
    _,_,pack,output,aliases=source
    p=next(p for p in pack['frames'][0]['controls']['points'] if p['control_id']=='C0-3')
    save_control_decision(pack,output,p['control_id'],native_xy=p['native_xy'],
        rgb_xy=[p['rgb_xy'][0]+20,p['rgb_xy'][1]],status='OK',reviewer='synthetic AI',note='Bad synthetic point',
        review_basis='AI_source_visual_review')
    aliases['binding']=source_joint_binding(pack,output,0)
    result=run(source)
    assert result['status']=='point_gate_failed' and result['registration'] is None
    assert len(result['raw_check_residuals'])==12
    assert max(r['error_native_px'] for r in result['raw_check_residuals'])>5


def test_all_raw_check_gate_can_reject_a_passing_unique_joint_fit(source):
    _,_,pack,output,aliases=source
    for cid in ('C0-1','duplicate-check'):
        p=next(p for p in pack['frames'][0]['controls']['points'] if p['control_id']==cid)
        save_control_decision(pack,output,cid,native_xy=p['native_xy'],
            rgb_xy=[p['rgb_xy'][0]+4.8,p['rgb_xy'][1]],status='OK',reviewer='synthetic AI',
            note='Synthetic duplicated check error',review_basis='AI_source_visual_review')
    aliases['binding']=source_joint_binding(pack,output,0)
    result=run(source)
    assert result['canonical_validation']['accepted']
    assert not result['all_raw_check_validation']['accepted']
    assert result['registration'] is None and result['status']=='point_gate_failed'


@pytest.mark.parametrize('groups',[[],[['unknown','C0-0']],[['C0-0','C0-0']],
    [['C0-0','shared-check'],['shared-check','C0-1']],[['C0-0','C0-1']]])
def test_invalid_or_disagreeing_aliases_are_not_merged(source,groups):
    source[4]['groups']=groups
    with pytest.raises(ValueError):run(source)


def test_stale_review_bindings_cannot_be_reused(source):
    old=deepcopy(source[4]['binding'])
    decide(source,'C0-1','Wait');source[4]['binding']=old
    with pytest.raises(ValueError,match='current review'):run(source)


def test_native_measurements_cannot_be_changed_under_source_aliases(source):
    source[0].temperature_c[0,0]+=1
    with pytest.raises(ValueError,match='different source'):run(source)


def test_review_change_during_fitting_discards_result(source,monkeypatch):
    import pvdelta.local_rgb_transfer as transfer
    original=transfer.fit_local_rgb_registration
    def fit(*a,**k):
        result=original(*a,**k);decide(source,'C0-1','Wait');return result
    monkeypatch.setattr(transfer,'fit_local_rgb_registration',fit)
    with pytest.raises(ValueError,match='changed during joint evaluation'):run(source)
