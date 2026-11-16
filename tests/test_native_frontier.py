import hashlib
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip('cv2')
from pvdelta.native_completion import complete_native_proposals
from pvdelta.native_frontier import frontier_native_candidates


def panel(x, identifier=None):
    return {'proposal_id':identifier or f'N{x}', 'polygon_native_xy':[[x,45],[x+100,45],[x+100,95],[x,95]],
        'source_method':'closed_contour', 'threshold_votes':2}


def baseline():
    return {'proposals':[panel(180),panel(290)]}


def frame(kind='frames'):
    y,x=np.mgrid[:240,:420]
    if kind=='flat':
        return np.full((240,420),30.,np.float32)
    distance=np.minimum.reduce([abs(x-v) for v in (70,170,180,280,290,390)]+[abs(y-45),abs(y-95)])
    values=(30-4*np.exp(-.5*(distance/.7)**2)).astype(np.float32)
    if kind=='hot_interior':
        values[55:85,90:150]+=4
    return values


def search(values, old=None, **kwargs):
    return frontier_native_candidates(values,baseline() if old is None else old,
        native_temperature_sha256=hashlib.sha256(values.tobytes()).hexdigest(),**kwargs)


@pytest.mark.parametrize('kind',['frames','hot_interior'])
def test_actual_four_ridges_recover_adjacent_geometry_without_closed_contour_input(kind):
    values=frame(kind)
    previous=values.copy()
    old=baseline()
    before=deepcopy(old)
    found=search(values,old)
    expected=np.array(panel(70)['polygon_native_xy'])
    assert any(np.max(np.abs(np.asarray(p['polygon_native_xy'])-expected)) < 1 for p in found['proposals'])
    for p in found['proposals']:
        assert p['native_supported_sides']==[0,1,2,3]
        assert p['segment_supported_sides']==[]
        assert min(p['side_coverage']) >= .8 and max(p['side_max_gap_fraction']) <= .18
        assert p['semantic_panel_membership']=='unverified' and not p['measurement_authorised']
    proposed=complete_native_proposals(old,found,native_shape=values.shape)
    assert len(proposed['proposals'])==3
    assert proposed['proposals'][:2]==before['proposals'] and old==before
    np.testing.assert_array_equal(values,previous)


@pytest.mark.parametrize('kind',['flat','slope','spot','step','three_sides'])
def test_neighbours_alone_and_incomplete_evidence_cannot_fill_a_panel(kind):
    y,x=np.mgrid[:240,:420]
    values=np.full((240,420),30.,np.float32)
    if kind=='slope':
        values+=x*.01+y*.005
    elif kind=='spot':
        values[60:70,110:120]+=10
    elif kind=='step':
        values+=5*(x>70)
    elif kind=='three_sides':
        distance=np.minimum.reduce([abs(x-70),abs(x-170),abs(y-45)])
        values-=4*np.exp(-.5*(distance/.7)**2)
    found=search(values)
    assert found['proposals']==[] and not found['measurement_authorised']


def test_no_supported_seed_means_no_search_or_forced_array():
    for old in ({'proposals':[]},{'proposals':[dict(panel(180),source_method='inferred_grid')]}):
        result=search(frame(),old)
        assert not result['decisions'] and not result['proposals']


def test_empty_ambiguous_baseline_interiors_cannot_seed_a_search():
    p=panel(180)
    old={'proposals':[p,dict(p,proposal_id='duplicate')]}
    result=search(frame(),old)
    assert result['counters']['seeds']==0 and result['proposals']==[]


def test_hypothesis_overflow_withholds_every_fit_instead_of_returning_first_success():
    result=search(frame(),max_hypotheses=1)
    assert result['status']=='frontier_hypothesis_budget_exceeded'
    assert result['hypotheses_evaluated']==1 and result['proposals']==[]


def test_seed_order_does_not_change_candidates_or_decisions():
    a=search(frame())
    old=baseline()
    old['proposals'].reverse()
    assert a==search(frame(),old)


def test_unresolved_fit_is_not_returned_as_an_observed_proposal(monkeypatch):
    monkeypatch.setattr('pvdelta.native_frontier._localize',lambda v,p,c:(None,{'status':'unresolved'}))
    assert search(frame())['proposals']==[]


def test_fitted_quad_touching_an_old_interior_is_withheld(monkeypatch):
    def unsafe(v,p,c):
        changed=deepcopy(p)
        changed['polygon_native_xy']=panel(180)['polygon_native_xy']
        return changed,{'status':'test_fitted'}
    monkeypatch.setattr('pvdelta.native_frontier._localize',unsafe)
    result=search(frame())
    assert result['proposals']==[] and result['counters']['protected_measurement_rejections']>0


def test_native_mutation_is_detected(monkeypatch):
    values=frame()
    def mutate(v,p,c):
        values[0,0]+=1
        return None,{'status':'unresolved'}
    monkeypatch.setattr('pvdelta.native_frontier._localize',mutate)
    with pytest.raises(RuntimeError,match='changed'):
        search(values)


@pytest.mark.parametrize('limit',[True,0,4097,512.])
def test_bad_budget_is_rejected(limit):
    with pytest.raises(ValueError,match='max_hypotheses'):
        search(frame(),max_hypotheses=limit)


@pytest.mark.parametrize('margin',[True,-1,2.])
def test_bad_margin_is_rejected(margin):
    with pytest.raises(ValueError,match='margin_px'):
        search(frame(),margin_px=margin)


@pytest.mark.parametrize('digest',[None,'z'*64,'A'*64,'0'*64])
def test_wrong_native_identity_fails_before_search(digest):
    with pytest.raises(ValueError,match='SHA256'):
        frontier_native_candidates(frame(),baseline(),native_temperature_sha256=digest)


@pytest.mark.parametrize('values',[np.ones(3),np.ones((20,40)),np.full((40,40),np.inf),
    np.ones((40,80))[:,::2],np.ones((40,40),bool)])
def test_bad_native_input_fails(values):
    with pytest.raises(ValueError,match='native'):
        search(values)


def test_wrong_baseline_native_identity_and_excessive_observation_count_fail():
    old=dict(baseline(),native_temperature_sha256='0'*64)
    with pytest.raises(ValueError,match='Baseline native'):
        search(frame(),old)
    many={'proposals':[panel(180,str(i)) for i in range(513)]}
    with pytest.raises(ValueError,match='512'):
        search(frame(),many)
