from copy import deepcopy

import numpy as np
import pytest

from pvdelta.native_boundaries import NativeBoundaryConfig
from pvdelta.native_localization import _edge_location, _localize, localize_native_alternatives


def frame():
    y,x=np.mgrid[:240,:320]
    distance=np.minimum.reduce([abs(x-60),abs(x-160),abs(y-45),abs(y-95)])
    return (30-4*np.exp(-.5*(distance/.7)**2)).astype(np.float32)


def proposal(identifier='p',quad=None):
    return {'proposal_id':identifier,'polygon_native_xy':quad or [[62,47],[158,47],[158,93],[62,93]],
        'source_method':'four_observed_lines','side_coverage':[1.]*4,'side_max_gap_fraction':[0.]*4}


def test_localization_recovers_observed_frame_with_all_fresh_edges_and_source_unchanged():
    values=frame();old=values.copy();p=proposal();original=deepcopy(p)
    refined,audit=_localize(values.astype(float),p,NativeBoundaryConfig())
    assert refined is not None,audit
    np.testing.assert_allclose(refined['polygon_native_xy'],[[60,45],[160,45],[160,95],[60,95]],atol=.75)
    assert min(refined['side_coverage'])>=.8 and max(refined['side_max_gap_fraction'])<=.18
    assert refined['native_localization']['parent_polygon_native_xy']==p['polygon_native_xy']
    assert not refined['measurement_authorised']
    assert p==original
    np.testing.assert_array_equal(values,old)


@pytest.mark.parametrize('kind',['constant','slope','single_step','spot'])
def test_weak_steps_and_isolated_hotspot_cannot_authorize_a_frame_snap(kind):
    y,x=np.mgrid[:240,:320]
    values=np.full((240,320),25.)
    if kind=='slope':values+=x*.02+y*.01
    if kind=='single_step':values+=5*(y>45)
    if kind=='spot':values[44:48,90:94]+=20
    refined,audit=_localize(values,proposal(),NativeBoundaryConfig())
    assert refined is None and audit['status']=='unchanged_unresolved_edges'


def test_multiple_equally_strong_ridges_are_ambiguous():
    y,_x=np.mgrid[:120,:200]
    values=30-4*np.exp(-.5*((y-48)/.5)**2)-4*np.exp(-.5*((y-52)/.5)**2)
    decision,line=_edge_location(values,np.array([20.,50.]),np.array([180.,50.]))
    assert line is None and decision['status']=='unresolved_strip_consensus'


def test_source_border_has_no_synthetic_padding_evidence():
    values=np.full((100,150),25.)
    decision,line=_edge_location(values,np.array([2.,3.]),np.array([100.,3.]))
    assert line is None and decision['status']=='insufficient_source_flanks'


def test_no_supported_geometry_or_labels_means_no_addition_and_inputs_not_mutated():
    values=np.full((240,320),25.)
    base={'proposals':[]};alternative={'proposals':[proposal()]}
    previous=deepcopy(alternative)
    result=localize_native_alternatives(values,base,alternative)
    assert result['proposals']==[]
    assert alternative==previous
    assert not result['native_localization_search']['measurement_authorised']
    assert result['native_localization_search']['decisions'][0]['status']=='unchanged_unresolved_edges'


def test_baseline_proposal_is_never_relocalized():
    p=proposal();base={'proposals':[p]}
    result=localize_native_alternatives(frame(),base,deepcopy(base))
    assert result['proposals']==[p]
    assert result['native_localization_search']['decisions'][0]['status']=='unchanged_duplicate_or_unobserved'


def test_shift_touching_protected_baseline_support_is_withheld(monkeypatch):
    base={'proposals':[proposal('base',[[168,45],[268,45],[268,95],[168,95]])]}
    alt={'proposals':[proposal()]}
    def unsafe(values,p,cfg):
        changed=deepcopy(p);changed['polygon_native_xy']=[[160,45],[260,45],[260,95],[160,95]]
        return changed,{'status':'localized_four_supported_ridges'}
    monkeypatch.setattr('pvdelta.native_localization._localize',unsafe)
    result=localize_native_alternatives(frame(),base,alt)
    assert result['proposals'][0]==base['proposals'][0]
    assert result['native_localization_search']['decisions'][0]['status']=='unchanged_baseline_protection'


@pytest.mark.parametrize('values',[np.ones(3),np.zeros((0,5)),np.ones((7,9)),np.array([[np.inf]*8]*8),np.array([['x']*8]*8)])
def test_invalid_native_grids_fail_closed(values):
    with pytest.raises(ValueError):localize_native_alternatives(values,{'proposals':[]},{'proposals':[]})


def test_invalid_config_and_geometry_are_rejected():
    with pytest.raises(TypeError):localize_native_alternatives(frame(),{'proposals':[]},{'proposals':[]},config={})
    with pytest.raises(ValueError):localize_native_alternatives(frame(),{'proposals':[proposal(quad=[[0,0]]*4)]},{'proposals':[]})


def test_native_mutation_is_detected(monkeypatch):
    values=frame()
    def mutate(native,p,cfg):
        values[0,0]+=1
        return None,{'status':'unresolved'}
    monkeypatch.setattr('pvdelta.native_localization._localize',mutate)
    with pytest.raises(RuntimeError,match='changed'):
        localize_native_alternatives(values,{'proposals':[]},{'proposals':[proposal()]})
