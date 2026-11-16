import hashlib
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip('cv2')
from pvdelta.native_boundaries import NativeBoundaryConfig
from pvdelta.native_completion import _observed_sides, complete_native_proposals
from pvdelta.native_frontier import frontier_native_candidates
from pvdelta.native_transition import localize_transition_quad, transition_edge_location
from pvdelta.regions import polygon_to_mask
from pvdelta.support_partition import partition_measurement_support

CFG=NativeBoundaryConfig(min_aspect=.5,max_aspect=6.1,max_corner_cosine=.95)


@pytest.mark.parametrize('family,width,sign',[
    ('ridge',.6,1),('ridge',1.2,-1),('ridge',2.4,-1),
    ('step',.6,1),('step',1.2,-1),('step',2.4,1),('step',3.6,-1)])
def test_native_ridge_and_broad_step_are_distinct_supported_models(family,width,sign):
    y,_x=np.mgrid[:240,:320]
    z=(y-101)/width
    values=30+sign*4*(np.exp(-.5*z*z) if family=='ridge' else np.tanh(z))
    old=values.copy()
    audit,line=transition_edge_location(values,[50,100],[230,100])
    assert line is not None,audit
    assert audit['family']==family and audit['polarity']==sign
    assert abs(audit['offset_px']-1)<=.5
    assert audit['coverage']>=.8 and audit['strip_consensus_count']>=3
    np.testing.assert_array_equal(values,old)


@pytest.mark.parametrize('kind',['flat','linear','spot','noise','partial'])
def test_non_edges_or_insufficient_along_edge_coverage_are_unresolved(kind):
    y,x=np.mgrid[:240,:320]
    values=np.full((240,320),30.)
    if kind=='linear':
        values+=.2*y+.1*x
    elif kind=='spot':
        values[98:103,130:135]+=20
    elif kind=='noise':
        values+=np.random.default_rng(108).normal(0,1,values.shape)
    elif kind=='partial':
        values+=4*np.tanh((y-101)/1.2)*(x<120)
    audit,line=transition_edge_location(values,[50,100],[230,100])
    assert line is None,audit


def test_two_parallel_narrow_ridges_are_not_forced_into_one_edge():
    y,_x=np.mgrid[:240,:320]
    values=30-4*np.exp(-.5*((y-98)/.6)**2)-4*np.exp(-.5*((y-102)/.6)**2)
    audit,line=transition_edge_location(values,[50,100],[230,100])
    assert line is None,audit


def test_far_neighbouring_seam_does_not_displace_the_local_source_edge():
    y,_x=np.mgrid[:240,:320]
    values=30-4*np.exp(-.5*((y-100)/.7)**2)-4*np.exp(-.5*((y-110)/.7)**2)
    audit,line=transition_edge_location(values,[50,100],[230,100])
    assert line is not None and audit['family']=='ridge' and abs(audit['offset_px'])<=.5
    far_only=30-4*np.exp(-.5*((y-110)/.7)**2)
    assert transition_edge_location(far_only,[50,100],[230,100])[1] is None


def test_large_contiguous_missing_segment_fails_even_if_most_of_the_edge_is_present():
    y,x=np.mgrid[:240,:320]
    present=(x<125)|(x>=160)
    values=30-4*np.exp(-.5*((y-100)/1.2)**2)*present
    audit,line=transition_edge_location(values,[50,100],[230,100])
    assert line is None,audit
    assert audit['status']=='insufficient_full_edge_transition' and audit['largest_gap_fraction']>.18


def test_spatial_strips_that_disagree_on_location_are_unresolved():
    y,x=np.mgrid[:240,:320]
    centre=np.where(x<140,98,102)
    values=30-4*np.exp(-.5*((y-centre)/1.2)**2)
    assert transition_edge_location(values,[50,100],[230,100])[1] is None


def test_source_border_cannot_be_padded_to_supply_a_step():
    audit,line=transition_edge_location(np.full((100,160),30.),[20,5],[130,5])
    assert line is None and audit['status']=='insufficient_source_flanks'


def panel(x):
    return {'proposal_id':f'N{x}','polygon_native_xy':[[x,45],[x+100,45],[x+100,95],[x,95]],
        'source_method':'closed_contour','threshold_votes':2}


def mixed_frame(hot=False):
    y,x=np.mgrid[:240,:420]
    values=25+2.5*(np.tanh((y-45)/1.2)-np.tanh((y-95)/1.2))
    distance=np.minimum.reduce([abs(x-v) for v in (70,170,180,280,290,390)])
    values-=4*np.exp(-.5*(distance/.7)**2)
    if hot:
        values[57:83,90:150]+=5
    return values.astype(np.float32)


@pytest.mark.parametrize('hot',[False,True])
def test_two_step_two_ridge_quad_recovers_without_changing_native_values(hot):
    values=mixed_frame(hot)
    old=values.copy()
    p=panel(70)
    original=deepcopy(p)
    fitted,audit=localize_transition_quad(values,p,CFG)
    assert fitted is not None,audit
    assert fitted['side_model_families']==['step','ridge','step','ridge']
    np.testing.assert_allclose(fitted['polygon_native_xy'],p['polygon_native_xy'],atol=.75)
    assert _observed_sides(fitted) and not fitted['measurement_authorised']
    assert p==original
    np.testing.assert_array_equal(values,old)


def test_four_step_warm_patch_cannot_supply_observed_panel_membership():
    y,x=np.mgrid[:240,:420]
    values=25+1.25*(np.tanh((x-70)/1.2)-np.tanh((x-170)/1.2))*(np.tanh((y-45)/1.2)-np.tanh((y-95)/1.2))
    fitted,audit=localize_transition_quad(values,panel(70),CFG)
    assert fitted is not None,audit
    assert fitted['side_model_families']==['step']*4
    assert not _observed_sides(fitted)
    baseline={'proposals':[panel(180),panel(290)]}
    completed=complete_native_proposals(baseline,{'proposals':[fitted]},native_shape=values.shape)
    assert completed['proposals']==baseline['proposals']
    assert completed['native_completion']['rejection_counts']=={'insufficient_observed_side_evidence':1}


def test_single_step_cannot_close_a_quad():
    y,_x=np.mgrid[:240,:420]
    fitted,audit=localize_transition_quad(25+5*np.tanh((y-45)/1.2),panel(70),CFG)
    assert fitted is None and audit['status']=='unchanged_unresolved_transitions'


def test_mixed_frontier_preserves_old_geometry_and_every_measurement_interior():
    values=mixed_frame(True)
    old=values.copy()
    baseline={'proposals':[panel(180),panel(290)]}
    previous=deepcopy(baseline)
    pool=frontier_native_candidates(values,baseline,edge_model='mixed',
        native_temperature_sha256=hashlib.sha256(values.tobytes()).hexdigest())
    assert pool['proposals']
    assert all(p['source_method']=='four_native_transitions' for p in pool['proposals'])
    completed=complete_native_proposals(baseline,pool,native_shape=values.shape)
    assert len(completed['proposals'])==3
    assert completed['proposals'][:2]==previous['proposals'] and baseline==previous
    supports=[partition_measurement_support([polygon_to_mask(p['polygon_native_xy'],values.shape)
        for p in geometry['proposals']],margin_px=2) for geometry in (baseline,completed)]
    assert all(np.array_equal(a.interior,b.interior) for a,b in zip(*supports))
    np.testing.assert_array_equal(values,old)


@pytest.mark.parametrize('change',['missing','bad_schema','measurement','coverage','gap','offset',
    'consensus','family','one_ridge','nan'])
def test_malformed_mixed_evidence_cannot_bypass_completion(change):
    fitted,audit=localize_transition_quad(mixed_frame(),panel(70),CFG)
    assert fitted is not None,audit
    if change=='missing':
        del fitted['native_transition_evidence']
    elif change=='bad_schema':
        fitted['native_transition_evidence']['schema']='other'
    elif change=='measurement':
        fitted['native_transition_evidence']['measurement_authorised']=True
    elif change=='coverage':
        fitted['side_coverage'][0]=.9
    elif change=='gap':
        fitted['side_max_gap_fraction'][0]=.1
    elif change=='one_ridge':
        fitted['side_model_families'][1]='step'
    else:
        edge=fitted['native_transition_evidence']['edges'][0]
        if change=='offset':edge['offset_px']=2
        elif change=='consensus':edge['strip_consensus_count']=2
        elif change=='family':edge['family']='ridge'
        else:edge['coverage']=float('nan')
    assert not _observed_sides(fitted)


@pytest.mark.parametrize('values,start,end',[
    (np.ones(5),[20,20],[50,20]),(np.full((50,50),np.nan),[10,25],[40,25]),
    (np.ones((50,50),bool),[10,25],[40,25]),(np.ones((50,50)),[10,25],[12,25]),
    (np.ones((50,50)),[float('inf'),25],[40,25])])
def test_invalid_grid_or_endpoints_fail_closed(values,start,end):
    with pytest.raises(ValueError):
        transition_edge_location(values,start,end)


def test_unknown_edge_model_is_rejected_and_default_ridge_stays_available():
    values=mixed_frame()
    with pytest.raises(ValueError,match='edge_model'):
        frontier_native_candidates(values,{'proposals':[]},native_temperature_sha256='0'*64,edge_model='unknown')
    result=frontier_native_candidates(values,{'proposals':[]},native_temperature_sha256=hashlib.sha256(values.tobytes()).hexdigest())
    assert result['proposals']==[]
