import hashlib
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip('cv2')
from pvdelta.convex_geometry import convex_iou
from pvdelta.native_completion import complete_native_proposals
from pvdelta.native_level_regions import level_region_candidates, supported_level_region
from pvdelta.regions import polygon_to_mask
from pvdelta.support_partition import partition_measurement_support


def scene(*, cold=False):
    v = np.full((200,300), 45. if cold else 15., dtype=np.float32)
    for i, x in enumerate((20,65,110)):
        v[25:120,x:x+40] = 30.+i
    return v


def baseline(found, ids=(1,2)):
    return {'proposals':[dict(deepcopy(found['proposals'][i]), proposal_id=f'N{i}') for i in ids]}


@pytest.mark.parametrize('cold',[False,True])
def test_observed_warm_or_cold_closed_regions_are_deterministic_and_native(cold):
    values = scene(cold=cold)
    old = values.copy()
    result = level_region_candidates(values)
    assert result == level_region_candidates(values)
    assert len(result['proposals']) == 3
    assert all(supported_level_region(p) for p in result['proposals'])
    assert [p['polygon_native_xy'][0] for p in result['proposals']] == [[20.,25.],[65.,25.],[110.,25.]]
    assert result['native_temperature_sha256'] == hashlib.sha256(values.tobytes()).hexdigest()
    np.testing.assert_array_equal(values,old)
    assert not result['radiometric_resampling'] and not result['measurement_authorised']
    assert all(not p['measurement_authorised'] and p['semantic_panel_membership']=='unverified'
               for p in result['proposals'])


def test_completion_adds_one_region_without_changing_existing_pixels_or_geometry():
    v = scene()
    found = level_region_candidates(v)
    old = baseline(found)
    original = deepcopy(old)
    completed = complete_native_proposals(old,found,native_shape=v.shape)
    assert old==original and completed['proposals'][:2]==old['proposals']
    assert len(completed['proposals'])==3
    assert completed['proposals'][-1]['source_method']=='native_level_regions'
    supports = [partition_measurement_support([polygon_to_mask(p['polygon_native_xy'],v.shape)
        for p in g['proposals']],margin_px=2) for g in (old,completed)]
    assert all(np.array_equal(a.interior,b.interior) for a,b in zip(*supports))


@pytest.mark.parametrize('ids',[(),(1,)])
def test_isolated_rectangle_is_not_approved_without_two_existing_references(ids):
    found=level_region_candidates(scene())
    old=baseline(found,ids)
    assert complete_native_proposals(old,found,native_shape=(200,300))['proposals']==old['proposals']


def test_internal_hot_patch_cannot_supply_full_module_scale():
    v=scene()
    v[40:70,170:195]=35
    found=level_region_candidates(v)
    old={'proposals':[dict(deepcopy(p),proposal_id=f'N{i}') for i,p in enumerate(found['proposals'])
                      if np.asarray(p['polygon_native_xy'])[:,0].mean()<160]}
    assert len(old['proposals'])==3
    completed=complete_native_proposals(old,found,native_shape=v.shape)
    assert completed['proposals']==old['proposals']


def test_holes_count_against_occupancy_and_do_not_become_a_filled_rectangle():
    v=scene()
    v[40:100,25:55]=15
    found=level_region_candidates(v)
    outer=[[20,25],[59,25],[59,119],[20,119]]
    assert all(convex_iou(p['polygon_native_xy'],outer)<.8 for p in found['proposals'])
    # The cold hole may itself be an observed region, but is not full-panel scale.
    old={'proposals':[dict(deepcopy(p),proposal_id=f'N{i}') for i,p in enumerate(found['proposals'])
                      if np.asarray(p['polygon_native_xy'])[:,0].mean()>60]}
    assert len(old['proposals'])==2
    assert complete_native_proposals(old,found,native_shape=v.shape)['proposals']==old['proposals']


def test_source_border_partial_regions_are_not_completed():
    v=scene()
    v[25:120,0:40]=30
    found=level_region_candidates(v)
    assert all(np.asarray(p['polygon_native_xy'])[:,0].min()>=3 for p in found['proposals'])


def test_budget_overflow_withholds_entire_pool():
    result=level_region_candidates(scene(),max_observations=1)
    assert result['status']=='level_region_budget_exceeded' and result['proposals']==[]
    assert result['raw_region_observations']==2


@pytest.mark.parametrize('change',['one_level','nonadjacent','same_threshold','low_fill','low_iou',
    'changed_corners','wrong_hash','wrong_shape','window','invalid_polarity','measurement_claim'])
def test_incomplete_or_tampered_region_evidence_is_not_completion_support(change):
    proposal=level_region_candidates(scene())['proposals'][0]
    evidence=proposal['native_level_region_evidence']
    if change=='one_level':evidence['support_levels'].pop()
    elif change=='nonadjacent':evidence['support_levels'][1]['level_index']+=2
    elif change=='same_threshold':evidence['support_levels'][1]['threshold_c']=evidence['support_levels'][0]['threshold_c']
    elif change=='low_fill':evidence['support_levels'][0]['occupancy']=.6
    elif change=='low_iou':evidence['support_levels'][0]['iou_to_representative']=.7
    elif change=='changed_corners':proposal['polygon_native_xy'][0][0]+=10
    elif change=='wrong_hash':evidence['native_temperature_sha256']='wrong'
    elif change=='wrong_shape':evidence['native_shape']=[False,300]
    elif change=='window':evidence['support_window_xyxy']=[0,0,500,500]
    elif change=='invalid_polarity':evidence['support_polarity']='any'
    elif change=='measurement_claim':evidence['measurement_authorised']=True
    assert not supported_level_region(proposal)


@pytest.mark.parametrize('values',[np.ones((20,40)), np.ones((40,40)), np.full((40,40),np.nan),
    np.ones((40,40),bool),np.zeros((40,80))[:,::2],np.ones((40,40,3))])
def test_invalid_or_flat_source_fails_explicitly(values):
    with pytest.raises(ValueError):level_region_candidates(values)


@pytest.mark.parametrize('budget',[True,0,8193,1.5])
def test_invalid_budget_fails_before_extraction(budget):
    with pytest.raises(ValueError):level_region_candidates(scene(),max_observations=budget)


def test_mutated_source_is_not_returned_as_original(monkeypatch):
    import cv2
    original=cv2.connectedComponentsWithStats
    v=scene()
    def changed(*args,**kwargs):
        result=original(*args,**kwargs)
        v[0,0]+=1
        return result
    monkeypatch.setattr(cv2,'connectedComponentsWithStats',changed)
    with pytest.raises(RuntimeError,match='changed'):level_region_candidates(v)
