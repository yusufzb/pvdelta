from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip('cv2')

from pvdelta.native_windows import _translate_proposal, native_window_proposals


def proposal(x, y):
    return {'polygon_native_xy':[[x,y],[x+40,y],[x+40,y+90],[x,y+90]],
            'source_method':'four_observed_lines','proposal_origin':'hybrid_structure'}


def payload(proposals):
    return {'proposals':proposals,'structure_pass':'hybrid_with_segment_baseline',
        'line_status':'unreviewed_observed_line_geometry','composition_diagnostics':{},
        'superseded_compound_proposals':[],'composition_rejected_proposals':[],
        'measurement_authorised':False}


def test_native_translation_includes_scale_references_without_mutating_input():
    p=proposal(10,15)
    p['internal_structure_resolution']={'reference_polygons_native_xy':[
        proposal(60,15)['polygon_native_xy'],proposal(110,15)['polygon_native_xy']]}
    original=deepcopy(p)
    result=_translate_proposal(p,[200,100,627,442])
    np.testing.assert_allclose(result['polygon_native_xy'],np.array(p['polygon_native_xy'])+[200,100])
    np.testing.assert_allclose(result['internal_structure_resolution']['reference_polygons_native_xy'],
        np.array(p['internal_structure_resolution']['reference_polygons_native_xy'])+[200,100])
    assert result['native_window_xyxy']==[200,100,627,442]
    assert p==original


def test_windows_are_four_exact_source_slices_with_native_offsets(monkeypatch):
    import pvdelta.native_windows as windows

    values=np.arange(240*300,dtype=float).reshape(240,300)
    original=values.copy();calls=[]
    def search(grid,**kwargs):
        calls.append(grid)
        assert np.shares_memory(grid,values)
        return payload([] if len(calls)==1 else [proposal(20,20),proposal(65,20)])
    monkeypatch.setattr(windows,'native_line_proposals',search)
    result=native_window_proposals(values)
    assert len(calls)==5
    for grid,(x,y) in zip(calls[1:],[(0,0),(100,0),(0,80),(100,80)]):
        np.testing.assert_array_equal(grid,values[y:y+160,x:x+200])
    windows_used={tuple(p['native_window_xyxy']) for p in result['proposals']}
    assert windows_used=={(0,0,200,160),(100,0,300,160),(0,80,200,240),(100,80,300,240)}
    assert not result['local_window_search']['radiometric_resampling']
    assert all(p['proposal_origin']=='local_native_window' for p in result['proposals'])
    np.testing.assert_array_equal(values,original)


def test_near_duplicate_windows_keep_global_geometry_and_provenance(monkeypatch):
    import pvdelta.native_windows as windows

    global_panels=[proposal(20,20),proposal(65,20)];count=0
    def search(grid,**kwargs):
        nonlocal count
        count+=1
        return payload(deepcopy(global_panels) if count==1 else
                       [proposal(22,20),proposal(67,20)] if count==2 else [])
    monkeypatch.setattr(windows,'native_line_proposals',search)
    result=native_window_proposals(np.ones((240,300)))
    assert [p['polygon_native_xy'] for p in result['proposals']]==[
        p['polygon_native_xy'] for p in global_panels]
    assert all(p['proposal_origin']=='hybrid_structure' for p in result['proposals'])
    assert not result['measurement_authorised']


def test_partial_open_frames_cannot_be_closed_by_window_borders():
    import cv2

    values=np.full((320,400),25.,np.float32)
    for x in (50,110,170):
        cv2.line(values,(x,70),(x,230),15.,2)
        cv2.line(values,(x,230),(x+50,230),15.,2)
        cv2.line(values,(x+50,230),(x+50,70),15.,2)
    before=values.copy()
    result=native_window_proposals(values)
    assert result['proposals']==[]
    np.testing.assert_array_equal(values,before)


def test_local_search_does_not_force_panels_in_noise():
    values=np.random.default_rng(13).normal(25,3,(320,400))
    assert native_window_proposals(values)['proposals']==[]


def local_half(y, x=20, width=85, height=45):
    p=proposal(x,y)
    p.update(polygon_native_xy=[[x,y],[x+width,y],[x+width,y+height],[x,y+height]],
             proposal_origin='local_native_window')
    return p


def test_cross_panel_halves_are_review_only_and_original_interiors_are_restored():
    from pvdelta.native_boundaries import NativeBoundaryConfig
    from pvdelta.native_windows import _withhold_cross_panel_locals
    from pvdelta.regions import polygon_to_mask
    from pvdelta.support_partition import partition_measurement_support

    panels=[proposal(20,20),proposal(65,20)]
    global_result=payload(panels)
    result=payload(panels+[local_half(20),local_half(65)])
    result['local_window_search']={}
    before=deepcopy((result,global_result))
    selected=_withhold_cross_panel_locals(result,global_result,NativeBoundaryConfig())
    assert len(selected['proposals'])==2
    conflicts=selected['local_orientation_conflicts']
    assert len(conflicts)==2
    assert all(len(p['reference_polygons_native_xy'])==2 for p in conflicts)
    assert all(not p['measurement_authorised'] for p in conflicts)
    assert (result,global_result)==before
    shape=(140,140)
    original=partition_measurement_support([
        polygon_to_mask(p['polygon_native_xy'],shape) for p in result['proposals']],margin_px=2)
    assert not original[0].interior.any() and not original[1].interior.any()
    supports=partition_measurement_support([
        polygon_to_mask(p['polygon_native_xy'],shape) for p in selected['proposals']],margin_px=2)
    assert all(s.interior.any() for s in supports)
    assert supports[0].interior[65,40] and supports[1].interior[65,85]
    assert all(p['proposal_origin']=='hybrid_structure' for p in selected['proposals'])


@pytest.mark.parametrize('baseline,local',[
    ([proposal(20,20)],local_half(20)),
    ([proposal(20,20),proposal(20,20)],local_half(20)),
    ([proposal(20,20),proposal(100,20)],local_half(20,width=120)),
    ([proposal(20,20),proposal(65,20)],local_half(20,height=10)),
    ([proposal(20,20),proposal(65,20)],proposal(20,20)),
])
def test_a_single_duplicate_far_grazing_or_parallel_reference_does_not_veto_local(baseline,local):
    from pvdelta.native_windows import _cross_panel_references

    assert _cross_panel_references(local,baseline)==[]


def test_superseded_global_envelopes_cannot_supply_conflict_votes():
    from pvdelta.native_boundaries import NativeBoundaryConfig
    from pvdelta.native_windows import _withhold_cross_panel_locals

    global_result=payload([proposal(20,20),proposal(65,20)])
    result=payload([local_half(20),local_half(65)]);result['local_window_search']={}
    selected=_withhold_cross_panel_locals(result,global_result,NativeBoundaryConfig())
    assert selected['local_orientation_conflicts']==[]
    assert selected['proposals']==result['proposals']


def raw_select(panels, candidates):
    from pvdelta.native_boundaries import NativeBoundaryConfig
    from pvdelta.native_windows import _supplement_raw_structure

    raw=payload(candidates)
    # _line_pass is a single pass and does not set the composition-level tag.
    raw.pop('structure_pass')
    return _supplement_raw_structure(payload(panels),raw,NativeBoundaryConfig())


def test_raw_restores_missing_envelope_with_retained_scale_and_adjacent_support():
    panels=[proposal(110,20),proposal(65,20)]
    candidates=[proposal(20,20)]
    before=deepcopy((panels,candidates))
    result=raw_select(panels,candidates)
    assert len(result['proposals'])==3
    assert [p['polygon_native_xy'] for p in result['proposals'][:2]]==[
        p['polygon_native_xy'] for p in panels]
    added=result['proposals'][-1]
    assert added['proposal_origin']=='raw_structure_supplement'
    assert len(added['raw_scale_support']['reference_polygons_native_xy'])==2
    assert (panels,candidates)==before
    assert not result['measurement_authorised']
    assert result['raw_structure_supplement']['retained_additions']==1


@pytest.mark.parametrize('references',[
    [],
    [proposal(65,20)],
    [proposal(65,20),proposal(65,20)],
    [proposal(155,20),proposal(200,20)],
    [local_half(20,x=65),local_half(20,x=155)],
])
def test_raw_cannot_use_missing_duplicate_distant_or_rotated_references(references):
    result=raw_select(references,[proposal(20,20)])
    assert result['raw_structure_supplement']['retained_additions']==0


def test_raw_cannot_bootstrap_scale_from_its_own_new_candidates():
    result=raw_select([proposal(155,20)],[proposal(20,20),proposal(65,20),proposal(110,20)])
    assert result['raw_structure_supplement']['retained_additions']==0


def test_raw_overlapping_a_third_existing_envelope_is_withheld():
    # Two valid scale references cannot hide a conflict with a different envelope.
    crossing=local_half(20,x=20)
    result=raw_select([proposal(65,20),proposal(110,20),crossing],[proposal(20,20)])
    assert result['raw_structure_supplement']['retained_additions']==0
    assert len(result['proposals'])==3


def test_raw_duplicate_cannot_shift_existing_geometry():
    panels=[proposal(20,20),proposal(65,20)]
    result=raw_select(panels,[proposal(22,20)])
    assert [p['polygon_native_xy'] for p in result['proposals']]==[
        p['polygon_native_xy'] for p in panels]
    assert result['raw_structure_supplement']['retained_additions']==0


def test_raw_contour_only_envelopes_are_not_line_supplements():
    candidate=proposal(20,20)
    candidate['source_method']='closed_contour'
    result=raw_select([proposal(65,20),proposal(110,20)],[candidate])
    assert result['raw_structure_supplement']['retained_additions']==0
    assert result['raw_structure_supplement']['rejected_proposals'][0]['reason']==(
        'raw_requires_four_supported_sides')


def test_raw_opt_in_preserves_source_and_previous_window_conflict_provenance(monkeypatch):
    import pvdelta.native_windows as windows

    calls=[]
    def search(grid,**kwargs):
        calls.append(grid.shape)
        return payload([proposal(65,20),proposal(110,20)] if len(calls)==1 else [])
    def raw(grid,**kwargs):
        assert kwargs['local_contrast'] is False and kwargs['support_mode']=='hybrid'
        result=payload([proposal(20,20)])
        result.pop('structure_pass')
        return result
    monkeypatch.setattr(windows,'native_line_proposals',search)
    monkeypatch.setattr(windows,'_line_pass',raw)
    values=np.arange(320*400,dtype=float).reshape(320,400)
    before=values.copy()
    result=native_window_proposals(values,raw_supplement=True)
    assert len(result['proposals'])==3
    assert len(calls)==5
    assert len(result['local_window_search']['windows'])==4
    assert result['local_orientation_conflicts']==[]
    assert 'global_search_rejections' in result
    np.testing.assert_array_equal(values,before)


def test_raw_flag_requires_explicit_boolean():
    with pytest.raises(TypeError,match='raw_supplement must be boolean'):
        native_window_proposals(np.ones((40,40)),raw_supplement='yes')
