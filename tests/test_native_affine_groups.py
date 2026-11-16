import numpy as np
import pytest

from pvdelta.native_affine_groups import _cycles, affine_native_groups


def scene(missing=None):
    import cv2
    values = np.full((320,400),20.,np.float32)
    # Projected aspect below the old 1.7 bound, with non-right-angle corners.
    basis = np.array([[40.,14.],[0.,60.]])
    for row in range(2):
        for col in range(2):
            if (row,col) == missing:continue
            origin = np.array([90.,60.])+basis@np.array([col*1.1,row*1.1])
            polygon=np.array([[0,0],[1,0],[1,1],[0,1]])@basis.T+origin
            cv2.fillConvexPoly(values,np.rint(polygon).astype(np.int32),35.)
    return values


def test_four_observed_sheared_cells_pass_without_changing_temperature():
    values=scene();before=values.copy();result=affine_native_groups(values)
    assert len(result['proposals'])==4 and result['groups']
    assert all(p['threshold_votes']>=2 and p['affine_group_ids'] for p in result['proposals'])
    assert all(len(p['neighbour_ids'])==2 for p in result['proposals'])
    assert not result['measurement_authorised'] and not result['efficacy_accepted']
    assert result['semantic_panel_membership']=='unverified'
    np.testing.assert_array_equal(values,before)


def test_three_observations_never_invent_missing_fourth_cell():
    assert affine_native_groups(scene(missing=(1,1)))['proposals']==[]


def test_regular_non_pv_rectangles_remain_explicitly_unverified():
    # Exactly the same structure could be roof tiles or windows.
    result=affine_native_groups(scene())
    assert result['proposals'] and all(p['semantic_panel_membership']=='unverified' for p in result['proposals'])


def test_one_dimensional_row_is_not_a_two_axis_group():
    base=np.array([[0,0],[40,0],[54,60],[14,60]],float)
    observed=[{'quad':base+[50+i*44,50]} for i in range(6)]
    assert _cycles(observed)==([],'complete')


def test_grid_budget_failure_is_explicit_not_a_partial_selected_sample():
    observed=[{'quad':np.array([[0,0],[30,0],[30,60],[0,60]])} for _ in range(513)]
    assert _cycles(observed)==([],'too_many_observed_quads')


def test_noise_does_not_become_a_forced_panel_array():
    values=np.random.default_rng(101).normal(25,2,(320,400))
    assert affine_native_groups(values)['proposals']==[]


def test_conflicting_two_axis_groups_are_all_withheld(monkeypatch):
    quad=np.array([[0,0],[30,0],[30,60],[0,60]],float)
    observed=[{'quad':quad+[70+col*33+offset,50+row*66],'votes':{0,1},'area':1800.}
              for offset in (0,8) for row in range(2) for col in range(2)]
    monkeypatch.setattr('pvdelta.native_affine_groups._observed_quads',lambda a:(observed,16))
    result=affine_native_groups(np.ones((240,300)))
    assert result['cycles_before_conflict_check']>0 and result['conflicted_quads_withheld']==8
    assert result['proposals']==[] and result['groups']==[]


def test_native_mutation_fails_the_whole_search(monkeypatch):
    def mutate(a):
        a[0,0]+=1
        return [],0
    monkeypatch.setattr('pvdelta.native_affine_groups._observed_quads',mutate)
    with pytest.raises(RuntimeError,match='changed'):
        affine_native_groups(np.ones((100,100)))


@pytest.mark.parametrize('values',[np.array([[np.nan]]),np.ones((10,10)),np.ones((40,40,3)),
    np.ones(4),np.array([['x']]),np.full((40,40),np.inf),np.ones((40,40))])
def test_invalid_or_flat_native_grid_fails_explicitly(values):
    with pytest.raises(ValueError):affine_native_groups(values)
