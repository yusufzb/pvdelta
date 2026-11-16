from copy import deepcopy

import numpy as np
import pytest

from pvdelta.native_local_search import local_native_alternatives


def panel(x, y):
    return {'proposal_id': 'old', 'polygon_native_xy': [[x,y],[x+30,y],[x+30,y+65],[x,y+65]],
            'source_method': 'closed_contour', 'threshold_votes': 3, 'neighbour_ids': ['old']}


def payload(proposals):
    return {'proposals': proposals, 'line_status': 'test_observed_geometry',
            'pair_search': {'search_complete': True}, 'composition_diagnostics': {}}


def test_nine_unresampled_windows_cover_odd_grid_and_correct_native_offsets(monkeypatch):
    values = np.arange(301*381, dtype=float).reshape(301,381); before = values.copy()
    calls = []
    def search(grid, **kwargs):
        assert np.shares_memory(grid, values)
        assert kwargs['support_mode'] == 'hybrid'
        calls.append(grid)
        return payload([panel(10,10),panel(44,10)])
    monkeypatch.setattr('pvdelta.native_local_search.native_line_proposals', search)
    result = local_native_alternatives(values)
    assert len(calls) == len(result['local_windows']) == 9
    for grid, info in zip(calls,result['local_windows']):
        x,y,right,bottom = info['native_window_xyxy']
        np.testing.assert_array_equal(grid,values[y:bottom,x:right])
        assert grid.shape == (151,191)
    assert len(result['proposals']) == 18
    assert all(p['neighbour_ids'] for p in result['proposals'])
    assert len({p['proposal_id'] for p in result['proposals']}) == 18
    assert max(w['native_window_xyxy'][2] for w in result['local_windows']) == 381
    assert max(w['native_window_xyxy'][3] for w in result['local_windows']) == 301
    assert not result['measurement_authorised'] and not result['efficacy_accepted']
    np.testing.assert_array_equal(values,before)


def test_duplicates_from_windows_do_not_vote_as_distinct_neighbours(monkeypatch):
    count = 0
    def search(grid, **kwargs):
        nonlocal count
        count += 1
        # All windows observe the same one panel in source space.
        x,y = [(x,y) for y in (0,75,150) for x in (0,95,190)][count-1]
        return payload([panel(200-x,170-y)])
    monkeypatch.setattr('pvdelta.native_local_search.native_line_proposals',search)
    result = local_native_alternatives(np.arange(300*380,dtype=float).reshape(300,380))
    assert result['deduplicated_observations'] == 8
    assert result['rejected_after_dedup_neighbours'] == 1
    assert result['proposals'] == []


def test_source_mutation_fails_instead_of_returning_misbound_alternatives(monkeypatch):
    values = np.arange(100*100,dtype=float).reshape(100,100)
    def search(grid,**kwargs):
        grid[0,0] += 1
        return payload([])
    monkeypatch.setattr('pvdelta.native_local_search.native_line_proposals',search)
    with pytest.raises(RuntimeError,match='changed'):
        local_native_alternatives(values)


@pytest.mark.parametrize('values',[np.array([[np.nan]]),np.array([[np.inf]]),
    np.empty((0,2)),np.ones(5),np.ones((2,2,2)),np.array([['x']])])
def test_invalid_native_input_fails_explicitly(values):
    with pytest.raises(ValueError): local_native_alternatives(values)


def test_small_grid_is_explicitly_not_searched():
    result = local_native_alternatives(np.ones((40,50)))
    assert result['status'] == 'native_grid_too_small' and result['local_windows'] == []


def test_flat_windows_are_unsearchable_not_negative_panel_evidence():
    result = local_native_alternatives(np.full((160,200),25.))
    assert result['status'] == 'no_searchable_local_windows'
    assert len(result['local_windows']) == 9 and result['proposals'] == []
    assert all(w['pair_search']['search_complete'] is False for w in result['local_windows'])


def test_real_noise_and_open_edges_do_not_become_a_filled_grid():
    import cv2
    noise = np.random.default_rng(100).normal(25,2,(160,200))
    assert local_native_alternatives(noise)['proposals'] == []
    values = np.full((240,300),25.,np.float32)
    for x in (45,95,145):
        cv2.line(values,(x,40),(x,160),15.,2)
        cv2.line(values,(x,160),(x+40,160),15.,2)
    before = deepcopy(values)
    assert local_native_alternatives(values)['proposals'] == []
    np.testing.assert_array_equal(values,before)


def test_actual_separated_panel_groups_with_different_directions_are_observed():
    import cv2
    values = np.full((480,640),20.,np.float32)
    for centre, angle in (([130,130],0),([500,350],25)):
        radians = np.deg2rad(angle)
        rotation = np.array([[np.cos(radians),-np.sin(radians)],
                             [np.sin(radians),np.cos(radians)]])
        for row in range(2):
            for col in range(3):
                polygon = np.array([[0,0],[32,0],[32,70],[0,70]])+[37*col-53,75*row-72]
                cv2.fillConvexPoly(values,np.rint(polygon@rotation.T+centre).astype(np.int32),35.)
    before = values.copy()
    result = local_native_alternatives(values)
    centres = [np.asarray(p['polygon_native_xy']).mean(axis=0) for p in result['proposals']]
    assert sum(np.linalg.norm(c-[130,130]) < 110 for c in centres) >= 4
    assert sum(np.linalg.norm(c-[500,350]) < 110 for c in centres) >= 4
    assert all(0 <= c[0] < 640 and 0 <= c[1] < 480 for c in centres)
    np.testing.assert_array_equal(values,before)
