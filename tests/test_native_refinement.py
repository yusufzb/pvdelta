from copy import deepcopy

import numpy as np
import pytest

import pvdelta.native_refinement as module


def geometry(count=1):
    return {'proposals': [{'proposal_id': chr(65+i),
        'polygon_native_xy': [[10+22*i, 10], [30+22*i, 10], [30+22*i, 50], [10+22*i, 50]]}
        for i in range(count)]}


def moved(proposal, shift):
    p = deepcopy(proposal)
    p.update(polygon_native_xy=(np.asarray(p['polygon_native_xy'])+[shift, 0]).tolist(),
             source_method='four_observed_lines', side_coverage=[.95]*4,
             side_max_gap_fraction=[.02]*4)
    return p, {'status': 'localized_four_supported_ridges'}


def test_changes_are_detached_and_old_masks_cannot_be_reused(monkeypatch):
    values = np.arange(10000, dtype=np.float32).reshape(100, 100)
    original = values.copy()
    g = geometry()
    old = deepcopy(g)
    monkeypatch.setattr(module, '_localize', lambda a, p, cfg: moved(p, 2))
    result = module.refine_native_geometry(values, g)
    assert result['proposals'][0]['polygon_native_xy'][0] == [12, 10]
    assert g == old
    np.testing.assert_array_equal(values, original)
    assert result['native_refinement']['localized_proposals'] == 1
    assert result['native_refinement']['requires_new_measurement_masks']
    assert not result['native_refinement']['old_measurement_reuse_authorised']
    result['proposals'][0]['polygon_native_xy'][0][0] = 99
    assert g == old


def test_unresolved_edge_falls_back_or_stays_original(monkeypatch):
    monkeypatch.setattr(module, '_localize', lambda *a: (None, {'status': 'unresolved_ridge'}))
    monkeypatch.setattr(module, 'localize_transition_quad', lambda a, p, cfg: moved(p, 1))
    g = geometry()
    result = module.refine_native_geometry(np.ones((100, 100)), g)
    assert result['native_refinement']['localized_proposals'] == 1
    monkeypatch.setattr(module, 'localize_transition_quad', lambda *a: (None, {'status': 'unresolved_transition'}))
    result = module.refine_native_geometry(np.ones((100, 100)), g)
    assert result['proposals'] == g['proposals']
    assert result['native_refinement']['decisions'][0]['audit']['status'] == 'unresolved_transition'


def test_conflicting_changes_reject_both_in_any_input_order(monkeypatch):
    monkeypatch.setattr(module, '_localize', lambda a, p, cfg: moved(p, 2 if p['proposal_id'] == 'A' else -2))
    for g in (geometry(2), {'proposals': geometry(2)['proposals'][::-1]}):
        result = module.refine_native_geometry(np.ones((100, 100)), g)
        assert result['proposals'] == g['proposals']
        assert result['native_refinement']['localized_proposals'] == 0


def test_restored_baseline_can_invalidate_another_change(monkeypatch):
    shifts = {'A': 2, 'B': -2, 'C': -4}
    monkeypatch.setattr(module, '_localize', lambda a, p, cfg: moved(p, shifts[p['proposal_id']]))
    g = geometry(3)
    result = module.refine_native_geometry(np.ones((100, 100)), g)
    assert result['proposals'] == g['proposals']
    assert result['native_refinement']['localized_proposals'] == 0


@pytest.mark.parametrize('kind', ['identity', 'movement', 'weak_side'])
def test_no_identity_substitution_large_move_or_weak_edge(monkeypatch, kind):
    def localize(a, p, cfg):
        p, audit = moved(p, 7 if kind == 'movement' else 1)
        if kind == 'identity':
            p['proposal_id'] = 'OTHER'
        if kind == 'weak_side':
            p['side_coverage'][0] = .4
        return p, audit
    monkeypatch.setattr(module, '_localize', localize)
    g = geometry()
    result = module.refine_native_geometry(np.ones((100, 100)), g)
    assert result['proposals'] == g['proposals']


def test_mutating_geometry_working_array_fails_without_changing_native(monkeypatch):
    def localize(a, p, cfg):
        a[0, 0] += 1
        return moved(p, 1)
    monkeypatch.setattr(module, '_localize', localize)
    values = np.ones((100, 100), np.float32)
    with pytest.raises(RuntimeError, match='changed source'):
        module.refine_native_geometry(values, geometry())
    assert values[0, 0] == 1


def test_actual_flat_source_cannot_create_edges():
    g = geometry()
    result = module.refine_native_geometry(np.ones((100, 100), np.float32), g)
    assert result['proposals'] == g['proposals']
    assert result['native_refinement']['localized_proposals'] == 0


@pytest.mark.parametrize('value', [np.ones((2, 2)), np.ones((10, 10), dtype=bool), np.full((20, 20), np.nan)])
def test_invalid_native_grid_rejected(value):
    with pytest.raises(ValueError):
        module.refine_native_geometry(value, geometry())
