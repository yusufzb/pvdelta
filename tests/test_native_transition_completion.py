from copy import deepcopy

import numpy as np
import pytest

import pvdelta.native_transition_completion as module
from pvdelta.native_search_trace import begin_trace
from pvdelta.regions import polygon_to_mask
from pvdelta.support_partition import partition_measurement_support


def panel(x, identifier):
    return {'proposal_id': identifier, 'polygon_native_xy': [[x, 10], [x+20, 10], [x+20, 50], [x, 50]],
            'source_method': 'four_observed_lines', 'side_coverage': [.95]*4, 'side_max_gap_fraction': [.01]*4}


@pytest.fixture
def setup(monkeypatch):
    baseline = {'proposals': [panel(10, 'N001'), panel(32, 'N002')]}
    events = [panel(54, 'candidate')['polygon_native_xy']]
    anchors = [True, True, False, False]
    calls = []
    def search(values, **kwargs):
        trace = begin_trace(values, support_mode='segments', local_contrast=True, pair_budget=250000)
        for q in events:
            trace.candidate('side_support', q, [(flag, .95 if flag else .3, .01 if flag else .5) for flag in anchors])
        return deepcopy(baseline)
    def localize(values, p, cfg):
        calls.append(p['proposal_id'])
        p.update(source_method='four_observed_lines', side_coverage=[.95]*4, side_max_gap_fraction=[.01]*4)
        return p, {'status': 'localized_four_supported_ridges'}
    monkeypatch.setattr(module, 'native_window_proposals', search)
    monkeypatch.setattr(module, '_localize', localize)
    return baseline, events, anchors, calls


def test_recovers_observed_gap_with_exact_old_geometry_and_pixel_preservation(setup):
    old, _, _, calls = setup
    values = np.ones((100, 100), np.float32)
    result = module.transition_completion_proposals(values)
    assert result['proposals'][:2] == old['proposals']
    assert len(old['proposals']) == 2
    assert len(result['proposals']) == 3 and len(calls) == 1
    assert result['transition_completion']['retained_additions'] == 1
    assert result['native_completion']['old_measurement_pixels_lost'] == 0
    before = partition_measurement_support([polygon_to_mask(p['polygon_native_xy'], values.shape) for p in old['proposals']], margin_px=2)
    after = partition_measurement_support([polygon_to_mask(p['polygon_native_xy'], values.shape) for p in result['proposals']], margin_px=2)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a.interior, b.interior)
    assert np.all(values == 1)


def test_weak_initial_anchor_cannot_generate_a_panel(setup):
    old, _, anchors, calls = setup
    anchors[1] = False
    result = module.transition_completion_proposals(np.ones((100, 100)))
    assert result['proposals'] == old['proposals'] and not calls


def test_pool_overflow_withholds_all_additions_and_keeps_complete_baseline(setup):
    old, events, _, calls = setup
    events.append(panel(76, 'other')['polygon_native_xy'])
    result = module.transition_completion_proposals(np.ones((100, 100)), max_candidates=1)
    assert result['proposals'] == old['proposals'] and not calls
    assert result['transition_completion']['status'] == 'pool_budget_exceeded_baseline_only'


def test_nearly_identical_source_passes_do_not_conflict_as_two_modules(setup):
    _, events, _, calls = setup
    events.append((np.asarray(events[0])+[.1, 0]).tolist())
    result = module.transition_completion_proposals(np.ones((100, 100)))
    assert len(calls) == 1 and result['transition_completion']['retained_additions'] == 1


def test_no_new_reference_or_missing_native_edges_never_fill_grid(setup, monkeypatch):
    old, _, _, _ = setup
    monkeypatch.setattr(module, '_localize', lambda *a: (None, {'status': 'no_ridge'}))
    monkeypatch.setattr(module, 'localize_transition_quad', lambda *a: (None, {'status': 'no_transition'}))
    result = module.transition_completion_proposals(np.ones((100, 100)))
    assert result['proposals'] == old['proposals']
    assert result['transition_completion']['decision_counts']['fresh_four_edge_evidence_unresolved'] == 1


def test_four_step_warm_rectangle_cannot_be_accepted(setup, monkeypatch):
    old, _, _, _ = setup
    def false_patch(a, p, cfg):
        p.update(source_method='four_native_transitions', side_model_families=['step']*4)
        return p, {'status': 'localized_four_native_transitions'}
    monkeypatch.setattr(module, '_localize', false_patch)
    result = module.transition_completion_proposals(np.ones((100, 100)))
    assert result['proposals'] == old['proposals']


def test_isolated_candidate_still_needs_two_baseline_scale_references(setup):
    old, _, _, _ = setup
    old['proposals'].pop(0)
    result = module.transition_completion_proposals(np.ones((100, 100)))
    assert result['proposals'] == old['proposals']
    assert result['native_completion']['rejection_counts']['requires_two_baseline_scale_references_one_adjacent'] == 1


def test_mutating_localization_work_array_is_rejected(setup, monkeypatch):
    def mutate(a, p, cfg):
        a[0, 0] = 2
        p.update(source_method='four_observed_lines', side_coverage=[.95]*4, side_max_gap_fraction=[.01]*4)
        return p, {'status': 'supported'}
    monkeypatch.setattr(module, '_localize', mutate)
    values = np.ones((100, 100), np.float32)
    with pytest.raises(RuntimeError, match='changed native'):
        module.transition_completion_proposals(values)
    assert values[0, 0] == 1


@pytest.mark.parametrize('kwargs', [{'max_candidates': True}, {'max_candidates': 0}, {'max_candidates': 513}, {'margin_px': -1}, {'raw_supplement': 1}])
def test_invalid_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        module.transition_completion_proposals(np.ones((100, 100)), **kwargs)
