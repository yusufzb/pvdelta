from copy import deepcopy

import numpy as np
import pytest

from pvdelta.native_observed_completion import complete_observed_surface_gaps
from pvdelta.native_search_trace import begin_trace
from pvdelta.regions import polygon_to_mask
from pvdelta.support_partition import partition_measurement_support


def panel(x):
    return {'proposal_id': f'N{x}', 'polygon_native_xy':
            [[x, 20], [x+40, 20], [x+40, 110], [x, 110]],
            'source_method': 'closed_contour', 'threshold_votes': 2}


def setup(monkeypatch, candidates=None, stage='four_sides_passed'):
    baseline = {'proposals': [panel(65), panel(110)]}
    values = np.arange(160*340, dtype=np.float32).reshape(160, 340)/1000
    mask = np.ones(values.shape, bool)
    candidates = [panel(20)] if candidates is None else candidates

    def search(grid, **kwargs):
        trace = begin_trace(grid, support_mode='hybrid', local_contrast=True, pair_budget=250000)
        for p in candidates:
            trace.candidate(stage, p['polygon_native_xy'], [(True, 1., 0.)]*4)
        # Simulate a larger envelope having suppressed these candidates during deduplication.
        return {'proposals': []}
    monkeypatch.setattr('pvdelta.native_observed_completion.native_line_proposals', search)
    return values, baseline, mask


def run(values, baseline, mask, **kwargs):
    return complete_observed_surface_gaps(values, baseline, panel_mask=mask,
                                         windows=[[0, 0, 340, 160]], **kwargs)


def test_recovers_observed_quad_hidden_by_dedup_without_changing_existing_measurements(monkeypatch):
    values, baseline, mask = setup(monkeypatch)
    old, original = deepcopy(baseline), values.copy()
    result = run(values, baseline, mask)
    assert baseline == old and np.array_equal(values, original)
    assert result['proposals'][:2] == baseline['proposals']
    assert len(result['proposals']) == 3 and not result['measurement_authorised']
    before = partition_measurement_support([
        polygon_to_mask(p['polygon_native_xy'], values.shape) for p in baseline['proposals']],
        margin_px=2)
    after = partition_measurement_support([
        polygon_to_mask(p['polygon_native_xy'], values.shape) for p in result['proposals']],
        margin_px=2)
    assert all(np.array_equal(a.interior, b.interior) for a, b in zip(before, after))


def test_one_background_or_unknown_pixel_withholds_candidate(monkeypatch):
    values, baseline, mask = setup(monkeypatch)
    mask[50, 30] = False
    result = run(values, baseline, mask)
    assert result['proposals'] == baseline['proposals']
    assert result['observed_surface_completion']['filter_counts']['outside_manual_panel'] == 1


def test_side_rejected_quad_cannot_be_promoted_by_manual_surface(monkeypatch):
    values, baseline, mask = setup(monkeypatch, stage='side_support')
    assert run(values, baseline, mask)['proposals'] == baseline['proposals']


def test_overflow_is_explicit_and_withholds_every_addition(monkeypatch):
    values, baseline, mask = setup(monkeypatch, candidates=[panel(20), panel(20)])
    result = run(values, baseline, mask, max_candidates=1)
    assert result['proposals'] == baseline['proposals']
    assert result['observed_surface_completion']['status'] == 'pool_budget_exceeded_baseline_only'


def test_duplicate_passes_do_not_make_conflicting_alternatives(monkeypatch):
    values, baseline, mask = setup(monkeypatch, candidates=[panel(20), panel(20)])
    assert len(run(values, baseline, mask)['proposals']) == 3


def test_genuinely_conflicting_alternatives_are_all_withheld(monkeypatch):
    values, baseline, mask = setup(monkeypatch, candidates=[panel(20), panel(17)])
    result = run(values, baseline, mask)
    assert result['proposals'] == baseline['proposals']
    assert result['native_completion']['rejection_counts'] == {
        'conflicting_alternative_envelopes_all_withheld': 2}


def test_no_bootstrapping_without_two_existing_references(monkeypatch):
    values, baseline, mask = setup(monkeypatch)
    baseline['proposals'].pop()
    assert run(values, baseline, mask)['proposals'] == baseline['proposals']


def test_isolated_option_requires_manual_surface_and_keeps_two_scale_references(monkeypatch):
    values,_,mask=setup(monkeypatch)
    baseline={'proposals':[panel(200),panel(245)]}
    assert run(values,baseline,mask)['proposals']==baseline['proposals']
    assert len(run(values,baseline,mask,allow_isolated=True)['proposals'])==3
    mask[50,30]=False
    assert run(values,baseline,mask,allow_isolated=True)['proposals']==baseline['proposals']
    with pytest.raises(ValueError,match='allow_isolated'):
        run(values,baseline,mask,allow_isolated=1)


@pytest.mark.parametrize('problem', ['numeric_mask', 'wrong_mask_shape', 'nan', 'noncontiguous'])
def test_invalid_inputs_are_rejected(monkeypatch, problem):
    values, baseline, mask = setup(monkeypatch)
    if problem == 'numeric_mask': mask = mask.astype(np.uint8)
    elif problem == 'wrong_mask_shape': mask = mask[:-1]
    elif problem == 'nan': values[0, 0] = np.nan
    else: values = values[:, ::-1]
    with pytest.raises(ValueError): run(values, baseline, mask)


def test_frame_without_panel_windows_preserves_baseline_without_search(monkeypatch):
    values, baseline, mask = setup(monkeypatch)
    def unexpected_search(*args, **kwargs):
        raise AssertionError('No eligible surface: search must not run')
    monkeypatch.setattr('pvdelta.native_observed_completion.native_line_proposals', unexpected_search)
    result = complete_observed_surface_gaps(values, baseline, panel_mask=mask, windows=[])
    assert result['proposals'] == baseline['proposals']
    assert result['observed_surface_completion']['status'] == 'no_eligible_windows_baseline_only'


@pytest.mark.parametrize('windows', [[[-1, 0, 340, 160]], [[0, 0, 341, 160]],
                                    [[0, 0, 12, 12]], [[0., 0, 340, 160]]])
def test_invalid_windows_are_rejected(monkeypatch, windows):
    values, baseline, mask = setup(monkeypatch)
    with pytest.raises(ValueError):
        complete_observed_surface_gaps(values, baseline, panel_mask=mask, windows=windows)


def test_raw_supplement_runs_after_nonempty_enhanced_search_and_preserves_baseline(monkeypatch):
    values, baseline, mask = setup(monkeypatch, candidates=[])
    calls = []

    def enhanced(grid, **kwargs):
        calls.append('enhanced')
        return deepcopy(baseline)

    def raw(grid, **kwargs):
        assert kwargs['local_contrast'] is False
        assert kwargs['support_mode'] == 'hybrid'
        assert np.shares_memory(values, grid)
        calls.append('raw')
        trace = begin_trace(grid, support_mode='hybrid', local_contrast=False, pair_budget=250000)
        trace.candidate('four_sides_passed', panel(20)['polygon_native_xy'], [(True, 1., 0.)]*4)
        return {'proposals': []}

    monkeypatch.setattr('pvdelta.native_observed_completion.native_line_proposals', enhanced)
    monkeypatch.setattr('pvdelta.native_observed_completion._line_pass', raw)
    assert run(values, baseline, mask)['proposals'] == baseline['proposals']
    assert calls == ['enhanced']
    calls.clear()
    result = run(values, baseline, mask, raw_supplement=True)
    assert calls == ['enhanced', 'raw']
    assert result['proposals'][:2] == baseline['proposals']
    assert len(result['proposals']) == 3
    assert result['proposals'][-1]['observed_pass']['local_contrast'] is False


@pytest.mark.parametrize('background,stage,budget', [(True,'four_sides_passed',512),
    (False,'side_support',512), (False,'four_sides_passed',1)])
def test_raw_supplement_cannot_bypass_mask_side_or_shared_budget(monkeypatch, background, stage, budget):
    values, baseline, mask = setup(monkeypatch, candidates=[])
    if background:
        mask[50, 30] = False

    def raw(grid, **kwargs):
        trace = begin_trace(grid, support_mode='hybrid', local_contrast=False, pair_budget=250000)
        for _ in range(2):
            trace.candidate(stage, panel(20)['polygon_native_xy'], [(True, 1., 0.)]*4)

    monkeypatch.setattr('pvdelta.native_observed_completion._line_pass', raw)
    result = run(values, baseline, mask, raw_supplement=True, max_candidates=budget)
    assert result['proposals'] == baseline['proposals']
    if budget == 1:
        assert result['observed_surface_completion']['status'] == 'pool_budget_exceeded_baseline_only'


def test_raw_supplement_flag_requires_boolean(monkeypatch):
    values, baseline, mask = setup(monkeypatch)
    with pytest.raises(ValueError, match='boolean'):
        run(values, baseline, mask, raw_supplement='false')
