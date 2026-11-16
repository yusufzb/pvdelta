import hashlib
from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip('cv2')
from pvdelta.native_completion import complete_native_proposals
from pvdelta.native_observed_candidates import observed_native_candidates
from pvdelta.native_search_trace import begin_trace
from pvdelta.regions import polygon_to_mask
from pvdelta.support_partition import partition_measurement_support


def quad(x):
    return np.array([[x,20], [x+40,20], [x+40,110], [x,110]], float)


def mock_search(monkeypatch, polygons, *, support=None, mutate=False):
    calls = []
    def search(values, **kwargs):
        calls.append(values)
        trace = begin_trace(values, support_mode='segments', local_contrast=True,
                            pair_budget=kwargs['line_config'].max_pair_product)
        if len(calls) == 1:
            for polygon in polygons:
                trace.candidate('four_sides_passed', polygon,
                                support if support is not None else [(True,.9,.1)]*4)
        if mutate:
            values[0,0] += 1
        trace.finish({'pair_search': {'search_complete': True}, 'line_status': 'test',
                      'line_diagnostics': {}})
        return {'proposals': []}
    monkeypatch.setattr('pvdelta.native_observed_candidates._observed_quads', lambda v: ([],0))
    monkeypatch.setattr('pvdelta.native_observed_candidates._line_pass', search)
    return calls


def values():
    return np.arange(160*340, dtype=float).reshape(160,340)


def test_single_observed_quad_is_retained_without_inventing_a_group(monkeypatch):
    calls = mock_search(monkeypatch, [quad(20)])
    native = values()
    old = native.copy()
    found = observed_native_candidates(native, max_pair_product=2_000_000)
    assert len(calls) == 5 and all(np.shares_memory(c, native) for c in calls)
    assert len(found['proposals']) == 1 and 'groups' not in found
    p = found['proposals'][0]
    assert p['polygon_native_xy'] == quad(20).tolist()
    assert p['original_side_support'] == [[True,.9,.1]]*4
    assert p['semantic_panel_membership'] == 'unverified' and not p['measurement_authorised']
    assert all(p['pair_budget'] == 2_000_000 for p in found['search_passes'])
    assert found['native_temperature_sha256'] == hashlib.sha256(native.tobytes()).hexdigest()
    np.testing.assert_array_equal(native, old)


def test_same_geometry_is_deduplicated_and_spatial_ids_are_stable(monkeypatch):
    mock_search(monkeypatch, [quad(110), quad(20), quad(21)])
    a = observed_native_candidates(values())
    mock_search(monkeypatch, [quad(21), quad(110), quad(20)])
    b = observed_native_candidates(values())
    assert a == b and [p['proposal_id'] for p in a['proposals']] == ['O0001','O0002']


def test_closed_contour_votes_preserve_the_existing_precedence(monkeypatch):
    mock_search(monkeypatch, [quad(21)])
    monkeypatch.setattr('pvdelta.native_observed_candidates._observed_quads',
        lambda v: ([{'quad':quad(20), 'votes':{0,1}}], 2))
    found = observed_native_candidates(values())
    assert len(found['proposals']) == 1
    assert found['proposals'][0]['source_method'] == 'closed_contour'
    assert found['proposals'][0]['threshold_votes'] == 2


def test_no_four_cell_cycle_does_not_bypass_two_existing_scale_references(monkeypatch):
    mock_search(monkeypatch, [quad(20)])
    found = observed_native_candidates(values())
    def baseline(xs):
        return {'proposals':[{'proposal_id':f'N{x}', 'polygon_native_xy':quad(x).tolist(),
            'source_method':'closed_contour', 'threshold_votes':2} for x in xs]}
    original = baseline([65,110])
    before = deepcopy(original)
    proposed = complete_native_proposals(original, found, native_shape=(160,340))
    assert original == before and proposed['proposals'][:2] == before['proposals']
    assert len(proposed['proposals']) == 3
    first = partition_measurement_support([polygon_to_mask(p['polygon_native_xy'], (160,340))
        for p in before['proposals']], margin_px=2)
    after = partition_measurement_support([polygon_to_mask(p['polygon_native_xy'], (160,340))
        for p in proposed['proposals']], margin_px=2)
    assert all(np.array_equal(a.interior,b.interior) for a,b in zip(first,after))
    for xs in ([], [65], [200,245]):
        old = baseline(xs)
        assert complete_native_proposals(old,found,native_shape=(160,340))['proposals'] == old['proposals']


def test_conflicting_observed_candidates_are_withheld_by_completion(monkeypatch):
    # Overlap below extractor dedup .65, but above completion's stricter .05.
    mock_search(monkeypatch, [quad(11), quad(21)])
    found = observed_native_candidates(values())
    baseline = {'proposals':[{'proposal_id':f'N{x}', 'polygon_native_xy':quad(x).tolist(),
        'source_method':'closed_contour', 'threshold_votes':2} for x in (60,105)]}
    result = complete_native_proposals(baseline, found, native_shape=(160,340))
    assert result['proposals'] == baseline['proposals']
    assert result['native_completion']['rejection_counts'] == {
        'conflicting_alternative_envelopes_all_withheld':2}


@pytest.mark.parametrize('support', [[(False,.79,.19)]*4, [(True,float('nan'),.1)]*4])
def test_missing_or_invalid_sides_cannot_enter_the_pool(monkeypatch, support):
    mock_search(monkeypatch, [quad(20)], support=support)
    with pytest.raises(ValueError, match='Four observed'):
        observed_native_candidates(values())


def test_event_overflow_withholds_the_entire_pool(monkeypatch):
    mock_search(monkeypatch, [quad(20)]*8193)
    found = observed_native_candidates(values())
    assert found['status'] == 'observed_candidate_budget_exceeded' and found['proposals'] == []


def test_deduplicated_quad_overflow_does_not_return_a_ranked_subset(monkeypatch):
    mock_search(monkeypatch, [])
    monkeypatch.setattr('pvdelta.native_observed_candidates._observed_quads',
        lambda v: ([{'quad':quad(x), 'votes':{0,1}} for x in range(513)], 513))
    monkeypatch.setattr('pvdelta.native_observed_candidates.convex_iou', lambda a,b: 0)
    found = observed_native_candidates(values())
    assert found['status'] == 'observed_candidate_budget_exceeded' and found['proposals'] == []


@pytest.mark.parametrize('budget', [True, 0, 2_000_001, 250_000.])
def test_bad_budget_fails_before_extraction(monkeypatch, budget):
    calls = mock_search(monkeypatch, [])
    with pytest.raises(ValueError):
        observed_native_candidates(values(), max_pair_product=budget)
    assert not calls


@pytest.mark.parametrize('native', [np.ones((20,40)), np.ones(5), np.full((40,40),np.inf),
    np.ones((40,80))[:,::2], np.ones((40,40,3)), np.array([['x']]), np.ones((40,40),bool)])
def test_invalid_input_fails_before_extraction(native):
    with pytest.raises(ValueError):
        observed_native_candidates(native)


def test_native_mutation_fails_explicitly(monkeypatch):
    mock_search(monkeypatch, [quad(20)], mutate=True)
    with pytest.raises(RuntimeError, match='changed'):
        observed_native_candidates(values())


def test_real_flat_grid_fails_explicitly_without_forced_geometry():
    native = np.full((80,100),25.)
    before = native.copy()
    with pytest.raises(ValueError, match='dynamic range'):
        observed_native_candidates(native)
    np.testing.assert_array_equal(native,before)
