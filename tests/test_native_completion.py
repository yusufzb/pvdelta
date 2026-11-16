from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from pvdelta.config import DetectionConfig
from pvdelta.models import ThermalFrame
from pvdelta.native_completion import complete_native_proposals
from pvdelta.pipeline import analyze_annotations
from pvdelta.regions import ModuleAnnotation, polygon_to_mask
from pvdelta.support_partition import partition_measurement_support

SHAPE = (160, 340)


def test_explicit_manual_support_can_replace_adjacency_but_preserves_old_pixels():
    baseline=[panel(200),panel(245)];candidate=panel(20)
    support=polygon_to_mask(candidate['polygon_native_xy'],SHAPE)
    assert complete(baseline,[candidate])['proposals']==baseline
    revised=complete(baseline,[candidate],manual_panel_mask=support)
    assert len(revised['proposals'])==3 and revised['proposals'][:2]==baseline
    assert revised['proposals'][-1]['neighbour_ids']==[]
    assert all(np.array_equal(a.interior,b.interior) for a,b in zip(masks(baseline),masks(revised['proposals'])))


@pytest.mark.parametrize('problem',['unknown_pixel','no_scale','one_scale','missing_side','conflict'])
def test_manual_support_does_not_bypass_other_completion_gates(problem):
    baseline=[panel(200),panel(245)];candidates=[panel(20)]
    support=polygon_to_mask(candidates[0]['polygon_native_xy'],SHAPE)
    if problem=='unknown_pixel':support[50,30]=False
    elif problem=='no_scale':baseline=[]
    elif problem=='one_scale':baseline=baseline[:1]
    elif problem=='missing_side':candidates[0].update(source_method='four_observed_lines',side_coverage=[.7]*4,side_max_gap_fraction=[0.]*4)
    else:
        candidates.append(panel(21));support|=polygon_to_mask(candidates[-1]['polygon_native_xy'],SHAPE)
    assert complete(baseline,candidates,manual_panel_mask=support)['proposals']==baseline


@pytest.mark.parametrize('mask',[np.ones(SHAPE,dtype=np.uint8),np.ones((10,10),bool)])
def test_manual_support_grid_must_be_explicit_boolean_native_mask(mask):
    with pytest.raises(ValueError,match='Manual panel support'):
        complete([panel(200),panel(245)],[panel(20)],manual_panel_mask=mask)


def panel(x, y=20, identifier=None):
    return {'proposal_id': identifier or f'N{x}_{y}',
            'polygon_native_xy': [[x, y], [x+40, y], [x+40, y+90], [x, y+90]],
            'source_method': 'closed_contour', 'threshold_votes': 2,
            'proposal_origin': 'original_observation', 'neighbour_ids': []}


def result(proposals):
    return {'proposals': proposals, 'measurement_authorised': False,
            'local_window_search': {'preserve': 'all existing diagnostics'}}


def complete(baseline, alternative, **kwargs):
    return complete_native_proposals(result(baseline), result(alternative), native_shape=SHAPE, **kwargs)


def masks(proposals):
    return partition_measurement_support([
        polygon_to_mask(p['polygon_native_xy'], SHAPE) for p in proposals], margin_px=2)


def test_supported_observed_envelope_is_added_without_mutating_originals_or_metadata():
    baseline = result([panel(65), panel(110)])
    alternative = result([panel(20)])
    original = deepcopy((baseline, alternative))
    proposed = complete_native_proposals(baseline, alternative, native_shape=SHAPE)
    assert (baseline, alternative) == original
    assert proposed['proposals'][:2] == baseline['proposals']
    assert proposed['local_window_search'] == baseline['local_window_search']
    added = proposed['proposals'][-1]
    assert added['proposal_id'] == 'E0001' and added['proposal_origin'] == 'supported_native_completion'
    assert len(added['completion_support']['reference_polygons_native_xy']) == 2
    assert added['completion_support']['alternative_proposal_id'] == 'N20_20'
    assert proposed['native_completion']['retained_additions'] == 1
    assert not proposed['measurement_authorised'] and not added['measurement_authorised']


@pytest.mark.parametrize('references', [[], [panel(65)], [panel(65), panel(65, identifier='duplicate')],
                                        [panel(200), panel(245)]])
def test_missing_duplicate_or_distant_references_do_not_authorise_completion(references):
    proposed = complete(references, [panel(20)])
    assert proposed['proposals'] == references
    assert proposed['native_completion']['retained_additions'] == 0


def test_alternatives_cannot_bootstrap_a_new_row_or_supply_the_second_reference():
    baseline = [panel(155)]
    proposed = complete(baseline, [panel(20), panel(65), panel(110)])
    assert proposed['proposals'] == baseline
    assert proposed['native_completion']['retained_additions'] == 0


def test_conflicting_alternatives_are_both_withheld_independent_of_order():
    baseline = [panel(65), panel(110)]
    alternatives = [panel(20), panel(21)]
    first = complete(baseline, alternatives)
    second = complete(baseline, alternatives[::-1])
    assert first == second
    assert first['proposals'] == baseline
    assert first['native_completion']['rejection_counts'] == {
        'conflicting_alternative_envelopes_all_withheld': 2}


def test_old_interior_is_protected_even_when_geometric_overlap_is_below_five_percent():
    baseline = [panel(57), panel(102)]
    proposed = complete(baseline, [panel(20)])
    assert proposed['proposals'] == baseline
    assert proposed['native_completion']['rejection_counts'] == {'touches_existing_measurement_pixels': 1}


def test_excluded_frame_edges_may_overlap_but_old_interiors_stay_bitwise_identical():
    baseline = [panel(59), panel(104)]
    proposed = complete(baseline, [panel(20)])
    assert len(proposed['proposals']) == 3
    before, after = masks(baseline), masks(proposed['proposals'])
    for old, new in zip(before, after):
        np.testing.assert_array_equal(old.interior, new.interior)
    assert np.max(np.sum([s.interior for s in after], axis=0)) == 1
    assert proposed['native_completion']['old_measurement_pixels_lost'] == 0


@pytest.mark.parametrize('change', ['low_votes', 'missing_votes', 'weak_side', 'gap', 'missing_sides', 'unsupported'])
def test_weak_or_missing_observed_side_evidence_is_withheld(change):
    candidate = panel(20)
    if change == 'low_votes':
        candidate['threshold_votes'] = 1
    elif change == 'missing_votes':
        del candidate['threshold_votes']
    elif change == 'unsupported':
        candidate['source_method'] = 'inferred_grid'
    else:
        candidate.update(source_method='four_observed_lines', side_coverage=[1]*4,
                         side_max_gap_fraction=[0]*4)
        if change == 'weak_side':
            candidate['side_coverage'][0] = .79
        elif change == 'gap':
            candidate['side_max_gap_fraction'][0] = .19
        else:
            del candidate['side_coverage']
    proposed = complete([panel(65), panel(110)], [candidate])
    assert proposed['native_completion']['rejection_counts'] == {'insufficient_observed_side_evidence': 1}


def test_four_observed_line_sides_are_supported_without_contour_votes():
    candidate = panel(20)
    candidate.update(source_method='observed_lines_and_native_ridges', side_coverage=[.9]*4,
                     side_max_gap_fraction=[.1]*4)
    del candidate['threshold_votes']
    assert complete([panel(65), panel(110)], [candidate])['native_completion']['retained_additions'] == 1


def test_unobserved_baseline_shapes_cannot_supply_scale_votes():
    baseline = [panel(65), panel(110)]
    for p in baseline:
        p['source_method'] = 'inferred_grid'
    assert complete(baseline, [panel(20)])['proposals'] == baseline


def test_ambiguous_baseline_shapes_with_no_measurement_interior_cannot_vote():
    baseline = [panel(65), panel(65, identifier='ambiguous_duplicate'), panel(110)]
    assert complete(baseline, [panel(20)])['proposals'] == baseline


def test_duplicate_alternative_does_not_shift_or_rename_existing_geometry():
    baseline = [panel(65), panel(110)]
    proposed = complete(baseline, [panel(66)])
    assert proposed['proposals'] == baseline
    assert proposed['native_completion']['rejection_counts'] == {'duplicates_existing_geometry': 1}


def test_added_ids_cannot_collide_with_existing_ids():
    baseline = [panel(65, identifier='E0001'), panel(110)]
    assert complete(baseline, [panel(20)])['proposals'][-1]['proposal_id'] == 'E0002'


@pytest.mark.parametrize('change', ['duplicate_id', 'outside', 'nonfinite', 'zero_area'])
def test_invalid_geometry_fails_instead_of_silently_approving_it(change):
    alternative = [panel(20)]
    if change == 'duplicate_id':
        alternative += deepcopy(alternative)
    elif change == 'outside':
        alternative[0]['polygon_native_xy'][0][0] = -1
    elif change == 'nonfinite':
        alternative[0]['polygon_native_xy'][0][0] = float('nan')
    else:
        alternative[0]['polygon_native_xy'] = [[20, 20]]*4
    with pytest.raises(ValueError):
        complete([panel(65), panel(110)], alternative)


@pytest.mark.parametrize('kwargs', [{'margin_px': -1}, {'margin_px': True}, {'max_overlap_iou': .1},
                                  {'max_overlap_iou': True}, {'max_overlap_iou': float('nan')}])
def test_unsafe_or_invalid_support_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        complete([], [], **kwargs)


def test_original_findings_including_a_boundary_hotspot_are_exactly_preserved():
    baseline = [panel(59), panel(104)]
    proposed = complete(baseline, [panel(20)])
    values = np.full(SHAPE, 25., dtype=np.float32)
    values[45:50, 62:67] = 40  # Near the original effective measurement boundary.
    values[70:75, 120:125] = 35
    original_values = values.copy()
    frame = ThermalFrame('test', 'native-source', 'source-sha', values, np.isfinite(values), SHAPE[1], SHAPE[0])

    def run(proposals):
        support = masks(proposals)
        annotations = [ModuleAnnotation(p['proposal_id'], tuple(tuple(v) for v in p['polygon_native_xy']))
                       for p in proposals]
        return analyze_annotations(frame, annotations, config=DetectionConfig(), direction='hot',
            module_masks={p['proposal_id']: s.interior for p, s in zip(proposals, support)})

    before = run(baseline)
    after = run(proposed['proposals'])
    assert sum(len(m.findings) for m in before) == 2
    assert [asdict(m) for m in after[:len(baseline)]] == [asdict(m) for m in before]
    np.testing.assert_array_equal(values, original_values)
