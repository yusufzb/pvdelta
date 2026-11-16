from copy import deepcopy

import pytest

from pvdelta.hotspot_coverage import diagnose_hotspot_coverage


def box(identifier, left, top, right, bottom):
    return {'region_id': identifier,
            'polygon_native_xy': [[left, top], [right, top], [right, bottom], [left, bottom]]}


def test_distinct_coverage_failures_do_not_hide_misses_or_mutate_inputs():
    refs = [box('outside', 0, 0, 5, 5), box('excluded', 20, 21, 21, 24),
            box('not_detected', 25, 25, 29, 29), box('low_iou', 40, 40, 42, 42),
            box('matched', 50, 50, 55, 55)]
    predictions = [box('broad_candidate', 39, 39, 47, 47), box('good_candidate', 50, 50, 55, 55)]
    envelopes = [box('panel', 20, 20, 60, 60)]
    original = deepcopy((refs, predictions, envelopes))
    result = diagnose_hotspot_coverage(refs, predictions, envelopes, native_shape=(100, 100),
                                      iou_threshold=.5)
    assert result['counts'] == {'no_proposed_envelope_overlap': 1,
                               'envelope_present_but_measurement_excluded': 1,
                               'measurement_present_no_candidate_overlap': 1,
                               'candidate_overlap_below_iou_gate': 1, 'matched': 1}
    assert result['scores']['false_negatives'] == 4
    assert result['references'][0]['frame_edge_distance_px'] == 0
    assert result['references'][-1]['fraction_in_measurement_support'] == 1
    assert not result['measurement_authorised']
    assert (refs, predictions, envelopes) == original


def test_every_shared_envelope_pixel_is_excluded_from_measurement():
    refs = [box('shared', 35, 30, 40, 35), box('safe', 23, 30, 28, 35)]
    envelopes = [box('left', 20, 20, 45, 60), box('right', 30, 20, 55, 60)]
    result = diagnose_hotspot_coverage(refs, [], envelopes, native_shape=(80, 80))
    shared, safe = result['references']
    assert shared['envelope_overlap_pixels'] == 25
    assert shared['measurement_overlap_pixels'] == 0
    assert safe['measurement_overlap_pixels'] == 25
    assert result['measurement_pixels'] < result['proposed_envelope_pixels']


def test_absent_geometry_is_a_miss_and_empty_labels_are_not_a_healthy_count():
    result = diagnose_hotspot_coverage([box('ref', 10, 10, 15, 15)], [], [], native_shape=(50, 50))
    assert result['counts'] == {'no_proposed_envelope_overlap': 1}
    assert result['scores']['false_negatives'] == 1
    assert result['measurement_pixels'] == result['proposed_envelope_pixels'] == 0
    result = diagnose_hotspot_coverage([], [box('candidate', 10, 10, 15, 15)], [], native_shape=(50, 50))
    assert result['counts'] == {} and result['scores']['false_positives'] == 1
    assert 'healthy' not in result


@pytest.mark.parametrize('invalid', ['duplicate', 'outside', 'nonfinite'])
def test_invalid_envelopes_are_rejected_before_coverage_is_reported(invalid):
    envelopes = [box('panel', 10, 10, 30, 30)]
    if invalid == 'duplicate':
        envelopes += deepcopy(envelopes)
    elif invalid == 'outside':
        envelopes[0]['polygon_native_xy'][0][0] = -1
    else:
        envelopes[0]['polygon_native_xy'][0][0] = float('nan')
    with pytest.raises(ValueError):
        diagnose_hotspot_coverage([], [], envelopes, native_shape=(50, 50))


@pytest.mark.parametrize('margin', [-1, True, 1.5])
def test_invalid_margin_is_rejected_even_when_no_envelopes_exist(margin):
    with pytest.raises(ValueError, match='margin_px'):
        diagnose_hotspot_coverage([], [], [], native_shape=(50, 50), margin_px=margin)


def test_subpixel_reference_is_explicitly_rejected_instead_of_claiming_coverage():
    with pytest.raises(ValueError, match='pixel centres'):
        diagnose_hotspot_coverage([box('tiny', .1, .1, .2, .2)], [], [], native_shape=(50, 50))


def test_one_to_one_competition_is_distinct_from_low_iou():
    refs = [box('ref1', 20, 20, 25, 25), box('ref2', 20, 20, 25, 25)]
    predictions = [box('candidate', 20, 20, 25, 25)]
    result = diagnose_hotspot_coverage(refs, predictions, [box('panel', 10, 10, 40, 40)],
                                      native_shape=(50, 50), iou_threshold=.5)
    assert result['counts'] == {'matched': 1, 'candidate_unmatched_one_to_one': 1}
    assert result['scores']['true_positives'] == 1 and result['scores']['false_negatives'] == 1


def test_numeric_coordinates_and_iterable_inputs_follow_the_evaluator_contract():
    ref = box('ref', 20, 20, 25, 25)
    ref['polygon_native_xy'] = [[str(v) for v in point] for point in ref['polygon_native_xy']]
    result = diagnose_hotspot_coverage(iter([ref]), iter([]), iter([box('panel', 10, 10, 40, 40)]),
                                      native_shape=(50, 50))
    assert result['references'][0]['frame_edge_distance_px'] == 20
    assert result['references'][0]['measurement_overlap_pixels'] == 25
