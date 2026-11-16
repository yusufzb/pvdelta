from copy import deepcopy

import numpy as np
import pytest

from pvdelta.panel_pair_check import observed_panel_pair_check


def panels(rows=4, columns=4, shift=(0, 0)):
    result = {'proposals': []}
    for y in range(rows):
        for x in range(columns):
            quad = np.array([[20+60*x, 20+40*y], [70+60*x, 20+40*y],
                             [70+60*x, 50+40*y], [20+60*x, 50+40*y]], float) + shift
            result['proposals'].append({'proposal_id': f'p{y}-{x}',
                'polygon_native_xy': quad.tolist(), 'source_method': 'closed_contour',
                'threshold_votes': 3})
    return result


def check(source, target, matrix=None):
    return observed_panel_pair_check(source, target, np.eye(3) if matrix is None else matrix,
                                      native_shape=(512, 640))


def test_exact_shift_passes_controls_but_never_confirms_physical_identity_or_transfers_pixels():
    a, b = panels(), panels(shift=(2, 3))
    old_a, old_b = deepcopy(a), deepcopy(b)
    report = check(a, b, [[1, 0, 2], [0, 1, 3], [0, 0, 1]])
    assert len(report['matches']) == 16 and report['heldout_p95_px'] < 1e-10
    assert report['control_check_passed'] is True
    assert not any(report[k] for k in ('same_panel_verified', 'measurement_authorised', 'glare_clearance'))
    assert report['status'] == 'withheld' and 'transform' not in report
    assert a == old_a and b == old_b


def test_inconsistent_controls_are_not_trimmed_to_pass():
    a, b = panels(), panels()
    b['proposals'][5]['polygon_native_xy'] = (np.asarray(b['proposals'][5]['polygon_native_xy'])+[8, 0]).tolist()
    report = check(a, b)
    assert len(report['matches']) == 16 and len(report['residuals_px']) == 16
    assert report['heldout_p95_px'] > 2
    assert not report['control_check_passed']
    assert report['reasons'] == ['held_out_panel_residual_too_large']


def test_single_axis_repetition_is_degenerate_not_a_panel_plane():
    report = check(panels(rows=1, columns=8), panels(rows=1, columns=8))
    assert not report['control_check_passed']
    assert report['reasons'] == ['degenerate_panel_control_layout']


def test_no_observed_boundaries_or_too_few_pairs_is_unknown():
    for count in (0, 1, 2):
        a = panels(rows=1, columns=count)
        report = check(a, a)
        assert report['reasons'] == ['insufficient_observed_panel_pairs']
    a = panels()
    for proposal in a['proposals']:
        proposal['threshold_votes'] = 1
    assert check(a, panels())['matches'] == []


def test_duplicate_target_candidates_are_ambiguous_not_double_counted():
    a, b = panels(), panels()
    b['proposals'].append(dict(deepcopy(b['proposals'][0]), proposal_id='duplicate'))
    report = check(a, b)
    assert len(report['matches']) == 15
    assert len({m['source_id'] for m in report['matches']}) == 15
    assert len({m['target_id'] for m in report['matches']}) == 15


@pytest.mark.parametrize('offset', [0., 1.])
@pytest.mark.parametrize('reverse', [False, True])
def test_competing_source_proposals_cannot_claim_same_target(offset, reverse):
    a, b = panels(), panels()
    competing = deepcopy(a['proposals'][0])
    competing['proposal_id'] = 'competing-source'
    competing['polygon_native_xy'] = (np.array(competing['polygon_native_xy'])+[offset, 0]).tolist()
    a['proposals'].append(competing)
    if reverse:
        a['proposals'].reverse()
    result = check(a, b)
    assert len(result['matches']) == 15
    assert not any(m['target_id'] == 'p0-0' for m in result['matches'])


def test_reported_centres_are_projective_rectangle_centres():
    from pvdelta.alignment import transform_points
    from pvdelta.tracking import quadrilateral_center

    a, b = panels(), panels()
    matrix = np.array([[1, .1, 0], [.05, 1, 0], [.001, .0015, 1]])
    for proposal in b['proposals']:
        proposal['polygon_native_xy'] = transform_points(np.array(proposal['polygon_native_xy']), matrix).tolist()
    report = check(a, b, matrix)
    assert len(report['matches']) == 16
    target_by_id = {p['proposal_id']: p for p in b['proposals']}
    for match in report['matches']:
        target = target_by_id[match['target_id']]['polygon_native_xy']
        np.testing.assert_allclose(match['target_center_xy'], quadrilateral_center(target), atol=1e-10)
        projected = transform_points(np.array([match['source_center_xy']]), matrix)[0]
        np.testing.assert_allclose(match['target_center_xy'], projected, atol=1e-10)


@pytest.mark.parametrize('matrix', [np.zeros((3, 3)), np.eye(2), [[1, 0, 0], [0, 1, 0], [0, 0, np.nan]]])
def test_invalid_initializer_fails(matrix):
    with pytest.raises(ValueError):
        check(panels(), panels(), matrix)


def test_initializer_projective_pole_fails_instead_of_creating_a_false_polygon():
    with pytest.raises(ValueError, match='pole'):
        check(panels(), panels(), [[1, 0, 0], [0, 1, 0], [1, 0, -30]])
