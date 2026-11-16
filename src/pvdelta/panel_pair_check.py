"""Held-out observed-panel controls for a coarse cross-frame initializer.

Geometry consistency is not physical identity, glare evidence or calibration.
No temperatures, hot-pixel locations, CAD or reference annotations are inputs.
"""
from __future__ import annotations

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.native_completion import _observed_sides, _validate
from pvdelta.tracking import quadrilateral_center


def observed_panel_pair_check(source, target, initial_source_to_target, *, native_shape):
    """Check mutual unambiguous panel-centre pairs with four module-held-out folds.

    All fit/check pairs are chosen before fitting. No residual trimming, corner
    splitting of one module or selecting a best fold. A coarse ground-feature
    homography is only an initializer, never accepted as a panel transform.
    """
    _validate(source, native_shape)
    _validate(target, native_shape)
    matrix = np.asarray(initial_source_to_target, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or abs(np.linalg.det(matrix)) < 1e-12):
        raise ValueError('Finite invertible 3x3 initializer required')
    observed_a = [p for p in source['proposals'] if _observed_sides(p)]
    observed_b = [p for p in target['proposals'] if _observed_sides(p)]
    config = {'min_initial_iou': .5, 'min_iou_margin': .25,
              'max_initial_center_error_px': 12., 'min_pairs': 8,
              'folds': 4, 'max_heldout_p95_px': 2.}
    report = {'schema': 'pvdelta.observed-panel-pair-check/v1', 'config': config,
              'observed_source_panels': len(observed_a), 'observed_target_panels': len(observed_b),
              'matches': [], 'matching_decisions': [],
              'center_method': 'opposite_diagonal_intersection',
              'heldout_p95_px': None, 'residuals_px': [],
              'control_check_passed': False, 'same_panel_verified': False,
              'measurement_authorised': False, 'glare_clearance': False,
              'status': 'withheld', 'reasons': []}
    moved = []
    for proposal in observed_a:
        q = np.asarray(proposal['polygon_native_xy'], dtype=float)
        xyw = np.column_stack((q, np.ones(4))) @ matrix.T
        if (np.abs(xyw[:, 2]) < 1e-9).any() or np.ptp(np.sign(xyw[:, 2])):
            raise ValueError('Initializer crosses a projective pole on panel geometry')
        moved.append(xyw[:, :2]/xyw[:, 2:])
    if len(observed_a) < 2 or len(observed_b) < 2:
        report['reasons'] = ['insufficient_observed_panel_pairs']
        return report
    scores = np.asarray([[convex_iou(q, t['polygon_native_xy']) for t in observed_b] for q in moved])
    for i, proposal in enumerate(observed_a):
        order = np.argsort(scores[i], kind='stable')[::-1]
        j = int(order[0])
        source_margin = float(scores[i, j]-scores[i, order[1]])
        target_margin = float(scores[i, j]-np.delete(scores[:, j], i).max())
        decision = {'source_id': proposal['proposal_id'],
            'target_id': observed_b[j]['proposal_id'], 'initial_iou': float(scores[i, j]),
            'source_iou_margin': source_margin, 'target_iou_margin': target_margin,
            'status': 'pending'}
        report['matching_decisions'].append(decision)
        if scores[i, j] < config['min_initial_iou']:
            decision['status'] = 'insufficient_overlap'
            continue
        if source_margin < config['min_iou_margin']:
            decision['status'] = 'ambiguous_target_proposals'
            continue
        # A target's argmax alone picks one source even when competing sources
        # tie (or nearly tie). Require the SAME separation in both directions.
        if target_margin < config['min_iou_margin']:
            decision['status'] = 'competing_source_proposals'
            continue
        a = np.asarray(quadrilateral_center(proposal['polygon_native_xy']))
        b = np.asarray(quadrilateral_center(observed_b[j]['polygon_native_xy']))
        projected = np.asarray(quadrilateral_center(moved[i]))
        decision['initial_center_error_px'] = float(np.linalg.norm(projected-b))
        if decision['initial_center_error_px'] > config['max_initial_center_error_px']:
            decision['status'] = 'center_correction_exceeds_bound'
            continue
        decision['status'] = 'mutual_unambiguous_geometry_pair'
        report['matches'].append({'source_id': proposal['proposal_id'],
            'target_id': observed_b[j]['proposal_id'], 'source_center_xy': a.tolist(),
            'target_center_xy': b.tolist(), 'initial_iou': float(scores[i, j])})
    report['matches'].sort(key=lambda r: (*r['target_center_xy'][::-1], r['target_id']))
    count = len(report['matches'])
    if count < config['min_pairs']:
        report['reasons'] = ['insufficient_observed_panel_pairs']
        return report
    a = np.asarray([r['source_center_xy'] for r in report['matches']])
    b = np.asarray([r['target_center_xy'] for r in report['matches']])
    design = np.column_stack((a, np.ones(count)))
    residuals = np.empty(count)
    for fold in range(config['folds']):
        check = np.arange(count) % config['folds'] == fold
        fit, _, rank, _ = np.linalg.lstsq(design[~check], b[~check], rcond=None)
        if rank < 3:
            report['reasons'] = ['degenerate_panel_control_layout']
            return report
        residuals[check] = np.linalg.norm(design[check] @ fit-b[check], axis=1)
    report['residuals_px'] = residuals.tolist()
    report['heldout_p95_px'] = float(np.percentile(residuals, 95))
    report['control_check_passed'] = report['heldout_p95_px'] <= config['max_heldout_p95_px']
    report['reasons'] = (['physical_identity_and_local_coverage_unverified']
                         if report['control_check_passed'] else ['held_out_panel_residual_too_large'])
    # Even perfect controls alone cannot establish the selected panel's identity,
    # local coverage or pixel-ray angle. Do not publish a transferred polygon.
    return report
