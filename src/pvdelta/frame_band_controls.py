"""Strict, per-sensor frame-band observations, separate from masks and identity."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from pvdelta.junction_registration import (
    _edge_pair_centres,
    _gradients,
    _paired_ridges,
    _response,
)


def observe_frame_band_corner(gray, seed_xy, directions, lengths, *, radius, minimum=15.):
    """Intersect two independently supported band CENTRES; no single-edge fallback.

    Only this sensor's original structural pixels, seed and local module vectors
    are inputs. No opposite-sensor observations or fitted mapping may move the
    search. Three inward spatial strips each need >=60% paired-edge support.
    RGB brightness or thermal palette colours are not temperature measurements.
    """
    return _observe_corner(_gradients(gray), seed_xy, directions, lengths,
                           radius=radius, minimum=minimum)


def _observe_corner(gradients, seed_xy, directions, lengths, *, radius, minimum=15.):
    import cv2

    seed, vectors, span = np.asarray(seed_xy, float), np.asarray(directions, float), np.asarray(lengths, float)
    if (seed.shape != (2,) or vectors.shape != (2, 2) or span.shape != (2,)
            or not np.isfinite(np.r_[seed, vectors.ravel(), span]).all()
            or (span < 16).any() or (np.linalg.norm(vectors, axis=1) <= 0).any()
            or type(radius) is not int or not 2 <= radius <= 24
            or not np.isfinite(minimum) or minimum <= 0):
        raise ValueError('Finite local corner vectors, lengths>=16 and bounded radius required')
    gx, gy = gradients
    result = {'status': 'unresolved', 'reason': None, 'seed_xy': seed.tolist(),
        'observed_xy': None, 'sides': [], 'single_edge_fallback': False,
        'source_only': True, 'measurement_authorised': False}
    lines = []
    half_width = max(2, min(8, round(float(span.min())*.06)))
    offsets = np.arange(-radius-half_width, radius+half_width+1, dtype=float)
    for direction, length in zip(vectors, span, strict=True):
        tangent = direction/np.linalg.norm(direction)
        normal = np.array([-tangent[1], tangent[0]])
        # Stay on the observed inward module arm, away from the intersection.
        along = np.linspace(.12, .72, 48)*length
        scan = seed+along[:, None, None]*tangent+offsets[None, :, None]*normal
        inside = ((scan >= 2).all(axis=2) & (scan[:, :, 0] < gx.shape[1]-2)
                  & (scan[:, :, 1] < gx.shape[0]-2))
        if not inside.all():
            result['reason'] = 'band_search_outside_source'; return result
        signed = _response(gx, gy, scan, normal, signed=True)
        ridge = _paired_ridges(signed, half_width)
        profile = np.percentile(ridge, 35, axis=0)
        best = int(np.argmax(profile))
        side = {'status': 'unresolved', 'paired_gradient_score': float(profile[best]),
                'half_width_px': half_width}
        result['sides'].append(side)
        if best in (0, len(profile)-1) or profile[best] < minimum:
            result['reason'] = 'paired_edges_unresolved'; return result
        alternative = np.abs(np.arange(len(profile))-best) > half_width+1
        if alternative.any() and profile[alternative].max() >= .85*profile[best]:
            result['reason'] = 'competing_parallel_bands'; return result
        centre = best-radius
        centres, valid = _edge_pair_centres(signed, offsets, centre, half_width, minimum)
        fractions = [float(v.mean()) for v in np.array_split(valid, 3)]
        side.update(paired_support_fraction=float(valid.mean()), spatial_strip_support=fractions)
        if min(fractions) < .6:
            result['reason'] = 'paired_support_not_continuous'; return result
        points = seed+along[:, None]*tangent+centres[:, None]*normal
        target = np.float32(points[valid])
        vx, vy, x0, y0 = cv2.fitLine(target, cv2.DIST_HUBER, 0, .01, .01).ravel()
        n = np.array([-vy, vx], float); constant = float(n @ [x0, y0])
        residual = float(np.percentile(np.abs(target @ n-constant), 80))
        side['p80_line_residual_px'] = residual
        if abs(np.dot([vx, vy], tangent)) < .985 or residual > 1.5:
            result['reason'] = 'band_line_consensus_unresolved'; return result
        side['status'] = 'paired_band_supported'
        lines.append((n, constant))
    matrix = np.asarray([n for n, _ in lines])
    if abs(np.linalg.det(matrix)) < .3:
        result['reason'] = 'degenerate_band_intersection'; return result
    observed = np.linalg.solve(matrix, [d for _, d in lines])
    if (np.linalg.norm(observed-seed) > radius or (observed < 3).any()
            or observed[0] >= gx.shape[1]-3 or observed[1] >= gx.shape[0]-3):
        result['reason'] = 'band_intersection_outside_bound'; return result
    result.update(status='observed_paired_band_corner', observed_xy=observed.tolist(),
                  displacement_px=float(np.linalg.norm(observed-seed)), reason=None)
    return result


def build_frame_band_controls(native_gray, rgb_gray, matches, split):
    """Observe both sensors separately with frozen module fit/check roles.

    Every held-out corner is mandatory. Missing fit observations may be withheld
    on image evidence only; their roles never change and their decisions remain.
    Close fit/check observations belong to checking, preventing shared-joint fit
    leakage. The caller must apply the existing point/count/hull/correction gates.
    No returned point is an independent human reference or a measured panel mask.
    """
    fit_ids, check_ids = list(split['fit_module_ids']), list(split['check_module_ids'])
    by_id = {m['module_id']: m for m in matches}
    ids = fit_ids+check_ids
    if (len(by_id) != len(matches) or len(set(ids)) != len(ids)
            or not set(ids) <= set(by_id) or len(check_ids) < 2 or len(fit_ids) < 2):
        raise ValueError('Frozen disjoint known module roles with >=2 modules each required')
    points, observations = [], []
    gradients = {'native': _gradients(native_gray), 'rgb': _gradients(rgb_gray)}
    for role, selected in (('fit', fit_ids), ('check', check_ids)):
        for mid in selected:
            match = by_id[mid]
            native, rgb = np.asarray(match['native_polygon'], float), np.asarray(match['rgb_polygon'], float)
            for corner in range(4):
                record = {'control_id': f'{mid}-{corner}', 'module_id': mid, 'role': role, 'sensors': {}}
                for name, quad, radius in (('native', native, 8), ('rgb', rgb, 20)):
                    vectors = quad[[(corner+1)%4, (corner-1)%4]]-quad[corner]
                    record['sensors'][name] = _observe_corner(gradients[name], quad[corner], vectors,
                        np.linalg.norm(vectors, axis=1), radius=radius)
                record['status'] = 'observed_both_sensors' if all(s['observed_xy'] is not None
                    for s in record['sensors'].values()) else 'source_observation_unresolved'
                observations.append(record)
                if record['status'] == 'observed_both_sensors':
                    points.append({'control_id': record['control_id'], 'native_xy': record['sensors']['native']['observed_xy'],
                        'rgb_xy': record['sensors']['rgb']['observed_xy'], 'role': role})
    missing = [r['control_id'] for r in observations if r['role'] == 'check' and r['status'] != 'observed_both_sensors']

    def close(a, b):
        return (np.linalg.norm(np.asarray(a['native_xy'])-b['native_xy']) < 3
                or np.linalg.norm(np.asarray(a['rgb_xy'])-b['rgb_xy']) < 8)

    checks, fits, excluded = [], [], []
    # All original check observations remain separately available for scoring.
    for p in (p for p in points if p['role'] == 'check'):
        if not any(close(p, q) for q in checks): checks.append(p)
    for p in (p for p in points if p['role'] == 'fit'):
        if any(close(p, q) for q in checks+fits): excluded.append(p['control_id'])
        else: fits.append(p)
    return {'schema': 'pvdelta.frame-band-controls-draft/v1',
        'status': 'unresolved' if missing or len(fits) < 8 or len(checks) < 4 else 'source_observed_controls_draft',
        'split': deepcopy(split), 'observations': observations, 'points': fits+checks,
        'all_observed_check_points': [p for p in points if p['role'] == 'check'],
        'missing_check_control_ids': missing, 'shared_fit_controls_excluded': excluded,
        'fit_point_count': len(fits), 'distinct_check_point_count': len(checks),
        'reference_review_status': 'AI_developer_draft', 'independent_review_verified': False,
        'physical_identity_verified': False, 'measurement_authorised': False}
