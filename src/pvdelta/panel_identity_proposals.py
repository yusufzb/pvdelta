"""Experimental cross-frame panel hypotheses; never an identity or fault verdict."""
from __future__ import annotations

import cv2
import numpy as np

from pvdelta.convex_geometry import convex_iou


def project(points, matrix):
    result = cv2.perspectiveTransform(np.float64(points).reshape(-1, 1, 2), matrix).reshape(-1, 2)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite projection')
    return result


def mutual_features(first, second):
    """Full-frame descriptors, with no GPS, panel index or anomaly prior in matching."""
    a, da = first
    b, db = second
    if da is None or db is None or min(len(da), len(db)) < 2:
        return np.empty((0, 2)), np.empty((0, 2))

    def links(x, y):
        return {(m.queryIdx, m.trainIdx) for pair in cv2.BFMatcher().knnMatch(x, y, k=2)
                if len(pair) == 2 for m, n in [pair] if m.distance < .70*n.distance}

    pairs = sorted(links(da, db) & {(j, i) for i, j in links(db, da)})
    used = []
    for i, j in pairs:
        p, q = np.asarray(a[i].pt), np.asarray(b[j].pt)
        if any(np.linalg.norm(p-u) < 1 or np.linalg.norm(q-v) < 1 for u, v in used):
            continue
        used.append((p, q))
    if not used:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.array([p for p, q in used]), np.array([q for p, q in used])


def frame_geometry(a, b):
    """Fixed spatial fit/check roles; check points cannot influence estimation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('Paired xy points required')
    check = (np.floor(a[:, 0]/16).astype(int)+np.floor(a[:, 1]/16).astype(int)) % 4 == 0
    result = {'status': 'insufficient_features', 'matches': len(a),
              'fit_points': int((~check).sum()), 'check_points': int(check.sum())}
    if (~check).sum() < 12 or check.sum() < 6:
        return result
    cv2.setRNGSeed(0)
    forward, mask = cv2.findHomography(a[~check], b[~check], cv2.RANSAC, 2.)
    cv2.setRNGSeed(0)
    reverse, reverse_mask = cv2.findHomography(b[~check], a[~check], cv2.RANSAC, 2.)
    if forward is None or reverse is None or min(mask.sum(), reverse_mask.sum()) < 12:
        return {**result, 'status': 'insufficient_fit_inliers'}
    errors = np.maximum(np.linalg.norm(project(a[check], forward)-b[check], axis=1),
                        np.linalg.norm(project(b[check], reverse)-a[check], axis=1))
    result.update(check_median_px=float(np.median(errors)), check_within_4px=float(np.mean(errors <= 4)))
    if np.median(errors) > 3 or np.mean(errors <= 4) < .6:
        return {**result, 'status': 'heldout_geometry_failed'}
    return {**result, 'status': 'frame_geometry_supported', 'forward': forward,
            'reverse': reverse, 'source_points': a, 'target_points': b, 'check': check}


def panel_pair(first, second, geometry, *, competitors=()):
    """Require reciprocal overlap, competing-panel separation and local held-out texture."""
    if geometry['status'] != 'frame_geometry_supported':
        return {'status': geometry['status']}
    first, second = np.asarray(first, float), np.asarray(second, float)
    mapped = project(first, geometry['forward'])
    iou = convex_iou(mapped, second)
    reverse_iou = convex_iou(project(second, geometry['reverse']), first)
    competitor = max((convex_iou(mapped, p) for p in competitors), default=0.)
    result = {'iou': iou, 'reverse_iou': reverse_iou, 'competing_iou': competitor,
              'mapped_polygon': mapped.tolist()}
    if min(iou, reverse_iou) < .65:
        return {**result, 'status': 'different_or_unresolved_panel'}
    if iou-competitor < .15:
        return {**result, 'status': 'ambiguous_neighbour'}
    a, b = geometry['source_points'], geometry['target_points']
    # Use only points directly on the saved panel polygon, excluding vegetation context.
    inside = np.array([cv2.pointPolygonTest(np.float32(first), tuple(p), False) > 0
                       and cv2.pointPolygonTest(np.float32(second), tuple(q), False) > 0
                       for p, q in zip(a, b)], bool)
    residual = np.maximum(np.linalg.norm(project(a, geometry['forward'])-b, axis=1),
                          np.linalg.norm(project(b, geometry['reverse'])-a, axis=1))
    good = inside & (residual <= 3)
    fit, check = good & ~geometry['check'], good & geometry['check']
    result.update(local_fit_points=int(fit.sum()), local_check_points=int(check.sum()))
    if fit.sum() < 4 or check.sum() < 2:
        return {**result, 'status': 'overlap_only_review_required'}
    # Avoid counting several points along one cell edge as two-dimensional support.
    local = a[good]
    singular = np.linalg.svd(local-local.mean(axis=0), compute_uv=False)
    if singular[1] < 3 or singular[1]/singular[0] < .1:
        return {**result, 'status': 'local_texture_degenerate'}
    return {**result, 'status': 'provisional_repeat_support'}
