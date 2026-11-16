"""Constrained table surface fitting; no temperature or camera-pose mutation."""
from __future__ import annotations

import numpy as np


def fit_south_plane(points, tilt_deg=25.0):
    """Fit only altitude offset in true ENU, with north edge raised by the fixed tilt.

    Deterministic alternating holdout is returned separately from the fit residual.
    Points should already be selected from the table interior, never ground pixels.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 40:
        raise ValueError("At least 40 table surface points required")
    if not np.isfinite(points).all() or not 0 < tilt_deg < 80:
        raise ValueError("Invalid plane input")
    ordered = points[np.lexsort((points[:, 0], points[:, 1]))]
    slope = np.tan(np.deg2rad(tilt_deg))
    intercept = float(np.median(ordered[::2, 2] - slope * ordered[::2, 1]))
    held_out = ordered[1::2, 2] - slope * ordered[1::2, 1] - intercept
    return intercept, {
        "points": len(points), "held_out_points": len(held_out),
        "held_out_median_abs_m": float(np.median(np.abs(held_out))),
        "held_out_p95_abs_m": float(np.percentile(np.abs(held_out), 95)),
        "validation_scope": "held-out reconstructed points, not independent survey",
    }


def brown_intrinsics(camera):
    """OpenSfM normalized Brown calibration in pixel units."""
    if camera.get("projection_type") != "brown":
        raise ValueError("Only calibrated OpenSfM Brown cameras are supported")
    w, h = camera["width"], camera["height"]
    scale = max(w, h)
    intrinsic = np.array([[camera["focal_x"] * scale, 0,
                           (w - 1)/2 + camera["c_x"] * scale],
                          [0, camera["focal_y"] * scale,
                           (h - 1)/2 + camera["c_y"] * scale], [0, 0, 1]], float)
    distortion = np.array([camera[k] for k in ("k1", "k2", "p1", "p2", "k3")])
    return intrinsic, distortion


def brown_project(points, rotation, translation, camera):
    """OpenSfM normalized Brown camera -> original RGB pixel coordinates."""
    import cv2

    intrinsic, distortion = brown_intrinsics(camera)
    r, _ = cv2.Rodrigues(np.asarray(rotation, dtype=float))
    xyz = np.ascontiguousarray(points, dtype=float)
    t = np.asarray(translation, dtype=float)
    pixels, _ = cv2.projectPoints(xyz, np.asarray(rotation, dtype=float), t,
                                 intrinsic, distortion)
    return pixels.reshape(-1, 2), (xyz @ r.T + t)[:, 2]


def fit_table_mapping(xy, observed_pixels, camera):
    """Local planar registration with independent held-out feature observations."""
    import cv2

    xy = np.asarray(xy, float)
    observed_pixels = np.asarray(observed_pixels, float)
    if (xy.ndim != 2 or xy.shape[1] != 2 or observed_pixels.shape != xy.shape
            or not np.isfinite(xy).all() or not np.isfinite(observed_pixels).all()):
        raise ValueError("Invalid local feature coordinates")
    if len(xy) < 40:
        raise ValueError("Insufficient local feature observations")
    intrinsic, distortion = brown_intrinsics(camera)
    uv = cv2.undistortPoints(np.asarray(observed_pixels, float).reshape(-1, 1, 2),
                            intrinsic, distortion, P=intrinsic).reshape(-1, 2)
    held = np.arange(len(xy)) % 4 == 0
    matrix, inliers = cv2.findHomography(xy[~held], uv[~held], cv2.RANSAC, 2.0)
    if matrix is None or not np.isfinite(matrix).all() or inliers.mean() < .65:
        raise ValueError("Local mapping has insufficient inliers")
    predicted = cv2.perspectiveTransform(xy[held].reshape(-1, 1, 2), matrix).reshape(-1, 2)
    error = np.linalg.norm(predicted - uv[held], axis=1)
    quality = {"held_out_features": int(held.sum()), "median_px": float(np.median(error)),
               "p95_px": float(np.percentile(error, 95))}
    if quality["median_px"] > 2 or quality["p95_px"] > 5:
        raise ValueError("Held-out registration residual exceeds pixel limits")
    quality['support_polygon'] = cv2.convexHull(
        xy[~held][inliers.ravel().astype(bool)].astype(np.float32)).reshape(-1, 2).tolist()
    return matrix, quality


def map_table_pixels(xy, matrix, camera):
    """Apply local registration then reapply lens distortion for source RGB sampling."""
    import cv2

    intrinsic, _ = brown_intrinsics(camera)
    uv = cv2.perspectiveTransform(np.asarray(xy, float).reshape(-1, 1, 2), matrix).reshape(-1, 2)
    rays = np.column_stack([(uv[:, 0]-intrinsic[0, 2])/intrinsic[0, 0],
                            (uv[:, 1]-intrinsic[1, 2])/intrinsic[1, 1], np.ones(len(uv))])
    return brown_project(rays, [0., 0., 0.], [0., 0., 0.], camera)[0]


def _piecewise_undistorted(xy, model):
    import cv2

    xy = np.asarray(xy, float)
    total = np.zeros((len(xy), 2))
    weights = np.zeros(len(xy))
    for piece in model['pieces']:
        weight = np.maximum(0, 1-np.abs(xy[:, 0]-piece['centre'])/model['spacing'])
        weight *= (xy[:, 0] >= piece['min_x']) & (xy[:, 0] <= piece['max_x'])
        weight *= within_feature_support(xy, piece['support_polygon'], margin_m=.5)
        prediction = cv2.perspectiveTransform(xy.reshape(-1, 1, 2),
                                             np.asarray(piece['matrix'])).reshape(-1, 2)
        total += prediction*weight[:, None]
        weights += weight
    mapped = weights > 1e-8
    uv = np.full_like(total, np.nan)
    uv[mapped] = total[mapped]/weights[mapped, None]
    return uv, mapped


def fit_piecewise_table_mapping(xy, observed_pixels, camera, east_bounds, window_m=8.0):
    """Fit overlapping spatial windows; blend coordinates, never RGB values.

    One common held-out set is excluded from *all* local fits. Unsupported windows
    stay unmapped, and the final blended field must pass the same pixel limits.
    """
    import cv2

    xy, observed = np.asarray(xy, float), np.asarray(observed_pixels, float)
    if (xy.ndim != 2 or xy.shape[1] != 2 or observed.shape != xy.shape or len(xy) < 40
            or not np.isfinite(xy).all() or not np.isfinite(observed).all()):
        raise ValueError('Insufficient or invalid piecewise observations')
    lo, hi = map(float, east_bounds)
    if not np.isfinite([lo, hi, window_m]).all() or hi <= lo or window_m <= 0:
        raise ValueError('Invalid piecewise bounds')
    intrinsic, distortion = brown_intrinsics(camera)
    uv = cv2.undistortPoints(observed.reshape(-1, 1, 2), intrinsic, distortion,
                            P=intrinsic).reshape(-1, 2)
    spacing = (hi-lo)/max(1, int(np.ceil((hi-lo)/window_m)))
    centres = np.arange(lo, hi+spacing*.5, spacing)
    held = np.arange(len(xy)) % 4 == 0
    pieces = []
    for centre in centres:
        train = (~held) & (np.abs(xy[:, 0]-centre) < spacing)
        if train.sum() < 30 or np.ptp(xy[train, 0]) < 1 or np.ptp(xy[train, 1]) < 1:
            continue
        matrix, inliers = cv2.findHomography(xy[train], uv[train], cv2.RANSAC, 2.0)
        if matrix is None or not np.isfinite(matrix).all() or inliers.mean() < .65:
            continue
        pieces.append({'centre': float(centre), 'matrix': matrix.tolist(),
                       'support_polygon': cv2.convexHull(xy[train][inliers.ravel().astype(bool)].astype(
                           np.float32)).reshape(-1, 2).tolist(),
                       'min_x': float(xy[train, 0].min()-.5),
                       'max_x': float(xy[train, 0].max()+.5),
                       'training_features': int(train.sum())})
    model = {'pieces': pieces, 'spacing': spacing}
    predicted, supported = _piecewise_undistorted(xy[held], model)
    if supported.sum() < 10 or supported.mean() < .8:
        raise ValueError('Insufficient held-out piecewise support')
    error = np.linalg.norm(predicted[supported]-uv[held][supported], axis=1)
    evidence = {'held_out_features': int(supported.sum()),
                'held_out_support_fraction': float(supported.mean()),
                'median_px': float(np.median(error)), 'p95_px': float(np.percentile(error, 95)),
                'piece_count': len(pieces)}
    if evidence['median_px'] > 2 or evidence['p95_px'] > 5:
        raise ValueError('Held-out piecewise residual exceeds pixel limits')
    return model, evidence


def map_piecewise_table_pixels(xy, model, camera):
    """Return source coordinates; unsupported domains remain NaN."""
    uv, supported = _piecewise_undistorted(xy, model)
    intrinsic, _ = brown_intrinsics(camera)
    if np.any(supported):
        part = uv[supported]
        rays = np.column_stack([(part[:, 0]-intrinsic[0, 2])/intrinsic[0, 0],
                                (part[:, 1]-intrinsic[1, 2])/intrinsic[1, 1],
                                np.ones(len(part))])
        uv[supported] = brown_project(rays, [0., 0., 0.], [0., 0., 0.], camera)[0]
    return uv


def within_feature_support(xy, polygon, margin_m=.5):
    """Limit both spatial axes to a convex training hull plus a declared small margin."""
    points, hull = np.asarray(xy, float), np.asarray(polygon, float)
    if hull.ndim != 2 or hull.shape[1] != 2 or len(hull) < 3 or margin_m < 0:
        raise ValueError('Invalid feature support hull')
    area = np.sum(hull[:, 0]*np.roll(hull[:, 1], -1)-hull[:, 1]*np.roll(hull[:, 0], -1))
    if not np.isfinite(hull).all() or abs(area) < 1e-10:
        raise ValueError('Degenerate feature support hull')
    sign = 1 if area > 0 else -1
    supported = np.isfinite(points).all(axis=1)
    for start, end in zip(hull, np.roll(hull, -1, axis=0)):
        edge = end-start
        cross = edge[0]*(points[:, 1]-start[1])-edge[1]*(points[:, 0]-start[0])
        supported &= sign*cross >= -margin_m*np.linalg.norm(edge)
    return supported


def panel_support(rgb):
    """Conservative RGB colour support, used only to propose geometry, not anomalies."""
    import cv2

    r, g, b = np.asarray(rgb, float).transpose(2, 0, 1)
    mask = ((b > g + 2) & (b > r + 3) & (g > 20)).astype("uint8") * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))


def propose_tables(rgb, resolution, module_length=2.278, module_width=1.113,
                   rows=4, tilt_deg=25, columns=(9, 27)):
    """Propose complete table rectangles; require observed gaps to split merged tables.

    The input must be a north-up metric orthophoto. Returned pixel boxes are candidates;
    georeferenced ENU plane fitting and support gates are still required.
    """
    import cv2

    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("Resolution must be positive metres/pixel")
    mask = panel_support(rgb)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates, rejected = [], []
    expected_depth = rows * module_width * np.cos(np.deg2rad(tilt_deg))
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if area * resolution**2 < 20:
            continue
        if x == 0 or y == 0 or x+w >= rgb.shape[1] or y+h >= rgb.shape[0]:
            rejected.append({"box": [x, y, x+w, y+h], "reason": "image_edge_partial"})
            continue
        if not 0.8 * expected_depth <= h*resolution <= 1.3 * expected_depth:
            rejected.append({"box": [x, y, x+w, y+h], "reason": "depth_inconsistent"})
            continue
        parts = [(x, x+w)]
        if abs(w*resolution/(18*module_length)-1) < 0.07:
            # Two nine-column tables: demand a dark, vertical seam near the centre.
            lo, hi = x + int(w*.46), x + int(w*.54)
            band = rgb[y+int(h*.15):y+int(h*.85), lo:hi].mean(axis=(0, 2))
            seam = lo + int(np.argmin(band))
            contrast = float(np.median(band) - np.min(band))
            if contrast < 12:
                rejected.append({"box": [x, y, x+w, y+h], "reason": "no_supported_table_split"})
                continue
            parts = [(x, seam), (seam+1, x+w)]
        for left, right in parts:
            length = (right-left)*resolution
            n = min(columns, key=lambda c: abs(length/(c*module_length)-1))
            mismatch = length/(n*module_length)-1
            if abs(mismatch) > 0.07:
                rejected.append({"box": [left, y, right, y+h], "reason": "length_inconsistent"})
                continue
            ys = []
            for col in range(left+int((right-left)*.1), right-int((right-left)*.1)):
                support = np.flatnonzero(labels[y:y+h, col] == label)
                if len(support) > h*.5:
                    ys.append((support[0]+y, support[-1]+y))
            if len(ys) < 20:
                continue
            top, bottom = np.median(ys, axis=0)
            candidates.append({"box": [left, float(top), right, float(bottom)],
                               "columns": n, "nominal_length_residual_fraction": mismatch})
    return candidates, rejected
