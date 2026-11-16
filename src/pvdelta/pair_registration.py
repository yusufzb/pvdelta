"""Bounded, per-table RGB-prior refinement on native thermal structural edges.

Check module indices are excluded from fitting. Adjacent modules can share borders,
so this is image consistency, not independent physical correspondence validation.
No result in this module establishes boresight calibration or measurement validity.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from pvdelta.boundaries import native_edge_check


@dataclass(frozen=True)
class PairRegistrationConfig:
    max_displacement_px: int = 12
    local_search_px: int = 4
    max_linear_change: float = .05
    max_check_p95_px: float = 2.0
    min_ambiguity_gap_px: float = .15

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 30
               for v in (self.max_displacement_px, self.local_search_px)):
            raise ValueError('search radii must be integers in [1, 30]')
        if (not all(math.isfinite(v) for v in (self.max_linear_change, self.max_check_p95_px,
                                              self.min_ambiguity_gap_px))
                or not 0 < self.max_linear_change <= .15 or self.max_check_p95_px <= 0
                or self.min_ambiguity_gap_px < 0):
            raise ValueError('invalid registration limits')


@dataclass(frozen=True)
class PairRegistrationResult:
    image_consistent: bool
    reasons: tuple[str, ...]
    correction: np.ndarray | None
    train_module_indices: tuple[int, ...]
    check_module_indices: tuple[int, ...]
    check_p95_px: float | None = None
    coarse_shift_xy: tuple[int, int] | None = None
    ambiguity_gap_px: float | None = None
    control_line_count: int = 0

    def to_dict(self):
        result = asdict(self)
        result['correction'] = self.correction.tolist() if self.correction is not None else None
        result['validation_scope'] = 'automatic_native_edge_consistency_provisional'
        result['measurement_status'] = 'withheld_requires_independent_correspondence_validation'
        return result


def apply_correction(points_xy, matrix):
    """Apply an affine correction to prior-native points, never temperatures."""
    points, affine = np.asarray(points_xy, float), np.asarray(matrix, float)
    if (points.ndim != 2 or points.shape[1] != 2 or affine.shape != (2, 3)
            or not np.isfinite(points).all() or not np.isfinite(affine).all()):
        raise ValueError('finite Nx2 points and 2x3 affine matrix are required')
    return np.column_stack((points, np.ones(len(points)))) @ affine.T


def check_registration_controls(result, prior_native_xy, observed_native_xy, *, max_p95_px=2.0):
    """Audit separate source-image controls; never optimise on these observations.

    The caller must establish point identity, exclude fit observations and limit the
    points to this table/pair. Passing checks only covers these controls, not the
    whole image, other pairs, field calibration or measurement conditions.
    """
    prior, observed = np.asarray(prior_native_xy, float), np.asarray(observed_native_xy, float)
    if (prior.ndim != 2 or prior.shape[1:] != (2,) or prior.shape != observed.shape
            or not np.isfinite(prior).all() or not np.isfinite(observed).all()
            or not math.isfinite(max_p95_px) or max_p95_px <= 0):
        raise ValueError('matching finite Nx2 controls and a positive limit are required')
    reasons = []
    if len(prior) < 4:
        reasons.append('at least four separate correspondence controls required')
    elif (np.linalg.matrix_rank(prior-prior.mean(axis=0)) < 2
          or np.linalg.matrix_rank(observed-observed.mean(axis=0)) < 2):
        reasons.append('correspondence controls are collinear')
    if not result.image_consistent or result.correction is None:
        reasons.append('automatic registration is withheld')
    if reasons:
        return {'accepted_at_controls': False, 'reasons': reasons,
                'control_count': len(prior), 'p95_native_px': None}
    errors = np.linalg.norm(apply_correction(prior, result.correction)-observed, axis=1)
    p95 = float(np.percentile(errors, 95))
    if p95 > max_p95_px:
        reasons.append('separate correspondence error exceeds limit')
    return {'accepted_at_controls': not reasons, 'reasons': reasons, 'control_count': len(prior),
            'p95_native_px': p95, 'max_native_px': float(errors.max()),
            'scope': 'provided separate controls in this pair/table only'}


def _oriented_maps(gray, polygons):
    import cv2

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    canny = cv2.Canny(gray, 25, 70) > 0
    angle = np.arctan2(gy, gx)
    maps = {}
    for polygon in polygons:
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            delta = end-start
            normal_angle = math.atan2(delta[0], -delta[1]) % math.pi
            key = round(normal_angle / math.pi * 16) % 16
            if key not in maps:
                expected = key * math.pi/16
                compatible = np.abs(np.cos(angle-expected)) >= math.cos(math.radians(22.5))
                selected = canny & compatible & (magnitude >= 15)
                maps[key] = cv2.distanceTransform((~selected).astype(np.uint8), cv2.DIST_L2, 5)
    return maps, gx, gy


def _samples(polygons, indices):
    points, keys = [], []
    for index in indices:
        polygon = polygons[index]
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
            delta = end-start
            key = round((math.atan2(delta[0], -delta[1]) % math.pi)/math.pi*16) % 16
            points.extend(start+np.linspace(.15, .85, 24)[:, None]*delta)
            keys.extend([key]*24)
    return np.asarray(points, np.float32), np.asarray(keys)


def _distances(points, keys, maps):
    import cv2

    values = np.empty(len(points), np.float32)
    for key in np.unique(keys):
        selected = keys == key
        values[selected] = cv2.remap(maps[int(key)], points[selected, 0][None].astype(np.float32),
                                     points[selected, 1][None].astype(np.float32), cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=50)[0]
    return values


def _line_controls(polygon, shift, gx, gy, radius):
    import cv2

    lines = []
    offsets = np.arange(-radius, radius+1, dtype=np.float32)
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        tangent = (end-start)/np.linalg.norm(end-start)
        normal = np.float32([-tangent[1], tangent[0]])
        source = start+np.linspace(.12, .88, 32)[:, None]*(end-start)
        search = (source[:, None]+shift+offsets[None, :, None]*normal).astype(np.float32)
        x, y = search[:, :, 0], search[:, :, 1]
        dx, dy = cv2.remap(gx, x, y, cv2.INTER_LINEAR), cv2.remap(gy, x, y, cv2.INTER_LINEAR)
        response = np.abs(dx*normal[0]+dy*normal[1])
        response *= np.abs(dx*normal[0]+dy*normal[1]) / np.maximum(1, np.hypot(dx, dy))
        peaks = np.argmax(response, axis=1)
        strength = response[np.arange(len(source)), peaks]
        good = strength >= max(15, float(np.percentile(strength, 20)))
        if good.mean() < .6:
            continue
        target = search[np.arange(len(source)), peaks][good]
        vx, vy, x0, y0 = cv2.fitLine(target, cv2.DIST_HUBER, 0, .01, .01).ravel()
        fitted_normal = np.array([-vy, vx])
        constant = float(fitted_normal @ [x0, y0])
        residual = np.abs(target @ fitted_normal-constant)
        if np.median(residual) <= 1.5 and abs(np.dot([vx, vy], tangent)) >= .98:
            lines.append((source[good], fitted_normal, constant))
    return lines


def refine_panel_registration(native_gray, initial_polygons_native, *, config=None):
    """Fit one bounded local affine correction for one visible table in one pair.

    Supply >=4 prior-native quadrilaterals belonging to the SAME surface/table.
    Checker module indices are fixed spatially before optimisation and not fitted.
    Shared table borders mean this is not independent point validation.
    The caller still needs independent correspondence checks before measurement use.
    """
    import cv2

    cfg = config or PairRegistrationConfig()
    gray = np.asarray(native_gray)
    if gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 32:
        raise ValueError('expected native uint8 grayscale structural image')
    polygons = [np.asarray(p, dtype=np.float32) for p in initial_polygons_native]
    if any(p.shape != (4, 2) or not np.isfinite(p).all() or not cv2.isContourConvex(p)
           or abs(cv2.contourArea(p)) < 16 for p in polygons):
        raise ValueError('finite, convex prior-native quadrilaterals are required')
    if len(polygons) < 4:
        return PairRegistrationResult(False, ('at least four same-table modules required',), None, (), ())
    centres = np.asarray([p.mean(axis=0) for p in polygons])
    order = np.lexsort((centres[:, 1], centres[:, 0]))
    train, check = tuple(map(int, order[::2])), tuple(map(int, order[1::2]))
    def rejected(reason, **kwargs):
        return PairRegistrationResult(False, (reason,), None, train, check, **kwargs)
    height, width = gray.shape
    if any((p < 3).any() or (p[:, 0] >= width-3).any() or (p[:, 1] >= height-3).any() for p in polygons):
        return rejected('prior polygon is outside the native frame margin')
    shortest = min(float(np.linalg.norm(p-np.roll(p, -1, axis=0), axis=1).min()) for p in polygons)
    radius = min(cfg.max_displacement_px, int(shortest*.25))
    if radius < 2:
        return rejected('native module is too small for unambiguous bounded refinement')
    smooth = cv2.GaussianBlur(gray, (5, 5), 1)
    maps, gx, gy = _oriented_maps(smooth, polygons)
    source_points, keys = _samples(polygons, train)
    scores = []
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            distance = _distances(source_points+[dx, dy], keys, maps)
            scores.append((float(np.minimum(distance, 12).mean()), dx, dy))
    scores.sort(key=lambda s: (s[0], math.hypot(s[1], s[2])))
    best, dx, dy = scores[0]
    if abs(dx) >= radius or abs(dy) >= radius:
        return rejected('best alignment reaches the displacement search limit', coarse_shift_xy=(dx, dy))
    alternatives = [s[0] for s in scores if math.hypot(s[1]-dx, s[2]-dy) >= 5]
    gap = min(alternatives)-best if alternatives else 0.0
    if gap < cfg.min_ambiguity_gap_px:
        return rejected('structural alignment is ambiguous', coarse_shift_xy=(dx, dy), ambiguity_gap_px=gap)
    lines = [line for index in train for line in _line_controls(
        polygons[index], np.float32([dx, dy]), gx, gy, cfg.local_search_px)]
    if len(lines) < 6:
        return rejected('insufficient coherent training frame lines', control_line_count=len(lines))
    origin = source_points.mean(axis=0)
    scale = max(float(np.ptp(source_points, axis=0).max()), 1)
    design, target = [], []
    for points, normal, constant in lines:
        normalized = (points-origin)/scale
        for p, (x, y) in zip(points, normalized):
            nx, ny = normal
            design.append([nx*x, nx*y, nx, ny*x, ny*y, ny])
            target.append(constant-normal @ p)
    design, target = np.asarray(design), np.asarray(target)
    if np.linalg.matrix_rank(design) < 6:
        return rejected('training line geometry is degenerate')
    weight = np.ones(len(target))
    for _ in range(4):
        solution = np.linalg.lstsq(design*weight[:, None], target*weight, rcond=None)[0]
        residual = np.abs(design @ solution-target)
        weight = np.sqrt(np.minimum(1, 1.5/np.maximum(residual, 1e-6)))
    delta = np.array([[solution[0], solution[1]], [solution[3], solution[4]]])/scale
    linear = np.eye(2)+delta
    translation = np.array([solution[2], solution[5]])-delta @ origin
    matrix = np.column_stack((linear, translation))
    all_corners = np.concatenate(polygons)
    corrected = apply_correction(all_corners, matrix)
    if (not np.isfinite(matrix).all() or np.linalg.det(linear) <= 0
            or np.linalg.norm(delta, ord=2) > cfg.max_linear_change
            or np.linalg.norm(corrected-all_corners, axis=1).max() > cfg.max_displacement_px):
        return rejected('affine correction exceeds the bounded geometry allowance')
    check_points, check_keys = _samples(polygons, check)
    check_points = apply_correction(check_points, matrix)
    if ((check_points < 3).any() or (check_points[:, 0] >= width-3).any()
            or (check_points[:, 1] >= height-3).any()):
        return rejected('held-out points leave the native frame margin')
    p95 = float(np.percentile(_distances(check_points, check_keys, maps), 95))
    if p95 > cfg.max_check_p95_px:
        return rejected('held-out module edge error exceeds limit', check_p95_px=p95)
    # Four-side gates use the unchanged native-edge criteria. They do not fit the transform.
    if not all(native_edge_check(smooth, apply_correction(polygons[i], matrix))['accepted'] for i in check):
        return rejected('held-out module four-side check rejected', check_p95_px=p95)
    matrix.setflags(write=False)
    return PairRegistrationResult(True, (), matrix, train, check, p95, (dx, dy), gap, len(lines))
