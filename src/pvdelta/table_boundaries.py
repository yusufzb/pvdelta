"""RGB-only, image-supported table separators. Outputs are unreviewed proposals.

The layout prior constrains separator spacing; it never fills a uniform grid. This
nadir-oriented proposer does not establish native thermal masks or physical IDs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from pvdelta.boundaries import order_quad, rgb_module_proposals


@dataclass(frozen=True)
class TableBoundaryConfig:
    rows: int = 4
    module_length_m: float = 2.278
    module_width_m: float = 1.113
    tilt_deg: float = 25.0
    max_dimension: int = 2016
    min_panel_support: float = .80
    edge_search_fraction: float = .18
    cell_columns: int = 24
    cell_rows: int = 6

    def __post_init__(self):
        if (isinstance(self.rows, bool) or not isinstance(self.rows, int) or self.rows < 2
                or isinstance(self.max_dimension, bool) or not isinstance(self.max_dimension, int)
                or self.max_dimension < 128
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 2
                       for v in (self.cell_columns, self.cell_rows))):
            raise ValueError('rows >= 2 and max_dimension >= 128 are required')
        if (not all(math.isfinite(v) for v in (self.module_length_m, self.module_width_m,
                                              self.tilt_deg, self.min_panel_support, self.edge_search_fraction))
                or self.module_length_m <= self.module_width_m or self.module_width_m <= 0
                or not 0 <= self.tilt_deg < 75 or not 0 < self.min_panel_support <= 1
                or not .05 <= self.edge_search_fraction <= .4):
            raise ValueError('invalid module dimensions, tilt or support threshold')


def _peaks(profile, *, distance, threshold):
    candidates = [i for i in range(2, len(profile)-2)
                  if profile[i] >= profile[i-1] and profile[i] > profile[i+1]
                  and profile[i] >= threshold]
    kept = []
    for index in sorted(candidates, key=lambda i: profile[i], reverse=True):
        if all(abs(index-other) >= distance for other in kept):
            kept.append(index)
    return sorted(kept)


def _refine_frame_edges(gray, polygon, search_fraction):
    """Fit four local bright frame lines; reject texture-only extrapolation."""
    import cv2

    edges = np.roll(polygon, -1, axis=0)-polygon
    radius = max(3, search_fraction*float(np.linalg.norm(edges, axis=1).min()))
    offsets = np.arange(-round(radius), round(radius)+1, dtype=np.float32)
    lines, residuals = [], []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        tangent = (end-start)/np.linalg.norm(end-start)
        normal = np.array([-tangent[1], tangent[0]], np.float32)
        samples = start+np.linspace(.08, .92, 45)[:, None]*(end-start)
        points = (samples[:, None]+offsets[None, :, None]*normal).astype(np.float32)
        intensity = cv2.remap(gray, points[:, :, 0], points[:, :, 1], cv2.INTER_LINEAR)
        # A contiguous frame is stronger than the cell busbars within the search band.
        peaks = np.argmax(intensity, axis=1)
        best = intensity[np.arange(len(samples)), peaks]
        good = best >= np.maximum(110, np.median(intensity, axis=1)+18)
        if good.mean() < .60:
            return None
        selected = points[np.arange(len(samples)), peaks][good]
        vx, vy, x, y = cv2.fitLine(selected, cv2.DIST_HUBER, 0, .01, .01).ravel()
        fitted_normal = np.array([-vy, vx])
        constant = float(fitted_normal @ [x, y])
        errors = np.abs(selected @ fitted_normal-constant)
        if np.median(errors) > 1.5 or abs(np.dot([vx, vy], tangent)) < .94:
            return None
        lines.append((fitted_normal, constant))
        residuals.append(float(np.median(errors)))
    corners = []
    for index in range(4):
        previous, current = lines[index-1], lines[index]
        system = np.stack([previous[0], current[0]])
        if abs(np.linalg.det(system)) < .1:
            return None
        corners.append(np.linalg.solve(system, [previous[1], current[1]]))
    refined = np.asarray(corners, np.float32)
    if (not cv2.isContourConvex(refined)
            or np.linalg.norm(refined-polygon, axis=1).max() > 2*radius):
        return None
    return refined, max(residuals)


def _fit_sides(panel, quad):
    """Fit the two observed long table sides, including tables clipped at their ends."""
    import cv2

    w = round(float(np.linalg.norm(quad[1]-quad[0])))
    h = round(float(np.linalg.norm(quad[3]-quad[0])))
    target = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    transform = cv2.getPerspectiveTransform(quad, target)
    mask = cv2.warpPerspective(panel, transform, (w, h), flags=cv2.INTER_NEAREST)
    samples = []
    for x in range(3, w-3, max(1, w//400)):
        ys = np.flatnonzero(mask[:, x])
        if len(ys) >= h*.6:
            samples.append((x, *np.percentile(ys, [1, 99])))
    if len(samples) < 20:
        return quad
    points = np.asarray(samples)
    lines = []
    for column in (1, 2):
        keep = np.ones(len(points), dtype=bool)
        for _ in range(3):
            coefficients = np.polyfit(points[keep, 0], points[keep, column], 1)
            errors = np.abs(np.polyval(coefficients, points[:, 0])-points[:, column])
            keep = errors <= max(2, float(np.median(errors))*3)
            if keep.sum() < 20:
                return quad
        lines.append(np.polyval(coefficients, [0, w-1]))
    local = np.float32([[0, lines[0][0]], [w-1, lines[0][1]],
                        [w-1, lines[1][1]], [0, lines[1][0]]])
    return cv2.perspectiveTransform(local[:, None], np.linalg.inv(transform))[:, 0]


def table_module_proposals(bgr: np.ndarray, *, config: TableBoundaryConfig | None = None,
                           gimbal_pitch_deg: float | None = None) -> dict:
    """Detect table envelopes and observed separators; return original RGB polygons.

    Requires near-nadir input and the declared landscape layout. Partial end modules
    are not extrapolated. Missing separators and unsupported tables remain explicit.
    """
    import cv2

    cfg = config or TableBoundaryConfig()
    if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3
            or bgr.shape[2] != 3 or min(bgr.shape[:2]) < 32):
        raise ValueError('expected uint8 BGR image with dimensions >= 32')
    if gimbal_pitch_deg is None or not math.isfinite(gimbal_pitch_deg):
        return {'schema': 'pvdelta.table-boundaries/v1', 'status': 'pose_unavailable',
                'modules': [], 'tables': []}
    if abs(gimbal_pitch_deg+90) > 10:
        return {'schema': 'pvdelta.table-boundaries/v1', 'status': 'deferred_oblique_geometry',
                'modules': [], 'tables': []}
    scale = min(1., cfg.max_dimension / max(bgr.shape[:2]))
    image = cv2.resize(bgr, (round(bgr.shape[1]*scale), round(bgr.shape[0]*scale)))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    panel = ((hsv[:, :, 0] > 85) & (hsv[:, :, 0] < 140)
             & (hsv[:, :, 1] < 155) & (hsv[:, :, 2] > 25)).astype(np.uint8)
    closed = cv2.morphologyEx(panel, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    modules, tables = [], []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < max(1600, image.shape[0]*image.shape[1]*.0015):
            continue
        rect = cv2.minAreaRect(contour)
        w, h = sorted(rect[1], reverse=True)
        if h < cfg.rows*16 or w < h or area / max(w*h, 1) < .65:
            continue
        quad = order_quad(cv2.boxPoints(rect))
        lengths = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
        if lengths[[0, 2]].mean() < lengths[[1, 3]].mean():
            quad = np.roll(quad, -1, axis=0)
        quad = _fit_sides(closed, quad)
        width = round(float(np.linalg.norm(quad[1]-quad[0])))
        height = round(float(np.linalg.norm(quad[3]-quad[0])))
        record = {'table_index': len(tables), 'polygon_rgb_xy': (quad/scale).tolist(),
                  'status': 'unresolved', 'reason': 'separators_not_supported',
                  'module_count': 0}
        tables.append(record)
        if min(width, height) < cfg.rows*12 or width*height > 8_000_000:
            continue
        target = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
        matrix = cv2.getPerspectiveTransform(quad, target)
        inverse = np.linalg.inv(matrix)
        rectified = cv2.warpPerspective(image, matrix, (width, height))
        support = cv2.warpPerspective(panel, matrix, (width, height), flags=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY).astype(float)
        pitch = height / cfg.rows
        y_profile = np.percentile(gray[:, max(2, width//20):-max(2, width//20)], 60, axis=1)
        y_profile = np.convolve(y_profile, [.25, .5, .25], mode='same')
        ys = [0]
        for row in range(1, cfg.rows):
            lo, hi = round(row*pitch-.16*pitch), round(row*pitch+.16*pitch)+1
            position = lo + int(np.argmax(y_profile[lo:hi]))
            baseline = float(np.median(y_profile[max(0, position-int(.4*pitch)):
                                                min(height, position+int(.4*pitch))]))
            if y_profile[position] < max(100, baseline+18):
                break
            ys.append(position)
        if len(ys) != cfg.rows:
            record['row_lines'] = ys
            continue
        ys.append(height-1)
        expected = pitch * cfg.module_length_m / (cfg.module_width_m*math.cos(math.radians(cfg.tilt_deg)))
        record.update(row_lines=ys, long_axis_lines=[], rectified_size=[width, height])
        # Each row has its own observed joins. A long straight seam assumption
        # across the entire table fails under off-axis lens distortion.
        for row, (top, bottom) in enumerate(pairwise(ys)):
            margin = max(2, round(.12*(bottom-top)))
            x_profile = np.percentile(gray[top+margin:bottom-margin], 65, axis=0)
            x_profile = np.convolve(x_profile, [.25, .5, .25], mode='same')
            baseline = float(np.median(x_profile))
            xs = _peaks(x_profile, distance=max(4, .12*expected), threshold=max(105, baseline+22))
            record['long_axis_lines'].append({'row_index': row, 'positions': xs})
            intervals = []
            for i, left in enumerate(xs):
                for right in xs[i+1:]:
                    ratio = (right-left)/expected
                    if .78 <= ratio <= 1.35:
                        score = float((x_profile[left]+x_profile[right])/(1+1.5*abs(ratio-1)))
                        intervals.append((score, left, right))
            chosen = []
            for _, left, right in sorted(intervals, reverse=True):
                if any(min(right, b)-max(left, a) > .05*expected for a, b in chosen):
                    continue
                chosen.append((left, right))
            for left, right in sorted(chosen):
                inset = max(2, round(.04*min(right-left, bottom-top)))
                local_support = support[top+inset:bottom-inset, left+inset:right-inset]
                if not local_support.size or local_support.mean() < cfg.min_panel_support:
                    continue
                local_quad = np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])
                polygon = cv2.perspectiveTransform(local_quad[:, None], inverse)[:, 0]
                fitted = _refine_frame_edges(image_gray, polygon, cfg.edge_search_fraction)
                if fitted is None:
                    continue
                polygon, line_residual = fitted
                if ((polygon < 3).any() or (polygon[:, 0] >= image.shape[1]-3).any()
                        or (polygon[:, 1] >= image.shape[0]-3).any()):
                    continue
                modules.append({'polygon_rgb_xy': (polygon/scale).tolist(),
                                'table_index': record['table_index'], 'row_index': row,
                                'source': 'observed_table_separators', 'visibility': 'full',
                                'panel_support': float(local_support.mean()),
                                'frame_line_median_residual_px': line_residual/scale,
                                'assessment_status': 'unreviewed_rgb_geometry'})
                record['module_count'] += 1
        if record['module_count']:
            record.update(status='proposed', reason='observed_separators_and_panel_support')
    return {'schema': 'pvdelta.table-boundaries/v1', 'status': 'unreviewed_rgb_geometry',
            'modules': modules, 'tables': tables}


def _full_module_signature(image, polygon, cell_columns, cell_rows):
    """Cell-grid evidence for the declared landscape module format."""
    import cv2

    quad = order_quad(polygon)
    lengths = np.linalg.norm(quad-np.roll(quad, -1, axis=0), axis=1)
    quad = np.roll(quad, -int(np.argmax(lengths)), axis=0)
    transform = cv2.getPerspectiveTransform(quad, np.float32(
        [[0, 0], [239, 0], [239, 119], [0, 119]]))
    crop = cv2.warpPerspective(image, transform, (240, 120))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    counts = []
    for axis in (0, 1):
        gradient = cv2.Sobel(gray, cv2.CV_32F, 1-axis, axis)
        profile = np.mean(np.abs(gradient), axis=axis)
        counts.append(len(_peaks(profile, distance=5, threshold=float(np.percentile(profile, 57)))))
    return (.75*cell_columns <= counts[0] <= 1.25*cell_columns
            and .66*cell_rows <= counts[1] <= 2.34*cell_rows)


def refined_module_proposals(bgr: np.ndarray, *, config: TableBoundaryConfig | None = None,
                             gimbal_pitch_deg: float | None = None,
                             contour_polygons: list[np.ndarray] | None = None) -> dict:
    """Combine observed table separators with full-cell-signature contour fallback.

    The fallback defaults to a 144-half-cell/24×6 layout. Both paths remain
    unreviewed RGB proposals; new geometry never becomes a thermal mask implicitly.
    """
    import cv2

    cfg = config or TableBoundaryConfig()
    result = table_module_proposals(bgr, config=cfg, gimbal_pitch_deg=gimbal_pitch_deg)
    if result['status'] != 'unreviewed_rgb_geometry':
        return result
    contours = rgb_module_proposals(bgr) if contour_polygons is None else contour_polygons
    modules = result['modules']
    for polygon in contours:
        quad = np.asarray(polygon, dtype=np.float32)
        if not _full_module_signature(bgr, quad, cfg.cell_columns, cfg.cell_rows):
            continue
        duplicate = False
        for other in modules:
            existing = np.asarray(other['polygon_rgb_xy'], dtype=np.float32)
            overlap, _ = cv2.intersectConvexConvex(quad, existing)
            # Suppress contained half/split contours as well as identical proposals.
            if overlap / max(1, min(abs(cv2.contourArea(quad)), abs(cv2.contourArea(existing)))) > .30:
                duplicate = True
                break
        if not duplicate:
            modules.append({'polygon_rgb_xy': quad.tolist(), 'table_index': None,
                            'row_index': None, 'source': 'contour_cell_signature',
                            'visibility': 'full', 'assessment_status': 'unreviewed_rgb_geometry'})
    return result
