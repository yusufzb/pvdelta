"""Experimental source-image junction matching within one table and capture pair.

Seed quads are provisional correspondences, not labels. Observed frame lines refine
both sensors separately. Structural displays never replace native radiometry for
anomaly measurement; native values are unchanged and no result authorises measurement.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from pvdelta.alignment import fit_homography, transform_points, validate_homography_checks


def radiometric_structure(temperature_c):
    """Create a monotonic structural display; never modify/return measured temperatures.

    Unlike palette grayscale this preserves temperature ordering. Percentile clipping
    is ONLY for registration gradients. Anomaly measurements must use the original grid.
    """
    values = np.asarray(temperature_c, float)
    if values.ndim != 2 or min(values.shape) < 32 or not np.isfinite(values).all():
        raise ValueError('finite 2D native radiometry required')
    low, high = np.percentile(values, [2, 98])
    if high-low <= 1e-6:
        raise ValueError('radiometry has insufficient structural dynamic range')
    display = np.rint(np.clip((values-low)/(high-low), 0, 1)*255).astype(np.uint8)
    return display, {'source': 'native_sdk_temperature_c', 'percentiles': [2, 98],
                     'low_c': float(low), 'high_c': float(high),
                     'purpose': 'registration_structure_only_not_measurement'}


@dataclass(frozen=True)
class JunctionConfig:
    rgb_merge_px: float = 8.0
    rgb_search_px: int = 16
    native_search_px: int = 16
    max_check_p95_px: float = 2.0
    min_gradient: float = 15.0
    module_side_only: bool = False
    paired_block_split: bool = False
    measurement_control_split: bool = False
    full_module_edge_support: bool = False

    def __post_init__(self):
        if not isinstance(self.full_module_edge_support, bool):
            raise TypeError('full_module_edge_support must be boolean')
        if self.full_module_edge_support and not self.module_side_only:
            raise ValueError('Full module-edge support requires module_side_only')
        if not isinstance(self.module_side_only, bool):
            raise TypeError('module_side_only must be boolean')
        if not isinstance(self.paired_block_split, bool):
            raise TypeError('paired_block_split must be boolean')
        if not isinstance(self.measurement_control_split, bool):
            raise TypeError('measurement_control_split must be boolean')
        if self.measurement_control_split and self.paired_block_split:
            raise ValueError('Choose one fixed control split')
        for radius in (self.rgb_search_px, self.native_search_px):
            if isinstance(radius, bool) or not isinstance(radius, int) or not 2 <= radius <= 30:
                raise ValueError('search radii must be integers in [2, 30]')
        if any(not math.isfinite(v) or v <= 0 for v in (
                self.rgb_merge_px, self.max_check_p95_px, self.min_gradient)):
            raise ValueError('finite positive junction limits required')


def _gradients(image):
    import cv2

    gray = np.asarray(image)
    if gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 32:
        raise ValueError('uint8 source-grid grayscale images required')
    smooth = cv2.GaussianBlur(gray, (5, 5), 1)
    return cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1)


def _response(gx, gy, points, normal, *, signed=False):
    import cv2

    shape = points.shape[:-1]
    points = np.asarray(points, np.float32).reshape(-1, 2)
    x, y = points[:, 0][None], points[:, 1][None]
    dx = cv2.remap(gx, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[0]
    dy = cv2.remap(gy, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[0]
    projected = dx*normal[0]+dy*normal[1]
    weighted = projected*np.abs(projected)/np.maximum(np.hypot(dx, dy), 1)
    return (weighted if signed else np.abs(weighted)).reshape(shape)


def _paired_ridges(signed, width):
    """Centre a thin frame using opposing gradients instead of choosing either edge."""
    count = signed.shape[1]-2*width
    ridge = np.zeros((signed.shape[0], count), float)
    for half in range(1, width+1):
        left = signed[:, width-half:width-half+count]
        right = signed[:, width+half:width+half+count]
        ridge = np.maximum(ridge, np.minimum(np.abs(left), np.abs(right))*(left*right < 0))
    return ridge


def _edge_pair_centres(signed, offsets, centre, half_width, minimum):
    """Localise opposite edges separately, then use their subpixel midpoint.

    A min-of-gradients ridge can have a flat maximum when one side has weaker
    contrast. Selecting its first maximum biases the frame toward that side.
    """
    local = np.flatnonzero(np.abs(offsets-centre) <= half_width+2)
    values = signed[:, local]
    left = offsets[local] < centre
    right = offsets[local] > centre
    if not left.any() or not right.any():
        return np.full(len(signed), centre), np.zeros(len(signed), bool)
    # Opposite edges must straddle the detected band. Unrestricted global extrema
    # can pick two cell-grid edges on the same side of a narrow RGB frame.
    lp = np.argmax(np.where(left, values, -np.inf), axis=1)
    ln = np.argmin(np.where(left, values, np.inf), axis=1)
    rp = np.argmax(np.where(right, values, -np.inf), axis=1)
    rn = np.argmin(np.where(right, values, np.inf), axis=1)
    rows = np.arange(len(signed))
    peak = np.minimum(values[rows, lp], -values[rows, rn])
    valley = np.minimum(-values[rows, ln], values[rows, rp])
    positive = np.where(peak >= valley, lp, rp)
    negative = np.where(peak >= valley, rn, ln)

    def extremum(indices, polarity):
        rows = np.arange(len(indices))
        full = local[indices]
        clipped = np.clip(full, 1, signed.shape[1]-2)
        left, mid, right = (polarity*signed[rows, clipped+delta] for delta in (-1, 0, 1))
        curvature = left-2*mid+right
        fractional = np.divide(.5*(left-right), curvature,
                               out=np.zeros_like(mid), where=np.abs(curvature) > 1e-6)
        location = offsets[full]+np.clip(fractional, -.5, .5)
        interior = (indices > 0) & (indices < len(local)-1)
        return location, polarity*signed[rows, full], interior

    p, ps, pi = extremum(positive, 1)
    n, ns, ni = extremum(negative, -1)
    midpoint = (p+n)*.5
    separation = np.abs(p-n)
    good = (pi & ni & (ps >= minimum) & (ns >= minimum) & (separation >= 1)
            & (separation <= 2*half_width+2) & (np.abs(midpoint-centre) <= half_width))
    return midpoint, good


def _find_junction(gradients, seed, directions, lengths, radius, minimum, *, module_side_only=False,
                   full_module_edge_support=False):
    import cv2

    gx, gy = gradients
    lines = []
    for direction, length in zip(directions, lengths, strict=True):
        tangent = direction/np.linalg.norm(direction)
        normal = np.array([-tangent[1], tangent[0]])
        # Span follows this sensor's module size, not the search radius. A fixed
        # native-pixel cap would shrink the RGB support to a few cell-grid pixels.
        span = max(6, length*.35)
        along = np.r_[np.linspace(-span, -3, 24), np.linspace(3, span, 24)]
        # The positive arm points into the selected observed module. Outside
        # vegetation must not determine its frame line. Both arms are still
        # inspected below for termination evidence, independently of fitting.
        if module_side_only:
            along = (np.linspace(max(3, length*.15), length*.85, 48)
                     if full_module_edge_support else along[along > 0])
        offsets = np.arange(-radius, radius+1)
        half_width = max(2, min(8, round(min(lengths)*.06)))
        padded = np.arange(-radius-half_width, radius+half_width+1)
        search = seed+along[:, None, None]*tangent+padded[None, :, None]*normal
        signed = _response(gx, gy, search, normal, signed=True)
        response = np.abs(signed[:, half_width:-half_width])
        ridge = _paired_ridges(signed, half_width)
        ridge_mode = np.percentile(ridge, 65, axis=0).max() >= max(
            minimum, .5*np.percentile(response, 65, axis=0).max())
        if ridge_mode:
            response = ridge
        points = seed+along[:, None, None]*tangent+offsets[None, :, None]*normal
        scores = np.percentile(response, 65, axis=0)
        best = int(np.argmax(scores))
        if best in (0, len(offsets)-1) or scores[best] < minimum:
            return None
        if ridge_mode:
            centres, good = _edge_pair_centres(signed, padded, offsets[best], half_width, minimum)
            targets = seed+along[:, None]*tangent+centres[:, None]*normal
        else:
            local = np.flatnonzero(np.abs(offsets-offsets[best]) <= 2)
            peaks = local[np.argmax(response[:, local], axis=1)]
            strength = response[np.arange(len(along)), peaks]
            good = strength >= minimum
            targets = points[np.arange(len(along)), peaks]
        if good.sum() < (math.ceil(.75*len(along)) if full_module_edge_support else 18):
            return None
        target = np.float32(targets[good])
        vx, vy, x0, y0 = cv2.fitLine(target, cv2.DIST_HUBER, 0, .01, .01).ravel()
        fitted_normal = np.array([-vy, vx], float)
        constant = float(fitted_normal @ [x0, y0])
        if (abs(np.dot([vx, vy], tangent)) < .985
                or float(np.percentile(np.abs(target @ fitted_normal-constant), 80)) > 1.5):
            return None
        lines.append((fitted_normal, constant, tangent, span, ridge_mode, half_width))
    matrix = np.array([line[0] for line in lines])
    if abs(np.linalg.det(matrix)) < .3:
        return None
    point = np.linalg.solve(matrix, [line[1] for line in lines])
    if (np.linalg.norm(point-seed) > radius*1.4 or (point < 3).any()
            or point[0] >= gx.shape[1]-3 or point[1] >= gx.shape[0]-3):
        return None
    arms = []
    for normal, _, tangent, span, ridge_mode, half_width in lines:
        support = []
        for sign in (-1, 1):
            sample = point+sign*np.linspace(4, span, 24)[:, None]*tangent
            inside = ((sample >= 2).all(axis=1) & (sample[:, 0] < gx.shape[1]-2)
                      & (sample[:, 1] < gx.shape[0]-2))
            if not inside.all():
                support.append('unavailable')
                continue
            if ridge_mode:
                scan = sample[:, None]+np.arange(-half_width, half_width+1)[None, :, None]*normal
                strength = _paired_ridges(_response(gx, gy, scan, normal, signed=True), half_width)[:, 0]
            else:
                strength = _response(gx, gy, sample, normal)
            fraction = float(np.mean(strength >= minimum))
            support.append('present' if fraction >= .65 else 'absent' if fraction <= .2 else 'uncertain')
        arms.append(support)
    return point, arms


def _nodes(rgb, native, merge):
    nodes = []
    for module_index, (r, n) in enumerate(zip(rgb, native, strict=True)):
        for corner in range(4):
            candidates = [node for node in nodes if np.linalg.norm(node['seed']-r[corner]) <= merge]
            if candidates:
                node = min(candidates, key=lambda item: np.linalg.norm(item['seed']-r[corner]))
            else:
                node = {'seed': r[corner], 'items': []}
                nodes.append(node)
            node['items'].append((module_index, corner))
    for node in nodes:
        members = node['items']
        node['seed'] = np.median([rgb[i][c] for i, c in members], axis=0)
        node['native_seed'] = np.median([native[i][c] for i, c in members], axis=0)
    return sorted(nodes, key=lambda node: tuple(node['seed']))


def _normalised_fit(source, target):
    """Condition both coordinate grids before DLT; return original pixel coordinates."""
    transforms = []
    points = []
    for xy in (source, target):
        centre = xy.mean(axis=0)
        rms = float(np.sqrt(np.mean(np.sum((xy-centre)**2, axis=1))))
        if rms < 1e-6 or np.linalg.matrix_rank(xy-centre) < 2:
            raise ValueError('degenerate fit geometry')
        scale = math.sqrt(2)/rms
        matrix = np.array([[scale, 0, -scale*centre[0]],
                           [0, scale, -scale*centre[1]], [0, 0, 1]])
        transforms.append(matrix)
        points.append(transform_points(xy, matrix))
    fit = fit_homography(*points)
    matrix = np.linalg.inv(transforms[1]) @ fit.matrix @ transforms[0]
    return matrix / matrix[2, 2]


def _resolve_matches(nodes, rgb_tolerance):
    """Enforce one-to-one observations in BOTH sensors before the fit/check split."""
    matches = [i for i, node in enumerate(nodes) if node['status'] == 'matched_provisional']
    parent = {i: i for i in matches}

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for offset, first in enumerate(matches):
        for second in matches[offset+1:]:
            rd = np.linalg.norm(np.array(nodes[first]['rgb_xy'])-nodes[second]['rgb_xy'])
            nd = np.linalg.norm(np.array(nodes[first]['native_xy'])-nodes[second]['native_xy'])
            if rd <= rgb_tolerance or nd < 3:
                parent[find(second)] = find(first)
    components = {}
    for index in matches:
        components.setdefault(find(index), []).append(index)
    resolved = []
    for members in components.values():
        rgb = np.array([nodes[i]['rgb_xy'] for i in members])
        native = np.array([nodes[i]['native_xy'] for i in members])
        rd = np.linalg.norm(rgb[:, None]-rgb[None], axis=2).max()
        nd = np.linalg.norm(native[:, None]-native[None], axis=2).max()
        if rd > rgb_tolerance or nd >= 3:
            for index in members:
                nodes[index].update(status='ambiguous_correspondence', is_corner_anchor=False)
            continue
        canonical = min(members)
        nodes[canonical].update(rgb_xy=np.median(rgb, axis=0).tolist(),
                                native_xy=np.median(native, axis=0).tolist(),
                                is_corner_anchor=all(nodes[i]['is_corner_anchor'] for i in members))
        resolved.append(canonical)
        for index in members:
            if index != canonical:
                nodes[index].update(status='duplicate_observed_junction',
                                    canonical_node_index=canonical, is_corner_anchor=False)
    return sorted(resolved)


def match_table_junctions(rgb_gray, native_gray, rgb_quads, prior_native_quads, *, config=None):
    """Match observed line intersections and fit native→RGB on fixed train nodes.

    Inputs must belong to ONE visible table in ONE pair with the same corner order.
    A bounded seed cannot resolve a wrong periodic prior. Two compatible L-corner
    observations are reported as anchor evidence, never proof of physical identity.
    Labels/check points must still validate this mapping separately before use.
    """
    import cv2

    cfg = config or JunctionConfig()
    rgb_grad, native_grad = _gradients(rgb_gray), _gradients(native_gray)
    rgb, native = [np.asarray(q, np.float32) for q in rgb_quads], [
        np.asarray(q, np.float32) for q in prior_native_quads]
    if len(rgb) != len(native):
        raise ValueError('matching RGB/native quad lists required')
    for quads in (rgb, native):
        if any(q.shape != (4, 2) or not np.isfinite(q).all() or not cv2.isContourConvex(q)
               or abs(cv2.contourArea(q)) < 16 for q in quads):
            raise ValueError('finite convex quads in matching corner order required')
    result = {'schema': 'pvdelta.junction-registration/v1', 'config': asdict(cfg),
              'image_consistent': False, 'anchor_evidence': 'unresolved', 'native_to_rgb': None,
              'measurement_status': 'withheld_requires_independent_correspondence_validation',
              'reasons': [], 'nodes': [], 'train_node_indices': [], 'check_node_indices': []}
    options = {'module_side_only': True} if cfg.module_side_only else {}
    if cfg.full_module_edge_support:
        options['full_module_edge_support'] = True
    for node in _nodes(rgb, native, cfg.rgb_merge_px):
        index, corner = node['items'][0]
        neighbours = [(corner+1) % 4, (corner-1) % 4]
        rgb_directions = np.array([rgb[index][c]-rgb[index][corner] for c in neighbours])
        native_directions = np.array([native[index][c]-native[index][corner] for c in neighbours])
        rgb_lengths = np.linalg.norm(rgb_directions, axis=1)
        native_lengths = np.linalg.norm(native_directions, axis=1)
        # Same ordered edge vectors in both sensors preserve arm signs under rotation.
        radius = min(cfg.native_search_px, int(native_lengths.min()*.3))
        record = {'seed_rgb_xy': node['seed'].tolist(), 'seed_native_xy': node['native_seed'].tolist(),
                  'status': 'rgb_junction_unresolved', 'is_corner_anchor': False}
        result['nodes'].append(record)
        if (node['native_seed'] < 3-radius).any() or (
                node['native_seed'][0] >= native_grad[0].shape[1]-3+radius
                or node['native_seed'][1] >= native_grad[0].shape[0]-3+radius):
            record['status'] = 'native_seed_outside_frame'
            continue
        if radius < 2:
            record['status'] = 'native_module_too_small'
            continue
        r = _find_junction(rgb_grad, node['seed'], rgb_directions, rgb_lengths,
                           min(cfg.rgb_search_px, max(2, int(rgb_lengths.min()*.15))), cfg.min_gradient, **options)
        if r is None:
            continue
        local_prior = cv2.getPerspectiveTransform(rgb[index], native[index])
        seed = transform_points(r[0][None], local_prior)[0]
        n = _find_junction(native_grad, seed, native_directions, native_lengths, radius, cfg.min_gradient, **options)
        if n is None:
            record.update(status='native_junction_unresolved', rgb_xy=r[0].tolist())
            continue
        record.update(status='matched_provisional', rgb_xy=r[0].tolist(), native_xy=n[0].tolist(),
                      rgb_arms=r[1], native_arms=n[1])
        # A real visible table corner must terminate BOTH frame lines in both sensors.
        record['is_corner_anchor'] = all(
            sorted(a) == ['absent', 'present'] and a == b for a, b in zip(r[1], n[1], strict=True))
    matches = _resolve_matches(result['nodes'], cfg.rgb_merge_px)
    anchors = [i for i in matches if result['nodes'][i]['is_corner_anchor']]
    if len(anchors) >= 2:
        distance = max(np.linalg.norm(np.array(result['nodes'][i]['native_xy'])-
                                     result['nodes'][j]['native_xy']) for i in anchors for j in anchors)
        shortest = min(np.linalg.norm(q-np.roll(q, -1, axis=0), axis=1).min() for q in native)
        if distance >= 1.5*shortest:
            result['anchor_evidence'] = 'two_compatible_visible_table_corners_provisional'
    if len(matches) < 10:
        result['reasons'].append('at least ten distinct observed junction matches required')
        return result
    if cfg.measurement_control_split and len(matches) < 12:
        result['reasons'].append('at least twelve distinct matches for eight fit and four check controls required')
        return result
    # Spatially deterministic split, fixed before model fitting, never selected by residual.
    fit_positions = [i for i in range(len(matches)) if (
        i % 3 != 1 if cfg.measurement_control_split else
        (i//2) % 2 == 0 if cfg.paired_block_split else i % 2 == 0)]
    check_positions = [i for i in range(len(matches)) if i not in fit_positions]
    train = [matches[i] for i in fit_positions]
    check = [matches[i] for i in check_positions]
    result.update(train_node_indices=train, check_node_indices=check)
    n = np.array([result['nodes'][i]['native_xy'] for i in matches])
    r = np.array([result['nodes'][i]['rgb_xy'] for i in matches])
    if np.linalg.matrix_rank(n-n.mean(axis=0)) < 2 or np.linalg.matrix_rank(r-r.mean(axis=0)) < 2:
        result['reasons'].append('observed junction geometry is collinear')
        return result
    for role, indices in (('fit', fit_positions), ('check', check_positions)):
        for points in (n[indices], r[indices]):
            # The combined set can span a rectangle while alternating roles
            # put all fit controls on one edge and all checks on the other.
            singular = np.linalg.svd(points-points.mean(axis=0), compute_uv=False)
            if singular[0] <= 0 or singular[1] <= singular[0]*1e-3:
                result['reasons'].append(f'{role} junction geometry is collinear')
                return result
    try:
        matrix = _normalised_fit(n[fit_positions], r[fit_positions])
        validation = validate_homography_checks(matrix, fit_point_count=len(train),
            check_source_xy=n[check_positions], check_target_xy=r[check_positions], max_p95_source_px=cfg.max_check_p95_px,
            min_fit_points=8 if cfg.measurement_control_split else 4)
        result['validation'] = asdict(validation)
        if not validation.accepted:
            result['reasons'].extend(validation.reasons)
            return result
        projected = transform_points(np.concatenate(rgb), np.linalg.inv(matrix))
        prior = np.concatenate(native)
        hull = cv2.convexHull(np.float32(n[fit_positions]))
        inside = np.array([cv2.pointPolygonTest(hull, tuple(map(float, p)), True) >= 0 for p in prior])
        if inside.sum() < 4:
            result['reasons'].append('insufficient prior corners inside fitted support')
            return result
        if (not np.isfinite(projected[inside]).all()
                or np.linalg.norm(projected[inside]-prior[inside], axis=1).max() > cfg.native_search_px*1.5):
            result['reasons'].append('model extrapolates beyond bounded native prior')
            return result
    except (ValueError, np.linalg.LinAlgError):
        result['reasons'].append('junction geometry yields an invalid projective model')
        return result
    result.update(image_consistent=True, native_to_rgb=matrix.tolist(),
                  check_support_hull_native_xy=cv2.convexHull(np.float32(n[check_positions])).reshape(-1, 2).tolist(),
                  support_hull_native_xy=cv2.convexHull(np.float32(n[fit_positions])).reshape(-1, 2).tolist())
    return result
