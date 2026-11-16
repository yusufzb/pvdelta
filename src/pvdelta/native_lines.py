"""Observed native line intersections supplement closed thermal contours.

All four sides need observed source support. No table grid is extrapolated;
geometric proposals remain unreviewed and never authorise radiometry by themselves.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.edge_support import combine_edge_evidence, native_edge_continuity
from pvdelta.junction_registration import radiometric_structure
from pvdelta.native_boundaries import NativeBoundaryConfig, _neighbours, native_panel_proposals
from pvdelta.native_search_trace import begin_trace


@dataclass(frozen=True)
class NativeLineConfig:
    min_segment_px: float = 24
    merge_distance_px: float = 3
    max_angle_deg: float = 5
    min_side_coverage: float = .8
    max_gap_fraction: float = .18
    max_pair_product: int = 250_000
    geometry_chunk_pairs: int = 16_384

    def __post_init__(self):
        values = tuple(asdict(self).values())
        if (any(isinstance(v, bool) for v in values) or not np.isfinite(values).all()
                or min(values) <= 0 or self.max_angle_deg > 10
                or self.min_side_coverage > 1 or self.max_gap_fraction >= 1):
            raise ValueError('finite positive native line limits required')
        for name, upper in (('max_pair_product', 2_000_000), ('geometry_chunk_pairs', 32_768)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                raise ValueError(f'{name} must be an integer in 1..{upper}')


def _coverage(intervals, low, high):
    """Union coverage and largest gap; disconnected intervals never fill gaps."""
    cursor, covered, gap = low, 0., 0.
    for start, end in sorted(intervals):
        start, end = max(start, low), min(end, high)
        if end <= start:
            continue
        gap = max(gap, start-cursor)
        covered += max(0., end-max(cursor, start))
        cursor = max(cursor, end)
    gap = max(gap, high-cursor)
    return covered/(high-low), gap/(high-low)


def _cluster_lines(segments, axis, distance):
    """Join collinear observations, retaining perspective slope in source space.

    A common rotation is only a search basis, not a rectangular panel constraint.
    Bound the residual spread of ALL endpoints so short fragments cannot chain
    two distinct parallel borders into one. Intervals retain every observed gap.
    """
    normal = 1-axis
    clusters = []
    for segment in sorted(segments, key=lambda s: -np.linalg.norm(s[1]-s[0])):
        choices = []
        for index, cluster in enumerate(clusters):
            points = np.concatenate([*cluster, segment])
            along, across = points[:, axis], points[:, normal]
            centred = along-along.mean()
            slope = float(centred @ (across-across.mean())/(centred @ centred))
            residual = across-slope*along
            spread = float(np.ptp(residual))
            if spread <= distance:
                choices.append((spread, index))
        if choices:
            clusters[min(choices)[1]].append(segment)
        else:
            clusters.append([segment])
    families = []
    for cluster in clusters:
        points = np.concatenate(cluster)
        along, across = points[:, axis], points[:, normal]
        centred = along-along.mean()
        slope = float(centred @ (across-across.mean())/(centred @ centred))
        families.append({'offset': float(np.mean(across-slope*along)), 'slope': slope,
            'intervals': [(float(s[:, axis].min()), float(s[:, axis].max())) for s in cluster]})
    return sorted(families, key=lambda line: line['offset'])


def _intersection(horizontal, vertical):
    """Intersect y=mh*x+bh and x=mv*y+bv in the search basis."""
    mh, mv = horizontal.get('slope', 0.), vertical.get('slope', 0.)
    x = (vertical['offset']+mv*horizontal['offset'])/(1-mh*mv)
    return np.array([x, mh*x+horizontal['offset']])


def _image_domain_pairs(family, perpendicular, along_bounds, min_side):
    """Conservative pair prefilter in the actual projected image domain.

    Intercept separation at coordinate zero is not a panel width: perspective
    lines may diverge (or reverse order) before reaching the observed panel.
    A crossing side has length gap(t)*sqrt(1+p**2)/abs(1-m*p), where
    p is its slope and m is the other family's slope. Use conservative bounds
    on both slopes. Maximum absolute separation of two affine lines over an
    interval occurs at an endpoint. This necessary bound
    cannot discard a finite-image quad satisfying the downstream side test.
    Final intersections, convexity, side support and budgets still apply.
    """
    max_slope = max((abs(line['slope']) for line in perpendicular), default=0.)
    pairs = []
    for i, first in enumerate(family):
        for j in range(i+1, len(family)):
            second = family[j]
            determinant_bound = max(0., 1-max(abs(first['slope']), abs(second['slope']))*max_slope)
            minimum_projected = min_side*determinant_bound/np.hypot(1., max_slope)
            separation = second['offset']-first['offset']
            slope = second['slope']-first['slope']
            maximum = max(abs(separation+slope*t) for t in along_bounds)
            if maximum+1e-9 >= minimum_projected:
                pairs.append((i, j, separation))
    return pairs


def _families(gray, cfg, *, extra_gray=None):
    import cv2

    sources = [gray] if extra_gray is None else [gray, extra_gray]
    detected = [cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(source)[0] for source in sources]
    detected = [np.asarray(lines).reshape(-1, 2, 2) for lines in detected if lines is not None]
    if not detected:
        return None
    # OpenCV versions expose either (N, 1, 4) or (N, 4).
    segments = np.concatenate(detected).astype(float)
    delta = segments[:, 1]-segments[:, 0]
    lengths = np.linalg.norm(delta, axis=1)
    segments, lengths, delta = segments[lengths >= cfg.min_segment_px], lengths[lengths >= cfg.min_segment_px], delta[lengths >= cfg.min_segment_px]
    if len(segments) < 4:
        return None
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    reduced = np.mod(angles, np.pi/2)
    # The histogram has periodic boundaries. Sum nearby bins on the circle.
    bins = np.linspace(0, np.pi/2, 91)[:-1]
    distance = np.abs((reduced[:, None]-bins[None, :]+np.pi/4) % (np.pi/2)-np.pi/4)
    weights = ((distance <= np.deg2rad(cfg.max_angle_deg))*lengths[:, None]).sum(axis=0)
    selected = distance[:, int(np.argmax(weights))] <= np.deg2rad(cfg.max_angle_deg)
    angle = float(np.angle(np.sum(lengths[selected]*np.exp(4j*angles[selected])))/4)
    basis = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    projected = segments @ basis
    direction = projected[:, 1]-projected[:, 0]
    groups = [[], []]
    axes = (np.abs(direction[:, 1]) > np.abs(direction[:, 0])).astype(int)
    # Perspective need not preserve right angles. Select each observed direction
    # independently; do not discard one rack's seams merely because another
    # direction dominates the common search basis.
    for axis in (0, 1):
        indices = np.flatnonzero(axes == axis)
        if not len(indices):
            continue
        vectors = direction[indices]
        slopes = np.arctan(vectors[:, 1-axis]/vectors[:, axis])
        bins = np.linspace(-np.pi/4, np.pi/4, 91)
        distance = np.abs(slopes[:, None]-bins[None, :])
        weights = ((distance <= np.deg2rad(cfg.max_angle_deg))*lengths[indices, None]).sum(axis=0)
        selected = distance[:, int(np.argmax(weights))] <= np.deg2rad(cfg.max_angle_deg)
        groups[axis] = list(projected[indices[selected]])
    families = [_cluster_lines(group, axis, cfg.merge_distance_px)
                for axis, group in enumerate(groups)]
    return basis, families, len(segments)


def _supported(line, low, high, cfg):
    coverage, gap = _coverage(line['intervals'], low, high)
    return coverage >= cfg.min_side_coverage and gap <= cfg.max_gap_fraction, coverage, gap


def _local_structure(temperature_c):
    """Local contrast for geometry only; the measured grid is never returned/changed."""
    import cv2

    values = np.asarray(temperature_c, np.float32)
    residual = values-cv2.GaussianBlur(values, (0, 0), 5)
    scale = np.sqrt(cv2.GaussianBlur(residual*residual, (0, 0), 10))
    normalised = residual/np.maximum(scale, .05)
    return np.rint(np.clip((normalised+3)/6, 0, 1)*255).astype(np.uint8)


def _quad_candidates(horizontal, vertical, pairs, cfg, pixel_count, *, chunk_pairs=16_384, trace=None):
    """Exact original pair order and filters with bounded intermediate arrays."""
    if isinstance(chunk_pairs, bool) or not isinstance(chunk_pairs, int) or not 1 <= chunk_pairs <= 32_768:
        raise ValueError('chunk_pairs must be an integer in 1..32768')
    if not pairs[0] or not pairs[1]:
        return
    vp, hp = np.asarray(pairs[0], int)[:, :2], np.asarray(pairs[1], int)[:, :2]
    h = np.array([[line['offset'], line['slope']] for line in horizontal])
    v = np.array([[line['offset'], line['slope']] for line in vertical])
    for start in range(0, len(vp)*len(hp), chunk_pairs):
        indices = np.arange(start, min(start+chunk_pairs, len(vp)*len(hp)))
        left, right = vp[indices//len(hp)].T
        top, bottom = hp[indices%len(hp)].T
        options = {'trace': trace} if trace is not None else {}
        yield from _quad_chunk(h, v, left, right, top, bottom, cfg, pixel_count, **options)


def _quad_chunk(h, v, left, right, top, bottom, cfg, pixel_count, *, trace=None):
    """Apply unchanged per-quad geometry tests to one bounded pair slice."""

    def intersections(hi, vi):
        bh, mh = h[hi].T
        bv, mv = v[vi].T
        x = (bv+mv*bh)/(1-mh*mv)
        return np.column_stack((x, mh*x+bh))

    corners = np.stack([intersections(top, left), intersections(top, right),
                        intersections(bottom, right), intersections(bottom, left)], axis=1)
    shifted = np.roll(corners, -1, axis=1)
    vectors = shifted-corners
    following = np.roll(vectors, -1, axis=1)
    lengths = np.linalg.norm(vectors, axis=2)
    width, height = (lengths[:, 0]+lengths[:, 2])/2, (lengths[:, 1]+lengths[:, 3])/2
    area = np.abs((corners[:, :, 0]*shifted[:, :, 1]-corners[:, :, 1]*shifted[:, :, 0]).sum(axis=1))/2
    cross = vectors[:, :, 0]*following[:, :, 1]-vectors[:, :, 1]*following[:, :, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.maximum(width, height)/np.minimum(width, height)
        cosine = (vectors*following).sum(axis=2)/(lengths*np.roll(lengths, -1, axis=1))
    valid = ((area >= cfg.min_area_px) & (area <= pixel_count*cfg.max_area_fraction)
        & (lengths.min(axis=1) >= cfg.min_side_px) & (ratio > cfg.min_aspect) & (ratio < cfg.max_aspect)
        & ((cross > 0).all(axis=1) | (cross < 0).all(axis=1))
        & (np.abs(cosine).max(axis=1) <= cfg.max_corner_cosine))
    if trace:
        trace.geometry_batch((
            ('area', (area >= cfg.min_area_px) & (area <= pixel_count*cfg.max_area_fraction)),
            ('min_side', lengths.min(axis=1) >= cfg.min_side_px),
            ('aspect', (ratio > cfg.min_aspect) & (ratio < cfg.max_aspect)),
            ('convexity', (cross > 0).all(axis=1) | (cross < 0).all(axis=1)),
            ('corner', np.abs(cosine).max(axis=1) <= cfg.max_corner_cosine)))
    for index in np.flatnonzero(valid):
        yield left[index], right[index], top[index], bottom[index], corners[index], width[index], height[index]


def _closed_scale_references(quad, contours):
    """Two disjoint closed contours can support full-module scale, not semantics.

    Used only for hybrid candidates with an ambiguous internal thermal line.
    Larger multi-panel envelopes and duplicate contours cannot supply this prior.
    """
    def shape(polygon):
        vectors = np.roll(polygon, -1, axis=0)-polygon
        lengths = np.linalg.norm(vectors, axis=1)
        return float(lengths.min()), float(lengths.max()), vectors[lengths.argmax()]/lengths.max()

    short, long, axis = shape(quad)
    references = []
    for proposal in contours:
        other = np.asarray(proposal['polygon_native_xy'], float)
        s, length, other_axis = shape(other)
        if (convex_iou(quad, other) > .05 or abs(short-s)/s > .15
                or abs(long-length)/length > .10 or abs(float(axis@other_axis)) < .985):
            continue
        for previous in references:
            if convex_iou(previous, other) <= .05:
                return [previous.tolist(), other.tolist()]
        references.append(other)
    return []


def _line_pass(temperature_c, *, config=None, line_config=None, local_contrast=True,
               support_mode='segments') -> dict:
    """Keep contour proposals and add only four-side, neighbour-supported envelopes."""
    cfg = config or NativeBoundaryConfig()
    line_cfg = line_config or NativeLineConfig()
    trace = begin_trace(temperature_c, support_mode=support_mode,
                        local_contrast=local_contrast, pair_budget=line_cfg.max_pair_product)
    result = native_panel_proposals(temperature_c, config=cfg, _trace=trace)
    gray, _ = radiometric_structure(temperature_c)
    result.update(schema='pvdelta.native-line-boundaries/v2', line_config=asdict(line_cfg))
    original = [dict(p, source_method='closed_contour') for p in result['proposals']]
    result['contour_proposal_count'] = len(original)
    result['line_proposal_count'] = 0
    result['line_support_mode'] = support_mode
    audit = dict.fromkeys(('geometric_candidates', 'rejected_side_support',
        'rejected_internal_divider', 'rejected_source_border', 'four_side_candidates',
        'deduplicated_candidates', 'rejected_neighbours'), 0)
    audit['rejected_hybrid_anchors'] = 0
    result['line_diagnostics'] = audit
    result['rejected_line_proposals'] = []
    result['line_structure'] = {'sources': ['global_percentile', 'local_contrast'] if local_contrast else ['global_percentile'],
        'local_background_sigma_px': 5, 'local_scale_sigma_px': 10,
        'scale_floor_k': .05, 'clip_sigma': 3, 'purpose': 'geometry_only_not_measurement'}
    result['pair_search'] = {'max_pair_product': line_cfg.max_pair_product,
        'geometry_chunk_pairs': line_cfg.geometry_chunk_pairs, 'pair_product': None,
        'search_complete': False, 'reason': 'insufficient_observed_segments'}
    found = _families(gray, line_cfg, extra_gray=_local_structure(temperature_c) if local_contrast else None)
    if found is None:
        result['proposals'] = original
        result['line_status'] = 'insufficient_observed_segments'
        if trace:
            trace.finish(result)
        return result
    basis, (horizontal, vertical), segment_count = found
    result['segment_count'] = segment_count
    result['line_family_counts'] = [len(horizontal), len(vertical)]
    if max(len(horizontal), len(vertical)) > 100:
        result['proposals'] = original
        result['line_status'] = 'too_many_line_families_for_bounded_search'
        result['pair_search']['reason'] = result['line_status']
        if trace:
            trace.finish(result)
        return result
    height, width = gray.shape
    image_corners = np.array([[0., 0.], [width-1., 0.],
                              [width-1., height-1.], [0., height-1.]]) @ basis
    pairs = [_image_domain_pairs(family, perpendicular,
                 (image_corners[:, axis].min(), image_corners[:, axis].max()), cfg.min_side_px)
             for family, perpendicular, axis in
             ((vertical, horizontal, 1), (horizontal, vertical, 0))]
    result['pair_search']['pair_product'] = len(pairs[0])*len(pairs[1])
    if len(pairs[0])*len(pairs[1]) > line_cfg.max_pair_product:
        result['proposals'] = original
        result['line_status'] = 'too_many_line_pairs_for_bounded_search'
        result['pair_search']['reason'] = result['line_status']
        if trace:
            trace.finish(result)
        return result
    result['pair_search'].update(search_complete=True, reason='all_pairs_geometrically_evaluated')
    candidates = []
    ridge_cache = {}
    native_values = np.asarray(temperature_c, float) if support_mode == 'hybrid' else None
    for left, right, top, bottom, corners, width, height in _quad_candidates(
            horizontal, vertical, pairs, cfg, gray.size, chunk_pairs=line_cfg.geometry_chunk_pairs,
            trace=trace):
            audit['geometric_candidates'] += 1
            support = [_supported(line, min(a[axis], b[axis]), max(a[axis], b[axis]), line_cfg)
                for line, a, b, axis in zip(
                    (horizontal[top], vertical[right], horizontal[bottom], vertical[left]),
                    corners, np.roll(corners, -1, axis=0), (0, 1, 0, 1))]
            segment_coverage = [float(s[1]) for s in support]
            native_sides = []
            if support_mode == 'hybrid' and not all(s[0] for s in support):
                # Require two already observed sides to anchor supplementation.
                # Ridge evidence alone cannot invent a four-sided panel envelope.
                if sum(s[0] for s in support) < 2:
                    audit['rejected_hybrid_anchors'] += 1
                    if trace:
                        trace.candidate('hybrid_anchors', corners @ basis.T, support)
                    continue
                source_corners = corners @ basis.T
                for side, (line, a, b, axis) in enumerate(zip(
                        (horizontal[top], vertical[right], horizontal[bottom], vertical[left]),
                        corners, np.roll(corners, -1, axis=0), (0, 1, 0, 1))):
                    if support[side][0]:
                        continue
                    key = tuple(source_corners[[side, (side+1)%4]].ravel())
                    if key not in ridge_cache:
                        ridge_cache[key] = native_edge_continuity(native_values,
                            source_corners[side], source_corners[(side+1)%4])
                    intervals = []
                    for low, high in line['intervals']:
                        start, end = sorted(((low-a[axis])/(b[axis]-a[axis]),
                                             (high-a[axis])/(b[axis]-a[axis])))
                        start, end = max(0., float(start)), min(1., float(end))
                        if end > start:
                            intervals.append([start, end])
                    combined = combine_edge_evidence(ridge_cache[key], intervals)
                    # Sampling alone must not turn a failing analytic LSD score
                    # into acceptance. Require genuinely additional ridge samples.
                    if combined['ridge_only_samples'] == 0:
                        continue
                    coverage, gap = combined['coverage'], combined['largest_gap_fraction']
                    support[side] = (coverage >= line_cfg.min_side_coverage and
                                     gap <= line_cfg.max_gap_fraction, coverage, gap)
                    native_sides.append(side)
            if not all(s[0] for s in support):
                audit['rejected_side_support'] += 1
                if trace:
                    trace.candidate('side_support', corners @ basis.T, support)
                continue
            # A full internal divider across the short dimension cannot be
            # explained away as a missing frame. Permit only one central half-cut
            # seam along the long dimension, never multiple merged panel columns.
            splitters = []
            for family, first, last, perpendicular, axis, long_axis in (
                (vertical, left, right, (horizontal[top], horizontal[bottom]), 1, width > height),
                (horizontal, top, bottom, (vertical[left], vertical[right]), 0, height > width)):
                positions = []
                # With perspective, ordering at the search origin need not be
                # ordering inside this panel. Inspect every other observed line.
                for k in range(len(family)):
                    if k in (first, last):
                        continue
                    points = np.array([_intersection(p, family[k]) if axis == 1 else
                                       _intersection(family[k], p) for p in perpendicular])
                    middle = points.mean(axis=0)
                    start = np.array([_intersection(p, family[first]) if axis == 1 else
                                      _intersection(family[first], p) for p in perpendicular]).mean(axis=0)
                    end = np.array([_intersection(p, family[last]) if axis == 1 else
                                    _intersection(family[last], p) for p in perpendicular]).mean(axis=0)
                    position = (middle[1-axis]-start[1-axis])/(end[1-axis]-start[1-axis])
                    if .12 < position < .88 and _supported(family[k], points[:, axis].min(),
                                                          points[:, axis].max(), line_cfg)[0]:
                        positions.append(position)
                splitters.append(bool(positions) and not (
                    long_axis and len(positions) == 1 and .4 < positions[0] < .6))
            if any(splitters):
                scale_refs = _closed_scale_references(corners @ basis.T, original) if (
                    support_mode == 'hybrid' and native_sides) else []
                if not scale_refs:
                    audit['rejected_internal_divider'] += 1
                    if trace:
                        trace.candidate('internal_divider', corners @ basis.T, support)
                    continue
            else:
                scale_refs = []
            quad = corners @ basis.T
            if ((quad < 3).any() or (quad[:, 0] >= gray.shape[1]-3).any()
                    or (quad[:, 1] >= gray.shape[0]-3).any()):
                audit['rejected_source_border'] += 1
                if trace:
                    trace.candidate('source_border', quad, support)
                continue
            audit['four_side_candidates'] += 1
            if trace:
                trace.candidate('four_sides_passed', quad, support)
            candidates.append({'quad': quad.astype(np.float32),
                'side_coverage': [float(s[1]) for s in support],
                'side_max_gap_fraction': [float(s[2]) for s in support],
                'native_supported_sides': native_sides,
                'scale_references': scale_refs,
                'segment_side_coverage': segment_coverage})
    # Preserve frozen contour proposals and deduplicate only supplements.
    proposals = list(original)
    for candidate in sorted(candidates, key=lambda c: min(c['side_coverage']), reverse=True):
        quad = candidate['quad']
        duplicate = False
        for proposal in proposals:
            other = np.asarray(proposal['polygon_native_xy'], np.float32)
            iou = convex_iou(quad, other)
            if iou > .65:
                duplicate = True
                audit['deduplicated_candidates'] += 1
                if trace:
                    trace.candidate('deduplicated', quad, [])
                break
        if not duplicate:
            proposals.append({'polygon_native_xy': quad.astype(float).tolist(),
                'source_method': 'four_observed_lines', 'side_coverage': candidate['side_coverage'],
                'side_max_gap_fraction': candidate['side_max_gap_fraction']})
            if candidate['native_supported_sides']:
                proposals[-1].update(source_method='observed_lines_and_native_ridges',
                    native_supported_sides=candidate['native_supported_sides'],
                    segment_side_coverage=candidate['segment_side_coverage'])
            if candidate['scale_references']:
                proposals[-1]['internal_structure_resolution'] = {
                    'status': 'same_scale_as_two_disjoint_closed_contours',
                    'reference_polygons_native_xy': candidate['scale_references'],
                    'semantic_panel_membership': 'unverified'}
    while proposals:
        kept = [p for p in proposals if p['source_method'] == 'closed_contour' or sum(
            q is not p and _neighbours(np.asarray(p['polygon_native_xy']), np.asarray(q['polygon_native_xy']))
            for q in proposals) >= cfg.min_neighbours]
        if len(kept) == len(proposals):
            break
        for proposal in proposals:
            if not any(proposal is survivor for survivor in kept):
                audit['rejected_neighbours'] += 1
                result['rejected_line_proposals'].append({
                    'polygon_native_xy': proposal['polygon_native_xy'],
                    'side_coverage': proposal['side_coverage'],
                    'side_max_gap_fraction': proposal['side_max_gap_fraction'],
                    'reason': 'insufficient_observed_neighbours',
                    'measurement_authorised': False})
                if trace:
                    trace.candidate('neighbours', proposal['polygon_native_xy'], [])
        proposals = kept
    for index, proposal in enumerate(proposals):
        proposal['proposal_id'] = f'N{index+1:03}'
        proposal['neighbour_ids'] = [f'N{j+1:03}' for j, other in enumerate(proposals)
            if j != index and _neighbours(np.asarray(proposal['polygon_native_xy']),
                                         np.asarray(other['polygon_native_xy']))]
    result['proposals'] = proposals
    result['line_proposal_count'] = len(proposals)-len(original)
    result['line_status'] = 'unreviewed_observed_line_geometry'
    if trace:
        trace.finish(result)
    return result


def _single_mode_proposals(temperature_c, *, config=None, line_config=None,
                           support_mode='segments') -> dict:
    """Try raw observed geometry if contrast supplementation removes every proposal.

    Both passes retain the same four-side, gap and neighbour gates. This is not
    a missing-edge/layout fallback and does not authorise measurement.
    Opt-in hybrid mode supplements failed sides with native ridge evidence,
    anchored by at least two already-supported segment sides.
    """
    if support_mode not in ('segments', 'hybrid'):
        raise ValueError('support_mode must be segments or hybrid')
    result = _line_pass(temperature_c, config=config, line_config=line_config,
                        support_mode=support_mode)
    result['structure_pass'] = 'global_plus_local'
    if result['proposals']:
        return result
    raw = _line_pass(temperature_c, config=config, line_config=line_config,
                     local_contrast=False, support_mode=support_mode)
    result['raw_fallback_proposal_count'] = len(raw['proposals'])
    result['raw_fallback_diagnostics'] = {key: raw.get(key) for key in (
        'line_status', 'line_diagnostics', 'rejected_line_proposals')}
    if not raw['proposals']:
        return result
    raw['structure_pass'] = 'global_only_fallback'
    raw['contrast_fallback_reason'] = 'combined_structure_pass_no_proposals'
    raw['combined_pass_line_status'] = result['line_status']
    return raw


def _area_axis(polygon):
    points = np.asarray(polygon, float)
    vectors = np.roll(points, -1, axis=0)-points
    lengths = np.linalg.norm(vectors, axis=1)
    shifted = points-points[0]
    area = abs(float(np.sum(shifted[:, 0]*np.roll(shifted[:, 1], -1)
                            - shifted[:, 1]*np.roll(shifted[:, 0], -1))))/2
    return area, lengths.min(), lengths.max(), vectors[lengths.argmax()]/lengths.max()


def _compound_children(parent, proposals):
    """Two adjacent observed children can expose an orthogonal multi-panel envelope.

    This is a structural conflict, not semantic panel approval. Require nearly
    contained, disjoint children of matching scale/orientation; a lone nested
    contour or a parallel thermal stripe cannot remove an envelope.
    """
    quad = np.asarray(parent['polygon_native_xy'], float)
    area, short, _, axis = _area_axis(quad)
    children = []
    for child in proposals:
        other = np.asarray(child['polygon_native_xy'], float)
        child_area, _, child_long, child_axis = _area_axis(other)
        if (child_area >= area*.6 or abs(float(axis@child_axis)) > .2
                or abs(child_long-short)/short > .2):
            continue
        iou = convex_iou(quad, other)
        intersection = iou*(area+child_area)/(1+iou)
        if intersection/child_area < .9:
            continue
        for previous in children:
            previous_quad = np.asarray(previous['polygon_native_xy'], float)
            if convex_iou(other, previous_quad) <= .05 and _neighbours(other, previous_quad):
                return [previous, child]
        children.append(child)
    return []


def _compose_hybrid(baseline, hybrid, cfg):
    """Stable segment geometry first; additional evidence must not replace it silently."""
    pool, duplicates = [], 0
    for origin, result in (('segment_baseline', baseline), ('hybrid_structure', hybrid)):
        for proposal in result['proposals']:
            if any(convex_iou(proposal['polygon_native_xy'], p['polygon_native_xy']) > .65
                   for p in pool):
                duplicates += 1
                continue
            pool.append(dict(proposal, proposal_origin=origin))
    # Smaller observed instances supply evidence against larger enclosing quads.
    # Do not use an already superseded envelope as a child of another envelope.
    kept, superseded = [], []
    for proposal in sorted(pool, key=lambda p: _area_axis(p['polygon_native_xy'])[0]):
        children = _compound_children(proposal, kept)
        if children:
            superseded.append(dict(proposal, reason='contains_adjacent_orthogonal_instances',
                child_polygons_native_xy=[p['polygon_native_xy'] for p in children],
                measurement_authorised=False))
        else:
            kept.append(proposal)
    # Keep original deterministic ordering after structural conflict resolution.
    proposals = [p for p in pool if any(p is survivor for survivor in kept)]
    rejected = []
    while proposals:
        kept = [p for p in proposals if p['source_method'] == 'closed_contour' or sum(
            q is not p and _neighbours(np.asarray(p['polygon_native_xy']),
                                      np.asarray(q['polygon_native_xy']))
            for q in proposals) >= cfg.min_neighbours]
        if len(kept) == len(proposals):
            break
        rejected.extend(dict(p, reason='insufficient_neighbours_after_composition',
                             measurement_authorised=False)
                        for p in proposals if not any(p is q for q in kept))
        proposals = kept
    for index, proposal in enumerate(proposals):
        proposal['proposal_id'] = f'N{index+1:03}'
        proposal['neighbour_ids'] = [f'N{j+1:03}' for j, q in enumerate(proposals)
            if index != j and _neighbours(np.asarray(proposal['polygon_native_xy']),
                                         np.asarray(q['polygon_native_xy']))]
    result = dict(hybrid, proposals=proposals, structure_pass='hybrid_with_segment_baseline',
        superseded_compound_proposals=superseded, composition_rejected_proposals=rejected,
        composition_diagnostics={'baseline_structure_pass':baseline['structure_pass'],
            'hybrid_structure_pass':hybrid['structure_pass'],
            'baseline_input_proposals':len(baseline['proposals']),
            'hybrid_input_proposals':len(hybrid['proposals']),
            'deduplicated_proposals':duplicates,'superseded_compounds':len(superseded),
            'rejected_neighbours':len(rejected)},
        measurement_authorised=False)
    result['contour_proposal_count'] = sum(p['source_method']=='closed_contour' for p in proposals)
    result['line_proposal_count'] = len(proposals)-result['contour_proposal_count']
    return result


def native_line_proposals(temperature_c, *, config=None, line_config=None,
                          support_mode='segments') -> dict:
    """Default observed segments, or opt-in hybrid with explicit conflict provenance.

    Hybrid always evaluates the segment baseline including its global fallback.
    It retains baseline geometry over near duplicates, and records multi-instance
    envelopes separately when two disjoint adjacent children expose a conflict.
    These remain unreviewed geometry proposals, never measurement approval.
    """
    if support_mode not in ('segments', 'hybrid'):
        raise ValueError('support_mode must be segments or hybrid')
    baseline = _single_mode_proposals(temperature_c, config=config, line_config=line_config)
    if support_mode == 'segments':
        return baseline
    hybrid = _single_mode_proposals(temperature_c, config=config, line_config=line_config,
                                    support_mode='hybrid')
    return _compose_hybrid(baseline, hybrid, config or NativeBoundaryConfig())
