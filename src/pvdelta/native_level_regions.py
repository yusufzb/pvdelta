"""Research closed level regions, complementary to narrow thermal edge models."""
from __future__ import annotations

import hashlib

import numpy as np

from pvdelta.boundaries import order_quad
from pvdelta.convex_geometry import convex_iou
from pvdelta.native_line_groups import _valid_affine


def level_region_candidates(temperature_c, *, max_observations=8192):
    """Collect full observed components persistent across temperature levels.

    Nineteen equally spaced levels within each native window's 2–98 percentile
    range, both polarities, five fixed windows. No Canny/LSD/ridge fit, inferred
    grid, reference corners or temperature resampling. The region's own convex
    hull supplies its corners; at least two adjacent distinct levels must agree
    at IoU .8 in one window and polarity. Multiple levels are NOT independent
    observations. Holes count against actual component occupancy.

    Regular non-PV rectangles can pass. Caller must separately check source
    identity, scale/neighbours/conflicts/old support and panel semantics. Budget
    overflow withholds the entire pool, never a temperature-ranked subset.
    """
    import cv2

    values = np.asarray(temperature_c)
    if (values.ndim != 2 or min(values.shape, default=0) < 32
            or values.dtype.kind not in 'fiu' or not values.flags.c_contiguous
            or not np.isfinite(values).all()):
        raise ValueError('Finite contiguous native numeric 2D grid, minimum 32 per axis, required')
    if type(max_observations) is not int or not 1 <= max_observations <= 8192:
        raise ValueError('max_observations must be an integer in 1..8192')
    low, high = np.percentile(values, [2, 98])
    if high-low <= 1e-6:
        raise ValueError('Native radiometry has insufficient structural dynamic range')
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    height, width = values.shape
    windows = [(0, 0, width, height)]
    if min(values.shape) >= 48:
        h, w = (2*height+2)//3, (2*width+2)//3
        windows.extend((x, y, x+w, y+h) for y in (0, height-h) for x in (0, width-w))
    observed, seen, skipped, overflow = [], 0, [], False
    for x0, y0, right, bottom in windows:
        grid = values[y0:bottom, x0:right]
        lo, hi = np.percentile(grid, [2, 98])
        if hi-lo <= 1e-6:
            skipped.append([x0, y0, right, bottom])
            continue
        for polarity in ('above', 'below'):
            for index, fraction in enumerate(np.linspace(.05, .95, 19)):
                threshold = float(lo+fraction*(hi-lo))
                mask = np.asarray(grid > threshold if polarity == 'above' else grid < threshold, np.uint8)
                _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
                    if not 500 <= area <= values.size*.08 or min(w, h) < 18:
                        continue
                    # Window-clipped components never supply a missing side.
                    if x < 3 or y < 3 or x+w >= grid.shape[1]-3 or y+h >= grid.shape[0]-3:
                        continue
                    component = np.asarray(labels[y:y+h, x:x+w] == label, np.uint8)
                    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) != 1:
                        continue
                    hull = cv2.convexHull(contours[0])
                    quad = cv2.approxPolyDP(hull, .018*cv2.arcLength(hull, True), True)
                    if len(quad) != 4:
                        continue
                    q = order_quad(quad.reshape(4, 2)).astype(float)+[x0+x, y0+y]
                    polygon_area = float(cv2.contourArea(q.astype(np.float32)))
                    occupancy = float(area)/polygon_area if polygon_area else 0.
                    if not .90 <= occupancy <= 1.12 or not _valid_affine(q, values.shape):
                        continue
                    seen += 1
                    if seen > max_observations:
                        overflow = True
                        break
                    observed.append({'quad': q, 'area': polygon_area, 'level_index': index,
                        'threshold_c': threshold, 'occupancy': occupancy, 'component_pixels': int(area),
                        'polarity': polarity, 'window_xyxy': [x0, y0, right, bottom]})
                if overflow:
                    break
            if overflow:
                break
        if overflow:
            break
    # Largest observed region is the representative; do not extrapolate corners.
    groups = []
    if not overflow:
        for item in sorted(observed, key=lambda p: (-p['area'], *p['quad'].ravel(),
                p['window_xyxy'], p['polarity'], p['level_index'])):
            group = next((g for g in groups if convex_iou(item['quad'], g[0]['quad']) >= .8), None)
            if group is None:
                groups.append([item])
                if len(groups) > 512:
                    overflow = True
                    break
            else:
                group.append(item)
    retained = []
    if not overflow:
        for group in groups:
            representative = group[0]
            streams = {}
            for item in group:
                key = (*item['window_xyxy'], item['polarity'])
                streams.setdefault(key, {})[item['level_index']] = item
            pairs = [(a, stream[i+1]) for stream in streams.values()
                     for i, a in stream.items() if i+1 in stream
                     and a['threshold_c'] != stream[i+1]['threshold_c']]
            if not pairs:
                continue
            a, b = min(pairs, key=lambda pair: (pair[0]['window_xyxy'], pair[0]['polarity'], pair[0]['level_index']))
            retained.append({'quad': representative['quad'], 'evidence':{
                'schema': 'pvdelta.native-level-region-evidence/v1', 'purpose': 'geometry_only',
                'native_temperature_sha256': digest, 'native_shape': list(values.shape),
                'representative_component_pixels': representative['component_pixels'],
                'representative_occupancy': representative['occupancy'],
                'representative_window_xyxy': representative['window_xyxy'],
                'representative_polarity': representative['polarity'],
                'representative_threshold_c': representative['threshold_c'],
                'support_window_xyxy': a['window_xyxy'], 'support_polarity': a['polarity'],
                'support_levels': [{'level_index': p['level_index'], 'threshold_c': p['threshold_c'],
                    'polygon_native_xy': p['quad'].tolist(), 'component_pixels': p['component_pixels'],
                    'occupancy': p['occupancy'], 'iou_to_representative': convex_iou(p['quad'], representative['quad'])}
                    for p in (a, b)],
                'measurement_authorised': False}})
        retained.sort(key=lambda p: (*p['quad'].mean(axis=0)[::-1], *p['quad'].ravel()))
    if hashlib.sha256(values.tobytes()).hexdigest() != digest:
        raise RuntimeError('Native grid changed during level-region extraction')
    return {'schema': 'pvdelta.native-level-regions/v1',
        'status': 'level_region_budget_exceeded' if overflow else 'unreviewed_level_region_geometry',
        'proposals': [] if overflow else [{'proposal_id': f'V{i+1:04d}',
            'polygon_native_xy': p['quad'].tolist(), 'source_method': 'native_level_regions',
            'native_level_region_evidence': p['evidence'], 'proposal_origin': 'native_level_region_search',
            'semantic_panel_membership': 'unverified', 'measurement_authorised': False} for i, p in enumerate(retained)],
        'native_temperature_sha256': digest, 'native_shape': list(values.shape),
        'raw_region_observations': seen, 'region_groups': len(groups), 'skipped_windows': skipped,
        'limits': {'levels_per_polarity': 19, 'min_occupancy': .90, 'persistence_iou': .8,
            'max_observations': max_observations, 'max_groups': 512},
        'radiometric_resampling': False, 'measurement_authorised': False,
        'semantic_panel_membership': 'unverified', 'efficacy_accepted': False}


def supported_level_region(proposal):
    """Validate declared region evidence; source identity remains caller's duty."""
    evidence = proposal.get('native_level_region_evidence')
    try:
        if (not isinstance(evidence, dict) or evidence['schema'] != 'pvdelta.native-level-region-evidence/v1'
                or evidence['purpose'] != 'geometry_only' or evidence['measurement_authorised'] is not False
                or evidence['support_polarity'] not in ('above', 'below')
                or evidence['representative_polarity'] not in ('above', 'below')
                or not .9 <= evidence['representative_occupancy'] <= 1.12):
            return False
        digest = evidence['native_temperature_sha256']
        shape = evidence['native_shape']
        if (not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                or not isinstance(shape, list) or len(shape) != 2
                or any(type(n) is not int or n < 32 for n in shape)):
            return False
        for name in ('support_window_xyxy', 'representative_window_xyxy'):
            window = evidence[name]
            if (not isinstance(window, list) or len(window) != 4 or any(type(n) is not int for n in window)
                    or not 0 <= window[0] < window[2] <= shape[1]
                    or not 0 <= window[1] < window[3] <= shape[0]):
                return False
        levels = evidence['support_levels']
        if not isinstance(levels, list) or len(levels) != 2:
            return False
        if (any(type(p['level_index']) is not int or not 0 <= p['level_index'] < 19 for p in levels)
                or levels[1]['level_index'] != levels[0]['level_index']+1
                or levels[1]['threshold_c'] <= levels[0]['threshold_c']):
            return False
        for p in levels:
            if (not np.isfinite([p['threshold_c'], p['occupancy'], p['iou_to_representative']]).all()
                    or not .9 <= p['occupancy'] <= 1.12 or type(p['component_pixels']) is not int
                    or p['component_pixels'] < 500 or not .8 <= p['iou_to_representative'] <= 1
                    or abs(convex_iou(p['polygon_native_xy'], proposal['polygon_native_xy'])
                           -p['iou_to_representative']) > 1e-9):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True
