"""Experimental native thermal panel envelopes, independent of RGB registration.

Only observed closed contours are proposed. Rectangle/neighbour evidence is not
semantic panel validation; this module never authorises measurement or deletes
findings. Radiometric temperatures are used unchanged by downstream consumers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from pvdelta.boundaries import order_quad
from pvdelta.junction_registration import radiometric_structure


@dataclass(frozen=True)
class NativeBoundaryConfig:
    min_area_px: float = 500
    min_side_px: float = 18
    max_area_fraction: float = .08
    min_aspect: float = 1.7
    max_aspect: float = 2.8
    max_corner_cosine: float = .15
    min_threshold_votes: int = 2
    min_neighbours: int = 1

    def __post_init__(self):
        values = (self.min_area_px, self.min_side_px, self.max_area_fraction,
                  self.min_aspect, self.max_aspect, self.max_corner_cosine)
        if (not all(np.isfinite(v) for v in values) or min(values) <= 0
                or self.max_area_fraction >= 1 or self.min_aspect >= self.max_aspect
                or self.max_corner_cosine >= 1):
            raise ValueError('native boundary geometry limits must be finite, positive and ordered')
        if (isinstance(self.min_threshold_votes, bool)
                or not isinstance(self.min_threshold_votes, int)
                or not 1 <= self.min_threshold_votes <= 3
                or isinstance(self.min_neighbours, bool)
                or not isinstance(self.min_neighbours, int) or self.min_neighbours < 1):
            raise ValueError('threshold votes must be integer 1..3 and neighbours positive integer')


def _geometry(quad):
    vectors = np.roll(quad, -1, axis=0)-quad
    lengths = np.linalg.norm(vectors, axis=1)
    index = int(np.argmax(lengths))
    return lengths, vectors[index]/lengths[index]


def _neighbours(first, second):
    lengths_a, axis = _geometry(first)
    lengths_b, other_axis = _geometry(second)
    short, long = (lengths_a.min()+lengths_b.min())/2, (lengths_a.max()+lengths_b.max())/2
    if (abs(float(axis @ other_axis)) < .985
            or abs(lengths_a.min()-lengths_b.min()) > short*.2
            or abs(lengths_a.max()-lengths_b.max()) > long*.2):
        return False
    delta = second.mean(axis=0)-first.mean(axis=0)
    along = abs(float(delta @ axis))
    across = abs(float(delta @ np.array([-axis[1], axis[0]])))
    return bool((abs(along-long) < long*.25 and across < short*.2)
                or (abs(across-short) < short*.25 and along < long*.2))


def native_panel_proposals(temperature_c, *, config: NativeBoundaryConfig | None = None, _trace=None) -> dict:
    """Propose full native contours with scale/orientation-compatible neighbours.

    No RGB polygon/pose/CAD is consumed. Default aspect bounds are experimental
    for the current landscape PV installation and need testing on other flights.
    Canny thresholds are votes from the same source image, not independent data.
    Nonfinite/flat frames fail explicitly; no temperature is interpolated.
    """
    import cv2

    cfg = config or NativeBoundaryConfig()
    gray, structure = radiometric_structure(temperature_c)
    smooth = cv2.GaussianBlur(gray, (3, 3), .6)
    candidates = []
    counts = dict.fromkeys(('observed_contours', 'area_rejected', 'not_four_corners',
                           'min_side_rejected', 'aspect_rejected', 'corner_or_fill_rejected',
                           'source_border_rejected', 'passed_geometry'), 0) if _trace else None
    for threshold_index, (low, high) in enumerate(((10, 30), (20, 60), (35, 100))):
        edge = cv2.Canny(smooth, low, high)
        for close in (False, True):
            binary = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) if close else edge
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if _trace:
                    counts['observed_contours'] += 1
                area = cv2.contourArea(contour)
                if not cfg.min_area_px <= area <= gray.size*cfg.max_area_fraction:
                    if _trace:
                        counts['area_rejected'] += 1
                    continue
                hull = cv2.convexHull(contour)
                quad = cv2.approxPolyDP(hull, .018*cv2.arcLength(hull, True), True)
                if len(quad) != 4:
                    if _trace:
                        counts['not_four_corners'] += 1
                    continue
                quad = order_quad(quad.reshape(4, 2))
                lengths, _ = _geometry(quad)
                if _trace:
                    aspect = float(lengths.max()/lengths.min())
                    trace_vectors = np.roll(quad, -1, axis=0)-quad
                    trace_cosine = float(np.abs(np.sum(trace_vectors*np.roll(trace_vectors, -1, axis=0),
                        axis=1)/(lengths*np.roll(lengths, -1))).max())
                    trace_fill = float(area/cv2.contourArea(quad))
                    trace_border = bool((quad < 3).any() or (quad[:, 0] >= gray.shape[1]-3).any()
                                        or (quad[:, 1] >= gray.shape[0]-3).any())
                    metrics = {'threshold_index': threshold_index, 'morphological_close': close,
                               'area_px': float(area), 'min_side_px': float(lengths.min()),
                               'aspect': aspect, 'max_corner_cosine': trace_cosine, 'fill': trace_fill,
                               'failed_gates': [name for name, failed in (
                                   ('min_side', lengths.min() < cfg.min_side_px),
                                   ('aspect', not cfg.min_aspect < aspect < cfg.max_aspect),
                                   ('corner', trace_cosine > cfg.max_corner_cosine),
                                   ('fill', trace_fill < .86), ('source_border', trace_border)) if failed]}
                if (lengths.min() < cfg.min_side_px
                        or not cfg.min_aspect < lengths.max()/lengths.min() < cfg.max_aspect):
                    if _trace:
                        stage = 'min_side_rejected' if lengths.min() < cfg.min_side_px else 'aspect_rejected'
                        counts[stage] += 1
                        _trace.contour(stage, quad, **metrics)
                    continue
                vectors = np.roll(quad, -1, axis=0)-quad
                cosines = np.sum(vectors*np.roll(vectors, -1, axis=0), axis=1)/(lengths*np.roll(lengths, -1))
                quad_area = float(cv2.contourArea(quad))
                if np.abs(cosines).max() > cfg.max_corner_cosine or area/quad_area < .86:
                    if _trace:
                        counts['corner_or_fill_rejected'] += 1
                        _trace.contour('corner_or_fill_rejected', quad, **metrics)
                    continue
                if ((quad < 3).any() or (quad[:, 0] >= gray.shape[1]-3).any()
                        or (quad[:, 1] >= gray.shape[0]-3).any()):
                    if _trace:
                        counts['source_border_rejected'] += 1
                        _trace.contour('source_border_rejected', quad, **metrics)
                    continue
                if _trace:
                    counts['passed_geometry'] += 1
                    _trace.contour('passed_geometry', quad, **metrics)
                candidates.append((quad, quad_area, threshold_index))
    # Deduplicate the same observed envelope, without suppressing smaller panels
    # merely because a larger enclosing contour happens to contain them.
    groups = []
    for quad, area, threshold in sorted(candidates, key=lambda row: row[1], reverse=True):
        matched = None
        for group in groups:
            intersection = cv2.intersectConvexConvex(quad, group['quad'])[0]
            if intersection/max(1, area+group['area']-intersection) > .65:
                matched = group
                break
        if matched is None:
            groups.append({'quad': quad, 'area': area, 'votes': {threshold}})
        else:
            matched['votes'].add(threshold)
    observed = [g for g in groups if len(g['votes']) >= cfg.min_threshold_votes]
    merged, retained = 0, []
    for candidate in observed:
        smaller = [other for other in observed if other['area'] < candidate['area']*.65
                   and cv2.intersectConvexConvex(candidate['quad'], other['quad'])[0]/other['area'] > .8]
        if len(smaller) >= 2:
            merged += 1
        else:
            retained.append(candidate)
    # Iteratively remove isolated candidates, so reported neighbours are present
    # in the returned set. This never fills a predicted row or missing module.
    while retained:
        supported = [g for g in retained if sum(
            other is not g and _neighbours(g['quad'], other['quad']) for other in retained
        ) >= cfg.min_neighbours]
        if len(supported) == len(retained):
            break
        retained = supported
    retained.sort(key=lambda g: tuple(g['quad'].mean(axis=0)[::-1]))
    if _trace:
        _trace.contour_summary(dict(counts, distinct_groups=len(groups),
            groups_passing_votes=len(observed), compound_rejected=merged,
            groups_retained=len(retained)))
    proposals = []
    for index, group in enumerate(retained):
        proposals.append({'proposal_id': f'N{index+1:03}',
            'polygon_native_xy': group['quad'].astype(float).tolist(),
            'threshold_votes': len(group['votes']),
            'neighbour_ids': [f'N{j+1:03}' for j, other in enumerate(retained)
                              if j != index and _neighbours(group['quad'], other['quad'])]})
    return {'schema': 'pvdelta.native-boundaries/v1', 'status': 'unreviewed_native_geometry',
        'config': asdict(cfg), 'structural_render': structure,
        'raw_contours_passing_geometry': len(candidates), 'distinct_observed_envelopes': len(observed),
        'merged_envelopes_rejected': merged, 'proposals': proposals,
        'semantic_panel_membership': 'unverified', 'measurement_authorised': False}
