"""Conservative native-ridge localization of observed, provisional alternatives."""
from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.edge_support import native_edge_continuity
from pvdelta.native_boundaries import NativeBoundaryConfig
from pvdelta.native_completion import _observed_sides, _validate, complete_native_proposals
from pvdelta.regions import polygon_to_mask
from pvdelta.support_partition import partition_measurement_support


def _edge_location(values, a, b):
    import cv2

    delta = b-a
    length = float(np.linalg.norm(delta))
    normal = np.array([-delta[1], delta[0]])/length
    positions = np.linspace(.12, .88, max(32, int(np.ceil(length))))
    offsets = np.arange(-7., 7.01, .25)
    points = a+positions[:, None]*delta
    coords = points[:, None, :]+offsets[None, :, None]*normal
    if ((coords < 0).any() or (coords[:, :, 0] > values.shape[1]-1).any()
            or (coords[:, :, 1] > values.shape[0]-1).any()):
        return {'status': 'insufficient_source_flanks'}, None
    samples = cv2.remap(values, coords[:, :, 0].astype(np.float32),
                        coords[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)
    centres = np.arange(12, 45)  # proposed displacements -4..+4, flanks ±2/3
    observations = []
    for region in np.array_split(samples, 4):
        profile = np.median(region-np.median(region, axis=1)[:, None], axis=0)
        cold = np.maximum(np.minimum(profile[centres-8]-profile[centres], profile[centres+8]-profile[centres]),
                          np.minimum(profile[centres-12]-profile[centres], profile[centres+12]-profile[centres]))
        hot = np.maximum(np.minimum(profile[centres]-profile[centres-8], profile[centres]-profile[centres+8]),
                         np.minimum(profile[centres]-profile[centres-12], profile[centres]-profile[centres+12]))
        polarity = 'cold' if cold.max() >= hot.max() else 'hot'
        scores = cold if polarity == 'cold' else hot
        peak = float(scores.max())
        near = np.flatnonzero(scores >= peak-max(.05, .05*peak))
        # A broad, multiple or boundary peak is unresolved, never a forced snap.
        good = (peak >= .15 and len(near) > 0 and near[0] > 0 and near[-1] < len(centres)-1
                and offsets[centres[near[-1]]]-offsets[centres[near[0]]] <= 1.5)
        observations.append({'valid': bool(good), 'polarity': polarity, 'contrast_k': peak,
            'offset_px': float(np.median(offsets[centres[near]])) if good else None})
    selected = None
    # At least three separated source strips must agree on polarity and location.
    for polarity in ('cold', 'hot'):
        valid = [o['offset_px'] for o in observations if o['valid'] and o['polarity'] == polarity]
        if len(valid) < 3:
            continue
        shift = float(np.median(valid))
        agreeing = [v for v in valid if abs(v-shift) <= .75]
        if len(agreeing) >= 3:
            selected = (float(np.median(agreeing)), polarity)
            break
    audit = {'status': 'unresolved_strip_consensus', 'strips': observations}
    if selected is None:
        return audit, None
    shift, polarity = selected
    evidence = native_edge_continuity(values, a+shift*normal, b+shift*normal)
    audit.update(offset_px=shift, polarity=polarity, coverage=evidence.coverage,
                 largest_gap_fraction=evidence.largest_gap_fraction)
    if evidence.coverage < .8 or evidence.largest_gap_fraction > .18 or evidence.polarity != polarity:
        audit['status'] = 'insufficient_full_edge_ridge'
        return audit, None
    audit['status'] = 'supported_native_ridge'
    return audit, (a+shift*normal, delta)


def _localize(values, proposal, cfg):
    original = np.asarray(proposal['polygon_native_xy'], float)
    edges, audit = [], []
    for a, b in zip(original, np.roll(original, -1, axis=0)):
        decision, edge = _edge_location(values, a, b)
        audit.append(decision)
        edges.append(edge)
    if any(edge is None for edge in edges):
        return None, {'status': 'unchanged_unresolved_edges', 'edges': audit}
    corners = []
    for i in range(4):
        a, u = edges[(i-1) % 4]
        b, v = edges[i]
        try:
            distance = np.linalg.solve(np.stack([u, -v], axis=1), b-a)[0]
        except np.linalg.LinAlgError:
            return None, {'status': 'unchanged_degenerate_intersections', 'edges': audit}
        corners.append(a+distance*u)
    quad = np.asarray(corners)
    lengths = np.linalg.norm(np.roll(quad, -1, axis=0)-quad, axis=1)
    vectors = np.roll(quad, -1, axis=0)-quad
    area = abs(float(np.sum(quad[:, 0]*np.roll(quad[:, 1], -1)-quad[:, 1]*np.roll(quad[:, 0], -1))))/2
    ratio = max((lengths[0]+lengths[2])/(lengths[1]+lengths[3]),
                (lengths[1]+lengths[3])/(lengths[0]+lengths[2]))
    cosine = np.abs((vectors*np.roll(vectors, -1, axis=0)).sum(axis=1)/(lengths*np.roll(lengths, -1)))
    if (not np.isfinite(quad).all() or (quad < 3).any()
            or (quad[:, 0] >= values.shape[1]-3).any() or (quad[:, 1] >= values.shape[0]-3).any()
            or np.linalg.norm(quad-original, axis=1).max() > 6 or lengths.min() < cfg.min_side_px
            or not cfg.min_area_px <= area <= values.size*cfg.max_area_fraction
            or not cfg.min_aspect < ratio < cfg.max_aspect or cosine.max() > cfg.max_corner_cosine):
        return None, {'status': 'unchanged_geometry_limit', 'edges': audit}
    try:
        convex_iou(quad, quad)
    except ValueError:
        return None, {'status': 'unchanged_nonconvex', 'edges': audit}
    fresh = [native_edge_continuity(values, a, b) for a, b in zip(quad, np.roll(quad, -1, axis=0))]
    if any(e.coverage < .8 or e.largest_gap_fraction > .18 for e in fresh):
        return None, {'status': 'unchanged_refitted_edge_support', 'edges': audit}
    refined = deepcopy(proposal)
    refined.update(polygon_native_xy=quad.tolist(), source_method='observed_lines_and_native_ridges',
        side_coverage=[e.coverage for e in fresh], side_max_gap_fraction=[e.largest_gap_fraction for e in fresh],
        native_supported_sides=[0, 1, 2, 3], measurement_authorised=False,
        native_localization={'parent_polygon_native_xy': original.tolist(), 'edges': audit,
            'scope': 'Native ridge geometry only; semantic panel membership and efficacy unverified',
            'max_search_offset_px': 4, 'measurement_authorised': False})
    return refined, {'status': 'localized_four_supported_ridges', 'edges': audit,
                     'polygon_native_xy': quad.tolist()}


def localize_native_alternatives(temperature_c, baseline, alternative, *, config=None, margin_px=2):
    """Localize observed alternatives, then apply unchanged protected completion.

    Both geometry inputs must belong to this exact native grid (caller verifies
    provenance, as with complete_native_proposals). No labels, CAD, RGB or grid
    extrapolation. All four edges need unambiguous three-of-four strip consensus
    within ±4 native pixels plus fresh full-edge .8 coverage/.18 gap. Otherwise
    retain the original alternative. Never replace baseline proposals or masks.
    A source flank outside the image is unresolved, not a fabricated border.
    """
    values = np.asarray(temperature_c)
    if (values.ndim != 2 or min(values.shape, default=0) < 8 or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all()):
        raise ValueError('Finite native numeric 2D grid, minimum 8 pixels per axis required')
    cfg = NativeBoundaryConfig() if config is None else config
    if not isinstance(cfg, NativeBoundaryConfig):
        raise TypeError('NativeBoundaryConfig required')
    _validate(baseline, values.shape)
    _validate(alternative, values.shape)
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    native = values.astype(float, copy=False)  # geometry sampling only
    original_masks = [polygon_to_mask(p['polygon_native_xy'], values.shape) for p in baseline['proposals']]
    supports = partition_measurement_support(original_masks, margin_px=margin_px)
    protected = np.zeros(values.shape, bool)
    for support in supports:
        protected |= support.interior
    proposals, decisions = [], []
    for proposal in alternative['proposals']:
        overlaps = [convex_iou(proposal['polygon_native_xy'], p['polygon_native_xy']) for p in baseline['proposals']]
        if any(iou > .65 for iou in overlaps) or not _observed_sides(proposal):
            proposals.append(deepcopy(proposal))
            decisions.append({'proposal_id': proposal['proposal_id'], 'status': 'unchanged_duplicate_or_unobserved'})
            continue
        refined, decision = _localize(native, proposal, cfg)
        if refined is not None:
            mask = polygon_to_mask(refined['polygon_native_xy'], values.shape)
            if ((mask & protected).any() or any(convex_iou(refined['polygon_native_xy'], p['polygon_native_xy']) > .05
                                               for p in baseline['proposals'])):
                decision['status'] = 'unchanged_baseline_protection'
                refined = None
        proposals.append(deepcopy(proposal) if refined is None else refined)
        decisions.append(dict(decision, proposal_id=proposal['proposal_id']))
    updated = deepcopy(alternative)
    updated['proposals'] = proposals
    completed = complete_native_proposals(baseline, updated, native_shape=values.shape, margin_px=margin_px)
    if hashlib.sha256(values.tobytes()).hexdigest() != digest:
        raise RuntimeError('Native temperature grid changed during localization')
    completed['native_localization_search'] = {'native_temperature_sha256': digest, 'decisions': decisions,
        'radiometric_resampling': False, 'baseline_geometry_and_measurement_unchanged': True,
        'measurement_authorised': False, 'efficacy_accepted': False}
    return completed
