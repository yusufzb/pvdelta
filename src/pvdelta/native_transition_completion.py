"""Opt-in recovery of rejected observed quads with four fresh native edges."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.native_boundaries import NativeBoundaryConfig
from pvdelta.native_completion import _observed_sides, complete_native_proposals
from pvdelta.native_localization import _localize
from pvdelta.native_search_trace import trace_native_search
from pvdelta.native_transition import localize_transition_quad
from pvdelta.native_windows import native_window_proposals


class _RejectedPool:
    def __init__(self, shape, limit):
        self.shape, self.limit = shape, limit
        self.quads, self.buckets = [], defaultdict(list)
        self.overflow = False
        self.events = 0

    def __call__(self, event):
        if event.get('event') != 'candidate' or event.get('stage') != 'side_support':
            return
        self.events += 1
        if self.overflow:
            return
        support = np.asarray(event['support'], float)
        if (support.shape != (4, 3) or not np.isfinite(support).all()
                or sum(bool(s[0]) and s[1] >= .8 and s[2] <= .18 for s in support) < 2):
            return
        q = np.asarray(event['polygon_native_xy'], float)
        if (q.shape != (4, 2) or not np.isfinite(q).all() or (q < 3).any()
                or (q[:, 0] >= self.shape[1]-3).any() or (q[:, 1] >= self.shape[0]-3).any()):
            return
        key = tuple(np.floor(q.mean(axis=0)/32).astype(int))
        for x in range(key[0]-1, key[0]+2):
            for y in range(key[1]-1, key[1]+2):
                for index in self.buckets[(x, y)]:
                    if convex_iou(q, self.quads[index]['polygon_native_xy']) > .9:
                        return
        if len(self.quads) >= self.limit:
            self.overflow = True
            return
        self.buckets[key].append(len(self.quads))
        self.quads.append({'proposal_id': f'R{len(self.quads)+1:04d}',
            'polygon_native_xy': q.tolist(), 'original_side_support': event['support'],
            'parent_pass_id': event['pass_id'], 'measurement_authorised': False})


def transition_completion_proposals(temperature_c, *, config=None, line_config=None,
                                    raw_supplement=True, margin_px=2, max_candidates=512):
    """Recover only directly observed, initially side-rejected candidates.

    At least two original supported sides must anchor a candidate. All four
    localized native edges then need the existing consensus/coverage checks;
    mixed step/ridge evidence requires at least two ridges. Existing protected
    completion additionally requires two baseline scale references, adjacency,
    conflict rejection and exact preservation of old measurement pixels.

    No rejected geometry is accepted merely because it fits a rack grid. No
    baseline boundaries, masks or temperatures are replaced. Overflow retains
    the complete baseline and withholds all additions, with an explicit status.
    This is research geometry, not semantic panel or fault authorization.
    """
    values = np.asarray(temperature_c)
    if (values.ndim != 2 or min(values.shape, default=0) < 8 or values.dtype.kind not in 'fiu'
            or not values.flags.c_contiguous or not np.isfinite(values).all()):
        raise ValueError('Finite contiguous native numeric 2D grid required')
    if type(max_candidates) is not int or not 1 <= max_candidates <= 512:
        raise ValueError('max_candidates must be an integer in 1..512')
    if type(margin_px) is not int or margin_px < 0 or type(raw_supplement) is not bool:
        raise ValueError('Explicit nonnegative pixel margin and boolean raw supplementation required')
    cfg = config or NativeBoundaryConfig()
    if not isinstance(cfg, NativeBoundaryConfig):
        raise TypeError('NativeBoundaryConfig required')
    before = hashlib.sha256(values.tobytes()).hexdigest()
    pool = _RejectedPool(values.shape, max_candidates)
    with trace_native_search(values, pool):
        baseline = native_window_proposals(values, config=cfg, line_config=line_config,
                                           raw_supplement=raw_supplement)
    decisions, alternatives = [], []
    native = values.astype(float, copy=True)
    for proposal in ([] if pool.overflow else pool.quads):
        if any(convex_iou(proposal['polygon_native_xy'], p['polygon_native_xy']) > .05
               for p in baseline['proposals']):
            decisions.append({'proposal_id': proposal['proposal_id'], 'status': 'overlaps_baseline_envelope'})
            continue
        localized, audit = _localize(native, deepcopy(proposal), cfg)
        if localized is None:
            localized, audit = localize_transition_quad(native, deepcopy(proposal), cfg)
        if localized is None or not _observed_sides(localized):
            decisions.append({'proposal_id': proposal['proposal_id'], 'status': 'fresh_four_edge_evidence_unresolved',
                              'audit_status': audit['status']})
            continue
        decisions.append({'proposal_id': proposal['proposal_id'], 'status': 'fresh_four_edges_passed'})
        alternatives.append(localized)
    # Multiple source passes may converge on the same observed boundary.
    selected = []
    for p in sorted(alternatives, key=lambda p: (-min(p['side_coverage']),
                     *np.mean(p['polygon_native_xy'], axis=0)[::-1], p['proposal_id'])):
        if not any(convex_iou(p['polygon_native_xy'], q['polygon_native_xy']) > .65 for q in selected):
            selected.append(p)
    result = complete_native_proposals(baseline, {'proposals': selected},
                                      native_shape=values.shape, margin_px=margin_px)
    if (hashlib.sha256(values.tobytes()).hexdigest() != before
            or not np.array_equal(native, values.astype(float))):
        raise RuntimeError('Transition completion changed native values')
    result['transition_completion'] = {
        'schema': 'pvdelta.observed-transition-completion/v1',
        'status': 'pool_budget_exceeded_baseline_only' if pool.overflow else 'unreviewed_observed_geometry',
        'native_temperature_sha256': before, 'source_rejected_events': pool.events,
        'distinct_anchored_candidates': len(pool.quads), 'candidate_limit': max_candidates,
        'pool_overflow': pool.overflow, 'decision_counts': dict(Counter(d['status'] for d in decisions)),
        'decisions': decisions, 'four_edge_alternatives': len(selected),
        'retained_additions': result['native_completion']['retained_additions'],
        'baseline_geometry_and_pixels_preserved': True, 'radiometric_resampling': False,
        'measurement_authorised': False, 'semantic_panel_membership': 'unverified'}
    return result
