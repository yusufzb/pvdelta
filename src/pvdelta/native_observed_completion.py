"""Opt-in surface-guided recovery of observed quads lost during deduplication."""
import hashlib
from collections import Counter
from copy import deepcopy
from dataclasses import replace

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.native_boundaries import NativeBoundaryConfig
from pvdelta.native_completion import complete_native_proposals
from pvdelta.native_lines import _line_pass, native_line_proposals
from pvdelta.native_search_trace import trace_native_search
from pvdelta.regions import polygon_to_mask


def complete_observed_surface_gaps(temperature_c, baseline, *, panel_mask, windows,
                                   max_candidates=512, raw_supplement=False, allow_isolated=False):
    """Retain four-side observations before a larger envelope can hide them.

    Explicit manual panel support is required; unknown/conflicting/background
    pixels must be False. Windows are native integer xyxy slices, normally manual
    surface bounding boxes with context. No reference panel coordinates or fault
    labels enter the search. Callers must verify that baseline, mask and native
    temperatures refer to the same source before calling this geometry helper.

    All original side/shape gates and the absolute area ceiling remain active.
    Completion requires two baseline scale references, adjacency and unchanged
    old measurement pixels. Pool overflow withholds all additions. Results stay
    provisional; this helper is not enabled in the default flight scanner.
    With raw_supplement=True, search raw structure even when enhanced contrast
    already found some panels. Both passes use the same four-side gates; the
    combined pool shares one budget, mask filter and protected completion step.
    With allow_isolated=True, explicit manual panel support replaces adjacency
    only; two observed baseline scale references are still required.
    """
    values = np.asarray(temperature_c)
    mask = np.asarray(panel_mask)
    if (values.ndim != 2 or min(values.shape, default=0) < 64
            or values.dtype.kind not in 'fiu' or not values.flags.c_contiguous
            or not np.isfinite(values).all()):
        raise ValueError('Finite contiguous native numeric grid of at least 64x64 required')
    if mask.dtype != np.bool_ or mask.shape != values.shape:
        raise ValueError('Explicit boolean panel mask on the same native grid required')
    if type(max_candidates) is not int or not 1 <= max_candidates <= 512:
        raise ValueError('max_candidates must be an integer in 1..512')
    if type(raw_supplement) is not bool:
        raise ValueError('raw_supplement must be boolean')
    if type(allow_isolated) is not bool:
        raise ValueError('allow_isolated must be boolean')
    if not isinstance(windows, (list, tuple)) or len(windows) > 32:
        raise ValueError('At most 32 explicit native windows required')
    for window in windows:
        if (not isinstance(window, (list, tuple)) or len(window) != 4
                or any(type(v) is not int for v in window)):
            raise ValueError('Native windows must contain four integers')
        x0, y0, x1, y1 = window
        if not (0 <= x0 < x1 <= values.shape[1] and 0 <= y0 < y1 <= values.shape[0]
                and min(x1-x0, y1-y0) >= 64):
            raise ValueError('Windows must be inside the native frame and at least 64x64')
    # Validate baseline before doing any search, including the no-candidate path.
    complete_native_proposals(baseline, {'proposals': []}, native_shape=values.shape)
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    observed, passes, rejected = [], {}, Counter()
    overflow = False

    def sink(event):
        nonlocal overflow
        if event['event'] == 'pass_begin':
            passes[event['pass_id']] = event
        if event['event'] != 'candidate' or event['stage'] != 'four_sides_passed':
            return
        quad = event['polygon_native_xy']
        if not np.all(mask[polygon_to_mask(quad, values.shape)]):
            rejected['outside_manual_panel'] += 1
            return
        if any(convex_iou(quad, p['polygon_native_xy']) > .05 for p in baseline['proposals']):
            rejected['overlaps_existing'] += 1
            return
        if len(observed) >= max_candidates:
            overflow = True
            return
        mode = passes[event['pass_id']]['support_mode']
        observed.append({
            'proposal_id': f'O{len(observed)+1:04}', 'polygon_native_xy': deepcopy(quad),
            'source_method': ('observed_lines_and_native_ridges' if mode == 'hybrid'
                              else 'four_observed_lines'),
            'side_coverage': [s[1] for s in event['support']],
            'side_max_gap_fraction': [s[2] for s in event['support']],
            'observed_trace': deepcopy(event),
            'observed_pass': deepcopy(passes[event['pass_id']]),
            'native_temperature_sha256': digest, 'measurement_authorised': False,
        })

    with trace_native_search(values, sink):
        for x0, y0, x1, y1 in windows:
            grid = values[y0:y1, x0:x1]
            config = replace(NativeBoundaryConfig(),
                             max_area_fraction=min(.999, .08*values.size/grid.size))
            native_line_proposals(grid, config=config, support_mode='hybrid')
            if raw_supplement:
                _line_pass(grid, config=config, support_mode='hybrid', local_contrast=False)
    selected = []
    for proposal in sorted(observed, key=lambda p: (-min(p['side_coverage']), p['proposal_id'])):
        if any(convex_iou(proposal['polygon_native_xy'], p['polygon_native_xy']) > .9
               for p in selected):
            rejected['duplicate_observed_quad'] += 1
            continue
        selected.append(proposal)
    result = complete_native_proposals(baseline, {'proposals': [] if overflow else selected},
        native_shape=values.shape, **({'manual_panel_mask': mask} if allow_isolated else {}))
    if hashlib.sha256(values.tobytes()).hexdigest() != digest:
        raise RuntimeError('Observed completion changed native temperatures')
    result['observed_surface_completion'] = {
        'schema': 'pvdelta.observed-surface-completion/v1',
        'status': ('pool_budget_exceeded_baseline_only' if overflow else
                   'unreviewed_geometry' if windows else 'no_eligible_windows_baseline_only'),
        'manual_panel_support_required': True, 'native_temperature_sha256': digest,
        'panel_mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest(),
        'windows': deepcopy(windows), 'maximum_candidates': max_candidates,
        'raw_supplement': raw_supplement,
        'observed_candidates': len(observed), 'selected_candidates': len(selected),
        'filter_counts': dict(rejected), 'measurement_authorised': False,
    }
    if allow_isolated:
        result['observed_surface_completion']['allow_isolated'] = True
    return result
