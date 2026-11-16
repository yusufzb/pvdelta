"""Research pool of actual native quads before the four-cell grouping gate."""
from __future__ import annotations

import hashlib

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.native_affine_groups import _observed_quads
from pvdelta.native_boundaries import NativeBoundaryConfig
from pvdelta.native_line_groups import _LineEvidence
from pvdelta.native_lines import NativeLineConfig, _line_pass
from pvdelta.native_search_trace import trace_native_search


def observed_native_candidates(temperature_c, *, max_pair_product=250_000):
    """Collect independently observed quads, without requiring a 2x2 cycle.

    The same five native windows, multiple contour votes, four segment sides
    and affine-shape limits as the line-group research extractor apply. No
    inferred side, ridge supplementation, temperature ranking or reference
    annotation enters this pool. Deduplication matches that extractor; output
    order is spatial and deterministic. Overflow withholds the entire pool.

    A quad is NOT a panel or an approved measurement mask. The caller must
    verify source/native identity and separately apply completion, conflict,
    old-support preservation and semantic review. Existing production and
    four-cell extraction are unchanged. No input temperature is resampled.
    """
    values = np.asarray(temperature_c)
    if (values.ndim != 2 or min(values.shape, default=0) < 32
            or values.dtype.kind not in 'fiu' or not values.flags.c_contiguous
            or not np.isfinite(values).all()):
        raise ValueError('Finite contiguous native numeric 2D grid, minimum 32 per axis, required')
    line_config = NativeLineConfig(min_segment_px=16, max_pair_product=max_pair_product)
    low, high = np.percentile(values, [2, 98])
    if high-low <= 1e-6:
        raise ValueError('Native radiometry has insufficient structural dynamic range')
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    closed, raw_closed = _observed_quads(values)
    sink = _LineEvidence(values.shape)
    height, width = values.shape
    windows = [(0, 0, width, height)]
    if min(values.shape) >= 48:
        tile_h, tile_w = (2*height+2)//3, (2*width+2)//3
        windows.extend((x, y, x+tile_w, y+tile_h)
            for y in (0, height-tile_h) for x in (0, width-tile_w))
    skipped = []
    config = NativeBoundaryConfig(min_aspect=.5, max_aspect=6.1, max_corner_cosine=.95)
    with trace_native_search(values, sink):
        for x, y, right, bottom in windows:
            grid = values[y:bottom, x:right]
            low, high = np.percentile(grid, [2, 98])
            if high-low <= 1e-6:
                skipped.append({'native_window_xyxy': [x, y, right, bottom],
                                'reason': 'insufficient_structural_dynamic_range'})
                continue
            _line_pass(grid, config=config, line_config=line_config,
                       local_contrast=True, support_mode='segments')
    if len(sink.passes) != len(sink.ends):
        raise RuntimeError('Incomplete native line evidence trace')
    pool = [{'quad': g['quad'], 'evidence': {
        'source_method': 'closed_contour', 'threshold_votes': len(g['votes'])}}
        for g in closed] + sink.observations
    retained = []
    overflow = sink.overflow
    if not overflow:
        for candidate in sorted(pool, key=lambda g: (
                g['evidence']['source_method'] != 'closed_contour',
                -min(g['evidence'].get('side_coverage', [1.])),
                *g['quad'].mean(axis=0)[::-1], *g['quad'].ravel())):
            if not any(convex_iou(candidate['quad'], other['quad']) > .65 for other in retained):
                retained.append(candidate)
                if len(retained) > 512:
                    overflow = True
                    break
    proposals = []
    if not overflow:
        retained.sort(key=lambda g: (*g['quad'].mean(axis=0)[::-1], *g['quad'].ravel()))
        proposals = [dict(g['evidence'], proposal_id=f'O{n+1:04d}',
            polygon_native_xy=g['quad'].tolist(), proposal_origin='observed_native_pool',
            semantic_panel_membership='unverified', measurement_authorised=False)
            for n, g in enumerate(retained)]
    if hashlib.sha256(values.tobytes()).hexdigest() != digest:
        raise RuntimeError('Native grid changed during observed-candidate search')
    return {'schema': 'pvdelta.native-observed-candidates/v1',
        'status': 'observed_candidate_budget_exceeded' if overflow else 'unreviewed_observed_geometry',
        'proposals': proposals, 'native_shape': list(values.shape),
        'native_temperature_sha256': digest,
        'raw_closed_observations': raw_closed, 'closed_two_vote_quads': len(closed),
        'raw_four_side_events': sink.seen, 'line_quads_rejected_affine': sink.affine_rejected,
        'search_passes': sink.passes, 'search_ends': sink.ends, 'skipped_windows': skipped,
        'limits': {'max_observed_quads': 512, 'max_line_events': 8192,
                   'max_pair_product_per_window': max_pair_product,
                   'min_side_coverage': .8, 'max_gap_fraction': .18},
        'radiometric_resampling': False, 'measurement_authorised': False,
        'semantic_panel_membership': 'unverified', 'efficacy_accepted': False}
