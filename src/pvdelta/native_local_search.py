"""Research-only smaller-window native geometry, without a forced panel lattice."""
from __future__ import annotations

import hashlib
from dataclasses import asdict

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.native_boundaries import NativeBoundaryConfig, _neighbours
from pvdelta.native_lines import NativeLineConfig, native_line_proposals
from pvdelta.native_windows import _translate_proposal


def local_native_alternatives(temperature_c, *, config=None, line_config=None):
    """Search nine overlapping half-size slices with independent local directions.

    This returns alternative geometry only, never a replacement scan or approved
    measurement mask. Callers must verify source/native identities before using
    ``complete_native_proposals`` to protect existing measurements. No reference
    labels, RGB, CAD, inferred grid cells or temperature ranking enter selection.
    Shape/side/divider/neighbour gates are the unchanged native line-search gates.
    Finer windows can miss large panels or lose edge context; efficacy is unproven.
    """
    values = np.asarray(temperature_c)
    if (values.ndim != 2 or min(values.shape, default=0) < 1
            or values.dtype.kind not in 'fiu' or not np.isfinite(values).all()):
        raise ValueError('A finite native numeric 2D grid is required')
    cfg = NativeBoundaryConfig() if config is None else config
    lc = NativeLineConfig() if line_config is None else line_config
    if not isinstance(cfg, NativeBoundaryConfig) or not isinstance(lc, NativeLineConfig):
        raise TypeError('Native boundary and line configuration objects required')
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    height, width = values.shape
    result = {'schema': 'pvdelta.native-local-alternatives/v1', 'proposals': [],
        'config': asdict(cfg), 'line_config': asdict(lc),
        'native_shape': list(values.shape), 'native_temperature_sha256': digest,
        'local_windows': [], 'radiometric_resampling': False,
        'semantic_panel_membership': 'unverified', 'measurement_authorised': False,
        'efficacy_accepted': False, 'status': 'native_grid_too_small'}
    if min(height, width) < 64:
        return result
    tile_h, tile_w = (height+1)//2, (width+1)//2
    ys = sorted({0, (height-tile_h)//2, height-tile_h})
    xs = sorted({0, (width-tile_w)//2, width-tile_w})
    pool = []
    for y in ys:
        for x in xs:
            window = [x, y, x+tile_w, y+tile_h]
            grid = values[y:y+tile_h, x:x+tile_w]
            low, high = np.percentile(grid, [2, 98])
            if high-low <= 1e-6:
                result['local_windows'].append({'native_window_xyxy': window, 'proposals': 0,
                    'line_status': 'insufficient_structural_dynamic_range',
                    'pair_search': {'search_complete': False, 'reason': 'insufficient_structural_dynamic_range'},
                    'composition_diagnostics': {}})
                continue
            local = native_line_proposals(grid, config=cfg,
                                           line_config=lc, support_mode='hybrid')
            result['local_windows'].append({'native_window_xyxy': window,
                'proposals': len(local['proposals']), 'line_status': local['line_status'],
                'pair_search': local['pair_search'],
                'composition_diagnostics': local['composition_diagnostics']})
            for p in local['proposals']:
                translated = _translate_proposal(p, window)
                quad = np.asarray(translated['polygon_native_xy'], float)
                # Prefer the observation furthest from its search boundary.
                # This ranks geometry only, never a hot pixel or supplied label.
                clearance = min(float(quad[:, 0].min()-x), float(x+tile_w-quad[:, 0].max()),
                                float(quad[:, 1].min()-y), float(y+tile_h-quad[:, 1].max()))
                translated['local_window_clearance_px'] = clearance
                pool.append(translated)
    retained = []
    for candidate in sorted(pool, key=lambda p: (-p['local_window_clearance_px'],
            *p['native_window_xyxy'], *np.asarray(p['polygon_native_xy']).ravel())):
        if not any(convex_iou(candidate['polygon_native_xy'], p['polygon_native_xy']) > .65 for p in retained):
            retained.append(candidate)
    deduplicated = len(pool)-len(retained)
    # Retained observations, not duplicate windows, must supply neighbours.
    initial = len(retained)
    while retained:
        kept = [p for p in retained if sum(q is not p and _neighbours(
            np.asarray(p['polygon_native_xy']), np.asarray(q['polygon_native_xy']))
            for q in retained) >= cfg.min_neighbours]
        if len(kept) == len(retained):
            break
        retained = kept
    retained.sort(key=lambda p: (*np.asarray(p['polygon_native_xy']).mean(axis=0)[::-1],
                                 *np.asarray(p['polygon_native_xy']).ravel()))
    for index, proposal in enumerate(retained):
        proposal['proposal_id'] = f'F{index+1:04d}'
        proposal['neighbour_ids'] = [f'F{j+1:04d}' for j, other in enumerate(retained)
            if j != index and _neighbours(np.asarray(proposal['polygon_native_xy']),
                                           np.asarray(other['polygon_native_xy']))]
    if hashlib.sha256(values.tobytes()).hexdigest() != digest:
        raise RuntimeError('Native grid changed during local alternative search')
    status = ('unreviewed_local_geometry' if any(w['line_status'] != 'insufficient_structural_dynamic_range'
              for w in result['local_windows']) else 'no_searchable_local_windows')
    result.update(status=status, proposals=retained,
        raw_local_observations=len(pool), deduplicated_observations=deduplicated,
        rejected_after_dedup_neighbours=initial-len(retained))
    return result
