"""Explicit higher-budget observed geometry search preserving baseline support."""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from pvdelta.native_boundaries import NativeBoundaryConfig
from pvdelta.native_completion import complete_native_proposals
from pvdelta.native_lines import NativeLineConfig
from pvdelta.native_windows import native_window_proposals


def protected_dense_completion_proposals(temperature_c, *, config=None, line_config=None,
        margin_px=2, raw_supplement=True, max_pair_product=2_000_000):
    """Compare baseline and explicit denser pair search on one unchanged array.

    No side, angle, aspect, border, neighbour or support gate is relaxed. All
    pair geometry is processed in bounded chunks; the original default budget
    remains 250000. Higher search cost is opt-in. Existing completion gates
    preserve baseline geometry, IDs and actual measurement pixels. References
    and labels cannot enter this search; no measurement/production approval.
    """
    values = np.asarray(temperature_c)
    if (values.ndim != 2 or min(values.shape, default=0) < 1
            or values.dtype.kind not in 'fiu' or not np.isfinite(values).all()):
        raise ValueError('Finite numeric native 2D temperature grid required')
    if isinstance(margin_px, bool) or not isinstance(margin_px, int) or margin_px < 0:
        raise ValueError('margin_px must be a nonnegative integer')
    if not isinstance(raw_supplement, bool):
        raise TypeError('raw_supplement must be boolean')
    cfg = NativeBoundaryConfig() if config is None else config
    lc = NativeLineConfig() if line_config is None else line_config
    if not isinstance(cfg, NativeBoundaryConfig) or not isinstance(lc, NativeLineConfig):
        raise TypeError('NativeBoundaryConfig and NativeLineConfig required')
    # Validate even when the caller's baseline already has the requested budget.
    candidate_lc = replace(lc, max_pair_product=max_pair_product)
    alternative_lc = replace(lc, max_pair_product=max(lc.max_pair_product, candidate_lc.max_pair_product))
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    baseline = native_window_proposals(values, config=cfg, line_config=lc, raw_supplement=raw_supplement)
    alternative = (baseline if alternative_lc == lc else native_window_proposals(values,
        config=cfg, line_config=alternative_lc, raw_supplement=raw_supplement))
    if hashlib.sha256(values.tobytes()).hexdigest() != digest:
        raise RuntimeError('Native temperature grid changed during dense geometry search')
    completed = complete_native_proposals(baseline, alternative, native_shape=values.shape, margin_px=margin_px)
    completed['dense_search'] = {'native_temperature_sha256': digest, 'native_shape': list(values.shape),
        'baseline_pair_budget': lc.max_pair_product, 'alternative_pair_budget': alternative_lc.max_pair_product,
        'geometry_chunk_pairs': lc.geometry_chunk_pairs, 'second_search_needed': alternative_lc != lc,
        'baseline_global_pair_search': baseline.get('pair_search'),
        'alternative_global_pair_search': alternative.get('pair_search'),
        'baseline_global_status': baseline.get('line_status'),
        'alternative_global_status': alternative.get('line_status'),
        'baseline_global_line_diagnostics': baseline.get('line_diagnostics'),
        'alternative_global_line_diagnostics': alternative.get('line_diagnostics'),
        'alternative_local_window_search': alternative.get('local_window_search'),
        'radiometric_resampling': False, 'measurement_authorised': False,
        'scope': 'Observed geometry only; semantic panel membership, efficacy and independent review unverified'}
    return completed
