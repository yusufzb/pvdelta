"""Protected research search for shorter, actually observed native edge fragments."""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from pvdelta.native_boundaries import NativeBoundaryConfig
from pvdelta.native_completion import complete_native_proposals
from pvdelta.native_dense_search import protected_dense_completion_proposals
from pvdelta.native_lines import NativeLineConfig
from pvdelta.native_windows import native_window_proposals


def protected_fragment_completion_proposals(temperature_c, *, config=None, line_config=None,
        margin_px=2, raw_supplement=True, max_pair_product=2_000_000):
    """Preserve dense baseline, then search observed fragments down to 16 pixels.

    This changes LSD input fragment length only. Four-side coverage/gap, shape,
    divider, neighbour and completion gates remain unchanged. No inferred edge,
    layout, reference label or temperature interpolation enters the search.
    Candidates still require two disjoint baseline scale references, one adjacent,
    and cannot touch existing measurement pixels. No production authorisation.
    """
    values = np.asarray(temperature_c)
    cfg = NativeBoundaryConfig() if config is None else config
    lc = NativeLineConfig() if line_config is None else line_config
    baseline = protected_dense_completion_proposals(values, config=cfg, line_config=lc,
        margin_px=margin_px, raw_supplement=raw_supplement, max_pair_product=max_pair_product)
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    alternative_lc = replace(lc, min_segment_px=min(lc.min_segment_px, 16),
                             max_pair_product=max(lc.max_pair_product, max_pair_product))
    if alternative_lc.min_segment_px == lc.min_segment_px:
        alternative = baseline
    else:
        alternative = native_window_proposals(values, config=cfg, line_config=alternative_lc,
                                               raw_supplement=raw_supplement)
    if hashlib.sha256(values.tobytes()).hexdigest() != digest:
        raise RuntimeError('Native grid changed during fragment search')
    completed = complete_native_proposals(baseline, alternative, native_shape=values.shape,
                                          margin_px=margin_px)
    completed['baseline_dense_completion'] = baseline['native_completion']
    completed['fragment_search'] = {'baseline_min_segment_px': lc.min_segment_px,
        'alternative_min_segment_px': alternative_lc.min_segment_px,
        'native_temperature_sha256': digest, 'native_shape': list(values.shape),
        'second_search_needed': alternative_lc.min_segment_px != lc.min_segment_px,
        'alternative_global_pair_search': alternative.get('pair_search'),
        'alternative_global_status': alternative.get('line_status'),
        'alternative_local_window_search': alternative.get('local_window_search'),
        'radiometric_resampling': False, 'measurement_authorised': False,
        'efficacy_accepted': False, 'scope': 'Unreviewed geometry research; independent reference review required'}
    return completed
