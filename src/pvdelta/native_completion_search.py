"""Run both completion geometry searches on one unchanged native source grid."""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from pvdelta.native_boundaries import NativeBoundaryConfig
from pvdelta.native_completion import complete_native_proposals
from pvdelta.native_windows import native_window_proposals


def supported_completion_proposals(temperature_c, *, config=None, line_config=None,
                                    margin_px=2, raw_supplement=True):
    """Opt-in research preview; never authorises masks or creates findings.

    Baseline search retains the caller's configuration. A separate observed
    search lowers only min_aspect to at most 1.3; strict completion gates decide
    which additions can be proposed. If the baseline already has that range,
    no redundant second search is run. Both searches see the same native grid,
    without resizing/interpolation. Source-grid SHA is recorded and checked.

    This cannot prove that preserved or added rectangles are physical panels.
    External pilots have not passed the hotspot efficacy gate. Keep results in
    explicit review; do not silently replace production masks or findings.
    """
    values = np.asarray(temperature_c)
    if (values.ndim != 2 or min(values.shape) < 1
            or values.dtype.kind not in 'fiu' or not np.isfinite(values).all()):
        raise ValueError('A finite numeric native 2D temperature grid is required')
    if isinstance(margin_px, bool) or not isinstance(margin_px, int) or margin_px < 0:
        raise ValueError('margin_px must be a nonnegative integer')
    if not isinstance(raw_supplement, bool):
        raise TypeError('raw_supplement must be boolean')
    cfg = NativeBoundaryConfig() if config is None else config
    if not isinstance(cfg, NativeBoundaryConfig):
        raise TypeError('config must be NativeBoundaryConfig')
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    baseline = native_window_proposals(values, config=cfg, line_config=line_config,
                                       raw_supplement=raw_supplement)
    alternative_cfg = replace(cfg, min_aspect=min(cfg.min_aspect, 1.3))
    alternative = (baseline if alternative_cfg == cfg else native_window_proposals(
        values, config=alternative_cfg, line_config=line_config, raw_supplement=raw_supplement))
    if hashlib.sha256(values.tobytes()).hexdigest() != digest:
        raise RuntimeError('Native temperature grid changed during geometry search')
    result = complete_native_proposals(baseline, alternative, native_shape=values.shape,
                                        margin_px=margin_px)
    result['native_completion_search'] = {'native_temperature_sha256': digest,
        'native_shape': list(values.shape), 'radiometric_resampling': False,
        'baseline_min_aspect': cfg.min_aspect, 'alternative_min_aspect': alternative_cfg.min_aspect,
        'raw_supplement': raw_supplement, 'second_search_needed': alternative_cfg != cfg,
        'measurement_authorised': False, 'hotspot_efficacy_accepted': False}
    return result
