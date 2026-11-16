"""Source-bound diagnostic of native ridge evidence on an unchanged quad.

This is one evidence channel, not the hybrid detector gate, a panel mask or a
temperature measurement. A failed ridge is unknown, never a healthy panel.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.edge_support import native_edge_continuity


def edge_audit_identity(values, quad, *, source_sha256, native_temperature_sha256):
    """Validate native bytes and geometry before computing a cache identity."""
    for digest in (source_sha256, native_temperature_sha256):
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Source and native SHA256 identities required')
    values = np.asarray(values)
    points = np.asarray(quad, dtype=float)
    if (values.ndim != 2 or min(values.shape, default=0) < 8
            or values.dtype.kind not in 'fiu' or not np.isfinite(values).all()):
        raise ValueError('Finite native numeric grid required')
    if hashlib.sha256(values.tobytes()).hexdigest() != native_temperature_sha256:
        raise ValueError('Native temperature SHA256 differs')
    if (points.shape != (4, 2) or not np.isfinite(points).all() or (points < 0).any()
            or (points[:, 0] > values.shape[1]-1).any()
            or (points[:, 1] > values.shape[0]-1).any()):
        raise ValueError('Four finite native corners inside the source required')
    convex_iou(points, points)
    engine = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in ('native_edge_audit.py', 'edge_support.py', 'convex_geometry.py')}
    payload = {'source_sha256': source_sha256, 'native_temperature_sha256': native_temperature_sha256,
               'native_shape': list(values.shape), 'polygon_native_xy': points.tolist(),
               'engine_source_sha256': engine,
               'limits': {'min_contrast_k': .15, 'min_snr': 3., 'coverage': .8, 'max_gap_fraction': .18}}
    token = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return token, payload


def audit_native_edges(values, quad, *, source_sha256, native_temperature_sha256):
    """Assess each original edge; do not refit, replace or approve any pixels."""
    token, payload = edge_audit_identity(values, quad, source_sha256=source_sha256,
                                         native_temperature_sha256=native_temperature_sha256)
    points = np.asarray(quad, dtype=float)
    sides = []
    for label, start, end in zip('ABCD', points, np.roll(points, -1, axis=0)):
        edge = native_edge_continuity(values, start, end)
        passed = edge.coverage >= .8 and edge.largest_gap_fraction <= .18
        sides.append({'side': label, 'start_native_xy': start.tolist(), 'end_native_xy': end.tolist(),
                      'native_ridge_supported': passed, **asdict(edge)})
    if hashlib.sha256(np.asarray(values).tobytes()).hexdigest() != native_temperature_sha256:
        raise RuntimeError('Native data changed during edge audit')
    return {'schema': 'pvdelta.native-edge-audit/v1', 'audit_token': token, **payload,
            'sides': sides, 'semantic_panel_membership': 'unverified',
            'measurement_authorised': False, 'mask_generated': False, 'database_written': False,
            'scope': 'Native ridge channel only; historical hybrid line support is a separate channel. '
                     'No boundary changes, mask approval, fault verdict or health assessment.'}
