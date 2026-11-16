"""Opt-in native boundary localization; changed geometry needs fresh measurements."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.native_boundaries import NativeBoundaryConfig
from pvdelta.native_completion import _observed_sides, _validate
from pvdelta.native_localization import _localize
from pvdelta.native_transition import localize_transition_quad


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def refine_native_geometry(temperature_c, geometry, *, config=None):
    """Localize four observed edges without adding modules or reusing old masks.

    Existing ridge/transition gates and 6px corner-motion bound apply. Each
    changed quad must have four observed sides. Conflicting changes are rejected
    symmetrically, then checked again against restored baselines. Reference
    labels, rack grids and hotspot temperatures are not localization inputs.
    The caller must recompute masks/statistics in a separate analysis run.
    """
    values = np.asarray(temperature_c)
    if (values.ndim != 2 or min(values.shape, default=0) < 8 or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all()):
        raise ValueError('Finite native numeric 2D grid required')
    cfg = config or NativeBoundaryConfig()
    if not isinstance(cfg, NativeBoundaryConfig):
        raise TypeError('NativeBoundaryConfig required')
    _validate(geometry, values.shape)
    native_digest = hashlib.sha256(values.tobytes(order='C')).hexdigest()
    original_digest = _digest(geometry)
    working = values.astype(np.float64, copy=True)
    work_digest = hashlib.sha256(working.tobytes()).hexdigest()
    old = geometry['proposals']
    proposed, decisions, changed = [], [], set()
    for index, item in enumerate(old):
        localized, audit = _localize(working, deepcopy(item), cfg)
        if localized is None:
            localized, audit = localize_transition_quad(working, deepcopy(item), cfg)
        accepted = localized is not None
        if accepted:
            _validate({'proposals': [localized]}, values.shape)
            q, before = np.asarray(localized['polygon_native_xy']), np.asarray(item['polygon_native_xy'])
            if (localized['proposal_id'] != item['proposal_id']
                    or np.linalg.norm(q-before, axis=1).max() > 6 or not _observed_sides(localized)):
                accepted, audit = False, {'status': 'localization_identity_motion_or_evidence_rejected'}
        if accepted and not np.array_equal(localized['polygon_native_xy'], item['polygon_native_xy']):
            changed.add(index)
            proposed.append(localized)
        else:
            proposed.append(deepcopy(item))
        decisions.append({'proposal_id': item['proposal_id'], 'localized': index in changed, 'audit': audit})
    # A rejected change can restore an old boundary that conflicts with another
    # change, so repeat until no newly restored baseline invalidates a proposal.
    while changed:
        reject = set()
        for i in range(len(proposed)):
            for j in range(i+1, len(proposed)):
                if not ({i, j} & changed):
                    continue
                if convex_iou(proposed[i]['polygon_native_xy'], proposed[j]['polygon_native_xy']) > .05:
                    reject.update({i, j} & changed)
        if not reject:
            break
        for index in reject:
            proposed[index] = deepcopy(old[index])
            decisions[index].update(localized=False, conflict_reason='overlap_with_selected_or_restored_geometry')
        changed -= reject
    output = deepcopy(geometry)
    output['proposals'] = proposed
    if (_digest(geometry) != original_digest or hashlib.sha256(values.tobytes(order='C')).hexdigest() != native_digest
            or hashlib.sha256(working.tobytes()).hexdigest() != work_digest):
        raise RuntimeError('Localization changed source geometry or native temperature values')
    _validate(output, values.shape)
    output['native_refinement'] = {'schema': 'pvdelta.native-boundary-refinement/v1',
        'baseline_geometry_sha256': original_digest, 'native_temperature_sha256': native_digest,
        'localized_proposals': len(changed), 'decisions': decisions,
        'requires_new_measurement_masks': True, 'old_measurement_reuse_authorised': False,
        'radiometric_resampling': False, 'semantic_panel_membership': 'unverified',
        'measurement_authorised': False, 'scope': 'Observed boundary localization only; no identity or fault approval'}
    output['measurement_authorised'] = False
    return output
