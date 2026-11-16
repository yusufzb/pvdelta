"""Bounded neighbour hypotheses require four fresh native ridge observations."""
from __future__ import annotations

import hashlib
import re

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.native_affine_groups import _basis
from pvdelta.native_boundaries import NativeBoundaryConfig
from pvdelta.native_completion import _observed_sides, _validate
from pvdelta.native_line_groups import _valid_affine
from pvdelta.native_localization import _localize
from pvdelta.regions import polygon_to_mask
from pvdelta.support_partition import partition_measurement_support


def frontier_native_candidates(temperature_c, baseline, *, native_temperature_sha256,
                               max_hypotheses=512, margin_px=2, edge_model='ridge'):
    """Search directly adjacent to observed geometry, never fill an array.

    Local seed axes/scale generate provisional search hypotheses at fixed
    cell steps 1.0, 1.1 and 1.2. Their edges are NOT observations. A hypothesis
    enters the returned pool only after all four sides independently pass the
    existing narrow native-ridge strip-consensus and fresh full-edge checks.
    This can expose frames split by internal thermal structure without relying
    on a closed Canny contour or a four-cell group.

    The caller verifies source identity and separately applies completion and
    semantic review. No fitted alternative replaces old geometry or pixels.
    Two existing scale references, overlap/conflict checks and old-support
    preservation remain necessary downstream. No accepted alternative seeds
    another search. Regular non-PV frames can pass: this is geometry research,
    not panel inventory, a fault detector or measurement authorisation.

    ``edge_model='mixed'`` is a separate opt-in ridge/step research model.
    Four source transitions must pass; downstream completion still requires
    at least two ridge sides. The default ridge extractor is unchanged.
    """
    values = np.asarray(temperature_c)
    if edge_model not in ('ridge', 'mixed'):
        raise ValueError('edge_model must be ridge or mixed')
    if edge_model == 'mixed':
        from pvdelta.native_transition import localize_transition_quad
        localizer = localize_transition_quad
    else:
        localizer = _localize
    if (values.ndim != 2 or min(values.shape, default=0) < 32
            or values.dtype.kind not in 'fiu' or not values.flags.c_contiguous
            or not np.isfinite(values).all()):
        raise ValueError('Finite contiguous native numeric 2D grid, minimum 32 per axis, required')
    if (not isinstance(native_temperature_sha256, str)
            or re.fullmatch('[0-9a-f]{64}', native_temperature_sha256) is None):
        raise ValueError('Expected lowercase native SHA256 required')
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    if digest != native_temperature_sha256:
        raise ValueError('Native temperature SHA256 mismatch')
    if type(max_hypotheses) is not int or not 1 <= max_hypotheses <= 4096:
        raise ValueError('max_hypotheses must be integer 1..4096')
    if type(margin_px) is not int or margin_px < 0:
        raise ValueError('margin_px must be a nonnegative integer')
    _validate(baseline, values.shape)
    if len(baseline['proposals']) > 512:
        raise ValueError('At most 512 baseline observations required')
    if baseline.get('native_temperature_sha256', digest) != digest:
        raise ValueError('Baseline native SHA256 mismatch')
    old = baseline['proposals']
    masks = [polygon_to_mask(p['polygon_native_xy'], values.shape) for p in old]
    supports = partition_measurement_support(masks, margin_px=margin_px)
    protected = np.zeros(values.shape, bool)
    for support in supports:
        protected |= support.interior
    cfg = NativeBoundaryConfig(min_aspect=.5, max_aspect=6.1, max_corner_cosine=.95)
    native = values.astype(float, copy=False)  # geometry sampling only
    decisions, fits = [], []
    counters = {'seeds':0, 'skipped_source_border':0, 'skipped_existing_envelope':0,
                'duplicate_fits':0, 'protected_measurement_rejections':0}
    overflow = False
    seeds = sorted(((p, s) for p, s in zip(old, supports)), key=lambda pair: pair[0]['proposal_id'])
    for seed, support in seeds:
        quad = np.asarray(seed['polygon_native_xy'], float)
        if not _observed_sides(seed) or not support.interior.any() or not _valid_affine(quad, values.shape):
            continue
        counters['seeds'] += 1
        axes = _basis(quad).T
        for axis_index, axis in enumerate(axes):
            for sign in (-1, 1):
                for step in (1., 1.1, 1.2):
                    projected = quad+sign*step*axis
                    if ((projected < 3).any() or (projected[:,0] >= values.shape[1]-3).any()
                            or (projected[:,1] >= values.shape[0]-3).any()):
                        counters['skipped_source_border'] += 1
                        continue
                    # Coarse pruning only; final fitted geometry is checked again.
                    if any(convex_iou(projected, p['polygon_native_xy']) > .2 for p in old):
                        counters['skipped_existing_envelope'] += 1
                        continue
                    if len(decisions) >= max_hypotheses:
                        overflow = True
                        break
                    hypothesis = {'proposal_id':f'H{len(decisions)+1:04d}',
                        'polygon_native_xy':projected.tolist(), 'source_method':'projected_hypothesis',
                        'measurement_authorised':False}
                    fitted, audit = localizer(native, hypothesis, cfg)
                    record = {'hypothesis_id':hypothesis['proposal_id'], 'seed_id':seed['proposal_id'],
                        'axis_index':axis_index, 'direction':sign, 'step':step,
                        'projected_polygon_native_xy':projected.tolist(), 'ridge_audit':audit}
                    decisions.append(record)
                    if fitted is None:
                        continue
                    fitted_quad = np.asarray(fitted['polygon_native_xy'])
                    if (polygon_to_mask(fitted_quad, values.shape) & protected).any():
                        counters['protected_measurement_rejections'] += 1
                        record['rejection'] = 'touches_old_measurement_interior'
                        continue
                    if any(convex_iou(fitted_quad, p['polygon_native_xy']) > .05 for p in old):
                        record['rejection'] = 'overlaps_existing_envelope'
                        continue
                    origin = ('four_observed_native_ridges_from_neighbour_hypothesis' if edge_model == 'ridge'
                              else 'four_observed_native_transitions_from_neighbour_hypothesis')
                    fitted.update(proposal_origin=origin,
                        semantic_panel_membership='unverified', segment_supported_sides=[],
                        frontier_hypothesis={'seed_id':seed['proposal_id'], 'axis_index':axis_index,
                            'direction':sign, 'step':step, 'hypothesis_id':hypothesis['proposal_id']})
                    fits.append(fitted)
                if overflow:
                    break
            if overflow:
                break
        if overflow:
            break
    retained = []
    if not overflow:
        for candidate in sorted(fits, key=lambda p: (*np.asarray(p['polygon_native_xy']).mean(axis=0)[::-1],
                *np.asarray(p['polygon_native_xy']).ravel())):
            q = np.asarray(candidate['polygon_native_xy'])
            same = any(min(np.linalg.norm(q-np.roll(np.asarray(p['polygon_native_xy']), shift, axis=0),
                axis=1).max() for shift in range(4)) <= 1 for p in retained)
            if same:
                counters['duplicate_fits'] += 1
            else:
                retained.append(candidate)
    for index, p in enumerate(retained):
        p['proposal_id'] = f'F{index+1:04d}'
    if hashlib.sha256(values.tobytes()).hexdigest() != digest:
        raise RuntimeError('Native grid changed during frontier search')
    return {'schema':'pvdelta.native-frontier-candidates/v1',
        'status':'frontier_hypothesis_budget_exceeded' if overflow else 'unreviewed_native_ridge_geometry',
        'proposals':retained, 'native_shape':list(values.shape), 'native_temperature_sha256':digest,
        'decisions':decisions, 'counters':counters, 'hypotheses_evaluated':len(decisions),
        'fitted_hypotheses':len(fits), 'max_hypotheses':max_hypotheses,
        'radiometric_resampling':False, 'measurement_authorised':False,
        'semantic_panel_membership':'unverified', 'efficacy_accepted':False}
