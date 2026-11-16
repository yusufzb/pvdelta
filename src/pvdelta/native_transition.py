"""Research native ridge/step geometry; never a semantic panel classifier."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

import numpy as np

from pvdelta.convex_geometry import convex_iou

_OFFSETS = np.arange(-12., 12.01, .5)
# Distant neighbouring seams must not dominate the local proposed boundary.
# Both the null and signal models use the same fixed source-centred weights.
_WEIGHTS = np.exp(-(_OFFSETS/8)**4)
_ROOT_WEIGHT = np.sqrt(_WEIGHTS)


def _design(family, centre, width):
    x = _OFFSETS
    signal = (np.exp(-.5*((x-centre)/width)**2) if family == 'ridge'
              else np.tanh((x-centre)/width))
    return np.column_stack((np.ones(len(x)), x/12, signal))


@lru_cache(maxsize=1)
def _models():
    params = [(family, float(centre), width)
        for family, widths in (('ridge', (.6, 1.2, 2.4)), ('step', (.6, 1.2, 2.4, 3.6)))
        for width in widths for centre in np.arange(-4., 4.01, .5)]
    design = np.stack([_design(*p) for p in params])
    return params, design, np.linalg.pinv(design*_ROOT_WEIGHT[None, :, None])*_ROOT_WEIGHT[None, None, :]


def _fit_strips(profiles, required):
    params, design, inverse = _models()
    null = design[0, :, :2]
    null_inverse = np.linalg.pinv(null*_ROOT_WEIGHT[:, None])*_ROOT_WEIGHT[None, :]
    baseline = (null @ (null_inverse@profiles.T)).T
    null_sse = np.sum((profiles-baseline)**2*_WEIGHTS, axis=1)
    coefficients = np.einsum('mpq,sq->smp', inverse, profiles)
    prediction = np.einsum('mqp,smp->smq', design, coefficients)
    sse = np.sum((prediction-profiles[:, None, :])**2*_WEIGHTS, axis=2)
    decisions = []
    for i in range(len(profiles)):
        best = int(np.argmin(sse[i]))
        family, centre, width = params[best]
        contrast = float(abs(coefficients[i, best, 2])*(2 if family == 'step' else 1))
        explained = 1-float(sse[i, best])/max(float(null_sse[i]), 1e-12)
        error = float(np.sqrt(sse[i, best]/_WEIGHTS.sum()))/max(contrast, 1e-12)
        near = np.flatnonzero(sse[i] <= sse[i, best]+max((.03*contrast)**2*_WEIGHTS.sum(), 1e-9))
        centres = [params[j][1] for j in near]
        families = {params[j][0] for j in near}
        uncertainty = max(centres)-min(centres)
        good = (contrast >= required and explained >= .75 and error <= .15
                and abs(centre) < 4 and uncertainty <= 1.5 and families == {family})
        decisions.append({'valid': bool(good), 'family': family, 'offset_px': centre,
            'width_px': width, 'polarity': 1 if coefficients[i, best, 2] >= 0 else -1,
            'contrast_k': contrast, 'explained_over_linear': explained,
            'relative_rmse': error, 'centre_uncertainty_px': uncertainty})
    return decisions


def transition_edge_location(temperature_c, start_xy, end_xy):
    """Fit ridge or monotonic step against a linear-background null model.

    Four separated source strips fit independently; three must agree on model
    family, polarity and centre. Full-edge samples then validate that fixed
    model. A linear ramp, spot, missing source flanks or ambiguous centre fails.
    Sampling/interpolation serves geometry only; input temperatures are unchanged.
    Limits are development settings, not thermography/IEC acceptance criteria.
    """
    import cv2

    values = np.asarray(temperature_c)
    endpoints = np.asarray([start_xy, end_xy], float)
    if (values.ndim != 2 or min(values.shape, default=0) < 32
            or values.dtype.kind not in 'fiu' or not np.isfinite(values).all()
            or endpoints.shape != (2, 2) or not np.isfinite(endpoints).all()):
        raise ValueError('Finite native numeric 2D grid and finite xy endpoints required')
    a, b = endpoints
    delta = b-a
    length = float(np.linalg.norm(delta))
    if length < 18 or length > 2*np.hypot(*values.shape):
        raise ValueError('Edge length outside bounded source search')
    normal = np.array([-delta[1], delta[0]])/length
    positions = np.linspace(0, 1, max(33, int(np.ceil(length))+1))
    coords = a+positions[:, None, None]*delta+_OFFSETS[None, :, None]*normal
    if ((coords < 0).any() or (coords[:, :, 0] > values.shape[1]-1).any()
            or (coords[:, :, 1] > values.shape[0]-1).any()):
        return {'status': 'insufficient_source_flanks'}, None
    samples = cv2.remap(values.astype(float, copy=False), coords[:, :, 0].astype(np.float32),
        coords[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)
    differences = np.diff(samples, axis=0)
    noise = float(np.median(np.abs(differences-np.median(differences, axis=0)))/.67448975/np.sqrt(2))
    required = max(.15, 3*noise)
    central = samples[(positions >= .12) & (positions <= .88)]
    profiles = np.stack([np.median(part, axis=0) for part in np.array_split(central, 4)])
    strips = _fit_strips(profiles, required)
    choices = []
    for family in ('ridge', 'step'):
        for polarity in (-1, 1):
            valid = [s for s in strips if s['valid'] and s['family'] == family and s['polarity'] == polarity]
            if len(valid) < 3:
                continue
            shift = float(np.median([s['offset_px'] for s in valid]))
            agreed = [s for s in valid if abs(s['offset_px']-shift) <= .75]
            if len(agreed) >= 3:
                choices.append((family, polarity, float(np.median([s['offset_px'] for s in agreed])),
                    float(np.median([s['width_px'] for s in agreed])), len(agreed)))
    audit = {'status': 'unresolved_transition_consensus', 'strips': strips,
             'structural_variation_k': noise, 'required_contrast_k': required}
    if len(choices) != 1:
        return audit, None
    family, polarity, shift, width, count = choices[0]
    design = _design(family, shift, width)
    inverse = np.linalg.pinv(design*_ROOT_WEIGHT[:, None])*_ROOT_WEIGHT[None, :]
    beta = (inverse@samples.T).T
    residual = np.sum((samples-beta@design.T)**2*_WEIGHTS, axis=1)
    null = design[:, :2]
    null_inverse = np.linalg.pinv(null*_ROOT_WEIGHT[:, None])*_ROOT_WEIGHT[None, :]
    null_beta = (null_inverse@samples.T).T
    null_sse = np.sum((samples-null_beta@null.T)**2*_WEIGHTS, axis=1)
    contrast = abs(beta[:, 2])*(2 if family == 'step' else 1)
    explained = 1-residual/np.maximum(null_sse, 1e-12)
    error = np.sqrt(residual/_WEIGHTS.sum())/np.maximum(contrast, 1e-12)
    supported = ((contrast >= required) & (explained >= .60) & (error <= .25)
                 & (np.sign(beta[:, 2]) == polarity))
    gap = longest = 0
    for ok in supported:
        gap = 0 if ok else gap+1
        longest = max(longest, gap)
    coverage, largest_gap = float(supported.mean()), longest/len(supported)
    audit.update(family=family, polarity=polarity, offset_px=shift, width_px=width,
        strip_consensus_count=count, coverage=coverage, largest_gap_fraction=largest_gap,
        supported_samples=supported.tolist(), sample_count=len(supported))
    if coverage < .8 or largest_gap > .18:
        audit['status'] = 'insufficient_full_edge_transition'
        return audit, None
    audit['status'] = 'supported_native_transition'
    return audit, (a+shift*normal, delta)


def localize_transition_quad(values, proposal, config):
    """Observe all four mixed edges of a provisional quad, without panel approval."""
    original = np.asarray(proposal['polygon_native_xy'], float)
    if original.shape != (4, 2) or not np.isfinite(original).all():
        raise ValueError('Finite quadrilateral required')
    convex_iou(original, original)
    observed = [transition_edge_location(values, a, b) for a, b in zip(original, np.roll(original, -1, axis=0))]
    audits = [o[0] for o in observed]
    if any(o[1] is None for o in observed):
        return None, {'status': 'unchanged_unresolved_transitions', 'edges': audits}
    corners = []
    for index in range(4):
        a, u = observed[(index-1) % 4][1]
        b, v = observed[index][1]
        try:
            distance = np.linalg.solve(np.stack([u, -v], axis=1), b-a)[0]
        except np.linalg.LinAlgError:
            return None, {'status': 'unchanged_degenerate_intersections', 'edges': audits}
        corners.append(a+distance*u)
    quad = np.asarray(corners)
    vectors = np.roll(quad, -1, axis=0)-quad
    lengths = np.linalg.norm(vectors, axis=1)
    area = abs(float(np.sum(quad[:, 0]*np.roll(quad[:, 1], -1)-quad[:, 1]*np.roll(quad[:, 0], -1))))/2
    ratio = max((lengths[0]+lengths[2])/(lengths[1]+lengths[3]),
                (lengths[1]+lengths[3])/(lengths[0]+lengths[2]))
    cosine = np.abs((vectors*np.roll(vectors, -1, axis=0)).sum(axis=1)/(lengths*np.roll(lengths, -1)))
    if (not np.isfinite(quad).all() or (quad < 3).any()
            or (quad[:, 0] >= values.shape[1]-3).any() or (quad[:, 1] >= values.shape[0]-3).any()
            or np.linalg.norm(quad-original, axis=1).max() > 6 or lengths.min() < config.min_side_px
            or not config.min_area_px <= area <= values.size*config.max_area_fraction
            or not config.min_aspect < ratio < config.max_aspect or cosine.max() > config.max_corner_cosine):
        return None, {'status': 'unchanged_geometry_limit', 'edges': audits}
    try:
        convex_iou(quad, quad)
    except ValueError:
        return None, {'status': 'unchanged_nonconvex', 'edges': audits}
    fresh = [transition_edge_location(values, a, b)[0] for a, b in zip(quad, np.roll(quad, -1, axis=0))]
    if any(e['status'] != 'supported_native_transition' or abs(e['offset_px']) > .75
           or e['family'] != old['family'] for e, old in zip(fresh, audits)):
        return None, {'status': 'unchanged_refitted_transition_support', 'edges': audits, 'fresh_edges': fresh}
    output = deepcopy(proposal)
    families = [e['family'] for e in fresh]
    output.update(polygon_native_xy=quad.tolist(), source_method='four_native_transitions',
        side_model_families=families, side_coverage=[e['coverage'] for e in fresh],
        side_max_gap_fraction=[e['largest_gap_fraction'] for e in fresh], native_supported_sides=[0, 1, 2, 3],
        native_ridge_sides=[i for i, f in enumerate(families) if f == 'ridge'],
        native_step_sides=[i for i, f in enumerate(families) if f == 'step'],
        native_transition_evidence={'schema': 'pvdelta.native-transition-evidence/v1',
            'parent_polygon_native_xy': original.tolist(), 'edges': fresh,
            'purpose': 'geometry_only', 'measurement_authorised': False}, measurement_authorised=False)
    return output, {'status': 'localized_four_native_transitions', 'edges': audits,
                    'fresh_edges': fresh, 'polygon_native_xy': quad.tolist()}
