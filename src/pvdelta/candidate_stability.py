"""Conditional reference and mask sensitivity, never a replacement detection run."""
from __future__ import annotations

import numpy as np

from pvdelta.config import DetectionConfig
from pvdelta.reference import erode


def candidate_stability(values, measurement, reference, regions, *, config: DetectionConfig):
    """Delete each image-axis quadrant from the already-selected reference once.

    Regions and reference membership stay frozen. These are conditional stress
    tests on the same frame, not an independent validation set or confidence
    intervals. Extra margins test support retention only; no temperature is warped.
    """
    values, measurement, reference = map(np.asarray, (values, measurement, reference))
    if (values.ndim != 2 or measurement.shape != values.shape or reference.shape != values.shape
            or measurement.dtype != np.bool_ or reference.dtype != np.bool_
            or not measurement.any() or not reference.any() or (reference & ~measurement).any()
            or not np.isfinite(values[measurement]).all()):
        raise ValueError('Finite native temperatures and nested nonempty boolean supports required')
    if not isinstance(regions, dict) or not regions:
        raise ValueError('Nonempty named exact-pixel regions required')
    points_by_id = {}
    for name, points in regions.items():
        p = np.asarray(points)
        if (not isinstance(name, str) or not name or p.ndim != 2 or p.shape[1] != 2 or not len(p)
                or not np.issubdtype(p.dtype, np.integer) or (p < 0).any()
                or (p[:, 0] >= values.shape[0]).any() or (p[:, 1] >= values.shape[1]).any()
                or len(np.unique(p, axis=0)) != len(p)):
            raise ValueError('Distinct native integer region pixels required')
        if not measurement[p[:, 0], p[:, 1]].all() or reference[p[:, 0], p[:, 1]].any():
            raise ValueError('Candidate must be inside measurement support and outside reference')
        points_by_id[name] = p
    positions = np.argwhere(measurement)
    centre = (positions.min(axis=0) + positions.max(axis=0)) / 2.
    scale = np.maximum(positions.max(axis=0) - positions.min(axis=0), 1.)

    def design(points):
        return np.column_stack(((points - centre) / scale, np.ones(len(points))))

    ref_points = np.argwhere(reference)
    # Boundaries depend only on the frozen measurement mask, not on hot pixels.
    blocks = (ref_points[:, 0] > centre[0]).astype(int)*2 + (ref_points[:, 1] > centre[1])
    ref_values = values[reference]
    folds = []
    deltas = {name: [] for name in regions}
    for index, label in enumerate(('upper_left', 'upper_right', 'lower_left', 'lower_right')):
        heldout = blocks == index
        training = ~heldout
        fold = {'excluded_block': label, 'training_pixels': int(training.sum()),
                'heldout_pixels': int(heldout.sum()), 'status': 'unavailable'}
        if min(training.sum(), heldout.sum()) < config.min_reference_px:
            fold['reason'] = 'Insufficient training or held-out reference pixels'
            folds.append(fold)
            continue
        if config.detrend_gradient:
            x = design(ref_points[training])
            if np.linalg.matrix_rank(x) != 3:
                fold['reason'] = 'Training reference cannot fit a two-dimensional plane'
                folds.append(fold)
                continue
            coefficients, *_ = np.linalg.lstsq(x, ref_values[training], rcond=None)

            def predict(p, coefficients=coefficients):
                return design(p) @ coefficients
        else:
            median = float(np.median(ref_values[training]))

            def predict(p, median=median):
                return np.full(len(p), median)
        error = ref_values[heldout] - predict(ref_points[heldout])
        fold.update(status='available', heldout_p95_abs_k=float(np.percentile(np.abs(error), 95)))
        for name, p in points_by_id.items():
            delta = float(np.median(values[p[:, 0], p[:, 1]] - predict(p)))
            deltas[name].append({'excluded_block': label, 'delta_k': delta})
        folds.append(fold)
    margins = {str(px): erode(measurement, px) for px in (1, 2)}
    results = {}
    for name, p in points_by_id.items():
        ds = [item['delta_k'] for item in deltas[name]]
        low, high = (min(ds), max(ds)) if ds else (None, None)
        if len(ds) != 4:
            state = 'insufficient_reference_blocks'
        elif low >= config.delta_threshold_k:
            state = 'above_threshold_in_all_folds'
        elif high < config.delta_threshold_k:
            state = 'below_threshold_in_all_folds'
        else:
            state = 'crosses_threshold'
        results[name] = {'reference_test': state, 'valid_folds': len(ds),
            'delta_min_k': low, 'delta_max_k': high, 'fold_deltas': deltas[name],
            'extra_margin_retained_fraction': {px: float(mask[p[:, 0], p[:, 1]].mean())
                for px, mask in margins.items()}, 'fault_confirmed': False}
    return {'schema': 'pvdelta.candidate-stability/v1', 'folds': folds, 'regions': results,
        'threshold_k': config.delta_threshold_k,
        'scope': 'Conditional sensitivity on frozen same-frame reference and regions. Not independent validation.',
        'margin_scope': 'One/two extra native pixels; support retention only, not a detector rerun or calibrated error bound.'}
