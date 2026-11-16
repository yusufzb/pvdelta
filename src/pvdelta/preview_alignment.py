"""Conservative JPEG-preview to native-thermal geometry checks, never thermometry."""
from __future__ import annotations

import cv2
import numpy as np

from pvdelta.hotspot_benchmark import evaluate_hotspot_regions


def _edges(array):
    array = cv2.GaussianBlur(array.astype(np.float32), (3, 3), .7)
    dx = cv2.Sobel(array, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(array, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = dx*dx + dy*dy
    if magnitude.ndim == 3:
        magnitude = magnitude.sum(axis=2)
    magnitude = np.sqrt(magnitude)
    ceiling = float(np.percentile(magnitude, 97))
    return (np.minimum(magnitude, ceiling) / max(ceiling, 1e-12)).astype(np.float32)


def _phase_uniqueness(first, second):
    product = np.fft.fft2(second) * np.conj(np.fft.fft2(first))
    correlation = np.abs(np.fft.ifft2(product / np.maximum(np.abs(product), 1e-12)))
    y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
    peak = float(correlation[y, x])
    yy, xx = np.indices(correlation.shape)
    dy = np.minimum(abs(yy-y), first.shape[0]-abs(yy-y))
    dx = np.minimum(abs(xx-x), first.shape[1]-abs(xx-x))
    runner = float(correlation[(dy > 5) | (dx > 5)].max())
    return peak / max(runner, 1e-12)


def _correlation(first, second, mask):
    if int(mask.sum()) < 100:
        return None
    a, b = first[mask].astype(float), second[mask].astype(float)
    if min(float(a.std()), float(b.std())) < .025:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def check_preview_alignment(native_values, preview_rgb):
    """Fit translation on central 60%, then validate on unused outer strips.

    Inputs must describe the SAME capture and have equal grids. Palette colours
    are used only for edge geometry; no temperatures are recovered from JPEG.
    Rotation/scale/distortion are intentionally unsupported. A failed check
    withholds scoring/overlays instead of assuming identity from equal shapes.
    Neither labels nor anomaly predictions are accepted as fitting inputs.
    """
    values, rgb = np.asarray(native_values), np.asarray(preview_rgb)
    if (values.ndim != 2 or min(values.shape, default=0) < 64
            or values.dtype.kind not in 'uif' or not np.isfinite(values).all()
            or rgb.shape != (*values.shape, 3) or rgb.dtype != np.uint8):
        raise ValueError('Finite native 2D grid >=64 pixels and matching uint8 RGB preview required')
    result = {'schema': 'pvdelta.preview-alignment/v1', 'status': 'unresolved',
              'method': 'central-edge translation fit / unused outer-strip validation',
              'native_shape': list(values.shape), 'preview_to_native': None,
              'measurement_authorised': False, 'geometry_only': True}

    def fail(reason):
        return dict(result, reason=reason)

    native, preview = _edges(values), _edges(rgb)
    height, width = values.shape
    by, bx = height//5, width//5
    a, b = native[by:-by, bx:-bx].copy(), preview[by:-by, bx:-bx].copy()
    if min(float(a.std()), float(b.std())) < .025:
        return fail('insufficient central edge texture')
    shift, response = cv2.phaseCorrelate(a, b)
    unique = _phase_uniqueness(a, b)
    result.update(phase_shift_xy=list(shift), phase_response=float(response), phase_peak_ratio=unique)
    if (not np.isfinite([*shift, response, unique]).all()
            or response < .25 or unique < 1.5):
        return fail('ambiguous central translation')
    limit = min(32., min(values.shape)*.08)
    if max(abs(v) for v in shift) > limit:
        return fail('translation exceeds supported range')
    forward = np.eye(2, 3, dtype=np.float32)
    forward[:, 2] = shift
    reverse = np.eye(2, 3, dtype=np.float32)
    reverse[:, 2] = -np.asarray(shift)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    try:
        score, forward = cv2.findTransformECC(a, b, forward, cv2.MOTION_TRANSLATION, criteria, None, 5)
        reverse_score, reverse = cv2.findTransformECC(b, a, reverse, cv2.MOTION_TRANSLATION, criteria, None, 5)
    except cv2.error:
        return fail('translation optimiser did not converge')
    displacement = forward[:, 2].astype(float)
    cycle = float(np.linalg.norm(displacement + reverse[:, 2]))
    result.update(training_ecc=float(score), reverse_training_ecc=float(reverse_score),
                  native_to_preview_shift_xy=displacement.tolist(), cycle_error_px=cycle)
    if (not np.isfinite([score, reverse_score, *displacement, cycle]).all()
            or min(score, reverse_score) < .65 or cycle > .75
            or max(abs(displacement)) > limit):
        return fail('fit strength, range or independent inverse check failed')
    # Only the preview edge representation is resampled, never native values.
    aligned = cv2.warpAffine(preview, forward, (width, height),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    valid = cv2.warpAffine(np.ones(values.shape, np.uint8), forward, (width, height),
                           flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP).astype(bool)
    # Keep validation >=8 px from the central fit region (Sobel/blur support).
    yy, xx = np.indices(values.shape)
    strips = [yy < by-8, yy >= height-by+8, xx < bx-8, xx >= width-bx+8]
    correlations = [_correlation(native, aligned, s & valid) for s in strips]
    usable = [v for v in correlations if v is not None]
    result['unused_strip_correlations_top_bottom_left_right'] = correlations
    if len(usable) < 3 or sum(v >= .60 for v in usable) < 3 or min(usable) < .35:
        return fail('unused outer strips disagree; rotation, distortion or wrong capture possible')
    result.update(status='accepted', reason='central translation and unused spatial checks passed',
                  preview_to_native=[[1., 0., -float(displacement[0])],
                                     [0., 1., -float(displacement[1])]])
    return result


def preview_regions_to_native(regions, alignment):
    """Transform all frozen labels/overlays together; never clip or drop a box.

    Rejection of ANY transformed region fails the frame. This utility is not an
    independent calibration certificate and does not infer physical faults.
    """
    if (alignment.get('schema') != 'pvdelta.preview-alignment/v1'
            or alignment.get('status') != 'accepted'):
        raise ValueError('Accepted preview/native geometry check required')
    shape = tuple(alignment['native_shape'])
    regions = list(regions)
    evaluate_hotspot_regions(regions, [], native_shape=shape)
    matrix = np.asarray(alignment['preview_to_native'], dtype=float)
    if (matrix.shape != (2, 3) or not np.isfinite(matrix).all()
            or not np.array_equal(matrix[:, :2], np.eye(2))):
        raise ValueError('Finite translation matrix required')
    transformed = []
    for region in regions:
        polygon = np.asarray(region['polygon_native_xy'], dtype=float)
        points = polygon + matrix[:, 2]
        transformed.append(dict(region, polygon_native_xy=points.tolist()))
    evaluate_hotspot_regions(transformed, [], native_shape=shape)
    return transformed
