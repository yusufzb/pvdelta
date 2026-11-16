"""Source-native ridge continuity evidence, never measurement-mask approval."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EdgeContinuity:
    sample_count: int
    valid_samples: int
    supported_samples: int
    coverage: float
    largest_gap_fraction: float
    noise_k: float
    required_contrast_k: float
    polarity: str
    supported: tuple[bool, ...]
    contrast_k: tuple[float, ...]
    purpose: str = 'geometry_only_not_temperature_measurement'
    measurement_authorised: bool = False


def native_edge_continuity(temperature_c, start_xy, end_xy, *,
                           min_contrast_k=.15, min_snr=3.) -> EdgeContinuity:
    """Measure a narrow opposing-edge ridge along an already proposed segment.

    Search is limited to ±1 native pixel about the given line, with flanks at
    ±1, ±2 or ±3 pixels. One-sided temperature steps do not count as frame
    ridges. Interpolation is only for geometry evidence; input values are never
    changed or returned as panel temperatures. Defaults are development limits.
    """
    import cv2

    values = np.asarray(temperature_c, float)
    endpoints = np.asarray([start_xy, end_xy], float)
    if (values.ndim != 2 or min(values.shape) < 8 or not np.isfinite(values).all()
            or endpoints.shape != (2, 2) or not np.isfinite(endpoints).all()):
        raise ValueError('finite native 2D grid and two finite xy endpoints required')
    limits = (min_contrast_k, min_snr)
    if any(isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in limits):
        raise ValueError('finite positive contrast and SNR limits required')
    delta = endpoints[1]-endpoints[0]
    length = float(np.linalg.norm(delta))
    if length < 2 or length > 2*np.hypot(*values.shape):
        raise ValueError('segment length outside bounded native search')
    normal = np.array([-delta[1], delta[0]])/length
    count = int(np.ceil(length))+1
    centres = endpoints[0]+np.linspace(0, 1, count)[:, None]*delta
    offsets = np.arange(-4, 5)
    points = centres[:, None, :]+offsets[None, :, None]*normal
    valid = ((points[:, :, 0] >= 0) & (points[:, :, 0] <= values.shape[1]-1)
             & (points[:, :, 1] >= 0) & (points[:, :, 1] <= values.shape[0]-1)).all(axis=1)
    profile = cv2.remap(values, points[:, :, 0].astype(np.float32),
                        points[:, :, 1].astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Adjacent along-line differences estimate variation, not a camera NETD.
    differences = np.diff(profile, axis=0)[valid[:-1] & valid[1:]]
    if differences.size:
        medians = np.median(differences, axis=0)
        noise = float(np.median(np.abs(differences-medians))/.67448975/np.sqrt(2))
    else:
        noise = 0.
    threshold = max(float(min_contrast_k), float(min_snr)*noise)
    cold, hot = np.zeros(count), np.zeros(count)
    for centre in (3, 4, 5):
        for radius in (1, 2, 3):
            left, middle, right = profile[:, centre-radius], profile[:, centre], profile[:, centre+radius]
            cold = np.maximum(cold, np.minimum(left-middle, right-middle))
            hot = np.maximum(hot, np.minimum(middle-left, middle-right))
    polarity = 'cold' if int(((cold >= threshold) & valid).sum()) >= int(((hot >= threshold) & valid).sum()) else 'hot'
    contrast = cold if polarity == 'cold' else hot
    supported = valid & (contrast >= threshold)
    gap = longest = 0
    for accepted in supported:
        gap = 0 if accepted else gap+1
        longest = max(longest, gap)
    return EdgeContinuity(count, int(valid.sum()), int(supported.sum()),
        float(supported.mean()), longest/count, noise, threshold, polarity,
        tuple(bool(v) for v in supported), tuple(float(v) for v in np.where(valid, contrast, 0)))


def combine_edge_evidence(continuity: EdgeContinuity, observed_intervals_fraction) -> dict:
    """Union observed line intervals and ridge samples on the same fixed edge.

    Fractions run from the SAME start endpoint to end endpoint. Coverage uses
    uniformly spaced source samples; gaps between evidence are never filled.
    This evidence does not replace geometric, semantic or neighbour checks.
    """
    intervals = np.asarray(observed_intervals_fraction, float)
    if intervals.size == 0:
        intervals = np.empty((0, 2))
    if (intervals.ndim != 2 or intervals.shape[1] != 2 or not np.isfinite(intervals).all()
            or (intervals < 0).any() or (intervals > 1).any()
            or (intervals[:, 1] <= intervals[:, 0]).any()):
        raise ValueError('ordered observed intervals in [0, 1] required')
    fractions = np.linspace(0, 1, continuity.sample_count)
    lines = np.zeros(continuity.sample_count, bool)
    for low, high in intervals:
        lines |= (fractions >= low) & (fractions <= high)
    ridge = np.asarray(continuity.supported, bool)
    union = lines | ridge
    gap = longest = 0
    for supported in union:
        gap = 0 if supported else gap+1
        longest = max(longest, gap)
    return {'coverage': float(union.mean()), 'largest_gap_fraction': longest/len(union),
            'line_only_samples': int((lines & ~ridge).sum()),
            'ridge_only_samples': int((ridge & ~lines).sum()),
            'shared_samples': int((ridge & lines).sum()),
            'supported': union.tolist(), 'measurement_authorised': False}
