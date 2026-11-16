"""Stable float64 convex clipping for near-coincident source-image boundaries."""
from __future__ import annotations

import numpy as np


def _cross(first, second):
    return first[0]*second[1]-first[1]*second[0]


def _signed_area(points):
    shifted = points-points[0]
    return .5*float(np.sum(shifted[:, 0]*np.roll(shifted[:, 1], -1)
                          - np.roll(shifted[:, 0], -1)*shifted[:, 1]))


def _polygon(points):
    polygon = np.asarray(points, dtype=np.float64)
    if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
            or not np.isfinite(polygon).all() or abs(_signed_area(polygon)) < 1e-9):
        raise ValueError('finite nonzero convex polygons required')
    edges = np.roll(polygon, -1, axis=0)-polygon
    turns = np.array([_cross(first, second) for first, second in zip(edges, np.roll(edges, -1, axis=0))])
    if np.any(turns > 1e-8) and np.any(turns < -1e-8):
        raise ValueError('convex polygon vertices must follow boundary order')
    return polygon


def convex_iou(first, second) -> float:
    """Symmetric-area IoU via convex clipping, including shared edges/corners.

    Avoids source float32 intersection results that can lose a triangle when two
    rectangles share nearly coincident vertices. Input polygons are never mutated.
    """
    first, second = _polygon(first), _polygon(second)
    output = first.copy()
    orientation = 1 if _signed_area(second) > 0 else -1
    for start, end in zip(second, np.roll(second, -1, axis=0)):
        if len(output) == 0:
            return 0.
        source, clipped = output, []
        previous = source[-1]
        previous_distance = orientation*_cross(end-start, previous-start)
        for current in source:
            distance = orientation*_cross(end-start, current-start)
            inside, was_inside = distance >= -1e-8, previous_distance >= -1e-8
            if inside != was_inside:
                denominator = previous_distance-distance
                fraction = np.clip(previous_distance/denominator, 0, 1)
                clipped.append(previous+fraction*(current-previous))
            if inside:
                clipped.append(current)
            previous, previous_distance = current, distance
        output = np.asarray(clipped, dtype=float).reshape(-1, 2)
    area_first, area_second = abs(_signed_area(first)), abs(_signed_area(second))
    intersection = min(area_first, area_second, abs(_signed_area(output))) if len(output) >= 3 else 0.
    return float(np.clip(intersection/(area_first+area_second-intersection), 0, 1))
