"""Image-local panel geometry for inventory-free Stage 1 tracking."""

from __future__ import annotations

import numpy as np


def quadrilateral_center(points_xy: np.ndarray) -> tuple[float, float]:
    """Intersect opposite diagonals of a complete, cyclically ordered panel quad.

    This is the projective image of a rectangular panel's physical centre. The
    vertex mean or visible-mask centroid need not be that centre under perspective.
    Callers must exclude clipped/partial panels before supplying the four corners.
    """
    points = np.asarray(points_xy, dtype=float)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("panel corners must be four finite cyclically ordered xy points")
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not (np.all(cross > 1e-9) or np.all(cross < -1e-9)):
        raise ValueError("panel corners must form a nondegenerate convex quadrilateral")
    a = points[2] - points[0]
    b = points[3] - points[1]
    parameters = np.linalg.solve(np.column_stack((a, -b)), points[1] - points[0])
    center = points[0] + parameters[0] * a
    return float(center[0]), float(center[1])
