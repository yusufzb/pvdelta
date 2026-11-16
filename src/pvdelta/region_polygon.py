"""Simple, possibly concave review polygons on the native pixel grid."""
from __future__ import annotations

import math

from pvdelta.regions import polygon_to_mask


def simple_polygon_mask(polygon, shape):
    if (not isinstance(polygon, list) or not 3 <= len(polygon) <= 1024
            or any(not isinstance(p, list) or len(p) != 2
                   or any(type(v) not in (int, float) or not math.isfinite(v) for v in p)
                   for p in polygon)):
        raise ValueError('A polygon needs 3–1024 finite native vertices')
    if any(x < 0 or y < 0 or x > shape[1] or y > shape[0] for x, y in polygon):
        raise ValueError('Polygon vertices must stay inside the native grid')
    if len({tuple(p) for p in polygon}) != len(polygon):
        raise ValueError('Polygon vertices must be distinct; do not repeat the first point')

    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])

    def touches(a, b, p):
        return (abs(cross(a, b, p)) <= 1e-9
                and min(a[0], b[0])-1e-9 <= p[0] <= max(a[0], b[0])+1e-9
                and min(a[1], b[1])-1e-9 <= p[1] <= max(a[1], b[1])+1e-9)

    n = len(polygon)
    for i, a in enumerate(polygon):
        b, c = polygon[(i+1) % n], polygon[(i+2) % n]
        # Collinear forward vertices are valid; backtracking overlaps an edge.
        if (abs(cross(a, b, c)) <= 1e-9
                and (b[0]-a[0])*(c[0]-b[0])+(b[1]-a[1])*(c[1]-b[1]) <= 0):
            raise ValueError('Polygon edges must not backtrack')
        for j in range(i+1, n):
            if j == i+1 or (i == 0 and j == n-1):
                continue
            c, d = polygon[j], polygon[(j+1) % n]
            if (cross(a, b, c)*cross(a, b, d) < 0
                    and cross(c, d, a)*cross(c, d, b) < 0
                    or any((touches(a, b, c), touches(a, b, d), touches(c, d, a), touches(c, d, b)))):
                raise ValueError('Polygon edges must not cross or touch themselves')
    return polygon_to_mask(polygon, shape)
