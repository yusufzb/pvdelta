"""RGB-only surface checks for projected native radiometric candidate pixels.

These checks flag disagreement for review. They never delete a thermal finding or
interpret display colours as temperatures, and cannot validate calibration alone.
"""

from __future__ import annotations

import numpy as np


def check_rgb_surface(
    rgb: np.ndarray, projected_pixels: np.ndarray, panel_polygon: np.ndarray,
    *, uncertainty_px: float,
) -> dict:
    import cv2

    points = np.asarray(projected_pixels, dtype=float)
    panel = np.asarray(panel_polygon, dtype=np.float32)
    if (points.ndim != 2 or points.shape[1] != 2 or not len(points)
            or not np.isfinite(points).all() or panel.shape != (4, 2)
            or not np.isfinite(panel).all() or not cv2.isContourConvex(panel)
            or not np.isfinite(uncertainty_px) or uncertainty_px < 0):
        raise ValueError("surface check requires finite geometry and non-negative uncertainty")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("surface check requires uint8 RGB")
    indices = np.rint(points).astype(int)
    inside = ((indices[:, 0] >= 0) & (indices[:, 0] < rgb.shape[1])
              & (indices[:, 1] >= 0) & (indices[:, 1] < rgb.shape[0]))
    if not inside.all():
        return {"status": "review_geometry", "reason": "candidate projects outside RGB frame"}
    distance = np.array([cv2.pointPolygonTest(panel, tuple(map(float, p)), True) for p in points])
    values = rgb[indices[:, 1], indices[:, 0]].astype(float)
    r, g, b = values.T
    green = ((g - (r + b) / 2) > 20) & (g > r * 1.10) & (g > b * 1.20)
    vegetation_ratio = float(green.mean())
    interior_ratio = float((distance > uncertainty_px).mean())
    result = {"interior_ratio": interior_ratio, "vegetation_ratio": vegetation_ratio,
              "projected_pixel_count": len(points), "registration_margin_rgb_px": uncertainty_px}
    if (distance < -uncertainty_px).mean() > .05:
        result.update(status="review_geometry", reason="candidate disagrees with the RGB module boundary")
    elif vegetation_ratio > .5:
        result.update(status="review_vegetation", reason="projected candidate has dominant green RGB support")
    elif interior_ratio < .95:
        result.update(status="review_edge", reason="candidate overlaps the registration uncertainty band")
    else:
        result.update(status="geometry_consistent", reason="candidate lies inside the RGB panel; still provisional")
    return result
