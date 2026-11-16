"""Conservative RGB module proposals and native-edge checks for flight screening.

These automatic proposals require review. Geometry uses image structure; temperatures
must come from the original SDK grid. No fixed rack grid is filled without evidence.
OpenCV is an optional dependency, imported only when these helpers are used.
"""
from __future__ import annotations

import numpy as np


def order_quad(points: np.ndarray) -> np.ndarray:
    quad = np.asarray(points, np.float32)
    centre = quad.mean(axis=0)
    quad = quad[np.argsort(np.arctan2(quad[:, 1]-centre[1], quad[:, 0]-centre[0]))]
    return np.roll(quad, -np.argmin(quad.sum(axis=1)), axis=0)


def rgb_module_proposals(bgr: np.ndarray) -> list[np.ndarray]:
    """Find full, rectangular module candidates; return original RGB coordinates.

    Adapted from the reviewed RGB contour experiments. Large enclosing rectangles
    suppress half-cut seams; multiple thresholds must support each proposal.
    This does not infer invisible cells or a physical panel count.
    """
    import cv2

    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("expected uint8 BGR image")
    scale = min(1.0, 2016 / bgr.shape[1])
    image = cv2.resize(bgr, (round(bgr.shape[1]*scale), round(bgr.shape[0]*scale)))
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), .65), cv2.COLOR_BGR2HSV)
    proposals = []
    for threshold in (90, 120, 145, 165, 180, 195, 210, 225):
        seed = ((hsv[:, :, 1] < 140) & (hsv[:, :, 2] < threshold)).astype(np.uint8)
        for kernel in (1, 3, 5, 7):
            mask = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if not 1000 <= area <= image.shape[0]*image.shape[1]*.04:
                    continue
                hull = cv2.convexHull(contour)
                quad = cv2.approxPolyDP(hull, .012*cv2.arcLength(hull, True), True)
                _, (w, h), _ = cv2.minAreaRect(hull)
                if min(w, h) < 15 or not 1.55 < max(w, h)/min(w, h) < 2.85:
                    continue
                if area/(w*h) < .88 or len(quad) != 4:
                    continue
                quad = order_quad(quad.reshape(4, 2))
                if (quad < 3).any() or (quad[:, 0] >= image.shape[1]-3).any() or (
                    quad[:, 1] >= image.shape[0]-3
                ).any():
                    continue
                # Require panel-like RGB support, excluding roofs/ground rectangles.
                x, y, bw, bh = cv2.boundingRect(quad)
                local = np.zeros((bh, bw), np.uint8)
                cv2.fillConvexPoly(local, (quad-[x, y]).astype(np.int32), 1)
                values = hsv[y:y+bh, x:x+bw][local.astype(bool)]
                blue = (values[:, 0] > 85) & (values[:, 0] < 140)
                if blue.mean() < .6:
                    continue
                proposals.append((quad, abs(cv2.contourArea(quad)), threshold))

    def iou(a, b):
        intersection, _ = cv2.intersectConvexConvex(a[0], b[0])
        return intersection / max(1, a[1]+b[1]-intersection)

    kept = []
    for candidate in sorted(proposals, key=lambda item: item[1], reverse=True):
        if any(iou(candidate, other) > .15 for other in kept):
            continue
        support = {other[2] for other in proposals if iou(candidate, other) > .8}
        if len(support) < 2:
            continue
        kept.append(candidate)
    # Recover only one neighbouring cell from an observed seed. Every proposal
    # still needs panel-colour support and subsequent native edge validation.
    observed = list(kept)
    panel = ((hsv[:, :, 0] > 85) & (hsv[:, :, 0] < 140)
             & (hsv[:, :, 1] < 150) & (hsv[:, :, 2] > 25)).astype(np.uint8)
    for quad, area, threshold in observed:
        for delta in (quad[1]-quad[0], quad[3]-quad[0], quad[0]-quad[1], quad[0]-quad[3]):
            proposed = quad+delta
            if (proposed < 3).any() or (proposed[:, 0] >= image.shape[1]-3).any() or (
                proposed[:, 1] >= image.shape[0]-3
            ).any():
                continue
            candidate = (proposed, area, threshold)
            if any(iou(candidate, other) > .12 for other in kept):
                continue
            x, y, w, h = cv2.boundingRect(proposed)
            mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(mask, np.rint(proposed-[x, y]).astype(np.int32), 1)
            if panel[y:y+h, x:x+w][mask.astype(bool)].mean() < .82:
                continue
            kept.append(candidate)
    return [item[0]/scale for item in sorted(kept, key=lambda item: tuple(item[0].mean(axis=0)[::-1]))]


def edge_samples(polygon: np.ndarray) -> np.ndarray:
    return np.concatenate([
        start + np.linspace(.15, .85, 24)[:, None]*(end-start)
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True)
    ])


def native_edge_check(gray: np.ndarray, polygon: np.ndarray) -> dict:
    """Check all four sides against native gradients without reading temperatures."""
    import cv2

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    records = []
    offsets = np.arange(-8, 9, dtype=np.float32)
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        tangent = end-start
        tangent = tangent / max(1e-6, np.linalg.norm(tangent))
        normal = np.array([-tangent[1], tangent[0]])
        base = start + np.linspace(.18, .82, 30)[:, None]*(end-start)
        points = base[:, None, :] + offsets[None, :, None]*normal
        x, y = points[:, :, 0].astype(np.float32), points[:, :, 1].astype(np.float32)
        response = np.abs(cv2.remap(gx, x, y, cv2.INTER_LINEAR)*normal[0]
                          + cv2.remap(gy, x, y, cv2.INTER_LINEAR)*normal[1])
        peaks = np.argmax(response, axis=1)
        strengths = response[np.arange(len(base)), peaks]
        selected = offsets[peaks][strengths >= np.percentile(strengths, 30)]
        median = float(np.median(selected))
        records.append((abs(median), float(np.median(np.abs(selected-median))),
                        float(np.median(strengths))))
    offset, mad, strength = max(r[0] for r in records), max(r[1] for r in records), min(r[2] for r in records)
    inside = bool((polygon >= 3).all() and (polygon[:, 0] < gray.shape[1]-3).all()
                  and (polygon[:, 1] < gray.shape[0]-3).all())
    return {"accepted": inside and offset <= 5 and mad <= 2 and strength >= 15,
            "inside": inside, "max_offset_px": offset, "max_mad_px": mad,
            "min_gradient": strength, "margin_px": int(np.ceil(offset))+3}
