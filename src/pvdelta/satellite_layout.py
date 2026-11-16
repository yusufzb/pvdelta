"""Coarse satellite screenshot priors; never survey control or panel inventory."""
from __future__ import annotations

import numpy as np


def fit_translation(source_xy, reference_xy, groups):
    """Keep orientation and scale fixed; hold each entire matched row out in turn."""
    source, reference = np.asarray(source_xy, float), np.asarray(reference_xy, float)
    groups = np.asarray(groups)
    if (source.ndim != 2 or source.shape[1] != 2 or reference.shape != source.shape
            or len(groups) != len(source) or not np.isfinite(source).all()
            or not np.isfinite(reference).all() or len(np.unique(groups)) < 3):
        raise ValueError("At least three finite, independently matched row groups are required")
    offsets = reference-source
    # Equal weight per row, even if individual rows have different point counts.
    group_offsets = np.array([np.median(offsets[groups == g], axis=0)
                              for g in np.unique(groups)])
    shift = np.median(group_offsets, axis=0)
    held = []
    for i, group in enumerate(np.unique(groups)):
        trial = np.median(np.delete(group_offsets, i, axis=0), axis=0)
        held.extend(np.linalg.norm(source[groups == group]+trial-reference[groups == group], axis=1))
    before = np.linalg.norm(offsets, axis=1)
    after = np.linalg.norm(source+shift-reference, axis=1)
    return shift, {"before_median": float(np.median(before)),
                   "after_median": float(np.median(after)),
                   "held_out_row_median": float(np.median(held)),
                   "held_out_row_p95": float(np.percentile(held, 95)),
                   "group_count": len(group_offsets),
                   "scale": 1.0, "rotation_deg": 0.0}


def screenshot_to_mercator(pixels, centre_xy, size, zoom):
    """Approximate north-up, unpitched map screenshot mapping in EPSG:3857 metres.

    Caller must verify map viewport, centre and zoom. These are screenshot-derived
    coordinates, not independent geographic control.
    """
    pixels = np.asarray(pixels, float)
    centre, size = np.asarray(centre_xy, float), np.asarray(size, float)
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or centre.shape != (2,)
            or size.shape != (2,) or np.any(size <= 0) or not 0 <= zoom <= 24
            or not np.isfinite(pixels).all() or not np.isfinite(centre).all()
            or not np.isfinite(size).all()):
        raise ValueError("Invalid screenshot geometry")
    resolution = 2*np.pi*6378137/(256*2**zoom)
    return centre + (pixels-size/2)*[resolution, -resolution]
