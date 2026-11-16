"""Native-grid finding support diagnostics; never accepts a mask or removes a finding."""
from __future__ import annotations

import numpy as np

from pvdelta.reference import erode


def review_region_support(
    envelope: np.ndarray, region_pixels_yx, *, measurement_mask: np.ndarray | None = None,
    insets: tuple[int, ...] = (0, 1, 2, 3, 5),
) -> dict:
    """Count exact finding pixels retained by geometric polygon insets.

    Insets are sensitivity probes, not calibrated registration uncertainty or a
    vegetation classifier. No temperature thresholds, bounding boxes, or enlarged
    display outlines are used to reconstruct the measured region.
    """
    if envelope.ndim != 2 or envelope.dtype != np.bool_ or not envelope.any():
        raise ValueError("envelope must be a nonempty native boolean mask")
    support = envelope if measurement_mask is None else measurement_mask
    if (support.shape != envelope.shape or support.dtype != np.bool_
            or np.any(support & ~envelope)):
        raise ValueError("measurement mask must be boolean, match envelope, and stay inside it")
    if (not insets or insets[0] != 0 or len(set(insets)) != len(insets)
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in insets)
            or tuple(sorted(insets)) != insets):
        raise ValueError("insets must be increasing nonnegative integers starting at zero")
    if region_pixels_yx is None or len(region_pixels_yx) == 0:
        return {"status": "unavailable", "reason": "exact finding pixels were not stored",
                "insets": [], "outside_envelope_pixels": None}
    pixels = np.asarray(region_pixels_yx)
    if (pixels.ndim != 2 or pixels.shape[1] != 2
            or pixels.dtype.kind not in "iu" or (pixels < 0).any()
            or (pixels[:, 0] >= envelope.shape[0]).any()
            or (pixels[:, 1] >= envelope.shape[1]).any()
            or len(np.unique(pixels, axis=0)) != len(pixels)):
        raise ValueError("finding pixels must be unique integer (y, x) indices inside native grid")
    y, x = pixels.T
    rows = []
    for inset in insets:
        mask = erode(support, inset)
        count = int(mask[y, x].sum())
        rows.append({"inset_px": inset, "remaining_region_pixels": count,
                     "retained_fraction": count / len(pixels),
                     "remaining_mask_pixels": int(mask.sum())})
    outside = int((~envelope[y, x]).sum())
    outside_support = len(pixels) - rows[0]["remaining_region_pixels"]
    status = "outside_envelope" if outside else "outside_measurement_support" if outside_support else (
        "inset_sensitive" if rows[-1]["remaining_region_pixels"] < len(pixels)
        else "retained_at_tested_insets"
    )
    return {"status": status, "region_pixels": len(pixels),
            "outside_envelope_pixels": outside, "insets": rows,
            "outside_measurement_pixels": outside_support,
            "probe_basis": "explicit_measurement_mask" if measurement_mask is not None else "module_polygon",
            "semantic_panel_membership": "unverified",
            "finding_action": "retain_for_review"}
