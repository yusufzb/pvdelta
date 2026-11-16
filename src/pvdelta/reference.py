"""Local reference temperatures and signed ΔT.

All statistics are computed on the native thermal grid. A mask is a boolean array of the same
shape as the temperature raster (True = pixel belongs to the region).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeltaT:
    delta_k: float          # signed: + warmer than reference, - cooler
    t_region_c: float       # statistic of the anomalous region (max for hot, min for cold)
    t_reference_c: float    # robust reference (median)
    n_region_px: int
    n_reference_px: int
    reference: str          # "module_median" | "row_median" | "string_median" | ...


def erode(mask: np.ndarray, px: int = 1) -> np.ndarray:
    """Shrink a mask by ``px`` pixels (4-neighbourhood) to drop frame / mixed edge pixels."""
    if mask.ndim != 2 or px < 0:
        raise ValueError("mask must be two-dimensional and px non-negative")
    m = mask.astype(bool).copy()
    for _ in range(px):
        inner = m.copy()
        inner[1:, :] &= m[:-1, :]
        inner[:-1, :] &= m[1:, :]
        inner[:, 1:] &= m[:, :-1]
        inner[:, :-1] &= m[:, 1:]
        inner[0, :] = inner[-1, :] = False
        inner[:, 0] = inner[:, -1] = False
        m = inner
    return m


def robust_median(t: np.ndarray, mask: np.ndarray, exclude: np.ndarray | None = None) -> tuple[float, int]:
    if t.ndim != 2 or mask.shape != t.shape:
        raise ValueError("temperature and mask shapes must match")
    sel = mask.astype(bool).copy()
    if exclude is not None:
        if exclude.shape != t.shape:
            raise ValueError("exclusion mask shape must match temperature")
        sel &= ~exclude.astype(bool)
    vals = t[sel]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("reference region is empty after exclusions")
    return float(np.median(vals)), int(vals.size)


def delta_t_region(
    t: np.ndarray,
    region: np.ndarray,
    reference_mask: np.ndarray,
    *,
    sign: str = "hot",
    reference: str = "module_median",
) -> DeltaT:
    """ΔT of ``region`` against the median of ``reference_mask`` with the region excluded.

    ``sign="hot"`` uses the region maximum, ``"cold"`` the minimum, ``"mean"`` the mean
    (appropriate for large-area anomalies such as a warm sub-string).
    """
    if t.ndim != 2 or region.shape != t.shape or reference_mask.shape != t.shape:
        raise ValueError("temperature and region/reference mask shapes must match")
    if sign not in {"hot", "cold", "mean"}:
        raise ValueError("sign must be hot, cold or mean")
    region = region.astype(bool)
    rvals = t[region]
    rvals = rvals[np.isfinite(rvals)]
    if rvals.size == 0:
        raise ValueError("anomaly region is empty")
    stat = {"hot": np.max, "cold": np.min, "mean": np.mean}[sign]
    t_reg = float(stat(rvals))
    t_ref, n_ref = robust_median(t, reference_mask, exclude=region)
    return DeltaT(t_reg - t_ref, t_reg, t_ref, int(rvals.size), n_ref, reference)


def group_median_delta(values: dict[str, float]) -> dict[str, float]:
    """Signed ΔT of each member (e.g. module or string median temperature) vs the group median.

    Use for whole-module or string-level comparisons inside one row / one inverter block.
    """
    if not values:
        return {}
    if not np.isfinite(list(values.values())).all():
        raise ValueError("group temperatures must be finite")
    med = float(np.median(list(values.values())))
    return {k: v - med for k, v in values.items()}
