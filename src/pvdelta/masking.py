"""Temperature-independent measurement support on the native radiometric grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pvdelta.reference import erode


@dataclass(frozen=True)
class MeasurementSupport:
    envelope: np.ndarray
    interior: np.ndarray
    excluded_edge: np.ndarray
    margin_px: int

    def valid_mask(self, temperature: np.ndarray) -> np.ndarray:
        if temperature.shape != self.interior.shape:
            raise ValueError("temperature must match the native mask shape")
        return self.interior & np.isfinite(temperature)

    def masked_temperature(self, temperature: np.ndarray) -> np.ndarray:
        """Preserve native values; excluded pixels are NaN, never zero degrees."""
        return np.where(self.valid_mask(temperature), temperature, np.nan)


@dataclass(frozen=True)
class TemperatureSummary:
    """Descriptive native-pixel statistics, not an anomaly or calibration decision."""

    status: str
    interior_pixels: int
    valid_pixels: int
    invalid_pixels: int
    min_c: float | None
    median_c: float | None
    p95_c: float | None
    max_c: float | None


def preview_mask_to_native(
    preview_mask: np.ndarray,
    native_shape: tuple[int, int],
    *,
    require_full_preview_coverage: bool = True,
) -> np.ndarray:
    """Reduce an aligned preview mask onto the native radiometric grid.

    Only exact integer grid ratios are accepted.  With the conservative default,
    every preview subpixel covering a native pixel must belong to the mask.
    Temperature values are never resampled by this operation.
    """
    preview = np.asarray(preview_mask)
    if preview.ndim != 2 or preview.dtype != np.bool_:
        raise ValueError("preview_mask must be a two-dimensional boolean array")
    if len(native_shape) != 2 or min(native_shape) <= 0:
        raise ValueError("native_shape must be a positive (height, width) pair")
    native_h, native_w = native_shape
    preview_h, preview_w = preview.shape
    if preview_h % native_h or preview_w % native_w:
        raise ValueError("preview and native grids require an exact integer ratio")
    scale_y, scale_x = preview_h // native_h, preview_w // native_w
    blocks = preview.reshape(native_h, scale_y, native_w, scale_x)
    native = blocks.all(axis=(1, 3)) if require_full_preview_coverage else blocks.any(axis=(1, 3))
    native.setflags(write=False)
    return native


def summarize_temperature(
    temperature: np.ndarray, support: MeasurementSupport
) -> TemperatureSummary:
    """Summarize only finite interior samples; empty support never means healthy.

    Callers must preserve source identity, radiometric settings and boundary-review status
    alongside this result. No resampling or environment correction is performed here.
    """
    values = temperature[support.valid_mask(temperature)]
    total = int(support.interior.sum())
    valid = int(values.size)
    if valid == 0:
        return TemperatureSummary("unavailable", total, 0, total, None, None, None, None)
    return TemperatureSummary(
        "finite_support" if valid == total else "incomplete_support",
        total, valid, total - valid,
        float(np.min(values)), float(np.median(values)),
        float(np.percentile(values, 95)), float(np.max(values)),
    )


def build_measurement_support(
    native_envelopes: list[np.ndarray], *, margin_px: int
) -> tuple[MeasurementSupport, ...]:
    """Shrink disjoint native envelopes without looking at temperature values.

    Inputs must already be registered to the native thermal grid. This function does not
    infer RGB registration, validate panel identity or accept a temperature-threshold mask
    as evidence of a panel boundary. Empty interiors are retained for explicit rejection.
    """
    if isinstance(margin_px, bool) or not isinstance(margin_px, int) or margin_px < 0:
        raise ValueError("margin_px must be a non-negative integer")
    if not native_envelopes:
        return ()
    shape = native_envelopes[0].shape
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("native envelopes must be non-empty two-dimensional arrays")
    occupied = np.zeros(shape, dtype=bool)
    result = []
    for envelope in native_envelopes:
        if envelope.shape != shape or envelope.dtype != np.bool_:
            raise ValueError("native envelopes must have matching shapes and boolean dtype")
        if not envelope.any():
            raise ValueError("panel envelope is empty")
        if np.any(occupied & envelope):
            raise ValueError("panel envelopes overlap; resolve instance boundaries first")
        occupied |= envelope
        full = envelope.copy()
        interior = erode(full, margin_px)
        edge = full & ~interior
        for array in (full, interior, edge):
            array.setflags(write=False)
        result.append(MeasurementSupport(full, interior, edge, margin_px))
    return tuple(result)
