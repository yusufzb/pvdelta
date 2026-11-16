"""Thermal-first anomaly candidates scoped to a reviewed table grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pvdelta.models import ThermalFrame
from pvdelta.reference import erode, robust_median
from pvdelta.regions import polygon_to_mask
from pvdelta.tables import TableGrid


@dataclass(frozen=True)
class CellTemperature:
    row: str
    column: int
    temperature_c: float
    n_pixels: int
    polygon_xy: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class TableCandidate:
    row: str
    column: int
    temperature_c: float
    reference_c: float
    delta_c: float
    reference_cells: tuple[tuple[str, int], ...]
    polygon_xy: tuple[tuple[float, float], ...]


def summarize_table(
    frame: ThermalFrame,
    grid: TableGrid,
    *,
    edge_erosion_px: int = 1,
    min_pixels: int = 4,
) -> tuple[CellTemperature, ...]:
    """Compute robust native-thermal medians for valid table cells."""
    if edge_erosion_px < 0 or min_pixels < 1:
        raise ValueError("edge_erosion_px must be non-negative and min_pixels positive")
    valid = np.isfinite(frame.temperature_c)
    if frame.valid_mask is not None:
        valid &= frame.valid_mask
    result: list[CellTemperature] = []
    for cell in grid.cells:
        try:
            mask = polygon_to_mask(list(cell.polygon_xy), frame.temperature_c.shape)
        except ValueError:
            continue
        mask = erode(mask, edge_erosion_px) & valid
        if int(mask.sum()) < min_pixels:
            continue
        median, count = robust_median(frame.temperature_c, mask)
        result.append(CellTemperature(cell.row, cell.column, median, count, cell.polygon_xy))
    return tuple(result)


def focus_table_anomalies(
    frame: ThermalFrame,
    grid: TableGrid,
    *,
    threshold_k: float = 2.5,
    neighbour_radius: int = 1,
    min_reference_cells: int = 3,
    edge_erosion_px: int = 1,
    min_pixels: int = 4,
) -> tuple[TableCandidate, ...]:
    """Return whole-cell thermal candidates against local table peers.

    This is a screening result, not a fault diagnosis.  Every candidate retains
    the reference cells used for its signed delta; cells with insufficient peers
    are omitted instead of being treated as normal.
    """
    if not np.isfinite(threshold_k) or threshold_k <= 0:
        raise ValueError("threshold_k must be finite and positive")
    if neighbour_radius < 1 or min_reference_cells < 1:
        raise ValueError("neighbour_radius and min_reference_cells must be positive")
    values = summarize_table(frame, grid, edge_erosion_px=edge_erosion_px, min_pixels=min_pixels)
    candidates: list[TableCandidate] = []
    for value in values:
        peers = []
        for peer in values:
            if peer is value:
                continue
            row_distance = abs("ABCD".index(value.row) - "ABCD".index(peer.row))
            if row_distance + abs(value.column - peer.column) <= neighbour_radius:
                peers.append(peer)
        if len(peers) < min_reference_cells:
            continue
        reference = float(np.median([peer.temperature_c for peer in peers]))
        delta = value.temperature_c - reference
        if abs(delta) < threshold_k:
            continue
        candidates.append(TableCandidate(
            row=value.row,
            column=value.column,
            temperature_c=value.temperature_c,
            reference_c=reference,
            delta_c=delta,
            reference_cells=tuple((peer.row, peer.column) for peer in peers),
            polygon_xy=value.polygon_xy,
        ))
    return tuple(candidates)
