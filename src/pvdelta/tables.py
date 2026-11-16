"""Table (sehpa) geometry for thermal-first module candidate analysis.

The physical table is the unit of geometry.  RGB registration is optional evidence;
thermal candidates can be assigned to a four-row table grid once its head and tail
are known or inferred from native-thermal panel evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np


class TableSize(str, Enum):
    NINE = "9"
    TWENTY_SEVEN = "27"

    @property
    def modules_along_length(self) -> int:
        return int(self.value)


@dataclass(frozen=True)
class TableSpec:
    """Physical layout prior.

    `modules_along_length` is deliberately separate from the total cell count:
    four tilt-direction rows are a site fact, while 9/27 describes the long axis.
    """

    size: TableSize
    rows: int = 4

    def __post_init__(self) -> None:
        if self.rows != 4:
            raise ValueError("the current table prior requires exactly four tilt-direction rows")

    @property
    def modules_along_length(self) -> int:
        return self.size.modules_along_length

    @property
    def cell_count(self) -> int:
        return self.rows * self.modules_along_length


@dataclass(frozen=True)
class TableCell:
    row: str
    column: int
    polygon_xy: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class TableGrid:
    spec: TableSpec
    head_xy: tuple[tuple[float, float], tuple[float, float]]
    tail_xy: tuple[tuple[float, float], tuple[float, float]]
    cells: tuple[TableCell, ...]
    evidence: str


def _cluster_1d(values: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic one-dimensional k-means used for visible row bands."""
    if count < 1 or len(values) < count:
        raise ValueError("cluster count must be positive and no larger than values")
    centres = np.quantile(values, np.linspace(0.08, 0.92, count))
    for _ in range(64):
        labels = np.argmin(np.abs(values[:, None] - centres[None, :]), axis=1)
        updated = np.asarray([
            values[labels == index].mean() if np.any(labels == index) else centres[index]
            for index in range(count)
        ])
        if np.allclose(updated, centres):
            break
        centres = updated
    order = np.argsort(centres)
    remap = np.empty(count, dtype=int)
    remap[order] = np.arange(count)
    return remap[labels], centres[order]


def index_visible_polygons(
    polygons: Sequence[Sequence[tuple[float, float]]],
    *,
    rows: int = 4,
    row_labels: Sequence[str] = ("A", "B", "C", "D"),
    long_axis: tuple[float, float] | None = None,
    row_axis: tuple[float, float] | None = None,
    evidence: str = "visible_panel_polygons_four_row_index",
) -> tuple[TableCell, ...]:
    """Index visible panel polygons on two table axes.

    Row bands are clustered across the tilt direction. Long-axis positions are clustered
    globally so a position keeps the same visible column number across rows. Columns are
    visible-frame indices, never CAD or absolute 9/27 IDs. Supply explicit axes once image
    orientation is tied to the physical table; otherwise PCA gives a deterministic image order.
    """
    del evidence  # retained in the API for callers that record provenance separately
    if rows != 4:
        raise ValueError("the current table prior requires exactly four rows")
    if len(row_labels) != rows or len(set(row_labels)) != rows:
        raise ValueError("row_labels must contain four unique labels")
    if len(polygons) < rows * 2:
        raise ValueError("at least eight visible polygons are required")
    arrays = [np.asarray(polygon, dtype=float) for polygon in polygons]
    if any(array.ndim != 2 or array.shape[1] != 2 or len(array) < 3
           or not np.isfinite(array).all() for array in arrays):
        raise ValueError("polygons must be finite sequences of at least three (x, y) points")
    centres = np.asarray([array.mean(axis=0) for array in arrays], dtype=float)
    centred = centres - centres.mean(axis=0)

    def unit(value: tuple[float, float] | np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        norm = float(np.linalg.norm(vector))
        if vector.shape != (2,) or not np.isfinite(vector).all() or norm <= 1e-9:
            raise ValueError(f"{name} must be a finite non-zero 2D vector")
        return vector / norm

    if long_axis is None and row_axis is None:
        _, _, vectors = np.linalg.svd(centred, full_matrices=False)
        long = unit(vectors[0], "long_axis")
        if np.dot(long, np.array([0.0, 1.0])) < 0:
            long = -long
        across = np.array([long[1], -long[0]])
        if np.dot(across, np.array([1.0, 0.0])) < 0:
            across = -across
    elif long_axis is not None:
        long = unit(long_axis, "long_axis")
        across = np.array([long[1], -long[0]])
        if row_axis is not None and np.dot(unit(row_axis, "row_axis"), across) < 0:
            across = -across
    else:
        across = unit(row_axis, "row_axis")
        long = np.array([-across[1], across[0]])
        if np.dot(long, np.array([0.0, 1.0])) < 0:
            long = -long

    row_indices, _ = _cluster_1d(centred @ across, rows)
    long_values = centred @ long
    order = np.argsort(long_values)
    projections = [array @ long for array in arrays]
    typical_length = float(np.median([value.max() - value.min() for value in projections]))
    if not np.isfinite(typical_length) or typical_length <= 1e-9:
        raise ValueError("polygons have no measurable long-axis extent")
    tolerance = typical_length * 0.35
    position_indices = np.empty(len(arrays), dtype=int)
    position = 0
    running: list[float] = [float(long_values[order[0]])]
    position_indices[order[0]] = position
    for index in order[1:]:
        value = float(long_values[index])
        if value - float(np.mean(running)) > tolerance:
            position += 1
            running = []
        running.append(value)
        position_indices[index] = position

    return tuple(
        TableCell(str(row_labels[row_indices[index]]), int(position_indices[index]) + 1,
                  tuple(tuple(float(value) for value in point) for point in arrays[index]))
        for index in range(len(arrays))
    )


def focus_cells(
    grid: TableGrid,
    candidates: set[tuple[str, int]],
    *,
    radius: int = 1,
) -> tuple[TableCell, ...]:
    """Return anomaly candidates plus nearby peers for thermal comparison.

    A candidate is never evaluated in isolation: its local table neighbours remain
    available as the reference set.  With no candidates the function returns no
    cells, allowing callers to avoid scanning every module in the reporting stage.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    keys = {(cell.row, cell.column) for cell in grid.cells}
    unknown = candidates - keys
    if unknown:
        raise ValueError(f"candidate cells are not in table grid: {sorted(unknown)!r}")
    selected = {
        (row, column)
        for row, column in keys
        if any(
            abs("ABCD".index(row) - "ABCD".index(candidate_row))
            + abs(column - candidate_column) <= radius
            for candidate_row, candidate_column in candidates
        )
    }
    return tuple(cell for cell in grid.cells if (cell.row, cell.column) in selected)


def _point(point: tuple[float, float] | np.ndarray) -> np.ndarray:
    value = np.asarray(point, dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError("table points must be finite (x, y) pairs")
    return value


def _bilinear(
    head_left: np.ndarray,
    head_right: np.ndarray,
    tail_right: np.ndarray,
    tail_left: np.ndarray,
    u: float,
    v: float,
) -> np.ndarray:
    """Interpolate from head (u=0) to tail (u=1), top to bottom (v=0..1)."""
    top = (1.0 - u) * head_left + u * tail_left
    bottom = (1.0 - u) * head_right + u * tail_right
    return (1.0 - v) * top + v * bottom


def build_table_grid(
    spec: TableSpec,
    head_xy: tuple[tuple[float, float], tuple[float, float]],
    tail_xy: tuple[tuple[float, float], tuple[float, float]],
    *,
    evidence: str = "reviewed_table_head_tail",
) -> TableGrid:
    """Create four-row cells between reviewed head and tail cross-sections."""
    head_left, head_right = (_point(value) for value in head_xy)
    tail_left, tail_right = (_point(value) for value in tail_xy)
    if np.linalg.norm(head_left - head_right) < 1e-6:
        raise ValueError("table head has zero width")
    if np.linalg.norm(tail_left - tail_right) < 1e-6:
        raise ValueError("table tail has zero width")
    cells: list[TableCell] = []
    row_names = ("A", "B", "C", "D")
    n = spec.modules_along_length
    for row, name in enumerate(row_names):
        v0, v1 = row / spec.rows, (row + 1) / spec.rows
        for column in range(n):
            u0, u1 = column / n, (column + 1) / n
            polygon = (
                tuple(_bilinear(head_left, head_right, tail_right, tail_left, u0, v0)),
                tuple(_bilinear(head_left, head_right, tail_right, tail_left, u1, v0)),
                tuple(_bilinear(head_left, head_right, tail_right, tail_left, u1, v1)),
                tuple(_bilinear(head_left, head_right, tail_right, tail_left, u0, v1)),
            )
            cells.append(TableCell(name, column + 1, polygon))
    return TableGrid(spec, tuple(tuple(value) for value in head_xy),
                     tuple(tuple(value) for value in tail_xy), tuple(cells), evidence)


def infer_table_head_tail(
    centres_xy: np.ndarray,
    spec: TableSpec,
    *,
    edge_fraction: float = 0.15,
) -> TableGrid:
    """Infer head/tail cross-sections from native-thermal candidate centres.

    This is a topology seed, not an accepted panel segmentation.  It requires at
    least eight centres so both ends and the four-row span have support.
    """
    centres = np.asarray(centres_xy, dtype=float)
    if centres.ndim != 2 or centres.shape[1] != 2 or not np.isfinite(centres).all():
        raise ValueError("centres_xy must be a finite (n, 2) array")
    if len(centres) < 8:
        raise ValueError("at least eight thermal candidate centres are required")
    centred = centres - centres.mean(axis=0)
    _, _, vectors = np.linalg.svd(centred, full_matrices=False)
    axis = vectors[0]
    if axis[0] < 0 or (abs(axis[0]) < 1e-9 and axis[1] < 0):
        axis = -axis
    cross = np.array([-axis[1], axis[0]])
    along = centred @ axis
    across = centred @ cross
    span = float(along.max() - along.min())
    if span <= 1e-6:
        raise ValueError("table candidates have no measurable long-axis span")
    band = max(span * edge_fraction, 1e-6)
    low = along <= along.min() + band
    high = along >= along.max() - band
    if low.sum() < 2 or high.sum() < 2:
        raise ValueError("table head and tail do not have enough candidate support")
    low_u = float(along[low].mean())
    high_u = float(along[high].mean())
    cross_low, cross_high = float(across.min()), float(across.max())
    origin = centres.mean(axis=0)
    head = (tuple(origin + low_u * axis + cross_low * cross),
            tuple(origin + low_u * axis + cross_high * cross))
    tail = (tuple(origin + high_u * axis + cross_low * cross),
            tuple(origin + high_u * axis + cross_high * cross))
    return build_table_grid(spec, head, tail, evidence="inferred_from_native_thermal_candidates")
