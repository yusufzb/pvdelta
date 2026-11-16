import numpy as np
import pytest

from pvdelta.tables import (
    TableSize,
    TableSpec,
    build_table_grid,
    focus_cells,
    index_visible_polygons,
    infer_table_head_tail,
)


def test_four_rows_and_long_axis_cell_count_are_explicit():
    grid = build_table_grid(
        TableSpec(TableSize.NINE),
        ((0, 0), (0, 4)),
        ((9, 0), (9, 4)),
    )
    assert len(grid.cells) == 36
    assert [(cell.row, cell.column) for cell in grid.cells[:3]] == [("A", 1), ("A", 2), ("A", 3)]
    assert grid.cells[-1].row == "D"
    assert grid.cells[-1].column == 9


def test_twenty_seven_table_generates_four_known_rows():
    grid = build_table_grid(
        TableSpec(TableSize.TWENTY_SEVEN),
        ((0, 0), (0, 4)),
        ((27, 0), (27, 4)),
    )
    assert grid.spec.cell_count == 108
    assert len(grid.cells) == 108
    assert grid.cells[27].row == "B"
    assert grid.cells[27].column == 1


def test_inference_recovers_head_tail_and_keeps_topology_prior():
    points = np.array([(x, y) for x in range(9) for y in range(4)], dtype=float)
    grid = infer_table_head_tail(points, TableSpec(TableSize.NINE))
    assert grid.evidence == "inferred_from_native_thermal_candidates"
    assert len(grid.cells) == 36
    assert {cell.row for cell in grid.cells} == {"A", "B", "C", "D"}


def test_invalid_table_row_count_is_rejected():
    with pytest.raises(ValueError, match="exactly four"):
        TableSpec(TableSize.NINE, rows=3)


def test_focus_keeps_candidate_and_local_reference_neighbours():
    grid = build_table_grid(TableSpec(TableSize.NINE), ((0, 0), (0, 4)), ((9, 0), (9, 4)))
    focused = focus_cells(grid, {("B", 4)}, radius=1)
    assert {("B", cell.column) for cell in focused} >= {("B", 3), ("B", 4), ("B", 5)}
    assert ("A", 4) in {(cell.row, cell.column) for cell in focused}
    assert ("D", 4) not in {(cell.row, cell.column) for cell in focused}


def test_visible_polygons_share_columns_across_four_rows():
    polygons = []
    for column in range(1, 4):
        for row in range(4):
            x0, y0 = row * 2.0 + 0.1 * column, column * 10.0
            polygons.append(((x0, y0), (x0 + 1.5, y0),
                             (x0 + 1.5, y0 + 8.0), (x0, y0 + 8.0)))
    cells = index_visible_polygons(list(reversed(polygons)))
    assert len(cells) == 12
    assert {(cell.row, cell.column) for cell in cells} == {
        (row, column) for row in "ABCD" for column in (1, 2, 3)
    }


def test_visible_polygon_axis_can_be_explicitly_oriented():
    polygons = [
        ((0, 0), (1, 0), (1, 5), (0, 5)),
        ((2, 0), (3, 0), (3, 5), (2, 5)),
        ((4, 0), (5, 0), (5, 5), (4, 5)),
        ((6, 0), (7, 0), (7, 5), (6, 5)),
        ((0, 10), (1, 10), (1, 15), (0, 15)),
        ((2, 10), (3, 10), (3, 15), (2, 15)),
        ((4, 10), (5, 10), (5, 15), (4, 15)),
        ((6, 10), (7, 10), (7, 15), (6, 15)),
    ]
    cells = index_visible_polygons(polygons, long_axis=(0, 1), row_axis=(1, 0))
    assert {(cell.row, cell.column) for cell in cells} == {
        (row, column) for row in "ABCD" for column in (1, 2)
    }
