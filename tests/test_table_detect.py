import datetime as dt

import numpy as np

from pvdelta.models import ThermalFrame
from pvdelta.table_detect import focus_table_anomalies, summarize_table
from pvdelta.tables import TableSize, TableSpec, build_table_grid


def _frame(values: np.ndarray) -> ThermalFrame:
    return ThermalFrame(
        frame_id="thermal-1",
        source_path="private/thermal.jpg",
        source_hash="a" * 64,
        temperature_c=values.astype(np.float32),
        valid_mask=np.ones(values.shape, dtype=bool),
        width=values.shape[1],
        height=values.shape[0],
        capture_time=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
    )


def test_table_summary_uses_native_cell_medians():
    grid = build_table_grid(TableSpec(TableSize.NINE), ((0, 0), (0, 40)), ((90, 0), (90, 40)))
    summary = summarize_table(_frame(np.full((40, 90), 25.0)), grid)
    assert len(summary) == 36
    assert all(item.n_pixels > 0 for item in summary)


def test_focus_returns_hot_cell_against_local_table_peers():
    values = np.full((40, 90), 25.0)
    values[10:20, 30:40] = 35.0  # row B, column 4
    grid = build_table_grid(TableSpec(TableSize.NINE), ((0, 0), (0, 40)), ((90, 0), (90, 40)))
    candidates = focus_table_anomalies(_frame(values), grid, threshold_k=2.5)
    assert len(candidates) == 1
    assert (candidates[0].row, candidates[0].column) == ("B", 4)
    assert candidates[0].delta_c > 9.0
    assert len(candidates[0].reference_cells) >= 3
