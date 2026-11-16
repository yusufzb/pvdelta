import csv
import json

import numpy as np
import pytest

from pvdelta.models import Finding, ModuleObservation, ThermalFrame
from pvdelta.report import ModuleReport, write_stage1_report


def _frame() -> ThermalFrame:
    temperature = np.full((20, 30), 25.0, dtype=np.float32)
    return ThermalFrame(
        "frame-1",
        "C:/özel/saha/termal.JPG",
        "sha256-value",
        temperature,
        None,
        30,
        20,
    )


def _modules() -> tuple[ModuleObservation, ModuleObservation]:
    return (
        ModuleObservation("M01", "frame-1", [(2, 2), (12, 2), (12, 16), (2, 16)]),
        ModuleObservation("M02", "frame-1", [(15, 2), (25, 2), (25, 16), (15, 16)]),
    )


def test_zero_finding_report_preserves_analyzed_and_unassessed_scope(tmp_path):
    first, second = _modules()
    artifacts = write_stage1_report(
        tmp_path / "Türkçe çıktı",
        "run-001",
        _frame(),
        (
            ModuleReport(first),
            ModuleReport(second, assessment_status="not_assessed", reasons=("invalid pixels",)),
        ),
        settings={"threshold_k": 3.0},
    )
    records = [json.loads(line) for line in artifacts.jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["finding_count"] == 0
    assert records[0]["analyzed_module_count"] == 1
    assert records[0]["not_assessed_module_count"] == 1
    assert records[0]["source_name"] == "termal.JPG"
    assert [item["assessment_status"] for item in records[1:]] == ["no_candidate", "not_assessed"]
    with artifacts.csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["module_status"] for row in rows] == ["no_candidate", "not_assessed"]
    assert "invalid pixels" in rows[1]["reasons"]


def test_finding_report_keeps_exact_support_and_native_overlay(tmp_path):
    first, _ = _modules()
    finding = Finding(
        "F01",
        "M01",
        "frame-1",
        "hot_region",
        5.0,
        31.0,
        [(5, 6), (7, 6), (7, 8), (5, 8)],
        reference_t_c=26.0,
        reference_method="module_reference",
        n_region_px=3,
        n_reference_px=50,
        region_pixels_yx=[(6, 5), (6, 6), (7, 6)],
        region_max_t_c=32.0,
        region_p95_t_c=31.9,
        region_max_residual_k=6.0,
    )
    artifacts = write_stage1_report(
        tmp_path,
        "run-002",
        _frame(),
        (ModuleReport(first, (finding,)),),
    )
    records = [json.loads(line) for line in artifacts.jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert records[2]["finding"]["region_pixels_yx"] == [[6, 5], [6, 6], [7, 6]]
    svg = artifacts.overlay_path.read_text(encoding="utf-8")
    assert 'viewBox="0 0 30 20"' in svg
    assert "M5 6h1v1h-1z" in svg
    assert "M01 +5.0 K" in svg


def test_report_refuses_implicit_overwrite_and_invalid_module_state(tmp_path):
    first, _ = _modules()
    write_stage1_report(tmp_path, "run-003", _frame(), (ModuleReport(first),))
    with pytest.raises(FileExistsError):
        write_stage1_report(tmp_path, "run-003", _frame(), (ModuleReport(first),))
    with pytest.raises(ValueError, match="not-assessed"):
        ModuleReport(
            first,
            (
                Finding(
                    "F",
                    "M01",
                    "frame-1",
                    "hot_region",
                    3.0,
                    30.0,
                    [(1, 1), (2, 1), (2, 2)],
                ),
            ),
            assessment_status="not_assessed",
        )
