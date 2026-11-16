import datetime as dt
import json

import numpy as np
import pytest

from pvdelta.conditions import ConditionObservation
from pvdelta.config import DetectionConfig
from pvdelta.models import ModuleObservation, ThermalFrame
from pvdelta.normalize import Extent, irradiance
from pvdelta.pipeline import analyze_annotations, analyze_module, execute_stage1
from pvdelta.regions import ModuleAnnotation


def _case():
    capture = dt.datetime(2026, 4, 29, 13, 0, tzinfo=dt.timezone.utc)
    grid = np.full((30, 30), 25.0, dtype=np.float32)
    grid[12:17, 12:17] = 30
    frame = ThermalFrame("f", "synthetic", "hash", grid, None, 30, 30,
                         capture_time=capture)
    module = ModuleObservation("m", "f", [(3, 3), (27, 3), (27, 27), (3, 27)])
    mask = np.zeros(grid.shape, dtype=bool)
    mask[3:27, 3:27] = True
    return capture, frame, module, mask


def test_pipeline_normalizes_only_time_and_scope_matched_conditions():
    capture, frame, module, mask = _case()
    fresh = ConditionObservation(irradiance(800), capture, "poa-roof", "plant")
    result = analyze_module(
        frame, module, mask, condition_observations=[fresh],
        extent=Extent.AREA,
    )
    finding = result.findings[0]
    assert finding.is_normalized
    assert finding.normalization_sources == ["poa-roof"]
    assert any("formula" in flag for flag in finding.normalization_flags)
    stale = ConditionObservation(irradiance(800), capture - dt.timedelta(minutes=2),
                                 "poa-old", "plant")
    result = analyze_module(
        frame, module, mask, condition_observations=[stale],
        extent=Extent.AREA,
    )
    assert result.findings[0].is_normalized is False
    assert any("stale" in flag for flag in result.findings[0].normalization_flags)


def test_pipeline_rejects_unusable_module_support():
    _, frame, module, mask = _case()
    frame.temperature_c[mask] = np.nan
    result = analyze_module(frame, module, mask)
    assert result.quality.image_analysis.value == "unavailable"
    assert result.findings == ()


def test_frame_pipeline_preserves_zero_findings_and_marks_boundary_candidates():
    temperature = np.full((30, 50), 25.0, dtype=np.float32)
    temperature[10:14, 10:14] = 31.0
    temperature[20:25, 42:46] = 31.0
    frame = ThermalFrame("f", "synthetic", "hash", temperature, None, 50, 30)
    annotations = [
        ModuleAnnotation("M01", ((2, 2), (22, 2), (22, 28), (2, 28))),
        ModuleAnnotation("M02", ((28, 2), (48, 2), (48, 28), (28, 28))),
        ModuleAnnotation("M03", ((23, 2), (27, 2), (27, 28), (23, 28))),
    ]
    results = analyze_annotations(
        frame,
        annotations,
        config=DetectionConfig(
            delta_threshold_k=3,
            min_region_px=4,
            min_reference_px=12,
            edge_erosion_px=0,
            detrend_gradient=False,
        ),
        direction="hot",
    )
    assert [len(item.findings) for item in results] == [1, 1, 0]
    assert results[0].findings[0].assessment_status == "candidate"
    assert results[1].findings[0].assessment_status == "candidate"
    assert results[2].assessment_status == "analyzed"


def test_execute_stage1_writes_matching_report_and_database(tmp_path):
    pytest.importorskip("duckdb")
    temperature = np.full((20, 20), 25.0, dtype=np.float32)
    temperature[8:12, 8:12] = 31.0
    frame = ThermalFrame("f", "synthetic", "hash", temperature, None, 20, 20)
    annotation = ModuleAnnotation("M01", ((2, 2), (18, 2), (18, 18), (2, 18)))
    database = tmp_path / "result.duckdb"
    result = execute_stage1(
        frame,
        [annotation],
        output_dir=tmp_path / "report",
        run_id="analysis-1",
        config=DetectionConfig(detrend_gradient=False),
        direction="hot",
        database_path=database,
    )
    assert result.store_summary is not None
    assert result.store_summary.findings == 1
    header = json.loads(result.report_artifacts.jsonl_path.read_text().splitlines()[0])
    assert header["finding_count"] == 1

    # An explicit overwrite rerun replaces both artifacts and DB slice without duplication.
    execute_stage1(
        frame,
        [annotation],
        output_dir=tmp_path / "report",
        run_id="analysis-1",
        config=DetectionConfig(detrend_gradient=False),
        direction="hot",
        database_path=database,
        overwrite=True,
    )
    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    assert connection.execute("SELECT count(*) FROM module_observations").fetchone()[0] == 1
    assert connection.execute("SELECT count(*) FROM findings").fetchone()[0] == 1
    connection.close()


def test_explicit_native_masks_are_used_without_polygon_rasterization():
    temperature = np.full((20, 20), 25.0, dtype=np.float32)
    temperature[8:12, 8:12] = 31.0
    frame = ThermalFrame("f", "synthetic", "hash", temperature, None, 20, 20)
    annotation = ModuleAnnotation(
        "M01",
        ((0, 0), (20, 0), (20, 20), (0, 20)),
        mask_label=1,
    )
    mask = np.zeros((20, 20), dtype=bool)
    mask[4:16, 4:16] = True
    results = analyze_annotations(
        frame,
        [annotation],
        module_masks={"M01": mask},
        config=DetectionConfig(detrend_gradient=False, edge_erosion_px=0),
        direction="hot",
    )
    assert len(results[0].findings) == 1
    assert results[0].findings[0].n_region_px == 16
    rejected = analyze_annotations(
        frame,
        [annotation],
        module_masks={"M01": mask},
        region_margin_px=1,
    )
    assert rejected[0].assessment_status == "not_assessed"
    assert "region_margin_px" in rejected[0].reasons[0]
