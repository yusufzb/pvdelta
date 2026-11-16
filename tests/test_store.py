import datetime as dt
import json

import pytest

duckdb = pytest.importorskip("duckdb")

from pvdelta.inventory import PanelAsset
from pvdelta.models import AssetMatch, Finding, MatchStatus, ModuleObservation
from pvdelta.radiometry import RadiometricFrameResult
from pvdelta.report import ModuleReport
from pvdelta.store import (
    CaptureRecord,
    initialize_schema,
    map_features,
    update_capture_footprint,
    update_capture_radiometry,
    upsert_capture,
    upsert_finding_review,
    upsert_panel_assets,
    upsert_run,
    write_asset_matches,
    write_stage1_results,
)


def test_panel_assets_round_trip_without_coordinates():
    connection = duckdb.connect(":memory:")
    initialize_schema(connection)
    assets = (
        PanelAsset("P01", "T01", "A", 1, 40.0, 26.0, "EPSG:4326", "S01"),
        PanelAsset("P02", "T01", "B", 1),
    )
    assert upsert_panel_assets(connection, assets) == 2
    rows = connection.execute(
        "SELECT panel_id, row_label, latitude, crs FROM panel_assets ORDER BY panel_id"
    ).fetchall()
    assert rows == [("P01", "A", 40.0, "EPSG:4326"), ("P02", "B", None, None)]
    upsert_panel_assets(connection, (PanelAsset("P01", "T02", "C", 3),))
    assert connection.execute(
        "SELECT table_id, row_label, position FROM panel_assets WHERE panel_id = 'P01'"
    ).fetchone() == ("T02", "C", 3)


def test_panel_assets_reject_invalid_inventory_without_partial_write():
    connection = duckdb.connect(":memory:")
    initialize_schema(connection)
    with pytest.raises(ValueError, match="invalid panel inventory"):
        upsert_panel_assets(connection, (PanelAsset("P01", "T01", "E", 0),))
    assert connection.execute("SELECT count(*) FROM panel_assets").fetchone()[0] == 0


def test_asset_matches_round_trip_is_separate_from_stage1_observations():
    connection = duckdb.connect(":memory:")
    initialize_schema(connection)
    matches = (
        AssetMatch("obs-1", MatchStatus.MATCHED, "P01", 0.9, ("P01",), "CAD-001"),
        AssetMatch("obs-2", MatchStatus.AMBIGUOUS, candidate_asset_ids=("P01", "P02")),
    )
    assert write_asset_matches(connection, matches) == 2
    rows = connection.execute(
        "SELECT observation_id, match_status, asset_id, candidate_asset_ids_json, evidence "
        "FROM asset_matches ORDER BY observation_id"
    ).fetchall()
    assert rows[0][:3] == ("obs-1", "matched", "P01")
    assert json.loads(rows[1][3]) == ["P01", "P02"]
    assert rows[0][4] == "CAD-001"


def test_finding_review_is_separate_and_updatable():
    connection = duckdb.connect(":memory:")
    initialize_schema(connection)
    connection.execute(
        """
        INSERT INTO findings (
            finding_id, observation_id, anomaly_type, assessment_status,
            delta_t_k, absolute_temperature_c, pixel_polygon_json
        ) VALUES ('run::M004:hot:1', 'obs-1', 'hot', 'review_required', 8.0, 42.0, '[]')
        """
    )

    upsert_finding_review(
        connection,
        finding_id="run::M004:hot:1",
        review_status="not_ok",
        reviewer="Synthetic reviewer",
        note="False positive",
    )
    upsert_finding_review(
        connection,
        finding_id="run::M004:hot:1",
        review_status="ok",
    )
    assert connection.execute(
        "SELECT review_status, reviewer, note FROM finding_reviews"
    ).fetchone() == ("ok", None, None)
    with pytest.raises(ValueError, match="review_status"):
        upsert_finding_review(
            connection,
            finding_id="run::M004:hot:1",
            review_status="maybe",
        )


def test_radiometry_summary_updates_indexed_capture():
    connection = duckdb.connect(":memory:")
    initialize_schema(connection)
    upsert_capture(connection, CaptureRecord("capture-1", "private/T.JPG", "hash-1", "thermal"))
    update_capture_radiometry(
        connection,
        RadiometricFrameResult(
            "private/T.JPG", "hash-1", 640, 512, 327680,
            20.0, 45.0, 27.0, 26.5, 0.95, 16.2, 18.0, 18.0,
        ),
    )
    assert connection.execute(
        "SELECT width_px, temperature_median_c, object_distance_m FROM captures"
    ).fetchone() == (640, 26.5, 16.2)


def test_capture_round_trip_exposes_camera_point_and_footprint():
    connection = duckdb.connect(":memory:")
    initialize_schema(connection)
    record = CaptureRecord(
        capture_id="frame-001",
        source_path="private/thermal-001.jpg",
        source_hash="abc123",
        image_kind="thermal",
        capture_time_utc=dt.datetime(2026, 1, 2, 3, 4, tzinfo=dt.timezone.utc),
        camera_latitude=0.25,
        camera_longitude=0.5,
        temperature_array_uri="private/temperature/frame-001.npy",
        temperature_min_c=20.0,
        temperature_max_c=45.0,
        footprint_geojson={
            "type": "Polygon",
            "coordinates": [[[0.49, 0.24], [0.51, 0.24], [0.51, 0.26], [0.49, 0.24]]],
        },
        footprint_method="synthetic-test",
        footprint_accuracy_m=1.5,
    )
    upsert_capture(connection, record)

    rows = map_features(connection)
    assert len(rows) == 1
    assert rows[0]["capture_id"] == "frame-001"
    assert rows[0]["temperature_max_c"] == 45.0
    assert rows[0]["footprint_geojson"]["type"] == "Polygon"


def test_upsert_replaces_mutable_capture_values():
    connection = duckdb.connect(":memory:")
    initialize_schema(connection)
    base = CaptureRecord("frame-001", "private/a.jpg", "hash", "thermal")
    upsert_capture(connection, base)
    upsert_capture(
        connection,
        CaptureRecord(
            "frame-001", "private/a.jpg", "hash", "thermal", temperature_max_c=51.0,
        ),
    )
    assert connection.execute(
        "SELECT count(*), max(temperature_max_c) FROM captures"
    ).fetchone() == (1, 51.0)


def test_capture_rejects_partial_coordinates():
    with pytest.raises(ValueError, match="provided together"):
        CaptureRecord("frame-001", "private/a.jpg", "hash", "thermal", camera_latitude=1)


def test_capture_rejects_footprint_without_method_and_accuracy():
    polygon = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]],
    }
    with pytest.raises(ValueError, match="footprint_method"):
        CaptureRecord(
            "frame-001", "private/a.jpg", "hash", "thermal", footprint_geojson=polygon
        )
    with pytest.raises(ValueError, match="footprint_accuracy_m"):
        CaptureRecord(
            "frame-001",
            "private/a.jpg",
            "hash",
            "thermal",
            footprint_geojson=polygon,
            footprint_method="synthetic-test",
        )


def test_reviewed_footprint_update_preserves_capture_metadata():
    connection = duckdb.connect(":memory:")
    initialize_schema(connection)
    upsert_capture(
        connection,
        CaptureRecord(
            "frame-001",
            "private/a.jpg",
            "hash",
            "thermal",
            camera_latitude=40.0,
            camera_longitude=29.0,
        ),
    )
    polygon = {
        "type": "Polygon",
        "coordinates": [[[29.0, 40.0], [29.1, 40.0], [29.0, 40.0]]],
    }
    update_capture_footprint(
        connection,
        capture_id="frame-001",
        footprint_geojson=polygon,
        method="reviewed-test",
        horizontal_accuracy_m=1.25,
    )
    row = connection.execute(
        """
        SELECT source_hash, camera_latitude, footprint_method, footprint_accuracy_m
        FROM captures WHERE capture_id = 'frame-001'
        """
    ).fetchone()
    assert row == ("hash", 40.0, "reviewed-test", 1.25)


def test_footprint_update_requires_existing_capture_and_accuracy():
    connection = duckdb.connect(":memory:")
    initialize_schema(connection)
    polygon = {"type": "Polygon", "coordinates": []}
    with pytest.raises(KeyError, match="unknown capture_id"):
        update_capture_footprint(
            connection,
            capture_id="missing",
            footprint_geojson=polygon,
            method="reviewed-test",
            horizontal_accuracy_m=1.0,
        )
    with pytest.raises(ValueError, match="positive finite"):
        update_capture_footprint(
            connection,
            capture_id="missing",
            footprint_geojson=polygon,
            method="reviewed-test",
            horizontal_accuracy_m=float("nan"),
        )


def test_map_hides_legacy_footprint_without_accuracy_metadata():
    connection = duckdb.connect(":memory:")
    initialize_schema(connection)
    upsert_capture(
        connection,
        CaptureRecord(
            "frame-001",
            "private/a.jpg",
            "hash",
            "thermal",
            camera_latitude=40.0,
            camera_longitude=29.0,
        ),
    )
    connection.execute(
        """
        UPDATE captures
        SET footprint_geojson = '{"type":"Polygon","coordinates":[]}'
        WHERE capture_id = 'frame-001'
        """
    )
    feature = map_features(connection)[0]
    assert feature["footprint_geojson"] is None


def test_stage1_results_are_idempotent_and_keep_sparse_evidence():
    connection = duckdb.connect(":memory:")
    initialize_schema(connection)
    upsert_run(
        connection,
        run_id="analysis-1",
        site_label="synthetic",
        started_at_utc=None,
        camera_model="test-camera",
        source_root="private",
    )
    upsert_capture(
        connection,
        CaptureRecord("capture-1", "private/T.JPG", "hash", "thermal"),
    )
    module = ModuleObservation(
        "M01",
        "frame-1",
        [(2, 2), (12, 2), (12, 16), (2, 16)],
        temp_min=20,
        temp_max=35,
        temp_mean=25,
        temp_median=24,
    )
    finding = Finding(
        "M01:hot:1",
        "M01",
        "frame-1",
        "hot_region",
        5.0,
        31.0,
        [(5, 6), (7, 6), (7, 8), (5, 8)],
        assessment_status="review_required",
        reference_t_c=26,
        reference_method="module_plane",
        n_region_px=3,
        n_reference_px=50,
        region_pixels_yx=[(6, 5), (6, 6), (7, 6)],
        region_max_t_c=32,
        region_p95_t_c=31.9,
        region_max_residual_k=6,
        delta_method="median_of_pixel_residuals",
    )
    summary = write_stage1_results(
        connection,
        analysis_run_id="analysis-1",
        capture_id="capture-1",
        module_results=(ModuleReport(module, (finding,), reasons=("boundary",)),),
    )
    assert (summary.module_observations, summary.findings, summary.review_required_findings) == (
        1,
        1,
        1,
    )
    row = connection.execute(
        """
        SELECT assessment_status, reference_method, region_pixels_yx_json
        FROM findings
        """
    ).fetchone()
    assert row[:2] == ("review_required", "module_plane")
    assert json.loads(row[2]) == [[6, 5], [6, 6], [7, 6]]

    # A rerun replaces this run/capture slice, so removed findings cannot remain stale.
    write_stage1_results(
        connection,
        analysis_run_id="analysis-1",
        capture_id="capture-1",
        module_results=(ModuleReport(module),),
    )
    assert connection.execute("SELECT count(*) FROM module_observations").fetchone()[0] == 1
    assert connection.execute("SELECT count(*) FROM findings").fetchone()[0] == 0


def test_stage1_results_require_known_run_and_capture():
    connection = duckdb.connect(":memory:")
    initialize_schema(connection)
    with pytest.raises(KeyError, match="analysis_run_id"):
        write_stage1_results(
            connection,
            analysis_run_id="missing",
            capture_id="missing",
            module_results=(),
        )
