import json

import numpy as np
import pytest

from pvdelta.store import connect_store, upsert_run, write_view_rankings
from pvdelta.view_selection import ViewCandidate, rank_view_candidates, score_rgb_array


def _checker(size=64):
    values = (np.indices((size, size)).sum(axis=0) % 2) * 180 + 40
    return np.repeat(values[..., None], 3, axis=2).astype(np.uint8)


def test_score_prefers_detail_and_penalizes_clipping():
    sharp = score_rgb_array(_checker())
    blurred = score_rgb_array(np.full((64, 64, 3), 128, dtype=np.uint8))
    clipped = score_rgb_array(np.full((64, 64, 3), 255, dtype=np.uint8))
    assert sharp["ranking_score"] > blurred["ranking_score"]
    assert clipped["clipped_fraction"] == 1.0
    assert sharp["clipped_fraction"] == 0.0


def test_ranking_is_grouped_and_ties_are_deterministic():
    images = {
        "a": _checker(),
        "b": np.full((64, 64, 3), 128, dtype=np.uint8),
        "c": _checker(),
    }
    candidates = [
        ViewCandidate("table-A-row-A-col-1", "b", "b", thermal_capture_id="tb"),
        ViewCandidate("table-A-row-A-col-1", "a", "a", thermal_capture_id="ta"),
        ViewCandidate("table-A-row-A-col-2", "c", "c"),
    ]
    result = rank_view_candidates(candidates, lambda path: images[path])
    assert [item.panel_key for item in result] == [
        "table-A-row-A-col-1", "table-A-row-A-col-1", "table-A-row-A-col-2"
    ]
    assert result[0].rgb_capture_id == "a"
    assert result[0].selection_status == "selected"
    assert result[1].selection_status == "candidate"
    assert result[2].rank == 1


def test_candidate_rejects_partial_coordinates():
    with pytest.raises(ValueError, match="provided together"):
        ViewCandidate("p", "rgb", "frame.jpg", latitude=1.0)


def test_rankings_write_to_local_store(tmp_path):
    pytest.importorskip("duckdb")
    connection = connect_store(tmp_path / "views.duckdb")
    try:
        upsert_run(connection, run_id="views-1", site_label=None, started_at_utc=None,
                   camera_model=None, source_root="local")
        images = {"a": _checker(), "b": np.full((64, 64, 3), 128, dtype=np.uint8)}
        rankings = rank_view_candidates([
            ViewCandidate("p", "a", "a"), ViewCandidate("p", "b", "b")
        ], lambda path: images[path])
        assert write_view_rankings(connection, run_id="views-1", rankings=rankings) == 2
        rows = connection.execute(
            "SELECT panel_key, rgb_capture_id, view_rank, selection_status FROM view_rankings ORDER BY view_rank"
        ).fetchall()
        assert rows[0][0:3] == ("p", "a", 1)
        assert rows[0][3] == "selected"
    finally:
        connection.close()


def test_rank_views_cli_writes_relative_report(monkeypatch, tmp_path, capsys):
    from pvdelta import cli

    manifest = tmp_path / "views.json"
    manifest.write_text(json.dumps({
        "schema_version": "pvdelta.views/v1",
        "run_id": "views-cli",
        "groups": [{"panel_key": "p1", "views": [
            {"rgb_capture_id": "b", "rgb_path": "b.jpg"},
            {"rgb_capture_id": "a", "rgb_path": "a.jpg"},
        ]}],
    }), encoding="utf-8")
    images = {"a.jpg": _checker(), "b.jpg": np.full((64, 64, 3), 128, dtype=np.uint8)}
    monkeypatch.setattr(cli, "load_rgb_file", lambda path: images[path.name])
    output = tmp_path / "out" / "rankings.json"
    assert cli.main(["rank-views", str(manifest), "--out", str(output)]) == 0
    capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["groups"] == 1
    assert payload["views"][0]["rgb_capture_id"] == "a"
    assert not str(tmp_path) in output.read_text(encoding="utf-8")
