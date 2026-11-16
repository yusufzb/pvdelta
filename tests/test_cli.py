import json

import numpy as np

from pvdelta import cli
from pvdelta.models import ThermalFrame


def test_analyze_cli_writes_report_without_sdk(monkeypatch, tmp_path, capsys):
    source = tmp_path / "synthetic_T.JPG"
    source.write_bytes(b"synthetic")
    regions = tmp_path / "regions.json"
    regions.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "coordinate_space": "thermal_native",
                "modules": [
                    {
                        "module_id": "M01",
                        "polygon": [[2, 2], [18, 2], [18, 18], [2, 18]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    temperature = np.full((20, 20), 25.0, dtype=np.float32)
    temperature[8:12, 8:12] = 31.0
    frame = ThermalFrame("f", str(source), "hash", temperature, None, 20, 20)
    monkeypatch.setattr(cli, "decode_frame", lambda path, sdk_lib: frame)
    output = tmp_path / "out"
    status = cli.main(
        [
            "analyze",
            str(source),
            "--regions",
            str(regions),
            "--out",
            str(output),
            "--run-id",
            "cli-test",
            "--direction",
            "hot",
        ]
    )
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["modules"] == 1
    assert payload["findings"] == 1
    assert (output / "cli-test.jsonl").is_file()
    assert (output / "cli-test.csv").is_file()


def test_cli_refuses_implicit_report_overwrite(monkeypatch, tmp_path, capsys):
    source = tmp_path / "synthetic_T.JPG"
    source.write_bytes(b"synthetic")
    regions = tmp_path / "regions.json"
    regions.write_text(
        '{"schema_version":"1.0","coordinate_space":"thermal_native",'
        '"modules":[{"module_id":"M01","polygon":[[1,1],[9,1],[9,9],[1,9]]}]}',
        encoding="utf-8",
    )
    temperature = np.arange(100, dtype=np.float32).reshape(10, 10)
    frame = ThermalFrame("f", str(source), "hash", temperature, None, 10, 10)
    monkeypatch.setattr(cli, "decode_frame", lambda path, sdk_lib: frame)
    arguments = [
        "analyze",
        str(source),
        "--regions",
        str(regions),
        "--out",
        str(tmp_path / "out"),
        "--run-id",
        "same-run",
    ]
    assert cli.main(arguments) == 0
    capsys.readouterr()
    assert cli.main(arguments) == 1
    assert "report output exists" in capsys.readouterr().err


def test_batch_cli_returns_partial_exit_code(monkeypatch, tmp_path, capsys):
    valid = tmp_path / "valid_T.JPG"
    valid.write_bytes(b"synthetic")
    regions = tmp_path / "regions.json"
    regions.write_text(
        '{"schema_version":"1.0","coordinate_space":"thermal_native",'
        '"modules":[{"module_id":"M01","polygon":[[2,2],[18,2],[18,18],[2,18]]}]}',
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "pvdelta.batch/v1",
                "run_id": "batch-cli",
                "defaults": {
                    "direction": "hot",
                    "detection": {"detrend_gradient": False},
                },
                "frames": [
                    {"input": valid.name, "regions": regions.name},
                    {"input": "missing_T.JPG", "regions": regions.name},
                ],
            }
        ),
        encoding="utf-8",
    )
    temperature = np.full((20, 20), 25.0, dtype=np.float32)
    temperature[8:12, 8:12] = 31.0

    def decode(path, sdk_lib):
        if not path.exists():
            raise OSError("missing")
        return ThermalFrame(path.stem, str(path), path.stem, temperature, None, 20, 20)

    monkeypatch.setattr(cli, "decode_frame", decode)
    status = cli.main(["batch", str(manifest), "--out", str(tmp_path / "out")])
    assert status == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "partial"
    assert payload["succeeded_frames"] == 1
    assert payload["failed_frames"] == 1


def test_import_inventory_cli_loads_csv_into_local_store(tmp_path, capsys):
    inventory = tmp_path / "panel_inventory.csv"
    inventory.write_text(
        "panel_id,table_id,row,position,latitude,longitude,crs\n"
        "P01,T01,A,1,40,26,EPSG:4326\n",
        encoding="utf-8",
    )
    database = tmp_path / "inspection.duckdb"
    assert cli.main(["import-inventory", str(inventory), "--database", str(database)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["assets"] == 1
    import duckdb
    connection = duckdb.connect(str(database), read_only=True)
    try:
        assert connection.execute("SELECT panel_id FROM panel_assets").fetchone() == ("P01",)
    finally:
        connection.close()
