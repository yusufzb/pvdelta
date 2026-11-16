import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from pvdelta.odm import file_hash, native_environment, prepare_project, run_native


def database(tmp_path, outside=False):
    root = tmp_path / "source"
    root.mkdir()
    db = tmp_path / "input.duckdb"
    with duckdb.connect(str(db)) as c:
        c.execute("CREATE TABLE inspection_runs(run_id VARCHAR, source_root VARCHAR)")
        c.execute("INSERT INTO inspection_runs VALUES ('flight', ?)", [str(root)])
        c.execute("""CREATE TABLE captures(capture_id VARCHAR, source_path VARCHAR,
                  paired_capture_id VARCHAR, gimbal_pitch_deg DOUBLE, run_id VARCHAR,
                  image_kind VARCHAR, capture_time_utc INTEGER, source_hash VARCHAR)""")
        for i in range(7):
            (root / f"{i}.JPG").write_bytes(f"source-exif-{i}".encode())
            c.execute("INSERT INTO captures VALUES (?,?,?,?,?,?,?,?)",
                      [f"rgb{i}", f"../{i}.JPG" if outside else f"{i}.JPG", f"thermal{i}",
                       -90 if i else -30, "flight", "rgb", i, file_hash(root / f"{i}.JPG")])
    return db, root


def test_pilot_copies_preserve_pairing_and_hashes_without_modifying_sources(tmp_path):
    db, root = database(tmp_path)
    job = tmp_path / "pilot"
    state = prepare_project(db, "flight", job, limit=3)
    assert state["selected_count"] == 3
    assert state["indexed_rgb_count"] == 7
    assert [e["capture_id"] for e in state["images"]] == ["rgb2", "rgb3", "rgb4"]
    for e in state["images"]:
        assert file_hash(job / "images" / e["image"]) == e["sha256"]
        assert e["thermal_capture_id"] == e["capture_id"].replace("rgb", "thermal")
    (job / "images" / state["images"][0]["image"]).write_bytes(b"changed")
    assert (root / "2.JPG").read_bytes() == b"source-exif-2"
    with pytest.raises(FileExistsError):
        prepare_project(db, "flight", job, limit=3)


def test_prepare_rejects_path_escape_before_writing_images(tmp_path):
    db, _ = database(tmp_path, outside=True)
    with pytest.raises(ValueError, match="out-of-root"):
        prepare_project(db, "flight", tmp_path / "pilot", 3)
    assert not (tmp_path / "pilot").exists()


def test_failed_runtime_does_not_become_success_or_leave_lock(tmp_path, monkeypatch):
    db, _ = database(tmp_path)
    job = tmp_path / "pilot"
    prepare_project(db, "flight", job, 3)
    home = tmp_path / "runtime"
    (home / "venv/Scripts").mkdir(parents=True)
    (home / "venv/Scripts/python.exe").touch()
    (home / "run.py").touch()
    (home / "VERSION").write_text("test")
    class Failure:
        returncode = 1
    monkeypatch.setattr("pvdelta.odm.subprocess.run", lambda *a, **kw: Failure())
    with pytest.raises(RuntimeError, match="exited with 1"):
        run_native(job, home)
    assert json.loads((job / "pvdelta-odm.json").read_text())["status"] == "failed"
    assert not (job / ".pvdelta-running").exists()


def test_duplicate_worker_cannot_change_running_status(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / ".pvdelta-running").write_text("another worker")
    home = tmp_path / "runtime"
    (home / "venv/Scripts").mkdir(parents=True)
    (home / "venv/Scripts/python.exe").touch()
    (home / "run.py").touch()
    with pytest.raises(FileExistsError):
        run_native(job, home)
    assert (job / ".pvdelta-running").read_text() == "another worker"


def test_environment_handles_unicode_and_removes_foreign_pythonhome(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHOME", "foreign")
    env = native_environment(tmp_path / "Masaüstü", tmp_path / "pilot")
    assert "PYTHONHOME" not in env
    assert Path(env["GDAL_DATA"]).is_relative_to(tmp_path)


def test_altered_input_is_rejected_before_starting_odm(tmp_path, monkeypatch):
    db, _ = database(tmp_path)
    job = tmp_path / "pilot"
    state = prepare_project(db, "flight", job, 3)
    (job / "images" / state["images"][0]["image"]).write_bytes(b"tampered")
    home = tmp_path / "runtime"
    (home / "venv/Scripts").mkdir(parents=True)
    (home / "venv/Scripts/python.exe").touch()
    (home / "run.py").touch()
    def never_run(*args, **kwargs):
        pytest.fail("ODM must not run on changed inputs")
    monkeypatch.setattr("pvdelta.odm.subprocess.run", never_run)
    with pytest.raises(ValueError, match="Input changed"):
        run_native(job, home)
    assert json.loads((job / "pvdelta-odm.json").read_text())["status"] == "failed"


def test_relaunch_does_not_erase_completed_status(tmp_path):
    from pvdelta.odm import write_status

    db, _ = database(tmp_path)
    job = tmp_path / "pilot"
    prepare_project(db, "flight", job, 3)
    write_status(job, status="needs_review")
    home = tmp_path / "runtime"
    (home / "venv/Scripts").mkdir(parents=True)
    (home / "venv/Scripts/python.exe").touch()
    (home / "run.py").touch()
    with pytest.raises(ValueError, match="Only a prepared"):
        run_native(job, home)
    assert json.loads((job / "pvdelta-odm.json").read_text())["status"] == "needs_review"


def test_stale_index_cannot_bind_new_bytes_to_old_capture_id(tmp_path):
    db, root = database(tmp_path)
    (root / "2.JPG").write_bytes(b"replaced after indexing")
    with pytest.raises(ValueError, match="database hash"):
        prepare_project(db, "flight", tmp_path / "pilot", 3)
    state = json.loads((tmp_path / "pilot/pvdelta-odm.json").read_text())
    assert state["status"] == "preparation_failed"


def test_zero_exit_without_readable_geotiff_is_failure(tmp_path, monkeypatch):
    import subprocess

    db, _ = database(tmp_path)
    job = tmp_path / "pilot"
    prepare_project(db, "flight", job, 3)
    home = tmp_path / "runtime"
    (home / "venv/Scripts").mkdir(parents=True)
    (home / "venv/Scripts/python.exe").touch()
    (home / "run.py").touch()
    (home / "VERSION").write_text("test")
    def execute(command, **kwargs):
        if "-c" in command:
            raise subprocess.CalledProcessError(1, command, stderr="Missing GeoTIFF")
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr("pvdelta.odm.subprocess.run", execute)
    with pytest.raises(subprocess.CalledProcessError):
        run_native(job, home)
    assert json.loads((job / "pvdelta-odm.json").read_text())["status"] == "failed"
