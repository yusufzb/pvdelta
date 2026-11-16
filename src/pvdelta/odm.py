"""Local ODM RGB jobs. Native radiometric observations remain the measurement source."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def write_status(project: Path, **values) -> dict:
    path = project / "pvdelta-odm.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    state.update(values, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)
    return state


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_project(database: Path, run_id: str, project: Path, limit: int = 80) -> dict:
    """Copy an ordered central pilot, or all RGBs with limit=0; preserve EXIF and provenance."""
    import duckdb

    if limit < 0 or (0 < limit < 3):
        raise ValueError("limit must be zero (full flight) or at least three")
    project = project.resolve()
    if project.exists():
        raise FileExistsError(f"Refusing to mix inputs or overwrite results: {project}")
    with duckdb.connect(str(database), read_only=True) as connection:
        root_row = connection.execute(
            "SELECT source_root FROM inspection_runs WHERE run_id=?", [run_id]
        ).fetchone()
        if root_row is None:
            raise ValueError(f"Unknown capture run: {run_id}")
        root = Path(root_row[0]).resolve()
        rows = connection.execute(
            """SELECT capture_id, source_path, paired_capture_id, gimbal_pitch_deg, source_hash
               FROM captures WHERE run_id=? AND image_kind='rgb'
               ORDER BY capture_time_utc, capture_id""", [run_id]
        ).fetchall()
    eligible = [r for r in rows if r[3] is not None and -95 <= r[3] <= -80]
    if limit:
        start = max(0, (len(eligible) - limit) // 2)
        selected = eligible[start:start + limit]
    else:
        selected = rows
    if len(selected) < 3:
        raise ValueError("Fewer than three eligible RGB frames")
    sources = []
    for capture_id, relative, thermal_id, pitch, indexed_hash in selected:
        source = (root / relative).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"Missing or out-of-root source: {relative}")
        sources.append((capture_id, source, thermal_id, pitch, indexed_hash))
    required = sum(s.stat().st_size for _, s, _, _, _ in sources)
    parent = project.parent
    parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(parent).free < required + 10 * 1024**3:
        raise ValueError("Insufficient free space for copies plus 10 GiB processing reserve")
    (project / "images").mkdir(parents=True)
    write_status(project, schema="pvdelta.odm/v1", status="preparing", run_id=run_id,
                 scope="pilot" if limit else "full", indexed_rgb_count=len(rows),
                 selected_count=len(sources), thermal_status="not_projected")
    entries = []
    try:
        for index, (capture_id, source, thermal_id, pitch, indexed_hash) in enumerate(sources):
            # Independent copies: ODM must never modify private source photographs.
            name = f"{index:05d}_{source.name}"
            target = project / "images" / name
            shutil.copy2(source, target)
            digest = file_hash(source)
            if not isinstance(indexed_hash, str) or digest != indexed_hash.removeprefix("sha256:"):
                raise ValueError(f"Source no longer matches database hash: {source.name}")
            if file_hash(target) != digest:
                raise ValueError(f"Copy integrity failed: {source.name}")
            entries.append({"image": name, "capture_id": capture_id, "source": str(source),
                            "thermal_capture_id": thermal_id, "sha256": digest, "pitch_deg": pitch})
        return write_status(project, status="prepared", images=entries,
                            copied_bytes=required, validation="not_run")
    except Exception as exc:
        write_status(project, status="preparation_failed", error=str(exc))
        raise


def native_environment(home: Path, project: Path) -> dict[str, str]:
    """Mirror ODM 3.6.2 win32env.bat without invoking a shell."""
    home = native_path(home)
    binary = home / "SuperBuild/install/bin"
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.update(ODMBASE=str(home), VIRTUAL_ENV=str(home / "venv"),
               PYTHONPATH=str(home / "venv"),
               PYTHONPYCACHEPREFIX=str(project / ".pycache"),
               GDAL_DATA=str(binary / "data/gdal"), GDAL_DRIVER_PATH=str(binary / "gdalplugins"),
               PROJ_LIB=str(binary / "data/proj"), PDAL_DRIVER_PATH=str(binary),
               PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    env["PATH"] = os.pathsep.join([str(home / "venv/Scripts"), str(binary),
                                  str(binary / "opensfm/bin"), env.get("PATH", "")])
    return env


def native_path(path: Path) -> Path:
    """Use existing Windows short paths for ODM C++ libraries with narrow-path I/O."""
    absolute = path.absolute()
    if os.name != "nt" or str(absolute).isascii() or not absolute.exists():
        return absolute
    import ctypes

    buffer = ctypes.create_unicode_buffer(32768)
    size = ctypes.windll.kernel32.GetShortPathNameW(str(absolute), buffer, len(buffer))
    if not size or size >= len(buffer) or not buffer.value.isascii():
        raise ValueError("ODM native libraries need an ASCII project/runtime path on this volume")
    return Path(buffer.value)


def run_native(project: Path, home: Path, concurrency: int = 4) -> dict:
    """Run an isolated RGB reconstruction; exit 0 still requires geometry review."""
    project, home = native_path(project), native_path(home)
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    python = home / "venv/Scripts/python.exe"
    if not python.is_file() or not (home / "run.py").is_file():
        raise FileNotFoundError(f"Native ODM runtime unavailable: {home}")
    lock = project / ".pvdelta-running"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    may_update_status = False
    try:
        state = json.loads((project / "pvdelta-odm.json").read_text(encoding="utf-8"))
        if state["status"] != "prepared":
            raise ValueError("Only a prepared, previously unprocessed job may run")
        may_update_status = True
        entries = state["images"]
        actual = {p.name for p in (project / "images").iterdir()}
        if actual != {e["image"] for e in entries}:
            raise ValueError("Input file set changed since preparation")
        for entry in entries:
            if file_hash(project / "images" / entry["image"]) != entry["sha256"]:
                raise ValueError(f"Input changed: {entry['image']}")
        version = (home / "VERSION").read_text().strip()
        command = [str(python), "-X", "utf8", str(home / "run.py"),
                   "--project-path", str(project.parent), project.name,
                   "--feature-quality", "high", "--pc-quality", "medium", "--dsm",
                   "--orthophoto-resolution", "2", "--max-concurrency", str(concurrency)]
        env = native_environment(home, project)
        write_status(project, status="running", pid=os.getpid(), version=version, command=command)
        with (project / "process.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=home, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            raise RuntimeError(f"ODM exited with {completed.returncode}; see process.log")
        # Inspect the actual georeferenced product with ODM's bundled GDAL.
        inspect = '''import json,sys
from osgeo import gdal
p=sys.argv[1]
d=gdal.Open(p+'/odm_orthophoto/odm_orthophoto.tif')
if d is None or not d.GetProjection(): raise ValueError('Missing georeferenced orthophoto')
if d.RasterCount < 3: raise ValueError('RGB bands missing')
gdal.Translate(p+'/rgb_preview.png', d, format='PNG', width=1600, height=0,
               bandList=[1,2,3], outputType=gdal.GDT_Byte)
print(json.dumps(dict(width=d.RasterXSize,height=d.RasterYSize,
                     crs_wkt=d.GetProjection(),transform=d.GetGeoTransform())))
'''
        checked = subprocess.run([str(python), "-X", "utf8", "-c", inspect, str(project)],
                                 env=env, cwd=home, capture_output=True, text=True,
                                 encoding="utf-8", check=True)
        raster = json.loads(checked.stdout)
        reconstruction = json.loads((project / "opensfm/reconstruction.json").read_text())
        shot_counts = [len(c.get("shots", {})) for c in reconstruction]
        return write_status(project, status="needs_review", raster=raster,
                            component_shot_counts=shot_counts,
                            largest_component_fraction=max(shot_counts, default=0)/len(entries),
                            validation="Visual seams and independent alignment checks pending")
    except Exception as exc:
        if may_update_status:
            write_status(project, status="failed", error=str(exc))
        raise
    finally:
        lock.unlink(missing_ok=True)
