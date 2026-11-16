import json
from pathlib import Path

import numpy as np
import pytest

from pvdelta.batch import run_batch
from pvdelta.cli import main
from pvdelta.manifest import prepare_manifest
from pvdelta.models import ThermalFrame


def _regions(path, *, label=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": "1.0", "coordinate_space": "thermal_native",
        "modules": [{"module_id": "M1", "polygon": [[1, 1], [19, 1], [19, 19], [1, 19]],
                     "mask_label": label}],
    }), encoding="utf-8")


def _folders(tmp_path):
    flight, regions = tmp_path / "flight", tmp_path / "regions"
    flight.mkdir()
    regions.mkdir()
    return flight, regions, tmp_path / "reports" / "manifest.json"


def test_mirrored_matching_no_basename_fallback_and_relative_paths(tmp_path):
    flight, regions, out = _folders(tmp_path)
    for folder in ("a", "b"):
        (flight / folder).mkdir()
        (flight / folder / "same_T.jPeG").touch()
    (flight / "rgb_W.JPG").touch()
    _regions(regions / "a" / "same_T.regions.json")
    result = prepare_manifest(flight, regions, out, run_id="test")
    assert result["preparation"] == {
        "status": "partial", "discovered_frames": 2, "matched_frames": 1,
        "unresolved_frames": 1,
        "unresolved": [{"input": "b/same_T.jPeG", "reason": "missing_regions"}],
    }
    assert len(result["frames"]) == 1
    for value in result["frames"][0].values():
        assert not Path(value).is_absolute()
        assert (out.parent / value).is_file()
    assert str(tmp_path) not in out.read_text(encoding="utf-8")


def test_explicit_mask_directory_never_falls_back_to_polygon(tmp_path):
    flight, regions, out = _folders(tmp_path)
    (flight / "one_T.JPG").touch()
    _regions(regions / "one_T.regions.json", label=1)
    masks = tmp_path / "masks"
    masks.mkdir()
    result = prepare_manifest(flight, regions, out, run_id="test", mask_dir=masks)
    assert result["frames"] == []
    assert result["preparation"]["unresolved"][0]["reason"] == "missing_mask_archive"
    np.savez(masks / "one_T.masks.npz", module_labels=np.ones((20, 20)),
             measurement=np.ones((20, 20), dtype=bool))
    result = prepare_manifest(flight, regions, out, run_id="test", mask_dir=masks, overwrite=True)
    assert result["preparation"]["status"] == "ready"
    assert result["frames"][0]["mask_archive"].endswith("one_T.masks.npz")


@pytest.mark.parametrize("body,reason", [
    ("invalid json", "invalid_regions"),
    ('{"schema_version":"1.0","coordinate_space":"thermal_native","modules":[]}',
     "empty_regions"),
])
def test_invalid_or_empty_annotations_are_unresolved(tmp_path, body, reason):
    flight, regions, out = _folders(tmp_path)
    (flight / "one_T.JPG").touch()
    (regions / "one_T.regions.json").write_text(body, encoding="utf-8")
    result = prepare_manifest(flight, regions, out, run_id="test")
    assert result["preparation"]["unresolved"][0]["reason"] == reason
    assert result["frames"] == []


def test_ambiguous_stem_and_overwrite_protection(tmp_path):
    flight, regions, out = _folders(tmp_path)
    (flight / "one_T.JPG").touch()
    (flight / "one_T.JPEG").touch()
    annotation = regions / "one_T.regions.json"
    _regions(annotation)
    result = prepare_manifest(flight, regions, out, run_id="test")
    assert result["frames"] == []
    assert all(item["reason"] == "ambiguous_capture_stem"
               for item in result["preparation"]["unresolved"])
    with pytest.raises(FileExistsError):
        prepare_manifest(flight, regions, out, run_id="test")
    before = annotation.read_bytes()
    with pytest.raises(ValueError, match="must not replace"):
        prepare_manifest(flight, regions, annotation, run_id="test", overwrite=True)
    assert annotation.read_bytes() == before


def test_cli_generated_manifest_runs_in_batch_without_sdk(tmp_path, capsys):
    flight, regions, out = _folders(tmp_path)
    (flight / "one_T.JPG").touch()
    _regions(regions / "one_T.regions.json")
    args = ["prepare-manifest", str(flight), "--regions-dir", str(regions),
            "--out", str(out), "--run-id", "test"]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"

    def decoder(path, sdk_lib):
        assert path.is_file()
        temperature = np.full((20, 20), 25.0, dtype=np.float32)
        temperature[8:12, 8:12] = 32.0
        return ThermalFrame(path.stem, str(path), "hash", temperature, None, 20, 20)

    batch = run_batch(out, tmp_path / "batch", decoder=decoder)
    assert batch.status == "complete"
    assert batch.frames[0].findings == 1
    (flight / "two_T.JPG").touch()
    assert main(args + ["--overwrite"]) == 3


def test_no_thermal_files_and_missing_directory(tmp_path, capsys):
    flight, regions, out = _folders(tmp_path)
    assert main(["prepare-manifest", str(flight), "--regions-dir", str(regions),
                 "--out", str(out), "--run-id", "test"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    assert json.loads(out.read_text())["frames"] == []
    with pytest.raises(ValueError, match="flight directory"):
        prepare_manifest(flight / "missing", regions, out, run_id="test", overwrite=True)
