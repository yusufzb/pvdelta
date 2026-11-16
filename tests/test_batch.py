import json

import numpy as np
import pytest

from pvdelta.batch import run_batch
from pvdelta.models import ThermalFrame


def _write_regions(path):
    path.write_text(
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


def _decoder(path, sdk_lib):
    if not path.is_file():
        raise OSError(f"missing test capture: {path.name}")
    temperature = np.full((20, 20), 25.0, dtype=np.float32)
    temperature[8:12, 8:12] = 31.0
    return ThermalFrame(path.stem, str(path), path.stem, temperature, None, 20, 20)


def test_explicit_per_frame_sdk_profile_and_source_hash_are_not_silently_ignored(tmp_path):
    (tmp_path/'first_T.JPG').write_bytes(b'first')
    (tmp_path/'second_T.JPG').write_bytes(b'second')
    _write_regions(tmp_path/'regions.json')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'schema_version': 'pvdelta.batch/v1', 'run_id': 'profiles',
        'frames': [{'input': 'first_T.JPG', 'regions': 'regions.json', 'source_sha256': 'first_T',
                    'measurement_overrides': {'distance_m': 19.}},
                   {'input': 'second_T.JPG', 'regions': 'regions.json', 'source_sha256': 'wrong'}]}))
    calls = []

    def decode(path, sdk, *, measurement_overrides=None):
        calls.append((path.name, measurement_overrides))
        return _decoder(path, sdk)

    result = run_batch(manifest, tmp_path/'out', decoder=decode)
    assert calls == [('first_T.JPG', {'distance_m': 19.}), ('second_T.JPG', None)]
    assert result.status == 'partial' and result.failed_frames == 1
    assert 'source hash differs' in result.frames[1].error


def test_batch_continues_after_frame_failure_and_writes_partial_summary(tmp_path):
    (tmp_path / "valid_T.JPG").write_bytes(b"synthetic")
    _write_regions(tmp_path / "regions.json")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "pvdelta.batch/v1",
                "run_id": "batch-1",
                "defaults": {
                    "direction": "hot",
                    "detection": {"detrend_gradient": False},
                },
                "frames": [
                    {"input": "valid_T.JPG", "regions": "regions.json"},
                    {"input": "missing_T.JPG", "regions": "regions.json"},
                ],
            }
        ),
        encoding="utf-8",
    )
    result = run_batch(manifest, tmp_path / "out", decoder=_decoder)
    assert result.status == "partial"
    assert (result.succeeded_frames, result.failed_frames) == (1, 1)
    assert result.frames[0].findings == 1
    assert result.frames[1].source_name == "missing_T.JPG"
    assert "missing test capture" in result.frames[1].error
    assert str(tmp_path) not in result.frames[1].error
    payload = json.loads(result.summary_json.read_text(encoding="utf-8"))
    assert payload["status"] == "partial"
    assert (tmp_path / "out" / "frames" / "0001-valid_T" / "batch-1.csv").is_file()


def test_batch_refuses_implicit_summary_overwrite(tmp_path):
    (tmp_path / "valid_T.JPG").write_bytes(b"synthetic")
    _write_regions(tmp_path / "regions.json")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "pvdelta.batch/v1",
                "run_id": "batch-1",
                "frames": [{"input": "valid_T.JPG", "regions": "regions.json"}],
            }
        ),
        encoding="utf-8",
    )
    run_batch(manifest, tmp_path / "out", decoder=_decoder)
    with pytest.raises(FileExistsError, match="batch output exists"):
        run_batch(manifest, tmp_path / "out", decoder=_decoder)
