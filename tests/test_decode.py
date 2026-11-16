"""Decode tests run only when the DJI Thermal SDK and a sample frame are available locally.

    PVDELTA_DIRP_LIB=/path/to/sdk/lib  PVDELTA_SAMPLE_RJPEG=/path/to/DJI_0001_T.JPG  pytest
"""

import os

import numpy as np
import pytest

lib = os.environ.get("PVDELTA_DIRP_LIB")
sample = os.environ.get("PVDELTA_SAMPLE_RJPEG")

pytestmark = [
    pytest.mark.sdk,
    pytest.mark.skipif(not (lib and sample), reason="DJI Thermal SDK or sample not set"),
]


def test_decode_native_grid():
    from pvdelta.decode import RJpeg

    with RJpeg(sample) as rj:
        h, w = rj.resolution
        t = rj.temperature()
        p = rj.measurement_params
    assert (h, w) == (512, 640)
    assert t.dtype == np.float32 and t.shape == (h, w)
    assert -40 < float(np.nanmedian(t)) < 150
    assert 0.1 <= p.emissivity <= 1.0


def test_sdk_processed_rgb_uses_the_native_grid():
    from pvdelta.decode import RJpeg

    with RJpeg(sample) as rj:
        h, w = rj.resolution
        image = rj.processed_rgb()
    assert image.shape == (h, w, 3)
    assert image.dtype == np.uint8
    assert image.max() > image.min()


def test_close_is_idempotent_and_closed_handle_is_rejected():
    from pvdelta.decode import DirpError, RJpeg

    image = RJpeg(sample)
    image.close()
    image.close()
    with pytest.raises(DirpError, match="closed"):
        _ = image.resolution


def test_corrupt_rjpeg_returns_sdk_error(tmp_path):
    from pvdelta.decode import DirpError, RJpeg

    corrupt = tmp_path / "corrupt_T.JPG"
    corrupt.write_bytes(b"not a radiometric jpeg")
    with pytest.raises(DirpError, match="dirp_create_from_rjpeg"):
        RJpeg(corrupt)


def test_explicit_sdk_correction_uses_supported_ranges_and_preserves_source_bytes():
    import hashlib
    from dataclasses import replace
    from pathlib import Path

    from pvdelta.decode import RJpeg

    before = hashlib.sha256(Path(sample).read_bytes()).hexdigest()
    with RJpeg(sample) as image:
        recorded = image.measurement_params
        original = image.temperature()
        low, high = image.measurement_param_ranges['distance_m']
        target = min(high, max(low, recorded.distance_m+10))
        if target == recorded.distance_m:
            target = low
        requested = replace(recorded, distance_m=target)
        effective = image.set_measurement_params(requested)
        corrected = image.temperature()
        assert effective.distance_m == pytest.approx(target)
        assert corrected.shape == original.shape and corrected.dtype == np.float32
        assert np.isfinite(corrected).all()
        assert not np.array_equal(original, corrected)
        with pytest.raises(ValueError, match='SDK range'):
            image.set_measurement_params(replace(recorded, distance_m=high+1))
        with pytest.raises(ValueError, match='SDK range'):
            image.set_measurement_params(replace(recorded, emissivity=float('nan')))
        image.set_measurement_params(recorded)
        np.testing.assert_array_equal(original, image.temperature())
    with RJpeg(sample) as image:
        assert image.measurement_params == recorded
        np.testing.assert_array_equal(original, image.temperature())
    assert hashlib.sha256(Path(sample).read_bytes()).hexdigest() == before
