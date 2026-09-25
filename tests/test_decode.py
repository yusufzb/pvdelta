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
