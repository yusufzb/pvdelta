import numpy as np

from pvdelta.radiometry import RadiometricFrameResult


def test_radiometric_frame_result_is_compact_and_finite():
    result = RadiometricFrameResult(
        "private/frame_T.JPG", "hash", 640, 512, 327680,
        20.0, 45.0, 27.0, 26.5, 0.95, 16.2, 18.0, 18.0,
    )
    assert result.width_px * result.height_px == result.valid_pixels
    assert np.isfinite(result.temperature_median_c)
