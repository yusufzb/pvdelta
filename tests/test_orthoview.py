import numpy as np

from pvdelta.orthoview import _local_enu, _thermal_display


def test_local_enu_is_zero_at_origin_and_scales_longitude_by_latitude():
    assert _local_enu(40.0, 29.0, 40.0, 29.0) == (0.0, 0.0)
    east, north = _local_enu(40.001, 29.001, 40.0, 29.0)
    assert east > 0
    assert north > 0
    assert east < 100
    assert north < 120


def test_thermal_display_keeps_invalid_pixels_black():
    temperature = np.asarray([[20.0, 30.0], [np.nan, 40.0]], dtype=np.float32)
    display = _thermal_display(temperature, 20.0, 40.0)
    assert tuple(display[0, 0]) == (0, 0, 255)
    assert tuple(display[1, 1]) == (255, 255, 0)
    assert tuple(display[1, 0]) == (0, 0, 0)
