import numpy as np
import pytest

pytest.importorskip("cv2")
from pvdelta.surface_support import check_rgb_surface


def test_rgb_check_distinguishes_panel_vegetation_and_uncertain_edge():
    rgb = np.full((120, 120, 3), [50, 150, 30], dtype=np.uint8)
    rgb[20:100, 20:100] = [70, 80, 95]
    panel = np.array([[20, 20], [100, 20], [100, 100], [20, 100]])
    assert check_rgb_surface(rgb, np.array([[50, 50]]), panel, uncertainty_px=3)['status'] == 'geometry_consistent'
    assert check_rgb_surface(rgb, np.array([[10, 10]]), panel, uncertainty_px=3)['status'] == 'review_geometry'
    assert check_rgb_surface(rgb, np.array([[21, 50]]), panel, uncertainty_px=3)['status'] == 'review_edge'
    rgb[50, 50] = [40, 160, 20]
    assert check_rgb_surface(rgb, np.array([[50, 50]]), panel, uncertainty_px=3)['status'] == 'review_vegetation'


def test_out_of_frame_and_nan_are_not_supported():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    panel = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
    assert check_rgb_surface(rgb, np.array([[-20, 30]]), panel, uncertainty_px=3)['status'] == 'review_geometry'
    with pytest.raises(ValueError):
        check_rgb_surface(rgb, np.array([[np.nan, 30]]), panel, uncertainty_px=3)
