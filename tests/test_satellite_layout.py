import numpy as np
import pytest

from pvdelta.satellite_layout import fit_translation, screenshot_to_mercator


def test_screenshot_centre_north_and_pixel_scale():
    centre = [100., 200.]
    pixels = [[400., 300.], [401., 300.], [400., 299.]]
    result = screenshot_to_mercator(pixels, centre, [800, 600], 19)
    np.testing.assert_allclose(result[0], centre)
    resolution = 2*np.pi*6378137/(256*2**19)
    np.testing.assert_allclose(result[1]-result[0], [resolution, 0])
    np.testing.assert_allclose(result[2]-result[0], [0, resolution])


def test_translation_is_recovered_without_rescaling_or_rotation():
    source = np.array([[0, 0], [20, 0], [0, 8], [20, 8], [0, 17], [20, 17]])
    shift, report = fit_translation(source, source+[1.2, -.3], ['a', 'a', 'b', 'b', 'c', 'c'])
    np.testing.assert_allclose(shift, [1.2, -.3])
    assert report['held_out_row_p95'] < 1e-12
    assert report['scale'] == 1
    assert report['rotation_deg'] == 0


def test_bad_row_is_exposed_in_group_holdout():
    source = np.array([[0, 0], [20, 0], [0, 8], [20, 8], [0, 17], [20, 17]])
    target = source + np.array([[1, 0]]*4 + [[15, 0]]*2)
    shift, report = fit_translation(source, target, ['a', 'a', 'b', 'b', 'c', 'c'])
    np.testing.assert_allclose(shift, [1, 0])
    assert report['held_out_row_p95'] > 10


def test_repeated_features_do_not_overweight_one_row():
    source = np.zeros((102, 2))
    target = np.array([[20, 0]]*100+[[1, 0], [2, 0]])
    shift, _ = fit_translation(source, target, ['a']*100+['b', 'c'])
    np.testing.assert_allclose(shift, [2, 0])


def test_too_few_rows_and_invalid_coordinates_rejected():
    with pytest.raises(ValueError):
        fit_translation([[0, 0]]*8, [[1, 0]]*8, ['a']*4+['b']*4)
    with pytest.raises(ValueError):
        fit_translation([[0, 0], [0, 1], [np.nan, 2]], [[1, 0]]*3, ['a', 'b', 'c'])
