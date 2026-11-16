import hashlib

import numpy as np
import pytest
from PIL import Image

from pvdelta.rgb_preview import quadratic_points, rgb_finding_preview


def test_original_rgb_keeps_source_pixels_and_rejects_changed_source(tmp_path):
    from pvdelta.rgb_preview import original_rgb_image

    path = tmp_path/'original.png'
    pixels = np.arange(18*24*3, dtype=np.uint8).reshape(18, 24, 3)
    Image.fromarray(pixels).save(path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    image = original_rgb_image(path, expected_source_hash=expected)
    np.testing.assert_array_equal(np.asarray(image), pixels)
    image.putpixel((0, 0), (255, 0, 0))
    np.testing.assert_array_equal(np.asarray(original_rgb_image(path, expected_source_hash=expected)), pixels)
    Image.new('RGB',(24,18),'white').save(path)
    with pytest.raises(ValueError, match='indexed source'):
        original_rgb_image(path, expected_source_hash=expected)


@pytest.fixture
def pair(tmp_path):
    rgb = tmp_path / 'frame_V.JPG'
    Image.new('RGB', (4032, 3024), '#303030').save(rgb)
    (tmp_path / 'calibration_T.JPG').write_bytes(b'calibration placeholder')
    registration = {'calibration_frame': 'calibration_T.JPG', 'model': {
        'centre': [0, 0], 'scale': [1, 1], 'coefficients': [[1000, 700], [3, 0],
            [0, 3], [0, 0], [0, 0], [0, 0]]}}
    settings = {'registration': registration, 'camera_model': 'M4T', 'native_size': (640, 512),
        'source_root': tmp_path, 'expected_rgb_hash': hashlib.sha256(rgb.read_bytes()).hexdigest(),
        'native_verified': True}
    return rgb, settings


def test_marks_mathematically_mapped_region_without_changing_source_or_claiming_validation(pair):
    path, settings = pair
    polygon = np.array([[300., 200.], [304., 200.], [304., 204.], [300., 204.]])
    original = polygon.copy()
    before = path.read_bytes()
    preview = rgb_finding_preview(path, polygon, **settings)
    assert preview.overlay_applied and preview.registration_status == 'approximate'
    assert 'unverified' in preview.message and '6×' in preview.message
    pixels = np.asarray(preview.image)
    assert ((pixels[:, :, 0] == 255) & (pixels[:, :, 1] == 32) & (pixels[:, :, 2] == 32)).any()
    assert preview.image.width >= 1000 and preview.image.height >= 1000
    assert path.read_bytes() == before
    np.testing.assert_array_equal(polygon, original)
    np.testing.assert_allclose(quadratic_points([[302, 202]], settings['registration']['model']), [[1906, 1306]])


@pytest.mark.parametrize('overrides', [
    {'camera_model': 'M30T'}, {'native_size': (1280, 1024)}, {'source_root': None},
    {'native_verified': False}, {'expected_rgb_hash': 'wrong'}, {'registration': None},
])
def test_incompatible_or_unverified_source_stays_unmarked(pair, overrides):
    path, settings = pair
    preview = rgb_finding_preview(path, [[300, 200], [304, 200], [304, 204], [300, 204]],
        **(settings | overrides))
    assert not preview.overlay_applied and preview.registration_status == 'unavailable'
    assert 'Unmarked RGB' in preview.message
    np.testing.assert_array_equal(np.asarray(preview.image), np.asarray(Image.open(path).convert('RGB')))


def test_wrong_flight_cannot_borrow_this_flights_model(pair):
    path, settings = pair
    settings['registration']['calibration_frame'] = 'unrelated_T.JPG'
    preview = rgb_finding_preview(path, [[1, 1], [4, 1], [4, 4], [1, 4]], **settings)
    assert not preview.overlay_applied and 'flight folder' in preview.message


def test_outside_projection_is_not_clipped_into_a_false_correspondence(pair):
    path, settings = pair
    settings['registration']['model']['coefficients'][0] = [10000, 10000]
    preview = rgb_finding_preview(path, [[1, 1], [4, 1], [4, 4], [1, 4]], **settings)
    assert not preview.overlay_applied and 'outside' in preview.message


def test_invalid_native_coordinates_or_model_fail(pair):
    path, settings = pair
    with pytest.raises(ValueError):
        rgb_finding_preview(path, [[-1, 1], [4, 1], [4, 4], [1, 4]], **settings)
    settings['registration']['model']['coefficients'][0] = [float('nan'), 0]
    with pytest.raises(ValueError):
        rgb_finding_preview(path, [[1, 1], [4, 1], [4, 4], [1, 4]], **settings)
