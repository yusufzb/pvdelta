import hashlib
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from pvdelta.panel_preview import rgb_panel_preview, thermal_panel_preview


def regions():
    return [{'label': 'A1', 'polygon': [[14, 14], [17, 14], [17, 17], [14, 17]], 'pixels_yx': [[15, 15]]},
            {'label': 'A2', 'polygon': [[34, 34], [37, 34], [37, 37], [34, 37]], 'pixels_yx': [[35, 35]]}]


def test_all_thermal_regions_share_one_sdk_decode_and_keep_hot_interiors(monkeypatch):
    from pvdelta import decode, measurement_profile

    pixels = np.full((64, 64, 3), 30, np.uint8)
    pixels[15, 15] = [250, 100, 10]
    pixels[35, 35] = [245, 90, 20]
    values = np.full((64, 64), 25., np.float32)
    values.setflags(write=False)
    calls = []

    class Reader:
        def __init__(self, *args):
            calls.append('decode')

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def processed_rgb(self):
            return pixels

    monkeypatch.setattr(decode, 'RJpeg', Reader)
    monkeypatch.setattr(measurement_profile, 'apply_saved_measurement_profile', lambda *args: values)
    original = deepcopy(regions())
    preview = thermal_panel_preview('unused', original, native_size=(64, 64), measurement_profile={},
        module_polygon=[[5, 5], [55, 5], [55, 55], [5, 55]])
    assert calls == ['decode'] and preview.overlay_applied
    assert set(preview.region_states) == {'A1', 'A2'}
    marked = np.asarray(preview.image)
    np.testing.assert_array_equal(marked[15, 15], pixels[15, 15])
    np.testing.assert_array_equal(marked[35, 35], pixels[35, 35])
    assert tuple(marked[14, 15]) == tuple(marked[34, 35]) == (255, 255, 255)
    assert original == regions()
    np.testing.assert_array_equal(values, np.full((64, 64), 25., np.float32))


def test_missing_native_profile_does_not_paint_any_regions(tmp_path):
    path = tmp_path/'thermal.jpg'
    Image.new('RGB', (64, 64), '#303030').save(path)
    preview = thermal_panel_preview(path, regions(), native_size=(64, 64))
    assert not preview.overlay_applied
    assert set(preview.region_states.values()) == {'unavailable'}
    np.testing.assert_array_equal(preview.image, Image.open(path).convert('RGB'))


def test_legacy_source_bound_panel_card_marks_every_region_without_claiming_old_radiometry(tmp_path, monkeypatch):
    from pvdelta import decode

    path = tmp_path/'original_T.JPG'
    Image.new('RGB', (128, 128), '#303030').save(path)
    native = np.full((64, 64, 3), 30, np.uint8)
    native[15, 15] = [250, 100, 10]

    class Reader:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def temperature(self):
            pytest.fail('A legacy location preview cannot certify historical temperatures')

        def processed_rgb(self):
            return native.copy()

    monkeypatch.setattr(decode, 'RJpeg', Reader)
    preview = thermal_panel_preview(path, regions(), native_size=(64, 64),
        expected_source_hash=hashlib.sha256(path.read_bytes()).hexdigest())
    assert preview.overlay_applied and not preview.radiometry_verified
    assert preview.region_states == {'A1': 'legacy_location', 'A2': 'legacy_location'}
    assert preview.image.size == (64, 64)  # SDK grid, never resized embedded JPEG
    np.testing.assert_array_equal(np.asarray(preview.image)[15, 15], native[15, 15])
    assert 'historical temperatures not reproduced' in preview.message


@pytest.mark.parametrize('failure', ['source_changed', 'grid_changed', 'invalid_profile', 'changed_during_decode'])
def test_legacy_location_mode_cannot_bypass_identity_grid_or_existing_profile_checks(tmp_path, monkeypatch, failure):
    from pvdelta import decode

    path = tmp_path/'original_T.JPG'
    Image.new('RGB', (128, 128), '#303030').save(path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    class Reader:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def processed_rgb(self):
            if failure == 'changed_during_decode':
                Image.new('RGB', (128, 128), '#404040').save(path)
            return np.zeros((63 if failure == 'grid_changed' else 64, 64, 3), np.uint8)

    monkeypatch.setattr(decode, 'RJpeg', Reader)
    preview = thermal_panel_preview(path, regions(), native_size=(64, 64),
        expected_source_hash='0'*64 if failure == 'source_changed' else expected,
        measurement_profile={'invalid': True} if failure == 'invalid_profile' else None)
    assert not preview.overlay_applied and not preview.radiometry_verified
    assert set(preview.region_states.values()) == {'unavailable'}
    np.testing.assert_array_equal(preview.image, Image.open(path).convert('RGB'))


def test_all_rgb_outlines_are_composed_on_one_image_with_consistent_labels(tmp_path):
    import hashlib

    import cv2

    path = tmp_path/'rgb.jpg'
    Image.new('RGB', (4032, 3024), '#303030').save(path)
    (tmp_path/'calibration_T.JPG').write_bytes(b'calibration placeholder')
    registration = {'calibration_frame': 'calibration_T.JPG', 'model': {'centre': [0, 0], 'scale': [1, 1],
        'coefficients': [[1000, 700], [3, 0], [0, 3], [0, 0], [0, 0], [0, 0]]}}
    items = [{'label': 'A1', 'polygon': [[200, 100], [204, 100], [204, 104], [200, 104]]},
             {'label': 'A2', 'polygon': [[400, 300], [404, 300], [404, 304], [400, 304]]}]
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    preview = rgb_panel_preview(path, items, registration=registration, camera_model='M4T',
        source_root=tmp_path, expected_rgb_hash=before, native_verified=True)
    assert preview.overlay_applied and preview.region_states == {'A1': 'approximate', 'A2': 'approximate'}
    pixels = np.asarray(preview.image)
    red = np.all(pixels == [255, 32, 32], axis=2).astype('uint8')
    assert cv2.connectedComponents(red)[0] == 3  # background + both separate outlines
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert '2/2 RGB regions' in preview.message
