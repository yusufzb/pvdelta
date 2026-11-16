import numpy as np
import pytest

pytest.importorskip('PIL')
from PIL import Image

from pvdelta.finding_preview import thermal_finding_preview
from pvdelta.mask_review import review_region_support


def test_mask_review_keeps_edge_hotspot_and_reports_exact_pixels_not_bbox():
    envelope = np.zeros((20, 30), dtype=bool)
    envelope[2:18, 2:28] = True
    points = [[2, 4], [3, 4], [10, 10]]
    before = envelope.copy()
    review = review_region_support(envelope, points)
    assert review['status'] == 'inset_sensitive'
    assert [r['remaining_region_pixels'] for r in review['insets']] == [3, 2, 1, 1, 1]
    assert review['finding_action'] == 'retain_for_review'
    np.testing.assert_array_equal(envelope, before)
    assert review_region_support(envelope, [[0, 0]])['outside_envelope_pixels'] == 1
    assert review_region_support(envelope, [[10, 10]])['status'] == 'retained_at_tested_insets'
    assert review_region_support(envelope, None)['status'] == 'unavailable'


def test_explicit_measurement_mask_checks_effective_boundary_not_original_polygon():
    envelope = np.ones((30, 30), dtype=bool)
    measurement = np.zeros_like(envelope)
    measurement[10:20, 10:20] = True
    result = review_region_support(envelope, [[10, 12]], measurement_mask=measurement)
    assert result['status'] == 'inset_sensitive'
    assert result['insets'][1]['remaining_region_pixels'] == 0
    assert result['probe_basis'] == 'explicit_measurement_mask'
    assert review_region_support(envelope, [[5, 5]], measurement_mask=measurement)[
        'status'] == 'outside_measurement_support'
    with pytest.raises(ValueError, match='inside'):
        review_region_support(measurement, [[12, 12]], measurement_mask=envelope)


@pytest.mark.parametrize('pixels', [[[1.5, 2]], [[-1, 2]], [[20, 2]],
                                  [[True, False]], [[2, 2], [2, 2]]])
def test_mask_review_does_not_truncate_clamp_or_duplicate_region(pixels):
    with pytest.raises(ValueError, match='unique integer'):
        review_region_support(np.ones((20, 20), dtype=bool), pixels)


@pytest.mark.parametrize('insets', [(1, 2), (0, True), (0, 2, 1), (0, 1.5), (0, 0)])
def test_mask_review_rejects_invalid_probe_distances(insets):
    with pytest.raises(ValueError, match='increasing'):
        review_region_support(np.ones((20, 20), dtype=bool), [[5, 5]], insets=insets)


def fake_sdk(monkeypatch, array=None, error=False):
    from pvdelta import decode

    class FakeSDK:
        def __init__(self, path, library):
            assert library == 'local-sdk'
            if error:
                raise RuntimeError('SDK unavailable')

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def processed_rgb(self):
            return array.copy()

    monkeypatch.setattr(decode, 'RJpeg', FakeSDK)


@pytest.mark.parametrize('sdk_failure', ['error', 'wrong_grid'])
def test_embedded_jpeg_cannot_receive_scaled_native_overlay(tmp_path, monkeypatch, sdk_failure):
    import hashlib

    path = tmp_path / 'thermal.jpg'
    Image.new('RGB', (120, 80), (20, 40, 60)).save(path)
    fake_sdk(monkeypatch, np.zeros((10, 10, 3), dtype=np.uint8), error=sdk_failure == 'error')
    result = thermal_finding_preview(path, [[10, 10], [15, 10], [15, 15], [10, 15]],
                                    sdk_library='local-sdk', native_size=(60, 40),
                                    expected_source_hash=hashlib.sha256(path.read_bytes()).hexdigest())
    assert not result.overlay_applied
    assert result.image.size == (120, 80)
    with Image.open(path) as original:
        np.testing.assert_array_equal(result.image, original.convert('RGB'))


def test_native_overlay_preserves_hot_pixel_interior_and_marks_exact_support(monkeypatch, tmp_path):
    import hashlib

    path = tmp_path/'original_T.JPG'
    path.write_bytes(b'original rjpeg fixture')
    array = np.full((40, 60, 3), 30, dtype=np.uint8)
    array[15, 15] = [250, 100, 10]
    fake_sdk(monkeypatch, array)
    result = thermal_finding_preview(path, [[14, 14], [17, 14], [17, 17], [14, 17]],
        sdk_library='local-sdk', native_size=(60, 40), region_pixels_yx=[[15, 15]],
        module_polygon=[[5, 5], [50, 5], [50, 35], [5, 35]],
        expected_source_hash=hashlib.sha256(path.read_bytes()).hexdigest())
    actual = np.asarray(result.image)
    assert result.overlay_applied
    np.testing.assert_array_equal(actual[15, 15], array[15, 15])
    assert tuple(actual[14, 15]) == (255, 255, 255)
    assert tuple(actual[5, 5]) == (0, 255, 255)
    assert 'Native SDK grid' in result.message
    assert not result.radiometry_verified
    np.testing.assert_array_equal(array[14, 15], [30, 30, 30])
