import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pvdelta.boundaries import native_edge_check, rgb_module_proposals


def test_blank_and_non_panel_rectangles_do_not_become_modules():
    image = np.full((800, 1200, 3), (40, 160, 40), np.uint8)
    cv2.rectangle(image, (200, 200), (360, 275), (40, 40, 40), -1)
    assert rgb_module_proposals(image) == []


def test_module_proposals_preserve_pixels_and_suppress_duplicates():
    image = np.full((800, 1200, 3), (40, 160, 40), np.uint8)
    cv2.rectangle(image, (200, 200), (360, 275), (110, 70, 65), -1)
    cv2.rectangle(image, (420, 200), (580, 275), (110, 70, 65), -1)
    original = image.copy()
    proposals = rgb_module_proposals(image)
    assert len(proposals) == 2
    assert np.array_equal(image, original)
    centres = sorted([p.mean(axis=0).tolist() for p in proposals])
    assert np.allclose(centres, [[280, 237.5], [500, 237.5]], atol=2)


def test_flat_native_frame_or_outside_polygon_cannot_pass_edge_check():
    polygon = np.float32([[30, 30], [130, 30], [130, 80], [30, 80]])
    gray = np.zeros((200, 200), np.uint8)
    assert not native_edge_check(gray, polygon)["accepted"]
    cv2.fillConvexPoly(gray, polygon.astype(np.int32), 200)
    assert native_edge_check(gray, polygon)["accepted"]
    assert not native_edge_check(gray, polygon-50)["accepted"]
