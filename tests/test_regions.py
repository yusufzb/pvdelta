import numpy as np
import pytest

from pvdelta.regions import extract_module_mask, load_annotations, polygon_to_mask


def test_polygon_pixel_centres_and_axis_order():
    mask = polygon_to_mask([(2, 2), (7, 2), (7, 7), (2, 7)], (10, 12))
    assert mask.shape == (10, 12)
    assert mask.dtype == np.bool_
    assert mask.sum() == 25
    assert mask[2, 2] and mask[6, 6]
    assert not mask[7, 7] and not mask[1, 1]
    assert extract_module_mask(12, 10, [(2, 2), (7, 2), (7, 7), (2, 7)]).shape == (10, 12)


def test_polygon_rejects_unusable_geometry():
    with pytest.raises(ValueError, match="zero area"):
        polygon_to_mask([(1, 1), (2, 2), (3, 3)], (10, 10))
    with pytest.raises(ValueError, match="finite"):
        polygon_to_mask([(0, 0), (3, 0), (np.nan, 3)], (10, 10))
    with pytest.raises(ValueError, match="outside"):
        polygon_to_mask([(20, 20), (23, 20), (23, 23)], (10, 10))


def test_annotation_schema_is_explicit(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text('''{"schema_version":"1.0","coordinate_space":"thermal_native",
      "modules":[{"module_id":"m1","polygon":[[0,0],[3,0],[3,3],[0,3]],
      "visibility":"partial","peer_group":"row-a","mask_label":7}]}''', encoding="utf-8")
    annotation = load_annotations(path)[0]
    assert annotation.visibility == "partial"
    assert annotation.peer_group == "row-a"
    assert annotation.mask_label == 7
