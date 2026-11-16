import json

import pytest

from pvdelta.camera_db import get_platform, list_platforms, load_database, lookup_by_xmp_model


def test_bundled_database_is_valid_and_lookup_uses_xmp_model():
    assert "DJI_M4T" in list_platforms()
    platform = lookup_by_xmp_model("M4T")
    assert platform == get_platform("DJI_M4T")
    assert platform is not None
    assert platform.thermal is not None
    assert platform.thermal.sensor_width_px == 640
    assert platform.thermal.actual_focal_length_mm is None
    assert platform.lrf is not None
    assert "0.15%" in (platform.lrf.accuracy or "")
    assert platform.source_urls[0].startswith("https://enterprise.dji.com/")


def test_unknown_xmp_model_is_nonfatal():
    assert lookup_by_xmp_model("FUTURE-DRONE") is None


def test_external_database_rejects_duplicate_model_ids(tmp_path):
    camera = {
        "sensor_type": "uncooled_vox",
        "sensor_width_px": 640,
        "sensor_height_px": 512,
        "equivalent_focal_length_mm": 40,
    }
    platform = {
        "manufacturer": "Example",
        "model": "Example",
        "xmp_model_id": "SAME",
        "source_urls": ["https://example.test/specs"],
        "cameras": {"thermal": camera},
    }
    path = tmp_path / "camera_db.json"
    path.write_text(
        json.dumps({"platforms": {"A": platform, "B": platform}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate xmp_model_id"):
        load_database(path)


def test_external_database_rejects_nonpositive_dimensions(tmp_path):
    path = tmp_path / "camera_db.json"
    path.write_text(json.dumps({"platforms": {"A": {
        "manufacturer": "Example",
        "model": "Example",
        "xmp_model_id": "A",
        "source_urls": ["https://example.test/specs"],
        "cameras": {"thermal": {
            "sensor_type": "uncooled_vox",
            "sensor_width_px": 0,
            "sensor_height_px": 512,
            "equivalent_focal_length_mm": 40,
        }},
    }}}), encoding="utf-8")
    with pytest.raises(ValueError, match="sensor_width_px"):
        load_database(path)
