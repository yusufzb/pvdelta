from datetime import datetime, timezone

import numpy as np
import pytest

from pvdelta.models import CameraProfile, Finding, RunManifest, ThermalFrame
from pvdelta.quality import QualityError, validate_module_observation, validate_thermal_frame


def frame(temp, valid_mask=None):
    return ThermalFrame(
        frame_id="f1", source_path="synthetic", source_hash="synthetic-hash",
        temperature_c=temp, valid_mask=valid_mask,
        width=temp.shape[1], height=temp.shape[0],
    )


def test_frame_rejects_inconsistent_and_unknown_data():
    temp = np.ones((10, 20), dtype=np.float32)
    assert frame(temp).width == 20
    with pytest.raises(ValueError, match="shape"):
        ThermalFrame("f1", "synthetic", "hash", temp, None, 10, 20)
    with pytest.raises(ValueError, match="valid_mask"):
        frame(temp, np.ones((10, 20), dtype=np.uint8))
    with pytest.raises(ValueError, match="source_hash"):
        ThermalFrame("f1", "synthetic", "", temp, None, 20, 10)
    assert CameraProfile().gimbal_pitch is None
    with pytest.raises(ValueError, match="timezone"):
        ThermalFrame("f1", "synthetic", "hash", temp, None, 20, 10,
                     capture_time=datetime(2026, 1, 1))  # noqa: DTZ001 - reject naive time
    assert RunManifest("run", datetime.now(timezone.utc)).run_id == "run"


def test_quality_handles_partial_and_unusable_frames():
    temp = np.full((10, 10), 25.0, dtype=np.float32)
    assert "Zero variance" in " ".join(validate_thermal_frame(temp))
    temp[0, 0] = np.nan
    assert any("NaN" in flag for flag in validate_thermal_frame(temp))
    with pytest.raises(QualityError, match="no finite"):
        validate_thermal_frame(np.full((10, 10), np.nan))
    with pytest.raises(QualityError, match="configured camera range"):
        validate_thermal_frame(np.full((10, 10), 600.0), max_c=500.0)


def test_module_validation_rejects_bad_geometry():
    polygon = [(5, 5), (15, 5), (15, 15), (5, 15)]
    assert validate_module_observation(20, 30, polygon, 100, 100) == []
    with pytest.raises(QualityError, match="temp_max"):
        validate_module_observation(30, 20, polygon, 100, 100)
    with pytest.raises(QualityError, match="completely outside"):
        validate_module_observation(20, 30, [(200, 200), (210, 200), (210, 210)], 100, 100)
    with pytest.raises(QualityError, match="zero area"):
        validate_module_observation(20, 30, [(1, 1), (2, 2), (3, 3)], 100, 100)


def test_finding_rejects_inconsistent_normalization_and_polygon():
    args = ("f1", "m1", "frame1", "hot_region", 5.0, 30.0)
    with pytest.raises(ValueError, match="status and value disagree"):
        Finding(*args, pixel_polygon=[], is_normalized=True)
    with pytest.raises(ValueError, match="finite"):
        Finding(*args, pixel_polygon=[(0, 0), (np.nan, 0), (1, 1)])
    assert Finding(*args, pixel_polygon=[], assessment_status="review_required").assessment_status == (
        "review_required"
    )
    with pytest.raises(ValueError, match="assessment_status"):
        Finding(*args, pixel_polygon=[], assessment_status="accepted_without_review")
