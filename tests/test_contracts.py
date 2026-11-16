import datetime as dt
import json

import numpy as np
import pytest

from pvdelta.contracts import (
    asset_match_from_dict,
    dumps_strict,
    location_from_dict,
    thermal_frame_from_dict,
    thermal_frame_to_dict,
)
from pvdelta.models import (
    AssetMatch,
    CameraProfile,
    LocationEstimate,
    LocationStatus,
    MatchStatus,
    ThermalFrame,
)


def test_frame_json_round_trip_preserves_grid_mask_time_and_profile():
    source = ThermalFrame(
        "f1", "private/input.jpg", "abc123",
        np.array([[20.25, 21.5]], dtype=np.float32),
        np.array([[True, False]], dtype=np.bool_), 2, 1,
        capture_time=dt.datetime(2026, 4, 29, 13, 0, tzinfo=dt.timezone.utc),
        camera_profile=CameraProfile(model="M4T", gimbal_pitch=-90.0),
        raw_metadata={"unit": "degC"},
    )
    payload = json.loads(dumps_strict(thermal_frame_to_dict(source)))
    restored = thermal_frame_from_dict(payload)
    assert restored.capture_time == source.capture_time
    assert restored.camera_profile == source.camera_profile
    np.testing.assert_array_equal(restored.temperature_c, source.temperature_c)
    np.testing.assert_array_equal(restored.valid_mask, source.valid_mask)


def test_non_finite_json_values_are_rejected():
    with pytest.raises(ValueError, match="NaN"):
        dumps_strict({"temperature_c": float("nan")})


def test_inventory_and_location_are_optional_and_explicit():
    assert AssetMatch("obs").status is MatchStatus.NOT_ATTEMPTED
    assert LocationEstimate("obs").status is LocationStatus.NOT_ATTEMPTED
    with pytest.raises(ValueError, match="asset_id"):
        AssetMatch("obs", MatchStatus.MATCHED)
    with pytest.raises(ValueError, match="provided together"):
        LocationEstimate("obs", LocationStatus.INSUFFICIENT_METADATA, latitude=40.0)
    location = LocationEstimate(
        "obs", LocationStatus.ESTIMATED, 40.0, 29.0, 1.5, "calibrated-raycast",
    )
    assert location_from_dict(json.loads(dumps_strict(location))) == location
    match = AssetMatch("obs", MatchStatus.AMBIGUOUS, candidate_asset_ids=("p1", "p2"))
    assert asset_match_from_dict(json.loads(dumps_strict(match))) == match
    evidenced = AssetMatch("obs-2", MatchStatus.MATCHED, "p1", 1.0, ("p1",), "CAD-001")
    assert asset_match_from_dict(json.loads(dumps_strict(evidenced))) == evidenced
