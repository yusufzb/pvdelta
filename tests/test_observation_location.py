import json

import pytest

from pvdelta.observation_location import project_observation_center


def test_lrf_raycast_returns_panel_plane_center():
    result = project_observation_center(
        json.dumps([[300, 240], [340, 240], [340, 280], [300, 280]]),
        target_latitude=10.125,
        target_longitude=-120.25,
        laser_distance_m=18.0,
        gimbal_pitch_deg=-90,
        gimbal_yaw_deg=-90.0,
        gimbal_roll_deg=180,
    )
    assert result["method"].endswith("provisional")
    assert result["latitude"] == pytest.approx(10.125, abs=0.001)
    assert result["longitude"] == pytest.approx(-120.25, abs=0.001)
    assert result["ray_distance_m"] > 0


def test_location_requires_a_complete_quad():
    with pytest.raises(ValueError):
        project_observation_center(
            [[0, 0], [1, 0], [1, 1]], target_latitude=40, target_longitude=26,
            laser_distance_m=18, gimbal_pitch_deg=-90, gimbal_yaw_deg=0, gimbal_roll_deg=0,
        )
