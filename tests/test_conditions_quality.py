import datetime as dt

import numpy as np

from pvdelta.conditions import ConditionObservation, match_conditions
from pvdelta.normalize import irradiance, string_current
from pvdelta.quality import Capability, assess_capabilities, assess_region_quality

UTC = dt.timezone.utc


def test_conditions_require_time_scope_and_source_suitability():
    capture = dt.datetime(2026, 4, 29, 13, 0, tzinfo=UTC)
    observations = [
        ConditionObservation(irradiance(800), capture, "poa-1", "plant"),
        ConditionObservation(irradiance(900), capture - dt.timedelta(minutes=5),
                             "poa-old", "plant"),
        ConditionObservation(string_current(8, 10), capture, "scada-s1", "string-1",
                             False, "string is offline or anomalous"),
    ]
    matched = match_conditions(capture, observations, scope_id="plant",
                               max_age=dt.timedelta(seconds=30))
    assert len(matched.conditions) == 1
    assert matched.sources == ("poa-1",)
    assert any("stale" in flag for flag in matched.flags)
    wrong_scope = match_conditions(capture, observations, scope_id="other",
                                   max_age=dt.timedelta(seconds=30))
    assert wrong_scope.conditions == ()


def test_missing_conditions_or_geometry_do_not_disable_image_analysis():
    assessment = assess_capabilities(
        np.arange(16, dtype=np.float32).reshape(4, 4),
        has_conditions=False, has_camera_pose=True, has_calibrated_geometry=False,
    )
    assert assessment.image_analysis is Capability.AVAILABLE
    assert assessment.normalization is Capability.UNAVAILABLE
    assert assessment.location is Capability.UNAVAILABLE


def test_region_coverage_and_saturation_are_explicit_quality_gates():
    grid = np.full((10, 10), 25.0, dtype=np.float32)
    region = np.ones(grid.shape, dtype=bool)
    valid = np.ones(grid.shape, dtype=bool)
    valid[:2] = False
    report = assess_region_quality(grid, region, valid_mask=valid)
    assert report.analysis is Capability.UNAVAILABLE
    assert report.valid_fraction == 0.8
    grid[5, 5] = 100
    report = assess_region_quality(
        grid, region, min_valid_fraction=.5, saturation_bounds_c=(-20, 100),
    )
    assert report.analysis is Capability.UNAVAILABLE
    assert any("saturation" in reason for reason in report.reasons)


def test_equally_close_conditions_stay_ambiguous_when_farther_observation_follows():
    capture = dt.datetime(2026, 4, 29, 13, 0, tzinfo=UTC)
    observations = [
        ConditionObservation(irradiance(800), capture - dt.timedelta(seconds=5),
                             "poa-a", "plant"),
        ConditionObservation(irradiance(850), capture + dt.timedelta(seconds=5),
                             "poa-b", "plant"),
        ConditionObservation(irradiance(900), capture + dt.timedelta(seconds=10),
                             "poa-farther", "plant"),
    ]
    matched = match_conditions(capture, observations, scope_id="plant",
                               max_age=dt.timedelta(seconds=30))
    assert matched.conditions == ()
    assert matched.sources == ()
    assert sum("equally close" in flag for flag in matched.flags) == 1


def test_condition_choice_is_independent_of_input_order():
    capture = dt.datetime(2026, 4, 29, 13, 0, tzinfo=UTC)
    observations = [
        ConditionObservation(irradiance(700), capture + dt.timedelta(seconds=20),
                             "far", "plant"),
        ConditionObservation(irradiance(800), capture + dt.timedelta(seconds=2),
                             "near", "plant"),
    ]
    forward = match_conditions(capture, observations, scope_id="plant",
                               max_age=dt.timedelta(seconds=30))
    reverse = match_conditions(capture, list(reversed(observations)), scope_id="plant",
                               max_age=dt.timedelta(seconds=30))
    assert forward == reverse
    assert forward.sources == ("near",)


def test_reference_pixels_are_counted_from_the_reference_mask():
    grid = np.full((10, 10), 25.0, dtype=np.float32)
    region = np.zeros(grid.shape, dtype=bool)
    region[2:5, 2:5] = True
    reference = np.zeros(grid.shape, dtype=bool)
    reference[6:10, :] = True
    reference[6, 0] = False
    report = assess_region_quality(
        grid, region, reference_mask=reference, min_reference_pixels=20,
    )
    assert report.valid_pixels == 9
    assert report.reference_pixels == 39
    assert report.analysis is Capability.AVAILABLE


def test_nonfinite_or_zero_variance_frame_disables_frame_analysis():
    nonfinite = np.arange(16, dtype=np.float32).reshape(4, 4)
    nonfinite[0, 0] = np.nan
    report = assess_capabilities(
        nonfinite, has_conditions=True, has_camera_pose=True,
        has_calibrated_geometry=True,
    )
    assert report.image_analysis is Capability.UNAVAILABLE
    assert any("NaN or Inf" in reason for reason in report.reasons)
    uniform = assess_capabilities(
        np.ones((4, 4), dtype=np.float32), has_conditions=True,
        has_camera_pose=True, has_calibrated_geometry=True,
    )
    assert uniform.image_analysis is Capability.UNAVAILABLE
