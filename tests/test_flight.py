import pytest

from pvdelta.flight import FlightPlanningError, diagonal_to_axis_fov, plan_flight


def _m4t_plan(**overrides):
    values = {
        "thermal_width_px": 640,
        "thermal_height_px": 512,
        "thermal_diagonal_fov_deg": 45.0,
        "height_above_module_m": 20.0,
        "panel_tilt_deg": 0.0,
        "camera_aligned_to_panel": False,
        "forward_overlap_pct": 70.0,
        "side_overlap_pct": 20.0,
        "capture_interval_s": 1.0,
        "planned_speed_m_s": 2.0,
        "smallest_cell_feature_mm": 156.0,
        "module_width_m": 1.13,
        "module_height_m": 2.28,
        "site_length_m": 100.0,
        "site_width_m": 50.0,
    }
    values.update(overrides)
    return plan_flight(**values)


def test_m4t_gsd_and_cell_resolution_are_derived_from_native_grid():
    plan = _m4t_plan()
    assert plan.thermal_gsd_cm_px == pytest.approx(2.02, abs=0.02)
    assert plan.pixels_per_smallest_cell_feature == pytest.approx(7.72, abs=0.1)
    assert plan.maximum_height_for_resolution_m > 30


def test_alignment_to_tilted_panel_increases_slant_range():
    plan = _m4t_plan(panel_tilt_deg=25, camera_aligned_to_panel=True)
    assert plan.slant_distance_m == pytest.approx(20 / 0.906307, rel=1e-5)
    assert plan.view_angle_from_normal_deg == 0
    assert plan.thermal_gsd_cm_px > 2.0


def test_nadir_projection_applies_to_selected_rack_axis():
    height_axis = _m4t_plan(panel_tilt_deg=25, tilt_along_image_height=True)
    width_axis = _m4t_plan(panel_tilt_deg=25, tilt_along_image_height=False)
    assert height_axis.footprint_height_m > width_axis.footprint_height_m
    assert width_axis.footprint_width_m > height_axis.footprint_width_m
    assert width_axis.thermal_gsd_cm_px == pytest.approx(
        height_axis.thermal_gsd_cm_px, rel=1e-6,
    )


def test_overlap_controls_step_speed_and_image_count():
    loose = _m4t_plan(forward_overlap_pct=50, side_overlap_pct=20)
    dense = _m4t_plan(forward_overlap_pct=80, side_overlap_pct=70)
    assert dense.along_track_step_m < loose.along_track_step_m
    assert dense.overlap_limited_speed_m_s < loose.overlap_limited_speed_m_s
    assert dense.estimated_images > loose.estimated_images


def test_image_count_is_omitted_without_both_site_dimensions():
    plan = _m4t_plan(site_length_m=None, site_width_m=None)
    assert plan.estimated_flight_lines is None
    assert plan.estimated_images is None


def test_partial_site_dimensions_are_rejected():
    with pytest.raises(FlightPlanningError, match="provided together"):
        _m4t_plan(site_length_m=None)


@pytest.mark.parametrize(
    "field,value",
    [("height_above_module_m", 0), ("forward_overlap_pct", 100), ("panel_tilt_deg", 90)],
)
def test_invalid_planning_values_are_rejected(field, value):
    with pytest.raises(FlightPlanningError):
        _m4t_plan(**{field: value})


def test_diagonal_fov_conversion_preserves_axis_order():
    horizontal, vertical = diagonal_to_axis_fov(45, 640, 512)
    assert horizontal > vertical
    assert 0 < vertical < 45
