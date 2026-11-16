import numpy as np
import pytest

from pvdelta.masking import (
    build_measurement_support,
    preview_mask_to_native,
    summarize_temperature,
)


def test_background_changes_cannot_change_measured_values_or_remove_hotspot():
    envelope = np.zeros((20, 30), dtype=bool)
    envelope[3:17, 4:26] = True
    support, = build_measurement_support([envelope], margin_px=2)
    temperature = np.full(envelope.shape, 25.0)
    temperature[9:11, 14:16] = 90.0
    expected = temperature[support.interior].copy()
    for outside in (-1000, 1000, np.nan, np.inf):
        altered = temperature.copy()
        altered[~support.interior] = outside
        masked = support.masked_temperature(altered)
        np.testing.assert_array_equal(masked[support.interior], expected)
        assert np.isnan(masked[~support.interior]).all()
        assert np.nanmax(masked) == 90
    assert not np.any(support.interior & support.excluded_edge)
    np.testing.assert_array_equal(support.interior | support.excluded_edge, envelope)


def test_invalid_inner_pixel_is_excluded_but_envelope_and_edge_remain_geometric():
    envelope = np.ones((10, 10), dtype=bool)
    support, = build_measurement_support([envelope], margin_px=1)
    grid = np.full((10, 10), 30.0)
    grid[4, 4] = np.nan
    assert support.interior[4, 4]
    assert not support.valid_mask(grid)[4, 4]
    assert not support.excluded_edge[4, 4]
    envelope[:] = False
    assert support.envelope.all()


def test_overlap_is_rejected_instead_of_assigning_pixels_to_two_modules():
    first = np.zeros((12, 12), dtype=bool)
    second = first.copy()
    first[1:8, 1:8] = True
    second[6:11, 6:11] = True
    with pytest.raises(ValueError, match="overlap"):
        build_measurement_support([first, second], margin_px=1)


def test_tiny_partial_panel_keeps_empty_interior_for_explicit_rejection():
    support, = build_measurement_support([np.ones((2, 2), dtype=bool)], margin_px=2)
    assert not support.interior.any()
    assert support.excluded_edge.all()
    assert np.isnan(support.masked_temperature(np.ones((2, 2)))).all()


@pytest.mark.parametrize("margin", [-1, 1.5, True])
def test_margin_requires_nonnegative_integer(margin):
    with pytest.raises(ValueError, match="integer"):
        build_measurement_support([], margin_px=margin)


def test_shape_and_dtype_are_checked():
    with pytest.raises(ValueError, match="boolean"):
        build_measurement_support([np.ones((3, 3))], margin_px=0)
    support, = build_measurement_support([np.ones((3, 3), dtype=bool)], margin_px=0)
    with pytest.raises(ValueError, match="shape"):
        support.masked_temperature(np.zeros((6, 6)))


def test_summary_uses_exact_interior_not_bounding_box_or_excluded_edge():
    envelope = np.zeros((6, 6), dtype=bool)
    envelope[1:5, 1:3] = True
    envelope[3:5, 3:5] = True
    support, = build_measurement_support([envelope], margin_px=0)
    grid = np.full((6, 6), 1000.0)
    grid[envelope] = 25.0
    grid[3, 3] = 70.0
    result = summarize_temperature(grid, support)
    assert result.max_c == 70.0
    assert result.median_c == 25.0
    assert result.valid_pixels == 12
    grid[~envelope] = np.nan
    assert summarize_temperature(grid, support) == result


def test_summary_empty_and_nonfinite_support_stay_explicit_and_json_safe():
    import json
    from dataclasses import asdict

    support, = build_measurement_support([np.ones((2, 2), dtype=bool)], margin_px=0)
    values = np.full((2, 2), np.nan)
    result = summarize_temperature(values, support)
    assert result.status == "unavailable"
    assert result.max_c is None
    json.dumps(asdict(result), allow_nan=False)
    values[0, 0] = 40
    result = summarize_temperature(values, support)
    assert result.status == "incomplete_support"
    assert result.invalid_pixels == 3
    assert result.max_c == 40


def test_preview_mask_reduction_is_conservative_and_temperature_independent():
    preview = np.zeros((6, 8), dtype=bool)
    preview[2:4, 2:4] = True
    preview[4:6, 4:8] = True
    preview[0, 0] = True
    conservative = preview_mask_to_native(preview, (3, 4))
    permissive = preview_mask_to_native(
        preview, (3, 4), require_full_preview_coverage=False
    )
    expected = np.zeros((3, 4), dtype=bool)
    expected[1, 1] = True
    expected[2, 2:4] = True
    np.testing.assert_array_equal(conservative, expected)
    assert permissive[0, 0]
    assert not conservative.flags.writeable


def test_preview_mask_reduction_rejects_implicit_crop_and_wrong_dtype():
    with pytest.raises(ValueError, match="integer ratio"):
        preview_mask_to_native(np.ones((7, 8), dtype=bool), (3, 4))
    with pytest.raises(ValueError, match="boolean"):
        preview_mask_to_native(np.ones((6, 8), dtype=np.uint8), (3, 4))
