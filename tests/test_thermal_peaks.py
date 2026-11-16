from copy import deepcopy

import numpy as np
import pytest

from pvdelta.thermal_peaks import PeakPartitionConfig, partition_thermal_peaks


def two_peaks():
    grid = np.full((15, 25), 90., dtype=float)  # Hot outside must never enter.
    grid[4:7, 5:18] = 30+np.array([3, 4, 7, 9, 7, 4, 3, 4, 7, 9, 7, 4, 3])
    return grid, [(y, x) for y in range(4, 7) for x in range(5, 18)]


def assert_conserved(result, original):
    parts = [set(map(tuple, r['region_pixels_yx'])) for r in result['regions']]
    bridge = set(map(tuple, result['ambiguous_bridge_pixels_yx']))
    assert all(not (p & bridge) for p in parts)
    assert all(not (p & q) for i, p in enumerate(parts) for q in parts[i+1:])
    assert set().union(bridge, *parts) == set(original)


def test_two_native_peaks_split_without_soil_pixels_or_arbitrary_bridge_ownership():
    grid, coords = two_peaks()
    before, old_coords = grid.copy(), deepcopy(coords)
    grid.setflags(write=False)
    result = partition_thermal_peaks(grid, coords)
    assert result['status'] == 'partitioned_for_review'
    assert len(result['regions']) == 2
    assert len(result['ambiguous_bridge_pixels_yx']) == 3
    assert_conserved(result, coords)
    assert max(r['temp_max_c'] for r in result['regions']) == 39
    assert not result['measurement_authorised'] and not result['child_reference_delta_available']
    np.testing.assert_array_equal(grid, before)
    assert coords == old_coords


def test_flat_warm_patch_has_no_forced_cell_count():
    grid, coords = two_peaks()
    for y, x in coords:
        grid[y, x] = 39
    result = partition_thermal_peaks(grid, coords)
    assert result['status'] == 'unsplit' and result['regions'] == []
    assert result['parent_pixels_yx'] == [list(p) for p in coords]


def test_shallow_saddle_or_unsupported_spike_does_not_create_extra_regions():
    grid, coords = two_peaks()
    for y, x in coords:
        grid[y, x] = 38.5 if x == 11 else 39
    result = partition_thermal_peaks(grid, coords)
    assert result['status'] == 'unsplit'
    grid, coords = two_peaks()
    for y, x in coords:
        grid[y, x] = 35
    grid[5, 8], grid[5, 14] = 40, 39
    assert partition_thermal_peaks(grid, coords)['status'] == 'unsplit'


def test_monotonic_temperature_gradient_cannot_become_multiple_hotspots():
    grid, coords = two_peaks()
    for y, x in coords:
        grid[y, x] = 30+x*.5
    assert partition_thermal_peaks(grid, coords)['status'] == 'unsplit'


def test_pixel_order_does_not_change_peak_identity_or_bridge_ties():
    grid, coords = two_peaks()
    assert partition_thermal_peaks(grid, coords) == partition_thermal_peaks(grid, coords[::-1])


def test_holes_and_exterior_stay_outside_children():
    grid, coords = two_peaks()
    coords.remove((4, 10))
    grid[4, 10] = float('nan')
    result = partition_thermal_peaks(grid, coords)
    assert result['status'] == 'partitioned_for_review'
    assert_conserved(result, coords)


def test_every_child_statistic_comes_from_its_exact_native_pixels():
    grid, coords = two_peaks()
    result = partition_thermal_peaks(grid, coords)
    for region in result['regions']:
        pixels = np.asarray(region['region_pixels_yx'])
        measured = grid[pixels[:, 0], pixels[:, 1]]
        assert region['temp_median_c'] == np.median(measured)
        assert region['temp_max_c'] == np.max(measured)
        assert region['n_region_px'] == len(pixels)
        assert region['peak']['prominence_k'] >= 1.25


@pytest.mark.parametrize('case', ['duplicate', 'disconnected', 'float_coords', 'outside', 'nonfinite', 'empty'])
def test_invalid_parent_support_is_rejected(case):
    grid, coords = two_peaks()
    if case == 'duplicate':
        coords.append(coords[0])
    elif case == 'disconnected':
        coords.append((12, 22))
    elif case == 'float_coords':
        coords = np.asarray(coords, dtype=float)
    elif case == 'outside':
        coords.append((-1, 0))
    elif case == 'nonfinite':
        grid[coords[0]] = float('inf')
    else:
        coords = []
    with pytest.raises(ValueError):
        partition_thermal_peaks(grid, coords)


@pytest.mark.parametrize('change', [{'min_prominence_k': 0}, {'min_prominence_k': True},
                                   {'min_prominence_k': float('nan')}, {'min_region_px': 1.5},
                                   {'max_peaks': False}, {'min_peak_support_px': 0},
                                   {'max_component_px': -1}])
def test_invalid_search_limits(change):
    with pytest.raises(ValueError):
        PeakPartitionConfig(**change)


@pytest.mark.parametrize('change,reason', [({'max_component_px': 20}, 'bounded search size'),
                                        ({'min_region_px': 40}, 'too few'),
                                        ({'max_peaks': 1}, 'too many peaks')])
def test_unsupported_partition_keeps_whole_parent(change, reason):
    grid, coords = two_peaks()
    result = partition_thermal_peaks(grid, coords, config=PeakPartitionConfig(**change))
    assert result['status'] == 'unsplit' and reason in result['reason']
    assert result['parent_pixels_yx'] == [list(p) for p in coords]
    assert not result['regions'] and not result['ambiguous_bridge_pixels_yx']
