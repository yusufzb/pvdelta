import numpy as np
import pytest

pytest.importorskip('cv2')
from pvdelta.native_boundaries import NativeBoundaryConfig, native_panel_proposals


def scene():
    # A 3x2 group of isolated 32x70 source rectangles; no RGB geometry given.
    grid = np.full((320, 400), 20., dtype=np.float32)
    for row in range(2):
        for column in range(3):
            y, x = 40+row*75, 70+column*37
            grid[y:y+70, x:x+32] = 35
    return grid


def test_native_observed_rectangles_have_neighbours_and_preserve_temperature():
    temperature = scene()
    original = temperature.copy()
    result = native_panel_proposals(temperature)
    assert len(result['proposals']) == 6
    assert all(len(p['neighbour_ids']) >= 1 and p['threshold_votes'] >= 2 for p in result['proposals'])
    assert result['semantic_panel_membership'] == 'unverified'
    assert not result['measurement_authorised']
    assert all(p['polygon_native_xy'] for p in result['proposals'])
    np.testing.assert_array_equal(temperature, original)


def test_rectangles_alone_are_not_a_semantic_vegetation_classifier():
    # Man-made non-PV rectangles could yield the exact same structure.
    result = native_panel_proposals(scene())
    assert result['proposals']
    assert result['status'] == 'unreviewed_native_geometry'
    assert not result['measurement_authorised']


def test_does_not_fill_missing_panel_or_isolated_rectangle():
    grid = scene()
    grid[40:110, 107:139] = 20  # missing source cell remains absent
    grid[235:305, 330:362] = 35  # isolated false rectangle is not retained
    result = native_panel_proposals(grid)
    assert len(result['proposals']) == 5
    centres = [np.mean(p['polygon_native_xy'], axis=0) for p in result['proposals']]
    assert all(np.linalg.norm(c-[123, 75]) > 15 for c in centres)
    assert all(c[0] < 250 for c in centres)


def test_noise_does_not_become_a_forced_panel_grid():
    grid = np.random.default_rng(7).normal(25, 2, (320, 400))
    assert native_panel_proposals(grid)['proposals'] == []


def test_enclosing_multiple_modules_is_not_retained_as_one_panel():
    import cv2

    grid = np.full((400, 600), 20., dtype=np.float32)
    cv2.rectangle(grid, (65, 35), (230, 120), 25., 2)
    for column in range(4):
        x = 75+column*37
        grid[43:113, x:x+32] = 35
    result = native_panel_proposals(grid)
    assert result['merged_envelopes_rejected'] >= 1
    assert len(result['proposals']) == 4
    assert all(cv2.contourArea(np.asarray(p['polygon_native_xy'], np.float32)) < 5000
               for p in result['proposals'])


@pytest.mark.parametrize('bad', [np.full((40, 40), np.nan), np.ones((40, 40)),
                                np.ones((20, 20)), np.ones((40, 40, 3))])
def test_invalid_or_uninformative_native_data_fail_explicitly(bad):
    with pytest.raises(ValueError):
        native_panel_proposals(bad)


@pytest.mark.parametrize('changes', [{'min_threshold_votes': True}, {'min_neighbours': 0},
    {'min_threshold_votes': 4}, {'min_aspect': 3, 'max_aspect': 2},
    {'min_side_px': float('nan')}, {'max_area_fraction': 1}])
def test_configuration_validation(changes):
    with pytest.raises(ValueError):
        NativeBoundaryConfig(**changes)
