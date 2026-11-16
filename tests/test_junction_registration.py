import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')

from pvdelta.alignment import transform_points
from pvdelta.junction_registration import (
    JunctionConfig,
    _edge_pair_centres,
    _resolve_matches,
    match_table_junctions,
    radiometric_structure,
)


def scene():
    native = np.full((512, 640), 30, np.uint8)
    quads = []
    for col in range(4):
        for row in range(3):
            x, y = 100+col*75, 100+row*95
            q = np.float32([[x, y], [x+75, y], [x+75, y+95], [x, y+95]])
            quads.append(q)
            cv2.fillConvexPoly(native, q.astype(np.int32), 140)
            cv2.polylines(native, [q.astype(np.int32)], True, 35, 2)
    matrix = np.float64([[2, .01, 23], [-.01, 2, 17], [0, 0, 1]])
    rgb = cv2.warpPerspective(255-native, matrix, (1280, 1024), borderValue=225)
    rgb_quads = [transform_points(q, matrix)+[1, -1] for q in quads]
    prior = [q+[4, -3] for q in quads]
    return rgb, native, rgb_quads, prior, quads, matrix


@pytest.mark.parametrize('paired_split', [False, True])
def test_module_side_mode_localises_full_scene_without_measurement_authority(paired_split):
    rgb, native, rgb_quads, prior, truth, matrix = scene()
    result = match_table_junctions(rgb, native, rgb_quads, prior,
                                  config=JunctionConfig(module_side_only=True, paired_block_split=paired_split))
    assert result['image_consistent'], result
    expected = transform_points(np.concatenate(truth), matrix)
    recovered = transform_points(expected, np.linalg.inv(result['native_to_rgb']))
    assert np.percentile(np.linalg.norm(recovered-np.concatenate(truth), axis=1), 95) < 2
    assert result['measurement_status'].startswith('withheld')


def test_collinear_fit_or_check_subset_is_rejected_before_solving(monkeypatch):
    import pvdelta.junction_registration as registration

    rgb, native, rgb_quads, prior, _, _ = scene()
    def forbidden(*args):
        raise AssertionError('A collinear role must never reach the homography solver')
    monkeypatch.setattr(registration, '_normalised_fit', forbidden)
    indices = [0, 3, 6, 9]
    result = match_table_junctions(rgb, native, [rgb_quads[i] for i in indices],
                                  [prior[i] for i in indices])
    assert not result['image_consistent']
    assert result['native_to_rgb'] is None
    assert any('junction geometry is collinear' in reason for reason in result['reasons'])


def test_module_side_flag_rejects_non_boolean():
    with pytest.raises(TypeError, match='boolean'):
        JunctionConfig(module_side_only=1)


def test_full_module_edge_support_recovers_known_mapping_without_temperature_changes():
    rgb,native,rgb_quads,prior,truth,matrix=scene()
    before=native.copy()
    result=match_table_junctions(rgb,native,rgb_quads,prior,
        config=JunctionConfig(module_side_only=True,measurement_control_split=True,
                              full_module_edge_support=True))
    assert result['image_consistent'],result
    expected=transform_points(np.concatenate(truth),matrix)
    recovered=transform_points(expected,np.linalg.inv(result['native_to_rgb']))
    assert np.percentile(np.linalg.norm(recovered-np.concatenate(truth),axis=1),95)<2.
    assert result['measurement_status'].startswith('withheld')
    np.testing.assert_array_equal(native,before)


def test_full_module_edge_support_requires_explicit_one_sided_mode():
    with pytest.raises(ValueError,match='requires module_side_only'):
        JunctionConfig(full_module_edge_support=True)
    with pytest.raises(TypeError,match='boolean'):
        JunctionConfig(full_module_edge_support=1,module_side_only=True)


def test_measurement_split_keeps_eight_fit_four_check_minimum_and_fixed_roles():
    rgb, native, rgb_quads, prior, _, _ = scene()
    result = match_table_junctions(rgb, native, rgb_quads, prior,
        config=JunctionConfig(module_side_only=True, measurement_control_split=True))
    assert result['image_consistent'], result
    fit, check = result['train_node_indices'], result['check_node_indices']
    assert len(fit) >= 8 and len(check) >= 4
    assert set(fit).isdisjoint(check)
    ordered = sorted(fit+check)
    assert check == [node for i,node in enumerate(ordered) if i % 3 == 1]
    assert result['check_support_hull_native_xy']
    assert result['measurement_status'].startswith('withheld')


def test_measurement_split_does_not_recycle_ten_junctions_or_combine_policies():
    rgb, native, rgb_quads, prior, _, _ = scene()
    indices = [0, 3, 6, 9]
    result = match_table_junctions(rgb, native, [rgb_quads[i] for i in indices],
        [prior[i] for i in indices], config=JunctionConfig(measurement_control_split=True))
    assert not result['image_consistent']
    assert result['native_to_rgb'] is None
    assert any('twelve' in reason for reason in result['reasons'])
    with pytest.raises(ValueError, match='one fixed'):
        JunctionConfig(measurement_control_split=True, paired_block_split=True)


def test_observed_cross_sensor_junctions_recover_known_mapping_without_labels():
    rgb, native, rgb_quads, prior, truth, matrix = scene()
    before = (rgb.copy(), native.copy())
    result = match_table_junctions(rgb, native, rgb_quads, prior)
    assert result['image_consistent'], result
    assert set(result['train_node_indices']).isdisjoint(result['check_node_indices'])
    expected = transform_points(np.concatenate(truth), matrix)
    recovered = transform_points(expected, np.linalg.inv(result['native_to_rgb']))
    assert np.percentile(np.linalg.norm(recovered-np.concatenate(truth), axis=1), 95) < 2
    assert result['measurement_status'].startswith('withheld')
    assert result['anchor_evidence'] == 'two_compatible_visible_table_corners_provisional'
    assert np.array_equal(rgb, before[0]) and np.array_equal(native, before[1])


def test_partial_modules_still_supply_visible_junctions_without_producing_masks():
    rgb, native, rgb_quads, prior, _, _ = scene()
    result = match_table_junctions(rgb, native[:340], rgb_quads, prior)
    assert result['image_consistent'], result
    assert any(n['status'] == 'native_seed_outside_frame' for n in result['nodes'])
    assert all(0 <= n['native_xy'][1] < 340 for n in result['nodes'] if n['status'] == 'matched_provisional')
    assert result['measurement_status'].startswith('withheld')


def test_bad_check_observations_are_not_used_to_refit_transform(monkeypatch):
    import pvdelta.junction_registration as registration

    rgb, native, rgb_quads, prior, _, _ = scene()
    original = registration._find_junction
    count = 0

    def corrupted_checks(gradients, *args, **kwargs):
        nonlocal count
        observed = original(gradients, *args, **kwargs)
        if gradients[0].shape == native.shape and observed is not None:
            count += 1
            if count % 2 == 0:
                return observed[0]+[0, 6], observed[1]
        return observed

    monkeypatch.setattr(registration, '_find_junction', corrupted_checks)
    result = match_table_junctions(rgb, native, rgb_quads, prior)
    assert not result['image_consistent']
    assert result['native_to_rgb'] is None
    assert result['validation']['check_p95_source_px'] > 4


def test_radiometric_structure_is_monotonic_and_leaves_measurements_unchanged():
    temperature = np.linspace(15, 45, 64*80).reshape(64, 80)
    before = temperature.copy()
    image, metadata = radiometric_structure(temperature)
    assert image.dtype == np.uint8 and image.shape == temperature.shape
    assert np.all(np.diff(image.ravel().astype(int)) >= 0)
    assert metadata['purpose'].endswith('not_measurement')
    np.testing.assert_array_equal(temperature, before)
    shifted, _ = radiometric_structure(temperature+20)
    np.testing.assert_allclose(shifted, image, atol=1)


@pytest.mark.parametrize('values', [np.full((32, 32), np.nan), np.ones((32, 32)), np.ones((3, 3))])
def test_invalid_radiometric_structure_abstains(values):
    with pytest.raises(ValueError):
        radiometric_structure(values)


def test_no_structure_does_not_fabricate_correspondences():
    rgb, native, rgb_quads, prior, _, _ = scene()
    result = match_table_junctions(rgb, np.full_like(native, 30), rgb_quads, prior)
    assert not result['image_consistent']
    assert result['native_to_rgb'] is None
    assert result['anchor_evidence'] == 'unresolved'
    assert all(n['status'] != 'matched_provisional' for n in result['nodes'])


def test_source_native_frame_difference_and_outside_prior_withhold_matches():
    rgb, native, rgb_quads, prior, _, _ = scene()
    result = match_table_junctions(rgb, native, rgb_quads, [q+[200, 0] for q in prior])
    assert not result['image_consistent']
    assert result['native_to_rgb'] is None


def test_interior_grid_cannot_claim_visible_table_corner_anchors():
    rgb, native, rgb_quads, prior, _, _ = scene()
    # Observed module subset, not a physical table end: actual frame lines continue.
    subset = [4, 5, 7, 8]
    result = match_table_junctions(rgb, native, [rgb_quads[i] for i in subset], [prior[i] for i in subset])
    assert result['anchor_evidence'] == 'unresolved'


def test_missing_rgb_lines_cannot_use_thermal_edges_alone():
    rgb, native, rgb_quads, prior, _, _ = scene()
    result = match_table_junctions(np.full_like(rgb, 225), native, rgb_quads, prior)
    assert not result['image_consistent']
    assert all(n['status'] == 'rgb_junction_unresolved' for n in result['nodes'])


@pytest.mark.parametrize('configuration', [
    {'rgb_search_px': True}, {'native_search_px': 0}, {'rgb_merge_px': -1},
    {'max_check_p95_px': float('nan')}, {'min_gradient': 0},
])
def test_invalid_config(configuration):
    with pytest.raises(ValueError):
        JunctionConfig(**configuration)


def test_invalid_grids_and_quads():
    rgb, native, rgb_quads, prior, _, _ = scene()
    with pytest.raises(ValueError):
        match_table_junctions(rgb.astype(float), native, rgb_quads, prior)
    with pytest.raises(ValueError):
        match_table_junctions(rgb, native, rgb_quads, prior[:2])
    bad = prior[0].copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        match_table_junctions(rgb, native, rgb_quads, [bad]+prior[1:])
    empty = match_table_junctions(rgb, native, [], [])
    assert not empty['image_consistent'] and not empty['nodes']


@pytest.mark.parametrize('points', [
    [([10, 10], [30, 30]), ([11, 10], [36, 30])],  # one RGB -> two thermal
    [([10, 10], [30, 30]), ([25, 10], [31, 30])],  # two RGB -> one thermal
    [([10, 10], [30, 30]), ([16, 10], [31, 30]), ([22, 10], [32, 30])],
])
def test_ambiguous_correspondence_in_either_sensor_never_enters_fit(points):
    nodes = [{'rgb_xy': r, 'native_xy': n, 'status': 'matched_provisional', 'is_corner_anchor': True}
             for r, n in points]
    assert _resolve_matches(nodes, 8) == []
    assert all(n['status'] == 'ambiguous_correspondence' for n in nodes)
    assert not any(n['is_corner_anchor'] for n in nodes)


def test_duplicate_junction_observations_are_one_unweighted_correspondence():
    nodes = [{'rgb_xy': r, 'native_xy': n, 'status': 'matched_provisional', 'is_corner_anchor': False}
             for r, n in [([10, 10], [30, 30]), ([12, 10], [31, 30])]]
    assert _resolve_matches(nodes, 8) == [0]
    assert nodes[0]['rgb_xy'] == [11, 10]
    assert nodes[0]['native_xy'] == [30.5, 30]
    assert nodes[1]['canonical_node_index'] == 0


@pytest.mark.parametrize('polarity', [-1, 1])
@pytest.mark.parametrize('contrast', [(100, 25), (25, 100), (100, 100)])
def test_frame_midpoint_is_not_biased_by_unequal_edge_contrast(polarity, contrast):
    offsets = np.arange(-15, 16, dtype=float)
    centre, half_width = .3, 3.5
    left = np.exp(-.5*((offsets-(centre-half_width))/.8)**2)*contrast[0]
    right = np.exp(-.5*((offsets-(centre+half_width))/.8)**2)*contrast[1]
    signed = np.tile((left-right)*polarity, (24, 1))
    located, valid = _edge_pair_centres(signed, offsets, -1, 5, 15)
    assert valid.all()
    np.testing.assert_allclose(located, centre, atol=.15)


def test_one_edge_cannot_be_used_as_a_two_edge_frame_midpoint():
    offsets = np.arange(-10, 11, dtype=float)
    signed = np.tile(np.exp(-.5*((offsets-2)/.8)**2)*100, (24, 1))
    _, valid = _edge_pair_centres(signed, offsets, 0, 4, 15)
    assert not valid.any()


def test_nearby_cell_grid_edge_cannot_replace_a_frame_edge_on_the_same_side():
    offsets = np.arange(-15, 16, dtype=float)
    centre = .2
    left = 40*np.exp(-.5*((offsets-(centre-3))/.6)**2)
    right = 25*np.exp(-.5*((offsets-(centre+3))/.6)**2)
    neighbouring_cell = 130*np.exp(-.5*((offsets-7)/.5)**2)
    signed = np.tile(left-right+neighbouring_cell, (24, 1))
    located, valid = _edge_pair_centres(signed, offsets, 0, 6, 15)
    assert valid.all()
    np.testing.assert_allclose(located, centre, atol=.15)
