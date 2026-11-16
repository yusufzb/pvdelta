import numpy as np

from pvdelta.panel_identity_proposals import frame_geometry, panel_pair, project


def _correspondence():
    # Simulate the return track's 180-degree image rotation with a perspective term.
    points = np.array([(x, y) for x in range(12, 220, 13) for y in range(12, 220, 13)], float)
    matrix = np.array([[-1., .01, 400], [.02, -1., 350], [.00001, .00002, 1.]])
    polygon = np.array([[4, 4], [230, 4], [230, 230], [4, 230]], float)
    return points, project(points, matrix), polygon, project(polygon, matrix)


def test_return_track_rotation_and_distinct_regions_can_share_a_panel():
    a, b, first, second = _correspondence()
    geometry = frame_geometry(a, b)
    result = panel_pair(first, second, geometry)
    assert result['status'] == 'provisional_repeat_support'
    assert result['iou'] > .99 and result['reverse_iou'] > .99
    # Evidence is about the panel boundary, independent of any anomaly position or count.
    assert result['local_check_points'] >= 2


def test_close_camera_or_same_appearance_cannot_merge_adjacent_or_ambiguous_panels():
    a, b, first, second = _correspondence()
    geometry = frame_geometry(a, b)
    adjacent = second + [240, 0]
    assert panel_pair(first, adjacent, geometry)['status'] == 'different_or_unresolved_panel'
    result = panel_pair(first, second, geometry, competitors=[second+[1, 0]])
    assert result['status'] == 'ambiguous_neighbour'


def test_background_alignment_without_local_panel_texture_stays_unresolved():
    a, b, first, second = _correspondence()
    first = first + [500, 500]
    geometry = frame_geometry(a, b)
    second = project(first, geometry['forward'])
    result = panel_pair(first, second, geometry)
    assert result['iou'] > .99
    assert result['status'] == 'overlap_only_review_required'
    assert result['local_fit_points'] == result['local_check_points'] == 0


def test_independent_check_failure_cannot_be_hidden_by_good_fit():
    a, b, first, second = _correspondence()
    check = (np.floor(a[:, 0]/16).astype(int)+np.floor(a[:, 1]/16).astype(int)) % 4 == 0
    b[check] += [25, -17]
    geometry = frame_geometry(a, b)
    assert geometry['status'] == 'heldout_geometry_failed'
    assert panel_pair(first, second, geometry)['status'] == 'heldout_geometry_failed'


def test_insufficient_features_is_not_evidence_of_different_panels():
    geometry = frame_geometry(np.empty((0, 2)), np.empty((0, 2)))
    assert geometry['status'] == 'insufficient_features'
