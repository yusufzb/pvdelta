import numpy as np
import pytest

pytest.importorskip('cv2')
from pvdelta.native_boundaries import NativeBoundaryConfig
from pvdelta.native_lines import (
    NativeLineConfig,
    _cluster_lines,
    _coverage,
    _families,
    _image_domain_pairs,
    _intersection,
    _local_structure,
    native_line_proposals,
)


def test_pair_filter_uses_observed_domain_not_origin_and_allows_reversed_order():
    perpendicular = [{'slope': 0.}]
    diverging = [{'offset': 50., 'slope': 0.}, {'offset': 51., 'slope': .1}]
    assert _image_domain_pairs(diverging, perpendicular, (200, 300), 18) == [(0, 1, 1.)]
    reversed_order = [{'offset': 50., 'slope': .1}, {'offset': 51., 'slope': 0.}]
    assert _image_domain_pairs(reversed_order, perpendicular, (200, 300), 18) == [(0, 1, 1.)]
    assert _image_domain_pairs(diverging, perpendicular, (0, 100), 18) == []


def test_pair_filter_preserves_minimum_length_on_sloped_crossing_side():
    family = [{'offset': 0., 'slope': 0.}, {'offset': 17.9, 'slope': 0.}]
    assert _image_domain_pairs(family, [{'slope': .2}], (0, 100), 18)
    assert not _image_domain_pairs(family, [{'slope': 0.}], (0, 100), 18)


def test_pair_eligibility_is_invariant_to_search_origin_translation():
    family = [{'offset': 50., 'slope': 0.}, {'offset': 51., 'slope': .1}]
    expected = [(0, 1)]
    for shift in (-1000., 0., 1000.):
        translated = [dict(line, offset=line['offset']+line['slope']*shift) for line in family]
        pairs = _image_domain_pairs(translated, [{'slope': 0.}], (200-shift, 300-shift), 18)
        assert [(i, j) for i, j, _ in pairs] == expected


def test_both_sloped_families_keep_a_quad_that_passes_actual_side_length():
    from pvdelta.native_lines import _quad_candidates

    h = [{'offset': y, 'slope': .05} for y in (20., 56.)]
    v = [{'offset': x, 'slope': .05} for x in (20., 37.96)]
    pairs = [[(0, 1, 17.96)], [(0, 1, 36.)]]
    actual = list(_quad_candidates(h, v, pairs, NativeBoundaryConfig(), 10000))
    assert len(actual) == 1  # All final geometry gates, including side length.
    assert _image_domain_pairs(v, h, (0., 99.), 18.)


def test_pair_search_recovers_observed_diverging_quad_without_loosening_geometry(monkeypatch):
    import pvdelta.native_lines as lines

    horizontal = [{'offset': y, 'slope': 0., 'intervals': [(45., 90.)]} for y in (250., 310.)]
    vertical = [{'offset': x, 'slope': slope, 'intervals': [(250., 310.)]}
                for x, slope in ((50., 0.), (51., .1))]
    neighbour = {'polygon_native_xy': [[20., 250.], [48., 250.], [48., 310.], [20., 310.]]}
    monkeypatch.setattr(lines, 'native_panel_proposals', lambda *a, **k:
                        {'proposals': [neighbour], 'measurement_authorised': False})
    monkeypatch.setattr(lines, '_families', lambda *a, **k: (np.eye(2), [horizontal, vertical], 4))
    values = np.indices((400, 400)).sum(axis=0).astype(float)
    result = lines._line_pass(values)
    assert result['line_proposal_count'] == 1
    expected = np.array([[50., 250.], [76., 250.], [82., 310.], [50., 310.]])
    np.testing.assert_allclose(result['proposals'][1]['polygon_native_xy'], expected)
    assert min(result['proposals'][1]['side_coverage']) >= .8
    assert not result['measurement_authorised']


def broken_frames(columns=3, shape=(320, 400)):
    import cv2

    frame = np.full(shape, 25., dtype=np.float32)
    for row in range(2):
        for column in range(columns):
            x, y = 65+column*42, 55+row*88
            cv2.rectangle(frame, (x, y), (x+36, y+80), 15., 2)
            frame[y+35:y+43, x+34:x+39] = 25  # visible 8px gap on the long edge
    return frame


def test_coverage_counts_observed_union_not_filled_span():
    fraction, gap = _coverage([(0, 2), (1, 3), (8, 10)], 0, 10)
    assert fraction == .5
    assert gap == .5
    assert _coverage([(-10, 20)], 0, 10) == (1., 0.)


def test_all_sides_need_observed_lines_and_temperature_is_preserved():
    frame = broken_frames()
    original = frame.copy()
    result = native_line_proposals(frame)
    assert len(result['proposals']) >= 3
    assert result['line_proposal_count'] >= 1
    assert not result['measurement_authorised']
    assert result['semantic_panel_membership'] == 'unverified'
    for proposal in result['proposals']:
        if proposal['source_method'] == 'four_observed_lines':
            assert min(proposal['side_coverage']) >= .8
            assert max(proposal['side_max_gap_fraction']) <= .18
    np.testing.assert_array_equal(frame, original)


@pytest.mark.parametrize('support_mode', ['segments', 'hybrid'])
def test_missing_side_does_not_become_a_full_panel(support_mode):
    frame = broken_frames()
    frame[54:58, 65:102] = 25  # remove entire top side of first frame
    result = native_line_proposals(frame,support_mode=support_mode)
    for proposal in result['proposals']:
        centre = np.mean(proposal['polygon_native_xy'], axis=0)
        assert np.linalg.norm(centre-[83, 95]) > 15


@pytest.mark.parametrize('support_mode', ['segments', 'hybrid'])
def test_noise_does_not_force_a_table_grid(support_mode):
    frame = np.random.default_rng(13).normal(25, 3, (320, 400))
    assert native_line_proposals(frame,support_mode=support_mode)['proposals'] == []


@pytest.mark.parametrize('nested', [True, False])
def test_opencv_line_array_layouts_are_both_supported(monkeypatch, nested):
    import cv2

    lines = np.array([[10, 10, 90, 10], [10, 70, 90, 70],
                      [10, 10, 10, 70], [90, 10, 90, 70]], np.float32)

    class Detector:
        def detect(self, gray):
            return (lines[:, None] if nested else lines, None, None, None)

    monkeypatch.setattr(cv2, 'createLineSegmentDetector', lambda *a: Detector())
    basis, families, count = _families(np.zeros((100, 100), np.uint8), NativeLineConfig())
    assert [len(family) for family in families] == [2, 2]
    assert count == 4
    np.testing.assert_allclose(basis.T@basis, np.eye(2), atol=1e-6)


def test_rotated_frame_retains_observed_line_support():
    result = native_line_proposals(np.rot90(broken_frames()).copy())
    assert result['line_proposal_count'] >= 1
    assert len(result['proposals']) >= 3


@pytest.mark.parametrize('support_mode', ['segments', 'hybrid'])
def test_multiple_panel_columns_cannot_form_a_super_panel(support_mode):
    result = native_line_proposals(broken_frames(columns=4, shape=(360, 600)),support_mode=support_mode)
    assert result['line_proposal_count'] > 0
    for proposal in result['proposals']:
        polygon = np.asarray(proposal['polygon_native_xy'])
        assert np.linalg.norm(np.roll(polygon, -1, axis=0)-polygon, axis=1).max() < 110


def test_one_central_half_cut_seam_does_not_hide_full_panel():
    import cv2

    frame = broken_frames()
    for row in range(2):
        for column in range(3):
            x, y = 65+column*42, 55+row*88
            cv2.line(frame, (x, y+40), (x+36, y+40), 15., 2)
    result = native_line_proposals(frame)
    assert len(result['proposals']) >= 3
    assert result['line_proposal_count'] > 0


@pytest.mark.parametrize('change', [{'min_segment_px': 0}, {'max_angle_deg': 11},
    {'max_gap_fraction': 1}, {'min_side_coverage': float('nan')}, {'merge_distance_px': True}])
def test_line_config_rejects_invalid_limits(change):
    with pytest.raises(ValueError):
        NativeLineConfig(**change)


def test_collinear_fragments_keep_slope_and_do_not_fill_missing_intervals():
    segments = [np.array([[10., 21.], [40., 24.]]),
                np.array([[80., 28.], [120., 32.]])]
    lines = _cluster_lines(segments, axis=0, distance=3)
    assert len(lines) == 1
    assert lines[0]['slope'] == pytest.approx(.1)
    assert lines[0]['offset'] == pytest.approx(20.)
    coverage, gap = _coverage(lines[0]['intervals'], 10, 120)
    assert coverage == pytest.approx(70/110)
    assert gap == pytest.approx(40/110)


def test_parallel_borders_and_chained_offsets_cannot_merge():
    segments = [np.array([[10., y], [110., y]]) for y in (20, 22, 24, 30)]
    lines = _cluster_lines(segments, axis=0, distance=3)
    assert len(lines) == 3
    assert sorted(len(line['intervals']) for line in lines) == [1, 1, 2]


def test_perspective_directions_are_not_forced_perpendicular(monkeypatch):
    import cv2

    expected = np.array([[60., 40.], [110., 40.], [126., 170.], [76., 170.]])
    lines = np.array([[*a, *b] for a, b in zip(expected, np.roll(expected, -1, axis=0))], np.float32)

    class Detector:
        def detect(self, gray):
            return (lines, None, None, None)

    monkeypatch.setattr(cv2, 'createLineSegmentDetector', lambda *args: Detector())
    basis, (horizontal, vertical), count = _families(np.zeros((220, 220), np.uint8), NativeLineConfig())
    assert count == 4
    assert (len(horizontal), len(vertical)) == (2, 2)
    recovered = np.array([_intersection(horizontal[h], vertical[v])
                          for h, v in ((0, 0), (0, 1), (1, 1), (1, 0))])@basis.T
    for point in expected:
        assert np.linalg.norm(recovered-point, axis=1).min() < 1e-5


def test_local_contrast_exposes_weak_edges_without_changing_measurements():
    import cv2

    frame = np.full((320, 400), 25., np.float32)
    frame[:100, :100] = 50.  # global dynamic range must not hide a weak panel border
    cv2.rectangle(frame, (200, 100), (240, 190), 24.8, 2)
    initial = frame.copy()
    display = _local_structure(frame)
    assert display.dtype == np.uint8 and display.shape == frame.shape
    assert int(display[140, 220])-int(display[140, 200]) > 30
    np.testing.assert_array_equal(initial, frame)


def test_skewed_broken_frames_retain_four_observed_sides():
    import cv2

    frame = broken_frames()
    skew = cv2.warpAffine(frame, np.array([[1., .10, 0], [0, 1., 0]]),
                          (400, 320), borderValue=25)
    result = native_line_proposals(skew)
    assert result['line_proposal_count'] >= 1
    assert len(result['proposals']) >= 3
    assert not result['measurement_authorised']
    for proposal in result['proposals']:
        if proposal['source_method'] == 'four_observed_lines':
            assert min(proposal['side_coverage']) >= .8
            assert max(proposal['side_max_gap_fraction']) <= .18


def test_internal_divider_is_checked_where_it_crosses_panel_not_origin_order(monkeypatch):
    import pvdelta.native_lines as lines

    horizontal = [{'offset': y, 'slope': 0., 'intervals': [(0., 600.)]} for y in (100., 300.)]
    vertical = [{'offset': x, 'slope': slope, 'intervals': [(100., 550.)]}
                for x, slope in ((50., .4), (100., 0.), (200., 0.))]
    neighbour = {'polygon_native_xy': [[100., 305.], [200., 305.], [200., 505.], [100., 505.]]}
    monkeypatch.setattr(lines, 'native_panel_proposals', lambda *a, **k:
                        {'proposals': [neighbour], 'measurement_authorised': False})
    monkeypatch.setattr(lines, '_families', lambda *a, **k:
                        (np.eye(2), [horizontal, vertical], 5))
    frame = np.indices((640, 640)).sum(axis=0).astype(float)
    result = lines.native_line_proposals(frame)
    # The divider has a smaller intercept than the left side, yet crosses the
    # proposed panel. Its observation cannot be ignored to create a super-panel.
    assert result['line_proposal_count'] == 0
    assert len(result['proposals']) == 1


def test_empty_contrast_pass_can_retain_strict_raw_observations(monkeypatch):
    import pvdelta.native_lines as lines

    calls = []
    observed = {'polygon_native_xy': [[10, 10], [30, 10], [30, 60], [10, 60]]}

    def line_pass(*args, local_contrast=True, **kwargs):
        calls.append(local_contrast)
        return {'proposals': [] if local_contrast else [observed],
                'line_status': 'unreviewed_observed_line_geometry', 'measurement_authorised': False}

    monkeypatch.setattr(lines, '_line_pass', line_pass)
    result = lines.native_line_proposals(np.zeros((64, 64)))
    assert calls == [True, False]
    assert result['proposals'] == [observed]
    assert result['structure_pass'] == 'global_only_fallback'
    assert not result['measurement_authorised']


def test_nonempty_contrast_pass_does_not_trigger_raw_retry(monkeypatch):
    import pvdelta.native_lines as lines

    calls = []

    def line_pass(*args, local_contrast=True, **kwargs):
        calls.append(local_contrast)
        return {'proposals': [{'proposal_id': 'N001'}], 'measurement_authorised': False}

    monkeypatch.setattr(lines, '_line_pass', line_pass)
    assert lines.native_line_proposals(np.zeros((64, 64)))['structure_pass'] == 'global_plus_local'
    assert calls == [True]


def test_rejected_neighbour_geometry_is_reviewable_without_becoming_a_mask(monkeypatch):
    import pvdelta.native_lines as lines

    monkeypatch.setattr(lines, 'native_panel_proposals', lambda *a, **k:
                        {'proposals': [], 'measurement_authorised': False})
    monkeypatch.setattr(lines, '_neighbours', lambda *a: False)
    result = lines.native_line_proposals(broken_frames())
    assert result['proposals'] == []
    for diagnostic in (result, result['raw_fallback_diagnostics']):
        audit = diagnostic['line_diagnostics']
        assert audit['geometric_candidates'] == sum(audit[k] for k in (
            'rejected_side_support', 'rejected_internal_divider',
            'rejected_source_border', 'four_side_candidates'))
        assert audit['rejected_neighbours'] == len(diagnostic['rejected_line_proposals']) > 0
        for rejected in diagnostic['rejected_line_proposals']:
            assert min(rejected['side_coverage']) >= .8
            assert rejected['reason'] == 'insufficient_observed_neighbours'
            assert not rejected['measurement_authorised']


@pytest.mark.parametrize('remove_native_side', [False, True])
def test_hybrid_recovers_observed_ridge_but_not_an_absent_side(monkeypatch, remove_native_side):
    import pvdelta.native_lines as lines

    frame = broken_frames()
    if remove_native_side:
        frame[58:132, 63:69] = 25.
    original = frame.copy()
    horizontal = [{'offset':y,'slope':0.,'intervals':[(65.,101.),(107.,143.),(149.,185.)]}
                  for y in (55.,135.,143.,223.)]
    vertical = [{'offset':x,'slope':0.,'intervals':[(55.,85.),(110.,135.),(143.,173.),(198.,223.)]}
                for x in (65.,101.,107.,143.,149.,185.)]
    monkeypatch.setattr(lines,'native_panel_proposals',lambda *a,**k:
                        {'proposals':[],'measurement_authorised':False})
    monkeypatch.setattr(lines,'_families',lambda *a,**k:
                        (np.eye(2),[horizontal,vertical],10))
    assert not lines.native_line_proposals(frame)['proposals']
    result = lines.native_line_proposals(frame,support_mode='hybrid')
    assert len(result['proposals']) >= 3
    first = [p for p in result['proposals'] if np.linalg.norm(
        np.mean(p['polygon_native_xy'],axis=0)-[83,95])<10]
    assert bool(first) == (not remove_native_side)
    for proposal in result['proposals']:
        assert proposal['source_method']=='observed_lines_and_native_ridges'
        assert min(proposal['side_coverage'])>=.8
        assert len(proposal['native_supported_sides'])<=2
    np.testing.assert_array_equal(frame,original)


def test_unknown_support_mode_fails_explicitly():
    with pytest.raises(ValueError,match='support_mode'):
        native_line_proposals(broken_frames(),support_mode='unchecked')


def test_scale_references_need_two_independent_full_module_contours():
    from pvdelta.native_lines import _closed_scale_references

    panel = np.array([[10.,10.],[50.,10.],[50.,100.],[10.,100.]])
    first = {'polygon_native_xy':(panel+[50.,0.]).tolist()}
    second = {'polygon_native_xy':(panel+[100.,0.]).tolist()}
    assert len(_closed_scale_references(panel,[first,second]))==2
    assert _closed_scale_references(panel,[first,first])==[]
    assert _closed_scale_references(panel,[{'polygon_native_xy':panel.tolist()},first])==[]
    merged = panel.copy()
    merged[1:3,0] += 40.
    assert _closed_scale_references(merged,[first,second])==[]
    rotated = np.array([[10.,10.],[100.,10.],[100.,50.],[10.,50.]])
    assert _closed_scale_references(rotated,[first,second])==[]


def _proposal(x, y, width=40, height=90, method='four_observed_lines'):
    return {'polygon_native_xy':[[x,y],[x+width,y],[x+width,y+height],[x,y+height]],
            'source_method':method}


def test_hybrid_retains_disjoint_baseline_fallback_and_does_not_mutate_inputs(monkeypatch):
    from copy import deepcopy

    import pvdelta.native_lines as lines

    baseline = {'proposals':[_proposal(500,20),_proposal(500,115)],
                'structure_pass':'global_only_fallback','measurement_authorised':False}
    hybrid = {'proposals':[_proposal(50,20),_proposal(50,115)],
              'structure_pass':'global_plus_local','measurement_authorised':False}
    before = deepcopy((baseline,hybrid))
    monkeypatch.setattr(lines,'_single_mode_proposals',lambda *a,support_mode='segments',**k:
                        baseline if support_mode=='segments' else hybrid)
    result = lines.native_line_proposals(np.zeros((320,640)),support_mode='hybrid')
    assert len(result['proposals'])==4
    assert result['composition_diagnostics']['baseline_structure_pass']=='global_only_fallback'
    assert (baseline,hybrid)==before
    assert not result['measurement_authorised']


def test_hybrid_near_duplicate_does_not_shift_baseline_geometry_or_neighbours():
    from pvdelta.native_lines import _compose_hybrid

    baseline = {'proposals':[_proposal(20,20),_proposal(65,20)],
                'structure_pass':'global_plus_local'}
    hybrid = {'proposals':[_proposal(23,20),_proposal(68,20)],
              'structure_pass':'global_plus_local'}
    result = _compose_hybrid(baseline,hybrid,NativeBoundaryConfig())
    assert [p['polygon_native_xy'] for p in result['proposals']]==[
        p['polygon_native_xy'] for p in baseline['proposals']]
    assert [p['neighbour_ids'] for p in result['proposals']]==[['N002'],['N001']]


@pytest.mark.parametrize('method',['four_observed_lines','closed_contour'])
def test_compound_envelope_is_review_only_while_observed_children_keep_interiors(method):
    from pvdelta.native_lines import _compose_hybrid
    from pvdelta.regions import polygon_to_mask
    from pvdelta.support_partition import partition_measurement_support

    parent = _proposal(10,10,200,100,method)
    children = [_proposal(10,10,45,100),_proposal(60,10,45,100)]
    baseline = {'proposals':[parent],'structure_pass':'global_plus_local'}
    hybrid = {'proposals':children,'structure_pass':'global_plus_local'}
    result = _compose_hybrid(baseline,hybrid,NativeBoundaryConfig())
    assert [p['polygon_native_xy'] for p in result['proposals']]==[
        p['polygon_native_xy'] for p in children]
    rejected = result['superseded_compound_proposals']
    assert len(rejected)==1 and rejected[0]['polygon_native_xy']==parent['polygon_native_xy']
    assert len(rejected[0]['child_polygons_native_xy'])==2
    assert not rejected[0]['measurement_authorised']
    masks = [polygon_to_mask(p['polygon_native_xy'],(140,240)) for p in result['proposals']]
    supports = partition_measurement_support(masks,margin_px=2)
    assert all(s.interior.any() for s in supports)
    assert not np.any(supports[0].interior & supports[1].interior)


@pytest.mark.parametrize('children',[
    [_proposal(10,10,45,100)],  # one child is not independent evidence
    [_proposal(10,10,45,100),_proposal(10,10,45,100)],  # duplicate
    [_proposal(10,10,45,100),_proposal(30,10,45,100)],  # overlapping
    [_proposal(10,10,45,100),_proposal(160,10,45,100)],  # not neighbours
    [_proposal(10,10,45,100),_proposal(60,-40,45,100)],  # mostly outside
    [_proposal(10,10,90,40),_proposal(105,10,90,40)],  # parallel thermal subregions
    [_proposal(10,10,20,45),_proposal(35,10,20,45)],  # incompatible scale
])
def test_compound_rejection_needs_independent_contained_adjacent_full_height_children(children):
    from pvdelta.native_lines import _compound_children

    assert _compound_children(_proposal(10,10,200,100),children)==[]


def test_composition_does_not_invent_neighbours_for_isolated_added_candidate():
    from pvdelta.native_lines import _compose_hybrid

    baseline = {'proposals':[_proposal(20,20),_proposal(65,20)],
                'structure_pass':'global_plus_local'}
    hybrid = {'proposals':[_proposal(300,20)],'structure_pass':'global_plus_local'}
    result = _compose_hybrid(baseline,hybrid,NativeBoundaryConfig())
    assert len(result['proposals'])==2
    assert len(result['composition_rejected_proposals'])==1
    assert not result['composition_rejected_proposals'][0]['measurement_authorised']
