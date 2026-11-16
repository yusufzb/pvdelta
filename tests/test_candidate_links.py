from copy import deepcopy

import pytest

from pvdelta import candidate_links, repeat_views, rgb_repeat


def setup(monkeypatch, *, reverse_index=0, target_findings=True, changed=False):
    quad = [[100, 100], [180, 100], [180, 220], [100, 220]]
    other = [[300, 100], [380, 100], [380, 220], [300, 220]]
    model = {'centre': [0, 0], 'scale': [1, 1],
        'coefficients': [[0, 0], [1, 0], [0, 1], [0, 0], [0, 0], [0, 0]]}
    rows = []
    for name in ('source', 'target'):
        rows.append({'capture_id': name, 'rgb_path': name+'_V.JPG', 'thermal_path': name+'_T.JPG',
            'rgb_sha256': name+'-rgb', 'thermal_sha256': name+'-thermal', 'gimbal_pitch_deg': -90,
            'module_id': 'M004', 'module_polygon': quad, 'camera_distance_m': 2., 'time_gap_s': 1.,
            'diagnostic': {'modules': [{'module': 'M004', 'native_polygon': quad}]},
            'observations': {'M004': {'observation_id': name+'-obs',
                                    'finding_ids': [name+'-finding'] if target_findings or name=='source' else []}}})
    binding = iter(('original', 'changed')) if changed else None
    monkeypatch.setattr(candidate_links, '_context', lambda *a: (deepcopy(rows), next(binding) if binding else 'same', 'source-obs'))
    verified = []
    monkeypatch.setattr(rgb_repeat, '_verify_source', lambda row, *a: verified.append(row['capture_id']))
    monkeypatch.setattr(rgb_repeat, '_observed', lambda *a: {'modules': [
        {'polygon_rgb_xy': quad}, {'polygon_rgb_xy': other}]})
    directions = []

    def match(source, target, polygon, candidates):
        directions.append((str(source), str(target)))
        return repeat_views.PanelMatch('matched_provisional', 'Checks passed',
            target_index=0 if len(directions)==1 else reverse_index, iou=.99,
            check_p95_px=.3, check_count=8, fit_inliers=25)

    monkeypatch.setattr(repeat_views, 'match_panel', match)
    return rows, {'model': model}, verified, directions


def test_two_rgb_directions_and_both_sdk_profiles_required_without_merging(monkeypatch):
    rows, registration, verified, directions = setup(monkeypatch)
    original = deepcopy(rows)
    result = candidate_links.candidate_card_links('db', 'f', registration, sdk_library='sdk')
    row = result['rows'][0]
    assert verified == ['source', 'target'] and len(directions)==2
    assert row['status']=='provisional_candidate_link' and row['target_observation_id']=='target-obs'
    assert row['target_finding_ids']==['target-finding']
    assert row['bidirectional_rgb_passed'] and not row['panel_identity_verified']
    assert not row['thermal_component_correspondence_verified'] and not row['glare_clearance']
    assert not result['physical_panels_merged'] and result['unique_fault_count'] is None
    assert not result['decisions_transferred'] and rows==original


def test_reverse_return_to_adjacent_panel_rejects_one_way_link(monkeypatch):
    _, registration, _, directions = setup(monkeypatch, reverse_index=1)
    row = candidate_links.candidate_card_links('db', 'f', registration)['rows'][0]
    assert len(directions)==2 and row['status']=='unavailable'
    assert row['target_observation_id'] is None and not row['target_finding_ids']
    assert row['rgb_polygon'] is None and 'Reverse' in row['reason']


def test_missing_target_candidate_is_explicit_not_healthy(monkeypatch):
    _, registration, _, _ = setup(monkeypatch, target_findings=False)
    row = candidate_links.candidate_card_links('db', 'f', registration)['rows'][0]
    assert row['status']=='matched_panel_without_candidate' and not row['target_finding_ids']
    assert not row['glare_clearance'] and 'does not establish a healthy' in row['reason']


def test_unbounded_target_anchor_keeps_attempt_without_link(monkeypatch):
    _, registration, _, _ = setup(monkeypatch)
    real = candidate_links._anchor

    def anchor(row, *a, **kw):
        if row['capture_id']=='target':
            raise ValueError('Ambiguous target native anchor')
        return real(row, *a, **kw)

    monkeypatch.setattr(candidate_links, '_anchor', anchor)
    row=candidate_links.candidate_card_links('db', 'f', registration)['rows'][0]
    assert row['status']=='unavailable' and row['target_observation_id'] is None


def test_changed_inputs_withhold_whole_comparison(monkeypatch):
    _, registration, _, _ = setup(monkeypatch, changed=True)
    with pytest.raises(ValueError, match='changed during'):
        candidate_links.candidate_card_links('db', 'f', registration)


@pytest.mark.parametrize('reverse_winding', [False, True])
def test_cyclic_corner_origin_does_not_change_panel_identity(monkeypatch, reverse_winding):
    import numpy as np

    _, registration, _, _ = setup(monkeypatch)
    real = candidate_links.match_module_quads

    def reordered(*args):
        matches = real(*args)
        for match in matches:
            q=np.asarray(match['rgb_polygon'])
            match['rgb_polygon']=np.roll(q[::-1] if reverse_winding else q, 1, axis=0).tolist()
        return matches

    monkeypatch.setattr(candidate_links, 'match_module_quads', reordered)
    row=candidate_links.candidate_card_links('db','f',registration)['rows'][0]
    assert row['status']=='provisional_candidate_link' and row['bidirectional_rgb_passed']


def test_empty_tracking_support_is_rejected_before_projecting(monkeypatch):
    from pathlib import Path

    import cv2
    import numpy as np

    gray=np.random.default_rng(0).integers(20,230,(64,64),dtype=np.uint8)
    points=np.array([(x,y) for x in range(12,52,8) for y in range(12,52,5)],np.float32)
    descriptors=np.arange(len(points)*128,dtype=np.float32).reshape(len(points),128)
    monkeypatch.setattr(repeat_views,'_load_features',lambda p:(points,descriptors,1.,gray))
    monkeypatch.setattr(cv2,'goodFeaturesToTrack',lambda *a,**k:points[:,None].copy())
    monkeypatch.setattr(cv2,'findHomography',lambda *a,**k:(np.eye(3),np.ones((len(points),1),np.uint8)))
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',lambda a,b,p,*args,**k:(p.copy(),np.zeros((len(p),1),np.uint8),None))
    quad=np.array([[10,10],[52,10],[52,52],[10,52]],np.float32)
    result=repeat_views.match_panel(Path('a'),Path('b'),quad,[quad])
    assert result.status=='unavailable' and 'too few forward/backward' in result.reason
