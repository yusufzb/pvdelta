import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from pvdelta.repeat_views import (
    assess_glare_evidence,
    component_evidence,
    location_consistency,
    match_panel,
    select_target_panel,
)


def evidence(cid, angle, *, present=True, analyzed=True, source=False):
    radians = np.radians(angle)
    return {'capture_id': cid, 'status': 'source' if source else 'matched_provisional',
            'thermal_geometry_accepted': True,
            'observation': {'assessment_status': 'analyzed' if analyzed else 'not_assessed'},
            'view_direction_enu': [float(np.sin(radians)), 0, float(np.cos(radians))],
            'component_evidence': 'present' if present else 'not_detected'}


def test_non_detection_does_not_clear_glare_or_count_as_clean():
    rows = [evidence('source', 0, source=True), evidence('target', 35, present=False)]
    result = assess_glare_evidence(rows)
    assert result["status"] == "unresolved"
    assert result["candidate_view_count"] == 1
    assert result["not_detected_view_count"] == 1
    assert 'clean_view_count' not in result
    assert result["angle_span_deg"] == pytest.approx(35)


def test_multi_angle_evidence_requires_angle_diversity():
    rows = [evidence('source', 10, source=True), evidence('target', 15)]
    result = assess_glare_evidence(rows)
    assert result["status"] == "angle_diversity_insufficient"


def test_multi_angle_evidence_marks_persistent_candidate():
    rows = [evidence('source', 0, source=True), evidence('target', 40)]
    assert assess_glare_evidence(rows)["status"] == "persistent_across_angles"


def test_unassessed_missing_rays_and_duplicate_captures_are_not_evidence():
    rows = [evidence('source', 0, source=True), evidence('target', 40, analyzed=False)]
    result = assess_glare_evidence(rows)
    assert result['status'] == 'unresolved'
    assert result['unknown_view_count'] == 1
    rows[1] = evidence('target', 40)
    rows[1].pop('view_direction_enu')
    assert assess_glare_evidence(rows)['status'] == 'unresolved'
    assert assess_glare_evidence([rows[0], rows[0]])['status'] == 'insufficient_views'


def test_diverse_negative_view_does_not_make_identical_positive_angles_persistent():
    rows = [evidence('source', 0, source=True), evidence('target', 1),
            evidence('negative', 45, present=False)]
    result = assess_glare_evidence(rows)
    assert result['status'] == 'unresolved'
    assert result['candidate_angle_span_deg'] == pytest.approx(1)


def test_nadir_yaw_rotation_does_not_change_centre_ray_but_offcentre_rays_do():
    from pvdelta.geolocation import LrfPlanarProjector

    def ray(yaw, x):
        projector = LrfPlanarProjector(38, 27, 20, -90, yaw, 0, 640, 512, 45, 25, 180)
        return list(projector.pixel_view_direction_enu(x, 256))
    source, target = evidence('source', 0, source=True), evidence('target', 0)
    source['view_direction_enu'], target['view_direction_enu'] = ray(0, 320), ray(90, 320)
    assert assess_glare_evidence([source, target])['angle_span_deg'] == pytest.approx(0)
    source['view_direction_enu'], target['view_direction_enu'] = ray(0, 40), ray(0, 600)
    assert assess_glare_evidence([source, target])['status'] == 'persistent_across_angles'


def test_component_correspondence_does_not_borrow_other_hotspots_or_cold_regions():
    panel = [[0, 0], [100, 0], [100, 200], [0, 200]]
    hot = {'finding_id': 'source', 'delta_t_k': 5,
           'pixel_polygon_json': [[10, 20], [30, 20], [30, 60], [10, 60]]}
    other = dict(hot, finding_id='elsewhere', pixel_polygon_json=
                 [[60, 120], [80, 120], [80, 160], [60, 160]])
    assert component_evidence(hot, panel, [other], panel)['component_evidence'] == 'not_detected'
    assert component_evidence(hot, panel, [dict(hot, delta_t_k=-5)], panel)[
        'component_evidence'] == 'not_detected'
    scaled_panel = (np.array(panel) * 2 + 50).tolist()
    scaled_hot = dict(hot, finding_id='same', pixel_polygon_json=
                      (np.array(hot['pixel_polygon_json']) * 2 + 50).tolist())
    result = component_evidence(hot, panel, [scaled_hot], scaled_panel)
    assert result['component_evidence'] == 'present'
    assert result['matching_finding_id'] == 'same'
    assert component_evidence(hot, panel, [dict(hot, pixel_polygon_json=[])], panel)[
        'component_evidence'] == 'unknown'


def test_location_veto_rejects_other_table_and_missing_location():
    source = {'latitude': 38.0, 'longitude': 27.0}
    assert location_consistency(source, source)['location_consistent']
    assert not location_consistency(source, {'latitude':38.0001, 'longitude':27.0})['location_consistent']
    assert not location_consistency(source, {})['location_consistent']


def test_target_match_rejects_ambiguous_and_absent_panels():
    quad = np.array([[20, 20], [80, 20], [80, 100], [20, 100]], dtype=float)
    assert select_target_panel(quad, [quad + 100, quad])[0] == 1
    assert select_target_panel(quad, [quad, quad + 1])[0] is None
    assert select_target_panel(quad, [quad + 100])[0] is None


def test_nearly_coincident_shared_edge_is_matched_without_relaxing_ambiguity():
    first = np.array([[501.7977905, 226.1936646], [565.2506104, 218.4847412],
                      [581.3344116, 350.8718567], [517.8815918, 358.5807800]])
    second = np.array([[501.2779236, 221.9145966], [564.7307129, 214.2056732],
                       [581.3344116, 350.8718567], [517.8815918, 358.5807800]])
    index, overlap = select_target_panel(first, [second])
    assert index == 0
    assert overlap > .96
    assert select_target_panel(first, [second, first])[0] is None


def test_shared_edge_component_uses_complete_panel_relative_intersection():
    panel = [[400, 0], [640, 0], [640, 512], [400, 512]]
    source = {'finding_id': 'source', 'delta_t_k': 5, 'pixel_polygon_json':
        [[501.7977905, 226.1936646], [565.2506104, 218.4847412],
         [581.3344116, 350.8718567], [517.8815918, 358.5807800]]}
    target = {'finding_id': 'target', 'delta_t_k': 5, 'pixel_polygon_json':
        [[501.2779236, 221.9145966], [564.7307129, 214.2056732],
         [581.3344116, 350.8718567], [517.8815918, 358.5807800]]}
    result = component_evidence(source, panel, [target], panel, min_iou=.9)
    assert result['component_evidence'] == 'present'
    assert result['component_overlap'] > .96


def test_tracking_recovers_panel_after_motion_and_exposure_change(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(42)
    texture = cv2.GaussianBlur(rng.integers(30, 180, (600, 800), dtype=np.uint8), (3, 3), .6)
    rgb = np.stack([texture // 2, texture, texture // 3], axis=-1)
    source = tmp_path / 'source.png'
    target = tmp_path / 'target.png'
    Image.fromarray(rgb).save(source)
    moved = cv2.warpAffine(rgb, np.array([[1, 0, 30], [0, 1, 25]], np.float32), (800, 600))
    Image.fromarray(np.clip(moved.astype(float) * .9 + 15, 0, 255).astype(np.uint8)).save(target)
    polygon = np.array([[300, 240], [380, 240], [380, 340], [300, 340]])
    result = match_panel(source, target, polygon, [polygon + [30, 25], polygon + [200, 25]])
    assert result.status == 'matched_provisional', result
    assert result.target_index == 0
    assert result.target_corner_indices == (0, 1, 2, 3)
    assert result.check_count >= 6
    assert result.check_p95_px <= 3
    rotated_order = np.roll(polygon + [30, 25], 2, axis=0)
    rotated = match_panel(source, target, polygon, [rotated_order])
    assert rotated.target_corner_indices == (2, 3, 0, 1)


def test_no_features_does_not_invent_correspondence(tmp_path):
    from PIL import Image

    source = tmp_path / 'flat.png'
    Image.new('RGB', (400, 400), 'gray').save(source)
    polygon = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
    assert match_panel(source, source, polygon, [polygon]).status == 'unavailable'


def test_rgb_return_track_rotates_patches_and_preserves_original_files(tmp_path):
    import hashlib

    from PIL import Image

    texture = cv2.GaussianBlur(np.random.default_rng(64).integers(30, 180, (600, 800), np.uint8), (3, 3), .6)
    rgb = np.stack([texture//2, texture, texture//3], axis=-1)
    source, target = tmp_path/'source.png', tmp_path/'return.png'
    Image.fromarray(rgb).save(source)
    Image.fromarray(cv2.rotate(rgb, cv2.ROTATE_180)).save(target)
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, target)]
    polygon = np.array([[300, 220], [419, 220], [419, 379], [300, 379]])
    observed = np.array([799, 599])-polygon
    result = match_panel(source, target, polygon, [observed])
    assert result.status == 'matched_provisional', result
    assert result.iou > .98 and result.check_p95_px < .3
    assert result.target_corner_indices == (0, 1, 2, 3)
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, target)]


def test_rgb_feature_cache_rejects_changed_bytes_even_with_preserved_stat(tmp_path):
    import os

    from PIL import Image

    from pvdelta.repeat_views import _load_features

    path = tmp_path/'source.bmp'
    texture = np.random.default_rng(8).integers(30, 180, (400, 400), np.uint8)
    Image.fromarray(np.stack([texture//2, texture, texture//3], axis=-1)).save(path)
    assert _load_features(path)[1] is not None
    stat = path.stat()
    Image.new('RGB', (400, 400), '#225522').save(path)
    assert path.stat().st_size == stat.st_size
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert _load_features(path)[1] is None


def test_panel_motion_is_not_rejected_because_ground_has_parallax(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(74)
    texture = cv2.GaussianBlur(rng.integers(30, 180, (600, 800), dtype=np.uint8), (3, 3), .6)
    ground = np.stack([texture // 2, texture, texture // 3], axis=-1)
    panel = cv2.GaussianBlur(rng.integers(30, 200, (160, 120), dtype=np.uint8), (3, 3), .6)
    panel = np.stack([panel // 2, panel // 2, panel], axis=-1)
    source_rgb = ground.copy()
    source_rgb[220:380, 300:420] = panel
    # The ground initializes motion at (+40,+25); the raised panel moves (+30,+25).
    target_rgb = cv2.warpAffine(ground, np.array([[1, 0, 40], [0, 1, 25]], np.float32), (800, 600))
    target_rgb[245:405, 330:450] = panel
    source, target = tmp_path / 'source.png', tmp_path / 'target.png'
    Image.fromarray(source_rgb).save(source)
    Image.fromarray(target_rgb).save(target)
    polygon = np.array([[300, 220], [419, 220], [419, 379], [300, 379]])
    result = match_panel(source, target, polygon, [polygon + [30, 25]])
    assert result.status == 'matched_provisional', result
    assert result.check_p95_px <= 3
    assert result.iou > .95
    # A tempting adjacent panel is not a substitute when the actual panel is absent.
    wrong = match_panel(source, target, polygon, [polygon + [150, 25]])
    assert wrong.status == 'unavailable'
    assert wrong.target_index is None


def test_repeat_evidence_does_not_borrow_another_modules_stronger_finding(tmp_path, monkeypatch):
    pytest.importorskip('duckdb')
    from pvdelta import repeat_views
    from pvdelta.store import connect_store

    database = tmp_path / 'flight.duckdb'
    c = connect_store(database)
    c.execute("INSERT INTO inspection_runs(run_id,source_root) VALUES ('flight', ?)", [str(tmp_path)])
    for index, cid in enumerate(['source', 'target']):
        c.execute("""INSERT INTO captures(capture_id,run_id,source_path,source_hash,image_kind,
            capture_time_utc,paired_capture_id,camera_model,width_px,height_px,
            laser_target_latitude,laser_target_longitude,laser_target_distance_m,
            gimbal_pitch_deg,gimbal_yaw_deg,gimbal_roll_deg)
            VALUES (?, 'flight', ?, ?, 'thermal', ?, ?, 'M4T',640,512,38,27,20,-90,90,0)
        """, [cid, cid+'.jpg', cid, f'2026-01-01 12:00:0{index}+00', cid+'-rgb'])
        c.execute("""INSERT INTO captures(capture_id,source_path,source_hash,image_kind)
            VALUES (?,?,?,'rgb')""", [cid+'-rgb',cid+'-rgb.jpg',cid+'-rgb'])
    polygon = [[200,200],[250,200],[250,300],[200,300]]
    for oid,cid,mid in [('source-1','source','M001'),('target-1','target','M001'),
                        ('target-2','target','M002')]:
        c.execute("""INSERT INTO module_observations(observation_id,capture_id,frame_module_id,
            pixel_polygon_json,visibility,analysis_run_id,assessment_status)
            VALUES (?,?,?,?,'full','analysis','analyzed')""", [oid,cid,mid,json.dumps(polygon)])
    for fid,oid,delta in [('source-f','source-1',3),('unrelated-hot','target-2',90)]:
        c.execute("""INSERT INTO findings(finding_id,observation_id,analysis_run_id,anomaly_type,
            assessment_status,delta_t_k,absolute_temperature_c,pixel_polygon_json)
            VALUES (?,?,'analysis','hot','review_required',?,95,?)""", [fid,oid,delta,json.dumps(polygon)])
    c.execute('CREATE TABLE flight_screening_frames(analysis_run_id VARCHAR,capture_id VARCHAR,diagnostics_json JSON)')
    diagnostic = {'heldout_p90_px':1,'modules':[{'module':'M001','accepted':True,
                  'rgb_polygon':polygon,'native_polygon':polygon}]}
    for cid in ['source','target']:
        c.execute('INSERT INTO flight_screening_frames VALUES (?,?,?)', ['analysis',cid,json.dumps(diagnostic)])
    c.close()
    profile = tmp_path / 'site.json'
    profile.write_text(json.dumps({'tables':{'surface_tilt_deg':25,'surface_facing_azimuth_true_deg':180}}))
    monkeypatch.setattr(repeat_views, 'match_panel', lambda *args: repeat_views.PanelMatch(
        'matched_provisional','test image correspondence',target_index=0,
        target_corner_indices=(0, 1, 2, 3)))
    rows = repeat_views.build_repeat_evidence(database,'source-f',site_profile=profile)
    assert len(rows) == 2
    assert rows[0]['status'] == 'source'
    assert rows[0]['findings'][0]['finding_id'] == 'source-f'
    assert rows[1]['status'] == 'matched_provisional'
    assert rows[1]['location_consistent']
    assert rows[1]['observation']['observation_id'] == 'target-1'
    assert rows[1]['findings'] == []  # M002's 90 K candidate must not replace M001 evidence.
    assert rows[1]['component_evidence'] == 'not_detected'
    assert rows[1]['glare_summary']['status'] != 'persistent_across_angles'

    manual = repeat_views.nearby_image_pairs(database, 'source-f')
    assert [r['capture_id'] for r in manual] == ['source', 'target']
    assert all(r['panel_identity_verified'] is False and r['glare_evidence'] == 'unresolved' for r in manual)
    assert all('native_polygon' not in r and 'view_direction_enu' not in r for r in manual)


@pytest.mark.parametrize('timed', [True, False])
def test_manual_pairs_stay_in_acquisition_and_time_window_without_geometry(tmp_path, timed):
    from pvdelta.repeat_views import nearby_image_pairs
    from pvdelta.store import connect_store

    database = tmp_path/'manual.duckdb'
    c = connect_store(database)
    c.execute("INSERT INTO inspection_runs(run_id,source_root) VALUES ('flight',?)", [str(tmp_path)])
    c.execute("INSERT INTO inspection_runs(run_id,source_root) VALUES ('other',?)", [str(tmp_path)])
    for cid, acq, seconds in [('source', 'flight', 0), ('near', 'flight', 10), ('far', 'flight', 40), ('other', 'other', 1)]:
        c.execute('''INSERT INTO captures(capture_id,run_id,source_path,source_hash,image_kind,capture_time_utc,paired_capture_id)
            VALUES (?,?,?,?,'thermal',?,?)''', [cid, acq, cid+'_T.JPG', cid,
            f'2026-01-01 12:00:{seconds:02d}+00' if timed else None, cid+'-rgb'])
        c.execute("INSERT INTO captures(capture_id,source_path,source_hash,image_kind) VALUES (?,?,?,'rgb')",
            [cid+'-rgb', cid+'_V.JPG', cid+'-rgb'])
    c.execute("INSERT INTO module_observations(observation_id,capture_id,frame_module_id,pixel_polygon_json,visibility,analysis_run_id) VALUES ('obs','source','N001','[]','full','native')")
    c.execute("INSERT INTO findings(finding_id,observation_id,analysis_run_id,anomaly_type,assessment_status,delta_t_k,absolute_temperature_c,pixel_polygon_json) VALUES ('finding','obs','native','hot','review_required',3,30,'[]')")
    c.close()
    rows = nearby_image_pairs(database, 'finding')
    assert [r['capture_id'] for r in rows] == (['source', 'near'] if timed else ['source'])
    assert all(r['status'] == 'unavailable' and r['glare_evidence'] == 'unresolved' for r in rows)
