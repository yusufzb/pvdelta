import datetime as dt
import hashlib

import pytest

from pvdelta.repeat_candidates import _distance, select_repeat_pairs
from pvdelta.store import CaptureRecord, connect_store, upsert_capture, upsert_run


@pytest.fixture
def database(tmp_path):
    path = tmp_path/'captures.duckdb'
    c = connect_store(path)
    for run in ('flight', 'other-flight', 'analysis'):
        upsert_run(c, run_id=run, source_root=str(tmp_path), started_at_utc=None, site_label=None, camera_model=None)
    for name, seconds, north_m, run in [('source', 0, 0, 'flight'), ('time', 1, 4, 'flight'),
        ('return', 40, 2, 'flight'), ('outside', 2, 30, 'flight'), ('other', 2, .1, 'other-flight'),
        ('unknown', 1, None, 'flight'), ('tie-b', 50, 3, 'flight'), ('tie-a', -50, 3, 'flight')]:
        timestamp = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)+dt.timedelta(seconds=seconds)
        for kind in ('thermal', 'rgb'):
            cid = name if kind == 'thermal' else name+'-rgb'
            upsert_capture(c, CaptureRecord(cid, cid+'.JPG', 'sha-'+cid, kind, run_id=run,
                capture_time_utc=timestamp, paired_capture_id=name+'-rgb' if kind == 'thermal' else name,
                camera_latitude=40+north_m/111_195 if north_m is not None else None,
                camera_longitude=26 if north_m is not None else None))
    c.execute("""INSERT INTO module_observations
        (observation_id,capture_id,frame_module_id,pixel_polygon_json,visibility,analysis_run_id)
        VALUES ('observation','source','M001','[[1,1],[2,1],[2,2],[1,2]]','full','analysis')""")
    c.execute("""INSERT INTO findings
        (finding_id,observation_id,anomaly_type,assessment_status,delta_t_k,absolute_temperature_c,
         reference_temperature_c,pixel_polygon_json,analysis_run_id)
        VALUES ('finding','observation','hot','review_required',10,40,30,
        '[[1,1],[2,1],[2,2],[1,2]]','analysis')""")
    c.close()
    return path


def test_gps_finds_return_track_beyond_time_window_and_never_crosses_acquisition(database):
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    rows = select_repeat_pairs(database, 'finding', max_views=4)
    assert [r['capture_id'] for r in rows] == ['source', 'return', 'tie-a', 'tie-b', 'time']
    assert rows[1]['time_gap_s'] == 40 and rows[1]['camera_distance_m'] == pytest.approx(2, abs=.01)
    assert all(r['search_method'] == 'camera_gps' and not r['panel_identity_verified'] for r in rows)
    assert all(r['glare_evidence'] == 'unresolved' for r in rows)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    time_rows = select_repeat_pairs(database, 'finding', search_mode='time')
    assert 'return' not in [r['capture_id'] for r in time_rows]


def test_missing_source_gps_falls_back_explicitly_and_missing_target_gps_is_not_zero(database):
    assert 'unknown' not in [r['capture_id'] for r in select_repeat_pairs(database, 'finding')]
    c = connect_store(database)
    c.execute("UPDATE captures SET camera_latitude=NULL,camera_longitude=NULL WHERE capture_id='source'")
    c.close()
    rows = select_repeat_pairs(database, 'finding')
    assert all(r['search_method'] == 'time_fallback_missing_source_gps' for r in rows)
    assert 'return' not in [r['capture_id'] for r in rows]


def test_missing_time_can_still_be_retrieved_by_gps_and_cap_is_deterministic(database):
    c = connect_store(database)
    c.execute("UPDATE captures SET capture_time_utc=NULL WHERE capture_id='return'")
    c.close()
    rows = select_repeat_pairs(database, 'finding', max_views=1)
    assert [r['capture_id'] for r in rows] == ['source', 'return']
    assert rows[1]['time_gap_s'] is None


def test_outside_root_and_wrong_pair_acquisition_are_rejected(database):
    c = connect_store(database)
    c.execute("UPDATE captures SET run_id='other-flight' WHERE capture_id='source-rgb'")
    c.close()
    with pytest.raises(ValueError, match='Source has no thermal/RGB pair'):
        select_repeat_pairs(database, 'finding')
    c = connect_store(database)
    c.execute("UPDATE captures SET run_id='flight' WHERE capture_id='source-rgb'")
    c.execute("UPDATE captures SET source_path='../outside.JPG' WHERE capture_id='return'")
    c.close()
    with pytest.raises(ValueError, match='outside its source root'):
        select_repeat_pairs(database, 'finding')


def test_haversine_handles_dateline_and_rejects_non_coordinates():
    assert _distance(0., 179.99999, 0., -179.99999) == pytest.approx(2.224, abs=.01)
    for lat, lon in ((None, 26.), (float('nan'), 26.), (91., 26.), (40., 181.), (True, 26.)):
        assert _distance(lat, lon, 40., 26.) is None


@pytest.mark.parametrize('kwargs', [{'max_views': True}, {'max_views': 33}, {'search_mode': 'guess'},
    {'max_distance_m': float('inf')}, {'max_distance_m': 0}, {'max_distance_m': True}])
def test_invalid_search_settings_fail_before_query(database, kwargs):
    with pytest.raises(ValueError):
        select_repeat_pairs(database, 'finding', **kwargs)
