import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest

pytest.importorskip('duckdb')

from pvdelta import cli
from pvdelta.decode import MeasurementParams
from pvdelta.measurement_profile import read_measurement_profile
from pvdelta.models import ThermalFrame
from pvdelta.screening_replay import attach_site_geometry, replay_screening, screening_runs
from pvdelta.store import CaptureRecord, connect_store, upsert_capture, upsert_run


@pytest.fixture
def source(tmp_path):
    flight = tmp_path/'flight'
    flight.mkdir()
    database = tmp_path/'original.duckdb'
    connection = connect_store(database)
    upsert_run(connection, run_id='acquisition', started_at_utc=None, source_root=str(flight), site_label=None, camera_model=None)
    upsert_run(connection, run_id='baseline', started_at_utc=None, source_root=None, site_label=None, camera_model=None)
    connection.execute('''CREATE TABLE flight_screening_frames (analysis_run_id VARCHAR,capture_id VARCHAR,
        source_name VARCHAR,status VARCHAR,rgb_candidates INTEGER,native_candidates INTEGER,
        measured INTEGER,findings INTEGER,diagnostics_json JSON)''')
    ids = []
    for name in ('A', 'B', 'C'):
        path = flight/f'{name}_T.JPG'
        path.write_bytes(name.encode())
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        cid = 'sha256:'+source_hash
        ids.append(source_hash)
        upsert_capture(connection, CaptureRecord(cid, path.name, source_hash, 'thermal',
            run_id='acquisition', width_px=80, height_px=64))
        modules = []
        for i, (x, y) in enumerate(((4, 4), (34, 4), (4, 34), (34, 34))):
            modules.append({'module': f'M{i}', 'accepted': i != 3,
                'native_polygon': [[x, y], [x+22, y], [x+22, y+25], [x, y+25]],
                'gate': {'accepted': i != 3, 'inside': True, 'margin_px': 1,
                         'max_offset_px': 1., 'max_mad_px': .2, 'min_gradient': 40.}})
        diagnostic = {'modules': modules, 'heldout_p90_px': 1., 'shift_xy': [0, 0],
                      'native_candidates': 4, 'rgb_candidates': 4}
        connection.execute('INSERT INTO flight_screening_frames VALUES (?,?,?,?,?,?,?,?,?)',
            ['baseline', cid, path.name, 'provisional', 4, 4, 3, 1, json.dumps(diagnostic)])
    connection.close()
    calls = []

    def decoder(path, sdk_library, *, measurement_overrides=None):
        calls.append(path.name)
        if path.name.startswith('B'):
            raise RuntimeError('corrupt test source')
        recorded = asdict(MeasurementParams(5., 50., .95, 20., 20.))
        effective = {**recorded, **(measurement_overrides or {})}
        values = np.full((64, 80), 25.+effective['distance_m']*.01, np.float32)
        values[12:17, 12:17] += 10
        return ThermalFrame(path.stem, str(path), hashlib.sha256(path.read_bytes()).hexdigest(), values,
            None, 80, 64, raw_metadata={'recorded_measurement_params': recorded,
                'measurement_params': effective, 'measurement_overrides': measurement_overrides or {}})

    return database, ids, decoder, calls


def test_all_sources_replayed_without_promoting_rejected_geometry_or_altering_source(source, tmp_path):
    database, hashes, decoder, calls = source
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    progress = []
    result = replay_screening(database, 'baseline', tmp_path/'out', 'replayed', decoder=decoder,
        measurement_overrides={'distance_m': 19.}, progress=lambda *args: progress.append(args))
    assert calls == ['A_T.JPG', 'B_T.JPG', 'C_T.JPG']
    assert len(progress) == 3 and result['status'] == 'partial'
    assert result['totals']['decoded_frames'] == 2
    assert result['totals']['analyzed_observations'] == 6
    assert result['totals']['not_assessed_observations'] == 2
    assert result['totals']['failed_frames'] == 1
    assert result['physical_panel_count'] is None and not result['geometry_semantically_verified']
    connection = connect_store(tmp_path/'out/inspection.duckdb', read_only=True)
    try:
        profile = read_measurement_profile(connection, 'replayed', 'sha256:'+hashes[0])
        assert profile['effective_parameters']['distance_m'] == 19.
        assert profile['recorded_parameters']['distance_m'] == 5.
        assert connection.execute('SELECT count(*) FROM module_observations').fetchone()[0] == 8
        assert connection.execute('SELECT count(*) FROM measurement_profiles').fetchone()[0] == 2
        assert connection.execute('SELECT count(*) FROM flight_screening_frames').fetchone()[0] == 3
    finally:
        connection.close()
    run = json.loads((tmp_path/'out/frames/0001/replayed.jsonl').read_text().splitlines()[0])
    assert run['radiometry']['profile_id'] == profile['profile_id']
    assert not run['settings']['geometry_semantically_verified']
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_per_frame_profiles_are_exact_source_bound_and_missing_sources_are_not_guessed(source, tmp_path):
    database, hashes, decoder, calls = source
    with pytest.raises(ValueError, match='exactly'):
        replay_screening(database, 'baseline', tmp_path/'wrong', 'new', decoder=decoder,
                         per_frame_overrides={hashes[0]: {'distance_m': 19.}})
    assert not calls and not (tmp_path/'wrong').exists()
    result = replay_screening(database, 'baseline', tmp_path/'out', 'new', decoder=decoder,
        per_frame_overrides={h: {'distance_m': 16.+i} for i, h in enumerate(hashes)})
    connection = connect_store(tmp_path/'out/inspection.duckdb', read_only=True)
    try:
        assert read_measurement_profile(connection, 'new', 'sha256:'+hashes[0])['effective_parameters']['distance_m'] == 16.
        assert read_measurement_profile(connection, 'new', 'sha256:'+hashes[2])['effective_parameters']['distance_m'] == 18.
    finally:
        connection.close()
    assert result['status'] == 'partial'


def test_inconsistent_stored_acceptance_does_not_authorize_measurement(source, tmp_path):
    database, _hashes, decoder, _calls = source
    connection = connect_store(database)
    records = connection.execute('SELECT capture_id,diagnostics_json FROM flight_screening_frames').fetchall()
    for cid, raw in records:
        data = json.loads(raw)
        data['heldout_p90_px'] = 20.
        connection.execute('UPDATE flight_screening_frames SET diagnostics_json=? WHERE capture_id=?', [json.dumps(data), cid])
    connection.close()
    result = replay_screening(database, 'baseline', tmp_path/'out', 'new', decoder=decoder)
    assert result['totals']['analyzed_observations'] == 0
    assert result['totals']['not_assessed_observations'] == 8
    assert result['totals']['finding_records'] == 0


def test_replay_requires_new_output_and_run_id(source, tmp_path):
    database, _hashes, decoder, _calls = source
    assert screening_runs(database) == ['baseline']
    with pytest.raises(ValueError, match='new run ID'):
        replay_screening(database, 'baseline', tmp_path/'wrong', 'baseline', decoder=decoder)
    out = tmp_path/'exists'
    out.mkdir()
    with pytest.raises(FileExistsError):
        replay_screening(database, 'baseline', out, 'new', decoder=decoder)


def test_site_prior_and_repeat_cache_follow_actual_analysis_state(source, tmp_path):
    from pvdelta.repeat_views import repeat_evidence_signature
    from pvdelta.store import upsert_finding_review

    database, _hashes, decoder, _calls = source
    out = tmp_path/'out'
    replay_screening(database, 'baseline', out, 'new', decoder=decoder)
    site = tmp_path/'site.json'
    specification = {'schema': 'pvdelta.site-geometry-prior/v1', 'tables': {
        'surface_tilt_deg': 25., 'surface_facing_azimuth_true_deg': 180.}}
    site.write_text(json.dumps(specification))
    result = attach_site_geometry(out, site)
    assert result['site_geometry_is_user_prior_not_field_validation']
    target = out/'inspection.duckdb'
    connection = connect_store(target)
    fid = connection.execute('SELECT finding_id FROM findings ORDER BY finding_id LIMIT 1').fetchone()[0]
    connection.close()
    first = repeat_evidence_signature(target, fid, out/'site-geometry.json')
    connection = connect_store(target)
    upsert_finding_review(connection, finding_id=fid, review_status='wait', reviewer='test')
    connection.close()
    assert repeat_evidence_signature(target, fid, out/'site-geometry.json') == first
    connection = connect_store(target)
    connection.execute("UPDATE measurement_profiles SET result_fingerprint='changed' WHERE analysis_run_id='new'")
    connection.close()
    second = repeat_evidence_signature(target, fid, out/'site-geometry.json')
    assert second != first
    copied = json.loads((out/'site-geometry.json').read_bytes())
    copied['tables']['surface_tilt_deg'] = 26.
    (out/'site-geometry.json').write_text(json.dumps(copied))
    assert repeat_evidence_signature(target, fid, out/'site-geometry.json') != second
    with pytest.raises(FileExistsError):
        attach_site_geometry(out, site)


def test_repeat_signature_changes_on_source_file_change(source, tmp_path):
    from pvdelta.repeat_views import repeat_evidence_signature

    database, _hashes, decoder, _calls = source
    out = tmp_path/'out'
    replay_screening(database, 'baseline', out, 'new', decoder=decoder)
    site = tmp_path/'site.json'
    site.write_text('{}')
    connection = connect_store(out/'inspection.duckdb', read_only=True)
    fid = connection.execute('SELECT finding_id FROM findings ORDER BY finding_id LIMIT 1').fetchone()[0]
    connection.close()
    first = repeat_evidence_signature(out/'inspection.duckdb', fid, site)
    (tmp_path/'flight/A_T.JPG').write_bytes(b'changed original file')
    assert repeat_evidence_signature(out/'inspection.duckdb', fid, site) != first


def test_cli_returns_partial_with_full_frame_failure_evidence(source, tmp_path, monkeypatch, capsys):
    database, _hashes, decoder, _calls = source
    monkeypatch.setattr(cli, 'decode_frame', decoder)
    assert cli.main(['replay-screening', '--database', str(database), '--baseline-run', 'baseline',
        '--run-id', 'new', '--out', str(tmp_path/'out')]) == 3
    result = json.loads(capsys.readouterr().out)
    assert result['totals']['failed_frames'] == 1 and 'frames' not in result


def test_replay_ui_executes_real_engine_and_clears_stale_success_on_failure(source, tmp_path, monkeypatch):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest

    import pvdelta.screening_replay_ui as ui

    database, _hashes, decoder, _calls = source
    real = replay_screening
    monkeypatch.setattr(ui, 'replay_screening', lambda *args, **kwargs: real(*args, decoder=decoder, **kwargs))
    app = AppTest.from_string('from pvdelta.screening_replay_ui import screening_replay_view\n'
        f'screening_replay_view({str(database)!r},None,output_root={str(tmp_path/"runs")!r},key="test")',
        default_timeout=30).run()

    def find(items, label):
        return next(e for e in items if e.label == label)

    assert not app.exception
    find(app.button, 'Run radiometric anomaly analysis').click().run()
    assert not app.exception and not app.error
    assert app.session_state['test-result'][1]['status'] == 'partial'
    find(app.button, 'Run radiometric anomaly analysis').click().run()
    assert not app.exception and app.error
    assert 'test-result' not in app.session_state
