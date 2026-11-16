import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest

pytest.importorskip('duckdb')

from pvdelta.decode import MeasurementParams
from pvdelta.measurement_profile import read_measurement_profile
from pvdelta.models import ThermalFrame
from pvdelta.native_flight import analyze_native_frame, native_acquisitions, scan_native_flight
from pvdelta.regions import polygon_to_mask
from pvdelta.store import CaptureRecord, connect_store, upsert_capture, upsert_run
from pvdelta.support_partition import partition_measurement_support


def _proposals(values, *, raw_supplement):
    assert raw_supplement
    return {'proposals': [{'proposal_id': 'N1', 'polygon_native_xy': [[3, 3], [28, 3], [28, 50], [3, 50]]},
        {'proposal_id': 'N2', 'polygon_native_xy': [[23, 3], [48, 3], [48, 50], [23, 50]]}]}


@pytest.fixture
def native_source(tmp_path):
    folder = tmp_path/'flight'
    folder.mkdir()
    db = tmp_path/'source.duckdb'
    c = connect_store(db)
    upsert_run(c, run_id='flight', source_root=str(folder), started_at_utc=None, site_label=None, camera_model=None)
    for name in ('A', 'B', 'C'):
        path = folder/f'{name}_T.JPG'
        path.write_bytes(name.encode())
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        upsert_capture(c, CaptureRecord('sha256:'+sha, path.name, sha, 'thermal', run_id='flight',
            width_px=64, height_px=64))
    c.close()
    calls = []

    def decoder(path, sdk, *, measurement_overrides=None):
        calls.append(path.name)
        if path.name == 'B_T.JPG':
            raise RuntimeError('corrupt source')
        recorded = asdict(MeasurementParams(5., 50., .95, 20., 20.))
        effective = recorded | (measurement_overrides or {})
        values = np.full((64, 64), 25., np.float32)
        values[15:20, 10:15] = 35.
        return ThermalFrame(path.stem, str(path), hashlib.sha256(path.read_bytes()).hexdigest(),
            values, None, 64, 64, raw_metadata={'measurement_params': effective,
                'recorded_measurement_params': recorded, 'measurement_overrides': measurement_overrides or {},
                'sdk_parameter_readback_verified': True})

    return db, decoder, calls


def test_native_scan_has_no_rgb_or_old_geometry_prerequisite_and_preserves_sources(native_source, tmp_path):
    database, decoder, calls = native_source
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    assert native_acquisitions(database) == ['flight']
    out = tmp_path/'native'
    result = scan_native_flight(database, 'flight', out, 'native-1', decoder=decoder,
        proposal_builder=_proposals, measurement_overrides={'distance_m': 19.})
    assert result['status'] == 'partial'
    assert calls == ['A_T.JPG', 'B_T.JPG', 'C_T.JPG']
    assert result['totals']['decoded_frames'] == 2
    assert result['totals']['failed_frames'] == 1
    assert result['totals']['analyzed_observations'] == 4
    assert result['totals']['finding_records'] == 2
    assert result['physical_panel_count'] is None and not result['geometry_semantically_verified']
    c = connect_store(out/'inspection.duckdb', read_only=True)
    try:
        assert c.execute("SELECT DISTINCT assessment_status FROM findings").fetchall() == [('review_required',)]
        cid = result['frames'][0]['capture_id']
        profile = read_measurement_profile(c, 'native-1', cid)
        assert profile['recorded_parameters']['distance_m'] == 5.
        assert profile['effective_parameters']['distance_m'] == 19.
        assert c.execute('SELECT count(*) FROM finding_reviews').fetchone()[0] == 0
    finally:
        c.close()
    first = json.loads((out/'frames/0001/native-1.jsonl').read_text().splitlines()[0])
    assert first['radiometry'] == profile
    diagnostic = json.loads((out/'frames/0001/native-geometry.json').read_bytes())
    assert 'heldout_p90_px' not in diagnostic and 'shift_xy' not in diagnostic
    assert diagnostic['rgb_candidates'] is None and not diagnostic['rgb_registration_required']
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    from pvdelta.screening_replay import replay_screening, screening_runs

    assert screening_runs(out/'inspection.duckdb') == []
    with pytest.raises(ValueError, match='scan-native-flight'):
        replay_screening(out/'inspection.duckdb', 'native-1', tmp_path/'wrong-replay', 'different')
    assert not (tmp_path/'wrong-replay').exists()


def test_flight_surface_index_is_source_specific_and_preserves_unreferenced_frames(native_source,tmp_path,monkeypatch):
    from pvdelta.candidate_triage import native_support_masks
    from pvdelta.measurement_profile import frame_measurement_profile

    database,decoder,_=native_source
    original=tmp_path/'flight'/'A_T.JPG'
    frame=decoder(original,'sdk')
    reference={'case_id':'C001','review_complete':True,'source_sha256':frame.source_hash,
        'native_temperature_sha256':frame_measurement_profile(frame)['native_temperature_sha256'],
        'native_height':64,'native_width':64,'panel_polygons':[],
        'non_panel_polygons':[{'polygon_native_xy':[[0,0],[64,0],[64,64],[0,64]]}]}
    monkeypatch.setattr('pvdelta.native_flight.configured_surface_index',lambda path:('f'*64,{frame.source_hash:reference}))
    result=scan_native_flight(database,'flight',tmp_path/'guarded','guarded',
        decoder=decoder,proposal_builder=_proposals,surface_manifest=tmp_path/'explicit-manifest.json')
    assert result['surface_constraints']['matching_source_frames']==1
    assert result['surface_constraints']['unreferenced_source_frames']==2
    outcomes={r['source_name']:r for r in result['frames']}
    assert outcomes['A_T.JPG']['measured']==0 and outcomes['A_T.JPG']['findings']==0
    assert outcomes['A_T.JPG']['status']=='no_assessable_modules'
    assert outcomes['C_T.JPG']['measured']>0
    snapshot_path = tmp_path/'guarded'/'surface-references.json'
    connection = connect_store(tmp_path/'guarded'/'inspection.duckdb', read_only=True)
    try:
        capture = outcomes['A_T.JPG']['capture_id']
        masks = native_support_masks(connection, 'guarded', capture, 1)
        assert masks and all(not mask.any() for mask in masks.values())
        snapshot = json.loads(snapshot_path.read_bytes())
        snapshot['references'][frame.source_hash]['non_panel_polygons'] = []
        snapshot_path.write_text(json.dumps(snapshot), encoding='utf-8')
        with pytest.raises(ValueError, match='differs from analysis binding'):
            native_support_masks(connection, 'guarded', capture, 1)
    finally:
        connection.close()


def test_shared_envelope_pixels_are_excluded_without_moving_or_warping_temperatures(native_source):
    database, decoder, _ = native_source
    c = connect_store(database, read_only=True)
    root = c.execute("SELECT source_root FROM inspection_runs WHERE run_id='flight'").fetchone()[0]
    c.close()
    from pathlib import Path

    frame = decoder(Path(root)/'A_T.JPG', None)
    original = frame.temperature_c.copy()
    reports, diagnostic = analyze_native_frame(frame, proposal_builder=_proposals)
    proposals = _proposals(original, raw_supplement=True)['proposals']
    full = [polygon_to_mask(p['polygon_native_xy'], original.shape) for p in proposals]
    supports = partition_measurement_support(full, margin_px=2)
    assert not (supports[0].interior & supports[1].interior).any()
    assert any(m['ambiguous_pixels_excluded'] > 0 for m in diagnostic['modules'])
    for report, support in zip(reports, supports):
        assert report.module.temp_mean == pytest.approx(float(original[support.interior].mean()))
    np.testing.assert_array_equal(frame.temperature_c, original)


def test_geometry_failure_remains_unassessed_and_later_files_run(native_source, tmp_path):
    database, decoder, calls = native_source

    def broken(values, **kwargs):
        raise ValueError('no usable native structure')

    result = scan_native_flight(database, 'flight', tmp_path/'native', 'new', decoder=decoder, proposal_builder=broken)
    assert result['status'] == 'failed' and len(calls) == 3
    assert result['totals']['decoded_frames'] == 2
    assert result['totals']['failed_frames'] == 3
    assert result['totals'].get('finding_records', 0) == 0
    assert all(row['status'] == 'failed' and 'error' in row for row in result['frames'])
    from pvdelta.screening_replay import screening_runs

    assert screening_runs(tmp_path/'native/inspection.duckdb') == []


def test_changed_source_is_not_analysed_and_empty_frame_is_not_healthy(native_source, tmp_path):
    from pathlib import Path

    database, decoder, calls = native_source
    (tmp_path/'flight/A_T.JPG').write_bytes(b'changed')
    result = scan_native_flight(database, 'flight', tmp_path/'native', 'new', decoder=decoder,
        proposal_builder=lambda *a, **k: {'proposals': []})
    assert calls == ['B_T.JPG', 'C_T.JPG']
    assert 'differs from indexed source' in result['frames'][0]['error']
    assert result['frames'][2]['status'] == 'no_assessable_modules'
    assert result['totals']['module_observations'] == 0
    assert Path(tmp_path/'native/frames/0003/native-geometry.json').is_file()


@pytest.mark.parametrize('workers', [0, 5, True])
def test_invalid_worker_count_cannot_create_output(native_source, tmp_path, workers):
    database, decoder, _ = native_source
    with pytest.raises(ValueError, match='workers'):
        scan_native_flight(database, 'flight', tmp_path/'out', 'new', workers=workers, decoder=decoder)
    assert not (tmp_path/'out').exists()


def test_native_cli_returns_partial_and_uses_explicit_acquisition(native_source, tmp_path, monkeypatch, capsys):
    from pvdelta import cli, native_flight

    database, decoder, _ = native_source
    real = native_flight.scan_native_flight
    monkeypatch.setattr(native_flight, 'scan_native_flight', lambda *args, **kwargs:
        real(*args, **kwargs, decoder=decoder, proposal_builder=_proposals))
    status = cli.main(['scan-native-flight', '--database', str(database), '--acquisition', 'flight',
        '--run-id', 'native-cli', '--out', str(tmp_path/'out')])
    assert status == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload['schema'] == 'pvdelta.native-flight/v1' and payload['totals']['failed_frames'] == 1


def test_native_dashboard_runs_engine_and_clears_stale_success_after_failed_attempt(native_source, tmp_path, monkeypatch):
    import inspect

    from streamlit.testing.v1 import AppTest

    from pvdelta import native_flight

    database, decoder, _ = native_source
    real = native_flight.scan_native_flight
    monkeypatch.setattr(native_flight, 'scan_native_flight', lambda *args, **kwargs:
        real(*args, **kwargs, decoder=decoder, proposal_builder=_proposals))
    monkeypatch.setenv('NATIVE_TEST_DATABASE', str(database))
    monkeypatch.setenv('NATIVE_TEST_OUTPUT', str(tmp_path/'private'))
    script = tmp_path/'app.py'
    script.write_text(inspect.getsource(_native_dashboard_app)+'\n_native_dashboard_app()\n', encoding='utf-8')
    app = AppTest.from_file(str(script)).run()
    app.button(key='native-flight-start').click().run()
    assert not app.exception
    assert app.session_state['native-flight-result'][1]['totals']['analyzed_observations'] == 4
    assert app.warning
    app.button(key='native-flight-start').click().run()
    assert not app.exception and app.error
    assert 'native-flight-result' not in app.session_state


def _native_dashboard_app():
    import os

    from pvdelta.native_flight_ui import native_flight_view

    native_flight_view(os.environ['NATIVE_TEST_DATABASE'], None, output_root=os.environ['NATIVE_TEST_OUTPUT'])


def test_native_support_preview_checks_temperatures_and_actual_mask_hash(native_source, monkeypatch):
    from copy import deepcopy
    from pathlib import Path

    from PIL import Image

    from pvdelta import decode
    from pvdelta.finding_preview import native_frame_preview
    from pvdelta.measurement_profile import frame_measurement_profile

    database, decoder, _ = native_source
    source = database.parent/'flight/A_T.JPG'
    Image.new('RGB', (128, 128), 'gray').save(source)
    frame = decoder(source, None)
    _reports, geometry = analyze_native_frame(frame, proposal_builder=_proposals)
    profile = frame_measurement_profile(frame)

    class FakeSDK:
        measurement_params = MeasurementParams(**profile['effective_parameters'])

        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def temperature(self):
            return frame.temperature_c

        def processed_rgb(self):
            return np.zeros((64, 64, 3), np.uint8)

    monkeypatch.setattr(decode, 'RJpeg', FakeSDK)
    shown = native_frame_preview(Path(source), geometry, profile)
    assert shown.overlay_applied and shown.image.size == (64, 64)
    assert shown.image.getpixel((25, 20)) == (255, 0, 255)
    changed = deepcopy(geometry)
    changed['modules'][0]['mask_sha256'] = 'wrong'
    withheld = native_frame_preview(source, changed, profile)
    assert not withheld.overlay_applied and withheld.image.size == (128, 128)
    assert 'support differs' in withheld.message
