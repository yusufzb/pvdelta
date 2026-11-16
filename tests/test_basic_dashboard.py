"""Exercise the user workflow, including read-only opening and explicit scan/review writes."""
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest

from pvdelta.store import CaptureRecord, connect_store, upsert_capture, upsert_run


def _store(path):
    from PIL import Image

    root = path.parent
    for kind in ('T', 'V'):
        Image.new('RGB', (64, 51), 'gray').save(root / f'example_{kind}.JPG')
    connection = connect_store(path)
    upsert_run(connection, run_id='photos', source_root=str(root), started_at_utc=None,
               site_label=None, camera_model=None)
    upsert_run(connection, run_id='analysis', source_root='', started_at_utc=None,
               site_label=None, camera_model=None)
    for kind, capture, pair in (('thermal', 'thermal', 'rgb'), ('rgb', 'rgb', 'thermal')):
        upsert_capture(connection, CaptureRecord(capture, f"example_{'T' if kind == 'thermal' else 'V'}.JPG",
            'test-sha', kind, run_id='photos', paired_capture_id=pair, width_px=64, height_px=51))
    polygon = json.dumps([[5, 5], [20, 5], [20, 30], [5, 30]])
    for index in range(2):
        observation, finding = f'obs-{index}', f'candidate-{index}'
        connection.execute('''INSERT INTO module_observations (observation_id,capture_id,frame_module_id,
            pixel_polygon_json,visibility,analysis_run_id,assessment_status,temperature_min_c,
            temperature_max_c,temperature_mean_c,temperature_median_c)
            VALUES (?, 'thermal', ?, ?, 'full','analysis','analyzed',25,40,30,29)''',
            [observation, f'M{index}', polygon])
        connection.execute('''INSERT INTO findings (finding_id,observation_id,anomaly_type,
            assessment_status,delta_t_k,absolute_temperature_c,reference_temperature_c,
            pixel_polygon_json,analysis_run_id) VALUES (?,?,'hot','review_required',10,40,30,?,'analysis')''',
            [finding, observation, polygon])
    connection.close()
    return path


def _app(database, tmp_path):
    return AppTest.from_string(
        'from pvdelta.basic_dashboard import basic_dashboard\n'
        f'basic_dashboard({str(database)!r}, "missing-sdk", output_root={str(tmp_path / "scans")!r}, review_version="individual")\n',
        default_timeout=30)


@pytest.mark.parametrize('version', ['individual', 'panel'])
def test_legacy_candidate_locations_reach_thermal_and_rgb_in_both_review_versions(tmp_path, monkeypatch, version):
    import numpy as np
    from PIL import Image

    from pvdelta import decode, rgb_preview

    database = _store(tmp_path/'legacy.duckdb')
    connection = connect_store(database)
    for capture, suffix in [('thermal', 'T'), ('rgb', 'V')]:
        connection.execute('UPDATE captures SET source_hash=? WHERE capture_id=?',
            [hashlib.sha256((tmp_path/f'example_{suffix}.JPG').read_bytes()).hexdigest(), capture])
    connection.close()
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    calls = []

    class Reader:
        def __init__(self, *args):
            calls.append('native')

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def processed_rgb(self):
            return np.full((51, 64, 3), 30, np.uint8)

    def rgb(path, polygon, **kwargs):
        assert kwargs['native_verified'] is True
        assert kwargs['measurement_profile'] is None
        calls.append('rgb')
        return rgb_preview.RGBPreview(Image.new('RGB', (64, 51)), True,
            'Approximate legacy location', 'approximate', ((5, 5), (20, 5), (20, 30), (5, 30)))

    monkeypatch.setattr(decode, 'RJpeg', Reader)
    monkeypatch.setattr(rgb_preview, 'rgb_finding_preview', rgb)
    app = AppTest.from_string('from pvdelta.basic_dashboard import basic_dashboard\n'
        f'basic_dashboard({str(database)!r}, "local-sdk", review_version={version!r})\n',
        default_timeout=30).run()
    assert not app.exception and not app.error
    assert calls == ['native', 'rgb']
    assert any('Legacy analysis: showing saved candidate locations' in w.value for w in app.warning)
    assert not any('overlay withheld' in w.value for w in app.warning)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('wrong_run', [False, True])
def test_saved_inspection_switch_is_exact_read_only_and_does_not_carry_unsaved_notes(tmp_path, wrong_run):
    for name in ('first','second'):
        (tmp_path/name).mkdir()
    first = _store(tmp_path/'first/inspection.duckdb')
    second = _store(tmp_path/'second/inspection.duckdb')
    connection = connect_store(second)
    connection.execute("DELETE FROM findings WHERE finding_id='candidate-1'")
    connection.close()
    library = tmp_path/'inspection-library.json'
    library.write_text(json.dumps({'schema':'pvdelta.inspection-library/v1','inspections':[
        {'id':'legacy','label':'First inspection','database_path':str(first),'analysis_run_id':'analysis'},
        {'id':'native','label':'Second inspection','database_path':str(second),
         'analysis_run_id':'nonexistent-run' if wrong_run else 'analysis'},
    ]}))
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in (first,second)]
    app = AppTest.from_string('from pvdelta.basic_dashboard import basic_dashboard\n'
        f'basic_dashboard({str(first)!r}, "missing-sdk", library_path={str(library)!r}, '
        'review_version="individual")\n',default_timeout=30).run()
    assert not app.exception and not app.error
    assert {m.label:m.value for m in app.metric}['Candidates to review'] == '2'
    app.text_input(key=f'basic-note-{first}-candidate-0').set_value('Unsaved first inspection note').run()
    app.radio(key='basic-review-filter').set_value('Wait').run()
    app.selectbox(key='basic-saved-inspection').set_value('native').run()
    assert not app.exception and app.text_input(key='basic-database').value == str(second)
    if wrong_run:
        assert any('does not match an analysis' in e.value for e in app.error)
        assert not app.metric  # Do not substitute the unrelated run or its candidate list.
    else:
        assert not app.error and app.selectbox(key='basic-analysis').value == 'analysis'
        assert app.radio(key='basic-review-filter').value == 'All'
        assert {m.label:m.value for m in app.metric}['Candidates to review'] == '1'
        assert app.text_input(key=f'basic-note-{second}-candidate-0').value == ''
        app.selectbox(key='basic-saved-inspection').set_value('legacy').run()
        assert not app.exception and not app.error
        assert app.text_input(key=f'basic-note-{first}-candidate-0').value == ''
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in (first,second)] == hashes


def test_open_is_read_only_navigation_review_and_filter_work(tmp_path):
    database = _store(tmp_path / 'inspection.duckdb')
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    app = _app(database, tmp_path).run()
    assert not app.exception and not app.error
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    assert {m.label: m.value for m in app.metric}['Candidates to review'] == '2'
    assert len(app.get('imgs')) == 2 or len(app.get('image')) == 2
    assert any('overlay withheld' in w.value for w in app.warning)
    app.button(key='basic-candidate-next').click().run()
    assert not app.exception and app.selectbox(key='basic-candidate').value == 'candidate-1'
    app.text_input(key=f'basic-note-{database}-candidate-1').set_value('Vegetation').run()
    app.button(key='basic-mark-not_ok').click().run()
    assert not app.exception
    connection = connect_store(database, read_only=True)
    assert connection.execute('SELECT finding_id,review_status,note FROM finding_reviews').fetchall() == [
        ('candidate-1', 'not_ok', 'Vegetation')]
    connection.close()
    app.radio(key='basic-review-filter').set_value('Unreviewed').run()
    assert app.selectbox(key='basic-candidate').value == 'candidate-0'
    app.radio(key='basic-review-filter').set_value('Not OK').run()
    assert app.selectbox(key='basic-candidate').value == 'candidate-1'
    assert not app.exception


def test_new_scan_is_explicit_and_opens_new_results(tmp_path, monkeypatch):
    from pvdelta import decode, ingest, native_flight

    template = _store(tmp_path / 'template.duckdb')
    calls = []
    monkeypatch.delenv('PVDELTA_DIRP_LIB', raising=False)
    monkeypatch.setattr(decode, 'load_library', lambda sdk: calls.append(('sdk', sdk)))

    def index(folder, database, **kwargs):
        calls.append(('index', folder, str(database)))
        return SimpleNamespace(thermal_captures=1)

    def scan(database, acquisition, output, run, **kwargs):
        assert kwargs['sdk_library'] == 'missing-sdk'
        assert acquisition == run + '-photos'
        calls.append(('scan', str(database)))
        output.mkdir()
        shutil.copyfile(template, output / 'inspection.duckdb')
        return {'totals': {'completed_frames': 1, 'failed_frames': 0}}

    monkeypatch.setattr(ingest, 'index_flight_folder', index)
    monkeypatch.setattr(native_flight, 'scan_native_flight', scan)
    app = _app(tmp_path / 'not-created.duckdb', tmp_path).run()
    assert app.button(key='basic-start').disabled
    assert not calls and not (tmp_path / 'scans').exists()
    app.text_input(key='basic-flight-folder').set_value(str(tmp_path)).run()
    assert not calls
    app.button(key='basic-start').click().run()
    assert not app.exception and not app.error
    assert [c[0] for c in calls] == ['sdk', 'index', 'scan']
    assert Path(app.text_input(key='basic-database').value).is_file()
    assert app.selectbox(key='basic-analysis').value == 'analysis'
    assert not (tmp_path / 'not-created.duckdb').exists()


def test_missing_sdk_does_not_create_scan_or_index(tmp_path, monkeypatch):
    from pvdelta import decode, ingest

    def fail(sdk):
        raise RuntimeError('SDK unavailable')

    monkeypatch.setattr(decode, 'load_library', fail)
    monkeypatch.setattr(ingest, 'index_flight_folder', lambda *a, **k: pytest.fail('must not index'))
    app = _app(tmp_path / 'missing.duckdb', tmp_path).run()
    app.text_input(key='basic-flight-folder').set_value(str(tmp_path)).run()
    app.button(key='basic-start').click().run()
    assert not app.exception and 'SDK unavailable' in app.error[0].value
    assert not (tmp_path / 'scans').exists()


def test_basic_routes_selected_finding_to_marked_rgb_preview(tmp_path, monkeypatch):
    from PIL import Image

    from pvdelta import basic_dashboard, rgb_preview
    from pvdelta.finding_preview import FindingPreview

    database = _store(tmp_path / 'inspection.duckdb')
    calls = []
    monkeypatch.setattr(basic_dashboard, 'read_measurement_profile', lambda *a: {'test_profile': True})
    monkeypatch.setattr(basic_dashboard, 'thermal_finding_preview', lambda *a, **k:
        FindingPreview(Image.new('RGB', (64, 51)), True, 'Native test preview'))

    def render(path, polygon, **kwargs):
        calls.append((polygon, kwargs))
        return rgb_preview.RGBPreview(Image.new('RGB', (64, 51)), True,
            'Red: approximate projected anomaly area', 'approximate')

    monkeypatch.setattr(rgb_preview, 'rgb_finding_preview', render)
    app = _app(database, tmp_path).run()
    assert not app.exception
    assert calls[-1][1]['native_verified'] is True
    assert calls[-1][1]['expected_rgb_hash'] == 'test-sha'
    assert 'approximate projected anomaly area' in str(app.get('image')[1].proto)
    app.button(key='basic-candidate-next').click().run()
    assert not app.exception and len(calls) == 2
    assert any('Approximate RGB location' in w.value for w in app.warning)


def test_full_dashboard_defaults_basic_and_keeps_advanced_tabs(tmp_path, monkeypatch):
    from pvdelta import native_flight_ui

    monkeypatch.setenv('PVDELTA_DATABASE_PATH', str(tmp_path / 'not-created.duckdb'))
    monkeypatch.setenv('PVDELTA_GAP_REVIEW_BUNDLE', str(tmp_path / 'no-bundle'))
    outputs = []
    monkeypatch.setattr(native_flight_ui, 'native_flight_view',
                        lambda *args, output_root: outputs.append(output_root))
    source = Path(__file__).resolve().parents[1] / 'dashboard' / 'flight_planner.py'
    checkout = tmp_path / 'standalone-checkout'
    script = checkout / 'dashboard' / source.name
    script.parent.mkdir(parents=True)
    script.write_bytes(source.read_bytes())
    app = AppTest.from_file(str(script), default_timeout=60).run()
    assert not app.exception and app.radio(key='interface-mode').value == 'Basic'
    assert not any(t.label == 'Flight plan' for t in app.tabs)
    app.radio(key='interface-mode').set_value('Advanced').run()
    assert not app.exception
    assert [t.label for t in app.tabs][:10] == ['Flight plan', 'Field checklist', 'Radiometric scan',
        'Panel boundaries', 'Unassigned evidence', 'RGB quality', 'Local data & map',
        'User review', 'Orthoview', 'Export']
    assert outputs == [checkout / 'data' / 'native-inspection-runs']
    app.radio(key='interface-mode').set_value('Basic').run()
    assert not app.exception and not (tmp_path / 'not-created.duckdb').exists()
