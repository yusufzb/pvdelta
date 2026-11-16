import csv
import hashlib
import json

import duckdb
import pytest
from test_coverage_audit import setup_store

from pvdelta import cli
from pvdelta.inspection_export import export_inspection, verify_inspection_export


@pytest.fixture
def database(tmp_path):
    path = tmp_path/'input.duckdb'
    connection = duckdb.connect(str(path))
    setup_store(connection)
    connection.close()
    return path


def load(path):
    return json.loads(path.read_bytes())


def test_export_scopes_all_tables_and_preserves_source_and_review_semantics(database, tmp_path):
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    out = tmp_path/'handover'
    result = export_inspection(database, 'analysis-A', out)
    payload = load(out/'inspection.json')
    assert result['counts']['module_observations'] == 5
    assert result['counts']['finding_records'] == 2
    assert {f['capture_id'] for f in payload['frames']} == {'A1', 'A2'}
    assert {f['finding_id'] for f in payload['findings']} == {'F1', 'F2'}
    assert payload['findings'][0]['review_status'] == 'not_ok'
    assert payload['findings'][1]['review_status'] is None
    assert payload['findings'][0]['assessment_status'] == 'review_required'
    assert payload['healthy_panel_count'] is None and payload['physical_panel_count'] is None
    assert payload['confirmed_fault_count'] is None
    assert payload['coverage']['counts']['analyzed_without_finding'] == 1
    assert (out/'review.html').read_text(encoding='utf-8').count('<details>') == 1
    assert verify_inspection_export(out)['files_verified'] == 7
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_csv_formulas_neutralized_html_escaped_and_json_original_preserved(database, tmp_path):
    note = '=1+1<script>alert(1)</script>'
    connection = duckdb.connect(str(database))
    connection.execute('UPDATE finding_reviews SET note=?', [note])
    connection.close()
    out = tmp_path/'handover'
    export_inspection(database, 'analysis-A', out)
    assert load(out/'inspection.json')['findings'][0]['review_note'] == note
    with (out/'findings.csv').open(encoding='utf-8', newline='') as stream:
        assert next(csv.DictReader(stream))['review_note'] == "'"+note
    page = (out/'review.html').read_text(encoding='utf-8')
    assert '<script>' not in page and '&lt;script&gt;' in page


def test_optional_review_and_profile_tables_remain_unknown_without_migration(database, tmp_path):
    connection = duckdb.connect(str(database))
    connection.execute('DROP TABLE finding_reviews')
    connection.execute('DROP TABLE measurement_profiles')
    connection.close()
    before = database.read_bytes()
    out = tmp_path/'handover'
    export_inspection(database, 'analysis-A', out)
    payload = load(out/'inspection.json')
    assert payload['coverage']['finding_review_counts'] is None
    assert all(f['review_status'] is None for f in payload['findings'])
    assert payload['measurement_profiles'] == []
    assert database.read_bytes() == before


def test_tampering_or_missing_manifest_entry_is_rejected(database, tmp_path):
    out = tmp_path/'handover'
    export_inspection(database, 'analysis-A', out)
    original = (out/'findings.csv').read_bytes()
    (out/'findings.csv').write_bytes(original+b'changed')
    with pytest.raises(ValueError, match='integrity'):
        verify_inspection_export(out)
    (out/'findings.csv').write_bytes(original)
    manifest = load(out/'manifest.json')
    manifest['files'].pop('findings.csv')
    (out/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='incomplete'):
        verify_inspection_export(out)


def test_no_overwrite_and_failed_export_never_publishes_partial_directory(database, tmp_path):
    out = tmp_path/'handover'
    out.mkdir()
    marker = out/'keep.txt'
    marker.write_text('preserved')
    with pytest.raises(FileExistsError): export_inspection(database, 'analysis-A', out)
    assert marker.read_text() == 'preserved'
    with pytest.raises(ValueError): export_inspection(database, 'missing-run', tmp_path/'missing')
    assert not (tmp_path/'missing').exists()
    connection = duckdb.connect(str(database))
    connection.execute("UPDATE findings SET delta_t_k='NaN'::DOUBLE WHERE finding_id='F1'")
    connection.close()
    with pytest.raises(ValueError): export_inspection(database, 'analysis-A', tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()


def test_cli_export_verify_and_missing_database_do_not_create_input(database, tmp_path, capsys):
    out = tmp_path/'handover'
    assert cli.main(['export-inspection', '--database', str(database), '--run-id', 'analysis-A',
                     '--out', str(out)]) == 0
    assert cli.main(['verify-inspection-export', str(out)]) == 0
    missing = tmp_path/'absent.duckdb'
    assert cli.main(['export-inspection', '--database', str(missing), '--run-id', 'analysis-A',
                     '--out', str(tmp_path/'bad')]) == 1
    assert not missing.exists() and not (tmp_path/'bad').exists()


def test_rgb_pair_is_same_acquisition_and_hash_bound(database, tmp_path):
    connection = duckdb.connect(str(database))
    connection.execute("INSERT INTO captures(capture_id,run_id,source_path,source_hash,image_kind) VALUES ('RGB','acq-A','paired_W.JPG','rgbhash','rgb')")
    connection.execute("UPDATE captures SET paired_capture_id='RGB' WHERE capture_id='A1'")
    connection.close()
    out = tmp_path/'handover'
    export_inspection(database, 'analysis-A', out)
    finding = load(out/'inspection.json')['findings'][0]
    assert finding['rgb_source_path'] == 'paired_W.JPG'
    assert finding['rgb_source_sha256'] == 'rgbhash'
    assert finding['thermal_source_sha256'] == 'A1'
