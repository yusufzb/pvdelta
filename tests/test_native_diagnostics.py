import hashlib
import json

import pytest

duckdb = pytest.importorskip('duckdb')
from pvdelta.native_diagnostics import read_native_diagnostic


@pytest.fixture
def sample(tmp_path):
    path = tmp_path/'DJI_1166_T.JPG'
    path.write_bytes(b'original encoded frame')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    c = duckdb.connect(':memory:')
    c.execute('CREATE TABLE captures(capture_id VARCHAR,source_path VARCHAR,source_hash VARCHAR,width_px INT,height_px INT)')
    c.execute('CREATE TABLE flight_screening_frames(analysis_run_id VARCHAR,capture_id VARCHAR,source_name VARCHAR,diagnostics_json VARCHAR)')
    c.execute('CREATE TABLE module_observations(analysis_run_id VARCHAR,capture_id VARCHAR,frame_module_id VARCHAR,pixel_polygon_json VARCHAR)')
    q = [[2, 2], [12, 2], [12, 18], [2, 18]]
    diagnostic = {'geometry_method': 'native_observed_boundaries', 'native_candidates': 1,
                  'native_geometry': {'proposals': [{'proposal_id': 'N001', 'polygon_native_xy': q}]}}
    c.execute('INSERT INTO captures VALUES (?,?,?,?,?)', ['capture-hash', str(path), digest, 20, 24])
    c.execute('INSERT INTO flight_screening_frames VALUES (?,?,?,?)', ['run', 'capture-hash', path.name, json.dumps(diagnostic)])
    c.execute('INSERT INTO module_observations VALUES (?,?,?,?)', ['run', 'capture-hash', 'N001', json.dumps(q)])
    yield c, path, diagnostic
    c.close()


def test_uses_capture_not_filename_sequence_and_reads_without_writes(sample):
    c, path, _ = sample
    before = c.execute('SELECT * FROM flight_screening_frames').fetchall()
    result = read_native_diagnostic(c, 'run', 'capture-hash', source_file=path)
    assert result['source_name'] == 'DJI_1166_T.JPG'
    assert result['source_bytes_verified'] and result['observations_verified']
    assert result['database_written'] is False and result['measurement_authorised'] is False
    assert c.execute('SELECT * FROM flight_screening_frames').fetchall() == before
    result['diagnostic']['native_geometry']['proposals'].clear()
    assert len(read_native_diagnostic(c, 'run', 'capture-hash')['diagnostic']['native_geometry']['proposals']) == 1


@pytest.mark.parametrize('run,capture', [('different', 'capture-hash'), ('run', 'different'), ('', 'capture-hash')])
def test_exact_run_and_capture_required(sample, run, capture):
    with pytest.raises(ValueError):
        read_native_diagnostic(sample[0], run, capture)


def test_changed_original_bytes_rejected(sample):
    c, path, _ = sample
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Original source bytes'):
        read_native_diagnostic(c, 'run', 'capture-hash', source_file=path)


def test_wrong_frame_diagnostic_even_with_same_module_id_rejected(sample):
    c, _, d = sample
    d['native_geometry']['proposals'][0]['polygon_native_xy'][0][0] += 1
    c.execute('UPDATE flight_screening_frames SET diagnostics_json=?', [json.dumps(d)])
    with pytest.raises(ValueError, match='polygon differs'):
        read_native_diagnostic(c, 'run', 'capture-hash')


def test_source_name_count_and_duplicated_diagnostic_rejected(sample):
    c, path, d = sample
    c.execute("UPDATE flight_screening_frames SET source_name='other_T.JPG'")
    with pytest.raises(ValueError, match='metadata'):
        read_native_diagnostic(c, 'run', 'capture-hash')
    c.execute('UPDATE flight_screening_frames SET source_name=?', [path.name])
    d['native_candidates'] = 0
    c.execute('UPDATE flight_screening_frames SET diagnostics_json=?', [json.dumps(d)])
    with pytest.raises(ValueError, match='proposals differ'):
        read_native_diagnostic(c, 'run', 'capture-hash')
    c.execute('INSERT INTO flight_screening_frames SELECT * FROM flight_screening_frames')
    with pytest.raises(ValueError, match='One native diagnostic'):
        read_native_diagnostic(c, 'run', 'capture-hash')


def test_zero_proposals_is_valid_diagnostic_not_a_healthy_label(sample):
    c, _, d = sample
    d['native_geometry']['proposals'] = []
    d['native_candidates'] = 0
    c.execute('DELETE FROM module_observations')
    c.execute('UPDATE flight_screening_frames SET diagnostics_json=?', [json.dumps(d)])
    r = read_native_diagnostic(c, 'run', 'capture-hash')
    assert r['observations_verified'] and not r['source_bytes_verified']
    assert 'healthy' not in r and not r['measurement_authorised']
