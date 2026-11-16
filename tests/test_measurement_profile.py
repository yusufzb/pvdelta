import csv
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from pvdelta.cli import decode_frame
from pvdelta.decode import MeasurementParams
from pvdelta.measurement_profile import (
    _digest,
    apply_saved_measurement_profile,
    frame_measurement_profile,
    read_measurement_profile,
    validate_overrides,
    verify_measurement_profile,
)
from pvdelta.models import ThermalFrame
from pvdelta.pipeline import execute_stage1
from pvdelta.regions import ModuleAnnotation
from pvdelta.store import connect_store, upsert_finding_review, write_stage1_results


def frame(path, *, distance=5.):
    parameters = asdict(MeasurementParams(5., 50., .95, 20., 20.))
    values = np.full((30, 30), 25., np.float32)
    values[12:16, 12:16] = 35.
    return ThermalFrame('test', str(path), hashlib.sha256(path.read_bytes()).hexdigest(), values,
        None, 30, 30, raw_metadata={'measurement_params': {**parameters, 'distance_m': distance},
            'recorded_measurement_params': parameters,
            'measurement_overrides': {'distance_m': distance} if distance != 5 else {},
            'sdk_parameter_readback_verified': True, 'private_device_serial': 'MUST-NOT-EXPORT'})


def roi():
    return [ModuleAnnotation('M01', ((3., 3.), (26., 3.), (26., 26.), (3., 26.)))]


def test_profile_hash_binds_effective_values_and_whitelists_private_metadata(tmp_path):
    source = tmp_path/'source_T.JPG'
    source.write_bytes(b'original')
    original, corrected = frame(source), frame(source, distance=19.)
    first, second = frame_measurement_profile(original), frame_measurement_profile(corrected)
    assert first['profile_id'] != second['profile_id']
    assert second['recorded_parameters']['distance_m'] == 5
    assert second['effective_parameters']['distance_m'] == 19
    assert 'MUST-NOT-EXPORT' not in json.dumps(second)
    assert not second['field_conditions_verified']
    assert verify_measurement_profile(second) == second
    second['effective_parameters']['distance_m'] = 20
    with pytest.raises(ValueError, match='identity'):
        verify_measurement_profile(second)


def test_missing_original_or_effective_parameters_are_not_guessed(tmp_path):
    source = tmp_path/'source_T.JPG'
    source.write_bytes(b'original')
    item = frame(source)
    item.raw_metadata.clear()
    profile = frame_measurement_profile(item)
    assert profile['effective_parameters'] is profile['recorded_parameters'] is None
    assert not profile['sdk_parameter_readback_verified']


@pytest.mark.parametrize('data', [{'height': 19}, {'emissivity': float('inf')},
    {'distance_m': True}, {'humidity_pct': '50'}, []])
def test_invalid_overrides_do_not_reach_sdk(data):
    with pytest.raises(ValueError):
        validate_overrides(data)


def test_report_csv_and_database_keep_per_run_profiles_without_changing_capture(tmp_path):
    source = tmp_path/'source_T.JPG'
    source.write_bytes(b'original')
    database = tmp_path/'test.duckdb'
    first = execute_stage1(frame(source), roi(), output_dir=tmp_path/'first',
        run_id='recorded', database_path=database)
    second = execute_stage1(frame(source, distance=19.), roi(), output_dir=tmp_path/'second',
        run_id='corrected', database_path=database)
    records = [json.loads(line) for line in second.report_artifacts.jsonl_path.read_text().splitlines()]
    profile = records[0]['radiometry']
    assert profile['recorded_parameters']['distance_m'] == 5
    assert profile['effective_parameters']['distance_m'] == 19
    with second.report_artifacts.csv_path.open(newline='') as stream:
        row = next(csv.DictReader(stream))
    assert float(row['measurement_distance_m']) == 19
    assert row['measurement_profile_id'] == profile['profile_id']
    connection = connect_store(database, read_only=True)
    try:
        cid = 'sha256:'+frame(source).source_hash
        assert read_measurement_profile(connection, 'corrected', cid) == profile
        assert read_measurement_profile(connection, 'recorded', cid)['effective_parameters']['distance_m'] == 5
        assert connection.execute('SELECT object_distance_m FROM captures').fetchone()[0] == 5
        assert connection.execute('SELECT count(*) FROM module_observations').fetchone()[0] == 2
        assert first.store_summary is not None
    finally:
        connection.close()


def test_first_corrected_run_does_not_insert_corrected_values_as_original_summary(tmp_path):
    source = tmp_path/'source_T.JPG'
    source.write_bytes(b'original')
    database = tmp_path/'test.duckdb'
    execute_stage1(frame(source, distance=19.), roi(), output_dir=tmp_path/'out',
                   run_id='corrected', database_path=database)
    connection = connect_store(database, read_only=True)
    try:
        row = connection.execute('SELECT object_distance_m,temperature_median_c FROM captures').fetchone()
        assert row == (5., None)
    finally:
        connection.close()


def test_changed_analysis_archives_reviews_but_identical_rerun_preserves_them(tmp_path):
    source = tmp_path/'source_T.JPG'
    source.write_bytes(b'original')
    database = tmp_path/'test.duckdb'
    f = frame(source)
    result = execute_stage1(f, roi(), output_dir=tmp_path/'out', run_id='one', database_path=database)
    connection = connect_store(database)
    try:
        fid = connection.execute('SELECT finding_id FROM findings').fetchone()[0]
        upsert_finding_review(connection, finding_id=fid, review_status='ok', reviewer='test', note='old evidence')
        arguments = {'analysis_run_id': 'one', 'capture_id': 'sha256:'+f.source_hash,
            'module_results': result.module_results, 'measurement_profile': frame_measurement_profile(f)}
        write_stage1_results(connection, **arguments)
        assert connection.execute('SELECT count(*) FROM finding_reviews').fetchone()[0] == 1
        assert connection.execute('SELECT count(*) FROM invalidated_finding_reviews').fetchone()[0] == 0
        arguments['measurement_profile'] = frame_measurement_profile(frame(source, distance=19.))
        write_stage1_results(connection, **arguments)
        assert connection.execute('SELECT count(*) FROM finding_reviews').fetchone()[0] == 0
        assert connection.execute('SELECT reviewer,note FROM invalidated_finding_reviews').fetchone() == ('test', 'old evidence')
        assert read_measurement_profile(connection, 'one', 'sha256:'+f.source_hash)['effective_parameters']['distance_m'] == 19
    finally:
        connection.close()


def test_invalid_profile_or_source_binding_cannot_replace_existing_results(tmp_path):
    source = tmp_path/'source_T.JPG'
    source.write_bytes(b'original')
    database = tmp_path/'test.duckdb'
    f = frame(source)
    result = execute_stage1(f, roi(), output_dir=tmp_path/'out', run_id='one', database_path=database)
    connection = connect_store(database)
    try:
        original = frame_measurement_profile(f)
        changed = {**original, 'source_sha256': 'other source'}
        changed['profile_id'] = 'sha256:'+_digest({k: v for k, v in changed.items() if k != 'profile_id'})
        with pytest.raises(ValueError, match='source'):
            write_stage1_results(connection, analysis_run_id='one', capture_id='sha256:'+f.source_hash,
                module_results=result.module_results, measurement_profile=changed)
        assert read_measurement_profile(connection, 'one', 'sha256:'+f.source_hash) == original
    finally:
        connection.close()


def test_transaction_rolls_back_profile_results_and_review_invalidation(tmp_path):
    source = tmp_path/'source_T.JPG'
    source.write_bytes(b'original')
    database = tmp_path/'test.duckdb'
    f = frame(source)
    result = execute_stage1(f, roi(), output_dir=tmp_path/'out', run_id='one', database_path=database)
    connection = connect_store(database)
    fid = connection.execute('SELECT finding_id FROM findings').fetchone()[0]
    upsert_finding_review(connection, finding_id=fid, review_status='ok', reviewer='test')

    class FailsOnce:
        failed = False

        def execute(self, sql, *args):
            if 'INSERT INTO findings' in sql and not self.failed:
                self.failed = True
                raise RuntimeError('injected write failure')
            return connection.execute(sql, *args)

    try:
        with pytest.raises(RuntimeError, match='injected'):
            write_stage1_results(FailsOnce(), analysis_run_id='one', capture_id='sha256:'+f.source_hash,
                module_results=result.module_results,
                measurement_profile=frame_measurement_profile(frame(source, distance=19.)))
        assert read_measurement_profile(connection, 'one', 'sha256:'+f.source_hash)['effective_parameters']['distance_m'] == 5
        assert connection.execute('SELECT count(*) FROM finding_reviews').fetchone()[0] == 1
        assert connection.execute('SELECT count(*) FROM invalidated_finding_reviews').fetchone()[0] == 0
        assert connection.execute('SELECT count(*) FROM findings').fetchone()[0] == 1
    finally:
        connection.close()


def test_old_read_only_store_without_profile_table_is_unknown():
    duckdb = pytest.importorskip('duckdb')
    connection = duckdb.connect(':memory:')
    try:
        assert read_measurement_profile(connection, 'unknown', 'capture') is None
        assert not connection.execute("SELECT table_name FROM information_schema.tables").fetchall()
    finally:
        connection.close()


@pytest.mark.sdk
@pytest.mark.skipif(not (os.environ.get('PVDELTA_DIRP_LIB') and os.environ.get('PVDELTA_SAMPLE_RJPEG')),
                    reason='DJI SDK and original source required')
def test_real_sdk_corrected_analysis_roundtrips_report_database_and_overlay(tmp_path):
    from pvdelta.decode import RJpeg
    from pvdelta.finding_preview import thermal_finding_preview

    source = Path(os.environ['PVDELTA_SAMPLE_RJPEG'])
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    item = decode_frame(source, os.environ['PVDELTA_DIRP_LIB'], measurement_overrides={'distance_m': 19.})
    profile = frame_measurement_profile(item)
    with RJpeg(source) as reader:
        values = apply_saved_measurement_profile(reader, source, profile)
    np.testing.assert_array_equal(values, item.temperature_c)
    assert profile['recorded_parameters']['distance_m'] == 5
    assert profile['effective_parameters']['distance_m'] == 19
    result = execute_stage1(item, roi(), output_dir=tmp_path/'out',
        run_id='real-corrected', database_path=tmp_path/'real.duckdb')
    record = json.loads(result.report_artifacts.jsonl_path.read_text().splitlines()[0])
    assert record['radiometry'] == profile
    preview = thermal_finding_preview(source, [[10., 10.], [15., 10.], [15., 15.], [10., 15.]],
        sdk_library=os.environ['PVDELTA_DIRP_LIB'], measurement_profile=profile)
    assert preview.overlay_applied and 'native hash checked' in preview.message
    invalid = {**profile, 'native_temperature_sha256': '0'*64}
    invalid['profile_id'] = 'sha256:'+_digest({k: v for k, v in invalid.items() if k != 'profile_id'})
    preview = thermal_finding_preview(source, [[10., 10.], [15., 10.], [15., 15.], [10., 15.]],
        sdk_library=os.environ['PVDELTA_DIRP_LIB'], measurement_profile=invalid)
    assert not preview.overlay_applied
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
