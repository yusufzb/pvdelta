import json

import pytest

duckdb = pytest.importorskip('duckdb')

from pvdelta import cli
from pvdelta.coverage_audit import audit_frame, build_coverage_audit, read_coverage_audit
from pvdelta.store import initialize_schema


def test_native_geometry_does_not_invent_an_rgb_failure_or_legacy_registration_score():
    raw = {'geometry_method': 'native_observed_boundaries', 'rgb_candidates': None,
        'native_candidates': 2, 'measured': 2, 'findings': 1,
        'modules': [{'accepted': True}, {'accepted': True}]}
    result = audit_frame(('A', 'native_T.JPG', 'provisional_native_measurements', None, 2, 2, 1, json.dumps(raw)))
    assert result['stage'] == 'native_provisional_measurements'
    assert result['legacy_frame_gate'] == 'not_applicable_native_geometry'
    assert result['legacy_edge_p90_px'] is None
    assert result['issues'] == []
    assert result['accepted_provisional_geometry'] == 2


def test_native_empty_geometry_is_unassessed_and_counter_mismatch_is_still_detected():
    raw = {'geometry_method': 'native_observed_boundaries', 'native_candidates': 0,
           'measured': 0, 'findings': 0, 'modules': []}
    result = audit_frame(('A', 'native_T.JPG', 'no_assessable_modules', None, 0, 0, 0, json.dumps(raw)))
    assert result['stage'] == 'no_native_structural_proposals' and result['issues'] == []
    result = audit_frame(('A', 'native_T.JPG', 'provisional_native_measurements', None, 0, 1, 0, json.dumps(raw)))
    assert 'measurement_without_native_structure' in result['issues']


def setup_store(connection, *, screening=True):
    initialize_schema(connection)
    for run in ('acq-A', 'acq-B', 'analysis-A', 'analysis-B'):
        connection.execute('INSERT INTO inspection_runs(run_id) VALUES (?)', [run])
    for cid, acq in [('A1', 'acq-A'), ('A2', 'acq-A'), ('B1', 'acq-B')]:
        connection.execute('''INSERT INTO captures(capture_id,run_id,source_path,source_hash,image_kind)
                              VALUES (?,?,'private.JPG',?,'thermal')''', [cid, acq, cid])
    for oid, cid, run, status in [('O1', 'A1', 'analysis-A', 'analyzed'),
                                   ('O2', 'A1', 'analysis-A', 'not_assessed'),
                                   ('O4', 'A1', 'analysis-A', 'not_assessed'),
                                   ('O5', 'A1', 'analysis-A', 'not_assessed'),
                                   ('O3', 'A2', 'analysis-A', 'analyzed'),
                                   ('OB', 'B1', 'analysis-B', 'analyzed')]:
        connection.execute('''INSERT INTO module_observations(observation_id,capture_id,
            frame_module_id,pixel_polygon_json,visibility,analysis_run_id,assessment_status,
            temperature_min_c,temperature_max_c,temperature_mean_c,temperature_median_c)
            VALUES (?,?,'M1','[]','full',?,?,20,30,25,25)''', [oid, cid, run, status])
    for fid, oid, run in [('F1', 'O1', 'analysis-A'), ('F2', 'O1', 'analysis-A'), ('FB', 'OB', 'analysis-B')]:
        connection.execute('''INSERT INTO findings(finding_id,observation_id,anomaly_type,
             assessment_status,delta_t_k,absolute_temperature_c,pixel_polygon_json,analysis_run_id)
             VALUES (?,?,'hot','review_required',4,30,'[]',?)''', [fid, oid, run])
    connection.execute("INSERT INTO finding_reviews(finding_id,review_status) VALUES ('F1','not_ok')")
    if screening:
        connection.execute('''CREATE TABLE flight_screening_frames (
            analysis_run_id VARCHAR,capture_id VARCHAR,source_name VARCHAR,status VARCHAR,
            rgb_candidates INTEGER,native_candidates INTEGER,measured INTEGER,findings INTEGER,
            diagnostics_json VARCHAR)''')
        connection.execute('INSERT INTO flight_screening_frames VALUES (?,?,?,?,?,?,?,?,?)',
                           ['analysis-A', *screening_row()])


def screening_row(*, rgb=10, native=4, measured=1, p90=2, shift=None, modules=None,
                  status='screened_provisional', error=None):
    if modules is None:
        modules = [
            {'accepted': True, 'gate': {'accepted': True, 'inside': True, 'max_offset_px': 1, 'max_mad_px': 0, 'min_gradient': 30}},
            *[{'accepted': False, 'gate': {'accepted': False, 'inside': True, 'max_offset_px': 6, 'max_mad_px': 0, 'min_gradient': 30}}]*3,
        ]
    raw = {'rgb_candidates': rgb, 'native_candidates': native, 'measured': measured,
           'findings': 2, 'modules': modules, 'error': error}
    if p90 is not None:
        raw.update(heldout_p90_px=p90, shift_xy=shift or [0, 0])
    return ('A1', 'synthetic_T.JPG', status, rgb, native, measured, 2, json.dumps(raw))


@pytest.fixture
def connection():
    con = duckdb.connect(':memory:')
    setup_store(con)
    yield con
    con.close()


def test_acquisition_and_analysis_are_scoped_and_duplicate_findings_do_not_inflate_panels(connection):
    before = connection.execute('SELECT count(*) FROM module_observations').fetchone()
    audit = build_coverage_audit(connection, 'analysis-A')
    assert audit['scope']['indexed_thermal_captures'] == 2
    assert audit['scope']['acquisition_run_ids'] == ['acq-A']
    assert audit['counts']['module_observations'] == 5
    assert audit['counts']['analyzed_observations'] == 2
    assert audit['counts']['analyzed_without_finding'] == 1
    assert audit['counts']['observations_with_findings'] == 1
    assert audit['counts']['finding_records'] == 2
    assert audit['counts']['scoped_captures_without_screening_record'] == 1
    assert audit['finding_review_counts'] == {'not_ok': 1, 'unreviewed': 1}
    assert audit['physical_panel_count'] is None
    assert audit['independently_verified_mask_count'] is None
    assert connection.execute('SELECT count(*) FROM module_observations').fetchone() == before


def test_legacy_store_without_screening_table_remains_read_only():
    con = duckdb.connect(':memory:')
    setup_store(con, screening=False)
    audit = build_coverage_audit(con, 'analysis-A')
    assert audit['counts']['screening_rows'] == 0
    assert audit['counts']['rgb_proposal_records'] is None
    assert audit['frame_stages'] == {}
    assert not con.execute("SELECT 1 FROM information_schema.tables WHERE table_name='flight_screening_frames'").fetchone()
    con.close()


def test_unknown_or_unlinked_acquisition_is_unknown_not_a_fully_processed_flight(connection):
    connection.execute("UPDATE captures SET run_id=NULL WHERE capture_id IN ('A1','A2')")
    audit = build_coverage_audit(connection, 'analysis-A')
    assert audit['scope']['indexed_thermal_captures'] is None
    assert not audit['scope']['scope_complete']
    assert audit['data_issues']['acquisition_scope_incomplete'] == 1


def test_nonfinite_statistics_do_not_become_no_finding_healthy_counts(connection):
    connection.execute("UPDATE module_observations SET temperature_mean_c='NaN'::DOUBLE WHERE observation_id='O3'")
    audit = build_coverage_audit(connection, 'analysis-A')
    assert audit['counts']['analyzed_without_finding'] == 0
    assert audit['data_issues']['analyzed_observations_missing_finite_temperature_statistics'] == 1
    json.dumps(audit, allow_nan=False)


def test_counter_mismatch_and_missing_capture_reference_are_reported(connection):
    connection.execute("UPDATE flight_screening_frames SET measured=2")
    connection.execute("DELETE FROM captures WHERE capture_id='A1'")
    audit = build_coverage_audit(connection, 'analysis-A')
    assert audit['data_issues']['counter_mismatch_measured'] == 1
    assert audit['data_issues']['measurement_count_disagrees_with_observations'] == 1
    assert audit['data_issues']['unindexed_thermal_capture_references'] == 1


@pytest.mark.parametrize('raw', ['{broken', 'null', '[]'])
def test_malformed_diagnostics_are_unavailable_not_healthy(raw):
    row = list(screening_row())
    row[-1] = raw
    report = audit_frame(row)
    assert report['stage'] == 'diagnostics_unavailable'
    assert report['issues'] == ['invalid_diagnostics']


def test_rejection_reasons_overlap_but_frame_stage_is_exclusive():
    modules = [{'accepted': False, 'gate': {'inside': False, 'max_offset_px': 7,
                                           'max_mad_px': 4, 'min_gradient': 10}}]
    report = audit_frame(screening_row(native=1, measured=0, p90=8, modules=modules))
    assert report['stage'] == 'frame_registration_rejected'
    assert report['module_rejection_reasons'] == {
        'frame_registration_rejected': 1, 'outside_native_margin': 1,
        'edge_offset_above_5px': 1, 'edge_spread_above_2px': 1, 'weak_frame_gradient': 1}


@pytest.mark.parametrize('row,stage', [
    (screening_row(rgb=0, native=0, measured=0, p90=None, modules=[]), 'no_rgb_proposals'),
    (screening_row(native=0, measured=0, p90=None, modules=[]), 'no_full_native_proposals'),
    (screening_row(native=4, measured=0, p90=None, modules=[]), 'frame_registration_unavailable'),
    (screening_row(native=1, measured=0, p90=None, modules=[]), 'insufficient_native_proposals_for_legacy_registration'),
    (screening_row(native=1, measured=0, shift=[20, 0]), 'frame_registration_rejected'),
    (screening_row(status='deferred_oblique_geometry', measured=0), 'deferred_geometry'),
    (screening_row(error='SDK failure', measured=0), 'processing_error'),
])
def test_frame_funnel_stages(row, stage):
    assert audit_frame(row)['stage'] == stage


def test_unknown_run_is_not_silently_a_zero_result(connection):
    with pytest.raises(ValueError, match='does not exist'):
        build_coverage_audit(connection, 'missing')


def test_cli_read_only_export_protects_database_and_existing_reports(tmp_path, capsys):
    database = tmp_path / 'source.duckdb'
    connection = duckdb.connect(str(database))
    setup_store(connection)
    connection.close()
    before = database.read_bytes()
    output = tmp_path / 'audit.json'
    args = ['coverage-audit', '--database', str(database), '--run-id', 'analysis-A', '--out', str(output)]
    assert cli.main(args) == 0
    result = json.loads(output.read_text())
    assert result['counts']['finding_records'] == 2
    assert database.read_bytes() == before
    assert cli.main(args) == 1
    assert cli.main(args[:-1]+[str(database), '--overwrite']) == 1
    assert database.read_bytes() == before
    capsys.readouterr()


def test_missing_database_is_not_created(tmp_path):
    missing = tmp_path / 'missing.duckdb'
    with pytest.raises(ValueError, match='Cannot audit'):
        read_coverage_audit(missing, 'anything')
    assert not missing.exists()


def test_exclusive_gate_combinations_do_not_double_count_rejected_observations():
    good = {'accepted': True, 'inside': True, 'max_offset_px': 1, 'max_mad_px': 0, 'min_gradient': 30}
    bad = {**good, 'accepted': False, 'max_offset_px': 7}
    report = audit_frame(screening_row(native=4, measured=0, p90=8,
        modules=[{'accepted': False, 'gate': gate} for gate in [good, bad, good, bad]]))
    assert report['module_gate_states'] == {
        'frame_registration_only_rejected': 2, 'frame_registration_and_edges_rejected': 2}
    assert sum(report['module_gate_states'].values()) == 4
    assert sum(report['module_rejection_reasons'].values()) > 4


def test_accepted_flag_does_not_conceal_contradictory_gate_values():
    module = {'accepted': True, 'gate': {'accepted': False, 'inside': True,
              'max_offset_px': 1, 'max_mad_px': 0, 'min_gradient': 30}}
    report = audit_frame(screening_row(modules=[module], native=4))
    assert 'module_edge_gate_inconsistent' in report['issues']
    assert 'accepted_geometry_without_passing_module_gate' in report['issues']


def test_legacy_four_module_fit_requirement_cannot_be_satisfied_by_two_modules():
    report = audit_frame(screening_row(native=2))
    assert report['legacy_frame_gate'] == 'rejected'
    assert 'accepted_geometry_without_passing_legacy_frame_gate' in report['issues']


def test_priority_queue_does_not_turn_edge_passes_into_measurement_acceptance(connection):
    row = screening_row(measured=0, p90=8, modules=[{'accepted': False, 'gate': {
        'accepted': True, 'inside': True, 'max_offset_px': 1, 'max_mad_px': 0, 'min_gradient': 30}}]*4)
    connection.execute('DELETE FROM flight_screening_frames')
    connection.execute('INSERT INTO flight_screening_frames VALUES (?,?,?,?,?,?,?,?,?)', ['analysis-A', *row])
    audit = build_coverage_audit(connection, 'analysis-A')
    assert audit['priority_review_frames'][0]['edge_pass_frame_reject_observations'] == 4
    assert audit['counts']['provisional_geometry_passes'] == 0
    assert audit['independently_verified_mask_count'] is None


@pytest.mark.parametrize('p90', [float('nan'), float('inf'), -1])
def test_invalid_registration_scores_do_not_leak_into_json_or_become_passing_geometry(p90):
    report = audit_frame(screening_row(p90=p90))
    assert report['legacy_frame_gate'] == 'unknown'
    assert report['legacy_edge_p90_px'] is None
    assert 'invalid_legacy_registration_metrics' in report['issues']
    json.dumps(report, allow_nan=False)


def test_boolean_diagnostic_count_is_not_an_integer_measurement_count():
    row = list(screening_row())
    diagnostic = json.loads(row[-1])
    diagnostic['measured'] = True
    row[-1] = json.dumps(diagnostic)
    assert 'invalid_diagnostic_measured_counter' in audit_frame(row)['issues']
