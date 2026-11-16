"""Read-only, run-scoped inspection coverage and legacy geometry gate audit.

Counts are observations, not physical panels. Missing verification is unknown,
and historical edge scores are not ground-truth registration error.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


def read_coverage_audit(database: str | Path, run_id: str):
    """Open an existing store read-only, including on the CLI error path."""
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Coverage audit requires pvdelta[storage]") from exc
    from pvdelta.store import connect_store

    try:
        connection = connect_store(database, read_only=True)
        try:
            return build_coverage_audit(connection, run_id)
        finally:
            connection.close()
    except duckdb.Error as exc:
        raise ValueError(f"Cannot audit database: {exc}") from exc


def _table_exists(connection, table):
    return bool(connection.execute(
        'SELECT 1 FROM information_schema.tables WHERE table_schema=? AND table_name=?',
        ['main', table]).fetchone())


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _audit_native_frame(record, diagnostic, modules):
    """Describe native-only structure without inventing legacy RGB registration scores."""
    accepted, unknown = 0, 0
    for module in modules:
        if not isinstance(module, dict) or type(module.get('accepted')) is not bool:
            unknown += 1
        elif module['accepted']:
            accepted += 1
    if unknown:
        record['issues'].append('invalid_module_decision')
    measured = record['measured_observations']
    if measured is not None and measured > accepted:
        record['issues'].append('measurement_without_native_structure')
    record.update(legacy_frame_gate='not_applicable_native_geometry',
        geometry_method='native_observed_boundaries', accepted_provisional_geometry=accepted,
        unknown_module_decisions=unknown, module_gate_states={'native_structure_only_unverified': accepted,
            'native_structure_withheld': len(modules)-accepted-unknown},
        module_rejection_reasons={})
    if diagnostic.get('error'):
        record['stage'] = 'processing_error'
    elif any(record[k] is None for k in ('native_proposals', 'measured_observations')):
        record['stage'] = 'diagnostics_unavailable'
    elif measured:
        record['stage'] = 'native_provisional_measurements'
    elif not record['native_proposals']:
        record['stage'] = 'no_native_structural_proposals'
    else:
        record['stage'] = 'native_geometry_measurement_unavailable'
    return record


def audit_frame(row):
    """Explain stored decisions. Never refit, decode or independently certify geometry."""
    capture, name, status, rgb, native, measured, findings, raw = row
    record = {'capture_id': capture, 'source_name': name, 'stored_status': status,
              'rgb_proposals': _count(rgb), 'native_proposals': _count(native),
              'measured_observations': _count(measured), 'finding_records': _count(findings),
              'issues': [], 'module_rejection_reasons': {}, 'legacy_edge_p90_px': None}
    try:
        diagnostic = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(diagnostic, dict):
            raise TypeError('diagnostic is not an object')
    except (TypeError, ValueError):
        record.update(stage='diagnostics_unavailable', issues=['invalid_diagnostics'])
        return record
    native_mode = diagnostic.get('geometry_method') == 'native_observed_boundaries'
    for key, value in [('rgb_candidates', rgb), ('native_candidates', native),
                       ('measured', measured), ('findings', findings)]:
        if native_mode and key == 'rgb_candidates' and value is None:
            continue
        if _count(value) is None:
            record['issues'].append('invalid_'+key+'_counter')
        if key in diagnostic and _count(diagnostic[key]) is None:
            record['issues'].append('invalid_diagnostic_'+key+'_counter')
        if key in diagnostic and diagnostic[key] != value:
            record['issues'].append('counter_mismatch_'+key)
    modules = diagnostic.get('modules')
    if not isinstance(modules, list):
        modules = []
        if native:
            record['issues'].append('module_diagnostics_missing')
    if _count(native) is not None and len(modules) != native:
        record['issues'].append('native_module_count_mismatch')
    if _count(rgb) is not None and _count(native) is not None and native > rgb:
        record['issues'].append('native_exceeds_rgb_counter')
    if _count(native) is not None and _count(measured) is not None and measured > native:
        record['issues'].append('measured_exceeds_native_counter')
    if native_mode:
        return _audit_native_frame(record, diagnostic, modules)
    p90 = diagnostic.get('heldout_p90_px')
    shift = diagnostic.get('shift_xy')
    fit_known = (_finite(p90) and p90 >= 0 and isinstance(shift, list) and len(shift) == 2
                 and all(_finite(v) for v in shift))
    if ('heldout_p90_px' in diagnostic or 'shift_xy' in diagnostic) and not fit_known:
        record['issues'].append('invalid_legacy_registration_metrics')
    fit_pass = fit_known and _count(native) is not None and native >= 4 and p90 <= 5 and max(abs(v) for v in shift) < 20
    record['legacy_frame_gate'] = 'passed' if fit_pass else 'rejected' if fit_known else 'unknown'
    if fit_known:
        record['legacy_edge_p90_px'] = float(p90)
        record['legacy_shift_xy_px'] = shift
    reasons, blockers, accepted, unknown = Counter(), Counter(), 0, 0
    for module in modules:
        if not isinstance(module, dict) or type(module.get('accepted')) is not bool:
            unknown += 1
            blockers['unknown_module_decision'] += 1
            continue
        gate = module.get('gate')
        if isinstance(gate, dict):
            raw_gate = gate.get('accepted')
            fields = [gate.get(k) for k in ('max_offset_px', 'max_mad_px', 'min_gradient')]
            if type(gate.get('inside')) is bool and all(_finite(v) for v in fields):
                calculated = gate['inside'] and fields[0] <= 5 and fields[1] <= 2 and fields[2] >= 15
                if type(raw_gate) is bool and calculated != raw_gate:
                    record['issues'].append('module_edge_gate_inconsistent')
        if module['accepted'] and (not isinstance(gate, dict) or gate.get('accepted') is not True):
            record['issues'].append('accepted_geometry_without_passing_module_gate')
        if module['accepted']:
            accepted += 1
            blockers['provisional_geometry_passed'] += 1
            continue
        if not isinstance(gate, dict):
            reasons['module_gate_unavailable'] += 1
            blockers['module_gate_unavailable'] += 1
            continue
        raw_gate = gate.get('accepted')
        if not fit_known:
            blockers['frame_gate_unavailable'] += 1
        elif type(raw_gate) is not bool:
            blockers['module_gate_unavailable'] += 1
        elif not fit_pass and raw_gate:
            blockers['frame_registration_only_rejected'] += 1
        elif not fit_pass:
            blockers['frame_registration_and_edges_rejected'] += 1
        elif not raw_gate:
            blockers['module_edges_only_rejected'] += 1
        else:
            blockers['unexplained_rejection'] += 1
        module_reasons = []
        if not fit_pass:
            module_reasons.append('frame_registration_rejected' if fit_known else 'frame_registration_unknown')
        if gate.get('inside') is False:
            module_reasons.append('outside_native_margin')
        for field, threshold, code, above in [
            ('max_offset_px', 5, 'edge_offset_above_5px', True),
            ('max_mad_px', 2, 'edge_spread_above_2px', True),
            ('min_gradient', 15, 'weak_frame_gradient', False),
        ]:
            value = gate.get(field)
            if not _finite(value):
                module_reasons.append('module_gate_values_unavailable')
            elif (value > threshold if above else value < threshold):
                module_reasons.append(code)
        reasons.update(set(module_reasons) or {'stored_rejection_without_recorded_cause'})
    record.update(accepted_provisional_geometry=accepted, unknown_module_decisions=unknown,
                  module_rejection_reasons=dict(reasons), module_gate_states=dict(blockers))
    if unknown:
        record['issues'].append('invalid_module_decision')
    if accepted and not fit_pass:
        record['issues'].append('accepted_geometry_without_passing_legacy_frame_gate')
    if _count(measured) is not None and measured > accepted:
        record['issues'].append('measurement_without_accepted_geometry')
    if diagnostic.get('error'):
        stage = 'processing_error'
    elif str(status).startswith('deferred') or status == 'unsupported_camera_grid':
        stage = 'deferred_geometry'
    elif any(record[k] is None for k in ('rgb_proposals', 'native_proposals', 'measured_observations')):
        stage = 'diagnostics_unavailable'
    elif measured > 0:
        stage = 'provisional_measurements'
    elif rgb == 0:
        stage = 'no_rgb_proposals'
    elif native == 0:
        stage = 'no_full_native_proposals'
    elif not fit_known:
        stage = ('insufficient_native_proposals_for_legacy_registration' if native < 4
                 else 'frame_registration_unavailable')
    elif not fit_pass:
        stage = 'frame_registration_rejected'
    elif accepted == 0:
        stage = 'module_edges_rejected'
    else:
        stage = 'geometry_passed_measurement_unavailable'
    record['stage'] = stage
    return record


def build_coverage_audit(connection, run_id):
    """Build an audit using only SELECTs, scoped to one analysis and its acquisitions.

    A missing screening table supports older stores without schema mutation.
    No source files are opened: SDK decode coverage cannot be inferred here.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError('analysis run ID required')
    rows = []
    if _table_exists(connection, 'flight_screening_frames'):
        rows = connection.execute('''
            SELECT capture_id,source_name,status,rgb_candidates,native_candidates,measured,
                   findings,diagnostics_json FROM flight_screening_frames
            WHERE analysis_run_id=? ORDER BY source_name,capture_id
        ''', [run_id]).fetchall()
    observations = connection.execute('''
        SELECT observation_id,capture_id,assessment_status,temperature_min_c,temperature_max_c,
               temperature_mean_c,temperature_median_c FROM module_observations
        WHERE analysis_run_id=?
    ''', [run_id]).fetchall()
    findings = connection.execute('SELECT finding_id,observation_id FROM findings WHERE analysis_run_id=?',
                                  [run_id]).fetchall()
    run_known = connection.execute('SELECT 1 FROM inspection_runs WHERE run_id=?', [run_id]).fetchone()
    if not run_known and not rows and not observations and not findings:
        raise ValueError('analysis run does not exist')
    all_captures = {cid: acquisition for cid, acquisition in connection.execute(
        "SELECT capture_id,run_id FROM captures WHERE image_kind='thermal'").fetchall()}
    referenced = {row[0] for row in rows} | {row[1] for row in observations}
    acquisitions = sorted({all_captures[cid] for cid in referenced
                           if cid in all_captures and all_captures[cid] is not None})
    expected = {cid for cid, acq in all_captures.items() if acq in acquisitions}
    processed = {row[0] for row in rows}
    stages, issues, reasons, blockers = Counter(), Counter(), Counter(), Counter()
    per_frame = [audit_frame(row) for row in rows]
    for frame in per_frame:
        stages[frame['stage']] += 1
        issues.update(frame['issues'])
        reasons.update(frame['module_rejection_reasons'])
        blockers.update(frame.get('module_gate_states', {}))
    if len(processed) != len(rows):
        issues['duplicate_capture_screening_rows'] += len(rows)-len(processed)
    if referenced-all_captures.keys():
        issues['unindexed_thermal_capture_references'] += len(referenced-all_captures.keys())
    if any(all_captures.get(cid) is None for cid in referenced):
        issues['acquisition_scope_incomplete'] += 1
    statuses = Counter(row[2] or 'unknown' for row in observations)
    finding_observations = {row[1] for row in findings}
    assessed = [row for row in observations if row[2] == 'analyzed']
    finite_assessed = [row for row in assessed if all(_finite(v) for v in row[3:])]
    if len(finite_assessed) != len(assessed):
        issues['analyzed_observations_missing_finite_temperature_statistics'] += len(assessed)-len(finite_assessed)
    if finding_observations-{row[0] for row in observations}:
        issues['findings_without_observation_in_selected_run'] += len(
            finding_observations-{row[0] for row in observations})
    observation_counts = Counter(row[1] for row in assessed)
    for frame in per_frame:
        actual = observation_counts[frame['capture_id']]
        frame['stored_analyzed_observations'] = actual
        if frame['measured_observations'] is not None and frame['measured_observations'] != actual:
            frame['issues'].append('measurement_count_disagrees_with_observations')
            issues['measurement_count_disagrees_with_observations'] += 1
    edge_values = [f['legacy_edge_p90_px'] for f in per_frame if f['legacy_edge_p90_px'] is not None]
    reviews = None
    if _table_exists(connection, 'finding_reviews'):
        reviews = dict(Counter(status or 'unreviewed' for (status,) in connection.execute('''
            SELECT r.review_status FROM findings f LEFT JOIN finding_reviews r USING(finding_id)
            WHERE f.analysis_run_id=?
        ''', [run_id]).fetchall()))

    def total(field):
        return (sum(frame[field] for frame in per_frame)
                if per_frame and all(frame[field] is not None for frame in per_frame) else None)

    priority = sorted([
        {'capture_id': f['capture_id'], 'source_name': f['source_name'],
         'edge_pass_frame_reject_observations': f.get('module_gate_states', {}).get('frame_registration_only_rejected', 0),
         'legacy_edge_p90_px': f['legacy_edge_p90_px']}
        for f in per_frame if f.get('module_gate_states', {}).get('frame_registration_only_rejected', 0)
    ], key=lambda f: (-f['edge_pass_frame_reject_observations'], f['source_name']))

    return {
        'schema': 'pvdelta.coverage-audit/v1', 'analysis_run_id': run_id,
        'scope': {'acquisition_run_ids': acquisitions,
                  'inferred_from_referenced_captures': True,
                  'indexed_thermal_captures': len(expected) if acquisitions else None,
                  'scope_complete': bool(acquisitions) and all(cid in all_captures and all_captures[cid]
                                                               for cid in referenced)},
        'counts': {'screening_rows': len(rows), 'screened_distinct_captures': len(processed),
                   'scoped_captures_without_screening_record': len(expected-processed) if acquisitions else None,
                   'rgb_proposal_records': total('rgb_proposals'),
                   'full_native_proposal_records': total('native_proposals'),
                   'screening_measured_counter_total': total('measured_observations'),
                   'module_observations': len(observations), 'analyzed_observations': len(assessed),
                   'analyzed_with_finite_statistics': len(finite_assessed),
                   'analyzed_without_finding': sum(row[0] not in finding_observations for row in finite_assessed),
                   'observations_with_findings': len(finding_observations & {row[0] for row in observations}),
                   'finding_records': len(findings),
                   'provisional_geometry_passes': sum(f.get('accepted_provisional_geometry', 0) for f in per_frame)},
        'frame_stages': dict(stages), 'module_assessment_statuses': dict(statuses),
        'overlapping_module_rejection_reasons': dict(reasons), 'data_issues': dict(issues),
        'exclusive_module_gate_states': dict(blockers),
        'priority_review_frames': priority,
        'finding_review_counts': reviews,
        'legacy_edge_scores': {'count': len(edge_values), 'statistic': 'per-frame heldout native edge-distance P90',
                               'median_px': float(np.median(edge_values)) if edge_values else None,
                               'p95_px': float(np.percentile(edge_values, 95)) if edge_values else None,
                               'meaning': 'historical image consistency, not source-corner or ground-truth error'},
        'physical_panel_count': None, 'independently_verified_mask_count': None,
        'verification_status': 'not_established_by_this_audit',
        'sdk_decode_coverage': 'not_revalidated_by_this_audit',
        'gate_policy': 'Native-only records use observed thermal structure and exclude shared envelope pixels; '
                       'RGB registration is not applicable and panel membership remains unverified. '
                       'Other records use EXP-043 legacy translation: >=4 modules, P90 <=5px, abs(dx/dy)<20px; '
                       'four-side offset<=5px, spread<=2px, gradient>=15. Not an IEC criterion.',
        'interpretation': 'Provisional observations only. No finding is not a healthy-panel confirmation. '
                          'RGB proposals outside the smaller thermal field of view are not proven missed panels. '
                          'Frame stages are exclusive; module rejection reasons overlap.',
        'frames': per_frame,
    }
