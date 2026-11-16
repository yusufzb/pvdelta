"""Replay stored native geometry into an isolated, profile-aware inspection run."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from pvdelta.config import DetectionConfig
from pvdelta.measurement_profile import frame_measurement_profile, validate_overrides
from pvdelta.models import ModuleObservation
from pvdelta.pipeline import analyze_annotations
from pvdelta.reference import erode
from pvdelta.regions import ModuleAnnotation, polygon_to_mask
from pvdelta.report import ModuleReport, write_stage1_report
from pvdelta.store import connect_store, upsert_run, write_stage1_results


def attach_site_geometry(output, source):
    """Keep the explicitly selected user site prior beside a completed replay."""
    destination = Path(output)
    content = Path(source).expanduser().read_bytes()
    site = json.loads(content)
    tables = site.get('tables', {})
    tilt, facing = tables.get('surface_tilt_deg'), tables.get('surface_facing_azimuth_true_deg')
    if (site.get('schema') != 'pvdelta.site-geometry-prior/v1'
            or any(type(v) not in (int, float) or not np.isfinite(v) for v in (tilt, facing))
            or not 0 <= tilt < 90 or not 0 <= facing < 360):
        raise ValueError('An explicit site geometry prior with finite tilt and facing is required')
    summary = json.loads((destination/'summary.json').read_bytes())
    if summary.get('schema') != 'pvdelta.screening-replay/v1' or summary.get('status') == 'running':
        raise ValueError('A completed or partial replay is required')
    target = destination/'site-geometry.json'
    if target.exists() and target.read_bytes() != content:
        raise FileExistsError('Different site geometry already exists for this replay')
    target.write_bytes(content)
    summary['site_geometry_sha256'] = hashlib.sha256(content).hexdigest()
    summary['site_geometry_is_user_prior_not_field_validation'] = True
    (destination/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    return summary


def screening_runs(database):
    connection = connect_store(database, read_only=True)
    try:
        if not connection.execute("SELECT 1 FROM information_schema.tables WHERE table_name='flight_screening_frames'").fetchone():
            return []
        return [r[0] for r in connection.execute("SELECT analysis_run_id FROM flight_screening_frames "
            "GROUP BY analysis_run_id HAVING sum(CASE WHEN "
            "json_extract_string(diagnostics_json,'$.geometry_method')='native_observed_boundaries' "
            "THEN 1 ELSE 0 END)=0 ORDER BY 1").fetchall()]
    finally:
        connection.close()


def _stored_gate_passes(diagnostic, module):
    """Check recorded legacy constraints without promoting them to field validation."""
    gate = module.get('gate', {})
    error, shift = diagnostic.get('heldout_p90_px'), diagnostic.get('shift_xy')
    def finite(value):
        return type(value) in (int, float) and np.isfinite(value)
    return (gate.get('accepted') is True and gate.get('inside') is True
        and finite(error) and 0 <= error <= 5
        and isinstance(shift, list) and len(shift) == 2 and all(finite(v) and abs(v) < 20 for v in shift)
        and all(finite(gate.get(k)) for k in ('max_offset_px', 'max_mad_px', 'min_gradient'))
        and 0 <= gate['max_offset_px'] <= 5 and 0 <= gate['max_mad_px'] <= 2 and gate['min_gradient'] >= 15)


def replay_screening(database, baseline_run, output, run_id, *, sdk_library=None,
                     measurement_overrides=None, per_frame_overrides=None, progress=None,
                     decoder=None):
    """Read all sources, retain rejected geometry, and never mutate the input store.

    Geometry is replayed, not revalidated or promoted. Native temperatures are
    freshly calculated by the SDK. No result is a confirmed healthy panel/fault.
    Per-frame overrides, when supplied, must name every exact source hash once.
    """
    from pvdelta.cli import decode_frame

    decoder = decoder or decode_frame
    overrides = validate_overrides(measurement_overrides)
    mapped = per_frame_overrides
    if mapped is not None:
        if not isinstance(mapped, dict) or overrides:
            raise ValueError('Use either fixed overrides or an explicit source-hash mapping')
        mapped = {k: validate_overrides(v) for k, v in mapped.items()}
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError('Choose a new inspection output directory')
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', run_id):
        raise ValueError('A portable analysis run ID is required')
    source = connect_store(database, read_only=True)
    try:
        source.execute('BEGIN TRANSACTION')
        rows = source.execute('''SELECT f.capture_id,f.source_name,f.diagnostics_json,
            c.source_path,r.source_root,c.source_hash FROM flight_screening_frames f
            JOIN captures c ON c.capture_id=f.capture_id
            JOIN inspection_runs r ON r.run_id=c.run_id
            WHERE f.analysis_run_id=? ORDER BY f.source_name,f.capture_id''', [baseline_run]).fetchall()
        if not rows:
            raise ValueError('No stored screening geometry for the selected analysis')
        if any(json.loads(row[2]).get('geometry_method') == 'native_observed_boundaries' for row in rows):
            raise ValueError('Native geometry uses scan-native-flight; legacy RGB geometry replay is not applicable')
        if source.execute('SELECT 1 FROM inspection_runs WHERE run_id=?', [run_id]).fetchone():
            raise ValueError('Use a new run ID; the original analysis stays separate')
        hashes = [r[5] for r in rows]
        if len({r[0] for r in rows}) != len(rows):
            raise ValueError('Duplicate capture diagnostics in selected scope')
        if mapped is not None and set(mapped) != set(hashes):
            raise ValueError('Per-frame overrides must cover exactly the selected source hashes')
        snapshot = {}
        for table in ('inspection_runs', 'captures'):
            cursor = source.execute(f'SELECT * FROM {table}')
            snapshot[table] = ([c[0] for c in cursor.description], cursor.fetchall())
        source.execute('COMMIT')
    finally:
        source.close()
    destination.mkdir(parents=True)
    (destination/'summary.json').write_text(json.dumps({'schema': 'pvdelta.screening-replay/v1',
        'run_id': run_id, 'baseline_run': baseline_run, 'status': 'running',
        'requested_frames': len(rows)}), encoding='utf-8')
    target = connect_store(destination/'inspection.duckdb')
    totals, outcomes = Counter(), []
    config = DetectionConfig(delta_threshold_k=2.5, min_region_px=4, min_reference_px=24,
                              edge_erosion_px=1, reject_boundary_findings=False)
    try:
        for table, (columns, values) in snapshot.items():
            names = ','.join('"'+n.replace('"', '""')+'"' for n in columns)
            if values:
                target.executemany(f'INSERT INTO {table} ({names}) VALUES ({",".join("?" for _ in columns)})', values)
        target.execute('''CREATE TABLE flight_screening_frames (
            analysis_run_id VARCHAR,capture_id VARCHAR,source_name VARCHAR,status VARCHAR,
            rgb_candidates INTEGER,native_candidates INTEGER,measured INTEGER,findings INTEGER,
            diagnostics_json JSON,PRIMARY KEY(analysis_run_id,capture_id))''')
        upsert_run(target, run_id=run_id, started_at_utc=None, source_root='',
                   site_label='Source-bound geometry replay; measurements require review', camera_model=None)
        with (destination/'frame-outcomes.jsonl').open('w', encoding='utf-8') as journal:
            for index, (capture_id, name, raw, relative, root, source_hash) in enumerate(rows, 1):
                outcome = {'capture_id': capture_id, 'source_name': name, 'status': 'failed',
                           'measured': 0, 'findings': 0, 'not_assessed': 0}
                diagnostic = {}
                try:
                    diagnostic = json.loads(raw)
                    modules = diagnostic.get('modules', [])
                    if not isinstance(modules, list) or any(not isinstance(m, dict) or type(m.get('accepted')) is not bool for m in modules):
                        raise ValueError('Explicit geometry decisions required for every stored module')
                    identifiers = [m.get('module') for m in modules]
                    if (any(not isinstance(i, str) or not i for i in identifiers)
                            or len(set(identifiers)) != len(identifiers)):
                        raise ValueError('Distinct explicit frame-local module IDs required')
                    source_root = Path(root).expanduser().resolve()
                    path = (source_root/relative).resolve()
                    if not path.is_relative_to(source_root):
                        raise ValueError('Original source lies outside indexed source root')
                    if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
                        raise ValueError('Original file differs from indexed source')
                    current = mapped[source_hash] if mapped is not None else overrides
                    frame = decoder(path, sdk_library, measurement_overrides=current) if current else decoder(path, sdk_library)
                    if frame.source_hash != source_hash:
                        raise ValueError('Decoded original source identity mismatch')
                    totals['decoded_frames'] += 1
                    annotations, masks, reports = [], {}, []
                    for module in modules:
                        polygon = module['native_polygon']
                        if not module['accepted'] or not _stored_gate_passes(diagnostic, module):
                            reports.append(ModuleReport(ModuleObservation(module['module'], frame.frame_id, polygon),
                                assessment_status='not_assessed', reasons=('original native registration/edge gate rejected',)))
                            continue
                        q = np.asarray(polygon, float)
                        h, w = frame.temperature_c.shape
                        margin = module.get('gate', {}).get('margin_px')
                        if (q.shape != (4, 2) or not np.isfinite(q).all() or (q < 0).any()
                                or (q[:, 0] >= w).any() or (q[:, 1] >= h).any()
                                or type(margin) is not int or margin < 0):
                            raise ValueError('Accepted geometry needs native corners and an explicit non-negative margin')
                        envelope = polygon_to_mask(polygon, (h, w))
                        mask = erode(envelope, margin)
                        if not mask.any():
                            raise ValueError('Accepted geometry has no measurement interior')
                        annotations.append(ModuleAnnotation(module['module'], tuple(map(tuple, polygon))))
                        masks[module['module']] = mask
                    occupancy = np.zeros(frame.temperature_c.shape, np.int32)
                    for mask in masks.values():
                        occupancy += mask
                    shared = occupancy > 1
                    excluded = int(shared.sum())
                    if excluded:
                        masks = {key: mask & ~shared for key, mask in masks.items()}
                    reports.extend(analyze_annotations(frame, annotations, config=config, direction='hot', module_masks=masks))
                    reports = tuple(replace(r, module=replace(r.module,
                        quality_flags=[*r.module.quality_flags, 'replayed_provisional_boundaries_not_revalidated',
                                       f'frame_shared_measurement_pixels_excluded:{excluded}']),
                        findings=tuple(replace(f, assessment_status='review_required') for f in r.findings)) for r in reports)
                    profile = frame_measurement_profile(frame)
                    folder = destination/'frames'/f'{index:04d}'
                    write_stage1_report(folder, run_id, frame, reports, settings={
                        'detection': asdict(config), 'direction': 'hot', 'geometry_replay_of': baseline_run,
                        'geometry_semantically_verified': False, 'explicit_native_masks': True,
                        'measurement_mask_sha256': {key: hashlib.sha256(mask.tobytes()).hexdigest()
                                                    for key, mask in masks.items()},
                        'shared_measurement_pixels_excluded': excluded})
                    write_stage1_results(target, analysis_run_id=run_id, capture_id=capture_id,
                                         module_results=reports, measurement_profile=profile)
                    measured = sum(r.assessment_status == 'analyzed' for r in reports)
                    findings = sum(len(r.findings) for r in reports)
                    outcome.update(status='reassessed_provisional' if measured else 'no_assessable_modules',
                        measured=measured, findings=findings, not_assessed=len(reports)-measured,
                        module_observations=len(reports), measurement_profile_id=profile['profile_id'],
                        report_directory=f'frames/{index:04d}')
                    totals.update(completed_frames=1, module_observations=len(reports), analyzed_observations=measured,
                        not_assessed_observations=len(reports)-measured, finding_records=findings,
                        frames_without_analyzed_modules=measured == 0)
                except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                    outcome['error'] = f'{type(exc).__name__}: {exc}'
                    totals['failed_frames'] += 1
                diagnostic.update(status=outcome['status'], measured=outcome['measured'], findings=outcome['findings'],
                    replay_of=baseline_run, measurement_profile_id=outcome.get('measurement_profile_id'),
                    geometry_semantically_verified=False, peer_comparison='unavailable_no_explicit_groups')
                if 'error' in outcome:
                    diagnostic['error'] = outcome['error']
                target.execute('INSERT INTO flight_screening_frames VALUES (?,?,?,?,?,?,?,?,?)',
                    [run_id, capture_id, name, outcome['status'], diagnostic.get('rgb_candidates'),
                     diagnostic.get('native_candidates'), outcome['measured'], outcome['findings'],
                     json.dumps(diagnostic, allow_nan=False)])
                journal.write(json.dumps(outcome, allow_nan=False)+'\n')
                journal.flush()
                outcomes.append(outcome)
                if progress is not None:
                    progress(index, len(rows), name)
        failed = totals['failed_frames']
        result = {'schema': 'pvdelta.screening-replay/v1', 'run_id': run_id, 'baseline_run': baseline_run,
            'status': 'complete' if not failed else 'failed' if failed == len(rows) else 'partial',
            'requested_frames': len(rows), 'totals': dict(totals), 'settings': asdict(config),
            'measurement_overrides': overrides, 'per_frame_overrides': mapped,
            'geometry_semantically_verified': False, 'physical_panel_count': None,
            'confirmed_fault_count': None, 'field_conditions_verified': False,
            'scope': 'Every selected source decoded; only stored accepted geometry remeasured. '
                'No finding and missing geometry are not healthy-panel confirmations.',
            'frames': outcomes}
        (destination/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        return result
    finally:
        target.close()
