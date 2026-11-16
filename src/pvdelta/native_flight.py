"""Isolated native-grid flight analysis without an RGB registration prerequisite.

Observed thermal structure supplies provisional envelopes, not semantic panel
acceptance. Every candidate remains review-required; no unique panel count is inferred.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from pvdelta.config import DetectionConfig
from pvdelta.measurement_profile import frame_measurement_profile, validate_overrides
from pvdelta.native_windows import native_window_proposals
from pvdelta.pipeline import analyze_annotations
from pvdelta.regions import ModuleAnnotation, polygon_to_mask
from pvdelta.report import write_stage1_report
from pvdelta.store import connect_store, upsert_run, write_stage1_results
from pvdelta.support_partition import PartitionedSupport, partition_measurement_support
from pvdelta.surface_constraints import (
    apply_configured_surface,
    apply_surface_reference,
    configured_surface_index,
    declared_surface_masks,
)

_CONFIG = DetectionConfig(delta_threshold_k=2.5, min_region_px=4, min_reference_px=24,
                          edge_erosion_px=1, reject_boundary_findings=False)


def _excluded_support_review(temperature, support):
    """Retain excluded geometry for review without assigning heat to a panel."""
    mask = support.excluded_edge
    y, x = np.where(mask)
    values = temperature[mask]
    finite = values[np.isfinite(values)]
    return {
        'status': 'review_required' if len(values) else 'empty',
        'scope': 'excluded_edge_and_ownership_uncertainty_not_anomaly_detection',
        'mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest(),
        'pixels': len(values), 'finite_pixels': len(finite),
        'invalid_pixels': len(values) - len(finite),
        'bbox_native_xyxy': [int(x.min()), int(y.min()), int(x.max()) + 1,
                             int(y.max()) + 1] if len(x) else None,
        'max_c': float(finite.max()) if len(finite) else None,
        'p95_c': float(np.percentile(finite, 95)) if len(finite) else None,
        'anomaly_assessed': False,
    }


def native_acquisitions(database):
    connection = connect_store(database, read_only=True)
    try:
        return [r[0] for r in connection.execute("SELECT DISTINCT run_id FROM captures "
            "WHERE image_kind='thermal' AND run_id IS NOT NULL ORDER BY 1").fetchall()]
    finally:
        connection.close()


def analyze_native_frame(frame, *, proposal_builder=None, surface_reference=None,
                         surface_reference_meaning=None):
    """Measure disjoint interiors on unchanged native pixels; return provisional results."""
    frame=apply_configured_surface(frame)
    if surface_reference is None and frame.raw_metadata.get('pvdelta_surface_reference'):
        surface_reference=frame.raw_metadata['pvdelta_surface_reference']
        surface_reference_meaning='panel_surface_and_non_panel'
    original = hashlib.sha256(frame.temperature_c.tobytes()).hexdigest()
    veto=None
    if surface_reference is not None:
        _declared_panel,veto,unlabelled=declared_surface_masks(
            frame,surface_reference,surface_reference_meaning,original)
    elif surface_reference_meaning is not None:
        raise ValueError('Surface meaning requires an actual source reference')
    geometry = (proposal_builder or native_window_proposals)(frame.temperature_c, raw_supplement=True)
    proposals = geometry['proposals']
    if len({p['proposal_id'] for p in proposals}) != len(proposals):
        raise ValueError('Native proposals require distinct identifiers')
    supports = partition_measurement_support([
        polygon_to_mask(p['polygon_native_xy'], frame.temperature_c.shape) for p in proposals], margin_px=2)
    surface_states=[]
    if veto is not None:
        constrained=[]
        for s in supports:
            removed=int((s.interior & veto).sum())
            unknown=int((s.interior & unlabelled).sum())
            state=('empty_baseline_interior' if not s.interior.any() else
                   'withheld_declared_non_panel_support' if removed==int(s.interior.sum()) else
                   'mixed_declared_surface_review_required' if removed else
                   'unlabelled_surface_review_required' if unknown else 'full_declared_panel_support')
            surface_states.append({'status':state,'vetoed_interior_pixels':removed,
                                   'unlabelled_interior_pixels':unknown,
                                   'baseline_interior_pixels':int(s.interior.sum())})
            if removed:
                interior=s.interior & ~veto
                excluded=s.envelope & ~interior
                interior.setflags(write=False);excluded.setflags(write=False)
                s=PartitionedSupport(s.envelope,interior,excluded,s.margin_px,s.excluded_ambiguous)
            constrained.append(s)
        supports=tuple(constrained)
    annotations = [ModuleAnnotation(p['proposal_id'], tuple(map(tuple, p['polygon_native_xy'])))
                   for p in proposals]
    masks = {p['proposal_id']: s.interior for p, s in zip(proposals, supports)}
    reports = analyze_annotations(frame, annotations, config=_CONFIG, direction='hot', module_masks=masks)
    reports = tuple(replace(r, module=replace(r.module, quality_flags=[*r.module.quality_flags,
        'native_structure_only_panel_membership_unverified']), findings=tuple(
            replace(f, assessment_status='review_required') for f in r.findings)) for r in reports)
    if hashlib.sha256(frame.temperature_c.tobytes()).hexdigest() != original:
        raise RuntimeError('Native geometry or analysis changed source temperatures')
    diagnostic = {'geometry_method': 'native_observed_boundaries', 'geometry_semantically_verified': False,
        'rgb_registration_required': False, 'rgb_candidates': None, 'native_candidates': len(proposals),
        'native_geometry': geometry, 'modules': [{'module': p['proposal_id'],
            'native_polygon': p['polygon_native_xy'], 'accepted': True,
            'acceptance_scope': 'observed_structure_only_not_semantic_panel_acceptance',
            'gate': {'method': 'native_observed_boundaries', 'margin_px': 2},
            'mask_sha256': hashlib.sha256(s.interior.tobytes()).hexdigest(),
            'interior_pixels': int(s.interior.sum()),
            'excluded_support_review': _excluded_support_review(frame.temperature_c, s),
            'ambiguous_pixels_excluded': int(s.excluded_ambiguous.sum())}
            for p, s in zip(proposals, supports)],
        'unique_interior_pixels': sum(int(s.interior.sum()) for s in supports),
        'empty_interiors': sum(not s.interior.any() for s in supports),
        'measurement_authorised': False, 'physical_panel_count': None}
    if veto is not None:
        reports=tuple(replace(r,module=replace(r.module,quality_flags=[*r.module.quality_flags,
            state['status']])) for r,state in zip(reports,surface_states,strict=True))
        for module,state in zip(diagnostic['modules'],surface_states,strict=True):
            module['declared_surface_support']=state
        diagnostic['declared_surface_policy']={
            'method':'exclude_explicit_non_panel_and_conflict_preserve_unlabelled',
            'reference_sha256':hashlib.sha256(json.dumps(surface_reference,sort_keys=True,allow_nan=False).encode()).hexdigest(),
            'source_sha256':frame.source_hash,'native_temperature_sha256':original,
            'withheld_observations':sum(s['status'].startswith('withheld') for s in surface_states),
            'masked_observations':sum(s['vetoed_interior_pixels']>0 for s in surface_states),
            'unlabelled_pixels_are_not_healthy':True,
            'scope':'Source-local user-declared surface support, not fault truth or independent panel identity'}
    if frame.raw_metadata.get('pvdelta_surface_constraint'):
        diagnostic['surface_constraint']=frame.raw_metadata['pvdelta_surface_constraint']
    return reports, diagnostic


def _worker_initialise():
    import cv2

    cv2.setNumThreads(1)


def _native_job(job, decoder=None, proposal_builder=None):
    import cv2

    from pvdelta.cli import decode_frame

    _capture_id, relative, root, source_hash, sdk, overrides = job[:6]
    surface=job[6] if len(job)>6 else None
    frame, profile = None, None
    try:
        source_root = Path(root).expanduser().resolve()
        path = (source_root/relative).resolve()
        if not path.is_relative_to(source_root):
            raise ValueError('Original source lies outside indexed source root')
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
            raise ValueError('Original file differs from indexed source')
        read = decoder or decode_frame
        frame = read(path, sdk, measurement_overrides=overrides) if overrides else read(path, sdk)
        if frame.source_hash != source_hash:
            raise ValueError('Decoded source identity mismatch')
        if surface is not None:
            frame=apply_surface_reference(frame,surface['reference'],surface['manifest_sha256'])
        profile = frame_measurement_profile(frame)
        reports, diagnostic = analyze_native_frame(frame, proposal_builder=proposal_builder)
        return frame, profile, reports, diagnostic, None
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, IndexError, cv2.error) as exc:
        return frame, profile, (), {}, f'{type(exc).__name__}: {exc}'


def scan_native_flight(database, acquisition, output, run_id, *, sdk_library=None,
                       measurement_overrides=None, workers=1, progress=None,
                       decoder=None, proposal_builder=None, surface_manifest=None):
    """Analyse all indexed acquisition thermals into a NEW, source-bound private DB.

    One worker failure does not certify a healthy frame or interrupt later frames.
    Worker processes only read images and return results; one parent owns DB writes.
    No mask, RGB warp, legacy finding or human review is imported as panel truth.
    """
    overrides = validate_overrides(measurement_overrides)
    surface_digest,surface_index=configured_surface_index(surface_manifest)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError('workers must be an integer in 1..4')
    if workers > 1 and (decoder is not None or proposal_builder is not None):
        raise ValueError('Injected processing functions require one worker')
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', run_id):
        raise ValueError('A portable new run ID is required')
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError('Choose a new native inspection output directory')
    source = connect_store(database, read_only=True)
    try:
        source.execute('BEGIN TRANSACTION')
        rows = source.execute('''SELECT c.capture_id,c.source_path,r.source_root,c.source_hash
            FROM captures c JOIN inspection_runs r ON r.run_id=c.run_id
            WHERE c.run_id=? AND c.image_kind='thermal' ORDER BY c.source_path,c.capture_id''', [acquisition]).fetchall()
        if not rows or source.execute('SELECT 1 FROM inspection_runs WHERE run_id=?', [run_id]).fetchone():
            raise ValueError('An indexed thermal acquisition and a new run ID are required')
        snapshot = {}
        for name in ('inspection_runs', 'captures'):
            result = source.execute(f'SELECT * FROM {name}')
            snapshot[name] = ([c[0] for c in result.description], result.fetchall())
        source.execute('COMMIT')
    finally:
        source.close()
    destination.mkdir(parents=True)
    base = {'schema': 'pvdelta.native-flight/v1', 'analysis_run_id': run_id,
        'acquisition': acquisition, 'requested_frames': len(rows), 'settings': asdict(_CONFIG),
        'measurement_overrides': overrides, 'geometry_semantically_verified': False,
        'physical_panel_count': None, 'confirmed_fault_count': None,
        'field_conditions_verified': False, 'measurement_authorised': False,
        'scope': 'Provisional native structural observations. Missing geometry and no candidate are not healthy labels.'}
    if surface_digest:
        matched=sum(sha in surface_index for _,_,_,sha in rows)
        base['surface_constraints']={'manifest_sha256':surface_digest,'known_reference_frames':len(surface_index),
            'matching_source_frames':matched,'unreferenced_source_frames':len(rows)-matched,
            'unreferenced_frames_verified':False,'unknown_pixels_are_not_healthy':True}
        (destination/'surface-references.json').write_text(json.dumps({
            'manifest_sha256':surface_digest,'references':surface_index},allow_nan=False),encoding='utf-8')
    from pvdelta.native_review import boundary_engine_sources

    base['engine_sources'] = boundary_engine_sources() | {name: hashlib.sha256(
        Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
        ('native_flight.py', 'cli.py', 'measurement_profile.py', 'pipeline.py', 'config.py',
         'detect.py', 'quality.py', 'report.py', 'store.py')}
    (destination/'summary.json').write_text(json.dumps(base | {'status': 'running'}), encoding='utf-8')
    target = connect_store(destination/'inspection.duckdb')
    pool, totals, outcomes = None, Counter(), []
    try:
        target.execute('BEGIN TRANSACTION')
        for name, (columns, values) in snapshot.items():
            names = ','.join('"'+n.replace('"', '""')+'"' for n in columns)
            if values:
                target.executemany(f'INSERT INTO {name} ({names}) VALUES ({",".join("?" for _ in columns)})', values)
        target.execute('COMMIT')
        target.execute('''CREATE TABLE flight_screening_frames (
            analysis_run_id VARCHAR,capture_id VARCHAR,source_name VARCHAR,status VARCHAR,
            rgb_candidates INTEGER,native_candidates INTEGER,measured INTEGER,findings INTEGER,
            diagnostics_json JSON,PRIMARY KEY(analysis_run_id,capture_id))''')
        upsert_run(target, run_id=run_id, started_at_utc=None, source_root='',
                   site_label='Provisional native structural analysis; panel membership unverified', camera_model=None)
        jobs = [(cid, relative, root, sha, str(sdk_library) if sdk_library else None, overrides,
                 {'reference':surface_index[sha],'manifest_sha256':surface_digest} if sha in surface_index else None)
                for cid, relative, root, sha in rows]
        if workers > 1:
            pool = ProcessPoolExecutor(max_workers=workers, initializer=_worker_initialise)
            pending = {}

            def processed():
                # Bound outstanding native matrices/results, including on older Python versions.
                for index in range(min(workers*2, len(jobs))):
                    pending[index] = pool.submit(_native_job, jobs[index])
                for index in range(len(jobs)):
                    value = pending.pop(index).result()
                    following = index + workers*2
                    if following < len(jobs):
                        pending[following] = pool.submit(_native_job, jobs[following])
                    yield value

            results = processed()
        else:
            results = (_native_job(j, decoder, proposal_builder) for j in jobs)
        with (destination/'frame-outcomes.jsonl').open('w', encoding='utf-8') as journal:
            for index, (row, value) in enumerate(zip(rows, results), 1):
                cid, relative, _root, _sha = row
                frame, profile, reports, diagnostic, error = value
                outcome = {'capture_id': cid, 'source_name': Path(relative).name, 'status': 'failed',
                    'module_observations': 0, 'measured': 0, 'not_assessed': 0, 'findings': 0}
                if profile is not None:
                    totals['decoded_frames'] += 1
                if error is None:
                    try:
                        folder = destination/'frames'/f'{index:04d}'
                        write_stage1_report(folder, run_id, frame, reports, settings={**base['settings'],
                            'geometry_method': 'native_observed_boundaries', 'geometry_semantically_verified': False,
                            'margin_px': 2, 'shared_envelope_pixels_excluded': True,
                            **({'surface_constraint':diagnostic['surface_constraint']}
                               if 'surface_constraint' in diagnostic else {})})
                        write_stage1_results(target, analysis_run_id=run_id, capture_id=cid,
                            module_results=reports, measurement_profile=profile)
                        (folder/'native-geometry.json').write_text(json.dumps(diagnostic, allow_nan=False), encoding='utf-8')
                        measured = sum(r.assessment_status == 'analyzed' for r in reports)
                        findings = sum(len(r.findings) for r in reports)
                        outcome.update(status='provisional_native_measurements' if measured else 'no_assessable_modules',
                            module_observations=len(reports), measured=measured, not_assessed=len(reports)-measured,
                            findings=findings, measurement_profile_id=profile['profile_id'],
                            report_directory=f'frames/{index:04d}')
                        totals.update(completed_frames=1, module_observations=len(reports),
                            analyzed_observations=measured, not_assessed_observations=len(reports)-measured,
                            finding_records=findings, frames_without_analyzed_modules=measured == 0,
                            unique_interior_pixels=diagnostic['unique_interior_pixels'])
                    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                        error = f'{type(exc).__name__}: {exc}'
                if error is not None:
                    outcome['error'] = error
                    totals['failed_frames'] += 1
                diagnostic.update(status=outcome['status'], measured=outcome['measured'], findings=outcome['findings'],
                    geometry_method='native_observed_boundaries', geometry_semantically_verified=False,
                    rgb_registration_required=False)
                if error:
                    diagnostic['error'] = error
                target.execute('INSERT INTO flight_screening_frames VALUES (?,?,?,?,?,?,?,?,?)',
                    [run_id, cid, outcome['source_name'], outcome['status'], None, diagnostic.get('native_candidates'),
                     outcome['measured'], outcome['findings'], json.dumps(diagnostic, allow_nan=False)])
                outcomes.append(outcome)
                journal.write(json.dumps(outcome, allow_nan=False)+'\n')
                journal.flush()
                if progress:
                    progress(index, len(rows), outcome['source_name'])
        failures = totals['failed_frames']
        result = base | {'status': 'complete' if failures == 0 else 'failed' if failures == len(rows) else 'partial',
                         'totals': dict(totals), 'frames': outcomes}
        (destination/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        return result
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
        target.close()
