"""Local source-bound temperature-check sidecars; never modify inspection/review rows."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import duckdb
import numpy as np

from pvdelta.measurement_profile import read_measurement_profile
from pvdelta.native_diagnostics import read_native_diagnostic
from pvdelta.store import connect_store

_ERRORS = (OSError, ValueError, TypeError, KeyError, RuntimeError, duckdb.Error)
_ENGINE = ('temperature_evidence_store.py', 'reference_evidence.py', 'candidate_stability.py',
    'candidate_triage.py', 'native_diagnostics.py', 'detect.py', 'config.py', 'reference.py',
    'quality.py', 'models.py', 'regions.py', 'support_partition.py', 'measurement_profile.py', 'decode.py')


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def evidence_path(database, run_id, observation_id):
    database = Path(database).resolve()
    return (database.parent/'temperature-evidence'/(_digest([database.name, run_id]))
            /(_digest(observation_id)+'.json'))


def _runtime(sdk_library):
    configured = sdk_library or os.environ.get('PVDELTA_DIRP_LIB')
    if not configured:
        raise ValueError('Explicit SDK location required to validate saved evidence')
    sdk = Path(configured).expanduser().resolve()
    folder = sdk if sdk.is_dir() else sdk.parent
    binaries = sorted(p for p in folder.iterdir()
        if p.is_file() and (p.suffix.lower() in ('.dll', '.dylib') or '.so' in p.name))
    if not binaries or (not sdk.is_dir() and sdk not in binaries):
        raise ValueError('SDK binary identity unavailable')
    return {'sdk_path': str(sdk), 'sdk_binaries': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries},
        'python': sys.version, 'numpy': np.__version__, 'opencv': cv2.__version__,
        'engine': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in _ENGINE}}


def _snapshot(connection, database, run_id, observation_id, runtime, source_hashes):
    summary = json.loads(Path(database).with_name('summary.json').read_bytes())
    if summary.get('schema') != 'pvdelta.native-flight/v1' or summary.get('analysis_run_id') != run_id:
        raise ValueError('Saved native run settings do not match')
    row = connection.execute('''SELECT m.capture_id,m.frame_module_id,m.pixel_polygon_json,
        c.source_path,r.source_root,c.source_hash FROM module_observations m
        JOIN captures c USING(capture_id) JOIN inspection_runs r ON r.run_id=c.run_id
        WHERE m.analysis_run_id=? AND m.observation_id=?''', [run_id, observation_id]).fetchone()
    if row is None:
        raise ValueError('Observation is outside selected analysis')
    cid, mid, polygon, relative, root, source_hash = row
    base = Path(root).resolve()
    path = (base/relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError('Source lies outside indexed flight folder')
    if path not in source_hashes:
        source_hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    if source_hashes[path] != source_hash:
        raise ValueError('Original source bytes changed')
    diagnostic = read_native_diagnostic(connection, run_id, cid)
    profile = read_measurement_profile(connection, run_id, cid)
    if profile is None:
        raise ValueError('Saved measurement profile missing')
    result = connection.execute('''SELECT * FROM findings WHERE analysis_run_id=?
        AND observation_id=? ORDER BY finding_id''', [run_id, observation_id])
    names = [v[0] for v in result.description]
    findings = [dict(zip(names, r, strict=True)) for r in result.fetchall()]
    if not findings:
        raise ValueError('No saved candidate population')
    return {'run_id': run_id, 'observation_id': observation_id, 'capture_id': cid,
        'module': mid, 'polygon': polygon, 'source_path': str(path), 'source_sha256': source_hash,
        'profile': profile, 'diagnostic': diagnostic, 'findings': findings,
        'settings': summary['settings'], 'runtime': runtime}


def _validate_report(report, snapshot):
    ids = {r['finding_id'] for r in snapshot['findings']}
    if (report['schema'] != 'pvdelta.reference-evidence/v1'
            or report['analysis_run_id'] != snapshot['run_id']
            or report['observation_id'] != snapshot['observation_id']
            or report['source_sha256'] != snapshot['source_sha256']
            or report['source_file'] != Path(snapshot['source_path']).name
            or report['profile_id'] != snapshot['profile']['profile_id']
            or report['native_temperature_sha256'] != snapshot['profile']['native_temperature_sha256']
            or report['settings'] != snapshot['settings']
            or report['complete_saved_population_reproduced'] is not True
            or report['database_written'] is not False
            or len(report['regions']) != len(ids)
            or {r['finding_id'] for r in report['regions']} != ids
            or set(report['sensitivity']['regions']) != ids):
        raise ValueError('Saved evidence does not describe the current complete population')
    allowed = {'above_threshold_in_all_folds', 'below_threshold_in_all_folds',
               'crosses_threshold', 'insufficient_reference_blocks'}
    if any(r['reference_test'] not in allowed for r in report['sensitivity']['regions'].values()):
        raise ValueError('Unknown reference-test state')
    _bytes(report)  # Reject non-finite data, including non-standard JSON NaN values.


def _read(path, snapshot):
    saved = json.loads(path.read_bytes())
    if (saved['schema'] != 'pvdelta.saved-temperature-evidence/v1'
            or saved['binding_sha256'] != _digest(snapshot)
            or saved['report_sha256'] != _digest(saved['report'])):
        raise ValueError('Temperature evidence is stale or its checksum changed')
    _validate_report(saved['report'], snapshot)
    return saved


def load_temperature_checks(database, run_id, *, sdk_library=None):
    """Check current DB/profile/geometry/engine/SDK/source bytes without SDK decoding."""
    connection = connect_store(database, read_only=True)
    result = {}
    try:
        connection.execute('BEGIN TRANSACTION')
        ids = [r[0] for r in connection.execute('''SELECT DISTINCT observation_id FROM findings
            WHERE analysis_run_id=? ORDER BY observation_id''', [run_id]).fetchall()]
        runtime, runtime_error, source_hashes = None, None, {}
        for oid in ids:
            path = evidence_path(database, run_id, oid)
            if not path.is_file():
                result[oid] = {'status': 'missing', 'reason': 'No saved temperature check'}
                continue
            try:
                if runtime_error:
                    raise ValueError(runtime_error)
                if runtime is None:
                    try:
                        runtime = _runtime(sdk_library)
                    except _ERRORS as exc:
                        runtime_error = str(exc)
                        raise
                snapshot = _snapshot(connection, database, run_id, oid, runtime, source_hashes)
                saved = _read(path, snapshot)
                result[oid] = {'status': 'available', 'checked_at_utc': saved['checked_at_utc'],
                               'report': saved['report']}
            except _ERRORS as exc:
                result[oid] = {'status': 'stale', 'reason': str(exc)}
        connection.execute('COMMIT')
    finally:
        connection.close()
    return result


def check_and_save_temperature_evidence(database, run_id, observation_id, *, sdk_library=None):
    from pvdelta.reference_evidence import check_temperature_evidence

    def snapshot():
        c = connect_store(database, read_only=True)
        try:
            c.execute('BEGIN TRANSACTION')
            result = _snapshot(c, database, run_id, observation_id, _runtime(sdk_library), {})
            c.execute('COMMIT')
            return result
        finally:
            c.close()

    before = snapshot()
    report = check_temperature_evidence(database, run_id, observation_id, sdk_library=sdk_library)
    after = snapshot()
    if _digest(before) != _digest(after):
        raise ValueError('Inputs changed during temperature check; evidence not saved')
    _validate_report(report, after)
    saved = {'schema': 'pvdelta.saved-temperature-evidence/v1', 'binding_sha256': _digest(after),
        'report_sha256': _digest(report), 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'report': report}
    path = evidence_path(database, run_id, observation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(_bytes(saved))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def build_temperature_checks(database, run_id, *, sdk_library=None, force=False, progress=None):
    old = load_temperature_checks(database, run_id, sdk_library=sdk_library)
    result = {'checked': 0, 'reused': 0, 'failed': []}
    for index, (oid, saved) in enumerate(old.items(), 1):
        if not force and saved['status'] == 'available':
            result['reused'] += 1
        else:
            try:
                check_and_save_temperature_evidence(database, run_id, oid, sdk_library=sdk_library)
                result['checked'] += 1
            except _ERRORS as exc:
                result['failed'].append({'observation_id': oid, 'reason': str(exc)})
        if progress:
            progress(index, len(old))
    return result


def temperature_groups(groups, checks, selection):
    """Whole cards only; never hide a sibling region or infer unique identity."""
    result = []
    for group in groups:
        saved = checks.get(group['observation_id'], {'status': 'missing'})
        if selection == 'Checks missing / stale':
            keep = saved['status'] != 'available'
        elif saved['status'] != 'available':
            keep = False
        elif selection == 'Reference-sensitive':
            keep = any(r['reference_test'] in ('crosses_threshold', 'below_threshold_in_all_folds')
                for r in saved['report']['sensitivity']['regions'].values())
        elif selection == 'Minimum pixel support':
            keep = any(r['minimum_pixel_support'] for r in saved['report']['regions'])
        else:
            raise ValueError('Unknown temperature evidence queue')
        if keep:
            result.append(group)
    return result
