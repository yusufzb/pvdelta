"""Source-bound surface-assisted scans, saved separately from inspection findings."""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pvdelta.cli import decode_frame
from pvdelta.measurement_profile import verify_measurement_profile
from pvdelta.native_flight import analyze_native_frame
from pvdelta.native_observed_completion import complete_observed_surface_gaps
from pvdelta.region_polygon import simple_polygon_mask


def recovery_engine_sources():
    from pvdelta.geometry_revision import revision_engine_sources
    from pvdelta.native_review import boundary_engine_sources

    result = revision_engine_sources() | boundary_engine_sources()
    for name in ('surface_recovery.py', 'native_observed_completion.py', 'native_completion.py',
                 'native_flight.py', 'region_polygon.py', 'measurement_profile.py'):
        result[name] = _sha(Path(__file__).with_name(name).read_bytes())
    return result


def _bytes(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def load_surface_manifest(path):
    content = Path(path).read_bytes()
    document = json.loads(content)
    if (document.get('schema') != 'pvdelta.surface-recovery-input/v1'
            or document.get('surface_meaning') != 'panel_surface_and_non_panel'
            or not isinstance(document.get('frames'), list) or not document['frames']
            or len(document['frames']) > 2000):
        raise ValueError('A nonempty source-bound panel-surface manifest is required')
    seen, sources = set(), set()
    for item in document['frames']:
        case = item['case_id']
        ref = item['surface_reference']
        profile = verify_measurement_profile(item['measurement_profile'])
        if (not isinstance(case, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', case)
                or case in seen or ref['source_sha256'] in sources
                or ref.get('review_complete') is not True or ref['case_id'] != case
                or ref['source_sha256'] != profile['source_sha256']
                or ref['native_temperature_sha256'] != profile['native_temperature_sha256']
                or [ref['native_height'], ref['native_width']] != profile['native_shape']
                or not profile['effective_parameters']):
            raise ValueError('Complete, unique surface/source/profile identities are required')
        if _sha(_bytes(item['baseline'])) != item['baseline_sha256']:
            raise ValueError('Baseline geometry hash mismatch')
        seen.add(case)
        sources.add(ref['source_sha256'])
    return document, _sha(content)


def _surface_inputs(reference, shape):
    if list(shape) != [reference['native_height'], reference['native_width']]:
        raise ValueError('Surface grid mismatch')
    panel, outside = np.zeros(shape, bool), np.zeros(shape, bool)
    windows = []
    for p in reference['panel_polygons']:
        q = np.asarray(p['polygon_native_xy'], float)
        panel |= simple_polygon_mask(q.tolist(), shape)
        lo = np.maximum(np.floor(q.min(axis=0)).astype(int)-24, 0)
        hi = np.minimum(np.ceil(q.max(axis=0)).astype(int)+24, [shape[1], shape[0]])
        if min(hi-lo) >= 64:
            windows.append([int(v) for v in [*lo, *hi]])
    for p in reference['non_panel_polygons']:
        outside |= simple_polygon_mask(p['polygon_native_xy'], shape)
    return panel & ~outside, windows


def recover_surface_frame(item, sdk_library, *, allow_isolated=False, exclude_declared_non_panel=False):
    """Decode the saved profile and measure fresh ROIs; do not write the scan DB."""
    source = Path(item['source_path'])
    if type(allow_isolated) is not bool:
        raise ValueError('allow_isolated must be boolean')
    if type(exclude_declared_non_panel) is not bool:
        raise ValueError('exclude_declared_non_panel must be boolean')
    ref = item['surface_reference']
    if _sha(source.read_bytes()) != ref['source_sha256']:
        raise ValueError('Thermal source changed since surface annotation')
    profile = verify_measurement_profile(item['measurement_profile'])
    frame = decode_frame(source, sdk_library, measurement_overrides=profile['effective_parameters'])
    if (_sha(frame.temperature_c.tobytes(order='C')) != ref['native_temperature_sha256']
            or frame.source_hash != ref['source_sha256']):
        raise ValueError('Native radiometry differs from the saved surface reference')
    mask, windows = _surface_inputs(ref, frame.temperature_c.shape)
    baseline = item['baseline']
    result = complete_observed_surface_gaps(frame.temperature_c, baseline,
        panel_mask=mask, windows=windows, raw_supplement=True,
        **({'allow_isolated': True} if allow_isolated else {}))
    before, before_detail = analyze_native_frame(frame, proposal_builder=lambda *a, **k: baseline)
    after, after_detail = analyze_native_frame(frame, proposal_builder=lambda *a, **k: result,
        **({'surface_reference':ref,'surface_reference_meaning':'panel_surface_and_non_panel'}
           if exclude_declared_non_panel else {}))
    old = len(baseline['proposals'])
    if (result['proposals'][:old] != baseline['proposals']
            or (not exclude_declared_non_panel and (
                [asdict(p) for p in before] != [asdict(p) for p in after[:len(before)]]
                or before_detail['modules'] != after_detail['modules'][:len(before)]))):
        raise RuntimeError('Surface recovery changed existing geometry or measurements')
    if _sha(source.read_bytes()) != ref['source_sha256']:
        raise ValueError('Thermal source changed during recovery')
    record = {'schema': 'pvdelta.surface-recovery-frame/v1', 'case_id': item['case_id'],
        'source_file': source.name, 'source_sha256': ref['source_sha256'],
        'native_temperature_sha256': ref['native_temperature_sha256'],
        'baseline_count': old, 'observation_count': len(result['proposals']),
        'added_observations': len(result['proposals'])-old,
        'candidate_regions': sum(len(r.findings) for r in after),
        'baseline_geometry': baseline, 'geometry': result,
        'module_reports': [asdict(p) for p in after], 'measurement_profile': profile,
        'old_measurements_preserved': not exclude_declared_non_panel, 'old_geometry_preserved':True,
        'database_written': False,
        'independent_accuracy_verified': False, 'field_conditions_verified': False}
    if allow_isolated:
        record['allow_isolated'] = True
    if exclude_declared_non_panel:
        record['declared_surface_policy']=after_detail['declared_surface_policy']
        record['measurement_diagnostics']=after_detail['modules']
    return record, _picture(frame.temperature_c, baseline, [], old), _picture(
        frame.temperature_c, result, after, old)


def _picture(values, geometry, reports, old_count):
    import cv2
    from PIL import Image, ImageDraw

    low, high = np.percentile(values, [2, 98])
    pixels = np.clip((values-low)/max(high-low, .01)*255, 0, 255).astype(np.uint8)
    image = Image.fromarray(cv2.cvtColor(cv2.applyColorMap(pixels, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for index, p in enumerate(geometry['proposals']):
        q = list(map(tuple, p['polygon_native_xy']))
        draw.line(q+q[:1], fill='#ffd447' if index >= old_count else '#29d8e8', width=1)
    for report in reports:
        if report.assessment_status=='not_assessed':
            q=list(map(tuple,report.module.pixel_polygon))
            draw.line(q+q[:1],fill='#9a9a9a',width=1)
        for finding in report.findings:
            q = list(map(tuple, finding.pixel_polygon))
            draw.line(q+q[:1], fill='#ff4949', width=1)
    return image


def run_surface_recovery(manifest_path, sdk_library, output_root, *, progress=None,
                         allow_isolated=False, exclude_declared_non_panel=False):
    """Process every configured source; retain failures explicitly and isolate runs."""
    if type(allow_isolated) is not bool:
        raise ValueError('allow_isolated must be boolean')
    if type(exclude_declared_non_panel) is not bool:
        raise ValueError('exclude_declared_non_panel must be boolean')
    manifest, digest = load_surface_manifest(manifest_path)
    engine = recovery_engine_sources()
    directory = Path(output_root).resolve() / ('run-'+uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    for index, item in enumerate(manifest['frames']):
        case = item['case_id']
        try:
            record, before, after = recover_surface_frame(item, sdk_library,
                **({'allow_isolated': True} if allow_isolated else {}),
                **({'exclude_declared_non_panel':True} if exclude_declared_non_panel else {}))
            payload = _bytes(record)
            (directory/f'{case}.json').write_bytes(payload)
            before.save(directory/f'{case}-before.png')
            after.save(directory/f'{case}-after.png')
            rows.append({k: record[k] for k in ('case_id', 'source_file', 'baseline_count',
                'observation_count', 'added_observations', 'candidate_regions')} | {
                'status': 'completed', 'record_sha256': _sha(payload),
                'before_sha256': _sha((directory/f'{case}-before.png').read_bytes()),
                'after_sha256': _sha((directory/f'{case}-after.png').read_bytes())})
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, ImportError) as exc:
            rows.append({'case_id': case, 'status': 'failed', 'error': str(exc)})
        if progress:
            progress(index+1, len(manifest['frames']))
    result = {'schema': 'pvdelta.surface-recovery-run/v1', 'manifest_sha256': digest,
        'started_at_utc': started,
        'engine_source_sha256': engine,
        'run_id': directory.name, 'configured_frames': len(rows),
        'completed_frames': sum(r['status']=='completed' for r in rows),
        'failed_frames': sum(r['status']=='failed' for r in rows), 'frames': rows,
        'manual_surface_required': True, 'database_written': False, 'stage1_accepted': False}
    result['allow_isolated'] = allow_isolated
    result['exclude_declared_non_panel'] = exclude_declared_non_panel
    (directory/'summary.json').write_bytes(_bytes(result))
    if _sha(Path(manifest_path).read_bytes()) != digest or engine != recovery_engine_sources():
        raise ValueError('Surface manifest changed during scan; results were not activated')
    latest = directory.parent/'latest.json'
    if latest.is_file():
        previous = json.loads(latest.read_bytes())
        _checked_run_id(previous['run_id'])
        previous_content = (directory.parent/previous['run_id']/'summary.json').read_bytes()
        if _sha(previous_content) != previous['summary_sha256']:
            raise ValueError('Previous run summary hash mismatch; refusing to replace it')
        _save_history_pointer(directory.parent, previous)
    saved = {'run_id': directory.name, 'summary_sha256': _sha(_bytes(result))}
    _save_history_pointer(directory.parent, saved)
    # A failed or partial retry must not displace the last complete inspection.
    if result['completed_frames'] and not result['failed_frames']:
        pointer = directory.parent/(directory.name+'-latest.tmp')
        pointer.write_bytes(_bytes(saved))
        pointer.replace(latest)
    return result, directory


def _checked_run_id(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r'run-[0-9a-f]{32}', run_id):
        raise ValueError('Invalid saved run ID')
    return run_id


def _save_history_pointer(root, pointer):
    history = root/'history'
    history.mkdir(exist_ok=True)
    path = history/(_checked_run_id(pointer['run_id'])+'.json')
    content = _bytes(pointer)
    try:
        with path.open('xb') as stream:
            stream.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError('Saved history identity changed; refusing overwrite')


def available_surface_runs(output_root):
    """List saved attempts, putting the last complete run first when available."""
    root = Path(output_root).resolve()
    latest = root/'latest.json'
    ids = []
    if latest.is_file():
        ids.append(_checked_run_id(json.loads(latest.read_bytes())['run_id']))
    paths = sorted((root/'history').glob('run-*.json'), key=lambda p: p.stat().st_mtime_ns,
                   reverse=True)
    for path in paths:
        identifier = _checked_run_id(path.stem)
        if identifier not in ids:
            ids.append(identifier)
    return ids


def load_surface_run(output_root, manifest_path, *, run_id=None):
    """Read a completed snapshot, never treat it as a fresh source verification."""
    root = Path(output_root).resolve()
    pointer_path = root/'latest.json'
    if run_id is not None:
        history = root/'history'/(_checked_run_id(run_id)+'.json')
        if history.is_file():
            pointer_path = history
    pointer = json.loads(pointer_path.read_bytes())
    _checked_run_id(pointer['run_id'])
    if run_id is not None and pointer['run_id'] != run_id:
        raise ValueError('Requested saved run ID has no matching history record')
    directory = root/pointer['run_id']
    content = (directory/'summary.json').read_bytes()
    if _sha(content) != pointer['summary_sha256']:
        raise ValueError('Saved run summary hash mismatch')
    summary = json.loads(content)
    if summary.get('run_id') != pointer['run_id']:
        raise ValueError('Saved summary run ID mismatch')
    _, digest = load_surface_manifest(manifest_path)
    if summary['manifest_sha256'] != digest:
        raise ValueError('Saved run belongs to a different surface configuration; run again')
    for row in summary['frames']:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', row['case_id']):
            raise ValueError('Invalid saved case ID')
        if row['status'] == 'completed':
            for suffix, key in [('.json', 'record_sha256'), ('-before.png', 'before_sha256'), ('-after.png', 'after_sha256')]:
                if _sha((directory/(row['case_id']+suffix)).read_bytes()) != row[key]:
                    raise ValueError('Saved result artifact hash mismatch')
    return summary, directory
