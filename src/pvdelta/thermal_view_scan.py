"""Candidate-independent nearby-view checks on original radiometric observations."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import fields
from pathlib import Path

import cv2
import duckdb
import numpy as np

from pvdelta import rgb_repeat
from pvdelta.candidate_links import _anchor, _same_quad
from pvdelta.candidate_triage import native_support_masks
from pvdelta.config import DetectionConfig
from pvdelta.decode import RJpeg
from pvdelta.detect import assess_intra_module_anomalies
from pvdelta.measurement_profile import apply_saved_measurement_profile, read_measurement_profile
from pvdelta.models import ModuleObservation, ThermalFrame
from pvdelta.native_diagnostics import read_native_diagnostic
from pvdelta.reference import erode
from pvdelta.rgb_pair_check import digest
from pvdelta.store import connect_store
from pvdelta.temperature_evidence_store import _runtime

ERRORS = (OSError, ValueError, RuntimeError, TypeError, KeyError, duckdb.Error, cv2.error)


def _analysis_settings(database, run):
    """Settings are optional legacy evidence, never guessed detector defaults."""
    path = Path(database).with_name('summary.json')
    if not path.is_file():
        return None, ('Saved native analysis settings are missing for this run. '
                      'Nearby photos remain available for manual comparison; automatic thermal replay is unavailable.')
    try:
        summary = json.loads(path.read_bytes())
        if (not isinstance(summary, dict) or summary.get('schema') != 'pvdelta.native-flight/v1'
                or summary.get('analysis_run_id') != run):
            raise ValueError('Saved settings do not belong to this native analysis')
        settings = {f.name: summary['settings'][f.name] for f in fields(DetectionConfig)}
        DetectionConfig(**settings)
        return settings, None
    except ERRORS as exc:
        return None, 'Saved native analysis settings cannot be verified: '+str(exc)


def _context(database, finding_id, registration, sdk_library, max_views):
    # Selection uses GPS/time only. Never consult findings before choosing targets.
    rows, base = rgb_repeat._snapshot(database, finding_id, registration, max_views, sdk_library)
    c = connect_store(database, read_only=True)
    try:
        c.execute('BEGIN TRANSACTION')
        run, oid = c.execute('SELECT analysis_run_id,observation_id FROM findings WHERE finding_id=?',
            [finding_id]).fetchone()
        for row in rows:
            cid = row['capture_id']
            row['input_error'] = None
            # Keep actual candidate counts even when old runs lack native replay evidence.
            result = c.execute('''SELECT frame_module_id,observation_id,pixel_polygon_json,assessment_status
                FROM module_observations WHERE analysis_run_id=? AND capture_id=? ORDER BY frame_module_id''',
                [run, cid]).fetchall()
            row['observations'] = {}
            for mid, observation, polygon, status in result:
                records = c.execute('''SELECT finding_id,region_pixels_yx_json,absolute_temperature_c,
                    delta_t_k,n_reference_px FROM findings WHERE analysis_run_id=? AND observation_id=?
                    ORDER BY finding_id''', [run, observation]).fetchall()
                row['observations'][mid] = {'observation_id': observation, 'polygon': json.loads(polygon),
                    'assessment_status': status, 'findings': records}
            try:
                row['profile'] = read_measurement_profile(c, run, cid)
                if row['profile'] is None:
                    raise ValueError('This legacy analysis has no saved native measurement profile. '
                        'Browse nearby photos manually; historical temperature replay and same-panel verification are unavailable.')
                row['diagnostic'] = read_native_diagnostic(c, run, cid)['diagnostic']
            except ERRORS as exc:
                row['input_error'] = str(exc)
        c.execute('COMMIT')
    finally:
        c.close()
    settings, settings_error = _analysis_settings(database, run)
    if settings_error:
        for row in rows:
            row['input_error'] = row['input_error'] or settings_error
    binding = digest({'base': base, 'rows': rows, 'run': run, 'observation': oid, 'settings': settings,
        'runtime': _runtime(sdk_library), 'engine': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'anchor_engine': hashlib.sha256(Path(__file__).with_name('candidate_links.py').read_bytes()).hexdigest()})
    return rows, run, oid, settings, binding


def thermal_views_signature(database, finding_id, registration, *, sdk_library=None, max_views=32):
    return _context(database, finding_id, registration, sdk_library, max_views)[-1]


def _replay(database, run, row, module_id, settings, sdk_library):
    item = row['observations'][module_id]
    if item['assessment_status'] != 'analyzed':
        raise ValueError('Target observation was not analyzed; absence of candidates is unknown')
    with RJpeg(row['thermal_path'], sdk_library) as reader:
        values = apply_saved_measurement_profile(reader, row['thermal_path'], row['profile'])
    c = connect_store(database, read_only=True)
    try:
        mask = native_support_masks(c, run, row['capture_id'], 0)[module_id]
    finally:
        c.close()
    config = DetectionConfig(**settings)
    frame = ThermalFrame(row['capture_id'], row['thermal_path'], row['thermal_sha256'], values,
        np.isfinite(values), values.shape[1], values.shape[0])
    assessment = assess_intra_module_anomalies(frame,
        ModuleObservation(module_id, row['capture_id'], item['polygon']), mask, config=config)
    if assessment.reasons:
        raise ValueError('Native replay is unavailable: '+'; '.join(assessment.reasons))
    def key(points):
        return tuple(sorted(map(tuple, points)))
    reproduced = {key(f.region_pixels_yx): f for f in assessment.findings if f.anomaly_type == 'hot_region'}
    records = item['findings']
    if len({key(json.loads(r[1])) for r in records}) != len(records) or set(reproduced) != {key(json.loads(r[1])) for r in records}:
        raise ValueError('Complete saved hot-region population does not reproduce')
    regions = []
    for fid, points, temperature, delta, count in records:
        found = reproduced[key(json.loads(points))]
        if found.absolute_t_c != temperature or abs(found.delta_t_c-delta) > 1e-9 or found.n_reference_px != count:
            raise ValueError('Saved region measurements do not reproduce')
        regions.append({'finding_id': fid, 'pixels_yx': json.loads(points), 'median_c': temperature,
            'delta_t_k': delta, 'reference_pixels': count})
    support = erode(mask, config.edge_erosion_px) & np.isfinite(values)
    if not support.any():
        raise ValueError('No finite measurement pixels')
    return {'status': 'analyzed_with_candidates' if regions else 'analyzed_no_candidates',
        'observation_id': item['observation_id'], 'module_id': module_id, 'native_polygon': item['polygon'],
        'regions': regions, 'profile': row['profile'], 'measured_pixels': int(support.sum()),
        'minimum_c': float(values[support].min()), 'median_c': float(np.median(values[support])),
        'maximum_c': float(values[support].max()), 'complete_saved_population_reproduced': True,
        'healthy_panel': None, 'glare_clearance': False}


def replay_review_pair(database, proposal, *, sdk_library=None):
    """Reproduce each reviewed pair member's saved measurements on its own source grid."""
    from pvdelta.panel_pair_review import _verify_sources

    verified_groups = _verify_sources(database, proposal)
    ordered_ids = (proposal['source'], proposal['target'])
    if set(ordered_ids) != verified_groups.keys():
        raise ValueError('Pair endpoints differ from the verified source binding.')
    run = proposal['binding']['run']
    settings, error = _analysis_settings(database, run)
    if error:
        raise ValueError(error)

    sources = []
    with connect_store(database, read_only=True) as connection:
        for observation_id in ordered_ids:
            group = verified_groups[observation_id]
            capture_id = group['capture_id']
            module_id = group['module_id']
            observation = connection.execute('''
                SELECT pixel_polygon_json, assessment_status FROM module_observations
                WHERE analysis_run_id=? AND capture_id=? AND frame_module_id=? AND observation_id=?
            ''', [run, capture_id, module_id, observation_id]).fetchone()
            if observation is None:
                raise ValueError('A paired observation is no longer present.')
            polygon, status = observation
            if polygon != group['regions'][0]['module_polygon']:
                raise ValueError('A paired observation changed during verification.')
            profile = read_measurement_profile(connection, run, capture_id)
            if profile is None:
                raise ValueError('A paired frame has no saved native measurement profile.')
            findings = connection.execute('''
                SELECT finding_id, region_pixels_yx_json, absolute_temperature_c, delta_t_k, n_reference_px
                FROM findings WHERE analysis_run_id=? AND observation_id=? ORDER BY finding_id
            ''', [run, observation_id]).fetchall()
            source = group['regions'][0]
            path = Path(source['source_root']) / source['thermal_path']
            sources.append((module_id, {
                'capture_id': capture_id, 'thermal_path': str(path),
                'thermal_sha256': source['thermal_hash'], 'profile': profile,
                'observations': {module_id: {'observation_id': observation_id, 'polygon': json.loads(polygon),
                                             'assessment_status': status, 'findings': findings}},
            }))

    evidence = []
    for module_id, row in sources:
        result = _replay(database, run, row, module_id, settings, sdk_library)
        evidence.append({'source_name': Path(row['thermal_path']).name, **result})
    return evidence


def scan_thermal_views(database, finding_id, registration, *, sdk_library=None, max_views=32, progress=None):
    from pvdelta.repeat_views import match_panel

    rows, run, oid, settings, binding = _context(database, finding_id, registration, sdk_library, max_views)
    source = rows[0]
    report = {'schema': 'pvdelta.thermal-view-scan/v1', 'binding': binding, 'analysis_run_id': run,
        'source_observation_id': oid, 'finding_id': finding_id, 'max_views': max_views,
        'source': {k: source[k] for k in ('capture_id','thermal_path','rgb_path','thermal_sha256','rgb_sha256')},
        'source_rgb_polygon': None, 'source_evidence': None, 'rows': [], 'status': 'unavailable', 'reason': '',
        'selection_uses_candidate_presence': False, 'requested_nearby_views': max_views,
        'retrieved_nearby_views': len(rows)-1, 'selection': 'same-acquisition GPS/time retrieval, bounded; not all visible physical-panel views',
        'physical_identity_verified': False, 'glare_clearance': False, 'database_written': False}
    try:
        if source['input_error']:
            raise ValueError(source['input_error'])
        rgb_repeat._verify_source(source, registration, sdk_library)
        observed = rgb_repeat._observed(source['rgb_path'], source['rgb_sha256'], source['gimbal_pitch_deg'])
        anchor = _anchor(source, observed, registration, module_id=source['module_id'])
        if not np.array_equal(anchor['native_polygon'], source['module_polygon']):
            raise ValueError('Selected source anchor differs from native observation')
        source_polygon = anchor['rgb_polygon']
        report['source_rgb_polygon'] = source_polygon
        report['source_evidence'] = _replay(database, run, source, source['module_id'], settings, sdk_library)
        report['status'] = 'available'
    except ERRORS as exc:
        report['reason'] = str(exc)
    for index, row in enumerate(rows[1:], 1):
        item = {k: row.get(k) for k in ('capture_id','thermal_path','rgb_path','thermal_sha256','rgb_sha256',
            'camera_distance_m','time_gap_s','gimbal_yaw_deg','gimbal_pitch_deg')}
        item.update(status='unavailable', reason='', thermal_evidence=None, rgb_polygon=None,
            forward=None, reverse=None, target_observation_id=None,
            capture_candidate_count=sum(len(m['findings']) for m in row.get('observations',{}).values()),
            panel_identity_verified=False, thermal_component_correspondence_verified=False,
            glare_clearance=False, measurement_authorised=False)
        if report['status'] != 'available':
            item.update(status='not_attempted', reason='Source anchor/replay unavailable: '+report['reason'])
        else:
            try:
                if row['input_error']:
                    raise ValueError(row['input_error'])
                rgb_repeat._verify_source(row, registration, sdk_library)
                target = rgb_repeat._observed(row['rgb_path'], row['rgb_sha256'], row['gimbal_pitch_deg'])
                forward = match_panel(Path(source['rgb_path']), Path(row['rgb_path']), np.asarray(source_polygon),
                    [np.asarray(m['polygon_rgb_xy']) for m in target['modules']])
                item['forward'] = forward.to_dict()
                if forward.status != 'matched_provisional':
                    raise ValueError('Forward RGB: '+forward.reason)
                if type(forward.target_index) is not int or not 0 <= forward.target_index < len(target['modules']):
                    raise ValueError('Invalid target panel index')
                polygon = target['modules'][forward.target_index]['polygon_rgb_xy']
                reverse = match_panel(Path(row['rgb_path']), Path(source['rgb_path']), np.asarray(polygon),
                    [np.asarray(m['polygon_rgb_xy']) for m in observed['modules']])
                item['reverse'] = reverse.to_dict()
                if (reverse.status != 'matched_provisional' or type(reverse.target_index) is not int
                        or not 0 <= reverse.target_index < len(observed['modules'])
                        or not _same_quad(observed['modules'][reverse.target_index]['polygon_rgb_xy'], source_polygon)):
                    raise ValueError('Reverse RGB did not return to the source panel: '+reverse.reason)
                target_anchor = _anchor(row, target, registration, rgb_polygon=polygon)
                mid = target_anchor['module_id']
                if not np.array_equal(target_anchor['native_polygon'], row['observations'][mid]['polygon']):
                    raise ValueError('Target anchor differs from its stored native observation')
                item.update(status='matched_provisional', rgb_polygon=polygon,
                    target_observation_id=row['observations'][mid]['observation_id'],
                    thermal_evidence=_replay(database, run, row, mid, settings, sdk_library),
                    reason='Both RGB directions and own-frame SDK replay passed; native/RGB anchors remain approximate')
            except ERRORS as exc:
                item['reason'] = str(exc)
        report['rows'].append(item)
        if progress:
            progress(index, len(rows)-1, item)
    if _context(database, finding_id, registration, sdk_library, max_views)[-1] != binding:
        raise ValueError('Inputs changed during thermal-view scan')
    report['content_sha256'] = digest(report)
    return report


def view_scan_path(database, finding_id, max_views=32):
    return Path(database).resolve().parent/'thermal-view-scans'/(digest([finding_id,max_views])+'.json')


def save_view_scan(database, report):
    path = view_scan_path(database, report['finding_id'], report['max_views'])
    if report['content_sha256'] != digest({k:v for k,v in report.items() if k!='content_sha256'}):
        raise ValueError('Changed thermal-view report')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(json.dumps(report, sort_keys=True, allow_nan=False).encode())
            stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_view_scan(database, finding_id, registration, *, sdk_library=None, max_views=32):
    path = view_scan_path(database, finding_id, max_views)
    if not path.is_file():
        return {'status':'missing'}
    try:
        report = json.loads(path.read_bytes())
        if (report['schema'] != 'pvdelta.thermal-view-scan/v1' or report['finding_id'] != finding_id
                or report['max_views'] != max_views or report['selection_uses_candidate_presence'] is not False
                or report['content_sha256'] != digest({k:v for k,v in report.items() if k!='content_sha256'})
                or report['binding'] != thermal_views_signature(database, finding_id, registration,
                    sdk_library=sdk_library, max_views=max_views)):
            raise ValueError('Saved thermal views are stale or belong to another source')
        return {'status':'available','report':report}
    except ERRORS as exc:
        return {'status':'stale','reason':str(exc)}


def thermal_view_images(report, capture_id, *, sdk_library=None):
    from pvdelta.linked_thermal import _native_image

    if report['content_sha256'] != digest({k:v for k,v in report.items() if k!='content_sha256'}):
        raise ValueError('Thermal-view report changed')
    row = next(r for r in report['rows'] if r['capture_id']==capture_id)
    if row['status'] != 'matched_provisional' or row['thermal_evidence'] is None:
        raise ValueError('A replayed target panel is required for native marks')
    evidence = [report['source_evidence'], row['thermal_evidence']]
    low = min(e['minimum_c'] for e in evidence);high = max(e['maximum_c'] for e in evidence)
    images = []
    for capture, e in zip((report['source'],row), evidence, strict=True):
        with RJpeg(capture['thermal_path'], sdk_library) as reader:
            values = apply_saved_measurement_profile(reader,capture['thermal_path'],e['profile'])
        records = [(r['finding_id'],json.dumps(r['pixels_yx']),r['median_c'],r['delta_t_k'],r['reference_pixels']) for r in e['regions']]
        images.append(_native_image(values,np.asarray(e['native_polygon']),records,low,high))
    return images, [low,high]
