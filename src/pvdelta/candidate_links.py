"""Bidirectional RGB hypotheses between existing candidate cards, never asset merges."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from pvdelta import rgb_repeat
from pvdelta.measurement_profile import read_measurement_profile
from pvdelta.native_diagnostics import read_native_diagnostic
from pvdelta.rgb_pair_check import digest, match_module_quads
from pvdelta.store import connect_store


def _context(database, finding_id, registration, sdk_library, max_targets, *, hash_file=None):
    if type(max_targets) is not int or not 1 <= max_targets <= 8:
        raise ValueError('max_targets must be in 1..8')
    kwargs = {'hash_file': hash_file} if hash_file is not None else {}
    rows, base_binding = rgb_repeat._snapshot(database, finding_id, registration, 32, sdk_library, **kwargs)
    c = connect_store(database, read_only=True)
    try:
        c.execute('BEGIN TRANSACTION')
        identity = c.execute('SELECT analysis_run_id,observation_id FROM findings WHERE finding_id=?',
            [finding_id]).fetchone()
        if identity is None:
            raise ValueError('Selected candidate no longer exists')
        run_id, source_observation = identity
        candidate_captures = {r[0] for r in c.execute('''SELECT DISTINCT m.capture_id FROM findings f
            JOIN module_observations m USING(observation_id) WHERE f.analysis_run_id=?
            AND m.analysis_run_id=?''', [run_id, run_id]).fetchall()}
        selected = [rows[0], *[r for r in rows[1:] if r['capture_id'] in candidate_captures][:max_targets]]
        for row in selected:
            cid = row['capture_id']
            row['diagnostic'] = read_native_diagnostic(c, run_id, cid)['diagnostic']
            row['profile'] = read_measurement_profile(c, run_id, cid)
            observations = c.execute('''SELECT frame_module_id,observation_id,pixel_polygon_json
                FROM module_observations WHERE analysis_run_id=? AND capture_id=? ORDER BY frame_module_id''',
                [run_id, cid]).fetchall()
            row['observations'] = {}
            for mid, oid, polygon in observations:
                findings = [r[0] for r in c.execute('''SELECT finding_id FROM findings
                    WHERE analysis_run_id=? AND observation_id=? ORDER BY finding_id''', [run_id, oid]).fetchall()]
                row['observations'][mid] = {'observation_id': oid, 'finding_ids': findings, 'polygon': polygon}
        c.execute('COMMIT')
    finally:
        c.close()
    payload = {'base_binding': base_binding, 'selected': selected, 'analysis_run_id': run_id,
        'source_observation_id': source_observation, 'max_targets': max_targets,
        'engine_sha256': rgb_repeat._file_hash(Path(__file__))}
    return selected, digest(payload), source_observation


def candidate_links_signature(database, finding_id, registration, *, sdk_library=None, max_targets=4, hash_file=None):
    kwargs = {'hash_file': hash_file} if hash_file is not None else {}
    return _context(database, finding_id, registration, sdk_library, max_targets, **kwargs)[1]


def _same_quad(first, second):
    a, b = np.asarray(first, float), np.asarray(second, float)
    return (a.shape == b.shape == (4, 2) and np.isfinite(a).all() and np.isfinite(b).all()
        and any(np.array_equal(a, np.roll(order, shift, axis=0))
            for order in (b, b[::-1]) for shift in range(4)))


def _anchor(row, observed, registration, module_id=None, rgb_polygon=None):
    matches = match_module_quads(row['diagnostic']['modules'], observed['modules'], registration['model'])
    if module_id is not None:
        selected = [m for m in matches if m['module_id'] == module_id]
    else:
        selected = [m for m in matches if _same_quad(m['rgb_polygon'], rgb_polygon)]
    if len(selected) != 1:
        raise ValueError('No unique bounded native/RGB anchor for the tracked panel')
    return selected[0]


def candidate_card_links(database, finding_id, registration, *, sdk_library=None, max_targets=4):
    from pvdelta.repeat_views import match_panel

    rows, binding, source_observation = _context(database, finding_id, registration, sdk_library, max_targets)
    source = rows[0]
    rgb_repeat._verify_source(source, registration, sdk_library)
    observed = deepcopy(rgb_repeat._observed(source['rgb_path'], source['rgb_sha256'], source['gimbal_pitch_deg']))
    anchor = _anchor(source, observed, registration, module_id=source['module_id'])
    if not np.array_equal(anchor['native_polygon'], source['module_polygon']):
        raise ValueError('Source anchor differs from the selected native observation')
    source_polygon = anchor['rgb_polygon']
    output = []
    for row in rows[1:]:
        item = {k: row.get(k) for k in ('capture_id', 'thermal_path', 'rgb_path', 'thermal_sha256',
            'rgb_sha256', 'camera_distance_m', 'time_gap_s')}
        item.update(status='unavailable', reason='', forward=None, reverse=None, rgb_polygon=None,
            target_observation_id=None, target_finding_ids=[], bidirectional_rgb_passed=False,
            panel_identity_verified=False, thermal_component_correspondence_verified=False,
            glare_clearance=False, measurement_authorised=False)
        try:
            # Both original temperature grids must reconstruct their own saved profiles.
            rgb_repeat._verify_source(row, registration, sdk_library)
            target = rgb_repeat._observed(row['rgb_path'], row['rgb_sha256'], row['gimbal_pitch_deg'])
            forward = match_panel(Path(source['rgb_path']), Path(row['rgb_path']), np.asarray(source_polygon),
                [np.asarray(m['polygon_rgb_xy']) for m in target['modules']])
            item['forward'] = forward.to_dict()
            if forward.status != 'matched_provisional':
                raise ValueError('Forward RGB check: '+forward.reason)
            if type(forward.target_index) is not int or not 0 <= forward.target_index < len(target['modules']):
                raise ValueError('Forward RGB target index is invalid')
            target_polygon = target['modules'][forward.target_index]['polygon_rgb_xy']
            reverse = match_panel(Path(row['rgb_path']), Path(source['rgb_path']), np.asarray(target_polygon),
                [np.asarray(m['polygon_rgb_xy']) for m in observed['modules']])
            item['reverse'] = reverse.to_dict()
            if reverse.status != 'matched_provisional':
                raise ValueError('Reverse RGB check: '+reverse.reason)
            if (type(reverse.target_index) is not int
                    or not 0 <= reverse.target_index < len(observed['modules'])
                    or not _same_quad(observed['modules'][reverse.target_index]['polygon_rgb_xy'], source_polygon)):
                raise ValueError('Reverse RGB check did not return to the original observed panel')
            target_anchor = _anchor(row, target, registration, rgb_polygon=target_polygon)
            target_observation = row['observations'][target_anchor['module_id']]
            item.update(bidirectional_rgb_passed=True, rgb_polygon=deepcopy(target_polygon),
                target_observation_id=target_observation['observation_id'],
                target_finding_ids=target_observation['finding_ids'], target_module_id=target_anchor['module_id'],
                source_anchor_iou=anchor['iou'], target_anchor_iou=target_anchor['iou'])
            if target_observation['finding_ids']:
                item.update(status='provisional_candidate_link', reason='Both RGB directions passed; both native/RGB anchors remain approximate')
            else:
                item.update(status='matched_panel_without_candidate', reason='Tracked observed panel has no saved candidate; this does not establish a healthy panel')
        except (OSError, ValueError, RuntimeError, TypeError, KeyError, cv2.error) as exc:
            item['reason'] = str(exc)
        output.append(item)
    if _context(database, finding_id, registration, sdk_library, max_targets)[1] != binding:
        raise ValueError('Sources, geometry or candidate population changed during comparison')
    report = {'schema': 'pvdelta.candidate-card-links/v1', 'binding': binding, 'finding_id': finding_id,
        'source_observation_id': source_observation,
        'source': {k: source[k] for k in ('capture_id', 'thermal_path', 'rgb_path', 'thermal_sha256', 'rgb_sha256')},
        'source_rgb_polygon': source_polygon, 'rows': output,
        'retrieval': 'Up to32 nearest GPS captures within20m (time fallback if source GPS missing); inspect up to max_targets captures containing candidates',
        'max_targets': max_targets, 'source_anchor_status': 'approximate_native_association',
        'database_written': False, 'physical_panels_merged': False, 'unique_fault_count': None,
        'panel_identity_verified': False, 'glare_clearance': False, 'decisions_transferred': False}
    report['content_sha256'] = digest(report)
    return report


def candidate_link_images(report, capture_id):
    if (report.get('schema') != 'pvdelta.candidate-card-links/v1'
            or report.get('content_sha256') != digest({k: v for k, v in report.items() if k != 'content_sha256'})):
        raise ValueError('Candidate-link report changed')
    converted = {'schema': 'pvdelta.observed-rgb-repeat/v1', 'source': report['source'],
        'source_rgb_polygon': report['source_rgb_polygon'], 'rows': [dict(r,
            status='matched_provisional' if r['bidirectional_rgb_passed'] else 'unavailable') for r in report['rows']]}
    converted['content_sha256'] = digest(converted)
    return rgb_repeat.repeat_pair_images(converted, capture_id)
