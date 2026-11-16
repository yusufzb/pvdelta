"""Explicit reviewed source-joint aliases; no residual-driven point selection."""
from copy import deepcopy
from dataclasses import asdict

import numpy as np

from pvdelta.control_review import _verify_pack, digest, latest_control_decisions


def source_joint_binding(pack, output, frame_index):
    _verify_pack(pack)
    if type(frame_index) is not int or not 0 <= frame_index < len(pack['frames']):
        raise ValueError('Explicit source pair index required')
    decisions = latest_control_decisions(pack, output)
    control = pack['frames'][frame_index]['controls']
    ids = [p['control_id'] for p in control['points']]
    return {'pack_id': pack['pack_id'], 'frame_index': frame_index, 'pair': control['pair'],
            'decisions_sha256': digest({cid: decisions.get(cid) for cid in ids})}


def evaluate_source_joints(frame, rgb_path, pack, output, aliases, *, frame_index,
                           diagnostic=False, rgb_calibration=None):
    """Retain every raw check while assigning shared joints to checking.

    Alias groups must be explicitly declared from source review and bind the exact
    current decisions. Coordinates must agree exactly in both sensors; the function
    never averages observations or infers aliases from distances/residuals.
    Pending diagnostic mode returns numbers but never a usable registration.
    """
    from pvdelta.alignment import validate_homography_checks
    from pvdelta.local_rgb_transfer import (
        _context,
        _native_matrix,
        _rgb_model_points,
        fit_local_rgb_registration,
    )

    if type(diagnostic) is not bool:
        raise ValueError('diagnostic must be boolean')
    binding = source_joint_binding(pack, output, frame_index)
    if (not isinstance(aliases, dict) or aliases.get('schema') != 'pvdelta.source-joint-aliases/v1'
            or aliases.get('binding') != binding or not isinstance(aliases.get('groups'), list)
            or not isinstance(aliases.get('note'), str) or not aliases['note'].strip()):
        raise ValueError('Explicit source aliases bound to current review decisions required')
    if _context(frame, rgb_path) != binding['pair']:
        raise ValueError('Reviewed joints belong to a different source pair/native grid')
    control = pack['frames'][frame_index]['controls']
    decisions = latest_control_decisions(pack, output)
    if digest({p['control_id']: decisions.get(p['control_id']) for p in control['points']}) != binding['decisions_sha256']:
        raise ValueError('Source reviews changed during joint preparation')
    raw = []
    for p in control['points']:
        decision = decisions.get(p['control_id'])
        if decision is None:
            raise ValueError('Every original control needs a recorded source decision')
        raw.append(dict(deepcopy(p), native_xy=decision['native_xy'], rgb_xy=decision['rgb_xy']))
    by_id = {p['control_id']: p for p in raw}
    used, groups = set(), []
    for group in aliases['groups']:
        if (not isinstance(group, list) or len(group) < 2
                or any(not isinstance(cid, str) or cid not in by_id for cid in group)
                or len(set(group)) != len(group) or used.intersection(group)):
            raise ValueError('Alias groups require distinct known controls, each used once')
        points = [by_id[cid] for cid in group]
        if any(p['native_xy'] != points[0]['native_xy'] or p['rgb_xy'] != points[0]['rgb_xy']
               for p in points):
            raise ValueError('Alias observations disagree; source review required, no averaging')
        used.update(group)
        groups.append(points)
    groups.extend([p] for p in raw if p['control_id'] not in used)
    canonical, audit = [], []
    for index, group in enumerate(groups):
        role = 'check' if any(p['role'] == 'check' for p in group) else 'fit'
        cid = f'J{index+1:03d}'
        canonical.append({'control_id': cid, 'role': role,
                          'native_xy': group[0]['native_xy'], 'rgb_xy': group[0]['rgb_xy']})
        audit.append({'joint_id': cid, 'role': role, 'original_controls': [
            {'control_id': p['control_id'], 'role': p['role']} for p in group]})
    pending = [p['control_id'] for p in raw if decisions[p['control_id']]['status'] != 'OK']
    rejected = [p['control_id'] for p in raw if decisions[p['control_id']]['status'] == 'Not OK']
    result = {'schema': 'pvdelta.source-joint-evaluation/v1', 'binding': binding,
        'aliases_sha256': digest(aliases), 'raw_observations': raw, 'joints': audit,
        'fit_joint_count': sum(p['role'] == 'fit' for p in canonical),
        'check_joint_count': sum(p['role'] == 'check' for p in canonical),
        'pending_control_ids': pending, 'review_complete': not pending,
        'registration': None, 'canonical_validation': None, 'all_raw_check_validation': None,
        'raw_check_residuals': [],
        'correspondence_verified': False, 'independent_review_verified': False,
        'measurement_authorised': False, 'database_written': False}
    if rejected or pending and not diagnostic:
        return result | {'status': 'source_review_pending'}
    reviewed = {'schema': 'pvdelta.local-rgb-controls/v1', 'pair': binding['pair'],
        'provenance': {'kind': 'explicit_reviewed_source_joints', 'note': aliases['note'],
            'aliases_sha256': digest(aliases), 'review_complete': not pending}, 'points': canonical}
    registration = fit_local_rgb_registration(frame, rgb_path, reviewed, rgb_calibration=rgb_calibration)
    result['canonical_validation'] = registration['validation']
    # Score ALL raw checks, even those aliased into a single canonical joint.
    raw_checks = [p for p in raw if p['role'] == 'check']
    joint_by_original = {p['control_id']: j['joint_id'] for j in audit for p in j['original_controls']}
    residual_by_joint = {r['control_id']: r for r in registration['control_residuals']}
    result['raw_check_residuals'] = [{'control_id': p['control_id'],
        'joint_id': joint_by_original[p['control_id']],
        'error_native_px': residual_by_joint[joint_by_original[p['control_id']]]['error_native_px']}
        for p in raw_checks]
    if registration['geometry_gate_passed']:
        raw_validation = validate_homography_checks(_native_matrix(registration),
            fit_point_count=result['fit_joint_count'],
            check_source_xy=np.asarray([p['native_xy'] for p in raw_checks]),
            check_target_xy=_rgb_model_points(np.asarray([p['rgb_xy'] for p in raw_checks]),
                registration['rgb_geometry']), max_p95_source_px=2., min_fit_points=8)
        result['all_raw_check_validation'] = asdict(raw_validation)
        if raw_validation.accepted and not pending:
            result['registration'] = registration
    result['status'] = ('pending_source_diagnostic_only' if pending else
                        'reviewed_joint_point_gate_passed' if result['registration'] else 'point_gate_failed')
    if source_joint_binding(pack, output, frame_index) != binding:
        raise ValueError('Source reviews changed during joint evaluation')
    return result
