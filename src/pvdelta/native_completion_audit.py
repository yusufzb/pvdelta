"""Compare optional completion against saved native support without store writes."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pvdelta.native_flight import analyze_native_frame
from pvdelta.native_transition_completion import transition_completion_proposals
from pvdelta.native_windows import native_window_proposals
from pvdelta.regions import polygon_to_mask
from pvdelta.support_partition import partition_measurement_support


def audit_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def save_audit_record(path, plan_sha256, capture_id, result):
    """Write an exclusive, digest-bound checkpoint after a comparison finishes."""
    if result.get('status') not in ('completed', 'failed'):
        raise ValueError('An explicit completed or failed audit result is required')
    payload = {'schema': 'pvdelta.native-completion-checkpoint/v1',
               'plan_sha256': plan_sha256, 'capture_id': capture_id, 'result': result}
    encoded = json.dumps({'payload': payload, 'sha256': audit_digest(payload)},
                         indent=2, allow_nan=False)
    # No overwrite: competing writers cannot silently replace a completed result.
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(encoded)


def read_audit_record(path, plan_sha256, capture_id):
    """Never reuse results from a different source/plan or an incomplete write.

    A digest detects corruption, not a malicious rewrite; this is a local audit
    checkpoint, not an authenticated external signature. Failed records remain
    failed and are not silently retried or dropped from the denominator.
    """
    try:
        record = json.loads(Path(path).read_text(encoding='utf-8'))
        payload = record['payload']
        if (record['sha256'] != audit_digest(payload)
                or payload['schema'] != 'pvdelta.native-completion-checkpoint/v1'
                or payload['plan_sha256'] != plan_sha256 or payload['capture_id'] != capture_id
                or payload['result']['status'] not in ('completed', 'failed')):
            raise ValueError('Audit checkpoint identity or digest differs')
        return payload['result']
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Incomplete or invalid audit checkpoint; preserve it for investigation') from exc


def _geometry_identity(proposals):
    return [{'proposal_id': p['proposal_id'], 'polygon_native_xy': p['polygon_native_xy']}
            for p in proposals]


def compare_native_completion(frame, saved_diagnostic, *, baseline_builder=None,
                              completion_builder=None):
    """Replay both searches and compare geometry, actual masks and old reports.

    Caller must first verify source bytes, SDK profile/native matrix and exact
    capture binding of saved_diagnostic. Report changes are measured separately
    from support: a new peer can alter inter-module findings even with identical
    old pixels. This function never infers physical identity or correctness.
    Exceptions withhold the comparison rather than certifying preservation.
    """
    native_hash = hashlib.sha256(frame.temperature_c.tobytes()).hexdigest()
    expected = saved_diagnostic['native_geometry']['proposals']
    start = time.monotonic()
    baseline = (baseline_builder or native_window_proposals)(frame.temperature_c, raw_supplement=True)
    baseline_seconds = time.monotonic()-start
    if _geometry_identity(baseline['proposals']) != _geometry_identity(expected):
        raise ValueError('Baseline replay differs from saved capture geometry')
    start = time.monotonic()
    candidate = (completion_builder or transition_completion_proposals)(
        frame.temperature_c, raw_supplement=True)
    completion_seconds = time.monotonic()-start
    old_count = len(expected)
    if (len(candidate['proposals']) < old_count
            or _geometry_identity(candidate['proposals'][:old_count]) != _geometry_identity(expected)):
        raise ValueError('Completion changed baseline IDs or geometry')

    # Independently rebuild supports; do not trust the builder's preservation flags.
    def supports(geometry):
        return partition_measurement_support([polygon_to_mask(p['polygon_native_xy'],
            frame.temperature_c.shape) for p in geometry['proposals']], margin_px=2)

    before, after = supports(baseline), supports(candidate)
    saved_masks = {m['module']: m['mask_sha256'] for m in saved_diagnostic['modules']}
    if set(saved_masks) != {p['proposal_id'] for p in expected}:
        raise ValueError('Saved mask identities differ from capture geometry')
    for p, old, new in zip(expected, before, after):
        if hashlib.sha256(old.interior.tobytes()).hexdigest() != saved_masks[p['proposal_id']]:
            raise ValueError('Baseline replay mask differs from saved measurement support')
        if not np.array_equal(old.interior, new.interior):
            raise ValueError('Completion changed baseline measurement pixels')
    occupied = np.zeros(frame.temperature_c.shape, bool)
    for support in after:
        if (occupied & support.interior).any():
            raise ValueError('Completion measurement supports overlap')
        occupied |= support.interior
        masked = support.masked_temperature(frame.temperature_c)
        if (not np.array_equal(masked[support.interior], frame.temperature_c[support.interior])
                or not np.isnan(masked[~support.interior]).all()):
            raise ValueError('Completion measurement resampled or leaked native values')

    old_reports, _ = analyze_native_frame(frame, proposal_builder=lambda *a, **k: baseline)
    new_reports, _ = analyze_native_frame(frame, proposal_builder=lambda *a, **k: candidate)
    changed = []
    for old, new in zip(old_reports, new_reports):
        if audit_digest(asdict(old)) != audit_digest(asdict(new)):
            changed.append(old.module.module_id)
    if hashlib.sha256(frame.temperature_c.tobytes()).hexdigest() != native_hash:
        raise RuntimeError('Completion audit changed native temperatures')
    completion = candidate['transition_completion']
    additions = len(candidate['proposals'])-old_count
    if completion['retained_additions'] != additions:
        raise ValueError('Completion addition count differs from actual geometry')
    return {'schema': 'pvdelta.native-completion-comparison/v1',
        'native_temperature_sha256': native_hash, 'baseline_count': old_count,
        'addition_count': additions, 'baseline_geometry_and_masks_preserved': True,
        'native_values_preserved': True, 'measurement_resampled': False,
        'baseline_reports_changed': changed,
        'baseline_report_sha256': audit_digest([asdict(r) for r in old_reports]),
        'completion_report_sha256': audit_digest([asdict(r) for r in new_reports]),
        'baseline_findings': sum(len(r.findings) for r in old_reports),
        'completion_findings': sum(len(r.findings) for r in new_reports),
        'added_observation_findings': sum(len(r.findings) for r in new_reports[old_count:]),
        'pool_overflow': completion['pool_overflow'],
        'decision_counts': completion['decision_counts'],
        'baseline_search_seconds': baseline_seconds, 'completion_search_seconds': completion_seconds,
        'proposals': candidate['proposals'], 'transition_completion': completion,
        'physical_panel_count': None, 'anomaly_accuracy': None,
        'independent_review_verified': False, 'production_promoted': False}
