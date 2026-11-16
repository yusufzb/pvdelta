"""Reference diagnostics for immutable native geometry, separate from approval."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from pvdelta.boundary_validation import _load_reference
from pvdelta.geometry_revision import verify_geometry_version
from pvdelta.instance_benchmark import evaluate_panel_instances
from pvdelta.regions import polygon_to_mask
from pvdelta.support_partition import partition_measurement_support


def revision_validation_sources():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('geometry_revision_validation.py', 'geometry_revision.py', 'boundary_validation.py',
                     'instance_benchmark.py', 'convex_geometry.py', 'regions.py', 'support_partition.py')}


def revision_validation_token(before, after, reference_bytes):
    verify_geometry_version(before)
    verify_geometry_version(after)
    value = {'before': before['version_id'], 'after': after['version_id'],
        'reference_file_sha256': hashlib.sha256(reference_bytes).hexdigest(),
        'scorer_sources': revision_validation_sources()}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _component_placement(version, reference):
    """Check actual measured pixels, never the enlarged display bounding box."""
    shape = tuple(version['source']['native_shape'])
    proposals = version['geometry']['proposals']
    masks = [polygon_to_mask(p['polygon_native_xy'], shape) for p in proposals]
    supports = partition_measurement_support(masks, margin_px=version['settings']['margin_px'])
    by_id = {p['proposal_id']: s.interior for p, s in zip(proposals, supports)}
    if len(by_id) != len(proposals):
        raise ValueError('Unique geometry IDs required')
    truth = [(p['instance_id'], polygon_to_mask(p['polygon_native_xy'], shape)) for p in reference['panels']]
    union = np.zeros(shape, bool)
    ignored = np.zeros(shape, bool)
    for _, mask in truth:
        union |= mask
    for region in reference.get('ignore', []):
        ignored |= polygon_to_mask(region['polygon_native_xy'], shape)
    rows, seen_modules, seen_findings = [], set(), set()
    for report in version['module_reports']:
        identifier = report['module']['module_id']
        if identifier not in by_id or identifier in seen_modules:
            raise ValueError('Reports must cover geometry IDs exactly once')
        seen_modules.add(identifier)
        for finding in report['findings']:
            fid = finding['finding_id']
            raw = finding.get('region_pixels_yx')
            if (not isinstance(fid, str) or not fid or fid in seen_findings
                    or finding['module_id'] != identifier or not isinstance(raw, list) or not raw
                    or any(not isinstance(p, list) or len(p) != 2 or any(type(n) is not int for n in p) for p in raw)):
                raise ValueError('Unique findings with explicit native integer component pixels required')
            pixels = np.asarray(raw, dtype=np.int64)
            y, x = pixels.T
            if (len(set(map(tuple, raw))) != len(raw) or finding['n_region_px'] != len(raw)
                    or (pixels < 0).any() or (y >= shape[0]).any() or (x >= shape[1]).any()):
                raise ValueError('Component pixels must be unique and inside the native grid')
            if not by_id[identifier][y, x].all():
                raise ValueError('Component extends outside its reconstructed measurement support')
            seen_findings.add(fid)
            inside, ignore = int(union[y, x].sum()), int(ignored[y, x].sum())
            outside = len(raw)-inside-ignore
            rows.append({'finding_id': fid, 'proposal_id': identifier, 'measured_pixels': len(raw),
                'in_reference_panel_pixels': inside, 'in_explicit_ignore_pixels': ignore,
                'outside_reference_panel_and_ignore_pixels': outside,
                'fraction_in_reference_panels': inside/len(raw),
                'reference_instance_ids': [i for i, m in truth if m[y, x].any()],
                'placement': ('inside_reference_panels' if inside == len(raw) else
                    'outside_reference_panels' if outside == len(raw) else
                    'explicit_ignore_only' if ignore == len(raw) else 'mixed_reference_coverage')})
    if seen_modules != by_id.keys():
        raise ValueError('Reports must cover geometry IDs exactly once')
    return {'components': rows, 'component_count': len(rows),
        'measured_pixel_observations': sum(r['measured_pixels'] for r in rows),
        'outside_reference_pixel_observations': sum(r['outside_reference_panel_and_ignore_pixels'] for r in rows),
        'scope': 'Exact component pixel placement relative to supplied panel labels; not semantic or fault approval'}


def compare_geometry_versions(before, after, reference_bytes):
    """Read-only diagnostics at fixed .5/.75 gates; no SDK decode or promotion.

    Versions may be non-adjacent within a session. Their complete source,
    settings and analysis-engine contexts must match. Declared label review
    status is preserved but never treated as independently verified evidence.
    """
    token = revision_validation_token(before, after, reference_bytes)
    if any(before[k] != after[k] for k in ('source', 'settings', 'engine')):
        raise ValueError('Version source, native data, context, settings and engine must match')
    source = before['source']
    reference, selected_hash = _load_reference(reference_bytes, source['source_sha256'])
    if ('native_temperature_sha256' in reference
            and reference['native_temperature_sha256'] != source['native_temperature_sha256']):
        raise ValueError('Reference native temperature identity mismatch')
    scores = {}
    for label, version in (('before', before), ('after', after)):
        scores[label] = {str(t): evaluate_panel_instances(reference, version['geometry']['proposals'],
            source_sha256=source['source_sha256'], native_shape=tuple(source['native_shape']), iou_threshold=t)
            for t in (.5, .75)}
    transitions = {}
    for threshold in ('0.5', '0.75'):
        a, b = scores['before'][threshold], scores['after'][threshold]
        missed_a, missed_b = set(a['false_negative_ids']), set(b['false_negative_ids'])
        fp_a, fp_b = set(a['false_positive_ids']), set(b['false_positive_ids'])
        transitions[threshold] = {'newly_matched_reference_ids': sorted(missed_a-missed_b),
            'newly_missed_reference_ids': sorted(missed_b-missed_a),
            'still_missed_reference_ids': sorted(missed_a & missed_b),
            'newly_unmatched_proposal_ids': sorted(fp_b-fp_a),
            'no_longer_unmatched_proposal_ids': sorted(fp_a-fp_b),
            'still_unmatched_proposal_ids': sorted(fp_a & fp_b)}
    return {'schema': 'pvdelta.geometry-version-reference-comparison/v1', 'validation_token': token,
        'before_version_id': before['version_id'], 'after_version_id': after['version_id'],
        'source_sha256': source['source_sha256'], 'native_temperature_sha256': source['native_temperature_sha256'],
        'native_shape': source['native_shape'], 'reference_file_sha256': hashlib.sha256(reference_bytes).hexdigest(),
        'selected_reference_sha256': selected_hash, 'scorer_sources': revision_validation_sources(),
        'declared_reference_review_status': reference['review_status'],
        'reference_provenance': reference.get('annotation_method'),
        'selected_version_provenance': {'before': before['provenance'], 'after': after['provenance']},
        'scores': scores, 'transitions': transitions,
        'component_placement': {label: _component_placement(v, reference)
                                for label, v in (('before', before), ('after', after))},
        'independent_review_verified': False, 'automatic_accuracy_improvement_claimed': False,
        'measurement_authorised': False, 'database_written': False, 'main_scan_updated': False,
        'native_values_redecoded': False,
        'scope': 'Supplied-reference diagnostics, including assisted edits; not independent accuracy, '
                 'unique inventory, healthy-panel confirmation, fault diagnosis or IEC acceptance'}
