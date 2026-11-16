"""Source-bound diagnostics for panel references, without detector predictions."""
import hashlib
import json
from dataclasses import asdict

import numpy as np

from pvdelta.edge_support import native_edge_continuity
from pvdelta.instance_benchmark import evaluate_panel_instances


def audit_reference_edges(frame, reference):
    """Inspect supplied edges on the unchanged native temperature grid.

    Low ridge support requests inspection; it does not invalidate a reference.
    A real outer panel edge may be a one-sided temperature step. Strong support
    likewise does not establish panel semantics, completeness or independence.
    This function never snaps labels to predictions, removes references, changes
    benchmark scores, or authorizes measurements.
    """
    values = np.asarray(frame.temperature_c)
    if (values.ndim != 2 or min(values.shape, default=0) < 8
            or values.dtype.kind not in 'fiu' or not np.isfinite(values).all()
            or values.shape != (frame.height, frame.width)):
        raise ValueError('Finite native temperature grid matching the frame required')
    digest = hashlib.sha256(values.tobytes(order='C')).hexdigest()
    if reference.get('native_temperature_sha256') != digest:
        raise ValueError('Reference/native temperature SHA256 mismatch')
    evaluate_panel_instances(reference, [], source_sha256=frame.source_hash,
                             native_shape=values.shape)
    canonical = json.dumps(reference, sort_keys=True, separators=(',', ':'), allow_nan=False)
    panels = []
    for panel in reference['panels']:
        quad = np.asarray(panel['polygon_native_xy'], float)
        edges = []
        for index, (a, b) in enumerate(zip(quad, np.roll(quad, -1, axis=0))):
            if np.linalg.norm(b-a) < 2:
                edges.append({'index': index, 'status': 'segment_too_short', 'ridge_supported': None})
                continue
            evidence = asdict(native_edge_continuity(values, a, b))
            supported = evidence['coverage'] >= .8 and evidence['largest_gap_fraction'] <= .18
            edges.append({'index': index, 'status': 'ridge_supported' if supported else 'inspect_source',
                          'ridge_supported': supported, 'evidence': evidence})
        panels.append({'instance_id': panel['instance_id'], 'edges': edges,
                       'source_inspection_requested': any(e['ridge_supported'] is not True for e in edges),
                       'reference_correctness': 'unverified'})
    if hashlib.sha256(values.tobytes(order='C')).hexdigest() != digest:
        raise RuntimeError('Reference audit changed native temperatures')
    return {'schema': 'pvdelta.reference-edge-audit/v1', 'source_sha256': frame.source_hash,
            'native_temperature_sha256': digest, 'native_shape': list(values.shape),
            'reference_sha256': hashlib.sha256(canonical.encode()).hexdigest(),
            'panels': panels, 'reference_correctness_verified': False,
            'independent_review_verified': False, 'prediction_geometry_consumed': False,
            'reference_modified': False, 'scores_modified': False, 'measurement_authorised': False,
            'interpretation': 'Edge evidence only. Unsupported ridges may be real step edges; '
                              'supported ridges may be non-panel structure. Inspect original sources.'}
