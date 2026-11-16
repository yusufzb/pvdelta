"""Bind native screening diagnostics to capture identity, never filename counters."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def read_native_diagnostic(connection, analysis_run_id, capture_id, *, source_file=None):
    """Read and cross-check saved geometry against exact run/capture observations.

    Flight report directories are processing indices, not DJI image numbers.
    Consumers must not infer a diagnostic path from a basename. Optional source
    bytes are verified separately; no migration, geometry approval or write occurs.
    """
    if any(not isinstance(v, str) or not v.strip() for v in (analysis_run_id, capture_id)):
        raise ValueError('Explicit analysis run and capture identity required')
    rows = connection.execute('''SELECT c.source_path,c.source_hash,c.width_px,c.height_px,
        f.source_name,f.diagnostics_json FROM flight_screening_frames f
        JOIN captures c ON c.capture_id=f.capture_id
        WHERE f.analysis_run_id=? AND f.capture_id=?''', [analysis_run_id, capture_id]).fetchall()
    if len(rows) != 1:
        raise ValueError('One native diagnostic for this exact run/capture is required')
    indexed_path, source_hash, width, height, name, encoded = rows[0]
    if (not isinstance(source_hash, str) or len(source_hash) != 64
            or any(c not in '0123456789abcdef' for c in source_hash)
            or Path(str(indexed_path)).name != name or type(width) is not int or type(height) is not int
            or min(width, height) < 1):
        raise ValueError('Diagnostic source metadata differs from the indexed capture')
    if source_file is not None and hashlib.sha256(Path(source_file).read_bytes()).hexdigest() != source_hash:
        raise ValueError('Original source bytes differ from indexed identity')
    diagnostic = json.loads(encoded)
    json.dumps(diagnostic, allow_nan=False)
    if (not isinstance(diagnostic, dict) or diagnostic.get('geometry_method') != 'native_observed_boundaries'
            or not isinstance(diagnostic.get('native_geometry'), dict)
            or not isinstance(diagnostic['native_geometry'].get('proposals'), list)):
        raise ValueError('Saved diagnostic is not native observed geometry')
    proposals = diagnostic['native_geometry']['proposals']
    if any(not isinstance(p, dict) for p in proposals):
        raise ValueError('Native proposals must be geometry objects')
    observed = connection.execute('''SELECT frame_module_id,pixel_polygon_json
        FROM module_observations WHERE analysis_run_id=? AND capture_id=?''',
        [analysis_run_id, capture_id]).fetchall()
    if len({r[0] for r in observed}) != len(observed):
        raise ValueError('Duplicate observation identities in this run/capture')
    by_id = dict(observed)
    ids = [p.get('proposal_id') for p in proposals]
    if (any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids)
            or set(ids) != set(by_id) or type(diagnostic.get('native_candidates')) is not int
            or diagnostic['native_candidates'] != len(proposals)):
        raise ValueError('Diagnostic proposals differ from saved capture observations')
    for p in proposals:
        q = np.asarray(p.get('polygon_native_xy'), float)
        expected = np.asarray(json.loads(by_id[p['proposal_id']]), float)
        if (q.shape != (4, 2) or not np.isfinite(q).all() or (q < 0).any()
                or (q[:, 0] > width).any() or (q[:, 1] > height).any()
                or not np.array_equal(q, expected)):
            raise ValueError('Diagnostic polygon differs from this capture native grid/observation')
    result = {'schema': 'pvdelta.capture-native-diagnostic/v1',
        'analysis_run_id': analysis_run_id, 'capture_id': capture_id,
        'source_name': name, 'source_sha256': source_hash, 'native_shape': [height, width],
        'source_bytes_verified': source_file is not None, 'observations_verified': True,
        'diagnostic': diagnostic, 'database_written': False, 'measurement_authorised': False}
    result['binding_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return result
