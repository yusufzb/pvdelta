"""Bound local unverified comparison frames to one exact gap-review snapshot."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pvdelta.gap_review_bundle import _digest, _engine_identity, _file_hash, _json, load_gap_bundle


def _validate_context(view, values):
    if (values.ndim != 2 or values.dtype.kind != 'f' or not np.isfinite(values).all()
            or min(values.shape) < 1 or max(values.shape) > 8192):
        raise ValueError('Finite native context grid required')
    if _digest(values.tobytes()) != view.get('native_temperature_sha256'):
        raise ValueError('Context native temperature SHA256 differs')
    if not isinstance(view.get('basis'), str) or not view['basis'].strip() or not isinstance(view.get('parameters'), dict):
        raise ValueError('Explicit context radiometric basis and parameters required')
    check = view.get('panel_check')
    if (not isinstance(check, dict) or check.get('schema') != 'pvdelta.observed-panel-pair-check/v1'
            or check.get('status') != 'withheld'
            or any(check.get(k) is not False for k in
                   ('same_panel_verified', 'measurement_authorised', 'glare_clearance'))):
        raise ValueError('Context is unverified; cannot authorise a panel transfer or glare clearance')
    p95 = check.get('heldout_p95_px')
    if (not isinstance(check.get('matches'), list) or not isinstance(check.get('config'), dict)
            or not isinstance(check.get('reasons'), list) or not check['reasons']
            or any(not isinstance(r, str) for r in check['reasons'])
            or (p95 is not None and (not isinstance(p95, (int, float))
                                    or not np.isfinite(p95) or p95 < 0))):
        raise ValueError('Context requires an explicit observed-panel control audit')


def write_gap_context(directory, record_id, views, *, engine_source_sha256):
    """Write new context once; original bundle manifest/decisions stay untouched.

    Each view supplies source path/hash, exact native data/hash, basis, parameters
    and observed-panel control report. No hot-region projection is persisted.
    """
    parents = load_gap_bundle(directory)
    parent = next((r for r in parents if r['record_id'] == record_id), None)
    if parent is None:
        raise ValueError('Unknown parent review snapshot')
    _engine_identity(engine_source_sha256)
    prepared, seen = [], {parent['report']['source_hash']}
    for raw in views:
        view = {k: v for k, v in raw.items() if k != 'native'}
        values = raw['native']
        _validate_context(view, values)
        source = Path(view['source_path']).resolve()
        if _file_hash(source) != view['source_sha256'] or view['source_sha256'] in seen:
            raise ValueError('Context source SHA256 differs or repeats a source')
        seen.add(view['source_sha256'])
        view['source_path'] = str(source)
        prepared.append((view, values))
    if not prepared:
        raise ValueError('At least one context frame required')
    folder = Path(directory)/'context'/record_id
    folder.mkdir(parents=True, exist_ok=False)
    entries = []
    for index, (view, values) in enumerate(prepared):
        path = folder/f'native-{index:04d}.npy'
        np.save(path, values, allow_pickle=False)
        entry = {**view, 'native_file': path.name, 'native_file_sha256': _file_hash(path)}
        entries.append({'view_id': _digest(_json(entry).encode()), **entry})
    payload = {'schema': 'pvdelta.gap-context/v1', 'parent_record_id': record_id,
               'engine_source_sha256': engine_source_sha256, 'views': entries,
               'selection': 'Explicit candidate neighbours; proximity does not establish capture order or same-panel identity'}
    payload['context_id'] = _digest(_json(payload).encode())
    (folder/'manifest.json').write_text(_json(payload), encoding='utf-8')
    return folder


def load_gap_context(directory, record_id):
    """Missing context is unknown; stale/corrupt context fails before any image."""
    if (not isinstance(record_id, str) or len(record_id) != 64
            or any(c not in '0123456789abcdef' for c in record_id)):
        raise ValueError('Exact parent snapshot SHA256 required')
    path = Path(directory)/'context'/record_id/'manifest.json'
    if not path.is_file():
        return None
    parent = next((r for r in load_gap_bundle(directory) if r['record_id'] == record_id), None)
    if parent is None:
        raise ValueError('Context parent snapshot is stale')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise TypeError('Context manifest must be an object')
    context_id = payload.pop('context_id')
    if (_digest(_json(payload).encode()) != context_id
            or payload.get('schema') != 'pvdelta.gap-context/v1'
            or payload.get('parent_record_id') != record_id
            or not isinstance(payload.get('views'), list) or not payload['views']):
        raise ValueError('Context identity differs or parent is stale')
    _engine_identity(payload.get('engine_source_sha256'))
    result, seen = [], {parent['report']['source_hash']}
    for original in payload['views']:
        entry = dict(original)
        identity = entry.pop('view_id')
        if _digest(_json(entry).encode()) != identity:
            raise ValueError('Context view identity differs')
        native = (path.parent/entry['native_file']).resolve()
        if native.parent != path.parent.resolve() or native.suffix != '.npy':
            raise ValueError('Context native file must remain inside its snapshot folder')
        if _file_hash(native) != entry['native_file_sha256']:
            raise ValueError('Context native file SHA256 differs')
        values = np.load(native, allow_pickle=False)
        _validate_context(entry, values)
        if _file_hash(entry['source_path']) != entry['source_sha256'] or entry['source_sha256'] in seen:
            raise ValueError('Context source SHA256 differs or repeats a source')
        seen.add(entry['source_sha256'])
        values.setflags(write=False)
        result.append({**original, 'native': values})
    return {**payload, 'context_id': context_id, 'views': result}
