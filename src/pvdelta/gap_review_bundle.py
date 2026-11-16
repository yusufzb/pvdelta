"""Local, source-bound snapshots for reviewing unassigned native thermal evidence.

This queue never writes inspection findings, panel inventory or health status.
Hashes detect stale/mixed artifacts; they are not a signature or field validation.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from copy import deepcopy
from pathlib import Path

import numpy as np


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _digest(stream.read())


def _validate(report, values):
    if (not isinstance(report, dict) or report.get('schema') != 'pvdelta.native-gap-review/v1'
            or report.get('production_enabled') is not False
            or report.get('baseline_unchanged') is not True):
        raise ValueError('Expected an isolated native-gap review report')
    if (values.ndim != 2 or values.dtype.kind != 'f' or not np.isfinite(values).all()
            or min(values.shape) < 1 or max(values.shape) > 8192):
        raise ValueError('Expected a finite floating native grid')
    if _digest(values.tobytes()) != report.get('native_temperature_sha256'):
        raise ValueError('Native temperature SHA256 differs from the report')
    source_hash = report.get('source_hash')
    if (not isinstance(source_hash, str) or len(source_hash) != 64
            or any(c not in '0123456789abcdef' for c in source_hash)):
        raise ValueError('Source SHA256 is required')
    evidence = report.get('evidence')
    if not isinstance(evidence, list):
        raise TypeError('Evidence list required (empty is not healthy)')
    seen = set()
    for item in evidence:
        if (not isinstance(item, dict)
                or item.get('status') != 'unassigned_thermal_evidence_review_required'
                or item.get('source_hash') != source_hash
                or any(item.get(k) is not False for k in ('confirmed_fault', 'measurement_authorised'))
                or any(item.get(k) is not None for k in ('module_identity', 'panel_location'))):
            raise ValueError('Unassigned evidence cannot authorise measurement or claim a panel/fault')
        quad = np.asarray(item.get('provisional_polygon_native_xy'), dtype=float)
        if (quad.shape != (4, 2) or not np.isfinite(quad).all()
                or np.any(quad < 0) or np.any(quad[:, 0] >= values.shape[1])
                or np.any(quad[:, 1] >= values.shape[0])):
            raise ValueError('Evidence polygon outside native grid')
        finding = item['native_finding']
        if not isinstance(finding, dict):
            raise TypeError('Native finding record required')
        pixels = np.asarray(finding.get('region_pixels_yx'))
        if (pixels.ndim != 2 or pixels.shape[1] != 2 or not len(pixels)
                or pixels.dtype.kind not in 'iu' or np.any(pixels < 0)
                or np.any(pixels[:, 0] >= values.shape[0])
                or np.any(pixels[:, 1] >= values.shape[1])):
            raise ValueError('Evidence needs exact integer native pixels')
        ordered = np.asarray(sorted(map(tuple, pixels.tolist())), dtype='<i4')
        if len(np.unique(ordered, axis=0)) != len(ordered) or finding.get('n_region_px') != len(ordered):
            raise ValueError('Native pixel count differs or contains duplicates')
        expected = _digest(source_hash.encode() + ordered.tobytes())[:24]
        if item.get('evidence_id') != expected or expected in seen:
            raise ValueError('Evidence identity differs from source and exact native pixels')
        seen.add(expected)
        peak = float(values[pixels[:, 0], pixels[:, 1]].max())
        if (finding.get('anomaly_type') != 'hot_region'
                or not np.isclose(peak, finding.get('region_max_t_c', float('nan')), rtol=0, atol=1e-6)
                or not np.isfinite(finding.get('delta_t_c', float('nan')))):
            raise ValueError('Native evidence statistics differ or are invalid')


def _engine_identity(hashes):
    if (not isinstance(hashes, dict) or not hashes
            or any(not isinstance(name, str) or not name or not isinstance(digest, str)
                   or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                   for name, digest in hashes.items())):
        raise ValueError('Named engine SHA256 identities required')


def write_gap_bundle(directory, frames, *, engine_source_sha256):
    """Export (report, native_array, source_path, basis, parameters) tuples locally.

    The caller explicitly supplies native data and its measurement basis. No
    decoder fallback, reference annotation or display JPEG is used here.
    Existing directories are never overwritten.
    """
    _engine_identity(engine_source_sha256)
    prepared = []
    for report, values, source, basis, parameters in frames:
        _validate(report, values)
        if not isinstance(basis, str) or not basis.strip() or not isinstance(parameters, dict):
            raise ValueError('Explicit radiometric basis and parameter record required')
        source = Path(source).resolve()
        if _file_hash(source) != report['source_hash']:
            raise ValueError('Source SHA256 differs from the report')
        prepared.append((deepcopy(report), values.copy(), str(source), basis, deepcopy(parameters)))
    if not prepared:
        raise ValueError('Bundle needs at least one inspected frame')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    entries = []
    for index, (report, values, source, basis, parameters) in enumerate(prepared):
        name = f'native-{index:04d}.npy'
        np.save(directory / name, values, allow_pickle=False)
        record = {'report': report, 'source_path': source, 'basis': basis, 'parameters': parameters,
                  'engine_source_sha256': deepcopy(engine_source_sha256),
                  'native_file': name, 'native_file_sha256': _file_hash(directory / name)}
        entries.append({'record_id': _digest(_json(record).encode()), **record})
    manifest = {'schema': 'pvdelta.native-gap-bundle/v1', 'frames': entries}
    (directory / 'manifest.json').write_text(_json(manifest), encoding='utf-8')
    return directory


def load_gap_bundle(directory):
    """Recheck raw source, snapshot and exact native data on every load."""
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if (not isinstance(manifest, dict) or manifest.get('schema') != 'pvdelta.native-gap-bundle/v1'
            or not isinstance(manifest.get('frames'), list) or not manifest['frames']):
        raise ValueError('Unsupported or empty native-gap bundle')
    frames, seen = [], set()
    for original in manifest['frames']:
        record = dict(original)
        identifier = record.pop('record_id')
        if _digest(_json(record).encode()) != identifier or identifier in seen:
            raise ValueError('Snapshot identity differs or is duplicated')
        seen.add(identifier)
        _engine_identity(record.get('engine_source_sha256'))
        if (not isinstance(record.get('basis'), str) or not record['basis'].strip()
                or not isinstance(record.get('parameters'), dict)):
            raise ValueError('Radiometric basis and parameters required')
        native_path = (directory / record['native_file']).resolve()
        if native_path.parent != directory or native_path.suffix != '.npy':
            raise ValueError('Native file must stay inside the bundle')
        if _file_hash(native_path) != record['native_file_sha256']:
            raise ValueError('Native file SHA256 differs')
        values = np.load(native_path, allow_pickle=False)
        _validate(record['report'], values)
        if _file_hash(record['source_path']) != record['report']['source_hash']:
            raise ValueError('Source SHA256 differs; stale evidence withheld')
        values.setflags(write=False)
        frames.append({**original, 'native': values})
    return frames


def evidence_snapshot(frame, item):
    """Bind a review to the whole frozen result as well as this evidence record."""
    if item not in frame['report']['evidence']:
        raise ValueError('Evidence is not in this snapshot')
    snapshot = {'record_id': frame['record_id'], 'evidence_id': item['evidence_id']}
    return _digest(_json(snapshot).encode()), _json(snapshot)


def gap_decisions(directory):
    """Read the separate review log without creating or migrating any database."""
    path = Path(directory) / 'reviews.sqlite3'
    if not path.is_file():
        return {}
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        rows = connection.execute('SELECT snapshot_id,status,reviewer,note,reviewed_at_utc '
                                  'FROM gap_review_events ORDER BY event_id').fetchall()
    return {row[0]: dict(zip(('status', 'reviewer', 'note', 'reviewed_at_utc'), row[1:]))
            for row in rows}


def save_gap_decision(directory, frame, item, status, *, reviewer='', note=''):
    """Append a triage decision; OK means useful evidence, never a confirmed fault."""
    if status not in ('ok', 'not_ok', 'wait') or not all(isinstance(s, str) for s in (reviewer, note)):
        raise ValueError('Expected ok / not_ok / wait and text reviewer/note')
    # Revalidate even if the source or bundle was replaced while a form was open.
    current = load_gap_bundle(directory)
    match = next((f for f in current if f['record_id'] == frame['record_id']), None)
    if match is None or item not in match['report']['evidence']:
        raise ValueError('Evidence snapshot changed; reload before saving')
    identity, snapshot = evidence_snapshot(match, item)
    with closing(sqlite3.connect(Path(directory) / 'reviews.sqlite3')) as connection, connection:
        connection.execute('''CREATE TABLE IF NOT EXISTS gap_review_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT, snapshot_id TEXT NOT NULL,
            snapshot_json TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('ok','not_ok','wait')),
            reviewer TEXT NOT NULL, note TEXT NOT NULL,
            reviewed_at_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))''')
        connection.execute('INSERT INTO gap_review_events '
                           '(snapshot_id,snapshot_json,status,reviewer,note) VALUES (?,?,?,?,?)',
                           (identity, snapshot, status, reviewer.strip(), note.strip()))
