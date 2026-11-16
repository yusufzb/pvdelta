"""Immutable boundary-result snapshots and append-only visual review decisions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pvdelta.store import connect_store

BOUNDARY_REASONS = {
    'missing_panel': 'Missing panel',
    'non_panel_region': 'Vegetation / ground selected',
    'merged_panels': 'Panels merged together',
    'split_panel': 'One panel split apart',
    'boundary_misalignment': 'Boundary misalignment',
    'incomplete_coverage': 'Incomplete frame coverage',
    'insufficient_view': 'Insufficient view / uncertain',
    'other': 'Other boundary issue',
}


def _reason_codes(values):
    if (not isinstance(values, (list, tuple))
            or any(not isinstance(v, str) or v not in BOUNDARY_REASONS for v in values)):
        raise ValueError('Boundary reason codes must be a list or tuple of supported codes')
    return [code for code in BOUNDARY_REASONS if code in values]


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _valid_hash(value):
    return isinstance(value, str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def boundary_engine_signature(hashes):
    if (not isinstance(hashes, dict) or not hashes
            or any(not isinstance(name, str) or not name or not _valid_hash(digest)
                   for name, digest in hashes.items())):
        raise ValueError('Boundary engine requires named SHA256 identities')
    return _hash(hashes)


def boundary_result_identity(report):
    """Bind decisions to exact source, native data, engine, geometry and settings.

    Session token is excluded: moving an identical source/SDK installation alone
    need not discard a decision about the exact same saved numerical result.
    """
    if not isinstance(report, dict) or report.get('schema')!='pvdelta.native-boundary-preview/v1':
        raise ValueError('Unsupported native boundary report')
    if not all(_valid_hash(report.get(k)) for k in ('source_sha256','native_temperature_sha256')):
        raise ValueError('Boundary report requires source and native-temperature SHA256')
    boundary_engine_signature(report.get('engine_source_sha256'))
    shape = report.get('native_shape')
    if (not isinstance(shape, list) or len(shape)!=2 or any(type(n) is not int or n<1 for n in shape)
            or not isinstance(report.get('source_name'), str) or not report['source_name']):
        raise ValueError('Boundary report requires source name and native grid')
    if (report.get('semantic_panel_membership')!='unverified'
            or any(report.get(k) is not False for k in ('measurement_authorised','database_written','anomaly_rerun'))
            or report.get('native_values_unchanged') is not True):
        raise ValueError('Visual boundary review cannot authorise measurement or replace analysis')
    for key, count in (('baseline','baseline_proposals'),('proposed','proposals')):
        result = report.get(key)
        proposals = result.get('proposals') if isinstance(result, dict) else None
        if (not isinstance(proposals, list) or type(report.get(count)) is not int
                or report[count]!=len(proposals)):
            raise ValueError('Boundary report candidate counts do not match geometry')
    snapshot = {k:v for k,v in report.items() if k!='preview_token'}
    return 'sha256:'+_hash(snapshot)


def save_boundary_review(database, report, review_status, *, reviewer=None, note=None, reason_codes=()):
    """Persist a human visual decision; leave findings/assessment statuses untouched."""
    if review_status not in ('ok','not_ok','wait'):
        raise ValueError('Boundary review status must be ok, not_ok or wait')
    if any(v is not None and not isinstance(v, str) for v in (reviewer,note)):
        raise ValueError('Reviewer and note must be text or None')
    reasons = _reason_codes(reason_codes)
    if review_status=='ok' and reasons:
        raise ValueError('OK cannot carry unresolved boundary reason codes')
    result_id = boundary_result_identity(report)
    snapshot = _canonical({k:v for k,v in report.items() if k!='preview_token'})
    signature = boundary_engine_signature(report['engine_source_sha256'])
    connection = connect_store(database)
    try:
        connection.execute('BEGIN TRANSACTION')
        try:
            connection.execute('''INSERT INTO native_boundary_snapshots
                (result_id, source_sha256, source_name, engine_signature, report_json)
                VALUES (?, ?, ?, ?, ?) ON CONFLICT (result_id) DO NOTHING''',
                [result_id,report['source_sha256'],report['source_name'],signature,snapshot])
            stored = connection.execute('SELECT report_json FROM native_boundary_snapshots WHERE result_id=?',
                                        [result_id]).fetchone()[0]
            if stored!=snapshot:
                raise ValueError('Saved boundary snapshot differs from its identity')
            event_id = connection.execute('''INSERT INTO native_boundary_review_events
                (result_id, review_status, reviewer, note) VALUES (?, ?, ?, ?) RETURNING event_id''',
                [result_id,review_status,(reviewer or '').strip() or None,(note or '').strip() or None]).fetchone()[0]
            if reasons:
                connection.executemany('''INSERT INTO native_boundary_review_reasons
                    (event_id, reason_code) VALUES (?, ?)''', [(event_id,code) for code in reasons])
            connection.execute('COMMIT')
        except Exception:
            connection.execute('ROLLBACK')
            raise
    finally:
        connection.close()
    return {'event_id':event_id,'result_id':result_id,'review_status':review_status,'reason_codes':reasons}


def _review_rows(database, *, source_sha256=None, result_id=None):
    """Read legacy stores without migrating; pre-v7 events have no recorded reasons."""
    if not Path(database).is_file():
        return []
    connection = connect_store(database, read_only=True)
    try:
        count = connection.execute('''SELECT count(*) FROM information_schema.tables
            WHERE table_schema='main' AND table_name IN
            ('native_boundary_snapshots','native_boundary_review_events')''').fetchone()[0]
        if count!=2:
            return []
        if result_id is not None:
            if not isinstance(result_id, str) or not result_id.startswith('sha256:') or not _valid_hash(result_id[7:]):
                raise ValueError('Boundary result SHA256 identity required')
            cursor = connection.execute('''SELECT e.event_id, e.result_id, e.review_status, e.reviewer,
                e.note, CAST(e.reviewed_at_utc AS VARCHAR) AS reviewed_at_utc,
                s.source_sha256, s.source_name, s.engine_signature
                FROM native_boundary_review_events e JOIN native_boundary_snapshots s USING (result_id)
                WHERE e.result_id=? ORDER BY e.event_id DESC LIMIT 1''', [result_id])
        elif source_sha256 is None:
            query = '''SELECT * FROM (
                SELECT e.event_id, e.result_id, e.review_status, e.reviewer, e.note,
                    CAST(e.reviewed_at_utc AS VARCHAR) AS reviewed_at_utc,
                    s.source_sha256, s.source_name, s.engine_signature,
                    row_number() OVER (PARTITION BY s.source_sha256 ORDER BY e.event_id DESC) AS position
                FROM native_boundary_review_events e JOIN native_boundary_snapshots s USING (result_id)
            ) WHERE position=1 ORDER BY event_id DESC'''
            cursor = connection.execute(query)
        else:
            if not _valid_hash(source_sha256):
                raise ValueError('Source SHA256 required for boundary review history')
            cursor = connection.execute('''SELECT e.event_id, e.result_id, e.review_status, e.reviewer,
                e.note, CAST(e.reviewed_at_utc AS VARCHAR) AS reviewed_at_utc,
                s.source_sha256, s.source_name, s.engine_signature
                FROM native_boundary_review_events e JOIN native_boundary_snapshots s USING (result_id)
                WHERE s.source_sha256=? ORDER BY e.event_id DESC LIMIT 50''', [source_sha256])
        columns = [c[0] for c in cursor.description]
        rows = [dict(zip(columns,row), reason_codes=[]) for row in cursor.fetchall()]
        has_reasons = connection.execute('''SELECT count(*) FROM information_schema.tables
            WHERE table_schema='main' AND table_name='native_boundary_review_reasons' ''').fetchone()[0]
        if rows and has_reasons:
            by_event = {r['event_id']:r for r in rows}
            placeholders = ','.join('?' for _ in by_event)
            reasons = connection.execute('SELECT event_id,reason_code FROM native_boundary_review_reasons '
                f'WHERE event_id IN ({placeholders}) ORDER BY reason_code', list(by_event)).fetchall()
            for event_id, code in reasons:
                by_event[event_id]['reason_codes'].append(code)
        return rows
    finally:
        connection.close()


def latest_boundary_reviews(database):
    return {r['source_sha256']:r for r in _review_rows(database)}


def boundary_review_history(database, source_sha256):
    """Latest 50 visual decisions for this source, including older engine versions."""
    return _review_rows(database, source_sha256=source_sha256)


def boundary_review_for_result(database, result_id):
    rows = _review_rows(database, result_id=result_id)
    return rows[0] if rows else None
