"""Read-only, run-scoped inspection handover without private image copies."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import defaultdict
from datetime import date, datetime, timezone
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory

from pvdelta.coverage_audit import _table_exists, build_coverage_audit
from pvdelta.store import connect_store

SCHEMA = 'pvdelta.inspection-export/v1'
FILES = {'inspection.json', 'coverage.json', 'frames.csv', 'observations.csv',
         'findings.csv', 'measurement-profiles.csv', 'review.html'}


def _rows(connection, sql, params):
    result = connection.execute(sql, params)
    names = [c[0] for c in result.description]
    return names, [dict(zip(names, row)) for row in result.fetchall()]


def _json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f'Unsupported export value: {type(value).__name__}')


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)


def _csv(columns, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(columns)
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, (datetime, date)):
                value = value.isoformat()
            if isinstance(value, str) and (value.lstrip().startswith(('=', '+', '-', '@'))
                                           or value.startswith(('\t', '\r', '\n'))):
                value = "'" + value
            cells.append(value)
        writer.writerow(cells)
    return stream.getvalue()


def _html(payload):
    audit = payload['coverage']
    counts = audit['counts']
    grouped = defaultdict(list)
    for finding in payload['findings']:
        grouped[finding['observation_id']].append(finding)
    cards = []
    for oid, findings in grouped.items():
        first = findings[0]
        regions = ''.join('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in (
            f['finding_id'], f['anomaly_type'], f['delta_t_k'], f['absolute_temperature_c'],
            f['review_status'] or 'unreviewed', f['review_note'] or '')) + '</tr>' for f in findings)
        cards.append(f'<details><summary>{escape(str(oid))} · {len(findings)} region(s)</summary>'
            f'<p>Thermal source: {escape(str(first["thermal_source_path"]))}<br>'
            f'Paired RGB: {escape(str(first["rgb_source_path"]))}<br>'
            f'Estimated finding location: {escape(str(first["estimated_latitude"]))}, '
            f'{escape(str(first["estimated_longitude"]))}; method: '
            f'{escape(str(first["location_method"]))}</p>'
            '<table><tr><th>Finding ID</th><th>Type</th><th>ΔT K</th><th>Temperature °C</th>'
            '<th>User review</th><th>Note</th></tr>'+regions+'</table></details>')
    metrics = ''.join(f'<li>{escape(k.replace("_", " "))}: {v}</li>' for k, v in counts.items()
                     if k in {'module_observations', 'analyzed_observations', 'analyzed_without_finding',
                              'observations_with_findings', 'finding_records',
                              'scoped_captures_without_screening_record'})
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>PV Delta inspection handover</title><style>'
        'body{font:16px system-ui;max-width:1200px;margin:24px auto;padding:16px;line-height:1.5}'
        'details{border:1px solid #aaa;padding:12px;margin:12px 0;overflow-wrap:anywhere;overflow:auto}'
        'table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aaa}'
        '.note{background:#fff2cc;padding:16px;color:#222}</style>'
        '<h1>PV Delta inspection handover</h1><p>Analysis run: '+escape(payload['analysis_run_id'])+'</p>'
        '<div class="note">Provisional screening, not a validated fault diagnosis or IEC acceptance. '
        'Counts refer to frame observations, not unique physical panels. No candidate does not mean '
        'healthy. OK / Not OK / Wait are operator review decisions. Camera GPS is not panel GPS. '
        'Missing data remain unknown. This export contains records, not the source photos or SDK.</div>'
        '<h2>Coverage</h2><ul>'+metrics+'</ul><p>Unique physical panels: unknown. Confirmed equipment '
        'faults: unknown.</p><p>Review counts: '+escape(_json(audit['finding_review_counts']))+'</p>'
        '<p>Data issues: '+escape(_json(audit['data_issues']))+'</p>'
        '<p><a href="inspection.json">Complete records (JSON)</a> · '
        '<a href="coverage.json">Coverage audit</a> · <a href="frames.csv">Frames</a> · '
        '<a href="observations.csv">Observations</a> · <a href="findings.csv">Findings/reviews</a> · '
        '<a href="measurement-profiles.csv">Measurement profiles</a></p>'
        '<h2>Candidate observations — all regions together</h2>'+''.join(cards)+'</html>')


def export_inspection(database, run_id, output):
    """Export one consistent read-only snapshot, publishing only complete bundles.

    JSON preserves original text; CSV neutralizes spreadsheet formula prefixes.
    Existing output directories are never overwritten. Unknown physical identity,
    healthy totals and confirmed fault totals are deliberately not inferred.
    """
    import duckdb

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError('Choose a new inspection export directory')
    connection = None
    try:
        connection = connect_store(database, read_only=True)
        connection.execute('BEGIN TRANSACTION')
        audit = build_coverage_audit(connection, run_id)
        acquisitions = audit['scope']['acquisition_run_ids']
        frames = _rows(connection, '''SELECT c.*,rgb.source_path AS rgb_source_path,
            rgb.source_hash AS rgb_source_hash FROM captures c LEFT JOIN captures rgb
            ON rgb.capture_id=c.paired_capture_id AND rgb.image_kind='rgb' AND rgb.run_id=c.run_id
            WHERE c.image_kind='thermal' AND c.run_id IN (SELECT unnest(?))
            ORDER BY c.source_path,c.capture_id''', [acquisitions])
        observations = _rows(connection, '''SELECT m.*,c.source_path AS thermal_source_path,
            c.source_hash AS thermal_source_sha256 FROM module_observations m
            LEFT JOIN captures c ON c.capture_id=m.capture_id
            WHERE m.analysis_run_id=? ORDER BY m.capture_id,m.observation_id''', [run_id])
        reviews = _table_exists(connection, 'finding_reviews')
        review_select = ('r.review_status,r.reviewer,r.note AS review_note,r.reviewed_at_utc'
                         if reviews else 'NULL AS review_status,NULL AS reviewer,'
                         'NULL AS review_note,NULL AS reviewed_at_utc')
        review_join = 'LEFT JOIN finding_reviews r ON r.finding_id=f.finding_id' if reviews else ''
        findings = _rows(connection, f'''SELECT f.*,m.capture_id,m.frame_module_id,
            c.source_path AS thermal_source_path,c.source_hash AS thermal_source_sha256,
            rgb.source_path AS rgb_source_path,rgb.source_hash AS rgb_source_sha256,
            c.camera_latitude,c.camera_longitude,{review_select}
            FROM findings f LEFT JOIN module_observations m
            ON m.observation_id=f.observation_id AND m.analysis_run_id=f.analysis_run_id
            LEFT JOIN captures c ON c.capture_id=m.capture_id
            LEFT JOIN captures rgb ON rgb.capture_id=c.paired_capture_id
            AND rgb.image_kind='rgb' AND rgb.run_id=c.run_id
            {review_join} WHERE f.analysis_run_id=? ORDER BY f.observation_id,f.finding_id''', [run_id])
        profiles = (_rows(connection, '''SELECT * FROM measurement_profiles
            WHERE analysis_run_id=? ORDER BY capture_id''', [run_id])
            if _table_exists(connection, 'measurement_profiles') else ([], []))
        connection.execute('COMMIT')
    except duckdb.Error as exc:
        raise ValueError(f'Cannot export inspection: {exc}') from exc
    finally:
        if connection is not None:
            connection.close()
    payload = {'schema': SCHEMA, 'analysis_run_id': run_id,
        'exported_at_utc': datetime.now(timezone.utc).isoformat(),
        'coverage': audit, 'frames': frames[1], 'observations': observations[1],
        'findings': findings[1], 'measurement_profiles': profiles[1],
        'physical_panel_count': None, 'healthy_panel_count': None, 'confirmed_fault_count': None,
        'stage1_acceptance': 'not_assessed_by_export', 'original_images_included': False,
        'csv_formula_prefixes_escaped': True}
    contents = {'inspection.json': _json(payload), 'coverage.json': _json(audit),
                'review.html': _html(payload)}
    for name, (columns, rows) in [('frames', frames), ('observations', observations),
                                 ('findings', findings), ('measurement-profiles', profiles)]:
        contents[name+'.csv'] = _csv(columns, rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.pvdelta-export-', dir=destination.parent) as temporary:
        stage = Path(temporary)/'bundle'
        stage.mkdir()
        for name, text in contents.items():
            (stage/name).write_text(text, encoding='utf-8', newline='')
        manifest = {'schema': SCHEMA, 'analysis_run_id': run_id,
            'files': {name: hashlib.sha256((stage/name).read_bytes()).hexdigest()
                      for name in sorted(FILES)}}
        (stage/'manifest.json').write_text(_json(manifest), encoding='utf-8')
        verify_inspection_export(stage)
        stage.rename(destination)
    return {'output': str(destination), 'analysis_run_id': run_id, 'counts': audit['counts'],
            'data_issues': audit['data_issues'], 'files_verified': len(FILES)}


def verify_inspection_export(directory):
    """Verify a complete export's bytes; hashes establish integrity, not truth."""
    folder = Path(directory)
    manifest = json.loads((folder/'manifest.json').read_bytes())
    if (manifest.get('schema') != SCHEMA or not isinstance(manifest.get('files'), dict)
            or set(manifest['files']) != FILES):
        raise ValueError('Unsupported or incomplete inspection manifest')
    for name, expected in manifest['files'].items():
        path = folder/name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Export integrity mismatch: {name}')
    payload = json.loads((folder/'inspection.json').read_bytes())
    if payload.get('schema') != SCHEMA or payload.get('analysis_run_id') != manifest.get('analysis_run_id'):
        raise ValueError('Export run/schema binding mismatch')
    return {'status': 'verified', 'analysis_run_id': manifest['analysis_run_id'],
            'files_verified': len(FILES)}
