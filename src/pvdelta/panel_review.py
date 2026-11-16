"""Review V2: group regions by exact run-scoped panel observation, never guessed asset identity."""
from __future__ import annotations


def panel_groups(connection, run_id):
    result = connection.execute('''SELECT f.*, m.capture_id,m.frame_module_id,
        m.pixel_polygon_json AS module_polygon,c.source_path AS thermal_path,c.source_hash AS thermal_hash,
        rgb.source_path AS rgb_path,r.source_root,c.camera_model,
        c.width_px,c.height_px,rgb.source_hash AS rgb_hash,
        coalesce(f.estimated_latitude,m.estimated_latitude) AS latitude,
        coalesce(f.estimated_longitude,m.estimated_longitude) AS longitude,
        coalesce(v.review_status,'unreviewed') AS review_status,coalesce(v.note,'') AS review_note
        FROM findings f JOIN module_observations m USING(observation_id)
        JOIN captures c ON c.capture_id=m.capture_id
        LEFT JOIN captures rgb ON rgb.capture_id=c.paired_capture_id
        LEFT JOIN inspection_runs r ON r.run_id=c.run_id
        LEFT JOIN finding_reviews v USING(finding_id)
        WHERE f.analysis_run_id=? AND m.analysis_run_id=?
        ORDER BY c.source_path,m.frame_module_id,f.finding_id''', [run_id, run_id])
    columns = [c[0] for c in result.description]
    groups = {}
    for values in result.fetchall():
        row = dict(zip(columns, values, strict=True))
        key = row['observation_id']
        if key not in groups:
            groups[key] = {'observation_id': key, 'capture_id': row['capture_id'],
                'module_id': row['frame_module_id'], 'thermal_path': row['thermal_path'],
                'analysis_run_id': run_id, 'regions': []}
        group = groups[key]
        if group['capture_id'] != row['capture_id'] or group['module_id'] != row['frame_module_id']:
            raise ValueError('Observation identity is inconsistent; grouping withheld')
        group['regions'].append(row)
    for group in groups.values():
        statuses = {r['review_status'] for r in group['regions']}
        group['review_status'] = next(iter(statuses)) if len(statuses) == 1 else 'mixed'
        group['region_count'] = len(group['regions'])
        group['max_delta_t_k'] = max(r['delta_t_k'] for r in group['regions'])
    return list(groups.values())


def filter_panel_groups(groups, selection):
    """Filter whole cards; never remove an individual region from a selected card."""
    if selection == 'All':
        return groups
    if selection == 'Contains unreviewed':
        return [g for g in groups if any(r['review_status'] == 'unreviewed' for r in g['regions'])]
    if selection == 'Contains Wait':
        return [g for g in groups if any(r['review_status'] == 'wait' for r in g['regions'])]
    status = {'All OK': 'ok', 'All Not OK': 'not_ok', 'Mixed decisions': 'mixed'}[selection]
    return [g for g in groups if g['review_status'] == status]


def save_panel_review(connection, run_id, observation_id, expected_finding_ids, status, *,
                      note=None, target_finding_id=None):
    """Atomically review the displayed population; reject stale group membership."""
    from pvdelta.store import upsert_finding_review

    expected = set(expected_finding_ids)
    if not expected or len(expected) != len(expected_finding_ids):
        raise ValueError('A nonempty, distinct displayed finding population is required')
    if target_finding_id is not None and target_finding_id not in expected:
        raise ValueError('Review target is outside the displayed panel')
    connection.execute('BEGIN TRANSACTION')
    try:
        actual = connection.execute('''SELECT f.finding_id,v.note FROM findings f
            LEFT JOIN finding_reviews v USING(finding_id)
            WHERE f.analysis_run_id=? AND f.observation_id=?''', [run_id, observation_id]).fetchall()
        if {r[0] for r in actual} != expected:
            raise ValueError('Panel findings changed since display; refresh before reviewing')
        for finding_id, old_note in actual:
            if target_finding_id is None or target_finding_id == finding_id:
                upsert_finding_review(connection, finding_id=finding_id, review_status=status,
                    note=note if note is not None else old_note)
        connection.execute('COMMIT')
    except Exception:
        connection.execute('ROLLBACK')
        raise
