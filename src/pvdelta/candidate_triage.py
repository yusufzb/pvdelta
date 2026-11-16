"""Read-only candidate support triage, never a fault or vegetation classifier."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np

from pvdelta.native_diagnostics import read_native_diagnostic
from pvdelta.reference import erode
from pvdelta.region_polygon import simple_polygon_mask
from pvdelta.regions import polygon_to_mask
from pvdelta.support_partition import partition_measurement_support

LABELS = {'interior': 'Interior candidate', 'boundary': 'Boundary review',
          'unavailable': 'Support unavailable'}


def region_support(mask, pixels, expected_count):
    """Use exact component pixels, not the enlarged display rectangle."""
    mask = np.asarray(mask)
    points = np.asarray(pixels)
    if (mask.ndim != 2 or mask.dtype != np.bool_ or not mask.any()
            or points.ndim != 2 or points.shape[1] != 2 or not len(points)
            or not np.issubdtype(points.dtype, np.integer)
            or len(points) != expected_count or len(np.unique(points, axis=0)) != len(points)
            or (points < 0).any() or (points[:, 0] >= mask.shape[0]).any()
            or (points[:, 1] >= mask.shape[1]).any()):
        raise ValueError('Exact distinct native component pixels and nonempty support required')
    yy, xx = points.T
    if not mask[yy, xx].all():
        raise ValueError('Candidate pixels extend outside saved measurement support')
    edge = mask & ~erode(mask, 1)
    touches = int(edge[yy, xx].sum())
    return {'support': 'boundary' if touches else 'interior',
            'boundary_pixels': touches, 'region_pixels': len(points),
            'measurement_pixels': int(mask.sum()), 'region_fraction': len(points) / int(mask.sum()),
            'reason': ('Component touches the effective measurement boundary; check panel edges and background.'
                       if touches else 'Component is inside the effective mask; panel membership and glare remain unverified.')}


def _saved_surface_veto(connection, bound):
    diagnostic = bound['diagnostic']
    policy = diagnostic['declared_surface_policy']
    constraint = diagnostic.get('surface_constraint', {})
    source_hash = bound['source_sha256']
    shape = bound['native_shape']
    row = connection.execute('SELECT path FROM duckdb_databases() WHERE database_name=current_database()').fetchone()
    if not row or not row[0]:
        raise ValueError('Saved surface constraints require a file-backed inspection')
    path = Path(row[0]).with_name('surface-references.json')
    try:
        snapshot = json.loads(path.read_bytes())
        reference = snapshot['references'][source_hash]
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError('Saved surface reference unavailable') from exc
    reference_hash = hashlib.sha256(
        json.dumps(reference, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if (snapshot.get('manifest_sha256') != constraint.get('manifest_sha256')
            or policy['source_sha256'] != source_hash
            or reference_hash != policy['reference_sha256']
            or reference.get('review_complete') is not True
            or reference.get('source_sha256') != source_hash
            or reference.get('native_temperature_sha256') != policy['native_temperature_sha256']
            or [reference.get('native_height'), reference.get('native_width')] != shape):
        raise ValueError('Saved surface reference differs from analysis binding')
    veto = np.zeros(shape, bool)
    for polygon in reference['non_panel_polygons']:
        veto |= simple_polygon_mask(polygon['polygon_native_xy'], shape)
    return veto


def native_support_masks(connection, run_id, capture_id, erosion):
    """Rebuild every capture mask together and verify recorded mask hashes."""
    bound = read_native_diagnostic(connection, run_id, capture_id)
    diagnostic = bound['diagnostic']
    proposals = diagnostic['native_geometry']['proposals']
    modules = {m['module']: m for m in diagnostic['modules']}
    if len(modules) != len(proposals):
        raise ValueError('Saved mask population differs from geometry')
    margins = {m.get('gate', {}).get('margin_px') for m in modules.values()}
    if len(margins) != 1 or any(type(v) is not int or v < 0 for v in margins):
        raise ValueError('One explicit saved partition margin is required')
    supports = partition_measurement_support([
        polygon_to_mask(p['polygon_native_xy'], bound['native_shape']) for p in proposals],
        margin_px=next(iter(margins)))
    veto = (_saved_surface_veto(connection, bound)
            if diagnostic.get('declared_surface_policy', {}).get('masked_observations', 0)
            else None)
    result = {}
    for proposal, support in zip(proposals, supports, strict=True):
        mid = proposal['proposal_id']
        interior = support.interior if veto is None else support.interior & ~veto
        if (modules[mid]['mask_sha256'] != hashlib.sha256(interior.tobytes()).hexdigest()
                or modules[mid]['interior_pixels'] != int(interior.sum())):
            raise ValueError('Rebuilt support differs from saved analysis mask')
        result[mid] = erode(interior, erosion)
    return result


def candidate_support_audit(connection, database, run_id):
    """No writes, no count target, no review or cross-frame identity inference."""
    rows = connection.execute('''SELECT f.finding_id,m.capture_id,m.frame_module_id,
        f.region_pixels_yx_json,f.n_region_px FROM findings f
        JOIN module_observations m USING(observation_id)
        WHERE f.analysis_run_id=? AND m.analysis_run_id=? ORDER BY f.finding_id''',
        [run_id, run_id]).fetchall()
    result = {r[0]: {'support': 'unavailable', 'reason': 'Saved native mask evidence unavailable.'} for r in rows}
    try:
        summary = json.loads(Path(database).with_name('summary.json').read_bytes())
        erosion = summary['settings']['edge_erosion_px']
        if (summary.get('schema') != 'pvdelta.native-flight/v1'
                or summary.get('analysis_run_id') != run_id
                or type(erosion) is not int or erosion < 0):
            raise ValueError('Saved run settings do not match selected native analysis')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        for record in result.values():
            record['reason'] = f'Support unavailable: {exc}'
        return result
    by_capture = {}
    for row in rows:
        by_capture.setdefault(row[1], []).append(row)
    for capture, findings in by_capture.items():
        try:
            masks = native_support_masks(connection, run_id, capture, erosion)
        except (ValueError, KeyError, TypeError, duckdb.Error) as exc:
            for row in findings:
                result[row[0]]['reason'] = f'Support unavailable: {exc}'
            continue
        for fid, _, mid, encoded, count in findings:
            try:
                result[fid] = region_support(masks[mid], json.loads(encoded or 'null'), count)
            except (ValueError, KeyError, TypeError) as exc:
                result[fid]['reason'] = f'Support unavailable: {exc}'
    return result


def support_groups(groups, audit, selection):
    """Keep a whole card if any member matches, preserving all sibling evidence."""
    if selection == 'All candidates':
        return groups
    wanted = {'Interior candidates': 'interior', 'Boundary review': 'boundary',
              'Support unavailable': 'unavailable'}[selection]
    return [g for g in groups if any(audit[r['finding_id']]['support'] == wanted for r in g['regions'])]
