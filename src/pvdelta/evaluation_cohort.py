"""Fixed, spatially separated evaluation cases and explicitly incomplete scoring."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

from pvdelta.instance_benchmark import evaluate_panel_instances


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def stratum(row):
    if row['finding_count']:
        return 'candidate'
    return 'no_candidate' if row['analyzed_observations'] else 'no_analyzed_modules'


def _farthest(pool, count, seed):
    """Deterministic geographic spread, balancing available heading families."""
    selected = []
    pool = sorted(pool, key=lambda r: digest([seed, r['source_sha256']]))
    while len(selected) < count:
        counts = Counter(r['heading_family'] for r in selected)
        available = {r['heading_family'] for r in pool}
        least = min(counts[h] for h in available)
        eligible = [r for r in pool if counts[r['heading_family']] == least]
        def distance(row):
            if not selected:
                return 0
            return min(math.hypot(row['east_m']-s['east_m'], row['north_m']-s['north_m'])
                       for s in selected)
        winner = max(eligible, key=distance)
        selected.append(winner)
        pool.remove(winner)
    return selected


def select_cohort(rows, *, seed, per_stratum=10, buffer_m=50, regression_hashes=()):
    """Select 30+30 by default; camera-space separation is not surveyed non-overlap.

    Only original nadir M4T grids with valid paired hashes/GPS enter this cohort.
    Known regression hashes are excluded before geographic partitioning. This is
    a locked future-development split of an exposed flight, not an unseen flight.
    """
    if (type(per_stratum) is not int or per_stratum < 1 or not isinstance(seed, str)
            or not seed or type(buffer_m) not in (int, float)
            or not math.isfinite(buffer_m) or buffer_m <= 0):
        raise ValueError('Positive cohort size, finite buffer and explicit seed required')
    seen, eligible, excluded = set(), [], []
    for original in rows:
        row = dict(original)
        source = row.get('source_sha256')
        if not _sha(source) or source in seen:
            raise ValueError('Unique original source hashes required')
        seen.add(source)
        for key in ('analyzed_observations', 'finding_count'):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError('Nonnegative frozen observation/finding counts required')
        if (type(row.get('native_width')) is not int or type(row.get('native_height')) is not int
                or row['native_width'] < 1 or row['native_height'] < 1):
            raise ValueError('Positive native grid required')
        reason = None
        lat, lon = row.get('latitude'), row.get('longitude')
        numbers = (lat, lon, row.get('yaw_deg'), row.get('pitch_deg'))
        if source in regression_hashes:
            reason = 'known_regression'
        elif row.get('camera_model') != 'M4T' or (row['native_width'], row['native_height']) != (640, 512):
            reason = 'outside_supported_camera_grid'
        elif not _sha(row.get('rgb_sha256')):
            reason = 'missing_original_rgb_pair'
        elif (any(type(n) not in (int, float) or not math.isfinite(n) for n in numbers)
              or not -90 <= lat <= 90 or not -180 <= lon <= 180 or (lat == 0 and lon == 0)):
            reason = 'invalid_camera_position_or_orientation'
        elif abs(row['pitch_deg']+90) > 2:
            reason = 'outside_nadir_cohort'
        if reason:
            excluded.append({'source_sha256': source, 'reason': reason})
            continue
        row['stratum'] = stratum(row)
        row['heading_family'] = 'forward' if math.cos(math.radians(row['yaw_deg'])) >= 0 else 'return'
        eligible.append(row)
    if not eligible:
        raise ValueError('No eligible original nadir pairs')
    eligible.sort(key=lambda r: r['source_sha256'])
    origin_lat = sum(r['latitude'] for r in eligible)/len(eligible)
    origin_lon = sum(r['longitude'] for r in eligible)/len(eligible)
    for row in eligible:
        row['east_m'] = (row['longitude']-origin_lon)*111320*math.cos(math.radians(origin_lat))
        row['north_m'] = (row['latitude']-origin_lat)*111320
    axes = sorted(('east_m', 'north_m'),
                  key=lambda a: (-(max(r[a] for r in eligible)-min(r[a] for r in eligible)), a))
    strata = ('candidate', 'no_candidate', 'no_analyzed_modules')
    selection = None
    for axis in axes:
        midpoint = (min(r[axis] for r in eligible)+max(r[axis] for r in eligible))/2
        pools = {name: [r for r in eligible if sign*(r[axis]-midpoint) >= buffer_m/2]
                 for name, sign in (('development', -1), ('locked', 1))}
        if all(sum(r['stratum'] == s for r in pools[name]) >= per_stratum
               for name in pools for s in strata):
            selection = []
            for name, pool in pools.items():
                for s in strata:
                    chosen = _farthest([r for r in pool if r['stratum'] == s], per_stratum,
                                       seed+':'+name+':'+s)
                    for row in chosen:
                        row['split'] = name
                    selection.extend(chosen)
            break
    if selection is None:
        raise ValueError('Insufficient spatially separated cases; do not silently shrink the buffer')
    selection.sort(key=lambda r: (r['split'], r['stratum'], r['source_sha256']))
    for index, row in enumerate(selection, 1):
        row['case_id'] = f'C{index:03d}'
    dev = [r for r in selection if r['split'] == 'development']
    locked = [r for r in selection if r['split'] == 'locked']
    minimum = min(math.hypot(a['east_m']-b['east_m'], a['north_m']-b['north_m'])
                  for a in dev for b in locked)
    return {'schema': 'pvdelta.fixed-evaluation-cohort/v1', 'seed': seed,
            'selection': 'spatial_block_then_stratum_then_heading_balanced_farthest_camera_position',
            'axis': axis, 'midpoint_m': midpoint, 'camera_buffer_m': buffer_m,
            'minimum_cross_split_camera_distance_m': minimum,
            'origin_latitude': origin_lat, 'origin_longitude': origin_lon,
            'independent_flight': False, 'surveyed_footprint_nonoverlap_verified': False,
            'prior_exposure_audited': False, 'input_rows': len(rows),
            'eligible_rows': len(eligible), 'excluded': excluded, 'frames': selection,
            'limits': 'Same previously exposed flight. Camera-space buffer is not verified panel-footprint separation.'}


def score_panels(manifest, predictions, references, *, split='development', unlock_locked=False):
    """Score explicit references only; pending is never an empty/healthy frame."""
    if manifest.get('schema') != 'pvdelta.fixed-evaluation-cohort/v1' or split not in ('development', 'locked'):
        raise ValueError('A fixed evaluation cohort and explicit split required')
    if split == 'locked' and not unlock_locked:
        raise ValueError('Locked evaluation requires explicit release after annotations are frozen')
    if (manifest.get('predictions_sha256') != digest(predictions)
            or predictions.get('schema') != 'pvdelta.frozen-panel-predictions/v1'):
        raise ValueError('Frozen predictions differ from the manifest')
    cases = [r for r in manifest['frames'] if r['split'] == split]
    if not cases or len({r['source_sha256'] for r in manifest['frames']}) != len(manifest['frames']):
        raise ValueError('A nonempty split of distinct sources required')
    refs, preds = {}, {}
    for items, target in ((references, refs), (predictions['frames'], preds)):
        for item in items:
            source = item.get('source_sha256')
            if source in target or not _sha(source):
                raise ValueError('Reference and prediction sources must be distinct SHA256 values')
            target[source] = item
    if set(refs)-{r['source_sha256'] for r in manifest['frames']}:
        raise ValueError('Reference source is outside the selected cohort')
    pending = [r['case_id'] for r in cases if refs.get(r['source_sha256'], {}).get('panels') is None
               or refs[r['source_sha256']].get('coverage') != 'exhaustive_full_frame'
               or refs[r['source_sha256']].get('review_status') not in ('developer_draft', 'independently_reviewed')]
    if split == 'locked' and pending:
        raise ValueError('Complete and freeze all locked references before opening scores')
    reports = []
    for case in cases:
        if case['case_id'] in pending:
            continue
        source = case['source_sha256']
        prediction = preds.get(source)
        if prediction is None or prediction.get('native_shape') != [case['native_height'], case['native_width']]:
            raise ValueError('Frozen prediction missing or native grid changed')
        result = evaluate_panel_instances(refs[source], prediction['proposals'], source_sha256=source,
                  native_shape=(case['native_height'], case['native_width']),
                  iou_threshold=manifest['acceptance']['panel_iou_threshold'],
                  ignore_fraction=manifest['acceptance']['ignore_fraction'])
        reports.append({'case_id': case['case_id'], **result})
    totals = {k: sum(r[k] for r in reports) for k in ('true_positives', 'false_positives', 'false_negatives')}
    tp, fp, fn = (totals[k] for k in ('true_positives', 'false_positives', 'false_negatives'))
    return {'schema': 'pvdelta.fixed-panel-score/v1', 'split': split, 'selected_frames': len(cases),
            'scored_frames': len(reports), 'pending_frames': pending,
            'status': 'references_pending' if pending else 'reference_geometry_scored',
            'totals': totals if reports else None, 'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None, 'frames': reports,
            'references_sha256': digest(references), 'predictions_sha256': digest(predictions),
            'independent_flight': False, 'stage1_accepted': False,
            'scope': 'Full-module geometry against declared references; not anomaly accuracy, identity or field approval'}
