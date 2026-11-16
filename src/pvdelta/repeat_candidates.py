"""Camera-GPS retrieval for cross-frame inspection, never physical panel identity."""
from __future__ import annotations

import math
from pathlib import Path


def _distance(a_lat, a_lon, b_lat, b_lon):
    values = (a_lat, a_lon, b_lat, b_lon)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        return None
    if not (-90 <= a_lat <= 90 and -90 <= b_lat <= 90
            and -180 <= a_lon <= 180 and -180 <= b_lon <= 180):
        return None
    a, b = math.radians(a_lat), math.radians(b_lat)
    dlat, dlon = b-a, math.radians(b_lon-a_lon)
    h = math.sin(dlat/2)**2+math.cos(a)*math.cos(b)*math.sin(dlon/2)**2
    return 2*6_371_008.8*math.asin(math.sqrt(min(1., max(0., h))))


def select_repeat_pairs(database, finding_id, *, max_views=8, search_mode='camera_gps', max_distance_m=20.):
    """Source plus bounded nearest camera positions in the same acquisition.

    Capture time does not veto a spatial candidate; return-track captures are
    otherwise missed. Missing source GPS falls back to the existing time search.
    Camera proximity is only retrieval evidence, never panel correspondence.
    """
    from pvdelta.repeat_views import nearby_image_pairs
    from pvdelta.store import connect_store

    if type(max_views) is not int or not 1 <= max_views <= 32:
        raise ValueError('max_views must be an integer in 1..32')
    if search_mode not in ('camera_gps', 'time'):
        raise ValueError('search_mode must be camera_gps or time')
    if (type(max_distance_m) not in (int, float) or not math.isfinite(max_distance_m)
            or not 0 < max_distance_m <= 100):
        raise ValueError('Finite camera search radius in (0,100] metres required')
    connection = connect_store(database, read_only=True)
    try:
        source = connection.execute('''SELECT c.capture_id,c.run_id,c.camera_latitude,c.camera_longitude
            FROM findings f JOIN module_observations m USING(observation_id)
            JOIN captures c ON c.capture_id=m.capture_id WHERE f.finding_id=?''', [finding_id]).fetchone()
        if source is None:
            raise ValueError('Finding no longer exists in this database')
        cid, acquisition, lat, lon = source
        source_gps = _distance(lat, lon, lat, lon) is not None
        if search_mode == 'time' or not source_gps:
            rows = None
        else:
            result = connection.execute('''SELECT c.capture_id,c.source_path AS thermal_path,
                rgb.source_path AS rgb_path,r.source_root,c.camera_latitude,c.camera_longitude,
                c.gimbal_yaw_deg,c.gimbal_pitch_deg,
                abs(epoch(c.capture_time_utc)-epoch(src.capture_time_utc)) AS time_gap_s
                FROM captures c JOIN captures src ON src.capture_id=?
                JOIN captures rgb ON rgb.capture_id=c.paired_capture_id
                    AND rgb.image_kind='rgb' AND rgb.run_id=c.run_id
                LEFT JOIN inspection_runs r ON r.run_id=c.run_id
                WHERE c.run_id=? AND c.image_kind='thermal' ''', [cid, acquisition])
            columns = [c[0] for c in result.description]
            all_rows = [dict(zip(columns, row, strict=True)) for row in result.fetchall()]
            original = next((r for r in all_rows if r['capture_id'] == cid), None)
            if original is None:
                raise ValueError('Source has no thermal/RGB pair in its acquisition')
            candidates = []
            for row in all_rows:
                distance = _distance(lat, lon, row['camera_latitude'], row['camera_longitude'])
                if row['capture_id'] == cid or distance is None or distance > max_distance_m:
                    continue
                row['camera_distance_m'] = distance
                candidates.append(row)
            candidates.sort(key=lambda r: (r['camera_distance_m'],
                r['time_gap_s'] if r['time_gap_s'] is not None else math.inf, r['capture_id']))
            original['camera_distance_m'] = 0.
            rows = [original, *candidates[:max_views]]
    finally:
        connection.close()
    if rows is None:
        method = 'time' if search_mode == 'time' else 'time_fallback_missing_source_gps'
        rows = nearby_image_pairs(database, finding_id, max_views=max_views)
        for row in rows:
            row.update(search_method=method, camera_distance_m=None, camera_search_radius_m=None)
        return rows
    for row in rows:
        for key in ('thermal_path', 'rgb_path'):
            root = Path(row['source_root'] or '').expanduser().resolve()
            path = (root/row[key]).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Indexed comparison image lies outside its source root')
            row[key] = str(path)
        row.update(search_method='camera_gps', camera_search_radius_m=float(max_distance_m),
            status='unavailable', panel_identity_verified=False, glare_evidence='unresolved',
            reason='camera proximity only; physical panel identity unverified')
    return rows
