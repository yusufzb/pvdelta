"""Native cross-frame geometry diagnostics; never physical identity or glare approval."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def native_motion_check(source_gray, target_gray, source_geometry, target_geometry):
    """Initialize from image texture; check separate, observed panel controls.

    Feature check roles are frozen before fitting or tracking. The existing four-fold module
    control gate is unchanged. No temperature or candidate location is fitted.
    """
    import cv2

    from pvdelta.panel_pair_check import observed_panel_pair_check

    a, b = np.asarray(source_gray), np.asarray(target_gray)
    if (a.ndim != 2 or a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8
            or min(a.shape) < 32):
        raise ValueError('Equal native-size uint8 grayscale images required')
    result = {'status': 'withheld', 'reasons': [], 'feature_pairs': 0,
        'fit_inliers': 0, 'feature_check_count': 0, 'feature_check_p95_px': None,
        'panel_controls': None, 'control_check_passed': False,
        'same_panel_verified': False, 'measurement_authorised': False, 'glare_clearance': False}

    def reject(reason):
        result['reasons'] = [reason]
        return result

    sift = cv2.SIFT_create(nfeatures=3000)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None or min(len(ka), len(kb)) < 24:
        return reject('insufficient_native_image_features')
    matcher = cv2.BFMatcher()
    forward = [p[0] for p in matcher.knnMatch(da, db, k=2)
               if len(p) == 2 and p[0].distance < .7*p[1].distance]
    reverse = {(p[0].trainIdx, p[0].queryIdx) for p in matcher.knnMatch(db, da, k=2)
               if len(p) == 2 and p[0].distance < .7*p[1].distance}
    pairs = sorted((p for p in forward if (p.queryIdx, p.trainIdx) in reverse),
                   key=lambda p: (ka[p.queryIdx].pt[1], ka[p.queryIdx].pt[0], p.queryIdx))
    result['feature_pairs'] = len(pairs)
    if len(pairs) < 24:
        return reject('insufficient_mutual_native_features')
    xy = np.float32([ka[p.queryIdx].pt for p in pairs])
    uv = np.float32([kb[p.trainIdx].pt for p in pairs])
    check = np.arange(len(pairs)) % 4 == 0
    cv2.setRNGSeed(0)
    matrix, inliers = cv2.findHomography(xy[~check], uv[~check], cv2.RANSAC, 2.)
    if (matrix is None or inliers is None or inliers.sum() < 12
            or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12):
        return reject('native_initializer_has_insufficient_support')
    # Descriptor matches initialize motion, not the final correspondence labels.
    # Keep the original check roles; validate optical tracks independently in
    # both directions and by normalized local texture before the final fit.
    params = {'winSize': (21, 21), 'maxLevel': 2,
        'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01)}
    starts = xy[:, None].copy()
    # Return tracks can rotate the camera by 180 degrees. LK assumes local
    # translation; an initial coordinate alone does not rotate its patches.
    # Compensate coarse rotation/scale on a DISPLAY copy, then map tracks back
    # to the original target grid. Native temperature arrays are never warped.
    aligned = cv2.warpPerspective(b, np.linalg.inv(matrix), (a.shape[1], a.shape[0]))
    overlap = cv2.warpPerspective(np.full_like(b, 255), np.linalg.inv(matrix),
        (a.shape[1], a.shape[0]), flags=cv2.INTER_NEAREST)
    overlap = cv2.erode(overlap, np.ones((15, 15), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    tracked, ok, _ = cv2.calcOpticalFlowPyrLK(a, aligned, starts, starts.copy(),
                                            flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **params)
    returned, back_ok, _ = cv2.calcOpticalFlowPyrLK(aligned, a, tracked, starts.copy(),
                                                 flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **params)
    valid = ((ok.ravel() == 1) & (back_ok.ravel() == 1)
        & (np.linalg.norm(returned[:, 0]-xy, axis=1) < .75)
        & (xy[:, 0] >= 8) & (xy[:, 1] >= 8)
        & (xy[:, 0] < a.shape[1]-8) & (xy[:, 1] < a.shape[0]-8))
    for i in np.flatnonzero(valid):
        tx, ty = np.rint(tracked[i, 0]).astype(int)
        if not (0 <= tx < a.shape[1] and 0 <= ty < a.shape[0] and overlap[ty, tx] == 255):
            valid[i] = False
            continue
        patch_a = cv2.getRectSubPix(a, (15, 15), tuple(xy[i])).astype(float)
        patch_b = cv2.getRectSubPix(aligned, (15, 15), tuple(tracked[i, 0])).astype(float)
        patch_a -= patch_a.mean()
        patch_b -= patch_b.mean()
        denom = np.linalg.norm(patch_a)*np.linalg.norm(patch_b)
        valid[i] = denom > 1 and np.sum(patch_a*patch_b)/denom >= .8
    fit, heldout = valid & ~check, valid & check
    result.update(validated_tracks=int(valid.sum()), feature_check_count=int(heldout.sum()))
    if fit.sum() < 12 or heldout.sum() < 6:
        return reject('insufficient_forward_backward_native_tracks')
    original_target = cv2.perspectiveTransform(tracked, matrix)[:, 0]
    cv2.setRNGSeed(0)
    matrix, inliers = cv2.findHomography(xy[fit], original_target[fit], cv2.RANSAC, 2.)
    if (matrix is None or inliers is None or inliers.sum() < 12
            or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12):
        return reject('native_local_fit_has_insufficient_support')
    result['fit_inliers'] = int(inliers.sum())
    predicted = cv2.perspectiveTransform(xy[heldout, None], matrix)[:, 0]
    error = np.linalg.norm(predicted-original_target[heldout], axis=1)
    result.update(feature_check_p95_px=float(np.percentile(error, 95)))
    if not np.isfinite(error).all() or result['feature_check_p95_px'] > 3.:
        return reject('heldout_native_features_disagree')
    controls = observed_panel_pair_check(source_geometry, target_geometry, matrix, native_shape=a.shape)
    result['panel_controls'] = controls
    result['control_check_passed'] = controls['control_check_passed']
    result['status'] = 'geometry_consistent_provisional' if controls['control_check_passed'] else 'withheld'
    result['reasons'] = controls['reasons']
    # Intentionally no transform or transferred anomaly polygon is published.
    return result


def _snapshot(database, finding_id, sdk_library, max_views, search_mode='camera_gps', max_distance_m=20.):
    import cv2

    from pvdelta.measurement_profile import read_measurement_profile
    from pvdelta.repeat_candidates import select_repeat_pairs
    from pvdelta.store import connect_store

    rows = select_repeat_pairs(database, finding_id, max_views=max_views,
                              search_mode=search_mode, max_distance_m=max_distance_m)
    connection = connect_store(database, read_only=True)
    try:
        identity = connection.execute('''SELECT f.analysis_run_id,m.capture_id,m.frame_module_id
            FROM findings f JOIN module_observations m USING(observation_id)
            WHERE f.finding_id=?''', [finding_id]).fetchone()
        if not identity or not rows or rows[0]['capture_id'] != identity[1]:
            raise ValueError('Source thermal/RGB pair unavailable in this acquisition')
        frames = []
        for row in rows:
            record = connection.execute('SELECT source_hash FROM captures WHERE capture_id=?',
                                        [row['capture_id']]).fetchone()
            actual = hashlib.sha256(Path(row['thermal_path']).read_bytes()).hexdigest()
            if not record or actual != record[0]:
                raise ValueError('Thermal source differs from indexed source: '+Path(row['thermal_path']).name)
            profile = read_measurement_profile(connection, identity[0], row['capture_id'])
            diagnostic = connection.execute('''SELECT diagnostics_json FROM flight_screening_frames
                WHERE analysis_run_id=? AND capture_id=?''', [identity[0], row['capture_id']]).fetchone()
            diagnostic = json.loads(diagnostic[0]) if diagnostic and diagnostic[0] else {}
            frames.append(dict(row, source_sha256=actual, profile=profile, diagnostic=diagnostic))
    finally:
        connection.close()
    configured = sdk_library or os.environ.get('PVDELTA_DIRP_LIB')
    sdk = Path(configured).expanduser().resolve() if configured else None
    binaries = sorted(p for p in sdk.glob('*.dll') if p.is_file()) if sdk and sdk.is_dir() else [sdk] if sdk else []
    if sdk and sdk.is_dir():
        binaries += sorted(p for p in sdk.glob('*.so*') if p.is_file())
    binding = {'finding_id': finding_id, 'analysis_run_id': identity[0], 'source_module': identity[2],
        'max_views': max_views, 'search_mode': search_mode, 'max_distance_m': max_distance_m,
        'frames': frames, 'sdk_path': str(sdk), 'opencv': cv2.__version__, 'numpy': np.__version__,
        'sdk_binaries': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries},
        'engine_sources': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('native_repeat.py', 'panel_pair_check.py', 'native_completion.py',
                         'native_level_regions.py', 'measurement_profile.py', 'decode.py', 'repeat_views.py',
                         'convex_geometry.py', 'regions.py', 'repeat_candidates.py', 'tracking.py',
                         'junction_registration.py')}}
    return frames, _digest(binding), identity[2]


def native_repeat_signature(database, finding_id, sdk_library, *, max_views=8,
                            search_mode='camera_gps', max_distance_m=20.):
    return _snapshot(database, finding_id, sdk_library, max_views, search_mode, max_distance_m)[1]


def _native_gray(row, sdk_library):
    """Derive motion structure from verified temperatures, never display colours."""
    from pvdelta.decode import RJpeg
    from pvdelta.junction_registration import radiometric_structure
    from pvdelta.measurement_profile import apply_saved_measurement_profile

    if row['profile'] is None:
        raise ValueError('Saved native measurement profile required')
    with RJpeg(row['thermal_path'], sdk_library) as reader:
        values = apply_saved_measurement_profile(reader, row['thermal_path'], row['profile'])
    # Palettes need not preserve temperature order, and can vary between frames.
    # This monotonic, clipped copy is only for motion; measured pixels stay native.
    image, _ = radiometric_structure(values)
    return image


def native_repeat_checks(database, finding_id, *, sdk_library=None, max_views=8,
                         search_mode='camera_gps', max_distance_m=20.):
    """Read-only, bounded SDK/structure checks without a legacy RGB/site prerequisite."""
    import cv2

    rows, binding, source_module = _snapshot(database, finding_id, sdk_library, max_views,
                                           search_mode, max_distance_m)
    source = rows[0]
    source_geometry = source['diagnostic'].get('native_geometry')
    if not isinstance(source_geometry, dict):
        raise TypeError('Source run has no saved native structural geometry')
    gray = _native_gray(source, sdk_library)
    output = []
    for target in rows[1:]:
        base = {'capture_id': target['capture_id'], 'thermal_file': Path(target['thermal_path']).name,
            'time_gap_s': target['time_gap_s'], 'status': 'unavailable', 'reasons': [],
            'camera_distance_m': target.get('camera_distance_m'),
            'control_check_passed': False, 'selected_control_target': None,
            'same_panel_verified': False, 'measurement_authorised': False, 'glare_clearance': False}
        try:
            geometry = target['diagnostic'].get('native_geometry')
            if not isinstance(geometry, dict):
                raise TypeError('Target has no saved native structural geometry')
            report = native_motion_check(gray, _native_gray(target, sdk_library), source_geometry, geometry)
            base.update(report)
            controls = report.get('panel_controls') or {}
            selected = next((p for p in controls.get('matches', []) if p['source_id'] == source_module), None)
            if selected:
                base['selected_control_target'] = selected['target_id']
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, cv2.error) as exc:
            base['reasons'] = [str(exc)]
        output.append(base)
    if _snapshot(database, finding_id, sdk_library, max_views, search_mode, max_distance_m)[1] != binding:
        raise ValueError('Sources or stored geometry changed during native comparison')
    return {'schema': 'pvdelta.native-repeat-checks/v1', 'binding': binding,
        'motion_input': {'source': 'verified_native_sdk_temperature_c',
            'mapping': 'monotonic_uint8', 'clip_percentiles': [2, 98],
            'normalization_scope': 'each_frame', 'purpose': 'geometry_only',
            'measurement_array_modified': False, 'palette_used': False},
        'source_file': Path(source['thermal_path']).name, 'source_module': source_module,
        'search_method': source.get('search_method', search_mode), 'max_views': max_views,
        'camera_search_radius_m': source.get('camera_search_radius_m'),
        'finding_id': finding_id, 'rows': output, 'same_panel_verified': False,
        'measurement_authorised': False, 'glare_clearance': False, 'database_written': False}
