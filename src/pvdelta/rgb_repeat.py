"""Observed RGB panel tracking for native runs; no thermal component transfer."""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np

from pvdelta.convex_geometry import convex_iou
from pvdelta.rgb_pair_check import digest, match_module_quads


def _file_hash(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _snapshot(database, finding_id, registration, max_views, sdk_library=None, *, hash_file=None,
              table_context=False):
    import cv2

    from pvdelta.measurement_profile import read_measurement_profile
    from pvdelta.repeat_candidates import select_repeat_pairs
    from pvdelta.store import connect_store

    if type(table_context) is not bool:
        raise ValueError('table_context must be boolean')
    hash_file = hash_file or _file_hash
    rows = select_repeat_pairs(database, finding_id, max_views=max_views)
    c = connect_store(database, read_only=True)
    try:
        identity = c.execute('''SELECT f.analysis_run_id,m.capture_id,m.frame_module_id,m.pixel_polygon_json
            FROM findings f JOIN module_observations m USING(observation_id) WHERE f.finding_id=?''',
            [finding_id]).fetchone()
        if not identity or not rows or rows[0]['capture_id'] != identity[1]:
            raise ValueError('Source pair is unavailable')
        for row in rows:
            source = c.execute('''SELECT t.source_hash,t.camera_model,t.width_px,t.height_px,
                rgb.source_hash,rgb.width_px,rgb.height_px FROM captures t
                JOIN captures rgb ON rgb.capture_id=t.paired_capture_id WHERE t.capture_id=?''',
                [row['capture_id']]).fetchone()
            row.update(thermal_sha256=source[0], camera_model=source[1], native_size=list(source[2:4]),
                rgb_sha256=source[4], rgb_size=list(source[5:7]), actual_rgb_sha256=hash_file(row['rgb_path']))
            row['actual_thermal_sha256'] = hash_file(row['thermal_path'])
        source = rows[0]
        source['actual_thermal_sha256'] = hash_file(source['thermal_path'])
        source['profile'] = read_measurement_profile(c, identity[0], identity[1])
        raw = c.execute('''SELECT diagnostics_json FROM flight_screening_frames
            WHERE analysis_run_id=? AND capture_id=?''', identity[:2]).fetchone()
        source['diagnostic'] = json.loads(raw[0]) if raw and raw[0] else {}
        source['module_id'], source['module_polygon'] = identity[2], json.loads(identity[3])
    finally:
        c.close()
    calibration = Path(registration['path']) if registration and registration.get('path') else None
    calibration_hash = hash_file(calibration) if calibration else None
    scope_path = (Path(registration.get('flight_binding') or calibration.with_name('rgb-search-scope.json'))
                  if calibration else None)
    calibration_name = registration.get('calibration_frame') if registration else None
    anchor = (Path(source['source_root'])/calibration_name if isinstance(calibration_name, str)
        and Path(calibration_name).name == calibration_name else None)
    configured = sdk_library or os.environ.get('PVDELTA_DIRP_LIB')
    sdk = Path(configured).expanduser().resolve() if configured else None
    binaries = (sorted(p for p in sdk.iterdir() if p.is_file() and (p.suffix == '.dll' or '.so' in p.name))
        if sdk and sdk.is_dir() else [sdk] if sdk else [])
    binding = digest({'database': str(Path(database).resolve()), 'finding_id': finding_id,
        'rows': rows, 'registration': registration, 'calibration_sha256': calibration_hash,
        'table_context': table_context,
        'flight_binding_sha256': hash_file(scope_path) if scope_path else None,
        'opencv': cv2.__version__, 'sdk_path': str(sdk), 'sdk_binaries': {str(p): hash_file(p) for p in binaries},
        'calibration_anchor_sha256': hash_file(anchor) if anchor else None,
        'engine_sources': {name: hash_file(Path(__file__).with_name(name)) for name in
            ('rgb_repeat.py', 'rgb_pair_check.py', 'rgb_preview.py', 'repeat_views.py',
             'repeat_candidates.py', 'table_boundaries.py', 'boundaries.py', 'measurement_profile.py',
             'decode.py', 'convex_geometry.py', 'native_boundaries.py')}})
    return rows, binding


def rgb_repeat_signature(database, finding_id, registration, *, sdk_library=None, max_views=8,
                         table_context=False):
    options = {'table_context': True} if table_context else {}
    if type(table_context) is not bool:
        raise ValueError('table_context must be boolean')
    return _snapshot(database, finding_id, registration, max_views, sdk_library, **options)[1]


@lru_cache(maxsize=16)
def _observed(path, source_sha256, pitch):
    import cv2

    from pvdelta.table_boundaries import refined_module_proposals

    encoded = Path(path).read_bytes()
    if hashlib.sha256(encoded).hexdigest() != source_sha256:
        raise ValueError('Original RGB source changed before extraction')
    bgr = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None or bgr.shape != (3024, 4032, 3):
        raise ValueError('Original M4T RGB grid required for this flight-specific search prior')
    result = refined_module_proposals(bgr, gimbal_pitch_deg=pitch)
    if result.get('status') != 'unreviewed_rgb_geometry':
        raise ValueError('Observed full-module RGB extractor withheld geometry')
    return result


def _verify_source(source, registration, sdk_library):
    from pvdelta.decode import RJpeg
    from pvdelta.measurement_profile import apply_saved_measurement_profile

    if (source['camera_model'] != 'M4T' or source['native_size'] != [640, 512]
            or source['rgb_size'] != [4032, 3024]):
        raise ValueError('This native-to-RGB search prior is limited to the original M4T grids')
    if (source['actual_rgb_sha256'] != source['rgb_sha256']
            or source['actual_thermal_sha256'] != source['thermal_sha256']):
        raise ValueError('Source thermal/RGB file differs from its indexed identity')
    if not registration or not registration.get('path'):
        raise ValueError('A compatible original-flight RGB search prior is required')
    payload = json.loads(Path(registration['path']).read_text(encoding='utf-8'))
    scope_path = Path(registration.get('flight_binding') or Path(registration['path']).with_name('rgb-search-scope.json'))
    scope = json.loads(scope_path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or not isinstance(scope, dict):
        raise TypeError('Calibration and flight scope must be JSON objects')
    name = registration.get('calibration_frame')
    if (not isinstance(name, str) or Path(name).name != name
            or not (Path(source['source_root'])/name).is_file()
            or payload.get('quadratic_diagnostic', {}).get('accepted') is not True
            or payload['quadratic_diagnostic'].get('forward') != registration.get('model')
            or payload.get('cross_frame_visual_audit', {}).get('calibration_frame') != name):
        raise ValueError('Search prior is not bound to this original flight/calibration file')
    if (scope.get('schema') != 'pvdelta.rgb-search-scope/v1'
            or Path(scope.get('source_root', '')).resolve() != Path(source['source_root']).resolve()
            or scope.get('calibration_sha256') != _file_hash(registration['path'])
            or scope.get('calibration_anchor_sha256') != _file_hash(Path(source['source_root'])/name)):
        raise ValueError('Original-flight RGB search binding differs; cross-flight reuse withheld')
    if source['profile'] is None:
        raise ValueError('Saved source measurement profile required')
    with RJpeg(source['thermal_path'], sdk_library) as reader:
        values = apply_saved_measurement_profile(reader, source['thermal_path'], source['profile'])
    if values.shape != (512, 640):
        raise ValueError('Reconstructed native grid differs from the scoped M4T search grid')


def _table_support(polygon, modules):
    """Freeze one-hop observed neighbours in the same explicitly indexed table."""
    from pvdelta.native_boundaries import _neighbours

    key = lambda p: tuple(sorted(map(tuple, p)))
    selected = [m for m in modules if key(m['polygon_rgb_xy']) == key(polygon)]
    if len(selected) != 1:
        raise ValueError('Selected observed source panel is absent or duplicated')
    support = [deepcopy(polygon)]
    table = selected[0].get('table_index')
    if type(table) is not int or table < 0:
        return support
    for module in modules:
        q = module['polygon_rgb_xy']
        if (type(module.get('table_index')) is int and module['table_index'] == table
                and key(q) != key(polygon) and convex_iou(polygon, q) <= .05
                and _neighbours(np.asarray(polygon, float), np.asarray(q, float))):
            support.append(deepcopy(q))
    if len(support) > 5:
        raise ValueError('Ambiguous observed table has more than four immediate neighbours')
    return support


def _reciprocal_match(source, target, source_polygon, source_modules, target_modules, *,
                       table_context=False):
    """Require an independently tracked return to the selected observed module.

    Every observed source module is a reverse competitor. A good forward fit
    cannot hide a reverse failure, neighbour jump or inconsistent corner cycle.
    This is still image consistency, not independently verified physical identity.
    """
    from pvdelta.repeat_views import match_panel

    if type(table_context) is not bool:
        raise ValueError('table_context must be boolean')
    source_quads = [np.asarray(m['polygon_rgb_xy']) for m in source_modules]
    target_quads = [np.asarray(m['polygon_rgb_xy']) for m in target_modules]
    source_indices = [i for i, q in enumerate(source_quads)
        if any(np.array_equal(np.roll(order, shift, axis=0), source_polygon)
               for order in (q, q[::-1]) for shift in range(4))]
    if len(source_indices) != 1:
        raise ValueError('Selected observed source panel is absent or duplicated')
    # The native search prior cyclically aligns the selected RGB vertices.
    # Keep that exact origin/winding on return so corner-cycle indices compose.
    source_quads[source_indices[0]] = np.asarray(source_polygon)
    source_support = _table_support(source_polygon, source_modules) if table_context else None
    options = {'source_support_polygons': source_support} if table_context else {}
    forward = match_panel(Path(source), Path(target), np.asarray(source_polygon), target_quads, **options)
    result = forward.to_dict() | {'forward_result': forward.to_dict(),
        'reverse_result': None, 'reciprocal_check_passed': False,
        'tracking_support_mode': 'observed_table_neighbours' if table_context else 'selected_panel_only',
        'source_tracking_polygons': source_support, 'target_tracking_polygons': None}
    if forward.status != 'matched_provisional':
        return result
    if type(forward.target_index) is not int or not 0 <= forward.target_index < len(target_quads):
        raise ValueError('RGB match has no valid observed target module')
    target_support = _table_support(target_quads[forward.target_index].tolist(), target_modules) if table_context else None
    options = {'source_support_polygons': target_support} if table_context else {}
    reverse = match_panel(Path(target), Path(source), target_quads[forward.target_index], source_quads, **options)
    result['target_tracking_polygons'] = target_support
    result['reverse_result'] = reverse.to_dict()
    reason = None
    if reverse.status != 'matched_provisional':
        reason = 'Reverse RGB check failed: ' + reverse.reason
    elif type(reverse.target_index) is not int or reverse.target_index != source_indices[0]:
        reason = 'Reverse RGB check returned a different source panel'
    elif forward.target_corner_indices is not None and reverse.target_corner_indices is not None:
        a, b = forward.target_corner_indices, reverse.target_corner_indices
        if (sorted(a) != list(range(4)) or sorted(b) != list(range(4))
                or any(b[a[i]] != i for i in range(4))):
            reason = 'RGB corner cycle disagrees between forward and reverse checks'
    if reason:
        return result | {'status': 'unavailable', 'reason': reason, 'target_index': None,
                         'target_corner_indices': None}
    return result | {'reason': 'Reciprocal RGB geometry returned to the selected observed panel',
                     'reciprocal_check_passed': True}


def observed_rgb_repeat(database, finding_id, registration, *, sdk_library=None, max_views=8,
                         table_context=False):
    """Use a labelled coarse native anchor, then fit/check RGB geometry locally.

    This is distinct from fine cross-camera registration. Source association
    remains approximate; no target thermal region, temperature or glare decision
    is assigned, even when the local RGB point/overlap checks pass.
    """
    import cv2

    if type(table_context) is not bool:
        raise ValueError('table_context must be boolean')
    options = {'table_context': True} if table_context else {}
    rows, binding = _snapshot(database, finding_id, registration, max_views, sdk_library, **options)
    source = rows[0]
    _verify_source(source, registration, sdk_library)
    try:
        observed = deepcopy(_observed(source['rgb_path'], source['rgb_sha256'], source['gimbal_pitch_deg']))
    except cv2.error as exc:
        raise ValueError('Source RGB extraction failed: '+str(exc)) from exc
    matches = match_module_quads(source['diagnostic'].get('modules', []), observed['modules'], registration['model'])
    anchor = next((m for m in matches if m['module_id'] == source['module_id']), None)
    if anchor is None or not np.array_equal(np.asarray(anchor['native_polygon'], float),
                                            np.asarray(source['module_polygon'], float)):
        raise ValueError('Selected native observation has no unique bounded observed RGB anchor')
    source_polygon = anchor['rgb_polygon']
    flags = {'panel_identity_verified': False, 'cross_sensor_verified': False,
        'thermal_component_correspondence_verified': False, 'measurement_authorised': False,
        'glare_clearance': False, 'component_evidence': 'unknown'}
    output = []
    for row in rows[1:]:
        item = {k: row.get(k) for k in ('capture_id', 'thermal_path', 'rgb_path', 'rgb_sha256',
            'thermal_sha256', 'camera_distance_m', 'time_gap_s', 'gimbal_pitch_deg', 'gimbal_yaw_deg')}
        item.update(status='unavailable', reason='', rgb_polygon=None,
            forward_result=None, reverse_result=None, reciprocal_check_passed=False, **flags)
        try:
            if row['camera_model'] != 'M4T' or row['rgb_size'] != [4032, 3024]:
                raise ValueError('Target camera/RGB grid differs from the scoped flight')
            if (row['actual_rgb_sha256'] != row['rgb_sha256']
                    or row['actual_thermal_sha256'] != row['thermal_sha256']):
                raise ValueError('Target thermal/RGB file differs from its indexed identity')
            target = _observed(row['rgb_path'], row['rgb_sha256'], row['gimbal_pitch_deg'])
            match = _reciprocal_match(source['rgb_path'], row['rgb_path'], source_polygon,
                observed['modules'], target['modules'], **options)
            item.update(match, observed_target_modules=len(target['modules']))
            if item['status'] == 'matched_provisional':
                item['rgb_polygon'] = deepcopy(target['modules'][item['target_index']]['polygon_rgb_xy'])
        except (OSError, ValueError, RuntimeError, TypeError, KeyError, cv2.error) as exc:
            item.update(status='unavailable', reason=str(exc))
        output.append(item)
    if _snapshot(database, finding_id, registration, max_views, sdk_library, **options)[1] != binding:
        raise ValueError('Sources or analysis changed during RGB comparison')
    report = {'schema': 'pvdelta.observed-rgb-repeat/v1', 'binding': binding, 'finding_id': finding_id,
        'source': {k: source.get(k) for k in ('capture_id', 'thermal_path', 'rgb_path', 'rgb_sha256', 'thermal_sha256', 'module_id')},
        'source_rgb_polygon': source_polygon, 'source_anchor_iou': anchor['iou'],
        'source_anchor_status': 'approximate_native_association',
        'rows': output, 'database_written': False,
        'tracking_support_mode': 'observed_table_neighbours' if table_context else 'selected_panel_only', **flags}
    report['content_sha256'] = digest(report)
    return report


def repeat_pair_images(report, capture_id):
    """Show only original paired photos; cyan is observed RGB geometry, never a hotspot."""
    import io

    from PIL import Image, ImageDraw

    if (report.get('schema') != 'pvdelta.observed-rgb-repeat/v1'
            or report.get('content_sha256') != digest({k: v for k, v in report.items() if k != 'content_sha256'})):
        raise ValueError('RGB comparison report content changed')
    source = report['source']
    if capture_id == source['capture_id']:
        row, polygon = source, report['source_rgb_polygon']
    else:
        row = next((r for r in report['rows'] if r['capture_id'] == capture_id), None)
        if row is None:
            raise ValueError('Capture is outside this comparison')
        polygon = row['rgb_polygon'] if row['status'] == 'matched_provisional' else None
    images = []
    for kind in ('thermal', 'rgb'):
        encoded = Path(row[kind+'_path']).read_bytes()
        if hashlib.sha256(encoded).hexdigest() != row[kind+'_sha256']:
            raise ValueError('Comparison source changed; overlay withheld')
        with Image.open(io.BytesIO(encoded)) as image:
            images.append(image.convert('RGB'))
    if polygon is not None:
        q = np.asarray(polygon, float)
        if (q.shape != (4, 2) or not np.isfinite(q).all() or (q < 0).any()
                or (q[:, 0] >= images[1].width).any() or (q[:, 1] >= images[1].height).any()):
            raise ValueError('Observed RGB polygon lies outside the original image')
        draw = ImageDraw.Draw(images[1])
        draw.line([*map(tuple, q), tuple(q[0])], fill='#00ffff', width=5)
        low = np.maximum(0, np.floor(q.min(axis=0)-500)).astype(int)
        high = np.minimum(images[1].size, np.ceil(q.max(axis=0)+500)).astype(int)
        images[1] = images[1].crop((*low, *high))
    return images[0], images[1]
