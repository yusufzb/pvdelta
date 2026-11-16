"""Own-frame radiometry for provisional card pairs; no transferred hotspot labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pvdelta.candidate_links import candidate_links_signature
from pvdelta.decode import RJpeg
from pvdelta.measurement_profile import apply_saved_measurement_profile, read_measurement_profile
from pvdelta.reference_evidence import check_temperature_evidence
from pvdelta.rgb_pair_check import _aligned_corners, digest
from pvdelta.rgb_preview import quadratic_points
from pvdelta.store import connect_store
from pvdelta.temperature_evidence_store import _runtime


def aligned_native_quads(source_native, target_native, source_rgb, target_rgb, indices, model):
    """Follow tracked RGB corners, including return flights, through approximate anchors."""
    if (not isinstance(indices, (list, tuple)) or len(indices) != 4
            or any(type(i) is not int for i in indices) or sorted(indices) != list(range(4))):
        raise ValueError('Tracked RGB corner correspondence unavailable')
    quads = [np.asarray(q, np.float64) for q in (source_native, target_native, source_rgb, target_rgb)]
    if any(q.shape != (4, 2) or not np.isfinite(q).all()
           or not cv2.isContourConvex(q.astype(np.float32)) for q in quads):
        raise ValueError('Convex finite native and RGB panel quads required')
    a, b, ar, br = quads
    expected_source = _aligned_corners(quadratic_points(a, model), ar)
    if not np.array_equal(expected_source, ar):
        raise ValueError('Source RGB corner order differs from its native anchor')
    expected_target = _aligned_corners(quadratic_points(b, model), br)
    # The tracker indexes the raw observed target RGB quad; the anchor orders it
    # in the native quad's order. Require exact vertex identity, not proximity.
    order = []
    for point in br[list(indices)]:
        hits = np.flatnonzero(np.all(expected_target == point, axis=1))
        if len(hits) != 1:
            raise ValueError('Target corner identity is ambiguous')
        order.append(int(hits[0]))
    ordered = b[order]
    if not cv2.isContourConvex(ordered.astype(np.float32)):
        raise ValueError('Tracked target corners cross')
    return a, ordered


def normalised_region(points_yx, quad):
    """Transform exact saved pixel centres; no hull, enlarged box or temperature warp."""
    points = np.asarray(points_yx)
    if (points.ndim != 2 or points.shape[1] != 2 or not len(points)
            or not np.issubdtype(points.dtype, np.integer)
            or len(np.unique(points, axis=0)) != len(points)):
        raise ValueError('Distinct integer candidate pixels required')
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(np.asarray(quad, np.float32), unit)
    uv = cv2.perspectiveTransform(points[:, ::-1, None].transpose(0, 2, 1).astype(np.float32), matrix)[:, 0]
    if not np.isfinite(uv).all() or (uv < 0).any() or (uv > 1).any():
        raise ValueError('Saved candidate pixel centres fall outside the panel')
    return {'pixel_count': len(points), 'centre_uv': uv.mean(axis=0).astype(float).tolist(),
        'bounds_uv': [*uv.min(axis=0).astype(float).tolist(), *uv.max(axis=0).astype(float).tolist()]}


def _pair_context(database, run_id, report, target_capture, registration, sdk_library):
    if (report.get('schema') != 'pvdelta.candidate-card-links/v1'
            or report.get('content_sha256') != digest({k:v for k,v in report.items() if k!='content_sha256'})
            or report.get('physical_panels_merged') is not False):
        raise ValueError('Current unmerged candidate-link report required')
    choices = [r for r in report['rows'] if r['capture_id'] == target_capture]
    if (len(choices) != 1 or choices[0]['status'] != 'provisional_candidate_link'
            or choices[0]['bidirectional_rgb_passed'] is not True):
        raise ValueError('A bidirectionally linked candidate target is required')
    target = choices[0]
    ids = [report['source_observation_id'], target['target_observation_id']]
    if ids[0] == ids[1]:
        raise ValueError('Two distinct observations required')
    connection = connect_store(database, read_only=True)
    data = []
    try:
        connection.execute('BEGIN TRANSACTION')
        for oid, capture in zip(ids, (report['source'], target), strict=True):
            row = connection.execute('''SELECT capture_id,pixel_polygon_json FROM module_observations
                WHERE analysis_run_id=? AND observation_id=?''', [run_id, oid]).fetchone()
            if row is None or row[0] != capture['capture_id']:
                raise ValueError('Observation differs from selected run/capture')
            path = Path(capture['thermal_path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != capture['thermal_sha256']:
                raise ValueError('Original thermal source changed')
            profile = read_measurement_profile(connection, run_id, row[0])
            if profile is None or profile['source_sha256'] != capture['thermal_sha256']:
                raise ValueError('Matching own-frame measurement profile required')
            result = connection.execute('''SELECT finding_id,region_pixels_yx_json,absolute_temperature_c,
                delta_t_k,n_reference_px FROM findings WHERE analysis_run_id=? AND observation_id=?
                ORDER BY finding_id''', [run_id, oid]).fetchall()
            expected = target['target_finding_ids'] if oid == ids[1] else None
            if not result or (expected is not None and [r[0] for r in result] != expected):
                raise ValueError('Complete saved candidate membership differs')
            data.append({'observation_id': oid, 'capture_id': row[0], 'path': str(path),
                'polygon': json.loads(row[1]), 'profile': profile, 'findings': result})
        if report['finding_id'] not in [r[0] for r in data[0]['findings']]:
            raise ValueError('Selected finding differs from source observation')
        connection.execute('COMMIT')
    finally:
        connection.close()
    quads = aligned_native_quads(data[0]['polygon'], data[1]['polygon'], report['source_rgb_polygon'],
        target['rgb_polygon'], target['forward'].get('target_corner_indices'), registration['model'])
    binding = digest({'run': run_id, 'data': data, 'quads': [q.tolist() for q in quads],
        'link_report': report['content_sha256'], 'registration': registration, 'runtime': _runtime(sdk_library),
        'engine': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    return data, quads, binding


def thermal_pair_signature(database, run_id, report, target_capture, registration, *, sdk_library=None):
    return _pair_context(database, run_id, report, target_capture, registration, sdk_library)[2]


def _native_image(values, quad, findings, low, high):
    """Common numeric colour scale on native pixels; outline only saved components."""
    gray = np.clip((values-low)/max(high-low, 1e-6)*255, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).resize((values.shape[1]*2, values.shape[0]*2), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    draw.line([tuple(p*2) for p in np.vstack((quad, quad[0]))], fill='cyan', width=2)
    for i, record in enumerate(findings, 1):
        points = np.asarray(json.loads(record[1]))
        mask = np.zeros(values.shape, np.uint8)
        mask[points[:, 0], points[:, 1]] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            xy = contour[:, 0]*2
            if len(xy) > 1:
                draw.line([tuple(p) for p in np.vstack((xy, xy[0]))], fill='lime', width=2)
            else:
                draw.point(tuple(xy[0]), fill='lime')
        y, x = points.mean(axis=0)*2
        draw.text((x+5, y-12), f'A{i}', fill='white', stroke_width=1, stroke_fill='black')
    # Entire panel and 50 native pixels context; images remain in their own camera orientation.
    x0, y0 = np.maximum(np.floor(quad.min(axis=0)-50), 0).astype(int)*2
    x1, y1 = np.minimum(np.ceil(quad.max(axis=0)+50), [values.shape[1], values.shape[0]]).astype(int)*2
    return image.crop((x0, y0, x1, y1))


def compare_linked_thermal(database, run_id, report, target_capture, registration, *, sdk_library=None):
    """Reproduce both full candidate populations and expose conditional positions, never merge."""
    if candidate_links_signature(database, report['finding_id'], registration, sdk_library=sdk_library,
            max_targets=report['max_targets']) != report['binding']:
        raise ValueError('Related candidate report is stale')
    data, quads, binding = _pair_context(database, run_id, report, target_capture, registration, sdk_library)
    temperatures, cards = [], []
    for item, quad in zip(data, quads, strict=True):
        replay = check_temperature_evidence(database, run_id, item['observation_id'], sdk_library=sdk_library)
        with RJpeg(item['path'], sdk_library) as reader:
            values = apply_saved_measurement_profile(reader, item['path'], item['profile'])
        if not np.isfinite(values).all():
            raise ValueError('Finite native SDK temperatures required')
        temperatures.append(values)
        regions = []
        for i, (fid, pixels, temperature, delta, count) in enumerate(item['findings'], 1):
            points = np.asarray(json.loads(pixels))
            if ((points < 0).any() or (points[:, 0] >= values.shape[0]).any()
                    or (points[:, 1] >= values.shape[1]).any()):
                raise ValueError('Candidate pixels lie outside native grid')
            regions.append({'label': f'A{i}', 'finding_id': fid,
                **normalised_region(points, quad), 'median_c': temperature, 'delta_t_k': delta,
                'reference_pixels': count})
        cards.append({'observation_id': item['observation_id'], 'source_file': Path(item['path']).name,
            'profile_id': item['profile']['profile_id'], 'native_sha256': replay['native_temperature_sha256'],
            'regions': regions})
    pairs = [{'source_region': a['label'], 'target_region': b['label'],
        'source_finding_id': a['finding_id'], 'target_finding_id': b['finding_id'],
        'centre_distance_uv': float(np.linalg.norm(np.asarray(a['centre_uv'])-b['centre_uv']))}
        for a in cards[0]['regions'] for b in cards[1]['regions']]
    # A common colour scale uses panel footprints only; never vegetation/sky statistics.
    populations = []
    for values, quad in zip(temperatures, quads, strict=True):
        mask = np.zeros(values.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 1)
        populations.append(values[mask.astype(bool)])
    low = float(min(v.min() for v in populations))
    high = float(max(v.max() for v in populations))
    images = [_native_image(v, q, item['findings'], low, high)
        for v, q, item in zip(temperatures, quads, data, strict=True)]
    if (_pair_context(database, run_id, report, target_capture, registration, sdk_library)[2] != binding
            or candidate_links_signature(database, report['finding_id'], registration, sdk_library=sdk_library,
                max_targets=report['max_targets']) != report['binding']):
        raise ValueError('Sources or evidence changed during thermal comparison')
    result = {'schema': 'pvdelta.linked-thermal/v1', 'binding': binding,
        'link_report_sha256': report['content_sha256'], 'cards': cards, 'all_region_pairs': pairs,
        'display_scale_c': [low, high], 'display': 'own native grids, nearest-neighbour display only',
        'spatial_status': 'conditional_on_approximate_native_rgb_anchors',
        'component_correspondence_verified': False, 'panel_identity_verified': False,
        'glare_clearance': False, 'physical_panels_merged': False, 'database_written': False}
    result['content_sha256'] = digest(result)
    return result, images
