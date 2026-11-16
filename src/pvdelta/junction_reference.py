"""Source-only corner references; no detector proposals or residual-selected roles."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pvdelta.local_rgb_transfer import _context


def _bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')


def slots():
    return [{'control_id': f'J{i+1:02}', 'role': 'check' if i % 3 == 1 else 'fit'} for i in range(12)]


def prepare_junction_references(pairs, destination):
    """Prepare new source-bound references from already decoded native frames."""
    if not 1 <= len(pairs) <= 30:
        raise ValueError('One to thirty explicit source pairs required')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Choose a new reference directory')
    records, images, assets = [], [], {}
    seen = set()
    for index, (frame, rgb_path) in enumerate(pairs):
        context = _context(frame, rgb_path)
        if context['source_sha256'] in seen:
            raise ValueError('Source thermal pairs must be distinct')
        seen.add(context['source_sha256'])
        low, high = np.percentile(frame.temperature_c, [2, 98])
        if high <= low:
            raise ValueError('Source has no structural contrast')
        gray = np.uint8(np.rint(np.clip((frame.temperature_c-low)/(high-low), 0, 1)*255))
        thermal = io.BytesIO()
        Image.fromarray(gray).save(thermal, format='PNG')
        native_bytes = thermal.getvalue()
        rgb_bytes = Path(rgb_path).read_bytes()
        if hashlib.sha256(rgb_bytes).hexdigest() != context['rgb_sha256']:
            raise ValueError('RGB source changed while preparing references')
        with Image.open(io.BytesIO(rgb_bytes)) as image:
            # Preserve encoded pixel orientation, including originals with EXIF rotation.
            if image.format == 'JPEG' and image.getexif().get(274, 1) == 1:
                rgb_preview, rgb_mime = rgb_bytes, 'image/jpeg'
            else:
                rgb = io.BytesIO()
                image.convert('RGB').save(rgb, format='PNG')
                rgb_preview, rgb_mime = rgb.getvalue(), 'image/png'
        thermal_asset = f'assets/P{index+1:02}-thermal.png'
        rgb_asset = f'assets/P{index+1:02}-rgb.'+('jpg' if rgb_mime == 'image/jpeg' else 'png')
        assets[thermal_asset], assets[rgb_asset] = native_bytes, rgb_preview
        records.append({'pair_id': f'P{index+1:02}', 'pair': context,
            'thermal_file': Path(frame.source_path).name, 'rgb_file': Path(rgb_path).name,
            'thermal_image': thermal_asset, 'rgb_image': rgb_asset,
            'thermal_preview_sha256': hashlib.sha256(native_bytes).hexdigest(),
            'rgb_preview_sha256': hashlib.sha256(rgb_preview).hexdigest(), 'slots': slots()})
        images.append({'thermal': thermal_asset, 'rgb': rgb_asset})
        if _context(frame, rgb_path) != context:
            raise ValueError('Source changed during reference preparation')
    pack = {'schema': 'pvdelta.junction-reference-pack/v1', 'frames': records,
        'scope': 'Source-only point collection. Fixed12 slots/8fit/4check per pair. No residuals or predictions shown. Not independent field or fault acceptance.'}
    pack['pack_id'] = hashlib.sha256(_bytes(pack)).hexdigest()
    template = Path(__file__).with_name('junction_reference_editor.html').read_text(encoding='utf-8')
    page = template.replace('__PACK_JSON__', _bytes(pack).decode().replace('<', chr(92)+'u003c'))
    page = page.replace('__IMAGES_JSON__', _bytes(images).decode())
    destination.mkdir(parents=True)
    (destination/'assets').mkdir()
    for name, content in assets.items():
        (destination/name).write_bytes(content)
    (destination/'pack.json').write_bytes(_bytes(pack))
    (destination/'review.html').write_text(page, encoding='utf-8')
    return pack


def validate_junction_references(pack, document):
    if (not isinstance(pack, dict) or pack.get('schema') != 'pvdelta.junction-reference-pack/v1'
            or not isinstance(pack.get('frames'), list) or not 1 <= len(pack['frames']) <= 30
            or any(not isinstance(f, dict) or f.get('slots') != slots() for f in pack['frames'])):
        raise ValueError('Canonical source pack with fixed twelve-slot roles required')
    expected = hashlib.sha256(_bytes({k:v for k,v in pack.items() if k != 'pack_id'})).hexdigest()
    if pack.get('pack_id') != expected:
        raise ValueError('Reference pack changed')
    if (not isinstance(document, dict) or document.get('schema') != 'pvdelta.junction-reference-work/v1'
            or document.get('pack_id') != expected):
        raise ValueError('Wrong reference source pack')
    if (not isinstance(document.get('reviewer'), str) or not 1 <= len(document['reviewer'].strip()) <= 200):
        raise ValueError('Reviewer required')
    frames = document.get('frames')
    if not isinstance(frames, list) or len(frames) != len(pack['frames']):
        raise ValueError('Every source pair must be retained')
    results = []
    for reference, frame in zip(pack['frames'], frames, strict=True):
        if (not isinstance(frame, dict) or frame.get('pair_id') != reference['pair_id']
                or frame.get('pair') != reference['pair']):
            raise ValueError('Source pair identity changed')
        points = frame.get('points')
        if not isinstance(points, list) or len(points) != 12:
            raise ValueError('All twelve fixed slots required')
        pending = []
        native, rgb = [], []
        for slot, point in zip(reference['slots'], points, strict=True):
            if (not isinstance(point, dict) or any(point.get(k) != slot[k] for k in ('control_id', 'role'))
                    or point.get('status') not in ('observed', 'unclear', 'unreviewed')):
                raise ValueError('Fixed source slots, roles and review states required')
            if point['status'] != 'observed':
                pending.append(slot['control_id'])
            for field, shape, collected in [('native_xy', reference['pair']['native_shape'], native),
                                             ('rgb_xy', reference['pair']['rgb_shape'], rgb)]:
                value = point.get(field)
                if value is None and point['status'] != 'observed':
                    continue
                if (not isinstance(value, list) or len(value) != 2
                        or any(type(v) not in (int, float) or not np.isfinite(v) for v in value)
                        or not 0 <= value[0] < shape[1] or not 0 <= value[1] < shape[0]):
                    raise ValueError('Source coordinates must be inside original pixel grids')
                collected.append(value)
        for coordinates in (native, rgb):
            q = np.array(coordinates, float).reshape(-1, 2)
            if len(q) > 1:
                distance = np.linalg.norm(q[:, None]-q[None], axis=2)
                np.fill_diagonal(distance, np.inf)
                if (distance < 1).any():
                    raise ValueError('Each slot must identify a distinct source junction')
        results.append({'pair_id': reference['pair_id'], 'pending_slots': pending,
                        'ready_for_numeric_check': not pending})
    return {'pairs': results, 'all_references_complete': all(not r['pending_slots'] for r in results),
            'registration_verified': False, 'measurement_authorised': False, 'database_updated': False}


def check_junction_references(pairs, pack, document):
    """Check a submitted source reference without changing masks, IDs or live data."""
    from pvdelta.local_rgb_transfer import fit_local_rgb_registration

    review = validate_junction_references(pack, document)
    if len(pairs) != len(pack['frames']):
        raise ValueError('Actual sources required for every reference pair')
    outcomes = []
    for actual, reference, work, state in zip(pairs, pack['frames'], document['frames'], review['pairs'], strict=True):
        frame, rgb_path = actual
        if _context(frame, rgb_path) != reference['pair']:
            raise ValueError('Actual source files or native grid differ from reference pack')
        result = {'pair_id': reference['pair_id'], 'status': 'source_reference_pending',
                  'registration': None, 'pending_slots': state['pending_slots']}
        if state['ready_for_numeric_check']:
            controls = {'schema': 'pvdelta.local-rgb-controls/v1', 'pair': reference['pair'],
                'provenance': {'kind': 'declared_source_only_reference',
                    'note': 'Fixed roles before numeric assessment; declared reviewer: '+document['reviewer']},
                'points': [{k:p[k] for k in ('control_id', 'role', 'native_xy', 'rgb_xy')}
                           for p in work['points']]}
            try:
                registration = fit_local_rgb_registration(frame, rgb_path, controls)
                result.update(status='point_gate_passed_provisional' if registration['geometry_gate_passed']
                              else 'point_gate_failed', registration=registration)
            except ValueError as error:
                result.update(status='point_geometry_invalid', reason=str(error))
        if _context(frame, rgb_path) != reference['pair']:
            raise ValueError('Sources changed during reference checking')
        outcomes.append(result)
    return {'pairs': outcomes, 'all_references_complete': review['all_references_complete'],
            'physical_correspondence_verified': False, 'independent_review_verified': False,
            'measurement_authorised': False, 'database_updated': False}
