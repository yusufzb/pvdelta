"""Source-bound control review, with immutable decisions and unchanged fit/check roles."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def prepare_control_review(controls, sources, output, *, sdk_library=None):
    """Create unmarked review images from exact originals, without fitting any model."""
    from PIL import Image

    from pvdelta.decode import RJpeg
    from pvdelta.local_rgb_transfer import _context
    from pvdelta.models import ThermalFrame

    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError('Choose a new control-review directory')
    frames = controls.get('frames', [controls])
    if not isinstance(frames, list) or not frames or len(frames) != len(sources):
        raise ValueError('One explicit original source pair per control frame required')
    prepared, identifiers = [], set()
    for control, source in zip(frames, sources, strict=True):
        thermal, rgb = Path(source['thermal']).resolve(), Path(source['rgb']).resolve()
        with RJpeg(thermal, sdk_library) as image:
            values, display = image.temperature(), image.processed_rgb()
        frame = ThermalFrame(thermal.stem, str(thermal), hashlib.sha256(thermal.read_bytes()).hexdigest(),
            values, None, values.shape[1], values.shape[0])
        if (_context(frame, rgb) != control.get('pair')
                or control.get('schema') != 'pvdelta.local-rgb-controls/v1'
                or display.dtype != np.uint8 or display.shape != (*values.shape, 3)):
            raise ValueError('Controls must match the exact original sources and native grid')
        points = control.get('points')
        if not isinstance(points, list) or not points:
            raise ValueError('Explicit source controls required')
        for point in points:
            identifier = point.get('control_id')
            if (not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers
                    or point.get('role') not in ('fit', 'check')):
                raise ValueError('Unique control IDs and fixed fit/check roles required')
            identifiers.add(identifier)
            _point(point['native_xy'], values.shape)
            _point(point['rgb_xy'], control['pair']['rgb_shape'])
        prepared.append((control, thermal, rgb, display))
    destination.mkdir(parents=True)
    entries = []
    for i, (control, thermal, rgb, display) in enumerate(prepared):
        native_name, rgb_name = f'{i:03d}-native.png', f'{i:03d}-original-rgb.jpg'
        Image.fromarray(display).save(destination/native_name)
        (destination/rgb_name).write_bytes(rgb.read_bytes())
        if hashlib.sha256((destination/rgb_name).read_bytes()).hexdigest() != control['pair']['rgb_sha256']:
            raise ValueError('Original RGB source changed during review preparation')
        entries.append({'controls': control, 'source_names': [thermal.name, rgb.name],
            'native_preview': native_name, 'original_rgb': rgb_name,
            'image_sha256': {n: hashlib.sha256((destination/n).read_bytes()).hexdigest()
                             for n in (native_name, rgb_name)}})
    pack = {'schema': 'pvdelta.control-review-pack/v1', 'parent_controls': controls,
        'parent_controls_sha256': digest(controls), 'frames': entries,
        'centre_contract': 'Intersection of the same physical frame-band centres in both sensors. '
            'Do not select a screw, either edge of a band, a cell line, or a model-predicted position.',
        'model_predictions_included': False, 'independent_review_verified': False}
    pack['pack_id'] = digest(pack)
    (destination/'review-pack.json').write_bytes(canonical(pack))
    return pack


def _verify_pack(pack):
    if (pack.get('schema') != 'pvdelta.control-review-pack/v1'
            or pack.get('pack_id') != digest({k: v for k, v in pack.items() if k != 'pack_id'})
            or digest(pack['parent_controls']) != pack['parent_controls_sha256']
            or [e['controls'] for e in pack['frames']] != pack['parent_controls'].get('frames', [pack['parent_controls']])):
        raise ValueError('Control review pack identity mismatch')


def load_review_pack(path):
    path = Path(path).resolve()
    pack = json.loads(path.read_bytes())
    _verify_pack(pack)
    for entry in pack['frames']:
        for key in ('native_preview', 'original_rgb'):
            image = (path.parent/entry[key]).resolve()
            if (not image.is_relative_to(path.parent) or hashlib.sha256(image.read_bytes()).hexdigest()
                    != entry['image_sha256'][entry[key]]):
                raise ValueError('Review image identity mismatch')
    return pack


def _point(xy, shape):
    if (not isinstance(xy, (list, tuple)) or len(xy) != 2
            or any(type(v) not in (int, float) or not np.isfinite(v) for v in xy)
            or not 0 <= xy[0] < shape[1] or not 0 <= xy[1] < shape[0]):
        raise ValueError('Finite coordinates inside the original pixel grid required')
    return list(xy)


def review_points(pack):
    return [(entry, point) for entry in pack['frames'] for point in entry['controls']['points']]


def save_control_decision(pack, output, control_id, *, native_xy, rgb_xy, status, reviewer, note,
                          review_basis='user_declared_source_visual_review'):
    _verify_pack(pack)
    pairs = [(e, p) for e, p in review_points(pack) if p['control_id'] == control_id]
    if (len(pairs) != 1 or status not in ('OK', 'Not OK', 'Wait') or not reviewer.strip()
            or review_basis not in ('user_declared_source_visual_review', 'AI_source_visual_review')):
        raise ValueError('Known control ID, explicit status and reviewer required')
    entry, point = pairs[0]
    previous = latest_control_decisions(pack, output).get(control_id)
    decision = {'schema': 'pvdelta.control-review-decision/v1', 'pack_id': pack['pack_id'],
        'control_id': control_id, 'role': point['role'], 'status': status,
        'native_xy': _point(native_xy, entry['controls']['pair']['native_shape']),
        'rgb_xy': _point(rgb_xy, entry['controls']['pair']['rgb_shape']),
        'reviewer': reviewer.strip(), 'note': note.strip(),
        'review_sequence': previous.get('review_sequence', 0)+1 if previous else 1,
        'reviewed_at': datetime.now(timezone.utc).isoformat(),
        'review_basis': review_basis, 'independent_review_verified': False}
    decision['decision_id'] = digest(decision)
    directory = Path(output)/pack['pack_id']
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/f"{decision['decision_id']}.json"
    with target.open('xb') as stream:
        stream.write(canonical(decision))
    return decision


def latest_control_decisions(pack, output):
    _verify_pack(pack)
    decisions = {}
    known = {p['control_id']: (e, p) for e, p in review_points(pack)}
    for path in sorted((Path(output)/pack['pack_id']).glob('*.json')):
        decision = json.loads(path.read_bytes())
        if (decision.get('schema') != 'pvdelta.control-review-decision/v1'
                or decision.get('decision_id') != digest({k: v for k, v in decision.items() if k != 'decision_id'})
                or decision.get('pack_id') != pack['pack_id'] or decision.get('control_id') not in known):
            raise ValueError('Saved control decision identity mismatch')
        identifier = decision['control_id']
        sequence = decision.get('review_sequence', 0)
        if type(sequence) is not int or sequence < 0:
            raise ValueError('Invalid control review sequence')
        entry, point = known[identifier]
        if decision['role'] != point['role'] or decision['status'] not in ('OK', 'Not OK', 'Wait'):
            raise ValueError('Saved control role or status mismatch')
        _point(decision['native_xy'], entry['controls']['pair']['native_shape'])
        _point(decision['rgb_xy'], entry['controls']['pair']['rgb_shape'])
        if identifier not in decisions or (sequence, decision['reviewed_at'], decision['decision_id']) > (
                decisions[identifier].get('review_sequence', 0), decisions[identifier]['reviewed_at'],
                decisions[identifier]['decision_id']):
            decisions[identifier] = decision
    return decisions


def export_reviewed_controls(pack, output, *, frame_index=None):
    """Export all controls for the pack or one explicitly selected source pair."""
    _verify_pack(pack)
    if frame_index is not None and (type(frame_index) is not int
            or not 0 <= frame_index < len(pack['frames'])):
        raise ValueError('A valid explicit source photo pair index is required')
    selected = pack['frames'] if frame_index is None else [pack['frames'][frame_index]]
    decisions = latest_control_decisions(pack, output)
    if any(decisions.get(p['control_id'], {}).get('status') != 'OK'
           for entry in selected for p in entry['controls']['points']):
        raise ValueError('Every original fit/check control needs an explicit OK; unresolved points cannot be dropped')
    frames = []
    for entry in selected:
        frame = deepcopy(entry['controls'])
        frame['provenance'] = {'kind': 'declared_source_visual_revision',
            'note': 'All original roles and source identities preserved. Revised observations are '
                'development controls, not independent validation after inspecting previous diagnostics.',
            'original_provenance': frame['provenance'], 'review_pack_id': pack['pack_id'],
            'parent_controls_sha256': pack['parent_controls_sha256'],
            'independent_review_verified': False,
            'decisions': [decisions[p['control_id']] for p in frame['points']]}
        if any(d['review_basis'] == 'AI_source_visual_review' for d in frame['provenance']['decisions']):
            frame['provenance']['kind'] = 'AI_assisted_source_visual_revision'
        if frame_index is not None:
            frame['provenance']['source_pair_index'] = frame_index
        for point in frame['points']:
            decision = decisions[point['control_id']]
            point.update(native_xy=decision['native_xy'], rgb_xy=decision['rgb_xy'])
        frames.append(frame)
    bundle = {'frames': frames}
    destination = Path(output)/f'controls-{digest(bundle)}.json'
    if not destination.exists():
        with destination.open('xb') as stream:
            stream.write(canonical(bundle))
    elif destination.read_bytes() != canonical(bundle):
        raise ValueError('Existing exported control bundle was modified')
    return destination
