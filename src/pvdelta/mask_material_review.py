"""Source-bound material descriptions, separate from masks and fault decisions."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import uuid
from pathlib import Path

import numpy as np

CLASSIFICATIONS = {'glass', 'frame_or_nonpanel', 'unclear', 'unreviewed'}
_ASSETS = ('crop', 'locator', 'unmarked_native', 'rgb_context')
_SAVE_LOCK = threading.RLock()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def asset_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('Relative review asset path required')
    base = Path(root).resolve()
    target = (base / relative).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        raise ValueError('Review asset is missing or outside its pack directory')
    return target


def load_material_pack(path):
    source = Path(path).expanduser().resolve()
    pack = json.loads(source.read_bytes())
    if not isinstance(pack, dict) or pack.get('schema') != 'pvdelta.mask-material-review-pack/v1':
        raise ValueError('Native material review pack required')
    identity = dict(pack)
    pack_id = identity.pop('pack_id', None)
    if pack_id != 'sha256:' + hashlib.sha256(_canonical(identity)).hexdigest():
        raise ValueError('Review pack integrity mismatch')
    if not isinstance(pack.get('items'), list) or not pack['items']:
        raise ValueError('Nonempty review items required')
    seen = set()
    for item in pack['items']:
        if not isinstance(item, dict):
            raise TypeError('Review item must be an object')
        identifier = item.get('item_id')
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError('Unique review item IDs required')
        seen.add(identifier)
        if not all(_digest(item.get(k)) for k in (
            'source_sha256', 'native_temperature_sha256', 'target_pixels_sha256',
        )):
            raise ValueError('Source, native-grid and pixel hashes required')
        shape = item.get('native_shape')
        coords = item.get('target_pixels_yx')
        if (not isinstance(shape, list) or len(shape) != 2
                or any(type(v) is not int or not 0 < v < 2**31 for v in shape)
                or not isinstance(coords, list) or not coords):
            raise ValueError('Native grid and target pixel coordinates required')
        for point in coords:
            if (not isinstance(point, list) or len(point) != 2
                    or any(type(v) is not int for v in point)
                    or not (0 <= point[0] < shape[0] and 0 <= point[1] < shape[1])):
                raise ValueError('Target pixel lies outside its native grid')
        if len({tuple(p) for p in coords}) != len(coords):
            raise ValueError('Duplicate native target pixels')
        if type(item.get('target_pixel_count')) is not int or item['target_pixel_count'] != len(coords):
            raise ValueError('Target pixel count mismatch')
        if hashlib.sha256(np.asarray(coords, dtype='<i4').tobytes()).hexdigest() != item['target_pixels_sha256']:
            raise ValueError('Native target pixel hash mismatch')
        hashes = item.get('asset_sha256', {})
        expected = {key for key in _ASSETS if item.get(key) is not None}
        if (not isinstance(hashes, dict)
                or not {'crop', 'locator', 'unmarked_native'} <= expected or set(hashes) != expected):
            raise ValueError('Complete native review assets and their hashes required')
        for key in expected:
            if not _digest(hashes[key]) or _sha(asset_path(source.parent, item[key])) != hashes[key]:
                raise ValueError('Review image changed after pack preparation')
        if not isinstance(item.get('source_name'), str) or not item['source_name']:
            raise ValueError('Source frame filename required')
        temperatures = item.get('temperature_c', {})
        if not isinstance(temperatures, dict) or any(
            type(temperatures.get(k)) not in (int, float) or not math.isfinite(temperatures[k])
            for k in ('min', 'median', 'max')
        ):
            raise ValueError('Finite native target temperature statistics required')
        if not temperatures['min'] <= temperatures['median'] <= temperatures['max']:
            raise ValueError('Native temperature statistics are inconsistent')
    return pack, source.parent


def new_material_work(pack, reviewer=''):
    return {
        'schema': 'pvdelta.mask-material-review-work/v1', 'pack_id': pack['pack_id'],
        'reviewer': reviewer,
        'items': [{k: item[k] for k in (
            'item_id', 'source_sha256', 'target_pixels_sha256',
        )} | {'classification': 'unreviewed', 'note': ''} for item in pack['items']],
        'scope': 'Declared material review only; no original label override or model/field approval',
    }


def validate_material_work(pack, content):
    identity = dict(pack)
    pack_id = identity.pop('pack_id', None)
    if pack_id != 'sha256:' + hashlib.sha256(_canonical(identity)).hexdigest():
        raise ValueError('Review pack integrity mismatch')
    work = json.loads(content)
    if (not isinstance(work, dict) or work.get('schema') != 'pvdelta.mask-material-review-work/v1'
            or work.get('pack_id') != pack['pack_id']):
        raise ValueError('Wrong material review schema or source pack')
    reviewer = work.get('reviewer')
    if not isinstance(reviewer, str) or len(reviewer) > 200:
        raise ValueError('Actual reviewer must be a short string')
    reviewer = reviewer.strip()
    supplied = work.get('items')
    if not isinstance(supplied, list) or len(supplied) != len(pack['items']):
        raise ValueError('All items must be retained, including unreviewed items')
    by_id = {}
    for item in supplied:
        if (not isinstance(item, dict) or not isinstance(item.get('item_id'), str)
                or item['item_id'] in by_id):
            raise ValueError('Unique valid review item IDs required')
        by_id[item['item_id']] = item
    if set(by_id) != {r['item_id'] for r in pack['items']}:
        raise ValueError('Unexpected or missing review item')
    checked = []
    for expected in pack['items']:
        item = by_id[expected['item_id']]
        if any(item.get(k) != expected[k] for k in ('source_sha256', 'target_pixels_sha256')):
            raise ValueError('Review is bound to a different source or native pixel set')
        value, note = item.get('classification'), item.get('note')
        if (not isinstance(value, str) or value not in CLASSIFICATIONS
                or not isinstance(note, str) or len(note) > 2000):
            raise ValueError('Invalid material classification or note')
        if value != 'unreviewed' and not reviewer:
            raise ValueError('Actual reviewer required for decisions')
        checked.append({
            'item_id': item['item_id'], 'source_sha256': expected['source_sha256'],
            'native_temperature_sha256': expected['native_temperature_sha256'],
            'target_pixels_sha256': expected['target_pixels_sha256'],
            'classification': value, 'note': note,
        })
    counts = {key: sum(r['classification'] == key for r in checked) for key in sorted(CLASSIFICATIONS)}
    raw = content.encode('utf-8') if isinstance(content, str) else content
    return {
        'schema': 'pvdelta.validated-mask-material-review/v1', 'pack_id': pack['pack_id'],
        'reviewer': reviewer, 'items': checked, 'counts': counts,
        'source_export_sha256': hashlib.sha256(raw).hexdigest(),
        'all_answered': counts['unreviewed'] == 0,
        'all_material_resolved': counts['unreviewed'] == counts['unclear'] == 0,
        'original_coarse_labels_changed': False, 'model_acceptance_changed': False,
        'database_changed': False, 'electrical_fault_or_glare_approval': False,
        'independent_reviewer_identity_verified': False,
        'authority': 'User-declared material descriptions only; not independent field or model acceptance.',
    }


def _stored_versions(pack, root):
    versions = []
    folder = Path(root) / 'operator-reviews'
    if folder.exists() and not folder.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Review output directory lies outside its pack')
    for path in folder.glob('*.json'):
        if re.fullmatch(r'[0-9a-f]{64}\.json', path.name) is None:
            continue
        stored = json.loads(path.read_bytes())
        if not isinstance(stored, dict):
            raise TypeError('Stored review must be an object')
        if stored.get('pack_id') != pack['pack_id']:
            continue
        digest = stored.get('source_export_sha256')
        if not _digest(digest) or path.stem != digest:
            raise ValueError('Stored review version identity mismatch')
        original = path.with_name(digest + '.source.json').read_bytes()
        checked = validate_material_work(pack, original)
        if checked != stored:
            raise ValueError('Stored review integrity mismatch')
        versions.append((path.stat().st_mtime_ns, path.name, json.loads(original), checked))
    return versions


def _head_path(pack, root):
    digest = pack['pack_id'].removeprefix('sha256:')
    if not _digest(digest):
        raise ValueError('Invalid review pack identity')
    return Path(root) / 'operator-reviews' / ('current-' + digest + '.json')


def _current_head(pack, root, versions):
    path = _head_path(pack, root)
    if not path.exists():
        # One-time migration of legacy versions that predate explicit commit order.
        reviewers = {}
        latest = None
        for row in sorted(versions, key=lambda row: (row[0], row[1])):
            latest = row[3]['source_export_sha256']
            reviewers[row[3]['reviewer']] = latest
        return {'schema': 'pvdelta.mask-material-review-head/v1', 'pack_id': pack['pack_id'],
                'revision': 0, 'latest': latest, 'reviewers': reviewers}
    head = json.loads(path.read_bytes())
    if (not isinstance(head, dict)
            or head.get('schema') != 'pvdelta.mask-material-review-head/v1'
            or head.get('pack_id') != pack['pack_id']
            or type(head.get('revision')) is not int or head['revision'] < 1
            or not isinstance(head.get('reviewers'), dict)):
        raise ValueError('Invalid review commit head')
    by_digest = {row[3]['source_export_sha256']: row[3] for row in versions}
    for reviewer, digest in head['reviewers'].items():
        if (not _digest(digest) or digest not in by_digest
                or by_digest[digest]['reviewer'] != reviewer):
            raise ValueError('Review commit head references an invalid version or reviewer')
    latest = head.get('latest')
    if (not _digest(latest) or latest not in by_digest
            or head['reviewers'].get(by_digest[latest]['reviewer']) != latest):
        raise ValueError('Review commit head references an invalid latest version')
    return head


def latest_material_work(pack, root, *, reviewer=None):
    with _SAVE_LOCK:
        versions = _stored_versions(pack, root)
        head = _current_head(pack, root, versions)
        digest = head['latest'] if reviewer is None else head['reviewers'].get(reviewer)
        for row in versions:
            if row[3]['source_export_sha256'] == digest:
                return row[2], row[3]
        work = new_material_work(pack, reviewer or '')
        return work, validate_material_work(pack, _canonical(work))


def save_material_decision(
    pack_path, item_id, classification, reviewer, *, note='', expected_pack_id=None,
):
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError('Actual reviewer required')
    pack, root = load_material_pack(pack_path)
    if expected_pack_id is not None and pack['pack_id'] != expected_pack_id:
        raise ValueError('Review pack changed; reopen it before saving')
    with _SAVE_LOCK:
        head = _current_head(pack, root, _stored_versions(pack, root))
        work, _ = latest_material_work(pack, root, reviewer=reviewer.strip())
        item = next((r for r in work['items'] if r['item_id'] == item_id), None)
        if item is None:
            raise ValueError('Unknown review item')
        item.update(classification=classification, note=note)
        work['reviewer'] = reviewer.strip()
        content = _canonical(work)
        checked = validate_material_work(pack, content)
        folder = root / 'operator-reviews'
        folder.mkdir(exist_ok=True)
        if not folder.resolve().is_relative_to(root):
            raise ValueError('Review output directory lies outside its pack')
        digest = checked['source_export_sha256']
        destination = folder / (digest + '.json')
        for path, data in (
            (folder / (digest + '.source.json'), content),
            (destination, json.dumps(checked, indent=2, allow_nan=False).encode('utf-8')),
        ):
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError('Existing immutable review version differs')
            else:
                with path.open('xb') as stream:
                    stream.write(data)
        # Commit order is explicit: filesystem timestamps may tie or run backwards.
        head['revision'] += 1
        head['latest'] = digest
        head['reviewers'][reviewer.strip()] = digest
        head_path = _head_path(pack, root)
        temporary = folder / ('.head-' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('xb') as stream:
                stream.write(_canonical(head))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, head_path)
        finally:
            temporary.unlink(missing_ok=True)
        return destination, work, checked
