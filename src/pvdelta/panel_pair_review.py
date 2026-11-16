"""Source-bound human decisions on experimental panel-pair proposals."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from pvdelta.identity_store import observation_binding, source_file_sha256
from pvdelta.panel_review import panel_groups
from pvdelta.store import connect_store


def load_pair_proposals(database, groups):
    pointer = Path(database).with_suffix('.panel-pairs.json')
    if not pointer.is_file() or not groups:
        return None
    config = json.loads(pointer.read_bytes())
    if (Path(config['database']).resolve() != Path(database).resolve()
            or config['run'] != groups[0]['analysis_run_id']):
        return None
    raw = Path(config['pairs']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config['pairs_sha256']:
        raise ValueError('Prepared pair proposals changed')
    pairs = json.loads(raw)
    rgb = {}
    if config.get('rgb_pairs'):
        rgb_raw = Path(config['rgb_pairs']).read_bytes()
        if hashlib.sha256(rgb_raw).hexdigest() != config['rgb_pairs_sha256']:
            raise ValueError('Prepared RGB pair evidence changed')
        rgb = {(r['source'], r['target']): r for r in json.loads(rgb_raw)}
    current = {g['observation_id']: g for g in groups}
    valid, stale = [], 0
    for pair in pairs:
        if pair.get('status') not in ('overlap_only_review_required', 'provisional_repeat_support'):
            continue
        ids = [pair['source'], pair['target']]
        if any(i not in current or config['bindings'].get(i) != observation_binding(current[i]) for i in ids):
            stale += 1
            continue
        first, second = [current[i] for i in ids]
        if first['capture_id'] == second['capture_id']:
            raise ValueError('Cross-frame proposal refers to a single source frame')
        binding = {'run': config['run'], 'observations': {
            i: observation_binding(current[i]) for i in sorted(ids)}}
        pair_id = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        rgb_evidence = rgb.get(tuple(ids), {})
        texture = rgb_evidence.get('original_rgb_texture')
        if texture is not None and (
                not isinstance(texture, dict) or texture.get('pair_id') != pair_id
                or type(texture.get('fit_points')) is not int or texture['fit_points'] < 4
                or type(texture.get('check_points')) is not int or texture['check_points'] < 2
                or type(texture.get('neighbour_controls')) is not int or texture['neighbour_controls'] < 4
                or type(texture.get('nominal_ray_span_deg')) not in (int, float)
                or not math.isfinite(texture['nominal_ray_span_deg'])
                or not 0 <= texture['nominal_ray_span_deg'] <= 180):
            raise ValueError('Original RGB texture evidence has an invalid pair binding or score')
        valid.append({**pair, 'pair_id': pair_id, 'binding': binding,
                      'first': first, 'second': second,
                      'rgb_status': rgb_evidence.get('rgb_status', 'not_checked'),
                      'original_rgb_texture': texture})
    if len({p['pair_id'] for p in valid}) != len(valid):
        raise ValueError('Duplicate or reversed pair proposals')
    return {'pairs': valid, 'stale': stale}


def _verify_sources(database, proposal):
    binding = proposal['binding']
    expected_id = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    if len(binding['observations']) != 2 or proposal['pair_id'] != expected_id:
        raise ValueError('A source-bound pair of distinct observations is required')
    connection = connect_store(database, read_only=True)
    try:
        current = {g['observation_id']: g for g in panel_groups(connection, proposal['binding']['run'])}
    finally:
        connection.close()
    frames = set()
    for oid, expected in proposal['binding']['observations'].items():
        group = current.get(oid)
        if group is None or observation_binding(group) != expected:
            raise ValueError('Panel source geometry changed; reload proposals')
        frames.add(group['capture_id'])
        row = group['regions'][0]
        for field in ('thermal', 'rgb'):
            actual = source_file_sha256(Path(row['source_root']) / row[field+'_path'])
            if actual != row[field+'_hash']:
                raise ValueError('Original source file changed; review rejected')
    if len(frames) != 2:
        raise ValueError('Pair review requires two different frames')
    return {oid: current[oid] for oid in binding['observations']}


def save_pair_decision(database, proposal, status, reviewer, note=''):
    if status not in ('same_panel', 'different_panels', 'wait'):
        raise ValueError('Explicit panel-pair review status required')
    if not isinstance(reviewer, str) or not 1 <= len(reviewer.strip()) <= 100:
        raise ValueError('Reviewer name is required (1–100 characters)')
    if not isinstance(note, str) or len(note) > 4000:
        raise ValueError('Review note must be at most 4000 characters')
    _verify_sources(database, proposal)
    document = {'schema': 'pvdelta.panel-pair-review/v1', 'pair_id': proposal['pair_id'],
                'binding': proposal['binding'], 'status': status, 'reviewer': reviewer.strip(),
                'note': note, 'identity_assignments_updated': False,
                'reviewed_at_utc': datetime.now(timezone.utc).isoformat()}
    content = json.dumps(document, sort_keys=True, ensure_ascii=False).encode('utf-8')
    digest = hashlib.sha256(content).hexdigest()
    folder = Path(database).with_suffix('.panel-pair-reviews')
    folder.mkdir(exist_ok=True)
    path = folder/(digest+'.json')
    try:
        with path.open('xb') as output:
            output.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError('Stored review content differs') from None
    return path


def pair_decisions(database, proposals):
    folder = Path(database).with_suffix('.panel-pair-reviews')
    current = {p['pair_id']: p['binding'] for p in proposals}
    decisions = {}
    for path in folder.glob('*.json'):
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != path.stem:
            raise ValueError('Saved panel review content changed')
        review = json.loads(content)
        pid = review['pair_id']
        if (review['binding'] == current.get(pid)
                and (pid not in decisions or review['reviewed_at_utc'] > decisions[pid]['reviewed_at_utc'])):
            decisions[pid] = review
    return decisions
