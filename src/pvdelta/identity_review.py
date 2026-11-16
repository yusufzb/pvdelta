"""Source-bound validation and local retention of operator table annotations."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from pvdelta.identity_store import observation_binding
from pvdelta.panel_review import panel_groups
from pvdelta.region_polygon import simple_polygon_mask
from pvdelta.store import connect_store


def workbench_package(database, run):
    """Resolve the optional, locally prepared package for this exact inspection."""
    database = Path(database).resolve()
    pointer = database.with_suffix('.identity-workbench.json')
    if not pointer.exists():
        return None
    config = json.loads(pointer.read_bytes())
    if Path(config['database']).resolve() != database or config['run'] != run:
        raise ValueError('Workbench belongs to a different database or run')
    pack_path = Path(config['pack'])
    pack = json.loads(pack_path.read_bytes())
    payload = {k: v for k, v in pack.items() if k != 'package_id'}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if (pack.get('schema') != 'pvdelta.table-identity-review-pack/v1'
            or pack.get('run') != run or pack.get('package_id') != expected):
        raise ValueError('Workbench manifest changed or belongs to a different run')
    return pack, Path(config['archive'])


def validate_decisions(database, pack, document):
    """Check source bytes and geometry; never change measurement or identity data."""
    if (not isinstance(document, dict)
            or document.get('schema') != 'pvdelta.table-identity-decisions/v1'
            or document.get('package_id') != pack['package_id']
            or document.get('run') != pack['run']):
        raise ValueError('Decision package differs from prepared sources')
    records = {r['observation_id']: r for r in pack['records']}
    decisions = document.get('decisions')
    if (not isinstance(decisions, list) or not decisions or len(decisions) > len(records)
            or any(not isinstance(d, dict) or not isinstance(d.get('observation_id'), str)
                   for d in decisions)):
        raise ValueError('A nonempty explicit decision list is required')
    if len({d['observation_id'] for d in decisions}) != len(decisions):
        raise ValueError('Duplicate observation decisions')
    connection = connect_store(database, read_only=True)
    try:
        current = {g['observation_id']: g for g in panel_groups(connection, pack['run'])}
    finally:
        connection.close()
    verified_sources = set()
    for decision in decisions:
        record = records.get(decision['observation_id'])
        group = current.get(decision['observation_id'])
        if (record is None or group is None or decision.get('binding') != record['binding']
                or record['binding'] != observation_binding(group)):
            raise ValueError('Source observation binding changed')
        row = group['regions'][0]
        for field, expected in [('rgb_sha256', row['rgb_hash']),
                                ('thermal_sha256', row['thermal_hash'])]:
            if decision.get(field) != expected:
                raise ValueError('Source file identity differs')
        for relative, expected in [(row['rgb_path'], row['rgb_hash']),
                                   (row['thermal_path'], row['thermal_hash'])]:
            path = Path(row['source_root']) / relative
            if (path, expected) not in verified_sources:
                with path.open('rb') as source:
                    actual = hashlib.file_digest(source, 'sha256').hexdigest()
                if actual != expected:
                    raise ValueError('Original source file changed')
                verified_sources.add((path, expected))
        status = decision.get('status')
        if status not in ('wait', 'confirmed_table', 'reject_link'):
            raise ValueError('Unknown review status')
        if not isinstance(decision.get('reviewer'), str) or not decision['reviewer'].strip():
            raise ValueError('Reviewer identity required')
        if status == 'confirmed_table':
            label = decision.get('table_label')
            if (not isinstance(label, str) or not 1 <= len(label.strip()) <= 100
                    or decision.get('source_layout_checked') is not True):
                raise ValueError('Confirmed table needs label and source-layout confirmation')
            width, height = record['rgb']['original_size']
            simple_polygon_mask(decision.get('polygon_rgb'), (height, width))
    return {'validated_records': len(decisions),
            'counts': dict(Counter(d['status'] for d in decisions)),
            'pending_source_cards': len(records) - len(decisions),
            'identity_database_updated': False}


def save_decisions(database, pack, document):
    """Keep an immutable content-addressed review after revalidation on save."""
    report = validate_decisions(database, pack, document)
    content = json.dumps(document, sort_keys=True, ensure_ascii=False, indent=2).encode('utf-8')
    digest = hashlib.sha256(content).hexdigest()
    folder = Path(database).with_suffix('.operator-reviews')
    folder.mkdir(exist_ok=True)
    path = folder / (digest + '.json')
    try:
        with path.open('xb') as output:
            output.write(content)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError('Existing review archive has different content') from None
    return path, report
