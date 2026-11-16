"""Explicit local inspection shortcuts; never discover or merge arbitrary databases."""
from __future__ import annotations

import json
from pathlib import Path


def load_inspection_library(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return []
    value = json.loads(path.read_bytes())
    if (not isinstance(value, dict) or value.get('schema') != 'pvdelta.inspection-library/v1'
            or not isinstance(value.get('inspections'), list)):
        raise ValueError('A pvdelta inspection library with an inspections list is required')
    result, seen = [], set()
    for item in value['inspections']:
        if (not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k].strip()
                for k in ('id', 'label', 'database_path', 'analysis_run_id'))):
            raise ValueError('Inspection ID, label, database path and exact analysis run are required')
        if item['id'] == 'current' or item['id'] in seen:
            raise ValueError('Distinct inspection IDs required; current is reserved')
        seen.add(item['id'])
        database = Path(item['database_path']).expanduser()
        if not database.is_absolute():
            database = path.parent/database
        result.append({k:item[k] for k in ('id', 'label', 'analysis_run_id')} |
                      {'database_path':str(database.resolve())})
    return result
