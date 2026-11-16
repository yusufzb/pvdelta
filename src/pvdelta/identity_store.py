"""Local panel/table identities and source-bound observation assignments."""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
from pathlib import Path

import duckdb

_LOCK = threading.RLock()


def identity_path(database):
    return Path(database).with_suffix('.identities.duckdb')


def observation_binding(group):
    row = group['regions'][0]
    fields = {k: row[k] for k in ('capture_id', 'module_polygon', 'thermal_hash', 'rgb_hash')}
    fields['run'] = group['analysis_run_id']
    fields['observation'] = group['observation_id']
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def _validate_current(database, group):
    from pvdelta.panel_review import panel_groups
    from pvdelta.store import connect_store

    connection=connect_store(database,read_only=True)
    try:
        groups=panel_groups(connection,group['analysis_run_id'])
        current=next((g for g in groups
                      if g['observation_id']==group['observation_id']),None)
    finally:
        connection.close()
    if current is None or observation_binding(current)!=observation_binding(group):
        raise ValueError('Observation changed or belongs to a different inspection')
    return {g['observation_id']:g for g in groups}


def _open(database):
    connection = duckdb.connect(str(identity_path(database)))
    connection.execute("""
        CREATE TABLE IF NOT EXISTS identity_metadata(key VARCHAR PRIMARY KEY,value VARCHAR);
        CREATE TABLE IF NOT EXISTS entities(
            uid VARCHAR PRIMARY KEY,kind VARCHAR,label VARCHAR,parent_uid VARCHAR,
            provenance VARCHAR, UNIQUE(kind,label,parent_uid));
        CREATE TABLE IF NOT EXISTS assignments(
            run VARCHAR,observation VARCHAR,binding VARCHAR,table_uid VARCHAR,panel_uid VARCHAR,
            basis VARCHAR,PRIMARY KEY(run,observation));
    """)
    connection.execute("INSERT INTO identity_metadata VALUES ('namespace',?) ON CONFLICT DO NOTHING",
                       [str(uuid.uuid4())])
    return connection


def identity_snapshot(database, groups):
    path = identity_path(database)
    if not path.is_file():
        return {'entities': [], 'assignments': {}, 'stale': []}
    with _LOCK, duckdb.connect(str(path), read_only=True) as connection:
        entities = [dict(zip(('uid','kind','label','parent_uid','provenance'),row,strict=True))
                    for row in connection.execute('SELECT * FROM entities ORDER BY kind,label').fetchall()]
        stored = connection.execute('SELECT * FROM assignments').fetchall()
    current = {(g['analysis_run_id'],g['observation_id']): g for g in groups}
    valid, stale = {}, []
    for run,observation,binding,table,panel,basis in stored:
        group = current.get((run,observation))
        if group:
            if binding == observation_binding(group):
                known={e['uid']:e for e in entities}
                if known.get(table,{}).get('kind')!='table' or (panel and (
                        known.get(panel,{}).get('kind')!='panel' or known[panel]['parent_uid']!=table)):
                    raise ValueError('Identity assignment references an incompatible entity')
                valid[observation] = {'table_uid':table,'panel_uid':panel,'basis':basis}
            else:
                stale.append(observation)
    return {'entities':entities,'assignments':valid,'stale':stale}


def assign_identity(database, group, table_label, panel_label=None, *, basis='operator_assignment'):
    table_label = str(table_label).strip().upper()
    panel_label = str(panel_label).strip().upper() if panel_label else None
    if not table_label or len(table_label)>100 or (panel_label and len(panel_label)>100):
        raise ValueError('A table label of1..100 characters is required')
    if basis not in ('operator_assignment','provisional_geometry'):
        raise ValueError('Explicit operator or provisional geometry provenance required')
    with _LOCK:
        current_groups=_validate_current(database,group)
        connection = _open(database)
        try:
            connection.execute('BEGIN')
            namespace = uuid.UUID(connection.execute("SELECT value FROM identity_metadata WHERE key='namespace'").fetchone()[0])
            table = connection.execute("SELECT uid FROM entities WHERE kind='table' AND label=? AND parent_uid IS NULL",
                                       [table_label]).fetchone()
            table_uid = table[0] if table else str(uuid.uuid5(namespace,'table/'+table_label))
            if not table:
                connection.execute('INSERT INTO entities VALUES (?,?,?,?,?)',
                                   [table_uid,'table',table_label,None,basis])
            panel_uid = None
            if panel_label:
                panel = connection.execute("SELECT uid FROM entities WHERE kind='panel' AND label=? AND parent_uid=?",
                                           [panel_label,table_uid]).fetchone()
                panel_uid = panel[0] if panel else str(uuid.uuid5(uuid.UUID(table_uid),panel_label))
                if not panel:
                    connection.execute('INSERT INTO entities VALUES (?,?,?,?,?)',
                                       [panel_uid,'panel',panel_label,table_uid,basis])
                peers=connection.execute('SELECT observation,binding FROM assignments WHERE run=? AND panel_uid=? AND observation<>?',
                    [group['analysis_run_id'],panel_uid,group['observation_id']]).fetchall()
                for observation,binding in peers:
                    peer=current_groups.get(observation)
                    if peer and peer['capture_id']==group['capture_id'] and observation_binding(peer)==binding:
                        raise ValueError('Panel ID already belongs to another observation in this frame')
            connection.execute('''INSERT INTO assignments VALUES (?,?,?,?,?,?)
                ON CONFLICT(run,observation) DO UPDATE SET binding=excluded.binding,
                table_uid=excluded.table_uid,panel_uid=excluded.panel_uid,basis=excluded.basis''',
                [group['analysis_run_id'],group['observation_id'],observation_binding(group),table_uid,panel_uid,basis])
            connection.execute('COMMIT')
            return table_uid,panel_uid
        except Exception:
            connection.execute('ROLLBACK')
            raise
        finally:
            connection.close()


def clear_identity(database, group):
    if not identity_path(database).is_file():
        return
    with _LOCK:
        _validate_current(database,group)
        with duckdb.connect(str(identity_path(database))) as connection:
            connection.execute('DELETE FROM assignments WHERE run=? AND observation=?',
                               [group['analysis_run_id'],group['observation_id']])


def assign_table_batch(database, groups, table_label):
    """Atomically assign selected observations to a table after source checks."""
    if not groups or len({g['observation_id'] for g in groups})!=len(groups):
        raise ValueError('Distinct selected observations required')
    if len({g['analysis_run_id'] for g in groups})!=1:
        raise ValueError('Select observations from one inspection')
    table_label=str(table_label).strip().upper()
    if not table_label or len(table_label)>100:
        raise ValueError('A table label of1..100 characters is required')
    with _LOCK:
        current=_validate_current(database,groups[0])
        if any(g['observation_id'] not in current or
               observation_binding(current[g['observation_id']])!=observation_binding(g) for g in groups):
            raise ValueError('Selected observation changed')
        connection=_open(database)
        try:
            connection.execute('BEGIN')
            namespace=uuid.UUID(connection.execute("SELECT value FROM identity_metadata WHERE key='namespace'").fetchone()[0])
            existing=connection.execute("SELECT uid FROM entities WHERE kind='table' AND label=? AND parent_uid IS NULL",[table_label]).fetchone()
            uid=existing[0] if existing else str(uuid.uuid5(namespace,'table/'+table_label))
            if not existing:
                connection.execute('INSERT INTO entities VALUES (?,?,?,?,?)',[uid,'table',table_label,None,'operator_assignment'])
            for group in groups:
                old=connection.execute('SELECT panel_uid FROM assignments WHERE run=? AND observation=?',
                    [group['analysis_run_id'],group['observation_id']]).fetchone()
                if old and old[0]:
                    raise ValueError('Remove an existing panel link before replacing it with a table-only link')
                connection.execute('''INSERT INTO assignments VALUES (?,?,?,?,NULL,'operator_assignment')
                    ON CONFLICT(run,observation) DO UPDATE SET binding=excluded.binding,
                    table_uid=excluded.table_uid,panel_uid=NULL,basis=excluded.basis''',
                    [group['analysis_run_id'],group['observation_id'],observation_binding(group),uid])
            connection.execute('COMMIT')
            return uid
        except Exception:
            connection.execute('ROLLBACK')
            raise
        finally:
            connection.close()


def import_registry(database, entities):
    """Import explicitly supplied identities while rejecting conflicting aliases."""
    by_uid = {e['uid']:e for e in entities}
    if len(by_uid)!=len(entities):
        raise ValueError('Distinct registry UUIDs required')
    for entity in entities:
        uuid.UUID(entity['uid'])
        if entity['kind'] not in ('table','panel') or not entity['label'].strip():
            raise ValueError('Explicit table/panel kind and label required')
        if entity['kind']=='panel' and by_uid.get(entity['parent_uid'],{}).get('kind')!='table':
            raise ValueError('Every imported panel requires its table')
    with _LOCK:
        connection=_open(database)
        try:
            connection.execute('BEGIN')
            for e in entities:
                by_id=connection.execute('SELECT kind,label,parent_uid FROM entities WHERE uid=?',[e['uid']]).fetchone()
                if by_id and by_id!=(e['kind'],e['label'],e['parent_uid']):
                    raise ValueError('Existing UUID refers to a different identity')
                stored=connection.execute('SELECT uid FROM entities WHERE kind=? AND label=? AND parent_uid IS NOT DISTINCT FROM ?',
                                          [e['kind'],e['label'],e['parent_uid']]).fetchone()
                if stored and stored[0]!=e['uid']:
                    raise ValueError('Existing identity label has a different UUID')
                connection.execute('INSERT INTO entities VALUES (?,?,?,?,?) ON CONFLICT(uid) DO NOTHING',
                    [e['uid'],e['kind'],e['label'],e['parent_uid'],e['provenance']])
            connection.execute('COMMIT')
        except Exception:
            connection.execute('ROLLBACK')
            raise
        finally:
            connection.close()


def identity_groups(groups, snapshot, *, prefer_table=False):
    """Use panel first, then table, then the individual unresolved observation."""
    entities = {e['uid']: e for e in snapshot['entities']}
    result = {}
    for observation in groups:
        link = snapshot['assignments'].get(observation['observation_id'])
        uid = (link['table_uid'] if prefer_table else link['panel_uid'] or link['table_uid']) if link else observation['observation_id']
        level = ('panel' if link['panel_uid'] and not prefer_table else 'table') if link else 'unassigned'
        if uid not in result:
            entity = entities.get(uid)
            label=entity['label'] if entity else 'Unassigned'
            if entity and level=='panel':
                label=entities[entity['parent_uid']]['label']+' / '+label
            result[uid] = {'uid':uid,'level':level,'label':label,
                           'observations':[],'regions':[]}
        result[uid]['observations'].append(observation)
        result[uid]['regions'].extend(observation['regions'])
    return list(result.values())
