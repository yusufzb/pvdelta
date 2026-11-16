"""Atomic local comparison records, including explicit unavailable source attempts."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import duckdb

from pvdelta.candidate_links import candidate_card_links, candidate_links_signature
from pvdelta.rgb_pair_check import digest
from pvdelta.store import connect_store

_ERRORS = (OSError, ValueError, RuntimeError, TypeError, KeyError, duckdb.Error, cv2.error)


def comparison_path(database, run_id, observation_id):
    database = Path(database).resolve()
    return database.parent/'candidate-comparisons'/digest([database.name, run_id])/(digest(observation_id)+'.json')


def candidate_population(database, run_id):
    c = connect_store(database, read_only=True)
    try:
        return c.execute('''SELECT m.observation_id,min(f.finding_id),c.source_path FROM findings f
            JOIN module_observations m USING(observation_id) JOIN captures c USING(capture_id)
            WHERE f.analysis_run_id=? AND m.analysis_run_id=? GROUP BY m.observation_id,c.source_path
            ORDER BY c.source_path,m.observation_id''', [run_id, run_id]).fetchall()
    finally:
        c.close()


def _binding(database, run_id, observation_id, finding_id, registration, sdk_library, max_targets, hash_file=None):
    signature = candidate_links_signature(database, finding_id, registration,
        sdk_library=sdk_library, max_targets=max_targets, **({'hash_file':hash_file} if hash_file else {}))
    return {'signature':signature, 'run_id':run_id, 'observation_id':observation_id,
        'finding_id':finding_id, 'max_targets':max_targets,
        'store_engine':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _validate(record, run_id, observation_id, finding_id):
    if (record.get('schema')!='pvdelta.saved-candidate-comparison/v1'
            or record['analysis_run_id']!=run_id or record['observation_id']!=observation_id
            or record['finding_id']!=finding_id or record['status'] not in ('available','unavailable')
            or record['content_sha256']!=digest({k:v for k,v in record.items() if k!='content_sha256'})):
        raise ValueError('Comparison record identity or checksum differs')
    if record['status']=='available':
        report=record['report']
        if (report['schema']!='pvdelta.candidate-card-links/v1' or report['finding_id']!=finding_id
                or report['source_observation_id']!=observation_id
                or report['binding']!=record['binding']['signature']
                or report['content_sha256']!=digest({k:v for k,v in report.items() if k!='content_sha256'})
                or report['physical_panels_merged'] is not False or report['decisions_transferred'] is not False):
            raise ValueError('Comparison report differs from its selected card')
    elif not isinstance(record.get('reason'),str) or not record['reason']:
        raise ValueError('Unavailable source requires an explicit reason')


def load_candidate_comparison(database, run_id, observation_id, finding_id, registration, *, sdk_library=None,
                              max_targets=4, hash_file=None):
    path=comparison_path(database,run_id,observation_id)
    if not path.is_file():
        return {'status':'missing','reason':'No saved comparison'}
    try:
        record=json.loads(path.read_bytes())
        _validate(record,run_id,observation_id,finding_id)
        binding=_binding(database,run_id,observation_id,finding_id,registration,sdk_library,max_targets,hash_file)
        if record['binding']!=binding:
            raise ValueError('Comparison inputs or engine changed')
        return record
    except _ERRORS as exc:
        return {'status':'stale','reason':str(exc)}


def check_and_save_comparison(database, run_id, observation_id, finding_id, registration, *,
                              sdk_library=None, max_targets=4):
    # Verify the representative belongs to the exact selected run/card.
    c=connect_store(database,read_only=True)
    try:
        if c.execute('SELECT observation_id FROM findings WHERE analysis_run_id=? AND finding_id=?',
                [run_id,finding_id]).fetchone()!=(observation_id,):
            raise ValueError('Finding is outside selected run/card')
    finally:
        c.close()
    before=_binding(database,run_id,observation_id,finding_id,registration,sdk_library,max_targets)
    record={'schema':'pvdelta.saved-candidate-comparison/v1','analysis_run_id':run_id,
        'observation_id':observation_id,'finding_id':finding_id,'binding':before,
        'checked_at_utc':datetime.now(timezone.utc).isoformat()}
    try:
        report=candidate_card_links(database,finding_id,registration,sdk_library=sdk_library,max_targets=max_targets)
        record.update(status='available',report=report)
    except _ERRORS as exc:
        record.update(status='unavailable',reason=str(exc))
    after=_binding(database,run_id,observation_id,finding_id,registration,sdk_library,max_targets)
    if before!=after:
        raise ValueError('Inputs changed during comparison; no record saved')
    record['content_sha256']=digest(record)
    _validate(record,run_id,observation_id,finding_id)
    path=comparison_path(database,run_id,observation_id);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(json.dumps(record,sort_keys=True,allow_nan=False).encode())
            stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def _initialize_worker():
    cv2.setNumThreads(1)


def _check_job(job):
    database,run_id,oid,fid,registration,sdk,max_targets=job
    try:
        record=check_and_save_comparison(database,run_id,oid,fid,registration,sdk_library=sdk,max_targets=max_targets)
        return {'observation_id':oid,'status':record['status'],
            'links':sum(r['status']=='provisional_candidate_link' for r in record.get('report',{}).get('rows',[])),
            'reason':record.get('reason','')}
    except _ERRORS as exc:
        return {'observation_id':oid,'status':'failed','reason':str(exc)}


def build_candidate_comparisons(database, run_id, registration, *, sdk_library=None, max_targets=4,
                                workers=2, force=False, progress=None):
    if type(workers) is not int or not 1<=workers<=4:
        raise ValueError('workers must be in 1..4')
    population=candidate_population(database,run_id)
    results=[];jobs=[];hashes={}
    def hash_file(path):
        path=Path(path).resolve()
        if path not in hashes:
            try:hashes[path]=hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:hashes[path]=None
        return hashes[path]
    for oid,fid,_ in population:
        saved=load_candidate_comparison(database,run_id,oid,fid,registration,sdk_library=sdk_library,
            max_targets=max_targets,hash_file=hash_file) if not force else {'status':'missing'}
        if saved['status'] in ('available','unavailable'):
            results.append({'observation_id':oid,'status':'reused','saved_status':saved['status']})
        else:
            jobs.append((str(database),run_id,oid,fid,registration,str(sdk_library) if sdk_library else None,max_targets))
    if workers==1:
        for job in jobs:
            results.append(_check_job(job))
            if progress:progress(len(results),len(population),results[-1])
    elif jobs:
        with ProcessPoolExecutor(max_workers=workers,initializer=_initialize_worker) as pool:
            # Bounded submitted work and memory; each process reads originals only.
            pending={i:pool.submit(_check_job,jobs[i]) for i in range(min(len(jobs),workers*2))}
            for i in range(len(jobs)):
                results.append(pending.pop(i).result())
                next_index=i+workers*2
                if next_index<len(jobs):pending[next_index]=pool.submit(_check_job,jobs[next_index])
                if progress:progress(len(results),len(population),results[-1])
    return {'population':len(population),'results':results,'database_written':False,
        'scope':'Bounded candidate-containing captures only; not every thermal view or physical panel.'}
