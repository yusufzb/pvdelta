"""Deterministic, prediction-free preparation of private native annotation packs."""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pvdelta.decode import RJpeg
from pvdelta.native_review import indexed_native_frames
from pvdelta.store import connect_store


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _valid_hash(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _time(value):
    try:
        stamp=dt.datetime.fromisoformat(value)
        return stamp.timestamp() if stamp.tzinfo is not None else None
    except (ValueError,TypeError,OverflowError):
        return None


def select_review_frames(frames, acquisition_id, *, count=12, seed='pvdelta-review-v1', excluded=()):
    """Use chronological quantile bins plus a seeded source-hash choice per bin.

    A separate untimed pool receives at least one slot when the budget allows.
    No detector scores, anomaly results or panel counts participate in selection.
    Adjacent/overlapping scenes can remain: this is not an independence certificate.
    """
    if type(count) is not int or count<1 or not isinstance(seed,str) or not seed:
        raise ValueError('Positive integer count and nonempty text seed required')
    excluded=set(excluded)
    if any(not _valid_hash(v) for v in excluded):
        raise ValueError('Exclusions require lowercase source SHA256 values')
    scoped=[f for f in frames if f['run_id']==acquisition_id]
    captures=[f['capture_id'] for f in scoped]
    if len(set(captures))!=len(captures):
        raise ValueError('Acquisition capture IDs must be unique')
    frequencies=Counter(f['source_hash'] for f in scoped)
    eligible=[];rejected=[]
    for frame in scoped:
        digest=frame['source_hash']
        reason=('invalid_source_hash' if not _valid_hash(digest) else
                'ambiguous_source_identity' if frequencies[digest]!=1 else
                'excluded_source' if digest in excluded else
                'source_path_unavailable' if not frame.get('source_path') else None)
        if reason:
            rejected.append({'capture_id':frame['capture_id'],'source_sha256':digest,'reason':reason})
        else:
            eligible.append(frame)
    timed=sorted((f for f in eligible if _time(f.get('captured_at')) is not None),
                 key=lambda f:(_time(f['captured_at']),f['source_hash']))
    untimed=sorted((f for f in eligible if _time(f.get('captured_at')) is None),key=lambda f:f['source_hash'])
    budget=min(count,len(eligible))
    reserved=1 if untimed and (budget>1 or not timed) else 0
    timed_count=min(len(timed),budget-reserved)
    untimed_count=budget-timed_count
    selected=[]
    for pool,slots,label in ((timed,timed_count,'chronological_quantile'),(untimed,untimed_count,'time_unavailable')):
        for index in range(slots):
            bucket=pool[index*len(pool)//slots:(index+1)*len(pool)//slots]
            choice=min(bucket,key=lambda f:(_sha((seed+'\0'+f['source_hash']).encode()),f['source_hash']))
            selected.append(dict(choice,selection_pool=label,selection_bin=index+1,
                bin_candidates=len(bucket),bin_first_capture_time=bucket[0].get('captured_at'),
                bin_last_capture_time=bucket[-1].get('captured_at')))
    return {'schema':'pvdelta.boundary-review-selection/v1','acquisition_id':acquisition_id,
        'requested_frames':count,'selected_frames':len(selected),'selection_shortfall':count-len(selected),
        'indexed_acquisition_frames':len(scoped),'eligible_frames':len(eligible),
        'eligible_timed_frames':len(timed),'eligible_untimed_frames':len(untimed),
        'rejection_counts':dict(sorted(Counter(r['reason'] for r in rejected).items())),
        'seed':seed,'selection_method':'Chronological quantile bins; seeded SHA256 choice within each bin; separate untimed pool',
        'predictions_used_for_selection':False,'scene_independence_verified':False,
        'scope':'Annotation preparation only; exclusion by exact bytes cannot remove all overlapping scenes',
        'selected':selected,'rejected':rejected}


def _load_exclusions(paths):
    documents=[];excluded=set()
    for path in paths:
        path=Path(path).expanduser().resolve();content=path.read_bytes();document=json.loads(content)
        if isinstance(document,list):
            hashes=document
        elif isinstance(document,dict) and isinstance(document.get('frames'),list):
            if any(not isinstance(f,dict) for f in document['frames']):
                raise ValueError('Exclusion frames must be objects with source_sha256')
            hashes=[f.get('source_sha256') for f in document['frames']]
        elif isinstance(document,dict):
            hashes=[document.get('source_sha256')]
        else:
            raise TypeError('Exclusions must be a reference object, frames bundle or SHA256 list')
        if any(not _valid_hash(v) for v in hashes):
            raise ValueError('Exclusions require lowercase source SHA256 values')
        excluded.update(hashes)
        documents.append((path,content))
    return excluded,documents


def _read_frames(database,acquisition_id):
    import duckdb

    try:
        frames=indexed_native_frames(database)
        connection=connect_store(database,read_only=True)
        try:
            cursor=connection.execute('''SELECT capture_id,camera_latitude,camera_longitude,
                gimbal_pitch_deg,gimbal_yaw_deg,camera_model,paired_capture_id
                FROM captures WHERE run_id=? AND image_kind='thermal' ''',[acquisition_id])
            columns=[c[0] for c in cursor.description]
            metadata={r[0]:dict(zip(columns,r)) for r in cursor.fetchall()}
        finally:
            connection.close()
    except duckdb.Error as exc:
        raise ValueError(f'Cannot read inspection index: {exc}') from exc
    return [dict(f,**{k:v for k,v in metadata[f['capture_id']].items() if k!='capture_id'})
            for f in frames if f['run_id']==acquisition_id]


def _render_native(path,sdk_library,source_sha256):
    from PIL import Image

    path=Path(path)
    if _sha(path.read_bytes())!=source_sha256:
        raise ValueError('Thermal source SHA256 differs from the indexed record')
    with RJpeg(path,sdk_library) as image:
        values=image.temperature();display=image.processed_rgb();parameters=asdict(image.measurement_params)
    if (values.ndim!=2 or min(values.shape)<1 or not np.isfinite(values).all() or display.dtype!=np.uint8
            or display.shape!=(*values.shape,3)):
        raise ValueError('SDK render and finite temperatures must share the exact native grid')
    if _sha(path.read_bytes())!=source_sha256:
        raise ValueError('Thermal source changed during rendering')
    return Image.fromarray(display),{'source_sha256':source_sha256,'native_shape':list(values.shape),
        'native_temperature_sha256':_sha(values.tobytes()),'radiometric_parameters':parameters,
        'panel_proposals_generated':False,'measurement_authorised':False}


def _write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def prepare_boundary_review(database,acquisition_id,output,*,count=12,seed='pvdelta-review-v1',
                            exclusions=(),sdk_library=None):
    """Freeze a selection, then render unmarked SDK images without running a detector."""
    excluded,documents=_load_exclusions(exclusions)
    selection=select_review_frames(_read_frames(database,acquisition_id),acquisition_id,
                                   count=count,seed=seed,excluded=excluded)
    if not selection['selected']:
        raise ValueError('No eligible thermal frames for this acquisition and exclusion set')
    code={name:_sha(Path(__file__).with_name(name).read_bytes())
          for name in ('boundary_review_pack.py','decode.py','native_review.py')}
    selection.update(exclusion_documents=[{'name':p.name,'sha256':_sha(c)} for p,c in documents],
                     source_code_sha256=code)
    output=Path(output).expanduser().resolve();output.mkdir(parents=True,exist_ok=False)
    _write_json(output/'selection.json',selection)
    rows=[{'capture_id':f['capture_id'],'source_sha256':f['source_hash'],
           'source_name':f['source_name'],'status':'pending'} for f in selection['selected']]
    templates=[]

    def progress(status):
        report={'schema':'pvdelta.boundary-review-pack/v1','status':status,
            'acquisition_id':acquisition_id,'requested_frames':count,'selected_frames':len(rows),
            'succeeded_frames':sum(r['status']=='succeeded' for r in rows),
            'failed_frames':sum(r['status']=='failed' for r in rows),
            'unprocessed_frames':sum(r['status']=='pending' for r in rows),
            'selection_shortfall':selection['selection_shortfall'],
            'eligible_frames':selection['eligible_frames'],
            'rejection_counts':selection['rejection_counts'],
            'invalidation_reason':'Exclusion documents or source code changed' if status=='invalidated' else None,
            'predictions_generated':False,'reference_labels_completed':False,
            'independent_validation_verified':False,'database_written':False,'frames':rows}
        _write_json(output/'progress.json',report)
        _write_json(output/'draft-references.json',{'provenance':'Unannotated templates. No panel/ignore labels or review decisions have been generated.',
                                                  'frames':templates})
        page='''<!doctype html><meta charset="utf-8"><title>Boundary annotation review pack</title>
<style>body{font:16px system-ui;background:#111827;color:#eee;max-width:1200px;margin:auto;padding:24px}a{color:#7dd3fc}img{max-width:100%}pre{white-space:pre-wrap}</style>
<h1>Unmarked native boundary review pack</h1><p>Annotation preparation, not accuracy results.
No detector predictions or panel counts were generated. Camera GPS is not panel GPS.
Exact-byte exclusions do not guarantee scene independence.</p>'''
        page+='<pre>'+html.escape(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2))+'</pre>'
        page+='<p><a href="selection.json">Frozen selection / exclusions / metadata</a> · <a href="draft-references.json">Unannotated reference templates</a></p>'
        for row in rows:
            page+='<h2>'+html.escape(row['source_name'])+'</h2><p>'+html.escape(row['status']+(': '+row['error'] if row.get('error') else ''))+'</p>'
            if row['status']=='succeeded':
                page+=f'<img src="frames/{row["source_sha256"]}/native.png" alt="Unmarked SDK native render">'
        temporary=output/'summary.html.tmp';temporary.write_text(page,encoding='utf-8');temporary.replace(output/'summary.html')
        return report

    def unchanged():
        try:
            return (all(p.read_bytes()==c for p,c in documents)
                    and all(_sha(Path(__file__).with_name(n).read_bytes())==h for n,h in code.items()))
        except OSError:
            return False

    progress('running')
    for frame,row in zip(selection['selected'],rows):
        try:
            if not unchanged():return progress('invalidated')
            picture,detail=_render_native(frame['source_path'],sdk_library,frame['source_hash'])
            if not unchanged():return progress('invalidated')
        except (OSError,ValueError,RuntimeError,TypeError,ImportError) as exc:
            row.update(status='failed',error=str(exc))
        else:
            folder=output/'frames'/frame['source_hash'];folder.mkdir(parents=True)
            picture.save(folder/'native.png');_write_json(folder/'native.json',detail)
            height,width=detail['native_shape']
            templates.append({'schema':'pvdelta.native-panel-instances/v1','source':frame['source_name'],
                'source_sha256':frame['source_hash'],'coordinate_space':'thermal_native',
                'native_width':width,'native_height':height,'coverage':'not_annotated',
                'review_status':'unreviewed','panels':None,'ignore':None})
            row.update(status='succeeded',native_temperature_sha256=detail['native_temperature_sha256'])
        progress('running')
    try:
        valid=unchanged()
    except OSError:
        valid=False
    status=('complete' if all(r['status']=='succeeded' for r in rows) else
            'partial' if any(r['status']=='succeeded' for r in rows) else 'failed')
    return progress(status if valid else 'invalidated')
