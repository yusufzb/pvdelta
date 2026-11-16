"""Read-only acquisition validation with explicit coverage and failure accounting."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

from pvdelta.boundary_validation import compare_boundary_reference, validation_engine_sources
from pvdelta.instance_benchmark import evaluate_panel_instances
from pvdelta.native_review import (
    boundary_engine_sources,
    indexed_native_frames,
    native_boundary_preview,
)


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _references(content):
    document = json.loads(content)
    if not isinstance(document,dict):
        raise TypeError('Reference JSON must be an object')
    frames = document.get('frames',[document])
    if not isinstance(frames,list) or not frames or any(not isinstance(r,dict) for r in frames):
        raise ValueError('Reference bundle requires a nonempty frames list')
    seen = set()
    for ref in frames:
        source = ref.get('source_sha256')
        if not isinstance(source,str) or source in seen:
            raise ValueError('References require unique source SHA256 identities')
        seen.add(source)
        shape = (ref.get('native_height'),ref.get('native_width'))
        if any(type(n) is not int for n in shape):
            raise ValueError('Reference requires integer native grid dimensions')
        # Validate every reference, including missing/unreadable source frames,
        # before creating output or decoding any SDK image. Discard these scores.
        try:
            evaluate_panel_instances(ref,[],source_sha256=source,native_shape=shape)
        except (TypeError, KeyError, AttributeError) as exc:
            raise ValueError('Invalid full-panel reference structure') from exc
    json.dumps(document,allow_nan=False)
    return frames


def _aggregate(rows):
    result = {}
    for mode in ('baseline','proposed'):
        result[mode] = {}
        for threshold in ('0.5','0.75'):
            scores = [r['scores'][mode][threshold] for r in rows if r['status']=='succeeded']
            if not scores:
                result[mode][threshold] = None
                continue
            tp,fp,fn = [sum(s[k] for s in scores)
                       for k in ('true_positives','false_positives','false_negatives')]
            result[mode][threshold] = {
                'scored_frame_records':len(scores),
                'reference_instances':sum(s['reference_instances'] for s in scores),
                'predictions':sum(s['predictions'] for s in scores),
                'ignored_predictions':sum(len(s['ignored_proposal_ids']) for s in scores),
                'true_positives':tp,'false_positives':fp,'false_negatives':fn,
                'precision':tp/(tp+fp) if tp+fp else None,
                'recall':tp/(tp+fn) if tp+fn else None,
            }
    return result


def _write_summary(output, report):
    report_json=json.dumps(report,indent=2,allow_nan=False)
    def esc(value):return html.escape(str(value),quote=True)
    page='''<!doctype html><meta charset="utf-8"><title>Boundary validation batch</title>
<style>body{font:16px system-ui;background:#111827;color:#eee;max-width:1350px;margin:auto;padding:24px}
a{color:#7dd3fc}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #526073;text-align:left}
.pair{display:flex;gap:12px}.pair img{width:49%;object-fit:contain}pre{white-space:pre-wrap}</style>
<h1>Acquisition boundary validation</h1>
<p>Frame-local geometry against supplied labels. Not site inventory, anomaly accuracy or IEC acceptance.</p>'''
    page+=f"<p>Status: <strong>{esc(report['status'])}</strong> · Acquisition: {esc(report['acquisition_id'])}</p>"
    page+=f"<p>Requested: {report['requested_frames']} · Succeeded: {report['succeeded_frames']} · Failed: {report['failed_frames']} · Not processed: {report['unprocessed_frames']}</p>"
    page+=f"<p>Reference instances requested: {report['requested_reference_instances']} · Scored: {report['scored_reference_instances']}. Indexed thermal records: {report['indexed_acquisition_frames']} · Without supplied references: {report['indexed_records_without_reference']}.</p>"
    page+='<p>Scores below use successful reference frames only. Failed/unprocessed frames are not healthy or zero-error results. Overlapping photos can contain the same physical panel.</p>'
    page+='<h2>Scores / provenance</h2><pre>'+esc(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2))+'</pre>'
    page+='<h2>All requested frames</h2><table><tr><th>Source / SHA256</th><th>Declared labels</th><th>Status / error</th><th>Baseline TP/FP/FN (.75)</th><th>Proposed TP/FP/FN (.75)</th></tr>'
    for row in report['frames']:
        cells=[row.get('source_name') or row['source_sha256'],row['declared_reference_review_status'],row['status']]
        if row.get('error'):cells[2]+=': '+row['error']
        for mode in ('baseline','proposed'):
            score=row.get('scores',{}).get(mode,{}).get('0.75')
            cells.append('/'.join(str(score[k]) for k in ('true_positives','false_positives','false_negatives')) if score else 'Not scored')
        page+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in cells)+'</tr>'
    page+='</table><p>Green TP: matched; amber FP: unmatched; dashed red FN: missed reference; grey IGN: explicit ignore. Geometry matching, not fault severity.</p>'
    for row in report['frames']:
        if row['status']=='succeeded':
            folder='frames/'+row['source_sha256']
            page+='<h2>'+esc(row['source_name'])+'</h2><p><a href="'+folder+'/comparison.json">Exact scores and identities</a></p>'
            page+=f'<div class="pair"><img src="{folder}/baseline.png" alt="Baseline errors"><img src="{folder}/proposed.png" alt="Proposed errors"></div>'
    for name,content in (('summary.json',report_json),('summary.html',page)):
        temporary=output/(name+'.tmp')
        temporary.write_text(content,encoding='utf-8')
        temporary.replace(output/name)


def run_boundary_validation(database, acquisition_id, references, output, *, sdk_library=None):
    """Validate only explicitly labelled frames; never write to the inspection DB.

    A new output directory is mandatory. Per-frame decode/input failures remain
    visible; code/reference changes invalidate the batch instead of mixing runs.
    JSON/HTML progress is checkpointed after each frame; there is no implicit resume.
    """
    import duckdb

    references = Path(references).expanduser().resolve()
    content = references.read_bytes()
    refs = _references(content)
    try:
        frames = [f for f in indexed_native_frames(database) if f['run_id']==acquisition_id]
    except duckdb.Error as exc:
        raise ValueError(f'Cannot read inspection index: {exc}') from exc
    if not frames:
        raise ValueError('Selected acquisition has no indexed thermal frames')
    sources = {}
    for frame in frames:
        sources.setdefault(frame['source_hash'],[]).append(frame)
    extractor = boundary_engine_sources()
    scorer = validation_engine_sources()
    runner_hash = _sha(Path(__file__).read_bytes())
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True,exist_ok=False)
    rows = [{'source_sha256':ref['source_sha256'],'source_name':None,'status':'pending',
             'reference_instances':len(ref['panels']),
             'declared_reference_review_status':ref['review_status']} for ref in refs]
    reference_hashes = {r['source_sha256'] for r in refs}

    def summary(status, error=None):
        successful = [r for r in rows if r['status']=='succeeded']
        return {'schema':'pvdelta.boundary-validation-batch/v1','status':status,
            'acquisition_id':acquisition_id,'requested_frames':len(rows),
            'succeeded_frames':len(successful),'failed_frames':sum(r['status']=='failed' for r in rows),
            'unprocessed_frames':sum(r['status']=='pending' for r in rows),
            'requested_reference_instances':sum(r['reference_instances'] for r in rows),
            'scored_reference_instances':sum(r['reference_instances'] for r in successful),
            'indexed_acquisition_frames':len(frames),
            'indexed_records_without_reference':sum(f['source_hash'] not in reference_hashes for f in frames),
            'score_scope':'Successful reference frames only; frame observations are not unique physical panels',
            'reference_file_sha256':_sha(content), 'extractor_source_sha256':extractor,
            'scorer_source_sha256':scorer,'runner_sha256':runner_hash,
            'error':error,'aggregate_scores':_aggregate(rows) if status!='invalidated' else None,
            'measurement_authorised':False,'database_written':False,'anomaly_rerun':False,
            'independent_validation_verified':False,'frames':rows}

    _write_summary(output,summary('running'))
    for ref,row in zip(refs,rows):
        def unchanged():
            return (references.read_bytes()==content and boundary_engine_sources()==extractor
                    and validation_engine_sources()==scorer and _sha(Path(__file__).read_bytes())==runner_hash)
        try:
            if not unchanged():
                raise RuntimeError('Reference or validation code changed during the batch')
        except (OSError,RuntimeError) as exc:
            report=summary('invalidated',str(exc));_write_summary(output,report)
            return report
        try:
            matches = sources.get(ref['source_sha256'],[])
            if len(matches)!=1:
                raise ValueError('Exactly one indexed capture with this source SHA256 is required in the acquisition')
            record=matches[0]
            row.update(source_name=record['source_name'],capture_id=record['capture_id'])
            if not record['source_path']:
                raise ValueError('Indexed source path is unavailable or escapes its acquisition root')
            preview=native_boundary_preview(record['source_path'],sdk_library,
                                            expected_sha256=ref['source_sha256'])
            comparison=compare_boundary_reference(preview,content)
            if not unchanged():
                raise RuntimeError('Reference or validation code changed during the batch')
        except (OSError,RuntimeError,ValueError,TypeError,KeyError,ImportError) as exc:
            row.update(status='failed',error=str(exc))
        else:
            folder=output/'frames'/ref['source_sha256'];folder.mkdir(parents=True)
            # Only mark success after every artifact is present; output I/O errors
            # propagate and leave the previous checkpoint clearly incomplete.
            (folder/'preview.json').write_text(json.dumps(preview.report,indent=2,allow_nan=False),encoding='utf-8')
            (folder/'comparison.json').write_text(json.dumps(comparison.report,indent=2,allow_nan=False),encoding='utf-8')
            comparison.baseline_image.save(folder/'baseline.png')
            comparison.proposed_image.save(folder/'proposed.png')
            row.update(status='succeeded',scores=comparison.report['scores'],
                       boundary_result_id=comparison.report['boundary_result_id'])
        _write_summary(output,summary('running'))
    try:
        valid=unchanged()
    except OSError:
        valid=False
    status=('complete' if all(r['status']=='succeeded' for r in rows) else
            'partial' if any(r['status']=='succeeded' for r in rows) else 'failed')
    report=summary(status if valid else 'invalidated',None if valid else 'Reference or code changed during the batch')
    _write_summary(output,report)
    return report
