"""Acquisition-scoped, historical review queue; never infer panel or fault counts."""
from pathlib import Path

from pvdelta.boundary_review_store import BOUNDARY_REASONS

REVIEW_STATES = {
    'no_review':'No saved review', 'ok':'Saved OK', 'not_ok':'Saved Not OK',
    'wait':'Saved Wait', 'other_engine':'Other engine version',
}


def boundary_review_queue(frames, saved, engine):
    """Latest event per source, with enough provenance to reopen the exact review.

    An engine match alone does not validate today's file/native data/settings.
    Queue entries remain historical until the exact preview is compared again.
    """
    rows = []
    for frame in frames:
        review = saved.get(frame['source_hash']) or {}
        state = (review['review_status'] if review.get('engine_signature')==engine
                 else 'other_engine') if review else 'no_review'
        rows.append({
            'frame':frame.get('source_name') or Path(frame.get('source_path') or '').name,
            'capture_id':frame['capture_id'], 'acquisition_id':frame['run_id'],
            'source_sha256':frame['source_hash'], 'state':state, 'saved_state':REVIEW_STATES[state],
            'latest_saved_status':review.get('review_status'),
            'reason_codes':list(review.get('reason_codes', [])),
            'result_id':review.get('result_id'), 'event_id':review.get('event_id'),
            'reviewer':review.get('reviewer'), 'note':review.get('note'),
            'reviewed_at':review.get('reviewed_at_utc'),
            'saved_engine_signature':review.get('engine_signature'),
            'current_engine_signature':engine, 'exact_current_result_checked':False,
        })
    return rows


def boundary_issue_counts(queue):
    """Count frame records once per reason in current-engine Not OK / Wait events.

    Categories overlap; do not sum these counts into a frame/panel total. Later
    OK events and other-engine history do not represent current correction tasks.
    """
    pending = [r for r in queue if r['state'] in ('not_ok','wait')]
    return [{'reason_code':code, 'issue':label,
             'frame_records':sum(code in r['reason_codes'] for r in pending)}
            for code,label in BOUNDARY_REASONS.items()
            if any(code in r['reason_codes'] for r in pending)]


def filter_boundary_queue(queue, status='All frames', reason='All issues'):
    rows = queue
    if status=='Needs review':
        rows = [r for r in rows if r['state']!='ok']
    elif status!='All frames':
        rows = [r for r in rows if r['saved_state']==status]
    if reason!='All issues':
        rows = [r for r in rows if r['state'] in ('not_ok','wait') and (
            not r['reason_codes'] if reason=='No recorded reason' else reason in r['reason_codes'])]
    return rows
