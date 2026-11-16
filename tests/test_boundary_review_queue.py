from pvdelta.boundary_review_queue import (
    boundary_issue_counts,
    boundary_review_queue,
    filter_boundary_queue,
)


def sample_queue():
    frames=[{'capture_id':str(i),'run_id':'flight-A','source_name':f'{i}_T.JPG',
             'source_hash':str(i)*64,'source_path':None} for i in range(5)]
    def review(status,reasons,engine='current'):
        return {'review_status':status,'reason_codes':reasons,'engine_signature':engine,
                'result_id':'sha256:result','event_id':10,'note':'Keep provenance'}
    saved={
        '0'*64:review('not_ok',['missing_panel','non_panel_region']),
        '1'*64:review('wait',['missing_panel']),
        '2'*64:review('ok',[]),
        '3'*64:review('not_ok',['missing_panel'],'older'),
    }
    return frames,saved,boundary_review_queue(frames,saved,'current')


def test_only_latest_current_engine_pending_reasons_are_counted_with_overlap():
    frames,saved,queue=sample_queue()
    counts={r['reason_code']:r['frame_records'] for r in boundary_issue_counts(queue)}
    assert counts=={'missing_panel':2,'non_panel_region':1}
    assert len(filter_boundary_queue(queue,'Needs review','missing_panel'))==2
    saved['0'*64]=dict(saved['0'*64],review_status='ok',reason_codes=[])
    updated=boundary_review_queue(frames,saved,'current')
    assert boundary_issue_counts(updated)==[
        {'reason_code':'missing_panel','issue':'Missing panel','frame_records':1}]
    assert len(filter_boundary_queue(updated,'Saved OK'))==2


def test_history_export_keeps_other_engine_tags_and_provenance_but_does_not_verify_files():
    _,_,queue=sample_queue();old=queue[3]
    assert old['state']=='other_engine' and old['reason_codes']==['missing_panel']
    assert old['latest_saved_status']=='not_ok'
    assert old['result_id']=='sha256:result' and old['event_id']==10
    assert old['note']=='Keep provenance' and old['source_sha256']=='3'*64
    assert old['acquisition_id']=='flight-A' and not old['exact_current_result_checked']
    assert filter_boundary_queue(queue,'Other engine version')==[old]
    assert filter_boundary_queue(queue,'Other engine version','missing_panel')==[]
    assert queue[4]['state']=='no_review' and queue[4]['reason_codes']==[]


def test_no_reason_filter_only_selects_current_engine_pending_events():
    frames,saved,_=sample_queue()
    saved['1'*64]['reason_codes']=[]
    queue=boundary_review_queue(frames,saved,'current')
    assert [r['capture_id'] for r in filter_boundary_queue(queue,reason='No recorded reason')]==['1']
