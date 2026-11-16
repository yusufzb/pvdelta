import hashlib
from copy import deepcopy

import pytest

pytest.importorskip('duckdb')

from pvdelta.boundary_review_store import (
    boundary_result_identity,
    boundary_review_for_result,
    boundary_review_history,
    latest_boundary_reviews,
    save_boundary_review,
)
from pvdelta.store import SCHEMA_VERSION, connect_store


def report():
    proposal={'proposal_id':'N001','polygon_native_xy':[[10,10],[50,10],[50,100],[10,100]]}
    return {'schema':'pvdelta.native-boundary-preview/v1','preview_token':'local-session-token',
        'source_name':'synthetic_T.JPG','source_sha256':'a'*64,'native_temperature_sha256':'b'*64,
        'engine_source_sha256':{'native_windows.py':'c'*64},'native_shape':[512,640],
        'semantic_panel_membership':'unverified','native_values_unchanged':True,
        'measurement_authorised':False,'database_written':False,'anomaly_rerun':False,
        'baseline_proposals':1,'proposals':1,'baseline':{'proposals':[proposal]},
        'proposed':{'proposals':[deepcopy(proposal)]},'radiometric_parameters':{'emissivity':.95}}


def test_decisions_append_and_survive_reopening_without_replacing_snapshot(tmp_path):
    database=tmp_path/'review.duckdb';r=report();before=deepcopy(r)
    first=save_boundary_review(database,r,'ok',reviewer=' Reviewer ',note=' checked ')
    save_boundary_review(database,r,'not_ok',note='Missed panel')
    last=save_boundary_review(database,r,'wait')
    assert r==before
    history=boundary_review_history(database,r['source_sha256'])
    assert [h['review_status'] for h in history]==['wait','not_ok','ok']
    assert first['event_id']<last['event_id']
    assert history[-1]['reviewer']=='Reviewer' and history[-1]['note']=='checked'
    assert latest_boundary_reviews(database)[r['source_sha256']]['review_status']=='wait'
    assert boundary_review_for_result(database,first['result_id'])['event_id']==last['event_id']
    connection=connect_store(database,read_only=True)
    assert connection.execute('SELECT count(*) FROM native_boundary_snapshots').fetchone()[0]==1
    assert connection.execute('SELECT count(*) FROM findings').fetchone()[0]==0
    assert connection.execute('SELECT count(*) FROM finding_reviews').fetchone()[0]==0
    connection.close()


@pytest.mark.parametrize('change',['source','temperature','engine','geometry','settings'])
def test_previous_ok_is_never_reused_for_a_changed_result(tmp_path,change):
    database=tmp_path/'review.duckdb';old=report()
    save_boundary_review(database,old,'ok');new=deepcopy(old)
    if change=='source':new['source_sha256']='d'*64
    elif change=='temperature':new['native_temperature_sha256']='d'*64
    elif change=='engine':new['engine_source_sha256']['native_windows.py']='d'*64
    elif change=='geometry':new['proposed']['proposals'][0]['polygon_native_xy'][0][0]+=1
    else:new['radiometric_parameters']['emissivity']=.9
    assert boundary_result_identity(old)!=boundary_result_identity(new)
    assert boundary_review_for_result(database,boundary_result_identity(new)) is None
    save_boundary_review(database,new,'wait')
    assert boundary_review_for_result(database,boundary_result_identity(old))['review_status']=='ok'


def test_session_token_and_dictionary_order_do_not_create_new_numerical_result():
    old=report();new=dict(reversed(list(old.items())))
    new['preview_token']='different-path-session'
    assert boundary_result_identity(old)==boundary_result_identity(new)


@pytest.mark.parametrize('field,value',[
    ('source_sha256','wrong'),('native_temperature_sha256',None),('engine_source_sha256',{}),
    ('measurement_authorised',True),('native_values_unchanged',False),('proposals',99),
    ('native_shape',[True,640]),('proposed',None),
])
def test_invalid_reports_cannot_create_a_database(tmp_path,field,value):
    database=tmp_path/'review.duckdb';r=report();r[field]=value
    with pytest.raises(ValueError):save_boundary_review(database,r,'ok')
    assert not database.exists()


def test_invalid_status_cannot_create_a_database(tmp_path):
    database=tmp_path/'review.duckdb'
    with pytest.raises(ValueError,match='status'):save_boundary_review(database,report(),'approved')
    assert not database.exists()


def test_old_store_is_read_only_until_a_decision_is_saved_and_migration_preserves_findings(tmp_path):
    database=tmp_path/'legacy.duckdb';connection=connect_store(database)
    connection.execute('DROP TABLE native_boundary_review_reasons')
    connection.execute('DROP TABLE native_boundary_review_events')
    connection.execute('DROP TABLE native_boundary_snapshots')
    connection.execute('DROP SEQUENCE native_boundary_review_seq')
    connection.execute("UPDATE store_metadata SET value='5' WHERE key='schema_version'")
    connection.execute('''INSERT INTO findings (finding_id,observation_id,anomaly_type,assessment_status,
        delta_t_k,absolute_temperature_c,pixel_polygon_json) VALUES ('F','O','hot','review_required',8,42,'[]')''')
    old=connection.execute('SELECT * FROM findings').fetchall();connection.close()
    before=hashlib.sha256(database.read_bytes()).hexdigest()
    assert latest_boundary_reviews(database)=={}
    assert boundary_review_history(database,report()['source_sha256'])==[]
    assert hashlib.sha256(database.read_bytes()).hexdigest()==before
    save_boundary_review(database,report(),'wait')
    connection=connect_store(database,read_only=True)
    assert connection.execute('SELECT * FROM findings').fetchall()==old
    assert connection.execute("SELECT value FROM store_metadata WHERE key='schema_version'").fetchone()[0] == SCHEMA_VERSION
    connection.close()


def test_history_display_limit_does_not_hide_the_current_result_decision(tmp_path):
    database=tmp_path/'review.duckdb';r=report()
    save_boundary_review(database,r,'ok')
    for index in range(51):
        other=deepcopy(r);other['iteration']=index
        save_boundary_review(database,other,'wait')
    assert len(boundary_review_history(database,r['source_sha256']))==50
    assert boundary_review_for_result(database,boundary_result_identity(r))['review_status']=='ok'


def test_reading_missing_store_does_not_create_files(tmp_path):
    database=tmp_path/'absent'/'review.duckdb'
    assert latest_boundary_reviews(database)=={}
    assert boundary_review_history(database,report()['source_sha256'])==[]
    assert not database.parent.exists()


def test_reason_tags_append_without_changing_snapshot_or_old_decisions(tmp_path):
    database=tmp_path/'reasons.duckdb';r=report();identity=boundary_result_identity(r)
    save_boundary_review(database,r,'not_ok',reason_codes=['missing_panel','non_panel_region','missing_panel'])
    save_boundary_review(database,r,'wait',reason_codes=['insufficient_view'])
    save_boundary_review(database,r,'ok')
    history=boundary_review_history(database,r['source_sha256'])
    assert [set(h['reason_codes']) for h in history]==[
        set(),{'insufficient_view'},{'missing_panel','non_panel_region'}]
    assert {h['result_id'] for h in history}=={identity}
    assert latest_boundary_reviews(database)[r['source_sha256']]['reason_codes']==[]
    connection=connect_store(database,read_only=True)
    assert connection.execute('SELECT count(*) FROM native_boundary_snapshots').fetchone()[0]==1
    assert connection.execute('SELECT count(*) FROM native_boundary_review_reasons').fetchone()[0]==3
    connection.close()


@pytest.mark.parametrize('status,codes',[
    ('not_ok','missing_panel'),('wait',None),('not_ok',['electrical_fault']),
    ('wait',[[]]),('ok',['missing_panel']),
])
def test_invalid_reason_input_cannot_create_database(tmp_path,status,codes):
    database=tmp_path/'invalid.duckdb'
    with pytest.raises(ValueError,match='reason'):save_boundary_review(database,report(),status,reason_codes=codes)
    assert not database.exists()


def test_v6_reviews_read_without_migration_and_upgrade_preserves_their_history(tmp_path):
    database=tmp_path/'v6.duckdb';r=report()
    save_boundary_review(database,r,'not_ok',note='Legacy missing panel')
    connection=connect_store(database)
    connection.execute('DROP TABLE native_boundary_review_reasons')
    connection.execute("UPDATE store_metadata SET value='6' WHERE key='schema_version'")
    old=connection.execute('SELECT * EXCLUDE(reviewed_at_utc) FROM native_boundary_review_events').fetchall()
    connection.close()
    before=hashlib.sha256(database.read_bytes()).hexdigest()
    latest=latest_boundary_reviews(database)[r['source_sha256']]
    assert latest['reason_codes']==[] and latest['note']=='Legacy missing panel'
    assert boundary_review_for_result(database,latest['result_id'])['reason_codes']==[]
    assert hashlib.sha256(database.read_bytes()).hexdigest()==before
    save_boundary_review(database,r,'wait',reason_codes=['insufficient_view'])
    history=boundary_review_history(database,r['source_sha256'])
    assert history[0]['reason_codes']==['insufficient_view']
    assert history[1]['note']=='Legacy missing panel' and history[1]['reason_codes']==[]
    connection=connect_store(database,read_only=True)
    assert connection.execute('SELECT * EXCLUDE(reviewed_at_utc) FROM native_boundary_review_events ORDER BY event_id LIMIT 1').fetchall()==old
    assert connection.execute("SELECT value FROM store_metadata WHERE key='schema_version'").fetchone()[0] == SCHEMA_VERSION
    connection.close()
