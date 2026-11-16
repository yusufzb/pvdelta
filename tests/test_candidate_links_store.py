import hashlib
import json

import pytest
from test_panel_review import panel_database  # noqa: F401

from pvdelta import candidate_links_store as store
from pvdelta.rgb_pair_check import digest
from pvdelta.store import connect_store, upsert_finding_review


def report(oid='obs1',fid='f1',binding='current'):
    p={'schema':'pvdelta.candidate-card-links/v1','source_observation_id':oid,'finding_id':fid,
        'binding':binding,'rows':[{'status':'provisional_candidate_link'}],
        'physical_panels_merged':False,'decisions_transferred':False}
    p['content_sha256']=digest(p)
    return p


def setup(monkeypatch):
    calls=[]
    monkeypatch.setattr(store,'candidate_links_signature',lambda *a,**k:'current')
    def compute(db,fid,*a,**k):
        calls.append(fid);return report('obs1' if fid=='f1' else 'obs2',fid)
    monkeypatch.setattr(store,'candidate_card_links',compute)
    return calls


def test_save_load_resume_and_review_independence(request,monkeypatch):
    db=request.getfixturevalue('panel_database');calls=setup(monkeypatch)
    before=hashlib.sha256(db.read_bytes()).hexdigest()
    assert store.load_candidate_comparison(db,'analysis','obs1','f1',{})['status']=='missing'
    assert not store.comparison_path(db,'analysis','obs1').parent.exists()
    store.check_and_save_comparison(db,'analysis','obs1','f1',{})
    assert store.load_candidate_comparison(db,'analysis','obs1','f1',{})['status']=='available'
    assert hashlib.sha256(db.read_bytes()).hexdigest()==before
    c=connect_store(db);upsert_finding_review(c,finding_id='f1',review_status='not_ok',note='User');c.close()
    assert store.load_candidate_comparison(db,'analysis','obs1','f1',{})['status']=='available'
    summary=store.build_candidate_comparisons(db,'analysis',{},workers=1)
    assert calls==['f1','f3'] and len(summary['results'])==2
    assert {v['status'] for v in summary['results']}=={'reused','available'}


def test_unavailable_source_saved_with_reason_and_not_retried_on_open(request,monkeypatch):
    db=request.getfixturevalue('panel_database');setup(monkeypatch)
    calls=[]
    def fail(*a,**k):calls.append(1);raise ValueError('No bounded source anchor')
    monkeypatch.setattr(store,'candidate_card_links',fail)
    saved=store.check_and_save_comparison(db,'analysis','obs1','f1',{})
    assert saved['status']=='unavailable' and saved['reason']=='No bounded source anchor'
    assert store.load_candidate_comparison(db,'analysis','obs1','f1',{})['status']=='unavailable'
    assert calls==[1]


@pytest.mark.parametrize('change',['binding','checksum','other_observation','run'])
def test_stale_or_mixed_records_are_withheld(request,monkeypatch,change):
    db=request.getfixturevalue('panel_database');setup(monkeypatch)
    store.check_and_save_comparison(db,'analysis','obs1','f1',{})
    if change=='binding':
        monkeypatch.setattr(store,'candidate_links_signature',lambda *a,**k:'changed-sdk-or-source')
    elif change=='checksum':
        p=store.comparison_path(db,'analysis','obs1');r=json.loads(p.read_bytes());r['report']['rows']=[];p.write_text(json.dumps(r))
    else:
        target=('analysis','obs2','f3') if change=='other_observation' else ('other-analysis','obs1','f1')
        path=store.comparison_path(db,target[0],target[1]);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(store.comparison_path(db,'analysis','obs1').read_bytes())
        assert store.load_candidate_comparison(db,*target,{})['status']=='stale'
        return
    assert store.load_candidate_comparison(db,'analysis','obs1','f1',{})['status']=='stale'


def test_change_during_matching_does_not_save(request,monkeypatch):
    db=request.getfixturevalue('panel_database');setup(monkeypatch)
    signatures=iter(('before','after'))
    monkeypatch.setattr(store,'candidate_links_signature',lambda *a,**k:next(signatures))
    with pytest.raises(ValueError,match='changed during'):
        store.check_and_save_comparison(db,'analysis','obs1','f1',{})
    assert not store.comparison_path(db,'analysis','obs1').exists()


def test_memoized_hashes_do_not_change_core_rgb_binding(request,monkeypatch):
    from pathlib import Path

    from pvdelta import rgb_repeat

    db=request.getfixturevalue('panel_database')
    c=connect_store(db)
    c.execute('CREATE TABLE flight_screening_frames (analysis_run_id VARCHAR,capture_id VARCHAR,diagnostics_json JSON)')
    c.close()
    # The fixture has original source bytes but no geometry/profile; snapshot
    # permits those unavailable fields and binds them identically in both paths.
    rows=[{'capture_id':'t1','thermal_path':str(db.parent/'t1.JPG'),'rgb_path':str(db.parent/'r1.JPG'),
        'source_root':str(db.parent),'gimbal_pitch_deg':-90}]
    from pvdelta import repeat_candidates
    monkeypatch.setattr(repeat_candidates,'select_repeat_pairs',lambda *a,**k:[dict(r) for r in rows])
    original=rgb_repeat._snapshot(db,'f1',None,4)[1]
    hashes={}
    def cached(path):
        path=Path(path)
        if path not in hashes:hashes[path]=rgb_repeat._file_hash(path)
        return hashes[path]
    assert rgb_repeat._snapshot(db,'f1',None,4,hash_file=cached)[1]==original
