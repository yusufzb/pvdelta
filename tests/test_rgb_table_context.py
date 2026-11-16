from copy import deepcopy

import numpy as np
import pytest

from pvdelta import repeat_views, rgb_repeat


def module(x, y=20, table=0):
    return {'polygon_rgb_xy': [[x, y], [x+40, y], [x+40, y+90], [x, y+90]],
            'table_index': table}


def test_support_is_one_hop_in_same_observed_table_and_does_not_invent_gap_pixels():
    selected=module(60)
    modules=[selected,module(20),module(100),module(140),module(60,115,table=1),module(60,220)]
    original=deepcopy(modules)
    support=rgb_repeat._table_support(selected['polygon_rgb_xy'],modules)
    assert support==[modules[i]['polygon_rgb_xy'] for i in (0,1,2)]
    mask=repeat_views._tracking_mask(np.asarray(support[0]),support,(340,220),1.)
    assert mask[40,40]==mask[40,80]==mask[40,120]==255
    assert mask[40,10]==mask[40,150]==mask[120,80]==0  # ground / non-neighbour / other table
    assert modules==original
    support[0][0][0]=-999
    assert modules==original


@pytest.mark.parametrize('table',[None,-1,True,'0'])
def test_missing_or_invalid_table_identity_keeps_only_the_selected_panel(table):
    selected=module(60,table=table)
    assert rgb_repeat._table_support(selected['polygon_rgb_xy'],[selected,module(100,table=table)])==[
        selected['polygon_rgb_xy']]


def test_overlapping_shapes_are_not_admitted_as_context():
    selected=module(60)
    assert rgb_repeat._table_support(selected['polygon_rgb_xy'],[selected,module(61)])==[
        selected['polygon_rgb_xy']]


@pytest.mark.parametrize('problem',['missing_selected','duplicate','outside','nonconvex','too_many','overlap'])
def test_invalid_support_is_rejected_before_tracking(problem):
    q=np.asarray(module(60)['polygon_rgb_xy'])
    support=[q.tolist(),module(100)['polygon_rgb_xy']]
    if problem=='missing_selected':support=support[1:]
    elif problem=='duplicate':support.append(q.tolist())
    elif problem=='outside':support[1][0][0]=-1
    elif problem=='nonconvex':support[1]=[[100,20],[140,110],[140,20],[100,110]]
    elif problem=='too_many':support=[q.tolist()]*6
    else:support.append(module(101)['polygon_rgb_xy'])
    with pytest.raises(ValueError):repeat_views._tracking_mask(q,support,(160,240),1.)


def test_context_is_forwarded_independently_in_both_directions_with_normalized_corner_origin(monkeypatch):
    source=[module(60),module(100),module(60,115,table=1)]
    target=[module(70),module(110),module(70,115,table=1)]
    q=np.roll(np.asarray(source[0]['polygon_rgb_xy']),1,axis=0)
    calls=[]
    def match(a,b,polygon,competitors,**kwargs):
        calls.append((polygon.copy(),[p.copy() for p in competitors],deepcopy(kwargs)))
        return repeat_views.PanelMatch('matched_provisional','local checks passed',target_index=0,
            iou=.99,check_p95_px=.1,check_count=8,fit_inliers=24)
    monkeypatch.setattr(repeat_views,'match_panel',match)
    original=deepcopy((source,target))
    report=rgb_repeat._reciprocal_match('source','target',q,source,target,table_context=True)
    assert report['reciprocal_check_passed'] and len(calls)==2
    assert len(calls[0][2]['source_support_polygons'])==len(calls[1][2]['source_support_polygons'])==2
    np.testing.assert_array_equal(calls[1][1][0],q)
    assert report['tracking_support_mode']=='observed_table_neighbours'
    assert (source,target)==original


def test_context_cannot_bypass_a_failed_heldout_or_reverse_gate(monkeypatch):
    source=[module(60),module(100)]
    target=[module(70),module(110)]
    outcomes=iter((repeat_views.PanelMatch('matched_provisional','forward',target_index=0),
        repeat_views.PanelMatch('unavailable','held-out RGB correspondence residual too large')))
    monkeypatch.setattr(repeat_views,'match_panel',lambda *a,**k:next(outcomes))
    report=rgb_repeat._reciprocal_match('source','target',source[0]['polygon_rgb_xy'],source,target,
        table_context=True)
    assert not report['reciprocal_check_passed'] and report['target_index'] is None
    assert 'Reverse RGB check failed' in report['reason']


@pytest.mark.parametrize('value',[1,None,'yes'])
def test_context_flag_is_explicit_boolean(value):
    with pytest.raises(ValueError,match='boolean'):
        rgb_repeat.observed_rgb_repeat('db','finding',{},table_context=value)


def test_ui_mode_change_invalidates_previous_evidence_and_requires_explicit_compute(tmp_path,monkeypatch):
    import hashlib

    from PIL import Image
    from streamlit.testing.v1 import AppTest

    from pvdelta import rgb_repeat_ui

    source={'capture_id':'source','status':'source','rgb_polygon':module(60)['polygon_rgb_xy']}
    target={'capture_id':'target','status':'matched_provisional','reason':'checked',
        'rgb_polygon':module(60)['polygon_rgb_xy'],'reciprocal_check_passed':True}
    for row in (source,target):
        for kind in ('rgb','thermal'):
            path=tmp_path/f'{row["capture_id"]}-{kind}.png'
            Image.new('RGB',(240,160),'#404040').save(path)
            row[kind+'_path']=str(path)
            row[kind+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    calls=[]
    def signature(db,fid,reg,**kwargs):return str(kwargs.get('table_context',False))
    def compute(db,fid,reg,**kwargs):
        enabled=kwargs.get('table_context',False);calls.append(enabled)
        report={'schema':'pvdelta.observed-rgb-repeat/v1','binding':str(enabled),
            'source':source,'source_rgb_polygon':source['rgb_polygon'],'rows':[target],
            'tracking_support_mode':'observed_table_neighbours' if enabled else 'selected_panel_only'}
        report['content_sha256']=rgb_repeat.digest(report)
        return report
    monkeypatch.setattr(rgb_repeat_ui,'rgb_repeat_signature',signature)
    monkeypatch.setattr(rgb_repeat_ui,'observed_rgb_repeat',compute)
    script=tmp_path/'app.py'
    script.write_text('from pvdelta.rgb_repeat_ui import observed_rgb_review\n'
        'observed_rgb_review("db","finding",{},"sdk",key="test")\n',encoding='utf-8')
    app=AppTest.from_file(str(script),default_timeout=30).run()
    assert not app.exception and not calls
    app.button(key='test-rgb-find').click().run()
    assert not app.exception and calls==[False] and len(app.get('image'))==2
    app.checkbox(key='test-rgb-table-context').check().run()
    assert not app.exception and calls==[False] and not app.get('image')
    assert 'test-rgb-result' not in app.session_state
    app.button(key='test-rgb-find').click().run()
    assert not app.exception and calls==[False,True] and len(app.get('image'))==2
    app.button(key='test-rgb-frame-finding-next').click().run()
    assert not app.exception and app.selectbox(key='test-rgb-frame-finding').value=='target'
