from copy import deepcopy

from streamlit.testing.v1 import AppTest

from pvdelta import thermal_views_ui as ui


def sample():
    return {'source_observation_id':'obs','source':{'capture_id':'source'},'source_rgb_polygon':None,
        'status':'available','reason':'','rows':[{'capture_id':cid,'thermal_path':cid+'.jpg',
            'rgb_path':cid+'-rgb.jpg','status':status,'thermal_evidence':evidence,
            'capture_candidate_count':0,'reason':'diagnostic'} for cid,status,evidence in
            [('first','matched_provisional',{'status':'analyzed_no_candidates','regions':[]}),
             ('second','unavailable',None)]]}


def script():
    return '''from pvdelta.thermal_views_ui import thermal_views_review
thermal_views_review('db','obs','f',{},'sdk',key='views')
'''


def test_all_view_scan_is_explicit_and_preserves_candidate_free_unknown_and_navigation(monkeypatch):
    calls=[];saved=[]
    monkeypatch.setattr(ui,'load_view_scan',lambda *a,**k:{'status':'missing'})
    def check(*a,**k):calls.append(k);return deepcopy(sample())
    monkeypatch.setattr(ui,'scan_thermal_views',check)
    monkeypatch.setattr(ui,'save_view_scan',lambda *a:saved.append(a[1]))
    app=AppTest.from_string(script(),default_timeout=30).run()
    assert not app.exception and not calls and not app.get('image')
    app.button(key='views-thermal-view-check').click().run()
    assert not app.exception and calls[0]['max_views']==32 and saved
    frame=app.dataframe[0].value
    assert frame['Native assessment'].tolist()==['analyzed_no_candidates','unknown']
    assert 'healthy-panel' in app.caption[0].value
    assert app.button(key='views-thermal-view-f-32-next').disabled is False


def test_persisted_views_open_without_reanalysis_and_stale_results_are_withheld(monkeypatch):
    monkeypatch.setattr(ui,'load_view_scan',lambda *a,**k:{'status':'available','report':sample()})
    def forbidden(*a,**k):raise AssertionError('Opening a saved card cannot rerun matching')
    monkeypatch.setattr(ui,'scan_thermal_views',forbidden)
    app=AppTest.from_string(script(),default_timeout=30).run()
    assert not app.exception and len(app.dataframe)==1
    app.button(key='views-thermal-view-f-32-next').click().run()
    assert not app.exception and app.selectbox(key='views-thermal-view-f-32').value=='second'
    monkeypatch.setattr(ui,'load_view_scan',lambda *a,**k:{'status':'stale','reason':'Source changed'})
    app.run()
    assert not app.exception and not app.dataframe
    assert any('Source changed' in w.value for w in app.warning)


def test_failed_check_does_not_raise_or_fabricate_healthy_results(monkeypatch):
    monkeypatch.setattr(ui,'load_view_scan',lambda *a,**k:{'status':'missing'})
    def fail(*a,**k):raise ValueError('SDK unavailable')
    monkeypatch.setattr(ui,'scan_thermal_views',fail)
    app=AppTest.from_string(script(),default_timeout=30).run()
    app.button(key='views-thermal-view-check').click().run()
    assert not app.exception and not app.dataframe and any('SDK unavailable' in w.value for w in app.warning)


def test_legacy_unavailable_replay_still_opens_nearby_photo_pairs(monkeypatch):
    from PIL import Image

    report = sample()
    report['status'] = 'unavailable'
    report['reason'] = 'This legacy analysis has no saved native measurement profile.'
    report['source'].update(thermal_path='source.jpg', rgb_path='source-rgb.jpg')
    for row in report['rows']:
        row.update(status='not_attempted', thermal_evidence=None, rgb_polygon=None)
    monkeypatch.setattr(ui, 'load_view_scan', lambda *a, **k:{'status':'available','report':report})
    calls = []
    def images(report, capture_id):
        calls.append(capture_id)
        return Image.new('RGB',(64,64)), Image.new('RGB',(64,64))
    monkeypatch.setattr(ui.rgb_repeat, 'repeat_pair_images', images)
    monkeypatch.setattr(ui, 'thermal_view_images', lambda *a, **k:(_ for _ in ()).throw(
        AssertionError('Legacy browsing cannot authorize native evidence')))
    app = AppTest.from_string(script(),default_timeout=30).run()
    assert not app.exception and not app.warning
    assert app.checkbox(key='views-thermal-view-show').value is True
    assert calls == ['source','first']
    assert any('Automatic thermal comparison unavailable' in item.value for item in app.info)
    assert app.dataframe[0].value['Native assessment'].tolist() == ['unknown','unknown']
    app.button(key='views-thermal-view-f-32-next').click().run()
    assert not app.exception and calls[-2:] == ['source','second']
