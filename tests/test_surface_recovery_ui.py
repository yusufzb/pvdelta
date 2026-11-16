import json

import pytest

pytest.importorskip('streamlit')
from PIL import Image
from streamlit.testing.v1 import AppTest

from pvdelta import surface_recovery_ui as ui


def test_saved_results_navigation_and_scan_are_available_without_recomputing_on_next(tmp_path,monkeypatch):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    root=tmp_path/'runs';root.mkdir();(root/'latest.json').write_text('{}')
    rows=[]
    for case in ('C001','C002'):
        rows.append({'case_id':case,'source_file':case+'.JPG','status':'completed',
                     'observation_count':3,'added_observations':1,'candidate_regions':0})
        (root/f'{case}.json').write_text(json.dumps({'source_file':case+'.JPG',
            'module_reports':[],'measurement_profile':{}}))
        for suffix in ('before','after'):
            Image.new('RGB',(20,20)).save(root/f'{case}-{suffix}.png')
    summary={'run_id':'saved-test','engine_source_sha256':{},'failed_frames':0,'frames':rows}
    monkeypatch.setattr(ui,'load_surface_manifest',lambda p:({'frames':rows,'flight_thermal_frames':20},'hash'))
    monkeypatch.setattr(ui,'available_surface_runs',lambda *a:['saved-test'])
    monkeypatch.setattr(ui,'load_surface_run',lambda *a,**k:(summary,root))
    monkeypatch.setattr(ui,'recovery_engine_sources',dict)
    calls=[]
    def run(*a,progress,allow_isolated=False,exclude_declared_non_panel=False):
        calls.append((allow_isolated,exclude_declared_non_panel));progress(2,2)
        return summary,root
    monkeypatch.setattr(ui,'run_surface_recovery',run)
    script=tmp_path/'surface_app.py'
    script.write_text('from pvdelta.surface_recovery_ui import surface_recovery_view\n'
        f'surface_recovery_view({str(manifest)!r}, "sdk", output_root={str(root)!r})', encoding='utf-8')
    app=AppTest.from_file(str(script)).run()
    assert not app.exception and not calls
    assert len(app.get('image'))==2
    app.button(key='surface-recovery-frame-next').click().run()
    assert not app.exception and not calls
    assert app.selectbox(key='surface-recovery-frame').value=='C002'
    app.button(key='surface-recovery-frame-previous').click().run()
    assert not app.exception and app.selectbox(key='surface-recovery-frame').value=='C001'
    app.checkbox(key='surface-recovery-allow-isolated').check().run()
    assert not app.exception and not calls
    app.button(key='surface-recovery-run').click().run()
    assert not app.exception and calls==[(True,False)]
    app.checkbox(key='surface-recovery-require-surface').check().run()
    assert not app.exception and len(calls)==1
    app.button(key='surface-recovery-run').click().run()
    assert not app.exception and calls==[(True,False),(True,True)]


def test_failed_retry_can_be_inspected_and_previous_results_reopened(tmp_path,monkeypatch):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    root=tmp_path/'runs';root.mkdir()
    row={'case_id':'C001','source_file':'source.JPG','status':'completed',
         'observation_count':3,'added_observations':1,'candidate_regions':0}
    success={'run_id':'complete','engine_source_sha256':{},'failed_frames':0,'frames':[row]}
    failure={'run_id':'failed','engine_source_sha256':{},'failed_frames':1,
             'frames':[{'case_id':'C001','status':'failed','error':'SDK unavailable'}]}
    (root/'C001.json').write_text(json.dumps({'source_file':'source.JPG',
        'module_reports':[],'measurement_profile':{}}))
    for suffix in ('before','after'):
        Image.new('RGB',(20,20)).save(root/f'C001-{suffix}.png')
    runs=['complete'];calls=[]
    monkeypatch.setattr(ui,'load_surface_manifest',lambda p:({'frames':[row]},'hash'))
    monkeypatch.setattr(ui,'available_surface_runs',lambda *a:list(runs))
    monkeypatch.setattr(ui,'load_surface_run',lambda *a,run_id:(success if run_id=='complete' else failure,root))
    monkeypatch.setattr(ui,'recovery_engine_sources',dict)
    def run(*a,progress):
        calls.append(True);runs.append('failed');progress(1,1)
        return failure,root
    monkeypatch.setattr(ui,'run_surface_recovery',run)
    script=tmp_path/'history_app.py'
    script.write_text('from pvdelta.surface_recovery_ui import surface_recovery_view\n'
        f'surface_recovery_view({str(manifest)!r}, "sdk", output_root={str(root)!r})',encoding='utf-8')
    app=AppTest.from_file(str(script)).run()
    assert not app.exception and len(app.get('image'))==2
    app.button(key='surface-recovery-run').click().run()
    assert not app.exception and calls==[True] and not app.get('image')
    assert app.selectbox(key='surface-recovery-saved-run').value=='failed'
    assert {m.label:m.value for m in app.metric}['Failed frames']=='1'
    assert any('previous complete' in w.value for w in app.warning)
    app.selectbox(key='surface-recovery-saved-run').set_value('complete').run()
    assert not app.exception and len(app.get('image'))==2 and calls==[True]
    app.run()
    assert not app.exception and app.selectbox(key='surface-recovery-saved-run').value=='complete'
    reopened=AppTest.from_file(str(script)).run()
    assert not reopened.exception and reopened.selectbox(key='surface-recovery-saved-run').value=='complete'
