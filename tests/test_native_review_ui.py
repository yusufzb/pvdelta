import hashlib
import inspect
import json
from pathlib import Path

import pytest

pytest.importorskip('streamlit')
pytest.importorskip('PIL')
from PIL import Image
from streamlit.testing.v1 import AppTest

from pvdelta import native_review_ui
from pvdelta.native_review import NativeBoundaryPreview, boundary_preview_token


def boundary_app():
    from pvdelta.native_review_ui import native_boundary_review

    native_boundary_review('unused.duckdb','local-sdk',key='test-boundaries')


@pytest.fixture
def review_app(tmp_path,monkeypatch):
    rows=[]
    for index,flight in enumerate(['flight-A','flight-A','flight-B']):
        path=tmp_path/f'{index}_T.JPG';path.write_bytes(f'source-{index}'.encode())
        rows.append({'capture_id':str(index),'source_path':str(path),'source_hash':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'run_id':flight,'captured_at':'2026-10-19 12:00:00'})
    monkeypatch.setattr(native_review_ui,'indexed_native_frames',lambda database:rows)
    engine={'native_windows.py':'3'*64}
    monkeypatch.setattr(native_review_ui,'boundary_engine_sources',lambda:engine)
    calls=[]
    def compute(path,sdk_library,*,expected_sha256=None):
        calls.append(Path(path).name)
        proposal={'proposal_id':'N001','polygon_native_xy':[[5,5],[20,5],[20,40],[5,40]]}
        report={'schema':'pvdelta.native-boundary-preview/v1',
            'preview_token':boundary_preview_token(path,sdk_library,expected_sha256=expected_sha256),
            'source_name':Path(path).name,'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'native_temperature_sha256':'1'*64,'native_shape':[51,64],
            'engine_source_sha256':engine,
            'baseline':{'proposals':[proposal]},'proposed':{'proposals':[proposal,dict(proposal,proposal_id='N002')]},
            'semantic_panel_membership':'unverified','measurement_authorised':False,
            'database_written':False,'anomaly_rerun':False,
            'radiometric_parameters':{'emissivity':.95},'native_values_unchanged':True,
            'baseline_proposals':1,'proposals':2,'raw_additions':1,'empty_interiors':0,
            'lost_baseline_proposal_ids':[],'old_pixels_lost_outside_shared_envelopes':0,
            'unique_interior_pixels':250,'shared_envelope_pixels':10,'old_interior_pixels_excluded':5,
            'descriptive_statistics':[{'proposal_id':'N001','max_c':32.}]}
        picture=Image.new('RGB',(64,51),'gray')
        return NativeBoundaryPreview(report,picture,picture,picture)
    monkeypatch.setattr(native_review_ui,'native_boundary_preview',compute)
    script=tmp_path/'boundary_app.py'
    script.write_text(inspect.getsource(boundary_app)+'\nboundary_app()\n',encoding='utf-8')
    app=AppTest.from_file(str(script)).run()
    return app,rows,calls


def test_compute_is_on_demand_and_support_toggle_does_not_decode_again(review_app):
    app,_,calls=review_app
    assert not app.exception and not calls and not app.get('image')
    app.button(key='test-boundaries-compare').click().run()
    assert not app.exception and calls==['0_T.JPG']
    assert len(app.get('image'))==2
    assert next(m.value for m in app.metric if m.label=='Supplemental candidates')=='1'
    app.checkbox(key='test-boundaries-show-support').check().run()
    assert not app.exception and calls==['0_T.JPG'] and len(app.get('image'))==3
    assert len(app.get('download_button'))==3


def test_next_previous_and_acquisition_change_hide_stale_results(review_app):
    app,_,calls=review_app
    app.button(key='test-boundaries-compare').click().run()
    app.button(key='test-boundaries-frame-next').click().run()
    assert not app.exception
    assert app.selectbox(key='test-boundaries-frame').value=='1'
    assert not app.get('image') and calls==['0_T.JPG']
    app.button(key='test-boundaries-compare').click().run()
    assert calls==['0_T.JPG','1_T.JPG']
    app.button(key='test-boundaries-frame-previous').click().run()
    assert not app.exception and not app.get('image')
    app.selectbox(key='test-boundaries-flight').select('flight-B').run()
    assert not app.exception and app.selectbox(key='test-boundaries-frame').value=='2'
    assert app.button(key='test-boundaries-frame-previous').disabled
    assert app.button(key='test-boundaries-frame-next').disabled


def test_replaced_source_cannot_keep_overlay_for_old_index_hash(review_app):
    app,rows,calls=review_app
    app.button(key='test-boundaries-compare').click().run()
    Path(rows[0]['source_path']).write_bytes(b'replaced-source')
    app.run()
    assert not app.exception and not app.get('image')
    assert any('SHA256 differs' in error.value for error in app.error)
    assert calls==['0_T.JPG']


def test_sdk_error_clears_previous_preview_instead_of_showing_old_boxes(review_app,monkeypatch):
    app,_,calls=review_app
    app.button(key='test-boundaries-compare').click().run()
    def failed(*args,**kwargs):raise RuntimeError('libdirp unavailable')
    monkeypatch.setattr(native_review_ui,'native_boundary_preview',failed)
    app.text_input(key='test-boundaries-sdk').set_value('missing-sdk').run()
    assert not app.get('image')
    app.button(key='test-boundaries-compare').click().run()
    assert not app.exception and not app.get('image')
    assert any('libdirp unavailable' in error.value for error in app.error)
    assert calls==['0_T.JPG']


def test_local_path_mode_does_not_reuse_indexed_frame_results(review_app):
    app,rows,calls=review_app
    app.button(key='test-boundaries-compare').click().run()
    app.radio(key='test-boundaries-source-mode').set_value('Local thermal file').run()
    assert not app.exception and not app.get('image')
    app.text_input(key='test-boundaries-path').set_value(rows[1]['source_path']).run()
    assert not app.get('image')
    app.button(key='test-boundaries-compare').click().run()
    assert not app.exception and calls==['0_T.JPG','1_T.JPG']


def test_missing_source_root_hides_old_overlay_and_keeps_local_file_alternative(review_app):
    app,rows,calls=review_app
    app.button(key='test-boundaries-compare').click().run()
    rows[0]['source_name']='0_T.JPG';rows[0]['source_path']=None
    app.run()
    assert not app.exception and not app.get('image')
    assert any('source root unavailable' in item.value for item in app.info)
    assert calls==['0_T.JPG']


def test_decision_persists_and_needs_review_filter_advances_without_copying_approval(review_app,tmp_path):
    from pvdelta.boundary_review_store import latest_boundary_reviews

    app,rows,_=review_app;database=tmp_path/'review.duckdb'
    app.text_input(key='test-boundaries-database').set_value(str(database)).run()
    app.selectbox(key='test-boundaries-review-filter').select('Needs review').run()
    app.button(key='test-boundaries-compare').click().run()
    app.button(key='test-boundaries-decision-ok').click().run()
    assert not app.exception
    saved=latest_boundary_reviews(database)
    assert saved[rows[0]['source_hash']]['review_status']=='ok'
    assert rows[1]['source_hash'] not in saved
    assert app.selectbox(key='test-boundaries-frame').value=='1'
    assert not app.get('image')
    app.selectbox(key='test-boundaries-review-filter').select('Saved OK').run()
    assert app.selectbox(key='test-boundaries-frame').value=='0'
    app.button(key='test-boundaries-compare').click().run()
    assert not app.exception and any('This comparison: OK' in item.value for item in app.markdown)


def test_not_ok_then_wait_preserves_both_notes_and_history(review_app,tmp_path):
    from pvdelta.boundary_review_store import boundary_review_history

    app,rows,_=review_app;database=tmp_path/'review.duckdb'
    app.text_input(key='test-boundaries-database').set_value(str(database)).run()
    app.button(key='test-boundaries-compare').click().run()
    note=next(t for t in app.text_input if t.label=='Boundary note (optional)')
    note.set_value('Missing panel at left').run()
    app.button(key='test-boundaries-decision-not_ok').click().run()
    assert not app.exception
    note=next(t for t in app.text_input if t.label=='Boundary note (optional)')
    note.set_value('Need another view').run()
    app.button(key='test-boundaries-decision-wait').click().run()
    assert not app.exception
    history=boundary_review_history(database,rows[0]['source_hash'])
    assert [(h['review_status'],h['note']) for h in history]==[
        ('wait','Need another view'),('not_ok','Missing panel at left')]
    app.selectbox(key='test-boundaries-review-filter').select('Saved Wait').run()
    assert app.selectbox(key='test-boundaries-frame').value=='0'


def test_older_engine_ok_is_visible_as_other_version_and_does_not_approve_current_preview(review_app,tmp_path):
    from pvdelta.boundary_review_store import save_boundary_review

    app,rows,_=review_app;database=tmp_path/'review.duckdb'
    app.text_input(key='test-boundaries-database').set_value(str(database)).run()
    old=native_review_ui.native_boundary_preview(Path(rows[0]['source_path']),'local-sdk')
    old.report['engine_source_sha256']={'native_windows.py':'4'*64}
    save_boundary_review(database,old.report,'ok')
    app.run()
    app.selectbox(key='test-boundaries-review-filter').select('Saved OK').run()
    assert not app.exception and not app.get('image')
    assert any('No frames match' in item.value for item in app.info)
    app.selectbox(key='test-boundaries-review-filter').select('Other engine version').run()
    app.button(key='test-boundaries-compare').click().run()
    assert not app.exception
    assert any('This comparison: No saved decision' in item.value for item in app.markdown)


def test_reason_filter_and_ok_clear_active_tags_without_losing_history(review_app,tmp_path):
    from pvdelta.boundary_review_store import boundary_review_history

    app,rows,calls=review_app;database=tmp_path/'tags.duckdb'
    app.text_input(key='test-boundaries-database').set_value(str(database)).run()
    app.button(key='test-boundaries-compare').click().run()
    app.multiselect[0].set_value(['missing_panel','non_panel_region']).run()
    app.button(key='test-boundaries-decision-not_ok').click().run()
    assert not app.exception
    assert set(app.multiselect[0].value)=={'missing_panel','non_panel_region'}
    app.selectbox(key='test-boundaries-issue-filter').select('missing_panel').run()
    assert app.selectbox(key='test-boundaries-frame').value=='0'
    assert app.button(key='test-boundaries-frame-next').disabled
    assert calls==['0_T.JPG']
    app.button(key='test-boundaries-decision-ok').click().run()
    assert not app.exception and not app.get('image')
    assert any('No frames match' in item.value for item in app.info)
    history=boundary_review_history(database,rows[0]['source_hash'])
    assert history[0]['review_status']=='ok' and history[0]['reason_codes']==[]
    assert set(history[1]['reason_codes'])=={'missing_panel','non_panel_region'}
    app.selectbox(key='test-boundaries-issue-filter').select('All issues').run()
    app.button(key='test-boundaries-compare').click().run()
    assert not app.exception and app.multiselect[0].value==[]


def test_reason_selection_is_not_copied_to_next_frame(review_app,tmp_path):
    from pvdelta.boundary_review_store import latest_boundary_reviews

    app,rows,_=review_app;database=tmp_path/'tags.duckdb'
    app.text_input(key='test-boundaries-database').set_value(str(database)).run()
    app.button(key='test-boundaries-compare').click().run()
    app.multiselect[0].set_value(['missing_panel']).run()
    app.button(key='test-boundaries-decision-wait').click().run()
    app.button(key='test-boundaries-frame-next').click().run()
    app.button(key='test-boundaries-compare').click().run()
    assert not app.exception and app.multiselect[0].value==[]
    saved=latest_boundary_reviews(database)
    assert saved[rows[0]['source_hash']]['reason_codes']==['missing_panel']
    assert rows[1]['source_hash'] not in saved


def _reference_file(tmp_path, source_hash):
    path=tmp_path/'reference.json'
    ref={'schema':'pvdelta.native-panel-instances/v1','coverage':'exhaustive_full_frame',
         'coordinate_space':'thermal_native','source_sha256':source_hash,
         'native_height':51,'native_width':64,'review_status':'developer_draft',
         'panels':[{'instance_id':'A','polygon_native_xy':[[5,5],[20,5],[20,40],[5,40]]}]}
    path.write_text(json.dumps(ref))
    return path,ref


def test_reference_scoring_is_on_demand_without_sdk_rerun_and_invalidates_changed_labels(review_app,tmp_path):
    app,rows,calls=review_app;path,ref=_reference_file(tmp_path,rows[0]['source_hash'])
    app.button(key='test-boundaries-compare').click().run()
    app.text_input(key='test-boundaries-reference-path').set_value(str(path)).run()
    assert len(app.get('image'))==2
    app.button(key='test-boundaries-validate-reference').click().run()
    assert not app.exception and len(app.get('image'))==4 and calls==['0_T.JPG']
    assert any('Declared reference status: developer_draft' in t.value for t in app.markdown)
    ref['panels']=[];path.write_text(json.dumps(ref))
    app.run()
    assert not app.exception and len(app.get('image'))==2
    app.button(key='test-boundaries-validate-reference').click().run()
    assert len(app.get('image'))==4 and calls==['0_T.JPG']
    path.write_bytes(b'broken reference')
    app.button(key='test-boundaries-validate-reference').click().run()
    assert not app.exception and len(app.get('image'))==2
    assert any('valid JSON' in e.value for e in app.error)


def test_reference_cannot_be_reused_for_next_source_or_missing_file(review_app,tmp_path):
    app,rows,calls=review_app;path,_=_reference_file(tmp_path,rows[0]['source_hash'])
    app.button(key='test-boundaries-compare').click().run()
    app.text_input(key='test-boundaries-reference-path').set_value(str(path)).run()
    app.button(key='test-boundaries-validate-reference').click().run()
    app.button(key='test-boundaries-frame-next').click().run()
    assert not app.get('image')
    app.button(key='test-boundaries-compare').click().run()
    assert not app.exception and len(app.get('image'))==2
    app.button(key='test-boundaries-validate-reference').click().run()
    assert not app.exception and len(app.get('image'))==2
    assert any('SHA256' in e.value for e in app.error)
    app.text_input(key='test-boundaries-reference-path').set_value(str(tmp_path/'missing.json')).run()
    assert not app.exception and len(app.get('image'))==2
    assert calls==['0_T.JPG','1_T.JPG']
