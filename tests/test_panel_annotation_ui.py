import io
import json

import pytest

pytest.importorskip('streamlit')
pytest.importorskip('PIL')
from PIL import Image
from streamlit.testing.v1 import AppTest

from pvdelta.panel_annotation import load_annotation_pack


@pytest.fixture
def setup_editor(tmp_path):
    Image.new('RGB',(80,60),'gray').save(tmp_path/'native.png')
    path = tmp_path/'pack.json'
    path.write_text(json.dumps({'schema':'pvdelta.pending-panel-reference-pack/v1', 'frames':[
        {'schema':'pvdelta.native-panel-instances/v1', 'case_id':'test', 'source_file':'test_T.JPG',
         'source_sha256':'a'*64, 'native_temperature_sha256':'b'*64, 'native_height':60, 'native_width':80,
         'coordinate_space':'thermal_native', 'coverage':'unresolved', 'review_status':'unreviewed',
         'panels':None, 'image':'native.png'}]}))
    script = tmp_path/'app.py'
    script.write_text('from pvdelta.panel_annotation_ui import panel_annotation_view\n'
        f'panel_annotation_view({str(path)!r},output={str(tmp_path/"saved")!r})\n',encoding='utf-8')
    return path, script


def test_open_editor_rerun_and_bad_pack_errors_do_not_crash(setup_editor):
    _, script = setup_editor
    app = AppTest.from_file(str(script)).run()
    assert not app.exception and len(app.get('iframe')) == 0
    app.checkbox(key='panel-reference-editor-open').check().run()
    assert not app.exception and len(app.get('iframe')) == 1
    app.text_input(key='panel-reference-editor-pack').set_value('missing-pack.json').run()
    assert not app.exception and any('Cannot load annotation pack' in e.value for e in app.error)


def test_uploaded_draft_saves_a_version_and_survives_rerun(setup_editor, monkeypatch):
    import streamlit as st

    path, script = setup_editor
    pack = load_annotation_pack(path)
    f = pack['frames'][0]
    document = {'schema':'pvdelta.panel-annotation-work/v1', 'pack_token':pack['pack_token'], 'frames':[
        {**{k:f[k] for k in ('source_sha256','native_temperature_sha256','native_width','native_height',
                            'display_png_sha256')}, 'panels':[], 'ignore':[], 'exhaustive':False,
         'reviewer':'', 'note':'', 'empty_reason':''}]}
    uploaded = io.BytesIO(json.dumps(document).encode())
    monkeypatch.setattr(st,'file_uploader',lambda *a,**k:uploaded)
    app = AppTest.from_file(str(script)).run()
    assert not app.exception
    app.button(key='panel-reference-editor-save').click().run()
    assert not app.exception and any('0/1 completed' in s.value for s in app.success)
    files = list((path.parent/'saved').glob('*.json'))
    assert len(files) == 1 and json.loads(files[0].read_text())['frames'][0]['panels'] is None
    app.run()
    assert not app.exception and any('0/1 completed' in s.value for s in app.success)
    assert len(list((path.parent/'saved').glob('*.json'))) == 1
    document['pack_token'] = 'stale'
    monkeypatch.setattr(st,'file_uploader',lambda *a,**k:io.BytesIO(json.dumps(document).encode()))
    app.run()
    assert not app.exception and any('validation failed' in e.value for e in app.error)
    assert not app.success and len(list((path.parent/'saved').glob('*.json'))) == 1
