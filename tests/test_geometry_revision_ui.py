import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip('streamlit')
pytest.importorskip('PIL')
from streamlit.testing.v1 import AppTest

from pvdelta.geometry_revision import load_geometry_version
from pvdelta.geometry_revision_ui import _decode_revision


@pytest.fixture
def ui(tmp_path,monkeypatch):
    source=tmp_path/'source_T.JPG'
    source.write_bytes(b'bound synthetic source')
    y,x=np.mgrid[:90,:140]
    values=(25+x*.02+y*.005).astype(np.float32)
    values[30:35,40:44]+=8
    @dataclass
    class Parameters:
        emissivity: float = .95
    class Reader:
        def __init__(self,*args,**kwargs):self.measurement_params=Parameters()
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def temperature(self):return values.copy()
        def processed_rgb(self):return np.full((*values.shape,3),128,np.uint8)
    monkeypatch.setattr('pvdelta.decode.RJpeg',Reader)
    proposals=[{'proposal_id':name,'polygon_native_xy':[[left,15],[right,15],[right,70],[left,70]]}
        for name,left,right in [('A',10,50),('B',45,80)]]
    report={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'native_temperature_sha256':hashlib.sha256(values.tobytes()).hexdigest(),
        'native_shape':list(values.shape),'radiometric_parameters':{'emissivity':.95},
        'proposed':{'proposals':proposals}}
    output=tmp_path/'versions'
    script=('from pvdelta.geometry_revision_ui import geometry_revision_view\n'
        f'geometry_revision_view({report!r}, {str(source)!r}, None, output={str(output)!r})\n')
    return AppTest.from_string(script,default_timeout=15),report,source,output


def element(items,label):
    return next(item for item in items if item.label==label)


def session(app):
    key=next(k for k in app.session_state if k.endswith('-session'))
    return app.session_state[key]


def start(app):
    app.run()
    element(app.button,'Start geometry revision session').click().run()
    assert not app.exception


def remove(app,identifier):
    element(app.selectbox,'Geometry operation').set_value('Remove region').run()
    element(app.selectbox,'Region ID').set_value(identifier).run()
    element(app.text_input,'Reason for this geometry change').input('Reviewed source geometry')
    element(app.button,'Preview revised analysis').click().run()
    assert not app.exception and not app.error


def test_remove_undo_branch_and_save_preserve_every_prior_version(ui):
    app,_report,_source,output=ui
    start(app)
    remove(app,'B')
    state=session(app)
    removed=state['active']
    assert len(state['versions'])==2
    assert [p['proposal_id'] for p in state['versions'][removed]['geometry']['proposals']]==['A']
    element(app.button,'Return to parent (keep this version in history)').click().run()
    assert session(app)['active']==session(app)['root']
    element(app.text_area,'Four native corners [[x,y], ...]').input('[[12,16],[48,16],[48,68],[12,68]]')
    element(app.text_input,'Reason for this geometry change').input('Align frame')
    element(app.button,'Preview revised analysis').click().run()
    state=session(app)
    assert len(state['versions'])==3 and removed in state['versions']
    assert state['versions'][state['active']]['parent_version_id']==state['root']
    element(app.button,'Save revision history (local JSON)').click().run()
    assert not app.exception and not app.error
    files=list(output.glob('geometry-version-*.json'))
    assert len(files)==3
    assert all(not load_geometry_version(p)['main_scan_updated'] for p in files)


def test_empty_version_can_return_to_parent_and_download_or_save(ui):
    app,*_rest=ui
    start(app)
    remove(app,'A')
    remove(app,'B')
    state=session(app)
    assert state['versions'][state['active']]['geometry']['proposals']==[]
    assert element(app.selectbox,'Geometry operation').value=='Add region'
    element(app.button,'Return to parent (keep this version in history)').click().run()
    assert not app.exception and len(session(app)['versions'])==3
    assert len(session(app)['versions'][session(app)['active']]['geometry']['proposals'])==1


def test_invalid_edit_leaves_the_active_version_and_history_unchanged(ui):
    app,*_rest=ui
    start(app)
    before=session(app)['active']
    element(app.text_area,'Four native corners [[x,y], ...]').input('[[0,0]]')
    element(app.text_input,'Reason for this geometry change').input('Invalid shape should fail')
    element(app.button,'Preview revised analysis').click().run()
    assert not app.exception and app.error
    assert session(app)['active']==before and len(session(app)['versions'])==1


def test_fresh_decode_must_match_the_preview_native_identity_and_parameters(ui):
    _app,report,source,_output=ui
    changed=dict(report,native_temperature_sha256='0'*64)
    with pytest.raises(ValueError,match='Native radiometry'):
        _decode_revision(changed,source,None)
    changed=dict(report,radiometric_parameters={'emissivity':.8})
    with pytest.raises(ValueError,match='parameters'):
        _decode_revision(changed,source,None)


def test_reference_evaluation_invalidates_after_edit_or_reference_change(ui):
    app,report,source,_output=ui
    start(app)
    ref={'schema':'pvdelta.native-panel-instances/v1','coverage':'exhaustive_full_frame',
        'coordinate_space':'thermal_native','source_sha256':report['source_sha256'],
        'native_height':90,'native_width':140,'review_status':'developer_draft',
        'panels':[{'instance_id':'R-A','polygon_native_xy':report['proposed']['proposals'][0]['polygon_native_xy']}],
        'ignore':[]}
    path=source.parent/'reference.json'
    path.write_text(json.dumps(ref))
    element(app.text_input,'Revision panel reference JSON path').input(str(path)).run()
    element(app.button,'Evaluate active revision').click().run()
    assert not app.exception and not app.error
    def results():
        return [app.session_state[k] for k in app.session_state if k.endswith('-validation-result')]
    assert len(results())==1
    assert results()[0]['scores']['before']['0.75']['true_positives']==1
    remove(app,'B')
    assert not results()  # A previous score must not describe a different active version.
    element(app.button,'Evaluate active revision').click().run()
    assert len(results())==1 and not app.exception
    ref['annotation_method']='changed reference file'
    path.write_text(json.dumps(ref))
    app.run()
    assert not results()


def test_reference_input_errors_remain_visible_instead_of_zero_scores(ui):
    app,_report,source,_output=ui
    start(app)
    element(app.text_input,'Revision panel reference JSON path').input(str(source.parent/'missing.json')).run()
    assert app.error and not app.exception
    path=source.parent/'bad.json'
    path.write_text('not JSON')
    element(app.text_input,'Revision panel reference JSON path').input(str(path)).run()
    element(app.button,'Evaluate active revision').click().run()
    assert app.error and not app.exception
    assert not [k for k in app.session_state if k.endswith('-validation-result')]
