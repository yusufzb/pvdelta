import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pytest

cv2=pytest.importorskip('cv2')
pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest

from pvdelta.local_rgb_transfer import _context
from pvdelta.local_rgb_transfer_ui import _prepare
from pvdelta.models import CameraProfile, ThermalFrame


@pytest.fixture
def inputs(tmp_path,monkeypatch):
    thermal=tmp_path/'original_T.JPG';thermal.write_bytes(b'original source bytes')
    rgb=tmp_path/'original_V.JPG'
    rgb.write_bytes(cv2.imencode('.png',np.full((440,520,3),120,np.uint8))[1].tobytes())
    values=np.linspace(20,35,220*260,dtype=np.float32).reshape(220,260)
    frame=ThermalFrame('frame',str(thermal),hashlib.sha256(thermal.read_bytes()).hexdigest(),values,None,260,220)
    @dataclass
    class Params:
        emissivity:float=.95
    class Reader:
        def __init__(self,*args,**kwargs):self.measurement_params=Params()
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def temperature(self):return values.copy()
        def processed_rgb(self):return np.full((220,260,3),120,np.uint8)
    monkeypatch.setattr('pvdelta.decode.RJpeg',Reader)
    monkeypatch.setattr('pvdelta.metadata.parse_camera_profile',lambda path:CameraProfile(gimbal_pitch=-90))
    def extract(*args,**kwargs):
        assert kwargs['gimbal_pitch_deg']==-90
        return {'status':'unreviewed_rgb_geometry','modules':[{'polygon_rgb_xy':[[192,178],[272,178],[272,258],[192,258]]}]}
    monkeypatch.setattr('pvdelta.table_boundaries.refined_module_proposals',extract)
    controls={'schema':'pvdelta.local-rgb-controls/v1','pair':_context(frame,rgb),
        'provenance':{'kind':'synthetic_controls','note':'Known unit-test correspondences'},
        'points':[{'control_id':f'{i}-{j}','native_xy':[x,y],'rgb_xy':[2*x+12,2*y+18],
                   'role':'fit' if (i+j)%2==0 else 'check'} for i,x in enumerate((30,70,110,150,190))
                  for j,y in enumerate((30,80,130,180))]}
    path=tmp_path/'controls.json';path.write_text(json.dumps({'frames':[controls]}))
    report={'source_sha256':frame.source_hash,'native_temperature_sha256':hashlib.sha256(values.tobytes()).hexdigest(),
            'native_shape':[220,260],'radiometric_parameters':{'emissivity':.95}}
    script=('from pvdelta.local_rgb_transfer_ui import local_rgb_transfer_view\n'
            f'local_rgb_transfer_view({report!r},{str(thermal)!r},None,key="test")\n')
    return AppTest.from_string(script,default_timeout=15),report,thermal,rgb,path,controls


def element(items,label):return next(e for e in items if e.label==label)


def preview(app,path):
    app.run()
    element(app.text_input,'Local pair controls JSON or bundle').input(str(path)).run()
    element(app.button,'Preview local RGB boundaries').click().run()


def test_preview_reads_original_native_and_shows_same_rgb_ids_without_db_write(inputs):
    app,_report,_thermal,_rgb,path,_controls=inputs
    preview(app,path)
    assert not app.exception and not app.error
    result=app.session_state['test-result'][1][0]
    assert result['transfer']['supported_proposals']==1
    assert result['native_version'] and not result['database_written']
    assert result['native_version']['source']['measurement_parameters']=={'emissivity':.95}
    assert 'local_rgb_transfer.py' in result['preview_engine_sources']


def test_tolerance_change_invalidates_preview_and_rechecks_fixed_controls(inputs):
    app,_report,_thermal,_rgb,path,controls=inputs
    for p in controls['points']:
        if p['role']=='check':p['rgb_xy'][0]+=4.2
    path.write_text(json.dumps(controls))
    preview(app,path)
    assert not app.exception
    assert not app.session_state['test-result'][1][0]['registration']['geometry_gate_passed']
    element(app.selectbox,'Development geometry tolerance (native px)').select(2.20).run()
    assert 'test-result' not in app.session_state
    element(app.button,'Preview local RGB boundaries').click().run()
    assert not app.exception and not app.error
    payload=app.session_state['test-result'][1][0]
    assert payload['registration']['geometry_gate_passed']
    assert payload['registration']['point_gate_policy']['max_check_p95_native_px']==2.20
    assert payload['transfer']['supported_proposals']==1
    assert payload['native_version']['settings']['margin_px']==3
    assert not payload['main_scan_updated'] and not payload['database_written']


def test_failed_check_model_shows_rejection_and_never_measures_transferred_panels(inputs):
    app,_report,_thermal,_rgb,path,controls=inputs
    for p in controls['points']:
        if p['role']=='check':p['rgb_xy'][0]+=12
    path.write_text(json.dumps(controls))
    preview(app,path)
    assert not app.exception and not app.error and app.warning
    result=app.session_state['test-result'][1][0]
    assert result['transfer']['proposals']==[] and result['native_version'] is None


def test_reference_edit_or_missing_input_clears_stale_preview(inputs):
    app,_report,_thermal,_rgb,path,controls=inputs
    preview(app,path)
    assert 'test-result' in app.session_state
    controls['provenance']['note']='edited file'
    path.write_text(json.dumps(controls))
    app.run()
    assert 'test-result' not in app.session_state
    element(app.text_input,'Local pair controls JSON or bundle').input(str(path.parent/'missing.json')).run()
    assert app.error and not app.exception and 'test-result' not in app.session_state


def test_changed_preview_native_identity_and_ambiguous_bundle_are_rejected(inputs):
    _app,report,thermal,rgb,path,controls=inputs
    wrong=dict(report,native_temperature_sha256='0'*64)
    with pytest.raises(ValueError,match='changed after'):_prepare(wrong,thermal,rgb,None,path.read_bytes())
    with pytest.raises(ValueError,match='exactly one'):
        _prepare(report,thermal,rgb,None,json.dumps({'frames':[controls,controls]}).encode())


def calibration_for(controls):
    return {'schema':'pvdelta.rgb-brown-calibration/v1','rgb_source_sha256':controls['pair']['rgb_sha256'],
        'camera':{'projection_type':'brown','width':520,'height':440,'focal_x':.8,'focal_y':.8,'c_x':0.,'c_y':0.,
                 'k1':0.,'k2':0.,'k3':0.,'p1':0.,'p2':0.},'provenance':{'note':'Fixed synthetic zero-distortion calibration'}}


def test_optional_calibration_is_explicit_and_changes_invalidate_cached_preview(inputs):
    app,_report,_thermal,_rgb,path,controls=inputs
    calibration=calibration_for(controls)
    lens=path.parent/'calibration.json';lens.write_text(json.dumps(calibration))
    app.run()
    element(app.text_input,'Local pair controls JSON or bundle').input(str(path))
    element(app.text_input,'RGB Brown calibration JSON (optional)').input(str(lens)).run()
    element(app.button,'Preview local RGB boundaries').click().run()
    assert not app.error and not app.exception
    payload=app.session_state['test-result'][1][0]
    assert payload['registration']['rgb_geometry']['model']=='brown_undistorted_rgb'
    assert payload['transfer']['supported_proposals']==1
    calibration['provenance']['note']='changed calibration identity'
    lens.write_text(json.dumps(calibration));app.run()
    assert 'test-result' not in app.session_state


def test_wrong_source_calibration_fails_without_falling_back_to_pinhole(inputs):
    app,_report,_thermal,_rgb,path,controls=inputs
    calibration=calibration_for(controls);calibration['rgb_source_sha256']='0'*64
    lens=path.parent/'wrong-lens.json';lens.write_text(json.dumps(calibration))
    app.run()
    element(app.text_input,'Local pair controls JSON or bundle').input(str(path))
    element(app.text_input,'RGB Brown calibration JSON (optional)').input(str(lens)).run()
    element(app.button,'Preview local RGB boundaries').click().run()
    assert app.error and not app.exception and 'test-result' not in app.session_state
