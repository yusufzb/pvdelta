import hashlib
import json
import os

import numpy as np
import pytest

pytest.importorskip('PIL')

from pvdelta import native_review
from pvdelta.decode import MeasurementParams


def fake_engine(monkeypatch, *, bad_shape=False, nonfinite=False, failing=False):
    values=np.full((60,80),25.,np.float32)
    values[20,15]=87.
    values[20,27]=999.  # Shared ownership must exclude this sample from both ROIs.
    if nonfinite:values[0,0]=np.nan
    calls=[]
    class SDK:
        def __init__(self,*args):
            calls.append('SDK')
            if failing:raise RuntimeError('libdirp unavailable')
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def temperature(self):return values
        def processed_rgb(self):return np.full((120 if bad_shape else 60,80,3),255,np.uint8)
        @property
        def measurement_params(self):return MeasurementParams(20.,50.,.95,16.,16.)
    def search(grid,*,raw_supplement=False):
        assert np.shares_memory(grid,values)
        p={'proposal_id':'N001','proposal_origin':'hybrid_structure',
           'polygon_native_xy':[[5,5],[30,5],[30,50],[5,50]]}
        q={'proposal_id':'N002','proposal_origin':'raw_structure_supplement',
           'polygon_native_xy':[[25,5],[50,5],[50,50],[25,50]]}
        return {'proposals':[p,q] if raw_supplement else [p],'measurement_authorised':False}
    monkeypatch.setattr(native_review,'RJpeg',SDK)
    monkeypatch.setattr(native_review,'native_window_proposals',search)
    return values,calls


def test_preview_uses_native_roi_values_and_excludes_shared_pixels_without_writes(tmp_path,monkeypatch):
    path=tmp_path/'source_T.JPG';path.write_bytes(b'original-rjpeg')
    values,calls=fake_engine(monkeypatch);before=values.copy()
    preview=native_review.native_boundary_preview(path,'sdk',expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report=preview.report
    assert calls==['SDK']
    assert report['baseline_proposals']==1 and report['raw_additions']==1
    assert report['descriptive_statistics'][0]['max_c']==87.
    assert report['descriptive_statistics'][1]['max_c']==25.
    assert report['old_interior_pixels_excluded']>0
    assert report['old_pixels_lost_outside_shared_envelopes']==0
    assert report['lost_baseline_proposal_ids']==[]
    assert report['radiometric_parameters']['emissivity']==.95
    assert report['native_values_unchanged'] and not report['measurement_authorised']
    assert not report['database_written'] and not report['anomaly_rerun']
    assert preview.proposed_image.size==(80,60)
    assert (np.asarray(preview.support_image)[20,27]==[255,0,255]).all()
    assert path.read_bytes()==b'original-rjpeg'
    assert list(tmp_path.iterdir())==[path]
    np.testing.assert_array_equal(values,before)
    json.dumps(report,allow_nan=False)


def test_changed_indexed_source_is_rejected_before_sdk_decode(tmp_path,monkeypatch):
    path=tmp_path/'source_T.JPG';path.write_bytes(b'changed')
    _,calls=fake_engine(monkeypatch)
    with pytest.raises(ValueError,match='SHA256 differs'):
        native_review.native_boundary_preview(path,'sdk',expected_sha256='0'*64)
    assert calls==[]


@pytest.mark.parametrize('settings',[{'bad_shape':True},{'nonfinite':True}])
def test_bad_sdk_render_or_nonfinite_grid_cannot_receive_native_boundaries(tmp_path,monkeypatch,settings):
    path=tmp_path/'source_T.JPG';path.write_bytes(b'source')
    fake_engine(monkeypatch,**settings)
    with pytest.raises(ValueError,match='identical native dimensions'):
        native_review.native_boundary_preview(path,'sdk')


def test_sdk_failure_never_falls_back_to_embedded_jpeg_geometry(tmp_path,monkeypatch):
    path=tmp_path/'source_T.JPG';path.write_bytes(b'not-an-embedded-image')
    fake_engine(monkeypatch,failing=True)
    with pytest.raises(RuntimeError,match='libdirp unavailable'):
        native_review.native_boundary_preview(path,'sdk')


def test_source_change_during_preview_cannot_publish_stale_overlay(tmp_path,monkeypatch):
    path=tmp_path/'source_T.JPG';path.write_bytes(b'original')
    fake_engine(monkeypatch)
    search=native_review.native_window_proposals
    def changed(grid,**kwargs):
        path.write_bytes(b'replaced')
        return search(grid,**kwargs)
    monkeypatch.setattr(native_review,'native_window_proposals',changed)
    with pytest.raises(ValueError,match='SHA256 differs'):
        native_review.native_boundary_preview(path,'sdk')


def test_preview_token_changes_for_replaced_content_or_sdk_selection(tmp_path,monkeypatch):
    path=tmp_path/'source_T.JPG';path.write_bytes(b'original')
    first=native_review.boundary_preview_token(path,'sdk')
    path.write_bytes(b'replaced')
    assert native_review.boundary_preview_token(path,'sdk')!=first
    assert native_review.boundary_preview_token(path,'sdk2')!=native_review.boundary_preview_token(path,'sdk')
    monkeypatch.setenv('PVDELTA_DIRP_LIB','sdk')
    assert native_review.boundary_preview_token(path,None)==native_review.boundary_preview_token(path,'sdk')


def test_indexed_frame_listing_is_read_only_and_excludes_rgb(tmp_path):
    duckdb=pytest.importorskip('duckdb')
    from pvdelta.store import CaptureRecord, connect_store, upsert_capture

    database=tmp_path/'flight.duckdb'
    connection=connect_store(database)
    for name,kind in [('first','thermal'),('second','thermal'),('visible','rgb')]:
        upsert_capture(connection,CaptureRecord(name,name+'.JPG','f'*64,kind,run_id='flight'))
    connection.close()
    before=hashlib.sha256(database.read_bytes()).hexdigest()
    frames=native_review.indexed_native_frames(database)
    assert [f['capture_id'] for f in frames]==['first','second']
    assert hashlib.sha256(database.read_bytes()).hexdigest()==before
    missing=tmp_path/'absent'/'no-store.duckdb'
    with pytest.raises(duckdb.IOException):native_review.indexed_native_frames(missing)
    assert not missing.parent.exists()


def test_indexed_relative_sources_use_the_acquisition_root_and_never_the_working_directory(tmp_path):
    pytest.importorskip('duckdb')
    from pvdelta.store import CaptureRecord, connect_store, upsert_capture, upsert_run

    database=tmp_path/'flight.duckdb';root=tmp_path/'flight'
    connection=connect_store(database)
    upsert_run(connection,run_id='flight',site_label=None,started_at_utc=None,
               camera_model=None,source_root=str(root))
    for capture,path,flight in [('relative','nested/source_T.JPG','flight'),
                                ('absolute',str(tmp_path/'absolute_T.JPG'),'flight'),
                                ('missing-root','source_T.JPG','unknown'),
                                ('escape','../outside_T.JPG','flight')]:
        upsert_capture(connection,CaptureRecord(capture,path,'f'*64,'thermal',run_id=flight))
    connection.close()
    rows={r['capture_id']:r for r in native_review.indexed_native_frames(database)}
    assert rows['relative']['source_path']==str(root/'nested/source_T.JPG')
    assert rows['absolute']['source_path']==str(tmp_path/'absolute_T.JPG')
    assert rows['missing-root']['source_path'] is None
    assert rows['missing-root']['source_name']=='source_T.JPG'
    assert rows['escape']['source_path'] is None


@pytest.mark.sdk
@pytest.mark.skipif(not(os.environ.get('PVDELTA_DIRP_LIB') and os.environ.get('PVDELTA_SAMPLE_RJPEG')),
                    reason='local SDK/sample not set')
def test_real_sdk_native_boundary_preview():
    pytest.importorskip('cv2')
    preview=native_review.native_boundary_preview(os.environ['PVDELTA_SAMPLE_RJPEG'],os.environ['PVDELTA_DIRP_LIB'])
    report=preview.report
    assert report['native_shape']==[512,640]
    assert preview.proposed_image.size==(640,512)
    assert report['native_values_unchanged'] and not report['database_written']
    assert report['old_pixels_lost_outside_shared_envelopes']==0
