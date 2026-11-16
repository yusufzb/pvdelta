import datetime as dt
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('PIL')
from PIL import Image

from pvdelta import boundary_review_pack as pack
from pvdelta import cli
from pvdelta.boundary_validation_batch import _references
from pvdelta.decode import MeasurementParams


def frame(index):
    return {'capture_id':str(index),'run_id':'flight-A','source_hash':hashlib.sha256(f'source-{index}'.encode()).hexdigest(),
            'source_name':f'{index}_T.JPG','source_path':f'/private/{index}_T.JPG',
            'captured_at':(dt.datetime(2026,10,19,10,tzinfo=dt.timezone.utc)+dt.timedelta(seconds=index)).isoformat()}


def test_selection_is_order_invariant_excludes_sources_and_spans_chronological_bins():
    frames=[frame(i) for i in range(22)]
    excluded={frames[0]['source_hash'],frames[-1]['source_hash']}
    first=pack.select_review_frames(frames,'flight-A',count=4,seed='fixed',excluded=excluded)
    second=pack.select_review_frames(frames[::-1],'flight-A',count=4,seed='fixed',excluded=excluded)
    assert first['selected']==second['selected']
    assert first['eligible_frames']==20 and first['rejection_counts']=={'excluded_source':2}
    assert all(
        1+i*5<=int(r['capture_id'])<1+(i+1)*5 for i,r in enumerate(first['selected']))
    assert not {r['source_hash'] for r in first['selected']} & excluded
    assert not first['predictions_used_for_selection'] and not first['scene_independence_verified']


def test_untimed_pool_is_explicit_and_source_duplicates_do_not_cross_acquisitions():
    frames=[frame(i) for i in range(6)]
    frames[4]['captured_at']=None;frames[5]['captured_at']='2026-10-19T11:00:00'  # timezone unknown
    frames.append(dict(frames[0],capture_id='other',run_id='flight-B'))
    result=pack.select_review_frames(frames,'flight-A',count=3)
    assert result['indexed_acquisition_frames']==6 and result['eligible_untimed_frames']==2
    assert [r['selection_pool'] for r in result['selected']].count('time_unavailable')==1
    only=pack.select_review_frames(frames[4:6],'flight-A',count=10)
    assert only['selected_frames']==2 and only['selection_shortfall']==8


def test_ambiguous_invalid_and_unavailable_sources_are_counted():
    frames=[frame(i) for i in range(4)]
    frames.append(dict(frames[0],capture_id='duplicate'))
    frames[1]['source_hash']='wrong';frames[2]['source_path']=None
    result=pack.select_review_frames(frames,'flight-A',excluded=[frames[3]['source_hash']])
    assert result['rejection_counts']=={'ambiguous_source_identity':2,'invalid_source_hash':1,
                                      'source_path_unavailable':1,'excluded_source':1}
    assert result['selected_frames']==0 and result['selection_shortfall']==12
    with pytest.raises(ValueError,match='capture IDs'):
        pack.select_review_frames([frame(0),frame(0)],'flight-A')


@pytest.mark.parametrize('count,seed',[(0,'x'),(-1,'x'),(True,'x'),(3,''),(3,None)])
def test_invalid_selection_settings_fail(count,seed):
    with pytest.raises(ValueError):pack.select_review_frames([frame(0)],'flight-A',count=count,seed=seed)


@pytest.fixture
def native_case(tmp_path,monkeypatch):
    frames=[];calls=[];values=np.full((30,40),25,np.float32);values[4,5]=99
    display=np.full((30,40,3),127,np.uint8)
    for i in range(4):
        record=frame(i);path=tmp_path/record['source_name'];path.write_bytes(f'source-{i}'.encode())
        record['source_path']=str(path);frames.append(record)
    class SDK:
        def __init__(self,path,sdk):calls.append((str(path),sdk))
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def temperature(self):return values
        def processed_rgb(self):return display
        @property
        def measurement_params(self):return MeasurementParams(20,50,.95,16,16)
    monkeypatch.setattr(pack,'RJpeg',SDK)
    monkeypatch.setattr(pack,'_read_frames',lambda database,acquisition:deepcopy(frames))
    from pvdelta import native_review
    def forbidden(*args,**kwargs):raise AssertionError('Detector must not run during annotation preparation')
    monkeypatch.setattr(native_review,'native_window_proposals',forbidden)
    return frames,calls,values,display


def test_pack_contains_unmarked_native_images_and_unscorable_drafts(native_case,tmp_path):
    _,calls,values,display=native_case;before=values.copy();output=tmp_path/'pack'
    result=pack.prepare_boundary_review('unused.duckdb','flight-A',output,count=3,sdk_library='explicit-sdk')
    assert result['status']=='complete' and result['succeeded_frames']==3 and len(calls)==3
    assert all(sdk=='explicit-sdk' for _,sdk in calls)
    assert not result['predictions_generated'] and not result['reference_labels_completed']
    assert not result['database_written'] and not result['independent_validation_verified']
    draft=json.loads((output/'draft-references.json').read_text())
    assert len(draft['frames'])==3
    with pytest.raises(ValueError):_references((output/'draft-references.json').read_bytes())
    for ref in draft['frames']:
        assert ref['coverage']=='not_annotated' and ref['review_status']=='unreviewed'
        assert ref['panels'] is None and ref['ignore'] is None
        assert (ref['native_width'],ref['native_height'])==(40,30)
        with Image.open(output/'frames'/ref['source_sha256']/'native.png') as picture:
            np.testing.assert_array_equal(np.asarray(picture),display)
        detail=json.loads((output/'frames'/ref['source_sha256']/'native.json').read_text())
        assert detail['native_temperature_sha256']==hashlib.sha256(values.tobytes()).hexdigest()
    np.testing.assert_array_equal(values,before)
    assert json.loads((output/'selection.json').read_text())['selected_frames']==3


def test_changed_source_is_not_decoded_or_silently_replaced(native_case,tmp_path):
    frames,calls,_,_=native_case
    Path(frames[0]['source_path']).write_bytes(b'changed')
    result=pack.prepare_boundary_review('unused','flight-A',tmp_path/'pack',count=4)
    assert result['status']=='partial' and result['failed_frames']==1 and result['succeeded_frames']==3
    assert len(calls)==3
    assert any('SHA256 differs' in r.get('error','') for r in result['frames'])


@pytest.mark.parametrize('change',['nonfinite','bad_render'])
def test_bad_sdk_data_cannot_become_an_annotation_grid(native_case,tmp_path,change):
    _,_,values,display=native_case
    if change=='nonfinite':values[0,0]=np.nan
    else:display.resize((15,40,3),refcheck=False)
    result=pack.prepare_boundary_review('unused','flight-A',tmp_path/'pack',count=1)
    assert result['status']=='failed' and result['succeeded_frames']==0
    assert json.loads((tmp_path/'pack/draft-references.json').read_text())['frames']==[]


def test_exclusion_documents_are_frozen_and_change_invalidates_pack(native_case,tmp_path,monkeypatch):
    frames,calls,_,_=native_case
    exclusion=tmp_path/'exclude.json';exclusion.write_text(json.dumps([frames[0]['source_hash']]))
    original=pack._render_native
    def altered(*args,**kwargs):
        value=original(*args,**kwargs);exclusion.write_text('[]');return value
    monkeypatch.setattr(pack,'_render_native',altered)
    result=pack.prepare_boundary_review('unused','flight-A',tmp_path/'pack',count=2,exclusions=[exclusion])
    assert result['status']=='invalidated' and result['invalidation_reason']
    assert len(calls)==1 and result['succeeded_frames']==0
    assert all(r['source_sha256']!=frames[0]['source_hash'] for r in result['frames'])


def test_existing_output_and_empty_eligible_pool_never_decode(native_case,tmp_path):
    frames,calls,_,_=native_case;output=tmp_path/'keep';output.mkdir()
    with pytest.raises(FileExistsError):pack.prepare_boundary_review('unused','flight-A',output)
    exclusion=tmp_path/'all.json';exclusion.write_text(json.dumps({'frames':[{'source_sha256':f['source_hash']} for f in frames]}))
    with pytest.raises(ValueError,match='No eligible'):
        pack.prepare_boundary_review('unused','flight-A',tmp_path/'absent',exclusions=[exclusion])
    assert not calls and not (tmp_path/'absent').exists()


def test_cli_reports_prepare_only_scope(native_case,tmp_path,capsys):
    output=tmp_path/'pack'
    assert cli.main(['prepare-boundary-review','--database','unused','--acquisition','flight-A',
                     '--count','2','--out',str(output)])==0
    payload=json.loads(capsys.readouterr().out)
    assert payload['succeeded_frames']==2 and not payload['reference_labels_completed']


def test_index_enrichment_is_read_only_and_camera_gps_is_not_panel_gps(tmp_path):
    pytest.importorskip('duckdb')
    from pvdelta.store import connect_store
    database=tmp_path/'index.duckdb';connection=connect_store(database)
    connection.execute("INSERT INTO inspection_runs (run_id,source_root) VALUES ('flight-A',?)",[str(tmp_path)])
    connection.execute('''INSERT INTO captures
        (capture_id,run_id,source_path,source_hash,image_kind,camera_latitude,camera_longitude)
        VALUES ('C','flight-A','thermal.JPG',?,'thermal',40,26)''',['a'*64])
    connection.close();before=database.read_bytes()
    rows=pack._read_frames(database,'flight-A')
    assert rows[0]['camera_latitude']==40 and rows[0]['camera_longitude']==26
    assert 'panel_latitude' not in rows[0] and rows[0]['source_name']=='thermal.JPG'
    assert database.read_bytes()==before
