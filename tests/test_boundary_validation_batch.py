import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip('duckdb')
pytest.importorskip('PIL')
from PIL import Image

from pvdelta import boundary_validation_batch as batch
from pvdelta import cli
from pvdelta.native_review import NativeBoundaryPreview, boundary_engine_sources
from pvdelta.store import connect_store


@pytest.fixture
def validation_case(tmp_path,monkeypatch):
    database=tmp_path/'inspection.duckdb'
    connection=connect_store(database)
    for flight in ('flight-A','flight-B'):
        connection.execute('INSERT INTO inspection_runs (run_id,source_root) VALUES (?,?)',[flight,str(tmp_path)])
    refs=[];paths=[];previews={};calls=[]
    polygon=[[10,10],[30,10],[30,60],[10,60]]
    for index in range(2):
        path=tmp_path/f'{index}_T.JPG';path.write_bytes(f'private-source-{index}'.encode());paths.append(path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        connection.execute('''INSERT INTO captures (capture_id,run_id,source_path,source_hash,image_kind)
            VALUES (?,?,?,?, 'thermal')''',[str(index),'flight-A',path.name,digest])
        # The same bytes in another acquisition must not make this flight ambiguous.
        if index==0:
            connection.execute('''INSERT INTO captures (capture_id,run_id,source_path,source_hash,image_kind)
                VALUES ('other-flight','flight-B',?,?, 'thermal')''',[path.name,digest])
        refs.append({'schema':'pvdelta.native-panel-instances/v1','coverage':'exhaustive_full_frame',
            'review_status':'developer_draft','coordinate_space':'thermal_native',
            'native_width':120,'native_height':100,'source_sha256':digest,
            'panels':[{'instance_id':'A','polygon_native_xy':polygon}]})
        proposal={'proposal_id':'N001','polygon_native_xy':polygon}
        report={'schema':'pvdelta.native-boundary-preview/v1','source_name':path.name,
            'source_sha256':digest,'native_temperature_sha256':'b'*64,
            'engine_source_sha256':boundary_engine_sources(),'native_shape':[100,120],
            'semantic_panel_membership':'unverified','native_values_unchanged':True,
            'measurement_authorised':False,'database_written':False,'anomaly_rerun':False,
            'baseline_proposals':index,'proposals':1,'baseline':{'proposals':[proposal] if index else []},
            'proposed':{'proposals':[proposal]}}
        image=Image.new('RGB',(120,100),'black')
        previews[str(path)]=NativeBoundaryPreview(report,image,image,image)
    connection.close()
    reference_path=tmp_path/'references.json'
    reference_path.write_text(json.dumps({'frames':refs}))
    def decode(path,sdk,*,expected_sha256):
        path=str(path);calls.append((path,sdk,expected_sha256))
        if not paths[int(Path(path).stem[0])].exists():raise FileNotFoundError('Missing thermal source')
        preview=previews[path]
        assert expected_sha256==preview.report['source_sha256']
        return preview
    monkeypatch.setattr(batch,'native_boundary_preview',decode)
    return {'database':database,'references':reference_path,'refs':refs,'paths':paths,
            'calls':calls,'output':tmp_path/'out','decode':decode}


def run(case,**kwargs):
    return batch.run_boundary_validation(case['database'],'flight-A',case['references'],
                                         case['output'],sdk_library='explicit-sdk',**kwargs)


def test_complete_batch_scopes_acquisition_and_preserves_database(validation_case):
    c=validation_case;before=c['database'].read_bytes();result=run(c)
    assert result['status']=='complete' and result['requested_frames']==result['succeeded_frames']==2
    assert result['failed_frames']==result['unprocessed_frames']==0
    assert result['indexed_acquisition_frames']==2
    for threshold in ('0.5','0.75'):
        before_score=result['aggregate_scores']['baseline'][threshold]
        after_score=result['aggregate_scores']['proposed'][threshold]
        assert (before_score['true_positives'],before_score['false_negatives'])==(1,1)
        assert (after_score['true_positives'],after_score['false_negatives'])==(2,0)
        assert after_score['precision']==after_score['recall']==1
    assert c['database'].read_bytes()==before
    assert len(c['calls'])==2 and all(call[1]=='explicit-sdk' for call in c['calls'])
    assert (c['output']/'summary.html').is_file()
    assert json.loads((c['output']/'summary.json').read_text())==result
    for row in result['frames']:
        folder=c['output']/'frames'/row['source_sha256']
        assert {p.name for p in folder.iterdir()}=={'preview.json','comparison.json','baseline.png','proposed.png'}
    assert not result['database_written'] and not result['measurement_authorised']


def test_missing_source_remains_failed_instead_of_silently_shrinking_denominator(validation_case):
    c=validation_case;c['paths'][1].unlink();result=run(c)
    assert result['status']=='partial' and result['failed_frames']==result['succeeded_frames']==1
    assert result['requested_reference_instances']==2 and result['scored_reference_instances']==1
    assert result['frames'][1]['status']=='failed' and 'scores' not in result['frames'][1]
    assert result['aggregate_scores']['proposed']['0.75']['reference_instances']==1
    assert 'Not scored' in (c['output']/'summary.html').read_text()


def test_complete_only_means_supplied_references_not_all_indexed_frames(validation_case):
    c=validation_case;connection=connect_store(c['database'])
    connection.execute("INSERT INTO captures SELECT * REPLACE ('unlabelled' AS capture_id, ? AS source_hash) FROM captures WHERE capture_id='0'",['e'*64])
    connection.close();result=run(c)
    assert result['status']=='complete' and result['requested_frames']==2
    assert result['indexed_acquisition_frames']==3 and result['indexed_records_without_reference']==1
    assert len(c['calls'])==2


@pytest.mark.parametrize('change',['absent','ambiguous','unknown_root'])
def test_bad_index_identity_is_explicit_and_never_uses_other_flight(validation_case,change):
    c=validation_case;connection=connect_store(c['database'])
    if change=='absent':connection.execute("DELETE FROM captures WHERE capture_id='0'")
    elif change=='ambiguous':
        connection.execute("INSERT INTO captures SELECT * REPLACE ('duplicate' AS capture_id) FROM captures WHERE capture_id='0'")
    else:connection.execute("UPDATE inspection_runs SET source_root=NULL WHERE run_id='flight-A'")
    connection.close();result=run(c)
    assert result['frames'][0]['status']=='failed'
    assert all(call[0]!=str(c['paths'][0]) for call in c['calls'])
    if change=='unknown_root':
        assert result['status']=='failed' and result['aggregate_scores']['proposed']['0.75'] is None
        assert not c['calls']


@pytest.mark.parametrize('change',['duplicate','sparse','invalid_other_frame'])
def test_every_reference_is_validated_before_output_or_decode(validation_case,change):
    c=validation_case;refs=deepcopy(c['refs'])
    if change=='duplicate':refs.append(refs[0])
    elif change=='sparse':refs[0]['coverage']='sparse'
    else:refs[1]['panels']=[None]
    c['references'].write_text(json.dumps({'frames':refs}))
    with pytest.raises(ValueError):run(c)
    assert not c['calls'] and not c['output'].exists()


def test_existing_output_and_unknown_acquisition_do_not_decode(validation_case):
    c=validation_case;c['output'].mkdir();sentinel=c['output']/'keep.txt';sentinel.write_text('keep')
    with pytest.raises(FileExistsError):run(c)
    assert sentinel.read_text()=='keep' and not c['calls']
    with pytest.raises(ValueError,match='no indexed'):
        batch.run_boundary_validation(c['database'],'unknown',c['references'],c['output'].parent/'new-out')
    assert not c['calls']


@pytest.mark.parametrize('change',['labels','engine'])
def test_mutating_inputs_invalidates_run_and_withholds_aggregate_scores(validation_case,monkeypatch,change):
    c=validation_case;original=batch.boundary_engine_sources()
    def altered(*args,**kwargs):
        value=c['decode'](*args,**kwargs)
        if change=='labels':c['references'].write_text(json.dumps({'frames':c['refs'],'changed':True}))
        else:monkeypatch.setattr(batch,'boundary_engine_sources',lambda:dict(original,extra='d'*64))
        return value
    monkeypatch.setattr(batch,'native_boundary_preview',altered)
    result=run(c)
    assert result['status']=='invalidated' and result['aggregate_scores'] is None
    assert result['unprocessed_frames']==1 and len(c['calls'])==1


def test_cli_reports_partial_and_refuses_a_missing_index(validation_case,capsys,tmp_path):
    c=validation_case;c['paths'][1].unlink()
    args=['validate-boundaries','--database',str(c['database']),'--acquisition','flight-A',
          '--references',str(c['references']),'--out',str(c['output'])]
    assert cli.main(args)==3
    assert json.loads(capsys.readouterr().out)['failed_frames']==1
    missing=tmp_path/'absent.duckdb'
    args[args.index('--database')+1]=str(missing)
    args[args.index('--out')+1]=str(tmp_path/'not-created')
    assert cli.main(args)==1
    assert 'inspection index' in capsys.readouterr().err and not missing.exists()
