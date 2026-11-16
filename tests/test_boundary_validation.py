import json
from copy import deepcopy

import pytest

pytest.importorskip('PIL')
from PIL import Image

from pvdelta.boundary_validation import compare_boundary_reference, reference_validation_token
from pvdelta.native_review import NativeBoundaryPreview


def quad(x):
    return [[x,10],[x+20,10],[x+20,60],[x,60]]


@pytest.fixture
def case():
    first={'proposal_id':'P1','polygon_native_xy':quad(10)}
    ground={'proposal_id':'ground','polygon_native_xy':quad(90)}
    second={'proposal_id':'P2','polygon_native_xy':quad(40)}
    r={'schema':'pvdelta.native-boundary-preview/v1','source_name':'synthetic_T.JPG',
       'source_sha256':'a'*64,'native_temperature_sha256':'b'*64,
       'engine_source_sha256':{'native_windows.py':'c'*64},'native_shape':[100,120],
       'semantic_panel_membership':'unverified','native_values_unchanged':True,
       'measurement_authorised':False,'database_written':False,'anomaly_rerun':False,
       'baseline_proposals':2,'proposals':2,'baseline':{'proposals':[first,ground]},
       'proposed':{'proposals':[first,second]}}
    ref={'schema':'pvdelta.native-panel-instances/v1','coverage':'exhaustive_full_frame',
         'coordinate_space':'thermal_native','source_sha256':'a'*64,
         'native_height':100,'native_width':120,'review_status':'developer_draft',
         'panels':[{'instance_id':'A','polygon_native_xy':quad(10)},
                   {'instance_id':'B','polygon_native_xy':quad(40)}],'ignore':[]}
    image=Image.new('RGB',(120,100),'black')
    return NativeBoundaryPreview(r,image,image,image),ref


def content(ref):
    return json.dumps(ref).encode()


def test_scores_and_error_overlays_preserve_input_and_do_not_authorise_measurement(case):
    preview,ref=case;original=deepcopy(preview.report);raw=preview.baseline_image.tobytes()
    result=compare_boundary_reference(preview,content(ref));scores=result.report['scores']
    for threshold in ('0.5','0.75'):
        assert [scores['baseline'][threshold][k] for k in ('true_positives','false_positives','false_negatives')]==[1,1,1]
        assert [scores['proposed'][threshold][k] for k in ('true_positives','false_positives','false_negatives')]==[2,0,0]
    assert result.baseline_image.getpixel((40,10))==(255,85,85)  # missed reference
    assert result.baseline_image.getpixel((90,40))==(255,173,51)  # false positive
    assert result.proposed_image.getpixel((40,40))==(53,232,139)  # matched
    assert preview.baseline_image.tobytes()==raw and preview.report==original
    assert result.report['declared_reference_review_status']=='developer_draft'
    assert result.report['reference_correctness_verified'] is False
    assert 'does not establish a detector error' in result.report['mismatch_interpretation']
    assert not any(result.report[k] for k in ('measurement_authorised','database_written','anomaly_rerun'))


def test_bundle_selects_by_source_hash_instead_of_filename_or_position(case):
    preview,ref=case;wrong=deepcopy(ref);wrong['source_sha256']='d'*64
    ref['source']='renamed-but-same-bytes.JPG'
    result=compare_boundary_reference(preview,content({'frames':[wrong,ref]}))
    assert result.report['scores']['proposed']['0.75']['true_positives']==2
    for frames in ([wrong],[ref,ref]):
        with pytest.raises(ValueError,match='exactly one'):
            compare_boundary_reference(preview,content({'frames':frames}))


@pytest.mark.parametrize('change',['hash','grid','sparse','bad_panels','nan','invalid_json','root','bool_grid'])
def test_wrong_or_malformed_reference_is_rejected(case,change):
    preview,ref=case
    if change=='hash':ref['source_sha256']='d'*64
    elif change=='grid':ref['native_width']=121
    elif change=='sparse':ref['coverage']='sparse_patches'
    elif change=='bad_panels':ref['panels']=[None]
    elif change=='nan':ref['panels'][0]['polygon_native_xy'][0][0]=float('nan')
    elif change=='root':ref=[]
    elif change=='bool_grid':ref['native_width']=True
    data=b'invalid' if change=='invalid_json' else content(ref)
    with pytest.raises(TypeError if change=='root' else ValueError):
        compare_boundary_reference(preview,data)


def test_changed_labels_result_or_scorer_invalidates_token(case,monkeypatch):
    from pvdelta import boundary_validation

    preview,ref=case;data=content(ref)
    old=reference_validation_token(preview.report,data)
    ref['annotation_method']='changed reference provenance'
    assert reference_validation_token(preview.report,content(ref))!=old
    changed=deepcopy(preview.report);changed['radiometric_parameters']={'emissivity':.9}
    assert reference_validation_token(changed,data)!=old
    monkeypatch.setattr(boundary_validation,'validation_engine_sources',lambda:{'scorer':'changed'})
    assert reference_validation_token(preview.report,data)!=old


def test_native_overlay_dimensions_must_match(case):
    preview,ref=case
    resized=NativeBoundaryPreview(preview.report,Image.new('RGB',(240,200)),preview.proposed_image,preview.support_image)
    with pytest.raises(ValueError,match='native image dimensions'):
        compare_boundary_reference(resized,content(ref))


def test_empty_reference_scores_candidates_as_unmatched_without_inventing_recall(case):
    preview,ref=case;ref['panels']=[]
    score=compare_boundary_reference(preview,content(ref)).report['scores']['proposed']['0.75']
    assert score['false_positives']==2 and score['recall'] is None


def test_reference_native_temperature_binding_is_explicit_and_checked(case):
    preview, ref = case
    result = compare_boundary_reference(preview, content(ref))
    assert result.report['reference_native_temperature_binding'] == 'not_supplied'
    ref['native_temperature_sha256'] = preview.report['native_temperature_sha256']
    result = compare_boundary_reference(preview, content(ref))
    assert result.report['reference_native_temperature_binding'] == 'matched'
    ref['native_temperature_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='measurement profile'):
        compare_boundary_reference(preview, content(ref))


@pytest.mark.parametrize('invalid', [None, '', 123, 'g' * 64, 'A' * 64])
def test_declared_native_temperature_hash_cannot_silently_disable_binding(case, invalid):
    preview, ref = case
    ref['native_temperature_sha256'] = invalid
    with pytest.raises(ValueError, match='SHA256'):
        compare_boundary_reference(preview, content(ref))
