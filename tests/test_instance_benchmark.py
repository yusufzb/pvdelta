import copy

import numpy as np
import pytest

from pvdelta.instance_benchmark import _maximum_matching, evaluate_panel_instances


def quad(x, y, width=20, height=40):
    return [[x,y],[x+width,y],[x+width,y+height],[x,y+height]]


def reference():
    return {'schema':'pvdelta.native-panel-instances/v1','coverage':'exhaustive_full_frame',
        'review_status':'developer_draft','coordinate_space':'thermal_native',
        'native_width':120,'native_height':100,'source_sha256':'a'*64,
        'panels':[{'instance_id':'A','polygon_native_xy':quad(10,10)},
                  {'instance_id':'B','polygon_native_xy':quad(35,10)}], 'ignore':[]}


def evaluate(ref, polygons, **kwargs):
    return evaluate_panel_instances(ref,[{'proposal_id':f'P{i}', 'polygon_native_xy':q}
        for i,q in enumerate(polygons)],source_sha256='a'*64,native_shape=(100,120),**kwargs)


def test_duplicate_does_not_count_as_second_panel_and_ground_is_false_positive():
    result=evaluate(reference(),[quad(10,10),quad(10,10),quad(80,10)])
    assert (result['true_positives'],result['false_positives'],result['false_negatives'])==(1,2,1)
    assert result['precision']==pytest.approx(1/3)
    assert result['recall']==.5
    assert not result['measurement_authorised']


def test_merged_and_split_panels_fail_strict_instance_overlap():
    merged=evaluate(reference(),[quad(10,10,45,40)],iou_threshold=.75)
    assert (merged['true_positives'],merged['false_positives'],merged['false_negatives'])==(0,1,2)
    split=evaluate(reference(),[quad(10,10,20,20),quad(10,30,20,20)],iou_threshold=.75)
    assert (split['true_positives'],split['false_positives'],split['false_negatives'])==(0,2,2)


def test_augmenting_path_avoids_greedy_under_count_and_is_permutation_stable():
    # P0 can serve either instance; P1 can only serve A. Greedy highest IoU loses B.
    overlaps=np.array([[.8,.7],[.75,.1]])
    matches=_maximum_matching(overlaps,.5,['P0','P1'],['A','B'])
    assert matches=={0:1,1:0}
    reversed_matches=_maximum_matching(overlaps[::-1],.5,['P1','P0'],['A','B'])
    assert reversed_matches=={0:0,1:1}


def test_partial_source_panels_are_ignored_without_hiding_ground_or_full_panel_misses():
    ref=reference()
    ref['ignore']=[{'region_id':'partial','polygon_native_xy':quad(85,70,35,30)}]
    result=evaluate(ref,[quad(85,70,35,30),quad(60,5)])
    assert result['true_positives']==0
    assert result['false_positives']==1 and result['false_negatives']==2
    assert result['ignored_proposal_ids']==['P0']


def test_no_predictions_or_no_full_labels_does_not_invent_healthy_panels():
    empty=evaluate(reference(),[])
    assert empty['false_negatives']==2 and empty['precision'] is None and empty['recall']==0
    ref=reference();ref['panels']=[]
    ground=evaluate(ref,[quad(10,10)])
    assert ground['false_positives']==1 and ground['recall'] is None
    assert evaluate(ref,[])['precision'] is None


@pytest.mark.parametrize('change',['hash','grid','sparse','missing_panels','review','duplicate',
                                  'overlap','ignore_overlap','outside','concave'])
def test_ambiguous_reference_contracts_fail_explicitly(change):
    ref=copy.deepcopy(reference())
    if change=='hash':ref['source_sha256']='b'*64
    elif change=='grid':ref['native_width']=121
    elif change=='sparse':ref['coverage']='sparse_patches'
    elif change=='missing_panels':ref.pop('panels')
    elif change=='review':ref.pop('review_status')
    elif change=='duplicate':ref['panels'][1]['instance_id']='A'
    elif change=='overlap':ref['panels'][1]['polygon_native_xy']=quad(15,10)
    elif change=='ignore_overlap':ref['ignore']=[{'region_id':'bad','polygon_native_xy':quad(10,10)}]
    elif change=='outside':ref['panels'][0]['polygon_native_xy']=quad(-1,10)
    else:ref['panels'][0]['polygon_native_xy']=[[10,10],[30,10],[20,20],[30,50],[10,50]]
    with pytest.raises(ValueError):evaluate(ref,[])


@pytest.mark.parametrize('threshold',[0,1.1,True,float('nan')])
def test_invalid_overlap_thresholds_are_not_silently_accepted(threshold):
    with pytest.raises(ValueError):evaluate(reference(),[],iou_threshold=threshold)
