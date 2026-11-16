import copy

import pytest

from pvdelta.hotspot_benchmark import evaluate_hotspot_regions, parse_yolo_hotspots


def box(identifier,left,right):
    return {'region_id':identifier,'polygon_native_xy':[[left,10],[right,10],[right,30],[left,30]]}


def test_yolo_native_coordinates_and_explicit_empty_label():
    r=parse_yolo_hotspots('0 0.5 0.25 0.2 0.1\n',native_shape=(100,200))
    assert r[0]['polygon_native_xy']==[[80,20],[120,20],[120,30],[80,30]]
    assert parse_yolo_hotspots('',native_shape=(100,200))==[]


@pytest.mark.parametrize('text',['1 .5 .5 .1 .1','0 .5 .5 .1','0 nan .5 .1 .1',
                                '0 .5 .5 -.1 .1','0 0 .5 .2 .1','0 .5 .5 1 2'])
def test_invalid_labels_fail_without_clipping_real_out_of_frame_boxes(text):
    with pytest.raises(ValueError):parse_yolo_hotspots(text,native_shape=(100,200))


def test_duplicate_predictions_do_not_inflate_true_positives_and_misses_are_retained():
    refs=[box('R1',10,30),box('R2',40,60)]
    pred=[box('P1',10,30),box('P2',10,30),box('P3',80,100)]
    r=evaluate_hotspot_regions(refs,pred,native_shape=(100,120))
    assert (r['true_positives'],r['false_positives'],r['false_negatives'])==(1,2,1)
    assert r['missed_reference_ids']==['R2'] and not r['measurement_authorised']


def test_augmenting_path_recovers_non_greedy_match_and_is_order_invariant():
    refs=[box('R1',10,30),box('R2',30,50)]
    pred=[box('P1',10,50),box('P2',10,30)]
    first=evaluate_hotspot_regions(refs,pred,native_shape=(100,120),iou_threshold=.5)
    second=evaluate_hotspot_regions(refs[::-1],pred[::-1],native_shape=(100,120),iou_threshold=.5)
    assert first['true_positives']==second['true_positives']==2


def test_no_geometry_cannot_hide_published_hotspots_as_healthy():
    r=evaluate_hotspot_regions([box('R',10,30)],[],native_shape=(100,120))
    assert r['false_negatives']==1 and r['recall']==0 and r['precision'] is None
    r=evaluate_hotspot_regions([],[],native_shape=(100,120))
    assert r['recall'] is None and r['precision'] is None


@pytest.mark.parametrize('change',['duplicate','outside','degenerate'])
def test_invalid_prediction_contracts_fail(change):
    pred=[box('P',10,30)]
    if change=='duplicate':pred.append(copy.deepcopy(pred[0]))
    elif change=='outside':pred[0]['polygon_native_xy'][0][0]=-1
    else:pred[0]['polygon_native_xy']=[[10,10],[20,10],[30,10]]
    with pytest.raises(ValueError):evaluate_hotspot_regions([],pred,native_shape=(100,120))


@pytest.mark.parametrize('threshold',[0,True,float('nan'),1.1])
def test_invalid_threshold_is_not_silently_accepted(threshold):
    with pytest.raises(ValueError):evaluate_hotspot_regions([],[],native_shape=(100,120),iou_threshold=threshold)
