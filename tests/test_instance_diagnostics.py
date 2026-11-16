from copy import deepcopy

import pytest

from pvdelta.instance_diagnostics import diagnose_panel_instances


def box(left, top, right, bottom):
    return [[left,top],[right,top],[right,bottom],[left,bottom]]


def reference():
    return {'schema':'pvdelta.native-panel-instances/v1','coverage':'exhaustive_full_frame',
        'review_status':'developer_draft','coordinate_space':'thermal_native',
        'native_width':100,'native_height':100,'source_sha256':'a'*64,
        'panels':[{'instance_id':'A','polygon_native_xy':box(10,10,30,50)},
                  {'instance_id':'B','polygon_native_xy':box(35,10,55,50)}], 'ignore':[]}


def diagnose(polygons, ref=None, **kwargs):
    return diagnose_panel_instances(ref or reference(),[
        {'proposal_id':f'P{i}', 'polygon_native_xy':p} for i,p in enumerate(polygons)],
        source_sha256='a'*64,native_shape=(100,100),**kwargs)


def test_merged_box_cannot_be_counted_as_two_correct_panels():
    result=diagnose([box(10,10,55,50)])
    assert result['merge_like_proposals']==1
    assert result['proposal_diagnostics'][0]['merge_like_reference_ids']==['A','B']
    assert result['scores']['true_positives']==0 and result['scores']['false_negatives']==2
    assert result['references_without_native_coverage']==0


def test_disjoint_fragments_are_split_like_but_not_valid_complete_panels():
    result=diagnose([box(10,10,30,30),box(10,30,30,50)])
    first=result['reference_diagnostics'][0]
    assert first['split_like'] and first['split_like_proposal_ids']==['P0','P1']
    assert first['fragment_union_pixels']==800 and first['fragment_repeated_pixels']==0
    assert result['scores']['false_positives']==2
    assert result['references_without_native_coverage']==1


def test_duplicate_full_proposals_are_not_mislabeled_as_fragments():
    result=diagnose([box(10,10,30,50)]*2)
    assert result['duplicate_like_references']==1 and result['split_like_references']==0
    assert result['scores']['true_positives']==1 and result['scores']['false_positives']==1


def test_overlapping_fragments_do_not_establish_a_disjoint_split():
    result=diagnose([box(10,10,30,35),box(10,25,30,50)])
    assert result['split_like_references']==0
    assert result['reference_diagnostics'][0]['fragment_repeated_pixels']==200


def test_outside_reference_diagnostic_does_not_invent_a_ground_or_healthy_label():
    result=diagnose([box(70,10,85,40)])
    row=result['proposal_diagnostics'][0]
    assert row['score_status']=='unmatched_mostly_outside_reference_and_ignore'
    assert row['fraction_outside_reference_and_ignore']==1
    assert result['references_without_native_coverage']==2
    assert 'ground' not in result and 'healthy' not in result


def test_explicit_ignore_is_visible_and_does_not_hide_a_full_reference_miss():
    ref=reference()
    ref['ignore']=[{'region_id':'partial','polygon_native_xy':box(70,70,100,100)}]
    result=diagnose([box(70,70,100,100)],ref)
    assert result['proposal_status_counts']=={'ignored':1}
    assert result['ignore_fraction_of_frame']==pytest.approx(.09)
    assert result['ignore_pixels']==900 and result['scores']['false_negatives']==2


def test_reference_status_and_inputs_are_preserved_without_detector_side_effects():
    ref=reference()
    proposals=[{'proposal_id':'good','polygon_native_xy':box(10,10,30,50)}]
    original=deepcopy((ref,proposals))
    result=diagnose_panel_instances(ref,iter(proposals),source_sha256='a'*64,native_shape=(100,100))
    assert result['reference_review_status']=='developer_draft'
    assert not result['measurement_authorised']
    assert (ref,proposals)==original


def test_no_predictions_and_explicit_empty_reference_do_not_create_health_statistics():
    assert diagnose([])['references_without_native_coverage']==2
    ref=reference();ref['panels']=[]
    result=diagnose([box(10,10,30,50)],ref)
    assert result['scores']['false_positives']==1 and result['scores']['recall'] is None
    assert result['reference_diagnostics']==[]


def test_subpixel_proposal_with_no_centres_is_explicitly_rejected_by_native_contract():
    with pytest.raises(ValueError, match='pixel centres'):
        diagnose([box(.1,.1,.2,.2)])


def test_permutations_do_not_change_diagnostic_identity_order():
    ref=reference()
    proposals=[{'proposal_id':'Q','polygon_native_xy':box(10,10,30,50)},
               {'proposal_id':'P','polygon_native_xy':box(35,10,55,50)}]
    a=diagnose_panel_instances(ref,proposals,source_sha256='a'*64,native_shape=(100,100))
    ref['panels'].reverse()
    b=diagnose_panel_instances(ref,proposals[::-1],source_sha256='a'*64,native_shape=(100,100))
    assert a['proposal_diagnostics']==b['proposal_diagnostics']
    assert a['reference_diagnostics']==b['reference_diagnostics']


@pytest.mark.parametrize('change',['hash','grid','unreviewed','not_exhaustive','overlap','ignore_overlap','outside'])
def test_invalid_or_unresolved_references_cannot_generate_confident_diagnostics(change):
    ref=reference()
    if change=='hash':ref['source_sha256']='b'*64
    elif change=='grid':ref['native_width']=99
    elif change=='unreviewed':ref['review_status']='unreviewed'
    elif change=='not_exhaustive':ref['coverage']='not_annotated';ref['panels']=None
    elif change=='overlap':ref['panels'][1]['polygon_native_xy']=box(20,20,40,50)
    elif change=='ignore_overlap':ref['ignore']=[{'region_id':'bad','polygon_native_xy':box(10,10,30,50)}]
    else:ref['panels'][0]['polygon_native_xy']=box(-1,10,20,50)
    with pytest.raises(ValueError):diagnose([],ref)
