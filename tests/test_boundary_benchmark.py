import copy

import numpy as np
import pytest

from pvdelta.boundary_benchmark import evaluate_labeled_support


def reference():
    return {'source_sha256': 'a'*64, 'native_width': 20, 'native_height': 10,
        'coordinate_space': 'thermal_native', 'review_status': 'developer_draft', 'regions': [
            {'region_id': 'panel', 'label': 'panel', 'polygon_native_xy': [[1,1],[5,1],[5,5],[1,5]]},
            {'region_id': 'ground', 'label': 'non_panel', 'polygon_native_xy': [[10,1],[14,1],[14,5],[10,5]]}]}


def evaluate(ref, masks):
    return evaluate_labeled_support(ref, masks, source_sha256='a'*64, native_shape=(10,20))


def test_sparse_semantics_reports_coverage_and_intrusion_without_claiming_precision():
    mask = np.zeros((10,20), bool)
    mask[1:5,1:3] = True
    mask[1:5,10:11] = True
    mask[6:9,1:19] = True
    result = evaluate(reference(), [mask])
    assert result['panel_patch_coverage'] == .5
    assert result['non_panel_patch_intrusion'] == .25
    assert result['unlabeled_predicted_pixels'] == 54
    assert result['proposals'][0]['status'] == 'contradicts_non_panel_label'
    assert result['regions'][0]['coverage'] == .5
    assert result['regions'][1]['coverage'] == .25
    assert not result['measurement_authorised']
    assert 'precision' not in result and 'recall' not in result


def test_empty_proposals_are_missed_label_support_not_healthy():
    result = evaluate(reference(), [])
    assert result['panel_patch_coverage'] == 0
    assert result['covered_non_panel_pixels'] == 0
    assert result['status'] == 'sparse_reference_only'


def test_overlapping_proposals_count_each_labeled_pixel_once_in_frame_union():
    mask = np.ones((10,20), bool)
    result = evaluate(reference(), [mask, mask])
    assert result['covered_panel_pixels'] == 16
    assert result['covered_non_panel_pixels'] == 16


def test_ignored_pixels_and_unlabeled_masks_do_not_become_semantic_evidence():
    ref = reference()
    ref['regions'].append({'region_id': 'uncertain', 'label': 'ignore',
                          'polygon_native_xy': [[1,1],[5,1],[5,5],[1,5]]})
    mask = np.zeros((10,20), bool)
    mask[1:5,1:5] = True
    result = evaluate(ref, [mask])
    assert result['panel_patch_coverage'] is None
    assert result['proposals'][0]['status'] == 'unlabeled'


@pytest.mark.parametrize('change', ['hash', 'grid', 'duplicate_id', 'overlap', 'outside', 'missing_hash'])
def test_reference_identity_geometry_and_contradictions_fail_explicitly(change):
    ref = copy.deepcopy(reference())
    if change == 'hash':
        ref['source_sha256'] = 'b'*64
    elif change == 'grid':
        ref['native_width'] = 21
    elif change == 'duplicate_id':
        ref['regions'][1]['region_id'] = 'panel'
    elif change == 'overlap':
        ref['regions'][1]['polygon_native_xy'] = ref['regions'][0]['polygon_native_xy']
    elif change == 'outside':
        ref['regions'][0]['polygon_native_xy'][0][0] = -1
    else:
        ref.pop('source_sha256')
    with pytest.raises(ValueError):
        evaluate(ref, [])
