import copy

import pytest

from pvdelta.evaluation_cohort import digest, score_panels, select_cohort


def rows():
    result = []
    for i in range(180):
        result.append({'source_sha256': f'{i+1:064x}', 'rgb_sha256': f'{i+1001:064x}',
            'camera_model': 'M4T', 'native_width': 640, 'native_height': 512,
            'latitude': 40+(i//90)*.002+(i%90//3)*.000001,
            'longitude': 26+(i%7)*.00001, 'pitch_deg': -90, 'yaw_deg': 0 if i%2 else 180,
            'analyzed_observations': 0 if i%3 == 0 else 1,
            'finding_count': 1 if i%3 == 2 else 0})
    return result


def test_selection_is_spatial_stratified_deterministic_and_order_independent():
    source = rows()
    before = copy.deepcopy(source)
    a = select_cohort(source, seed='frozen')
    b = select_cohort(source[::-1], seed='frozen')
    assert a['frames'] == b['frames']
    assert source == before
    assert len(a['frames']) == 60
    assert a['minimum_cross_split_camera_distance_m'] >= 50
    for split in ('development', 'locked'):
        selected = [r for r in a['frames'] if r['split'] == split]
        assert len(selected) == 30
        for stratum in ('candidate', 'no_candidate', 'no_analyzed_modules'):
            sub = [r for r in selected if r['stratum'] == stratum]
            assert len(sub) == 10
            assert {r['heading_family'] for r in sub} == {'forward', 'return'}
    assert a['independent_flight'] is False
    assert a['prior_exposure_audited'] is False
    assert a['surveyed_footprint_nonoverlap_verified'] is False


def test_exclusions_are_explicit_and_not_relabelled_normal():
    source = rows()
    source[0]['latitude'] = None
    source[1]['pitch_deg'] = -45
    source[2]['rgb_sha256'] = None
    source[3]['camera_model'] = 'XT2'
    result = select_cohort(source, seed='fixed', regression_hashes=[source[4]['source_sha256']])
    assert {e['reason'] for e in result['excluded']} == {
        'invalid_camera_position_or_orientation', 'outside_nadir_cohort',
        'missing_original_rgb_pair', 'outside_supported_camera_grid', 'known_regression'}
    assert not {r['source_sha256'] for r in result['frames']} & {r['source_sha256'] for r in source[:5]}


def test_insufficient_geographic_separation_does_not_reduce_buffer():
    source = rows()
    for r in source:
        r['latitude'], r['longitude'] = 40, 26
    with pytest.raises(ValueError, match='Insufficient spatially separated'):
        select_cohort(source, seed='fixed')


@pytest.mark.parametrize('mutate', [lambda r: r.append(r[0]),
    lambda r: r[0].update(source_sha256='bad'), lambda r: r[0].update(finding_count=-1)])
def test_invalid_inventory_is_rejected(mutate):
    source = rows()
    mutate(source)
    with pytest.raises(ValueError):
        select_cohort(source, seed='fixed')


def fixture():
    case = {'case_id': 'C001', 'source_sha256': 'a'*64,
            'split': 'development', 'native_width': 20, 'native_height': 20}
    predictions = {'schema': 'pvdelta.frozen-panel-predictions/v1', 'frames': [
        {'source_sha256': 'a'*64, 'native_shape': [20, 20], 'proposals': []}]}
    manifest = {'schema': 'pvdelta.fixed-evaluation-cohort/v1', 'frames': [case],
                'predictions_sha256': digest(predictions),
                'acceptance': {'panel_iou_threshold': .75, 'ignore_fraction': .9}}
    reference = {'schema': 'pvdelta.native-panel-instances/v1', 'source_sha256': 'a'*64,
        'coordinate_space': 'thermal_native', 'native_width': 20, 'native_height': 20,
        'coverage': 'exhaustive_full_frame', 'review_status': 'developer_draft',
        'panels': [{'instance_id': 'P001', 'polygon_native_xy': [[2, 2], [10, 2], [10, 10], [2, 10]]}],
        'ignore': []}
    return manifest, predictions, reference


def test_missing_truth_is_pending_not_negative_or_perfect_recall():
    manifest, predictions, _ = fixture()
    result = score_panels(manifest, predictions, [])
    assert result['status'] == 'references_pending'
    assert result['scored_frames'] == 0
    assert result['pending_frames'] == ['C001']
    assert result['totals'] is None
    assert result['precision'] is None and result['recall'] is None
    assert result['stage1_accepted'] is False


def test_no_predictions_still_counts_annotated_missed_panels():
    manifest, predictions, reference = fixture()
    result = score_panels(manifest, predictions, [reference])
    assert result['totals'] == {'true_positives': 0, 'false_positives': 0, 'false_negatives': 1}
    assert result['recall'] == 0
    assert result['frames'][0]['reference_review_status'] == 'developer_draft'
    assert result['stage1_accepted'] is False


def test_explicit_exhaustive_empty_reference_counts_false_positives():
    manifest, predictions, reference = fixture()
    predictions['frames'][0]['proposals'] = [
        {'proposal_id': 'X1', 'polygon_native_xy': reference['panels'][0]['polygon_native_xy']}]
    manifest['predictions_sha256'] = digest(predictions)
    reference['panels'] = []
    result = score_panels(manifest, predictions, [reference])
    assert result['totals']['false_positives'] == 1
    assert result['recall'] is None


def test_changed_predictions_or_reference_identity_cannot_score():
    manifest, predictions, reference = fixture()
    predictions['frames'][0]['proposals'].append({'proposal_id': 'changed'})
    with pytest.raises(ValueError, match='Frozen predictions'):
        score_panels(manifest, predictions, [reference])
    manifest, predictions, reference = fixture()
    reference['source_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='outside the selected cohort'):
        score_panels(manifest, predictions, [reference])
    reference['source_sha256'] = 'a'*64
    reference['native_width'] = 21
    with pytest.raises(ValueError, match='native source grid'):
        score_panels(manifest, predictions, [reference])


def test_locked_set_requires_release_and_complete_annotations():
    manifest, predictions, reference = fixture()
    manifest['frames'][0]['split'] = 'locked'
    with pytest.raises(ValueError, match='explicit release'):
        score_panels(manifest, predictions, [reference], split='locked')
    with pytest.raises(ValueError, match='Complete and freeze all locked'):
        score_panels(manifest, predictions, [], split='locked', unlock_locked=True)
    result = score_panels(manifest, predictions, [reference], split='locked', unlock_locked=True)
    assert result['recall'] == 0 and result['stage1_accepted'] is False
