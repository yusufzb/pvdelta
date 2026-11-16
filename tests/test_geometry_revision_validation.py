import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest

from pvdelta.geometry_revision import analyze_geometry_version, apply_geometry_patch
from pvdelta.geometry_revision_validation import (
    compare_geometry_versions,
    revision_validation_token,
)
from pvdelta.models import ThermalFrame


def quad(x):
    return [[x, 15], [x+40, 15], [x+40, 70], [x, 70]]


def encode(value):
    return json.dumps(value).encode()


def resign(value):
    value['version_id'] = 'sha256:'+hashlib.sha256(json.dumps(
        {k:v for k,v in value.items() if k != 'version_id'},
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return value


@pytest.fixture
def data(tmp_path):
    source = tmp_path/'source.bin'
    source.write_bytes(b'geometry evaluation synthetic source')
    y,x = np.mgrid[:90, :180]
    values = (25+x*.01+y*.005).astype(np.float32)
    for left in (25, 75, 125):
        values[30:35, left:left+5] += 8
    frame = ThermalFrame('frame', str(source), hashlib.sha256(source.read_bytes()).hexdigest(), values, None, 180, 90)
    root = analyze_geometry_version(frame, {'proposals':[
        {'proposal_id':'A', 'polygon_native_xy':quad(10)},
        {'proposal_id':'ground', 'polygon_native_xy':quad(110)}]},
        measurement_parameters={'decoder':'synthetic'}, provenance={'kind':'automatic', 'actor':'test'})
    ref = {'schema':'pvdelta.native-panel-instances/v1', 'coordinate_space':'thermal_native',
        'coverage':'exhaustive_full_frame', 'source_sha256':frame.source_hash,
        'native_height':90, 'native_width':180, 'review_status':'developer_draft',
        'panels':[{'instance_id':i,'polygon_native_xy':quad(x)} for i,x in [('R-A',10),('R-B',60)]], 'ignore':[]}
    def revise(operations, parent=root):
        patch = {'schema':'pvdelta.native-geometry-patch/v1', 'parent_version_id':parent['version_id'],
            'source_sha256':frame.source_hash, 'native_temperature_sha256':parent['source']['native_temperature_sha256'],
            'operations':operations}
        return analyze_geometry_version(frame, apply_geometry_patch(parent, patch),
            measurement_parameters={'decoder':'synthetic'}, provenance={'kind':'user_edit', 'actor':'test'},
            parent=parent, patch=patch)
    return root, ref, revise


def test_gains_losses_and_exact_component_placement_are_separate(data):
    root,ref,revise = data
    child = revise([{'op':'remove','proposal_id':'ground','reason':'synthetic ground'},
        {'op':'add','proposal_id':'B','reason':'synthetic missed panel','polygon_native_xy':quad(60)}])
    saved = deepcopy((root,child))
    result = compare_geometry_versions(root, child, encode(ref))
    for threshold in ('0.5', '0.75'):
        before,after = [result['scores'][s][threshold] for s in ('before','after')]
        assert [before[k] for k in ('true_positives','false_positives','false_negatives')] == [1,1,1]
        assert [after[k] for k in ('true_positives','false_positives','false_negatives')] == [2,0,0]
        t = result['transitions'][threshold]
        assert t['newly_matched_reference_ids'] == ['R-B']
        assert t['no_longer_unmatched_proposal_ids'] == ['ground']
        assert t['newly_missed_reference_ids'] == []
    placements = result['component_placement']
    assert {r['placement'] for r in placements['before']['components']} == {'inside_reference_panels','outside_reference_panels'}
    assert placements['before']['outside_reference_pixel_observations'] > 0
    assert placements['after']['outside_reference_pixel_observations'] == 0
    assert all(r['reference_instance_ids'] for r in placements['after']['components'])
    assert (root,child) == saved
    assert not any(result[k] for k in ('independent_review_verified','automatic_accuracy_improvement_claimed',
        'measurement_authorised','database_written','main_scan_updated','native_values_redecoded'))


def test_lost_matches_are_not_hidden_by_decreasing_candidate_count(data):
    root,ref,revise = data
    child = revise([{'op':'remove','proposal_id':'A','reason':'counterfactual bad edit'}])
    result = compare_geometry_versions(root, child, encode(ref))
    assert result['transitions']['0.75']['newly_missed_reference_ids'] == ['R-A']
    assert result['transitions']['0.75']['still_missed_reference_ids'] == ['R-B']
    assert result['transitions']['0.75']['still_unmatched_proposal_ids'] == ['ground']


def test_nonadjacent_versions_and_empty_versions_are_comparable(data):
    root,ref,revise = data
    child = revise([{'op':'remove','proposal_id':'ground','reason':'remove'}])
    empty = revise([{'op':'remove','proposal_id':'A','reason':'remove'}], child)
    result = compare_geometry_versions(root, empty, encode(ref))
    assert result['scores']['after']['0.75']['precision'] is None
    assert result['scores']['after']['0.75']['false_negatives'] == 2
    assert result['component_placement']['after']['components'] == []


def test_declared_independence_and_assisted_inputs_do_not_establish_verification(data):
    root,ref,_revise = data
    root = deepcopy(root)
    root['provenance'].update(reference_assisted=True,kind='AI_developer_draft')
    resign(root)
    ref['review_status'] = 'independently_reviewed'
    result = compare_geometry_versions(root,root,encode(ref))
    assert result['declared_reference_review_status'] == 'independently_reviewed'
    assert result['selected_version_provenance']['after']['reference_assisted']
    assert not result['independent_review_verified'] and not result['automatic_accuracy_improvement_claimed']


@pytest.mark.parametrize('change',['hash','grid','sparse','native','ambiguous_bundle','missing_bundle'])
def test_wrong_references_fail_without_scoring_as_zero(data,change):
    root,ref,_revise = data
    if change == 'hash': ref['source_sha256'] = '0'*64
    elif change == 'grid': ref['native_width'] += 1
    elif change == 'sparse': ref['coverage'] = 'sparse_patches'
    elif change == 'native': ref['native_temperature_sha256'] = '0'*64
    elif change == 'ambiguous_bundle': ref = {'frames':[ref,ref]}
    elif change == 'missing_bundle': ref = {'frames':[]}
    with pytest.raises(ValueError):
        compare_geometry_versions(root,root,encode(ref))


def test_bundle_uses_hash_and_reports_selected_reference_identity(data):
    root,ref,_revise = data
    other = dict(ref, source_sha256='0'*64)
    result = compare_geometry_versions(root,root,encode({'frames':[other,ref]}))
    assert result['scores']['before']['0.75']['true_positives'] == 1
    assert result['reference_file_sha256'] and result['selected_reference_sha256']


@pytest.mark.parametrize('change',['source','settings','engine'])
def test_mixed_context_versions_are_rejected_even_with_valid_content_hashes(data,change):
    root,ref,_revise = data
    changed = deepcopy(root)
    changed[change]['new_context'] = True
    resign(changed)
    with pytest.raises(ValueError,match='must match'):
        compare_geometry_versions(root,changed,encode(ref))


@pytest.mark.parametrize('change',['no_pixels','duplicate','fractional','outside','wrong_module','not_in_support'])
def test_invalid_component_payload_does_not_turn_into_plausible_placement(data,change):
    root,ref,_revise = data
    root = deepcopy(root)
    finding = root['module_reports'][0]['findings'][0]
    if change == 'no_pixels': finding['region_pixels_yx'] = None
    elif change == 'duplicate': finding['region_pixels_yx'][0] = finding['region_pixels_yx'][1]
    elif change == 'fractional': finding['region_pixels_yx'][0][0] += .5
    elif change == 'outside': finding['region_pixels_yx'][0] = [1000,1000]
    elif change == 'wrong_module': finding['module_id'] = 'ground'
    elif change == 'not_in_support': finding['region_pixels_yx'][0] = [80,170]
    resign(root)
    with pytest.raises(ValueError):
        compare_geometry_versions(root,root,encode(ref))


def test_explicit_ignore_pixels_are_separate_from_outside_pixels(data):
    root,ref,_revise = data
    ref['ignore'] = [{'region_id':'ignore', 'polygon_native_xy':quad(110)}]
    result = compare_geometry_versions(root,root,encode(ref))
    p = result['component_placement']['before']
    assert p['outside_reference_pixel_observations'] == 0
    assert any(r['placement'] == 'explicit_ignore_only' for r in p['components'])
    assert result['scores']['before']['0.75']['ignored_proposal_ids'] == ['ground']


def test_pixel_placement_does_not_use_enlarged_display_box(data):
    root,ref,_revise = data
    before = compare_geometry_versions(root,root,encode(ref))
    root = deepcopy(root)
    for report in root['module_reports']:
        for finding in report['findings']:
            finding['pixel_polygon'] = [[0,0],[179,0],[179,89],[0,89]]
    resign(root)
    after = compare_geometry_versions(root,root,encode(ref))
    assert before['component_placement'] == after['component_placement']


def test_partial_component_coverage_preserves_pixels_outside_labels(data):
    root,ref,_revise = data
    ref['panels'][0]['polygon_native_xy'] = [[10,15],[27,15],[27,70],[10,70]]
    result = compare_geometry_versions(root,root,encode(ref))
    row = next(r for r in result['component_placement']['before']['components'] if r['proposal_id']=='A')
    assert row['placement'] == 'mixed_reference_coverage'
    assert 0 < row['fraction_in_reference_panels'] < 1
    assert row['outside_reference_panel_and_ignore_pixels'] > 0
    assert row['measured_pixels'] == row['in_reference_panel_pixels'] + row['outside_reference_panel_and_ignore_pixels']


def test_changed_reference_version_and_scorer_invalidate_token(data,monkeypatch):
    from pvdelta import geometry_revision_validation as validation
    root,ref,revise = data
    old = revision_validation_token(root,root,encode(ref))
    ref['annotation_method'] = 'new annotation provenance'
    assert revision_validation_token(root,root,encode(ref)) != old
    ref.pop('annotation_method')
    child = revise([{'op':'remove','proposal_id':'ground','reason':'remove'}])
    assert revision_validation_token(root,child,encode(ref)) != old
    monkeypatch.setattr(validation,'revision_validation_sources',lambda:{'changed':'1'})
    assert revision_validation_token(root,root,encode(ref)) != old
