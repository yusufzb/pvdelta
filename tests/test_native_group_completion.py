from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from pvdelta.config import DetectionConfig
from pvdelta.models import ThermalFrame
from pvdelta.native_group_completion import complete_local_native_groups
from pvdelta.pipeline import analyze_annotations
from pvdelta.regions import ModuleAnnotation, polygon_to_mask
from pvdelta.support_partition import partition_measurement_support

SHAPE=(300,400)
DIGEST='a'*64


def cell(row,col,prefix='A',shift=0):
    x,y=60+col*44+shift,50+row*66
    return {'proposal_id':f'{prefix}{row}{col}', 'polygon_native_xy':[[x,y],[x+40,y],[x+40,y+60],[x,y+60]],
        'source_method':'closed_contour', 'threshold_votes':2, 'affine_group_ids':[prefix], 'neighbour_ids':[]}


def alternative():
    cells=[cell(r,c) for r in range(2) for c in range(2)]
    return {'schema':'pvdelta.native-line-affine-groups/v1', 'status':'unreviewed_line_affine_geometry',
        'proposals':cells, 'groups':[{'group_id':'A','proposal_ids':[p['proposal_id'] for p in cells]}],
        'native_shape':list(SHAPE), 'native_temperature_sha256':DIGEST, 'radiometric_resampling':False,
        'measurement_authorised':False, 'semantic_panel_membership':'unverified'}


def baseline(items=()):
    return {'proposals':list(items), 'diagnostics':{'unchanged':True}}


def run(alt=None,old=None,**kwargs):
    return complete_local_native_groups(baseline() if old is None else old,alternative() if alt is None else alt,
        native_shape=SHAPE,native_temperature_sha256=DIGEST,**kwargs)


def test_observed_group_supplies_local_scale_without_any_old_anchor():
    alt,old=alternative(),baseline();original=deepcopy((alt,old))
    out=run(alt,old)
    assert (alt,old)==original and len(out['proposals'])==4
    assert out['diagnostics']==old['diagnostics']
    assert all(len(p['neighbour_ids'])==2 for p in out['proposals'])
    assert all(p['completion_support']['scale_basis']=='four_actual_local_affine_cells' for p in out['proposals'])
    assert not out['measurement_authorised'] and not out['local_group_completion']['efficacy_accepted']


def test_regular_non_pv_geometry_is_explicitly_not_semantically_authorised():
    out=run()  # Identical geometry can describe roof tiles: no source semantic evidence supplied.
    assert len(out['proposals'])==4
    assert all(p['semantic_panel_membership']=='unverified' and not p['measurement_authorised'] for p in out['proposals'])


def test_three_observations_without_a_group_never_fill_a_fourth_cell():
    alt=alternative();alt['proposals'].pop();alt['groups']=[]
    for p in alt['proposals']:p['affine_group_ids']=[]
    assert run(alt)['proposals']==[]


@pytest.mark.parametrize('change',['missing_member','repeated_member','wrong_membership','duplicate_group','dangling_group'])
def test_group_identity_and_membership_are_validated(change):
    alt=alternative()
    if change=='missing_member':alt['groups'][0]['proposal_ids'].pop()
    elif change=='repeated_member':alt['groups'][0]['proposal_ids'][0]=alt['groups'][0]['proposal_ids'][1]
    elif change=='wrong_membership':alt['proposals'][0]['affine_group_ids']=[]
    elif change=='duplicate_group':alt['groups'].append(deepcopy(alt['groups'][0]))
    else:alt['groups'][0]['proposal_ids'][0]='missing'
    with pytest.raises(ValueError):run(alt)


@pytest.mark.parametrize('change',['one_vote','inferred','hybrid','shape','single_row','overlap'])
def test_one_invalid_member_withholds_the_entire_group(change):
    alt=alternative();p=alt['proposals'][0]
    if change=='one_vote':p['threshold_votes']=1
    elif change=='inferred':p['source_method']='inferred_grid'
    elif change=='hybrid':p.update(source_method='observed_lines_and_native_ridges',side_coverage=[1]*4,side_max_gap_fraction=[0]*4)
    elif change=='shape':p['polygon_native_xy'][1][0]+=15
    elif change=='single_row':
        for i,p in enumerate(alt['proposals']):
            p['polygon_native_xy']=[[20+i*44,50],[60+i*44,50],[60+i*44,110],[20+i*44,110]]
    else:alt['proposals'][1]['polygon_native_xy']=deepcopy(p['polygon_native_xy'])
    assert run(alt)['proposals']==[]


@pytest.mark.parametrize('change',['valid','missing_trace','false_side','changed_trace','oversized_coverage'])
def test_four_segment_evidence_requires_original_support_metrics(change):
    alt=alternative();p=alt['proposals'][0]
    p.update(source_method='four_observed_lines',support_mode='segments',side_coverage=[.9]*4,
        side_max_gap_fraction=[.1]*4,original_side_support=[[True,.9,.1]]*4)
    if change=='missing_trace':p.pop('original_side_support')
    elif change=='false_side':p['original_side_support']=[[False,.9,.1]]*4
    elif change=='changed_trace':p['original_side_support']=[[True,.8,.1]]*4
    elif change=='oversized_coverage':p['side_coverage']=[1.01]*4
    assert len(run(alt)['proposals'])==(4 if change=='valid' else 0)


def test_old_interior_contact_below_overlap_limit_invalidates_group():
    p=cell(0,0);p['proposal_id']='original'
    p['polygon_native_xy']=[[23,50],[63,50],[63,110],[23,110]]
    out=run(old=baseline([p]))
    assert out['proposals']==[p]
    assert out['local_group_completion']['rejection_counts']['touches_existing_measurement_pixels']==1


def test_shared_excluded_edges_are_allowed_and_old_support_stays_identical():
    p=cell(0,0);p['proposal_id']='original'
    p['polygon_native_xy']=[[21,50],[61,50],[61,110],[21,110]]
    old=baseline([p]);out=run(old=old)
    assert len(out['proposals'])==5 and out['proposals'][0]==p
    def supports(geometry):
        return partition_measurement_support([polygon_to_mask(x['polygon_native_xy'],SHAPE) for x in geometry['proposals']],margin_px=2)
    assert np.array_equal(supports(old)[0].interior,supports(out)[0].interior)


def test_conflicting_group_members_invalidate_both_groups_without_order_ranking():
    alt=alternative();more=[cell(r,c,prefix='B',shift=1) for r in range(2) for c in range(2)]
    alt['proposals']+=more;alt['groups'].append({'group_id':'B','proposal_ids':[p['proposal_id'] for p in more]})
    first=run(alt);alt['proposals'].reverse();alt['groups'].reverse()
    assert first==run(alt) and first['proposals']==[]


def test_empty_support_cannot_supply_a_hidden_group_vote():
    assert run(margin_px=100)['proposals']==[]
    p=cell(0,0);p['proposal_id']='old1';other=deepcopy(p);other['proposal_id']='old2'
    assert run(old=baseline([p,other]))['proposals']==[p,other]


def test_existing_group_member_and_new_ids_are_preserved_and_mapped():
    p=cell(0,0);p['proposal_id']='C0001'
    out=run(old=baseline([p]))
    assert out['proposals'][0]==p and len(out['proposals'])==4
    assert {x['proposal_id'] for x in out['proposals']}=={'C0001','C0002','C0003','C0004'}
    assert sum('C0001' in x['neighbour_ids'] for x in out['proposals'][1:])==2


@pytest.mark.parametrize('change',['hash','shape','schema','resampling','approved','overflow','old_hash'])
def test_identity_or_budget_mismatch_fails_closed(change):
    alt=alternative();old=baseline()
    if change=='hash':alt['native_temperature_sha256']='b'*64
    elif change=='shape':alt['native_shape']=[640,512]
    elif change=='schema':alt['schema']='arbitrary'
    elif change=='resampling':alt['radiometric_resampling']=True
    elif change=='approved':alt['measurement_authorised']=True
    elif change=='overflow':alt['groups']*=4097
    else:old['native_temperature_sha256']='b'*64
    with pytest.raises(ValueError):run(alt,old)


def test_old_boundary_hotspot_report_is_bitwise_preserved():
    p=cell(0,0);p['proposal_id']='original';old=baseline([p]);out=run(old=old)
    values=np.full(SHAPE,25.,dtype=np.float32);values[53:58,63:68]=40
    untouched=values.copy();frame=ThermalFrame('test','synthetic','source',values,None,SHAPE[1],SHAPE[0])
    def reports(geometry):
        ps=geometry['proposals'];supports=partition_measurement_support([
            polygon_to_mask(x['polygon_native_xy'],SHAPE) for x in ps],margin_px=2)
        anns=[ModuleAnnotation(x['proposal_id'],tuple(map(tuple,x['polygon_native_xy']))) for x in ps]
        return [asdict(r) for r in analyze_annotations(frame,anns,config=DetectionConfig(),direction='hot',
            module_masks={x['proposal_id']:s.interior for x,s in zip(ps,supports)})]
    before=reports(old);after=reports(out)
    assert before[0]['findings'] and after[:1]==before
    np.testing.assert_array_equal(values,untouched)
