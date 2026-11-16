from dataclasses import asdict

import numpy as np
import pytest

from pvdelta.native_boundaries import NativeBoundaryConfig
from pvdelta.native_dense_search import protected_dense_completion_proposals
from pvdelta.native_lines import NativeLineConfig, _line_pass, _quad_candidates


def families(count, spacing, slope):
    return [{'offset': 20.+i*spacing, 'slope':slope, 'intervals':[(0.,2000.)]} for i in range(count)]


def pairs(horizontal,vertical):
    return [[(i,j,b['offset']-a['offset']) for i,a in enumerate(f) for j,b in enumerate(f) if i<j]
            for f in (vertical,horizontal)]


def scalar_oracle(horizontal,vertical,pair_list,cfg,pixels):
    # Independent scalar reference: explicit line intersections and four edges.
    result=[]
    for left,right,_ in pair_list[0]:
        for top,bottom,_ in pair_list[1]:
            vertices=[]
            for hi,vi in [(top,left),(top,right),(bottom,right),(bottom,left)]:
                h,v=horizontal[hi],vertical[vi]
                x=(v['offset']+v['slope']*h['offset'])/(1-h['slope']*v['slope'])
                vertices.append([x,h['slope']*x+h['offset']])
            q=np.array(vertices)
            edges=np.roll(q,-1,axis=0)-q
            lengths=np.sqrt((edges**2).sum(axis=1))
            width=(lengths[0]+lengths[2])/2; height=(lengths[1]+lengths[3])/2
            area=abs(sum(q[i,0]*q[(i+1)%4,1]-q[(i+1)%4,0]*q[i,1] for i in range(4)))/2
            turns=[np.linalg.det(np.stack([edges[i],edges[(i+1)%4]])) for i in range(4)]
            cosines=[np.dot(edges[i],edges[(i+1)%4])/(lengths[i]*lengths[(i+1)%4]) for i in range(4)]
            if (cfg.min_area_px<=area<=pixels*cfg.max_area_fraction and min(lengths)>=cfg.min_side_px
                    and cfg.min_aspect<max(width,height)/min(width,height)<cfg.max_aspect
                    and (min(turns)>0 or max(turns)<0) and max(map(abs,cosines))<=cfg.max_corner_cosine):
                result.append((left,right,top,bottom,q,width,height))
    return result


@pytest.mark.parametrize('chunk',[1,7,16384,32768])
def test_chunked_geometry_matches_scalar_enumeration_and_order(chunk):
    h,v=families(7,26,.015),families(6,49,-.01)
    cfg=NativeBoundaryConfig();pair_list=pairs(h,v)
    expected=scalar_oracle(h,v,pair_list,cfg,200_000)
    actual=list(_quad_candidates(h,v,pair_list,cfg,200_000,chunk_pairs=chunk))
    assert len(actual)==len(expected)>0
    for first,second in zip(actual,expected):
        assert first[:4]==second[:4]
        np.testing.assert_allclose(first[4],second[4],rtol=0,atol=1e-12)
        np.testing.assert_allclose(first[5:],second[5:],rtol=0,atol=1e-12)


def test_dense_enumeration_never_allocates_a_whole_pair_product(monkeypatch):
    import pvdelta.native_lines as lines
    h,v=families(50,21,0),families(30,41,0)
    pair_list=pairs(h,v)
    total=len(pair_list[0])*len(pair_list[1])
    assert total>250_000
    sizes=[];original=lines._quad_chunk
    def observed(h,v,left,right,top,bottom,cfg,pixels):
        sizes.append(len(left))
        yield from original(h,v,left,right,top,bottom,cfg,pixels)
    monkeypatch.setattr(lines,'_quad_chunk',observed)
    count=sum(1 for _ in lines._quad_candidates(h,v,pair_list,NativeBoundaryConfig(),2_000_000,chunk_pairs=4096))
    assert count>0 and max(sizes)<=4096 and sum(sizes)==total and len(sizes)>1


def test_budget_skip_and_complete_search_are_distinct_and_side_gates_unchanged(monkeypatch):
    import pvdelta.native_lines as lines
    h,v=families(4,22,0),families(4,42,0)
    monkeypatch.setattr(lines,'native_panel_proposals',lambda *a,**k:{'proposals':[]})
    monkeypatch.setattr(lines,'radiometric_structure',lambda a:(np.zeros(a.shape,np.uint8),{}))
    monkeypatch.setattr(lines,'_local_structure',lambda a:np.zeros(a.shape,np.uint8))
    monkeypatch.setattr(lines,'_families',lambda *a,**k:(np.eye(2),(h,v),100))
    values=np.ones((140,180))
    skipped=_line_pass(values,line_config=NativeLineConfig(max_pair_product=5))
    complete=_line_pass(values,line_config=NativeLineConfig(max_pair_product=2_000_000))
    assert skipped['proposals']==[] and not skipped['pair_search']['search_complete']
    assert skipped['line_status']=='too_many_line_pairs_for_bounded_search'
    assert complete['pair_search']['search_complete'] and complete['proposals']
    assert all(min(p['side_coverage'])>=.8 and max(p['side_max_gap_fraction'])<=.18 for p in complete['proposals'])


def test_dense_wrapper_changes_only_pair_budget_on_same_native_grid(monkeypatch):
    import pvdelta.native_dense_search as search
    values=np.arange(160*340,dtype=np.float32).reshape(160,340)
    before=values.copy();calls=[]
    def geometry(grid,**kwargs):
        assert grid is values
        calls.append(kwargs)
        return {'proposals':[]}
    monkeypatch.setattr(search,'native_window_proposals',geometry)
    cfg=NativeBoundaryConfig(min_neighbours=2)
    lc=NativeLineConfig(min_segment_px=27,geometry_chunk_pairs=128)
    result=protected_dense_completion_proposals(values,config=cfg,line_config=lc,raw_supplement=False)
    assert len(calls)==2
    assert calls[0]['config'] is cfg and calls[1]['config'] is cfg
    first,second=[asdict(c['line_config']) for c in calls]
    assert {k for k in first if first[k]!=second[k]}=={'max_pair_product'}
    assert first['max_pair_product']==250_000 and second['max_pair_product']==2_000_000
    assert all(c['raw_supplement'] is False for c in calls)
    assert not result['measurement_authorised']
    np.testing.assert_array_equal(values,before)


def test_already_dense_baseline_is_not_repeated_or_narrowed(monkeypatch):
    import pvdelta.native_dense_search as search
    calls=[]
    def geometry(grid,**kwargs):
        calls.append(kwargs)
        return {'proposals':[]}
    monkeypatch.setattr(search,'native_window_proposals',geometry)
    result=protected_dense_completion_proposals(np.ones((80,80)),
        line_config=NativeLineConfig(max_pair_product=2_000_000),max_pair_product=500_000)
    assert len(calls)==1 and not result['dense_search']['second_search_needed']


def test_mutated_grid_cannot_be_returned_as_preserved(monkeypatch):
    import pvdelta.native_dense_search as search
    def broken(grid,**kwargs):
        grid[0,0]+=1
        return {'proposals':[]}
    monkeypatch.setattr(search,'native_window_proposals',broken)
    with pytest.raises(RuntimeError,match='changed'):
        protected_dense_completion_proposals(np.ones((80,80)))


@pytest.mark.parametrize('changes',[{'max_pair_product':0},{'max_pair_product':True},
    {'max_pair_product':250000.5},{'max_pair_product':2000001},{'geometry_chunk_pairs':0},
    {'geometry_chunk_pairs':False},{'geometry_chunk_pairs':32769}])
def test_invalid_or_unbounded_search_settings_fail(changes):
    with pytest.raises(ValueError):NativeLineConfig(**changes)


@pytest.mark.parametrize('kwargs',[{'max_pair_product':True},{'max_pair_product':2000001},
    {'margin_px':-1},{'margin_px':True},{'raw_supplement':'yes'},{'config':'wide'},{'line_config':{}}])
def test_invalid_dense_preview_settings_fail_before_any_search(kwargs):
    with pytest.raises((ValueError,TypeError)):
        protected_dense_completion_proposals(np.ones((80,80)),**kwargs)


@pytest.mark.parametrize('values',[np.ones(10),np.ones((0,10)),[[float('nan')]],[[True]]])
def test_invalid_native_arrays_fail_before_search(values):
    with pytest.raises(ValueError):protected_dense_completion_proposals(values)
