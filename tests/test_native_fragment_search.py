import numpy as np
import pytest

from pvdelta.native_fragment_search import protected_fragment_completion_proposals
from pvdelta.native_lines import NativeLineConfig


def stub(monkeypatch):
    import pvdelta.native_fragment_search as search
    calls=[]
    baseline={'proposals':[], 'native_completion':{'retained_additions':0}}
    def dense(values,**kwargs):calls.append(('dense',values,kwargs));return baseline
    def window(values,**kwargs):calls.append(('fragment',values,kwargs));return {'proposals':[]}
    def complete(first,second,**kwargs):
        calls.append(('complete',first,second));return dict(first,native_completion={'retained_additions':0})
    monkeypatch.setattr(search,'protected_dense_completion_proposals',dense)
    monkeypatch.setattr(search,'native_window_proposals',window)
    monkeypatch.setattr(search,'complete_native_proposals',complete)
    return calls,baseline


def test_only_fragment_length_and_budget_change_and_dense_baseline_is_protected(monkeypatch):
    calls,baseline=stub(monkeypatch)
    values=np.ones((64,80))
    lc=NativeLineConfig()
    result=protected_fragment_completion_proposals(values,line_config=lc)
    assert calls[0][1] is values and calls[1][1] is values
    cfg=calls[1][2]['line_config']
    assert cfg.min_segment_px==16 and cfg.max_pair_product==2_000_000
    for name in ('min_side_coverage','max_gap_fraction','merge_distance_px','max_angle_deg','geometry_chunk_pairs'):
        assert getattr(cfg,name)==getattr(lc,name)
    assert calls[2][1] is baseline
    assert result['baseline_dense_completion']==baseline['native_completion']
    assert result['fragment_search']['radiometric_resampling'] is False


@pytest.mark.parametrize('minimum',[12,16])
def test_does_not_repeat_or_narrow_already_short_fragment_search(monkeypatch,minimum):
    calls,baseline=stub(monkeypatch)
    result=protected_fragment_completion_proposals(np.ones((64,80)),line_config=NativeLineConfig(min_segment_px=minimum))
    assert [r[0] for r in calls]==['dense','complete']
    assert calls[-1][2] is baseline
    assert not result['fragment_search']['second_search_needed']


def test_source_mutation_aborts(monkeypatch):
    stub(monkeypatch)
    def mutate(values,**kwargs):values[0,0]=5;return {'proposals':[]}
    monkeypatch.setattr('pvdelta.native_fragment_search.native_window_proposals',mutate)
    with pytest.raises(RuntimeError,match='changed'):
        protected_fragment_completion_proposals(np.ones((64,80)))


@pytest.mark.parametrize('values',[np.ones(3),np.array([[np.inf]]),np.array([['x']])])
def test_invalid_grid_is_rejected_by_dense_entrypoint(values):
    with pytest.raises(ValueError):protected_fragment_completion_proposals(values)
