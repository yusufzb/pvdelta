import numpy as np
import pytest

pytest.importorskip('cv2')
from pvdelta.native_line_groups import _LineEvidence, line_affine_native_groups
from pvdelta.native_search_trace import begin_trace


def observed_quad(row,col):
    basis=np.array([[40.,14.],[0.,60.]])
    return np.array([[0,0],[1,0],[1,1],[0,1]])@basis.T+[80.,50.]+basis@np.array([col*1.1,row*1.1])


def mock_search(monkeypatch, count=4, mutate=False):
    calls=[]
    def search(values,**kwargs):
        calls.append(values)
        trace=begin_trace(values,support_mode='segments',local_contrast=True,
                          pair_budget=kwargs['line_config'].max_pair_product)
        if len(calls)==1:
            for i in range(count):
                trace.candidate('four_sides_passed',observed_quad(i//2,i%2),[(True,.9,.1)]*4)
        if mutate:values[0,0]+=1
        trace.finish({'pair_search':{'search_complete':True},'line_status':'test',
                      'line_diagnostics':{'four_side_candidates':count if len(calls)==1 else 0}})
        # Legacy neighbours may reject every quad: evidence precedes that filter.
        return {'proposals':[]}
    monkeypatch.setattr('pvdelta.native_line_groups._observed_quads',lambda v:([],0))
    monkeypatch.setattr('pvdelta.native_line_groups._line_pass',search)
    return calls


def test_non_contour_four_side_hypotheses_form_groups_before_legacy_selection(monkeypatch):
    calls=mock_search(monkeypatch)
    values=np.arange(320*400,dtype=float).reshape(320,400);before=values.copy()
    result=line_affine_native_groups(values)
    assert len(calls)==5 and all(np.shares_memory(v,values) for v in calls)
    assert len(result['proposals'])==4 and result['groups']
    assert all(p['source_method']=='four_observed_lines' and 'threshold_votes' not in p for p in result['proposals'])
    assert all(p['side_coverage']==[.9]*4 and len(p['neighbour_ids'])==2 for p in result['proposals'])
    assert not result['measurement_authorised'] and not result['efficacy_accepted']
    assert result['search_passes'][1]['window_xyxy']==[0,0,267,214]
    assert result['search_passes'][-1]['window_xyxy']==[133,106,400,320]
    np.testing.assert_array_equal(values,before)


def test_three_supported_quads_never_invent_a_fourth(monkeypatch):
    mock_search(monkeypatch,count=3)
    assert line_affine_native_groups(np.arange(320*400,dtype=float).reshape(320,400))['proposals']==[]


def test_explicit_dense_budget_uses_same_native_windows_and_gates(monkeypatch):
    mock_search(monkeypatch)
    result=line_affine_native_groups(np.arange(320*400,dtype=float).reshape(320,400),max_pair_product=2_000_000)
    assert len(result['proposals'])==4
    assert all(p['pair_budget']==2_000_000 for p in result['search_passes'])
    assert result['pair_budget_skipped_windows']==0


@pytest.mark.parametrize('budget',[0,2_000_001,250_000.,True])
def test_invalid_pair_budget_fails_before_search(budget):
    with pytest.raises(ValueError):
        line_affine_native_groups(np.arange(100*100,dtype=float).reshape(100,100),max_pair_product=budget)


def test_failed_side_cannot_enter_four_side_evidence():
    sink=_LineEvidence((320,400));sink({'event':'pass_begin','pass_id':1,'window_xyxy':[0,0,400,320]})
    with pytest.raises(ValueError,match='Four observed'):
        sink({'event':'candidate','stage':'four_sides_passed','pass_id':1,
              'polygon_native_xy':observed_quad(0,0).tolist(),'support':[[False,.79,.19]]*4})


def test_evidence_budget_withholds_all_groups(monkeypatch):
    mock_search(monkeypatch,count=8193)
    result=line_affine_native_groups(np.arange(320*400,dtype=float).reshape(320,400))
    assert result['status']=='line_evidence_budget_exceeded' and result['proposals']==[]


def test_native_mutation_fails_the_search(monkeypatch):
    mock_search(monkeypatch,mutate=True)
    with pytest.raises(RuntimeError,match='changed'):
        line_affine_native_groups(np.arange(320*400,dtype=float).reshape(320,400))


@pytest.mark.parametrize('values',[np.ones((20,20)),np.ones((40,40)),np.array([[np.nan]]),
    np.ones((80,160))[:,::2],np.ones(4),np.ones((40,40,3)),np.array([['x']])])
def test_invalid_flat_or_resampled_input_fails_explicitly(values):
    with pytest.raises(ValueError):line_affine_native_groups(values)


def test_actual_noise_does_not_become_a_forced_array():
    values=np.random.default_rng(102).normal(25,2,(160,200));before=values.copy()
    result=line_affine_native_groups(values)
    assert result['proposals']==[] and not result['measurement_authorised']
    np.testing.assert_array_equal(values,before)
