import numpy as np
import pytest

from pvdelta.native_lines import NativeLineConfig, _line_pass
from pvdelta.native_search_trace import trace_native_search


def setup_lines(monkeypatch):
    import pvdelta.native_lines as lines
    h=[{'offset':y,'slope':0.,'intervals':[(0.,180.)]} for y in (15.,40.,65.,90.)]
    v=[{'offset':x,'slope':0.,'intervals':[(0.,120.)]} for x in (15.,65.,115.,165.)]
    monkeypatch.setattr(lines,'native_panel_proposals',lambda *a,**k:{'proposals':[]})
    monkeypatch.setattr(lines,'radiometric_structure',lambda a:(np.zeros(a.shape,np.uint8),{}))
    monkeypatch.setattr(lines,'_local_structure',lambda a:np.zeros(a.shape,np.uint8))
    monkeypatch.setattr(lines,'_families',lambda *a,**k:(np.eye(2),(h,v),24))


def test_trace_is_observational_and_detached_with_native_window_offsets(monkeypatch):
    setup_lines(monkeypatch)
    root=np.ones((150,220));window=root[10:130,20:200]
    expected=_line_pass(window)
    records=[]
    with trace_native_search(root,records.append):
        actual=_line_pass(window)
    assert actual==expected
    begin,end=records[0],records[-1]
    assert begin['window_xyxy']==[20,10,200,130]
    assert end['event']=='pass_end' and end['pair_search']['search_complete']
    candidates=[r for r in records if r['event']=='candidate']
    assert candidates
    for p in actual['proposals']:
        assert any(np.allclose(np.asarray(p['polygon_native_xy'])+[20,10],r['polygon_native_xy'])
                   for r in candidates)
    candidates[0]['polygon_native_xy'][0][0]=-999
    assert actual==expected


def test_budget_skip_is_explicit_with_no_candidate_events(monkeypatch):
    setup_lines(monkeypatch)
    a=np.ones((120,180));records=[]
    with trace_native_search(a,records.append):
        _line_pass(a,line_config=NativeLineConfig(max_pair_product=1))
    assert [r['event'] for r in records]==['pass_begin','pass_end']
    assert records[-1]['pair_search']['reason']=='too_many_line_pairs_for_bounded_search'


def test_rejection_counts_match_actual_decision_events(monkeypatch):
    setup_lines(monkeypatch)
    monkeypatch.setattr('pvdelta.native_lines._supported',lambda *a:(False,0.,1.))
    a=np.ones((120,180));records=[]
    with trace_native_search(a,records.append):
        result=_line_pass(a,support_mode='hybrid')
    rejected=[r for r in records if r.get('stage')=='hybrid_anchors']
    assert len(rejected)==result['line_diagnostics']['geometric_candidates']>0
    assert len(rejected)==result['line_diagnostics']['rejected_hybrid_anchors']


@pytest.mark.parametrize('values',[np.zeros((0,2)),np.ones((3,3))[:,::2],np.array([[np.nan]]),np.ones(3),np.array([['x']])])
def test_trace_rejects_invalid_native_root(values):
    with pytest.raises(ValueError), trace_native_search(values,lambda e:None):pass


def test_nested_trace_and_resampled_grid_are_rejected_and_context_is_reset(monkeypatch):
    setup_lines(monkeypatch)
    a=np.ones((120,180))
    with trace_native_search(a,lambda e:None):
        with pytest.raises(RuntimeError), trace_native_search(a,lambda e:None):pass
        with pytest.raises(ValueError):_line_pass(a.copy())
    _line_pass(a.copy())


def test_sink_failure_is_propagated_and_context_reset(monkeypatch):
    setup_lines(monkeypatch)
    a=np.ones((120,180))
    def failed(e):raise OSError('trace unavailable')
    with pytest.raises(OSError), trace_native_search(a,failed):_line_pass(a)
    _line_pass(a)


def test_source_mutation_is_detected():
    a=np.ones((20,30))
    with pytest.raises(RuntimeError,match='changed'), trace_native_search(a,lambda e:None):
        a[0,0]=2


def test_callable_sink_required():
    with pytest.raises(TypeError), trace_native_search(np.ones((2,3)),None):pass


def test_pregeometry_batches_partition_rejections_without_changing_candidates(monkeypatch):
    setup_lines(monkeypatch)
    values = np.ones((240, 360)); records = []
    expected = _line_pass(values)
    with trace_native_search(values, records.append):
        actual = _line_pass(values, line_config=NativeLineConfig(geometry_chunk_pairs=7))
    # The requested chunk size is metadata; polygons and actual decisions agree.
    assert actual['proposals'] == expected['proposals']
    assert actual['line_diagnostics'] == expected['line_diagnostics']
    batches = [r for r in records if r['event'] == 'geometry_batch']
    assert sum(r['pair_intersections'] for r in batches) == actual['pair_search']['pair_product']
    assert sum(r['passed_geometry'] for r in batches) == actual['line_diagnostics']['geometric_candidates']
    assert any(r['gates']['aspect']['sole_failure'] > 0 for r in batches)
    for batch in batches:
        rejected = batch['pair_intersections']-batch['passed_geometry']
        assert sum(g['first_failure'] for g in batch['gates'].values()) == rejected
        assert all(0 <= g['sole_failure'] <= g['failed'] for g in batch['gates'].values())
    batches[0]['gates']['area']['failed'] = -999
    assert actual['proposals'] == expected['proposals']


def test_observed_contour_failures_and_native_offsets_are_auditable():
    values = np.full((300, 380), 20.)
    # Two short-aspect rectangles and six panel-like rectangles. No labels input.
    for row in range(2):
        for column in range(3):
            y, x = 40+row*75, 70+column*37
            values[y:y+70, x:x+32] = 35
    values[210:250, 70:105] = 35
    values[210:250, 110:145] = 35
    window = values[10:280, 20:360]; records = []
    expected = _line_pass(window)
    before = values.copy()
    with trace_native_search(values, records.append):
        actual = _line_pass(window)
    assert actual == expected
    np.testing.assert_array_equal(values, before)
    contours = [r for r in records if r['event'] == 'contour_candidate']
    summaries = [r for r in records if r['event'] == 'contour_summary']
    assert len(summaries) == 1
    counts = summaries[0]['counts']
    assert counts['aspect_rejected'] > 0
    for stage in ('min_side_rejected', 'aspect_rejected', 'corner_or_fill_rejected',
                  'source_border_rejected', 'passed_geometry'):
        assert counts[stage] == sum(r['stage'] == stage for r in contours)
    partition = ('area_rejected', 'not_four_corners', 'min_side_rejected', 'aspect_rejected',
                 'corner_or_fill_rejected', 'source_border_rejected', 'passed_geometry')
    assert sum(counts[key] for key in partition) == counts['observed_contours']
    # Reported coordinates refer to the source, not the search-window origin.
    rejected = [r for r in contours if r['stage'] == 'aspect_rejected']
    assert any(r['failed_gates'] == ['aspect'] for r in rejected)
    assert any(np.linalg.norm(np.mean(r['polygon_native_xy'], axis=0)-[87, 230]) < 4 for r in rejected)


def test_sink_cannot_change_later_native_coordinates_through_pass_metadata(monkeypatch):
    setup_lines(monkeypatch)
    root = np.ones((150, 220)); window = root[10:130, 20:200]
    clean = []; changed = []
    with trace_native_search(root, clean.append):
        expected = _line_pass(window)
    def mutating_sink(event):
        changed.append(event)
        if event['event'] == 'pass_begin':
            event['window_xyxy'][:] = [9999, 9999, 9999, 9999]
    with trace_native_search(root, mutating_sink):
        actual = _line_pass(window)
    assert actual == expected
    assert [e for e in clean if e['event'] == 'candidate'] == [e for e in changed if e['event'] == 'candidate']
