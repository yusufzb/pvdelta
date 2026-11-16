from copy import deepcopy

import numpy as np
import pytest

from pvdelta.models import ThermalFrame
from pvdelta.native_gap_review import NativeGapReview


def test_export_reconstructs_saved_native_profile_and_withholds_changed_geometry(tmp_path, monkeypatch):
    import hashlib
    from dataclasses import asdict, replace

    from pvdelta import decode, native_windows
    from pvdelta.decode import MeasurementParams
    from pvdelta.gap_review_bundle import load_gap_bundle
    from pvdelta.measurement_profile import frame_measurement_profile
    from pvdelta.native_gap_review import export_native_frame_gaps
    from pvdelta.native_search_trace import begin_trace

    frame, base, event = setup()
    source = tmp_path/'source_T.JPG'
    source.write_bytes(b'explicit synthetic SDK source')
    params = MeasurementParams(5., 50., .95, 20., 20.)
    frame = replace(frame, source_path=str(source), source_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
        raw_metadata={'measurement_params': asdict(params), 'recorded_measurement_params': asdict(params),
                      'sdk_parameter_readback_verified': True})
    profile = frame_measurement_profile(frame)

    class FakeSDK:
        measurement_params = params

        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def temperature(self):
            return frame.temperature_c

    def propose(values, **kwargs):
        trace = begin_trace(values, support_mode='hybrid', local_contrast=False, pair_budget=250000)
        trace.candidate('side_support', np.array(event['polygon_native_xy']), event['support'])
        return deepcopy(base)

    monkeypatch.setattr(decode, 'RJpeg', FakeSDK)
    monkeypatch.setattr(native_windows, 'native_window_proposals', propose)
    diagnostic = {'geometry_method': 'native_observed_boundaries', 'native_geometry': base}
    folder, report = export_native_frame_gaps(source, diagnostic, profile, tmp_path/'export')
    assert len(report['evidence']) == 1 and not report['production_enabled']
    assert report['measurement_profile_id'] == profile['profile_id']
    assert load_gap_bundle(folder)[0]['parameters'] == profile
    monkeypatch.setattr(native_windows, 'native_window_proposals', lambda *a, **k: {'proposals': []})
    with pytest.raises(ValueError, match='engine or proposal geometry changed'):
        export_native_frame_gaps(source, diagnostic, profile, tmp_path/'stale')
    assert not (tmp_path/'stale').exists()


def setup():
    values=np.full((140,240),25.,np.float32)
    values[53:57,68:72]=35.
    values[12:16,12:16]=50.  # warmer ground outside the structural gap
    frame=ThermalFrame('frame','sample_T.JPG','source-hash',values,None,240,140)
    def panel(name,q):return {'proposal_id':name,'polygon_native_xy':q,'source_method':'closed_contour','threshold_votes':3}
    base={'proposals':[panel('right',[[100,40],[160,40],[160,70],[100,70]]),
                       panel('below',[[40,70],[100,70],[100,100],[40,100]])]}
    event={'event':'candidate','pass_id':1,'stage':'side_support',
           'polygon_native_xy':[[40,40],[100,40],[100,70],[40,70]],
           'support':[[False,.4,.6],[True,1.,0.],[True,1.,0.],[False,.5,.5]]}
    return frame,base,event


def begin(collector,mode='hybrid'):
    collector({'event':'pass_begin','pass_id':1,'support_mode':mode,'window_xyxy':[0,0,240,140]})


def test_enclosed_unresolved_gap_preserves_hot_native_evidence_and_excludes_hotter_ground():
    frame,base,event=setup();old=frame.temperature_c.copy();snapshot=deepcopy(base)
    collector=NativeGapReview(frame,base);begin(collector);collector(event)
    result=collector.evaluate()
    assert len(result['evidence'])==1
    evidence=result['evidence'][0];finding=evidence['native_finding']
    assert evidence['module_identity'] is None and evidence['panel_location'] is None
    assert not evidence['confirmed_fault'] and not evidence['measurement_authorised']
    assert evidence['status']=='unassigned_thermal_evidence_review_required'
    assert finding['n_region_px']==16 and abs(finding['delta_t_c']-10)<1e-4
    assert finding['absolute_t_c']==35
    assert set(map(tuple,finding['region_pixels_yx']))=={(y,x) for y in range(53,57) for x in range(68,72)}
    assert base==snapshot
    np.testing.assert_array_equal(frame.temperature_c,old)


def test_neighbours_on_only_one_axis_do_not_form_a_panel_context():
    frame,base,event=setup();base['proposals'][1]['polygon_native_xy']=[[100,70],[160,70],[160,100],[100,100]]
    collector=NativeGapReview(frame,base);begin(collector);collector(event)
    assert collector.evaluate()['evidence']==[] and collector.candidates=={}


@pytest.mark.parametrize('mode',['segments',None])
def test_only_actual_hybrid_side_rejections_are_consumed(mode):
    frame,base,event=setup();collector=NativeGapReview(frame,base)
    if mode is not None:begin(collector,mode)
    collector(event);assert collector.evaluate()['evidence']==[]


def test_baseline_overlap_does_not_create_an_extra_panel_or_evidence():
    frame,base,event=setup();event['polygon_native_xy']=[[100,40],[160,40],[160,70],[100,70]]
    collector=NativeGapReview(frame,base);begin(collector);collector(event)
    assert collector.evaluate()['evidence']==[]


def test_empty_baseline_does_not_scan_unassigned_hot_vegetation():
    frame,_base,event=setup();collector=NativeGapReview(frame,{'proposals':[]});begin(collector);collector(event)
    assert collector.evaluate()['evidence']==[]


def test_zero_hot_evidence_is_not_a_healthy_panel_record():
    frame,base,event=setup();frame.temperature_c.fill(25)
    collector=NativeGapReview(frame,base);begin(collector);collector(event);result=collector.evaluate()
    assert result['evidence']==[] and len(result['analyses'])==1
    assert 'health assessment' in result['scope'] and 'healthy_count' not in result


def test_budget_evictions_are_explicit_and_hypotheses_never_rank_by_temperature():
    frame,base,event=setup();collector=NativeGapReview(frame,base,max_hypotheses=2,max_analyses=1);begin(collector)
    for shift in (0,.1,.2):
        e=deepcopy(event);e['polygon_native_xy']=(np.asarray(e['polygon_native_xy'])+[shift,0]).tolist();collector(e)
    result=collector.evaluate()
    assert result['stored_hypotheses']==2 and result['counters']['budget_evictions']==1
    assert len(result['evidence'])==1


def test_repeated_geometry_does_not_duplicate_native_evidence():
    frame,base,event=setup();collector=NativeGapReview(frame,base);begin(collector)
    collector(event);collector(deepcopy(event))
    assert len(collector.evaluate()['evidence'])==1
    assert collector.evaluate()['evidence'][0]['evidence_id']==collector.evaluate()['evidence'][0]['evidence_id']


def test_out_of_frame_rejected_geometry_is_skipped_without_aborting_trace():
    frame,base,event=setup();event['polygon_native_xy']=(np.asarray(event['polygon_native_xy'])-[60,0]).tolist()
    collector=NativeGapReview(frame,base);begin(collector);collector(event)
    assert collector.evaluate()['evidence']==[]


@pytest.mark.parametrize('change',[lambda e:e.update(support=[[True,.2,.8]]*4),
                                  lambda e:e.update(support=[[False,float('nan'),.5]]*4),
                                  lambda e:e.update(polygon_native_xy=[[0,0]]),
                                  lambda e:e.update(support=[[False,.5,.5]]*3)])
def test_corrupt_event_is_rejected(change):
    frame,base,event=setup();collector=NativeGapReview(frame,base);begin(collector);change(event)
    with pytest.raises(ValueError):collector(event)


@pytest.mark.parametrize('kwargs',[{'max_hypotheses':False},{'max_hypotheses':0},{'max_analyses':257},
                                  {'max_hypotheses':1,'max_analyses':2}])
def test_invalid_budgets_fail_closed(kwargs):
    frame,base,_event=setup()
    with pytest.raises(ValueError):NativeGapReview(frame,base,**kwargs)


def test_changed_native_source_aborts_review():
    frame,base,_event=setup();collector=NativeGapReview(frame,base);frame.temperature_c[0,0]+=1
    with pytest.raises(RuntimeError,match='changed'):collector.evaluate()


def test_nonfinite_native_source_is_rejected():
    frame,base,_event=setup();frame.temperature_c[0,0]=np.nan
    with pytest.raises(ValueError):NativeGapReview(frame,base)


def test_machine_scale_interval_roundoff_does_not_abort_a_valid_trace():
    frame,base,event=setup();event['support'][1][1]=1+1e-12
    collector=NativeGapReview(frame,base);begin(collector);collector(event)
    assert len(collector.evaluate()['evidence'])==1


def test_materially_invalid_coverage_remains_rejected():
    frame,base,event=setup();event['support'][1][1]=1.0001
    collector=NativeGapReview(frame,base);begin(collector)
    with pytest.raises(ValueError):collector(event)
