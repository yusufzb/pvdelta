from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from pvdelta.models import ThermalFrame
from pvdelta.native_completion_audit import compare_native_completion
from pvdelta.native_flight import analyze_native_frame


def quad(identifier, x):
    return {'proposal_id': identifier,
            'polygon_native_xy': [[x, 5], [x+20, 5], [x+20, 55], [x, 55]]}


@pytest.fixture
def sample():
    values = np.full((64, 100), 25., np.float32)
    values[20:24, 10:14] = 35.
    frame = ThermalFrame('frame', 'original.JPG', '0'*64, values, None, 100, 64)
    base = {'proposals': [quad('N001', 5), quad('N002', 30)]}
    candidate = deepcopy(base)
    candidate['proposals'].append(quad('E0001', 55))
    candidate['transition_completion'] = {'retained_additions': 1, 'pool_overflow': False,
                                          'decision_counts': {}}
    _, saved = analyze_native_frame(frame, proposal_builder=lambda *a, **k: deepcopy(base))
    return frame, base, candidate, saved


def compare(sample):
    frame, base, candidate, saved = sample
    return compare_native_completion(frame, saved,
        baseline_builder=lambda *a, **k: deepcopy(base),
        completion_builder=lambda *a, **k: deepcopy(candidate))


def test_exact_masks_native_values_and_intra_module_candidate_survive(sample):
    original = sample[0].temperature_c.copy()
    result = compare(sample)
    assert result['baseline_count'] == 2 and result['addition_count'] == 1
    assert result['baseline_geometry_and_masks_preserved']
    assert result['baseline_findings'] >= 1
    assert result['native_values_preserved'] and not result['measurement_resampled']
    assert result['physical_panel_count'] is None and result['anomaly_accuracy'] is None
    np.testing.assert_array_equal(original, sample[0].temperature_c)


def test_wrong_capture_or_engine_baseline_withheld(sample):
    sample[1]['proposals'][0]['polygon_native_xy'][0][0] += 1
    with pytest.raises(ValueError, match='saved capture geometry'):
        compare(sample)


def test_baseline_mask_hash_mismatch_is_not_treated_as_preserved(sample):
    sample[3]['modules'][0]['mask_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='saved measurement support'):
        compare(sample)


def test_candidate_cannot_move_old_polygon(sample):
    sample[2]['proposals'][0]['polygon_native_xy'][0][0] += 1
    with pytest.raises(ValueError, match='changed baseline IDs or geometry'):
        compare(sample)


def test_shared_new_envelope_cannot_remove_old_pixels(sample):
    sample[2]['proposals'][-1] = quad('E0001', 12)
    with pytest.raises(ValueError, match='changed baseline measurement pixels'):
        compare(sample)


def test_report_change_is_not_hidden_behind_mask_preservation(sample, monkeypatch):
    import pvdelta.native_completion_audit as module

    original = module.analyze_native_frame
    calls = []

    def changed(*args, **kwargs):
        reports, diagnostic = original(*args, **kwargs)
        calls.append(True)
        if len(calls) == 2:
            reports = (replace(reports[0], reasons=('new peer changed interpretation',)), *reports[1:])
        return reports, diagnostic

    monkeypatch.setattr(module, 'analyze_native_frame', changed)
    result = compare(sample)
    assert result['baseline_reports_changed'] == ['N001']
    assert result['baseline_geometry_and_masks_preserved']


def test_pool_overflow_is_explicit_and_all_baseline_reports_still_compared(sample):
    sample[2]['proposals'].pop()
    sample[2]['transition_completion'].update(retained_additions=0, pool_overflow=True)
    result = compare(sample)
    assert result['pool_overflow'] and result['addition_count'] == 0
    assert result['baseline_reports_changed'] == []
    assert result['baseline_report_sha256'] == result['completion_report_sha256']


def test_forged_addition_counter_withheld(sample):
    sample[2]['transition_completion']['retained_additions'] = 2
    with pytest.raises(ValueError, match='addition count'):
        compare(sample)


def test_resume_is_bound_to_exact_plan_capture_and_content(tmp_path):
    from pvdelta.native_completion_audit import read_audit_record, save_audit_record

    path = tmp_path/'checkpoint.json'
    result = {'status': 'completed', 'addition_count': 1}
    save_audit_record(path, 'plan-A', 'capture-A', result)
    assert read_audit_record(path, 'plan-A', 'capture-A') == result
    for plan, capture in [('plan-B', 'capture-A'), ('plan-A', 'capture-B')]:
        with pytest.raises(ValueError, match='identity or digest'):
            read_audit_record(path, plan, capture)
    with pytest.raises(FileExistsError):
        save_audit_record(path, 'plan-A', 'capture-A', result)
    path.write_text(path.read_text().replace('"addition_count": 1', '"addition_count": 9'))
    with pytest.raises(ValueError, match='identity or digest'):
        read_audit_record(path, 'plan-A', 'capture-A')


def test_failed_and_truncated_resume_records_are_not_silently_successful(tmp_path):
    from pvdelta.native_completion_audit import read_audit_record, save_audit_record

    path = tmp_path/'checkpoint.json'
    save_audit_record(path, 'plan', 'capture', {'status': 'failed', 'error': 'SDK failed'})
    assert read_audit_record(path, 'plan', 'capture')['status'] == 'failed'
    path.write_text('{"payload":')
    with pytest.raises(ValueError, match='Incomplete or invalid'):
        read_audit_record(path, 'plan', 'capture')
