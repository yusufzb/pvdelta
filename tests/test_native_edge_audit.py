import hashlib

import numpy as np
import pytest

from pvdelta.native_edge_audit import audit_native_edges, edge_audit_identity


def inspect(values, quad=None, expected=None):
    return audit_native_edges(values, quad or [[60, 45], [160, 45], [160, 95], [60, 95]],
        source_sha256='1'*64,
        native_temperature_sha256=expected or hashlib.sha256(values.tobytes()).hexdigest())


def test_four_observed_ridges_are_diagnostic_only_and_preserve_native_values():
    y, x = np.mgrid[:150, :220]
    distance = np.minimum.reduce([abs(x-60), abs(x-160), abs(y-45), abs(y-95)])
    values = (30-4*np.exp(-.5*(distance/.7)**2)).astype(np.float32)
    before = values.copy()
    report = inspect(values)
    assert all(s['native_ridge_supported'] for s in report['sides'])
    assert [s['side'] for s in report['sides']] == list('ABCD')
    assert not any(report[k] for k in ('measurement_authorised', 'mask_generated', 'database_written'))
    assert report['semantic_panel_membership'] == 'unverified'
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize('kind', ['flat', 'step', 'hotspot', 'slope'])
def test_unknown_steps_or_hot_pixels_are_not_approved_boundaries_or_healthy(kind):
    y, x = np.mgrid[:150, :220]
    values = np.full((150, 220), 25.)
    if kind == 'step': values += 5*(y > 45)
    if kind == 'hotspot': values[70:75, 100:105] += 15
    if kind == 'slope': values += .01*x+.01*y
    report = inspect(values)
    assert not any(s['native_ridge_supported'] for s in report['sides'])
    assert 'healthy_count' not in report and 'mask' not in report
    assert report['mask_generated'] is False


def test_missing_source_flanks_do_not_turn_padding_into_support():
    values = np.full((150, 220), 25.)
    report = inspect(values, [[1, 1], [100, 1], [100, 51], [1, 51]])
    assert report['sides'][0]['valid_samples'] == 0
    assert report['sides'][0]['coverage'] == 0


@pytest.mark.parametrize('quad', [[[0, 0]]*4, [[-1, 2], [30, 2], [30, 40], [2, 40]],
                                  [[60, 45], [160, 95], [160, 45], [60, 95]]])
def test_invalid_quad_fails(quad):
    with pytest.raises(ValueError):
        inspect(np.full((150, 220), 25.), quad)


def test_changed_native_hash_and_nonfinite_source_fail_closed():
    values = np.full((150, 220), 25.)
    with pytest.raises(ValueError, match='SHA256 differs'):
        inspect(values, expected='0'*64)
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match='Finite native'):
        inspect(values)


def test_diagnostic_identity_binds_geometry_engine_and_native_bytes():
    values = np.full((150, 220), 25.)
    first = inspect(values)
    second = inspect(values, [[61, 45], [160, 45], [160, 95], [61, 95]])
    assert first['audit_token'] != second['audit_token']
    assert 'edge_support.py' in first['engine_source_sha256']


def test_native_mutation_during_audit_is_detected(monkeypatch):
    from pvdelta import native_edge_audit

    values = np.full((150, 220), 25.)
    original = native_edge_audit.native_edge_continuity
    def mutate(array, a, b):
        result = original(array, a, b)
        array[0, 0] += 1
        return result
    monkeypatch.setattr(native_edge_audit, 'native_edge_continuity', mutate)
    with pytest.raises(RuntimeError, match='changed during'):
        inspect(values)


def test_incorrect_hash_identity_types_fail():
    with pytest.raises(ValueError, match='SHA256 identities'):
        edge_audit_identity(np.full((150, 220), 25.), [[60, 45], [160, 45], [160, 95], [60, 95]],
                            source_sha256='not-a-hash', native_temperature_sha256='1'*64)
