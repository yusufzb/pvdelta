from dataclasses import asdict

import numpy as np
import pytest

from pvdelta.native_boundaries import NativeBoundaryConfig
from pvdelta.native_completion_search import supported_completion_proposals


def test_both_searches_use_exact_same_native_grid_and_change_only_aspect(monkeypatch):
    import pvdelta.native_completion_search as search

    values = np.arange(160*340, dtype=np.float32).reshape(160, 340)
    before = values.copy()
    calls = []

    def geometry(grid, **kwargs):
        assert grid is values and np.shares_memory(grid, values)
        calls.append(kwargs)
        return {'proposals': [], 'measurement_authorised': False}

    monkeypatch.setattr(search, 'native_window_proposals', geometry)
    cfg = NativeBoundaryConfig(min_side_px=22, min_neighbours=2)
    result = supported_completion_proposals(values, config=cfg, raw_supplement=False)
    assert len(calls) == 2
    first, second = [asdict(c['config']) for c in calls]
    assert first == asdict(cfg)
    assert {k for k in first if first[k] != second[k]} == {'min_aspect'}
    assert second['min_aspect'] == 1.3
    assert all(c['raw_supplement'] is False for c in calls)
    np.testing.assert_array_equal(values, before)
    assert not result['native_completion_search']['measurement_authorised']
    assert not result['native_completion_search']['hotspot_efficacy_accepted']


def test_existing_wide_configuration_does_not_run_a_redundant_search(monkeypatch):
    import pvdelta.native_completion_search as search

    calls = []

    def geometry(grid, **kwargs):
        calls.append(kwargs)
        return {'proposals': []}

    monkeypatch.setattr(search, 'native_window_proposals', geometry)
    result = supported_completion_proposals(np.ones((80, 80)), config=NativeBoundaryConfig(min_aspect=1.2))
    assert len(calls) == 1 and not result['native_completion_search']['second_search_needed']


def test_mutated_source_fails_instead_of_returning_a_preview(monkeypatch):
    import pvdelta.native_completion_search as search

    def broken(grid, **kwargs):
        grid[0, 0] += 1
        return {'proposals': []}

    monkeypatch.setattr(search, 'native_window_proposals', broken)
    with pytest.raises(RuntimeError, match='changed'):
        supported_completion_proposals(np.ones((80, 80)))


@pytest.mark.parametrize('values', [np.ones(10), np.ones((0, 10)), [[float('nan')]],
                                  [[float('inf')]], [['not temperature']], [[True]]])
def test_invalid_grids_fail_before_search(values):
    with pytest.raises(ValueError, match='native 2D'):
        supported_completion_proposals(values)


@pytest.mark.parametrize('kwargs', [{'margin_px': True}, {'margin_px': -1}, {'raw_supplement': 'yes'},
                                  {'config': 'wide'}, {'config': False}, {'config': {}}])
def test_invalid_preview_settings_are_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        supported_completion_proposals(np.ones((80, 80)), **kwargs)
