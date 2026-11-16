import numpy as np
import pytest

from pvdelta.masking import build_measurement_support, summarize_temperature
from pvdelta.support_partition import partition_measurement_support


def envelopes():
    first = np.zeros((20, 30), bool)
    second = first.copy()
    first[2:18, 2:17] = True
    second[2:18, 13:28] = True
    return [first, second]


def test_shared_pixels_are_unassigned_to_all_instances_without_shifting_boundaries():
    masks = envelopes()
    first, second = partition_measurement_support(masks, margin_px=1)
    shared = masks[0] & masks[1]
    for source, support in zip(masks, (first, second)):
        np.testing.assert_array_equal(support.envelope, source)
        assert not np.any(support.interior & shared)
        assert support.excluded_ambiguous.any()
        assert not support.interior.flags.writeable
        np.testing.assert_array_equal(support.interior | support.excluded_edge, source)
    assert not np.any(first.interior & second.interior)
    with pytest.raises(ValueError, match='overlap'):
        build_measurement_support(masks, margin_px=1)


def test_unique_hotspot_remains_and_ambiguous_hotspot_stays_in_source_not_zero_degrees():
    masks = envelopes()
    support = partition_measurement_support(masks, margin_px=1)[0]
    temperature = np.full(masks[0].shape, 25.)
    temperature[9, 7], temperature[9, 14] = 90., 100.
    before = temperature.copy()
    masked = support.masked_temperature(temperature)
    assert masked[9, 7] == 90
    assert np.isnan(masked[9, 14])
    assert summarize_temperature(temperature, support).max_c == 90
    np.testing.assert_array_equal(temperature, before)
    masks[0][:] = False
    assert support.envelope.any()


def test_identical_envelopes_keep_empty_support_explicit():
    mask = np.ones((10, 10), bool)
    supports = partition_measurement_support([mask, mask], margin_px=1)
    assert all(not s.interior.any() for s in supports)
    assert all(summarize_temperature(np.ones((10, 10)), s).status == 'unavailable' for s in supports)


def test_disjoint_partition_matches_strict_builder():
    masks = [np.eye(12, dtype=bool), np.fliplr(np.eye(12, dtype=bool))]
    strict = build_measurement_support(masks, margin_px=0)
    partitioned = partition_measurement_support(masks, margin_px=0)
    for before, after in zip(strict, partitioned):
        np.testing.assert_array_equal(before.interior, after.interior)
        assert not after.excluded_ambiguous.any()


@pytest.mark.parametrize('margin', [-1, True, 1.5])
def test_partition_margin_is_not_coerced(margin):
    with pytest.raises(ValueError):
        partition_measurement_support([], margin_px=margin)


@pytest.mark.parametrize('masks', [[np.ones((3, 3))], [np.zeros((3, 3), bool)],
    [np.ones((3, 3), bool), np.ones((4, 4), bool)]])
def test_partition_rejects_wrong_mask_contracts(masks):
    with pytest.raises(ValueError):
        partition_measurement_support(masks, margin_px=0)
