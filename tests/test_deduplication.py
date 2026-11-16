import pytest

from pvdelta.deduplication import ObservationPoint, cluster_observations


def test_same_capture_neighbours_are_not_merged():
    clusters = cluster_observations(
        [
            ObservationPoint("a", "frame-1", 40.0, 26.0),
            ObservationPoint("b", "frame-1", 40.0, 26.000004),
        ],
        radius_m=1.0,
    )
    assert len(clusters) == 2


def test_transitive_chain_cannot_merge_two_points_from_one_capture():
    clusters = cluster_observations(
        [
            ObservationPoint("a", "frame-1", 40.0, 26.0),
            ObservationPoint("b", "frame-2", 40.0, 26.000004),
            ObservationPoint("c", "frame-1", 40.0, 26.000008),
        ],
        radius_m=1.0,
    )
    assert sorted(len(cluster.observation_ids) for cluster in clusters) == [1, 2]


def test_different_captures_within_tolerance_are_merged():
    clusters = cluster_observations(
        [
            ObservationPoint("a", "frame-1", 40.0, 26.0),
            ObservationPoint("b", "frame-2", 40.0, 26.000004),
        ],
        radius_m=1.0,
    )
    assert len(clusters) == 1
    assert clusters[0].observation_ids == ("a", "b")
    assert clusters[0].capture_count == 2


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        cluster_observations([], radius_m=0)
    with pytest.raises(ValueError):
        cluster_observations([ObservationPoint("a", "f", float("nan"), 26.0)])
