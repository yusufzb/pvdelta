import pytest

from pvdelta.inventory import PanelAsset
from pvdelta.matching import (
    ReviewedTopologyAnchor,
    load_topology_anchors_csv,
    match_observations_to_assets,
    match_reviewed_topology_anchors,
)
from pvdelta.models import LocationEstimate, LocationStatus, MatchStatus


def _asset(panel_id: str, latitude: float, longitude: float) -> PanelAsset:
    return PanelAsset(panel_id, "T01", "A", 1, latitude, longitude, "EPSG:4326")


def test_matching_uses_explicit_wgs84_distance_and_keeps_unlocated_observation_unmatched():
    results = match_observations_to_assets(
        (
            LocationEstimate("obs-1", LocationStatus.ESTIMATED, 40.0, 26.0, 2.0, "raycast"),
            LocationEstimate("obs-2", LocationStatus.NOT_ATTEMPTED),
        ),
        (_asset("P01", 40.0, 26.0),),
    )
    assert results[0].status is MatchStatus.MATCHED
    assert results[0].asset_id == "P01"
    assert results[1].status is MatchStatus.UNMATCHED


def test_matching_keeps_close_tie_ambiguous_and_rejects_local_crs():
    observation = LocationEstimate("obs-1", LocationStatus.ESTIMATED, 40.0, 26.00002, 1.0, "raycast")
    results = match_observations_to_assets(
        (observation,),
        (_asset("P01", 40.0, 26.0), _asset("P02", 40.0, 26.00004)),
        min_distance_margin_m=1.0,
    )
    assert results[0].status is MatchStatus.AMBIGUOUS
    assert results[0].candidate_asset_ids == ("P01", "P02")
    local = PanelAsset("P03", "T01", "A", 3, 40.0, 26.0, "EPSG:32635")
    unmatched = match_observations_to_assets((observation,), (local,))
    assert unmatched[0].status is MatchStatus.UNMATCHED


def test_matching_rejects_duplicate_observations_and_invalid_threshold():
    location = LocationEstimate("obs-1", LocationStatus.ESTIMATED, 40.0, 26.0, 1.0, "raycast")
    with pytest.raises(ValueError, match="duplicate observation_id"):
        match_observations_to_assets((location, location), (_asset("P01", 40.0, 26.0),))
    with pytest.raises(ValueError, match="max_distance_m"):
        match_observations_to_assets((location,), (_asset("P01", 40.0, 26.0),), max_distance_m=0)


def test_reviewed_topology_anchor_resolves_panel_without_inventing_visible_position():
    assets = (
        PanelAsset("P01", "T01", "D", 4),
        PanelAsset("P02", "T01", "D", 5),
    )
    anchor = ReviewedTopologyAnchor("obs-1", "T01", "d", 4, "CAD-review-001")
    result = match_reviewed_topology_anchors((anchor,), assets)
    assert result[0].status is MatchStatus.MATCHED
    assert result[0].asset_id == "P01"


def test_reviewed_topology_anchor_keeps_duplicate_key_ambiguous_and_missing_unmatched():
    assets = (
        PanelAsset("P01", "T01", "A", 1),
        PanelAsset("P01-duplicate", "T01", "A", 1),
    )
    anchors = (
        ReviewedTopologyAnchor("obs-1", "T01", "A", 1, "manual-review"),
        ReviewedTopologyAnchor("obs-2", "T01", "B", 1, "manual-review"),
    )
    result = match_reviewed_topology_anchors(anchors, assets)
    assert result[0].status is MatchStatus.AMBIGUOUS
    assert result[1].status is MatchStatus.UNMATCHED


def test_topology_anchor_csv_loader_is_bom_safe(tmp_path):
    path = tmp_path / "anchors.csv"
    path.write_text(
        "observation_id,table_id,row,position,evidence\nobs-1,T01,d,4,CAD-001\n",
        encoding="utf-8-sig",
    )
    anchors = load_topology_anchors_csv(path)
    assert anchors == (ReviewedTopologyAnchor("obs-1", "T01", "D", 4, "CAD-001"),)
