"""Conservative Stage 2 matching of reviewed geographic observations to panel assets."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pvdelta.inventory import PanelAsset, validate_panel_assets
from pvdelta.models import AssetMatch, LocationEstimate, LocationStatus, MatchStatus

_WGS84_NAMES = {"EPSG:4326", "WGS84", "CRS84"}
_EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class MatchCandidate:
    """A ranked candidate retained for audit; score is not a probability."""

    asset_id: str
    distance_m: float
    ranking_score: float


@dataclass(frozen=True)
class ReviewedTopologyAnchor:
    """A human/CAD-reviewed link from a frame observation to table topology."""

    observation_id: str
    table_id: str
    row: str
    position: int
    evidence: str

    def __post_init__(self) -> None:
        if not self.observation_id.strip() or not self.table_id.strip():
            raise ValueError("observation_id and table_id are required")
        if self.row.upper() not in {"A", "B", "C", "D"}:
            raise ValueError("row must be A, B, C or D")
        if self.position < 1:
            raise ValueError("position must be positive")
        if not self.evidence.strip():
            raise ValueError("evidence is required for a reviewed topology anchor")


def load_topology_anchors_csv(path: str | Path) -> tuple[ReviewedTopologyAnchor, ...]:
    """Load an explicit reviewed anchor manifest without inferring physical positions."""
    required = {"observation_id", "table_id", "row", "position", "evidence"}
    anchors: list[ReviewedTopologyAnchor] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"anchor CSV missing required columns: {sorted(missing)!r}")
        for line, row in enumerate(reader, start=2):
            try:
                anchors.append(ReviewedTopologyAnchor(
                    observation_id=(row.get("observation_id") or "").strip(),
                    table_id=(row.get("table_id") or "").strip(),
                    row=(row.get("row") or "").strip().upper(),
                    position=int(row["position"]),
                    evidence=(row.get("evidence") or "").strip(),
                ))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid anchor CSV line {line}: {exc}") from exc
    return tuple(anchors)


def match_observations_to_assets(
    observations: Iterable[LocationEstimate],
    assets: Iterable[PanelAsset],
    *,
    max_distance_m: float = 5.0,
    min_distance_margin_m: float = 0.5,
) -> tuple[AssetMatch, ...]:
    """Match explicit WGS84 observation estimates to nearby panel centres.

    The matcher is intentionally geographic and conservative. Observations without a
    validated estimated location, assets without explicit WGS84 coordinates, and candidates
    outside ``max_distance_m`` remain unmatched. A close tie remains ambiguous instead of
    being resolved by list order.
    """
    _validate_threshold("max_distance_m", max_distance_m, positive=True)
    _validate_threshold("min_distance_margin_m", min_distance_margin_m, positive=False)
    asset_items = tuple(assets)
    validation = validate_panel_assets(asset_items)
    if validation.errors:
        raise ValueError("cannot match invalid panel inventory: " + "; ".join(validation.errors))
    usable_assets = tuple(
        asset for asset in asset_items
        if asset.latitude is not None
        and asset.crs is not None
        and asset.crs.upper() in _WGS84_NAMES
    )
    results: list[AssetMatch] = []
    seen_observations: set[str] = set()
    for observation in observations:
        if observation.observation_id in seen_observations:
            raise ValueError(f"duplicate observation_id: {observation.observation_id}")
        seen_observations.add(observation.observation_id)
        if (
            observation.status is not LocationStatus.ESTIMATED
            or observation.latitude is None
            or observation.longitude is None
        ):
            results.append(AssetMatch(observation.observation_id, MatchStatus.UNMATCHED))
            continue
        candidates = sorted(
            (
                MatchCandidate(
                    asset.panel_id,
                    _haversine_m(
                        observation.latitude,
                        observation.longitude,
                        asset.latitude,
                        asset.longitude,
                    ),
                    0.0,
                )
                for asset in usable_assets
            ),
            key=lambda item: (item.distance_m, item.asset_id),
        )
        candidates = tuple(
            MatchCandidate(
                candidate.asset_id,
                candidate.distance_m,
                max(0.0, 1.0 - candidate.distance_m / max_distance_m),
            )
            for candidate in candidates
            if candidate.distance_m <= max_distance_m
        )
        candidate_ids = tuple(candidate.asset_id for candidate in candidates)
        if not candidates:
            results.append(AssetMatch(observation.observation_id, MatchStatus.UNMATCHED))
        elif len(candidates) == 1 or (
            candidates[1].distance_m - candidates[0].distance_m >= min_distance_margin_m
        ):
            best = candidates[0]
            results.append(AssetMatch(
                observation.observation_id,
                MatchStatus.MATCHED,
                best.asset_id,
                best.ranking_score,
                candidate_ids,
            ))
        else:
            results.append(AssetMatch(
                observation.observation_id,
                MatchStatus.AMBIGUOUS,
                ranking_score=candidates[0].ranking_score,
                candidate_asset_ids=candidate_ids,
            ))
    return tuple(results)


def match_reviewed_topology_anchors(
    anchors: Iterable[ReviewedTopologyAnchor],
    assets: Iterable[PanelAsset],
) -> tuple[AssetMatch, ...]:
    """Resolve reviewed table/row/position anchors without inferring topology.

    The anchor is the evidence boundary: visible-frame indices are not accepted as physical
    positions unless an upstream review has recorded the table, row and position explicitly.
    Duplicate inventory keys remain ambiguous instead of being resolved by input order.
    """
    asset_items = tuple(assets)
    validation = validate_panel_assets(asset_items)
    if validation.errors:
        raise ValueError("cannot match invalid panel inventory: " + "; ".join(validation.errors))
    by_topology: dict[tuple[str, str, int], list[PanelAsset]] = {}
    for asset in asset_items:
        key = (asset.table_id, asset.row.upper(), asset.position)
        by_topology.setdefault(key, []).append(asset)
    results: list[AssetMatch] = []
    seen_observations: set[str] = set()
    for anchor in anchors:
        if anchor.observation_id in seen_observations:
            raise ValueError(f"duplicate observation_id: {anchor.observation_id}")
        seen_observations.add(anchor.observation_id)
        candidates = tuple(sorted(
            by_topology.get((anchor.table_id, anchor.row.upper(), anchor.position), ()),
            key=lambda asset: asset.panel_id,
        ))
        candidate_ids = tuple(asset.panel_id for asset in candidates)
        if len(candidates) == 1:
            results.append(AssetMatch(
                anchor.observation_id,
                MatchStatus.MATCHED,
                candidates[0].panel_id,
                1.0,
                candidate_ids,
                anchor.evidence,
            ))
        elif candidates:
            results.append(AssetMatch(
                anchor.observation_id,
                MatchStatus.AMBIGUOUS,
                ranking_score=1.0,
                candidate_asset_ids=candidate_ids,
                evidence=anchor.evidence,
            ))
        else:
            results.append(AssetMatch(
                anchor.observation_id,
                MatchStatus.UNMATCHED,
                evidence=anchor.evidence,
            ))
    return tuple(results)


def _validate_threshold(name: str, value: float, *, positive: bool) -> None:
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        comparison = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {comparison}")


def _haversine_m(latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float) -> float:
    lat_a, lat_b = math.radians(latitude_a), math.radians(latitude_b)
    delta_lat = lat_b - lat_a
    delta_lon = math.radians(longitude_b - longitude_a)
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2) ** 2
    return _EARTH_RADIUS_M * 2 * math.asin(math.sqrt(min(1.0, a)))
