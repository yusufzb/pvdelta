"""Optional Stage 2 panel inventory contracts and CSV/GeoJSON-safe validation."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PanelAsset:
    """One surveyed/CAD panel record; identity is external to image coordinates."""

    panel_id: str
    table_id: str
    row: str
    position: int
    latitude: float | None = None
    longitude: float | None = None
    crs: str | None = None
    string_id: str | None = None


@dataclass(frozen=True)
class SiteProfile:
    """Editable site facts used to constrain, never invent, table topology."""

    module_length_m: float = 2.278
    module_width_m: float = 1.113
    module_orientation: str = "landscape"
    panel_tilt_deg: float = 25.0
    panel_azimuth_deg: float = 180.0
    rack_rows: int = 4
    table_lengths: tuple[int, ...] = (9, 27)

    def __post_init__(self) -> None:
        if self.module_length_m <= 0 or self.module_width_m <= 0:
            raise ValueError("module dimensions must be positive")
        if self.module_orientation not in {"portrait", "landscape"}:
            raise ValueError("module_orientation must be portrait or landscape")
        if not 0 <= self.panel_tilt_deg <= 90:
            raise ValueError("panel_tilt_deg must be between 0 and 90")
        if not 0 <= self.panel_azimuth_deg < 360:
            raise ValueError("panel_azimuth_deg must be in [0, 360)")
        if self.rack_rows != 4:
            raise ValueError("the current site profile requires four A-D rows")
        if not self.table_lengths or any(length not in {9, 27} for length in self.table_lengths):
            raise ValueError("table_lengths must contain 9 and/or 27")


@dataclass(frozen=True)
class InventoryValidation:
    valid: bool
    asset_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def validate_panel_assets(assets: list[PanelAsset] | tuple[PanelAsset, ...]) -> InventoryValidation:
    """Validate IDs, four-row topology and coordinate/CRS declarations."""
    items = tuple(assets)
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for asset in items:
        prefix = f"{asset.panel_id or '<missing panel_id>'}: "
        if not asset.panel_id.strip():
            errors.append(prefix + "panel_id is required")
        elif asset.panel_id in seen:
            errors.append(prefix + "duplicate panel_id")
        seen.add(asset.panel_id)
        if not asset.table_id.strip():
            errors.append(prefix + "table_id is required")
        if asset.row.upper() not in {"A", "B", "C", "D"}:
            errors.append(prefix + "row must be A, B, C or D")
        if asset.position < 1:
            errors.append(prefix + "position must be a positive visible/absolute index")
        coords = (asset.latitude, asset.longitude)
        if any(value is not None for value in coords):
            if any(value is None for value in coords):
                errors.append(prefix + "latitude and longitude must be supplied together")
            elif not asset.crs:
                errors.append(prefix + "CRS is required when coordinates are supplied")
            elif asset.crs.upper() in {"EPSG:4326", "WGS84", "CRS84"} and (
                not -90 <= float(asset.latitude) <= 90
                or not -180 <= float(asset.longitude) <= 180
            ):
                errors.append(prefix + "WGS84 coordinate is outside valid bounds")
            if not all(math.isfinite(float(value)) for value in coords if value is not None):
                errors.append(prefix + "coordinates must be finite")
        elif asset.crs:
            warnings.append(prefix + "CRS is present without coordinates")
        else:
            warnings.append(prefix + "coordinates are missing; location matching remains unavailable")
    return InventoryValidation(not errors, len(items), tuple(errors), tuple(warnings))


def load_inventory_csv(path: str | Path) -> tuple[PanelAsset, ...]:
    """Read a documented CSV without assuming missing coordinates are WGS84."""
    required = {"panel_id", "table_id", "row", "position"}
    assets: list[PanelAsset] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = required - fields
        if missing:
            raise ValueError(f"inventory CSV missing required columns: {sorted(missing)!r}")
        for line, row in enumerate(reader, start=2):
            try:
                assets.append(PanelAsset(
                    panel_id=(row.get("panel_id") or "").strip(),
                    table_id=(row.get("table_id") or "").strip(),
                    row=(row.get("row") or "").strip().upper(),
                    position=int(row["position"]),
                    latitude=_optional_float(row.get("latitude")),
                    longitude=_optional_float(row.get("longitude")),
                    crs=(row.get("crs") or "").strip() or None,
                    string_id=(row.get("string_id") or "").strip() or None,
                ))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid inventory CSV line {line}: {exc}") from exc
    return tuple(assets)


def load_inventory_geojson(
    path: str | Path,
    *,
    source_crs: str | None = None,
) -> tuple[PanelAsset, ...]:
    """Read panel-centre points from a GeoJSON FeatureCollection.

    The source CRS must be declared by the caller or in the legacy GeoJSON ``crs`` member.
    This adapter accepts Point features only; it does not infer panel centres from polygons
    and does not transform coordinates. Callers can pass the resulting assets to an explicit
    GIS transformation step before exporting them as WGS84.
    """
    with Path(path).open(encoding="utf-8-sig") as handle:
        try:
            document = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid inventory GeoJSON: {exc.msg}") from exc
    if document.get("type") != "FeatureCollection":
        raise ValueError("inventory GeoJSON must be a FeatureCollection")
    declared_crs = source_crs or _geojson_crs(document.get("crs"))
    if not declared_crs or not declared_crs.strip():
        raise ValueError("source_crs is required; coordinate CRS must be explicit")
    assets: list[PanelAsset] = []
    for index, feature in enumerate(document.get("features", ()), start=1):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError(f"invalid GeoJSON feature at index {index}")
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Point":
            raise ValueError(f"feature {index} must have a Point geometry")
        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 2:
            raise ValueError(f"feature {index} Point must contain [x, y]")
        properties = feature.get("properties") or {}
        try:
            assets.append(PanelAsset(
                panel_id=str(properties.get("panel_id") or "").strip(),
                table_id=str(properties.get("table_id") or "").strip(),
                row=str(properties.get("row") or "").strip().upper(),
                position=int(properties["position"]),
                latitude=float(coordinates[1]),
                longitude=float(coordinates[0]),
                crs=declared_crs.strip(),
                string_id=(str(properties["string_id"]).strip()
                           if properties.get("string_id") is not None else None),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid inventory GeoJSON feature {index}: {exc}") from exc
    validation = validate_panel_assets(assets)
    if validation.errors:
        raise ValueError("invalid inventory GeoJSON: " + "; ".join(validation.errors))
    return tuple(assets)


def assets_to_geojson(assets: list[PanelAsset] | tuple[PanelAsset, ...]) -> dict[str, Any]:
    """Export only explicit WGS84 coordinates; no CRS transformation is guessed."""
    validation = validate_panel_assets(assets)
    if validation.errors:
        raise ValueError("cannot export invalid inventory: " + "; ".join(validation.errors))
    features = []
    for asset in assets:
        if asset.latitude is None:
            continue
        if asset.crs is None or asset.crs.upper() not in {"EPSG:4326", "WGS84", "CRS84"}:
            raise ValueError(f"{asset.panel_id}: GeoJSON export requires explicit WGS84 CRS")
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [asset.longitude, asset.latitude]},
            "properties": {"panel_id": asset.panel_id, "table_id": asset.table_id,
                           "row": asset.row, "position": asset.position, "string_id": asset.string_id},
        })
    return {"type": "FeatureCollection", "features": features}


def _optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    result = float(value.strip().replace(",", "."))
    if not math.isfinite(result):
        raise ValueError("coordinate must be finite")
    return result


def _geojson_crs(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        properties = value.get("properties") or {}
        name = properties.get("name") or properties.get("href")
        return str(name) if name else None
    return None
