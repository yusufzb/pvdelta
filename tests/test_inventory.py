import pytest

from pvdelta.inventory import (
    PanelAsset,
    SiteProfile,
    assets_to_geojson,
    load_inventory_csv,
    load_inventory_geojson,
    validate_panel_assets,
)


def test_four_row_inventory_and_geojson_export():
    assets = tuple(PanelAsset(f"P{i}", "T01", row, 1, 40.0 + i / 10000, 26.0, "EPSG:4326")
                   for i, row in enumerate("ABCD", 1))
    validation = validate_panel_assets(assets)
    assert validation.valid
    geojson = assets_to_geojson(assets)
    assert len(geojson["features"]) == 4
    assert geojson["features"][0]["properties"]["row"] == "A"


def test_inventory_rejects_duplicate_ids_bad_rows_and_missing_crs():
    validation = validate_panel_assets((
        PanelAsset("P01", "T01", "E", 0, 40.0, 26.0),
        PanelAsset("P01", "T01", "A", 1),
    ))
    assert not validation.valid
    assert any("duplicate" in error for error in validation.errors)
    assert any("row" in error for error in validation.errors)
    assert any("CRS" in error for error in validation.errors)


def test_inventory_does_not_treat_local_crs_as_wgs84():
    asset = PanelAsset("P01", "T01", "A", 1, 10.0, 20.0, "EPSG:32635")
    with pytest.raises(ValueError, match="WGS84"):
        assets_to_geojson((asset,))


def test_site_profile_encodes_known_four_row_site_facts():
    profile = SiteProfile()
    assert profile.rack_rows == 4
    assert profile.table_lengths == (9, 27)
    with pytest.raises(ValueError, match="four A-D"):
        SiteProfile(rack_rows=3)


def test_inventory_csv_loader_is_bom_safe_and_keeps_optional_coordinates(tmp_path):
    path = tmp_path / "inventory.csv"
    path.write_text(
        "panel_id,table_id,row,position,latitude,longitude,crs,string_id\n"
        "P01,T01,a,1,40,26,EPSG:4326,S01\n"
        "P02,T01,B,2,,,,\n",
        encoding="utf-8-sig",
    )
    assets = load_inventory_csv(path)
    assert assets[0].row == "A"
    assert assets[0].latitude == 40.0
    assert assets[1].latitude is None
    assert validate_panel_assets(assets).valid


def test_inventory_geojson_loader_requires_crs_and_accepts_point_features(tmp_path):
    path = tmp_path / "inventory.geojson"
    path.write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature",'
        '"geometry":{"type":"Point","coordinates":[26.1,40.1]},'
        '"properties":{"panel_id":"P01","table_id":"T01","row":"a",'
        '"position":1,"string_id":"S01"}}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source_crs"):
        load_inventory_geojson(path)
    assets = load_inventory_geojson(path, source_crs="EPSG:4326")
    assert assets == (PanelAsset("P01", "T01", "A", 1, 40.1, 26.1, "EPSG:4326", "S01"),)


def test_inventory_geojson_loader_does_not_infer_centres_from_polygons(tmp_path):
    path = tmp_path / "inventory.geojson"
    path.write_text(
        '{"type":"FeatureCollection","features":[{"type":"Feature",'
        '"geometry":{"type":"Polygon","coordinates":[]},"properties":{}}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Point"):
        load_inventory_geojson(path, source_crs="EPSG:4326")
