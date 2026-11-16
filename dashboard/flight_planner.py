"""Streamlit flight-planning and local inspection-data dashboard for pvdelta."""

from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import ExifTags, Image

from pvdelta.camera_db import get_platform, list_platforms
from pvdelta.coverage_audit import build_coverage_audit
from pvdelta.coverage_ui import coverage_audit_view
from pvdelta.finding_preview import DISPLAY_OUTLINE_SCALE
from pvdelta.flight import FlightPlanningError, plan_flight
from pvdelta.flight_scan import scan_radiometric_flight
from pvdelta.map_route import build_route_segments, simplify_route
from pvdelta.mask_material_review_ui import material_review_view
from pvdelta.native_flight_ui import native_flight_view
from pvdelta.review_ui import finding_navigation, repeat_view_review
from pvdelta.surface_support import check_rgb_surface

MODULE_PRESETS = {
    "144 half-cell / M10": (2.278, 1.113, 91.0, 144, "M10", "half-cut"),
    "120 half-cell": (1.75, 1.04, 83.0, 120, None, "half-cut"),
    "60 full-cell": (1.65, 0.99, 156.0, 60, None, "full-cell"),
    "Custom": (2.278, 1.113, 91.0, 144, "M10", "half-cut"),
}

OVERLAP_PRESETS = {
    "Detailed mapping (recommended)": (80.0, 70.0),
    "Fast screening": (70.0, 20.0),
    "Custom": (80.0, 70.0),
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_WORKSPACE = PROJECT_ROOT.parent.parent / 'knowledge basis'
PRIVATE_WORKSPACE = (
    LEGACY_WORKSPACE
    if PROJECT_ROOT.parent.name.casefold() == 'github' and LEGACY_WORKSPACE.is_dir()
    else PROJECT_ROOT / 'data'
)
SURFACE_MANIFEST = Path(os.environ.get('PVDELTA_SURFACE_MANIFEST')
                        or PRIVATE_WORKSPACE / 'surface-recovery' / 'manifest.json')
DEFAULT_DATABASE_PATH = Path(os.environ.get('PVDELTA_DATABASE_PATH') or
                             PROJECT_ROOT / "data" / "pvdelta.duckdb")
DEFAULT_SDK_LIBRARY_PATH = (
    PROJECT_ROOT
    / "sdk"
    / "tsdk-core"
    / "lib"
    / "windows"
    / "release_x64"
)


def _render_repeat_view_review(database_path: Path, finding_id: str, *, key: str, sdk_library=None) -> None:
    """Render repeat-view evidence across hot-reloaded review_ui versions.

    Streamlit keeps imported modules alive between reruns.  During development a
    dashboard rerun can therefore hold an older ``repeat_view_review`` function
    whose signature predates the optional SDK argument.  Inspect the live
    callable before invoking it so a stale module cannot turn the review screen
    into a TypeError.  The fallback sets the SDK path for the older callable via
    the decoder's supported environment variable.
    """
    sdk_library = Path(sdk_library) if sdk_library else DEFAULT_SDK_LIBRARY_PATH
    parameters = inspect.signature(repeat_view_review).parameters
    if "sdk_library" in parameters:
        repeat_view_review(database_path, finding_id, key=key, sdk_library=sdk_library)
        return
    if (sdk_library / ("libdirp.dll" if os.name == "nt" else "libdirp.so")).is_file():
        os.environ.setdefault("PVDELTA_DIRP_LIB", str(sdk_library))
    repeat_view_review(database_path, finding_id, key=key)

BASEMAPS = {
    "OpenStreetMap": {
        "tiles": "OpenStreetMap",
        "attr": None,
        "max_zoom": 19,
    },
    "Terrain / elevation (OpenTopoMap)": {
        "tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": (
            "Map data © OpenStreetMap contributors; map style © OpenTopoMap "
            "(CC-BY-SA)"
        ),
        "max_zoom": 17,
    },
    "Satellite (Google high-resolution)": {
        "tiles": (
            "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
        ),
        "attr": "Imagery © Google",
        "max_zoom": 22,
        # The provider serves detailed aerial tiles through z=21. Leaflet keeps
        # the layer visible and enlarges the last level beyond that zoom.
        "max_native_zoom": 21,
    },
    "Satellite fallback (Esri World Imagery)": {
        "tiles": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/"
            "{z}/{y}/{x}"
        ),
        "attr": "Imagery © Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        "max_zoom": 19,
        "max_native_zoom": 19,
    },
}


def _safe_ratio(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) else None


def _optional_float(value: str) -> float | None:
    value = value.strip().replace(",", ".")
    if not value:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("value must be finite")
    return result


def _analyze_rgb(uploaded_file: Any) -> dict[str, Any]:
    """Return non-location EXIF and scene-relative sharpness without saving the upload."""
    payload = uploaded_file.getvalue()
    with Image.open(io.BytesIO(payload)) as image:
        width, height = image.size
        exif = image.getexif()
        exif_ifd = exif.get_ifd(0x8769)
        tag = {name: key for key, name in ExifTags.TAGS.items()}

        def field(name: str) -> Any:
            key = tag.get(name)
            return exif_ifd.get(key) if key is not None else None

        preview = image.convert("L")
        preview.thumbnail((1200, 1200))
        gray = np.asarray(preview, dtype=np.float32)
        if min(gray.shape) < 3:
            sharpness = math.nan
        else:
            laplacian = (
                -4 * gray[1:-1, 1:-1]
                + gray[:-2, 1:-1]
                + gray[2:, 1:-1]
                + gray[1:-1, :-2]
                + gray[1:-1, 2:]
            )
            sharpness = float(laplacian.var())
        exposure_s = _safe_ratio(field("ExposureTime"))
        return {
            "file": uploaded_file.name,
            "dimensions": f"{width}×{height}",
            "sharpness_score": round(sharpness, 1),
            "exposure_s": exposure_s,
            "shutter": None if not exposure_s else f"1/{round(1 / exposure_s)} s",
            "iso": field("ISOSpeedRatings") or field("PhotographicSensitivity"),
            "f_number": _safe_ratio(field("FNumber")),
            "focal_length_mm": _safe_ratio(field("FocalLength")),
            "equivalent_focal_length_mm": field("FocalLengthIn35mmFilm"),
            "digital_zoom": _safe_ratio(field("DigitalZoomRatio")),
        }


def _check(label: str, status: str, detail: str) -> dict[str, str]:
    return {"check": label, "status": status, "detail": detail}


def _pass_review(label: str, passed: bool, detail: str) -> dict[str, str]:
    return _check(label, "PASS" if passed else "REVIEW", detail)


def _table_exists(connection: Any, table_name: str) -> bool:
    """Allow the dashboard to read older local stores before a schema migration."""
    return bool(connection.execute(
        """
        SELECT count(*)
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
        """,
        [table_name],
    ).fetchone()[0])


def _resolve_capture_path(source_root: str | None, source_path: str | None) -> Path | None:
    if not source_path:
        return None
    path = Path(source_path)
    if not path.is_absolute() and source_root:
        path = Path(source_root) / path
    return path if path.is_file() else None


def _frame_preview(path: Path, *, max_size: tuple[int, int] = (1600, 1200)) -> Image.Image:
    """Load a local frame for the timeline without retaining an open file handle."""
    with Image.open(path) as source:
        preview = source.convert("RGB")
        preview.thumbnail(max_size)
        return preview.copy()


def _quadratic_points(points: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    """Evaluate the reviewed native-thermal to RGB quadratic registration."""
    array = np.asarray(points, dtype=float)
    centre = np.asarray(model["centre"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or centre.shape != (2,) or scale.shape != (2,):
        raise ValueError("registration model or points have an invalid shape")
    if np.any(scale == 0) or coefficients.shape != (6, 2):
        raise ValueError("registration model has invalid scale or coefficient shape")
    x, y = ((array - centre) / scale).T
    basis = np.column_stack((np.ones(len(array)), x, y, x * y, x * x, y * y))
    result = basis @ coefficients
    if not np.isfinite(result).all():
        raise ValueError("registration maps the polygon to non-finite RGB coordinates")
    return result


def _load_rgb_registration() -> dict[str, Any] | None:
    """Load a local reviewed calibration; never invent a display transform."""
    configured = os.environ.get("PVDELTA_RGB_THERMAL_CALIBRATION")
    candidates = [Path(configured)] if configured else []
    candidates.append(
        PRIVATE_WORKSPACE
        / "experiments"
        / "EXP-011-assets"
        / "native-rgb-registration.json"
    )
    for path in candidates:
        if path is None or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            diagnostic = payload.get("quadratic_diagnostic", {})
            if diagnostic.get("accepted") and diagnostic.get("forward"):
                return {"model": diagnostic["forward"], "path": str(path),
                    "calibration_frame": payload.get('cross_frame_visual_audit', {}).get('calibration_frame')}
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def _rgb_finding_crop(
    path: Path,
    polygon_json: str,
    *,
    registration: dict[str, Any],
    native_shift_xy: tuple[float, float] = (0.0, 0.0),
) -> Image.Image:
    """Project a native finding onto RGB and return a marked local crop."""
    polygon = np.asarray(json.loads(polygon_json), dtype=float)
    if polygon.shape != (4, 2):
        raise ValueError("finding polygon must contain four points")
    polygon_centre = polygon.mean(axis=0)
    polygon = polygon_centre + DISPLAY_OUTLINE_SCALE * (polygon - polygon_centre)
    # EXP-043 fitted a bounded native translation after the global calibration.
    # Undo it before applying the stored native-to-RGB model.
    native_for_registration = polygon - np.asarray(native_shift_xy, dtype=float)
    rgb_polygon = _quadratic_points(native_for_registration, registration["model"])
    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.size != (4032, 3024):
        raise ValueError('loaded calibration is only validated for 4032×3024 paired RGB')
    points = [(round(float(x)), round(float(y))) for x, y in rgb_polygon]
    if (max(x for x, _ in points) <= 0 or min(x for x, _ in points) >= image.width
            or max(y for _, y in points) <= 0 or min(y for _, y in points) >= image.height):
        raise ValueError("projected finding is outside the paired RGB frame")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    draw.line(points + [points[0]], fill="#ff2020", width=max(5, image.width // 700), joint="curve")
    crop_padding = 180
    left = max(0, min(point[0] for point in points) - crop_padding)
    top = max(0, min(point[1] for point in points) - crop_padding)
    right = min(image.width, max(point[0] for point in points) + crop_padding)
    bottom = min(image.height, max(point[1] for point in points) + crop_padding)
    return image.crop((left, top, right, bottom))


def _render_local_map(
    features: list[dict[str, Any]],
    *,
    basemap: str | None,
    show_laser_targets: bool,
    show_camera_positions: bool = True,
    show_flight_route: bool = True,
    fit_to_route: bool = True,
    selected_frame: dict[str, Any] | None = None,
    selected_anomaly: dict[str, Any] | None = None,
) -> None:
    import folium
    from folium.plugins import PolyLineTextPath
    from streamlit_folium import st_folium

    valid_features = []
    for item in features:
        try:
            latitude = float(item["camera_latitude"])
            longitude = float(item["camera_longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(latitude) and math.isfinite(longitude):
            valid_features.append(item)
    if not valid_features:
        st.warning("No valid camera GPS points are available for the map.")
        return
    features = valid_features
    center = [
        sum(item["camera_latitude"] for item in features) / len(features),
        sum(item["camera_longitude"] for item in features) / len(features),
    ]
    inspection_map = folium.Map(
        location=center,
        zoom_start=18,
        max_zoom=22,
        tiles=None,
        control_scale=True,
    )
    if basemap is not None:
        for name, config in BASEMAPS.items():
            layer_kwargs = {
                "tiles": config["tiles"],
                "name": name,
                "overlay": False,
                "control": True,
                "show": name == basemap,
                "max_zoom": config["max_zoom"],
            }
            if config["attr"] is not None:
                layer_kwargs["attr"] = config["attr"]
            if config.get("max_native_zoom") is not None:
                layer_kwargs["max_native_zoom"] = config["max_native_zoom"]
            folium.TileLayer(**layer_kwargs).add_to(inspection_map)
    camera_layer = folium.FeatureGroup(
        name="Thermal camera positions", show=show_camera_positions
    ).add_to(
        inspection_map
    )
    route_layer = folium.FeatureGroup(
        name="Flight route", show=show_flight_route
    ).add_to(inspection_map)
    frame_layer = folium.FeatureGroup(
        name="Selected frame", show=selected_frame is not None
    ).add_to(inspection_map)
    target_layer = folium.FeatureGroup(name="Laser targets", show=show_laser_targets).add_to(
        inspection_map
    )
    anomaly_layer = folium.FeatureGroup(name="Selected anomaly", show=True).add_to(
        inspection_map
    )
    camera_track = [
        [item["camera_latitude"], item["camera_longitude"]] for item in features
    ]
    for item in features:
        details = [Path(item["source_path"]).name, item["image_kind"]]
        if item["capture_time_utc"]:
            details.append(item["capture_time_utc"])
        if item["temperature_min_c"] is not None and item["temperature_max_c"] is not None:
            details.append(
                f'{item["temperature_min_c"]:.1f}–{item["temperature_max_c"]:.1f} °C'
            )
        folium.CircleMarker(
            [item["camera_latitude"], item["camera_longitude"]],
            radius=2,
            color="#2563eb",
            fill=True,
            weight=1,
            opacity=0.45,
            fill_opacity=0.30,
            tooltip=" · ".join(details),
        ).add_to(camera_layer)
        if (
            show_laser_targets
            and item["laser_target_latitude"] is not None
            and item["laser_target_longitude"] is not None
        ):
            folium.CircleMarker(
                [item["laser_target_latitude"], item["laser_target_longitude"]],
                radius=3,
                color="#16a34a",
                fill=True,
                tooltip=f'{Path(item["source_path"]).name} · laser target',
            ).add_to(target_layer)
        if item["footprint_geojson"] is not None:
            folium.GeoJson(
                item["footprint_geojson"],
                tooltip=(
                    f'{item["capture_id"]} coverage · '
                    f'{item["footprint_method"] or "method not recorded"}'
                ),
                style_function=lambda _feature: {
                    "color": "#f97316",
                    "weight": 2,
                    "fillOpacity": 0.12,
                },
            ).add_to(inspection_map)
    if selected_anomaly is not None:
        folium.Marker(
            [selected_anomaly["latitude"], selected_anomaly["longitude"]],
            tooltip=(
                f'{selected_anomaly["file_name"]} · {selected_anomaly["delta_t_k"]:.2f} K'
            ),
            popup=(
                f'<b>Selected anomaly</b><br>{selected_anomaly["file_name"]}<br>'
                f'lat {selected_anomaly["latitude"]:.7f}<br>'
                f'lon {selected_anomaly["longitude"]:.7f}'
            ),
            icon=folium.Icon(color="red", icon="fire", prefix="fa"),
        ).add_to(anomaly_layer)
    route_segments = build_route_segments(features)
    route_bounds = [
        point for segment in route_segments if len(segment) > 1 for point in segment
    ] or camera_track
    if show_flight_route:
        for segment in route_segments:
            route = simplify_route(segment)
            if len(route) > 1:
                line = folium.PolyLine(
                    route,
                    color="#2563eb",
                    weight=3,
                    opacity=0.85,
                    tooltip="Thermal capture flight route",
                ).add_to(route_layer)
                PolyLineTextPath(
                    line,
                    "➤",
                    repeat=True,
                    offset=7,
                    attributes={"fill": "#1d4ed8", "font-weight": "bold", "font-size": "16"},
                ).add_to(route_layer)
        if route_segments:
            first = route_segments[0][0]
            last = route_segments[-1][-1]
            folium.Marker(first, tooltip="Flight start", icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(route_layer)
            folium.Marker(last, tooltip="Flight end", icon=folium.Icon(color="black", icon="stop", prefix="fa")).add_to(route_layer)
    if selected_frame is not None:
        frame_point = [selected_frame["camera_latitude"], selected_frame["camera_longitude"]]
        frame_name = Path(selected_frame["source_path"]).name
        frame_layer.add_child(
            folium.Marker(
                frame_point,
                tooltip=f"Selected frame · {frame_name}",
                popup=(
                    f"<b>Selected frame</b><br>{frame_name}<br>"
                    f"lat {frame_point[0]:.7f}<br>lon {frame_point[1]:.7f}"
                ),
                icon=folium.Icon(color="orange", icon="camera", prefix="fa"),
            )
        )
        if selected_frame.get("footprint_geojson") is not None:
            folium.GeoJson(
                selected_frame["footprint_geojson"],
                name="Selected frame coverage",
                tooltip=(
                    f"{frame_name} coverage · {selected_frame.get('footprint_method') or 'calibrated'} · "
                    f"±{selected_frame.get('footprint_accuracy_m'):.1f} m"
                ),
                style_function=lambda _feature: {
                    "color": "#f59e0b",
                    "weight": 3,
                    "fillOpacity": 0.16,
                    "dashArray": "6 4",
                },
            ).add_to(frame_layer)
    if fit_to_route and route_bounds:
        inspection_map.fit_bounds(route_bounds, padding=(18, 18))
    if basemap is not None or show_flight_route or show_camera_positions or show_laser_targets or selected_frame is not None:
        folium.LayerControl(collapsed=False).add_to(inspection_map)
    st_folium(
        inspection_map,
        key="inspection-map-v3",
        use_container_width=True,
        height=560,
        returned_objects=[],
    )


st.set_page_config(page_title="pvdelta Flight Planner", page_icon="🌡️", layout="wide")
interface_mode = st.radio(
    "Interface", ("Basic", "Advanced"), horizontal=True, key="interface-mode",
    help="Basic: open a flight, scan and review. Advanced: all planning and inspection tools.",
)
DEFAULT_DATABASE_PATH = Path(st.session_state.get('active-inspection-database') or DEFAULT_DATABASE_PATH)
material_packs = list(PRIVATE_WORKSPACE.glob('stage1-*/material-review-pack.json'))
latest_material_pack = max(material_packs, key=lambda path: path.stat().st_mtime_ns, default=None)
default_material_pack = str(latest_material_pack) if latest_material_pack else ''
material_review_view(os.environ.get('PVDELTA_MASK_MATERIAL_PACK', default_material_pack),
                     advanced=interface_mode == 'Advanced')
if interface_mode == "Basic":
    from pvdelta.basic_dashboard import basic_dashboard

    basic_dashboard(DEFAULT_DATABASE_PATH, DEFAULT_SDK_LIBRARY_PATH,
                    rgb_registration=_load_rgb_registration(),
                    surface_manifest=SURFACE_MANIFEST)
    st.stop()

st.title("pvdelta · PV thermal flight planner")
st.caption(
    "Plan capture geometry, record the measurements actually available on site, and keep "
    "inspection data local."
)

with st.sidebar:
    st.header("Mission")
    site_label = st.text_input("Site label", "local-pilot")
    mission_date = st.date_input("Planned date")

    st.header("Camera and route")
    platform_keys = list_platforms()
    default_platform = platform_keys.index("DJI_M4T") if "DJI_M4T" in platform_keys else 0
    platform_key = st.selectbox(
        "Aircraft / camera",
        platform_keys,
        index=default_platform,
        format_func=lambda key: get_platform(key).model,
    )
    platform = get_platform(platform_key)
    thermal = platform.thermal
    if thermal is None or thermal.dfov_deg is None:
        st.error("The selected platform has no reviewed thermal field of view.")
        st.stop()

    flight_altitude_agl_m = st.number_input(
        "Flight altitude above lower-edge ground (m AGL)",
        1.0,
        120.0,
        16.0,
        0.5,
        help="Add the take-off/terrain elevation offset in the mission controller when needed.",
    )
    geometry_mode = st.radio(
        "Camera view",
        ("Nadir (straight down)", "Perpendicular to panel"),
        help="Nadir keeps the gimbal vertical. Perpendicular points the camera along the panel normal.",
    )
    speed_m_s = st.number_input("Flight speed (m/s)", 0.1, 15.0, 2.5, 0.1)
    capture_interval_s = st.number_input("Photo interval (s)", 0.7, 60.0, 1.0, 0.1)
    overlap_preset = st.selectbox("Coverage preset", list(OVERLAP_PRESETS))
    default_front, default_side = OVERLAP_PRESETS[overlap_preset]
    if overlap_preset == "Custom":
        forward_overlap = st.number_input("Forward overlap (%)", 0.0, 95.0, 80.0, 1.0)
        side_overlap = st.number_input("Side overlap (%)", 0.0, 95.0, 70.0, 1.0)
    else:
        forward_overlap, side_overlap = default_front, default_side
        st.caption(f"{forward_overlap:.0f}% forward · {side_overlap:.0f}% side overlap")

    st.header("Array geometry")
    module_preset = st.selectbox("Module format", list(MODULE_PRESETS))
    (
        preset_length,
        preset_width,
        preset_cell,
        preset_cell_count,
        preset_cell_format,
        preset_cell_cut,
    ) = MODULE_PRESETS[module_preset]
    module_orientation = st.radio(
        "Module orientation",
        ("Portrait", "Landscape"),
        help="Portrait: the long module side runs up the rack slope. Landscape: the short side does.",
        horizontal=True,
        index=1,
    )
    panel_tilt_deg = st.number_input("Panel tilt (°)", 0.0, 80.0, 25.0, 1.0)
    rack_rows = st.number_input("Modules up the rack slope", 1, 20, 4, 1)
    lower_edge_height_m = st.number_input(
        "Lower module edge above ground (m)", 0.0, 5.0, 0.0, 0.1
    )

    with st.expander("Advanced geometry"):
        if module_preset == "Custom":
            module_length_m = st.number_input("Module length (m)", 0.2, 3.5, 2.278, 0.001)
            module_width_m = st.number_input("Module width (m)", 0.2, 2.0, 1.113, 0.001)
            smallest_cell_mm = st.number_input(
                "Smallest cell dimension (mm)", 20.0, 250.0, 91.0, 1.0
            )
            module_cell_count = st.number_input("Cell count", 1, 300, 144, 1)
            module_cell_format = st.text_input("Cell format", "M10")
            module_cell_cut = st.selectbox("Cell layout", ("half-cut", "full-cell", "other"))
        else:
            module_length_m, module_width_m, smallest_cell_mm = (
                preset_length,
                preset_width,
                preset_cell,
            )
            module_cell_count = preset_cell_count
            module_cell_format = preset_cell_format
            module_cell_cut = preset_cell_cut
            st.write(
                f"Preset geometry: {module_length_m:.3f} × {module_width_m:.3f} m; "
                f"{module_cell_count} {module_cell_format or ''} {module_cell_cut} cells; "
                f"smallest cell dimension {smallest_cell_mm:.0f} mm."
            )
        row_gap_m = st.number_input("Gap between module rows (m)", 0.0, 1.0, 0.0, 0.01)
        st.caption(
            "The route is assumed to follow the table length, with the thermal image's "
            "640 px edge along the route and its 512 px edge across the rack."
        )

    with st.expander("Optional asset metadata"):
        module_model = st.text_input("Module manufacturer / model", "")
        module_technology = st.selectbox(
            "Cell technology",
            ("Unknown", "TOPCon", "Mono PERC", "HJT", "Thin-film", "Other"),
        )
        bifacial = st.checkbox("Bifacial module", False)
        aircraft_label = st.text_input("Aircraft label", "")
        firmware_version = st.text_input("Aircraft / camera firmware", "")

    estimate_image_count = st.toggle("Estimate image count for a rectangular area", False)
    if estimate_image_count:
        site_length_m = st.number_input(
            "Area length along flight lines (m)", 1.0, 10000.0, 200.0, 10.0
        )
        site_width_m = st.number_input(
            "Area width across flight lines (m)", 1.0, 10000.0, 100.0, 10.0
        )
    else:
        site_length_m = None
        site_width_m = None

rack_module_span_m = module_length_m if module_orientation == "Portrait" else module_width_m
rack_slope_length_m = rack_rows * rack_module_span_m + (rack_rows - 1) * row_gap_m
rack_top_elevation_m = lower_edge_height_m + rack_slope_length_m * math.sin(
    math.radians(panel_tilt_deg)
)
rack_center_elevation_m = (lower_edge_height_m + rack_top_elevation_m) / 2
aligned_to_panel = geometry_mode == "Perpendicular to panel"
height_reference_elevation_m = rack_center_elevation_m if aligned_to_panel else lower_edge_height_m
height_above_reference_m = flight_altitude_agl_m - height_reference_elevation_m
if height_above_reference_m <= 0:
    st.error("Flight altitude must be above the selected rack reference elevation.")
    st.stop()

try:
    flight = plan_flight(
        thermal_width_px=thermal.sensor_width_px,
        thermal_height_px=thermal.sensor_height_px,
        thermal_diagonal_fov_deg=thermal.dfov_deg,
        height_above_module_m=height_above_reference_m,
        panel_tilt_deg=panel_tilt_deg,
        camera_aligned_to_panel=aligned_to_panel,
        forward_overlap_pct=forward_overlap,
        side_overlap_pct=side_overlap,
        capture_interval_s=capture_interval_s,
        planned_speed_m_s=speed_m_s,
        smallest_cell_feature_mm=smallest_cell_mm,
        module_width_m=module_width_m,
        module_height_m=module_length_m,
        site_length_m=site_length_m,
        site_width_m=site_width_m,
        long_image_axis_along_flight=True,
        tilt_along_image_height=True,
    )
except FlightPlanningError as exc:
    st.error(str(exc))
    st.stop()

maximum_flight_agl_m = flight.maximum_height_for_resolution_m + height_reference_elevation_m
camera_to_lower_edge_m = flight_altitude_agl_m - lower_edge_height_m
camera_to_top_edge_m = flight_altitude_agl_m - rack_top_elevation_m
rack_axis_footprint_m = flight.footprint_height_m
resolution_ok = flight.pixels_per_smallest_cell_feature >= 5
overlap_speed_ok = speed_m_s <= flight.overlap_limited_speed_m_s
angle_ok = flight.view_angle_from_normal_deg <= 60
rack_coverage_ok = rack_axis_footprint_m >= rack_slope_length_m
recommended_speed = min(3.0, flight.overlap_limited_speed_m_s * 0.9)

plan_tab, field_tab, inspect_tab, boundary_tab, gap_tab, rgb_tab, data_tab, review_tab, ortho_tab, export_tab, panel_tab = st.tabs(
    (
        "Flight plan",
        "Field checklist",
        "Radiometric scan",
        "Panel boundaries",
        "Unassigned evidence",
        "RGB quality",
        "Local data & map",
        "User review",
        "Orthoview",
        "Export",
        "Panel review (v2)",
    )
)

with plan_tab:
    st.subheader("Capture geometry")
    if resolution_ok:
        st.success(
            f"The selected {flight_altitude_agl_m:.1f} m AGL meets the 5 px/cell planning target."
        )
    else:
        st.error(
            f"Reduce altitude to {maximum_flight_agl_m:.1f} m AGL or lower for the "
            "5 px/cell planning target."
        )
    st.caption(
        "The calculated maximum is a resolution ceiling, not an operational target. "
        "Leave margin for terrain, altitude variation and model uncertainty."
    )

    row1 = st.columns(4)
    row1[0].metric("Thermal GSD", f"{flight.thermal_gsd_cm_px:.2f} cm/px")
    row1[1].metric("Smallest cell", f"{flight.pixels_per_smallest_cell_feature:.1f} px")
    row1[2].metric("Resolution ceiling", f"{maximum_flight_agl_m:.1f} m AGL")
    row1[3].metric("Suggested speed", f"≤ {recommended_speed:.1f} m/s")
    row2 = st.columns(4)
    row2[0].metric(
        "Thermal footprint",
        f"{flight.footprint_width_m:.1f} × {flight.footprint_height_m:.1f} m",
    )
    row2[1].metric("Along-track step", f"{flight.along_track_step_m:.1f} m")
    row2[2].metric("Cross-track step", f"{flight.cross_track_step_m:.1f} m")
    row2[3].metric("View from panel normal", f"{flight.view_angle_from_normal_deg:.1f}°")

    plan_checks = [
        _pass_review(
            "Thermal cell resolution",
            resolution_ok,
            f"{flight.pixels_per_smallest_cell_feature:.1f} px; target ≥5 px",
        ),
        _pass_review(
            "Speed / photo interval",
            overlap_speed_ok,
            f"Planned {speed_m_s:.1f} m/s; overlap limit {flight.overlap_limited_speed_m_s:.1f} m/s",
        ),
        _pass_review(
            "Viewing angle",
            angle_ok,
            f"{flight.view_angle_from_normal_deg:.1f}° from panel normal; target ≤60°",
        ),
        _pass_review(
            "Whole rack in one frame",
            rack_coverage_ok,
            f"{rack_axis_footprint_m:.2f} m coverage across a {rack_slope_length_m:.2f} m rack",
        ),
    ]
    st.dataframe(plan_checks, hide_index=True, width="stretch")
    if not rack_coverage_ok:
        st.info(
            "One frame does not cover every module row. That is acceptable when adjacent "
            "flight lines provide the planned side overlap; it is not a missing-image warning."
        )

    with st.expander("How this rack was modelled"):
        st.write(
            f"{rack_rows} {module_orientation.lower()} modules span "
            f"{rack_slope_length_m:.2f} m up the {panel_tilt_deg:.1f}° slope. The lower edge is "
            f"{lower_edge_height_m:.2f} m above ground and the upper edge is "
            f"{rack_top_elevation_m:.2f} m above ground. At the selected altitude, the "
            f"vertical camera separations are about {camera_to_lower_edge_m:.2f} m at the "
            f"lower edge and {camera_to_top_edge_m:.2f} m at the upper edge."
        )

    if flight.estimated_images is not None and flight.estimated_flight_lines is not None:
        st.subheader("Rough rectangular-area estimate")
        st.metric("Estimated photos", flight.estimated_images)
        st.write(
            f"{flight.estimated_flight_lines} flight lines × capture positions derived from "
            f"{site_length_m:.0f} × {site_width_m:.0f} m, the thermal footprint and "
            f"{forward_overlap:.0f}/{side_overlap:.0f}% overlap. Turns, irregular boundaries, "
            "obstacles and line overshoot are not included."
        )

    st.caption(
        "Detailed mapping uses 80/70 as a conservative starting preset for repetitive PV "
        "rows. Fast screening uses the published DJI 70/20 solar example. Validate the final "
        "route on site."
    )

with field_tab:
    st.subheader("Capture settings")
    left, right = st.columns(2)
    with left:
        gain_mode = st.selectbox("Thermal gain", ("High gain", "Low gain"))
        radiometric_format = st.checkbox("Save original radiometric R-JPEG", True)
    with right:
        ffc_done = st.checkbox("FFC checked before the quantitative pass", True)
        rtk_enabled = st.checkbox("RTK enabled", True)

    st.info(
        "High gain is the normal PV starting point while the complete scene stays below "
        "150 °C. Switch to low gain if clipping is possible; gain does not replace the "
        "radiometric correction settings."
    )

    st.subheader("Measurements recorded on site")
    st.write(
        "These values are observations made during the flight. Leave this section off while "
        "planning; do not invent values to make a checklist pass."
    )
    record_conditions = st.toggle("I have measurements from the flight", False)
    if record_conditions:
        col1, col2, col3 = st.columns(3)
        with col1:
            irradiance_w_m2 = st.number_input("POA irradiance (W/m²)", 0.0, 1500.0, 800.0, 10.0)
            irradiance_change_pct_min = st.number_input(
                "Irradiance change (%/min)", 0.0, 100.0, 2.0, 0.5
            )
        with col2:
            wind_kmh = st.number_input("Wind speed (km/h)", 0.0, 100.0, 8.0, 1.0)
            ambient_c = st.number_input("Air temperature (°C)", -30.0, 70.0, 25.0, 0.5)
        with col3:
            humidity_pct = st.number_input("Relative humidity (%)", 0.0, 100.0, 50.0, 1.0)
            cloud_okta = st.number_input("Cumulus cloud cover (okta)", 0, 8, 0, 1)
        irradiance_source = st.selectbox(
            "POA measurement source",
            ("POA pyranometer", "POA reference cell", "Plant sensor", "Other / unknown"),
        )
    else:
        irradiance_w_m2 = None
        irradiance_change_pct_min = None
        wind_kmh = None
        ambient_c = None
        humidity_pct = None
        cloud_okta = None
        irradiance_source = None

    with st.expander("Advanced radiometric record"):
        st.write(
            "Reflected apparent temperature is the effective infrared radiation reflected "
            "by the glass toward the camera. It is not the air temperature. Leave it blank "
            "unless it was measured or deliberately set. The FLIR reflector method uses "
            "crumpled and re-flattened aluminium foil at the target position, viewed from the "
            "measurement direction with emissivity set to 1.0."
        )
        st.markdown(
            "Method reference: [FLIR thermographic measurement techniques]"
            "(https://support.flir.com/docdownload/assets/dl/t559495-en-us%24aa.pdf)."
        )
        emissivity = st.number_input("Module glass emissivity", 0.10, 1.00, 0.95, 0.01)
        reflected_text = st.text_input(
            "Reflected apparent temperature (°C)",
            "",
            placeholder="Leave blank unless measured",
        )
        try:
            reflected_c = _optional_float(reflected_text)
        except ValueError:
            reflected_c = None
            st.warning("Reflected apparent temperature must be a finite number.")
        st.caption(
            "Object distance should normally come from each image's verified metadata or "
            "calibrated geometry, not from one mission-wide guess."
        )

    capture_checks = [
        _pass_review("Radiometric source", radiometric_format, "Keep the original R-JPEG"),
        _pass_review("FFC", ffc_done, "Check FFC before the quantitative pass"),
        _pass_review("RTK", rtk_enabled, "Useful for camera pose; not a Panel ID proof"),
    ]
    if record_conditions:
        capture_checks.extend(
            [
                _pass_review(
                    "POA irradiance",
                    irradiance_w_m2 >= 600,
                    f"{irradiance_w_m2:.0f} W/m²; target ≥600",
                ),
                _pass_review(
                    "Irradiance stability",
                    irradiance_change_pct_min <= 10,
                    f"{irradiance_change_pct_min:.1f}%/min; target ≤10",
                ),
                _pass_review("Wind", wind_kmh <= 28, f"{wind_kmh:.1f} km/h; target ≤28"),
                _pass_review("Cloud", cloud_okta <= 2, f"{cloud_okta} okta; target ≤2"),
            ]
        )
    else:
        capture_checks.append(
            _check("Environmental measurements", "NOT RECORDED", "Complete this on site")
        )
    st.dataframe(capture_checks, hide_index=True, width="stretch")

with inspect_tab:
    st.subheader("Start a local radiometric scan")
    st.write(
        "Select an indexed flight folder and decode each original thermal R-JPEG on the DJI "
        "native grid. Files remain in place; only compact summaries are written to DuckDB."
    )
    scan_folder_text = st.text_input("Flight folder to scan", "", key="radiometric_scan_folder")
    scan_database_text = st.text_input(
        "Scan database path", str(DEFAULT_DATABASE_PATH), key="radiometric_scan_database"
    )
    sdk_default = os.environ.get("PVDELTA_DIRP_LIB", "") or (
        str(DEFAULT_SDK_LIBRARY_PATH) if (DEFAULT_SDK_LIBRARY_PATH / "libdirp.dll").is_file() else ""
    )
    sdk_library_text = st.text_input(
        "DJI Thermal SDK library folder", sdk_default,
        help="Windows: .../sdk/tsdk-core/lib/windows/release_x64; Linux: .../release_x64.",
    )
    scan_folder = Path(scan_folder_text).expanduser()
    scan_database = Path(scan_database_text).expanduser()
    resume_scan = st.checkbox("Resume unchanged successful SDK frames", value=True)
    with st.expander("Optional explicit radiometric corrections"):
        measurement_profile_text = st.text_input("SDK measurement overrides JSON file (optional)",
            help='Leave empty to preserve recorded values. Example: {"distance_m": 19.0}. '
                 'Use camera-to-target distance, not flight altitude. Never infer reflected temperature from ambient.')
        st.caption('Corrections are calculated by the DJI SDK on a separate native measurement '
            'profile. Original JPEGs and recorded parameters are preserved; matching/geometry '
            'requires checks for the selected profile. A value supplied here is not field verification.')
    if st.button("Start radiometric scan", disabled=not scan_folder_text.strip(), type="primary"):
        st.session_state.pop("radiometric_scan_results", None)
        st.session_state.pop("radiometric_scan_report", None)
        try:
            sdk_library = Path(sdk_library_text).expanduser()
            sdk_filename = "libdirp.dll" if os.name == "nt" else "libdirp.so"
            if not sdk_library.is_dir() or not (sdk_library / sdk_filename).is_file():
                raise ValueError(
                    f"SDK library folder must contain {sdk_filename}: {sdk_library}. "
                    f"Expected local folder: {DEFAULT_SDK_LIBRARY_PATH}"
                )
            scan_progress = st.progress(0.0)
            scan_status = st.empty()

            def report_scan(position: int, total: int, path: Path) -> None:
                scan_progress.progress(position / total)
                scan_status.write(f"{position}/{total}: {path.name}")

            overrides = json.loads(Path(measurement_profile_text).expanduser().read_bytes()) \
                if measurement_profile_text.strip() else None
            cache_id = hashlib.sha256((str(scan_folder.resolve())+json.dumps(overrides, sort_keys=True)).encode()).hexdigest()[:16]
            cache_path = scan_database.resolve().parent / "radiometric-scans" / cache_id
            report, results = scan_radiometric_flight(
                scan_folder, cache_path,
                sdk_lib=str(sdk_library),
                resume=resume_scan,
                progress=report_scan,
                measurement_overrides=overrides,
            )
            st.session_state["radiometric_scan_report"] = report
            from pvdelta.store import connect_store, update_capture_radiometry
            written, unindexed = 0, []
            if not overrides:
                connection = connect_store(scan_database)
                try:
                    connection.execute("BEGIN TRANSACTION")
                    for result in results:
                        try:
                            update_capture_radiometry(connection, result)
                            written += 1
                        except KeyError:
                            unindexed.append(Path(result.source_path).name)
                    connection.execute("COMMIT")
                except Exception:
                    connection.execute("ROLLBACK")
                    raise
                finally:
                    connection.close()
            else:
                st.info('Corrected native measurements are saved in a separate private profile. '
                    'Existing database summaries and findings retain their original parameters; '
                    'they have not been reanalyzed with these corrections.')
            st.session_state["radiometric_scan_results"] = [
                {
                    "file": Path(result.source_path).name,
                    "native_grid": f"{result.width_px}×{result.height_px}",
                    "valid_pixels": result.valid_pixels,
                    "min_°C": round(result.temperature_min_c, 2),
                    "median_°C": round(result.temperature_median_c, 2),
                    "max_°C": round(result.temperature_max_c, 2),
                    "emissivity": round(result.emissivity, 3),
                    "distance_m": round(result.distance_m, 3),
                }
                for result in results
            ]
            st.success(f"SDK-readable: {len(results)}/{report['requested_frames']}. "
                f"Database summaries updated: {written}. Fresh decodes: {report['fresh_sdk_decodes']}; "
                f"reused: {report['reused_frames']}.")
            st.caption(f"Private scan journal: {cache_path}")
            if unindexed:
                st.warning(f"{len(unindexed)} decoded frames have no unique indexed thermal record. "
                           "Index the flight before updating their database summaries.")
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            st.error(f"Radiometric scan failed: {exc}")

    if st.session_state.get("radiometric_scan_report"):
        report = st.session_state["radiometric_scan_report"]
        failed = [{"File": r['source_name'], "Error": r['error']}
                  for r in report['frames'] if r['status'] == 'failed']
        if failed:
            st.warning(f"{len(failed)} frames failed SDK validation and remain unassessed.")
            st.dataframe(failed, hide_index=True, width="stretch")
        st.download_button("Download SDK scan outcomes (JSON)",
            json.dumps(report, indent=2, allow_nan=False), file_name="sdk-flight-scan.json",
            mime="application/json")

    if st.session_state.get("radiometric_scan_results"):
        scan_results = st.session_state["radiometric_scan_results"]
        metrics = st.columns(4)
        metrics[0].metric("Thermal frames", len(scan_results))
        metrics[1].metric("Native grid", scan_results[0]["native_grid"])
        metrics[2].metric(
            "Median range",
            f"{min(row['median_°C'] for row in scan_results):.1f}–"
            f"{max(row['median_°C'] for row in scan_results):.1f} °C",
        )
        metrics[3].metric(
            "Peak observed", f"{max(row['max_°C'] for row in scan_results):.1f} °C"
        )
        st.dataframe(scan_results, hide_index=True, width="stretch")
        st.info(
            "This scan reports native radiometry, not a panel count. Panel count becomes "
            "measurable after reviewed/automatic panel boundaries create module observations; "
            "the dashboard will show those stored observations separately."
        )

    from pvdelta.screening_replay_ui import screening_replay_view

    screening_replay_view(scan_database, sdk_library_text,
        output_root=PRIVATE_WORKSPACE/'inspection-runs')
    native_flight_view(scan_database, sdk_library_text,
        output_root=PRIVATE_WORKSPACE/'native-inspection-runs')

with boundary_tab:
    from pvdelta.control_review_ui import control_review_view
    from pvdelta.native_review_ui import native_boundary_review
    from pvdelta.panel_annotation_ui import panel_annotation_view
    from pvdelta.region_annotation_ui import region_annotation_view

    private_workspace = PRIVATE_WORKSPACE
    from pvdelta.surface_recovery_ui import surface_recovery_view

    surface_recovery_view(SURFACE_MANIFEST,
        sdk_library_text.strip() or None, output_root=private_workspace / 'surface-recovery' / 'runs')
    region_packs = sorted(private_workspace.glob('stage1-region-reference-*/development-reference-pack.json'))
    region_annotation_view(os.environ.get('PVDELTA_THERMAL_REFERENCE_PACK',
        str(region_packs[-1]) if region_packs else ''),
        output=private_workspace / 'thermal-region-references')
    control_pack = private_workspace / 'stage1-closeout' / 'control-review' / 'review-pack.json'
    band_control_pack = private_workspace / 'stage1-c3-band-centres-20261025' / 'source-review' / 'review-pack.json'
    control_pack_options = {label: str(path) for label, path in (
        ('Existing source controls', control_pack), ('Frame-band source controls', band_control_pack))
        if path.is_file()}
    control_review_view(str(control_pack) if control_pack.is_file() else '',
                        output=private_workspace/'control-reviews', pack_options=control_pack_options)
    pending_packs = sorted((private_workspace / 'experiments').glob(
        'EXP-*-assets/full-panel-review/pending-references.json'))
    panel_annotation_view(os.environ.get('PVDELTA_PANEL_REFERENCE_PACK',
        str(pending_packs[-1]) if pending_packs else ''),
        output=private_workspace / 'panel-reference-reviews')
    native_boundary_review(scan_database, sdk_library_text,
        revision_output=private_workspace / 'geometry-revisions')

with gap_tab:
    from pvdelta.gap_review_ui import native_gap_review

    pending_gap_bundle = st.session_state.pop('native-gap-pending-bundle', None)
    if pending_gap_bundle:
        st.session_state['native-gap-directory'] = pending_gap_bundle
    native_gap_review(os.environ.get('PVDELTA_GAP_REVIEW_BUNDLE',
        str(DEFAULT_DATABASE_PATH.parent / 'native-gap-review')))

with rgb_tab:
    st.subheader("RGB sharpness comparison")
    st.write(
        "Upload several JPEGs from the same route. Files are inspected in memory and are not "
        "written to the local database. GPS and serial-number fields are not shown or exported."
    )
    rgb_files = st.file_uploader(
        "RGB JPEG files", type=("jpg", "jpeg"), accept_multiple_files=True
    )
    rgb_results: list[dict[str, Any]] = []
    for uploaded in rgb_files:
        try:
            rgb_results.append(_analyze_rgb(uploaded))
        except (OSError, ValueError) as exc:
            st.warning(f"{uploaded.name}: could not be read ({exc})")
    if rgb_results:
        scores = np.asarray([item["sharpness_score"] for item in rgb_results], dtype=float)
        median_score = float(np.nanmedian(scores))
        for item in rgb_results:
            item["relative_sharpness"] = round(item["sharpness_score"] / median_score, 2)
        st.dataframe(rgb_results, hide_index=True, width="stretch")
        if any((item["digital_zoom"] or 1) > 1 for item in rgb_results):
            st.warning("At least one frame used digital zoom; it does not add optical detail.")
        slow = [
            item for item in rgb_results
            if item["exposure_s"] and item["exposure_s"] > 1 / 1000
        ]
        if slow and speed_m_s > 2:
            st.warning("Some frames are slower than 1/1000 s; motion smear should be reviewed.")
        st.caption(
            "Compare relative scores only within similar scenes. Texture, focus, motion, "
            "atmosphere and JPEG processing all affect the score."
        )

finding_scope = "All raw candidates"

with data_tab:
    st.subheader("Local DuckDB inspection store")
    st.write(
        "DuckDB holds searchable EXIF, camera pose, radiometric settings, per-frame and "
        "per-module temperature summaries, findings and optional geographic geometry. Dense "
        "temperature arrays remain external local files referenced by path and hash."
    )
    st.code(
        "inspection_runs → captures → module_observations → findings\n"
        "                                      ↘ asset_matches (optional Stage 2)",
        language="text",
    )
    database_path_text = st.text_input("Local database path", str(DEFAULT_DATABASE_PATH))
    st.session_state['active-inspection-database'] = database_path_text
    database_path = Path(database_path_text).expanduser()
    try:
        from pvdelta.ingest import index_flight_folder
        from pvdelta.store import connect_store, map_features, upsert_finding_review

        duckdb_available = True
    except (ImportError, RuntimeError):
        duckdb_available = False

    if not duckdb_available:
        st.warning("Install the dashboard extra to enable DuckDB and Folium.")
        st.code("python -m pip install -e '.[dashboard]'", language="bash")
    else:
        if st.button("Create / open local database"):
            connection = connect_store(database_path)
            connection.close()
            st.success(f"Local schema is ready at {database_path}.")

        with st.expander("Index a local DJI flight folder"):
            st.write(
                "This reads the local files in place. It stores relative paths, hashes and "
                "metadata; it does not copy or upload imagery."
            )
            flight_folder_text = st.text_input("Flight folder", "")
            ingest_run_id = st.text_input("Run ID", "pilot-001")
            if st.button("Index flight metadata", disabled=not flight_folder_text.strip()):
                ingest_progress = st.progress(0.0)
                ingest_status = st.empty()

                def report_progress(position: int, total: int, path: Path) -> None:
                    ingest_progress.progress(position / total)
                    if position == 1 or position == total or position % 100 == 0:
                        ingest_status.write(f"{position}/{total}: {path.name}")

                try:
                    summary = index_flight_folder(
                        flight_folder_text,
                        database_path,
                        run_id=ingest_run_id,
                        site_label=site_label or None,
                        progress=report_progress,
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    st.error(f"Indexing failed: {exc}")
                else:
                    st.success(
                        f"Indexed {summary.indexed_captures} captures: "
                        f"{summary.thermal_captures} thermal, {summary.rgb_captures} RGB; "
                        f"{summary.geolocated_captures} with camera coordinates."
                    )

        selected_run = None
        if database_path.is_file():
            # Older stores still migrate; current stores remain read-only on reruns.
            from pvdelta.dashboard_store import ensure_dashboard_schema

            ensure_dashboard_schema(database_path)
            connection = connect_store(database_path, read_only=True)
            features = map_features(connection)
            analysis_runs = [row[0] for row in connection.execute(
                """SELECT r.run_id FROM inspection_runs r WHERE EXISTS (
                    SELECT 1 FROM module_observations m WHERE m.analysis_run_id=r.run_id
                ) ORDER BY r.created_at_utc DESC, r.run_id DESC"""
            ).fetchall()]
            preferred_run = (
                "exp049-balanced-review" if "exp049-balanced-review" in analysis_runs
                else "exp047-red-team" if "exp047-red-team" in analysis_runs else analysis_runs[0]
                if analysis_runs else None
            )
            selected_run = (
                st.selectbox(
                    "Stage 1 analysis run", analysis_runs,
                    index=analysis_runs.index(preferred_run),
                )
                if analysis_runs else None
            )
            if selected_run and selected_run != preferred_run:
                st.warning(
                    f"{selected_run} is an older screening run. The default evidence-aware "
                    f"review run is {preferred_run}."
                )
            evidence_table_exists = _table_exists(connection, "finding_evidence")
            if evidence_table_exists:
                finding_scope = st.radio(
                    "Candidate scope",
                    ["All raw candidates", "Legacy palette shortlist (experimental)"],
                    horizontal=True,
                    help=(
                        "All candidates are shown by default. The legacy shortlist uses display "
                        "colours and is not validated panel segmentation; it can miss real anomalies."
                    ),
                )
            counts = connection.execute(
                """
                SELECT
                    (SELECT count(*) FROM captures) AS captures,
                    (SELECT count(*) FROM module_observations WHERE analysis_run_id = ?) AS modules,
                    (SELECT count(*) FROM findings WHERE analysis_run_id = ?) AS findings
                """, [selected_run, selected_run]
            ).fetchone()
            connection.close()
            cols = st.columns(3)
            cols[0].metric("Captures", counts[0])
            cols[1].metric("Module proposals — selected run", counts[1])
            cols[2].metric("Findings — selected run", counts[2])
            st.subheader("Inspection overview")
            connection = connect_store(database_path, read_only=True)
            try:
                inventory_table_exists = _table_exists(connection, "panel_assets")
                matches_table_exists = _table_exists(connection, "asset_matches")
                clusters_table_exists = _table_exists(connection, "provisional_module_clusters")
                inventory_count = (
                    connection.execute("SELECT count(*) FROM panel_assets").fetchone()[0]
                    if inventory_table_exists else 0
                )
                matched_assets = (
                    connection.execute(
                        "SELECT count(*) FROM asset_matches WHERE match_status = 'matched'"
                    ).fetchone()[0]
                    if matches_table_exists else 0
                )
                observation_counts = connection.execute(
                    """
                    SELECT count(*) AS observations,
                           count(DISTINCT (capture_id, frame_module_id)) AS frame_local_keys,
                           count(*) FILTER (
                               WHERE observation_id IN (
                                   SELECT DISTINCT observation_id FROM findings
                               )
                           ) AS anomaly_observations
                    FROM module_observations WHERE analysis_run_id = ?
                    """, [selected_run]
                ).fetchone()
                capture_counts = connection.execute(
                    """
                    SELECT count(*) AS total,
                           count(*) FILTER (WHERE image_kind = 'thermal') AS thermal,
                           count(*) FILTER (WHERE image_kind = 'rgb') AS rgb,
                           count(DISTINCT run_id) AS runs,
                           count(*) FILTER (WHERE paired_capture_id IS NOT NULL) AS paired,
                           count(*) FILTER (
                               WHERE camera_latitude IS NOT NULL
                                 AND camera_longitude IS NOT NULL
                           ) AS camera_gps,
                           count(*) FILTER (WHERE temperature_array_uri IS NOT NULL) AS radiometric,
                           count(*) FILTER (WHERE temperature_median_c IS NOT NULL) AS temp_summary
                    FROM captures
                    """
                ).fetchone()
                observation_coverage = connection.execute(
                    """
                    SELECT count(*) AS observations,
                           count(DISTINCT capture_id) AS observation_captures,
                           count(DISTINCT (capture_id, frame_module_id)) AS frame_local_keys,
                           count(*) FILTER (WHERE visibility = 'full') AS visible,
                           count(*) FILTER (WHERE estimated_latitude IS NOT NULL
                                                   AND estimated_longitude IS NOT NULL) AS located,
                           count(*) FILTER (
                               WHERE observation_id IN (
                                   SELECT DISTINCT observation_id FROM findings
                               )
                           ) AS anomaly_observations,
                           count(*) FILTER (
                               WHERE assessment_status = 'analyzed' AND observation_id NOT IN (
                                   SELECT DISTINCT observation_id FROM findings
                               )
                           ) AS normal_observations
                    FROM module_observations WHERE analysis_run_id = ?
                    """, [selected_run]
                ).fetchone()
                cluster_summary = (0, 0, 0, 0.0)
                if clusters_table_exists:
                    cluster_summary = connection.execute(
                        """
                        SELECT count(*) AS clusters,
                               coalesce(sum(observation_count), 0) AS assigned_observations,
                               count(*) FILTER (WHERE anomaly_observation_count > 0) AS clusters_with_findings,
                               coalesce(max(spread_m), 0) AS max_spread_m
                        FROM provisional_module_clusters
                        WHERE analysis_run_id = ?
                        """, [selected_run]
                    ).fetchone()
                finding_types = connection.execute(
                    """
                    SELECT anomaly_type, count(*) AS count
                    FROM findings
                    WHERE analysis_run_id = ?
                    GROUP BY anomaly_type
                    ORDER BY count DESC, anomaly_type
                    """, [selected_run]
                ).fetchdf()
                finding_statuses = connection.execute(
                    """
                    SELECT assessment_status, count(*) AS count
                    FROM findings
                    WHERE analysis_run_id = ?
                    GROUP BY assessment_status
                    ORDER BY count DESC, assessment_status
                    """, [selected_run]
                ).fetchdf()
                delta_values = connection.execute(
                    """
                    SELECT finding_id, anomaly_type, delta_t_k,
                           absolute_temperature_c, reference_temperature_c
                    FROM findings
                    WHERE analysis_run_id = ?
                    ORDER BY abs(delta_t_k) DESC, finding_id
                    """, [selected_run]
                ).fetchdf()
                temperature_trend = connection.execute(
                    """
                    SELECT CAST(capture_time_utc AS VARCHAR) AS capture_time_utc,
                           temperature_median_c, temperature_min_c,
                           temperature_max_c
                    FROM captures
                    WHERE image_kind = 'thermal' AND temperature_median_c IS NOT NULL
                    ORDER BY capture_time_utc NULLS LAST, capture_id
                    """
                ).fetchdf()
            finally:
                connection.close()
            overview_cols = st.columns(4)
            overview_cols[0].metric("Inventory panels", inventory_count)
            overview_cols[1].metric("Stored module observations", observation_counts[0])
            overview_cols[2].metric("Matched Panel IDs", matched_assets)
            overview_cols[3].metric(
                "No finding (assessed)",
                observation_coverage[6],
            )
            anomaly_cols = st.columns(3)
            anomaly_cols[0].metric("Anomaly observations", observation_counts[2])
            anomaly_cols[1].metric("Provisional findings", counts[2])
            anomaly_cols[2].metric(
                "Unique physical modules",
                "Not established",
            )
            if clusters_table_exists:
                st.markdown("**Provisional geographic clusters**")
                cluster_cols = st.columns(4)
                cluster_cols[0].metric("Location clusters", cluster_summary[0])
                cluster_cols[1].metric("Assigned observations", cluster_summary[1])
                cluster_cols[2].metric("Clusters with findings", cluster_summary[2])
                cluster_cols[3].metric("Largest cluster spread", f"{cluster_summary[3]:.2f} m")
                st.caption(
                    "These are proximity-based review groups from geolocated observations. "
                    "They are not Panel IDs, and the count is not a validated physical-module count."
                )
            else:
                st.info(
                    "No provisional location-cluster table exists for this database. "
                    "Run the EXP-045 clustering step after geolocation."
                )
            st.markdown("**Data coverage**")
            coverage_table = pd.DataFrame(
                [
                    {"Metric": "Indexed captures", "Count": capture_counts[0], "Meaning": "All indexed image files"},
                    {"Metric": "Inspection runs", "Count": capture_counts[3], "Meaning": "Distinct indexed flight runs"},
                    {"Metric": "Thermal captures", "Count": capture_counts[1], "Meaning": "DJI *_T.JPG records"},
                    {"Metric": "RGB captures", "Count": capture_counts[2], "Meaning": "DJI *_V.JPG records"},
                    {"Metric": "Paired capture records", "Count": capture_counts[4], "Meaning": "Thermal/RGB pairing links"},
                    {"Metric": "Captures with camera GPS", "Count": capture_counts[5], "Meaning": "Parsed latitude and longitude"},
                    {"Metric": "Saved temperature arrays", "Count": capture_counts[6], "Meaning": "External grid path recorded; summaries alone do not save arrays"},
                    {"Metric": "Captures with temperature summary", "Count": capture_counts[7], "Meaning": "Min/median/max available"},
                    {"Metric": "Stored module observations", "Count": observation_coverage[0], "Meaning": "Boundary observations, may repeat a panel"},
                    {"Metric": "Captures containing observations", "Count": observation_coverage[1], "Meaning": "Frames that reached module observation stage"},
                    {"Metric": "Frame-local module keys", "Count": observation_coverage[2], "Meaning": "Unique keys within frames, not physical IDs"},
                    {"Metric": "Full module proposals", "Count": observation_coverage[3], "Meaning": "Full geometry; does not imply assessment succeeded"},
                    {"Metric": "Located module observations", "Count": observation_coverage[4], "Meaning": "Observation center has coordinates"},
                    {"Metric": "No finding (assessed)", "Count": observation_coverage[6], "Meaning": "Analysis ran with no candidate; not a confirmed healthy panel"},
                    {"Metric": "Anomalous observations", "Count": observation_coverage[5], "Meaning": "At least one finding linked"},
                ]
            )
            st.dataframe(coverage_table, hide_index=True, width="stretch")
            if capture_counts[1] and observation_coverage[1] < capture_counts[1]:
                st.warning(
                    f"Only {observation_coverage[1]} of {capture_counts[1]} thermal captures currently "
                    "contain module proposals in this run. Other frames may be pending or rejected; "
                    "see the flight screening coverage below for per-frame reasons."
                )
            audit_roots = sorted(
                (Path(__file__).resolve().parents[1] / "data").glob("red-team-*/summary.json"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            )
            audit_summary_path = audit_roots[0] if audit_roots else None
            if audit_summary_path is not None and audit_summary_path.is_file():
                try:
                    audit_summary = json.loads(
                        audit_summary_path.read_text(encoding="utf-8")
                    )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    audit_summary = None
                if isinstance(audit_summary, dict):
                    totals = audit_summary.get("totals", {})
                    with st.expander("Red-team anomaly audit (EXP-045)", expanded=False):
                        st.caption(
                            "This is a separate local replay of the full-flight screening. "
                            "It does not replace the selected analysis run."
                        )
                        audit_cols = st.columns(5)
                        audit_cols[0].metric("Frames in scope", int(totals.get("frames", 0)))
                        audit_cols[1].metric("Decoded with SDK", int(totals.get("decoded_frames", 0)))
                        audit_cols[2].metric("Module observations", int(totals.get("modules", 0)))
                        audit_cols[3].metric("Measured", int(totals.get("measured", 0)))
                        audit_cols[4].metric("Not assessed", int(totals.get("not_assessed", 0)))
                        st.write(
                            f"Baseline findings: **{int(audit_summary.get('baseline_findings', 0))}**; "
                            f"replay findings: **{int(totals.get('findings', 0))}**; "
                            f"errors: **{len(audit_summary.get('errors', []))}**."
                        )
                        reason_rows = [
                            {"Reason": reason, "Count": count}
                            for reason, count in audit_summary.get("reasons", {}).items()
                        ]
                        if reason_rows:
                            st.dataframe(pd.DataFrame(reason_rows), hide_index=True, width="stretch")
                        if (audit_summary.get("boundaries_revalidated", False)
                                and audit_summary.get("all_thermal_sources_redecoded", False)):
                            st.success(
                                "All indexed thermal sources were decoded and every accepted "
                                "boundary passed the native-frame sanity gate."
                            )
                        else:
                            st.warning(
                                "Boundary or SDK coverage is incomplete in this replay. "
                                "Findings remain review_required and are not a validated physical-panel count."
                            )
                        if not audit_summary.get("unique_panel_count_verified", False):
                            st.caption(
                                "Unique physical module count and Panel ID identity remain unverified; "
                                "overlapping frames can repeat the same panel."
                            )
                        artifact = audit_summary.get("artifact_database")
                        st.caption(
                            f"Audit artifact: {artifact or (audit_summary_path.parent / 'flight.duckdb')}"
                        )
            st.caption(
                "All module and finding counts use the selected analysis run. Overlapping frames "
                "can observe the same physical panel repeatedly; a unique site count requires "
                "validated tracking. No finding means no candidate under this run's settings, "
                "not confirmation that the panel is healthy. Automatic geometry remains provisional."
            )
            if not inventory_table_exists or not matches_table_exists:
                st.info(
                    "This DuckDB predates the optional Stage 2 tables. Inventory and matched "
                    "Panel ID counts are shown as zero until you click Create / open local "
                    "database once or run the inventory import command."
                )
            coverage_connection = connect_store(database_path, read_only=True)
            try:
                frame_screening = None
                if _table_exists(coverage_connection, "flight_screening_frames"):
                    frame_screening = coverage_connection.execute(
                        """SELECT source_name, status, rgb_candidates, native_candidates,
                                  measured, findings, diagnostics_json, capture_id
                           FROM flight_screening_frames WHERE analysis_run_id = ?
                           ORDER BY source_name""", [selected_run]
                    ).fetchdf()
                module_statuses = coverage_connection.execute(
                    """SELECT assessment_status, count(*) AS count FROM module_observations
                       WHERE analysis_run_id = ? GROUP BY assessment_status""", [selected_run]
                ).fetchdf()
                coverage_audit = build_coverage_audit(coverage_connection, selected_run)
            finally:
                coverage_connection.close()
            coverage_audit_view(coverage_audit)
            if not module_statuses.empty:
                st.markdown("**Module assessment coverage**")
                st.bar_chart(module_statuses.set_index("assessment_status")["count"])
            if frame_screening is not None and not frame_screening.empty:
                st.subheader("Full-flight screening coverage")
                attempted = coverage_audit['counts']['screened_distinct_captures']
                scoped_total = coverage_audit['scope']['indexed_thermal_captures']
                measured_frames = int((frame_screening["measured"] > 0).sum())
                checked = st.columns(4)
                checked[0].metric("Captures with screening records",
                                  f"{attempted} / {scoped_total if scoped_total is not None else 'Unknown'}")
                checked[1].metric("Frames with measured interiors", measured_frames)
                checked[2].metric("Measured module observations", int(frame_screening["measured"].sum()))
                checked[3].metric("Frames with findings", int((frame_screening["findings"] > 0).sum()))
                if scoped_total and coverage_audit['scope']['scope_complete']:
                    st.progress(min(1.0, attempted / scoped_total))
                st.bar_chart(frame_screening.groupby("status").size().rename("frames"))
                st.caption(
                    "Processed includes frames rejected by geometry checks. Measurements use native "
                    "DJI SDK temperatures inside eroded module boundaries. The current automatic "
                    "screen detects local hot regions; no-candidate observations are not certified "
                    "normal, and this does not yet cover every whole-module or electrical fault."
                )
                export_columns = [c for c in frame_screening.columns if c != "diagnostics_json"]
                st.download_button("Download per-frame screening statistics",
                    frame_screening[export_columns].to_csv(index=False),
                    file_name=f"{selected_run}-coverage.csv", mime="text/csv")
                with st.expander("Frame results and paired boundary previews"):
                    frame_filter = st.selectbox("Screening status", ["All", *sorted(frame_screening["status"].unique())])
                    visible_frames = frame_screening if frame_filter == "All" else frame_screening[
                        frame_screening["status"] == frame_filter
                    ]
                    st.dataframe(visible_frames[export_columns], hide_index=True, width="stretch")
                    selected_frame = st.selectbox("Frame preview", visible_frames["source_name"].tolist())
                    diagnostic = json.loads(visible_frames.loc[
                        visible_frames["source_name"] == selected_frame, "diagnostics_json"
                    ].iloc[0])
                    preview_path = diagnostic.get("preview")
                    if preview_path and Path(preview_path).is_file():
                        st.image(preview_path, use_container_width=True,
                            caption="Thermal + paired RGB. Green: provisional measured interior; "
                                    "grey: rejected boundary; red: local thermal candidate.")
                    elif diagnostic.get('geometry_method') == 'native_observed_boundaries':
                        st.caption('Native thermal structure; RGB registration is not required for these proposals.')
                        show_native_support = st.button('Show native boundaries and measurement support', key='show-native-frame-support')
                        review_native_gaps = st.button('Review unresolved thermal regions', key='review-native-frame-gaps')
                        if show_native_support or review_native_gaps:
                            from pvdelta.finding_preview import native_frame_preview
                            from pvdelta.measurement_profile import read_measurement_profile

                            selected_capture_id = visible_frames.loc[
                                visible_frames['source_name'] == selected_frame, 'capture_id'].iloc[0]
                            preview_connection = connect_store(database_path, read_only=True)
                            try:
                                profile = read_measurement_profile(preview_connection, selected_run, selected_capture_id)
                                image_source = preview_connection.execute('''SELECT c.source_path,r.source_root
                                    FROM captures c LEFT JOIN inspection_runs r ON r.run_id=c.run_id
                                    WHERE c.capture_id=?''', [selected_capture_id]).fetchone()
                            finally:
                                preview_connection.close()
                            source_path = Path(image_source[1] or '')/image_source[0]
                        if show_native_support:
                            native_preview = native_frame_preview(source_path, diagnostic, profile,
                                sdk_library=sdk_library_text.strip() or None)
                            st.image(native_preview.image, caption=native_preview.message, width='stretch')
                            if not native_preview.overlay_applied:
                                st.warning('Native support overlay withheld; check source, profile and geometry.')
                        if review_native_gaps:
                            from pvdelta.native_gap_review import export_native_frame_gaps

                            gap_output = (PRIVATE_WORKSPACE/'native-gap-reviews'/
                                (str(selected_capture_id).replace(':','-')+'-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S-%f')))
                            try:
                                folder, gap_report = export_native_frame_gaps(source_path, diagnostic, profile,
                                    gap_output, sdk_library=sdk_library_text.strip() or None)
                                st.session_state['native-gap-pending-bundle'] = str(folder)
                                st.session_state['native-gap-last-export'] = (selected_frame, str(folder), len(gap_report['evidence']))
                                st.rerun()
                            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                                st.error(f'Unresolved-region review failed: {exc}')
                        last_gap_export = st.session_state.get('native-gap-last-export')
                        if last_gap_export and last_gap_export[0] == selected_frame:
                            st.info(f'{last_gap_export[2]} unassigned evidence records. Open Unassigned evidence to review. '
                                    'These are unresolved regions, not confirmed panels or faults.')
                    else:
                        st.info(f"No boundary preview for this frame: {diagnostic['status']}")
                    if diagnostic.get("error"):
                        st.error(diagnostic["error"])
            charts_left, charts_right = st.columns(2)
            with charts_left:
                st.markdown("**Anomaly categories**")
                if finding_types.empty:
                    st.info("No findings stored yet.")
                else:
                    st.bar_chart(finding_types.set_index("anomaly_type")["count"])
            with charts_right:
                st.markdown("**Assessment status**")
                if finding_statuses.empty:
                    st.info("No assessment statuses stored yet.")
                else:
                    st.bar_chart(finding_statuses.set_index("assessment_status")["count"])
            if not delta_values.empty:
                st.markdown("**Finding ΔT and temperatures**")
                st.dataframe(delta_values, hide_index=True, width="stretch")
            if not temperature_trend.empty:
                st.markdown("**Recorded source temperature trend**")
                trend = temperature_trend.copy()
                trend["capture_time_utc"] = trend["capture_time_utc"].fillna(
                    "unknown capture time"
                )
                st.line_chart(
                    trend.set_index("capture_time_utc")[[
                        "temperature_median_c", "temperature_min_c", "temperature_max_c"
                    ]]
                )
                st.caption('Capture-level source summaries; these do not represent a different '
                           'measurement profile selected for anomaly reanalysis.')
            st.caption(
                "Anomaly categories are detection outputs, not confirmed physical causes. "
                "Cause attribution needs reviewed imagery, field evidence or SCADA."
            )
            selected_anomaly_map_point = None
            selected_timeline_frame = None
            if counts[2]:
                connection = connect_store(database_path, read_only=True)
                finding_scope_join = (
                    "LEFT JOIN finding_evidence AS fe ON fe.finding_id = f.finding_id"
                    if evidence_table_exists else ""
                )
                finding_evidence_select = (
                    "fe.panel_support_status, fe.panel_support_ratio, "
                    "fe.panel_texture_std, fe.evidence_group_id, fe.repeat_capture_count"
                    if evidence_table_exists else
                    "NULL AS panel_support_status, NULL AS panel_support_ratio, "
                    "NULL AS panel_texture_std, "
                    "NULL AS evidence_group_id, NULL AS repeat_capture_count"
                )
                finding_scope_clause = (
                    "AND fe.panel_support_status = 'panel_supported'"
                    if evidence_table_exists and finding_scope == "Legacy palette shortlist (experimental)"
                    else ""
                )
                finding_rows = connection.execute(
                    f"""
                    SELECT f.finding_id,
                           f.analysis_run_id AS run,
                           m.frame_module_id AS module,
                           f.anomaly_type,
                           f.assessment_status,
                           round(f.delta_t_k, 2) AS delta_t_k,
                           round(f.absolute_temperature_c, 2) AS temperature_c,
                           round(f.reference_temperature_c, 2) AS reference_c,
                           c.source_path AS thermal_file,
                           rgb.source_path AS rgb_file,
                           {finding_evidence_select},
                           round(coalesce(f.estimated_latitude, m.estimated_latitude), 7) AS latitude,
                           round(coalesce(f.estimated_longitude, m.estimated_longitude), 7) AS longitude
                    FROM findings AS f
                    JOIN module_observations AS m USING (observation_id)
                    JOIN captures AS c ON c.capture_id = m.capture_id
                    LEFT JOIN captures AS rgb ON rgb.capture_id = c.paired_capture_id
                    {finding_scope_join}
                    WHERE f.analysis_run_id = ? {finding_scope_clause}
                    ORDER BY f.analysis_run_id, m.frame_module_id, f.finding_id
                    """,
                    [selected_run],
                ).fetchdf()
                connection.close()
                if "thermal_file" in finding_rows:
                    finding_rows["thermal_file"] = finding_rows["thermal_file"].map(
                        lambda value: Path(value).name if value else value
                    )
                if "rgb_file" in finding_rows:
                    finding_rows["rgb_file"] = finding_rows["rgb_file"].map(
                        lambda value: Path(value).name if value else value
                    )
                st.subheader("Stage 1 findings")
                if evidence_table_exists and finding_scope == "Legacy palette shortlist (experimental)":
                    supported_group_count = int(
                        finding_rows["evidence_group_id"].nunique()
                        if not finding_rows.empty else 0
                    )
                    st.caption(
                        f"Legacy palette shortlist: {len(finding_rows)} candidates in "
                        f"{supported_group_count} provisional physical-location groups. "
                        "Excluded candidates are unverified, not confirmed vegetation. Use All raw candidates."
                    )
                else:
                    st.caption(
                        "Showing all raw detector candidates. Panel support and repeat-view "
                        "evidence are displayed in the table for red-team review."
                    )
                st.dataframe(finding_rows, hide_index=True, width="stretch")
                st.caption(
                    "The overview and findings table use the same selected analysis run. "
                    "Findings are radiometric candidates that require review."
                )
                finding_ids = finding_rows["finding_id"].tolist()
                selected_finding_id = finding_navigation(
                    finding_ids, key="stage1-finding-id", label="Finding detail",
                )
                detail_connection = connect_store(database_path, read_only=True)
                screening_available = _table_exists(detail_connection, "flight_screening_frames")
                screening_select = "s.diagnostics_json" if screening_available else "NULL"
                screening_join = (
                    "LEFT JOIN flight_screening_frames AS s "
                    "ON s.analysis_run_id = f.analysis_run_id AND s.capture_id = c.capture_id"
                    if screening_available else ""
                )
                detail = detail_connection.execute(
                    f"""
                    SELECT f.finding_id, f.anomaly_type, f.assessment_status,
                           f.delta_t_k, f.absolute_temperature_c,
                           f.reference_temperature_c, f.confidence,
                           f.n_region_px, f.n_reference_px,
                           f.region_max_temperature_c, f.region_p95_temperature_c,
                           f.region_max_residual_k, f.reference_method,
                           f.delta_method, f.pixel_polygon_json,
                           c.source_path AS thermal_path, c.run_id,
                           c.capture_id AS thermal_capture_id,
                           m.observation_id, m.frame_module_id,
                           coalesce(f.estimated_latitude, m.estimated_latitude) AS latitude,
                           coalesce(f.estimated_longitude, m.estimated_longitude) AS longitude,
                           r.source_root, rgb.source_path AS rgb_path,
                           c.width_px AS thermal_width, c.height_px AS thermal_height,
                           {screening_select} AS screening_diagnostics, f.region_pixels_yx_json,
                           m.pixel_polygon_json AS module_polygon_json, c.camera_model,
                           rgb.source_hash AS rgb_hash, c.source_hash AS thermal_hash
                    FROM findings AS f
                    JOIN module_observations AS m USING (observation_id)
                    JOIN captures AS c ON c.capture_id = m.capture_id
                    LEFT JOIN captures AS rgb ON rgb.capture_id = c.paired_capture_id
                    LEFT JOIN inspection_runs AS r ON r.run_id = c.run_id
                    {screening_join}
                    WHERE f.finding_id = ? AND f.analysis_run_id = ?
                    """,
                    [selected_finding_id, selected_run],
                ).fetchone()
                from pvdelta.measurement_profile import read_measurement_profile

                selected_measurement_profile, measurement_profile_error = None, None
                try:
                    if detail is not None:
                        selected_measurement_profile = read_measurement_profile(detail_connection, selected_run, detail[17])
                except (ValueError, TypeError, KeyError) as exc:
                    measurement_profile_error = str(exc)
                finally:
                    detail_connection.close()
                if detail is not None:
                    detail_columns = [
                        "finding_id", "anomaly_type", "assessment_status", "delta_t_k",
                        "absolute_temperature_c", "reference_temperature_c", "confidence",
                        "n_region_px", "n_reference_px", "region_max_temperature_c",
                        "region_p95_temperature_c", "region_max_residual_k", "reference_method",
                        "delta_method", "pixel_polygon_json", "thermal_path", "run_id",
                        "thermal_capture_id", "observation_id", "frame_module_id", "latitude",
                        "longitude", "source_root", "rgb_path", "thermal_width", "thermal_height", "screening_diagnostics", "region_pixels_yx_json", "module_polygon_json",
                        "camera_model", "rgb_hash", "thermal_hash",
                    ]
                    detail_record = dict(zip(detail_columns, detail, strict=True))
                    st.markdown("**Selected anomaly detail**")
                    with st.expander('Measurement parameters used for this finding'):
                        if measurement_profile_error:
                            st.error('Saved measurement profile is invalid: '+measurement_profile_error)
                        elif selected_measurement_profile is None:
                            st.info('This legacy analysis has no saved per-analysis measurement profile. '
                                    'Capture metadata alone does not establish which settings produced this finding.')
                        else:
                            st.json(selected_measurement_profile)
                            st.caption('Recorded and effective SDK settings belong to this analysis and source. '
                                       'Field conditions and fault cause still require verification.')
                    stats = {
                        key: detail_record[key]
                        for key in (
                            "anomaly_type", "assessment_status", "delta_t_k",
                            "absolute_temperature_c", "reference_temperature_c", "confidence",
                            "n_region_px", "n_reference_px", "region_max_temperature_c",
                            "region_p95_temperature_c", "region_max_residual_k",
                            "reference_method", "delta_method",
                            "latitude", "longitude", "frame_module_id",
                        )
                    }
                    stats["thermal_file"] = Path(detail_record["thermal_path"]).name
                    stats["rgb_file"] = (
                        Path(detail_record["rgb_path"]).name
                        if detail_record["rgb_path"] else None
                    )
                    stats.pop("thermal_path", None)
                    stats.pop("rgb_path", None)
                    st.dataframe([stats], hide_index=True, width="stretch")
                    anomaly_latitude = _safe_ratio(detail_record["latitude"])
                    anomaly_longitude = _safe_ratio(detail_record["longitude"])
                    selected_anomaly_map_point = (
                        {
                            "latitude": anomaly_latitude,
                            "longitude": anomaly_longitude,
                            "file_name": Path(detail_record["thermal_path"]).name,
                            "delta_t_k": float(detail_record["delta_t_k"]),
                        }
                        if anomaly_latitude is not None and anomaly_longitude is not None
                        else None
                    )
                    thermal_path = _resolve_capture_path(
                        detail_record["source_root"], detail_record["thermal_path"]
                    )
                    rgb_path = _resolve_capture_path(
                        detail_record["source_root"], detail_record["rgb_path"]
                    )
                    image_left, image_right, image_original = st.columns(3)
                    thermal_preview = None
                    with image_left:
                        st.markdown("**Thermal anomaly crop**")
                        if thermal_path is None:
                            st.info("Thermal source image is not available at its indexed path.")
                        elif measurement_profile_error:
                            st.warning('Thermal overlay withheld: the saved measurement profile failed its identity check.')
                        else:
                            try:
                                from pvdelta.finding_preview import thermal_finding_preview
                                thermal_preview = thermal_finding_preview(
                                    thermal_path, json.loads(detail_record["pixel_polygon_json"]),
                                    sdk_library=Path(sdk_library_text).expanduser() if sdk_library_text.strip() else None,
                                    native_size=(detail_record['thermal_width'], detail_record['thermal_height']),
                                    region_pixels_yx=json.loads(detail_record['region_pixels_yx_json'] or 'null'),
                                    module_polygon=json.loads(detail_record['module_polygon_json']),
                                    measurement_profile=selected_measurement_profile,
                                    expected_source_hash=detail_record['thermal_hash'],
                                )
                                st.image(
                                    thermal_preview.image,
                                    caption=thermal_preview.message,
                                    use_container_width=True,
                                )
                                if not thermal_preview.overlay_applied:
                                    st.warning("Native overlay withheld. Check the DJI Thermal SDK library folder.")
                            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                                st.warning(f"Thermal preview could not be rendered: {exc}")
                    with image_right:
                        st.markdown("**Paired RGB frame with anomaly overlay**")
                        if rgb_path is None:
                            st.info("No paired RGB frame is available at its indexed path.")
                        else:
                            try:
                                registration = _load_rgb_registration()
                                diagnostics = (
                                    json.loads(detail_record["screening_diagnostics"])
                                    if detail_record["screening_diagnostics"]
                                    else {}
                                )
                                shift = tuple(diagnostics.get("shift_xy", [0.0, 0.0]))
                                matching_module = next((m for m in diagnostics.get('modules', [])
                                    if m.get('module') == detail_record['frame_module_id']), {})
                                local_validated = (
                                    matching_module.get('accepted', False)
                                    and diagnostics.get('geometry_semantically_verified') is not False
                                    and diagnostics.get('heldout_p90_px', float('inf')) <= 5
                                    and len(shift) == 2 and np.isfinite(shift).all()
                                    and (detail_record['thermal_width'], detail_record['thermal_height']) == (640, 512)
                                )
                                if registration is None or not local_validated:
                                    from pvdelta.rgb_pair_check import rgb_pair_check_view
                                    from pvdelta.rgb_preview import rgb_finding_preview

                                    image_slot = st.container()
                                    pair_check = rgb_pair_check_view(thermal_path, rgb_path,
                                        selected_measurement_profile, diagnostics, detail_record['frame_module_id'],
                                        registration, sdk_library=sdk_library_text.strip() or None,
                                        key='advanced-rgb-check', automatic_junctions=True) if thermal_preview and thermal_preview.overlay_applied else None
                                    rgb_preview = rgb_finding_preview(rgb_path,
                                        json.loads(detail_record['pixel_polygon_json']),
                                        registration=registration, camera_model=detail_record['camera_model'],
                                        native_size=(detail_record['thermal_width'], detail_record['thermal_height']),
                                        source_root=detail_record['source_root'], expected_rgb_hash=detail_record['rgb_hash'],
                                        native_verified=(thermal_preview is not None and thermal_preview.overlay_applied),
                                        module_polygon=json.loads(detail_record['module_polygon_json']),
                                        pair_check=pair_check, measurement_profile=selected_measurement_profile)
                                    with image_slot:
                                        st.image(
                                            rgb_preview.image, caption=rgb_preview.message,
                                            use_container_width=True,
                                        )
                                        if rgb_preview.overlay_applied:
                                            if rgb_preview.registration_status == 'approximate':
                                                st.warning('Approximate RGB location. Local cross-camera alignment is not verified.')
                                            else:
                                                st.caption('Module-edge checks passed; physical panel identity remains unverified.')
                                        else:
                                            st.info(rgb_preview.message)
                                else:
                                    native_pixels = np.asarray(json.loads(
                                        detail_record["region_pixels_yx_json"] or "[]"
                                    ), dtype=float)
                                    if native_pixels.ndim == 2 and native_pixels.shape[1] == 2 and len(native_pixels):
                                        xy = native_pixels[:, ::-1] - np.asarray(shift)
                                        mapped = _quadratic_points(xy, registration["model"])
                                        centre = xy.mean(axis=0)
                                        basis = _quadratic_points(np.array([
                                            centre, centre + [1, 0], centre + [0, 1]
                                        ]), registration["model"])
                                        jacobian = np.column_stack((basis[1] - basis[0], basis[2] - basis[0]))
                                        margin = max(1.0, float(np.linalg.norm(jacobian, 2)) * max(
                                            float(diagnostics["heldout_p90_px"]),
                                            float(matching_module.get("gate", {}).get("max_offset_px", 0)),
                                        ))
                                        with Image.open(rgb_path) as source_rgb:
                                            surface = check_rgb_surface(
                                                np.asarray(source_rgb.convert("RGB")), mapped,
                                                np.asarray(matching_module["rgb_polygon"]),
                                                uncertainty_px=margin,
                                            )
                                        if surface["status"] == "geometry_consistent":
                                            st.caption("RGB projection agrees with the stored polygon; thermal panel membership still needs review.")
                                        else:
                                            st.warning("RGB surface check: " + surface["reason"] + ". Review required; candidate retained.")
                                        with st.expander("RGB surface-check measurements"):
                                            st.json(surface)
                                    rgb_crop = _rgb_finding_crop(
                                        rgb_path,
                                        detail_record["pixel_polygon_json"],
                                        registration=registration,
                                        native_shift_xy=shift,
                                    )
                                    st.image(
                                        rgb_crop,
                                        caption=(
                                            "Thermal finding projected onto paired RGB; "
                                            f"red outline is enlarged {DISPLAY_OUTLINE_SCALE}× for visibility, not measured extent"
                                        ),
                                        use_container_width=True,
                                    )
                            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                                st.warning(f"RGB preview could not be rendered: {exc}")
                    with image_original:
                        st.markdown("**Original RGB · no marks**")
                        try:
                            if rgb_path:
                                from pvdelta.rgb_preview import original_rgb_image

                                st.image(original_rgb_image(rgb_path, expected_source_hash=detail_record['rgb_hash']),
                                         caption=rgb_path.name, use_container_width=True)
                            else:
                                st.info("Paired RGB source image unavailable.")
                        except (OSError, ValueError, TypeError) as exc:
                            st.warning(f"Original RGB unavailable: {exc}")
                    st.caption(
                        "The thermal polygon is kept in the native 640×512 grid. The RGB outline "
                        "is a visual correspondence, not a second temperature measurement. A fine "
                        "hotspot outline requires a compatible calibration and passing local geometry; "
                        "a cyan module proposal alone does not establish that correspondence."
                    )
                    with st.expander("Native mask support / edge sensitivity"):
                        from pvdelta.mask_review import review_region_support
                        from pvdelta.regions import polygon_to_mask
                        try:
                            support_review = review_region_support(
                                polygon_to_mask(json.loads(detail_record['module_polygon_json']),
                                    (detail_record['thermal_height'], detail_record['thermal_width'])),
                                json.loads(detail_record['region_pixels_yx_json'] or 'null'),
                            )
                            st.write("Finding support: " + support_review['status'].replace('_', ' '))
                            if support_review['insets']:
                                st.dataframe(support_review['insets'], hide_index=True, width='stretch')
                            st.caption(
                                "Exact stored finding pixels retained after 0/1/2/3/5 native-pixel "
                                "polygon insets. This probes boundary sensitivity; it does not "
                                "validate panel membership or classify vegetation. Edge hotspots "
                                "remain visible and require review."
                            )
                        except (TypeError, ValueError, json.JSONDecodeError) as exc:
                            st.info(f"Mask support unavailable: {exc}")
                    _render_repeat_view_review(database_path, selected_finding_id, key="stage1",
                        sdk_library=sdk_library_text.strip() or None)
                    with st.expander("User review", expanded=False):
                        review_connection = connect_store(database_path, read_only=True)
                        review_state = review_connection.execute(
                            """
                            SELECT review_status, reviewer, note,
                                   CAST(reviewed_at_utc AS VARCHAR) AS reviewed_at_utc
                            FROM finding_reviews
                            WHERE finding_id = ?
                            """,
                            [selected_finding_id],
                        ).fetchone()
                        review_connection.close()
                        current_status = review_state[0] if review_state else "wait"
                        st.caption(
                            "OK confirms the anomaly; Not OK marks a false positive; "
                            "Wait keeps it open for more evidence."
                        )
                        review_form_left, review_form_right = st.columns([2, 1])
                        with review_form_left:
                            detail_note = st.text_input(
                                "Review note (optional)",
                                value=review_state[2] if review_state and review_state[2] else "",
                                key=f"crop-review-note-{selected_finding_id}",
                            )
                        with review_form_right:
                            detail_reviewer = st.text_input(
                                "Reviewer",
                                value=review_state[1] if review_state and review_state[1] else "",
                                key=f"crop-reviewer-{selected_finding_id}",
                            )
                        detail_check_left, detail_check_right = st.columns(2)
                        detail_panel_support = detail_check_left.checkbox(
                            "Panel support confirmed",
                            key=f"crop-review-panel-support-{selected_finding_id}",
                        )
                        detail_glare_checked = detail_check_right.checkbox(
                            "Glare / reflection checked",
                            key=f"crop-review-glare-{selected_finding_id}",
                        )
                        detail_repeat_view = detail_check_left.checkbox(
                            "Repeat or alternate view checked",
                            key=f"crop-review-repeat-{selected_finding_id}",
                        )
                        detail_conditions_documented = detail_check_right.checkbox(
                            "Radiometric conditions documented",
                            key=f"crop-review-conditions-{selected_finding_id}",
                        )
                        detail_checklist_passed = all(
                            (
                                detail_panel_support,
                                detail_glare_checked,
                                detail_repeat_view,
                                detail_conditions_documented,
                            )
                        )
                        st.caption(
                            "OK is enabled only after this evidence gate passes. It is a project "
                            "review decision, not an IEC conformity certificate."
                        )
                        status_line, review_button_row = st.columns([1, 3])
                        status_line.caption(f"Current: **{current_status}**")
                        with review_button_row:
                            detail_buttons = st.columns(3)
                        detail_actions = (
                            (detail_buttons[0], "OK", "ok"),
                            (detail_buttons[1], "Not OK", "not_ok"),
                            (detail_buttons[2], "Wait", "wait"),
                        )
                        for button_column, label, status in detail_actions:
                            if button_column.button(
                                label,
                                key=f"crop-review-{status}-{selected_finding_id}",
                                disabled=status == "ok" and not detail_checklist_passed,
                            ):
                                detail_checklist_note = (
                                    "Evidence checklist: "
                                    f"panel_support={'PASS' if detail_panel_support else 'PENDING'}; "
                                    f"glare_checked={'PASS' if detail_glare_checked else 'PENDING'}; "
                                    f"repeat_view={'PASS' if detail_repeat_view else 'PENDING'}; "
                                    f"radiometric_conditions={'PASS' if detail_conditions_documented else 'PENDING'}."
                                )
                                detail_combined_note = " ".join(
                                    item for item in (detail_note.strip(), detail_checklist_note) if item
                                )
                                write_connection = connect_store(database_path)
                                try:
                                    upsert_finding_review(
                                        write_connection,
                                        finding_id=selected_finding_id,
                                        review_status=status,
                                        reviewer=detail_reviewer.strip() or None,
                                        note=detail_combined_note or None,
                                    )
                                finally:
                                    write_connection.close()
                                st.success(f"Saved review: {label}.")
                                st.rerun()
            ranking_connection = connect_store(database_path, read_only=True)
            selected_views = ranking_connection.execute(
                """
                SELECT panel_key, rgb_capture_id, thermal_capture_id,
                       round(ranking_score, 3) AS score, view_rank,
                       latitude, longitude
                FROM view_rankings
                WHERE selection_status = 'selected'
                ORDER BY run_id, panel_key
                """
            ).fetchdf() if _table_exists(ranking_connection, "view_rankings") else pd.DataFrame()
            ranking_connection.close()
            if not selected_views.empty:
                st.subheader("Selected RGB views")
                st.dataframe(selected_views, hide_index=True, width="stretch")
                st.caption(
                    "Panel keys come from reviewed geometry, CAD or a validated tracker. "
                    "RGB selection does not select the thermal measurement."
                )
            if features:
                st.subheader("Frame timeline")
                timeline_key = "local-map-frame-index"
                findings_only = st.checkbox(
                    "Show frames with findings only",
                    False,
                    help="Keep the full timeline by default; enable this to step through frames that contain stored anomaly candidates.",
                )
                timeline_frames = (
                    [item for item in features if item.get("finding_count", 0) > 0]
                    if findings_only else features
                )
                if not timeline_frames:
                    st.info("No indexed frames contain findings in the selected scope.")
                    timeline_frames = features
                if not 1 <= int(st.session_state.get(timeline_key, 1)) <= len(timeline_frames):
                    st.session_state[timeline_key] = 1
                frame_position = int(st.session_state.get(timeline_key, 1))
                previous_frame, next_frame = st.columns(2)
                with previous_frame:
                    if st.button(
                        "← Previous frame",
                        key="local-map-frame-previous",
                        disabled=frame_position <= 1,
                        use_container_width=True,
                    ):
                        st.session_state[timeline_key] = frame_position - 1
                        st.rerun()
                with next_frame:
                    if st.button(
                        "Next frame →",
                        key="local-map-frame-next",
                        disabled=frame_position >= len(timeline_frames),
                        use_container_width=True,
                    ):
                        st.session_state[timeline_key] = frame_position + 1
                        st.rerun()
                frame_position = st.slider(
                    "Thermal frame",
                    min_value=1,
                    max_value=len(timeline_frames),
                    key=timeline_key,
                    help="Move through thermal captures in acquisition order. The map marker and previews follow this frame.",
                )
                selected_timeline_frame = timeline_frames[frame_position - 1]
                timeline_name = Path(selected_timeline_frame["source_path"]).name
                timeline_time = selected_timeline_frame.get("capture_time_utc") or "time unavailable"
                timeline_lat = selected_timeline_frame["camera_latitude"]
                timeline_lon = selected_timeline_frame["camera_longitude"]
                timeline_meta = {
                    "frame": f"{frame_position} / {len(timeline_frames)}",
                    "thermal_file": timeline_name,
                    "capture_time": timeline_time,
                    "camera_latitude": round(float(timeline_lat), 7),
                    "camera_longitude": round(float(timeline_lon), 7),
                    "relative_altitude_m": selected_timeline_frame.get("relative_altitude_m"),
                    "temperature_min_c": selected_timeline_frame.get("temperature_min_c"),
                    "temperature_max_c": selected_timeline_frame.get("temperature_max_c"),
                    "module_observations": selected_timeline_frame.get("module_observation_count", 0),
                    "finding_candidates": selected_timeline_frame.get("finding_count", 0),
                    "footprint": (
                        f"calibrated ±{selected_timeline_frame['footprint_accuracy_m']:.1f} m"
                        if selected_timeline_frame.get("footprint_geojson") is not None
                        else "camera GPS only"
                    ),
                }
                st.dataframe([timeline_meta], hide_index=True, width="stretch")
                timeline_rgb_path = _resolve_capture_path(
                    selected_timeline_frame.get("source_root"),
                    selected_timeline_frame.get("rgb_source_path"),
                )
                timeline_thermal_path = _resolve_capture_path(
                    selected_timeline_frame.get("source_root"),
                    selected_timeline_frame.get("source_path"),
                )
                timeline_left, timeline_right = st.columns(2)
                with timeline_left:
                    st.markdown("**Thermal frame**")
                    if timeline_thermal_path is None:
                        st.info("Thermal source image is not available at its indexed path.")
                    else:
                        try:
                            st.image(
                                _frame_preview(timeline_thermal_path),
                                caption=timeline_name,
                                use_container_width=True,
                            )
                        except (OSError, ValueError) as exc:
                            st.warning(f"Thermal frame could not be opened: {exc}")
                with timeline_right:
                    st.markdown("**Paired RGB frame**")
                    if timeline_rgb_path is None:
                        st.info("No paired RGB frame is available for this thermal capture.")
                    else:
                        try:
                            st.image(
                                _frame_preview(timeline_rgb_path),
                                caption=timeline_rgb_path.name,
                                use_container_width=True,
                            )
                        except (OSError, ValueError) as exc:
                            st.warning(f"RGB frame could not be opened: {exc}")
                st.caption(
                    "The orange map marker is the camera position for this frame. "
                    "It becomes a photographed-panel location only when a calibrated footprint is stored."
                )
                online_basemaps = st.checkbox(
                    "Enable online map backgrounds",
                    False,
                    help=(
                        "Off by default for privacy. Enabling a background requests the viewed "
                        "map area from OpenStreetMap, OpenTopoMap or Google satellite imagery."
                    ),
                )
                basemap = (
                    st.selectbox("Map background", list(BASEMAPS))
                    if online_basemaps
                    else None
                )
                map_options_left, map_options_right = st.columns(2)
                with map_options_left:
                    show_flight_route = st.checkbox(
                        "Show flight route", True,
                        help="Draw the time-ordered thermal capture track and mark its start/end.",
                    )
                    show_camera_positions = st.checkbox(
                        "Show thermal camera positions", True,
                        help="Show the individual GPS points used to build the route.",
                    )
                with map_options_right:
                    fit_to_route = st.checkbox(
                        "Fit map to full route", True,
                        help="Zoom to the complete valid flight track after each refresh.",
                    )
                show_laser_targets = st.checkbox("Show recorded laser target points", False)
                try:
                    _render_local_map(
                        features,
                        basemap=basemap,
                        show_laser_targets=show_laser_targets,
                        show_camera_positions=show_camera_positions,
                        show_flight_route=show_flight_route,
                        fit_to_route=fit_to_route,
                        selected_frame=selected_timeline_frame,
                        selected_anomaly=selected_anomaly_map_point,
                    )
                except ImportError:
                    st.warning(
                        "Folium map packages are not installed. Reinstall the dashboard extra."
                    )
                st.caption(
                    "Use the map control to switch between the selected street, terrain/elevation "
                    "and satellite background. The blue line is the time-ordered flight route; "
                    "its start and end are marked separately. Optional blue points are thermal "
                    "camera positions and green points are recorded laser targets. Orange polygons "
                    "appear only when a calibrated footprint has been stored; camera GPS alone is "
                    "not treated as the photographed panel location."
                )
            else:
                st.info(
                    "The database has no geolocated captures yet. Folder ingestion and "
                    "calibrated footprint generation are the next pipeline step."
                )
        else:
            st.info(
                "No database exists at this path. Create it when you are ready to ingest a flight."
            )

with ortho_tab:
    st.subheader("ODM photogrammetry")
    st.caption(
        "Local RGB reconstruction from overlapping source photographs. "
        "Pilot coverage is separate from full-flight coverage. "
        "Thermal measurements remain in the original DJI radiometric frames."
    )
    with st.expander("Prepare a new RGB pilot"):
        st.caption("Copies a contiguous near-nadir block with original EXIF; checks every copy. "
                   "Start with a pilot before processing the complete flight.")
        if database_path.is_file():
            import duckdb

            with duckdb.connect(str(database_path), read_only=True) as odm_connection:
                odm_runs = [r[0] for r in odm_connection.execute(
                    "SELECT DISTINCT run_id FROM captures WHERE image_kind='rgb' ORDER BY run_id"
                ).fetchall()]
            if odm_runs:
                odm_run = st.selectbox("Source flight", odm_runs, key="odm_source_flight")
                odm_limit = st.selectbox("Pilot RGB frames", [60, 100, 160], key="odm_limit")
                if st.button("Prepare RGB pilot"):
                    from pvdelta.odm import prepare_project

                    odm_new = database_path.parent / "odm" / dt.datetime.now(dt.timezone.utc).strftime(
                        "pilot-%Y%m%d-%H%M%S")
                    try:
                        with st.spinner("Copying and verifying pilot images"):
                            prepare_project(database_path, odm_run, odm_new, odm_limit)
                        st.rerun()
                    except (OSError, ValueError) as exc:
                        st.error(str(exc))
    odm_jobs = sorted((database_path.parent / "odm").glob("*/pvdelta-odm.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if st.button("Refresh reconstruction status", key="odm_refresh"):
        st.rerun()
    if not odm_jobs:
        st.info("No local ODM project prepared yet.")
    else:
        odm_job_path = st.selectbox("RGB reconstruction job", odm_jobs,
                                   format_func=lambda p: p.parent.name)
        try:
            odm_state = json.loads(odm_job_path.read_text(encoding="utf-8"))
            odm_cols = st.columns(4)
            odm_cols[0].metric("Processing status", odm_state.get("status", "unknown"))
            odm_cols[1].metric("Scope", odm_state.get("scope", "unknown"))
            odm_cols[2].metric("Selected RGB frames", odm_state.get("selected_count", 0))
            odm_cols[3].metric("Indexed RGB frames", odm_state.get("indexed_rgb_count", 0))
            st.caption(f"Updated (UTC): {odm_state.get('updated_utc', 'unknown')}")
            odm_home = Path(__file__).resolve().parents[1] / "data/tools/ODM"
            if odm_state.get("status") == "prepared":
                runtime_ready = (odm_home / "venv/Scripts/python.exe").is_file()
                if not runtime_ready:
                    st.warning("Install the native ODM runtime under data/tools/ODM to start.")
                if st.button("Start RGB reconstruction", disabled=not runtime_ready):
                    runner = Path(__file__).resolve().parents[1] / "examples/build_odm.py"
                    with (odm_job_path.parent / "worker.log").open("ab") as worker_log:
                        subprocess.Popen(
                            [sys.executable, str(runner), "run", "--project",
                             str(odm_job_path.parent), "--odm-home", str(odm_home)],
                            cwd=runner.parent.parent, stdout=worker_log, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                        )
                    st.success("Worker started. Refresh status to follow reconstruction.")
            if odm_state.get("error"):
                st.error(odm_state["error"])
            if "largest_component_fraction" in odm_state:
                st.write("Images in largest reconstructed component: "
                         f"{odm_state['largest_component_fraction']:.1%}")
            odm_stats_path = odm_job_path.parent / "opensfm/stats/stats.json"
            if odm_stats_path.is_file():
                odm_stats = json.loads(odm_stats_path.read_text(encoding="utf-8"))
                odm_reconstruction = odm_stats.get("reconstruction_statistics", {})
                stat_cols = st.columns(3)
                stat_cols[0].metric("Reconstructed cameras",
                                    odm_reconstruction.get("reconstructed_shots_count", 0))
                stat_cols[1].metric("Components", odm_reconstruction.get("components", 0))
                reprojection = odm_reconstruction.get("reprojection_error_pixels")
                stat_cols[2].metric("Reprojection error (px)",
                                    f"{reprojection:.2f}" if reprojection is not None else "—")
                st.caption("Reprojection error measures internal image alignment, "
                           "not absolute GPS accuracy.")
            if odm_state.get("status") == "running":
                depth_dir = odm_job_path.parent / "opensfm/undistorted/openmvs/depthmaps"
                if depth_dir.is_dir():
                    st.caption(f"Depth-map files written: {len(list(depth_dir.glob('*.dmap')))}. "
                               "Surface and orthophoto processing follow this stage.")
            odm_preview = odm_job_path.parent / "rgb_preview.png"
            if odm_state.get("status") == "needs_review" and odm_preview.is_file():
                st.image(str(odm_preview), caption="ODM RGB reconstruction — geometry review required",
                         use_container_width=True)
                if odm_state.get("visual_review"):
                    st.warning(odm_state["visual_review"])
                odm_detail = odm_job_path.parent / "rgb_detail.png"
                if odm_detail.is_file():
                    with st.expander("Native-resolution panel detail"):
                        st.image(str(odm_detail), use_container_width=True)
                st.code(str(odm_job_path.parent / "odm_orthophoto/odm_orthophoto.tif"))
            refinements = sorted(odm_job_path.parent.glob("table-refinement*/report.json"),
                                 key=lambda path: path.stat().st_mtime, reverse=True)
            if refinements:
                with st.expander("Table geometry refinement — before / after", expanded=True):
                    refinement_path = st.selectbox("Table refinement experiment", refinements,
                                                   format_func=lambda path: path.parent.name)
                    refinement = json.loads(refinement_path.read_text(encoding="utf-8"))
                    refined_tables = refinement.get("tables", [])
                    st.caption(f"Experiment: {refinement_path.parent.name}. "
                               f"{len(refined_tables)} tables re-rendered; "
                               f"{len(refinement.get('rejected', []))} candidates withheld.")
                    st.caption(f"Registration: {refinement.get('registration_mode', 'global')}. "
                               "Coverage and feature checks do not certify seam-free geometry.")
                    st.info("Experimental local RGB registration with 25° south-facing table "
                            "surface checks. Global camera poses and thermal results are unchanged. "
                            "Absolute positions and table edges still need validation.")
                    comparison = refinement_path.parent / "comparison.png"
                    if comparison.is_file():
                        st.image(str(comparison), use_container_width=True)
                    if refined_tables:
                        st.dataframe([
                            {"Table": i, "Layout": f"{table['columns']} × 4",
                             "Surface P95 (m)": table['held_out_p95_abs_m'],
                             "Source coverage (%)": 100*table.get('coverage_fraction', 1),
                             "Used-source P95 (px)": max(
                                 table['local_registration'][str(n)]['p95_px']
                                 for n in table['source_frame_ids']),
                             "Source frames": ", ".join(
                                 refinement['source_frames'][str(n)]['image']
                                 for n in table['source_frame_ids'])}
                            for i, table in enumerate(refined_tables, 1)
                        ], hide_index=True, use_container_width=True)
                    if (refinement_path.parent / 'coverage.png').is_file():
                        with st.expander("Refinement coverage map"):
                            st.image(str(refinement_path.parent / 'coverage.png'),
                                     caption="Green T: re-rendered. Orange R: withheld; original "
                                             "ODM pixels retained. These are candidate table areas.",
                                     use_container_width=True)
                    if refinement.get('rejected'):
                        with st.expander("Withheld areas and reasons"):
                            st.dataframe([
                                {"Bounds (pixels)": str(area['box']), "Reason": area['reason'],
                                 "Qualified coverage (%)": (100*area['coverage_fraction']
                                                            if 'coverage_fraction' in area else None)}
                                for area in refinement['rejected']
                            ], hide_index=True, use_container_width=True)
                    st.download_button("Download table refinement report",
                                       refinement_path.read_bytes(), "table-refinement.json",
                                       mime="application/json")
                    st.code(str(refinement_path.parent / "rgb_constrained.tif"))
            satellite_reports = sorted(odm_job_path.parent.glob("satellite-prior*/report.json"),
                                       key=lambda path: path.stat().st_mtime, reverse=True)
            if satellite_reports:
                with st.expander("Satellite layout — approximate alignment", expanded=True):
                    satellite_path = satellite_reports[0]
                    satellite = json.loads(satellite_path.read_text(encoding="utf-8"))
                    metrics = satellite['diagnostics_grid_m']
                    st.caption(f"{len(satellite['row_bands'])} estimated row bands in the central "
                               f"flight block; {metrics['group_count']} matched pilot rows. "
                               "Row bands are not confirmed tables or module counts.")
                    st.write(f"Median outline difference: {metrics['before_median']:.2f} → "
                             f"{metrics['after_median']:.2f} m. Held-out row P95: "
                             f"{metrics['held_out_row_p95']:.2f} m.")
                    st.info("Approximate satellite reference; image acquisition date and absolute "
                            "accuracy are unknown. Translation only: no scale, yaw, camera-pose "
                            "or temperature correction. Original map and database positions remain unchanged.")
                    if (satellite_path.parent / 'comparison.jpg').is_file():
                        st.image(str(satellite_path.parent / 'comparison.jpg'),
                                 caption="Yellow: satellite outline estimate. Cyan: drone outline, "
                                         "before (left) and after (right).",
                                 use_container_width=True)
                    with st.expander("Estimated layout and source"):
                        if (satellite_path.parent / 'satellite-layout.jpg').is_file():
                            st.image(str(satellite_path.parent / 'satellite-layout.jpg'),
                                     use_container_width=True)
                        st.markdown(f"[Open satellite reference]({satellite['source_url']})")
                        st.caption(satellite['attribution'])
                    st.download_button("Download estimated layout (GeoJSON)",
                                       (satellite_path.parent / 'layout.geojson').read_bytes(),
                                       'estimated-layout.geojson', mime='application/geo+json')
                    st.download_button("Download satellite alignment report",
                                       satellite_path.read_bytes(), 'satellite-alignment.json',
                                       mime='application/json')
                    if satellite.get('derivative'):
                        st.code(str(satellite_path.parent / satellite['derivative']))
            st.info("Thermal orthomosaic: not yet projected onto the ODM surface. "
                    "RGB–thermal alignment must pass validation first.")
            with st.expander("Reconstruction log and provenance"):
                odm_log = odm_job_path.parent / "process.log"
                if odm_log.is_file():
                    with odm_log.open("rb") as log_stream:
                        log_stream.seek(max(0, odm_log.stat().st_size - 12000))
                        st.code(log_stream.read().decode("utf-8", errors="replace"))
                st.download_button("Download job manifest", odm_job_path.read_bytes(),
                                   "pvdelta-odm.json", mime="application/json")
        except (OSError, ValueError, KeyError) as exc:
            st.warning(f"Cannot read reconstruction status: {exc}")
    st.divider()
    st.subheader("RGB + thermal orthoview")
    st.warning("Legacy planar preview: visible geometric seams remain. "
               "This is not an accepted photogrammetric orthophoto.")
    st.write(
        "These are provisional planar orthoviews generated from the recorded LRF target, "
        "camera pose and declared panel surface. The thermal PNG is a display product; "
        "the Celsius grid is retained separately as a NumPy file. RGB pixels are selected "
        "after frame registration by centre/clarity score to avoid double-panel blur."
    )
    orthoview_root = database_path.parent / "orthoviews"
    manifests = sorted(orthoview_root.glob("*/orthoview_manifest.json"))
    if not manifests:
        st.info(
            "No orthoview has been generated yet. Run the local orthoview builder for a "
            "selected flight after checking the surface tilt and azimuth."
        )
    else:
        selected_manifest_path = st.selectbox(
            "Orthoview build",
            manifests,
            format_func=lambda path: path.parent.name,
        )
        manifest = json.loads(selected_manifest_path.read_text(encoding="utf-8"))
        summary_cols = st.columns(4)
        summary_cols[0].metric("Status", manifest.get("status", "unknown"))
        summary_cols[1].metric("Frames", manifest.get("accepted_frames", 0))
        summary_cols[2].metric("Rejected", manifest.get("rejected_frames", 0))
        summary_cols[3].metric(
            "Grid",
            f"{manifest.get('resolution_m_per_px', 0):.2f} m/px",
        )
        build_dir = selected_manifest_path.parent
        rgb_orthoview = build_dir / "rgb_orthoview.png"
        thermal_orthoview = build_dir / "thermal_orthoview.png"
        coverage_orthoview = build_dir / "coverage_count.png"
        image_left, image_right = st.columns(2)
        with image_left:
            st.markdown("**RGB orthoview**")
            if rgb_orthoview.is_file():
                st.image(rgb_orthoview, use_container_width=True)
            else:
                st.warning("RGB orthoview image is missing.")
        with image_right:
            st.markdown("**Radiometric thermal orthoview**")
            if thermal_orthoview.is_file():
                st.image(thermal_orthoview, use_container_width=True)
            else:
                st.warning("Thermal orthoview image is missing.")
        if coverage_orthoview.is_file():
            with st.expander("Coverage count", expanded=False):
                st.image(coverage_orthoview, use_container_width=True)
        st.caption(
            "The current build assumes a 25° south-facing panel plane. Validate it with CAD or "
            "independent control points before using the orthoview for survey-grade measurements."
        )
        downloads = st.columns(3)
        for column, path, label, mime in (
            (downloads[0], rgb_orthoview, "Download RGB orthoview", "image/png"),
            (downloads[1], thermal_orthoview, "Download thermal orthoview", "image/png"),
            (
                downloads[2],
                build_dir / "thermal_temperature_c.npy",
                "Download thermal Celsius grid",
                "application/octet-stream",
            ),
        ):
            if path.is_file():
                column.download_button(label, path.read_bytes(), path.name, mime=mime)

with review_tab:
    st.subheader("User review")
    st.write(
        "Review each radiometric candidate without changing the detector result. "
        "Use **OK** for a confirmed anomaly, **Not OK** for a false positive, and "
        "**Wait** when more evidence is needed."
    )
    if not duckdb_available:
        st.info("Install the dashboard extra to enable finding review.")
    elif not database_path.is_file() or selected_run is None:
        st.info("Open a database with a Stage 1 analysis run to start reviewing findings.")
    else:
        review_connection = connect_store(database_path, read_only=True)
        review_evidence_table = _table_exists(review_connection, "finding_evidence")
        review_evidence_join = (
            "LEFT JOIN finding_evidence AS fe ON fe.finding_id = f.finding_id"
            if review_evidence_table else ""
        )
        review_evidence_select = (
            "fe.panel_support_status, fe.panel_support_ratio, fe.panel_texture_std, "
            "fe.evidence_group_id, "
            "fe.repeat_capture_count"
            if review_evidence_table else
            "NULL AS panel_support_status, NULL AS panel_support_ratio, NULL AS panel_texture_std, "
            "NULL AS evidence_group_id, NULL AS repeat_capture_count"
        )
        review_evidence_clause = (
            "AND fe.panel_support_status = 'panel_supported'"
            if review_evidence_table and finding_scope == "Legacy palette shortlist (experimental)"
            else ""
        )
        review_rows = review_connection.execute(
            f"""
            SELECT f.finding_id,
                   f.analysis_run_id AS run,
                   m.frame_module_id AS module,
                   m.capture_id,
                   c.run_id AS capture_run_id,
                   f.anomaly_type,
                   f.assessment_status AS detector_status,
                   coalesce(fr.review_status, 'wait') AS review_status,
                   CAST(fr.reviewed_at_utc AS VARCHAR) AS reviewed_at_utc,
                   round(f.delta_t_k, 2) AS delta_t_k,
                   round(f.absolute_temperature_c, 2) AS temperature_c,
                   round(coalesce(f.estimated_latitude, m.estimated_latitude), 7) AS latitude,
                   round(coalesce(f.estimated_longitude, m.estimated_longitude), 7) AS longitude,
                   c.source_path AS thermal_path,
                   c.paired_capture_id,
                   CAST(c.capture_time_utc AS VARCHAR) AS capture_time_utc,
                   c.camera_latitude,
                   c.camera_longitude,
                   r.source_root,
                   {review_evidence_select}
            FROM findings AS f
            JOIN module_observations AS m USING (observation_id)
            JOIN captures AS c ON c.capture_id = m.capture_id
            LEFT JOIN inspection_runs AS r ON r.run_id = c.run_id
            LEFT JOIN finding_reviews AS fr USING (finding_id)
            {review_evidence_join}
            WHERE f.analysis_run_id = ? {review_evidence_clause}
            ORDER BY CASE coalesce(fr.review_status, 'wait')
                         WHEN 'wait' THEN 0 WHEN 'not_ok' THEN 1 ELSE 2 END,
                     abs(f.delta_t_k) DESC, f.finding_id
            """,
            [selected_run],
        ).fetchdf()
        review_connection.close()
        if review_rows.empty:
            st.info("No findings match this run and candidate scope.")
        else:
            review_filter = st.selectbox(
                "Show findings",
                ["All", "wait", "ok", "not_ok"],
                format_func=lambda value: {
                    "All": "All",
                    "wait": "Wait / not reviewed",
                    "ok": "OK",
                    "not_ok": "Not OK",
                }[value],
            )
            visible_reviews = review_rows if review_filter == "All" else review_rows[
                review_rows["review_status"] == review_filter
            ]
            review_counts = st.columns(4)
            review_counts[0].metric("Total findings", len(review_rows))
            review_counts[1].metric("Wait", int((review_rows["review_status"] == "wait").sum()))
            review_counts[2].metric("OK", int((review_rows["review_status"] == "ok").sum()))
            review_counts[3].metric("Not OK", int((review_rows["review_status"] == "not_ok").sum()))
            st.dataframe(visible_reviews, hide_index=True, width="stretch")
            if visible_reviews.empty:
                st.info("No findings match this review filter.")
            else:
                review_ids = visible_reviews["finding_id"].tolist()
                selected_review_id = finding_navigation(
                    review_ids, key="review-finding-id", label="Finding to review",
                )
                selected_review_row = visible_reviews[
                    visible_reviews["finding_id"] == selected_review_id
                ].iloc[0]
                st.caption(
                    f"Current review status: **{selected_review_row['review_status']}** · "
                    f"detector status: `{selected_review_row['detector_status']}` · "
                    f"ΔT: {selected_review_row['delta_t_k']:.2f} K"
                )
                st.info(
                    "IEC review gate: an OK decision means this candidate passed the project "
                    "evidence checklist. It is not an IEC conformity assessment or certificate. "
                    "Glare/reflection must be ruled out with comparable radiometric views."
                )
                _render_repeat_view_review(database_path, selected_review_id, key="review")
                review_note = st.text_area(
                    "Review note (optional)",
                    key=f"review-note-{selected_review_id}",
                    placeholder="Why is this candidate confirmed, rejected or deferred?",
                )
                reviewer = st.text_input(
                    "Reviewer",
                    value="",
                    key=f"reviewer-{selected_review_id}",
                )
                checklist_columns = st.columns(2)
                panel_support = checklist_columns[0].checkbox(
                    "Panel support confirmed",
                    key=f"review-panel-support-{selected_review_id}",
                )
                glare_checked = checklist_columns[1].checkbox(
                    "Glare / reflection checked",
                    key=f"review-glare-{selected_review_id}",
                )
                repeat_view = checklist_columns[0].checkbox(
                    "Repeat or alternate view checked",
                    key=f"review-repeat-{selected_review_id}",
                )
                conditions_documented = checklist_columns[1].checkbox(
                    "Radiometric conditions documented",
                    key=f"review-conditions-{selected_review_id}",
                )
                checklist_passed = all(
                    (panel_support, glare_checked, repeat_view, conditions_documented)
                )
                st.caption(
                    "The OK button remains disabled until all evidence checks are recorded. "
                    "This is a project review gate, not an IEC certificate."
                )
                review_buttons = st.columns(3)
                review_actions = (
                    (review_buttons[0], "OK", "ok"),
                    (review_buttons[1], "Not OK", "not_ok"),
                    (review_buttons[2], "Wait", "wait"),
                )
                for button_column, label, status in review_actions:
                    if button_column.button(
                        label,
                        key=f"review-{status}-{selected_review_id}",
                        disabled=status == "ok" and not checklist_passed,
                    ):
                        checklist_note = (
                            "Evidence checklist: "
                            f"panel_support={'PASS' if panel_support else 'PENDING'}; "
                            f"glare_checked={'PASS' if glare_checked else 'PENDING'}; "
                            f"repeat_view={'PASS' if repeat_view else 'PENDING'}; "
                            f"radiometric_conditions={'PASS' if conditions_documented else 'PENDING'}."
                        )
                        combined_note = " ".join(
                            item for item in (review_note.strip(), checklist_note) if item
                        )
                        write_connection = connect_store(database_path)
                        try:
                            upsert_finding_review(
                                write_connection,
                                finding_id=selected_review_id,
                                review_status=status,
                                reviewer=reviewer.strip() or None,
                                note=combined_note or None,
                            )
                        finally:
                            write_connection.close()
                        st.success(f"Saved review: {label}.")
                        st.rerun()

with panel_tab:
    if duckdb_available and database_path.is_file() and selected_run:
        import duckdb

        from pvdelta.panel_review_ui import panel_review_view

        st.caption(f'Using the database and analysis selected in Local data & map: {selected_run}')
        try:
            panel_review_view(database_path, selected_run, sdk_library_text.strip() or None,
                              _load_rgb_registration(), key='advanced-panels-v2')
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, duckdb.Error) as exc:
            st.warning(f'Panel review unavailable: {exc}')
    else:
        st.info('Open an inspection and select an analysis in Local data & map.')

with export_tab:
    geometry_ready = resolution_ok and overlap_speed_ok and angle_ok
    condition_status = (
        all(item["status"] == "PASS" for item in capture_checks)
        if record_conditions
        else None
    )
    payload = {
        "schema": "pvdelta.flight-plan/v1",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project": {"site_label": site_label, "mission_date": mission_date.isoformat()},
        "equipment": {
            "platform_key": platform_key,
            "platform_model": platform.model,
            "aircraft_label": aircraft_label or None,
            "firmware_version": firmware_version or None,
            "thermal_gain_mode": gain_mode,
            "thermal_native_width_px": thermal.sensor_width_px,
            "thermal_native_height_px": thermal.sensor_height_px,
            "thermal_dfov_deg": thermal.dfov_deg,
            "radiometric_rjpeg": radiometric_format,
            "ffc_checked": ffc_done,
            "rtk_enabled": rtk_enabled,
        },
        "module": {
            "preset": module_preset,
            "orientation": module_orientation.lower(),
            "technology": None if module_technology == "Unknown" else module_technology,
            "model_or_datasheet": module_model or None,
            "bifacial": bifacial,
            "length_m": module_length_m,
            "width_m": module_width_m,
            "smallest_cell_feature_mm": smallest_cell_mm,
            "cell_count": module_cell_count,
            "cell_format": module_cell_format or None,
            "cell_layout": module_cell_cut,
            "tilt_deg": panel_tilt_deg,
            "modules_up_slope": rack_rows,
            "row_gap_m": row_gap_m,
            "lower_edge_height_m": lower_edge_height_m,
            "rack_slope_length_m": rack_slope_length_m,
            "rack_top_elevation_m": rack_top_elevation_m,
        },
        "flight": {
            "flight_altitude_agl_m": flight_altitude_agl_m,
            "height_reference": "rack_center" if aligned_to_panel else "lower_edge_ground",
            "height_above_reference_m": height_above_reference_m,
            "camera_geometry": "panel_normal" if aligned_to_panel else "nadir",
            "speed_m_s": speed_m_s,
            "capture_interval_s": capture_interval_s,
            "forward_overlap_pct": forward_overlap,
            "side_overlap_pct": side_overlap,
            "site_length_m": site_length_m,
            "site_width_m": site_width_m,
        },
        "conditions": {
            "recorded": record_conditions,
            "poa_irradiance_w_m2": irradiance_w_m2,
            "irradiance_source": irradiance_source,
            "irradiance_change_pct_min": irradiance_change_pct_min,
            "ambient_temperature_c": ambient_c,
            "reflected_apparent_temperature_c": reflected_c,
            "relative_humidity_pct": humidity_pct,
            "emissivity": emissivity,
            "wind_kmh": wind_kmh,
            "cloud_okta": cloud_okta,
        },
        "derived": flight.to_dict(),
        "readiness": {
            "geometry_ready": geometry_ready,
            "field_conditions_ready": condition_status,
            "checks": [*plan_checks, *capture_checks],
        },
    }
    output_json = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    st.code(output_json, language="json")
    st.download_button(
        "Download flight-plan JSON",
        output_json,
        file_name=f"{site_label or 'pvdelta'}-flight-plan.json",
        mime="application/json",
    )
    st.caption("Uploaded RGB files and their location/serial metadata are not included.")

st.divider()
st.markdown(
    "Sources: [DJI Matrice 4 specifications]"
    "(https://enterprise.dji.com/matrice-4-series/specs), "
    "[DJI solar inspection example]"
    "(https://enterprise-insights.dji.com/blog/dji-dock-solar-inspection-lessons-learned), "
    "[IEA-PVPS O&M Guidelines]"
    "(https://iea-pvps.org/wp-content/uploads/2022/11/IEA-PVPS-Report-T13-25-2022-OandM-Guidelines.pdf)."
)
