"""Build provisional RGB and radiometric thermal orthoviews from DJI captures.

The orthoview is deliberately separate from detection.  Each frame is ray-cast onto a
declared planar surface, then resampled into a local ENU grid.  The result is useful for
visual review, but remains provisional until independent survey/CAD control validates the
surface and camera boresight.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pvdelta.decode import RJpeg
from pvdelta.geolocation import FootprintRaycastError, LrfPlanarProjector


@dataclass(frozen=True)
class OrthoCapture:
    """One paired thermal/RGB capture with pose metadata from the local store."""

    thermal_path: Path
    rgb_path: Path | None
    target_latitude: float
    target_longitude: float
    laser_distance_m: float
    gimbal_pitch_deg: float
    gimbal_yaw_deg: float
    gimbal_roll_deg: float
    thermal_width_px: int
    thermal_height_px: int
    rgb_width_px: int | None
    rgb_height_px: int | None
    thermal_fov_deg: float
    rgb_fov_deg: float | None
    frame_id: str


@dataclass(frozen=True)
class OrthoBuildResult:
    """Paths and diagnostics for one RGB/thermal orthoview build."""

    thermal_png: Path
    thermal_temperature_npy: Path
    rgb_png: Path
    coverage_png: Path
    manifest_json: Path
    frame_count: int
    accepted_frames: int
    rejected_frames: int
    bounds_m: tuple[float, float, float, float]
    resolution_m_per_px: float


def _local_enu(
    latitude: float,
    longitude: float,
    origin_latitude: float,
    origin_longitude: float,
) -> tuple[float, float]:
    """Convert a small WGS84 difference to local east/north metres."""
    earth_radius_m = 6_378_137.0
    east_scale = earth_radius_m * math.cos(math.radians(origin_latitude))
    east = math.radians(longitude - origin_longitude) * east_scale
    north = math.radians(latitude - origin_latitude) * earth_radius_m
    return east, north


def _project_corners(
    capture: OrthoCapture,
    *,
    image_width_px: int,
    image_height_px: int,
    diagonal_fov_deg: float,
    origin_latitude: float,
    origin_longitude: float,
    surface_tilt_deg: float,
    surface_azimuth_deg: float,
) -> np.ndarray:
    projector = LrfPlanarProjector(
        target_latitude=capture.target_latitude,
        target_longitude=capture.target_longitude,
        laser_distance_m=capture.laser_distance_m,
        gimbal_pitch_deg=capture.gimbal_pitch_deg,
        gimbal_yaw_deg=capture.gimbal_yaw_deg,
        gimbal_roll_deg=capture.gimbal_roll_deg,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        diagonal_fov_deg=diagonal_fov_deg,
        surface_tilt_deg=surface_tilt_deg,
        surface_azimuth_deg=surface_azimuth_deg,
    )
    points = []
    for corner in projector.footprint().corners:
        east, north = _local_enu(
            corner.latitude,
            corner.longitude,
            origin_latitude,
            origin_longitude,
        )
        points.append((east, north))
    return np.asarray(points, dtype=np.float32)


def _read_rgb(path: Path, max_dimension: int = 2400) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    scale = min(1.0, max_dimension / max(rgb.shape[:2]))
    if scale < 1:
        resized = Image.fromarray(rgb).resize(
            (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)),
            Image.Resampling.LANCZOS,
        )
        rgb = np.asarray(resized)
    return rgb


def _frame_sharpness(image: np.ndarray) -> float:
    """Return a display-only focus score used to avoid averaging blurry overlaps."""
    try:
        import cv2
    except ImportError:  # pragma: no cover - optional dashboard dependency
        return 1.0
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.float32)
    small = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    return max(1e-6, float(cv2.Laplacian(small, cv2.CV_32F).var()))


def _centre_weight(image_shape: tuple[int, ...]) -> np.ndarray:
    height, width = image_shape[:2]
    y, x = np.ogrid[:height, :width]
    distance = np.sqrt(
        ((x - (width - 1) / 2) / max(1.0, width / 2)) ** 2
        + ((y - (height - 1) / 2) / max(1.0, height / 2)) ** 2
    )
    return np.clip(1.0 - 0.45 * distance, 0.15, 1.0).astype(np.float32)


def _estimate_translation(
    reference_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    overlap: np.ndarray,
    *,
    max_shift_px: float = 40.0,
) -> tuple[float, float, float]:
    """Estimate a small translation from the current mosaic to a new RGB frame."""
    try:
        import cv2
    except ImportError:  # pragma: no cover - optional dashboard dependency
        return 0.0, 0.0, 0.0
    ys, xs = np.where(overlap)
    if len(xs) < 2_000:
        return 0.0, 0.0, 0.0
    margin = 12
    left, right = max(0, int(xs.min()) - margin), min(reference_rgb.shape[1], int(xs.max()) + margin + 1)
    top, bottom = max(0, int(ys.min()) - margin), min(reference_rgb.shape[0], int(ys.max()) + margin + 1)
    if right - left < 32 or bottom - top < 32:
        return 0.0, 0.0, 0.0
    reference = cv2.cvtColor(reference_rgb[top:bottom, left:right], cv2.COLOR_RGB2GRAY).astype(np.float32)
    candidate = cv2.cvtColor(candidate_rgb[top:bottom, left:right], cv2.COLOR_RGB2GRAY).astype(np.float32)
    local_mask = overlap[top:bottom, left:right]
    reference[~local_mask] = 0
    candidate[~local_mask] = 0
    window = cv2.createHanningWindow((reference.shape[1], reference.shape[0]), cv2.CV_32F)
    (shift_x, shift_y), response = cv2.phaseCorrelate(reference, candidate, window)
    magnitude = math.hypot(shift_x, shift_y)
    if not math.isfinite(response) or response < 0.08 or magnitude > max_shift_px:
        return 0.0, 0.0, float(response)
    return float(shift_x), float(shift_y), float(response)


def _thermal_display(temperature_c: np.ndarray, low: float, high: float) -> np.ndarray:
    valid = np.isfinite(temperature_c)
    scaled = np.zeros(temperature_c.shape, dtype=np.uint8)
    if high > low and valid.any():
        values = np.clip((temperature_c - low) / (high - low), 0, 1)
        scaled[valid] = np.round(values[valid] * 255).astype(np.uint8)
    # A compact blue-to-yellow-red palette without requiring matplotlib.
    red = scaled
    green = np.minimum(255, scaled.astype(np.uint16) * 2).astype(np.uint8)
    blue = (255 - scaled).astype(np.uint8)
    display = np.stack((red, green, blue), axis=-1)
    display[~valid] = 0
    return display


def _homography(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dashboard dependency
        raise RuntimeError("Orthoview generation requires opencv-python-headless") from exc
    matrix, status = cv2.findHomography(source, target, method=0)
    if matrix is None or status is None:
        raise ValueError("could not compute image-to-orthoview homography")
    return matrix.astype(np.float64)


def _warp(
    image: np.ndarray,
    matrix: np.ndarray,
    canvas_shape: tuple[int, int],
    *,
    interpolation: int,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    height, width = canvas_shape
    mask = np.full(image.shape[:2], 255, dtype=np.uint8)
    warped = cv2.warpPerspective(image, matrix, (width, height), flags=interpolation)
    warped_mask = cv2.warpPerspective(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
    return warped, warped_mask > 0


def build_orthoview(
    captures: list[OrthoCapture],
    output_dir: str | Path,
    *,
    surface_tilt_deg: float = 25.0,
    surface_azimuth_deg: float = 180.0,
    resolution_m_per_px: float = 0.05,
    sdk_library: str | Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> OrthoBuildResult:
    """Build paired provisional orthoviews from validated pose records.

    Thermal output includes a float32 Celsius ``.npy`` grid.  PNGs are display products;
    the temperature grid remains the measurement product and contains NaN outside coverage.
    """
    if not captures:
        raise ValueError("at least one capture is required")
    if not math.isfinite(resolution_m_per_px) or resolution_m_per_px <= 0:
        raise ValueError("resolution_m_per_px must be positive and finite")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    origin_latitude = captures[0].target_latitude
    origin_longitude = captures[0].target_longitude
    corners: list[np.ndarray] = []
    accepted: list[tuple[OrthoCapture, np.ndarray, np.ndarray]] = []
    rejected = 0
    for capture in captures:
        try:
            thermal_polygon = _project_corners(
                capture,
                image_width_px=capture.thermal_width_px,
                image_height_px=capture.thermal_height_px,
                diagonal_fov_deg=capture.thermal_fov_deg,
                origin_latitude=origin_latitude,
                origin_longitude=origin_longitude,
                surface_tilt_deg=surface_tilt_deg,
                surface_azimuth_deg=surface_azimuth_deg,
            )
            rgb_width = capture.rgb_width_px or capture.thermal_width_px
            rgb_height = capture.rgb_height_px or capture.thermal_height_px
            rgb_polygon = _project_corners(
                capture,
                image_width_px=rgb_width,
                image_height_px=rgb_height,
                diagonal_fov_deg=capture.rgb_fov_deg or capture.thermal_fov_deg,
                origin_latitude=origin_latitude,
                origin_longitude=origin_longitude,
                surface_tilt_deg=surface_tilt_deg,
                surface_azimuth_deg=surface_azimuth_deg,
            )
            if not np.isfinite(thermal_polygon).all() or not np.isfinite(rgb_polygon).all():
                raise FootprintRaycastError("non-finite projected footprint")
            accepted.append((capture, thermal_polygon, rgb_polygon))
            corners.extend((thermal_polygon, rgb_polygon))
        except (FootprintRaycastError, ValueError, TypeError):
            rejected += 1
    if not accepted:
        raise ValueError("no capture could be projected onto the selected plane")
    all_corners = np.concatenate(corners, axis=0)
    min_east = float(np.floor(all_corners[:, 0].min() / resolution_m_per_px) * resolution_m_per_px)
    max_east = float(np.ceil(all_corners[:, 0].max() / resolution_m_per_px) * resolution_m_per_px)
    min_north = float(np.floor(all_corners[:, 1].min() / resolution_m_per_px) * resolution_m_per_px)
    max_north = float(np.ceil(all_corners[:, 1].max() / resolution_m_per_px) * resolution_m_per_px)
    width = round((max_east - min_east) / resolution_m_per_px) + 1
    height = round((max_north - min_north) / resolution_m_per_px) + 1
    if width * height > 40_000_000:
        raise ValueError("orthoview canvas exceeds 40 million pixels; use a coarser resolution")

    rgb_grid = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_best_score = np.full((height, width), -np.inf, dtype=np.float32)
    thermal_grid = np.full((height, width), np.nan, dtype=np.float32)
    thermal_best_score = np.full((height, width), -np.inf, dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.uint16)
    thermal_low, thermal_high = math.inf, -math.inf
    source_entries: list[dict[str, Any]] = []
    source_points = np.asarray(
        [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32
    )
    for index, (capture, thermal_polygon, rgb_polygon) in enumerate(accepted, start=1):
        thermal_target_points = np.column_stack(
            (
                (thermal_polygon[:, 0] - min_east) / resolution_m_per_px,
                (max_north - thermal_polygon[:, 1]) / resolution_m_per_px,
            )
        ).astype(np.float32)
        with RJpeg(capture.thermal_path, sdk_library) as rjpeg:
            temperature = rjpeg.temperature()
            thermal_height, thermal_width = temperature.shape
            thermal_low = min(thermal_low, float(np.nanmin(temperature)))
            thermal_high = max(thermal_high, float(np.nanmax(temperature)))
        rgb = None
        rgb_target_points = None
        rgb_source = None
        rgb_sharpness = 1.0
        shift_x = shift_y = 0.0
        registration_response = 0.0
        rgb_alignment_accepted = True
        if capture.rgb_path is not None and capture.rgb_path.is_file():
            rgb = _read_rgb(capture.rgb_path)
            rgb_target_points = np.column_stack(
                (
                    (rgb_polygon[:, 0] - min_east) / resolution_m_per_px,
                    (max_north - rgb_polygon[:, 1]) / resolution_m_per_px,
                )
            ).astype(np.float32)
            rgb_source = source_points * np.asarray(
                [rgb.shape[1] - 1, rgb.shape[0] - 1], dtype=np.float32
            )
            rgb_matrix = _homography(rgb_source, rgb_target_points)
            initial_rgb, initial_rgb_mask = _warp(
                rgb,
                rgb_matrix,
                (height, width),
                interpolation=1,
            )
            shift_x, shift_y, registration_response = _estimate_translation(
                rgb_grid,
                initial_rgb,
                initial_rgb_mask & np.isfinite(rgb_best_score),
            )
            has_existing_rgb = bool(np.isfinite(rgb_best_score).any())
            if has_existing_rgb and registration_response < 0.08:
                # Do not create double panel edges by blending an unregistered frame.
                # Thermal data can still contribute on its native radiometric grid.
                rgb_alignment_accepted = False
            rgb_sharpness = _frame_sharpness(rgb)

        thermal_target_points = thermal_target_points + np.asarray([shift_x, shift_y], dtype=np.float32)
        thermal_source = source_points * np.asarray(
            [thermal_width - 1, thermal_height - 1], dtype=np.float32
        )
        thermal_matrix = _homography(thermal_source, thermal_target_points)
        warped_temperature, thermal_mask = _warp(
            temperature,
            thermal_matrix,
            (height, width),
            interpolation=0,
        )
        valid = thermal_mask & np.isfinite(warped_temperature)
        thermal_score, _ = _warp(
            _centre_weight(temperature.shape),
            thermal_matrix,
            (height, width),
            interpolation=1,
        )
        thermal_score = thermal_score.astype(np.float32) * _frame_sharpness(temperature)
        take_thermal = valid & (thermal_score > thermal_best_score)
        thermal_grid[take_thermal] = warped_temperature[take_thermal]
        thermal_best_score[take_thermal] = thermal_score[take_thermal]
        coverage[thermal_mask] = np.minimum(coverage[thermal_mask] + 1, np.iinfo(np.uint16).max)
        if (rgb_alignment_accepted and rgb is not None
                and rgb_target_points is not None and rgb_source is not None):
            rgb_target_points = rgb_target_points + np.asarray([shift_x, shift_y], dtype=np.float32)
            rgb_matrix = _homography(rgb_source, rgb_target_points)
            rgb_warped, rgb_mask = _warp(rgb, rgb_matrix, (height, width), interpolation=1)
            rgb_weight, _ = _warp(_centre_weight(rgb.shape), rgb_matrix, (height, width), interpolation=1)
            rgb_score = rgb_weight.astype(np.float32) * rgb_sharpness
            # Select one source pixel per orthoview pixel. Averaging frames with
            # sub-pixel registration error is the main source of visible blur.
            take_rgb_max = rgb_mask & (rgb_score > rgb_best_score)
            rgb_grid[take_rgb_max] = rgb_warped[take_rgb_max]
            rgb_best_score[take_rgb_max] = rgb_score[take_rgb_max]

        source_entries.append({
            "frame_id": capture.frame_id,
            "thermal": str(capture.thermal_path),
            "registration_shift_px": [shift_x, shift_y],
            "rgb_registration_response": registration_response,
            "rgb_alignment": "accepted" if rgb_alignment_accepted else "rejected_low_confidence",
        })
        if progress is not None:
            progress(index, len(accepted))

    display_low, display_high = np.nanpercentile(thermal_grid, [2, 98])
    if not math.isfinite(float(display_low)) or not math.isfinite(float(display_high)):
        display_low, display_high = thermal_low, thermal_high
    thermal_png = output / "thermal_orthoview.png"
    thermal_temperature_npy = output / "thermal_temperature_c.npy"
    rgb_png = output / "rgb_orthoview.png"
    coverage_png = output / "coverage_count.png"
    manifest_json = output / "orthoview_manifest.json"
    Image.fromarray(_thermal_display(thermal_grid, display_low, display_high)).save(thermal_png)
    np.save(thermal_temperature_npy, thermal_grid)
    # Keep the highest quality registered RGB sample at each pixel. Averaging
    # misregistered frames creates double panel lines and visible blur. Apply
    # only a display-side unsharp mask; source pixels and radiometric data are
    # untouched.
    try:
        from PIL import ImageFilter

        rgb_display = Image.fromarray(rgb_grid).filter(
            ImageFilter.UnsharpMask(radius=0.8, percent=70, threshold=2)
        )
    except (ImportError, ValueError):  # pragma: no cover - Pillow is a dashboard dependency
        rgb_display = Image.fromarray(rgb_grid)
    rgb_display.save(rgb_png)
    Image.fromarray(np.clip(coverage / max(1, int(coverage.max())) * 255, 0, 255).astype(np.uint8)).save(
        coverage_png
    )
    manifest = {
        "schema": "pvdelta.orthoview/v1",
        "status": "provisional",
        "surface": {"tilt_deg": surface_tilt_deg, "azimuth_deg": surface_azimuth_deg},
        "origin": {"latitude": origin_latitude, "longitude": origin_longitude},
        "bounds_m": [min_east, min_north, max_east, max_north],
        "resolution_m_per_px": resolution_m_per_px,
        "frame_count": len(captures),
        "accepted_frames": len(accepted),
        "rejected_frames": rejected,
        "temperature_range_c": [thermal_low, thermal_high],
        "display_range_c": [float(display_low), float(display_high)],
        "sources": source_entries,
    }
    shifts = np.asarray(
        [entry["registration_shift_px"] for entry in source_entries], dtype=np.float32
    )
    if len(shifts):
        magnitudes = np.linalg.norm(shifts, axis=1)
        manifest["registration_shift_stats_px"] = {
            "mean_abs": float(np.mean(magnitudes)),
            "p95_abs": float(np.percentile(magnitudes, 95)),
            "max_abs": float(np.max(magnitudes)),
        }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return OrthoBuildResult(
        thermal_png,
        thermal_temperature_npy,
        rgb_png,
        coverage_png,
        manifest_json,
        len(captures),
        len(accepted),
        rejected,
        (min_east, min_north, max_east, max_north),
        resolution_m_per_px,
    )
