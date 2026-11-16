"""Rasterize module polygons on the native thermal pixel grid using NumPy only."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pvdelta.models import CoordinateSpace


@dataclass(frozen=True)
class ModuleAnnotation:
    module_id: str
    polygon: tuple[tuple[float, float], ...]
    coordinate_space: CoordinateSpace = CoordinateSpace.THERMAL_NATIVE
    visibility: str = "full"
    peer_group: str | None = None
    mask_label: int | None = None

    def __post_init__(self) -> None:
        if not self.module_id.strip():
            raise ValueError("module_id must not be empty")
        if self.visibility not in {"full", "partial"}:
            raise ValueError("visibility must be full or partial")
        if self.mask_label is not None and (
            isinstance(self.mask_label, bool)
            or not isinstance(self.mask_label, (int, np.integer))
            or self.mask_label < 1
        ):
            raise ValueError("mask_label must be a positive integer")
        points = np.asarray(self.polygon, dtype=float)
        if points.shape != (len(self.polygon), 2) or len(points) < 3:
            raise ValueError("polygon needs at least three (x, y) points")
        if not np.isfinite(points).all():
            raise ValueError("polygon points must be finite")
        xs, ys = points[:, 0], points[:, 1]
        if abs(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys)) < 1e-9:
            raise ValueError("polygon has zero area")


def load_annotations(path: str | Path) -> list[ModuleAnnotation]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported annotation schema_version")
    if payload.get("coordinate_space") != CoordinateSpace.THERMAL_NATIVE.value:
        raise ValueError("annotations must use the native thermal coordinate space")
    result = [ModuleAnnotation(
        module_id=item["module_id"],
        polygon=tuple(tuple(float(v) for v in point) for point in item["polygon"]),
        visibility=item.get("visibility", "full"),
        peer_group=item.get("peer_group"),
        mask_label=item.get("mask_label"),
    ) for item in payload.get("modules", [])]
    if len({item.module_id for item in result}) != len(result):
        raise ValueError("duplicate module_id in annotations")
    return result


def load_mask_archive(
    path: str | Path,
    annotations: list[ModuleAnnotation],
    shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Load reviewed native measurement masks mapped by explicit positive labels."""
    with np.load(path) as archive:
        if not {"module_labels", "measurement"}.issubset(archive.files):
            raise ValueError("mask archive needs module_labels and measurement arrays")
        labels = archive["module_labels"].copy()
        measurement = archive["measurement"].copy()
    if labels.shape != shape or measurement.shape != labels.shape:
        raise ValueError("mask archive arrays must match the native temperature grid")
    result = {}
    for annotation in annotations:
        if annotation.mask_label is None:
            raise ValueError("each annotation needs mask_label with a mask archive")
        mask = (labels == annotation.mask_label) & measurement.astype(bool)
        if not mask.any():
            raise ValueError(f"mask_label {annotation.mask_label} is empty for {annotation.module_id}")
        result[annotation.module_id] = mask
    return result


def polygon_to_mask(
    pixel_polygon: list[tuple[float, float]],
    shape: tuple[int, int],
) -> np.ndarray:
    """Mask pixel centres inside a polygon; coordinates are (x, y), shape (h, w)."""
    if len(shape) != 2 or any(not isinstance(v, int) or v <= 0 for v in shape):
        raise ValueError("shape must be positive (height, width)")
    if len(pixel_polygon) < 3:
        raise ValueError("Polygon must have at least 3 vertices")
    points = np.asarray(pixel_polygon, dtype=float)
    if points.shape != (len(pixel_polygon), 2) or not np.isfinite(points).all():
        raise ValueError("Polygon vertices must be finite (x, y) pairs")
    xs, ys = points[:, 0], points[:, 1]
    area2 = np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys)
    if abs(area2) < 1e-9:
        raise ValueError("Polygon has zero area")
    height, width = shape
    x0, x1 = max(0, math.floor(xs.min())), min(width, math.ceil(xs.max()))
    y0, y1 = max(0, math.floor(ys.min())), min(height, math.ceil(ys.max()))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Polygon is outside the frame")
    x, y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    inside = np.zeros(x.shape, dtype=bool)
    for index in range(len(points)):
        ax, ay = points[index]
        bx, by = points[(index + 1) % len(points)]
        if ay == by:
            continue
        crosses = (ay > y) != (by > y)
        x_cross = ax + (y - ay) * (bx - ax) / (by - ay)
        inside ^= crosses & (x < x_cross)
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = inside
    if not mask.any():
        raise ValueError("Polygon contains no thermal pixel centres")
    return mask


def extract_module_mask(
    frame_width: int,
    frame_height: int,
    pixel_polygon: list[tuple[float, float]],
) -> np.ndarray:
    return polygon_to_mask(pixel_polygon, (frame_height, frame_width))
