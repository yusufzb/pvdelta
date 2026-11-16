"""Rank repeated RGB views while preserving explicit panel-group provenance."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

VIEWS_SCHEMA_VERSION = "pvdelta.views/v1"


@dataclass(frozen=True)
class ViewCandidate:
    panel_key: str
    rgb_capture_id: str
    rgb_path: str
    thermal_capture_id: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    horizontal_accuracy_m: float | None = None

    def __post_init__(self) -> None:
        if not self.panel_key.strip() or not self.rgb_capture_id.strip():
            raise ValueError("panel_key and rgb_capture_id must not be empty")
        if not self.rgb_path.strip():
            raise ValueError("rgb_path must not be empty")
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be provided together")
        if self.latitude is not None and not -90 <= self.latitude <= 90:
            raise ValueError("latitude is outside [-90, 90]")
        if self.longitude is not None and not -180 <= self.longitude <= 180:
            raise ValueError("longitude is outside [-180, 180]")


@dataclass(frozen=True)
class RankedView:
    panel_key: str
    rgb_capture_id: str
    rgb_path: str
    thermal_capture_id: str | None
    latitude: float | None
    longitude: float | None
    horizontal_accuracy_m: float | None
    ranking_score: float
    sharpness: float
    clipped_fraction: float
    contrast: float
    rank: int
    selection_status: str


def _luminance(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array.astype(np.float32)
    elif array.ndim == 3 and array.shape[2] in {1, 3, 4}:
        channels = array[..., :3].astype(np.float32)
        gray = channels[..., 0] * 0.2126 + channels[..., 1] * 0.7152 + channels[..., 2] * 0.0722
    else:
        raise ValueError("RGB image must have shape (height, width, 3/4)")
    if min(gray.shape) < 3 or not np.isfinite(gray).all():
        raise ValueError("RGB image must be finite and at least 3 pixels wide/high")
    if gray.max() <= 1.0:
        gray *= 255.0
    return gray


def score_rgb_array(image: np.ndarray) -> dict[str, float]:
    """Return scene-relative focus/exposure metrics for one RGB crop or frame."""
    gray = _luminance(image)
    laplacian = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    sharpness = float(np.var(laplacian))
    clipped = float(np.mean((gray <= 2.0) | (gray >= 253.0)))
    contrast = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    score = math.log1p(sharpness) + 0.02 * contrast - 4.0 * clipped
    return {
        "ranking_score": score,
        "sharpness": sharpness,
        "clipped_fraction": clipped,
        "contrast": contrast,
    }


def rank_view_candidates(
    candidates: list[ViewCandidate] | tuple[ViewCandidate, ...],
    loader: Callable[[str], np.ndarray],
) -> tuple[RankedView, ...]:
    """Rank within each explicit panel group; never infer group identity here."""
    grouped: dict[str, list[tuple[ViewCandidate, dict[str, float]]]] = {}
    for candidate in candidates:
        metrics = score_rgb_array(loader(candidate.rgb_path))
        grouped.setdefault(candidate.panel_key, []).append((candidate, metrics))
    result: list[RankedView] = []
    for panel_key in sorted(grouped):
        ordered = sorted(
            grouped[panel_key],
            key=lambda item: (-item[1]["ranking_score"], item[0].rgb_capture_id),
        )
        for rank, (candidate, metrics) in enumerate(ordered, 1):
            result.append(RankedView(
                panel_key=candidate.panel_key,
                rgb_capture_id=candidate.rgb_capture_id,
                rgb_path=candidate.rgb_path,
                thermal_capture_id=candidate.thermal_capture_id,
                latitude=candidate.latitude,
                longitude=candidate.longitude,
                horizontal_accuracy_m=candidate.horizontal_accuracy_m,
                ranking_score=metrics["ranking_score"],
                sharpness=metrics["sharpness"],
                clipped_fraction=metrics["clipped_fraction"],
                contrast=metrics["contrast"],
                rank=rank,
                selection_status="selected" if rank == 1 else "candidate",
            ))
    return tuple(result)


def ranked_view_to_dict(view: RankedView) -> dict:
    return asdict(view)


def load_rgb_file(path: str | Path) -> np.ndarray:
    """Load an RGB file without copying it into project storage."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("RGB ranking requires Pillow: pip install -e '.[dashboard]'") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))
