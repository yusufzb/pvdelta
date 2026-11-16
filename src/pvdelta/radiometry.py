"""Private-flight native radiometry scan helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pvdelta.decode import RJpeg


@dataclass(frozen=True)
class RadiometricFrameResult:
    source_path: str
    source_hash: str
    width_px: int
    height_px: int
    valid_pixels: int
    temperature_min_c: float
    temperature_max_c: float
    temperature_mean_c: float
    temperature_median_c: float
    emissivity: float
    distance_m: float
    ambient_temp_c: float
    reflected_temp_c: float


def scan_thermal_files(
    paths: Iterable[str | Path],
    *,
    sdk_lib: str | Path | None = None,
    progress: Callable[[int, int, Path], None] | None = None,
) -> tuple[RadiometricFrameResult, ...]:
    """Decode local thermal R-JPEGs on the native DJI grid and return compact summaries."""
    files = tuple(Path(path) for path in paths)
    results: list[RadiometricFrameResult] = []
    for index, path in enumerate(files, start=1):
        with RJpeg(path, sdk_lib) as image:
            temperature = image.temperature()
            params = image.measurement_params
        valid = temperature[np.isfinite(temperature)]
        if valid.size == 0:
            raise ValueError(f"thermal frame contains no finite pixels: {path}")
        results.append(RadiometricFrameResult(
            source_path=str(path),
            source_hash=_sha256(path),
            width_px=int(temperature.shape[1]),
            height_px=int(temperature.shape[0]),
            valid_pixels=int(valid.size),
            temperature_min_c=float(np.min(valid)),
            temperature_max_c=float(np.max(valid)),
            temperature_mean_c=float(np.mean(valid)),
            temperature_median_c=float(np.median(valid)),
            emissivity=float(params.emissivity),
            distance_m=float(params.distance_m),
            ambient_temp_c=float(params.ambient_temp_c),
            reflected_temp_c=float(params.reflected_temp_c),
        ))
        if progress is not None:
            progress(index, len(files), path)
    return tuple(results)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
