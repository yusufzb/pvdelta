"""R-JPEG → float32 °C via the DJI Thermal SDK (libdirp), using pvdelta's own ctypes binding.

The SDK is *not* bundled. Download it from DJI and point ``PVDELTA_DIRP_LIB`` at the folder that
contains ``libdirp.so`` (Linux) or ``libdirp.dll`` (Windows).

Only the small, stable subset of ``dirp_api.h`` is bound here:

    dirp_create_from_rjpeg, dirp_destroy, dirp_get_rjpeg_resolution,
    dirp_get_measurement_params, dirp_set_measurement_params,
    dirp_get_measurement_params_range, dirp_process, dirp_measure_ex

The local DJI Thermal SDK v1.8 header specifies five float measurement fields.
Verify the binding again when using a different SDK version.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np

DIRP_SUCCESS = 0
ENV_VAR = "PVDELTA_DIRP_LIB"

_lib: ctypes.CDLL | None = None
_lib_path: Path | None = None
_dll_dir: object | None = None
_RJpegT = TypeVar("_RJpegT", bound="RJpeg")


class DirpError(RuntimeError):
    """Raised when the DJI Thermal SDK returns a non-zero status or cannot be loaded."""


class _Resolution(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int32), ("height", ctypes.c_int32)]


class _MeasureParams(ctypes.Structure):
    _fields_ = [
        ("distance", ctypes.c_float),
        ("humidity", ctypes.c_float),
        ("emissivity", ctypes.c_float),
        ("reflection", ctypes.c_float),
        ("ambient_temp", ctypes.c_float),
    ]


class _ParameterRange(ctypes.Structure):
    _fields_ = [("min", ctypes.c_float), ("max", ctypes.c_float)]


class _MeasurementRanges(ctypes.Structure):
    _fields_ = [(name, _ParameterRange) for name in
                ("distance", "humidity", "emissivity", "reflection", "ambient_temp")]


@dataclass(frozen=True)
class MeasurementParams:
    distance_m: float
    humidity_pct: float
    emissivity: float
    reflected_temp_c: float
    ambient_temp_c: float


def _lib_names() -> list[str]:
    if sys.platform.startswith("win"):
        return ["libdirp.dll"]
    if sys.platform.startswith("linux"):
        return ["libdirp.so"]
    raise DirpError("DJI ships the Thermal SDK for Windows and Linux only.")


def load_library(path: str | os.PathLike | None = None) -> ctypes.CDLL:
    """Load libdirp once. ``path`` may be the library file or its folder."""
    global _lib, _lib_path, _dll_dir
    configured = path if path is not None else os.environ.get(ENV_VAR)
    if configured is None or not str(configured).strip():
        raise DirpError(f"Set {ENV_VAR} to the DJI Thermal SDK library folder.")
    root = Path(configured)
    libfile = root if root.is_file() else next(
        (root / n for n in _lib_names() if (root / n).exists()), None
    )
    if libfile is None:
        raise DirpError(f"libdirp not found in {root}")
    libfile = libfile.resolve()
    if _lib is not None:
        if libfile != _lib_path:
            raise DirpError("A different DJI Thermal SDK library is already loaded in this process")
        return _lib

    folder = libfile.parent
    if sys.platform.startswith("win"):
        try:
            _dll_dir = os.add_dll_directory(str(folder))  # type: ignore[attr-defined]
            lib = ctypes.CDLL(str(libfile))
        except OSError as exc:
            raise DirpError(f"cannot load DJI SDK library or one of its dependencies: {exc}") from exc
    else:
        # libdirp depends on sibling libv_*.so; preload them globally so the loader finds them
        # even without LD_LIBRARY_PATH.
        for dep in sorted(glob.glob(str(folder / "libv_*.so"))):
            ctypes.CDLL(dep, mode=ctypes.RTLD_GLOBAL)
        lib = ctypes.CDLL(str(libfile), mode=ctypes.RTLD_GLOBAL)

    h = ctypes.c_void_p
    lib.dirp_create_from_rjpeg.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(h)]
    lib.dirp_destroy.argtypes = [h]
    lib.dirp_get_rjpeg_resolution.argtypes = [h, ctypes.POINTER(_Resolution)]
    lib.dirp_get_measurement_params.argtypes = [h, ctypes.POINTER(_MeasureParams)]
    lib.dirp_set_measurement_params.argtypes = [h, ctypes.POINTER(_MeasureParams)]
    lib.dirp_get_measurement_params_range.argtypes = [h, ctypes.POINTER(_MeasurementRanges)]
    lib.dirp_process.argtypes = [h, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int32]
    lib.dirp_measure_ex.argtypes = [h, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
    for fn in ("dirp_create_from_rjpeg", "dirp_destroy", "dirp_get_rjpeg_resolution",
               "dirp_get_measurement_params", "dirp_set_measurement_params",
               "dirp_get_measurement_params_range", "dirp_process", "dirp_measure_ex"):
        getattr(lib, fn).restype = ctypes.c_int32

    _lib = lib
    _lib_path = libfile
    return lib


def _check(code: int, what: str) -> None:
    if code != DIRP_SUCCESS:
        raise DirpError(f"{what} failed with DIRP code {code}")


class RJpeg:
    """Context manager around one R-JPEG handle."""

    def __init__(self, path: str | os.PathLike, lib_path: str | os.PathLike | None = None):
        self._lib = load_library(lib_path)
        self._data = Path(path).read_bytes()
        if not self._data:
            raise DirpError(f"R-JPEG is empty: {path}")
        if len(self._data) > 2**31 - 1:
            raise DirpError("R-JPEG is too large for the DJI SDK int32 input length")
        self._h = ctypes.c_void_p()
        _check(
            self._lib.dirp_create_from_rjpeg(self._data, len(self._data), ctypes.byref(self._h)),
            f"dirp_create_from_rjpeg({path})",
        )

    # Keep Python 3.10 support without adding typing_extensions just for Self.
    def __enter__(self: _RJpegT) -> _RJpegT:  # noqa: PYI019
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._h:
            self._lib.dirp_destroy(self._h)
            self._h = ctypes.c_void_p()

    def _require_open(self) -> None:
        if not self._h:
            raise DirpError("R-JPEG handle is closed")

    @property
    def resolution(self) -> tuple[int, int]:
        self._require_open()
        r = _Resolution()
        _check(self._lib.dirp_get_rjpeg_resolution(self._h, ctypes.byref(r)), "resolution")
        if r.width <= 0 or r.height <= 0 or r.width * r.height > 100_000_000:
            raise DirpError(f"invalid thermal resolution: {r.width}x{r.height}")
        return r.height, r.width

    @property
    def measurement_params(self) -> MeasurementParams:
        self._require_open()
        b = _MeasureParams()
        _check(self._lib.dirp_get_measurement_params(self._h, ctypes.byref(b)), "params")
        return MeasurementParams(
            float(b.distance), float(b.humidity), float(b.emissivity),
            float(b.reflection), float(b.ambient_temp),
        )

    @property
    def measurement_param_ranges(self) -> dict[str, tuple[float, float]]:
        """Read this R-JPEG's supported limits from DJI, without guessing camera limits."""
        self._require_open()
        result = _MeasurementRanges()
        _check(self._lib.dirp_get_measurement_params_range(self._h, ctypes.byref(result)),
               "measurement parameter ranges")
        return {name: (float(getattr(result, field).min), float(getattr(result, field).max))
                for name, field in zip(MeasurementParams.__dataclass_fields__,
                    ("distance", "humidity", "emissivity", "reflection", "ambient_temp"), strict=True)}

    def set_measurement_params(self, parameters: MeasurementParams) -> MeasurementParams:
        """Explicitly remeasure through DJI; change only this handle, never the source JPEG.

        Callers must retain the recorded and effective parameters separately. Native
        array arithmetic is not a substitute for the SDK's atmospheric correction.
        """
        self._require_open()
        if not isinstance(parameters, MeasurementParams):
            raise TypeError("MeasurementParams required")
        limits = self.measurement_param_ranges
        values = []
        for name in MeasurementParams.__dataclass_fields__:
            value = getattr(parameters, name)
            low, high = limits[name]
            if (type(value) not in (int, float) or not np.isfinite(value)
                    or not np.isfinite([low, high]).all() or not low <= value <= high):
                raise ValueError(f"{name} must be within this frame's SDK range [{low}, {high}]")
            values.append(value)
        packed = _MeasureParams(*values)
        _check(self._lib.dirp_set_measurement_params(self._h, ctypes.byref(packed)),
               "dirp_set_measurement_params")
        effective = self.measurement_params
        if not np.allclose([getattr(effective, n) for n in MeasurementParams.__dataclass_fields__],
                           values, rtol=1e-6, atol=1e-5):
            raise DirpError("SDK effective parameters differ from the explicitly requested values")
        return effective

    def temperature(self) -> np.ndarray:
        """Native-resolution float32 °C raster. Never resampled."""
        hgt, wid = self.resolution
        out = np.empty((hgt, wid), dtype=np.float32)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _check(self._lib.dirp_measure_ex(self._h, ptr, out.nbytes), "dirp_measure_ex")
        return out

    def processed_rgb(self) -> np.ndarray:
        """Native-resolution RGB pseudo-color image produced by DJI's ISP."""
        hgt, wid = self.resolution
        out = np.empty((hgt, wid, 3), dtype=np.uint8)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        _check(self._lib.dirp_process(self._h, ptr, out.nbytes), "dirp_process")
        return out


def read_temperature(path: str | os.PathLike, lib_path: str | os.PathLike | None = None) -> np.ndarray:
    """Convenience: decode one R-JPEG to a float32 °C array."""
    with RJpeg(path, lib_path) as rj:
        return rj.temperature()
