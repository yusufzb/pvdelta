"""R-JPEG → float32 °C via the DJI Thermal SDK (libdirp), using pvdelta's own ctypes binding.

The SDK is *not* bundled. Download it from DJI and point ``PVDELTA_DIRP_LIB`` at the folder that
contains ``libdirp.so`` (Linux) or ``libdirp.dll`` (Windows).

Only the small, stable subset of ``dirp_api.h`` is bound here:

    dirp_create_from_rjpeg, dirp_destroy, dirp_get_rjpeg_resolution,
    dirp_get_measurement_params, dirp_measure_ex

VERIFY against ``tsdk-core/api/dirp_api.h`` of your SDK version before relying on
``measurement_params``: the layout of ``dirp_measurement_params_t`` has changed between SDK
releases (some versions add fields). pvdelta reads the struct into a zero-padded buffer and only
interprets the first four floats (distance, humidity, emissivity, reflection).
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
_RJpegT = TypeVar("_RJpegT", bound="RJpeg")


class DirpError(RuntimeError):
    """Raised when the DJI Thermal SDK returns a non-zero status or cannot be loaded."""


class _Resolution(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int32), ("height", ctypes.c_int32)]


# Oversized buffer so a newer SDK with extra fields cannot overrun our memory.
class _MeasureParamsBuf(ctypes.Structure):
    _fields_ = [("f", ctypes.c_float * 16)]


@dataclass(frozen=True)
class MeasurementParams:
    distance_m: float
    humidity_pct: float
    emissivity: float
    reflected_temp_c: float


def _lib_names() -> list[str]:
    if sys.platform.startswith("win"):
        return ["libdirp.dll"]
    if sys.platform.startswith("linux"):
        return ["libdirp.so"]
    raise DirpError("DJI ships the Thermal SDK for Windows and Linux only.")


def load_library(path: str | os.PathLike | None = None) -> ctypes.CDLL:
    """Load libdirp once. ``path`` may be the library file or its folder."""
    global _lib
    if _lib is not None:
        return _lib

    root = Path(path or os.environ.get(ENV_VAR, ""))
    if not str(root):
        raise DirpError(f"Set {ENV_VAR} to the DJI Thermal SDK library folder.")
    libfile = root if root.is_file() else next(
        (root / n for n in _lib_names() if (root / n).exists()), None
    )
    if libfile is None:
        raise DirpError(f"libdirp not found in {root}")

    folder = libfile.parent
    if sys.platform.startswith("win"):
        os.add_dll_directory(str(folder))  # type: ignore[attr-defined]
        lib = ctypes.CDLL(str(libfile))
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
    lib.dirp_get_measurement_params.argtypes = [h, ctypes.POINTER(_MeasureParamsBuf)]
    lib.dirp_measure_ex.argtypes = [h, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
    for fn in ("dirp_create_from_rjpeg", "dirp_destroy", "dirp_get_rjpeg_resolution",
               "dirp_get_measurement_params", "dirp_measure_ex"):
        getattr(lib, fn).restype = ctypes.c_int32

    _lib = lib
    return lib


def _check(code: int, what: str) -> None:
    if code != DIRP_SUCCESS:
        raise DirpError(f"{what} failed with DIRP code {code}")


class RJpeg:
    """Context manager around one R-JPEG handle."""

    def __init__(self, path: str | os.PathLike, lib_path: str | os.PathLike | None = None):
        self._lib = load_library(lib_path)
        self._data = Path(path).read_bytes()
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

    @property
    def resolution(self) -> tuple[int, int]:
        r = _Resolution()
        _check(self._lib.dirp_get_rjpeg_resolution(self._h, ctypes.byref(r)), "resolution")
        return r.height, r.width

    @property
    def measurement_params(self) -> MeasurementParams:
        b = _MeasureParamsBuf()
        _check(self._lib.dirp_get_measurement_params(self._h, ctypes.byref(b)), "params")
        return MeasurementParams(*(float(b.f[i]) for i in range(4)))

    def temperature(self) -> np.ndarray:
        """Native-resolution float32 °C raster. Never resampled."""
        hgt, wid = self.resolution
        out = np.empty((hgt, wid), dtype=np.float32)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _check(self._lib.dirp_measure_ex(self._h, ptr, out.nbytes), "dirp_measure_ex")
        return out


def read_temperature(path: str | os.PathLike, lib_path: str | os.PathLike | None = None) -> np.ndarray:
    """Convenience: decode one R-JPEG to a float32 °C array."""
    with RJpeg(path, lib_path) as rj:
        return rj.temperature()
