"""Strict FLIR FFF raw-count reader; apparent temperature only, not surface correction.

Format references: ExifTool's FLIR.pm and JPEG APP1 handler (P. Harvey).
Implemented independently from their documented binary field layout. PNG/raw
compression variants are explicitly unsupported; never infer values from JPEG
colours or guess byte order from plausible temperatures.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class FlirDecodeError(ValueError):
    """Missing, malformed or unsupported native FLIR radiometric payload."""


@dataclass(frozen=True)
class PlanckCalibration:
    r1: float
    b: float
    f: float
    o: float
    r2: float

    def __post_init__(self):
        if (not all(np.isfinite(v) for v in (self.r1, self.b, self.f, self.o, self.r2))
                or min(self.r1, self.b, self.r2) <= 0 or self.f < 0):
            raise FlirDecodeError('Finite, valid embedded Planck coefficients required')

    def apparent_temperature_c(self, counts):
        """Invert embedded blackbody calibration, without environmental correction.

        T = B / ln(R1 / (R2 * (raw + O)) + F) - 273.15.
        Any invalid pixel fails explicitly; no clipping, byte swapping guesses,
        defaults, interpolation or NaN removal is performed.
        """
        raw = np.asarray(counts)
        if (raw.ndim != 2 or raw.size == 0 or raw.dtype.kind not in 'ui'
                or (raw < 0).any() or (raw > 65535).any()):
            raise FlirDecodeError('Nonempty native integer raw-count grid required')
        denominator = self.r2 * (raw.astype(np.float64) + self.o)
        if not np.isfinite(denominator).all() or (denominator <= 0).any():
            raise FlirDecodeError('Raw counts fall outside the embedded Planck conversion domain')
        argument = self.r1/denominator + self.f
        if not np.isfinite(argument).all() or (argument <= 1).any():
            raise FlirDecodeError('Invalid logarithm or nonpositive Kelvin temperature')
        result = self.b/np.log(argument) - 273.15
        if not np.isfinite(result).all():
            raise FlirDecodeError('Nonfinite apparent temperature')
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class FlirObservationSettings:
    emissivity: float
    object_distance_m: float
    reflected_temp_c: float
    atmospheric_temp_c: float
    window_temp_c: float
    window_transmission: float
    relative_humidity_fraction: float


@dataclass(frozen=True)
class FlirRawFrame:
    raw_counts: np.ndarray
    calibration: PlanckCalibration
    observation_settings: FlirObservationSettings
    source_sha256: str
    payload_sha256: str
    fff_byte_order: str
    raw_byte_order: str
    camera_byte_order: str
    camera_model: str

    def apparent_temperature_c(self):
        return self.calibration.apparent_temperature_c(self.raw_counts)


def _payload(data):
    if len(data) < 4 or not data.startswith(b'\xff\xd8'):
        raise FlirDecodeError('JPEG SOI required')
    chunks = {}
    expected = None
    position = 2
    while position < len(data):
        if data[position] != 255:
            raise FlirDecodeError('Malformed JPEG marker before scan data')
        while position < len(data) and data[position] == 255:
            position += 1
        if position >= len(data):
            raise FlirDecodeError('Truncated JPEG marker')
        marker = data[position]
        position += 1
        if marker in (0xda, 0xd9):
            break
        if marker in (0, 0xd8, 1) or 0xd0 <= marker <= 0xd7:
            raise FlirDecodeError('Unexpected standalone JPEG marker')
        if position+2 > len(data):
            raise FlirDecodeError('Truncated JPEG segment length')
        size = int.from_bytes(data[position:position+2], 'big')
        if size < 2 or position+size > len(data):
            raise FlirDecodeError('Truncated JPEG segment')
        segment = data[position+2:position+size]
        position += size
        if marker != 0xe1 or not segment.startswith(b'FLIR\0'):
            continue
        if len(segment) < 8 or segment[:6] != b'FLIR\0\x01':
            raise FlirDecodeError('Unsupported FLIR APP1 chunk header')
        index, total = segment[6], segment[7]+1
        if index >= total or (expected is not None and total != expected) or index in chunks:
            raise FlirDecodeError('Inconsistent or duplicate FLIR chunks')
        expected = total
        chunks[index] = segment[8:]
    if not chunks or expected is None or set(chunks) != set(range(expected)):
        raise FlirDecodeError('Missing FLIR APP1 payload or incomplete chunks')
    return b''.join(chunks[i] for i in range(expected))


def _records(payload):
    if len(payload) < 64 or payload[:4] != b'FFF\0':
        raise FlirDecodeError('Supported FFF header required')
    orders = [o for o in ('>', '<') if 100 <= struct.unpack_from(o+'I', payload, 20)[0] < 200]
    if len(orders) != 1:
        raise FlirDecodeError('Unsupported FFF version/byte order')
    order = orders[0]
    offset, count = struct.unpack_from(order+'II', payload, 24)
    end = offset+count*32
    if offset < 64 or count > 1024 or end > len(payload):
        raise FlirDecodeError('Invalid or truncated FFF record directory')
    records = {}
    ranges = []
    for i in range(count):
        kind, subtype = struct.unpack_from(order+'HH', payload, offset+i*32)
        if not kind:
            continue
        start, size = struct.unpack_from(order+'II', payload, offset+i*32+12)
        if not size or start < end or start+size > len(payload):
            raise FlirDecodeError('FFF record points outside payload or into directory')
        if any(start < b and a < start+size for a, b in ranges):
            raise FlirDecodeError('Overlapping FFF records')
        ranges.append((start, start+size))
        if kind in (1, 32):
            if kind in records:
                raise FlirDecodeError('Ambiguous duplicate raw/calibration record')
            records[kind] = subtype, payload[start:start+size]
    if 1 not in records or 32 not in records:
        raise FlirDecodeError('Both raw data and camera calibration records required')
    return order, records


def read_flir_rjpeg(source):
    """Read uncompressed uint16 FFF records; never silently replace DJI decoding.

    The returned calibration produces apparent blackbody temperature ONLY.
    Stored emissivity/distance/environment are recorded but not applied. No
    corrected surface-temperature or field/SDK validation capability follows.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.stat().st_size > 64*1024*1024:
            raise FlirDecodeError('JPEG exceeds supported payload size')
        data = path.read_bytes()
    elif isinstance(source, bytes):
        data = source
    else:
        raise TypeError('Path or JPEG bytes required')
    if len(data) > 64*1024*1024:
        raise FlirDecodeError('JPEG exceeds supported payload size')
    payload = _payload(data)
    order, records = _records(payload)
    subtype, raw = records[1]
    if subtype not in (1, 2) or len(raw) < 32:
        raise FlirDecodeError('Only uncompressed BE/LE uint16 FLIR raw data supported')
    raw_order = '>' if subtype == 1 else '<'
    marker, width, height = struct.unpack_from(raw_order+'HHH', raw)
    if marker != 2 or not width or not height or width*height > 16_000_000:
        raise FlirDecodeError('Invalid raw byte order or native dimensions')
    if len(raw) != 32+width*height*2:
        raise FlirDecodeError('Raw length does not exactly match native dimensions')
    camera = records[32][1]
    if len(camera) < 0x310:
        raise FlirDecodeError('Camera calibration record truncated')
    camera_orders = [o for o in ('>', '<') if struct.unpack_from(o+'H', camera)[0] == 2]
    if len(camera_orders) != 1:
        raise FlirDecodeError('Camera record byte order is unknown')
    camera_order = camera_orders[0]
    if struct.unpack_from(camera_order+'HH', camera, 2) != (width, height):
        raise FlirDecodeError('Raw and camera native dimensions differ')

    def floating(offset):
        value = float(struct.unpack_from(camera_order+'f', camera, offset)[0])
        if not np.isfinite(value):
            raise FlirDecodeError('Nonfinite camera metadata')
        return value

    calibration = PlanckCalibration(floating(0x58), floating(0x5c), floating(0x60),
        int(struct.unpack_from(camera_order+'i', camera, 0x308)[0]), floating(0x30c))
    humidity = floating(0x3c)
    humidity = humidity/100 if humidity > 2 else humidity
    settings = FlirObservationSettings(floating(0x20), floating(0x24), floating(0x28)-273.15,
        floating(0x2c)-273.15, floating(0x30)-273.15, floating(0x34), humidity)
    if (not 0 < settings.emissivity <= 1 or settings.object_distance_m < 0
            or not 0 < settings.window_transmission <= 1 or not 0 <= humidity <= 1):
        raise FlirDecodeError('Invalid recorded observation parameters')
    counts = np.frombuffer(raw, dtype=raw_order+'u2', offset=32).reshape(height, width).astype(np.uint16)
    counts.setflags(write=False)
    model = camera[0xd4:0xf4].split(b'\0', 1)[0].decode('ascii', errors='replace')
    return FlirRawFrame(counts, calibration, settings, hashlib.sha256(data).hexdigest(),
        hashlib.sha256(payload).hexdigest(), order, raw_order, camera_order, model)
