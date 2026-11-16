"""Resumable original-grid SDK scans, with one explicit outcome per input file."""
from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from pvdelta.decode import MeasurementParams, RJpeg, _lib_names
from pvdelta.radiometry import RadiometricFrameResult, _sha256


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(_canonical(value))
    temporary.replace(path)


def _identity(sdk_lib):
    configured = sdk_lib or os.environ.get('PVDELTA_DIRP_LIB')
    if not configured:
        raise ValueError('DJI SDK library folder required')
    path = Path(configured).expanduser().resolve()
    library = path if path.is_file() else next((path/n for n in _lib_names() if (path/n).is_file()), None)
    if library is None or library.name not in _lib_names():
        raise ValueError('DJI SDK library not found in configured path')
    binaries = sorted(p for p in library.parent.iterdir()
                      if p.is_file() and (p.suffix.lower() in ('.dll', '.ini') or '.so' in p.name))
    return {'schema': 'pvdelta.flight-scan-engine/v1', 'sdk_library': str(library),
        'sdk_binaries': {p.name: _sha256(p) for p in binaries},
        'engine': {n: _sha256(Path(__file__).with_name(n)) for n in
                   ('flight_scan.py', 'decode.py', 'radiometry.py')},
        'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()}


def _decode(path, sdk_lib, overrides):
    before = _sha256(path)
    with RJpeg(path, sdk_lib) as image:
        recorded = image.measurement_params
        if overrides:
            image.set_measurement_params(replace(recorded, **overrides))
        values = image.temperature()
        parameters = asdict(image.measurement_params)
    if (values.ndim != 2 or min(values.shape) < 1 or values.dtype != np.float32
            or not np.isfinite(values).all()):
        raise ValueError('A finite float32 original native grid is required; frame not assessed')
    if not all(type(v) in (int, float) and np.isfinite(v) for v in parameters.values()):
        raise ValueError('Finite recorded radiometric parameters required')
    if before != _sha256(path):
        raise ValueError('Source file changed during SDK decoding')
    result = RadiometricFrameResult(str(path), before, int(values.shape[1]), int(values.shape[0]),
        int(values.size), float(values.min()), float(values.max()), float(values.mean()),
        float(np.median(values)), parameters['emissivity'], parameters['distance_m'],
        parameters['ambient_temp_c'], parameters['reflected_temp_c'])
    return {'summary': asdict(result), 'recorded_parameters': asdict(recorded),
        'effective_parameters': parameters,
        'native_temperature_sha256': hashlib.sha256(values.tobytes(order='C')).hexdigest(),
        'native_dtype': str(values.dtype), 'native_values_resampled': False}


def scan_radiometric_flight(flight, output, *, sdk_lib=None, resume=False, progress=None,
                            measurement_overrides=None):
    """Decode every discovered DJI thermal file; errors never become healthy panels.

    Cached successes require the same source bytes, SDK binaries and engine, plus
    an intact cache document. Failures are retried. This is a private local cache,
    not an independent certificate or protection against a malicious cache author.
    A fresh run performs SDK decoding for every frame. No DB or inputs are modified.
    """
    root, destination = Path(flight).expanduser().resolve(), Path(output).expanduser().resolve()
    overrides = {} if measurement_overrides is None else measurement_overrides
    if (not isinstance(overrides, dict) or set(overrides)-set(MeasurementParams.__dataclass_fields__)
            or any(type(v) not in (int, float) or not np.isfinite(v) for v in overrides.values())):
        raise ValueError('Only finite, explicit MeasurementParams field overrides are supported')
    overrides = dict(overrides)
    if not root.is_dir():
        raise ValueError('Existing flight directory required')
    if destination.exists() and not resume:
        raise FileExistsError('Scan output exists; choose a new folder or resume explicitly')
    # Do not follow a redirected source or output subtree into another flight.
    files = sorted(p for p in root.rglob('*') if p.is_file()
        and p.name.upper().endswith(('_T.JPG', '_T.JPEG'))
        and p.resolve().is_relative_to(root) and not p.resolve().is_relative_to(destination))
    if not files:
        raise ValueError('No original DJI *_T.JPG or *_T.JPEG files found')
    if root == destination or root.is_relative_to(destination):
        raise ValueError('Output must not contain or replace the source flight directory')
    identity = {**_identity(sdk_lib), 'measurement_overrides': overrides}
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination / 'frames'
    cache.mkdir(exist_ok=True)
    # A running/restarted scan must never leave an old complete summary visible.
    _write(destination/'scan-summary.json', {'schema': 'pvdelta.flight-scan/v1',
        'status': 'running', 'requested_frames': len(files), 'engine': identity})
    records, successes, decoded, reused = [], [], 0, 0
    expected_errors = (OSError, RuntimeError, ValueError, TypeError, KeyError)
    for index, source in enumerate(files, 1):
        relative = source.relative_to(root).as_posix()
        key = hashlib.sha256(relative.encode()).hexdigest()
        target = cache/f'{key}.json'
        record = {'source_name': relative, 'status': 'failed', 'reused': False}
        try:
            source_hash = _sha256(source)
            binding = {'source_sha256': source_hash, 'engine_sha256': _digest(identity)}
            stored = None
            if resume and target.is_file():
                try:
                    candidate = json.loads(target.read_bytes())
                    payload = {k: v for k, v in candidate.items() if k != 'record_sha256'}
                    if (candidate.get('record_sha256') == _digest(payload)
                            and candidate.get('binding') == binding and candidate.get('status') == 'decoded'
                            and candidate.get('data', {}).get('summary', {}).get('source_hash') == source_hash):
                        stored = candidate['data']
                except (OSError, ValueError, TypeError):
                    pass  # Corrupt or incomplete cache is decoded again, never trusted.
            data = stored if stored is not None else _decode(source, sdk_lib, overrides)
            if data['summary']['source_hash'] != source_hash:
                raise ValueError('Source bytes changed between scan inventory and decoding')
            data['summary']['source_path'] = str(source)
            saved = {'source_name': relative, 'binding': binding, 'status': 'decoded', 'data': data}
            _write(target, {**saved, 'record_sha256': _digest(saved)})
            record.update(status='decoded', reused=stored is not None, **data)
            successes.append(RadiometricFrameResult(**data['summary']))
            reused += stored is not None
            decoded += stored is None
        except expected_errors as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            _write(target, record)  # Failed entries are not reusable successes.
        records.append(record)
        if progress is not None:
            progress(index, len(files), source)
    if identity != {**_identity(sdk_lib), 'measurement_overrides': overrides}:
        raise RuntimeError('SDK binaries or engine changed during scanning; resume with the final engine')
    failed = len(files)-len(successes)
    report = {'schema': 'pvdelta.flight-scan/v1',
        'status': 'complete' if not failed else 'partial' if successes else 'failed',
        'source_root': str(root), 'engine': identity, 'requested_frames': len(files),
        'successful_frames': len(successes), 'failed_frames': failed,
        'fresh_sdk_decodes': decoded, 'reused_frames': reused,
        'physical_panel_count': None, 'anomaly_count': None, 'field_radiometry_verified': False,
        'scope': 'SDK decode coverage only; not panel coverage, anomaly accuracy or IEC acceptance',
        'frames': records}
    _write(destination/'scan-summary.json', report)
    return report, tuple(successes)
