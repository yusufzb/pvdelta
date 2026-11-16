"""Per-analysis radiometry identity, separate from original capture metadata."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pvdelta.decode import MeasurementParams


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def validate_overrides(value):
    if value is None:
        return {}
    if (not isinstance(value, dict) or set(value)-set(MeasurementParams.__dataclass_fields__)
            or any(type(v) not in (int, float) or not np.isfinite(v) for v in value.values())):
        raise ValueError('Finite, explicit SDK measurement parameter fields required')
    return dict(value)


def _parameters(value):
    if value is None:
        return None
    result = validate_overrides(value)
    if set(result) != set(MeasurementParams.__dataclass_fields__):
        raise ValueError('All five SDK measurement parameters required when a profile is supplied')
    return result


def frame_measurement_profile(frame):
    """Whitelist measurement fields; never export arbitrary raw GPS/device metadata."""
    metadata = frame.raw_metadata
    effective = _parameters(metadata.get('measurement_params'))
    recorded = _parameters(metadata.get('recorded_measurement_params'))
    overrides = validate_overrides(metadata.get('measurement_overrides'))
    if overrides and effective is None:
        raise ValueError('Requested corrections need effective SDK parameter readback')
    profile = {'schema': 'pvdelta.measurement-profile/v1', 'source_sha256': frame.source_hash,
        'native_shape': list(frame.temperature_c.shape), 'native_dtype': str(frame.temperature_c.dtype),
        'native_temperature_sha256': hashlib.sha256(frame.temperature_c.tobytes(order='C')).hexdigest(),
        'finite_native_pixels': int(np.isfinite(frame.temperature_c).sum()),
        'recorded_parameters': recorded, 'effective_parameters': effective,
        'requested_overrides': overrides,
        'sdk_parameter_readback_verified': metadata.get('sdk_parameter_readback_verified') is True,
        'field_conditions_verified': False}
    profile['profile_id'] = 'sha256:'+_digest(profile)
    return profile


def verify_measurement_profile(profile):
    if (not isinstance(profile, dict) or profile.get('schema') != 'pvdelta.measurement-profile/v1'
            or profile.get('profile_id') != 'sha256:'+_digest({k: v for k, v in profile.items() if k != 'profile_id'})):
        raise ValueError('Measurement profile identity mismatch')
    _parameters(profile['recorded_parameters'])
    _parameters(profile['effective_parameters'])
    validate_overrides(profile['requested_overrides'])
    return profile


def read_measurement_profile(connection, analysis_run_id, capture_id):
    """Read older stores without migration or guessed defaults."""
    if not connection.execute("SELECT 1 FROM information_schema.tables WHERE table_name='measurement_profiles'").fetchone():
        return None
    row = connection.execute('SELECT profile_json FROM measurement_profiles '
        'WHERE analysis_run_id=? AND capture_id=?', [analysis_run_id, capture_id]).fetchone()
    if row is None or row[0] is None:
        return None
    profile = verify_measurement_profile(json.loads(row[0]))
    source = connection.execute('SELECT source_hash FROM captures WHERE capture_id=?', [capture_id]).fetchone()
    if source is None or profile['source_sha256'] != source[0]:
        raise ValueError('Stored measurement profile belongs to another source')
    return profile


def apply_saved_measurement_profile(reader, path, profile):
    """Reconstruct and check the exact measured native grid before showing evidence."""
    verify_measurement_profile(profile)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != profile['source_sha256']:
        raise ValueError('Measurement profile source mismatch')
    effective = profile['effective_parameters']
    if effective is None:
        raise ValueError('Saved effective SDK parameters unavailable')
    if asdict(reader.measurement_params) != effective:
        reader.set_measurement_params(MeasurementParams(**effective))
    if asdict(reader.measurement_params) != effective:
        raise ValueError('SDK parameter readback differs from saved measurement profile')
    values = reader.temperature()
    if (list(values.shape) != profile['native_shape'] or str(values.dtype) != profile['native_dtype']
            or hashlib.sha256(values.tobytes(order='C')).hexdigest() != profile['native_temperature_sha256']):
        raise ValueError('Reconstructed native temperatures differ from the saved analysis')
    return values
