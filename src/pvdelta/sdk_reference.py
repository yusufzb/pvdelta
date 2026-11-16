"""Compare the Python decoder against DJI's local float32 CLI, never field accuracy."""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from pvdelta.decode import RJpeg
from pvdelta.measurement_profile import apply_saved_measurement_profile


def compare_dji_cli(source, profile, executable, *, sdk_library, timeout=60):
    source, executable = Path(source).resolve(), Path(executable).resolve()
    if not executable.is_file():
        raise ValueError('DJI dji_irp executable is required')
    with RJpeg(source, sdk_library) as reader:
        native = apply_saved_measurement_profile(reader, source, profile)
    parameters = profile['effective_parameters']
    names = {'distance_m': '--distance', 'humidity_pct': '--humidity', 'emissivity': '--emissivity',
             'ambient_temp_c': '--ambient', 'reflected_temp_c': '--reflection'}
    if set(parameters) != set(names):
        raise ValueError('All five effective SDK parameters are required')
    # DJI's narrow-path Windows sample may not open Unicode paths. A byte-exact
    # temporary copy avoids changing original files or measurement metadata.
    temp_root = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix='pvdelta-dji-', dir=temp_root) as directory:
        folder = Path(directory).resolve()
        if not folder.is_relative_to(temp_root):
            raise ValueError('Temporary path lies outside selected temporary root')
        copy, output = folder/'source.jpg', folder/'temperature.raw'
        copy.write_bytes(source.read_bytes())
        if hashlib.sha256(copy.read_bytes()).hexdigest() != profile['source_sha256']:
            raise ValueError('Reference input differs from saved source')
        args = [str(executable), '-s', str(copy), '-a', 'measure', '-o', str(output), '--measurefmt', 'float32']
        for name, flag in names.items():
            args.extend([flag, format(parameters[name], '.17g')])
        environment = os.environ.copy()
        environment['OMP_NUM_THREADS'] = '1'
        result = subprocess.run(args, cwd=executable.parent, env=environment,
            capture_output=True, timeout=timeout, check=False, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode != 0 or not output.is_file() or output.stat().st_size != native.size*4:
            raise ValueError('DJI CLI failed or returned an incomplete native float32 grid: '+
                result.stdout.decode(errors='replace')[-1000:])
        reference = np.frombuffer(output.read_bytes(), dtype='<f4').reshape(native.shape)
        if not np.isfinite(reference).all() or not np.isfinite(native).all():
            raise ValueError('Both native grids must be finite')
        errors = np.abs(native.astype(np.float64)-reference.astype(np.float64))
    if hashlib.sha256(source.read_bytes()).hexdigest() != profile['source_sha256']:
        raise ValueError('Original source changed during reference comparison')
    return {'schema': 'pvdelta.dji-cli-comparison/v1', 'source_sha256': profile['source_sha256'],
        'source_file': source.name, 'profile_id': profile['profile_id'], 'parameters': parameters,
        'pixels': native.size, 'shape': list(native.shape), 'exact_equal': bool(np.array_equal(native, reference)),
        'max_absolute_error_k': float(errors.max()), 'mean_absolute_error_k': float(errors.mean()),
        'python_native_sha256': hashlib.sha256(native.tobytes()).hexdigest(),
        'cli_native_sha256': hashlib.sha256(reference.tobytes()).hexdigest(),
        'cli_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
        'scope': 'DJI CLI/Python integration agreement using the same effective parameters; '
                 'not independent instrument accuracy or field validation.', 'field_accuracy_verified': False}
