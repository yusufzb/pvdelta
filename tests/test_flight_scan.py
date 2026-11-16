import json
import os
from dataclasses import replace

import numpy as np
import pytest

from pvdelta import cli, flight_scan
from pvdelta.decode import MeasurementParams


@pytest.fixture
def scene(tmp_path, monkeypatch):
    flight = tmp_path/'flight'
    flight.mkdir()
    good = flight/'A_T.JPG'
    good.write_bytes(b'good')
    bad = flight/'B_T.jpeg'
    bad.write_bytes(b'broken')
    (flight/'A_V.JPG').write_bytes(b'rgb')
    sdk = tmp_path/'sdk'
    sdk.mkdir()
    (sdk/flight_scan._lib_names()[0]).write_bytes(b'fake-sdk')
    calls = []
    parameters = MeasurementParams(19., 50., .95, 20., 21.)

    class Decoder:
        def __init__(self, path, library):
            self.path = path
            self.params = parameters
            calls.append(path.name)
            if path.read_bytes() == b'broken':
                raise RuntimeError('corrupt input')

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def temperature(self):
            return np.arange(20, dtype=np.float32).reshape(4, 5)+20

        @property
        def measurement_params(self):
            return self.params

        def set_measurement_params(self, parameters):
            self.params = parameters
            return parameters

    monkeypatch.setattr(flight_scan, 'RJpeg', Decoder)
    return flight, sdk, tmp_path/'scan', calls, Decoder


def test_partial_scan_keeps_later_success_and_recorded_native_parameters(scene):
    flight, sdk, output, calls, _ = scene
    (flight/'C_T.JPG').write_bytes(b'good2')
    progress = []
    report, results = flight_scan.scan_radiometric_flight(flight, output, sdk_lib=sdk,
        progress=lambda n, total, p: progress.append((n, total, p.name)))
    assert report['status'] == 'partial'
    assert report['requested_frames'] == 3 and report['failed_frames'] == 1
    assert report['successful_frames'] == report['fresh_sdk_decodes'] == 2
    assert len(results) == 2 and len(calls) == len(progress) == 3
    success = report['frames'][0]
    assert success['recorded_parameters']['humidity_pct'] == 50
    assert len(success['native_temperature_sha256']) == 64
    assert success['native_dtype'] == 'float32' and success['native_values_resampled'] is False
    assert report['physical_panel_count'] is None and report['anomaly_count'] is None
    assert json.loads((output/'scan-summary.json').read_bytes()) == report


def test_resume_reuses_only_intact_successes_and_retries_failures(scene):
    flight, sdk, output, calls, _ = scene
    flight_scan.scan_radiometric_flight(flight, output, sdk_lib=sdk)
    calls.clear()
    report, _ = flight_scan.scan_radiometric_flight(flight, output, sdk_lib=sdk, resume=True)
    assert calls == ['B_T.jpeg']
    assert report['reused_frames'] == 1 and report['failed_frames'] == 1
    (flight/'B_T.jpeg').write_bytes(b'fixed')
    report, _ = flight_scan.scan_radiometric_flight(flight, output, sdk_lib=sdk, resume=True)
    assert report['status'] == 'complete' and report['fresh_sdk_decodes'] == 1


@pytest.mark.parametrize('change', ['source', 'sdk', 'engine', 'cache'])
def test_resume_invalidates_changed_binding_or_corrupt_record(scene, monkeypatch, change):
    flight, sdk, output, calls, _ = scene
    (flight/'B_T.jpeg').unlink()
    flight_scan.scan_radiometric_flight(flight, output, sdk_lib=sdk)
    calls.clear()
    if change == 'source':
        (flight/'A_T.JPG').write_bytes(b'new original')
    elif change == 'sdk':
        (sdk/flight_scan._lib_names()[0]).write_bytes(b'new sdk')
    elif change == 'engine':
        identity = flight_scan._identity
        monkeypatch.setattr(flight_scan, '_identity', lambda path: {**identity(path), 'new_version': 2})
    else:
        cached = next((output/'frames').glob('*.json'))
        data = json.loads(cached.read_bytes())
        data['data']['summary']['temperature_max_c'] = 999
        cached.write_text(json.dumps(data))
    report, _ = flight_scan.scan_radiometric_flight(flight, output, sdk_lib=sdk, resume=True)
    assert calls == ['A_T.JPG'] and report['reused_frames'] == 0
    assert report['frames'][0]['summary']['temperature_max_c'] == 39


@pytest.mark.parametrize('invalid', ['nan', 'inf', 'empty', 'wrong_dtype', 'bad_parameters'])
def test_invalid_measurements_are_failed_and_never_cached_as_success(scene, monkeypatch, invalid):
    flight, sdk, output, _, decoder = scene
    if invalid == 'bad_parameters':
        monkeypatch.setattr(decoder, 'measurement_params', property(lambda self:
            replace(MeasurementParams(19., 50., .95, 20., 21.), emissivity=float('nan'))))
    else:
        pixels = {'nan': np.array([[np.nan]], np.float32), 'inf': np.array([[np.inf]], np.float32),
            'empty': np.empty((0, 2), np.float32), 'wrong_dtype': np.ones((4, 5), np.uint8)}[invalid]
        monkeypatch.setattr(decoder, 'temperature', lambda self: pixels)
    report, results = flight_scan.scan_radiometric_flight(flight, output, sdk_lib=sdk)
    assert report['status'] == 'failed' and not results
    assert report['successful_frames'] == 0 and report['failed_frames'] == 2


def test_source_change_during_decode_is_failed(scene, monkeypatch):
    flight, sdk, output, _, decoder = scene
    method = decoder.temperature

    def changed(self):
        self.path.write_bytes(b'changed')
        return method(self)

    monkeypatch.setattr(decoder, 'temperature', changed)
    report, results = flight_scan.scan_radiometric_flight(flight, output, sdk_lib=sdk)
    assert not results and report['failed_frames'] == 2
    assert 'changed during' in report['frames'][0]['error']


def test_interrupted_resume_replaces_old_complete_summary_with_running(scene):
    flight, sdk, output, _, _ = scene
    (flight/'B_T.jpeg').unlink()
    report, _ = flight_scan.scan_radiometric_flight(flight, output, sdk_lib=sdk)
    assert report['status'] == 'complete'

    def interrupted(*args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        flight_scan.scan_radiometric_flight(flight, output, sdk_lib=sdk, resume=True, progress=interrupted)
    assert json.loads((output/'scan-summary.json').read_bytes())['status'] == 'running'


def test_paths_with_same_basename_never_share_a_cache_slot(scene):
    flight, sdk, output, calls, _ = scene
    (flight/'leg2').mkdir()
    (flight/'leg2/A_T.JPG').write_bytes(b'other')
    report, _ = flight_scan.scan_radiometric_flight(flight, output, sdk_lib=sdk)
    assert report['requested_frames'] == 3 and len(list((output/'frames').glob('*.json'))) == 3
    calls.clear()
    report, _ = flight_scan.scan_radiometric_flight(flight, output, sdk_lib=sdk, resume=True)
    assert report['reused_frames'] == 2 and calls == ['B_T.jpeg']


def test_output_requires_explicit_resume_and_cannot_replace_flight(scene):
    flight, sdk, output, _, _ = scene
    flight_scan.scan_radiometric_flight(flight, output, sdk_lib=sdk)
    with pytest.raises(FileExistsError):
        flight_scan.scan_radiometric_flight(flight, output, sdk_lib=sdk)
    with pytest.raises(ValueError):
        flight_scan.scan_radiometric_flight(flight, flight, sdk_lib=sdk, resume=True)
    assert (flight/'A_T.JPG').read_bytes() == b'good'


def test_cli_partial_exit_code_and_explicit_resume(scene, capsys):
    flight, sdk, output, _, _ = scene
    args = ['scan-flight', str(flight), '--sdk-lib', str(sdk), '--out', str(output)]
    assert cli.main(args) == 3
    report = json.loads(capsys.readouterr().out)
    assert report['failed_frames'] == 1 and 'frames' not in report
    assert cli.main(args) == 1
    capsys.readouterr()
    assert cli.main(args+['--resume']) == 3
    assert json.loads(capsys.readouterr().out)['reused_frames'] == 1


def test_explicit_parameter_profile_preserves_recorded_values_and_invalidates_cache(scene):
    flight, sdk, output, calls, _ = scene
    (flight/'B_T.jpeg').unlink()
    report, _ = flight_scan.scan_radiometric_flight(flight, output, sdk_lib=sdk)
    assert report['frames'][0]['recorded_parameters'] == report['frames'][0]['effective_parameters']
    calls.clear()
    report, results = flight_scan.scan_radiometric_flight(flight, output, sdk_lib=sdk, resume=True,
        measurement_overrides={'distance_m': 23.})
    assert calls == ['A_T.JPG'] and report['reused_frames'] == 0
    assert report['frames'][0]['recorded_parameters']['distance_m'] == 19.
    assert report['frames'][0]['effective_parameters']['distance_m'] == 23.
    assert results[0].distance_m == 23.
    calls.clear()
    report, _ = flight_scan.scan_radiometric_flight(flight, output, sdk_lib=sdk, resume=True,
        measurement_overrides={'distance_m': 23.})
    assert not calls and report['reused_frames'] == 1


@pytest.mark.parametrize('overrides', [{'distance_m': float('nan')}, {'distance_m': True},
    {'height': 19.}, [], 'distance=19'])
def test_invalid_overrides_fail_globally_before_scanning(scene, overrides):
    flight, sdk, output, calls, _ = scene
    with pytest.raises(ValueError, match='MeasurementParams'):
        flight_scan.scan_radiometric_flight(flight, output, sdk_lib=sdk, measurement_overrides=overrides)
    assert not calls and not output.exists()


@pytest.mark.sdk
@pytest.mark.skipif(not (os.environ.get('PVDELTA_DIRP_LIB') and os.environ.get('PVDELTA_SAMPLE_RJPEG')),
                    reason='original DJI sample and SDK required')
def test_real_sdk_native_values_match_direct_decode_with_fault_isolation(tmp_path):
    from pathlib import Path
    from shutil import copyfile

    from pvdelta.decode import RJpeg

    flight = tmp_path/'flight'
    flight.mkdir()
    source = flight/'A_T.JPG'
    copyfile(Path(os.environ['PVDELTA_SAMPLE_RJPEG']), source)
    (flight/'B_T.JPG').write_bytes(b'not a radiometric jpeg')
    report, _ = flight_scan.scan_radiometric_flight(flight, tmp_path/'report',
        sdk_lib=os.environ['PVDELTA_DIRP_LIB'])
    with RJpeg(source) as image:
        expected = image.temperature()
    import hashlib

    assert report['status'] == 'partial'
    assert report['fresh_sdk_decodes'] == 1 and report['failed_frames'] == 1
    assert report['frames'][0]['native_temperature_sha256'] == hashlib.sha256(expected.tobytes()).hexdigest()
