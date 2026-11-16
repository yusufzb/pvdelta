import hashlib
import os
from types import SimpleNamespace

import numpy as np
import pytest

from pvdelta import sdk_reference


def setup(monkeypatch,tmp_path):
    source=tmp_path/'source.jpg';source.write_bytes(b'original')
    exe=tmp_path/'dji_irp.exe';exe.write_bytes(b'reference')
    params={'distance_m':19.,'humidity_pct':50.,'emissivity':.95,'ambient_temp_c':20.,'reflected_temp_c':18.}
    profile={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'effective_parameters':params,'profile_id':'p'}
    values=np.arange(16,dtype=np.float32).reshape(4,4)
    class Reader:
        def __init__(self,*a):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
    monkeypatch.setattr(sdk_reference,'RJpeg',Reader)
    monkeypatch.setattr(sdk_reference,'apply_saved_measurement_profile',lambda *a:values)
    monkeypatch.setattr(sdk_reference.tempfile,'gettempdir',lambda:str(tmp_path))
    return source,exe,profile,values


def test_cli_uses_float32_and_all_effective_parameters(monkeypatch,tmp_path):
    source,exe,profile,values=setup(monkeypatch,tmp_path)
    from pathlib import Path
    def run(args,**kwargs):
        assert args[args.index('--measurefmt')+1]=='float32' and kwargs['check'] is False
        for flag in ('--distance','--humidity','--emissivity','--ambient','--reflection'):assert flag in args
        Path(args[args.index('-o')+1]).write_bytes(values.tobytes())
        return SimpleNamespace(returncode=0,stdout=b'')
    monkeypatch.setattr(sdk_reference.subprocess,'run',run)
    report=sdk_reference.compare_dji_cli(source,profile,exe,sdk_library='sdk')
    assert report['exact_equal'] and report['max_absolute_error_k']==0 and not report['field_accuracy_verified']
    assert source.read_bytes()==b'original'


def test_cli_incomplete_output_is_an_error_not_zero_temperature(monkeypatch,tmp_path):
    source,exe,profile,_values=setup(monkeypatch,tmp_path)
    monkeypatch.setattr(sdk_reference.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout=b'bad output'))
    with pytest.raises(ValueError,match='incomplete'):
        sdk_reference.compare_dji_cli(source,profile,exe,sdk_library='sdk')


@pytest.mark.sdk
@pytest.mark.skipif(not all(os.environ.get(k) for k in ('PVDELTA_DIRP_LIB','PVDELTA_SAMPLE_RJPEG','PVDELTA_DJI_CLI')),
    reason='Original R-JPEG, DJI SDK and DJI CLI are required')
def test_genuine_sdk_matches_dji_float32_cli():
    from pathlib import Path

    from pvdelta.cli import decode_frame
    from pvdelta.measurement_profile import frame_measurement_profile

    source=Path(os.environ['PVDELTA_SAMPLE_RJPEG']);sdk=os.environ['PVDELTA_DIRP_LIB']
    frame=decode_frame(source,sdk)
    report=sdk_reference.compare_dji_cli(source,frame_measurement_profile(frame),os.environ['PVDELTA_DJI_CLI'],sdk_library=sdk)
    assert report['exact_equal'] and report['max_absolute_error_k']==0.
