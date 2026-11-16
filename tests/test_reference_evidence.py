import hashlib
import json
from contextlib import nullcontext

import numpy as np
import pytest
from test_native_flight import _proposals, native_source  # noqa: F401

from pvdelta.config import DetectionConfig
from pvdelta.detect import assess_intra_module_anomalies
from pvdelta.models import ModuleObservation, ThermalFrame
from pvdelta.reference_evidence import reference_geometry


def test_reference_hull_distinguishes_interpolation_from_extrapolation():
    yy, xx = np.indices((30, 30))
    background = 20. + .1*xx + .2*yy
    reference = np.zeros((30, 30), bool)
    reference[5:25, 5:15] = True
    reference[10, 10] = False
    values = background.copy()
    values[10, 10] += 5
    values[10, 20] += 5
    inside = reference_geometry(values, reference, background, [[10, 10]], minimum_pixels=1)
    outside = reference_geometry(values, reference, background, [[10, 20]], minimum_pixels=1)
    assert inside['reference_status'] == 'interpolated_reference'
    assert outside['extrapolated_fraction'] == 1 and outside['median_residual_k'] == 5
    assert outside['minimum_pixel_support'] and outside['reference_mad_k'] == 0
    assert not outside['fault_confirmed']


def test_contaminated_degenerate_and_nonfinite_reference_is_withheld():
    ref = np.zeros((20, 20), bool)
    ref[5:15, 5:15] = True
    values = np.ones((20, 20))
    with pytest.raises(ValueError, match='contaminates'):
        reference_geometry(values, ref, values, [[10, 10]], minimum_pixels=1)
    values[5, 5] = np.nan
    with pytest.raises(ValueError, match='finite'):
        reference_geometry(values, ref, np.ones_like(values), [[3, 3]], minimum_pixels=1)
    ref[:] = False
    ref[10, 5:15] = True
    with pytest.raises(ValueError, match='two-dimensional'):
        reference_geometry(np.ones_like(values), ref, np.ones_like(values), [[3, 3]], minimum_pixels=1)


def test_opt_in_background_does_not_change_detector_and_excludes_small_outliers():
    yy, xx = np.indices((50, 50))
    values = (25+.01*xx+.02*yy).astype(np.float32)
    values[20:24, 20:24] += 8
    values[10, 10] += 9  # below display component minimum; still excluded from reference
    values[30:34, 30:34] -= 7
    original = hashlib.sha256(values.tobytes()).hexdigest()
    frame = ThermalFrame('f', 'source', 'sha', values, None, 50, 50)
    # Polygon edges use pixel boundaries; this mask includes all50x50 pixel centres.
    module = ModuleObservation('m', 'f', [(0, 0), (50, 0), (50, 50), (0, 50)])
    config = DetectionConfig(min_reference_px=24)
    mask = np.ones(values.shape, bool)
    ordinary = assess_intra_module_anomalies(frame, module, mask, config=config)
    traced = assess_intra_module_anomalies(frame, module, mask, config=config, include_background=True)
    assert ordinary.findings == traced.findings and len(traced.findings) == 2
    assert ordinary.reference_mask is None and ordinary.background_c is None
    assert not traced.reference_mask[10, 10]
    for f in traced.findings:
        points = np.asarray(f.region_pixels_yx)
        assert not traced.reference_mask[points[:, 0], points[:, 1]].any()
    json.dumps(reference_geometry(values, traced.reference_mask, traced.background_c,
        traced.findings[0].region_pixels_yx, minimum_pixels=4), allow_nan=False)
    assert hashlib.sha256(values.tobytes()).hexdigest() == original


def test_replay_refuses_changed_saved_delta_and_incomplete_population(request, tmp_path, monkeypatch):
    from pvdelta import reference_evidence
    from pvdelta.native_flight import scan_native_flight
    from pvdelta.store import connect_store

    source, decoder, _ = request.getfixturevalue('native_source')
    out = tmp_path/'native'
    scan_native_flight(source, 'flight', out, 'native-1', decoder=decoder, proposal_builder=_proposals)
    db = out/'inspection.duckdb'
    monkeypatch.setattr(reference_evidence, 'RJpeg', lambda *a: nullcontext(None))
    monkeypatch.setattr(reference_evidence, 'apply_saved_measurement_profile',
        lambda reader, path, profile: decoder(path, None).temperature_c)
    c = connect_store(db)
    fid, oid = c.execute('SELECT finding_id,observation_id FROM findings LIMIT 1').fetchone()
    c.close()
    result = reference_evidence.check_temperature_evidence(db, 'native-1', oid)
    assert result['complete_saved_population_reproduced'] and not result['database_written']
    c = connect_store(db)
    c.execute('UPDATE findings SET delta_t_k=delta_t_k+1 WHERE finding_id=?', [fid])
    c.close()
    with pytest.raises(ValueError, match='measurements'):
        reference_evidence.check_temperature_evidence(db, 'native-1', oid)
    c = connect_store(db)
    c.execute("UPDATE findings SET region_pixels_yx_json='[[15,10]]' WHERE finding_id=?", [fid])
    c.close()
    with pytest.raises(ValueError, match='population'):
        reference_evidence.check_temperature_evidence(db, 'native-1', oid)
