import hashlib
from copy import deepcopy

import numpy as np
import pytest

from pvdelta.models import ThermalFrame
from pvdelta.reference_edge_audit import audit_reference_edges


def sample(shift=0):
    _yy, xx = np.mgrid[:100, :140]
    values = (25-2*np.exp(-.5*((xx-30)/.7)**2)-2*np.exp(-.5*((xx-70)/.7)**2)).astype(np.float32)
    frame = ThermalFrame('synthetic', 'synthetic_T.JPG', 'a'*64, values, None, 140, 100)
    reference = {'schema': 'pvdelta.native-panel-instances/v1', 'coordinate_space': 'thermal_native',
        'source_sha256': 'a'*64, 'native_temperature_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
        'native_height': 100, 'native_width': 140, 'coverage': 'exhaustive_full_frame',
        'review_status': 'developer_draft', 'panels': [{'instance_id': 'R1',
        'polygon_native_xy': [[30+shift, 10], [70+shift, 10], [70+shift, 90], [30+shift, 90]]}], 'ignore': []}
    return frame, reference


def test_shifted_draft_is_diagnosed_from_native_source_without_predictions():
    frame, ref = sample()
    _, shifted = sample(9)
    original, pixels = deepcopy(ref), frame.temperature_c.copy()
    aligned = audit_reference_edges(frame, ref)
    displaced = audit_reference_edges(frame, shifted)
    assert all(aligned['panels'][0]['edges'][i]['ridge_supported'] for i in (1, 3))
    assert not any(displaced['panels'][0]['edges'][i]['ridge_supported'] for i in (1, 3))
    assert ref == original and np.array_equal(pixels, frame.temperature_c)
    assert not any(aligned[k] for k in ('prediction_geometry_consumed', 'reference_correctness_verified',
                                       'independent_review_verified', 'reference_modified', 'scores_modified'))


def test_flat_and_step_only_edges_remain_unverified_not_invalid_labels():
    frame, ref = sample()
    frame.temperature_c[:] = 25
    frame.temperature_c[:, 30:] = 30
    ref['native_temperature_sha256'] = hashlib.sha256(frame.temperature_c.tobytes()).hexdigest()
    result = audit_reference_edges(frame, ref)
    assert result['panels'][0]['source_inspection_requested']
    assert result['panels'][0]['reference_correctness'] == 'unverified'
    assert not result['scores_modified'] and not result['measurement_authorised']


@pytest.mark.parametrize('change', ['source', 'native_hash', 'grid', 'nan', 'overlap'])
def test_wrong_source_or_grid_never_produces_audit(change):
    frame, ref = sample()
    if change == 'source': ref['source_sha256'] = 'c'*64
    elif change == 'native_hash': ref['native_temperature_sha256'] = 'c'*64
    elif change == 'grid': ref['native_width'] = 141
    elif change == 'nan': frame.temperature_c[0, 0] = np.nan
    else: ref['panels'].append(dict(deepcopy(ref['panels'][0]), instance_id='R2'))
    with pytest.raises(ValueError): audit_reference_edges(frame, ref)
