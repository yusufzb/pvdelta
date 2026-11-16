import json

import numpy as np
import pytest

from pvdelta.models import ThermalFrame
from pvdelta.native_flight import analyze_native_frame


@pytest.mark.parametrize('edge_value', [85., np.nan, np.inf])
def test_native_scan_retains_excluded_hot_or_invalid_edge_without_changing_findings(edge_value):
    values = np.full((40, 40), 25., dtype=np.float32)
    values[4, 10] = edge_value
    before = values.copy()
    frame = ThermalFrame('edge', 'edge_T.JPG', 'test', values, None, 40, 40)

    def proposals(grid, **kwargs):
        return {'proposals': [{'proposal_id': 'P1',
            'polygon_native_xy': [[4, 4], [35, 4], [35, 35], [4, 35]]}]}

    reports, diagnostic = analyze_native_frame(frame, proposal_builder=proposals)
    review = diagnostic['modules'][0]['excluded_support_review']
    assert review['status'] == 'review_required'
    assert review['pixels'] > 0
    assert review['pixels'] == review['finite_pixels'] + review['invalid_pixels']
    assert review['anomaly_assessed'] is False
    assert review['bbox_native_xyxy'] == [4, 4, 35, 35]
    if np.isfinite(edge_value):
        assert review['max_c'] == 85.
        assert review['invalid_pixels'] == 0
    else:
        assert review['invalid_pixels'] == 1
        assert review['max_c'] == 25.
    assert not reports[0].findings
    np.testing.assert_array_equal(values, before)
    json.dumps(review, allow_nan=False)
