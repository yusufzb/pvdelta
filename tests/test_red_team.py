import numpy as np
import pytest

from pvdelta.config import DetectionConfig
from pvdelta.models import ThermalFrame
from pvdelta.pipeline import analyze_annotations
from pvdelta.regions import ModuleAnnotation


def scene():
    t = np.full((40, 80), 25., np.float32)
    t[12:16, 12:16] = 35
    return t, ModuleAnnotation('m', ((5, 5), (25, 5), (25, 30), (5, 30)))


def frame(t, valid=None):
    return ThermalFrame('f', 'synthetic', 'synthetic', t, valid, t.shape[1], t.shape[0])


def test_invalid_pixel_outside_panel_does_not_hide_hotspot():
    t, a = scene(); t[0, 0] = np.nan
    result, = analyze_annotations(frame(t), [a])
    assert result.assessment_status == 'analyzed'
    assert len(result.findings) == 1


def test_eroded_away_panel_is_not_reported_as_analyzed():
    t, _ = scene()
    a = ModuleAnnotation('thin', ((30, 5), (32, 5), (32, 35), (30, 35)))
    result, = analyze_annotations(frame(t), [a])
    assert result.assessment_status == 'not_assessed'


def test_mostly_invalid_panel_cannot_count_as_no_finding():
    t, a = scene(); valid = np.ones(t.shape, bool); valid[5:26, 5:25] = False
    result, = analyze_annotations(frame(t, valid), [a])
    assert result.assessment_status == 'not_assessed'


def test_mask_outside_declared_panel_is_rejected():
    t, a = scene(); mask = np.ones(t.shape, bool)
    result, = analyze_annotations(frame(t), [a], module_masks={'m': mask})
    assert result.assessment_status == 'not_assessed'


def test_edge_touching_candidate_is_excluded_by_measurement_mask():
    t = np.full((40, 80), 25., np.float32)
    annotation = ModuleAnnotation('edge', ((5, 5), (25, 5), (25, 30), (5, 30)))
    t[28:30, 12:16] = 35.0
    result, = analyze_annotations(
        frame(t), [annotation],
        config=DetectionConfig(detrend_gradient=False, reject_boundary_findings=True),
    )
    assert result.findings == ()
    assert 'edge-touching candidates excluded' in result.reasons[0]


def test_edge_touching_candidate_stays_visible_for_review_by_default():
    t = np.full((40, 80), 25., np.float32)
    annotation = ModuleAnnotation('edge', ((5, 5), (25, 5), (25, 30), (5, 30)))
    t[28:30, 12:16] = 35.0
    result, = analyze_annotations(
        frame(t), [annotation], config=DetectionConfig(detrend_gradient=False)
    )
    assert result.findings
    assert all(f.assessment_status == 'review_required' for f in result.findings)


def test_explicit_peer_groups_detect_uniform_hot_module_in_pipeline():
    t = np.full((30, 100), 25., np.float32)
    annotations = []
    for i in range(4):
        x = 3+24*i
        annotations.append(ModuleAnnotation(str(i), ((x, 3), (x+18, 3), (x+18, 27), (x, 27)),
                                            peer_group='row'))
        if i == 3: t[3:27, x:x+18] = 35
    results = analyze_annotations(frame(t), annotations)
    assert any(f.anomaly_type == 'whole_module_hot_candidate' for f in results[3].findings)


def test_overlapping_panel_masks_cannot_double_count_hotspot():
    t, a = scene(); b = ModuleAnnotation('other', a.polygon)
    results = analyze_annotations(frame(t), [a, b])
    assert all(r.assessment_status == 'not_assessed' for r in results)


def test_pixel_count_settings_reject_fractional_values():
    with pytest.raises(ValueError):
        DetectionConfig(edge_erosion_px=1.5)


def test_piecewise_registration_cannot_extrapolate_far_beyond_feature_support():
    pytest.importorskip('cv2')
    from pvdelta.table_surface import fit_piecewise_table_mapping, map_piecewise_table_pixels
    camera = {'projection_type': 'brown', 'width': 1600, 'height': 1200, 'focal_x': .7,
              'focal_y': .7, 'c_x': 0., 'c_y': 0., 'k1': 0., 'k2': 0., 'k3': 0., 'p1': 0., 'p2': 0.}
    rng = np.random.default_rng(2)
    xy = rng.uniform([-10, -2], [10, 2], (800, 2))
    pixels = xy*40+[800, 600]
    model, _ = fit_piecewise_table_mapping(xy, pixels, camera, [-10, 10])
    mapped = map_piecewise_table_pixels(np.array([[0., 0.], [0., 20.]]), model, camera)
    assert np.isfinite(mapped[0]).all()
    assert np.isnan(mapped[1]).all()


def test_reference_never_includes_another_detected_region():
    from pvdelta.detect import assess_intra_module_anomalies
    from pvdelta.models import ModuleObservation
    rng = np.random.default_rng(33)
    for _ in range(2):
        t = rng.normal(25, 1, (20, 20)).astype('float32')
        t[3:9, 3:9] += 5; t[11:17, 11:17] -= 5
    module = ModuleObservation('m', 'f', [(0,0),(20,0),(20,20),(0,20)])
    result = assess_intra_module_anomalies(frame(t), module, np.ones(t.shape, bool),
                                          config=DetectionConfig(edge_erosion_px=0))
    assert result.findings
    pixels = {p for f in result.findings for p in f.region_pixels_yx}
    assert all(f.n_reference_px+len(pixels) <= t.size for f in result.findings)
