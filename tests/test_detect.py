import numpy as np
import pytest

from pvdelta.config import DetectionConfig
from pvdelta.detect import (
    assess_inter_module_anomalies,
    detect_inter_module_anomalies,
    detect_intra_module_anomalies,
)
from pvdelta.models import ModuleObservation, ThermalFrame
from pvdelta.normalize import Extent, irradiance


def fixture():
    t = np.full((50, 60), 25.0, dtype=np.float32)
    frame = ThermalFrame("f1", "synthetic", "synthetic-hash", t, None, 60, 50)
    module = ModuleObservation("m1", "f1", [(10, 10), (40, 10), (40, 40), (10, 40)])
    mask = np.zeros(t.shape, dtype=bool)
    mask[10:40, 10:40] = True
    return frame, module, mask


def test_separate_candidates_and_cold_region():
    frame, module, mask = fixture()
    frame.temperature_c[16:20, 16:20] = 35.0
    frame.temperature_c[25:29, 25:29] = 34.0
    frame.temperature_c[31:35, 16:20] = 20.0
    findings = detect_intra_module_anomalies(frame, module, mask)
    assert [f.anomaly_type for f in findings] == ["hot_region", "hot_region", "cold_region"]
    assert all(f.confidence is None and f.reference_t_c == pytest.approx(25.0) for f in findings)
    assert all(f.n_region_px == 16 and f.pixel_polygon for f in findings)
    assert [f.delta_t_c for f in findings] == pytest.approx([10.0, 9.0, -5.0])
    assert len({f.finding_id for f in findings}) == 3


def test_small_noise_and_invalid_pixels_do_not_become_findings():
    frame, module, mask = fixture()
    frame.temperature_c[20, 20] = 100.0
    frame.temperature_c[21, 21] = np.nan
    assert detect_intra_module_anomalies(frame, module, mask) == []
    frame.temperature_c[20:25, 20:25] = 30.0
    valid = np.ones(frame.temperature_c.shape, dtype=bool)
    valid[20:25, 20:25] = False
    excluded = ThermalFrame("f1", "synthetic", "synthetic-hash", frame.temperature_c,
                            valid, 60, 50)
    assert detect_intra_module_anomalies(excluded, module, mask) == []


def test_extent_is_required_for_normalization():
    frame, module, mask = fixture()
    frame.temperature_c[20:25, 20:25] = 30.0
    untyped = detect_intra_module_anomalies(frame, module, mask, conditions=[irradiance(800)])[0]
    assert untyped.is_normalized is False
    assert "extent not established" in untyped.normalization_flags[0]
    typed = detect_intra_module_anomalies(frame, module, mask,
                                          conditions=[irradiance(800)], extent=Extent.AREA)[0]
    assert typed.delta_t_c == pytest.approx(5.0)
    assert typed.normalized_delta_t_c == pytest.approx(6.25)
    assert typed.normalization_basis == "poa_pyranometer"
    assert any("formula" in flag for flag in typed.normalization_flags)


def test_uniform_module_requires_explicit_local_peers():
    t = np.full((40, 100), 25.0, dtype=np.float32)
    frame = ThermalFrame("f1", "synthetic", "synthetic-hash", t, None, 100, 40)
    modules = []
    masks = {}
    for i in range(4):
        x0 = 5 + 22 * i
        mid = f"m{i}"
        modules.append(ModuleObservation(mid, "f1", [(x0, 5), (x0+10, 5),
                                                    (x0+10, 20), (x0, 20)]))
        mask = np.zeros(t.shape, dtype=bool)
        mask[5:20, x0:x0+10] = True
        masks[mid] = mask
    t[masks["m3"]] = 35.0
    assert detect_intra_module_anomalies(frame, modules[3], masks["m3"]) == []
    groups = {m.module_id: "row-a" for m in modules}
    findings = detect_inter_module_anomalies(frame, modules, masks, groups)
    assert len(findings) == 1
    assert findings[0].module_id == "m3"
    assert findings[0].anomaly_type == "whole_module_hot_candidate"
    assert findings[0].delta_t_c == 10.0
    assert detect_inter_module_anomalies(frame, modules, masks,
                                          {**groups, "m3": "different"}) == []


def test_sufficient_reference_required():
    frame, module, mask = fixture()
    frame.temperature_c[16:20, 16:20] = 35.0
    strict = DetectionConfig(min_reference_px=10000)
    assert detect_intra_module_anomalies(frame, module, mask, config=strict) == []


def test_direct_detector_rejects_hot_pixels_outside_declared_panel():
    frame, module, mask = fixture()
    frame.temperature_c[2:8, 2:8] = 40.
    mask[2:8, 2:8] = True
    before = frame.temperature_c.copy()
    with pytest.raises(ValueError, match='outside module polygon'):
        detect_intra_module_anomalies(frame, module, mask)
    np.testing.assert_array_equal(frame.temperature_c, before)


def test_peer_detector_uses_the_same_panel_mask_guard():
    frame, module, mask = fixture()
    mask[2:8, 2:8] = True
    with pytest.raises(ValueError, match='outside module polygon'):
        assess_inter_module_anomalies(frame, [module], {'m1': mask}, {'m1': 'row'})


def test_whole_module_comparison_cannot_promote_small_valid_hot_patch():
    t = np.full((40, 100), 25., dtype=np.float32)
    valid = np.ones(t.shape, dtype=bool)
    modules, masks = [], {}
    for index in range(4):
        x = 5+22*index
        mid = str(index)
        modules.append(ModuleObservation(mid, 'f', [(x, 5), (x+10, 5), (x+10, 20), (x, 20)]))
        mask = np.zeros(t.shape, dtype=bool)
        mask[5:20, x:x+10] = True
        masks[mid] = mask
    t[masks['3']] = 35.
    valid[masks['3']] = False
    valid[7:10, 72:80] = True  # Enough reference pixels, far below the valid-fraction gate.
    frame = ThermalFrame('f', 'synthetic', 'hash', t, valid, 100, 40)
    result = assess_inter_module_anomalies(frame, modules, masks, {str(i): 'row' for i in range(4)})
    assert result.findings == ()
    assert any('3:' in flag and 'valid pixel fraction' in flag for flag in result.flags)
    # When the full panel is usable, the existing real whole-module case is retained.
    valid[:] = True
    result = assess_inter_module_anomalies(frame, modules, masks, {str(i): 'row' for i in range(4)})
    assert len(result.findings) == 1 and result.findings[0].module_id == '3'
    assert result.findings[0].delta_t_c == pytest.approx(10.)


def test_smooth_module_gradient_is_removed_before_candidate_detection():
    frame, module, mask = fixture()
    yy, xx = np.indices(frame.temperature_c.shape)
    frame.temperature_c[:] = 20 + .12 * xx + .08 * yy
    assert detect_intra_module_anomalies(frame, module, mask) == []
    frame.temperature_c[20:25, 20:25] += 5
    findings = detect_intra_module_anomalies(frame, module, mask)
    assert len(findings) == 1
    assert 4.5 < findings[0].delta_t_c < 5.5


def test_widespread_peer_group_deviation_is_reported_as_insufficient():
    t = np.full((40, 100), 25.0, dtype=np.float32)
    frame = ThermalFrame("f1", "synthetic", "hash", t, None, 100, 40)
    modules, masks = [], {}
    for index in range(4):
        x0 = 5 + 22 * index
        module_id = f"m{index}"
        polygon = [(x0, 5), (x0 + 10, 5), (x0 + 10, 20), (x0, 20)]
        modules.append(ModuleObservation(module_id, "f1", polygon))
        masks[module_id] = np.zeros(t.shape, dtype=bool)
        masks[module_id][5:20, x0:x0 + 10] = True
    t[masks["m2"] | masks["m3"]] = 35
    result = assess_inter_module_anomalies(
        frame, modules, masks, {m.module_id: "row" for m in modules},
    )
    assert result.findings == ()
    assert result.insufficient_groups == ("row",)


def test_median_reference_is_recomputed_after_all_candidates_are_excluded():
    t = np.full((10, 10), 20.0, dtype=np.float32)
    t[4:8] = 21.0
    t[8:] = 30.0
    frame = ThermalFrame("f", "synthetic", "hash", t, None, 10, 10)
    module = ModuleObservation("m", "f", [(0, 0), (10, 0), (10, 10), (0, 10)])
    finding, = detect_intra_module_anomalies(
        frame, module, np.ones(t.shape, dtype=bool),
        config=DetectionConfig(detrend_gradient=False, edge_erosion_px=0),
    )
    assert finding.reference_t_c == 20.5  # Whole-module median was 21.0.
    assert finding.delta_t_c == 9.5
    assert finding.n_reference_px == 80
    assert finding.reference_method == "module_median_excluding_candidates"


def test_component_maximum_does_not_include_invalid_pixels_inside_its_bbox():
    t = np.full((20, 20), 20.0, dtype=np.float32)
    t[5:11, 5] = 30
    t[10, 5:11] = 30
    t[7, 8] = 99  # Inside the L-shaped candidate's box, outside its support.
    valid = np.ones(t.shape, dtype=bool)
    valid[7, 8] = False
    frame = ThermalFrame("f", "synthetic", "hash", t, valid, 20, 20)
    module = ModuleObservation("m", "f", [(0, 0), (20, 0), (20, 20), (0, 20)])
    finding, = detect_intra_module_anomalies(
        frame, module, np.ones(t.shape, dtype=bool),
        config=DetectionConfig(detrend_gradient=False, edge_erosion_px=0),
    )
    assert finding.region_max_t_c == 30
    assert finding.region_p95_t_c == 30
    assert finding.region_max_residual_k == 10
    assert finding.n_region_px == 11
    assert (7, 8) not in finding.region_pixels_yx
    yy, xx = np.asarray(finding.region_pixels_yx).T
    assert np.max(t[yy, xx]) == finding.region_max_t_c
