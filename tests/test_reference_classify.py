import numpy as np

from pvdelta.classify import Cause, Pattern, classify, industry_bin
from pvdelta.reference import delta_t_region, erode, group_median_delta


def synthetic_module(hot_k: float = 12.0):
    t = np.full((60, 100), 45.0, dtype=np.float32)
    t += np.random.default_rng(0).normal(0, 0.2, t.shape).astype(np.float32)
    module = np.zeros_like(t, dtype=bool)
    module[5:55, 5:95] = True
    cell = np.zeros_like(module)
    cell[20:26, 40:46] = True
    t[cell] += hot_k
    return t, module, cell


def test_hot_cell_delta():
    t, module, cell = synthetic_module(12.0)
    d = delta_t_region(t, cell, erode(module), sign="hot")
    assert 11.5 < d.delta_k < 13.0
    assert d.reference == "module_median"


def test_cold_region_is_negative():
    t, module, cell = synthetic_module(-4.0)
    d = delta_t_region(t, cell, module, sign="cold")
    assert d.delta_k < -3.5


def test_erode_shrinks():
    m = np.zeros((10, 10), bool)
    m[2:8, 2:8] = True
    assert erode(m).sum() == 16


def test_group_median_delta():
    d = group_median_delta({"S1": 45.0, "S2": 45.3, "S3": 50.1, "S4": 44.8})
    assert d["S3"] > 4.5


def test_classify_pattern_first():
    assert classify(Pattern.HOT_CELL, 25.0).candidate_coa == 2
    assert classify(Pattern.HOT_CELL, 45.0).candidate_coa == 3
    assert classify(Pattern.BROKEN_GLASS, 0.0).candidate_coa == 3
    assert classify(Pattern.SUBSTRING, 1.0).candidate_coa == 1
    assert classify(Pattern.HOT_CELL, 25.0).coa is None
    assert classify(Pattern.NONE, None).coa == 1
    assert classify(Pattern.UNKNOWN, None).coa is None


def test_external_cause_keeps_class():
    c = classify(Pattern.HOT_CELL, 18.0, Cause.BIRD_DROPPING)
    assert c.candidate_coa == 2
    assert "external cause" in c.note


def test_industry_bins_separate():
    assert industry_bin(8) == "low"
    assert industry_bin(15) == "medium"
    assert industry_bin(25) == "high"
