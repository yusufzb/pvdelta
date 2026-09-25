import math

import pytest

from pvdelta.normalize import Basis, Extent, irradiance, normalize, string_current


def test_area_fault_is_linear():
    n = normalize(5.0, [irradiance(800)], Extent.AREA)
    assert n.primary == "poa_pyranometer"
    assert math.isclose(n.delta_ref_k, 5.0 * 1000 / 800)
    assert n.flags == []


def test_point_fault_uses_power_law():
    n = normalize(8.0, [irradiance(750)], Extent.POINT, x_point=1.65)
    assert math.isclose(n.delta_ref_k, (1000 / 750) ** 1.65 * 8.0)
    assert 12.5 < n.delta_ref_k < 13.0  # (1000/750)^1.65 · 8 ≈ 12.86 K


def test_point_fault_below_60_percent_is_refused():
    n = normalize(8.0, [irradiance(550)], Extent.POINT)
    assert n.delta_ref_k is None
    assert any("not extrapolated" in f for f in n.flags)
    assert any("below IEC minimum" in f for f in n.flags)


def test_string_current_basis():
    n = normalize(4.0, [string_current(8.5, 10.0)], Extent.AREA)
    assert n.primary == "string_current"
    assert math.isclose(n.delta_ref_k, 4.0 / 0.85)


def test_all_bases_kept_and_divergence_flagged():
    conds = [
        irradiance(900, Basis.POA_PYRANOMETER),
        irradiance(780, Basis.POA_REFCELL),
        string_current(9.6, 10.0),
    ]
    n = normalize(3.0, conds, Extent.AREA)
    assert set(n.per_basis) == {"poa_pyranometer", "poa_refcell", "string_current"}
    assert n.primary == "poa_pyranometer"
    assert any("diverge" in f for f in n.flags)


def test_consistent_bases_not_flagged():
    conds = [irradiance(900), string_current(9.1, 10.0)]
    n = normalize(3.0, conds, Extent.AREA)
    assert not any("diverge" in f for f in n.flags)


def test_bad_exponent_rejected():
    with pytest.raises(ValueError):
        normalize(1.0, [irradiance(900)], Extent.POINT, x_point=2.5)
