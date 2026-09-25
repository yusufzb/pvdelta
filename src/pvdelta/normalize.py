"""ΔT normalization to reference conditions — IEC TS 62446-3:2017 §7.4.1.

    ΔT_ref = (G_ref / G_meas) ** x * ΔT_meas

The standard allows G to be irradiance *or DC current*. pvdelta supports three bases and keeps
them side by side, so a divergence between them is visible instead of hidden:

    poa_pyranometer   thermopile pyranometer in plane of array   [W/m²]
    poa_refcell       silicon reference cell in plane of array   [W/m²]
    string_current    SCADA string / MPPT DC current vs its STC value [A]

For bifacial modules the string-current basis captures the rear-side contribution that a
front-only reference cell misses.

Exponent x: 1 for faults covering one cell or more; ~1.5–1.8 for small point faults.
The standard gives no correction for point faults on crystalline modules below 60 % of reference
irradiance; pvdelta refuses to extrapolate those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

G_REF_W_M2 = 1000.0
MIN_POA_W_M2 = 600.0          # IEC TS 62446-3 minimum irradiance in module plane
POINT_FAULT_MIN_RATIO = 0.60  # below this, no point-fault extrapolation
X_AREA = 1.0
X_POINT_DEFAULT = 1.65        # midpoint of 1.5–1.8
DIVERGENCE_WARN = 0.10        # 10 % between bases


class Basis(str, Enum):
    POA_PYRANOMETER = "poa_pyranometer"
    POA_REFCELL = "poa_refcell"
    STRING_CURRENT = "string_current"


class Extent(str, Enum):
    AREA = "area"     # ≥ 1 cell: sub-string, module, string
    POINT = "point"   # < 1 cell: point hot spot, bad contact


@dataclass(frozen=True)
class Condition:
    """One measurement of the driving quantity, in its own units."""
    basis: Basis
    value: float        # W/m² for irradiance bases, A for string current
    reference: float    # 1000 W/m², or the string's STC current (e.g. Impp_STC) in A

    @property
    def ratio(self) -> float:
        """G_meas / G_ref (dimensionless)."""
        if self.value <= 0 or self.reference <= 0:
            raise ValueError(f"{self.basis.value}: non-positive value or reference")
        return self.value / self.reference


def irradiance(value_w_m2: float, basis: Basis = Basis.POA_PYRANOMETER) -> Condition:
    return Condition(basis, value_w_m2, G_REF_W_M2)


def string_current(i_meas_a: float, i_stc_a: float) -> Condition:
    return Condition(Basis.STRING_CURRENT, i_meas_a, i_stc_a)


def exponent(extent: Extent, x_point: float = X_POINT_DEFAULT) -> float:
    if not 1.5 <= x_point <= 1.8:
        raise ValueError("point-fault exponent must be within 1.5–1.8")
    return X_AREA if extent is Extent.AREA else x_point


@dataclass
class Normalized:
    delta_meas_k: float
    extent: Extent
    per_basis: dict[str, float | None] = field(default_factory=dict)
    ratios: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    primary: str | None = None

    @property
    def delta_ref_k(self) -> float | None:
        return self.per_basis.get(self.primary) if self.primary else None


def normalize(
    delta_meas_k: float,
    conditions: list[Condition],
    extent: Extent,
    *,
    x_point: float = X_POINT_DEFAULT,
    preferred: tuple[Basis, ...] = (Basis.POA_PYRANOMETER, Basis.STRING_CURRENT, Basis.POA_REFCELL),
    divergence_warn: float = DIVERGENCE_WARN,
) -> Normalized:
    """Normalize a measured ΔT with every available basis and flag inconsistencies."""
    if not conditions:
        raise ValueError("at least one condition (irradiance or string current) is required")

    x = exponent(extent, x_point)
    out = Normalized(delta_meas_k, extent)

    for c in conditions:
        r = c.ratio
        out.ratios[c.basis.value] = r
        if c.basis is not Basis.STRING_CURRENT and c.value < MIN_POA_W_M2:
            out.flags.append(f"{c.basis.value}: {c.value:.0f} W/m² below IEC minimum 600 W/m²")
        if extent is Extent.POINT and r < POINT_FAULT_MIN_RATIO:
            out.per_basis[c.basis.value] = None
            out.flags.append(f"{c.basis.value}: point fault at {r:.0%} of reference — not extrapolated")
            continue
        out.per_basis[c.basis.value] = (1.0 / r) ** x * delta_meas_k

    valid = [v for v in out.ratios.values()]
    if len(valid) > 1:
        lo, hi = min(valid), max(valid)
        if (hi - lo) / hi > divergence_warn:
            out.flags.append(
                f"bases diverge by {(hi - lo) / hi:.0%} (ratios {', '.join(f'{k}={v:.2f}' for k, v in out.ratios.items())})"
            )

    for b in preferred:
        if out.per_basis.get(b.value) is not None:
            out.primary = b.value
            break
    return out
