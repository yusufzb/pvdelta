"""Pattern → Class of Abnormality (IEC TS 62446-3:2017).

    CoA 1  no anomaly / OK
    CoA 2  thermal abnormality
    CoA 3  safety-relevant abnormality

The standard's class table has no fixed ΔT bands. ΔT figures come from its informative examples
(at 1000 W/m²). Every rule below carries its provenance and whether it has been checked against
a purchased copy of the standard (``verified``). Rules with ``verified=False`` are provisional.

Industry severity bins are kept in ``INDUSTRY_BINS`` and are never mixed into the CoA.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Pattern(str, Enum):
    NONE = "none"
    HOT_CELL = "hot_cell"
    MULTI_HOT_CELL = "multi_hot_cell"
    POINT_HOTSPOT = "point_hotspot"            # bad contact / sub-cell
    SUBSTRING = "substring"                    # 1/3 (or 1/2) of module uniformly warm
    MODULE_OPEN = "module_open"                # whole module uniformly warm
    STRING_OFFLINE = "string_offline"          # whole string uniformly warm vs neighbours
    JUNCTION_BOX = "junction_box"
    PATCHWORK = "patchwork"                    # PID-like
    BROKEN_GLASS = "broken_glass"              # from RGB
    COLD_REGION = "cold_region"                # e.g. delamination


class Cause(str, Enum):
    INTERNAL = "internal"
    SOILING = "soiling"
    BIRD_DROPPING = "bird_dropping"
    SHADOW = "shadow"
    VEGETATION = "vegetation"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Rule:
    pattern: Pattern
    coa_below: int
    threshold_k: float | None
    coa_at_or_above: int
    source: str
    verified: bool


# ΔT here is normalized to 1000 W/m² (see normalize.py).
RULES: dict[Pattern, Rule] = {
    Pattern.HOT_CELL: Rule(Pattern.HOT_CELL, 2, 40.0, 3,
                           "IEC TS 62446-3 Annex example: single hot cell 10–40 K CoA 2, >40 K CoA 3", False),
    Pattern.MULTI_HOT_CELL: Rule(Pattern.MULTI_HOT_CELL, 2, 40.0, 3,
                                 "provisional: treated like HOT_CELL", False),
    Pattern.POINT_HOTSPOT: Rule(Pattern.POINT_HOTSPOT, 2, None, 2,
                                "IEC Annex example: bad contact >10 K, CoA 2–3 (escalation manual)", False),
    Pattern.SUBSTRING: Rule(Pattern.SUBSTRING, 2, None, 2,
                            "IEC Annex example: open sub-string 2–7 K, CoA 2 (2–3)", False),
    Pattern.MODULE_OPEN: Rule(Pattern.MODULE_OPEN, 2, None, 2,
                              "IEC Annex example: open-circuit module 2–7 K uniform, CoA 2", False),
    Pattern.STRING_OFFLINE: Rule(Pattern.STRING_OFFLINE, 2, None, 2,
                                 "derived from module-open example; confirm with SCADA current", False),
    Pattern.JUNCTION_BOX: Rule(Pattern.JUNCTION_BOX, 2, None, 2,
                               "IEC Annex example: ≥3 K warmer than comparable junction box", False),
    Pattern.PATCHWORK: Rule(Pattern.PATCHWORK, 2, None, 2,
                            "IEA-PVPS T13 2018: PID patchwork; CoA provisional", False),
    Pattern.BROKEN_GLASS: Rule(Pattern.BROKEN_GLASS, 3, None, 3,
                               "IEC Annex example: broken glass CoA 3 regardless of ΔT", False),
    Pattern.COLD_REGION: Rule(Pattern.COLD_REGION, 2, None, 2,
                              "IEA-PVPS T13 2018: delamination appears colder; CoA provisional", False),
}

# Minimum |ΔT| to call something an anomaly at all. IEA-PVPS T13 (2018) cites ≥ 2.5 K as
# significant; below that, measurement uncertainty dominates. Not an IEC class boundary.
DETECTION_THRESHOLD_K = 2.5

# Industry convention — NOT IEC. Reported in a separate column.
INDUSTRY_BINS = {
    "raptor_maps_hotspot": [(10.0, "low"), (20.0, "medium"), (float("inf"), "high")],
}


@dataclass(frozen=True)
class Classification:
    coa: int
    pattern: Pattern
    cause: Cause
    delta_ref_k: float | None
    rule_source: str
    provisional: bool
    note: str = ""


def classify(pattern: Pattern, delta_ref_k: float | None, cause: Cause = Cause.UNKNOWN) -> Classification:
    """Assign a CoA from pattern first, normalized ΔT second.

    Hot cells under soiling / droppings / shadow are real heating: they keep their CoA and get a
    cause tag so the action becomes 'clean' instead of 'replace'.
    """
    if pattern is Pattern.NONE:
        return Classification(1, pattern, cause, delta_ref_k, "no anomaly", False)

    rule = RULES[pattern]

    if pattern is not Pattern.BROKEN_GLASS:
        if delta_ref_k is None:
            return Classification(2, pattern, cause, None, rule.source, True,
                                  "ΔT not normalizable; CoA from pattern only")
        if abs(delta_ref_k) < DETECTION_THRESHOLD_K:
            return Classification(1, pattern, cause, delta_ref_k, "below detection threshold", False,
                                  f"|ΔT| < {DETECTION_THRESHOLD_K} K")

    coa = rule.coa_below
    if rule.threshold_k is not None and delta_ref_k is not None and delta_ref_k >= rule.threshold_k:
        coa = rule.coa_at_or_above

    note = ""
    if cause in (Cause.SOILING, Cause.BIRD_DROPPING, Cause.SHADOW, Cause.VEGETATION):
        note = "external cause: clean / trim and re-inspect before replacement"
    return Classification(coa, pattern, cause, delta_ref_k, rule.source, not rule.verified, note)


def industry_bin(delta_k: float, scheme: str = "raptor_maps_hotspot") -> str:
    for upper, label in INDUSTRY_BINS[scheme]:
        if delta_k < upper:
            return label
    return INDUSTRY_BINS[scheme][-1][1]
