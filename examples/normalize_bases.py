"""Same measured ΔT, three normalization bases side by side (bifacial plant example)."""

from pvdelta import Basis, Extent, Pattern, classify, irradiance, normalize, string_current

measured_dk = 6.0  # sub-string warmer than module median, K

n = normalize(
    measured_dk,
    [
        irradiance(820, Basis.POA_PYRANOMETER),
        irradiance(790, Basis.POA_REFCELL),       # front-only cell
        string_current(9.05, 10.2),               # SCADA string current vs Impp_STC
    ],
    Extent.AREA,
)

for basis, value in n.per_basis.items():
    print(f"{basis:16s} ratio={n.ratios[basis]:.2f}  ΔT_1000={value:.2f} K")
print("flags:", n.flags or "none")

c = classify(Pattern.SUBSTRING, n.delta_ref_k)
print(
    f"verified CoA={c.coa}, provisional candidate={c.candidate_coa} "
    f"({c.pattern.value}) — {c.rule_source}"
)
