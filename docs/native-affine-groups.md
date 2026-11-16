# Observed affine quad groups — research only

`pvdelta.native_affine_groups.affine_native_groups(native_temperature_c)` finds
2×2 groups of four **observed** closed contours. No missing fourth cell, row,
table length or CAD geometry is filled. At least two distinct Canny thresholds
must support each quad; morphological closing is not an independent vote.

The former projected aspect/right-angle priors are replaced in this alternative
by compatible local quad shapes and two-axis displacements. Opposite edges must
agree within 20%; basis condition is bounded by 6. Four observed cells must form
a non-overlapping cycle, with local shape/position tolerance .15 and neighbour
step .9–1.3. This is a **local affine approximation**, not calibrated projective
rectification or proof of physical module dimensions. Existing native area,
minimum side, contour-fill and source-border requirements remain.

Search is bounded to 512 observed quads and 4096 cycles. Budget failures withhold
all proposals and identify the reason. Conflicting overlapping groups are all
withheld. Votes/windows/groups are same-frame evidence, not different flight
angles or glare validation. Diagonal group members are not direct neighbours.

The output remains `semantic_panel_membership='unverified'`,
`measurement_authorised=False`, `efficacy_accepted=False`. Regular windows, roof
tiles, half-module contours or other non-PV quads may also satisfy these tests.
Group support cannot establish full-panel scale or replace semantic review.

For a fixed source/grid, `complete_native_proposals` may separately combine this
alternative with a checked baseline. It still requires two disjoint existing
scale references, one adjacent, no old measurement-pixel contact and no conflicting
new envelopes. New affine groups do not automatically bypass those conditions.
Callers must verify raw-source and exact native temperature hashes before
composition. No dashboard default or production mask changes here.

Geometry uses structural rendering only. Radiometry always consumes the unchanged
native temperature grid, never a warped/rectified display. Unknown or zero
proposals cannot be counted as healthy panels. Compare full-panel draft scores,
strict localisation, partial/non-PV controls and hotspot region accuracy together;
published hotspot boxes are not full-panel boundaries. Independent human/field
validation and physical same-panel identity remain separate tasks.
