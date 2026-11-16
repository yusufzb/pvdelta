# Mixed native transition research

The optional `edge_model="mixed"` in `frontier_native_candidates` fits narrow or
broad Gaussian ridge profiles and monotonic step profiles against a linear
background null model. The same fixed source-centred weights are used for all
models. Four spatial strips fit independently; three must agree on family,
polarity and centre, with bounded centre uncertainty. A fixed model then needs
full-edge coverage at least .8 and maximum contiguous gap at most .18.

All four sides are intersected and checked again at the fitted quadrilateral.
Source flanks must exist; no synthetic padding or missing side is accepted.
Geometry interpolation does not alter the native temperature measurement grid.
Search/contrast/model limits are development settings, not IEC acceptance limits.

`four_native_transitions` evidence records every side's model and support audit.
Completion requires at least two ridge sides, two existing disjoint scale
references including one adjacent, and unchanged baseline measurement support.
A four-step warm rectangle is insufficient. Existing ridge defaults and legacy
line/contour support checks are preserved. No fitted candidate recursively seeds
a new search, and no temperature or reference label ranks alternatives.

This remains geometry research. Regular framed non-PV objects can pass; source
support does not prove PV membership, full-module identity, physical inventory,
fault, glare clearance or field/IEC compliance. Pure-step or unresolved cases
are not healthy panels. Verified RGB support is a separate source of evidence.
