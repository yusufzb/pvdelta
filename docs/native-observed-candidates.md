# Observed native candidates before grouping

`observed_native_candidates` is a research extraction API. It exposes actual
closed quads and four-supported-segment quads before the four-cell cycle gate.
The five native windows, shape limits, contour votes and segment coverage/gap
requirements match `line_affine_native_groups`. Missing sides are never filled.
Existing production and grouped extraction are unchanged.

The caller verifies the source and native SHA256, then applies a separate
completion policy. For example, `complete_native_proposals` still requires two
disjoint existing scale references, at least one adjacent, and preserves every
old measurement interior. This pool alone cannot authorise a measurement mask.
New candidates cannot supply each other's baseline scale references.

Input temperatures are unchanged; structural rendering is only for geometry.
There is no temperature ranking, reference-label input, CAD prior or inferred
grid. Overflow withholds the full pool instead of returning an arbitrary subset.
All geometry stays semantically unverified: regular non-PV structures may pass.
Results do not establish physical inventory, fault, glare clearance or IEC field
compliance. Empty output is not a healthy-panel determination.
