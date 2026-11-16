# Source-observed frame-band controls

`pvdelta.frame_band_controls` is an opt-in research observer, separate from the
native detector, measurement masks and accepted RGB overlays. It does not update
the main inspection or authorize a panel identity.

`observe_frame_band_corner` receives one sensor's uint8 structural image, a local
seed and the two inward module-edge vectors. It finds opposite signed gradients,
localizes their subpixel midpoint and intersects two supported band centre lines.
Three longitudinal strips must each retain at least60% paired support. Competing
parallel bands, missing edge pairs, poor consensus, out-of-source sampling and
excessive displacement return explicit unresolved reasons. There is no single-edge
fallback. This is intentionally conservative: an actual outer module boundary may
look like a single step, so withholding is not proof that the panel/frame is absent.

For thermal structure use the existing `radiometric_structure` monotonic display
from a verified saved-profile SDK array. The structural display is for geometry;
all temperatures remain on the unchanged original native grid. RGB brightness and
palette colours must never be interpreted as temperature.

`build_frame_band_controls` observes both sensors independently, using a frozen
module fit/check split. Every held-out corner must be observed in both sensors.
Missing fit observations are withheld on source evidence alone; their roles and
rejection records remain. Nearby fit/check observations belong to checking, avoiding
shared-frame-joint leakage. All original observed check corners are separately
retained for scoring, including duplicates excluded from the fitting API.

A draft still needs source identity, >=8 distinct fit/4 check controls, original
source-grid checks and the existing2-native-pixel held-out gate. No bad check may
be fitted, trimmed or silently omitted. Passing a numerical gate does not establish
correct physical correspondence, sufficient local hull coverage, masks, repeat
identity, glare or inspection acceptance. Do not activate measurement polygons
from centre-control geometry without separate support/semantic validation.

The local12-frame operational comparison kept all cases and original match/role
assignments: five had no eligible frozen split; seven lacked mandatory observed
check support. No new point-consistent transform was accepted. This observer is
not enabled by default. Its unresolved cases feed the existing source-review
workflow rather than deleting findings or treating regions as healthy.
