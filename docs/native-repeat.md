# Native frame geometry checks

In V2, enable **Compare other photos**, select a region and press **Check native frame geometry**. This path supports the current native thermal analysis without requiring legacy RGB module polygons or a site geometry file. The existing RGB correspondence workflow and unverified nearby-photo browser remain available separately.

The check reads the source and up to eight paired thermal captures nearest its camera GPS within a 20 m horizontal radius in the same acquisition. Later return tracks are eligible even outside 30 seconds. This is retrieval, not footprint overlap, panel coordinates or independent location accuracy. Missing source GPS falls back explicitly to the original 30-second time search; invalid/missing target GPS is not treated as zero distance. Retrieval is deterministic and does not cross acquisition boundaries. The Python API retains `search_mode='time'` for comparison with the previous method.

Every original thermal file must match its indexed SHA256. Each SDK decode must reconstruct the saved measurement profile and native-temperature hash. Native diagnostic geometry, source files, GPS selection, measurement profiles, SDK binaries and relevant engine sources bind the result. A changed region, candidate set or binding clears the previous display. No computation runs just because the comparison checkbox is enabled.

## Geometry gates

1. Mutual SIFT descriptor matches initialize native image motion. A fixed spatial ordering assigns every fourth correspondence to checking before fitting or optical tracking.
2. Coarse rotation/scale is compensated on a structural display copy before forward/backward optical tracking and normalized local patch checks. This supports 180-degree return tracks; an initial point coordinate alone cannot rotate a tracking patch. Border padding cannot provide valid patch evidence. Tracks are mapped back onto the original target grid before the final fit. Check tracks never enter the initializer or final RANSAC fit. All surviving held-out tracks are scored without residual trimming; P95 must be at most three native pixels. Native temperature arrays are never warped.
3. The existing observed-panel checker independently tests unambiguous native structural controls: at least eight mutual pairs and four module-held-out folds, with P95 at most two native pixels. Weakly observed sides and ambiguous repeated proposals are excluded by the existing checker.

The module matcher requires IoU >=0.5 and a >=0.25 margin over competing
proposals in **both** directions. A target's highest score alone is insufficient:
two source proposals may tie for the same target. All observed source proposals
retain a `matching_decisions` entry describing overlap, both margins and any
centre-distance rejection. These are diagnostic decisions, not anomaly labels.

Centres are the intersection of opposite diagonals of a complete cyclic panel
quadrilateral. Unlike the vertex mean, this preserves a rectangular panel's
physical centre under a projective camera mapping. The initializer is checked
against that centre with the unchanged12px bound. A correct centre calculation
does not validate the input boundary or establish surveyed coordinates. The
centre implementation participates in the repeat-result cache binding.

These are development geometry limits, not thermography or IEC acceptance thresholds.
Motion uses a monotonic uint8 copy of the saved-profile, hash-verified SDK
temperature array, with each frame's2nd/98th percentiles mapped to0/255. It never
reads palette RGB pixels: display colours need not preserve temperature order.
Flat/insufficient-contrast frames are withheld. Percentile clipping applies only
to geometry; original temperatures, anomaly detection and measurement masks are
unchanged. Frame-specific display scaling is not a shared Celsius scale.
`motion_input` records this policy; the conversion implementation and NumPy
version participate in cache identity.

In a fixed32-pair same-flight development comparison, image-motion passes changed
from9 to13 (8 gained,4 lost,5 retained). Global panel-control passes remained0;
the counts are not independent correspondence accuracy or glare validation.

## Meaning and limitations

The table separates image-feature failures, missing structural controls and panel-control residual failures. `geometry_consistent_provisional` means the global structural control test passed. It does **not** establish the selected physical panel's identity, local coverage, component correspondence or viewing-angle diversity. A selected control target can be listed as diagnostic metadata even when the overall gate fails; no mark or temperature is transferred to it.

Every result keeps `same_panel_verified`, `measurement_authorised`, `glare_clearance` and `database_written` false. Neither a pass nor a failure confirms a fault, establishes glare or clears another region on the card. Source temperatures, findings and human reviews remain unchanged. The downloadable JSON retains all attempted targets and failure reasons.
