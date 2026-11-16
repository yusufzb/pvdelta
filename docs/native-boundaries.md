# Experimental native thermal boundaries

Full-envelope TP/FP/FN evaluation is documented in
[Full-panel instance evaluation](full-panel-evaluation.md). Sparse patch coverage
alone cannot establish module detection recall.

`pvdelta.native_boundaries.native_panel_proposals(temperature_c)` proposes closed
panel-shaped contours directly on a finite native SDK temperature array. The input
is not modified. A monotonic percentile display supports geometric gradients only;
temperature measurements must continue to use the original array.

```python
from pvdelta.decode import RJpeg
from pvdelta.native_boundaries import native_panel_proposals

with RJpeg(thermal_path, sdk_library) as image:
    temperature_c = image.temperature()
result = native_panel_proposals(temperature_c)
```

The method consumes no RGB polygon, calibration, GPS or CAD. It checks observed
contour closure, size, rectangular corners and aspect ratio. Duplicates combine
edge-threshold votes from the same image; these votes are not independent evidence.
Large envelopes containing multiple smaller observed envelopes are rejected.
Remaining envelopes need neighbours with compatible dimensions, orientation and
one-module spacing. No missing row or panel is filled from an assumed layout.

Returned coordinates are native `(x, y)` pixel coordinates. Each proposal lists
its observed neighbour IDs. Partial frames and unclear contours are omitted;
zero proposals does not establish an empty or healthy site.

All output is `unreviewed_native_geometry`, with semantic membership `unverified`
and `measurement_authorised=False`. Other rectangular surfaces could pass these
geometric checks. This is not a vegetation classifier, physical panel counter,
registration validator or anomaly detector. Default area, aspect and neighbour
bounds are experimental for the current flight; no general camera/altitude
transfer or IEC conformity claim has been established.

The detector does not authorise a physical panel measurement. The isolated
[native flight workflow](native-flight.md) can now compute provisional numerical
observations and review-required candidates inside these structural masks, with
source-image/semantic acceptance still unverified. RGB overlays and cross-frame
panel identity retain their own validation requirements. Existing findings are
not replaced automatically.

## Supplementary observed lines

`pvdelta.native_lines.native_line_proposals(temperature_c)` preserves the contour
proposals and can supplement them where a short gap prevents contour closure.
Line segments are grouped into two independently observed orientation families.
The common rotation is a search basis; it does not force perspective-projected
panel corners to be right angles. Collinear fragments are fitted together with a
bounded residual spread of every endpoint, retaining slope and observed gaps.
Pair eligibility is evaluated over the finite image rectangle projected into
the search basis. Separation at the coordinate origin is not a panel width:
perspective lines can diverge or reverse their ordering before reaching the
visible panel. The conservative prefilter accounts for both line-family slopes;
the actual intersection polygon must still pass every size, angle, convexity,
source-border and observed-side test. Search budgets are unchanged; exceeding
them remains explicit rather than implying an empty frame. This correctness
fix does not establish semantic vegetation rejection or better field recall.
Each proposed side needs at least 80% observed interval coverage, with no single
gap larger than 18% of side length. Disconnected segments are not counted as a
solid line. A completely missing side cannot be supplied by the table layout.

Internal supported separators reject larger rectangles spanning multiple panels.
One central separator along the long dimension can represent a half-cut seam.
New proposals require compatible observed neighbours and are deduplicated against
the preserved contour set. Search limits leave dense/ambiguous frames explicit
rather than running an unbounded combinatorial search.

Results record `source_method`, per-side coverage/gap evidence and line status.
OpenCV `(N, 4)` and `(N, 1, 4)` line arrays are supported. All thresholds remain
experimental development settings; additional outlines do not establish precision,
recall, vegetation exclusion, calibrated boundary accuracy or measurement approval.

The v2 line search combines segments from the global percentile display and a
local contrast display of native radiometry. The latter subtracts a 5 px sigma
background and divides by a 10 px sigma local RMS (floor 0.05 K), clipping only
the geometry display at ±3 local scales. These are experimental structure
parameters, not thermal anomaly thresholds. `line_structure` records them.
Neither display changes the source temperature grid or supplies measurement
values. Four-side coverage, gaps, internal separators, neighbour checks and
bounded search limits remain mandatory. Low contrast enhancement does not
establish semantic panel membership or resolve all missing boundaries.

In default segment mode, if the combined structure pass produces no proposals, a separate global-only
pass retries the same observed-side, gap, separator and neighbour checks. A
successful retry is marked `structure_pass="global_only_fallback"` with an
explicit reason. No missing side is extrapolated, no threshold is relaxed, and
no raw pass is added to a nonempty combined result. Both passes remain unreviewed.

`line_diagnostics` counts geometrically eligible candidates rejected by side
support, internal dividers, source borders and neighbour checks. Counts describe
search candidates, not distinct panels. `rejected_line_proposals` preserves
deduplicated, four-side-supported envelopes rejected by the neighbour check,
with their reason and `measurement_authorised=False`. These are review evidence
and must not be converted to measurement masks or anomaly findings. A failed raw
retry retains its diagnostics under `raw_fallback_diagnostics`.

## Opt-in hybrid proposals

`native_line_proposals(temperature_c, support_mode="hybrid")` adds native ridge
continuity to missing segment support. The default remains `"segments"`.
At least two sides must already pass the segment coverage/gap gate. Failed
sides can then union their observed intervals with source-native ridge samples;
at least one newly supported ridge sample is required. Sample discretisation
alone cannot promote a failing analytic segment score. Unsupported gaps remain
unsupported. Existing image-border, shape, deduplication and neighbour checks apply.

An internal thermal line is not always an inter-module frame. In hybrid candidates
that actually used ridge supplementation, this ambiguity can be resolved only
when two non-overlapping closed contours outside the candidate independently
support its full-module scale and orientation. The short side must agree within
15%, the long side within 10%, and the absolute long-axis dot product must be at
least .985. The candidate and references, and the two references, must have
IoU <= .05. Otherwise the internal-divider rejection remains. Provenance is
recorded under `internal_structure_resolution`; this is geometric scale evidence,
not semantic PV validation or a measured module inventory.

Supplemented outlines use `source_method="observed_lines_and_native_ridges"`,
with zero-based `native_supported_sides` and original `segment_side_coverage`.
These thresholds remain development settings. Outputs are always unreviewed and
measurement-unauthorised. Hybrid mode has not replaced production findings or
established independent-flight accuracy.

### Hybrid candidate composition

Hybrid also computes the unchanged segment baseline, including its global-only
fallback even when hybrid's combined pass is nonempty. Baseline geometry has
priority over IoU > .65 duplicates, preventing newly supplemented scores from
silently shifting existing boundaries or removing their compatible neighbours.
`structure_pass="hybrid_with_segment_baseline"` and `composition_diagnostics`
record both input passes and selection counts. Source arrays are unchanged.

A wide quad can represent several adjacent modules while still satisfying the
experimental aspect limits. When two separate, adjacent, compatible child quads
are at least 90% contained in it, the larger envelope is retained separately under
`superseded_compound_proposals`, with its child polygons and measurement disabled.
Children must each occupy less than 60% of its area, have long sides within 20%
of its short side, and point across its long direction (absolute axis dot <= .2).
The children must have IoU <= .05 and pass the existing neighbour check. A lone
nested quad, duplicate, parallel stripe, wrong-scale or mostly outside child
cannot supply that evidence. Smaller, not-already-superseded instances are used
first; closed contours can also be flagged as compounds in this opt-in mode.

After composition, neighbours are checked again and identifiers regenerated;
additional rejections remain under `composition_rejected_proposals`. Remaining
overlap is still ambiguous and is excluded by the separate support partition.
This is structural conflict resolution, not verified panel segmentation or an
automatic anomaly acceptance gate. Default segment mode is unchanged.

Supplement deduplication uses `pvdelta.convex_geometry.convex_iou`, a float64
convex clipping implementation with shared-edge/near-coincident regression tests.
The same overlap function is used in repeat-view panel selection and component
correspondence. A reproduced source case yielded a triangular intersection from
the local float32 OpenCV routine even though one quad nearly contained the other;
the corrected IoU is approximately 0.969. Ambiguous multi-panel matches still fail.
Historical contour/RGB proposal algorithms retain their frozen overlap routines
for controlled comparison; this is not a migration of every geometry caller.

## Optional local native windows

`pvdelta.native_windows.native_window_proposals(temperature_c)` retains global
hybrid proposals and supplements them with four overlapping native crops. Each
window spans two thirds of each source dimension, rounded up, and is anchored
at a source corner. This handles locally different observed directions that a
single frame-wide line-family mode can discard.

Each crop uses the existing segment/hybrid extractor and all its side-support,
gap, source-border, shape, deduplication and neighbour checks. Area fractions
are relative to the crop; large global envelopes remain available from global
search. Windows are analysis regions, not predicted module rows or grid cells.
No temperature resizing, interpolation or geometric image warp is performed.

Returned polygons and scale-reference polygons are translated to the original
native grid. Additional proposals carry `native_window_xyxy` and
`proposal_origin="local_native_window"`. Global geometry wins near-duplicate
selection, with its original provenance retained. Composition checks neighbours
and compounds again. `local_window_search` records every window and counts;
`global_search_rejections` preserves earlier global rejection evidence.

Source grids smaller than 24 px on either axis keep global search only. Bounds,
source values and existing default APIs remain unchanged. Final measurement
support must still use the full native source grid; remaining shared pixels
are ambiguous and excluded. Outputs remain unreviewed and cannot authorise
measurement, establish physical Panel IDs or replace the main anomaly run.
Independent full-panel evaluation and broader regression remain necessary.

Local envelopes that cross two retained global panels in an orthogonal direction
are withheld from the measurement proposal list under `local_orientation_conflicts`.
Both global references must be disjoint (IoU <= .05), pass the existing neighbour
check, overlap at least 25% of each reference, and together cover at least 80% of
the local envelope. Absolute long-axis dot <= .2 defines the orthogonal conflict.
A lone, duplicate, distant, grazing, parallel or already-superseded reference
cannot supply that evidence. Full geometry and reference polygons remain in the
review record with measurement disabled; this does not verify global semantics.
Remaining proposals are recomposed, their neighbours and compounds checked, and
their source provenance retained. Pre-conflict rejection history is also saved.

### Optional raw-structure supplementation

`native_window_proposals(temperature_c, raw_supplement=True)` also evaluates one
global hybrid pass without the local contrast render. This addresses a contrast
line that falsely triggers the internal-divider gate while the separate raw
pass observes a supported four-sided envelope. Existing side-coverage, gap,
hybrid-anchor, divider, border and neighbour limits are unchanged; this is not
an instruction to overlook a failed edge or draw a layout grid.

A supplemental envelope needs two disjoint, still-retained baseline envelopes
at the existing scale/orientation limits (short side within 15%, long within
10%, absolute axis dot >= .985); one must pass the existing neighbour check.
New supplements cannot supply these references to each other. Candidates that
overlap any existing envelope by IoU > .05 are withheld, except near duplicates
above .65, which are simply skipped so existing geometry keeps priority. Closed
contours alone are not admitted by this extra pass; four observed-line/ridge
envelopes use the same source support gates as global hybrid.

`raw_structure_supplement` records counts and rejections. Added envelopes carry
`proposal_origin="raw_structure_supplement"` and the two reference polygons in
`raw_scale_support`. Earlier window/conflict provenance remains available. Raw
passes still consume exact native arrays and never resize temperatures. The
option defaults to false and does not update the main anomaly run. Scale support
is unreviewed geometry evidence, not proof of panel semantics or measurement
approval. Same-flight tuning results require separate-flight validation.
