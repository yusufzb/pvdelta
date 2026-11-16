# Native panel measurement masks

Native flight diagnostics retain `excluded_support_review` per provisional module:
excluded edge/ownership-mask hash, half-open native bbox, pixel/finite/invalid
counts, maximum and P95 temperature. Nonempty excluded support requires review;
these statistics do not detect an anomaly or establish panel membership. They do
not change the interior mask, source temperature raster or finding population.
Modules missing from geometry extraction have no such record and remain a
separate coverage gap. Existing runs require a separate rerun to gain these fields.

`pvdelta.masking.build_measurement_support` accepts disjoint boolean panel envelopes already
registered to the native thermal grid. It returns an interior measurement mask and a separate
excluded edge band. Overlapping instances are rejected instead of counting a pixel twice.

For DJI R-JPEG, `RJpeg.processed_rgb()` returns the SDK pseudo-color image on the same native
grid as `RJpeg.temperature()`. Use that image when a visible thermal rendering is needed for
native-grid segmentation or registration. A 1280×1024 embedded high-resolution JPEG must not be
treated as an exact 2× copy of a 640×512 radiometric raster merely because its dimensions double.

`preview_mask_to_native` is intentionally strict: it accepts only an exact integer grid ratio and
keeps a native pixel only when every covering preview subpixel is selected. Call it only after the
preview-to-native crop, orientation and pixel-centre mapping have been independently validated for
that image product. It does not make an uncalibrated preview safe.

The margin is an explicit integer in native thermal pixels. Select it from the frame/mixed-pixel
allowance and measured registration uncertainty. The current erosion uses four-neighbour steps;
this is a grid margin, not a calibrated Euclidean error bound.

Temperature is never used to choose the panel envelope or its interior. A hot pixel inside the
interior remains available for measurement. Non-finite interior pixels are excluded by
`valid_mask`, but do not change the geometric boundary or become part of the edge band.

`masked_temperature` produces a display/export array with NaN outside valid support. Keep the
original native temperature array for analysis and pass the boolean mask separately: the current
frame quality check rejects NaN frames. Empty interiors must be reported as unavailable, never
healthy. Excluded edges and partial panels remain unassessed until their own review is complete.

This utility does not detect panels or validate RGB-to-thermal registration. Automatic boundary
candidates still require instance-level validation for missed, split and merged modules. A fixed
RGB threshold is not sufficient evidence of reliable whole-flight segmentation.

`summarize_temperature` reports min/median/P95/max on finite interior pixels, plus valid and
invalid counts. Empty support returns `unavailable` and null temperatures; incomplete finite
support remains explicit. These are descriptive statistics, not an anomaly decision. Keep
source hashes, the radiometric settings and boundary-review status with each exported record.

## Experimental RGB boundary proposals

`pvdelta.table_boundaries.refined_module_proposals` combines table-first separators and
cell-grid-supported contours for near-nadir, landscape arrays. The editable defaults describe
four rows across the slope, 2278 × 1113 mm modules, 25° tilt and a 24 × 6 half-cell layout.
The visible segment may contain any number of complete modules: 9/27 table lengths are not
used to fill unseen panels or force a module count.

The table envelope is rectified, row separators are detected from image structure, and
long-axis joins are estimated separately within each row. Four local frame lines refine each
module quadrilateral. A contour fallback requires the declared cell-grid signature; overlapping
and contained proposals are suppressed. Colour support helps exclude vegetation, but these
site-dependent thresholds are not a general-purpose segmentation model.

```powershell
./.venv/Scripts/python.exe -m pvdelta.cli propose-boundaries private/flight/DJI_0001_V.JPG --out private/rgb-boundaries.json
```

The CLI reads gimbal pitch from image metadata. Unknown pitch yields `pose_unavailable`;
views more than 10 degrees away from nadir yield `deferred_oblique_geometry`. `--pitch-deg`
records an explicit override. Other options include `--rows`, `--tilt-deg`, `--module-length-m`,
`--module-width-m`, `--cell-columns` and `--cell-rows`.

Outputs contain the source hash, original RGB pixel dimensions, configuration, table diagnostics
and per-module provenance (`observed_table_separators` or `contour_cell_signature`). Their
status is `unreviewed_rgb_geometry`. Writing over an existing report requires `--overwrite`;
the source image is protected even with this flag. Transfer to native thermal coordinates
requires a separate registration check before these proposals can become measurement masks.
Image-local row indices are not physical A–D labels or Panel IDs.

Local development checks showed fewer split-module errors in a small labelled area, but
whole-flight segmentation accuracy and RGB-to-native registration remain open. The existing
automatic analysis route is not switched to this experimental proposer.

## Experimental pair/table registration checks

`pvdelta.pair_registration.refine_panel_registration(native_gray, prior_quads)` fits
a bounded local affine correction for at least four modules on one table in one
RGB/thermal pair. Input polygons are already in native thermal pixels through an
explicit provisional prior. The structural image must use the same native grid.
No temperature pixels are resampled or inferred from the display colours.

Module indices are split spatially before fitting. Oriented edge distance,
ambiguity, transform bounds and unchanged four-side checks can withhold the result.
Adjacent modules may share borders, so the excluded module check is image consistency,
not independent correspondence validation. Repetitive neighbouring panels can pass
this image gate despite wrong identity.

`check_registration_controls(result, prior_points, observed_native_points)` evaluates
separate source-image controls without fitting them. Use independently identified
corners within the fitted table area and the same pair; an affine correction must
not be extrapolated across a whole frame or reused on other pairs. The default
2 native pixel P95 limit is an experimental check, not an IEC requirement. Caller
review must establish control provenance and physical correspondence.

Results explicitly retain `measurement_status=withheld_requires_independent_correspondence_validation`.
Even a passing control-point audit covers only those controls; panel boundaries,
view identity, radiometric parameters and field conditions need their own acceptance.
This optional API does not alter the default analysis pipeline or existing findings.

## Experimental observed-junction registration

`measurement_control_split=True` optionally assigns every third distinct observed
node (index modulo3 equals1) to checking and the remainder to fitting. It requires
at least12 matches and validates at least8fit/4check, with separate two-dimensional
role geometry. This fixed policy is incompatible with `paired_block_split`; no
residual-based split search is performed. The option defaults false. A qualifying
development result can be rechecked by the existing source-bound local-control
transfer API; physical correspondence and measurement authority remain unverified.

Optional `JunctionConfig(module_side_only=True, paired_block_split=True)` fits
frame lines on the arm pointing into the observed module; both arms remain in
the independent termination diagnostic. Fixed pairs of consecutive observed nodes
alternate FIT/CHECK blocks, rather than assigning opposite table edges to different
roles. There is no residual-based role search or fallback. Each role is checked
for collinear/near-collinear geometry (minor/major singular value <=0.001) before
solving. Both options default false pending broader evaluation. A passing local
development check still does not authorise measurement or cross-frame transfer.

`pvdelta.junction_registration.match_table_junctions(rgb_gray, native_structure,
rgb_quads, prior_native_quads)` observes frame intersections separately in both
sensors for one table/pair. The paired quads supply bounded search priors with matching
corner order. Visible corners from partial modules may support registration; no partial
module measurement mask is produced. Inputs and outputs use their original pixel grids.

`radiometric_structure(temperature_c)` creates a monotonic P2–P98 structural image and
normalisation metadata. It leaves the native temperature array unchanged. This image
is for gradient geometry only: measure anomalies on the original radiometric values.
Non-finite, flat or insufficient-sized structural inputs fail explicitly.

The matcher rejects both one-to-many and many-to-one correspondences before a fixed
FIT/CHECK split. Duplicate observations carry one weight. Normalised DLT fits a
native-to-RGB homography; check error is in native pixels. Compatible L-corner arm
terminations are separate provisional anchor evidence. Unobserved arms at a crop border
are not treated as missing physical structure. Repetitive interior grids cannot prove
same-panel identity.

Only image-consistent models return a `native_to_rgb` matrix. Limit use to the fitted
support hull in the same pair; independent controls and physical identity remain
mandatory. `measurement_status` stays withheld. The default 2 px check is experimental,
not an IEC requirement. Current real-flight replay did not pass this gate, so this API
remains a review experiment and does not change the production detector or dashboard.

Frame centres use separately localised opposite edges and their subpixel midpoint.
This avoids choosing the first maximum of a contrast-dependent ridge plateau. Both
edges must straddle the detected band; a stronger neighbouring cell-grid edge on
the same side cannot substitute for the opposite frame edge. A missing edge does
not yield a two-edge midpoint. Source-corner QA and physical identity are still
required: an automatic image check can pass while separate controls fail.
