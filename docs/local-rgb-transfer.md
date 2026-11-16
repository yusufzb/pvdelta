# Local source-bound RGB boundary preview

Basic and Advanced candidate review now enable automatic **Local panel boundary
transfer** for compatible native M4T/source-profile/calibration inputs. Selecting
a source starts the guarded per-table geometry pipeline. Green outlines show
provisional panel boundaries on the native thermal rendering and original RGB;
they are not anomaly outlines. Counts and per-table rejection reasons stay visible.
Missing prerequisites are shown as unavailable, never healthy.

The per-session cache holds at most8 frame results, bound to source/profile/
geometry/calibration/engine hashes and SDK path. Different findings on the same
frame share a result. Failures are cached to avoid retry loops; **Recheck panel
boundary transfer** retries explicitly. A changed SDK path or source binding gets
a new check. These automatic previews do not change temperature masks, findings,
review decisions or database records. Wider-source accuracy remains an open gate.

For table-isolated automatic development proposals, use
`propose_observed_junction_transfer(frame, rgb_path, matched_modules, rgb_proposals)`.
`matched_modules` are the existing bounded matcher records (`rgb_table`,
`native_polygon`, `rgb_polygon`). Each requested RGB proposal requires a unique
`proposal_id`, `polygon_rgb_xy` and explicit `table_index`. Missing table identity
retains an unavailable decision; another table's model is never substituted.

The function observes junctions in the original pair using fixed module-side
support and the8fit/4check split, then rechecks qualifying controls through the
existing source-bound local registration/transfer APIs. Failed tables retain all
requested proposal decisions. Successful transfers still require full boundary
support in both control hulls and the native frame margin. Every call validates
source/native identity before and after processing. The returned geometry remains
provisional: no temperature warp, measurement authority, database write or main
scan activation. It does not convert numerical correspondence into physical panel
identity. The current wider-source accuracy gate has failed; this API is an
integration path for guarded research/review, not a promoted automatic detector.

In **Panel boundaries**, compare a frame and open **Local RGB → thermal boundary
preview (control-assisted)**. Supply the original RGB file and local control JSON
or bundle. For DJI names, the filename suggestion requires a unique RGB file
with the same capture counter and a timestamp within one second, including across
midnight. Multiple candidates are withheld, even when one has the exact timestamp.
Other filename conventions retain the exact `_T.JPG` → `_V.JPG` suggestion.
This is a convenience only: filenames do not establish correspondence, and source
hashes/controls still verify the selected pair. Other offsets need an explicit path.

Controls use `pvdelta.local-rgb-controls/v1`, with `pair` binding the original
thermal/RGB SHA256, native temperature SHA256, native/RGB `(height, width)` grids
and coordinate spaces `thermal_native` / `original_rgb_pixels`. RGB coordinates
refer to the original encoded grid; EXIF display rotation is not applied.
`points` contain unique `control_id`, `native_xy`, `rgb_xy`, and declared `role`
(`fit` or `check`). `provenance.kind` and a nonempty `provenance.note` document
how correspondences were obtained and their limitations. Bundles contain `frames`;
exactly one pair must match both source hashes.

`fit_local_rgb_registration` fits native→RGB only on the fixed fit controls.
It requires at least eight fit and four held-out check points, distinct
observations in each sensor and two-dimensional coverage in both sets. The
default numerical gate is P95 <=2 native pixels. No check trimming, split search
or check-driven refit is performed. A failed model exposes no usable transform.

For an explicitly accepted development tolerance, call
`fit_local_rgb_registration(..., max_check_p95_px=2.20, tolerance_note="...")`.
The limit must be finite, positive and at most2.20 native pixels. A nonempty
note is mandatory above2px. `point_gate_policy` records the exact limit, reason
and development scope inside the hashed registration; changing it invalidates
the registration identity. Existing callers retain2px. This option does not
verify panel identity, make the check points independent, change an IEC limit,
activate findings or relax boundary-support/native-frame checks. The exact
source coordinates and role assignments remain unchanged.

The Streamlit control-assisted preview exposes **Development geometry tolerance
(native px)** with2.00(default) and2.20. Changing it invalidates a cached preview;
the new model must be computed explicitly. The2.20 selection records its reason
and uses a3px native interior margin; this margin is not an accuracy guarantee.

`propose_observed_junction_transfer` also accepts `full_module_edge_support=True`,
`max_check_p95_px=2.20` and an explicit `tolerance_note`. This opt-in locator fits
the middle70% of each selected module side with75% gradient support. The same
declared tolerance is used for both junction and source-bound point gates.
Default calls preserve the earlier short-support/2px behavior. Tables, sources,
missing regions and full-perimeter support decisions remain separate.

`transfer_local_rgb_boundaries` maps observed full RGB quadrilaterals into the
unchanged native grid. Every corner must lie inside **both** fit and check hulls
and the native frame margin. Unsupported, out-of-frame and failed-model regions
retain explicit decisions; they are never clipped or extrapolated. Derived GPS,
old edge evidence and physical panel identity are not inherited.

The preview extracts RGB regions from the original image using the existing
declared landscape table profile. Supported local geometry receives a separate
provisional native analysis using recorded SDK parameters, 2px ROI margin and
exploratory defaults. It does not activate findings or write to DuckDB. Save the
downloaded JSON to retain the controls hash, model, decisions, SDK parameters and
native analysis. No finding is not a healthy-panel confirmation.

Control provenance remains declared metadata: a coherent but wrong periodic
assignment can pass numerical checks. Geometry acceptance therefore **never**
sets `correspondence_verified`, `independent_review_verified` or
`measurement_authorised` true. Developer control labels are assisted development
evidence, not automatic calibration, verified panel inventory, glare clearance,
fault diagnosis or IEC field acceptance. CAD remains optional in Stage 2.

An optional **RGB Brown calibration JSON** uses
`pvdelta.rgb-brown-calibration/v1`, an exact `rgb_source_sha256`, a finite
source-sized OpenSfM Brown `camera` and `provenance.note`. Coefficients are
supplied inputs, never estimated from held-out thermal controls. A bundle may
contain `frames`; exactly one original RGB source must match. An incompatible
calibration fails explicitly rather than silently falling back to a pinhole model.

With Brown correction, fit/check RGB points are undistorted to pixel coordinates
before fitting. The accepted matrix is named `native_to_undistorted_rgb`;
`native_to_rgb` stays null because original RGB projection is nonlinear.
`project_local_native_to_rgb` reapplies distortion for supported original-image
overlays. Inverse correction must round-trip within 0.01 original RGB pixels.
OpenCV 4/5 iterative point APIs are handled separately.

For a four-corner measurement region, 17 points on every original RGB edge
are also mapped. Their perpendicular deviation from the native edge chord must
stay <=0.5 native pixels, and the full sampled perimeter must remain inside
both support hulls/frame margin. Curved edges are withheld if a quad would
misrepresent them. Only coordinates are transformed; native temperatures are
never warped. Control residuals remain visible even for failed models, without
moving, dropping or refitting check observations. Numerical diagnostics cannot
by themselves determine whether a residual is label uncertainty, lens geometry
or another physical cause.
