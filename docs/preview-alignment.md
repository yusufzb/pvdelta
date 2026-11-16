# Preview / native thermal coordinates

Equal JPEG and native radiometric dimensions do **not** establish pixel identity.
A label may match the JPEG preview exactly while occupying different coordinates
in the embedded raw thermal data. Verify both relationships before scoring:
label-image → source preview, and source preview → native grid.

`pvdelta.preview_alignment.check_preview_alignment(native_values, preview_rgb)`
is an optional, conservative **geometry** check for the same thermal capture.
It does not replace RGB/thermal camera registration and must not be used with a
separate visible-light image. It never estimates temperature from palette colours.
No labels, proposals or detector candidates are supplied to the fit.

The native array stays unchanged. Gaussian/Sobel edge representations are built
separately for the native array and RGB-encoded thermal preview. Translation is
fitted on the central 60% of each dimension. The outer strips, separated from the
fit region by 8 pixels, are used only for subsequent validation. Geometry-only
preview edges may be interpolated; native measurements are never resampled.

Fixed gates in version 1:

- Equal grids, minimum 64×64, finite real native values and uint8 three-channel
  preview. Invalid inputs raise an error.
- Central phase response ≥0.25 and peak/runner-up ratio ≥1.5, with a 5-pixel
  exclusion around the peak. Flat or repetitive/ambiguous texture is withheld.
- Translation ≤min(32 px, 8% of the shorter frame dimension).
- Forward and independently fitted reverse ECC ≥0.65; round-trip error ≤0.75 px.
- At least three textured unused strips, at least three correlations ≥0.60,
  and no usable strip correlation below 0.35. Each strip needs ≥100 valid pixels
  and standard deviations ≥0.025 in the normalised edge representations.

The result is `accepted` or `unresolved`, with explicit diagnostics. Only accepted
results include `preview_to_native`. A translation from native to preview of
`(dx, dy)` means preview labels move by `(-dx, -dy)` to enter the native grid.
No global fixed offset is assumed for a camera model.

`preview_regions_to_native(regions, alignment)` preserves all region identifiers
and class metadata, transforms coordinates only and validates the complete set.
If any transformed label is outside the frame, reject scoring for the entire
frame: do not clip/drop difficult labels. An unresolved result never falls back
to identity. Original source, labels, alignment record and native predictions
should remain available for audit.

This gate supports translation only. Rotation, scale, lens distortion, mixed
capture content or spatially ambiguous alignment may require another calibrated
method. An accepted image fit is not a calibration certificate or measured
coordinate uncertainty. Examine fixed coordinate perturbations and show
unresolved counts alongside scored denominators. Compare old and transformed
coordinates on **identical accepted frames and identical predictions**; improved
matching after moving references is a coordinate correction, not detector recall
improvement. No panel semantics, unique inventory, glare cause, surface-temperature
accuracy or IEC compliance follows. The existing production detector, database
and dashboard do not automatically use this experimental check.
