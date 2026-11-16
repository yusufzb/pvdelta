# Provisional orthoviews

**Quality status:** the sample planar mosaic still has visible geometric seams and is not
an accepted orthophoto. See [local ODM reconstruction](odm.md) for the replacement geometry
pipeline. Passing software tests or sharpening a preview does not validate image alignment.

`pvdelta.orthoview` creates paired RGB and thermal orthoviews on a local ENU grid. Each
frame is ray-cast from its DJI laser target onto a declared planar surface, then warped into
the common grid with a projective transform. Thermal frames are decoded through the DJI SDK;
the generated `thermal_temperature_c.npy` file remains the Celsius measurement grid while the
PNG is only a display rendering.

The current sample build uses a 25° south-facing plane and 0.05 m/px. It is explicitly marked
`provisional`: camera GPS and the LRF target anchor do not prove the exact panel plane or
boresight. Validate the surface, azimuth and registration with CAD or independent surveyed
control before making survey-grade distance or module-location claims.

Overlapping frames are never averaged. A display-only centre/clarity score selects the strongest
registered source pixel, which removes the double edges and blur caused by small pose or plane
errors. After the first RGB frame, a phase-correlation response below 0.08 rejects that frame's
RGB contribution and records `rejected_low_confidence` in the manifest; thermal radiometric
pixels remain independently projected.
Thermal display colours use a robust 2nd–98th percentile stretch; the Celsius grid is untouched.

Each build writes:

- `rgb_orthoview.png` — RGB display mosaic;
- `thermal_orthoview.png` — temperature-colour display mosaic;
- `thermal_temperature_c.npy` — float32 Celsius grid with NaN outside coverage;
- `coverage_count.png` — number of contributing frames per pixel;
- `orthoview_manifest.json` — plane, bounds, resolution, frame counts and source provenance.

RGB and thermal use their own camera FOVs but share the same projected surface coordinates.
Anomaly detection continues to use native radiometric pixels; the orthoview is a review and
navigation product rather than a replacement measurement grid.
