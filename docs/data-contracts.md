# Data contracts

All temperatures are degrees Celsius; temperature differences are kelvin-equivalent values.
Pixel points are `(x, y)` in the native radiometric grid and carry the explicit
`thermal_native` coordinate-space value. Geographic coordinates, when available, use WGS 84
latitude/longitude and always carry a method plus an estimated horizontal accuracy.

An observation may have neither a location nor an inventory match. `not_attempted`,
`insufficient_metadata`, `ambiguous`, and `unmatched` are data values; they must not be converted
to zero coordinates, a healthy assessment, or a forced panel ID. Match `ranking_score` is an
uncalibrated ordering value and is never reported as probability or confidence.

JSON writers use strict JSON and reject NaN/Infinity. The real flight manifest, source paths,
hashes, imagery, coordinates and site identifiers remain outside the repository. The public
example below is synthetic.

Intra-module findings retain `region_pixels_yx`, the exact valid component pixels in
native array order `(y, x)`. `pixel_polygon` is its display bounding box, not the
measurement mask. `absolute_t_c` is the component median; `region_max_t_c` and
`region_p95_t_c` use only the component support. `delta_t_c` is the median of
pixel-wise temperature minus the fitted reference (`delta_method`). For a varying
background this need not equal the difference of the two medians.
`region_max_residual_k` is the maximum signed residual, not cold-anomaly severity.

Capture metadata preserves `laser_target_distance_m` and `laser_status` separately
from camera altitude and the SDK radiometric distance setting. A laser range to one
target is not automatically the distance to every module in the image. Ambient
and reflected temperatures in SDK parameters are settings whose field provenance
must be recorded; they are not inferred air-temperature measurements.

An image footprint is persisted only with its projection method and a positive reviewed
horizontal-accuracy estimate. The LRF target may anchor the image centre, but LRF range accuracy
alone does not establish footprint location accuracy because GNSS, gimbal attitude, camera/LRF
boresight and the selected surface model also contribute. See
[Planar image geolocation](geolocation.md).

The optional local DuckDB store is an index over these contracts, not the canonical container
for dense temperature pixels. It stores capture, module, finding and location summaries while
referencing each native temperature array by local URI and hash. See
[Local inspection store](local-inspection-store.md).
