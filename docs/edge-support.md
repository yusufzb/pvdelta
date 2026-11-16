# Native ridge continuity evidence

`edge_support.native_edge_continuity(temperature_c, start_xy, end_xy)` measures
two-sided narrow-ridge support along an existing native segment. It searches at
most ±1 pixel from the supplied centreline, with flanks 1–3 pixels away. A
one-sided temperature step does not count as a frame ridge. Only the dominant
hot/cold ridge polarity is used for continuity.

Returned evidence includes sampled support flags, coverage, largest unsupported
gap, valid sample count, local variation estimate and required contrast. The
variation estimate uses robust adjacent along-line differences; it is **not a
camera NETD measurement**. The contrast floor (0.15 K) and SNR multiplier (3) are
experimental geometry settings, not anomaly thresholds or field calibration.

Border samples requiring unavailable flanks remain unsupported. Disconnected
support is not filled from endpoints, table grids or nearby structures. Bilinear
sampling is for geometry evidence only; input temperatures remain unchanged and
these samples must not replace native pixels in temperature measurements.

`measurement_authorised=False` is unconditional. This is not a panel classifier
or a replacement for the existing boundary acceptance gates. Some true frame
edges are broad or one-sided and fail this narrow-ridge test. Combining it with
observed line support needs separate semantic and boundary validation.

`combine_edge_evidence(continuity, observed_intervals_fraction)` unions the ridge
flags with observed line intervals on the exact same start-to-end segment.
Intervals use ordered fractions in [0, 1]. Coverage is sampled, not the exact
analytic interval length. Unsupported gaps remain unsupported; two endpoints
do not fill the space between them. The result is always measurement-unauthorised
and does not bypass neighbour, image-border or semantic acceptance checks.
