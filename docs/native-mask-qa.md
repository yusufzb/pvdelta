# Native mask ownership and sparse semantic QA

`support_partition.partition_measurement_support(envelopes, margin_px=2)` offers
an explicit alternative to the existing strict disjoint-envelope builder. It
preserves every observed envelope, erodes each by the specified margin, then
excludes every shared envelope pixel from **all** measurement interiors. Shared
pixels are never assigned by proposal order, nearest centre or temperature.

Returned `PartitionedSupport` instances retain the original envelope and an
immutable unique interior. For compatibility `excluded_edge` is the entire
excluded support, including geometric margins and ownership uncertainty;
`excluded_ambiguous` identifies the latter subset that was inside the eroded ROI.
`masked_temperature` preserves selected native values exactly and uses NaN for
excluded samples. Identical overlapping envelopes can yield empty interiors;
these remain explicitly unavailable, not healthy. The strict builder is unchanged.

This proves unique native-pixel ownership only. It cannot prove that an envelope
belongs to a PV panel. Excluded pixels and original imagery must remain available
for review, especially for edge hotspots. The function does not authorise analysis
or remove an existing finding.

`boundary_benchmark.evaluate_labeled_support(reference, masks, source_sha256=...,
native_shape=...)` checks explicitly labelled `panel`, `non_panel` and `ignore`
patches on the native grid. A reference frame supplies its original file SHA256,
native dimensions, `coordinate_space="thermal_native"`, review status and regions
with unique IDs and native polygons. Wrong source identity, grid mismatch,
contradictory labels and out-of-frame reference geometry fail explicitly.

Outputs include panel-patch coverage, non-panel-patch intrusion, uncovered labelled
regions, per-proposal labelled evidence and the number of predicted pixels left
unlabelled. Frame totals use mask unions so duplicate proposals cannot inflate
coverage. These partial labels do not establish module recall, whole-mask
precision, correct instance identity, IEC conformity or release readiness.
Developer-generated labels require independent review. A missed labelled panel
patch is a known regression case; zero intrusion on sparse negative patches does
not clear the remainder of an image or a flight.
