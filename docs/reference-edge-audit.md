# Reference edge audit

`audit_reference_edges(frame, reference)` in `pvdelta.reference_edge_audit` checks
supplied panel edges against the original radiometric pixel grid. It requires
matching source and native-temperature hashes, native dimensions and valid
non-overlapping full-frame references. It consumes no detector predictions.

Each edge receives native ridge continuity evidence and an inspection status.
The function never moves polygons, removes labels, changes TP/FP/FN or verifies
independent accuracy. A real outer edge may be a temperature step rather than a
ridge, so low ridge support is a request to inspect the source, not a rejection.
Strong support is not proof that a region is a panel either.

Before changing a detector to improve agreement with AI-interpolated drafts,
inspect label placement on unmarked original sources. Uniform reference offsets
can masquerade as localization errors. Correct labels require a separately saved
reviewed version, provenance and a documented change rationale. Do not snap labels
to current predictions and call the improved score independent accuracy. Existing
comparison reports retain the original metrics and explicitly state that unmatched
references do not establish detector errors.

The boundary comparison also validates a reference's declared native-temperature
SHA256 against the preview, rejecting malformed or mismatched hashes before
scoring. Older references without that field remain usable for geometry, but the
report explicitly records `reference_native_temperature_binding: not_supplied`;
the UI does not claim that their measurement profile has been verified.
