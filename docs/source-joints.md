# Explicit source-joint review

`pvdelta.source_joints` evaluates explicit aliases for reviewed observations of the
same physical frame joint. It does not discover identities by proximity or residuals.
Call `source_joint_binding(pack, review_directory, frame_index)` to bind an alias
document to the exact pack, source pair and current source-review decisions:

```python
aliases = {
    "schema": "pvdelta.source-joint-aliases/v1",
    "binding": source_joint_binding(pack, reviews, 0),
    "groups": [["module-a-corner", "module-b-corner"]],
    "note": "Explain the original-source evidence for these physical joint aliases.",
}
result = evaluate_source_joints(frame, rgb_path, pack, reviews, aliases, frame_index=0)
```

Every original observation and role remains in the result. Aliased observations
must agree exactly in both sensors after source review; disagreement requires review,
never automatic averaging. A joint with any original check role belongs to checking
and cannot enter fitting. All raw check records, including duplicates, retain their
individual error assessment. Existing minimum unique-control counts, support hulls,
source identities and the 2-native-pixel point gate apply. A registration is exposed
only if source review is complete and both unique and all-raw check gates pass.

Wait blocks fitting by default. Explicit `diagnostic=True` can evaluate recorded
Wait coordinates, but returns no usable registration while any source decision is
pending. Not OK or missing source observations cannot be fitted through diagnostic
mode. Changed review decisions invalidate the alias binding. A numerical pass remains
development geometry; physical correspondence and independent calibration are not
automatically certified. Existing RGB transfer and its no-extrapolation support gates
remain applicable. This API does not change inspection findings, masks or temperatures.
