# Smaller native search windows (research only)

`local_native_alternatives(temperature_c)` searches nine half-size windows with
50% overlap in each axis. Each window estimates its own observed line families;
the default four larger windows cover two thirds of the source dimension.
This can isolate differently oriented racks, but can also lose context, truncate
large modules or expose partial/false contours. Efficacy is not accepted.

```python
from pvdelta.native_local_search import local_native_alternatives
from pvdelta.native_lines import NativeLineConfig
from pvdelta.native_completion import complete_native_proposals

alternative = local_native_alternatives(
    native_temperature_c,
    line_config=NativeLineConfig(min_segment_px=16, max_pair_product=250_000),
)
# Verify raw-source identity and the baseline's exact native temperature hash
# against alternative['native_temperature_sha256'] before composing.
completed = complete_native_proposals(
    baseline, alternative, native_shape=native_temperature_c.shape, margin_px=2,
)
```

Native source slices are used directly. Temperature values are never resized,
ranked or corrected here. Existing geometry, side coverage/gap, divider and
neighbour tests remain unchanged. Near duplicate windows do not supply extra
neighbours: candidates are deduplicated by IoU > .65, preferring greatest native
clearance from their own window border, then neighbour support is checked again.
These are multiple observations of one frame, not multiple flight angles.

The function does **not** approve measurements or alter dashboard defaults.
Protected completion separately requires two baseline scale references (one
adjacent), no old measurement-pixel contact, and no conflicting new envelopes.
Consequently an entirely new rack with no compatible baseline evidence can
remain unresolved. Geometry is not a semantic vegetation classifier.

Small grids and windows with insufficient structural percentile range are
explicitly unsearched. Pair-budget failures remain in window diagnostics;
zero proposals never establishes a negative/healthy observation. Unexpected
search errors propagate. Native mutation fails the whole call.

Assess full-panel localisation and extra unmatched proposals as well as hotspot
box coverage on fixed source cohorts. Published hotspot boxes cannot be used as
full-panel reference masks. Developer draft references are not independent human
validation; unlabelled sources remain unscored. This feature supplies neither a
physical inventory, panel GPS identity, glare conclusion nor IEC acceptance.
