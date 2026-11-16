# Native frame localization (research only)

`localize_native_alternatives(native_temperature, baseline, alternative)` locates
the native thermal ridge near an already observed provisional quadrilateral.
Both geometry inputs must belong to that exact native source/grid; the caller
must verify provenance, as with `complete_native_proposals`. No RGB, CAD,
reference annotations, fixed rack pitch or assumed image orientation enters it.

Each edge is sampled in a bounded ±4 native-pixel search with ±2/3-pixel
opposing flanks. Four separated along-edge strips provide robust profiles.
At least three must agree on ridge polarity and position. Broad/multiple or
search-boundary peaks, weak evidence and unavailable source flanks stay
unresolved. All four edges must pass. After line intersections, geometry limits
and fresh full-edge ridge coverage ≥0.8 / maximum gap ≤0.18 are checked again.
Geometry sampling does not change native temperatures or measurement values.

If any check fails, retain the original alternative. If movement would overlap
the protected baseline envelope or touch any baseline measurement pixel, retain
the original. Remaining proposals pass the unchanged completion/scale/neighbour
gates; old baseline dictionaries, IDs and partitioned masks stay protected.
Refined provisional additions can have different measurement pixels and must
be analyzed again. Their previous temperature statistics must not be reused.

```python
from pvdelta.native_localization import localize_native_alternatives

# baseline / alternative were obtained on this exact native grid.
geometry = localize_native_alternatives(temperature_c, baseline, alternative)
```

EXP-093 improved three development envelope IoUs against AI developer drafts
from 0.835/0.759/0.769 to 0.929/0.880/0.891. Nominal and ±1px full-instance
match counts did not improve. The original 0.749398 tight-IoU mismatch remained
unchanged because all refitted edges did not pass the support check. Neither
the reference nor the 0.75 evaluation threshold was altered to force success.
Two frozen controls showed no localization gain. The earlier 16px-fragment
search recovered three panel drafts in the new XT2 frame, without additional
unmatched geometry relative to dense search. These are frame-local developer
drafts, not independent human ground truth or physical site inventory.

More importantly, a visibly warm region in that new XT2 source lay entirely
outside the automatic envelopes. A **posthoc reference-only diagnostic** on
the full-panel draft produced a 37-pixel hot candidate around +3.015K; the
automatic geometry covered none of its pixels. That diagnostic is not an
automatic mask or a reviewed real defect label. FLIR values remain apparent
blackbody temperatures. The missed geometry means zero hot findings cannot be
treated as successful anomaly detection or a healthy observation.

This module is not enabled in Streamlit or automatic scans. Independent panel
review, geometry coverage and real positive-flight fault evaluation remain open;
there is no surface-temperature, glare or IEC acceptance claim.

See [decision tracing](native-search-tracing.md),
[short fragments](native-fragment-search.md) and
[protected completion](native-completion.md).

[Unassigned native evidence review](native-gap-review.md) can preserve a warm
component when a boundary stays unresolved. It does not approve that boundary
or add a physical panel/fault to the scan totals.
