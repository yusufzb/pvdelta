# Native closed level regions (research)

`level_region_candidates` searches observed connected components at 19 fixed
temperature levels in each of five native windows, both polarities. The levels
are evenly spaced inside each window's 2–98 percentile range. A region must
have a full in-window boundary, a convex four-corner hull and at least 90%
actual component occupancy. Holes count against occupancy. Two adjacent,
distinct levels must support the representative at IoU >= 0.8.

This differs from narrow ridge/step fitting: it uses closed region membership,
not inferred edges, source grid extrapolation or reference coordinates. Adjacent
levels are correlated evidence, not independent views. Regular roofs or warm
rectangles may also pass. Region persistence alone cannot establish PV semantics.

The optional completion helper recognises this evidence separately from line
segments. Its existing two baseline scale references, one adjacent, overlap,
conflict and unchanged-old-support rules still apply. A small hotspot cannot
substitute for a full panel merely because it is rectangular. Conflicts and
budget overflow are withheld rather than ranked by temperature.

The native temperature array is unchanged. Derived binary masks are geometry
inputs only; downstream temperatures must be measured on original SDK/native
pixels. Source/native identity must be checked by the caller. This research
pool does not replace the default extractor, approve masks, clear glare, count
unique physical panels or establish IEC field compliance.
