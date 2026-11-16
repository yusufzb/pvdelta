# Bounded dense observed-line search (research preview)

Dense scenes can exceed the line-pair budget even with fewer than 100 observed
lines in each family. Previously, a product above 250,000 skipped the entire
pass and retained only contour proposals. Local windows can recover some
geometry but do not prove complete coverage.

`NativeLineConfig` exposes two explicit resource settings:

- `max_pair_product=250_000`: unchanged default. Integer 1–2,000,000.
- `geometry_chunk_pairs=16_384`: integer 1–32,768. Intermediate geometry slices
  are bounded by this size; no whole-product repeat/tile allocation.

Pair order and geometry tests are unchanged. Chunks enumerate the complete
Cartesian product in the original order, applying identical area, side length,
aspect, convexity and angle gates. Chunk size does not alter candidates. A pass
over its total budget still stops explicitly, never processes an arbitrary
prefix. The 100-family cap remains. This bounds intermediate array size, not
total runtime.

`pair_search` reports the product, limits, completion flag and skip reason.
`search_complete` refers only to pair geometry enumeration in that pass: it
does not mean all physical panels or image edges were found. Side support,
ridge, divider, source-border, deduplication and neighbour gates still apply.

```python
from pvdelta.native_dense_search import protected_dense_completion_proposals

preview = protected_dense_completion_proposals(native_temperature_c)
assert preview['native_completion']['old_measurement_pixels_lost'] == 0
assert preview['measurement_authorised'] is False
```

This opt-in API runs the baseline and a second search with a maximum pair
product of 2,000,000 on the **same native array**. Only the pair budget changes.
Temperature thresholds, source geometry, line length/angle, coverage, gap,
neighbour and aspect requirements are preserved. An already dense baseline is
not repeated or narrowed. Labels, CAD and an assumed grid cannot enter.

Existing `complete_native_proposals` gates decide additions: observed sides,
two disjoint retained scale references including a neighbour, conflicts
withheld, no old measurement pixel touched, and old masks verified bit-for-bit.
Baseline proposal dictionaries and IDs remain unchanged. False baseline
geometry is preserved too; this does not approve semantic panel membership.
Addition counts are not physical site inventory.

Native arrays are never resized/resampled; SHA checks bind both passes to the
same grid. Record whether values are SDK measurements or apparent temperatures.
This API does not convert raw counts, compensate radiometry or infer temperature
from JPEG colours.

Results include global pass statuses/counters, alternative local window summary
and completion rejection reasons. Default metadata now includes search settings
and audit fields. Source-bound caches should refresh; complete historical JSON
headers can differ while baseline polygons, IDs and masks stay identical.

The default search budget and dashboard route are unchanged. Validate frozen
full-panel references, matched-reference gains/losses, false/ignored proposals,
pixel preservation and coordinate sensitivity on separate sources. AI drafts
are not independent acceptance, field thermometry or IEC compliance.

Research follow-ups: [actual decision tracing](native-search-tracing.md) and
[protected shorter observed fragments](native-fragment-search.md). These are
not enabled in the dashboard or automatic scans.
