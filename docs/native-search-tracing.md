# Native search decision tracing

`trace_native_search(native_temperature, sink)` streams actual decisions made by
the existing line-search implementation. It is opt-in and does not change
proposal selection, thresholds, measurement masks or default configuration.

```python
from pvdelta.native_dense_search import protected_dense_completion_proposals
from pvdelta.native_search_trace import trace_native_search

# A finite C-contiguous 2D native temperature array, not a JPEG palette.
with trace_native_search(temperature_c, bounded_diagnostic_sink):
    geometry = protected_dense_completion_proposals(temperature_c)
```

The callable sink receives detached dictionaries. `pass_begin` identifies the
native window, support mode, contrast source and pair budget. `candidate` contains
the full-source native polygon, actual gate and four side-support tuples
`[passes, coverage, largest_gap_fraction]` where available. Stages include hybrid
anchors, side support, internal divider, source border, four-side pass,
deduplication and neighbour rejection. `pass_end` supplies the actual counters and
pair-search status, including explicit budget skips. Candidate gate counts can
be checked against those counters. Multiple events can describe one polygon.

`geometry_batch` adds bounded pre-support line-pair counters: total intersections,
geometry passes, and per-gate failures (area, minimum side, aspect, convexity,
corner angle). `first_failure` partitions rejections in gate order; `failed`
counts overlap; `sole_failure` means all other **early geometry** gates passed.
It does not imply that relaxing that gate would pass the later four-side,
divider, neighbour or composition checks. Batches never retain all pair polygons.

`contour_candidate` records area-admissible closed contours approximated by four
corners, with native source coordinates, Canny threshold, closing flag, measured
shape metrics, first rejection stage and **all** remaining geometry failures.
`contour_summary` accounts for discarded area/non-quadrilateral observations,
votes, compound rejection and retained groups. These events are emitted by line
passes; a standalone `native_panel_proposals` call is not traced. Repeated
thresholds/windows are repeated observations, not independent panels. The sink
must distinguish these event types from supported line candidates.

The extractor never receives reference labels. Compare recorded decisions with
references afterwards. A matching rejected candidate does not establish that it
is the sole cause of a missed panel. Aggregate pair counters cannot locate a
particular missing panel. No overlapping contour leaves line generation and
earlier area/non-quadrilateral contour rejection unresolved. A four-side pass is
not final selection; later window composition and protected completion can still
withhold it. This API traces line passes, not every later selection function.

Sinks must bound their own storage or stream to disk. Exceptions abort the audit
rather than silently discarding evidence. Nested contexts, resampled/copied
search windows, invalid roots and source mutation are rejected. Tracing is not
thread/process propagation: install a context separately in each worker.

EXP-092 additionally used private CPython branch/return tracing for later
composition decisions; those local diagnostics are not a dashboard feature.
