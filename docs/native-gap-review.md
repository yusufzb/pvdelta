# Unassigned native thermal evidence (research review only)

`NativeGapReview` is a bounded sink for `trace_native_search`. It retains
native hot evidence when observed panel geometry fails its four-side gate.
The result is a **separate review queue**, not accepted panel geometry,
primary findings, a physical inventory or a healthy-module count.

```python
from pvdelta.native_gap_review import NativeGapReview
from pvdelta.native_lines import NativeLineConfig
from pvdelta.native_search_trace import trace_native_search
from pvdelta.native_windows import native_window_proposals

# Caller verifies that the baseline geometry belongs to this exact frame.
review = NativeGapReview(frame, baseline_geometry)
with trace_native_search(frame.temperature_c, review):
    unchanged_geometry = native_window_proposals(
        frame.temperature_c,
        line_config=NativeLineConfig(min_segment_px=16, max_pair_product=2_000_000),
        raw_supplement=True,
    )
review_queue = review.evaluate()
```

Only actual hybrid side-support rejections are considered. At least two sides
must already satisfy coverage ≥0.8 / gap ≤0.18. Two disjoint, observed,
scale-compatible baseline neighbours must border the gap along **different
module axes**. Neighbours along only one axis are insufficient. No rack cell is
extrapolated and reference polygons, RGB or CAD are not inputs. Tiny floating
roundoff in analytic support ranges is clipped within 1e-9; material invalid
coverage and contradictory side-status values fail explicitly.

Hypotheses are ranked by geometry, not temperature. Defaults retain at most
512 hypotheses, deduplicate near-identical quadrilaterals and analyze at most
64. Budget evictions and withheld analyses are explicit. The collector only
sees emitted candidates: consult the upstream geometry audit for skipped line
passes. A complete queue is not exhaustive flight coverage.

The existing native detector examines provisional interiors with a two-pixel
margin and excludes **all baseline envelopes**. Every source temperature stays
unchanged. Exact component pixels, native statistics, source filename/hash,
geometry rejection and neighbour IDs accompany each item. Similar component
supports are deduplicated without merging or enlarging their pixels. Evidence
IDs use source hash and exact pixel support. Native changes abort the review.

Each item is `unassigned_thermal_evidence_review_required`. Panel identity and
panel GPS remain null; confirmed-fault and measurement approval stay false.
The stored finding has only an internal gap hypothesis ID, not a physical
panel ID. A warm candidate can still be ground, reflection or uncertain panel
geometry. Zero queue items cannot establish health. Do not combine this queue
with module/anomaly statistics or use its provisional mask as accepted output.

EXP-094 explored four adjacent XT2 filenames. Registration did not establish
reliable panel correspondence, so no boundary or temperature was transferred
and no glare evidence was accepted. The review queue uses the target native
array only. External XT2 values remain FLIR apparent blackbody temperatures,
not environmentally corrected surface temperatures. No IEC acceptance.

The collector is not enabled in automatic scans. Streamlit's **Unassigned
evidence** tab can review an explicitly exported, frozen local bundle. Independent
review and fixed positive/negative flight evaluation remain required. See
[decision tracing](native-search-tracing.md) and
[native localization](native-localization.md).

## Local review bundle

```python
from pvdelta.gap_review_bundle import write_gap_bundle

# Native array must match review_queue's native-temperature SHA256. The raw
# source must still exist and match its SHA256. Include effective parameters
# and an honest basis (e.g. FFF apparent blackbody, not surface temperature).
write_gap_bundle(
    "data/native-gap-review",
    [(review_queue, frame.temperature_c, frame.source_path, basis, parameters)],
    engine_source_sha256=frozen_engine_hashes,
)
```

Existing directories are never overwritten. A bundle contains an integrity-bound
report, exact `.npy` native data, raw-source path/hash, engine identities and
measurement parameters. All are private local artifacts under ignored `data/`.
The dashboard defaults to `data/native-gap-review`; use the **Review source**
path field or `PVDELTA_GAP_REVIEW_BUNDLE` to select a different bundle.

Each rerun rechecks the native grid and the current raw source; a missing or
changed source withholds overlays rather than reusing stale geometry. This
verifies artifact consistency, not scientific accuracy or an authenticated
signature. The screen renders native data with a frame-local percentile scale,
an exterior white pixel ring, an amber unresolved panel hypothesis and a 5× red
display outline. Native values and measured support are never expanded or
resampled. The 500×500 crop shifts within the source frame at its edges.

Previous/Next, status filtering and **OK / Not OK / Wait** apply to individual
evidence records. Decisions append to `reviews.sqlite3` inside the bundle;
they bind to source, native grid, parameters, frozen engine and full result.
No inspection database is written. OK means retain useful evidence for follow-up,
never a confirmed module, fault, glare clearance or IEC assessment. Changing the
whole result requires a new review even when the hot-pixel evidence ID is equal.
Multiple records may refer to one panel. Zero evidence is not a healthy count.
Unavailable RGB registration, panel GPS and same-panel multi-angle evidence
remain explicitly unavailable. Viewing does not decode a new flight or rerun
the collector. A new scan must export a new bundle.

## Nearby native comparison frames

An optional source-bound context bundle adds **Nearby thermal frames — unverified
comparison** below the native preview. The selected source keeps its exact
markings; comparison frames remain unmarked. Previous/Next switches comparison
frames independently of the evidence selector. A nearby filename is not proof of
same-panel identity, capture sequence, overlap or viewing-angle diversity.

`observed_panel_pair_check` checks a coarse cross-frame initializer against
observed native panel geometries. It keeps mutual one-to-one unambiguous matches
(initial IoU ≥.5, margin ≥.25, centre distance ≤12px), requires at least8 pairs,
and fits an affine model with four whole-module-held-out folds. All chosen pairs
are checked; residual outliers are not removed to force a pass. Degenerate
one-axis controls fail. The exploratory2px P95 gate is a project diagnostic,
not an IEC threshold. A good global control check still does not prove the
selected panel's identity or local support, so this API never authorises
transferred polygons, measurements or glare clearance. It sees geometry only,
not hot regions, temperatures, CAD or reference annotations.

`write_gap_context(directory, parent_record_id, views,
engine_source_sha256=hashes)` persists explicit source/hash, native/hash,
radiometric basis, parameters and control reports in a new `context/<record_id>/`
subdirectory. It never overwrites context or changes the parent manifest/reviews.
`load_gap_context` rechecks the parent, each native array and current raw source.
Missing/changed context withholds only comparison images; valid primary evidence
stays visible. No nearby frame means unknown, not absent anomaly or a healthy
panel. Context counts are not assessed same-panel views. Display scales are
per frame and cannot establish thermal persistence by colour.

## Native boundary diagnostic

Expand **Panel boundary support — diagnostic** and press **Inspect native
boundary evidence**. This checks the four unchanged provisional edges on the
verified native grid. Cyan marks observed narrow-ridge samples; magenta marks
unresolved samples. The table shows native ridge coverage/gap and, where
available, the original hybrid proposal coverage. Hybrid support can include
observed line segments, so the two channels are not interchangeable.

Project diagnostic limits remain .8 coverage/.18 gap, with .15K minimum
contrast and3× estimated along-edge variation. The variation is not camera
NETD, and none of these project gates is an IEC acceptance criterion. A
one-sided exterior step can lack a narrow ridge; failed ridge support means
unresolved evidence, not proof of no panel or no anomaly. Conversely, four
supported ridges do not establish semantic panel membership.

The diagnostic generates no mask, moves no edge, changes no hot-pixel support
and writes no inspection/review record. Interpolated samples serve geometry
only and are never reported as panel temperatures. JSON export binds source,
native bytes, geometry, limits and engine identities. A changed native result,
geometry or audit engine invalidates the cached overlay; changing evidence
does not copy the prior diagnostic onto another component.
