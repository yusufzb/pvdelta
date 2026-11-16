# Full-panel instance evaluation

`pvdelta.instance_benchmark.evaluate_panel_instances` evaluates complete native
panel envelopes. Sparse patches remain under `evaluate_labeled_support`; they
cannot establish full-panel recall or precision.

Reference fields: `schema: "pvdelta.native-panel-instances/v1"`,
`coverage: "exhaustive_full_frame"`, exact `source_sha256`,
`coordinate_space: "thermal_native"`, native width/height, and `review_status`
(`developer_draft` or `independently_reviewed`). An explicit `panels` list is
mandatory, even when empty. Entries have unique `instance_id` and convex
`polygon_native_xy`. Optional `ignore` regions have unique `region_id` and convex
native polygons, for example partial panels at the source boundary. They must
not overlap full reference instances. Partition complex regions explicitly.

Predictions have unique `proposal_id` and convex `polygon_native_xy`. Vertices
must be finite and in the source grid. Invalid identity, grid, coverage and
overlapping reference contracts raise errors.

Matching uses continuous convex-polygon IoU and maximum-cardinality one-to-one
matching above the requested threshold. IoU orders traversal; this does not claim
maximum total IoU. A duplicate cannot count as another panel, and a merged
envelope cannot count as several panels. Fixed .5/.75 thresholds in the local
experiment are development metrics, not IEC acceptance limits.

Unmatched predictions are false positives unless at least `ignore_fraction`
(default .9) of their native raster pixels lie in explicit ignore regions.
Unmatched full references are false negatives. Ignore coverage uses pixel centres,
separately from polygon IoU. Envelopes are scored before measurement erosion and
shared-pixel exclusion, so measurement masks cannot conceal incorrect geometry.

Outputs include TP/FP/FN, precision/recall, matched IDs/IoUs, missed IDs, false
positive IDs, ignored IDs and reference review status. Undefined ratios are `None`.
No true negatives, healthy panels, physical site inventory, anomaly accuracy,
IEC conformity or measurement approval are inferred.

Freeze reference annotations and extractor settings before viewing predictions.
Developer labels remain drafts regardless of scores. Reference review,
representative positive/negative scenes and separate-flight validation remain
required for acceptance.

`pvdelta.instance_diagnostics.diagnose_panel_instances` adds native-pixel
coverage and merge/split/duplicate-like overlap patterns without changing
matching or review status. See [instance diagnostics](instance-diagnostics.md).
Unresolved panel identity remains unscored; it must not be resolved by assuming
every roof rectangle is a panel or by importing detector proposals as truth.

## Local before/after comparison

`pvdelta.boundary_validation.compare_boundary_reference(preview, reference_bytes)`
scores the existing native boundary preview at fixed IoU .5/.75. It performs no
SDK decode, label edits, database writes or review acceptance. Input can be one
reference object or a `frames` bundle. Bundles must contain exactly one entry
matching the preview's source SHA256; filenames and array position are not keys.
Source identity, native grid and exhaustive reference contracts remain mandatory.

The output binds the exact boundary result ID, native-temperature hash, raw
reference file hash, canonical selected-reference hash and scorer source hashes.
The supplied review status is recorded as *declared*, not independently verified.
Changing labels, source/result or scorer invalidates the comparison cache.

Both baseline and proposed overlays use IoU .75: green TP for matched candidates,
amber FP for unmatched candidates, dashed red FN for missed reference instances,
and grey IGN for candidates covered by explicit ignore regions. Undefined ratios
remain null. These colours describe geometry matching, not anomaly severity.
JSON scores and a proposed error PNG can be exported locally.

## Acquisition batch command

Use the same scorer on all explicitly supplied references for one indexed
acquisition, without changing the inspection database:

```powershell
pvdelta validate-boundaries --database data/inspection.duckdb `
  --acquisition flight-id --references private/full-panel-references.json `
  --out reports/boundary-validation-001 --sdk-lib sdk/path/to/native-library
```

Install the storage, OpenCV and Pillow dependencies (available together through
`pip install -e ".[dashboard,vision]"`) and configure the local DJI Thermal SDK.
The `--out` directory must be new; existing results are never overwritten and
there is no implicit resume. Outputs are private derived data: `summary.json`,
`summary.html`, and source-SHA folders containing the exact preview/comparison
JSON and native-size baseline/proposed error images. No images or dense arrays
are inserted into DuckDB, and no review decision or anomaly analysis is saved.

Every reference is checked before decoding begins. Source SHA256 matches must
identify exactly one indexed thermal capture in the selected acquisition; a
missing/ambiguous index entry, unavailable path, changed source or SDK failure
remains an explicit failed frame. Same-source records in other acquisitions do
not select or duplicate the frame. All requested references remain in the report.

Summary counts distinguish requested/scored reference instances, successful,
failed and unprocessed frame records, and indexed acquisition records without
supplied references. `complete` means all **supplied reference frames** succeeded;
it never means all indexed frames or all physical panels were validated. Aggregate
TP/FP/FN and micro precision/recall use successful frames only. When no frames
succeed, scores are null; failure does not become zero-error or healthy evidence.

JSON and HTML checkpoints are replaced after each frame. If reference bytes or
extractor/scorer/runner files change while running, the batch is `invalidated`
and aggregate scores are withheld. Per-frame artifacts retain their identities.
Output I/O failures stop the command and leave the last checkpoint incomplete.
Exit codes: `0` complete, `3` partial, `1` failed/invalidated/setup error.

Reference `review_status` remains a supplied declaration. The command does not
certify label independence, detect data leakage, infer unique site inventory or
establish anomaly accuracy/IEC conformity. Freeze and review new flight labels
separately before using them for acceptance.

## Prepare unmarked annotation images

When full-panel references are missing, prepare a private review pack before
scoring or tuning against predictions:

```powershell
pvdelta prepare-boundary-review --database data/inspection.duckdb `
  --acquisition flight-id --count 12 --seed review-v1 `
  --exclude private/previous-reference-bundle.json `
  --out reports/annotation-pack-001 --sdk-lib sdk/path/to/native-library
```

`--exclude` may repeat and accepts one reference object, a `frames` bundle or a
JSON list of lowercase source SHA256 strings. Exclusions remove exact source
bytes. They do not prove the absence of overlapping scenes or earlier use of
other frames from the same site/flight. A new output directory is required.

Selection is frozen before SDK rendering. It sorts eligible captures by recorded
timezone-aware capture time, divides them into chronological quantile bins, and
chooses the minimum seeded source-hash rank within each bin. Missing/invalid/
timezone-unknown times use a separate pool, receiving a slot when the budget
allows. Selection does not use detector output. Ambiguous source hashes,
unavailable acquisition-root paths and excluded identities are counted explicitly.
Requested/eligible/selected counts and shortfalls remain visible.

The pack contains `selection.json` with camera metadata and exclusions, unmarked
SDK `native.png` images with native-temperature hash/settings in `native.json`,
checkpointed `progress.json`, `summary.html`, and `draft-references.json`.
Camera GPS describes the aircraft, not panel centres. No detector is run, no
panel count/anomaly verdict is inferred, and no inspection DB/review is written.
Source bytes are checked before/after rendering and the SDK render must share
the finite native temperature grid. Exclusion/code changes invalidate the pack.

Templates deliberately start with `coverage: "not_annotated"`,
`review_status: "unreviewed"`, and `panels: null`, `ignore: null`. Existing
validators reject them. An empty template must never masquerade as an exhaustive
negative frame. An annotator must explicitly label every full panel and ignore
region, then set the matching coverage/review status. A genuinely negative frame
requires an explicit reviewed empty panel list. Developer labels remain drafts;
neither the tool nor selecting new hashes certifies independent validation.
