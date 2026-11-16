# External hotspot-region evaluation

`pvdelta.hotspot_benchmark` evaluates hotspot region labels. These labels must
remain separate from full-panel polygons used by the boundary benchmark.

`parse_yolo_hotspots(text, native_shape=(height, width))` reads class-0 YOLO
centre/size boxes and converts them to native coordinates. The caller must first
verify that the labelled image dimensions AND spatial coordinates match the
native temperature grid. Matching dimensions or label/source JPEG pixels alone
is insufficient: embedded raw data can be shifted relative to the preview.
See [preview/native geometry checks](preview-alignment.md). Preserve originals,
map references only after a passing check, and withhold unresolved frames with
their requested label counts. Do not silently apply identity, clip or drop
references. Empty
text means an existing, explicitly empty published label file; a missing label
must raise an input error. Nonfinite, unknown-class, invalid or out-of-frame boxes
are rejected. No panel geometry or electrical fault type is inferred.

`evaluate_hotspot_regions(references, predictions, native_shape=..., iou_threshold=.5)`
uses unique `region_id` values and finite convex `polygon_native_xy` values.
Maximum-cardinality one-to-one matching prevents duplicate candidates from
inflating the number of matches. Unmatched predictions and supplied reference
boxes remain separate. Undefined precision/recall are null.

An end-to-end pilot should freeze source/label hashes, sample selection, code,
boundary settings and detection thresholds before inspecting predictions. Run
geometry extraction and detection without reference coordinates; load reference
boxes only afterwards for scoring. Missing module coverage must not erase a
published hotspot from the evaluated reference population. Failed SDK frames
remain separately visible with requested/scored label counts.

Report provisional module coverage and not-assessed modules alongside region
scores. Candidate display boxes are bounding rectangles of measured components,
not segmentation truth. IoU cutoffs are geometric diagnostics, not thermal
severity or IEC limits. An unmatched candidate is a benchmark false positive
against the supplied annotations; it is not proof that the underlying module is
electrically healthy. Empty published labels likewise do not establish health.

Native SDK decoding demonstrates radiometric readability, not independent
absolute-temperature calibration. Preserve the SDK's effective emissivity,
distance, humidity, reflected and ambient temperature parameters. Field
conditions, absolute calibration, glare, module identity and panel GPS require
their own evidence. Do not infer temperatures from rendered palette colours or
switch unsupported formats to an image-derived temperature estimate.

Public datasets and downloaded originals remain local. Dataset metadata can
describe a format that the installed decoder rejects; probe the actual files
before claiming SDK support. Separate RGB images, zero-byte placeholders and
inconsistent file counts explicitly from radiometric success.

`pvdelta.hotspot_coverage.diagnose_hotspot_coverage` adds post-prediction
diagnostics. Pass the references, frozen candidates and actual proposed module
envelopes as region dictionaries on the same native grid. It reconstructs the
measurement supports using the analysis partition rule and the explicit erosion
margin. A miss remains in the score even when its panel was never proposed.

The report separates absent envelope coverage, envelope pixels excluded from
measurement, measurement coverage without candidate overlap, low-IoU overlap,
and one-to-one matching competition. Native reference pixel counts, measurement
fractions and distance to the source frame edge remain visible per reference.
This helps distinguish an internal coverage gap from a cropped frame edge.
An overlapping box is not proof that the labelled hot pixels were measured;
these fractions are diagnostics, not semantic panel validation or fault causes.
Subpixel labels containing no native pixel centres fail explicitly. Empty labels
do not create a healthy-module count, and shared envelope pixels are never
assigned to an arbitrary panel.

If an evaluated test group informs a geometry or threshold change, mark it as
development feedback. Freeze another group before the next comparison. Excluding
filename prefixes and exact source hashes reduces direct reuse, but does not
establish independent scenes, module identity or a new flight. Report both match
gains and losses as well as unmatched candidates; more proposals alone are not
an improvement. A broader aspect-ratio filter must remain an experiment until
its coverage and false-candidate tradeoffs have been checked.
