# Flight planning dashboard

The optional Streamlit dashboard separates values known while planning from measurements
recorded during the flight. It keeps field imagery and inspection coordinates local.

## Basic and Advanced

Prepared source-bound material questions are available in both modes under **Mask material review** with direct local saving. See [material review](mask-material-review.md).

The **Interface** selector at the top opens in **Basic** by default:

An optional **Saved inspection** selector opens explicitly configured local
database/run pairs. Each analysis retains its own findings and decisions; no
scan starts or review transfers. The header distinguishes legacy location-only
records from recorded measurement settings. See [saved inspections](inspection-library.md).

- **Start a scan** asks for the local DJI photo folder. One explicit button checks
  the configured SDK, indexes the photos and runs the existing native thermal
  inspection engine. Each attempt gets a new private folder under
  `data/inspections/`; completed results open automatically. Original photos and
  existing inspections are preserved. Embedded measurement settings are used;
  parameter overrides belong in Advanced.
- **Overview** shows flight-scoped processing coverage, observation and candidate
  counts, candidate types, review progress and a downloadable candidate list.
  Missing boundaries and failed/unassessed frames remain visible.
- **Review candidates** provides thermal/RGB previews, filenames, available
  estimated panel coordinates, Previous/Next, review filters and collapsible
  OK / Not OK / Wait controls. Saved measurement profiles and native hashes are
  checked before thermal overlays. For the original compatible M4T flight,
  the saved quadratic diagnostic supplies a **labelled approximate RGB location**:
  red 6× display outline and cyan projected provisional module, with generous
  surrounding context. This visual search aid is not locally validated registration.
  Other camera/flight/image sizes, changed sources or unverifiable native grids
  remain unmarked with an explanation. Legacy analyses lacking a saved measurement
  profile can show source/grid-checked **saved candidate locations**, explicitly
  separate from historical temperature verification. An invalid existing profile
  still blocks overlays. See [review versions](review-versions.md).
  Verified registration tools remain in
  Advanced. Optional nearby pairs support manual
  comparison without claiming same-panel or glare verification.

  **Check RGB alignment** optionally compares complete RGB module boundaries with
  the stored native proposals. The selected module stays outside fitting; unused
  module checks, shared-corner exclusion and source-bound control support gate any
  correction. Failures retain the approximate preview and show a concrete reason.
  See [per-pair RGB checks](rgb-pair-check.md); the first 12-frame sample did not
  establish any acceptable local correction.

**Advanced** retains all existing flight planning, field measurements, SDK scan,
boundary, unassigned evidence, RGB quality, database/map, review, orthoview and
export tools. The selected inspection database follows the mode switch. Basic
opens current databases read-only; use Advanced for older schema migration.
Changing modes or opening results does not start a scan. Counts remain repeated
observations and candidate regions, not unique physical panels or confirmed faults.

The inspection tab can reanalyse stored provisional panel boundaries with fresh SDK
measurements into a new private database. It retains per-analysis measurement settings,
checks native temperature hashes before drawing thermal evidence, and invalidates stale
reviews and repeat-view caches when their inputs change. See
[measurement profiles and reanalysis](measurement-profiles.md) for scope and operation.

## Run

```powershell
& .\.venv\Scripts\python.exe -m pip install -e '.[dashboard]'
& .\.venv\Scripts\python.exe -m streamlit run dashboard\flight_planner.py
```

```bash
.venv/bin/python -m pip install -e '.[dashboard]'
.venv/bin/python -m streamlit run dashboard/flight_planner.py
```

## Planning inputs

The main form asks only for the camera, altitude, camera view, speed, photo interval,
coverage preset and array geometry. Module orientation uses the familiar terms:

- **Portrait:** the module's long side runs up the rack slope.
- **Landscape:** the module's short side runs up the rack slope.

The standard route assumption is stated rather than asked repeatedly: the flight follows the
table length, the native thermal image's 640 px edge follows the route and its 512 px edge
crosses the rack. Module dimensions, row gaps and asset labels are under advanced sections.

The default M10 preset records a 2.278 × 1.113 m, 144-cell half-cut module. It opens in
landscape orientation with four modules up the rack slope. The export keeps the exact
dimensions, cell count, M10 format and half-cut layout so later boundary detection can use
known geometry instead of estimating it from every image.

Site length and width are optional. The dashboard shows an image-count estimate only when
the rectangular-area estimate is enabled. The count is the number of calculated flight lines
times capture positions per line. It excludes irregular boundaries, turns, obstacles and line
overshoot, so it is a rough planning value rather than a predicted deliverable count.

Altitude is AGL relative to the ground below the rack's lower edge. For a nadir camera, that
lower edge is the conservative thermal-resolution reference. Add any difference between the
mission controller's take-off reference and this ground level separately.

## Field record

Capture settings such as gain, original R-JPEG, FFC and RTK are separate from environmental
measurements. POA irradiance, irradiance stability, wind, air temperature, humidity and cloud
cover stay blank until the operator selects **I have measurements from the flight**.

Reflected apparent temperature is an advanced radiometric value. It represents infrared
radiation from the surroundings reflected by the module glass toward the camera; it is not air
temperature. Leave it blank unless it was measured or deliberately set. One established
measurement procedure is the reflector method in the
[FLIR thermographic measurement guide](https://support.flir.com/docdownload/assets/dl/t559495-en-us%24aa.pdf):
crumple and flatten aluminium foil, place it at the target, view it from the measurement
direction and measure its apparent temperature with emissivity set to 1.0.

High gain is the normal M4T PV starting point while the entire scene remains below its upper
range. Low gain is required when clipping is possible. Gain does not replace emissivity,
distance, humidity or reflected-radiation corrections.

## Local data and map

The dashboard can create or open an ignored local DuckDB file. It stores searchable capture
metadata, temperature summaries, module observations, findings and optional geometry. Dense
native temperature matrices remain external local files referenced by path and hash; they are
not expanded into hundreds of millions of database rows.

The folder indexer reads DJI `*_T.JPG` and `*_V.JPG` files in place, stores their relative path,
hash and available EXIF/XMP fields, and pairs captures by DJI sequence number. It does not copy
or upload the original imagery.

The Folium view distinguishes the ordered thermal camera track, optional recorded laser target
points and image coverage. A camera GPS point can be shown as soon as it is parsed. A coverage
polygon is shown only after calibrated geometry has produced a footprint. Online map
backgrounds are disabled by default because requesting a tile reveals the viewed geographic
area to the provider. When enabled, the map control offers OpenStreetMap, OpenTopoMap terrain
and Google high-resolution satellite imagery. The satellite layer remains visible through zoom
22; beyond the provider's native level it enlarges the last available tile instead of leaving a
blank layer. Provider attribution remains visible. Satellite tiles show site context and do not
replace the drone's radiometric measurements.

See [Local inspection store](local-inspection-store.md) for the schema and privacy boundary.

## Radiometric scan

The **Radiometric scan** tab starts a local scan of the original `*_T.JPG` files with the DJI
Thermal SDK. Enter the SDK library folder and the flight folder, then start the scan. The UI
stores only compact native-grid summaries in the selected local DuckDB and leaves the source
images in place. `PVDELTA_DIRP_LIB` can prefill the SDK field.

The scan's thermal-frame count is not a panel count. Module proposals and measured observations
are frame-local and can repeat across overlapping captures. A unique physical panel count needs
validated tracking or optional inventory matching. GPS points, image counts and the four-row
prior are not silently converted into panel IDs or module counts.

## DuckDB charts

The Local data & map tab also provides an inspection overview: inventory panels, stored module
observations, matched Panel IDs, finding totals, anomaly-category bars, assessment-status bars,
finding ΔT/temperature details and the native radiometric min/median/max trend. Repeated
frame-local observations are labelled separately from physical panels. Categories such as
`hot_region` remain candidate labels; the chart does not claim an electrical or physical cause.

The same tab includes a data-coverage table for indexed thermal/RGB captures, thermal/RGB pairing,
camera GPS, native radiometry, temperature summaries, observations with coordinates and observations
linked to findings. **No finding (assessed)** counts only observations with an `analyzed` status and
no linked finding. Unassessed proposals are excluded; no finding is not a healthy-panel diagnosis. A warning is
shown when indexed thermal captures outnumber captures containing module observations. The finding
detail view shows the native thermal crop and its radiometric polygon together with the paired RGB
frame. When the accepted local quadratic registration is available, the same finding polygon is
projected onto a marked RGB crop; the detail table also shows the thermal/RGB filenames and the
estimated finding coordinates. The RGB overlay is a visual correspondence and is not used as a
second temperature measurement. The selected finding is also marked in the Folium map with its
filename and coordinates.

Select a **Stage 1 analysis run** before reading module counts. Statistics and findings use that
same run, so reprocessing one photograph in three experiments does not triple the displayed count.
If `flight_screening_frames` is present, **Full-flight screening coverage** shows attempted frames,
frames with measured interiors, assessment/rejection reasons and a per-frame CSV download.
The expandable frame browser displays saved thermal/RGB boundary previews with their quality status.

The **User review** tab stores a separate human decision for each finding. Select a candidate and
use **OK**, **Not OK** or **Wait**; an optional reviewer name and note are stored in the local
`finding_reviews` table. These decisions do not overwrite the detector's `assessment_status`, so
the radiometric result and the human confirmation remain auditable independently. The same controls
are also available in a collapsed **User review** section directly below the thermal anomaly crop.

The current automatic flight screening is provisional: RGB contours supply geometry candidates,
native thermal edge checks gate the eroded measurement interior, and temperatures come from the
original DJI SDK grid. It screens local hot regions using the run's recorded thresholds. Oblique
frames and rejected registrations remain explicitly unassessed. It does not establish exhaustive
module coverage, unique panel identity, or detection of every whole-module/electrical fault.

### Run-scoped coverage audit

Under **Local data & map**, expand **Coverage audit & geometry rejection reasons**.
This read-only view explains stored decisions for the selected analysis run. Its
indexed-flight denominator comes from the acquisitions referenced by that run,
not unrelated flights in the database. Unknown acquisition scope is explicit.

Frame pipeline stages and module gate combinations count each record once.
Detailed rejection reasons overlap and must not be summed. The review queue lists
observations whose legacy edges passed while shared frame registration failed;
these remain unverified and are not automatically measured. Historical edge scores
describe image consistency, not ground-truth correspondence error or IEC conformity.

The JSON download includes per-frame reasons, consistency issues, finite analyzed
observation counts and run-specific review totals. Multiple findings on one observation
are counted separately from observations with findings. Analyzed without findings is
not a confirmed healthy-panel count. Physical panel count and independent mask
verification remain unknown until supported by separate evidence. This audit does
not decode the flight again or rerun the experimental registration matcher.

### Native finding overlays and mask support

The Stage 1 thermal finding preview draws on the DJI SDK's native RGB render,
with dimensions checked against the indexed thermal grid. Native coordinates are
never scaled onto the embedded JPEG. If SDK rendering fails, an unmarked embedded
image remains available and the overlay is explicitly withheld.

Red is a 6× display outline, 20% wider than the previous 5×. A thin white ring follows the exterior of
the exact stored finding pixels, leaving those pixels visible; cyan traces the
stored provisional module polygon. These outlines do not independently verify
the panel boundary. The requested 500px crop context is preserved on the native grid.

**Native mask support / edge sensitivity** counts exact stored region pixels
retained by 0/1/2/3/5px insets of the module polygon. This UI probes polygon
sensitivity; it does not reconstruct a historical measurement mask from unknown
run settings. The core diagnostic also accepts an explicit measurement mask when
the caller has that evidence. Missing exact pixels remain unavailable rather than
being replaced with a bounding rectangle. No findings or human decisions are
deleted or changed by this diagnostic, and retained pixels do not prove absence
of vegetation or glare.

### Panel boundary comparison

Open **Panel boundaries**, choose an indexed flight and thermal frame, then press
**Compare panel boundaries**. Previous/Next navigate only that acquisition.
Alternatively choose **Local thermal file** and enter an original DJI R-JPEG
path. **Source settings** contains the inspection database and local SDK path;
the initial values come from Radiometric scan.

The comparison decodes one real SDK grid and evaluates the baseline native-window
search and the supported raw supplement on those same unchanged temperatures.
Cyan traces existing geometry; amber marks raw supplements. Candidate counts are
frame observations, not unique site panels or anomaly counts. Geometry remains
unreviewed and does not authorise measurements or change stored findings.

**Show pixel support** displays unique 2px-inset interiors with a cyan tint and
shared envelope pixels in magenta. Shared pixels remain excluded from every ROI.
Descriptive native ROI statistics and recorded SDK settings are expandable;
palette colours never supply temperatures. An empty proposal list is not a
healthy-panel result. The JSON and marked PNG downloads are local review exports.

Indexed relative paths resolve under their acquisition's recorded source root;
missing roots and paths escaping that root are withheld. Source bytes must match
the indexed SHA256 before decoding. The preview is retained only in the current
Streamlit session and invalidated by frame/source/settings/engine changes. A new
comparison runs only on request; navigating or changing display options does not
automatically process the flight. SDK/render failure displays an error without
reusing an old overlay or mapping native polygons onto an embedded JPEG.

The preview service opens DuckDB read-only and performs no schema migrations or
finding/review writes. Other existing dashboard tabs retain their own operations.
Restart the running Streamlit server after code updates to load fresh imported
modules; a browser reload alone can retain old Python callables.

### Persistent boundary review queue

**Boundary review queue** shows the latest saved visual decisions for the selected
acquisition. Filter **Needs review**, **Saved OK**, **Saved Not OK**, **Saved Wait**,
or **Other engine version**; Previous/Next use that filtered population. Needs
review includes unsaved, rejected, waiting and other-engine records. Queue states
describe saved history: compare a frame to check the exact current numerical
result, including SDK settings and native data. The queue does not decode or
revalidate every indexed file in the background.

After comparison, **Boundary review** offers OK / Not OK / Wait with an optional
reviewer and note. These decisions concern the displayed boundaries and frame
coverage, not whether a module has a fault. They are stored separately from the
anomaly User review controls. An engine, native-data, geometry or settings change
requires a decision on the new result. The current decision is looked up by the
exact result identity, even when the queue has a more recent different result.

Saving appends to DuckDB; previous notes and decisions remain in history (the UI
shows the latest 50 for that source). With Needs review selected, saving OK removes
that frame from the filtered list and advances to the next remaining frame. Its
preview must be requested separately; the accepted overlay/decision is never
copied to it. This is visual review evidence, not independent full-panel labels,
an anomaly verdict, or IEC conformity. Only explicit review buttons save decisions;
opening or comparing a frame does not create an acceptance record.

**Boundary issues (optional)** records one or more visual error reasons with
Not OK / Wait. OK saves no active issue tags and preserves previous reasons in
history. **Boundary issue filter** combines with the status filter; it selects
current-engine Not OK / Wait records for the chosen reason, or **No recorded
reason** for unclassified decisions. A frame change does not copy issue selections.
If no records match both filters, clear one filter to continue reviewing.

The queue shows issue counts for the selected acquisition's latest current-engine
Not OK / Wait events. Counts refer to indexed frame records, can overlap across
reasons, and must not be summed into a unique panel count. Older-engine tags are
retained in history and the JSON export but excluded from current correction
counts. The export includes acquisition/capture/source identities, saved result
and event IDs, reviewer, note, reason codes and engine signatures. It explicitly
marks exact current results as unchecked until a fresh comparison is made; queue
statistics do not validate source files or SDK settings in the background.

### Reference validation

After comparing a frame, expand **Reference validation (optional)** and enter a
local full-panel reference JSON path. A bundle with a `frames` list is supported;
the frame is selected by source SHA256, not filename. Press **Validate against
reference** to score baseline and proposed boundaries without decoding the SDK
again or saving any human decision. Full-frame exhaustive references are required;
sparse panel patches cannot establish full-panel precision or recall.

The table reports reference instances, matched candidates (TP), unmatched
candidates (FP), missed references (FN), ignored candidates, precision and recall
at IoU .5/.75. Two annotated images show matching errors at .75; these thresholds
are geometric diagnostics, not IEC limits. Developer drafts remain development
evidence. A declared independently-reviewed label status is metadata and is not
certified by the dashboard.

Reference file changes, a different preview result or scorer code changes remove
old validation results. Missing/malformed/wrong-source files display an error;
no previous scores or error overlays are reused. The selected bundle path survives
frame navigation so multi-frame references can be reviewed with Previous/Next.
Each new result still requires the explicit validation button. Downloads include
reference/result/scorer provenance and a native-size proposed error overlay.

### Unassigned evidence

The **Unassigned evidence** tab shows a separate frozen native thermal review
bundle with 500px context, exact hot-pixel outlines, source filenames, explicit
radiometric basis and Previous/Next navigation. OK / Not OK / Wait decisions are
saved in a separate append-only local log, never added to main findings, module
inventory or healthy counts. Missing or changed source/native data withholds
the overlay. This is optional review of unresolved boundaries, not an automatic
flight scan or confirmed panel fault. See [bundle format and scope](native-gap-review.md).

**Nearby thermal frames — unverified comparison** optionally displays the marked
source beside an unmarked comparison, with its own Previous/Next navigation,
filename, measurement basis and observed-panel control failure. These frames
have no accepted same-panel transfer or angle separation; no anomaly outline
is copied onto them. Corrupt or stale context does not hide valid primary evidence.


## Panel review versions

Basic results default to **Panel cards (v2)**, with **Individual anomalies (v1)** available in the Review version selector. Advanced adds a separate **Panel review (v2)** tab. See [grouping, review scope and checkpoint behavior](review-versions.md).
