# Original-source RGB/thermal control review

Use source controls to diagnose a specific original RGB/thermal pair. A numerical residual
cannot identify the correct physical frame centre. This tool shows the original observations
without a fitted prediction and preserves all declared fit/check roles.

Prepare a private pack:

```powershell
pvdelta prepare-control-review --controls .\pair-controls.json --sources .\sources.json `
  --source-root .\private-flight --sdk-lib .\sdk\tsdk-core\lib\windows\release_x64 `
  --out .\control-review
```

`pair-controls.json` is the source-bound control document/bundle described in
[local RGB transfer](local-rgb-transfer.md). `sources.json` lists the corresponding original
pairs **in the same order**:

```json
{"frames": [{"thermal": "capture_T.JPG", "rgb": "capture_V.JPG"}]}
```

An indexed pair can have different timestamp text in its thermal/RGB filenames. Supply the
actual files; do not infer correspondence by replacing a filename letter. Preparation checks
the exact source bytes, native temperature hash and both original pixel grids against the
controls. It creates unmarked native SDK previews and exact local copies of original RGBs.
This pack contains private imagery and must remain outside public release artifacts.

In Streamlit **Panel boundaries → RGB–thermal source control review**, enter `review-pack.json`.
When multiple local packs are available, **Source review dataset** switches between
them. Each dataset has a separate path widget; navigation, form state and decisions
remain bound to the selected pack identity. A supplied custom pack path is still
supported. Selecting another dataset does not import review decisions or approve
its coordinates.
Previous/Next navigates the controls. Inspect the same physical frame-band centre in each sensor;
do not substitute a hot screw, one band edge, or a cell line. Original encoded pixel orientation
is preserved, with no EXIF display rotation. X/Y fields use original source pixels, not enlarged
display pixels. The magenta cross is your current observation; save to refresh the crop.

**Source photo pair** filters to one capture pair. **Control review status** offers
All, Needs review (everything except OK), Unreviewed, Wait, Not OK and OK. Counts
always describe the whole pack; navigation uses only the visible queue. Filtering
never removes controls from export requirements. An empty queue still shows the
whole-pack export button and its actual completion state.

When one **Source photo pair** is selected, **Export selected photo pair** becomes
available once every original fit/check point in that pair has an explicit OK.
Other photo pairs may remain pending. Status filters never remove points from
this requirement. The exported one-frame bundle preserves the exact source pair,
all original IDs/roles, review provenance and original pack identity, and works
with the local RGB transfer preview. A later Wait/Not OK disables that pair's
export; an earlier immutable export stays on disk. **Export revised controls**
continues to require every control in the whole pack.

Enable **Show full source frames** to locate the saved point within each original
frame. These thumbnails preserve aspect ratio and encoded orientation; magenta
indicates the saved source observation, not a verified match or an anomaly. Local
crops clipped by a source edge are padded instead of stretched. Coordinate inputs
continue to use original pixels. **Save and next** writes one explicit decision
and advances within the filtered queue. A failed save never advances. At the end
it stays on the last point, unless the saved status removes it from the queue.

Each save records OK / Not OK / Wait, the declared reviewer, coordinates, note and time in an
immutable decision file. The latest decision for the exact pack is used. A later Wait/Not OK
reopens the control. All original controls in the chosen export scope must explicitly be OK; unresolved
checks cannot be silently omitted. The new bundle preserves IDs, roles, source identities and
original provenance. It can be supplied directly to the local RGB transfer preview.

New decisions have a monotonically increasing per-control `review_sequence` so
rapid edits sharing a Windows clock timestamp still select the newer revision.
Legacy decisions without a sequence remain readable as revision zero.

For explicitly reviewed repeated physical joints, see [source-joint evaluation](source-joints.md).
The adapter keeps all original roles and check records and assigns any shared
fit/check joint exclusively to checking. Pending diagnostics expose no usable transform.

Export does not fit a model, relax a threshold, update findings, edit the originals, authorise
temperature measurement or claim an independent calibration. Reviewing previously diagnosed
controls produces revised development data; validate the resulting mapping separately on new
source pairs. This tool does not make a local transformation valid across an entire flight.

Programmatic AI source review must explicitly set `review_basis="AI_source_visual_review"`
and name the AI reviewer. The UI identifies these observations as AI development
review; they are never represented as a human or independent calibration. Exports
containing AI observations retain that basis and use AI-assisted provenance. Wait
still blocks export, regardless of who reviewed the other controls.

The optional **Frame-band source controls** local dataset contains the unchanged
source observations from a fixed paired-camera regression comparison. Its fit/check
roles are preserved and all review states initially remain pending. It contains no
fitted predictions or automatically approved replacement coordinates. Inspect the
same physical band centre in both sensors; if the thermal frame band cannot be
identified, use Wait or Not OK rather than choosing the fitted position.
