# Thermal reference review

In **Advanced → Panel boundaries → Thermal reference review**, select **Load
thermal reference pack**, then open the editor or download
its standalone HTML. `PVDELTA_THERMAL_REFERENCE_PACK` overrides the suggested
local development pack. No images, annotations or model weights ship with pvdelta.

Review all visible modules in each full native thermal frame. Add three or more
vertices around each thermal pattern that needs investigation, including missed
patterns outside proposed panel boundaries. Several regions may belong to the same
module. Use **Uncertain / exclude from scoring** for ambiguous regions; do not
force reflection, supports or unclear locations into a positive/negative decision.
Polygons can have inward corners. Click **Finish polygon** or the first point to
close; four clicks no longer close automatically. Polygons are review extents,
not confirmed faults. Non-overlapping
regions are required; use one region for a connected pattern.

The expandable RGB view provides the paired full-frame context and its filename.
It is not registered; no thermal marker is projected into it. Neither view shows
detector predictions. Previous/Next navigates the fixed pack. Display zoom never
changes exported native coordinates or source measurements.
Crossed/self-touching polygons and repeated vertices are rejected. Drag existing
vertices to correct them. Old four-corner JSON exports remain compatible. Export
open work before refreshing an updated standalone editor.

Export JSON before leaving the page or changing Streamlit controls. Resume the
JSON in the editor; upload it in Streamlit and **Save thermal reference version**.
The local version is content-addressed and previous versions are retained. Source,
native matrix, display and optional RGB hashes bind the review to this pack; a
changed image or manifest invalidates the previous export. No inspection DB writes.

Incomplete frames retain `regions:null`; an untouched image is not a negative.
Declare full review only after checking the entire frame. Completion needs the
actual reviewer's name; a zero-region frame also needs a reason. Ambiguous areas
remain explicit ignores. Imported independent-review/fault/measurement claims are
discarded. Completion remains a development declaration, not verified independence.

Region scoring requires a separately frozen protocol: see
[thermal region evaluation](thermal-region-evaluation.md). The existing panel-instance scorer must not
be used to report anomaly accuracy. Simple-polygon geometry validation checks
bounds, self-intersections, native raster support and overlaps. Review labels do not establish field fault
cause, glare clearance, temperature accuracy or IEC acceptance.
