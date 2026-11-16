# Frozen thermal-region development comparison

## Polygon protocol v2

From 2026-10-29 the editor supports simple polygons with inward corners. A separate
`evaluation-protocol-v2.json` uses native pixel-centre mask intersection/union for
IoU. Concave notches are preserved rather than filled by a convex hull or bounding
rectangle. The original v1 protocol and continuous convex scores remain intact.
Streamlit prefers v2 when present. Freeze v2 explicitly with
`freeze_region_protocol(pack, predictions, version=2)`; default v1 remains backward
compatible. Existing four-corner exports work with either geometry, but raster
and continuous IoU may differ. Compare only reports with the same protocol hash.
The thresholds, pending/uncertain rules and matching policy below remain the same.

`freeze_region_protocol(pack, predictions)` in `pvdelta.region_reference_benchmark`
binds one native reference pack to previously frozen region predictions. It consumes
no reference labels. The separate protocol has a content checksum, selected source
population, original prediction-document hash and explicit rules. Keep the protocol
and its checkpoint before collecting region references. Checksums detect accidental
changes; they do not prove when a file was created or who approved it.

Version 1 rules, fixed before human region review:

- Primary: native convex extent IoU >= 0.5; auxiliary: IoU >= 0.1.
- One-to-one maximum-cardinality matching, using the existing hotspot benchmark.
- Match positives first. For unmatched predictions only, exclude those with at
  least 90% of their rasterised extent in explicitly uncertain regions.
- Include every completed full-frame reference, even if the detector proposed no
  panels or no findings. A missing mask cannot remove a missed thermal reference.
- Incomplete frames remain pending with their prediction counts visible. They
  contribute neither FP nor FN. With zero completed frames, totals are null.
- A completed zero-region frame requires reviewer and empty-frame reason, and
  can contribute unmatched predictions. Undefined precision/recall remain null.

These thresholds are project geometry diagnostics, not IEC limits. Review quads
and detector extent polygons are compared as convex extents, not exact temperature
component masks. Broad annotations can produce low IoU despite a small component
inside; inspect both fixed outputs and missed IDs instead of selecting the most
favourable threshold. No thermal severity, fault cause or healthy-panel count
follows from these scores.

`evaluate_region_work(pack, editor_export, protocol)` reuses the source/grid/image
and manual-geometry validator. The separately supplied protocol authorises this
specific development comparison; the editor export still grants no independent
review or measurement authorisation. Duplicate predictions cannot inflate TP.
Reference labels and the operational inspection database are never modified.

In **Advanced → Panel boundaries → Thermal reference review**, load the pack and
upload the editor's raw JSON export. With `evaluation-protocol.json` beside the
pack, select **Compare with frozen baseline**. The JSON download lists all frames,
pending counts, matched IDs, missed references, unmatched and ignored predictions,
and reference/protocol hashes. This compares the frozen baseline, not a later
detector implementation. Human declarations still require independent checking
before acceptance; partial development scores cannot establish full-flight accuracy.
