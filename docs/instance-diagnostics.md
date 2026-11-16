# Full-panel geometry diagnostics

`pvdelta.instance_diagnostics.diagnose_panel_instances(reference, proposals,
source_sha256=..., native_shape=...)` adds post-prediction overlap diagnostics
to `evaluate_panel_instances`. The same native identity, exhaustive reference,
review status, convexity, frame bounds and non-overlapping label/ignore
contracts apply. Subpixel polygons with no thermal pixel centres fail explicitly.
Unannotated templates cannot be scored as empty negative frames.

Inputs and existing one-to-one matching are preserved. The result includes the
original score, native pixel coverage, ignore area and deterministic per-ID
diagnostics. Defaults use IoU 0.75 and ignore fraction 0.9; these are geometric
development settings, not IEC limits.

Fixed overlap patterns:

- **Merge-like proposal:** covers at least 80% of each of two or more full
  reference instances. It still cannot count as several correct panels.
- **Duplicate-like reference:** at least two proposals each cover at least 80%
  of the reference and have at least 80% of their own pixels inside it.
- **Split-like reference:** at least two fragments each cover 10%–less than 80%
  of the reference, each has at least 80% purity, their union covers at least
  80% of the reference, and repeated overlap is at most 10% of reference area.
  This avoids calling overlapping near-duplicates a disjoint split.
- **Absent native coverage:** no proposed raster pixel enters a full reference.
  Partial coverage and the fraction covered by any proposal remain separate.
- **Unmatched mostly outside reference and ignore:** at least 90% of an
  unmatched proposal lies outside both declared regions. This is an overlap
  description, not a vegetation/ground classifier or proof of a healthy module.

Patterns are non-exclusive geometric descriptions under the supplied labels;
they are not established causes or independent semantic validation. Native
raster fractions differ from continuous polygon IoU used by matching. Ignore
regions remain visible with their fraction of the complete frame; they must not
silently conceal unresolved panel identity or be selected after seeing errors.

AI/developer annotations stay `developer_draft`, even after a high score. If a
thermal-only scene cannot distinguish panels from roof objects or establish
full visibility, keep it `not_annotated` / `unreviewed` and withhold the entire
frame from full-frame scoring. Retain it in the requested population and do not
replace it with an easier frame. No supplied hotspot box or detector proposal
should be reused as a full-panel reference.

Compare thresholds and reference-coordinate perturbations separately. A fixed
±1-pixel sensitivity check is not a measured calibration uncertainty. Record
the exact reference hash frozen before predictions and require independent
review before acceptance. These diagnostics do not approve measurement masks,
create database records, infer unique site inventory, establish thermal anomaly
accuracy or certify field/IEC compliance.
