# Observed RGB panel views

In V2, enable **Compare other photos** and press **Find observed RGB panel views**. This is a separate RGB geometry path; it does not bypass or change the native structural checker. It works on native-flight runs without legacy RGB geometry stored in the database.

The source thermal is reconstructed with its saved SDK measurement profile. A flight-scoped quadratic prior supplies an **approximate search location** for the selected native observation. Existing full-module RGB extraction and bounded, mutual quad overlap must find one observed RGB module at that location. A missing/ambiguous source anchor aborts this search instead of choosing a different panel. This association is still not independently verified.

Up to eight GPS-nearby paired captures in the same acquisition are then inspected. Background RGB texture initializes displacement; local tracks inside the selected observed module fit the module plane. Coarse rotation and exposure are compensated on a display copy, including return tracks. Forward/backward tracks, local patch agreement, held-out points, bounded feature support and unambiguous observed target-module overlap retain the existing RGB tracking gates. P95 is measured on the feature image (long side at most 1600 px), not in native thermal pixels. No detector or native measurement thresholds are changed.

Every forward panel match now requires a separate reverse `match_panel` call from
the observed target polygon to **all** observed source polygons. The reverse match
must select the original source module, with the same feature/support/overlap gates.
A neighbour return or failed reverse check withholds the target outline. When both
directions provide corner correspondences, their composed corner permutation must
be the identity. Both direction reports and failure reasons are retained; no best
direction or residual trimming is used. This reduces a one-way correspondence
failure mode but cannot rule out a coherent repetitive-array alias in both directions.

The optional **Use observed table neighbours for RGB tracking** checkbox extends
the tracking-point mask to the selected observed module and at most four immediate,
scale/orientation-compatible neighbours with the same explicit observed table index.
Support is frozen before tracking/fitting. Polygon interiors are used separately;
the ground between them and unobserved grid cells are never filled. Other tables,
overlapping proposals, missing/invalid table identities and distant same-table
modules cannot supply support. This assumes the grouped table modules are coplanar;
the RGB grouping is still provisional, not surveyed geometry.

The selected panel's feature-hull bound, held-out 3px limit, unique target overlap,
reciprocal return and corner-cycle checks remain active. More tracking points do
not enlarge the measurement ROI or anomaly outline, change thermal masks, transfer
temperatures or clear glare. Both directional reports retain the support polygons.
The default stays selected-panel-only. Changing modes invalidates the previous
cached result and requires another explicit computation; failures remain visible.

## Reading the result

The table retains every attempted target with its status and reason. **matched_provisional** means local RGB image checks passed. It is not a confirmed physical asset identity, fine cross-camera alignment or fault verdict. The source anchor is also explicitly labelled approximate. Repetitive arrays can produce coherent but wrong correspondences; independent review remains necessary.

Previous/Next includes the source and unsuccessful target views. Cyan outlines show **observed RGB module geometry** for the source anchor and passing targets, with a 500-original-pixel context crop. They are not projected hotspot outlines. Additional thermal views are unmarked embedded photos: no source anomaly outline, target temperature, strongest neighbouring finding or review decision is transferred. The main card continues to show its original SDK-native measurements and all saved regions.

Every report keeps `panel_identity_verified`, `cross_sensor_verified`, `thermal_component_correspondence_verified`, `measurement_authorised`, `glare_clearance` and `database_written` false. RGB tracking alone cannot clear glare, establish viewing-incidence diversity, count unique physical panels or certify IEC compliance. CAD is not required for this diagnostic.

## Source and flight binding

The search prior is restricted to the original M4T flight and its original native 640×512 / RGB 4032×3024 grids. Original source/target file SHA256 values must match the index. The source profile, observed geometry, model, SDK binaries, engine sources and selected captures bind the cached result. Original RGB feature caches also use content hashes, even if file size/time are preserved. Changed inputs invalidate previous evidence; overlays verify photo/report hashes again at rendering time. One unavailable target does not replace or hide valid primary evidence.

The prior requires a local `rgb-search-scope.json` beside its calibration JSON, or an explicit `registration.flight_binding` path. It contains schema `pvdelta.rgb-search-scope/v1`, the exact resolved `source_root`, `calibration_sha256` of the original model file and `calibration_anchor_sha256` of its original calibration photo. This separate binding prevents silent reuse on another flight with the same filename. It documents search provenance, not a validated calibration. Models, bindings, photos and SDK binaries remain local and are not bundled into the repository.

The explicit button computes and caches read-only evidence; the JSON download preserves its source binding and scope. It does not change findings, measurements or user reviews. Missing compatible configuration leaves the unmarked GPS photo browser and native review available.
