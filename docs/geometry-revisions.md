# Versioned native geometry edits

In **Panel boundaries**, compare a source frame, then open **Revise boundaries
and reanalyze (separate versions)**. Start a session, select a frame-local ID,
and replace its four native corners, remove it, or add a new region. Supply a
reason and preview the revised analysis. A changed polygon discards stale edge,
GPS and other derived evidence. It does not inherit independent approval.

The entire frame is reanalyzed: changing one envelope can change neighbouring
pixel ownership, and explicit peer dependencies can affect other reports.
Temperatures always come from the unchanged original SDK/native grid. A changed
measurement ROI can legitimately change its reported temperatures and findings.

Previous versions remain in session history, including edits returned to a
parent. Download a version or save the local immutable JSON history before
leaving the session. This does not activate a version in the main scan or write
inspection findings to DuckDB. The preview uses recorded SDK parameters and
default exploratory Stage 1 settings, shown in the panel.

The Python API provides `analyze_geometry_version`, `apply_geometry_patch`,
`geometry_version_impact`, and content-addressed save/load. Patches bind the exact
parent, source and native SHA256 and explicitly identify add/replace/remove IDs.
Versions bind configuration, ROI policy, valid mask, capture context, decoder
parameters and analysis-engine/runtime identities. Changed settings or engines
need a fresh baseline for a fair paired geometry comparison.

Version history proves provenance and preserves old numerical results; it is not
human ground truth, physical panel identity, fault/glare clearance, healthy-panel
confirmation or IEC field compliance. CAD/panel registration stays optional in
Stage 2. RGB support requires verified correspondence; it is not transferred from
a replaced thermal polygon.

Open **Evaluate revision against panel reference** inside a revision session to
compare the active version with the session baseline. Supply an exhaustive
native full-panel reference JSON or a bundle, using the same format as the
boundary reference evaluator. Source/grid identity must match; a declared native
temperature digest must also match. Missing or incompatible references fail
explicitly instead of becoming zero scores.

Fixed IoU 0.5/0.75 diagnostics show matches, missed references and unmatched
proposals, including newly gained and lost matches. Measured component pixels
are separately checked against the supplied panel and explicit ignore masks.
The enlarged display box is never used to decide pixel placement. Temperatures
are not re-decoded by this read-only comparison. Reference changes, version
changes and scorer changes invalidate cached results.

Label review status remains declared metadata. AI-assisted geometry edits cannot
be evaluated as independent automatic accuracy gains against the same drafts.
The comparison does not mark panels healthy, clear glare, certify a fault,
activate a scan version or write inspection results to DuckDB.
