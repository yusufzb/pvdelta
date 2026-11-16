# User-declared panel surfaces

`surface_reference.import_surface_work` imports native polygon-editor exports only
after explicit confirmation that R regions represent panel/table surfaces and U
regions represent non-panel background. These are distinct from hotspot labels
and individual-module references. The original JSON and vertices remain intact.

Native labels are 0 unlabelled, 1 panel, 2 non-panel, 3 overlapping/conflicting
claims. Conflict and unlabelled pixels remain unknown; no erosion, priority rule
or automatic removal resolves them. Completion, reviewer and source/display/native
identity checks reuse the editor validator. Pending frames are not scored.

`audit_surface_predictions` reports proposed envelope support and exact saved
candidate pixel support against the declared surfaces. Candidate bounding boxes
are not used for this support check. A candidate entirely on a labelled panel can
still be glare, a joint or another false positive; no fault label follows. The
envelope union's panel-pixel coverage is not complete-module recall and includes
partial modules. No physical module count or extrapolated whole-flight label set.

These functions return reference/diagnostic records. They do not modify an
inspection DB, remove findings or authorise radiometric measurements. Manual
semantic mistakes must remain visible for review; a developer flag does not
silently relabel a user's polygon. Independent review remains a separate step.
