# Fixed evaluation cohorts

`pvdelta.evaluation_cohort` prepares deterministic frame cohorts for geometry
diagnostics. It accepts indexed original M4T nadir pairs and frozen observation/
finding counts. Frames are assigned to separated camera-coordinate blocks, then
stratified by candidate-bearing, analyzed-without-candidate and no-analysis status.
Within strata, camera-position spread and available heading families guide selection.
Known regression hashes are excluded explicitly. Insufficient data raises an error;
the spatial buffer is never silently reduced.

This same-flight split does not establish surveyed footprint separation, audited
absence of prior exposure, independent-flight performance or unique physical panels.
Outcome-stratified sampling does not estimate anomaly prevalence. Check scene/flight
independence separately and do not transfer the initial site's geometry to other sites.

Freeze the manifest, input/model/profile hashes and predictions before annotations or
tuning. Store private imagery and evaluation artifacts outside the public repository.
Unmarked SDK-derived displays can be fed to the existing source-bound panel annotation
editor; its exports are developer drafts until independently reviewed.

`score_panels` checks frozen predictions and scores exhaustive full-panel references
using the existing maximum-cardinality one-to-one instance evaluator. Missing or
incomplete references remain pending. Empty predictions against labeled panels count
as misses; only explicitly exhaustive empty truth counts as a zero-panel reference.
Changed sources/grids/predictions are rejected. Locked scoring requires explicit
release and completed references for every selected locked frame. The caller should
also verify the immutable manifest and artifact listing; hashes are integrity checks,
not proof of reviewer identity or absence of earlier exposure.

Report panel geometry, candidate-region detection, repeat association and field fault
verification independently. This utility does not declare Stage 1 acceptance or IEC
compliance. A locked test must be replaced if its results are used for model tuning.
