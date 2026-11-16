# Per-pair RGB image checks

**Check RGB alignment** is available beside native-run RGB previews in Basic and
Advanced. It explicitly processes the selected original pair. It never reruns
anomaly detection, changes temperatures, changes findings or writes the inspection
database. Results and their detailed JSON download remain local.

The original compatible M4T quadratic diagnostic is an initial visual prior.
Full-module RGB boundaries are extracted from the original visible frame. Matches
must have mutual unique overlap (IoU ≥.5, both ambiguity margins ≥.2) and corner
displacements within 12 native pixels. No source proposal is forced into a target.
The observed RGB table envelope bounds the local group; a rack grid is not filled.

The selected module and its nearest spatial neighbour are fixed as checks before
fitting. Shared corner observations are assigned to checking, excluded from fitting,
and deduplicated before the existing local registration API is called. Every raw
held-out corner is additionally scored before a correction can be used. There must
be at least two fit/two check modules and eight distinct fit/four check controls.
The local point gate stays P95 ≤2 native pixels. A usable correction must improve
the held-out P95, remain within the 12-native-pixel displacement bound and have no
projective pole across its support. These are development geometry limits, not
IEC acceptance thresholds.

Even a passing result is **provisional image consistency**, not independently
verified physical panel identity or glare/fault clearance. Repeated frames and
automatically inferred corners are not independent correspondence ground truth.

Before displaying a correction, the saved result must match the current RGB hash,
thermal/profile/native identities, prior, selected module and result content hash.
The candidate must lie inside both fit and check control hulls. The enlarged 5×
outline is a visual display boundary, not an enlarged measured anomaly.

If any prerequisite or check fails, the existing labelled approximate projection
is retained. The candidate is not deleted or called healthy. A failed attempt,
changed source, geometry, profile, model, selected module or engine clears cached
checks. Missing optional vision dependencies leave ordinary previews available.

The initial private development sample used 12 evenly spaced candidate-bearing
frames out of 139, one candidate per frame. **0/12 qualified**: five failed the
point gate, two lacked distinct controls after shared-corner exclusion, one lacked
enough modules, and four lacked a unique bounded full-module match for the target.
This provides no evidence of improved accepted alignment. Source-bound native
boundary/correspondence quality remains an open Stage 1 requirement.

## Numerical conditioning and rejection details

The shared `fit_homography` solver centres and isotropically normalizes both fit
coordinate sets before homogeneous DLT, then returns a matrix and residuals in
the original pixel domains. Rank-deficient, collinear and zero-spread controls
are rejected. The minimal four-pair solve retains the ninth right singular vector;
larger fits avoid an unnecessary quadratic-size left SVD basis. No held-out point
participates in normalization or fitting. This improves numerical consistency
under pixel origin/unit changes; it cannot establish correct correspondence.

`explain_module_matches` returns both the original accepted `matches` list and a
decision for every native source proposal. `match_module_quads` remains a wrapper
returning the list for existing callers. Decisions distinguish missing observed
RGB geometry, insufficient overlap, ambiguous RGB targets, competing native
sources, invalid prior pixel scale and excessive native corner correction.
The existing IoU/margin/12px limits are unchanged. `check_rgb_pair` includes these
as `matching_diagnostics` and `selected_matching_diagnostic` in its source-bound
JSON and existing alignment-details expander/download.

Unmatched geometry stays unknown. It must not be silently classified as vegetation,
discarded from anomaly recall or treated as healthy. Geometry evidence still needs
same-physical-frame-band centre observations, original-source identity and suitable
independent controls before it can support thermal masks or repeated-panel identity.

A fresh 12-case SDK regression comparison of the conditioning change completed all
cases with unchanged quad matches, controls and fit/check roles. Held-out P95 fell
in three of five fitted cases and rose in two; accepted frames remained **0/12**.
This is a numerical correction and better error visibility, not a demonstrated
solution to the RGB/thermal alignment or panel/vegetation problem.
