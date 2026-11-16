# Research local-group completion

`complete_local_native_groups` is an opt-in geometry research helper. Production
extraction and the existing two-baseline-anchor completion rule remain unchanged.
It accepts native line-group evidence from `line_affine_native_groups` and a
caller-verified baseline from the exact same source/grid. The expected native
temperature SHA256 must match; the caller still verifies source identity.

Four actually observed nonoverlapping cells must form a two-axis affine cycle.
Closed contours need multiple threshold votes. Segment candidates need four
supported sides and their original support metrics; inferred sides and hybrid
ridge completion cannot seed a group. Every group member must be an eligible
new envelope or a duplicate of existing geometry with nonempty native support.
Conflicts and empty measurement interiors invalidate whole affected groups.
There is no missing-cell interpolation or temperature/label-based winner selection.

Every existing dictionary/ID and every native measurement interior is preserved.
Shared excluded edges are permitted, but new masks may not touch old measurement
pixels. New IDs and direct group-neighbour IDs are mapped deterministically.
Inputs are bounded to512 observations and4096 groups.

This removes dependence on two old scale anchors only for this research branch.
Regular ground/roof equipment can satisfy the same geometry. All additions stay
semantically unverified, measurement-unauthorised and efficacy-unaccepted. A valid
group, native digest or positive draft benchmark score does not establish a PV
module, physical panel identity, fault, glare clearance or IEC field compliance.
