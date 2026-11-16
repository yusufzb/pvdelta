# Protected native completion (research preview)

`pvdelta.native_completion_search.supported_completion_proposals` runs an
explicit, opt-in geometry comparison on one native temperature grid. It does
not replace production boundaries, create database findings, or authorise
measurement masks. External hotspot efficacy has not passed acceptance.

For scenes exceeding the observed line-pair budget, the separate
[bounded dense search](native-dense-search.md) raises only that explicit budget
while retaining completion/pixel-preservation rules. It does not widen aspect.

```python
from pvdelta.decode import RJpeg
from pvdelta.native_completion_search import supported_completion_proposals

with RJpeg(source_path, sdk_library) as image:
    temperature_c = image.temperature()

preview = supported_completion_proposals(temperature_c, margin_px=2)
assert preview['native_completion']['old_measurement_pixels_lost'] == 0
assert preview['measurement_authorised'] is False
```

The baseline uses the caller's `NativeBoundaryConfig` and `line_config`.
A separate observed search lowers only `min_aspect` to at most 1.3. If the
baseline already has that range, a redundant search is avoided. This aspect
range is experimental image geometry, not a universal module size or a camera
calibration. Both searches use exactly the same array; its SHA and dimensions
are recorded and no radiometric resampling is performed. A mutated/nonfinite
source fails explicitly.

An alternative must provide a closed contour observed at two or three Canny
thresholds, or all four line sides with at least .8 coverage and at most .18
maximum gap. These are structural observations of one image; threshold votes
are not independent views or semantic panel approval. The method invents no
missing sides, layout cells or CAD coordinates.

Completion preserves existing proposal dictionaries, polygons and IDs. An
addition needs two disjoint baseline scale references, one adjacent. Only
baseline references with observed-side evidence and a nonempty measurement
interior may vote. Alternatives cannot vote for each other. Strongly
overlapping eligible alternatives are all withheld rather than resolved by
source order. Rejection reasons and reference geometry remain available.

Any alternative touching an old measurement pixel is withheld, even if its
geometric IoU is small. Limited overlap with already excluded envelope edges
is permitted. After composition, supports are rebuilt using the existing
partition rule, and every old interior is checked bit-for-bit. Shared pixels
are never assigned to an arbitrary module. Added IDs avoid existing IDs;
old neighbour metadata is retained and new neighbour links refer to baseline
observations. This is not a reconstructed complete adjacency graph.

The pixel guarantee applies to the returned `margin_px`. Changing that margin,
source, engine or measurement configuration requires a new preview and review.
It does not establish that the baseline was correct: baseline false positives
and wrong rectangles are preserved too. A closed rectangle may still contain
vegetation or roof pixels. A valid addition is provisional geometry, not a
verified physical panel or an electrically faulty/healthy module.

For cached geometry, the lower-level
`pvdelta.native_completion.complete_native_proposals` accepts baseline and
alternative result dictionaries. The caller must verify that both came from
the same source SHA, native grid, SDK temperature array and effective parameters.
Prefer the one-call entry point when those input identities are not already
pinned. Neither function receives published reference coordinates.

Validation should separately check full-panel semantics, old-pixel/report
preservation, region match gains/losses and unmatched candidate growth. Hotspot
boxes cannot serve as full-panel references. Keep an evaluated group as
development feedback once it informs a change, then freeze another group.
Repeated-view identity, glare evidence, field calibration, unique panel counts,
GPS accuracy and IEC acceptance require separate validation.
