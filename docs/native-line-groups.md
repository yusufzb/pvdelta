# Four-supported-line affine groups — research only

`line_affine_native_groups(native_temperature_c)` supplements closed contours
with actual four-side segment hypotheses. It consumes native search trace events
**after** side support, internal-divider and source-border checks, but before
legacy dedup/neighbour selection. A closed Canny contour is not required for the
line hypotheses. Each side still needs coverage ≥.8 and largest gap ≤.18.
Missing sides are not inferred, and native-ridge/hybrid supplementation is off.

The search runs on the full native frame and four overlapping corner windows,
each covering two thirds of the source dimensions. No temperature grid is warped,
resized or ranked. Windows with insufficient structural range and pair-budget
skips are recorded explicitly. A finite contiguous numeric native grid is required;
the function owns its trace context and cannot run inside another active trace.

Extraction uses a broad numerical envelope (aspect .5–6.1, corner cosine .95,
16px segments, default 250k pair product per window). The explicit
`max_pair_product` argument accepts an integer up to 2M for bounded research
searches; shape/side/group gates are unchanged. Pair and family budget skip
counts are reported separately. This deliberately differs from the
default shape prior. Subsequent locally affine shape/position checks and four
observed non-overlapping cells in a two-axis 2×2 cycle supply the alternative
consistency test. This is not calibrated projective rectification. Closed quads
retain their actual two-threshold votes; line evidence has four side metrics,
never fabricated threshold votes. Equivalent closed observations take priority
over near-duplicate line observations, without temperature/reference input.

Storage is bounded to 8192 four-side events, then 512 distinct quads and 4096
cycles. Evidence/group-budget failures withhold all group proposals. Conflicting
groups are all withheld. Full-pass status and counters remain available, including
unsearched areas; zero proposals is not evidence of healthy modules.

Output is research geometry: semantic membership, physical module scale and
measurement authorisation are unresolved. Regular non-PV structures or submodule
patterns can satisfy the tests. Same-frame windows/segments are not independent
viewing angles and cannot resolve glare.

For verified matching raw-source/native hashes, `complete_native_proposals` may
separately preserve a baseline and add eligible geometry. Its two existing scale
references, one adjacent reference, old measurement-pixel protection and conflict
checks are unchanged. A raw warm cue withheld by those checks belongs in a
separate source/pixel-bound review artifact, not accepted fault/normal totals.
This function changes no Streamlit defaults, primary database or previous review
bundle. Full-panel localisation, hotspot error costs and independent review are
required before any promotion.
