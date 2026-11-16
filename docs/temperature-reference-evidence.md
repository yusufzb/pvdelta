# Temperature reference evidence

Panel review v2 → **Temperature reference evidence** → **Check temperature evidence**
reconstructs the saved native DJI temperature grid and the entire observation's
hot-region population. The operation is explicit and read-only. It verifies saved
mask hashes, measurement profiles, exact component pixels, median temperatures,
delta values and reference pixel counts. A mismatch with the current detector
withholds diagnostics rather than displaying unrelated evidence.

The same converged background/reference used by the detector is exposed only on
request. All hot/cold outliers, including components too small for display, stay
excluded from the reference. The ordinary detector output is unchanged.

The table reports:

- **Extrapolated region (%)**: component pixel centres outside the convex hull of
  retained reference pixel centres. The background plane is extended into this
  area; inspecting the mask and thermal gradient is necessary.
- **Reference noise MAD (K)**: median absolute deviation of background residuals.
  It is not camera accuracy, a detection confidence, or a probability.
- **At minimum pixel support**: region contains exactly the configured minimum
  native pixels. A warning to obtain more evidence, not automatic rejection.
- Reference and region counts, median and peak residuals.

Interpolation does not prove semantic panel membership, a valid physical plane,
or glare clearance. Extrapolation does not prove a false positive. No finding is
removed or promoted to a confirmed fault. The JSON download retains source/run/
observation/profile identities. Diagnostics are recomputed on request rather
than reused for another observation. Existing V1 and native scan results remain.

## Conditional sensitivity checks

The same button also runs four fixed spatial deletion checks. The midpoint of
the effective measurement mask's bounding box defines four image-axis blocks.
Each check omits one block of the already-selected reference pixels, refits the
configured plane (or median), and measures the original exact candidate pixels.
Training and held-out blocks must each have at least the configured minimum
reference count; plane fits must have full rank. Missing checks are unavailable,
never a pass. The table shows the ΔT range and whether it crosses the existing
threshold. Held-out residuals and block counts are retained in the JSON.

This is conditional on the original reference selection. It is not independent
cross-validation of anomaly detection, a confidence interval or fault probability.
A common biased reference can remain consistent in every check.

One- and two-pixel extra mask margins also report how many original candidate
pixels remain. Temperatures and candidate coordinates are unchanged. This is
support-retention stress only: no detection rerun, physical error bound, or reason
to discard edge faults. These checks never delete or approve findings.
