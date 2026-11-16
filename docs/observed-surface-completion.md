# Observed surface completion (opt-in)

`pvdelta.native_observed_completion.complete_observed_surface_gaps` recovers
observed panel quadrilaterals that an earlier deduplication step can hide behind
a larger envelope. It consumes actual `four_sides_passed` search events before
deduplication. It does not lower edge thresholds or extrapolate a rack grid.

Set `raw_supplement=True` to also search raw structure inside each supplied window
when enhanced contrast has already found some modules. The previous fallback only
searched raw structure when an enhanced pass returned no proposals. Partial
success therefore did not establish complete search coverage. The additional pass
uses the same four-side, geometry, mask and protected-completion gates and shares
the same candidate budget. The default remains false for existing callers.

The caller supplies the unchanged native temperature array, the existing boundary
result, explicit integer search windows, and a boolean manual panel-surface mask.
Unknown, conflicting and non-panel pixels must be false. The caller must verify
that all inputs belong to the same source and native pixel grid. Manual surfaces
are not hotspot labels, individual panel identities or independent acceptance data.

Each candidate must lie completely inside the supplied mask. The existing protected
completion step requires two compatible baseline scale references, an adjacent
reference, four observed sides and unchanged existing measurement support. Similar
observations are deduplicated; conflicting alternatives are withheld. If the bounded
pool overflows, all additions are withheld and the baseline is preserved.

The original temperature array is never modified or resampled. Geometry results
remain provisional with `measurement_authorised=False`. A separate radiometric
analysis is required; an added boundary is not an anomaly or a healthy diagnosis.

This is an explicit research helper, not enabled in the default scanner or
dashboard. Fixed development images showed a recovery, but independent flight
validation and automatic panel-surface reliability are still outstanding. Original
photos, masks, experiment output and databases remain outside the public package.
