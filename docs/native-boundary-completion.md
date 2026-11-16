# Observed native boundary completion

The default native detector remains available unchanged. The opt-in
`pvdelta.native_transition_completion.transition_completion_proposals` recovers
some already observed line intersections rejected for incomplete side evidence.
It never fills a rack from dimensions, row counts, CAD or reference labels.

During the ordinary native search, it collects side-rejected quads anchored by
at least two supported sides. Nearby duplicate observations are consolidated in
a bounded 512-candidate pool. Overflow retains the entire baseline and withholds
all additions with an explicit status. It is not evidence that the frame is healthy.

Each candidate must then independently pass all four existing native ridge or
mixed transition checks. Mixed evidence needs at least two ridge sides; a warm
four-step patch is insufficient. Existing scale, adjacency, overlap/conflict and
measurement support gates decide whether the candidate can be appended. At least
two baseline scale references, one adjacent, are required. Baseline polygon IDs,
geometry and actual measurement pixels are preserved exactly. Input temperature
values are checked for mutation; interpolation serves edge geometry only.

For a validated decoded `ThermalFrame`, a separate research analysis can use:

```python
from pvdelta.native_flight import analyze_native_frame
from pvdelta.native_transition_completion import transition_completion_proposals

reports, diagnostic = analyze_native_frame(
    frame, proposal_builder=transition_completion_proposals
)
```

The acquisition scanner also accepts this callable through `proposal_builder`
with `workers=1`. Always use a new output directory/run ID and freeze a source,
SDK and parameter manifest for injected research builders. Do not overwrite an
existing inspection or transfer its review decisions to new observations.

Additional geometry is unverified panel evidence, not a confirmed anomaly or a
unique asset. Recompute candidate measurements on original native pixels. Glare,
independent identity, field radiometry and inspection acceptance remain separate.

## Boundary relocation is a different operation

`pvdelta.native_refinement.refine_native_geometry` is a separate research helper
that relocates existing quads using four observed edges. It rejects conflicting
changes symmetrically, including conflicts introduced by restoring a rejected
baseline. It explicitly requires new measurement masks and disallows reusing old
statistics. It must not be confused with completion's preservation guarantee.

The local development comparison did not support promoting boundary relocation;
its default use is not enabled. Test references were AI drafts, not independent
ground truth. A limited completion gain on those drafts does not establish
whole-flight or external accuracy.

## Operational comparison before promotion

`pvdelta.native_completion_audit.compare_native_completion(frame, saved_diagnostic)`
replays the default and optional searches on the same original SDK matrix. First
verify the source bytes and saved measurement profile, and obtain the diagnostic
with `read_native_diagnostic` for the exact analysis run/capture. The comparison:

- rejects baseline geometry that differs from that capture's saved observations;
- independently rebuilds old/new masks and checks saved mask SHA256 values;
- verifies unchanged native values, disjoint interiors and excluded-pixel NaNs;
- compares every old module report separately from mask preservation;
- reports additions, pool overflow and both search timings, including zero gains.

Adding a peer can affect a future peer-based detector even when masks are
unchanged. `baseline_reports_changed` therefore remains a separate check. Current
native observations do not carry explicit validated peer groups; this comparison
does not establish whole-module anomaly coverage. A new boundary candidate is
not an anomaly, a verified panel or evidence that other panels are healthy.

For long private comparisons, `save_audit_record` and `read_audit_record` bind a
checkpoint to the full frozen plan digest and capture ID. Existing records cannot
be overwritten. Reuse verifies the payload digest; failed records remain failures
in the denominator. A truncated record stops reuse and needs investigation.
These local digests detect corruption, not malicious changes or reviewer approval.

The caller must freeze source/SDK/code/parameters, record runtime dependencies,
verify them on resume, prevent concurrent writers, verify any linked artifacts,
and retain all selected failures. Use a fresh inspection for changed inputs. Keep
reference scoring separate; an unlabeled operational sample measures preservation
and cost, not precision/recall. Preserve locked evaluation cases for their intended
release protocol.

## Diagnostic identity

`pvdelta.native_diagnostics.read_native_diagnostic` binds screening diagnostics to
the exact analysis run and capture, validates source metadata and matches every
proposal ID/polygon against stored observations. Optional original bytes are
hash-checked. A diagnostic with no proposals is a valid unknown-coverage result.

Processing folders such as `frames/1165` are not DJI filename counters such as
`..._1166_T.JPG`. For exported files, use the run summary's `report_directory`,
check containment in the report root and compare the file with the capture-bound
diagnostic. Never guess a diagnostic path from the image basename.
