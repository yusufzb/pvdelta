# Surface-assisted scan

The dashboard exposes saved manual-surface recovery in **Basic → Surface-assisted
scan** and **Advanced → Panel boundaries → Surface-assisted scan**. Availability
depends on a local `knowledge basis/surface-recovery/manifest.json` outside the
repository. No private paths, images, annotations or database are packaged.

Press **Run surface-assisted scan** to process every eligible configured frame.
Enable **Recover isolated tables inside saved panel areas** to replace the adjacent
baseline requirement with explicit manual panel support. Four observed sides,
two disjoint observed scale references, overlap/conflict checks and protection of
old measurement pixels still apply. Every candidate pixel must be inside the
eligible manual panel mask; unknown, conflict and non-panel pixels exclude it.
The option defaults off. Each saved run records its own mode; changing the checkbox
does not change an existing run or the automatic whole-flight scanner.
Saved results reopen after refresh. Previous/Next navigates successful frames;
**Saved scan run** reopens earlier attempts without recomputing. A failed or partial
retry is retained for diagnosis and does not replace the last complete run as the
default. All-failed attempts show their errors even when no successful run exists.
frames with additions appear first. Before/after images show cyan retained module
boundaries, yellow additions and red thermal candidate regions. The table shows
native temperature statistics and candidate counts. Failures remain explicit and
are not counted as empty or normal frames. A JSON download retains the profile,
source identity, geometry and all module reports.

This is a separately saved, manual-surface-assisted run, independent of the
inspection selected in the rest of the dashboard. It does not overwrite its
findings, reviews or counts. Configured-frame coverage is explicit; remaining
flight frames have not acquired reviewed masks by implication. Observation counts
are not unique physical modules, and absence of a local candidate is not a healthy
diagnosis. Whole-module peers, reflection and field-condition acceptance remain
separate requirements.

## Local configuration and integrity

`pvdelta.surface-recovery-input/v1` contains a confirmed
`panel_surface_and_non_panel` meaning and a list of frames. Each item binds a safe,
unique case ID, source path, complete surface reference, verified saved measurement
profile and baseline geometry with its canonical SHA256. Source identity, native
temperature hash and dimensions must agree. Unknown/conflicting/non-panel pixels
are excluded from recovery. No hotspot labels are inferred from surface labels.

The runner decodes original SDK pixels with saved parameters, verifies the native
array hash, adds only supported boundaries inside the mask, and checks exact
preservation of previous geometry and analysis. Results use unique run directories;
each finished attempt has a separate checksum-bound history record. The latest
pointer is published only when every configured frame succeeds. Existing legacy
latest records are retained in history before that pointer is replaced. The UI verifies JSON
and image hashes and identifies older engine versions. Saved results are snapshots,
not a fresh recheck of source files on every page visit. Changing the manifest
requires another run. Source/manifest/engine changes detected during processing
prevent successful activation. Reports remain provisional, not IEC acceptance.
