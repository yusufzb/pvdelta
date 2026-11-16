# Source-bound measurement exclusions

`PVDELTA_SURFACE_MANIFEST` configures one existing surface-recovery manifest for
source-bound non-panel/conflict veto throughout the application. It is optional;
without configuration the prior behavior is preserved. The Windows launcher
accepts `-SurfaceManifest` and discovers the existing private workspace manifest
when available. A malformed/missing configured file is an error, not a silent
fallback to unguarded measurements.

The shared layer matches the exact original source hash, native temperature
hash, pixel grid, completed review and declared panel/non-panel meaning. It
combines exclusions with any existing validity mask; it never re-enables invalid
sensor pixels or changes native temperature values. Explicit non-panel and
conflicting labels are excluded from measurement and reference sampling. Labelled
panel and unlabelled pixels are preserved; unknown does not mean healthy.
Different parameters producing a different native hash need new validated source
binding rather than automatic reuse of a stale reference.

The decoder, annotation pipeline, direct intra/peer detectors, native analysis
and geometry remeasurement use the shared validity mask. Native diagnostics,
geometry versions and Stage1 report settings preserve the constraint binding.
Configured manifest changes invalidate preview/version engine identities.
Raw whole-photo temperature summaries remain summaries of the original full
image, not falsely presented as panel-only statistics.

```powershell
.\scripts\Start-PvDelta.ps1 -SurfaceManifest 'C:\private\surface-recovery\manifest.json'
.venv\Scripts\python.exe -m pvdelta.cli scan-native-flight --database 'C:\private\index.duckdb' --acquisition flight --run-id new-guarded-run --out 'C:\private\new-run' --surface-manifest 'C:\private\surface-recovery\manifest.json'
```

Native scans snapshot the reference set and report source-matched versus
unreferenced frame counts. Each original source remains an explicit outcome.
An unreferenced source receives no guessed/copied surface mask; its observations
and candidates remain provisional. A new database preserves older inspections.
This mechanism does not infer boundaries, unique physical panel identities,
glare clearance, independent fault accuracy or IEC field acceptance.
