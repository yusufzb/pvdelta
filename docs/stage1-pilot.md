# Stage 1 local pilot

This is an operator-reviewed development pilot for original DJI M4T RGB/R-JPEG
flights without CAD or an external panel inventory. The software decodes native
SDK temperatures, indexes photos, produces provisional observations/candidates,
supports source review and exports inspection evidence. Automatic whole-flight
accuracy, unique physical-panel counts, glare clearance, field temperature accuracy
and independent Stage 1 acceptance remain open. Orthoview remains experimental.

## Install and start on Windows

From the checkout root, with Python 3.10 or newer installed:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e '.[dashboard,vision,storage]'
.\scripts\Start-PvDelta.ps1 -CheckOnly
.\scripts\Start-PvDelta.ps1
```

The proprietary DJI SDK is supplied separately. The launcher uses an explicit
`-SdkLibrary`, then `PVDELTA_DIRP_LIB`, then `sdk/tsdk-core/lib/windows/release_x64`.
It checks the SDK and dependencies before launching. The photo folder is never
used as an SDK directory. No Python activation-policy change is needed. If script
execution is restricted, use the direct command in [flight dashboard](flight-dashboard.md).

When the existing private workspace surface manifest is available, the launcher
enables its source-bound exclusions throughout measurement and reference sampling.
Use `-SurfaceManifest` or `PVDELTA_SURFACE_MANIFEST` for another explicit manifest.
Only matching source/native/grid/review bindings are eligible; remaining frames
stay unverified. See [shared surface constraints](surface-constraints.md).

Open `http://127.0.0.1:8765/`. The default address also permits access from devices
on the same LAN using the computer's LAN IP and port8765, subject to network/firewall
settings. The launcher does not change firewall rules. To bind only locally:

```powershell
.\scripts\Start-PvDelta.ps1 -BindAddress 127.0.0.1
```

## Operator workflow

1. **Basic → Start a new inspection:** select original photos and start indexing
   and native scanning. The SDK must be installed. Each scan creates a separate
   local inspection; earlier inspections remain available.
2. Review candidate thermal/RGB evidence, filenames, provisional locations and
   processing coverage. Candidate regions and frame observations are different
   from confirmed faults and unique physical modules. A frame without measured
   regions is not automatically empty or healthy.
3. Use **OK / Not OK / Wait** with source evidence. Nearby images are manual context
   until the same physical panel and viewing geometry have been validated. A failed
   RGB registration withholds precise overlays.
   For a prepared pair, **Recheck both thermal measurements** reads each original
   R-JPEG with its saved SDK settings and shows the reproduced local ΔT.
   Use the paired images to review physical identity. Viewing angle and reflection
   require their own evidence.
4. When a source-bound manual-surface manifest is available, **Surface-assisted scan**
   can recover boundaries inside those saved areas. **Recover isolated tables inside
   saved panel areas** is optional. Coverage is limited to configured frames. Its
   saved results remain separate from the main inspection. **Saved scan run** opens
   older runs; failed retries retain explicit errors and preserve the last complete run.
   **Exclude saved non-panel areas from measurements** is optional and source-bound.
   It vetoes only explicitly labelled non-panel/conflicting native pixels, retaining
   labelled-panel and unlabelled pixels. Unknown support remains review-required;
   it is not a healthy-panel label. Empty eligible masks are not assessed. This
   option changes the new run's measurements while preserving original geometry
   and all earlier runs; it does not supply masks for unannotated flight frames.
5. Advanced source controls preserve original fit/check roles. AI observations are
   labeled as such. Wait/Not OK blocks export; explicit source-joint aliases prevent
   shared fit/check leakage. A point-gate pass is not independent physical calibration.
   The local RGB boundary preview has an explicit2.00/2.20native-pixel development
   tolerance selector. Its choice is bound to the saved model and invalidates stale
   previews. This does not activate candidate findings or change field acceptance.
6. Export one selected stored analysis for handover, then verify its integrity:

```powershell
.venv\Scripts\python.exe -m pvdelta.cli export-inspection --help
.venv\Scripts\python.exe -m pvdelta.cli verify-inspection-export --help
```

## Preserve and restore

Keep the checkout, exact wheel/checkpoint SHA, environment requirements, DJI SDK,
private original photos, local databases and review sidecars. SDK, originals,
credentials and databases are not bundled in the public package. Archive output
directories alongside their hash manifests. Restore into a separate folder;
point Advanced at the restored DB and SDK. Verify exports before use and check
source/profile identity before recomputing measurements. An immutable saved report
is a historical snapshot, not a fresh check of later source/review changes.

Do not publish private image/control/inspection artifacts. This pilot provides no
IEC field-compliance certificate or validated automatic full-flight fault count.
The acceptance work is tracked separately from software delivery in the local plan.
