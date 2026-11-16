# Source-bound measurement profiles and reanalysis

An SDK decode, a panel boundary and a confirmed fault are separate results. The
inspection workflow now retains the measurement settings that actually produced
each analysis; source-level metadata is not used as a substitute for that record.

## Native measurement identity

Every new Stage 1 JSONL report and DuckDB run/capture slice includes a
`pvdelta.measurement-profile/v1` record with:

- Original source SHA256 and native matrix shape, dtype and SHA256.
- All five recorded and effective SDK settings, explicit overrides and SDK
  parameter readback status. Missing settings remain unknown.
- A content-derived profile ID and an explicit, unverified field-conditions flag.

CSV rows repeat the profile ID, native temperature hash and effective settings.
The report does not export arbitrary raw EXIF/XMP metadata. Profile identity is
provenance, not evidence that the chosen atmospheric settings were measured on site.

`pvdelta analyze` accepts `--measurement-params .\measurement.json`, for example:

```json
{"distance_m": 19.0, "emissivity": 0.95}
```

Only specified fields change. The SDK validates supported ranges and recalculates
the native matrix; the input JPEG is never modified. Use this only with justified
inputs. Aircraft altitude and a laser target distance are not automatically the
camera-to-panel distance for every pixel. Omitted fields retain recorded values,
which are not thereby validated.

## Full-flight reanalysis of stored geometry

```powershell
pvdelta replay-screening --database .\private\inspection.duckdb `
  --baseline-run original-screening --run-id measurement-review-001 `
  --out .\private\measurement-review-001 `
  --sdk-lib C:\path\to\dji-sdk\release_x64
```

Optional inputs are `--measurement-params` for a fixed profile, or
`--per-frame-params` for a JSON object mapping every selected source SHA256 to
explicit overrides. The two inputs cannot be combined. Per-frame mappings must
cover the exact selected source set. `--site-geometry` copies an explicitly chosen
site prior beside the output for repeat-view checks; it is not field validation.

The source database is opened read-only. The command creates a new
`inspection.duckdb`, a summary, per-frame outcome journal and JSONL/CSV/SVG reports.
Every selected original is hash-checked and decoded, including frames with no
assessable panel. Stored accepted boundaries must also pass their recorded
numeric consistency checks. Rejected boundaries remain `not_assessed`, shared
measurement pixels are excluded, and findings remain `review_required`.

This command remeasures stored provisional boundaries. It does not discover
missing panels, validate panel membership or establish a unique physical panel
count. Missing coverage and a zero-candidate observation are not healthy labels.
Failures are isolated to their frame; exit codes are 0 complete, 3 partial and
1 failed. A new output directory and run ID are required.

In Streamlit, open **Radiometric scan → Run anomaly analysis on stored panel
boundaries**. After completion, copy the displayed database path into **Local
database path** to inspect that analysis. `PVDELTA_DATABASE_PATH` can set the
dashboard's initial database without changing the repository's main store.

## Review and preview integrity

The thermal preview reapplies the saved effective SDK settings and checks the
reconstructed native matrix hash. A mismatch withholds its overlay and shows an
unmarked source image. Legacy analyses explicitly report that a saved profile is
unavailable. RGB anomaly transfer still requires compatible geometry; a filename
pair alone cannot establish a pixel mapping.

Schema v8 stores profiles and a result fingerprint per analysis/capture. Replacing
different results archives prior human decisions in `invalidated_finding_reviews`
and removes active stale decisions and finding evidence atomically. An identical
rerun preserves those decisions. Rollback preserves the entire previous slice.

Repeat-view UI cache identity includes measurement results, panel geometry, site
prior, source availability and relevant processing code. It never reuses old
multi-angle evidence after these inputs change. The bounded search examines up to
eight nearby captures within 30 seconds; provisional correspondence or insufficient
angles does not confirm a fault, establish a healthy module or rule out glare.
