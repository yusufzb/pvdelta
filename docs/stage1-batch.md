# Stage 1 batch processing

`pvdelta batch` processes an explicit local manifest. Every frame keeps its own report directory;
one corrupt or missing frame does not prevent later frames from running.

## Prepare a manifest from a flight folder

```powershell
pvdelta prepare-manifest .\flight `
  --regions-dir .\regions `
  --mask-dir .\masks `
  --out .\batch-manifest.json --run-id flight-001
```

Discovery recursively selects filenames ending in `_T.JPG` or `_T.JPEG`, ignoring case.
This naming check does not establish that the image is radiometric; decoding happens in `batch`.
Sidecars must mirror the flight's subfolders and preserve the capture stem:

| Capture | Native regions | Optional exact mask |
|---|---|---|
| `flight/leg1/DJI_0001_T.JPG` | `regions/leg1/DJI_0001_T.regions.json` | `masks/leg1/DJI_0001_T.masks.npz` |

Only matched frames enter `frames`. Missing, empty or malformed annotations appear in
`preparation.unresolved` with a reason. Repeated basenames in different subfolders never share
annotations; duplicate capture stems with different JPEG extensions are rejected as ambiguous.
When `--mask-dir` is supplied, a missing mask or missing annotation `mask_label` prevents inclusion.
Omit `--mask-dir` only when polygon rasterization is intended. Mask array validation happens during
batch execution. File matching alone does not validate panel alignment or source provenance.

All file references are relative to the generated manifest; keep the relative directory layout
when moving it. Inputs and output must be on the same drive. These manifests may reveal local
filenames and remain private experiment data.

Review `defaults` before analysis: generated values match `DetectionConfig` defaults (2.5 K,
4 region pixels, 16 reference pixels, 1 pixel erosion; both hot and cold). These are exploratory
settings, not fault severity thresholds. Preparation does not measure temperatures or write the DB.

Preparation exits with `0` when all discovered frames are matched, `3` when some are unresolved,
or `1` when none can be included or preparation fails. A zero-match output is a useful inventory
but cannot be run by `batch` until regions are supplied and the manifest is regenerated.
Existing manifests require `--overwrite`; capture and annotation inputs cannot be overwritten.

## Run the manifest

```powershell
pvdelta batch .\batch-manifest.json `
  --out .\reports\batch-001 `
  --database .\data\pvdelta.duckdb `
  --sdk-lib C:\path\to\dji-sdk\release_x64
```

Example manifest:

```json
{
  "schema_version": "pvdelta.batch/v1",
  "run_id": "batch-001",
  "site_label": "local pilot",
  "defaults": {
    "direction": "hot",
    "region_margin_px": 5,
    "detection": {
      "delta_threshold_k": 3.0,
      "min_region_px": 8,
      "min_reference_px": 24,
      "edge_erosion_px": 0
    }
  },
  "frames": [
    {"input": "flight/DJI_0001_T.JPG", "regions": "regions/0001.json"},
    {"input": "flight/DJI_0002_T.JPG", "regions": "regions/0002.json"}
  ]
}
```

Paths are resolved relative to the manifest. A frame can also specify `mask_archive`; when used,
the default `region_margin_px` must be zero and the region annotations need matching `mask_label`
values.

A frame may supply `measurement_overrides` (or inherit that object from `defaults`). These
are explicit SDK settings, not post-processing temperature offsets. An optional
`source_sha256` must match the decoded original; a mismatch fails only that frame.
Each successful report and database slice retains its own [measurement profile](measurement-profiles.md).

The batch root contains `batch-summary.json` and `batch-summary.csv`. Successful frame reports are
under `frames/0001-<source-stem>/`. Summary errors contain the source filename and sanitized error
detail; absolute local paths are not persisted.

Exit codes:

| Code | Meaning |
|---:|---|
| 0 | Every frame completed |
| 1 | Manifest/global failure, or every frame failed |
| 3 | Partial result: at least one frame completed and at least one failed |

Without `--overwrite`, an existing batch summary is rejected. With it, report files are replaced
and each successful run/capture DB slice is atomically refreshed. Failed frames are listed in the
summary and are not written as successful observations.
