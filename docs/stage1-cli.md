# Stage 1 explicit-region CLI

The current CLI analyzes one original DJI thermal R-JPEG using module regions already expressed
on its native radiometric grid. It is inventory-independent: module IDs are frame-local labels,
not plant Panel IDs.

```powershell
pvdelta analyze .\DJI_0001_T.JPG `
  --regions .\regions.json `
  --out .\reports\run-001 `
  --run-id run-001 `
  --database .\data\pvdelta.duckdb `
  --sdk-lib C:\path\to\dji-sdk\release_x64 `
  --margin-px 5 `
  --delta-k 3 --min-region-px 8 --min-reference-px 24 `
  --edge-erosion-px 0 --direction hot
```

The output directory contains:

- `<run-id>.jsonl`: run, every assessed/not-assessed module and complete finding provenance;
- `<run-id>.csv`: one review row per module/finding, including zero-candidate coverage;
- `<run-id>-overlay.svg`: vector evidence on the native thermal coordinate grid.

The DuckDB write is optional. A run/capture slice is replaced atomically on rerun, so stale
findings cannot survive and counts do not grow accidentally. Different run IDs preserve different
algorithm/configuration runs of the same capture.

## Region file

```json
{
  "schema_version": "1.0",
  "coordinate_space": "thermal_native",
  "modules": [
    {
      "module_id": "M01",
      "polygon": [[10, 20], [80, 20], [80, 150], [10, 150]],
      "visibility": "full"
    }
  ]
}
```

The polygon path is rasterized using pixel centres and then reduced by `--margin-px`. Temperature
does not influence the module mask.

For a previously reviewed exact native mask, use `--mask-archive masks.npz`. The archive must
contain a `module_labels` integer array and a boolean `measurement` array matching the decoded
temperature shape. Each module entry then needs `"mask_label": 1` (or its corresponding positive
label), and `--margin-px` must be zero. This preserves a reviewed mask exactly instead of trying to
reconstruct it from a simplified polygon.

`--direction hot`, `cold` or `both` is an explicit reporting filter. A component touching the
effective measurement boundary is stored as `review_required`. This status is not a diagnosis or
fault class.

The command requires the separately installed DJI Thermal SDK. SDK binaries, original images,
mask archives, reports and DuckDB files must remain local and are ignored by the repository.

For explicit SDK measurement corrections, per-analysis native hashes, review invalidation
and source-bound full-flight reanalysis, see [measurement profiles](measurement-profiles.md).
