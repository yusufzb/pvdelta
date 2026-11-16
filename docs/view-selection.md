# Repeated RGB view selection

`pvdelta rank-views` ranks several RGB observations that have already been assigned to the same
explicit `panel_key`. It stores a selected view and the remaining candidates in the local DuckDB;
the original files stay in place.

```powershell
pvdelta rank-views .\panel-views.json `
  --out .\reports\panel-views.json `
  --database .\data\pvdelta.duckdb
```

Manifest shape:

```json
{
  "schema_version": "pvdelta.views/v1",
  "run_id": "pilot-views-001",
  "groups": [
    {
      "panel_key": "table-07-row-B-col-04",
      "views": [
        {
          "rgb_capture_id": "sha256:...",
          "rgb_path": "flight/DJI_0001_V.JPG",
          "thermal_capture_id": "sha256:...",
          "latitude": 39.0,
          "longitude": 32.0,
          "horizontal_accuracy_m": 8.0
        }
      ]
    }
  ]
}
```

The score combines Laplacian sharpness, non-clipped luminance fraction and contrast. It is a
within-group ranking score, not a probability and not a thermal quality assessment. The selected
RGB view does not automatically become the selected thermal measurement; the thermal candidate
still needs its own native-grid quality and anomaly checks.

The command deliberately does not infer `panel_key` from a filename. A key must come from a
reviewed panel/table geometry, CAD match, or a future validated tracker. A thermal/RGB filename
pair proves correspondence between sensors for one capture; it does not prove that two different
captures show the same physical panel. GPS is retained as provenance and is not used as the sole
identity key.

DuckDB table `view_rankings` keeps the score, rank, selection status, RGB and thermal capture IDs,
source path and available coordinates. Re-running the same `run_id` replaces only the affected
panel groups.
