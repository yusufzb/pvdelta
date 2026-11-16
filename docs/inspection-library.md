# Saved inspections

Basic mode can offer a **Saved inspection** selector above the review workflow.
Keep an optional local `data/inspection-library.json`:

```json
{
  "schema": "pvdelta.inspection-library/v1",
  "inspections": [
    {
      "id": "original-flight",
      "label": "Original flight · radiometric inspection",
      "database_path": "inspections/original/inspection.duckdb",
      "analysis_run_id": "original-native-run"
    }
  ]
}
```

Paths resolve relative to the library file; absolute local paths also work.
`PVDELTA_INSPECTION_LIBRARY` or the Python `library_path` argument can select
another local library. No database discovery, migration, merge or scan runs when
opening the selector. Missing optional libraries leave the previous workflow intact.

Choosing an entry opens that exact database and analysis run. A mismatched run
produces an error instead of substituting another analysis. Switching resets review
filters to **All candidates / All** and clears unsaved review input and comparison
state; saved decisions remain in their own
databases. No decisions transfer between analyses. Opening results is read-only.

Recorded measurement-profile counts describe stored settings, not field
calibration or confirmed fault counts. Native SDK previews must reproduce their
saved profile/hash; legacy previews are explicitly limited to saved locations.
Different analyses can have different provisional boundaries and candidate counts.
Those counts are not additive, unique panels or confirmed faults.

Keep the library private: `data/` is ignored by Git, and paths can identify local
flight data. Include matching `summary.json` when moving a native inspection so
its own detector settings remain available to temperature/repeat evidence checks.
