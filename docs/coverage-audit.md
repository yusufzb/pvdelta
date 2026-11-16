# Stored coverage audit

```powershell
./.venv/Scripts/python.exe -m pvdelta.cli coverage-audit --database private/inspection.duckdb --run-id inspection-001 --out private/coverage-audit.json
```

Requires `pvdelta[storage]`. This opens an existing database read-only and never
creates a missing one. Existing reports require `--overwrite`; the source database
is protected. Original photographs and SDK libraries are not needed for this audit.

`pvdelta.coverage_audit.build_coverage_audit(connection, run_id)` accepts an open
connection and issues SELECT queries only. `read_coverage_audit(database, run_id)`
opens and closes its own read-only connection. Unknown analysis IDs fail explicitly.

The report scopes captures through the acquisitions referenced by the selected run.
It separates indexed captures, screening records, analyzed/unassessed observations,
observations with findings and individual finding records. It does not deduplicate
physical panels or label observations healthy. Missing independent verification is
unknown, not zero. An absent optional `flight_screening_frames` table is supported
without schema changes; required core tables/columns must already exist.

The legacy gate interpretation uses at least four native proposals, frame edge
P90 ≤5 px, absolute x/y shift <20 px, module edge offset ≤5 px, MAD ≤2 px and gradient
≥15. These are the historical EXP-043 image gates, not IEC criteria. The audit does
not fit transforms or establish correspondence. Detailed rejection reasons overlap;
frame stages and module gate-state combinations are exclusive.

`priority_review_frames` lists edges-pass/frame-reject cases for investigation only.
No finding is added or deleted and no rejected mask is automatically accepted.
Contradictory or malformed diagnostics remain visible through `data_issues`.
SDK decode coverage, radiometric parameter validity, field calibration, independent
mask accuracy and registration accuracy are not verified by a database count audit.

The same report is available in the dashboard's **Coverage audit & geometry rejection
reasons** expander. Use separately labelled native masks and source correspondence
controls before changing measurement acceptance based on this queue.
