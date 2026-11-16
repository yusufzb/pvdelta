# Inspection handover

Export a stored analysis run without changing the database:

```powershell
pvdelta export-inspection --database inspection.duckdb --run-id my-analysis --out handover
pvdelta verify-inspection-export handover
```

Choose a new output directory. Export reads a consistent database transaction,
writes and verifies a temporary bundle, then publishes the complete directory.
It never overwrites an earlier handover or migrates the input database.

Open `handover/review.html` for coverage and candidate cards. All regions belonging
to one frame-local observation appear together. `inspection.json` contains the
complete export; CSV files contain frames, module observations, findings with
operator reviews, and effective measurement profiles. `coverage.json` records
missing screening records and detected inconsistencies. `manifest.json` stores
file hashes so a copied bundle can be verified without the original database.

Thermal/RGB source names and hashes, camera coordinates, estimated finding
coordinates and location methods remain separate. RGB names are included only
for stored same-acquisition pairs; the exporter does not guess a correspondence.
Original photographs, temperature arrays, the SDK and the database are not copied.
The exported records still contain private site information and review notes;
keep this directory local unless you explicitly intend to share it.

Counts are frame-local observations and candidate regions, not unique physical
panels or confirmed faults. A missing candidate does not mean healthy. User
`ok`, `not_ok` and `wait` decisions remain separate from detector assessments.
Missing review/profile tables remain unknown; the export does not create them.
Unknown physical totals and field acceptance are never filled from observation
counts. Export verification checks byte integrity, not detection accuracy or IEC
compliance. JSON preserves text exactly; CSV neutralizes spreadsheet formula
prefixes. Unsupported schemas or non-finite numeric records fail explicitly.
