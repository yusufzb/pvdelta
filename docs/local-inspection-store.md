# Local inspection store

pvdelta uses DuckDB as a local analytical index. The database, original imagery, temperature
arrays and real coordinates are ignored by Git and must remain outside public examples.

## Data layout

| Table | Purpose |
|---|---|
| `inspection_runs` | One import or processing run and its local source root |
| `captures` | Thermal/RGB file identity, EXIF/XMP pose, radiometric settings, temperature summary and optional footprint |
| `module_observations` | Frame-local module polygon, temperature summary and optional estimated location |
| `findings` | Candidate/reviewed anomaly, ΔT, temperature, confidence and optional location |
| `finding_reviews` | Human OK / Not OK / Wait decision, reviewer, note and timestamp |
| `measurement_profiles` | Analysis/capture-specific SDK settings, native matrix hash and result identity |
| `invalidated_finding_reviews` | Previous decisions archived when their analysis result or measurement profile changes |
| `native_boundary_snapshots` | Immutable source/native-temperature/engine-bound geometry review result |
| `native_boundary_review_events` | Append-only visual boundary decisions, notes and reviewer history |
| `native_boundary_review_reasons` | Optional boundary issue codes attached to each review event |
| `asset_matches` | Optional Stage 2 match to a plant Panel ID |
| `view_rankings` | RGB candidates ranked within an explicit panel key, with linked thermal capture and coordinates |

Schema v2 adds an explicit `analysis_run_id` to module observations and findings. Findings retain
their reference method, pixel counts, normalization provenance, exact sparse region support and
`review_required` state. `write_stage1_results()` atomically replaces one analysis-run/capture
slice, so a rerun cannot leave stale findings in the database. Schema v5 adds the separate
`finding_reviews` table so human decisions never overwrite detector output.

Schema v6 adds separate boundary snapshots and review events. Snapshots contain the
preview report, not images or dense thermal arrays. Their content identity binds
source bytes, native temperature hash, engine hashes, geometry and settings; a local
session token is excluded. Changed results never inherit an earlier OK decision.
An event sequence preserves ordering; later OK / Not OK / Wait decisions append
to history instead of overwriting it. Read-only access to older stores returns an
empty boundary review list until the store is migrated by a write-capable open.
Visual boundary acceptance does not change findings, anomaly review decisions,
measurement authorisation or independent validation status.

Schema v7 adds an event/reason junction table. A decision and its deduplicated
reason codes are saved in the same transaction. Existing snapshots, events and
notes remain unchanged; read-only v6 access returns an empty reason list without
migration or inferred labels. Not OK / Wait can carry optional visual error tags
(missing panel, vegetation/ground selection, merged/split panels, misalignment,
incomplete coverage, insufficient view or other). OK carries no active tags;
earlier issue tags remain attached to their historical events. These are boundary
review reasons, not electrical fault diagnoses.

Schema v8 adds per-analysis/capture measurement profiles and result fingerprints.
Different results archive existing finding decisions, clear active stale evidence
and replace the slice in one transaction. Exact reruns retain decisions. Capture
metadata keeps its source-level meaning; corrected analysis temperatures belong
to the analysis profile. Legacy read-only stores return no saved profile instead
of inventing one. See [measurement profiles](measurement-profiles.md).

Original thermal matrices are stored as external `.npy` files initially and referenced by
`temperature_array_uri` plus `source_hash`. A later Zarr adapter can improve chunked access
without changing the relational model. Per-pixel long-form rows are deliberately avoided.
Sparse finding support is the exception: only the exact pixels belonging to a reported candidate
are stored as JSON evidence. The full radiometric matrix remains external.

`camera_latitude` and `camera_longitude` locate the aircraft at exposure. They do not locate
every module visible in the image. `footprint_geojson` remains null until calibrated camera
intrinsics, orientation and a surface model or verified image-to-layout transform produce a
defensible coverage polygon. The record also carries its method and estimated accuracy.

The schema is created without loading DuckDB's spatial extension. Geometry is stored as
GeoJSON so the core remains offline and reproducible; Folium reads the same geometry for local
visual review.

## Folder indexing

The dashboard and `pvdelta.ingest.index_flight_folder` can index a local DJI flight directory
in place. The importer recursively discovers `*_T.JPG` and `*_V.JPG` files, hashes each source,
parses available EXIF/XMP metadata and stores relative paths. It neither copies nor uploads the
imagery. Re-running the same import is idempotent.

Thermal and RGB files are paired by their parent directory and DJI sequence number. The
timestamp portion of paired filenames is not required to match because observed DJI pairs can
differ by one second. A filename pair remains only a capture association; it is not proof of
pixel registration.

`view_rankings` is populated by `pvdelta rank-views`. Its input must provide an explicit
`panel_key` from reviewed geometry, CAD or a validated tracker. The score chooses a convenient RGB
inspection view; it does not select a thermal measurement and it does not infer that two captures
show the same physical panel from filenames alone.

The local map draws the ordered thermal camera positions as a flight track. Recorded laser
target points can be enabled separately. Neither geometry is called a panel location or image
footprint. Coverage polygons appear only after a calibrated method and its accuracy have been
stored.
