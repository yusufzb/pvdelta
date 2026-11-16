# Native thermal flight analysis

`scan-native-flight` connects the existing observed thermal-boundary detector to
source-bound reports and a new private DuckDB. It can inspect original thermal
captures without waiting for an RGB registration or a plant inventory.

```powershell
pvdelta scan-native-flight --database .\private\source.duckdb `
  --acquisition indexed-flight --run-id native-review-001 `
  --out .\private\native-review-001 --workers 2 `
  --sdk-lib C:\path\to\dji-sdk\release_x64
```

First index the original flight. The command reads every indexed thermal capture
in the chosen acquisition, including frames with no panel proposals. It checks
original source hashes and freshly decodes native temperatures with the DJI SDK.
The source database stays read-only. Outputs and original imagery remain local.

The detector uses `native_window_proposals(..., raw_supplement=True)` with its
existing geometry and observed-side gates. No panel grid, CAD slot, RGB warp or
reference annotation fills a missing boundary. Default size/aspect assumptions are
experimental and require checking for different cameras, altitudes and arrays.
This conservative structural detector can miss full panels and admit other
rectangular surfaces. It does not establish semantic panel membership.

All proposed envelopes are retained in the native coordinate grid. Measurement
support excludes a two-pixel margin and every shared envelope pixel from all
instances, with no nearest-centre ownership rule. Temperatures are never warped,
blended or offset after SDK measurement. Local hot-region detection uses the
existing 2.5 K / four-pixel configuration, requires at least 24 reference pixels,
and keeps every finding `review_required`. These are exploratory settings.
Whole-module peer groups are not inferred from proximity.

The output contains `inspection.duckdb`, `summary.json`, an incremental
`frame-outcomes.jsonl` journal and per-frame JSONL/CSV/SVG reports plus
`native-geometry.json`. Each successful analysis retains its exact
[measurement profile](measurement-profiles.md) and native mask hashes.
`--measurement-params` accepts an explicit fixed SDK override JSON; recorded
settings remain separately available. No automatic LRF distance correction occurs.

`--workers` accepts 1–4 processes. Workers read originals; a single parent owns
database writes, with bounded pending results. A failed source or geometry remains
unassessed while later files continue. The summary starts as `running` and records
the completed outcome. Exit codes are 0 complete, 3 partial and 1 failed. An
existing output directory or run ID is refused; automatic resume is not provided.

In Streamlit, open **Radiometric scan → Native thermal boundaries & anomaly
scan**. Open the resulting database in **Local data & map**. The coverage audit
recognises native structure and does not invent RGB registration scores. Expand
**Frame results and paired boundary previews**, select a frame and press **Show
native boundaries and measurement support** to inspect all proposals, including
frames without findings. The preview reconstructs the saved native temperature
hash and verifies each displayed mask against its stored hash before drawing.

Fine RGB hotspot transfer remains withheld without validated registration. The
**Browse nearby image pairs (unverified)** button is available even without a
site profile or reliable same-panel mapping. It shows the source and up to eight
paired captures within ±30 seconds for manual comparison, with Previous/Next.
No transferred outline, temperature, panel identity or glare verdict is inferred.
Unavailable timestamps restrict comparison to the source instead of guessing.

For a frame with missing boundaries, **Review unresolved thermal regions** reruns
the existing native side-support diagnostics on the exact saved SDK profile. It
retains review hypotheses with observed support and neighbouring panel context,
without relaxing accepted-panel gates. Results open in **Unassigned evidence** as
a separate immutable bundle. They carry no panel ID, panel GPS or confirmed-fault
status and do not enter module/fault/healthy totals. Multiple evidence records may
refer to the same physical region. Changed source, native data or re-extracted
baseline geometry prevents export instead of reusing stale support.

An analyzed observation is a numerical result inside a provisional ROI, not a
verified panel measurement or a confirmed healthy module. Zero proposals and zero
findings do not establish a healthy frame. Repeated observations are not unique
physical panels. Independent full-panel references, same-panel correspondence and
field measurements remain separate acceptance requirements. Legacy geometry
replay is deliberately unavailable for native-only runs; re-extract them with
`scan-native-flight` instead of applying irrelevant RGB gates.
