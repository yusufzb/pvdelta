# Whole-flight SDK scan

This step checks original DJI radiometric decoding on every discovered thermal file. It does
not detect panels, establish their unique physical count, confirm a fault or certify field
measurement conditions. CAD and a panel registry are not required.

```powershell
pvdelta scan-flight .\flight --sdk-lib .\sdk\tsdk-core\lib\windows\release_x64 --out .\private-scan
pvdelta scan-flight .\flight --sdk-lib .\sdk\tsdk-core\lib\windows\release_x64 --out .\private-scan --resume
```

Discovery recursively selects `_T.JPG` and `_T.JPEG`, ignoring case. RGB files are excluded;
the naming convention is only a discovery rule. Actual SDK decoding determines support.
Each file has an explicit decoded/failed outcome. A corrupt image or a NaN/Inf native grid
does not stop later images, and failed files remain unassessed. An entirely finite float32
native grid is required for a successful validation record.

The private `scan-summary.json` contains the native grid hash, dimensions, temperature summary
and all five recorded and effective SDK parameters, including humidity. It also records the SDK binaries,
SDK INI files, engine hashes and runtime. Source temperatures are never resized or warped.
This records the parameters used; it does not prove they match actual field conditions.

## Explicit corrections

Recorded radiometric distance is not necessarily the flight altitude or the recorded LRF range.
Check it before treating absolute temperatures as field-validated measurements. To calculate
a separate profile, provide a JSON file with only the values you explicitly intend to change:

```json
{"distance_m": 19.0, "emissivity": 0.95}
```

```powershell
pvdelta scan-flight .\flight --sdk-lib .\sdk\tsdk-core\lib\windows\release_x64 `
  --measurement-params .\measured-conditions.json --out .\private-corrected-scan
```

The numbers above illustrate syntax, not a recommended distance for any flight. Other accepted
keys are `humidity_pct`, `ambient_temp_c` and `reflected_temp_c`. Unspecified values remain
recorded values. Limits come from `dirp_get_measurement_params_range` for each R-JPEG; unsupported
values fail rather than being silently clamped. The SDK's `dirp_set_measurement_params` and
`dirp_measure_ex` perform the new calculation on the original native grid. No empirical offset
or palette-based temperature estimate is used. Recorded/effective values and native hashes are
kept separately; overrides also participate in cache identity.

Do not replace the whole flight's distance with one laser observation automatically: the LRF
target and the panel under inspection must correspond. Likewise, reflected apparent temperature
is a separate environmental input; an embedded value equal to ambient does not verify it.
Supplying a profile is an explicit assumption/input, not proof of field conditions.

The `frames/` journal writes successful outcomes atomically. Explicit resume reuses only
intact successes bound to identical original file bytes, SDK and engine. Changed inputs,
changed SDK or changed engine are decoded again; failed files are always retried. A resumed
run distinguishes fresh decodes from reused records. This is a local reproducibility cache,
not an independent certificate or protection against a malicious cache author.

An interrupted operation leaves `status: running`, replacing an older complete summary.
Resume processes the currently discovered files and reconstructs the summary; removed files
are not counted just because their historical cache files remain. Use one process per output
folder. Inputs and the source database are never modified by this CLI.

Exit codes: `0` all files decoded, `3` some failed, `1` all failed or a global/input error.

In Streamlit, **Start a local radiometric scan** uses the same engine. It also updates summaries
for uniquely indexed thermal source hashes. Unindexed/ambiguous records and SDK failures are
reported separately. Download the JSON for every file's outcome. The local journal lives beside
the selected database under `radiometric-scans/`; it contains private source names and paths.
Corrected profiles are stored separately and do not overwrite capture summaries or activate
existing findings calculated with a different parameter set. Rerun module analysis with the
selected profile and source-bound geometry before comparing anomaly results.

SDK-readable frames, frames with usable panel masks, analyzed panel observations, and unique
physical panels are separate quantities. A full SDK pass does not close Stage 1 detection acceptance.
