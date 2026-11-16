# Development

Run commands from the repository root. Use a locally installed Python 3.10 or newer; CI is configured for Python 3.10 and 3.12 on Windows and Linux. A working Python interpreter can be passed by its absolute path if it is not on PATH.

## Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e '.[dev]'
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m pytest -m 'not sdk' -q -ra
```

The Python launcher must have an installed interpreter. If `py -0p` reports none, use an available interpreter's absolute path for the first command.

Virtual-environment activation is optional. Calling its interpreter directly avoids PowerShell activation-policy changes and ensures pip, pytest and Ruff use the same environment.

## Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pip check
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -m 'not sdk' -q -ra
```

## SDK integration tests

On Windows, use an ignored repository-local temporary directory if the default Temp
directory is not writable. Give each run a fresh directory name, for example
`--basetemp .test-tmp/run-001 -p no:cacheprovider`. Streamlit AppTest additionally needs
a runtime where its local server can start; a stalled sandbox run is not a passing test.

The SDK is not required for the numerical unit tests. Real-frame tests carry the `sdk` marker and require both a local SDK library and an original radiometric sample:

```powershell
$env:PVDELTA_DIRP_LIB = 'C:\path\to\sdk\library-folder'
$env:PVDELTA_SAMPLE_RJPEG = 'C:\path\to\original_T.JPG'
& .\.venv\Scripts\python.exe -m pytest -m sdk -q -ra
```

For this checkout, the local Windows x64 SDK folder is
`sdk\tsdk-core\lib\windows\release_x64`; point `PVDELTA_SAMPLE_RJPEG` at any private
original `*_T.JPG` from the flight folder. The SDK DLL and flight images remain local and must
not be committed. With both variables set, the four real-frame tests run; with either variable
missing, pytest reports four explicit skips by design.

Before running a full flight, verify the local DLL and one private frame directly:

```powershell
& .\\.venv\\Scripts\\python.exe -m pvdelta.cli sdk-check `
  'C:\\path\\to\\private\\DJI_0001_T.JPG' `
  --sdk-lib 'C:\\path\\to\\dji-thermal-sdk\\tsdk-core\\lib\\windows\\release_x64'
```

The command reports the native temperature shape, finite-pixel count, summary values and
the SDK measurement parameters. It never writes the image, temperature raster or private
metadata to the repository. A successful check proves that this local binding can decode the
selected frame; it does not by itself prove field calibration or IEC conformity.

Run `python -m pytest -q -ra` with the environment's interpreter to include both groups. Missing SDK/sample settings produce an explicit skip. Invalid configured paths should fail; do not label such failures as successful skips. The current integration test is a draft M4T-shaped smoke test; full parameter and temperature equivalence checks remain T05 work.

Hosted CI does not install or redistribute the proprietary SDK. It runs the SDK-independent suite and reports the unavailable integration test separately. A green unit-test job does not validate camera decoding, field performance or provisional CoA rules. A local one-frame comparison with DJI's official CLI matched pixel-for-pixel; broader device and field validation remains open.

## Optional Stage 2 inventory import

Use [panel-inventory.csv](templates/panel-inventory.csv) for declared panel centres and topology.
The import keeps missing coordinates as warnings and requires an explicit CRS for GeoJSON:

```powershell
& .\.venv\Scripts\python.exe -m pvdelta.cli import-inventory .\panel-inventory.csv `
  --database .\private\inspection.duckdb
```

The [topology-anchors.csv](templates/topology-anchors.csv) template is for reviewed
observation-to-table/row/position links. Frame-local `V` indices must not be copied into its
`position` column without CAD or manual evidence.

## Data and outputs

Keep field imagery, panel inventories, CAD/GIS files, SCADA data, experiment logs and working notes outside this repository. The public project is for reviewed code, usage documentation and small synthetic automated tests. No real field data is intended for publication, including under `tests/fixtures/`.

The ignore rules also exclude common field-data formats, local outputs and SDK binaries if accidentally placed inside the project. Review the actual files selected for publication: ignore rules do not remove already tracked files and cannot identify private information inside arbitrary JSON or text files. Local trial scripts and experiment histories are not release deliverables.
