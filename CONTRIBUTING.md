# Contributing

1. Open an issue first for anything beyond a small fix.
2. Fork → branch (`feat/...`, `fix/...`) → pull request against `main`.
3. `pip install -e ".[dev]"`, then `ruff check .` and `pytest` must pass.
4. Physics or standard-related changes: cite the source (clause, table, paper) in the PR and in the code docstring.

## Data

- Never commit DJI SDK binaries or real field data, including anonymized flight images.
- Keep GPS metadata, panel identities, CAD/GIS layouts, SCADA data and experiment outputs local and outside the repository.
- Public tests and bug reproductions should use generated synthetic data. Do not upload private files through issues or pull requests.
- Publish reviewed, reusable code and documentation; keep local trial scripts, working notes and experiment history outside the public project.

## Style

English for code, comments and docs. Signed ΔT in kelvin, temperatures in °C, irradiance in W/m².
