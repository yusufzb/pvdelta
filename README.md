# pvdelta

An open-source project for detecting and locating temperature anomalies in photovoltaic modules from DJI R-JPEG thermal images.

The goal is to turn per-pixel temperature measurements into explainable findings with **local ΔT, image evidence and location information**. Analysis should work without a panel inventory. When a plant layout and panel records are available, an optional second stage links findings to real **Panel IDs**, rows, tables and strings.

```
Stage 1 — no panel inventory required
R-JPEG → native temperature grid → module regions → anomaly candidates + local ΔT
       → image evidence + inspection IDs + estimated map locations when possible

Stage 2 — optional inventory matching
Stage 1 findings + plant layout / panel records
       → Panel ID + row / table / string + position within table + match status
```

The first validation target is the **DJI Matrice 4T (M4T)**. **Matrice 30T (M30T)** and other DJI thermal cameras are candidates for later validation. Shared analysis will operate on a temperature grid; decoding, metadata and camera geometry must be checked separately for each model.

> **Status: design and prototype stage (v0.1.0-dev).** The repository contains draft reference-temperature, normalization, classification and SDK-binding code, plus initial tests. It is not yet an end-to-end detector. Automatic module/anomaly detection, geolocation, inventory matching and reporting remain to be built and validated on real imagery. The workflow below describes the intended system. IEC TS 62446-3:2017 informs the method; the current rules are provisional, not a claim of standards compliance.

---

## Why this exists

pvdelta focuses on the radiometric temperature layer rather than relying on the displayed colour palette:

- a palette changes with the pilot's span setting;
- a "red cell" is not a temperature;
- severity is not "how red" — it is **how many kelvin warmer than the local reference, under known irradiance, and what the pattern is**.

A useful finding should state the temperature difference, its reference and the measurement conditions, with an image that lets someone inspect the evidence. Absolute temperature alone is not a diagnosis, and relative ΔT still has measurement uncertainty. Normalized ΔT should be reported only when suitable condition measurements are available.

## Two-stage workflow

### Stage 1: detect and locate without an inventory

The required input is an original radiometric thermal image. Paired RGB images, camera pose, positioning metadata and environmental measurements improve the analysis but do not require a plant asset database.

Planned outputs:

- Module and anomaly regions in the original image, measured ΔT and the local reference used.
- A frame-local ID such as `frame_001/module_007`. Tracking across overlapping frames can later associate observations with one inspection-level module ID; neither is a permanent plant Panel ID.
- An annotated image and links to the original observations.
- An estimated geographic location only when positioning metadata and camera-to-site geometry are sufficient. Image coordinates remain available when geolocation is unavailable.
- Separate measurement-quality, anomaly-assessment and location-quality fields. Missing data must remain visible rather than being treated as a normal result.

The drone's capture coordinates are not the coordinates of every panel in the image. Geolocation needs camera orientation, projection and a suitable surface model or image-to-layout registration. RTK status is useful evidence, not proof that a finding belongs to a particular panel.

Without irradiance or a suitable current reference, retain measured ΔT and mark normalization as unavailable. Without electrical topology, use image-local spatial references rather than assuming neighbouring panels share a string.

### Stage 2: match to a plant inventory (optional)

When available, use panel centre coordinates, panel polygons or a CAD/GIS layout to match detected modules to existing assets. Preserve the source coordinate reference system and explicitly transform coordinates before spatial matching.

Optional inventory fields include:

| Field | Purpose |
|---|---|
| Panel ID | Existing asset identity |
| Centre coordinates / polygon | Spatial matching and recorded panel location |
| Row and table ID | Layout constraints and field navigation |
| Position within table | Disambiguate modules within a table |
| String and inverter ID | Electrical context, separate from spatial adjacency |

For example, a site may have four module rows within each table, labelled **a** at the top through **d** at the bottom. This is a configurable site convention, not a four-row assumption in the core. Define top/bottom in the fixed layout orientation, since the image orientation changes with the flight direction. A column or slot identifier, or the coordinates, distinguishes panels sharing the same within-table row.

Use spatial proximity together with layout consistency and, when available, observations from overlapping frames. Keep anomaly confidence separate from identity-match confidence. An ambiguous match should retain candidate IDs or remain unmatched instead of forcing the nearest Panel ID.

Reports should distinguish an **image-estimated location** from an **inventory-sourced panel location**. A panel centre identifies the asset; it is not necessarily the exact geographic position of the hotspot within that panel. Stage 1 findings remain usable even if Stage 2 cannot find a match.

## Method in one screen

These are proposed method choices backed by draft utilities. Real-frame validation and source verification remain part of the roadmap.

### 1. Temperature
The draft decoder uses the DJI Thermal SDK (`dirp_measure_ex` → float32 °C) through pvdelta's own `ctypes` binding. No third-party parser is vendored. Read the **native radiometric dimensions from the decoded frame** instead of hardcoding one shape for all cameras. Display super-resolution does not define the measurement grid. Extract ΔT on the native grid before any resampling for visualization.

### 2. Reference temperature
ΔT is **signed** and always local:

| Reference | Use |
|---|---|
| module median (robust, excluding the anomaly) | cell / sub-string anomalies |
| neighbouring modules in the same row | whole-module anomalies |
| string / row median | string- and inverter-level anomalies |

Arrays carry several kelvin of natural gradient (wind, edge rows, trackers), so a plant-wide mean is never used.

### 3. Normalization — IEC TS 62446-3 §7.4.1

$$\Delta T_{1000} = \left(\frac{G_{1000}}{G_{meas}}\right)^{x}\,\Delta T_{meas}$$

- `x = 1` for faults covering one cell or more (sub-string, open module, string).
- `x ≈ 1.5–1.8` for small point faults (default 1.65, configurable).
- The standard allows **G to be irradiance or DC current.** pvdelta supports all three bases and keeps them side by side:

| Basis | Source | Notes |
|---|---|---|
| `poa_pyranometer` | thermopile pyranometer in plane of array | reference basis |
| `poa_refcell` | silicon reference cell in plane of array | spectral/angular differences; front-side only |
| `string_current` | SCADA string/MPPT current vs STC current | captures rear-side gain on **bifacial** modules |

When more than one basis is available, pvdelta reports each and flags divergence (default > 10 %).

Measurements below **600 W/m² POA** are marked as outside the standard's conditions. Point-fault extrapolation is refused below 60 % of reference irradiance.

### 4. Classification — pattern first, ΔT second

IEC TS 62446-3 uses **Classes of Abnormality** — severity increases with the number:

| CoA | Meaning |
|---|---|
| **1** | no anomaly / OK |
| **2** | thermal abnormality — investigate, monitor |
| **3** | safety-relevant abnormality — act |

The class table itself has **no fixed ΔT bands**. ΔT figures appear only in the informative examples (Annex, at 1000 W/m²), e.g.:

| Pattern | Typical ΔT | CoA |
|---|---|---|
| open-circuit module / sub-string, uniform | +2…+7 K | 2 |
| single hot cell | 10–40 K | 2 |
| single hot cell | > 40 K | 3 |
| broken glass | any, even 0 K | 3 |

The draft mapping lives in [`src/pvdelta/classify.py`](src/pvdelta/classify.py), with a source and a verification flag per rule. All current rules are marked unverified. Check the source wording, threshold boundaries and applicability before presenting a standards-based classification. The draft classifier accepts a supplied pattern; it does not detect that pattern from an image.

**Industry bins are reported separately and labelled as convention**, never as IEC (e.g. Raptor Maps hot-spot severity < 10 / 10–20 / > 20 K).

### 5. What is *not* a false positive
Soiling, bird droppings and hard shadows can cause real heating. The intended workflow retains such findings with an *external cause* tag when supported by evidence. It separately assesses optical/measurement artefacts such as glint, reflections, cloud transients and motion smear. Automatic cause attribution and artefact rejection are not yet implemented.

## Development setup

These commands install the draft utilities and run their tests from a local source checkout. There is no complete inspection CLI yet. Real-frame decoding additionally requires a separately installed DJI Thermal SDK and a compatible sample image.

See [Development](docs/development.md) for separate PowerShell/Linux commands and SDK-independent versus real-frame test runs.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -m 'not sdk' -q
```

For Windows, use the PowerShell commands in [Development](docs/development.md). The repository is being prepared under the `yusufzb` GitHub account; clone instructions will be added once its URL is live.

Decoding real frames additionally needs:

1. **DJI Thermal SDK v1.8** — download from DJI ([Matrice 4 downloads](https://enterprise.dji.com/matrice-4-series/downloads)). **Do not commit the binaries**; they are under DJI's own license.
2. Point pvdelta at the library folder:
   ```bash
   export PVDELTA_DIRP_LIB=/path/to/dji_thermal_sdk/tsdk-core/lib/linux/release_x64   # Linux
   set    PVDELTA_DIRP_LIB=C:\path\to\dji_thermal_sdk\tsdk-core\lib\windows\release_x64 # Windows
   ```
   DJI ships Windows and Linux x86/x64 builds only (no macOS).

```python
from pvdelta.decode import read_temperature
T = read_temperature("DJI_0001_T.JPG")     # float32 °C, native frame dimensions
```

## Flight notes that actually matter

- **High Gain**, R-JPEG on, **constant gain** over the whole plant.
- Preserve emissivity, distance, humidity and reflected-temperature settings with each observation. Validate the draft SDK parameter reader before relying on those values; report integration is planned. IEA-PVPS reports glass emissivity ≈ 0.85 for viewing angles within 60° of normal; DJI Terra's PV preset is 0.85.
- Resolution: **≤ 3 cm of module per thermal pixel** (IEC TS 62446-3). At 640×512 that is roughly ≤ 28–30 m AGL for M4T and ≤ 22 m for M3T/M30T (flat ground, nadir). Read `CalibratedFocalLength` from XMP rather than trusting published figures.
- Viewing angle: as close to normal as possible **while avoiding reflections** of the sun, the drone and nearby structures; never more than 60° from normal.
- Conditions: POA ≥ 600 W/m², wind ≤ 28 km/h (4 Bft), ≤ 2 okta cumulus, low soiling. After an irradiance change > 10 %/min, wait ~15 min.
- Log POA irradiance (and SCADA string currents if available) with timestamps; normalization needs them.

## Roadmap

This is a high-level product roadmap. Private development notes, experiment logs, field imagery and plant inventories are kept outside the public repository.

### First reproducible experiment — Stage 1

- [ ] Select a small M4T image set with manually reviewed normal regions and anomaly candidates.
- [ ] Validate SDK decoding and measurement parameters against the installed SDK headers and DJI analysis software, including selected points/regions and min/max values.
- [ ] Start with manually marked module regions and a simple local-ΔT anomaly baseline.
- [ ] Export frame-local IDs, measured ΔT, annotated images and CSV findings without an inventory dependency.
- [ ] Estimate geographic locations where metadata and geometry allow; record unavailable or uncertain locations explicitly.
- [ ] Evaluate missed anomalies and false alarms on held-out observations, keeping repeated views of a module in the same evaluation split.

### Optional inventory matching — Stage 2

- [ ] Define a small inventory schema and adapters for site-specific records / CAD/GIS layouts.
- [ ] Match modules using coordinates, row/table structure and configurable positions within tables.
- [ ] Export Panel IDs, candidate matches, match status and coordinate provenance; add GeoJSON for findings with valid geographic geometry.
- [ ] Measure identity-match accuracy, unmatched/ambiguous rates and location error separately from anomaly-detection performance.
- [ ] Validate both stages on the same site subset: withhold the inventory from Stage 1, then use it for matching and manually checked evaluation.

### Further automation and validation

- [ ] Automatic module masks, with optional RGB assistance, and native-grid region statistics.
- [ ] Multi-view module tracking and persistence checks for anomaly and location consistency.
- [ ] Pattern detection: hot cell, multi-cell, sub-string, whole module and junction box.
- [ ] Verify normalization and provisional CoA rules against source guidance and real observations.
- [ ] Investigate reflection rejection using multiple views and sun/camera geometry.
- [ ] Evaluate optional SCADA integration using verified string mappings and suitable current references.
- [ ] Validate M30T and additional camera models independently; compare the same modules under comparable capture conditions where possible.
- [ ] Package the validated workflow as a CLI (`pvdelta scan <folder>`).

## Non-goals (v0)

- Training a CNN on palettized JPEGs
- Replacing I–V curve or EL testing
- Claiming factory-calibration accuracy without a cross-check on *your* airframe
- Orthomosaic temperature analysis (blending lowers hot-spot peaks; orthos are for navigation only)

## Related work

- **[PV-Hawk](https://github.com/LukasBommes/PV-Hawk)** (Bommes et al.) — module detection, tracking and georeferencing from IR video; its [Viewer](https://github.com/LukasBommes/PV-Hawk-Viewer) computes a per-module ΔT (module max − median of neighbours' max). pvdelta's intended focus is DJI still R-JPEG input, local signed ΔT, condition-aware normalization, explainable findings and optional matching to an existing panel inventory. These are development goals, not a claim that temperature-based PV anomaly detection is new.

## Guidance used

- **IEA-PVPS Task 13**, *Review on Infrared and Electroluminescence Imaging for PV Field Applications*, Report IEA-PVPS T13-10:2018 — source for the 2.5 K detection threshold, glass emissivity ≈ 0.85 (≤ 60° from normal), 5–15 min thermal settling after irradiance changes, and PID / delamination patterns.
- **IEA-PVPS Task 13**, *Qualification of Photovoltaic (PV) Power Plants using Mobile Test Equipment*, Report IEA-PVPS T13-24:2021 — drone-specific practice (wind, resolution per cell).

## Contributing

Contributions should contain reviewed code, documentation and synthetic reproductions. Keep original images, GPS metadata, plant layouts, Panel IDs, SCADA records and experiment outputs local; do not attach them to public issues or pull requests. Camera compatibility can be discussed using model/version information and synthetic examples. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Code: [MIT](LICENSE).
DJI Thermal SDK: DJI's own license — not redistributed here.

IEC TS 62446-3 is a copyrighted IEC publication. pvdelta implements its method from the author's reading; it does not reproduce the standard. Buy the standard for normative use.
