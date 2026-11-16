# Local ODM reconstruction

ODM estimates RGB camera poses and surface geometry from overlapping photographs. It replaces
the geometry assumptions of the provisional planar preview; sharpening cannot repair those
assumptions. No CAD or panel inventory is required for this Stage 1 reconstruction.

The adapter is tested against the official **ODM 3.6.2 native Windows distribution**.
Install it separately; binaries and private imagery stay under ignored `data/`.
The upstream installer can encode generated configuration files incorrectly when its path
contains non-ASCII characters. Use an ASCII installation path, or rewrite the generated
`venv/pyvenv.cfg` and `cv2/config*.py` files as UTF-8 with the actual installation paths.
The adapter uses Windows short paths for existing Unicode directories because OpenSfM's
native track-file I/O cannot reliably open those paths. If short names are unavailable,
it fails with an actionable error instead of silently continuing.

From the repository root in PowerShell:

```powershell
.venv/Scripts/python.exe examples/build_odm.py prepare --project data/odm/pilot --limit 60
.venv/Scripts/python.exe examples/build_odm.py run --project data/odm/pilot --odm-home data/tools/ODM
```

The pilot selects a central time-ordered block of near-nadir RGB captures, preserving
overlap rather than sampling every Nth image. It is only a pilot, not full-flight coverage.
`--limit 0` prepares all indexed RGB captures. Inspect cameras, turns and overlap before a
full run. Use a new project directory for another experiment; existing results are preserved.

Inputs are independent byte-for-byte copies with EXIF intact, SHA-256 hashes, original
capture IDs and thermal pair IDs in `pvdelta-odm.json`. A file lock prevents concurrent
workers, and input hashes are checked again before execution. `process.log` retains ODM
diagnostics. The native runtime version and exact argument list are recorded. Processing
uses high feature quality, medium dense cloud quality, DSM, a target 2 cm/pixel orthophoto,
and four CPU workers. A requested grid size is not an accuracy guarantee.

Only a zero process exit plus a readable georeferenced RGB GeoTIFF and reconstruction
metadata produce `needs_review`. Missing products and process errors produce `failed`.
The adapter reports component sizes and the largest component's fraction of input images;
these statistics do not prove global accuracy or remove the need to inspect panel seams.

Streamlit → **Orthoview → ODM photogrammetry** displays job state, scope, log, manifest,
and the generated RGB preview. Use **Refresh reconstruction status** after processing.
Full-resolution output is `odm_orthophoto/odm_orthophoto.tif`; `rgb_preview.png` is downsampled
for display. The legacy planar product remains explicitly labelled as unaccepted.

## Acceptance gates and remaining work

### Optional table geometry refinement

`examples/build_odm.py refine` runs a separate experimental adapter using the native ODM
runtime. Supply a private site JSON with `module.length_m`, `module.width_m`,
`module.orientation="landscape"`, and `tables` fields `surface_facing_azimuth_true_deg=180`,
`long_axis_azimuth_true_deg=90`, `surface_tilt_deg`, `rows_along_slope`, and
`columns_along_east_west` (for example `[9,27]`). Existing output directories are refused.

```powershell
.venv/Scripts/python.exe examples/build_odm.py refine --project data/odm/pilot --site data/site-geometry.json --output-name table-refinement-v1
```

The adapter proposes complete rectangles from RGB support, checks their dimensions, and
fits a separate altitude offset to each south-facing plane in true ENU coordinates. Table
spacing stays independent. Nominal module dimensions gate proposals; they do not force
an unverified global scale correction. Partial tables and inconsistent surfaces are withheld.
Sparse point observations inconsistent with the original reconstruction by over 3 pixels
are removed before local registration. A quarter of remaining features is withheld from
each homography fit; median residual must be <=2 px and P95 <=5 px. Surface holdout P95
must be <=0.35 m and qualified image coverage >=99%. These are experimental gates,
not independent geographic or edge accuracy certification.

Original RGB sources are resampled through a local planar mapping with Brown lens distortion.
The mapping is fitted to observed features: global camera poses are not optimized and this
does not calibrate wind-induced yaw. Sources are selected rather than averaged. The output
retains the baseline elsewhere and records a per-pixel source ID (0 = unchanged baseline).
`table_height_enu.tif` stores fitted ENU heights, not surveyed elevations. Temperature data
and native-frame anomaly results are unchanged.

Streamlit's selected reconstruction shows the latest `table-refinement*/report.json` and
its before/after image under **Table geometry refinement — before / after**. GeoTIFF,
source IDs, report and preview are separate products; nothing replaces the original ODM
output automatically. Residual seam and edge review remains required before thermal transfer.

For local deformation, `refine --registration-mode hybrid` first attempts one homography
per table/view, then overlapping spatial mappings for views that fail. A common 25% feature
holdout is excluded from every fit; the final blended coordinate field must still satisfy
median <=2 px / P95 <=5 px. At least 80% of that view's held-out observations must have
spatial support. Unsupported windows remain unmapped. The coordinates are blended;
RGB values from separate photos are never averaged. Table coverage still must reach 99%
and the original 0.35 m surface P95 gate remains unchanged.

If a single camera-orientation group provides sufficient coverage, hybrid source selection
uses that group to avoid opposite image orientations at seams. This groups reconstructed
camera axes, not proven flight passes or measured wind errors. Local deformation and
camera poses are not globally recalibrated. `coverage.png` marks re-rendered areas in green
and withheld candidates in orange. New reports consistently use left/top/right/bottom pixel
bounds. The dashboard experiment selector preserves access to earlier results and reports
used-source residuals, coverage and reasons for withholding candidates.

### Remaining acceptance gates

An optional **satellite layout prior** compares manually observed row outlines with the
local pilot. It is distinct from camera calibration and radiometry. The private observations
JSON records provider attribution, screenshot viewport/centre/zoom, row boxes and explicit
pilot correspondences. Row bands and possible 9/27 partitions are hypotheses, not inventory.

```powershell
.venv/Scripts/python.exe examples/build_odm.py satellite-prior --project data/odm/pilot --observations data/satellite-layout/observations.json --output-name satellite-prior-v1
```

Use `--reference-rgb` to retain a table-refined RGB GeoTIFF on the same original grid.
The adapter fits translation only, with equal weight per row and leave-one-entire-row-out
checks. A separate experimental derivative is produced only for a shift <=5 m, improved
median outline difference and held-out P95 <=3 m. These are exploratory rejection gates,
not absolute accuracy certification. Pixel values, scale and rotation remain unchanged;
the original raster, database positions, camera poses and temperatures are not modified.
Satellite acquisition date may be unknown. Screenshot centre/zoom mapping is approximate.
It cannot resolve local seams or blur, establish surveyed coordinates, or distinguish every
table partition. Keep provider attribution with local screenshots and derived comparisons.

The dashboard exposes the estimated layout, before/after outlines, metrics and downloadable
GeoJSON/report under **Satellite layout — approximate alignment** for the selected pilot.

1. Inspect the pilot: continuous table edges, no repeated or missing panel rows, consistent
   overlap alignment, adequate component coverage, and reconstruction residuals.
2. Reconstruct the full RGB flight and repeat spatial checks across flight strips and turns.
   Keep camera GPS accuracy separate from image alignment residuals.
3. Use reconstructed camera poses and surface geometry with the calibrated thermal-to-RGB
   transform. Validate the transform on held-out pairs across the flight, not only one frame.
4. Project SDK Celsius arrays onto that geometry, preserve per-pixel source frame provenance
   and masks, and compare temperatures against native pixels. A colour JPEG is never used
   to infer temperature. Native-frame anomaly analysis remains authoritative.
5. Display registered RGB/thermal GeoTIFF layers and anomaly locations with uncertainty.
   Independent geographic control may improve absolute accuracy; CAD remains optional Stage 2.

ODM supports some thermal cameras, but its thermal support page does not establish M4T
compatibility. Direct ODM decoding of M4T temperatures is therefore not assumed validated.

Sources: [ODM repository](https://github.com/OpenDroneMap/ODM),
[3.6.2 release](https://github.com/OpenDroneMap/ODM/releases/tag/v3.6.2),
[thermal support](https://docs.opendronemap.org/thermal/),
[orthophoto resolution](https://docs.opendronemap.org/arguments/orthophoto-resolution/).
