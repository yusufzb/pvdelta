# Experimental XT2 / FLIR FFF reading

`pvdelta.flir_fff.read_flir_rjpeg` reads native uint16 sensor counts from FLIR
FFF records in JPEG APP1 chunks. It is an explicit research adapter; the DJI
`RJpeg` decoder and production/dashboard routing are unchanged. A DJI SDK
rejection is not silently converted into successful surface radiometry.

```python
from pvdelta.flir_fff import read_flir_rjpeg

record = read_flir_rjpeg(source_path)
native_counts = record.raw_counts            # read-only uint16 grid
apparent_c = record.apparent_temperature_c()  # read-only float64 grid
```

The adapter verifies chunk numbering/completeness, FFF directory bounds,
nonoverlapping records, raw dimensions, byte-order markers and per-image
Planck coefficients. Container, raw-data and camera-record byte orders are
independent. Only uncompressed big/little-endian uint16 records are supported.
PNG/compressed raw variants, missing/invalid coefficients and impossible
conversion domains fail explicitly. There are no palette-derived temperatures,
default camera coefficients, byte-order guesses, resizing or pixel imputation.

The conversion is the inverse embedded blackbody calibration:

```text
T_apparent_C = B / ln(R1 / (R2 * (raw + O)) + F) - 273.15
```

These are apparent blackbody temperatures, not independently validated panel
surface temperatures. Recorded emissivity, object distance, reflected and
atmospheric temperature, humidity and window parameters remain available in
`observation_settings`, but no environmental/surface compensation is applied.
A recorded distance of zero does not establish flight height or physical
camera distance. Do not substitute the current site's module settings for an
unrelated public flight without evidence.

Retain the measurement basis and source/native hashes with any exported
candidate. Apparent-temperature screening results must not be merged with
SDK surface-temperature results as if their basis and calibration were equal.
Field calibration, operating conditions, fault causes, glare and IEC acceptance
remain separate requirements. Do not enable production routing solely because
the payload can be decoded.

The format layout was checked against [ExifTool FLIR.pm](https://github.com/exiftool/exiftool/blob/master/lib/Image/ExifTool/FLIR.pm)
and its [JPEG APP1 handler](https://github.com/exiftool/exiftool/blob/master/lib/Image/ExifTool.pm).
The Planck inversion is also documented by the author of
[Thermimage raw2temp](https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R).
This adapter is independently implemented; that project's environmental
correction algorithm is not used.

# Published LabelMe regions

`pvdelta.labelme_regions.parse_labelme_rectangles` imports explicit rectangle
labels while preserving the supplied class name. It never opens `imagePath` or
decodes `imageData`. The caller verifies the source basename, native dimensions,
orientation and source/reference-image correspondence before scoring. Reverse
opposite corners are ordered; unsupported shapes, nonfinite/out-of-frame
coordinates and zero-area boxes fail as a whole rather than losing references.

An explicit empty shapes list differs from a missing label. An unlabelled photo
is not a negative or a healthy module. Labels such as `F1` are the publisher's
region labels, not full-panel boundaries or a validated electrical fault type.
Keep missing raw counterparts and embedded-image alignment failures separate
from detector false negatives. A rendered PNG embedded in an orphan label
cannot supply the missing native radiometric counts.

**Preview/native correction:** Matching the label's embedded image to the JPEG
does not establish alignment with the embedded native array, even when both are
640×512. A separate geometry check is required. See
[preview/native alignment](preview-alignment.md) for the conservative translation
API, rejection rules and coordinate-only reference mapping. Do not use the
unverified identity assumption from historical XT2 pilot scores as validation.
