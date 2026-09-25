# Method notes

Status: proposed method with draft numerical utilities. This document describes the intended workflow; it is not evidence of an implemented detector or standards compliance. See the [project README](../README.md) for scope and implementation status.

## Stage 1 — inventory-independent analysis

```text
original radiometric image
  → SDK decode: native float32 temperature grid + measurement parameters
  → metadata and independent quality checks
  → module regions (manual first; automatic detection later)
  → local and neighbouring-module references
  → signed ΔT and anomaly candidates
  → optional condition normalization and pattern assessment
  → image evidence + frame-local IDs + CSV/JSON findings
  → optional geographic estimates and multi-view tracking
```

An original radiometric thermal image is the primary input. RGB, irradiance, camera pose and panel inventories are not mandatory inputs to the analysis core. Each optional input enables a particular capability; missing data is recorded explicitly.

Analysis uses the native radiometric dimensions read from each frame. Display super-resolution and orthomosaic blending are not measurement grids. Keep masks and temperature statistics in the original thermal pixel coordinates.

Initially, manually annotated module polygons delimit the search area. The detector must produce anomaly candidates itself; supplying a manual anomaly mask is only a measurement example, not a detection experiment. Local references exclude candidate regions when enough valid reference pixels remain. Whole-module heating needs a separate neighbouring-module comparison; the module's own median can hide it.

## Optional normalization

The draft utility implements a configurable power-law correction:

$$\Delta T_{ref} = \left(\frac{G_{ref}}{G_{meas}}\right)^x\Delta T_{meas}$$

The proposed exponents and applicability limits must be checked against the relevant source guidance and the measurement setup. Record the driving quantity, units, reference value, exponent, timestamp and provenance. An appropriate current reference is an optional basis; an anomalous or offline string's current must not silently become an irradiance proxy.

Without valid and time-aligned conditions, retain measured ΔT and report normalization as unavailable. Unknown pattern, missing conditions and low-quality measurements are not evidence that a panel is healthy. Current CoA rules are provisional; the draft classifier accepts supplied patterns rather than detecting them in an image.

## Location and identity

Every finding has an image location. Geographic estimates additionally require usable camera pose, calibration and a surface model or independently validated registration. Preserve coordinate systems, axis conventions and altitude references. Capture coordinates must not be reported as panel coordinates.

RGB-assisted registration can be added when paired imagery exists. A plane model is a candidate for a sufficiently planar site subset, not an assumption that all tables and terrain share one plane. Validate registration on independent control points and explicitly record where it is unavailable.

Multi-view tracking is planned to link observations of the same module and assess persistence. Repeated panel patterns can cause incorrect associations, and persistence alone does not establish that a feature is a real fault. Keep individual observations and aggregate statistics separately.

## Stage 2 — optional inventory matching

Stage 1 findings may be matched to a plant inventory containing panel centres or polygons, a declared CRS, Panel IDs and optional row/table/string/slot fields. Use spatial candidates together with layout consistency; ambiguous matches remain ambiguous.

Within-table row and column labels are configurable site conventions. Define orientation in the fixed layout; flight/image rotation must not change those labels. Spatial adjacency is distinct from electrical string membership.

Stage 2 preserves the original Stage 1 results and adds asset matches. Report image-estimated geometry separately from inventory-sourced panel centres, and distinguish anomaly assessment from location quality and identity-match confidence. Findings without usable geographic coordinates remain in CSV/JSON even if they cannot appear on a map.

## Verification sequence

1. Compare real SDK temperatures and parameter interpretation against the installed SDK headers and DJI analysis software using the same settings.
2. Evaluate a small local-ΔT baseline with manually defined module regions and held-out observations.
3. Measure location errors on independent controls and track association errors separately.
4. Evaluate optional Panel ID matching with manually verified asset identities.
5. Add automatic masks, patterns, reflection checks, SCADA and additional cameras only with separate evidence of their effect.

Numerical thresholds and camera assumptions require source verification and local validation. No aircraft-specific GSD or alignment constant is treated as universal.

## Reference material to verify

- IEC TS 62446-3:2017, outdoor infrared thermography of PV modules and plants.
- IEA-PVPS Task 13, *Review on Infrared and Electroluminescence Imaging for PV Field Applications* (2018).
- IEA-PVPS Task 13, *Qualification of Photovoltaic (PV) Power Plants using Mobile Test Equipment* (2021).
- Bommes et al., *Georeferencing of Photovoltaic Modules from Aerial Infrared Videos using Structure-from-Motion* (2022).

Record exact edition, section and applicability when a source is used to justify a production rule.
