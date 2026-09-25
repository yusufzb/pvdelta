# Changelog

## [Unreleased] — 0.1.0.dev0

- IEC TS 62446-3 CoA 1–3 scheme replaces the earlier "<2 / 2–10 / ≥10 K" table (that table is industry convention, not IEC).
- §7.4.1 normalization with exponent by fault extent; three bases: POA pyranometer, POA reference cell, SCADA string current; divergence flag.
- Signed local ΔT (module / row / string median).
- Own `ctypes` binding for DJI Thermal SDK `dirp_measure_ex` (unvalidated on hardware).
