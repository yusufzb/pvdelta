# Source-only thermal/RGB corner references

`prepare_junction_references` creates a new offline source-reference folder from
explicit `(ThermalFrame, original_rgb_path)` pairs. Original file hashes, native
temperature-array hash, both grid sizes and preview hashes bind every pair.
Thermal images are monotonic structural previews; RGB uses its encoded pixel
orientation. Neither sensor shows detector seeds or predicted registration.
Copy the entire folder with `assets/` when moving the editor.

Select one of12 fixed junction slots, then click the same physical panel-frame
intersection in both images. Zoom and scroll preserve original coordinates.
Avoid cell busbars, shadows and arbitrary hot-region edges. Choose a spatially
spread set: both fit and check points must cover two dimensions. The fixed roles
are8 fit and4 check, assigned before source collection. They cannot be switched
to improve residuals. Missing or uncertain correspondence stays Unclear.

Enter the actual reviewer, mark observed pairs and export JSON. A single complete
source pair can be checked while other pairs remain explicitly pending. Export
retains all pairs and12 slots each. `validate_junction_references` checks source
identities, fixed IDs/roles, reviewer, coordinate ranges and distinct junctions.
`check_junction_references` rechecks actual source bytes/native grid and delegates
completed pairs to the existing8fit/4check/P95≤2 native-pixel gate. Degenerate
layouts or failed point gates are retained as failures; pending points expose no
transform. No scan, mask, finding or identity is changed.

A successful numeric gate does not prove physical correspondence, independent
review, glare clearance or field accuracy. In particular, a filename pair is not
a synchronization certificate. Reported exposure times and camera positions can
differ; those metadata are not a directly measured rig baseline or a calibrated
pixel-motion correction. Source-specific controls are required to investigate
such effects before reusing a single-pair prior across a moving flight.
