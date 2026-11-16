# Native panel reference editor

In **Panel boundaries → Draw full-panel references (development validation)**,
choose an unresolved `pvdelta.pending-panel-reference-pack/v1` JSON and open the editor.
`PVDELTA_PANEL_REFERENCE_PACK` can provide its path. If a private workspace contains
prepared experiment packs, the latest matching pack is suggested. No samples or labels ship with the package.

Draw four corners in boundary order for each full module. Use **Ignore / partial**
for explicitly ambiguous or partial regions. Zoom is display-only; coordinates remain
on the original thermal grid. Click and drag a corner to correct it; Undo and Delete
remove mistakes. Previous/Next switch between frames inside the editor.

Export the JSON **before changing Streamlit controls or leaving the page**. Work is
not autosaved. The standalone HTML download has the same editor and needs no network.
Use **Resume saved draft** inside the editor to restore an export. Upload that export
in Streamlit to validate all identities, native bounds, convexity, IDs and raster overlaps.

Leave incomplete frames unchecked: they retain `panels:null` and cannot be scored.
Only declare completion after reviewing the whole frame and recording the actual
reviewer. An empty completed frame needs an explicit reason. Completion is a declaration,
not proof of exhaustive correctness. All completed exports are `developer_draft`;
this tool never grants `independently_reviewed` status or measurement authorisation.

**Save validated reference version** re-reads the original pack to reject stale images
or manifests, then writes a content-addressed JSON under the private workspace's
`knowledge basis/panel-reference-reviews`. Prior versions are not overwritten.
The inspection database, finding statuses and radiometric values are untouched.

Use the saved bundle path in **Reference validation (optional)** for a native preview
of the same source. Full-panel IoU diagnostics do not certify anomaly accuracy, physical
site inventory, glare rejection or IEC field compliance. A hotspot box is not a module boundary.
