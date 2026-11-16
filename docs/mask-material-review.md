# Mask material review

Basic and Advanced show a **Mask material review** section when a prepared local
review pack is available. Open the section, enable **Open material review** and
enter the actual reviewer name. Use Previous/Next to select a source region.

Classify the pink native pixels as **Panel glass**, **Frame / gap** (including
clamps and nonpanel ground), or **Unclear**. Each button saves the decision and
note directly on the server, so a phone session does not need to export/upload
JSON. Reset returns an item to unreviewed. Unclear does not resolve the material;
unreviewed never means nonpanel or healthy.

The crop is a display enlargement. Temperature statistics come from the prepared
source-bound native SDK data under its saved parameters; they are not independent
field measurements. Paired RGB is unregistered context, without a point overlay.

Decisions are separate material descriptions. They do not alter original coarse
panel labels, masks, findings, temperatures, model acceptance or the inspection
database. They do not approve a fault, glare verdict or IEC compliance.

The pack binds source, native-temperature, target-pixel and image hashes. Modified
assets, stale packs and mismatched source/pixel identities are rejected. Exact
exports and validated content versions are retained in the pack's
`operator-reviews` directory. Re-selecting an older answer survives reloads;
concurrent saves in one dashboard process merge the same reviewer's decisions.
An atomically replaced commit head selects the current immutable version for each
reviewer; equal or backwards filesystem timestamps cannot select an older answer.
Interrupted commits retain the previous head. Existing versions are migrated on
their first save. A damaged head is reported instead of silently reverting history.
This coordination covers sessions in one server process, not multiple independent
server processes writing the same pack.
Different reviewer names retain separate review sets. A reviewer name is a
declaration, not authenticated independent-review certification.

The dashboard discovers local `knowledge basis/stage1-*/material-review-pack.json`
files. Advanced also accepts a pack path. `PVDELTA_MASK_MATERIAL_PACK` overrides
the default. No private pack, images or review answers belong in the public repo.
**Download material review JSON** remains available for local export/resume tools.
