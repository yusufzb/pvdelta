# Saved temperature checks

Basic → Panel cards (v2) → Evidence queue now includes:

- **Reference-sensitive**: a card has a region whose fixed reference-deletion
  checks cross the threshold or all fall below it.
- **Minimum pixel support**: a region equals the configured minimum pixel count.
- **Checks missing / stale**: no current saved check is available.

Filters retain entire cards, including sibling regions with different evidence.
These are review priorities, never confirmed fault/healthy labels or physical
panel identities. Existing boundary queues and All candidates remain available.

**Manage saved temperature checks** offers Check missing / stale and Recheck all.
Check temperature evidence on a selected card also saves its fresh result.
Current saved tables open without recalculating the detector. Thermal image
previews may still decode original SDK imagery independently.

Sidecars are stored under `temperature-evidence/` beside the selected inspection
database, scoped by database filename, analysis run and observation. They are
local derived evidence, not finding or review rows. No source images are copied.
Writes use a flushed temporary file and atomic replacement. Failed checks retain
existing evidence files; stale data is withheld when subsequently loaded.

Validity is bound to exact source bytes, the saved measurement profile, capture
geometry/masks and all observation findings, run settings, engine source hashes,
SDK binaries and runtime versions. Those inputs are checked before and after a
fresh calculation. Loaded reports must match their binding and payload checksum.
These checks detect stale or accidentally corrupted files; checksums are not a
security signature against an adversary rewriting local files.

Human review decisions are intentionally excluded from this computational
binding: marking a region Not OK must not force an unnecessary SDK recalculation.
Opening a screen does not create sidecars or write to the inspection DB. Missing,
stale and failed evidence is explicit; it is never interpreted as a healthy panel.
