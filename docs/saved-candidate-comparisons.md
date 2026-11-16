# Persistent related-card comparisons

Related candidate cards now reopens its saved comparison automatically. An
explicit Check related candidate cards reruns the selected card. Check all
missing / stale comparisons processes the candidate-card population and resumes
completed work; Recheck all comparisons calculates every card again.

Two local worker processes are used by default, with bounded submitted jobs and
one OpenCV thread per worker. Each worker reads originals and writes only its
own atomic comparison sidecar. The inspection database, temperature findings
and user decisions remain separate. A stopped run retains completed records.

Records live in `candidate-comparisons/` beside the inspection database. Both
successful reports and unavailable source attempts are retained. An unavailable
source has an explicit reason and does not indicate a healthy panel. Changes to
source files, profiles, geometry, candidate membership, calibration, SDK or
matching code invalidate the input binding. Corrupted reports are withheld.
Bindings are checked before and after computation; changed inputs prevent saving.
The payload checksum detects accidental corruption, not malicious rewriting.

Batch resume can share a file-hash memo only within the current validation pass.
It still hashes current original bytes and never reuses timestamps as proof of
unchanged content. Individual card opening validates its own current binding.
Persistent comparison loading does not rerun RGB tracking or native-temperature
checks; original image previews still verify photo hashes.

The current comparison scope remains up to four candidate-containing captures
from32 retrieved neighbours. It does not enumerate every thermal observation of
a physical panel. Repeated directed links are not unique fault counts. No
transitive asset grouping, physical merge, anomaly/temperature transfer, copied
review decision or glare clearance is performed. Original-flight M4T search-prior
requirements and the existing two-direction geometry gates remain in force.
