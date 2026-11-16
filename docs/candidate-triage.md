# Candidate support queues

Panel cards (v2) separate **Interior candidates**, **Boundary review**, and
**Support unavailable**. All candidates remains available. This is evidence
triage, not false-positive removal, physical-panel deduplication or a diagnosis.

The read-only audit uses exact stored region pixels, never the enlarged display
rectangle. For native-flight runs it reconstructs all capture masks together,
including shared-envelope exclusions, verifies each saved mask hash and pixel
count, then applies the saved detector erosion from the matching run summary.
Touching any effective measurement edge, including an exclusion hole, requires
boundary review. Missing or inconsistent evidence remains unavailable.

The default queue is interior candidates when present. A card retains every
sibling region; mixed cards therefore appear in more than one queue. Per-region
support and reasons appear in the table and its export. Review decisions,
findings, measured temperatures and detection thresholds are unchanged. Older
analyses without the required evidence remain accessible through All candidates.

Interior support does not establish a semantic panel mask, glare clearance or
independent fault confirmation. Real edge/substring faults may occur in Boundary
review and must not be discarded. Unique faulty-panel totals remain unknown
until physical correspondence and fault evidence are established.
