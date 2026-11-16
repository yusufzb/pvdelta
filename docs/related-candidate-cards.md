# Related candidate cards

Basic → Panel cards (v2) → **Related candidate cards · provisional** →
**Check related candidate cards** checks up to four nearby paired captures that
contain saved candidates. Retrieval considers up to32 nearest camera positions
within20m in the same acquisition, with the existing time fallback for missing
source GPS. This is a bounded search, not every observation of a physical panel.

The original-flight native/RGB search prior must be available. Original source
and target thermal grids reconstruct their own saved SDK profiles. Native/RGB
anchors use existing mutual bounded observed-quad association. RGB module tracking
must pass in both directions and the reverse match must return to the same
observed source quadrilateral. Cyclic vertex origin/winding does not change
quadrilateral identity; exact coordinates must still match. Existing overlap,
ambiguity, track-support and held-out-error thresholds remain.

The table retains unsuccessful attempts with reasons. Cyan comparison outlines
show observed RGB module geometry. The optional thermal comparison shows unmarked
embedded originals; no temperature or hotspot is transferred. **Open related
candidate card** opens the target's own saved regions and native measurements.
Navigation resets incompatible filters before widget instantiation. Review
decisions remain independent on each observation.

Native/RGB anchors are approximate and repetitive arrays can still cause wrong
associations. Links remain provisional, no physical panels are merged, and no
unique faulty-panel count or glare clearance is inferred. A tracked panel with
no saved candidate is explicit and is not classified healthy. Missing source
anchors remain failures, rather than being assigned to another panel.

Comparison evidence binds original file hashes, analysis/profile/geometry and
candidate membership, calibration/scope files, SDK and engine. Changed inputs or
modified reports invalidate the session result. No inspection data is written.
