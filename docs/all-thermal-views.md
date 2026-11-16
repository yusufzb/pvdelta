# Review nearby thermal views without candidate-selection bias

In V2 open **All nearby thermal views · provisional**. Select a retrieval limit
(32 by default, or16/8) and choose **Check nearby thermal views**. Candidate
presence is never a target-selection input. Retrieval uses the existing bounded
camera-GPS/time search within the acquisition; it does not prove that every view
of the physical panel has been found.

Each target retains its outcome, including unattempted source-anchor failures,
failed RGB/anchor checks, missing native analysis and successful provisional
matches. Both RGB tracking directions must return to the original observed panel.
Each matched native panel must reproduce its own complete saved hot-region
population with its own original SDK matrix, measurement profile, detector
settings and support mask. An empty saved population must also reproduce; it is
reported as `analyzed_no_candidates`, never healthy or glare-cleared. A failed
check remains unknown. Existing candidate/review rows are not changed.

Older databases may lack native measurement profiles or their matching
`summary.json` settings. The check then retains nearby paired captures and their
actual saved candidate counts for manual browsing, with an explicit explanation
that automatic thermal replay is unavailable. No default settings are guessed,
and no files from another run are borrowed. Photo pairs open automatically in
this unavailable state; unmatched thermal images remain unmarked. This is not
same-panel, negative thermal or glare-clearance evidence.

Results persist in separate `thermal-view-scans/` sidecars, bound to current
source bytes, run, geometry, complete memberships, settings, SDK/runtime and
engine. Changed/corrupt evidence is withheld. Previous/Next retains rejected
views. **Show selected thermal/RGB pair** explicitly decodes selected original
SDK evidence. Green marks each capture's own saved hot pixels; cyan marks its
provisional panel. Both images use a common numeric scale and retain their own
camera orientation. Failed targets show unmarked original thermal images.

The native/RGB anchors remain approximate. Neither candidate persistence nor
absence, photo count, camera proximity or yaw change confirms a physical hot
cell, adequate angular diversity, an electrical fault or absence of reflection.
This workflow fixes a selection bias; it does not establish association accuracy.

# DJI reference integration check

`pvdelta.sdk_reference.compare_dji_cli` compares native float32 matrices against
the locally installed DJI `dji_irp` utility. All five effective saved parameters
are passed explicitly. Nonzero exit status, missing/truncated output or nonfinite
values are errors. The original R-JPEG is never changed. SDK binaries and private
test images remain local.

Exact agreement validates the Python integration against DJI's implementation
using those parameters. It does not validate the environmental inputs or field
temperature accuracy. In the current original flight all1239 saved profiles use
5m while recorded laser ranges span15.417–24.104m. Laser range is to its recorded
target, not automatically every panel's range. A fixed eight-frame distance-only
scenario changed full-frame medians by1.038–2.086K, including non-panel pixels;
it is not a panel-temperature correction or an uncertainty bound. Profile changes
require explicit assumptions and a new analysis rather than relabeling old results.
