# Review versions

Basic mode offers **Review version** above the result tabs:

- **Panel cards (v2)** groups all candidate regions for one saved panel observation into one thermal/RGB pair. A1, A2 and subsequent labels correspond between the images and the region table. Each region retains its original finding ID, temperature, reference, delta T, and review decision.
- **Individual anomalies (v1)** keeps the previous single-region review workflow.

Advanced mode retains its existing tabs and adds **Panel review (v2)**.

Grouping uses the exact observation ID within the selected analysis run. Repeated local module names in different frames are separate cards. These are panel observations, not a verified count of unique physical assets. Cross-frame identity still requires validated correspondence; CAD is not required for this view.

Previous/Next moves between cards. Review filters select whole cards, retaining all regions on each selected card. In **User review**, apply OK / Not OK / Wait to all regions together or select one A label. Group updates are transactional and reject changed group membership. An empty note preserves existing notes. Reviews do not change detector findings or certify faults.

Thermal candidate locations require the original indexed source hash and the SDK's
native grid, never a resized embedded JPEG. When a saved measurement profile exists,
its effective parameters and native temperature hash must also reproduce exactly.
Older analyses without a profile can show **saved candidate locations only** after
source/grid verification; historical temperatures remain explicitly unverified.
An invalid or changed existing profile cannot enter this legacy mode. Source/SDK/grid
failures still leave the photo unmarked.

RGB overlays retain the source, camera, flight, image-size and registration gates;
the original-flight projection remains labelled approximate. Missing historical
radiometry alone does not erase source-bound geometric locations. Local registration
and temperature-evidence checks still require the saved profile. Unavailable RGB
mappings are reported per region rather than silently dropping findings. Red outlines
use a 6x display enlargement (20% wider than the previous 5x) and do not represent the measured extent.
Region measurements, findings and review decisions are unchanged.

## Other photos and glare review

Enable **Compare other photos** below the panel's records, then choose **Region for comparison** (A1, A2, etc.). The existing **Same-panel views / glare evidence** controls are available directly from the V2 card. Evidence remains specific to the chosen finding; switching regions or panel cards does not display the previous selection's results.

For the current native thermal analysis, **Check native frame geometry** provides source-bound SDK and held-out structural diagnostics without legacy RGB polygons or a site geometry file. See [native checks and their limits](native-repeat.md). These diagnostics do not authorize physical identity or anomaly transfer.

**Browse GPS-nearby image pairs** opens the source and up to eight paired photos selected by horizontal camera distance within 20 m in the same flight, including later return tracks. It works without an SDK or site profile. Filenames, horizontal distance, time gap, yaw/pitch and Previous/Next are shown. Images remain unmarked; camera proximity and yaw change alone do not prove the same panel or viewing-incidence diversity. Missing source GPS uses the time search and is labelled explicitly. Changing region uses a separate navigation state.

**Find observed RGB panel views** adds local tracking between directly observed RGB module boundaries when a compatible original-flight search prior and SDK profile are available. Passing targets receive a cyan RGB outline; paired thermal photos stay unmarked. The source native association remains approximate, so no thermal component, temperature or glare verdict is transferred. See [source binding and interpretation](observed-rgb-repeat.md).

**Browse nearby image pairs (unverified)** works without a site geometry profile. It includes the source and up to eight time-near paired captures within 30 seconds, with filenames and Previous/Next navigation. These unmarked photos may contain different panels; proximity alone establishes neither physical identity nor angle diversity. Missing timestamps or RGB pairs can reduce the available set.

**Find matching panel views** additionally requires a site geometry profile and passing existing correspondence/geometry checks. A match is provisional, and a failed match does not establish glare. The search is bounded, not an exhaustive list of every photograph of the physical panel. A result for A1 does not resolve A2 or the whole card.

Comparison runs only after an explicit button press. It does not alter findings or review decisions. An unavailable comparison leaves the primary panel image pair and region table visible. The configured SDK is passed to the native evidence renderer; embedded JPEG fallbacks remain unmarked where native geometry cannot be validated.

## Local checkpoints

Before enabling V2, preserve source files, inspection databases, configuration, and the separate unassigned-evidence review bundle in a local checkpoint outside the public repository. Retain a SHA256 manifest and a frozen launch script on a separate loopback port. Raw images, virtual environments and SDK binaries remain in their original local locations. Never publish a private checkpoint.

V1 and V2 use the same active database when selected in the current dashboard; their review decisions are shared. The frozen V1 launcher instead uses separate runtime copies of its saved baseline database and review bundle.

Manual browsing is labelled **panel matching not requested**. It is not reported as a failed automatic match.
