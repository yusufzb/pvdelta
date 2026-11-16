# Reviewing candidates and repeat views

The dashboard defaults to **All raw candidates**. The legacy palette shortlist is an
experimental comparison filter; excluded records must not be interpreted as false alarms.
Use Previous/Next in either Stage 1 or User review to navigate the current filtered list.

The RGB surface check compares projected native candidate pixels with the stored RGB panel
polygon, image content and registration uncertainty. It is a consistency check, not independent
validation of the thermal mask. Candidates are retained even when the check requests review.

Open **Same-panel views / glare evidence** and select **Find matching panel views**. Supply
the current flight's site geometry profile (the local default is beside the database).
The implementation includes the original candidate frame and examines up to eight other
paired captures within 30 seconds. This is a bounded neighbourhood, not an exhaustive
whole-flight search.
It uses RGB background features for initialization, exposure-adjusted tracking **inside the
selected panel**, held-out
geometry, unique panel overlap and an LRF panel-centre consistency veto. Site tilt/azimuth
come from the selected profile; thermal FOV comes from the capture's camera-model profile.

Ground and tilted modules must not share one local motion model: parallax between them
can reject a correct panel correspondence. The local fit now uses panel-interior corners.
Its training-inlier hull must cover the inner 70% quadrilateral (2 px tolerance on the
feature grid); the outer frame is a bounded extrapolation. The 3 px held-out P95 limit,
forward/backward tracking, appearance checks, unique overlap and location veto remain.
This is still provisional association, particularly on repetitive cells and uncertain boundaries.

Matching results have Previous/Next controls and filename labels. When association fails,
**Nearby pairs — manual comparison (panel identity unverified)** still displays the source
RGB/thermal pair for that nearby capture. It does not transfer a panel outline, borrow
another module's finding, or assign its temperature to the selected candidate. The panel
may be outside that frame. Failure to match is not evidence of glare or a fault.

The profile contains `tables.surface_tilt_deg` and
`tables.surface_facing_azimuth_true_deg`. A missing profile or unavailable camera/LRF geometry
must not imply a successful match. The 1.5 m location tolerance is not a precision estimate.
Background initialization currently relies on green scene texture; results may be unavailable
on bare-ground sites. This algorithm is provisional and needs independent validation.

Cyan outlines identify corresponding panel hypotheses. Red outlines show only findings
recorded for that target module, enlarged fivefold for display. Thermal boundaries are shown
only when that observation's stored native geometry passed its checks. The annotated
thermal view is decoded with
the SDK on its native grid. If native rendering is unavailable, the embedded preview is
shown for manual inspection with the thermal overlay withheld. No finding does not
mean healthy; an unassessed observation is unknown. Yaw/pitch are camera metadata, not a
computed incidence-angle separation. Pixel-ray directions at the selected panel centre now
provide the estimated angular separation in a common ENU frame. A yaw change at nadir alone
does not establish angle diversity. The pose/FOV model remains provisional.

The summary counts unique captures and requires accepted native geometry and an `analyzed`
observation before using a view as thermal evidence. Missing observations and rejected geometry
are unknown. No recorded finding means only `not_detected`, never a clean/healthy panel or
confirmed glare. Component polygons of the same temperature-difference sign are compared
in panel-relative coordinates, using physical corner correspondence carried through the RGB
match. A different hotspot in the same panel cannot substitute for the selected region.
Unsupported corner correspondence or component polygons remain unknown.

`persistent_across_angles` requires corresponding candidate regions in at least two assessed
views with pixel-ray separation of at least 20 degrees. Both 20 degrees and component IoU
0.15 are exploratory configurable helper thresholds, not IEC requirements or validated field
limits. Angle diversity must occur between the positive views themselves. Negative or
unassessed views cannot manufacture evidence of persistence. Unproven persistence leaves
glare `unresolved`; even persistent candidates still need human and field review.

All temperature values remain from the stored native radiometric analysis. Matching does not
calculate temperatures from RGB or thermal display colours, confirm a fault, or certify compliance.
