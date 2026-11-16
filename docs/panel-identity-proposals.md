# Experimental panel-pair evidence

`panel_identity_proposals` evaluates already observed panel boundaries across
thermal display images. Camera GPS is used by the local trial runner to retrieve
nearby candidate-bearing frames within the same acquisition. It does not provide
panel coordinates or establish identity.

Descriptor matching uses full-frame mutual SIFT matches with ratio0.70 and
duplicate coordinates removed. Fixed source spatial cells reserve check matches
before forward and reverse homographies are estimated. At least12 fit inliers
and6 check matches are required; check median error must be at most3 native
pixels, with at least60% within4 pixels. Saved panel quads must overlap with
IoU≥0.65 in both directions and beat the other available candidate-panel quads
by0.15. This competitor set is not a complete panel inventory.

An overlap alone remains `overlap_only_review_required`. Extra local texture
support requires4 fit and2 reserved check matches directly within both panel
polygons, residual≤3 pixels and two-dimensional spatial support. Even the
`provisional_repeat_support` result is a hypothesis requiring independent
identity validation. Similar cell patterns, bad saved boundaries, parallax and
missing competitors remain possible errors.

The trial does not merge findings, assign permanent UUIDs, infer temperature
from colours, or change measurement masks. Pair labels in its CSV identify
review proposals only. Multiple regions on one panel and repeated views of
one region remain distinct records.
