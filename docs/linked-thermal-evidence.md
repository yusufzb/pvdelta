# Own-frame thermal comparison

In V2, open **Related candidate cards · provisional**, select a provisional
linked target, then choose **Compare own-frame SDK hotspots**. This is explicit:
opening a card does not trigger another thermal analysis. Each image shows its
own native DJI SDK temperature grid, its own saved candidate pixels in green,
and its provisional panel outline in cyan. The display uses one numeric colour
scale for both panels and nearest-neighbour enlargement. Original camera
orientations are retained. No hotspot geometry or measurement is copied from
one image to the other.

Both complete saved hot-region populations must reproduce using each capture's
own saved measurement profile and detector settings. Temperatures, ΔT, region
pixels and references remain those of that capture. Source hashes, profiles,
membership, input bindings, SDK/runtime and engine are checked; changed inputs
withhold the result. The inspection database and review decisions are read-only.
Failure leaves the main card and related-card navigation available. Results are
session-local, downloadable, and cleared on source/target changes; stale results
are withheld.

Panel-relative u/v coordinates follow the RGB tracker's physical corner order,
including return-flight rotation, through the approximate native/RGB anchors.
Missing or crossing corner correspondence is rejected rather than guessed.
u/v is the mean of the exact saved native pixel centres after this conditional
geometric mapping. Every source/target region combination is reported, with its
centre distance in unit-panel coordinates. There is no best-region substitution,
distance cutoff, automatic component match or fault decision. The distance is
not metres and is not registration uncertainty.

Approximate cross-camera anchors and provisional panel identity can associate
different physical areas. Similar positions and temperatures do not verify a
repeated hot cell or clear glare. A panel with no recorded candidate is not
classified as healthy. This bounded comparison covers candidate-containing
captures retrieved by the existing search; it does not enumerate every thermal
view, establish independent accuracy, merge physical assets, or satisfy field
acceptance requirements.
