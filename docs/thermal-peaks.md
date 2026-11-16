# Native peak separation (research only)

`pvdelta.thermal_peaks.partition_thermal_peaks(temperature_c, region_pixels_yx)`
offers child **review regions** inside one existing hot candidate. It does not
replace detector findings, approve a panel mask or create database records. The
first external comparison did not pass the efficacy gate; production routing is
disabled.

Inputs are a real native thermal array and a nonempty, unique, four-connected
integer `(y, x)` pixel list taken from an existing candidate. Duplicate,
disconnected, nonfinite measured pixels and out-of-grid coordinates are errors.
Nonfinite values outside the parent are irrelevant and never enter a child.
The function receives no labels, panel layout, RGB image or CAD.
The caller must supply decoded temperatures in degrees Celsius, never detector
raw counts, and retain whether the basis is apparent or surface-compensated.

Regional maxima are connected equal-temperature plateaus. A peak's saddle is
the highest path-minimum temperature on any four-connected path to an elder
peak. Higher peaks are elders; equal peaks use fixed spatial ordering. The
highest peak's prominence is its height above the parent minimum. Plateau
ordering is deterministic and does not depend on input pixel order.

`PeakPartitionConfig` fixes exploratory defaults:

- Minimum prominence 1.25 K; at least two connected near-peak pixels above
  `peak - min_prominence / 2`.
- At least four unambiguous pixels in each child, at most 16 retained peaks,
  at most 4096 parent pixels. More than four times the peak limit in raw
  regional maxima also withholds a split before expensive path searches.
- At least two supported peaks are needed; a uniform warm region, shallow
  saddle or unsupported spike does not force another region.

These are research settings, not IEC limits, physical cell dimensions or NETD
measurements. Peaks are found from native temperatures, without interpolation,
palette thermometry or temperature threshold reductions. The module detector's
gradient plane is not reconstructed here: a peak topology is not a detrended
anomaly classification.

Children follow shortest paths inside the parent with positive node cost
`1 + (parent_max - native_pixel_temperature) / min_prominence`. This is an
explicit partition rule, not a physical temperature model. Equal-cost ties are
withheld in `ambiguous_bridge_pixels_yx`; no spatial priority assigns them to a
child. A too-small or disconnected child cancels the complete split. The parent
is then retained unchanged.

For a successful partition, children are disjoint and
`union(children, ambiguous_bridge) == parent`. No outside, shared module edge,
vegetation or reference pixel can be newly added. This guarantee does not prove
the parent was correctly located: a false parent over vegetation remains a
false parent. Exact child pixels and native min/max/median accompany display
bounding rectangles; a rectangle is not a measurement mask.

The result never supplies child background delta, normalisation, confidence,
severity or fault type. Do not copy the parent's fitted-background delta to a
child. Preserve the original finding and bind each partition to its source,
native-array hash, parent ID and exact parent pixel list in an audit record.
Do not count parent and children simultaneously as independent anomalies.

An experimental benchmark may compare unchanged parent predictions with child
regions on the exact same eligible frames. Record both gains and lost matched
references, unmatched-candidate growth, unsplit reasons and bridge conservation.
Do not choose peak count or thresholds from reference boxes. Extra regions do
not establish extra physical panels, cells or faults; poorly resolved or
overlapping labels may still compete in one-to-one matching. Full-panel
semantics and independent reference review remain separate requirements.
