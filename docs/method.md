# Radiometric method and inspection scope

PV Delta screens photovoltaic modules for thermal anomalies in original DJI
radiometric JPEGs. It produces source-linked pixel evidence for operator review.
This page describes the implemented pilot and the evidence needed to validate an
inspection. For setup and outputs, see the [operator guide](stage1-pilot.md).

## 1. Radiometry and source integrity

A radiometric JPEG carries thermal measurements and camera calibration data.
The visible palette is a rendering whose colors depend on display settings.
PV Delta calls DJI Thermal SDK `dirp_measure_ex` to obtain an unresampled
`float32` temperature array $T(x,y)$ in °C. For a 640 × 512 capture,
each array element represents one original thermal pixel. The separate
`dirp_process` output is used for the native-grid color overlay.

A simplified model for an opaque target illustrates why the radiometric
parameters matter. At wavelength $\lambda$, the measured radiance contains
emission from the target, reflected radiance and atmospheric emission.

$$
L_{\lambda,\mathrm{cam}} =
\tau_\lambda \left[\varepsilon_\lambda B_\lambda(T_s)
+ (1-\varepsilon_\lambda)L_{\lambda,\mathrm{refl}}\right]
+ (1-\tau_\lambda)B_\lambda(T_{\mathrm{atm}}).
$$

$B_\lambda$ denotes blackbody spectral radiance, $T_s$ the surface temperature,
$\varepsilon_\lambda$ emissivity and $\tau_\lambda$ path transmittance. This is
the physical measurement model. The DJI SDK applies the camera-specific
calibration and inversion. The software records object distance, humidity,
emissivity, air temperature and reflected apparent temperature, including
explicit overrides and effective SDK readback. Reflected apparent temperature
describes radiation reflected toward the camera. It can differ from air
temperature. Loss of focus, saturated data, mixed edge pixels, reflections and
uncertain settings can bias temperatures and ΔT. [Thermographic measurement guide](https://support.flir.com/docdownload/assets/dl/t559495-en-us%24aa.pdf)

A [measurement profile](measurement-profiles.md) binds the original source and
the decoded native array to their SHA-256 hashes and the effective parameters.
Replays check the source, dimensions and hash. SDK/CLI numerical agreement
verifies decoding consistency with the same settings. Instrument accuracy
requires an appropriate external reference and a known measurement uncertainty.

## 2. Select the measurement support

The explicit-region workflow accepts reviewed module quadrilaterals and
optional masks. The experimental native-flight workflow proposes
quadrilaterals from observed thermal edges and line groups, using geometric
filters and bounded local searches. These are provisional module regions.
A proposal can miss an edge or include part of the surrounding scene.
[Automatic geometry](native-flight.md) · [Native mask diagnostics](native-mask-qa.md)

For module $i$, a polygon rasterizes to a native-grid mask $M_i$. The flight
path partitions overlapping envelopes and removes multiply assigned pixels.
The measurement support $S_i$ excludes the configured boundary margin,
nonfinite temperatures, invalid sensor samples and any source-bound reviewed
non-panel veto. The pilot native-flight profile uses a two-pixel partition
margin and one further pixel of detector erosion. Candidate and reference
pixels stay on the same native grid. An empty or insufficiently valid support
is marked unassessed. Boundary-sensitive results can be reviewed separately.

## 3. Estimate the local background

A single module can have a smooth temperature gradient from illumination,
wind, mounting and viewing geometry. With detrending enabled, the detector
models that background as a plane $B_i(x,y)=a_i x+b_i y+c_i$ over the module.
It seeds a robust fit from bounded, deterministic samples and uses least
squares on retained reference pixels $R_i \subseteq S_i$.

$$
(a_i,b_i,c_i)=
\underset{a,b,c}{\operatorname{argmin}}
\sum_{(x,y)\in R_i}\left[T(x,y)-ax-by-c\right]^2 .
$$

During reference estimation, the code uses robust residual rejection based on
the median and median absolute deviation. It then excludes warm and cool
outliers beyond the candidate threshold and refits until the retained
reference stabilizes or the iteration limit is reached. The optional
non-detrended path uses a retained-pixel median. Too few reference pixels,
degenerate plane geometry or a reference that fails to converge prevents a
reported intra-module assessment.

## 4. Extract and measure anomaly candidates

The residual at each eligible pixel is $r_i(x,y)=T(x,y)-B_i(x,y)$. A hot
candidate comes from a four-connected component $C$ of pixels with
$r_i(x,y)\geq\tau$. The corresponding cold test is $r_i(x,y)\leq-\tau$.
The native-flight scan selects the hot direction. It currently uses
$\tau=2.5$ K, a minimum component of 4 native pixels, at least 24 retained
reference pixels and a minimum valid support fraction of 0.95. These are
project screening settings. Changing them changes the detection tradeoff.

$$
\Delta T_C = \operatorname{median}_{(x,y)\in C}\left[T(x,y)-B_i(x,y)\right].
$$

The report preserves the exact component pixels, their bounding box, region
median, reference median, maximum, 95th percentile and signed ΔT. The median
of pixel-level differences can differ from subtracting the two separately
reported medians. A 1 °C difference has the same numerical magnitude as
1 K. An enlarged display rectangle is a locator. It does not change the
component area or the measurement support.

For an explicitly supplied group of comparable modules, the whole-module
comparison subtracts the leave-one-out median of peer module medians from
the target module median. The comparison requires at least three usable
members and withholds groups with widespread deviations. This is a separate
test because an intra-module reference can absorb uniform heating.

## 5. Conditions, context and localization

Optional normalization uses
$\Delta T_{\rm ref}=(Q_{\rm ref}/Q_{\rm meas})^x\Delta T_{\rm meas}$.
$Q$ is either plane-of-array irradiance with like units on both sides or
a separately justified DC-current reference. Conditions must be usable,
time-matched and from the right scope. The method needs an explicit physical
extent. The implementation uses $x=1$ for area cases and a configurable
point exponent in the range 1.5–1.8. It withholds irradiance-based
normalization below 600 W/m² and point projections below a 0.60
measured/reference ratio. Without suitable evidence the report keeps the
measured ΔT. [Condition inputs](flight-dashboard.md#field-record)

The thermal image locates the observation in its own pixel grid. Paired RGB
and the registration check can help associate it with visible soiling,
debris and module boundaries. Similar-looking modules and reflections make
unverified projections and repeat views provisional. Field checks, comparable
viewing angles, local evidence and electrical inspection can help evaluate
cause. An empty candidate list does not establish panel health.

Map coordinates depend on pose, camera geometry and the assumed surface.
The camera GPS position is the aircraft position. Stage 1 uses frame-local
observation IDs. When a surveyed layout is available, optional Stage 2 can
associate observations with asset IDs and report unresolved matches.
[Geolocation](geolocation.md) · [Panel identity](panel-identities.md) · [Inspection export](inspection-export.md)

## 6. IEC TS 62446-3 and field evidence

[IEC TS 62446-3:2017](https://webstore.iec.ch/en/publication/28628) covers
outdoor infrared inspection of PV modules and plants in operation. Its scope
includes equipment, conditions, procedure, reporting, personnel and thermal
abnormality guidance. The [IEA PVPS Task 13 O&M report](https://iea-pvps.org/wp-content/uploads/2022/11/IEA-PVPS-Report-T13-25-2022-OandM-Guidelines.pdf)
gives field guidance including stable, on-site plane-of-array irradiance of
at least 600 W/m² for module thermography and sufficient sampling across
each cell. Consult the applicable specification and inspection scope before
issuing a conformity statement.

| Evidence area | Supported by PV Delta | Evidence needed for a field decision |
| --- | --- | --- |
| Equipment and resolution | Native grid, camera metadata where available, SDK parameters and source identity | Instrument suitability and calibration, focus, usable range, sampling at the actual target distance and viewing geometry |
| Environment and operation | Record conditions, associate usable inputs by scope and time, and flag unavailable normalization | Measured POA irradiance and stability, wind and site conditions, operating state and documented capture procedure |
| Interpretation | Signed ΔT, local reference, measured pixels, thermal/RGB evidence and operator review | Qualified review, cause assessment, supporting observations and any needed field or electrical tests |
| Report | Source-linked findings, scope and availability flags, review record and export | Complete inspection coverage, traceable measurement uncertainty, personnel qualification and approval for the actual inspection |

The public debris example is a documented thermal candidate with an
operator-reported cause. It does not establish an IEC class, automated
cross-site accuracy or field conformity. The pilot supports the inspection
record. PV Delta does not issue a compliance certificate.

## 7. Evaluate the stages independently

- **Radiometry.** Report source count, compared pixel count, effective
  parameters and numerical error against the comparison implementation.
- **Geometry.** Score native module boundaries against independently
  reviewed polygons and report views where usable support is missing.
- **Candidate detection.** Use a frozen set of independently annotated
  hot-region observations. Match spatially one-to-one at a stated IoU
  threshold, then report $TP/(TP+FP)$ and $TP/(TP+FN)$ with counts and
  exclusions. A thermal-region label and a field-confirmed cause are
  separate targets. [Hotspot evaluation](hotspot-evaluation.md)
- **Registration and identity.** Report held-out geometric residuals,
  verified repeat matches and unresolved cases. Validate any map-error
  claims against independent controls.
