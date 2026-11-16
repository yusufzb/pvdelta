# Computational methodology

## Capture geometry

The planner converts the diagonal thermal field of view to horizontal and vertical fields of view using the native sensor aspect ratio. In an approximate planar model, the span along camera axis $j$ is $F_j=2d\tan(\theta_j/2)s_j$. The span $F_j$ and slant distance $d$ are in metres. The dimensionless projection factor $s_j$ is 1 except along the designated tilt axis in nadir mode, where it is $1/\cos\beta$ for module tilt $\beta$. For a panel-normal view, the planner uses $d=h/\cos\beta$ where $h$ is vertical height above the module plane at the ray intersection. For nadir it uses $d=h$.

The conservative sample spacing is $g=\max(F_x/N_x,F_y/N_y)$ in m/px. A feature of length $\ell$ metres has approximate sampling $\ell/g$ pixels. With planned overlap $o$ as a fraction, capture spacing is $F(1-o)$ along the corresponding axis. Along-track spacing divided by photo interval (seconds) gives an overlap-limited speed in m/s. Actual terrain, altitude variation, pose, blur and flight constraints affect realised capture. [Planner implementation](../src/pvdelta/flight.py)

## Radiometric decoding and support

The DJI Thermal SDK decodes an original R-JPEG into a native temperature grid. Source hashes and effective settings, including emissivity, distance, humidity and reflected temperature, bind a saved measurement to its source. The thermal palette is used for display. RGB provides boundary and interpretation evidence where supported. [Decoder and profile](../src/pvdelta/decode.py) · [Measurement profiles](measurement-profiles.md)

Panel support comes from native observations or reviewed geometry. Native-flight processing partitions target and reference support and applies insets. Source-bound surface exclusions can veto known non-panel pixels. Unresolved geometry can be retained separately for operator review. [Support partition](../src/pvdelta/support_partition.py) · [Surface constraints](surface-constraints.md)

## Local background and detection

For module $i$, the fitted background is $B_i(x,y)=a_i+b_ix+c_iy$. Coordinates $(x,y)$ are native thermal pixels, the intercept is in °C and the slopes are in °C/px. The implementation uses least squares with robust residual clipping. Thresholded residual pixels are grouped by four-neighbour connectivity. Native-flight screening uses a 2.5 K residual threshold, at least four connected pixels, at least 24 retained background pixels and 95% finite eligible target support. These values specify the pilot screening path. Review should account for boundary error, reflection, noisy readings and the limits of the local planar reference. [Detector](../src/pvdelta/detect.py) · [Native scan](../src/pvdelta/native_flight.py)

For component $C$, the median measured temperature and median residual are separate statistics. An optional irradiance correction requires usable time- and scope-matched conditions and a documented exponent. The detailed method specifies the correction, units, assumptions, peer comparisons and evidence prerequisites. [Detailed equations and assumptions](method.md) · [Scientific pipeline diagram](diagrams/README.md#scientific-computation)
