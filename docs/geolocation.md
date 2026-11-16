# Planar image geolocation

`pvdelta.geolocation.LrfPlanarProjector` maps a thermal-image coordinate to a reviewed planar
surface. It is the geometry core for Stage 1 approximate locations; it does not require a plant
asset inventory.

## Inputs and convention

The projector uses the recorded laser target coordinate as the image-centre anchor, the recorded
laser distance, DJI gimbal pitch/yaw/roll, image dimensions, camera diagonal field of view and a
surface plane. Yaw and surface azimuth are clockwise from north. Gimbal pitch is 0° at the horizon
and -90° at nadir. A south-facing 25° module plane is represented by a 25° tilt and 180° facing
azimuth.

The image corner rays and any module/anomaly pixel coordinate can be intersected with that plane.
Results carry the method `lrf_center_planar_raycast_v1`. The centre pixel always maps to the
recorded LRF target by construction.

## Persistence gate

A ray intersection is geometry, not proof of horizontal accuracy. `CaptureRecord` and
`update_capture_footprint` require all stored coverage polygons to include a non-empty method and
a positive reviewed horizontal-accuracy value. Camera GPS remains separate from LRF target,
coverage polygon and estimated module/finding location.

The public M4T specification gives a 45°±0.3° thermal diagonal FOV and LRF distance accuracy of
±(0.2 m + 0.0015×D) beyond 3 m. That range specification alone is not a complete horizontal
location accuracy: DJI's manual also states that RNG positioning depends on GNSS position and
gimbal attitude accuracy. A surveyed reference, calibrated camera/LRF boresight or equivalent
independent validation is therefore needed before storing real footprints with an accuracy claim.

## Stage 1 boresight gate

`validate_boresight_checks()` is the Stage 1 quality gate for this dependency. It reports median,
P95 and maximum horizontal residuals and accepts a result only when the points are independent
surveyed/asymmetric controls and the P95 threshold passes. Camera GPS compared with the LRF/gimbal
reconstruction is useful as a diagnostic, but it is not an independent boresight calibration; that
run remains `not_validated` and keeps estimated panel coordinates `review_required`.

Sources: [DJI Matrice 4 specifications](https://enterprise.dji.com/matrice-4-series/specs) and
[DJI Matrice 4 Series User Manual](https://dl.djicdn.com/downloads/DJI_Matrice_4_Series/DJI_Matrice_4_Series_User_Manual_en.pdf).

## Current limits

- The surface is an infinite plane. Rack boundaries are applied later by module segmentation or
  optional site geometry.
- Per-image calibrated optical-centre and distortion fields were not present in the inspected M4T
  thermal XMP, so the reviewed product DFOV is used.
- The LRF centre anchor must still be checked against the thermal optical centre for the aircraft.
- Terrain, multiple rack planes and objects above the selected plane require separate surfaces.
