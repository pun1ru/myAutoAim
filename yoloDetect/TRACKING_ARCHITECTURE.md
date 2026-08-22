# Whole-Vehicle EKF Architecture

Read the implementation in this order:

1. `src/tracking/tracker_measurement_adapter.cpp`
2. `src/tracking/constrained_yaw_solver.cpp`
3. `src/tracking/whole_vehicle_ekf.hpp`
4. `src/tracking/whole_vehicle_ekf.cpp`
5. `src/app/main.cpp` at the `makeTrackerMeasurement()` call and drawing code
6. `src/test/whole_vehicle_ekf_test.cpp`

## Module boundaries

- PnP (`src/pose`) estimates an armor center in OpenCV camera frame C. Its
  translation is converted from mm to m before entering tracking.
- Coordinates (`src/coordinates`) reads the pose belonging to the same
  `source_sequence` and `capture_timestamp_ns`, then converts the PnP center
  from C to tracker frame T. T is ROS odom during one tracking session:
  +x forward, +y left, +z up.
- Constrained yaw follows the `sp_vision_25` operational solver: it keeps the
  PnP center fixed, uses a `+15 deg` ordinary-armor or `-15 deg` outpost pitch
  prior, searches the exposure-time gimbal yaw +/- `70 deg` at `1 deg` steps,
  and selects the four-corner reprojection minimum (with a local refinement
  inside the winning one-degree cell). The IPPE rotation remains
  available as raw full armor pose, but is not silently reused as EKF yaw. As
  in the reference, large balance armor skips this fixed-pitch yaw step. An
  unreliable constrained yaw leaves a position-only measurement.
- The measurement adapter is the only interface into the EKF. It bundles the
  timestamp, T-frame position, optional reliable yaw, quality, and target
  identity. It does not assign E0-E3.
- `WholeVehicleEkf` contains prediction, association, update, and the tracking
  state machine. It uses fixed-size Eigen matrices only.

## State and armor model

The 11D state is:

```text
[cx, vx, cy, vy, cz, vz, theta, omega, r0, dr, dz]
```

`c` and `v` are the vehicle center and velocity in T. `theta` is the
continuous inward yaw of physical slot E0. `omega` is the angular speed.
`r0`, `dr`, and `dz` describe the alternating geometry of four armor plates.

For physical slot `i` in `[0, 3]`:

```text
parity = i % 2
phi    = theta + i*pi/2
r      = r0 + parity*dr
z      = cz + parity*dz
armor  = [cx-r*cos(phi), cy-r*sin(phi), z]
yaw    = phi
```

`number_id` and `color_id` identify the vehicle only. They are never physical
armor slot numbers.

## Per-frame flow

```text
image header (sequence, exposure timestamp)
  -> PnP armor center in C
  -> same-exposure C->T transform + constrained yaw
  -> Measurement list
  -> predict with dt from exposure timestamps
  -> test every measurement against every E0..E3 slot
  -> reject candidates by identity, visibility, yaw consistency, and NIS
  -> select a one-to-one minimum-cost assignment
  -> one stacked fixed-size EKF update using Joseph covariance form
  -> update Confirming/Tracking/TemporarilyLost/Lost state machine
  -> decode E0..E3 and project them for web yellow markers
```

Prediction uses constant center velocity and constant angular velocity. The
process covariance is a continuous white-acceleration model. `theta` remains
continuous internally; only yaw innovations are wrapped to `[-pi, pi)`.

## Association and update

For each candidate measurement/slot pair, the EKF predicts `h(x, slot)` and
computes the innovation. Position-only observations use 3D NIS; observations
with reliable yaw use 4D NIS. NIS is solved with LDLT, never an explicit matrix
inverse. Candidates outside the configured chi-square gates are rejected.

All accepted observations are stacked into one joint update. This matters when
two plates are visible: their relative positions constrain the center and the
alternating radius/height geometry. Geometry is only enabled after the
multi-armor observability checks pass. A single plate cannot independently
identify both center and radius, so it cannot update geometry.

The covariance update is Joseph form, followed by symmetrization and finite /
physical-bound checks. A failed update restores the predicted state.

## State machine

- `Uninitialized`: needs a reliable-yaw observation to define E0 and infer an
  initial center from `radius_prior_m`.
- `Confirming`: requires consecutive successful associations.
- `Tracking`: normal predict, associate, and update operation.
- `TemporarilyLost`: no accepted observation; only prediction and process noise
  growth occur.
- `Lost`: loss limit exceeded; state is cleared and the next reliable yaw may
  initialize a new track.

## Debugging order

Use the web panel before changing Q/R:

1. `D#` is the raw detector center, `M#` is the same PnP center transformed to
   T and projected back to the image, and yellow `E#` is EKF predicted armor.
2. If D/M disagree, investigate PnP, calibration, or exposure transform first.
3. If M is correct but E is wrong, inspect associated E slot, innovations, NIS,
   radius, `dr`, `dz`, and reliable-yaw status.
4. Only after association and geometry are correct, tune measurement R or
   process Q. Do not use larger R to hide a sign, timestamp, or slot error.

The synthetic regression suite is `src/test/whole_vehicle_ekf_test.cpp`. It
covers prediction/observation consistency, analytic Jacobian, yaw wrapping,
slot switching, NIS rejection, multi-armor geometry, loss/reset, covariance,
irregular dt, and exposure snapshot matching.
