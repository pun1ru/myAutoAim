# Yaw Spin Tuning

## Baseline

Recorded before spin testing on 2026-08-24.

| Parameter | Value |
|---|---:|
| Yaw max reprojection RMS | 4.0 px |
| Yaw max standard deviation | 0.45 rad |
| Yaw minimum facing cosine | 0.50 |
| Yaw opposite-solution margin | 0.50 px |
| Planar-distance cost weight | 1.0 |
| Yaw-difference cost weight | 0.60 m/rad |
| Front-facing yaw variance | 0.60 rad^2 |
| Dynamic prediction delay | 0.060 s |

## Acceptance

- At both 120 deg/s and 180 deg/s, every valid armor observation must have a
  reliable yaw.
- Consecutive associated slots must advance consistently around E0..E3; an
  adjacent direction reversal is a failure.

## Results

Sampled for approximately eight seconds per speed after an EKF reset.

| Spin speed | Observed samples | Samples without reliable yaw | Slot reversals | Result |
|---|---:|---:|---:|---|
| 120 deg/s | 77 | 0 | 0 | Pass |
| 180 deg/s | 75 | 0 | 0 | Pass |

## 200 deg/s Drop Diagnosis

Target speed `200 deg/s = 3.491 rad/s`. Runtime logs:

- Baseline `spin200-baseline-20260824.csv`: omega range `1.60..2.60
  rad/s`; 137/150 samples were below `2.5 rad/s`.
- Increasing `maximum_omega_correction_rad_s` from `0.15` to `0.5` did not
  recover the speed (`max 2.40 rad/s`).
- Lowering Rtheta from `1` to `0.09` also did not recover it (`max 1.62
  rad/s`).
- `spin200-no-large-omega-block.csv` after the fix: omega average `3.260
  rad/s`, range `2.49..3.94 rad/s`, only 2/120 samples below `2.5 rad/s`,
  and zero slot-direction reversals.

Root cause: `applyJointUpdate()` previously zeroed the Omega Kalman gain when
the yaw innovation exceeded half of `maximum_yaw_update_innovation_rad`.
At high spin this innovation is a normal lag signal; in the baseline 86/119
observed samples exceeded that half-gate. The code now zeroes Omega only for a
real discrete slot transition, while continuous yaw innovation can update the
angular speed.

The yaw-validity values above were retained. The prior E0 reversals were
caused by immediate E0 reset after a one-frame detector blackout, not by a
reliable-yaw rejection. The tracker now retains E0 during detector blackouts
and resets only after 20 consecutive missed frames.
