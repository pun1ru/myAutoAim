# YOLO Armor Detector

Receives RGB/RGBA frames through DaedalusSimSdk, runs the exported YOLO11 Pose
model with ONNX Runtime CUDA on the NVIDIA GPU, and displays armor boxes and
four keypoints in real time. OpenCV is used for image preprocessing and the
visualization window.

The keypoint order is preserved exactly as trained:

1. bottom-left
2. top-left
3. top-right
4. bottom-right

## Source layout

The source tree follows the current auto-aim pipeline boundaries:

```text
src/
  app/          Application entry point and simulator/inference orchestration
  ballistics/   Standalone gravity-only projectile trajectory solver
  control/      Absolute gimbal-angle advice from odom targets and ballistics
  coordinates/  Explicit O/B/G/C transforms and simulator pose adapter
  detection/    YOLO pose model adapter and armor detection results
  pose/         Camera-frame armor PnP and coordinate definitions
  web/          Headless HTTP/MJPEG debug stream and pose/aim telemetry
  test/         Detector, PnP, ballistics, coordinate, and aim tests
```

`app` currently owns the end-to-end loop, including TCP frame acquisition,
scene control, camera-frame PnP, synchronized pose lookup, display, recording,
and browser debug streaming. Coordinate transforms, ballistics, and gimbal
angle advice are independent modules. Tracking/prediction remains external and
feeds a predicted odom target through the same aim API. No gimbal command or
firing request is sent.

## Build

From the workspace root in a Visual Studio developer shell:

```powershell
cmake --preset windows-vs2022 -S .\yoloDetect
cmake --build .\yoloDetect\build\windows-vs2022 --config Release
ctest --test-dir .\yoloDetect\build\windows-vs2022 -C Release --output-on-failure
```

The default model is `models/armor_pose_0815_640.onnx`; the build copies it and
all required runtime DLLs next to the executable. CUDA 12.6 and cuDNN 9 runtime
DLLs are loaded from `trains/.venv/Lib/site-packages/torch/lib`, so keep the
training virtual environment available when running this local build. Override
the default model at runtime with `--model <path>` when needed.

For the deployed Daedalus 1.3.1 AutoDL environment, use the Linux preset:

```bash
cd /root/autoaim-dev/myAutoAim/yoloDetect
cmake --preset linux-autodl
cmake --build build/linux-autodl --parallel
```

The Linux build copies locally available ONNX models and ONNX Runtime provider
libraries next to `build/linux-autodl/yolo_detect`.

## Run

Start Daedalus Simulator first, then run:

```powershell
.\yoloDetect\build\windows-vs2022\Release\yolo_detect.exe
```

CUDA device 0 is the default. The program prints `backend=cuda device=0` before
loading the model. Select another CUDA device or explicitly fall back to CPU
with:

```powershell
.\yoloDetect\build\windows-vs2022\Release\yolo_detect.exe --device 0
.\yoloDetect\build\windows-vs2022\Release\yolo_detect.exe --cpu
```

The default 0815 model uses a low object-confidence threshold of `0.01`.
Override it with a normal threshold such as `0.25` after validating that value
against the deployed model:

```powershell
.\yoloDetect\build\windows-vs2022\Release\yolo_detect.exe --conf 0.25
```

Run `yolo_detect.exe --help` for all connection, model, threshold, and display
options. Press Q or Escape to close the visualization window.

## Camera-frame PnP

Each accepted detection is solved independently with OpenCV IPPE. Image and
model points use the fixed `BL, TL, TR, BR` order. The armor frame origin is at
the plate center, `+x_A` is the plate normal, `+y_A` points toward plate-left,
and `+z_A` points up. `PoseResult::tvec_m` is therefore the armor center in the
OpenCV camera frame (`+x` right, `+y` down, `+z` forward), in meters.
No component of `rvec` is interpreted as vehicle or gimbal yaw.

The executable uses the Daedalus 1.3.1 fixed 1440x1080 calibration and rejects
frames with a different resolution. Select the physical plate template
explicitly; the detector class IDs are not used to guess its size:

```bash
./build/linux-autodl/yolo_detect --armor-size small
./build/linux-autodl/yolo_detect --armor-size large
```

The annotated image shows the armor axes, camera-frame center, reprojection RMS,
and IPPE candidate count. The web page presents the same values in a live table;
its machine-readable endpoint is `/api/status`.

## Vacuum ballistics

`ballistics::VacuumBallisticSolver` solves a stationary target analytically with
constant gravity and no aerodynamic drag. Its input is deliberately limited to
muzzle-frame scalar geometry: horizontal distance and vertical offset, both in
meters, with vertical positive upward. It does not accept OpenCV camera-frame
coordinates or produce the simulator's offset pitch command.

The projectile defaults match Daedalus 1.3.1: 25 m/s muzzle speed, 5 s lifetime,
0.05 s firing cooldown, 17 mm diameter, 3.2 g mass, and zero linear damping. The
solver uses an explicit, configurable gravity magnitude of 9.81 m/s^2 and returns
low- or high-arc pitch, flight time, launch velocity components,
and gravity drop. It rejects invalid input, unreachable targets, non-zero linear
damping, and trajectories that exceed projectile lifetime. Diameter and mass are
retained as physical configuration but do not enter the vacuum equations.

The aim pipeline connects this solver to PnP through explicit coordinate and
control layers. The ballistic solver itself remains independent of camera
geometry, tracking, SDK command transport, and firing decisions.

## Coordinate frames and gimbal aim

The runtime coordinate contract is:

- `O`: ROS odom, `+x` forward, `+y` left, `+z` up.
- `B`: chassis ROS frame with the same axis convention.
- `G`: gimbal/muzzle frame, `+x` along the launch direction, `+y` left,
  `+z` up.
- `C`: OpenCV camera frame, `+x` right, `+y` down, `+z` forward.

The fixed camera-axis mapping is
`p_G = (z_C, -x_C, -y_C)`. For each image, the adapter reads the exposure
state matching `source_sequence`. The chassis quaternion supplies `R_OB`,
and the gimbal rotation is `Rz(yaw) * Ry(-elevation)`. Camera and muzzle
offsets are read from SDK `PoseMeta` at runtime; the reconstructed camera
position must agree with the exposure position within 0.01 m.

A valid PnP center is transformed into `target_center_odom_m`.
`GimbalAimSolver` iterates the command angle because the muzzle position moves
with yaw and pitch, solves the low vacuum trajectory at every iteration, and
returns absolute simulator `yaw_command_deg` and `pitch_command_deg`, where
90 degrees is level. It also returns time of flight, gravity drop, target and
muzzle odom positions, and explicit failure status. Pitch advice is limited to
the simulator's 45 to 135 degree range.

Current and future predicted targets use the same odom target API. Prediction
stays outside the solver and is carried only as `predicted` and
`prediction_horizon_s` metadata. The application does not send a gimbal command
until the operator explicitly enables static follow or requests a shot from the
web page.

Static follow captures the valid armor center with the lowest PnP reprojection
RMS once in `O`, then keeps solving and sending absolute yaw/pitch commands
against that fixed world point. It does not replace the target with later
detections. Stopping follow clears the captured target and cancels any pending
shot.

## Headless web debugging

On a server without a desktop, use `--no-display --web` to publish the same
annotated frame used by the local OpenCV window as an MJPEG stream:

```bash
./yolo_detect --no-display --web 8080 \
  --ipc-dir /root/autoaim-dev/daedalus-simulator-1.3.1/runtime/talos-ipc
```

The web server binds to `127.0.0.1` by default, so it is not exposed publicly.
From Windows, forward the port through SSH and open the printed URL in a local
browser:

```powershell
ssh -N -L 8080:127.0.0.1:8080 autodl-4090
```

Then browse to `http://127.0.0.1:8080/`. The raw stream and a single JPEG are
also available at `/stream.mjpg` and `/snapshot.jpg`. To deliberately expose
the stream on a trusted private network, add `--web-bind 0.0.0.0`.

The JPEG quality is configurable with `--web-quality <1..100>` (default 80).
Web mode draws and encodes annotated frames even when `--no-display` is set.

The page also exposes the simulator controls that were previously available by
keyboard in the OpenCV window: Shooting Range, Energy, reset, vehicle motion,
and speed adjustment. Commands are queued and executed on the detector thread,
so the browser never accesses the Daedalus SDK concurrently. Daedalus 1.3.1
contest builds only expose Shooting Range and Energy; unavailable scenes are
not shown on the web page.

The `Start Static Follow` button enables fixed-world-point gimbal following.
The `Fire Once` button may be used with or without follow: it captures a target
if necessary, continuously aims, and waits until both absolute yaw and pitch
errors are at most 0.5 degrees. The request is cancelled if alignment is not
reached within 3 seconds. After alignment, exactly one tracked UDP command is
sent with `fire_advice=true`; ordinary follow commands always set it to false.
No gimbal or fire datagram is sent by default.

The web status reports the latched odom target, controller state, last tracked
command ID, and whether the last sent command requested fire.

## Scene and vehicle controls

The visualization window also owns a Daedalus scene-control session. Click the
window once, then use:

- `1`: Armor scene
- `2`: Energy scene
- `3`: Outpost scene
- `4`: Shooting Range scene
- `0`: Reset the current scene
- `S`: Stop the shooting-range vehicle
- `L`: Linear vehicle motion
- `P`: Spin in place
- `B`: Linear motion and spin
- `+` / `-`: Increase or decrease linear speed

The default vehicle is target 3, with a speed of 1.5 m/s and a speed step of
0.25 m/s. Override these values at startup when needed:

```powershell
.\yoloDetect\build\windows-vs2022\Release\yolo_detect.exe `
  --target 3 --speed 2.0 --speed-step 0.5
```

Vehicle motion commands apply to the Shooting Range scene. A rejected SDK
command is shown in the window and logged to the console without stopping image
reception or YOLO inference.

## Headless benchmark

To measure SDK reception and inference without any drawing, resizing, window
refresh, or keyboard handling, run a fixed number of frames:

```powershell
.\yoloDetect\build\windows-vs2022\Release\yolo_detect.exe `
  --no-display --frames 300
```

The final summary reports processed frames per second, average model inference
time, and source frames skipped while inference was busy.

On the local RTX 4060 Laptop GPU, a 300-frame SDK benchmark measured 63.5 FPS
and 12.4 ms average inference time. The previous CPU backend measured 24.0 FPS
and 32.4 ms average inference time under the same headless workflow.
