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
  detection/    YOLO pose model adapter and armor detection results
  web/          Headless HTTP/MJPEG debug stream
  test/         Detector smoke and inference regression tests
```

`app` currently owns the end-to-end loop, including TCP frame acquisition,
scene control, display, recording, and browser debug streaming. PnP, tracking, prediction, ballistics,
and gimbal output are not implemented in this executable yet; their future
implementations should become separate `pose`, `tracking`, `prediction`,
`ballistics`, and `control` directories instead of expanding the detector
adapter.

## Build

From the workspace root in a Visual Studio developer shell:

```powershell
cmake --preset windows-vs2022 -S .\yoloDetect
cmake --build .\yoloDetect\build\windows-vs2022 --config Release
ctest --test-dir .\yoloDetect\build\windows-vs2022 -C Release --output-on-failure
```

The build copies `models/armor_pose.onnx` and all required runtime DLLs next to
the executable. CUDA 12.6 and cuDNN 9 runtime DLLs are loaded from
`trains/.venv/Lib/site-packages/torch/lib`, so keep the training virtual
environment available when running this local build.

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

The demonstration model is undertrained and has very low confidence. Its
default threshold is therefore `0.01`. Use a normal threshold such as `0.25`
after training a production model:

```powershell
.\yoloDetect\build\windows-vs2022\Release\yolo_detect.exe --conf 0.25
```

Run `yolo_detect.exe --help` for all connection, model, threshold, and display
options. Press Q or Escape to close the visualization window.

## Headless web debugging

On a server without a desktop, use `--no-display --web` to publish the same
annotated frame used by the local OpenCV window as an MJPEG stream:

```bash
./yolo_detect --no-display --web 8080
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
