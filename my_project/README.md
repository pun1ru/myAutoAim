# Windows Auto-Aim Project

Windows x64 C++17 consumer for Daedalus Simulator 1.1.1. The first executable
is a read-only OpenCV viewer that receives the latest TCP image, reads the
exposure-synchronized gimbal state, and displays transport statistics.

## Source Layout

| Directory | Responsibility |
| --- | --- |
| `src/sdk` | Daedalus SDK adapter and transport boundary |
| `src/acquisition` | Frame ingestion and buffering |
| `src/synchronization` | Image/state exposure synchronization |
| `src/preprocessing` | Image and model-input preparation |
| `src/detection` | Target detection and classification |
| `src/pose` | Calibration, PnP, and coordinate transforms |
| `src/tracking` | Target association and state estimation |
| `src/prediction` | Motion and latency prediction |
| `src/ballistics` | Projectile compensation |
| `src/control` | Gimbal output and safety gates |
| `src/app` | Complete application composition |
| `src/test` | Diagnostics and integration tests |

## Configure And Build

OpenCV is installed through the project-local vcpkg checkout in `downloads`.

```powershell
Set-Location 'E:\DaedalusSimulator\myAutoAim\my_project'
cmake --preset windows-vs2022
cmake --build --preset windows-release
ctest --preset windows-release
```

## Run The Image Viewer

Start Daedalus first:

```powershell
Set-Location 'E:\DaedalusSimulator\1.1.1'
.\start-simulator.ps1
```

Then run the viewer from another PowerShell window:

```powershell
Set-Location 'E:\DaedalusSimulator\myAutoAim\my_project'
.\build\windows-vs2022\Release\daedalus_image_viewer.exe `
  --ipc-dir '..\1.1.1\runtime\talos-ipc'
```

Press `Q` or `Esc` to close. The viewer only receives images and metadata; it
does not create a UDP client and never sends gimbal or firing commands.

## Headless Receive Benchmark

Close the viewer before benchmarking so it does not compete for the image
stream. Pure TCP receive:

```powershell
.\build\windows-vs2022\Release\daedalus_receive_benchmark.exe `
  --seconds 10
```

TCP receive plus exposure-time gimbal lookup for every frame:

```powershell
.\build\windows-vs2022\Release\daedalus_receive_benchmark.exe `
  --seconds 10 `
  --sync `
  --ipc-dir '..\1.1.1\runtime\talos-ipc'
```

The benchmark creates no OpenCV window and never creates a UDP command client.
