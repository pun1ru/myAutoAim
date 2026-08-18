# New training data

This project captures raw simulator images for a new armor-pose dataset.

The collector runs one fixed sequence:

1. 4 seconds in the outpost scene.
2. 4 seconds in the shooting range, target 3, linear speed 2.0 m/s,
   span 4 m, direction 90 degrees, and spin speed 180 degrees/s.

The verified 300-image capture is stored as JPG files in
`data_300/raw/outpost` and `data_300/raw/shooting_range_linear_spin`.
`data_300/manifests` records the source sequence and capture time for every
image. The images are intentionally raw and unannotated; annotate the four
armor corners in CVAT before converting
them with `trains/prepare_armor_pose.py`.

The verified 300-image JPG capture is in `data_300/raw` with its manifest in
`data_300/manifests`. The original BMP files are retained in
`data_300/raw_bmp`. The earlier smaller capture remains in `data/raw`.

## Build and capture

Configure with the repository's SDK and vcpkg toolchain, then build:

```powershell
cmake -S . -B build/windows-vs2022 `
  -DCMAKE_TOOLCHAIN_FILE=../my_project/downloads/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DDaedalusSimSdk_DIR=../1.1.1/sdk/lib/cmake/DaedalusSimSdk `
  -DOpenCV_DIR=../my_project/vcpkg_installed/x64-windows/share/opencv4
cmake --build build/windows-vs2022 --config Release
```

Start the simulator first, then run:

```powershell
.\build\windows-vs2022\Release\new_trains_capture.exe
```

An optional output directory can be supplied as the first argument.

To convert a future BMP capture to JPG, run:

```powershell
python scripts/convert_capture_to_jpg.py data_300 data_300_jpg
```
