# Traditional Armor Detector

This project receives RGB/RGBA frames from Daedalus Simulator and detects armor
without a neural network.

Pipeline:

1. Convert the SDK image to BGR.
2. Build a blue or red light-bar mask from color difference and brightness.
3. Apply morphological closing.
4. Extract elongated contours as light bars.
5. Pair compatible light bars and output four armor corners in the order
   top-left, top-right, bottom-right, bottom-left.

Build on Windows:

```powershell
cmake --preset windows-vs2022
cmake --build build/traditional-windows-vs2022 --config Release
```

Run while Daedalus Simulator is active:

```powershell
& .\build\traditional-windows-vs2022\Release\traditional_detect.exe
```

The application opens the YOLO-style detection window, a `controls` window, and
a `mask` window. Click a value in `controls`, type a number, and press Enter to
apply every detector threshold and geometry constraint in real time. The
initial values come from the macros at the top of `src/main.cpp`.

Broken light bars are handled without increasing the morphology kernel: nearby,
collinear contour fragments are merged before the normal two-light-bar armor
pairing stage. Tune these controls when needed:

- `Fragment minimum length` and `Fragment minimum aspect ratio` reject tiny
  noise before fragment matching.
- `Fragment merge maximum gap (px)` limits the dark gap that may be bridged.
- `Fragment merge maximum angle difference (deg)` requires the pieces to point
  in nearly the same direction.
- `Fragment merge maximum lateral offset (px)` keeps the pieces on the same
  light-bar centerline.

Scene keys match the YOLO project: `1` Armor, `2` Energy, `3` Outpost, `4`
Shooting Range, `0` Reset. Motion keys are `S` Stop, `L` Linear, `P` Spin, `B`
Linear plus Spin, and `+`/`-` adjust speed. Press `Q` or `Esc` to exit.
