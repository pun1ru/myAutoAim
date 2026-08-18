# myAutoAim

Source code for an auto-aim development workspace. This repository contains:

- YOLO pose armor detection and ONNX Runtime inference code
- Training and dataset preparation scripts
- Traditional detector experiments
- CMake-based C++ project scaffolding

## Not Included

The repository intentionally excludes the Daedalus Simulator distribution,
model weights, datasets, recordings, Python virtual environments, and build
artifacts. Obtain the simulator and model assets separately, then configure
the relevant project paths locally.

## Projects

- `yoloDetect`: YOLO pose detector and simulator integration example
- `trains`: training and export scripts
- `new_trains`: newer training and conversion experiments
- `traditionalDetect`: traditional image-processing detector experiments
- `first-traditional-detector`: initial traditional detector implementation
- `my_project`: C++ project experiments and configuration
- `RobotDetectionModel`: robot detection inference source

## Development Environment

The intended development stack is C++17, CMake, OpenCV, ONNX Runtime GPU,
Python 3.10, PyTorch CUDA, and Ultralytics. The Daedalus Simulator SDK is an
external dependency and is not bundled here.
