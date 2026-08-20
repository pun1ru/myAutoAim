# YOLO 装甲板检测器

本程序通过 DaedalusSimSdk 接收 RGB/RGBA 图像帧，使用 NVIDIA GPU 上的 ONNX Runtime CUDA
运行导出的 YOLO11 Pose 模型，并实时显示装甲板检测框和四个关键点。OpenCV 用于图像预处理和
可视化窗口。

关键点顺序与训练时完全一致：

1. 左下（bottom-left）
2. 左上（top-left）
3. 右上（top-right）
4. 右下（bottom-right）

## 源码结构

```text
src/
  app/          程序入口、模拟器与推理流程编排
  ballistics/   独立的仅受重力影响的弹道求解器
  control/      基于里程计目标和弹道的云台绝对角度建议
  coordinates/  显式 O/B/G/C 坐标变换和模拟器位姿适配器
  detection/    YOLO 姿态模型适配器与装甲板检测结果
  pose/         相机坐标系装甲板 PnP 与坐标定义
  web/          无界面 HTTP/MJPEG 调试流与位姿/瞄准遥测
  test/         检测、PnP、弹道、坐标和瞄准测试
```

`app` 负责 TCP 图像接收、场景控制、相机坐标系 PnP、同步位姿查询、显示、录像和浏览器调试
推流。坐标变换、弹道和云台角度建议是独立模块。跟踪/预测在模块外部进行，并通过同一瞄准 API
传入预测的里程计目标。默认不会发送云台指令或开火请求。

## 构建

在工作区根目录的 Visual Studio 开发者命令行中执行：

```powershell
cmake --preset windows-vs2022 -S .\yoloDetect
cmake --build .\yoloDetect\build\windows-vs2022 --config Release
ctest --test-dir .\yoloDetect\build\windows-vs2022 -C Release --output-on-failure
```

默认模型为 `models/szu_best2_sim_416.onnx`（深圳大学模型）。构建时会将它和所需运行时 DLL 复制到
可执行文件旁。CUDA 12.6 和 cuDNN 9 运行时 DLL 从
`trains/.venv/Lib/site-packages/torch/lib` 加载。可通过 `--model <path>` 覆盖默认模型。

部署到 Daedalus 1.3.1 AutoDL 环境时使用 Linux 预设：

```bash
cd /root/autoaim-dev/myAutoAim/yoloDetect
cmake --preset linux-autodl
cmake --build build/linux-autodl --parallel
```

Linux 构建会将本地可用 ONNX 模型和 ONNX Runtime Provider 库复制到
`build/linux-autodl/yolo_detect` 旁。

## 运行

先启动 Daedalus Simulator，再执行：

```powershell
.\yoloDetect\build\windows-vs2022\Release\yolo_detect.exe
```

默认使用 CUDA 设备 0，加载模型前程序会输出 `backend=cuda device=0`。选择其他设备或显式
使用 CPU：

```powershell
.\yoloDetect\build\windows-vs2022\Release\yolo_detect.exe --device 0
.\yoloDetect\build\windows-vs2022\Release\yolo_detect.exe --cpu
```

深圳大学模型默认使用目标置信度阈值 `0.70`。可按场景通过命令行覆盖：

```powershell
.\yoloDetect\build\windows-vs2022\Release\yolo_detect.exe --conf 0.70
```

执行 `yolo_detect.exe --help` 查看连接、模型、阈值和显示选项。按 `Q` 或 `Esc` 关闭可视化窗口。

## 相机坐标系 PnP

每个通过校验的检测结果均使用 OpenCV IPPE 独立求解。图像点和模型点固定使用 `BL, TL, TR, BR`
顺序。装甲板坐标系原点位于板中心，`+x_A` 为板法线方向，`+y_A` 指向板左侧，`+z_A` 向上。
`PoseResult::tvec_m` 是以米为单位的装甲板中心在 OpenCV 相机坐标系中的位置（`+x` 向右、
`+y` 向下、`+z` 向前）。`rvec` 的任何分量都不解释为车辆或云台偏航角。

程序使用 Daedalus 1.3.1 固定的 1440x1080 标定，并拒绝其他分辨率的帧。请显式选择物理装甲板
模板；检测器类别 ID 不用于推断尺寸：

```bash
./build/linux-autodl/yolo_detect --armor-size small
./build/linux-autodl/yolo_detect --armor-size large
```

标注图像会显示装甲板轴、相机坐标系中心、重投影 RMS 和 IPPE 候选数量。网页会在实时表格中展示
相同数据；机器可读接口为 `/api/status`。

## 真空弹道

`ballistics::VacuumBallisticSolver` 使用恒定重力、无空气阻力的解析公式求解静止目标。它的输入
限制为炮口坐标系标量几何量：以米为单位的水平距离和竖直偏移，竖直向上为正。该模块不接受
OpenCV 相机坐标，也不生成模拟器带偏移的俯仰指令。

弹丸默认参数与 Daedalus 1.3.1 一致：初速 25 m/s、寿命 5 s、射击冷却 0.05 s、直径 17 mm、
质量 3.2 g、线性阻尼为零。求解器使用显式且可配置的 9.81 m/s^2 重力值，返回低/高弹道俯仰、
飞行时间、发射速度分量和重力下坠量。它会拒绝无效输入、不可达目标、非零线性阻尼和超过弹丸
寿命的轨迹。

## 坐标系与云台瞄准

运行时坐标约定如下：

- `O`：ROS odom，`+x` 向前、`+y` 向左、`+z` 向上。
- `B`：底盘 ROS 坐标系，轴向约定相同。
- `G`：云台/炮口坐标系，`+x` 沿发射方向、`+y` 向左、`+z` 向上。
- `C`：OpenCV 相机坐标系，`+x` 向右、`+y` 向下、`+z` 向前。

固定相机轴映射为 `p_G = (z_C, -x_C, -y_C)`。适配器针对每张图像读取与 `source_sequence`
对应的曝光状态。底盘四元数提供 `R_OB`，云台旋转为 `Rz(yaw) * Ry(-elevation)`。相机和炮口偏移
从 SDK `PoseMeta` 在运行时读取；重建的相机位置必须在 0.01 m 内匹配曝光位置。

有效 PnP 中心会转换为 `target_center_odom_m`。`GimbalAimSolver` 因炮口随偏航和俯仰移动而对
指令角迭代求解，每轮计算低弹道，返回模拟器绝对 `yaw_command_deg` 和 `pitch_command_deg`，其中
90 度表示水平。结果还包含飞行时间、重力下坠、目标和炮口 odom 位置，以及明确失败状态。俯仰
建议限制在模拟器 45 至 135 度范围内。

当前目标和未来预测目标使用同一 odom 目标 API。预测仅通过 `predicted` 与
`prediction_horizon_s` 元数据传递。只有操作员在网页中显式启用静态跟随或请求射击后，程序才会
发送云台命令。

静态跟随会在 `O` 中一次性捕获 PnP 重投影 RMS 最低的有效装甲板中心，之后持续针对该固定世界点
求解并发送绝对偏航/俯仰命令，不会被后续检测替换。停止跟随会清除已捕获目标并取消待处理射击。

## 整车跟踪

`tracking/whole_vehicle_ekf` 使用固定尺寸 Eigen 矩阵实现 11 维整车 EKF：车体中心和速度、连续装甲参考 yaw 与角速度，以及四装甲的交替半径和高度几何参数。它与 PnP、曝光坐标变换和云台控制保持独立；关联在四个物理槽位中选择 NIS 最小且通过门限的候选，`number_id` 与 `color_id` 仅用于身份一致性检查。

运行时测量仅由 `FrameHeader::capture_timestamp_ns` 对应的曝光姿态、PnP 中心和检测质量生成。当前尚未接入独立的约束重投影 yaw 解算，因此适配层只生成 position-only 测量并显式禁用整车 EKF 初始化；不会使用 IPPE Rodrigues 向量或模拟器 GroundTruth 作为估计输入。`whole_vehicle_ekf_test` 与 `tracker_measurement_adapter_test` 分别验证滤波数学和曝光同步。

## 无界面 Web 调试

在没有桌面环境的服务器上，使用 `--no-display --web` 将本地 OpenCV 窗口使用的同一标注帧发布为
MJPEG 流：

```bash
./yolo_detect --no-display --web 8080 \
  --ipc-dir /root/autoaim-dev/daedalus-simulator-1.3.1/runtime/talos-ipc
```

Web 服务器默认绑定到 `127.0.0.1`，不会对公网暴露。从 Windows 通过 SSH 转发端口，再用本地
浏览器打开输出的 URL：

```powershell
ssh -N -L 8080:127.0.0.1:8080 autodl-4090
```

随后访问 `http://127.0.0.1:8080/`。原始流和单帧 JPEG 分别为 `/stream.mjpg` 与
`/snapshot.jpg`。如需在可信私有网络中主动暴露服务，添加 `--web-bind 0.0.0.0`。JPEG 质量由
`--web-quality <1..100>` 设置，默认值为 80。即便设置 `--no-display`，Web 模式仍会绘制并编码
标注帧。

网页还提供 OpenCV 窗口键盘可用的模拟器控制：Shooting Range、Energy、重置、车辆运动和速度
调整。命令会排队并由检测线程执行，因此浏览器不会并发访问 Daedalus SDK。Daedalus 1.3.1 竞赛
构建仅提供 Shooting Range 和 Energy，网页不会显示不可用场景。

`Start Static Follow` 启用固定世界点云台跟随。`Fire Once` 可在跟随开启或关闭时使用：必要时先
捕获目标，持续瞄准，并等待绝对偏航和俯仰误差都不超过 0.5 度；3 秒内未对准则取消。对准后仅
发送一条带 `fire_advice=true` 的已跟踪 UDP 指令，普通跟随指令始终为 false。默认不会发送云台
或开火数据报。Web 状态会报告锁定的 odom 目标、控制器状态、最后跟踪指令 ID 以及最后命令是否
请求开火。

## 场景与车辆控制

可视化窗口维护一个 Daedalus 场景控制会话。单击窗口使其获得焦点后，可使用：

- `1`：装甲板场景
- `2`：能量机关场景
- `3`：前哨站场景
- `4`：射击场场景
- `0`：重置当前场景
- `S`：停止射击场车辆
- `L`：车辆直线运动
- `P`：原地旋转
- `B`：直线运动并旋转
- `+` / `-`：提高/降低直线速度

默认车辆目标 ID 为 3，速度为 1.5 m/s，速度步长为 0.25 m/s。可在启动时覆盖：

```powershell
.\yoloDetect\build\windows-vs2022\Release\yolo_detect.exe `
  --target 3 --speed 2.0 --speed-step 0.5
```

车辆运动命令应用于射击场场景。SDK 命令被拒绝时会在窗口和控制台显示，但不会中断图像接收或
YOLO 推理。

## 无界面基准测试

在不绘制、不缩放、不刷新窗口且不处理键盘的情况下测量 SDK 接收与推理性能：

```powershell
.\yoloDetect\build\windows-vs2022\Release\yolo_detect.exe `
  --no-display --frames 300
```

最终汇总会报告处理帧率、平均模型推理时间，以及推理繁忙期间跳过的源帧数量。本地 RTX 4060
Laptop GPU 的 300 帧 SDK 基准测试为 63.5 FPS 和 12.4 ms 平均推理时间；同样流程下先前 CPU
后端为 24.0 FPS 和 32.4 ms。
