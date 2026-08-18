# 第一个传统视觉识别器

这是“自瞄教程”识别模块的可运行传统视觉参考程序。它连接 Daedalus Simulator 1.1.1、切换
Shooting Range、读取最新图像，并完成颜色分割、轮廓筛选、灯条配对与角点绘制；不识别数字。

构建前安装 OpenCV 4，并下载 Daedalus Simulator 1.1.1 对应平台的完整 Release。

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=<Daedalus发布目录>/sdk
cmake --build build --parallel
```

Windows 使用 vcpkg 时，同时传入：

```text
-DCMAKE_TOOLCHAIN_FILE=C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake
```

运行模拟器后启动 `first_traditional_detector`。按 `1` 切换静止目标，按 `2` 切换直线
运动，按 `3` 切换直线并自转，按 `Q` 或 `Esc` 退出。`detector` 窗口中：橙色框为候选灯条、
黄色线为灯条长轴、绿色四边形和圆点为两灯条配对得到的角点；`mask` 窗口显示二值图。
