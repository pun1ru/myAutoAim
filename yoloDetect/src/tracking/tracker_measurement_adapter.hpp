#pragma once

#include "coordinates/coordinate_frames.hpp"
#include "detection/yolo_pose_detector.hpp"
#include "pose/armor_pose_estimator.hpp"
#include "tracking/constrained_yaw_solver.hpp"
#include "tracking/whole_vehicle_ekf.hpp"

#include <cstdint>
#include <optional>

namespace yolo_detect::tracking {

struct TrackerFrame {
  // 与 FrameHeader 一一对应，用于拒绝“图像与云台姿态来自不同曝光”的量测。
  std::uint64_t source_sequence = 0;
  std::uint64_t capture_timestamp_ns = 0;
};

// 将 PnP、检测质量、同曝光坐标快照和约束 yaw 汇总为 EKF Measurement。
// 当 yaw 求解不可靠时仍返回 position-only 量测，但首次初始化需要可靠 yaw。
[[nodiscard]] std::optional<Measurement> makeTrackerMeasurement(
    const TrackerFrame& frame,
    const coordinates::CoordinateSnapshot& exposure_snapshot,
    const PoseResult& pose, const ArmorDetection& detection,
    const ConstrainedYawSolver& yaw_solver,
    ReliableYaw* reliable_yaw = nullptr);

}  // namespace yolo_detect::tracking
