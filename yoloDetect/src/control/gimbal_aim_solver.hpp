#pragma once

#include "ballistics/vacuum_ballistic_solver.hpp"
#include "coordinates/coordinate_frames.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <limits>

namespace yolo_detect::control {

enum class AimStatus {
  Success,
  InvalidCoordinateSnapshot,
  NonFiniteTarget,
  BallisticFailure,
  PitchOutsideLimits,
  DidNotConverge,
};

struct AimTarget {
  cv::Vec3d center_odom_m{0.0, 0.0, 0.0};
  bool predicted = false;
  double prediction_horizon_s = 0.0;
};

struct GimbalAimOptions {
  double minimum_pitch_command_deg = 45.0;
  double maximum_pitch_command_deg = 135.0;
  // Physical target-height correction applied before ballistic solving. A
  // negative value aims below the reconstructed armor center.
  double target_height_offset_m = -0.05;
  // Match sp_vision_25: at most ten flight-time fixed-point iterations,
  // converging when adjacent time-of-flight estimates differ by under 1 ms.
  std::size_t maximum_iterations = 10;
  double flight_time_convergence_tolerance_s = 1e-3;
  // The reference pipeline has no muzzle-offset fixed point. This additional
  // check ensures the simulator's moving muzzle has settled before a command
  // is emitted.
  double muzzle_direction_tolerance_rad = 1e-7;
};

struct GimbalAimResult {
  bool valid = false;
  AimStatus status = AimStatus::InvalidCoordinateSnapshot;
  ballistics::BallisticStatus ballistic_status =
      ballistics::BallisticStatus::NumericalFailure;
  bool predicted = false;
  double prediction_horizon_s = 0.0;
  // The height-offset-adjusted point used by the ballistic solution.
  cv::Vec3d target_center_odom_m{0.0, 0.0, 0.0};
  cv::Vec3d muzzle_center_odom_m{0.0, 0.0, 0.0};
  double yaw_command_deg = std::numeric_limits<double>::quiet_NaN();
  double pitch_command_deg = std::numeric_limits<double>::quiet_NaN();
  double launch_elevation_rad = std::numeric_limits<double>::quiet_NaN();
  double time_of_flight_s = std::numeric_limits<double>::quiet_NaN();
  double gravity_drop_m = std::numeric_limits<double>::quiet_NaN();
  double horizontal_distance_m = std::numeric_limits<double>::quiet_NaN();
  double vertical_offset_m = std::numeric_limits<double>::quiet_NaN();
  std::size_t iterations = 0;
};

// 将瞄准结果状态转换为可读的诊断信息。
[[nodiscard]] const char* aimStatusName(AimStatus status) noexcept;

class GimbalAimSolver {
 public:
  // 使用已校验的弹道和瞄准选项创建绝对角度求解器。
  explicit GimbalAimSolver(
      ballistics::VacuumBallisticSolver ballistic_solver =
          ballistics::VacuumBallisticSolver{},
      GimbalAimOptions options = GimbalAimOptions{});

  // 返回模拟器绝对指令角度。pitch_command_deg 遵循模拟器约定：90 度表示水平，
  // 更大值表示向上瞄准。本方法只计算建议值，不发送指令也不触发开火。
  [[nodiscard]] GimbalAimResult solve(
      const AimTarget& target,
      const coordinates::CoordinateSnapshot& snapshot) const noexcept;

  // 返回用于轨迹计算的弹道求解器。
  [[nodiscard]] const ballistics::VacuumBallisticSolver& ballisticSolver()
      const noexcept;
  // 返回已校验的角度限制和迭代设置。
  [[nodiscard]] const GimbalAimOptions& options() const noexcept;

 private:
  ballistics::VacuumBallisticSolver ballistic_solver_;
  GimbalAimOptions options_;
};

}  // namespace yolo_detect::control
