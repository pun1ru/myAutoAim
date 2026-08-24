#include "control/gimbal_aim_solver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace yolo_detect::control {
namespace {

constexpr double kPi = 3.14159265358979323846;

// 在坐标或弹道计算前检查标量输入。
bool finite(double value) noexcept { return std::isfinite(value); }

// 检查三维向量的每个分量。
bool finite(const cv::Vec3d& value) noexcept {
  return finite(value[0]) && finite(value[1]) && finite(value[2]);
}

// 将弧度角转换为模拟器使用的角度。
double radiansToDegrees(double radians) noexcept {
  return radians * 180.0 / kPi;
}

// 将角度归一化到带符号的主值弧度范围。
double wrapRadians(double angle) noexcept {
  return std::remainder(angle, 2.0 * kPi);
}

// 将角度归一化到带符号的主值角度范围。
double wrapDegrees(double angle) noexcept {
  return std::remainder(angle, 360.0);
}

// 拒绝无效的俯仰限制和不动点迭代设置。
void validateOptions(const GimbalAimOptions& options) {
  if (!finite(options.minimum_pitch_command_deg) ||
      !finite(options.maximum_pitch_command_deg) ||
      !finite(options.target_height_offset_m) ||
      !finite(options.dynamic_prediction_delay_s) ||
      !finite(options.flight_time_convergence_tolerance_s) ||
      !finite(options.muzzle_direction_tolerance_rad)) {
    throw std::invalid_argument("gimbal aim options must be finite");
  }
  if (options.minimum_pitch_command_deg >=
      options.maximum_pitch_command_deg) {
    throw std::invalid_argument(
        "minimum pitch command must be less than maximum pitch command");
  }
  if (options.dynamic_prediction_delay_s < 0.0) {
    throw std::invalid_argument("dynamic prediction delay must be non-negative");
  }
  if (options.maximum_iterations == 0) {
    throw std::invalid_argument("maximum iterations must be positive");
  }
  if (options.flight_time_convergence_tolerance_s <= 0.0) {
    throw std::invalid_argument("convergence tolerance must be positive");
  }
}

}  // namespace

// 返回瞄准求解结果的诊断信息。
const char* aimStatusName(AimStatus status) noexcept {
  switch (status) {
    case AimStatus::Success:
      return "success";
    case AimStatus::InvalidCoordinateSnapshot:
      return "coordinate snapshot is invalid";
    case AimStatus::NonFiniteTarget:
      return "target center or prediction metadata is non-finite";
    case AimStatus::BallisticFailure:
      return "ballistic solution failed";
    case AimStatus::PitchOutsideLimits:
      return "required pitch is outside simulator limits";
    case AimStatus::DidNotConverge:
      return "muzzle offset iteration did not converge";
  }
  return "unknown aim status";
}

// 保存轨迹求解器并校验指令约束。
GimbalAimSolver::GimbalAimSolver(
    ballistics::VacuumBallisticSolver ballistic_solver,
    GimbalAimOptions options)
    : ballistic_solver_(std::move(ballistic_solver)), options_(options) {
  // Preserve aggregate-initialization compatibility with callers that were
  // compiled against the previous four-field options structure.
  if (options_.muzzle_direction_tolerance_rad <= 0.0) {
    options_.muzzle_direction_tolerance_rad = 1e-7;
  }
  validateOptions(options_);
}

// Iterate the muzzle-offset solution with sp_vision_25's flight-time
// convergence condition rather than an arbitrary yaw/pitch delta.
GimbalAimResult GimbalAimSolver::solve(
    const AimTarget& target,
    const coordinates::CoordinateSnapshot& snapshot) const noexcept {
  GimbalAimResult result;
  result.predicted = target.predicted;
  result.prediction_horizon_s = target.prediction_horizon_s;
  if (!snapshot.valid) {
    result.status = AimStatus::InvalidCoordinateSnapshot;
    return result;
  }
  if (!finite(target.center_odom_m) ||
      !finite(target.prediction_horizon_s) ||
      target.prediction_horizon_s < 0.0) {
    result.status = AimStatus::NonFiniteTarget;
    return result;
  }
  const cv::Vec3d ballistic_target_odom_m(
      target.center_odom_m[0], target.center_odom_m[1],
      target.center_odom_m[2] + options_.target_height_offset_m);
  result.target_center_odom_m = ballistic_target_odom_m;

  const double muzzle_direction_tolerance_rad =
      options_.muzzle_direction_tolerance_rad > 0.0
          ? options_.muzzle_direction_tolerance_rad
          : 1e-7;
  double yaw_rad = snapshot.gimbal_yaw_rad;
  double elevation_rad = snapshot.gimbal_elevation_rad;
  double previous_flight_time_s = std::numeric_limits<double>::quiet_NaN();
  ballistics::BallisticSolution ballistic;

  for (std::size_t iteration = 1;
       iteration <= options_.maximum_iterations; ++iteration) {
    const cv::Vec3d muzzle = coordinates::muzzlePositionOdom(
        snapshot, yaw_rad, elevation_rad);
    const cv::Vec3d relative = ballistic_target_odom_m - muzzle;
    const double horizontal = std::hypot(relative[0], relative[1]);
    const double vertical = relative[2];
    ballistic = ballistic_solver_.solve(
        {horizontal, vertical}, ballistics::TrajectoryArc::Low);

    result.iterations = iteration;
    result.muzzle_center_odom_m = muzzle;
    result.horizontal_distance_m = horizontal;
    result.vertical_offset_m = vertical;
    result.ballistic_status = ballistic.status;
    if (!ballistic.valid) {
      result.status = AimStatus::BallisticFailure;
      return result;
    }

    const double azimuth_odom = std::atan2(relative[1], relative[0]);
    const double cos_elevation = std::cos(ballistic.pitch_rad);
    const cv::Vec3d fire_direction_odom(
        cos_elevation * std::cos(azimuth_odom),
        cos_elevation * std::sin(azimuth_odom),
        std::sin(ballistic.pitch_rad));
    const cv::Vec3d fire_direction_chassis =
        snapshot.R_OB.t() * fire_direction_odom;
    if (!finite(fire_direction_chassis)) {
      result.status = AimStatus::NonFiniteTarget;
      return result;
    }

    const double next_yaw_rad =
        std::atan2(fire_direction_chassis[1],
                   fire_direction_chassis[0]);
    const double next_elevation_rad =
        std::atan2(fire_direction_chassis[2],
                   std::hypot(fire_direction_chassis[0],
                              fire_direction_chassis[1]));
    const double pitch_command_deg =
        90.0 + radiansToDegrees(next_elevation_rad);
    if (pitch_command_deg < options_.minimum_pitch_command_deg ||
        pitch_command_deg > options_.maximum_pitch_command_deg) {
      result.status = AimStatus::PitchOutsideLimits;
      result.yaw_command_deg = wrapDegrees(radiansToDegrees(next_yaw_rad));
      result.pitch_command_deg = pitch_command_deg;
      return result;
    }

    const bool flight_time_converged =
        finite(previous_flight_time_s) &&
        std::abs(ballistic.time_of_flight_s - previous_flight_time_s) <
            options_.flight_time_convergence_tolerance_s;
    const bool muzzle_direction_converged =
        std::abs(wrapRadians(next_yaw_rad - yaw_rad)) <=
            muzzle_direction_tolerance_rad &&
        std::abs(next_elevation_rad - elevation_rad) <=
            muzzle_direction_tolerance_rad;
    const bool converged = flight_time_converged && muzzle_direction_converged;
    previous_flight_time_s = ballistic.time_of_flight_s;
    yaw_rad = next_yaw_rad;
    elevation_rad = next_elevation_rad;
    if (!converged) continue;

    result.valid = true;
    result.status = AimStatus::Success;
    result.yaw_command_deg = wrapDegrees(radiansToDegrees(yaw_rad));
    result.pitch_command_deg = 90.0 + radiansToDegrees(elevation_rad);
    result.launch_elevation_rad = elevation_rad;
    result.time_of_flight_s = ballistic.time_of_flight_s;
    result.gravity_drop_m = ballistic.gravity_drop_m;
    result.muzzle_center_odom_m = coordinates::muzzlePositionOdom(
        snapshot, yaw_rad, elevation_rad);
    return result;
  }

  result.status = AimStatus::DidNotConverge;
  result.yaw_command_deg = wrapDegrees(radiansToDegrees(yaw_rad));
  result.pitch_command_deg = 90.0 + radiansToDegrees(elevation_rad);
  return result;
}

// 返回此瞄准求解器持有的轨迹求解器。
const ballistics::VacuumBallisticSolver& GimbalAimSolver::ballisticSolver()
    const noexcept {
  return ballistic_solver_;
}

// 返回已校验的瞄准约束。
const GimbalAimOptions& GimbalAimSolver::options() const noexcept {
  return options_;
}

}  // namespace yolo_detect::control
