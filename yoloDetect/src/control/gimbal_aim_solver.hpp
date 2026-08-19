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
  std::size_t maximum_iterations = 12;
  double convergence_tolerance_rad = 1e-7;
};

struct GimbalAimResult {
  bool valid = false;
  AimStatus status = AimStatus::InvalidCoordinateSnapshot;
  ballistics::BallisticStatus ballistic_status =
      ballistics::BallisticStatus::NumericalFailure;
  bool predicted = false;
  double prediction_horizon_s = 0.0;
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

[[nodiscard]] const char* aimStatusName(AimStatus status) noexcept;

class GimbalAimSolver {
 public:
  explicit GimbalAimSolver(
      ballistics::VacuumBallisticSolver ballistic_solver =
          ballistics::VacuumBallisticSolver{},
      GimbalAimOptions options = GimbalAimOptions{});

  // Returns absolute simulator command angles. pitch_command_deg uses the
  // simulator convention: 90 degrees is level and larger values aim upward.
  // This method only computes advice; it never sends a command or fires.
  [[nodiscard]] GimbalAimResult solve(
      const AimTarget& target,
      const coordinates::CoordinateSnapshot& snapshot) const noexcept;

  [[nodiscard]] const ballistics::VacuumBallisticSolver& ballisticSolver()
      const noexcept;
  [[nodiscard]] const GimbalAimOptions& options() const noexcept;

 private:
  ballistics::VacuumBallisticSolver ballistic_solver_;
  GimbalAimOptions options_;
};

}  // namespace yolo_detect::control
