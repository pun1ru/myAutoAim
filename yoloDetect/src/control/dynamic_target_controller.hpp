#pragma once

#include "control/gimbal_aim_solver.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

namespace yolo_detect::control {

struct DynamicTargetControllerOptions {
  double alignment_tolerance_deg = 2.0;
  std::chrono::milliseconds fire_timeout{3000};
};

struct DynamicTargetCommand {
  bool valid = false;
  bool fire = false;
  GimbalAimResult aim;
  double distance_m = 0.0;
  double yaw_error_deg = 0.0;
  double pitch_error_deg = 0.0;
};

// Follows the latest EKF-predicted armor target. Target selection and future
// position prediction remain upstream; this controller owns only aiming,
// alignment, fire timeout, and UDP acknowledgement state.
class DynamicTargetController {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  explicit DynamicTargetController(
      GimbalAimSolver aim_solver = GimbalAimSolver{},
      DynamicTargetControllerOptions options = {});

  void toggleFollowing();
  void requestFire(TimePoint now = Clock::now());
  [[nodiscard]] DynamicTargetCommand update(
      const std::optional<AimTarget>& predicted_target,
      const coordinates::CoordinateSnapshot& snapshot,
      TimePoint now = Clock::now());
  void acknowledgeCommand(std::uint64_t command_id, bool fired);
  void reportSendFailure(const std::string& message);

  [[nodiscard]] bool following() const noexcept;
  [[nodiscard]] bool firePending() const noexcept;
  [[nodiscard]] bool lastCommandFired() const noexcept;
  [[nodiscard]] std::uint64_t lastCommandId() const noexcept;
  [[nodiscard]] const std::optional<cv::Vec3d>& activeTargetOdomM()
      const noexcept;
  [[nodiscard]] const std::string& status() const noexcept;

 private:
  GimbalAimSolver aim_solver_;
  DynamicTargetControllerOptions options_;
  bool following_ = false;
  bool continuous_fire_ = false;
  bool fire_pending_ = false;
  bool last_command_fired_ = false;
  TimePoint fire_requested_at_{};
  std::uint64_t last_command_id_ = 0;
  std::optional<cv::Vec3d> active_target_odom_m_;
  std::string status_ = "dynamic follow idle";
};

}  // namespace yolo_detect::control
