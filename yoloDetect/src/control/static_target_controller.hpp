#pragma once

#include "control/gimbal_aim_solver.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

namespace yolo_detect::control {

struct StaticTargetControllerOptions {
  double alignment_tolerance_deg = 0.5;
  std::chrono::milliseconds fire_timeout{3000};
};

struct StaticTargetCommand {
  bool valid = false;
  bool fire = false;
  GimbalAimResult aim;
  double distance_m = 0.0;
  double yaw_error_deg = 0.0;
  double pitch_error_deg = 0.0;
};

// Latches one target in O and keeps solving it as the gimbal moves. This class
// is transport-independent: the application decides how to send the command.
class StaticTargetController {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  explicit StaticTargetController(
      GimbalAimSolver aim_solver = GimbalAimSolver{},
      StaticTargetControllerOptions options =
          StaticTargetControllerOptions{});

  void toggleFollowing();
  void requestFire(TimePoint now = Clock::now());

  [[nodiscard]] StaticTargetCommand update(
      const std::optional<cv::Vec3d>& capture_candidate_odom_m,
      const coordinates::CoordinateSnapshot& snapshot,
      TimePoint now = Clock::now());

  void acknowledgeCommand(std::uint64_t command_id, bool fired);
  void reportSendFailure(const std::string& message);

  [[nodiscard]] bool following() const noexcept;
  [[nodiscard]] bool capturePending() const noexcept;
  [[nodiscard]] bool firePending() const noexcept;
  [[nodiscard]] bool lastCommandFired() const noexcept;
  [[nodiscard]] std::uint64_t lastCommandId() const noexcept;
  [[nodiscard]] const std::optional<cv::Vec3d>& staticTargetOdomM()
      const noexcept;
  [[nodiscard]] const std::string& status() const noexcept;

 private:
  GimbalAimSolver aim_solver_;
  StaticTargetControllerOptions options_;
  bool following_ = false;
  bool capture_pending_ = false;
  bool fire_pending_ = false;
  bool last_command_fired_ = false;
  TimePoint fire_requested_at_{};
  std::uint64_t last_command_id_ = 0;
  std::optional<cv::Vec3d> static_target_odom_m_;
  std::string status_ = "gimbal idle";
};

}  // namespace yolo_detect::control
