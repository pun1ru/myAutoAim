#pragma once

#include "control/gimbal_aim_solver.hpp"
#include "tracking/whole_vehicle_ekf.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

namespace yolo_detect::control {

struct PredictiveArmorFireControllerOptions {
  double alignment_tolerance_deg = 1.0;
  double command_latency_s = 0.05;
  double future_visibility_cosine = 0.0;
  double slot_switch_visibility_margin = 0.08;
  // Positive values aim above the armor center in the world-up direction.
  // The current downward bias compensates the observed high impacts.
  double target_vertical_offset_m = -0.05;
  double flight_time_tolerance_s = 0.001;
  std::size_t maximum_prediction_iterations = 6;
  int stable_slot_frames = 2;
  std::chrono::milliseconds fire_timeout{3000};
};

struct PredictiveArmorFireCommand {
  bool valid = false;
  bool fire = false;
  int armor_slot = -1;
  int stable_slot_frames = 0;
  double prediction_horizon_s = 0.0;
  double visibility_cosine = 0.0;
  double distance_m = 0.0;
  double yaw_error_deg = 0.0;
  double pitch_error_deg = 0.0;
  GimbalAimResult aim;
};

// Selects an armor predicted to be visible when the projectile arrives. Fire
// remains explicitly armed by the operator and is rejected for stale tracks.
class PredictiveArmorFireController {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  explicit PredictiveArmorFireController(
      GimbalAimSolver aim_solver = GimbalAimSolver{},
      PredictiveArmorFireControllerOptions options =
          PredictiveArmorFireControllerOptions{});

  void toggleFollowing();
  void requestFire(TimePoint now = Clock::now());

  [[nodiscard]] PredictiveArmorFireCommand update(
      const tracking::TrackOutput& tracker_output,
      const coordinates::CoordinateSnapshot& snapshot,
      double processing_latency_s, TimePoint now = Clock::now());

  void acknowledgeCommand(std::uint64_t command_id, bool fired);
  void reportSendFailure(const std::string& message);

  [[nodiscard]] bool following() const noexcept;
  [[nodiscard]] bool firePending() const noexcept;
  [[nodiscard]] bool lastCommandFired() const noexcept;
  [[nodiscard]] std::uint64_t lastCommandId() const noexcept;
  [[nodiscard]] const std::optional<cv::Vec3d>& targetOdomM() const noexcept;
  [[nodiscard]] std::optional<int> selectedArmorSlot() const noexcept;
  [[nodiscard]] const std::string& status() const noexcept;

 private:
  struct Candidate {
    int armor_slot = -1;
    cv::Vec3d target_odom_m{0.0, 0.0, 0.0};
    double prediction_horizon_s = 0.0;
    double visibility_cosine = -1.0;
    GimbalAimResult aim;
  };

  [[nodiscard]] std::optional<Candidate> predictCandidate(
      const tracking::TrackOutput& tracker_output, int armor_slot,
      const coordinates::CoordinateSnapshot& snapshot,
      double base_horizon_s) const;
  [[nodiscard]] std::optional<Candidate> chooseCandidate(
      const tracking::TrackOutput& tracker_output,
      const coordinates::CoordinateSnapshot& snapshot,
      double base_horizon_s) const;

  GimbalAimSolver aim_solver_;
  PredictiveArmorFireControllerOptions options_;
  bool following_ = false;
  bool fire_pending_ = false;
  bool last_command_fired_ = false;
  TimePoint fire_requested_at_{};
  std::uint64_t last_command_id_ = 0;
  std::optional<int> selected_slot_;
  int stable_slot_frames_ = 0;
  std::optional<cv::Vec3d> target_odom_m_;
  std::string status_ = "predictive fire idle";
};

}  // namespace yolo_detect::control
