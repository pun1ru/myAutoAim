#include "control/predictive_armor_fire_controller.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace yolo_detect::control {
namespace {

constexpr double kPi = 3.14159265358979323846;

bool finite(double value) noexcept { return std::isfinite(value); }

bool finite(const cv::Vec3d& value) noexcept {
  return finite(value[0]) && finite(value[1]) && finite(value[2]);
}

double wrappedErrorDegrees(double target, double actual) noexcept {
  return std::abs(std::remainder(target - actual, 360.0));
}

void validateOptions(const PredictiveArmorFireControllerOptions& options) {
  if (!finite(options.alignment_tolerance_deg) ||
      options.alignment_tolerance_deg <= 0.0 ||
      options.alignment_tolerance_deg > 10.0 ||
      !finite(options.command_latency_s) || options.command_latency_s < 0.0 ||
      !finite(options.future_visibility_cosine) ||
      options.future_visibility_cosine < -1.0 ||
      options.future_visibility_cosine > 1.0 ||
      !finite(options.slot_switch_visibility_margin) ||
      options.slot_switch_visibility_margin < 0.0 ||
      !finite(options.target_vertical_offset_m) ||
      !finite(options.flight_time_tolerance_s) ||
      options.flight_time_tolerance_s <= 0.0 ||
      options.maximum_prediction_iterations == 0 ||
      options.stable_slot_frames <= 0 || options.fire_timeout.count() <= 0) {
    throw std::invalid_argument("invalid predictive fire controller options");
  }
}

}  // namespace

PredictiveArmorFireController::PredictiveArmorFireController(
    GimbalAimSolver aim_solver, PredictiveArmorFireControllerOptions options)
    : aim_solver_(std::move(aim_solver)), options_(options) {
  validateOptions(options_);
}

void PredictiveArmorFireController::toggleFollowing() {
  following_ = !following_;
  fire_pending_ = false;
  last_command_fired_ = false;
  selected_slot_.reset();
  stable_slot_frames_ = 0;
  target_odom_m_.reset();
  status_ = following_ ? "predictive follow enabled" : "predictive follow stopped";
}

void PredictiveArmorFireController::requestFire(TimePoint now) {
  fire_pending_ = true;
  last_command_fired_ = false;
  fire_requested_at_ = now;
  status_ = "predictive fire armed";
}

std::optional<PredictiveArmorFireController::Candidate>
PredictiveArmorFireController::predictCandidate(
    const tracking::TrackOutput& tracker_output, int armor_slot,
    const coordinates::CoordinateSnapshot& snapshot,
    double base_horizon_s) const {
  double horizon_s = base_horizon_s;
  for (std::size_t iteration = 0;
       iteration < options_.maximum_prediction_iterations; ++iteration) {
    const tracking::DecodedArmor armor = tracking::decodeArmor(
        tracker_output.state, armor_slot, horizon_s);
    cv::Vec3d target(armor.position_T_m.x(), armor.position_T_m.y(),
                     armor.position_T_m.z());
    target[2] += options_.target_vertical_offset_m;
    const GimbalAimResult aim = aim_solver_.solve(
        {target, true, horizon_s}, snapshot);
    if (!aim.valid || !finite(aim.time_of_flight_s) ||
        aim.time_of_flight_s < 0.0) {
      return std::nullopt;
    }
    const double next_horizon_s = base_horizon_s + aim.time_of_flight_s;
    if (std::abs(next_horizon_s - horizon_s) >
        options_.flight_time_tolerance_s) {
      horizon_s = next_horizon_s;
      continue;
    }

    const cv::Vec3d to_muzzle = aim.muzzle_center_odom_m - target;
    const double range = cv::norm(to_muzzle);
    if (!finite(to_muzzle) || !finite(range) || range <= 1e-9) {
      return std::nullopt;
    }
    const double yaw = armor.inward_yaw_T_rad;
    const cv::Vec3d outward(-std::cos(yaw), -std::sin(yaw), 0.0);
    const double visibility = outward.dot(to_muzzle) / range;
    if (!finite(visibility) ||
        visibility < options_.future_visibility_cosine) {
      return std::nullopt;
    }
    Candidate candidate;
    candidate.armor_slot = armor_slot;
    candidate.target_odom_m = target;
    candidate.prediction_horizon_s = next_horizon_s;
    candidate.visibility_cosine = visibility;
    candidate.aim = aim;
    candidate.aim.prediction_horizon_s = next_horizon_s;
    return candidate;
  }
  return std::nullopt;
}

std::optional<PredictiveArmorFireController::Candidate>
PredictiveArmorFireController::chooseCandidate(
    const tracking::TrackOutput& tracker_output,
    const coordinates::CoordinateSnapshot& snapshot,
    double base_horizon_s) const {
  std::optional<Candidate> best;
  std::optional<Candidate> retained;
  for (int slot = 0; slot < tracking::kArmorSlotCount; ++slot) {
    const auto candidate = predictCandidate(tracker_output, slot, snapshot,
                                            base_horizon_s);
    if (!candidate) continue;
    if (!best || candidate->visibility_cosine > best->visibility_cosine ||
        (candidate->visibility_cosine == best->visibility_cosine &&
         candidate->aim.horizontal_distance_m < best->aim.horizontal_distance_m)) {
      best = candidate;
    }
    if (selected_slot_ && candidate->armor_slot == *selected_slot_) {
      retained = candidate;
    }
  }
  if (retained && best &&
      retained->visibility_cosine + options_.slot_switch_visibility_margin >=
          best->visibility_cosine) {
    return retained;
  }
  return best;
}

PredictiveArmorFireCommand PredictiveArmorFireController::update(
    const tracking::TrackOutput& tracker_output,
    const coordinates::CoordinateSnapshot& snapshot,
    double processing_latency_s, TimePoint now) {
  PredictiveArmorFireCommand command;
  last_command_fired_ = false;
  if (fire_pending_ && now - fire_requested_at_ > options_.fire_timeout) {
    fire_pending_ = false;
    status_ = "predictive fire cancelled: alignment timeout";
  }
  if (!following_ && !fire_pending_) return command;
  if (!snapshot.valid) {
    status_ = "predictive fire blocked: coordinate snapshot invalid";
    return command;
  }
  if (!tracker_output.has_state ||
      tracker_output.tracking_state != tracking::TrackingState::Tracking ||
      tracker_output.consecutive_misses != 0) {
    status_ = "predictive fire waiting for a fresh tracking state";
    return command;
  }
  const bool fresh_yaw = std::any_of(
      tracker_output.associated_observations.begin(),
      tracker_output.associated_observations.end(),
      [](const tracking::AssociatedObservation& observation) {
        return observation.includes_yaw;
      });
  const double base_horizon_s = options_.command_latency_s +
      std::max(0.0, finite(processing_latency_s) ? processing_latency_s : 0.0);
  const auto candidate = chooseCandidate(tracker_output, snapshot, base_horizon_s);
  if (!candidate) {
    target_odom_m_.reset();
    selected_slot_.reset();
    stable_slot_frames_ = 0;
    status_ = "predictive fire waiting for a future-visible armor";
    return command;
  }

  if (selected_slot_ && *selected_slot_ == candidate->armor_slot) {
    ++stable_slot_frames_;
  } else {
    selected_slot_ = candidate->armor_slot;
    stable_slot_frames_ = 1;
  }
  target_odom_m_ = candidate->target_odom_m;
  command.valid = true;
  command.armor_slot = candidate->armor_slot;
  command.stable_slot_frames = stable_slot_frames_;
  command.prediction_horizon_s = candidate->prediction_horizon_s;
  command.visibility_cosine = candidate->visibility_cosine;
  command.aim = candidate->aim;
  command.distance_m = cv::norm(candidate->target_odom_m -
                                candidate->aim.muzzle_center_odom_m);
  double gate_yaw_rad = snapshot.gimbal_yaw_rad;
  double gate_elevation_rad = snapshot.gimbal_elevation_rad;
  if (snapshot.has_gimbal_velocity) {
    // The command and its fire advice take effect after the command horizon,
    // so evaluate alignment at that time rather than at image exposure.
    gate_yaw_rad +=
        snapshot.gimbal_yaw_velocity_rad_s * base_horizon_s;
    gate_elevation_rad +=
        snapshot.gimbal_elevation_velocity_rad_s * base_horizon_s;
  }
  const double gate_yaw_deg = gate_yaw_rad * 180.0 / kPi;
  const double gate_pitch_deg = 90.0 + gate_elevation_rad * 180.0 / kPi;
  command.yaw_error_deg = wrappedErrorDegrees(
      candidate->aim.yaw_command_deg, gate_yaw_deg);
  command.pitch_error_deg = std::abs(
      candidate->aim.pitch_command_deg - gate_pitch_deg);
  const bool aligned = command.yaw_error_deg <= options_.alignment_tolerance_deg &&
                       command.pitch_error_deg <= options_.alignment_tolerance_deg;
  const bool stable = stable_slot_frames_ >= options_.stable_slot_frames;
  command.fire = fire_pending_ && fresh_yaw && stable && aligned;
  if (fire_pending_) {
    if (!fresh_yaw) {
      status_ = "predictive fire blocked: current yaw was not accepted";
    } else if (!stable) {
      status_ = "predictive fire waiting for a stable armor slot";
    } else if (!aligned) {
      status_ = "predictive fire waiting for gimbal alignment";
    } else {
      status_ = "predictive fire command ready";
    }
  } else {
    status_ = "predictive following E" +
              std::to_string(candidate->armor_slot);
  }
  return command;
}

void PredictiveArmorFireController::acknowledgeCommand(
    std::uint64_t command_id, bool fired) {
  last_command_id_ = command_id;
  last_command_fired_ = fired;
  if (fired) {
    fire_pending_ = false;
    status_ = following_ ? "shot sent; predictive follow active" : "shot sent";
  }
}

void PredictiveArmorFireController::reportSendFailure(const std::string& message) {
  last_command_fired_ = false;
  status_ = "gimbal UDP send failed: " + message;
}

bool PredictiveArmorFireController::following() const noexcept {
  return following_;
}

bool PredictiveArmorFireController::firePending() const noexcept {
  return fire_pending_;
}

bool PredictiveArmorFireController::lastCommandFired() const noexcept {
  return last_command_fired_;
}

std::uint64_t PredictiveArmorFireController::lastCommandId() const noexcept {
  return last_command_id_;
}

const std::optional<cv::Vec3d>&
PredictiveArmorFireController::targetOdomM() const noexcept {
  return target_odom_m_;
}

std::optional<int> PredictiveArmorFireController::selectedArmorSlot() const noexcept {
  return selected_slot_;
}

const std::string& PredictiveArmorFireController::status() const noexcept {
  return status_;
}

}  // namespace yolo_detect::control
