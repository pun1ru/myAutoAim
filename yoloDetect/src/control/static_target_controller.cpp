#include "control/static_target_controller.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace yolo_detect::control {
namespace {

// Checks one controller input component for finiteness.
bool finite(double value) noexcept { return std::isfinite(value); }

// Checks a candidate world target before latching it.
bool finite(const cv::Vec3d& value) noexcept {
  return finite(value[0]) && finite(value[1]) && finite(value[2]);
}

// Returns the smallest absolute yaw difference across the +/-180 degree seam.
double wrappedErrorDegrees(double target, double actual) noexcept {
  return std::abs(std::remainder(target - actual, 360.0));
}

// Rejects impossible alignment tolerances and fire timeouts.
void validateOptions(const StaticTargetControllerOptions& options) {
  if (!finite(options.alignment_tolerance_deg) ||
      options.alignment_tolerance_deg <= 0.0 ||
      options.alignment_tolerance_deg > 10.0) {
    throw std::invalid_argument(
        "alignment tolerance must be in (0, 10] degrees");
  }
  if (options.fire_timeout.count() <= 0) {
    throw std::invalid_argument("fire timeout must be positive");
  }
}

// Formats the current residual errors for operator-facing telemetry.
std::string alignmentStatus(double yaw_error, double pitch_error) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(2)
         << "aligning: yaw error " << yaw_error
         << " deg, pitch error " << pitch_error << " deg";
  return stream.str();
}

}  // namespace

// Stores the aim solver and validates the state-machine constraints.
StaticTargetController::StaticTargetController(
    GimbalAimSolver aim_solver, StaticTargetControllerOptions options)
    : aim_solver_(std::move(aim_solver)), options_(options) {
  validateOptions(options_);
}

// Enters target-capture mode or clears the active follow and fire state.
void StaticTargetController::toggleFollowing() {
  if (following_ || capture_pending_) {
    following_ = false;
    capture_pending_ = false;
    fire_pending_ = false;
    last_command_fired_ = false;
    static_target_odom_m_.reset();
    status_ = "static follow stopped";
    return;
  }

  following_ = true;
  capture_pending_ = true;
  last_command_fired_ = false;
  static_target_odom_m_.reset();
  status_ = "waiting for a valid static target";
}

// Arms a single shot and begins its alignment timeout window.
void StaticTargetController::requestFire(TimePoint now) {
  fire_pending_ = true;
  last_command_fired_ = false;
  fire_requested_at_ = now;
  if (!static_target_odom_m_) capture_pending_ = true;
  status_ = "fire armed: waiting for alignment";
}

// Latches a candidate if needed and produces the current gimbal command.
StaticTargetCommand StaticTargetController::update(
    const std::optional<cv::Vec3d>& capture_candidate_odom_m,
    const coordinates::CoordinateSnapshot& snapshot, TimePoint now) {
  StaticTargetCommand command;

  if (capture_pending_ && capture_candidate_odom_m &&
      finite(*capture_candidate_odom_m)) {
    static_target_odom_m_ = *capture_candidate_odom_m;
    capture_pending_ = false;
    status_ = "static target locked";
  }

  if (fire_pending_ && now - fire_requested_at_ > options_.fire_timeout) {
    fire_pending_ = false;
    if (!following_) capture_pending_ = false;
    status_ = "fire cancelled: alignment timeout";
  }

  if (!following_ && !fire_pending_) return command;
  if (!static_target_odom_m_) {
    status_ = fire_pending_ ? "fire armed: waiting for a valid target"
                            : "waiting for a valid static target";
    return command;
  }
  if (!snapshot.valid) {
    status_ = "gimbal command blocked: coordinate snapshot invalid";
    return command;
  }

  command.aim =
      aim_solver_.solve({*static_target_odom_m_, false, 0.0}, snapshot);
  if (!command.aim.valid) {
    status_ = std::string("gimbal command blocked: ") +
              aimStatusName(command.aim.status);
    return command;
  }

  const double actual_yaw_deg = snapshot.gimbal_yaw_rad * 180.0 / CV_PI;
  const double actual_pitch_deg =
      90.0 + snapshot.gimbal_elevation_rad * 180.0 / CV_PI;
  command.yaw_error_deg =
      wrappedErrorDegrees(command.aim.yaw_command_deg, actual_yaw_deg);
  command.pitch_error_deg =
      std::abs(command.aim.pitch_command_deg - actual_pitch_deg);
  const bool aligned =
      command.yaw_error_deg <= options_.alignment_tolerance_deg &&
      command.pitch_error_deg <= options_.alignment_tolerance_deg;

  command.valid = true;
  command.fire = fire_pending_ && aligned;
  command.distance_m =
      cv::norm(*static_target_odom_m_ - command.aim.muzzle_center_odom_m);
  if (fire_pending_) {
    status_ = aligned ? "aligned: fire command ready"
                      : alignmentStatus(command.yaw_error_deg,
                                        command.pitch_error_deg);
  } else {
    status_ = "static target following";
  }
  return command;
}

// Records transport success and consumes a pending fire request when fired.
void StaticTargetController::acknowledgeCommand(
    std::uint64_t command_id, bool fired) {
  last_command_id_ = command_id;
  last_command_fired_ = fired;
  if (fired) {
    fire_pending_ = false;
    status_ = following_ ? "shot sent; static follow active" : "shot sent";
  } else if (!fire_pending_) {
    status_ = "static target following";
  }
}

// Preserves controller state while surfacing an outgoing command failure.
void StaticTargetController::reportSendFailure(const std::string& message) {
  last_command_fired_ = false;
  status_ = "gimbal UDP send failed: " + message;
}

// Reports whether static follow mode is active.
bool StaticTargetController::following() const noexcept { return following_; }

// Reports whether a new valid target still needs to be latched.
bool StaticTargetController::capturePending() const noexcept {
  return capture_pending_;
}

// Reports whether a one-shot fire request is still armed.
bool StaticTargetController::firePending() const noexcept {
  return fire_pending_;
}

// Reports whether the latest acknowledged command carried a fire request.
bool StaticTargetController::lastCommandFired() const noexcept {
  return last_command_fired_;
}

// Returns the identifier assigned to the latest acknowledged command.
std::uint64_t StaticTargetController::lastCommandId() const noexcept {
  return last_command_id_;
}

const std::optional<cv::Vec3d>&
// Returns the fixed odom target, if capture has completed.
StaticTargetController::staticTargetOdomM() const noexcept {
  return static_target_odom_m_;
}

// Returns the latest state-machine status for UI and telemetry.
const std::string& StaticTargetController::status() const noexcept {
  return status_;
}

}  // namespace yolo_detect::control
