#include "control/dynamic_target_controller.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace yolo_detect::control {
namespace {

bool finite(double value) noexcept { return std::isfinite(value); }

bool finite(const cv::Vec3d& value) noexcept {
  return finite(value[0]) && finite(value[1]) && finite(value[2]);
}

double wrappedErrorDegrees(double target, double actual) noexcept {
  return std::abs(std::remainder(target - actual, 360.0));
}

void validateOptions(const DynamicTargetControllerOptions& options) {
  if (!finite(options.alignment_tolerance_deg) ||
      options.alignment_tolerance_deg <= 0.0 ||
      options.alignment_tolerance_deg > 10.0) {
    throw std::invalid_argument("alignment tolerance must be in (0, 10] degrees");
  }
  if (options.fire_timeout.count() <= 0) {
    throw std::invalid_argument("fire timeout must be positive");
  }
}

std::string alignmentStatus(double yaw_error, double pitch_error) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(2)
         << "dynamic aligning: yaw error " << yaw_error
         << " deg, pitch error " << pitch_error << " deg";
  return stream.str();
}

}  // namespace

DynamicTargetController::DynamicTargetController(
    GimbalAimSolver aim_solver, DynamicTargetControllerOptions options)
    : aim_solver_(std::move(aim_solver)), options_(options) {
  validateOptions(options_);
}

void DynamicTargetController::toggleFollowing() {
  if (following_) {
    following_ = false;
    continuous_fire_ = false;
    fire_pending_ = false;
    last_command_fired_ = false;
    active_target_odom_m_.reset();
    status_ = "dynamic follow stopped";
    return;
  }
  following_ = true;
  last_command_fired_ = false;
  active_target_odom_m_.reset();
  status_ = "dynamic follow: waiting for EKF target";
}

void DynamicTargetController::requestFire(TimePoint now) {
  continuous_fire_ = !continuous_fire_;
  fire_pending_ = continuous_fire_;
  last_command_fired_ = false;
  if (continuous_fire_) {
    fire_requested_at_ = now;
    status_ = "continuous dynamic fire armed: waiting for alignment";
  } else {
    status_ = "continuous dynamic fire stopped";
  }
}

DynamicTargetCommand DynamicTargetController::update(
    const std::optional<AimTarget>& predicted_target,
    const coordinates::CoordinateSnapshot& snapshot, TimePoint now) {
  DynamicTargetCommand command;
  static_cast<void>(now);
  if (!following_ && !fire_pending_) return command;
  if (!predicted_target || !finite(predicted_target->center_odom_m)) {
    active_target_odom_m_.reset();
    status_ = fire_pending_ ? "continuous dynamic fire: waiting for EKF target"
                            : "dynamic follow: waiting for EKF target";
    return command;
  }
  if (!snapshot.valid) {
    status_ = "dynamic command blocked: coordinate snapshot invalid";
    return command;
  }

  active_target_odom_m_ = predicted_target->center_odom_m;
  command.aim = aim_solver_.solve(*predicted_target, snapshot);
  if (!command.aim.valid) {
    status_ = std::string("dynamic command blocked: ") +
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
  command.distance_m = cv::norm(command.aim.target_center_odom_m -
                                command.aim.muzzle_center_odom_m);
  status_ = fire_pending_
                ? (aligned ? "continuous dynamic fire: aligned"
                           : alignmentStatus(command.yaw_error_deg,
                                             command.pitch_error_deg))
                : "dynamic target following";
  return command;
}

void DynamicTargetController::acknowledgeCommand(std::uint64_t command_id,
                                                 bool fired) {
  last_command_id_ = command_id;
  last_command_fired_ = fired;
  if (fired) {
    status_ = following_ ? "continuous shot sent; dynamic follow active"
                         : "continuous shot sent";
  }
}

void DynamicTargetController::reportSendFailure(const std::string& message) {
  last_command_fired_ = false;
  status_ = "dynamic gimbal UDP send failed: " + message;
}

bool DynamicTargetController::following() const noexcept { return following_; }
bool DynamicTargetController::firePending() const noexcept { return fire_pending_; }
bool DynamicTargetController::lastCommandFired() const noexcept {
  return last_command_fired_;
}
std::uint64_t DynamicTargetController::lastCommandId() const noexcept {
  return last_command_id_;
}
const std::optional<cv::Vec3d>& DynamicTargetController::activeTargetOdomM()
    const noexcept {
  return active_target_odom_m_;
}
const std::string& DynamicTargetController::status() const noexcept {
  return status_;
}

}  // namespace yolo_detect::control
