#include "control/static_target_controller.hpp"

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

std::string alignmentStatus(double yaw_error, double pitch_error) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(2)
         << "aligning: yaw error " << yaw_error
         << " deg, pitch error " << pitch_error << " deg";
  return stream.str();
}

}  // namespace

StaticTargetController::StaticTargetController(
    GimbalAimSolver aim_solver, StaticTargetControllerOptions options)
    : aim_solver_(std::move(aim_solver)), options_(options) {
  validateOptions(options_);
}

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

void StaticTargetController::requestFire(TimePoint now) {
  fire_pending_ = true;
  last_command_fired_ = false;
  fire_requested_at_ = now;
  if (!static_target_odom_m_) capture_pending_ = true;
  status_ = "fire armed: waiting for alignment";
}

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

void StaticTargetController::reportSendFailure(const std::string& message) {
  last_command_fired_ = false;
  status_ = "gimbal UDP send failed: " + message;
}

bool StaticTargetController::following() const noexcept { return following_; }

bool StaticTargetController::capturePending() const noexcept {
  return capture_pending_;
}

bool StaticTargetController::firePending() const noexcept {
  return fire_pending_;
}

bool StaticTargetController::lastCommandFired() const noexcept {
  return last_command_fired_;
}

std::uint64_t StaticTargetController::lastCommandId() const noexcept {
  return last_command_id_;
}

const std::optional<cv::Vec3d>&
StaticTargetController::staticTargetOdomM() const noexcept {
  return static_target_odom_m_;
}

const std::string& StaticTargetController::status() const noexcept {
  return status_;
}

}  // namespace yolo_detect::control
