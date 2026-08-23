#include "control/predictive_armor_fire_controller.hpp"
#include "coordinates/coordinate_frames.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

constexpr double kPi = 3.14159265358979323846;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

yolo_detect::coordinates::CoordinateSnapshot identitySnapshot() {
  namespace coordinates = yolo_detect::coordinates;
  coordinates::CoordinateObservation observation;
  observation.ros_odom_world = true;
  observation.has_chassis_pose = true;
  observation.has_gimbal_pose = true;
  observation.has_camera_pose = true;
  const auto snapshot = coordinates::makeCoordinateSnapshot(observation);
  require(snapshot.valid, "identity coordinate snapshot must be valid");
  return snapshot;
}

yolo_detect::tracking::TrackOutput trackedOutput() {
  namespace tracking = yolo_detect::tracking;
  tracking::TrackOutput output;
  output.has_state = true;
  output.tracking_state = tracking::TrackingState::Tracking;
  output.consecutive_misses = 0;
  output.state.x[tracking::CenterX] = 8.0;
  output.state.x[tracking::CenterY] = 0.0;
  output.state.x[tracking::CenterZ] = 1.0;
  output.state.x[tracking::Theta] = 0.0;
  output.state.x[tracking::Omega] = 2.0;
  output.state.x[tracking::RadiusEven] = 0.2;
  output.associated_observations.push_back({0, 0, 0.0, true});
  return output;
}

void testPredictsFutureArmorAndFiresWhenAligned() {
  namespace control = yolo_detect::control;
  namespace coordinates = yolo_detect::coordinates;
  control::PredictiveArmorFireControllerOptions options;
  options.command_latency_s = 0.0;
  options.stable_slot_frames = 1;
  control::PredictiveArmorFireController controller(
      control::GimbalAimSolver{}, options);
  const auto output = trackedOutput();
  const auto snapshot = identitySnapshot();
  const control::PredictiveArmorFireController::TimePoint now{};

  controller.toggleFollowing();
  const auto first = controller.update(output, snapshot, 0.0, now);
  require(first.valid, "tracking state must produce a predictive aim command");
  require(first.aim.predicted,
          "predictive controller must mark the selected target as predicted");
  require(first.prediction_horizon_s > 0.0,
          "prediction horizon must include projectile flight time");
  require(first.armor_slot >= 0 && first.armor_slot < 4,
          "predictive controller selected an invalid armor slot");
  require(!first.fire, "following alone must not fire");

  controller.requestFire(now);
  auto aligned = coordinates::CoordinateObservation{};
  aligned.ros_odom_world = true;
  aligned.has_chassis_pose = true;
  aligned.has_gimbal_pose = true;
  aligned.has_camera_pose = true;
  aligned.gimbal_yaw_rad = first.aim.yaw_command_deg * kPi / 180.0;
  aligned.gimbal_elevation_rad =
      (first.aim.pitch_command_deg - 90.0) * kPi / 180.0;
  const auto aligned_snapshot = coordinates::makeCoordinateSnapshot(aligned);
  require(aligned_snapshot.valid, "aligned coordinate snapshot must be valid");
  const auto fire = controller.update(output, aligned_snapshot, 0.0, now);
  require(fire.valid && fire.fire,
          "aligned predictive fire request must emit one fire command");
}

void testRejectsStaleOrYawlessTrackForFire() {
  namespace control = yolo_detect::control;
  control::PredictiveArmorFireControllerOptions options;
  options.command_latency_s = 0.0;
  options.stable_slot_frames = 1;
  control::PredictiveArmorFireController controller(
      control::GimbalAimSolver{}, options);
  auto output = trackedOutput();
  output.associated_observations.front().includes_yaw = false;
  const auto snapshot = identitySnapshot();
  const control::PredictiveArmorFireController::TimePoint now{};
  controller.requestFire(now);
  const auto command = controller.update(output, snapshot, 0.0, now);
  require(command.valid && !command.fire,
          "yawless current observation must block predictive fire");
}

void testAppliesVerticalAimOffset() {
  namespace control = yolo_detect::control;
  control::PredictiveArmorFireControllerOptions options;
  options.command_latency_s = 0.0;
  options.target_vertical_offset_m = -0.05;
  control::PredictiveArmorFireController controller(
      control::GimbalAimSolver{}, options);
  const auto output = trackedOutput();
  controller.toggleFollowing();
  const auto command = controller.update(
      output, identitySnapshot(), 0.0,
      control::PredictiveArmorFireController::TimePoint{});
  require(command.valid, "vertical-offset test needs a valid aim command");
  require(std::abs(command.aim.target_center_odom_m[2] - 0.95) < 1e-9,
          "predictive aim target must include the configured vertical offset");
}

void testPredictsGimbalAtCommandHorizonForFireGate() {
  namespace control = yolo_detect::control;
  namespace coordinates = yolo_detect::coordinates;
  control::PredictiveArmorFireControllerOptions options;
  options.command_latency_s = 0.1;
  options.stable_slot_frames = 1;
  control::PredictiveArmorFireController controller(
      control::GimbalAimSolver{}, options);
  const auto output = trackedOutput();
  const auto identity = identitySnapshot();
  const control::PredictiveArmorFireController::TimePoint now{};

  controller.toggleFollowing();
  const auto reference = controller.update(output, identity, 0.0, now);
  require(reference.valid, "gimbal prediction test needs a valid aim command");
  controller.requestFire(now);

  coordinates::CoordinateObservation observation;
  observation.ros_odom_world = true;
  observation.has_chassis_pose = true;
  observation.has_gimbal_pose = true;
  observation.has_camera_pose = true;
  constexpr double kCurrentYawLagRad = 0.02;
  observation.gimbal_yaw_rad =
      reference.aim.yaw_command_deg * kPi / 180.0 - kCurrentYawLagRad;
  observation.gimbal_elevation_rad =
      (reference.aim.pitch_command_deg - 90.0) * kPi / 180.0;
  observation.has_gimbal_velocity = true;
  observation.gimbal_yaw_velocity_rad_s =
      kCurrentYawLagRad / options.command_latency_s;
  const auto predicted_snapshot =
      coordinates::makeCoordinateSnapshot(observation);
  require(predicted_snapshot.valid,
          "predicted-gimbal coordinate snapshot must be valid");
  const auto fire = controller.update(output, predicted_snapshot, 0.0, now);
  require(fire.valid && fire.fire,
          "command-horizon gimbal prediction must permit aligned fire");
}

}  // namespace

int main() {
  try {
    testPredictsFutureArmorAndFiresWhenAligned();
    testRejectsStaleOrYawlessTrackForFire();
    testAppliesVerticalAimOffset();
    testPredictsGimbalAtCommandHorizonForFireGate();
    std::cout << "predictive fire controller tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "predictive fire controller test failed: " << error.what()
              << '\n';
    return 1;
  }
}
