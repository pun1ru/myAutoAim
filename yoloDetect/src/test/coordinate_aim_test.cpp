#include "control/gimbal_aim_solver.hpp"
#include "control/static_target_controller.hpp"
#include "coordinates/coordinate_frames.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

constexpr double kPi = 3.14159265358979323846;

// 在条件不成立时抛出带上下文的测试失败信息。
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

// 验证有限浮点值在给定误差内相等。
void requireNear(double actual, double expected, double tolerance,
                 const std::string& message) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(message + ": actual=" + std::to_string(actual) +
                             " expected=" + std::to_string(expected));
  }
}

// 构造零位姿、单位旋转的有效坐标观测。
yolo_detect::coordinates::CoordinateObservation identityObservation() {
  yolo_detect::coordinates::CoordinateObservation observation;
  observation.frame_sequence = 42;
  observation.ros_odom_world = true;
  observation.has_chassis_pose = true;
  observation.has_gimbal_pose = true;
  observation.has_camera_pose = true;
  return observation;
}

// 验证固定的 OpenCV 相机轴到云台轴映射。
void testCameraAxisMapping() {
  namespace coordinates = yolo_detect::coordinates;
  auto observation = identityObservation();
  observation.gimbal_position_odom_m = {1.0, 2.0, 3.0};
  observation.camera_offset_gimbal_m = {0.2, 0.0, 0.1};
  observation.measured_camera_position_odom_m = {1.2, 2.0, 3.1};

  const coordinates::CoordinateSnapshot snapshot =
      coordinates::makeCoordinateSnapshot(observation);
  require(snapshot.valid, coordinates::coordinateStatusName(snapshot.status));
  const cv::Vec3d transformed =
      coordinates::cameraPointToOdom(snapshot, {1.0, 2.0, 3.0});
  requireNear(transformed[0], 4.2, 1e-12, "camera +z must map to G +x");
  requireNear(transformed[1], 1.0, 1e-12, "camera +x must map to G -y");
  requireNear(transformed[2], 1.1, 1e-12, "camera +y must map to G -z");
}
// 验证模拟器相机/炮口偏移和坐标重建回归结果。
void testSimulatorCalibrationRegression() {
  namespace coordinates = yolo_detect::coordinates;
  auto observation = identityObservation();
  observation.chassis_position_odom_m = {5.40056, 5.79997, 0.236291};
  observation.gimbal_position_odom_m = observation.chassis_position_odom_m;
  const double chassis_yaw = 3.13936;
  observation.chassis_quaternion_odom_wxyz = {
      std::cos(chassis_yaw * 0.5), 0.0, 0.0,
      std::sin(chassis_yaw * 0.5)};
  observation.gimbal_yaw_rad = 0.0;
  observation.gimbal_elevation_rad = (63.0 - 90.0) * kPi / 180.0;
  observation.camera_offset_gimbal_m =
      {0.191966, 0.00183105, 0.194812};
  observation.muzzle_offset_gimbal_m =
      {0.010704, 0.00183058, 0.110288};
  observation.measured_camera_position_odom_m =
      {5.14107, 5.79872, 0.322719};

  const coordinates::CoordinateSnapshot snapshot =
      coordinates::makeCoordinateSnapshot(observation, 1e-3);
  require(snapshot.valid, coordinates::coordinateStatusName(snapshot.status));
  require(snapshot.camera_position_error_m < 1e-3,
          "simulator camera extrinsic regression exceeded tolerance");
  requireNear(snapshot.camera_position_odom_m[0], 5.14107, 1e-3,
              "simulator camera odom x regression");
  requireNear(snapshot.camera_position_odom_m[1], 5.79872, 1e-3,
              "simulator camera odom y regression");
  requireNear(snapshot.camera_position_odom_m[2], 0.322719, 1e-3,
              "simulator camera odom z regression");
}


// 验证不一致的相机偏移会使快照失效。
void testCameraOffsetMismatchRejected() {
  namespace coordinates = yolo_detect::coordinates;
  auto observation = identityObservation();
  observation.camera_offset_gimbal_m = {0.2, 0.0, 0.1};
  observation.measured_camera_position_odom_m = {0.5, 0.0, 0.1};
  const coordinates::CoordinateSnapshot snapshot =
      coordinates::makeCoordinateSnapshot(observation, 0.01);
  require(!snapshot.valid, "camera position mismatch must be rejected");
  require(snapshot.status == coordinates::CoordinateStatus::CameraOffsetMismatch,
          "camera position mismatch returned wrong status");
}

// 用返回的瞄准结果正向验证弹丸能够命中目标。
void verifyProjectileHit(
    const yolo_detect::control::GimbalAimResult& aim,
    const yolo_detect::coordinates::CoordinateSnapshot& snapshot,
    const cv::Vec3d& expected_target) {
  namespace coordinates = yolo_detect::coordinates;
  constexpr double kSpeed = 25.0;
  constexpr double kGravity = 9.81;
  const double yaw = aim.yaw_command_deg * kPi / 180.0;
  const double elevation = (aim.pitch_command_deg - 90.0) * kPi / 180.0;
  const cv::Vec3d direction =
      snapshot.R_OB *
      coordinates::rotationGimbalFromNeutral(yaw, elevation) *
      cv::Vec3d(1.0, 0.0, 0.0);
  cv::Vec3d impact =
      aim.muzzle_center_odom_m + direction * (kSpeed * aim.time_of_flight_s);
  impact[2] -=
      0.5 * kGravity * aim.time_of_flight_s * aim.time_of_flight_s;
  require(cv::norm(impact - expected_target) < 2e-5,
          "computed command does not intersect target");
}

// 验证水平底盘下的瞄准指令和弹道解。
void testLevelChassisAimAndBallistics() {
  namespace control = yolo_detect::control;
  namespace coordinates = yolo_detect::coordinates;
  auto observation = identityObservation();
  observation.muzzle_offset_gimbal_m = {0.2, 0.0, 0.1};
  const coordinates::CoordinateSnapshot snapshot =
      coordinates::makeCoordinateSnapshot(observation);
  require(snapshot.valid, "identity snapshot should be valid");

  const cv::Vec3d target(8.0, 2.0, 1.0);
  const control::GimbalAimResult aim =
      control::GimbalAimSolver{}.solve({target, false, 0.0}, snapshot);
  require(aim.valid, control::aimStatusName(aim.status));
  require(aim.pitch_command_deg > 90.0,
          "gravity compensation should command upward pitch");
  require(aim.iterations >= 2 && aim.iterations <= 12,
          "muzzle-offset iteration count is invalid");
  require(aim.time_of_flight_s > 0.0, "time of flight must be positive");
  verifyProjectileHit(aim, snapshot, target);
}

// 验证底盘偏航被正确从云台绝对指令中消除。
void testChassisYawIsRemovedFromCommand() {
  namespace control = yolo_detect::control;
  namespace coordinates = yolo_detect::coordinates;
  auto observation = identityObservation();
  const double half_yaw = kPi / 4.0;
  observation.chassis_quaternion_odom_wxyz =
      {std::cos(half_yaw), 0.0, 0.0, std::sin(half_yaw)};
  const coordinates::CoordinateSnapshot snapshot =
      coordinates::makeCoordinateSnapshot(observation);
  require(snapshot.valid, "rotated chassis snapshot should be valid");

  const control::GimbalAimResult aim =
      control::GimbalAimSolver{}.solve({{10.0, 0.0, 0.0}, true, 0.2},
                                       snapshot);
  require(aim.valid, control::aimStatusName(aim.status));
  requireNear(aim.yaw_command_deg, -90.0, 1e-8,
              "world target must be converted into chassis command yaw");
  require(aim.predicted && aim.prediction_horizon_s == 0.2,
          "prediction metadata must pass through unchanged");
}

// 验证不可达目标不会产生有效云台指令。
void testUnreachableTargetRejected() {
  namespace control = yolo_detect::control;
  namespace coordinates = yolo_detect::coordinates;
  const coordinates::CoordinateSnapshot snapshot =
      coordinates::makeCoordinateSnapshot(identityObservation());
  const control::GimbalAimResult aim =
      control::GimbalAimSolver{}.solve({{1000.0, 0.0, 0.0}}, snapshot);
  require(!aim.valid, "unreachable target must fail");
  require(aim.status == control::AimStatus::BallisticFailure,
          "unreachable target returned wrong aim status");
  require(aim.ballistic_status ==
              yolo_detect::ballistics::BallisticStatus::Unreachable,
          "unreachable target returned wrong ballistic status");
}

// 验证静态目标锁定、对准后单次开火和停止跟随行为。
void testStaticTargetFollowAndSingleFire() {
  namespace control = yolo_detect::control;
  namespace coordinates = yolo_detect::coordinates;
  using Controller = control::StaticTargetController;

  const coordinates::CoordinateSnapshot initial_snapshot =
      coordinates::makeCoordinateSnapshot(identityObservation());
  require(initial_snapshot.valid, "static target snapshot should be valid");

  Controller controller;
  const Controller::TimePoint start{};
  const cv::Vec3d target(8.0, 2.0, 1.0);
  controller.toggleFollowing();
  require(controller.following(), "follow toggle must enable following");
  require(controller.capturePending(),
          "follow toggle must wait for one target capture");

  const control::StaticTargetCommand first =
      controller.update(target, initial_snapshot, start);
  require(first.valid, "captured static target must produce a command");
  require(!first.fire, "following alone must never request fire");
  require(controller.staticTargetOdomM().has_value(),
          "static target was not latched");
  require(cv::norm(*controller.staticTargetOdomM() - target) < 1e-12,
          "wrong static target was latched");

  const cv::Vec3d later_detection(3.0, -4.0, 2.0);
  static_cast<void>(
      controller.update(later_detection, initial_snapshot, start));
  require(cv::norm(*controller.staticTargetOdomM() - target) < 1e-12,
          "latched target must not follow later detections");

  controller.requestFire(start);
  const control::StaticTargetCommand misaligned =
      controller.update(std::nullopt, initial_snapshot, start);
  require(misaligned.valid, "armed controller must keep aiming");
  require(!misaligned.fire, "misaligned target must not fire");
  require(controller.firePending(),
          "misaligned fire request must remain pending");

  auto aligned_observation = identityObservation();
  aligned_observation.gimbal_yaw_rad =
      misaligned.aim.yaw_command_deg * kPi / 180.0;
  aligned_observation.gimbal_elevation_rad =
      (misaligned.aim.pitch_command_deg - 90.0) * kPi / 180.0;
  const coordinates::CoordinateSnapshot aligned_snapshot =
      coordinates::makeCoordinateSnapshot(aligned_observation);
  require(aligned_snapshot.valid, "aligned snapshot should be valid");

  const control::StaticTargetCommand aligned =
      controller.update(std::nullopt, aligned_snapshot, start);
  require(aligned.valid && aligned.fire,
          "aligned pending request must emit one fire command");
  require(aligned.yaw_error_deg <= 0.5 &&
              aligned.pitch_error_deg <= 0.5,
          "fire command exceeded alignment tolerance");

  controller.acknowledgeCommand(1234, true);
  require(!controller.firePending(),
          "successful fire send must clear pending state");
  require(controller.lastCommandFired(),
          "successful fire send must be observable");
  require(controller.lastCommandId() == 1234,
          "tracked command id was not retained");
  const control::StaticTargetCommand after_fire =
      controller.update(std::nullopt, aligned_snapshot, start);
  require(after_fire.valid && !after_fire.fire,
          "acknowledged fire must not repeat");

  controller.toggleFollowing();
  require(!controller.following() &&
              !controller.staticTargetOdomM().has_value(),
          "stopping follow must clear the static target");
}

// 验证未在期限内对准的开火请求会超时取消。
void testStaticTargetFireTimeout() {
  namespace control = yolo_detect::control;
  namespace coordinates = yolo_detect::coordinates;
  control::StaticTargetControllerOptions options;
  options.fire_timeout = std::chrono::milliseconds(100);
  control::StaticTargetController controller(control::GimbalAimSolver{},
                                               options);
  const auto start = control::StaticTargetController::TimePoint{};
  controller.requestFire(start);
  const auto snapshot =
      coordinates::makeCoordinateSnapshot(identityObservation());
  const control::StaticTargetCommand command = controller.update(
      cv::Vec3d(8.0, 0.0, 1.0), snapshot,
      start + std::chrono::milliseconds(101));
  require(!command.valid && !controller.firePending(),
          "expired fire request must not produce a command");
  require(controller.status().find("timeout") != std::string::npos,
          "expired fire request must report timeout");
}

}  // namespace

// 运行全部坐标与云台瞄准单元测试。
int main() {
  try {
    testCameraAxisMapping();
    testSimulatorCalibrationRegression();
    testCameraOffsetMismatchRejected();
    testLevelChassisAimAndBallistics();
    testChassisYawIsRemovedFromCommand();
    testUnreachableTargetRejected();
    testStaticTargetFollowAndSingleFire();
    testStaticTargetFireTimeout();
    std::cout << "coordinate and gimbal aim tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "coordinate/aim test failed: " << error.what() << '\n';
    return 1;
  }
}
