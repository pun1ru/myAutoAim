#include "coordinates/coordinate_frames.hpp"

#include <cmath>

namespace yolo_detect::coordinates {
namespace {

// Checks scalar coordinate input for a finite value.
bool finite(double value) noexcept { return std::isfinite(value); }

// Checks all components of a three-dimensional coordinate.
bool finite(const cv::Vec3d& value) noexcept {
  return finite(value[0]) && finite(value[1]) && finite(value[2]);
}

// Checks all components of a WXYZ quaternion.
bool finite(const cv::Vec4d& value) noexcept {
  return finite(value[0]) && finite(value[1]) && finite(value[2]) &&
         finite(value[3]);
}

}  // namespace

// Returns a diagnostic message for coordinate snapshot validation.
const char* coordinateStatusName(CoordinateStatus status) noexcept {
  switch (status) {
    case CoordinateStatus::Success:
      return "success";
    case CoordinateStatus::NonFiniteInput:
      return "coordinate input contains a non-finite value";
    case CoordinateStatus::InvalidQuaternion:
      return "chassis quaternion is invalid";
    case CoordinateStatus::InvalidWorldFrame:
      return "exposure world frame is not ROS odom";
    case CoordinateStatus::MissingPoseState:
      return "exposure is missing chassis, gimbal, or camera pose";
    case CoordinateStatus::CameraOffsetMismatch:
      return "camera offset does not match exposure camera position";
    case CoordinateStatus::ExposureTimestampMismatch:
      return "exposure timestamp does not match image capture timestamp";
  }
  return "unknown coordinate status";
}

// Constructs a right-handed rotation about the x axis.
cv::Matx33d rotationX(double angle_rad) noexcept {
  const double c = std::cos(angle_rad);
  const double s = std::sin(angle_rad);
  return {1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c};
}

// Constructs a right-handed rotation about the y axis.
cv::Matx33d rotationY(double angle_rad) noexcept {
  const double c = std::cos(angle_rad);
  const double s = std::sin(angle_rad);
  return {c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c};
}

// Constructs a right-handed rotation about the z axis.
cv::Matx33d rotationZ(double angle_rad) noexcept {
  const double c = std::cos(angle_rad);
  const double s = std::sin(angle_rad);
  return {c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0};
}

// Normalizes a WXYZ quaternion and returns its corresponding rotation matrix.
cv::Matx33d quaternionWxyzToRotation(
    const cv::Vec4d& quaternion_wxyz, bool* valid) noexcept {
  if (valid != nullptr) *valid = false;
  if (!finite(quaternion_wxyz)) return cv::Matx33d::eye();

  const double norm_squared = quaternion_wxyz.dot(quaternion_wxyz);
  if (!finite(norm_squared) || norm_squared < 1e-18) {
    return cv::Matx33d::eye();
  }
  const double inverse_norm = 1.0 / std::sqrt(norm_squared);
  const double w = quaternion_wxyz[0] * inverse_norm;
  const double x = quaternion_wxyz[1] * inverse_norm;
  const double y = quaternion_wxyz[2] * inverse_norm;
  const double z = quaternion_wxyz[3] * inverse_norm;

  if (valid != nullptr) *valid = true;
  return {
      1.0 - 2.0 * (y * y + z * z),
      2.0 * (x * y - z * w),
      2.0 * (x * z + y * w),
      2.0 * (x * y + z * w),
      1.0 - 2.0 * (x * x + z * z),
      2.0 * (y * z - x * w),
      2.0 * (x * z - y * w),
      2.0 * (y * z + x * w),
      1.0 - 2.0 * (x * x + y * y),
  };
}

// Returns the fixed axis permutation from OpenCV camera to gimbal frame.
const cv::Matx33d& rotationGimbalFromCamera() noexcept {
  static const cv::Matx33d kR_GC(
      0.0, 0.0, 1.0,
      -1.0, 0.0, 0.0,
      0.0, -1.0, 0.0);
  return kR_GC;
}

// Composes simulator yaw and elevation into the B-from-G rotation.
cv::Matx33d rotationGimbalFromNeutral(
    double yaw_rad, double elevation_rad) noexcept {
  return rotationZ(yaw_rad) * rotationY(-elevation_rad);
}

// Validates exposure metadata and reconstructs all O/B/G/C transforms.
CoordinateSnapshot makeCoordinateSnapshot(
    const CoordinateObservation& observation,
    double camera_position_tolerance_m) noexcept {
  CoordinateSnapshot snapshot;
  snapshot.frame_sequence = observation.frame_sequence;
  snapshot.capture_timestamp_ns = observation.capture_timestamp_ns;

  if (!observation.ros_odom_world) {
    snapshot.status = CoordinateStatus::InvalidWorldFrame;
    return snapshot;
  }
  if (!observation.has_chassis_pose || !observation.has_gimbal_pose ||
      !observation.has_camera_pose) {
    snapshot.status = CoordinateStatus::MissingPoseState;
    return snapshot;
  }
  if (!finite(observation.chassis_position_odom_m) ||
      !finite(observation.chassis_quaternion_odom_wxyz) ||
      !finite(observation.gimbal_position_odom_m) ||
      !finite(observation.measured_camera_position_odom_m) ||
      !finite(observation.gimbal_yaw_rad) ||
      !finite(observation.gimbal_elevation_rad) ||
      !finite(observation.camera_offset_gimbal_m) ||
      !finite(observation.muzzle_offset_gimbal_m) ||
      !finite(camera_position_tolerance_m) ||
      camera_position_tolerance_m < 0.0) {
    snapshot.status = CoordinateStatus::NonFiniteInput;
    return snapshot;
  }

  bool quaternion_valid = false;
  const cv::Matx33d R_OB = quaternionWxyzToRotation(
      observation.chassis_quaternion_odom_wxyz, &quaternion_valid);
  if (!quaternion_valid) {
    snapshot.status = CoordinateStatus::InvalidQuaternion;
    return snapshot;
  }

  snapshot.chassis_position_odom_m = observation.chassis_position_odom_m;
  snapshot.R_OB = R_OB;
  snapshot.gimbal_position_odom_m = observation.gimbal_position_odom_m;
  snapshot.gimbal_yaw_rad = observation.gimbal_yaw_rad;
  snapshot.gimbal_elevation_rad = observation.gimbal_elevation_rad;
  snapshot.camera_offset_gimbal_m = observation.camera_offset_gimbal_m;
  snapshot.muzzle_offset_gimbal_m = observation.muzzle_offset_gimbal_m;

  const cv::Matx33d R_BG = rotationGimbalFromNeutral(
      snapshot.gimbal_yaw_rad, snapshot.gimbal_elevation_rad);
  snapshot.camera_position_odom_m =
      snapshot.gimbal_position_odom_m +
      snapshot.R_OB * R_BG * snapshot.camera_offset_gimbal_m;
  snapshot.camera_position_error_m = cv::norm(
      snapshot.camera_position_odom_m -
      observation.measured_camera_position_odom_m);
  if (!finite(snapshot.camera_position_error_m) ||
      snapshot.camera_position_error_m > camera_position_tolerance_m) {
    snapshot.status = CoordinateStatus::CameraOffsetMismatch;
    return snapshot;
  }

  snapshot.valid = true;
  snapshot.status = CoordinateStatus::Success;
  return snapshot;
}

// Reconstructs the O-from-C rotation from the current chassis and gimbal state.
cv::Matx33d cameraRotationOdom(
    const CoordinateSnapshot& snapshot) noexcept {
  return snapshot.R_OB *
         rotationGimbalFromNeutral(snapshot.gimbal_yaw_rad,
                                   snapshot.gimbal_elevation_rad) *
         rotationGimbalFromCamera();
}

// Transforms a metric OpenCV-camera point to its ROS odom position.
cv::Vec3d cameraPointToOdom(
    const CoordinateSnapshot& snapshot,
    const cv::Vec3d& point_camera_m) noexcept {
  return snapshot.camera_position_odom_m +
         cameraRotationOdom(snapshot) * point_camera_m;
}

cv::Vec3d cameraPointToTracker(
    const CoordinateSnapshot& snapshot,
    const cv::Vec3d& point_camera_m) noexcept {
  return cameraPointToOdom(snapshot, point_camera_m);
}

// Recomputes muzzle position for a candidate yaw/elevation command.
cv::Vec3d muzzlePositionOdom(
    const CoordinateSnapshot& snapshot, double yaw_rad,
    double elevation_rad) noexcept {
  return snapshot.gimbal_position_odom_m +
         snapshot.R_OB *
             rotationGimbalFromNeutral(yaw_rad, elevation_rad) *
             snapshot.muzzle_offset_gimbal_m;
}

}  // namespace yolo_detect::coordinates
