#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <limits>

namespace yolo_detect::coordinates {

// Coordinate frames:
// O: ROS odom, +x forward, +y left, +z up.
// B: chassis ROS frame, +x forward, +y left, +z up.
// G: gimbal/muzzle frame at zero angle, +x fire, +y left, +z up.
// C: OpenCV camera frame, +x right, +y down, +z forward.
// R_XY maps a vector expressed in Y into frame X.

enum class CoordinateStatus {
  Success,
  NonFiniteInput,
  InvalidQuaternion,
  InvalidWorldFrame,
  MissingPoseState,
  CameraOffsetMismatch,
};

struct CoordinateObservation {
  std::uint64_t frame_sequence = 0;
  bool ros_odom_world = false;
  bool has_chassis_pose = false;
  bool has_gimbal_pose = false;
  bool has_camera_pose = false;
  cv::Vec3d chassis_position_odom_m{0.0, 0.0, 0.0};
  cv::Vec4d chassis_quaternion_odom_wxyz{1.0, 0.0, 0.0, 0.0};
  cv::Vec3d gimbal_position_odom_m{0.0, 0.0, 0.0};
  cv::Vec3d measured_camera_position_odom_m{0.0, 0.0, 0.0};
  double gimbal_yaw_rad = 0.0;
  // Simulator internal elevation: zero is level, positive is upward.
  double gimbal_elevation_rad = 0.0;
  cv::Vec3d camera_offset_gimbal_m{0.0, 0.0, 0.0};
  cv::Vec3d muzzle_offset_gimbal_m{0.0, 0.0, 0.0};
};

struct CoordinateSnapshot {
  bool valid = false;
  CoordinateStatus status = CoordinateStatus::NonFiniteInput;
  std::uint64_t frame_sequence = 0;
  cv::Vec3d chassis_position_odom_m{0.0, 0.0, 0.0};
  cv::Matx33d R_OB = cv::Matx33d::eye();
  cv::Vec3d gimbal_position_odom_m{0.0, 0.0, 0.0};
  double gimbal_yaw_rad = 0.0;
  double gimbal_elevation_rad = 0.0;
  cv::Vec3d camera_offset_gimbal_m{0.0, 0.0, 0.0};
  cv::Vec3d muzzle_offset_gimbal_m{0.0, 0.0, 0.0};
  cv::Vec3d camera_position_odom_m{0.0, 0.0, 0.0};
  double camera_position_error_m = std::numeric_limits<double>::quiet_NaN();
};

inline constexpr double kDefaultCameraPositionToleranceM = 0.01;

[[nodiscard]] const char* coordinateStatusName(
    CoordinateStatus status) noexcept;

[[nodiscard]] cv::Matx33d rotationX(double angle_rad) noexcept;
[[nodiscard]] cv::Matx33d rotationY(double angle_rad) noexcept;
[[nodiscard]] cv::Matx33d rotationZ(double angle_rad) noexcept;

[[nodiscard]] cv::Matx33d quaternionWxyzToRotation(
    const cv::Vec4d& quaternion_wxyz, bool* valid = nullptr) noexcept;

// Fixed OpenCV-camera to gimbal-axis mapping:
// (x_G, y_G, z_G) = (z_C, -x_C, -y_C).
[[nodiscard]] const cv::Matx33d& rotationGimbalFromCamera() noexcept;

[[nodiscard]] cv::Matx33d rotationGimbalFromNeutral(
    double yaw_rad, double elevation_rad) noexcept;

[[nodiscard]] CoordinateSnapshot makeCoordinateSnapshot(
    const CoordinateObservation& observation,
    double camera_position_tolerance_m =
        kDefaultCameraPositionToleranceM) noexcept;

[[nodiscard]] cv::Matx33d cameraRotationOdom(
    const CoordinateSnapshot& snapshot) noexcept;

[[nodiscard]] cv::Vec3d cameraPointToOdom(
    const CoordinateSnapshot& snapshot,
    const cv::Vec3d& point_camera_m) noexcept;

[[nodiscard]] cv::Vec3d muzzlePositionOdom(
    const CoordinateSnapshot& snapshot, double yaw_rad,
    double elevation_rad) noexcept;

}  // namespace yolo_detect::coordinates
