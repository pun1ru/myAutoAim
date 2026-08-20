#include "coordinates/coordinate_frames.hpp"
#include "tracking/constrained_yaw_solver.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

double wrapToPi(double angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}

cv::Matx33d rotationTrackerFromArmor(double inward_yaw_T_rad) {
  const double c = std::cos(inward_yaw_T_rad);
  const double s = std::sin(inward_yaw_T_rad);
  const cv::Vec3d x_out_T(-c, -s, 0.0);
  const cv::Vec3d z_up_T(0.0, 0.0, 1.0);
  const cv::Vec3d y_left_T = z_up_T.cross(x_out_T);
  return {
      x_out_T[0], y_left_T[0], z_up_T[0],
      x_out_T[1], y_left_T[1], z_up_T[1],
      x_out_T[2], y_left_T[2], z_up_T[2],
  };
}

yolo_detect::coordinates::CoordinateSnapshot exposureSnapshot() {
  namespace coordinates = yolo_detect::coordinates;
  coordinates::CoordinateObservation observation;
  observation.frame_sequence = 7;
  observation.capture_timestamp_ns = 123456789ULL;
  observation.ros_odom_world = true;
  observation.has_chassis_pose = true;
  observation.has_gimbal_pose = true;
  observation.has_camera_pose = true;
  const auto snapshot = coordinates::makeCoordinateSnapshot(observation);
  require(snapshot.valid, "synthetic exposure snapshot must be valid");
  return snapshot;
}

yolo_detect::CameraCalibration calibration() {
  yolo_detect::CameraCalibration result;
  result.camera_matrix = {1000.0, 0.0, 720.0,
                          0.0, 1000.0, 540.0,
                          0.0, 0.0, 1.0};
  result.image_size = {1440, 1080};
  return result;
}

std::array<cv::Point2f, yolo_detect::kArmorPointCount> imagePoints(
    const yolo_detect::coordinates::CoordinateSnapshot& snapshot,
    const yolo_detect::CameraCalibration& camera, double yaw_T_rad,
    const cv::Vec3d& center_camera_m) {
  const cv::Matx33d R_CA =
      yolo_detect::coordinates::cameraRotationOdom(snapshot).t() *
      rotationTrackerFromArmor(yaw_T_rad);
  cv::Mat rvec;
  cv::Rodrigues(R_CA, rvec);
  const auto object_point_array =
      yolo_detect::ArmorPoseEstimator::objectPoints(
          yolo_detect::ArmorSize::Small);
  const std::vector<cv::Point3d> object_points(object_point_array.begin(),
                                                object_point_array.end());
  std::vector<cv::Point2d> projected;
  cv::projectPoints(object_points, rvec, center_camera_m, camera.camera_matrix,
                    camera.distortion_coefficients, projected);
  require(projected.size() == yolo_detect::kArmorPointCount,
          "must project all synthetic corners");
  std::array<cv::Point2f, yolo_detect::kArmorPointCount> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = cv::Point2f(projected[index]);
  }
  return result;
}

void testRecoversYawWithoutPnpRotation() {
  const auto snapshot = exposureSnapshot();
  const auto camera = calibration();
  constexpr double kExpectedYaw = 0.35;
  yolo_detect::PoseResult pose;
  pose.valid = true;
  pose.armor_size = yolo_detect::ArmorSize::Small;
  pose.center_camera_m = {0.0, 0.0, 5.0};
  pose.rvec_rad = {99.0, -77.0, 55.0};
  pose.rotation_camera_from_armor = cv::Matx33d::eye();

  const auto points = imagePoints(snapshot, camera, kExpectedYaw,
                                  pose.center_camera_m);
  const yolo_detect::tracking::ConstrainedYawSolver solver(camera);
  const auto result = solver.solve(snapshot, pose, points);
  require(result.valid, "perfect constrained projection must produce reliable yaw");
  require(std::abs(wrapToPi(result.inward_yaw_T_rad - kExpectedYaw)) < 1e-3,
          "constrained yaw must match synthetic tracker yaw");
  require(result.reprojection_rms_px < 1e-3,
          "synthetic constrained yaw reprojection must be near zero");
}

void testRejectsInconsistentCorners() {
  const auto snapshot = exposureSnapshot();
  const auto camera = calibration();
  yolo_detect::PoseResult pose;
  pose.valid = true;
  pose.armor_size = yolo_detect::ArmorSize::Small;
  pose.center_camera_m = {0.0, 0.0, 5.0};
  auto points = imagePoints(snapshot, camera, 0.0, pose.center_camera_m);
  for (cv::Point2f& point : points) point.x += 20.0F;

  const yolo_detect::tracking::ConstrainedYawSolver solver(camera);
  const auto result = solver.solve(snapshot, pose, points);
  require(!result.valid, "inconsistent corners must not produce reliable yaw");
}

}  // namespace

int main() {
  try {
    testRecoversYawWithoutPnpRotation();
    testRejectsInconsistentCorners();
    std::cout << "constrained yaw solver tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "constrained yaw solver test failed: " << error.what() << '\n';
    return 1;
  }
}
