#include "pose/armor_pose_estimator.hpp"

#include <opencv2/calib3d.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

int failures = 0;

// 在条件失败时记录测试断言。
void expect(bool condition, const char* message) {
  if (condition) return;
  std::cerr << "FAIL: " << message << '\n';
  ++failures;
}

// 返回与模拟器一致的固定测试相机标定。
yolo_detect::CameraCalibration testCalibration() {
  yolo_detect::CameraCalibration calibration;
  calibration.camera_matrix = cv::Matx33d(
      1303.67532368147, 0.0, 720.0, 0.0, 1303.67532368147, 540.0, 0.0,
      0.0, 1.0);
  calibration.distortion_coefficients = {0.0, 0.0, 0.0, 0.0, 0.0};
  calibration.image_size = {1440, 1080};
  return calibration;
}

// 使用已知位姿投影出一组无噪声装甲板图像点。
std::array<cv::Point2f, yolo_detect::kArmorPointCount> syntheticImagePoints(
    const yolo_detect::CameraCalibration& calibration, cv::Vec3d& rvec,
    cv::Vec3d& tvec) {
  const cv::Matx33d front_facing_rotation(
      0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0);
  cv::Mat tilt_matrix;
  cv::Rodrigues(cv::Vec3d(0.08, -0.16, 0.04), tilt_matrix);
  const cv::Matx33d tilt(
      tilt_matrix.at<double>(0, 0), tilt_matrix.at<double>(0, 1),
      tilt_matrix.at<double>(0, 2), tilt_matrix.at<double>(1, 0),
      tilt_matrix.at<double>(1, 1), tilt_matrix.at<double>(1, 2),
      tilt_matrix.at<double>(2, 0), tilt_matrix.at<double>(2, 1),
      tilt_matrix.at<double>(2, 2));
  const cv::Matx33d rotation_camera_from_armor =
      front_facing_rotation * tilt;
  cv::Mat rvec_matrix;
  cv::Rodrigues(rotation_camera_from_armor, rvec_matrix);
  rvec = {rvec_matrix.at<double>(0, 0), rvec_matrix.at<double>(1, 0),
          rvec_matrix.at<double>(2, 0)};
  tvec = {0.12, -0.05, 4.0};
  const auto points = yolo_detect::ArmorPoseEstimator::objectPoints(
      yolo_detect::ArmorSize::Small);
  const std::vector<cv::Point3d> object_points(points.begin(), points.end());
  std::vector<cv::Point2d> projected;
  cv::projectPoints(object_points, rvec, tvec, calibration.camera_matrix,
                    calibration.distortion_coefficients, projected);
  return {cv::Point2f(projected[0]), cv::Point2f(projected[1]),
          cv::Point2f(projected[2]), cv::Point2f(projected[3])};
}

// 验证大小装甲板模型尺寸及其关键点顺序。
void testObjectPointDefinitions() {
  const auto small = yolo_detect::ArmorPoseEstimator::objectPoints(
      yolo_detect::ArmorSize::Small);
  const auto large = yolo_detect::ArmorPoseEstimator::objectPoints(
      yolo_detect::ArmorSize::Large);
  expect(std::abs(small[0].y - 0.0675) < 1e-12,
         "small armor width must be 0.135 m");
  expect(std::abs(large[0].y - 0.1125) < 1e-12,
         "large armor width must be 0.225 m");
  expect(std::abs(small[0].z + 0.0275) < 1e-12,
         "armor height must be 0.055 m");
  expect(small[0].y == small[1].y && small[2].y == small[3].y,
         "object point order must begin with BL, TL");
  expect(small[1].z == small[2].z && small[3].z == small[0].z,
         "object point order must end with TR, BR");
}

// 验证 IPPE 能恢复合成小装甲板的位姿。
void testSyntheticSmallArmorPose() {
  const auto calibration = testCalibration();
  const yolo_detect::ArmorPoseEstimator estimator(calibration);
  cv::Vec3d expected_rvec;
  cv::Vec3d expected_tvec;
  const auto image_points =
      syntheticImagePoints(calibration, expected_rvec, expected_tvec);
  const yolo_detect::PoseResult pose = estimator.estimate(
      image_points, calibration.image_size, yolo_detect::ArmorSize::Small);
  if (!pose.valid) {
    std::cerr << "synthetic pose status: "
              << yolo_detect::poseStatusName(pose.status) << '\n';
  }

  expect(pose.valid, "synthetic small armor pose must solve");
  expect(pose.status == yolo_detect::PoseStatus::Success,
         "successful pose must report Success");
  expect(pose.candidate_count >= 1,
         "IPPE must return at least one complete candidate");
  expect(cv::norm(pose.tvec_m - expected_tvec) < 1e-3,
         "translation must recover within 1 mm");
  expect(pose.tvec_m[2] > 0.0, "selected translation must have positive depth");
  expect(pose.reprojection_rms_px < 1e-3,
         "noise-free reprojection RMS must be below 0.001 px");
  expect(cv::norm(pose.center_camera_m - pose.tvec_m) < 1e-12,
         "armor center must equal tvec because the model origin is centered");
  cv::Mat expected_rotation;
  cv::Rodrigues(expected_rvec, expected_rotation);
  cv::Mat returned_rotation;
  cv::Rodrigues(pose.rvec_rad, returned_rotation);
  double rotation_error = 0.0;
  double rvec_matrix_error = 0.0;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      const double expected = expected_rotation.at<double>(row, column);
      const double solved = pose.rotation_camera_from_armor(row, column);
      rotation_error += (expected - solved) * (expected - solved);
      const double from_rvec = returned_rotation.at<double>(row, column);
      rvec_matrix_error += (from_rvec - solved) * (from_rvec - solved);
    }
  }
  expect(std::sqrt(rotation_error) < 1e-3,
         "R_CA must be returned in the public armor coordinate frame");
  expect(std::sqrt(rvec_matrix_error) < 1e-9,
         "rvec must be the Rodrigues representation of R_CA");
  expect(std::abs(cv::norm(pose.armor_normal_camera) - 1.0) < 1e-9,
         "rotated armor normal must remain a unit vector");
}

// 验证分辨率、非有限点和错误点顺序的处理。
void testInputValidationAndOrdering() {
  const auto calibration = testCalibration();
  const yolo_detect::ArmorPoseEstimator estimator(calibration);
  cv::Vec3d rvec;
  cv::Vec3d tvec;
  const auto points = syntheticImagePoints(calibration, rvec, tvec);

  const auto wrong_size = estimator.estimate(
      points, {1280, 720}, yolo_detect::ArmorSize::Small);
  expect(wrong_size.status == yolo_detect::PoseStatus::ImageSizeMismatch,
         "calibration/image resolution mismatch must be explicit");

  auto non_finite = points;
  non_finite[0].x = std::numeric_limits<float>::quiet_NaN();
  expect(estimator
             .estimate(non_finite, calibration.image_size,
                       yolo_detect::ArmorSize::Small)
             .status == yolo_detect::PoseStatus::NonFiniteImagePoint,
         "non-finite point must be rejected");

  for (std::size_t first = 0; first < points.size(); ++first) {
    for (std::size_t second = first + 1; second < points.size(); ++second) {
      auto swapped = points;
      std::swap(swapped[first], swapped[second]);
      const auto pose = estimator.estimate(
          swapped, calibration.image_size, yolo_detect::ArmorSize::Small);
      expect(!pose.valid || pose.reprojection_rms_px > 0.5,
             "a pairwise keypoint swap must fail or increase RMS");
    }
  }
}

// 验证无效相机内参在构造估计器时被拒绝。
void testInvalidCalibration() {
  auto calibration = testCalibration();
  calibration.camera_matrix(0, 0) = 0.0;
  bool rejected = false;
  try {
    const yolo_detect::ArmorPoseEstimator estimator(calibration);
    (void)estimator;
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  expect(rejected, "non-positive focal length must reject calibration");
}

}  // namespace

// 运行全部 PnP 单元测试并返回断言结果。
int main() {
  testObjectPointDefinitions();
  testSyntheticSmallArmorPose();
  testInputValidationAndOrdering();
  testInvalidCalibration();
  if (failures != 0) {
    std::cerr << failures << " PnP test assertion(s) failed\n";
    return 1;
  }
  std::cout << "PnP synthetic pose and validation tests passed\n";
  return 0;
}
