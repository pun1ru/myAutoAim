#include "tracking/constrained_yaw_solver.hpp"

#include <Eigen/QR>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace yolo_detect::tracking {
namespace {

constexpr double kPi = 3.14159265358979323846;

bool finite(double value) { return std::isfinite(value); }

bool finite(const cv::Vec3d& value) {
  return finite(value[0]) && finite(value[1]) && finite(value[2]);
}

bool finite(const cv::Point2f& point) {
  return finite(point.x) && finite(point.y);
}

double wrapToPi(double angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}

cv::Matx33d rotationTrackerFromArmor(double inward_yaw_T_rad) {
  const double c = std::cos(inward_yaw_T_rad);
  const double s = std::sin(inward_yaw_T_rad);
  // ArmorPoseEstimator's right-handed point convention has +x_A point
  // inward. Its +y_A is the visual left direction and +z_A is up.
  const cv::Vec3d x_in_T(c, s, 0.0);
  const cv::Vec3d z_up_T(0.0, 0.0, 1.0);
  const cv::Vec3d y_left_T = z_up_T.cross(x_in_T);
  return {
      x_in_T[0], y_left_T[0], z_up_T[0],
      x_in_T[1], y_left_T[1], z_up_T[1],
      x_in_T[2], y_left_T[2], z_up_T[2],
  };
}

struct ProjectionEvaluation {
  double squared_error_px = std::numeric_limits<double>::infinity();
  cv::Matx33d rotation_camera_from_armor = cv::Matx33d::eye();
  cv::Vec3d center_camera_m{0.0, 0.0, 0.0};
  std::array<cv::Point2d, kArmorPointCount> projected_points{};
};

using NormalizedImagePoints = std::array<cv::Point2d, kArmorPointCount>;

double facingCosine(const ProjectionEvaluation& evaluation) {
  const cv::Vec3d outward_normal_camera(
      -evaluation.rotation_camera_from_armor(0, 0),
      -evaluation.rotation_camera_from_armor(1, 0),
      -evaluation.rotation_camera_from_armor(2, 0));
  const double range = cv::norm(evaluation.center_camera_m);
  if (!finite(range) || range <= 1e-9) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return outward_normal_camera.dot(-evaluation.center_camera_m) / range;
}

bool solveConstrainedTranslation(
    const cv::Matx33d& rotation_camera_from_armor, ArmorSize armor_size,
    const NormalizedImagePoints& normalized_points,
    cv::Vec3d* center_camera_m) {
  constexpr int kEquationCount = 2 * static_cast<int>(kArmorPointCount);
  Eigen::Matrix<double, kEquationCount, 3> system;
  Eigen::Matrix<double, kEquationCount, 1> right_hand_side;
  const auto object_points = ArmorPoseEstimator::objectPoints(armor_size);
  for (std::size_t index = 0; index < object_points.size(); ++index) {
    const cv::Point3d& point = object_points[index];
    const cv::Vec3d rotated = rotation_camera_from_armor *
                              cv::Vec3d(point.x, point.y, point.z);
    const double u = normalized_points[index].x;
    const double v = normalized_points[index].y;
    const int row = 2 * static_cast<int>(index);
    system.row(row) << 1.0, 0.0, -u;
    right_hand_side[row] = u * rotated[2] - rotated[0];
    system.row(row + 1) << 0.0, 1.0, -v;
    right_hand_side[row + 1] = v * rotated[2] - rotated[1];
  }

  Eigen::ColPivHouseholderQR<decltype(system)> qr(system);
  if (qr.rank() != 3) return false;
  const Eigen::Vector3d translation = qr.solve(right_hand_side);
  if (!translation.allFinite()) return false;
  const cv::Vec3d translation_cv(translation.x(), translation.y(),
                                 translation.z());
  for (const cv::Point3d& point : object_points) {
    const cv::Vec3d camera_point =
        rotation_camera_from_armor * cv::Vec3d(point.x, point.y, point.z) +
        translation_cv;
    if (!finite(camera_point) || camera_point[2] <= 1e-6) return false;
  }
  *center_camera_m = translation_cv;
  return true;
}

ProjectionEvaluation evaluateYaw(
    double inward_yaw_T_rad, const cv::Matx33d& rotation_tracker_from_camera,
    const CameraCalibration& calibration, ArmorSize armor_size,
    const NormalizedImagePoints& normalized_points,
    const std::array<cv::Point2f, kArmorPointCount>& image_points) {
  ProjectionEvaluation evaluation;
  // 候选 yaw 先构成 T-from-A，再用同曝光 R_TC 转到 OpenCV 相机系投影。
  const cv::Matx33d rotation_camera_from_tracker =
      rotation_tracker_from_camera.t();
  evaluation.rotation_camera_from_armor =
      rotation_camera_from_tracker *
      rotationTrackerFromArmor(inward_yaw_T_rad);
  if (!solveConstrainedTranslation(evaluation.rotation_camera_from_armor,
                                   armor_size, normalized_points,
                                   &evaluation.center_camera_m)) {
    return evaluation;
  }

  cv::Mat rvec;
  cv::Rodrigues(evaluation.rotation_camera_from_armor, rvec);
  const auto object_point_array = ArmorPoseEstimator::objectPoints(armor_size);
  const std::vector<cv::Point3d> object_points(object_point_array.begin(),
                                                object_point_array.end());
  std::vector<cv::Point2d> projected;
  cv::projectPoints(object_points, rvec, evaluation.center_camera_m,
                    calibration.camera_matrix,
                    calibration.distortion_coefficients, projected);
  if (projected.size() != image_points.size()) return evaluation;

  double squared_error = 0.0;
  for (std::size_t index = 0; index < image_points.size(); ++index) {
    if (!finite(projected[index])) return evaluation;
    evaluation.projected_points[index] = projected[index];
    const double dx = static_cast<double>(projected[index].x) -
                      static_cast<double>(image_points[index].x);
    const double dy = static_cast<double>(projected[index].y) -
                      static_cast<double>(image_points[index].y);
    squared_error += dx * dx + dy * dy;
  }
  evaluation.squared_error_px = squared_error;
  return evaluation;
}

}  // namespace

const char* reliableYawStatusName(ReliableYawStatus status) noexcept {
  switch (status) {
    case ReliableYawStatus::Success:
      return "success";
    case ReliableYawStatus::InvalidInput:
      return "invalid yaw solver input";
    case ReliableYawStatus::ReprojectionTooLarge:
      return "constrained yaw reprojection error is too large";
    case ReliableYawStatus::WeakYawInformation:
      return "constrained yaw is poorly observable";
    case ReliableYawStatus::AmbiguousOrientation:
      return "constrained yaw is ambiguous";
    case ReliableYawStatus::BackFacingArmor:
      return "constrained yaw puts armor back-facing the camera";
  }
  return "unknown reliable yaw status";
}

ConstrainedYawSolver::ConstrainedYawSolver(CameraCalibration calibration,
                                           ConstrainedYawOptions options)
    : calibration_(std::move(calibration)), options_(options) {
  if (calibration_.image_size.width <= 0 || calibration_.image_size.height <= 0 ||
      options_.coarse_search_steps < 16 ||
      !finite(options_.max_reprojection_rms_px) ||
      !finite(options_.max_yaw_std_rad) || !finite(options_.min_facing_cosine) ||
      !finite(options_.min_opposite_margin_px) ||
      options_.max_reprojection_rms_px <= 0.0 ||
      options_.max_yaw_std_rad <= 0.0 || options_.min_facing_cosine < -1.0 ||
      options_.min_facing_cosine > 1.0 || options_.min_opposite_margin_px < 0.0) {
    throw std::invalid_argument("invalid constrained yaw solver configuration");
  }
}

ReliableYaw ConstrainedYawSolver::solve(
    const coordinates::CoordinateSnapshot& exposure_snapshot,
    const PoseResult& pose,
    const std::array<cv::Point2f, kArmorPointCount>& image_points) const {
  ReliableYaw result;
  if (!exposure_snapshot.valid || !pose.valid || !finite(pose.center_camera_m) ||
      pose.center_camera_m[2] <= 0.0 ||
      !std::all_of(image_points.begin(), image_points.end(),
                   [](const cv::Point2f& point) { return finite(point); })) {
    return result;
  }

  // 相机朝向取自图像曝光瞬间，不能取检测处理完成时刻的云台姿态。
  const cv::Matx33d rotation_tracker_from_camera =
      coordinates::cameraRotationOdom(exposure_snapshot);
  std::vector<cv::Point2f> distorted_points(image_points.begin(),
                                             image_points.end());
  std::vector<cv::Point2f> normalized_vector;
  cv::undistortPoints(distorted_points, normalized_vector,
                      calibration_.camera_matrix,
                      calibration_.distortion_coefficients);
  if (normalized_vector.size() != image_points.size()) return result;
  NormalizedImagePoints normalized_points{};
  for (std::size_t index = 0; index < normalized_points.size(); ++index) {
    if (!finite(normalized_vector[index])) return result;
    normalized_points[index] = normalized_vector[index];
  }
  double best_yaw = 0.0;
  ProjectionEvaluation best;
  const double step = 2.0 * kPi / options_.coarse_search_steps;
  // 先全局扫描，避免从 IPPE 姿态或局部初值继承平面解的错误分支。
  for (int index = 0; index < options_.coarse_search_steps; ++index) {
    const double yaw = -kPi + step * index;
    const ProjectionEvaluation candidate = evaluateYaw(
        yaw, rotation_tracker_from_camera, calibration_, pose.armor_size,
        normalized_points, image_points);
    if (facingCosine(candidate) < options_.min_facing_cosine) continue;
    if (candidate.squared_error_px < best.squared_error_px) {
      best_yaw = yaw;
      best = candidate;
    }
  }
  if (!finite(best.squared_error_px)) return result;

  // 在最佳网格附近做有界黄金分割，保持 yaw 求解独立于 IPPE。
  double lower = best_yaw - step;
  double upper = best_yaw + step;
  constexpr double kGolden = 0.6180339887498948482;
  double left = upper - kGolden * (upper - lower);
  double right = lower + kGolden * (upper - lower);
  double left_cost = evaluateYaw(left, rotation_tracker_from_camera, calibration_,
                                 pose.armor_size, normalized_points,
                                 image_points)
                         .squared_error_px;
  double right_cost = evaluateYaw(right, rotation_tracker_from_camera, calibration_,
                                  pose.armor_size, normalized_points,
                                  image_points)
                          .squared_error_px;
  for (int iteration = 0; iteration < 30; ++iteration) {
    if (left_cost < right_cost) {
      upper = right;
      right = left;
      right_cost = left_cost;
      left = upper - kGolden * (upper - lower);
      left_cost = evaluateYaw(left, rotation_tracker_from_camera, calibration_,
                              pose.armor_size, normalized_points,
                              image_points)
                      .squared_error_px;
    } else {
      lower = left;
      left = right;
      left_cost = right_cost;
      right = lower + kGolden * (upper - lower);
      right_cost = evaluateYaw(right, rotation_tracker_from_camera, calibration_,
                               pose.armor_size, normalized_points,
                               image_points)
                       .squared_error_px;
    }
  }
  best_yaw = 0.5 * (lower + upper);
  best = evaluateYaw(best_yaw, rotation_tracker_from_camera, calibration_,
                     pose.armor_size, normalized_points, image_points);
  if (!finite(best.center_camera_m) || best.center_camera_m[2] <= 0.0) {
    return result;
  }
  result.inward_yaw_T_rad = wrapToPi(best_yaw);
  result.refined_center_camera_m = best.center_camera_m;
  result.reprojection_rms_px =
      std::sqrt(best.squared_error_px / (2.0 * image_points.size()));
  if (!finite(result.reprojection_rms_px) ||
      result.reprojection_rms_px > options_.max_reprojection_rms_px) {
    result.status = ReliableYawStatus::ReprojectionTooLarge;
    return result;
  }

  // 通过误差曲率估计单板 yaw 可观测性；曲率过小意味着 yaw 不可信。
  const double finite_difference_rad = 0.005;
  const double left_error = evaluateYaw(
      best_yaw - finite_difference_rad, rotation_tracker_from_camera,
      calibration_, pose.armor_size, normalized_points, image_points)
                                .squared_error_px;
  const double right_error = evaluateYaw(
      best_yaw + finite_difference_rad, rotation_tracker_from_camera,
      calibration_, pose.armor_size, normalized_points, image_points)
                                 .squared_error_px;
  const double information = (left_error - 2.0 * best.squared_error_px +
                              right_error) /
                             (2.0 * finite_difference_rad * finite_difference_rad);
  if (!finite(information) || information <= 1e-9) {
    result.status = ReliableYawStatus::WeakYawInformation;
    return result;
  }
  result.yaw_std_rad = std::sqrt(1.0 / information);
  if (!finite(result.yaw_std_rad) || result.yaw_std_rad > options_.max_yaw_std_rad) {
    result.status = ReliableYawStatus::WeakYawInformation;
    return result;
  }

  // 与相差 pi 的反向法线比较，拒绝两种朝向几乎等价的平面退化情况。
  const ProjectionEvaluation opposite = evaluateYaw(
      best_yaw + kPi, rotation_tracker_from_camera, calibration_, pose.armor_size,
      normalized_points, image_points);
  const double opposite_rms = std::sqrt(
      opposite.squared_error_px / (2.0 * image_points.size()));
  const double opposite_facing_cosine = facingCosine(opposite);
  if (finite(opposite_rms) && finite(opposite_facing_cosine) &&
      opposite_facing_cosine >= options_.min_facing_cosine &&
      opposite_rms - result.reprojection_rms_px <
          options_.min_opposite_margin_px) {
    result.status = ReliableYawStatus::AmbiguousOrientation;
    return result;
  }

  result.facing_cosine = facingCosine(best);
  if (!finite(result.facing_cosine) ||
      result.facing_cosine < options_.min_facing_cosine) {
    result.status = ReliableYawStatus::BackFacingArmor;
    return result;
  }

  result.valid = true;
  result.status = ReliableYawStatus::Success;
  return result;
}

}  // namespace yolo_detect::tracking
