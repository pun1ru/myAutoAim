#include "tracking/constrained_yaw_solver.hpp"

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

cv::Matx33d rotationTrackerFromArmor(double inward_yaw_T_rad,
                                     double pitch_rad) {
  const double c = std::cos(inward_yaw_T_rad);
  const double s = std::sin(inward_yaw_T_rad);
  const double cp = std::cos(pitch_rad);
  const double sp = std::sin(pitch_rad);
  // Match sp_vision_25::reproject_armor: +x_A points inward, +y_A is
  // panel-left, and +z_A is up. Positive pitch tilts +x_A downward.
  const cv::Vec3d x_in_T(c * cp, s * cp, -sp);
  const cv::Vec3d y_left_T(-s, c, 0.0);
  const cv::Vec3d z_up_T(c * sp, s * sp, cp);
  return {
      x_in_T[0], y_left_T[0], z_up_T[0],
      x_in_T[1], y_left_T[1], z_up_T[1],
      x_in_T[2], y_left_T[2], z_up_T[2],
  };
}

struct ProjectionEvaluation {
  double squared_error_px = std::numeric_limits<double>::infinity();
  cv::Matx33d rotation_camera_from_armor = cv::Matx33d::eye();
  std::array<cv::Point2d, kArmorPointCount> projected_points{};
};

ProjectionEvaluation evaluateYaw(
    double inward_yaw_T_rad, const cv::Matx33d& rotation_tracker_from_camera,
    const CameraCalibration& calibration, ArmorSize armor_size,
    const cv::Vec3d& center_camera_m,
    const std::array<cv::Point2f, kArmorPointCount>& image_points,
    double pitch_rad) {
  ProjectionEvaluation evaluation;
  // 候选 yaw 先构成 T-from-A，再用同曝光 R_TC 转到 OpenCV 相机系投影。
  const cv::Matx33d rotation_camera_from_tracker =
      rotation_tracker_from_camera.t();
  evaluation.rotation_camera_from_armor =
      rotation_camera_from_tracker *
      rotationTrackerFromArmor(inward_yaw_T_rad, pitch_rad);
  cv::Mat rvec;
  cv::Rodrigues(evaluation.rotation_camera_from_armor, rvec);
  const auto object_point_array = ArmorPoseEstimator::objectPoints(armor_size);
  const std::vector<cv::Point3d> object_points(object_point_array.begin(),
                                                object_point_array.end());
  std::vector<cv::Point2d> projected;
  cv::projectPoints(object_points, rvec, center_camera_m,
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

double facingCosine(const ProjectionEvaluation& evaluation,
                    const cv::Vec3d& center_camera_m) {
  if (!finite(center_camera_m)) {
    return -std::numeric_limits<double>::infinity();
  }
  const double range = cv::norm(center_camera_m);
  if (!finite(range) || range <= 1e-9) {
    return -std::numeric_limits<double>::infinity();
  }
  const cv::Vec3d outward_normal_camera(
      -evaluation.rotation_camera_from_armor(0, 0),
      -evaluation.rotation_camera_from_armor(1, 0),
      -evaluation.rotation_camera_from_armor(2, 0));
  return outward_normal_camera.dot(-center_camera_m) / range;
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
      !finite(options_.search_half_range_rad) ||
      !finite(options_.search_step_rad) ||
      !finite(options_.armor_pitch_rad) ||
      !finite(options_.outpost_pitch_rad) ||
      options_.search_half_range_rad <= 0.0 ||
      options_.search_step_rad <= 0.0 ||
      !finite(options_.max_reprojection_rms_px) ||
      !finite(options_.max_yaw_std_rad) || !finite(options_.min_facing_cosine) ||
      !finite(options_.min_opposite_margin_px) ||
      options_.max_reprojection_rms_px <= 0.0 ||
      options_.max_yaw_std_rad <= 0.0 || options_.min_facing_cosine < -1.0 ||
      options_.min_facing_cosine > 1.0 || options_.min_opposite_margin_px < 0.0) {
    throw std::invalid_argument("invalid constrained yaw solver configuration");
  }
}

bool ConstrainedYawSolver::setOptions(ConstrainedYawOptions options) noexcept {
  try {
    ConstrainedYawSolver validated(calibration_, options);
    options_ = validated.options_;
    return true;
  } catch (...) {
    return false;
  }
}

ReliableYaw ConstrainedYawSolver::solve(
    const coordinates::CoordinateSnapshot& exposure_snapshot,
    const PoseResult& pose,
    const std::array<cv::Point2f, kArmorPointCount>& image_points,
    int number_id) const {
  ReliableYaw result;
  if (!exposure_snapshot.valid || !pose.valid || !finite(pose.center_camera_m) ||
      pose.center_camera_m[2] <= 0.0 ||
      !std::all_of(image_points.begin(), image_points.end(),
                   [](const cv::Point2f& point) { return finite(point); })) {
    return result;
  }

  // 相机朝向取自图像曝光瞬间，不能取检测处理完成时刻的云台姿态。
  // sp_vision_25 skips fixed-pitch yaw optimization for balance armor because
  // the pitch prior does not hold. Keep those observations position-only.
  if (options_.skip_balance_large_armor &&
      pose.armor_size == ArmorSize::Large && number_id >= 3 && number_id <= 5) {
    result.status = ReliableYawStatus::WeakYawInformation;
    return result;
  }

  const cv::Matx33d rotation_tracker_from_camera =
      coordinates::cameraRotationOdom(exposure_snapshot);
  // Keep yaw independent from the PnP Rodrigues yaw, but use the PnP plate
  // normal only as a pitch constraint. Pose +x_A is the outward normal;
  // rotationTrackerFromArmor expects the inward normal.
  double pitch_rad = number_id == options_.outpost_number_id
                         ? options_.outpost_pitch_rad
                         : options_.armor_pitch_rad;
  const cv::Matx33d& rotation_camera_from_armor =
      pose.rotation_camera_from_armor;
  const cv::Vec3d outward_camera(rotation_camera_from_armor(0, 0),
                                 rotation_camera_from_armor(1, 0),
                                 rotation_camera_from_armor(2, 0));
  const cv::Vec3d inward_tracker =
      -(rotation_tracker_from_camera * outward_camera);
  if (finite(inward_tracker) && cv::norm(inward_tracker) > 1e-9) {
    pitch_rad = std::asin(std::clamp(-inward_tracker[2] /
                                         cv::norm(inward_tracker),
                                     -1.0, 1.0));
  }
  double best_yaw = 0.0;
  ProjectionEvaluation best;
  const double step = options_.search_step_rad;
  const cv::Matx33d rotation_tracker_from_gimbal =
      rotation_tracker_from_camera * coordinates::rotationGimbalFromCamera().t();
  const double gimbal_yaw_rad =
      std::atan2(rotation_tracker_from_gimbal(1, 0),
                 rotation_tracker_from_gimbal(0, 0));
  const int candidate_count = std::max(
      1, static_cast<int>(std::ceil(2.0 * options_.search_half_range_rad / step)));
  // 先全局扫描，避免从 IPPE 姿态或局部初值继承平面解的错误分支。
  for (int index = 0; index < candidate_count; ++index) {
    const double yaw = gimbal_yaw_rad -
                       options_.search_half_range_rad + index * step;
    const ProjectionEvaluation candidate = evaluateYaw(
        yaw, rotation_tracker_from_camera, calibration_, pose.armor_size,
        pose.center_camera_m, image_points, pitch_rad);
    if (facingCosine(candidate, pose.center_camera_m) <
        options_.min_facing_cosine) {
      continue;
    }
    if (candidate.squared_error_px < best.squared_error_px) {
      best_yaw = yaw;
      best = candidate;
    }
  }
  if (!finite(best.squared_error_px)) return result;

  // 在最佳网格附近做有界黄金分割，保持 yaw 求解独立于 IPPE。
  // The grid is the sp_vision_25 search. Refine only inside the winning
  // one-degree cell so the result remains smooth without changing branches.
  double lower = best_yaw - step;
  double upper = best_yaw + step;
  constexpr double kGolden = 0.6180339887498948482;
  double left = upper - kGolden * (upper - lower);
  double right = lower + kGolden * (upper - lower);
  auto yawCost = [&](double yaw) {
    return evaluateYaw(yaw, rotation_tracker_from_camera, calibration_,
                       pose.armor_size, pose.center_camera_m, image_points,
                       pitch_rad)
        .squared_error_px;
  };
  double left_cost = yawCost(left);
  double right_cost = yawCost(right);
  for (int iteration = 0; iteration < 24; ++iteration) {
    if (left_cost < right_cost) {
      upper = right;
      right = left;
      right_cost = left_cost;
      left = upper - kGolden * (upper - lower);
      left_cost = yawCost(left);
    } else {
      lower = left;
      left = right;
      left_cost = right_cost;
      right = lower + kGolden * (upper - lower);
      right_cost = yawCost(right);
    }
  }
  best_yaw = 0.5 * (lower + upper);
  best = evaluateYaw(best_yaw, rotation_tracker_from_camera, calibration_,
                     pose.armor_size, pose.center_camera_m, image_points,
                     pitch_rad);
  result.inward_yaw_T_rad = wrapToPi(best_yaw);
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
      calibration_, pose.armor_size, pose.center_camera_m, image_points, pitch_rad)
                                .squared_error_px;
  const double right_error = evaluateYaw(
      best_yaw + finite_difference_rad, rotation_tracker_from_camera,
      calibration_, pose.armor_size, pose.center_camera_m, image_points, pitch_rad)
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

  result.facing_cosine = facingCosine(best, pose.center_camera_m);
  if (!finite(result.facing_cosine) ||
      result.facing_cosine < options_.min_facing_cosine) {
    result.status = ReliableYawStatus::BackFacingArmor;
    return result;
  }

  // Only a second front-facing solution is a physical ambiguity. The
  // algebraically equivalent back-facing planar solution is not observable.
  const ProjectionEvaluation opposite = evaluateYaw(
      best_yaw + kPi, rotation_tracker_from_camera, calibration_, pose.armor_size,
      pose.center_camera_m, image_points, pitch_rad);
  const double opposite_rms = std::sqrt(
      opposite.squared_error_px / (2.0 * image_points.size()));
  const double opposite_facing_cosine =
      facingCosine(opposite, pose.center_camera_m);
  if (opposite_facing_cosine >= options_.min_facing_cosine &&
      (!finite(opposite_rms) ||
       opposite_rms - result.reprojection_rms_px <
           options_.min_opposite_margin_px)) {
    result.status = ReliableYawStatus::AmbiguousOrientation;
    return result;
  }

  result.valid = true;
  result.status = ReliableYawStatus::Success;
  return result;
}

}  // namespace yolo_detect::tracking
