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

struct ProjectionEvaluation {
  double squared_error_px = std::numeric_limits<double>::infinity();
  cv::Matx33d rotation_camera_from_armor = cv::Matx33d::eye();
};

ProjectionEvaluation evaluateYaw(
    double inward_yaw_T_rad, const cv::Matx33d& rotation_tracker_from_camera,
    const CameraCalibration& calibration, ArmorSize armor_size,
    const cv::Vec3d& center_camera_m,
    const std::array<cv::Point2f, kArmorPointCount>& image_points) {
  ProjectionEvaluation evaluation;
  const cv::Matx33d rotation_camera_from_tracker =
      rotation_tracker_from_camera.t();
  evaluation.rotation_camera_from_armor =
      rotation_camera_from_tracker *
      rotationTrackerFromArmor(inward_yaw_T_rad);

  cv::Vec3d rvec;
  cv::Rodrigues(evaluation.rotation_camera_from_armor, rvec);
  std::vector<cv::Point2f> projected;
  cv::projectPoints(ArmorPoseEstimator::objectPoints(armor_size), rvec,
                    center_camera_m, calibration.camera_matrix,
                    calibration.distortion_coefficients, projected);
  if (projected.size() != image_points.size()) return evaluation;

  double squared_error = 0.0;
  for (std::size_t index = 0; index < image_points.size(); ++index) {
    if (!finite(projected[index])) return evaluation;
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

  const cv::Matx33d rotation_tracker_from_camera =
      coordinates::cameraRotationOdom(exposure_snapshot);
  double best_yaw = 0.0;
  ProjectionEvaluation best;
  const double step = 2.0 * kPi / options_.coarse_search_steps;
  for (int index = 0; index < options_.coarse_search_steps; ++index) {
    const double yaw = -kPi + step * index;
    const ProjectionEvaluation candidate = evaluateYaw(
        yaw, rotation_tracker_from_camera, calibration_, pose.armor_size,
        pose.center_camera_m, image_points);
    if (candidate.squared_error_px < best.squared_error_px) {
      best_yaw = yaw;
      best = candidate;
    }
  }
  if (!finite(best.squared_error_px)) return result;

  // A bounded golden-section search keeps the yaw result independent of IPPE.
  double lower = best_yaw - step;
  double upper = best_yaw + step;
  constexpr double kGolden = 0.6180339887498948482;
  double left = upper - kGolden * (upper - lower);
  double right = lower + kGolden * (upper - lower);
  double left_cost = evaluateYaw(left, rotation_tracker_from_camera, calibration_,
                                 pose.armor_size, pose.center_camera_m,
                                 image_points)
                         .squared_error_px;
  double right_cost = evaluateYaw(right, rotation_tracker_from_camera, calibration_,
                                  pose.armor_size, pose.center_camera_m,
                                  image_points)
                          .squared_error_px;
  for (int iteration = 0; iteration < 30; ++iteration) {
    if (left_cost < right_cost) {
      upper = right;
      right = left;
      right_cost = left_cost;
      left = upper - kGolden * (upper - lower);
      left_cost = evaluateYaw(left, rotation_tracker_from_camera, calibration_,
                              pose.armor_size, pose.center_camera_m,
                              image_points)
                      .squared_error_px;
    } else {
      lower = left;
      left = right;
      left_cost = right_cost;
      right = lower + kGolden * (upper - lower);
      right_cost = evaluateYaw(right, rotation_tracker_from_camera, calibration_,
                               pose.armor_size, pose.center_camera_m,
                               image_points)
                       .squared_error_px;
    }
  }
  best_yaw = 0.5 * (lower + upper);
  best = evaluateYaw(best_yaw, rotation_tracker_from_camera, calibration_,
                     pose.armor_size, pose.center_camera_m, image_points);
  result.inward_yaw_T_rad = wrapToPi(best_yaw);
  result.reprojection_rms_px =
      std::sqrt(best.squared_error_px / (2.0 * image_points.size()));
  if (!finite(result.reprojection_rms_px) ||
      result.reprojection_rms_px > options_.max_reprojection_rms_px) {
    result.status = ReliableYawStatus::ReprojectionTooLarge;
    return result;
  }

  const double finite_difference_rad = 0.005;
  const double left_error = evaluateYaw(
      best_yaw - finite_difference_rad, rotation_tracker_from_camera,
      calibration_, pose.armor_size, pose.center_camera_m, image_points)
                                .squared_error_px;
  const double right_error = evaluateYaw(
      best_yaw + finite_difference_rad, rotation_tracker_from_camera,
      calibration_, pose.armor_size, pose.center_camera_m, image_points)
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

  const ProjectionEvaluation opposite = evaluateYaw(
      best_yaw + kPi, rotation_tracker_from_camera, calibration_, pose.armor_size,
      pose.center_camera_m, image_points);
  const double opposite_rms = std::sqrt(
      opposite.squared_error_px / (2.0 * image_points.size()));
  if (!finite(opposite_rms) ||
      opposite_rms - result.reprojection_rms_px < options_.min_opposite_margin_px) {
    result.status = ReliableYawStatus::AmbiguousOrientation;
    return result;
  }

  const cv::Vec3d outward_normal_camera(
      best.rotation_camera_from_armor(0, 0),
      best.rotation_camera_from_armor(1, 0),
      best.rotation_camera_from_armor(2, 0));
  const double range = cv::norm(pose.center_camera_m);
  result.facing_cosine = outward_normal_camera.dot(-pose.center_camera_m) / range;
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
