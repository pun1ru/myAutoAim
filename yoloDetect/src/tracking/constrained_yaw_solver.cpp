#include "tracking/constrained_yaw_solver.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Core>
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
  std::array<cv::Point2d, kArmorPointCount> projected_points{};
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

using Residual8 = Eigen::Matrix<double, 8, 1>;
using Jacobian8x4 = Eigen::Matrix<double, 8, 4>;
using Matrix4 = Eigen::Matrix<double, 4, 4>;
using Vector4 = Eigen::Matrix<double, 4, 1>;

Residual8 residuals(const ProjectionEvaluation& evaluation,
                    const std::array<cv::Point2f, kArmorPointCount>& points) {
  Residual8 result;
  for (std::size_t index = 0; index < points.size(); ++index) {
    result[2 * index] = evaluation.projected_points[index].x - points[index].x;
    result[2 * index + 1] =
        evaluation.projected_points[index].y - points[index].y;
  }
  return result;
}

bool finiteEigen(const Vector4& value) {
  return value.array().isFinite().all();
}

Jacobian8x4 numericalJacobian(
    double yaw, const cv::Vec3d& center_camera_m,
    const ProjectionEvaluation& evaluation,
    const cv::Matx33d& rotation_tracker_from_camera,
    const CameraCalibration& calibration, ArmorSize armor_size,
    const std::array<cv::Point2f, kArmorPointCount>& image_points) {
  Jacobian8x4 jacobian;
  constexpr double kPositionStepM = 1e-4;
  constexpr double kYawStepRad = 1e-4;
  for (int column = 0; column < 4; ++column) {
    double perturbed_yaw = yaw;
    cv::Vec3d perturbed_center = center_camera_m;
    double step = kPositionStepM;
    if (column == 3) {
      perturbed_yaw += kYawStepRad;
      step = kYawStepRad;
    } else {
      perturbed_center[column] += kPositionStepM;
    }
    const ProjectionEvaluation perturbed = evaluateYaw(
        perturbed_yaw, rotation_tracker_from_camera, calibration, armor_size,
        perturbed_center, image_points);
    jacobian.col(column) =
        (residuals(perturbed, image_points) - residuals(evaluation, image_points)) /
        step;
  }
  return jacobian;
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
    case ReliableYawStatus::CenterAdjustmentTooLarge:
      return "constrained yaw requires an excessive center adjustment";
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
      !finite(options_.max_center_adjustment_m) ||
      options_.max_reprojection_rms_px <= 0.0 ||
      options_.max_yaw_std_rad <= 0.0 || options_.min_facing_cosine < -1.0 ||
      options_.min_facing_cosine > 1.0 || options_.min_opposite_margin_px < 0.0 ||
      options_.max_center_adjustment_m <= 0.0) {
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
  cv::Vec3d refined_center_camera_m = pose.center_camera_m;
  double damping = 1e-3;
  for (int iteration = 0; iteration < 16; ++iteration) {
    best = evaluateYaw(best_yaw, rotation_tracker_from_camera, calibration_,
                       pose.armor_size, refined_center_camera_m, image_points);
    if (!finite(best.squared_error_px)) return result;
    const Residual8 residual = residuals(best, image_points);
    const Jacobian8x4 jacobian = numericalJacobian(
        best_yaw, refined_center_camera_m, best, rotation_tracker_from_camera,
        calibration_, pose.armor_size, image_points);
    const Matrix4 normal = jacobian.transpose() * jacobian;
    const Vector4 gradient = jacobian.transpose() * residual;
    Eigen::LDLT<Matrix4> factor(normal + damping * Matrix4::Identity());
    if (factor.info() != Eigen::Success) break;
    Vector4 delta = factor.solve(-gradient);
    if (factor.info() != Eigen::Success || !finiteEigen(delta)) break;

    const double position_step = delta.template head<3>().norm();
    if (position_step > 0.05) delta.template head<3>() *= 0.05 / position_step;
    delta[3] = std::clamp(delta[3], -0.15, 0.15);
    const cv::Vec3d candidate_center = refined_center_camera_m +
        cv::Vec3d(delta[0], delta[1], delta[2]);
    if (!finite(candidate_center) || candidate_center[2] <= 0.0) break;
    const ProjectionEvaluation candidate = evaluateYaw(
        best_yaw + delta[3], rotation_tracker_from_camera, calibration_,
        pose.armor_size, candidate_center, image_points);
    if (candidate.squared_error_px < best.squared_error_px) {
      best_yaw += delta[3];
      refined_center_camera_m = candidate_center;
      best = candidate;
      damping = std::max(1e-7, damping * 0.3);
      if (delta.norm() < 1e-6) break;
    } else {
      damping = std::min(1e8, damping * 10.0);
    }
  }
  best = evaluateYaw(best_yaw, rotation_tracker_from_camera, calibration_,
                     pose.armor_size, refined_center_camera_m, image_points);
  if (!finite(best.squared_error_px)) return result;
  result.inward_yaw_T_rad = wrapToPi(best_yaw);
  result.reprojection_rms_px =
      std::sqrt(best.squared_error_px / (2.0 * image_points.size()));
  if (cv::norm(refined_center_camera_m - pose.center_camera_m) >
      options_.max_center_adjustment_m) {
    result.status = ReliableYawStatus::CenterAdjustmentTooLarge;
    return result;
  }
  if (!finite(result.reprojection_rms_px) ||
      result.reprojection_rms_px > options_.max_reprojection_rms_px) {
    result.status = ReliableYawStatus::ReprojectionTooLarge;
    return result;
  }

  const Jacobian8x4 final_jacobian = numericalJacobian(
      best_yaw, refined_center_camera_m, best, rotation_tracker_from_camera,
      calibration_, pose.armor_size, image_points);
  Eigen::LDLT<Matrix4> final_factor(
      final_jacobian.transpose() * final_jacobian);
  Vector4 yaw_basis = Vector4::Zero();
  yaw_basis[3] = 1.0;
  const Vector4 yaw_covariance_column = final_factor.solve(yaw_basis);
  if (final_factor.info() != Eigen::Success ||
      !finiteEigen(yaw_covariance_column) || yaw_covariance_column[3] <= 0.0) {
    result.status = ReliableYawStatus::WeakYawInformation;
    return result;
  }
  result.yaw_std_rad = std::sqrt(yaw_covariance_column[3]);
  if (!finite(result.yaw_std_rad) || result.yaw_std_rad > options_.max_yaw_std_rad) {
    result.status = ReliableYawStatus::WeakYawInformation;
    return result;
  }

  const ProjectionEvaluation opposite = evaluateYaw(
      best_yaw + kPi, rotation_tracker_from_camera, calibration_, pose.armor_size,
      refined_center_camera_m, image_points);
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
  const double range = cv::norm(refined_center_camera_m);
  result.facing_cosine =
      outward_normal_camera.dot(-refined_center_camera_m) / range;
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
