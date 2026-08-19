#include "pose/armor_pose_estimator.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace yolo_detect {
namespace {

constexpr double kSmallArmorWidthM = 0.135;
constexpr double kLargeArmorWidthM = 0.225;
constexpr double kArmorHeightM = 0.055;
constexpr double kMinQuadrilateralAreaPx = 4.0;
constexpr double kGeometryEpsilon = 1e-6;

bool isFinite(double value) { return std::isfinite(value); }

bool isFinite(const cv::Vec3d& value) {
  return isFinite(value[0]) && isFinite(value[1]) && isFinite(value[2]);
}

bool isFinite(const cv::Matx33d& value) {
  return std::all_of(value.val, value.val + 9,
                     [](double element) { return isFinite(element); });
}

double signedArea(
    const std::array<cv::Point2f, kArmorPointCount>& points) {
  double twice_area = 0.0;
  for (std::size_t index = 0; index < points.size(); ++index) {
    const cv::Point2f& point = points[index];
    const cv::Point2f& next = points[(index + 1) % points.size()];
    twice_area += static_cast<double>(point.x) * next.y -
                  static_cast<double>(point.y) * next.x;
  }
  return twice_area * 0.5;
}

double orientation(const cv::Point2f& a, const cv::Point2f& b,
                   const cv::Point2f& c) {
  return (static_cast<double>(b.x) - a.x) *
             (static_cast<double>(c.y) - a.y) -
         (static_cast<double>(b.y) - a.y) *
             (static_cast<double>(c.x) - a.x);
}

bool onSegment(const cv::Point2f& a, const cv::Point2f& b,
               const cv::Point2f& point) {
  return point.x >= std::min(a.x, b.x) - kGeometryEpsilon &&
         point.x <= std::max(a.x, b.x) + kGeometryEpsilon &&
         point.y >= std::min(a.y, b.y) - kGeometryEpsilon &&
         point.y <= std::max(a.y, b.y) + kGeometryEpsilon;
}

bool segmentsIntersect(const cv::Point2f& a, const cv::Point2f& b,
                       const cv::Point2f& c, const cv::Point2f& d) {
  const double o1 = orientation(a, b, c);
  const double o2 = orientation(a, b, d);
  const double o3 = orientation(c, d, a);
  const double o4 = orientation(c, d, b);
  const bool first_straddles =
      (o1 > kGeometryEpsilon && o2 < -kGeometryEpsilon) ||
      (o1 < -kGeometryEpsilon && o2 > kGeometryEpsilon);
  const bool second_straddles =
      (o3 > kGeometryEpsilon && o4 < -kGeometryEpsilon) ||
      (o3 < -kGeometryEpsilon && o4 > kGeometryEpsilon);
  if (first_straddles && second_straddles) return true;
  if (std::abs(o1) <= kGeometryEpsilon && onSegment(a, b, c)) return true;
  if (std::abs(o2) <= kGeometryEpsilon && onSegment(a, b, d)) return true;
  if (std::abs(o3) <= kGeometryEpsilon && onSegment(c, d, a)) return true;
  if (std::abs(o4) <= kGeometryEpsilon && onSegment(c, d, b)) return true;
  return false;
}

bool selfIntersects(
    const std::array<cv::Point2f, kArmorPointCount>& points) {
  return segmentsIntersect(points[0], points[1], points[2], points[3]) ||
         segmentsIntersect(points[1], points[2], points[3], points[0]);
}

bool matToVector(const cv::Mat& source, cv::Vec3d& destination) {
  if (source.total() != 3) return false;
  cv::Mat converted;
  source.reshape(1, 1).convertTo(converted, CV_64F);
  destination = {converted.at<double>(0, 0), converted.at<double>(0, 1),
                 converted.at<double>(0, 2)};
  return isFinite(destination);
}

cv::Matx33d matToMatrix(const cv::Mat& source) {
  cv::Mat converted;
  source.convertTo(converted, CV_64F);
  cv::Matx33d result;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      result(row, column) = converted.at<double>(row, column);
    }
  }
  return result;
}

bool allCornersInFront(
    const std::array<cv::Point3d, kArmorPointCount>& object_points,
    const cv::Matx33d& rotation, const cv::Vec3d& translation) {
  for (const cv::Point3d& point : object_points) {
    const cv::Vec3d camera_point =
        rotation * cv::Vec3d(point.x, point.y, point.z) + translation;
    if (!isFinite(camera_point) || camera_point[2] <= 0.0) return false;
  }
  return true;
}

void validateCalibration(const CameraCalibration& calibration) {
  if (calibration.image_size.width <= 0 ||
      calibration.image_size.height <= 0) {
    throw std::invalid_argument("camera calibration image size must be positive");
  }
  if (!isFinite(calibration.camera_matrix)) {
    throw std::invalid_argument("camera matrix K contains a non-finite value");
  }
  if (calibration.camera_matrix(0, 0) <= 0.0 ||
      calibration.camera_matrix(1, 1) <= 0.0) {
    throw std::invalid_argument("camera focal lengths fx and fy must be positive");
  }
  if (std::abs(calibration.camera_matrix(2, 0)) > kGeometryEpsilon ||
      std::abs(calibration.camera_matrix(2, 1)) > kGeometryEpsilon ||
      std::abs(calibration.camera_matrix(2, 2) - 1.0) > kGeometryEpsilon) {
    throw std::invalid_argument(
        "camera matrix K must have homogeneous row [0, 0, 1]");
  }
  const std::size_t count = calibration.distortion_coefficients.size();
  if (count != 0 && count != 4 && count != 5 && count != 8 && count != 12 &&
      count != 14) {
    throw std::invalid_argument(
        "distortion D must contain 4, 5, 8, 12, or 14 coefficients");
  }
  if (!std::all_of(calibration.distortion_coefficients.begin(),
                   calibration.distortion_coefficients.end(),
                   [](double value) { return isFinite(value); })) {
    throw std::invalid_argument("distortion D contains a non-finite value");
  }
}

}  // namespace

const char* armorSizeName(ArmorSize armor_size) noexcept {
  switch (armor_size) {
    case ArmorSize::Small:
      return "small";
    case ArmorSize::Large:
      return "large";
  }
  return "unknown";
}

const char* poseStatusName(PoseStatus status) noexcept {
  switch (status) {
    case PoseStatus::Success:
      return "success";
    case PoseStatus::InsufficientKeypointConfidence:
      return "keypoint confidence is below the PnP threshold";
    case PoseStatus::ImageSizeMismatch:
      return "camera calibration does not match the image size";
    case PoseStatus::NonFiniteImagePoint:
      return "image point contains a non-finite coordinate";
    case PoseStatus::ImagePointOutsideFrame:
      return "image point is outside the calibrated frame";
    case PoseStatus::DegenerateQuadrilateral:
      return "image quadrilateral is degenerate";
    case PoseStatus::SelfIntersectingQuadrilateral:
      return "image quadrilateral is self-intersecting";
    case PoseStatus::InvalidPointOrder:
      return "image points are not ordered BL, TL, TR, BR";
    case PoseStatus::SolverFailure:
      return "IPPE failed to return pose candidates";
    case PoseStatus::NoValidCandidate:
      return "IPPE returned no finite positive-depth candidate";
  }
  return "unknown pose status";
}

ArmorPoseEstimator::ArmorPoseEstimator(CameraCalibration calibration)
    : calibration_(std::move(calibration)) {
  validateCalibration(calibration_);
}

PoseResult ArmorPoseEstimator::estimate(
    const std::array<cv::Point2f, kArmorPointCount>& image_points,
    cv::Size image_size, ArmorSize armor_size) const {
  PoseResult result;
  result.armor_size = armor_size;
  if (image_size != calibration_.image_size) {
    result.status = PoseStatus::ImageSizeMismatch;
    return result;
  }
  for (const cv::Point2f& point : image_points) {
    if (!isFinite(point.x) || !isFinite(point.y)) {
      result.status = PoseStatus::NonFiniteImagePoint;
      return result;
    }
    if (point.x < 0.0F || point.y < 0.0F || point.x >= image_size.width ||
        point.y >= image_size.height) {
      result.status = PoseStatus::ImagePointOutsideFrame;
      return result;
    }
  }
  if (selfIntersects(image_points)) {
    result.status = PoseStatus::SelfIntersectingQuadrilateral;
    return result;
  }
  const double area = signedArea(image_points);
  if (std::abs(area) < kMinQuadrilateralAreaPx) {
    result.status = PoseStatus::DegenerateQuadrilateral;
    return result;
  }
  if (area < 0.0) {
    result.status = PoseStatus::InvalidPointOrder;
    return result;
  }

  const auto object_point_array = objectPoints(armor_size);
  const std::vector<cv::Point3d> object_points(object_point_array.begin(),
                                               object_point_array.end());
  // OpenCV 4.5 IPPE requires its internal model plane to be z=0. The public
  // armor frame remains x_A=0; this proper rotation maps A to the temporary
  // IPPE frame P as p_P=(y_A,z_A,x_A).
  const cv::Matx33d rotation_ippe_from_armor(
      0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0);
  std::vector<cv::Point3d> ippe_object_points;
  ippe_object_points.reserve(object_point_array.size());
  for (const cv::Point3d& point : object_point_array) {
    const cv::Vec3d transformed =
        rotation_ippe_from_armor * cv::Vec3d(point.x, point.y, point.z);
    ippe_object_points.emplace_back(transformed[0], transformed[1],
                                    transformed[2]);
  }
  const std::vector<cv::Point2f> detected_points(image_points.begin(),
                                                 image_points.end());
  const cv::Mat distortion(calibration_.distortion_coefficients, true);
  std::vector<cv::Mat> rvecs;
  std::vector<cv::Mat> tvecs;
  try {
    cv::solvePnPGeneric(ippe_object_points, detected_points,
                        calibration_.camera_matrix, distortion, rvecs, tvecs,
                        false, cv::SOLVEPNP_IPPE);
  } catch (const cv::Exception&) {
    result.status = PoseStatus::SolverFailure;
    return result;
  }
  result.candidate_count = std::min(rvecs.size(), tvecs.size());
  if (result.candidate_count == 0) {
    result.status = PoseStatus::SolverFailure;
    return result;
  }

  double best_rms = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < result.candidate_count; ++index) {
    cv::Vec3d ippe_rvec;
    cv::Vec3d tvec;
    if (!matToVector(rvecs[index], ippe_rvec) ||
        !matToVector(tvecs[index], tvec) || tvec[2] <= 0.0) {
      continue;
    }

    cv::Mat ippe_rotation_mat;
    cv::Rodrigues(ippe_rvec, ippe_rotation_mat);
    const cv::Matx33d rotation =
        matToMatrix(ippe_rotation_mat) * rotation_ippe_from_armor;
    if (!isFinite(rotation) ||
        !allCornersInFront(object_point_array, rotation, tvec)) {
      continue;
    }
    cv::Mat armor_rvec_mat;
    cv::Rodrigues(rotation, armor_rvec_mat);
    cv::Vec3d armor_rvec;
    if (!matToVector(armor_rvec_mat, armor_rvec)) continue;

    std::vector<cv::Point2d> projected_points;
    cv::projectPoints(object_points, armor_rvec, tvec,
                      calibration_.camera_matrix,
                      distortion, projected_points);
    if (projected_points.size() != detected_points.size()) continue;
    double squared_error = 0.0;
    bool projection_valid = true;
    for (std::size_t point_index = 0;
         point_index < projected_points.size(); ++point_index) {
      const cv::Point2d& projected = projected_points[point_index];
      if (!isFinite(projected.x) || !isFinite(projected.y)) {
        projection_valid = false;
        break;
      }
      const double dx = projected.x - detected_points[point_index].x;
      const double dy = projected.y - detected_points[point_index].y;
      squared_error += dx * dx + dy * dy;
    }
    if (!projection_valid) continue;
    const double rms = std::sqrt(squared_error / detected_points.size());
    if (!isFinite(rms) || rms >= best_rms) continue;

    best_rms = rms;
    result.valid = true;
    result.status = PoseStatus::Success;
    result.rvec_rad = armor_rvec;
    result.rotation_camera_from_armor = rotation;
    result.tvec_m = tvec;
    result.center_camera_m = tvec;
    result.armor_normal_camera = rotation * cv::Vec3d(1.0, 0.0, 0.0);
    result.reprojection_rms_px = rms;
  }
  if (!result.valid) result.status = PoseStatus::NoValidCandidate;
  return result;
}

const CameraCalibration& ArmorPoseEstimator::calibration() const noexcept {
  return calibration_;
}

std::array<cv::Point3d, kArmorPointCount> ArmorPoseEstimator::objectPoints(
    ArmorSize armor_size) {
  const double width = armor_size == ArmorSize::Small ? kSmallArmorWidthM
                                                      : kLargeArmorWidthM;
  const double half_width = width * 0.5;
  const double half_height = kArmorHeightM * 0.5;
  return {
      cv::Point3d(0.0, +half_width, -half_height),  // bottom-left
      cv::Point3d(0.0, +half_width, +half_height),  // top-left
      cv::Point3d(0.0, -half_width, +half_height),  // top-right
      cv::Point3d(0.0, -half_width, -half_height),  // bottom-right
  };
}

}  // namespace yolo_detect
