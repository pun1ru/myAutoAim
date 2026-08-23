#include "tracking/tracker_measurement_adapter.hpp"

#include <algorithm>
#include <cmath>

namespace yolo_detect::tracking {
namespace {

bool finite(double value) { return std::isfinite(value); }

bool finite(const cv::Vec3d& value) {
  return finite(value[0]) && finite(value[1]) && finite(value[2]);
}

}  // namespace

std::optional<Measurement> makeTrackerMeasurement(
    const TrackerFrame& frame,
    const coordinates::CoordinateSnapshot& exposure_snapshot,
    const PoseResult& pose, const ArmorDetection& detection,
    const ConstrainedYawSolver& yaw_solver, ReliableYaw* reliable_yaw) {
  // This adapter is the only handoff from vision/coordinates to the EKF.
  // Keeping it separate prevents PnP, camera transforms, and tracker state
  // from silently sharing mutable state or using a processing-time pose.
  if (reliable_yaw != nullptr) *reliable_yaw = ReliableYaw{};
  // 量测、图像和坐标姿态必须是同一曝光，任意序列号/时间戳不一致均拒绝。
  if (frame.source_sequence == 0 || frame.capture_timestamp_ns == 0 ||
      !exposure_snapshot.valid ||
      exposure_snapshot.frame_sequence != frame.source_sequence ||
      exposure_snapshot.capture_timestamp_ns != frame.capture_timestamp_ns ||
      !pose.valid || !finite(pose.center_camera_m) ||
      !finite(pose.reprojection_rms_px) || !finite(detection.confidence)) {
    return std::nullopt;
  }
  Measurement measurement;
  measurement.timestamp_ns = frame.capture_timestamp_ns;
  // pose.center_camera_m is the PnP armor center in OpenCV C, already meters.
  // cameraPointToTracker applies the matching exposure-time C->T transform.
  // The tracker must never use a later gimbal pose for this image.
  // PnP tvec/center 已统一为米；相机到 T 的变换只使用 exposure_snapshot。
  const cv::Vec3d tracker_position = coordinates::cameraPointToTracker(
      exposure_snapshot, pose.center_camera_m);
  measurement.position_T_m = Eigen::Vector3d(
      tracker_position[0], tracker_position[1], tracker_position[2]);
  measurement.camera_range_m = cv::norm(pose.center_camera_m);
  if (!finite(measurement.camera_range_m) || measurement.camera_range_m <= 0.0) {
    return std::nullopt;
  }
  measurement.reprojection_rms_px = pose.reprojection_rms_px;
  const cv::Matx33d R_TC = coordinates::cameraRotationOdom(exposure_snapshot);
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      measurement.R_TC(row, column) = R_TC(row, column);
    }
  }
  measurement.camera_position_T_m = {
      exposure_snapshot.camera_position_odom_m[0],
      exposure_snapshot.camera_position_odom_m[1],
      exposure_snapshot.camera_position_odom_m[2]};
  measurement.has_exposure_camera_geometry =
      measurement.R_TC.allFinite() &&
      measurement.camera_position_T_m.allFinite();
  const cv::Matx33d& rotation_camera_from_armor =
      pose.rotation_camera_from_armor;
  // PoseResult::rotation_camera_from_armor.col(0) is the armor +x_A axis.
  // The constrained solver defines +x_A as the inward normal, so do not
  // negate it when deriving the diagnostic pitch.
  const cv::Vec3d inward_camera(rotation_camera_from_armor(0, 0),
                                rotation_camera_from_armor(1, 0),
                                rotation_camera_from_armor(2, 0));
  const cv::Vec3d inward_tracker = R_TC * inward_camera;
  const double inward_norm = cv::norm(inward_tracker);
  if (finite(inward_tracker) && finite(inward_norm) && inward_norm > 1e-9) {
    measurement.pnp_inward_pitch_T_rad = std::asin(std::clamp(
        -inward_tracker[2] / inward_norm, -1.0, 1.0));
    measurement.has_pnp_inward_pitch_T = true;
  }
  measurement.confidence = detection.confidence;
  measurement.color_id = detection.color_id;
  measurement.number_id = detection.number_id;
  measurement.keypoint_quality = *std::min_element(
      detection.keypoint_confidences.begin(), detection.keypoint_confidences.end());
  measurement.view_quality = 1.0;
  // 约束重投影 yaw 独立于 IPPE rvec；失败时保留 position-only 观测。
  const ReliableYaw yaw = yaw_solver.solve(exposure_snapshot, pose,
                                           detection.keypoints,
                                           detection.number_id);
  if (reliable_yaw != nullptr) *reliable_yaw = yaw;
  if (yaw.has_candidate_yaw) {
    measurement.inward_yaw_T_rad = yaw.inward_yaw_T_rad;
    measurement.has_raw_inward_yaw = true;
  }
  if (yaw.valid) {
    measurement.inward_yaw_T_rad = yaw.inward_yaw_T_rad;
    measurement.yaw_std_rad = yaw.yaw_std_rad;
    measurement.has_inward_yaw = true;
    measurement.view_quality = std::clamp(yaw.facing_cosine, 0.05, 1.0);
  }
  return measurement;
}

}  // namespace yolo_detect::tracking
