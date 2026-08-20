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
    const PoseResult& pose, const ArmorDetection& detection) {
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
  measurement.position_T_m = coordinates::cameraPointToTracker(
      exposure_snapshot, pose.center_camera_m);
  measurement.reprojection_rms_px = pose.reprojection_rms_px;
  measurement.confidence = detection.confidence;
  measurement.color_id = detection.color_id;
  measurement.number_id = detection.number_id;
  measurement.keypoint_quality = *std::min_element(
      detection.keypoint_confidences.begin(), detection.keypoint_confidences.end());
  measurement.view_quality = 1.0;
  measurement.has_inward_yaw = false;
  return measurement;
}

}  // namespace yolo_detect::tracking
