#include "tracking/tracker_measurement_adapter.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void requireNear(double actual, double expected, double tolerance,
                 const char* message) {
  require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
          message);
}

yolo_detect::coordinates::CoordinateSnapshot exposureSnapshot() {
  namespace coordinates = yolo_detect::coordinates;
  coordinates::CoordinateObservation observation;
  observation.frame_sequence = 42;
  observation.capture_timestamp_ns = 987654321ULL;
  observation.ros_odom_world = true;
  observation.has_chassis_pose = true;
  observation.has_gimbal_pose = true;
  observation.has_camera_pose = true;
  const coordinates::CoordinateSnapshot snapshot =
      coordinates::makeCoordinateSnapshot(observation);
  if (!snapshot.valid) throw std::runtime_error("failed to make exposure snapshot");
  return snapshot;
}

void testCameraToTrackerUsesMatchingExposure() {
  const auto snapshot = exposureSnapshot();
  yolo_detect::CameraCalibration calibration;
  calibration.image_size = {1440, 1080};
  const yolo_detect::tracking::ConstrainedYawSolver yaw_solver(calibration);
  yolo_detect::PoseResult pose;
  pose.valid = true;
  pose.center_camera_m = {1.0, 2.0, 3.0};
  pose.reprojection_rms_px = 0.5;
  yolo_detect::ArmorDetection detection;
  detection.confidence = 0.9F;
  detection.keypoint_confidences.fill(0.8F);
  detection.color_id = 1;
  detection.number_id = 3;
  const yolo_detect::tracking::TrackerFrame frame{42, 987654321ULL};

  const auto measurement = yolo_detect::tracking::makeTrackerMeasurement(
      frame, snapshot, pose, detection, yaw_solver);
  require(measurement.has_value(), "matching frame exposure must create measurement");
  require(measurement->timestamp_ns == frame.capture_timestamp_ns,
          "tracker measurement must retain image capture timestamp");
  requireNear(measurement->position_T_m.x(), 3.0, 1e-12,
              "camera plus z must become tracker plus x");
  requireNear(measurement->position_T_m.y(), -1.0, 1e-12,
              "camera plus x must become tracker minus y");
  requireNear(measurement->position_T_m.z(), -2.0, 1e-12,
              "camera plus y must become tracker minus z");
  requireNear(measurement->camera_range_m, std::sqrt(14.0), 1e-12,
              "measurement range must be measured from the camera center");
  require(!measurement->has_inward_yaw,
          "adapter must not derive yaw from PnP Rodrigues components");
}

void testMismatchedExposureRejected() {
  const auto snapshot = exposureSnapshot();
  yolo_detect::CameraCalibration calibration;
  calibration.image_size = {1440, 1080};
  const yolo_detect::tracking::ConstrainedYawSolver yaw_solver(calibration);
  yolo_detect::PoseResult pose;
  pose.valid = true;
  pose.center_camera_m = {1.0, 2.0, 3.0};
  pose.reprojection_rms_px = 0.5;
  yolo_detect::ArmorDetection detection;
  detection.confidence = 0.9F;
  detection.keypoint_confidences.fill(0.8F);
  const yolo_detect::tracking::TrackerFrame wrong_sequence{43, 987654321ULL};
  const yolo_detect::tracking::TrackerFrame wrong_timestamp{42, 987654322ULL};
  require(!yolo_detect::tracking::makeTrackerMeasurement(
               wrong_sequence, snapshot, pose, detection, yaw_solver),
          "different source sequence must be rejected");
  require(!yolo_detect::tracking::makeTrackerMeasurement(
               wrong_timestamp, snapshot, pose, detection, yaw_solver),
          "different capture timestamp must be rejected");
}

}  // namespace

int main() {
  try {
    testCameraToTrackerUsesMatchingExposure();
    testMismatchedExposureRejected();
    std::cout << "tracker measurement adapter tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "tracker measurement adapter test failed: " << error.what()
              << '\n';
    return 1;
  }
}
