#pragma once

#include "coordinates/coordinate_frames.hpp"
#include "pose/armor_pose_estimator.hpp"

#include <array>

namespace yolo_detect::tracking {

enum class ReliableYawStatus {
  Success,
  InvalidInput,
  ReprojectionTooLarge,
  WeakYawInformation,
  AmbiguousOrientation,
  BackFacingArmor,
  CenterAdjustmentTooLarge,
};

struct ConstrainedYawOptions {
  double max_reprojection_rms_px = 2.0;
  double max_yaw_std_rad = 0.45;
  double min_facing_cosine = 0.20;
  double min_opposite_margin_px = 0.50;
  double max_center_adjustment_m = 0.25;
  int coarse_search_steps = 360;
};

struct ReliableYaw {
  bool valid = false;
  ReliableYawStatus status = ReliableYawStatus::InvalidInput;
  double inward_yaw_T_rad = 0.0;
  double reprojection_rms_px = 0.0;
  double yaw_std_rad = 0.0;
  double facing_cosine = 0.0;
};

// Estimates the horizontal inward armor normal with an upright-armor
// constraint. It intentionally never reads PoseResult::rvec_rad or
// PoseResult::rotation_camera_from_armor.
class ConstrainedYawSolver {
 public:
  explicit ConstrainedYawSolver(CameraCalibration calibration,
                                ConstrainedYawOptions options = {});

  [[nodiscard]] ReliableYaw solve(
      const coordinates::CoordinateSnapshot& exposure_snapshot,
      const PoseResult& pose,
      const std::array<cv::Point2f, kArmorPointCount>& image_points) const;

 private:
  CameraCalibration calibration_;
  ConstrainedYawOptions options_;
};

[[nodiscard]] const char* reliableYawStatusName(
    ReliableYawStatus status) noexcept;

}  // namespace yolo_detect::tracking
