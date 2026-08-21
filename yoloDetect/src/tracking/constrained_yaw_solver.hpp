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
};

struct ConstrainedYawOptions {
  // 只有同时满足误差、可观测性、正反面歧义和可见面约束才输出可靠 yaw。
  double max_reprojection_rms_px = 4.0;
  double max_yaw_std_rad = 0.45;
  double min_facing_cosine = 0.20;
  double min_opposite_margin_px = 0.50;
  int coarse_search_steps = 360;
};

struct ReliableYaw {
  // yaw_std_rad 由重投影误差对 yaw 的有限差分曲率估计，直接进入 EKF R。
  bool valid = false;
  ReliableYawStatus status = ReliableYawStatus::InvalidInput;
  double inward_yaw_T_rad = 0.0;
  // Translation jointly refined while armor roll/pitch remain constrained.
  // This prevents a biased free-IPPE tvec from being absorbed by yaw.
  cv::Vec3d refined_center_camera_m{0.0, 0.0, 0.0};
  double reprojection_rms_px = 0.0;
  double yaw_std_rad = 0.0;
  double facing_cosine = 0.0;
};

// 以“装甲板竖直”约束，从四个角点独立估计 T 系水平 inward yaw。
// 故意不读取 PoseResult::rvec_rad 或 rotation_camera_from_armor，以规避
// IPPE 平面姿态的 yaw 翻转和不稳定性。
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
