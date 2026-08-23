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
  // sp_vision_25-style constrained reprojection search. The center remains
  // the PnP center; only the candidate armor orientation is changed.
  double search_half_range_rad = 90.0 * 3.14159265358979323846 / 180.0;
  double search_step_rad = 1.0 * 3.14159265358979323846 / 180.0;
  double armor_pitch_rad = 15.0 * 3.14159265358979323846 / 180.0;
  double outpost_pitch_rad = -15.0 * 3.14159265358979323846 / 180.0;
  int outpost_number_id = 6;
  bool skip_balance_large_armor = true;
  // 只有同时满足误差、可观测性、正反面歧义和可见面约束才输出可靠 yaw。
  double max_reprojection_rms_px = 4.0;
  double max_yaw_std_rad = 0.45;
  // Strongly oblique plates produce deceptively sharp but systematically
  // biased yaw minima from detector corners. Keep them position-only.
  double min_facing_cosine = 0.65;
  double min_opposite_margin_px = 0.50;
  int coarse_search_steps = 360;
};

struct ReliableYaw {
  // yaw_std_rad 由重投影误差对 yaw 的有限差分曲率估计，直接进入 EKF R。
  bool valid = false;
  // A finite optimizer candidate may still fail a later reliability gate.
  // This is diagnostic only; EKF use still requires valid=true.
  bool has_candidate_yaw = false;
  ReliableYawStatus status = ReliableYawStatus::InvalidInput;
  double inward_yaw_T_rad = 0.0;
  double reprojection_rms_px = 0.0;
  double yaw_std_rad = 0.0;
  double facing_cosine = 0.0;
};

// 以“装甲板竖直”约束，从四个角点独立估计 T 系水平 inward yaw。
// 故意不读取 PoseResult::rvec_rad 或 rotation_camera_from_armor，以规避
// IPPE 平面姿态的 yaw 翻转和不稳定性。
// Four-corner reprojection with the sp_vision_25 fixed-pitch prior estimates
// horizontal inward yaw. Raw IPPE rotation remains diagnostic only because a
// planar IPPE branch can flip between frames.
class ConstrainedYawSolver {
 public:
  explicit ConstrainedYawSolver(CameraCalibration calibration,
                                ConstrainedYawOptions options = {});

  [[nodiscard]] const ConstrainedYawOptions& options() const noexcept {
    return options_;
  }

  // Validate and apply runtime solver thresholds without changing calibration.
  [[nodiscard]] bool setOptions(ConstrainedYawOptions options) noexcept;

  [[nodiscard]] ReliableYaw solve(
      const coordinates::CoordinateSnapshot& exposure_snapshot,
      const PoseResult& pose,
      const std::array<cv::Point2f, kArmorPointCount>& image_points,
      // RobotDetectionModel number id is used only to select the outpost
      // pitch prior; it is never treated as a physical EKF slot.
      int number_id = -1) const;

 private:
  CameraCalibration calibration_;
  ConstrainedYawOptions options_;
};

[[nodiscard]] const char* reliableYawStatusName(
    ReliableYawStatus status) noexcept;

}  // namespace yolo_detect::tracking
