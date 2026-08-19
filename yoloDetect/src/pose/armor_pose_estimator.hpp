#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

namespace yolo_detect {

inline constexpr std::size_t kArmorPointCount = 4;

enum class ArmorSize {
  Small,
  Large,
};

enum class PoseStatus {
  Success,
  InsufficientKeypointConfidence,
  ImageSizeMismatch,
  NonFiniteImagePoint,
  ImagePointOutsideFrame,
  DegenerateQuadrilateral,
  SelfIntersectingQuadrilateral,
  InvalidPointOrder,
  SolverFailure,
  NoValidCandidate,
};

struct CameraCalibration {
  cv::Matx33d camera_matrix = cv::Matx33d::eye();
  std::vector<double> distortion_coefficients;
  cv::Size image_size;
};

struct PoseResult {
  bool valid = false;
  PoseStatus status = PoseStatus::SolverFailure;
  ArmorSize armor_size = ArmorSize::Small;
  cv::Vec3d rvec_rad{};
  cv::Matx33d rotation_camera_from_armor = cv::Matx33d::eye();
  cv::Vec3d tvec_m{};
  cv::Vec3d center_camera_m{};
  cv::Vec3d armor_normal_camera{};
  double reprojection_rms_px = std::numeric_limits<double>::infinity();
  std::size_t candidate_count = 0;
};

// 将装甲板尺寸枚举转换为配置使用的板型名称。
[[nodiscard]] const char* armorSizeName(ArmorSize armor_size) noexcept;
// 将位姿估计状态转换为可读的诊断信息。
[[nodiscard]] const char* poseStatusName(PoseStatus status) noexcept;

class ArmorPoseEstimator {
 public:
  // 校验相机标定参数后创建估计器。
  explicit ArmorPoseEstimator(CameraCalibration calibration);

  [[nodiscard]] PoseResult estimate(
      const std::array<cv::Point2f, kArmorPointCount>& image_points,
      cv::Size image_size, ArmorSize armor_size) const;

  // 返回所有位姿估计使用的相机标定参数。
  [[nodiscard]] const CameraCalibration& calibration() const noexcept;
  // 按 BL、TL、TR、BR 顺序返回以中心为原点的装甲板坐标角点。
  [[nodiscard]] static std::array<cv::Point3d, kArmorPointCount>
  objectPoints(ArmorSize armor_size);

 private:
  CameraCalibration calibration_;
};

}  // namespace yolo_detect
