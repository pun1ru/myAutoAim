#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

namespace yolo_detect {

// 单块装甲板使用的角点数量，顺序固定为 BL、TL、TR、BR。
inline constexpr std::size_t kArmorPointCount = 4;

// 用于 PnP 物理模型的装甲板尺寸。
enum class ArmorSize {
  Small,  // 宽度为 0.135 m 的小装甲板。
  Large,  // 宽度为 0.225 m 的大装甲板。
};

// 位姿估计的成功状态或输入/求解失败原因。
enum class PoseStatus {
  Success,                        // 成功选出有效的正深度位姿。
  InsufficientKeypointConfidence, // 关键点置信度低于调用方阈值。
  ImageSizeMismatch,              // 输入图像分辨率与标定不一致。
  NonFiniteImagePoint,            // 图像点包含 NaN 或无穷值。
  ImagePointOutsideFrame,         // 图像点位于标定图像边界外。
  DegenerateQuadrilateral,        // 四边形面积过小，无法稳定求解。
  SelfIntersectingQuadrilateral,  // 四边形对边相交。
  InvalidPointOrder,              // 点顺序不符合 BL、TL、TR、BR。
  SolverFailure,                  // OpenCV IPPE 未返回候选解。
  NoValidCandidate,               // 所有候选解均无效或位于相机后方。
};

// OpenCV PnP 所需的相机内参、畸变系数和标定分辨率。
struct CameraCalibration {
  cv::Matx33d camera_matrix = cv::Matx33d::eye();  // 3x3 相机内参矩阵 K。
  std::vector<double> distortion_coefficients;     // OpenCV 畸变参数 D。
  cv::Size image_size;  // 与该标定严格匹配的输入图像分辨率。
};

// 单次 IPPE 求解的输出；仅当 valid 为 true 时位姿字段可用。
struct PoseResult {
  bool valid = false;  // 是否获得可用的正深度候选解。
  PoseStatus status = PoseStatus::SolverFailure;  // 求解结果状态。
  ArmorSize armor_size = ArmorSize::Small;  // 本次使用的物理板型。
  cv::Vec3d rvec_rad{};  // R_CA 的 Rodrigues 旋转向量，单位为弧度。
  cv::Matx33d rotation_camera_from_armor = cv::Matx33d::eye();  // R_CA。
  cv::Vec3d tvec_m{};  // 装甲板原点在相机 C 中的位置，单位为米。
  cv::Vec3d center_camera_m{};  // 装甲板中心在相机 C 中的位置，单位为米。
  cv::Vec3d armor_normal_camera{};  // 装甲板 +x_A 法线在相机 C 中的方向。
  double reprojection_rms_px = std::numeric_limits<double>::infinity();  // RMS 重投影误差。
  std::size_t candidate_count = 0;  // IPPE 返回的原始候选解数量。
  // The two raw IPPE orientation candidates are retained for web diagnostics.
  // They are not used by the constrained yaw solver or EKF.
  std::array<cv::Matx33d, 2> ippe_rotation_camera_from_armor{};
  std::array<bool, 2> has_ippe_rotation{};
};

// 将装甲板尺寸枚举转换为配置使用的板型名称。
[[nodiscard]] const char* armorSizeName(ArmorSize armor_size) noexcept;
// 将位姿估计状态转换为可读的诊断信息。
[[nodiscard]] const char* poseStatusName(PoseStatus status) noexcept;

class ArmorPoseEstimator {
 public:
  // 校验相机标定参数后创建估计器。
  explicit ArmorPoseEstimator(CameraCalibration calibration);

  // 使用 IPPE 估计装甲板位姿。image_points 必须为 BL、TL、TR、BR 顺序，且
  // image_size 必须等于标定分辨率。失败时返回 valid=false 和具体 status。
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
