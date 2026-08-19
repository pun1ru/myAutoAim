#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <filesystem>
#include <memory>
#include <vector>

namespace yolo_detect {

// 顺序与 CVAT 标注数据及训练模型保持一致。
enum ArmorKeypoint : std::size_t {
  BottomLeft = 0,
  TopLeft = 1,
  TopRight = 2,
  BottomRight = 3,
  KeypointCount = 4,
};

struct ArmorDetection {
  cv::Rect2f box;
  std::array<cv::Point2f, KeypointCount> keypoints;
  std::array<float, KeypointCount> keypoint_confidences{};
  float confidence = 0.0F;
  // RobotDetectionModel 元数据；-1 表示模型未提供该字段。
  int color_id = -1;
  int number_id = -1;
};

struct DetectorConfig {
  int input_width = 640;
  int input_height = 640;
  float confidence_threshold = 0.01F;
  float nms_threshold = 0.45F;
  bool use_cuda = true;
  int cuda_device = 0;
  std::filesystem::path cuda_library_directory;
};

class YoloPoseDetector {
 public:
  // 加载 ONNX 姿态模型并配置选定的推理后端。
  YoloPoseDetector(const std::filesystem::path& model_path,
                   DetectorConfig config);
  // 释放 ONNX Runtime 会话和推理后端资源。
  ~YoloPoseDetector();

  YoloPoseDetector(const YoloPoseDetector&) = delete;
  YoloPoseDetector& operator=(const YoloPoseDetector&) = delete;
  YoloPoseDetector(YoloPoseDetector&&) noexcept;
  YoloPoseDetector& operator=(YoloPoseDetector&&) noexcept;

  // 对 BGR 图像执行推理，并将检测结果还原到原始像素坐标。
  [[nodiscard]] std::vector<ArmorDetection> detect(const cv::Mat& bgr);
  // 返回构造期间校验过的检测器配置。
  [[nodiscard]] const DetectorConfig& config() const noexcept;

 private:
  struct LetterboxTransform {
    float scale = 1.0F;
    int left = 0;
    int top = 0;
  };

  // 将图像缩放到模型画布，同时保留逆变换几何信息。
  [[nodiscard]] cv::Mat letterbox(const cv::Mat& source,
                                  LetterboxTransform& transform) const;
  // 将模型空间点映射回限制在边界内的原图坐标。
  [[nodiscard]] cv::Point2f restorePoint(
      float x, float y, const LetterboxTransform& transform,
      const cv::Size& source_size) const;

  struct Impl;
  std::unique_ptr<Impl> impl_;
  DetectorConfig config_;
};

}  // namespace yolo_detect
