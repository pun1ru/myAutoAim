#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <filesystem>
#include <memory>
#include <vector>

namespace yolo_detect {

// The order is preserved from the CVAT data and the trained model.
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
  // RobotDetectionModel metadata. -1 means the model did not provide it.
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
  YoloPoseDetector(const std::filesystem::path& model_path,
                   DetectorConfig config);
  ~YoloPoseDetector();

  YoloPoseDetector(const YoloPoseDetector&) = delete;
  YoloPoseDetector& operator=(const YoloPoseDetector&) = delete;
  YoloPoseDetector(YoloPoseDetector&&) noexcept;
  YoloPoseDetector& operator=(YoloPoseDetector&&) noexcept;

  [[nodiscard]] std::vector<ArmorDetection> detect(const cv::Mat& bgr);
  [[nodiscard]] const DetectorConfig& config() const noexcept;

 private:
  struct LetterboxTransform {
    float scale = 1.0F;
    int left = 0;
    int top = 0;
  };

  [[nodiscard]] cv::Mat letterbox(const cv::Mat& source,
                                  LetterboxTransform& transform) const;
  [[nodiscard]] cv::Point2f restorePoint(
      float x, float y, const LetterboxTransform& transform,
      const cv::Size& source_size) const;

  struct Impl;
  std::unique_ptr<Impl> impl_;
  DetectorConfig config_;
};

}  // namespace yolo_detect
