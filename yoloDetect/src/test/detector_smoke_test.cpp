#include "detection/yolo_pose_detector.hpp"

#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

// 加载指定模型和图像，验证一次端到端检测结果。
int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "usage: yolo_pose_smoke_test <model.onnx> <image> [--cpu]\n";
    return 2;
  }

  const cv::Mat image = cv::imread(argv[2]);
  if (image.empty()) {
    std::cerr << "could not read image: " << argv[2] << '\n';
    return 3;
  }

  yolo_detect::DetectorConfig config;
  config.confidence_threshold = 0.01F;
  if (argc == 4 && std::string(argv[3]) == "--cpu") {
    config.use_cuda = false;
  }
#ifdef YOLO_DETECT_DEFAULT_CUDA_LIB_DIR
  config.cuda_library_directory = YOLO_DETECT_DEFAULT_CUDA_LIB_DIR;
#endif
  yolo_detect::YoloPoseDetector detector(std::filesystem::path(argv[1]),
                                          config);
  const auto detections = detector.detect(image);
  if (detections.empty()) {
    std::cerr << "model returned no detections at confidence 0.01\n";
    return 4;
  }

  for (const auto& detection : detections) {
    if (!std::isfinite(detection.confidence) || detection.confidence < 0.01F) {
      std::cerr << "invalid detection confidence\n";
      return 5;
    }
    for (const cv::Point2f& point : detection.keypoints) {
      if (!std::isfinite(point.x) || !std::isfinite(point.y) || point.x < 0.0F ||
          point.y < 0.0F || point.x >= image.cols || point.y >= image.rows) {
        std::cerr << "invalid restored keypoint\n";
        return 6;
      }
    }
  }

  const auto& first = detections.front();
  std::cout << "detections=" << detections.size()
            << " best_confidence=" << first.confidence << '\n'
            << "point_order=bottom-left,top-left,top-right,bottom-right\n";
  for (std::size_t index = 0; index < first.keypoints.size(); ++index) {
    std::cout << "point" << (index + 1) << '=' << first.keypoints[index].x << ','
              << first.keypoints[index].y
              << " confidence=" << first.keypoint_confidences[index] << '\n';
  }
  return 0;
}
