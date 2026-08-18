#include "yolo_pose_detector.hpp"

#include <onnxruntime_cxx_api.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif

namespace yolo_detect {

namespace {

constexpr int kBoxValues = 4;
constexpr int kClassValues = 1;
constexpr int kValuesPerKeypoint = 3;
constexpr int kExpectedValues =
    kBoxValues + kClassValues +
    static_cast<int>(KeypointCount) * kValuesPerKeypoint;
constexpr int kRobotValues = 22;
constexpr int kRobotConfidenceIndex = 8;
constexpr int kRobotColorBegin = 9;
constexpr int kRobotColorCount = 4;
constexpr int kRobotNumberBegin = 13;
constexpr int kRobotNumberCount = 9;

float clamp(float value, float lower, float upper) {
  return std::max(lower, std::min(value, upper));
}

float sigmoid(float value) {
  if (value >= 0.0F) {
    const float e = std::exp(-value);
    return 1.0F / (1.0F + e);
  }
  const float e = std::exp(value);
  return e / (1.0F + e);
}

float intersectionOverUnion(const cv::Rect2f& lhs, const cv::Rect2f& rhs) {
  const float left = std::max(lhs.x, rhs.x);
  const float top = std::max(lhs.y, rhs.y);
  const float right = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
  const float bottom = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
  const float intersection =
      std::max(0.0F, right - left) * std::max(0.0F, bottom - top);
  const float union_area = lhs.area() + rhs.area() - intersection;
  return union_area > 0.0F ? intersection / union_area : 0.0F;
}

void addCudaLibraryDirectory(const std::filesystem::path& directory) {
#ifdef _WIN32
  if (directory.empty() || !std::filesystem::is_directory(directory)) {
    throw std::runtime_error("CUDA library directory not found: " +
                             directory.string());
  }
  if (!SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
                                LOAD_LIBRARY_SEARCH_USER_DIRS)) {
    throw std::runtime_error("SetDefaultDllDirectories failed");
  }
  if (AddDllDirectory(directory.c_str()) == nullptr) {
    throw std::runtime_error("AddDllDirectory failed for: " +
                             directory.string());
  }
#else
  (void)directory;
#endif
}

}  // namespace

struct YoloPoseDetector::Impl {
  Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "yolo_detect"};
  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> session;
  std::string input_name;
  std::string output_name;
  bool input_float16 = false;
};

YoloPoseDetector::YoloPoseDetector(const std::filesystem::path& model_path,
                                   DetectorConfig config)
    : impl_(std::make_unique<Impl>()), config_(config) {
  if (config_.input_width <= 0 || config_.input_height <= 0) {
    throw std::invalid_argument("model input size must be positive");
  }
  if (config_.confidence_threshold < 0.0F ||
      config_.confidence_threshold > 1.0F) {
    throw std::invalid_argument("confidence threshold must be in [0, 1]");
  }
  if (!std::filesystem::is_regular_file(model_path)) {
    throw std::runtime_error("model not found: " + model_path.string());
  }

  if (config_.use_cuda) {
    addCudaLibraryDirectory(config_.cuda_library_directory);
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = config_.cuda_device;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
    cuda_options.do_copy_in_default_stream = 1;
    impl_->session_options.AppendExecutionProvider_CUDA(cuda_options);
  }

  impl_->session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  impl_->session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  impl_->session = std::make_unique<Ort::Session>(
      impl_->environment, model_path.c_str(), impl_->session_options);

  Ort::AllocatorWithDefaultOptions allocator;
  const auto input_name = impl_->session->GetInputNameAllocated(0, allocator);
  const auto output_name = impl_->session->GetOutputNameAllocated(0, allocator);
  impl_->input_name = input_name.get();
  impl_->output_name = output_name.get();

  const std::vector<std::int64_t> input_shape =
      impl_->session->GetInputTypeInfo(0)
          .GetTensorTypeAndShapeInfo()
          .GetShape();
  const auto input_type = impl_->session->GetInputTypeInfo(0)
                              .GetTensorTypeAndShapeInfo()
                              .GetElementType();
  impl_->input_float16 =
      input_type == ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  if (input_shape.size() != 4 ||
      (input_shape[2] > 0 && input_shape[2] != config_.input_height) ||
      (input_shape[3] > 0 && input_shape[3] != config_.input_width)) {
    throw std::runtime_error("configured input size does not match ONNX model");
  }
}

YoloPoseDetector::~YoloPoseDetector() = default;
YoloPoseDetector::YoloPoseDetector(YoloPoseDetector&&) noexcept = default;
YoloPoseDetector& YoloPoseDetector::operator=(YoloPoseDetector&&) noexcept =
    default;

cv::Mat YoloPoseDetector::letterbox(
    const cv::Mat& source, LetterboxTransform& transform) const {
  const float width_scale =
      static_cast<float>(config_.input_width) / source.cols;
  const float height_scale =
      static_cast<float>(config_.input_height) / source.rows;
  transform.scale = std::min(width_scale, height_scale);

  const int resized_width =
      std::max(1, cvRound(source.cols * transform.scale));
  const int resized_height =
      std::max(1, cvRound(source.rows * transform.scale));
  transform.left = (config_.input_width - resized_width) / 2;
  transform.top = (config_.input_height - resized_height) / 2;

  cv::Mat resized;
  cv::resize(source, resized, {resized_width, resized_height}, 0.0, 0.0,
             transform.scale < 1.0F ? cv::INTER_AREA : cv::INTER_LINEAR);
  cv::Mat canvas(config_.input_height, config_.input_width, CV_8UC3,
                 cv::Scalar(114, 114, 114));
  resized.copyTo(canvas(cv::Rect(transform.left, transform.top, resized_width,
                                 resized_height)));
  return canvas;
}

cv::Point2f YoloPoseDetector::restorePoint(
    float x, float y, const LetterboxTransform& transform,
    const cv::Size& source_size) const {
  x = (x - transform.left) / transform.scale;
  y = (y - transform.top) / transform.scale;
  return {clamp(x, 0.0F, static_cast<float>(source_size.width - 1)),
          clamp(y, 0.0F, static_cast<float>(source_size.height - 1))};
}

std::vector<ArmorDetection> YoloPoseDetector::detect(const cv::Mat& bgr) {
  if (bgr.empty() || bgr.type() != CV_8UC3) {
    throw std::invalid_argument("detector input must be a non-empty BGR image");
  }

  LetterboxTransform transform;
  const cv::Mat letterboxed = letterbox(bgr, transform);
  cv::Mat rgb;
  cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
  cv::Mat normalized;
  rgb.convertTo(normalized, CV_32FC3, 1.0 / 255.0);

  const std::size_t plane_size =
      static_cast<std::size_t>(config_.input_width) * config_.input_height;
  std::vector<float> input_values(plane_size * 3);
  std::array<cv::Mat, 3> input_planes = {
      cv::Mat(config_.input_height, config_.input_width, CV_32F,
              input_values.data()),
      cv::Mat(config_.input_height, config_.input_width, CV_32F,
              input_values.data() + plane_size),
      cv::Mat(config_.input_height, config_.input_width, CV_32F,
              input_values.data() + plane_size * 2),
  };
  cv::split(normalized, input_planes.data());

  const std::array<std::int64_t, 4> input_shape = {
      1, 3, config_.input_height, config_.input_width};
  const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(
      OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor{nullptr};
  std::vector<Ort::Float16_t> input_values_fp16;
  if (impl_->input_float16) {
    input_values_fp16.reserve(input_values.size());
    for (const float value : input_values) {
      input_values_fp16.emplace_back(value);
    }
    input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
        memory, input_values_fp16.data(), input_values_fp16.size(),
        input_shape.data(), input_shape.size());
  } else {
    input_tensor = Ort::Value::CreateTensor<float>(
        memory, input_values.data(), input_values.size(), input_shape.data(),
        input_shape.size());
  }

  const char* input_names[] = {impl_->input_name.c_str()};
  const char* output_names[] = {impl_->output_name.c_str()};
  std::vector<Ort::Value> outputs = impl_->session->Run(
      Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
  if (outputs.size() != 1 || !outputs[0].IsTensor()) {
    throw std::runtime_error("ONNX model did not return one tensor");
  }

  const std::vector<std::int64_t> output_shape =
      outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  if (output_shape.size() != 3 || output_shape[0] != 1) {
    throw std::runtime_error("unexpected YOLO output rank");
  }

  const bool pose_channel_first = output_shape[1] == kExpectedValues;
  const bool pose_channel_last = output_shape[2] == kExpectedValues;
  const bool robot_channel_first = output_shape[1] == kRobotValues;
  const bool robot_channel_last = output_shape[2] == kRobotValues;
  const bool robot_model = robot_channel_first || robot_channel_last;
  const bool channel_first = robot_model ? robot_channel_first
                                         : pose_channel_first;
  const bool channel_last = robot_model ? robot_channel_last
                                        : pose_channel_last;
  if (!channel_first && !channel_last) {
    throw std::runtime_error(
        "unexpected output: expected 17 (YOLO Pose) or 22 (0526 armor) values per candidate");
  }
  const int values_per_candidate = robot_model ? kRobotValues : kExpectedValues;
  const int candidate_count = static_cast<int>(
      channel_first ? output_shape[2] : output_shape[1]);
  const float* output = outputs[0].GetTensorData<float>();
  const auto value_at = [&](int candidate, int value) {
    return channel_first
               ? output[static_cast<std::size_t>(value) * candidate_count +
                        candidate]
               : output[static_cast<std::size_t>(candidate) * values_per_candidate +
                        value];
  };

  std::vector<ArmorDetection> candidates;
  candidates.reserve(candidate_count);

  for (int candidate_index = 0; candidate_index < candidate_count;
       ++candidate_index) {
    const float confidence = robot_model
                                 ? sigmoid(value_at(candidate_index,
                                                    kRobotConfidenceIndex))
                                 : value_at(candidate_index, kBoxValues);
    if (confidence < config_.confidence_threshold) continue;

    ArmorDetection detection;
    detection.confidence = confidence;
    if (robot_model) {
      // 0526 native order is TL, BL, BR, TR. Expose BL, TL, TR, BR.
      constexpr std::array<int, KeypointCount> kOutputOrder = {1, 0, 3, 2};
      std::array<cv::Point2f, KeypointCount> native_points;
      for (std::size_t point_index = 0; point_index < KeypointCount;
           ++point_index) {
        const int offset = static_cast<int>(point_index) * 2;
        native_points[point_index] = restorePoint(
            value_at(candidate_index, offset),
            value_at(candidate_index, offset + 1), transform, bgr.size());
      }
      float min_x = native_points[0].x;
      float max_x = native_points[0].x;
      float min_y = native_points[0].y;
      float max_y = native_points[0].y;
      for (const cv::Point2f& point : native_points) {
        min_x = std::min(min_x, point.x);
        max_x = std::max(max_x, point.x);
        min_y = std::min(min_y, point.y);
        max_y = std::max(max_y, point.y);
      }
      detection.box = {min_x, min_y, std::max(1.0F, max_x - min_x),
                       std::max(1.0F, max_y - min_y)};
      for (std::size_t point_index = 0; point_index < KeypointCount;
           ++point_index) {
        detection.keypoints[point_index] = native_points[kOutputOrder[point_index]];
        detection.keypoint_confidences[point_index] = 1.0F;
      }
      int color_id = 0;
      int number_id = 0;
      for (int index = 1; index < kRobotColorCount; ++index) {
        if (value_at(candidate_index, kRobotColorBegin + index) >
            value_at(candidate_index, kRobotColorBegin + color_id)) {
          color_id = index;
        }
      }
      for (int index = 1; index < kRobotNumberCount; ++index) {
        if (value_at(candidate_index, kRobotNumberBegin + index) >
            value_at(candidate_index, kRobotNumberBegin + number_id)) {
          number_id = index;
        }
      }
      detection.color_id = color_id;
      detection.number_id = number_id;
    } else {
      const float center_x = value_at(candidate_index, 0);
      const float center_y = value_at(candidate_index, 1);
      const float width = value_at(candidate_index, 2);
      const float height = value_at(candidate_index, 3);
      const cv::Point2f top_left =
          restorePoint(center_x - width * 0.5F, center_y - height * 0.5F,
                       transform, bgr.size());
      const cv::Point2f bottom_right =
          restorePoint(center_x + width * 0.5F, center_y + height * 0.5F,
                       transform, bgr.size());
      detection.box = {top_left.x, top_left.y,
                       std::max(1.0F, bottom_right.x - top_left.x),
                       std::max(1.0F, bottom_right.y - top_left.y)};
      for (std::size_t point_index = 0; point_index < KeypointCount;
           ++point_index) {
        const int offset = kBoxValues + kClassValues +
                           static_cast<int>(point_index) * kValuesPerKeypoint;
        detection.keypoints[point_index] = restorePoint(
            value_at(candidate_index, offset),
            value_at(candidate_index, offset + 1), transform, bgr.size());
        detection.keypoint_confidences[point_index] =
            value_at(candidate_index, offset + 2);
      }
    }

    candidates.push_back(detection);
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const ArmorDetection& lhs, const ArmorDetection& rhs) {
              return lhs.confidence > rhs.confidence;
            });

  std::vector<ArmorDetection> detections;
  detections.reserve(candidates.size());
  for (const ArmorDetection& candidate : candidates) {
    bool suppressed = false;
    for (const ArmorDetection& selected : detections) {
      if (intersectionOverUnion(candidate.box, selected.box) >
          config_.nms_threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) detections.push_back(candidate);
  }
  return detections;
}

const DetectorConfig& YoloPoseDetector::config() const noexcept {
  return config_;
}

}  // namespace yolo_detect
