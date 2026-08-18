#include <daedalus_sim_sdk/scene_control_client.hpp>
#include <daedalus_sim_sdk/tcp_image_client.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// 默认图像阈值：亮度越高越严格，颜色差越高越能抑制白色噪声。
#define TRADITIONAL_DEFAULT_BRIGHTNESS_THRESHOLD 180
#define TRADITIONAL_DEFAULT_COLOR_DIFFERENCE 60
// 形态学闭运算半径：用于连接断裂灯条，过大会粘连相邻目标。
#define TRADITIONAL_DEFAULT_MORPHOLOGY_RADIUS 1
// 轮廓面积下限：过滤小噪点，过大会漏掉远处装甲板。
#define TRADITIONAL_DEFAULT_MIN_CONTOUR_AREA 0
// 1 检测蓝色，0 检测红色（输入图像已经是 BGR）。
#define TRADITIONAL_DEFAULT_ENEMY_IS_BLUE 1

// 灯条自身的几何筛选条件。
#define TRADITIONAL_LIGHTBAR_MIN_LENGTH 3.0F
#define TRADITIONAL_LIGHTBAR_MAX_LENGTH 100.0F
#define TRADITIONAL_LIGHTBAR_MIN_ASPECT_RATIO 4.0F
#define TRADITIONAL_LIGHTBAR_MAX_ASPECT_RATIO 30.0F
#define TRADITIONAL_FRAGMENT_MIN_LENGTH 8.0F
#define TRADITIONAL_FRAGMENT_MIN_ASPECT_RATIO 3.0F
#define TRADITIONAL_FRAGMENT_MERGE_MAX_GAP_PX 3.0F
#define TRADITIONAL_FRAGMENT_MERGE_MAX_ANGLE_DIFFERENCE_DEG 10.0F
#define TRADITIONAL_FRAGMENT_MERGE_MAX_LATERAL_OFFSET_PX 4.0F
// 两根灯条组成装甲板时的匹配条件。
#define TRADITIONAL_PAIR_MAX_LENGTH_RATIO 2.0F
#define TRADITIONAL_PAIR_MAX_ANGLE_DIFFERENCE_DEG 12.0F
#define TRADITIONAL_PAIR_MAX_HEIGHT_DIFFERENCE 3.0F
#define TRADITIONAL_PAIR_MIN_SEPARATION 0.5F
#define TRADITIONAL_PAIR_MAX_SEPARATION 5.0F
// 配对排序时，角度差在评分中的权重。
#define TRADITIONAL_PAIR_ANGLE_SCORE_WEIGHT 0.05F

// 模拟器靶场控制的默认参数。
#define TRADITIONAL_TARGET_ID 3
#define TRADITIONAL_MOTION_DIRECTION_DEG 90.0F
#define TRADITIONAL_MOTION_LINEAR_SPEED_MPS 2.0F
#define TRADITIONAL_MOTION_LINEAR_SPAN_M 4.0F
#define TRADITIONAL_MOTION_SPIN_SPEED_DEG_S 180.0F
#define TRADITIONAL_MOTION_SPEED_STEP_MPS 0.25F
#define TRADITIONAL_DISPLAY_WIDTH 1100

using namespace daedalus::sim::sdk::v1;
using Clock = std::chrono::steady_clock;

struct DetectorParameters {
  int brightness_threshold = TRADITIONAL_DEFAULT_BRIGHTNESS_THRESHOLD;
  int color_difference_threshold = TRADITIONAL_DEFAULT_COLOR_DIFFERENCE;
  int morphology_size = TRADITIONAL_DEFAULT_MORPHOLOGY_RADIUS;
  int minimum_contour_area = TRADITIONAL_DEFAULT_MIN_CONTOUR_AREA;
  int enemy_is_blue = TRADITIONAL_DEFAULT_ENEMY_IS_BLUE;
  float lightbar_min_length = TRADITIONAL_LIGHTBAR_MIN_LENGTH;
  float lightbar_max_length = TRADITIONAL_LIGHTBAR_MAX_LENGTH;
  float lightbar_min_aspect_ratio = TRADITIONAL_LIGHTBAR_MIN_ASPECT_RATIO;
  float lightbar_max_aspect_ratio = TRADITIONAL_LIGHTBAR_MAX_ASPECT_RATIO;
  float fragment_min_length = TRADITIONAL_FRAGMENT_MIN_LENGTH;
  float fragment_min_aspect_ratio = TRADITIONAL_FRAGMENT_MIN_ASPECT_RATIO;
  float fragment_merge_max_gap_px = TRADITIONAL_FRAGMENT_MERGE_MAX_GAP_PX;
  float fragment_merge_max_angle_difference_deg =
      TRADITIONAL_FRAGMENT_MERGE_MAX_ANGLE_DIFFERENCE_DEG;
  float fragment_merge_max_lateral_offset_px =
      TRADITIONAL_FRAGMENT_MERGE_MAX_LATERAL_OFFSET_PX;
  float pair_max_length_ratio = TRADITIONAL_PAIR_MAX_LENGTH_RATIO;
  float pair_max_angle_difference_deg =
      TRADITIONAL_PAIR_MAX_ANGLE_DIFFERENCE_DEG;
  float pair_max_height_difference = TRADITIONAL_PAIR_MAX_HEIGHT_DIFFERENCE;
  float pair_min_separation = TRADITIONAL_PAIR_MIN_SEPARATION;
  float pair_max_separation = TRADITIONAL_PAIR_MAX_SEPARATION;
  float pair_angle_score_weight = TRADITIONAL_PAIR_ANGLE_SCORE_WEIGHT;
};

struct LightBar {
  cv::RotatedRect rect;
  cv::Point2f top;
  cv::Point2f bottom;
  float length = 0.0F;
  float width = 0.0F;
  float angle_deg = 0.0F;
};

struct DetectionResult {
  cv::Mat mask;
  std::size_t fragment_count = 0;
  std::vector<LightBar> light_bars;
  // 顺序固定为：左上、右上、右下、左下。
  std::vector<std::array<cv::Point2f, 4>> armor_corners;
};

float normalizeAngle180(float angle_deg) {
  angle_deg = std::fmod(angle_deg, 180.0F);
  return angle_deg < 0.0F ? angle_deg + 180.0F : angle_deg;
}

float angleDifference180(float lhs_deg, float rhs_deg) {
  const float difference = std::abs(normalizeAngle180(lhs_deg) -
                                    normalizeAngle180(rhs_deg));
  return std::min(difference, 180.0F - difference);
}

LightBar makeLightBar(const cv::RotatedRect& rect) {
  LightBar bar;
  bar.rect = rect;
  bar.length = std::max(rect.size.width, rect.size.height);
  bar.width = std::max(1.0F, std::min(rect.size.width, rect.size.height));
  bar.angle_deg = rect.angle;
  // OpenCV's angle follows the rectangle width edge. Rotate it only when
  // that edge is the short side, so the direction follows the light bar.
  if (rect.size.width < rect.size.height) {
    bar.angle_deg += 90.0F;
  }

  const float angle_rad = bar.angle_deg * static_cast<float>(CV_PI / 180.0);
  const cv::Point2f direction(std::cos(angle_rad), std::sin(angle_rad));
  const cv::Point2f endpoint_a = rect.center + direction * (bar.length * 0.5F);
  const cv::Point2f endpoint_b = rect.center - direction * (bar.length * 0.5F);
  if (endpoint_a.y < endpoint_b.y) {
    bar.top = endpoint_a;
    bar.bottom = endpoint_b;
  } else {
    bar.top = endpoint_b;
    bar.bottom = endpoint_a;
  }
  return bar;
}

float endpointGap(const LightBar& lhs, const LightBar& rhs) {
  return std::min({cv::norm(lhs.top - rhs.top), cv::norm(lhs.top - rhs.bottom),
                   cv::norm(lhs.bottom - rhs.top),
                   cv::norm(lhs.bottom - rhs.bottom)});
}

float lateralOffset(const LightBar& reference, const LightBar& candidate) {
  const cv::Point2f axis = reference.bottom - reference.top;
  const float axis_length = cv::norm(axis);
  if (axis_length <= 1.0e-3F) return std::numeric_limits<float>::infinity();
  const cv::Point2f unit_axis = axis * (1.0F / axis_length);
  const cv::Point2f delta = candidate.rect.center - reference.rect.center;
  return std::abs(unit_axis.x * delta.y - unit_axis.y * delta.x);
}

bool canMergeFragments(const LightBar& lhs, const LightBar& rhs,
                       const DetectorParameters& parameters) {
  return angleDifference180(lhs.angle_deg, rhs.angle_deg) <=
             parameters.fragment_merge_max_angle_difference_deg &&
         endpointGap(lhs, rhs) <= parameters.fragment_merge_max_gap_px &&
         lateralOffset(lhs, rhs) <=
             parameters.fragment_merge_max_lateral_offset_px &&
         lateralOffset(rhs, lhs) <=
             parameters.fragment_merge_max_lateral_offset_px;
}

LightBar mergeFragmentGroup(const std::vector<LightBar>& fragments,
                            const std::vector<int>& members) {
  if (members.size() == 1) return fragments[members.front()];

  std::vector<cv::Point2f> endpoints;
  endpoints.reserve(members.size() * 2);
  float widest_fragment = 1.0F;
  for (const int member : members) {
    endpoints.push_back(fragments[member].top);
    endpoints.push_back(fragments[member].bottom);
    widest_fragment = std::max(widest_fragment, fragments[member].width);
  }

  cv::Vec4f line;
  cv::fitLine(endpoints, line, cv::DIST_L2, 0.0, 0.01, 0.01);
  cv::Point2f direction(line[0], line[1]);
  direction *= 1.0F / cv::norm(direction);
  const cv::Point2f origin(line[2], line[3]);
  float minimum_projection = std::numeric_limits<float>::max();
  float maximum_projection = std::numeric_limits<float>::lowest();
  for (const cv::Point2f& point : endpoints) {
    const float projection = (point - origin).dot(direction);
    minimum_projection = std::min(minimum_projection, projection);
    maximum_projection = std::max(maximum_projection, projection);
  }

  const cv::Point2f endpoint_a = origin + direction * minimum_projection;
  const cv::Point2f endpoint_b = origin + direction * maximum_projection;
  LightBar merged;
  merged.top = endpoint_a.y <= endpoint_b.y ? endpoint_a : endpoint_b;
  merged.bottom = endpoint_a.y <= endpoint_b.y ? endpoint_b : endpoint_a;
  merged.length = cv::norm(merged.bottom - merged.top);
  merged.width = widest_fragment;
  merged.angle_deg = std::atan2(merged.bottom.y - merged.top.y,
                                merged.bottom.x - merged.top.x) *
                     static_cast<float>(180.0 / CV_PI);
  merged.rect = cv::RotatedRect(
      (merged.top + merged.bottom) * 0.5F,
      cv::Size2f(merged.length, merged.width), merged.angle_deg);
  return merged;
}

DetectionResult runTraditionalDetector(const cv::Mat& bgr,
                                       const DetectorParameters& parameters) {
  DetectionResult result;

  // 通过 B-R 或 R-B 提取敌方颜色，再与亮度掩膜取交集。
  std::vector<cv::Mat> channels;
  cv::split(bgr, channels);
  cv::Mat color_score;
  if (parameters.enemy_is_blue != 0) {
    cv::subtract(channels[0], channels[2], color_score);  // B - R
  } else {
    cv::subtract(channels[2], channels[0], color_score);  // R - B
  }

  cv::Mat brightness;
  cv::max(channels[0], channels[1], brightness);
  cv::max(brightness, channels[2], brightness);
  cv::Mat color_mask;
  cv::Mat brightness_mask;
  cv::threshold(color_score, color_mask, parameters.color_difference_threshold,
                255, cv::THRESH_BINARY);
  cv::threshold(brightness, brightness_mask, parameters.brightness_threshold, 255,
                cv::THRESH_BINARY);
  cv::bitwise_and(color_mask, brightness_mask, result.mask);

  const int radius = std::max(parameters.morphology_size, 0);
  if (radius > 0) {
    const int size = radius * 2 + 1;
    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(size, size));
    cv::morphologyEx(result.mask, result.mask, cv::MORPH_CLOSE, kernel);
  }

  // 每个细长轮廓视为一根候选灯条。
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(result.mask, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  std::vector<LightBar> fragments;
  fragments.reserve(contours.size());
  for (const std::vector<cv::Point>& contour : contours) {
    const double area = cv::contourArea(contour);
    if (area < static_cast<double>(parameters.minimum_contour_area)) {
      continue;
    }
    const cv::RotatedRect rect = cv::minAreaRect(contour);
    const float length = std::max(rect.size.width, rect.size.height);
    const float width = std::max(1.0F, std::min(rect.size.width, rect.size.height));
    const float aspect_ratio = length / width;
    if (length < parameters.fragment_min_length ||
        length > parameters.lightbar_max_length ||
        aspect_ratio < parameters.fragment_min_aspect_ratio ||
        aspect_ratio > parameters.lightbar_max_aspect_ratio) {
      continue;
    }
    fragments.push_back(makeLightBar(rect));
  }
  result.fragment_count = fragments.size();

  // Join only nearby, nearly collinear pieces. This avoids a larger morphology
  // kernel while reconstructing a light bar interrupted by a dark gap.
  std::vector<int> parent(fragments.size());
  for (int index = 0; index < static_cast<int>(parent.size()); ++index) {
    parent[index] = index;
  }
  const auto findRoot = [&parent](int index) {
    int root = index;
    while (parent[root] != root) root = parent[root];
    while (parent[index] != index) {
      const int next = parent[index];
      parent[index] = root;
      index = next;
    }
    return root;
  };
  for (int first = 0; first < static_cast<int>(fragments.size()); ++first) {
    for (int second = first + 1;
         second < static_cast<int>(fragments.size()); ++second) {
      if (!canMergeFragments(fragments[first], fragments[second], parameters)) {
        continue;
      }
      const int first_root = findRoot(first);
      const int second_root = findRoot(second);
      if (first_root != second_root) parent[second_root] = first_root;
    }
  }

  std::vector<std::vector<int>> fragment_groups(fragments.size());
  for (int index = 0; index < static_cast<int>(fragments.size()); ++index) {
    fragment_groups[findRoot(index)].push_back(index);
  }
  result.light_bars.reserve(fragments.size());
  for (const std::vector<int>& group : fragment_groups) {
    if (group.empty()) continue;
    const LightBar merged = mergeFragmentGroup(fragments, group);
    const float aspect_ratio = merged.length / std::max(1.0F, merged.width);
    if (merged.length < parameters.lightbar_min_length ||
        merged.length > parameters.lightbar_max_length ||
        aspect_ratio < parameters.lightbar_min_aspect_ratio ||
        aspect_ratio > parameters.lightbar_max_aspect_ratio) {
      continue;
    }
    result.light_bars.push_back(merged);
  }

  struct PairCandidate {
    int left = 0;
    int right = 0;
    float score = 0.0F;
  };
  // 遍历灯条两两组合，筛选出几何关系合理的装甲板。
  std::vector<PairCandidate> pairs;
  for (int first = 0; first < static_cast<int>(result.light_bars.size()); ++first) {
    for (int second = first + 1;
         second < static_cast<int>(result.light_bars.size()); ++second) {
      int left = first;
      int right = second;
      if (result.light_bars[left].rect.center.x > result.light_bars[right].rect.center.x) {
        std::swap(left, right);
      }
      const LightBar& lhs = result.light_bars[left];
      const LightBar& rhs = result.light_bars[right];
      const float average_length = (lhs.length + rhs.length) * 0.5F;
      const float length_ratio = std::max(lhs.length, rhs.length) /
                                 std::max(1.0F, std::min(lhs.length, rhs.length));
      const float angle_difference = angleDifference180(lhs.angle_deg, rhs.angle_deg);
      const float height_difference =
          std::abs(lhs.rect.center.y - rhs.rect.center.y) / average_length;
      const float separation =
          std::abs(lhs.rect.center.x - rhs.rect.center.x) / average_length;
      if (length_ratio > parameters.pair_max_length_ratio ||
          angle_difference > parameters.pair_max_angle_difference_deg ||
          height_difference > parameters.pair_max_height_difference ||
          separation < parameters.pair_min_separation ||
          separation > parameters.pair_max_separation) {
        continue;
      }
      pairs.push_back({left, right,
                       length_ratio +
                           angle_difference * parameters.pair_angle_score_weight +
                           height_difference});
    }
  }

  std::sort(pairs.begin(), pairs.end(), [](const PairCandidate& lhs,
                                            const PairCandidate& rhs) {
    return lhs.score < rhs.score;
  });
  // 一个灯条只允许参与一组配对，避免同一目标重复输出。
  std::vector<bool> used(result.light_bars.size(), false);
  for (const PairCandidate& pair : pairs) {
    if (used[pair.left] || used[pair.right]) {
      continue;
    }
    const LightBar& left = result.light_bars[pair.left];
    const LightBar& right = result.light_bars[pair.right];
    result.armor_corners.push_back(
        {left.top, right.top, right.bottom, left.bottom});
    used[pair.left] = true;
    used[pair.right] = true;
  }
  return result;
}

bool sceneSucceeded(const ClientResult<SceneControlResponse>& response,
                    std::string_view operation) {
  if (!response) {
    std::cerr << operation << " failed: " << response.status.message << '\n';
    return false;
  }
  if (response.value->status != SceneControlStatus::Ok) {
    std::cerr << operation << " rejected: " << response.value->message << '\n';
    return false;
  }
  return true;
}

struct SceneState {
  std::string scene = "shooting-range";
  RangeMotionMode motion = RangeMotionMode::Stationary;
  std::uint8_t target = TRADITIONAL_TARGET_ID;
  float vehicle_speed = TRADITIONAL_MOTION_LINEAR_SPEED_MPS;
  float linear_span = TRADITIONAL_MOTION_LINEAR_SPAN_M;
  float direction_deg = TRADITIONAL_MOTION_DIRECTION_DEG;
  float spin_speed_deg_s = TRADITIONAL_MOTION_SPIN_SPEED_DEG_S;
};

bool applyMotion(SceneControlClient& scene, const SceneState& state) {
  RangeTargetMotion motion;
  motion.target = state.target;
  motion.mode = state.motion;
  motion.direction_deg = state.direction_deg;
  const bool has_linear = state.motion == RangeMotionMode::Linear ||
                          state.motion == RangeMotionMode::LinearAndSpin;
  const bool has_spin = state.motion == RangeMotionMode::Spin ||
                        state.motion == RangeMotionMode::LinearAndSpin;
  motion.linear_speed_mps = has_linear ? state.vehicle_speed : 0.0F;
  motion.linear_span_m = has_linear ? state.linear_span : 0.0F;
  motion.spin_deg_s = has_spin ? state.spin_speed_deg_s : 0.0F;
  return sceneSucceeded(scene.setRangeTargetMotion(motion), "setRangeTargetMotion");
}

void drawLightBar(cv::Mat& display, const LightBar& bar) {
  cv::Point2f vertices[4];
  bar.rect.points(vertices);
  for (int index = 0; index < 4; ++index) {
    cv::line(display, vertices[index], vertices[(index + 1) % 4],
             cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
  }
  cv::line(display, bar.top, bar.bottom, cv::Scalar(0, 255, 255), 2,
           cv::LINE_AA);
}

std::string fixed(double value, int precision = 1) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

void drawText(cv::Mat& image, const std::string& text, int line,
              const cv::Scalar& color = {235, 235, 235}) {
  const cv::Point origin(16, 28 + line * 25);
  cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.58,
              cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
  cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.58, color, 1,
              cv::LINE_AA);
}

constexpr int kControlRowHeight = 34;
constexpr int kControlFirstRowY = 54;
constexpr int kControlParameterCount = 20;
constexpr int kControlPanelHeight =
    kControlFirstRowY + kControlParameterCount * kControlRowHeight + 20;

struct ParameterEditor {
  int selected = -1;
  std::string input;
  bool replace_on_next_key = false;
};

struct ControlsContext {
  DetectorParameters* parameters = nullptr;
  ParameterEditor editor;
};

const std::array<const char*, kControlParameterCount> kParameterNames = {
    "Brightness threshold", "Color difference", "Morphology radius",
    "Minimum contour area", "Enemy color (0 red, 1 blue)",
    "Lightbar minimum length", "Lightbar maximum length",
    "Lightbar aspect ratio minimum", "Lightbar aspect ratio maximum",
    "Fragment minimum length", "Fragment minimum aspect ratio",
    "Fragment merge maximum gap (px)",
    "Fragment merge maximum angle difference (deg)",
    "Fragment merge maximum lateral offset (px)",
    "Pair maximum length ratio", "Pair maximum angle difference",
    "Pair maximum height difference", "Pair minimum separation",
    "Pair maximum separation", "Pair angle score weight"};

std::string parameterValue(const DetectorParameters& parameters, int index) {
  switch (index) {
    case 0: return std::to_string(parameters.brightness_threshold);
    case 1: return std::to_string(parameters.color_difference_threshold);
    case 2: return std::to_string(parameters.morphology_size);
    case 3: return std::to_string(parameters.minimum_contour_area);
    case 4: return std::to_string(parameters.enemy_is_blue);
    case 5: return fixed(parameters.lightbar_min_length, 2);
    case 6: return fixed(parameters.lightbar_max_length, 2);
    case 7: return fixed(parameters.lightbar_min_aspect_ratio, 2);
    case 8: return fixed(parameters.lightbar_max_aspect_ratio, 2);
    case 9: return fixed(parameters.fragment_min_length, 2);
    case 10: return fixed(parameters.fragment_min_aspect_ratio, 2);
    case 11: return fixed(parameters.fragment_merge_max_gap_px, 2);
    case 12: return fixed(parameters.fragment_merge_max_angle_difference_deg, 1);
    case 13: return fixed(parameters.fragment_merge_max_lateral_offset_px, 2);
    case 14: return fixed(parameters.pair_max_length_ratio, 2);
    case 15: return fixed(parameters.pair_max_angle_difference_deg, 1);
    case 16: return fixed(parameters.pair_max_height_difference, 2);
    case 17: return fixed(parameters.pair_min_separation, 2);
    case 18: return fixed(parameters.pair_max_separation, 2);
    case 19: return fixed(parameters.pair_angle_score_weight, 3);
  }
  return {};
}

bool applyParameterValue(DetectorParameters& parameters, int index,
                         const std::string& input) {
  try {
    if (index <= 4) {
      const int value = std::stoi(input);
      switch (index) {
        case 0:
          parameters.brightness_threshold = std::clamp(value, 0, 255);
          break;
        case 1:
          parameters.color_difference_threshold = std::clamp(value, 0, 255);
          break;
        case 2:
          parameters.morphology_size = std::clamp(value, 0, 20);
          break;
        case 3:
          parameters.minimum_contour_area = std::clamp(value, 0, 10000);
          break;
        case 4:
          parameters.enemy_is_blue = value == 0 ? 0 : 1;
          break;
      }
      return true;
    }

    const float value = std::stof(input);
    switch (index) {
      case 5: parameters.lightbar_min_length = std::clamp(value, 0.0F, 1000.0F); break;
      case 6: parameters.lightbar_max_length = std::clamp(value, 0.0F, 10000.0F); break;
      case 7: parameters.lightbar_min_aspect_ratio = std::clamp(value, 0.1F, 100.0F); break;
      case 8: parameters.lightbar_max_aspect_ratio = std::clamp(value, 0.1F, 100.0F); break;
      case 9: parameters.fragment_min_length = std::clamp(value, 0.5F, 1000.0F); break;
      case 10: parameters.fragment_min_aspect_ratio = std::clamp(value, 1.0F, 100.0F); break;
      case 11: parameters.fragment_merge_max_gap_px = std::clamp(value, 0.0F, 100.0F); break;
      case 12: parameters.fragment_merge_max_angle_difference_deg = std::clamp(value, 0.0F, 90.0F); break;
      case 13: parameters.fragment_merge_max_lateral_offset_px = std::clamp(value, 0.0F, 100.0F); break;
      case 14: parameters.pair_max_length_ratio = std::clamp(value, 1.0F, 20.0F); break;
      case 15: parameters.pair_max_angle_difference_deg = std::clamp(value, 0.0F, 90.0F); break;
      case 16: parameters.pair_max_height_difference = std::clamp(value, 0.0F, 10.0F); break;
      case 17: parameters.pair_min_separation = std::clamp(value, 0.0F, 20.0F); break;
      case 18: parameters.pair_max_separation = std::clamp(value, 0.0F, 20.0F); break;
      case 19: parameters.pair_angle_score_weight = std::clamp(value, 0.0F, 10.0F); break;
      default: return false;
    }
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

void controlsMouseCallback(int event, int, int y, int, void* userdata) {
  if (event != cv::EVENT_LBUTTONDOWN) return;
  auto* context = static_cast<ControlsContext*>(userdata);
  if (context == nullptr || context->parameters == nullptr) return;
  const int index = (y - kControlFirstRowY) / kControlRowHeight;
  if (y < kControlFirstRowY || index < 0 || index >= kControlParameterCount) {
    return;
  }
  context->editor.selected = index;
  context->editor.input = parameterValue(*context->parameters, index);
  context->editor.replace_on_next_key = true;
}

bool handleParameterKey(ControlsContext& context, int key) {
  ParameterEditor& editor = context.editor;
  if (editor.selected < 0 || context.parameters == nullptr) return false;
  if (key == 13 || key == 10) {
    const bool applied =
        applyParameterValue(*context.parameters, editor.selected, editor.input);
    if (applied) editor.selected = -1;
    return true;
  }
  if (key == 27) {
    editor.selected = -1;
    return true;
  }
  if (key == 8 || key == 127) {
    if (editor.replace_on_next_key) {
      editor.input.clear();
      editor.replace_on_next_key = false;
    } else if (!editor.input.empty()) {
      editor.input.pop_back();
    }
    return true;
  }
  const bool numeric = key >= '0' && key <= '9';
  const bool decimal_point = key == '.' && editor.selected >= 5 &&
                             editor.input.find('.') == std::string::npos;
  if (!numeric && !decimal_point) return true;
  if (editor.replace_on_next_key) {
    editor.input.clear();
    editor.replace_on_next_key = false;
  }
  editor.input.push_back(static_cast<char>(key));
  return true;
}

void drawControlsPanel(cv::Mat& panel, const ControlsContext& context) {
  panel = cv::Mat(kControlPanelHeight, 760, CV_8UC3, cv::Scalar(35, 35, 35));
  drawText(panel, "Click a value, type a number, then press Enter", 0,
           cv::Scalar(80, 255, 120));
  for (int index = 0; index < kControlParameterCount; ++index) {
    const int y = kControlFirstRowY + index * kControlRowHeight;
    const bool selected = context.editor.selected == index;
    if (selected) {
      cv::rectangle(panel, cv::Rect(8, y - 22, panel.cols - 16, 29),
                    cv::Scalar(75, 75, 25), cv::FILLED);
    }
    const std::string value = selected ? context.editor.input
                                       : parameterValue(*context.parameters, index);
    cv::putText(panel, kParameterNames[index], {16, y},
                cv::FONT_HERSHEY_SIMPLEX, 0.52, cv::Scalar(235, 235, 235), 1,
                cv::LINE_AA);
    cv::putText(panel, value, {570, y}, cv::FONT_HERSHEY_SIMPLEX, 0.58,
                selected ? cv::Scalar(80, 255, 120) : cv::Scalar(80, 220, 255),
                1, cv::LINE_AA);
  }
}

const char* motionName(RangeMotionMode mode) {
  switch (mode) {
    case RangeMotionMode::Stationary:
      return "stationary";
    case RangeMotionMode::Linear:
      return "linear";
    case RangeMotionMode::LinearAndSpin:
      return "linear+spin";
    case RangeMotionMode::Spin:
      return "spin";
  }
  return "unknown";
}

void drawArmor(cv::Mat& image,
               const std::array<cv::Point2f, 4>& corners,
               int armor_index) {
  // Detector order is TL, TR, BR, BL; display order follows YOLO: BL, TL, TR, BR.
  const std::array<int, 4> display_order = {3, 0, 1, 2};
  const std::array<cv::Scalar, 4> colors = {
      cv::Scalar(0, 255, 255), cv::Scalar(0, 255, 0),
      cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255)};

  std::vector<cv::Point> polygon;
  polygon.reserve(corners.size());
  for (const cv::Point2f& corner : corners) {
    polygon.emplace_back(cvRound(corner.x), cvRound(corner.y));
  }
  cv::polylines(image, polygon, true, cv::Scalar(30, 210, 255), 2,
                cv::LINE_AA);

  for (std::size_t index = 0; index < display_order.size(); ++index) {
    const cv::Point2f& point = corners[display_order[index]];
    cv::circle(image, point, 5, colors[index], cv::FILLED, cv::LINE_AA);
    cv::putText(image, std::to_string(index + 1),
                cv::Point(cvRound(point.x) + 6, cvRound(point.y) - 6),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[index], 2,
                cv::LINE_AA);
  }

  const cv::Point label_origin(
      cvRound(corners[0].x), std::max(20, cvRound(corners[0].y) - 8));
  const std::string label = "armor traditional " + std::to_string(armor_index);
  cv::putText(image, label, label_origin, cv::FONT_HERSHEY_SIMPLEX, 0.55,
              cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
  cv::putText(image, label, label_origin, cv::FONT_HERSHEY_SIMPLEX, 0.55,
              cv::Scalar(255, 180, 30), 1, cv::LINE_AA);
}

struct RuntimeOptions {
  std::filesystem::path record_path;
  double record_fps = 60.0;
  bool no_display = false;
};

RuntimeOptions parseRuntimeOptions(int argc, char** argv) {
  RuntimeOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--no-display") {
      options.no_display = true;
    } else if (argument == "--record" && index + 1 < argc) {
      options.record_path = argv[++index];
    } else if (argument == "--record-fps" && index + 1 < argc) {
      options.record_fps = std::stod(argv[++index]);
      if (options.record_fps <= 0.0 || options.record_fps > 240.0) {
        throw std::runtime_error("record-fps must be in (0, 240]");
      }
    } else if (argument == "--help") {
      std::cout << "Usage: traditional_detect [--no-display] [--record path] "
                   "[--record-fps fps]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown or incomplete option: " + argument);
    }
  }
  return options;
}

int main(int argc, char** argv) {
  RuntimeOptions runtime_options;
  try {
    runtime_options = parseRuntimeOptions(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  const bool recording = !runtime_options.record_path.empty();

  SceneState scene_state;
  SceneControlOptions scene_options;
  scene_options.session_id = "traditional-detect";
  SceneControlClient scene(scene_options);
  if (!sceneSucceeded(scene.createSession(), "createSession") ||
      !sceneSucceeded(scene.setScene(recording ? SceneMode::Outpost
                                                : SceneMode::ShootingRange),
                      "setScene") ||
      !sceneSucceeded(scene.resetScene(), "resetScene") ||
      (recording ? false : !applyMotion(scene, scene_state))) {
    return 1;
  }
  if (recording) {
    scene_state.scene = "outpost";
  }

  TcpImageClient images;
  const ClientStatus connected = images.connect();
  if (!connected) {
    std::cerr << "connect image stream failed: " << connected.message << '\n';
    return 2;
  }

  DetectorParameters parameters;
  ControlsContext controls_context;
  controls_context.parameters = &parameters;
  constexpr char kWindowName[] = "Traditional Armor Detection - Q/Esc to close";
  if (!runtime_options.no_display) {
    cv::namedWindow(kWindowName, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::namedWindow("controls", cv::WINDOW_NORMAL);
    cv::resizeWindow("controls", 760, kControlPanelHeight);
    cv::setMouseCallback("controls", controlsMouseCallback, &controls_context);
    cv::moveWindow(kWindowName, 20, 20);
    cv::resizeWindow(kWindowName, TRADITIONAL_DISPLAY_WIDTH,
                     TRADITIONAL_DISPLAY_WIDTH * 3 / 4);
    cv::namedWindow("mask", cv::WINDOW_NORMAL);
  }

  std::uint64_t previous_sequence = 0;
  std::uint64_t received_count = 0;
  std::uint64_t skipped_count = 0;
  std::uint64_t rate_received = 0;
  auto rate_started = Clock::now();
  double receive_fps = 0.0;
  double traditional_ms = 0.0;
  std::uint64_t recorded_frames = 0;
  cv::VideoWriter recorder;
  bool recorder_opened = false;
  const std::uint64_t scene_switch_frame = static_cast<std::uint64_t>(
      std::llround(runtime_options.record_fps * 8.0));
  const std::uint64_t max_record_frames = static_cast<std::uint64_t>(
      std::llround(runtime_options.record_fps * 16.0));
  bool timed_scene_switched = false;

  auto requestScene = [&](SceneMode requested, const char* name) {
    if (sceneSucceeded(scene.setScene(requested),
                       std::string("setScene ") + name)) {
      scene_state.scene = name;
      if (requested == SceneMode::ShootingRange) {
        applyMotion(scene, scene_state);
      }
    }
  };
  auto requestMotion = [&](RangeMotionMode requested) {
    SceneState next = scene_state;
    next.motion = requested;
    if (applyMotion(scene, next)) {
      scene_state = next;
    }
  };
  auto adjustSpeed = [&](float delta) {
    SceneState next = scene_state;
    next.vehicle_speed = std::clamp(
        next.vehicle_speed + delta, 0.0F, 20.0F);
    if (next.vehicle_speed <= 0.0F) {
      if (next.motion == RangeMotionMode::Linear) {
        next.motion = RangeMotionMode::Stationary;
      } else if (next.motion == RangeMotionMode::LinearAndSpin) {
        next.motion = RangeMotionMode::Spin;
      }
    } else if (next.motion == RangeMotionMode::Stationary) {
      next.motion = RangeMotionMode::Linear;
    } else if (next.motion == RangeMotionMode::Spin) {
      next.motion = RangeMotionMode::LinearAndSpin;
    }
    if (applyMotion(scene, next)) {
      scene_state = next;
    }
  };

  while (true) {
    auto frame = images.waitForLatest(previous_sequence,
                                      std::chrono::milliseconds(1000));
    if (!frame) {
      if (frame.status.error != ClientError::Timeout) {
        std::cerr << "waitForLatest failed: " << frame.status.message << '\n';
      }
      continue;
    }
    if (previous_sequence != 0 &&
        frame.value->header.source_sequence > previous_sequence + 1) {
      skipped_count += frame.value->header.source_sequence - previous_sequence - 1;
    }
    previous_sequence = frame.value->header.source_sequence;
    ++received_count;
    ++rate_received;

    if (recording && !timed_scene_switched &&
        recorded_frames >= scene_switch_frame) {
      scene_state.motion = RangeMotionMode::LinearAndSpin;
      if (sceneSucceeded(scene.setScene(SceneMode::ShootingRange),
                         "setScene shooting-range")) {
        scene_state.scene = "shooting-range";
        applyMotion(scene, scene_state);
      }
      timed_scene_switched = true;
    }

    const double rate_elapsed =
        std::chrono::duration<double>(Clock::now() - rate_started).count();
    if (rate_elapsed >= 1.0) {
      receive_fps = static_cast<double>(rate_received) / rate_elapsed;
      rate_received = 0;
      rate_started = Clock::now();
    }

    const auto& source = *frame.value;
    const int type = source.header.format == tcp_image::PixelFormat::Rgba32
                         ? CV_8UC4
                         : CV_8UC3;
    cv::Mat view(static_cast<int>(source.header.height),
                 static_cast<int>(source.header.width), type,
                 const_cast<std::uint8_t*>(source.payload.data()));

    cv::Mat bgr;
    if (source.header.format == tcp_image::PixelFormat::Rgba32) {
      cv::cvtColor(view, bgr, cv::COLOR_RGBA2BGR);
    } else {
      cv::cvtColor(view, bgr, cv::COLOR_RGB2BGR);
    }

    const auto inference_started = Clock::now();
    DetectionResult detection = runTraditionalDetector(bgr, parameters);
    traditional_ms = std::chrono::duration<double, std::milli>(
                         Clock::now() - inference_started)
                         .count();
    cv::Mat display = bgr.clone();
    for (const LightBar& light_bar : detection.light_bars) {
      drawLightBar(display, light_bar);
    }
    for (std::size_t index = 0; index < detection.armor_corners.size();
         ++index) {
      drawArmor(display, detection.armor_corners[index],
                static_cast<int>(index + 1));
    }

    drawText(display,
             "Traditional armor: " +
                 std::to_string(detection.armor_corners.size()) +
                 "   fragments: " +
                 std::to_string(detection.fragment_count) + " -> bars: " +
                 std::to_string(detection.light_bars.size()) +
                 "   process: " + fixed(traditional_ms) + " ms (" +
                 fixed(traditional_ms > 0.0 ? 1000.0 / traditional_ms : 0.0) +
                 " FPS)",
             0, cv::Scalar(80, 255, 120));
    drawText(display,
             "SDK RX: " + fixed(receive_fps) +
                 " FPS   frame: " +
                 std::to_string(source.header.source_sequence) +
                 "   received: " + std::to_string(received_count) +
                 "   skipped: " + std::to_string(skipped_count),
             1);
    drawText(display,
             "thresholds: brightness " +
                 std::to_string(parameters.brightness_threshold) +
                 "   color " +
                 std::to_string(parameters.color_difference_threshold) +
                 "   area " + std::to_string(parameters.minimum_contour_area),
             2, cv::Scalar(80, 220, 255));
    drawText(display,
             "scene: " + scene_state.scene +
                 "   control: ready   target: " +
                 std::to_string(scene_state.target),
             3, cv::Scalar(80, 255, 120));
    drawText(display,
             "motion: " + std::string(motionName(scene_state.motion)) +
                 "   vehicle speed: " +
                 fixed(scene_state.vehicle_speed, 2) +
                 " m/s   spin: " +
                 fixed(scene_state.spin_speed_deg_s, 1) + " deg/s",
             4, cv::Scalar(80, 220, 255));
    drawText(display,
             "color: " + std::string(parameters.enemy_is_blue != 0 ? "blue" : "red") +
                 "   points: 1 BL  2 TL  3 TR  4 BR",
             5, cv::Scalar(80, 220, 255));

    if (recording) {
      if (!recorder_opened) {
        const std::filesystem::path parent =
            runtime_options.record_path.parent_path();
        if (!parent.empty()) {
          std::error_code error;
          std::filesystem::create_directories(parent, error);
          if (error) {
            std::cerr << "record directory creation failed: "
                      << error.message() << '\n';
            break;
          }
        }
        const int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        recorder_opened = recorder.open(runtime_options.record_path.string(),
                                         codec, runtime_options.record_fps,
                                         display.size(), true);
        if (!recorder_opened) {
          std::cerr << "record open failed: "
                    << runtime_options.record_path.string() << '\n';
          break;
        }
        std::cout << "recording annotated video: "
                  << runtime_options.record_path.string() << " at "
                  << fixed(runtime_options.record_fps, 1) << " FPS\n";
      }
      recorder.write(display);
      ++recorded_frames;
      if (recorded_frames >= max_record_frames) {
        break;
      }
    }

    if (!runtime_options.no_display) {
      cv::Mat resized_display;
      const double display_scale =
          std::min(1.0, static_cast<double>(TRADITIONAL_DISPLAY_WIDTH) /
                            display.cols);
      cv::resize(display, resized_display, {}, display_scale, display_scale,
                 display_scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
      cv::imshow(kWindowName, resized_display);
      cv::Mat controls_panel;
      drawControlsPanel(controls_panel, controls_context);
      cv::imshow("controls", controls_panel);
      cv::imshow("mask", detection.mask);

      const int key = cv::waitKey(1) & 0xff;
      if (handleParameterKey(controls_context, key)) continue;
      if (key == 27 || key == 'q' || key == 'Q') break;
      if (key == '1') requestScene(SceneMode::Armor, "armor");
      if (key == '2') requestScene(SceneMode::Energy, "energy");
      if (key == '3') requestScene(SceneMode::Outpost, "outpost");
      if (key == '4') requestScene(SceneMode::ShootingRange, "shooting-range");
      if (key == '0' && sceneSucceeded(scene.resetScene(), "resetScene")) {
        scene_state.scene = "reset";
        scene_state.motion = RangeMotionMode::Stationary;
      }
      if (key == 's' || key == 'S') requestMotion(RangeMotionMode::Stationary);
      if (key == 'l' || key == 'L') requestMotion(RangeMotionMode::Linear);
      if (key == 'p' || key == 'P') requestMotion(RangeMotionMode::Spin);
      if (key == 'b' || key == 'B') requestMotion(RangeMotionMode::LinearAndSpin);
      if (key == '+' || key == '=') adjustSpeed(TRADITIONAL_MOTION_SPEED_STEP_MPS);
      if (key == '-' || key == '_') adjustSpeed(-TRADITIONAL_MOTION_SPEED_STEP_MPS);
    }
  }

  images.close();
  recorder.release();
  cv::destroyAllWindows();
  return 0;
}
