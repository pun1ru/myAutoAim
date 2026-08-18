#include <daedalus_sim_sdk/scene_control_client.hpp>
#include <daedalus_sim_sdk/tcp_image_client.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace daedalus::sim::sdk::v1;

struct DetectorParameters {
  int brightness_threshold = 180;
  int color_difference_threshold = 40;
  int morphology_size = 1;
  int minimum_contour_area = 8;
  int enemy_is_blue = 1;
};

struct LightBar {
  cv::RotatedRect rect;
  cv::Point2f top;
  cv::Point2f bottom;
  float length = 0.0F;
  float angle_deg = 0.0F;
};

struct DetectionResult {
  cv::Mat mask;
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

DetectionResult runTraditionalDetector(const cv::Mat& bgr,
                                       const DetectorParameters& parameters) {
  DetectionResult result;

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

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(result.mask, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  for (const std::vector<cv::Point>& contour : contours) {
    const double area = cv::contourArea(contour);
    if (area < static_cast<double>(parameters.minimum_contour_area)) {
      continue;
    }
    const cv::RotatedRect rect = cv::minAreaRect(contour);
    const float length = std::max(rect.size.width, rect.size.height);
    const float width = std::max(1.0F, std::min(rect.size.width, rect.size.height));
    if (length < 4.0F || length / width < 1.8F) {
      continue;
    }
    result.light_bars.push_back(makeLightBar(rect));
  }

  struct PairCandidate {
    int left = 0;
    int right = 0;
    float score = 0.0F;
  };
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
      if (length_ratio > 1.6F || angle_difference > 20.0F ||
          height_difference > 0.65F || separation < 0.6F || separation > 6.0F) {
        continue;
      }
      pairs.push_back({left, right, length_ratio + angle_difference / 20.0F +
                                       height_difference});
    }
  }

  std::sort(pairs.begin(), pairs.end(), [](const PairCandidate& lhs,
                                            const PairCandidate& rhs) {
    return lhs.score < rhs.score;
  });
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

bool setTargetMotion(SceneControlClient& scene, RangeMotionMode mode) {
  RangeTargetMotion motion;
  motion.target = 3;
  motion.mode = mode;
  motion.direction_deg = 90.0F;
  motion.linear_speed_mps = mode == RangeMotionMode::Stationary ? 0.0F : 1.5F;
  motion.linear_span_m = mode == RangeMotionMode::Stationary ? 0.0F : 4.0F;
  motion.spin_deg_s = mode == RangeMotionMode::LinearAndSpin ? 60.0F : 0.0F;
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

int main() {
  SceneControlOptions scene_options;
  scene_options.session_id = "first-traditional-detector";
  SceneControlClient scene(scene_options);
  if (!sceneSucceeded(scene.createSession(), "createSession") ||
      !sceneSucceeded(scene.setScene(SceneMode::ShootingRange), "setScene") ||
      !sceneSucceeded(scene.resetScene(), "resetScene") ||
      !setTargetMotion(scene, RangeMotionMode::Stationary)) {
    return 1;
  }

  TcpImageClient images;
  const ClientStatus connected = images.connect();
  if (!connected) {
    std::cerr << "connect image stream failed: " << connected.message << '\n';
    return 2;
  }

  DetectorParameters parameters;
  cv::namedWindow("controls", cv::WINDOW_NORMAL);
  cv::namedWindow("detector", cv::WINDOW_NORMAL);
  cv::namedWindow("mask", cv::WINDOW_NORMAL);
  cv::createTrackbar("brightness", "controls",
                     &parameters.brightness_threshold, 255);
  cv::createTrackbar("color difference", "controls",
                     &parameters.color_difference_threshold, 255);
  cv::createTrackbar("morphology radius", "controls",
                     &parameters.morphology_size, 10);
  cv::createTrackbar("minimum area", "controls",
                     &parameters.minimum_contour_area, 500);
  cv::createTrackbar("enemy: red / blue", "controls",
                     &parameters.enemy_is_blue, 1);

  std::uint64_t previous_sequence = 0;
  while (true) {
    auto frame = images.waitForLatest(previous_sequence,
                                      std::chrono::milliseconds(1000));
    if (!frame) {
      if (frame.status.error != ClientError::Timeout) {
        std::cerr << "waitForLatest failed: " << frame.status.message << '\n';
      }
      continue;
    }
    previous_sequence = frame.value->header.source_sequence;

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

    DetectionResult detection = runTraditionalDetector(bgr, parameters);
    cv::Mat display = bgr.clone();
    for (const LightBar& light_bar : detection.light_bars) {
      drawLightBar(display, light_bar);
    }
    for (const auto& corners : detection.armor_corners) {
      std::vector<cv::Point> polygon;
      polygon.reserve(corners.size());
      for (const cv::Point2f& corner : corners) {
        polygon.emplace_back(cvRound(corner.x), cvRound(corner.y));
        cv::circle(display, corner, 4, cv::Scalar(0, 255, 0), cv::FILLED,
                   cv::LINE_AA);
      }
      cv::polylines(display, polygon, true, cv::Scalar(0, 255, 0), 2,
                    cv::LINE_AA);
    }

    const std::string identity =
        "seq=" + std::to_string(source.header.source_sequence) +
        " timestamp_ns=" + std::to_string(source.header.capture_timestamp_ns);
    const std::string summary =
        "light bars=" + std::to_string(detection.light_bars.size()) +
        " armor pairs=" + std::to_string(detection.armor_corners.size());
    cv::putText(display, identity, {20, 35}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    cv::putText(display, summary, {20, 65}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

    cv::imshow("detector", display);
    cv::imshow("mask", detection.mask);

    const int key = cv::waitKey(1) & 0xff;
    if (key == 27 || key == 'q') break;
    if (key == '1') setTargetMotion(scene, RangeMotionMode::Stationary);
    if (key == '2') setTargetMotion(scene, RangeMotionMode::Linear);
    if (key == '3') setTargetMotion(scene, RangeMotionMode::LinearAndSpin);
  }

  images.close();
  return 0;
}
