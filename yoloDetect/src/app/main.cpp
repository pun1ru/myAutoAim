#include "control/gimbal_aim_solver.hpp"
#include "control/dynamic_target_controller.hpp"
#include "control/static_target_controller.hpp"
#include "coordinates/simulator_pose_adapter.hpp"
#include "detection/yolo_pose_detector.hpp"
#include "pose/armor_pose_estimator.hpp"
#include "tracking/tracker_measurement_adapter.hpp"
#include "tracking/whole_vehicle_ekf.hpp"
#include "web/mjpeg_server.hpp"

#include <daedalus_sim_sdk/scene_control_client.hpp>
#include <daedalus_sim_sdk/tcp_image_client.hpp>
#include <daedalus_sim_sdk/udp_gimbal_client.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace daedalus_sdk = daedalus::sim::sdk::v1;
namespace tcp_image = daedalus::sim::sdk::v1::tcp_image;
using Clock = std::chrono::steady_clock;

#ifndef YOLO_DETECT_DEFAULT_CUDA_LIB_DIR
#define YOLO_DETECT_DEFAULT_CUDA_LIB_DIR ""
#endif

namespace {

enum class ModelProfile {
  SzuSim,
  Robot0526,
  Robot0708,
  SpvisionBest2Sim,
  ArmorPose0815,
};

struct ModelProfileDefinition {
  const char* name;
  const char* filename;
  int input_size;
};

constexpr std::array<ModelProfileDefinition, 6> kModelProfiles = {{
    {"szu-sim", "szu_best2_sim_416.onnx", 640},
    {"robot-0526", "robot_armor_0526.onnx", 640},
    {"robot-0708", "robot_armor_0708.onnx", 640},
    {"spvision-best2-sim", "spvision_best2_sim.onnx", 640},
    {"armor-pose-0815", "armor_pose_0815_640.onnx", 640},
    {"armor-pose-0815-export", "armor_pose_0815_4pt_640_export.onnx", 640},
}};

struct Options {
  std::string host = "127.0.0.1";
  std::uint16_t port = daedalus_sdk::kTcpImagePort;
  std::uint16_t control_port = daedalus_sdk::kUdpSceneControlPort;
  std::uint16_t gimbal_port = daedalus_sdk::kUdpCommandPort;
  std::filesystem::path model;
  std::optional<ModelProfile> model_profile;
  bool model_explicit = false;
  bool input_size_explicit = false;
  float confidence = 0.70F;
  float keypoint_confidence = 0.25F;
  yolo_detect::ArmorSize armor_size = yolo_detect::ArmorSize::Small;
  float nms = 0.45F;
  int input_size = 640;
  int display_width = 1100;
  std::uint16_t web_port = 0;
  std::string web_bind = "127.0.0.1";
  int web_jpeg_quality = 65;
  std::filesystem::path ipc_directory;
  std::uint8_t target = 3;
  float vehicle_speed = 1.5F;
  float speed_step = 0.25F;
  float linear_span = 4.0F;
  float direction_deg = 90.0F;
  float spin_speed_deg_s = 60.0F;
  bool no_display = false;
  std::filesystem::path record_path;
  double record_fps = 60.0;
  std::string startup_scene = "unchanged";
  std::string scene_after = "unchanged";
  double scene_switch_after = 0.0;
  daedalus_sdk::RangeMotionMode startup_motion =
      daedalus_sdk::RangeMotionMode::Stationary;
  std::uint64_t max_frames = 0;
  bool use_cuda = true;
  int cuda_device = 0;
  std::filesystem::path cuda_library_directory =
      YOLO_DETECT_DEFAULT_CUDA_LIB_DIR;
};

struct SceneState {
  std::string scene = "unchanged";
  daedalus_sdk::RangeMotionMode motion =
      daedalus_sdk::RangeMotionMode::Stationary;
  std::uint8_t target = 3;
  float vehicle_speed = 1.5F;
  float linear_span = 4.0F;
  float direction_deg = 90.0F;
  float spin_speed_deg_s = 60.0F;
};

struct DetectionAim {
  bool coordinate_valid = false;
  cv::Vec3d center_odom_m{0.0, 0.0, 0.0};
  yolo_detect::control::GimbalAimResult aim;
};

struct ProjectionDebugState {
  bool enabled = false;
  bool anchor_observed = true;
  bool has_reference = false;
  double center_x_T_m = 0.0;
  double center_y_T_m = 0.0;
  double center_z_T_m = 0.0;
  double theta_rad = 0.0;
  double radius_even_m = 0.15;
  double radius_odd_delta_m = 0.0;
  double height_odd_delta_m = 0.0;
};

// 根据可执行文件位置推导随程序部署的默认模型路径。
std::filesystem::path defaultModelPath(const char* executable) {
  std::error_code error;
  std::filesystem::path path = std::filesystem::absolute(executable, error);
  if (error) path = executable;
  return path.parent_path() / "robot_armor_0526.onnx";
}

std::optional<ModelProfile> parseModelProfile(std::string_view name) {
  for (std::size_t index = 0; index < kModelProfiles.size(); ++index) {
    if (name == kModelProfiles[index].name) {
      return static_cast<ModelProfile>(index);
    }
  }
  return std::nullopt;
}

const ModelProfileDefinition& modelProfileDefinition(ModelProfile profile) {
  return kModelProfiles.at(static_cast<std::size_t>(profile));
}

std::filesystem::path modelProfilePath(const char* executable,
                                       ModelProfile profile) {
  std::error_code error;
  std::filesystem::path path = std::filesystem::absolute(executable, error);
  if (error) path = executable;
  return path.parent_path() / modelProfileDefinition(profile).filename;
}

// Windows development normally keeps yoloDetect and the simulator package in
// the same workspace. Resolve that package automatically, while preserving an
// explicit --ipc-dir or TALOS_IPC_DIR as the authoritative configuration.
std::filesystem::path defaultIpcDirectory(const char* executable) {
  if (const char* value = std::getenv("TALOS_IPC_DIR");
      value != nullptr && value[0] != '\0') {
    return value;
  }
#ifdef _WIN32
  std::error_code error;
  std::filesystem::path directory =
      std::filesystem::absolute(executable, error).parent_path();
  if (error) return {};
  while (!directory.empty()) {
    const std::filesystem::path candidate =
        directory / "1.1.1" / "runtime" / "talos-ipc";
    if (std::filesystem::is_directory(candidate, error) && !error) {
      return candidate;
    }
    const std::filesystem::path parent = directory.parent_path();
    if (parent == directory) break;
    directory = parent;
  }
#else
  static_cast<void>(executable);
#endif
  return {};
}

// 返回 Daedalus 1.3.1 固定分辨率对应的相机标定参数。
yolo_detect::CameraCalibration simulatorCameraCalibration() {
  yolo_detect::CameraCalibration calibration;
  calibration.camera_matrix = cv::Matx33d(
      1303.67532368147, 0.0, 720.0, 0.0, 1303.67532368147, 540.0, 0.0,
      0.0, 1.0);
  calibration.distortion_coefficients = {0.0, 0.0, 0.0, 0.0, 0.0};
  calibration.image_size = {1440, 1080};
  return calibration;
}

// 输出命令行参数、键盘控制和关键点顺序说明。
void printUsage() {
  std::cout
      << "Daedalus YOLO armor pose detector\n\n"
      << "Usage: yolo_detect [options]\n"
      << "  --model <path>       Custom ONNX model (default: robot_armor_0526)\n"
      << "  --model-profile <name> Built-in model: szu-sim, robot-0526, robot-0708,\n"
      << "                       spvision-best2-sim, armor-pose-0815, or armor-pose-0815-export\n"
      << "  --host <address>     SDK image host (default: 127.0.0.1)\n"
      << "  --port <port>        SDK image port (default: 5602)\n"
      << "  --control-port <port> Scene control UDP port (default: 5603)\n"
      << "  --gimbal-port <port> Gimbal/fire UDP port (default: 5601)\n"
      << "  --conf <0..1>        Object confidence (default: 0.70 for SZU model)\n"
      << "  --kpt-conf <0..1>    Point/PnP confidence (default: 0.25)\n"
      << "  --armor-size <name>  PnP plate model: small or large (default: small)\n"
      << "  --nms <0..1>         NMS IoU threshold (default: 0.45)\n"
      << "  --imgsz <pixels>     Square model input size (profile default if selected)\n"
      << "  --width <pixels>     Display width (default: 1100)\n"
      << "  --web <port>         Serve annotated MJPEG frames over HTTP\n"
      << "  --web-bind <address> Web bind address (default: 127.0.0.1)\n"
      << "  --web-quality <1..100> JPEG quality for --web (default: 65)\n"
      << "  --ipc-dir <path>     Talos IPC directory for synchronized poses\n"
      << "  --target <0..255>    Shooting-range vehicle ID (default: 3)\n"
      << "  --speed <m/s>        Initial vehicle speed (default: 1.5)\n"
      << "  --speed-step <m/s>   +/- adjustment step (default: 0.25)\n"
      << "  --no-display         Disable drawing and the visualization window\n"
      << "  --record <path>      Write annotated video; works with --no-display\n"
      << "  --record-fps <fps>   Recorded video playback FPS (default: 60)\n"
      << "  --scene <name>       Startup scene: unchanged, armor, energy, outpost, shooting-range\n"
      << "  --scene-after <name> Switch to this scene after --scene-switch-after seconds\n"
      << "  --scene-switch-after <s> Timed scene switch (default: disabled)\n"
      << "  --motion <name>      Startup motion: stationary, linear, spin, linear-spin\n"
      << "  --spin-speed <deg/s> Startup self-rotation speed (default: 60)\n"
      << "  --frames <count>     Stop after N received frames (default: unlimited)\n"
      << "  --device <index>     CUDA device index (default: 0)\n"
      << "  --cuda-libs <path>   CUDA/cuDNN runtime DLL directory\n"
      << "  --cpu                Explicitly use CPU instead of CUDA\n"
      << "  --help               Show this help\n\n"
      << "Keypoint order: 1 bottom-left, 2 top-left, 3 top-right, "
         "4 bottom-right.\n"
      << "Scene keys: 1 Armor, 2 Energy, 3 Outpost, 4 Shooting Range, "
         "0 Reset.\n"
      << "Vehicle keys: S Stop, L Linear, P Spin, B Linear+Spin, "
         "+/- Speed.\n"
      << "Press Q or Esc to close.\n";
}

// 读取当前命令行选项的必填参数值，并推进参数索引。
std::string requireValue(int& index, int argc, char** argv) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + argv[index]);
  }
  return argv[++index];
}

// 解析并校验位于闭区间 [0, 1] 的浮点选项。
float parseUnitFloat(const std::string& value, const char* name) {
  const float parsed = std::stof(value);
  if (parsed < 0.0F || parsed > 1.0F) {
    throw std::runtime_error(std::string(name) + " must be in [0, 1]");
  }
  return parsed;
}

bool hasPrefix(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

// 解析并校验全部命令行选项。
Options parseOptions(int argc, char** argv) {
  Options options;
  options.model = defaultModelPath(argv[0]);
  options.ipc_directory = defaultIpcDirectory(argv[0]);
  options.use_cuda = !options.cuda_library_directory.empty();
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      printUsage();
      std::exit(0);
    }
    if (argument == "--model") {
      options.model = requireValue(index, argc, argv);
      options.model_explicit = true;
    } else if (argument == "--model-profile") {
      const std::string value = requireValue(index, argc, argv);
      const std::optional<ModelProfile> profile = parseModelProfile(value);
      if (!profile) {
        throw std::runtime_error("unknown model-profile: " + value);
      }
      options.model_profile = *profile;
    } else if (argument == "--host") {
      options.host = requireValue(index, argc, argv);
    } else if (argument == "--port") {
      const int value = std::stoi(requireValue(index, argc, argv));
      if (value < 1 || value > 65535) {
        throw std::runtime_error("port must be in [1, 65535]");
      }
      options.port = static_cast<std::uint16_t>(value);
    } else if (argument == "--control-port") {
      const int value = std::stoi(requireValue(index, argc, argv));
      if (value < 1 || value > 65535) {
        throw std::runtime_error("control-port must be in [1, 65535]");
      }
      options.control_port = static_cast<std::uint16_t>(value);
    } else if (argument == "--gimbal-port") {
      const int value = std::stoi(requireValue(index, argc, argv));
      if (value < 1 || value > 65535) {
        throw std::runtime_error("gimbal-port must be in [1, 65535]");
      }
      options.gimbal_port = static_cast<std::uint16_t>(value);
    } else if (argument == "--conf") {
      options.confidence =
          parseUnitFloat(requireValue(index, argc, argv), "conf");
    } else if (argument == "--kpt-conf") {
      options.keypoint_confidence =
          parseUnitFloat(requireValue(index, argc, argv), "kpt-conf");
    } else if (argument == "--armor-size") {
      const std::string value = requireValue(index, argc, argv);
      if (value == "small") {
        options.armor_size = yolo_detect::ArmorSize::Small;
      } else if (value == "large") {
        options.armor_size = yolo_detect::ArmorSize::Large;
      } else {
        throw std::runtime_error("armor-size must be small or large");
      }
    } else if (argument == "--nms") {
      options.nms = parseUnitFloat(requireValue(index, argc, argv), "nms");
    } else if (argument == "--imgsz") {
      options.input_size = std::stoi(requireValue(index, argc, argv));
      options.input_size_explicit = true;
      if (options.input_size < 32 || options.input_size > 2048) {
        throw std::runtime_error("imgsz must be in [32, 2048]");
      }
    } else if (argument == "--width") {
      options.display_width = std::stoi(requireValue(index, argc, argv));
      if (options.display_width < 320 || options.display_width > 3840) {
        throw std::runtime_error("width must be in [320, 3840]");
      }
    } else if (argument == "--web") {
      const int value = std::stoi(requireValue(index, argc, argv));
      if (value < 1 || value > 65535) {
        throw std::runtime_error("web port must be in [1, 65535]");
      }
      options.web_port = static_cast<std::uint16_t>(value);
    } else if (argument == "--web-bind") {
      options.web_bind = requireValue(index, argc, argv);
      if (options.web_bind.empty()) {
        throw std::runtime_error("web-bind must not be empty");
      }
    } else if (argument == "--web-quality") {
      options.web_jpeg_quality = std::stoi(requireValue(index, argc, argv));
      if (options.web_jpeg_quality < 1 || options.web_jpeg_quality > 100) {
        throw std::runtime_error("web-quality must be in [1, 100]");
      }
    } else if (argument == "--ipc-dir") {
      options.ipc_directory = requireValue(index, argc, argv);
      if (options.ipc_directory.empty()) {
        throw std::runtime_error("ipc-dir must not be empty");
      }
    } else if (argument == "--target") {
      const int value = std::stoi(requireValue(index, argc, argv));
      if (value < 0 || value > 255) {
        throw std::runtime_error("target must be in [0, 255]");
      }
      options.target = static_cast<std::uint8_t>(value);
    } else if (argument == "--speed") {
      options.vehicle_speed = std::stof(requireValue(index, argc, argv));
      if (options.vehicle_speed < 0.0F || options.vehicle_speed > 20.0F) {
        throw std::runtime_error("speed must be in [0, 20] m/s");
      }
    } else if (argument == "--speed-step") {
      options.speed_step = std::stof(requireValue(index, argc, argv));
      if (options.speed_step <= 0.0F || options.speed_step > 5.0F) {
        throw std::runtime_error("speed-step must be in (0, 5] m/s");
      }
    } else if (argument == "--no-display") {
      options.no_display = true;
    } else if (argument == "--record") {
      options.record_path = requireValue(index, argc, argv);
      if (options.record_path.empty()) {
        throw std::runtime_error("record path must not be empty");
      }
    } else if (argument == "--record-fps") {
      options.record_fps = std::stod(requireValue(index, argc, argv));
      if (options.record_fps <= 0.0 || options.record_fps > 240.0) {
        throw std::runtime_error("record-fps must be in (0, 240]");
      }
    } else if (argument == "--scene") {
      options.startup_scene = requireValue(index, argc, argv);
      if (options.startup_scene != "unchanged" &&
          options.startup_scene != "armor" &&
          options.startup_scene != "energy" &&
          options.startup_scene != "outpost" &&
          options.startup_scene != "shooting-range") {
        throw std::runtime_error(
            "scene must be unchanged, armor, energy, outpost, or shooting-range");
      }
    } else if (argument == "--scene-after") {
      options.scene_after = requireValue(index, argc, argv);
      if (options.scene_after != "unchanged" &&
          options.scene_after != "armor" && options.scene_after != "energy" &&
          options.scene_after != "outpost" &&
          options.scene_after != "shooting-range") {
        throw std::runtime_error(
            "scene-after must be unchanged, armor, energy, outpost, or shooting-range");
      }
    } else if (argument == "--scene-switch-after") {
      options.scene_switch_after =
          std::stod(requireValue(index, argc, argv));
      if (options.scene_switch_after <= 0.0 ||
          options.scene_switch_after > 3600.0) {
        throw std::runtime_error(
            "scene-switch-after must be in (0, 3600] seconds");
      }
    } else if (argument == "--motion") {
      const std::string motion = requireValue(index, argc, argv);
      if (motion == "stationary") {
        options.startup_motion = daedalus_sdk::RangeMotionMode::Stationary;
      } else if (motion == "linear") {
        options.startup_motion = daedalus_sdk::RangeMotionMode::Linear;
      } else if (motion == "spin") {
        options.startup_motion = daedalus_sdk::RangeMotionMode::Spin;
      } else if (motion == "linear-spin") {
        options.startup_motion = daedalus_sdk::RangeMotionMode::LinearAndSpin;
      } else {
        throw std::runtime_error(
            "motion must be stationary, linear, spin, or linear-spin");
      }
    } else if (argument == "--spin-speed") {
      options.spin_speed_deg_s = std::stof(requireValue(index, argc, argv));
      if (options.spin_speed_deg_s < 0.0F ||
          options.spin_speed_deg_s > 720.0F) {
        throw std::runtime_error("spin-speed must be in [0, 720] deg/s");
      }
    } else if (argument == "--frames") {
      const long long value = std::stoll(requireValue(index, argc, argv));
      if (value <= 0) {
        throw std::runtime_error("frames must be greater than zero");
      }
      options.max_frames = static_cast<std::uint64_t>(value);
    } else if (argument == "--device") {
      options.cuda_device = std::stoi(requireValue(index, argc, argv));
      if (options.cuda_device < 0 || options.cuda_device > 15) {
        throw std::runtime_error("device must be in [0, 15]");
      }
    } else if (argument == "--cuda-libs") {
      options.cuda_library_directory = requireValue(index, argc, argv);
      options.use_cuda = true;
    } else if (argument == "--cpu") {
      options.use_cuda = false;
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (options.model_profile) {
    if (options.model_explicit) {
      throw std::runtime_error("--model and --model-profile cannot be used together");
    }
    const ModelProfileDefinition& profile =
        modelProfileDefinition(*options.model_profile);
    options.model = modelProfilePath(argv[0], *options.model_profile);
    if (!options.input_size_explicit) options.input_size = profile.input_size;
  }
  return options;
}

// 将射击场车辆运动模式转换为显示名称。
const char* motionName(daedalus_sdk::RangeMotionMode mode) {
  switch (mode) {
    case daedalus_sdk::RangeMotionMode::Stationary:
      return "stationary";
    case daedalus_sdk::RangeMotionMode::Linear:
      return "linear";
    case daedalus_sdk::RangeMotionMode::Spin:
      return "spin";
    case daedalus_sdk::RangeMotionMode::LinearAndSpin:
      return "linear+spin";
  }
  return "unknown";
}

// 将 SDK 场景控制调用结果转换为状态文本和成功标记。
bool reportControl(
    const daedalus_sdk::ClientResult<daedalus_sdk::SceneControlResponse>& result,
    std::string_view operation, std::string& message) {
  if (!result) {
    message = std::string(operation) + " failed: " + result.status.message;
    std::cerr << message << '\n';
    return false;
  }
  const auto& response = *result.value;
  if (response.status != daedalus_sdk::SceneControlStatus::Ok) {
    message = std::string(operation) + " rejected: " + response.message;
    std::cerr << message << '\n';
    return false;
  }
  message = std::string(operation) + " applied at frame " +
            std::to_string(response.applied_frame_seq);
  std::cout << message << '\n';
  return true;
}

// 按当前场景状态向 SDK 下发目标车辆运动设置。
bool applyMotion(daedalus_sdk::SceneControlClient& client,
                 const SceneState& state, std::string& message) {
  daedalus_sdk::RangeTargetMotion motion;
  motion.target = state.target;
  motion.mode = state.motion;
  motion.direction_deg = state.direction_deg;
  const bool has_linear =
      state.motion == daedalus_sdk::RangeMotionMode::Linear ||
      state.motion == daedalus_sdk::RangeMotionMode::LinearAndSpin;
  const bool has_spin = state.motion == daedalus_sdk::RangeMotionMode::Spin ||
                        state.motion ==
                            daedalus_sdk::RangeMotionMode::LinearAndSpin;
  motion.linear_speed_mps = has_linear ? state.vehicle_speed : 0.0F;
  motion.linear_span_m = has_linear ? state.linear_span : 0.0F;
  motion.spin_deg_s = has_spin ? state.spin_speed_deg_s : 0.0F;
  return reportControl(client.setRangeTargetMotion(motion),
                       "setRangeTargetMotion", message);
}

// 按指定小数位数格式化遥测数值。
std::string fixed(double value, int precision = 1) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

// 在调试图像的指定状态行绘制文本。
void drawText(cv::Mat& image, const std::string& text, int line,
              const cv::Scalar& color = {235, 235, 235}) {
  const cv::Point origin(16, 28 + line * 25);
  cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.58,
              cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
  cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.58, color, 1,
              cv::LINE_AA);
}

// 绘制检测框、关键点、PnP 位姿和可用的瞄准信息。
void drawDetection(cv::Mat& image,
                   const yolo_detect::ArmorDetection& detection,
                   const yolo_detect::PoseResult& pose,
                   const DetectionAim& detection_aim,
                   const yolo_detect::CameraCalibration& calibration,
                   float keypoint_threshold) {
  const cv::Rect box(cvRound(detection.box.x), cvRound(detection.box.y),
                     cvRound(detection.box.width),
                     cvRound(detection.box.height));
  cv::rectangle(image, box, cv::Scalar(255, 120, 30), 1, cv::LINE_AA);

  const std::array<cv::Scalar, yolo_detect::KeypointCount> colors = {
      cv::Scalar(0, 255, 255),    // bottom-left
      cv::Scalar(0, 255, 0),      // top-left
      cv::Scalar(255, 255, 0),    // top-right
      cv::Scalar(255, 0, 255),    // bottom-right
  };

  std::vector<cv::Point> polygon;
  polygon.reserve(yolo_detect::KeypointCount);
  bool all_visible = true;
  for (std::size_t index = 0; index < yolo_detect::KeypointCount; ++index) {
    const cv::Point point(cvRound(detection.keypoints[index].x),
                          cvRound(detection.keypoints[index].y));
    polygon.push_back(point);
    if (detection.keypoint_confidences[index] < keypoint_threshold) {
      all_visible = false;
      continue;
    }
    cv::circle(image, point, 3, colors[index], cv::FILLED, cv::LINE_AA);
    cv::putText(image, std::to_string(index + 1), point + cv::Point(6, -6),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[index], 2,
                cv::LINE_AA);
  }
  if (all_visible) {
    cv::polylines(image, polygon, true, cv::Scalar(30, 210, 255), 1,
                  cv::LINE_AA);
  }

  // RobotDetectionModel/0526.onnx uses this class order: 0=blue, 1=red,
  // 2=gray, 3=purple. Keep the model's IDs unchanged and only map them to
  // display names here.
  const std::array<const char*, 4> color_names = {"blue", "red", "gray",
                                                   "purple"};
  const std::array<const char*, 9> number_names = {
      "G", "1", "2", "3", "4", "5", "O", "Bs", "Bb"};
  std::string label = "armor " + fixed(detection.confidence, 3);
  if (detection.color_id >= 0 &&
      detection.color_id < static_cast<int>(color_names.size())) {
    label += " " + std::string(color_names[detection.color_id]);
  }
  if (detection.number_id >= 0 &&
      detection.number_id < static_cast<int>(number_names.size())) {
    label += " " + std::string(number_names[detection.number_id]);
  }
  const cv::Point label_origin(box.x, std::max(20, box.y - 7));
  cv::putText(image, label, label_origin, cv::FONT_HERSHEY_SIMPLEX, 0.55,
              cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
  cv::putText(image, label, label_origin, cv::FONT_HERSHEY_SIMPLEX, 0.55,
              cv::Scalar(255, 180, 30), 1, cv::LINE_AA);

  if (pose.valid) {
    cv::drawFrameAxes(image, calibration.camera_matrix,
                      calibration.distortion_coefficients, pose.rvec_rad,
                      pose.tvec_m, 0.06F, 2);
    const std::string pose_label =
        "C[m] x=" + fixed(pose.center_camera_m[0], 3) +
        " y=" + fixed(pose.center_camera_m[1], 3) +
        " z=" + fixed(pose.center_camera_m[2], 3) +
        " rms=" + fixed(pose.reprojection_rms_px, 2) +
        " n=" + std::to_string(pose.candidate_count);
    const int pose_y =
        std::min(image.rows - 8, std::max(20, box.y + box.height + 20));
    const cv::Point pose_origin(std::max(0, box.x), pose_y);
    cv::putText(image, pose_label, pose_origin, cv::FONT_HERSHEY_SIMPLEX,
                0.48, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
    cv::putText(image, pose_label, pose_origin, cv::FONT_HERSHEY_SIMPLEX,
                0.48, cv::Scalar(90, 255, 150), 1, cv::LINE_AA);
    if (detection_aim.coordinate_valid) {
      std::string aim_label =
          "O[m] x=" + fixed(detection_aim.center_odom_m[0], 3) +
          " y=" + fixed(detection_aim.center_odom_m[1], 3) +
          " z=" + fixed(detection_aim.center_odom_m[2], 3);
      if (detection_aim.aim.valid) {
        aim_label +=
            " aim yaw=" + fixed(detection_aim.aim.yaw_command_deg, 2) +
            " pitch=" + fixed(detection_aim.aim.pitch_command_deg, 2) +
            " tof=" + fixed(detection_aim.aim.time_of_flight_s, 3);
      } else {
        aim_label += " aim=" +
            std::string(yolo_detect::control::aimStatusName(
                detection_aim.aim.status));
      }
      const cv::Point aim_origin(
          pose_origin.x, std::min(image.rows - 8, pose_origin.y + 19));
      cv::putText(image, aim_label, aim_origin, cv::FONT_HERSHEY_SIMPLEX,
                  0.45, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
      cv::putText(image, aim_label, aim_origin, cv::FONT_HERSHEY_SIMPLEX,
                  0.45, cv::Scalar(80, 220, 255), 1, cv::LINE_AA);
    }
  }
}

std::optional<cv::Point> projectTrackerPoint(
    const Eigen::Vector3d& point_tracker_m, const cv::Size image_size,
    const yolo_detect::coordinates::CoordinateSnapshot& exposure_snapshot,
    const yolo_detect::CameraCalibration& calibration) {
  if (!exposure_snapshot.valid || !point_tracker_m.allFinite()) {
    return std::nullopt;
  }
  const cv::Matx33d R_CO =
      yolo_detect::coordinates::cameraRotationOdom(exposure_snapshot).t();
  const cv::Vec3d point_tracker(point_tracker_m.x(), point_tracker_m.y(),
                                point_tracker_m.z());
  const cv::Vec3d point_camera =
      R_CO * (point_tracker - exposure_snapshot.camera_position_odom_m);
  if (!std::isfinite(point_camera[0]) || !std::isfinite(point_camera[1]) ||
      !std::isfinite(point_camera[2]) || point_camera[2] <= 0.0) {
    return std::nullopt;
  }
  std::vector<cv::Point2d> projected;
  cv::projectPoints(std::vector<cv::Point3d>{
                        {point_camera[0], point_camera[1], point_camera[2]}},
                    cv::Vec3d(0.0, 0.0, 0.0), cv::Vec3d(0.0, 0.0, 0.0),
                    calibration.camera_matrix,
                    calibration.distortion_coefficients, projected);
  if (projected.size() != 1 || !std::isfinite(projected[0].x) ||
      !std::isfinite(projected[0].y) || projected[0].x < 0.0 ||
      projected[0].y < 0.0 || projected[0].x >= image_size.width ||
      projected[0].y >= image_size.height) {
    return std::nullopt;
  }
  return cv::Point(cvRound(projected[0].x), cvRound(projected[0].y));
}

// Projects whole-vehicle EKF armor-center predictions using the same exposure
// snapshot that produced the current image frame.
void drawPredictedArmorCenters(
    cv::Mat& image, const yolo_detect::tracking::TrackOutput& tracker_output,
    const yolo_detect::coordinates::CoordinateSnapshot& exposure_snapshot,
    const yolo_detect::CameraCalibration& calibration) {
  if (!tracker_output.has_state || !exposure_snapshot.valid) return;
  for (const yolo_detect::tracking::DecodedArmor& armor :
       tracker_output.predicted_armors) {
    const std::optional<cv::Point> projected = projectTrackerPoint(
        armor.position_T_m, image.size(), exposure_snapshot, calibration);
    if (!projected) continue;
    const cv::Point center = *projected;
    cv::circle(image, center, 7, cv::Scalar(0, 0, 0), cv::FILLED, cv::LINE_AA);
    cv::circle(image, center, 5, cv::Scalar(0, 255, 255), cv::FILLED,
               cv::LINE_AA);
    cv::putText(image, "E" + std::to_string(armor.armor_slot),
                center + cv::Point(7, -7), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
  }
}

void drawProjectionDebugCenters(
    cv::Mat& image, const ProjectionDebugState& debug_state,
    const std::optional<yolo_detect::tracking::Measurement>& reference,
    const yolo_detect::coordinates::CoordinateSnapshot& exposure_snapshot,
    const yolo_detect::CameraCalibration& calibration) {
  if (!debug_state.enabled || !exposure_snapshot.valid) return;
  yolo_detect::tracking::State state;
  state.x[yolo_detect::tracking::CenterX] = debug_state.center_x_T_m;
  state.x[yolo_detect::tracking::CenterY] = debug_state.center_y_T_m;
  state.x[yolo_detect::tracking::CenterZ] = debug_state.center_z_T_m;
  state.x[yolo_detect::tracking::Theta] = debug_state.theta_rad;
  state.x[yolo_detect::tracking::RadiusEven] = debug_state.radius_even_m;
  state.x[yolo_detect::tracking::RadiusOddDelta] =
      debug_state.radius_odd_delta_m;
  state.x[yolo_detect::tracking::HeightOddDelta] =
      debug_state.height_odd_delta_m;
  for (int slot = 0; slot < yolo_detect::tracking::kArmorSlotCount; ++slot) {
    const auto armor = yolo_detect::tracking::decodeArmor(state, slot, 0.0);
    const std::optional<cv::Point> projected = projectTrackerPoint(
        armor.position_T_m, image.size(), exposure_snapshot, calibration);
    if (!projected) continue;
    const cv::Point center = *projected;
    cv::circle(image, center, 8, cv::Scalar(0, 0, 0), cv::FILLED,
               cv::LINE_AA);
    cv::circle(image, center, 6, cv::Scalar(0, 165, 255), cv::FILLED,
               cv::LINE_AA);
    cv::putText(image, "D" + std::to_string(slot), center + cv::Point(8, -8),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 165, 255), 2,
                cv::LINE_AA);
  }
  if (reference && reference->position_T_m.allFinite()) {
    const auto projected = projectTrackerPoint(
        reference->position_T_m, image.size(), exposure_snapshot, calibration);
    if (projected) {
      cv::drawMarker(image, *projected, cv::Scalar(255, 255, 255),
                     cv::MARKER_CROSS, 18, 2, cv::LINE_AA);
      cv::putText(image, "PnP ref", *projected + cv::Point(8, 18),
                  cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255),
                  2, cv::LINE_AA);
    }
  }
}

// Select the armor whose outward normal most directly faces the current
// camera, then solve the target position at the self-consistent flight time.
std::optional<yolo_detect::control::AimTarget> makeDynamicAimTarget(
    const yolo_detect::tracking::TrackOutput& tracker_output,
    const yolo_detect::coordinates::CoordinateSnapshot& snapshot,
    const yolo_detect::control::GimbalAimSolver& aim_solver) {
  if (!tracker_output.has_state || !snapshot.valid) return std::nullopt;

  double horizon_s = 0.0;
  std::optional<yolo_detect::control::AimTarget> selected;
  for (int iteration = 0; iteration < 10; ++iteration) {
    double best_facing = -std::numeric_limits<double>::infinity();
    std::optional<yolo_detect::tracking::DecodedArmor> best_armor;
    for (int slot = 0; slot < yolo_detect::tracking::kArmorSlotCount; ++slot) {
      const yolo_detect::tracking::DecodedArmor armor =
          yolo_detect::tracking::decodeArmor(tracker_output.state, slot,
                                              horizon_s);
      const Eigen::Vector3d camera_to_armor =
          armor.position_T_m - Eigen::Vector3d(
                                    snapshot.camera_position_odom_m[0],
                                    snapshot.camera_position_odom_m[1],
                                    snapshot.camera_position_odom_m[2]);
      const double range = camera_to_armor.norm();
      if (!armor.position_T_m.allFinite() || !std::isfinite(range) ||
          range <= 1e-6) {
        continue;
      }
      const Eigen::Vector3d outward_normal(
          -std::cos(armor.inward_yaw_T_rad),
          -std::sin(armor.inward_yaw_T_rad), 0.0);
      const double facing = outward_normal.dot(-camera_to_armor / range);
      if (std::isfinite(facing) && facing > best_facing) {
        best_facing = facing;
        best_armor = armor;
      }
    }
    if (!best_armor) return std::nullopt;
    selected = yolo_detect::control::AimTarget{
        {best_armor->position_T_m.x(), best_armor->position_T_m.y(),
         best_armor->position_T_m.z()},
        true, horizon_s};
    const yolo_detect::control::GimbalAimResult trial =
        aim_solver.solve(*selected, snapshot);
    if (!trial.valid || !std::isfinite(trial.time_of_flight_s)) {
      return std::nullopt;
    }
    if (std::abs(trial.time_of_flight_s - horizon_s) < 1e-3) {
      return selected;
    }
    horizon_s = trial.time_of_flight_s;
  }
  return selected;
}

// 将本帧的位姿、坐标和控制器状态组装为网页遥测。
yolo_detect::WebFrameTelemetry makeWebTelemetry(
    std::uint64_t source_sequence,
    const std::vector<yolo_detect::PoseResult>& poses,
    const std::vector<DetectionAim>& detection_aims,
    const yolo_detect::coordinates::CoordinateSnapshot& coordinate_snapshot,
    const std::string& coordinate_message,
    const yolo_detect::control::DynamicTargetController& gimbal_control,
    const SceneState& scene_state,
    const yolo_detect::tracking::TrackOutput& tracker_output,
    const yolo_detect::tracking::WholeVehicleEkfOptions& ekf_options,
    const yolo_detect::tracking::ConstrainedYawOptions& yaw_options,
    const ProjectionDebugState& projection_debug,
    const std::vector<yolo_detect::tracking::ReliableYaw>& reliable_yaws,
    const std::vector<yolo_detect::tracking::Measurement>& tracker_measurements) {
  yolo_detect::WebFrameTelemetry telemetry;
  telemetry.source_sequence = source_sequence;
  telemetry.scene = scene_state.scene;
  telemetry.motion = motionName(scene_state.motion);
  telemetry.vehicle_speed_mps = scene_state.vehicle_speed;
  telemetry.spin_speed_deg_s = scene_state.spin_speed_deg_s;
  auto& tuning = telemetry.ekf_tuning;
  tuning.initial_position_std_m = ekf_options.initial_position_std_m;
  tuning.initial_velocity_std_mps = ekf_options.initial_velocity_std_mps;
  tuning.initial_theta_std_rad = ekf_options.initial_theta_std_rad;
  tuning.initial_omega_std_rad_s = ekf_options.initial_omega_std_rad_s;
  tuning.initial_geometry_std_m = ekf_options.initial_geometry_std_m;
  tuning.q_linear_acceleration = ekf_options.q_linear_acceleration;
  tuning.q_angular_acceleration = ekf_options.q_angular_acceleration;
  tuning.q_geometry = ekf_options.q_geometry;
  tuning.position_std_x_m = ekf_options.position_std_x_m;
  tuning.position_std_y_m = ekf_options.position_std_y_m;
  tuning.position_std_z_m = ekf_options.position_std_z_m;
  tuning.yaw_facing_base_variance_rad2 =
      ekf_options.yaw_facing_base_variance_rad2;
  tuning.yaw_facing_log_variance_scale_rad2 =
      ekf_options.yaw_facing_log_variance_scale_rad2;
  tuning.single_armor_position_variance_scale =
      ekf_options.single_armor_position_variance_scale;
  tuning.association_position_variance_scale =
      ekf_options.association_position_variance_scale;
  tuning.maximum_multi_armor_position_residual_m =
      ekf_options.maximum_multi_armor_position_residual_m;
  tuning.maximum_yaw_update_innovation_rad =
      ekf_options.maximum_yaw_update_innovation_rad;
  tuning.maximum_yaw_association_innovation_rad =
      ekf_options.maximum_yaw_association_innovation_rad;
  tuning.yaw_phase_cost_std_rad = ekf_options.yaw_phase_cost_std_rad;
  tuning.adjacent_slot_penalty = ekf_options.adjacent_slot_penalty;
  tuning.opposite_slot_penalty = ekf_options.opposite_slot_penalty;
  tuning.minimum_visibility_cosine = ekf_options.minimum_visibility_cosine;
  tuning.slot_position_cost_weight = ekf_options.slot_position_cost_weight;
  tuning.slot_yaw_cost_weight_m_per_rad =
      ekf_options.slot_yaw_cost_weight_m_per_rad;
  tuning.geometry_yaw_consistency_rad = ekf_options.geometry_yaw_consistency_rad;
  tuning.geometry_minimum_baseline_m = ekf_options.geometry_minimum_baseline_m;
  tuning.geometry_confirming_frames = ekf_options.geometry_confirming_frames;
  tuning.nis_gate_3d = ekf_options.nis_gate_3d;
  tuning.nis_gate_4d = ekf_options.nis_gate_4d;
  tuning.maximum_angular_speed_rad_s = ekf_options.maximum_angular_speed_rad_s;
  tuning.maximum_omega_correction_rad_s =
      ekf_options.maximum_omega_correction_rad_s;
  tuning.yaw_max_reprojection_rms_px = yaw_options.max_reprojection_rms_px;
  tuning.yaw_max_std_rad = yaw_options.max_yaw_std_rad;
  tuning.yaw_min_facing_cosine = yaw_options.min_facing_cosine;
  tuning.yaw_min_opposite_margin_px = yaw_options.min_opposite_margin_px;
  telemetry.projection_debug.enabled = projection_debug.enabled;
  telemetry.projection_debug.anchor_observed = projection_debug.anchor_observed;
  telemetry.projection_debug.has_reference = projection_debug.has_reference;
  telemetry.projection_debug.center_x_T_m = projection_debug.center_x_T_m;
  telemetry.projection_debug.center_y_T_m = projection_debug.center_y_T_m;
  telemetry.projection_debug.center_z_T_m = projection_debug.center_z_T_m;
  telemetry.projection_debug.theta_rad = projection_debug.theta_rad;
  telemetry.projection_debug.radius_even_m = projection_debug.radius_even_m;
  telemetry.projection_debug.radius_odd_delta_m =
      projection_debug.radius_odd_delta_m;
  telemetry.projection_debug.height_odd_delta_m =
      projection_debug.height_odd_delta_m;
  telemetry.tracker_state = yolo_detect::tracking::trackingStateName(
      tracker_output.tracking_state);
  telemetry.tracker_has_state = tracker_output.has_state;
  telemetry.tracker_observation_count = tracker_measurements.size();
  telemetry.reliable_yaw_count = static_cast<std::size_t>(std::count_if(
      reliable_yaws.begin(), reliable_yaws.end(),
      [](const yolo_detect::tracking::ReliableYaw& yaw) { return yaw.valid; }));
  if (!reliable_yaws.empty()) {
    const yolo_detect::tracking::ReliableYaw& yaw = reliable_yaws.front();
    telemetry.tracker_yaw_status =
        yolo_detect::tracking::reliableYawStatusName(yaw.status);
    telemetry.tracker_yaw_diagnostic_valid =
        std::isfinite(yaw.reprojection_rms_px);
    telemetry.tracker_yaw_rms_px = yaw.reprojection_rms_px;
  }
  if (tracker_output.has_state) {
    telemetry.tracker_center_x_T_m = tracker_output.center_T_m.x();
    telemetry.tracker_center_y_T_m = tracker_output.center_T_m.y();
    telemetry.tracker_center_z_T_m = tracker_output.center_T_m.z();
    telemetry.tracker_velocity_x_T_mps = tracker_output.velocity_T_mps.x();
    telemetry.tracker_velocity_y_T_mps = tracker_output.velocity_T_mps.y();
    telemetry.tracker_velocity_z_T_mps = tracker_output.velocity_T_mps.z();
    telemetry.tracker_theta_rad = tracker_output.theta_rad;
    telemetry.tracker_omega_rad_s = tracker_output.omega_rad_s;
    telemetry.tracker_radius_even_m = tracker_output.radius_even_m;
    telemetry.tracker_radius_odd_delta_m = tracker_output.radius_odd_delta_m;
    telemetry.tracker_radius_odd_m = tracker_output.radius_even_m +
                                     tracker_output.radius_odd_delta_m;
    telemetry.tracker_height_odd_delta_m = tracker_output.height_odd_delta_m;
    telemetry.tracker_consecutive_hits = tracker_output.consecutive_hits;
    telemetry.tracker_consecutive_misses = tracker_output.consecutive_misses;
    if (tracker_output.associated_slot) {
      telemetry.tracker_association_valid = true;
      telemetry.tracker_associated_slot = *tracker_output.associated_slot;
    }
    if (tracker_output.nis && std::isfinite(*tracker_output.nis)) {
      telemetry.tracker_nis_valid = true;
      telemetry.tracker_nis = *tracker_output.nis;
    }
    telemetry.predicted_armors.reserve(tracker_output.predicted_armors.size());
    for (const yolo_detect::tracking::DecodedArmor& armor :
         tracker_output.predicted_armors) {
      const Eigen::Vector3d& position = armor.position_T_m;
      if (!position.array().isFinite().all()) continue;
      telemetry.predicted_armors.push_back(
          {armor.armor_slot, position.x(), position.y(), position.z()});
    }
  }
  telemetry.tracker_measurements.reserve(tracker_measurements.size());
  for (std::size_t measurement_index = 0;
       measurement_index < tracker_measurements.size(); ++measurement_index) {
    const yolo_detect::tracking::Measurement& measurement =
        tracker_measurements[measurement_index];
    const auto association = std::find_if(
        tracker_output.associated_observations.begin(),
        tracker_output.associated_observations.end(),
        [measurement_index](const yolo_detect::tracking::AssociatedObservation& item) {
          return item.measurement_index == static_cast<int>(measurement_index);
        });
    const bool associated = association !=
                            tracker_output.associated_observations.end();
    telemetry.tracker_measurements.push_back(
        {measurement.timestamp_ns,
         measurement.position_T_m.x(), measurement.position_T_m.y(),
         measurement.position_T_m.z(), measurement.camera_range_m,
         measurement.has_exposure_camera_geometry,
         measurement.camera_position_T_m.x(), measurement.camera_position_T_m.y(),
         measurement.has_inward_yaw, measurement.has_raw_inward_yaw,
         measurement.has_pnp_inward_pitch_T, measurement.inward_yaw_T_rad,
         measurement.pnp_inward_pitch_T_rad,
         measurement.has_ippe_inward_yaw_0_T,
         measurement.ippe_inward_yaw_0_T_rad,
         measurement.has_ippe_inward_yaw_1_T,
         measurement.ippe_inward_yaw_1_T_rad,
         measurement.has_raw_inward_yaw, measurement.inward_yaw_T_rad,
         measurement.yaw_std_rad, measurement.reprojection_rms_px,
         measurement.confidence, measurement.keypoint_quality,
         measurement.view_quality, measurement.color_id, measurement.number_id,
         associated, measurement.selected_for_ekf,
         associated ? association->armor_slot : -1,
         associated ? association->nis : 0.0,
         associated ? association->predicted_position_T_m.x() : 0.0,
         associated ? association->predicted_position_T_m.y() : 0.0,
         associated ? association->predicted_position_T_m.z() : 0.0,
         associated ? association->position_innovation_T_m.x() : 0.0,
         associated ? association->position_innovation_T_m.y() : 0.0,
         associated ? association->position_innovation_T_m.z() : 0.0,
         associated ? association->radial_innovation_m : 0.0,
         associated ? association->predicted_inward_yaw_T_rad : 0.0,
         associated ? association->yaw_innovation_rad : 0.0});
  }
  telemetry.coordinate_valid = coordinate_snapshot.valid;
  telemetry.coordinate_status = coordinate_message;
  telemetry.camera_position_error_m =
      coordinate_snapshot.camera_position_error_m;
  telemetry.actual_yaw_deg =
      coordinate_snapshot.gimbal_yaw_rad * 180.0 / CV_PI;
  telemetry.actual_pitch_command_deg =
      90.0 + coordinate_snapshot.gimbal_elevation_rad * 180.0 / CV_PI;
  telemetry.gimbal_following = gimbal_control.following();
  telemetry.fire_pending = gimbal_control.firePending();
  telemetry.gimbal_status = gimbal_control.status();
  telemetry.last_gimbal_command_id = gimbal_control.lastCommandId();
  telemetry.last_command_fired = gimbal_control.lastCommandFired();
  telemetry.static_target_valid =
      gimbal_control.activeTargetOdomM().has_value();
  if (gimbal_control.activeTargetOdomM()) {
    telemetry.static_target_odom_x_m =
        (*gimbal_control.activeTargetOdomM())[0];
    telemetry.static_target_odom_y_m =
        (*gimbal_control.activeTargetOdomM())[1];
    telemetry.static_target_odom_z_m =
        (*gimbal_control.activeTargetOdomM())[2];
  }
  telemetry.poses.reserve(poses.size());
  for (std::size_t index = 0; index < poses.size(); ++index) {
    const yolo_detect::PoseResult& pose = poses[index];
    const DetectionAim* detection_aim =
        index < detection_aims.size() ? &detection_aims[index] : nullptr;
    yolo_detect::WebPoseTelemetry item;
    item.detection_index = index;
    item.valid = pose.valid;
    item.armor_size = yolo_detect::armorSizeName(pose.armor_size);
    item.status = yolo_detect::poseStatusName(pose.status);
    item.x_m = pose.center_camera_m[0];
    item.y_m = pose.center_camera_m[1];
    item.z_m = pose.center_camera_m[2];
    item.reprojection_rms_px = pose.reprojection_rms_px;
    item.candidate_count = pose.candidate_count;
    item.coordinate_valid =
        detection_aim != nullptr && detection_aim->coordinate_valid;
    item.coordinate_status =
        item.coordinate_valid
            ? "success"
            : (pose.valid ? coordinate_message : "PnP pose invalid");
    if (detection_aim != nullptr) {
      item.odom_x_m = detection_aim->center_odom_m[0];
      item.odom_y_m = detection_aim->center_odom_m[1];
      item.odom_z_m = detection_aim->center_odom_m[2];
      item.aim_valid = detection_aim->aim.valid;
      item.aim_status =
          yolo_detect::control::aimStatusName(detection_aim->aim.status);
      item.ballistic_status =
          yolo_detect::ballistics::ballisticStatusName(
              detection_aim->aim.ballistic_status);
      item.yaw_command_deg = detection_aim->aim.yaw_command_deg;
      item.pitch_command_deg = detection_aim->aim.pitch_command_deg;
      item.time_of_flight_s = detection_aim->aim.time_of_flight_s;
      item.gravity_drop_m = detection_aim->aim.gravity_drop_m;
      item.muzzle_odom_x_m = detection_aim->aim.muzzle_center_odom_m[0];
      item.muzzle_odom_y_m = detection_aim->aim.muzzle_center_odom_m[1];
      item.muzzle_odom_z_m = detection_aim->aim.muzzle_center_odom_m[2];
      item.predicted = detection_aim->aim.predicted;
      item.prediction_horizon_s =
          detection_aim->aim.prediction_horizon_s;
    } else {
      item.aim_status = "not evaluated";
      item.ballistic_status = "not evaluated";
    }
    telemetry.poses.push_back(std::move(item));
  }
  return telemetry;
}

}  // namespace

// 初始化 SDK、检测与控制模块，并运行主图像处理循环。
int main(int argc, char** argv) {
  Options options;
  try {
    options = parseOptions(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "argument error: " << error.what() << '\n';
    printUsage();
    return 2;
  }

  yolo_detect::DetectorConfig detector_config;
  detector_config.input_width = options.input_size;
  detector_config.input_height = options.input_size;
  detector_config.confidence_threshold = options.confidence;
  detector_config.nms_threshold = options.nms;
  detector_config.use_cuda = options.use_cuda;
  detector_config.cuda_device = options.cuda_device;
  detector_config.cuda_library_directory = options.cuda_library_directory;

  std::cout << "loading model: " << options.model.string() << '\n'
            << "backend=" << (options.use_cuda ? "cuda" : "cpu");
  if (options.use_cuda) {
    std::cout << " device=" << options.cuda_device
              << " cuda_libs=" << options.cuda_library_directory.string();
  }
  std::cout << '\n';
  std::unique_ptr<yolo_detect::YoloPoseDetector> detector;
  try {
    detector = std::make_unique<yolo_detect::YoloPoseDetector>(
        options.model, detector_config);
  } catch (const std::exception& error) {
    std::cerr << "model load failed: " << error.what() << '\n';
    return 3;
  }

  const yolo_detect::CameraCalibration camera_calibration =
      simulatorCameraCalibration();
  const yolo_detect::ArmorPoseEstimator pose_estimator(camera_calibration);
  yolo_detect::tracking::ConstrainedYawSolver constrained_yaw_solver(
      camera_calibration);
  const yolo_detect::control::GimbalAimSolver aim_solver;
  yolo_detect::tracking::WholeVehicleEkf whole_vehicle_tracker;
  ProjectionDebugState projection_debug;
  yolo_detect::coordinates::SimulatorPoseAdapter simulator_pose;
  std::string coordinate_message =
      "coordinate transform disabled: provide --ipc-dir";
  if (!options.ipc_directory.empty()) {
    if (simulator_pose.open(options.ipc_directory)) {
      coordinate_message = "success";
      const cv::Vec3d& camera_offset =
          simulator_pose.cameraOffsetGimbalM();
      const cv::Vec3d& muzzle_offset =
          simulator_pose.muzzleOffsetGimbalM();
      std::cout << "coordinate adapter ready: camera_G=("
                << camera_offset << ") m muzzle_G=(" << muzzle_offset
                << ") m\n";
    } else {
      coordinate_message = simulator_pose.lastError();
      std::cerr << coordinate_message << '\n';
    }
  }

  SceneState scene_state;
  scene_state.target = options.target;
  scene_state.vehicle_speed = options.vehicle_speed;
  scene_state.linear_span = options.linear_span;
  scene_state.direction_deg = options.direction_deg;
  scene_state.spin_speed_deg_s = options.spin_speed_deg_s;

  daedalus_sdk::SceneControlOptions scene_options;
  scene_options.endpoint = {options.host, options.control_port};
  scene_options.session_id = "yolo-detect";
  auto scene =
      std::make_unique<daedalus_sdk::SceneControlClient>(scene_options);
  std::string control_message;
  bool control_available = reportControl(
      scene->createSession(), "createSession", control_message);
  bool control_ok = control_available;
  daedalus_sdk::UdpGimbalClient gimbal(
      {options.host, options.gimbal_port});
  yolo_detect::control::DynamicTargetController gimbal_control;

  if (control_available && options.startup_scene != "unchanged") {
    daedalus_sdk::SceneMode startup_scene;
    if (options.startup_scene == "armor") {
      startup_scene = daedalus_sdk::SceneMode::Armor;
    } else if (options.startup_scene == "energy") {
      startup_scene = daedalus_sdk::SceneMode::Energy;
    } else if (options.startup_scene == "outpost") {
      startup_scene = daedalus_sdk::SceneMode::Outpost;
    } else {
      startup_scene = daedalus_sdk::SceneMode::ShootingRange;
    }
    control_ok = reportControl(
        scene->setScene(startup_scene),
        std::string("setScene ") + options.startup_scene, control_message);
    if (control_ok) scene_state.scene = options.startup_scene;
  }
  if (control_ok && options.startup_scene == "shooting-range") {
    scene_state.motion = options.startup_motion;
    control_ok = applyMotion(*scene, scene_state, control_message);
  }

  daedalus_sdk::TcpImageClient images({options.host, options.port});
  const daedalus_sdk::ClientStatus connected = images.connect();
  if (!connected) {
    std::cerr << "SDK image connection failed: " << connected.message << '\n'
              << "endpoint: tcp://" << options.host << ':' << options.port
              << '\n';
    return 4;
  }

  std::cout << "connected: tcp://" << options.host << ':' << options.port
            << " control=udp://" << options.host << ':'
            << options.control_port << " conf=" << options.confidence
            << " gimbal=udp://" << options.host << ':'
            << options.gimbal_port
            << " point-order=BL,TL,TR,BR armor-size="
            << yolo_detect::armorSizeName(options.armor_size)
            << " pnp-frame=C pnp-unit=m\n";

  constexpr char kWindowName[] = "YOLO Armor Detection - Q/Esc to close";
  if (!options.no_display) {
    cv::namedWindow(kWindowName, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::moveWindow(kWindowName, 20, 20);
    cv::resizeWindow(kWindowName, options.display_width,
                     options.display_width * 3 / 4);
  }

  const auto run_started = Clock::now();
  std::uint64_t previous_sequence = 0;
  std::uint64_t received_count = 0;
  std::uint64_t skipped_count = 0;
  auto rate_started = Clock::now();
  std::uint64_t rate_received = 0;
  double receive_fps = 0.0;
  double inference_ms = 0.0;
  double total_inference_ms = 0.0;
  std::size_t latest_detection_count = 0;
  std::size_t latest_pose_count = 0;
  cv::VideoWriter recorder;
  bool recorder_opened = false;

  auto requestScene = [&](daedalus_sdk::SceneMode requested,
                          const char* name) {
    if (!control_available) {
      control_message = "scene control unavailable";
      control_ok = false;
      return;
    }
    control_ok = reportControl(scene->setScene(requested),
                               std::string("setScene ") + name,
                               control_message);
    if (control_ok) scene_state.scene = name;
  };

  auto requestMotion = [&](daedalus_sdk::RangeMotionMode requested) {
    if (!control_available) {
      control_message = "scene control unavailable";
      control_ok = false;
      return;
    }
    SceneState next = scene_state;
    next.motion = requested;
    control_ok = applyMotion(*scene, next, control_message);
    if (control_ok) scene_state = next;
  };

  auto adjustVehicleSpeed = [&](float delta) {
    if (!control_available) {
      control_message = "scene control unavailable";
      control_ok = false;
      return;
    }
    SceneState next = scene_state;
    next.vehicle_speed =
        std::clamp(next.vehicle_speed + delta, 0.0F, 20.0F);
    if (next.vehicle_speed <= 0.0F) {
      if (next.motion == daedalus_sdk::RangeMotionMode::Linear) {
        next.motion = daedalus_sdk::RangeMotionMode::Stationary;
      } else if (next.motion ==
                 daedalus_sdk::RangeMotionMode::LinearAndSpin) {
        next.motion = daedalus_sdk::RangeMotionMode::Spin;
      }
    } else if (next.motion ==
               daedalus_sdk::RangeMotionMode::Stationary) {
      next.motion = daedalus_sdk::RangeMotionMode::Linear;
    } else if (next.motion == daedalus_sdk::RangeMotionMode::Spin) {
      next.motion = daedalus_sdk::RangeMotionMode::LinearAndSpin;
    }
    control_ok = applyMotion(*scene, next, control_message);
    if (control_ok) scene_state = next;
  };

  std::mutex web_command_mutex;
  std::deque<std::string> web_commands;
  const auto applyEkfTuning = [&](std::string_view action) {
    constexpr std::string_view prefix = "ekf-param:";
    if (!hasPrefix(action, prefix)) return false;
    const std::string payload(action.substr(prefix.size()));
    const std::size_t separator = payload.find(':');
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 >= payload.size()) {
      return false;
    }
    const std::string name = payload.substr(0, separator);
    const std::string value_text = payload.substr(separator + 1);
    try {
      const double value = std::stod(value_text);
      if (!std::isfinite(value)) return false;
      if (name == "yaw_max_reprojection_rms_px" ||
          name == "yaw_max_std_rad" || name == "yaw_min_facing_cosine" ||
          name == "yaw_min_opposite_margin_px") {
        auto tuned = constrained_yaw_solver.options();
        if (name == "yaw_max_reprojection_rms_px") {
          tuned.max_reprojection_rms_px = value;
        } else if (name == "yaw_max_std_rad") {
          tuned.max_yaw_std_rad = value;
        } else if (name == "yaw_min_facing_cosine") {
          tuned.min_facing_cosine = value;
        } else {
          tuned.min_opposite_margin_px = value;
        }
        if (!constrained_yaw_solver.setOptions(tuned)) return false;
        control_message = "Yaw validity parameter updated: " + name;
        return true;
      }
      auto tuned = whole_vehicle_tracker.options();
      if (name == "initial_position_std_m") tuned.initial_position_std_m = value;
      else if (name == "initial_velocity_std_mps") tuned.initial_velocity_std_mps = value;
      else if (name == "initial_theta_std_rad") tuned.initial_theta_std_rad = value;
      else if (name == "initial_omega_std_rad_s") tuned.initial_omega_std_rad_s = value;
      else if (name == "initial_geometry_std_m") tuned.initial_geometry_std_m = value;
      else if (name == "q_linear_acceleration") tuned.q_linear_acceleration = value;
      else if (name == "q_angular_acceleration") tuned.q_angular_acceleration = value;
      else if (name == "q_geometry") tuned.q_geometry = value;
      else if (name == "position_std_x_m") tuned.position_std_x_m = value;
      else if (name == "position_std_y_m") tuned.position_std_y_m = value;
      else if (name == "position_std_z_m") tuned.position_std_z_m = value;
      else if (name == "yaw_facing_base_variance_rad2") tuned.yaw_facing_base_variance_rad2 = value;
      else if (name == "yaw_facing_log_variance_scale_rad2") tuned.yaw_facing_log_variance_scale_rad2 = value;
      else if (name == "single_armor_position_variance_scale") tuned.single_armor_position_variance_scale = value;
      else if (name == "association_position_variance_scale") tuned.association_position_variance_scale = value;
      else if (name == "maximum_multi_armor_position_residual_m") tuned.maximum_multi_armor_position_residual_m = value;
      else if (name == "maximum_yaw_update_innovation_rad") tuned.maximum_yaw_update_innovation_rad = value;
      else if (name == "maximum_yaw_association_innovation_rad") tuned.maximum_yaw_association_innovation_rad = value;
      else if (name == "yaw_phase_cost_std_rad") tuned.yaw_phase_cost_std_rad = value;
      else if (name == "adjacent_slot_penalty") tuned.adjacent_slot_penalty = value;
      else if (name == "opposite_slot_penalty") tuned.opposite_slot_penalty = value;
      else if (name == "minimum_visibility_cosine") tuned.minimum_visibility_cosine = value;
      else if (name == "slot_position_cost_weight") tuned.slot_position_cost_weight = value;
      else if (name == "slot_yaw_cost_weight_m_per_rad") tuned.slot_yaw_cost_weight_m_per_rad = value;
      else if (name == "geometry_yaw_consistency_rad") tuned.geometry_yaw_consistency_rad = value;
      else if (name == "geometry_minimum_baseline_m") tuned.geometry_minimum_baseline_m = value;
      else if (name == "geometry_confirming_frames") {
        const double rounded = std::round(value);
        if (std::abs(value - rounded) > 1e-9) return false;
        tuned.geometry_confirming_frames = static_cast<int>(rounded);
      } else if (name == "nis_gate_3d") tuned.nis_gate_3d = value;
      else if (name == "nis_gate_4d") tuned.nis_gate_4d = value;
      else if (name == "maximum_angular_speed_rad_s") {
        tuned.maximum_angular_speed_rad_s = value;
      } else if (name == "maximum_omega_correction_rad_s") {
        tuned.maximum_omega_correction_rad_s = value;
      }
      else return false;
      if (!whole_vehicle_tracker.setOptions(tuned)) return false;
      control_message = "EKF parameter updated: " + name;
      return true;
    } catch (const std::exception&) {
      return false;
    }
  };
  const auto applyProjectionDebug = [&](std::string_view action) {
    constexpr std::string_view prefix = "projection-debug-param:";
    if (!hasPrefix(action, prefix)) return false;
    const std::string payload(action.substr(prefix.size()));
    const std::size_t separator = payload.find(':');
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 >= payload.size()) {
      return false;
    }
    const std::string name = payload.substr(0, separator);
    try {
      const double value = std::stod(payload.substr(separator + 1));
      if (!std::isfinite(value)) return false;
      // Observed anchoring rewrites center/theta every frame. A manual edit
      // must take ownership immediately instead of being overwritten by the
      // next telemetry update.
      projection_debug.anchor_observed = false;
      if (name == "center_x_T_m") projection_debug.center_x_T_m = value;
      else if (name == "center_y_T_m") projection_debug.center_y_T_m = value;
      else if (name == "center_z_T_m") projection_debug.center_z_T_m = value;
      else if (name == "theta_rad") projection_debug.theta_rad = value;
      else if (name == "radius_even_m") projection_debug.radius_even_m = value;
      else if (name == "radius_odd_delta_m") {
        if (std::abs(value) > 1e-12) return false;
        projection_debug.radius_odd_delta_m = 0.0;
      }
      else if (name == "height_odd_delta_m") {
        if (std::abs(value) > 1e-12) return false;
        projection_debug.height_odd_delta_m = 0.0;
      }
      else return false;
      if (projection_debug.radius_even_m < 0.05 ||
          projection_debug.radius_even_m > 0.5 ||
          std::abs(projection_debug.radius_odd_delta_m) > 0.2 ||
          std::abs(projection_debug.height_odd_delta_m) > 0.5) {
        return false;
      }
      control_message = "Projection debug parameter updated: " + name;
      return true;
    } catch (const std::exception&) {
      return false;
    }
  };
  const auto queueWebCommand = [&](std::string_view action) {
    constexpr std::array<std::string_view, 14> kKnownActions = {
        "scene-shooting-range", "scene-energy", "reset", "motion-stop",
        "motion-linear", "motion-spin", "motion-linear-spin", "speed-down",
        "speed-up", "gimbal-follow-toggle", "gimbal-fire", "ekf-reset",
        "projection-debug-toggle", "projection-debug-anchor-toggle"};
    const std::string_view spin_speed_prefix = "spin-speed:";
    const bool is_spin_speed = hasPrefix(action, spin_speed_prefix);
    const bool is_ekf_param = hasPrefix(action, "ekf-param:");
    const bool is_projection_debug_param =
        hasPrefix(action, "projection-debug-param:");
    if (std::find(kKnownActions.begin(), kKnownActions.end(), action) ==
            kKnownActions.end() &&
        !is_spin_speed && !is_ekf_param && !is_projection_debug_param) {
      return false;
    }
    std::lock_guard<std::mutex> lock(web_command_mutex);
    web_commands.emplace_back(action);
    return true;
  };

  const auto processWebCommands = [&] {
    std::deque<std::string> pending;
    {
      std::lock_guard<std::mutex> lock(web_command_mutex);
      pending.swap(web_commands);
    }
    for (const std::string& action : pending) {
      if (hasPrefix(action, "ekf-param:")) {
        if (!applyEkfTuning(action)) {
          control_ok = false;
          control_message = "invalid EKF parameter command";
        } else {
          control_ok = true;
        }
      } else if (hasPrefix(action, "projection-debug-param:")) {
        if (!applyProjectionDebug(action)) {
          control_ok = false;
          control_message = "invalid projection debug parameter command";
        } else {
          control_ok = true;
        }
      } else if (action == "projection-debug-toggle") {
        projection_debug.enabled = !projection_debug.enabled;
        control_ok = true;
        control_message = projection_debug.enabled
                              ? "Projection debug enabled."
                              : "Projection debug disabled.";
      } else if (action == "projection-debug-anchor-toggle") {
        projection_debug.anchor_observed = !projection_debug.anchor_observed;
        control_ok = true;
        control_message = projection_debug.anchor_observed
                              ? "Projection debug anchored to observed E0."
                              : "Projection debug using manual center/theta.";
      } else if (action == "ekf-reset") {
        whole_vehicle_tracker.reset();
        control_ok = true;
        control_message = "EKF track reset.";
      } else if (action == "scene-shooting-range") {
        requestScene(daedalus_sdk::SceneMode::ShootingRange, "shooting-range");
        if (control_ok) requestMotion(scene_state.motion);
      } else if (action == "scene-energy") {
        requestScene(daedalus_sdk::SceneMode::Energy, "energy");
      } else if (action == "reset" && control_available) {
        control_ok = reportControl(scene->resetScene(), "resetScene",
                                   control_message);
        if (control_ok) {
          scene_state.motion = daedalus_sdk::RangeMotionMode::Stationary;
        }
      } else if (action == "motion-stop") {
        requestMotion(daedalus_sdk::RangeMotionMode::Stationary);
      } else if (action == "motion-linear") {
        requestMotion(daedalus_sdk::RangeMotionMode::Linear);
      } else if (action == "motion-spin") {
        requestMotion(daedalus_sdk::RangeMotionMode::Spin);
      } else if (action == "motion-linear-spin") {
        requestMotion(daedalus_sdk::RangeMotionMode::LinearAndSpin);
      } else if (action == "speed-down") {
        adjustVehicleSpeed(-options.speed_step);
      } else if (action == "speed-up") {
        adjustVehicleSpeed(options.speed_step);
      } else if (hasPrefix(action, "spin-speed:")) {
        try {
          const std::string value = action.substr(std::string("spin-speed:").size());
          std::size_t parsed = 0;
          const float speed = std::stof(value, &parsed);
          if (parsed != value.size() || !std::isfinite(speed) || speed < 0.0F ||
              speed > 720.0F) {
            throw std::runtime_error("spin speed out of range");
          }
          SceneState next = scene_state;
          next.spin_speed_deg_s = speed;
          control_ok = applyMotion(*scene, next, control_message);
          if (control_ok) scene_state = next;
        } catch (const std::exception&) {
          control_ok = false;
          control_message = "invalid spin speed command";
        }
      } else if (action == "gimbal-follow-toggle") {
        gimbal_control.toggleFollowing();
      } else if (action == "gimbal-fire") {
        gimbal_control.requestFire();
      }
    }
  };

  std::unique_ptr<yolo_detect::MjpegServer> web_server;
  if (options.web_port != 0) {
    try {
      web_server = std::make_unique<yolo_detect::MjpegServer>(
          yolo_detect::WebServerOptions{options.web_bind, options.web_port,
                                        options.web_jpeg_quality},
          queueWebCommand);
      std::cout << "web debug stream: " << web_server->url()
                << " (open /stream.mjpg for the raw MJPEG endpoint)\n";
    } catch (const std::exception& error) {
      std::cerr << "web server start failed: " << error.what() << '\n';
      return 5;
    }
  }

  bool timed_scene_switched = false;
  const std::uint64_t scene_switch_recorded_frame =
      options.record_path.empty()
          ? 0
          : static_cast<std::uint64_t>(options.scene_switch_after *
                                       options.record_fps);

  while (images.connected()) {
    processWebCommands();
    auto frame = images.waitForLatest(previous_sequence,
                                      std::chrono::milliseconds(
                                          web_server ? 100 : 1500));
    if (!frame) {
      if (frame.status.error != daedalus_sdk::ClientError::Timeout) {
        std::cerr << "SDK receive failed: " << frame.status.message << '\n';
        break;
      }
      if (!options.no_display) {
        const int key = cv::waitKey(1) & 0xff;
        if (key == 27 || key == 'q' || key == 'Q') break;
      }
      continue;
    }

    const auto& source = *frame.value;
    const auto& header = source.header;
    yolo_detect::coordinates::CoordinateSnapshot coordinate_snapshot;
    coordinate_snapshot.frame_sequence = header.source_sequence;
    if (simulator_pose.isOpen()) {
      coordinate_snapshot =
          simulator_pose.snapshotForFrame(header.source_sequence,
                                          header.capture_timestamp_ns);
      coordinate_message = coordinate_snapshot.valid
                               ? "success"
                               : simulator_pose.lastError();
    }
    if (previous_sequence != 0 && header.source_sequence > previous_sequence + 1) {
      skipped_count += header.source_sequence - previous_sequence - 1;
    }
    previous_sequence = header.source_sequence;
    ++received_count;
    ++rate_received;

    if (!timed_scene_switched && options.scene_after != "unchanged" &&
        options.scene_switch_after > 0.0 &&
        ((!options.record_path.empty() &&
          received_count >= scene_switch_recorded_frame) ||
         (options.record_path.empty() &&
          std::chrono::duration<double>(Clock::now() - run_started).count() >=
              options.scene_switch_after))) {
      daedalus_sdk::SceneMode requested_scene;
      if (options.scene_after == "armor") {
        requested_scene = daedalus_sdk::SceneMode::Armor;
      } else if (options.scene_after == "energy") {
        requested_scene = daedalus_sdk::SceneMode::Energy;
      } else if (options.scene_after == "outpost") {
        requested_scene = daedalus_sdk::SceneMode::Outpost;
      } else {
        requested_scene = daedalus_sdk::SceneMode::ShootingRange;
      }
      requestScene(requested_scene, options.scene_after.c_str());
      if (control_ok && options.scene_after == "shooting-range") {
        requestMotion(options.startup_motion);
      }
      timed_scene_switched = true;
    }

    const double rate_elapsed =
        std::chrono::duration<double>(Clock::now() - rate_started).count();
    if (rate_elapsed >= 1.0) {
      receive_fps = static_cast<double>(rate_received) / rate_elapsed;
      rate_received = 0;
      rate_started = Clock::now();
      if (options.no_display) {
        std::cout << "frames=" << received_count
                  << " process_fps=" << fixed(receive_fps)
                  << " inference_ms=" << fixed(inference_ms)
                  << " detections=" << latest_detection_count
                  << " poses=" << latest_pose_count
                  << " skipped=" << skipped_count << '\n';
      }
    }

    const int input_type = header.format == tcp_image::PixelFormat::Rgba32
                               ? CV_8UC4
                               : CV_8UC3;
    cv::Mat sdk_image(static_cast<int>(header.height),
                      static_cast<int>(header.width), input_type,
                      const_cast<std::uint8_t*>(source.payload.data()));
    cv::Mat bgr;
    cv::cvtColor(sdk_image, bgr,
                 header.format == tcp_image::PixelFormat::Rgba32
                     ? cv::COLOR_RGBA2BGR
                     : cv::COLOR_RGB2BGR);

    const auto inference_started = Clock::now();
    std::vector<yolo_detect::ArmorDetection> detections;
    try {
      detections = detector->detect(bgr);
    } catch (const std::exception& error) {
      std::cerr << "inference failed: " << error.what() << '\n';
      break;
    }
    const double current_inference_ms =
        std::chrono::duration<double, std::milli>(Clock::now() -
                                                  inference_started)
            .count();
    total_inference_ms += current_inference_ms;
    inference_ms = inference_ms == 0.0
                       ? current_inference_ms
                       : inference_ms * 0.9 + current_inference_ms * 0.1;
    latest_detection_count = detections.size();

    std::vector<yolo_detect::PoseResult> poses;
    poses.reserve(detections.size());
    for (const yolo_detect::ArmorDetection& detection : detections) {
      const bool keypoints_confident = std::all_of(
          detection.keypoint_confidences.begin(),
          detection.keypoint_confidences.end(), [&](float confidence) {
            return std::isfinite(confidence) &&
                   confidence >= options.keypoint_confidence;
          });
      if (!keypoints_confident) {
        yolo_detect::PoseResult skipped_pose;
        skipped_pose.status =
            yolo_detect::PoseStatus::InsufficientKeypointConfidence;
        skipped_pose.armor_size = options.armor_size;
        poses.push_back(skipped_pose);
        continue;
      }
      poses.push_back(pose_estimator.estimate(
          detection.keypoints, bgr.size(), options.armor_size));
    }
    latest_pose_count = static_cast<std::size_t>(std::count_if(
        poses.begin(), poses.end(),
        [](const yolo_detect::PoseResult& pose) { return pose.valid; }));

    std::vector<yolo_detect::tracking::Measurement> tracker_measurements;
    std::vector<yolo_detect::tracking::ReliableYaw> reliable_yaws;
    if (coordinate_snapshot.valid) {
      const yolo_detect::tracking::TrackerFrame tracker_frame{
          header.source_sequence, header.capture_timestamp_ns};
      for (std::size_t index = 0; index < poses.size(); ++index) {
        yolo_detect::tracking::ReliableYaw reliable_yaw;
        const auto measurement = yolo_detect::tracking::makeTrackerMeasurement(
            tracker_frame, coordinate_snapshot, poses[index], detections[index],
            constrained_yaw_solver, &reliable_yaw);
        reliable_yaws.push_back(reliable_yaw);
        if (measurement) {
          tracker_measurements.push_back(*measurement);
        }
      }
    }
    if (!tracker_measurements.empty()) {
      // A single board feeds the EKF. When two or more are visible, choose
      // the lower constrained-reprojection RMS while retaining all boards for
      // E0..E3 slot association and diagnostic rendering.
      for (auto& measurement : tracker_measurements) {
        measurement.selected_for_ekf = false;
      }
      const auto primary = std::min_element(
          tracker_measurements.begin(), tracker_measurements.end(),
          [](const yolo_detect::tracking::Measurement& left,
             const yolo_detect::tracking::Measurement& right) {
            return left.reprojection_rms_px < right.reprojection_rms_px;
          });
      primary->selected_for_ekf = true;
    }
    std::optional<yolo_detect::tracking::Measurement> projection_reference;
    if (projection_debug.enabled && projection_debug.anchor_observed) {
      projection_debug.has_reference = false;
      for (const auto& measurement : tracker_measurements) {
        if (!measurement.has_inward_yaw) continue;
        projection_reference = measurement;
        projection_debug.theta_rad = measurement.inward_yaw_T_rad;
        projection_debug.center_x_T_m =
            measurement.position_T_m.x() +
            projection_debug.radius_even_m * std::cos(projection_debug.theta_rad);
        projection_debug.center_y_T_m =
            measurement.position_T_m.y() +
            projection_debug.radius_even_m * std::sin(projection_debug.theta_rad);
        projection_debug.center_z_T_m = measurement.position_T_m.z();
        projection_debug.has_reference = true;
        break;
      }
    } else if (projection_debug.enabled) {
      projection_debug.has_reference = false;
    }
    const yolo_detect::tracking::TrackOutput tracker_output =
        whole_vehicle_tracker.update(header.capture_timestamp_ns,
                                    tracker_measurements);

    std::vector<DetectionAim> detection_aims(poses.size());
    if (coordinate_snapshot.valid) {
      for (std::size_t index = 0; index < poses.size(); ++index) {
        if (!poses[index].valid) continue;
        DetectionAim& detection_aim = detection_aims[index];
        detection_aim.coordinate_valid = true;
        detection_aim.center_odom_m =
            yolo_detect::coordinates::cameraPointToOdom(
                coordinate_snapshot, poses[index].center_camera_m);
        detection_aim.aim = aim_solver.solve(
            {detection_aim.center_odom_m, false, 0.0}, coordinate_snapshot);
      }
    }
    const std::size_t latest_aim_count =
        static_cast<std::size_t>(std::count_if(
            detection_aims.begin(), detection_aims.end(),
            [](const DetectionAim& item) { return item.aim.valid; }));

    const std::optional<yolo_detect::control::AimTarget> dynamic_target =
        makeDynamicAimTarget(tracker_output, coordinate_snapshot, aim_solver);
    const yolo_detect::control::DynamicTargetCommand gimbal_command =
        gimbal_control.update(dynamic_target, coordinate_snapshot);
    if (gimbal_command.valid) {
      daedalus_sdk::UdpGimbalCommand command;
      command.yaw_deg =
          static_cast<float>(gimbal_command.aim.yaw_command_deg);
      command.pitch_deg =
          static_cast<float>(gimbal_command.aim.pitch_command_deg);
      command.distance_m = static_cast<float>(gimbal_command.distance_m);
      command.fire_advice = gimbal_command.fire;
      const auto sent = gimbal.sendTracked(command);
      if (sent.ok()) {
        gimbal_control.acknowledgeCommand(*sent.value,
                                          gimbal_command.fire);
      } else {
        gimbal_control.reportSendFailure(sent.status.message);
      }
    }

    if (!options.no_display || !options.record_path.empty() || web_server) {
      cv::Mat annotated = bgr.clone();
      for (std::size_t index = 0; index < detections.size(); ++index) {
        drawDetection(annotated, detections[index], poses[index],
                      detection_aims[index], camera_calibration,
                      options.keypoint_confidence);
      }
      drawPredictedArmorCenters(annotated, tracker_output, coordinate_snapshot,
                                camera_calibration);
      drawProjectionDebugCenters(annotated, projection_debug,
                                 projection_reference, coordinate_snapshot,
                                 camera_calibration);

      drawText(annotated,
             "YOLO armor: " + std::to_string(detections.size()) +
                 "   inference: " + fixed(inference_ms) + " ms (" +
                 fixed(inference_ms > 0.0 ? 1000.0 / inference_ms : 0.0) +
                 " FPS)",
             0, cv::Scalar(80, 255, 120));
      drawText(annotated,
             "SDK RX: " + fixed(receive_fps) +
                 " FPS   frame: " + std::to_string(header.source_sequence) +
                 "   received: " + std::to_string(received_count) +
                 "   skipped: " + std::to_string(skipped_count),
             1);
      drawText(annotated,
             "conf: " + fixed(options.confidence, 3) +
                 "   input: " + std::to_string(options.input_size) +
                 "   points: 1 BL  2 TL  3 TR  4 BR",
             2, cv::Scalar(80, 220, 255));
      drawText(annotated,
             "PnP C: " + std::to_string(latest_pose_count) + "/" +
                 std::to_string(detections.size()) + " valid   plate: " +
                 yolo_detect::armorSizeName(options.armor_size) +
                 "   position unit: m",
             3, latest_pose_count > 0 ? cv::Scalar(80, 255, 120)
                                      : cv::Scalar(80, 180, 255));
      drawText(
          annotated,
          coordinate_snapshot.valid
              ? "Coordinates O: actual yaw=" +
                    fixed(coordinate_snapshot.gimbal_yaw_rad * 180.0 / CV_PI, 2) +
                    " pitch=" +
                    fixed(90.0 + coordinate_snapshot.gimbal_elevation_rad *
                                     180.0 / CV_PI, 2) +
                    " aim=" + std::to_string(latest_aim_count) + "/" +
                    std::to_string(latest_pose_count) + " | " +
                    gimbal_control.status()
              : "Coordinates O: " + coordinate_message,
          4, coordinate_snapshot.valid ? cv::Scalar(80, 255, 120)
                                       : cv::Scalar(80, 180, 255));
      drawText(annotated,
             "scene: " + scene_state.scene +
                 "   control: " +
                 (control_available ? "ready" : "unavailable") +
                 "   target: " + std::to_string(scene_state.target),
             5, control_available ? cv::Scalar(80, 255, 120)
                                  : cv::Scalar(80, 180, 255));
      drawText(annotated,
             "motion: " + std::string(motionName(scene_state.motion)) +
                 "   vehicle speed: " + fixed(scene_state.vehicle_speed, 2) +
                 " m/s   spin: " + fixed(scene_state.spin_speed_deg_s, 1) +
                 " deg/s",
             6, cv::Scalar(80, 220, 255));
      drawText(annotated, control_message, 7,
             control_ok ? cv::Scalar(80, 255, 120)
                        : cv::Scalar(80, 180, 255));
      drawText(
          annotated,
          "tracker T: " +
              std::string(yolo_detect::tracking::trackingStateName(
                  tracker_output.tracking_state)) +
              " observations: " + std::to_string(tracker_measurements.size()) +
              " yaw reliable: " +
              std::to_string(std::count_if(
                  reliable_yaws.begin(), reliable_yaws.end(),
                  [](const yolo_detect::tracking::ReliableYaw& yaw) {
                    return yaw.valid;
                  })) +
              "/" + std::to_string(reliable_yaws.size()) +
              (reliable_yaws.empty() || reliable_yaws.front().valid
                   ? ""
                   : " (" + std::string(yolo_detect::tracking::reliableYawStatusName(
                                  reliable_yaws.front().status)) +
                         " rms=" + fixed(
                             reliable_yaws.front().reprojection_rms_px, 2) + ")"),
          8, tracker_output.has_state ? cv::Scalar(80, 255, 120)
                                      : cv::Scalar(80, 180, 255));

      if (web_server) {
        web_server->publish(
            annotated,
            makeWebTelemetry(header.source_sequence, poses, detection_aims,
                             coordinate_snapshot, coordinate_message,
                             gimbal_control, scene_state, tracker_output,
                             whole_vehicle_tracker.options(),
                             constrained_yaw_solver.options(),
                             projection_debug,
                             reliable_yaws, tracker_measurements));
      }

      if (!options.record_path.empty()) {
        if (!recorder_opened) {
          const std::filesystem::path parent =
              options.record_path.parent_path();
          if (!parent.empty()) {
            std::error_code error;
            std::filesystem::create_directories(parent, error);
            if (error) {
              std::cerr << "record directory creation failed: "
                        << parent.string() << ": " << error.message() << '\n';
              break;
            }
          }
          const std::string extension = options.record_path.extension().string();
          const int codec = extension == ".avi" || extension == ".AVI"
                                ? cv::VideoWriter::fourcc('M', 'J', 'P', 'G')
                                : cv::VideoWriter::fourcc('m', 'p', '4', 'v');
          recorder_opened = recorder.open(options.record_path.string(), codec,
                                          options.record_fps, annotated.size(),
                                          true);
          if (!recorder_opened) {
            std::cerr << "record open failed: " << options.record_path.string()
                      << " (try an .avi path)\n";
            break;
          }
          std::cout << "recording annotated video: "
                    << options.record_path.string() << " at "
                    << fixed(options.record_fps, 1) << " FPS\n";
        }
        recorder.write(annotated);
      }

      if (!options.no_display) {
        cv::Mat display;
        const double display_scale = std::min(
            1.0, static_cast<double>(options.display_width) / annotated.cols);
        cv::resize(annotated, display, {}, display_scale, display_scale,
                   display_scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR);
        cv::imshow(kWindowName, display);

        const int key = cv::waitKey(1) & 0xff;
        if (key == 27 || key == 'q' || key == 'Q') break;
        if (key == '1') requestScene(daedalus_sdk::SceneMode::Armor, "armor");
        if (key == '2') requestScene(daedalus_sdk::SceneMode::Energy, "energy");
        if (key == '3') {
          requestScene(daedalus_sdk::SceneMode::Outpost, "outpost");
        }
        if (key == '4') {
          requestScene(daedalus_sdk::SceneMode::ShootingRange,
                       "shooting-range");
          if (control_ok) requestMotion(scene_state.motion);
        }
        if (key == '0' && control_available) {
          control_ok = reportControl(scene->resetScene(), "resetScene",
                                     control_message);
          if (control_ok) {
            scene_state.motion = daedalus_sdk::RangeMotionMode::Stationary;
          }
        }
        if (key == 's' || key == 'S') {
          requestMotion(daedalus_sdk::RangeMotionMode::Stationary);
        }
        if (key == 'l' || key == 'L') {
          requestMotion(daedalus_sdk::RangeMotionMode::Linear);
        }
        if (key == 'p' || key == 'P') {
          requestMotion(daedalus_sdk::RangeMotionMode::Spin);
        }
        if (key == 'b' || key == 'B') {
          requestMotion(daedalus_sdk::RangeMotionMode::LinearAndSpin);
        }
        if (key == '+' || key == '=') {
          adjustVehicleSpeed(options.speed_step);
        }
        if (key == '-' || key == '_') {
          adjustVehicleSpeed(-options.speed_step);
        }
        if (cv::getWindowProperty(kWindowName, cv::WND_PROP_VISIBLE) < 1.0) {
          break;
        }
      }
    }
    if (options.max_frames > 0 && received_count >= options.max_frames) break;
  }

  images.close();
  recorder.release();
  if (!options.no_display) cv::destroyAllWindows();
  const double run_seconds =
      std::chrono::duration<double>(Clock::now() - run_started).count();
  std::cout << "summary frames=" << received_count
            << " elapsed_s=" << fixed(run_seconds, 2)
            << " process_fps="
            << fixed(run_seconds > 0.0 ? received_count / run_seconds : 0.0)
            << " avg_inference_ms="
            << fixed(received_count > 0 ? total_inference_ms / received_count
                                        : 0.0)
            << " skipped=" << skipped_count << '\n';
  return 0;
}
