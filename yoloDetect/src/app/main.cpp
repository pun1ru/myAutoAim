#include "detection/yolo_pose_detector.hpp"
#include "web/mjpeg_server.hpp"

#include <daedalus_sim_sdk/scene_control_client.hpp>
#include <daedalus_sim_sdk/tcp_image_client.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace daedalus_sdk = daedalus::sim::sdk::v1;
namespace tcp_image = daedalus::sim::sdk::v1::tcp_image;
using Clock = std::chrono::steady_clock;

#ifndef YOLO_DETECT_DEFAULT_CUDA_LIB_DIR
#define YOLO_DETECT_DEFAULT_CUDA_LIB_DIR ""
#endif

namespace {

struct Options {
  std::string host = "127.0.0.1";
  std::uint16_t port = daedalus_sdk::kTcpImagePort;
  std::uint16_t control_port = daedalus_sdk::kUdpSceneControlPort;
  std::filesystem::path model;
  float confidence = 0.65F;
  float keypoint_confidence = 0.25F;
  float nms = 0.45F;
  int input_size = 640;
  int display_width = 1100;
  std::uint16_t web_port = 0;
  std::string web_bind = "127.0.0.1";
  int web_jpeg_quality = 80;
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

std::filesystem::path defaultModelPath(const char* executable) {
  std::error_code error;
  std::filesystem::path path = std::filesystem::absolute(executable, error);
  if (error) path = executable;
  return path.parent_path() / "robot_armor_0526.onnx";
}

void printUsage() {
  std::cout
      << "Daedalus YOLO armor pose detector\n\n"
      << "Usage: yolo_detect [options]\n"
      << "  --model <path>       ONNX model (default: RobotDetectionModel 0526)\n"
      << "  --host <address>     SDK image host (default: 127.0.0.1)\n"
      << "  --port <port>        SDK image port (default: 5602)\n"
      << "  --control-port <port> Scene control UDP port (default: 5603)\n"
      << "  --conf <0..1>        Object confidence (default: 0.65 for 0526)\n"
      << "  --kpt-conf <0..1>    Point display confidence (default: 0.25)\n"
      << "  --nms <0..1>         NMS IoU threshold (default: 0.45)\n"
      << "  --imgsz <pixels>     Square model input size (default: 640)\n"
      << "  --width <pixels>     Display width (default: 1100)\n"
      << "  --web <port>         Serve annotated MJPEG frames over HTTP\n"
      << "  --web-bind <address> Web bind address (default: 127.0.0.1)\n"
      << "  --web-quality <1..100> JPEG quality for --web (default: 80)\n"
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

std::string requireValue(int& index, int argc, char** argv) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + argv[index]);
  }
  return argv[++index];
}

float parseUnitFloat(const std::string& value, const char* name) {
  const float parsed = std::stof(value);
  if (parsed < 0.0F || parsed > 1.0F) {
    throw std::runtime_error(std::string(name) + " must be in [0, 1]");
  }
  return parsed;
}

Options parseOptions(int argc, char** argv) {
  Options options;
  options.model = defaultModelPath(argv[0]);
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      printUsage();
      std::exit(0);
    }
    if (argument == "--model") {
      options.model = requireValue(index, argc, argv);
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
    } else if (argument == "--conf") {
      options.confidence =
          parseUnitFloat(requireValue(index, argc, argv), "conf");
    } else if (argument == "--kpt-conf") {
      options.keypoint_confidence =
          parseUnitFloat(requireValue(index, argc, argv), "kpt-conf");
    } else if (argument == "--nms") {
      options.nms = parseUnitFloat(requireValue(index, argc, argv), "nms");
    } else if (argument == "--imgsz") {
      options.input_size = std::stoi(requireValue(index, argc, argv));
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
    } else if (argument == "--cpu") {
      options.use_cuda = false;
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  return options;
}

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

void drawDetection(cv::Mat& image,
                   const yolo_detect::ArmorDetection& detection,
                   float keypoint_threshold) {
  const cv::Rect box(cvRound(detection.box.x), cvRound(detection.box.y),
                     cvRound(detection.box.width),
                     cvRound(detection.box.height));
  cv::rectangle(image, box, cv::Scalar(255, 120, 30), 2, cv::LINE_AA);

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
    cv::circle(image, point, 5, colors[index], cv::FILLED, cv::LINE_AA);
    cv::putText(image, std::to_string(index + 1), point + cv::Point(6, -6),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, colors[index], 2,
                cv::LINE_AA);
  }
  if (all_visible) {
    cv::polylines(image, polygon, true, cv::Scalar(30, 210, 255), 2,
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
}

}  // namespace

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

  std::unique_ptr<yolo_detect::MjpegServer> web_server;
  if (options.web_port != 0) {
    try {
      web_server = std::make_unique<yolo_detect::MjpegServer>(
          yolo_detect::WebServerOptions{options.web_bind, options.web_port,
                                        options.web_jpeg_quality});
      std::cout << "web debug stream: " << web_server->url()
                << " (open /stream.mjpg for the raw MJPEG endpoint)\n";
    } catch (const std::exception& error) {
      std::cerr << "web server start failed: " << error.what() << '\n';
      return 5;
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
            << " point-order=BL,TL,TR,BR\n";

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

  bool timed_scene_switched = false;
  const std::uint64_t scene_switch_recorded_frame =
      options.record_path.empty()
          ? 0
          : static_cast<std::uint64_t>(options.scene_switch_after *
                                       options.record_fps);

  while (images.connected()) {
    auto frame = images.waitForLatest(previous_sequence,
                                      std::chrono::milliseconds(1500));
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

    if (!options.no_display || !options.record_path.empty() || web_server) {
      cv::Mat annotated = bgr.clone();
      for (const auto& detection : detections) {
        drawDetection(annotated, detection, options.keypoint_confidence);
      }

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
             "scene: " + scene_state.scene +
                 "   control: " +
                 (control_available ? "ready" : "unavailable") +
                 "   target: " + std::to_string(scene_state.target),
             3, control_available ? cv::Scalar(80, 255, 120)
                                  : cv::Scalar(80, 180, 255));
      drawText(annotated,
             "motion: " + std::string(motionName(scene_state.motion)) +
                 "   vehicle speed: " + fixed(scene_state.vehicle_speed, 2) +
                 " m/s   spin: " + fixed(scene_state.spin_speed_deg_s, 1) +
                 " deg/s",
             4, cv::Scalar(80, 220, 255));
      drawText(annotated, control_message, 5,
             control_ok ? cv::Scalar(80, 255, 120)
                        : cv::Scalar(80, 180, 255));

      if (web_server) web_server->publish(annotated);

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
