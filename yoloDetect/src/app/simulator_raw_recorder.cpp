#include <daedalus_sim_sdk/scene_control_client.hpp>
#include <daedalus_sim_sdk/tcp_image_client.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace daedalus_sdk = daedalus::sim::sdk::v1;
namespace tcp_image = daedalus::sim::sdk::v1::tcp_image;
using Clock = std::chrono::steady_clock;

namespace {

struct Options {
  std::filesystem::path output_path;
  std::string host = "127.0.0.1";
  std::uint16_t image_port = daedalus_sdk::kTcpImagePort;
  std::uint16_t control_port = daedalus_sdk::kUdpSceneControlPort;
  std::uint8_t target = 3;
  float spin_speed_deg_s = 60.0F;
  double duration_s = 8.0;
  double output_fps = 60.0;
  double warmup_s = 1.0;
};

std::string requireValue(int& index, int argc, char** argv) {
  if (++index >= argc) throw std::runtime_error("missing option value");
  return argv[index];
}

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--output") {
      options.output_path = requireValue(index, argc, argv);
    } else if (argument == "--seconds") {
      options.duration_s = std::stod(requireValue(index, argc, argv));
    } else if (argument == "--fps") {
      options.output_fps = std::stod(requireValue(index, argc, argv));
    } else if (argument == "--warmup") {
      options.warmup_s = std::stod(requireValue(index, argc, argv));
    } else if (argument == "--target") {
      const int target = std::stoi(requireValue(index, argc, argv));
      if (target < 0 || target > 255) throw std::runtime_error("invalid target");
      options.target = static_cast<std::uint8_t>(target);
    } else if (argument == "--spin-speed") {
      options.spin_speed_deg_s = std::stof(requireValue(index, argc, argv));
    } else if (argument == "--host") {
      options.host = requireValue(index, argc, argv);
    } else if (argument == "--image-port") {
      options.image_port = static_cast<std::uint16_t>(
          std::stoul(requireValue(index, argc, argv)));
    } else if (argument == "--control-port") {
      options.control_port = static_cast<std::uint16_t>(
          std::stoul(requireValue(index, argc, argv)));
    } else if (argument == "--help") {
      std::cout << "Usage: simulator_raw_recorder --output <avi|mp4> "
                   "[--seconds 8] [--fps 60] [--warmup 1] "
                   "[--target 3] [--spin-speed 60]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + argument);
    }
  }
  if (options.output_path.empty() || options.duration_s <= 0.0 ||
      options.output_fps <= 0.0 || options.output_fps > 240.0 ||
      options.warmup_s < 0.0 || options.spin_speed_deg_s < 0.0F ||
      options.spin_speed_deg_s > 720.0F) {
    throw std::runtime_error("invalid recorder options");
  }
  return options;
}

bool sceneSucceeded(
    const daedalus_sdk::ClientResult<daedalus_sdk::SceneControlResponse>& result,
    const char* operation) {
  if (!result) {
    std::cerr << operation << " failed: " << result.status.message << '\n';
    return false;
  }
  if (result.value->status != daedalus_sdk::SceneControlStatus::Ok) {
    std::cerr << operation << " rejected: " << result.value->message << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parseOptions(argc, argv);
    daedalus_sdk::SceneControlOptions scene_options;
    scene_options.endpoint = {options.host, options.control_port};
    scene_options.session_id = "raw-simulator-recorder";
    daedalus_sdk::SceneControlClient scene(scene_options);
    if (!sceneSucceeded(scene.createSession(), "createSession") ||
        !sceneSucceeded(scene.setScene(daedalus_sdk::SceneMode::ShootingRange),
                        "setScene") ||
        !sceneSucceeded(scene.resetScene(), "resetScene")) {
      return 2;
    }
    daedalus_sdk::RangeTargetMotion motion;
    motion.target = options.target;
    motion.mode = daedalus_sdk::RangeMotionMode::Spin;
    motion.spin_deg_s = options.spin_speed_deg_s;
    if (!sceneSucceeded(scene.setRangeTargetMotion(motion),
                        "setRangeTargetMotion")) {
      return 2;
    }

    daedalus_sdk::TcpImageClient images({options.host, options.image_port});
    const auto connected = images.connect();
    if (!connected) {
      std::cerr << "TCP image connection failed: " << connected.message << '\n';
      return 3;
    }
    std::uint64_t previous_sequence = 0;
    const auto warmup_end = Clock::now() +
                            std::chrono::duration<double>(options.warmup_s);
    while (Clock::now() < warmup_end) {
      const auto frame = images.waitForLatest(
          previous_sequence, std::chrono::milliseconds(1500));
      if (!frame) continue;
      previous_sequence = frame.value->header.source_sequence;
    }

    const std::filesystem::path parent = options.output_path.parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    cv::VideoWriter recorder;
    std::uint64_t recorded_frames = 0;
    const auto record_started = Clock::now();
    while (std::chrono::duration<double>(Clock::now() - record_started).count() <
           options.duration_s) {
      const auto frame = images.waitForLatest(
          previous_sequence, std::chrono::milliseconds(1500));
      if (!frame) continue;
      previous_sequence = frame.value->header.source_sequence;
      const auto& header = frame.value->header;
      const int type = header.format == tcp_image::PixelFormat::Rgba32
                           ? CV_8UC4
                           : CV_8UC3;
      cv::Mat source(static_cast<int>(header.height),
                     static_cast<int>(header.width), type,
                     const_cast<std::uint8_t*>(frame.value->payload.data()));
      cv::Mat bgr;
      cv::cvtColor(source, bgr,
                   header.format == tcp_image::PixelFormat::Rgba32
                       ? cv::COLOR_RGBA2BGR
                       : cv::COLOR_RGB2BGR);
      if (!recorder.isOpened()) {
        const std::string extension = options.output_path.extension().string();
        const int codec = extension == ".avi" || extension == ".AVI"
                              ? cv::VideoWriter::fourcc('M', 'J', 'P', 'G')
                              : cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        if (!recorder.open(options.output_path.string(), codec,
                           options.output_fps, bgr.size(), true)) {
          throw std::runtime_error("unable to open output video");
        }
      }
      recorder.write(bgr);
      ++recorded_frames;
    }
    recorder.release();
    images.close();
    std::cout << "recorded " << recorded_frames << " raw frames to "
              << options.output_path.string() << '\n';
    return recorded_frames > 0 ? 0 : 4;
  } catch (const std::exception& error) {
    std::cerr << "raw recorder failed: " << error.what() << '\n';
    return 1;
  }
}
