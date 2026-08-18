#include <daedalus_sim_sdk/scene_control_client.hpp>
#include <daedalus_sim_sdk/tcp_image_client.hpp>
#include <daedalus_sim_sdk/tcp_image_v1.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

using namespace daedalus::sim::sdk::v1;
namespace tcp_image = daedalus::sim::sdk::v1::tcp_image;
using Clock = std::chrono::steady_clock;

constexpr double kOutpostSeconds = 4.0;
constexpr double kRangeSeconds = 4.0;
constexpr int kCaptureIntervalMs = 27;  // About 37 FPS, 150 images over 4 seconds.
constexpr double kCaptureGraceSeconds = 0.25;
constexpr std::uint64_t kImagesPerScene = 150;
constexpr std::uint8_t kTargetId = 3;
constexpr float kDirectionDeg = 90.0F;
constexpr float kLinearSpeedMps = 2.0F;
constexpr float kLinearSpanM = 4.0F;
constexpr float kSpinSpeedDegS = 180.0F;

bool sceneOk(const ClientResult<SceneControlResponse>& result,
            const char* operation) {
  if (!result) {
    std::cerr << operation << " failed: " << result.status.message << '\n';
    return false;
  }
  if (result.value->status != SceneControlStatus::Ok) {
    std::cerr << operation << " rejected: " << result.value->message << '\n';
    return false;
  }
  return true;
}

bool setRangeMotion(SceneControlClient& scene) {
  RangeTargetMotion motion;
  motion.target = kTargetId;
  motion.mode = RangeMotionMode::LinearAndSpin;
  motion.direction_deg = kDirectionDeg;
  motion.linear_speed_mps = kLinearSpeedMps;
  motion.linear_span_m = kLinearSpanM;
  motion.spin_deg_s = kSpinSpeedDegS;
  return sceneOk(scene.setRangeTargetMotion(motion), "setRangeTargetMotion");
}

bool stopRangeMotion(SceneControlClient& scene) {
  RangeTargetMotion motion;
  motion.target = kTargetId;
  motion.mode = RangeMotionMode::Stationary;
  return sceneOk(scene.setRangeTargetMotion(motion), "stopRangeMotion");
}

std::filesystem::path projectRoot(const char* executable) {
  std::error_code error;
  const auto absolute = std::filesystem::absolute(executable, error);
  const auto path = error ? std::filesystem::path(executable) : absolute;
  // Release -> windows-vs2022 -> build -> new_trains.
  return path.parent_path().parent_path().parent_path().parent_path();
}

std::string imageName(const char* scene_name, std::uint64_t index,
                      std::uint64_t source_sequence) {
  std::ostringstream name;
  name << scene_name << '_' << std::setfill('0') << std::setw(5) << index
       << "_seq_" << source_sequence << ".bmp";
  return name.str();
}

bool saveImage(const std::filesystem::path& path, const cv::Mat& bgr) {
  return cv::imwrite(path.string(), bgr);
}

cv::Mat toBgr(const TcpImageFrame& frame) {
  const auto& header = frame.header;
  const int type = header.format == tcp_image::PixelFormat::Rgba32
                       ? CV_8UC4
                       : CV_8UC3;
  cv::Mat input(static_cast<int>(header.height), static_cast<int>(header.width),
                type, const_cast<std::uint8_t*>(frame.payload.data()));
  cv::Mat bgr;
  cv::cvtColor(input, bgr, header.format == tcp_image::PixelFormat::Rgba32
                              ? cv::COLOR_RGBA2BGR
                              : cv::COLOR_RGB2BGR);
  return bgr;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--help") {
    std::cout << "Captures 4s outpost + 4s shooting-range linear-spin images.\n"
                 "Usage: new_trains_capture [output-directory]\n";
    return 0;
  }

  const std::filesystem::path output = argc > 1
                                           ? std::filesystem::path(argv[1])
                                           : projectRoot(argv[0]) / "data" /
                                                 "raw";
  const auto outpost_dir = output / "outpost";
  const auto range_dir = output / "shooting_range_linear_spin";
  const auto manifest_dir = output.parent_path() / "manifests";
  std::error_code directory_error;
  std::filesystem::create_directories(outpost_dir, directory_error);
  std::filesystem::create_directories(range_dir, directory_error);
  std::filesystem::create_directories(manifest_dir, directory_error);
  if (directory_error) {
    std::cerr << "create output directories failed: "
              << directory_error.message() << '\n';
    return 1;
  }

  const auto manifest_path = manifest_dir / "capture_4s_outpost_4s_range.csv";
  std::ofstream manifest(manifest_path, std::ios::trunc);
  if (!manifest) {
    std::cerr << "open manifest failed: " << manifest_path << '\n';
    return 1;
  }
  manifest << "file,scene,source_sequence,elapsed_seconds\n";

  SceneControlOptions scene_options;
  scene_options.session_id = "new-trains-capture";
  SceneControlClient scene(scene_options);
  if (!sceneOk(scene.createSession(), "createSession") ||
      !sceneOk(scene.setScene(SceneMode::Outpost), "setScene outpost") ||
      !sceneOk(scene.resetScene(), "resetScene")) {
    return 2;
  }

  TcpImageClient images;
  const ClientStatus connected = images.connect();
  if (!connected) {
    std::cerr << "connect image stream failed: " << connected.message << '\n';
    return 3;
  }

  std::cout << "Capturing to: " << output << '\n'
            << "Outpost 0-4s; shooting range 4-8s; capture interval "
            << kCaptureIntervalMs << " ms\n";
  const auto started = Clock::now();
  auto next_capture = started;
  std::uint64_t previous_sequence = 0;
  std::uint64_t outpost_count = 0;
  std::uint64_t range_count = 0;
  bool range_started = false;

  while (true) {
    const auto elapsed =
        std::chrono::duration<double>(Clock::now() - started).count();
    if (elapsed >= kOutpostSeconds + kRangeSeconds + kCaptureGraceSeconds) {
      break;
    }

    auto frame = images.waitForLatest(previous_sequence,
                                      std::chrono::milliseconds(1000));
    if (!frame) {
      std::cerr << "waitForLatest failed: " << frame.status.message << '\n';
      break;
    }
    previous_sequence = frame.value->header.source_sequence;

    const auto frame_elapsed =
        std::chrono::duration<double>(Clock::now() - started).count();
    if (!range_started && frame_elapsed >= kOutpostSeconds &&
        outpost_count >= kImagesPerScene) {
      if (!sceneOk(scene.setScene(SceneMode::ShootingRange),
                   "setScene shooting-range") ||
          !setRangeMotion(scene)) {
        break;
      }
      range_started = true;
      std::cout << "Switched to shooting range at " << frame_elapsed << "s\n";
    }

    if (Clock::now() < next_capture) continue;
    if ((!range_started && outpost_count >= kImagesPerScene) ||
        (range_started && range_count >= kImagesPerScene)) {
      next_capture += std::chrono::milliseconds(kCaptureIntervalMs);
      continue;
    }
    const cv::Mat bgr = toBgr(*frame.value);
    const char* scene_name = range_started ? "shooting_range" : "outpost";
    const auto directory = range_started ? range_dir : outpost_dir;
    const std::uint64_t scene_index = range_started ? range_count++ : outpost_count++;
    const auto path = directory / imageName(scene_name, scene_index,
                                             frame.value->header.source_sequence);
    if (!saveImage(path, bgr)) {
      std::cerr << "save image failed: " << path << '\n';
      break;
    }
    manifest << std::filesystem::relative(path, output.parent_path()).generic_string()
             << ',' << (range_started ? "shooting_range_linear_spin" : "outpost")
             << ',' << frame.value->header.source_sequence << ','
             << std::fixed << std::setprecision(3) << frame_elapsed << '\n';
    next_capture += std::chrono::milliseconds(kCaptureIntervalMs);
  }

  stopRangeMotion(scene);
  images.close();
  manifest.close();
  std::cout << "Captured outpost=" << outpost_count
            << " shooting_range=" << range_count << '\n'
            << "Manifest: " << manifest_path << '\n';
  return 0;
}
