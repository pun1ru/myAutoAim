#include "sdk/daedalus_frame_source.hpp"

#include <daedalus_sim_sdk/tcp_image_v1.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace daedalus_sdk = daedalus::sim::sdk::v1;
namespace tcp_image = daedalus::sim::sdk::v1::tcp_image;
using Clock = std::chrono::steady_clock;

namespace {

struct ViewerOptions {
  std::string host = "127.0.0.1";
  std::uint16_t port = daedalus_sdk::kTcpImagePort;
  std::filesystem::path ipc_directory = my_project::sdk::defaultIpcDirectory();
  int display_width = 900;
};

void printUsage() {
  std::cout
      << "Daedalus TCP image viewer (read-only)\n\n"
      << "Usage: daedalus_image_viewer [options]\n"
      << "  --host <address>   TCP image host (default: 127.0.0.1)\n"
      << "  --port <port>      TCP image port (default: 5602)\n"
      << "  --ipc-dir <path>   Daedalus runtime/talos-ipc directory\n"
      << "  --width <pixels>   Initial display width (default: 900)\n"
      << "  --help             Show this help\n\n"
      << "Press Q or Esc to close. This program never sends control commands.\n";
}

std::string requireValue(int& index, int argc, char** argv) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + argv[index]);
  }
  return argv[++index];
}

ViewerOptions parseOptions(int argc, char** argv) {
  ViewerOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      printUsage();
      std::exit(0);
    }
    if (argument == "--host") {
      options.host = requireValue(index, argc, argv);
    } else if (argument == "--port") {
      const int value = std::stoi(requireValue(index, argc, argv));
      if (value < 1 || value > 65535) throw std::runtime_error("invalid port");
      options.port = static_cast<std::uint16_t>(value);
    } else if (argument == "--ipc-dir") {
      options.ipc_directory = requireValue(index, argc, argv);
    } else if (argument == "--width") {
      options.display_width = std::stoi(requireValue(index, argc, argv));
      if (options.display_width < 320 || options.display_width > 3840) {
        throw std::runtime_error("display width must be between 320 and 3840");
      }
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  return options;
}

std::string pixelFormatName(tcp_image::PixelFormat format) {
  return format == tcp_image::PixelFormat::Rgba32 ? "RGBA32" : "RGB24";
}

void drawText(cv::Mat& image, const std::string& text, int x, int line,
              const cv::Scalar& color = {235, 235, 235}) {
  const int baseline_y = 25 + line * 23;
  cv::putText(image, text, {x, baseline_y}, cv::FONT_HERSHEY_SIMPLEX, 0.54,
              {0, 0, 0}, 4, cv::LINE_AA);
  cv::putText(image, text, {x, baseline_y}, cv::FONT_HERSHEY_SIMPLEX, 0.54,
              color, 1, cv::LINE_AA);
}

std::string fixed(double value, int precision = 1) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

}  // namespace

int main(int argc, char** argv) {
  ViewerOptions options;
  try {
    options = parseOptions(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "argument error: " << error.what() << '\n';
    printUsage();
    return 2;
  }

  my_project::sdk::FrameSourceConfig config;
  config.host = options.host;
  config.port = options.port;
  config.ipc_directory = options.ipc_directory;

  my_project::sdk::DaedalusFrameSource source(config);
  auto connected = source.connect();
  if (!connected) {
    std::cerr << "connect failed: " << connected.message << '\n'
              << "endpoint: tcp://" << options.host << ':' << options.port << '\n'
              << "ipc directory: " << options.ipc_directory.string() << '\n';
    return 3;
  }

  std::cout << "connected: tcp://" << options.host << ':' << options.port
            << " metadata=" << (options.ipc_directory / "talos_ipc_meta").string()
            << " commands=disabled\n";

  constexpr char kWindowName[] = "Daedalus Image Viewer - Q/Esc to close";
  cv::namedWindow(kWindowName, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
  cv::moveWindow(kWindowName, 20, 20);
  cv::resizeWindow(kWindowName, options.display_width,
                   options.display_width * 3 / 4);

  auto rate_started = Clock::now();
  std::uint64_t rate_first_sequence = 0;
  std::uint64_t rate_received = 0;
  std::uint64_t total_received = 0;
  std::uint64_t previous_sequence = 0;
  std::uint64_t skipped_frames = 0;
  double receive_fps = 0.0;
  double source_fps = 0.0;

  while (source.connected()) {
    auto received = source.receive();
    if (!received) {
      std::cerr << "receive failed: " << received.status.message << '\n';
      break;
    }

    const auto& image = received.value->image;
    const auto& header = image.header;
    if (previous_sequence != 0 && header.source_sequence > previous_sequence + 1) {
      skipped_frames += header.source_sequence - previous_sequence - 1;
    }
    previous_sequence = header.source_sequence;
    ++total_received;
    ++rate_received;
    if (rate_first_sequence == 0) rate_first_sequence = header.source_sequence;

    const auto elapsed =
        std::chrono::duration<double>(Clock::now() - rate_started).count();
    if (elapsed >= 1.0) {
      receive_fps = static_cast<double>(rate_received) / elapsed;
      source_fps = header.source_sequence >= rate_first_sequence
                       ? static_cast<double>(header.source_sequence -
                                             rate_first_sequence + 1) /
                             elapsed
                       : 0.0;
      rate_started = Clock::now();
      rate_first_sequence = 0;
      rate_received = 0;
    }

    const int input_type = header.format == tcp_image::PixelFormat::Rgba32
                               ? CV_8UC4
                               : CV_8UC3;
    cv::Mat input(static_cast<int>(header.height), static_cast<int>(header.width),
                  input_type, const_cast<std::uint8_t*>(image.payload.data()));
    cv::Mat bgr;
    cv::cvtColor(input, bgr,
                 header.format == tcp_image::PixelFormat::Rgba32
                     ? cv::COLOR_RGBA2BGR
                     : cv::COLOR_RGB2BGR);

    cv::Mat display;
    const double scale = std::min(
        1.0, static_cast<double>(options.display_width) / bgr.cols);
    cv::resize(bgr, display, {}, scale, scale, cv::INTER_AREA);

    drawText(display, "RX FPS: " + fixed(receive_fps) +
                          "   Source FPS: " + fixed(source_fps),
             15, 0, {80, 255, 120});
    drawText(display, "Frame: " + std::to_string(header.source_sequence) +
                          "   Received: " + std::to_string(total_received) +
                          "   Skipped: " + std::to_string(skipped_frames),
             15, 1);
    drawText(display, "Image: " + std::to_string(header.width) + "x" +
                          std::to_string(header.height) + " " +
                          pixelFormatName(header.format) + "   Bytes: " +
                          std::to_string(header.payload_bytes),
             15, 2);
    drawText(display, "Epoch: " + std::to_string(header.producer_epoch), 15, 3);
    drawText(display, "Capture ns: " +
                          std::to_string(header.capture_timestamp_ns),
             15, 4);
    if (received.value->exposure_gimbal.has_value()) {
      const auto& gimbal = *received.value->exposure_gimbal;
      drawText(display, "Sync: yes   Yaw: " + fixed(gimbal.yaw_deg, 2) +
                            "   Pitch: " + fixed(gimbal.pitch_deg, 2),
               15, 5, {80, 255, 120});
    } else {
      drawText(display, "Exposure sync: no", 15, 5, {80, 180, 255});
      drawText(display, received.value->synchronization_message, 15, 6,
               {80, 180, 255});
    }
    drawText(display, "TCP: " + options.host + ':' +
                          std::to_string(options.port) +
                          "   Commands: disabled",
             15, 7, {80, 255, 120});

    cv::imshow(kWindowName, display);
    const int key = cv::waitKey(1) & 0xff;
    if (key == 27 || key == 'q' || key == 'Q') break;
    if (cv::getWindowProperty(kWindowName, cv::WND_PROP_VISIBLE) < 1.0) break;
  }

  source.close();
  cv::destroyAllWindows();
  return 0;
}
