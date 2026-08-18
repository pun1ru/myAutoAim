#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace cv {
class Mat;
}

namespace yolo_detect {

struct WebServerOptions {
  std::string bind_address = "127.0.0.1";
  std::uint16_t port = 8080;
  int jpeg_quality = 80;
};

// A small, dependency-free HTTP server for local MJPEG debugging.
class MjpegServer {
 public:
  struct State;

  explicit MjpegServer(WebServerOptions options);
  ~MjpegServer();

  MjpegServer(const MjpegServer&) = delete;
  MjpegServer& operator=(const MjpegServer&) = delete;
  MjpegServer(MjpegServer&&) = delete;
  MjpegServer& operator=(MjpegServer&&) = delete;

  void publish(const cv::Mat& bgr_frame);
 [[nodiscard]] std::string url() const;

 private:
  std::shared_ptr<State> state_;
};

}  // namespace yolo_detect
