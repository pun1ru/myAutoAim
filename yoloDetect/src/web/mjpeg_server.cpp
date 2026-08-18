#include "web/mjpeg_server.hpp"

#include <opencv2/imgcodecs.hpp>

#include <array>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace yolo_detect {

namespace {

#ifdef _WIN32
using Socket = SOCKET;
constexpr Socket kInvalidSocket = INVALID_SOCKET;

void closeSocket(Socket socket) {
  if (socket != kInvalidSocket) closesocket(socket);
}

class WinsockSession {
 public:
  WinsockSession() {
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
      throw std::runtime_error("WSAStartup failed");
    }
  }
  ~WinsockSession() { WSACleanup(); }
};
#else
using Socket = int;
constexpr Socket kInvalidSocket = -1;

void closeSocket(Socket socket) {
  if (socket != kInvalidSocket) close(socket);
}
#endif

bool sendAll(Socket socket, const void* data, std::size_t size) {
  const auto* bytes = static_cast<const char*>(data);
  while (size > 0) {
#ifdef MSG_NOSIGNAL
    const int sent = send(socket, bytes, size, MSG_NOSIGNAL);
#else
    const int sent = send(socket, bytes, static_cast<int>(size), 0);
#endif
    if (sent <= 0) return false;
    bytes += sent;
    size -= static_cast<std::size_t>(sent);
  }
  return true;
}

bool sendText(Socket socket, const std::string& response) {
  return sendAll(socket, response.data(), response.size());
}

std::string htmlPage() {
  return R"(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>YOLO Armor Debug</title>
<style>
  body { margin: 0; background: #15191d; color: #eef3f6; font-family: system-ui, sans-serif; }
  header { padding: 14px 18px; background: #20282e; border-bottom: 1px solid #39464f; }
  h1 { margin: 0; font-size: 18px; font-weight: 600; }
  main { padding: 16px; }
  img { display: block; width: min(100%, 1400px); height: auto; background: #090b0d; }
  p, #status { color: #aab7c0; font-size: 14px; }
  .controls { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(132px, 1fr)); max-width: 760px; margin: 16px 0; }
  button { min-height: 38px; border: 1px solid #52616b; border-radius: 4px; background: #273139; color: #eef3f6; cursor: pointer; font: inherit; }
  button:hover { background: #34434e; }
  button.scene { border-color: #2f9d76; }
  button.stop { border-color: #c65b56; }
</style>
</head>
<body>
<header><h1>YOLO Armor Debug Stream</h1></header>
<main>
<img src="/stream.mjpg" alt="Waiting for detector frames">
<section class="controls" aria-label="Simulator controls">
  <button class="scene" data-action="scene-shooting-range">Shooting Range</button>
  <button class="scene" data-action="scene-energy">Energy</button>
  <button data-action="reset">Reset Scene</button>
  <button class="stop" data-action="motion-stop">Stop Vehicle</button>
  <button data-action="motion-linear">Linear</button>
  <button data-action="motion-spin">Spin</button>
  <button data-action="motion-linear-spin">Linear + Spin</button>
  <button data-action="speed-down">Speed -</button>
  <button data-action="speed-up">Speed +</button>
</section>
<p id="status">Controls are sent to the detector command queue.</p>
</main>
<script>
const status = document.getElementById('status');
for (const button of document.querySelectorAll('[data-action]')) {
  button.addEventListener('click', async () => {
    const action = button.dataset.action;
    button.disabled = true;
    status.textContent = 'Sending ' + action + '...';
    try {
      const response = await fetch('/api/control?action=' + encodeURIComponent(action), { cache: 'no-store' });
      const result = await response.json();
      status.textContent = result.message;
    } catch (_) {
      status.textContent = 'Control request failed.';
    } finally {
      button.disabled = false;
    }
  });
}
</script>
</body>
</html>
)";
}

}  // namespace

struct MjpegServer::State {
  WebServerOptions options;
  Socket listener = kInvalidSocket;
  std::atomic<bool> running{true};
  std::thread accept_thread;
  std::mutex frame_mutex;
  std::condition_variable frame_ready;
  std::vector<unsigned char> latest_jpeg;
  std::uint64_t frame_sequence = 0;
  ControlHandler control_handler;
#ifdef _WIN32
  std::unique_ptr<WinsockSession> winsock;
#endif
};

namespace {

void sendNotFound(Socket client) {
  sendText(client, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n"
                   "Connection: close\r\n\r\n");
}

void sendJson(Socket client, int status, const char* message) {
  const std::string body = std::string("{\"message\":\"") + message + "\"}";
  const std::string reason = status == 202 ? "Accepted" : "Bad Request";
  sendText(client, "HTTP/1.1 " + std::to_string(status) + " " + reason +
                   "\r\nContent-Type: application/json\r\nCache-Control: no-store\r\n"
                   "Content-Length: " + std::to_string(body.size()) +
                   "\r\nConnection: close\r\n\r\n" + body);
}

std::string controlAction(const std::string& request) {
  const std::size_t path_end = request.find(' ', 4);
  if (path_end == std::string::npos) return {};
  const std::string target = request.substr(4, path_end - 4);
  constexpr std::string_view kPrefix = "/api/control?action=";
  if (target.rfind(kPrefix, 0) != 0) return {};
  std::string action = target.substr(kPrefix.size());
  const std::size_t next_parameter = action.find('&');
  if (next_parameter != std::string::npos) action.resize(next_parameter);
  if (action.empty() || action.size() > 64) return {};
  for (const unsigned char character : action) {
    if (!std::isalnum(character) && character != '-') return {};
  }
  return action;
}

void streamFrames(Socket client, const std::shared_ptr<MjpegServer::State>& state) {
  constexpr char kBoundary[] = "yolo-frame";
  if (!sendText(client, "HTTP/1.1 200 OK\r\n"
                        "Cache-Control: no-store, no-cache, must-revalidate\r\n"
                        "Pragma: no-cache\r\n"
                        "Connection: close\r\n"
                        "Content-Type: multipart/x-mixed-replace; boundary=" +
                        std::string(kBoundary) + "\r\n\r\n")) {
    return;
  }

  std::uint64_t sent_sequence = 0;
  while (state->running.load()) {
    std::vector<unsigned char> jpeg;
    {
      std::unique_lock<std::mutex> lock(state->frame_mutex);
      state->frame_ready.wait(lock, [&] {
        return !state->running.load() ||
               (state->frame_sequence != sent_sequence &&
                !state->latest_jpeg.empty());
      });
      if (!state->running.load()) return;
      jpeg = state->latest_jpeg;
      sent_sequence = state->frame_sequence;
    }

    const std::string header = "--" + std::string(kBoundary) + "\r\n"
        "Content-Type: image/jpeg\r\n"
        "Content-Length: " + std::to_string(jpeg.size()) + "\r\n\r\n";
    if (!sendText(client, header) || !sendAll(client, jpeg.data(), jpeg.size()) ||
        !sendText(client, "\r\n")) {
      return;
    }
  }
}

void handleClient(Socket client, const std::shared_ptr<MjpegServer::State>& state) {
  std::array<char, 4096> request{};
  const int received = recv(client, request.data(),
                            static_cast<int>(request.size() - 1), 0);
  if (received <= 0) {
    closeSocket(client);
    return;
  }

  const std::string request_line(request.data(),
                                 static_cast<std::size_t>(received));
  const bool get_root = request_line.rfind("GET / ", 0) == 0 ||
                        request_line.rfind("GET /HTTP", 0) == 0;
  const bool get_stream = request_line.rfind("GET /stream.mjpg ", 0) == 0;
  const bool get_snapshot = request_line.rfind("GET /snapshot.jpg ", 0) == 0;
  const std::string action = controlAction(request_line);

  if (get_root) {
    const std::string page = htmlPage();
    sendText(client, "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                     "Cache-Control: no-store\r\nContent-Length: " +
                         std::to_string(page.size()) + "\r\nConnection: close\r\n\r\n" +
                         page);
  } else if (get_stream) {
    streamFrames(client, state);
  } else if (get_snapshot) {
    std::vector<unsigned char> jpeg;
    {
      std::lock_guard<std::mutex> lock(state->frame_mutex);
      jpeg = state->latest_jpeg;
    }
    if (jpeg.empty()) {
      sendText(client, "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n"
                       "Connection: close\r\n\r\n");
    } else {
      const std::string header = "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\n"
          "Cache-Control: no-store\r\nContent-Length: " +
          std::to_string(jpeg.size()) + "\r\nConnection: close\r\n\r\n";
      sendText(client, header);
      sendAll(client, jpeg.data(), jpeg.size());
    }
  } else if (!action.empty()) {
    const bool accepted = state->running.load() && state->control_handler &&
                          state->control_handler(action);
    sendJson(client, accepted ? 202 : 400,
             accepted ? "Control command queued." : "Unknown control command.");
  } else {
    sendNotFound(client);
  }
  closeSocket(client);
}

void acceptClients(const std::shared_ptr<MjpegServer::State>& state) {
  while (state->running.load()) {
    sockaddr_storage peer{};
#ifdef _WIN32
    int peer_size = sizeof(peer);
#else
    socklen_t peer_size = sizeof(peer);
#endif
    const Socket client = accept(state->listener,
                                 reinterpret_cast<sockaddr*>(&peer),
                                 &peer_size);
    if (client == kInvalidSocket) {
      if (!state->running.load()) return;
      continue;
    }
    std::thread(handleClient, client, state).detach();
  }
}

}  // namespace

MjpegServer::MjpegServer(WebServerOptions options, ControlHandler control_handler)
    : state_(std::make_shared<State>()) {
  if (options.port == 0) throw std::invalid_argument("web port must be non-zero");
  if (options.jpeg_quality < 1 || options.jpeg_quality > 100) {
    throw std::invalid_argument("web JPEG quality must be in [1, 100]");
  }
  state_->options = std::move(options);
  state_->control_handler = std::move(control_handler);
#ifdef _WIN32
  state_->winsock = std::make_unique<WinsockSession>();
#endif

  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;
  addrinfo* addresses = nullptr;
  const std::string port = std::to_string(state_->options.port);
  const int resolved = getaddrinfo(state_->options.bind_address.c_str(),
                                   port.c_str(), &hints, &addresses);
  if (resolved != 0) {
    throw std::runtime_error("invalid web bind address: " +
                             state_->options.bind_address);
  }

  for (addrinfo* address = addresses; address != nullptr;
       address = address->ai_next) {
    const Socket listener = socket(address->ai_family, address->ai_socktype,
                                   address->ai_protocol);
    if (listener == kInvalidSocket) continue;
    int enabled = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&enabled), sizeof(enabled));
    if (bind(listener, address->ai_addr,
             static_cast<int>(address->ai_addrlen)) == 0 &&
        listen(listener, 8) == 0) {
      state_->listener = listener;
      break;
    }
    closeSocket(listener);
  }
  freeaddrinfo(addresses);
  if (state_->listener == kInvalidSocket) {
    throw std::runtime_error("cannot listen on " + state_->options.bind_address +
                             ":" + port);
  }
  state_->accept_thread = std::thread(acceptClients, state_);
}

MjpegServer::~MjpegServer() {
  state_->running.store(false);
  state_->frame_ready.notify_all();
#ifdef _WIN32
  if (state_->listener != kInvalidSocket) shutdown(state_->listener, SD_BOTH);
#else
  if (state_->listener != kInvalidSocket) shutdown(state_->listener, SHUT_RDWR);
#endif
  closeSocket(state_->listener);
  state_->listener = kInvalidSocket;
  if (state_->accept_thread.joinable()) state_->accept_thread.join();
}

void MjpegServer::publish(const cv::Mat& bgr_frame) {
  if (bgr_frame.empty()) return;
  std::vector<unsigned char> jpeg;
  const std::vector<int> parameters = {cv::IMWRITE_JPEG_QUALITY,
                                       state_->options.jpeg_quality};
  if (!cv::imencode(".jpg", bgr_frame, jpeg, parameters)) return;
  {
    std::lock_guard<std::mutex> lock(state_->frame_mutex);
    state_->latest_jpeg = std::move(jpeg);
    ++state_->frame_sequence;
  }
  state_->frame_ready.notify_all();
}

std::string MjpegServer::url() const {
  return "http://" + state_->options.bind_address + ":" +
         std::to_string(state_->options.port) + "/";
}

}  // namespace yolo_detect
