#include "web/mjpeg_server.hpp"

#include <opencv2/imgcodecs.hpp>

#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>
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
  h2 { margin: 0; font-size: 15px; font-weight: 600; }
  main { padding: 16px; }
  img { display: block; width: min(100%, 1400px); height: auto; background: #090b0d; }
  p, #control-status, #pose-frame { color: #aab7c0; font-size: 14px; }
  .pose-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; max-width: 1400px; margin-top: 18px; }
  .frame-meta { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 14px; color: #aab7c0; font-size: 14px; }
  .pose-table-wrap { max-width: 1400px; overflow-x: auto; margin-top: 8px; border-top: 1px solid #39464f; border-bottom: 1px solid #39464f; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  th, td { padding: 9px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid #2b353c; font-size: 13px; }
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:last-child, td:last-child { text-align: left; }
  th { color: #aab7c0; font-weight: 500; }
  tr.valid td { color: #d8f5e8; }
  tr.warning td { color: #f4d58b; }
  tr.invalid td { color: #efb6a8; }
  .controls { display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(132px, 1fr)); max-width: 760px; margin: 16px 0; }
  .aim-controls { max-width: 520px; }
  .gimbal-state { grid-column: 1 / -1; color: #aab7c0; font-size: 13px; }
  button.follow { border-color: #d8a43a; }
  button.follow.active { background: #66511e; border-color: #f0c75e; }
  button.fire { background: #612c2c; border-color: #d4655f; }
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
<section aria-labelledby="pose-title">
  <div class="pose-heading">
    <h2 id="pose-title">Pose and Aim</h2>
    <div class="frame-meta">
      <span id="coordinate-state">Coordinates unavailable</span>
      <span id="pose-frame">Frame 0</span>
    </div>
  </div>
  <div class="pose-table-wrap">
    <table>
      <thead><tr><th>Target</th><th>Plate</th><th>C X</th><th>C Y</th><th>C Z</th><th>O X</th><th>O Y</th><th>O Z</th><th>Yaw deg</th><th>Pitch deg</th><th>TOF s</th><th>Drop m</th><th>RMS px</th><th>Candidates</th><th>PnP</th><th>Aim</th></tr></thead>
      <tbody id="pose-rows"><tr><td colspan="16">Waiting for detections</td></tr></tbody>
    </table>
  </div>
</section>
<section class="controls aim-controls" aria-label="Aim controls">
  <button id="follow-button" class="follow" data-action="gimbal-follow-toggle" aria-pressed="false">Start Static Follow</button>
  <button id="fire-button" class="fire" data-action="gimbal-fire">Fire Once</button>
  <span id="gimbal-state" class="gimbal-state">Gimbal idle</span>
</section>
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
<p id="control-status">Controls are sent to the detector command queue.</p>
</main>
<script>
const controlStatus = document.getElementById('control-status');
const followButton = document.getElementById('follow-button');
const fireButton = document.getElementById('fire-button');
const gimbalState = document.getElementById('gimbal-state');
for (const button of document.querySelectorAll('[data-action]')) {
  button.addEventListener('click', async () => {
    const action = button.dataset.action;
    button.disabled = true;
    controlStatus.textContent = 'Sending ' + action + '...';
    try {
      const response = await fetch('/api/control?action=' + encodeURIComponent(action), { cache: 'no-store' });
      const result = await response.json();
      controlStatus.textContent = result.message;
    } catch (_) {
      controlStatus.textContent = 'Control request failed.';
    } finally {
      button.disabled = false;
    }
  });
}

const poseRows = document.getElementById('pose-rows');
const poseFrame = document.getElementById('pose-frame');
const coordinateState = document.getElementById('coordinate-state');
const cell = (text) => {
  const element = document.createElement('td');
  element.textContent = text;
  return element;
};
const metric = (value, digits) => typeof value === 'number' ? value.toFixed(digits) : '-';
async function refreshPose() {
  try {
    const response = await fetch('/api/status', { cache: 'no-store' });
    if (!response.ok) return;
    const state = await response.json();
    poseFrame.textContent = 'Frame ' + state.source_sequence;
    coordinateState.textContent = state.coordinate_valid
      ? 'Odom ready | camera offset error ' + metric(state.camera_position_error_m, 4) + ' m'
      : state.coordinate_status;
    followButton.textContent =
      state.gimbal_following ? 'Stop Static Follow' : 'Start Static Follow';
    followButton.classList.toggle('active', state.gimbal_following);
    followButton.setAttribute('aria-pressed', String(state.gimbal_following));
    fireButton.disabled = state.fire_pending;
    const staticTarget = state.static_target_valid
      ? ' | target O=(' + metric(state.static_target_odom_x_m, 3) + ', ' +
        metric(state.static_target_odom_y_m, 3) + ', ' +
        metric(state.static_target_odom_z_m, 3) + ')'
      : '';
    gimbalState.textContent = state.gimbal_status + staticTarget;
    const rows = [];
    for (const pose of state.poses) {
      const row = document.createElement('tr');
      row.className = !pose.valid ? 'invalid' : (pose.aim_valid ? 'valid' : 'warning');
      row.append(cell('#' + (pose.detection_index + 1)));
      row.append(cell(pose.armor_size));
      row.append(cell(metric(pose.x_m, 3)));
      row.append(cell(metric(pose.y_m, 3)));
      row.append(cell(metric(pose.z_m, 3)));
      row.append(cell(metric(pose.odom_x_m, 3)));
      row.append(cell(metric(pose.odom_y_m, 3)));
      row.append(cell(metric(pose.odom_z_m, 3)));
      row.append(cell(metric(pose.yaw_command_deg, 2)));
      row.append(cell(metric(pose.pitch_command_deg, 2)));
      row.append(cell(metric(pose.time_of_flight_s, 3)));
      row.append(cell(metric(pose.gravity_drop_m, 3)));
      row.append(cell(metric(pose.reprojection_rms_px, 2)));
      row.append(cell(String(pose.candidate_count)));
      row.append(cell(pose.status));
      const downstreamStatus = pose.coordinate_valid
        ? pose.aim_status + (pose.ballistic_status !== 'success'
          ? ' / ' + pose.ballistic_status
          : '')
        : pose.coordinate_status;
      row.append(cell(downstreamStatus));
      rows.push(row);
    }
    if (rows.length === 0) {
      const row = document.createElement('tr');
      const empty = cell('No armor detections');
      empty.colSpan = 16;
      row.append(empty);
      rows.push(row);
    }
    poseRows.replaceChildren(...rows);
  } catch (_) {
  }
}
refreshPose();
setInterval(refreshPose, 250);
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
  std::string latest_telemetry_json =
      R"({"source_sequence":0,"coordinate_valid":false,)"
      R"("coordinate_status":"waiting for first frame",)"
      R"("camera_position_error_m":null,"actual_yaw_deg":null,)"
      R"("actual_pitch_command_deg":null,"gimbal_following":false,)"
      R"("fire_pending":false,"gimbal_status":"gimbal idle",)"
      R"("last_gimbal_command_id":0,"last_command_fired":false,)"
      R"("static_target_valid":false,"static_target_odom_x_m":null,)"
      R"("static_target_odom_y_m":null,"static_target_odom_z_m":null,)"
      R"("poses":[]})";
  std::uint64_t frame_sequence = 0;
  ControlHandler control_handler;
#ifdef _WIN32
  std::unique_ptr<WinsockSession> winsock;
#endif
};

namespace {

std::string jsonEscape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char character : value) {
    if (character == '\\' || character == '"') escaped.push_back('\\');
    escaped.push_back(character);
  }
  return escaped;
}

std::string telemetryJson(const WebFrameTelemetry& telemetry) {
  std::ostringstream stream;
  const bool coordinate_metrics_valid =
      telemetry.coordinate_valid &&
      std::isfinite(telemetry.camera_position_error_m) &&
      std::isfinite(telemetry.actual_yaw_deg) &&
      std::isfinite(telemetry.actual_pitch_command_deg);
  const bool static_target_metrics_valid =
      telemetry.static_target_valid &&
      std::isfinite(telemetry.static_target_odom_x_m) &&
      std::isfinite(telemetry.static_target_odom_y_m) &&
      std::isfinite(telemetry.static_target_odom_z_m);
  stream << std::setprecision(8) << "{\"source_sequence\":"
         << telemetry.source_sequence
         << ",\"coordinate_valid\":"
         << (coordinate_metrics_valid ? "true" : "false")
         << ",\"coordinate_status\":\""
         << jsonEscape(telemetry.coordinate_status) << "\"";
  if (coordinate_metrics_valid) {
    stream << ",\"camera_position_error_m\":"
           << telemetry.camera_position_error_m
           << ",\"actual_yaw_deg\":" << telemetry.actual_yaw_deg
           << ",\"actual_pitch_command_deg\":"
           << telemetry.actual_pitch_command_deg;
  } else {
    stream << ",\"camera_position_error_m\":null,"
              "\"actual_yaw_deg\":null,"
              "\"actual_pitch_command_deg\":null";
  }
  stream << ",\"gimbal_following\":"
         << (telemetry.gimbal_following ? "true" : "false")
         << ",\"fire_pending\":"
         << (telemetry.fire_pending ? "true" : "false")
         << ",\"gimbal_status\":\"" << jsonEscape(telemetry.gimbal_status)
         << "\",\"last_gimbal_command_id\":"
         << telemetry.last_gimbal_command_id
         << ",\"last_command_fired\":"
         << (telemetry.last_command_fired ? "true" : "false")
         << ",\"static_target_valid\":"
         << (static_target_metrics_valid ? "true" : "false");
  if (static_target_metrics_valid) {
    stream << ",\"static_target_odom_x_m\":"
           << telemetry.static_target_odom_x_m
           << ",\"static_target_odom_y_m\":"
           << telemetry.static_target_odom_y_m
           << ",\"static_target_odom_z_m\":"
           << telemetry.static_target_odom_z_m;
  } else {
    stream << ",\"static_target_odom_x_m\":null,"
              "\"static_target_odom_y_m\":null,"
              "\"static_target_odom_z_m\":null";
  }
  stream << ",\"poses\":[";
  for (std::size_t index = 0; index < telemetry.poses.size(); ++index) {
    if (index != 0) stream << ',';
    const WebPoseTelemetry& pose = telemetry.poses[index];
    const bool pnp_metrics_valid =
        pose.valid && std::isfinite(pose.x_m) && std::isfinite(pose.y_m) &&
        std::isfinite(pose.z_m) && std::isfinite(pose.reprojection_rms_px);
    const bool coordinate_valid =
        pose.coordinate_valid && std::isfinite(pose.odom_x_m) &&
        std::isfinite(pose.odom_y_m) && std::isfinite(pose.odom_z_m);
    const bool aim_valid =
        pose.aim_valid && std::isfinite(pose.yaw_command_deg) &&
        std::isfinite(pose.pitch_command_deg) &&
        std::isfinite(pose.time_of_flight_s) &&
        std::isfinite(pose.gravity_drop_m) &&
        std::isfinite(pose.muzzle_odom_x_m) &&
        std::isfinite(pose.muzzle_odom_y_m) &&
        std::isfinite(pose.muzzle_odom_z_m);
    stream << "{\"detection_index\":" << pose.detection_index
           << ",\"valid\":" << (pnp_metrics_valid ? "true" : "false")
           << ",\"armor_size\":\"" << jsonEscape(pose.armor_size)
           << "\",\"status\":\"" << jsonEscape(pose.status)
           << "\",\"coordinate_valid\":"
           << (coordinate_valid ? "true" : "false")
           << ",\"coordinate_status\":\""
           << jsonEscape(pose.coordinate_status)
           << "\",\"aim_valid\":" << (aim_valid ? "true" : "false")
           << ",\"aim_status\":\"" << jsonEscape(pose.aim_status)
           << "\",\"ballistic_status\":\""
           << jsonEscape(pose.ballistic_status) << "\""
           << ",\"predicted\":" << (pose.predicted ? "true" : "false");
    if (std::isfinite(pose.prediction_horizon_s)) {
      stream << ",\"prediction_horizon_s\":"
             << pose.prediction_horizon_s;
    } else {
      stream << ",\"prediction_horizon_s\":null";
    }
    if (pnp_metrics_valid) {
      stream << ",\"x_m\":" << pose.x_m << ",\"y_m\":" << pose.y_m
             << ",\"z_m\":" << pose.z_m
             << ",\"reprojection_rms_px\":"
             << pose.reprojection_rms_px;
    } else {
      stream << ",\"x_m\":null,\"y_m\":null,\"z_m\":null,"
                "\"reprojection_rms_px\":null";
    }
    if (coordinate_valid) {
      stream << ",\"odom_x_m\":" << pose.odom_x_m
             << ",\"odom_y_m\":" << pose.odom_y_m
             << ",\"odom_z_m\":" << pose.odom_z_m;
    } else {
      stream << ",\"odom_x_m\":null,\"odom_y_m\":null,\"odom_z_m\":null";
    }
    if (aim_valid) {
      stream << ",\"yaw_command_deg\":" << pose.yaw_command_deg
             << ",\"pitch_command_deg\":" << pose.pitch_command_deg
             << ",\"time_of_flight_s\":" << pose.time_of_flight_s
             << ",\"gravity_drop_m\":" << pose.gravity_drop_m
             << ",\"muzzle_odom_x_m\":" << pose.muzzle_odom_x_m
             << ",\"muzzle_odom_y_m\":" << pose.muzzle_odom_y_m
             << ",\"muzzle_odom_z_m\":" << pose.muzzle_odom_z_m;
    } else {
      stream << ",\"yaw_command_deg\":null,\"pitch_command_deg\":null,"
                "\"time_of_flight_s\":null,\"gravity_drop_m\":null,"
                "\"muzzle_odom_x_m\":null,\"muzzle_odom_y_m\":null,\"muzzle_odom_z_m\":null";
    }
    stream << ",\"candidate_count\":" << pose.candidate_count << '}';
  }
  stream << "]}";
  return stream.str();
}

void sendNotFound(Socket client) {
  sendText(client, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n"
                   "Connection: close\r\n\r\n");
}

void sendJsonDocument(Socket client, const std::string& body) {
  sendText(client, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                   "Cache-Control: no-store\r\nContent-Length: " +
                       std::to_string(body.size()) +
                       "\r\nConnection: close\r\n\r\n" + body);
}

void sendControlJson(Socket client, int status, const char* message) {
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
  const bool get_status = request_line.rfind("GET /api/status ", 0) == 0;
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
  } else if (get_status) {
    std::string telemetry;
    {
      std::lock_guard<std::mutex> lock(state->frame_mutex);
      telemetry = state->latest_telemetry_json;
    }
    sendJsonDocument(client, telemetry);
  } else if (!action.empty()) {
    const bool accepted = state->running.load() && state->control_handler &&
                          state->control_handler(action);
    sendControlJson(client, accepted ? 202 : 400,
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
  publish(bgr_frame, WebFrameTelemetry{});
}

void MjpegServer::publish(const cv::Mat& bgr_frame,
                          const WebFrameTelemetry& telemetry) {
  if (bgr_frame.empty()) return;
  std::vector<unsigned char> jpeg;
  const std::vector<int> parameters = {cv::IMWRITE_JPEG_QUALITY,
                                       state_->options.jpeg_quality};
  if (!cv::imencode(".jpg", bgr_frame, jpeg, parameters)) return;
  const std::string telemetry_json = telemetryJson(telemetry);
  {
    std::lock_guard<std::mutex> lock(state_->frame_mutex);
    state_->latest_jpeg = std::move(jpeg);
    state_->latest_telemetry_json = telemetry_json;
    ++state_->frame_sequence;
  }
  state_->frame_ready.notify_all();
}

std::string MjpegServer::url() const {
  return "http://" + state_->options.bind_address + ":" +
         std::to_string(state_->options.port) + "/";
}

}  // namespace yolo_detect
