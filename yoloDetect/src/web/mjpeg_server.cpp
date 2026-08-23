#include "web/mjpeg_server.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <optional>
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

// 在 Winsock 句柄有效时关闭它。
void closeSocket(Socket socket) {
  if (socket != kInvalidSocket) closesocket(socket);
}

class WinsockSession {
 public:
  // 为 HTTP 服务器生命周期初始化 Winsock。
  WinsockSession() {
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
      throw std::runtime_error("WSAStartup failed");
    }
  }
  // 释放进程级 Winsock 初始化引用。
  ~WinsockSession() { WSACleanup(); }
};
#else
using Socket = int;
constexpr Socket kInvalidSocket = -1;

// 在 POSIX 套接字有效时关闭它。
void closeSocket(Socket socket) {
  if (socket != kInvalidSocket) close(socket);
}
#endif

// 发送完整字节缓冲区，直到对端接收所有字节。
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

// 通过完整缓冲区发送助手发送 HTTP 头或文本正文。
bool sendText(Socket socket, const std::string& response) {
  return sendAll(socket, response.data(), response.size());
}

// 构建根路径返回的独立浏览器界面。
std::string htmlPage() {
  std::string page = R"(<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>YOLO Armor Debug</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #15191d; color: #eef3f6; font-family: system-ui, sans-serif; overflow: hidden; }
  header { padding: 14px 18px; background: #20282e; border-bottom: 1px solid #39464f; }
  h1 { margin: 0; font-size: 18px; font-weight: 600; }
  h2 { margin: 0; font-size: 15px; font-weight: 600; }
  main { display: grid; grid-template-columns: minmax(0, 1fr) clamp(560px, 44vw, 780px); height: calc(100vh - 50px); min-height: 0; }
  .video-pane { min-width: 0; min-height: 0; display: grid; place-items: center; padding: 12px; background: #090b0d; }
  .video-pane img { display: block; width: 100%; height: 100%; object-fit: contain; background: #090b0d; }
  .debug-pane { min-width: 0; min-height: 0; overflow-y: auto; padding: 0 14px 18px; border-left: 1px solid #39464f; background: #171d21; scrollbar-gutter: stable; }
  .debug-section { min-width: 0; border-bottom: 1px solid #39464f; padding-bottom: 10px; }
  .debug-section.pose { height: 260px; }
  .debug-section.predicted { height: 214px; }
  .debug-section.state { height: 154px; }
   .debug-section.observations { height: 628px; }
  p, #control-status, #pose-frame { color: #aab7c0; font-size: 14px; }
  .pose-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; min-height: 42px; padding-top: 12px; }
  .frame-meta { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 14px; color: #aab7c0; font-size: 14px; }
   .pose-table-wrap { height: calc(100% - 48px); overflow: auto; border-top: 1px solid #39464f; border-bottom: 1px solid #39464f; scrollbar-gutter: stable; }
   .observation-table-wrap { height: 244px; overflow: auto; border-top: 1px solid #39464f; border-bottom: 1px solid #39464f; scrollbar-gutter: stable; }
   .xy-map { height: 310px; margin-top: 12px; border: 1px solid #39464f; background: #111619; }
   .xy-map svg { display: block; width: 100%; height: 100%; }
   .xy-map text { font-family: system-ui, sans-serif; font-size: 11px; fill: #aab7c0; }
   .xy-map .axis { stroke: #596771; stroke-width: 1; }
   .xy-map .grid { stroke: #2b353c; stroke-width: 1; }
   .xy-map .predicted { fill: #2f9d76; stroke: #d8f5e8; stroke-width: 1.5; }
   .xy-map .observed { fill: #15191d; stroke: #f0c75e; stroke-width: 2; }
   .xy-map .camera { fill: #d4655f; stroke: #ffe0d3; stroke-width: 1.5; }
   .xy-map .legend { font-size: 11px; fill: #d8e1e6; }
  table { width: max-content; min-width: 100%; table-layout: fixed; border-collapse: collapse; font-variant-numeric: tabular-nums; }
  th, td { height: 36px; padding: 7px 9px; text-align: right; white-space: nowrap; border-bottom: 1px solid #2b353c; font-size: 12px; }
  th, td { min-width: 82px; max-width: 180px; overflow: hidden; text-overflow: ellipsis; }
  th:first-child, td:first-child { min-width: 64px; }
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:last-child, td:last-child { text-align: left; }
  th { color: #aab7c0; font-weight: 500; }
  tr.valid td { color: #d8f5e8; }
  tr.warning td { color: #f4d58b; }
  tr.invalid td { color: #efb6a8; }
  .control-panel { margin: 12px 0 0; border-top: 1px solid #39464f; }
  .control-row { display: grid; grid-template-columns: 126px repeat(auto-fit, minmax(118px, 1fr)); gap: 10px; align-items: center; padding: 10px 0; border-bottom: 1px solid #2b353c; }
  .control-label { color: #aab7c0; font-size: 13px; }
  .spin-input { min-width: 0; width: 100%; box-sizing: border-box; min-height: 38px; border: 1px solid #52616b; border-radius: 4px; background: #1b2227; color: #eef3f6; font: inherit; padding: 0 10px; }
  .spin-input:focus { outline: 2px solid #2f9d76; outline-offset: 1px; }
  .tuning-panel { padding-bottom: 18px; }
  .tuning-heading { padding: 14px 0 6px; }
  .tuning-group { color: #7fcab0; font-size: 12px; padding: 10px 0 2px; border-bottom: 1px solid #39464f; }
  .ekf-input { min-width: 0; width: 100%; min-height: 34px; border: 1px solid #52616b; border-radius: 4px; background: #1b2227; color: #eef3f6; font: inherit; padding: 0 8px; }
  .ekf-input:focus { outline: 2px solid #2f9d76; outline-offset: 1px; }
  .ekf-unit { color: #82919a; font-size: 12px; white-space: nowrap; }
  .ekf-apply { min-height: 34px; padding: 0 9px; }
  @media (max-width: 560px) { .control-row { grid-template-columns: repeat(2, minmax(0, 1fr)); } .control-row > .control-label:first-child { grid-column: 1 / -1; } }
  .gimbal-state { grid-column: 1 / -1; color: #aab7c0; font-size: 13px; }
  button.follow { border-color: #d8a43a; }
  button.follow.active { background: #66511e; border-color: #f0c75e; }
  button.fire { background: #612c2c; border-color: #d4655f; }
  button { min-height: 38px; border: 1px solid #52616b; border-radius: 4px; background: #273139; color: #eef3f6; cursor: pointer; font: inherit; }
  button:hover { background: #34434e; }
  button.scene { border-color: #2f9d76; }
  button.stop { border-color: #c65b56; }
  @media (max-width: 1050px) {
    body { overflow: auto; }
    main { display: block; height: auto; }
    .video-pane { height: min(62vh, 720px); }
    .debug-pane { border-left: 0; border-top: 1px solid #39464f; overflow: visible; }
  }
</style>
</head>
<body>
<header><h1>YOLO Armor Debug Stream</h1></header>
<main>
<div class="video-pane"><img src="/stream.mjpg" alt="Waiting for detector frames"></div>
<aside class="debug-pane" aria-label="Detector and EKF diagnostics">
<section class="debug-section pose" aria-labelledby="pose-title">
  <div class="pose-heading">
    <h2 id="pose-title">Pose and Aim</h2>
    <div class="frame-meta">
      <span id="coordinate-state">Coordinates unavailable</span>
      <span id="tracker-state">EKF unavailable</span>
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
<section class="debug-section predicted" aria-labelledby="ekf-title">
  <div class="pose-heading"><h2 id="ekf-title">EKF Predicted Armor Centers (T m)</h2></div>
  <div class="pose-table-wrap">
    <table>
      <thead><tr><th>Armor</th><th>T X m</th><th>T Y m</th><th>T Z m</th></tr></thead>
      <tbody id="ekf-rows"><tr><td colspan="4">No active EKF state</td></tr></tbody>
    </table>
  </div>
</section>
<section class="debug-section state" aria-labelledby="ekf-state-title">
  <div class="pose-heading"><h2 id="ekf-state-title">EKF State (T)</h2></div>
  <div class="pose-table-wrap">
    <table>
      <thead><tr><th>C X m</th><th>C Y m</th><th>C Z m</th><th>V X m/s</th><th>V Y m/s</th><th>V Z m/s</th><th>Theta deg</th><th>Omega deg/s</th><th>r0 (E0/E2) m</th><th>dr m</th><th>r1 (E1/E3) m</th><th>dz m</th><th>Slot</th><th>NIS</th><th>Hits</th><th>Misses</th></tr></thead>
      <tbody id="ekf-state-rows"><tr><td colspan="16">No active EKF state</td></tr></tbody>
    </table>
  </div>
</section>
<section class="debug-section observations" aria-labelledby="ekf-observation-title">
  <div class="pose-heading"><h2 id="ekf-observation-title">EKF Measurements and Association (T)</h2></div>
  <div class="observation-table-wrap">
    <table>
      <thead><tr><th>Obs</th><th>T X m</th><th>T Y m</th><th>T Z m</th><th>Camera range m</th><th>Yaw valid</th><th>Yaw used</th><th>Measured yaw deg</th><th>IPPE pitch deg</th><th>Predicted yaw deg</th><th>Yaw std deg</th><th>PnP RMS px</th><th>Confidence</th><th>Keypoint Q</th><th>View Q</th><th>Color ID</th><th>Number ID</th><th>Slot</th><th>NIS</th><th>Pred x</th><th>Pred y</th><th>Pred z</th><th>dx</th><th>dy</th><th>dz</th><th>Radial d</th><th>Yaw d deg</th></tr></thead>
      <tbody id="ekf-observation-rows"><tr><td colspan="27">No EKF measurements</td></tr></tbody>
    </table>
  </div>
  <div class="xy-map" aria-label="EKF armor and camera XY projection">
    <svg id="ekf-xy-map" viewBox="0 0 560 310" role="img" aria-label="Armor and camera positions in tracker XY coordinates"></svg>
  </div>
</section>
<section class="control-panel" aria-label="Simulator controls">
  <div class="control-row"><span class="control-label">Scene</span><button class="scene" data-action="scene-shooting-range">Shooting Range</button><button class="scene" data-action="scene-energy">Energy</button><button data-action="reset">Reset Scene</button></div>
  <div class="control-row"><span class="control-label">Vehicle motion</span><button class="stop" data-action="motion-stop">Stop Vehicle</button><button data-action="motion-linear">Linear</button><button data-action="motion-spin">Spin</button><button data-action="motion-linear-spin">Linear + Spin</button></div>
  <div class="control-row"><span class="control-label">Linear speed</span><button data-action="speed-down">Speed -</button><button data-action="speed-up">Speed +</button><span id="linear-speed-state" class="control-label">0.00 m/s</span></div>
  <div class="control-row"><label class="control-label" for="spin-speed">Spin speed</label><input id="spin-speed" class="spin-input" type="number" min="0" max="720" step="1" inputmode="decimal" aria-label="Spin speed in degrees per second"><button id="spin-speed-apply" type="button">Apply deg/s</button><span id="spin-speed-state" class="control-label">0.0 deg/s</span></div>
  <div class="control-row"><span class="control-label">Gimbal aim</span><button id="follow-button" class="follow" data-action="gimbal-follow-toggle" aria-pressed="false">Start Predictive Follow</button><button id="fire-button" class="fire" data-action="gimbal-fire">Fire Once</button><span id="gimbal-state" class="control-label">Gimbal idle</span></div>
</section>
<section class="control-panel tuning-panel" aria-label="EKF tuning controls">
  <div class="pose-heading tuning-heading"><h2>EKF Tuning</h2><span class="frame-meta">Live Q/R and gates; initial uncertainties apply after reset</span><button data-action="ekf-reset" type="button">Reset EKF Track</button></div>
  <div id="ekf-tuning-controls"></div>
</section>
<section class="control-panel tuning-panel" aria-label="Projection debug controls">
  <div class="pose-heading tuning-heading"><h2>Projection Debug</h2><span class="frame-meta">Does not modify EKF</span><button data-action="projection-debug-toggle" type="button">Enable</button><button data-action="projection-debug-anchor-toggle" type="button">Anchor: observed</button></div>
  <div id="projection-debug-controls"></div>
</section>
<p id="control-status">Controls are sent to the detector command queue.</p>
</aside>
</main>
)";
  page += R"(<script>
const controlStatus = document.getElementById('control-status');
const followButton = document.getElementById('follow-button');
const fireButton = document.getElementById('fire-button');
const gimbalState = document.getElementById('gimbal-state');
const linearSpeedState = document.getElementById('linear-speed-state');
const spinSpeedInput = document.getElementById('spin-speed');
const spinSpeedState = document.getElementById('spin-speed-state');
const spinSpeedApply = document.getElementById('spin-speed-apply');
const ekfTuningControls = document.getElementById('ekf-tuning-controls');
const projectionDebugControls = document.getElementById('projection-debug-controls');
async function sendControl(action, button) {
  if (button) button.disabled = true;
  controlStatus.textContent = 'Sending ' + action + '...';
  try {
    const response = await fetch('/api/control?action=' + encodeURIComponent(action), { cache: 'no-store' });
    const result = await response.json();
    controlStatus.textContent = result.message;
  } catch (_) {
    controlStatus.textContent = 'Control request failed.';
  } finally {
    if (button) button.disabled = false;
  }
}
for (const button of document.querySelectorAll('[data-action]')) {
  button.addEventListener('click', () => sendControl(button.dataset.action, button));
}
spinSpeedApply.addEventListener('click', () => {
  const speed = Number(spinSpeedInput.value);
  if (!Number.isFinite(speed) || speed < 0 || speed > 720) {
    controlStatus.textContent = 'Spin speed must be between 0 and 720 deg/s.';
    return;
  }
  sendControl('spin-speed:' + speed, spinSpeedApply);
});

const ekfTuningFields = [
  ['Initial uncertainty', 'initial_position_std_m', 'Initial position std', 'm', 0, 2, 0.01],
  ['Initial uncertainty', 'initial_velocity_std_mps', 'Initial velocity std', 'm/s', 0, 10, 0.01],
  ['Initial uncertainty', 'initial_theta_std_rad', 'Initial theta std', 'rad', 0.001, 3.2, 0.01],
  ['Initial uncertainty', 'initial_omega_std_rad_s', 'Initial omega std', 'rad/s', 0.001, 30, 0.01],
  ['Initial uncertainty', 'initial_geometry_std_m', 'Initial geometry std', 'm', 0.001, 1, 0.01],
  ['Process noise Q', 'q_linear_acceleration', 'Linear acceleration density', 'm^2/s^3', 0, 100, 0.001],
  ['Process noise Q', 'q_angular_acceleration', 'Angular acceleration density', 'rad^2/s^3', 0, 100, 0.001],
  ['Process noise Q', 'q_geometry', 'Geometry random walk', 'm^2/s', 0, 10, 0.001],
  ['Measurement noise R', 'position_std_xy_m', 'Position XY std', 'm', 0.001, 1, 0.001],
  ['Measurement noise R', 'position_std_z_m', 'Position Z std', 'm', 0.001, 2, 0.001],
  ['Measurement noise R', 'yaw_std_rad', 'Yaw std', 'rad', 0.001, 3.2, 0.001],
  ['Measurement noise R', 'reprojection_rms_scale', 'Reprojection RMS scale', '', 0, 10, 0.001],
  ['Measurement noise R', 'range_noise_scale_per_m', 'Range noise scale', 'per m', 0, 2, 0.001],
  ['Measurement noise R', 'minimum_quality', 'Minimum quality', '', 0.001, 1, 0.001],
  ['Multi-armor weighting', 'single_armor_position_variance_scale', 'Single-plate variance scale', 'x', 1, 1000, 1],
  ['Multi-armor weighting', 'association_position_variance_scale', 'Association position scale', 'x', 1, 2000, 1],
  ['Multi-armor weighting', 'maximum_multi_armor_position_residual_m', 'Max multi-plate residual', 'm', 0.001, 2, 0.001],
  ['Slot association', 'maximum_yaw_update_innovation_rad', 'Yaw used gate (EKF update)', 'rad', 0.001, 3.2, 0.001],
  ['Slot association', 'maximum_yaw_association_innovation_rad', 'Yaw association gate', 'rad', 0.001, 3.2, 0.001],
  ['Slot association', 'yaw_phase_cost_std_rad', 'Yaw phase cost std', 'rad', 0.001, 3.2, 0.001],
  ['Slot association', 'adjacent_slot_penalty', 'Adjacent slot penalty', '', 0, 100, 0.01],
  ['Slot association', 'opposite_slot_penalty', 'Opposite slot penalty', '', 0, 100, 0.01],
  ['Slot association', 'minimum_visibility_cosine', 'Minimum visibility cosine', '', -1, 1, 0.01],
  ['Geometry gate', 'geometry_yaw_consistency_rad', 'Geometry yaw consistency', 'rad', 0.001, 3.2, 0.001],
  ['Geometry gate', 'geometry_minimum_baseline_m', 'Minimum plate baseline', 'm', 0.001, 2, 0.001],
  ['Geometry gate', 'geometry_confirming_frames', 'Geometry confirming frames', 'frames', 1, 100, 1],
  ['NIS gate', 'nis_gate_3d', 'Position NIS gate', '', 0.001, 100, 0.001],
  ['NIS gate', 'nis_gate_4d', 'Position+yaw NIS gate', '', 0.001, 100, 0.001],
  ['Angular limits', 'maximum_angular_speed_rad_s', 'Maximum angular speed', 'rad/s', 0.01, 30, 0.01],
  ['Angular limits', 'maximum_omega_correction_rad_s', 'Maximum omega correction/frame', 'rad/s', 0.001, 5, 0.001],
  ['Yaw validity', 'yaw_max_reprojection_rms_px', 'Yaw max reprojection RMS', 'px', 0.001, 20, 0.001],
  ['Yaw validity', 'yaw_max_std_rad', 'Yaw max standard deviation', 'rad', 0.001, 3.2, 0.001],
  ['Yaw validity', 'yaw_min_facing_cosine', 'Yaw minimum facing cosine', '', -1, 1, 0.001],
  ['Yaw validity', 'yaw_min_opposite_margin_px', 'Yaw opposite-solution margin', 'px', 0, 20, 0.001]
];
let tuningGroup = '';
for (const [group, name, label, unit, min, max, step] of ekfTuningFields) {
  if (group !== tuningGroup) {
    const heading = document.createElement('div');
    heading.className = 'tuning-group';
    heading.textContent = group;
    ekfTuningControls.append(heading);
    tuningGroup = group;
  }
  const row = document.createElement('div');
  row.className = 'control-row';
  row.innerHTML = '<label class="control-label" for="ekf-' + name + '">' + label + '</label>' +
    '<input id="ekf-' + name + '" class="ekf-input" type="number" min="' + min + '" max="' + max + '" step="' + step + '" data-ekf-name="' + name + '">' +
    '<span class="ekf-unit">' + unit + '</span><button class="ekf-apply" type="button">Apply</button>';
  const input = row.querySelector('input');
  const button = row.querySelector('button');
  button.addEventListener('click', () => {
    const value = Number(input.value);
    if (!Number.isFinite(value) || value < min || value > max) {
      controlStatus.textContent = label + ' is out of range.';
      return;
    }
    sendControl('ekf-param:' + name + ':' + value, button);
  });
  ekfTuningControls.append(row);
}

const projectionDebugFields = [
  ['center_x_T_m', 'Debug center cx', 'm', -20, 20, 0.001],
  ['center_y_T_m', 'Debug center cy', 'm', -20, 20, 0.001],
  ['center_z_T_m', 'Debug center cz', 'm', -20, 20, 0.001],
  ['theta_rad', 'Debug theta', 'rad', -100, 100, 0.001],
  ['radius_even_m', 'Debug r0', 'm', 0.05, 0.5, 0.001],
  ['radius_odd_delta_m', 'Debug dr (fixed)', 'm', 0, 0, 0.001],
  ['height_odd_delta_m', 'Debug dz (fixed)', 'm', 0, 0, 0.001]
];
for (const [name, label, unit, min, max, step] of projectionDebugFields) {
  const row = document.createElement('div');
  row.className = 'control-row';
  row.innerHTML = '<label class="control-label" for="projection-' + name + '">' + label + '</label>' +
    '<input id="projection-' + name + '" class="ekf-input" type="number" min="' + min + '" max="' + max + '" step="' + step + '" data-projection-name="' + name + '">' +
    '<span class="ekf-unit">' + unit + '</span><button class="ekf-apply" type="button">Apply</button>';
  const input = row.querySelector('input');
  const button = row.querySelector('button');
  button.addEventListener('click', () => {
    const value = Number(input.value);
    if (!Number.isFinite(value) || value < min || value > max) {
      controlStatus.textContent = label + ' is out of range.';
      return;
    }
    sendControl('projection-debug-param:' + name + ':' + value, button);
  });
  projectionDebugControls.append(row);
}

)";
  page += R"(const poseRows = document.getElementById('pose-rows');
const ekfRows = document.getElementById('ekf-rows');
const ekfStateRows = document.getElementById('ekf-state-rows');
const ekfObservationRows = document.getElementById('ekf-observation-rows');
const poseFrame = document.getElementById('pose-frame');
const coordinateState = document.getElementById('coordinate-state');
const trackerState = document.getElementById('tracker-state');
const cell = (text) => {
  const element = document.createElement('td');
  element.textContent = text;
  return element;
};
const metric = (value, digits) => typeof value === 'number' ? value.toFixed(digits) : '-';
const trackerXyMap = document.getElementById('ekf-xy-map');
const svgElement = (name, attributes = {}) => {
  const element = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  return element;
};
const finiteXy = (point) => Number.isFinite(point.x) && Number.isFinite(point.y);
function renderTrackerXyMap(state) {
  const predicted = (state.predicted_armors || [])
    .map((armor) => ({ x: armor.x_T_m, y: armor.y_T_m, slot: armor.armor_slot }))
    .filter(finiteXy);
  const observed = (state.tracker_measurements || [])
    .map((measurement, index) => ({ x: measurement.x_T_m, y: measurement.y_T_m, index }))
    .filter(finiteXy);
  const cameraMeasurement = (state.tracker_measurements || []).find(
    (measurement) => measurement.has_exposure_camera_geometry &&
      Number.isFinite(measurement.camera_x_T_m) && Number.isFinite(measurement.camera_y_T_m));
  const camera = cameraMeasurement
    ? { x: cameraMeasurement.camera_x_T_m, y: cameraMeasurement.camera_y_T_m }
    : null;
  const points = [...predicted, ...observed];
  if (camera) points.push(camera);
  trackerXyMap.replaceChildren();
  if (points.length === 0) {
    const message = svgElement('text', { x: 280, y: 155, 'text-anchor': 'middle' });
    message.textContent = 'No tracker XY data';
    trackerXyMap.append(message);
    return;
  }

  const left = 60, top = 24, width = 480, height = 236;
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const centerX = (Math.min(...xs) + Math.max(...xs)) * 0.5;
  const centerY = (Math.min(...ys) + Math.max(...ys)) * 0.5;
  const metersPerPixel = Math.max(
    (Math.max(...xs) - Math.min(...xs)) / width,
    (Math.max(...ys) - Math.min(...ys)) / height,
    0.40 / Math.min(width, height));
  const halfX = metersPerPixel * width * 0.5 + 0.08;
  const halfY = metersPerPixel * height * 0.5 + 0.08;
  const xMin = centerX - halfX, xMax = centerX + halfX;
  const yMin = centerY - halfY, yMax = centerY + halfY;
  const mapX = (x) => left + (x - xMin) / (xMax - xMin) * width;
  const mapY = (y) => top + (yMax - y) / (yMax - yMin) * height;

  trackerXyMap.append(svgElement('rect', { x: left, y: top, width, height, fill: 'none', class: 'axis' }));
  for (let tick = 0; tick <= 4; ++tick) {
    const x = left + width * tick / 4;
    const y = top + height * tick / 4;
    trackerXyMap.append(svgElement('line', { x1: x, y1: top, x2: x, y2: top + height, class: 'grid' }));
    trackerXyMap.append(svgElement('line', { x1: left, y1: y, x2: left + width, y2: y, class: 'grid' }));
    const xLabel = svgElement('text', { x, y: top + height + 18, 'text-anchor': 'middle' });
    xLabel.textContent = (xMin + (xMax - xMin) * tick / 4).toFixed(2);
    trackerXyMap.append(xLabel);
    const yLabel = svgElement('text', { x: left - 8, y: y + 4, 'text-anchor': 'end' });
    yLabel.textContent = (yMax - (yMax - yMin) * tick / 4).toFixed(2);
    trackerXyMap.append(yLabel);
  }
  const xAxis = svgElement('text', { x: left + width * 0.5, y: 302, 'text-anchor': 'middle' });
  xAxis.textContent = 'T X (m)';
  trackerXyMap.append(xAxis);
  const yAxis = svgElement('text', { x: 14, y: top + height * 0.5, transform: 'rotate(-90 14 ' + (top + height * 0.5) + ')', 'text-anchor': 'middle' });
  yAxis.textContent = 'T Y (m)';
  trackerXyMap.append(yAxis);

  for (const armor of predicted) {
    const x = mapX(armor.x), y = mapY(armor.y);
    trackerXyMap.append(svgElement('circle', { cx: x, cy: y, r: 6, class: 'predicted' }));
    const label = svgElement('text', { x: x + 9, y: y - 8, class: 'legend' });
    label.textContent = 'E' + armor.slot;
    trackerXyMap.append(label);
  }
  for (const measurement of observed) {
    const x = mapX(measurement.x), y = mapY(measurement.y);
    trackerXyMap.append(svgElement('path', { d: 'M ' + x + ' ' + (y - 7) + ' L ' + (x + 7) + ' ' + y + ' L ' + x + ' ' + (y + 7) + ' L ' + (x - 7) + ' ' + y + ' Z', class: 'observed' }));
    const label = svgElement('text', { x: x + 9, y: y + 16, class: 'legend' });
    label.textContent = 'M' + (measurement.index + 1);
    trackerXyMap.append(label);
  }
  if (camera) {
    const x = mapX(camera.x), y = mapY(camera.y);
    trackerXyMap.append(svgElement('path', { d: 'M ' + x + ' ' + (y - 8) + ' L ' + (x + 7) + ' ' + (y + 6) + ' L ' + (x - 7) + ' ' + (y + 6) + ' Z', class: 'camera' }));
    const label = svgElement('text', { x: x + 9, y: y - 10, class: 'legend' });
    label.textContent = 'Camera';
    trackerXyMap.append(label);
  }
  const legend = svgElement('text', { x: left + 4, y: top + 15, class: 'legend' });
  legend.textContent = 'filled: predicted  diamond: measurement  triangle: camera';
  trackerXyMap.append(legend);
}
// 独立于 MJPEG 图像流刷新遥测数据。
async function refreshPose() {
  if (refreshPose.pending) return;
  refreshPose.pending = true;
  try {
    const response = await fetch('/api/status', { cache: 'no-store' });
    if (!response.ok) return;
    const state = await response.json();
    renderTrackerXyMap(state);
    poseFrame.textContent = 'Frame ' + state.source_sequence;
    coordinateState.textContent = state.coordinate_valid
      ? 'Odom ready | camera offset error ' + metric(state.camera_position_error_m, 4) + ' m'
      : state.coordinate_status;
    const yawDiagnostic = state.tracker_yaw_rms_px === null
      ? state.tracker_yaw_status
      : state.tracker_yaw_status + ' | RMS ' + metric(state.tracker_yaw_rms_px, 2) + ' px';
    trackerState.textContent = 'EKF ' + state.tracker_state +
      ' | yaw ' + state.reliable_yaw_count + '/' + state.tracker_observation_count +
      ' | ' + yawDiagnostic;
    followButton.textContent =
      state.gimbal_following ? 'Stop Predictive Follow' : 'Start Predictive Follow';
    followButton.classList.toggle('active', state.gimbal_following);
    followButton.setAttribute('aria-pressed', String(state.gimbal_following));
    fireButton.disabled = state.fire_pending;
    const staticTarget = state.static_target_valid
      ? ' | target O=(' + metric(state.static_target_odom_x_m, 3) + ', ' +
        metric(state.static_target_odom_y_m, 3) + ', ' +
        metric(state.static_target_odom_z_m, 3) + ')'
      : '';
    gimbalState.textContent = state.gimbal_status + staticTarget;
    linearSpeedState.textContent = metric(state.vehicle_speed_mps, 2) + ' m/s';
    spinSpeedState.textContent = metric(state.spin_speed_deg_s, 1) + ' deg/s';
    if (document.activeElement !== spinSpeedInput) {
      spinSpeedInput.value = metric(state.spin_speed_deg_s, 1);
    }
    const tuning = state.ekf_tuning || {};
    for (const input of ekfTuningControls.querySelectorAll('input[data-ekf-name]')) {
      if (document.activeElement !== input && typeof tuning[input.dataset.ekfName] === 'number') {
        input.value = tuning[input.dataset.ekfName];
      }
    }
    const debug = state.projection_debug || {};
    const debugToggle = document.querySelector('[data-action="projection-debug-toggle"]');
    const debugAnchorToggle = document.querySelector('[data-action="projection-debug-anchor-toggle"]');
    if (debugToggle) debugToggle.textContent = debug.enabled ? 'Disable' : 'Enable';
    if (debugAnchorToggle) debugAnchorToggle.textContent = debug.anchor_observed ? 'Anchor: observed' : 'Anchor: manual';
    for (const input of projectionDebugControls.querySelectorAll('input[data-projection-name]')) {
      if (document.activeElement !== input && typeof debug[input.dataset.projectionName] === 'number') {
        input.value = debug[input.dataset.projectionName];
      }
    }
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
    const predictedRows = [];
    for (const armor of state.predicted_armors || []) {
      const row = document.createElement('tr');
      row.className = 'valid';
      row.append(cell('E' + armor.armor_slot));
      row.append(cell(metric(armor.x_T_m, 3)));
      row.append(cell(metric(armor.y_T_m, 3)));
      row.append(cell(metric(armor.z_T_m, 3)));
      predictedRows.push(row);
    }
    if (predictedRows.length === 0) {
      const row = document.createElement('tr');
      const empty = cell('No active EKF state');
      empty.colSpan = 4;
      row.append(empty);
      predictedRows.push(row);
    }
    ekfRows.replaceChildren(...predictedRows);
    const stateRows = [];
    if (state.tracker_has_state) {
      const row = document.createElement('tr');
      row.className = 'valid';
      row.append(cell(metric(state.tracker_center_x_T_m, 3)));
      row.append(cell(metric(state.tracker_center_y_T_m, 3)));
      row.append(cell(metric(state.tracker_center_z_T_m, 3)));
      row.append(cell(metric(state.tracker_velocity_x_T_mps, 3)));
      row.append(cell(metric(state.tracker_velocity_y_T_mps, 3)));
      row.append(cell(metric(state.tracker_velocity_z_T_mps, 3)));
      row.append(cell(metric(state.tracker_theta_rad * 180 / Math.PI, 2)));
      row.append(cell(metric(state.tracker_omega_rad_s * 180 / Math.PI, 2)));
      row.append(cell(metric(state.tracker_radius_even_m, 3)));
      row.append(cell(metric(state.tracker_radius_odd_delta_m, 3)));
      row.append(cell(metric(state.tracker_radius_odd_m, 3)));
      row.append(cell(metric(state.tracker_height_odd_delta_m, 3)));
      row.append(cell(state.tracker_association_valid ? 'E' + state.tracker_associated_slot : '-'));
      row.append(cell(state.tracker_nis_valid ? metric(state.tracker_nis, 3) : '-'));
      row.append(cell(String(state.tracker_consecutive_hits)));
      row.append(cell(String(state.tracker_consecutive_misses)));
      stateRows.push(row);
    } else {
      const row = document.createElement('tr');
      const empty = cell('No active EKF state');
      empty.colSpan = 16;
      row.append(empty);
      stateRows.push(row);
    }
    ekfStateRows.replaceChildren(...stateRows);
)";
  page += R"(    const observationRows = [];
    for (const [index, observation] of (state.tracker_measurements || []).entries()) {
      const row = document.createElement('tr');
      row.className = observation.has_inward_yaw ? 'valid' : 'warning';
      row.append(cell('#' + (index + 1)));
      row.append(cell(metric(observation.x_T_m, 3)));
      row.append(cell(metric(observation.y_T_m, 3)));
      row.append(cell(metric(observation.z_T_m, 3)));
      row.append(cell(metric(observation.camera_range_m, 3)));
      row.append(cell(observation.has_inward_yaw ? 'yes' : 'no'));
      row.append(cell(observation.yaw_used ? 'yes' : 'no'));
      row.append(cell(observation.has_raw_inward_yaw
        ? metric(observation.inward_yaw_rad * 180 / Math.PI, 2) : '-'));
      row.append(cell(observation.has_pnp_inward_pitch_T
        ? metric(observation.pnp_inward_pitch_rad * 180 / Math.PI, 2) : '-'));
      row.append(cell(observation.association_valid ? metric(observation.predicted_yaw_rad * 180 / Math.PI, 2) : '-'));
      row.append(cell(observation.has_inward_yaw ? metric(observation.yaw_std_rad * 180 / Math.PI, 2) : '-'));
      row.append(cell(metric(observation.reprojection_rms_px, 2)));
      row.append(cell(metric(observation.confidence, 3)));
      row.append(cell(metric(observation.keypoint_quality, 3)));
      row.append(cell(metric(observation.view_quality, 3)));
      row.append(cell(String(observation.color_id)));
      row.append(cell(String(observation.number_id)));
      row.append(cell(observation.association_valid ? 'E' + observation.associated_slot : '-'));
      row.append(cell(observation.association_valid ? metric(observation.nis, 3) : '-'));
      row.append(cell(observation.association_valid ? metric(observation.predicted_x_T_m, 3) : '-'));
      row.append(cell(observation.association_valid ? metric(observation.predicted_y_T_m, 3) : '-'));
      row.append(cell(observation.association_valid ? metric(observation.predicted_z_T_m, 3) : '-'));
      row.append(cell(observation.association_valid ? metric(observation.innovation_x_T_m, 3) : '-'));
      row.append(cell(observation.association_valid ? metric(observation.innovation_y_T_m, 3) : '-'));
      row.append(cell(observation.association_valid ? metric(observation.innovation_z_T_m, 3) : '-'));
      row.append(cell(observation.association_valid ? metric(observation.radial_innovation_m, 3) : '-'));
      row.append(cell(observation.association_valid && observation.has_inward_yaw
        ? metric(observation.yaw_innovation_rad * 180 / Math.PI, 2) : '-'));
      observationRows.push(row);
    }
    if (observationRows.length === 0) {
      const row = document.createElement('tr');
      const empty = cell('No EKF measurements');
      empty.colSpan = 27;
      row.append(empty);
      observationRows.push(row);
    }
    ekfObservationRows.replaceChildren(...observationRows);
  } catch (_) {
  } finally {
    refreshPose.pending = false;
  }
}
refreshPose.pending = false;
refreshPose();
setInterval(refreshPose, 100);
</script>
</body>
</html>
)";
  return page;
}

}  // namespace

struct MjpegServer::State {
  WebServerOptions options;
  Socket listener = kInvalidSocket;
  std::atomic<bool> running{true};
  std::thread accept_thread;
  std::mutex frame_mutex;
  std::mutex publish_mutex;
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
  std::chrono::steady_clock::time_point last_image_publish{};
  ControlHandler control_handler;
#ifdef _WIN32
  std::unique_ptr<WinsockSession> winsock;
#endif
};

namespace {

// 转义可能使 JSON 字符串失效的字符。
std::string jsonEscape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char character : value) {
    if (character == '\\' || character == '"') escaped.push_back('\\');
    escaped.push_back(character);
  }
  return escaped;
}

// 将最新图像、位姿和云台遥测序列化为状态 API 数据。
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
         << ",\"scene\":\"" << jsonEscape(telemetry.scene)
         << "\",\"motion\":\"" << jsonEscape(telemetry.motion)
         << "\",\"vehicle_speed_mps\":" << telemetry.vehicle_speed_mps
         << ",\"spin_speed_deg_s\":" << telemetry.spin_speed_deg_s
         << ",\"ekf_tuning\":{\"initial_position_std_m\":"
         << telemetry.ekf_tuning.initial_position_std_m
         << ",\"initial_velocity_std_mps\":"
         << telemetry.ekf_tuning.initial_velocity_std_mps
         << ",\"initial_theta_std_rad\":"
         << telemetry.ekf_tuning.initial_theta_std_rad
         << ",\"initial_omega_std_rad_s\":"
         << telemetry.ekf_tuning.initial_omega_std_rad_s
         << ",\"initial_geometry_std_m\":"
         << telemetry.ekf_tuning.initial_geometry_std_m
         << ",\"q_linear_acceleration\":"
         << telemetry.ekf_tuning.q_linear_acceleration
         << ",\"q_angular_acceleration\":"
         << telemetry.ekf_tuning.q_angular_acceleration
         << ",\"q_geometry\":" << telemetry.ekf_tuning.q_geometry
         << ",\"position_std_xy_m\":"
         << telemetry.ekf_tuning.position_std_xy_m
         << ",\"position_std_z_m\":"
         << telemetry.ekf_tuning.position_std_z_m
         << ",\"yaw_std_rad\":" << telemetry.ekf_tuning.yaw_std_rad
         << ",\"reprojection_rms_scale\":"
         << telemetry.ekf_tuning.reprojection_rms_scale
         << ",\"range_noise_scale_per_m\":"
         << telemetry.ekf_tuning.range_noise_scale_per_m
         << ",\"minimum_quality\":"
         << telemetry.ekf_tuning.minimum_quality
         << ",\"single_armor_position_variance_scale\":"
         << telemetry.ekf_tuning.single_armor_position_variance_scale
         << ",\"association_position_variance_scale\":"
         << telemetry.ekf_tuning.association_position_variance_scale
         << ",\"maximum_multi_armor_position_residual_m\":"
         << telemetry.ekf_tuning.maximum_multi_armor_position_residual_m
         << ",\"maximum_yaw_update_innovation_rad\":"
         << telemetry.ekf_tuning.maximum_yaw_update_innovation_rad
         << ",\"maximum_yaw_association_innovation_rad\":"
         << telemetry.ekf_tuning.maximum_yaw_association_innovation_rad
         << ",\"yaw_phase_cost_std_rad\":"
         << telemetry.ekf_tuning.yaw_phase_cost_std_rad
         << ",\"adjacent_slot_penalty\":"
         << telemetry.ekf_tuning.adjacent_slot_penalty
         << ",\"opposite_slot_penalty\":"
         << telemetry.ekf_tuning.opposite_slot_penalty
         << ",\"minimum_visibility_cosine\":"
         << telemetry.ekf_tuning.minimum_visibility_cosine
         << ",\"geometry_yaw_consistency_rad\":"
         << telemetry.ekf_tuning.geometry_yaw_consistency_rad
         << ",\"geometry_minimum_baseline_m\":"
         << telemetry.ekf_tuning.geometry_minimum_baseline_m
         << ",\"geometry_confirming_frames\":"
         << telemetry.ekf_tuning.geometry_confirming_frames
         << ",\"nis_gate_3d\":" << telemetry.ekf_tuning.nis_gate_3d
         << ",\"nis_gate_4d\":" << telemetry.ekf_tuning.nis_gate_4d
         << ",\"maximum_angular_speed_rad_s\":"
         << telemetry.ekf_tuning.maximum_angular_speed_rad_s
         << ",\"maximum_omega_correction_rad_s\":"
         << telemetry.ekf_tuning.maximum_omega_correction_rad_s
         << ",\"yaw_max_reprojection_rms_px\":"
         << telemetry.ekf_tuning.yaw_max_reprojection_rms_px
         << ",\"yaw_max_std_rad\":" << telemetry.ekf_tuning.yaw_max_std_rad
         << ",\"yaw_min_facing_cosine\":"
         << telemetry.ekf_tuning.yaw_min_facing_cosine
         << ",\"yaw_min_opposite_margin_px\":"
         << telemetry.ekf_tuning.yaw_min_opposite_margin_px << "}"
         << ",\"projection_debug\":{"
         << "\"enabled\":"
         << (telemetry.projection_debug.enabled ? "true" : "false")
         << ",\"anchor_observed\":"
         << (telemetry.projection_debug.anchor_observed ? "true" : "false")
         << ",\"has_reference\":"
         << (telemetry.projection_debug.has_reference ? "true" : "false")
         << ",\"center_x_T_m\":"
         << telemetry.projection_debug.center_x_T_m
         << ",\"center_y_T_m\":"
         << telemetry.projection_debug.center_y_T_m
         << ",\"center_z_T_m\":"
         << telemetry.projection_debug.center_z_T_m
         << ",\"theta_rad\":" << telemetry.projection_debug.theta_rad
         << ",\"radius_even_m\":"
         << telemetry.projection_debug.radius_even_m
         << ",\"radius_odd_delta_m\":"
         << telemetry.projection_debug.radius_odd_delta_m
         << ",\"height_odd_delta_m\":"
         << telemetry.projection_debug.height_odd_delta_m << "}"
         << ",\"tracker_state\":\""
         << jsonEscape(telemetry.tracker_state)
         << "\",\"tracker_has_state\":"
         << (telemetry.tracker_has_state ? "true" : "false")
         << ",\"tracker_observation_count\":"
         << telemetry.tracker_observation_count
         << ",\"reliable_yaw_count\":" << telemetry.reliable_yaw_count
         << ",\"tracker_yaw_status\":\""
         << jsonEscape(telemetry.tracker_yaw_status) << "\"";
  if (telemetry.tracker_yaw_diagnostic_valid &&
      std::isfinite(telemetry.tracker_yaw_rms_px)) {
    stream << ",\"tracker_yaw_rms_px\":" << telemetry.tracker_yaw_rms_px;
  } else {
    stream << ",\"tracker_yaw_rms_px\":null";
  }
  if (telemetry.tracker_has_state) {
    stream << ",\"tracker_center_x_T_m\":" << telemetry.tracker_center_x_T_m
           << ",\"tracker_center_y_T_m\":" << telemetry.tracker_center_y_T_m
           << ",\"tracker_center_z_T_m\":" << telemetry.tracker_center_z_T_m
           << ",\"tracker_velocity_x_T_mps\":"
           << telemetry.tracker_velocity_x_T_mps
           << ",\"tracker_velocity_y_T_mps\":"
           << telemetry.tracker_velocity_y_T_mps
           << ",\"tracker_velocity_z_T_mps\":"
           << telemetry.tracker_velocity_z_T_mps
           << ",\"tracker_theta_rad\":" << telemetry.tracker_theta_rad
           << ",\"tracker_omega_rad_s\":" << telemetry.tracker_omega_rad_s
           << ",\"tracker_radius_even_m\":"
           << telemetry.tracker_radius_even_m
           << ",\"tracker_radius_odd_delta_m\":"
           << telemetry.tracker_radius_odd_delta_m
           << ",\"tracker_radius_odd_m\":"
           << telemetry.tracker_radius_odd_m
           << ",\"tracker_height_odd_delta_m\":"
           << telemetry.tracker_height_odd_delta_m
           << ",\"tracker_consecutive_hits\":"
           << telemetry.tracker_consecutive_hits
           << ",\"tracker_consecutive_misses\":"
           << telemetry.tracker_consecutive_misses;
  } else {
    stream << ",\"tracker_center_x_T_m\":null,"
              "\"tracker_center_y_T_m\":null,"
              "\"tracker_center_z_T_m\":null,"
              "\"tracker_velocity_x_T_mps\":null,"
              "\"tracker_velocity_y_T_mps\":null,"
              "\"tracker_velocity_z_T_mps\":null,"
              "\"tracker_theta_rad\":null,"
              "\"tracker_omega_rad_s\":null,"
              "\"tracker_radius_even_m\":null,"
              "\"tracker_radius_odd_delta_m\":null,"
              "\"tracker_radius_odd_m\":null,"
              "\"tracker_height_odd_delta_m\":null,"
              "\"tracker_consecutive_hits\":0,"
              "\"tracker_consecutive_misses\":0";
  }
  if (telemetry.tracker_association_valid) {
    stream << ",\"tracker_associated_slot\":"
           << telemetry.tracker_associated_slot;
  } else {
    stream << ",\"tracker_associated_slot\":null";
  }
  if (telemetry.tracker_nis_valid && std::isfinite(telemetry.tracker_nis)) {
    stream << ",\"tracker_nis\":" << telemetry.tracker_nis;
  } else {
    stream << ",\"tracker_nis\":null";
  }
  stream
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
  stream << ",\"predicted_armors\":[";
  for (std::size_t index = 0; index < telemetry.predicted_armors.size();
       ++index) {
    if (index != 0) stream << ',';
    const WebPredictedArmorTelemetry& armor =
        telemetry.predicted_armors[index];
    stream << "{\"armor_slot\":" << armor.armor_slot
           << ",\"x_T_m\":" << armor.x_T_m
           << ",\"y_T_m\":" << armor.y_T_m
           << ",\"z_T_m\":" << armor.z_T_m << '}';
  }
  stream << "],\"tracker_measurements\":[";
  for (std::size_t index = 0; index < telemetry.tracker_measurements.size();
       ++index) {
    if (index != 0) stream << ',';
    const WebTrackerMeasurementTelemetry& measurement =
        telemetry.tracker_measurements[index];
    stream << "{\"timestamp_ns\":" << measurement.timestamp_ns
           << ",\"x_T_m\":" << measurement.x_T_m
           << ",\"y_T_m\":" << measurement.y_T_m
           << ",\"z_T_m\":" << measurement.z_T_m
           << ",\"camera_range_m\":" << measurement.camera_range_m
           << ",\"has_exposure_camera_geometry\":"
           << (measurement.has_exposure_camera_geometry ? "true" : "false")
           << ",\"camera_x_T_m\":" << measurement.camera_x_T_m
           << ",\"camera_y_T_m\":" << measurement.camera_y_T_m
           << ",\"has_inward_yaw\":"
           << (measurement.has_inward_yaw ? "true" : "false")
           << ",\"has_raw_inward_yaw\":"
           << (measurement.has_raw_inward_yaw ? "true" : "false")
           << ",\"has_pnp_inward_pitch_T\":"
           << (measurement.has_pnp_inward_pitch_T ? "true" : "false")
           << ",\"inward_yaw_rad\":" << measurement.inward_yaw_rad
           << ",\"pnp_inward_pitch_rad\":"
           << measurement.pnp_inward_pitch_rad
           << ",\"yaw_std_rad\":" << measurement.yaw_std_rad
           << ",\"reprojection_rms_px\":"
           << measurement.reprojection_rms_px
           << ",\"confidence\":" << measurement.confidence
           << ",\"keypoint_quality\":" << measurement.keypoint_quality
           << ",\"view_quality\":" << measurement.view_quality
           << ",\"color_id\":" << measurement.color_id
           << ",\"number_id\":" << measurement.number_id
           << ",\"association_valid\":"
           << (measurement.association_valid ? "true" : "false")
           << ",\"yaw_used\":"
           << (measurement.yaw_used ? "true" : "false")
           << ",\"associated_slot\":" << measurement.associated_slot
           << ",\"nis\":" << measurement.nis
           << ",\"predicted_x_T_m\":" << measurement.predicted_x_T_m
           << ",\"predicted_y_T_m\":" << measurement.predicted_y_T_m
           << ",\"predicted_z_T_m\":" << measurement.predicted_z_T_m
           << ",\"innovation_x_T_m\":" << measurement.innovation_x_T_m
           << ",\"innovation_y_T_m\":" << measurement.innovation_y_T_m
           << ",\"innovation_z_T_m\":" << measurement.innovation_z_T_m
           << ",\"radial_innovation_m\":" << measurement.radial_innovation_m
           << ",\"predicted_yaw_rad\":" << measurement.predicted_yaw_rad
           << ",\"yaw_innovation_rad\":" << measurement.yaw_innovation_rad
           << '}';
  }
  stream << "],\"poses\":[";
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

// 对不支持的 HTTP 路由返回精简的 404 响应。
void sendNotFound(Socket client) {
  sendText(client, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n"
                   "Connection: close\r\n\r\n");
}

// 发送禁用缓存且内容长度正确的 JSON 文档。
void sendJsonDocument(Socket client, const std::string& body) {
  sendText(client, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                   "Cache-Control: no-store\r\nContent-Length: " +
                       std::to_string(body.size()) +
                       "\r\nConnection: close\r\n\r\n" + body);
}

// 以精简 JSON 响应返回控制端点结果。
void sendControlJson(Socket client, int status, const char* message) {
  const std::string body = std::string("{\"message\":\"") + message + "\"}";
  const std::string reason = status == 202 ? "Accepted" : "Bad Request";
  sendText(client, "HTTP/1.1 " + std::to_string(status) + " " + reason +
                   "\r\nContent-Type: application/json\r\nCache-Control: no-store\r\n"
                   "Content-Length: " + std::to_string(body.size()) +
                   "\r\nConnection: close\r\n\r\n" + body);
}

// Decode the query value before validating it. Browser fetch() percent-encodes
// the ':' used by the variable spin-speed command.
std::optional<std::string> decodeQueryValue(std::string_view encoded) {
  std::string decoded;
  decoded.reserve(encoded.size());
  const auto hexValue = [](char character) -> int {
    if (character >= '0' && character <= '9') return character - '0';
    if (character >= 'a' && character <= 'f') return character - 'a' + 10;
    if (character >= 'A' && character <= 'F') return character - 'A' + 10;
    return -1;
  };
  for (std::size_t index = 0; index < encoded.size(); ++index) {
    if (encoded[index] != '%') {
      decoded.push_back(encoded[index]);
      continue;
    }
    if (index + 2 >= encoded.size()) return std::nullopt;
    const int high = hexValue(encoded[index + 1]);
    const int low = hexValue(encoded[index + 2]);
    if (high < 0 || low < 0) return std::nullopt;
    decoded.push_back(static_cast<char>((high << 4) | low));
    index += 2;
  }
  return decoded;
}

// Extract and validate a control operation token from an HTTP request line.
std::string controlAction(const std::string& request) {
  const std::size_t path_end = request.find(' ', 4);
  if (path_end == std::string::npos) return {};
  const std::string target = request.substr(4, path_end - 4);
  constexpr std::string_view kPrefix = "/api/control?action=";
  if (target.rfind(kPrefix, 0) != 0) return {};
  std::string action = target.substr(kPrefix.size());
  const std::size_t next_parameter = action.find('&');
  if (next_parameter != std::string::npos) action.resize(next_parameter);
  const std::optional<std::string> decoded_action = decodeQueryValue(action);
  if (!decoded_action) return {};
  action = *decoded_action;
  if (action.empty() || action.size() > 160) return {};
  for (const unsigned char character : action) {
    if (!std::isalnum(character) && character != '-' && character != ':' &&
        character != '.' && character != '_') {
      return {};
    }
  }
  return action;
}

// 在每帧 JPEG 编码完成后持续向单个客户端推送。
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

// 解析单个 HTTP 请求，并路由到页面、图像、状态或控制逻辑。
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

// 接受连接，并将每个独立请求交给工作线程。
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

// 校验选项、绑定 HTTP 监听器并启动接收循环。
MjpegServer::MjpegServer(WebServerOptions options, ControlHandler control_handler)
    : state_(std::make_shared<State>()) {
  if (options.port == 0) throw std::invalid_argument("web port must be non-zero");
  if (options.jpeg_quality < 1 || options.jpeg_quality > 100) {
    throw std::invalid_argument("web JPEG quality must be in [1, 100]");
  }
  if (options.max_image_width < 160 || options.max_frame_rate < 1) {
    throw std::invalid_argument("invalid web image size or frame rate");
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

// 停止监听器并等待接收循环结束后释放共享状态。
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

// 使用默认空遥测记录发布图像帧。
void MjpegServer::publish(const cv::Mat& bgr_frame) {
  publish(bgr_frame, WebFrameTelemetry{});
}

// 对图像帧进行 JPEG 编码，原子替换共享数据并唤醒客户端。
void MjpegServer::publish(const cv::Mat& bgr_frame,
                          const WebFrameTelemetry& telemetry) {
  const std::string telemetry_json = telemetryJson(telemetry);
  {
    std::lock_guard<std::mutex> lock(state_->frame_mutex);
    state_->latest_telemetry_json = telemetry_json;
  }

  if (bgr_frame.empty()) return;
  std::lock_guard<std::mutex> publish_lock(state_->publish_mutex);
  const auto now = std::chrono::steady_clock::now();
  const auto publish_interval =
      std::chrono::milliseconds(1000 / state_->options.max_frame_rate);
  if (state_->last_image_publish.time_since_epoch().count() != 0 &&
      now - state_->last_image_publish < publish_interval) {
    return;
  }
  state_->last_image_publish = now;

  cv::Mat web_frame;
  if (bgr_frame.cols > state_->options.max_image_width) {
    const double scale = static_cast<double>(state_->options.max_image_width) /
                         static_cast<double>(bgr_frame.cols);
    cv::resize(bgr_frame, web_frame, {}, scale, scale, cv::INTER_AREA);
  } else {
    web_frame = bgr_frame;
  }
  std::vector<unsigned char> jpeg;
  const std::vector<int> parameters = {cv::IMWRITE_JPEG_QUALITY,
                                       state_->options.jpeg_quality};
  if (!cv::imencode(".jpg", web_frame, jpeg, parameters)) return;
  {
    std::lock_guard<std::mutex> lock(state_->frame_mutex);
    state_->latest_jpeg = std::move(jpeg);
    ++state_->frame_sequence;
  }
  state_->frame_ready.notify_all();
}

// 格式化此服务器对外可用的根地址。
std::string MjpegServer::url() const {
  return "http://" + state_->options.bind_address + ":" +
         std::to_string(state_->options.port) + "/";
}

}  // namespace yolo_detect
