#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cv {
class Mat;
}

namespace yolo_detect {

struct WebServerOptions {
  std::string bind_address = "127.0.0.1";
  std::uint16_t port = 8080;
  int jpeg_quality = 80;
};

struct WebPoseTelemetry {
  std::size_t detection_index = 0;
  bool valid = false;
  std::string armor_size;
  std::string status;
  double x_m = 0.0;
  double y_m = 0.0;
  double z_m = 0.0;
  double reprojection_rms_px = 0.0;
  std::size_t candidate_count = 0;
  bool coordinate_valid = false;
  std::string coordinate_status;
  double odom_x_m = 0.0;
  double odom_y_m = 0.0;
  double odom_z_m = 0.0;
  bool aim_valid = false;
  std::string aim_status;
  std::string ballistic_status;
  double yaw_command_deg = 0.0;
  double pitch_command_deg = 0.0;
  double time_of_flight_s = 0.0;
  double gravity_drop_m = 0.0;
  double muzzle_odom_x_m = 0.0;
  double muzzle_odom_y_m = 0.0;
  double muzzle_odom_z_m = 0.0;
  bool predicted = false;
  double prediction_horizon_s = 0.0;
};

struct WebPredictedArmorTelemetry {
  int armor_slot = -1;
  double x_T_m = 0.0;
  double y_T_m = 0.0;
  double z_T_m = 0.0;
};

// Mirrors one tracker Measurement before data association and EKF update.
struct WebTrackerMeasurementTelemetry {
  std::uint64_t timestamp_ns = 0;
  double x_T_m = 0.0;
  double y_T_m = 0.0;
  double z_T_m = 0.0;
  double camera_range_m = 0.0;
  bool has_inward_yaw = false;
  double inward_yaw_rad = 0.0;
  double yaw_std_rad = 0.0;
  double reprojection_rms_px = 0.0;
  double confidence = 0.0;
  double keypoint_quality = 0.0;
  double view_quality = 0.0;
  int color_id = -1;
  int number_id = -1;
  bool association_valid = false;
  int associated_slot = -1;
  double nis = 0.0;
  double predicted_x_T_m = 0.0;
  double predicted_y_T_m = 0.0;
  double predicted_z_T_m = 0.0;
  double innovation_x_T_m = 0.0;
  double innovation_y_T_m = 0.0;
  double innovation_z_T_m = 0.0;
  double radial_innovation_m = 0.0;
  double predicted_yaw_rad = 0.0;
  double yaw_innovation_rad = 0.0;
};

struct WebFrameTelemetry {
  std::uint64_t source_sequence = 0;
  std::string scene;
  std::string motion;
  double vehicle_speed_mps = 0.0;
  double spin_speed_deg_s = 0.0;
  std::string tracker_state = "uninitialized";
  bool tracker_has_state = false;
  std::size_t tracker_observation_count = 0;
  std::size_t reliable_yaw_count = 0;
  std::string tracker_yaw_status = "no armor observations";
  bool tracker_yaw_diagnostic_valid = false;
  double tracker_yaw_rms_px = 0.0;
  double tracker_center_x_T_m = 0.0;
  double tracker_center_y_T_m = 0.0;
  double tracker_center_z_T_m = 0.0;
  double tracker_velocity_x_T_mps = 0.0;
  double tracker_velocity_y_T_mps = 0.0;
  double tracker_velocity_z_T_mps = 0.0;
  double tracker_theta_rad = 0.0;
  double tracker_omega_rad_s = 0.0;
  double tracker_radius_even_m = 0.0;
  double tracker_radius_odd_delta_m = 0.0;
  double tracker_height_odd_delta_m = 0.0;
  bool tracker_association_valid = false;
  int tracker_associated_slot = -1;
  bool tracker_nis_valid = false;
  double tracker_nis = 0.0;
  int tracker_consecutive_hits = 0;
  int tracker_consecutive_misses = 0;
  bool coordinate_valid = false;
  std::string coordinate_status;
  double camera_position_error_m = 0.0;
  double actual_yaw_deg = 0.0;
  double actual_pitch_command_deg = 90.0;
  bool gimbal_following = false;
  bool fire_pending = false;
  bool static_target_valid = false;
  std::string gimbal_status;
  double static_target_odom_x_m = 0.0;
  double static_target_odom_y_m = 0.0;
  double static_target_odom_z_m = 0.0;
  std::uint64_t last_gimbal_command_id = 0;
  bool last_command_fired = false;
  std::vector<WebPredictedArmorTelemetry> predicted_armors;
  std::vector<WebTrackerMeasurementTelemetry> tracker_measurements;
  std::vector<WebPoseTelemetry> poses;
};

// 用于本地 MJPEG 调试的小型无外部依赖 HTTP 服务器。
class MjpegServer {
 public:
  struct State;
  using ControlHandler = std::function<bool(std::string_view)>;

  // 启动本地 HTTP 监听器并关联控制回调。
  MjpegServer(WebServerOptions options, ControlHandler control_handler = {});
  // 停止监听器并唤醒所有阻塞的流客户端。
  ~MjpegServer();

  MjpegServer(const MjpegServer&) = delete;
  MjpegServer& operator=(const MjpegServer&) = delete;
  MjpegServer(MjpegServer&&) = delete;
  MjpegServer& operator=(MjpegServer&&) = delete;

  // 编码并发布不带附加遥测的图像帧。
  void publish(const cv::Mat& bgr_frame);
  // 编码并发布图像帧及其最新遥测快照。
  void publish(const cv::Mat& bgr_frame,
               const WebFrameTelemetry& telemetry);
  // 返回已绑定本地服务器的根 URL。
  [[nodiscard]] std::string url() const;

 private:
  std::shared_ptr<State> state_;
};

}  // namespace yolo_detect
