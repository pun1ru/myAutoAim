#pragma once

#include <Eigen/Core>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace yolo_detect::tracking {

// 固定维度用于避免运行时动态分配；所有位置使用 T 系米，角度使用 rad。
inline constexpr int kWholeVehicleStateDimension = 11;
inline constexpr int kArmorObservationDimension = 4;
inline constexpr int kArmorSlotCount = 4;

enum StateIndex : int {
  // 状态排列为 [cx,vx, cy,vy, cz,vz, theta,omega, r0,dr,dz]。
  CenterX = 0,
  VelocityX = 1,
  CenterY = 2,
  VelocityY = 3,
  CenterZ = 4,
  VelocityZ = 5,
  Theta = 6,
  Omega = 7,
  RadiusEven = 8,
  RadiusOddDelta = 9,
  HeightOddDelta = 10,
};

using Vector11 = Eigen::Matrix<double, kWholeVehicleStateDimension, 1>;
using Matrix11 = Eigen::Matrix<double, kWholeVehicleStateDimension,
                               kWholeVehicleStateDimension>;
using Vector4 = Eigen::Matrix<double, kArmorObservationDimension, 1>;
using Matrix4 = Eigen::Matrix<double, kArmorObservationDimension,
                               kArmorObservationDimension>;
using Jacobian = Eigen::Matrix<double, kArmorObservationDimension,
                               kWholeVehicleStateDimension>;
inline constexpr int kMaximumStackedObservationDimension =
    kArmorObservationDimension * kArmorSlotCount;
using Vector16 =
    Eigen::Matrix<double, kMaximumStackedObservationDimension, 1>;
using Matrix16 = Eigen::Matrix<double, kMaximumStackedObservationDimension,
                               kMaximumStackedObservationDimension>;
using StackedJacobian =
    Eigen::Matrix<double, kMaximumStackedObservationDimension,
                  kWholeVehicleStateDimension>;

// 单块装甲板在本次曝光时刻的量测。position_T_m 由 PnP 平移和同曝光
// 坐标快照转换而来；yaw 必须来自独立约束重投影，不能使用 PnP rvec。
struct Measurement {
  std::uint64_t timestamp_ns = 0;
  Eigen::Vector3d position_T_m = Eigen::Vector3d::Zero();
  // Euclidean distance from the exposure-time camera center to this armor,
  // measured in the OpenCV camera frame. It is not a T/odom-origin norm.
  double camera_range_m = 0.0;
  double inward_yaw_T_rad = 0.0;
  // The constrained-reprojection yaw uncertainty, in radians. Zero selects
  // the EKF's configured yaw standard deviation for synthetic observations.
  double yaw_std_rad = 0.0;
  double reprojection_rms_px = 0.0;
  double confidence = 0.0;
  int color_id = -1;
  int number_id = -1;
  double keypoint_quality = 1.0;
  double view_quality = 1.0;
  bool has_inward_yaw = false;
  // Exposure-time camera geometry. R_TC maps camera vectors into tracker T.
  Eigen::Matrix3d R_TC = Eigen::Matrix3d::Identity();
  Eigen::Vector3d camera_position_T_m = Eigen::Vector3d::Zero();
  bool has_exposure_camera_geometry = false;
};

struct State {
  // x 是连续 theta 的内部状态，不在每帧 wrap；covariance 与 x 一一对应。
  Vector11 x = Vector11::Zero();
  Matrix11 covariance = Matrix11::Identity();
};

struct DecodedArmor {
  // 由整车状态解码出的物理槽位，而不是网络的 number_id。
  int armor_slot = -1;
  Eigen::Vector3d position_T_m = Eigen::Vector3d::Zero();
  double inward_yaw_T_rad = 0.0;
};

// A measurement accepted during the current frame. Measurement indices refer
// to the vector passed to WholeVehicleEkf::update, rather than detector IDs.
struct AssociatedObservation {
  int measurement_index = -1;
  int armor_slot = -1;
  double nis = 0.0;
  bool includes_yaw = false;
  Eigen::Vector3d predicted_position_T_m = Eigen::Vector3d::Zero();
  Eigen::Vector3d position_innovation_T_m = Eigen::Vector3d::Zero();
  double predicted_inward_yaw_T_rad = 0.0;
  double yaw_innovation_rad = 0.0;
  // Positive means the observation lies toward the predicted vehicle center.
  double radial_innovation_m = 0.0;
};

// 纯函数：按匀速、匀角速度模型预测均值，不修改协方差。
[[nodiscard]] State predictState(const State& state, double dt_s);
// 纯函数：返回指定物理槽位的 h(x,i)=[x,y,z,inward_yaw]。
[[nodiscard]] Measurement observe(const State& state, int armor_slot);
// h(x,i) 对 11 维状态的解析 Jacobian，供关联和 EKF 更新共用。
[[nodiscard]] Jacobian observationJacobian(const State& state, int armor_slot);
// 将状态预测 horizon_s 后解码一个物理装甲板；theta 仍保持连续。
[[nodiscard]] DecodedArmor decodeArmor(const State& state, int armor_slot,
                                        double horizon_s);
[[nodiscard]] double wrapToPi(double angle_rad) noexcept;

enum class TrackingState {
  Uninitialized,
  Confirming,
  Tracking,
  TemporarilyLost,
  Lost,
};

struct WholeVehicleEkfOptions {
  // 几何先验：首次只看到一块板时，中心只能由该半径反推。
  double radius_prior_m = 0.26;
  double minimum_radius_m = 0.05;
  double maximum_radius_m = 0.50;
  double maximum_radius_difference_m = 0.12;
  double maximum_height_difference_m = 0.20;
  // 连续白噪声谱密度，分别用于平动、角运动和几何随机游走。
  double q_linear_acceleration = 4.0;
  double q_angular_acceleration = 16.0;
  // Vehicle geometry is static during one track. It is updated only by a
  // geometrically consistent multi-armor frame, not by process random walk.
  double q_geometry = 0.0;
  // 基础观测标准差。R 会再按质量、重投影 RMS 和相机量测距离放大。
  double position_std_xy_m = 0.03;
  double position_std_z_m = 0.08;
  double yaw_std_rad = 0.12;
  double reprojection_rms_scale = 0.15;
  double range_noise_scale_per_m = 0.025;
  double minimum_quality = 0.05;
  // Association uses yaw more permissively than the EKF update. A large but
  // plausible phase residual may select a slot while remaining position-only.
  double maximum_yaw_update_innovation_rad = 0.35;
  double maximum_yaw_association_innovation_rad = 1.40;
  double yaw_phase_cost_std_rad = 0.35;
  double adjacent_slot_penalty = 0.5;
  double opposite_slot_penalty = 4.0;
  double minimum_visibility_cosine = -0.35;
  // Geometry is observable only from distinct slots with consistent yaw.
  double geometry_yaw_consistency_rad = 0.35;
  double geometry_minimum_baseline_m = 0.08;
  // 3D/4D 卡方 NIS 门限，默认分别接近 95% 分位。
  double nis_gate_3d = 7.815;
  double nis_gate_4d = 9.488;
  // 状态机确认、丢失和时间跳变策略。
  int confirming_hits = 3;
  int lost_frame_limit = 10;
  double lost_time_limit_s = 0.5;
  double maximum_frame_dt_s = 0.25;
  double initial_position_std_m = 0.25;
  double initial_velocity_std_mps = 3.0;
  double initial_theta_std_rad = 0.35;
  double initial_omega_std_rad_s = 8.0;
  double initial_geometry_std_m = 0.15;
};

struct TrackOutput {
  // 每帧的完整快照，既供黄点投影，也供网页/API 输出。
  std::uint64_t timestamp_ns = 0;
  TrackingState tracking_state = TrackingState::Uninitialized;
  bool has_state = false;
  State state;
  Eigen::Vector3d center_T_m = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity_T_mps = Eigen::Vector3d::Zero();
  double theta_rad = 0.0;
  double omega_rad_s = 0.0;
  double radius_even_m = 0.0;
  double radius_odd_delta_m = 0.0;
  double height_odd_delta_m = 0.0;
  std::optional<int> associated_slot;
  std::optional<double> nis;
  std::vector<AssociatedObservation> associated_observations;
  int consecutive_hits = 0;
  int consecutive_misses = 0;
  std::array<DecodedArmor, kArmorSlotCount> predicted_armors{};
};

class WholeVehicleEkf {
 public:
  explicit WholeVehicleEkf(WholeVehicleEkfOptions options = {});

  // 一次调用严格对应一个 capture_timestamp_ns：先 predict，再关联并更新。
  [[nodiscard]] TrackOutput update(
      std::uint64_t timestamp_ns, const std::vector<Measurement>& measurements);
  void reset() noexcept;

  [[nodiscard]] const WholeVehicleEkfOptions& options() const noexcept;
  [[nodiscard]] TrackingState trackingState() const noexcept;
  [[nodiscard]] bool hasState() const noexcept;
  [[nodiscard]] const State& state() const noexcept;

 private:
  struct Association {
    // 本帧被接受的“检测量测 <-> 物理槽位”组合。
    int measurement_index = -1;
    int armor_slot = -1;
    double nis = 0.0;
    bool includes_yaw = false;
    double cost = 0.0;
    Matrix4 covariance = Matrix4::Zero();
  };

  // Q、R、身份门控、关联、Joseph 更新和初始化均保持为私有步骤。
  [[nodiscard]] Matrix11 processNoise(double dt_s) const;
  [[nodiscard]] Matrix4 measurementCovariance(
      const Measurement& measurement) const;
  [[nodiscard]] bool identityConsistent(const Measurement& measurement) const;
  // Kept as a small single-candidate primitive for tests and diagnostics.
  [[nodiscard]] std::optional<Association> associate(
      const std::vector<Measurement>& measurements,
      const std::vector<bool>& used_measurements,
      const std::array<bool, kArmorSlotCount>& used_slots) const;
  [[nodiscard]] bool applyUpdate(const Measurement& measurement,
                                 const Association& association);
  [[nodiscard]] bool slotVisible(const Measurement& measurement,
                                 int armor_slot) const;
  [[nodiscard]] double slotTransitionPenalty(int armor_slot) const;
  [[nodiscard]] std::optional<Association> makeAssociationCandidate(
      const std::vector<Measurement>& measurements, int measurement_index,
      int armor_slot) const;
  [[nodiscard]] std::vector<Association> associateAll(
      const std::vector<Measurement>& measurements) const;
  [[nodiscard]] bool geometryObservable(
      const std::vector<Measurement>& measurements,
      const std::vector<Association>& associations) const;
  [[nodiscard]] bool applyJointUpdate(
      const std::vector<Measurement>& measurements,
      const std::vector<Association>& associations, bool update_geometry);
  [[nodiscard]] bool initialize(const Measurement& measurement);
  [[nodiscard]] bool stateFiniteAndPhysical() const;
  [[nodiscard]] TrackOutput makeOutput(
      std::uint64_t timestamp_ns,
      const std::vector<AssociatedObservation>& associations = {}) const;

  WholeVehicleEkfOptions options_;
  State state_;
  TrackingState tracking_state_ = TrackingState::Uninitialized;
  bool has_state_ = false;
  std::uint64_t last_timestamp_ns_ = 0;
  std::uint64_t last_hit_timestamp_ns_ = 0;
  int consecutive_hits_ = 0;
  int consecutive_misses_ = 0;
  int tracked_color_id_ = -1;
  int tracked_number_id_ = -1;
  std::array<bool, kArmorSlotCount> last_associated_slots_{};
};

[[nodiscard]] const char* trackingStateName(TrackingState state) noexcept;

}  // namespace yolo_detect::tracking
