#pragma once

#include <Eigen/Core>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace yolo_detect::tracking {

// 固定维度用于避免运行时动态分配；所有位置使用 T 系米，角度使用 rad。
// Whole-vehicle tracking architecture, in reading order:
//   1. tracker_measurement_adapter turns one PnP pose plus the matching
//      exposure snapshot into a Measurement in tracker frame T.
//   2. constrained_yaw_solver independently validates inward yaw by
//      constrained reprojection; PnP rvec is never used as vehicle yaw.
//   3. WholeVehicleEkf predicts this 11-state vehicle model and associates
//      every visible measurement with one physical slot E0..E3. Only the
//      selected primary observation updates the EKF state.
//   4. decodeArmor expands the posterior back into E0..E3 for the yellow
//      image markers and web telemetry.
//
// T is fixed to ROS odom for one track: +x forward, +y left, +z up. Every
// position is meters, every timestamp is exposure time in ns, and angles are
// radians. Target GroundTruth is deliberately not an input to this pipeline.
inline constexpr int kWholeVehicleStateDimension = 11;
inline constexpr int kArmorObservationDimension = 4;
inline constexpr int kArmorSlotCount = 4;

enum StateIndex : int {
  // theta is the inward yaw of E0 and is wrapped to [-pi, pi) after every
  // prediction and update.
  // For slot i: phi=theta+i*pi/2, r=r0+(i%2)*dr, z=cz+(i%2)*dz, and
  // armor center=(cx-r*cos(phi), cy-r*sin(phi), z).
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
  // Raw constrained-yaw candidate for diagnostics. It never enables EKF yaw.
  bool has_raw_inward_yaw = false;
  // Pitch inferred from the IPPE plate normal in T. It is diagnostic only.
  bool has_pnp_inward_pitch_T = false;
  double pnp_inward_pitch_T_rad = 0.0;
  // Both raw IPPE orientation candidates transformed into T. Diagnostics only.
  bool has_ippe_inward_yaw_0_T = false;
  double ippe_inward_yaw_0_T_rad = 0.0;
  bool has_ippe_inward_yaw_1_T = false;
  double ippe_inward_yaw_1_T_rad = 0.0;
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
  // The application selects one board per frame for the EKF update. Other
  // visible boards still participate in slot association and telemetry.
  bool selected_for_ekf = true;
  // Exposure-time camera geometry. R_TC maps camera vectors into tracker T.
  Eigen::Matrix3d R_TC = Eigen::Matrix3d::Identity();
  Eigen::Vector3d camera_position_T_m = Eigen::Vector3d::Zero();
  bool has_exposure_camera_geometry = false;
};

struct State {
  // theta is wrapped after prediction and update; covariance is aligned with
  // this same state representation.
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
  double radius_prior_m = 0.20;
  double minimum_radius_m = 0.05;
  double maximum_radius_m = 0.50;
  double maximum_radius_difference_m = 0.12;
  double maximum_height_difference_m = 0.20;
  // Match sp_vision_25's discrete piecewise-white-acceleration model:
  // v * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]. Units are m^2/s^4 and
  // rad^2/s^4 respectively, not continuous-time spectral densities.
  double q_linear_acceleration = 100.0;
  double q_angular_acceleration = 400.0;
  // Vehicle geometry is static during one track. It is updated only by a
  // geometrically consistent multi-armor frame, not by process random walk.
  double q_geometry = 0.0;
  // 固定的单帧 PnP 观测标准差，分别对应相机 X/Y/深度方向。
  double position_std_x_m = 0.03;
  double position_std_y_m = 0.03;
  double position_std_z_m = 0.10;
  // Rtheta = base + log(1 + facing_angle) * scale, following the
  // sp_vision_25 yaw-noise form. Values are variances in rad^2.
  double yaw_facing_base_variance_rad2 = 1.0;
  double yaw_facing_log_variance_scale_rad2 = 1.0 / 200.0;
  // A single plate observes center only through the configured radius and is
  // therefore much weaker than a simultaneous multi-plate observation.
  double single_armor_position_variance_scale = 25.0;
  // Association needs a wider position gate than the state update so a track
  // can recover after a motion reversal without accepting the residual at
  // full EKF weight.
  double association_position_variance_scale = 100.0;
  // Association has no rejection threshold. It selects the minimum weighted
  // x-y distance plus wrapped yaw difference.
  double slot_position_cost_weight = 1.0;
  double slot_yaw_cost_weight_m_per_rad = 0.20;
  // Legacy tuning fields are retained so saved runtime tuning requests remain
  // accepted. The current cost-based association does not use them.
  double maximum_yaw_update_innovation_rad = 0.35;
  double maximum_yaw_association_innovation_rad = 1.80;
  double yaw_phase_cost_std_rad = 0.35;
  double adjacent_slot_penalty = 4.0;
  double opposite_slot_penalty = 6.0;
  double minimum_visibility_cosine = -0.35;
  // Geometry is observable only from distinct slots with a 3D baseline. When
  // both boards have reliable yaw their phase must also be consistent.
  double geometry_yaw_consistency_rad = 0.35;
  double geometry_minimum_baseline_m = 0.08;
  // A consistent simultaneous two-board observation updates fixed geometry
  // immediately. Single-board updates still have geometry gain rows disabled.
  int geometry_confirming_frames = 1;
  // 3D/4D 卡方 NIS 门限，默认分别接近 95% 分位。
  double nis_gate_3d = 7.815;
  double nis_gate_4d = 9.488;
  // 状态机确认、丢失和时间跳变策略。
  int confirming_hits = 3;
  // A detector blackout during fast rotation must not redefine E0. Reset only
  // after this many consecutive missed frames; time is diagnostic only.
  int lost_frame_limit = 20;
  double lost_time_limit_s = 1.5;
  double maximum_frame_dt_s = 0.25;
  // sqrt([1,64,1,64,1,64,0.4,100,1,1,1]), matching sp_vision_25's
  // ordinary four-armor P0 diagonal.
  double initial_position_std_m = 1.0;
  double initial_velocity_std_mps = 8.0;
  double initial_theta_std_rad = 0.6324555320336759;
  double initial_omega_std_rad_s = 10.0;
  double initial_geometry_std_m = 1.0;
  // 10 rad/s is a reachable chassis speed; retain margin for estimation error.
  double maximum_angular_speed_rad_s = 12.0;
  double maximum_omega_correction_rad_s = 0.5;
  double maximum_multi_armor_position_residual_m = 0.18;
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
  // Applies tuning without clearing the current track. Initial covariance
  // fields take effect on the next initialization; Q/R and gates are live.
  [[nodiscard]] bool setOptions(WholeVehicleEkfOptions options) noexcept;
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
  int consecutive_geometry_observations_ = 0;
  bool motion_observed_ = false;
  std::array<bool, kArmorSlotCount> last_associated_slots_{};
};

[[nodiscard]] const char* trackingStateName(TrackingState state) noexcept;

}  // namespace yolo_detect::tracking
