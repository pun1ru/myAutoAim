#pragma once

#include <Eigen/Core>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace yolo_detect::tracking {

inline constexpr int kWholeVehicleStateDimension = 11;
inline constexpr int kArmorObservationDimension = 4;
inline constexpr int kArmorSlotCount = 4;

enum StateIndex : int {
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

// A single PnP-derived armor observation expressed in the tracker frame T.
// `has_inward_yaw` remains false until an independent constrained-reprojection
// yaw estimator is available; PnP Rodrigues components are never used here.
struct Measurement {
  std::uint64_t timestamp_ns = 0;
  Eigen::Vector3d position_T_m = Eigen::Vector3d::Zero();
  double inward_yaw_T_rad = 0.0;
  double reprojection_rms_px = 0.0;
  double confidence = 0.0;
  int color_id = -1;
  int number_id = -1;
  double keypoint_quality = 1.0;
  double view_quality = 1.0;
  bool has_inward_yaw = false;
};

struct State {
  Vector11 x = Vector11::Zero();
  Matrix11 covariance = Matrix11::Identity();
};

struct DecodedArmor {
  int armor_slot = -1;
  Eigen::Vector3d position_T_m = Eigen::Vector3d::Zero();
  double inward_yaw_T_rad = 0.0;
};

// Advances the mean state with constant linear and angular velocity.
[[nodiscard]] State predictState(const State& state, double dt_s);
// Returns the 4D [position_T, inward_yaw_T] observation of one physical slot.
[[nodiscard]] Measurement observe(const State& state, int armor_slot);
// Analytic derivative of observe(state, armor_slot) with respect to state.x.
[[nodiscard]] Jacobian observationJacobian(const State& state, int armor_slot);
// Decodes one future physical armor position without wrapping theta.
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
  double radius_prior_m = 0.26;
  double minimum_radius_m = 0.02;
  double q_linear_acceleration = 4.0;
  double q_angular_acceleration = 16.0;
  double q_geometry = 1e-4;
  double position_std_xy_m = 0.03;
  double position_std_z_m = 0.08;
  double yaw_std_rad = 0.12;
  double reprojection_rms_scale = 0.15;
  double range_noise_scale_per_m = 0.025;
  double minimum_quality = 0.05;
  double nis_gate_3d = 7.815;
  double nis_gate_4d = 9.488;
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
  int consecutive_hits = 0;
  int consecutive_misses = 0;
  std::array<DecodedArmor, kArmorSlotCount> predicted_armors{};
};

class WholeVehicleEkf {
 public:
  explicit WholeVehicleEkf(WholeVehicleEkfOptions options = {});

  [[nodiscard]] TrackOutput update(
      std::uint64_t timestamp_ns, const std::vector<Measurement>& measurements);
  void reset() noexcept;

  [[nodiscard]] const WholeVehicleEkfOptions& options() const noexcept;
  [[nodiscard]] TrackingState trackingState() const noexcept;
  [[nodiscard]] bool hasState() const noexcept;
  [[nodiscard]] const State& state() const noexcept;

 private:
  struct Association {
    int measurement_index = -1;
    int armor_slot = -1;
    double nis = 0.0;
    bool includes_yaw = false;
    Matrix4 covariance = Matrix4::Zero();
  };

  [[nodiscard]] Matrix11 processNoise(double dt_s) const;
  [[nodiscard]] Matrix4 measurementCovariance(
      const Measurement& measurement) const;
  [[nodiscard]] bool identityConsistent(const Measurement& measurement) const;
  [[nodiscard]] std::optional<Association> associate(
      const std::vector<Measurement>& measurements) const;
  [[nodiscard]] bool applyUpdate(const Measurement& measurement,
                                 const Association& association);
  [[nodiscard]] bool initialize(const Measurement& measurement);
  [[nodiscard]] bool stateFiniteAndPhysical() const;
  [[nodiscard]] TrackOutput makeOutput(std::uint64_t timestamp_ns,
                                       std::optional<int> associated_slot,
                                       std::optional<double> nis) const;

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
};

[[nodiscard]] const char* trackingStateName(TrackingState state) noexcept;

}  // namespace yolo_detect::tracking
