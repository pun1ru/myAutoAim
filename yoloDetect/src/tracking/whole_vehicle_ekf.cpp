#include "tracking/whole_vehicle_ekf.hpp"

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace yolo_detect::tracking {
double wrapToPi(double angle_rad) noexcept;

namespace {

constexpr double kPi = 3.14159265358979323846;

bool finite(double value) { return std::isfinite(value); }

bool validSlot(int armor_slot) {
  return armor_slot >= 0 && armor_slot < kArmorSlotCount;
}

bool validMeasurement(const Measurement& measurement) {
  return measurement.timestamp_ns != 0 && measurement.position_T_m.allFinite() &&
         finite(measurement.camera_range_m) && measurement.camera_range_m > 0.0 &&
         finite(measurement.reprojection_rms_px) &&
         finite(measurement.confidence) && finite(measurement.keypoint_quality) &&
         finite(measurement.view_quality) &&
         (!measurement.has_inward_yaw || finite(measurement.inward_yaw_T_rad));
}

double square(double value) { return value * value; }

AssociatedObservation makeAssociatedObservation(
    const State& state, const Measurement& measurement,
    int measurement_index, int armor_slot, double nis, bool includes_yaw) {
  const Measurement predicted = observe(state, armor_slot);
  AssociatedObservation output;
  output.measurement_index = measurement_index;
  output.armor_slot = armor_slot;
  output.nis = nis;
  output.includes_yaw = includes_yaw;
  output.predicted_position_T_m = predicted.position_T_m;
  output.position_innovation_T_m =
      measurement.position_T_m - predicted.position_T_m;
  output.predicted_inward_yaw_T_rad = predicted.inward_yaw_T_rad;
  if (includes_yaw) {
    output.yaw_innovation_rad = wrapToPi(
        measurement.inward_yaw_T_rad - predicted.inward_yaw_T_rad);
  }
  const Eigen::Vector3d center(state.x[CenterX], state.x[CenterY],
                               state.x[CenterZ]);
  const Eigen::Vector3d toward_center = center - predicted.position_T_m;
  if (toward_center.squaredNorm() > 1e-12) {
    output.radial_innovation_m = output.position_innovation_T_m.dot(
        toward_center.normalized());
  }
  return output;
}

}  // namespace

double wrapToPi(double angle_rad) noexcept {
  if (!finite(angle_rad)) return std::numeric_limits<double>::quiet_NaN();
  angle_rad = std::fmod(angle_rad + kPi, 2.0 * kPi);
  if (angle_rad < 0.0) angle_rad += 2.0 * kPi;
  return angle_rad - kPi;
}

State predictState(const State& state, double dt_s) {
  if (!finite(dt_s) || dt_s < 0.0) {
    throw std::invalid_argument("prediction dt must be finite and non-negative");
  }
  // 仅积分平动速度和自转角速度；半径与高差作为慢变几何保留。
  State predicted = state;
  predicted.x[CenterX] += predicted.x[VelocityX] * dt_s;
  predicted.x[CenterY] += predicted.x[VelocityY] * dt_s;
  predicted.x[CenterZ] += predicted.x[VelocityZ] * dt_s;
  predicted.x[Theta] += predicted.x[Omega] * dt_s;
  return predicted;
}

Measurement observe(const State& state, int armor_slot) {
  if (!validSlot(armor_slot)) throw std::out_of_range("invalid armor slot");
  // 偶/奇槽位允许不同半径和高度；相邻物理槽位固定相差 90 度。
  const double parity = static_cast<double>(armor_slot % 2);
  const double phi = state.x[Theta] + armor_slot * (kPi * 0.5);
  const double radius = state.x[RadiusEven] + parity * state.x[RadiusOddDelta];
  Measurement measurement;
  measurement.position_T_m = {
      state.x[CenterX] - radius * std::cos(phi),
      state.x[CenterY] - radius * std::sin(phi),
      state.x[CenterZ] + parity * state.x[HeightOddDelta],
  };
  measurement.inward_yaw_T_rad = phi;
  measurement.has_inward_yaw = true;
  return measurement;
}

Jacobian observationJacobian(const State& state, int armor_slot) {
  if (!validSlot(armor_slot)) throw std::out_of_range("invalid armor slot");
  const double parity = static_cast<double>(armor_slot % 2);
  const double phi = state.x[Theta] + armor_slot * (kPi * 0.5);
  const double cosine = std::cos(phi);
  const double sine = std::sin(phi);
  const double radius = state.x[RadiusEven] + parity * state.x[RadiusOddDelta];

  Jacobian jacobian = Jacobian::Zero();
  jacobian(0, CenterX) = 1.0;
  jacobian(0, Theta) = radius * sine;
  jacobian(0, RadiusEven) = -cosine;
  jacobian(0, RadiusOddDelta) = -parity * cosine;
  jacobian(1, CenterY) = 1.0;
  jacobian(1, Theta) = -radius * cosine;
  jacobian(1, RadiusEven) = -sine;
  jacobian(1, RadiusOddDelta) = -parity * sine;
  jacobian(2, CenterZ) = 1.0;
  jacobian(2, HeightOddDelta) = parity;
  jacobian(3, Theta) = 1.0;
  return jacobian;
}

DecodedArmor decodeArmor(const State& state, int armor_slot, double horizon_s) {
  const State predicted = predictState(state, horizon_s);
  const Measurement observation = observe(predicted, armor_slot);
  return {armor_slot, observation.position_T_m, observation.inward_yaw_T_rad};
}

const char* trackingStateName(TrackingState state) noexcept {
  switch (state) {
    case TrackingState::Uninitialized:
      return "uninitialized";
    case TrackingState::Confirming:
      return "confirming";
    case TrackingState::Tracking:
      return "tracking";
    case TrackingState::TemporarilyLost:
      return "temporarily-lost";
    case TrackingState::Lost:
      return "lost";
  }
  return "unknown";
}

WholeVehicleEkf::WholeVehicleEkf(WholeVehicleEkfOptions options)
    : options_(options) {
  if (!finite(options_.radius_prior_m) || options_.radius_prior_m <= 0.0 ||
      !finite(options_.minimum_radius_m) || options_.minimum_radius_m <= 0.0 ||
      options_.minimum_radius_m >= options_.radius_prior_m ||
      !finite(options_.q_linear_acceleration) ||
      !finite(options_.q_angular_acceleration) ||
      !finite(options_.q_geometry) || options_.q_linear_acceleration < 0.0 ||
      options_.q_angular_acceleration < 0.0 || options_.q_geometry < 0.0 ||
      options_.confirming_hits <= 0 || options_.lost_frame_limit <= 0 ||
      !finite(options_.lost_time_limit_s) || options_.lost_time_limit_s <= 0.0 ||
      !finite(options_.maximum_frame_dt_s) || options_.maximum_frame_dt_s <= 0.0) {
    throw std::invalid_argument("invalid whole-vehicle EKF options");
  }
}

void WholeVehicleEkf::reset() noexcept {
  state_ = State{};
  has_state_ = false;
  last_timestamp_ns_ = 0;
  last_hit_timestamp_ns_ = 0;
  consecutive_hits_ = 0;
  consecutive_misses_ = 0;
  tracked_color_id_ = -1;
  tracked_number_id_ = -1;
  tracking_state_ = TrackingState::Uninitialized;
}

const WholeVehicleEkfOptions& WholeVehicleEkf::options() const noexcept {
  return options_;
}

TrackingState WholeVehicleEkf::trackingState() const noexcept {
  return tracking_state_;
}

bool WholeVehicleEkf::hasState() const noexcept { return has_state_; }

const State& WholeVehicleEkf::state() const noexcept { return state_; }

Matrix11 WholeVehicleEkf::processNoise(double dt_s) const {
  Matrix11 noise = Matrix11::Zero();
  // 连续白噪声加速度离散化：q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]。
  const auto addConstantVelocityNoise = [&](int position, int velocity,
                                            double spectral_density) {
    noise(position, position) = spectral_density * dt_s * dt_s * dt_s / 3.0;
    noise(position, velocity) = spectral_density * dt_s * dt_s / 2.0;
    noise(velocity, position) = noise(position, velocity);
    noise(velocity, velocity) = spectral_density * dt_s;
  };
  addConstantVelocityNoise(CenterX, VelocityX, options_.q_linear_acceleration);
  addConstantVelocityNoise(CenterY, VelocityY, options_.q_linear_acceleration);
  addConstantVelocityNoise(CenterZ, VelocityZ, options_.q_linear_acceleration);
  addConstantVelocityNoise(Theta, Omega, options_.q_angular_acceleration);
  noise(RadiusEven, RadiusEven) = options_.q_geometry * dt_s;
  noise(RadiusOddDelta, RadiusOddDelta) = options_.q_geometry * dt_s;
  noise(HeightOddDelta, HeightOddDelta) = options_.q_geometry * dt_s;
  return noise;
}

Matrix4 WholeVehicleEkf::measurementCovariance(
    const Measurement& measurement) const {
  // 低置信度、差关键点、差视角、大重投影 RMS 或远距离都会降低本帧权重。
  const double confidence = std::max(options_.minimum_quality, measurement.confidence);
  const double keypoint_quality =
      std::max(options_.minimum_quality, measurement.keypoint_quality);
  const double view_quality = std::max(options_.minimum_quality, measurement.view_quality);
  const double range_m = measurement.camera_range_m;
  const double scale =
      (1.0 + options_.reprojection_rms_scale * std::max(0.0, measurement.reprojection_rms_px)) *
      (1.0 + options_.range_noise_scale_per_m * range_m) /
      (confidence * keypoint_quality * view_quality);
  Matrix4 covariance = Matrix4::Zero();
  covariance(0, 0) = square(options_.position_std_xy_m * scale);
  covariance(1, 1) = square(options_.position_std_xy_m * scale);
  covariance(2, 2) = square(options_.position_std_z_m * scale);
  double yaw_std = options_.yaw_std_rad * scale;
  if (measurement.has_inward_yaw && finite(measurement.yaw_std_rad) &&
      measurement.yaw_std_rad > 0.0) {
    yaw_std = std::max(yaw_std, measurement.yaw_std_rad);
  }
  covariance(3, 3) = square(yaw_std);
  return covariance;
}

bool WholeVehicleEkf::identityConsistent(const Measurement& measurement) const {
  const bool color_consistent = tracked_color_id_ < 0 || measurement.color_id < 0 ||
                                tracked_color_id_ == measurement.color_id;
  const bool number_consistent = tracked_number_id_ < 0 || measurement.number_id < 0 ||
                                 tracked_number_id_ == measurement.number_id;
  return color_consistent && number_consistent;
}

std::optional<WholeVehicleEkf::Association> WholeVehicleEkf::associate(
    const std::vector<Measurement>& measurements,
    const std::vector<bool>& used_measurements,
    const std::array<bool, kArmorSlotCount>& used_slots) const {
  // number/color 只做车辆身份门控。实际 E0-E3 槽位由几何预测和 NIS 决定。
  std::optional<Association> best;
  for (std::size_t measurement_index = 0; measurement_index < measurements.size();
       ++measurement_index) {
    if (measurement_index >= used_measurements.size() ||
        used_measurements[measurement_index]) {
      continue;
    }
    const Measurement& measurement = measurements[measurement_index];
    if (!validMeasurement(measurement) || !identityConsistent(measurement)) continue;
    const Matrix4 covariance = measurementCovariance(measurement);
    for (int armor_slot = 0; armor_slot < kArmorSlotCount; ++armor_slot) {
      if (used_slots[static_cast<std::size_t>(armor_slot)]) continue;
      const Measurement predicted = observe(state_, armor_slot);
      const Jacobian h = observationJacobian(state_, armor_slot);
      if (measurement.has_inward_yaw) {
        // yaw 残差必须环绕到 [-pi, pi)，否则跨 +/-pi 会被误判为大跳变。
        Vector4 innovation;
        innovation.template head<3>() = measurement.position_T_m - predicted.position_T_m;
        innovation[3] = wrapToPi(measurement.inward_yaw_T_rad -
                                  predicted.inward_yaw_T_rad);
        const Matrix4 innovation_covariance = h * state_.covariance * h.transpose() + covariance;
        // NIS = y^T S^-1 y，LDLT 求解避免形成 S 的显式逆。
        Eigen::LDLT<Matrix4> ldlt(innovation_covariance);
        if (ldlt.info() != Eigen::Success) continue;
        const Vector4 solution = ldlt.solve(innovation);
        if (!solution.allFinite()) continue;
        const double nis = innovation.dot(solution);
        if (!finite(nis) || nis > options_.nis_gate_4d) continue;
        if (!best || nis < best->nis) {
          best = Association{static_cast<int>(measurement_index), armor_slot, nis,
                             true, covariance};
        }
      } else {
        const Eigen::Matrix<double, 3, kWholeVehicleStateDimension> h_position =
            h.template topRows<3>();
        const Eigen::Vector3d innovation = measurement.position_T_m - predicted.position_T_m;
        const Eigen::Matrix3d covariance_position = covariance.template topLeftCorner<3, 3>();
        const Eigen::Matrix3d innovation_covariance =
            h_position * state_.covariance * h_position.transpose() + covariance_position;
        Eigen::LDLT<Eigen::Matrix3d> ldlt(innovation_covariance);
        if (ldlt.info() != Eigen::Success) continue;
        const Eigen::Vector3d solution = ldlt.solve(innovation);
        if (!solution.allFinite()) continue;
        const double nis = innovation.dot(solution);
        if (!finite(nis) || nis > options_.nis_gate_3d) continue;
        if (!best || nis < best->nis) {
          best = Association{static_cast<int>(measurement_index), armor_slot, nis,
                             false, covariance};
        }
      }
    }
  }
  return best;
}

bool WholeVehicleEkf::applyUpdate(const Measurement& measurement,
                                  const Association& association) {
  const Measurement predicted = observe(state_, association.armor_slot);
  const Jacobian h = observationJacobian(state_, association.armor_slot);
  const Matrix11 identity = Matrix11::Identity();
  if (association.includes_yaw) {
    Vector4 innovation;
    innovation.template head<3>() = measurement.position_T_m - predicted.position_T_m;
    innovation[3] = wrapToPi(measurement.inward_yaw_T_rad - predicted.inward_yaw_T_rad);
    const Matrix4 innovation_covariance =
        h * state_.covariance * h.transpose() + association.covariance;
    Eigen::LDLT<Matrix4> ldlt(innovation_covariance);
    if (ldlt.info() != Eigen::Success) return false;
    // K=P*H^T*S^-1；求解 S*(H*P) 后转置得到 K。
    const Eigen::Matrix<double, 4, kWholeVehicleStateDimension> solved =
        ldlt.solve(h * state_.covariance);
    if (!solved.allFinite()) return false;
    const Eigen::Matrix<double, kWholeVehicleStateDimension, 4> gain =
        solved.transpose();
    state_.x += gain * innovation;
    const Matrix11 residual = identity - gain * h;
    // Joseph 形式在浮点误差下仍能维持协方差半正定。
    state_.covariance = residual * state_.covariance * residual.transpose() +
                        gain * association.covariance * gain.transpose();
  } else {
    const Eigen::Matrix<double, 3, kWholeVehicleStateDimension> h_position =
        h.template topRows<3>();
    const Eigen::Vector3d innovation = measurement.position_T_m - predicted.position_T_m;
    const Eigen::Matrix3d covariance_position =
        association.covariance.template topLeftCorner<3, 3>();
    const Eigen::Matrix3d innovation_covariance =
        h_position * state_.covariance * h_position.transpose() + covariance_position;
    Eigen::LDLT<Eigen::Matrix3d> ldlt(innovation_covariance);
    if (ldlt.info() != Eigen::Success) return false;
    const Eigen::Matrix<double, 3, kWholeVehicleStateDimension> solved =
        ldlt.solve(h_position * state_.covariance);
    if (!solved.allFinite()) return false;
    const Eigen::Matrix<double, kWholeVehicleStateDimension, 3> gain =
        solved.transpose();
    state_.x += gain * innovation;
    const Matrix11 residual = identity - gain * h_position;
    state_.covariance = residual * state_.covariance * residual.transpose() +
                        gain * covariance_position * gain.transpose();
  }
  state_.covariance = 0.5 * (state_.covariance + state_.covariance.transpose());
  return stateFiniteAndPhysical();
}

bool WholeVehicleEkf::initialize(const Measurement& measurement) {
  if (!validMeasurement(measurement) || !measurement.has_inward_yaw) return false;
  // 单块板不能同时解出中心和半径：将该板定义为 E0，并使用 r0 先验反推中心。
  state_ = State{};
  state_.x[Theta] = measurement.inward_yaw_T_rad;
  state_.x[RadiusEven] = options_.radius_prior_m;
  state_.x[CenterX] = measurement.position_T_m.x() +
                      options_.radius_prior_m * std::cos(state_.x[Theta]);
  state_.x[CenterY] = measurement.position_T_m.y() +
                      options_.radius_prior_m * std::sin(state_.x[Theta]);
  state_.x[CenterZ] = measurement.position_T_m.z();
  state_.covariance = Matrix11::Zero();
  state_.covariance(CenterX, CenterX) = square(options_.initial_position_std_m);
  state_.covariance(CenterY, CenterY) = square(options_.initial_position_std_m);
  state_.covariance(CenterZ, CenterZ) = square(options_.initial_position_std_m);
  state_.covariance(VelocityX, VelocityX) = square(options_.initial_velocity_std_mps);
  state_.covariance(VelocityY, VelocityY) = square(options_.initial_velocity_std_mps);
  state_.covariance(VelocityZ, VelocityZ) = square(options_.initial_velocity_std_mps);
  state_.covariance(Theta, Theta) = square(options_.initial_theta_std_rad);
  state_.covariance(Omega, Omega) = square(options_.initial_omega_std_rad_s);
  state_.covariance(RadiusEven, RadiusEven) = square(options_.initial_geometry_std_m);
  state_.covariance(RadiusOddDelta, RadiusOddDelta) = square(options_.initial_geometry_std_m);
  state_.covariance(HeightOddDelta, HeightOddDelta) = square(options_.initial_geometry_std_m);
  if (!stateFiniteAndPhysical()) return false;
  has_state_ = true;
  tracking_state_ = TrackingState::Confirming;
  last_timestamp_ns_ = measurement.timestamp_ns;
  last_hit_timestamp_ns_ = measurement.timestamp_ns;
  consecutive_hits_ = 1;
  consecutive_misses_ = 0;
  tracked_color_id_ = measurement.color_id;
  tracked_number_id_ = measurement.number_id;
  return true;
}

bool WholeVehicleEkf::stateFiniteAndPhysical() const {
  return state_.x.allFinite() && state_.covariance.allFinite() &&
         state_.x[RadiusEven] >= options_.minimum_radius_m &&
         state_.x[RadiusEven] + state_.x[RadiusOddDelta] >=
             options_.minimum_radius_m;
}

TrackOutput WholeVehicleEkf::makeOutput(
    std::uint64_t timestamp_ns,
    const std::vector<AssociatedObservation>& associations) const {
  TrackOutput output;
  output.timestamp_ns = timestamp_ns;
  output.tracking_state = tracking_state_;
  output.has_state = has_state_;
  output.associated_observations = associations;
  if (!associations.empty()) {
    output.associated_slot = associations.front().armor_slot;
    output.nis = associations.front().nis;
  }
  output.consecutive_hits = consecutive_hits_;
  output.consecutive_misses = consecutive_misses_;
  if (has_state_) {
    output.state = state_;
    output.center_T_m = {state_.x[CenterX], state_.x[CenterY],
                         state_.x[CenterZ]};
    output.velocity_T_mps = {state_.x[VelocityX], state_.x[VelocityY],
                             state_.x[VelocityZ]};
    output.theta_rad = state_.x[Theta];
    output.omega_rad_s = state_.x[Omega];
    output.radius_even_m = state_.x[RadiusEven];
    output.radius_odd_delta_m = state_.x[RadiusOddDelta];
    output.height_odd_delta_m = state_.x[HeightOddDelta];
    for (int slot = 0; slot < kArmorSlotCount; ++slot) {
      output.predicted_armors[static_cast<std::size_t>(slot)] =
          decodeArmor(state_, slot, 0.0);
    }
  }
  return output;
}

TrackOutput WholeVehicleEkf::update(
    std::uint64_t timestamp_ns, const std::vector<Measurement>& measurements) {
  if (timestamp_ns == 0) return makeOutput(timestamp_ns);
  for (const Measurement& measurement : measurements) {
    if (!validMeasurement(measurement) || measurement.timestamp_ns != timestamp_ns) {
      return makeOutput(timestamp_ns);
    }
  }

  // 未初始化时只接受可靠 yaw；position-only 量测不会凭空确定整车朝向。
  if (!has_state_) {
    std::vector<AssociatedObservation> associations;
    std::vector<bool> used_measurements(measurements.size(), false);
    std::array<bool, kArmorSlotCount> used_slots{};
    for (std::size_t index = 0; index < measurements.size(); ++index) {
      const Measurement& measurement = measurements[index];
      if (!initialize(measurement)) continue;

      // The initializing armor defines E0. Updating its zero residual also
      // records its measurement covariance before another armor constrains
      // center and radius in the same exposure.
      const Association initializer{static_cast<int>(index), 0, 0.0,
                                    measurement.has_inward_yaw,
                                    measurementCovariance(measurement)};
      const AssociatedObservation accepted = makeAssociatedObservation(
          state_, measurement, static_cast<int>(index), 0, 0.0,
          measurement.has_inward_yaw);
      if (!applyUpdate(measurement, initializer)) {
        reset();
        break;
      }
      associations.push_back(accepted);
      used_measurements[index] = true;
      used_slots[0] = true;
      break;
    }
    if (has_state_) {
      while (associations.size() < kArmorSlotCount) {
        const std::optional<Association> association =
            associate(measurements, used_measurements, used_slots);
        if (!association) {
          break;
        }
        const AssociatedObservation accepted = makeAssociatedObservation(
            state_, measurements[association->measurement_index],
            association->measurement_index, association->armor_slot,
            association->nis, association->includes_yaw);
        if (!applyUpdate(measurements[association->measurement_index], *association)) {
          break;
        }
        associations.push_back(accepted);
        used_measurements[static_cast<std::size_t>(association->measurement_index)] = true;
        used_slots[static_cast<std::size_t>(association->armor_slot)] = true;
      }
      return makeOutput(timestamp_ns, associations);
    }
    return makeOutput(timestamp_ns);
  }

  if (timestamp_ns <= last_timestamp_ns_) {
    return makeOutput(timestamp_ns);
  }
  const double dt_s = static_cast<double>(timestamp_ns - last_timestamp_ns_) * 1e-9;
  if (!finite(dt_s) || dt_s <= 0.0 || dt_s > options_.maximum_frame_dt_s) {
    reset();
    tracking_state_ = TrackingState::Lost;
    // A valid current observation may start a fresh track after a long gap.
    return update(timestamp_ns, measurements);
  }

  // dt 必须来自相邻曝光时间，不能使用处理耗时或固定 FPS。
  Matrix11 transition = Matrix11::Identity();
  transition(CenterX, VelocityX) = dt_s;
  transition(CenterY, VelocityY) = dt_s;
  transition(CenterZ, VelocityZ) = dt_s;
  transition(Theta, Omega) = dt_s;
  state_ = predictState(state_, dt_s);
  state_.covariance = transition * state_.covariance * transition.transpose() +
                      processNoise(dt_s);
  state_.covariance = 0.5 * (state_.covariance + state_.covariance.transpose());
  last_timestamp_ns_ = timestamp_ns;
  if (!stateFiniteAndPhysical()) {
    reset();
    tracking_state_ = TrackingState::Lost;
    return makeOutput(timestamp_ns);
  }

  // Associate without replacement. Each accepted detection and each physical
  // E0-E3 slot is used at most once; after an update, association is repeated
  // against the corrected state so two visible armors jointly constrain it.
  std::vector<AssociatedObservation> associations;
  std::vector<bool> used_measurements(measurements.size(), false);
  std::array<bool, kArmorSlotCount> used_slots{};
  while (associations.size() < kArmorSlotCount) {
    const std::optional<Association> association =
        associate(measurements, used_measurements, used_slots);
    if (!association) {
      break;
    }
    const AssociatedObservation accepted = makeAssociatedObservation(
        state_, measurements[association->measurement_index],
        association->measurement_index, association->armor_slot,
        association->nis, association->includes_yaw);
    if (!applyUpdate(measurements[association->measurement_index], *association)) {
      break;
    }
    associations.push_back(accepted);
    used_measurements[static_cast<std::size_t>(association->measurement_index)] = true;
    used_slots[static_cast<std::size_t>(association->armor_slot)] = true;
  }
  if (!associations.empty()) {
    const Measurement& measurement = measurements[associations.front().measurement_index];
    if (tracked_color_id_ < 0) tracked_color_id_ = measurement.color_id;
    if (tracked_number_id_ < 0) tracked_number_id_ = measurement.number_id;
    ++consecutive_hits_;
    consecutive_misses_ = 0;
    last_hit_timestamp_ns_ = timestamp_ns;
    if (tracking_state_ == TrackingState::Confirming &&
        consecutive_hits_ >= options_.confirming_hits) {
      tracking_state_ = TrackingState::Tracking;
    } else if (tracking_state_ == TrackingState::TemporarilyLost) {
      tracking_state_ = TrackingState::Tracking;
    }
    return makeOutput(timestamp_ns, associations);
  }

  consecutive_hits_ = 0;
  ++consecutive_misses_;
  const double since_hit_s = static_cast<double>(timestamp_ns - last_hit_timestamp_ns_) * 1e-9;
  if (consecutive_misses_ > options_.lost_frame_limit ||
      since_hit_s > options_.lost_time_limit_s) {
    reset();
    tracking_state_ = TrackingState::Lost;
  } else {
    tracking_state_ = TrackingState::TemporarilyLost;
  }
  return makeOutput(timestamp_ns);
}

}  // namespace yolo_detect::tracking
