#include "tracking/whole_vehicle_ekf.hpp"

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <functional>
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
  const bool camera_geometry_valid =
      !measurement.has_exposure_camera_geometry ||
      (measurement.R_TC.allFinite() &&
       measurement.camera_position_T_m.allFinite());
  return measurement.timestamp_ns != 0 && measurement.position_T_m.allFinite() &&
         finite(measurement.camera_range_m) && measurement.camera_range_m > 0.0 &&
         finite(measurement.reprojection_rms_px) &&
         finite(measurement.confidence) && finite(measurement.keypoint_quality) &&
         finite(measurement.view_quality) && camera_geometry_valid &&
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
  if (measurement.has_inward_yaw) {
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
      !finite(options_.maximum_radius_m) ||
      options_.maximum_radius_m <= options_.radius_prior_m ||
      !finite(options_.maximum_radius_difference_m) ||
      options_.maximum_radius_difference_m < 0.0 ||
      !finite(options_.maximum_height_difference_m) ||
      options_.maximum_height_difference_m < 0.0 ||
      !finite(options_.q_linear_acceleration) ||
      !finite(options_.q_angular_acceleration) ||
      !finite(options_.q_geometry) || options_.q_linear_acceleration < 0.0 ||
      options_.q_angular_acceleration < 0.0 || options_.q_geometry < 0.0 ||
      !finite(options_.maximum_yaw_update_innovation_rad) ||
      options_.maximum_yaw_update_innovation_rad <= 0.0 ||
      !finite(options_.maximum_yaw_association_innovation_rad) ||
      options_.maximum_yaw_association_innovation_rad <
          options_.maximum_yaw_update_innovation_rad ||
      !finite(options_.yaw_phase_cost_std_rad) ||
      options_.yaw_phase_cost_std_rad <= 0.0 ||
      !finite(options_.geometry_yaw_consistency_rad) ||
      options_.geometry_yaw_consistency_rad <= 0.0 ||
      !finite(options_.geometry_minimum_baseline_m) ||
      options_.geometry_minimum_baseline_m <= 0.0 ||
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
  last_associated_slots_.fill(false);
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
  Eigen::Matrix3d position_covariance_camera = Eigen::Matrix3d::Zero();
  position_covariance_camera(0, 0) =
      square(options_.position_std_xy_m * scale);
  position_covariance_camera(1, 1) =
      square(options_.position_std_xy_m * scale);
  position_covariance_camera(2, 2) =
      square(options_.position_std_z_m * scale);
  const Eigen::Matrix3d position_covariance_tracker =
      measurement.has_exposure_camera_geometry
          ? measurement.R_TC * position_covariance_camera *
                measurement.R_TC.transpose()
          : position_covariance_camera;
  covariance.template topLeftCorner<3, 3>() =
      0.5 * (position_covariance_tracker +
             position_covariance_tracker.transpose());
  double yaw_std = options_.yaw_std_rad * scale;
  if (measurement.has_inward_yaw && finite(measurement.yaw_std_rad) &&
      measurement.yaw_std_rad > 0.0) {
    yaw_std = std::max(yaw_std, measurement.yaw_std_rad);
  }
  covariance(3, 3) = square(yaw_std);
  return covariance;
}

bool WholeVehicleEkf::identityConsistent(const Measurement& measurement) const {
  // The number classifier is the stronger target identity. Color classifiers
  // can disagree between two faces of the same vehicle, so only use color as
  // the fallback identity when either number is unavailable.
  if (tracked_number_id_ >= 0 && measurement.number_id >= 0) {
    return tracked_number_id_ == measurement.number_id;
  }
  return tracked_color_id_ < 0 || measurement.color_id < 0 ||
         tracked_color_id_ == measurement.color_id;
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
                             true, nis, covariance};
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
                             false, nis, covariance};
        }
      }
    }
  }
  return best;
}

bool WholeVehicleEkf::applyUpdate(const Measurement& measurement,
                                  const Association& association) {
  const State prior = state_;
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
  if (!stateFiniteAndPhysical()) {
    state_ = prior;
    return false;
  }
  return true;
}

bool WholeVehicleEkf::slotVisible(const Measurement& measurement,
                                  int armor_slot) const {
  if (!measurement.has_exposure_camera_geometry) return true;
  const Measurement predicted = observe(state_, armor_slot);
  if (measurement.has_inward_yaw) {
    const double yaw_innovation = wrapToPi(
        measurement.inward_yaw_T_rad - predicted.inward_yaw_T_rad);
    if (!finite(yaw_innovation) ||
        std::abs(yaw_innovation) >
            options_.maximum_yaw_update_innovation_rad) {
      // A normal inferred from unreliable yaw must not veto a valid position.
      return true;
    }
  }
  const Eigen::Vector3d to_camera =
      measurement.camera_position_T_m - predicted.position_T_m;
  if (to_camera.squaredNorm() <= 1e-12) return false;
  const double phi = predicted.inward_yaw_T_rad;
  const Eigen::Vector3d outward_normal(-std::cos(phi), -std::sin(phi), 0.0);
  const double facing_cosine = outward_normal.dot(to_camera.normalized());
  return finite(facing_cosine) &&
         facing_cosine >= options_.minimum_visibility_cosine;
}

double WholeVehicleEkf::slotTransitionPenalty(int armor_slot) const {
  bool has_previous_slot = false;
  int minimum_distance = kArmorSlotCount;
  for (int previous = 0; previous < kArmorSlotCount; ++previous) {
    if (!last_associated_slots_[static_cast<std::size_t>(previous)]) continue;
    has_previous_slot = true;
    const int direct = std::abs(armor_slot - previous);
    minimum_distance = std::min(minimum_distance,
                                std::min(direct, kArmorSlotCount - direct));
  }
  if (!has_previous_slot || minimum_distance == 0) return 0.0;
  const double loss_scale = consecutive_misses_ > 0 ? 0.5 : 1.0;
  return loss_scale * (minimum_distance == 1
                           ? options_.adjacent_slot_penalty
                           : options_.opposite_slot_penalty);
}

std::optional<WholeVehicleEkf::Association>
WholeVehicleEkf::makeAssociationCandidate(
    const std::vector<Measurement>& measurements, int measurement_index,
    int armor_slot) const {
  if (measurement_index < 0 ||
      measurement_index >= static_cast<int>(measurements.size()) ||
      !validSlot(armor_slot)) {
    return std::nullopt;
  }
  const Measurement& measurement =
      measurements[static_cast<std::size_t>(measurement_index)];
  if (!validMeasurement(measurement) || !identityConsistent(measurement) ||
      !slotVisible(measurement, armor_slot)) {
    return std::nullopt;
  }

  const Measurement predicted = observe(state_, armor_slot);
  const Jacobian h = observationJacobian(state_, armor_slot);
  const Matrix4 covariance = measurementCovariance(measurement);
  const Eigen::Vector3d position_innovation =
      measurement.position_T_m - predicted.position_T_m;
  const Eigen::Matrix<double, 3, kWholeVehicleStateDimension> h_position =
      h.template topRows<3>();
  const Eigen::Matrix3d position_covariance =
      covariance.template topLeftCorner<3, 3>();

  double yaw_innovation = 0.0;
  const bool has_yaw = measurement.has_inward_yaw;
  if (has_yaw) {
    yaw_innovation = wrapToPi(measurement.inward_yaw_T_rad -
                              predicted.inward_yaw_T_rad);
    if (!finite(yaw_innovation) ||
        std::abs(yaw_innovation) >
            options_.maximum_yaw_association_innovation_rad) {
      return std::nullopt;
    }
  }

  const bool include_yaw =
      has_yaw && std::abs(yaw_innovation) <=
                     options_.maximum_yaw_update_innovation_rad;
  double nis = 0.0;
  if (include_yaw) {
    Vector4 innovation;
    innovation.template head<3>() = position_innovation;
    innovation[3] = yaw_innovation;
    const Matrix4 innovation_covariance =
        h * state_.covariance * h.transpose() + covariance;
    Eigen::LDLT<Matrix4> ldlt(innovation_covariance);
    if (ldlt.info() != Eigen::Success) return std::nullopt;
    const Vector4 solved = ldlt.solve(innovation);
    if (!solved.allFinite()) return std::nullopt;
    nis = innovation.dot(solved);
    if (!finite(nis) || nis > options_.nis_gate_4d) return std::nullopt;
  } else {
    const Eigen::Matrix3d innovation_covariance =
        h_position * state_.covariance * h_position.transpose() +
        position_covariance;
    Eigen::LDLT<Eigen::Matrix3d> ldlt(innovation_covariance);
    if (ldlt.info() != Eigen::Success) return std::nullopt;
    const Eigen::Vector3d solved = ldlt.solve(position_innovation);
    if (!solved.allFinite()) return std::nullopt;
    nis = position_innovation.dot(solved);
    if (!finite(nis) || nis > options_.nis_gate_3d) return std::nullopt;
  }

  double cost = nis + slotTransitionPenalty(armor_slot);
  if (has_yaw && !include_yaw) {
    cost += square(yaw_innovation / options_.yaw_phase_cost_std_rad);
  }
  return Association{measurement_index, armor_slot, nis, include_yaw, cost,
                     covariance};
}

std::vector<WholeVehicleEkf::Association> WholeVehicleEkf::associateAll(
    const std::vector<Measurement>& measurements) const {
  std::vector<std::array<std::optional<Association>, kArmorSlotCount>> candidates(
      measurements.size());
  for (std::size_t measurement_index = 0;
       measurement_index < measurements.size(); ++measurement_index) {
    for (int slot = 0; slot < kArmorSlotCount; ++slot) {
      candidates[measurement_index][static_cast<std::size_t>(slot)] =
          makeAssociationCandidate(measurements,
                                   static_cast<int>(measurement_index), slot);
    }
  }

  std::vector<Association> best;
  std::vector<Association> current;
  std::vector<bool> used_measurements(measurements.size(), false);
  double best_cost = std::numeric_limits<double>::infinity();
  std::function<void(int, double)> search = [&](int slot, double cost) {
    if (slot == kArmorSlotCount) {
      if (current.size() > best.size() ||
          (current.size() == best.size() && cost < best_cost)) {
        best = current;
        best_cost = cost;
      }
      return;
    }
    search(slot + 1, cost);
    for (std::size_t measurement_index = 0;
         measurement_index < measurements.size(); ++measurement_index) {
      if (used_measurements[measurement_index]) continue;
      const auto& candidate =
          candidates[measurement_index][static_cast<std::size_t>(slot)];
      if (!candidate) continue;
      used_measurements[measurement_index] = true;
      current.push_back(*candidate);
      search(slot + 1, cost + candidate->cost);
      current.pop_back();
      used_measurements[measurement_index] = false;
    }
  };
  search(0, 0.0);
  return best;
}

bool WholeVehicleEkf::geometryObservable(
    const std::vector<Measurement>& measurements,
    const std::vector<Association>& associations) const {
  if (associations.size() < 2) return false;

  bool has_reliable_yaw = false;
  bool has_even_slot = false;
  bool has_odd_slot = false;
  for (std::size_t first = 0; first < associations.size(); ++first) {
    const Association& a = associations[first];
    const Measurement& measurement_a =
        measurements[static_cast<std::size_t>(a.measurement_index)];
    has_reliable_yaw = has_reliable_yaw || a.includes_yaw;
    has_even_slot = has_even_slot || (a.armor_slot % 2 == 0);
    has_odd_slot = has_odd_slot || (a.armor_slot % 2 != 0);
    for (std::size_t second = first + 1; second < associations.size(); ++second) {
      const Association& b = associations[second];
      const Measurement& measurement_b =
          measurements[static_cast<std::size_t>(b.measurement_index)];
      if ((measurement_a.position_T_m - measurement_b.position_T_m).norm() <
          options_.geometry_minimum_baseline_m) {
        return false;
      }
      // An unreliable yaw is deliberately position-only. Only compare phase
      // when both constrained-yaw solutions were accepted for this update.
      if (!a.includes_yaw || !b.includes_yaw) continue;
      const double measured_delta = wrapToPi(
          measurement_a.inward_yaw_T_rad - measurement_b.inward_yaw_T_rad);
      const double expected_delta = wrapToPi(
          (a.armor_slot - b.armor_slot) * (kPi * 0.5));
      if (std::abs(wrapToPi(measured_delta - expected_delta)) >
          options_.geometry_yaw_consistency_rad) {
        return false;
      }
    }
  }
  return has_reliable_yaw && has_even_slot && has_odd_slot;
}

bool WholeVehicleEkf::applyJointUpdate(
    const std::vector<Measurement>& measurements,
    const std::vector<Association>& associations, bool update_geometry) {
  if (associations.empty() || associations.size() > kArmorSlotCount) return false;
  const State prior = state_;
  StackedJacobian h_stacked = StackedJacobian::Zero();
  Vector16 innovation = Vector16::Zero();
  Matrix16 covariance = Matrix16::Identity();

  for (std::size_t index = 0; index < associations.size(); ++index) {
    const Association& association = associations[index];
    const Measurement& measurement =
        measurements[static_cast<std::size_t>(association.measurement_index)];
    const Measurement predicted = observe(prior, association.armor_slot);
    Jacobian h = observationJacobian(prior, association.armor_slot);
    Matrix4 block_covariance = association.covariance;
    Vector4 block_innovation = Vector4::Zero();
    block_innovation.template head<3>() =
        measurement.position_T_m - predicted.position_T_m;

    if (!update_geometry) {
      h.col(RadiusEven).setZero();
      h.col(RadiusOddDelta).setZero();
      h.col(HeightOddDelta).setZero();
    }
    if (association.includes_yaw) {
      block_innovation[3] = wrapToPi(
          measurement.inward_yaw_T_rad - predicted.inward_yaw_T_rad);
    } else {
      h.row(3).setZero();
      block_covariance.row(3).setZero();
      block_covariance.col(3).setZero();
      block_covariance(3, 3) = 1.0;
    }

    const int row = static_cast<int>(index) * kArmorObservationDimension;
    h_stacked.template block<kArmorObservationDimension,
                             kWholeVehicleStateDimension>(row, 0) = h;
    innovation.template segment<kArmorObservationDimension>(row) =
        block_innovation;
    covariance.template block<kArmorObservationDimension,
                              kArmorObservationDimension>(row, row) =
        block_covariance;
  }

  const Matrix16 innovation_covariance =
      h_stacked * prior.covariance * h_stacked.transpose() + covariance;
  Eigen::LDLT<Matrix16> ldlt(innovation_covariance);
  if (ldlt.info() != Eigen::Success) return false;
  const Eigen::Matrix<double, kMaximumStackedObservationDimension,
                      kWholeVehicleStateDimension>
      solved = ldlt.solve(h_stacked * prior.covariance);
  if (!solved.allFinite()) return false;
  Eigen::Matrix<double, kWholeVehicleStateDimension,
                kMaximumStackedObservationDimension>
      gain = solved.transpose();
  if (!update_geometry) {
    gain.row(RadiusEven).setZero();
    gain.row(RadiusOddDelta).setZero();
    gain.row(HeightOddDelta).setZero();
  }

  state_.x = prior.x + gain * innovation;
  const Matrix11 residual = Matrix11::Identity() - gain * h_stacked;
  state_.covariance = residual * prior.covariance * residual.transpose() +
                      gain * covariance * gain.transpose();
  state_.covariance =
      0.5 * (state_.covariance + state_.covariance.transpose());
  if (!stateFiniteAndPhysical()) {
    state_ = prior;
    return false;
  }
  return true;
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
  const double even_radius = state_.x[RadiusEven];
  const double odd_radius = even_radius + state_.x[RadiusOddDelta];
  return state_.x.allFinite() && state_.covariance.allFinite() &&
         even_radius >= options_.minimum_radius_m &&
         even_radius <= options_.maximum_radius_m &&
         odd_radius >= options_.minimum_radius_m &&
         odd_radius <= options_.maximum_radius_m &&
         std::abs(state_.x[RadiusOddDelta]) <=
             options_.maximum_radius_difference_m &&
         std::abs(state_.x[HeightOddDelta]) <=
             options_.maximum_height_difference_m;
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
    for (const Measurement& measurement : measurements) {
      if (!initialize(measurement)) continue;
      const std::vector<Association> matched = associateAll(measurements);
      std::vector<AssociatedObservation> accepted;
      accepted.reserve(matched.size());
      for (const Association& association : matched) {
        accepted.push_back(makeAssociatedObservation(
            state_, measurements[static_cast<std::size_t>(
                        association.measurement_index)],
            association.measurement_index, association.armor_slot,
            association.nis, association.includes_yaw));
      }
      const bool update_geometry = geometryObservable(measurements, matched);
      if (!matched.empty() &&
          !applyJointUpdate(measurements, matched, update_geometry) &&
          !applyJointUpdate(measurements, matched, false)) {
        reset();
        return makeOutput(timestamp_ns);
      }
      last_associated_slots_.fill(false);
      for (const Association& association : matched) {
        last_associated_slots_[static_cast<std::size_t>(
            association.armor_slot)] = true;
      }
      return makeOutput(timestamp_ns, accepted);
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

  // Every candidate is evaluated against the same predicted snapshot. The
  // accepted one-to-one assignment is then applied as one fixed-size joint
  // update, so detector ordering cannot change the posterior.
  const std::vector<Association> matched = associateAll(measurements);
  std::vector<AssociatedObservation> accepted;
  accepted.reserve(matched.size());
  for (const Association& association : matched) {
    accepted.push_back(makeAssociatedObservation(
        state_, measurements[static_cast<std::size_t>(
                    association.measurement_index)],
        association.measurement_index, association.armor_slot,
        association.nis, association.includes_yaw));
  }
  bool update_succeeded = false;
  if (!matched.empty()) {
    const bool update_geometry = geometryObservable(measurements, matched);
    update_succeeded = applyJointUpdate(measurements, matched, update_geometry);
    if (!update_succeeded && update_geometry) {
      update_succeeded = applyJointUpdate(measurements, matched, false);
    }
  }
  if (update_succeeded) {
    const Measurement& measurement = measurements[static_cast<std::size_t>(
        matched.front().measurement_index)];
    if (tracked_color_id_ < 0) tracked_color_id_ = measurement.color_id;
    if (tracked_number_id_ < 0) tracked_number_id_ = measurement.number_id;
    last_associated_slots_.fill(false);
    for (const Association& association : matched) {
      last_associated_slots_[static_cast<std::size_t>(association.armor_slot)] =
          true;
    }
    ++consecutive_hits_;
    consecutive_misses_ = 0;
    last_hit_timestamp_ns_ = timestamp_ns;
    if (tracking_state_ == TrackingState::Confirming &&
        consecutive_hits_ >= options_.confirming_hits) {
      tracking_state_ = TrackingState::Tracking;
    } else if (tracking_state_ == TrackingState::TemporarilyLost) {
      tracking_state_ = TrackingState::Tracking;
    }
    return makeOutput(timestamp_ns, accepted);
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
