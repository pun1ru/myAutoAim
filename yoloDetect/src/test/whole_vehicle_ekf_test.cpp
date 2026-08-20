#include "tracking/whole_vehicle_ekf.hpp"

#include <Eigen/Eigenvalues>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tracking = yolo_detect::tracking;

namespace {

constexpr double kPi = 3.14159265358979323846;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void requireNear(double actual, double expected, double tolerance,
                 const std::string& message) {
  require(std::abs(actual - expected) <= tolerance,
          message + " actual=" + std::to_string(actual) +
              " expected=" + std::to_string(expected));
}

tracking::State syntheticState() {
  tracking::State state;
  state.x << 4.0, 1.2, -1.5, -0.4, 0.7, 0.1, 0.35, 2.0, 0.28, 0.02,
      0.04;
  state.covariance = tracking::Matrix11::Identity() * 0.1;
  return state;
}

tracking::Measurement observationAt(const tracking::State& state, int slot,
                                    std::uint64_t timestamp_ns,
                                    int color_id = 1, int number_id = 3) {
  const tracking::Measurement predicted = tracking::observe(state, slot);
  tracking::Measurement measurement;
  measurement.timestamp_ns = timestamp_ns;
  measurement.position_T_m = predicted.position_T_m;
  measurement.inward_yaw_T_rad = predicted.inward_yaw_T_rad;
  measurement.reprojection_rms_px = 0.2;
  measurement.confidence = 0.99;
  measurement.keypoint_quality = 1.0;
  measurement.view_quality = 1.0;
  measurement.color_id = color_id;
  measurement.number_id = number_id;
  measurement.has_inward_yaw = true;
  return measurement;
}

void testNoiselessPredictObserve() {
  const tracking::State state = syntheticState();
  const tracking::State predicted = tracking::predictState(state, 0.25);
  requireNear(predicted.x[tracking::CenterX], 4.3, 1e-12,
              "constant velocity x prediction");
  requireNear(predicted.x[tracking::Theta], 0.85, 1e-12,
              "constant angular velocity prediction");
  const tracking::Measurement slot_three = tracking::observe(state, 3);
  const double phi = state.x[tracking::Theta] + 1.5 * kPi;
  const double radius = state.x[tracking::RadiusEven] +
                        state.x[tracking::RadiusOddDelta];
  requireNear(slot_three.position_T_m.x(),
              state.x[tracking::CenterX] - radius * std::cos(phi), 1e-12,
              "slot observation x");
  requireNear(slot_three.position_T_m.y(),
              state.x[tracking::CenterY] - radius * std::sin(phi), 1e-12,
              "slot observation y");
  requireNear(slot_three.position_T_m.z(),
              state.x[tracking::CenterZ] + state.x[tracking::HeightOddDelta],
              1e-12, "slot observation z");
}

void testAnalyticJacobianAgainstFiniteDifference() {
  const tracking::State state = syntheticState();
  constexpr double kStep = 1e-6;
  for (int slot = 0; slot < tracking::kArmorSlotCount; ++slot) {
    const tracking::Jacobian analytic = tracking::observationJacobian(state, slot);
    for (int column = 0; column < tracking::kWholeVehicleStateDimension;
         ++column) {
      tracking::State plus = state;
      tracking::State minus = state;
      plus.x[column] += kStep;
      minus.x[column] -= kStep;
      const tracking::Measurement z_plus = tracking::observe(plus, slot);
      const tracking::Measurement z_minus = tracking::observe(minus, slot);
      const tracking::Vector4 delta = {
          z_plus.position_T_m.x() - z_minus.position_T_m.x(),
          z_plus.position_T_m.y() - z_minus.position_T_m.y(),
          z_plus.position_T_m.z() - z_minus.position_T_m.z(),
          tracking::wrapToPi(z_plus.inward_yaw_T_rad - z_minus.inward_yaw_T_rad),
      };
      const tracking::Vector4 numeric = delta / (2.0 * kStep);
      require((numeric - analytic.col(column)).cwiseAbs().maxCoeff() < 2e-6,
              "analytic observation Jacobian differs from finite difference");
    }
  }
}

void testYawWrap() {
  const double innovation = tracking::wrapToPi(-179.0 * kPi / 180.0 -
                                                179.0 * kPi / 180.0);
  requireNear(innovation, 2.0 * kPi / 180.0, 1e-12,
              "yaw innovation must cross pi boundary through two degrees");
}

tracking::TrackOutput initializeTracker(tracking::WholeVehicleEkf& ekf,
                                        const tracking::State& state,
                                        std::uint64_t timestamp_ns) {
  tracking::Measurement first = observationAt(state, 0, timestamp_ns);
  const tracking::TrackOutput output = ekf.update(timestamp_ns, {first});
  require(output.has_state && output.tracking_state == tracking::TrackingState::Confirming,
          "first reliable yaw observation must initialize tracker");
  return output;
}

void testSlotSwitchDoesNotJumpCenter() {
  tracking::WholeVehicleEkfOptions options;
  options.confirming_hits = 2;
  tracking::WholeVehicleEkf ekf(options);
  tracking::State state = syntheticState();
  state.x[tracking::RadiusEven] = options.radius_prior_m;
  state.x[tracking::RadiusOddDelta] = 0.0;
  state.x[tracking::HeightOddDelta] = 0.0;
  state.x[tracking::VelocityX] = 0.0;
  state.x[tracking::VelocityY] = 0.0;
  state.x[tracking::VelocityZ] = 0.0;
  state.x[tracking::Omega] = 0.0;
  const std::uint64_t t0 = 1'000'000'000ULL;
  initializeTracker(ekf, state, t0);
  const tracking::State before_switch = ekf.state();
  const tracking::Measurement slot_one = observationAt(state, 1, t0 + 10'000'000ULL);
  const tracking::TrackOutput output = ekf.update(slot_one.timestamp_ns, {slot_one});
  require(output.associated_slot.has_value() && *output.associated_slot == 1,
          "yaw-aware association must select the visible next physical slot");
  const Eigen::Vector3d center_before(before_switch.x[tracking::CenterX],
                                      before_switch.x[tracking::CenterY],
                                      before_switch.x[tracking::CenterZ]);
  const Eigen::Vector3d center_after(output.state.x[tracking::CenterX],
                                     output.state.x[tracking::CenterY],
                                     output.state.x[tracking::CenterZ]);
  require((center_after - center_before).norm() < 0.04,
          "changing visible armor slot must not jump vehicle center");
}

void testNisOutlierRejected() {
  tracking::WholeVehicleEkf ekf;
  const tracking::State state = syntheticState();
  const std::uint64_t t0 = 2'000'000'000ULL;
  initializeTracker(ekf, state, t0);
  tracking::Measurement outlier = observationAt(state, 0, t0 + 10'000'000ULL);
  outlier.position_T_m += Eigen::Vector3d(20.0, -15.0, 10.0);
  const tracking::TrackOutput output = ekf.update(outlier.timestamp_ns, {outlier});
  require(!output.associated_slot.has_value() && !output.nis.has_value(),
          "NIS-gated outlier must not update state");
  require(output.tracking_state == tracking::TrackingState::TemporarilyLost,
          "outlier frame must transition to temporarily lost");
}

void testLostPredictionAndReset() {
  tracking::WholeVehicleEkfOptions options;
  options.lost_frame_limit = 2;
  options.lost_time_limit_s = 1.0;
  options.maximum_frame_dt_s = 0.2;
  tracking::WholeVehicleEkf ekf(options);
  tracking::State state = syntheticState();
  state.x[tracking::VelocityX] = 2.0;
  const std::uint64_t t0 = 3'000'000'000ULL;
  initializeTracker(ekf, state, t0);
  const tracking::TrackOutput predicted = ekf.update(t0 + 20'000'000ULL, {});
  require(predicted.has_state &&
              predicted.tracking_state == tracking::TrackingState::TemporarilyLost,
          "missed frame must retain predicted track");
  require(predicted.state.x[tracking::CenterX] > ekf.options().radius_prior_m,
          "missed frame must advance state by exposure timestamp dt");
  static_cast<void>(ekf.update(t0 + 40'000'000ULL, {}));
  const tracking::TrackOutput reset = ekf.update(t0 + 60'000'000ULL, {});
  require(!reset.has_state && reset.tracking_state == tracking::TrackingState::Lost,
          "loss thresholds must clear tracker state");
}

void testJosephCovariance() {
  tracking::WholeVehicleEkfOptions options;
  options.confirming_hits = 1;
  tracking::WholeVehicleEkf ekf(options);
  const tracking::State state = syntheticState();
  const std::uint64_t t0 = 4'000'000'000ULL;
  initializeTracker(ekf, state, t0);
  const tracking::Measurement update = observationAt(state, 0, t0 + 20'000'000ULL);
  const tracking::TrackOutput output = ekf.update(update.timestamp_ns, {update});
  const tracking::Matrix11 asymmetry = output.state.covariance - output.state.covariance.transpose();
  require(output.state.covariance.allFinite() && asymmetry.cwiseAbs().maxCoeff() < 1e-12,
          "Joseph covariance must be finite and symmetric");
  const Eigen::SelfAdjointEigenSolver<tracking::Matrix11> eigenvalues(output.state.covariance);
  require(eigenvalues.info() == Eigen::Success &&
              eigenvalues.eigenvalues().minCoeff() > -1e-10,
          "Joseph covariance must remain positive semidefinite");
}

tracking::Vector11 simulateDifferentDt(const std::vector<double>& steps) {
  tracking::WholeVehicleEkfOptions options;
  options.confirming_hits = 1;
  tracking::WholeVehicleEkf ekf(options);
  tracking::State truth = syntheticState();
  truth.x[tracking::VelocityX] = 0.7;
  truth.x[tracking::VelocityY] = -0.2;
  truth.x[tracking::Omega] = 0.4;
  std::uint64_t timestamp_ns = 5'000'000'000ULL;
  initializeTracker(ekf, truth, timestamp_ns);
  for (double dt_s : steps) {
    truth = tracking::predictState(truth, dt_s);
    timestamp_ns += static_cast<std::uint64_t>(std::llround(dt_s * 1e9));
    const tracking::Measurement measurement = observationAt(truth, 0, timestamp_ns);
    static_cast<void>(ekf.update(timestamp_ns, {measurement}));
  }
  return ekf.state().x;
}

void testIrregularDtTrend() {
  const tracking::Vector11 uniform = simulateDifferentDt({0.02, 0.02, 0.02, 0.02, 0.02});
  const tracking::Vector11 irregular = simulateDifferentDt({0.013, 0.031, 0.009, 0.027, 0.020});
  require((uniform - irregular).template head<8>().norm() < 0.08,
          "irregular exposure intervals must preserve motion trend");
}

}  // namespace

int main() {
  try {
    testNoiselessPredictObserve();
    testAnalyticJacobianAgainstFiniteDifference();
    testYawWrap();
    testSlotSwitchDoesNotJumpCenter();
    testNisOutlierRejected();
    testLostPredictionAndReset();
    testJosephCovariance();
    testIrregularDtTrend();
    std::cout << "whole vehicle EKF tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "whole vehicle EKF test failed: " << error.what() << '\n';
    return 1;
  }
}
