#include "tracking/whole_vehicle_ekf.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
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
  measurement.camera_range_m = 5.0;
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

void testTwoVisibleArmorsUpdateOneFrame() {
  tracking::WholeVehicleEkfOptions options;
  options.radius_prior_m = 0.26;
  options.confirming_hits = 1;
  tracking::WholeVehicleEkf ekf(options);
  tracking::State truth = syntheticState();
  truth.x[tracking::RadiusEven] = 0.32;
  truth.x[tracking::RadiusOddDelta] = 0.0;
  truth.x[tracking::HeightOddDelta] = 0.0;
  truth.x[tracking::VelocityX] = 0.0;
  truth.x[tracking::VelocityY] = 0.0;
  truth.x[tracking::VelocityZ] = 0.0;
  truth.x[tracking::Omega] = 0.0;
  const std::uint64_t timestamp_ns = 2'500'000'000ULL;
  tracking::Measurement second = observationAt(truth, 1, timestamp_ns);
  second.color_id = 2;  // Simulate an inconsistent color classification.
  const tracking::TrackOutput output = ekf.update(
      timestamp_ns, {observationAt(truth, 0, timestamp_ns),
                     second});
  require(output.associated_observations.size() == 2,
          "two visible armors must both update the same EKF frame");
  require(output.associated_observations[0].armor_slot == 0 &&
              output.associated_observations[1].armor_slot == 1,
          "one-to-one association must keep the two physical armor slots distinct");
  require(output.associated_observations[1].measurement_index == 1,
          "matching number IDs must allow a second armor despite color disagreement");
  require(std::abs(output.radius_even_m - truth.x[tracking::RadiusEven]) <
              std::abs(options.radius_prior_m - truth.x[tracking::RadiusEven]),
          "the second armor observation must reduce radius-prior error");
}

void testSingleArmorDoesNotChangeGeometry() {
  tracking::WholeVehicleEkfOptions options;
  options.confirming_hits = 1;
  tracking::WholeVehicleEkf ekf(options);
  tracking::State truth = syntheticState();
  truth.x[tracking::RadiusEven] = options.radius_prior_m;
  truth.x[tracking::RadiusOddDelta] = 0.0;
  truth.x[tracking::HeightOddDelta] = 0.0;
  truth.x[tracking::VelocityX] = 0.0;
  truth.x[tracking::VelocityY] = 0.0;
  truth.x[tracking::VelocityZ] = 0.0;
  truth.x[tracking::Omega] = 0.0;
  const std::uint64_t t0 = 2'700'000'000ULL;
  initializeTracker(ekf, truth, t0);

  tracking::Measurement measurement =
      observationAt(truth, 0, t0 + 20'000'000ULL);
  measurement.position_T_m += Eigen::Vector3d(0.025, -0.015, 0.01);
  const tracking::TrackOutput output =
      ekf.update(measurement.timestamp_ns, {measurement});
  require(output.associated_observations.size() == 1,
          "single visible armor must still update motion state");
  requireNear(output.radius_even_m, options.radius_prior_m, 1e-12,
              "single armor must not change even radius");
  requireNear(output.radius_odd_delta_m, 0.0, 1e-12,
              "single armor must not change radius difference");
  requireNear(output.height_odd_delta_m, 0.0, 1e-12,
              "single armor must not change height difference");
}

void testLargeYawResidualBecomesPositionOnly() {
  tracking::WholeVehicleEkfOptions options;
  options.confirming_hits = 1;
  tracking::WholeVehicleEkf ekf(options);
  tracking::State truth = syntheticState();
  truth.x[tracking::RadiusEven] = options.radius_prior_m;
  truth.x[tracking::RadiusOddDelta] = 0.0;
  truth.x[tracking::HeightOddDelta] = 0.0;
  truth.x[tracking::VelocityX] = 0.0;
  truth.x[tracking::VelocityY] = 0.0;
  truth.x[tracking::VelocityZ] = 0.0;
  truth.x[tracking::Omega] = 0.0;
  const std::uint64_t t0 = 2'800'000'000ULL;
  initializeTracker(ekf, truth, t0);
  tracking::Measurement measurement =
      observationAt(truth, 0, t0 + 20'000'000ULL);
  measurement.inward_yaw_T_rad += 0.5;
  const tracking::TrackOutput output =
      ekf.update(measurement.timestamp_ns, {measurement});
  require(output.associated_observations.size() == 1,
          "moderate yaw error must retain the position observation");
  require(!output.associated_observations.front().includes_yaw,
          "large yaw innovation must not update theta or omega");
  requireNear(output.state.x[tracking::Theta], truth.x[tracking::Theta], 1e-9,
              "position-only zero residual must not pull theta");
}

void testSecondArmorWithPoorYawStillUpdatesGeometry() {
  tracking::WholeVehicleEkfOptions options;
  options.confirming_hits = 1;
  tracking::WholeVehicleEkf ekf(options);
  tracking::State truth = syntheticState();
  truth.x[tracking::RadiusEven] = 0.22;
  truth.x[tracking::RadiusOddDelta] = -0.04;
  truth.x[tracking::HeightOddDelta] = 0.03;
  truth.x[tracking::VelocityX] = 0.0;
  truth.x[tracking::VelocityY] = 0.0;
  truth.x[tracking::VelocityZ] = 0.0;
  truth.x[tracking::Omega] = 0.0;
  const std::uint64_t timestamp_ns = 2'850'000'000ULL;
  tracking::Measurement slot_zero =
      observationAt(truth, 0, timestamp_ns);
  tracking::Measurement slot_one =
      observationAt(truth, 1, timestamp_ns);
  slot_one.inward_yaw_T_rad -= 1.15;
  slot_one.has_exposure_camera_geometry = true;
  slot_one.camera_position_T_m = Eigen::Vector3d::Zero();
  const tracking::TrackOutput output =
      ekf.update(timestamp_ns, {slot_zero, slot_one});
  require(output.associated_observations.size() == 2,
          "poor second-board yaw must not discard its position");
  const auto second = std::find_if(
      output.associated_observations.begin(),
      output.associated_observations.end(),
      [](const tracking::AssociatedObservation& association) {
        return association.measurement_index == 1;
      });
  require(second != output.associated_observations.end() &&
              second->armor_slot == 1 && !second->includes_yaw,
          "poor yaw must select the adjacent slot as position-only");
  require(std::abs(output.radius_even_m - truth.x[tracking::RadiusEven]) <
              std::abs(options.radius_prior_m - truth.x[tracking::RadiusEven]),
          "one reliable yaw plus two positions must reduce radius-prior error");
  const Eigen::Vector3d expected_center(
      truth.x[tracking::CenterX], truth.x[tracking::CenterY],
      truth.x[tracking::CenterZ]);
  require((output.center_T_m - expected_center).norm() < 0.08,
          "mixed-yaw armor pair must correct the initialized vehicle center");
}

void testJointUpdateIndependentOfDetectionOrder() {
  tracking::WholeVehicleEkfOptions options;
  options.confirming_hits = 1;
  tracking::WholeVehicleEkf first_order(options);
  tracking::WholeVehicleEkf second_order(options);
  tracking::State truth = syntheticState();
  truth.x[tracking::RadiusEven] = options.radius_prior_m;
  truth.x[tracking::RadiusOddDelta] = 0.0;
  truth.x[tracking::HeightOddDelta] = 0.0;
  truth.x[tracking::VelocityX] = 0.0;
  truth.x[tracking::VelocityY] = 0.0;
  truth.x[tracking::VelocityZ] = 0.0;
  truth.x[tracking::Omega] = 0.0;
  const std::uint64_t t0 = 2'900'000'000ULL;
  initializeTracker(first_order, truth, t0);
  initializeTracker(second_order, truth, t0);
  tracking::Measurement slot_zero =
      observationAt(truth, 0, t0 + 20'000'000ULL);
  tracking::Measurement slot_three =
      observationAt(truth, 3, t0 + 20'000'000ULL);
  slot_zero.position_T_m += Eigen::Vector3d(0.01, -0.005, 0.002);
  slot_three.position_T_m += Eigen::Vector3d(-0.006, 0.008, -0.003);
  const tracking::TrackOutput output_ab = first_order.update(
      slot_zero.timestamp_ns, {slot_zero, slot_three});
  const tracking::TrackOutput output_ba = second_order.update(
      slot_zero.timestamp_ns, {slot_three, slot_zero});
  require(output_ab.associated_observations.size() == 2 &&
              output_ba.associated_observations.size() == 2,
          "both observation orders must retain both armors");
  require((output_ab.state.x - output_ba.state.x).norm() < 1e-10,
          "joint posterior must not depend on detector ordering");
  require((output_ab.state.covariance - output_ba.state.covariance)
              .cwiseAbs()
              .maxCoeff() < 1e-10,
          "joint covariance must not depend on detector ordering");
}

void testRotatingSlotSequenceKeepsCenterStable() {
  tracking::WholeVehicleEkfOptions options;
  options.confirming_hits = 1;
  tracking::WholeVehicleEkf ekf(options);
  tracking::State truth = syntheticState();
  truth.x[tracking::RadiusEven] = options.radius_prior_m;
  truth.x[tracking::RadiusOddDelta] = 0.0;
  truth.x[tracking::HeightOddDelta] = 0.0;
  truth.x[tracking::VelocityX] = 0.0;
  truth.x[tracking::VelocityY] = 0.0;
  truth.x[tracking::VelocityZ] = 0.0;
  truth.x[tracking::Omega] = 1.2;
  std::uint64_t timestamp_ns = 3'100'000'000ULL;
  initializeTracker(ekf, truth, timestamp_ns);
  const Eigen::Vector3d expected_center(
      truth.x[tracking::CenterX], truth.x[tracking::CenterY],
      truth.x[tracking::CenterZ]);
  const std::vector<int> visible_slots{3, 3, 2, 2, 1, 1, 0};
  for (int slot : visible_slots) {
    constexpr double kDt = 0.1;
    truth = tracking::predictState(truth, kDt);
    timestamp_ns += 100'000'000ULL;
    const tracking::Measurement measurement =
        observationAt(truth, slot, timestamp_ns);
    const tracking::TrackOutput output = ekf.update(timestamp_ns, {measurement});
    require(output.associated_slot.has_value() &&
                *output.associated_slot == slot,
            "rotating target must advance the physical armor slot");
    require((output.center_T_m - expected_center).norm() < 0.05,
            "slot transitions must not move a stationary vehicle center");
  }
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
    testTwoVisibleArmorsUpdateOneFrame();
    testSingleArmorDoesNotChangeGeometry();
    testLargeYawResidualBecomesPositionOnly();
    testSecondArmorWithPoorYawStillUpdatesGeometry();
    testJointUpdateIndependentOfDetectionOrder();
    testRotatingSlotSequenceKeepsCenterStable();
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
