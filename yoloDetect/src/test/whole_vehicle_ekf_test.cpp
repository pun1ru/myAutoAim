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
  tracking::State state = syntheticState();
  state.x[tracking::Theta] = 0.3;
  state.x[tracking::Omega] = 200.0;
  const tracking::State predicted = tracking::predictState(state, 10.0);
  require(std::abs(predicted.x[tracking::Theta]) <= kPi,
          "predicted yaw must be wrapped before later angular calculations");
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

void testSpVisionInitialCovariance() {
  tracking::WholeVehicleEkf ekf;
  const tracking::State truth = syntheticState();
  const std::uint64_t timestamp_ns = 900'000'000ULL;
  const tracking::TrackOutput output =
      ekf.update(timestamp_ns, {observationAt(truth, 0, timestamp_ns)});
  require(output.has_state, "reference P0 test must initialize the EKF");
  const std::array<double, tracking::kWholeVehicleStateDimension> expected = {
      1.0, 64.0, 1.0, 64.0, 1.0, 64.0, 0.4, 100.0, 1.0, 1.0, 1.0};
  for (int index = 0; index < tracking::kWholeVehicleStateDimension; ++index) {
    requireNear(output.state.covariance(index, index),
                expected[static_cast<std::size_t>(index)], 1e-12,
                "sp_vision P0 diagonal mismatch");
  }
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

void testMinimumCostAssociationDoesNotReject() {
  tracking::WholeVehicleEkf ekf;
  const tracking::State state = syntheticState();
  const std::uint64_t t0 = 2'000'000'000ULL;
  initializeTracker(ekf, state, t0);
  tracking::Measurement outlier = observationAt(state, 0, t0 + 10'000'000ULL);
  outlier.position_T_m += Eigen::Vector3d(2.0, -1.5, 0.2);
  const tracking::TrackOutput output = ekf.update(outlier.timestamp_ns, {outlier});
  require(output.associated_slot.has_value() && output.nis.has_value(),
          "slot association must select the minimum-cost slot without a gate");
}

void testTwoVisibleArmorsAssociateButOnlyPrimaryUpdates() {
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
  initializeTracker(ekf, truth, timestamp_ns);
  tracking::Measurement first = observationAt(truth, 0, timestamp_ns + 20'000'000ULL);
  tracking::Measurement second = observationAt(truth, 1, first.timestamp_ns);
  second.color_id = 2;  // Simulate an inconsistent color classification.
  first.selected_for_ekf = true;
  second.selected_for_ekf = false;
  const tracking::TrackOutput output = ekf.update(
      first.timestamp_ns, {first, second});
  require(output.associated_observations.size() == 2,
          "both visible armors must remain available for slot association");
  const auto secondary = std::find_if(
      output.associated_observations.begin(), output.associated_observations.end(),
      [](const tracking::AssociatedObservation& observation) {
        return observation.measurement_index == 1;
      });
  require(secondary != output.associated_observations.end() &&
              secondary->armor_slot == 1,
          "one-to-one association must keep the two physical armor slots distinct");
  requireNear(output.radius_even_m, options.radius_prior_m, 1e-12,
              "secondary armor must not update EKF geometry");
  requireNear(output.radius_odd_delta_m, 0.0, 1e-12,
              "fixed odd-radius delta must remain zero");
  requireNear(output.height_odd_delta_m, 0.0, 1e-12,
              "fixed odd-height delta must remain zero");
}

void testReliableSingleArmorHasLowWeightAndCannotMoveGeometry() {
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
  const tracking::State prior = ekf.state();

  tracking::Measurement measurement =
      observationAt(truth, 0, t0 + 20'000'000ULL);
  measurement.position_T_m += Eigen::Vector3d(0.025, -0.015, 0.01);
  const tracking::TrackOutput output =
      ekf.update(measurement.timestamp_ns, {measurement});
  require(output.associated_observations.size() == 1 &&
              output.associated_observations.front().includes_yaw,
          "reliable single armor must remain a four-dimensional update");
  requireNear(output.radius_even_m, options.radius_prior_m, 1e-12,
              "single armor must not change even radius");
  requireNear(output.radius_odd_delta_m, 0.0, 1e-12,
              "single armor must not change radius difference");
  requireNear(output.height_odd_delta_m, 0.0, 1e-12,
              "single armor must not change height difference");
  const double motion_correction =
      (output.state.x.template head<6>() - prior.x.template head<6>()).norm();
  require(std::isfinite(motion_correction) && motion_correction > 1e-6 &&
              motion_correction < 0.10,
          "sp_vision P0 must keep a single-armor motion update finite and bounded");
}

void testGeometryUpdatesOnFirstConsistentMultiArmorFrame() {
  tracking::WholeVehicleEkfOptions options;
  options.radius_prior_m = 0.26;
  options.confirming_hits = 1;
  options.geometry_confirming_frames = 1;
  tracking::WholeVehicleEkf ekf(options);
  tracking::State truth = syntheticState();
  truth.x[tracking::RadiusEven] = 0.32;
  truth.x[tracking::RadiusOddDelta] = 0.0;
  truth.x[tracking::HeightOddDelta] = 0.0;
  truth.x[tracking::VelocityX] = 0.0;
  truth.x[tracking::VelocityY] = 0.0;
  truth.x[tracking::VelocityZ] = 0.0;
  truth.x[tracking::Omega] = 0.0;
  std::uint64_t timestamp_ns = 2'600'000'000ULL;

  timestamp_ns += 20'000'000ULL;
  const tracking::TrackOutput output = ekf.update(
      timestamp_ns, {observationAt(truth, 0, timestamp_ns),
                     observationAt(truth, 1, timestamp_ns)});
  require(std::abs(output.radius_even_m - truth.x[tracking::RadiusEven]) <
              std::abs(options.radius_prior_m - truth.x[tracking::RadiusEven]),
          "the first consistent multi-armor frame must update radius");
  requireNear(output.radius_odd_delta_m, 0.0, 1e-12,
              "fixed odd-radius delta must remain zero");
  requireNear(output.height_odd_delta_m, 0.0, 1e-12,
              "fixed odd-height delta must remain zero");
}

void testTemporallyInconsistentYawDoesNotUpdateAngularState() {
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
          "temporally inconsistent yaw may still associate by position");
  require(!output.associated_observations.front().includes_yaw,
          "temporally inconsistent yaw must not enter the EKF update");
  requireNear(output.state.x[tracking::Theta], truth.x[tracking::Theta], 1e-12,
              "rejected yaw must not correct theta");
}

void testAngularVelocityCorrectionIsBounded() {
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
  const std::uint64_t t0 = 2'825'000'000ULL;
  initializeTracker(ekf, truth, t0);

  tracking::Measurement noisy_yaw =
      observationAt(truth, 0, t0 + 20'000'000ULL);
  noisy_yaw.inward_yaw_T_rad += 0.34;
  const tracking::TrackOutput output =
      ekf.update(noisy_yaw.timestamp_ns, {noisy_yaw});
  require(std::abs(output.omega_rad_s) <=
              options.maximum_omega_correction_rad_s + 1e-12,
          "one yaw observation must have a bounded omega correction");
}

void testPositionOnlySecondArmorConstrainsChassis() {
  tracking::WholeVehicleEkfOptions options;
  options.confirming_hits = 1;
  tracking::WholeVehicleEkf ekf(options);
  tracking::State truth = syntheticState();
  truth.x[tracking::RadiusEven] = 0.32;
  truth.x[tracking::RadiusOddDelta] = 0.04;
  truth.x[tracking::HeightOddDelta] = 0.0;
  truth.x[tracking::VelocityX] = 0.0;
  truth.x[tracking::VelocityY] = 0.0;
  truth.x[tracking::VelocityZ] = 0.0;
  truth.x[tracking::Omega] = 0.0;
  const std::uint64_t timestamp_ns = 2'850'000'000ULL;
  tracking::Measurement slot_zero =
      observationAt(truth, 0, timestamp_ns);
  tracking::Measurement slot_one =
      observationAt(truth, 1, timestamp_ns);
  slot_one.has_inward_yaw = false;
  const Eigen::Vector3d expected_center(
      truth.x[tracking::CenterX], truth.x[tracking::CenterY],
      truth.x[tracking::CenterZ]);
  const tracking::TrackOutput output =
      ekf.update(timestamp_ns, {slot_zero, slot_one});
  require(output.associated_observations.size() == 2,
          "position-only second board must remain visible in association telemetry");
  const auto second = std::find_if(
      output.associated_observations.begin(),
      output.associated_observations.end(),
      [](const tracking::AssociatedObservation& association) {
        return association.measurement_index == 1;
      });
  require(second != output.associated_observations.end() &&
              !second->includes_yaw,
          "yaw-less second board must remain position-only");
  require(second->armor_slot == 1,
          "the second board must keep its nearest physical slot: " +
              std::to_string(second->armor_slot));
  require(std::abs(output.radius_even_m - truth.x[tracking::RadiusEven]) <
              std::abs(options.radius_prior_m - truth.x[tracking::RadiusEven]),
          "a reliable yaw plus position-only armor must update even radius: " +
              std::to_string(output.radius_even_m));
  requireNear(output.radius_odd_delta_m, 0.0, 1e-12,
              "fixed odd-radius delta must remain zero");
  requireNear(output.height_odd_delta_m, 0.0, 1e-12,
              "fixed odd-height delta must remain zero");
  const double center_error = (output.center_T_m - expected_center).norm();
  require(center_error < 0.06,
          "a noisy second board must not make the center jump: " +
              std::to_string(center_error));
}

void testTwoPositionOnlyArmorsConstrainChassis() {
  tracking::WholeVehicleEkfOptions options;
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
  const std::uint64_t t0 = 2'875'000'000ULL;
  initializeTracker(ekf, truth, t0);
  const tracking::State prior = ekf.state();
  tracking::Measurement first = observationAt(truth, 0, t0 + 20'000'000ULL);
  tracking::Measurement second = observationAt(truth, 1, first.timestamp_ns);
  first.has_inward_yaw = false;
  second.has_inward_yaw = false;
  first.position_T_m += Eigen::Vector3d(0.03, -0.02, 0.01);
  second.position_T_m += Eigen::Vector3d(-0.02, 0.03, -0.01);

  const tracking::TrackOutput output =
      ekf.update(first.timestamp_ns, {first, second});
  require(output.associated_observations.size() == 2,
          "two position-only detections must still be associated");
  require((output.state.x.template head<6>() - prior.x.template head<6>())
                  .norm() >
              1e-6,
          "two distinct armor positions must constrain center and velocity");
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

void testMultiArmorFramesRecoverLinearVelocity() {
  tracking::WholeVehicleEkfOptions options;
  options.confirming_hits = 1;
  tracking::WholeVehicleEkf ekf(options);
  tracking::State truth = syntheticState();
  truth.x[tracking::RadiusEven] = options.radius_prior_m;
  truth.x[tracking::RadiusOddDelta] = 0.0;
  truth.x[tracking::HeightOddDelta] = 0.0;
  truth.x[tracking::VelocityX] = 1.0;
  truth.x[tracking::VelocityY] = -0.35;
  truth.x[tracking::VelocityZ] = 0.0;
  truth.x[tracking::Omega] = 0.7;
  std::uint64_t timestamp_ns = 3'300'000'000ULL;
  initializeTracker(ekf, truth, timestamp_ns);

  tracking::TrackOutput output;
  for (int frame = 0; frame < 80; ++frame) {
    constexpr double kDt = 0.02;
    truth = tracking::predictState(truth, kDt);
    timestamp_ns += 20'000'000ULL;
    const int first_slot = (frame / 20) % tracking::kArmorSlotCount;
    const int second_slot = (first_slot + 1) % tracking::kArmorSlotCount;
    output = ekf.update(timestamp_ns,
                        {observationAt(truth, first_slot, timestamp_ns),
                         observationAt(truth, second_slot, timestamp_ns)});
  }
  requireNear(output.velocity_T_mps.x(), truth.x[tracking::VelocityX], 0.08,
              "multi-armor frames must recover vehicle x velocity");
  requireNear(output.velocity_T_mps.y(), truth.x[tracking::VelocityY], 0.08,
              "multi-armor frames must recover vehicle y velocity");
  require((output.center_T_m -
           Eigen::Vector3d(truth.x[tracking::CenterX],
                           truth.x[tracking::CenterY],
                           truth.x[tracking::CenterZ]))
              .norm() < 0.05,
          "multi-armor motion update must follow translating vehicle center");
}

void testSingleReliableArmorFramesDoNotInventLinearVelocity() {
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
  truth.x[tracking::Omega] = 0.7;
  std::uint64_t timestamp_ns = 3'700'000'000ULL;
  initializeTracker(ekf, truth, timestamp_ns);

  tracking::TrackOutput output;
  for (int frame = 0; frame < 160; ++frame) {
    constexpr double kDt = 0.02;
    truth = tracking::predictState(truth, kDt);
    timestamp_ns += 20'000'000ULL;
    const int visible_slot = (frame / 40) % tracking::kArmorSlotCount;
    tracking::Measurement measurement =
        observationAt(truth, visible_slot, timestamp_ns);
    measurement.position_T_m +=
        Eigen::Vector3d(0.03 * std::sin(frame * 0.31),
                        0.02 * std::cos(frame * 0.23),
                        0.01 * std::sin(frame * 0.17));
    output = ekf.update(timestamp_ns, {measurement});
  }
  require(output.velocity_T_mps.norm() < 0.10,
          "single-armor PnP jitter must not create large chassis velocity");
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
  const tracking::TrackOutput reset = ekf.update(t0 + 20'000'000ULL, {});
  require(!reset.has_state && reset.tracking_state == tracking::TrackingState::Lost,
          "losing armor observations must immediately reset E0 and the track");
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
    testSpVisionInitialCovariance();
    testSlotSwitchDoesNotJumpCenter();
    testMinimumCostAssociationDoesNotReject();
    testTwoVisibleArmorsAssociateButOnlyPrimaryUpdates();
    testReliableSingleArmorHasLowWeightAndCannotMoveGeometry();
    testRotatingSlotSequenceKeepsCenterStable();
    testSingleReliableArmorFramesDoNotInventLinearVelocity();
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
