#include "ballistics/vacuum_ballistic_solver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace yolo_detect::ballistics {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDiscriminantTolerance = 1e-12;

bool isFinite(double value) { return std::isfinite(value); }

void validateParameters(const ProjectileParameters& parameters) {
  if (!isFinite(parameters.muzzle_speed_mps) ||
      !isFinite(parameters.lifetime_s) ||
      !isFinite(parameters.cooldown_s) ||
      !isFinite(parameters.diameter_m) || !isFinite(parameters.mass_kg) ||
      !isFinite(parameters.linear_damping_per_s) ||
      !isFinite(parameters.gravity_mps2)) {
    throw std::invalid_argument("projectile parameters must be finite");
  }
  if (parameters.muzzle_speed_mps <= 0.0) {
    throw std::invalid_argument("projectile muzzle speed must be positive");
  }
  if (parameters.lifetime_s <= 0.0) {
    throw std::invalid_argument("projectile lifetime must be positive");
  }
  if (parameters.cooldown_s <= 0.0) {
    throw std::invalid_argument("projectile cooldown must be positive");
  }
  if (parameters.diameter_m <= 0.0 || parameters.mass_kg <= 0.0) {
    throw std::invalid_argument("projectile diameter and mass must be positive");
  }
  if (parameters.linear_damping_per_s != 0.0) {
    throw std::invalid_argument(
        "vacuum ballistic solver requires zero linear damping");
  }
  if (parameters.gravity_mps2 <= 0.0) {
    throw std::invalid_argument("gravity magnitude must be positive");
  }
}

}  // namespace

const char* trajectoryArcName(TrajectoryArc arc) noexcept {
  switch (arc) {
    case TrajectoryArc::Low:
      return "low";
    case TrajectoryArc::High:
      return "high";
  }
  return "unknown trajectory arc";
}

const char* ballisticStatusName(BallisticStatus status) noexcept {
  switch (status) {
    case BallisticStatus::Success:
      return "success";
    case BallisticStatus::NonFiniteInput:
      return "ballistic target contains a non-finite value";
    case BallisticStatus::NonPositiveHorizontalDistance:
      return "horizontal target distance must be positive";
    case BallisticStatus::Unreachable:
      return "target is unreachable at the configured muzzle speed";
    case BallisticStatus::FlightTimeExceeded:
      return "trajectory exceeds the configured projectile lifetime";
    case BallisticStatus::NumericalFailure:
      return "ballistic calculation produced a non-finite value";
  }
  return "unknown ballistic status";
}

VacuumBallisticSolver::VacuumBallisticSolver(ProjectileParameters parameters)
    : parameters_(parameters) {
  validateParameters(parameters_);
}

BallisticSolution VacuumBallisticSolver::solve(
    const BallisticTarget& target, TrajectoryArc arc) const noexcept {
  BallisticSolution result;
  result.arc = arc;
  result.target = target;
  if (!isFinite(target.horizontal_distance_m) ||
      !isFinite(target.vertical_offset_m)) {
    result.status = BallisticStatus::NonFiniteInput;
    return result;
  }
  if (target.horizontal_distance_m <= 0.0) {
    result.status = BallisticStatus::NonPositiveHorizontalDistance;
    return result;
  }

  const double distance = target.horizontal_distance_m;
  const double height = target.vertical_offset_m;
  const double speed = parameters_.muzzle_speed_mps;
  const double gravity = parameters_.gravity_mps2;
  const double speed_squared = speed * speed;
  const double speed_fourth = speed_squared * speed_squared;
  const double gravity_term =
      gravity * (gravity * distance * distance +
                 2.0 * height * speed_squared);
  const double discriminant = speed_fourth - gravity_term;
  if (!isFinite(speed_fourth) || !isFinite(gravity_term) ||
      !isFinite(discriminant)) {
    result.status = BallisticStatus::NumericalFailure;
    return result;
  }
  const double discriminant_scale =
      std::max({1.0, std::abs(speed_fourth), std::abs(gravity_term)});
  if (discriminant < -kDiscriminantTolerance * discriminant_scale) {
    result.status = BallisticStatus::Unreachable;
    return result;
  }

  const double root = std::sqrt(std::max(0.0, discriminant));
  double tangent = 0.0;
  if (arc == TrajectoryArc::Low) {
    // Rationalized low-arc root avoids cancellation at short range.
    tangent =
        (gravity * distance * distance + 2.0 * height * speed_squared) /
        (distance * (speed_squared + root));
  } else {
    tangent = (speed_squared + root) / (gravity * distance);
  }
  const double pitch = std::atan(tangent);
  const double cosine = std::cos(pitch);
  const double horizontal_velocity = speed * cosine;
  if (!isFinite(tangent) || !isFinite(pitch) || !isFinite(horizontal_velocity) ||
      horizontal_velocity <= 0.0) {
    result.status = BallisticStatus::NumericalFailure;
    return result;
  }

  const double time_of_flight = distance / horizontal_velocity;
  const double vertical_velocity = speed * std::sin(pitch);
  const double gravity_drop =
      0.5 * gravity * time_of_flight * time_of_flight;
  if (!isFinite(time_of_flight) || !isFinite(vertical_velocity) ||
      !isFinite(gravity_drop)) {
    result.status = BallisticStatus::NumericalFailure;
    return result;
  }
  if (time_of_flight > parameters_.lifetime_s) {
    result.status = BallisticStatus::FlightTimeExceeded;
    return result;
  }

  result.valid = true;
  result.status = BallisticStatus::Success;
  result.pitch_rad = pitch;
  result.pitch_deg = pitch * 180.0 / kPi;
  result.time_of_flight_s = time_of_flight;
  result.launch_horizontal_velocity_mps = horizontal_velocity;
  result.launch_vertical_velocity_mps = vertical_velocity;
  result.gravity_drop_m = gravity_drop;
  return result;
}

const ProjectileParameters& VacuumBallisticSolver::parameters() const noexcept {
  return parameters_;
}

double VacuumBallisticSolver::maximumFireRateHz() const noexcept {
  return 1.0 / parameters_.cooldown_s;
}

}  // namespace yolo_detect::ballistics
