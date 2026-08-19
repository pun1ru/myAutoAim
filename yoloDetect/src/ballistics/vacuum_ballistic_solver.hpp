#pragma once

#include <limits>

namespace yolo_detect::ballistics {

enum class TrajectoryArc {
  Low,
  High,
};

enum class BallisticStatus {
  Success,
  NonFiniteInput,
  NonPositiveHorizontalDistance,
  Unreachable,
  FlightTimeExceeded,
  NumericalFailure,
};

// Daedalus 1.3.1 projectile defaults plus an explicit solver gravity.
// Diameter and mass do not enter the vacuum trajectory equations.
struct ProjectileParameters {
  double muzzle_speed_mps = 25.0;
  double lifetime_s = 5.0;
  double cooldown_s = 0.05;
  double diameter_m = 0.017;
  double mass_kg = 0.0032;
  double linear_damping_per_s = 0.0;
  double gravity_mps2 = 9.81;
};

// Scalar geometry in the muzzle frame. Horizontal distance is measured in the
// level plane; vertical offset is positive upward from the muzzle.
struct BallisticTarget {
  double horizontal_distance_m = 0.0;
  double vertical_offset_m = 0.0;
};

struct BallisticSolution {
  bool valid = false;
  BallisticStatus status = BallisticStatus::NumericalFailure;
  TrajectoryArc arc = TrajectoryArc::Low;
  BallisticTarget target;
  double pitch_rad = std::numeric_limits<double>::quiet_NaN();
  double pitch_deg = std::numeric_limits<double>::quiet_NaN();
  double time_of_flight_s = std::numeric_limits<double>::quiet_NaN();
  double launch_horizontal_velocity_mps =
      std::numeric_limits<double>::quiet_NaN();
  double launch_vertical_velocity_mps =
      std::numeric_limits<double>::quiet_NaN();
  double gravity_drop_m = std::numeric_limits<double>::quiet_NaN();
};

[[nodiscard]] const char* trajectoryArcName(TrajectoryArc arc) noexcept;
[[nodiscard]] const char* ballisticStatusName(BallisticStatus status) noexcept;

class VacuumBallisticSolver {
 public:
  explicit VacuumBallisticSolver(
      ProjectileParameters parameters = ProjectileParameters{});

  // Solves a stationary target using constant gravity and no aerodynamic drag.
  // pitch_rad is zero at level aim and positive upward. It is not a simulator
  // pitch command; coordinate transforms and the simulator's 90-degree offset
  // belong to the future control layer.
  [[nodiscard]] BallisticSolution solve(
      const BallisticTarget& target,
      TrajectoryArc arc = TrajectoryArc::Low) const noexcept;

  [[nodiscard]] const ProjectileParameters& parameters() const noexcept;
  [[nodiscard]] double maximumFireRateHz() const noexcept;

 private:
  ProjectileParameters parameters_;
};

}  // namespace yolo_detect::ballistics
