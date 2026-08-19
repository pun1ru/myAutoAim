#include "ballistics/vacuum_ballistic_solver.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace ballistics = yolo_detect::ballistics;

namespace {

int failures = 0;

// 在条件失败时记录测试断言。
void expect(bool condition, const char* message) {
  if (condition) return;
  std::cerr << "FAIL: " << message << '\n';
  ++failures;
}

// 以给定绝对误差比较两个浮点数。
bool nearlyEqual(double actual, double expected, double tolerance) {
  return std::abs(actual - expected) <= tolerance;
}

// 用弹道解正向计算命中时的竖直高度。
double forwardHeight(const ballistics::BallisticSolution& solution,
                     double gravity_mps2) {
  const double time = solution.time_of_flight_s;
  return solution.launch_vertical_velocity_mps * time -
         0.5 * gravity_mps2 * time * time;
}

// 验证求解器默认参数与模拟器弹丸配置一致。
void testSimulatorDefaults() {
  const ballistics::VacuumBallisticSolver solver;
  const auto& parameters = solver.parameters();
  expect(parameters.muzzle_speed_mps == 25.0,
         "default muzzle speed must match Daedalus 1.3.1");
  expect(parameters.lifetime_s == 5.0,
         "default lifetime must match Daedalus 1.3.1");
  expect(parameters.cooldown_s == 0.05,
         "default cooldown must match Daedalus 1.3.1");
  expect(parameters.diameter_m == 0.017,
         "default projectile diameter must be 17 mm");
  expect(parameters.mass_kg == 0.0032,
         "default projectile mass must be 3.2 g");
  expect(parameters.linear_damping_per_s == 0.0,
         "vacuum model must default to zero linear damping");
  expect(parameters.gravity_mps2 == 9.81,
         "default solver gravity must be explicit and testable");
  expect(nearlyEqual(solver.maximumFireRateHz(), 20.0, 1e-12),
         "0.05 s cooldown must correspond to 20 Hz");
}

// 验证水平目标低弹道的解析解和正向积分结果。
void testLevelLowArc() {
  const ballistics::VacuumBallisticSolver solver;
  const ballistics::BallisticTarget target{10.0, 0.0};
  const auto solution = solver.solve(target, ballistics::TrajectoryArc::Low);
  expect(solution.valid, "10 m level target must have a low-arc solution");
  expect(solution.status == ballistics::BallisticStatus::Success,
         "valid solution must report Success");
  const double expected_pitch =
      0.5 * std::asin(solver.parameters().gravity_mps2 *
                      target.horizontal_distance_m /
                      (solver.parameters().muzzle_speed_mps *
                       solver.parameters().muzzle_speed_mps));
  expect(nearlyEqual(solution.pitch_rad, expected_pitch, 1e-12),
         "level-target low arc must match the independent closed form");
  expect(nearlyEqual(solution.launch_horizontal_velocity_mps *
                         solution.time_of_flight_s,
                     target.horizontal_distance_m, 1e-10),
         "forward integration must recover horizontal distance");
  expect(nearlyEqual(forwardHeight(solution, solver.parameters().gravity_mps2),
                     target.vertical_offset_m, 1e-10),
         "forward integration must recover level target height");
  expect(nearlyEqual(solution.gravity_drop_m,
                     solution.launch_vertical_velocity_mps *
                         solution.time_of_flight_s,
                     1e-10),
         "level-target launch rise must equal gravity drop");
}

// 验证高于炮口的目标可由运动方程反算命中。
void testElevatedTargetRoundTrip() {
  const ballistics::VacuumBallisticSolver solver;
  const ballistics::BallisticTarget target{8.0, 1.25};
  const auto solution = solver.solve(target);
  expect(solution.valid, "elevated target must have a low-arc solution");
  expect(solution.pitch_rad > std::atan2(target.vertical_offset_m,
                                         target.horizontal_distance_m),
         "gravity compensation must aim above line of sight");
  expect(nearlyEqual(forwardHeight(solution, solver.parameters().gravity_mps2),
                     target.vertical_offset_m, 1e-10),
         "elevated target must round-trip through the motion equation");
}

// 验证不可达目标和超寿命高弹道会被拒绝。
void testUnreachableAndLifetimeLimits() {
  const ballistics::VacuumBallisticSolver solver;
  const auto unreachable = solver.solve({100.0, 0.0});
  expect(!unreachable.valid &&
             unreachable.status == ballistics::BallisticStatus::Unreachable,
         "target beyond vacuum range must be unreachable");

  const auto high = solver.solve({10.0, 0.0}, ballistics::TrajectoryArc::High);
  expect(!high.valid &&
             high.status ==
                 ballistics::BallisticStatus::FlightTimeExceeded,
         "high arc exceeding the 5 s lifetime must be rejected");
}

// 验证非法输入及不支持的弹丸参数会被明确拒绝。
void testInputAndParameterValidation() {
  const ballistics::VacuumBallisticSolver solver;
  const auto non_finite = solver.solve(
      {std::numeric_limits<double>::quiet_NaN(), 0.0});
  expect(non_finite.status == ballistics::BallisticStatus::NonFiniteInput,
         "non-finite target input must be explicit");
  const auto behind = solver.solve({-1.0, 0.0});
  expect(behind.status ==
             ballistics::BallisticStatus::NonPositiveHorizontalDistance,
         "non-positive horizontal distance must be rejected");

  auto parameters = solver.parameters();
  parameters.linear_damping_per_s = 0.1;
  bool damping_rejected = false;
  try {
    const ballistics::VacuumBallisticSolver invalid(parameters);
    (void)invalid;
  } catch (const std::invalid_argument&) {
    damping_rejected = true;
  }
  expect(damping_rejected,
         "vacuum solver must reject non-zero linear damping");

  parameters = solver.parameters();
  parameters.muzzle_speed_mps = 0.0;
  bool speed_rejected = false;
  try {
    const ballistics::VacuumBallisticSolver invalid(parameters);
    (void)invalid;
  } catch (const std::invalid_argument&) {
    speed_rejected = true;
  }
  expect(speed_rejected, "non-positive muzzle speed must be rejected");
}

}  // namespace

// 运行全部真空弹道单元测试。
int main() {
  testSimulatorDefaults();
  testLevelLowArc();
  testElevatedTargetRoundTrip();
  testUnreachableAndLifetimeLimits();
  testInputAndParameterValidation();
  if (failures != 0) {
    std::cerr << failures << " ballistic test assertion(s) failed\n";
    return 1;
  }
  std::cout << "Vacuum ballistic solver and validation tests passed\n";
  return 0;
}
