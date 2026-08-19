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

// Daedalus 1.3.1 的弹丸默认参数，以及显式的求解器重力值。
// 直径和质量不参与真空弹道方程。
struct ProjectileParameters {
  double muzzle_speed_mps = 25.0;
  double lifetime_s = 5.0;
  double cooldown_s = 0.05;
  double diameter_m = 0.017;
  double mass_kg = 0.0032;
  double linear_damping_per_s = 0.0;
  double gravity_mps2 = 9.81;
};

// 炮口坐标系中的标量几何量。水平距离在水平面中测量，竖直偏移以炮口向上为正。
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

// 将弹道弧线枚举转换为稳定的诊断名称。
[[nodiscard]] const char* trajectoryArcName(TrajectoryArc arc) noexcept;
// 将弹道求解状态转换为可读的诊断信息。
[[nodiscard]] const char* ballisticStatusName(BallisticStatus status) noexcept;

class VacuumBallisticSolver {
 public:
  // 校验弹丸物理参数后创建求解器。
  explicit VacuumBallisticSolver(
      ProjectileParameters parameters = ProjectileParameters{});

  // 在恒定重力、无空气阻力条件下求解静止目标。
  // pitch_rad 在水平瞄准时为零，向上为正。它不是模拟器俯仰指令；坐标变换及
  // 模拟器的 90 度偏移由控制层处理。
  [[nodiscard]] BallisticSolution solve(
      const BallisticTarget& target,
      TrajectoryArc arc = TrajectoryArc::Low) const noexcept;

  // 返回此求解器使用的已校验弹丸参数。
  [[nodiscard]] const ProjectileParameters& parameters() const noexcept;
  // 计算由冷却时间限制的最大射速。
  [[nodiscard]] double maximumFireRateHz() const noexcept;

 private:
  ProjectileParameters parameters_;
};

}  // namespace yolo_detect::ballistics
