#pragma once

#include "coordinates/coordinate_frames.hpp"
#include "detection/yolo_pose_detector.hpp"
#include "pose/armor_pose_estimator.hpp"
#include "tracking/constrained_yaw_solver.hpp"
#include "tracking/whole_vehicle_ekf.hpp"

#include <cstdint>
#include <optional>

namespace yolo_detect::tracking {

struct TrackerFrame {
  std::uint64_t source_sequence = 0;
  std::uint64_t capture_timestamp_ns = 0;
};

// Builds a tracker-frame position observation from metric PnP and the
// exposure state matching exactly the image frame. Yaw is unavailable until a
// separate constrained-reprojection estimator is introduced.
[[nodiscard]] std::optional<Measurement> makeTrackerMeasurement(
    const TrackerFrame& frame,
    const coordinates::CoordinateSnapshot& exposure_snapshot,
    const PoseResult& pose, const ArmorDetection& detection,
    const ConstrainedYawSolver& yaw_solver,
    ReliableYaw* reliable_yaw = nullptr);

}  // namespace yolo_detect::tracking
