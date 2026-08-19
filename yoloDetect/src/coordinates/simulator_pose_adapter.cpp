#include "coordinates/simulator_pose_adapter.hpp"

#include <daedalus_sim_sdk/endpoints_v1.hpp>
#include <daedalus_sim_sdk/talos_v1.hpp>

#include <cmath>

namespace yolo_detect::coordinates {
namespace {

namespace sdk = daedalus::sim::sdk::v1;

cv::Vec3d vec3(const float value[3]) {
  return {static_cast<double>(value[0]), static_cast<double>(value[1]),
          static_cast<double>(value[2])};
}

cv::Vec4d vec4(const float value[4]) {
  return {static_cast<double>(value[0]), static_cast<double>(value[1]),
          static_cast<double>(value[2]), static_cast<double>(value[3])};
}

bool finite(const cv::Vec3d& value) {
  return std::isfinite(value[0]) && std::isfinite(value[1]) &&
         std::isfinite(value[2]);
}

}  // namespace

bool SimulatorPoseAdapter::open(
    const std::filesystem::path& ipc_directory) {
  close();
  if (ipc_directory.empty()) {
    last_error_ = "IPC directory must not be empty";
    return false;
  }

  const std::filesystem::path metadata_path =
      ipc_directory / std::string(sdk::kMetaFileName);
  const sdk::ClientStatus opened = mapping_.open(metadata_path.string());
  if (!opened) {
    last_error_ = "metadata mapping failed: " + opened.message;
    return false;
  }

  const auto reader_result = mapping_.reader();
  if (!reader_result) {
    last_error_ = "metadata reader failed: " + reader_result.status.message;
    mapping_.close();
    return false;
  }
  const sdk::TalosMetadataReader& reader = *reader_result.value;
  const auto camera_pose = reader.readLatestPose(sdk::kCameraPoseIndex);
  if (!camera_pose) {
    last_error_ = "camera local pose unavailable: " +
                  camera_pose.status.message;
    mapping_.close();
    return false;
  }
  const auto muzzle_pose = reader.readLatestPose(sdk::kMuzzlePoseIndex);
  if (!muzzle_pose) {
    last_error_ = "muzzle local pose unavailable: " +
                  muzzle_pose.status.message;
    mapping_.close();
    return false;
  }

  camera_offset_gimbal_m_ = vec3(camera_pose.value->position);
  muzzle_offset_gimbal_m_ = vec3(muzzle_pose.value->position);
  if (!finite(camera_offset_gimbal_m_) ||
      !finite(muzzle_offset_gimbal_m_)) {
    last_error_ = "camera or muzzle local offset is non-finite";
    mapping_.close();
    return false;
  }

  offsets_valid_ = true;
  last_error_.clear();
  return true;
}

void SimulatorPoseAdapter::close() noexcept {
  mapping_.close();
  offsets_valid_ = false;
}

bool SimulatorPoseAdapter::isOpen() const noexcept {
  return mapping_.isOpen() && offsets_valid_;
}

CoordinateSnapshot SimulatorPoseAdapter::snapshotForFrame(
    std::uint64_t frame_sequence,
    double camera_position_tolerance_m) const {
  CoordinateSnapshot invalid;
  invalid.frame_sequence = frame_sequence;
  invalid.status = CoordinateStatus::MissingPoseState;
  if (!isOpen()) {
    last_error_ = "simulator pose adapter is not open";
    return invalid;
  }

  const auto reader_result = mapping_.reader();
  if (!reader_result) {
    last_error_ = "metadata reader failed: " + reader_result.status.message;
    return invalid;
  }
  const auto exposure =
      reader_result.value->readExposureStateForFrame(frame_sequence);
  if (!exposure) {
    last_error_ = "frame " + std::to_string(frame_sequence) +
                  " exposure state unavailable: " + exposure.status.message;
    return invalid;
  }

  const sdk::ExposureState& state = *exposure.value;
  CoordinateObservation observation;
  observation.frame_sequence = state.frame_seq;
  observation.ros_odom_world =
      state.world_frame == sdk::kGroundTruthFrameRosOdom;
  observation.has_chassis_pose =
      (state.state_flags & sdk::kExposureStateHasChassisWorldPose) != 0;
  observation.has_gimbal_pose =
      (state.state_flags & sdk::kExposureStateHasGimbalWorldPose) != 0;
  observation.has_camera_pose =
      (state.state_flags & sdk::kExposureStateHasCameraWorldPose) != 0;
  observation.chassis_position_odom_m = vec3(state.chassis_position_world);
  observation.chassis_quaternion_odom_wxyz =
      vec4(state.chassis_quaternion_world_wxyz);
  observation.gimbal_position_odom_m = vec3(state.gimbal_position_world);
  observation.measured_camera_position_odom_m =
      vec3(state.camera_position_world);
  observation.gimbal_yaw_rad = state.gimbal_yaw_rad;
  observation.gimbal_elevation_rad = state.gimbal_pitch_rad;
  observation.camera_offset_gimbal_m = camera_offset_gimbal_m_;
  observation.muzzle_offset_gimbal_m = muzzle_offset_gimbal_m_;

  CoordinateSnapshot snapshot =
      makeCoordinateSnapshot(observation, camera_position_tolerance_m);
  last_error_ = coordinateStatusName(snapshot.status);
  return snapshot;
}

const cv::Vec3d& SimulatorPoseAdapter::cameraOffsetGimbalM() const noexcept {
  return camera_offset_gimbal_m_;
}

const cv::Vec3d& SimulatorPoseAdapter::muzzleOffsetGimbalM() const noexcept {
  return muzzle_offset_gimbal_m_;
}

const std::string& SimulatorPoseAdapter::lastError() const noexcept {
  return last_error_;
}

}  // namespace yolo_detect::coordinates
