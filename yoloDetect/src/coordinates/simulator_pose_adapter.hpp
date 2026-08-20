#pragma once

#include "coordinates/coordinate_frames.hpp"

#include <daedalus_sim_sdk/talos_metadata_reader.hpp>

#include <cstdint>
#include <filesystem>
#include <string>

namespace yolo_detect::coordinates {

// Reads frame-synchronous simulator poses and converts the SDK representation
// into the coordinate module's explicit O/B/G/C contract.
class SimulatorPoseAdapter {
 public:
  SimulatorPoseAdapter() = default;

  // Opens the Talos metadata mapping and reads fixed camera/muzzle offsets.
  [[nodiscard]] bool open(const std::filesystem::path& ipc_directory);
  // Closes the metadata mapping and clears cached offset validity.
  void close() noexcept;
  // Reports whether the Talos metadata mapping is open.
  [[nodiscard]] bool isOpen() const noexcept;

  // Converts the metadata matching a source frame into a coordinate snapshot.
  [[nodiscard]] CoordinateSnapshot snapshotForFrame(
      std::uint64_t frame_sequence, std::uint64_t capture_timestamp_ns,
      double camera_position_tolerance_m =
          kDefaultCameraPositionToleranceM) const;

  // Returns the fixed camera offset expressed in gimbal coordinates.
  [[nodiscard]] const cv::Vec3d& cameraOffsetGimbalM() const noexcept;
  // Returns the fixed muzzle offset expressed in gimbal coordinates.
  [[nodiscard]] const cv::Vec3d& muzzleOffsetGimbalM() const noexcept;
  // Returns the most recent adapter error encountered by an operation.
  [[nodiscard]] const std::string& lastError() const noexcept;

 private:
  daedalus::sim::sdk::v1::TalosMetadataMapping mapping_;
  cv::Vec3d camera_offset_gimbal_m_{0.0, 0.0, 0.0};
  cv::Vec3d muzzle_offset_gimbal_m_{0.0, 0.0, 0.0};
  bool offsets_valid_ = false;
  mutable std::string last_error_;
};

}  // namespace yolo_detect::coordinates
