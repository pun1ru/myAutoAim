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

  [[nodiscard]] bool open(const std::filesystem::path& ipc_directory);
  void close() noexcept;
  [[nodiscard]] bool isOpen() const noexcept;

  [[nodiscard]] CoordinateSnapshot snapshotForFrame(
      std::uint64_t frame_sequence,
      double camera_position_tolerance_m =
          kDefaultCameraPositionToleranceM) const;

  [[nodiscard]] const cv::Vec3d& cameraOffsetGimbalM() const noexcept;
  [[nodiscard]] const cv::Vec3d& muzzleOffsetGimbalM() const noexcept;
  [[nodiscard]] const std::string& lastError() const noexcept;

 private:
  daedalus::sim::sdk::v1::TalosMetadataMapping mapping_;
  cv::Vec3d camera_offset_gimbal_m_{0.0, 0.0, 0.0};
  cv::Vec3d muzzle_offset_gimbal_m_{0.0, 0.0, 0.0};
  bool offsets_valid_ = false;
  mutable std::string last_error_;
};

}  // namespace yolo_detect::coordinates
