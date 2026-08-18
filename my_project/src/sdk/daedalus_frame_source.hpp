#pragma once

#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <daedalus_sim_sdk/tcp_image_client.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace my_project::sdk {

struct FrameSourceConfig {
  std::string host = "127.0.0.1";
  std::uint16_t port = daedalus::sim::sdk::v1::kTcpImagePort;
  std::filesystem::path ipc_directory;
  std::chrono::milliseconds frame_timeout{1500};
};

struct ReceivedFrame {
  daedalus::sim::sdk::v1::TcpImageFrame image;
  std::optional<daedalus::sim::sdk::v1::GimbalState> exposure_gimbal;
  std::string synchronization_message;
};

class DaedalusFrameSource {
 public:
  explicit DaedalusFrameSource(FrameSourceConfig config);

  [[nodiscard]] daedalus::sim::sdk::v1::ClientStatus connect();
  [[nodiscard]] daedalus::sim::sdk::v1::ClientResult<ReceivedFrame> receive();
  void close() noexcept;

  [[nodiscard]] bool connected() const noexcept;
  [[nodiscard]] const FrameSourceConfig& config() const noexcept;

 private:
  FrameSourceConfig config_;
  daedalus::sim::sdk::v1::TcpImageClient images_;
  daedalus::sim::sdk::v1::TalosMetadataMapping metadata_;
  std::optional<daedalus::sim::sdk::v1::TalosMetadataReader> reader_;
  std::uint64_t previous_sequence_ = 0;
  std::uint64_t producer_epoch_ = 0;
};

[[nodiscard]] std::filesystem::path defaultIpcDirectory();

}  // namespace my_project::sdk
