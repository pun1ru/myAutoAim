#include "sdk/daedalus_frame_source.hpp"

#include <cstdlib>
#include <memory>
#include <utility>

namespace daedalus_sdk = daedalus::sim::sdk::v1;

namespace my_project::sdk {

namespace {

std::filesystem::path metadataPath(const std::filesystem::path& ipc_directory) {
  return ipc_directory / std::string(daedalus_sdk::kMetaFileName);
}

}  // namespace

DaedalusFrameSource::DaedalusFrameSource(FrameSourceConfig config)
    : config_(std::move(config)), images_({config_.host, config_.port}) {}

daedalus_sdk::ClientStatus DaedalusFrameSource::connect() {
  if (config_.ipc_directory.empty()) {
    return daedalus_sdk::ClientStatus::failure(
        daedalus_sdk::ClientError::InvalidArgument,
        "IPC directory is empty");
  }

  auto opened = metadata_.open(metadataPath(config_.ipc_directory).string());
  if (!opened) return opened;

  auto made_reader = metadata_.reader();
  if (!made_reader) {
    metadata_.close();
    return made_reader.status;
  }
  reader_ = *made_reader.value;

  auto connected = images_.connect();
  if (!connected) {
    reader_.reset();
    metadata_.close();
    return connected;
  }

  previous_sequence_ = 0;
  producer_epoch_ = 0;
  return daedalus_sdk::ClientStatus::success();
}

daedalus_sdk::ClientResult<ReceivedFrame> DaedalusFrameSource::receive() {
  auto frame = images_.waitForLatest(previous_sequence_, config_.frame_timeout);
  if (!frame) {
    return daedalus_sdk::ClientResult<ReceivedFrame>::failure(
        frame.status.error, frame.status.message);
  }

  if (producer_epoch_ != 0 &&
      producer_epoch_ != frame.value->header.producer_epoch) {
    previous_sequence_ = 0;
  }
  producer_epoch_ = frame.value->header.producer_epoch;
  previous_sequence_ = frame.value->header.source_sequence;

  ReceivedFrame result;
  result.image = std::move(*frame.value);
  if (reader_.has_value()) {
    auto synced = reader_->readGimbalStateForFrame(
        result.image.header.source_sequence);
    if (synced) {
      result.exposure_gimbal = *synced.value;
    } else {
      result.synchronization_message = synced.status.message;
    }
  }

  return daedalus_sdk::ClientResult<ReceivedFrame>::success(std::move(result));
}

void DaedalusFrameSource::close() noexcept {
  images_.close();
  reader_.reset();
  metadata_.close();
  previous_sequence_ = 0;
  producer_epoch_ = 0;
}

bool DaedalusFrameSource::connected() const noexcept {
  return images_.connected();
}

const FrameSourceConfig& DaedalusFrameSource::config() const noexcept {
  return config_;
}

std::filesystem::path defaultIpcDirectory() {
#ifdef _WIN32
  char* raw_value = nullptr;
  std::size_t value_size = 0;
  if (_dupenv_s(&raw_value, &value_size, "TALOS_IPC_DIR") == 0) {
    const std::unique_ptr<char, decltype(&std::free)> value(raw_value, &std::free);
    if (value && value_size > 1) return std::filesystem::path(value.get());
  }
#else
  if (const char* value = std::getenv("TALOS_IPC_DIR");
      value != nullptr && value[0] != '\0') {
    return std::filesystem::path(value);
  }
#endif

  const auto sibling_release = std::filesystem::current_path() / ".." / "1.1.1" /
                               "runtime" / "talos-ipc";
  return sibling_release.lexically_normal();
}

}  // namespace my_project::sdk
