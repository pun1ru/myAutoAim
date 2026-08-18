#include "sdk/daedalus_frame_source.hpp"

#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <daedalus_sim_sdk/tcp_image_client.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace daedalus_sdk = daedalus::sim::sdk::v1;
using Clock = std::chrono::steady_clock;

namespace {

struct Options {
  std::string host = "127.0.0.1";
  std::uint16_t port = daedalus_sdk::kTcpImagePort;
  std::filesystem::path ipc_directory = my_project::sdk::defaultIpcDirectory();
  double warmup_seconds = 2.0;
  double sample_seconds = 10.0;
  bool synchronize = false;
};

void printUsage() {
  std::cout
      << "Daedalus headless receive benchmark (read-only)\n\n"
      << "Usage: daedalus_receive_benchmark [options]\n"
      << "  --host <address>    TCP host (default: 127.0.0.1)\n"
      << "  --port <port>       TCP port (default: 5602)\n"
      << "  --ipc-dir <path>    runtime/talos-ipc directory\n"
      << "  --warmup <seconds>  Warmup duration (default: 2)\n"
      << "  --seconds <seconds> Sample duration (default: 10)\n"
      << "  --sync              Query exposure gimbal state for every frame\n"
      << "  --help              Show this help\n\n"
      << "No OpenCV window is created and no UDP command is sent.\n";
}

std::string requireValue(int& index, int argc, char** argv) {
  if (index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + argv[index]);
  }
  return argv[++index];
}

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      printUsage();
      std::exit(0);
    }
    if (argument == "--host") {
      options.host = requireValue(index, argc, argv);
    } else if (argument == "--port") {
      const int value = std::stoi(requireValue(index, argc, argv));
      if (value < 1 || value > 65535) throw std::runtime_error("invalid port");
      options.port = static_cast<std::uint16_t>(value);
    } else if (argument == "--ipc-dir") {
      options.ipc_directory = requireValue(index, argc, argv);
    } else if (argument == "--warmup") {
      options.warmup_seconds = std::stod(requireValue(index, argc, argv));
    } else if (argument == "--seconds") {
      options.sample_seconds = std::stod(requireValue(index, argc, argv));
    } else if (argument == "--sync") {
      options.synchronize = true;
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (options.warmup_seconds < 0.0 || options.warmup_seconds > 60.0) {
    throw std::runtime_error("warmup must be between 0 and 60 seconds");
  }
  if (options.sample_seconds <= 0.0 || options.sample_seconds > 300.0) {
    throw std::runtime_error("sample duration must be between 0 and 300 seconds");
  }
  return options;
}

struct Counters {
  std::uint64_t frames = 0;
  std::uint64_t payload_bytes = 0;
  std::uint64_t first_sequence = 0;
  std::uint64_t last_sequence = 0;
  std::uint64_t first_capture_ns = 0;
  std::uint64_t last_capture_ns = 0;
  std::uint64_t sync_ok = 0;
  std::uint64_t sync_failed = 0;
  std::uint64_t timeouts = 0;
};

void recordFrame(Counters& counters, const daedalus_sdk::TcpImageFrame& frame) {
  if (counters.frames == 0) {
    counters.first_sequence = frame.header.source_sequence;
    counters.first_capture_ns = frame.header.capture_timestamp_ns;
  }
  ++counters.frames;
  counters.payload_bytes += frame.payload.size();
  counters.last_sequence = frame.header.source_sequence;
  counters.last_capture_ns = frame.header.capture_timestamp_ns;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  try {
    options = parseOptions(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "argument error: " << error.what() << '\n';
    printUsage();
    return 2;
  }

  daedalus_sdk::TalosMetadataMapping metadata;
  std::optional<daedalus_sdk::TalosMetadataReader> reader;
  if (options.synchronize) {
    const auto metadata_path = options.ipc_directory / "talos_ipc_meta";
    auto opened = metadata.open(metadata_path.string());
    if (!opened) {
      std::cerr << "metadata open failed: " << opened.message << '\n';
      return 3;
    }
    auto made_reader = metadata.reader();
    if (!made_reader) {
      std::cerr << "metadata reader failed: " << made_reader.status.message << '\n';
      return 3;
    }
    reader = *made_reader.value;
  }

  daedalus_sdk::TcpImageClient images({options.host, options.port});
  auto connected = images.connect();
  if (!connected) {
    std::cerr << "TCP connect failed: " << connected.message << '\n';
    return 4;
  }

  std::uint64_t previous_sequence = 0;
  const auto warmup_end =
      Clock::now() + std::chrono::duration<double>(options.warmup_seconds);
  while (Clock::now() < warmup_end) {
    auto frame = images.waitForLatest(previous_sequence,
                                      std::chrono::milliseconds(1500));
    if (!frame) {
      std::cerr << "warmup receive failed: " << frame.status.message << '\n';
      return 5;
    }
    previous_sequence = frame.value->header.source_sequence;
    if (reader) {
      (void)reader->readGimbalStateForFrame(previous_sequence);
    }
  }

  Counters counters;
  const auto sample_start = Clock::now();
  const auto sample_end =
      sample_start + std::chrono::duration<double>(options.sample_seconds);
  while (Clock::now() < sample_end) {
    auto frame = images.waitForLatest(previous_sequence,
                                      std::chrono::milliseconds(1500));
    if (!frame) {
      if (frame.status.error == daedalus_sdk::ClientError::Timeout) {
        ++counters.timeouts;
        continue;
      }
      std::cerr << "receive failed: " << frame.status.message << '\n';
      return 6;
    }

    previous_sequence = frame.value->header.source_sequence;
    recordFrame(counters, *frame.value);
    if (reader) {
      auto synced = reader->readGimbalStateForFrame(previous_sequence);
      if (synced) {
        ++counters.sync_ok;
      } else {
        ++counters.sync_failed;
      }
    }
  }
  const double elapsed =
      std::chrono::duration<double>(Clock::now() - sample_start).count();
  images.close();

  const std::uint64_t sequence_delta =
      counters.frames > 1 ? counters.last_sequence - counters.first_sequence : 0;
  const std::uint64_t skipped =
      sequence_delta + (counters.frames > 0 ? 1 : 0) > counters.frames
          ? sequence_delta + 1 - counters.frames
          : 0;
  const double received_hz = counters.frames / elapsed;
  const double source_hz = sequence_delta / elapsed;
  const double payload_mib_s =
      counters.payload_bytes / elapsed / (1024.0 * 1024.0);
  const double capture_span =
      counters.last_capture_ns > counters.first_capture_ns
          ? static_cast<double>(counters.last_capture_ns -
                                counters.first_capture_ns) /
                1'000'000'000.0
          : 0.0;
  const double capture_sequence_hz =
      capture_span > 0.0 ? sequence_delta / capture_span : 0.0;

  std::cout << std::fixed << std::setprecision(3)
            << "mode=" << (options.synchronize ? "tcp+sync" : "tcp-only")
            << '\n'
            << "elapsed_s=" << elapsed << '\n'
            << "frames_received=" << counters.frames << '\n'
            << "receive_hz=" << received_hz << '\n'
            << "first_sequence=" << counters.first_sequence << '\n'
            << "last_sequence=" << counters.last_sequence << '\n'
            << "source_sequence_hz=" << source_hz << '\n'
            << "capture_sequence_hz=" << capture_sequence_hz << '\n'
            << "frames_skipped=" << skipped << '\n'
            << "payload_mib_s=" << payload_mib_s << '\n'
            << "timeouts=" << counters.timeouts << '\n';
  if (options.synchronize) {
    std::cout << "sync_ok=" << counters.sync_ok << '\n'
              << "sync_failed=" << counters.sync_failed << '\n';
  }
  return counters.frames > 0 && counters.timeouts == 0 ? 0 : 7;
}
