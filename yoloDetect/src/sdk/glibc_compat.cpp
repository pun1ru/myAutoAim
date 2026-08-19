#include <cstdlib>

#ifdef __linux__
// Daedalus Simulator 1.3.1 SDK 基于 glibc 2.38 构建。Ubuntu 22.04 提供
// 行为相同的 C23 之前入口点，故为 SDK 提供兼容实现。
extern "C" long long __isoc23_strtoll(const char* text, char** end, int base) {
  return std::strtoll(text, end, base);
}
#endif
