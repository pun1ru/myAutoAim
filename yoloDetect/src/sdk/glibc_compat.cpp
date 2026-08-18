#include <cstdlib>

#ifdef __linux__
// Daedalus Simulator 1.3.1 SDK was built against glibc 2.38. Ubuntu 22.04
// provides the pre-C23 entry point, which has the same behavior for this SDK.
extern "C" long long __isoc23_strtoll(const char* text, char** end, int base) {
  return std::strtoll(text, end, base);
}
#endif
