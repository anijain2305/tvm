#ifndef TVM_CYCLE_COUNTER_H
#define TVM_CYCLE_COUNTER_H

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#ifdef CYCLE_ARM_CORTEX
#include "cycle_platform_arm_cortex_m.h"
#else
#include "cycle_platform_x86.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CYCLE_COUNTERS 32


//////////////////////////////////////////////////////////////////////////////
/// @brief Platform-specific callback for obtaining cycle count.
///
typedef prlLong (*CycleCounter_getCount)(void);


//////////////////////////////////////////////////////////////////////////////
/// @brief Initialize Cycle Count Tracker
///
/// @param getCount Callback for obtaining cycle count.
///
/// @return Zero for success, non-zero error code otherwise.
int cycleCounter_init(CycleCounter_getCount getCount);


//////////////////////////////////////////////////////////////////////////////
/// @brief Start counting cycles.
void cycleCounter_start(int counterIndex);


//////////////////////////////////////////////////////////////////////////////
/// @brief Stop counting cycles.
///
/// @return The elapsed cycle count.
prlInt cycleCounter_stop(int counterIndex);


#define MAX_LOG_CHARS (256)


inline void prl_assert_msg(bool condition, const char* file, int line, const char* format, ...)
{
  if (!condition)
  {
    // first format the message
    va_list ap;

    // NOTE this assumes single-threaded only!
    static char message[MAX_LOG_CHARS];

    va_start(ap, format);
    int ret = vsnprintf(message, sizeof(message), format, ap);
    va_end(ap);
    if(ret < 0)
    {
#ifdef _MSC_VER
      strerror_s(message, sizeof(message), ret);
            fprintf(stderr, "Warning: String formatting failed! %s", message);
#else
      fprintf(stderr, "Warning: String formatting failed! %s", strerror(ret));
#endif
    }

    // write to stderr
    // only need the filename, not the full path in the output. The strrchr calls strip the path.
    fprintf(stderr, "Assert Failed: %s: %d, '%s'\n", (strrchr(file, '/') ? strrchr(file, '/') + 1 : file), line, message);


  }
}

#define PRL_ASSERT_MSG(condition, format, ...) prl_assert_msg(condition, __FILE__, __LINE__, format, ##__VA_ARGS__)


/// @brief function to demote a long integer to integer, with saturation
inline int32_t prlMathLongToIntSat(int64_t input)
{
  if (input > INT32_MAX)
  {
    return INT32_MAX;
  }
  else if (input < INT32_MIN)
  {
    return INT32_MIN;
  }
  else
  {
    return (int32_t)input;
  }
}



#ifdef __cplusplus
} // extern "C"
#endif

#endif  // TVM_CYCLE_COUNTER_H
