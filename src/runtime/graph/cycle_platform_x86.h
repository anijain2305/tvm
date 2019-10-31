#ifndef TVM_CYCLE_PLATFORM_X86_H
#define TVM_CYCLE_PLATFORM_X86_H

#include <stdint.h>
#include <stdbool.h>
#include <limits.h>

#ifndef _MSC_VER
#include <sys/time.h>
#else
#include <windows.h>
#endif

typedef int32_t prlInt;   // int (at least 16-bit)
typedef int64_t prlLong;  // long int (at least 32-bit)
#define PRL_INT_MAX    INT32_MAX
#define PRL_LONG_MAX   INT64_MAX
#define PRL_LONG_MIN   INT64_MIN

// @brief: return current clock in usec
static long inline prl_get_clock(void)
{
  long r;
#ifdef _MSC_VER
  // TODO: Use the other tick counter w/ clock frequency
    r = 1000 * GetTickCount();
#else
  struct timeval result;
  gettimeofday(&result, 0);
  r = 1000000 * result.tv_sec + result.tv_usec;
#endif
  return r;
}


#define PRL_GET_CLOCK prl_get_clock
#define PRL_INIT_CLOCK()



#endif  // TVM_CYCLE_PLATFORM_X86_H
