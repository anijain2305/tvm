

//------ Include files -------------------------------------------------------

#include "cycle_counter.h"
#include <stddef.h> // For NULL
#include <limits.h> // For UINT_MAX
#include <stdio.h>

//------ Private macros ------------------------------------------------------
//------ Private Variables ---------------------------------------------------

static CycleCounter_getCount gGetCount = NULL; ///< Callback for getting cycle count

static prlLong gStartCount[CYCLE_COUNTERS]; ///< Start count for computing differential

//------ Private Function Declarations ---------------------------------------
//------ Function Definitions ----- ------------------------------------------

// See declaration for function description
int cycleCounter_init(CycleCounter_getCount getCount)
{
  int result;

  if ((gGetCount == NULL) && (getCount != NULL)) {
    gGetCount = getCount;
    result = 0;
  }
  else { // Callback already registered
    result = 1;
  }

  return result;
}


// See declaration for function description
void cycleCounter_start(int counterIndex)
{
  PRL_ASSERT_MSG(counterIndex < CYCLE_COUNTERS, "counter index out of range");
  if (gGetCount != NULL) {
    gStartCount[counterIndex] = gGetCount();
  }

  return;
}


// See declaration for function description
prlInt cycleCounter_stop(int counterIndex)
{
  prlLong diff = 0;

  PRL_ASSERT_MSG(counterIndex < CYCLE_COUNTERS, "counter index out of range");
  if (gGetCount != NULL) {
    prlLong endCount = gGetCount();

    // Multiple wraps not handled.
    if (endCount >= gStartCount[counterIndex]) { // No wrapping detected
      diff = endCount - gStartCount[counterIndex];
    }
    else { // Handle single-wrap condition
      diff = (PRL_LONG_MAX - gStartCount[counterIndex]) + (endCount - PRL_LONG_MIN) + 1;
    }
  }

  return prlMathLongToIntSat(diff);
}
