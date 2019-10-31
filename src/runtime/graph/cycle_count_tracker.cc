//------ Include files -------------------------------------------------------

#include "cycle_count_tracker.h"
#include <dmlc/logging.h>

//------ Private macros ------------------------------------------------------

#define ARRAY_LEN(x) (sizeof(x) / sizeof((x)[0])) ///< Report length of array

//------ Public Variables ---------------------------------------------------

SPrlCycleCountRecord gCycleCountBuffer[CYCLE_COUNT_TRACKER_BUF_LEN]; ///< History of counted cycles
#if CYCLE_COUNT_TRACKER_BUF_LEN > PRL_INT_MAX
LOG(ERROR) << "You have way too many entries in your cycle count buffer."
#endif
int gCycleCountBufIdx; ///< Index for history buffer
prlInt  gCycleCountMax; ///< Maximum recorded cycle count
prlInt  gCycleCountMin; ///< Minimum recorded cycle count
const char* gCycleCountRecordNames[CYCLE_COUNTERS] = {0};

//------ Private Variables ---------------------------------------------------

//------ Private Function Declarations ---------------------------------------
static int cycleCountLookupName(const char *str);
static prlLong getClock(void);
static void initClock(void);

/// @brief clock getter
static prlLong getClock(void)
{
  prlLong ret = PRL_GET_CLOCK();
  //printf("clock=" PRIld "\n", FORMAT_LONG(ret));
  return ret;
}

static void initClock(void)
{
  PRL_INIT_CLOCK();
  return;
}

//////////////////////////////////////////////////////////////////////////////
/// @brief Lookup a string in the record names array.
/// if the name is found, return the index.
/// if the name is not found, return the name index available, and update the record name array
/// if the record name array is full, return -1
/// TODO: use a hashmap, rather than a lookup array.
static int cycleCountLookupName(const char *str)
{
  int index = 0;
  while (index < CYCLE_COUNTERS) {
    if (gCycleCountRecordNames[index] == NULL && (strnlen(str, CYCLE_COUNT_MAX_RECORD_NAME_STRLEN + 1) <= CYCLE_COUNT_MAX_RECORD_NAME_STRLEN)) {
      gCycleCountRecordNames[index] = str;
      return index;
    }

    if ((strncmp(str, gCycleCountRecordNames[index], CYCLE_COUNT_MAX_RECORD_NAME_STRLEN)) == 0)
    {
      return index;
    }
    index++;
  }
  LOG(WARNING) << "Warning: cycleCountLookupName failed";
  return -1;
}
//------ Function Definitions ----- ------------------------------------------

// See declaration for function description
int cycleCountTracker_init(void)
{
  size_t i;
  for (i = 0; i < ARRAY_LEN(gCycleCountBuffer); i++) {
    gCycleCountBuffer[i].cycleCount = 0;
    gCycleCountBuffer[i].recordName = NULL;
  }
  gCycleCountBufIdx = 0;
  gCycleCountMax = 0;
  gCycleCountMin = PRL_INT_MAX;

  initClock();

  return cycleCounter_init(getClock);
}


// See declaration for function description
void cycleCountTracker_start(const char *str)
{
  int counterIndex;

  counterIndex = cycleCountLookupName(str);
  if (counterIndex == -1) {
    return;
  }

  cycleCounter_start(counterIndex);
}


// See declaration for function description
prlInt cycleCountTracker_stop(const char *str)
{
  int counterIndex;

  counterIndex = cycleCountLookupName(str);
  if (counterIndex == -1) {
    return 0;
  }

  return cycleCounter_stop(counterIndex);
}

// See declaration for function description
void cycleCountTracker_record(const char *str, prlInt elapsed)
{
  int counterIndex;

  counterIndex = cycleCountLookupName(str);
  if (counterIndex == -1) {
    return;
  }

  gCycleCountBuffer[gCycleCountBufIdx].cycleCount = elapsed;
  gCycleCountBuffer[gCycleCountBufIdx++].recordName = str;

  if (elapsed > gCycleCountMax) {
    gCycleCountMax = elapsed;
  }

  if (elapsed < gCycleCountMin) {
    gCycleCountMin = elapsed;
  }

  if (gCycleCountBufIdx >= CYCLE_COUNT_TRACKER_BUF_LEN) {
    gCycleCountBufIdx = 0;
  }

}

void cycleCountTracker_dump(const char *filename)
{
  FILE *f;
  int i;
  int is_stdout = 0;

  f = fopen(filename, "w");
  if (f == nullptr)
  {
    f = stdout;
    is_stdout = 1;
  }

  fprintf(f, "min count = %d, max count = %d\n", gCycleCountMin, gCycleCountMax);
  fprintf(f, "Cycle count buffer index = %d\n", gCycleCountBufIdx);
  for (i = 0; i < gCycleCountBufIdx; i++) {
    fprintf(f, "%s, %d\n", gCycleCountBuffer[i].recordName, gCycleCountBuffer[i].cycleCount);
  }

  if (is_stdout == 0)
  {
    fclose(f);
    // also print to stdout.
    f = stdout;
    fprintf(f, "min count = %d, max count = %d\n", gCycleCountMin, gCycleCountMax);
    fprintf(f, "Cycle count buffer index = %d\n", gCycleCountBufIdx);
    for (i = 0; i < gCycleCountBufIdx; i++) {
      fprintf(f, "%s, %d\n", gCycleCountBuffer[i].recordName, gCycleCountBuffer[i].cycleCount);
    }
  }
}
