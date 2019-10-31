#ifndef TVM_CYCLE_COUNT_TRACKER_H
#define TVM_CYCLE_COUNT_TRACKER_H


#include "cycle_counter.h"
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <string.h>

#define CYCLE_COUNT_REPORTING
#ifdef CYCLE_COUNT_REPORTING
#define CYC_PROF_INIT() cycleCountTracker_init()
#define CYC_PROF_START(str) cycleCountTracker_start(str)
#define CYC_PROF_STOP(str) cycleCountTracker_record(str, cycleCountTracker_stop(str))
#define CYC_PROF_DUMP(filename) cycleCountTracker_dump(filename)
#else
#define CYC_PROF_INIT()
#define CYC_PROF_START(str)
#define CYC_PROF_STOP(str)
#define CYC_PROF_DUMP(filename)
//#error No cycle count
#endif


#define CYCLE_COUNT_TRACKER_BUF_LEN (65536) ///< Length of cycle count history buffer

#define CYCLE_COUNT_MAX_RECORD_NAME_STRLEN  (64)

typedef struct SPrlCycleCountRecord {
  prlInt cycleCount;
  const char *recordName;
} SPrlCycleCountRecord;

extern SPrlCycleCountRecord gCycleCountBuffer[CYCLE_COUNT_TRACKER_BUF_LEN]; ///< History of counted cycles
extern int gCycleCountBufIdx; ///< Index for history buffer (next write)
extern prlInt gCycleCountMax; ///< Maximum recorded cycle count
extern prlInt gCycleCountMin; ///< Minimum recorded cycle count
extern const char* gCycleCountRecordNames[CYCLE_COUNTERS];

//////////////////////////////////////////////////////////////////////////////
/// @brief Initialize Cycle Count Tracker
///
/// Initialization routine for tracking of cycle counts.
///
/// @return Zero for success, non-zero error code otherwise.
int cycleCountTracker_init(void);

//////////////////////////////////////////////////////////////////////////////
/// @brief Start counting cycles.
void cycleCountTracker_start(const char *str);


//////////////////////////////////////////////////////////////////////////////
/// @brief Stop counting cycles.
///
/// @return Elapsed cycle count since start.
prlInt cycleCountTracker_stop(const char *str);


//////////////////////////////////////////////////////////////////////////////
/// @brief Record elapsed count to history, update min/max counts
///
/// @param elapsed [in] Cycles elapsed (return value from stop function)
void cycleCountTracker_record(const char *str, prlInt elapsed);

//////////////////////////////////////////////////////////////////////////////
/// @brief Dump cycle count records to file
/// if file is NULL, dump to stdout
///
void cycleCountTracker_dump(const char *filename);

#ifdef __cplusplus
} // extern "C"
#endif


#endif  // TVM_CYCLE_COUNT_TRACKER_H
