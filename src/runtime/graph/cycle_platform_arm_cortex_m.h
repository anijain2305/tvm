#ifndef TVM_CYCLE_ARM_CORTEX_M_H
#define TVM_CYCLE_ARM_CORTEX_M_H

#include <cstdint>
#include <ctime>


typedef int32_t prlInt;   // int (at least 16-bit)
typedef int64_t prlLong;  // long int (at least 32-bit)
#define PRL_INT_MAX    INT32_MAX
#define PRL_LONG_MAX   INT64_MAX
#define PRL_LONG_MIN   INT64_MIN

// @brief: return current clock count
static long inline prl_get_clock(void)
{
  //DWT_CYCCNT is the cycle count register on ARM CortexM cores.
  //Reading DWT_CYCCNT register provides the cycle count.
  //Documentation regarding this can be found at links:
  // http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.ddi0489c/BABJFFGJ.html
  volatile uint32_t *DWT_CYCCNT = (uint32_t *) 0xE0001004;
  return (*DWT_CYCCNT);
}
static void inline prl_start_clock(void){
  //Initialise cycle counter using various registers on ARM Cortex M cores.
  //Documentation regarding this can be found at links:
  // http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.ddi0489c/BABJFFGJ.html
  // https://mcuoneclipse.com/2017/01/30/cycle-counting-on-arm-cortex-m-with-dwt/
  volatile uint32_t *DWT_CONTROL = (uint32_t *) 0xE0001000;
  volatile uint32_t *DWT_CYCCNT = (uint32_t *) 0xE0001004;
  volatile uint32_t *DEMCR = (uint32_t *) 0xE000EDFC;
  volatile uint32_t *LAR  = (uint32_t *) 0xE0001FB0;
  // enable trace
  *DEMCR = *DEMCR | 0x01000000;
  // provides unlock access to DWT register
  *LAR = 0xC5ACCE55;
  // clear DWT cycle counter
  *DWT_CYCCNT = 0;
  // enable DWT cycle counter
  *DWT_CONTROL = *DWT_CONTROL | 1;

}
#define PRL_GET_CLOCK prl_get_clock
#define PRL_INIT_CLOCK() prl_start_clock()


#endif  // TVM_CYCLE_ARM_CORTEX_M_H
