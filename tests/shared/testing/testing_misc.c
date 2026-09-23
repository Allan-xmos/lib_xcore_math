// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
// XMOS Public License: Version 1

#include "testing.h"

#include <math.h>
#include <stdio.h>

#ifdef __xcore__
 #include "xcore/hwtimer.h"
#endif


unsigned get_seed(
    const char* str, 
    const unsigned len)
{
  unsigned seed = 0;
  int left = len;

  while(left > 0){
    unsigned v = ((unsigned*)str)[0];
    seed = seed ^ v;
    left -= 4;
    str = &str[4];
  }

  return seed;
}

int64_t vect_s16_dot_expected(
    const int16_t b[],
    const int16_t c[],
    const unsigned length)
{
  int64_t sum = 0;

  for(unsigned int k = 0; k < length; k++){
    sum += ((int32_t) b[k]) * c[k];
#if defined(__VX4B__)
    if((b[k] & 1) && (c[k] & 1))
      sum -= 1;
#endif
  }

  return sum;
}

unsigned getTimestamp()
{
#if __xcore__
  return get_reference_time();
#else
  return 0;
#endif
}