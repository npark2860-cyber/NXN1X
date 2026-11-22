// Windows port: Mach absolute time = the QueryPerformanceCounter count, timebase from QueryPerformanceFrequency (dm/windows/compat/darwin_compat.c);
// on Windows on Arm that count is the generic timer (CNTVCT), as Mach time is on Apple silicon.
#pragma once
#include <stdint.h>
#include <time.h>
typedef struct { uint32_t numer, denom; } mach_timebase_info_data_t, *mach_timebase_info_t;
int mach_timebase_info(mach_timebase_info_t tb);
uint64_t mach_absolute_time(void);
uint64_t mach_continuous_time(void);
int mach_wait_until(uint64_t deadline);
#ifndef DM_CLOCK_NSEC_NP
#define DM_CLOCK_NSEC_NP
static inline uint64_t clock_gettime_nsec_np(clockid_t c) { struct timespec t; if (clock_gettime(c, &t)) return 0; return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
#endif
