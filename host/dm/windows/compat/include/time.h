// Windows port: the reentrant time calls (mingw has them with _POSIX_THREAD_SAFE_FUNCTIONS), timegm = _mkgmtime, Darwin's raw clocks = CLOCK_MONOTONIC.
// struct tm has no tm_zone / tm_gmtoff here: dm_tm_zone / dm_tm_gmtoff give them (posix_win.c).
#pragma once
#ifndef _POSIX_THREAD_SAFE_FUNCTIONS
#define _POSIX_THREAD_SAFE_FUNCTIONS 200112L
#endif
#include_next <time.h>
#include <pthread_time.h>
#ifndef CLOCK_MONOTONIC_RAW
#define CLOCK_MONOTONIC_RAW CLOCK_MONOTONIC
#endif
#ifndef CLOCK_UPTIME_RAW
#define CLOCK_UPTIME_RAW CLOCK_MONOTONIC
#endif
#define timegm _mkgmtime
const char *dm_tm_zone(const struct tm *tm);
long dm_tm_gmtoff(const struct tm *tm);
