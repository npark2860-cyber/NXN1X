// Windows port: process / thread CPU time (GetProcessTimes / GetThreadTimes) and Win32 priorities for nice values (posix_win.c)
#pragma once
#include <sys/time.h>
#include <sys/types.h>
#define RUSAGE_SELF 0
#define RUSAGE_CHILDREN (-1)
#define RUSAGE_THREAD 1
#define PRIO_PROCESS 0
#define PRIO_PGRP 1
#define PRIO_USER 2
#define PRIO_DARWIN_THREAD 3
#define PRIO_DARWIN_BG 0x1000
#define RLIMIT_NOFILE 8
#define RLIMIT_STACK 3
#define RLIMIT_CORE 4
#define RLIM_INFINITY (~0ull)
typedef unsigned long long rlim_t;
struct rlimit { rlim_t rlim_cur, rlim_max; };
struct rusage { struct timeval ru_utime, ru_stime; long ru_maxrss, ru_ixrss, ru_idrss, ru_isrss, ru_minflt, ru_majflt, ru_nswap, ru_inblock, ru_oublock, ru_msgsnd, ru_msgrcv, ru_nsignals, ru_nvcsw, ru_nivcsw; };
int getrusage(int who, struct rusage *ru);
int getrlimit(int res, struct rlimit *rl);
int setrlimit(int res, const struct rlimit *rl);
int getpriority(int which, int who);
int setpriority(int which, int who, int prio);
