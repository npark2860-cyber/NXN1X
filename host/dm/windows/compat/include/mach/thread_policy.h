// Windows port: Mach thread policies. The time-constraint (real-time) class maps to THREAD_PRIORITY_TIME_CRITICAL for the calling thread
// (dm/windows/compat/darwin_compat.c); thread_policy_get reports the default.
#pragma once
#include <stdint.h>
typedef int *thread_policy_t;
typedef int thread_policy_flavor_t;
#define THREAD_TIME_CONSTRAINT_POLICY 2
typedef struct { uint32_t period, computation, constraint; int preemptible; } thread_time_constraint_policy_data_t;
#define THREAD_TIME_CONSTRAINT_POLICY_COUNT ((unsigned)(sizeof(thread_time_constraint_policy_data_t) / sizeof(int)))
int thread_policy_set(int thread, thread_policy_flavor_t flavor, thread_policy_t policy, unsigned count);
int thread_policy_get(int thread, thread_policy_flavor_t flavor, thread_policy_t policy, unsigned *count, int *get_default);
