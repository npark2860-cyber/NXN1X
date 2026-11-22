// Windows port: Darwin's arm64 general register state, shared by mach/mach.h (thread_get_state) and sys/ucontext.h (signal handlers) without either
// pulling in the other (winpthreads' <pthread.h> includes <signal.h>, which includes sys/ucontext.h, before pthread_t exists)
#pragma once
#include <stdint.h>
struct __darwin_arm_thread_state64 { uint64_t __x[29], __fp, __lr, __sp, __pc; uint32_t __cpsr, __pad; };
