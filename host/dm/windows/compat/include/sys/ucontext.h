// Windows port: Darwin's arm64 ucontext, which the host's signal handlers read and write (uc->uc_mcontext->__ss / __ns / __es).
// The compat layer (posix_win.c) fills it from the Windows CONTEXT before calling a handler and copies __ss / __ns back afterwards.
// __es.__esr is made up from the exception: a data abort (EC 0x24) with WnR set for a write, an instruction abort (EC 0x20) for an execute fault.
#pragma once
#include <stdint.h>
#include <signal.h>
#include <mach/arm_thread_state.h>
struct __darwin_arm_exception_state64 { uint64_t __far; uint32_t __esr, __exception; };
struct __darwin_arm_neon_state64 { __uint128_t __v[32]; uint32_t __fpsr, __fpcr; };
struct __darwin_mcontext64 { struct __darwin_arm_exception_state64 __es; struct __darwin_arm_thread_state64 __ss; struct __darwin_arm_neon_state64 __ns; };
typedef struct __darwin_mcontext64 *mcontext_t;
#define _STRUCT_ARM_NEON_STATE64 struct __darwin_arm_neon_state64
#define _STRUCT_ARM_EXCEPTION_STATE64 struct __darwin_arm_exception_state64
#define _STRUCT_MCONTEXT64 struct __darwin_mcontext64
typedef struct dm_ucontext { int uc_onstack; sigset_t uc_sigmask; stack_t uc_stack; struct dm_ucontext *uc_link; size_t uc_mcsize; struct __darwin_mcontext64 *uc_mcontext;
                             struct __darwin_mcontext64 __mcontext_data; } ucontext_t;
