// Windows port: force-included (-include) into every host source. The Darwin and POSIX calls the host makes that mingw-w64 lacks are declared here
// or in the wrapper headers beside it (signal.h, unistd.h, ... include mingw's header with #include_next and add the rest), and implemented in
// dm/windows/compat/*.c. <windows.h> is deliberately not included here: its macros (near, far, min, max, ERROR, ...) collide with host names.
#pragma once
#include <TargetConditionals.h>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>   // Darwin's headers bring bool along; host files use it without including stdbool.h
#include <sys/types.h>
#ifndef _POSIX
typedef _sigset_t sigset_t;   // mingw defines it only under _POSIX
#endif
#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <fcntl.h>

// Darwin's pthread_setname_np names only the calling thread (one argument); winpthreads takes the thread too
int dm_setname_self(const char *name);
#define DM_SETNAME_PICK(_1, _2, NAME, ...) NAME
#define pthread_setname_np(...) DM_SETNAME_PICK(__VA_ARGS__, pthread_setname_np, dm_setname_self, x)(__VA_ARGS__)
// the thread's stack: Darwin returns its top (highest address) and size; here from the thread's TEB
void *pthread_get_stackaddr_np(pthread_t t);
size_t pthread_get_stacksize_np(pthread_t t);
int pthread_threadid_np(pthread_t t, uint64_t *id);
// winpthreads' pthread_kill cancels the thread for any signal but 0: signals go through the compat layer's delivery instead (posix_win.c)
int dm_pthread_kill(pthread_t t, int sig);
// Darwin's wait with a relative timeout (posix_win.c): newer winpthreads have one of their own inline in <pthread.h>, older ones none, so the name goes to ours
int dm_cond_timedwait_rel(pthread_cond_t *c, pthread_mutex_t *m, const struct timespec *rel);
#define pthread_cond_timedwait_relative_np dm_cond_timedwait_rel
#define pthread_kill(t, sig) dm_pthread_kill((t), (sig))
static inline int getdtablesize(void) { return 8192; }
#ifndef DM_CLOCK_NSEC_NP
#define DM_CLOCK_NSEC_NP
static inline uint64_t clock_gettime_nsec_np(clockid_t c) { struct timespec t; if (clock_gettime(c, &t)) return 0; return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
#endif
static inline size_t malloc_good_size(size_t n) { return (n + 15) & ~(size_t)15; }
#include <pthread/qos.h>   // Darwin's <pthread.h> brings the QoS classes with it
#define MACH_PORT_NULL 0
#ifdef __BLOCKS__
// The blocks runtime (blocks_runtime.c), declared so that clang links its objects directly rather than as DLL imports (its default on COFF)
extern void *_NSConcreteStackBlock[32], *_NSConcreteGlobalBlock[32];
void *_Block_copy(const void *b);
void _Block_release(const void *b);
void _Block_object_assign(void *dest, const void *obj, int flags);
void _Block_object_dispose(const void *obj, int flags);
#endif
