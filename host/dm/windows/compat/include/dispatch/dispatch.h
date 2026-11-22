// Windows port: the libdispatch subset the host uses (dm/windows/compat/dispatch_win.c). Blocks work with -fblocks and the small runtime in
// dm/windows/compat/blocks_runtime.c.
//   semaphores: counting semaphores on winpthreads' POSIX semaphores, a dispatch_time being an absolute CLOCK_REALTIME instant in ns;
//   the global queues: the Win32 thread pool; a queue from dispatch_queue_create: serial, one worker thread in submission order;
//   dispatch_after: a thread-pool timer; dispatch_once: InitOnceExecuteOnce. Signal sources are not provided (hle_lifecycle.c takes the console path).
#pragma once
#include <semaphore.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <pthread/qos.h>
typedef sem_t *dispatch_semaphore_t;
typedef uint64_t dispatch_time_t;
#define DISPATCH_TIME_NOW 0ull
#define DISPATCH_TIME_FOREVER (~0ull)
#ifndef NSEC_PER_SEC
#define NSEC_PER_SEC 1000000000ull
#define NSEC_PER_MSEC 1000000ull
#define NSEC_PER_USEC 1000ull
#define USEC_PER_SEC 1000000ull
#endif
static inline dispatch_time_t dispatch_time(dispatch_time_t when, int64_t delta) {
    if (when == DISPATCH_TIME_FOREVER) return when;
    if (when == DISPATCH_TIME_NOW) { struct timespec t; clock_gettime(CLOCK_REALTIME, &t); when = (uint64_t)t.tv_sec * NSEC_PER_SEC + (uint64_t)t.tv_nsec; }
    return when + (uint64_t)delta;
}
static inline dispatch_semaphore_t dispatch_semaphore_create(long v) { sem_t *s = malloc(sizeof *s); sem_init(s, 0, (unsigned)v); return s; }
static inline long dispatch_semaphore_signal(dispatch_semaphore_t s) { sem_post(s); return 0; }
static inline long dispatch_semaphore_wait(dispatch_semaphore_t s, dispatch_time_t t) {
    if (t == DISPATCH_TIME_FOREVER) { while (sem_wait(s) && errno == EINTR) ; return 0; }
    struct timespec ts = {(time_t)(t / NSEC_PER_SEC), (long)(t % NSEC_PER_SEC)};
    for (;;) { if (!sem_timedwait(s, &ts)) return 0; if (errno != EINTR) return 1; }
}
static inline void dispatch_release(void *o) { if (o) { sem_destroy((sem_t *)o); free(o); } }

typedef struct dm_dispatch_queue *dispatch_queue_t;
typedef void *dispatch_queue_attr_t;
typedef void (*dispatch_function_t)(void *);
typedef void (^dispatch_block_t)(void);
typedef intptr_t dispatch_once_t;
#define DISPATCH_QUEUE_SERIAL ((dispatch_queue_attr_t)0)
#define DISPATCH_QUEUE_CONCURRENT ((dispatch_queue_attr_t)1)
#define DISPATCH_QUEUE_PRIORITY_HIGH 2
#define DISPATCH_QUEUE_PRIORITY_DEFAULT 0
#define DISPATCH_QUEUE_PRIORITY_LOW (-2)
#define DISPATCH_QUEUE_PRIORITY_BACKGROUND INT16_MIN
dispatch_queue_t dispatch_get_global_queue(intptr_t prio, uintptr_t flags);
dispatch_queue_t dispatch_queue_create(const char *label, dispatch_queue_attr_t attr);
void dispatch_async_f(dispatch_queue_t q, void *ctx, dispatch_function_t fn);
void dispatch_sync_f(dispatch_queue_t q, void *ctx, dispatch_function_t fn);
void dispatch_after_f(dispatch_time_t when, dispatch_queue_t q, void *ctx, dispatch_function_t fn);
void dispatch_once_f(dispatch_once_t *once, void *ctx, dispatch_function_t fn);
void dispatch_async(dispatch_queue_t q, dispatch_block_t b);
void dispatch_sync(dispatch_queue_t q, dispatch_block_t b);
void dispatch_after(dispatch_time_t when, dispatch_queue_t q, dispatch_block_t b);
void dispatch_once(dispatch_once_t *once, dispatch_block_t b);
