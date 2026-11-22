// Windows port: the libdispatch queues the host uses (dm/windows/compat/include/dispatch/dispatch.h): the global queues on the Win32 thread pool, a created
// queue as one worker thread running its items in submission order, dispatch_after on a thread that sleeps, dispatch_once on InitOnceExecuteOnce
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dispatch/dispatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Block.h>

typedef struct item { struct item *next; void *ctx; dispatch_function_t fn; HANDLE done; } item_t;
struct dm_dispatch_queue { int global; SRWLOCK lock; CONDITION_VARIABLE cv; item_t *head, *tail; HANDLE thread; char label[64]; };
static struct dm_dispatch_queue g_global = {.global = 1};

dispatch_queue_t dispatch_get_global_queue(intptr_t prio, uintptr_t flags) { (void)prio; (void)flags; return &g_global; }
static DWORD WINAPI queue_main(LPVOID p) {
    dispatch_queue_t q = p;
    for (;;) {
        AcquireSRWLockExclusive(&q->lock);
        while (!q->head) SleepConditionVariableSRW(&q->cv, &q->lock, INFINITE, 0);
        item_t *it = q->head; q->head = it->next; if (!q->head) q->tail = NULL;
        ReleaseSRWLockExclusive(&q->lock);
        it->fn(it->ctx);
        if (it->done) SetEvent(it->done); else free(it);
    }
}
dispatch_queue_t dispatch_queue_create(const char *label, dispatch_queue_attr_t attr) {
    (void)attr;   // concurrent queues are made serial: none of the host's needs more
    dispatch_queue_t q = calloc(1, sizeof *q); if (!q) return NULL;
    InitializeSRWLock(&q->lock); InitializeConditionVariable(&q->cv);
    snprintf(q->label, sizeof q->label, "%s", label ? label : "queue");
    q->thread = CreateThread(NULL, 0, queue_main, q, 0, NULL);
    return q;
}
static void enqueue(dispatch_queue_t q, item_t *it) {
    AcquireSRWLockExclusive(&q->lock);
    if (q->tail) q->tail->next = it; else q->head = it;
    q->tail = it;
    ReleaseSRWLockExclusive(&q->lock); WakeConditionVariable(&q->cv);
}
typedef struct { void *ctx; dispatch_function_t fn; } pool_job;
static VOID CALLBACK pool_run(PTP_CALLBACK_INSTANCE inst, PVOID p) { (void)inst; pool_job j = *(pool_job *)p; free(p); j.fn(j.ctx); }
void dispatch_async_f(dispatch_queue_t q, void *ctx, dispatch_function_t fn) {
    if (q->global) {
        pool_job *j = malloc(sizeof *j); j->ctx = ctx; j->fn = fn;
        if (!TrySubmitThreadpoolCallback(pool_run, j, NULL)) { free(j); fn(ctx); }
        return;
    }
    item_t *it = calloc(1, sizeof *it); it->ctx = ctx; it->fn = fn; enqueue(q, it);
}
void dispatch_sync_f(dispatch_queue_t q, void *ctx, dispatch_function_t fn) {
    if (q->global || GetThreadId(q->thread) == GetCurrentThreadId()) { fn(ctx); return; }
    item_t it = {.ctx = ctx, .fn = fn, .done = CreateEventW(NULL, FALSE, FALSE, NULL)};
    enqueue(q, &it); WaitForSingleObject(it.done, INFINITE); CloseHandle(it.done);
}
typedef struct { dispatch_time_t when; dispatch_queue_t q; void *ctx; dispatch_function_t fn; } after_t;
static DWORD WINAPI after_main(LPVOID p) {
    after_t *a = p; struct timespec t; clock_gettime(CLOCK_REALTIME, &t);
    uint64_t now = (uint64_t)t.tv_sec * NSEC_PER_SEC + (uint64_t)t.tv_nsec;
    if (a->when > now) Sleep((DWORD)((a->when - now + 999999) / 1000000));
    dispatch_async_f(a->q, a->ctx, a->fn); free(a); return 0;
}
void dispatch_after_f(dispatch_time_t when, dispatch_queue_t q, void *ctx, dispatch_function_t fn) {
    after_t *a = malloc(sizeof *a); *a = (after_t){when, q, ctx, fn};
    HANDLE h = CreateThread(NULL, 0, after_main, a, 0, NULL); if (h) CloseHandle(h); else { free(a); dispatch_async_f(q, ctx, fn); }
}
typedef struct { void *ctx; dispatch_function_t fn; } once_job;
static BOOL CALLBACK once_run(PINIT_ONCE o, PVOID p, PVOID *c) { (void)o; (void)c; once_job *j = p; j->fn(j->ctx); return TRUE; }
void dispatch_once_f(dispatch_once_t *once, void *ctx, dispatch_function_t fn) {
    _Static_assert(sizeof(dispatch_once_t) == sizeof(INIT_ONCE), "dispatch_once_t holds an INIT_ONCE");
    once_job j = {ctx, fn}; InitOnceExecuteOnce((PINIT_ONCE)once, once_run, &j, NULL);
}
static void run_block(void *b) { dispatch_block_t blk = b; blk(); Block_release(blk); }
static void run_block_keep(void *b) { dispatch_block_t blk = b; blk(); }
void dispatch_async(dispatch_queue_t q, dispatch_block_t b) { dispatch_async_f(q, Block_copy(b), run_block); }
void dispatch_sync(dispatch_queue_t q, dispatch_block_t b) { dispatch_sync_f(q, (void *)b, run_block_keep); }
void dispatch_after(dispatch_time_t when, dispatch_queue_t q, dispatch_block_t b) { dispatch_after_f(when, q, Block_copy(b), run_block); }
void dispatch_once(dispatch_once_t *once, dispatch_block_t b) { dispatch_once_f(once, (void *)b, run_block_keep); }
