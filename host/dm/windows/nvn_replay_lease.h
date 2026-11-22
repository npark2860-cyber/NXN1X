// SPDX-License-Identifier: GPL-3.0-or-later
// A suspended owner keeps logical transaction ownership, not a held mutex.
// Other entrants sleep on the condition variable and cannot mutate its records.
// GPU completion and the private archive worker never enter this gate.
#pragma once
#include <pthread.h>
#include <errno.h>

typedef struct nvnr_gate {
    pthread_mutex_t mutex;
    pthread_cond_t changed;
    pthread_t owner;
    int suspended;
} nvnr_gate;
#define NVNR_GATE_INITIALIZER {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0, 0}
static inline int nvnr_is_owner(const nvnr_gate *g) {
    return g->suspended && pthread_equal(g->owner,pthread_self());
}
static inline int nvnr_lock(nvnr_gate *g) {
    int r=pthread_mutex_lock(&g->mutex);
    if (r) return r;
    while (g->suspended && !nvnr_is_owner(g)) {
        r=pthread_cond_wait(&g->changed,&g->mutex);
        if (r) { pthread_mutex_unlock(&g->mutex); return r; }
    }
    return 0;
}
static inline int nvnr_trylock(nvnr_gate *g) {
    int r=pthread_mutex_trylock(&g->mutex);
    if (r) return r;
    if (g->suspended && !nvnr_is_owner(g)) {
        pthread_mutex_unlock(&g->mutex); return EBUSY;
    }
    return 0;
}
static inline int nvnr_unlock(nvnr_gate *g) { return pthread_mutex_unlock(&g->mutex); }
// Caller holds the mutex. Nested suspension must not discard the outer lease.
static inline int nvnr_suspend(nvnr_gate *g) {
    if (g->suspended) return 0;
    g->owner=pthread_self(); g->suspended=1; return 1;
}
// Caller has reacquired its mutex; it remains held after releasing the lease.
static inline int nvnr_resume(nvnr_gate *g) {
    if (!nvnr_is_owner(g)) return 0;
    g->suspended=0; pthread_cond_broadcast(&g->changed); return 1;
}
