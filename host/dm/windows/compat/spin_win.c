// Windows port (r128, 11-10): pthread_spin_* in user mode, in place of winpthreads' own (libpthread.a, libwinpthread_la-spinlock.o).
// The installed winpthreads keeps an auto-reset event per spin lock and calls WaitForSingleObject on every lock and SetEvent on every unlock,
// contended or not. winpthreads takes these spin locks inside pthread_rwlock_rdlock / _unlock (two each: eight kernel calls per read lock held
// and released) and inside pthread_getspecific / _setspecific. On Donkey Kong's floor 200 the replay thread spent about three quarters of its
// samples in those two kernel calls (WPR, r127); Bayonetta 3's MainDraw lost a third of its time to pthread_getspecific (r98).
// These definitions come first on the link line, so the library's spinlock.o is not pulled in and winpthreads' own calls land here.
// The lock is the pthread_spinlock_t word itself: -1 free (PTHREAD_SPINLOCK_INITIALIZER), 0 invalid (destroyed), otherwise the owner's thread id.
// The return values follow the installed library: EDEADLK when the owner locks again, EPERM when another thread unlocks, EBUSY from
// trylock and from destroying a held lock, EINVAL for NULL or a destroyed lock, ENOSYS for PTHREAD_PROCESS_SHARED.
// -DDM_SPIN_WINPTHREADS=1 (DM_WIN_CFLAGS) leaves this file empty: the library's version is linked again.
#ifndef DM_SPIN_WINPTHREADS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <pthread.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

_Static_assert(sizeof(pthread_spinlock_t) == sizeof(intptr_t), "pthread_spinlock_t is not one pointer-sized word: spin_win.c does not fit this winpthreads");

#define SPIN_FREE ((intptr_t)-1)

// the static initialiser must be the free value: winpthreads' own static spin locks (rwl_global, ...) start as PTHREAD_SPINLOCK_INITIALIZER
__attribute__((constructor)) static void spin_check_init(void) {
    pthread_spinlock_t s = PTHREAD_SPINLOCK_INITIALIZER;
    if ((intptr_t)s != SPIN_FREE) { fprintf(stderr, "[spin_win] PTHREAD_SPINLOCK_INITIALIZER is %#llx, not -1: rebuild with DM_WIN_CFLAGS=-DDM_SPIN_WINPTHREADS=1\n", (unsigned long long)(intptr_t)s); abort(); }
}

static inline intptr_t spin_self(void) { return (intptr_t)GetCurrentThreadId(); }

int pthread_spin_init(pthread_spinlock_t *lock, int pshared) {
    if (!lock) return EINVAL;
    if (pshared != PTHREAD_PROCESS_PRIVATE && pshared != PTHREAD_PROCESS_SHARED) return EINVAL;
    if (pshared == PTHREAD_PROCESS_SHARED) return ENOSYS;
    __atomic_store_n((intptr_t *)lock, SPIN_FREE, __ATOMIC_RELEASE);
    return 0;
}

int pthread_spin_destroy(pthread_spinlock_t *lock) {
    if (!lock) return EINVAL;
    intptr_t v = SPIN_FREE;
    if (__atomic_compare_exchange_n((intptr_t *)lock, &v, 0, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) return 0;
    return v == 0 ? EINVAL : EBUSY;
}

int pthread_spin_lock(pthread_spinlock_t *lock) {
    if (!lock) return EINVAL;
    intptr_t *w = (intptr_t *)lock, me = spin_self();
    for (unsigned n = 0;; n++) {
        intptr_t v = __atomic_load_n(w, __ATOMIC_RELAXED);
        if (v == 0) return EINVAL;
        if (v == me) return EDEADLK;
        if (v == SPIN_FREE && __atomic_compare_exchange_n(w, &v, me, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return 0;
        // a held lock: spin briefly, then give the core away (the owner may be preempted or suspended), sleeping now and then
        if (n < 128) YieldProcessor(); else if (n & 1023) SwitchToThread(); else Sleep(1);
    }
}

int pthread_spin_trylock(pthread_spinlock_t *lock) {
    if (!lock) return EINVAL;
    intptr_t v = SPIN_FREE;
    if (__atomic_compare_exchange_n((intptr_t *)lock, &v, spin_self(), 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return 0;
    return v == 0 ? EINVAL : EBUSY;
}

int pthread_spin_unlock(pthread_spinlock_t *lock) {
    if (!lock) return EINVAL;
    intptr_t *w = (intptr_t *)lock, v = __atomic_load_n(w, __ATOMIC_RELAXED);
    if (v == 0) return EINVAL;
    if (v != spin_self()) return EPERM;
    __atomic_store_n(w, SPIN_FREE, __ATOMIC_RELEASE);
    return 0;
}
#endif
