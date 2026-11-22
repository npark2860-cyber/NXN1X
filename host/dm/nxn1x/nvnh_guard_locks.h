// SPDX-License-Identifier: GPL-3.0-or-later
// Forced into native C rendering units only, never the NCE fault path or C++.
#pragma once
#include <pthread.h>
#include "nvnh_memory_guard.h"
int nvnh_guard_mutex_lock(pthread_mutex_t *);
int nvnh_guard_mutex_trylock(pthread_mutex_t *);
int nvnh_guard_mutex_unlock(pthread_mutex_t *);
int nvnh_guard_rwlock_rdlock(pthread_rwlock_t *);
int nvnh_guard_rwlock_wrlock(pthread_rwlock_t *);
int nvnh_guard_rwlock_unlock(pthread_rwlock_t *);
#define pthread_mutex_lock nvnh_guard_mutex_lock
#define pthread_mutex_trylock nvnh_guard_mutex_trylock
#define pthread_mutex_unlock nvnh_guard_mutex_unlock
#define pthread_rwlock_rdlock nvnh_guard_rwlock_rdlock
#define pthread_rwlock_wrlock nvnh_guard_rwlock_wrlock
#define pthread_rwlock_unlock nvnh_guard_rwlock_unlock
