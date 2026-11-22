// SPDX-License-Identifier: GPL-3.0-or-later
#include "nvnh_memory_guard.h"
#include <atomic>
#include <array>
#include <cstdlib>
#include <cstring>
#include <new>
#include <pthread.h>

namespace {
struct Stop {};
struct Lease { void (*release)(void *); void *object; };
thread_local std::array<Lease,256> leases;
thread_local unsigned used{}, depth{};
std::atomic<unsigned> stopped{}, notified{};
std::atomic<bool> enabled{};
nvnh_stop_info info{};
std::atomic<void (*)(void)> notification{};
void Notify() {
    if(!enabled.load(std::memory_order_acquire)) return;
    if (stopped.load(std::memory_order_acquire)!=2) return;
    auto fn=notification.load(std::memory_order_acquire);
    if (fn && !notified.exchange(1,std::memory_order_acq_rel)) fn();
}
void Unlock(void *m) { pthread_mutex_unlock(static_cast<pthread_mutex_t*>(m)); }
void ReadUnlock(void *m) { pthread_rwlock_unlock(static_cast<pthread_rwlock_t*>(m)); }
}
extern "C" int nvnh_guard_active(void) { return depth!=0; }
extern "C" int nvnh_guard_enabled(void) { return enabled.load(std::memory_order_relaxed); }
extern "C" void nvnh_guard_enable(int native_switch2) {
    bool on=native_switch2!=0;
    for(const char *name:{"DM_VK_FOUNDATION_F1","DM_VK_FOUNDATION_F4","DM_VK_FOUNDATION_F5","DM_VK_FOUNDATION_F5R"}) {
        const char *v=std::getenv(name); if(v && !std::strcmp(v,"0")) on=false;
    }
    enabled.store(on,std::memory_order_release);
}
extern "C" int nvnh_memory_stopped(void) { return enabled.load(std::memory_order_acquire) && stopped.load(std::memory_order_acquire)!=0; }
extern "C" int nvnh_memory_stop_info(nvnh_stop_info *out) {
    if (stopped.load(std::memory_order_acquire)!=2 || !out) return 0;
    *out=info; return 1;
}
extern "C" void nvnh_guard_set_notify(void (*fn)(void)) { notification.store(fn,std::memory_order_release); }
extern "C" void nvnh_guard_notify(void) { Notify(); }
extern "C" void nvnh_memory_latch(int reason,int result,const char *site,uint64_t bytes) {
    if(!nvnh_guard_enabled()) return;
    unsigned expected=0;
    if (stopped.compare_exchange_strong(expected,1,std::memory_order_acq_rel)) {
        info={reason,result,bytes,site}; stopped.store(2,std::memory_order_release);
    }
}
extern "C" void nvnh_memory_stop(int reason,int result,const char *site,uint64_t bytes) {
    nvnh_memory_latch(reason,result,site,bytes);
    // Only a host error boundary can unwind. Background transfer callbacks
    // return their normal error and leave the original image alive.
    if (depth && nvnh_memory_stopped()) throw Stop{};
}
extern "C" void nvnh_guard_check(void) { if (depth && nvnh_memory_stopped()) throw Stop{}; }
extern "C" void nvnh_guard_track(void (*release)(void *),void *object) {
    if (!depth) return;
    if (used==leases.size()) {
        release(object);
        nvnh_memory_stop(NVNH_STOP_INTERNAL,0,"native lock tracking exhausted",0);
        return;
    }
    leases[used++]={release,object};
}
extern "C" void nvnh_guard_untrack(void (*release)(void *),void *object) {
    if (!depth) return;
    for (unsigned i=used;i;i--) if(leases[i-1].release==release && leases[i-1].object==object) {
        for(unsigned j=i;j<used;j++) leases[j-1]=leases[j];
        --used; return;
    }
}
extern "C" int nvnh_guard_run(void (*body)(void *),void *context) {
    if (!nvnh_guard_enabled()) { body(context); return 1; }
    if (depth) { nvnh_guard_check(); body(context); return 1; }
    if (nvnh_memory_stopped()) { Notify(); return 0; }
    const int work=nvnh_work_get();
    depth=1;
    try {
        body(context);
        nvnh_guard_check();
        if (used) nvnh_memory_stop(NVNH_STOP_INTERNAL,0,"native boundary retained locks",0);
        depth=0; return 1;
    } catch (const Stop&) {
        // C++ stack objects have unwound normally. Only native C locks are
        // registered here; their storage outlives this boundary.
        while(used) { const auto lease=leases[--used]; lease.release(lease.object); }
        nvnh_work_restore(work); depth=0; Notify(); return 0;
    } catch (const std::bad_alloc&) {
        while(used) { const auto lease=leases[--used]; lease.release(lease.object); }
        nvnh_work_restore(work); depth=0;
        nvnh_memory_stop(NVNH_STOP_INTERNAL,0,"host allocation failed at native boundary",0);
        Notify(); return 0;
    } catch (...) {
        while(used) { const auto lease=leases[--used]; lease.release(lease.object); }
        nvnh_work_restore(work); depth=0;
        if(nvnh_memory_stopped()) { Notify(); return 0; }
        throw;
    }
}
extern "C" int nvnh_guard_mutex_lock(pthread_mutex_t *m) {
    const int r=pthread_mutex_lock(m); if(!r) nvnh_guard_track(Unlock,m); return r;
}
extern "C" int nvnh_guard_mutex_trylock(pthread_mutex_t *m) {
    const int r=pthread_mutex_trylock(m); if(!r) nvnh_guard_track(Unlock,m); return r;
}
extern "C" int nvnh_guard_mutex_unlock(pthread_mutex_t *m) {
    nvnh_guard_untrack(Unlock,m); return pthread_mutex_unlock(m);
}
extern "C" int nvnh_guard_rwlock_rdlock(pthread_rwlock_t *m) {
    const int r=pthread_rwlock_rdlock(m); if(!r) nvnh_guard_track(ReadUnlock,m); return r;
}
extern "C" int nvnh_guard_rwlock_wrlock(pthread_rwlock_t *m) {
    const int r=pthread_rwlock_wrlock(m); if(!r) nvnh_guard_track(ReadUnlock,m); return r;
}
extern "C" int nvnh_guard_rwlock_unlock(pthread_rwlock_t *m) {
    nvnh_guard_untrack(ReadUnlock,m); return pthread_rwlock_unlock(m);
}
