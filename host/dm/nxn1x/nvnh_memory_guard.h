// SPDX-License-Identifier: GPL-3.0-or-later
// Session error boundary. No guest code or process termination crosses this API.
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
enum { NVNH_STOP_POLICY=1, NVNH_STOP_DRIVER=2, NVNH_STOP_INTERNAL=3 };
typedef struct nvnh_stop_info {
    int reason, result;
    uint64_t bytes;
    const char *site; // static storage only
} nvnh_stop_info;
int nvnh_guard_run(void (*body)(void *), void *context);
int nvnh_guard_active(void);
int nvnh_guard_enabled(void);
void nvnh_guard_enable(int native_switch2);
int nvnh_memory_stopped(void);
int nvnh_memory_stop_info(nvnh_stop_info *out);
void nvnh_guard_set_notify(void (*notify)(void)); // lifecycle registration
void nvnh_guard_notify(void); // no owner/budget locks held
void nvnh_memory_latch(int reason, int result, const char *site, uint64_t bytes);
void nvnh_memory_stop(int reason, int result, const char *site, uint64_t bytes);
void nvnh_guard_check(void);
// Native lock ownership is retained on the host stack during normal execution.
// On a session error, the outer host boundary releases these in reverse order.
void nvnh_guard_track(void (*release)(void *), void *object);
void nvnh_guard_untrack(void (*release)(void *), void *object);
void nvnh_memory_wake_waiters(void);
void nvnh_replay_shutdown(void);
void nvnh_vk_completion_shutdown(void);
int nvnh_work_get(void);
void nvnh_work_restore(int value);
#ifdef __cplusplus
}
#endif
