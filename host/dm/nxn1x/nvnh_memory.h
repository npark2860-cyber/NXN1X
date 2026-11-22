// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <stddef.h>
#include <stdint.h>

// F1 notifications remain observation-only. F4 adds separate growth-ticket
// callbacks and pressure polling; callbacks never reclaim or call an owner.
// Handles are correlation tokens, never dereferenced across this boundary.
// notify copies the event into bounded storage and returns an opaque event ID.
// It must not wait, allocate, log, invoke an owner, or call Vulkan. Zero means
// unavailable/dropped; consumers must mark incomplete accounting Unknown.
enum { NVNH_MEMORY_VERSION = 3, NVNH_MEMORY_OBSERVE = 1, NVNH_MEMORY_GROWTH = 2 };
enum { NVNH_OWNER_VMA = 1, NVNH_OWNER_NATIVE = 2, NVNH_OWNER_COUNT = 3 };
enum nvnh_memory_op {
    NVNH_MEM_ALLOCATE, NVNH_MEM_FREE_BEGIN, NVNH_MEM_FREE_END, NVNH_MEM_RETIRE,
    NVNH_MEM_BIND, NVNH_MEM_RESULT, NVNH_MEM_RESTORE_BEGIN, NVNH_MEM_RESTORE_END,
    NVNH_MEM_VMA_LOGICAL, NVNH_MEM_ATTACH, NVNH_MEM_DETACH, NVNH_MEM_LOCAL_REJECT
};
enum nvnh_memory_class {
    NVNH_MEM_OTHER, NVNH_MEM_VMA_BLOCK, NVNH_MEM_IMAGE, NVNH_MEM_SPARSE,
    NVNH_MEM_IMPORT, NVNH_MEM_STAGING, NVNH_MEM_POOL, NVNH_MEM_PRESENT, NVNH_MEM_SCRATCH,
    NVNH_MEM_CLASS_COUNT
};
typedef struct nvnh_memory_event {
    uint32_t op, owner, kind, memory_type;
    int32_t result; // original VkResult ONLY for ALLOCATE/RESULT/BIND
    uint32_t line;
    uint64_t id, related, memory, bytes, object, offset, time_ns, growth_ticket;
    const char *site; // static strings; must outlive device teardown/drain
    const char *operation;
} nvnh_memory_event;
enum { NVNH_POLICY_ENABLE=1, NVNH_POLICY_TAKE_PRESSURE, NVNH_POLICY_RECLAIMED };
typedef struct nvnh_memory_api {
    uint32_t version, size, capabilities, reserved;
    void *context;
    uint64_t (*notify)(void *context, const nvnh_memory_event *event);
    // Separate normal-thread metadata operations. reserve covers the actual
    // vkAllocateMemory size, finish settles it once, policy queues owner work.
    // Cooperative deployment: required allocations retain original VkResults.
    uint64_t (*reserve)(void *context, uint32_t memory_type, uint64_t bytes);
    void (*finish)(void *context, uint64_t ticket, int32_t result);
    uint64_t (*policy)(void *context, uint32_t op, uint32_t owner, uint64_t value);
    uint64_t (*block_size)(void *context, uint32_t memory_type, uint64_t desired, uint64_t minimum, uint64_t quantum);
    // Optional work must obtain an actual block ticket. Zero denies growth;
    // unavailable/stale samples never borrow the required-allocation debt path.
    uint64_t (*reserve_optional)(void *context, uint32_t memory_type, uint64_t bytes);
} nvnh_memory_api;

static inline int nvnh_memory_api_valid(const nvnh_memory_api *api) {
    return api && api->version == NVNH_MEMORY_VERSION && api->size >= sizeof(*api) &&
        (api->capabilities & NVNH_MEMORY_OBSERVE) && api->context && api->notify;
}
#ifdef __cplusplus
extern "C" {
#endif
int nvnh_memory_attach(const nvnh_memory_api *api);
// Called outside owner/queue locks. Close admission, then drain callback readers
// before the C++ context/device is destroyed. No game resource is freed here.
void nvnh_memory_detach(void);
// Constructor-unwind fallback: detach only this owner, never another device.
// Like attach/detach, called under the renderer's lifecycle serialization.
void nvnh_memory_detach_context(const void *context);
#ifdef __cplusplus
}
#endif
