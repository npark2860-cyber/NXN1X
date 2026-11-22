// SPDX-License-Identifier: GPL-3.0-or-later
// No Vulkan dispatch patching: OFF calls each original function exactly once.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#define NVNH_MEMORY_IMPLEMENTATION
#include "nvn_memory_observe.h"
#include <stdatomic.h>
#include <string.h>
#include <sched.h>
#include "nvn_memory_content.h"
_Static_assert(ATOMIC_LLONG_LOCK_FREE == 2, "F1 callbacks require lock-free 64-bit atomics");

#define DECLARE(f) extern PFN_##f f;
DECLARE(vkAllocateMemory) DECLARE(vkFreeMemory) DECLARE(vkCreateBuffer)
DECLARE(vkCreateImage) DECLARE(vkBindBufferMemory) DECLARE(vkBindImageMemory)
DECLARE(vkMapMemory)
#undef DECLARE

// Close bit and reader count are one atomic: a detach cannot miss a reader
// which was about to acquire the context. Attach/detach are lifecycle-serialized.
#define CLOSED (UINT64_C(1) << 63)
static _Atomic uint64_t readers = CLOSED;
static nvnh_memory_api bridge;
static _Thread_local uint64_t restore_id;
int nvnh_memory_active(void) { return !(atomic_load_explicit(&readers, memory_order_acquire) & CLOSED); }
static uint64_t emit(nvnh_memory_event e) {
    if (!nvnh_memory_active()) return 0;
    uint64_t before = atomic_fetch_add_explicit(&readers, 1, memory_order_acq_rel);
    uint64_t id = 0;
    if (!(before & CLOSED)) {
        e.owner = NVNH_OWNER_NATIVE;
        id = bridge.notify(bridge.context, &e);
    }
    atomic_fetch_sub_explicit(&readers, 1, memory_order_release);
    return id;
}
int nvnh_memory_attach(const nvnh_memory_api *api) {
    if (!nvnh_memory_api_valid(api) || atomic_load_explicit(&readers, memory_order_acquire) != CLOSED) return -1;
    bridge = *api;
    atomic_store_explicit(&readers, 0, memory_order_release);
    emit((nvnh_memory_event){.op=NVNH_MEM_ATTACH, .site="native", .operation="attach"});
    return 0;
}
void nvnh_memory_detach(void) {
    nvnc_stop(); // no deferred resolver work after owner teardown
    atomic_fetch_or_explicit(&readers, CLOSED, memory_order_acq_rel);
    while (atomic_load_explicit(&readers, memory_order_acquire) != CLOSED) sched_yield();
    if (bridge.notify) {
        nvnh_memory_event e = {.op=NVNH_MEM_DETACH, .owner=NVNH_OWNER_NATIVE, .site="native", .operation="detach"};
        bridge.notify(bridge.context, &e);
        memset(&bridge, 0, sizeof bridge);
    }
}
void nvnh_memory_detach_context(const void *context) {
    // Lifecycle callers are serialized, so the API cannot be replaced between
    // this identity check and closing admission. Callback readers use atomics.
    if (bridge.context == context) nvnh_memory_detach();
}
static unsigned classify(const char *site, const char *unit) {
    if (strstr(site, "nvna_workspace")) return NVNH_MEM_STAGING;
    if (strstr(site, "s2sp") || strstr(site, "s2tile") || strstr(site, "sparse")) return NVNH_MEM_SPARSE;
    if (strstr(site, "image_memory")) return NVNH_MEM_IMAGE;
    if (strstr(site, "nvn2_alloc")) return NVNH_MEM_SCRATCH;
    if (strstr(site, "pool") || strstr(site, "bridge_zero")) return NVNH_MEM_POOL;
    if (strstr(unit, "gfx_vk_present.c")) return NVNH_MEM_PRESENT;
    return NVNH_MEM_OTHER;
}
uint64_t nvnh_memory_policy(uint32_t op, uint64_t value) {
    if (!nvnh_memory_active()) return 0;
    uint64_t before=atomic_fetch_add_explicit(&readers,1,memory_order_acq_rel), out=0;
    if (!(before&CLOSED) && (bridge.capabilities&NVNH_MEMORY_GROWTH) && bridge.policy)
        out=bridge.policy(bridge.context,op,NVNH_OWNER_NATIVE,value);
    atomic_fetch_sub_explicit(&readers,1,memory_order_release); return out;
}
uint64_t nvnh_memory_block_size(uint32_t type, uint64_t desired, uint64_t minimum, uint64_t quantum) {
    if (!nvnh_memory_active()) return desired;
    uint64_t before=atomic_fetch_add_explicit(&readers,1,memory_order_acq_rel), out=desired;
    if (!(before&CLOSED) && (bridge.capabilities&NVNH_MEMORY_GROWTH) && bridge.block_size)
        out=bridge.block_size(bridge.context,type,desired,minimum,quantum);
    atomic_fetch_sub_explicit(&readers,1,memory_order_release);
    return out>=minimum && out<=desired && quantum && !(out%quantum) ? out : desired;
}
VkResult nvnh_observe_allocate(VkDevice d, const VkMemoryAllocateInfo *a, const VkAllocationCallbacks *c, VkDeviceMemory *m, const char *site, unsigned line, const char *unit) {
    uint64_t reader=CLOSED, ticket=0;
    const int acquired=nvnh_memory_active();
    if (acquired) reader=atomic_fetch_add_explicit(&readers,1,memory_order_acq_rel);
    const int entered=!(reader&CLOSED);
    if (entered && (bridge.capabilities&NVNH_MEMORY_GROWTH) && bridge.reserve && bridge.finish)
        ticket=bridge.reserve(bridge.context,a->memoryTypeIndex,a->allocationSize);
    // The bridge reader spans the driver call and settlement. Detach cannot
    // destroy the coordinator while an outstanding ticket still names it.
    VkResult r=vkAllocateMemory(d,a,c,m);
    if (entered) {
        if (ticket) bridge.finish(bridge.context,ticket,r);
        nvnh_memory_event e={.op=NVNH_MEM_ALLOCATE,.owner=NVNH_OWNER_NATIVE,.kind=classify(site,unit),.memory_type=a->memoryTypeIndex,
            .result=r,.line=line,.related=restore_id,.memory=r==VK_SUCCESS?(uint64_t)*m:0,
            .bytes=a->allocationSize,.site=site,.operation="vkAllocateMemory",.growth_ticket=ticket};
        for(const VkBaseInStructure *p=a->pNext;p;p=p->pNext) if(p->sType==VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT) {
            e.kind=NVNH_MEM_IMPORT; e.object=(uint64_t)((const VkImportMemoryHostPointerInfoEXT*)p)->pHostPointer;
        }
        bridge.notify(bridge.context,&e);
    }
    if (acquired) atomic_fetch_sub_explicit(&readers,1,memory_order_release);
    return r;
}
void nvnh_observe_free(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks *c, const char *site, unsigned line) {
    uint64_t id = m ? emit((nvnh_memory_event){.op=NVNH_MEM_FREE_BEGIN, .memory=(uint64_t)m, .site=site, .line=line, .operation="vkFreeMemory"}) : 0;
    vkFreeMemory(d, m, c);
    if (id) emit((nvnh_memory_event){.op=NVNH_MEM_FREE_END, .memory=(uint64_t)m, .related=id, .site=site, .line=line, .operation="vkFreeMemory"});
}
VkResult nvnh_memory_allocate_optional(VkDevice d, const VkMemoryAllocateInfo *a,
                                     const VkAllocationCallbacks *c, VkDeviceMemory *m) {
    *m=VK_NULL_HANDLE;
    if (!nvnh_memory_active()) return VK_NOT_READY;
    const uint64_t before=atomic_fetch_add_explicit(&readers,1,memory_order_acq_rel);
    VkResult result=VK_NOT_READY;
    if (!(before&CLOSED) && (bridge.capabilities&NVNH_MEMORY_GROWTH) &&
        bridge.reserve_optional && bridge.finish) {
        const uint64_t ticket=bridge.reserve_optional(bridge.context,a->memoryTypeIndex,a->allocationSize);
        if (ticket) {
            result=vkAllocateMemory(d,a,c,m);
            bridge.finish(bridge.context,ticket,result);
            const nvnh_memory_event e={.op=NVNH_MEM_ALLOCATE,.owner=NVNH_OWNER_NATIVE,
                .kind=NVNH_MEM_STAGING,.memory_type=a->memoryTypeIndex,.result=result,
                .memory=result==VK_SUCCESS?(uint64_t)*m:0,.bytes=a->allocationSize,
                .growth_ticket=ticket,.site="optional-preservation",.operation="vkAllocateMemory"};
            bridge.notify(bridge.context,&e);
        }
        // A local denial is not a driver OOM and has no allocation event.
    }
    atomic_fetch_sub_explicit(&readers,1,memory_order_release);
    return result;
}
static void result(VkResult r, const char *op, const char *site, unsigned line, uint64_t bytes, uint64_t memory) {
    if (r != VK_SUCCESS) emit((nvnh_memory_event){.op=NVNH_MEM_RESULT, .result=r, .memory_type=UINT32_MAX,
        .bytes=bytes, .memory=memory, .related=restore_id, .site=site, .line=line, .operation=op});
}
VkResult nvnh_observe_create_buffer(VkDevice d, const VkBufferCreateInfo *i, const VkAllocationCallbacks *a, VkBuffer *b, const char *s, unsigned l) {
    VkResult r = vkCreateBuffer(d,i,a,b); result(r,"vkCreateBuffer",s,l,i->size,0); return r;
}
VkResult nvnh_observe_create_image(VkDevice d, const VkImageCreateInfo *i, const VkAllocationCallbacks *a, VkImage *b, const char *s, unsigned l) {
    VkResult r = vkCreateImage(d,i,a,b); result(r,"vkCreateImage",s,l,0,0); return r;
}
static void binding(VkResult r, const char *op, const char *s, unsigned l, uint64_t object, VkDeviceMemory m, uint64_t offset) {
    emit((nvnh_memory_event){.op=NVNH_MEM_BIND, .result=r, .memory=(uint64_t)m, .object=object, .offset=offset,
        .memory_type=UINT32_MAX, .site=s, .line=l, .operation=op, .related=restore_id});
}
VkResult nvnh_observe_bind_buffer(VkDevice d, VkBuffer b, VkDeviceMemory m, VkDeviceSize o, const char *s, unsigned l) {
    VkResult r = vkBindBufferMemory(d,b,m,o); binding(r,"vkBindBufferMemory",s,l,(uint64_t)b,m,o); return r;
}
VkResult nvnh_observe_bind_image(VkDevice d, VkImage b, VkDeviceMemory m, VkDeviceSize o, const char *s, unsigned l) {
    VkResult r = vkBindImageMemory(d,b,m,o); binding(r,"vkBindImageMemory",s,l,(uint64_t)b,m,o); return r;
}
VkResult nvnh_observe_map(VkDevice d, VkDeviceMemory m, VkDeviceSize o, VkDeviceSize n, VkMemoryMapFlags f, void **p, const char *s, unsigned l) {
    VkResult r = vkMapMemory(d,m,o,n,f,p); result(r,"vkMapMemory",s,l,n,(uint64_t)m); return r;
}
VkResult nvnh_observe_host_properties(PFN_vkGetMemoryHostPointerPropertiesEXT f, VkDevice d, VkExternalMemoryHandleTypeFlagBits t, const void *p, VkMemoryHostPointerPropertiesEXT *v, const char *s, unsigned l) {
    VkResult r = f(d,t,p,v); result(r,"vkGetMemoryHostPointerPropertiesEXT",s,l,0,0); return r;
}
nvnh_restore_scope nvnh_memory_restore_begin(uint32_t type, uint64_t growth, uint64_t payload) {
    nvnh_restore_scope s = {.previous=restore_id, .begun=1};
    s.id = emit((nvnh_memory_event){.op=NVNH_MEM_RESTORE_BEGIN, .memory_type=type, .bytes=growth,
        .object=payload, .site="img_archive_restore", .operation="restore-growth"});
    restore_id = s.id;
    return s;
}
void nvnh_memory_restore_end(nvnh_restore_scope s, int success) {
    if (!s.begun) return;
    if (s.id) emit((nvnh_memory_event){.op=NVNH_MEM_RESTORE_END, .related=s.id, .result=success,
        .site="img_archive_restore", .operation="restore-outcome-local"});
    restore_id = s.previous;
}
void nvnh_memory_retire(VkDeviceMemory m) {
    if (m) emit((nvnh_memory_event){.op=NVNH_MEM_RETIRE, .memory=(uint64_t)m, .site="df_add", .operation="deferred-free"});
}
void nvnh_memory_local_reject(const char *s, const char *reason, uint64_t bytes) {
    emit((nvnh_memory_event){.op=NVNH_MEM_LOCAL_REJECT, .site=s, .operation=reason, .bytes=bytes, .memory_type=UINT32_MAX});
}
