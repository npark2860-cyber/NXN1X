// SPDX-License-Identifier: GPL-3.0-or-later
// CPU-only: real F1 wrappers/bridge linked against stub Vulkan functions.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#define NVNH_MEMORY_IMPLEMENTATION
#include "../nvn_memory_observe.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <pthread.h>
#include <sched.h>

static _Atomic int checks;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while(0)
static nvnh_memory_event events[128];
static _Atomic unsigned count;
static _Atomic int blocking, entered, release_reader, detached;
static VkResult answer;
static int original_calls;
static uint64_t notify(void *ctx, const nvnh_memory_event *e) {
    CHECK(ctx == events);
    unsigned n = atomic_fetch_add(&count, 1);
    CHECK(n < 128); events[n] = *e;
    if (atomic_load(&blocking) && e->op == NVNH_MEM_ALLOCATE) {
        atomic_store(&entered, 1);
        while (!atomic_load(&release_reader)) sched_yield();
    }
    return n + 1;
}
static VKAPI_ATTR VkResult VKAPI_CALL alloc(VkDevice d, const VkMemoryAllocateInfo *a, const VkAllocationCallbacks *c, VkDeviceMemory *m) {
    (void)d; (void)a; (void)c; ++original_calls;
    if (answer == VK_SUCCESS) *m = (VkDeviceMemory)(uintptr_t)0x1234;
    return answer;
}
static VKAPI_ATTR void VKAPI_CALL release(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks *c) {
    (void)d; (void)m; (void)c; ++original_calls;
    if (nvnh_memory_active() && m) CHECK(events[atomic_load(&count)-1].op == NVNH_MEM_FREE_BEGIN);
}
static VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice d, const VkBufferCreateInfo *i, const VkAllocationCallbacks *a, VkBuffer *b) { (void)d; (void)i; (void)a; (void)b; ++original_calls; return answer; }
static VKAPI_ATTR VkResult VKAPI_CALL create_image(VkDevice d, const VkImageCreateInfo *i, const VkAllocationCallbacks *a, VkImage *b) { (void)d; (void)i; (void)a; (void)b; ++original_calls; return answer; }
static VKAPI_ATTR VkResult VKAPI_CALL bind_buffer(VkDevice d, VkBuffer b, VkDeviceMemory m, VkDeviceSize o) { (void)d; (void)b; (void)m; (void)o; ++original_calls; return answer; }
static VKAPI_ATTR VkResult VKAPI_CALL bind_image(VkDevice d, VkImage b, VkDeviceMemory m, VkDeviceSize o) { (void)d; (void)b; (void)m; (void)o; ++original_calls; return answer; }
static VKAPI_ATTR VkResult VKAPI_CALL map(VkDevice d, VkDeviceMemory m, VkDeviceSize o, VkDeviceSize n, VkMemoryMapFlags f, void **p) { (void)d; (void)m; (void)o; (void)n; (void)f; (void)p; ++original_calls; return answer; }
static VKAPI_ATTR VkResult VKAPI_CALL host(VkDevice d, VkExternalMemoryHandleTypeFlagBits t, const void *p, VkMemoryHostPointerPropertiesEXT *v) { (void)d; (void)t; (void)p; (void)v; ++original_calls; return answer; }
PFN_vkAllocateMemory vkAllocateMemory = alloc;
PFN_vkFreeMemory vkFreeMemory = release;
PFN_vkCreateBuffer vkCreateBuffer = create_buffer;
PFN_vkCreateImage vkCreateImage = create_image;
PFN_vkBindBufferMemory vkBindBufferMemory = bind_buffer;
PFN_vkBindImageMemory vkBindImageMemory = bind_image;
PFN_vkMapMemory vkMapMemory = map;
static VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, 1024, 3};
static void *reader(void *ignored) { (void)ignored; VkDeviceMemory m; nvnh_observe_allocate(NULL,&ai,NULL,&m,"thread",1,__FILE__); return NULL; }
static void *closer(void *ignored) { (void)ignored; nvnh_memory_detach_context(events); atomic_store(&detached,1); return NULL; }
int main(void) {
    VkDeviceMemory memory = (VkDeviceMemory)(uintptr_t)99;
    CHECK(!nvnh_memory_active());
    CHECK(nvnh_observe_allocate(NULL,&ai,NULL,&memory,"off",1,__FILE__) == VK_SUCCESS);
    nvnh_observe_free(NULL,memory,NULL,"off",2);
    CHECK(original_calls == 2 && count == 0);
    nvnh_memory_api api = {NVNH_MEMORY_VERSION,sizeof(api),NVNH_MEMORY_OBSERVE,0,events,notify};
    api.version++; CHECK(nvnh_memory_attach(&api) == -1); api.version--;
    api.size--; CHECK(nvnh_memory_attach(&api) == -1); api.size++;
    CHECK(nvnh_memory_attach(&api) == 0); CHECK(nvnh_memory_attach(&api) == -1);
    CHECK(nvnh_memory_active());
    nvnh_memory_detach_context(&api); CHECK(nvnh_memory_active());
    const VkResult failures[] = {VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_DEVICE_LOST, VK_ERROR_INVALID_EXTERNAL_HANDLE};
    for (unsigned i=0; i<4; ++i) {
        answer = failures[i]; unsigned before = count; int calls = original_calls;
        CHECK(nvnh_observe_allocate(NULL,&ai,NULL,&memory,"failed",17,__FILE__) == answer);
        CHECK(original_calls == calls + 1 && count == before + 1);
        CHECK(events[before].result == answer && events[before].memory == 0);
        CHECK(events[before].bytes == 1024 && events[before].memory_type == 3 && events[before].line == 17);
    }
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 4096};
    VkImageCreateInfo ii = {.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    VkBuffer buffer = NULL; VkImage image = NULL; void *mapped = NULL;
    VkMemoryHostPointerPropertiesEXT hp = {.sType=VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    CHECK(nvnh_observe_create_buffer(NULL,&bi,NULL,&buffer,"create",1) == answer);
    CHECK(nvnh_observe_create_image(NULL,&ii,NULL,&image,"create",2) == answer);
    CHECK(nvnh_observe_bind_buffer(NULL,NULL,memory,0,"bind",3) == answer);
    CHECK(nvnh_observe_bind_image(NULL,NULL,memory,0,"bind",4) == answer);
    CHECK(nvnh_observe_map(NULL,memory,0,1024,0,&mapped,"map",5) == answer);
    CHECK(nvnh_observe_host_properties(host,NULL,VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,NULL,&hp,"host",6) == answer);
    for (unsigned i=count-6; i<count; ++i) CHECK(events[i].result == answer);
    answer = VK_SUCCESS;
    VkImportMemoryHostPointerInfoEXT imp = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, (void*)(uintptr_t)0x10000};
    ai.pNext = &imp;
    nvnh_restore_scope scope = nvnh_memory_restore_begin(3, 8192, 1024);
    unsigned before = count;
    CHECK(nvnh_observe_allocate(NULL,&ai,NULL,&memory,"import",27,__FILE__) == VK_SUCCESS);
    CHECK(events[before].kind == NVNH_MEM_IMPORT && events[before].object == 0x10000);
    CHECK(events[before].related == scope.id);
    nvnh_memory_restore_end(scope,1);
    nvnh_memory_retire(memory);
    before = count;
    nvnh_observe_free(NULL,memory,NULL,"free",33);
    CHECK(count == before + 2 && events[before+1].op == NVNH_MEM_FREE_END);
    CHECK(events[before+1].related == before+1);
    nvnh_memory_local_reject("import","no-type",999);
    CHECK(events[count-1].op == NVNH_MEM_LOCAL_REJECT);
    // Detach waits for an existing callback, closes new callbacks immediately,
    // and never changes the driver's answer. No GPU or game is involved.
    atomic_store(&blocking,1);
    pthread_t r, c; CHECK(!pthread_create(&r,NULL,reader,NULL));
    while (!atomic_load(&entered)) sched_yield();
    CHECK(!pthread_create(&c,NULL,closer,NULL));
    while (nvnh_memory_active()) sched_yield();
    CHECK(!atomic_load(&detached));
    before = count;
    CHECK(nvnh_observe_allocate(NULL,&ai,NULL,&memory,"closed",1,__FILE__) == VK_SUCCESS);
    CHECK(count == before);
    atomic_store(&release_reader,1);
    CHECK(!pthread_join(r,NULL)); CHECK(!pthread_join(c,NULL));
    CHECK(atomic_load(&detached) && !nvnh_memory_active());
    before = count; nvnh_memory_detach(); CHECK(count == before);
    atomic_store(&blocking,0);
    CHECK(nvnh_memory_attach(&api) == 0); nvnh_memory_detach();
    printf("PASS native wrappers/ABI/detach: %d checks; Vulkan/GPU/game NOT_RUN\n",checks);
    return 0;
}
