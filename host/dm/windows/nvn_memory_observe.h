// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "../nxn1x/nvnh_memory.h"

// Included after the Vulkan function-pointer declarations in gfx_vk.h.
typedef struct { uint64_t id, previous; int begun; } nvnh_restore_scope;
int nvnh_memory_active(void);
VkResult nvnh_memory_allocate_optional(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory*);
uint64_t nvnh_memory_policy(uint32_t op, uint64_t value);
uint64_t nvnh_memory_block_size(uint32_t type, uint64_t desired, uint64_t minimum, uint64_t quantum);
nvnh_restore_scope nvnh_memory_restore_begin(uint32_t type, uint64_t growth, uint64_t payload);
void nvnh_memory_restore_end(nvnh_restore_scope scope, int success);
void nvnh_memory_retire(VkDeviceMemory memory);
void nvnh_memory_local_reject(const char *site, const char *reason, uint64_t bytes);
VkResult nvnh_observe_allocate(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory*, const char*, unsigned, const char*);
void nvnh_observe_free(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*, const char*, unsigned);
VkResult nvnh_observe_create_buffer(VkDevice, const VkBufferCreateInfo*, const VkAllocationCallbacks*, VkBuffer*, const char*, unsigned);
VkResult nvnh_observe_create_image(VkDevice, const VkImageCreateInfo*, const VkAllocationCallbacks*, VkImage*, const char*, unsigned);
VkResult nvnh_observe_bind_buffer(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize, const char*, unsigned);
VkResult nvnh_observe_bind_image(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize, const char*, unsigned);
VkResult nvnh_observe_map(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags, void**, const char*, unsigned);
VkResult nvnh_observe_host_properties(PFN_vkGetMemoryHostPointerPropertiesEXT, VkDevice, VkExternalMemoryHandleTypeFlagBits, const void*, VkMemoryHostPointerPropertiesEXT*, const char*, unsigned);

#ifndef NVNH_MEMORY_IMPLEMENTATION
#define vkAllocateMemory(d,i,a,m) nvnh_observe_allocate(d,i,a,m,__func__,__LINE__,__FILE__)
#define vkFreeMemory(d,m,a) nvnh_observe_free(d,m,a,__func__,__LINE__)
#define vkCreateBuffer(d,i,a,b) nvnh_observe_create_buffer(d,i,a,b,__func__,__LINE__)
#define vkCreateImage(d,i,a,b) nvnh_observe_create_image(d,i,a,b,__func__,__LINE__)
#define vkBindBufferMemory(d,b,m,o) nvnh_observe_bind_buffer(d,b,m,o,__func__,__LINE__)
#define vkBindImageMemory(d,b,m,o) nvnh_observe_bind_image(d,b,m,o,__func__,__LINE__)
#define vkMapMemory(d,m,o,s,f,p) nvnh_observe_map(d,m,o,s,f,p,__func__,__LINE__)
#define NVNH_HOST_PROPERTIES(f,d,t,p,r) nvnh_observe_host_properties(f,d,t,p,r,__func__,__LINE__)
#endif
