// SPDX-License-Identifier: GPL-3.0-or-later
// F2 observation only. None of these records authorizes an upload, free or writeback.
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define NVNC_ABI 1u
typedef struct nvnc_backing {
    uint64_t id, offset, length;
} nvnc_backing;
typedef struct nvnc_resolver {
    uint32_t version, size;
    // Normal consumer only. Resolve at most 64 KiB, never create an import/view.
    int (*resolve)(uint64_t address, uint64_t max_length, nvnc_backing *out);
    // Atomics only: also called before a faulting NCE store resumes.
    // UINT64_MAX means a mapping change is in progress.
    uint64_t (*epoch)(void);
} nvnc_resolver;

enum {
    NVNC_CPU_PRESTORE = 1, NVNC_CPU_FAULT, NVNC_CPU_FLUSH,
    NVNC_GPU_WRITEBACK, NVNC_DEVICE_WRITE, NVNC_UNCLASSIFIED, NVNC_CPU_UPLOAD
};
enum {
    NVNC_UNKNOWN_NCE = 1u, NVNC_UNKNOWN_PRESTORE = 2u,
    NVNC_UNKNOWN_MAPPING = 4u, NVNC_UNKNOWN_OVERFLOW = 8u,
    NVNC_UNKNOWN_PARTIAL = 16u, NVNC_UNKNOWN_FAILED = 32u,
    NVNC_UNKNOWN_IDENTITY = 64u, NVNC_UNKNOWN_RAW = 128u,
    NVNC_UNKNOWN_ORDER = 256u
};
typedef struct nvnc_region_snapshot {
    uint64_t backing, block_offset, all_gen, cpu_gen, device_gen, writeback_gen;
    uint64_t map_epoch, identity_epoch;
    uint32_t unknown;
} nvnc_region_snapshot;
typedef struct nvnc_stats {
    uint64_t metadata_bytes, queued, consumed, dropped, unknown_epoch, backlog;
    uint64_t unknown_reasons, unsupported_events;
    uint64_t pools, images, regions, cpu_events, fault_events, flush_events, writebacks;
    uint64_t cpu_bytes, writeback_bytes, device_bytes, unresolved_bytes, alias_hits;
    uint64_t intents, upload_intents, accepted_intents, rejected_intents, gpu_completed, cpu_completed;
    uint64_t first_error_sub, first_error_part;
    int32_t first_error;
    uint64_t pool_protected_checks, pool_protected_bytes, no_image_write_checks, pool_protected_true;
    uint64_t observed_cpu_regions, observed_gpu_regions, observed_mixed_regions;
    uint64_t drain_ns, drain_max_ns, bounded_yields;
    uint64_t range_raw, range_coalesced, range_pending, range_passes, range_dropped, queue_dropped;
} nvnc_stats;

int nvnc_set_resolver(const nvnc_resolver *resolver); // before start, lifecycle serialized
void nvnc_start(int native_nvn2); // NXN1X_NVN_CONTENT=0: no allocation or hooks
void nvnc_stop(void); // outside owner locks; drain resolver reader, retain bounded fault storage
int nvnc_active(void);
// CPU/fault producers: bounded lock-free queue, no resolver, allocator, I/O or mutex.
void nvnc_cpu(uint64_t address, uint64_t length, unsigned origin);
unsigned nvnc_origin_push(unsigned origin);
void nvnc_origin_pop(unsigned old_origin);
unsigned nvnc_origin(void);
void nvnc_unknown(unsigned reason);
void nvnc_pool_add(uint64_t key, uint64_t address, uint64_t size);
void nvnc_pool_del(uint64_t key);
uint64_t nvnc_image_add(uint64_t pool_key, uint64_t address, uint64_t size, uint64_t layout);
void nvnc_image_del(uint64_t image);
void nvnc_pool_protected(uint64_t image, uint64_t bytes, int image_written, int pool_written);
uint64_t nvnc_sub_begin(void);
void nvnc_part(unsigned part);
void nvnc_image_intent(uint64_t image);
void nvnc_image_upload(uint64_t image); // CPU source transported by a GPU command
void nvnc_range_intent(uint64_t address, uint64_t size);
void nvnc_accept(uint64_t sub, unsigned first, unsigned last, int32_t result);
void nvnc_error(unsigned part, int32_t result);
void nvnc_sub_end(void);
void nvnc_gpu_done(uint64_t sub);
void nvnc_cpu_done(uint64_t sub);
// Single bounded consumer at an owner safe point. No owner callbacks or data copies.
void nvnc_drain(unsigned event_limit, unsigned region_limit);
void nvnc_get_stats(nvnc_stats *out);
// Diagnostic only; Unknown is sticky. Never interpret a zero cpu_gen as guest validity.
int nvnc_probe(uint64_t address, nvnc_region_snapshot *out);
#ifdef __cplusplus
}
#endif
