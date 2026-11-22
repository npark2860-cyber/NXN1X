// SPDX-License-Identifier: GPL-2.0-or-later
// Bounded preservation worker. The owner pins the source record until release.
// Callbacks run without engine/owner locks; only the owner publishes a copy.
#pragma once
#include <stdint.h>
#include <stdatomic.h>
typedef struct nvnp_job {
    struct nvnp_job *queue_next;
    const uint64_t *live_version;
    uint64_t image_id, version, bytes;
    _Atomic int cancel, done;
    int result, admitted;
} nvnp_job;
typedef int (*nvnp_save_fn)(nvnp_job *job);
int nvnp_start(nvnp_save_fn save);
int nvnp_submit(nvnp_job *job);
int nvnp_changed(const nvnp_job *job);
void nvnp_cancel(nvnp_job *job);
int nvnp_done(const nvnp_job *job);
int nvnp_publishable(const nvnp_job *job, uint64_t image_id, int same_image, int orphan);
void nvnp_release(nvnp_job *job); // only after done; caller frees outside owner locks
uint64_t nvnp_pending_bytes(void);
uint64_t nvnp_progress(void);
int nvnp_working(void);
int nvnp_wait_progress(uint64_t observed, uint64_t timeout_ns); // outside owner/submission locks
void nvnp_stop(void); // cancel, join, then owner collects; no owner/queue lock held
enum { NVNP_JOBS = 64, NVNP_BYTES = 128 << 20 };
