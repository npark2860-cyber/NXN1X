// SPDX-License-Identifier: GPL-2.0-or-later
// F0 diagnostic bookkeeping only. No Vulkan calls, allocation, waits or policy.
#pragma once
#include <stdint.h>
#include <string.h>

typedef struct {
    uint64_t id, frame, first_error_event, fence;
    const char *first_site;
    int32_t first_result;
    uint32_t first_part, errors, accepted_parts, accepted_last_part, final_buffers;
    int dropped;
} nvf0_submission;

static inline int nvf0_error(nvf0_submission *s, int32_t result, const char *site,
                            uint32_t part, uint64_t event) {
    if (!s || !s->id || result >= 0) return 0;
    s->errors++;
    if (!s->first_result) {
        s->first_result = result; s->first_site = site;
        s->first_part = part; s->first_error_event = event;
    }
    return 1;
}
static inline void nvf0_accepted(nvf0_submission *s, uint32_t last_part, uint32_t count) {
    if (!s || !s->id || !count) return;
    s->accepted_parts += count; s->accepted_last_part = last_part;
}

enum {
    NVF0_SLOT_WAIT, NVF0_POOL_WAIT, NVF0_POOL_SYNC, NVF0_TRIM,
    NVF0_SAVE, NVF0_RESTORE, NVF0_IMAGE_ALLOC, NVF0_TRANSFER,
    NVF0_GPU_WAIT, NVF0_COPYBACK, NVF0_MIRROR, NVF0_COMPLETION,
    NVF0_KIND_COUNT, NVF0_MAX_DEPTH = 8
};
typedef struct nvf0_scope {
    struct nvf0_scope *previous;
    uint64_t id, parent, submission, frame, bytes, start, child_ns, elapsed, self_ns;
    uint32_t kind, depth;
} nvf0_scope;

static inline int nvf0_enter(nvf0_scope *s, nvf0_scope **top, uint64_t id,
                            uint32_t kind, uint64_t now, uint64_t submission,
                            uint64_t frame, uint64_t bytes) {
    memset(s, 0, sizeof *s);
    if (!id || kind >= NVF0_KIND_COUNT || (*top && (*top)->depth >= NVF0_MAX_DEPTH)) return 0;
    s->previous = *top; s->parent = *top ? (*top)->id : 0;
    s->depth = *top ? (*top)->depth + 1 : 1;
    s->id = id; s->kind = kind; s->start = now;
    s->submission = submission; s->frame = frame; s->bytes = bytes; *top = s;
    return 1;
}
static inline int nvf0_leave(nvf0_scope *s, nvf0_scope **top, uint64_t now) {
    if (!s->id || *top != s) return 0;
    s->elapsed = now >= s->start ? now - s->start : 0;
    s->self_ns = s->elapsed >= s->child_ns ? s->elapsed - s->child_ns : 0;
    *top = s->previous;
    if (*top) {
        uint64_t old = (*top)->child_ns;
        (*top)->child_ns = old > UINT64_MAX - s->elapsed ? UINT64_MAX : old + s->elapsed;
    }
    return 1;
}
