// s46pc (OOT_BARRIER_DESIGN.md 15): what nvnTextureFinalize does with replay work still queued for the replay thread (hle_nvn.c s46, DM_VK_ASYNC > 0).
// The queued lists hold their own copies of the textures they name (tex_snap), and the texture pool registrations are copies kept per queued job (mtl_reg_tex), so
// the finalized object itself is read later only by a queued present (s46_job.t points at the game's object). What finalization does on the host (ft_del, the
// release hint of mtl_tex_release) must come after the queued jobs, in order: it is queued as a job of its own instead of waiting for the queue to empty
// (OoT, 11-21: ~500 such waits per 600 frames, 5-8 ms a frame of the game thread waiting for the replay thread)
#pragma once
#include <stddef.h>
#include <stdint.h>

enum { S46F_NOW = 0, S46F_QUEUE = 1, S46F_DRAIN = 2 };
// jobs: the queue (cap entries of stride bytes; kind at kind_off: 0 submit, 1 present, 2 finalize; the present's texture pointer at tex_off); [tail, head) queued.
// NOW: nothing queued, finalize at once; QUEUE: queue the finalization behind the jobs; DRAIN: wait for the queue to empty first (deferral off, or a queued present
// shows this texture)
static inline int s46_fin_plan(int defer_on, uint64_t head, uint64_t tail, const void *jobs, unsigned cap, size_t stride, size_t kind_off, size_t tex_off, const void *tex) {
    if (head == tail) return S46F_NOW;
    if (!defer_on) return S46F_DRAIN;
    for (uint64_t i = tail; i != head; i++) {
        const unsigned char *j = (const unsigned char *)jobs + (size_t)(i % cap) * stride;
        int kind; const void *t;
        __builtin_memcpy(&kind, j + kind_off, sizeof kind); __builtin_memcpy(&t, j + tex_off, sizeof t);
        if (kind == 1 && t == tex) return S46F_DRAIN;
    }
    return S46F_QUEUE;
}
