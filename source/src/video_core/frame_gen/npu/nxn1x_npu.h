// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <stdint.h>
#include <wchar.h>

#if defined(_WIN32) && defined(NXN1X_NPU_BUILD_DLL)
#define NXN1X_NPU_API __declspec(dllexport)
#else
#define NXN1X_NPU_API
#endif
#if defined(_WIN32)
#define NXN1X_NPU_CALL __cdecl
#else
#define NXN1X_NPU_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum { NXN1X_NPU_ABI = 1, NXN1X_NPU_WIDTH = 1280, NXN1X_NPU_HEIGHT = 720 };

typedef struct NXN1XNpu NXN1XNpu;

typedef enum NXN1XNpuResult {
    NXN1X_NPU_OK = 0,
    NXN1X_NPU_EMPTY = 1,
    NXN1X_NPU_BUSY = 2,
    NXN1X_NPU_WARMING = 3,
    NXN1X_NPU_UNAVAILABLE = 4,
    NXN1X_NPU_STALE = 5,
    NXN1X_NPU_INVALID = -1,
    NXN1X_NPU_ERROR = -2
} NXN1XNpuResult;

typedef enum NXN1XNpuState {
    NXN1X_NPU_INITIALIZING = 0,
    NXN1X_NPU_READY = 1,
    NXN1X_NPU_FAILED = 2,
    NXN1X_NPU_STOPPING = 3
} NXN1XNpuState;

typedef enum NXN1XNpuPixelFormat {
    NXN1X_NPU_RGB24 = 0,
    NXN1X_NPU_RGBA8 = 1,
    NXN1X_NPU_BGRA8 = 2
} NXN1XNpuPixelFormat;

typedef struct NXN1XNpuConfig {
    uint32_t struct_size;
    uint32_t abi_version;
    const wchar_t* runtime_directory;
    const wchar_t* motion_model_path;
    const wchar_t* anvil_model_path;
    // 1 or 2. Counts pending, executing, and completed-but-not-taken jobs together.
    uint32_t capacity;
    uint32_t reserved;
} NXN1XNpuConfig;

typedef struct NXN1XNpuImage {
    const uint8_t* pixels;
    uint64_t data_bytes;
    uint32_t width;
    uint32_t height;
    uint32_t row_bytes;
    uint32_t format;
} NXN1XNpuImage;

typedef struct NXN1XNpuPair {
    uint32_t struct_size;
    uint32_t reserved;
    uint64_t epoch;
    uint64_t previous_id;
    uint64_t current_id;
    // Use NXN1XNpuNowNs(), not wall-clock or guest emulation time.
    int64_t deadline_ns;
    NXN1XNpuImage previous;
    NXN1XNpuImage current;
} NXN1XNpuPair;

typedef struct NXN1XNpuOutput {
    uint32_t struct_size;
    uint32_t reserved;
    uint64_t epoch;
    uint64_t previous_id;
    uint64_t current_id;
    int64_t deadline_ns;
    double job_ms;
    double motion_ms;
    double prepare_ms;
    double anvil_ms;
    double compose_ms;
    uint32_t scene_cut;
    uint32_t valid_pixels;
} NXN1XNpuOutput;

typedef struct NXN1XNpuStats {
    uint32_t struct_size;
    uint32_t state;
    uint64_t epoch;
    uint64_t accepted;
    uint64_t completed;
    uint64_t delivered;
    uint64_t busy;
    uint64_t expired_or_reset;
    uint64_t errors;
    uint32_t occupied;
    uint32_t capacity;
    double last_job_ms;
    char last_error[384];
} NXN1XNpuStats;

NXN1X_NPU_API uint32_t NXN1X_NPU_CALL NXN1XNpuAbiVersion(void);
NXN1X_NPU_API int64_t NXN1X_NPU_CALL NXN1XNpuNowNs(void);

// Returns immediately after starting the initialization worker. READY only means
// both HTP sessions are ready; it does not mean a generated frame was displayed.
NXN1X_NPU_API int32_t NXN1X_NPU_CALL NXN1XNpuCreate(const NXN1XNpuConfig* config,
                                                          NXN1XNpu** output);

// Submit/Take/Stats/Reset may run concurrently; input/output buffers remain
// caller-owned for the duration of each call. Destroy requires all callers to
// have stopped, is shutdown-only, and waits for an active Run.
NXN1X_NPU_API void NXN1X_NPU_CALL NXN1XNpuDestroy(NXN1XNpu* context);

// Invalidate queued/completed output without waiting for active inference.
// Caller must use the returned epoch for all later submissions.
NXN1X_NPU_API uint64_t NXN1X_NPU_CALL NXN1XNpuReset(NXN1XNpu* context);

// Call only after GPU capture completion + mapped-memory invalidation. Successful
// submit copies both inputs during this call. No caller pointer is retained.
// BUSY/WARMING/UNAVAILABLE: keep presenting originals, never wait for capacity.
NXN1X_NPU_API int32_t NXN1X_NPU_CALL NXN1XNpuSubmit(NXN1XNpu* context,
                                                          const NXN1XNpuPair* pair);

// Copies RGBA8 (alpha 255) into caller-owned storage and consumes one completion.
// EMPTY/BUSY: no output. Caller must still reject stale source IDs before upload,
// and pin its upload buffer until the GPU has finished reading it.
NXN1X_NPU_API int32_t NXN1X_NPU_CALL NXN1XNpuTake(NXN1XNpu* context,
                                                        NXN1XNpuOutput* output, uint8_t* rgba,
                                                        uint64_t data_bytes, uint32_t row_bytes);

NXN1X_NPU_API int32_t NXN1X_NPU_CALL NXN1XNpuGetStats(NXN1XNpu* context,
                                                            NXN1XNpuStats* output);

#ifdef __cplusplus
}
#endif
