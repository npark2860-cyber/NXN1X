// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <stdint.h>

// No pointer is a content identity. This classifier makes no guest-memory
// publication and never infers whole-image coverage from an attachment write.
enum nvni_source { NVNI_KEEP, NVNI_ARCHIVE, NVNI_LEGACY_GUEST, NVNI_SAVE_REQUIRED };
enum nvni_block { NVNI_PIN=1, NVNI_LAYOUT=2, NVNI_VERSION=4, NVNI_IDENTITY=8 };
typedef struct nvni_state {
    uint64_t image, archive_image, version, archive_version;
    uint64_t last_use, cpu_done;
    unsigned shared_keys, external_reader, transfer_pending;
    unsigned supported_layout, archive_complete, legacy_guest;
} nvni_state;
typedef struct nvni_decision { unsigned source, blocked; } nvni_decision;
static inline nvni_decision nvni_eligibility(nvni_state s) {
    if (s.shared_keys || s.external_reader || s.transfer_pending || s.last_use>s.cpu_done)
        return (nvni_decision){NVNI_KEEP,NVNI_PIN};
    // Preserve the existing packaged reload choice, even if an old clean file
    // also exists: subsequent lookup must still take the current guest source.
    if (s.legacy_guest) return (nvni_decision){NVNI_LEGACY_GUEST,0};
    if (s.archive_complete) {
        if (!s.image || s.image!=s.archive_image) return (nvni_decision){NVNI_KEEP,NVNI_IDENTITY};
        if (s.version==s.archive_version && s.version!=UINT64_MAX)
            return (nvni_decision){NVNI_ARCHIVE,0};
    }
    // Existing packaged reload route only. F2 Unknown records cannot broaden
    // this class or permit writeback into unexcluded NCE-writable memory.
    return s.supported_layout ? (nvni_decision){NVNI_SAVE_REQUIRED,NVNI_VERSION}
                              : (nvni_decision){NVNI_KEEP,NVNI_LAYOUT};
}
