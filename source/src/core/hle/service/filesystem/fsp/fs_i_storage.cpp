// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstring>
#include "core/file_sys/errors.h"
#include "core/hle/service/cmif_serialization.h"
#include "core/hle/service/filesystem/fsp/fs_i_storage.h"
#include "core/s49_probe.h"

namespace Service::FileSystem {

IStorage::IStorage(Core::System& system_, FileSys::VirtualFile backend_)
    : ServiceFramework{system_, "IStorage"}, backend(std::move(backend_)) {
    static const FunctionInfo functions[] = {
        {0, D<&IStorage::Read>, "Read"},
        {1, nullptr, "Write"},
        {2, nullptr, "Flush"},
        {3, nullptr, "SetSize"},
        {4, D<&IStorage::GetSize>, "GetSize"},
        {5, nullptr, "OperateRange"},
    };
    RegisterHandlers(functions);
}

Result IStorage::Read(
    OutBuffer<BufferAttr_HipcMapAlias | BufferAttr_HipcMapTransferAllowsNonSecure> out_bytes,
    s64 offset, s64 length) {
    LOG_DEBUG(Service_FS, "called, offset={:#x}, length={}", offset, length);

    R_UNLESS(length >= 0, FileSys::ResultInvalidSize);
    R_UNLESS(offset >= 0, FileSys::ResultInvalidOffset);

    // Read the data from the Storage backend
    const uint64_t s49_t0 = S49::On() ? S49::Now() : 0;   // s49: fs read time and size
    R_UNLESS(static_cast<u64>(length) <= out_bytes.size(), FileSys::ResultInvalidSize);
    const auto bytes_read = backend->Read(out_bytes.data(), length, offset);
    if (bytes_read >= 12) {
        // Movie archives can be decoded in guest code without opening a host decoder session.
        const auto* data = out_bytes.data();
        const char* container = nullptr;
        if (!std::memcmp(data, "\x1a\x45\xdf\xa3", 4)) container = "EBML";
        else if (!std::memcmp(data, "CRID", 4)) container = "CRI USM";
        else if (!std::memcmp(data + 4, "ftyp", 4)) container = "ISO BMFF";
        else if (!std::memcmp(data, "DKIF", 4)) container = "IVF";
        else if (!std::memcmp(data, "KB2", 3) || !std::memcmp(data, "BIK", 3)) container = "Bink";
        if (container) {
            static std::atomic_uint reports{};
            if (reports.fetch_add(1, std::memory_order_relaxed) < 64)
                LOG_INFO(Service_FS, "Movie container header read: {} storage_offset={:#x} bytes={}", container, offset, bytes_read);
        }
    }
    if (s49_t0)
        S49::FsDone(false, static_cast<uint64_t>(length), s49_t0);

    R_SUCCEED();
}

Result IStorage::GetSize(Out<u64> out_size) {
    *out_size = backend->GetSize();

    LOG_DEBUG(Service_FS, "called, size={}", *out_size);

    R_SUCCEED();
}

} // namespace Service::FileSystem
