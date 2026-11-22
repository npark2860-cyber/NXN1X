// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <optional>
#include "common/common_funcs.h"
#include "common/common_types.h"
#include "common/virtual_buffer.h"

namespace Common {

enum class MemoryPermission : u32 {
    Read = 1 << 0,
    Write = 1 << 1,
    ReadWrite = Read | Write,
    Execute = 1 << 2,
};
DECLARE_ENUM_FLAG_OPERATORS(MemoryPermission)

// vfix (Windows): called with [address, address+length) after HostMemory itself makes guest pages writable (a partial unmap's remainders mapped back at
// PAGE_READWRITE); the core sets it to tell the NVN layer's write watch (memory.cpp)
void SetHostWritableHook(void (*fn)(uintptr_t address, size_t length));
// nceresume R9 (Windows): the NCE handler's resume check (an EXCEPTION_POINTERS*), run before HostMemory's own S36 handler resumes a thread
void SetResumeCheckHook(void (*fn)(void* exception_pointers));

/**
 * A low level linear memory buffer, which supports multiple mappings
 * Its purpose is to rebuild a given sparse memory layout, including mirrors.
 */
class HostMemory {
public:
    explicit HostMemory(size_t backing_size_, size_t virtual_size_);
    ~HostMemory();

    /**
     * Copy constructors. They shall return a copy of the buffer without the mappings.
     * TODO: Implement them with COW if needed.
     */
    HostMemory(const HostMemory& other) = delete;
    HostMemory& operator=(const HostMemory& other) = delete;

    /**
     * Move constructors. They will move the buffer and the mappings to the new object.
     */
    HostMemory(HostMemory&& other) noexcept;
    HostMemory& operator=(HostMemory&& other) noexcept;

#ifdef _WIN32
    /**
     * Owns a temporary MEM_PRIVATE replacement for one exact direct-mapped HostMemory range.
     * Destruction restores the original section-backed mapping and copies private writes back to
     * the HostMemory backing store. This is intentionally scoped to Windows direct mappings.
     */
    class PrivateMappingLease {
    public:
        ~PrivateMappingLease();

        PrivateMappingLease(const PrivateMappingLease&) = delete;
        PrivateMappingLease& operator=(const PrivateMappingLease&) = delete;
        PrivateMappingLease(PrivateMappingLease&& other) noexcept;
        PrivateMappingLease& operator=(PrivateMappingLease&& other) noexcept;

        // live_begin: lowest live address (the guest SP). When it lies inside the lease only
        // [page(live_begin) - LiveMargin, end) is copied back; 0 copies the whole lease.
        [[nodiscard]] bool Restore(u64 live_begin = 0) noexcept;
        [[nodiscard]] bool ContainsAddress(u64 address) const noexcept;
        [[nodiscard]] bool SyncToBacking() noexcept;
        [[nodiscard]] bool SyncFromBacking() noexcept;

    private:
        friend class HostMemory;

        PrivateMappingLease(HostMemory* owner_, u8* virtual_address_, size_t host_offset_,
                            size_t length_, MemoryPermission perms_) noexcept;

        HostMemory* owner{};
        u8* virtual_address{};
        size_t host_offset{};
        size_t length{};
        MemoryPermission perms{};
        bool active{};
    };

    [[nodiscard]] std::optional<PrivateMappingLease> AcquireDirectMappedPrivateLease(
        void* virtual_address, size_t host_offset, size_t length, MemoryPermission perms,
        u64 live_begin = 0);
#endif

    void Map(size_t virtual_offset, size_t host_offset, size_t length, MemoryPermission perms,
             bool separate_heap);

    void Unmap(size_t virtual_offset, size_t length, bool separate_heap);

    void Protect(size_t virtual_offset, size_t length, MemoryPermission perms);

    void EnableDirectMappedAddress();

    void ClearBackingRegion(size_t physical_offset, size_t length, u32 fill_value);

    // nx2 port: Lichii marks cold guest memory here (MADV_COLD / page-out). NXN1X keeps its own
    // backing (Windows NCE); this is a hint only, so it does nothing here.
    void HintCold([[maybe_unused]] size_t physical_offset, [[maybe_unused]] size_t length,
                  [[maybe_unused]] std::optional<size_t> virtual_offset = std::nullopt) const {}

    [[nodiscard]] u8* BackingBasePointer() noexcept {
        return backing_base;
    }
    [[nodiscard]] const u8* BackingBasePointer() const noexcept {
        return backing_base;
    }

    [[nodiscard]] u8* VirtualBasePointer() noexcept {
        return virtual_base;
    }
    [[nodiscard]] const u8* VirtualBasePointer() const noexcept {
        return virtual_base;
    }

    // s28 (NVN path): a new view of the backing memory, the runs [offsets[i], offsets[i] + lengths[i]) side by side at one fresh address range, or null.
    // Windows only; cached until backing destruction. The NVN layer imports a memory pool there when the game's range is not one run of the backing (it crosses two views)
    [[nodiscard]] u8* MapBackingRuns(const size_t* offsets, const size_t* lengths, size_t count);

    bool IsInVirtualRange(void* address) const noexcept {
        return address >= virtual_base && address < virtual_base + virtual_size;
    }

private:
#ifdef _WIN32
    void UnmapForPrivateLease(size_t virtual_offset, size_t length);
#endif

    size_t backing_size{};
    size_t virtual_size{};

#if !(defined(__OPENORBIS__) || defined(__managarm__))
    // Low level handler for the platform dependent memory routines
    class Impl;
    std::unique_ptr<Impl> impl;
#endif
    u8* backing_base{};
    u8* virtual_base{};
    size_t virtual_base_offset{};
    bool direct_mapped_address{};
    // Windows requires it for kernels whom lack proper support for some functions!
    std::optional<Common::VirtualBuffer<u8>> fallback_buffer;
};

} // namespace Common
