// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifdef _WIN32

#include <iterator>
#include <ankerl/unordered_dense.h>
#include <boost/icl/separate_interval_set.hpp>
#include <boost/icl/interval_set.hpp>
#include <windows.h>
#include <atomic>
#include <cstdlib>
#include "common/dynamic_library.h"

#else // ^^^ Windows ^^^ vvv POSIX vvv

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <boost/icl/interval_set.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "common/scope_exit.h"

#if defined(__linux__)
#include <sys/random.h>
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/random.h>
#include <mach/vm_map.h>
#include <mach/mach.h>
#elif defined(__FreeBSD__)
#include <sys/shm.h>
#elif defined(__OPENORBIS__)
#include <orbis/libkernel.h>
#endif

// FreeBSD
#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0
#endif
// Solaris 11 and illumos
#ifndef MAP_ALIGNED_SUPER
#define MAP_ALIGNED_SUPER 0
#endif
// macOS
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

#endif // ^^^ POSIX ^^^

#include <limits>
#include <mutex>
#include <random>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/free_region_manager.h"
#include "common/host_memory.h"
#include <map>
#include <vector>
#include "common/logging.h"

#if defined(__ANDROID__) && __ANDROID_API__ < 30
#include <sys/syscall.h>
#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif
static int memfd_create(const char* name, unsigned int flags) {
    return syscall(__NR_memfd_create, name, flags);
}
#endif

namespace Common {

[[maybe_unused]] constexpr size_t PageAlignment = 0x1000;
[[maybe_unused]] constexpr size_t HugePageSize = 0x200000;
constexpr size_t DirectMappedAddressSpaceSize = 1ULL << 39;

#ifdef _WIN32

// Manually imported for MinGW compatibility
#ifndef MEM_RESERVE_PLACEHOLDER
#define MEM_RESERVE_PLACEHOLDER 0x00040000
#endif
#ifndef MEM_REPLACE_PLACEHOLDER
#define MEM_REPLACE_PLACEHOLDER 0x00004000
#endif
#ifndef MEM_COALESCE_PLACEHOLDERS
#define MEM_COALESCE_PLACEHOLDERS 0x00000001
#endif
#ifndef MEM_PRESERVE_PLACEHOLDER
#define MEM_PRESERVE_PLACEHOLDER 0x00000002
#endif

using PFN_CreateFileMapping2 = _Ret_maybenull_ HANDLE(WINAPI*)(
    _In_ HANDLE File, _In_opt_ SECURITY_ATTRIBUTES* SecurityAttributes, _In_ ULONG DesiredAccess,
    _In_ ULONG PageProtection, _In_ ULONG AllocationAttributes, _In_ ULONG64 MaximumSize,
    _In_opt_ PCWSTR Name,
    _Inout_updates_opt_(ParameterCount) MEM_EXTENDED_PARAMETER* ExtendedParameters,
    _In_ ULONG ParameterCount);

using PFN_VirtualAlloc2 = _Ret_maybenull_ PVOID(WINAPI*)(
    _In_opt_ HANDLE Process, _In_opt_ PVOID BaseAddress, _In_ SIZE_T Size,
    _In_ ULONG AllocationType, _In_ ULONG PageProtection,
    _Inout_updates_opt_(ParameterCount) MEM_EXTENDED_PARAMETER* ExtendedParameters,
    _In_ ULONG ParameterCount);

using PFN_MapViewOfFile3 = _Ret_maybenull_ PVOID(WINAPI*)(
    _In_ HANDLE FileMapping, _In_opt_ HANDLE Process, _In_opt_ PVOID BaseAddress,
    _In_ ULONG64 Offset, _In_ SIZE_T ViewSize, _In_ ULONG AllocationType, _In_ ULONG PageProtection,
    _Inout_updates_opt_(ParameterCount) MEM_EXTENDED_PARAMETER* ExtendedParameters,
    _In_ ULONG ParameterCount);

using PFN_UnmapViewOfFile2 = BOOL(WINAPI*)(_In_ HANDLE Process, _In_ PVOID BaseAddress,
                                           _In_ ULONG UnmapFlags);

template <typename T>
static void GetFuncAddress(Common::DynamicLibrary& dll, const char* name, T& pfn) {
    if (!dll.GetSymbol(name, &pfn)) {
        LOG_CRITICAL(HW_Memory, "Failed to load {}", name);
        throw std::bad_alloc{};
    }
}

static DWORD GetWindowsProtection(bool read, bool write, bool execute) {
    if (write && !read) {
        UNIMPLEMENTED_MSG("Protection flag combination read={} write={} execute={}", read, write,
                          execute);
        return PAGE_NOACCESS;
    }

    if (execute) {
        if (read && write) {
            return PAGE_EXECUTE_READWRITE;
        }
        if (read) {
            return PAGE_EXECUTE_READ;
        }
        return PAGE_EXECUTE;
    }

    if (read && write) {
        return PAGE_READWRITE;
    }
    if (read) {
        return PAGE_READONLY;
    }
    return PAGE_NOACCESS;
}

// s36 (NXN1X lane A, TotK runs 51/53/54/55, 2026-11-13): UnmapOnePlaceholder's panic region (a partial unmap unmaps the whole view, then maps the
// remainders back) left the remainders unmapped for ~ms; the game's threads (NCE guest code) and the host (nvnBufferMap reading a guest NVNbuffer) faulted
// there during loads and the process died. The region is now a published window, and an access violation inside it waits for the window to close and is
// retried when the page allows the access. DM_S36_REMAP_WAIT=0: off
// vfix: who to tell when HostMemory makes guest pages writable on its own (SetHostWritableHook)
static std::atomic<void (*)(uintptr_t, size_t)> g_host_writable_hook{nullptr};
void SetHostWritableHook(void (*fn)(uintptr_t, size_t)) { g_host_writable_hook.store(fn, std::memory_order_release); }
// nceresume R9: the NCE handler's resume check, run before S36 resumes a thread itself (EXCEPTION_CONTINUE_EXECUTION); it fails fast on a guest
// thread whose SP it cannot place
static std::atomic<void (*)(void*)> g_resume_check_hook{nullptr};
void SetResumeCheckHook(void (*fn)(void*)) { g_resume_check_hook.store(fn, std::memory_order_release); }
namespace {
std::atomic<uintptr_t> s36_lo{0}, s36_hi{0};
std::atomic<uint32_t> s36_open{0};
std::atomic<uint64_t> s36_until{0};   // GetTickCount64 until which the last window's range is still checked
std::atomic<uint64_t> s36_waits{0};
bool S36Allows(const MEMORY_BASIC_INFORMATION& mi, ULONG_PTR access) {   // access: 0 read, 1 write, 8 execute
    if (mi.State != MEM_COMMIT || (mi.Protect & PAGE_GUARD) != 0) {
        return false;
    }
    const DWORD p = mi.Protect & 0xff;
    if (access == 1) {
        return p == PAGE_READWRITE || p == PAGE_WRITECOPY || p == PAGE_EXECUTE_READWRITE || p == PAGE_EXECUTE_WRITECOPY;
    }
    if (access == 8) {
        return p == PAGE_EXECUTE || p == PAGE_EXECUTE_READ || p == PAGE_EXECUTE_READWRITE || p == PAGE_EXECUTE_WRITECOPY;
    }
    return p != 0 && p != PAGE_NOACCESS;
}
LONG CALLBACK S36Handler(EXCEPTION_POINTERS* ep) {
    const EXCEPTION_RECORD* er = ep->ExceptionRecord;
    if (er->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || er->NumberParameters < 2) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const uintptr_t va = er->ExceptionInformation[1];
    if (va < s36_lo.load(std::memory_order_acquire) || va >= s36_hi.load(std::memory_order_acquire)) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    if (!s36_open.load(std::memory_order_acquire) && GetTickCount64() > s36_until.load(std::memory_order_relaxed)) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    for (int n = 0; s36_open.load(std::memory_order_acquire) != 0; ++n) {
        if (n < 64) {
            YieldProcessor();
        } else {
            SwitchToThread();
        }
    }
    MEMORY_BASIC_INFORMATION mi{};
    if (VirtualQuery(reinterpret_cast<LPCVOID>(va), &mi, sizeof(mi)) == 0 || !S36Allows(mi, er->ExceptionInformation[0])) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    s36_waits.fetch_add(1, std::memory_order_relaxed);
    if (const auto check = g_resume_check_hook.load(std::memory_order_acquire); check != nullptr) check(ep);   // nceresume R9 (SetResumeCheckHook)
    return EXCEPTION_CONTINUE_EXECUTION;
}
PVOID s36_handle;   // under placeholder_mutex (Init, then UnmapOnePlaceholder)
// bak84: the reservations whose own map answers nxn1x_host_range_readable (registered when made, removed when gone)
using NXN1XRangeFn = int (*)(void*, uintptr_t, size_t);
static std::mutex g_nxn1x_range_mu;
static void* g_nxn1x_range_self[4];
static NXN1XRangeFn g_nxn1x_range_fn[4];
static void NXN1XRangeRegister(void* self, NXN1XRangeFn fn) {
    std::scoped_lock lock{g_nxn1x_range_mu};
    for (int i = 0; i < 4; i++) if (g_nxn1x_range_self[i] == self) { g_nxn1x_range_self[i] = fn ? self : nullptr; g_nxn1x_range_fn[i] = fn; return; }
    if (!fn) return;
    for (int i = 0; i < 4; i++) if (!g_nxn1x_range_self[i]) { g_nxn1x_range_self[i] = self; g_nxn1x_range_fn[i] = fn; return; }
}
extern "C" int nxn1x_host_range_readable(const void* p, uint64_t n) {   // -1 in no reservation, else that reservation's answer (1 readable, 0 not known)
    std::scoped_lock lock{g_nxn1x_range_mu};
    for (int i = 0; i < 4; i++) if (g_nxn1x_range_self[i]) {
        const int r = g_nxn1x_range_fn[i](g_nxn1x_range_self[i], reinterpret_cast<uintptr_t>(p), static_cast<size_t>(n));
        if (r >= 0) return r;
    }
    return -1;
}
bool s36_on;
void S36Install() {
    static std::once_flag once;
    std::call_once(once, [] {
        const char* e = std::getenv("DM_S36_REMAP_WAIT");
        if (e != nullptr && *e != 0 && std::atoi(e) == 0) {
            LOG_WARNING(HW_Memory, "s36: remap windows not guarded (DM_S36_REMAP_WAIT=0)");
            return;
        }
        s36_on = true;
        s36_handle = AddVectoredExceptionHandler(1, S36Handler);
        if (s36_handle == nullptr) {
            LOG_ERROR(HW_Memory, "s36: the remap window handler could not be added");
        } else {
            LOG_WARNING(HW_Memory, "s36: accesses inside a partial unmap's remap window wait for it and are retried");
        }
    });
}
void S36Open(uintptr_t lo, uintptr_t hi) {
    // first in the chain at each window: NCE adds its handler first (ArmNce::Initialize) after this file's Init, and guest-code faults must meet this one
    // before NCE's (which sends them to the guest thread's host side)
    if (s36_on) {
        if (s36_handle != nullptr) {
            RemoveVectoredExceptionHandler(s36_handle);
        }
        s36_handle = AddVectoredExceptionHandler(1, S36Handler);
    }
    s36_lo.store(lo, std::memory_order_relaxed);
    s36_hi.store(hi, std::memory_order_relaxed);
    s36_open.store(1, std::memory_order_seq_cst);
}
void S36Close() {
    s36_until.store(GetTickCount64() + 1000, std::memory_order_relaxed);
    s36_open.store(0, std::memory_order_seq_cst);
    static uint64_t logged;
    const uint64_t w = s36_waits.load(std::memory_order_relaxed);
    if (w != logged && (w <= 16 || (w & (w - 1)) == 0)) {
        logged = w;
        LOG_WARNING(HW_Memory, "s36: {} accesses met a remap window and were retried", w);
    }
}
} // namespace

static DWORD GetWindowsProtection(MemoryPermission perms) {
    return GetWindowsProtection(True(perms & MemoryPermission::Read),
                                True(perms & MemoryPermission::Write),
                                True(perms & MemoryPermission::Execute));
}

class HostMemory::Impl {
public:
    explicit Impl(size_t backing_size_, size_t virtual_size_)
        : backing_size{backing_size_}
        , virtual_size{virtual_size_}
        , process{GetCurrentProcess()}
        , kernelbase_dll("Kernelbase")
    {}

    bool Init() {
        if (!kernelbase_dll.IsOpen()) {
            LOG_CRITICAL(HW_Memory, "Failed to load Kernelbase.dll");
            return false;
        }
        GetFuncAddress(kernelbase_dll, "CreateFileMapping2", pfn_CreateFileMapping2);
        GetFuncAddress(kernelbase_dll, "VirtualAlloc2", pfn_VirtualAlloc2);
        GetFuncAddress(kernelbase_dll, "MapViewOfFile3", pfn_MapViewOfFile3);
        GetFuncAddress(kernelbase_dll, "UnmapViewOfFile2", pfn_UnmapViewOfFile2);
        S36Install();   // s36

        if (!pfn_CreateFileMapping2 || !pfn_VirtualAlloc2 || !pfn_MapViewOfFile3 || !pfn_UnmapViewOfFile2) {
            LOG_CRITICAL(HW_Memory, "Failed to find functions for virtual allocs");
            return false;
        }

        // Windows mapped views must be created with execute capability up front if they may later
        // transition between writable and executable protections. Keep the backing alias RW; guest
        // views are created RWX-capable and immediately narrowed to the requested protection.
        // SEC_COMMIT charges all of it to the commit limit up front; when the free commit is only
        // just above its size the call fails without growing a system-managed page file (Switch 2:
        // 14 GiB on a 16 GB host). A private commit of the same size does grow the page file, so
        // make one, release it and try again (from Lichii). Only runs after a failure.
        constexpr int BackingAttempts = 5;
        for (int attempt = 1; attempt <= BackingAttempts && !backing_handle; ++attempt) {
            backing_handle = pfn_CreateFileMapping2(
                INVALID_HANDLE_VALUE, nullptr,
                SECTION_MAP_READ | SECTION_MAP_WRITE | SECTION_MAP_EXECUTE,
                PAGE_EXECUTE_READWRITE, SEC_COMMIT, backing_size, nullptr, nullptr, 0);
            if (backing_handle) {
                if (attempt > 1) {
                    LOG_INFO(HW_Memory, "Allocated {} MiB of backing memory on attempt {}",
                             backing_size >> 20, attempt);
                }
                break;
            }
            const DWORD error = GetLastError();
            MEMORYSTATUSEX status{};
            status.dwLength = sizeof(status);
            GlobalMemoryStatusEx(&status);
            LOG_WARNING(HW_Memory,
                        "Failed to allocate {} MiB of backing memory (attempt {}, error {}, {} MiB "
                        "of commit free)",
                        backing_size >> 20, attempt, error, status.ullAvailPageFile >> 20);
            if (void* const probe =
                    VirtualAlloc(nullptr, backing_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)) {
                VirtualFree(probe, 0, MEM_RELEASE);
            }
            if (attempt < BackingAttempts) {
                Sleep(50);
            }
        }
        if (!backing_handle) {
            LOG_CRITICAL(HW_Memory, "Failed to allocate {} MiB of backing memory", backing_size >> 20);
            return false;
        }
        // Allocate a virtual memory for the backing file map as placeholder
        backing_base = static_cast<u8*>(pfn_VirtualAlloc2(process, nullptr, backing_size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0));
        if (!backing_base) {
            Release();
            LOG_CRITICAL(HW_Memory, "Failed to reserve {} MiB of virtual memory", backing_size >> 20);
            return false;
        }
        // Map backing placeholder
        void* const ret = pfn_MapViewOfFile3(backing_handle, process, backing_base, 0, backing_size, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0);
        if (ret != backing_base) {
            Release();
            LOG_CRITICAL(HW_Memory, "Failed to map {} MiB of virtual memory", backing_size >> 20);
            return false;
        }

        // NCE embeds the host fastmem base into the guest 39-bit address space. POSIX explicitly
        // reserves its arena between 36 and 39 bits; Windows must enforce the same invariant rather
        // than accepting the much higher address that VirtualAlloc2 commonly chooses by default.
        MEM_EXTENDED_PARAMETER* virtual_parameters = nullptr;
        ULONG virtual_parameter_count = 0;
#if defined(ARCHITECTURE_arm64) && defined(HAS_NCE)
        MEM_ADDRESS_REQUIREMENTS nce_address_requirements{};
        nce_address_requirements.LowestStartingAddress = reinterpret_cast<PVOID>(1ULL << 36);
        nce_address_requirements.HighestEndingAddress = reinterpret_cast<PVOID>((1ULL << 39) - 1);
        nce_address_requirements.Alignment = HugePageSize;

        MEM_EXTENDED_PARAMETER nce_address_parameter{};
        nce_address_parameter.Type = MemExtendedParameterAddressRequirements;
        nce_address_parameter.Pointer = &nce_address_requirements;
        virtual_parameters = &nce_address_parameter;
        virtual_parameter_count = 1;
#endif

        // Allocate virtual address placeholder.
        // Keep the existing Windows ARM64 NCE placement policy as the first attempt. Some Windows
        // process layouts fragment the [2^36, 2^39) interval such that no 256-GiB contiguous hole
        // remains even though a suitable hole exists below 2^36. In that case only, relax the
        // application-imposed lower bound while preserving the 39-bit ceiling, alignment,
        // reservation size, and single-arena direct-map contract.
        virtual_base = static_cast<u8*>(pfn_VirtualAlloc2(
            process, nullptr, virtual_size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS,
            virtual_parameters, virtual_parameter_count));
#if defined(ARCHITECTURE_arm64) && defined(HAS_NCE)
        DWORD constrained_error = ERROR_SUCCESS;
        if (!virtual_base) {
            constrained_error = GetLastError();
            nce_address_requirements.LowestStartingAddress = nullptr;
            virtual_base = static_cast<u8*>(pfn_VirtualAlloc2(
                process, nullptr, virtual_size, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS, virtual_parameters, virtual_parameter_count));
            if (virtual_base) {
                LOG_WARNING(HW_Memory,
                            "Windows NCE virtual reservation above 64 GiB failed with error {}; "
                            "using relaxed-lower-bound arena at {:#x}",
                            constrained_error, reinterpret_cast<uintptr_t>(virtual_base));
            }
        }
#endif
        if (!virtual_base) {
#if defined(ARCHITECTURE_arm64) && defined(HAS_NCE)
            const DWORD relaxed_error = GetLastError();
            Release();
            LOG_CRITICAL(HW_Memory,
                         "Failed to reserve {} GiB of Windows NCE virtual memory "
                         "(constrained error {}, relaxed error {})",
                         virtual_size >> 30, constrained_error, relaxed_error);
#else
            Release();
            LOG_CRITICAL(HW_Memory, "Failed to reserve {} GiB of virtual memory", virtual_size >> 30);
#endif
            return false;
        }
        virtual_map_base = virtual_base;
        NXN1XRangeRegister(this, &Impl::RangeReadableOf);   // bak84
        return true;
    }

    ~Impl() {
        for (const auto& [layout, base] : stitched_views) {
            u8* view = base;
            for (const auto& run : layout) {
                pfn_UnmapViewOfFile2(process, view, 0);
                view += run.second;
            }
        }
        NXN1XRangeRegister(this, nullptr);   // bak84
        Release();
    }

    void Map(size_t virtual_offset, size_t host_offset, size_t length, MemoryPermission perms) {
        AdjustMap(&virtual_offset, &length);
        // AdjustMap reports ranges outside the direct-mapped reservation as offset 0 / length 0.
        // Never forward that to Split/MapViewOfFile3: a zero-sized view maps the rest of the section.
        if (length == 0) {
            return;
        }

        std::unique_lock lock{placeholder_mutex};
        if (!IsNiechePlaceholder(virtual_offset, length)) {
            Split(virtual_offset, length);
        }
        ASSERT(placeholders.find({virtual_offset, virtual_offset + length}) == placeholders.end());
        TrackPlaceholder(virtual_offset, host_offset, length);

        const DWORD protection = GetWindowsProtection(perms);
        MapView(virtual_offset, host_offset, length, protection);
        if (True(perms & MemoryPermission::Execute) &&
            !FlushInstructionCache(process, virtual_base + virtual_offset, length)) {
            LOG_CRITICAL(HW_Memory, "Failed to flush instruction cache after executable mapping");
        }
    }

    void Unmap(size_t virtual_offset, size_t length, bool allow_coalesce) {
        AdjustMap(&virtual_offset, &length);
        if (length == 0) {
            return;
        }

        std::scoped_lock lock{placeholder_mutex};
        noread_ranges.subtract(boost::icl::interval<size_t>::right_open(virtual_offset, virtual_offset + length));   // bak84: no mapping, no protection left

        // Unmap until there are no more placeholders. Private NCE leases must preserve the exact
        // placeholder boundary produced by UnmapViewOfFile2 so MEM_REPLACE_PLACEHOLDER can replace
        // the same address/size immediately afterwards.
        while (UnmapOnePlaceholder(virtual_offset, length, allow_coalesce)) {
        }
    }

    void Protect(size_t virtual_offset, size_t length, bool read, bool write, bool execute) {
        AdjustMap(&virtual_offset, &length);
        if (length == 0) {
            return;
        }

        const DWORD new_flags = GetWindowsProtection(read, write, execute);
        const size_t virtual_end = virtual_offset + length;

        std::scoped_lock lock{placeholder_mutex};
        if (read) noread_ranges.subtract(boost::icl::interval<size_t>::right_open(virtual_offset, virtual_end));   // bak84
        else noread_ranges.add(boost::icl::interval<size_t>::right_open(virtual_offset, virtual_end));
        // Look up the first byte, then visit each intersecting mapped view in address order.
        // A multi-view interval lookup can skip the view containing the start of the range.
        auto it = placeholders.lower_bound({virtual_offset, virtual_offset + 1});
        while (it != placeholders.end() && it->lower() < virtual_end) {
            const size_t offset = (std::max)(it->lower(), virtual_offset);
            const size_t protect_length = (std::min)(it->upper(), virtual_end) - offset;
            DWORD old_flags{};
            if (!VirtualProtect(virtual_base + offset, protect_length, new_flags, &old_flags)) {
                LOG_CRITICAL(HW_Memory, "Failed to change virtual memory protect rules");
            } else if (execute &&
                       !FlushInstructionCache(process, virtual_base + offset, protect_length)) {
                LOG_CRITICAL(HW_Memory,
                             "Failed to flush instruction cache after executable protection");
            }
            ++it;
        }
    }

    void EnableDirectMappedAddress() {
        direct_mapped = true;
    }

    const size_t backing_size; ///< Size of the backing memory in bytes
    const size_t virtual_size; ///< Size of the virtual address placeholder in bytes

    u8* backing_base{};
    u8* virtual_base{};

    // s28 (HostMemory::MapBackingRuns): the runs of the backing side by side in a new placeholder, each mapped there as a view of the backing section (the same
    // physical pages as the game's views of them). Owned by this backing until its destruction. Null, with a log line, when a step fails (what was made is undone)
    u8* MapRuns(const size_t* offsets, const size_t* lengths, size_t count) {
        size_t total = 0;
        for (size_t i = 0; i < count; ++i) {
            if (lengths[i] == 0 || ((offsets[i] | lengths[i]) & (PageAlignment - 1)) != 0 || offsets[i] > backing_size ||
                lengths[i] > backing_size - offsets[i]) {
                return nullptr;
            }
            total += lengths[i];
        }
        if (count == 0) {
            return nullptr;
        }
        RunLayout layout;
        for (size_t i = 0; i < count; ++i) layout.emplace_back(offsets[i], lengths[i]);
        std::scoped_lock lock{stitched_mutex};
        auto [entry, inserted] = stitched_views.try_emplace(std::move(layout), nullptr);
        if (!inserted) return entry->second;
        u8* const base = static_cast<u8*>(pfn_VirtualAlloc2(process, nullptr, total, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                                             PAGE_NOACCESS, nullptr, 0));
        if (base == nullptr) {
            stitched_views.erase(entry);
            LOG_ERROR(HW_Memory, "s28: no placeholder of {:#x} bytes for {} runs of the backing (error {})", total, count, GetLastError());
            return nullptr;
        }
        size_t at = 0;
        size_t mapped = 0;
        bool split = false;
        const char* step = nullptr;
        DWORD error = 0;
        for (; mapped < count; ++mapped) {
            split = false;
            if (mapped + 1 < count) {
                if (!VirtualFreeEx(process, base + at, lengths[mapped], MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
                    step = "splitting";
                    error = GetLastError();
                    break;
                }
                split = true;
            }
            if (pfn_MapViewOfFile3(backing_handle, process, base + at, offsets[mapped], lengths[mapped], MEM_REPLACE_PLACEHOLDER,
                                   PAGE_READWRITE, nullptr, 0) == nullptr) {
                step = "mapping";
                error = GetLastError();
                break;
            }
            at += lengths[mapped];
        }
        if (step == nullptr) {
            entry->second = base;
            if (stitched_views.size() == 65 || stitched_views.size() % 256 == 0)
                LOG_INFO(HW_Memory, "s28: {} active zero-copy pool layouts", stitched_views.size());
            return base;
        }
        stitched_views.erase(entry);
        // Undone: the views made, then the placeholders left (the run that failed, if split off, and the rest)
        size_t pos = 0;
        for (size_t i = 0; i < mapped; ++i) {
            pfn_UnmapViewOfFile2(process, base + pos, 0);
            pos += lengths[i];
        }
        if (split) {
            VirtualFreeEx(process, base + pos, 0, MEM_RELEASE);
            pos += lengths[mapped];
        }
        if (pos < total) {
            VirtualFreeEx(process, base + pos, 0, MEM_RELEASE);
        }
        LOG_ERROR(HW_Memory, "s28: {} run {} of {} ({:#x} +{:#x} of the backing) failed (error {})", step, mapped, count, offsets[mapped],
                  lengths[mapped], error);
        return nullptr;
    }

private:
    /// Release all resources in the object
    void Release() {
        if (!placeholders.empty()) {
            for (const auto& placeholder : placeholders) {
                if (!pfn_UnmapViewOfFile2(process, virtual_base + placeholder.lower(),
                                          MEM_PRESERVE_PLACEHOLDER)) {
                    LOG_CRITICAL(HW_Memory, "Failed to unmap virtual memory placeholder");
                }
            }
            Coalesce(0, virtual_size);
        }
        if (virtual_base) {
            if (!VirtualFree(virtual_base, 0, MEM_RELEASE)) {
                LOG_CRITICAL(HW_Memory, "Failed to free virtual memory");
            }
        }
        if (backing_base) {
            if (!pfn_UnmapViewOfFile2(process, backing_base, MEM_PRESERVE_PLACEHOLDER)) {
                LOG_CRITICAL(HW_Memory, "Failed to unmap backing memory placeholder");
            }
            if (!VirtualFreeEx(process, backing_base, 0, MEM_RELEASE)) {
                LOG_CRITICAL(HW_Memory, "Failed to free backing memory");
            }
        }
        if (!CloseHandle(backing_handle)) {
            LOG_CRITICAL(HW_Memory, "Failed to free backing memory file handle");
        }
    }

    void AdjustMap(size_t* virtual_offset, size_t* length) const {
        if (!direct_mapped) {
            return;
        }

        const size_t intended_start = *virtual_offset;
        const size_t intended_end = intended_start + *length;
        const size_t address_space_start = reinterpret_cast<size_t>(virtual_map_base);
        const size_t address_space_end = address_space_start + virtual_size;

        if (intended_end <= address_space_start || intended_start >= address_space_end) {
            *virtual_offset = 0;
            *length = 0;
            return;
        }

        const size_t mapped_start = (std::max)(intended_start, address_space_start);
        const size_t mapped_end = (std::min)(intended_end, address_space_end);
        *virtual_offset = mapped_start - address_space_start;
        *length = mapped_end - mapped_start;
    }

    /// Unmap one placeholder in the given range (partial unmaps are supported)
    /// Return true when there are no more placeholders to unmap
    bool UnmapOnePlaceholder(size_t virtual_offset, size_t length, bool allow_coalesce) {
        const auto it = placeholders.find({virtual_offset, virtual_offset + length});
        const auto begin = placeholders.begin();
        const auto end = placeholders.end();
        if (it == end) {
            return false;
        }
        const size_t placeholder_begin = it->lower();
        const size_t placeholder_end = it->upper();
        const size_t unmap_begin = (std::max)(virtual_offset, placeholder_begin);
        const size_t unmap_end = (std::min)(virtual_offset + length, placeholder_end);
        ASSERT(unmap_begin >= placeholder_begin && unmap_begin < placeholder_end);
        ASSERT(unmap_end <= placeholder_end && unmap_end > placeholder_begin);

        const auto host_pointer_it = placeholder_host_pointers.find(placeholder_begin);
        ASSERT(host_pointer_it != placeholder_host_pointers.end());
        const size_t host_offset = host_pointer_it->second;

        const bool split_left = unmap_begin > placeholder_begin;
        const bool split_right = unmap_end < placeholder_end;

        // s36: the remainders are gone until mapped again below: accesses there wait (S36Handler)
        const bool s36_window = split_left || split_right;
        if (s36_window) {
            S36Open(reinterpret_cast<uintptr_t>(virtual_base + placeholder_begin), reinterpret_cast<uintptr_t>(virtual_base + placeholder_end));
        }
        if (!pfn_UnmapViewOfFile2(process, virtual_base + placeholder_begin,
                                  MEM_PRESERVE_PLACEHOLDER)) {
            const DWORD error = GetLastError();
            LOG_CRITICAL(HW_Memory, "Failed to unmap placeholder {:#018x} +{:#x} (error {}, s23)",
                         reinterpret_cast<uintptr_t>(virtual_base + placeholder_begin),
                         placeholder_end - placeholder_begin, error);
        }
        // If we have to remap memory regions due to partial unmaps, we are in a data race as
        // Windows doesn't support remapping memory without unmapping first. Avoid adding any extra
        // logic within the panic region described below.

        // Panic region, we are in a data race right now
        if (split_left || split_right) {
            Split(unmap_begin, unmap_end - unmap_begin);
        }
        if (split_left) {
            MapView(placeholder_begin, host_offset, unmap_begin - placeholder_begin);
        }
        if (split_right) {
            MapView(unmap_end, host_offset + unmap_end - placeholder_begin,
                    placeholder_end - unmap_end);
        }
        // End panic region
        if (s36_window) {
            S36Close();
        }
        // vfix (11-20): the remainders came back as new views at PAGE_READWRITE (MapView's default), whatever their pages' protection was before: the
        // NVN write watch's read-only pages among them are writable behind its back (memory.cpp HostProtectWritable has the whole story). It is told
        // here, after the views are back
        if (const auto hook = g_host_writable_hook.load(std::memory_order_acquire); hook != nullptr) {
            if (split_left) hook(reinterpret_cast<uintptr_t>(virtual_base + placeholder_begin), unmap_begin - placeholder_begin);
            if (split_right) hook(reinterpret_cast<uintptr_t>(virtual_base + unmap_end), placeholder_end - unmap_end);
        }

        if (allow_coalesce) {
            size_t coalesce_begin = unmap_begin;
            if (!split_left) {
                // Try to coalesce pages to the left
                coalesce_begin = it == begin ? 0 : std::prev(it)->upper();
                if (coalesce_begin != placeholder_begin) {
                    Coalesce(coalesce_begin, unmap_end - coalesce_begin);
                }
            }
            if (!split_right) {
                // Try to coalesce pages to the right
                const auto next = std::next(it);
                const size_t next_begin = next == end ? virtual_size : next->lower();
                if (placeholder_end != next_begin) {
                    // We can coalesce to the right
                    Coalesce(coalesce_begin, next_begin - coalesce_begin);
                }
            }
        }
        // Remove and reinsert placeholder trackers
        UntrackPlaceholder(it);
        if (split_left) {
            TrackPlaceholder(placeholder_begin, host_offset, unmap_begin - placeholder_begin);
        }
        if (split_right) {
            TrackPlaceholder(unmap_end, host_offset + unmap_end - placeholder_begin,
                             placeholder_end - unmap_end);
        }
        return true;
    }

    void MapView(size_t virtual_offset, size_t host_offset, size_t length,
                 DWORD protection = PAGE_READWRITE) {
        void* const mapped = pfn_MapViewOfFile3(
            backing_handle, process, virtual_base + virtual_offset, host_offset, length,
            MEM_REPLACE_PLACEHOLDER, PAGE_EXECUTE_READWRITE, nullptr, 0);
        if (mapped != virtual_base + virtual_offset) {
            const DWORD error = GetLastError();
            LOG_CRITICAL(HW_Memory, "Failed to map placeholder {:#018x} +{:#x} (error {}, s23)",
                         reinterpret_cast<uintptr_t>(virtual_base + virtual_offset), length, error);
            return;
        }

        DWORD old_flags{};
        if (!VirtualProtect(virtual_base + virtual_offset, length, protection, &old_flags)) {
            LOG_CRITICAL(HW_Memory, "Failed to narrow mapped placeholder protection");
        }
    }

    void Split(size_t virtual_offset, size_t length) {
        auto* const address = virtual_base + virtual_offset;
        MEMORY_BASIC_INFORMATION region{};
        const SIZE_T queried = VirtualQuery(address, &region, sizeof(region));
        MEMORY_BASIC_INFORMATION allocation_region{};
        const auto allocation_base = reinterpret_cast<std::uintptr_t>(region.AllocationBase);
        const SIZE_T allocation_queried =
            queried != 0 ? VirtualQuery(region.AllocationBase, &allocation_region,
                                        sizeof(allocation_region))
                         : 0;
        const auto region_begin = allocation_base;
        const auto target_begin = reinterpret_cast<std::uintptr_t>(address);
        const auto region_end =
            region_begin + static_cast<std::uintptr_t>(allocation_region.RegionSize);
        const auto target_end = target_begin + length;
        if (queried == 0 || allocation_queried == 0 || region.State != MEM_RESERVE ||
            allocation_region.State != MEM_RESERVE || target_begin < region_begin ||
            target_end < target_begin || target_end > region_end) {
            LOG_CRITICAL(HW_Memory,
                         "NCE_V18_SPLIT_RANGE_INVALID addr={:#018x} length={:#x} query={} "
                         "state={:#x} region_base={:#018x} region_size={:#x}",
                         target_begin, length, queried,
                         queried != 0 ? static_cast<unsigned long>(region.State) : 0UL,
                         region_begin, static_cast<size_t>(region.RegionSize));
            return;
        }

        // A private HostMemory lease returns to an exact preserved placeholder before Restore()
        // remaps the section-backed view. Use AllocationBase to identify that placeholder: on
        // Windows ARM64, VirtualQuery's BaseAddress/RegionSize can describe the suffix beginning
        // at the queried page rather than the full placeholder extent.
        if (region_begin == target_begin && static_cast<size_t>(allocation_region.RegionSize) == length) {
            return;
        }

        // MEM_REPLACE_PLACEHOLDER requires an exact placeholder match. Split at both requested
        // boundaries so the requested subrange becomes its own placeholder.
        if (target_begin > region_begin &&
            !VirtualFreeEx(process, reinterpret_cast<LPVOID>(region_begin),
                           target_begin - region_begin, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            const DWORD error = GetLastError();
            LOG_CRITICAL(HW_Memory,
                         "NCE_V18_SPLIT_START_FAIL addr={:#018x} length={:#x} error={} "
                         "region_base={:#018x} region_size={:#x}",
                         target_begin, length, error, region_begin,
                         static_cast<size_t>(region.RegionSize));
            return;
        }

        if (target_end < region_end &&
            !VirtualFreeEx(process, reinterpret_cast<LPVOID>(target_begin), length,
                           MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            const DWORD error = GetLastError();
            LOG_CRITICAL(HW_Memory,
                         "NCE_V18_SPLIT_END_FAIL addr={:#018x} length={:#x} error={} "
                         "region_end={:#018x}",
                         target_begin, length, error, region_end);
        }
    }

    void Coalesce(size_t virtual_offset, size_t length) {
        auto* const address = virtual_base + virtual_offset;
        const auto begin = reinterpret_cast<std::uintptr_t>(address);
        const auto end = begin + length;
        if (end < begin) {
            LOG_CRITICAL(HW_Memory,
                         "NCE_V18_COALESCE_RANGE_OVERFLOW addr={:#018x} length={:#x}", begin,
                         length);
            return;
        }

        // MEM_COALESCE_PLACEHOLDERS is valid only when the complete requested range consists of
        // exact adjacent placeholders. Private NCE leases deliberately occupy holes in that
        // topology, so defer coalescing while any part of the range is committed/non-placeholder.
        auto cursor = begin;
        size_t region_count = 0;
        while (cursor < end) {
            MEMORY_BASIC_INFORMATION region{};
            const SIZE_T queried =
                VirtualQuery(reinterpret_cast<const void*>(cursor), &region, sizeof(region));
            if (queried == 0) {
                LOG_CRITICAL(HW_Memory,
                             "NCE_V18_COALESCE_QUERY_FAIL addr={:#018x} length={:#x} cursor={:#018x} error={}",
                             begin, length, cursor, GetLastError());
                return;
            }

            const auto region_begin = reinterpret_cast<std::uintptr_t>(region.BaseAddress);
            const auto region_end = region_begin + static_cast<std::uintptr_t>(region.RegionSize);
            if (region_end <= region_begin || region_begin != cursor || region_end > end ||
                region.State != MEM_RESERVE) {
                LOG_INFO(HW_Memory,
                         "NCE_V18_COALESCE_DEFER addr={:#018x} length={:#x} cursor={:#018x} "
                         "state={:#x} type={:#x} region_base={:#018x} region_size={:#x}",
                         begin, length, cursor, static_cast<unsigned long>(region.State),
                         static_cast<unsigned long>(region.Type), region_begin,
                         static_cast<size_t>(region.RegionSize));
                return;
            }

            cursor = region_end;
            ++region_count;
        }

        if (region_count <= 1) {
            LOG_INFO(HW_Memory,
                     "NCE_V18_COALESCE_NOOP_SINGLE_PLACEHOLDER addr={:#018x} length={:#x}",
                     begin, length);
            return;
        }

        if (!VirtualFreeEx(process, reinterpret_cast<LPVOID>(address), length,
                           MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)) {
            LOG_CRITICAL(HW_Memory,
                         "NCE_V18_COALESCE_FAIL addr={:#018x} length={:#x} error={} regions={}",
                         begin, length, GetLastError(), region_count);
        }
    }

    void TrackPlaceholder(size_t virtual_offset, size_t host_offset, size_t length) {
        placeholders.insert({virtual_offset, virtual_offset + length});
        placeholder_host_pointers.emplace(virtual_offset, host_offset);
    }

    void UntrackPlaceholder(boost::icl::separate_interval_set<size_t>::iterator it) {
        placeholder_host_pointers.erase(it->lower());
        placeholders.erase(it);
    }

    /// Return true when a given memory region is a "nieche" and the placeholders don't have to be
    /// split.
    bool IsNiechePlaceholder(size_t virtual_offset, size_t length) const {
        const auto it = placeholders.upper_bound({virtual_offset, virtual_offset + length});
        if (it != placeholders.end() && it->lower() == virtual_offset + length) {
            return it == placeholders.begin() ? virtual_offset == 0
                                              : std::prev(it)->upper() == virtual_offset;
        }
        return false;
    }

    using RunLayout = std::vector<std::pair<size_t, size_t>>;
    std::map<RunLayout, u8*> stitched_views;
    std::mutex stitched_mutex;
    HANDLE process{};        ///< Current process handle
    HANDLE backing_handle{}; ///< File based backing memory

    DynamicLibrary kernelbase_dll;
    PFN_CreateFileMapping2 pfn_CreateFileMapping2{};
    PFN_VirtualAlloc2 pfn_VirtualAlloc2{};
    PFN_MapViewOfFile3 pfn_MapViewOfFile3{};
    PFN_UnmapViewOfFile2 pfn_UnmapViewOfFile2{};

    u8* virtual_map_base{}; ///< Original placeholder reservation base for direct mappings
    bool direct_mapped{};

    std::mutex placeholder_mutex;                                 ///< Mutex for placeholders
    boost::icl::separate_interval_set<size_t> placeholders;       ///< Mapped placeholders
    ankerl::unordered_dense::map<size_t, size_t> placeholder_host_pointers; ///< Placeholder backing offset
    boost::icl::interval_set<size_t> noread_ranges;               ///< bak84: ranges a Protect took reading from (under placeholder_mutex)

public:
    // bak84 (NVN2 s24 guard, host/dm/nxn1x/nvnh_rt.c nvnh_range_readable_hm): whether [p, p + n) is readable, from this map alone: -1 outside the reservation, 1 when
    // mapped views (placeholders) cover all of it and no Protect took reading from any of it, 0 otherwise (the caller asks the OS then: memory outside these views, such as
    // a private lease, is not known here). No system call
    int RangeReadable(uintptr_t p, size_t n) {
        const uintptr_t vb = reinterpret_cast<uintptr_t>(virtual_base);
        if (!virtual_base || !n || p < vb || n > virtual_size || p - vb > virtual_size - n) return -1;
        const size_t lo = p - vb, hi = lo + n;
        std::scoped_lock lock{placeholder_mutex};
        size_t at = lo;
        auto it = placeholders.lower_bound({lo, lo + 1});
        for (; it != placeholders.end() && at < hi; ++it) {
            if (it->lower() > at) return 0;
            if (it->upper() > at) at = it->upper();
        }
        if (at < hi) return 0;
        if (boost::icl::intersects(noread_ranges, boost::icl::interval<size_t>::right_open(lo, hi))) return 0;
        return 1;
    }
    static int RangeReadableOf(void* self, uintptr_t p, size_t n) { return static_cast<Impl*>(self)->RangeReadable(p, n); }
};

#elif defined(__OPENORBIS__) || defined(__managarm__)
// None of the luxuries of POSIX, all of the suffering
// For managarm: see https://github.com/managarm/managarm/issues/1370
#else // ^^^ Windows ^^^ vvv POSIX vvv

#ifdef ARCHITECTURE_arm64

static void* ChooseVirtualBase(size_t virtual_size) {
    constexpr uintptr_t Map39BitSize = (1ULL << 39);
    constexpr uintptr_t Map36BitSize = (1ULL << 36);

    // This is not a cryptographic application, we just want something random.
    std::mt19937_64 rng;

    // We want to ensure we are allocating at an address aligned to the L2 block size.
    // For Qualcomm devices, we must also allocate memory above 36 bits.
    const size_t lower = Map36BitSize / HugePageSize;
    const size_t upper = (Map39BitSize - virtual_size) / HugePageSize;
    const size_t range = upper - lower;

    // Try up to 64 times to allocate memory at random addresses in the range.
    for (int i = 0; i < 64; i++) {
        // Calculate a possible location.
        uintptr_t hint_address = ((rng() % range) + lower) * HugePageSize;

        // Try to map.
        // Note: we may be able to take advantage of MAP_FIXED_NOREPLACE here.
        void* map_pointer =
            mmap(reinterpret_cast<void*>(hint_address), virtual_size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);

        // If we successfully mapped, we're done.
        if (reinterpret_cast<uintptr_t>(map_pointer) == hint_address) {
            return map_pointer;
        }

        // Unmap if necessary, and try again.
        if (map_pointer != MAP_FAILED) {
            munmap(map_pointer, virtual_size);
        }
    }

    return MAP_FAILED;
}

#else

static void* ChooseVirtualBase(size_t virtual_size) {
#if defined(__FreeBSD__) || defined(__DragonFly__) || defined(__OpenBSD__) || defined(__sun__) || defined(__HAIKU__) || defined(__managarm__) || defined(__AIX__)
    void* virtual_base = mmap(nullptr, virtual_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_ALIGNED_SUPER, -1, 0);
    if (virtual_base != MAP_FAILED)
        return virtual_base;
#endif
    return mmap(nullptr, virtual_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
}

#endif

#if defined(__sun__) || defined(__HAIKU__) || defined(__NetBSD__) || defined(__DragonFly__)
/// Most Unices don't have a portable shm_open (AIX, OpenBSD, NetBSD, Solaris 11, OpenIndiana)
/// Portable implementation of shm_open(SHM_ANON, ...) - roughly equivalent but without
/// OS support - may fail sporadically, beware!
static int shm_open_anon(int flags, mode_t mode) {
    char name[16] = "/shm-";
    char *const limit = name + sizeof(name) - 1;
    *limit = '\0';
    char *start = name + strlen(name);
    for (int tries = 0; tries < 4; tries++) {
        struct timespec tv;
        clock_gettime(CLOCK_REALTIME, &tv);
        unsigned long r = (unsigned long)tv.tv_sec + (unsigned long)tv.tv_nsec;
        for (char *fill = start; fill < limit; r /= 8)
            *fill++ = '0' + (r % 8);
        int fd = shm_open(name, flags, mode);
        if (fd != -1) {
            if (shm_unlink(name) == -1) {
                int tmp = errno;
                close(fd);
                errno = tmp;
                return -1;
            }
            return fd;
        }
        if (errno != EEXIST)
            break;
    }
    return -1;
}
#elif defined(__OpenBSD__)
/// Except OpenBSD which explicitly uses shm_mkstemp instead (as a more secure alternative)
static int shm_open_anon(int flags, mode_t mode) {
    char name[16] = "/shm-XXXXXXXXXX";
    int fd;
    if ((fd = shm_mkstemp(name)) == -1)
        return -1;
    if (shm_unlink(name) == -1) {
        int tmp = errno;
        close(fd);
        errno = tmp;
        return -1;
    }
    return fd;
}
#endif

class HostMemory::Impl {
public:
    explicit Impl(size_t backing_size_, size_t virtual_size_)
        : backing_size{backing_size_}
        , virtual_size{virtual_size_}
    {}

    bool Init() {
        long page_size = sysconf(_SC_PAGESIZE);
        ASSERT_MSG(page_size == 0x1000, "page size {:#x} is incompatible with 4K paging", page_size);
        // Backing memory initialization
#if defined(__sun__) || defined(__HAIKU__) || defined(__NetBSD__) || defined(__DragonFly__)
        fd = shm_open_anon(O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
#elif defined(__OpenBSD__)
        fd = shm_open_anon(O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
#elif defined(__FreeBSD__)
        fd = shm_open(SHM_ANON, O_RDWR, 0600);
#elif defined(__APPLE__) || defined(__managarm__)
        // macOS doesn't have memfd_create, use anonymous temporary file
        char template_path[] = "/tmp/eden_mem_XXXXXX";
        fd = mkstemp(template_path);
        if (fd >= 0) {
            unlink(template_path);
        }
#else
        fd = memfd_create("HostMemory", 0);
#endif
        bool use_anon = false;
        if (fd <= 0) {
            LOG_WARNING(Common_Memory, "memfd_create: {}", strerror(errno));
            use_anon = true;
        }
        if (!use_anon) {
            // Defined to extend the file with zeros
            int ret = ftruncate(fd, backing_size);
            if (ret != 0) {
                LOG_WARNING(Common_Memory, "ftruncate: {} (likely out-of-emory)", strerror(errno));
                use_anon = true;
            }
        }
        if (use_anon) {
            LOG_WARNING(Common_Memory, "Using private mappings instead of shared ones");
            backing_base = static_cast<u8*>(mmap(nullptr, backing_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0));
            if (fd > 0) {
                fd = -1;
                close(fd);
            }
        } else {
            backing_base = static_cast<u8*>(mmap(nullptr, backing_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        }
        if (backing_base == MAP_FAILED) {
            LOG_CRITICAL(HW_Memory, "mmap failed: {}", strerror(errno));
            return false;
        }

        // Virtual memory initialization
        virtual_base = virtual_map_base = static_cast<u8*>(ChooseVirtualBase(virtual_size));
        if (virtual_base == MAP_FAILED) {
            LOG_CRITICAL(HW_Memory, "mmap failed: {}", strerror(errno));
            return false;
        }
#if defined(__linux__)
        madvise(virtual_base, virtual_size, MADV_HUGEPAGE);
#endif
        free_manager.SetAddressSpace(virtual_base, virtual_size);
        return true;
    }

    ~Impl() {
        Release();
    }

    void Map(size_t virtual_offset, size_t host_offset, size_t length, MemoryPermission perms) {
        // Intersect the range with our address space.
        AdjustMap(&virtual_offset, &length);

        // We are removing a placeholder.
        free_manager.AllocateBlock(virtual_base + virtual_offset, length);

        // Deduce mapping protection flags.
        int prot_flags = PROT_NONE;
        if (True(perms & MemoryPermission::Read))
            prot_flags |= PROT_READ;
        if (True(perms & MemoryPermission::Write))
            prot_flags |= PROT_WRITE;
#ifdef ARCHITECTURE_arm64
        if (True(perms & MemoryPermission::Execute))
            prot_flags |= PROT_EXEC;
#endif
        int flags = (fd >= 0 ? MAP_SHARED : MAP_PRIVATE) | MAP_FIXED;
        void* ret = mmap(virtual_base + virtual_offset, length, prot_flags, flags, fd, host_offset);
        ASSERT_MSG(ret != MAP_FAILED, "mmap: {} {}", strerror(errno), fd);
    }

    // Same signature as the Windows implementation (non-Windows builds compile-check only).
    void Unmap(size_t virtual_offset, size_t length, [[maybe_unused]] bool allow_coalesce) {
        Unmap(virtual_offset, length);
    }

    void Unmap(size_t virtual_offset, size_t length) {
        // The method name is wrong. We're still talking about the virtual range.
        // We don't want to unmap, we want to reserve this memory.

        // Intersect the range with our address space.
        AdjustMap(&virtual_offset, &length);

        // Merge with any adjacent placeholder mappings.
        auto [merged_pointer, merged_size] =
            free_manager.FreeBlock(virtual_base + virtual_offset, length);

        void* ret = mmap(merged_pointer, merged_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
        ASSERT_MSG(ret != MAP_FAILED, "mmap: {}", strerror(errno));
    }

    void Protect(size_t virtual_offset, size_t length, bool read, bool write, bool execute) {
        // Intersect the range with our address space.
        AdjustMap(&virtual_offset, &length);

        int flags = PROT_NONE;
        if (read) {
            flags |= PROT_READ;
        }
        if (write) {
            flags |= PROT_WRITE;
        }
#ifdef HAS_NCE
        if (execute) {
            flags |= PROT_EXEC;
        }
#endif
        int ret = mprotect(virtual_base + virtual_offset, length, flags);
        ASSERT_MSG(ret == 0, "mprotect failed: {}", strerror(errno));
    }

    void EnableDirectMappedAddress() {
        virtual_base = nullptr;
    }

    const size_t backing_size; ///< Size of the backing memory in bytes
    const size_t virtual_size; ///< Size of the virtual address placeholder in bytes

    u8* backing_base{reinterpret_cast<u8*>(MAP_FAILED)};
    u8* virtual_base{reinterpret_cast<u8*>(MAP_FAILED)};
    u8* virtual_map_base{reinterpret_cast<u8*>(MAP_FAILED)};

private:
    /// Release all resources in the object
    void Release() {
        if (virtual_map_base != MAP_FAILED) {
            int ret = munmap(virtual_map_base, virtual_size);
            ASSERT_MSG(ret == 0, "munmap failed: {}", strerror(errno));
        }

        if (backing_base != MAP_FAILED) {
            int ret = munmap(backing_base, backing_size);
            ASSERT_MSG(ret == 0, "munmap failed: {}", strerror(errno));
        }

        if (fd != -1) {
            int ret = close(fd);
            ASSERT_MSG(ret == 0, "close failed: {}", strerror(errno));
        }
    }

    void AdjustMap(size_t* virtual_offset, size_t* length) {
        if (virtual_base != nullptr) {
            return;
        }

        // If we are direct mapped, we want to make sure we are operating on a region
        // that is in range of our virtual mapping.
        const size_t intended_start = *virtual_offset;
        const size_t intended_end = intended_start + *length;
        const size_t address_space_start = reinterpret_cast<size_t>(virtual_map_base);
        const size_t address_space_end = address_space_start + virtual_size;

        if (intended_end <= address_space_start || intended_start >= address_space_end) {
            *virtual_offset = 0;
            *length = 0;
        } else {
            const size_t mapped_start = (std::max)(intended_start, address_space_start);
            const size_t mapped_end = (std::min)(intended_end, address_space_end);
            *virtual_offset = mapped_start;
            *length = mapped_end - mapped_start;
        }
    }

    int fd{-1}; // memfd file descriptor, -1 is the error value of memfd_create
    FreeRegionManager free_manager{};
};

#endif // ^^^ POSIX ^^^

HostMemory::HostMemory(size_t backing_size_, size_t virtual_size_)
    : backing_size(backing_size_)
    , virtual_size(virtual_size_)
{
#if defined(__OPENORBIS__) || defined(__managarm__)
    LOG_WARNING(HW_Memory, "Platform doesn't support fastmem");
    fallback_buffer.emplace(backing_size);
    backing_base = fallback_buffer->data();
    virtual_base = nullptr;
#else
    // Try to allocate a fastmem arena.
    // The implementation will fail with std::bad_alloc on errors.
    impl = std::make_unique<HostMemory::Impl>(AlignUp(backing_size, PageAlignment), AlignUp(virtual_size, PageAlignment) + HugePageSize);
    if (impl->Init()) {
        backing_base = impl->backing_base;
        virtual_base = impl->virtual_base;
        if (virtual_base) {
            // Ensure the virtual base is aligned to the L2 block size.
            virtual_base = reinterpret_cast<u8*>(Common::AlignUp(uintptr_t(virtual_base), HugePageSize));
            virtual_base_offset = virtual_base - impl->virtual_base;
        }
    } else {
        impl.reset();
        LOG_WARNING(HW_Memory, "Platform can support fastmem, but can't create it");
        fallback_buffer.emplace(backing_size);
        backing_base = fallback_buffer->data();
        virtual_base = nullptr;
    }
#endif
}

HostMemory::~HostMemory() = default;

HostMemory::HostMemory(HostMemory&&) noexcept = default;

HostMemory& HostMemory::operator=(HostMemory&&) noexcept = default;

u8* HostMemory::MapBackingRuns(const size_t* offsets, const size_t* lengths, size_t count) {
#ifdef _WIN32
    return impl ? impl->MapRuns(offsets, lengths, count) : nullptr;
#else
    (void)offsets;
    (void)lengths;
    (void)count;
    return nullptr;
#endif
}

// NVN2 (the NVN host layer's s24 guard, host/dm/windows/gfx_vk_tex.c s24_readable): moved before and after every change of the guest's mappings that can make guest
// memory unreadable (Map, Unmap, UnmapForPrivateLease, a Protect without read), so a "readable" answer cached with the generation it was taken at holds until it moves.
// The dirty tracker's RW/RO protections (nvn_dirty, VirtualProtect directly) do not go through here and never make memory unreadable
static std::atomic<uint64_t> g_nxn1x_map_gen{1};
extern "C" uint64_t nxn1x_host_map_gen(void) { return g_nxn1x_map_gen.load(std::memory_order_acquire); }
static void MapGenBump() { g_nxn1x_map_gen.fetch_add(1, std::memory_order_acq_rel); }
#if !defined(_WIN32)
extern "C" int nxn1x_host_range_readable(const void* p, uint64_t n) { (void)p; (void)n; return -1; }   // bak84: the host map answers on Windows only
#endif
// bak82 (11-16, MKW 14:17 race: the s24 guard's cached answers were dropped at every move of the generation, several a present: about 10k VirtualQuery calls of about
// 2.4 ms per 30 s, 21-25 s of every 30). Each change that moves the generation (two moves each) is also recorded with its host address range in a ring, so that
// gfx_vk_tex.c s24_readable keeps a cached "readable" answer when no change since touched its range. An event is recorded before its first move (a reader that sees the
// move sees the event); busy counts the changes under way, and an answer taken while one is under way is not kept by range (nxn1x_map_ev_mark). Ranges are supersets
// (a direct-mapped address is not clipped to the reservation). Entries are read seqlock style (seq written last, read before and after)
namespace {
constexpr uint64_t kMapEvN = 1024;
struct MapEv { std::atomic<uint64_t> lo{0}, hi{0}, seq{0}; };
MapEv g_map_ev[kMapEvN];
std::atomic<uint64_t> g_map_ev_n{0};
std::atomic<int64_t> g_map_ev_busy{0};
} // namespace
extern "C" void nxn1x_map_ev_begin(const void* a, uint64_t n) {
    g_map_ev_busy.fetch_add(1);
    const uint64_t i = g_map_ev_n.fetch_add(1);
    MapEv& e = g_map_ev[i % kMapEvN];
    const uint64_t lo = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(a));
    e.seq.store(0);
    e.lo.store(lo);
    e.hi.store(n > UINT64_MAX - lo ? UINT64_MAX : lo + n);
    e.seq.store(i + 1);
}
extern "C" void nxn1x_map_ev_end(void) { g_map_ev_busy.fetch_sub(1); }
// The ring position for an answer about to be taken; UINT64_MAX while a change is under way (the answer is then never kept by range)
extern "C" uint64_t nxn1x_map_ev_mark(void) { const uint64_t e = g_map_ev_n.load(); return g_map_ev_busy.load() ? UINT64_MAX : e; }
// 1: no event in [from, now) touches [a, a + n), *now set; 0: one does, or one is not written yet; -1: the ring no longer holds them all; -2: from was taken while busy
extern "C" int nxn1x_map_ev_clear(uint64_t from, const void* a, uint64_t n, uint64_t* now) {
    if (from == UINT64_MAX) return -2;
    const uint64_t to = g_map_ev_n.load();
    if (to < from || to - from > kMapEvN) return -1;
    const uint64_t lo = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(a)), hi = n > UINT64_MAX - lo ? UINT64_MAX : lo + n;
    for (uint64_t j = from; j < to; j++) {
        const MapEv& e = g_map_ev[j % kMapEvN];
        if (e.seq.load() != j + 1) return 0;
        const uint64_t elo = e.lo.load(), ehi = e.hi.load();
        if (e.seq.load() != j + 1) return 0;
        if (elo < hi && lo < ehi) return 0;
    }
    *now = to;
    return 1;
}
// The host address of a guest range as the platform implementation maps it: a direct-mapped caller passes absolute addresses
#define NXN1X_MAP_EV_ADDR(off) (direct_mapped_address ? reinterpret_cast<const void*>(static_cast<uintptr_t>(off)) : static_cast<const void*>(virtual_base + (off)))

void HostMemory::Map(size_t virtual_offset, size_t host_offset, size_t length, MemoryPermission perms, bool separate_heap) {
#if !(defined(__OPENORBIS__) || defined(__managarm__))
    ASSERT(virtual_offset % PageAlignment == 0);
    ASSERT(host_offset % PageAlignment == 0);
    ASSERT(length % PageAlignment == 0);
    ASSERT(length <= std::numeric_limits<size_t>::max() - virtual_offset);
    ASSERT(length <= std::numeric_limits<size_t>::max() - host_offset);
    ASSERT(direct_mapped_address ? virtual_offset + length <= DirectMappedAddressSpaceSize
                                 : virtual_offset + length <= virtual_size);
    ASSERT(host_offset + length <= backing_size);
    if (length == 0 || !virtual_base || !impl) {
        return;
    }
    nxn1x_map_ev_begin(NXN1X_MAP_EV_ADDR(virtual_offset), length);   // bak82
    MapGenBump(); impl->Map(virtual_offset + virtual_base_offset, host_offset, length, perms); MapGenBump();
    nxn1x_map_ev_end();
#endif
}

void HostMemory::Unmap(size_t virtual_offset, size_t length, bool separate_heap) {
#if !(defined(__OPENORBIS__) || defined(__managarm__))
    ASSERT(virtual_offset % PageAlignment == 0);
    ASSERT(length % PageAlignment == 0);
    ASSERT(length <= std::numeric_limits<size_t>::max() - virtual_offset);
    ASSERT(direct_mapped_address ? virtual_offset + length <= DirectMappedAddressSpaceSize
                                 : virtual_offset + length <= virtual_size);
    if (length == 0 || !virtual_base || !impl) {
        return;
    }
    nxn1x_map_ev_begin(NXN1X_MAP_EV_ADDR(virtual_offset), length);   // bak82
    MapGenBump(); impl->Unmap(virtual_offset + virtual_base_offset, length, true); MapGenBump();
    nxn1x_map_ev_end();
#endif
}

#ifdef _WIN32
void HostMemory::UnmapForPrivateLease(size_t virtual_offset, size_t length) {
#if !(defined(__OPENORBIS__) || defined(__managarm__))
    ASSERT(virtual_offset % PageAlignment == 0);
    ASSERT(length % PageAlignment == 0);
    ASSERT(virtual_offset + length <= virtual_size);
    if (length == 0 || !virtual_base || !impl) {
        return;
    }
    nxn1x_map_ev_begin(NXN1X_MAP_EV_ADDR(virtual_offset), length);   // bak82
    MapGenBump(); impl->Unmap(virtual_offset + virtual_base_offset, length, false); MapGenBump();
    nxn1x_map_ev_end();
#endif
}
#endif

void HostMemory::Protect(size_t virtual_offset, size_t length, MemoryPermission perm) {
#if !(defined(__OPENORBIS__) || defined(__managarm__))
    ASSERT(virtual_offset % PageAlignment == 0);
    ASSERT(length % PageAlignment == 0);
    ASSERT(length <= std::numeric_limits<size_t>::max() - virtual_offset);
    ASSERT(direct_mapped_address ? virtual_offset + length <= DirectMappedAddressSpaceSize
                                 : virtual_offset + length <= virtual_size);
    if (length == 0 || !virtual_base || !impl) {
        return;
    }
    const bool read = True(perm & MemoryPermission::Read);
    const bool write = True(perm & MemoryPermission::Write);
    const bool execute = True(perm & MemoryPermission::Execute);
    if (!read) { nxn1x_map_ev_begin(NXN1X_MAP_EV_ADDR(virtual_offset), length); MapGenBump(); }   // (only a protection that takes reading away can make memory unreadable; bak82: recorded)
    impl->Protect(virtual_offset + virtual_base_offset, length, read, write, execute);
    if (!read) { MapGenBump(); nxn1x_map_ev_end(); }
#endif
}

void HostMemory::ClearBackingRegion(size_t physical_offset, size_t length, u32 fill_value) {
    std::memset(backing_base + physical_offset, fill_value, length);
}

void HostMemory::EnableDirectMappedAddress() {
#if !(defined(__OPENORBIS__) || defined(__managarm__))
    if (impl && !direct_mapped_address) {
        impl->EnableDirectMappedAddress();
#ifdef _WIN32
        // Direct-mapped callers pass absolute guest addresses. The Windows implementation
        // converts those back to offsets from the original placeholder reservation itself.
        virtual_base_offset = 0;
#endif
        virtual_size += reinterpret_cast<uintptr_t>(virtual_base);
        direct_mapped_address = true;
    }
#endif
}

} // namespace Common
