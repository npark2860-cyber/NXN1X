// SPDX-License-Identifier: GPL-2.0-or-later

#if defined(NXN1X_NVN_HOST)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cxxabi.h>
#include <exception>
#include <mutex>
#include <string>
#include <typeinfo>

#include <fmt/format.h>
#endif

#include <atomic>
#include <cstring>

#include "common/logging.h"
#include "common/settings.h"
#include "core/nvn/nvn_layer.h"

#if defined(NXN1X_NVN_HOST)
#include "common/host_memory.h"
#include "common/thread.h"   // s46pc
#include "common/cpu_features.h"   // guestclk: Common::g_wall_clock
#include <utility>
#include <vector>
#include "core/memory.h"
#include "core/nvn/nvn_gate.h"
#include "core/nvn/sv1.h"
#include "nvnh_api.h"
#endif

namespace Core::NvnLayer {

namespace {
std::atomic<bool> g_wanted{false};
// Write faults on pages of the layer's pools that the layer did not cause (OnWatchedWrite): how many, and the first address
std::atomic<u64> g_foreign_faults{0};
std::atomic<u64> g_foreign_first{0};
// Faults on the same page again right after "run it again" (OnWatchedWrite), the longest run and where, neighbouring pages released
std::atomic<u64> g_repeat_faults{0};
std::atomic<u64> g_repeat_worst{0};
std::atomic<u64> g_repeat_addr{0};
std::atomic<u64> g_neighbour_releases{0};
// Per thread: the page OnWatchedWrite last let run again, when (QueryPerformanceCounter), and how many times in a row it came back at once
struct WatchRepeat {
    u64 page;
    s64 when;
    u64 count;
};
thread_local WatchRepeat t_repeat{};

#if defined(NXN1X_NVN_HOST)
// s23 (SetGuestMemory): [mem, mem + size) of the game's memory as one range of NXN1X's backing memory, or null when a page of it is unmapped
// or its pages are not one physically contiguous run. The host layer imports a pool chunk there (nvnh_import_alias)
std::atomic<Core::Memory::Memory*> g_guest_memory{nullptr};

// s28 (SetBackingMemory): a range whose pages are runs of the backing that do not follow each other (it crosses two of the views the game mapped: the host
// layer's s13 bridges) gets a view of its own, those runs side by side (HostMemory::MapBackingRuns), and the layer imports that: the same physical pages, so
// the GPU sees what the game writes with no copy. Xenoblade 3 run 32 (11-12): its first bridge was copied, the copy's write watch faulted thousands of times a
// second, and the game froze 25 s later. Views are kept (an import can outlive its pool) and found again for the same range and pages; at most 64, of at most
// 256 MB and 64 runs each
std::atomic<Common::HostMemory*> g_backing{nullptr};
std::atomic<u64> g_content_backing_id{0};
extern "C" uint64_t nxn1x_host_map_gen(void);
extern "C" uint64_t nxn1x_map_ev_mark(void);
extern "C" uint64_t nvn_sparse_gen(void);

// F2: only atomic reads here; the producer also runs on the NCE fault path.
uint64_t ContentEpoch() {
    if (nxn1x_map_ev_mark() == UINT64_MAX || nvnh_content_sparse_busy()) return UINT64_MAX;
    const u64 host = nxn1x_host_map_gen(), sparse = nvn_sparse_gen();
    if (host >= UINT64_MAX - sparse) return UINT64_MAX;
    const u64 epoch = host + sparse;
    if (nxn1x_map_ev_mark() == UINT64_MAX || nvnh_content_sparse_busy()) return UINT64_MAX;
    return epoch;
}

// Called only by the bounded normal-thread observer. No import, allocation or
// guest data read: GetPointerSilent resolves the stable backing view, including
// sparse aliases resolved through the native page table under its own lock.
int ContentResolve(u64 address, u64 length, nvnc_backing* out) {
    auto* memory = g_guest_memory.load(std::memory_order_acquire);
    auto* backing = g_backing.load(std::memory_order_acquire);
    const u64 id = g_content_backing_id.load(std::memory_order_acquire);
    if (!memory || !backing || !id || !length || length > 65536 || address > UINT64_MAX - length) return 0;
    const u64 base = reinterpret_cast<u64>(backing->BackingBasePointer());
    if (!base) return 0;
    u64 first = 0, covered = 0;
    while (covered < length) {
        const u64 a = address + covered;
        const u64 chunk = std::min<u64>(length - covered, 4096 - (a & 4095));
        auto* pointer = memory->GetPointerSilent(a);
        if (!pointer) {
            const u64 physical = nvnh_content_sparse_address(a, chunk);
            if (!physical || physical == a) break;
            pointer = memory->GetPointerSilent(physical);
        }
        const u64 value = reinterpret_cast<u64>(pointer);
        if (!pointer || value < base || (covered && value != first + covered)) break;
        if (!covered) first = value;
        covered += chunk;
    }
    if (!covered || id != g_content_backing_id.load(std::memory_order_acquire)) return 0;
    *out = nvnc_backing{id, first - base, covered};
    return 1;
}
constexpr size_t MaxStitchRuns = 64;
constexpr u64 MaxStitchSize = 256ULL << 20;
void* StitchRuns(Core::Memory::Memory& memory, u64 address, u64 size) {
    Common::HostMemory* const backing = g_backing.load(std::memory_order_acquire);
    if (backing == nullptr || size > MaxStitchSize) {
        return nullptr;
    }
    constexpr u64 Page = 4096;
    const u8* const backing_base = backing->BackingBasePointer();
    std::array<size_t, MaxStitchRuns> offsets{};
    std::array<size_t, MaxStitchRuns> lengths{};
    size_t count = 0;
    for (u64 at = 0; at < size; at += Page) {
        const u8* const pointer = memory.GetPointerSilent(address + at);
        if (pointer == nullptr || backing_base == nullptr || pointer < backing_base) {
            return nullptr;
        }
        const size_t offset = static_cast<size_t>(pointer - backing_base);
        if (count != 0 && offsets[count - 1] + lengths[count - 1] == offset) {
            lengths[count - 1] += Page;
            continue;
        }
        if (count == MaxStitchRuns) {
            return nullptr;
        }
        offsets[count] = offset;
        lengths[count] = Page;
        ++count;
    }
    return backing->MapBackingRuns(offsets.data(), lengths.data(), count);
}
void* ImportAlias(const void* mem, uint64_t size) {
    Core::Memory::Memory* const memory = g_guest_memory.load(std::memory_order_acquire);
    constexpr u64 Page = 4096;
    const u64 address = reinterpret_cast<u64>(mem);
    if (memory == nullptr || size == 0 || ((address | size) & (Page - 1)) != 0) {
        return nullptr;
    }
    u8* const base = memory->GetPointerSilent(address);
    if (base == nullptr) {
        return nullptr;
    }
    for (u64 offset = Page; offset < size; offset += Page) {
        if (memory->GetPointerSilent(address + offset) != base + offset) {
            return StitchRuns(*memory, address, size); // s28
        }
    }
    return base;
}
#endif

#if defined(NXN1X_NVN_HOST)
// The host layer's "[dm] ..." lines (dm_log_line, nvnh_rt.c) into NXN1X's log; [RED] marks its errors
void HostLogLine(const char* line) {
    if (std::strstr(line, "[RED]") != nullptr) {
        LOG_ERROR(Render_Vulkan, "[nvn] {}", line);
    } else {
        LOG_INFO(Render_Vulkan, "[nvn] {}", line);
    }
}

// s46pc: the host layer's replay thread (hle_nvn.c s46, DM_VK_ASYNC > 0) gets the guest-core threads' policy (cpu_manager.cpp RunThread: Critical priority, the
// performance-class CPU sets and no power throttling), and can ask whether it runs on the fastest core class (the "s46pc:" line)
struct CpuClasses {
    std::vector<std::pair<u32, u8>> lp;   // (group << 8 | logical processor index, efficiency class)
    u8 highest = 0;
    bool hybrid = false;
};
const CpuClasses& HostCpuClasses() {
    static const CpuClasses classes = [] {
        CpuClasses c;
        ULONG need = 0;
        GetSystemCpuSetInformation(nullptr, 0, &need, GetCurrentProcess(), 0);
        if (need == 0) return c;
        std::vector<u8> buffer(need);
        auto* const info = reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buffer.data());
        if (!GetSystemCpuSetInformation(info, need, &need, GetCurrentProcess(), 0)) return c;
        u8 lowest = 0xff;
        for (ULONG off = 0; off < need;) {
            const auto* s = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + off);
            if (s->Type == CpuSetInformation) {
                c.lp.emplace_back(static_cast<u32>(s->CpuSet.Group) << 8 | s->CpuSet.LogicalProcessorIndex, s->CpuSet.EfficiencyClass);
                c.highest = (std::max)(c.highest, s->CpuSet.EfficiencyClass);
                lowest = (std::min)(lowest, s->CpuSet.EfficiencyClass);
            }
            off += s->Size;
        }
        c.hybrid = !c.lp.empty() && lowest != c.highest;
        return c;
    }();
    return classes;
}
int ReplayCpuClass(int* cpu) {
    PROCESSOR_NUMBER pn{};
    GetCurrentProcessorNumberEx(&pn);
    const u32 key = static_cast<u32>(pn.Group) << 8 | pn.Number;
    if (cpu != nullptr) *cpu = static_cast<int>(key);
    const CpuClasses& c = HostCpuClasses();
    for (const auto& [k, cls] : c.lp) {
        if (k == key) return cls == c.highest ? 1 : 0;
    }
    return -1;
}
// s46pc2: the guest-core policy in full (TIME_CRITICAL on the same P-core CPU sets as the 6 guest-core threads) halved OoT with DM_VK_ASYNC=1 in every scene, light
// ones too (11-21 09:18: 22-42 fps where it had 52-60, GPU 0.3 ms, the replay thread busy 2.7 ms a frame): the replay thread now only asks not to be throttled,
// one step above normal. DM_VK_S46_POLICY: 0 nothing (as before s46pc), 1 no power throttling + ABOVE_NORMAL (default), 2 + the P-core CPU sets,
// 3 + TIME_CRITICAL (the s46pc build)
void ReplayThreadPolicy() {
    const char* e = std::getenv("DM_VK_S46_POLICY");
    const int policy = e != nullptr && *e >= '0' && *e <= '3' ? *e - '0' : 1;
    if (policy >= 1) {
        Common::SetCurrentThreadToPerformanceCores();   // the execution-speed QoS: no power throttling
        Common::SetCurrentThreadPriority(policy >= 3 ? Common::ThreadPriority::Critical : Common::ThreadPriority::High);
    }
    if (policy >= 2) {
        Common::SetCurrentThreadToPerformanceCoreSets();
    }
    const CpuClasses& c = HostCpuClasses();
    LOG_INFO(Core, "s46pc: NVN replay thread policy {} (DM_VK_S46_POLICY: 0 none, 1 no power throttling + above normal, 2 + efficiency class {} CPU sets, 3 + time critical){}",
             policy, c.highest, c.hybrid ? "" : "; one core class");
}

// guestclk (11-21, NPARK; SnapRYU2 host/docs/reference/frame-rate.md: Donkey Kong Bananza held at 20 fps): its frame controller (main+0x204c20) measures
// ConvertToTick(nvnDeviceGetTimestampInNanoseconds(report)) minus the frame's start in CPU ticks (main+0x204618), so the GPU's time and the CPU counter must be
// one clock, as on the device. The CPU counter the game reads is the NCE handler's (patcher.cpp WriteCntpctHandler: CNTVCT_EL0 times the guest CNTFRQ factor,
// 31.25 MHz for Switch 2 titles); the NVN host's guest_ns was QueryPerformanceCounter (nvnh_rt.c now_ns, no clock set), another base. Here guest_ns becomes
// that same counter in nanoseconds: report timestamps, nvnDeviceGetCurrentTimestampInNanoseconds and the replay's timestamp clock follow the CPU counter.
// DM_NVN_GUEST_CLOCK=0: QueryPerformanceCounter as before
static uint64_t GuestCounterNs() {
#if defined(HAS_NCE) && defined(__aarch64__)
    u64 c;
    asm volatile("mrs %0, cntvct_el0" : "=r"(c));
    const auto f = Common::g_wall_clock.GetGuestCNTFRQFactor();
    const u64 lo = static_cast<u64>(f), hi = static_cast<u64>(f >> 64);
    const u64 ticks = static_cast<u64>((static_cast<unsigned __int128>(c) * lo) >> 64) + c * hi;   // the NCE handler's umulh + madd
    const u64 frq = Common::g_wall_clock.GetGuestCNTFRQ();
    return frq ? static_cast<u64>(static_cast<unsigned __int128>(ticks) * 1'000'000'000ULL / frq) : 0;
#else
    return 0;
#endif
}
extern "C" uint64_t nxn1x_guest_counter_ns(void) { return GuestCounterNs(); }   // guestclk: hle_nvn.c dkctl_log compares it with now_ns
static void InstallGuestClock() {
#if defined(HAS_NCE) && defined(__aarch64__)
    const char* const e = std::getenv("DM_NVN_GUEST_CLOCK");
    const bool on = !(e && *e == '0') && Common::g_wall_clock.GetGuestCNTFRQ() != 0;   // (on since 11-21 13:00: the check showed the NVN clock 199,745 s ahead of the CPU counter, same rate)
    if (on) nvnh_set_guest_clock(GuestCounterNs);
    LOG_INFO(Core, "guestclk: the NVN host's clock (timestamps, GetCurrentTimestampInNanoseconds) is {} (guest counter {} Hz; DM_NVN_GUEST_CLOCK={})",
             on ? "the guest CPU counter (CNTVCT_EL0 at the guest CNTFRQ)" : "QueryPerformanceCounter", Common::g_wall_clock.GetGuestCNTFRQ(), e && *e ? e : "1 (default)");
#endif
}

void ReportLinkedOnce() {
    static bool reported = false;
    if (reported) {
        return;
    }
    reported = true;
    nvnh_set_log(HostLogLine);
    InstallGuestClock();   // guestclk
    nvnh_replay_thread_hook = ReplayThreadPolicy;   // s46pc
    nvnh_cpu_class_hook = ReplayCpuClass;
    const nvnh_force_t* force{};
    const size_t nforce = nvnh_force_entries(&force);
    LOG_INFO(Core, "NVN host layer linked: {} NVN functions, {} take-overs by symbol (first: {})",
             nvnh_nfunc(), nforce, nforce ? force[0].name : "-");
}

// ---- NXN1X_CXX_THROW_TRACE (diagnostic, off by default) ----
// Logs every C++ throw (libunwind on Windows ARM64 raises it as SEH exception 0x20474343, the first argument is the
// _Unwind_Exception) and every unhandled exception: thread, whether the NVN gate is on, type, what() for std::exception
// types, and the stack as module+RVA from the raise point (symbolize with llvm-symbolizer --relative-address).
// "1" logs to NXN1X's log only. Any other value is also a file path the lines are appended to and flushed at once,
// because the log thread dies with the process when the exception is unhandled.
constexpr DWORD GccThrow = 0x20474343;
constexpr u64 LibcxxabiPrimaryClass = 0x434C4E47432B2B00ULL; // "CLNGC++\0"
// libc++abi, 64-bit: __cxa_exception has 96 bytes of fields before its _Unwind_Exception (SEH: class, cleanup,
// private_[6] = 64 bytes); exceptionType is its third field; the thrown object follows the header
constexpr ptrdiff_t TypeFromUnwind = -96 + 16;
constexpr ptrdiff_t ObjectFromUnwind = 64;
constexpr u32 MaxTraceRecords = 200;

std::mutex g_trace_mutex;
std::FILE* g_trace_file{};
std::atomic<u32> g_trace_records{0};
LPTOP_LEVEL_EXCEPTION_FILTER g_prev_filter{};

// The build has no RTTI (-fno-rtti), so the type is only known by its mangled name. what() is read only for types
// known to derive from std::exception at offset 0 (single inheritance): the standard ones and NXN1X's own
bool IsStdExceptionAtZero(const char* mangled) {
    static constexpr const char* known[] = {
        "St9exception", "St13runtime_error", "St11logic_error", "St12out_of_range", "St16invalid_argument",
        "St12length_error", "St11range_error", "St14overflow_error", "St15underflow_error", "St12domain_error",
        "St9bad_alloc", "St20bad_array_new_length", "NSt3__112system_errorE", "NSt3__14__fs10filesystem16filesystem_errorE",
        "N6Vulkan2vk9ExceptionE", "N6Shader9ExceptionE", "N6Shader10LogicErrorE", "N6Shader12RuntimeErrorE",
        "N6Shader23NotImplementedExceptionE", "N6Shader15InvalidArgumentE"};
    for (const char* name : known) {
        if (std::strcmp(mangled, name) == 0) {
            return true;
        }
    }
    return false;
}

std::string TraceThreadName() {
    using GetDescription = HRESULT(WINAPI*)(HANDLE, PWSTR*);
    static const auto get = reinterpret_cast<GetDescription>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetThreadDescription")));
    std::string name;
    PWSTR wide{};
    if (get != nullptr && SUCCEEDED(get(GetCurrentThread(), &wide)) && wide != nullptr) {
        for (const wchar_t* p = wide; *p != 0 && name.size() < 64; ++p) {
            name.push_back(*p < 0x80 ? static_cast<char>(*p) : '?');
        }
        LocalFree(wide);
    }
    return name;
}

void TraceStack(const CONTEXT& start, std::string& out) {
    CONTEXT ctx = start;
    for (int i = 0; i < 40 && ctx.Pc != 0; ++i) {
        const DWORD64 pc = ctx.Pc;
        HMODULE module{};
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCSTR>(pc), &module)) {
            char path[MAX_PATH] = "?";
            GetModuleFileNameA(module, path, sizeof(path));
            const char* const slash = std::strrchr(path, '\\');
            out += fmt::format(" {}+{:#x}", slash != nullptr ? slash + 1 : path, pc - reinterpret_cast<DWORD64>(module));
        } else {
            out += fmt::format(" {:#x}", pc); // guest code or the gate block: no unwind data, the walk ends here
            break;
        }
        DWORD64 image_base{};
        auto* const function = RtlLookupFunctionEntry(pc, &image_base, nullptr);
        if (function == nullptr) {
            break;
        }
        void* handler_data{};
        DWORD64 establisher{};
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, pc, function, &ctx, &handler_data, &establisher, nullptr);
        if (ctx.Pc == pc) {
            break;
        }
    }
}

void TraceRecord(const char* kind, EXCEPTION_POINTERS* ep) {
    const u32 n = g_trace_records.fetch_add(1, std::memory_order_relaxed);
    if (n >= MaxTraceRecords) {
        return;
    }
    const auto& rec = *ep->ExceptionRecord;
    std::string line = fmt::format("C++ throw trace #{} {}: thread {} '{}' gate {}", n, kind, GetCurrentThreadId(),
                                   TraceThreadName(), GateLive() ? "on" : "off");
    if (rec.ExceptionCode == GccThrow && rec.NumberParameters >= 1 && rec.ExceptionInformation[0] != 0) {
        const auto unwind = static_cast<uintptr_t>(rec.ExceptionInformation[0]);
        const u64 cls = *reinterpret_cast<const u64*>(unwind);
        if (cls == LibcxxabiPrimaryClass) {
            const auto* const type = *reinterpret_cast<const std::type_info* const*>(unwind + TypeFromUnwind);
            int status = -1;
            char* const demangled = type != nullptr ? abi::__cxa_demangle(type->name(), nullptr, nullptr, &status) : nullptr;
            line += fmt::format(" type {}", demangled != nullptr ? demangled : (type != nullptr ? type->name() : "?"));
            std::free(demangled);
            if (type != nullptr && IsStdExceptionAtZero(type->name())) {
                const auto* const e = reinterpret_cast<const std::exception*>(unwind + ObjectFromUnwind);
                line += fmt::format(" what \"{}\"", e->what());
            }
        } else {
            line += fmt::format(" exception class {:#x}", cls);
        }
    } else {
        line += fmt::format(" code {:#x} at {:#x}", static_cast<u32>(rec.ExceptionCode),
                            reinterpret_cast<uintptr_t>(rec.ExceptionAddress));
    }
    line += " stack:";
    TraceStack(*ep->ContextRecord, line);
    {
        std::scoped_lock lock{g_trace_mutex};
        if (g_trace_file != nullptr) {
            std::fputs(line.c_str(), g_trace_file);
            std::fputc('\n', g_trace_file);
            std::fflush(g_trace_file);
        }
    }
    LOG_CRITICAL(Core, "{}", line);
}

LONG CALLBACK ThrowTraceHandler(EXCEPTION_POINTERS* ep) {
    if (ep->ExceptionRecord->ExceptionCode == GccThrow) {
        TraceRecord("throw", ep);
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

LONG WINAPI ThrowTraceUnhandled(EXCEPTION_POINTERS* ep) {
    TraceRecord("UNHANDLED", ep);
    return g_prev_filter != nullptr ? g_prev_filter(ep) : EXCEPTION_CONTINUE_SEARCH;
}

void InstallThrowTraceOnce() {
    static std::once_flag once;
    std::call_once(once, [] {
        const char* const value = std::getenv("NXN1X_CXX_THROW_TRACE");
        if (value == nullptr || *value == 0 || std::strcmp(value, "0") == 0) {
            return;
        }
        if (std::strcmp(value, "1") != 0) {
            g_trace_file = std::fopen(value, "a");
        }
        AddVectoredExceptionHandler(0, ThrowTraceHandler);
        g_prev_filter = SetUnhandledExceptionFilter(ThrowTraceUnhandled);
        LOG_INFO(Core, "NXN1X_CXX_THROW_TRACE: C++ throws and unhandled exceptions are logged{}",
                 g_trace_file != nullptr ? fmt::format(" and appended to {}", value) : std::string{});
    });
}
#endif
} // Anonymous namespace

#if defined(NXN1X_NVN_HOST)
void OutputAttach(); // nvn_output.cpp
#endif

bool Available() {
#if defined(NXN1X_NVN_HOST)
    return true;
#else
    return false;
#endif
}

#if defined(NXN1X_NVN_HOST)
extern "C" void nvnh_set_title(unsigned long long id);   // s31 (host dm/nxn1x/nvnh_rt.c)
#endif
bool RanInProcess() {   // u1
#if defined(NXN1X_NVN_HOST)
    return nvnh_vk_active() != 0;
#else
    return false;
#endif
}

void OnAppLoad() {
    g_wanted.store(false, std::memory_order_relaxed);
#if defined(NXN1X_NVN_HOST)
    InstallThrowTraceOnce();
    GateBeginBoot();
    ReportLinkedOnce();
    if (Settings::values.nce_gpu_path.GetValue() != Settings::NceGpuPath::Nvn) {
        LOG_INFO(Core, "GPU path: Emulation (nce_gpu_path)");
        return;
    }
    // NVN2: a Switch 2 program takes the layer too (nce_gpu_path = Nvn only). The gate reads the sdk module's NVN2 constants
    // (nvn_gate.cpp ReadNvn2) and the shader bridge lowers its Ampere code as the Maxwell path does (video_core/nvn/nvn_shader_env.cpp)
    if (!Settings::IsNceEnabled()) {
        LOG_WARNING(Core, "GPU path: Emulation; NVN was asked for but NCE does not run this game");
        return;
    }
    if (Settings::values.renderer_backend.GetValue() != Settings::RendererBackend::Vulkan) {
        // the layer runs on the Vulkan renderer's device (NVN_HOST_BRIDGE.md 4.3)
        LOG_WARNING(Core, "GPU path: Emulation; NVN was asked for but the renderer is not Vulkan");
        return;
    }
    if (nvnh_vk_active()) {
        // vk_init ran on an earlier boot's device: the layer's pools, caches and threads are not reset between boots
        LOG_WARNING(Core, "GPU path: Emulation; the NVN layer already ran in this process (restart NXN1X to use NVN again)");
        return;
    }
    if (Settings::CurrentProgramIsSwitch2()) {
        LOG_INFO(Core, "GPU path: NVN host layer (Switch 2, NVN2)");
#if defined(_WIN32)
        // NVN2: Turnip's TU_GS_ONCE (NXN1X's Turnip build, ir3_nir_lower_tess.c gs_once_enabled): a geometry shader's storage / global / image writes and atomics
        // run on its thread 0 only. Turnip runs a geometry shader's body once per output vertex, so Donkey Kong Bananza's terrain GS (5d537f7630ec3671, which
        // appends its visible triangles to a list with storage atomics) appended each triangle 3 times and overflowed the list: black terrain (the old DK host's
        // h4/h4b, RUN_DK_WIN.cmd; NPARK 11-16 09:54 with NXN1X.exe started directly, without nvn2_*.bat). Turnip reads it once, at the first geometry shader it
        // compiles, from the process environment (GetEnvironmentVariableA): set here, at the boot's path decision, before the game's shaders. A value the
        // environment already has (TU_GS_ONCE=0 included) is kept.
        // 11-16 (NPARK: "nvn2_run.bat 이건 기본값으로 해줘"): the settings nvn2_run.bat sets are the Switch 2 defaults, so NXN1X.exe started directly behaves as the bat.
        // DM_VK_TBUF_FRESH=1 (host gfx_vk_tex.c tbuf_fresh_on): a texel buffer's copy is made again on the GPU after a GPU write into its storage. Each only when the
        // environment has none (a user's value, 0 included, wins). One line per Switch 2 boot with what was set and what the environment already had
        struct Nvn2Default { const char* name; const char* value; };
        static constexpr Nvn2Default nvn2_defaults[] = {{"TU_GS_ONCE", "1"}, {"DM_VK_TBUF_FRESH", "1"}};
        std::string applied, kept;
        for (const Nvn2Default& d : nvn2_defaults) {
            if (const char* const have = std::getenv(d.name); have == nullptr) {
                _putenv_s(d.name, d.value);
                applied += fmt::format("{}{}={}", applied.empty() ? "" : ", ", d.name, d.value);
            } else {
                kept += fmt::format("{}{}={}", kept.empty() ? "" : ", ", d.name, have);
            }
        }
        _putenv_s("NXN1X_NVN2_HOST", "1");   // bak86: the shader translation's NVN2-only rules (vk_pipeline_cache.cpp first_store) take this boot's path from it
        LOG_INFO(Core, "NVN2 defaults (as nvn2_run.bat): set {}; from the environment {} (a value in the environment overrides each)",
                 applied.empty() ? "none" : applied, kept.empty() ? "none" : kept);
#endif
    } else {
        LOG_INFO(Core, "GPU path: NVN host layer");
    }
    g_wanted.store(true, std::memory_order_relaxed);
#endif
}

// s31: the title for the layer's per-title profile defaults (render_cond for TotK)
void SetTitleId(unsigned long long title_id) {
#if defined(NXN1X_NVN_HOST)
    nvnh_set_title(title_id);
#else
    (void)title_id;
#endif
}

bool Wanted() {
    return g_wanted.load(std::memory_order_relaxed);
}

void Refuse(std::string_view why) {
#if defined(NXN1X_NVN_HOST)
    if (!g_wanted.exchange(false, std::memory_order_relaxed)) {
        return;
    }
    GateSetLive(false);
    LOG_WARNING(Core, "GPU path: Emulation; {}", why);
#else
    (void)why;
#endif
}

void OnModulesPlanned() {
#if defined(NXN1X_NVN_HOST)
    if (Wanted() && !GateBootstrapHooked()) {
        Refuse("no module of the game defines nvnBootstrapLoader");
    }
#endif
}

void GateOn() {
#if defined(NXN1X_NVN_HOST)
    if (!Wanted()) {
        return;
    }
    const size_t hooks = GateHookCount();
    if (hooks == 0) {
        Refuse("no NVN definition was taken over");
        return;
    }
    GateSetLive(true);
    LOG_INFO(Core, "NVN gate on: {} definitions taken over ({} without an original start to fall back to)", hooks,
             GateHooksWithoutStart());
    OutputAttach(); // step 6: the layer's presents go to the compositor (nvn_output.cpp)
#endif
}

namespace {
std::atomic<u64> g_wf_gh{}, g_wf_gp{}, g_wf_hh{}, g_wf_hp{}, g_wf_noroom{};
std::atomic<u32> g_wf_n{};
WatchFaultKinds::First g_wf_first[8]{};
std::atomic<u32> g_wf_published{};
} // namespace

void NoteWatchFault(bool host_stack, bool room, bool handled, u64 address, u64 pc, u64 sp) {   // vbvol (nvn_layer.h)
    if (!room) g_wf_noroom.fetch_add(1, std::memory_order_relaxed);
    (host_stack ? (handled ? g_wf_hh : g_wf_hp) : (handled ? g_wf_gh : g_wf_gp)).fetch_add(1, std::memory_order_relaxed);
    const u32 i = g_wf_n.fetch_add(1, std::memory_order_relaxed);
    if (i < 8) {
        g_wf_first[i] = {static_cast<u32>(GetCurrentThreadId()), host_stack, room, handled, address, pc, sp};
        g_wf_published.fetch_add(1, std::memory_order_release);
    }
}

WatchFaultKinds GetWatchFaultKinds() {
    WatchFaultKinds k{};
    k.guest_handled = g_wf_gh.load(std::memory_order_relaxed);
    k.guest_passed = g_wf_gp.load(std::memory_order_relaxed);
    k.host_handled = g_wf_hh.load(std::memory_order_relaxed);
    k.host_passed = g_wf_hp.load(std::memory_order_relaxed);
    k.no_room = g_wf_noroom.load(std::memory_order_relaxed);
    k.nfirst = std::min<u32>(g_wf_published.load(std::memory_order_acquire), 8);
    for (u32 i = 0; i < k.nfirst; ++i) k.first[i] = g_wf_first[i];
    return k;
}

bool OnWatchedWrite(u64 address) {
#if defined(NXN1X_NVN_HOST)
    if (!g_wanted.load(std::memory_order_relaxed)) {
        return false;
    }
    // nvn_dirty.c registers on_fault with its first pool; it reports only its own protected pages, and only for writes
    const nvnh_fault_fn on_fault = nvnh_fault_hook();
    if (on_fault == nullptr || on_fault(reinterpret_cast<void*>(address), 1, nullptr) == 0) {
        return false;
    }
    // on_fault also says "ours" for a page of its pools that it unprotected earlier (another thread may have just done so, and the write
    // only has to run again). That is the usual case and costs nothing more. If the same thread faults on the same page again at once
    // (within about 2 ms), the write did not get through: it may reach into the next page (a store pair or vector store across the boundary
    // can report the address of its first byte), or something other than the layer protected the page. Then the neighbouring pages are
    // released too, and the page is checked with VirtualQuery (slow on a large view: about 1.5 ms in run 7, so never on the usual path)
    static const s64 window = [] {
        LARGE_INTEGER f{};
        QueryPerformanceFrequency(&f);
        return f.QuadPart / 500;
    }();
    constexpr u64 Page = 4096; // nvnh_posix.c getpagesize: the page nvn_dirty protects
    const u64 page = address & ~(Page - 1);
    LARGE_INTEGER now{};
    QueryPerformanceCounter(&now);
    WatchRepeat& r = t_repeat;
    const bool again = r.page == page && now.QuadPart - r.when < window;
    r.when = now.QuadPart;
    if (!again) {
        r.page = page;
        r.count = 0;
        return true;
    }
    ++r.count;
    g_repeat_faults.fetch_add(1, std::memory_order_relaxed);
    if (r.count > g_repeat_worst.load(std::memory_order_relaxed)) {
        g_repeat_worst.store(r.count, std::memory_order_relaxed);
        g_repeat_addr.store(address, std::memory_order_relaxed);
    }
    if (on_fault(reinterpret_cast<void*>(page + Page), 1, nullptr) != 0) {
        g_neighbour_releases.fetch_add(1, std::memory_order_relaxed);
    }
    if (page >= Page && on_fault(reinterpret_cast<void*>(page - 1), 1, nullptr) != 0) {
        g_neighbour_releases.fetch_add(1, std::memory_order_relaxed);
    }
    MEMORY_BASIC_INFORMATION info{};
    constexpr DWORD Writable = PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    if (VirtualQuery(reinterpret_cast<void*>(address), &info, sizeof(info)) != 0 && (info.Protect & Writable) != 0 &&
        (info.Protect & (PAGE_GUARD | PAGE_NOACCESS)) == 0) {
        return true;
    }
    g_foreign_faults.fetch_add(1, std::memory_order_relaxed);
    u64 none = 0;
    g_foreign_first.compare_exchange_strong(none, address, std::memory_order_relaxed);
    return false;
#else
    (void)address;
    return false;
#endif
}

WatchStats GetWatchStats() {
    WatchStats s{};
    s.foreign = g_foreign_faults.load(std::memory_order_relaxed);
    s.foreign_first = g_foreign_first.load(std::memory_order_relaxed);
    s.repeats = g_repeat_faults.load(std::memory_order_relaxed);
    s.repeat_worst = g_repeat_worst.load(std::memory_order_relaxed);
    s.repeat_addr = g_repeat_addr.load(std::memory_order_relaxed);
    s.neighbours = g_neighbour_releases.load(std::memory_order_relaxed);
    return s;
}

void BeforeGuestWrite(u64 address, u64 size) {
#if defined(NXN1X_NVN_HOST)
    if (size == 0 || !g_wanted.load(std::memory_order_relaxed)) {
        return;
    }
    // nvn_dirty.c registers prep_write with its first pool (nothing to do before that)
    if (const nvnh_prep_fn prep = nvnh_write_prep_hook()) {
        prep(reinterpret_cast<void*>(address), static_cast<size_t>(size));
    }
#else
    (void)address;
    (void)size;
#endif
}

void ObserveGuestWrite(u64 address, u64 size) {
#if defined(NXN1X_NVN_HOST)
    if (size && g_wanted.load(std::memory_order_relaxed)) nvnc_cpu(address, size, NVNC_CPU_PRESTORE);
#else
    (void)address; (void)size;
#endif
}

void SetGuestMemory(Core::Memory::Memory& memory) {
#if defined(NXN1X_NVN_HOST)
    if (!Wanted()) {
        return;
    }
    g_guest_memory.store(&memory, std::memory_order_release);
    nvnh_set_import_alias(ImportAlias);
    const nvnc_resolver content{NVNC_ABI, sizeof(nvnc_resolver), ContentResolve, ContentEpoch};
    nvnc_set_resolver(&content);
    nvnc_start(Settings::CurrentProgramIsSwitch2());
    if (nvnc_active()) {
        nvnc_stats stats{}; nvnc_get_stats(&stats);
        LOG_INFO(Core, "NVN2 F2 content tracking: OBSERVE; metadata cap {} bytes; unobserved NCE stores remain Unknown; allocation and reconstruction policy unchanged", stats.metadata_bytes);
    }
#else
    (void)memory;
#endif
}

#if defined(NXN1X_NVN_HOST)
extern "C" uint8_t* dm_fps_state;   // host dm/gfx/nvn/hle_nvn.c (weak): the frame step state page (r63 realtime: +0xc0 / +0xc4, rt_step_tick)
#endif
void SetRealtimeState(u64 address) {
#if defined(NXN1X_NVN_HOST)
    Core::Memory::Memory* const memory = g_guest_memory.load(std::memory_order_acquire);
    u8* const p = Wanted() && memory ? memory->GetPointerSilent(address) : nullptr;
    if (!p) {
        LOG_WARNING(Core, "realtime game step: the state page at {:#x} has no host pointer; the game keeps its own steps", address);
        return;
    }
    dm_fps_state = p;
    LOG_INFO(Core, "realtime game step: the host's frame step state is the guest page at {:#x}", address);
#else
    (void)address;
#endif
}

void SetBackingMemory(Common::HostMemory& backing) {
#if defined(NXN1X_NVN_HOST)
    if (!Wanted()) {
        return;
    }
    g_backing.store(&backing, std::memory_order_release);
    g_content_backing_id.fetch_add(1, std::memory_order_acq_rel);
#else
    (void)backing;
#endif
}

void SlowGateCall(Core::ArmInterface& cpu, u32 svc, Core::System& system) {
#if defined(NXN1X_NVN_HOST)
    Core::Sv1::StartWatchdog(system);   // sv1: once per run, log only
    GateSlowCall(cpu, svc, system);
#else
    (void)cpu;
    (void)system;
    LOG_CRITICAL(Core, "NVN S call {:#x} in a build without the NVN layer", svc);
#endif
}

void OnFrame() {
#if defined(NXN1X_NVN_HOST)
    if (!Wanted() || GateBootstrapCalled()) {
        return;
    }
    Refuse(GateLive() ? "the game drew a frame before calling nvnBootstrapLoader (not an NVN game)"
                      : "NXN1X's device was never handed over to the NVN layer");
#endif
}

} // namespace Core::NvnLayer
