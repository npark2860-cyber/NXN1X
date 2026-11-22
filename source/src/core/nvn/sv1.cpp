// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// sv1 (lane A): log-only guest thread dumps when the NVN path stops presenting (docs/lane_a/apply_sv1.py).
// The thread list is walked without the kernel's locks, as debugger/gdbstub.cpp does: the watchdog is a host thread the kernel does not
// know (KScopedSchedulerLock / the process list lock need a kernel thread), and the hang being looked at may hold them. The contexts are
// those KScheduler::Unload saved (PhysicalCore::SaveContext); a thread current on a core is marked, its saved context may be stale.

#include "core/nvn/sv1.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include <algorithm>
#include <cstring>
#include <map>

#include "core/arm/nce/hwa_pool.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hardware_properties.h"
#include <cstdio>
#include "core/arm/debug.h"   // thrname: GetThreadName
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"

#if defined(NXN1X_NVN_HOST)
extern "C" void (*nvnh_thread_cpu_hook)(char* out, size_t n, double secs);   // cpudiag: host dm/gfx/nvn/hle_nvn.c dm_thread_cpu
extern "C" void nvnh_hang_dump(const char* why);   // hangstk (host dm/nxn1x/nvnh_rt.c): every host thread's stack at the first dump
#endif

namespace Core::Sv1 {
namespace {

struct Module {
    std::string name;
    u64 base;
    u64 size;
};

std::mutex g_mod_mu;
std::vector<Module> g_modules;
std::atomic<u64> g_presents{0};
std::atomic<bool> g_started{false};
// combined1120: the watchdog threads were detached and kept a Core::System* after the system was destroyed: at shutdown one called IsPoweredOn on freed memory
// (18:41, c0000005 in Core::System::IsPoweredOn from sv1 Watch; dump NXN1X.exe.28216.dmp). Now they wait on g_stop_cv and are joined by StopWatchdog
// (System::Impl::ShutdownMainProcess, before the system goes away)
std::mutex g_stop_mu;
std::condition_variable g_stop_cv;
bool g_stop = false;
std::vector<std::thread> g_threads;
template <class D> bool SleepOrStop(D d) {   // false: stop requested
    std::unique_lock lk{g_stop_mu};
    return !g_stop_cv.wait_for(lk, d, [] { return g_stop; });
}

using Clock = std::chrono::steady_clock;

std::string Where(u64 address) {
    std::scoped_lock lock{g_mod_mu};
    for (const Module& m : g_modules) {
        if (address >= m.base && address - m.base < m.size) {
            return fmt::format("{}+{:#x}", m.name, address - m.base);
        }
    }
    return fmt::format("{:#x}", address);
}

const char* StateName(u32 state) {
    switch (state & static_cast<u32>(Kernel::ThreadState::Mask)) {
    case 0:
        return "Initialized";
    case 1:
        return "Waiting";
    case 2:
        return "Runnable";
    case 3:
        return "Terminated";
    default:
        return "?";
    }
}

const char* WaitName(Kernel::ThreadWaitReasonForDebugging reason) {
    switch (reason) {
    case Kernel::ThreadWaitReasonForDebugging::None:
        return "None";
    case Kernel::ThreadWaitReasonForDebugging::Sleep:
        return "Sleep";
    case Kernel::ThreadWaitReasonForDebugging::IPC:
        return "IPC";
    case Kernel::ThreadWaitReasonForDebugging::Synchronization:
        return "Synchronization";
    case Kernel::ThreadWaitReasonForDebugging::ConditionVar:
        return "ConditionVar";
    case Kernel::ThreadWaitReasonForDebugging::Arbitration:
        return "Arbitration";
    case Kernel::ThreadWaitReasonForDebugging::Suspended:
        return "Suspended";
    default:
        return "?";
    }
}

void Watch(Core::System* system) {
    constexpr auto Quiet = std::chrono::seconds(8);
    constexpr auto Apart = std::chrono::seconds(10);
    constexpr int MaxDumps = 3;
    u64 seen = g_presents.load(std::memory_order_relaxed);
    auto last_progress = Clock::now();
    auto last_dump = Clock::time_point{};
    int dumps = 0;
    while (dumps < MaxDumps) {
        if (!SleepOrStop(std::chrono::seconds(1))) return;
        if (!system->IsPoweredOn() || system->ApplicationProcess() == nullptr) {
            LOG_INFO(Core, "sv1: watchdog ends (the application is gone), {} dumps", dumps);
            return;
        }
        const auto now = Clock::now();
        const u64 presents = g_presents.load(std::memory_order_relaxed);
        if (presents != seen) {
            seen = presents;
            last_progress = now;
            continue;
        }
        if (now - last_progress < Quiet || (dumps > 0 && now - last_dump < Apart)) {
            continue;
        }
        ++dumps;
        last_dump = now;
        const auto quiet_s = std::chrono::duration_cast<std::chrono::seconds>(now - last_progress).count();
        const std::string why = fmt::format("no present for {} s ({} presents so far), dump {} of {}", quiet_s, presents, dumps, MaxDumps);
        DumpGuestThreads(*system, why.c_str());
#if defined(NXN1X_NVN_HOST)
        if (dumps == 1) nvnh_hang_dump(why.c_str());   // hangstk
#endif
    }
    LOG_INFO(Core, "sv1: watchdog ends after {} dumps", dumps);
}

// wakediag (NXN1X_HWA_POOLDIAG=1, log only): condition variable traffic recorded by k_condition_variable.cpp (Signal / Wait) into a
// ring without locks; PoolDiag summarizes the window of a pool burst. An entry being overwritten while it is read is possible (diagnostic)
struct CvRecord {
    u64 ns;
    u64 key;
    u64 tid;
    s32 a;
    s32 b;
    u32 kind;
};
constexpr size_t CvRingSize = 1u << 17;
std::array<CvRecord, CvRingSize> g_cv_ring{};
std::atomic<u64> g_cv_n{0};

u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

void SummarizeWakes(Core::System& system, u64 from_ns, u64 to_ns, int burst, u32 peak) {
    const u64 n = g_cv_n.load(std::memory_order_acquire);
    const u64 first = n > CvRingSize ? n - CvRingSize : 0;
    struct KeyStats {
        u32 signals{}, signals_empty{}, signals_all{}, taken{}, woken{}, to_mutex{}, waits{}, returns{};
        std::vector<double> lat_ms;
        std::vector<u64> tids;
    };
    std::map<u64, KeyStats> keys;
    std::map<u64, std::pair<u64, u64>> pending;   // tid -> (signal ns, key) for Woken / ToMutex, matched with its WaitEnd
    std::vector<double> all_lat;
    u64 used = 0, oldest = ~0ull;
    for (u64 i = first; i < n; ++i) {
        const CvRecord r = g_cv_ring[i % CvRingSize];
        oldest = (std::min)(oldest, r.ns);
        if (r.ns < from_ns || r.ns > to_ns) {
            continue;
        }
        ++used;
        KeyStats& k = keys[r.key];
        switch (static_cast<CvEvent>(r.kind)) {
        case CvEvent::Signal:
            ++k.signals;
            k.signals_empty += r.b == 0;
            k.signals_all += r.a <= 0;
            k.taken += static_cast<u32>(r.b);
            break;
        case CvEvent::Woken:
        case CvEvent::ToMutex:
            ++(static_cast<CvEvent>(r.kind) == CvEvent::Woken ? k.woken : k.to_mutex);
            pending[r.tid] = {r.ns, r.key};
            if (std::find(k.tids.begin(), k.tids.end(), r.tid) == k.tids.end()) k.tids.push_back(r.tid);
            break;
        case CvEvent::WaitBegin:
            ++k.waits;
            if (std::find(k.tids.begin(), k.tids.end(), r.tid) == k.tids.end()) k.tids.push_back(r.tid);
            break;
        case CvEvent::WaitEnd:
            ++k.returns;
            if (auto it = pending.find(r.tid); it != pending.end() && it->second.second == r.key) {
                const double ms = static_cast<double>(r.ns - it->second.first) / 1e6;
                k.lat_ms.push_back(ms);
                all_lat.push_back(ms);
                pending.erase(it);
            }
            break;
        }
    }
    const auto pct = [](std::vector<double> v, double q) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[(std::min)(v.size() - 1, static_cast<size_t>(q * static_cast<double>(v.size())))];
    };
    std::map<u64, s32> prio;
    if (Kernel::KProcess* const process = system.ApplicationProcess()) {
        int nt = 0;
        for (auto& thread : process->GetThreadList()) {
            if (++nt > 256) break;
            prio[thread.GetThreadId()] = thread.GetPriority();
        }
    }
    LOG_INFO(Core, "wakediag: burst {} (peak {} live, {:.0f} ms window): {} condvar events (ring {}; {}), {} keys; signal->back-from-wait "
                   "latency p50 {:.2f} p90 {:.2f} max {:.2f} ms over {} wakes; {} signalled threads not back from Wait by the window's end",
             burst, peak, static_cast<double>(to_ns - from_ns) / 1e6, used, n - first,
             oldest > from_ns && first > 0 ? "OVERFLOWED: the window's start is lost" : "complete", keys.size(), pct(all_lat, 0.5),
             pct(all_lat, 0.9), all_lat.empty() ? 0.0 : *std::max_element(all_lat.begin(), all_lat.end()), all_lat.size(), pending.size());
    std::vector<std::pair<u64, const KeyStats*>> order;
    for (const auto& [key, k] : keys) order.emplace_back(key, &k);
    std::sort(order.begin(), order.end(), [](const auto& x, const auto& y) {
        return x.second->waits + x.second->signals > y.second->waits + y.second->signals;
    });
    int shown = 0;
    for (const auto& [key, k] : order) {
        if (++shown > 32) break;
        std::string tids;
        for (size_t i = 0; i < k->tids.size() && i < 8; ++i) {
            const auto p = prio.find(k->tids[i]);
            tids += fmt::format(" {}(p{})", k->tids[i], p != prio.end() ? p->second : -1);
        }
        LOG_INFO(Core, "wakediag:   key {:#x}: waits {} back {} | signals {} (all-waiters {}, none waiting {}) took {} woken {} to-mutex {} | "
                       "latency p50 {:.2f} max {:.2f} ms | threads{}{}",
                 key, k->waits, k->returns, k->signals, k->signals_all, k->signals_empty, k->taken, k->woken, k->to_mutex, pct(k->lat_ms, 0.5),
                 k->lat_ms.empty() ? 0.0 : *std::max_element(k->lat_ms.begin(), k->lat_ms.end()), tids, k->tids.size() > 8 ? " …" : "");
    }
}

// pooldiag (NXN1X_HWA_POOLDIAG=1, log only): Hyrule Warriors' slot-48 block pool (core/arm/nce/hwa_pool.h) fills to 16000 of 16000
// blocks while loading and the game crashes on the null block. Allocs and frees run as guest code without host traps, so this samples
// the pool's live count from a host thread: is it a steady climb (frees too slow) or frees stopping (workers not woken)? The pool object
// is found once by its vtable (a scan of the heap region's mapped pages); the guest threads are listed as DumpGuestThreads does.
void PoolDiag(Core::System* system) {
    namespace Pool = Core::NCE::HwaPool;
    constexpr auto Step = std::chrono::milliseconds(50);
    constexpr u32 Hot = 12000;            // per-sample lines and thread states only near the 16000 cap
    constexpr int MaxThreadDumps = 120;   // bounded: thread states at most every 250 ms while hot
    const auto t0 = Clock::now();
    const auto secs = [&t0] { return std::chrono::duration<double>(Clock::now() - t0).count(); };
    u64 receiver = 0;
    auto next_scan = Clock::now();
    int scans = 0;
    while (receiver == 0) {
        if (!SleepOrStop(std::chrono::milliseconds(200))) return;
        if (!system->IsPoweredOn()) {
            return;
        }
        Kernel::KProcess* const process = system->ApplicationProcess();
        const u64 main_base = Pool::probe_base.load(std::memory_order_relaxed);
        if (process == nullptr || main_base == 0 || Clock::now() < next_scan) {
            continue;
        }
        if (++scans > 30) {
            LOG_WARNING(Core, "pooldiag: no slot-48 pool object found after 30 scans; off");
            return;
        }
        next_scan = Clock::now() + std::chrono::seconds(2);
        const u64 vtable = main_base + Pool::AllocatorVtable;
        auto& memory = process->GetMemory();
        const u64 start = GetInteger(process->GetPageTable().GetHeapRegionStart());
        const u64 size = process->GetPageTable().GetHeapRegionSize();
        const auto ts = Clock::now();
        u64 mapped = 0;
        int found = 0;
        for (u64 page = start; page < start + size && found < 4; page += 0x1000) {
            const u8* const p = memory.GetPointerSilent(Common::ProcessAddress{page});
            if (p == nullptr) {
                continue;
            }
            mapped += 0x1000;
            for (u64 o = 0; o + 8 <= 0x1000; o += 16) {
                u64 v;
                std::memcpy(&v, p + o, sizeof v);
                if (v != vtable || !memory.IsValidVirtualAddressRange(page + o, 0x820) ||
                    memory.Read32(page + o + 0x28) != 0x80) {   // block size: initialized pools only
                    continue;
                }
                ++found;
                LOG_INFO(Core, "pooldiag: pool object {:#x} (vtable main+{:#x}): live {} of 16000, release {:#x}", page + o, Pool::AllocatorVtable,
                         memory.Read32(page + o + 0x2c), memory.Read32(page + o + 0x818));
                if (receiver == 0) {
                    receiver = page + o;
                }
            }
        }
        LOG_INFO(Core, "pooldiag: scan {} of heap {:#x} +{:#x}: {} MB mapped, {} pool objects, {:.0f} ms", scans, start, size, mapped >> 20,
                 found, std::chrono::duration<double, std::milli>(Clock::now() - ts).count());
    }
    LOG_INFO(Core, "pooldiag: sampling pool {:#x} every 50 ms (a line per sample at {} or more live blocks, else one a second; thread "
                   "states while hot, at most {} times)", receiver, Hot, MaxThreadDumps);
    u32 lo = ~0u, hi = 0, last = 0;
    int thread_dumps = 0;
    bool in_burst = false;
    u64 burst_from = 0;
    u32 burst_peak = 0;
    int bursts = 0;
    auto next_second = Clock::now() + std::chrono::seconds(1);
    auto next_threads = Clock::now();
    while (system->IsPoweredOn()) {
        if (!SleepOrStop(Step)) return;
        Kernel::KProcess* const process = system->ApplicationProcess();
        if (process == nullptr) {
            break;
        }
        auto& memory = process->GetMemory();
        if (!memory.IsValidVirtualAddressRange(receiver, 0x820)) {
            LOG_INFO(Core, "pooldiag: pool {:#x} is no longer mapped; ends", receiver);
            return;
        }
        const u32 live = memory.Read32(receiver + 0x2c);
        const u32 release = memory.Read32(receiver + 0x818);
        lo = (std::min)(lo, live);
        hi = (std::max)(hi, live);
        const auto now = Clock::now();
        if (live >= Hot) {
            LOG_INFO(Core, "pooldiag: t {:.2f} s live {} (delta {:+d}) release {:#x}", secs(), live, static_cast<int>(live - last), release);
        }
        if (now >= next_second) {
            LOG_INFO(Core, "pooldiag: t {:.1f} s live {} (last second min {} max {})", secs(), live, lo, hi);
            lo = ~0u;
            hi = 0;
            next_second = now + std::chrono::seconds(1);
        }
        last = live;
        // wakediag: a burst starts at 4000 live blocks and ends below 1000; its condvar traffic (from 300 ms before) is summarized
        if (!in_burst && live >= 4000) {
            in_burst = true;
            burst_from = NowNs() - 300'000'000ull;
            burst_peak = live;
        } else if (in_burst) {
            burst_peak = (std::max)(burst_peak, live);
            if (live < 1000 || NowNs() - burst_from > 10'000'000'000ull) {
                in_burst = false;
                if (bursts < 4) {
                    SummarizeWakes(*system, burst_from, NowNs(), ++bursts, burst_peak);
                }
            }
        }
        if (live < Hot || thread_dumps >= MaxThreadDumps || now < next_threads) {
            continue;
        }
        ++thread_dumps;
        next_threads = now + std::chrono::milliseconds(250);
        // Thread states (as DumpGuestThreads, without the kernel's locks): counts by state / wait, then each prio-44 worker (the job
        // workers that allocated from this pool, tids 231-235 in the 11-20 dumps)
        auto& kernel = system->Kernel();
        std::array<const Kernel::KThread*, Core::Hardware::NUM_CPU_CORES> on_core{};
        for (size_t i = 0; i < on_core.size(); ++i) {
            on_core[i] = kernel.Scheduler(i).GetSchedulerCurrentThread();
        }
        int runnable = 0, waiting = 0, running = 0;
        std::string workers;
        int nthreads = 0;
        for (auto& thread : process->GetThreadList()) {
            if (++nthreads > 256) {
                break;
            }
            const u32 state = static_cast<u32>(thread.GetRawState());
            int core = -1;
            for (size_t i = 0; i < on_core.size(); ++i) {
                if (on_core[i] == &thread) {
                    core = static_cast<int>(i);
                }
            }
            running += core >= 0;
            runnable += (state & static_cast<u32>(Kernel::ThreadState::Mask)) == 2;
            waiting += (state & static_cast<u32>(Kernel::ThreadState::Mask)) == 1;
            if (thread.GetPriority() == 44) {
                workers += fmt::format(" [{} {}/{}{} key {:#x} pc {}]", thread.GetThreadId(), StateName(state),
                                       WaitName(thread.GetWaitReasonForDebugging()), core >= 0 ? fmt::format(" on core {}", core) : std::string{},
                                       GetInteger(thread.GetAddressKey()), Where(thread.GetContext().pc));
            }
        }
        LOG_INFO(Core, "pooldiag: t {:.2f} s live {} threads {}: runnable {} waiting {} on a core {}; prio-44 workers:{}", secs(), live,
                 nthreads, runnable, waiting, running, workers.empty() ? std::string{" none"} : workers);
    }
    LOG_INFO(Core, "pooldiag: ends (the application is gone)");
}

} // Anonymous namespace

void NoteModule(std::string_view name, u64 base, u64 size) {
    std::scoped_lock lock{g_mod_mu};
    for (Module& m : g_modules) {
        if (m.name == name) {   // a new run in the same process: replaced
            m.base = base;
            m.size = size;
            return;
        }
    }
    g_modules.push_back(Module{std::string{name}, base, size});
}

bool WakeTraceOn() {
    static const bool on = [] {
        const char* e = std::getenv("NXN1X_HWA_POOLDIAG");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

void NoteCv(CvEvent kind, u64 key, u64 tid, s32 a, s32 b) {
    const u64 i = g_cv_n.fetch_add(1, std::memory_order_relaxed);
    g_cv_ring[i % CvRingSize] = CvRecord{NowNs(), key, tid, a, b, static_cast<u32>(kind)};
}

void NotePresent() {
    g_presents.fetch_add(1, std::memory_order_relaxed);
}

// cpudiag (NXN1X_THREADCPU=1, log only; OOT_BARRIER_DESIGN.md 14). Run = the time a guest thread held a core between the scheduler's switches (Eden's own
// KThread CPU time: guest code and its SVCs and HLE services, also while a service blocks the core's host thread); wait = the rest of the window. A thread that
// keeps a core without switching is counted at its next switch
namespace {
constexpr size_t TcSlots = 512;
struct TcSlot {
    std::atomic<u64> tid{0};   // tid + 1 (0: free)
    std::atomic<s64> ticks{0};
    std::atomic<s32> prio{0};
    std::atomic<s32> core{0};
};
std::array<TcSlot, TcSlots> g_tc{};
std::array<std::atomic<s64>, Core::Hardware::NUM_CPU_CORES> g_tc_idle{};
std::atomic<Core::System*> g_tc_system{nullptr};

std::mutex g_entry_mu;
std::map<u64, u64> g_entry;   // thrname: tid -> entry point (svc CreateThread)

void ThreadCpuLine(char* out, size_t n, double secs) {
    if (n == 0) return;
    Core::System* const system = g_tc_system.load(std::memory_order_acquire);
    static std::array<s64, TcSlots> prev{};
    static std::array<u64, TcSlots> prev_tid{};
    static std::array<s64, Core::Hardware::NUM_CPU_CORES> prev_idle{};
    static u64 prev_clock = 0, prev_presents = 0;
    static bool based = false;
    if (system == nullptr) { std::snprintf(out, n, "-"); return; }
    const u64 clock = system->CoreTiming().GetClockTicks();
    const u64 presents = g_presents.load(std::memory_order_relaxed);
    struct Row { u64 tid; s32 prio, core; s64 d; };
    std::vector<Row> rows;
    for (size_t i = 0; i < TcSlots; ++i) {
        const u64 t = g_tc[i].tid.load(std::memory_order_acquire);
        if (t == 0) continue;
        const s64 v = g_tc[i].ticks.load(std::memory_order_relaxed);
        const s64 d = prev_tid[i] == t ? v - prev[i] : v;
        prev[i] = v; prev_tid[i] = t;
        if (d > 0) rows.push_back(Row{t - 1, g_tc[i].prio.load(std::memory_order_relaxed), g_tc[i].core.load(std::memory_order_relaxed), d});
    }
    std::array<s64, Core::Hardware::NUM_CPU_CORES> idle{};
    for (size_t c = 0; c < idle.size(); ++c) { const s64 v = g_tc_idle[c].load(std::memory_order_relaxed); idle[c] = v - prev_idle[c]; prev_idle[c] = v; }
    const u64 window = clock - prev_clock, frames = presents - prev_presents;
    prev_clock = clock; prev_presents = presents;
    if (!based || window == 0) { based = true; std::snprintf(out, n, "-"); return; }   // the first call only takes the baselines
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.d > b.d; });
    const double w = static_cast<double>(window), ms_frame = frames ? secs * 1e3 / static_cast<double>(frames) : 0.0;
    s64 total = 0;
    for (const Row& r : rows) total += r.d;
    std::string s = fmt::format("{} presents, frame {:.1f} ms; guest threads {:.2f} cores busy; cores idle", frames, ms_frame, static_cast<double>(total) / w);
    for (size_t c = 0; c < idle.size(); ++c) s += fmt::format(" {:.0f}%", 100.0 * static_cast<double>(idle[c]) / w);
    s += "; top threads (tid prio core: run ms per frame, share of the window):";
    // thrname: each top thread's nnsdk name, entry and where it is now (the process's thread list as DumpGuestThreads reads it, without the kernel's locks)
    Kernel::KProcess* const process = system->ApplicationProcess();
    const auto who = [&](u64 tid) -> std::string {
        std::string name = "?", pc = "?";
        if (process != nullptr) {
            int n = 0;
            for (auto& thread : process->GetThreadList()) {
                if (++n > 512) break;
                if (thread.GetThreadId() != tid) continue;
                if (const auto nm = Core::GetThreadName(std::addressof(thread))) name = *nm;
                pc = Where(thread.GetContext().pc);
                break;
            }
        }
        u64 entry = 0;
        { std::scoped_lock lk{g_entry_mu}; if (const auto it = g_entry.find(tid); it != g_entry.end()) entry = it->second; }
        return fmt::format(" \"{}\" entry {} now {}", name, entry ? Where(entry) : std::string{"?"}, pc);
    };
    for (size_t i = 0; i < rows.size() && i < 8; ++i) {
        const double share = static_cast<double>(rows[i].d) / w;
        s += fmt::format(" t{} p{} c{}{}: run {:.1f} ms ({:.0f}%), wait {:.1f} ms;", rows[i].tid, rows[i].prio, rows[i].core, i < 4 ? who(rows[i].tid) : std::string{},
                         share * ms_frame, 100.0 * share, (1.0 - share) * ms_frame);
    }
    std::snprintf(out, n, "%s", s.c_str());
}
} // namespace

bool ThreadCpuOn() {
    static const bool on = [] {
        const char* e = std::getenv("NXN1X_THREADCPU");
        return e != nullptr && e[0] == '1';
    }();
    return on;
}

void NoteThreadEntry(u64 tid, u64 entry) {
    if (!ThreadCpuOn()) return;
    std::scoped_lock lk{g_entry_mu};
    g_entry[tid] = entry;
}

void NoteThreadTime(u64 tid, s32 prio, s32 core, s64 ticks, bool idle) {
    if (idle) {
        if (core >= 0 && static_cast<size_t>(core) < g_tc_idle.size()) g_tc_idle[core].fetch_add(ticks, std::memory_order_relaxed);
        return;
    }
    const u64 key = tid + 1;
    size_t i = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 55) % TcSlots;
    for (size_t probe = 0; probe < TcSlots; ++probe, i = (i + 1) % TcSlots) {
        u64 t = g_tc[i].tid.load(std::memory_order_acquire);
        if (t == 0 && g_tc[i].tid.compare_exchange_strong(t, key, std::memory_order_acq_rel)) t = key;
        if (t != key) continue;
        g_tc[i].ticks.fetch_add(ticks, std::memory_order_relaxed);
        g_tc[i].prio.store(prio, std::memory_order_relaxed);
        g_tc[i].core.store(core, std::memory_order_relaxed);
        return;
    }
}

void StartWatchdog(Core::System& system) {
    if (g_started.exchange(true)) {
        return;
    }
    { std::scoped_lock lk{g_stop_mu}; g_stop = false; }
#if defined(NXN1X_NVN_HOST)
    if (ThreadCpuOn()) {   // cpudiag: log only, off unless set
        g_tc_system.store(&system, std::memory_order_release);
        nvnh_thread_cpu_hook = ThreadCpuLine;
        LOG_INFO(Core, "cpudiag: on (NXN1X_THREADCPU=1): guest thread CPU from the scheduler in the \"threads up to frame\" lines");
    }
#endif
    if (const char* e = std::getenv("NXN1X_HWA_POOLDIAG"); e != nullptr && e[0] == '1') {   // pooldiag: log only, off unless set
        LOG_INFO(Core, "pooldiag: on (NXN1X_HWA_POOLDIAG=1): Hyrule Warriors slot-48 pool sampling");
        std::scoped_lock lk{g_stop_mu}; g_threads.emplace_back(PoolDiag, &system);
    }
    if (const char* e = std::getenv("NXN1X_SV1"); e != nullptr && e[0] == '0') {
        LOG_INFO(Core, "sv1: off (NXN1X_SV1=0)");
        return;
    }
    LOG_INFO(Core, "sv1: watchdog on: guest threads are dumped after 8 s without a present (at most 3 dumps, 10 s apart; NXN1X_SV1=0 off)");
    { std::scoped_lock lk{g_stop_mu}; g_threads.emplace_back(Watch, &system); }
}

void StopWatchdog() {
    std::vector<std::thread> threads;
    g_tc_system.store(nullptr, std::memory_order_release);   // cpudiag: the line prints "-" from here on
    { std::scoped_lock lk{g_stop_mu}; g_stop = true; threads.swap(g_threads); }
    g_stop_cv.notify_all();
    for (auto& t : threads) if (t.joinable()) t.join();
    g_started.store(false);
}

void DumpGuestThreads(Core::System& system, const char* why) {
    Kernel::KProcess* const process = system.ApplicationProcess();
    if (process == nullptr) {
        LOG_INFO(Core, "sv1: {}: no application process", why);
        return;
    }
    auto& kernel = system.Kernel();
    std::array<const Kernel::KThread*, Core::Hardware::NUM_CPU_CORES> on_core{};
    for (size_t i = 0; i < on_core.size(); ++i) {
        on_core[i] = kernel.Scheduler(i).GetSchedulerCurrentThread();
    }
    std::vector<Kernel::KThread*> threads;
    for (auto& thread : process->GetThreadList()) {
        threads.push_back(std::addressof(thread));
        if (threads.size() >= 256) {
            break;
        }
    }
    LOG_INFO(Core, "sv1: ===== guest thread dump: {}: {} threads", why, threads.size());
    {
        std::scoped_lock lock{g_mod_mu};
        for (const Module& m : g_modules) {
            LOG_INFO(Core, "sv1: module {} {:#x} +{:#x}", m.name, m.base, m.size);
        }
    }
    auto& memory = process->GetMemory();
    for (Kernel::KThread* thread : threads) {
        const Kernel::Svc::ThreadContext ctx = thread->GetContext();
        const u32 state = static_cast<u32>(thread->GetRawState());
        int core = -1;
        for (size_t i = 0; i < on_core.size(); ++i) {
            if (on_core[i] == thread) {
                core = static_cast<int>(i);
            }
        }
        const std::string running = core >= 0 ? fmt::format(" ON CORE {} (context may be stale)", core) : std::string{};
        LOG_INFO(Core,
                 "sv1: thread {} prio {} core active {} current {} state {} ({:#x}) wait {} svc {} ({:#x}) suspend {:#x} addrkey {:#x} "
                 "arbkey {:#x} tls {:#x}{}",
                 thread->GetThreadId(), thread->GetPriority(), thread->GetActiveCore(), thread->GetCurrentCore(), StateName(state), state,
                 WaitName(thread->GetWaitReasonForDebugging()), thread->IsCallingSvc() ? "in" : "out", thread->GetSvcId(),
                 thread->GetSuspendFlags(), GetInteger(thread->GetAddressKey()), thread->GetAddressArbiterKey(),
                 GetInteger(thread->GetTlsAddress()), running);
        LOG_INFO(Core, "sv1:   pc {} lr {} sp {:#x} fp {:#x} x0 {:#x} x1 {:#x} x2 {:#x} x3 {:#x}", Where(ctx.pc), Where(ctx.lr), ctx.sp,
                 ctx.fp, ctx.r[0], ctx.r[1], ctx.r[2], ctx.r[3]);
        // The frame-pointer chain: [fp] = the caller's fp, [fp + 8] = the return address
        std::string frames;
        u64 fp = ctx.fp;
        int n = 0;
        for (; n < 24; ++n) {
            if (fp == 0 || (fp & 0xF) != 0 || !memory.IsValidVirtualAddressRange(fp, 16)) {
                break;
            }
            const u64 next = memory.Read64(fp);
            const u64 ret = memory.Read64(fp + 8);
            if (ret == 0) {
                break;
            }
            frames += fmt::format(" {}", Where(ret));
            if (next <= fp) {
                frames += " (fp chain ends)";
                ++n;
                break;
            }
            fp = next;
        }
        LOG_INFO(Core, "sv1:   frames {}:{}", n, frames.empty() ? std::string{" (none)"} : frames);
    }
    LOG_INFO(Core, "sv1: ===== end of dump");
}

} // namespace Core::Sv1
