// SPDX-License-Identifier: GPL-3.0-or-later
// s49 (lane A, 2026-11-13): MEASUREMENT ONLY. What the guest waits on during TotK's ~22-31 s loading screen (run 66: process ~0.8 core, no thread
// above 19%, disk 12-22 MB/s, renderer idle at vsync). Counts and times, per 2 s of wall clock: (A) kernel SVCs per id (all of Svc::Call, so the blocking
// ones include their wait) plus SleepThread by requested time; (B) HLE IPC handlers per (service, command); (C) fsp IStorage::Read / IFile::Read with a size
// histogram. The NVN guest-side waits (D) are counted in host/dm/gfx/nvn/hle_nvn.c (C code, its own 2 s line). One reporter: whichever hooked call first
// sees the 2 s boundary writes the lines ("s49: ..."). NXN1X_S49=0: off (nothing counted, nothing logged).
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "common/logging.h"

namespace S49 {

inline bool On() {
    static const bool on = [] { const char* e = std::getenv("NXN1X_S49"); return e && e[0] == '1'; }();   // s54: default off (NXN1X_S49=1 on)
    return on;
}
inline uint64_t Now() {   // steady_clock = QueryPerformanceCounter on Windows
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
struct Cnt {
    std::atomic<uint64_t> n{0}, ns{0}, max{0};
    void Add(uint64_t d) {
        n.fetch_add(1, std::memory_order_relaxed); ns.fetch_add(d, std::memory_order_relaxed);
        uint64_t m = max.load(std::memory_order_relaxed);
        while (d > m && !max.compare_exchange_weak(m, d, std::memory_order_relaxed)) {}
    }
    struct Snap { uint64_t n, ns, max; };
    Snap Take() { return {n.exchange(0, std::memory_order_relaxed), ns.exchange(0, std::memory_order_relaxed), max.exchange(0, std::memory_order_relaxed)}; }
};

inline const char* SvcName(uint32_t id) {
    static const char* const names[256] = {"","SetHeapSize","SetMemoryPermission","SetMemoryAttribute","MapMemory","UnmapMemory","QueryMemory","ExitProcess","CreateThread","StartThread","ExitThread","SleepThread","GetThreadPriority","SetThreadPriority","GetThreadCoreMask","SetThreadCoreMask","GetCurrentProcessorNumber","SignalEvent","ClearEvent","MapSharedMemory","UnmapSharedMemory","CreateTransferMemory","CloseHandle","ResetSignal","WaitSynchronization","CancelSynchronization","ArbitrateLock","ArbitrateUnlock","WaitProcessWideKeyAtomic","SignalProcessWideKey","GetSystemTick","ConnectToNamedPort","SendSyncRequestLight","SendSyncRequest","SendSyncRequestWithUserBuffer","SendAsyncRequestWithUserBuffer","GetProcessId","GetThreadId","Break","OutputDebugString","ReturnFromException","GetInfo","FlushEntireDataCache","FlushDataCache","MapPhysicalMemory","UnmapPhysicalMemory","GetDebugFutureThreadInfo","GetLastThreadInfo","GetResourceLimitLimitValue","GetResourceLimitCurrentValue","SetThreadActivity","GetThreadContext3","WaitForAddress","SignalToAddress","SynchronizePreemptionState","GetResourceLimitPeakValue","","CreateIoPool","CreateIoRegion","","KernelDebug","ChangeKernelTraceState","","","CreateSession","AcceptSession","ReplyAndReceiveLight","ReplyAndReceive","ReplyAndReceiveWithUserBuffer","CreateEvent","MapIoRegion","UnmapIoRegion","MapPhysicalMemoryUnsafe","UnmapPhysicalMemoryUnsafe","SetUnsafeLimit","CreateCodeMemory","ControlCodeMemory","SleepSystem","ReadWriteRegister","SetProcessActivity","CreateSharedMemory","MapTransferMemory","UnmapTransferMemory","CreateInterruptEvent","QueryPhysicalAddress","QueryIoMapping","CreateDeviceAddressSpace","AttachDeviceAddressSpace","DetachDeviceAddressSpace","MapDeviceAddressSpaceByForce","MapDeviceAddressSpaceAligned","","UnmapDeviceAddressSpace","InvalidateProcessDataCache","StoreProcessDataCache","FlushProcessDataCache","DebugActiveProcess","BreakDebugProcess","TerminateDebugProcess","GetDebugEvent","ContinueDebugEvent","GetProcessList","GetThreadList","GetDebugThreadContext","SetDebugThreadContext","QueryDebugProcessMemory","ReadDebugProcessMemory","WriteDebugProcessMemory","SetHardwareBreakPoint","GetDebugThreadParam","","GetSystemInfo","CreatePort","ManageNamedPort","ConnectToPort","SetProcessMemoryPermission","MapProcessMemory","UnmapProcessMemory","QueryProcessMemory","MapProcessCodeMemory","UnmapProcessCodeMemory","CreateProcess","StartProcess","TerminateProcess","GetProcessInfo","CreateResourceLimit","SetResourceLimitLimitValue","CallSecureMonitor","","","","","","","","","","","","","","","","","MapInsecureMemory","UnmapInsecureMemory","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""};
    return id < 256 && names[id][0] ? names[id] : "?";
}
inline std::array<Cnt, 256>& Svc() { static std::array<Cnt, 256> a; return a; }
inline std::array<std::atomic<uint64_t>, 4>& Sleep() { static std::array<std::atomic<uint64_t>, 4> a{}; return a; }   // yield (<=0), <1 ms, 1-10 ms, >10 ms

struct IpcE { uint64_t n = 0, ns = 0, max = 0; };
inline std::mutex& IpcMu() { static std::mutex m; return m; }
inline std::map<std::pair<const char*, uint32_t>, IpcE>& Ipc() { static std::map<std::pair<const char*, uint32_t>, IpcE> m; return m; }

struct Fs {
    Cnt storage, file;
    std::atomic<uint64_t> storage_bytes{0}, file_bytes{0};
    std::array<std::atomic<uint64_t>, 4> hist{};   // <=16 KB, <=256 KB, <=4 MB, >4 MB (both kinds)
};
inline Fs& FsC() { static Fs f; return f; }

inline void Report(double win_s) {
    std::string s;
    {   // A: SVCs, top 10 by total time
        std::vector<std::pair<uint32_t, Cnt::Snap>> v;
        for (uint32_t i = 0; i < 256; i++) { auto x = Svc()[i].Take(); if (x.n) v.push_back({i, x}); }
        std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second.ns > b.second.ns; });
        uint64_t calls = 0; for (auto& e : v) calls += e.second.n;
        s = fmt::format("s49: svc last {:.1f} s, {} calls, top 10 by total:", win_s, calls);
        for (size_t k = 0; k < v.size() && k < 10; k++)
            s += fmt::format(" | {:#x}:{} n={} tot={:.1f}ms max={:.1f}ms", v[k].first, SvcName(v[k].first), v[k].second.n, v[k].second.ns / 1e6, v[k].second.max / 1e6);
        LOG_INFO(Service, "{}", s);
        auto& sl = Sleep();
        LOG_INFO(Service, "s49: sleep last {:.1f} s: SleepThread requested yield(<=0) {} | <1ms {} | 1-10ms {} | >10ms {}", win_s,
                 sl[0].exchange(0), sl[1].exchange(0), sl[2].exchange(0), sl[3].exchange(0));
    }
    {   // B: IPC handlers, top 10 by total time
        std::vector<std::pair<std::pair<const char*, uint32_t>, IpcE>> v;
        { std::lock_guard lk{IpcMu()}; for (auto& e : Ipc()) if (e.second.n) v.push_back(e); Ipc().clear(); }
        std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second.ns > b.second.ns; });
        uint64_t calls = 0, tot = 0; for (auto& e : v) { calls += e.second.n; tot += e.second.ns; }
        s = fmt::format("s49: ipc last {:.1f} s, {} handler calls {:.1f} ms, top 10 by total:", win_s, calls, tot / 1e6);
        for (size_t k = 0; k < v.size() && k < 10; k++)
            s += fmt::format(" | svc={} cmd={} n={} tot={:.1f}ms max={:.1f}ms", v[k].first.first ? v[k].first.first : "?", v[k].first.second, v[k].second.n, v[k].second.ns / 1e6, v[k].second.max / 1e6);
        LOG_INFO(Service, "{}", s);
    }
    {   // C: FS reads
        auto& f = FsC(); auto a = f.storage.Take(), b = f.file.Take();
        LOG_INFO(Service, "s49: fs last {:.1f} s: IStorage::Read n={} {:.1f} MB tot={:.1f}ms max={:.1f}ms | IFile::Read n={} {:.1f} MB tot={:.1f}ms max={:.1f}ms | sizes <=16K {} <=256K {} <=4M {} >4M {}",
                 win_s, a.n, f.storage_bytes.exchange(0) / 1048576.0, a.ns / 1e6, a.max / 1e6, b.n, f.file_bytes.exchange(0) / 1048576.0, b.ns / 1e6, b.max / 1e6,
                 f.hist[0].exchange(0), f.hist[1].exchange(0), f.hist[2].exchange(0), f.hist[3].exchange(0));
    }
}
inline void Tick() {   // the 2 s reporter: the first hooked call past the boundary reports
    static std::atomic<uint64_t> next{0}, last{0};
    uint64_t now = Now(), nx = next.load(std::memory_order_relaxed);
    if (nx == 0) { uint64_t z = 0; if (next.compare_exchange_strong(z, now + 2000000000ull)) last.store(now); return; }
    if (now < nx || !next.compare_exchange_strong(nx, now + 2000000000ull)) return;
    uint64_t l = last.exchange(now);
    Report((now - l) / 1e9);
}

inline void SvcDone(uint32_t imm, uint64_t t0, int64_t sleep_ns, bool is_sleep) {
    Svc()[imm & 255].Add(Now() - t0);
    if (is_sleep) Sleep()[sleep_ns <= 0 ? 0 : sleep_ns < 1000000 ? 1 : sleep_ns <= 10000000 ? 2 : 3].fetch_add(1, std::memory_order_relaxed);
    Tick();
}
inline void IpcDone(const char* svc, uint32_t cmd, uint64_t t0) {
    uint64_t d = Now() - t0;
    { std::lock_guard lk{IpcMu()}; auto& e = Ipc()[{svc, cmd}]; e.n++; e.ns += d; if (d > e.max) e.max = d; }
    Tick();
}
inline void FsDone(bool file, uint64_t bytes, uint64_t t0) {
    auto& f = FsC(); uint64_t d = Now() - t0;
    (file ? f.file : f.storage).Add(d); (file ? f.file_bytes : f.storage_bytes).fetch_add(bytes, std::memory_order_relaxed);
    f.hist[bytes <= 16384 ? 0 : bytes <= 262144 ? 1 : bytes <= 4194304 ? 2 : 3].fetch_add(1, std::memory_order_relaxed);
}

}  // namespace S49
