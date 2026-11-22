// r49: DM_VK_HWWATCH=<address hex>[,<address hex>][#n] (live, read in gfx_vk_exec.c x_read): an ARM64 hardware watchpoint on stores to the 4 bytes at each address
// (two at most: the CPU has two watchpoints a thread), set on every thread of the process. The next n (16) stores that hit are logged with the storing thread's pc
// and lr (module+offset), the instruction, the value it stores (Rt / Rt2, as integer and float), the value there before, the registers and a frame pointer backtrace.
// That answers which game code writes a constant (DK Bananza's scene constant c8+0x218) and from what. Off (0) clears the watchpoints.
// The watchpoint is taken before the store completes: the hit thread's watchpoints are cleared (by the helper thread, a thread cannot set its own) so the store goes
// through when the handler returns; the helper sets them again on every thread once a second while hits are left.
#include "dm.h"
#include <windows.h>
#include <mach/mach.h>
#include <stdlib.h>
#include <string.h>

static volatile uint64_t g_hw_a[2]; static volatile LONG g_hw_left, g_hw_n; static volatile LONG g_hw_on;
static HANDLE g_hw_th, g_hw_req, g_hw_done; static volatile DWORD g_hw_req_tid; static SRWLOCK g_hw_req_lock = SRWLOCK_INIT; static DWORD g_hw_tid;
static PVOID g_hw_veh;

static uint32_t wcr_of(uint64_t a) { return a ? 1u | (2u << 1) | (2u << 3) | ((0xfu << (a & 7)) << 5) : 0; }   // enabled, EL0, stores, the 4 bytes at a
// One thread's watchpoints: arm (the current addresses) or clear. 1 = set and read back as written
static int hw_thread(DWORD tid, int arm, uint32_t *readback) {
    HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, FALSE, tid); if (!h) return 0;
    int ok = 0;
    if (SuspendThread(h) != (DWORD)-1) {   // nothing between suspend and resume allocates or logs: the thread may hold the heap or stderr lock
        CONTEXT c; memset(&c, 0, sizeof c); c.ContextFlags = CONTEXT_DEBUG_REGISTERS;
        if (GetThreadContext(h, &c)) {
            for (int i = 0; i < 2; i++) { uint64_t a = arm ? g_hw_a[i] : 0; c.Wcr[i] = wcr_of(a); c.Wvr[i] = a & ~7ull; }
            c.ContextFlags = CONTEXT_DEBUG_REGISTERS;
            if (SetThreadContext(h, &c)) {
                CONTEXT r; memset(&r, 0, sizeof r); r.ContextFlags = CONTEXT_DEBUG_REGISTERS;
                ok = GetThreadContext(h, &r) && r.Wcr[0] == c.Wcr[0] && r.Wvr[0] == c.Wvr[0] && r.Wcr[1] == c.Wcr[1] && r.Wvr[1] == c.Wvr[1];
                if (readback) *readback = r.Wcr[0];
            }
        }
        ResumeThread(h);
    }
    CloseHandle(h); return ok;
}
static void hw_all(int arm) {   // every thread of the process but this one (task_threads: darwin_compat.c, the thread list of a Toolhelp snapshot)
    thread_act_array_t list; mach_msg_type_number_t n = 0;
    if (task_threads(mach_task_self(), &list, &n) != KERN_SUCCESS) { LOG("[RED] [vk] hw watch: cannot list the threads"); return; }
    unsigned set = 0, failed = 0; uint32_t rb = 0; DWORD self = GetCurrentThreadId();
    for (unsigned i = 0; i < n; i++) if ((DWORD)list[i] != self) { if (hw_thread((DWORD)list[i], arm, &rb)) set++; else failed++; }
    vm_deallocate(mach_task_self(), (vm_address_t)list, 4096 * sizeof *list);
    static int said; if (said++ < 3 || failed) LOG("%s[vk] hw watch: %s on %u threads (%u failed), Wcr read back %#x", failed ? "[RED] " : "", arm ? "set" : "cleared", set, failed, rb);
}
static DWORD WINAPI hw_main(void *p) {
    (void)p; g_hw_tid = GetCurrentThreadId();
    for (;;) {
        DWORD r = WaitForSingleObject(g_hw_req, 1000);
        if (r == WAIT_OBJECT_0) { hw_thread(g_hw_req_tid, 0, NULL); SetEvent(g_hw_done); continue; }   // a hit: clear that thread's watchpoints so its store can go through
        if (InterlockedCompareExchange(&g_hw_on, 0, 0) == 1) {
            if (InterlockedCompareExchange(&g_hw_left, 0, 0) > 0) hw_all(1);
            else { hw_all(0); InterlockedExchange(&g_hw_on, 0); LOG("[vk] hw watch: done, watchpoints cleared"); }
        } else if (InterlockedCompareExchange(&g_hw_on, 2, 2) == 2) { hw_all(0); InterlockedExchange(&g_hw_on, 0); }   // turned off
    }
    return 0;
}
static void reg_name(char *o, size_t n, uint64_t a) {
    module_t *m = module_of((void *)(uintptr_t)a);
    if (m) snprintf(o, n, "%s+%#llx", m->name, (unsigned long long)(a - (uint64_t)(uintptr_t)m->base)); else snprintf(o, n, "%#llx", (unsigned long long)a);
}
static void val_of(char *o, size_t n, const CONTEXT *c, int vec, unsigned r) {
    uint64_t x; if (vec) memcpy(&x, &c->V[r], 8); else x = r == 31 ? 0 : (&c->X0)[r];
    uint32_t lo = (uint32_t)x; float f; memcpy(&f, &lo, 4);
    snprintf(o, n, "%s%u=%#llx (low word %#x = %g)", vec ? "v" : "x", r, (unsigned long long)x, lo, f);
}
static LONG CALLBACK hw_veh(EXCEPTION_POINTERS *ep) {
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (!InterlockedCompareExchange(&g_hw_on, 0, 0) || (code != EXCEPTION_SINGLE_STEP && code != EXCEPTION_BREAKPOINT)) return EXCEPTION_CONTINUE_SEARCH;
    CONTEXT *c = ep->ContextRecord; uint32_t insn = *(const uint32_t *)(uintptr_t)c->Pc;
    if ((insn & 0xffe0001fu) == 0xd4200000u || (insn & 0x0a000000u) != 0x08000000u) return EXCEPTION_CONTINUE_SEARCH;   // a BRK, or not a load / store: not ours
    LONG k = InterlockedDecrement(&g_hw_left);
    if (k >= 0) {
        char pc[96], lr[96], v1[96], v2[96]; reg_name(pc, sizeof pc, c->Pc); reg_name(lr, sizeof lr, c->Lr);
        int vec = (insn >> 26) & 1; val_of(v1, sizeof v1, c, vec, insn & 31); val_of(v2, sizeof v2, c, vec, (insn >> 10) & 31);
        float m[2]; memcpy(&m[0], (const void *)(uintptr_t)g_hw_a[0], 4); if (g_hw_a[1]) memcpy(&m[1], (const void *)(uintptr_t)g_hw_a[1], 4); else m[1] = 0;
        LOG("[vk] hw watch: store #%ld on thread %lu (code %#lx, %lu parameters, first %#llx %#llx): pc %s lr %s instruction %08x, Rt %s, Rt2 %s; memory now %g / %g",
            (long)(g_hw_n - k), (unsigned long)GetCurrentThreadId(), (unsigned long)code, (unsigned long)ep->ExceptionRecord->NumberParameters,
            (unsigned long long)(ep->ExceptionRecord->NumberParameters > 0 ? ep->ExceptionRecord->ExceptionInformation[0] : 0),
            (unsigned long long)(ep->ExceptionRecord->NumberParameters > 1 ? ep->ExceptionRecord->ExceptionInformation[1] : 0), pc, lr, insn, v1, v2, m[0], m[1]);
        char line[1400]; int o = 0;
        for (int i = 0; i < 29 && o < (int)sizeof line - 40; i++) o += snprintf(line + o, sizeof line - o, " x%d=%llx", i, (unsigned long long)(&c->X0)[i]);
        LOG("[vk] hw watch: registers%s fp=%llx sp=%llx", line, (unsigned long long)c->Fp, (unsigned long long)c->Sp);
        dm_backtrace(c->Fp);
    }
    AcquireSRWLockExclusive(&g_hw_req_lock);
    g_hw_req_tid = GetCurrentThreadId(); SetEvent(g_hw_req); WaitForSingleObject(g_hw_done, 2000);
    ReleaseSRWLockExclusive(&g_hw_req_lock);
    for (int i = 0; i < 2; i++) { c->Wcr[i] = 0; c->Wvr[i] = 0; }   // and the context this thread resumes with, should it carry the debug registers
    return EXCEPTION_CONTINUE_EXECUTION;
}
void dm_hwwatch_set(uint64_t a0, uint64_t a1, int n) {
    if (!g_hw_th) {
        g_hw_req = CreateEventW(NULL, FALSE, FALSE, NULL); g_hw_done = CreateEventW(NULL, FALSE, FALSE, NULL);
        g_hw_veh = AddVectoredExceptionHandler(1, hw_veh);   // first: before the host's (posix_win.c), which would take the hit for a crash
        g_hw_th = CreateThread(NULL, 0, hw_main, NULL, 0, NULL);
        if (!g_hw_th || !g_hw_veh) { LOG("[RED] [vk] hw watch: cannot start"); return; }
    }
    if (!a0) { InterlockedExchange(&g_hw_on, 2); LOG("[vk] hw watch: off"); return; }
    if (n <= 0) n = 16;
    g_hw_a[0] = a0; g_hw_a[1] = a1; g_hw_n = n; InterlockedExchange(&g_hw_left, n); InterlockedExchange(&g_hw_on, 1);
    LOG("[vk] hw watch: stores to %#llx and %#llx, the next %d logged (set on every thread within a second)", (unsigned long long)a0, (unsigned long long)a1, n);
}
