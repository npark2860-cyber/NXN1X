// Windows port: the POSIX calls the host makes that mingw-w64 lacks, on Win32 (headers in dm/windows/compat/include).
//   memory: anonymous mmap / mprotect / munmap on VirtualAlloc (see sys/mman.h for the mapping of the semantics)
//   signals: faults through a vectored exception handler, pthread_kill through a context redirect, alarm on a timer (see signal.h)
//   threads: the host keys of the rewritten code are TEB TLS slots (loader.c); thread names, stacks and IDs from the TEB / Win32
//   files: pread / pwrite on positioned ReadFile / WriteFile, extended attributes on alternate data streams, F_GETPATH, futimens
//   and the small libc additions (strlcpy, memmem, setenv, mergesort, arc4random, ...)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>
#include <psapi.h>
#include <mmsystem.h>
#include <io.h>
#include <direct.h>
#include <fcntl.h>
#include <errno.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/ucontext.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/xattr.h>
#include <dlfcn.h>
#include <time.h>
#include <unistd.h>

// ---------------------------------------------------------------- memory ----------------------------------------------------------------
static DWORD win_prot(int p) {
    switch (p & (PROT_READ | PROT_WRITE | PROT_EXEC)) {
    case PROT_NONE: return PAGE_NOACCESS;
    case PROT_READ: return PAGE_READONLY;
    case PROT_WRITE: case PROT_READ | PROT_WRITE: return PAGE_READWRITE;
    case PROT_EXEC: return PAGE_EXECUTE;
    case PROT_READ | PROT_EXEC: return PAGE_EXECUTE_READ;
    default: return PAGE_EXECUTE_READWRITE;
    }
}
static int set_errno_win(void) {
    DWORD e = GetLastError();
    errno = e == ERROR_NOT_ENOUGH_MEMORY || e == ERROR_COMMITMENT_LIMIT || e == ERROR_INVALID_ADDRESS ? ENOMEM : e == ERROR_ACCESS_DENIED ? EACCES : EINVAL;
    return -1;
}
// A file: a view of a file mapping object (the view keeps the mapping alive after its handle and the file descriptor are closed, as POSIX keeps a mapping
// after close); MAP_PRIVATE writes go to copy-on-write pages. Taken anywhere (addr and MAP_FIXED are not honoured for files: the host maps files only at an
// address the system picks, the Vulkan shader bundle being the one: gfx_vk_shader.c vk_shaders_open, 2.2 GB for DK)
static void *mmap_file(size_t len, int prot, int flags, int fd, off_t off) {
    HANDLE f = (HANDLE)_get_osfhandle(fd);
    if (f == INVALID_HANDLE_VALUE) { errno = EBADF; return MAP_FAILED; }
    int wr = (prot & PROT_WRITE) != 0, priv = (flags & MAP_PRIVATE) != 0, ex = (prot & PROT_EXEC) != 0;
    DWORD mp = wr && !priv ? (ex ? PAGE_EXECUTE_READWRITE : PAGE_READWRITE) : wr ? (ex ? PAGE_EXECUTE_WRITECOPY : PAGE_WRITECOPY) : ex ? PAGE_EXECUTE_READ : PAGE_READONLY;
    DWORD acc = wr ? (priv ? FILE_MAP_COPY : FILE_MAP_WRITE) : FILE_MAP_READ; if (ex) acc |= FILE_MAP_EXECUTE;
    HANDLE m = CreateFileMappingW(f, NULL, mp, 0, 0, NULL);   // the file's size (a view past its end fails)
    if (!m) { set_errno_win(); return MAP_FAILED; }
    uint64_t o = (uint64_t)off;
    void *p = MapViewOfFile(m, acc, (DWORD)(o >> 32), (DWORD)o, len);
    if (!p) set_errno_win();
    CloseHandle(m);
    return p ? p : MAP_FAILED;
}
void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    if (!(flags & MAP_ANON) && fd != -1) return len ? mmap_file(len, prot, flags, fd, off) : (errno = EINVAL, MAP_FAILED);
    (void)off;
    if (fd != -1 || !(flags & MAP_ANON)) { errno = ENODEV; return MAP_FAILED; }
    if (!len) { errno = EINVAL; return MAP_FAILED; }
    DWORD wp = win_prot(prot);
    if (flags & MAP_FIXED) {
        MEMORY_BASIC_INFORMATION mi;
        if (VirtualQuery(addr, &mi, sizeof mi) && mi.State != MEM_FREE) {   // inside a reservation of ours: the range becomes fresh zero pages
            if (!VirtualFree(addr, len, MEM_DECOMMIT)) { set_errno_win(); return MAP_FAILED; }
            if (prot != PROT_NONE && !VirtualAlloc(addr, len, MEM_COMMIT, wp)) { set_errno_win(); return MAP_FAILED; }
            return addr;
        }
        void *p = VirtualAlloc(addr, len, MEM_RESERVE | (prot != PROT_NONE ? MEM_COMMIT : 0), wp);
        if (p != addr) { if (p) VirtualFree(p, 0, MEM_RELEASE); errno = ENOMEM; return MAP_FAILED; }
        return p;
    }
    DWORD type = MEM_RESERVE | (prot != PROT_NONE ? MEM_COMMIT : 0);
    void *p = addr ? VirtualAlloc(addr, len, type, wp) : NULL;   // a hint: taken when free
    if (!p) p = VirtualAlloc(NULL, len, type, wp);
    if (!p) { set_errno_win(); return MAP_FAILED; }
    return p;
}
// Walk [a, a + len) region by region (VirtualQuery): fn gets each piece with its region information; a free piece fails with ENOMEM
static int each_region(void *addr, size_t len, int (*fn)(uint8_t *p, size_t n, const MEMORY_BASIC_INFORMATION *mi, void *ctx), void *ctx) {
    uint8_t *a = addr, *end = a + len;
    while (a < end) {
        MEMORY_BASIC_INFORMATION mi;
        if (!VirtualQuery(a, &mi, sizeof mi)) { errno = ENOMEM; return -1; }
        uint8_t *rend = (uint8_t *)mi.BaseAddress + mi.RegionSize; if (rend > end) rend = end;
        if (mi.State == MEM_FREE) { errno = ENOMEM; return -1; }
        if (fn(a, (size_t)(rend - a), &mi, ctx)) return -1;
        a = rend;
    }
    return 0;
}
static int unmap_piece(uint8_t *p, size_t n, const MEMORY_BASIC_INFORMATION *mi, void *ctx) {
    (void)ctx;
    if (mi->State == MEM_COMMIT && !VirtualFree(p, n, MEM_DECOMMIT)) return set_errno_win();
    return 0;
}
int munmap(void *addr, size_t len) {
    MEMORY_BASIC_INFORMATION mi;
    if (!len || !VirtualQuery(addr, &mi, sizeof mi)) { errno = EINVAL; return -1; }
    if (mi.State == MEM_FREE) return 0;
    if (mi.Type == MEM_MAPPED) { if (mi.AllocationBase != addr) { errno = EINVAL; return -1; }   // a file view (mmap_file): unmapped whole
        return UnmapViewOfFile(addr) ? 0 : set_errno_win(); }
    if (mi.AllocationBase == addr) {   // the unmap starts at an allocation: release it when the range covers all of it
        size_t total = 0; MEMORY_BASIC_INFORMATION r;
        for (uint8_t *q = addr; VirtualQuery(q, &r, sizeof r) && r.AllocationBase == addr && r.State != MEM_FREE; q += r.RegionSize) total += r.RegionSize;
        if (len >= total) {
            if (!VirtualFree(addr, 0, MEM_RELEASE)) return set_errno_win();
            if (len > total) return munmap((uint8_t *)addr + total, len - total);
            return 0;
        }
    }
    return each_region(addr, len, unmap_piece, NULL);   // a part: decommitted, the reservation stays
}
static int protect_piece(uint8_t *p, size_t n, const MEMORY_BASIC_INFORMATION *mi, void *ctx) {
    int prot = *(int *)ctx; DWORD wp = win_prot(prot), old;
    if (mi->State == MEM_RESERVE) { if (prot != PROT_NONE && !VirtualAlloc(p, n, MEM_COMMIT, wp)) return set_errno_win(); return 0; }
    if (!VirtualProtect(p, n, wp, &old)) return set_errno_win();
    return 0;
}
int mprotect(void *addr, size_t len, int prot) {
    DWORD old;
    if (!len) return 0;
    int r = VirtualProtect(addr, len, win_prot(prot), &old) ? 0 : each_region(addr, len, protect_piece, &prot);   // first: the common case, committed pages of one allocation
    if (!r && (prot & PROT_EXEC)) FlushInstructionCache(GetCurrentProcess(), addr, len);   // code just copied in (loader.c map_image) must reach instruction fetch
    return r;
}
static int dontneed_piece(uint8_t *p, size_t n, const MEMORY_BASIC_INFORMATION *mi, void *ctx) {
    (void)ctx;
    if (mi->State != MEM_COMMIT) return 0;
    DWORD keep = mi->Protect;
    if (!VirtualFree(p, n, MEM_DECOMMIT) || !VirtualAlloc(p, n, MEM_COMMIT, keep)) return set_errno_win();
    return 0;
}
static int reset_piece(uint8_t *p, size_t n, const MEMORY_BASIC_INFORMATION *mi, void *ctx) {
    (void)ctx; if (mi->State == MEM_COMMIT) VirtualAlloc(p, n, MEM_RESET, PAGE_NOACCESS); return 0;
}
int madvise(void *addr, size_t len, int advice) {
    if (advice == MADV_DONTNEED) return each_region(addr, len, dontneed_piece, NULL);   // reads as zeros afterwards, as on Linux
    if (advice == MADV_FREE || advice == MADV_FREE_REUSABLE) return each_region(addr, len, reset_piece, NULL);
    return 0;
}
int msync(void *addr, size_t len, int flags) { (void)addr; (void)len; (void)flags; return 0; }
int mlock(const void *addr, size_t len) { (void)addr; (void)len; return 0; }
int munlock(const void *addr, size_t len) { (void)addr; (void)len; return 0; }

// ---------------------------------------------------------------- signals ----------------------------------------------------------------
static struct sigaction g_sa[DM_NSIG];
static SRWLOCK g_sa_lock = SRWLOCK_INIT;
static __thread int t_pass, t_final, t_depth;
static int real_stack(NT_TIB *tib, uint8_t **lo, uint8_t **hi);   // the thread's own stack (threads section)
static __thread uint8_t *t_alt; static __thread size_t t_alt_n;   // sigaltstack: the handlers with SA_ONSTACK run there (crash.c gives each thread 64 KiB)
// A line straight to stderr, for the dispatcher's own failures (no CRT stream, no allocation)
static void raw_err(const char *fmt, ...) {
    char b[256]; va_list ap; va_start(ap, fmt); int n = vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
    DWORD w; if (n > 0) WriteFile(GetStdHandle(STD_ERROR_HANDLE), b, (DWORD)(n < (int)sizeof b ? n : (int)sizeof b - 1), &w, NULL);
}
// Call fn(arg) with sp at top (16-byte aligned), and come back to this stack
__attribute__((naked)) static void call_on_stack(void (*fn)(void *), void *arg, void *top) {
    __asm__ volatile("stp x29, x30, [sp, #-16]!\n mov x29, sp\n mov x3, sp\n and x2, x2, #~15\n mov sp, x2\n str x3, [sp, #-16]!\n"
                     "mov x9, x0\n mov x0, x1\n blr x9\n ldr x3, [sp], #16\n mov sp, x3\n ldp x29, x30, [sp], #16\n ret\n");
}

static void ctx_to_uc(const CONTEXT *c, ucontext_t *uc, uint64_t fault, uint32_t esr) {
    memset(uc, 0, offsetof(ucontext_t, __mcontext_data));
    struct __darwin_mcontext64 *m = &uc->__mcontext_data; uc->uc_mcontext = m; uc->uc_mcsize = sizeof *m;
    m->__es.__far = fault; m->__es.__esr = esr; m->__es.__exception = 0;
    memcpy(m->__ss.__x, &c->X0, 29 * 8);
    m->__ss.__fp = c->Fp; m->__ss.__lr = c->Lr; m->__ss.__sp = c->Sp; m->__ss.__pc = c->Pc; m->__ss.__cpsr = c->Cpsr; m->__ss.__pad = 0;
    memcpy(m->__ns.__v, c->V, sizeof m->__ns.__v);
    m->__ns.__fpsr = c->Fpsr; m->__ns.__fpcr = c->Fpcr;
}
static void uc_to_ctx(const ucontext_t *uc, CONTEXT *c) {   // x18 stays the thread's (the TEB): no handler has reason to move it
    const struct __darwin_mcontext64 *m = uc->uc_mcontext;
    DWORD64 x18 = c->X18;
    memcpy(&c->X0, m->__ss.__x, 29 * 8); c->X18 = x18;
    c->Fp = m->__ss.__fp; c->Lr = m->__ss.__lr; c->Sp = m->__ss.__sp; c->Pc = m->__ss.__pc; c->Cpsr = m->__ss.__cpsr;
    memcpy(c->V, m->__ns.__v, sizeof m->__ns.__v);
    c->Fpsr = m->__ns.__fpsr; c->Fpcr = m->__ns.__fpcr;
}
static int is_brk(uint64_t pc) {   // BRK #imm: Windows reports its own (#0xf000) as a breakpoint and may report others as illegal instructions
    MEMORY_BASIC_INFORMATION mi;
    if (!VirtualQuery((void *)pc, &mi, sizeof mi) || mi.State != MEM_COMMIT || (mi.Protect & (PAGE_NOACCESS | PAGE_GUARD))) return 0;
    return (*(const uint32_t *)pc & 0xffe0001fu) == 0xd4200000u;
}
// The signal an exception stands for (0: none, the exception goes on as it is), with the fault address and a made-up ESR
static int sig_of(const EXCEPTION_RECORD *er, uint64_t pc, siginfo_t *si, uint32_t *esr) {
    memset(si, 0, sizeof *si); *esr = 0;
    switch (er->ExceptionCode) {
    case EXCEPTION_ACCESS_VIOLATION: {
        ULONG_PTR kind = er->NumberParameters >= 2 ? er->ExceptionInformation[0] : 0;
        si->si_addr = (void *)(er->NumberParameters >= 2 ? er->ExceptionInformation[1] : 0); si->si_code = SEGV_ACCERR;
        if (kind == 8) *esr = (0x20u << 26) | (1u << 25);                       // instruction abort
        else *esr = (0x24u << 26) | (1u << 25) | (kind == 1 ? 1u << 6 : 0);    // data abort, WnR for a write
        return SIGSEGV; }
    case EXCEPTION_IN_PAGE_ERROR:
        si->si_addr = (void *)(er->NumberParameters >= 2 ? er->ExceptionInformation[1] : 0); *esr = (0x24u << 26) | (1u << 25); return SIGBUS;
    case EXCEPTION_DATATYPE_MISALIGNMENT: si->si_addr = er->ExceptionAddress; si->si_code = BUS_ADRALN; return SIGBUS;
    case EXCEPTION_STACK_OVERFLOW: si->si_addr = (void *)(er->NumberParameters >= 2 ? er->ExceptionInformation[1] : 0); return SIGSEGV;
    case EXCEPTION_BREAKPOINT: case EXCEPTION_SINGLE_STEP: si->si_addr = er->ExceptionAddress; si->si_code = TRAP_BRKPT; return SIGTRAP;
    case EXCEPTION_ILLEGAL_INSTRUCTION: case EXCEPTION_PRIV_INSTRUCTION:
        si->si_addr = er->ExceptionAddress;
        if (is_brk(pc)) { si->si_code = TRAP_BRKPT; return SIGTRAP; }
        si->si_code = ILL_ILLOPC; return SIGILL;
    case EXCEPTION_FLT_DIVIDE_BY_ZERO: case EXCEPTION_FLT_INVALID_OPERATION: case EXCEPTION_FLT_OVERFLOW: case EXCEPTION_FLT_UNDERFLOW:
    case EXCEPTION_FLT_INEXACT_RESULT: case EXCEPTION_FLT_DENORMAL_OPERAND: case EXCEPTION_FLT_STACK_CHECK: case EXCEPTION_INT_DIVIDE_BY_ZERO: case EXCEPTION_INT_OVERFLOW:
        si->si_addr = er->ExceptionAddress; return SIGFPE;
    default: return 0;
    }
}
static int has_handler(int sig, struct sigaction *out) {
    AcquireSRWLockShared(&g_sa_lock); *out = g_sa[sig]; ReleaseSRWLockShared(&g_sa_lock);
    return out->sa_handler != SIG_DFL && out->sa_handler != SIG_IGN && out->sa_handler != NULL;
}
static void on_alt(void *p) {
    struct { const struct sigaction *sa; int sig; siginfo_t *si; ucontext_t *uc; } *a = p;
    if (a->sa->sa_flags & SA_SIGINFO) a->sa->sa_sigaction(a->sig, a->si, a->uc); else a->sa->sa_handler(a->sig);
}
// Run the handler on this thread with c as its context; 1 = the handler resumed (c now holds its changes), 0 = it asked to pass the fault on
static int run_handler(int sig, const struct sigaction *sa, siginfo_t *si, CONTEXT *c, uint32_t esr, int final) {
    ucontext_t uc; ctx_to_uc(c, &uc, (uint64_t)(uintptr_t)si->si_addr, esr);
    si->si_signo = sig;
    int pass0 = t_pass, final0 = t_final; t_pass = 0; t_final = final;
    uint8_t *sp = (uint8_t *)__builtin_frame_address(0);
    if ((sa->sa_flags & SA_ONSTACK) && t_alt && !(sp >= t_alt && sp < t_alt + t_alt_n)) {   // not already on it (a fault inside the handler)
        struct { const struct sigaction *sa; int sig; siginfo_t *si; ucontext_t *uc; } a = {sa, sig, si, &uc};
        call_on_stack(on_alt, &a, t_alt + t_alt_n);
    } else if (sa->sa_flags & SA_SIGINFO) sa->sa_sigaction(sig, si, &uc); else sa->sa_handler(sig);
    int pass = t_pass; t_pass = pass0; t_final = final0;
    if (pass) return 0;
    uc_to_ctx(&uc, c);
    return 1;
}
static LONG CALLBACK veh(EXCEPTION_POINTERS *ep) {
    siginfo_t si; uint32_t esr; struct sigaction sa;
    int sig = sig_of(ep->ExceptionRecord, ep->ContextRecord->Pc, &si, &esr);
    if (!sig || !has_handler(sig, &sa)) return EXCEPTION_CONTINUE_SEARCH;
    // A fault inside a handler: say so before it goes on, as nothing else will (the dispatcher cannot walk a guest or alternate stack, and the process ends without a word)
    if (t_depth) raw_err("[dm] fault inside the signal handler (depth %d): code %#lx pc %#llx sp %#llx address %p\n", t_depth, (unsigned long)ep->ExceptionRecord->ExceptionCode,
                         (unsigned long long)ep->ContextRecord->Pc, (unsigned long long)ep->ContextRecord->Sp, si.si_addr);
    t_depth++;
    int r = run_handler(sig, &sa, &si, ep->ContextRecord, esr, 0);
    t_depth--;
    return r ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}
static LONG WINAPI unhandled(EXCEPTION_POINTERS *ep) {   // a fault passed on that nobody handled: the host's handler gets it as final
    siginfo_t si; uint32_t esr; struct sigaction sa;
    int sig = sig_of(ep->ExceptionRecord, ep->ContextRecord->Pc, &si, &esr);
    if (!sig || !has_handler(sig, &sa)) return EXCEPTION_CONTINUE_SEARCH;
    return run_handler(sig, &sa, &si, ep->ContextRecord, esr, 1) ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}
int dm_win_pass_fault(void *ucv) {
    if (t_final) return 0;
    ucontext_t *uc = ucv; HMODULE m = NULL;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCWSTR)(uintptr_t)uc->uc_mcontext->__ss.__pc, &m) ||
        !m || m == GetModuleHandleW(NULL)) return 0;   // guest code (no module), or the host itself
    // The DLL's handlers are found by walking the stack, which Windows does only within the thread's own stack (TEB limits); on a guest stack the walk fails and the
    // process ends silently, so such a fault is the host's to report
    uint8_t *lo, *hi; uint64_t sp = uc->uc_mcontext->__ss.__sp;
    char name[MAX_PATH]; if (!GetModuleFileNameA(m, name, sizeof name)) strcpy(name, "?");
    if (!real_stack((NT_TIB *)NtCurrentTeb(), &lo, &hi) || sp < (uint64_t)(uintptr_t)lo || sp >= (uint64_t)(uintptr_t)hi) {   // the thread's own stack, not the opened TEB bounds
        raw_err("[dm] fault in %s +%#llx on a stack Windows does not know (sp %#llx): reported here\n", name, (unsigned long long)(uc->uc_mcontext->__ss.__pc - (uint64_t)(uintptr_t)m), (unsigned long long)sp);
        return 0;
    }
    raw_err("[dm] fault in %s +%#llx: passed on to its own handlers\n", name, (unsigned long long)(uc->uc_mcontext->__ss.__pc - (uint64_t)(uintptr_t)m));
    t_pass = 1; return 1;
}
// A signal raised on the calling thread (raise, alarm, abort through the CRT, pthread_kill to oneself): the handler sees this thread's registers at the call
static void raise_here(int sig) {
    struct sigaction sa; if (!has_handler(sig, &sa)) return;
    CONTEXT c; memset(&c, 0, sizeof c); c.ContextFlags = CONTEXT_FULL; RtlCaptureContext(&c);
    siginfo_t si; memset(&si, 0, sizeof si); si.si_pid = (int)GetCurrentProcessId();
    run_handler(sig, &sa, &si, &c, 0, 1);
}
static void __cdecl crt_signal(int sig) { signal(sig, crt_signal); raise_here(sig); }
static INIT_ONCE g_veh_once = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK veh_install(PINIT_ONCE o, PVOID p, PVOID *ctx) {
    (void)o; (void)p; (void)ctx;
    AddVectoredExceptionHandler(1, veh); SetUnhandledExceptionFilter(unhandled);
    return TRUE;
}
int sigaction(int sig, const struct sigaction *sa, struct sigaction *old) {
    if (sig <= 0 || sig >= DM_NSIG) { errno = EINVAL; return -1; }
    AcquireSRWLockExclusive(&g_sa_lock);
    if (old) *old = g_sa[sig];
    if (sa) g_sa[sig] = *sa;
    ReleaseSRWLockExclusive(&g_sa_lock);
    if (sa) {
        if (sig == SIGSEGV || sig == SIGBUS || sig == SIGILL || sig == SIGTRAP || sig == SIGFPE) InitOnceExecuteOnce(&g_veh_once, veh_install, NULL, NULL);
        if (sig == SIGABRT || sig == SIGINT || sig == SIGTERM) signal(sig, sa->sa_handler == SIG_IGN ? SIG_IGN : sa->sa_handler == SIG_DFL ? SIG_DFL : crt_signal);
    }
    return 0;
}
int sigaltstack(const stack_t *ss, stack_t *old) {
    if (old) { memset(old, 0, sizeof *old); old->ss_sp = t_alt; old->ss_size = t_alt_n; old->ss_flags = t_alt ? 0 : SS_DISABLE; }
    if (ss) { if (ss->ss_flags & SS_DISABLE) { t_alt = NULL; t_alt_n = 0; } else { t_alt = ss->ss_sp; t_alt_n = ss->ss_size; } }
    return 0;
}

// pthread_kill: the target is suspended and pointed at sig_trampoline with a frame (its saved registers) pushed on its own stack; the trampoline runs the handler there
// and returns to the saved registers with RtlRestoreContext. A target inside a system call (its pc in ntdll) is let run a little and tried again: changing the
// registers of a thread in the kernel would lose the call's result
typedef struct { CONTEXT ctx; int sig; } sigframe_t;
static void sig_trampoline(sigframe_t *f) {
    struct sigaction sa;
    if (has_handler(f->sig, &sa)) { siginfo_t si; memset(&si, 0, sizeof si); si.si_pid = (int)GetCurrentProcessId(); run_handler(f->sig, &sa, &si, &f->ctx, 0, 1); }
    RtlRestoreContext(&f->ctx, NULL);
}
static int in_ntdll(uint64_t pc) {
    static HMODULE nt; static uint64_t lo, hi;
    if (!nt) { nt = GetModuleHandleW(L"ntdll.dll"); MODULEINFO mi; if (nt && K32GetModuleInformation(GetCurrentProcess(), nt, &mi, sizeof mi)) { lo = (uint64_t)(uintptr_t)mi.lpBaseOfDll; hi = lo + mi.SizeOfImage; } }
    return pc >= lo && pc < hi;
}
int dm_pthread_kill(pthread_t t, int sig) {
    if (sig == 0) return 0;
    if (sig < 0 || sig >= DM_NSIG) return EINVAL;
    if (t == pthread_self()) { raise_here(sig); return 0; }
    HANDLE h = pthread_gethandle(t); if (!h) return ESRCH;
    struct sigaction sa; if (!has_handler(sig, &sa)) return 0;
    for (int tries = 0;; tries++) {
        if (SuspendThread(h) == (DWORD)-1) return ESRCH;
        CONTEXT c; memset(&c, 0, sizeof c); c.ContextFlags = CONTEXT_FULL;
        if (!GetThreadContext(h, &c)) { ResumeThread(h); return ESRCH; }
        if (in_ntdll(c.Pc) && tries < 200) { ResumeThread(h); Sleep(tries < 20 ? 0 : 1); continue; }
        uint64_t sp = (c.Sp - sizeof(sigframe_t) - 64) & ~15ull;
        sigframe_t *f = (sigframe_t *)(uintptr_t)sp;
        memcpy(&f->ctx, &c, sizeof c); f->sig = sig;
        c.X0 = sp; c.Sp = sp; c.Pc = (DWORD64)(uintptr_t)sig_trampoline; c.Lr = 0; c.Fp = 0;
        BOOL ok = SetThreadContext(h, &c);
        ResumeThread(h);
        return ok ? 0 : ESRCH;
    }
}
int kill(pid_t pid, int sig) {
    if (pid == 0 || pid == (pid_t)GetCurrentProcessId()) { if (sig == SIGKILL) TerminateProcess(GetCurrentProcess(), 128 + SIGKILL); raise_here(sig); return 0; }
    if (sig == 0) { HANDLE p = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)pid); if (!p) { errno = ESRCH; return -1; } CloseHandle(p); return 0; }
    HANDLE p = OpenProcess(PROCESS_TERMINATE, FALSE, (DWORD)pid); if (!p) { errno = ESRCH; return -1; }
    BOOL ok = TerminateProcess(p, 128 + (unsigned)sig); CloseHandle(p); return ok ? 0 : (errno = EPERM, -1);
}
static HANDLE g_alarm; static ULONGLONG g_alarm_at;
static VOID CALLBACK alarm_fire(PVOID p, BOOLEAN fired) { (void)p; (void)fired; raise_here(SIGALRM); }
unsigned alarm(unsigned seconds) {
    unsigned left = 0; ULONGLONG now = GetTickCount64();
    if (g_alarm) { if (g_alarm_at > now) left = (unsigned)((g_alarm_at - now + 999) / 1000); DeleteTimerQueueTimer(NULL, g_alarm, NULL); g_alarm = NULL; }
    if (seconds) { g_alarm_at = now + seconds * 1000ull; CreateTimerQueueTimer(&g_alarm, NULL, alarm_fire, NULL, seconds * 1000u, 0, WT_EXECUTEONLYONCE | WT_EXECUTELONGFUNCTION); }
    return left;
}
int pause(void) { for (;;) Sleep(INFINITE); }

// ---------------------------------------------------------------- threads ----------------------------------------------------------------
unsigned dm_tls_alloc(void) { DWORD i = TlsAlloc(); return i == TLS_OUT_OF_INDEXES ? ~0u : (unsigned)i; }
void dm_tsd_set(pthread_key_t k, const void *v) { TlsSetValue((DWORD)k, (LPVOID)v); }
void *dm_tsd_get(pthread_key_t k) { return TlsGetValue((DWORD)k); }
int dm_setname_self(const char *name) {
    typedef HRESULT (WINAPI *setdesc_fn)(HANDLE, PCWSTR);
    static setdesc_fn fn; static int looked;
    if (!looked) { fn = (setdesc_fn)(void *)GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadDescription"); looked = 1; }
    if (!fn) return 0;
    WCHAR w[128]; if (!MultiByteToWideChar(CP_UTF8, 0, name, -1, w, 128)) return EINVAL;
    return SUCCEEDED(fn(GetCurrentThread(), w)) ? 0 : EINVAL;
}
// another thread's TEB (NtQueryInformationThread, ThreadBasicInformation)
typedef struct { LONG ExitStatus; PVOID TebBaseAddress; ULONG_PTR ClientId[2]; ULONG_PTR AffinityMask; LONG Priority, BasePriority; } dm_tbi;
static NT_TIB *tib_of(pthread_t t) {
    if (t == pthread_self()) return (NT_TIB *)NtCurrentTeb();
    typedef LONG (NTAPI *qit_fn)(HANDLE, int, PVOID, ULONG, PULONG);
    static qit_fn q; if (!q) q = (qit_fn)(void *)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationThread");
    HANDLE h = pthread_gethandle(t); dm_tbi bi;
    if (!q || !h || q(h, 0, &bi, sizeof bi, NULL) < 0) return NULL;
    return (NT_TIB *)bi.TebBaseAddress;
}
// A thread's own stack: its whole reservation, from the TEB's DeallocationStack (+0x1478, the word before the TLS slots) to the reservation's end. The TEB's StackBase /
// StackLimit cannot say it: they are opened to all of user space (stack_open below)
#define TEB_DEALLOCATION_STACK 0x1478
static int real_stack(NT_TIB *tib, uint8_t **lo, uint8_t **hi) {
    uint8_t *base = *(uint8_t **)((uint8_t *)tib + TEB_DEALLOCATION_STACK), *q = base; MEMORY_BASIC_INFORMATION mi;
    if (!base) return 0;
    while (VirtualQuery(q, &mi, sizeof mi) && mi.AllocationBase == base && mi.State != MEM_FREE) q = (uint8_t *)mi.BaseAddress + mi.RegionSize;
    if (q == base) return 0;
    *lo = base; *hi = q; return 1;
}
void *pthread_get_stackaddr_np(pthread_t t) { NT_TIB *tib = tib_of(t); uint8_t *lo, *hi; return tib && real_stack(tib, &lo, &hi) ? hi : NULL; }
size_t pthread_get_stacksize_np(pthread_t t) { NT_TIB *tib = tib_of(t); uint8_t *lo, *hi; return tib && real_stack(tib, &lo, &hi) ? (size_t)(hi - lo) : 0; }   // the whole reservation, as Darwin's size covers the thread's whole stack
// Windows delivers an exception to user mode, and continues a context (NtContinue, RtlRestoreContext), only when sp lies within the TEB's StackLimit..StackBase
// (TotK donor-NCE: "Windows must validate the first context against guest StackLimit"); otherwise the process ends at once, before any handler, without a word
// (DK stage 2, 11-03: a fault on a guest thread right after nnMain's add-on query). The host runs code on stacks Windows did not make: guest thread stacks (in the
// guest address space), gate stacks and the large host stack (mapped), sigaltstack (malloc). So every thread opens its TEB bounds to all of user space when it
// starts (TLS callback below); the thread's own stack is still known through DeallocationStack (real_stack). Its whole reservation is committed first: Windows'
// own __chkstk probes only below StackLimit, so with StackLimit opened a large frame in a system DLL would step past the guard page that grows the stack
static void stack_open(void) {
    NT_TIB *tib = (NT_TIB *)NtCurrentTeb(); uint8_t *lo, *hi; DWORD old;
    if (real_stack(tib, &lo, &hi) && VirtualAlloc(lo, (size_t)(hi - lo), MEM_COMMIT, PAGE_READWRITE)) VirtualProtect(lo, (size_t)(hi - lo), PAGE_READWRITE, &old);   // also lifts the guard page
    tib->StackLimit = (void *)(uintptr_t)0x10000; tib->StackBase = (void *)(uintptr_t)0x800000000000ull;
}
static void NTAPI tls_cb(PVOID h, DWORD reason, PVOID r) { (void)h; (void)r; if (reason == DLL_PROCESS_ATTACH || reason == DLL_THREAD_ATTACH) stack_open(); }
__attribute__((used, section(".CRT$XLD"))) static const PIMAGE_TLS_CALLBACK g_tls_cb = tls_cb;
int pthread_threadid_np(pthread_t t, uint64_t *id) {
    if (!t || t == pthread_self()) { *id = GetCurrentThreadId(); return 0; }
    HANDLE h = pthread_gethandle(t); if (!h) return ESRCH;
    *id = GetThreadId(h); return 0;
}
static void (*g_ctrl_cb)(const char *why);
static BOOL WINAPI on_ctrl(DWORD ev) {
    if (!g_ctrl_cb) return FALSE;
    g_ctrl_cb(ev == CTRL_C_EVENT ? "Ctrl+C" : ev == CTRL_BREAK_EVENT ? "Ctrl+Break" : ev == CTRL_CLOSE_EVENT ? "console close" : ev == CTRL_LOGOFF_EVENT ? "logoff" : "shutdown");
    Sleep(INFINITE);   // the callback ends the process
    return TRUE;
}
// Darwin's wait with a relative timeout: the deadline on the clock winpthreads' timed wait measures (CLOCK_REALTIME)
int dm_cond_timedwait_rel(pthread_cond_t *c, pthread_mutex_t *m, const struct timespec *rel) {
    struct timespec t; clock_gettime(CLOCK_REALTIME, &t);
    t.tv_sec += rel->tv_sec; t.tv_nsec += rel->tv_nsec; if (t.tv_nsec >= 1000000000) { t.tv_sec++; t.tv_nsec -= 1000000000; }
    return pthread_cond_timedwait(c, m, &t);
}
int dm_console_ctrl_install(void (*fn)(const char *why)) { g_ctrl_cb = fn; return SetConsoleCtrlHandler(on_ctrl, TRUE) ? 0 : -1; }

// ---------------------------------------------------------------- unistd ----------------------------------------------------------------
long sysconf(int name) {
    SYSTEM_INFO si; GetSystemInfo(&si);
    switch (name) {
    case _SC_PAGESIZE: return (long)si.dwPageSize;
    case _SC_NPROCESSORS_CONF: case _SC_NPROCESSORS_ONLN: return (long)si.dwNumberOfProcessors;
    case _SC_OPEN_MAX: return 8192;
    case _SC_CLK_TCK: return 100;
    case _SC_PHYS_PAGES: { MEMORYSTATUSEX ms = {.dwLength = sizeof ms}; GlobalMemoryStatusEx(&ms); return (long)(ms.ullTotalPhys / si.dwPageSize); }
    default: errno = EINVAL; return -1;
    }
}
int getpagesize(void) { return 4096; }
// Positioned I/O: the offset goes in the OVERLAPPED. On a descriptor opened for synchronous I/O (the CRT's) Windows also moves the file position; the host reads
// its RomFS and save files with pread only, so nothing reads the position
ssize_t pread(int fd, void *buf, size_t n, off_t off) {
    HANDLE h = (HANDLE)_get_osfhandle(fd); if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
    if (n > 0x7ffff000u) n = 0x7ffff000u;
    OVERLAPPED o; memset(&o, 0, sizeof o); o.Offset = (DWORD)((uint64_t)off & 0xffffffffu); o.OffsetHigh = (DWORD)((uint64_t)off >> 32);
    DWORD got = 0;
    if (!ReadFile(h, buf, (DWORD)n, &got, &o)) { DWORD e = GetLastError(); if (e == ERROR_HANDLE_EOF) return 0; errno = e == ERROR_INVALID_USER_BUFFER || e == ERROR_NOACCESS ? EFAULT : EIO; return -1; }
    return (ssize_t)got;
}
ssize_t pwrite(int fd, const void *buf, size_t n, off_t off) {
    HANDLE h = (HANDLE)_get_osfhandle(fd); if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
    if (n > 0x7ffff000u) n = 0x7ffff000u;
    OVERLAPPED o; memset(&o, 0, sizeof o); o.Offset = (DWORD)((uint64_t)off & 0xffffffffu); o.OffsetHigh = (DWORD)((uint64_t)off >> 32);
    DWORD put = 0;
    if (!WriteFile(h, buf, (DWORD)n, &put, &o)) { errno = GetLastError() == ERROR_DISK_FULL ? ENOSPC : EIO; return -1; }
    return (ssize_t)put;
}
int fsync(int fd) { return _commit(fd); }
int symlink(const char *target, const char *path) {
    DWORD a = GetFileAttributesA(target);
    if (CreateSymbolicLinkA(path, target, (a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY) ? SYMBOLIC_LINK_FLAG_DIRECTORY : 0) | SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE)) return 0;
    errno = EPERM; return -1;
}
ssize_t readlink(const char *path, char *buf, size_t cap) { (void)path; (void)buf; (void)cap; errno = EINVAL; return -1; }
int link(const char *from, const char *to) { if (CreateHardLinkA(to, from, NULL)) return 0; errno = GetLastError() == ERROR_ALREADY_EXISTS ? EEXIST : EPERM; return -1; }
int fcntl(int fd, int cmd, ...) {
    if (cmd == F_GETPATH) {
        va_list ap; va_start(ap, cmd); char *out = va_arg(ap, char *); va_end(ap);
        HANDLE h = (HANDLE)_get_osfhandle(fd); if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
        char b[1100]; DWORD n = GetFinalPathNameByHandleA(h, b, sizeof b, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
        if (!n || n >= sizeof b) { errno = EBADF; return -1; }
        const char *p = strncmp(b, "\\\\?\\", 4) ? b : b + 4;
        snprintf(out, 1024, "%s", p); return 0;
    }
    if (cmd == F_FULLFSYNC) return _commit(fd);
    return (HANDLE)_get_osfhandle(fd) == INVALID_HANDLE_VALUE ? (errno = EBADF, -1) : 0;
}
static FILETIME ft_of(const struct timespec *t) {
    uint64_t v = ((uint64_t)t->tv_sec + 11644473600ull) * 10000000ull + (uint64_t)t->tv_nsec / 100; FILETIME f = {(DWORD)v, (DWORD)(v >> 32)}; return f;
}
int futimens(int fd, const struct timespec t[2]) {
    HANDLE h = (HANDLE)_get_osfhandle(fd); if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
    FILETIME a, m, now; GetSystemTimeAsFileTime(&now);
    const FILETIME *pa = NULL, *pm = NULL;
    if (!t) pa = pm = &now;
    else {
        if (t[0].tv_nsec != UTIME_OMIT) { a = t[0].tv_nsec == UTIME_NOW ? now : ft_of(&t[0]); pa = &a; }
        if (t[1].tv_nsec != UTIME_OMIT) { m = t[1].tv_nsec == UTIME_NOW ? now : ft_of(&t[1]); pm = &m; }
    }
    if (!SetFileTime(h, NULL, pa, pm)) { errno = EPERM; return -1; }
    return 0;
}

// ---------------------------------------------------------------- extended attributes (alternate data streams) ----------------------------------------------------------------
static int ads_path(char *out, size_t cap, const char *path, const char *name) {
    if (strchr(name, ':') || strchr(name, '\\') || strchr(name, '/')) { errno = EINVAL; return -1; }
    if ((size_t)snprintf(out, cap, "%s:%s", path, name) >= cap) { errno = ENAMETOOLONG; return -1; }
    return 0;
}
static int ads_errno(void) { DWORD e = GetLastError(); errno = e == ERROR_FILE_NOT_FOUND ? ENOATTR : e == ERROR_PATH_NOT_FOUND ? ENOENT : e == ERROR_ACCESS_DENIED ? EACCES : EIO; return -1; }
ssize_t getxattr(const char *path, const char *name, void *value, size_t size, uint32_t pos, int opt) {
    (void)pos; (void)opt; char p[1400]; if (ads_path(p, sizeof p, path, name)) return -1;
    HANDLE h = CreateFileA(p, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
    if (h == INVALID_HANDLE_VALUE) return ads_errno();
    LARGE_INTEGER sz; if (!GetFileSizeEx(h, &sz)) { CloseHandle(h); errno = EIO; return -1; }
    if (!value) { CloseHandle(h); return (ssize_t)sz.QuadPart; }
    if ((uint64_t)sz.QuadPart > size) { CloseHandle(h); errno = ERANGE; return -1; }
    DWORD got = 0; BOOL ok = ReadFile(h, value, (DWORD)sz.QuadPart, &got, NULL); CloseHandle(h);
    if (!ok) { errno = EIO; return -1; }
    return (ssize_t)got;
}
int setxattr(const char *path, const char *name, const void *value, size_t size, uint32_t pos, int opt) {
    (void)pos; char p[1400]; if (ads_path(p, sizeof p, path, name)) return -1;
    DWORD disp = opt & XATTR_CREATE ? CREATE_NEW : opt & XATTR_REPLACE ? TRUNCATE_EXISTING : CREATE_ALWAYS;
    HANDLE h = CreateFileA(p, GENERIC_WRITE, FILE_SHARE_READ, NULL, disp, FILE_FLAG_BACKUP_SEMANTICS, NULL);
    if (h == INVALID_HANDLE_VALUE) { if (GetLastError() == ERROR_FILE_EXISTS) { errno = EEXIST; return -1; } return ads_errno(); }
    DWORD put = 0; BOOL ok = WriteFile(h, value, (DWORD)size, &put, NULL); CloseHandle(h);
    if (!ok || put != size) { errno = EIO; return -1; }
    return 0;
}
int removexattr(const char *path, const char *name, int opt) {
    (void)opt; char p[1400]; if (ads_path(p, sizeof p, path, name)) return -1;
    return DeleteFileA(p) ? 0 : ads_errno();
}

// ---------------------------------------------------------------- strings / stdlib / time ----------------------------------------------------------------
size_t strlcpy(char *dst, const char *src, size_t cap) {
    size_t n = strlen(src);
    if (cap) { size_t k = n < cap - 1 ? n : cap - 1; memcpy(dst, src, k); dst[k] = 0; }
    return n;
}
size_t strlcat(char *dst, const char *src, size_t cap) {
    size_t d = strnlen(dst, cap); if (d == cap) return cap + strlen(src);
    return d + strlcpy(dst + d, src, cap - d);
}
void *memmem(const void *hay, size_t hn, const void *needle, size_t nn) {
    if (!nn) return (void *)hay;
    if (hn < nn) return NULL;
    const uint8_t *h = hay, *n = needle, *end = h + hn - nn + 1;
    for (; h < end; h++) { h = memchr(h, n[0], (size_t)(end - h)); if (!h) return NULL; if (!memcmp(h, n, nn)) return (void *)h; }
    return NULL;
}
char *strsep(char **s, const char *delim) {
    char *b = *s; if (!b) return NULL;
    char *e = b + strcspn(b, delim);
    if (*e) { *e = 0; *s = e + 1; } else *s = NULL;
    return b;
}
char *strnstr(const char *hay, const char *needle, size_t n) {
    size_t k = strlen(needle); if (!k) return (char *)hay;
    for (size_t i = 0; i + k <= n && hay[i]; i++) if (!strncmp(hay + i, needle, k)) return (char *)hay + i;
    return NULL;
}
char *strcasestr(const char *hay, const char *needle) {
    size_t k = strlen(needle); if (!k) return (char *)hay;
    for (; *hay; hay++) if (!_strnicmp(hay, needle, k)) return (char *)hay;
    return NULL;
}
volatile long dm_env_gen;   // r53: changed by every setenv / unsetenv (DM_CTL "env" lines), so readers can skip getenv while it stays the same (gfx_vk_exec.c x_read)
int setenv(const char *name, const char *value, int overwrite) {
    if (!name || !*name || strchr(name, '=')) { errno = EINVAL; return -1; }
    if (!overwrite && getenv(name)) return 0;
    int r = _putenv_s(name, value) ? (errno = ENOMEM, -1) : 0; InterlockedIncrement(&dm_env_gen); return r;
}
int unsetenv(const char *name) { int r = _putenv_s(name, "") ? (errno = EINVAL, -1) : 0; InterlockedIncrement(&dm_env_gen); return r; }
// stable merge sort (BSD's mergesort is stable; the host relies on it): top-down with one scratch buffer
static void msort(uint8_t *a, uint8_t *tmp, size_t n, size_t sz, int (*cmp)(const void *, const void *)) {
    if (n < 2) return;
    if (n <= 8) {   // insertion sort: stable
        for (size_t i = 1; i < n; i++) { size_t j = i; memcpy(tmp, a + i * sz, sz); while (j && cmp(a + (j - 1) * sz, tmp) > 0) { memcpy(a + j * sz, a + (j - 1) * sz, sz); j--; } memcpy(a + j * sz, tmp, sz); }
        return;
    }
    size_t h = n / 2; msort(a, tmp, h, sz, cmp); msort(a + h * sz, tmp, n - h, sz, cmp);
    if (cmp(a + (h - 1) * sz, a + h * sz) <= 0) return;
    memcpy(tmp, a, h * sz);
    size_t i = 0, j = h, k = 0;
    while (i < h && j < n) { if (cmp(a + j * sz, tmp + i * sz) < 0) memcpy(a + k++ * sz, a + j++ * sz, sz); else memcpy(a + k++ * sz, tmp + i++ * sz, sz); }
    while (i < h) memcpy(a + k++ * sz, tmp + i++ * sz, sz);
}
int mergesort(void *base, size_t n, size_t size, int (*cmp)(const void *, const void *)) {
    if (n < 2) return 0;
    uint8_t *tmp = malloc((n / 2 + 1) * size + size); if (!tmp) { errno = ENOMEM; return -1; }
    msort(base, tmp, n, size, cmp); free(tmp); return 0;
}
void arc4random_buf(void *buf, size_t n) {
    if (BCryptGenRandom(NULL, buf, (ULONG)n, BCRYPT_USE_SYSTEM_PREFERRED_RNG) < 0) for (size_t i = 0; i < n; i++) ((uint8_t *)buf)[i] = (uint8_t)rand();
}
uint32_t arc4random(void) { uint32_t v; arc4random_buf(&v, sizeof v); return v; }
uint32_t arc4random_uniform(uint32_t bound) {
    if (bound < 2) return 0;
    uint32_t min = (uint32_t)-bound % bound, r;
    do r = arc4random(); while (r < min);
    return r % bound;
}
const char *getprogname(void) {
    static char name[MAX_PATH];
    if (!name[0]) { char p[MAX_PATH]; DWORD n = GetModuleFileNameA(NULL, p, sizeof p); if (n && n < sizeof p) { const char *b = strrchr(p, '\\'); snprintf(name, sizeof name, "%s", b ? b + 1 : p); char *dot = strrchr(name, '.'); if (dot) *dot = 0; } }
    return name;
}
char *realpath(const char *path, char *resolved) {
    char b[MAX_PATH * 4]; DWORD n = GetFullPathNameA(path, sizeof b, b, NULL);
    if (!n || n >= sizeof b) { errno = ENAMETOOLONG; return NULL; }
    if (GetFileAttributesA(b) == INVALID_FILE_ATTRIBUTES) { errno = ENOENT; return NULL; }
    if (!resolved) return _strdup(b);
    snprintf(resolved, 1024, "%s", b); return resolved;
}
const char *dm_tm_zone(const struct tm *tm) { static int once; if (!once) { _tzset(); once = 1; } return _tzname[tm->tm_isdst > 0 ? 1 : 0]; }
long dm_tm_gmtoff(const struct tm *tm) { struct tm a = *tm, b = *tm; time_t u = _mkgmtime(&a), l = mktime(&b); return u == (time_t)-1 || l == (time_t)-1 ? 0 : (long)(u - l); }
// gmtime_r / localtime_r as functions: mingw has them only inline, and hle_table.c hands their addresses to the guest (assembler names, so the inline ones stay as they are)
struct tm *dm_gmtime_r(const time_t *t, struct tm *r) __asm__("gmtime_r");
struct tm *dm_gmtime_r(const time_t *t, struct tm *r) { return _gmtime64_s(r, t) ? NULL : r; }
struct tm *dm_localtime_r(const time_t *t, struct tm *r) __asm__("localtime_r");
struct tm *dm_localtime_r(const time_t *t, struct tm *r) { return _localtime64_s(r, t) ? NULL : r; }

// ---------------------------------------------------------------- dynamic loading ----------------------------------------------------------------
static __thread char t_dlerr[256];
static __thread char t_dlname[MAX_PATH];
void *dlopen(const char *path, int mode) {
    (void)mode;
    if (!path) return GetModuleHandleW(NULL);
    HMODULE m = (mode & RTLD_NOLOAD) ? GetModuleHandleA(path) : LoadLibraryA(path);
    if (!m) snprintf(t_dlerr, sizeof t_dlerr, "%s: error %lu", path, GetLastError());
    return m;
}
void *dlsym(void *handle, const char *name) {
    if (handle != RTLD_DEFAULT && handle != RTLD_SELF && handle != RTLD_NEXT) {
        void *p = (void *)(uintptr_t)GetProcAddress((HMODULE)handle, name);
        if (!p) snprintf(t_dlerr, sizeof t_dlerr, "%s: not found", name);
        return p;
    }
    HMODULE mods[1024]; DWORD need = 0;
    if (!K32EnumProcessModules(GetCurrentProcess(), mods, sizeof mods, &need)) need = 0;
    void *p = (void *)(uintptr_t)GetProcAddress(GetModuleHandleW(NULL), name);
    for (DWORD i = 0; !p && i < need / sizeof *mods && i < 1024; i++) p = (void *)(uintptr_t)GetProcAddress(mods[i], name);
    if (!p) snprintf(t_dlerr, sizeof t_dlerr, "%s: not found", name);
    return p;
}
int dlclose(void *handle) { return handle && handle != GetModuleHandleW(NULL) ? !FreeLibrary((HMODULE)handle) : 0; }
const char *dlerror(void) { if (!t_dlerr[0]) return NULL; static __thread char out[256]; memcpy(out, t_dlerr, sizeof out); t_dlerr[0] = 0; return out; }
int dladdr(const void *addr, Dl_info *info) {
    HMODULE m = NULL;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCWSTR)addr, &m) || !m) return 0;
    DWORD n = GetModuleFileNameA(m, t_dlname, sizeof t_dlname); if (!n || n >= sizeof t_dlname) t_dlname[0] = 0;
    info->dli_fname = t_dlname; info->dli_fbase = m; info->dli_sname = NULL; info->dli_saddr = NULL;
    return 1;
}

// ---------------------------------------------------------------- resources ----------------------------------------------------------------
static struct timeval tv_of(FILETIME f) { uint64_t v = ((uint64_t)f.dwHighDateTime << 32 | f.dwLowDateTime) / 10; struct timeval t = {(long)(v / 1000000), (long)(v % 1000000)}; return t; }
int getrusage(int who, struct rusage *ru) {
    memset(ru, 0, sizeof *ru); FILETIME c, e, k, u;
    BOOL ok = who == RUSAGE_THREAD ? GetThreadTimes(GetCurrentThread(), &c, &e, &k, &u) : GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
    if (!ok) { errno = EINVAL; return -1; }
    ru->ru_utime = tv_of(u); ru->ru_stime = tv_of(k);
    PROCESS_MEMORY_COUNTERS pm; if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pm, sizeof pm)) { ru->ru_maxrss = (long)(pm.PeakWorkingSetSize / 1024); ru->ru_majflt = (long)pm.PageFaultCount; }
    return 0;
}
int getrlimit(int res, struct rlimit *rl) {
    if (res == RLIMIT_NOFILE) { rl->rlim_cur = (rlim_t)_getmaxstdio(); rl->rlim_max = 8192; return 0; }
    rl->rlim_cur = rl->rlim_max = RLIM_INFINITY; return 0;
}
int setrlimit(int res, const struct rlimit *rl) {
    if (res == RLIMIT_NOFILE) { int n = rl->rlim_cur > 8192 ? 8192 : (int)rl->rlim_cur; return _setmaxstdio(n) == -1 ? (errno = EINVAL, -1) : 0; }
    return 0;
}
static int prio_of_nice(int nice) { return nice <= -10 ? THREAD_PRIORITY_HIGHEST : nice < 0 ? THREAD_PRIORITY_ABOVE_NORMAL : nice == 0 ? THREAD_PRIORITY_NORMAL : nice < 10 ? THREAD_PRIORITY_BELOW_NORMAL : THREAD_PRIORITY_LOWEST; }
static int nice_of_prio(int p) { return p >= THREAD_PRIORITY_HIGHEST ? -10 : p > 0 ? -4 : p == 0 ? 0 : p > THREAD_PRIORITY_LOWEST ? 4 : 10; }
// PRIO_PROCESS with a thread ID names that thread (as Linux and the host's callers use it); 0 or this process's ID = the calling thread
int getpriority(int which, int who) {
    (void)which; HANDLE h = GetCurrentThread(); int own = 0;
    if (who && (DWORD)who != GetCurrentThreadId() && (DWORD)who != GetCurrentProcessId()) { h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)who); own = 1; if (!h) { errno = ESRCH; return -1; } }
    int p = GetThreadPriority(h); if (own) CloseHandle(h);
    return nice_of_prio(p);
}
int setpriority(int which, int who, int prio) {
    (void)which; HANDLE h = GetCurrentThread(); int own = 0;
    if (who && (DWORD)who != GetCurrentThreadId() && (DWORD)who != GetCurrentProcessId()) { h = OpenThread(THREAD_SET_LIMITED_INFORMATION, FALSE, (DWORD)who); own = 1; if (!h) { errno = ESRCH; return -1; } }
    BOOL ok = SetThreadPriority(h, prio_of_nice(prio)); if (own) CloseHandle(h);
    return ok ? 0 : (errno = EPERM, -1);
}

// ---------------------------------------------------------------- process start ----------------------------------------------------------------
// Every descriptor in binary mode (game data read with open / fopen without "b" must not be translated); 1 ms timer resolution for the host's sleeps and
// waits (frame pacing, audio); no Windows error dialog on a crash (the host prints its own report)
__attribute__((constructor)) static void posix_win_init(void) {
    _fmode = _O_BINARY;
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);   // the DM_OODLE_SERVE answers (hle_oodle.c write(1, ...)) are binary: a text-mode stdout turns each 0x0a into 0d 0a
    timeBeginPeriod(1);
    SetErrorMode(GetErrorMode() | SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
}
