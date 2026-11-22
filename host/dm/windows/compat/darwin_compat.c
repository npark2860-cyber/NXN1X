// Windows port: the Darwin / Mach calls the host makes, on Win32 (headers in dm/windows/compat/include). The same subset dm/android/compat implements on Linux.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>
#include <psapi.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <pthread/qos.h>
#include <CoreFoundation/CoreFoundation.h>
#include <sys/clonefile.h>
#include <sys/sysctl.h>
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

// ---------------- Mach time: the QueryPerformanceCounter count ----------------
static mach_timebase_info_data_t g_tb;
static LARGE_INTEGER g_qpf;
static void tb_init(void) {
    if (g_tb.denom) return;
    QueryPerformanceFrequency(&g_qpf);
    uint64_t n = 1000000000ull, d = (uint64_t)g_qpf.QuadPart, a = n, b = d;
    while (b) { uint64_t t = a % b; a = b; b = t; }
    g_tb.numer = (uint32_t)(n / a); g_tb.denom = (uint32_t)(d / a);
}
int mach_timebase_info(mach_timebase_info_t tb) { tb_init(); *tb = g_tb; return 0; }
uint64_t mach_absolute_time(void) { LARGE_INTEGER c; QueryPerformanceCounter(&c); return (uint64_t)c.QuadPart; }
uint64_t mach_continuous_time(void) { return mach_absolute_time(); }   // the counter keeps counting through sleep on Windows
// Sleep until the counter reaches deadline: a high-resolution waitable timer for the bulk, then a short spin for the last 50 µs
int mach_wait_until(uint64_t deadline) {
    tb_init();
    static __thread HANDLE timer;
    if (!timer) timer = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    for (;;) {
        uint64_t now = mach_absolute_time(); if (now >= deadline) return 0;
        uint64_t ns = (deadline - now) * g_tb.numer / g_tb.denom;
        if (ns <= 50000) { while (mach_absolute_time() < deadline) YieldProcessor(); return 0; }
        LARGE_INTEGER due; due.QuadPart = -(LONGLONG)((ns - 50000) / 100);
        if (timer && SetWaitableTimer(timer, &due, 0, NULL, NULL, FALSE)) WaitForSingleObject(timer, INFINITE);
        else Sleep((DWORD)((ns - 50000) / 1000000));
    }
}
// This thread's cumulative {instructions, cycles} (a libSystem private call on Darwin). Not available here: the host's counters stay 0.
int thread_selfcounts(int type, void *buf, size_t nbytes) { (void)type; (void)buf; (void)nbytes; return -1; }

// ---------------- ports ----------------
mach_port_t mach_task_self(void) { return (mach_port_t)GetCurrentProcessId(); }
mach_port_t mach_thread_self(void) { return (mach_port_t)GetCurrentThreadId(); }
kern_return_t mach_port_deallocate(task_t task, mach_port_t name) { (void)task; (void)name; return KERN_SUCCESS; }
mach_port_t pthread_mach_thread_np(pthread_t t) { HANDLE h = pthread_gethandle(t); return h ? (mach_port_t)GetThreadId(h) : 0; }
vm_size_t vm_page_size = 4096;

// ---------------- memory ----------------
// A read that fails instead of faulting
kern_return_t vm_read_overwrite(task_t task, vm_address_t addr, vm_size_t size, vm_address_t data, vm_size_t *outsize) {
    (void)task; SIZE_T got = 0;
    BOOL ok = ReadProcessMemory(GetCurrentProcess(), (LPCVOID)addr, (LPVOID)data, size, &got);
    if (outsize) *outsize = (vm_size_t)got;
    return ok && got == size ? KERN_SUCCESS : KERN_INVALID_ADDRESS;
}
static int writable(DWORD p) { return (p & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_WRITECOPY)) && !(p & PAGE_GUARD); }
kern_return_t vm_write(task_t task, vm_address_t addr, vm_offset_t data, mach_msg_type_number_t size) {
    (void)task;
    for (uint8_t *a = (uint8_t *)addr, *end = a + size; a < end;) {   // refuse a page not writable right now, as Mach does
        MEMORY_BASIC_INFORMATION mi;
        if (!VirtualQuery(a, &mi, sizeof mi) || mi.State != MEM_COMMIT || !writable(mi.Protect)) return KERN_PROTECTION_FAILURE;
        a = (uint8_t *)mi.BaseAddress + mi.RegionSize;
    }
    SIZE_T put = 0;
    return WriteProcessMemory(GetCurrentProcess(), (LPVOID)addr, (LPCVOID)data, size, &put) && put == size ? KERN_SUCCESS : KERN_INVALID_ADDRESS;
}
kern_return_t vm_allocate(task_t task, vm_address_t *addr, vm_size_t size, int flags) {
    (void)task;
    void *p = VirtualAlloc((flags & VM_FLAGS_ANYWHERE) ? NULL : (void *)*addr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!p) return KERN_NO_SPACE;
    *addr = (vm_address_t)p; return KERN_SUCCESS;
}
kern_return_t vm_deallocate(task_t task, vm_address_t addr, vm_size_t size) { (void)task; return munmap((void *)addr, size) ? KERN_INVALID_ADDRESS : KERN_SUCCESS; }
kern_return_t mach_vm_deallocate(task_t task, mach_vm_address_t addr, mach_vm_size_t size) { return vm_deallocate(task, (vm_address_t)addr, (vm_size_t)size); }
kern_return_t vm_protect(task_t task, vm_address_t addr, vm_size_t size, boolean_t set_max, vm_prot_t prot) {
    (void)task; if (set_max) return KERN_SUCCESS;
    return mprotect((void *)addr, size, prot) ? KERN_PROTECTION_FAILURE : KERN_SUCCESS;
}
kern_return_t mach_vm_protect(task_t task, mach_vm_address_t addr, mach_vm_size_t size, boolean_t set_max, vm_prot_t prot) {
    return vm_protect(task, (vm_address_t)addr, (vm_size_t)size, set_max, prot);
}
static int prot_of(DWORD p) {
    if (p & (PAGE_NOACCESS | PAGE_GUARD)) return VM_PROT_NONE;
    switch (p & 0xff) {
    case PAGE_READONLY: return VM_PROT_READ;
    case PAGE_READWRITE: case PAGE_WRITECOPY: return VM_PROT_READ | VM_PROT_WRITE;
    case PAGE_EXECUTE: return VM_PROT_EXECUTE;
    case PAGE_EXECUTE_READ: return VM_PROT_READ | VM_PROT_EXECUTE;
    case PAGE_EXECUTE_READWRITE: case PAGE_EXECUTE_WRITECOPY: return VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE;
    default: return VM_PROT_NONE;
    }
}
// The region containing addr, or the next one above it (as Mach answers); a region is a run of pages with the same state and protection
kern_return_t vm_region_64(task_t task, vm_address_t *addr, vm_size_t *size, vm_region_flavor_t flavor, vm_region_info_t info, mach_msg_type_number_t *count, mach_port_t *obj) {
    (void)task; (void)count;
    if (flavor != VM_REGION_BASIC_INFO_64) return KERN_INVALID_ARGUMENT;
    uint8_t *a = (uint8_t *)*addr; MEMORY_BASIC_INFORMATION mi;
    for (;;) {
        if (!VirtualQuery(a, &mi, sizeof mi)) return KERN_INVALID_ADDRESS;
        if (mi.State != MEM_FREE) break;
        a = (uint8_t *)mi.BaseAddress + mi.RegionSize;
    }
    vm_region_basic_info_data_64_t *bi = (vm_region_basic_info_data_64_t *)info; memset(bi, 0, sizeof *bi);
    bi->protection = mi.State == MEM_COMMIT ? prot_of(mi.Protect) : VM_PROT_NONE; bi->max_protection = VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE;
    bi->shared = mi.Type == MEM_MAPPED;
    *addr = (vm_address_t)mi.BaseAddress; *size = (vm_size_t)mi.RegionSize; if (obj) *obj = 0;
    return KERN_SUCCESS;
}
kern_return_t mach_vm_region(task_t task, mach_vm_address_t *addr, mach_vm_size_t *size, vm_region_flavor_t flavor, vm_region_info_t info, mach_msg_type_number_t *count, mach_port_t *obj) {
    vm_address_t a = (vm_address_t)*addr; vm_size_t s = 0;
    kern_return_t r = vm_region_64(task, &a, &s, flavor, info, count, obj); *addr = a; *size = s; return r;
}
// Sharing the source's pages (copy=FALSE) needs section-backed memory; the host's is VirtualAlloc memory. copy=TRUE: an eager copy into fresh pages
kern_return_t vm_remap(task_t target, vm_address_t *addr, vm_size_t size, vm_address_t mask, int flags, task_t src_task, vm_address_t src, boolean_t copy,
                       vm_prot_t *cur, vm_prot_t *max, vm_inherit_t inherit) {
    (void)target; (void)mask; (void)src_task; (void)inherit;
    if (!copy) return KERN_NOT_SUPPORTED;
    void *hint = (flags & VM_FLAGS_ANYWHERE) ? NULL : (void *)*addr;
    void *p = mmap(hint, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | (hint ? MAP_FIXED : 0), -1, 0);
    if (p == MAP_FAILED) return KERN_NO_SPACE;
    memcpy(p, (void *)src, size);
    *addr = (vm_address_t)p; if (cur) *cur = VM_PROT_READ | VM_PROT_WRITE; if (max) *max = VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE;
    return KERN_SUCCESS;
}
kern_return_t mach_vm_remap(task_t target, mach_vm_address_t *addr, mach_vm_size_t size, mach_vm_address_t mask, int flags, task_t src_task, mach_vm_address_t src,
                            boolean_t copy, vm_prot_t *cur, vm_prot_t *max, vm_inherit_t inherit) {
    vm_address_t a = (vm_address_t)*addr;
    kern_return_t r = vm_remap(target, &a, (vm_size_t)size, (vm_address_t)mask, flags, src_task, (vm_address_t)src, copy, cur, max, inherit);
    *addr = a; return r;
}

// ---------------- thread and task information ----------------
static uint64_t ft_ns(FILETIME f) { return ((uint64_t)f.dwHighDateTime << 32 | f.dwLowDateTime) * 100; }
kern_return_t thread_info(thread_act_t t, thread_flavor_t flavor, thread_info_t out, mach_msg_type_number_t *count) {
    (void)count;
    HANDLE h = (DWORD)t == GetCurrentThreadId() ? GetCurrentThread() : OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, (DWORD)t);
    if (!h) return KERN_INVALID_ARGUMENT;
    FILETIME c, e, k, u; BOOL ok = GetThreadTimes(h, &c, &e, &k, &u); int prio = GetThreadPriority(h);
    if (h != GetCurrentThread()) CloseHandle(h);
    if (!ok) return KERN_INVALID_ARGUMENT;
    uint64_t un = ft_ns(u), kn = ft_ns(k);
    if (flavor == THREAD_BASIC_INFO) {   // Windows does not say whether a thread is running: TH_STATE_RUNNING is reported for all
        thread_basic_info_data_t *bi = (thread_basic_info_data_t *)out; memset(bi, 0, sizeof *bi);
        bi->user_time.seconds = (integer_t)(un / 1000000000ull); bi->user_time.microseconds = (integer_t)(un % 1000000000ull / 1000);
        bi->system_time.seconds = (integer_t)(kn / 1000000000ull); bi->system_time.microseconds = (integer_t)(kn % 1000000000ull / 1000);
        bi->run_state = TH_STATE_RUNNING;
        return KERN_SUCCESS;
    }
    if (flavor == THREAD_EXTENDED_INFO) {
        thread_extended_info_data_t *xi = (thread_extended_info_data_t *)out; memset(xi, 0, sizeof *xi);
        xi->pth_user_time = un; xi->pth_system_time = kn; xi->pth_run_state = TH_STATE_RUNNING; xi->pth_curpri = xi->pth_priority = prio;
        return KERN_SUCCESS;
    }
    return KERN_INVALID_ARGUMENT;
}
kern_return_t task_info(task_t task, task_flavor_t flavor, task_info_t out, mach_msg_type_number_t *count) {
    (void)task; (void)count;
    if (flavor != TASK_VM_INFO) return KERN_FAILURE;   // no energy / power accounting
    task_vm_info_data_t *vi = (task_vm_info_data_t *)out; memset(vi, 0, sizeof *vi);
    PROCESS_MEMORY_COUNTERS_EX pm; if (!K32GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pm, sizeof pm)) return KERN_FAILURE;
    vi->resident_size = vi->internal = pm.WorkingSetSize; vi->resident_size_peak = pm.PeakWorkingSetSize;
    vi->phys_footprint = pm.PrivateUsage; vi->virtual_size = pm.PrivateUsage; vi->page_size = 4096;
    return KERN_SUCCESS;
}
kern_return_t task_threads(task_t task, thread_act_array_t *list, mach_msg_type_number_t *count) {
    (void)task;
    HANDLE s = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0); if (s == INVALID_HANDLE_VALUE) return KERN_FAILURE;
    size_t cap = 4096; mach_port_t *a = mmap(NULL, cap * sizeof *a, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);   // freed by vm_deallocate (munmap)
    if (a == MAP_FAILED) { CloseHandle(s); return KERN_NO_SPACE; }
    unsigned n = 0; DWORD pid = GetCurrentProcessId(); THREADENTRY32 te = {.dwSize = sizeof te};
    for (BOOL ok = Thread32First(s, &te); ok && n < cap; ok = Thread32Next(s, &te)) if (te.th32OwnerProcessID == pid) a[n++] = (mach_port_t)te.th32ThreadID;
    CloseHandle(s); *list = a; *count = n; return KERN_SUCCESS;
}
// The time-constraint (real-time) class: the calling thread at THREAD_PRIORITY_TIME_CRITICAL (the host asks it for its frame thread)
int thread_policy_set(int thread, thread_policy_flavor_t flavor, thread_policy_t policy, unsigned count) {
    (void)policy; (void)count;
    if (flavor != THREAD_TIME_CONSTRAINT_POLICY || (DWORD)thread != GetCurrentThreadId()) return KERN_NOT_SUPPORTED;
    return SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL) ? KERN_SUCCESS : KERN_FAILURE;
}
int thread_policy_get(int thread, thread_policy_flavor_t flavor, thread_policy_t policy, unsigned *count, int *get_default) {
    (void)thread; (void)flavor; (void)count; if (get_default) *get_default = 1;
    memset(policy, 0, sizeof(thread_time_constraint_policy_data_t)); return KERN_SUCCESS;
}

// ---------------- thread suspension (thread dumps) ----------------
#define MAX_SUSP 64
static struct { DWORD tid; HANDLE h; } g_susp[MAX_SUSP];
static SRWLOCK g_susp_lock = SRWLOCK_INIT;
kern_return_t thread_suspend(thread_act_t t) {
    if ((DWORD)t == GetCurrentThreadId()) return KERN_INVALID_ARGUMENT;
    HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, (DWORD)t);
    if (!h) return KERN_INVALID_ARGUMENT;
    if (SuspendThread(h) == (DWORD)-1) { CloseHandle(h); return KERN_FAILURE; }
    AcquireSRWLockExclusive(&g_susp_lock);
    int i = 0; while (i < MAX_SUSP && g_susp[i].tid) i++;
    if (i < MAX_SUSP) { g_susp[i].tid = (DWORD)t; g_susp[i].h = h; }
    ReleaseSRWLockExclusive(&g_susp_lock);
    if (i == MAX_SUSP) { ResumeThread(h); CloseHandle(h); return KERN_NO_SPACE; }
    return KERN_SUCCESS;
}
static HANDLE susp_take(DWORD tid, int remove) {
    HANDLE h = NULL; AcquireSRWLockExclusive(&g_susp_lock);
    for (int i = 0; i < MAX_SUSP; i++) if (g_susp[i].tid == tid) { h = g_susp[i].h; if (remove) g_susp[i].tid = 0, g_susp[i].h = NULL; break; }
    ReleaseSRWLockExclusive(&g_susp_lock); return h;
}
kern_return_t thread_get_state(thread_act_t t, thread_flavor_t flavor, thread_state_t out, mach_msg_type_number_t *count) {
    (void)count; HANDLE h = susp_take((DWORD)t, 0);
    if (flavor != ARM_THREAD_STATE64 || !h) return KERN_INVALID_ARGUMENT;
    CONTEXT c; memset(&c, 0, sizeof c); c.ContextFlags = CONTEXT_FULL;
    if (!GetThreadContext(h, &c)) return KERN_FAILURE;
    arm_thread_state64_t *s = (arm_thread_state64_t *)out;
    memcpy(s->__x, &c.X0, 29 * 8); s->__fp = c.Fp; s->__lr = c.Lr; s->__sp = c.Sp; s->__pc = c.Pc; s->__cpsr = c.Cpsr; s->__pad = 0;
    return KERN_SUCCESS;
}
kern_return_t thread_resume(thread_act_t t) {
    HANDLE h = susp_take((DWORD)t, 1); if (!h) return KERN_INVALID_ARGUMENT;
    ResumeThread(h); CloseHandle(h); return KERN_SUCCESS;
}

// ---------------- QoS classes → thread priorities ----------------
static __thread qos_class_t t_qos = QOS_CLASS_DEFAULT;
static int qos_prio(qos_class_t q) {
    switch (q) { case QOS_CLASS_USER_INTERACTIVE: return THREAD_PRIORITY_HIGHEST; case QOS_CLASS_USER_INITIATED: return THREAD_PRIORITY_ABOVE_NORMAL;
                 case QOS_CLASS_UTILITY: return THREAD_PRIORITY_BELOW_NORMAL; case QOS_CLASS_BACKGROUND: return THREAD_PRIORITY_LOWEST; default: return THREAD_PRIORITY_NORMAL; }
}
int pthread_set_qos_class_self_np(qos_class_t q, int rel) { (void)rel; t_qos = q; SetThreadPriority(GetCurrentThread(), qos_prio(q)); return 0; }
int pthread_get_qos_class_np(pthread_t t, qos_class_t *q, int *rel) { (void)t; if (q) *q = t_qos; if (rel) *rel = 0; return 0; }
qos_class_t qos_class_self(void) { return t_qos; }

// ---------------- CFDictionary (pointer keys, no callbacks) ----------------
struct dm_cfdict { size_t cap, n; const void **k, **v; };
static const void *const TOMB = (const void *)1;
static size_t slot_of(const struct dm_cfdict *d, const void *key) {
    uintptr_t h = (uintptr_t)key; h ^= h >> 33; h *= 0xff51afd7ed558ccdull; h ^= h >> 33;
    return (size_t)h & (d->cap - 1);
}
CFMutableDictionaryRef CFDictionaryCreateMutable(CFAllocatorRef a, CFIndex cap, const void *kcb, const void *vcb) {
    (void)a; (void)cap; (void)kcb; (void)vcb;
    struct dm_cfdict *d = calloc(1, sizeof *d); d->cap = 64; d->k = calloc(d->cap, sizeof *d->k); d->v = calloc(d->cap, sizeof *d->v); return d;
}
static size_t find(const struct dm_cfdict *d, const void *key) {   // the key's slot, or cap when absent
    for (size_t i = slot_of(d, key), n = 0; n < d->cap; i = (i + 1) & (d->cap - 1), n++) { if (!d->k[i]) return d->cap; if (d->k[i] == key) return i; }
    return d->cap;
}
const void *CFDictionaryGetValue(CFDictionaryRef d, const void *key) { size_t i = find(d, key); return i < d->cap ? d->v[i] : NULL; }
static void put(struct dm_cfdict *d, const void *key, const void *value) {
    size_t i = slot_of(d, key);
    while (d->k[i] && d->k[i] != TOMB) i = (i + 1) & (d->cap - 1);
    d->k[i] = key; d->v[i] = value; d->n++;
}
void CFDictionarySetValue(CFMutableDictionaryRef d, const void *key, const void *value) {
    size_t i = find(d, key); if (i < d->cap) { d->v[i] = value; return; }
    if (2 * (d->n + 1) > d->cap) {   // grow (tombstones are dropped)
        size_t oc = d->cap; const void **ok = d->k, **ov = d->v;
        d->cap *= 2; d->n = 0; d->k = calloc(d->cap, sizeof *d->k); d->v = calloc(d->cap, sizeof *d->v);
        for (size_t j = 0; j < oc; j++) if (ok[j] && ok[j] != TOMB) put(d, ok[j], ov[j]);
        free(ok); free(ov);
    }
    put(d, key, value);
}
void CFDictionaryRemoveValue(CFMutableDictionaryRef d, const void *key) { size_t i = find(d, key); if (i < d->cap) { d->k[i] = TOMB; d->v[i] = NULL; d->n--; } }
CFIndex CFDictionaryGetCount(CFDictionaryRef d) { return (CFIndex)d->n; }

// ---------------- clonefile / renamex_np ----------------
// A recursive copy: CopyFile keeps the alternate data streams, which carry the host's extended attributes (sys/xattr.h)
int clonefile(const char *src, const char *dst, uint32_t flags) {
    (void)flags;
    DWORD a = GetFileAttributesA(src); if (a == INVALID_FILE_ATTRIBUTES) { errno = ENOENT; return -1; }
    if (GetFileAttributesA(dst) != INVALID_FILE_ATTRIBUTES) { errno = EEXIST; return -1; }
    if (!(a & FILE_ATTRIBUTE_DIRECTORY)) { if (CopyFileA(src, dst, TRUE)) return 0; errno = EIO; return -1; }
    if (!CreateDirectoryExA(src, dst, NULL)) { errno = EIO; return -1; }   // with the template's attributes and streams
    DIR *d = opendir(src); if (!d) return -1;
    struct dirent *e; int r = 0;
    while (!r && (e = readdir(d))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        char x[MAX_PATH * 4], y[MAX_PATH * 4]; snprintf(x, sizeof x, "%s/%s", src, e->d_name); snprintf(y, sizeof y, "%s/%s", dst, e->d_name);
        r = clonefile(x, y, flags);
    }
    int er = errno; closedir(d); errno = er; return r;
}
// RENAME_SWAP: two entries exchanged through a temporary name (APFS does it atomically; NTFS has no exchange call). RENAME_EXCL: no replacing
int renamex_np(const char *from, const char *to, unsigned flags) {
    if (flags & RENAME_SWAP) {
        char tmp[MAX_PATH * 4]; snprintf(tmp, sizeof tmp, "%s.dm-swap-%lu", to, GetCurrentThreadId());
        if (!MoveFileExA(to, tmp, 0)) { errno = ENOENT; return -1; }
        if (!MoveFileExA(from, to, 0)) { MoveFileExA(tmp, to, 0); errno = EIO; return -1; }
        if (!MoveFileExA(tmp, from, 0)) { errno = EIO; return -1; }
        return 0;
    }
    if (MoveFileExA(from, to, (flags & RENAME_EXCL) ? 0 : MOVEFILE_REPLACE_EXISTING)) return 0;
    errno = GetLastError() == ERROR_ALREADY_EXISTS || GetLastError() == ERROR_FILE_EXISTS ? EEXIST : EIO; return -1;
}

// ---------------- sysctl: the process start time ----------------
int sysctl(int *name, unsigned namelen, void *old, size_t *oldlen, void *newp, size_t newlen) {
    (void)newp; (void)newlen;
    if (namelen != 4 || name[0] != CTL_KERN || name[1] != KERN_PROC || name[2] != KERN_PROC_PID || !old || !oldlen || *oldlen < sizeof(struct kinfo_proc)) { errno = ENOTSUP; return -1; }
    FILETIME c, e, k, u; if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) { errno = ESRCH; return -1; }
    uint64_t t = ((uint64_t)c.dwHighDateTime << 32 | c.dwLowDateTime) / 10 - 11644473600000000ull;   // µs since 1970
    struct kinfo_proc *kp = old; memset(kp, 0, sizeof *kp);
    kp->kp_proc.p_starttime.tv_sec = (long)(t / 1000000); kp->kp_proc.p_starttime.tv_usec = (long)(t % 1000000);
    *oldlen = sizeof *kp; return 0;
}
// hw.model / hw.machine: the processor name from the registry (save_sync.c records which machine wrote a save)
int sysctlbyname(const char *name, void *old, size_t *oldlen, void *newp, size_t newlen) {
    (void)newp; (void)newlen;
    if ((!strcmp(name, "hw.model") || !strcmp(name, "hw.machine")) && old && oldlen && *oldlen) {
        DWORD n = (DWORD)*oldlen;
        if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "ProcessorNameString", RRF_RT_REG_SZ, NULL, old, &n) == ERROR_SUCCESS) { *oldlen = n; return 0; }
    }
    errno = ENOENT; return -1;
}
