// NXN1X build of the host NVN layer: the POSIX calls the NVN layer and the Vulkan backend make, copied verbatim from dm/windows/compat/posix_win.c.
// posix_win.c itself is not built into NXN1X: it installs a vectored exception handler (signals), a TLS callback that rewrites every thread's TEB stack bounds,
// and a constructor that changes the CRT file mode and the error mode. Those belong to the host process; NXN1X owns its threads, faults and CRT state.
// Names are nvnh_ ones (nvnh_rename.h). Keep the copied bodies in step with posix_win.c.
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

// ---- memory (posix_win.c) ----
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

// ---- system (posix_win.c) ----
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

// ---- environment (posix_win.c) ----
volatile long dm_env_gen;   // r53: changed by every setenv / unsetenv (DM_CTL "env" lines), so readers can skip getenv while it stays the same (gfx_vk_exec.c x_read)
int setenv(const char *name, const char *value, int overwrite) {
    if (!name || !*name || strchr(name, '=')) { errno = EINVAL; return -1; }
    if (!overwrite && getenv(name)) return 0;
    int r = _putenv_s(name, value) ? (errno = ENOMEM, -1) : 0; InterlockedIncrement(&dm_env_gen); return r;
}

// ---- dynamic loading (posix_win.c) ----
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

// ---- resources (posix_win.c) ----
static struct timeval tv_of(FILETIME f) { uint64_t v = ((uint64_t)f.dwHighDateTime << 32 | f.dwLowDateTime) / 10; struct timeval t = {(long)(v / 1000000), (long)(v % 1000000)}; return t; }
int getrusage(int who, struct rusage *ru) {
    memset(ru, 0, sizeof *ru); FILETIME c, e, k, u;
    BOOL ok = who == RUSAGE_THREAD ? GetThreadTimes(GetCurrentThread(), &c, &e, &k, &u) : GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
    if (!ok) { errno = EINVAL; return -1; }
    ru->ru_utime = tv_of(u); ru->ru_stime = tv_of(k);
    PROCESS_MEMORY_COUNTERS pm; if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pm, sizeof pm)) { ru->ru_maxrss = (long)(pm.PeakWorkingSetSize / 1024); ru->ru_majflt = (long)pm.PageFaultCount; }
    return 0;
}
static int prio_of_nice(int nice) { return nice <= -10 ? THREAD_PRIORITY_HIGHEST : nice < 0 ? THREAD_PRIORITY_ABOVE_NORMAL : nice == 0 ? THREAD_PRIORITY_NORMAL : nice < 10 ? THREAD_PRIORITY_BELOW_NORMAL : THREAD_PRIORITY_LOWEST; }
int setpriority(int which, int who, int prio) {
    (void)which; HANDLE h = GetCurrentThread(); int own = 0;
    if (who && (DWORD)who != GetCurrentThreadId() && (DWORD)who != GetCurrentProcessId()) { h = OpenThread(THREAD_SET_LIMITED_INFORMATION, FALSE, (DWORD)who); own = 1; if (!h) { errno = ESRCH; return -1; } }
    BOOL ok = SetThreadPriority(h, prio_of_nice(prio)); if (own) CloseHandle(h);
    return ok ? 0 : (errno = EPERM, -1);
}

// pcverify (gfx_vk_core.c pc_verify): what Windows says the page's protection is (MEMORY_BASIC_INFORMATION Protect / Type); 0 when it cannot be queried
unsigned long dm_win_page_protect(const void *p, unsigned long *type) { MEMORY_BASIC_INFORMATION m = {0}; if (!VirtualQuery(p, &m, sizeof m)) { *type = 0; return 0; } *type = m.Type; return m.Protect; }

// stage2c: die()'s message (dm.h), for NXN1X's abort report
char dm_fatal_msg[512];
void (*dm_fatal_hook)(void);   // stage2c3: set by NXN1X (arm_nce_windows.cpp InstallAbortReport): die() writes the abort report through it
