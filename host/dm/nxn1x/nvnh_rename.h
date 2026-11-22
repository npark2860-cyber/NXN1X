// NXN1X build of the host NVN layer (dm/nxn1x): force-included before darwin_compat.h.
// The host defines a few POSIX names itself (dm/windows/compat/posix_win.c). Inside NXN1X those definitions would be global symbols of the emulator
// executable, where another library may define or expect the same names. The NXN1X build therefore compiles the needed ones (nvnh_posix.c) under
// nvnh_ names; this header renames every use and declaration in the host sources the same way.
#pragma once
#define DM_NXN1X 1
#define mmap nvnh_mmap
#define munmap nvnh_munmap
#define mprotect nvnh_mprotect
#define sysconf nvnh_sysconf
#define getpagesize nvnh_getpagesize
#define setenv nvnh_setenv
#define getrusage nvnh_getrusage
#define setpriority nvnh_setpriority
#define dlopen nvnh_dlopen
#define dlsym nvnh_dlsym
#define dlerror nvnh_dlerror
#define dladdr nvnh_dladdr
#define dlclose nvnh_dlclose
