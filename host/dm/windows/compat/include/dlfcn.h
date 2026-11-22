// Windows port: the dynamic loader calls on Win32 (dm/windows/compat/posix_win.c). dladdr names the module (GetModuleFileName) and leaves
// dli_sname NULL: there is no symbol table to look in. dlsym(RTLD_DEFAULT) looks the name up among the exports of the executable and the loaded DLLs.
#pragma once
#define RTLD_LAZY 0x1
#define RTLD_NOW 0x2
#define RTLD_LOCAL 0x4
#define RTLD_GLOBAL 0x8
#define RTLD_NOLOAD 0x10
#define RTLD_DEFAULT ((void *)-2)
#define RTLD_SELF ((void *)-3)
#define RTLD_NEXT ((void *)-1)
typedef struct { const char *dli_fname; void *dli_fbase; const char *dli_sname; void *dli_saddr; } Dl_info;
void *dlopen(const char *path, int mode);
void *dlsym(void *handle, const char *name);
int dlclose(void *handle);
const char *dlerror(void);
int dladdr(const void *addr, Dl_info *info);
