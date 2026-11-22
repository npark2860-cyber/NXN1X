// Windows port: anonymous mappings on VirtualAlloc (dm/windows/compat/posix_win.c). The host maps only anonymous memory:
//   mmap PROT_NONE = a reservation; mmap with access = reserved and committed; MAP_FIXED inside a reservation = its pages decommitted (and committed again
//   with the access asked for), so they read as zeros as a fresh mapping does; mprotect commits reserved pages it gives access to;
//   munmap of a whole reservation releases it, of a part decommits that part. File mappings (fd >= 0) fail with ENODEV.
#pragma once
#include <stddef.h>
#include <sys/types.h>
#define PROT_NONE 0
#define PROT_READ 1
#define PROT_WRITE 2
#define PROT_EXEC 4
#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_FIXED 0x10
#define MAP_ANON 0x20
#define MAP_ANONYMOUS MAP_ANON
#define MAP_NORESERVE 0x4000
#define MAP_JIT 0x800
#define MAP_FAILED ((void *)-1)
#define MADV_NORMAL 0
#define MADV_RANDOM 1
#define MADV_SEQUENTIAL 2
#define MADV_WILLNEED 3
#define MADV_DONTNEED 4
#define MADV_FREE 5
#define MADV_FREE_REUSABLE 7
#define MADV_FREE_REUSE 8
#define MS_ASYNC 1
#define MS_INVALIDATE 2
#define MS_SYNC 4
void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off);
int munmap(void *addr, size_t len);
int mprotect(void *addr, size_t len, int prot);
int madvise(void *addr, size_t len, int advice);
int msync(void *addr, size_t len, int flags);
int mlock(const void *addr, size_t len);
int munlock(const void *addr, size_t len);
