// Windows port: POSIX calls on top of mingw's <unistd.h> (dm/windows/compat/posix_win.c). pread / pwrite use positioned ReadFile / WriteFile
// on the descriptor's handle, so they leave the file position alone as POSIX's do.
#pragma once
#include_next <unistd.h>
#include <stddef.h>
#include <sys/types.h>
#define _SC_PAGESIZE 29
#define _SC_PAGE_SIZE _SC_PAGESIZE
#define _SC_NPROCESSORS_CONF 83
#define _SC_NPROCESSORS_ONLN 84
#define _SC_OPEN_MAX 4
#define _SC_CLK_TCK 2
#define _SC_PHYS_PAGES 85
long sysconf(int name);
int getpagesize(void);
ssize_t pread(int fd, void *buf, size_t n, off_t off);
ssize_t pwrite(int fd, const void *buf, size_t n, off_t off);
int fsync(int fd);
unsigned alarm(unsigned seconds);
int pause(void);
int symlink(const char *target, const char *path);
ssize_t readlink(const char *path, char *buf, size_t cap);
int link(const char *from, const char *to);
static inline unsigned getuid(void) { return 0; }
static inline unsigned geteuid(void) { return 0; }
static inline unsigned getgid(void) { return 0; }
