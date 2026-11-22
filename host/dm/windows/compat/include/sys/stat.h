// Windows port: POSIX's mkdir(path, mode) (mingw's takes the path only), lstat = stat (no symbolic links the host makes), and the mode bits mingw lacks
#pragma once
#include_next <sys/stat.h>
#include <io.h>
#include <direct.h>
#define DM_MKDIR_PICK(_1, _2, NAME, ...) NAME
#define DM_MKDIR2(p, m) ((void)(m), _mkdir(p))
#define mkdir(...) DM_MKDIR_PICK(__VA_ARGS__, DM_MKDIR2, _mkdir, x)(__VA_ARGS__)
#define lstat stat
#ifndef S_ISLNK
#define S_ISLNK(m) 0
#endif
#ifndef S_IFLNK
#define S_IFLNK 0120000
#endif
#ifndef S_ISSOCK
#define S_ISSOCK(m) 0
#endif
#ifndef S_IRWXG
#define S_IRWXG 0070
#define S_IRGRP 0040
#define S_IWGRP 0020
#define S_IXGRP 0010
#define S_IRWXO 0007
#define S_IROTH 0004
#define S_IWOTH 0002
#define S_IXOTH 0001
#endif
static inline int fchmod(int fd, mode_t m) { (void)fd; (void)m; return 0; }
// Darwin's st_mtimespec: seconds only here (mingw's struct stat has st_mtime); futimens sets the times through SetFileTime (posix_win.c)
#define DM_ST_MTIME(st) ((int64_t)(st)->st_mtime)
#ifndef UTIME_NOW
#define UTIME_NOW ((1l << 30) - 1l)
#define UTIME_OMIT ((1l << 30) - 2l)
#endif
struct timespec;
int futimens(int fd, const struct timespec t[2]);
