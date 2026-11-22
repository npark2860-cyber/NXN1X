// Windows port: open flags mingw lacks (no meaning on Windows: 0), and binary mode for every descriptor (posix_win.c sets _fmode).
// fcntl: F_GETPATH (GetFinalPathNameByHandle), F_GETFL / F_SETFL / F_GETFD / F_SETFD accepted and reporting 0.
#pragma once
#include_next <fcntl.h>
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif
#ifndef O_DIRECTORY
#define O_DIRECTORY 0
#endif
#ifndef O_NONBLOCK
#define O_NONBLOCK 0
#endif
#ifndef O_SYNC
#define O_SYNC 0
#endif
#ifndef O_DSYNC
#define O_DSYNC 0
#endif
#ifndef O_NOCTTY
#define O_NOCTTY 0
#endif
#ifndef AT_FDCWD
#define AT_FDCWD (-100)
#endif
#define F_GETFD 1
#define F_SETFD 2
#define F_GETFL 3
#define F_SETFL 4
#define F_GETPATH 50
#define F_FULLFSYNC 51
#define FD_CLOEXEC 1
int fcntl(int fd, int cmd, ...);
