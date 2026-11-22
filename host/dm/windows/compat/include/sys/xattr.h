// Windows port: Darwin's extended attribute calls on NTFS alternate data streams: attribute <name> of <path> is the stream "<path>:<name>"
// (posix_win.c). Position and options are ignored (no symbolic links the host makes, so XATTR_NOFOLLOW changes nothing).
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <errno.h>
#define XATTR_NOFOLLOW 0x0001
#define XATTR_CREATE 0x0002
#define XATTR_REPLACE 0x0004
#ifndef ENOATTR
#define ENOATTR ENODATA
#endif
ssize_t getxattr(const char *path, const char *name, void *value, size_t size, uint32_t pos, int opt);
int setxattr(const char *path, const char *name, const void *value, size_t size, uint32_t pos, int opt);
int removexattr(const char *path, const char *name, int opt);
