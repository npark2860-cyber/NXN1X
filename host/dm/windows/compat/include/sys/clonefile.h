// Windows port: clonefile = a recursive copy (NTFS has no copy-on-write clones; save directories are small), renamex_np(RENAME_SWAP) = two renames
// through a temporary name (not atomic as on APFS), RENAME_EXCL = MoveFileEx without replace
#pragma once
#include <stdint.h>
#define CLONE_NOFOLLOW 0x1
#define RENAME_SWAP 0x2
#define RENAME_EXCL 0x4
int clonefile(const char *src, const char *dst, uint32_t flags);
int renamex_np(const char *from, const char *to, unsigned flags);
