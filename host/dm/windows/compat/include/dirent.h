// Windows port: directory reading on FindFirstFileEx (dm/windows/compat/dirent_win.c) with the entry type (d_type) that mingw's dirent lacks:
// the host lists RomFS directories by type without a stat per entry. dirfd gives a descriptor-like number for fstatat, which answers from the
// entry just read (its size, times and attributes come with the listing) and otherwise stats "<directory>/<name>".
#pragma once
#include <sys/types.h>
#include <sys/stat.h>
#include <limits.h>
#define DT_UNKNOWN 0
#define DT_FIFO 1
#define DT_CHR 2
#define DT_DIR 4
#define DT_BLK 6
#define DT_REG 8
#define DT_LNK 10
#define DT_SOCK 12
#ifndef AT_FDCWD
#define AT_FDCWD (-100)
#endif
#define AT_SYMLINK_NOFOLLOW 0x100
struct dirent { uint64_t d_ino; uint16_t d_reclen, d_namlen; uint8_t d_type; char d_name[260 * 3 + 1]; };
typedef struct dm_dir DIR;
DIR *dm_opendir(const char *path);
struct dirent *dm_readdir(DIR *d);
int dm_closedir(DIR *d);
void dm_rewinddir(DIR *d);
int dm_dirfd(DIR *d);
int dm_fstatat(int dfd, const char *name, struct stat *st, int flags);
#define opendir dm_opendir
#define readdir dm_readdir
#define closedir dm_closedir
#define rewinddir dm_rewinddir
#define dirfd dm_dirfd
#define fstatat dm_fstatat
