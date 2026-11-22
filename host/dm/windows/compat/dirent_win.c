// Windows port: opendir / readdir on FindFirstFileEx (dm/windows/compat/include/dirent.h). The listing brings each entry's type, size, times and attributes;
// fstatat on the directory's number answers from the entry just read without another system call (the host lists RomFS directories of thousands of files)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

struct dm_dir { HANDLE h; int first, done, slot; char path[MAX_PATH * 4]; WIN32_FIND_DATAA fd; struct dirent ent; };
#define MAX_DIRS 1024
static DIR *g_dirs[MAX_DIRS];
static SRWLOCK g_dirs_lock = SRWLOCK_INIT;

static HANDLE find_first(const char *path, WIN32_FIND_DATAA *fd) {
    char pat[MAX_PATH * 4 + 4]; size_t n = strlen(path);
    snprintf(pat, sizeof pat, "%s%s*", path, n && (path[n - 1] == '/' || path[n - 1] == '\\') ? "" : "\\");
    return FindFirstFileExA(pat, FindExInfoBasic, fd, FindExSearchNameMatch, NULL, FIND_FIRST_EX_LARGE_FETCH);
}
DIR *dm_opendir(const char *path) {
    DWORD a = GetFileAttributesA(path);
    if (a == INVALID_FILE_ATTRIBUTES) { errno = ENOENT; return NULL; }
    if (!(a & FILE_ATTRIBUTE_DIRECTORY)) { errno = ENOTDIR; return NULL; }
    DIR *d = calloc(1, sizeof *d); if (!d) { errno = ENOMEM; return NULL; }
    snprintf(d->path, sizeof d->path, "%s", path);
    d->h = find_first(path, &d->fd);
    if (d->h == INVALID_HANDLE_VALUE) { if (GetLastError() != ERROR_FILE_NOT_FOUND) { free(d); errno = EACCES; return NULL; } d->done = 1; }
    d->first = 1; d->slot = -1;
    AcquireSRWLockExclusive(&g_dirs_lock);
    for (int i = 0; i < MAX_DIRS; i++) if (!g_dirs[i]) { g_dirs[i] = d; d->slot = i; break; }
    ReleaseSRWLockExclusive(&g_dirs_lock);
    return d;
}
static uint8_t type_of(DWORD a) { return a & FILE_ATTRIBUTE_REPARSE_POINT ? DT_LNK : a & FILE_ATTRIBUTE_DIRECTORY ? DT_DIR : DT_REG; }
struct dirent *dm_readdir(DIR *d) {
    if (d->done) return NULL;
    if (!d->first && !FindNextFileA(d->h, &d->fd)) { d->done = 1; return NULL; }
    d->first = 0;
    size_t n = strlen(d->fd.cFileName); if (n >= sizeof d->ent.d_name) n = sizeof d->ent.d_name - 1;
    memcpy(d->ent.d_name, d->fd.cFileName, n); d->ent.d_name[n] = 0;
    d->ent.d_namlen = (uint16_t)n; d->ent.d_reclen = sizeof d->ent; d->ent.d_ino = 1; d->ent.d_type = type_of(d->fd.dwFileAttributes);
    return &d->ent;
}
int dm_closedir(DIR *d) {
    if (!d) { errno = EBADF; return -1; }
    if (d->h != INVALID_HANDLE_VALUE && d->h) FindClose(d->h);
    if (d->slot >= 0) { AcquireSRWLockExclusive(&g_dirs_lock); g_dirs[d->slot] = NULL; ReleaseSRWLockExclusive(&g_dirs_lock); }
    free(d); return 0;
}
void dm_rewinddir(DIR *d) {
    if (d->h != INVALID_HANDLE_VALUE && d->h) FindClose(d->h);
    d->h = find_first(d->path, &d->fd); d->first = 1; d->done = d->h == INVALID_HANDLE_VALUE;
}
// The directory's number: 1,000,000 + its table slot (never a CRT descriptor)
int dm_dirfd(DIR *d) { if (d->slot < 0) { errno = ENOTSUP; return -1; } return 1000000 + d->slot; }
static time_t time_of(FILETIME f) { uint64_t v = (uint64_t)f.dwHighDateTime << 32 | f.dwLowDateTime; return (time_t)(v / 10000000ull - 11644473600ull); }
int dm_fstatat(int dfd, const char *name, struct stat *st, int flags) {
    (void)flags;
    if (dfd == AT_FDCWD) return stat(name, st);
    DIR *d = NULL; int slot = dfd - 1000000;
    if (slot >= 0 && slot < MAX_DIRS) { AcquireSRWLockShared(&g_dirs_lock); d = g_dirs[slot]; ReleaseSRWLockShared(&g_dirs_lock); }
    if (!d) { errno = EBADF; return -1; }
    if (!d->first && !d->done && !strcmp(name, d->ent.d_name) && !(d->fd.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) {   // the entry just read
        memset(st, 0, sizeof *st);
        int dir = (d->fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
        st->st_mode = (unsigned short)(dir ? _S_IFDIR | 0755 : _S_IFREG | ((d->fd.dwFileAttributes & FILE_ATTRIBUTE_READONLY) ? 0444 : 0644));
        st->st_size = dir ? 0 : (long long)((uint64_t)d->fd.nFileSizeHigh << 32 | d->fd.nFileSizeLow);
        st->st_mtime = time_of(d->fd.ftLastWriteTime); st->st_atime = time_of(d->fd.ftLastAccessTime); st->st_ctime = time_of(d->fd.ftCreationTime);
        st->st_nlink = 1;
        return 0;
    }
    char p[MAX_PATH * 8]; snprintf(p, sizeof p, "%s/%s", d->path, name);
    return stat(p, st);
}
