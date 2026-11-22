// Windows port: POSIX / BSD calls on top of mingw's <stdlib.h> (dm/windows/compat/posix_win.c). arc4random reads the system RNG (BCryptGenRandom).
#pragma once
#include_next <stdlib.h>
#include <stdint.h>
#include <stddef.h>
int setenv(const char *name, const char *value, int overwrite);
int unsetenv(const char *name);
int mergesort(void *base, size_t n, size_t size, int (*cmp)(const void *, const void *));
uint32_t arc4random(void);
void arc4random_buf(void *buf, size_t n);
uint32_t arc4random_uniform(uint32_t bound);
const char *getprogname(void);
char *realpath(const char *path, char *resolved);
