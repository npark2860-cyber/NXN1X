// Windows port: BSD string calls on top of mingw's <string.h> (dm/windows/compat/posix_win.c)
#pragma once
#include_next <string.h>
#include <stddef.h>
size_t strlcpy(char *dst, const char *src, size_t cap);
size_t strlcat(char *dst, const char *src, size_t cap);
void *memmem(const void *hay, size_t hn, const void *needle, size_t nn);
char *strsep(char **s, const char *delim);
char *strnstr(const char *hay, const char *needle, size_t n);
char *strcasestr(const char *hay, const char *needle);
