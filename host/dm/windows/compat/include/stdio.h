// Windows port: the POSIX stream locks on top of mingw's <stdio.h> (the UCRT's _lock_file / _unlock_file, recursive as POSIX's are).
// ftrylockfile has no CRT counterpart: it takes the lock (blocking) and reports success.
#pragma once
#include_next <stdio.h>
#define flockfile(f) _lock_file(f)
#define funlockfile(f) _unlock_file(f)
static inline int ftrylockfile(FILE *f) { _lock_file(f); return 0; }
