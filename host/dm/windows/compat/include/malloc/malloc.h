// Windows port: malloc zones are the one CRT heap (an own zone only kept Unreal's allocations apart for accounting). Zone allocations come from
// _aligned_malloc (a zone pointer may be memalign'ed, and _aligned_free must free every one of them); malloc_size works on zone pointers only.
#pragma once
#include <malloc.h>
#include <stdlib.h>
typedef struct dm_malloc_zone { int unused; } malloc_zone_t;
static inline malloc_zone_t *malloc_create_zone(size_t start, unsigned flags) { (void)start; (void)flags; static malloc_zone_t z; return &z; }
static inline malloc_zone_t *malloc_default_zone(void) { static malloc_zone_t z; return &z; }
static inline void malloc_set_zone_name(malloc_zone_t *z, const char *n) { (void)z; (void)n; }
static inline void *malloc_zone_malloc(malloc_zone_t *z, size_t n) { (void)z; return _aligned_malloc(n ? n : 1, 16); }
static inline void *malloc_zone_calloc(malloc_zone_t *z, size_t c, size_t n) { (void)z; void *p = _aligned_malloc(c * n ? c * n : 1, 16); if (p) __builtin_memset(p, 0, c * n); return p; }
static inline void *malloc_zone_memalign(malloc_zone_t *z, size_t a, size_t n) { (void)z; return _aligned_malloc(n ? n : 1, a < 16 ? 16 : a); }
static inline void *malloc_zone_realloc(malloc_zone_t *z, void *p, size_t n) { (void)z; return _aligned_realloc(p, n ? n : 1, 16); }
static inline void malloc_zone_free(malloc_zone_t *z, void *p) { (void)z; _aligned_free(p); }
static inline size_t malloc_size(const void *p) { return p ? _aligned_msize((void *)p, 16, 0) : 0; }
