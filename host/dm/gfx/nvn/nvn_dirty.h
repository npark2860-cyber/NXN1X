// Write-protect dirty-page tracking for guest memory (nvn_dirty.c)
#pragma once
#include <stddef.h>
#include <stdint.h>

void nvn_dirty_add_range(const void *mem, size_t size);           // Register a memory pool as a tracked range (the first call registers the page-fault hook and the host pre-write hook)
void nvn_dirty_stat(int *live, int *peak, int *cap);                // number of registered memory pools, peak, tracking table capacity (for the memory readout line)
void nvn_dirty_remove_range(const void *mem);                     // Pool destruction: restore protected pages to writable and mark invalid
int nvn_dirty_tracked(const void *addr, size_t len);
int nvn_dirty_gpu_write(const void *addr, size_t len);  /* s56b2: PROT pages are stamped and stay protected; only BUSY returns 1 */
uint64_t nvn_dirty_gpu_write_prot(void);   // s56b2: protected pages stamped without unprotecting since the last call   // s56b: 0 = no page of the range protected/busy (stamped), 1 = some is, -1 = untracked
void nvn_dirty_protect(const void *addr, size_t len);              // Set the covered pages read-only
void nvn_dirty_protect_sparse(const void *addr, size_t len);       // The same, for the pages behind a placement sparse tile: a later write to one moves nvn_dirty_sparse_writes
uint64_t nvn_dirty_sparse_writes(void);                            // Writes so far to pages protected that way
uint64_t nvn_dirty_now(void);                                      // Current generation: take the content's stamp before nvn_dirty_protect and before reading it
int nvn_dirty_since(const void *addr, size_t len, uint64_t stamp); // Covered pages were written after the stamp -> 1
const void *nvn_dirty_span(const void *addr, size_t len);          // The tracked range holding the span (a handle to keep; NULL = untracked)
int nvn_dirty_since_cached(const void **span, const void *addr, size_t len, uint64_t stamp);   // nvn_dirty_since keeping the last range found in *span
int nvn_dirty_since_span(const void *span, const void *addr, size_t len, uint64_t stamp);   // nvn_dirty_since through a kept handle (rechecked, scanned again only if it no longer holds the span)
void nvn_dirty_stats(uint64_t *faults, uint64_t *preps);           // Cumulative: number of page faults triggered by game writes, number of pages unprotected before host writes
void nvn_dirty_mp_stats(uint64_t *calls, uint64_t *pages, uint64_t *ns);   // Cumulative: this module's mprotect call count, pages covered, time spent
void nvn_dirty_slot_stats(int *live, int *used_slots, uint64_t *adds_total); // Currently live slots, slot table high-water mark, total number of registrations
void nvn_dirty_set_snap_hook(void (*fn)(uintptr_t page, size_t len, uint64_t now));   // Called with each protected page just before it is made writable (still read-only: its content as protected), and the clock then; from the fault path, so the hook may use only atomics and memcpy
