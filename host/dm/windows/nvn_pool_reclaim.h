// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <stdint.h>
#include <string.h>

// Owner-local, bounded import-allocation planner. All indices and handles stay
// pinned by the pool write lock until the selected buffers are retired.
// A two-range sparse bridge joins both allocations into one reclaim cohort.
// Missing holders or one recent holder pins the entire connected component.
#define NVNP_RANGES 512u
typedef struct nvnp_range {
    uint64_t bytes, used, group_bytes, group_used;
    unsigned parent, expected, seen, blocked, group_blocked, selected;
} nvnp_range;
typedef struct nvnp_plan { nvnp_range r[NVNP_RANGES]; unsigned n; } nvnp_plan;
static inline void nvnp_init(nvnp_plan *p,unsigned n) {
    memset(p,0,sizeof *p); p->n=n<=NVNP_RANGES?n:NVNP_RANGES;
    for(unsigned i=0;i<p->n;i++) p->r[i].parent=i;
}
static inline unsigned nvnp_root(nvnp_plan *p,unsigned i) {
    while(p->r[i].parent!=i) { p->r[i].parent=p->r[p->r[i].parent].parent; i=p->r[i].parent; }
    return i;
}
static inline void nvnp_set(nvnp_plan *p,unsigned i,uint64_t bytes,int holders,int usable) {
    if(i>=p->n) return;
    p->r[i].bytes=bytes; p->r[i].expected=holders<0?0:(unsigned)holders;
    p->r[i].blocked=!usable || holders<0;
}
static inline void nvnp_bind(nvnp_plan *p,unsigned a,unsigned b,uint64_t used,int cold) {
    // 1-based pool-table indices; zero means no range on that side.
    unsigned ids[2]={a,b}; int invalid=(a>p->n || b>p->n);
    for(unsigned k=0;k<2;k++) if(ids[k] && ids[k]<=p->n) {
        nvnp_range *r=&p->r[ids[k]-1]; r->seen++;
        if(used>r->used) r->used=used;
        r->blocked|=!cold || invalid;
    }
    if(a && b && !invalid) {
        unsigned x=nvnp_root(p,a-1), y=nvnp_root(p,b-1);
        if(x!=y) p->r[y].parent=x;
    }
}
static inline uint64_t nvnp_select(nvnp_plan *p,uint64_t want) {
    for(unsigned i=0;i<p->n;i++) {
        nvnp_range *r=&p->r[i], *g=&p->r[nvnp_root(p,i)];
        g->group_blocked|=r->blocked || r->seen!=r->expected;
        if(UINT64_MAX-g->group_bytes<r->bytes) g->group_blocked=1;
        else g->group_bytes+=r->bytes;
        if(r->used>g->group_used) g->group_used=r->used;
    }
    uint64_t got=0;
    while(got<want) {
        unsigned oldest=p->n;
        for(unsigned i=0;i<p->n;i++) {
            const nvnp_range *g=&p->r[i];
            if(g->parent!=i || g->selected || g->group_blocked || !g->group_bytes) continue;
            if(oldest==p->n || g->group_used<p->r[oldest].group_used) oldest=i;
        }
        if(oldest==p->n) break;
        p->r[oldest].selected=1;
        uint64_t n=p->r[oldest].group_bytes;
        got=UINT64_MAX-got<n?UINT64_MAX:got+n;
    }
    return got;
}
static inline int nvnp_selected(nvnp_plan *p,unsigned one_based) {
    return one_based && one_based<=p->n && p->r[nvnp_root(p,one_based-1)].selected;
}
