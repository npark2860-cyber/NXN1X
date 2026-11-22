// SPDX-License-Identifier: GPL-3.0-or-later
// Bounded content provenance ledger. No Vulkan, guest reads/writes or authority decisions.
#include "nvn_memory_content.h"
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

_Static_assert(ATOMIC_LLONG_LOCK_FREE == 2 && ATOMIC_INT_LOCK_FREE == 2,
               "F2 fault producers require lock-free atomics");
#ifndef NVNC_QUEUE_CAP
#define NVNC_QUEUE_CAP 32768u
#endif
#ifndef NVNC_REGION_CAP
#define NVNC_REGION_CAP 131072u
#endif
#ifndef NVNC_OBJECT_CAP
#define NVNC_OBJECT_CAP 16384u
#endif
#ifndef NVNC_INTENT_CAP
#define NVNC_INTENT_CAP 32768u
#endif
#ifndef NVNC_RANGE_CAP
#define NVNC_RANGE_CAP 65536u
#endif
#define SUB_CAP 64u
#define BLOCK_SIZE UINT64_C(65536)
#define NONE UINT32_MAX
#define CLOSED (UINT64_C(1) << 63)
enum { E_POOL_ADD=1, E_POOL_DEL, E_IMAGE_ADD, E_IMAGE_DEL, E_CPU,
       E_SUB_BEGIN, E_INTENT_IMAGE, E_INTENT_RANGE, E_ACCEPT, E_ERROR,
       E_GPU_DONE, E_CPU_DONE, E_PROTECTED, E_UNKNOWN };
typedef struct event {
    uint64_t id, address, length, object, epoch, life, serial;
    uint32_t kind, part, aux;
    int32_t result;
} event;
typedef struct slot { _Atomic uint64_t sequence; event data; } slot;
// Range writers never occupy the lifecycle queue. A slot is exclusively owned
// with one bounded CAS, including while a consumer resolves its current slice.
// Large ranges retain a cursor; new observations request another pass instead
// of repeatedly discarding the tail. All records remain diagnostic/Unknown.
typedef struct range_slot {
    _Atomic unsigned state; // 0 free, 1 owned, 2 ready
    event data;
    uint64_t cursor, pass_serial;
    unsigned extra;
} range_slot;
typedef struct pool_record { uint64_t key, id, address, length; int live; } pool_record;
typedef struct image_record {
    uint64_t id, pool_key, pool_id, address, length, layout, epoch;
    uint64_t recorded, accepted;
    int live;
} image_record;
typedef struct region {
    nvnc_region_snapshot value;
    uint64_t virtual_block;
} region;
typedef struct intent {
    event e;
    uint64_t pool_id;
    uint32_t next;
} intent;
typedef struct submission {
    uint64_t id, accepted;
    uint32_t first, last, count;
    int gpu_done;
} submission;
typedef struct ledger {
    slot queue[NVNC_QUEUE_CAP];
    range_slot ranges[NVNC_RANGE_CAP];
    unsigned range_scan;
    pool_record pools[NVNC_OBJECT_CAP];
    image_record images[NVNC_OBJECT_CAP];
    region regions[NVNC_REGION_CAP];
    intent intents[NVNC_INTENT_CAP];
    submission subs[SUB_CAP];
    uint32_t free_intent;
    uint64_t read;
    nvnc_stats stats;
} ledger;

static nvnc_resolver resolver;
static ledger *state; // published once, never freed while fault producers might exist
static _Atomic uint64_t admission = CLOSED, write_pos, next_id = 1, life_epoch = 1;
static _Atomic uint64_t lost, unknown_epoch, unknown_reasons;
static _Atomic uint64_t serial_number, range_pending, range_raw, range_coalesced;
static _Atomic uint64_t range_dropped, queue_dropped;
static _Atomic uint64_t raw_cpu, raw_fault, raw_flush, raw_writeback;
static _Atomic int identity_exhausted;
static int started;
static pthread_mutex_t consume_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t drain_mutex = PTHREAD_MUTEX_INITIALIZER;
static _Thread_local unsigned write_origin;
static _Thread_local uint64_t recording;
static _Thread_local unsigned part;
static _Thread_local struct { uint64_t sub, image, address, size; unsigned part, origin; } recent[128];

static uint64_t hash64(uint64_t v) {
    v ^= v >> 30; v *= UINT64_C(0xbf58476d1ce4e5b9);
    v ^= v >> 27; v *= UINT64_C(0x94d049bb133111eb);
    return v ^ (v >> 31);
}
static void loss(void) {
    atomic_fetch_add_explicit(&lost, 1, memory_order_relaxed);
    // Saturate, rather than allowing generation wrap to produce a clean epoch.
    atomic_store_explicit(&unknown_epoch, 1, memory_order_release);
    atomic_fetch_or_explicit(&unknown_reasons, NVNC_UNKNOWN_OVERFLOW, memory_order_relaxed);
}
static uint64_t identity(void) {
    if (atomic_load_explicit(&identity_exhausted,memory_order_acquire)) return 0;
    uint64_t n = atomic_fetch_add_explicit(&next_id, 1, memory_order_relaxed);
    if (!n || n >= CLOSED) { atomic_store_explicit(&identity_exhausted,1,memory_order_release); loss(); return 0; }
    return n;
}
static void advance_lifetime(void) {
    uint64_t old=atomic_load_explicit(&life_epoch,memory_order_acquire);
    for (unsigned i=0;i<8;i++) {
        if (old>=CLOSED-1) break;
        if (atomic_compare_exchange_weak_explicit(&life_epoch,&old,old+1,memory_order_acq_rel,memory_order_acquire)) return;
    }
    loss(); // never wrap and reuse an earlier lifetime epoch
}
int nvnc_active(void) { return !(atomic_load_explicit(&admission, memory_order_acquire) & CLOSED); }
int nvnc_set_resolver(const nvnc_resolver *r) {
    if (started || !r || r->version != NVNC_ABI || r->size < sizeof *r || !r->resolve || !r->epoch) return 0;
    resolver = *r;
    return 1;
}
void nvnc_start(int native_nvn2) {
    if (!native_nvn2 || started) return;
    started = 1;
    const char *e = getenv("NXN1X_NVN_CONTENT");
    if ((e && !atoi(e)) || !resolver.resolve || !resolver.epoch) return;
    state = calloc(1, sizeof *state);
    if (!state) { loss(); return; }
    for (unsigned i=0; i<NVNC_QUEUE_CAP; i++) atomic_init(&state->queue[i].sequence, i);
    for (unsigned i=0; i<NVNC_INTENT_CAP; i++) state->intents[i].next = i+1 < NVNC_INTENT_CAP ? i+1 : NONE;
    state->free_intent = 0;
    state->stats.metadata_bytes = sizeof *state;
    atomic_store_explicit(&admission, 0, memory_order_release);
}
void nvnc_stop(void) {
    atomic_fetch_or_explicit(&admission, CLOSED, memory_order_acq_rel);
    // Lifecycle caller holds no owner lock. Finish a normal-thread mapping
    // resolver before its owner disappears. Fault producers never take this lock.
    pthread_mutex_lock(&drain_mutex);
    pthread_mutex_unlock(&drain_mutex);
    // No free: an admitted fault producer may still finish in the bounded queue.
}
static int enqueue(event e) {
    if (!nvnc_active()) return 0;
    uint64_t reader = atomic_fetch_add_explicit(&admission, 1, memory_order_acq_rel);
    if (reader & CLOSED) { atomic_fetch_sub_explicit(&admission, 1, memory_order_release); return 0; }
    int ok = 0;
    uint64_t pos = atomic_load_explicit(&write_pos, memory_order_relaxed);
    for (unsigned attempt=0; attempt<8; attempt++) {
        if (pos >= CLOSED) break;
        slot *s = &state->queue[pos % NVNC_QUEUE_CAP];
        uint64_t seq = atomic_load_explicit(&s->sequence, memory_order_acquire);
        if (seq < pos) break;
        if (seq != pos) { pos = atomic_load_explicit(&write_pos, memory_order_relaxed); continue; }
        if (!atomic_compare_exchange_weak_explicit(&write_pos, &pos, pos+1,
                                                  memory_order_relaxed, memory_order_relaxed)) continue;
        e.serial = atomic_fetch_add_explicit(&serial_number,1,memory_order_relaxed)+1;
        s->data = e;
        atomic_store_explicit(&s->sequence, pos+1, memory_order_release);
        ok = 1;
        break;
    }
    if (!ok) { atomic_fetch_add_explicit(&queue_dropped,1,memory_order_relaxed); loss(); }
    atomic_fetch_sub_explicit(&admission, 1, memory_order_release);
    return ok;
}
static void range_enqueue(event e, unsigned extra) {
    if (!nvnc_active() || !e.length) return;
    atomic_fetch_add_explicit(&range_raw,1,memory_order_relaxed);
    if (e.length>UINT64_MAX-e.address) { loss(); atomic_fetch_add(&range_dropped,1); return; }
    if (!e.serial) e.serial=atomic_fetch_add_explicit(&serial_number,1,memory_order_relaxed)+1;
    // Small writes coalesce only inside one block and one origin/epoch. Widened
    // byte coverage is explicitly partial; it is never a valid-source proof.
    uint64_t base=e.address, length=e.length;
    if (length<=BLOCK_SIZE-(base&(BLOCK_SIZE-1))) {
        base &= ~(BLOCK_SIZE-1); length=BLOCK_SIZE; extra|=NVNC_UNKNOWN_PARTIAL;
    }
    size_t h=hash64(base ^ hash64(length) ^ ((uint64_t)e.aux<<48))%NVNC_RANGE_CAP;
    for (unsigned k=0;k<16;k++) {
        range_slot *r=&state->ranges[(h+k)%NVNC_RANGE_CAP];
        unsigned old=atomic_load_explicit(&r->state,memory_order_acquire);
        if (old==1 || !atomic_compare_exchange_strong_explicit(&r->state,&old,1,memory_order_acquire,memory_order_relaxed)) continue;
        if (!old) {
            e.address=base; e.length=length; r->data=e; r->cursor=0; r->pass_serial=e.serial; r->extra=extra;
            atomic_fetch_add_explicit(&range_pending,1,memory_order_relaxed);
            atomic_store_explicit(&r->state,2,memory_order_release); return;
        }
        if (r->data.address==base && r->data.length==length && r->data.aux==e.aux &&
            r->data.epoch==e.epoch && r->data.life==e.life) {
            if (r->data.serial<e.serial) r->data.serial=e.serial;
            // A not-yet-started pass may immediately use the newest version.
            if (!r->cursor) r->pass_serial=r->data.serial;
            r->extra|=extra;
            atomic_fetch_add_explicit(&range_coalesced,1,memory_order_relaxed);
            atomic_store_explicit(&r->state,2,memory_order_release); return;
        }
        atomic_store_explicit(&r->state,2,memory_order_release);
    }
    atomic_fetch_add_explicit(&range_dropped,1,memory_order_relaxed); loss();
}
static event make_event(unsigned kind) {
    event e = {0}; e.kind = kind;
    e.epoch = resolver.epoch ? resolver.epoch() : UINT64_MAX;
    e.life = atomic_load_explicit(&life_epoch, memory_order_acquire);
    return e;
}
unsigned nvnc_origin_push(unsigned origin) { unsigned old = write_origin; write_origin = origin; return old; }
void nvnc_origin_pop(unsigned old) { write_origin = old; }
unsigned nvnc_origin(void) { return write_origin ? write_origin : NVNC_CPU_PRESTORE; }
void nvnc_cpu(uint64_t a, uint64_t n, unsigned origin) {
    if (!n || !nvnc_active()) return;
    event e = make_event(E_CPU); e.address = a; e.length = n; e.aux = origin;
    if (origin==NVNC_GPU_WRITEBACK) atomic_fetch_add_explicit(&raw_writeback,1,memory_order_relaxed);
    else {
        atomic_fetch_add_explicit(&raw_cpu,1,memory_order_relaxed);
        if (origin==NVNC_CPU_FAULT) atomic_fetch_add_explicit(&raw_fault,1,memory_order_relaxed);
        if (origin==NVNC_CPU_FLUSH) atomic_fetch_add_explicit(&raw_flush,1,memory_order_relaxed);
    }
    range_enqueue(e,0);
}
void nvnc_unknown(unsigned reason) {
    if (!nvnc_active()) return;
    event e = make_event(E_UNKNOWN); e.aux = reason; enqueue(e);
}
void nvnc_pool_add(uint64_t key, uint64_t a, uint64_t n) {
    if (!nvnc_active()) return;
    advance_lifetime();
    event e = make_event(E_POOL_ADD); e.id = identity(); e.object = key; e.address = a; e.length = n; enqueue(e);
}
void nvnc_pool_del(uint64_t key) {
    if (!nvnc_active()) return;
    advance_lifetime();
    event e = make_event(E_POOL_DEL); e.object = key; enqueue(e);
}
uint64_t nvnc_image_add(uint64_t key, uint64_t a, uint64_t n, uint64_t layout) {
    if (!nvnc_active()) return 0;
    event e = make_event(E_IMAGE_ADD); e.id = identity(); e.object = key;
    e.address = a; e.length = n; e.part = (uint32_t)layout; e.aux = (uint32_t)(layout >> 32);
    return enqueue(e) ? e.id : 0;
}
void nvnc_image_del(uint64_t id) { if (id && nvnc_active()) { event e=make_event(E_IMAGE_DEL); e.id=id; enqueue(e); } }
void nvnc_pool_protected(uint64_t id, uint64_t bytes, int written, int pool_written) {
    if (!id || !nvnc_active()) return;
    event e=make_event(E_PROTECTED); e.id=id; e.length=bytes; e.aux=written != 0; e.part=pool_written != 0; enqueue(e);
}
uint64_t nvnc_sub_begin(void) {
    recording=0; part=1;
    if (!nvnc_active()) return 0;
    event e=make_event(E_SUB_BEGIN); e.id=identity();
    if (enqueue(e)) recording=e.id;
    return recording;
}
void nvnc_part(unsigned n) { part=n; }
static void record_intent(uint64_t image, uint64_t a, uint64_t n, unsigned origin) {
    if (!nvnc_active()) return;
    if (!recording || !part || part>32) { nvnc_unknown(NVNC_UNKNOWN_ORDER); return; }
    unsigned at=(unsigned)hash64(image ^ a ^ (n<<1) ^ ((uint64_t)origin<<32)) & 127;
    if (recent[at].sub==recording && recent[at].part==part && recent[at].image==image &&
        recent[at].address==a && recent[at].size==n && recent[at].origin==origin) return;
    event e=make_event(image ? E_INTENT_IMAGE : E_INTENT_RANGE);
    e.id=recording; e.object=image; e.address=a; e.length=n; e.part=part; e.aux=origin;
    if (enqueue(e)) {
        recent[at].sub=recording; recent[at].part=part; recent[at].image=image;
        recent[at].address=a; recent[at].size=n;
        recent[at].origin=origin;
    }
}
void nvnc_image_intent(uint64_t id) { if (id) record_intent(id,0,0,NVNC_DEVICE_WRITE); }
void nvnc_image_upload(uint64_t id) { if (id) record_intent(id,0,0,NVNC_CPU_UPLOAD); }
void nvnc_range_intent(uint64_t a, uint64_t n) { if (n) record_intent(0,a,n,NVNC_DEVICE_WRITE); }
void nvnc_accept(uint64_t sub, unsigned first, unsigned last, int32_t result) {
    if (!sub || !nvnc_active()) return;
    event e=make_event(E_ACCEPT); e.id=sub; e.part=first; e.aux=last; e.result=result; enqueue(e);
}
void nvnc_error(unsigned p, int32_t result) {
    if (!recording || result>=0 || !nvnc_active()) return;
    event e=make_event(E_ERROR); e.id=recording; e.part=p; e.result=result; enqueue(e);
}
void nvnc_sub_end(void) { recording=0; part=0; }
void nvnc_gpu_done(uint64_t sub) { if (sub && nvnc_active()) { event e=make_event(E_GPU_DONE); e.id=sub; enqueue(e); } }
void nvnc_cpu_done(uint64_t sub) { if (sub && nvnc_active()) { event e=make_event(E_CPU_DONE); e.id=sub; enqueue(e); } }

static pool_record *pool_get(uint64_t key, int create) {
    if (!key) return NULL;
    size_t h=(size_t)hash64(key) % NVNC_OBJECT_CAP;
    pool_record *dead=NULL;
    for (unsigned k=0; k<64; k++) {
        pool_record *p=&state->pools[(h+k)%NVNC_OBJECT_CAP];
        if (p->key==key) return p;
        if (!p->live && !dead) dead=p;
        if (!p->key) break;
    }
    if (create && dead) { memset(dead,0,sizeof *dead); dead->key=key; return dead; }
    if (create) loss();
    return NULL;
}
static image_record *image_get(uint64_t id, int create) {
    if (!id) return NULL;
    size_t h=(size_t)hash64(id)%NVNC_OBJECT_CAP;
    image_record *dead=NULL;
    for (unsigned k=0; k<64; k++) {
        image_record *p=&state->images[(h+k)%NVNC_OBJECT_CAP];
        if (p->id==id) return p;
        if (!p->live && !dead) dead=p;
        if (!p->id) break;
    }
    if (create && dead) { memset(dead,0,sizeof *dead); dead->id=id; return dead; }
    if (create) loss();
    return NULL;
}
static region *region_get(uint64_t backing, uint64_t offset, int create) {
    offset &= ~(BLOCK_SIZE-1);
    size_t h=(size_t)hash64(backing ^ hash64(offset))%NVNC_REGION_CAP;
    for (unsigned k=0; k<128; k++) {
        region *r=&state->regions[(h+k)%NVNC_REGION_CAP];
        if (r->value.backing==backing && r->value.block_offset==offset) return r;
        if (!r->value.backing) {
            if (!create) return NULL;
            r->value.backing=backing; r->value.block_offset=offset;
            // First-write faults do not cover later stores to writable pages.
            r->value.unknown=NVNC_UNKNOWN_NCE;
            state->stats.regions++;
            return r;
        }
    }
    if (create) loss();
    return NULL;
}
static void unresolved(uint64_t bytes) {
    state->stats.unresolved_bytes += bytes;
    atomic_store_explicit(&unknown_epoch,1,memory_order_release);
    atomic_fetch_or_explicit(&unknown_reasons,NVNC_UNKNOWN_MAPPING|NVNC_UNKNOWN_ORDER,memory_order_relaxed);
}
// At most the caller's region budget is resolved. Large/unknown spans remain
// Unknown; no scan of an entire pool and no memory-content comparison is added.
static uint64_t apply_range(const event *e, unsigned origin, unsigned extra, unsigned *budget) {
    uint64_t a=e->address, n=e->length;
    if (!n) return 0;
    if (n>UINT64_MAX-a || e->epoch==UINT64_MAX || e->epoch!=resolver.epoch()) { unresolved(n); return e->length; }
    unsigned pieces=0;
    while (n && *budget && pieces<64) {
        uint64_t max=BLOCK_SIZE-(a&(BLOCK_SIZE-1)); if (max>n) max=n;
        nvnc_backing b={0}; (*budget)--; pieces++;
        // drain_mutex pins the sole consumer; do not hold the content lock
        // while the resolver takes the native sparse registry's owner lock.
        pthread_mutex_unlock(&consume_mutex);
        int resolved=resolver.resolve(a,max,&b);
        pthread_mutex_lock(&consume_mutex);
        if (!resolved || !b.id || !b.length || b.length>max ||
            b.offset>UINT64_MAX-b.length || e->epoch!=resolver.epoch()) { unresolved(n); return e->length; }
        uint64_t size=b.length;
        if (size>BLOCK_SIZE-(b.offset&(BLOCK_SIZE-1))) size=BLOCK_SIZE-(b.offset&(BLOCK_SIZE-1));
        region *r=region_get(b.id,b.offset,1);
        if (!r) { unresolved(n); return e->length; }
        uint64_t va=a-(b.offset&(BLOCK_SIZE-1));
        if (r->virtual_block && r->virtual_block!=va) state->stats.alias_hits++;
        r->virtual_block=va;
        if (r->value.map_epoch && r->value.map_epoch!=e->epoch) r->value.unknown|=NVNC_UNKNOWN_MAPPING;
        r->value.map_epoch=e->epoch;
        if (r->value.identity_epoch && r->value.identity_epoch!=e->life) r->value.unknown|=NVNC_UNKNOWN_IDENTITY;
        r->value.identity_epoch=e->life;
        r->value.unknown|=extra;
        if (size!=BLOCK_SIZE || (b.offset&(BLOCK_SIZE-1))) r->value.unknown|=NVNC_UNKNOWN_PARTIAL;
        if (origin!=NVNC_UNCLASSIFIED) r->value.all_gen=r->value.all_gen>e->serial ? r->value.all_gen : e->serial;
        if (origin==NVNC_GPU_WRITEBACK) {
            r->value.writeback_gen=r->value.writeback_gen>e->serial ? r->value.writeback_gen : e->serial; state->stats.writeback_bytes+=size;
            r->value.unknown|=NVNC_UNKNOWN_PRESTORE|NVNC_UNKNOWN_ORDER;
        } else if (origin==NVNC_DEVICE_WRITE) {
            r->value.device_gen=r->value.device_gen>e->serial ? r->value.device_gen : e->serial; state->stats.device_bytes+=size;
            // An accepted attachment/storage binding is possible partial work,
            // not proof that a whole image or even one texel was overwritten.
            r->value.unknown|=NVNC_UNKNOWN_PARTIAL|NVNC_UNKNOWN_ORDER;
        } else if (origin==NVNC_UNCLASSIFIED) {
            r->value.unknown|=NVNC_UNKNOWN_RAW;
        } else {
            r->value.cpu_gen=r->value.cpu_gen>e->serial ? r->value.cpu_gen : e->serial; state->stats.cpu_bytes+=size;
            if (origin!=NVNC_CPU_FLUSH) r->value.unknown|=NVNC_UNKNOWN_PRESTORE;
        }
        a+=size; n-=size;
    }
    if (n) state->stats.bounded_yields++;
    return e->length-n;
}
static submission *sub_get(uint64_t id) {
    if (!id) return NULL;
    for (unsigned i=0;i<SUB_CAP;i++) if (state->subs[i].id==id) return &state->subs[i];
    return NULL;
}
static void first_error(const event *e) {
    if (e->result<0 && !state->stats.first_error) {
        state->stats.first_error=e->result; state->stats.first_error_sub=e->id;
        state->stats.first_error_part=e->part;
    }
}
static void consume(const event *e, unsigned *budget) {
    (void)budget;
    pool_record *p; image_record *im; submission *s;
    switch(e->kind) {
    case E_POOL_ADD:
        p=pool_get(e->object,1);
        if (p) { if (!p->live) state->stats.pools++; p->live=1; p->id=e->id; p->address=e->address; p->length=e->length; }
        break;
    case E_POOL_DEL:
        p=pool_get(e->object,0); if (p && p->live) { p->live=0; state->stats.pools--; }
        break;
    case E_IMAGE_ADD:
        im=image_get(e->id,1); p=pool_get(e->object,0);
        if (im) { im->live=1; im->pool_key=e->object; im->pool_id=p && p->live ? p->id : 0;
            im->address=e->address; im->length=e->length; im->layout=((uint64_t)e->aux<<32)|e->part; im->epoch=e->epoch; state->stats.images++; }
        break;
    case E_IMAGE_DEL:
        im=image_get(e->id,0); if (im && im->live) { im->live=0; state->stats.images--; }
        break;
    case E_SUB_BEGIN:
        s=NULL;
        for (unsigned i=0;i<SUB_CAP;i++) if (!state->subs[i].id) { s=&state->subs[i]; break; }
        if (!s || !e->id) { loss(); break; }
        *s=(submission){.id=e->id,.first=NONE,.last=NONE};
        break;
    case E_INTENT_IMAGE: case E_INTENT_RANGE: {
        s=sub_get(e->id);
        if (!s || s->gpu_done || s->count>=2048 || state->free_intent==NONE) { loss(); break; }
        event write=*e; uint64_t pool_id=0;
        if (e->kind==E_INTENT_IMAGE) {
            im=image_get(e->object,0); p=im ? pool_get(im->pool_key,0) : NULL;
            if (!im || !im->live || !p || !p->live || im->pool_id!=p->id) { loss(); break; }
            im->recorded+=e->aux!=NVNC_CPU_UPLOAD;
            state->stats.upload_intents+=e->aux==NVNC_CPU_UPLOAD;
            write.address=im->address; write.length=im->length; pool_id=im->pool_id;
        }
        uint32_t i=state->free_intent; intent *in=&state->intents[i]; state->free_intent=in->next;
        *in=(intent){.e=write,.pool_id=pool_id,.next=NONE};
        if (s->last!=NONE) state->intents[s->last].next=i; else s->first=i;
        s->last=i; s->count++; state->stats.intents++;
        break;
    }
    case E_ACCEPT:
        first_error(e); s=sub_get(e->id);
        if (!s || e->result || !e->part || e->aux<e->part || e->aux>32) { if (!s) loss(); break; }
        for (unsigned k=e->part; k<=e->aux; k++) s->accepted|=UINT64_C(1)<<(k-1);
        break;
    case E_ERROR: first_error(e); break;
    case E_GPU_DONE:
        s=sub_get(e->id); if (!s || s->gpu_done) { loss(); break; }
        s->gpu_done=1; state->stats.gpu_completed++;
        for (uint32_t i=s->first; i!=NONE; i=state->intents[i].next) {
            intent *in=&state->intents[i]; event write=in->e; write.serial=e->serial;
            int accepted=write.part && write.part<=32 && (s->accepted&(UINT64_C(1)<<(write.part-1)));
            if (accepted) state->stats.accepted_intents++; else state->stats.rejected_intents++;
            if (in->e.kind==E_INTENT_IMAGE) {
                im=image_get(in->e.object,0); p=im ? pool_get(im->pool_key,0) : NULL;
                if (!im || !im->live || !p || !p->live || p->id!=in->pool_id) { unresolved(write.length); continue; }
                im->accepted+=accepted;
            }
            write.aux=accepted ? NVNC_DEVICE_WRITE : NVNC_UNCLASSIFIED;
            range_enqueue(write,accepted ? 0 : NVNC_UNKNOWN_FAILED);
        }
        break;
    case E_CPU_DONE:
        s=sub_get(e->id); if (!s || !s->gpu_done) { loss(); break; }
        for (uint32_t i=s->first; i!=NONE;) { uint32_t next=state->intents[i].next;
            state->intents[i].next=state->free_intent; state->free_intent=i; i=next; }
        memset(s,0,sizeof *s); state->stats.cpu_completed++;
        break;
    case E_PROTECTED:
        state->stats.pool_protected_checks++; state->stats.pool_protected_bytes+=e->length;
        state->stats.pool_protected_true+=e->part;
        im=image_get(e->id,0);
        if (!e->aux && im && im->live && !im->recorded) state->stats.no_image_write_checks++;
        break;
    case E_UNKNOWN:
        state->stats.unsupported_events++;
        atomic_store_explicit(&unknown_epoch,1,memory_order_release);
        atomic_fetch_or_explicit(&unknown_reasons,e->aux,memory_order_relaxed);
        break;
    default: loss(); break;
    }
}
static uint64_t monotonic_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (uint64_t)t.tv_sec*UINT64_C(1000000000)+(uint64_t)t.tv_nsec;
}
void nvnc_drain(unsigned event_limit, unsigned region_limit) {
    if (!nvnc_active() || !event_limit || !region_limit) return;
    if (pthread_mutex_trylock(&drain_mutex)) return;
    if (!nvnc_active()) { pthread_mutex_unlock(&drain_mutex); return; }
    pthread_mutex_lock(&consume_mutex);
    uint64_t start=monotonic_ns(); unsigned count=0;
    while (count<event_limit) {
        uint64_t pos=state->read; slot *s=&state->queue[pos%NVNC_QUEUE_CAP];
        if (atomic_load_explicit(&s->sequence,memory_order_acquire)!=pos+1) break;
        event e=s->data;
        atomic_store_explicit(&s->sequence,pos+NVNC_QUEUE_CAP,memory_order_release);
        state->read++; consume(&e,&region_limit); count++;
        if (!(count & 31) && monotonic_ns()-start >= UINT64_C(1000000)) {
            state->stats.bounded_yields++; break;
        }
    }
    state->stats.consumed+=count;
    // Fair bounded scan. A continuation is not lost simply because this frame
    // has exhausted its region budget. Lifecycle consumption is independent.
    for (unsigned scanned=0; scanned<NVNC_RANGE_CAP && region_limit; scanned++) {
        if (scanned && !(scanned&31) && monotonic_ns()-start>=UINT64_C(1000000)) break;
        range_slot *r=&state->ranges[state->range_scan++%NVNC_RANGE_CAP];
        unsigned ready=2;
        if (!atomic_compare_exchange_strong_explicit(&r->state,&ready,1,memory_order_acquire,memory_order_relaxed)) continue;
        event e=r->data; e.address+=r->cursor; e.length-=r->cursor; e.serial=r->pass_serial;
        r->cursor+=apply_range(&e,e.aux,r->extra,&region_limit);
        if (r->cursor==r->data.length) {
            state->stats.range_passes++;
            if (r->pass_serial!=r->data.serial) { r->cursor=0; r->pass_serial=r->data.serial; }
            else { atomic_fetch_sub_explicit(&range_pending,1,memory_order_relaxed); atomic_store_explicit(&r->state,0,memory_order_release); continue; }
        }
        atomic_store_explicit(&r->state,2,memory_order_release);
    }
    uint64_t elapsed=monotonic_ns()-start;
    state->stats.drain_ns+=elapsed;
    if (elapsed>state->stats.drain_max_ns) state->stats.drain_max_ns=elapsed;
    pthread_mutex_unlock(&consume_mutex);
    pthread_mutex_unlock(&drain_mutex);
}
void nvnc_get_stats(nvnc_stats *out) {
    memset(out,0,sizeof *out);
    pthread_mutex_lock(&consume_mutex);
    if (state) {
        *out=state->stats;
        out->queued=atomic_load_explicit(&write_pos,memory_order_acquire);
        out->range_pending=atomic_load_explicit(&range_pending,memory_order_acquire);
        out->backlog=out->queued-state->read+out->range_pending;
        out->range_raw=atomic_load(&range_raw); out->range_coalesced=atomic_load(&range_coalesced);
        out->range_dropped=atomic_load(&range_dropped); out->queue_dropped=atomic_load(&queue_dropped);
        out->cpu_events=atomic_load(&raw_cpu); out->fault_events=atomic_load(&raw_fault);
        out->flush_events=atomic_load(&raw_flush); out->writebacks=atomic_load(&raw_writeback);
        for (unsigned i=0;i<NVNC_REGION_CAP;i++) {
            const nvnc_region_snapshot *r=&state->regions[i].value;
            if (!r->backing) continue;
            int cpu=r->cpu_gen!=0, gpu=r->device_gen||r->writeback_gen;
            out->observed_mixed_regions+=cpu&&gpu;
            out->observed_cpu_regions+=cpu&&!gpu;
            out->observed_gpu_regions+=!cpu&&gpu;
        }
    }
    out->dropped=atomic_load_explicit(&lost,memory_order_acquire);
    out->unknown_epoch=atomic_load_explicit(&unknown_epoch,memory_order_acquire);
    out->unknown_reasons=atomic_load_explicit(&unknown_reasons,memory_order_acquire);
    pthread_mutex_unlock(&consume_mutex);
}
int nvnc_probe(uint64_t address, nvnc_region_snapshot *out) {
    memset(out,0,sizeof *out); out->unknown=NVNC_UNKNOWN_NCE;
    if (!nvnc_active()) return 0;
    pthread_mutex_lock(&drain_mutex);
    if (!nvnc_active()) { pthread_mutex_unlock(&drain_mutex); return 0; }
    uint64_t epoch=resolver.epoch(); nvnc_backing b={0};
    int ok=epoch!=UINT64_MAX && resolver.resolve(address,1,&b) && b.id && b.length==1 && resolver.epoch()==epoch;
    pthread_mutex_lock(&consume_mutex);
    region *r=ok ? region_get(b.id,b.offset,0) : NULL;
    if (r) { *out=r->value; if (out->map_epoch!=epoch) out->unknown|=NVNC_UNKNOWN_MAPPING; }
    if (r && out->identity_epoch!=atomic_load_explicit(&life_epoch,memory_order_acquire)) out->unknown|=NVNC_UNKNOWN_IDENTITY;
    out->unknown|=(uint32_t)atomic_load_explicit(&unknown_reasons,memory_order_acquire);
    if (!ok) out->unknown|=NVNC_UNKNOWN_MAPPING;
    if (state->read!=atomic_load_explicit(&write_pos,memory_order_acquire) || atomic_load_explicit(&range_pending,memory_order_acquire)) out->unknown|=NVNC_UNKNOWN_ORDER;
    pthread_mutex_unlock(&consume_mutex);
    pthread_mutex_unlock(&drain_mutex);
    return r!=NULL;
}
