// CPU-only tests of the production ledger. No Vulkan or guest execution.
// Including the production C unit allows bounded-capacity/wrap adversarial tests
// without exposing reset or corruption APIs in the shipped host.
#include "../nvn_memory_content.c"
#include <stdio.h>
#include <sched.h>

#define BASE UINT64_C(0x100000000)
#define ALIAS UINT64_C(0x200000000)
#define SIZE UINT64_C(0x10000000)
static _Atomic uint64_t map_version=1;
static _Atomic int resolving;
static _Atomic int block_resolver, resolver_entered, release_resolver, stopped;
static unsigned checks;
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); exit(1); } } while(0)
static uint64_t epoch(void) { return atomic_load(&map_version); }
static int resolve(uint64_t a,uint64_t n,nvnc_backing *b) {
    atomic_fetch_add(&resolving,1);
    if (atomic_load(&block_resolver)) {
        nvnc_stats nested; nvnc_get_stats(&nested); // resolver never owns content mutex
        atomic_store(&resolver_entered,1);
        uint64_t start=monotonic_ns();
        while (!atomic_load(&release_resolver)) {
            if (monotonic_ns()-start>UINT64_C(3000000000)) abort();
            sched_yield();
        }
    }
    uint64_t off;
    if (a>=BASE && a-BASE<SIZE) off=a-BASE;
    else if (a>=ALIAS && a-ALIAS<SIZE) off=a-ALIAS;
    else return 0;
    if (n>SIZE-off) n=SIZE-off;
    *b=(nvnc_backing){1,off,n}; return 1;
}
static void drain(void) {
    for (unsigned i=0;i<10000;i++) {
        nvnc_stats s; nvnc_get_stats(&s);
        if (!s.backlog) return;
        nvnc_drain(4096,4096);
    }
    CHECK(!"queue failed to drain");
}
static nvnc_region_snapshot probe(uint64_t a) {
    nvnc_region_snapshot s; CHECK(nvnc_probe(a,&s)); return s;
}
static uint64_t image(uint64_t key,uint64_t a) {
    nvnc_pool_add(key,a,2*BLOCK_SIZE);
    uint64_t im=nvnc_image_add(key,a,BLOCK_SIZE,37); CHECK(im!=0); drain(); return im;
}
static void versions(void) {
    uint64_t im=image(0x1234,BASE);
    int calls=atomic_load(&resolving);
    nvnc_cpu(BASE,4096,NVNC_CPU_FAULT);
    CHECK(atomic_load(&resolving)==calls); // enqueue must not resolve in a fault
    drain(); nvnc_region_snapshot a=probe(BASE);
    CHECK(a.cpu_gen && a.all_gen==a.cpu_gen);
    CHECK((a.unknown&(NVNC_UNKNOWN_PRESTORE|NVNC_UNKNOWN_NCE))==(NVNC_UNKNOWN_PRESTORE|NVNC_UNKNOWN_NCE));
    nvnc_cpu(ALIAS,4096,NVNC_GPU_WRITEBACK); drain();
    nvnc_region_snapshot b=probe(BASE);
    CHECK(b.cpu_gen==a.cpu_gen && b.all_gen>a.all_gen && b.writeback_gen==b.all_gen);
    CHECK(b.backing==a.backing && b.block_offset==a.block_offset);
    uint64_t sub=nvnc_sub_begin(); nvnc_image_intent(im); nvnc_image_intent(im); drain();
    CHECK(probe(BASE).device_gen==0); // recording is not execution
    nvnc_accept(sub,1,1,0); nvnc_sub_end(); drain();
    CHECK(probe(BASE).device_gen==0); // accepted is not completed
    nvnc_gpu_done(sub); drain(); b=probe(BASE);
    CHECK(b.device_gen && b.cpu_gen==a.cpu_gen);
    nvnc_stats s; nvnc_get_stats(&s);
    CHECK(s.gpu_completed==1 && s.cpu_completed==0 && s.intents==1 && s.accepted_intents==1);
    nvnc_cpu_done(sub); nvnc_cpu(BASE+4,1,NVNC_CPU_PRESTORE); drain();
    nvnc_region_snapshot c=probe(ALIAS);
    CHECK(c.cpu_gen>b.cpu_gen && c.all_gen==c.cpu_gen);
    CHECK(c.unknown & NVNC_UNKNOWN_ORDER); // no inferred last writer or safe guest publication
    nvnc_get_stats(&s); CHECK(s.cpu_completed==1 && s.alias_hits>=1);
    unsigned old=nvnc_origin_push(NVNC_GPU_WRITEBACK); CHECK(nvnc_origin()==NVNC_GPU_WRITEBACK);
    nvnc_origin_pop(old); CHECK(nvnc_origin()==NVNC_CPU_PRESTORE);
}
static void partial_failure(void) {
    uint64_t im=image(0x1234,BASE);
    nvnc_cpu(BASE+BLOCK_SIZE,BLOCK_SIZE,NVNC_CPU_FLUSH); drain();
    nvnc_region_snapshot untouched=probe(BASE+BLOCK_SIZE);
    uint64_t sub=nvnc_sub_begin();
    nvnc_image_intent(im); nvnc_accept(sub,1,1,0);
    nvnc_part(2); nvnc_range_intent(BASE+BLOCK_SIZE,64);
    nvnc_error(2,-2); nvnc_accept(sub,2,2,-2); nvnc_sub_end();
    nvnc_gpu_done(sub); drain();
    CHECK(probe(BASE).device_gen>0);
    nvnc_region_snapshot failed=probe(BASE+BLOCK_SIZE);
    CHECK(failed.device_gen==0 && failed.all_gen==untouched.all_gen && failed.cpu_gen==untouched.cpu_gen);
    CHECK(failed.unknown&NVNC_UNKNOWN_FAILED);
    nvnc_stats s; nvnc_get_stats(&s);
    CHECK(s.accepted_intents==1 && s.rejected_intents==1 && s.first_error==-2 && s.first_error_part==2);
    nvnc_cpu_done(sub); drain(); CHECK(sub_get(sub)==NULL);
}
static void uploads(void) {
    nvnc_pool_add(0x1234,BASE,BLOCK_SIZE);
    uint64_t im=nvnc_image_add(0x1234,BASE,BLOCK_SIZE,UINT64_C(0x1234567887654321)); drain();
    CHECK(image_get(im,0)->layout==UINT64_C(0x1234567887654321));
    uint64_t sub=nvnc_sub_begin(); nvnc_image_upload(im); nvnc_image_upload(im);
    nvnc_pool_protected(im,BLOCK_SIZE,0,1); drain();
    nvnc_stats s; nvnc_get_stats(&s);
    CHECK(s.intents==1 && s.upload_intents==1 && s.no_image_write_checks==1);
    nvnc_image_intent(im); nvnc_pool_protected(im,BLOCK_SIZE,0,1); drain();
    nvnc_get_stats(&s); CHECK(s.intents==2 && s.no_image_write_checks==1);
    nvnc_accept(sub,1,1,0); nvnc_sub_end(); nvnc_gpu_done(sub); nvnc_cpu_done(sub); drain();
    nvnc_region_snapshot r=probe(BASE); CHECK(!r.cpu_gen && r.device_gen && r.unknown);
}
static void identity_reuse(void) {
    uint64_t im=image(0x1234,BASE);
    uint64_t sub=nvnc_sub_begin(); nvnc_image_intent(im); nvnc_accept(sub,1,1,0); nvnc_sub_end(); drain();
    nvnc_pool_del(0x1234); nvnc_image_del(im);
    uint64_t replacement=image(0x1234,BASE); CHECK(replacement!=im);
    nvnc_gpu_done(sub); nvnc_cpu_done(sub); drain();
    nvnc_region_snapshot s; CHECK(!nvnc_probe(BASE,&s)); CHECK(s.unknown);
    nvnc_stats stats; nvnc_get_stats(&stats);
    CHECK(stats.device_bytes==0 && stats.unresolved_bytes==BLOCK_SIZE && stats.images==1 && stats.pools==1);
    nvnc_image_intent(im); drain(); // no open submission -> Unknown, never a stale dereference
    nvnc_get_stats(&stats); CHECK(stats.unsupported_events==1);
}
static void aliases_and_remap(void) {
    nvnc_pool_add(10,BASE,BLOCK_SIZE); nvnc_pool_add(11,ALIAS,BLOCK_SIZE); drain();
    nvnc_cpu(BASE,64,NVNC_CPU_FLUSH); drain();
    nvnc_region_snapshot a=probe(BASE), b=probe(ALIAS); CHECK(a.cpu_gen==b.cpu_gen);
    nvnc_pool_del(10); drain(); // freeing one view must not erase shared content records
    CHECK(probe(ALIAS).cpu_gen==a.cpu_gen);
    nvnc_cpu(ALIAS,64,NVNC_CPU_PRESTORE);
    atomic_fetch_add(&map_version,1); drain();
    b=probe(ALIAS); CHECK(b.cpu_gen==a.cpu_gen && (b.unknown&NVNC_UNKNOWN_MAPPING));
    nvnc_cpu(ALIAS+BLOCK_SIZE,1,NVNC_CPU_FLUSH); drain();
    CHECK(probe(ALIAS).cpu_gen==a.cpu_gen); // partial block update cannot clean adjacent region
    CHECK(probe(ALIAS+BLOCK_SIZE).cpu_gen>a.cpu_gen);
}
static void bounded(void) {
    nvnc_cpu(BASE,16*BLOCK_SIZE,NVNC_CPU_PRESTORE);
    for (unsigned i=0;i<10000 && !state->stats.regions;i++) nvnc_drain(1,1);
    nvnc_stats s; nvnc_get_stats(&s);
    CHECK(s.regions==1 && s.cpu_bytes==BLOCK_SIZE && !s.unresolved_bytes);
    CHECK(s.range_pending==1 && s.bounded_yields); drain();
    nvnc_get_stats(&s); CHECK(s.regions==16 && !s.range_pending && !s.unresolved_bytes);
    for (unsigned i=0;i<1000000;i++) nvnc_cpu(BASE,4,NVNC_GPU_WRITEBACK);
    uint64_t sub=nvnc_sub_begin(); nvnc_error(1,-2); nvnc_sub_end();
    nvnc_gpu_done(sub); nvnc_cpu_done(sub);
    nvnc_get_stats(&s); CHECK(!s.dropped && s.range_coalesced==999999 && s.queued==4);
    drain(); nvnc_get_stats(&s);
    CHECK(s.first_error==-2 && s.gpu_completed==1 && s.cpu_completed==1 && !s.backlog);
    // Critical queue overflow is separate and sticky, never promoted to clean.
    for(unsigned i=0;i<NVNC_QUEUE_CAP+17;i++) nvnc_unknown(NVNC_UNKNOWN_RAW);
    nvnc_get_stats(&s); CHECK(s.queue_dropped==17 && !s.range_dropped); drain();
    CHECK(probe(BASE).unknown&NVNC_UNKNOWN_OVERFLOW);
    atomic_store(&next_id,CLOSED);
    CHECK(nvnc_image_add(1,BASE,1,0)==0); drain();
    atomic_store(&next_id,1); CHECK(nvnc_image_add(1,BASE,1,0)==0); drain();
    atomic_store(&life_epoch,CLOSED-1); nvnc_pool_add(1,BASE,1); drain();
    CHECK(atomic_load(&life_epoch)==CLOSED-1);
}
static _Atomic int producers_done;
static void *producer(void *arg) {
    uint64_t index=(uint64_t)(uintptr_t)arg;
    for (unsigned i=0;i<20000;i++) nvnc_cpu(BASE+index*BLOCK_SIZE,16,(i&1)?NVNC_CPU_PRESTORE:NVNC_GPU_WRITEBACK);
    atomic_fetch_add(&producers_done,1); return NULL;
}
static void concurrent(void) {
    pthread_t threads[4];
    for (unsigned i=0;i<4;i++) CHECK(pthread_create(&threads[i],NULL,producer,(void *)(uintptr_t)i)==0);
    while (atomic_load(&producers_done)<4) nvnc_drain(1024,1024);
    for (unsigned i=0;i<4;i++) CHECK(pthread_join(threads[i],NULL)==0);
    drain(); nvnc_stats s; nvnc_get_stats(&s);
    CHECK(s.range_raw==80000 && !s.backlog && !s.queue_dropped);
    CHECK(s.cpu_events+s.writebacks==80000);
    for (unsigned i=0;i<4;i++) { nvnc_region_snapshot r=probe(BASE+(uint64_t)i*BLOCK_SIZE);
        CHECK(r.cpu_gen && r.writeback_gen && r.all_gen>=r.cpu_gen && r.all_gen>=r.writeback_gen && r.unknown); }
    printf("concurrency: events=%llu dropped=%llu drain_max_ms=%.3f\n",(unsigned long long)s.consumed,(unsigned long long)s.dropped,s.drain_max_ns/1e6);
}
static void off(void) {
    CHECK(!nvnc_active() && !state);
    nvnc_cpu(BASE,4096,NVNC_CPU_FAULT); nvnc_pool_add(1,BASE,4096);
    CHECK(!nvnc_image_add(1,BASE,4096,0)); CHECK(!nvnc_sub_begin());
    nvnc_unknown(NVNC_UNKNOWN_RAW); nvnc_drain(100,100);
    nvnc_stats s; nvnc_get_stats(&s); CHECK(!s.metadata_bytes && !s.queued && !s.dropped);
}
static void *drainer(void *p) { (void)p; while(!atomic_load(&resolver_entered)) nvnc_drain(10,10); return NULL; }
static void *stopper(void *p) { (void)p; nvnc_stop(); atomic_store(&stopped,1); return NULL; }
static void lifecycle(void) {
    atomic_store(&block_resolver,1);
    nvnc_cpu(BASE,1,NVNC_CPU_PRESTORE);
    pthread_t reader, closer;
    CHECK(!pthread_create(&reader,NULL,drainer,NULL));
    while (!atomic_load(&resolver_entered)) sched_yield();
    CHECK(!pthread_create(&closer,NULL,stopper,NULL));
    while (nvnc_active()) sched_yield();
    CHECK(!atomic_load(&stopped)); // owner cannot disappear under a mapping reader
    uint64_t before=atomic_load(&write_pos); nvnc_cpu(BASE,1,NVNC_CPU_FAULT);
    CHECK(atomic_load(&write_pos)==before); // no wait on the blocked resolver
    atomic_store(&release_resolver,1);
    CHECK(!pthread_join(reader,NULL)); CHECK(!pthread_join(closer,NULL));
    CHECK(atomic_load(&stopped));
}
int main(int argc,char **argv) {
    CHECK(argc==2);
    nvnc_resolver r={NVNC_ABI,sizeof r,resolve,epoch};
    CHECK(nvnc_set_resolver(&r));
    nvnc_start(strcmp(argv[1],"off")!=0);
    if (!strcmp(argv[1],"off")) off();
    else {
        CHECK(nvnc_active());
        if (!strcmp(argv[1],"versions")) versions();
        else if (!strcmp(argv[1],"uploads")) uploads();
        else if (!strcmp(argv[1],"failure")) partial_failure();
        else if (!strcmp(argv[1],"reuse")) identity_reuse();
        else if (!strcmp(argv[1],"alias")) aliases_and_remap();
        else if (!strcmp(argv[1],"bounded")) bounded();
        else if (!strcmp(argv[1],"concurrent")) concurrent();
        else if (!strcmp(argv[1],"lifecycle")) lifecycle();
        else CHECK(!"unknown test case");
    }
    nvnc_stop(); CHECK(!nvnc_active());
    uint64_t before=atomic_load(&write_pos); nvnc_cpu(BASE,1,NVNC_CPU_FAULT); CHECK(before==atomic_load(&write_pos));
    printf("PASS %s %u checks\n",argv[1],checks);
    return 0;
}
