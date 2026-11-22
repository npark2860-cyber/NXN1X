// Link the actual production dirty-tracker object to the actual F2 ledger.
// OS protection is a stub; test the real hooks/generations without a GPU/game.
#include "../nvn_memory_content.h"
#include "../../gfx/nvn/nvn_dirty.h"
#include <stdatomic.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define BASE UINT64_C(0x100000000)
static int (*fault_hook)(void*,int,void*);
static void (*prep_hook)(void*,size_t);
static unsigned checks, protects, notifications;
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); return 1; } } while(0)
void dm_fault_hook_register(int (*f)(void*,int,void*)) { fault_hook=f; }
void dm_guest_write_prep_register(void (*f)(void*,size_t)) { prep_hook=f; }
void dm_log_line(const char *format,...) { (void)format; notifications++; }
int nvnh_getpagesize(void) { return 4096; }
int nvnh_mprotect(void *p,size_t n,int prot) { (void)p;(void)n;(void)prot; protects++; return 0; }
uint64_t mach_absolute_time(void) { struct timespec t; timespec_get(&t,TIME_UTC); return (uint64_t)t.tv_sec*1000000000+(uint64_t)t.tv_nsec; }
int mach_timebase_info(void *p) { uint32_t *v=p; v[0]=v[1]=1; return 0; }
static uint64_t epoch(void) { return 1; }
static int resolve(uint64_t a,uint64_t n,nvnc_backing *out) {
    if (a<BASE || a-BASE>=65536) return 0;
    *out=(nvnc_backing){1,a-BASE,n}; return 1;
}
void nvn_dirty_stamp_unprotected(int);
static void drain(void) {
    for (unsigned i=0;i<10000;i++) {
        nvnc_drain(1024,1024); nvnc_stats s; nvnc_get_stats(&s);
        if (!s.backlog) return;
    }
    abort();
}
int main(void) {
    nvnc_resolver r={NVNC_ABI,sizeof r,resolve,epoch}; CHECK(nvnc_set_resolver(&r)); nvnc_start(1);
    nvnc_pool_add(1,BASE,65536); drain();
    nvn_dirty_add_range((void *)(uintptr_t)BASE,65536); CHECK(fault_hook && prep_hook);
    nvn_dirty_stamp_unprotected(1);
    uint64_t before=nvn_dirty_now();
    prep_hook((void *)(uintptr_t)BASE,16); drain();
    CHECK(nvn_dirty_since((void *)(uintptr_t)BASE,16,before));
    nvnc_region_snapshot first; CHECK(nvnc_probe(BASE,&first) && first.cpu_gen);
    nvn_dirty_protect((void *)(uintptr_t)BASE,4096);
    unsigned protected_calls=protects;
    CHECK(nvn_dirty_gpu_write((void *)(uintptr_t)BASE,4096)==0); drain();
    CHECK(protects==protected_calls); // GPU stamp must not unprotect the page
    nvnc_region_snapshot gpu; CHECK(nvnc_probe(BASE,&gpu));
    CHECK(gpu.cpu_gen==first.cpu_gen && gpu.writeback_gen>first.all_gen);
    CHECK(fault_hook((void *)(uintptr_t)BASE,1,NULL)); drain();
    nvnc_region_snapshot cpu; CHECK(nvnc_probe(BASE,&cpu));
    CHECK(cpu.cpu_gen>gpu.cpu_gen && (cpu.unknown&NVNC_UNKNOWN_PRESTORE));
    unsigned old=nvnc_origin_push(NVNC_GPU_WRITEBACK);
    prep_hook((void *)(uintptr_t)BASE,16); nvnc_origin_pop(old); drain();
    CHECK(nvnc_probe(BASE,&gpu) && gpu.cpu_gen==cpu.cpu_gen && gpu.writeback_gen>cpu.all_gen);
    nvnc_stats stats; nvnc_get_stats(&stats); uint64_t cpu_events=stats.cpu_events;
    nvn_dirty_protect((void *)(uintptr_t)BASE,4096);
    nvn_dirty_remove_range((void *)(uintptr_t)BASE); drain(); nvnc_get_stats(&stats);
    CHECK(stats.cpu_events==cpu_events); // teardown's unprotect is not a host write
    CHECK(!nvn_dirty_tracked((void *)(uintptr_t)BASE,1));
    CHECK(!notifications); CHECK(stats.fault_events==1 && stats.writebacks==2);
    nvnc_stop(); printf("PASS production dirty hooks %u checks\n",checks); return 0;
}
