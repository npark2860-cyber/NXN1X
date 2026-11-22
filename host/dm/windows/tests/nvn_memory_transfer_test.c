// CPU-only test of the production worker and publication predicates.
#include "../nvn_memory_transfer.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static unsigned checks;
#define CHECK(x) do { ++checks; if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);} } while(0)
static _Atomic int entered, gpu_ready, stop_done, output_result;
static void tick(void) { struct timespec t={0,1000000}; nanosleep(&t,0); }
static void await(_Atomic int *p) { for(int n=0;n<5000 && !atomic_load(p);n++) tick(); CHECK(atomic_load(p)); }
static int save(nvnp_job *j) {
    atomic_store(&entered,1);
    // A cancellation cannot invent completion of an already accepted GPU read.
    while(!atomic_load(&gpu_ready)) tick();
    if(nvnp_changed(j)) return 1;
    return atomic_load(&output_result);
}
static void *stop(void *p) { (void)p; nvnp_stop(); atomic_store(&stop_done,1); return 0; }
static nvnp_job job(uint64_t *v,uint64_t id,uint64_t bytes) {
    nvnp_job j={.live_version=v,.image_id=id,.version=*v,.bytes=bytes}; return j;
}
int main(void) {
    uint64_t version=1;
    nvnp_job a=job(&version,1,4096);
    CHECK(!nvnp_submit(&a)); CHECK(!nvnp_pending_bytes()); // OFF: no worker/callback
    CHECK(nvnp_start(save)); CHECK(nvnp_submit(&a)); CHECK(!nvnp_submit(&a)); await(&entered);
    CHECK(!nvnp_done(&a) && !nvnp_publishable(&a,1,1,0));
    __atomic_store_n(&version,2,__ATOMIC_RELEASE); // mutation between chunks
    CHECK(nvnp_changed(&a)); CHECK(!nvnp_done(&a));
    atomic_store(&gpu_ready,1); await(&a.done);
    CHECK(a.result==1 && !nvnp_publishable(&a,1,1,0));
    CHECK(nvnp_pending_bytes()==4096); nvnp_release(&a); nvnp_release(&a);
    CHECK(!nvnp_pending_bytes());
    nvnp_job b=job(&version,2,8192); CHECK(nvnp_submit(&b)); await(&b.done);
    CHECK(nvnp_publishable(&b,2,1,0));
    CHECK(!nvnp_publishable(&b,3,1,0)); // pool/address reincarnated
    CHECK(!nvnp_publishable(&b,2,0,0)); // upgraded handle
    CHECK(!nvnp_publishable(&b,2,1,1)); // finalized pool
    __atomic_store_n(&version,3,__ATOMIC_RELEASE);
    CHECK(!nvnp_publishable(&b,2,1,0)); // write after file completion, before publication
    nvnp_release(&b);
    atomic_store(&output_result,-13);
    nvnp_job fail=job(&version,4,4096); CHECK(nvnp_submit(&fail)); await(&fail.done);
    CHECK(fail.result==-13 && !nvnp_publishable(&fail,4,1,0)); nvnp_release(&fail);
    atomic_store(&output_result,0); atomic_store(&gpu_ready,0); atomic_store(&entered,0);
    nvnp_job jobs[NVNP_JOBS+1];
    for(int i=0;i<NVNP_JOBS;i++) { jobs[i]=job(&version,(uint64_t)i+10,4096); CHECK(nvnp_submit(&jobs[i])); }
    jobs[NVNP_JOBS]=job(&version,1000,4096); CHECK(!nvnp_submit(&jobs[NVNP_JOBS]));
    await(&entered); CHECK(nvnp_pending_bytes()==NVNP_JOBS*4096);
    nvnp_job unadmitted=job(&version,2000,4096);
    atomic_store(&unadmitted.done,1); nvnp_release(&unadmitted);
    CHECK(nvnp_pending_bytes()==NVNP_JOBS*4096); // cannot credit another job's reservation
    pthread_t stopper; CHECK(!pthread_create(&stopper,0,stop,0));
    for(int i=0;i<20;i++) tick(); CHECK(!atomic_load(&stop_done)); // teardown retains pins until GPU completion
    atomic_store(&gpu_ready,1); CHECK(!pthread_join(stopper,0)); CHECK(atomic_load(&stop_done));
    for(int i=0;i<NVNP_JOBS;i++) { CHECK(nvnp_done(&jobs[i])); CHECK(!nvnp_publishable(&jobs[i],i+10,1,0)); nvnp_release(&jobs[i]); }
    CHECK(!nvnp_pending_bytes()); CHECK(nvnp_start(save));
    atomic_store(&gpu_ready,0);
    nvnp_job big=job(&version,80,NVNP_BYTES), extra=job(&version,81,1);
    CHECK(nvnp_submit(&big)); CHECK(!nvnp_submit(&extra)); atomic_store(&gpu_ready,1);
    await(&big.done); nvnp_release(&big);
    uint64_t exhausted=UINT64_MAX; nvnp_job wrap=job(&exhausted,90,1);
    CHECK(nvnp_submit(&wrap)); await(&wrap.done); CHECK(!nvnp_publishable(&wrap,90,1,0)); nvnp_release(&wrap);
    nvnp_stop(); CHECK(!nvnp_pending_bytes());
    printf("PASS transfer worker %u checks; bounded queue, source pins, mutation, I/O failure, incarnation, teardown; GPU NOT_RUN\n",checks);
}
