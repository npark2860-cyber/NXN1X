// SPDX-License-Identifier: GPL-2.0-or-later
#include "nvn_memory_transfer.h"
#include <pthread.h>
#include <limits.h>
#include <time.h>
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t wake = PTHREAD_COND_INITIALIZER;
static pthread_cond_t progress_wake = PTHREAD_COND_INITIALIZER;
static uint64_t completed;
static pthread_t thread;
static nvnp_job *head, *tail, *active;
static nvnp_save_fn run;
static uint64_t bytes;
static unsigned count;
static int running, stopping;
int nvnp_changed(const nvnp_job *j) {
    return atomic_load_explicit(&j->cancel,memory_order_acquire) || !j->image_id ||
        j->version==UINT64_MAX || !j->live_version ||
        __atomic_load_n(j->live_version,__ATOMIC_ACQUIRE)!=j->version;
}
void nvnp_cancel(nvnp_job *j) { atomic_store_explicit(&j->cancel,1,memory_order_release); }
int nvnp_done(const nvnp_job *j) { return atomic_load_explicit(&j->done,memory_order_acquire); }
int nvnp_publishable(const nvnp_job *j,uint64_t image_id,int same_image,int orphan) {
    return nvnp_done(j) && !j->result && !nvnp_changed(j) &&
        j->image_id==image_id && same_image && !orphan;
}
static void *worker(void *unused) {
    (void)unused;
    for (;;) {
        pthread_mutex_lock(&mu);
        while (!head && !stopping) pthread_cond_wait(&wake,&mu);
        if (!head && stopping) { pthread_mutex_unlock(&mu); break; }
        nvnp_job *j=head; head=j->queue_next; if (!head) tail=0;
        active=j; pthread_mutex_unlock(&mu);
        j->result=nvnp_changed(j)?1:run(j);
        pthread_mutex_lock(&mu); active=0;
        // All callback access to the source ends before publishing done.
        atomic_store_explicit(&j->done,1,memory_order_release);
        ++completed; pthread_cond_broadcast(&progress_wake);
        pthread_mutex_unlock(&mu);
    }
    return 0;
}
int nvnp_start(nvnp_save_fn save) {
    pthread_mutex_lock(&mu);
    if (running) { pthread_mutex_unlock(&mu); return 1; }
    if (count || !save) { pthread_mutex_unlock(&mu); return 0; }
    stopping=0; run=save;
    running=pthread_create(&thread,0,worker,0)==0;
    pthread_mutex_unlock(&mu); return running;
}
int nvnp_submit(nvnp_job *j) {
    pthread_mutex_lock(&mu);
    if (!running || stopping || j->admitted || nvnp_done(j) || count>=NVNP_JOBS || !j->bytes || j->bytes>NVNP_BYTES-bytes) {
        pthread_mutex_unlock(&mu); return 0;
    }
    j->queue_next=0; j->admitted=1; bytes+=j->bytes; ++count;
    if (tail) tail->queue_next=j; else head=j;
    tail=j; pthread_cond_signal(&wake); pthread_mutex_unlock(&mu); return 1;
}
void nvnp_release(nvnp_job *j) {
    pthread_mutex_lock(&mu);
    if (j->admitted && nvnp_done(j) && j->bytes && count && bytes>=j->bytes) {
        bytes-=j->bytes; --count; j->bytes=0; j->admitted=0;
    }
    pthread_mutex_unlock(&mu);
}
uint64_t nvnp_pending_bytes(void) {
    pthread_mutex_lock(&mu); uint64_t n=bytes; pthread_mutex_unlock(&mu); return n;
}
uint64_t nvnp_progress(void) {
    pthread_mutex_lock(&mu); uint64_t n=completed; pthread_mutex_unlock(&mu); return n;
}
int nvnp_working(void) {
    pthread_mutex_lock(&mu); int busy=head!=0 || active!=0; pthread_mutex_unlock(&mu); return busy;
}
int nvnp_wait_progress(uint64_t observed,uint64_t timeout_ns) {
    if (timeout_ns>1000000000ull) timeout_ns=1000000000ull;
    struct timespec deadline; clock_gettime(CLOCK_REALTIME,&deadline);
    uint64_t ns=(uint64_t)deadline.tv_nsec+timeout_ns;
    deadline.tv_sec+=(time_t)(ns/1000000000ull); deadline.tv_nsec=(long)(ns%1000000000ull);
    pthread_mutex_lock(&mu);
    while (completed==observed && running && (head || active))
        if (pthread_cond_timedwait(&progress_wake,&mu,&deadline)) break;
    int advanced=completed!=observed; pthread_mutex_unlock(&mu); return advanced;
}
void nvnp_stop(void) {
    pthread_mutex_lock(&mu);
    if (!running) { pthread_mutex_unlock(&mu); return; }
    stopping=1;
    if (active) nvnp_cancel(active);
    for (nvnp_job *j=head;j;j=j->queue_next) nvnp_cancel(j);
    pthread_cond_signal(&wake); pthread_mutex_unlock(&mu);
    pthread_join(thread,0);
    pthread_mutex_lock(&mu); running=0; pthread_mutex_unlock(&mu);
}
