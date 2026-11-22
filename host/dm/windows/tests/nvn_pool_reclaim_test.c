#include "../nvn_pool_reclaim.h"
#include <stdio.h>
#include <stdlib.h>
static unsigned checks;
#define CHECK(x) do {checks++;if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);}} while(0)
#define M (UINT64_C(1)<<20)
int main(void) {
    nvnp_plan p;
    // Releasing one small buffer in a range with a hot holder frees no memory.
    nvnp_init(&p,3); nvnp_set(&p,0,32*M,2,1); nvnp_set(&p,1,32*M,1,1); nvnp_set(&p,2,32*M,0,1);
    nvnp_bind(&p,1,0,10,1); nvnp_bind(&p,1,0,100,0); nvnp_bind(&p,2,0,20,1);
    CHECK(nvnp_select(&p,1)==32*M);
    CHECK(!nvnp_selected(&p,1) && !nvnp_selected(&p,2) && nvnp_selected(&p,3));
    // A bridge into a hot range pins both, including the apparently cold side.
    nvnp_init(&p,3); for(unsigned i=0;i<3;i++)nvnp_set(&p,i,32*M,i==1?2:1,1);
    nvnp_bind(&p,1,2,10,1); nvnp_bind(&p,2,0,100,0); nvnp_bind(&p,3,0,25,1);
    CHECK(nvnp_select(&p,64*M)==32*M);
    CHECK(!nvnp_selected(&p,1) && !nvnp_selected(&p,2) && nvnp_selected(&p,3));
    // Transitive bridge closure: never retire only one allocation of a cohort.
    nvnp_init(&p,4); for(unsigned i=0;i<4;i++)nvnp_set(&p,i,32*M,i==1?2:1,1);
    nvnp_bind(&p,1,2,20,1); nvnp_bind(&p,2,3,30,1); nvnp_bind(&p,4,0,40,1);
    CHECK(nvnp_select(&p,1)==96*M);
    CHECK(nvnp_selected(&p,1) && nvnp_selected(&p,2) && nvnp_selected(&p,3) && !nvnp_selected(&p,4));
    // Refcounts cover all owners. An unseen holder is a pin, not a free byte.
    nvnp_init(&p,2); nvnp_set(&p,0,32*M,2,1); nvnp_set(&p,1,32*M,1,1);
    nvnp_bind(&p,1,2,5,1); CHECK(!nvnp_select(&p,UINT64_MAX));
    CHECK(!nvnp_selected(&p,1) && !nvnp_selected(&p,2));
    // Two references to the same imported allocation count twice, bytes once.
    nvnp_init(&p,1); nvnp_set(&p,0,32*M,2,1); nvnp_bind(&p,1,1,5,1);
    CHECK(nvnp_select(&p,64*M)==32*M);
    nvnp_init(&p,2); nvnp_set(&p,0,32*M,1,1); nvnp_set(&p,1,32*M,0,0);
    nvnp_bind(&p,1,9,5,1); CHECK(!nvnp_select(&p,1));
    // Selection stops at requested allocation bytes; no free-all idle sweep.
    nvnp_init(&p,NVNP_RANGES);
    for(unsigned i=0;i<NVNP_RANGES;i++)nvnp_set(&p,i,32*M,0,1);
    CHECK(nvnp_select(&p,33*M)==64*M);
    unsigned n=0; for(unsigned i=1;i<=NVNP_RANGES;i++)n+=nvnp_selected(&p,i);
    CHECK(n==2);
    nvnp_init(&p,1); nvnp_set(&p,0,32*M,0,1); CHECK(!nvnp_select(&p,0));
    CHECK(!nvnp_selected(&p,0) && !nvnp_selected(&p,2));
    printf("PASS pool cohorts %u checks; GPU NOT_RUN\n",checks);
}
