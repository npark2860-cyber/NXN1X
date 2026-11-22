// SPDX-License-Identifier: GPL-3.0-or-later
// CPU-only tests of the production ledger and notification queue.
#include "video_core/vulkan_common/nvn_memory_ledger.h"
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
using namespace Vulkan::NvnMemory;
static unsigned checks;
#define CHECK(x) do { ++checks; if (!(x)) { std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); std::exit(1); } } while (0)
int main() {
    Ledger<16,2> l;
    nvnh_memory_event e{};
    e.op=NVNH_MEM_ALLOCATE; e.owner=NVNH_OWNER_NATIVE; e.kind=NVNH_MEM_IMAGE;
    e.memory=0x1000; e.bytes=32*MiB; e.id=1;
    l.Apply(e,0); CHECK(l.heaps[0].live==32*MiB && l.heaps[0].blocks==1);
    e.op=NVNH_MEM_BIND;
    for (unsigned i=0;i<64;++i) { e.object=i+1; e.offset=i*4096; l.Apply(e,0); }
    CHECK(l.binds==64 && l.heaps[0].live==32*MiB); // logical aliases/suballocations
    e.op=NVNH_MEM_ALLOCATE; e.result=-2; l.Apply(e,0);
    CHECK(l.failures==1 && l.heaps[0].live==32*MiB);
    e.result=0; e.op=NVNH_MEM_RETIRE; l.Apply(e,0); l.Apply(e,0);
    CHECK(l.heaps[0].pending_retirement==32*MiB && l.heaps[0].live==32*MiB);
    e.op=NVNH_MEM_FREE_BEGIN; e.id=2; l.Apply(e,0);
    CHECK(l.heaps[0].live==32*MiB); // not yet returned from vkFreeMemory
    e.op=NVNH_MEM_ALLOCATE; e.id=3; e.bytes=8*MiB; l.Apply(e,0); // reused raw handle
    CHECK(l.heaps[0].live==40*MiB && l.heaps[0].blocks==2);
    e.op=NVNH_MEM_FREE_END; e.related=2; l.Apply(e,0);
    CHECK(l.heaps[0].live==8*MiB && l.heaps[0].freed==32*MiB && !l.heaps[0].pending_retirement);
    l.Apply(e,0); CHECK(l.unknown==1 && l.heaps[0].live==8*MiB); // duplicate completion
    e.op=NVNH_MEM_FREE_BEGIN; e.id=4; l.Apply(e,0);
    e.op=NVNH_MEM_FREE_END; e.related=4; l.Apply(e,0); CHECK(!l.heaps[0].live);
    e.op=NVNH_MEM_ALLOCATE; e.id=5; e.memory=0x2000; e.bytes=4*MiB; e.kind=NVNH_MEM_IMPORT; l.Apply(e,1);
    e.memory=0x3000; e.id=6; e.object=0xabcdef; l.Apply(e,1);
    CHECK(l.heaps[1].live==8*MiB && l.heaps[1].kind[NVNH_MEM_IMPORT]==8*MiB); // no virtual alias dedup guess
    e.op=NVNH_MEM_RESTORE_BEGIN; e.id=7; e.bytes=96*MiB; l.Apply(e,0);
    e.id=8; e.bytes=32*MiB; l.Apply(e,0);
    CHECK(l.heaps[0].restore_pending==128*MiB && l.RestoreReserve(0)==132*MiB);
    e.id=9; l.Apply(e,0); CHECK(l.unknown==2); // bounded restore metadata
    e.op=NVNH_MEM_RESTORE_END; e.related=7; e.result=0; l.Apply(e,0);
    CHECK(l.heaps[0].restore_pending==32*MiB && l.heaps[0].restore_failures==1);
    e.related=8; e.result=1; l.Apply(e,0); CHECK(!l.heaps[0].restore_pending);
    e.op=NVNH_MEM_VMA_LOGICAL; e.bytes=400*MiB; l.Apply(e,0);
    CHECK(l.heaps[0].live==0 && l.heaps[0].logical_vma==400*MiB);
    Budget b{700*MiB,1024*MiB,100*MiB,1,100,true};
    CHECK(b.Estimate(120*MiB)==720*MiB && b.Headroom(120*MiB)==304*MiB);
    CHECK(b.Estimate(100*MiB)==700*MiB); // frees never reduce old sample
    b.budget=600*MiB; CHECK(!b.Headroom(120*MiB)); // shrinking driver budget
    CHECK(b.Fresh(100) && !b.Fresh(99) && !b.Fresh(1000000101));
    b.available=false; CHECK(!b.Fresh(100));
    CHECK(Add(UINT64_MAX-1,2)==UINT64_MAX && Sub(1,2)==0);
    Ledger<2> small;
    e={}; e.op=NVNH_MEM_ALLOCATE; e.owner=NVNH_OWNER_VMA; e.kind=NVNH_MEM_VMA_BLOCK; e.bytes=1;
    for (unsigned i=1;i<4;++i) { e.memory=i; e.id=i; small.Apply(e,0); }
    CHECK(small.heaps[0].live==2 && small.unknown==1);
    Events<2> bounded; e.id=1; CHECK(bounded.Push(e)); e.id=2; CHECK(bounded.Push(e));
    CHECK(!bounded.Push(e)); CHECK(bounded.Pop(e)&&e.id==1); CHECK(bounded.Pop(e)&&e.id==2); CHECK(!bounded.Pop(e));
    // Concurrent producer loss is observable, no overwritten/duplicated events.
    Events<1024> queue; constexpr size_t producers=4, each=20000;
    std::atomic<size_t> done{}, accepted{}, dropped{};
    std::vector<std::thread> threads;
    std::vector<bool> seen(producers*each);
    for (size_t p=0;p<producers;++p) threads.emplace_back([&,p] {
        nvnh_memory_event v{};
        for (size_t i=0;i<each;++i) { v.id=p*each+i; if(queue.Push(v)) ++accepted; else ++dropped; }
        ++done;
    });
    size_t received=0;
    while (done.load()!=producers || received<accepted.load()) {
        if (queue.Pop(e)) { CHECK(e.id<seen.size()&&!seen[e.id]); seen[e.id]=true; ++received; }
        else std::this_thread::yield();
    }
    for (auto& t:threads) t.join();
    CHECK(received==accepted && accepted+dropped==producers*each);
    std::printf("PASS production ledger/queue: %u checks, accepted=%zu observable_drops=%zu; GPU/game NOT_RUN\n",checks,received,dropped.load());
}
