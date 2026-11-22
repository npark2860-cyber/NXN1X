#include "video_core/vulkan_common/nvn_memory_admission.h"
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
using namespace Vulkan::NvnMemory;
static unsigned checks;
#define CHECK(x) do {++checks;if(!(x)){std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);std::exit(1);}}while(0)
int main() {
 Admission<8> a; constexpr uint64_t now=1000000000;
 CHECK(a.Reserve(0,8*MiB,now,false,false).status==AdmissionStatus::Unknown);
 CHECK(a.BlockSize(0,32*MiB,2*MiB,4096,now)==2*MiB);
 a.Sample(0,{128*MiB,256*MiB,0,1,now,true});
 CHECK(a.BlockSize(0,32*MiB,2*MiB,4096,now)==32*MiB);
 auto one=a.Reserve(0,32*MiB,now,true,false); CHECK(one.id);
 auto two=a.Reserve(0,64*MiB,now,true,false); CHECK(!two.id && two.status==AdmissionStatus::NeedReclaim);
 CHECK(a.Settle(one.id,false)); CHECK(!a.Settle(one.id,true)); CHECK(a.Snapshot().reserved==0 && a.Snapshot().growth==0);
 // Actual 128 MiB VMA block is charged, not its 1 MiB logical suballocation.
 auto block=a.Reserve(0,128*MiB,now,true,false); CHECK(!block.id);
 // Stale samples never admit optional work, even in cooperative mode.
 CHECK(!a.Reserve(0,MiB,now+1000000001,false,true).id);
 auto required=a.Reserve(0,16*MiB,now+1000000001,true,true); CHECK(required.id);
 CHECK(a.Settle(required.id,true) && a.Snapshot().growth==16*MiB);
 CHECK(a.TakePressure()>0 && !a.TakePressure());
 a.Restore(0,128*MiB); CHECK(a.Snapshot().reserve>=128*MiB+4*MiB);
 auto baseline=a.Growth(0); a.Sample(0,{64*MiB,512*MiB,baseline,2,now+2000000000,true});
 one=a.Reserve(0,16*MiB,now+2000000000,true,false); CHECK(one.id);
 // Budget shrinks while a ticket is in flight. Never cancel an existing copy
 // or credit a pending retirement; the next ticket observes the smaller budget.
 a.Sample(0,{64*MiB,80*MiB,baseline,3,now+2000000001,true});
 CHECK(!a.Reserve(0,MiB,now+2000000001,true,false).id); CHECK(a.Settle(one.id,false));
 CHECK(a.BlockSize(0,32*MiB,2*MiB,6144,now+2000000001)>=2*MiB);
 CHECK(a.BlockSize(0,32*MiB,2*MiB,6144,now+2000000001)%6144==0);
 Admission<1024> concurrent; concurrent.Sample(0,{0,UINT64_C(1)<<42,0,1,now,true});
 std::atomic<unsigned> failed=0; std::vector<std::thread> threads;
 for(int k=0;k<8;k++) threads.emplace_back([&]{for(int n=0;n<2000;n++) {auto t=concurrent.Reserve(0,4096,now,true,false); if(!t.id||!concurrent.Settle(t.id,n&1)) failed++;}});
 for(auto& t:threads)t.join(); auto s=concurrent.Snapshot();
 CHECK(!failed && s.issued==16000 && s.settled==16000 && !s.reserved && !s.invalid);
 CHECK(s.growth==8000*4096);
 std::printf("PASS admission %u checks + 16000 concurrent tickets; GPU NOT_RUN\n",checks);
}
