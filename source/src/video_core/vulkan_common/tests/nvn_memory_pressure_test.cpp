#include "video_core/vulkan_common/nvn_memory_pressure.h"
#include "video_core/vulkan_common/nvn_memory_admission.h"
#include <cstdio>
#include <cstdlib>
using namespace Vulkan::NvnMemory;
static unsigned checks;
#define CHECK(x) do { ++checks;if(!(x)){std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);std::exit(1);} }while(0)
int main() {
 constexpr uint64_t G=1024*MiB,t=1000000000;
 CHECK(SystemPressure::Reserve(8*G)==G);
 CHECK(SystemPressure::Reserve(12*G)==1536*MiB);
 CHECK(SystemPressure::Reserve(16*G)==512*MiB);
 SystemPressure p;
 p.Update({16*G,4*G,10*G,t,true,false});
 CHECK(!p.State() && p.Limit(0,8*G,7*G,3*G,t)==8*G);
 p.Update({16*G,800*MiB,10*G,t,true,false});
 CHECK(p.State()==1 && p.Limit(0,8*G,7*G,3*G,t)==7*G+288*MiB);
 const auto boot=p.Limit(0,8*G,546*MiB,0,t);
 CHECK(boot==834*MiB);
 CHECK(p.Limit(0,8*G,5*G,3*G,t)==5*G+288*MiB); // no frozen bootstrap ceiling
 p.Update({16*G,344*MiB,10*G,t,true,false});
 CHECK(p.State()==2 && p.Limit(0,8*G,7*G,3*G,t)==7*G);
 CHECK(p.Limit(0,8*G,7*G-64*MiB,3*G,t+250000000)==7*G-64*MiB);
 p.Update({16*G,300*MiB,10*G,t+1000000000,true,false});
 CHECK(p.Limit(0,8*G,7*G-128*MiB,3*G,t+1000000000)==7*G-128*MiB);
 p.Update({});
 CHECK(p.Limit(0,8*G,7*G,3*G,t+2000000000)==8*G-256*MiB);
 p.Update({16*G,1200*MiB,10*G,t+3000000000,true,false});
 CHECK(!p.State() && p.Limit(0,8*G,7*G,3*G,t+3000000000)==7*G+688*MiB);
 p.Update({16*G,4*G,200*MiB,t+4000000000,true,false});
 CHECK(p.State()==2 && p.Limit(0,8*G,7*G,256*MiB,t+4000000000)==7*G);
 p.Update({16*G,300*MiB,10*G,t+5000000000,true,true});
 CHECK(p.State()==2);
 const auto constrained=p.Limit(0,8*G,7*G,3*G,t+5000000000);
 Admission<> a; a.Sample(0,{7*G,constrained,0,1,t,true});
 CHECK(a.CurrentNeed(0,t)>0);
 CHECK(!a.Reserve(0,MiB,t,true,false).id); // required growth is denied too
 CHECK(a.BlockSize(0,32*MiB,2*MiB,4096,t)==2*MiB);
 CHECK(p.Limit(0,6*G,7*G,3*G,t+5000000000)<=6*G);
 // OS reserve applies even before a pressure transition. Growth between
 // samples is reserved atomically against the same finite allowance.
 p.Update({16*G,2*G,10*G,t+6000000000,true,false});
 CHECK(!p.State());
 const auto budget=p.Limit(0,8*G,546*MiB,0,t+6000000000);
 CHECK(budget==546*MiB+1536*MiB);
 Admission<8> race; race.Sample(0,{546*MiB,budget,0,1,t,true});
 auto one=race.Reserve(0,512*MiB,t,true,false);
 auto two=race.Reserve(0,512*MiB,t,true,false);
 auto three=race.Reserve(0,512*MiB,t,true,false);
 CHECK(one.id && !two.id && !three.id); // largest mandatory + staging reserve
 CHECK(race.Settle(one.id,false));
 CHECK(!race.Settle(one.id,true));
 CHECK(!race.Snapshot().reserved);
 std::printf("PASS Windows pressure policy %u checks; no Vulkan device or game\n",checks);
}
