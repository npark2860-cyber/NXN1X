// SPDX-License-Identifier: GPL-3.0-or-later
#include "video_core/vulkan_common/nvn_memory_coordinator.h"
#include "video_core/vulkan_common/nvn_memory_ledger.h"
#include "video_core/vulkan_common/nvn_memory_admission.h"
#include "video_core/vulkan_common/vma.h"
#include "common/logging.h"
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

namespace Vulkan {
namespace {
using namespace NvnMemory;
uint64_t Now() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
struct Dispatch {
    std::atomic<VkDevice> device{};
    NvnMemoryCoordinator* owner{};
};
// Vulkan function signatures have no user data. Registration is lifecycle-only;
// a VMA call reads no other device's object. Unregister only AFTER vmaDestroyAllocator.
std::array<Dispatch, 8> dispatches;
std::mutex registration;
NvnMemoryCoordinator* Lookup(VkDevice device) {
    for (auto& d : dispatches) if (d.device.load(std::memory_order_acquire) == device) return d.owner;
    std::terminate(); // adapter lifetime invariant, never a fabricated Vulkan OOM
}
VKAPI_ATTR VkResult VKAPI_CALL VmaAllocate(VkDevice d, const VkMemoryAllocateInfo* i,
                                         const VkAllocationCallbacks* a, VkDeviceMemory* m) {
    return Lookup(d)->Allocate(i, a, m);
}
VKAPI_ATTR void VKAPI_CALL VmaFree(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks* a) {
    Lookup(d)->Free(m, a);
}
const char* ResultName(int32_t r) {
    switch (r) {
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
    case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
    default: return "VkResult";
    }
}
bool Due(uint64_t n) { return n <= 4 || !(n & (n - 1)); }
} // namespace

struct NvnMemoryCoordinator::Impl {
    VkPhysicalDevice physical;
    VkDevice device;
    PFN_vkGetPhysicalDeviceMemoryProperties2 query;
    PFN_vkAllocateMemory allocate;
    PFN_vkFreeMemory free_memory;
    bool budget_extension;
    VkPhysicalDeviceMemoryProperties properties{};
    Events<8192> events;
    Ledger<> ledger;
    Admission<> admission;
    std::atomic<bool> cooperative{};
    std::array<std::atomic<uint64_t>,NVNH_OWNER_COUNT> owner_need{}, owner_freed{}, owner_polls{};
    bool policy_allowed{true};
    std::array<Budget, HeapCount> budgets{};
    std::array<std::atomic<uint64_t>, HeapCount> growth{};
    std::array<uint64_t, HeapCount> restore_short{}, restore_stale{};
    struct Failure { const char* site{}; const char* op{}; int32_t result{}; uint64_t count{}; };
    std::array<Failure, 64> failure_kinds{};
    uint64_t unclassified_failures{}, samples{}, last_report{}, start{Now()};
    std::atomic<uint64_t> next_id{1}, drops{}, pressure{};
    std::atomic<bool> stopped{};
    std::mutex sleep_mutex;
    std::condition_variable wake;
    std::thread worker;
    Dispatch* dispatch{};
    nvnh_memory_api api{};

    Impl(VkPhysicalDevice p, VkDevice d, PFN_vkGetPhysicalDeviceMemoryProperties2 q,
         PFN_vkAllocateMemory a, PFN_vkFreeMemory f, bool b)
        : physical(p), device(d), query(q), allocate(a), free_memory(f), budget_extension(b) {
        VkPhysicalDeviceMemoryProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        query(physical, &props);
        properties = props.memoryProperties;
        api = {NVNH_MEMORY_VERSION, sizeof(nvnh_memory_api), NVNH_MEMORY_OBSERVE | NVNH_MEMORY_GROWTH, 0, this,
            [](void* ctx, const nvnh_memory_event* event) -> uint64_t {
                return static_cast<Impl*>(ctx)->Record(*event);
            },
            [](void* ctx,uint32_t type,uint64_t bytes)->uint64_t { return static_cast<Impl*>(ctx)->Reserve(type,bytes); },
            [](void* ctx,uint64_t ticket,int32_t result) { static_cast<Impl*>(ctx)->admission.Settle(ticket,result==VK_SUCCESS); },
            [](void* ctx,uint32_t op,uint32_t owner,uint64_t value)->uint64_t { return static_cast<Impl*>(ctx)->Policy(op,owner,value); },
            [](void* ctx,uint32_t type,uint64_t desired,uint64_t minimum,uint64_t quantum)->uint64_t {
                auto* owner=static_cast<Impl*>(ctx);
                if(!owner->cooperative.load() || type>=owner->properties.memoryTypeCount) return desired;
                return owner->admission.BlockSize(owner->properties.memoryTypes[type].heapIndex,desired,minimum,quantum,Now());
            },
            [](void* ctx,uint32_t type,uint64_t bytes)->uint64_t {
                auto* owner=static_cast<Impl*>(ctx);
                if (!owner->cooperative.load(std::memory_order_acquire) ||
                    type>=owner->properties.memoryTypeCount) return 0;
                return owner->admission.Reserve(owner->properties.memoryTypes[type].heapIndex,
                                                bytes,Now(),false,false).id;
            }};
        const char* f4=std::getenv("DM_VK_FOUNDATION_F4"); policy_allowed=!f4 || std::strcmp(f4,"0")!=0;
    }
    uint64_t Reserve(uint32_t type,uint64_t bytes) {
        if (!cooperative.load(std::memory_order_acquire) || type>=properties.memoryTypeCount) return 0;
        return admission.Reserve(properties.memoryTypes[type].heapIndex,bytes,Now(),true,true).id;
    }
    uint64_t Policy(uint32_t op,uint32_t owner,uint64_t value) {
        if (owner==0 || owner>=NVNH_OWNER_COUNT) return 0;
        if (op==NVNH_POLICY_ENABLE) {
            if (owner==NVNH_OWNER_NATIVE && value && policy_allowed) cooperative.store(true,std::memory_order_release);
            return cooperative.load(std::memory_order_acquire);
        }
        if (!cooperative.load(std::memory_order_acquire)) return 0;
        if (op==NVNH_POLICY_TAKE_PRESSURE) { owner_polls[owner]++; return owner_need[owner].exchange(0); }
        if (op==NVNH_POLICY_RECLAIMED) owner_freed[owner].fetch_add(value);
        return 0;
    }
    void PublishPressure() {
        const auto n=admission.TakePressure(); if(!n || !cooperative.load()) return;
        for(unsigned i=1;i<NVNH_OWNER_COUNT;i++) {
            auto old=owner_need[i].load(); while(old<n && !owner_need[i].compare_exchange_weak(old,n)) {}
        }
    }
    uint32_t HeapFor(const nvnh_memory_event& e) const {
        if (e.op == NVNH_MEM_VMA_LOGICAL) return e.memory_type; // heap, not memory type
        return e.memory_type < properties.memoryTypeCount ? properties.memoryTypes[e.memory_type].heapIndex : UINT32_MAX;
    }
    uint64_t Record(nvnh_memory_event e) noexcept {
        if (e.op==NVNH_MEM_DETACH) {
            // A borrowed device can outlive one native game. Do not let a
            // later emulation/Switch 1 session inherit cooperative activation.
            cooperative.store(false,std::memory_order_release);
            for(auto& need:owner_need) need.store(0,std::memory_order_release);
        }
        e.id = next_id.fetch_add(1, std::memory_order_relaxed);
        e.time_ns = Now();
        const auto heap = HeapFor(e);
        if (e.op == NVNH_MEM_ALLOCATE && !e.result && heap < HeapCount) {
            growth[heap].fetch_add(e.bytes,std::memory_order_relaxed);
        }
        if ((e.op == NVNH_MEM_ALLOCATE || e.op == NVNH_MEM_RESULT || e.op == NVNH_MEM_BIND) && e.result)
            pressure.fetch_add(1, std::memory_order_relaxed);
        if (!e.id || !events.Push(e)) {
            drops.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
        return e.id;
    }
    void FailureLog(const nvnh_memory_event& e) {
        uint64_t n = 0;
        for (auto& f : failure_kinds) {
            if (!f.count || (f.site == e.site && f.op == e.operation && f.result == e.result)) {
                f.site = e.site; f.op = e.operation; f.result = e.result; n = ++f.count; break;
            }
        }
        if (!n) n = ++unclassified_failures;
        if (!Due(n)) return;
        const auto heap = HeapFor(e);
        const Budget b = heap < HeapCount ? budgets[heap] : Budget{};
        LOG_WARNING(Render_Vulkan,
            "NVN-F1 raw-result op={} site={}:{} VkResult={}({}) owner={} class={} "
            "bytes={} type={} heap={} memory={:#x} related={} event={} event_ms={} count={} "
            "sample={} sample_age_ms={} usage={} budget={} policy=observe",
            e.operation, e.site, e.line, e.result, ResultName(e.result), e.owner, e.kind,
            e.bytes, e.memory_type, heap, e.memory, e.related, e.id, (e.time_ns - start) / 1000000,
            n, b.serial, b.serial ? (Now() - b.time_ns) / 1000000 : UINT64_MAX, b.usage, b.budget);
    }
    void Process(const nvnh_memory_event& e) {
        const auto heap = HeapFor(e);
        ledger.Apply(e, heap);
        if (e.op==NVNH_MEM_ALLOCATE && !e.result && !e.growth_ticket && heap<HeapCount) admission.Unreserved(heap,e.bytes);
        if (e.op==NVNH_MEM_RESTORE_BEGIN && heap<HeapCount) admission.Restore(heap,e.bytes);
        if ((e.op == NVNH_MEM_ALLOCATE || e.op == NVNH_MEM_RESULT || e.op == NVNH_MEM_BIND) && e.result)
            FailureLog(e);
        if (e.op == NVNH_MEM_LOCAL_REJECT && Due(ledger.local_rejects))
            LOG_WARNING(Render_Vulkan, "NVN-F1 local-reject site={} reason={} bytes={} count={} Vulkan_call=no",
                        e.site, e.operation, e.bytes, ledger.local_rejects);
        if (e.op == NVNH_MEM_RESTORE_BEGIN && heap < properties.memoryHeapCount) {
            const auto& b = budgets[heap];
            const bool fresh = b.Fresh(e.time_ns);
            const auto positive = growth[heap].load(std::memory_order_relaxed);
            const auto required = Add(ledger.heaps[heap].restore_pending, ledger.RestoreReserve(heap));
            const auto shortfall = fresh ? Sub(required, b.Headroom(positive)) : 0;
            if (!fresh) ++restore_stale[heap];
            if (shortfall) ++restore_short[heap];
            if (Due(ledger.heaps[heap].restore_requests) || (shortfall && Due(restore_short[heap])))
                LOG_INFO(Render_Vulkan,
                    "NVN-F1 restore-request id={} heap={} payload={} growth={} pending={} reserve={} "
                    "headroom={} shortfall={} sample={} fresh={} event_ms={} policy=observe admitted=unchanged",
                    e.id, heap, e.object, e.bytes, ledger.heaps[heap].restore_pending, ledger.RestoreReserve(heap),
                    b.Headroom(positive), shortfall, b.serial, fresh, (e.time_ns - start) / 1000000);
        }
    }
    void Sample() {
        // Never call VMA from this worker: the allocator is externally synchronized.
        // Snapshot growth BEFORE the driver query. Racing allocations remain a
        // conservative positive delta; frees do not manufacture stale headroom.
        std::array<uint64_t, HeapCount> before{};
        std::array<uint64_t,HeapCount> admission_before{};
        for (uint32_t i=0;i<HeapCount;++i) { before[i]=growth[i].load(std::memory_order_relaxed); admission_before[i]=admission.Growth(i); }
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        props.pNext = budget_extension ? &budget : nullptr;
        const auto time = Now();
        query(physical, &props);
        ++samples;
        for (uint32_t i=0;i<properties.memoryHeapCount;++i) {
            budgets[i]={budget.heapUsage[i],budget.heapBudget[i],before[i],samples,time,budget_extension && budget.heapBudget[i]!=0};
            auto b=budgets[i]; b.growth_base=admission_before[i]; admission.Sample(i,b);
        }
        PublishPressure();
    }
    void Report(bool final) {
        const auto now = Now();
        for (uint32_t i = 0; i < properties.memoryHeapCount; ++i) {
            const auto& h = ledger.heaps[i]; const auto& b = budgets[i];
            const auto positive = growth[i].load(std::memory_order_relaxed);
            LOG_INFO(Render_Vulkan,
                "NVN-F1 heap={} sample={} age_ms={} available={} usage={} budget={} estimate={} "
                "known={} VMA={} native={} blocks={} retired_pending={} final={}",
                i, b.serial, (now - b.time_ns) / 1000000, b.available, b.usage, b.budget, b.Estimate(positive),
                h.live, h.owner[NVNH_OWNER_VMA], h.owner[NVNH_OWNER_NATIVE], h.blocks, h.pending_retirement, final);
            LOG_INFO(Render_Vulkan,
                "NVN-F1 classes heap={} sample={} imports={} images={} sparse={} staging={} pools={} "
                "present={} scratch={} other={} final={}",
                i, b.serial, h.kind[NVNH_MEM_IMPORT],
                h.kind[NVNH_MEM_IMAGE], h.kind[NVNH_MEM_SPARSE], h.kind[NVNH_MEM_STAGING], h.kind[NVNH_MEM_POOL],
                h.kind[NVNH_MEM_PRESENT], h.kind[NVNH_MEM_SCRATCH], h.kind[NVNH_MEM_OTHER], final);
            // Keep each record below the logger's line limit. The old combined
            // line lost restore_short/stale/failed once allocations grew large.
            LOG_INFO(Render_Vulkan,
                "NVN-F1 allocation heap={} sample={} allocated_total={} freed_total={} growth_since_sample={} "
                "driver_residual={} accounted_excess={} vma_logical={} logical_age_ms={} final={}",
                i, b.serial, h.allocated, h.freed, Sub(positive, b.growth_base), Sub(b.usage, h.live), Sub(h.live, b.usage),
                h.logical_vma, h.logical_time_ns ? (now - h.logical_time_ns) / 1000000 : UINT64_MAX, final);
            LOG_INFO(Render_Vulkan,
                "NVN-F1 restore heap={} sample={} restore_pending={} restore_peak={} reserve={} "
                "headroom={} restore_short={} restore_stale={} restore_failed={} final={}",
                i, b.serial,
                h.restore_pending, h.restore_peak, ledger.RestoreReserve(i), b.Headroom(positive),
                restore_short[i], restore_stale[i], h.restore_failures, final);
        }
        LOG_INFO(Render_Vulkan,
            "NVN-F1 accounting failures={} local_rejects={} bindings={} dropped={} unknown={} "
            "native_attached={} metadata_bytes={} unclassified_errors={} state={} policy=observe enforcement=disabled "
            "imports_are_not_additional_OS_RAM=true final={}",
            ledger.failures, ledger.local_rejects, ledger.binds, drops.load(), ledger.unknown,
            ledger.native_attached, sizeof(Impl), unclassified_failures,
            (drops.load() || ledger.unknown || !ledger.native_attached) ? "Unknown" : "observed-explicit-blocks", final);
        if (cooperative.load()) {
            const auto a=admission.Snapshot();
            LOG_INFO(Render_Vulkan,"NVN-F4 tickets issued={} settled={} pending={} required_debt={} strict_rejected={} invalid={} reserve={} mode=cooperative enforcement=disabled",
                     a.issued,a.settled,a.reserved,a.debt,a.rejected,a.invalid,a.reserve);
            LOG_INFO(Render_Vulkan,"NVN-F4 owners native_polls={} VMA_polls={} native_actual_freed={} pending_native={} pending_VMA={} emergency_retry=new-path-disabled",
                     owner_polls[NVNH_OWNER_NATIVE].load(),owner_polls[NVNH_OWNER_VMA].load(),owner_freed[NVNH_OWNER_NATIVE].load(),owner_need[NVNH_OWNER_NATIVE].load(),owner_need[NVNH_OWNER_VMA].load());
        }
        last_report = now;
    }
    void Run() {
        Sample();
        uint64_t sampled_at = Now(), seen_pressure = 0;
        while (!stopped.load(std::memory_order_acquire)) {
            nvnh_memory_event e;
            for (unsigned i = 0; i < 4096 && events.Pop(e); ++i) Process(e);
            auto now = Now();
            const auto failures = pressure.load(std::memory_order_relaxed);
            if (now - sampled_at >= 250000000 || (failures != seen_pressure && now - sampled_at >= 100000000)) {
                Sample(); sampled_at = Now(); seen_pressure = failures;
            }
            if (!last_report || now - last_report >= 5000000000) Report(false);
            std::unique_lock lock(sleep_mutex);
            wake.wait_for(lock, std::chrono::milliseconds(25), [this] { return stopped.load(); });
        }
        nvnh_memory_event e;
        while (events.Pop(e)) Process(e);
        Sample(); Report(true);
    }
};

bool NvnMemoryCoordinator::Enabled() {
    const char* value = std::getenv("DM_VK_FOUNDATION_F1");
    return !value || std::strcmp(value, "0") != 0;
}
NvnMemoryCoordinator::NvnMemoryCoordinator(std::unique_ptr<Impl> p) : impl(std::move(p)) {}
std::shared_ptr<NvnMemoryCoordinator> NvnMemoryCoordinator::Create(
    VkPhysicalDevice physical, VkDevice device, PFN_vkGetPhysicalDeviceMemoryProperties2 query,
    PFN_vkAllocateMemory allocate, PFN_vkFreeMemory free_memory, bool budget_extension) {
    if (!Enabled() || !query || !allocate || !free_memory) return {};
    try {
        auto result = std::shared_ptr<NvnMemoryCoordinator>(new NvnMemoryCoordinator(
            std::make_unique<Impl>(physical, device, query, allocate, free_memory, budget_extension)));
        {
            std::lock_guard lock(registration);
            for (auto& d : dispatches) if (!d.device.load()) {
                d.owner = result.get(); d.device.store(device, std::memory_order_release);
                result->impl->dispatch = &d; break;
            }
        }
        if (!result->impl->dispatch) {
            LOG_WARNING(Render_Vulkan, "NVN-F1 unavailable: device registry full; baseline allocator retained");
            return {};
        }
        result->impl->worker = std::thread([p = result->impl.get()] {
            try { p->Run(); }
            catch (...) {
                // Diagnostic/logging allocation failure must not terminate the
                // game. No fresh heartbeat/sample means observation is Unknown.
                p->drops.fetch_add(1, std::memory_order_relaxed);
                p->stopped.store(true, std::memory_order_release);
            }
        });
        LOG_INFO(Render_Vulkan, "NVN-F1 enabled mode=observe interval_ms=250 pressure_min_ms=100 queue=8192 ledger=32768 reserve_floor=64MiB archive_margin=4MiB; no admission/reclaim changes");
        return result;
    } catch (const std::exception& e) {
        LOG_WARNING(Render_Vulkan, "NVN-F1 unavailable: {}; baseline allocator retained", e.what());
        return {};
    }
}
NvnMemoryCoordinator::~NvnMemoryCoordinator() {
    // Also covers RendererVulkan constructor unwind after a successful handover,
    // where its destructor (and normal NvnBridge::Detach) would not run.
    nvnh_memory_detach_context(impl.get());
    impl->stopped.store(true, std::memory_order_release);
    impl->wake.notify_one();
    if (impl->worker.joinable()) impl->worker.join();
    if (impl->dispatch) {
        std::lock_guard lock(registration);
        impl->dispatch->device.store(VK_NULL_HANDLE, std::memory_order_release);
        impl->dispatch->owner = nullptr;
    }
}
const nvnh_memory_api* NvnMemoryCoordinator::Api() const { return &impl->api; }
void NvnMemoryCoordinator::ConfigureVma(VmaVulkanFunctions& functions) {
    // VMA's pfnFree notification runs BEFORE vkFreeMemory. Observe the actual
    // Vulkan boundary instead, including raw failures and completed frees.
    functions.vkAllocateMemory = VmaAllocate;
    functions.vkFreeMemory = VmaFree;
}
void NvnMemoryCoordinator::LogicalVma(uint32_t heap, uint64_t bytes) {
    nvnh_memory_event e{};
    e.op = NVNH_MEM_VMA_LOGICAL; e.owner = NVNH_OWNER_VMA; e.memory_type = heap; e.bytes = bytes;
    impl->Record(e);
}
VkResult NvnMemoryCoordinator::Allocate(const VkMemoryAllocateInfo* i, const VkAllocationCallbacks* a, VkDeviceMemory* m) {
    const auto ticket=impl->Reserve(i->memoryTypeIndex,i->allocationSize);
    const auto r = impl->allocate(impl->device, i, a, m);
    if(ticket) impl->admission.Settle(ticket,r==VK_SUCCESS);
    nvnh_memory_event e{};
    e.op = NVNH_MEM_ALLOCATE; e.owner = NVNH_OWNER_VMA; e.kind = NVNH_MEM_VMA_BLOCK;
    e.result = r; e.memory_type = i->memoryTypeIndex; e.bytes = i->allocationSize; e.growth_ticket=ticket;
    e.memory = r == VK_SUCCESS ? reinterpret_cast<uint64_t>(*m) : 0;
    e.site = "VMA-block"; e.operation = "vkAllocateMemory";
    impl->Record(e);
    return r;
}
uint64_t NvnMemoryCoordinator::OwnerPressure(uint32_t owner) {
    return impl->Policy(NVNH_POLICY_TAKE_PRESSURE,owner,0);
}
void NvnMemoryCoordinator::Free(VkDeviceMemory m, const VkAllocationCallbacks* a) {
    nvnh_memory_event e{};
    e.op = NVNH_MEM_FREE_BEGIN; e.owner = NVNH_OWNER_VMA; e.memory = reinterpret_cast<uint64_t>(m);
    e.site = "VMA-block"; e.operation = "vkFreeMemory";
    const auto id = m ? impl->Record(e) : 0;
    impl->free_memory(impl->device, m, a);
    if (id) { e.op = NVNH_MEM_FREE_END; e.related = id; impl->Record(e); }
}
} // namespace Vulkan
