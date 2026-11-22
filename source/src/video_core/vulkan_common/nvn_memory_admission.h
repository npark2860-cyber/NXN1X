// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include <cstdint>
#include <mutex>
#include "video_core/vulkan_common/nvn_memory_ledger.h"

namespace Vulkan::NvnMemory {
enum class AdmissionStatus : uint32_t { Ready, NeedReclaim, Stale, Unknown, Full };
struct GrowthTicket { uint64_t id{}, bytes{}; uint32_t heap{}; AdmissionStatus status{AdmissionStatus::Unknown}; };
struct AdmissionSnapshot {
    uint64_t reserved{}, growth{}, reserve{}, pressure{}, issued{}, settled{}, debt{}, rejected{}, invalid{};
};
// All methods hold only this private metadata lock. No callbacks, driver calls,
// allocation, owner locks, waits or logs occur while it is held. A ticket covers
// an ACTUAL VkDeviceMemory request (including VMA block slack), never an image.
template<size_t Capacity=1024> class Admission {
    struct Ticket { uint64_t id{}, bytes{}; uint32_t heap{}; };
    struct State { Budget sample{}; uint64_t pending{}, growth{}, restore{}, mandatory{}, requested{}; };
    std::array<State,HeapCount> heaps{};
    std::array<Ticket,Capacity> tickets{};
    mutable std::mutex mutex;
    uint64_t serial{}, issued{}, settled{}, debt{}, rejected{}, invalid{};
    static uint64_t ReserveBytes(const State& h) {
        // At least 64 MiB; otherwise measured restoration growth + largest
        // mandatory request + the existing 4 MiB archive transfer reserve.
        return std::max<uint64_t>(64*MiB,Add(Add(h.restore,h.mandatory),4*MiB));
    }
public:
    uint64_t Growth(uint32_t heap) const {
        std::lock_guard lock(mutex); return heap<HeapCount?heaps[heap].growth:0;
    }
    void Unreserved(uint32_t heap,uint64_t bytes) {
        std::lock_guard lock(mutex); if (heap<HeapCount) heaps[heap].growth=Add(heaps[heap].growth,bytes);
    }
    void Sample(uint32_t heap, Budget b) {
        std::lock_guard lock(mutex);
        if (heap>=HeapCount) return;
        auto& h=heaps[heap]; h.sample=b;
        if (b.available) h.requested=std::max(h.requested,Sub(Add(b.Estimate(h.growth),Add(h.pending,ReserveBytes(h))),b.budget));
    }
    void Restore(uint32_t heap,uint64_t bytes) {
        std::lock_guard lock(mutex); if (heap<HeapCount) heaps[heap].restore=std::max(heaps[heap].restore,bytes);
    }
    uint64_t BlockSize(uint32_t heap,uint64_t desired,uint64_t minimum,uint64_t quantum,uint64_t now) const {
        if (!quantum || !minimum || minimum>desired) return desired;
        const uint64_t lower=minimum/quantum+(minimum%quantum!=0);
        if (lower>desired/quantum) return desired;
        std::lock_guard lock(mutex);
        uint64_t available=0;
        if (heap<HeapCount) {
            const auto& h=heaps[heap];
            if(h.sample.Fresh(now)) available=Sub(h.sample.Headroom(h.growth),Add(h.pending,ReserveBytes(h)));
        }
        // Only speculative spare slots shrink. The required aligned slot/floor
        // remains, so this never substitutes a missing or undersized image.
        return std::max(lower,std::min(desired,available)/quantum)*quantum;
    }
    GrowthTicket Reserve(uint32_t heap,uint64_t bytes,uint64_t now,bool mandatory,bool permit_debt) {
        std::lock_guard lock(mutex);
        GrowthTicket out{0,bytes,heap,AdmissionStatus::Unknown};
        if (heap>=HeapCount || !bytes) return out;
        auto& h=heaps[heap]; const auto& b=h.sample;
        if (mandatory) h.mandatory=std::max(h.mandatory,bytes);
        const uint64_t reserve=ReserveBytes(h);
        const uint64_t target=Add(Add(b.Estimate(h.growth),h.pending),Add(bytes,reserve));
        out.status=!b.available ? AdmissionStatus::Unknown : !b.Fresh(now) ? AdmissionStatus::Stale :
            target>b.budget ? AdmissionStatus::NeedReclaim : AdmissionStatus::Ready;
        if (out.status!=AdmissionStatus::Ready) {
            h.requested=std::max(h.requested,b.available?Sub(target,b.budget):bytes);
            if (!mandatory || !permit_debt) { ++rejected; return out; }
            ++debt; // explicit required-growth debt in cooperative mode, not strict admission
        }
        Ticket* free=nullptr; for(auto& t:tickets) if(!t.id) {free=&t;break;}
        if (!free || serial==UINT64_MAX) { out.status=AdmissionStatus::Full; ++rejected; return out; }
        out.id=++serial; *free={out.id,bytes,heap}; h.pending=Add(h.pending,bytes); ++issued;
        return out;
    }
    bool Settle(uint64_t id,bool allocated) {
        if (!id) return false;
        std::lock_guard lock(mutex);
        for(auto& t:tickets) if(t.id==id) {
            auto& h=heaps[t.heap]; h.pending=Sub(h.pending,t.bytes);
            if(allocated) h.growth=Add(h.growth,t.bytes);
            else h.requested=std::max(h.requested,t.bytes);
            t={}; ++settled; return true;
        }
        ++invalid; return false; // duplicate settlement never credits headroom
    }
    uint64_t TakePressure() {
        std::lock_guard lock(mutex); uint64_t n=0;
        for(auto& h:heaps) { n=std::max(n,h.requested); h.requested=0; }
        return n;
    }
    AdmissionSnapshot Snapshot() const {
        std::lock_guard lock(mutex); AdmissionSnapshot s{};
        for(const auto& h:heaps) {s.reserved=Add(s.reserved,h.pending); s.growth=Add(s.growth,h.growth);
            s.reserve=std::max(s.reserve,ReserveBytes(h)); s.pressure=std::max(s.pressure,h.requested);}
        s.issued=issued; s.settled=settled; s.debt=debt; s.rejected=rejected; s.invalid=invalid; return s;
    }
};
} // namespace Vulkan::NvnMemory
