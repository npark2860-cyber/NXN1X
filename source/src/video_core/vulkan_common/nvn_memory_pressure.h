// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "video_core/vulkan_common/nvn_memory_ledger.h"
namespace Vulkan::NvnMemory {
struct SystemPressureSample {
    uint64_t total{}, available{}, commit_available{}, time_ns{};
    bool valid{}, low{};
};
// PR 138's hold/shrink policy, fed by Windows physical pressure rather than
// Linux PSI. RAM/commit are constraints on a unified heap, never added to usage.
class SystemPressure {
    std::array<uint64_t,HeapCount> discount{}, last_shrink{};
    SystemPressureSample sample{};
    unsigned state{};
public:
    static uint64_t Reserve(uint64_t total) {
        return total && total<=12*1024*MiB ? total/8 : std::max<uint64_t>(512*MiB,total/32);
    }
    void Update(SystemPressureSample s) {
        sample=s;
        if (!s.valid) return; // retain the last caution on an unavailable OS sample
        const uint64_t reserve=Reserve(s.total);
        if (s.low || s.available<reserve || s.commit_available<512*MiB) state=2;
        else if (s.available<2*reserve || s.commit_available<1024*MiB) state=1;
        else if (s.available>=Add(2*reserve,128*MiB) && s.commit_available>=1280*MiB) {
            state=0; discount.fill(0); last_shrink.fill(0);
        } else if (state==2) state=1;
    }
    uint64_t Limit(uint32_t heap,uint64_t raw,uint64_t usage,uint64_t image_bytes,uint64_t now) {
        if (heap>=HeapCount || !raw) return raw;
        // Pressure reduces the actual driver budget, not the usage observed
        // before game loading. A bootstrap sample must not freeze the whole
        // heap at a few hundred MiB for the remainder of loading.
        uint64_t& cut=discount[heap];
        const bool fresh=sample.valid && now>=sample.time_ns && now-sample.time_ns<=1000000000;
        if (state==2 && fresh && (!last_shrink[heap] || now-last_shrink[heap]>=1000000000)) {
            // One bounded step per second. Do not make 250 ms budget sampling
            // itself force repeated whole-cache shrink requests.
            const uint64_t step=std::min<uint64_t>(128*MiB,std::max(image_bytes/8,
                std::max(Sub(Reserve(sample.total),sample.available),Sub(512*MiB,sample.commit_available))));
            cut=std::min(raw,Add(cut,step)); last_shrink[heap]=now;
        }
        // Imported/other live allocations are not disposable image cache.
        // Rebase this floor each sample instead of keeping an obsolete minimum.
        const uint64_t floor=Add(Sub(usage,image_bytes),std::min<uint64_t>(image_bytes,512*MiB));
        uint64_t allowed=std::min(raw,std::max(floor,Sub(raw,cut)));
        if(fresh) {
            // Keep OS headroom in addition to the GPU restoration reserve.
            // Imports still obey the GPU budget; they are never counted as a
            // second independent OS allocation in the ledger.
            const uint64_t room=std::min(Sub(sample.available,Reserve(sample.total)),
                                         Sub(sample.commit_available,512*MiB));
            allowed=std::min(allowed,Add(usage,room));
        }
        return allowed;
    }
    unsigned State() const { return state; }
    const SystemPressureSample& Sample() const { return sample; }
};
} // namespace Vulkan::NvnMemory
