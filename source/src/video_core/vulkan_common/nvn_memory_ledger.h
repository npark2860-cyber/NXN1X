// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include "nvnh_memory.h"

namespace Vulkan::NvnMemory {
static_assert(std::atomic<uint64_t>::is_always_lock_free);
constexpr uint64_t MiB = 1024 * 1024;
constexpr size_t HeapCount = 16;
inline uint64_t Add(uint64_t a, uint64_t b) {
    return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}
inline uint64_t Sub(uint64_t a, uint64_t b) { return a > b ? a - b : 0; }

// Bounded MPSC queue, with bounded producer CAS retries. The sole consumer never
// exposes a slot until its producer published it. Drops are permanent Unknown.
template <size_t Capacity>
class Events {
    static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0);
    struct Cell { std::atomic<uint64_t> sequence{}; nvnh_memory_event event{}; };
    std::array<Cell, Capacity> cells{};
    alignas(64) std::atomic<uint64_t> write{};
    alignas(64) uint64_t read{};
public:
    Events() { for (size_t i = 0; i < Capacity; ++i) cells[i].sequence.store(i); }
    bool Push(const nvnh_memory_event& e) noexcept {
        auto pos = write.load(std::memory_order_relaxed);
        for (unsigned attempt = 0; attempt < 8; ++attempt) {
            auto& cell = cells[pos & (Capacity - 1)];
            const auto seq = cell.sequence.load(std::memory_order_acquire);
            if (seq == pos) {
                if (write.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    cell.event = e;
                    cell.sequence.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (seq < pos) {
                return false;
            } else {
                pos = write.load(std::memory_order_relaxed);
            }
        }
        return false;
    }
    bool Pop(nvnh_memory_event& e) noexcept {
        auto& cell = cells[read & (Capacity - 1)];
        if (cell.sequence.load(std::memory_order_acquire) != read + 1) return false;
        e = cell.event;
        cell.sequence.store(read + Capacity, std::memory_order_release);
        ++read;
        return true;
    }
};

struct Heap {
    uint64_t live{}, pending_retirement{}, allocated{}, freed{}, blocks{};
    std::array<uint64_t, NVNH_OWNER_COUNT> owner{};
    std::array<uint64_t, NVNH_MEM_CLASS_COUNT> kind{};
    uint64_t restore_pending{}, restore_peak{}, restore_requests{}, restore_failures{};
    uint64_t logical_vma{}, logical_time_ns{}; // never added to actual blocks
};
struct Budget {
    uint64_t usage{}, budget{}, growth_base{}, serial{}, time_ns{};
    bool available{};
    uint64_t Estimate(uint64_t positive_growth) const {
        // No credit for frees against an old sample. Concurrent growth may be
        // counted twice for one epoch; that is deliberately conservative.
        return Add(usage, Sub(positive_growth, growth_base));
    }
    uint64_t Headroom(uint64_t positive_growth) const { return Sub(budget, Estimate(positive_growth)); }
    bool Fresh(uint64_t now_ns) const { return available && now_ns >= time_ns && now_ns - time_ns <= 1'000'000'000; }
};

template <size_t Capacity = 32768, size_t RestoreCapacity = 128>
class Ledger {
    struct Entry {
        uint64_t memory{}, id{}, freeing{}, bytes{};
        uint32_t owner{}, kind{}, heap{};
        uint8_t state{}; // 0 never used, 1 live, 2 tombstone
        bool retired{};
    };
    struct Restore { uint64_t id{}, bytes{}; uint32_t heap{}; };
    std::array<Entry, Capacity> entries{};
    std::array<Restore, RestoreCapacity> restores{};
    static size_t Hash(uint64_t m, uint32_t o) {
        return static_cast<size_t>(((m >> 4) ^ (m >> 32) ^ (uint64_t{o} * 0x9e3779b9)) % Capacity);
    }
    Entry* Find(uint64_t m, uint32_t o, uint64_t freeing = 0) {
        for (size_t i = 0, pos = Hash(m, o); i < Capacity; ++i, pos = (pos + 1) % Capacity) {
            auto& e = entries[pos];
            if (!e.state) return nullptr;
            if (e.state == 1 && e.memory == m && e.owner == o && e.freeing == freeing) return &e;
        }
        return nullptr;
    }
    void Retire(Entry& e) {
        if (!e.retired) { heaps[e.heap].pending_retirement += e.bytes; e.retired = true; }
    }
public:
    std::array<Heap, HeapCount> heaps{};
    uint64_t unknown{}, failures{}, binds{}, local_rejects{}, native_attached{};
    void Apply(const nvnh_memory_event& v, uint32_t heap) {
        if (v.owner == 0 || v.owner >= NVNH_OWNER_COUNT) { ++unknown; return; }
        switch (v.op) {
        case NVNH_MEM_ALLOCATE: {
            if (v.result != 0) { ++failures; return; }
            if (heap >= HeapCount || !v.memory || !v.id || v.kind >= NVNH_MEM_CLASS_COUNT || Find(v.memory, v.owner)) { ++unknown; return; }
            for (size_t i = 0, pos = Hash(v.memory, v.owner); i < Capacity; ++i, pos = (pos + 1) % Capacity) {
                auto& e = entries[pos];
                if (e.state == 1) continue;
                e = {v.memory, v.id, 0, v.bytes, v.owner, v.kind, heap, 1, false};
                auto& h = heaps[heap];
                h.live += v.bytes; h.allocated += v.bytes; ++h.blocks;
                h.owner[v.owner] += v.bytes; h.kind[v.kind] += v.bytes;
                return;
            }
            ++unknown; return;
        }
        case NVNH_MEM_RETIRE:
        case NVNH_MEM_FREE_BEGIN: {
            auto* e = Find(v.memory, v.owner);
            if (!e) { ++unknown; return; }
            Retire(*e);
            if (v.op == NVNH_MEM_FREE_BEGIN) e->freeing = v.id;
            return;
        }
        case NVNH_MEM_FREE_END: {
            // Handle reuse between the real free and this event must not free
            // a new allocation with the same Vulkan handle.
            auto* e = v.related ? Find(v.memory, v.owner, v.related) : nullptr;
            if (!e) { ++unknown; return; }
            auto& h = heaps[e->heap];
            h.live -= e->bytes; h.freed += e->bytes; --h.blocks;
            h.owner[e->owner] -= e->bytes; h.kind[e->kind] -= e->bytes;
            if (e->retired) h.pending_retirement -= e->bytes;
            e->state = 2;
            return;
        }
        case NVNH_MEM_BIND:
            if (v.result != 0) ++failures;
            else ++binds; // suballocation/alias binding never creates heap bytes
            return;
        case NVNH_MEM_RESULT: if (v.result != 0) ++failures; return;
        case NVNH_MEM_LOCAL_REJECT: ++local_rejects; return;
        case NVNH_MEM_RESTORE_BEGIN:
            if (heap >= HeapCount) { ++unknown; return; }
            for (auto& r : restores) {
                if (r.id) continue;
                r = {v.id, v.bytes, heap};
                auto& h = heaps[heap];
                h.restore_pending += v.bytes;
                h.restore_peak = std::max(h.restore_peak, h.restore_pending);
                ++h.restore_requests;
                return;
            }
            ++unknown; return;
        case NVNH_MEM_RESTORE_END:
            for (auto& r : restores) {
                if (r.id != v.related || !r.id) continue;
                auto& h = heaps[r.heap];
                h.restore_pending -= r.bytes;
                if (!v.result) ++h.restore_failures;
                r = {};
                return;
            }
            ++unknown; return;
        case NVNH_MEM_VMA_LOGICAL:
            if (heap >= HeapCount) { ++unknown; return; }
            heaps[heap].logical_vma = v.bytes; heaps[heap].logical_time_ns = v.time_ns; return;
        case NVNH_MEM_ATTACH: native_attached = 1; return;
        case NVNH_MEM_DETACH: native_attached = 0; return;
        default: ++unknown; return;
        }
    }
    uint64_t RestoreReserve(uint32_t heap) const {
        // Forecast only: no extra VkDeviceMemory is preallocated. Existing 4 MiB
        // transfer workspace is counted as a block, and this margin remains
        // conservative even when that workspace already exists.
        return std::max(uint64_t{64} * MiB, Add(heaps[heap].restore_peak, 4 * MiB));
    }
};
} // namespace Vulkan::NvnMemory
