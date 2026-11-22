// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <type_traits>
#include <ankerl/unordered_dense.h>
#include <vector>

#include "common/common_types.h"
#include "common/thread_worker.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/shader_cache.h"

namespace Core {
class System;
}

namespace Shader::IR {
struct Program;
}

namespace VideoCore {
class ShaderNotify;
}

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

struct ComputePipelineCacheKey {
    u64 unique_hash;
    u32 shared_memory_size;
    std::array<u32, 3> workgroup_size;

    size_t Hash() const noexcept;

    bool operator==(const ComputePipelineCacheKey& rhs) const noexcept;

    bool operator!=(const ComputePipelineCacheKey& rhs) const noexcept {
        return !operator==(rhs);
    }
};
static_assert(std::has_unique_object_representations_v<ComputePipelineCacheKey>);
static_assert(std::is_trivially_copyable_v<ComputePipelineCacheKey>);
static_assert(std::is_trivially_constructible_v<ComputePipelineCacheKey>);

} // namespace Vulkan

namespace std {

template <>
struct hash<Vulkan::ComputePipelineCacheKey> {
    size_t operator()(const Vulkan::ComputePipelineCacheKey& k) const noexcept {
        return k.Hash();
    }
};

} // namespace std

namespace Vulkan {

class ComputePipeline;
class DescriptorPool;
class Device;
class PipelineStatistics;
class RenderPassCache;
class Scheduler;

using VideoCommon::ShaderInfo;

struct ShaderPools {
    void ReleaseContents() {
        flow_block.ReleaseContents();
        block.ReleaseContents();
        inst.ReleaseContents();
    }

    Shader::ObjectPool<Shader::IR::Inst> inst{8192};
    Shader::ObjectPool<Shader::IR::Block> block{32};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_block{32};
};

// bak86: a store a compute shader made on its dispatch's first invocation only, moved to the host (vk_pipeline_cache.cpp FirstStorePass): before each dispatch the
// host writes value at the 64-bit address in constant buffer cbuf_index at cbuf_offset, plus byte_offset
struct NvnFirstStore {
    u32 cbuf_index;
    u32 cbuf_offset;
    u32 byte_offset;
    u32 value;
};
[[nodiscard]] std::vector<NvnFirstStore> NvnFirstStoresOf(const ComputePipeline* pipeline);

class PipelineCache : public VideoCommon::ShaderCache {
public:
    explicit PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, const Device& device,
                           Scheduler& scheduler, DescriptorPool& descriptor_pool,
                           GuestDescriptorQueue& guest_descriptor_queue,
                           DescriptorBufferRing& descriptor_buffer_ring,
                           RenderPassCache& render_pass_cache, BufferCache& buffer_cache,
                           TextureCache& texture_cache, VideoCore::ShaderNotify& shader_notify_);
    void TickNvnDriverCache();
    ~PipelineCache();

    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipeline();

    [[nodiscard]] ComputePipeline* CurrentComputePipeline();

    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback);

    // NVN path (video_core/nvn, NXN1X docs/lane_b/SHADER_EDEN_B.md): the Eden pipeline for
    // NVN shader environments, made by the same translation from the same key as on the Maxwell
    // path and recorded in the same vulkan.bin. envs: the present stages in program order (as
    // CreateGraphicsPipeline consumes them), each already sized (GenericEnvironment::Analyze).
    // Called from the NVN replay thread; a game never runs the NVN and Maxwell paths together.
    [[nodiscard]] GraphicsPipeline* NvnGraphicsPipeline(
        const GraphicsPipelineCacheKey& key, std::span<VideoCommon::GenericEnvironment* const> envs);
    // bak85: the translation settings the NVN path's pipelines are made with (nvn_shader_bridge.cpp nvn_eden_dump_ir)
    [[nodiscard]] const Shader::HostTranslateInfo& NvnHostInfo() const noexcept { return host_info; }
    [[nodiscard]] ComputePipeline* NvnComputePipeline(const ComputePipelineCacheKey& key,
                                                      VideoCommon::GenericEnvironment& env);
    [[nodiscard]] const Shader::Profile& NvnProfile() const noexcept {
        return profile;
    }
    [[nodiscard]] const DynamicFeatures& NvnDynamicFeatures() const noexcept {
        return dynamic_features;
    }

private:
    void ReportCacheUsage(const char* phase) const;

    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipelineSlowPath();

    [[nodiscard]] GraphicsPipeline* BuiltPipeline(GraphicsPipeline* pipeline) const noexcept;

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline();

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline(
        ShaderPools& pools, const GraphicsPipelineCacheKey& key,
        std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
        bool build_in_parallel);

    std::unique_ptr<ComputePipeline> CreateComputePipeline(const ComputePipelineCacheKey& key,
                                                           const ShaderInfo* shader);

    std::unique_ptr<ComputePipeline> CreateComputePipeline(ShaderPools& pools,
                                                           const ComputePipelineCacheKey& key,
                                                           Shader::Environment& env,
                                                           PipelineStatistics* statistics,
                                                           bool build_in_parallel);

    void QueueNvnRecipe(std::span<const char> key, std::span<const VideoCommon::GenericEnvironment* const> envs);
    bool SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                      const vk::PipelineCache& pipeline_cache, u32 cache_version);

    vk::PipelineCache LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                              u32 expected_cache_version);

    const Device& device;
    Scheduler& scheduler;
    DescriptorPool& descriptor_pool;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;
    RenderPassCache& render_pass_cache;
    BufferCache& buffer_cache;
    TextureCache& texture_cache;
    VideoCore::ShaderNotify& shader_notify;
    bool use_asynchronous_shaders{};
    bool use_vulkan_pipeline_cache{};

    GraphicsPipelineCacheKey graphics_key{};
    GraphicsPipeline* current_pipeline{};

    ankerl::unordered_dense::map<ComputePipelineCacheKey, std::unique_ptr<ComputePipeline>> compute_cache;
    ankerl::unordered_dense::map<GraphicsPipelineCacheKey, std::unique_ptr<GraphicsPipeline>> graphics_cache;

    ShaderPools main_pools;
    ShaderPools nvn_pools;          // NVN path: its translations, under nvn_mutex
    std::mutex nvn_mutex;

    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;

    std::filesystem::path pipeline_cache_filename;

    std::filesystem::path vulkan_pipeline_cache_filename;
    vk::PipelineCache vulkan_pipeline_cache;

    // Count only cache loading and new build requests, never the per-draw hit path.
    // Vulkan's opaque driver cache does not expose entry or recompilation counts here.
    std::atomic<u64> disk_graphics_recipes{};
    std::atomic<u64> disk_compute_recipes{};
    std::atomic<u64> disk_shader_environments{};
    std::atomic<u64> runtime_graphics_requests{};
    std::atomic<u64> runtime_compute_requests{};
    std::atomic<u64> nvn_graphics_requests{};
    std::atomic<u64> nvn_compute_requests{};
    std::atomic<u64> nvn_unserializable_recipes{};
    u64 observed_nvn_builds{};
    std::atomic<u64> checkpoint_nvn_builds{};
    std::chrono::steady_clock::time_point last_nvn_build{}, last_checkpoint{};
    std::atomic_bool checkpoint_pending{};
    u64 driver_cache_bytes_supplied{};

    Common::ThreadWorker workers;
    Common::ThreadWorker serialization_thread;
    DynamicFeatures dynamic_features;
};

} // namespace Vulkan
