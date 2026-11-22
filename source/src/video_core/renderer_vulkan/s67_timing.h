// SPDX-License-Identifier: GPL-3.0-or-later
// s67 (apply_s67.py, diagnostic only): where an Eden graphics pipeline's build time goes.
// PipelineCache::CreateGraphicsPipeline (the calling thread) fills g_front: t0 its start, a = environments + TranslateProgram / MergeDualVertexPrograms
// (+ the varying pack), b = runtime info + EmitSPIRV, c = vkCreateShaderModule (BuildShader), e its end. The GraphicsPipeline constructor takes it
// (Take) and the worker lambda logs, when the total is >= DM_VK_S67_MS (default 50 ms, 0 = every pipeline), one "s67: pipeline" line with
// layout (descriptor / pipeline layouts, templates: the constructor before the queue), queue (waiting for a pipeline worker), build (render pass, Validate,
// MakePipeline), driver (vkCreateGraphicsPipelines alone, g_driver_ns set around it in MakePipeline) and total (t0 to the end of the build).
// DM_VK_S67_KEYS=1 (default 1): every other pipeline gets a short "s67: built" line with its shader keys, once (a pipeline object is built once).
// Both env values are read once, at the first pipeline. Times by std::chrono::steady_clock; thread_local only, no locks.
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"

namespace Vulkan::S67 {

struct Front {
    int64_t t0 = 0, a = 0, b = 0, c = 0, e = 0;
    bool set = false;
};
inline thread_local Front g_front;
inline thread_local int64_t g_driver_ns = -1;

inline int64_t Now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
inline Front Take() {
    const Front f = g_front;
    g_front = {};
    return f;
}
inline double Ms(int64_t ns) {
    return static_cast<double>(ns) / 1e6;
}
inline double Threshold() {
    static const double t = [] {
        const char* e = std::getenv("DM_VK_S67_MS");
        return e && *e ? std::atof(e) : 50.0;
    }();
    return t;
}
inline bool Keys() {
    static const bool k = [] {
        const char* e = std::getenv("DM_VK_S67_KEYS");
        return !(e && *e && std::atoi(e) == 0);
    }();
    return k;
}
inline const char* TopologyName(u32 t) {
    static constexpr const char* n[15] = {"POINTS", "LINES", "LINE_LOOP", "LINE_STRIP", "TRIANGLES",
                                          "TRIANGLE_STRIP", "TRIANGLE_FAN", "QUADS", "QUAD_STRIP",
                                          "POLYGON", "LINES_ADJ", "LINE_STRIP_ADJ", "TRIANGLES_ADJ",
                                          "TRIANGLE_STRIP_ADJ", "PATCHES"};
    return t < 15 ? n[t] : "?";
}
// unique_hashes: 0 vertex A, 1 vertex B, 2 tess control, 3 tess eval, 4 geometry, 5 fragment
template <typename Hashes>
inline std::string KeyText(const Hashes& h, u32 topology) {
    std::string s = fmt::format("VS {:016x} FS {:016x}", h[1], h[5]);
    static constexpr const char* other[5] = {"VA", "", "TC", "TE", "GS"};
    for (int i : {0, 2, 3, 4}) {
        if (h[i] != 0) {
            s += fmt::format(" {} {:016x}", other[i], h[i]);
        }
    }
    s += fmt::format(" prim {}", TopologyName(topology));
    return s;
}
// on the thread that built the pipeline, right after its build (ok) or its failure
template <typename Hashes>
inline void Report(u64 hash, const Hashes& h, u32 topology, const Front& f, int64_t enq, int64_t run,
                   bool ok) {
    const int64_t end = Now();
    const int64_t start = f.set ? f.t0 : enq;
    const double total = Ms(end - start);
    const double thr = Threshold();
    if (total >= thr || thr <= 0.0 || !ok) {
        const std::string drv = g_driver_ns >= 0 ? fmt::format("{:.1f}", Ms(g_driver_ns)) : "-";
        LOG_INFO(Render_Vulkan,
                 "s67: pipeline {:016x} {}{} front a={:.1f} spirv b={:.1f} modules c={:.1f} "
                 "layout={:.1f} queue q={:.1f} build={:.1f} driver d={} total e={:.1f} ms",
                 hash, KeyText(h, topology), ok ? "" : " [FAILED]", Ms(f.a), Ms(f.b), Ms(f.c),
                 f.set ? Ms(enq - f.e) : 0.0, Ms(run - enq), Ms(end - run), drv, total);
    } else if (Keys()) {
        LOG_INFO(Render_Vulkan, "s67: built {:016x} {} in {:.1f} ms", hash, KeyText(h, topology),
                 total);
    }
}

} // namespace Vulkan::S67
