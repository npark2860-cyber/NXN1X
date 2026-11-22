// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <bitset>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "common/assert.h"
#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "shader_recompiler/exception.h"
#include "core/core.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#ifdef ARCHITECTURE_x86_64
#include "video_core/macro_turing_jit.h"
#endif
#include "video_core/rasterizer_interface.h"
#include "video_core/textures/decoders.h"

namespace Tegra::Engines {

KeplerCompute::KeplerCompute(MemoryManager& memory_manager_)
    : memory_manager{memory_manager_}
    , upload_state{memory_manager, regs.upload}
{
    // EDEN_NX2_QMD_TRACE: log the QMD scheduler's loads, dependency counts and launches.
    if (std::getenv("EDEN_NX2_QMD_TRACE")) {
        native_qmd.trace = [this](std::string_view event, u64 address, s32 value) {
            LOG_INFO(HW_GPU, "QMD trace {} {:#x} {} (engine {} as {})", event, address, value,
                     fmt::ptr(this), memory_manager.GetID());
        };
    }
    native_qmd.skipped_release = [](u64 address, std::size_t slot, std::span<const u32, 4> words) {
        static std::unordered_set<u32> logged;
        if (!logged.insert(words[1]).second) return;
        LOG_WARNING(HW_GPU, "Native QMD {:#x} skipped release slot {} with unsupported format, words={:08x}",
                    address, slot, fmt::join(words, ","));
    };
    execution_mask.reset();
    execution_mask[KEPLER_COMPUTE_REG_INDEX(exec_upload)] = true;
    execution_mask[KEPLER_COMPUTE_REG_INDEX(data_upload)] = true;
    execution_mask[KEPLER_COMPUTE_REG_INDEX(launch)] = true;
    // Everything ProcessNativeMethod acts on besides storing the value; the QMD and 0x5a
    // trace range too while native draw diagnostics trace it.
    native_plain_methods.set();
    for (const u32 method : {0x45U, 0x46U, 0x47U, 0x48U, 0xaeU, 0xafU, 0xb0U, 0x107U, 0x6c3U,
                             static_cast<u32>(KEPLER_COMPUTE_REG_INDEX(exec_upload)),
                             static_cast<u32>(KEPLER_COMPUTE_REG_INDEX(data_upload))}) {
        native_plain_methods[method] = false;
    }
    if (std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr) {
        native_plain_methods[0x5a] = false;
        for (u32 method = 0xc6; method <= 0x108; ++method) {
            native_plain_methods[method] = false;
        }
    }
}

KeplerCompute::~KeplerCompute() {
    Nx2ListGrid::ReleaseSlots(this);
}

void KeplerCompute::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
    upload_state.BindRasterizer(rasterizer);
}

void KeplerCompute::ConsumeSinkImpl(Core::System& system) {
    for (auto [method, value] : method_sink) {
        if (method < Regs::NUM_REGS) {
            regs.reg_array[method] = value;
        }
    }
    method_sink.clear();
}

// SET_REPORT_SEMAPHORE_A-D (0x6c0-0x6c3): address high/low, payload, then operation in bits 1:0
// (release / acquire / report only / trap), reduction enable bit 3 with the op in bits 11:9, and
// a one-word (no timestamp) structure in bit 28. NVN2 releases queue syncs from compute this way.
void KeplerCompute::ProcessNativeReportSemaphore(u32 argument) {
    const GPUVAddr address = (GPUVAddr{native_registers[0x6c0] & 0x1ffffff} << 32) |
                             native_registers[0x6c1];
    const u32 payload = native_registers[0x6c2];
    const u32 operation = argument & 3;
    const bool one_word = (argument >> 28) & 1;
    static u32 logged{};
    if (logged++ < 32) {
        LOG_INFO(HW_GPU, "Native compute report semaphore {:#x} = {:#x}, control {:#x}", address,
                 payload, argument);
    }
    if (operation != 0 && operation != 2) {
        LOG_WARNING(HW_GPU, "Native compute report semaphore operation {} not supported", operation);
        return;
    }
    if ((argument >> 3) & 1) {
        const u32 reduction = (argument >> 9) & 7;
        rasterizer->SignalFence([this, address, payload, reduction] {
            memory_manager.Write<u32>(address, ReduceSemaphore(reduction, false,
                                                               memory_manager.Read<u32>(address), payload));
        });
        return;
    }
    rasterizer->Query(address, VideoCommon::QueryType::Payload,
                      one_word ? VideoCommon::QueryPropertiesFlags::IsAFence
                               : VideoCommon::QueryPropertiesFlags::HasTimeout,
                      payload, 0);
}

bool KeplerCompute::ProcessNativeMethod(Core::System& system, u32 method, u32 argument, bool is_last_call) {
    if (native_context && !front_macros) {
        if (method == 0x45 || method == 0x47) native_compute_uploaded = true;
        native_macro = native_compute_uploaded ? &native_context->compute_engine : &native_context->engine;
        native_macro_positions = native_compute_uploaded ? std::span<u32>{native_context->compute_positions} : std::span<u32>{native_context->positions};
    }
    if (method >= native_registers.size()) {
        LOG_ERROR(HW_GPU, "Native compute method out of range: {:#x}", method);
        return true;
    }
    if (method >= 0xe00) {
        if (!native_executing_macro) {
            if (method & 1) { LOG_ERROR(HW_GPU, "Native compute macro starts on argument method"); return true; }
            native_executing_macro = method;
        }
        native_parameters.push_back(argument);
        if (!is_last_call) return true;
        const u32 entry = ((*native_executing_macro - 0xe00) / 2) % native_macro_positions.size();
        native_executing_macro_logged = native_executing_macro;
        native_executing_macro.reset();
        static const u32 watch = [] {
            const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
            return value ? static_cast<u32>(std::strtoull(value, nullptr, 16)) : 0U;
        }();
        const bool watched = watch != 0 && std::ranges::any_of(native_parameters,
            [](u32 parameter) { return (parameter & ~0xfU) == watch; });
        if (watched) {
            LOG_INFO(HW_GPU, "Semaphore watch: compute macro start={:#x} parameters={:08x}",
                     native_macro_positions[entry], fmt::join(native_parameters, ","));
        }
        // EDEN_NX2_MME_DUMP: write the compute macro code once per engine, and log each macro
        // call with its entry and parameters (diagnostics).
        if (static const bool mme_dump = std::getenv("EDEN_NX2_MME_DUMP") != nullptr; mme_dump) {
            if (!native_mme_dumped && native_macro_positions[entry] == 0xda) {
                native_mme_dumped = true;
                std::ofstream dump(fmt::format("native-compute-mme-{}.bin", fmt::ptr(this)),
                                   std::ios::binary);
                dump.write(reinterpret_cast<const char*>(native_macro->native_code.data()),
                           sizeof(native_macro->native_code));
            }
            LOG_INFO(HW_GPU, "MME call engine {} method {:#x} start={:#x} parameters={:08x}",
                     fmt::ptr(this), *native_executing_macro_logged, native_macro_positions[entry],
                     fmt::join(native_parameters, ","));
        }
        if (native_data_ram) native_macro->native_interpreter.active_dram = native_data_ram;
        const u32 dram0_before = (*native_macro->native_interpreter.active_dram)[0x27];
        const u32 start = native_macro_positions[entry];
        const auto emit = [&](u32 address, u32 value) {
            static u32 diagnostic_writes{};
            static const bool log_emits = std::getenv("EDEN_NX2_SEM_WATCH_EMITS") != nullptr;
            if (watched && log_emits) LOG_INFO(HW_GPU, "Semaphore watch: compute macro emit {:#x} = {:#x}", address, value);
            if (diagnostic_writes++ < 4096) LOG_DEBUG(HW_GPU, "Native compute MME emit method={:#x}, value={:#x}", address, value);
            if (address >= 0xe00) return false;
            CallMethod(system, address, value, true);
            return true;
        };
#ifdef ARCHITECTURE_x86_64
        // Compiled macro code, as for 3D macros (macro_turing_jit.h); EDEN_NX2_MME_JIT=0 off.
        static const bool use_jit = [] {
            const char* value = std::getenv("EDEN_NX2_MME_JIT");
            return value == nullptr || value[0] != '0';
        }();
        if (use_jit && !watched) {
            if (!native_macro->native_jit) {
                native_macro->native_jit = std::make_shared<TuringMacroJit>();
            }
            native_macro->native_jit->Prepare(native_macro->native_code,
                                              native_macro->native_code_generation);
            const auto result = native_macro->native_jit->Execute(
                start, native_parameters,
                std::span<u32>{*native_macro->native_interpreter.active_dram},
                [&](u32 address) { return address < native_registers.size() ? native_registers[address] : 0U; },
                emit);
            if (!result.success) {
                native_macro->native_execution_failed = true;
                LOG_ERROR(HW_GPU, "Native MME (JIT) failed: compute start={:#x}, consumed={}/{}, error={}",
                          start, result.consumed, native_parameters.size(), result.error);
            }
            native_parameters.clear();
            return true;
        }
#endif
        // Repeated calls replay their writes, as for 3D macros (macro_memo.h).
        static const bool use_memo = [] {
            const char* value = std::getenv("EDEN_NX2_MME_MEMO");
            return value == nullptr || value[0] != '0';
        }();
        auto& memo = native_macro->native_memo;
        if (use_memo && !watched) {
            memo.Synchronize(native_macro->native_code_generation);
            if (std::remove_reference_t<decltype(memo)>::Effects effects;
                memo.Find(start, native_parameters, *native_macro->native_interpreter.active_dram,
                          effects)) {
                auto& data_ram = *native_macro->native_interpreter.active_dram;
                for (size_t i = 0; i < effects.data_ram_writes.size(); i += 2) {
                    data_ram[effects.data_ram_writes[i]] = effects.data_ram_writes[i + 1];
                }
                for (size_t i = 0; i < effects.writes.size(); i += 2) {
                    emit(effects.writes[i], effects.writes[i + 1]);
                }
                native_parameters.clear();
                return true;
            }
            native_macro->native_writes.clear();
        }
        // The interpreter consumes native_parameters as its FIFO: keep the call's own parameters.
        native_memo_parameters.assign(native_parameters.begin(), native_parameters.end());
        const u64 code_generation = native_macro->native_code_generation;
        const auto result = native_macro->native_interpreter.Execute(native_macro->native_code,
            start, native_parameters,
            [&](u32 address) { return address < native_registers.size() ? native_registers[address] : 0U; },
            [&](u32 address, u32 value) {
                if (use_memo) {
                    native_macro->native_writes.emplace_back(address, value);
                }
                return emit(address, value);
            });
        const auto& interpreter = native_macro->native_interpreter;
        if (use_memo && !watched && result.success && !interpreter.read_registers &&
            !interpreter.reread_data_ram && code_generation == native_macro->native_code_generation) {
            memo.Store(start, native_memo_parameters, interpreter.data_ram_reads,
                       interpreter.data_ram_writes, native_macro->native_writes);
        }
        if (static const bool mme_dump = std::getenv("EDEN_NX2_MME_DUMP") != nullptr;
            mme_dump && (*native_macro->native_interpreter.active_dram)[0x27] != dram0_before) {
            LOG_INFO(HW_GPU, "MME data RAM[0x27] {:#x} -> {:#x} by engine {} macro start={:#x}",
                     dram0_before, (*native_macro->native_interpreter.active_dram)[0x27], fmt::ptr(this),
                     native_macro_positions[entry]);
        }
        if (!result.success) {
            native_macro->native_execution_failed = true;
            if (static const bool diagnostics = std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr;
                diagnostics) {
                std::ofstream dump("native-compute-mme.bin", std::ios::binary);
                dump.write(reinterpret_cast<const char*>(native_macro->native_code.data()), sizeof(native_macro->native_code));
            }
            LOG_ERROR(HW_GPU, "Native MME failed: compute start={:#x}, pc={:#x}, consumed={}/{}, error={}, parameters={:08x}",
                native_macro_positions[entry], result.pc, result.consumed, native_parameters.size(), result.error,
                fmt::join(native_parameters, ","));
        }
        native_parameters.clear();
        return true;
    }
    native_registers[method] = argument;
    if (method == 0x6c3) { // SET_REPORT_SEMAPHORE_D: the compute class's semaphore release
        ProcessNativeReportSemaphore(argument);
        return true;
    }
    if (method == 0x107) {
        const GPUVAddr address = ((GPUVAddr{native_registers[0xc6]} << 32) |
            native_registers[0xc7]) << 8;
        const GPUVAddr program = (GPUVAddr{native_registers[0xc8 + 49] & 0x1ffff} << 32) |
            native_registers[0xc8 + 48];
        const u32 control = native_registers[0xa7];
        LOG_DEBUG(HW_GPU, "Native inline QMD: address={:#x}, program={:#x}, control={:#x}, grid={}/{}/{}",
            address, program, control, native_registers[0xc8 + 12],
            native_registers[0xc8 + 13] & 0xffff, native_registers[0xc8 + 14] & 0xffff);
        if (static const bool diagnostics = std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr;
            diagnostics) {
            std::ofstream dump("native-inline-qmd.bin", std::ios::binary);
            dump.write(reinterpret_cast<const char*>(native_registers.data() + 0xc8), 256);
            std::ofstream registers("native-inline-compute-registers.bin", std::ios::binary);
            registers.write(reinterpret_cast<const char*>(native_registers.data()), sizeof(native_registers));
            if (memory_manager.IsFullyMappedRange(program, 0x10000)) {
                std::vector<u8> code(0x10000);
                memory_manager.ReadBlockUnsafe(program, code.data(), code.size());
                std::ofstream shader("native-inline-compute-shader.bin", std::ios::binary);
                shader.write(reinterpret_cast<const char*>(code.data()), code.size());
            }
        }
        if (!memory_manager.IsFullyMappedRange(address, 256)) {
            native_macro->native_execution_failed = true;
            LOG_ERROR(HW_GPU, "Unmapped native inline QMD destination");
            return true;
        }
        rasterizer->AccelerateInlineToMemory(address, 256,
            std::span<const u8>{reinterpret_cast<const u8*>(native_registers.data() + 0xc8), 256});
        if ((control & (1U << 24)) || !(native_registers[0xcc] & (1U << 12))) {
            const u32 previous = native_registers[0xad];
            native_registers[0xad] = static_cast<u32>(address >> 8);
            ProcessNativeMethod(system, 0xb0, 3, true);
            native_registers[0xad] = previous;
        }
        return true;
    }
    if (front_macros && method >= 0x45 && method <= 0x48) {
        return true; // Loaded and bound by the split GPU thread's front (FrontApply).
    }
    switch (method) {
    case 0x45: native_macro->ClearCode(argument); return true;
    case 0x46: native_macro->AddCode(0, argument); return true;
    case 0x47: native_bind_pointer = argument; return true;
    case 0x48:
        if (native_bind_pointer < native_macro_positions.size()) native_macro_positions[native_bind_pointer++] = argument;
        else LOG_ERROR(HW_GPU, "Native compute macro bind out of bounds");
        return true;
    case 0xae: case 0xaf: case 0xb0: {
        const GPUVAddr qmd_address = GPUVAddr{native_registers[0xad]} << 8;
        if (method == 0xae || (method == 0xaf && !(argument & 2)) ||
            (method == 0xb0 && ((argument & 15) == 0 || (argument & 15) == 1))) return true;
        static const bool qmd_diagnostics = std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr;
        if ((argument & 15) == 12 && qmd_diagnostics) {
            static u32 captured{};
            if (captured++ == 0 && memory_manager.IsFullyMappedRange(qmd_address, 256)) {
                std::array<u32, 64> descriptor{};
                memory_manager.ReadBlockUnsafe(qmd_address, descriptor.data(), sizeof(descriptor));
                std::ofstream dump("native-increment-qmd.bin", std::ios::binary);
                dump.write(reinterpret_cast<const char*>(descriptor.data()), sizeof(descriptor));
                LOG_WARNING(HW_GPU, "Native increment dependency descriptor: address={:#x}, words={:08x}",
                    qmd_address, fmt::join(descriptor, ","));
            }
        }
        const auto error = native_qmd.Signal(qmd_address, argument,
            [&](u64 address, NativeQmdScheduler::Descriptor& descriptor) {
                if (!memory_manager.IsFullyMappedRange(address, sizeof(descriptor))) return false;
                memory_manager.ReadBlockUnsafe(address, descriptor.data(), sizeof(descriptor));
                // Grid markers (nx2_list_grid.h): each word's list is the one the back patched at
                // its address and still keeps. Any other marker (its slot reused, the word
                // written since) takes the real words its upload patched in on the host GPU.
                qmd_grid_lists = {};
                bool unknown = false;
                for (u32 i = 0; i < 3; ++i) {
                    const u32 word = descriptor[12 + i];
                    if (!Nx2ListGrid::IsMarker(word)) {
                        continue;
                    }
                    const auto device = memory_manager.GpuToCpuAddress(address + (12 + i) * 4);
                    const auto list =
                        device ? rasterizer->Nx2FindKeptWord(*device, word) : std::nullopt;
                    qmd_grid_lists[i] = list.value_or(0);
                    unknown = unknown || !list;
                }
                if (const auto grid = memory_manager.GpuToCpuAddress(address + 12 * sizeof(u32));
                    unknown && grid) {
                    rasterizer->FlushRegion(*grid, 3 * sizeof(u32));
                    memory_manager.ReadBlockUnsafe(address + 12 * sizeof(u32), descriptor.data() + 12,
                                                   3 * sizeof(u32));
                    qmd_grid_lists = {};
                }
                if (static u32 logged{}; (descriptor[53] & (1U << 23)) && logged < 8) {
                    ++logged;
                    LOG_INFO(HW_GPU, "Native QMD {:#x} releases: {:08x} | {:08x} | {:08x}", address,
                             fmt::join(std::span{descriptor}.subspan(24, 4), ","),
                             fmt::join(std::span{descriptor}.subspan(28, 4), ","),
                             fmt::join(std::span{descriptor}.subspan(52, 4), ","));
                }
                return true;
            },
            [&](u64 address, u32 payload, u32 flags) {
                const bool four_words = (flags >> 30) == 0;
                if (!memory_manager.IsFullyMappedRange(address, four_words ? 16 : sizeof(payload))) return false;
                if (flags & (1U << 26)) {
                    rasterizer->SignalFence([this, address, payload, flags] {
                        const auto combine = [&](u32 current) {
                            return ReduceSemaphore((flags >> 20) & 7, (flags >> 24) & 1, current, payload);
                        };
                        const u32 guest = memory_manager.Read<u32>(address);
                        const u32 value = combine(guest);
                        memory_manager.Write<u32>(address, value);
                        if (const auto device = memory_manager.GpuToCpuAddress(address)) {
                            Tegra::Nx2SemaphoreWaiters::Get().OnWrite(
                                *device, Tegra::Nx2MirrorShadow::Get().NoteCombinedRelease(
                                             *device, address, guest, value, combine));
                        }
                        static const u64 watch = [] {
                            const char* text = std::getenv("EDEN_NX2_SEM_WATCH");
                            return text ? u64{std::strtoull(text, nullptr, 16)} : u64{};
                        }();
                        if (watch != 0 && (address & ~u64{0xf}) == watch) {
                            LOG_INFO(HW_GPU, "Semaphore watch: QMD release reduction {:#x} -> {:#x}", address, value);
                        }
                    });
                } else if (four_words) {
                    rasterizer->Query(address, VideoCommon::QueryType::Payload,
                        VideoCommon::QueryPropertiesFlags::IsAFence | VideoCommon::QueryPropertiesFlags::HasTimeout,
                        payload, 0);
                } else if (flags & (1U << 19)) {
                    // NVN2 tracks QMD completion with a max release (bit 19) into a word other
                    // releases also advance: a plain write would move it backwards and leave
                    // GEQ acquires waiting forever.
                    rasterizer->SignalFence([this, address, payload] {
                        const auto combine = [payload](u32 current) { return (std::max)(current, payload); };
                        const u32 guest = memory_manager.Read<u32>(address);
                        const u32 value = combine(guest);
                        memory_manager.Write<u32>(address, value);
                        if (const auto device = memory_manager.GpuToCpuAddress(address)) {
                            Tegra::Nx2SemaphoreWaiters::Get().OnWrite(
                                *device, Tegra::Nx2MirrorShadow::Get().NoteCombinedRelease(
                                             *device, address, guest, value, combine));
                        }
                    });
                } else {
                    static const u64 watch = [] {
                        const char* text = std::getenv("EDEN_NX2_SEM_WATCH");
                        return text ? u64{std::strtoull(text, nullptr, 16)} : u64{};
                    }();
                    if (watch != 0 && (address & ~u64{0xf}) == (watch & ~u64{0xf})) {
                        LOG_INFO(HW_GPU, "Semaphore watch: QMD release {:#x} = {:#x} flags {:#x}", address, payload, flags);
                    }
                    rasterizer->Query(address, VideoCommon::QueryType::Payload,
                        VideoCommon::QueryPropertiesFlags::IsAFence, payload, 0);
                }
                return true;
            }, [&](const auto& descriptor) { return DispatchNativeCompute(descriptor); });
        if (error.empty()) {
            LOG_DEBUG(HW_GPU, "Native QMD control action={:#x}, QMD={:#x}", argument, qmd_address);
            return true;
        }
        native_macro->native_execution_failed = true;
        std::array<u32, 128> qmd{};
        if (memory_manager.IsFullyMappedRange(qmd_address, sizeof(qmd))) {
            memory_manager.ReadBlockUnsafe(qmd_address, qmd.data(), sizeof(qmd));
        }
        // Log a failing program's first few dispatches only (QMD words 48-49 hold its code
        // address); it fails the same way every frame.
        static std::unordered_map<u64, u32> failures;
        const u64 program = (u64{qmd[49]} << 32) | qmd[48];
        if (failures[program ^ std::hash<std::string>{}(std::string{error})]++ >= 2) {
            return true;
        }
        LOG_ERROR(HW_GPU, "Native QMD failed: {}", error);
        if (qmd_diagnostics && memory_manager.IsFullyMappedRange(qmd_address, sizeof(qmd))) {
            std::ofstream dump("native-qmd.bin", std::ios::binary);
            dump.write(reinterpret_cast<const char*>(qmd.data()), sizeof(qmd));
            std::ofstream macros("native-dispatch-mme.bin", std::ios::binary);
            macros.write(reinterpret_cast<const char*>(native_macro->native_code.data()), sizeof(native_macro->native_code));
            std::ofstream registers("native-dispatch-registers.bin", std::ios::binary);
            registers.write(reinterpret_cast<const char*>(native_registers.data()), sizeof(native_registers));
        }
        LOG_ERROR(HW_GPU, "Native compute dispatch unsupported: method={:#x}, value={:#x}, QMD={:#x}, words={:08x}", method, argument,
                  qmd_address, fmt::join(qmd, ","));
        return true;
    }
    default:
        // The inline upload register block is shared with the older class.
        if (method < Regs::NUM_REGS) regs.reg_array[method] = argument;
        return method != KEPLER_COMPUTE_REG_INDEX(exec_upload) && method != KEPLER_COMPUTE_REG_INDEX(data_upload);
    }
}

void KeplerCompute::CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    if (front_macros && !VideoCommon::GpuSplit::OnBackThread()) {
        ReleaseFront();
    }
    if (native_mode) {
        // Most of the compute class's register writes (QMD fields replayed by its macros for
        // each of thousands of dispatches a second) only store the value: ~4% of title 1's GPU
        // thread went through ProcessNativeMethod's checks for them.
        if (method < native_plain_methods.size() && native_plain_methods[method]) {
            native_registers[method] = method_argument;
            if (method < Regs::NUM_REGS) {
                regs.reg_array[method] = method_argument;
            }
            return;
        }
        if (ProcessNativeMethod(system, method, method_argument, is_last_call)) return;
    }
    if (method >= Regs::NUM_REGS) {
        LOG_ERROR(HW_GPU, "Compute register out of range: {:#x}", method);
        return;
    }

    regs.reg_array[method] = method_argument;

    switch (method) {
    case KEPLER_COMPUTE_REG_INDEX(exec_upload): {
        UploadInfo info{.upload_address = upload_address,
                        .exec_address = upload_state.ExecTargetAddress(),
                        .copy_size = upload_state.GetUploadSize(),
                        .was_dirty = upload_dirty};
        uploads.push_back(info);
        upload_state.ProcessExec(regs.exec_upload.linear != 0);
        break;
    }
    case KEPLER_COMPUTE_REG_INDEX(data_upload): {
        upload_address = current_dma_segment;
        upload_dirty = current_dirty;
        current_dirty = false;
        upload_state.ProcessData(method_argument, is_last_call);
        break;
    }
    case KEPLER_COMPUTE_REG_INDEX(launch): {
        const GPUVAddr launch_desc_loc = regs.launch_desc_loc.Address();

        for (auto& data : uploads) {
            const GPUVAddr offset = data.exec_address - launch_desc_loc;
            if (offset / sizeof(u32) == LAUNCH_REG_INDEX(grid_dim_x)) {
                const bool source_dirty = memory_manager.IsMemoryDirty(data.upload_address, data.copy_size);
                if (data.was_dirty || source_dirty) {
                    indirect_compute = {data.upload_address};
                }
            }
        }
        uploads.clear();
        ProcessLaunch();
        indirect_compute = std::nullopt;
        break;
    }
    default:
        break;
    }
}

void KeplerCompute::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    if (native_mode) {
        for (u32 i = 0; i < amount; ++i) CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        return;
    }
    switch (method) {
    case KEPLER_COMPUTE_REG_INDEX(data_upload):
        upload_address = current_dma_segment;
        upload_dirty = current_dirty;
        current_dirty = false;
        upload_state.ProcessData(base_start, amount);
        return;
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

bool KeplerCompute::DispatchNativeCompute(const NativeQmdScheduler::Descriptor& qmd) {
    launch_description = {};
    std::array<u32, 3> grid{qmd[12], qmd[13], qmd[14]};
    kept_grid.reset();
    SCOPE_EXIT {
        kept_grid.reset();
    };
    // Markers of a list-fed grid macro (nx2_list_grid.h), their lists checked by the QMD read
    // (qmd_grid_lists): a list's three, in order, dispatch from its kept words; any other mix
    // gets them read back.
    if (const auto source = Nx2ListGrid::QmdGridSource(grid[0], grid[1], grid[2]);
        source.kind != Nx2ListGrid::GridKind::Direct) {
        const auto& lists = qmd_grid_lists;
        if (source.kind == Nx2ListGrid::GridKind::Indirect && lists[0] != 0 &&
            lists[0] == lists[1] && lists[1] == lists[2]) {
            kept_grid = lists[0];
            grid = {1, 1, 1};
        } else {
            for (u32 i = 0; i < 3; ++i) {
                u32& word = grid[i];
                if (!Nx2ListGrid::IsMarker(word)) {
                    continue;
                }
                std::array<u32, 3> words{};
                if (lists[i] == 0 || !rasterizer->Nx2ReadKeptListWords(lists[i], words)) {
                    // Unreachable unless a marker reached memory unpatched (the guard admits
                    // only patched destinations): never dispatched on.
                    LOG_ERROR(HW_GPU, "List-fed grid: QMD grid word {:#x} names no kept list",
                              word);
                    word = 0;
                    continue;
                }
                word = words[Nx2ListGrid::MarkerIndex(word)];
            }
        }
    }
    launch_description.grid_dim_x.Assign(grid[0]);
    launch_description.grid_dim_y.Assign(grid[1] & 0xffff);
    launch_description.grid_dim_z.Assign(grid[2] & 0xffff);
    launch_description.shared_alloc.Assign(qmd[17] & 0x3ffff);
    launch_description.block_dim_x.Assign(qmd[18] >> 16);
    launch_description.block_dim_y.Assign(qmd[19] & 0xffff);
    launch_description.block_dim_z.Assign(qmd[19] >> 16);
    launch_description.const_buffer_enable_mask.Assign(qmd[20] & 0xff);
    launch_description.local_pos_alloc.Assign(qmd[23] & 0xffffff);
    launch_description.local_crs_alloc.Assign(qmd[50] & 0xffffff);
    regs.code_loc.address_low = qmd[48];
    regs.code_loc.address_high = qmd[49] & 0x1ffff;
    for (u32 slot = 0; slot < NumConstBuffers; ++slot) {
        const u32 high = qmd[33 + slot * 2];
        if (high & 0x1ff00) {
            LOG_ERROR(HW_GPU, "Native compute constant address exceeds supported GPU width");
            return false;
        }
        auto& buffer = launch_description.const_buffer_config[slot];
        buffer.address_low = qmd[32 + slot * 2];
        buffer.address_high.Assign(high & 0xff);
        buffer.size.Assign(((high >> 19) & 0x1fff) << 4);
    }
    if (!launch_description.block_dim_x || !launch_description.block_dim_y || !launch_description.block_dim_z)
        return false;
    bool dispatched = true;
    try {
        rasterizer->DispatchCompute();
    } catch (const Shader::Exception& error) {
        static std::unordered_set<std::string> logged;
        if (logged.insert(error.what()).second) {
            LOG_ERROR(HW_GPU, "Native compute shader rejected: {}", error.what());
        }
        dispatched = false;
        if (std::string_view{error.what()}.find("matrix multiply") !=
                   std::string_view::npos) {
            DescribeNativeResources(qmd);
            LOG_ERROR(HW_GPU, "Native tensor dispatch is unsupported; no validated replacement was available");
        }
    }
    return dispatched;
}

// Logs the images a rejected program's handles (constant bank 0, bytes 0x80-0x11c) point to, so a
// substitute (e.g. an upscale blit for a tensor-core pass) can find its inputs and output.
void KeplerCompute::DescribeNativeResources(const NativeQmdScheduler::Descriptor& qmd) {
    static u32 described{};
    if (described++ >= 2) return;
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr cbuf_address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    LOG_INFO(HW_GPU, "Native resources: cbuf0 {:#x} size {:#x}, grid {}x{}x{}, block {}x{}x{}, tic pool {:#x}",
             cbuf_address, cbuf.size.Value(), qmd[12], qmd[13] & 0xffff, qmd[14] & 0xffff,
             launch_description.block_dim_x.Value(), launch_description.block_dim_y.Value(),
             launch_description.block_dim_z.Value(), regs.tic.Address());
    for (u32 offset = 0x80; offset < 0x120; offset += 4) {
        const u32 handle = memory_manager.Read<u32>(cbuf_address + offset);
        const u32 tic_index = handle & 0xfffff;
        if (handle == 0 || tic_index > regs.tic.limit) continue;
        const auto tic = GetTICEntry(tic_index);
        LOG_INFO(HW_GPU, "  handle @{:#x} = {:#x}: tic {} address {:#x} {}x{} format {:#x} types {}/{}/{}/{}",
                 offset, handle, tic_index, tic.Address(), tic.Width(), tic.Height(),
                 static_cast<u32>(tic.format.Value()), static_cast<u32>(tic.r_type.Value()),
                 static_cast<u32>(tic.g_type.Value()), static_cast<u32>(tic.b_type.Value()),
                 static_cast<u32>(tic.a_type.Value()));
    }
}

std::optional<Texture::TICEntry> KeplerCompute::ConstBuffer0Image(u32 offset) const {
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr cbuf_address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    if (!(launch_description.const_buffer_enable_mask & 1) || cbuf.size.Value() < offset + 4 ||
        !memory_manager.IsFullyMappedRange(cbuf_address + offset, 4)) {
        return std::nullopt;
    }
    const u32 tic_index = memory_manager.Read<u32>(cbuf_address + offset) & 0xfffff;
    if (tic_index > regs.tic.limit ||
        !memory_manager.IsFullyMappedRange(regs.tic.Address() +
                                               u64{tic_index} * sizeof(Texture::TICEntry),
                                           sizeof(Texture::TICEntry))) {
        return std::nullopt;
    }
    return GetTICEntry(tic_index);
}

void KeplerCompute::ProcessLaunch() {
    const GPUVAddr launch_desc_loc = regs.launch_desc_loc.Address();
    memory_manager.ReadBlockUnsafe(launch_desc_loc, &launch_description,
                                   LaunchParams::NUM_LAUNCH_PARAMETERS * sizeof(u32));
    rasterizer->DispatchCompute();
}

Texture::TICEntry KeplerCompute::GetTICEntry(u32 tic_index) const {
    const GPUVAddr tic_address_gpu{regs.tic.Address() + tic_index * sizeof(Texture::TICEntry)};

    Texture::TICEntry tic_entry;
    memory_manager.ReadBlockUnsafe(tic_address_gpu, &tic_entry, sizeof(Texture::TICEntry));
    return tic_entry;
}

Texture::TSCEntry KeplerCompute::GetTSCEntry(u32 tsc_index) const {
    const GPUVAddr tsc_address_gpu{regs.tsc.Address() + tsc_index * sizeof(Texture::TSCEntry)};

    Texture::TSCEntry tsc_entry;
    memory_manager.ReadBlockUnsafe(tsc_address_gpu, &tsc_entry, sizeof(Texture::TSCEntry));
    return tsc_entry;
}

// Split GPU thread with macros on its front half (gpu_split.h, EDEN_NX2_SPLIT=2): the front
// runs compute macros against its own copy of the native registers (which hold the last value
// written to each, on the back too) and loads and binds the compute macro code itself.
struct KeplerCompute::FrontMacro {
    const MacroEngine* engine{};
    u64 generation{};
    u32 start{};
    u64 hash{};
    std::optional<Nx2ListGrid::GridCells> cells; ///< An HleMacros entry's store words
};

struct KeplerCompute::FrontState {
    std::array<u32, 0x1000> registers{};
    std::optional<u32> executing_macro;
    std::vector<u32> parameters;
    bool compute_uploaded{};
    u32 bind_pointer{};
    MacroEngine* macro{}; ///< Without a native context: the one macro engine
    std::vector<FrontMacro> macros; ///< FrontMacroInfo's, by engine, code generation and start
    Nx2ListGrid::UploadState upload;
    std::vector<u64> upload_lists; ///< Lists whose markers the upload still on its way carries
    /// The running call's markers into upload data (FrontGuardListGrid), by emit.
    std::vector<Nx2ListGrid::UploadMarker> upload_markers;
};

KeplerCompute::FrontState& KeplerCompute::Front(VideoCommon::GpuSplit& split) {
    if (!front) [[unlikely]] {
        // Copy the back's state and take the macros over while it is idle.
        split.Drain(VideoCommon::DrainSite::FrontStart);
        front = std::make_unique<FrontState>();
        front->registers = native_registers;
        // Data the upload took before is unknown here: a marker into it isn't patched.
        for (const u32 method : {0x60U, 0x61U, 0x62U, 0x63U, 0x6cU}) {
            front->upload.Apply(method, native_registers[method]);
        }
        front->upload.offset = front->upload.Size();
        front->executing_macro = native_executing_macro;
        front->parameters = std::move(native_parameters);
        native_parameters.clear();
        native_executing_macro.reset();
        front->compute_uploaded = native_compute_uploaded;
        front->bind_pointer = native_bind_pointer;
        front->macro = native_macro;
        front_macros = true;
    }
    return *front;
}

// As Maxwell3D::ReleaseFront: the split stopped, the engine runs its macros again.
void KeplerCompute::ReleaseFront() {
    // The engine's macros run unguarded from now on: markers get their real words.
    if (grid_marker != nullptr && grid_marker->set) {
        if (native_data_ram) {
            Nx2ListGrid::ResolveCells(*grid_marker, *native_data_ram);
        }
        for (u32 i = 0; i < 3; ++i) {
            u32& value = native_registers[Nx2ListGrid::QmdGridRegister + i];
            if (Nx2ListGrid::IsMarker(value) && grid_marker->Find(value) != nullptr) {
                value = Nx2ListGrid::ResolveWord(*grid_marker, value).value_or(0);
            }
        }
    }
    native_executing_macro = front->executing_macro;
    native_parameters = std::move(front->parameters);
    native_compute_uploaded = front->compute_uploaded;
    native_bind_pointer = front->bind_pointer;
    native_macro = front->macro;
    front.reset();
    front_macros = false;
    LOG_INFO(HW_GPU, "GPU thread split stopped: compute macros back on the engine");
}

void KeplerCompute::FrontApply(FrontState& fs, u32 method, u32 argument) {
    // As ProcessNativeMethod does with the macro code and bindings.
    if (method == 0x45 || method == 0x47) {
        fs.compute_uploaded = true;
    }
    fs.registers[method] = argument;
    if (method == Nx2ListGrid::ExecUploadMethod) {
        fs.upload_lists.clear();
    } else if (method == Nx2ListGrid::DataUploadMethod && grid_marker != nullptr &&
               Nx2ListGrid::IsMarker(argument)) {
        if (const auto* const list = grid_marker->Find(argument)) {
            fs.upload_lists.push_back(list->keep_id);
        }
    } else if (method == 0x107 && grid_marker != nullptr) {
        // The inline QMD ProcessNativeMethod writes: its grid words are the grid registers'.
        const GPUVAddr qmd = ((GPUVAddr{fs.registers[0xc6]} << 32) | fs.registers[0xc7]) << 8;
        for (u32 i = 0; i < 3; ++i) {
            const u32 word = fs.registers[Nx2ListGrid::QmdGridRegister + i];
            if (Nx2ListGrid::IsMarker(word)) {
                FrontExpectKeptWord(qmd + (12 + i) * sizeof(u32), word);
            }
        }
    }
    fs.upload.Apply(method, argument);
    if (method == Nx2ListGrid::DataUploadMethod && !fs.upload.InFlight()) {
        fs.upload_lists.clear();
    }
    switch (method) {
    case 0x45:
        FrontMacroEngine(fs).ClearCode(argument);
        break;
    case 0x46:
        FrontMacroEngine(fs).AddCode(0, argument);
        break;
    case 0x47:
        fs.bind_pointer = argument;
        break;
    case 0x48: {
        const auto positions = FrontMacroPositions(fs);
        if (fs.bind_pointer < positions.size()) {
            positions[fs.bind_pointer++] = argument;
        } else {
            LOG_ERROR(HW_GPU, "Native compute macro bind out of bounds");
        }
        break;
    }
    default:
        break;
    }
}

MacroEngine& KeplerCompute::FrontMacroEngine(FrontState& fs) {
    if (native_context) {
        return fs.compute_uploaded ? native_context->compute_engine : native_context->engine;
    }
    return *fs.macro;
}

std::span<u32> KeplerCompute::FrontMacroPositions(FrontState& fs) {
    if (native_context) {
        return fs.compute_uploaded ? std::span<u32>{native_context->compute_positions}
                                      : std::span<u32>{native_context->positions};
    }
    return native_macro_positions;
}

void KeplerCompute::FrontCallMethod(Core::System& system, VideoCommon::GpuSplit& split,
                                    u32 method, u32 argument, bool is_last_call,
                                    const GPUVAddr* segment) {
    auto& fs = Front(split);
    if (method >= fs.registers.size()) {
        split.Method(this, method, argument, is_last_call, segment);
        return;
    }
    if (method >= 0xe00) {
        if (!fs.executing_macro) {
            if (method & 1) {
                LOG_ERROR(HW_GPU, "Native compute macro starts on argument method");
                return;
            }
            fs.executing_macro = method;
        }
        fs.parameters.push_back(argument);
        if (is_last_call) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    FrontApply(fs, method, argument);
    split.Method(this, method, argument, is_last_call, segment);
    if (method == 0x6c3) {
        FrontNoteReport(split, fs, argument);
    }
}

bool KeplerCompute::FrontRunsMacroWord(u32 method) const {
    const auto& executing = front ? front->executing_macro : native_executing_macro;
    return method >= 0xe00 && method < native_registers.size() &&
           (executing || (method & 1) == 0);
}

void KeplerCompute::FrontNoteReport(VideoCommon::GpuSplit& split, const FrontState& fs,
                                    u32 control) {
    // SET_REPORT_SEMAPHORE_D was sent: what ProcessNativeReportSemaphore will write on the back.
    const u32 operation = control & 3;
    if (operation != 0 && operation != 2) {
        return;
    }
    const GPUVAddr address =
        (GPUVAddr{fs.registers[0x6c0] & 0x1ffffff} << 32) | fs.registers[0x6c1];
    const auto device = memory_manager.GpuToCpuAddress(address);
    if (!device) {
        return;
    }
    // A reduction combines the payload with the word's value then.
    const bool reduction = ((control >> 3) & 1) != 0;
    split.NoteSemaphoreWrite(*device, fs.registers[0x6c2], !reduction);
    if (!reduction && ((control >> 28) & 1) == 0) {
        // 64 bits (a timestamp follows).
        split.NoteSemaphoreWrite(*device + 4, 0, true);
    }
}

const KeplerCompute::FrontMacro& KeplerCompute::FrontMacroInfo(FrontState& fs,
                                                               const MacroEngine& engine,
                                                               u32 start) {
    const auto known = std::ranges::find_if(fs.macros, [&](const FrontMacro& entry) {
        return entry.engine == &engine && entry.generation == engine.native_code_generation &&
               entry.start == start;
    });
    if (known != fs.macros.end()) {
        return *known;
    }
    if (fs.macros.size() >= 256) {
        fs.macros.clear();
    }
    const std::span<const u32> code{engine.native_code};
    const u64 hash = TuringMacroInterpreter::CodeHash(
        code, start, TuringMacroInterpreter::AnalyzeEffects(code, start));
    const auto cells = Nx2ListGrid::RecognizeHleMacro(code, start, hash);
    if (cells) {
        LOG_INFO(HW_GPU,
                 "List-fed grid: compute macro {:016x} at {:#x} stores its parameters in data RAM "
                 "words {:#x}, {:#x}, {:#x}; its lists are taken unread",
                 hash, start, (*cells)[0], (*cells)[1], (*cells)[2]);
    } else if (Nx2ListGrid::FindHleMacro(hash) != nullptr) {
        LOG_WARNING(HW_GPU,
                    "List-fed grid: compute macro {:016x} at {:#x} does not store its parameters "
                    "as its entry says; its lists are read",
                    hash, start);
    }
    fs.macros.push_back({&engine, engine.native_code_generation, start, hash, cells});
    return fs.macros.back();
}

std::optional<u64> KeplerCompute::FrontClaimListGrid(VideoCommon::GpuSplit& split, u32 method) {
    auto& fs = Front(split);
    // The three words must be the whole call.
    if (fs.executing_macro || !fs.parameters.empty() || method < 0xe00 ||
        method >= fs.registers.size() || (method & 1) != 0 || grid_marker == nullptr ||
        rasterizer == nullptr || !rasterizer->Nx2ListGridSupported()) {
        return std::nullopt;
    }
    const auto positions = FrontMacroPositions(fs);
    if (positions.empty()) {
        return std::nullopt;
    }
    const u32 start = positions[((method - 0xe00) / 2) % positions.size()];
    if (!FrontMacroInfo(fs, FrontMacroEngine(fs), start).cells) {
        return std::nullopt;
    }
    const u64 keep_id = Nx2ListGrid::NextKeepId();
    const Nx2ListGrid::SlotOwner owner{
        keep_id,
        [](void* self, u64 evicted) {
            return static_cast<KeplerCompute*>(self)->FrontEvictList(evicted);
        },
        this};
    if (!Nx2ListGrid::ClaimSlot(owner)) {
        return std::nullopt;
    }
    return keep_id;
}

bool KeplerCompute::FrontEvictList(u64 keep_id) {
    namespace ListGrid = Nx2ListGrid;
    // Without a front, ReleaseFront resolved the markers.
    if (!front || grid_marker == nullptr) {
        return true;
    }
    auto& fs = *front;
    // Its markers are in upload data the back patches once the upload is complete.
    if (std::ranges::find(fs.upload_lists, keep_id) != fs.upload_lists.end()) {
        return false;
    }
    ListGrid::KeptList* const list = grid_marker->Find(ListGrid::Marker(keep_id, 0));
    if (list == nullptr || list->keep_id != keep_id) {
        return true;
    }
    auto& data_ram = native_data_ram ? *native_data_ram
                                     : *FrontMacroEngine(fs).native_interpreter.active_dram;
    std::array<bool, 3> forced = grid_marker->keep_id == keep_id ? grid_marker->Marked(data_ram)
                                                                 : std::array<bool, 3>{};
    ListGrid::EvictFromRam(*grid_marker, data_ram, keep_id);
    for (u32 i = 0; i < 3; ++i) {
        const u32 address = ListGrid::QmdGridRegister + i;
        const u32 value = fs.registers[address];
        if (ListGrid::IsMarker(value) && grid_marker->Find(value) == list) {
            const u32 word = ListGrid::ResolveWord(*grid_marker, value).value_or(0);
            FrontApply(fs, address, word);
            VideoCommon::GpuSplit::Get().MacroWrite(this, address, word);
            forced[ListGrid::MarkerIndex(value)] = true;
        }
    }
    if (ListGrid::g_verify && list->verified) {
        for (u32 i = 0; i < 3; ++i) {
            if (!forced[i]) {
                continue;
            }
            ListGrid::NoteVerify(
                ListGrid::Check::Forced, list->hash, list->resolved && list->words[i] == list->real[i],
                fmt::format("list {} word {} resolved as {:#x} before its slot was reused, the "
                            "real word is {:#x}",
                            keep_id, i, list->words[i], list->real[i]));
        }
    }
    return true;
}

void KeplerCompute::FrontTakeListGrid(VideoCommon::GpuSplit& split, u32 method, u64 keep_id,
                                      GPUVAddr address, const Nx2ListGrid::ListReader& reader,
                                      std::span<const u32> real) {
    namespace ListGrid = Nx2ListGrid;
    auto& fs = Front(split);
    const auto positions = FrontMacroPositions(fs);
    const u32 start = positions[((method - 0xe00) / 2) % positions.size()];
    auto& macro = FrontMacroEngine(fs);
    const FrontMacro& info = FrontMacroInfo(fs, macro, start);
    if (native_data_ram) {
        macro.native_interpreter.active_dram = native_data_ram;
    }
    auto& data_ram = *macro.native_interpreter.active_dram;
    // As the macro, with a marker for each word.
    ListGrid::KeptList list{
        .keep_id = keep_id, .address = address, .hash = info.hash, .reader = reader};
    if (real.size() == list.real.size()) {
        list.verified = true;
        std::ranges::copy(real, list.real.begin());
    }
    for (u32 i = 0; i < 3; ++i) {
        data_ram[(*info.cells)[i]] = ListGrid::Marker(keep_id, i);
    }
    grid_marker->cells = *info.cells;
    grid_marker->keep_id = keep_id;
    grid_marker->set = true;
    grid_marker->Remember(list);
    // At the list's place in the stream, the back keeps a copy of the words. It writes no guest
    // memory.
    const u64 hash = info.hash;
    if (list.verified) {
        split.DeferWords(
            [this, keep_id, address, hash](std::span<const u32> words) {
                rasterizer->Nx2KeepListWords(keep_id, address, hash, words);
            },
            real, VideoCommon::GpuSplit::Writes::Noted);
    } else {
        split.Defer(
            [this, keep_id, address, hash] {
                rasterizer->Nx2KeepListWords(keep_id, address, hash, {});
            },
            0, VideoCommon::GpuSplit::Writes::Noted);
    }
}

void KeplerCompute::FrontExpectKeptWord(GPUVAddr address, u32 marker) {
    const auto* const list = grid_marker->Find(marker);
    if (list == nullptr) {
        return;
    }
    VideoCommon::GpuSplit::Get().Defer(
        [this, address, keep_id = list->keep_id, index = Nx2ListGrid::MarkerIndex(marker)] {
            if (const auto device = memory_manager.GpuToCpuAddress(address)) {
                rasterizer->Nx2ExpectKeptWord(*device, keep_id, index);
            }
        },
        0, VideoCommon::GpuSplit::Writes::Noted);
}

void KeplerCompute::FrontGuardListGrid(VideoCommon::GpuSplit& split, FrontState& fs,
                                       MacroEngine& macro, u32 start) {
    namespace ListGrid = Nx2ListGrid;
    auto& data_ram = *macro.native_interpreter.active_dram;
    // Markers reach registers only as moves into the QMD grid registers (Admit).
    const auto marker_register = [this](u32 address, u32 value) {
        return address - ListGrid::QmdGridRegister < 3 && ListGrid::IsMarker(value) &&
               grid_marker->Find(value) != nullptr;
    };
    const auto marked = grid_marker->Marked(data_ram);
    bool live = marked[0] || marked[1] || marked[2];
    for (u32 i = 0; i < 3 && !live; ++i) {
        live = marker_register(ListGrid::QmdGridRegister + i,
                               fs.registers[ListGrid::QmdGridRegister + i]);
    }
    if (!live) {
        return;
    }
    const auto read = [&fs](u32 address) {
        return address < fs.registers.size() ? fs.registers[address] : 0U;
    };
    const std::span<const u32> code{macro.native_code};
    const auto result = ListGrid::DryRun(code, start, fs.parameters, data_ram, grid_marker->cells,
                                         marked, read, marker_register);
    const auto admission = ListGrid::Admit(result, fs.upload);
    if (admission == ListGrid::Admission::Unaffected) {
        return;
    }
    if (admission == ListGrid::Admission::Moves) {
        // The back patches exactly the upload words the markers go to (PlanPatches), told as
        // the call writes each (FrontRunMacro).
        fs.upload_markers = ListGrid::UploadMarkerWords(result, fs.upload);
        if (ListGrid::g_verify) {
            // The same call on the real words, where every list's were read the old way.
            const auto real_word = [this](u32 marker) -> std::optional<u32> {
                const ListGrid::KeptList* const list = grid_marker->Find(marker);
                if (list == nullptr || !list->verified) {
                    return std::nullopt;
                }
                return list->real[ListGrid::MarkerIndex(marker)];
            };
            bool known = true;
            auto real_ram = std::make_unique<ListGrid::DataRam>(data_ram);
            for (u32 i = 0; i < 3; ++i) {
                if (marked[i]) {
                    const auto word = real_word(data_ram[grid_marker->cells[i]]);
                    known = known && word.has_value();
                    (*real_ram)[grid_marker->cells[i]] = word.value_or(0);
                }
            }
            const auto real_read = [&](u32 address) {
                const u32 value = read(address);
                if (!marker_register(address, value)) {
                    return value;
                }
                const auto word = real_word(value);
                known = known && word.has_value();
                return word.value_or(0);
            };
            const auto real = ListGrid::DryRun(code, start, fs.parameters, *real_ram,
                                               grid_marker->cells, {}, real_read,
                                               [](u32, u32) { return false; });
            if (known) {
                // An admitted call stores no marker in the data RAM (Admit): its data RAM
                // writes compare as they are.
                const bool match =
                    real.completed &&
                    ListGrid::MatchesRealRun(result.emits, real.emits, real_word) &&
                    result.data_ram_writes == real.data_ram_writes;
                ListGrid::NoteVerify(ListGrid::Check::Reader,
                                     FrontMacroInfo(fs, macro, start).hash, match,
                                     fmt::format("compute macro at {:#x}: {} writes on the markers, "
                                                 "{} on the real words",
                                                 start, result.emits.size(), real.emits.size()));
            }
        }
        return;
    }
    if (static u32 logged{}; logged < 16) {
        ++logged;
        LOG_INFO(HW_GPU,
                 "List-fed grid: compute macro {:016x} at {:#x} gets the real words: {}",
                 FrontMacroInfo(fs, macro, start).hash, start,
                 result.reason.empty() ? std::string_view{"a marker goes elsewhere"}
                                       : result.reason);
    }
    ListGrid::ResolveCells(*grid_marker, data_ram);
    for (u32 i = 0; i < 3; ++i) {
        const u32 address = ListGrid::QmdGridRegister + i;
        if (marker_register(address, fs.registers[address])) {
            const u32 value = ListGrid::ResolveWord(*grid_marker, fs.registers[address]).value_or(0);
            FrontApply(fs, address, value);
            split.MacroWrite(this, address, value);
        }
    }
}

void KeplerCompute::FrontCallMultiMethod(Core::System& system, VideoCommon::GpuSplit& split,
                                         u32 method, const u32* base_start, u32 amount,
                                         u32 methods_pending, const GPUVAddr* segment) {
    auto& fs = Front(split);
    if (method >= 0xe00 && method < fs.registers.size()) {
        for (u32 i = 0; i < amount; ++i) {
            FrontCallMethod(system, split, method, base_start[i], methods_pending - i <= 1, segment);
        }
        return;
    }
    if (method < fs.registers.size()) {
        for (u32 i = 0; i < amount; ++i) {
            FrontApply(fs, method, base_start[i]);
        }
    }
    split.MultiMethod(this, method, base_start, amount, methods_pending, segment);
    if (method == 0x6c3) {
        for (u32 i = 0; i < amount; ++i) {
            FrontNoteReport(split, fs, base_start[i]);
        }
    }
}

void KeplerCompute::FrontRunMacro(Core::System& system, VideoCommon::GpuSplit& split,
                                  FrontState& fs) {
    const auto positions = FrontMacroPositions(fs);
    const u32 method = *fs.executing_macro;
    const u32 start = positions[((method - 0xe00) / 2) % positions.size()];
    fs.executing_macro.reset();
    auto& macro = FrontMacroEngine(fs);
    if (native_data_ram) {
        macro.native_interpreter.active_dram = native_data_ram;
    }
    fs.upload_markers.clear();
    if (grid_marker != nullptr && grid_marker->set) {
        FrontGuardListGrid(split, fs, macro, start);
    }
    const auto read = [&fs](u32 address) {
        return address < fs.registers.size() ? fs.registers[address] : 0U;
    };
    std::size_t emitted = 0;
    std::size_t next_marker = 0;
    const auto emit = [&](u32 address, u32 value) {
        if (next_marker < fs.upload_markers.size() &&
            fs.upload_markers[next_marker].emit == emitted) {
            const auto& marker = fs.upload_markers[next_marker++];
            FrontExpectKeptWord(marker.address, marker.marker);
        }
        ++emitted;
        if (address >= 0xe00) {
            return false;
        }
        FrontApply(fs, address, value);
        split.MacroWrite(this, address, value);
        if (address == 0x6c3) {
            FrontNoteReport(split, fs, value);
        }
        return true;
    };
#ifdef ARCHITECTURE_x86_64
    if (!macro.native_jit) {
        macro.native_jit = std::make_shared<TuringMacroJit>();
    }
    macro.native_jit->Prepare(macro.native_code, macro.native_code_generation);
    const auto result = macro.native_jit->Execute(
        start, fs.parameters, std::span<u32>{*macro.native_interpreter.active_dram}, read, emit);
#else
    const auto result =
        macro.native_interpreter.Execute(macro.native_code, start, fs.parameters, read, emit);
#endif
    if (!result.success) {
        macro.native_execution_failed = true;
        LOG_ERROR(HW_GPU, "Native MME (split front) failed: compute start={:#x}, consumed={}, error={}",
                  start, result.consumed, result.error);
    }
    fs.parameters.clear();
}

} // namespace Tegra::Engines
