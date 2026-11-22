// SPDX-License-Identifier: GPL-3.0-or-later
// nanguard (NXN1X 11-21, NPARK: Ocarina of Time's whole-picture white flash): from Longshot's gpu/vk/eden/shader_recompiler/ir_opt/fragment_output_guard.h
// (refer\Longshot, GPL-3.0; the same idea as NX2navi's patch 0027). Longshot: "One such pixel in the scene went through the bloom's chain of averages to the
// whole picture: the one-frame white flash, the HUD over it" (its RX_VK_PROBE caught a NaN in the scene target, the blur after it all at its top).
// Called from vk_pipeline_cache.cpp for fragment programs; DM_VK_NAN_GUARD=0 turns it off.

#pragma once

#include <utility>
#include <vector>

#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/program.h"

namespace Shader::Optimization {

/// Not a NaN or an infinity into a float colour target (an integer target's bits go as they are). float_targets: bit i, target i is a float one.
/// (NXN1X: clamped to 65000, under the largest half float (65504) and B10G11R11 value (65024), not Longshot's 1e30: OoT's scene and bloom targets are R11G11B10 and RGBA16F, where 1e30 is an infinity again)
/// Returns the colour component writes guarded
inline size_t GuardFragmentOutputs(IR::Program& program, u32 float_targets) {
    if (program.stage != Stage::Fragment) return 0;
    std::vector<std::pair<IR::Block*, IR::Inst*>> writes;
    for (IR::Block* block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::SetFragColor && inst.Arg(0).IsImmediate() &&
                (float_targets >> inst.Arg(0).U32() & 1) != 0 && !inst.Arg(2).IsImmediate()) {
                writes.emplace_back(block, &inst);
            }
        }
    }
    for (const auto& [block, inst] : writes) {
        IR::IREmitter ir{*block, block->Instructions().iterator_to(*inst)};
        const IR::F32 value{inst->Arg(2)};
        const IR::F32 limited{ir.FPClamp(value, ir.Imm32(-65000.0f), ir.Imm32(65000.0f))};
        const IR::F32 safe{ir.Select(ir.FPIsNan(value), ir.Imm32(0.0f), limited)};
        inst->SetArg(2, safe);
    }
    return writes.size();
}

} // namespace Shader::Optimization
