# NXN1X 2.0 Technical Document

Korean original: NXN1X_2.0_TECHNICAL.md

Written: 2026-11-22. Scope: the first NXN1X release (2026-11-22 build, `build\release\bin\NXN1X.exe`).

This document only collects what was confirmed in the repository's documents, design notes, source comments, and the project memory notes. The "Source" line at the end of each item names the files where that content was confirmed. Paths are relative to the repository root (`nxn1x\`). Where documents disagree or there is no evidence, the text says "Mismatch" or "Unconfirmed". The full list is in section 8.3.

Terms:

| Term | Meaning |
|---|---|
| NCE | Native Code Execution. Eden's CPU backend that runs guest ARM64 code directly on the host CPU |
| NVN / NVN2 | The graphics API of Switch 1 / Switch 2 |
| NVN host layer | The `host\dm` code that receives the game's NVN calls via HLE and replays them on Vulkan. Taken from the nxnative line |
| Emulation path | Eden's original Maxwell GPU emulation |
| import / copy / hybrid | How game memory pools are made visible to Vulkan. Section 4.4 |
| replay | The step that re-records recorded NVN commands into Vulkan command buffers |
| HWAoI | Hyrule Warriors: Age of Imprisonment |
| OoT | Ocarina of Time (Switch 2 edition, 0400486024DEC000) |
| DK | Donkey Kong Bananza (040074A01BF12000) |
| XCDE / XC2 | Xenoblade Chronicles DE / Xenoblade Chronicles 2 (Switch 2 Edition) |

---

## 1. Overview

### 1.1 What NXN1X is

NXN1X 2.0 is a Nintendo Switch / Switch 2 emulator based on Eden (a yuzu derivative). The target is Windows ARM64, and the current reference hardware is an NVIDIA RTX Spark N1X laptop.

- Version: 2.0 (window title "NXN1X 2.0", `source\src\yuzu\yuzu.rc` FileDescription/ProductName).
- Skeleton: **SnapRYU 2.0**. NXN1X 2.0 is built on the SnapRYU 2.0 skeleton (it keeps only the name and is not Ryujinx-based). Go mode also comes from it and is hidden in 2.0 (`NXN1X_GO_MODE=1` shows it).
- Mainly changed: `video_core` (NVN path, Vulkan renderer), `core` (NCE execution path: instruction handling, x18/FPCR, patcher; loaders), `shader_recompiler`, the `yuzu` UI, `host/dm/windows` (Vulkan backend).
- Target device: SnapRYU 2.0 targets the Snapdragon X Elite (Adreno/Turnip); NXN1X 2.0 targets the NVIDIA N1X.

- GPU: Blackwell family, PCI 10de:2e03, driver 621.24, Vulkan 1.4.357. Windows 11 ARM64 28000.3501, WDDM 3.2, hardware scheduling on.
- CPU: hybrid configuration with 10 P-cores + 10 E-cores. VRAM budget about 80 GB.
- Executable: `build\release\bin\NXN1X.exe`. Diagnostic instrumentation is separated by the CMake option `NXN1X_DIAGNOSTICS` (default OFF).
- License: derived from Eden/yuzu. Original author attribution and the GPL license are kept.

Source: `docs\DLSS_ARM64_CAPABILITIES.md`, memory note `host-tuned-for-adreno.md`, `source\CMakeLists.txt`, `source\README.md`.

### 1.2 The Adreno period (record)

NXN1X was originally a project for the Qualcomm Snapdragon X Elite (Adreno GPU, Turnip/Mesa driver, 12 uniform cores).

- It started as a copy of SnapRYU (the X Elite project at the time). The current skeleton is SnapRYU 2.0 (see 1.1).
- Since 2026-11 it runs on the N1X. Right after the move to the N1X, NCE did not work at all, and even after the fix, performance was low from the start (not a later regression).
- The host layer (`host\dm\windows`) was tuned on Adreno/Turnip. When a cost or a bug shows up on the N1X, first check whether it is an Adreno-era default. Adreno compatibility is no longer maintained.

Adreno-era documents and traces:
- The old `source\README.md` (before 2.0): a plan targeting "Windows ARM64 and Qualcomm Adreno": NPU frame generation (QNN/ORT), TotK 1080p lock, `tools/build.ps1`, etc. It was rewritten for 2.0 and now points to the repository README.
- `README.md` (root): a description of a source snapshot that contains Turnip/Mesa sources, the F1R/F2R driver overlay, and LSFG integration.
- `docs\nvn-memory\`: memory-based work from 2026-11-16 to 18 (stages F2 to F5Z). Assumes a 12 GB-class device and Turnip. Most of it ended in a "runtime pending" state.
- Traces left in code: on Switch 2 boot, `TU_GS_ONCE=1` (a Turnip-only environment variable) is set (`source\src\core\nvn\nvn_layer.cpp` `OnAppLoad`). `source\src\video_core\renderer_vulkan\vk_npu_frame_gen.*` also remains.

Source: memory note `host-tuned-for-adreno.md`, `source\README.md`, `README.md`, `docs\nvn-memory\FOUNDATION_DESIGN.txt`, `docs\nvn-memory\IMPLEMENTATION_PLAN.json`.

### 1.3 GPU path: Emulation and the NVN host layer

For games run with NCE, the GPU path is chosen with the setting `nce_gpu_path` (Cpu category). It is read once at boot.

| Value | Behavior |
|---|---|
| `Emulation` (code default) | Eden's Maxwell GPU emulation |
| `Nvn` | Sends the game's NVN calls to the NVN host layer (`source\src\nvn_host` → `host\dm`) |

Even with `Nvn` selected, the following cases fall back to Emulation and leave one log line (`core\nvn\nvn_layer.cpp` `OnAppLoad`, `Refuse`).
- NCE does not run that game
- The renderer is not Vulkan
- The NVN layer already ran once in the same process (NXN1X must be restarted)

Differences between Switch 1 and Switch 2:
- **Switch 1 (NVN1)**: log `GPU path: NVN host layer`.
- **Switch 2 (NVN2)**: log `GPU path: NVN host layer (Switch 2, NVN2)`. The gate reads the NVN2 constants of the sdk module (`nvn_gate.cpp` ReadNvn2). The shader bridge lowers Ampere code to Maxwell (section 4.3). If the environment has no value, it sets `TU_GS_ONCE=1` and `DM_VK_TBUF_FRESH=1` and sets `NXN1X_NVN2_HOST=1`.
- The default resolution for Switch 2 programs is 1x (`ns2res1x`, `common\settings.cpp` `Switch2Res1x()`). If a per-game setting exists, it is used. With `NXN1X_NS2_RES_1X=0`, the global setting is used as before.

Unconfirmed: whether `nce_gpu_path` is actually `Nvn` in the release profile cannot be known from the source alone. The code default is `Emulation` (`common\settings.h`).

Source: `source\src\common\settings.h`, `source\src\common\settings.cpp`, `source\src\core\nvn\nvn_layer.cpp`, `docs\RELEASE_2026-11-22.md` section 3.

---

## 2. Architecture

### 2.1 Overall structure

```
Guest code (NCE, runs directly on the host CPU)
  │  nvnBootstrapLoader hook → H gate (core\nvn\nvn_gate.cpp)
  ▼
NVN host layer (host\dm\gfx\nvn\hle_nvn.c)
  │  calls with meaning are implemented as real objects; GPU state and draws are recorded as commands
  ▼
Vulkan replay (host\dm\windows\gfx_vk_exec.c, gfx_vk_eden.inc, gfx_vk_core.c, gfx_vk_tex.c)
  │  uses Eden pipelines (video_core\nvn bridge → Eden shader recompiler)
  ▼
present (gfx_vk_present.c) → core\nvn\nvn_output.cpp → Eden compositor (nvn_display.cpp)
  │  DLSS SR / DLSS-G (Streamline), optical-flow frame interpolation (fsr3_frame_gen)
  ▼
Vulkan swapchain
```

The NVN host layer is built as one C static library, `nvn_host` (`source\src\nvn_host\CMakeLists.txt`). It uses the host tree's own build method as is (GNU C from `dm\windows\build.py`, blocks, Darwin compatibility headers). It includes these files:
- `gfx/nvn/hle_nvn.c`, `nvn_dirty.c`, `nvn_tex.c`, `nvn_capture.c`, `nvn_entry.S`, `nvn_thunks.S`
- `windows/gfx_vk_core.c`, `gfx_vk_exec.c`, `gfx_vk_present.c`, `gfx_vk_shader.c`, `gfx_vk_tex.c`, `gfx_vk_win_glue.c`, `nvn_memory_*.c`
- `nxn1x/nvnh_rt.c`, `nvnh_posix.c`, `nvnh_memory_guard.cpp`

`dm/host` (kernel and services for the host's own process), `posix_win.c`, and the host's window are not built. Threads, module loading, exceptions, the window, and the performance panel are handled by NXN1X (Eden). `nvnh_rt.c` holds the replacement functions.

Source: `source\src\nvn_host\CMakeLists.txt`, `host\dm\nxn1x\nvnh_rt.c` header comment.

### 2.2 CPU: Windows NCE

#### 2.2.1 Layout

The NCE folder is `source\src\core\arm\nce\`. The Windows build compiles the following (`core\CMakeLists.txt` around lines 1304-1335).
- `arm_nce_windows.cpp`, `windows_*`, `x18_*`, `windows_nce_entry.asm` (armasm64), `core\hle\service\ro\ro_nce_windows.cpp`

`arm_nce.cpp`, `arm_nce.s`, and `interpreter_visitor.cpp` are compiled only on the non-Windows (Linux/Android) path.

Size (at the 2026-11 review):

| Item | Value |
|---|---|
| Whole NCE folder | 39 files, 15,864 lines |
| Windows only | about 8,410 lines |
| Linux/Android only (Eden original) | 4,366 lines |
| Shared (patcher, etc.) | 3,086 lines |
| `ArmNce::RunThread` (Windows) | 693 lines (Eden original: 59 lines) |

Source: `design\NCE_CLEANUP_REVIEW.md` sections 0 and 5.

#### 2.2.2 Handling needed on Windows

| Concern | Windows implementation |
|---|---|
| Exception entry | VEH (`AddVectoredExceptionHandler`). Exceptions are handled on the current stack of the faulting thread, that is, the guest stack |
| Leaving the guest after a fault | `RedirectToHost` + `NtContinue` enter `WindowsNceHostStackBridge` |
| Guest entry | `WindowsNceEnterGuest` / `EnterGuestContext` (armasm64) |
| x18 | On Windows it is the TEB pointer, so the guest cannot own it. Section 2.2.3 |
| Thread-state pointer in generated code | TEB TLS slot (`LDR [x18,#slot]`) or a C getter. `yuzu\main.cpp:102` reserves the slot at startup |
| TEB StackBase/StackLimit | `RtlRestoreContext` checks SP, so they must match SP (tebfix, nceresume). Section 2.2.4 |
| Stopping another thread | Per-core worker does Suspend/GetThreadContext/SetThreadContext/Resume. Delivery gate state machine and BreakRescue |
| EL0 cache instructions, CTR_EL0 | Windows blocks them, so they are emulated in the VEH. The patcher skips DC CIVAC and emulates CTR_EL0 |
| Data aborts that cannot be handled | Dynarmic single-step (the JIT's read-as-zero semantics); skip as a last resort |
| fail-fast | `RaiseFailFastException`, codes 0xE0CE0001, 0xE0CE0004, 0xE04E4352, 0xE04E4353 |

Source: `design\NCE_CLEANUP_REVIEW.md` sections 5.1 and 2.1.

#### 2.2.3 x18 and FPCR

On Windows ARM64, x18 is the platform register that points to the TEB. Guest instructions that use x18 are handled by `Patcher::PatchText` (`patcher.cpp`) in 9 stages.

| Stage | Target instructions | Method |
|---|---|---|
| a | CBZ/CBNZ/TBZ/TBNZ on x18 | Save x16 on the guest stack, look up the TEB slot, native branch |
| b | SBFM/BFM/UBFM with x18 | Save 2 scratch registers on the guest stack |
| c | LSE atomics with x18 | 0x130 save frame, call a C getter |
| d | LDXR/STXR pairs with x18, single exclusives | Same 0x130 frame and getter |
| e | Register substitution verified by the decoder | TEB slot spill (15 stages) |
| f | Remaining linear x18 instructions | Save context → host stack → Dynarmic single-step (`WindowsNceExecuteDirectX18`) |
| g | Other instructions (branches, SP changes) | BRK #0xF000 at load time, VEH redirect, Dynarmic single-step |
| h | MRS CNTFRQ/CNTPCT/CNTVCT/TPIDR into x18 | Special handling in the system register handler |
| i | Interruption caught inside stage a/b/e or an MRS helper | Recipe re-execution (`TryRecoverX18Substitution`) |

FPCR: the direct x18 path (stage f) runs with the host FPCR. `EDEN_NCE_X18_HOST_FPCR=1` is a diagnostic that audits that handling (`x18_host_fpcr.h`). The H gate also enters with the host's TEB stack bounds and FPCR (`nvn_gate.cpp` header comment).

In the 2026-11-22 DK measurement, x18 cost was not the bottleneck. The direct x18 stage ran 50,000-90,000 times per second at about 190 ns, which is 10-16 ms per second (memory note `dk-bananza-perf.md`).

Source: `design\NCE_CLEANUP_REVIEW.md` sections 3.1 and 4.1, `source\src\core\nvn\nvn_gate.cpp`.

#### 2.2.4 Exception return (tebfix → nceresume)

On the N1X, 0xd fast-fails (the SP check in `RtlRestoreContext`) kept happening. The causes and fixes went in this order.

| Build | Cause | Fix |
|---|---|---|
| tebfix, tebfix2 | The game faulted and resumed on a different stack (fiber), but the TEB bounds were for the old stack | Set the bounds to the allocation that contains SP |
| x18fix → tebfix3 | The bridge mistook the guest x18 for the TEB | Pass the target thread's TEB in x4 and `mov x18,x4` |
| tebfix4 | While delivering its own fault, another thread's break moved PC into the bridge | Include the bridge range in what gets fixed up |
| tebfix5 | A log inside the VEH raised an OutputDebugString exception | Unified into one rule that only looks at SP; removed logging from the VEH path |
| nceresume | Applied review items R1-R17 | Common `ResumeChecked` (rewinds PC in transition ranges; 0xE0CE0001 fail-fast if it cannot be fixed up), binding seqlock, etc. |

According to HISTORY, there were no NCE crashes after that, and the tebfix5 "left" counter is 0. What remains is the P7 per-stage injection test and the STP/exclusive re-execution test.

Source: `design\HISTORY_20261120.md` section 3, `design\ASTRA_NCE_RESUME_REVIEW_REQUEST.md`, `design\NCE_RESUME_REVIEW.md`, `design\ASTRA_NCE_RESUME_REVIEW.md`.

#### 2.2.5 patcher, loader order, IPS

- The NSO loader (`core\loader\nso.cpp`) splits the plan pass (`PatchText`) and the load pass.
- Before 2026-11-22, the plan pass had no program ID, so title patches selected by program ID went in after the NCE plan. The SVC/MRS/x18 instructions those patches added did not get NCE patches (the HWAoI "stubsvc" bug).
- From the 2026-11-22 release, title patches and IPS patches are applied before the NCE plan (`nso.cpp:202`, log `titleplan:`; with `NXN1X_TITLE_PATCH_PLAN=0`, the old order).
- In the same release, unused patcher wrappers were removed and CASP is retried. x18 LSE that cannot be planned is skipped with a warning. The self-test is enabled with `NXN1X_NCE_SELFTEST`.

Source: `docs\RELEASE_2026-11-22.md` section 2, `design\NCE_CLEANUP_REVIEW.md` section 1, `docs\NCE_CLEANUP_AUDIT_2026-11-22.md`.

#### 2.2.6 crash ring and crash reports

- **crash ring**: exceptions that reach the VEH are recorded once at entry and once at return into a 256-entry file-mapped ring `crashring.bin` (next to the exe). It is the only record a fast-fail 0xd leaves. It is included or excluded with the compile definition `NXN1X_CRASH_RING` (default 1). `crashring_dump` prints the contents (`arm_nce_windows.cpp:1325-1439`).
- **abort report**: on SIGABRT, CRT invalid-parameter, purecall, and `std::terminate`, `nxn1x_abort_<pid>.txt` is written next to the exe.
- **crash report**: `nxn1x_crash_<pid>.txt` (exception, module+offset addresses, x0-x30/sp/pc, frame-pointer chain) and `nxn1x_crash_<pid>.dmp` are written next to the exe (`host\dm\nxn1x\nvnh_rt.c:324`). From the `crashsym` build, NXN1X.exe frames get names from the PDB.
- llvm-symbolizer and llvm-objdump (clangarm64) can read NXN1X.exe offsets from the PDB.

Source: `source\src\core\arm\nce\arm_nce_windows.cpp`, `host\dm\nxn1x\nvnh_rt.c`, `BUILD_LOG.txt` (crashsym), memory note `dk-bananza-perf.md`.

#### 2.2.7 Workarounds and diagnostics left in the default path

`design\NCE_CLEANUP_REVIEW.md` (a read-only review, no changes) lists the items that run on the default path. The main items are:
- **s39 lock steal**: `LockContext` in generated code takes a lock that is still held after 2^23 tries. A workaround for the TotK hang (11-13), on by default (`DM_S39`).
- **BreakRescue**: a watchdog thread that polls every 5 ms. The 2026-11-22 cleanup (P2) decided to keep it on in every build (release: always; diagnostic builds: off with `EDEN_NCE_BREAK_RESCUE=0`). The comment was corrected; it logs `NCE_BREAK_RESCUE on` at start and a count every 2 s (`windows_cross_thread_break.h:176`).
- **HWAoI pool retry stub**: retries the allocator's slot 48 full-pool exit with `svc #0xb` (1 ms sleep) up to 128 times. Classified as a workaround whose cause was not found in the code (`hwa_pool.h`).
- **DK dkctl snapshot**: since the 2026-11-22 cleanup (P3) it is built only in `NXN1X_DIAGNOSTICS` builds.

Source: `design\NCE_CLEANUP_REVIEW.md` sections 1, 2, and 6.

### 2.3 NVN host layer

#### 2.3.1 Entry: hook and H gate

From the `core\nvn\nvn_gate.cpp` header comment:
1. **Hook**: the first pass of the NCE patcher replaces the first instruction of the `nvnBootstrapLoader` definition with a branch to the patch section. When the gate is off (before device handoff or after Refuse), the original code runs and boot goes to the Emulation path.
2. **Gate**: `nvnBootstrapLoader` returns a gate-block entry for each NVN function. The entry puts `&nvn_slot[i]` in x16 and enters the H gate. The H gate moves from the guest stack to that guest thread's host (fiber) stack and calls `nvn_slot[i].fn` with the host's TEB stack bounds and FPCR. x0-x8 and v0-v7 pass through unchanged.

NVN calls run as host functions on the guest thread. Example: `nvnSyncWait` is an S gate function and runs on core 0's host thread while that thread holds the core (memory note `dk-bananza-perf.md`).

#### 2.3.2 HLE and command recording (`host\dm\gfx\nvn\hle_nvn.c`)

From the `hle_nvn.c` header comment:
- It replaces the `nvnBootstrapLoader` that `main` imports. The sdk's NVN driver is not run.
- All 523 NVN names return a counting thunk (`nvn_thunks.S`).
  - Those with meaning (device, memory pools, textures and samplers, queues, windows, sync, events, command buffer memory) are implemented with NVN semantics and real objects are created.
  - Those with no CPU-visible result (state objects, bindings, draws, clears, etc.) are recorded as commands and replayed at submit.
  - The rest (307 that main loads but never calls, and those with no prototype in the header) go to `nvn_unimpl_entry`, which prints a failure and the name if called.
- GPU addresses equal CPU addresses. Memory pools use the real memory the game provides.
- Commands are recorded into the command memory the game provides and replayed at submit.

This code comes from the nxnative line (Metal replay on Mac/iOS). Metal terms in the comments, such as "zero-copy MTLBuffer", are traces of that.

Source: `host\dm\gfx\nvn\hle_nvn.c` header comment.

#### 2.3.3 Vulkan replay (`host\dm\windows\gfx_vk_*.c`)

This is the Android port's Vulkan backend moved to Windows.

| File | Role (per header comment) |
|---|---|
| `gfx_vk_core.c` | Device, memory pools, submission, completion. Submission uses a slot ring (command pool, descriptor pool, scratch, objects freed after the GPU is past them). A submission signals a fence on a timeline semaphore, and the completion thread applies CPU-visible results (event writes, counter reports) |
| `gfx_vk_exec.c` | Command replay (`mtl_exec`). State records update the NVN state shadow, and draw/dispatch become pipeline, descriptor set, and draw. A render pass opens at the first draw/clear after SetRenderTargets and closes before anything that is not a draw |
| `gfx_vk_eden.inc` | Draw/dispatch in Eden mode (`vk_eden_mode`, default). Section 2.3.4 |
| `gfx_vk_tex.c` | Formats, textures, views, samplers, ID tables. Textures with the same pool, offset, and description share one image. Packaged textures are deswizzled on the CPU and filled through an upload command buffer, and are refilled when dirty tracking sees a write |
| `gfx_vk_shader.c` | Shaders for the bundle path (the host's own translator). Not used in Eden mode |
| `gfx_vk_present.c` | present. In NXN1X, `mtl_present` passes each present to `core\nvn\nvn_output.cpp` |
| `gfx_vk_win_glue.c` | Connects Android port names to the current host interface |
| `gfx_vk_fg_auto.inc`, `gfx_vk_fg_capture.inc`, `gfx_vk_fg_math.h`, `nvn_fg_input.h` | Detection and capture of DLSS-G inputs (camera, depth, motion) |

replay threads:
- By default, recording happens on the game thread (`DM_VK_ASYNC=0`, `DM_VK_PARALLEL=0`).
- `DM_VK_ASYNC=1` (s46) records on a separate replay thread.
- `DM_VK_PARALLEL=<n>` is parallel recording (section 5.15).
- DK uses ASYNC=1, PARALLEL=4 as title defaults in `nvnh_rt.c` (`dkpar` build).

Source: each file's header comment, memory note `dk-bananza-perf.md`, `BUILD_LOG.txt` (dkpar+optdiag).

#### 2.3.4 Eden pipeline connection (`source\src\video_core\nvn\`)

In Eden mode, for each draw `gfx_vk_eden.inc` uses the Eden `GraphicsPipeline` that the Maxwell path would build for the same shaders and state (`nvn_eden_graphics_pipeline`).
- Set 0 is filled in Eden order (per-stage uniform, storage, texel, storage texel, texture, image) through an update template.
- Push constants and dynamic state also use Eden's.
- Replay-specific rules that change the image are kept: viewport flip (`set_dynamic`) and the winding that follows from it, and the Bayonetta 3 depth EQUAL relaxation (r57).

Bridge files:

| File | Role |
|---|---|
| `nvn_shader_bridge.{h,cpp}` | C interface that translates NVN shaders with the Eden shader recompiler. One NVN shader is one Eden environment, and it uses the same hash (CityHash64) as the Maxwell path so both paths find the same entry in `vulkan.bin` |
| `nvn_pipeline_key.{h,cpp}` | NVN fixed-function state → Eden `FixedPipelineState`. Render target format mapping |
| `nvn_shader_env.{h,cpp}` | NVN shader environment. Connects NVN2 Ampere code lowering |
| `renderer_vulkan\nvn_bridge_device.cpp` | Hands Eden's Vulkan device, scheduler, and pipeline cache to the host. Shares external synchronization of the graphics queue (`Scheduler::submit_mutex`) with the host |

Benefits of the decision to switch to the Eden-style shader path: Switch 1 stability, pipeline cache reuse, and easier comparison with other Eden forks (Lichii, nxnative PR).

Source: `host\dm\windows\gfx_vk_eden.inc`, `source\src\video_core\nvn\nvn_shader_bridge.h`, `source\src\video_core\renderer_vulkan\nvn_bridge_device.cpp`, memory note (2026-11-21, feedback note on the decision to switch to Eden-style shaders).

### 2.4 Shader translation

#### 2.4.1 Ampere → Maxwell lowering (Switch 2)

`source\src\shader_recompiler\frontend\ampere\lower_to_maxwell.h`:
- Only operations whose meaning the existing Maxwell frontend can express are lowered. It creates new instructions instead of reinterpreting the original bytes. The environment adapter attaches the schedule/control words.
- Options: geometry, tensor (compact matrix), tessellation stages, verified trap list.
- Uniform registers UR28 and above are mapped to reserved registers below R216.
- Encoding reference: Mesa NAK `sm70_encode.rs` (MIT).
- Call sites: `video_core\native_lowering.cpp` `LowerNativeProgram`, `video_core\shader_environment.cpp:667`.

Items confirmed in the FE porting review (already in the code): scaled shared memory addresses, IMAD.HI pairs and IMAD.X carry, NVN2 workgroup count c0+4, SDK 22 bindless TEX formats 0x361/0xf60/0x370.

HMMA: guest HMMA goes through the Ampere translator. The existence of a cooperative-matrix emitter (`emit_spirv_tensor.cpp`) is not evidence that hardware Tensor Cores are used. Nothing in the current shader profile turns on the cooperative HMMA flag.

Source: `lower_to_maxwell.h`, `docs\RELEASE_2026-11-22.md` section 7.6, `docs\DLSS_ARM64_ARCHITECTURE.md`.

#### 2.4.2 handle program and handle table

- When a bindless texture handle must be computed for each draw, it becomes a "handle program" (`shader_recompiler\environment.h` `handle_program`, `ir_opt\texture_pass.cpp`). The host fills the value with `nvn_eden_eval_handle` (including `HandleOp::Index`).
- `htarr` (2026-11-22): `HostTranslateInfo::support_handle_table_arrays` was never set, so the per-invocation handle table path was dead code. It is turned on for NVN host devices that support non-uniform sampled image array indexing (`vk_pipeline_cache.cpp:727`; off with `NXN1X_HANDLE_TABLE_ARRAYS=0`).
- `cidx` (2026-11-22): an index computed by the shader (an operand multiplied by the table stride) is also accepted as an element index. The index is clamped to `HANDLE_TABLE_SIZE` 64 (`texture_pass.cpp:881`; off with `NXN1X_HANDLE_CIDX=0`).
- The diagnostic `handletree` (`NXN1X_HANDLETREE=1`) logs the expression of bindless handles that cannot be traced.

Source: `docs\RELEASE_2026-11-22.md` sections 7.2 and 8.

#### 2.4.3 Other shader handling

- `nanguard`: in the float color outputs of every fragment program, NaN is replaced with 0 and values are clamped to ±65000 (port of Longshot GuardFragmentOutputs, GPL-3.0). Off with `DM_VK_NAN_GUARD=0`.
- `EDEN_MAXE` (the work array of set 0 entries for one pipeline) was raised from 1024 to 4096 (`maxe4k`). 1024 was the limit for one indexed texture array (`texture_pass.cpp:99 DESCRIPTOR_MAX_COUNT`) but was wrongly used as the limit for the whole pipeline.
- Async shaders: skipping draws whose pipeline was still building caused missing text glyphs. So draws without a depth test, and compute, wait for compilation (`glyphwait`).
- Pipeline cache: `vulkan.bin`, `vulkan_pipelines.bin` (version 20). The driver cache (61 MB) is saved during play when builds have been quiet for 20 seconds, and at least every 5 minutes (`pcache1`; with `NXN1X_PCACHE_PERIODIC=1`, every 30 seconds as before).

Source: `BUILD_LOG.txt` (nanguard), `docs\RELEASE_2026-11-22.md` section 7.1, `design\HISTORY_20261120.md` section 4, `design\OOT_BARRIER_DESIGN.md` section 16.

### 2.5 Memory

#### 2.5.1 The N1X driver's import memory problem

NXN1X imports game memory pools with `VK_EXT_external_memory_host` (zero-copy, GPU address = CPU address). For pools whose import the driver refuses, a GPU copy is updated on every submission (`gfx_vk_core.c` header comment). Import memory performance measured on N1X driver 621.24:

| Access | import / device-local | Source |
|---|---|---|
| Fixed-function vertex/index fetch | 100-250x slower (membench), 76-94x (membench3 conditions) | membench, membench3 |
| atomic | about 25x | membench5 |
| Repeated reads of the same address | about 36x | membench5 |
| SSBO store | about 1.8x | membench5 |
| SSBO/UBO/texel reads, shader load | same | membench3/4 |
| Copy (import → device-local) | 125-190 GB/s | membench |
| texel store, buffer→image, indirect args, xfb, query copy | 1.0-1.1x | membench5 |
| attachment writes, transfer dst | not measured | `GPU_MEMORY_HYBRID_DESIGN.md` section 2 |

- Only driver-private (non-exportable) allocations are fast. Every method that makes memory CPU-visible at an address we choose (OPAQUE_WIN32 section, importing our own section, host-pointer import variants) has slow fixed-function fetch.
- Large pages could not be tested (no SeLockMemoryPrivilege).
- The idea of using driver-allocated HOST_VISIBLE memory as the guest original was judged "blocked in the current structure". The reasons are the game's right to map the same pages at a second address, and the pointer contracts of `SetHeapSize`, `MapView`, and `nvnMemoryPoolBuilderSetStorage`.

Source: memory note `n1x-imported-vertex-fetch.md`, `design\GPU_MEMORY_CACHE_DESIGN.md` section 1, `design\GPU_MEMORY_HYBRID_DESIGN.md` section 1, `design\ASTRA_HOST_VISIBLE_STATIC_REVIEW.md`.

#### 2.5.2 Countermeasures

| Method | Switch | Description | Status |
|---|---|---|---|
| vbmir (vertex mirror) | off with `NXN1X_VB_MIRROR=0` | Persistent device-local mirror per pool chunk, 64 KB block tracking. Re-copies only on CPU dirty (nvn_dirty) or GPU writes (sub_gpu_wrote) | On by default. HWAoI 16 → 60 fps |
| vbvol | (part of vbmir) | Narrows protection to 4 KB pages and adds volatile pages. Faults 0-200/s | Default path |
| Vertex pulling | `NXN1X_VTX_PULL=1` | The VS reads directly from the import pool through a fetch table (BDA); indices are mirrored | Stage B implemented, default OFF |
| copy mode | `NXN1X_POOL_MODE=copy` | Device-local copy of every pool, CPU changes uploaded on every submission, GPU changes merged | Implemented, not default |
| hybrid | `NXN1X_POOL_MODE=hybrid` | Copies only chunks that atomics touch | hybrid1/2 runs were slow |
| hy3 (atomic staging) | `DM_VK_HY3=0/1/2` (default 2) | Moves only the bindings of atomic dispatches to device-local staging, and the GPU merges them | Implemented, effect confirmed in OoT |
| fstage | off with `DM_VK_FSTAGE=0` | Stages SSBOs written by fragment shaders per draw | Implemented, confirmed on the DK title |
| import (default) | `NXN1X_POOL_MODE=import` | Existing zero-copy | Default |

Current structural decision (2026-11-20 23:49): **"mixed = vertex pulling + import + atomic exception path"** is the long-term default path. The premise is that the driver may not be fixed.

copy mode makes the GPU side faster, but the CPU write tracking cost is high.
- Bayonetta 3 depth pass: 40-57 ms → 0.6 ms.
- DK gameplay: 8 fps in copy mode. It reads and compares about 19.5 GB over 120 submissions. So copy mode is not the solution for DK.

Source: memory notes `n1x-imported-vertex-fetch.md`, `bayo3-depth-pass-import-writes.md`, `dk-bananza-perf.md`, `design\GPU_MEMORY_HYBRID_DESIGN.md` section 7.0, `design\ATOMIC_STAGING_DESIGN.md` section 8.2, `design\ASTRA_PHASE_B_REVIEW_REQUEST.md`.

#### 2.5.3 Write tracking (nvn_dirty) and its pitfalls

- `host\dm\gfx\nvn\nvn_dirty.c` protects pages as read-only to detect CPU writes and leaves a stamp.
- Pitfall: the core's `memory.cpp` `MapRegion`/`ProtectRegion`, uncached/debug-off, and remapping after a partial unmap in HostMemory changed watched pages to RW without nvn_dirty knowing. That broke vertices in copy mode.
- Fix `vfix`: right after such a path changes them, it calls `NvnLayer::BeforeGuestWrite`.
- For stale-GPU-data bugs, first compare the tracker state with Windows protection using `DM_VK_PC_VERIFY` (`RUN_PCVERIFY_COPY.cmd`).

Texture ownership rules:
- `gwrange`: whether the GPU wrote is judged per texture byte range, not per pool (`gfx_vk_tex.c` `storage_gpu_written`; off with `DM_VK_GPUW_RANGE=0`).
- `gpuown`: a registered texture that the CPU rewrote is treated as GPU-owned only when the GPU wrote it after the CPU's last write (`hle_nvn.c` `nvn_ft_poll`; off with `DM_NVN_GPU_OWN_AFTER_CPU=0`). It causes the OoT Deku Tree holes, so the launch scripts turn it off (section 5.1).
- `relovl`: fixed GPU-only images being kept forever under newer textures (47,000 images and 86 GB in 11 minutes). Off with `DM_VK_KEPT_OVERWRITE_DROP=0`.

Source: memory notes `core-writable-bypasses-nvn-dirty.md`, `oot-terrain-gwrange.md`, `oot-deku-terrain-gpuown.md`, `oot-speed-relovl.md`.

#### 2.5.4 Command memory

- The cause of StarFox aborting at about 115 seconds was the 1 GB upper limit on NVN command memory. The game provides memory only through a callback, and host segments lent before the callback answered were never reclaimed, so they piled up to 65,489.
- Fix: segments are tied to the chunk the callback answered with and reclaimed on reuse (combined1120). Normal up to 274 seconds, steady at 773 segments.

Source: `design\HISTORY_20261120.md` section 4.

### 2.6 Output: compositor, DLSS, frame generation

#### 2.6.1 present path

- The host's `mtl_present` (`gfx_vk_present.c`) does not create its own window or swapchain. It passes each present to `core\nvn\nvn_output.cpp`.
- On each vsync, `HardwareComposer::ComposeLocked` composites the present whose turn has come as the game layer.
- The renderer copies the image (`renderer_vulkan\nvn_display.cpp`) and returns it when that copy finishes on the GPU. Presents are returned in order.
- The host runs `vkQueueSubmit` and present on the same graphics queue under `Scheduler::submit_mutex`.

Source: `source\src\core\nvn\nvn_output.cpp`, `nvn_display.cpp`, `nvn_bridge_device.cpp` header comments.

#### 2.6.2 DLSS SR (Streamline)

- One Streamline 2.14.16 Vulkan interposer is used. The SR plugin evaluates NGX. The NGX DLSS/DLSS-G DLLs are 310.9.16.0. Six ARM64 DLLs are placed in `streamline\` next to the exe.
- Code: `renderer_vulkan\nvn_streamline.{h,cpp}` (built with `NXN1X_STREAMLINE_SDK`).
- 2026-11-19 probe: SR offscreen output passed. But synchronization validation reported 2 WRITE_AFTER_WRITE errors inside `nv.ngx.dlss.Evaluate`, so the document at that time said "production integration blocked".

DLSS in NVN2 games (`OP_DLSS`):
- `nvnCommandBufferEvaluateDlss` in `hle_nvn.c` records 6 texture roles, sizes, motion scale, jitter, pre-exposure, and history reset.
- The answer order of `nvnDlssNetworkBuilderGetOptimalSettings` is optimal, **smallest**, **largest** (not the NGX order). The range is output/3..output (`dlssrange`, `hle_nvn.c:3567`; the old order with `DM_NVN_DLSS_RANGE=0`).

OoT (no `OP_DLSS`):
- The game's own upscale network is replaced with DLSS SR (`DM_VK_OOT_DLSS`). Section 6.1.

Settings and hotkeys:
- "NVIDIA DLSS Super Resolution" under Graphics Advanced, F7 (`dlsstoggle`).

**Mismatch**: `docs\DLSS_ARM64_ARCHITECTURE.md` says "SR and FG are off in the game" and "The game executable has no SDK DLL dependency". But later builds (ootdlss7 to 11, dlsstoggle) and the memory notes record that DLSS SR and DLSS-G work in game. This architecture document should be read as a record from the time of the initial audit (its first paragraph also says it "describes the initial audit").

Source: `docs\DLSS_ARM64_ARCHITECTURE.md`, `docs\DLSS_ARM64_CAPABILITIES.md`, `docs\RELEASE_2026-11-22.md` section 1, `BUILD_LOG.txt`, memory notes `oot-dlss-plan.md`, `xcde-dlss-range-order.md`.

#### 2.6.3 DLSS-G and optical-flow frame interpolation

- **DLSS-G**: the swapchain interposer receives the final composited color. Depth and motion are tagged in the producing command buffer, and camera constants and tags are bound to the original present ticket. If inputs are missing, stale, or ambiguous, it does not turn on.
- **Automatic input detection**: finds a rigid view, perspective projection, and view-projection in the bound vertex UBOs. It compares 16 depth/motion samples with camera reprojection, and after three matches it fixes the RG attachment and its units and sign (`gfx_vk_fg_auto.inc`, `gfx_vk_fg_math.h`).
- **Optical-flow interpolation**: original NVN tickets without DLSS inputs use optical interpolation (port of `fsr3_frame_gen` from Lichii commit 60ee2ef). It interpolates the composited color including the HUD.
  - Fractional target planning: for example, 43 original fps + 17 generated = 60.
  - `fgfill`: optical interpolation fills in when DLSS-G does not generate.
  - `optfill`: presents without an NVN ticket also get their own original number (bit 62) (off with `NXN1X_OPTFILL_ANY=0`). This is the fix that made interpolation work in DK.
- DLSS-G only supports integer multiples. With a 60 Hz target and about 53 original fps, it does not generate. A 120/Auto target is needed.

Settings and hotkeys:
- Frame Generation under Graphics Advanced (Off / DLSS Frame Generation ×2 / LSFG ×2), "Frame Generation source" (Auto / DLSS-G only / optical only, can be changed while running), F10 (Toggle Frame Generation).
- The overlay FG row counts DLSS-G, optical, and original frames over a 0.5-second window (`fgoverlay`).

**Mismatch**: the choices change even within `docs\DLSS_ARM64_TEST_REPORT.md`. An earlier section lists "OFF, Auto, 30, 60, 90 or 120 FPS" with LSFG disabled; a later section lists "Off, NVIDIA DLSS Frame Generation ×2, or LSFG ×2". Which one is the current UI needs a check of the source UI files (unconfirmed).

Source: `docs\DLSS_ARM64_TEST_REPORT.md`, `design\OOT_DLSS_DESIGN.md` section 1, `BUILD_LOG.txt` (ootdlss10, fgoverlay, fgsource, optfill), memory notes `oot-dlss-plan.md`, `dk-bananza-perf.md`.

#### 2.6.4 Other output handling

- `opthold`: frame generation waits until the size and format of the NVN output copy have held for 8 original frames (off with `NXN1X_OPT_HOLD=0`).
- `viewusage`: when frame generation is on, frame images get STORAGE usage. Sampled/attachment views state their own usage with `VkImageViewUsageCreateInfo` (off with `NXN1X_VIEW_USAGE=0`).
- Swapchain recreation waits for idle under `submit_mutex` (`DM_VK_SWAPCHAIN_WAIT`).
- GPU checkpoints (`DM_VK_CHECKPOINTS`) report on the first device lost.
- `BlitScreen::WaitIdle` takes the NVN submit mutex after `scheduler.Finish()` (response to a race seen by the validation layer).

Source: `docs\RELEASE_2026-11-22.md` section 7.5, memory note `oot-dlss-plan.md`, `docs\DLSS_ARM64_TEST_REPORT.md`.

### 2.7 Audio and video

- NXN1X's NVN host library does not build `dm/host` (`source\src\nvn_host\CMakeLists.txt`). So audio and in-game video appear to be handled by Eden's HLE services. However, there is no NXN1X-specific audio/video design document (unconfirmed).
- `host\dm\windows\movie_stub_win.c` (AVC shows a black screen, AV1 has no output) is for the standalone host Windows port. It is not in the `nvn_host` build list.
- `host\docs\reference\audio.md` and `movies.md` are nxnative documents (standalone host process, Mac/iOS). They do not apply to NXN1X as is.
- `source\CMakeLists.txt` has Eden's `ENABLE_CUBEB` (ON) and `YUZU_USE_BUNDLED_FFMPEG` options.

Source: `source\src\nvn_host\CMakeLists.txt`, `host\dm\windows\movie_stub_win.c`, `host\docs\reference\README.md`, `source\CMakeLists.txt`.

---

## 3. Game file formats

### 3.1 Supported formats and detection order

Loaders in `source\src\core\loader\`: NSP, XCI, NCA, NAX, NSO, NRO, KIP, NUP, DeconstructedRomDirectory (dump folder).
- Detection is `IdentifyFile` in `loader.cpp`. From 2026-11-22, dump folder detection runs before NSO detection (`loader.cpp:106`). So `exefs/main` boots as a dump folder, not as a single NSO.
- NSP/XCI/NCA use Eden's original loaders. Keys (prod.keys) are needed.

Source: `source\src\core\loader\loader.cpp`, `docs\RELEASE_2026-11-22.md` section 5.

### 3.2 NUP (NX Package, SPEC v1)

Per `source\src\core\file_sys\nup_container.h`:
- One file holds the ExeFS/RomFS trees of the base program, updates, and DLC as they are.
- Structure:
  - Superblock 0x80 bytes, magic `"NUP\x01"`
  - TOC: packages, directory/file nodes, children arrays, string pool
  - Blob area: per-file extents, raw or zstd, 16-byte aligned
- Package kinds: `Base`=0, `Update`=1, `Addon`=2. There is a `standalone_update` flag.
- Per-package meta TLV: NPDM, title name (default + localized), publisher, source dump, icon, display version.
- Integers are little-endian. The zstd decode cache is 512 MB by default, and members larger than half of that are unpacked to a temporary file.
- Files are not encrypted, so no keys are needed.
- Loader: `core\loader\nup.cpp` (includes mapping of the 16 NACP language slots).

Containers without a base: if there is no base record (standalone_update on all updates), or there is a base record with no files (title ID 0), the chosen update is used as the whole game. The flag does not guarantee that (`pack_nup.py` header comment).

SPEC original: https://github.com/a55uka/nup (SPEC.md v1, the link in the `pack_nup.py` comment).

### 3.3 make_nup.py: dump folder → NUP

`host\scripts\make_nup.py`:

```
python host\scripts\make_nup.py <dump dir> <out.nup> --title-id <16 hex> --version <n> --name "<name>" [--base] [--display-version 1.2.0] [--resume]
```

| Argument | Meaning |
|---|---|
| `<dump dir>` | A folder with `exefs/` + `romfs/` |
| `--title-id`, `--version`, `--name` | Required |
| `--base` | Make a base package. The default is a standalone update (kind 1, flag 1) |
| `--display-version` | Display version |
| `--resume` | Reuse the files a previous run wrote to `<out>.part` |

- Files are stored raw (codec 0) with a SHA-256 per file. zstd compression is not used.
- The meta block holds NPDM, the title name, the original dump label, and the display version.
- Data is written first (including hashes), and the TOC and superblock last. Windows paths longer than 260 characters are handled.
- There are 32 read threads, because most files are small.
- Real case: Fire Emblem Fortune's Weave (v131072), 29.1 GB, 252,693 files, every SHA-256 verified.

`host\scripts\pack_nup.py` goes the other way. For nxnative's IPA build script, it unpacks a NUP into the extracted layout (ExeFS, RomFS image, NACP, icon, DLC images). It needs the zstd module.

Source: `host\scripts\make_nup.py`, `host\scripts\pack_nup.py`, `docs\RELEASE_2026-11-22.md` section 6.

### 3.4 Dump folders (`exefs/` + `romfs/`)

Items fixed in the 2026-11-22 release:

| Name | Location | Description | Switch / log |
|---|---|---|---|
| `dirfs` | `core\loader\deconstructed_rom_directory.cpp:296` | Booting a loose `exefs/main` had no FS controller registration, so `fsp SetCurrentProcess` failed with 2002-1002 and the game exited on its own (FE endless loading). Now RomFS is registered from a `.romfs` file in `exefs/` or a `romfs/` folder next to it | `NXN1X_DIRFS=0`, `dirfs:` |
| `maxopen` | `core\file_sys\vfs\vfs_real.cpp:43` | The VFS keeps up to 8192 files open with CRT `FILE*`, but the CRT limit was 2048. A folder with 250,000 RomFS files hit "Too many open files". The CRT limit was raised to 8192, and other uses stay at 1024 | `NXN1X_VFS_MAXOPEN=<n>`, `maxopen:` |
| `fsfail` | `fs_i_filesystem.cpp:22`, `fsp_srv.cpp:275` | Logs the paths of failed guest FS calls as Warning, the first 64 (because LOG_DEBUG is dropped in release) | `NXN1X_FS_FAIL_LOG=0`, on by default |

Source: `docs\RELEASE_2026-11-22.md` section 5.

---

## 4. Design document summaries

There are 26 documents in the `design\` folder. Most were written on 2026-11-20 to 21. "Review request (REQUEST)" and "review (REVIEW)" come in pairs. "Astra" in the documents is an external reviewer; a REVIEW without a name is an internal review.

### 4.1 HISTORY_20261120.md

- **Scope**: record of N1X performance and crash work on 2026-11-19 to 20.
- **Content**:
  - Measuring slow import memory, and vbmir/vbvol
  - copy mode stages 1 and 2 (poolcopy → stage2a/2c/2b → combined1120)
  - NCE 0xd crash (tebfix to nceresume)
  - StarFox command memory abort, exit crash, missing glyphs (glyphwait)
- **Conclusion**: NCE crash solved. HWAoI copy mode is 52-57 fps while moving (import + vbmir holds 60). The next direction is hybrid. Open items: endless loading (no sv1 dump), T15 (CPU rewriting old values while the GPU writes, a known limit), NCE P7 injection test.

### 4.2 GPU_MEMORY_CACHE_DESIGN.md (copy mode stage 1)

- **Problem**: on the N1X, GPU access to import memory is slow (section 2.5.1).
- **Method**: on NVIDIA, all pools run as driver-allocated device-local copies (copied pools).
  - CPU writes: incremental 4 KB uploads based on nvn_dirty
  - GPU writes: written back by `gw[]` ranges
  - Per-driver policy: copies on NVIDIA, import on Adreno/Mac
- **Status**: copy location A (device-local) decided. Implemented result: Bayonetta 3 depth pass 0.6 ms. HWAoI is 22-24 fps (CPU cost), so stage 2 became necessary.

### 4.3 ASTRA_GPU_MEMORY_CACHE_REVIEW_REQUEST.md / ASTRA_GPU_MEMORY_CACHE_REVIEW.md / GPU_MEMORY_CACHE_REVIEW.md

- **Request**: barrier order, gaps in the `g_done` and `nvnSyncWait` structure, risk of misses in deferred uploads of binding ranges, amount of write-back, scope of Eden BufferCache reuse.
- **Astra verdict**: "The direction of plan A is sound. Implement after completing the correctness contract."
  - P0: fix the default write-back path to a separate readback staging
  - P0: separate 4 KB dirty from ownership of the newest data
  - P0: make every alias and access path use the same copy
- **Internal verdict**: "Proceed with a comparison implementation of (A); hold off on switching the default."
  - P1 R9: a direct GPU copy from device-local → import overwrites CPU changes
  - P1 R1/R2: misses for writes of unknown size and when the raw global write log saturates
- Both reviews state that they looked only at the design documents, without the project source or logs.

### 4.4 GPU_MEMORY_CACHE_STAGE2_DESIGN.md / ASTRA_STAGE2_REVIEW_REQUEST.md / ASTRA_GPU_MEMORY_CACHE_STAGE2_REVIEW.md / GPU_MEMORY_CACHE_STAGE2_REVIEW.md

- **Problem**: in copy stage 1, HWAoI CPU cost was high. Waiting for the previous submission 8-10 ms/frame, comparison 6-7 ms, write exceptions 16,000/s.
- **Original plan**: remove serialization by changing the baseline from "the copy at submission start" to "a first-write snapshot per epoch S[e]". 4 KB dirty comparison, volatile pages (enter after 3 consecutive writes, return after 8 without writes).
- **Review**: both reviews found counterexamples to the original plan's merge rule. Example: if the GPU writes X and the CPU writes Y, the result should be (2,8) but (1,8) comes out. They said the 4 KB comparison can go ahead separately.
- **Applied (section 8)**: changed to remove serialization with GPU-side pre/post readback and a "mirror of G" shadow.
- **Result (HISTORY)**:
  - stage2c: HWAoI wait 12 → 0 ms, 27 → 37-57 fps
  - fault-ahead (on sequential faults, release P+1..P+15 at once): 52-57 fps
  - self-test: case 0-12, T1-T17, V1-V7, FA1-FA3

### 4.5 GPU_MEMORY_HYBRID_DESIGN.md / ASTRA_GPU_MEMORY_HYBRID_REVIEW_REQUEST.md

- **Problem**: copy makes the GPU fast but has a CPU write tracking cost (HWAoI while moving: faults 20-35k/s, mprotect 30-50k/s).
- **Method**: per backing range (a 256 MB-aligned shared copy or a private chunk), copy only the ranges that have slow GPU writes (shader store, atomic), and leave the rest as import. One-way switch; turned off automatically if the driver becomes fast.
- **Structural decision (7.0)**: mixed = vertex pulling + import + atomic exception path.
- **Result**:
  - hybrid1 (atomic chunk switch): the 256 MB block closure rule made 452 chunks (493 MB) into copies, giving 4-5 fps.
  - hybrid2 (exact ranges, read-only excluded, A/B with `NXN1X_SSBO_WRITES=all`): 55 chunks, still slower than import + ootfadd.
- **Status**: premise validation (section 1.1 counters) remained "unverified". The atomic exception path was replaced by hy3 in section 4.6.

### 4.6 ATOMIC_STAGING_DESIGN.md (hybrid3, hy3)

- **Problem**: in hybrid1/2, the cost of the CPU comparing and merging the GPU results of atomic bindings on every submission (replay pool-sync 17.5 s, copyback 15.8 s).
- **Method**: game memory stays import. Only during atomic dispatches are the binding ranges moved to device-local staging (S) and a snapshot (P), and the GPU writes them back with a 3-way merge (`hy3_merge.comp`). The CPU is not involved. The starting point is the existing TotK-only `n1x_atomic`.
- **Stage 2 (window grouping)**: consecutive dispatches in one submission are grouped into a window, which closes before any GPU work that is not a dispatch. Binding limit 16 MB, window total 64 MB. `DM_VK_HY3=0` off / `1` stage 1 / `2` default.
- **Result (stage2s1, OoT)**:
  - Self-test HYBRID 93/93, COPY 73/73 PASS
  - dispatch → dispatch gap 46 µs → 18.9 µs
  - fps 27-34
  - Next bottleneck is the part boundary 4.7 ms (the GPU waits for the CPU's next part)
- **Extension**: `fstage`, widened to fragment shader storage writes in DK (section 6.2).

### 4.7 VERTEX_PULLING_DESIGN.md / ASTRA_REVIEW_REQUEST.md / ASTRA_VERTEX_PULLING_REVIEW.md / VERTEX_PULLING_REVIEW.md / ASTRA_PHASE_B_REVIEW_REQUEST.md

- **Problem**: the vbmir mirror uses CPU write detection (page protection and faults).
- **Method**: the VS reads vertex attributes directly from the import pool via BDA instead of fixed-function input, and indices stay in a device-local mirror.
- **Review**: both reviews said "direction sound; the current design cannot be applied as default as is." Main accepted points:
  - Put only proven contiguous ranges into the table
  - Branch after the check, then load (do not mask with select)
  - Barriers for the change in consuming stage
  - Drop the comparison that runs game draws twice
  - push constant offset 48, range 56
- **Stage A (membench4)**: exhaustive format comparison, 0 mismatches (excluding NaN payloads). Pulling + device-local IB runs at the same speed as the current mirror (13.8 vs 12.1 µs, 29.1 vs 29.7 µs).
- **Stage B**: implemented only on the Eden path. Only direct draws that meet the conditions are covered, default OFF (`NXN1X_VTX_PULL=1`). Changed files include `runtime_info.h`, `emit_spirv.h`, `spirv_emit_context`, `emit_spirv_context_get_set.cpp`, `fixed_pipeline_state.h`, `vk_vertex_pulling.h`.
- **Status**: confirmed up to the stage B review request. No result document for the game comparison (stage C) was found (unconfirmed).

### 4.8 ASTRA_HOST_VISIBLE_STATIC_REVIEW.md

- **Question**: can driver-allocated HOST_VISIBLE memory be used as the guest original (an alternative that avoids slow import)?
- **Verdict**: for all game memory, "blocked in the current structure." The basis is the address contracts set by `KPageTableBase::SetHeapSize`, `HostMemory::Impl::MapView`, and `nvnMemoryPoolBuilderSetStorage`. The game keeps the right to map the same pages at a second address. Plain `vkMapMemory` has no contract for creating a second view at a given address.

### 4.9 ASTRA_NCE_RESUME_REVIEW_REQUEST.md / ASTRA_NCE_RESUME_REVIEW.md / NCE_RESUME_REVIEW.md

- **Problem**: is the NCE exception/redirect/resume design, fixed five times in one day (tebfix2 → tebfix5), complete?
- **Premises (W1-W5)**: the SP acceptance rule of `RtlRestoreContext`, the VEH running on the guest stack, `NtContinue`, `SetThreadContext`, etc.
- **Verdicts**:
  - Astra: "tebfix5 is an improvement but does not prove the whole design complete." Keep the bridge, and make the bounds selection policy and final resume state check common. Priority 1 is the match of target thread, TEB, and GuestContext; the fact that `CONTEXT_FULL` does not include x18; and reconstruction rules per transition stage.
  - Internal: "Keep the bridge." P1 is the branch that resumes as is even when `FitTebToSp` fails, and exceptions during the handler prologue.
- **Result**: the nceresume build applied R1-R17 (section 2.2.4).

### 4.10 NCE_CLEANUP_REVIEW.md

- **Scope**: read-only review of NCE and its connecting code (no build or run).
- **Content**:
  - 13 per-title hooks (HWAoI, XC2/BF2, MKW, DK, OoT)
  - Diagnostics and workarounds in the default path
  - Duplication across the 9 x18 stages
  - Switch list
  - Platform-dependent parts
  - Cleanup proposals P1-P6
- **Findings**:
  - Structural risk that the loader plan pass has no program ID
  - Mismatch between the BreakRescue comment and code
  - 0x0400E95004038001 is called "XC2 3.0.1" by the NCE code and "BF2" by `k_system_control.cpp`
- **Follow-up**: `docs\NCE_CLEANUP_AUDIT_2026-11-22.md` narrowed the first deletion targets to 9 unused private overloads in the patcher (Windows ARM64 syntax check 18/18 passed; linking and game runs not included). The cleanup went into the 2026-11-22 release (section 2.2.5).

### 4.11 OCCLUSION_REDESIGN.md

- **Problem**: haze, jitter, and flashing in OoT Kokiri. The cause was substitute occlusion values taken from the previous frame (occkey, occaddr, occland, fixed 4096).
- **References**: HeyListen (GPL-3.0, mechanism only), NX2navi, Longshot, Eden query cache.
- **Method (`DM_VK_OCC2`)**:
  - s1: a query slot per draw per stream serial (64K ring)
  - s2: GPU resolve and landing in stream order, substitute values removed
  - s3: GPU-side render condition evaluation with `VK_EXT_conditional_rendering`
  - s4/s5: worker occlusion segments, adoption (not started)
- **Result**: in s3, overexposure and flashing disappeared (user confirmed 2026-11-21, A/B with `RUN_OCC_S0`). Cost is 31-40 fps vs about 47 fps for the previous path. `occ-s3fast` (reusing Eden `queries_prefix_scan_sum`, merging condition blocks) is in progress.
- **Status**: the default is still 0. It will be changed after an fps A/B and a Switch 1 / Zelda regression check.

### 4.12 OOT_BARRIER_DESIGN.md

- **Problem**: in OoT, 13,265 full barriers per second. Two fulls go into one gap.
- **Cause**: OoT records a guest barrier per dispatch, and OP_BARRIER bumps `g_bar_seq`, which breaks r89's compute-only barrier condition.
- **Fixes**:
  - `cbar1` (off with `DM_VK_CBAR=0`): condition for skipping guest barriers between dispatches
  - `cbar2` (off with `DM_VK_CBAR2=0`): merging consecutive barriers
  - Diagnostics tsdiag/cpudiag
- **Later sections**:
  - s46 async replay A/B: game-thread submit+present goes from 17-20 → 0.3-0.5 ms, but fps is the same
  - s46pc: the replay thread TIME_CRITICAL + P-core policy caused a regression, so s46pc2 changed it to ABOVE_NORMAL
  - `nvnTextureFinalize` deferral (off with `DM_VK_FIN_DEFER=0`)
  - pcache1 (the 30-second cache save problem)
- **Remaining**: the s48 range import 130-180 ms spikes and the 2-3 s scene entry were not touched.

### 4.13 OOT_DLSS_DESIGN.md

- **Problem**: OoT does not use NVN DLSS commands; it upscales 536 → 1440 with its own upscale network (a335c1db → d99fbeeb). Optical-flow FG interpolated the screen including the HUD, and ghosting was suspected.
- **Method**: reuses the following from NX2navi's (a static recompilation of the same game, MIT) `vk_dlaa.c` and `dlaa_mv.comp`.
  - Camera block, frame shape, upscaler location
  - jitter (Halton 2,3), camera-based MV, quality ratio
  - Stages: D1a survey → D1b SR → D1c jitter → D2 DLSS-G
- **Result (memory note)**: in ootdlss7, DLSS SR replaced the game network at every DRS size (1138x640..1920x1080 → 2560x1440). User comment: "DLSS가 훨씬 좋네" ("DLSS is much better"). Later: ootdlss9 to 11 (NaN guard, srfit, output stall detection), resfix (fixed scene ratio 0.5), nanguard, expbound/flashdiag.
- **Mismatch**: the design document gives the camera as "c3의 0x3500바이트 scene context" ("the 0x3500-byte scene context in c3") (per NX2navi). The memory note says the implementation uses "vertex UBO0 >=0x230 (Longshot layout: VP 0x30, proj 0x70, size 0x120)" with runtime validation. It seems to have changed in implementation after the design, but the design document was not updated.
- **Mismatch**: section 3 of the design document assumes a separate build per stage (D1a, D1b, ...). The "one build" principle in `CLAUDE.md` was set on 2026-11-21, after this design.

### 4.14 PARALLEL_REPLAY_DESIGN.md

- **Problem**: one replay thread records all NVN lists serially (OoT about 15 ms/frame, about 2,300 draws). The GPU idles about 6 ms waiting for parts.
- **Method**: follows the HeyListen model.
  - The replay thread walks the stream once in order and makes all order-dependent decisions (dry pass: reports, events, render conditions, uniform update lists, occlusion, render target marking, part splitting).
  - Draw batches are cut into segments, and worker threads re-execute them from an NVN state snapshot into their own primary command buffers.
  - Chunks are submitted in stream order.
  - In stage 1, all operations that write GPU memory stay on the replay thread.
- **Stages**: R1-R8 refactor, S1a (worker base), S1b (dry pass, `DM_VK_PARALLEL_DRY=1`, `DM_VK_PARALLEL_SERIALSEG=1`), S1c (thread safety), S1d, S1e (worker recording, `DM_VK_PARALLEL=<n>`).
- **Result**:
  - OoT S1b/S1d: SERIALSEG mismatch 0, identical image
  - DK: `uppool` (per-slot upload command pool, off with `DM_VK_UPPOOL=0`) fixed the parallel recording crash (2026-11-22). With PARALLEL=4, about 85% of draws go to workers, and 20 → 26-30 fps
- **Mismatch**: the document header says "NVN1 titles first (Zelda Ocarina of Time is the target)". But `OOT_DLSS_DESIGN.md` and the memory notes treat OoT as an NVN2 (Switch 2) title.
- **Mismatch**: the document status line says S1e is "syntax-checked only, not run", but the memory notes have later game run results. The document was not updated.

---

## 5. Per-game status

Status is as of 2026-11-22. "Confirmed" means the user checked it in game; "log" means only the logs were read.

### 5.1 Zelda: Ocarina of Time (Switch 2, 0400486024DEC000)

| Item | Status | Basis |
|---|---|---|
| Kokiri terrain "water"/sky reflection | Fixed (gwrange), confirmed | `oot-terrain-gwrange.md` |
| haze, jitter, flashing | Fixed with `DM_VK_OCC2=s3`, confirmed. Default is still 0 | `oot-occlusion-s3.md` |
| Slow motion below 60 fps | `ootspeed` (step patch at main+0x6e6cc4, off with `DM_REALTIME_PATCH=0`), confirmed | `oot-speed-relovl.md` |
| Slowdown over long sessions | `relovl`, stays at 2-3 GB in a 16-minute run, confirmed | same |
| DLSS SR | Works since ootdlss7, confirmed. Switch `DM_VK_OOT_DLSS` | `oot-dlss-plan.md` |
| Full-screen white flash | OoT work for this week ends after the nanguard + flashdiag build. expbound was removed later (flashdiag2) | `oot-wrapup-friday.md`, `BUILD_LOG.txt` |
| Deku Tree floor holes / ghost terrain | Unresolved. Cause is gpuown. The launch scripts (`RUN_PAR2`, `RUN_PAR4`, `RUN_DK*`) set `DM_NVN_GPU_OWN_AFTER_CPU=0`. The shader that turns transparent is FS c72b6ea746c00433 | `oot-deku-terrain-gpuown.md` |
| Striped textures | Postponed to next week. The F3 stripediag dump is in the build | `oot-wrapup-friday.md` |
| Stutter right after start | About 1,800 Eden pipelines are created on first use (30-90 s, fps 32-53). Prewarm planned | `pipeline-prewarm-todo.md` |
| device lost | Recorded 3 times (once during an area change), cause not determined | `oot-dlss-plan.md` |
| ootdlss9 run "unplayable" | Frame intervals are even (45-49 fps), cause unconfirmed | `oot-dlss-plan.md` |

Note `oot-wrapup-friday.md`: OoT wraps up this week. Additional bugs and fix cases from other projects will be handed over on 2026-11-27 (Fri).

### 5.2 Donkey Kong Bananza (040074A01BF12000)

| Item | Status | Basis |
|---|---|---|
| "GPU 208 ms" | A misread figure. That line was one long frame, and the GPU column is the sum of submission timestamp intervals, so it includes CPU replay time | `dk-bananza-perf.md` |
| Resolution | Per-game setting 1X (resolution_setup=3), global is 2X | same |
| Gameplay bottleneck | Replay CPU. Most of the main thread's 50 ms is `nvnSyncWait` (waiting on a GPU fence). Not x18 (EDEN_NCE_STATS) | same |
| Title screen 20 fps | One 1440x810 R32F full-screen draw writes storage into the import pool, taking 38-40 ms. With `fstage`, title screen 60 fps, GPU 0.7 ms | same |
| copy mode | Title screen 60 fps but gameplay 8 fps. Not the solution | same |
| Parallel recording | Crash fixed with uppool, confirmed (about 16.8K presents, no device lost). 26-30 fps with PARALLEL=4 (about 20 before). The replay thread is still 21-23 ms/frame | same |
| Defaults | DK uses ASYNC=1, PARALLEL=4 (`nvnh_rt.c`) | `BUILD_LOG.txt` dkpar |
| Optical interpolation | Works with `optfill`, confirmed | `dk-bananza-perf.md` |
| DLSS SR (instead of agl_ndsr) | Next week | same |
| Next week's task | Waiting at replay part boundaries (about 6 part ends per frame, 3.4 ms wait) | same |
| Built-in game NDSR | The original guest compute is the default. `DM_VK_NDSR=0` gives GSR1. A purple screen was reported at one point; after a storage view split fix it is recorded as waiting for in-game confirmation | `DLSS_ARM64_TEST_REPORT.md` |

The 11-19 measurement in `DLSS_ARM64_TEST_REPORT.md` (2X, 3.1-5.5 fps, median frame 213 ms) is old compared with later builds.

### 5.3 Xenoblade Chronicles DE / Xenoblade Chronicles 2 (Switch 2 Edition)

- **XCDE (0400FF500E34A001)**: the title background was black and the intro froze (it moved for only about 1 second at the day/night branch).
  - Cause: the answer order of DLSS GetOptimalSettings. The game treats arguments 3/4 as the lower bound and 5/6 as the upper bound, and evaluates DLSS only when the input is in range (main+0x7742c0, main+0x7745e0). The title drew at 1280x720, so DLSS was skipped, and the next compute pass stretched an unwritten 1920x1080 to 4K.
  - Fix `dlssrange`. Confirmed 2026-11-22.
- **XC2**: same symptom, and also no 3D. The same fix applies (`RELEASE_2026-11-22.md` section 1). No record of an in-game XC2 check was found (unconfirmed).
- 0x0400E95004038001: reserves a dedicated 64 GiB NX2 address region (`k_system_control.cpp:133-136`). The name of this ID differs inside the code: "XC2 3.0.1" and "BF2".
- Waiting for regression check: XCDE impact of htarr/cidx/copymirror.

Source: `docs\RELEASE_2026-11-22.md` sections 1 and 9, memory note `xcde-dlss-range-order.md`, `design\NCE_CLEANUP_REVIEW.md` section 1.

### 5.4 Mario Kart World (0400C3F00006E000)

- What is in the source:
  - 30 FPS / dynamic FPS patches (`mkw_30fps`, `mkw_dynamic_fps` settings, `nso.cpp:195-282`)
  - Double-free probe (`NXN1X_NX2_PROBE`, off by default, marked "Temporary")
- The FPS patch UI controls were removed from the Graphics Advanced panel. The setting data is kept.
- **Mismatch (minor)**: `nx2_probe_trap.h:41` statically initializes a 65,536-entry ring in every process even when the probe is off (review estimate about 3.5 MiB).
- Current game status record: only that it is on the htarr/cidx/copymirror regression check waiting list for the 2026-11-22 release. No other N1X run record was found (unconfirmed).
- `docs\nvn-memory\MKW_REFERENCE.txt` is comparison material on the memory design of the mkw-recomp project. It is not a game status document.

Source: `design\NCE_CLEANUP_REVIEW.md` section 1, `docs\DLSS_ARM64_TEST_REPORT.md`, `docs\RELEASE_2026-11-22.md` section 9.

### 5.5 Hyrule Warriors: Age of Imprisonment (04003F40232DE000)

- The cause of 16 fps was vertex fetch from import memory. With vbmir: 60 fps, GPU p50 10 ms, about 6 MB copied per frame, 15 mirrors at 1.27 GB.
- copy mode gives no GPU gain (vertices are already mirrored). 52-57 fps while moving.
- Unresolved crash: 0xc0000409 (fast-fail 0xd, ntdll+0x34a10).
  - Once during the opening video, not reproduced, cause not proven
  - The same form once in Bayonetta 3 on relaunch after a forced exit
- E-core placement once caused a loading hang (`host-tuned-for-adreno.md`).
- NCE has HWAoI-only workarounds: pool retry stub, diagnostic BRK, pooldiag (section 2.2.7).
- The user deliberately removed the `hyrule_30fps` / `hyrule_dynamic_fps` options (root-fix principle). However, the global frame-interval detection in `nso.cpp:284-341` (the frame pacing feature) remains.

Source: memory notes `n1x-imported-vertex-fetch.md`, `bayo3-depth-pass-import-writes.md`, `root-fix-principle.md`, `design\HISTORY_20261120.md`, `design\NCE_CLEANUP_REVIEW.md`.

### 5.6 Fire Emblem Fortune's Weave (04007EF01F8AE000, v131072)

Fixed in the 2026-11-22 release:

| Name | Symptom → fix | Switch |
|---|---|---|
| dirfs, maxopen | Endless loading from a dump folder → FS registration, CRT file limit | `NXN1X_DIRFS=0`, `NXN1X_VFS_MAXOPEN` |
| maxe4k | Draws skipped for a pipeline with 1035 descriptors → `EDEN_MAXE=4096` | none (constant) |
| htarr + cidx | 32 lighting texture handles that could not be computed → 0 | `NXN1X_HANDLE_TABLE_ARRAYS=0`, `NXN1X_HANDLE_CIDX=0` |
| copymirror | Character cards flipped upside down → `MIRROR_X/Y` copies done with `vkCmdBlitImage`, confirmed | `NXN1X_COPY_MIRROR=0` |
| progforget | A finalized program stayed in the queue binding (port of nxnative 1440a73) | `DM_VK_PROG_FORGET=0` |
| opthold, viewusage | Frame generation related changes (device lost not fixed) | `NXN1X_OPT_HOLD=0`, `NXN1X_VIEW_USAGE=0` |

Open issues:
1. The tutorial rectangle is darker than on the console. Scene luminance is 137.25, which matches nxnative's console-matched value of 134.4, so exposure stage 1 is right. The 3x1 exposure image has not been captured yet.
2. With frame generation on from the start, device lost at about 3 seconds (tick 6). Starting with it off works. Turning it on with F10 while running does not help. The next step is NV checkpoints on the compositor queue.

Source: `docs\RELEASE_2026-11-22.md` sections 5-9, memory note `fe-fortunes-weave.md`.

### 5.7 Bayonetta 3 (01004A4010FEA000)

- In 2X import mode, GPU 25-55 ms. Of that, one depth-only pass (1600x896, 500-850 draws) is 40-57 ms. This is because the fragment shader writes `ssbo[objectId] = 65535` into import pool memory for every pixel.
- In copy mode, that pass is 0.50-0.68 ms, GPU p50 7.4-9.0 ms, frame 18-20 ms. The copy mode vertex corruption was fixed with vfix, confirmed.
- copy mode is not the default. Stage 2 was the precondition for switching the default.
- fstage has a limit of 16 draws per shader per present, so the Bayo3 depth pass (500+ draws per pass) hits the limit. A per-pass window is a follow-up task.
- ID mismatch in the records: the memory note says 01004A4010FEA000, and `NCE_CLEANUP_REVIEW.md` mentions "Bayonetta 3 0400E88025B1A000" based on a comment. The documents do not say whether the two IDs are different editions (Switch 1 edition / Switch 2 Edition) (unconfirmed).

Source: memory notes `bayo3-depth-pass-import-writes.md`, `core-writable-bypasses-nvn-dirty.md`, `dk-bananza-perf.md`, `design\NCE_CLEANUP_REVIEW.md` section 1.

### 5.8 Switch 1 regression titles

- On the 2026-11-21 build, the user said Switch 1 games run well. The original wording was "젤다" ("Zelda") 1/2 and Bayonetta 1/2.
- Zelda 1 appears not to use occlusion queries (the control group for occlusion changes). Zelda 2 appears to use them.
- Other titles that use them: HWAoI, DK, Pokémon Z-A (on hold).
- Principle: a default flip that touches the NVN1 path (for example `DM_VK_OCC2=s3`) is done after an A/B on OoT and one Switch 1 title (Bayonetta 2 or Zelda).
- **Mismatch**: the memory notes do not say which titles "Zelda 1/2" specifically are (BotW/TotK, etc.) (unconfirmed).
- Other recorded titles:
  - StarFox: command memory abort fixed. A one-off `vk::Exception(MasterSemaphore::Wait)` has an undetermined cause.
  - TotK (0100F2C0115B6000): s39 lock steal workaround, `n1x_atomic`, s56 GPU→guest write-back (`DM_VK_S56`). On 11-19, FG detection observed 60 cameras and 0 motion candidates, and stayed at 30 fps.

Source: memory note `regression-games.md`, `design\HISTORY_20261120.md` section 4, `docs\DLSS_ARM64_TEST_REPORT.md`, `design\NCE_CLEANUP_REVIEW.md`.

---

## 6. Settings, switches, logs

### 6.1 Data and log locations

| Item | Location |
|---|---|
| User data | Portable mode if a `user` folder exists next to the exe; otherwise `%APPDATA%\NXN1X` |
| Game log | `%APPDATA%\NXN1X\log\nxn1x_log.txt` (the previous run is `.old.txt`). In portable mode, `user\log\nxn1x_log.txt` |
| crash ring | `crashring.bin` next to the exe. The previous run's ring is renamed and kept at startup |
| Crash reports | `nxn1x_crash_<pid>.txt`, `.dmp`, `nxn1x_abort_<pid>.txt` next to the exe |
| Pipeline cache | Eden `vulkan.bin`, `vulkan_pipelines.bin`, driver cache |
| DLSS-G profiles | `streamline/profiles/<16-digit lowercase hex program ID>.fg` (no verified game profiles ship in the release) |
| Launch script log copies | `RUN_*.cmd` copies the log to `logs_poolcopy\` after the run |

- Saves, keys, firmware, and game files are not included in the release.
- Diagnostic logs can affect game timing.

Source: `source\README.md`, memory note `nxn1x-build.md`, `arm_nce_windows.cpp`, `nvnh_rt.c`, `docs\DLSS_ARM64_TEST_REPORT.md`, `RUN_DK.cmd`.

### 6.2 DM_CTL: switches while running

- When run with `DM_CTL=<file>`, lines from that file are read while running (`gfx_vk_exec.c:775`). `env K=V` lines change environment switches live. Example: `DM_VK_RC_OCC=0` and `NXN1X_VTX_PULL=1` take effect from the next present.
- `DM_CTL_FROM_END=1` reads from the end of the file (`RUN_DK.cmd`).
- `ctl.txt` in the repository is the control file for the launch scripts.
- The `DM_CTL` description in `host\docs\env_switches.tsv` lists "(Mac)" and the button, `shot`, `passdump`, `texdump`, `env KEY=VALUE`, `peek`, `memdump`, and `drag` commands. The document does not say which of these commands work on Windows/NXN1X (unconfirmed). The `dlsstoggle` build added a "DM_CTL line-count reader".

Source: memory note `nxn1x-build.md`, `RUN_DK.cmd`, `host\docs\env_switches.tsv`, `BUILD_LOG.txt`, `design\ASTRA_PHASE_B_REVIEW_REQUEST.md`.

### 6.3 Main environment switches (NXN1X)

Below are the switches that appear in the 2026-11-22 release document, the design documents, and the memory notes. In principle, switches are for A/B measurement and for reverting on regression; they are not solutions (root-fix principle).

**Memory pools**

| Switch | Default | Meaning |
|---|---|---|
| `NXN1X_POOL_MODE` | import | `import` / `copy` / `hybrid` |
| `NXN1X_VB_MIRROR` | 1 | `0` turns vbmir off |
| `NXN1X_VTX_PULL` | 0 | Vertex pulling stage B |
| `NXN1X_SSBO_WRITES` | — | `all` treats every SSBO as written (A/B) |
| `DM_VK_HY3` | 2 | atomic staging: 0 off, 1 stage 1, 2 window grouping |
| `DM_VK_FSTAGE` | on | fragment storage staging |
| `DM_VK_GPUW_RANGE` | on | gwrange |
| `DM_NVN_GPU_OWN_AFTER_CPU` | on (code) | gpuown. The launch scripts use 0 |
| `DM_VK_KEPT_OVERWRITE_DROP` | on | relovl |
| `DM_VK_S56` | TotK only | GPU → guest write-back |
| `DM_VK_PC_VERIFY` | off | Compare the copy mode tracker with Windows protection |

**replay and synchronization**

| Switch | Default | Meaning |
|---|---|---|
| `DM_VK_ASYNC` | 0 (DK 1) | Async replay (s46) |
| `DM_VK_PARALLEL` | 0 (DK 4) | Number of parallel recording workers |
| `DM_VK_PARALLEL_WKTEST` | — | For the first N presents, re-record worker chunks on the replay thread |
| `DM_VK_UPPOOL` | on | Per-slot upload command pool |
| `DM_VK_CBAR`, `DM_VK_CBAR2` | on | Barrier skipping and merging |
| `DM_VK_FIN_DEFER` | on | Texture finalize deferral |
| `DM_VK_S46_POLICY` | — | replay thread policy diagnostic (0-3) |
| `NXN1X_THREADCPU` | — | Thread CPU diagnostic |
| `DM_VK_PROG_FORGET` | on | progforget |

**occlusion and rendering**

| Switch | Default | Meaning |
|---|---|---|
| `DM_VK_OCC2` | 0 | `s1`/`s2`/`s3` new occlusion path |
| `DM_VK_OCCDIAG` | off | occlusion instrumentation |
| `DM_VK_RC_OCC` | — | `0` makes every occlusion value unknown (→ draw) |
| `DM_VK_NAN_GUARD` | on | NaN handling in fragment output |
| `NXN1X_COPY_MIRROR` | on | Mirror copies |
| `NXN1X_HANDLE_TABLE_ARRAYS`, `NXN1X_HANDLE_CIDX` | on | handle table |
| `DM_VK_TBUF_FRESH` | NVN2 1 | Recreate the texel buffer copy after GPU writes |

**DLSS and frame generation**

| Switch | Default | Meaning |
|---|---|---|
| `DM_NVN_DLSS_RANGE` | on | dlssrange |
| `DM_VK_OOT_DLSS` | — | OoT DLSS SR |
| `DM_VK_OOT_RESFIX` | on | OoT fixed scene ratio |
| `DM_VK_NDSR` | original | `0` gives GSR1 |
| `NXN1X_OPT_HOLD`, `NXN1X_VIEW_USAGE`, `NXN1X_OPTFILL_ANY` | on | Frame generation helpers |
| `DM_VK_SWAPCHAIN_WAIT`, `DM_VK_CHECKPOINTS` | — | Swapchain wait, GPU checkpoints |

**CPU, loader, FS, resolution**

| Switch | Default | Meaning |
|---|---|---|
| `DM_REALTIME_PATCH` | on | DK/OoT realtime step patch |
| `NXN1X_TITLE_PATCH_PLAN` | on | Title/IPS patches before the NCE plan |
| `NXN1X_NCE_SELFTEST` | off | Resume rule self-test |
| `EDEN_NCE_STATS` | off | NCE counters per exit kind, top trap PCs |
| `NXN1X_NS2_RES_1X` | on | Switch 2 default 1x |
| `NXN1X_DIRFS`, `NXN1X_FS_FAIL_LOG` | on | Dump folder FS, FS failure log |
| `NXN1X_VFS_MAXOPEN` | — | Set the number of open files directly |
| `NXN1X_PCACHE_PERIODIC` | off | Cache save every 30 seconds (old behavior) |

**Mismatch**: the header of `RELEASE_2026-11-22.md` says "Each change has one switch (environment variable)". But the crash ring's `NXN1X_CRASH_RING` is a compile definition, not an environment variable (`-DNXN1X_CRASH_RING=0`, `arm_nce_windows.cpp:1326`). `maxe4k` has no switch to turn it off.

Source: `docs\RELEASE_2026-11-22.md`, `design\*.md`, all memory notes, `source\src\core\nvn\nvn_layer.cpp`, `source\src\common\settings.cpp`.

### 6.4 env_switches.tsv

- `host\docs\env_switches.tsv` has 362 entries (diagnostic 154, switch 122, path 54, player 23, test 9).
- This table is the switch list of nxnative (the Mac/iOS Metal host). Most entries are Metal-, iOS-, or Mac-only. Examples: `DM_NVN_TBDR`, `DM_NVN_ENC_THREAD`, `DM_PRESENT_LINK`, `DM_SHOW_FPS`.
- It has almost none of the `DM_VK_*` and `NXN1X_*` switches that NXN1X uses. `DM_VK_CAPTURE` and `DM_VK_LOG` are also for the Vulkan-on-Metal layer.

Diagnostics that appear usable in the common NVN layer (`hle_nvn.c`):
- `DM_NVN_TSV` (per-frame NVN call table, used for the XCDE investigation)
- `DM_NVN_COUNT`, `DM_NVN_FRAMES`, `DM_NVN_PASSDUMP`, `DM_NVN_CMDSTAT`, `DM_NVN_CBLOG`, `DM_NVN_WAITS`

However, the table does not say whether each of them works on the Windows Vulkan backend (unconfirmed).

Source: `host\docs\env_switches.tsv`.

### 6.5 Diagnostic tools

| Tool | Usage | Description |
|---|---|---|
| pass times | `DM_VK_PASS_TIMES=1` | r138 "GPU busy" line, time per pass. When on, all S1e parallel segments are refused |
| ootgap / longframe | (log) | Breaks GPU span into timed passes + part boundaries + gaps |
| dispatch trace | `DM_VK_TRACE_DISPATCH` | Dispatch tracing (XCDE investigation) |
| image watch | `DM_VK_WATCH` | Watches image writes (overlap-aware) |
| NVN gate trace | `NXN1X_NVN_GATE_TRACE=1` | Guest code dump. Disassemble it to confirm what a call means |
| NVN call time | `NXN1X_S51=1`, `NXN1X_S49=1` | Splits the main thread into host NVN calls and guest code |
| NCE statistics | `EDEN_NCE_STATS=1` | Counts per exit kind and top trap PCs (`windows_nce_stats.h`) |
| gpuck | (build default) | On device lost, reports the pass and draw the GPU was on |
| hangstk | (first sv1 "no present" dump) | PC/LR/frame chain of every host thread, names from the PDB |
| sv1 | (NVN path) | Dumps guest threads when present stops (log only) |
| cpudiag / thrname | (diagnostic build) | CPU per thread, nnsdk thread names |
| crash ring | default | Section 2.2.6 |
| fsfail | on by default | Failed guest FS paths |
| handletree | `NXN1X_HANDLETREE=1` | Expressions of bindless handles that cannot be computed |
| fmtdiag | `NXN1X_FMTDIAG=1` | NVN/Vulkan formats of textures sampled by fragments |
| expoprobe | `NXN1X_EXPOPROBE=1` | Luminance and exposure values, every 2 seconds |
| stripediag | F3 | Dumps image info for the last 3 presents, re-uploads packaged textures |
| FG input capture | Ctrl+Shift+F11 (`tools\test_ndsr.ps1 -FrameGeneration -CaptureFGInputs`) | 32x32 samples over 2 frames, 4096 bytes of UBO |
| flashdiag(2) | (OoT) | Sudden brightness changes in DLSS input and output |
| copy self-test | `RUN_POOLCOPY_SELFTEST_COPY.cmd` | HYBRID / COPY self-test |
| DLSS probe | `tools\dlss_probe.cpp` | Streamline feature probe, SR offscreen readback |

The same principle applies to diagnostics. Do not ask for separate A/B runs that the logs can answer. Verification is done with runtime checks in the build, detection logs, and automatic fallback.

Source: `docs\RELEASE_2026-11-22.md` section 8, memory notes `dk-bananza-perf.md`, `xcde-dlss-range-order.md`, `feedback-no-busywork.md`, `BUILD_LOG.txt`, `docs\DLSS_ARM64_TEST_REPORT.md`.

### 6.6 Hotkeys and UI

| Key / location | Action |
|---|---|
| F7 | Toggle DLSS SR |
| F10 | Toggle frame generation (if not on at start, there is no STORAGE usage, so it does not help the FE device lost) |
| F3 | stripediag texture dump |
| Ctrl+Shift+F11 | FG input capture (diagnostic run) |
| View → Show Performance Overlay | FPS, frame time, CPU%, GPU%, DLSS/FG rows |
| Graphics Advanced | DLSS SR, Frame Generation, Frame Generation source, HDR checkbox |

- The HDR checkbox uses the existing HDR policy. The presenter still prefers an SDR swapchain, so HDR10 output is not guaranteed.
- On the NVN path, controls that only apply to the emulated GPU (texture decoding/recompression, anisotropy, buffer cache, GPU unswizzle) are hidden.

**Mismatch (overlay meaning)**: the "GPU" value in the overlay and the s67 log (`mtl_gpu_ms`) is the sum of submission timestamp intervals. It includes GPU waits between parts, so actual GPU work should be read from r138 "GPU busy" (`DM_VK_PASS_TIMES=1`). Meanwhile, `source\README.md` defines GPU% as "가장 바쁜 Windows GPU 엔진 값" ("the value of the busiest Windows GPU engine"). The two are different metrics.

Source: `BUILD_LOG.txt` (dlsstoggle, stripediag), memory notes `oot-dlss-plan.md`, `dk-bananza-perf.md`, `docs\DLSS_ARM64_TEST_REPORT.md`, `source\README.md`.

---

## 7. Build

### 7.1 Current method

```powershell
$env:PATH = "C:\msys64\clangarm64\bin;C:\msys64\usr\bin;$env:PATH"
ninja -C build/release                      # or: ninja -C build/release bin/NXN1X.exe
```

- Output: `build\release\bin\NXN1X.exe`.
- Toolchain: MSYS2 clangarm64 (a syntax check was recorded with Clang 23.1.5), CMake/Ninja.
- The build machine has no git. Originals of changes are kept in `_backup\<tag>\`, and each build leaves one line in `BUILD_LOG.txt`.
- The exe cannot be replaced while NXN1X is running. Build when NXN1X is closed. The same folder is shared with other work threads, so do not build over someone else's build.
- CMake options:
  - `NXN1X_DIAGNOSTICS` (default OFF): investigation traces and detailed cost counters
  - `NXN1X_CRASH_RING` (compile definition, default 1)
  - `NXN1X_STREAMLINE_SDK`: Streamline integration
  - `ENABLE_LSFG`: there is a record of the optical-flow build being built with it OFF
- The NVN host is built from `NVN_HOST_DM` (default `../host/dm`). If `hle_nvn.c` is missing, configuration stops. It generates the game profile header (`game_profile.h`).
- Streamline: unpack the official ARM64 SDK 2.14.16 to `build/sl-sdk`, and put the 6 runtime DLLs in `build\release\bin\streamline\`. Keep the SDK's license file with them.

Source: memory notes `nxn1x-build.md`, `root-fix-principle.md`, `dk-bananza-perf.md`, `docs\DLSS_ARM64_TEST_REPORT.md`, `docs\DLSS_ARM64_CAPABILITIES.md`, `docs\NCE_CLEANUP_AUDIT_2026-11-22.md`, `source\CMakeLists.txt`, `source\src\nvn_host\CMakeLists.txt`, `design\HISTORY_20261120.md` section 6.

### 7.2 Working rules (`CLAUDE.md`)

- Do not split work into survey builds and step-by-step builds. When earlier implementations (refer, Eden, nxnative, Longshot, NX2navi) or logs already show the cause or the method, put the diagnostics and the change in one build.
- Guard the risky parts within that one build in three ways.
  - Checks while running
  - On a failed check, fall back to the original path and log the reason
  - One switch to turn the change off
- Other operating principles (HISTORY section 6, memory notes):
  - Fix only root causes. Do not use workarounds such as 30 fps modes, retries, sleeps, or stubs as solutions.
  - Fix after confirming the cause with evidence.
  - For problems another project (nxnative, BananaNative, etc.) has already fixed, map and port that fix first.

Source: `CLAUDE.md`, `design\HISTORY_20261120.md` section 6, memory notes `root-fix-principle.md`, `feedback-one-build.md`, `feedback-port-known-fixes.md`.

### 7.3 Old build tools (Adreno period)

- `tools\build_nx2.py`: CMake/Ninja + MSYS2 ARM64 (root `README.md`).
- `tools\build_turnip_f1r.py`, `build_turnip_f2r.py`: Turnip/Mesa incremental builds. They depend on objects from another donor build, so they are not clean build commands.
- **(Resolved)** The old `source\README.md` gave a missing `tools/build.ps1` (`-Diagnostic`) and two exes (release and diagnostic) as the build. The 2.0 README rewrite states the build as `ninja -C build/release`.

Source: `README.md`, `source\README.md`, `tools\` listing.

---

## 8. Known problems and next tasks

### 8.1 Known problems

| # | Problem | Status | Basis |
|---|---|---|---|
| 1 | The FE tutorial rectangle is darker than on the console | Exposure image capture needed | `RELEASE_2026-11-22.md` section 9 |
| 2 | In FE, device lost when frame generation is on from the start | Unresolved. Next: NV checkpoints on the compositor queue | same |
| 3 | Impact of htarr/cidx/copymirror on other Switch 2 games (MKW, DK, XCDE) | Waiting for regression check | same |
| 4 | OoT Deku Tree floor holes (gpuown) | Deferred. gpuown off in scripts | `oot-deku-terrain-gpuown.md` |
| 5 | OoT striped textures | Next week. Waiting for the F3 dump | `oot-wrapup-friday.md` |
| 6 | OoT occlusion s3 cost (31-40 vs about 47 fps) | occ-s3fast in progress. Default 0 | `oot-occlusion-s3.md` |
| 7 | OoT device lost, 3 times | Cause not determined | `oot-dlss-plan.md` |
| 8 | Pipeline creation stutter after start (OoT about 1,800) | Prewarm planned | `pipeline-prewarm-todo.md` |
| 9 | DK gameplay replay CPU bottleneck (replay thread 21-23 ms) | Part boundary wait is next week's task | `dk-bananza-perf.md` |
| 10 | 0xc0000409 fast-fail 0xd (once in the HWAoI opening, once on a Bayo3 relaunch) | Cause not proven | `n1x-imported-vertex-fetch.md` |
| 11 | StarFox one-off `vk::Exception(MasterSemaphore::Wait)` | Diagnostics added, cause not determined | `HISTORY_20261120.md` |
| 12 | Endless loading | No log with an sv1 dump yet | same |
| 13 | T15: CPU rewriting old values while a GPU write is in progress | Documented as a known limit | same |
| 14 | s48 range import 130-180 ms spikes, scene entry 2-3 s | Not touched | `OOT_BARRIER_DESIGN.md` section 16 |
| 15 | 2 DLSS SR synchronization validation errors (inside NGX Evaluate) | Recorded in the 11-19 probe. No later re-validation record | `DLSS_ARM64_CAPABILITIES.md` |
| 16 | BreakRescue is always on in release | Resolved (P2, 11-22): on in every build, comment and log corrected | `NCE_CLEANUP_REVIEW.md` |
| 17 | The HDR checkbox does not guarantee HDR10 output | — | `DLSS_ARM64_TEST_REPORT.md` |
| 18 | No Vulkan validation layer on the laptop | Synchronization validation limited | `dk-bananza-perf.md` |

### 8.2 Next tasks

1. DK: reduce the replay part boundary wait, and put DLSS SR in place of agl_ndsr (next week).
2. OoT: receive additional bugs and fix cases from other projects on 2026-11-27. Remaining: stripes, Deku terrain (gpuown; check a write-back like TotK s56), occ-s3fast, switching the s3 default (after an OoT + Switch 1 A/B).
3. Pipeline prewarm: reuse nxnative's `prewarm_start` and the Eden `vulkan.bin` list (do not build a new mechanism).
4. Memory: continue the mixed structure (vertex pulling stages C/D, a per-pass window for fstage (Bayo3), deciding the vertex pulling default).
5. Vertex mirror improvement: only the range a draw reads, GPU copy before submission, lifetime kept until the fence. The memory note still lists it as "waiting for approval".
6. NCE: P7 injection test, STP/exclusive re-execution test, cleanup proposals P2-P6.
7. Re-check Adreno-era defaults: title keys in `nvnh_rt.c`, NVN2 defaults in `nvn_layer.cpp` (`TU_GS_ONCE`, etc.), Turnip-conditional code, fp16 rules.
8. FE: exposure image capture, compositor queue checkpoints for the FG device lost.

### 8.3 Mismatches and gaps between documents

| # | Content | Related files |
|---|---|---|
| 1 | (Resolved) The old `source\README.md` and root `README.md` described the Adreno period (NPU frame generation, a missing `tools/build.ps1`, two exes). Both were rewritten for 2.0 | `README.md`, `source\README.md` |
| 2 | The root `README.md` describes a snapshot with Turnip/Mesa and LSFG. The document does not say how it relates to the current N1X build | `README.md` |
| 3 | `host\docs\` (development, playing, settings, frontends, env_switches.tsv, reference) are nxnative (iOS/Mac, Metal) documents. There is no user documentation for NXN1X on Windows | `host\docs\*` |
| 4 | `NVN_HOST_BRIDGE.md`, `docs/lane_b/SHADER_EDEN_B.md`, `docs/lane_a/A1_BUILD.md`, and `docs/lane_a/apply_sv1.py`, which code comments point to, are not in the repository | `nvn_gate.cpp`, `nvn_shader_bridge.h`, `nvnh_rt.c`, `sv1.cpp` |
| 5 | "SR and FG are off in the game" in `DLSS_ARM64_ARCHITECTURE.md` does not match the later implementation (a record of the initial audit) | Section 2.6.2 |
| 6 | Within `DLSS_ARM64_TEST_REPORT.md`, the FG setting choices are written in two ways | Section 2.6.3 |
| 7 | The camera location in `OOT_DLSS_DESIGN.md` (c3 0x3500) differs from the implementation in the memory note (UBO0 ≥0x230, Longshot layout) | Section 4.13 |
| 8 | `PARALLEL_REPLAY_DESIGN.md` calls OoT an NVN1 title; other documents call it NVN2. Its status line was also not updated with run results | Section 4.14 |
| 9 | The header and section 6 title of `OCCLUSION_REDESIGN.md` say s2/s3 are "syntax-checked only, not run", but the memory notes have an in-game s3 confirmation | Section 4.11 |
| 10 | 0x0400E95004038001 is called "XC2 3.0.1" and "BF2" in different places in the code | `arm_nce_windows.cpp`, `k_system_control.cpp` |
| 11 | The Bayonetta 3 ID differs: 01004A4010FEA000 (memory) and 0400E88025B1A000 (comment basis in the NCE review). The documents do not say whether these are different editions of the same game | Section 5.7 |
| 12 | (Resolved) The BreakRescue comment ("off by default") differed from the code. Corrected on 11-22 to "on in every build" | `windows_cross_thread_break.h:176` |
| 13 | The RELEASE document says "an environment variable switch for every change", but `NXN1X_CRASH_RING` is a compile definition and `maxe4k` has no switch | Section 6.3 |
| 14 | `BUILD_LOG.txt` ends at optfill (11-22 around 08:40). The 11-22 release items (dlssrange, NCE cleanup, dirfs, maxe4k, htarr, etc.) are missing | `BUILD_LOG.txt` |
| 15 | The GPU value has two meanings: overlay/s67 "GPU" (sum of submission intervals, includes CPU replay) and the README's GPU% (Windows GPU engine) | Section 6.6 |
| 16 | The code default of `nce_gpu_path` is Emulation. No document says whether the NVN path is the default in the release profile | Section 1.3 |
| 17 | No results were found for vertex pulling stage C (game comparison), an in-game XC2 check, or MKW N1X run records | Sections 4.7, 5.3, 5.4 |
| 18 | There is no NXN1X-specific audio/video document (presumed to be Eden services) | Section 2.7 |
| 19 | The workspace path (`Documents/ChatGPT/NXN1X`) and premises (12 GB device, Turnip) in `docs\nvn-memory\` differ from the current tree. Most of it ended as runtime pending | `docs\nvn-memory\FOUNDATION_DESIGN.txt` |
| 20 | There is no record of which games the Switch 1 regression titles "Zelda 1/2" specifically are | `regression-games.md` |

---

## 9. Change history

| Date | Document | Content |
|---|---|---|
| 2026-11-16~18 | `docs\nvn-memory\` | Adreno-era NVN memory-based work (F2-F5Z) |
| 2026-11-19 | `docs\DLSS_ARM64_CAPABILITIES.md`, `docs\DLSS_ARM64_TEST_REPORT.md` | N1X feature probe, Streamline integration, automatic FG input detection, optical-flow interpolation |
| 2026-11-19~20 | [`design\HISTORY_20261120.md`](../design/HISTORY_20261120.md) | Import memory measurements, vbmir, copy mode stages 1 and 2, NCE crash fixes, StarFox command memory |
| 2026-11-20~21 | `design\*.md` | Designs for hybrid, atomic staging, vertex pulling, occlusion, barriers, OoT DLSS, parallel replay |
| 2026-11-21 | `BUILD_LOG.txt` | OoT DLSS (ootdlss7~11), resfix, fgoverlay/fgsource, nanguard, flashdiag, cbguard, upswap |
| 2026-11-22 | `BUILD_LOG.txt`, memory notes | DK: fstage, hangstk, parck, uppool, dkpar, optfill |
| 2026-11-22 | [`docs\RELEASE_2026-11-22.md`](RELEASE_2026-11-22.md) | First release: dlssrange, NCE cleanup and loader order, ns2res1x, icon, dump folder boot (dirfs, maxopen, fsfail), make_nup.py, FE fixes (maxe4k, htarr/cidx, copymirror, progforget, opthold, viewusage) |
| 2026-11-22 | [`docs\NCE_CLEANUP_AUDIT_2026-11-22.md`](NCE_CLEANUP_AUDIT_2026-11-22.md) | NCE cleanup pre-check |

Per-build details are in `BUILD_LOG.txt` (one line per build, times in UTC). Originals before changes are in `_backup\<tag>\`.
