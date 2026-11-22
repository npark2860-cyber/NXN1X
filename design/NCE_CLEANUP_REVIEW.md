# NCE cleanup review (read-only)

Scope: `source/src/core/arm/nce/*` plus the glue it depends on (`core/loader/nso.cpp`, `core/memory.cpp`,
`core/hle/kernel/physical_core.cpp`, `core/hle/kernel/board/nintendo/nx/k_system_control.cpp`,
`core/hle/service/ro/ro_nce_windows.cpp`, `core/nvn/nvn_layer.cpp`, `core/nvn/sv1.cpp`, `common/game_presets.h`,
`common/hang_diag.h`, `yuzu/main.cpp`). Nothing was built or run. All paths below are relative to
`source/src/` unless stated. "Guess" marks anything not confirmed by reading code.

Build facts used throughout:
- `NXN1X_DIAGNOSTICS` is a CMake option, default OFF (`source/CMakeLists.txt:6`), and the current build cache has it OFF
  (`build/.../CMakeCache.txt:1072`). So "diag-only" below means "compiled out of the build in use".
- `common/hang_diag.h:11` makes `HangDiag::COMPILED` false when `NXN1X_DIAGNOSTICS` is off, so `BreakWatch` and
  `Vulkan::HangWatch::Scope` compile to nothing.
- On Windows the build compiles `arm_nce_windows.cpp` + `windows_*` + `x18_*` + `windows_nce_entry.asm` +
  `ro_nce_windows.cpp`; `arm_nce.cpp`, `arm_nce.s`, `interpreter_visitor.cpp` are only compiled on the non-Windows
  branch (`core/CMakeLists.txt` ~1304-1335).

## 0. Key numbers

| Item | Value |
|---|---|
| NCE folder total | 15,864 lines in 39 files |
| Windows-only NCE code | ~8,410 lines (arm_nce_windows.cpp 3,137; windows_* 3,447; x18_* 784; nx2_probe* 314; hwa_pool.h 107; nce_write_step.h 187; +303 in ro_nce_windows.cpp) |
| Linux/Android-only (upstream Eden) | 4,366 lines (arm_nce.cpp 421, arm_nce.s 268, interpreter_visitor 894, visitor_base.h 2,783) |
| Shared | 3,086 lines (patcher.cpp 2,185 with 29 `defined(_WIN32)` blocks, patcher.h, instructions.h, guest_context.h, arm_nce.h, current_nce_context.h, asm defs) |
| `ArmNce::RunThread` (Windows) | `arm_nce_windows.cpp:2188-2880` = 693 lines (upstream `arm_nce.cpp:207-265` = 59 lines) |
| `#if NXN1X_DIAGNOSTICS` in arm_nce_windows.cpp | 28 blocks, ~687 lines; plus 33 `NXN1X_R2A(...)` macro sites |
| Diagnostic code **not** behind that flag in arm_nce_windows.cpp | ~420 lines (crash ring, abort report, TEB-fit/nceresume counters, self-test, title logs, HWAoI pool diag, Stats call sites) |
| `NCE::Stats::` call sites in arm_nce_windows.cpp | 39 |
| Environment switches read by NCE code | 10 (table in section 4), plus 9 in the glue |
| Title/build-keyed hooks | 6 titles, 13 sites (section 1) |
| x18 handling tiers | 9 (section 3.1) |

## 1. Game-specific hooks

Classification: **fix** (the right behaviour for that guest), **workaround** (masks a cause elsewhere),
**diagnostic** (logs or probes, no behaviour change), **feature** (user-visible option).

| # | Title (ID) | Where | What it does | Kind | Default path? |
|---|---|---|---|---|---|
| 1 | Hyrule Warriors: Age of Imprisonment (`Settings::HyruleWarriorsProgramId` = 0x04003F40232DE000, `common/game_presets.h:10`) | `core/arm/nce/hwa_pool.h:13-105` (Install), called from `core/loader/nso.cpp:538-558` for every `main` module, keyed on build ID (`hwa_pool.h:23-26`) and code words | Rewrites the allocator's slot-48 entry (main+0x1adce64) and main+0x1add1a4 into a 36-word stub placed in the .text padding (main+0x1ed71c0). The stub retries the full-pool exit up to 128 times with `svc #0xb` (SleepThread 1 ms) | **Workaround** (pool pressure, the cause is not identified in code) | Yes, for that build. The build-ID check costs nothing for other titles |
| 2 | HWAoI | `nso.cpp:579-587` | Adds 5 extra BRK #0xF000 sites (3 failure exits, the stub's Recovered and Failed NOPs) to the x18 fallback list so they trap to the host | **Diagnostic** (probes) | Yes, for that build. Each hit is a full BRK, VEH, NtContinue, Dynarmic step cycle |
| 3 | HWAoI | `arm_nce_windows.cpp:2712-2770` | On every x18 fallback return (`hr == ReturnMarker`) reads `HwaPool::probe_base`. If it is set, decodes the HWAoI-specific exits, logs "pool retry recovered", "slot48 occupancy", "pool terminal", "pool failure" with hard-coded main offsets (0x1adcecc, 0x1add054, 0x1add17c, 0x1adcec0, ...) | **Diagnostic** | The relaxed load and branch run for every title on every x18 trap. The decode runs only when HWAoI is loaded |
| 4 | HWAoI | `arm_nce_windows.cpp:2593-2626` | In the "unresolvable data abort" path, matches fault 0x18, `str x22,[x0,#0x18]` and the BLR pattern, then logs "HWAoI null allocation" with the receiver's vtable slot 48 | **Diagnostic** | Only on that fault |
| 5 | HWAoI | `nso.cpp:284-341` | Finds the frame-interval global by instruction pattern. Nothing is patched; the present path writes it | **Feature** (frame pacing) | Yes, for that title |
| 6 | HWAoI | `core/nvn/sv1.cpp:273+` (pooldiag) | Scans the heap for pool objects using `HwaPool::probe_base` and `AllocatorVtable` (`hwa_pool.h:20-22`) | **Diagnostic** | Yes, when HWAoI is loaded (a host thread polling every 200 ms, up to 30 scans) |
| 7 | 0x0400E95004038001 (the NCE code calls it "XC2 3.0.1"; `k_system_control.cpp:134` calls it "BF2". **The two names disagree**) | `arm_nce_windows.cpp:2565-2592` | On a null store fault (`0x39000001`, x0=0, x2=0x20000) logs the startup arena fields from x19/x20 | **Diagnostic** | Only on that fault |
| 8 | same ID | `core/hle/kernel/board/nintendo/nx/k_system_control.cpp:133-136` | Reserves a dedicated 64 GiB NX2 address region | **Fix/compat** (memory layout), outside NCE but behind the fault in row 7 | Yes |
| 9 | Mario Kart World (0x0400C3F00006E000) | `core/arm/nce/nx2_probe.h:33-87`, `nx2_probe_trap.h`, sites added at `nso.cpp:578`, trap hooks `arm_nce_windows.cpp:2655-2657, 2772-2775` | Double-free probe: 3 hard-coded `main` offsets (0x47ff44, 0xa13d4, 0xa1640) turned into BRK sites that log alloc/free/event records. Keyed on module name `"main"` plus word match, **not** on program ID | **Diagnostic**, labelled "Temporary" (`nx2_probe.h:6`) | Off unless `NXN1X_NX2_PROBE`. **But** `nx2_probe_trap.h:41` sizes `g_trap.ring` with a 65,536-entry `std::vector<Record>` during static initialization in every process (guess: about 3.5 MiB at 56 B per record) |
| 10 | Mario Kart World | `nso.cpp:195-282`, `684-686` | 30 FPS or dynamic FPS patch found by pattern match (`mkw_30fps`, `mkw_dynamic_fps` settings) | **Feature** | Only when the setting is on |
| 11 | Donkey Kong Bananza (0x040074A01BF12000, build E25C90C1...) | `nso.cpp:343-482` | "realtime" patch: 8 sites (`Clocks` + `Controller`) rewritten to stubs in the .text padding that read the host-measured frame time | **Workaround/feature** (slow motion when frames are slow) | **On by default** (`DM_REALTIME_PATCH`, `nso.cpp:371-372`) |
| 12 | Donkey Kong Bananza | `nso.cpp:425-475` (dkctl) | A stub at main+0x204c20 snapshots x0-x30, d0-d7 and CNTVCT into a record after .bss | **Diagnostic** | **On by default** (`DM_DK_CTL_LOG=0` turns it off, `nso.cpp:434,449`) |
| 13 | Ocarina of Time (0x0400486024DEC000, build E0F3CCFA...) | `nso.cpp:353-366` (logs the words at two sites), `483-531` (ootspeed: `fmax` of the game's step and the measured step at main+0x6e6cc4) | Log; time-step patch | **Diagnostic** + **workaround/feature** | ootspeed on by default (same `DM_REALTIME_PATCH`) |

Titles that appear only in comments (Gunvolt, TotK, Bayonetta 3 0400E88025B1A000) are evidence for decisions and
are not hooks.

**Structural hazard in the loader (applies to rows 5, 10, 11, 13):** `nso.cpp:546-549` records that the planning
pass (`load_into_process == false`, which runs `Patcher::PatchText`) has no program ID yet. Every patch keyed on
`process.GetProgramId()` therefore lands only in the loading pass, after the NCE patch plan was made from the
unpatched text. Any SVC, `MRS TPIDR*`, `MRS CNT*` or x18 instruction such a patch adds is never NCE-patched. This was
the "stubsvc" bug for HWAoI, fixed only there by switching to a build-ID key. The DK dkctl stub's
`mrs x10, cntvct_el0` (0xD53BE04A, `nso.cpp:465`) stays raw for the same reason. That happens to match its comment
("raw CNTVCT_EL0"), but it holds only because of the pass order. A patch that overwrote an instruction the plan
patched would also be silently overwritten by `RelocateAndCopy`'s branch.

## 2. Temporary workarounds and diagnostic code

"Default" = runs in the build in use (`NXN1X_DIAGNOSTICS` OFF, no env set).

### 2.1 On the default path (not behind any flag)

| Item | Where | Cost / risk |
|---|---|---|
| **crashring**: every exception that reaches the VEH is recorded twice (enter and return) in a 256-entry file-backed ring `crashring.bin`, plus `MoveFileExW` of the previous run's ring at startup | `arm_nce_windows.cpp:1324-1477`, init `2092`, record calls `1651, 1708, 1722, 1726` | Per exception: 2x QPC, 2x shared `__atomic_fetch_add` on one header cache line (contended by all cores), 2x128 B stores to a mapped file, `GetCurrentThreadStackLimits`. The NVN watched-write path and every guest fault pay it. Comment calls it "diagnostic, log only" |
| **Abort report**: installs process-wide SIGABRT, CRT invalid-parameter, purecall and `std::terminate` handlers and `dm_fatal_hook`, writing `nxn1x_abort_<pid>.txt` | `arm_nce_windows.cpp:1343-1399`, called from `CrashRingInit` (`1401`) | One-time. A process-wide side effect owned by the NCE VEH install; belongs in the frontend or common crash code |
| **tebfix5 TEB fitting + nceresume checked resume** | `1479-1711` (`FitTebToSp`, `ResumeChecked`, `NormalizeTransitionPc`) | **Functional**, not diagnostic: it keeps RtlRestoreContext from fast-failing (0xd). The counters and logs around it (`g_teb_fit`, `LogTebFitCounters 1573-1605`, `LogNceResumeCounters 1655-1664`) are diagnostic and run on every `PrepareRunThread` (cheap relaxed loads) |
| **NCE fast-fail codes** 0xE0CE0001, 0xE0CE0004, 0xE04E4352, 0xE04E4353 | `1633-1641`, `windows_nce_transition.cpp` RedirectToHost, `865-873`, `975-988` | Functional (fail closed). Three separate fail-fast helpers do the same thing |
| **s39 parameter-lock steal**: generated `LockContext` takes over a lock still held after 2^23 tries; a detached thread logs the count every 2 s | `patcher.cpp:2068-2147` | **Workaround** (dated 2026-11-13, TotK freezes). Default ON (`DM_S39`). The steal counter is a plain LDR/ADD/STR from generated code (`2135-2138`), so concurrent steals can lose counts (diagnostic only). The logger thread exists in every process. The code is in the shared patcher, so a Linux build would get it too |
| **BreakRescue "experiment"**: a watch thread polls every 5 ms (`POLL_MS`, `windows_cross_thread_break.h:203`). A delivery stuck in GetThreadContext gets GetThreadContext or Suspend/Resume "kicks" on target and sender after 20 ms, 1 s, 2 s and 3 s | `windows_cross_thread_break.h:178-518`, `Register 594-601`, per delivery `676-694` | **The comment (`:178`) says "off unless EDEN_NCE_BREAK_RESCUE=1". The code (`:235-245`) makes it ON when the variable is unset in diag builds, and ALWAYS ON in release (`return true`).** In release it runs as an unlogged workaround: a 200 Hz poll thread, 2 duplicated handles per core thread, and BeginWait/EndWait atomics plus 2 `steady_clock` reads on every delivery |
| **CurrentNceContext history**: QPC + ring store + `IsDebuggerPresent` on every Install/Clear (each native entry and exit); `OutputDebugStringA` under a debugger | `current_nce_context.h:91-150` | One QPC and a few stores per native entry/exit. Consumed only by the null-context crash record `windows_nce_transition.cpp` (`g_eden_nce_null_context_report`) |
| **vbvol watched-write counters** `NvnLayer::NoteWatchFault` | VEH `arm_nce_windows.cpp:1772-1774`, `core/nvn/nvn_layer.cpp:579-587` | 1-2 shared global atomics per write fault from every thread |
| **Title diagnostics** (rows 3, 4, 7, 9 of section 1) | as listed | A probe_base load per x18 trap; the rest is fault-only |
| **"NCE emulated faulting guest instruction"**, "skipped", read-guard fallback logs | `2639-2654, 2671-2675, 2449-2455` | Capped (8/16) logs. Reasonable to keep |
| **NCE_FRAME_MISMATCH** | `2321-2338` | Behind `Stats::Enabled()` |
| **LogWindowsNceAbort**: dumps words around the PC on PrefetchAbort | `111-135`, `2832-2834` | Fault-only. Keep |
| HWAoI pooldiag thread | `core/nvn/sv1.cpp:265+` | HWAoI only |
| dkctl register snapshot stub | `nso.cpp:425-475` | Default on for DK |

### 2.2 Env-gated (default off, read once)

`EDEN_NCE_STATS` (counters and a 1 s report; when off, every `Stats::Add/Now` is one cached-bool test),
`EDEN_NCE_X18_HOST_FPCR` (audit of the FPCR fix; still adds a static-guard check per direct x18 step,
`windows_x18_fallback_runner.cpp:121`), `NXN1X_NX2_PROBE`, `NXN1X_NCE_SELFTEST` (`1672-1703`, a startup self-test of
the resume rule), `EDEN_NCE_INLINE_FAULT` (an experiment that "hung TotK", `1927-1929`), `EDEN_NCE_LEASE`
(an A/B for the abandoned private stack lease, `257-271`).

### 2.3 Compiled out (NXN1X_DIAGNOSTICS)

R2a delivery trace (`windows_nce_r2a_trace.h`, 761 lines, writes `nce-r2a\`), deferred fault records b56/b58/b60
(`990-1322`), `EDEN_NCE_RECOVER_OFF` (`569-611`), `EDEN_NCE_WRITE_STEP` (`2462-2516`, `nce_write_step.h`),
NCE_WRITE_NOTE / TILE_TRACE (`1967-2041`, `2526-2552`), read-guard watch lines (`2406-2448`), EpochGuard
(`windows_nce_epoch_guard.h`, `physical_core.cpp:119-136`), X18_ATOMIC_PATCH counts (`patcher.cpp:106-119, 1292-1300`),
BreakWatch.

### 2.4 Dated or labelled investigation notes in code

147 tag hits in `arm_nce_windows.cpp` alone (tebfix/nceresume/crashring/stage2*/b5x-b7x/R2b-*/P2x-P3x/V16/V72/IMP-00x/
s25/vbvol), 20 in patcher.cpp, 25 in r2a_trace, 20 in recovery_metadata, 11 in windows_nce_transition.cpp. Many
comments narrate crash dates ("HWAoI 11:11, 11:22; Bayonetta 3 11:53", "16:54", "17:06") rather than stating the
invariant. Stale references: `windows_nce_entry.asm:10` points at `kGuestSpHostRanges`, which no longer exists
(tebfix4 was replaced by tebfix5). `windows_nce_transition.cpp` includes `common/logging.h` "x18fix diagnostic" but
makes no `LOG_` call. Twelve NCE log lines use the `Render_Vulkan` category (e.g. `963, 1235-1282, 2422-2446`).

## 3. Duplicate, overlapping or unused paths

### 3.1 x18 (Windows platform register) handling: 9 tiers

Order of precedence in `Patcher::PatchText` (`patcher.cpp:1025-1290`), then load-time BRK, then runtime:

| Tier | Instructions | Mechanism | Where |
|---|---|---|---|
| a | CBZ/CBNZ/TBZ/TBNZ on x18 | x16 spill on guest stack, TEB-slot lookup, native branch; R2b-B2x2 recipe | `patcher.cpp:781-886, 1173-1186` |
| b | SBFM/BFM/UBFM with x18 (P2A) | 2 scratches spilled on guest stack; recipe B2x1 | `596-682, 1188-1201` |
| c | LSE touching x18 | full 0x130 save frame, **C-call getter** (BLR) | `488-594, 1203-1226`; plan in `windows_x18_lse.h` |
| d | LDXR/STXR pair, single exclusive with x18 | same 0x130 frame and C-call getter | `310-486, 1229-1256`; `windows_x18_exclusive.h` |
| e | Any decoder-validated register substitution | TEB-slot spill (15 steps), recipe B2 | `121-187, 693-779, 1260-1272` |
| f | Remaining linear/non-SP x18 (P3A) | save context, switch to host stack, Dynarmic single step (`WindowsNceExecuteDirectX18`) | `888-1022, 1274-1288`; `windows_x18_fallback_runner.cpp:111-183` |
| g | Everything else `SupportedOrdinary` (branches, SP-changing) | BRK #0xF000 at load, VEH redirect, RunThread `Dispatch`, Dynarmic step | `x18_site_patcher.cpp`, `nso.cpp:577,625`, VEH `1805-1818`, RunThread `2776-2812` |
| h | MRS CNTFRQ/CNTPCT/CNTVCT/TPIDR into x18 | special cases inside the system-register handlers | `patcher.cpp:1975-2066`, MRS handler |
| i | Break capture inside a tier a/b/e or MRS helper | recipe replay (`TryRecoverX18Substitution`) | `arm_nce_windows.cpp:613-759`, `windows_patch_recovery_metadata.h` |

Overlaps and candidates:
- **b vs e:** P2A bitfield predates the general substitution. Substitution validates any register-only form against
  the Dynarmic decoder, which would cover bitfields (guess, needs a check that `BuildWindowsX18SubstitutionPlan`
  accepts `0x331F7A4F`). P2A is shorter (7 vs 15 words) but uses the guest stack. It is a candidate to fold in, or a
  documented "fast form".
- **c and d** each repeat the same ~40-instruction save/getter/restore sequence three times (`328-403`, `423-486`,
  `512-594`). They also call the C getter `GetCurrentNceContextForGeneratedCode` even though `yuzu/main.cpp:102`
  reserves the TEB slot at startup (`ReserveDirectSlots`), so the one-load TEB lookup used by every other tier
  would work. The C-call fallback in `windows_generated_context.h:31-90` is reached only if `TlsAlloc` fails.
- **f vs g:** both do a Dynarmic single step. g pays the VEH and NtContinue on top. g is still needed for
  branch/SP forms; f's own stack-refresh marker (`DirectStackRefreshMarker`, runner `:150-155`, RunThread
  `2373-2377`) overlaps with the later **nceresume R8** refresh (`WindowsNceRefreshGuestStackBounds`,
  `arm_nce_windows.cpp:2050-2060`, runner `:121-128`). Both exist because guest fibers change SP inside an epoch.
- **P2B/P2C private-stack-sync decisions** (`402-510`) only matter when a private stack lease exists, and the lease
  is off by default (3.2).

### 3.2 Paths that are dead by default

- **Private guest stack lease** (`EDEN_NCE_LEASE`, default off since it "cost most of the frame time ... found
  unnecessary", `257-260`). All of this is dormant: `EnsureWindowsGuestStackLease` after the early return
  (`288-399`), `P2BGuestX18Register`..`P2CNeedsPrivateStackSync` (`402-510`), RunThread lease handling
  (`2229, 2287-2293, 2685-2709, 2785-2790, 2814-2830`), runner `SetDirectPrivateStackView` / private memory view,
  lease stats, `Common::HostMemory::PrivateMappingLease` (`common/host_memory_windows_lease.h`). That is roughly
  300+ lines.
- `WindowsNceNoLeaseExperiment()` is tested twice in a row (`284` and `296`). The second test is unreachable.
- **Inline fault resolution** in the VEH (`1922-1938`): off by default, and the comment says turning it on hung TotK.
- `(void)stack_limit;` (`1746, 1760`): a computed and unused value left over from before R7.
- Exported asm labels `WindowsNceEnterGuestEnd` and `WindowsNceGuestStackBridgeEnd` (`windows_nce_entry.asm:11-12`) have
  no C++ reference.
- `WriteWindowsX18DirectTrampoline` takes `load_ctx` and discards it (`patcher.cpp:893`).
- `ArmNce::StepThread` returns `StepThread` without stepping (`2882-2885`), the same as upstream: debugger single
  step is not supported under NCE.

### 3.3 Code repeated inside the Windows path

- **The VirtualQuery allocation walk appears 3 times:** `InstallGuestTebStackBounds 174-200`,
  `EnsureWindowsGuestStackLease 336-351`, `AllocationOf 1509-1522` (the newest). `nx2_probe_trap.h:Readable` is a fourth
  variant.
- **The RtlRestoreContext SP rule appears twice:** `TebAcceptsSp 1504-1506` and inline at `CrashRingRecord 1469`.
- **"Read the guest instruction at pc"** (`GetPointerSilent` + memcpy) appears 6 times in RunThread (`2418, 2437, 2481,
  2520, 2530, 2561`), plus raw memcpy reads in the VEH (`1044, 1204, 1852`).
- **The guest-stack headroom rule** appears twice (watched-write `1757-1768`, inline fault `1924-1926`).
- **Three NVN watched-write entry points:** the VEH `1742-1778`, `Memory::InvalidateNCE` (`memory.cpp:1318-1322`), and
  through RunThread's InvalidateNCE. The comments say why (stack room), but `OnWatchedWrite` is asked again on
  the fallback path.
- **Cache maintenance is handled twice:** `patcher.cpp:1119-1134` removes `DC CIVAC` statically, and the VEH emulates
  IC IVAU / DC CVAU / CVAC / CIVAC / CVAP / CVADP at runtime (`1843-1877`). Together they cover static and
  self-modified code, but the static pass could rewrite all six.
- **Fail-fast helpers:** `FailFastOnJobOverwrite`, `FailFastOnResumeFailure`, `NceFailFast`, and RedirectToHost's
  inline one.

### 3.4 Repeated across the platforms

- `ArmNce::UnlockThread` is the same in `arm_nce.cpp:191-205` and `arm_nce_windows.cpp:2128-2142`. `LockThread`,
  `GetSvcArguments`, `SetSvcArguments`, `GetContext`, `SetContext` and `GetSvcNumber` are near-identical; Windows
  `SetContext` also sets `nzcv` (`2931`), Linux does not.
- `x18_fallback.cpp:CopyGuestToThreadContext/CopyThreadContextToGuest` (`48-81`) repeat `GetContext/SetContext` with
  NZCV handled differently.

### 3.5 Switches that are always one way

- `BreakRescue::Enabled()` is constant true in release (2.1).
- `DM_S39`, `DM_REALTIME_PATCH` and `DM_DK_CTL_LOG` are on unless set to 0. Their "off" branches exist only to
  restore old behaviour, which goes against the root-fix-only preference.
- `CurrentNceContext::HasTebSlot()` is effectively always true because `ReserveDirectSlots` runs in `main()`. Every
  `if (HasTebSlot())` else branch, and the C-call lookup, is fallback code that never runs.

### 3.6 Latent bug found during the review (needs confirmation)

`patcher.cpp:1206-1213`: when an LSE instruction touches x18 and `BuildWindowsX18LsePlan` returns `nullopt`,
`PatchText` returns `false`. Callers read `false` as "module out of branch range, use a new Patcher":
`nso.cpp:569-572` loops `while (!patch->PatchText(...)) patch = &patches->emplace_back();`. A fresh Patcher hits the
same word again, so (guess) the load spins forever and keeps allocating Patchers. `nullopt` is returned for CASP
with an odd register pair (`windows_x18_lse.h:140-144`), a reserved encoding that can still occur as data inside
.text. The early return also leaves `curr_patch` half-built (`modules.emplace_back()` at `289`). Fix: treat the word
as data (leave it), or route it to the BRK tier (g), and never overload the range signal.

## 4. Switch list

### 4.1 Read by NCE code

| Name | Default | Effect | Read at | When read |
|---|---|---|---|---|
| `EDEN_NCE_LEASE` | off | private guest stack lease (A/B) | `arm_nce_windows.cpp:263` | once (static, first RunThread) |
| `EDEN_NCE_INLINE_FAULT` | off | resolve write faults inside the VEH | `arm_nce_windows.cpp:2094` | once at VEH install |
| `NXN1X_NCE_SELFTEST` | off | run the resume-rule self-test at VEH install | `arm_nce_windows.cpp:2098` | once |
| `EDEN_NCE_RECOVER_OFF` | none | disable R2b recovery per kind/step (`subst,bitfield,mrs,cond,all,kind:N`) | `arm_nce_windows.cpp:608` | once; diag builds only |
| `EDEN_NCE_WRITE_STEP` | off | single-step faults on GPU-cached pages through Dynarmic | `nce_write_step.h:50` | once; diag builds only |
| `EDEN_NCE_BREAK_RESCUE` | **on** (diag); **forced on** (release) | break-rescue kicker thread | `windows_cross_thread_break.h:238` | once; ignored in release |
| `EDEN_NCE_STATS` | off | hot-path counters, 1 s NCE_STATS line | `windows_nce_stats.h:62` | once |
| `EDEN_NCE_X18_HOST_FPCR` | off | FPCR audit around the direct x18 helper | `x18_host_fpcr.h:56` | once (first direct x18 step) |
| `NXN1X_NX2_PROBE` | off | MKW double-free probe | `nx2_probe.h:49` | once (module load) |
| `DM_S39` | **on** | generated LockContext steals a lock after 2^23 tries; logger thread | `patcher.cpp:2075` | once (first LockContext emission, at patch time) |
| `Settings::values.use_reactive_flushing` | setting | write-note reads skipped when on | `arm_nce_windows.cpp:2000` | live; diag only |

### 4.2 Read by NCE glue

| Name | Default | Effect | Read at | When read |
|---|---|---|---|---|
| `Settings::IsNceEnabled()`, `use_reactive_flushing` | setting | NCE read guard on/off | `memory.cpp:820-827` | live, every guard/fault call |
| `DM_REALTIME_PATCH` | on | DK and OoT realtime patches | `nso.cpp:371` | per module load |
| `DM_DK_CTL_LOG` | on | DK register-snapshot stub | `nso.cpp:434` | per module load |
| `mkw_30fps`, `mkw_dynamic_fps` | settings | MKW FPS patch | `nso.cpp:202-204` | per module load |
| `dump_nso` | setting | NSO patch/dump | `nso.cpp:183` | per load |
| `EDEN_CGT` | off | CPU/GPU trace; RunThread records NCE faults | `common/cgt_trace.h:202`, used `arm_nce_windows.cpp:2518`, `memory.cpp:1338` | once |
| `EDEN_UB_KEEP` | off | copies before NCE store faults | `common/ub_keep.h:672`, `memory.cpp:1020` | once |
| `EDEN_NX2_IDLE_STATS` | off | core idle stats | `physical_core.cpp:34` | static init |
| `NVN_HOST_GAME` (CMake) / `NXN1X_NVN_HOST` (define) | build | NVN gate, EntryHooks, watched writes | `core/CMakeLists.txt:1352` | compile time |

Not reviewed but NCE-named, in video_core: `EDEN_NCE_BUF_GPU`, `EDEN_NCE_GPU_READBACK`, `EDEN_NCE_IMAGE_READBACK`.

## 5. Platform-dependent parts

### 5.1 Windows-only mechanisms, and what Linux/Android has instead

| Concern | Windows (this fork) | Linux/Android (upstream path in the same folder) |
|---|---|---|
| Fault entry | VEH `AddVectoredExceptionHandler(1, ...)` (`2099`), runs **on the faulting (guest) stack** | `sigaction` + `sigaltstack` (`arm_nce.cpp:294-350`), handler on its own stack |
| Leaving guest code after a fault | `RedirectToHost` + `NtContinue` into `WindowsNceHostStackBridge` (`windows_nce_transition.cpp`, asm) | edit `ucontext` and return from the signal (`SaveGuestContext`) |
| Entering guest code | `WindowsNceEnterGuest` / `EnterGuestContext` in armasm64 (`windows_nce_entry.asm`, built with a separate armasm64 step in CMake) | `arm_nce.s` (GNU as), `ReturnToRunCodeByTrampoline` / `ByExceptionLevelChange` |
| Platform register x18 | TEB pointer, must never be guest-owned. Hence tiers a-i of 3.1, recovery recipes, BRK fallback, Dynarmic fallback runner (`x18_*`, `windows_x18_*`, ~2,500 lines) | free for the guest; **none of this is needed** |
| Thread state pointer for generated code | TEB TLS slot via `LDR [x18,#slot]` or a C getter (`windows_generated_context.h`, `current_nce_context.h`) | `TPIDR_EL0` (`patcher.cpp:2111-2115` `#else`) |
| TEB StackBase/StackLimit | must match SP for RtlRestoreContext and exception dispatch: tebfix/tebfix5/nceresume (`1479-1729`), publish windows in asm, `NormalizeTransitionPc` | not applicable |
| Cross-thread interrupt | per-core worker thread doing Suspend/GetThreadContext/SetThreadContext/Resume, delivery gate state machine, BreakRescue (`windows_cross_thread_break.h`, `878-943, 2938-3127`) | `tkill(BreakFromRunCodeSignal)` (`arm_nce.cpp:386-404`) |
| EL0 cache maintenance, CTR_EL0 | trapped by Windows: VEH emulation (`1843-1877`), patcher skips DC CIVAC and emulates CTR_EL0 (`patcher.cpp:1104-1134`) | allowed natively |
| Guest stack bookkeeping | `VirtualQuery` allocation walks, `PrepareRunThread` guest page-table cache (`2144-2186`), lease | not needed |
| Alignment faults | **no handler** (an `EXCEPTION_DATATYPE_MISALIGNMENT` would go to CONTINUE_SEARCH; guess: rare on ARM64 Windows except for exclusives/atomics) | `interpreter_visitor.cpp` emulates through SIGBUS |
| Unresolvable data abort | Dynarmic single step with JIT read-as-zero semantics, skip as a last resort (`2554-2677`) | skip the instruction (`arm_nce.cpp:124-127`) |
| Fail-fast | `RaiseFailFastException` with custom codes | abort |
| Process-wide crash plumbing | crashring, abort report (Windows files, CRT hooks) | none |

### 5.2 Portable (or should be)

- Patcher core: SVC/MRS/MSR/CNTPCT/CNTFRQ, exclusive ordering, relocation, split mode, EntryHooks (NVN).
  `guest_context.h`, `instructions.h`, `patcher.h`.
- **The NCE read guard** (`memory.cpp:812-910`, RunThread `2387-2460`) fixes a platform-independent problem: native
  guest reads of GPU-modified pages. `NceReadGuardEnabled` returns false off Windows. Guess: Linux NCE has the same
  stale-read bug and would need a PROT_NONE + SIGSEGV read path.
- The fault-emulate-before-skip policy (portable; Linux still skips).
- s39 lock steal (already shared through patcher.cpp).
- `PrepareRunThread` (an NXN1X addition to `ArmInterface`, `arm_interface.h:62`; only Windows overrides it).

### 5.3 Porting effort

Porting this fork's NCE to Linux/Android means keeping upstream's `arm_nce.cpp`/`arm_nce.s`/interpreter (already in
the tree) and carrying over only the portable fixes in 5.2. Nothing in the x18, TEB, VEH or break-worker code is
needed there. Risk: the non-Windows path is not built in this project (guess, based on CMake and no Linux build
dir), so it may have bit-rotted against shared changes (`NativeExecutionParameters` fields, `CurrentNceContext::Install`
site argument, `RelocateAndCopy` signature with `#if _WIN32` parameter, `patcher.cpp` s39 and EntryHook changes).
A CI compile of `HAS_NCE` on Linux arm64 would show this.

## 6. Cleanup proposals (ordered by value/risk)

Each item lists what, why, risk, and how to check for regressions. None of these were applied.

**P1. Fix the PatchText "unplannable LSE" return (3.6).**
What: treat it as data or as a BRK-tier site; reserve `false` for the range signal only.
Why: a latent load hang.
Risk: low.
Verify: a unit test that feeds `PatchText` a text segment containing an odd-pair CASP with x18 (e.g. `rs=17`) and
asserts it returns `true` and patches nothing there. Then load the usual titles and diff `X18_ATOMIC_PATCH` counts
(diag build).

**P2. Make BreakRescue an explicit decision (2.1, 3.5).**
What: either (a) it is needed in release: rename it as a fix, keep it, fix the comment, log once that it is active;
or (b) it is an experiment: gate it like the others (off in release).
Why: the code contradicts its own documentation, and a 200 Hz poll thread plus thread kicks run unobserved in
release.
Risk: medium. If (b) is chosen and the GetThreadContext wait recurs (seen on build 26220), the guest stalls again.
Verify: a long soak on TotK/HWAoI with the diag build's `NCE_BREAK_STATS` / `NCE_BREAK_RESCUE_RESULT` lines; count
rescues over N hours before deciding.

**P3. Move default-path diagnostics behind a compile flag (2.1).**
What: put crashring, the abort report, `CurrentNceContext` QPC history, the s39 logger thread, vbvol counters, the
title logs (XC2/HWAoI rows 3, 4, 7), the HWAoI probe BRK sites (row 2), dkctl (row 12) and the static 65,536-entry
NX2 ring behind `NXN1X_DIAGNOSTICS`, or a narrower `NXN1X_CRASH_FORENSICS` if crashring must stay in field builds.
Move `InstallAbortReport` out of NCE into the frontend.
Why: removes per-exception file-mapped writes and contended atomics; makes `RunThread` and the VEH read as policy.
Risk: low-medium. crashring is currently the only record for 0xd fast-fails, so keep it available through the flag.
Do not touch `FitTebToSp`/`ResumeChecked` (functional).
Verify: A/B the frame time on a fault-heavy title (Bayonetta 3 copy mode, NVN watched writes) with `EDEN_NCE_STATS=1`
for exit counts. The nceresume counters must show the same entries and resumes.

**P4. Split RunThread into named stages (pure refactor).**
What, with the current line ranges:
1. `OpenEpoch` (2189-2232)
2. `PrepareNativeEntry`: pending reason, stack state, TEB bounds (2234-2293)
3. `EnterGuest` (2295-2316)
4. `AccountExit`: stats (2318-2367)
5. `HandlePendingFault` (2379-2683), split into `TryReadGuard`, `TryWriteStep` (diag), `NoteFaultForTraces`
   (CGT/write note), `TryInvalidate`, `EmulateOrSkip`, plus a `TitleFaultDiagnostics` hook
6. `DispatchX18Fallback` (2685-2812)
7. `RetireEpoch` (2814-2879)

Also add helpers `ReadGuestInstruction(pc)` (6 copies) and one `AllocationOf` (3 copies).
Why: 693 lines with 9 `break` and 7 `continue` exits are hard to review; the R2a phase markers already name these
stages.
Risk: low if each stage keeps the exact `hr` / `abort_site` / `continue` contract. Take care with the ordering
comments (R2b-B3: HostSide must be set before any path back to the loop top).
Verify: diag build. R2a phase histograms, NCE_STATS exit categories and the nceresume counters must be identical
over the same scripted session. Also check that the object-code size of RunThread shrinks or stays the same (no new
copies).

**P5. Remove the dormant private stack lease and inline-fault path (3.2).**
What: remove `EDEN_NCE_LEASE`, the lease code, P2B/P2C, the runner's private views and `PrivateMappingLease` if no
other user exists; remove `EDEN_NCE_INLINE_FAULT`; remove the duplicate `WindowsNceNoLeaseExperiment` call.
Why: about 350 lines that never run by default, the A/B is concluded per its own comment, and it fits the root-fix
preference.
Risk: low.
Verify: grep the launchers (`*.cmd`, `launchers_old`) for both variables; run the title set. The exit histogram
must not change, because those paths are skipped today.

**P6. Consolidate x18 handling (3.1).**
What:
- Keep one table (section 3.1) as a header comment in a new `windows_x18_patcher.cpp`, and move all
  `#if defined(_WIN32)` x18 lambdas out of `patcher.cpp` (about 950 lines).
- Replace the C getter in tiers c/d with `WriteWindowsCurrentNceParametersLookup` and assert the TEB slot exists at
  startup. That removes the C-call fallback in `windows_generated_context.h`.
- Factor out the shared 0x130 frame code.
- Measure whether tier b can become tier e.
- Merge the P3C refresh marker with R8.

Why: the rules are spread across 9 places and 6 files, and the same frame code is written three times.
Risk: medium. Generated-code changes alter recipe shapes. The recipes guard themselves with `code_words` compare
and fall back to "retry capture", so a shape change makes captures slower, not wrong.
Verify: a diag build. `X18_ATOMIC_PATCH` counts and the per-kind R2a `RecoveredStep` counts must be the same.
`NCE_STATS` DirectX18Steps and ExitX18Fallback must not increase. Run a TotK/HWAoI/MKW boot and play session.

**P7. Move title hooks out of NCE core (section 1).**
What: add a small `core/game_fixes/` (or `loader/title_patches/`) with a registration API: `OnModuleText(name,
build_id, text)`, `ExtraTrapSites`, `OnX18Trap`, `OnUnresolvedFault`. Move HwaPool, the HWAoI/XC2 logs, MKW/DK/OoT
patches and nx2_probe there. Key every code patch on **build ID**, which the planning pass has, not on program ID.
That fixes the pass-order hazard.
Why: RunThread and the VEH would carry no title constants. All patches would be visible to the NCE planner.
Risk: medium for the DK/OoT/MKW patches (pass order changes which instructions the NCE planner sees). For example,
the dkctl `mrs cntvct_el0` would then be patched into the scaled guest counter.
Verify: the boot log lines "realtime game step: 8/8", "ootspeed: ... patched", "HWAoI pool-pressure retry
installed=true", "Mario Kart World: ... FPS mode enabled". Compare the codeset bytes between the old and new order.

**P8. Fold the root-fix switches (`DM_S39=0`, `DM_REALTIME_PATCH=0`, `DM_DK_CTL_LOG=0`) and document the rest (section 4).**
What: remove the old-behaviour branches. Read all NCE switches in one place (`ArmNce::Initialize` or an
`nce_switches.h`) and log them once, as `NCE_RECOVER_OFF` already does. Find the root cause behind s39 (who leaves
the lock taken: the break worker's SetFailed give-back race, per its comment) rather than keeping the steal.
Risk: low for the switches. s39's root cause is open work.
Verify: a TotK boot loop (the s39 repro, pid notes 23344/12388). The s39 counter must stay at 0 once the root cause
is fixed.

**P9. Share the platform-neutral ArmNce members (3.4).**
What: one `arm_nce_common.cpp` for Lock/Unlock/Get/SetContext/SVC args (fixing the NZCV difference), and use it from
`x18_fallback.cpp`.
Risk: low.
Verify: a Windows build plus a Linux arm64 `HAS_NCE` compile in CI. This also answers the bit-rot question in 5.3.

**P10. Write down the platform boundary, and tidy small things.**
What:
- A short `docs` page with the table in 5.1.
- Remove the stale `kGuestSpHostRanges` comment, the unused exports, `(void)stack_limit`, the unused logging
  include and the unused `load_ctx` parameter.
- Move the 12 NCE log lines from `Render_Vulkan` to `Core_ARM`.
- Replace dated crash narratives with invariant statements, and link the design docs
  (`design/ASTRA_NCE_RESUME_REVIEW*.md`).
- Fold the 4 fail-fast helpers into one.
- Settle the XC2/BF2 name for 0x0400E95004038001.

Risk: none (comments, logs, dead symbols).
Verify: build, then grep the logs for any tooling that parses `Render_Vulkan` NCE lines (e.g. `crashring_dump`, log
scripts) before changing the category.
