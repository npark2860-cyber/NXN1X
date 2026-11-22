# Parallel replay recording (DM_VK_PARALLEL): design

Status: proposal, 2026-11-21. Prerequisite refactors R1, R1b, R2, R3, R6, R7, R8 (first steps) landed in serial mode (R6-R8 syntax-checked only); R4 deferred, R5 analysed.
S1a (worker infrastructure, `DM_VK_PARALLEL=<n>`, inert at 0) landed, syntax-checked only; nothing records on the workers yet (see "S1a" under Stage 1).
S1b (dry pass with CALL-interior segmentation, `DM_VK_PARALLEL_DRY=1`, and the same-thread resume harness `DM_VK_PARALLEL_SERIALSEG=1`, both off by default) landed,
syntax-checked only (see "S1b" under Stage 1); verified in game on OoT 11-21 (DRY: 73% of draws in segments; SERIALSEG: all mismatch counters 0, picture identical).
S1c (the recording path made thread-safe for a worker, workers still off; `DM_VK_PARALLEL_SELFTEST=2` stress test) landed, syntax-checked only (see "S1c" under Stage 1).
S1d (the order-dependent CPU work in the dry pass with a replay that takes its results, full per-segment snapshots checked in SERIALSEG, segment-local uniform update lists,
part cuts placed by the dry pass; workers still record nothing) landed, syntax-checked only (see "S1d" under Stage 1). S1d/S1d2 verified in game on OoT 11-21 (DRY: 80% of 1,900 draws
in 12.5 segments a frame, dry pass 1.68 ms a frame; SERIALSEG: every mismatch counter 0, picture identical).
S1e (segments recorded on the workers, `DM_VK_PARALLEL=<n>` > 0 implies the dry pass; strict eligibility, occlusion untouched; chunks submitted in stream order; workers on the
P-cores) landed, syntax-checked only, not run (see "S1e" under Stage 1).
Scope: NXN1X Windows ARM64 (NVIDIA N1X), NVN-HLE -> Vulkan replay,
Eden render path (`vk_eden_mode()`), NVN1 titles first (Zelda Ocarina of Time is the target).

## 0. Summary

- Today one replay thread records every submitted NVN list serially: ~15 ms a frame in OoT (~2,300 draws, 6 submissions,
  ~2 parts each, parts sent early by `vk_cmd_split`). The GPU does ~14 ms of work and idles ~6 ms waiting for parts.
- Proposal: HeyListen's model, adapted. The replay thread walks the stream once in stream order and does every
  order-dependent decision itself ("dry pass": reports, events, render conditions, uniform-update list, occlusion
  bookkeeping, render-target resolution and marks, part cuts). Runs of draws are cut into **segments** that worker threads
  re-execute from a snapshot of the NVN state into **their own primary command buffers**; all chunks of a part are submitted
  in stream order.
- Stage 1 keeps every GPU-memory-writing op (copies, fills, dispatches, storage-writing draws) on the replay thread, so the
  submission's GPU-write list, pool sync and staging windows never change inside a segment. That is what makes the first
  version tractable.
- Expected: replay thread 15 -> ~8-9 ms, the frame then GPU-bound (~14-16 ms instead of ~20). See section 6.
- Biggest blockers: about 60 file-static caches and counters in `gfx_vk_exec.c` / `gfx_vk_eden.inc` that assume one thread;
  occlusion queries whose slots and post ops are allocated as passes end (must become per-draw slots by stream serial);
  `vk_tex` upload/first-use paths and the shared upload command buffer; unknown thread safety of the `nvn_eden_*` library.

## 1. Reference: HeyListen's parallel replay (GPL-3.0, read only)

Files: `refer/HeyListen-main/engine/gpu/vk/vk_replay.cpp` (VR), `vk_draw.cpp` (VD), `vk_texture.cpp` (VT),
`vk_internal.h` (VI), `docs/SWITCHES.md` (SW), `docs/benchmarks/*` (BM).

### 1.1 Shape

- One interpreter, three modes: `enum class Mode { Serial, Dry, Worker }` with a `thread_local rmode` (VR:1275-1278).
  `replay()` (VR:1827) branches per op on `dry` / `worker` (e.g. events VR:1867-1889, buffer copy skipped in dry VR:1961,
  `UPDATE_UB` VR:2055, `RENDER_COND` VR:2278, draws VR:2380-2448).
- `replay_handle()` (VR:3315): if `par_threads() >= 2` and not dumping/profiling (VR:3319), run a **dry pass** over the whole
  command buffer (VR:3329-3456), choose segment boundaries among saved points (VR:3464-3495), submit what came before
  (`submit_recording(r)`, VR:3531-3537), give each segment its own `Recording` (VR:3542-3605), hand segments to the pool
  (VR:3650-3659), record segment 0 on the replay thread (`RX_VK_PARALLEL_SELF`, VR:3661), help with queued ones while waiting
  (`SELF=2`, VR:3668-3679), and submit each segment **in order** as soon as it and all before it are done (VR:3665-3707).
  What follows the segments goes into a fresh recording (VR:3771-3778).
- A segment is `{p, end, const NvnState* s, ub, cond, serial}` (VR:2784-2789); `record_segment()` (VR:2792-2806) copies the
  state, sets `ub_limit`/`cond_at`/`occ_wserial`, records `full_barrier`, replays `[p, end)`, ends the pass, `full_barrier`.
- Every image lives in `VK_IMAGE_LAYOUT_GENERAL` (VT:4, `VULKAN.md:48`) with full barriers, so no layout tracking crosses
  segments; only a new image's first transition matters. Each segment has an **init command buffer** for those first
  transitions (`texture_segments_begin`/`texture_segment_end`, VT:4630-4645), submitted before its upload and draw buffers
  (VR:3690-3698); losing one is fatal (`replay_fatal`, VR:2812).

### 1.2 What the dry pass computes (serially, in stream order)

- **NVN state points**: a full `NvnState` copy (~27 KB) every `par_point_every()` = 32 draws/dispatches (VR:1298-1304,
  1330-1334, 2456-2470) with the uniform-update count, render-condition count and draw serial at that point.
  `RX_VK_DRY_LEAN` / `RX_VK_DRY_WANTED` only cut the copy cost (VR:1316-1327, 3400-3438).
- **Render conditions**: evaluated on the CPU, results appended to `par_conds` (VR:1279-1297, 2307-2321); workers read
  `par_conds[cond_at++]` (VR:2279-2283). A condition on a report not read yet stops the dry pass there (`occ_stop_at`,
  VR:2287-2291) and the rest goes on after a GPU sync (`occ_go_on`, VR:3781).
- **Counter reports** are written on the CPU in stream order (VR:1940-1951); small `CLEAR_BUF`s on the CPU too (VR:2000-2011).
- **Uniform updates**: appended to the global `ub_updates` deque (VR:2055-2079); a worker only advances `ub_limit` and
  `ub_gen` at each `UPDATE_UB` (VR:2056-2058) and overlays the first `ub_limit` entries on its uniform reads.
- **Events**: `ev_values` updated in order on the dry pass (VR:1867-1876); waits may stop the dry pass (VR:1890-1899).
- **Texture-written marks**: `dry_mark_written()` (VD:11076-11107) marks every colour/depth target of each draw and every
  storage image of each dispatch as GPU-written before any worker runs; clears too since D4 (`dry_clear_marks`,
  VR:1306-1315). Texture-to-texture copy targets likewise (`dry_mark_copy_target`, VR:2050).
- **Occlusion slots**: one query slot per draw by stream serial (`OccSlot`, `occ_serial` / `occ_wserial`, VR:2386,
  2451-2455), so dry pass and workers agree without sharing a counter (SW:3252).
- **Mirrored pool pre-sync**: `memory_sync_mirrors_before_segments()` once, before the segments (VR:2997, 3534), optionally
  on a worker during the dry pass (`RX_VK_PRESYNC_EARLY`, VR:2821-2919). Readback refresh once a part, not per segment
  (VR:2655-2661, 2938-2939).

### 1.3 Per-thread state and shared state

- Per recording: `RecordingRes` (descriptor pool + upload stream, VD:397-438), `thread_local rec_cmd/rec_value` (VD:394);
  per segment `texture_begin_recording(upload)` / `draw_begin_recording(slot, cmd)` (VR:3620-3623).
- Per-thread caches: 79 `thread_local` in `vk_draw.cpp` alone (pipelines `pipe_fast`, view tables, sampler maps,
  program picks). Objects with constructors go through `TLS_OBJECT` plain pointers because llvm-mingw's TLS init function
  cost 2.4% of the replay thread (VD:84-118).
- Shared structures under one recursive lock `ParLock` (VI:292-317), active only while `par_active`; spins 4000 before
  sleeping (VR:3509-3517). Uniform list and condition list growth under `ub_grow_mu` when segments run during the dry pass
  (VR:1335-1373).
- Order repair rule: a texture uploaded in this submission by a *later* segment is uploaded again by the earlier reader
  (VT:3511-3516, 4493, 4551).

### 1.4 Segmentation and scheduling

- Segments of ~160 draws (128 with `RX_VK_FINE_SEGMENTS`, default under 16 hardware threads), at most `seg_cap()` = 6 / 10
  (VR:2545-2568, 3349); count from `segment_plan::segment_count`; boundaries balanced by a per-command-buffer time profile
  (`RX_VK_PARALLEL_BALANCE`, VR:3359-3374, 3737-3747), because a segment's first draws cost ~2x (cold caches).
- Threads: 4 recorders with >= 8 hardware threads, 5 with >= 16 (VR:2662-2669); replay thread records one (`SELF`).
- `RX_VK_SEG_EARLY` hands segments out while the dry pass still runs (VR:2931-2948, 2984-3290); `RX_VK_EARLY_GPU` submits
  the work before the dry pass and each finished segment in order (VR:2949-2970, 3048-3097).

### 1.5 Measured (desktop x86 + RTX/3050)

| Change | Result | Source |
|---|---|---|
| Parallel replay on (4 threads), after the readback-refresh fix | Kokiri 54->65 fps, Hyrule 49->62, p99 26->20 / 66->19 ms | VR:2663-2664 |
| First version (2026-11-08) | replay 12.6 -> 8.5 ms, 54 -> 58 fps (60 Hz cap) | VR:2655-2656 |
| Dry pass cost | 0.33-0.48 ms per big command buffer (~800 draws) | BM tick050 |
| Segments vs serial | segments sum ~5 ms vs ~3 ms serial; slowest segment ~1.25 ms whatever the count | BM tick050 |
| ParLock | 13% of worker samples waiting (fixed to 8.7%); first 32 draws of a segment 6.9-7.4 us vs 2.8-3.1 later | BM tick052 |
| ROUNDS+SELF+BALANCE+POINT=32+5 threads (adopted TICK-054) | kokiri 72.3->76.1 fps, castle 68.3->70.2, hyrule 69.9->72.1; frame prep 8.11->6.77 ms | BM tick054 |
| FINE_SEGMENTS | replay thread's own segment 4.7 -> 3.6 ms, 36.0 -> 37.7 fps | VR:2546-2550 |
| EARLY_GPU | 36.45 -> 38.05 fps, GPU idle ~1 ms shorter a frame | VR:2956-2957 |

Lessons: the large win came from the first working version; later tuning gave +2-5%. Contention and cold per-thread caches
inflate draw cost ~1.6x; the replay thread's own serial work (dry pass, pre-sync, its segment) becomes the critical path.

### 1.6 Pitfalls they hit (each a picture bug first)

1. Out-of-order recording left one-frame black scenes (48-93 per walk) until targets were marked in the dry pass
   (VD:11070-11075) and clears too (D4, VR:1306-1311).
2. Washed-out fog / missing shadows: each segment refreshed a readback page and wiped earlier segments' writes (VR:2659-2661).
3. A segment read a mirror before it was filled (houses missing, SW:6540); pre-sync must precede all segments.
4. Uniform/condition list growth while workers read (deque/vector reallocation) needed a lock (VR:1335-1373).
5. Render conditions on unread reports cannot be decided in a dry pass: stop and sync (VR:2287-2291).
6. Driver bug: a command buffer starting with mesh draws (Intel) -> cuts only before plain draws (VR:2570-2583).
7. GPU ASTC decode and DLSS are not allowed during segments (SW:5838, SW:3292).

## 2. Our replay: what recording a draw touches

The interpreter is `exec_ops()` (gfx_vk_exec.c:4639-4831), entered from `mtl_exec()` (4599). The state shadow is the static
`S` (gfx_vk_exec.c:25-42). A draw goes `OP_DRAW` (4746) -> `draw()` (3477) -> `draw_body()` (3519) -> `prepare_draw()` (2969)
-> `begin_pass()` (1960) / `pass_commit()` (1263) -> `eden_draw()` (gfx_vk_eden.inc:1243) -> `eden_payload()` (1109),
`eden_bind_set()` (1181), `set_dynamic()` (gfx_vk_exec.c:2858), vertex buffers (eden 1351-1382). Every vkCmd goes to
`vk_cmd()` = `cur()->main` (gfx_vk_core.c:3668, 3600); uploads to `vk_upload_cmd()` (core:3671), which is submitted with its
part ahead of the part's main buffer (`vk_cmd_split`, core:3621-3667).

### 2.1 Inventory (O = order-dependent, must be decided in stream order; P = per-segment/per-thread)

| # | State | Where | O/P | Stage-1 plan |
|---|---|---|---|---|
| 1 | `S` (bindings, RT set, `pass_open`, `bound_pipe`, `rp_pass`) | exec:25-42 | O (stream state) | Dry pass keeps the master; snapshot at each boundary; worker owns a TLS copy |
| 2 | Open pass: `P_att`, `P_ca`, `P_da/P_sa`, `P_ri`, `P_cleared`, `g_pass_draws`, `g_pass_layered` | begin_pass 1960-2031, pass_commit 1263, 1948-1959 | P | TLS; a segment starts and ends with no pass open |
| 3 | s64 pending barrier `g_s64_pend`, `g_s64_pwrite`, s64a counters | exec:4685, 4776-4785, 1903, 1937 | P within a pass | TLS; a worker's final `vk_end_pass` emits a pending barrier as today |
| 4 | Scale path: `pending_scale_count`, `carry_one_draw`, `pass_scale`, `materialize_scale` | exec:4669-4683, 1088, pass_commit | O (crosses passes) | Stage 1: parallel off when resolution scale is on |
| 5 | Render condition `g_rc_skip` | `render_cond` exec:775, `rc_value` 728 | O | Dry pass evaluates; per-segment result array; worker reads |
| 6 | Reports: `cnt_note`, `rc_note`, `occ_key_make`, `mtl_sub_counter`, queue `g_rq` | exec:4692-4693, 717-728, 4427; core:4124-4216 | O | Dry pass only |
| 7 | Occlusion queries `g_oq`, `g_oq_active`, `s->noq`, `g_oq_epoch`, `g_oq_counting`; `oq_end` posts at report **and at pass end** | core:2501, 4083-4094 | O | Prerequisite R4: one slot per draw by stream serial (HeyListen TICK-033); dry pass emits every post |
| 8 | Post ops `s->post`, `npost`, `wseq` | `post` core:3934 | O | Dry pass only; a worker never posts |
| 9 | `OP_SIGNAL_EVENT`, `OP_FENCE_SYNC` | exec:4691, 4695 | O (CPU only) | Dry pass only; worker skips |
| 10 | Deferred uniform updates `g_ubd`, `g_nubd`, `g_ubd_gen` (append, supersede by memmove, land) | `vk_sub_update` core:4320, `vk_ubd_land` 4363 | O | Dry pass runs `vk_sub_update` (scratch copy made once); logs each list op; worker replays the log on a TLS list; a land ends the segment |
| 11 | In-flight landed updates `g_infl` | core:4355-4361, read in `vk_ubd_read` 4582 | O (pruned by replay thread) | Prune only between parts; workers read a frozen array |
| 12 | Shadow cache `c[NC]` in `vk_ubd_shadow` | core:4594-4602 | P | TLS |
| 13 | `vk_rpt_flush` before CPU reads standing for GPU reads | `vk_ubd_read` core:4583 | O | Dry pass flushes before handing a segment; a queued report range overlapping a UBO the segment binds ends the segment |
| 14 | GPU-write lists `s->gw`, `s->wr`, `wseq`, `g_gw_seq`; `vk_sub_wver()` | `sub_gpu_wrote` core:4392-4414, 4476 | O | No writer inside a Stage-1 segment, so constant per segment |
| 15 | Pool sync / copied pools (`pool_sync_pending`, `pc_pre_note`) | core:2380, 4399, 3633 | O | Only from serial ops and part boundaries |
| 16 | Vertex mirrors (`vbm_bind`, `g_vbm_stale`, `vk_vbm_up_close`) | exec:475, 434; eden:1372 | mixed | Global lock; copies into the part's upload buffer under the lock |
| 17 | Upload command buffer `s->up`, `up_used`, `g_up_gen` | core:3671-3675 | P (one per part, runs before it) | One VkCommandBuffer: every upload sequence under the lock (Stage 2: one per chunk) |
| 18 | Image registry `g_by_key/g_by_serial`, creation, `init_fill`/`zero_fill`/`cpu_first_use`/`pkg_upload`/`lin_copy` | gfx_vk_tex.c:129, `vk_tex` 2178-2265 (map under `g_img_mu` 2192-2243, uploads after unlock 2244-2263) | O (first use decides content) | Dry pass resolves render targets at each draw/clear (R6); sampled lookups in workers: whole `vk_tex` body under the lock |
| 19 | Write marks `im->gpu_w`, `wseq` (`++g_wseq`), `wpres`, `scaled_valid`, `nvnc_image_intent` | tex:2612, 177-187; exec:2602 | O | Dry pass marks targets in order; worker attachment lookups skip the mark |
| 20 | First layout change `im->layout` UNDEFINED->GENERAL | `vk_tex_prepare` tex:1386-1391, callers begin_pass 1974/1995, tex:2362 | O | Check-and-set under the lock, recorded into the part's upload buffer, never into a worker buffer |
| 21 | Draw caches `g_tvc` (2550), `g_setc` (2624), `g_bc` (2846), `g_eden_vbm` (eden:1220), `vlk` (eden:1359), `g_eden_vpm` (1212), `g_s45_st/db/dm/lctx` (eden:261, 390, 1214-1215), `g_s35_*` (1178-1179), static `pay[]` (1260), `g_bind_gen`, `g_ubo_gen` | as listed | P (keyed by `g_cmd_gen`) | TLS; `g_cmd_gen` per thread, bumped when a worker begins a chunk |
| 22 | Descriptor pools `s->dp`, `dp_cur` | `vk_alloc_set` core:3910-3931 | P | Per thread per slot |
| 23 | Scratch `s->blk`, Eden rings `g_eden_uring/g_eden_dring` | `vk_scratch` core:3725, eden:358 | P | Per thread per slot (reset in `slot_recycle` core:3593) |
| 24 | Barriers: s63 `lcb/lwork` (TLS already), `bpend_hold` table (mutex, keyed by cb), `g_bar_seq`, `g_disp_seq/g_disp_cb` | core:4260-4279, 60-76; exec:4071 | P | Already thread-safe or TLS; `g_bar_seq` TLS |
| 25 | Atomic staging window `g_hy3w` and `n1x_atomic` | eden:460-515, 626-698; `vk_hy3_close_for` in every s63 wrapper (core:84-91) | O | Draws calling `n1x_atomic_begin` (two shader keys, eden:476-477) end the segment; dispatches stay serial |
| 26 | Current command buffer | `vk_cmd()` core:3668, `g_slot_main_cb()` | P | TLS override (`tl_cb`) |
| 27 | Part cuts `g_cb_draws`, `g_ord_parts`, `g_psplit_now` | begin_pass 2012-2017, prepare_draw 2970-3013 | O | Dry pass decides; workers never cut |
| 28 | `g_deq_dir` (direction of the last depth-writing test) | prepare_draw 3069, eden:1291 | O (crosses draws) | Dry pass tracks it per draw (pure function of `S.ds`); part of the snapshot |
| 29 | Rasterizer-discard runs (`g_rd_prev` since S1b, `wa[]`, `rd_read`) | prepare_draw 3006-3009 | O, writes memory | Such draws end the segment (serial) |
| 30 | Pipelines `eden_graphics`, `eden_ready_w`, `nvn_eden_cmd_*`; shader flags written at draw time (`vpull_known`, `layer_out`) | eden:224, 277, 1253, 1257 | shared | Audit (open question 3); lock creation, TLS front cache |
| 31 | `OP_CALL` decisions `dm_nvn_call_check`, `OP_JUMP` checks | exec:4689-4690 | O (depends on handle bytes now) | Dry pass records each decision; worker reads them. Boundaries at any depth: segment carries the call-stack position (Stage 1, Boundaries) |
| 32 | Indirect prescan, MDI scan | `ind_prescan` 4528, `mdi_scan` 3446 | O | At list start / in dry pass |
| 33 | Pass timestamps `vk_pass_ts`, ootgap `og_*`, crumbs, `g_r107`, `wb_trace`, bisection `g_bis_*`, paydiag, oplog | core:2862, 2855-2861; exec:820-860 | diagnostics | Parallel refused while any is on (as HeyListen refuses dumps/profiles, VR:3319) |
| 34 | Counters `g_vkst`, `g_r131`, `g_r142`, `g_s64_*`, `eden_prof` | many | P | TLS, summed at part end |
| 35 | Transform feedback `g_tf*`, DLSS, `fg_*` capture, `ndsr_*` | exec:43, 4816-4817, 4733, 1911-1923 | O | Serial ops; parallel off while `fg_frames_left` |
| 36 | `oot_bilerp_rt` | set_rt 4407 | ? | Check what it keeps (open question) |
| 37 | Content ledger `nvnc_part`, `nvnc_accept`, `nvnc_range_intent` | core:3644, 3663, 4395 | O | Part-level calls stay on the replay thread |
| 38 | Copies, fills, clears of textures, dispatches | exec:4696-4697, 4811-4814, 4773 | O (GPU writes) | Serial ops on the replay thread in Stage 1 |

## 3. Proposed design

Principles: one code path. The same `exec_ops` runs in three modes (`RM_SERIAL`, `RM_DRY`, `RM_WORKER`, TLS `g_rmode`). No
behaviour toggles beyond the rollout gate `DM_VK_PARALLEL=<workers>` (default 0 = today's serial path, bit for bit). Every
prerequisite refactor (R1-R8) also runs, and is tested, in serial mode before parallelism is turned on.

### Stage 0: measurements (no behaviour change)

0.1 Where the 15 ms go: `DM_VK_REPLAYDIAG=1` already gives per-op time ("record ops, ms per frame", exec:4592-4596), draw
    payload per binding kind (4587-4591), and `eden_prof` (eden:1225-1242). Need the split: (a) state ops + reports +
    conditions + updates (dry-pass cost), (b) `eden_draw` (worker cost), (c) serial ops (copies, dispatches), (d) part
    submit/begin. Decision rule: Stage 1 is worth it only if (b) >= ~60% of the 15 ms.
0.2 Shape of the stream: draws per pass (s64a histogram, exec:4794-4797 and s61_log), length of runs of worker-safe ops
    between serial ops (copies/dispatches/storage-writing draws), draws inside `OP_CALL` (depth > 0), reports and render
    conditions per frame, storage-writing graphics pipelines (`ep->s64w == 2`, eden:1248-1250), `n1x_atomic` draws,
    timestamp reports through `g_cq`. A header-only walker (like HeyListen's `count_work`, VR:3304) logged per submission.
0.3 TLS cost probe: build with `S` and the item-21 caches as `_Thread_local` and compare serial replay time (HeyListen paid
    2.4% for C++ TLS init; C `_Thread_local` on clang/Windows ARM64 is a TEB load + index, expected < 1%).
0.4 CPU budget: which cores are idle while OoT runs (N1X hybrid P/E: the memory note says Adreno-era defaults need a
    re-check); measure one worker thread pinned to a P-core vs an E-core recording the same segment.
0.5 GPU idle timeline: `DM_VK_PASS_TIMES` / ootgap gaps (core:2855-2869) to confirm the 6 ms idle is between parts.

**Measured (par0, `DM_VK_REPLAYDIAG`, OoT heavy scene, 2026-11-21):**

| Quantity | Value |
|---|---|
| Draws per frame | 3,186 |
| Draws inside an `OP_CALL` (depth > 0) | 99.9% |
| Draws in runs of >= 48 draws between serial ops | 90-96% |
| `eden_draw` share of replay | 12.2 of 21.8 ms (56%) |
| Draws per pass | most passes hold 1 draw; ~0.9 passes per submission hold 64+ draws |

Consequences: (1) depth-0-only boundaries would give no segments at all, so Stage 1 boundaries must sit inside `OP_CALL`
frames (below); (2) runs are long enough for `SEG_MIN` = 48; (3) `eden_draw` at 56% is just under the 60% rule of 0.1, but
the worker also takes `prepare_draw`/`begin_pass`/state ops, so the parallel share is higher than 56%; (4) one big pass
per submission holds a large share of the draws, so in-pass splits (the `g_psplit` rule) are needed from the start, not
only `SET_RT` boundaries.

### Stage 1: minimal viable parallelism

**Unit.** A submission part (between `vk_cmd_split` cuts) becomes an ordered list of *chunks*, each a primary command
buffer: `[up] R0 W1 R1 W2 ... Rn`. `R` chunks are recorded by the replay thread (serial ops and anything not worker-safe),
`W` chunks are segments recorded by workers. At the part's end the replay thread waits for its `W` chunks (helping with
queued ones, HeyListen `SELF=2`) and submits all chunks in one `vkQueueSubmit` in order, exactly where `vk_cmd_split` /
`mtl_sub_commit` submit today (core:3634-3651, 4850+). Secondary command buffers are not used in Stage 1:
`vkCmdBeginRenderPass` must be in a primary, `vkCmdExecuteCommands` needs ended secondaries (the primary could not go on
until each worker is done), and Eden passes record inline queries/clears/depth-bias inside the pass. On an immediate-mode
NVIDIA GPU the extra pass ends at segment boundaries are cheap (HeyListen measured on RTX; our own `g_psplit` already splits
passes at draws, prepare_draw:3011).

**Boundaries.** The dry pass may open a segment at any op where no pass needs to stay open:
- right before an `OP_SET_RT` that is not `rt_same` (set_rt:4397-4398 ends the pass anyway), or
- before a draw where a pass split is legal today (no scale pending, `S.pass_open` would be 1, not a layered change),
  i.e. the conditions of `g_psplit` (prepare_draw:3011-3013) minus the command-buffer limit;
- at any `OP_CALL` depth (par0: 99.9% of draws are at depth > 0). The dry pass walks the CALL stack as `exec_ops` does
  (`dm_nvn_call_check` per CALL, JUMPs followed with `dm_nvn_jump_ok`) and keeps, per depth, the record pointer to resume
  at: `pos[0..d-1]` = the record after each open CALL (return points), `pos[d]` = the segment's first record in the
  innermost list. The segment carries `{d, pos[0..d], end, end_depth}`, `end` being the first record not in the segment
  (in whichever list it lies, depth `end_depth`, which may be lower or higher than `d`). Depth limit as today (8).
- only if the header-only lookahead finds >= `SEG_MIN` (48) worker-safe draws before the next serial op.
A segment ends at ~`SEG_TARGET` (128) draws at the next legal boundary, at the first op that is not worker-safe, at the
part's cut, or at a `vk_sub_update` that would land (`g_nubd == UBD_MAX`, core:4327).

**Worker-safe ops** (everything else ends the segment): all pure state ops (the `S64_ALLOW` set, exec:4615), `OP_SET_RT`,
`OP_CLEAR_COLOR/DS`, `OP_BARRIER`, `OP_DRAW`, `OP_DRAW_IDX`, `OP_MULTI_DRAW` (direct or indirect, args only read),
`OP_UPDATE_UB`, `OP_REPORT_COUNTER` (samples-passed and "not measured" types; not `g_cq` timestamps), `OP_RESET_COUNTER`,
`OP_RENDER_COND`, `OP_SIGNAL_EVENT`, `OP_FENCE_SYNC`, `OP_JUMP`, `OP_CALL` (decision from the dry pass). Excluded draws:
pipelines that write storage (needs a per-program flag known without building the pipeline: R7), rasterizer-discard runs,
the `n1x_atomic` shader keys, transform feedback, `OP_DRAW_TEX`, `OP_DLSS`.

**Resuming inside nested lists.** A worker runs `exec_ops_at(seg, 0)`: at depth i < d it calls
`exec_ops_at(seg, i + 1)` first (the inner list from `pos[i+1]`), then goes on in list i from `pos[i]` (the record after
the CALL); at depth d it starts at `pos[d]`. Every list walk checks `p == seg->end` before each record (one compare, TLS
`g_seg_end`, NULL in serial mode) and on a hit sets `g_seg_done`, which every enclosing frame returns on, so the worker
unwinds without running the rest of any outer list. CALLs the worker meets inside the segment use the dry pass's decision
array; JUMPs are followed with `dm_nvn_jump_ok` exactly as `exec_ops` does (a failed check ends that list, as today, and
the dry pass saw the same result). `full_barrier_current` starts at 0 in each resumed frame (conservative). The per-list
work of `mtl_exec` at depth 1 (`g_s45_kgen`, `ind_prescan`, `ind_args_window`) stays on the replay thread.

**Dry pass (replay thread, `RM_DRY`).** Same switch as `exec_ops`; per op:
- state ops, `SET_RT`: as today on `S` (no pass is opened, no vkCmd recorded);
- draw / clear: `g_occ_draws++`, render-condition skip, draw serial `++g_draw_serial`, `g_deq_dir` update, worker-safety
  check, boundary bookkeeping, and `dry_resolve_targets()` (R6) when the target set changed since the last mark;
- `REPORT_COUNTER`, `RESET_COUNTER`, `SIGNAL_EVENT`, `FENCE_SYNC`, `RENDER_COND`: the full CPU work of today (they record no
  GPU command once R4 is in); `RENDER_COND` result appended to the segment's array;
- `UPDATE_UB`: `vk_sub_update` as today, plus one log entry `{dst, n, sb, so, p, superseded index}` for the segment;
- `OP_CALL`: `dm_nvn_call_check` decision appended to the segment's array, then descend (push the return point);
  `OP_JUMP`: `dm_nvn_jump_ok`, replace `pos[depth]`; list end: pop.
At the segment's end the dry pass publishes the job: call-stack position `{d, pos[]}`, end record and its depth, `S` snapshot, `g_ubd` snapshot (<= 512
entries, ~20 KB), its update log, condition and call arrays, first draw serial, `g_deq_dir`, and `g_rq` flushed
(`vk_rpt_flush`, core:4540) so the worker's CPU reads see every earlier report.

**Worker (`RM_WORKER`).** Begins its primary from a per-thread command pool of the slot (`nvn2_begin`, TLS `tl_cb`,
`g_cmd_gen++`, TLS caches invalid), records `vk_img_barrier_all` (the boundary's dependency, as HeyListen's
`full_barrier`, VR:2799), then runs `exec_ops` from the start pointer on its TLS `S`: draws and passes exactly as today;
`UPDATE_UB` applies the next log entry to its TLS list; `RENDER_COND` / `OP_CALL` read the arrays; reports, events, fence
syncs are skipped; draws wrap their occlusion query with slot = draw serial (R4). At the end: `vk_end_pass`,
`vk_img_barrier_all`, `vkEndCommandBuffer`, sum TLS counters into the part, signal done.

**Replay thread between segments.** After handing out segment k it opens a new `R` chunk, continues in `RM_SERIAL` for
serial ops (its `S` is the dry pass's, `pass_open` 0), and may open the next segment immediately. `vk_end_pass_land`
in serial ops lands `g_ubd` after the segment's draws in queue order, as today.

**Prerequisite refactors (each first landed in serial mode):**

*Stage 1 first version landed as S1e (2026-11-21), not run yet.* Done = landed in serial mode, syntax-checked; R1-R3, S1b, S1c, S1d/S1d2 verified in game. S1e checklist
(details in the S1e section): [x] workers record segments (`DM_VK_PARALLEL=<n>`, dry pass implied) with strict eligibility; [x] chunks in each part's submit in stream order
(`par_cbs`, R chunks the replay thread's); [x] frozen GPU-write view, frozen `g_infl`, no `vk_rpt_flush` on workers; [x] R6's resolution and marks at the hand-out, workers' attachment
lookups none; [x] ootbilerp's kept targets, r150's fresh copies, `g_occ_draws` / `g_cb_draws` at the hand-out; [x] occlusion queries of a worker's passes kept exactly (frozen epoch,
atomic slots, post ops back in the segment's place); [x] image objects kept while segments are out; [x] P-core pinning; [ ] R4 (samples-passed reports in segments: still the
largest refusal reason expected); [ ] report and event commits at the hand-out; [ ] per-site lock counters; [ ] Stage 2 (early chunk submission, balancing).
- [x] S1d: reports' `cnt_note`/`rc_note`, resets' `cnt_note`, fence syncs, render conditions (stop rule), timestamp query slots, `UPDATE_UB`'s range check and scratch copy
  done by the dry pass and taken by the replay (`g_sd.dec`); full snapshot per segment (`seg_snap`) checked in SERIALSEG; segment-local `g_ubd` lists with a supersede log (R5's
  list part); part cuts placed by the dry pass. [ ] The commits that stay the replay thread's at the record (report queue / post ops, events), occlusion (R4), frozen `g_infl`
  and GPU-write view (below, "Remains for Stage 1 proper").
- [x] R1/R1b recording contexts (`g_rec`, `g_rec_e`). [x] S1c: `g_vkst`, `g_r131`, `g_set_hits/new`, s64a / s64 / s65 / cbar counters per context, folded
  into the replay thread's (`vk_rec_ctx_fold`). [ ] Still global: the bundle-path diagnostics' counters (`g_r132/g_r133/g_r135/g_r142/g_r151/g_r152/g_r143`;
  bundle path only, parallel is Eden mode only), order-dependent `g_sub_pass`, `g_draw_i`, `g_cb_draws`, `g_psplit_now` (items 27, 31: the dry pass's).
- [x] R2 recording target `g_vk_rt`. [x] S1a: worker command pools and primaries per slot, chunk list per slot. [ ] Chunks in the part's submit (ordered); `MAIN_CBS` limits.
- [x] R3 `struct vk_arena`, Eden rings behind `g_rec_e`. [x] S1a: worker arenas per slot, rewound in `slot_recycle`; worker ring pair; `vk_slot_stats` / `scratch_trim` /
  `vk_buf_size` over them.
- [x] S1c: shared mutable state in the Eden entry points (S1a audit below): per-thread `g_vpull_*`, `lop_kept`/`df_kept` into the stream state (`rec_xs_t`),
  eden_ubo's c1 and shadow buffers, get-or-insert for `eden_pipe`, once-publication of `e->pi`/`e->ready`, draw-time shader-object writes, counters; the library's
  `CopyCode`. [ ] The dry pass computing the `[S]` fields of `rec_xs_t` (snapshot per segment); the GPU-write list frozen per segment (S1c, "Remains").
- [ ] R4 occlusion queries by draw serial: deferred, needs the in-game A/B of (e).
- [x] R5 uniform-update log, the list part (S1d): `struct vk_ubl {gen, n, e[]}` behind a TLS pointer, `g_ubd_gen` a macro over the thread's list's generation (one counter for
  all lists), `vk_sub_update_pre` / `vk_sub_update_apply`, segment lists with a logged supersede index. [ ] Frozen `g_infl`, no flush on workers (`vk_ubd_read`).
- [x] R6 resolve step `pass_resolve_targets` split from `begin_pass`. [ ] Its dry-pass caller; `pass_commit`'s attachment marks
  (`wb_use(WB_ATTACHMENT)`, `scaled_valid = 0`) moved to the dry pass or shown idempotent; a "no mark" role for workers' lookups.
- [x] R7 per-shader storage-write flag `vkshader_t.eden_sw` (`eden_prog_sw`, `eden_cs_sw`; S1b: the dry pass reads `eden_prog_sw`). [ ] Unknown (0) at a program's first draw:
  that draw stays serial; a pre-build answer needs a bridge call (below).
- [x] R8 locks `vk_par_lock` (outer) / `vk_up_lock` (leaf). [x] S1c: `eden_graphics` / `eden_compute` get-or-insert and once-publication; the library's shader
  code publication (`CopyCode`); draw-time writes to shader objects atomic / once. [ ] Per-site take/wait counters; pipeline creation still serialized by the
  library's `nvn_mutex` (noted, not redesigned: S1c).
- [x] S1b: the dry pass walks each top-level list as `exec_ops` does (CALL descent, JUMPs, per-depth return points), applies the worker-safe set and the boundary
  rules, cuts segments inside CALL frames, and owns the CALL/JUMP decisions (the replay reads them); `DM_VK_PARALLEL_SERIALSEG` resumes `exec_ops` at segment
  positions on the replay thread (correctness harness). [ ] Discovery only: the order-dependent CPU work of reports, occlusion keys, events, fence syncs, render
  conditions and uniform updates is still done by the replay, not the dry pass (S1b, "What the dry pass does not do").
- [x] S1c: the `defer` handle copy of `bind_handles_rec` (not idempotent): no longer written into the record; read once in stream order by the dry pass when it walks
  the list, which the replay (and a segment) takes; the cross-record globals grouped in `rec_xs_t` (part of each recording context).
- [x] S1d: order-dependent CPU work of reports, resets, fence syncs and render conditions split from recording (dry pass). [ ] Events and the report commits (post ops,
  rptfix queue) stay at the record on the replay thread (S1d, "What stays the replay's").
- R1 Recording context: move `S`, the open-pass globals (item 2), and the caches of item 21 behind one `_Thread_local`
  context; counters (items 3, 34) per thread.
  First step done (2026-11-21, no behaviour change, syntax-checked only): `struct rec_ctx` (exec: `S`, `g_tvc`, `g_setc`,
  `g_drvc`, `g_bc`, `g_s45_kgen/list`) and `struct rec_eden` (eden: `g_eden_lk/lp`, s45/s35 memos, `g_eden_vpm`, `g_eden_vbm`,
  `vlk`, both `pay[]`), each one static instance reached through a `_Thread_local` pointer (`g_rec`, `g_rec_e`);
  `g_vk_storage_want`, `g_storage_ro` `_Thread_local`. Still global: item 2 open-pass globals, `g_an`/`g_ir`, counters.
  R1b (2026-11-21, no behaviour change, syntax-checked only): into `struct rec_ctx` the open pass (`P_ca`, `P_da`, `P_sa`,
  `P_ri`, `P_cleared`, `P_att`, `g_pass_draws`, `g_pass_layered`, `g_pass_bb`; `P_ri` points at its own context's `P_ca/P_da/P_sa`)
  and the run memos `g_mg` (r142), `g_an` (r152), `g_ir` (r151); `g_bc`'s type named `rec_bc_t`. Into `struct rec_eden` the
  descriptor-buffer memo of `eden_bind_set` (were function statics) and the counters `g_s35_skip/push/ns`, `g_s45_nk/ns/nd/bad`,
  `g_vlk_hit/miss` (the logs read the replay thread's instance; a worker's would have to be summed in later). `pass_commit`,
  `begin_pass`, `vk_end_pass`, `prepare_draw` read `g_rec` once into a local. Left global: the replaydiag accumulators
  (`g_rd_ns/g_rd_n`, `g_ss_*`: extern, shared with core, written only while `DM_VK_REPLAYDIAG` is on, which item 33 already
  excludes from parallel mode), `g_vkst`, `g_s64_*`/s64a, `g_r142/g_r151` counters (item 34, still to do), and the order-dependent
  `g_sub_pass`, `g_draw_i`, `g_cb_draws`, `g_psplit_now` (items 27, 31).
- R2 `vk_cmd()` / `g_slot_main_cb()` return the thread's current chunk; chunks per slot per thread from their own
  `VkCommandPool` (a pool is externally synchronized); `MAIN_CBS` (core:2488, 32) grows into a chunk list.
  First step done (2026-11-21, no behaviour change, syntax-checked only): `vk_rec_target {cb, gen}` (gfx_vk.h) and
  `_Thread_local vk_rec_target *g_vk_rt`, which in every thread points at the replay thread's `g_vk_rt_main` (core).
  `vk_cmd()` and `g_slot_main_cb()` return `g_vk_rt->cb` (`vk_cmd` still dies when it is NULL, as `cur()` did outside a
  submission); `g_cmd_gen` is a macro for `g_vk_rt->gen`, so every memo check (`g_bc`, s35 `pgen`, s45, `g_eden_vbm`,
  `vlk`, `g_mg/g_an/g_ir`, the masked clear's and r129's bumps) uses the thread's target. The core keeps `g_vk_rt_main`
  equal to `cur()->main` explicitly (not through `g_vk_rt`): `mtl_sub_begin` (at `g_cur =` and at `s->main = mains[0]`, gen++
  after the begin), `vk_cmd_split` (new part, gen++), the commit's empty-recovery begin (cb, gen++), and the commit's end
  (`g_cur = -1` -> cb NULL). Part-level recording in core (`s->main` for query resets, timestamps, pool sync, ootgap) stays on
  the slot's buffer. No worker pools yet: a worker would set `g_vk_rt` to its own target and bump its own gen at each begin.
- R3 Per-thread per-slot arenas: descriptor pools (`vk_alloc_set`), scratch blocks (`vk_scratch`), Eden rings; recycled in
  `slot_recycle`.
  First step done (2026-11-21, no behaviour change, syntax-checked only): `struct vk_arena` (core: `blk/nblk/capblk`,
  `scr_peak/scr_age`, `dp/ndp/capdp/dp_cur`, `dp_used/dp_sets/dp_counted`, were fields of `struct vksub`), one per slot as
  `s->ar`. `vk_rec_target` (gfx_vk.h) gained `struct vk_arena *ar`; `rec_arena()` returns `g_vk_rt->ar`, or when NULL (every
  target so far, `g_vk_rt_main` included) `&cur()->ar`, so it still dies outside a submission. `vk_scratch`, `vk_alloc_set` and
  `vk_buf_size` allocate from / search `rec_arena()`; `slot_recycle`, `scratch_trim` (now takes an arena) and `vk_slot_stats` work
  on `s->ar`. Eden: `EDEN_CHUNK`, `eden_chunk`, `eden_ring`, `g_eden_dring/g_eden_uring` moved above `struct rec_eden`, which
  gained `dring`/`uring` pointers (`g_rec_e_main` points at the two shared rings); `eden_addr_copy`, r75's `g_drv_lb/lo` (read
  from the same ring the copy came from), `eden_bind_set`'s descriptor buffer and the vertex pull table allocate from
  `g_rec_e->dring/uring`. Hot-path cost: one TLS pointer load and a branch per allocation. To do with workers: a worker's arena
  per slot (own descriptor pools and scratch blocks, rewound in that slot's `slot_recycle` next to `s->ar`; `vk_buf_size` then
  searches the thread's arena only, which is what its callers want), a worker's own `eden_ring` pair (chunks tagged with
  `mtl_last_fence() + 1` as now, so the reuse rule holds per ring), and `vk_slot_stats` summing the workers' arenas.
- R4 Occlusion queries by draw serial: each counted draw `vkCmdBeginQuery/EndQuery` with slot = serial in a per-slot ring;
  `mtl_sub_counter` posts the report with its serial window (reset .. report) and the completion thread sums the slots.
  Removes slot allocation and posts at pass end (`oq_end` in `vk_end_pass`, exec:1907). One path for serial too.
  Not done (2026-11-21): identical occlusion results cannot be shown without an in-game A/B. Today a query (`vk_oq_draw`, core)
  is begun at the first counted draw of a pass after a reset/report and ended at the pass end (`vk_oq_pass_end` in
  `vk_end_pass`; also before the masked clear's draw, exec:2169), a report (`mtl_sub_counter`, the occkey path too) or a reset
  (`vk_oq_reset`); one kind-4 post per query carries `{epoch, q}`; the completion thread sums kind-4 results per epoch
  (`oq_sum`, `oq_lost`) and the next SAMPLES_PASSED report (kind 2) takes the sum (occkey `occk_put`). What R4 needs to keep the
  sums equal:
  (a) the per-draw query must wrap exactly what the pass-span query counts: everything recorded in the pass between
  `vk_oq_draw` and the span end, i.e. the draw's own `vkCmdDraw*` and also `mdi_draw` runs, r131 duplicate draws, xfb draws, the
  mesh path (exec:3560), and any in-pass helper draw recorded while a span is open (audit: such a draw is counted today and
  would not be per draw); a draw with `g_ok == 0` gets a zero query or none, sums equal;
  (b) `VK_QUERY_CONTROL_PRECISE_BIT` (`g_oq_flags`): without it a query may return any nonzero value for nonzero samples, so N
  per-draw values sum differently from one span: R4 must keep span queries (or refuse parallel) when not precise;
  (c) capacity: `OQ` = 4096 queries per slot counts passes today (DK overflowed 512); per draw it must hold one submission's
  counted draws (OoT ~2,300 a frame over 6 submissions; measure the max per submission in DK and OoT), else lost queries turn a
  report into `1` / "some lost" where it is exact today; the completion thread's stack array `occlusion[OQ]` must move to the heap;
  (d) posts: one post per window `{epoch, q0, q1}` (first and last serial of the epoch in this submission) instead of one per
  query, so post count and completion cost stay as today; the completion sums `occlusion[q0..q1)` and keeps the `oq_lost` rule;
  (e) validation before adoption: log every report (address, value) with span queries and with R4 over the same frames at
  `DM_VK_PARALLEL=0`, require identical values and an equal occkey table (`occk_put`); measure the GPU cost of ~2,300
  begin/end pairs a frame on N1X.
- R5 Deferred uniform updates as data: `g_ubd` operations reproducible from a log; `g_infl` pruned only between parts.
  Analysis (2026-11-21). Only code change: `vk_ubd_shadow`'s cache `c[NC]` is `_Thread_local` (item 12; no behaviour change while
  only the replay thread builds shadows). The rest is not mechanical, so not done yet:
  - The list (`g_ubd[UBD_MAX]`, `g_nubd`) has four mutators: append and supersede in `vk_sub_update` (an exact-range match removes
    the older entry by memmove, so indices shift), land in `vk_ubd_land` (records copies and barriers into the command buffer and
    `sub_gpu_wrote` per entry: a GPU-write-list change, i.e. a serial op), drop in `vk_ubd_land_drop` (the commit's empty-recovery
    paths). A full list (`g_nubd == UBD_MAX`) lands from inside `vk_sub_update` through `vk_end_pass_land`, which also ends the
    pass: the dry pass must end the segment there.
  - The log a worker replays: `{dst, n, sb, so, p, superseded index or -1}` per `UPDATE_UB`. The `vk_scratch` copy of the update's
    bytes is made once on the dry pass (in the replay thread's arena of the slot, kept until its recycle, so workers may point at
    it). A worker starts from the `g_ubd` snapshot at its boundary and applies the log in order.
  - `g_ubd_gen` (extern; compared by the `g_setc` / r142 / s45 `m->ubd` memos and the shadow cache) bumps on every append, land and
    drop. Per thread it only has to be monotonic within that thread's memos: a worker's counter started from the snapshot's value
    and bumped per applied log entry keeps the equality checks exact. One struct `{g_ubd, g_nubd, g_ubd_gen}` behind a TLS pointer
    touches the extern declarations in exec (2703, 2735) and the s45 memo (eden:445).
  - `g_infl` (landed updates still in flight) is appended by `vk_ubd_land` and compacted by `infl_prune` at every land: with
    workers reading it in `vk_ubd_read` that is a reallocation/compaction under their feet. Needed: append at lands only (serial),
    prune only between parts (part cut, commit), never while segments are out; workers read a frozen `{ptr, count}`.
  - `vk_ubd_read` also flushes queued report writes (`if (g_nrq) vk_rpt_flush()`) and checks `vk_gpu_pending` (`g_gw`): both
    order-dependent; the dry pass flushes before handing out a segment (item 13) and a worker must not flush (it would race the
    replay thread's queue). It reads guest memory at worker time (open question 12).
  - Landing order: updates land at serial pass ends (`vk_end_pass_land`) after the segment's draws in queue order; a worker never
    lands, and a land inside a segment's range ends that segment.
- R6 `dry_resolve_targets()`: the `vk_tex_view(..., role 1)` lookups of `begin_pass` (1972, 1993) made at the first draw or
  clear after a target change, in stream order, with creation, marks and the first layout change; workers' attachment
  lookups then hit and skip marks (`tex_view` with a "no mark" flag).
  First step done (2026-11-21, no behaviour change, syntax-checked only): `pass_resolve_targets(cb, col[8], &dep)` (exec, before
  `begin_pass`) does, in stream order, each set colour target's `vk_tex_view(role 1)` (image found/made, first-use upload or
  fill, marks `note_image_write`, `wseq`, `wpres`, `gpu_w`, `vk_tex_gpu_mark`) and its `vk_tex_prepare(cb)`, then the depth
  target's; it returns `{view, image, vk_view_fmt, vk_view_layers}` per slot (`pass_tgt`). `begin_pass` calls it first and its
  loops (record step: `S.cfmt`, attachment infos, sizes, dup check, `P_att`) read the results. The tex-side calls and their order
  are exactly as before; what moved after them is only exec-local state (`S.cfmt[i] = UNDEFINED`, `ca[i]`) and two rate-limited
  log lines (r76 dup slot, "pass samples"), which now print after the depth lookup's logs. (An image dropped by a later slot's
  `evict_overlaps` was already used after that lookup before, through `P_att[i].im`.) Still to do for Stage 1: (a) the dry pass
  calls it at the first draw/clear after a target change and records the prepares into its `R` chunk; a worker's `begin_pass`
  then sees views made and layouts `GENERAL` (`vk_tex_prepare` skips) but still runs the marks again: harmless for
  `gpu_w`/`vk_tex_gpu_mark` (idempotent), not for `wseq = ++g_wseq` and `note_image_write` (NVN2 `image_version_advance`, off in
  parallel mode; `nvnc_image_intent`): needs a `VK_VIEW_ATTACHMENT_NOMARK` role, or the worker uses the dry pass's `pass_tgt`
  directly (preferred: no second lookup at all); (b) `pass_commit`'s `wb_use(P_att[i].im, WB_ATTACHMENT)` and
  `scaled_valid = 0`, and the clears' `wb_use(WB_OTHER)` / `vk_tex_content_changed` (`clear_whole`), move to the dry pass
  (they are marks, not commands); (c) `materialize_scale` and scale paths stay excluded (parallel off with scale on).
- R7 Per-program "writes memory" flag (storage buffer written, storage image, transform feedback), from translator
  reflection at program bind, so the dry pass can exclude such draws without building the pipeline.
  First step done (2026-11-21, no behaviour change, syntax-checked only): the bridge has no per-shader reflection call (only
  `nvn_eden_pipeline_info.binds` of a built pipeline, keyed by the whole draw state), so the flag is cached per shader object:
  `vkshader_t.eden_sw` (gfx_vk.h; 0 not known, 1 none, 2 writes; sticky at 2). `eden_pipe_sw` is s64w's rule (IMG, IMG_TBUF,
  SSBO `is_written`), now a function; at a pipeline's first draw (where s64w was computed) `eden_sw_note` sets each stage's
  shader from its binds (`nvn_stage`); a writing bind of no present stage marks every shader of the program. Compute:
  `eden_dispatch_prep` sets the compute pipeline's `s64w` and the shader's `eden_sw` (compute `s64w` read nowhere else). Queries
  for the dry pass: `eden_prog_sw(S.sh)` (2 / 1 / 0 unknown), `eden_cs_sw(cs)`; nothing reads them yet. Transform feedback is
  not a bind: the dry pass checks `g_tf_on && S.sh[0]->xfb_layout` itself (as `eden_fill_state`). A program's first draw is
  unknown (0) and must stay serial; to know it before any pipeline, the bridge would need `nvn_eden_shader_writes(sh)` from
  Eden's `Shader::Info` (storage buffers written, image writes, global stores) for the shader's translation: a library change.
- R8 One global recursive lock `g_par_mu` (active only while workers run, as `ParLock`), taken by: the whole of `vk_tex`
  (including its upload tail 2244-2263), `vk_tex_prepare`, `vbm_bind`, any recording into the upload buffer, pipeline
  creation in `eden_graphics`, writes to shader objects at draw time. Its takes and wait time counted per site from day one
  (HeyListen needed this, BM tick052).
  First step done (2026-11-21, no behaviour change, syntax-checked only). One lock cannot do it: the pool-copy uploads record into
  the upload buffer under the pool lock (`pc_pre_note` -> `pc_pre_range`, `pool_make_0` -> `pc_upload`), while `vbm_bind` and
  the `vk_tex` tail take the pool lock (`vk_pool_of` / `vk_buf_at`, which may `pool_make` under the write lock) while they would
  hold it: an inversion. So two locks, each recursive per thread (TLS depth over a plain pthread mutex; one uncontended
  lock/unlock at the outermost take), core `vk_par_lock/unlock`, `vk_up_lock/unlock`, order
  `vk_par_lock -> pool lock -> g_img_mu -> vk_up_lock`:
  - `vk_par_lock` (outer): `vk_tex`'s tail after `vk_sub_open()` (s2tile refresh, `lin_copy`, packaged / `cpu_first_use` /
    `init_fill` / `zero_fill`: the image's upload state) — not its hash lookup (already under `g_img_mu`) nor `vk_tex_view`'s
    view cache (`g_view_mu`), nor exec's `g_tvc` hits, which never reach `vk_tex`; `vbm_bind` whole (wrapper over `vbm_bind_0`:
    `g_vbm` is realloc'ed by `vbm_get`, so even its hit path reads shared state); `vk_vbm_gpu_wrote`. The tail's bak86 `g_img_mu`
    takes are under it (NVN2 only, parallel off).
  - `vk_up_lock` (leaf; nothing inside takes the pool lock, `g_img_mu` or `vk_par_lock`): from `vk_upload_cmd` to the last command
    recorded into the upload buffer, with its `g_up_gen` memos, in `pc_upload`, `pc_stage_upload`, `pc_stage_regions`,
    `pc_pre_range` (core), `lin_copy`, `zero_fill`, `pkg_upload_s` (from the upload buffer on: scratch, dirty protection and
    deswizzle inside), `img_upgrade` (under `g_img_mu`), `vk_dummy_view`, `vk_tbuf_rows_view`, `vk_tbuf_copy_view` (tex),
    `vbm_bind_0`'s copy and `vk_vbm_up_close` (exec); and `vk_tex_prepare`'s slow path (layout checked again under it).
  - Left: `s2tile_upload_tile/tail` record under the outer lock only (NVN2 only; their static buffer is the replay thread's); the
    commit's upload-buffer close (core, after the workers are joined); a debug check in `vk_upload_cmd` that the leaf lock is
    held once workers exist; per-site take/wait counters (with the workers, not before: no measuring now); pipeline creation in
    `eden_graphics`/`eden_compute` (their maps are under `g_eden_mu`; whether `nvn_eden_graphics_pipeline` and
    `nvn_eden_pipeline_get` may run on two threads is the open question of the library); draw-time writes to shader objects
    (`vpull_known`, the pkm2 `layer_out` mark, `eden_sw`, `ep->s64w`: same-value writes, to be made atomic or moved to the dry pass).

**S1a (parallel recording): worker infrastructure (2026-11-21, syntax-checked only; inert unless `DM_VK_PARALLEL` > 0).**
- `DM_VK_PARALLEL=<n>` (0..6, default 0) read in `slots_init` (`par_start`, core). At 0 nothing is made: no thread, pool, context or ring; `slot_recycle` loops
  over the slot's own arena only and takes no new lock; `vk_slot_stats` adds nothing. The one change on the default path: `vk_scratch` builds a new block in a
  local and appends it under `g_ar_mu` (rare path, uncontended).
- Per worker (`struct vk_worker`, core): thread `dm-vk-rec-<i>` (no affinity yet: Stage 0.4 decides), `vk_rec_ctx_new()` / `vk_rec_eden_new()` (exec: zeroed
  `rec_ctx` as `g_rec_main`; `rec_eden` with `g_rec_e_main`'s initial values and its own `eden_ring` pair, same usage flags, chunks made on first use by
  `eden_ring_alloc` and tagged `mtl_last_fence() + 1` as for the shared rings), its `vk_rec_target rt`; in its thread `g_vk_rt = &rt` and
  `vk_rec_ctx_bind(rec, rec_e)`. Per slot (`vk_par_slot`): a `VkCommandPool` (RESET_COMMAND_BUFFER_BIT as the slots' pools, for `nvn2_begin`'s retry), primaries
  allocated as used (`used`/`ncb`), a `vk_arena`.
- Jobs: `vk_par_run(w, fn, arg)` (ring of 64, mutex + condvar, blocks when full), run in order on worker w; `vk_par_drain()` waits for every queue to empty.
  On the worker: `vk_par_chunk_begin(slot)` takes the slot's next primary, begins it, sets `rt = {cb, gen + 1, &slot arena}`; `vk_par_chunk_end()` ends it and
  clears `rt.cb`/`rt.ar` (so `vk_cmd` dies between chunks; `rec_arena()` would fall back to the slot's arena there: nothing may allocate on a worker between chunks).
- Chunks: `vk_chunk {cb, worker, done}` (gfx_vk.h), per slot `chunk[]`, `nchunk`, `chunk0` (first chunk of the open part). `vk_chunk_add(worker)` (replay thread,
  grows under `g_chunk_mu`), `vk_chunk_finish(slot, i, cb, ok)` (worker; broadcast), `vk_chunk_wait_part()` (waits for chunks `chunk0..nchunk`, then the next part
  starts after them; returns the failed count), `vk_chunk_list()`. Submission unchanged: nothing calls these but the self-test.
- `slot_recycle`: for the slot's arena and each worker's arena of that slot: scratch_trim (NVN2), blocks rewound, descriptor pools reset; each worker's pool of the slot
  `vkResetCommandPool(…, 0)` when it gave out a primary; the chunk list emptied. `g_scr_hw` is the sum over the submission's arenas. Valid because the slot's fence
  has passed and workers record only into the open slot (Stage 1 waits for every chunk of a part before the part is submitted).
- `vk_slot_stats`: blocks, bytes, descriptor pools and primaries of the workers' arenas added (o[0], o[1], o[7], o[3]). `vk_buf_size`: the thread's arena first; on a
  worker then the slot's own arena under `g_ar_mu` (the dry pass's scratch copies, R5, are made there).
- `DM_VK_PARALLEL_SELFTEST=1` (with workers): at the first `mtl_sub_begin`, each worker records an empty chunk of the open slot plus one `vk_scratch(256)`, checks
  `vk_cmd() == cb`, `rec_arena()` is its slot arena, `vk_rec_ctx_cur()` its context, `vk_buf_size` finds the block, `vkEndCommandBuffer` succeeds; one line
  `S1a: parallel recording self-test: k of n workers ...` (`[RED]` on a failure). The chunks are dropped from the list, never submitted, reset at the slot's next recycle.

**S1a audit: shared mutable state in the Eden entry points recording uses (step 3 prep; nothing changed).** Library (`nvn_shader_bridge.cpp`):
- `nvn_eden_graphics_pipeline` / `nvn_eden_compute_pipeline`: Eden's `PipelineCache::graphics_cache` / `compute_cache` and `nvn_pools` under `nvn_mutex`
  (vk_pipeline_cache.cpp:1337, 1375): thread-safe, but a miss holds the lock for the whole `CreateGraphicsPipeline` (environments, and translation when not
  async), so one thread's miss stalls every other thread's miss; per-shader `hash`/`analyzed_size`/`lowered` under `nvn_eden_shader::lower_mutex`
  (`PrepareProgram`, `KeepLowered`); `g_records`, `g_records_*`, `BridgeCoverage` statics under `g_pipe_mutex` (`Record`); `EnsureTables` `call_once`;
  `FillNvnRegs` target `regs` is `thread_local`; the environments call back into the host (`eden_q_cbuf` reads the calling thread's `S`, `eden_q_texture` reads
  the texture table through `vk_tex_by_id`: to check under the lock rules); recipes queued to Eden's serialization thread. `nvn_eden_shader_set_code` /
  `CopyCode` assigns `source.code` outside `lower_mutex`: unsafe if a shader's code changes while another thread builds with it.
- `nvn_eden_pipeline_get`: reads the `PipelineRecord` (immutable after `Record`, stable `unique_ptr`), `IsBuilt` (atomic), `NvnWaitForBuild` (the pipeline's own
  mutex/condvar), `NvnFirstStoresOf` (const): safe.
- `nvn_eden_cmd_state`, `nvn_eden_cmd_vertex_buffers`, `nvn_eden_cmd_vertex_table`, `nvn_eden_cmd_push_descriptors`, `nvn_eden_cmd_dbuf`: record into the
  caller's command buffer from immutable pipeline data and the device (`NO_RESCALING` constexpr): safe with per-thread command buffers.
- `nvn_eden_update_set` (vkUpdateDescriptorSetWithTemplate: the set is the caller's, from its own arena's pool), `nvn_eden_write_dbuf` (caller's ring memory),
  `nvn_eden_dbuf_alignment`, `nvn_eden_resolution_x2`, `nvn_eden_eval_handle` (reads `Shader::Info`; its word reader `eden_pool_word` takes the pool lock): safe.

Host side (gfx_vk_eden.inc), what a worker would share:
- Order-dependent, must become stream state (snapshot per segment): `eden_fill_state`'s `static lop_kept` / `static df_kept` (B7 register memory, 134, 144);
  `g_deq_dir` (item 28).
- Per-draw scratch globals, must become per thread: `g_vpull_want`, `g_vpull_why`, `g_vpull_tab` (vpull_decide -> eden_fill_state / s45_ctx_fill / eden_draw),
  `g_pd_draw`, `g_pd_tsample`, `g_eden_dbuf_bad` (eden_payload), `g_eden_cs_key`, eden_ubo's `static uint8_t c[65536]` (c1 copy, 475), `g_drv_lb/lo`
  (compute only). `vpull_frame` (`g_vpull_on`, `g_vpull_fr`: per present) on the replay thread only.
- Pipelines: `eden_graphics` miss path is check-then-create: two threads missing one key both build (Eden dedups) and may both `calloc` an `eden_pipe` for one
  Eden pipeline (`g_eden_pe` get and put in separate `g_eden_mu` sections) -> needs get-or-insert under one hold; `eden_ready_w` fills `e->pi`, `e->ready`,
  `vk_dsl_register` (its own mutex) and `g_eden_kdyn` without publication order (a reader may see `ready` before `pi`) -> once-init with release/acquire;
  `g_eden_kdyn` is read by every key (`eden_key_norm`, `s45_ctx_fill`). Same for `eden_compute` (dispatches stay serial in Stage 1).
- Draw-time writes to shared objects (same values, still races): `ep->s64w`, `eden_sw_note` (`vkshader_t.eden_sw`), `S.sh[0]->vpull_inputs/vpull_known`,
  `layer_out` (pkm2), `g_s64_pwrite` / `vk_par0_storage_draw`.
- Counters and diagnostics (per thread then summed, or refused with item 33): `g_vkst.*` (draws, skip_*, ns_desc, max_pass_draws), `g_r131.*`, `g_set_hits/new`,
  `g_eden_kmiss/kdup`, `g_eden_skip_build/wait_ns/wait_kept`, `g_vps_*`, `eden_prof` statics, `pd_seq`, `g_pd_*`, `g_sd_*` + `seen[]`, `g_wt_*`, `g_atom*`,
  s30b / pkm2 / pkm6 `static n`, the cull-tex diagnostic's `last[][]`. `VKRED_ONCE` and getenv-once statics are benign.
- Already per thread or locked: rings (`g_rec_e->dring/uring`, chunk reuse by the completed fence), `rec_eden` memos, `vbm_bind` (`vk_par_lock`), texture lookups
  (`vk_tex` tail under `vk_par_lock`, `g_tvc` per thread). Serial by Stage 1 rules: the hy3 window `g_hy3w` and `n1x_atomic` (item 25), scale (`pass_scale`,
  `scale_writes`), `fg_observe_draw` / `fg_capture_image` (frame-gen capture off).

**S1b (parallel recording): the dry pass and the same-thread resume harness (2026-11-21, syntax-checked only; off by default).** Code: the end of
`gfx_vk_exec.c` (`seg_*`), the piece hooks in `exec_ops`, small splits listed below.
- Switches (live, read when the environment changes; the first that is set wins): `DM_VK_PARALLEL_SERIALSEG=1` (dry pass + replay in pieces),
  `DM_VK_PARALLEL_DRY=1` (dry pass + replay as today). Neither needs `DM_VK_PARALLEL` workers. Both off: `mtl_exec` calls `exec_ops` as before; the default path gains
  one TLS test per record (`g_seg_on`), one TLS pointer test per CALL / JUMP (`g_seg_flow`) and one per CALL return (`g_seg_done`), and `seg_mode()` (one compare of
  `dm_env_gen`) per top-level list.
- Default-path refactors (same behaviour, needed by the dry pass): `bind_program_on(state, bp)`, `rt_same_on(state, r)`, `rt_assign_on(state, r)` (set_rt's state
  change) on a given `rec_state_t`; `oot_bilerp_live()` / `oot_bilerp_wide()` split from `oot_bilerp_rt` (its switch statics now file statics `g_oob_*`, same order of
  the environment read, stand-down and stand-in); prepare_draw's rasterizer-discard `static prev` is the file static `g_rd_prev`; `exec_ops`' `full_barrier_current`
  switch is the mask `FB_KEEP`; its `DM_VK_SKIP_OPS` static is `skip_ops()`; `n1x_atomic_prog(fs)` (eden.inc) names n1x_atomic_begin's two fragment programs;
  `vk_ubd_ranges()` (core) lists the deferred update ranges, read only.
- **Dry pass** (`seg_dry`, replay thread, at each top-level `mtl_exec` before the replay): copies `S`, `g_tf_on`, `g_rd_prev` and walks the list as `exec_ops`:
  same record validity check (a bad record ends that list), `DM_VK_SKIP_OPS` records passed over, CALLs descended on `dm_nvn_call_check`, JUMPs followed on
  `dm_nvn_jump_ok`, no depth limit (as `exec_ops`; segments only up to depth 8). Every record walked has an index in stream order; per depth the return point of each
  open CALL (`g_sd.ret`). On its copy of the state it applies `BIND_PROGRAM` (`bind_program_on`), `RAST_DISCARD`, `SET_RT` (`rt_assign_on`), `TF_STATE` (on/off).
  Per record: worker-safe (`SEG_SAFE`: the `S64_ALLOW` state ops, `BARRIER`, clears, `RESET_COUNTER`, `SIGNAL_EVENT`, `FENCE_SYNC`, `RENDER_COND`, non-timestamp
  `REPORT_COUNTER`, `UPDATE_UB` that cannot land, draws below), a legal boundary (a worker-safe draw that is not an indirect-count follower; a `SET_RT` that is not
  `rt_same` on the dry state), or serial (ends the segment; the name logged): copies, fills, texture clears, dispatches, `DRAW_TEX`, `DLSS`, `TF_BIND`, `TF_STATE`,
  timestamp reports (`g_cq`), a `SET_RT` the ootbilerp stand-in may copy at (`oot_bilerp_may_copy`, a superset of its copy cases), an `UPDATE_UB` that may land, a
  CALL into depth 9+, a bad record, the top list's end; draws whose program writes storage (`eden_prog_sw` 2) or is not known yet (0), rasterizer-discard draws and the
  first draw after such a run (`rd_read`), draws under transform feedback capture, direct draws of the two `n1x_atomic` programs. A segment opens at the first legal
  boundary of a run (depth <= 8), is cut again at the next boundary once it holds `SEG_TARGET` = 128 draws, and ends before the first serial record; a run that ends
  with fewer than `SEG_MIN` = 48 draws is refused, except the tail of a run cut at 128, which the segment before takes (so 128..175 draws). A cut is never at a record
  of `FB_KEEP` (static assert), so resuming a list with `full_barrier_current` 0 is exact.
- "May land" bound for `UPDATE_UB`: a set M of ranges, seeded with the real list at the list's start (`vk_ubd_ranges`), every update's range added. The real list
  is a subset of M (lands only empty it), so it can land only at an update whose range is new to M while |M| >= 512, or already in M while |M| > 512. Conservative:
  after a real land in a long list the bound stays high until the next list.
- Segment descriptor (`seg_desc`): `at` = `{d, pos[0..d]}` (return points, first record), `endp` = the first record not in it with its own `{d, pos[]}` (end depth
  `endp.d`, lower or higher than `at.d`), record indices `rec0/rec1`, CALL/JUMP decision range `flow0/flow1` into `g_sd.flow`, `UPDATE_UB` log range `ub0/ub1` into
  `g_sd.ub` (`{dst, n, data}` as recorded), `RENDER_COND` record range `cond0/cond1` into `g_sd.cond`, first draw serial in the submission `draw0` (reset when
  `g_set_epoch` changes), `ndraw`, the reason it ended, and the dry state at its start (program, shaders, rasterizer discard, a hash of the render-target fields).
- **Side effects.** The dry pass is discovery only, with one exception: it is the one that asks hle_nvn.c for each CALL / JUMP (`g_sd.flow`, bit 0 taken, bit 1 a
  JUMP); the replay of that list (DRY and SERIALSEG) reads the decisions (`seg_flow_take`) instead of asking again, so nothing is asked twice and the replay is
  guaranteed to walk the path the dry pass cut (a decision whose kind does not match, or none left, is `[RED]`-logged and the rest of the list asks again). Its other
  effects are not order-dependent: `vk_shader` lookups / creation by key for `BIND_PROGRAM` (the missing-shader log is once per key either way) and the
  ootbilerp switch read. It records no command and does no report, occlusion key, event, fence, render condition, uniform update, texture lookup or mark.
- **What the dry pass does not do (left to the worker step):** the order-dependent CPU work of `REPORT_COUNTER` (`cnt_note`, `rc_note`, `occ_key_make`,
  `mtl_sub_counter`), `RESET_COUNTER`, `SIGNAL_EVENT`, `FENCE_SYNC`, `RENDER_COND` (its value needs the reports before it: the descriptor lists the records, not the
  results), `UPDATE_UB` (`vk_sub_update`'s supersede index and scratch copy: the log holds `{dst, n, data}` only), `g_occ_draws`, `g_deq_dir`, R6's target
  resolution and marks, part cuts (`g_cb_draws`, `DM_VK_CB_DRAWS`, `g_psplit`: not predicted, so a segment may span a part cut), the `defer` handle copy of
  `bind_handles_rec` (it wrote guest handles into the record at replay time; S1c: the dry pass now reads them in stream order and the replay takes its copy; the dry state's texture/sampler/image handles are still not maintained). Draws under a
  false render condition and draws `prepare_draw` drops early count as draws (and for the rasterizer-discard run rule).
- **SERIALSEG** (`seg_pieces`): the list replayed on the replay thread into the normal command buffer, in stream order, as pieces: the records before segment 0
  (from the list start), segment 0 resumed at `at`, the records between, ..., the rest of the list. A piece runs a counted number of records (`g_seg_on`,
  `g_seg_left`, checked at the top of every `exec_ops` iteration) and stops before the next: `g_seg_done`, every frame returns (after its CALL). Resuming
  (`seg_resume`, design "Resuming inside nested lists"): list i+1 first, then list i from its return point, `g_par_depth` set per level. The end is a record count, not
  `p == end`: a list called twice inside one segment would meet its end record early; the pointer is checked at the stop instead. Nothing is ended or begun at a cut
  (no pass end, no barrier), so the commands are those of the uncut replay. Checks: the record a piece stopped at is the dry pass's `endp` record (else `[RED]`
  "stopped at another record"), the list did not end before a cut (`[RED]` "list ended early": the remaining pieces are not run), each piece starts at its decision
  index, all decisions were used, the stream state at a segment's start (program, shaders, rasterizer discard, render targets) equals the dry pass's copy. It does
  not test a worker's own state (the replay thread carries `S`, the open pass, `g_s64_pend`, `g_mdi_*` across cuts).
- Refused lists (nothing walked, replayed as before): not Eden mode, NVN2 (`dm_sdk_ounce`), resolution scale (`nvn_eden_resolution_x2`), frame-gen capture
  (`fg_frames_left`), bisection or oplog on.
- **Logs** (every 300 presents, `g_tex_frame`; per frame = averaged over those presents):
  - `S1b: dry pass per frame (N presents): lists L (refused R: not Eden, NVN2, resolution scale, frame-gen capture, diagnostics); draws D, in segments Ds (P%);
    segments S, draws per segment mean M max X; segment starts by call depth 0..8: h0 .. h8; CALL/JUMP decisions F; dry pass T ms`: L top-level lists a frame;
    D draw records walked in the lists not refused; Ds of them inside segments (the share a worker could take under the Stage-1 rules); S segments a frame; the
    depth histogram of segment starts (par0 predicts nearly all at depth >= 1); F decisions moved from the replay to the dry pass; T the dry pass's own time a frame.
  - `S1b: segments ended a frame by: <reason> n, ...; runs refused (under 48 draws) n a frame holding d draws, by: <reason> n, ...`: the six commonest reasons, by
    op name (`DISPATCH`, `COPY_T2T`, ... a serial op) or rule (`segment target (128 draws)`, `program writes storage`, `program's storage writes not known yet`,
    `rasterizer discard`, `draw after a rasterizer-discard run`, `transform feedback capture`, `n1x_atomic program`, `timestamp report`, `uniform update may land`,
    `SET_RT the ootbilerp stand-in may copy at`, `CALL deeper than 8`, `end of the list`, `bad record`). Ends mostly by `segment target` = long runs, good.
  - SERIALSEG only: `S1b: serialseg: S segments resumed a frame (P pieces); since start: stopped at another record a, list ended early b, CALL/JUMP decisions off the
    dry pass's c, stream state at a segment's start differs d` (`[RED]` at the end when any is nonzero; the first occurrences also have `[RED] S1b:` lines). All four
    must stay 0 and the picture must equal `DM_VK_PARALLEL_SERIALSEG=0` (A/B shots).
- NPARK rule: S1b adds no descriptor cache (no new key; `g_vk_storage_want` untouched), so the `[RED] ootu32: no storage view` absence is unaffected.
- **Remains for Eden thread safety and the workers (S1c+)** (S1c did (4), the grouping of (3)'s globals and the `defer` copy; the rest is in S1c's "Remains for Stage 1 proper"): (1) the order-dependent CPU work into the dry pass with a replay that skips it (reports, events,
  fence syncs, render conditions with HeyListen's stop rule for unread reports, `vk_sub_update` with the supersede log, `g_occ_draws`/occkey, R4 per-draw occlusion
  slots); (2) the dry pass deciding part cuts and R6's target resolution; (3) a full `rec_state_t` snapshot per segment (the dry copy at `at`, ~40 KB) with the
  `defer` handle copy of `bind_handles_rec` moved to the dry pass (or proven idempotent), plus the cross-record globals a worker must get from the snapshot or own:
  `g_mdi_cnt/n/stride`, `g_s64_pend`/`g_s64_pwrite`, `g_rd_prev` and `rd_read`'s `wa[]`, `g_deq_dir`, `g_tf_*`, `g_rc_skip` (from the condition array),
  `graphics_pipeline`'s `g_last_k/g_last_p` (replay-thread memo), `eden_fill_state`'s `lop_kept/df_kept`; (4) the Eden entry points of the S1a audit (per-thread
  `g_vpull_*`, eden_ubo's c1 buffer, `eden_pipe` get-or-insert, publication of `e->pi`/`e->ready`, draw-time shader-object writes `eden_sw_note`/`vpull_known`/`s64w`,
  counters `g_vkst`, `g_r131`, `eden_prof`); (5) chunks in the part's submission in order, a worker's `vk_end_pass` + barriers at its ends; (6) the first draw of
  a program stays serial until the bridge answers `nvn_eden_shader_writes` (R7).

**S1c (parallel recording): the draw-recording path made thread-safe (2026-11-21, syntax-checked only; nothing records on the workers yet).** Every change is a pure
refactor with one recording thread or a lock / atomic that is uncontended with one thread; code tagged "S1c (parallel recording)". Files: `gfx_vk_exec.c`,
`gfx_vk_eden.inc`, `gfx_vk_core.c`, `gfx_vk.h`, `source/src/video_core/nvn/nvn_shader_bridge.cpp`.
- **Per-thread scratch** (`_Thread_local`): `g_vpull_want`, `g_vpull_why`, `g_vpull_tab` (type `vpull_ent`), `g_vpull_direct`, `g_pd_draw`, `g_pd_tsample`, `g_pd_ov`, eden_payload's
  `pd_seq`, `g_eden_dbuf_bad`, `g_eden_cs_key`, `n1x_atomic` (begun and ended by the thread drawing; a worker's is always empty, so its `eden_ssbo` never redirects to another
  thread's staging), `g_draw_binst`, `mesh_draw_args`, the bundle diagnostics' per-draw flags (`g_r131_dupok`, `g_r132_instok`, `g_r142_ok`, `g_pofs_st/words`, `g_irun_on`,
  `g_r152_cand`, `g_var_now`, `g_tracing`, `g_pkm7_tr`, which eden_draw clears every draw), exec_ops' `g_xo_next`/`g_xo_op`, `bad_record`'s `g_rec_prev*`/`g_rec_jump_*`.
  Large buffers in `struct rec_eden` instead (TLS would give every process thread a copy): eden_ubo's c1 copy `c1buf[65536]` and deferred-update shadow `ubobuf[65536+256]`.
  Rule kept: TLS only for what is set and used within one draw, record walk or dispatch, never for state carried between submissions: the replay of a submission runs on the
  s46 replay thread or on the submitting thread (hle_nvn.c, under `g_sub_mu`), and that thread may change between submissions; carried state stays in the replay thread's
  contexts (`g_rec_main`, `g_rec_e_main`), which every non-worker thread reaches as before.
- **Stream state `rec_xs_t`** (`struct rec_ctx` member `X`; old names are macros): `[S]` = the value at a segment's start must come from the dry pass, `[O]` = the thread's own
  (a segment starts with no pass open). `mdi_cnt/n/stride` [O] (a run's followers are never a segment's first record), `s64_pend/pwrite` [O] (per pass), `rd_prev` [S] (0 at every
  segment start by the S1b rules) + `rd_wa[16]/rd_nwa` [O] (were prepare_draw's statics), `deq_dir` [S] (only read with `DM_VK_DEPTH_EQ` > 0, which keeps workers off), the
  transform feedback block `tf[4]/tf_off/tf_on/tf_in_pass/tf_verts` (`tf_on` [S], the dry pass already tracks it; the rest only capturing draws use: serial), `rc_skip` [S] (to come
  from the dry pass's condition results), `lop_kept/df_kept` [S] (eden_fill_state's B7 statics), s64a/s65 pass state `s64_afterbar/s64_fb/s65_rdpass` [O]. `mtl_queue_activate`'s
  per-queue save/restore of `S`, `g_tf*`, `g_rc_skip` works through the macros unchanged.
- **Counters** per context, the replay thread's being the globals the logs read: `vkst_t` (`g_vkst`, extern to core/present by name; type named), `r131_t` (`g_r131_all`),
  `g_set_hits/new`, `rec_pcnt_t` (s64a `g_s64_end/passes/bar/bar_open/bit/dpp/ab_n/ab_fb`, s64 `npend/kept/fbend/opend`, s65 `rdp/rddraws/rdsplit/cut`, `g_repeated_barriers`,
  cbar `skip/kept/g_cbk`), r80's viewport table; in `struct rec_eden` the vtxpull counters `g_vps_pulled/g_vps_fb` and eden_prof's sums (`prof_n`, `prof_s`; it logs when
  `prof_n` >= 2^17 on the replay thread's context, averages over `prof_n`: the same lines in serial mode). `vk_rec_ctx_fold(rec, rec_e)` adds a worker's into the replay
  thread's and zeroes them (max_pass_draws as a maximum, r80 merged by size); `vk_chunk_wait_part` calls it for every worker with a chunk in the joined part (a chunk
  job's `vk_chunk_finish` is its last count). Atomic instead (rare, shared): `g_eden_skip_build/wait_ns/wait_kept`, s30b / pkm2 / pkm6 / NVN2 push-constant / NVN2 indirect-SSBO
  log counters; under `g_eden_mu`: `g_eden_kmiss/kdup`; under a new `g_sd_mu`: the NVN2 SSBO diagnostic (`g_sd_*`, `seen[]`; always on in NVN2 titles, so locked, not refused).
- **Pipelines.** `eden_pipe_put(k, p, &made)`: after a miss (the bridge asked outside any lock: Eden makes the pipeline once and gives every asker the same one) the key's entry
  is got-or-inserted in one hold of `g_eden_mu`, with its Eden pipeline's one `eden_pipe` (found by pipeline or `calloc`'ed with `p`/`k0` set before it is put): no second
  `eden_pipe` per Eden pipeline, counts only for the inserter. `eden_compute` likewise. `eden_ready_w` asks `nvn_eden_pipeline_get` into a local (its wait may take seconds: no
  lock held), then `eden_pipe_publish` once under `g_eden_mu`: `pi`, the set layout's `vk_dsl_register`, the s21 rule `g_eden_kdyn` (atomic; read atomically by
  `eden_key_norm`/`s45_ctx_fill`), then `ready` with release; readers acquire `ready` before reading `pi`. `s64w` once by compare-and-swap (`eden_pipe_sw_once`): the winner
  does the first use's work (`eden_sw_note`, the compute shader's `eden_sw`), as the first draw/dispatch did.
- **Shader objects** (shared by all threads drawing with them): `eden_sw` sticky at 2 by atomic store, 1 only over 0 by compare-and-swap (`eden_sw_set`), read atomically by
  `eden_prog_sw`/`eden_cs_sw` (the dry pass); `vpull_known/vpull_inputs` published once (`vpull_inputs_publish`: CAS 0 -> 2 claims, inputs stored, 1 with release; `vpull_decide`
  acquires `== 1`); `layer_out` set once by CAS (`layer_out_mark`), read atomically by `draw_layered`; the vtxpull switch `g_vpull_on` atomic (written by `vpull_frame` on the
  replay thread only; a worker returns from `vpull_frame` at once). hy3's window: `vk_hy3_close_for` returns on any other thread than the replay thread's (dispatches stay serial).
- **Library** (`nvn_shader_bridge.cpp`): `nvn_eden_shader::has_code` is `std::atomic<bool>`; `CopyCode` writes code, stage (`DetectStage`), `hash`, `lowered` under `lower_mutex` and
  publishes `has_code` last (release; it was set before the stage was detected); every reader of the code already gates on `has_code` (environments' constructors copy it), and
  `nvn_eden_shader_c1` / `c1_hash` now do too. Published code is immutable: a second `CopyCode` with the same bytes is a no-op, with other bytes refused (logged): an environment
  may be copying it on another thread, and pipelines made from it stay in Eden's cache under its hash. No caller passes new code today (`nvn_eden_shader_set_code` has no host
  caller; `nvn_eden_shader_new` sets it before the object is published). Not redesigned (noted): pipeline creation is serialized under `nvn_mutex` for a whole miss.
- **Deferred binds** (`bind_handles_rec`, `defer`): not idempotent: it wrote the game's handles into the record (`b->h`, command memory) at every replay: two recorders of one
  record race on that write, and `src` is game memory rewritten between submissions, so a worker would read it later than stream order. Now the record is never written: the
  handles are read into a local (default path: from `src` at replay, values into `S` unchanged), or, while the dry pass walks the list (DRY / SERIALSEG), the dry pass reads them
  once in stream order (`g_sd.bh`: `{n, src, n handles}` per record) and the replay takes them in order (`seg_bh_take`, like the CALL/JUMP decisions; a mismatch is `[RED]`-logged
  and the rest read from `src`); a segment's range is `bh0..bh1` in `seg_desc`. SERIALSEG checks each piece's start index and that all were taken; its log line adds
  "deferred-handle reads off the dry pass's" (must be 0). In DRY/SERIALSEG the handles are read when the dry pass walks the list, before its replay (earlier, never later, than
  before). Side effect: command memory keeps the recording-time `h` (the capture tool, `nvn_capture.c`, now sees those, not the last replay's).
- `graphics_pipeline`'s `g_last_k/g_last_p` (992-byte key: TLS would put it into every thread): kept the replay thread's; any other context skips the memo (bundle path only).
- `seg_worker_refuse()` (unused until the workers record): diagnostics whose tables are the replay thread's (paydiag, atomdiag, wtrace/vsres, replaydiag/par0, s15, vbytes, vtxmir
  A/B, frame-gen collection `nvn_fg_collecting`) and `DM_VK_DEPTH_EQ` keep a list's segments off the workers (`seg_refuse` keeps covering what the dry pass refuses).
- **Self-test** `DM_VK_PARALLEL_SELFTEST=2` (with workers; after the S1a empty chunks): all workers at once, 32 rounds with a start barrier, no draw recorded:
  `eden_pipe_put` on fresh synthetic keys (pairs per Eden pipeline, one key without), `eden_pipe_publish` of a synthetic compute info (no layout: nothing registered),
  `eden_pipe_sw_once` + `eden_sw_note`, `vpull_inputs_publish`, `layer_out_mark` on fresh shader objects, counting into the worker's own contexts, the TLS scratch. The replay
  thread checks one insert / one publisher / one first per object, equal entries across workers, the maps, final marks (2 for the writing stage, 1 else), the fold into a scratch
  context pair (none into the logs'), then deletes the test keys from `g_eden_gp`/`g_eden_pe` and restores `g_eden_kmiss/kdup` (held at 16 meanwhile so no s21 diff line is
  spent). One line `S1c: parallel recording stress self-test: N workers x 32 rounds: ...` (`[RED]` on a failure).
- NPARK rule: S1c adds no descriptor cache and no cache key (`g_vk_storage_want` untouched).
- **Left as is (judged unsafe or out of scope here):** the GPU-write list / `vk_sub_gw_overlaps` / `vk_sub_wver` read by workers while the replay thread's serial ops append to
  it (needs a frozen per-segment view, below); `g_occ_draws`, `g_rc_n`, `g_rcd_*`, `g_sub_ops`, `vk_og_op` (ootgap), `g_par_depth`/par0, `g_draw_i` (bisection order), `g_cb_draws`,
  `g_psplit_now`, `g_s61_*` (order-dependent or the replay thread's: the dry pass's or refused); `g_tf_ctl` (transform feedback control buffers: serial ops); the compute-only
  statics (csdiag, cull-tex `last[][]`, csres, upsprobe, `g_drv_lb/lo`, hy3 `g_hy3`); `lop_kept`'s update point (see Remains 2).

**S1d (parallel recording): the dry pass does the order-dependent CPU work, full snapshots, segment update lists, part cuts (2026-11-21, syntax-checked only; only with
`DM_VK_PARALLEL_DRY=1` / `DM_VK_PARALLEL_SERIALSEG=1`).** Code tagged "S1d (parallel recording)": `gfx_vk_exec.c` (the seg_* section after `exec_ops`, `state_op_on`,
`bind_handles_on`, `rc_value_e` / `rc_eval` / `render_cond_e` / `render_cond_note`, `part_cut`), `gfx_vk_core.c` (`struct vk_ubl`, `vk_sub_update_pre/apply`, `vk_ubl_*`,
`sub_counter` / `vk_sub_counter_q`, `vk_sub_ts_next`), `gfx_vk.h`. `gfx_vk_eden.inc` and `gfx_vk_tex.c` unchanged.
- **Handed-over decisions** (`struct seg_dec`, `g_sd.dec`): the dry pass pushes one entry per record it does work for, `{record index, kind, value, ...}`; the replay numbers the
  records it walks the same way (`g_seg_ri`, counted in `seg_hot` at the top of each `exec_ops` iteration, the one test S1b's `g_seg_on` had) and takes the next entry
  (`seg_dec_take`); an entry whose index or kind does not match is `[RED]`-logged ("decisions off the dry pass's") and the rest of the list is done as without the dry pass. A
  piece in SERIALSEG starts at its segment's `dec0` (checked like `flow0`); all must be taken at the list's end. Per record type, what moved and what stays:

  | Record | Moved to the dry pass (stream order) | Stays the replay's (at the record) | Stays recording |
  |---|---|---|---|
  | `REPORT_COUNTER` timestamp | `cnt_note`, `rc_note`, the query slot (`vk_sub_ts_next` at the list's start, one per timestamp; none with `DM_VK_TS_REPLAY`, then serial as before) | the post op (`post`, kind 2 with the slot) | `vkCmdWriteTimestamp` at the named slot (`vk_sub_counter_q`; `ncq` advanced past it; a slot other than the one the old allocation gives is counted) |
  | `REPORT_COUNTER` other counters | `cnt_note`, `rc_note` | rptfix's queued write (`report_now`: `gw_seq`, `wseq`, `npost` at the record) or its post | - |
  | `REPORT_COUNTER` samples passed | `cnt_note`, `rc_note` (same values: `rc_note`'s fence is `mtl_last_fence() + 1`, which only a commit changes) | `occ_key_make`, the occkey / fixed write, the post | `oq_end` (the query) |
  | `RESET_COUNTER` | `cnt_note` | `g_occ_draws = 0`, `vk_oq_reset` (occlusion) | `oq_end` |
  | `FENCE_SYNC` | all of it (the submit's ticket into the sync object; idempotent) | - | - |
  | `SIGNAL_EVENT` | nothing | the post op (kind 1): its place among the posts | - |
  | `RENDER_COND` | the evaluation (`rc_eval`) with the report table as at the record and one fence value for the list (`vk_fence_done()` at its start), the census and diagnostics (`render_cond_note`) | `g_rc_skip` set from the result; stop rule: evaluated as before | - |
  | `UPDATE_UB` | the range check and the copy into scratch memory (`vk_sub_update_pre`) | the list operation (`vk_sub_update_apply`: supersede, land when full, append) | the land's copies (serial ops only) |

  Why the commits stay at the record: the dry pass runs over a whole top-level list before its replay, so the replay-time state they capture (`g_gw_seq`, `wseq`, `npost`: which
  GPU writes and post ops were recorded before the report) is not there yet when the dry pass walks them, and post ops from the dry pass would come before the posts of earlier
  serial records (an event signalled before an earlier report's late write). Inside one segment `g_gw_seq` / `wseq` cannot change (no GPU writer in a segment), so the worker
  step can commit a segment's reports and events on the replay thread when it hands the segment out, in order, except for their place against the post ops recording makes
  inside the segment (`oq_end` at pass ends, samples-passed reports): see "Remains" 1.
- **Render conditions, the stop rule.** HeyListen stops its dry pass at a condition on a report not read yet. Ours never needs that for reports (a report of this submission or in
  flight is "unknown, keep drawing" by r52's rule, which the dry pass applies with the table it keeps in order). What the dry pass cannot know is memory a GPU write may still
  change: a condition whose words a write in flight covers (`vk_gpu_pending(addr, 32)`) or that follows any GPU-writing record of the list (copies, fills, dispatches, texture
  clears, storage-writing / rasterizer-discard / atomic / transform-feedback / indirect-quad draws, updates that may land, the ootbilerp copy) is left to the replay (value -1):
  evaluated there as before (`render_cond_e`, with the table's entries copied at the record, so the dry pass having run ahead does not change it, and the live fence), it ends the
  segment ("render condition left to the replay"), and the draws until the next decided condition are serial ("draw under a render condition left to the replay"); no segment
  starts meanwhile. (S1d2: the rule is applied only to conditions whose replay outcome is not fixed yet; one on a report of the same submission is decided: see S1d2.) SERIALSEG evaluates every decided condition again with the same entries and fence value: only guest memory changed by the game since could differ.
  Behaviour change (DRY/SERIALSEG only): a condition sees the fence done at its list's start instead of at its replay (a submission completing during the list's replay is seen
  one list later: unknown, keeps drawing).
- **Uniform updates as data (R5's list).** `struct vk_ubl {gen, n, e[512]}` (core); the thread's current list behind `_Thread_local g_ubl` (the main list `g_ubl_main` by default),
  `g_ubd`/`g_nubd` macros over it; `g_ubd_gen` (gfx_vk.h) is a macro over `g_ubd_genp`, the thread's list's generation; every bump (`ubd_bump`: append, land, drop) takes the next
  value of one atomic counter, so memos (`g_setc`, r142, s45 `m->ubd`, the shadow cache) never match across lists. DRY: each `UPDATE_UB` takes the dry pass's copy and applies the
  list operation on the main list at the record. SERIALSEG, where segment k starts (`seg_ub_open`): the main list is copied into the segment's list (`vk_ubl_snapshot`: the
  per-segment `g_ubd` snapshot), then the segment's own updates are applied to the main list in order (`vk_sub_update_apply`, superseded index logged into the decision), and
  the segment is replayed on its list (`vk_ubl_use`), each `UPDATE_UB` applying its logged update by index (`vk_ubl_log_apply`, no search; checked against a search in
  SERIALSEG). At its end the segment's list must equal the main list (entries' ranges and data). A segment never lands (the dry pass's may-land bound, plus "indirect quad draw"
  now serial: `ixq_args` lands the list and records a dispatch, which S1b had missed); a land of a segment list is counted (`g_ubl_seg_lands`), and an update applied at a segment's
  start that landed is counted. Not done: `g_infl` frozen per segment (appended only at lands, never in segments, but pruned at lands while a worker would read it) and no
  `vk_rpt_flush` on a worker (`vk_ubd_read`).
- **Full snapshot per segment** (`seg_snap`, `g_sd.snap[k]` beside `g_sd.seg[k]`): `rec_state_t` up to `pass_open` (`SEG_SNAP_BYTES`: program, shaders, all bindings, fixed
  state, viewports, push constants, render targets, strides; the open-pass fields are the recorder's own) and `rec_xs_t`'s [S] fields `rc_skip`, `tf_on`, `rd_prev`, `deq_dir`,
  `lop_kept`, `df_kept`. The dry pass now applies every pure state record to its copy through the same functions the replay uses (`state_op_on`: the 25 state records exec_ops
  had inline, now one function with `live` = bump memo generations and log; `bind_handles_on` for textures, samplers, images with the deferred handles it reads itself;
  `bind_program_on`, `rt_assign_on` as S1b). `rc_skip` comes from its condition results. `lop_kept` / `df_kept` / `deq_dir` follow key builds (`eden_fill_state`, `eden_draw`), so
  the dry pass applies the same rules at every draw that reaches `prepare_draw`'s key (not dropped by a false condition, not an indirect-count follower, with a program), on its
  state: the logic op, and for the depth target the image `vk_tex_peek` finds (format, depth aspect; memoised per target). A field a draw under an undecided condition would
  change, or the depth target's image not made yet (made when the replay begins its pass), makes that field unknown (`XU_*`) until a decided draw sets it; no segment starts while
  any is unknown ("segment starts refused (state not known)"). Not modelled: draws `prepare_draw` drops after its program check (no usable target, `pass_commit`) or `eden_draw`
  returns before its depth rule; the B7 root fix (registers written where the game writes them) stays open (below). SERIALSEG compares the replay's state with the snapshot at
  every segment start (`seg_check_snap`: the stream part by `memcmp`, naming the first differing field group, each `rec_xs_t` field separately); any difference is `[RED]`.
  `[O]` fields (mdi run, s64 pending barrier, open pass, s64a/s65 state, `rd_wa`) are the recorder's: a worker starts them as a fresh context with no pass open (S1c); SERIALSEG
  carries the live values across cuts and does not compare them.
- **Part cuts** (`g_sd.cut`, `seg_cut_take`, `part_cut`): while a list takes the dry pass's cuts (`g_seg_cuts`), `begin_pass` makes no DM_VK_CB_DRAWS cut and `prepare_draw`
  no DM_VK_PASS_SPLIT pass end. The dry pass counts draws as `prepare_draw` would (`g_cb_draws` at the list's start, then each draw not dropped by a false condition and not an
  indirect-count follower) and plans an ordinary cut at the first record after the count reaches DM_VK_CB_DRAWS where the pass ends anyway: a `SET_RT` to other targets, a copy,
  fill, texture clear or dispatch (`SEG_PASS_END`), or a draw where rasterizer discard toggles; at most `g_s61_maxcut` ordinary cuts a submission (over the cap: counted, the
  count restarts, as `begin_pass`). DM_VK_PASS_SPLIT: a cut before a draw once the dry pass's pass (draws since the last pass-ending record) holds that many draws, while parts are
  left. A cut closes the segment being gathered ("part cut"), so no segment spans one; the replay ends the pass and cuts before the record (`seg_cut_take`; in SERIALSEG a cut at a
  segment's first record is taken before the segment's update list is set up). Positions differ from today's: today's cut is at the first pass *begin* after the count, which the
  dry pass cannot know (passes also end at barriers, s64 decisions, layered changes, feedback); the dry pass cuts at the pass *end* before it (a few draws earlier or the same
  place; `OP_BARRIER` is not a cut point: it is an `FB_KEEP` record, where a resumed list would differ). Cuts stay at pass ends, so no extra pass end is made, except
  DM_VK_PASS_SPLIT's own (as before).
- **Segment rules added:** "indirect quad draw (compute conversion)", "render condition left to the replay", "draw under a render condition left to the replay", "part cut";
  "timestamp report" only with `DM_VK_TS_REPLAY=1` now. `gw_dirty` (any GPU-writing record walked in the list) feeds the stop rule.
- **Logs** (every 300 presents, after the two S1b lines):
  - `S1d: handed over a frame: counter reports R (timestamps with a slot named T), counter resets Z, fence syncs F, render conditions C (left to the replay S), uniform updates U
    (in segments Us); part cuts P (pass splits Q, over the cap K, taken X); samples-passed reports inside segments O (left to the replay: occlusion, R4); segment starts refused
    (state not known) N`: what the dry pass did a frame; Us the updates applied through segment lists (SERIALSEG); X must equal P + Q; O is the occlusion work that still blocks
    handing those segments to workers.
  - `S1d: serialseg|dry since start: decisions off the dry pass's a, timestamp slots off b, render conditions evaluated differently at replay c, segment update lists off the main
    list d, segment update log entries off e, segment updates that landed f, segment lists landed g, part cuts not taken or inside a segment h; snapshot fields differing at a
    segment's start: stream state, render condition, transform feedback, rasterizer-discard run, depth direction, logic op, depth format ...` (`[RED]` when any is nonzero). All
    must be 0 (c may only move if the game rewrote a condition's words between the dry pass and the replay; the logic op / depth format / depth direction counts measure the
    key-build approximation above). The S1b serialseg line's "stream state at a segment's start differs" now counts any snapshot difference.
- **Expected coverage** (OoT, from S1b's 86% of 2,204 draws): timestamp reports were the top segment ender (~300 draws a frame refused); they no longer end segments, so most of
  those runs join their neighbours (expected ~90-95% of draws in segments, longer segments, fewer refused runs). Lost again: ~1-2 segment ends per part cut (OoT ~6 submissions x
  ~2 parts: about 6-12 cuts a frame, a few short runs refused), segment starts while a depth target's image is new (first frames of a scene). Render conditions and indirect quads:
  none expected in OoT (render_cond profile key off; quads not seen). Measured: wrong for conditions, OoT records ~480 a frame (46% of draws in segments; see S1d2).
- NPARK rule: S1d adds no descriptor cache and no cache key (`g_vk_storage_want` untouched).
- Known limits: with an allocation failure in the dry pass the CPU work it already did stays done and the replay does it again (counter census counted twice, report table
  entries written twice with the same values, fence syncs again, scratch copies again); the dry pass's rules for `lop_kept` / `df_kept` / `deq_dir` are approximations of
  "draws that reach the key" (measured by SERIALSEG); `seg_snap` is ~40 KB, copied at every segment start candidate (refused runs included).

**S1d2 (parallel recording): the first in-game run's two SERIALSEG faults and the render-condition coverage regression (2026-11-21, syntax-checked only; only with
`DM_VK_PARALLEL_DRY=1` / `DM_VK_PARALLEL_SERIALSEG=1`).** Code tagged "S1d2 (parallel recording)", `gfx_vk_exec.c` only (`rc_word_class`, `seg_dry_list`, `seg_open`,
`seg_end_at`, `seg_close`, `seg_log`). In game (OoT 0400486024DEC000, Kokiri village, ~30 s each), S1d gave: DRY 46% of 870 draws in segments (S1b/S1c: 73-86%), segments ended
and runs refused mostly by "render condition left to the replay" (242.1 of 482.1 conditions a frame left to the replay); SERIALSEG 602 decisions off the dry pass's
(`[RED] S1d: a piece begins at decision 268, the dry pass cut it at 269`) and 266 part cuts inside a segment (`[RED] S1d: a part cut at record 2891 is inside a segment (from 2232)`).
- **Decisions one off (cause).** A segment's ranges (`flow1`, `dec1`, `bh1`) were the dry pass's counts at the moment `seg_close` ran, and `seg_open`'s the same at `seg_open`.
  The records that end a segment *after* pushing their own decision, `RENDER_COND` left to the replay (stop rule: `seg_dec_push` first, then `seg_serial(SW_RCSTOP)`) and a
  timestamp under `DM_VK_TS_REPLAY` (push, then `seg_serial(SW_TS)`), put their own decision into the segment they end. The segment's piece stops *before* that record
  (`seg_stop`), having taken none of its decisions, so the next piece began at decision n while `dec1` said n + 1; `seg_piece` then set `g_seg_di = dec1`, which skipped the
  record's own entry, its `seg_dec_take` met the next record's entry and dropped the hand-over for the rest of the list (2 `dec_bad` per list; its part cuts were then made the
  old way). Fix: the dry pass notes `nflow` / `ndec` / `nbh` at the top of every record it walks (`g_sd.r_flow/r_dec/r_bh`), and a segment's bounds are those of its first record
  and of the first record not in it, i.e. exactly what the replay has taken when it reaches that record, whatever a record does before it ends or opens a segment. (S1b's CALL /
  JUMP serial ends were already before their pushes; the bounds now no longer depend on that order.)
- **Part cut inside a segment (cause).** `seg_close` merges a run of fewer than `SEG_MIN` draws into the segment before when that one ended where the run began
  (`l->rec1 == c->rec0`): meant for the short tail of a run cut at `SEG_TARGET` (`seg_boundary` closes and opens at the same record). A part cut does the same: `seg_cut_at`
  closes the segment at the cut record ("part cut") and `seg_boundary` then opens the next candidate at that same record (a `SET_RT` to other targets, a draw where
  DM_VK_PASS_SPLIT cuts). When that candidate stayed short it was merged backwards across the cut, so the merged segment spanned it. Fix: the tail is merged only into a segment
  that `SEG_TARGET` ended (`l->why == SW_TARGET`); a short run after a part cut is refused like any other (counted, "short runs after a part cut"). Cut positions are unchanged;
  segments are closed before them by construction now (every cut closes the candidate, and nothing extends a closed segment across a cut).
- **Render conditions: decided when the replay's outcome is fixed (coverage).** What the replay reads for each word of a condition (`rc_value_e`, both words evaluated, the
  condition "known" only when both are): `rc_word_class(entry, fd, cur)`, `fd` = `vk_fence_done()` at the list's start, `cur` = `mtl_last_fence() + 1`, the open submission's
  fence (`g_fence` moves only at the commit, so no fence the replay can see reaches `cur`):
  `RCW_UNK` unknown whatever the fence and memory (non-timestamp entry never measured `f == ~0`, `DM_VK_RC_OCC=0`, or a samples-passed report of this submission `f == cur`);
  `RCW_ZERO` a timestamp of this submission (0, no read); `RCW_FENCE` a report of an earlier submission not done at `fd` (unknown or memory, as the live fence says);
  `RCW_MEM` read from guest memory (no entry, or its submission done). A condition with a word `RCW_UNK` is "unknown, keep drawing" at the replay whatever the fence and memory;
  two `RCW_ZERO` words are 0 == 0. Such a condition is decided by the dry pass even after a GPU write or with one pending (same `rc_eval`, same result as the replay's
  `render_cond_e` would give: occlusion semantics untouched, r52's rule as it is); the stop rule now applies only to conditions with a `RCW_MEM` word ("memory a GPU write may
  change") or, without one, a `RCW_FENCE` word ("a report in flight": the replay may see it complete and read memory). OoT: its conditions read samples-passed reports of the
  same submission (render_cond census "occlusion values 0 real, ~94% this submission"), so nearly all of the 242 a frame are decided again. Census note: for a decided condition
  whose other word is `RCW_FENCE`, `g_rc_n` counts that word with the list's fence instead of the live one (census only; the outcome is fixed by the `RCW_UNK` word).
- **Logs.** New line after the S1d line: `S1d2: render conditions a frame: C; decided by the dry pass: enable/disable E, outcome fixed without memory (a report of this submission
  or never measured) F, from memory or the list's fence with no GPU write before M; left to the replay S: memory a GPU write may change Sm, a report in flight Sf; short runs after
  a part cut (not merged across it) T` (C = E + F + M + S, S = Sm + Sf; the S1d line's "left to the replay" equals S).
- **Expected (OoT, Kokiri village).** SERIALSEG: "decisions off the dry pass's" 0, "part cuts not taken or inside a segment" 0, no `[RED] S1d` piece / cut lines, the S1b
  serialseg counters 0, snapshot fields 0. DRY: "render conditions left to the replay" from ~242 to the conditions on reports in flight or on memory after a GPU write (expected
  well under 20 a frame, mostly "a report in flight"); "render condition left to the replay" and "draw under a render condition left to the replay" nearly gone from the segment
  ends and refused runs; draws in segments back to at least S1c's 73-86%, more with the timestamps S1d took off the serial list (target 85-90%); "segment starts refused (state not
  known)" lower (fewer `rc_unk` stretches); "short runs after a part cut" about one per part cut at most (≤ 4 a frame). The default path (both switches off) is unchanged: every
  change is in the dry pass and the segment bookkeeping.
- NPARK rule: S1d2 adds no descriptor cache and no cache key (`g_vk_storage_want` untouched).

**S1e (parallel recording): segments recorded on the workers (2026-11-21, syntax-checked only, not run).** Code tagged "S1e (parallel recording)": `gfx_vk_exec.c` (the
S1e block before `seg_top`: `seg_job`, `seg_par_handout`, `seg_job_main`, `vk_seg_part_join`, `seg_par_pieces`, `seg_par_log`; hooks under `g_wk` in `exec_ops`, `prepare_draw`,
`begin_pass`, `pass_commit`, `vk_end_pass`, `set_rt`, `clear_whole`, `clear_draw`; the dry pass's `seg_tg_*`, `seg_end_snap`, `WR_*`), `gfx_vk_eden.inc` (one guard),
`gfx_vk_core.c` (`vk_par_view`, `vk_par_handout`, `vk_par_merge_posts`, `par_cbs`, the helper context, `vk_par_help_*`, `vk_chunk_wait_all`), `gfx_vk_tex.c` (worker lookups,
held image objects, `vk_tex_prepare`), `gfx_vk_present.c` (`vk_par_pin_self`), `gfx_vk.h`.
- **Switches.** `DM_VK_PARALLEL=<n>` (1..6, read at device init): n worker threads plus a helper context (no thread) for the replay thread; `seg_mode()` is then 3 (the dry pass on,
  segments handed out) unless `DM_VK_PARALLEL_SERIALSEG=1` (the harness wins). `DM_VK_PARALLEL_PIN=0` (device init): workers not pinned. `DM_VK_PARALLEL=0` (default): no thread, no
  helper; the default path gains only TLS tests (`g_wk` per record and in the pass code; `tl_wv` in the GPU-write lookups and `post`), atomic adds for the occlusion slot and a few
  counters, `g_bar_seq` / `g_draw_i` / `g_psplit_now` / `g_seg_cuts` / `g_par_depth` / `g_oq_active` thread-local, `g_bind_gen` / `g_ubo_gen` / `pass_scale` / `scale_writes` /
  `scale_dirty` in the recording context, one atomic load per image drop, a mutex around the masked clears' pipeline cache.
- **Per list (replay thread, `seg_par_pieces`).** SERIALSEG's walk: serial pieces and part cuts as before; at each segment either the hand-out or, when it is kept, the same-thread
  resume (verified path: its own update list, `seg_piece`). Kept when (`seg_desc.wk`, set while the dry pass gathers it, then `seg_par_why`): a samples-passed report (occlusion, R4),
  any other counter report (its commit's place among the post ops and rptfix's queue), a samples-passed reset, an event, an indirect or multi-draw, more than 16 target states, a pass on
  a linear target with z4 on (write-back), the B7 / r57 state at its end not known, a diagnostic of `seg_worker_refuse` (paydiag, atomdiag, wtrace, vsres, replaydiag, s15, vbytes,
  vtxmir, `DM_VK_DEPTH_EQ`, frame-gen collection, pass timestamps / ootgap, crumbs, pass / exposure / indirect dumps, traces, `DM_VK_SKIP_FS`, drawprof, idxmir, wb traces, UBO log,
  watches, pkm7, s56), the list's hand-over dropped (the walk left the dry pass's path), no memory for a job. Already ended by the dry pass's rules: render conditions left to the
  replay, updates that may land, storage-writing / unknown programs, rasterizer discard, transform feedback, n1x_atomic, the ootbilerp copy, timestamps under `DM_VK_TS_REPLAY`.
- **Hand-out (`seg_par_handout`, replay thread, at the segment's first record, part cuts taken).** In stream order before the worker's chunk: image objects held
  (`vk_tex_par_hold`), the replay thread's pass ended; `g_occ_draws += ndraw`, `g_cb_draws += ncb` (the segment's draws as `exec_ops` / `prepare_draw` count them); for each target
  state (`seg_tg`: the one at its start, then each SET_RT to other targets) `rt_assign_on` on a scratch copy of the targets and `oot_bilerp_rt` (its kept-target bookkeeping; a
  SET_RT it may copy at is never in a segment), and for each one a draw or clear will begin a pass on, R6's `pass_resolve_targets_on` (made, first-use fill, `note_image_write`,
  `wseq`, `gpu_w`, `vk_tex_gpu_mark`, the first transition into the replay thread's chunk) plus `pass_commit`'s marks (`wb_use`, `scaled_valid = 0`), the views and images kept in the
  job with the targets' bytes; r150's `vk_tbuf_fresh`; `vk_rpt_flush`. Then the job: the replay thread's state now (S up to the open pass, `rec_xs_t`'s [S] fields: the dry pass's
  state at the segment's start, SERIALSEG's check), a copy of the main update list, the segment's updates applied to the main list (superseded indices logged, as `seg_ub_open`),
  copies of its CALL/JUMP decisions, deferred handles and decisions. `vk_par_handout` (core): the frozen view (the submission's `gw` list, `ngml`, `wseq`, `g_nsubs`; `g_gw`'s
  entries in flight; `g_infl`'s entries in flight; `g_oq_counting` / `g_oq_epoch`), a kind-5 post op placeholder, the replay thread's R chunk ended and listed, the W chunk listed,
  a new R chunk begun (`rcb[]`, from the slot's pool). The job goes to the least loaded worker. The replay thread then takes the dry pass's end snapshot of the segment (`esnap`, made
  for accepted segments only), bumps its memo generations (`g_bind_gen`, `g_ubo_gen`, `g_s45_kgen`), skips the segment's decisions and goes on at its end record.
- **Worker (`seg_job_main`).** Its chunk begun (`vk_par_chunk_begin`), the view set (`vk_par_wk_begin`), S and `rec_xs_t` from the job ([O] fields fresh), its memo generations
  bumped, its own copy of the update list, the job's decision arrays, no part cuts; a full barrier; `seg_resume` at the segment's position for its records; the pass ended, a full
  barrier, the chunk ended; `vk_chunk_finish` last. What it skips or does differently (`g_wk`): `g_occ_draws`, `g_cb_draws`, `g_sub_pass`, the render-condition census, `g_sub_ops`
  / ootgap op marks, ootbilerp, r150, frame-gen's pass profile, pass_commit's marks; attachment lookups (`seg_wk_targets`: the job's table by the targets' bytes, nothing looked up,
  marked or transitioned); `vk_ubd_read` without a report flush, with the frozen in-flight lists; `vk_sub_gw_overlaps` / `vk_sub_wver` / `vk_gpu_pending` on the frozen view; its
  occlusion queries counted as its view says, slots by atomic add, their kind-4 posts into the view; a first image transition into the part's upload buffer. It aborts (the segment is
  then recorded again on the replay thread) when: `vk_tex` would make an image, restore an archived one, upgrade one for storage, or refresh NVN2 sparse tiles (`g_vk_tex_wk`); a pass
  begins on targets the hand-out did not resolve; a decision or deferred-handle read is off the dry pass's; a record the rules keep off the workers is met (report, event, SP reset,
  multi-draw, storage-writing pipeline); a whole-target clear (r76 `clear_whole`: its marks); the walk does not stop at the dry pass's end; the chunk's begin or end fails ([RED]).
- **Join (`vk_seg_part_join`, from `vk_cmd_split` before the part ends and from `mtl_sub_commit` after its pass end).** Queued jobs run on the replay thread (`vk_par_help_one`:
  stolen from the queue tail, recorded in the helper context with the replay thread's segment-walk TLS kept aside), then `vk_chunk_wait_all`; each failed job recorded again
  there (`rerun`: images may be made, unresolved targets resolved there; `[RED]` and the segment's draws left out if that fails too); each job's post ops put at its placeholder
  (`vk_par_merge_posts`, the queued reports' `npost` moved with it); counters folded; held image objects freed. Submission: `par_cbs` lists the upload buffer, the part's chunks in
  order (failed ones out), the open R chunk; one `vkQueueSubmit` as before. A part not sent early lists its R chunk for the commit (`DM_VK_SUBMIT_SPLIT=0`); a refused submission drops
  the chunk list with the rest.
- **Occlusion (not changed).** The queries are the same per-pass-span queries; a worker's passes begin and end their own (the replay thread ends its pass at the hand-out, so a pass
  that went on across the boundary is two spans: the same sum with `VK_QUERY_CONTROL_PRECISE_BIT`, which the N1X has); slots are taken by atomic add (no gaps: each is begun and ended);
  the posts come back in stream order at the segment's place. No samples-passed report or reset is inside a handed-out segment, so the epoch is constant there.
- **Dry pass cost (item 4).** The full start snapshot (`seg_snap`, ~40 KB) was copied at every candidate, most of which are refused (under 48 draws, or a run with no draw at all
  after a SET_RT boundary). Now it is copied only with SERIALSEG (mode 2, its check); the workers start from the replay thread's live state, and mode 3 copies the end snapshot of
  accepted segments only. DRY (mode 1) copies none. Estimate: the dry pass's 1.68 ms a frame (OoT, DRY) drops by (candidates − 0) × ~4 µs in DRY and (candidates − segments) × ~4 µs
  with workers; with ~150-250 candidates a frame (12.5 segments, the refused runs and the empty SET_RT candidates) that is ~0.5-0.9 ms, i.e. ~0.8-1.2 ms a frame left (to be measured:
  the S1b line's "dry pass ... ms").
- **Thread placement.** `vk_par_pin_self` (gfx_vk_present.c): the CPU sets of the highest EfficiencyClass (parked ones included, as Eden's `PerformanceCpuSets`), power throttling
  off; nothing on a host with one class; one line (`S1e: recording workers on the efficiency class ...`).
- **Logs** (every 300 presents, after the S1d2 line): `S1e: parallel recording a frame (N workers): segments handed to workers H (D draws), kept on the replay thread K (D draws) by:
  <reason> n, ...; worker record T ms (t ms a segment), hand-out X ms, the replay thread waited W ms at part ends (ran J queued jobs itself); parts with worker chunks P, chunks per such
  part C; recorded again on the replay thread R by: <reason> n; since start: worker failures F, segments lost L` (`[RED]` when F > 0). `[RED] S1e:` lines for a chunk that could not
  begin or end, a segment lost, a placeholder gone, a segment update that landed, a worker that could not be pinned.
- **NPARK rule:** no descriptor cache and no cache key added (`g_vk_storage_want` untouched); the worker records through the same descriptor paths.
- **Known limits / risks:** (1) coverage: OoT reports ~2,000 samples-passed counters a frame, so most segments may hold one (WR_SPREP) until R4 or a stream-ordered commit of
  reports at the hand-out; the S1e line's "kept ... by" says how many. (2) Guest memory read at worker time (uniform shadows, handles from uniform buffers, texture dirty checks) is
  read later than in stream order (open question 12), and a report the replay thread flushes after the hand-out may be seen by a worker's uniform read of the same bytes. (3) A failed
  segment is recorded at the part's end: its image creations, evictions and (when unresolved) target marks then come later than in stream order. (4) `nvn_dirty` and the content
  ledger's dedup table (`nvnc_*` `recent[]`) are called from workers on packaged re-uploads (under `vk_par_lock` among recorders, not against the replay thread's other calls).
  (5) The render-condition census counts no draw a false condition drops inside a worker's segment. (6) Pass numbers (`g_sub_pass`) count the replay thread's passes only.

**Remains for Stage 1 proper (after S1d), in order:** (S1e: items 4-6 and 8 done in the first version, 7 by refusal and worker-time aborts; 1-3 remain)
1. Posts in stream order: the commits that stay the replay thread's at the record (rptfix's queue entries, report and event post ops, timestamps' posts) must move to the replay
   thread's hand-out of a segment for workers. Exact once the post ops recording makes inside a segment (occlusion: `oq_end` at pass ends and at reports, samples-passed reports)
   carry their place in the stream, i.e. with R4 (d) (one post per window, made by the dry pass) or a stream key per post op that the completion thread orders by. Until then a
   segment holding a samples-passed report (or any occlusion query) must stay on the replay thread (S1d logs how many there are: "samples-passed reports inside segments").
2. The B7 registers (`lop_kept` / `df_kept`) and r57's `deq_dir` per segment exactly: the root fix is to update them where the Maxwell registers are written (the colour state
   bind, `OP_BIND_STATE` kind 1, for the logic op; the depth target at `SET_RT` for the zeta format), a pure function of the stream. It changes Eden pipeline keys only where a
   logic-op state is bound but not drawn before a COPY draw (picture identical: the op is disabled there), so it needs an A/B of pipeline-key counts before adoption. S1d's
   key-build approximation and its SERIALSEG counters show whether it is needed.
3. R4 occlusion by draw serial: deferred by NPARK ("최적화 하고 나중에 잡는게 나을듯": optimise first, fix later), to be done together with its in-game A/B (R4 (e): every
   report value and the occkey table equal at `DM_VK_PARALLEL=0`). S1d keeps occlusion exactly as before (`g_occ_draws`, occkey, the queries, the posts).
4. R6's target resolution and marks in the dry pass (a worker's attachment lookup must not mark: `wseq`, `note_image_write`); the depth image's format then comes from the dry
   pass's own resolution instead of `vk_tex_peek`.
5. A frozen GPU-write view per segment (`wseq` at the segment's start for `vk_sub_gw_overlaps`, `vk_sub_wver` and `vbm_bind`'s stale check), since the replay thread's serial ops
   between segments append to `s->gw` while workers record earlier ones; and `g_infl` frozen per segment (`{ptr, count}` taken at hand-out, pruned only between parts: R5), and
   `vk_ubd_read`'s `vk_rpt_flush` done by the replay thread before it hands a segment out (a worker never flushes).
6. Chunk submission order: each part's `[up] R0 W1 R1 ...` chunks in one `vkQueueSubmit` in stream order where `vk_cmd_split` / `mtl_sub_commit` submit; a worker's chunk ends with
   `vk_end_pass` + `vk_img_barrier_all` and starts with the barrier; `MAIN_CBS` grows into the chunk list; the upload buffer's commands from workers under `vk_up_lock`.
7. NVN2 paths per segment or serial: `s2tile_refresh` (records binds and uploads per lookup, outer lock only), residency history, `image_version_advance` / `note_image_write`
   (via R6 marks), F0 drop accounting per part: either made per segment (dry pass marks, part-level accounting on the replay thread) or the draws reaching them kept serial.
8. Then the worker step itself: hand segments to workers (`seg_desc` + `seg_snap` + the CALL/JUMP, decision and deferred-handle ranges; the worker binds a fresh context, copies
   the snapshot into its `S` / `X`, takes a segment list from `vk_ubl_snapshot` and applies the logged updates), the replay thread committing each handed-out segment's reports
   and events in order (item 1), `seg_worker_refuse`, the first draw of a program serial until `nvn_eden_shader_writes` (R7), per-site lock counters, Stage 0.4 pinning.

**Threads.** `DM_VK_PARALLEL=N` workers (N <= 6), created at device init (S1a, `slots_init`; S1e: pinned to the P-cores with `SetThreadSelectedCpuSets`, `DM_VK_PARALLEL_PIN=0`
off; 0.4 still to measure), plus the replay thread recording queued jobs when it waits (S1e, the helper context). Off when: `!vk_eden_mode()` (NVN2 is not refused: OoT is NVN2),
resolution scale on, frame-gen capture active, any diagnostic of item 33 on.

**Validation.** Interleaved A/B on fixed scenes (shots every N presents, brightness series, as HeyListen BM tick054),
`pcverify` for stale GPU data (memory note), counters for: segments per part, draws per segment, dry-pass ms, wait ms,
lock takes/waits, serial-op chunks per part. Promote the default only after a clean run on 3 scenes.

### Stage 2: overlap and early GPU work

- Per-chunk upload command buffers and init buffers (HeyListen VR:3690-3698): each `W` chunk carries its own uploads, so
  chunks can be submitted as soon as they and all before them are done (an ordered submitter, VR:3087-3097). The fixed
  draw-count part cut (`DM_VK_CB_DRAWS`, exec:640, 2012) is then replaced by chunk submission; the GPU starts within one
  segment instead of one part (HeyListen: ~1 ms less GPU idle a frame).
- The dry pass runs ahead across parts while workers record (HeyListen `SEG_EARLY`); the replay thread waits only at the
  submission's commit (`mtl_sub_commit`).
- Balanced boundaries from a per-submission-order time profile (`RX_VK_PARALLEL_BALANCE`), segment affinity to keep
  per-thread caches warm, smaller lock scopes (TLS front cache for `vk_tex_view` hits).

### Stage 3: wider coverage

- Copies, fills, dispatches, storage-writing draws inside segments: the dry pass keeps the GPU-write list
  (`sub_gpu_wrote`), pool sync and the hy3 window in order (as `dry_mark_written` does for images, VD:11076), workers only
  record. Timestamp reports with dry-assigned slots.
- NVN2 paths (`s2tile_refresh`, residency, `image_version_advance`, F0 drop accounting per part), scale paths, the
  non-Eden bundle path.
- Secondary command buffers to split a single giant pass without store/load, if Stage 0 shows most draws in a few passes:
  needs the replay thread's primary to be assembled after the workers (a deferred primary).

## 4. Risks

- **Picture correctness.** HeyListen's history (section 1.6) is the checklist. Ours specifically: a texture first created
  by a worker's sampled lookup with `init_fill`/`cpu_first_use` (tex:2246-2263) while an earlier segment renders into its
  storage; R6 covers targets, but storage writes of excluded draws are recorded serially *after* the earlier segments were
  handed out, which is fine in queue order. Uniform shadows (`vk_ubd_read`, core:4582) read guest memory at worker time,
  later than serial: a segment can see newer CPU data than the segment after it (open question 12). The occlusion change R4
  alters which samples a report counts (per draw window instead of per pass span): must keep occkey's keyed values equal
  (compare report values serial vs R4 before parallelism).
- **Races with `g_img_mu`.** Only the map is under it (tex:2192-2243); the upload tail runs unlocked and records into the
  shared upload buffer. Eviction and pressure (`img_evict_idle` and the trylock paths tex:1176-1251) run at submission start
  on the replay thread; with Stage 2 overlap they may meet a worker's lookup: images used by a worker carry
  `last_use = mtl_last_fence() + 1` (tex:2239), so eviction must keep checking it against the completed fence, never the
  submitted one. `vk_tex_sweep` / `vk_tex_drop_pool` (tex:2282, 2324) must not run while segments are out.
- **Uploads.** One upload buffer per part in Stage 1: uploads from different workers land in arbitrary order inside it;
  correct while they touch different images or write the same bytes (the order repair rule of VT:3511 is not needed while
  every upload precedes the whole part, as today). `nvn_dirty_*` stamps and protection calls from worker threads need the
  dirty tracker to be thread-safe (open question 8; the memory note on `MapRegion/ProtectRegion` bypassing `nvn_dirty`
  makes stale data there easy to miss).
- **NVN2 specifics.** Sparse tile refresh records binds and uploads per lookup; residency history and F0 drop accounting
  are per part. Ocarina of Time (the target) is an NVN2 title, so NVN2 cannot be refused: since 11-21 the dry pass and SERIALSEG run on NVN2 lists (same thread, same order); the worker step must make these paths per segment or keep the draws that reach them serial.
- **Hybrid cores.** A worker on an E-core makes the slowest segment the critical path (HeyListen: the slowest segment
  ~1.25 ms whatever the count). Pinning and per-segment timing are part of Stage 1.
- **Performance traps.** Cold per-thread caches (first draws ~2x), lock contention (HeyListen 13%), the replay thread's
  serial remainder, and extra command buffers/pass ends per part. Mitigated by `SEG_MIN`, counters from day one, and
  Stage 2 balancing.
- **Rollout.** `DM_VK_PARALLEL` defaults to 0 until validated; with 0 nothing of the parallel path runs. The prerequisite
  refactors are root changes (one path) and land first with their own A/B at `DM_VK_PARALLEL=0`.

## 5. Expected gain

Assume (to be checked in Stage 0): 75% of the 15 ms is in worker-eligible draws (~11.3 ms, ~4.9 us a draw), the dry pass
costs ~10% of that (~1.1 ms; HeyListen 0.33-0.48 ms per ~800 draws vs ~3 ms of recording), the remaining 3.7 ms stays
serial, contention and cold caches inflate worker draws 1.6x (HeyListen), 4 recorders (3 workers + replay thread).
Worker time 11.3 x 1.6 / 4 = 4.5 ms, overlapped with the replay thread's 4.8 ms of dry pass + serial ops; with Stage 1's
per-part waits, ~1-2 ms of waiting remains. Replay thread: 15 -> ~8-9 ms. The GPU (14 ms of work) then becomes the
bottleneck and its idle shrinks from ~6 ms to ~2 ms (Stage 1) or ~1 ms (Stage 2): frame ~20 -> ~15-16 ms, about +25-30% fps
at best, +10-15% if only half the draws are eligible. HeyListen's first version gave +20-26% (54->65, 49->62), later tuning
+3-5%; our serial share is larger (more bookkeeping per draw), so the Stage-0 split decides the real number.

## 6. Open questions

1. Split of the replay time (Stage 0.1): `eden_draw` measured at 12.2 of 21.8 ms (56%); still open: how much is how much state ops, reports (~2,000 a frame in OoT, rptfix
   comment core:4115-4123), conditions and serial ops?
2. Answered by par0: 90-96% of draws in runs >= 48; most passes hold 1 draw but ~0.9 passes per submission hold 64+, so
   in-pass splits are needed from the start.
3. Thread safety of `nvn_eden_*` (`eden_graphics`, `eden_ready_w`, `nvn_eden_cmd_state`, `nvn_eden_cmd_push_descriptors`,
   `nvn_eden_cmd_vertex_buffers`, `nvn_eden_update_set`): per-pipeline mutable state inside the library?
4. How often do OoT draws write storage (`s64w == 2`) or hit the `n1x_atomic` keys (eden:476-477)? Each one ends a segment.
5. Answered by par0: 99.9% of draws are inside `OP_CALL` lists; Stage 1 boundaries carry the call-stack position.
6. Timestamp reports through `g_cq` (core:4215): frequency in OoT; cut or dry-assigned slots?
7. Is the resolution scale (`nvn_eden_resolution_x2`, `pass_scale`) or frame-gen capture on in the OoT configuration?
8. Is `nvn_dirty` (protect/release/since/now) safe to call from several threads?
9. Does the per-draw occlusion model (R4) reproduce occkey/occaddr values (core:4152-4210) exactly? Is the query ring large
   enough (~2,300 counted draws a frame, several frames in flight)?
10. Answered by S1b: `oot_bilerp_rt` keeps the last scene target and the output layer (statics) and records a `copy_t2t` at a SET_RT (a GPU write), so it is
    order-dependent; the dry pass keeps every SET_RT it may copy at on the replay thread (`oot_bilerp_may_copy`). Off once ootu32's storage view is used.
11. Content ledger (`nvnc_*`): does it accept several command buffers per part with parts' accounting unchanged?
12. Guest reads at worker time (`vk_ubd_read`, texture dirty checks): is the game writing next-frame data into those
    ranges during the replay, and does a later read change pictures?
13. N1X cores free during OoT play, and the cost of a P-core worker vs an E-core one (Stage 0.4).
14. Stage 2: can `DM_VK_CB_DRAWS` cuts be dropped in favour of chunk submission without hitting a driver limit on
    submissions a frame?
