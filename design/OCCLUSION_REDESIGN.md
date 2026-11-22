# Occlusion redesign: real counts, per-draw slots, GPU-side render conditions

Status: design, 2026-11-21. Steps 0 and 1 implemented and verified in game (OoT); steps 2 and 3 implemented (`DM_VK_OCC2=s2` / `s3`, syntax-checked only, not
run; see section 6.9); steps 4-5 not started. Read-only research of `refer/HeyListen-main` (GPL-3.0: mechanisms only, no code
copied), `refer/NX2navi-main`, `refer/Longshot`, and the Eden query cache in our own tree (`source/src/video_core`). Note:
`refer/_snapryu2` was **empty** when this was written, so the Eden/yuzu query cache is described from `source/src/video_core`
(same lineage). If the SnapRYU2 tree differs, re-check section 1.3.

Scope (NPARK, 2026-11-21): (a) per-draw query slots by stream serial (R4); (b) no 4096-slot overflow; (c) render conditions
decided by real GPU-measured values; (d) remove every substituted value from an earlier frame (occkey keyed table, occaddr,
occland, rptfix-style fixed 4096). One switch for the whole new path, default on, off switch for the old path.

---------------------------------------------------------------------------------------------------------------------------------

## 1. How the references do it

### 1.1 HeyListen (Vulkan replay, TICK-032/033)

| Piece | Mechanism | Where |
|---|---|---|
| Default report value | `REPORT_COUNTER` is written **on the CPU, in stream order, at replay** as `{count, timestamp}`; SAMPLES_PASSED count = `RX_VK_OCC_FIXED` (4096). Workers skip reports (dry pass writes them). Address marked `memory_cpu_owned` so mirror write-backs never overwrite it. | `engine/gpu/vk/vk_replay.cpp:1910-1951` |
| Reset | `RESET_COUNTER` samples-passed: window start = current stream serial (`occ_win_lo = occ_serial`, replay thread only). | `vk_replay.cpp:1953-1960` |
| Stream serial | Every draw **and dispatch** record advances a serial, **whether or not it is recorded** (render condition false, raster discard, dry pass). Replay/dry thread: `occ_serial`; a segment's thread: `occ_wserial`, seeded from the dry pass's serial at the segment start. Each draw sets the TLS slot via `OccSlot`. | `vk_replay.cpp:175-193`, `2380-2455`, `2797` |
| Allocation | **One query per draw**, slot = `serial % 64K` in one host-reset ring (`kOccRing = 1<<16`, chunks of 4K). Entering a chunk: wait for the timeline value of its last lap, read its pending reports, `vkResetQueryPool` (host) the chunk. Precise bit if supported. | `engine/gpu/vk/vk_draw.cpp:6474-6517`, `6653-6673` |
| Query scope | `OccQuery` begins right before the draw is recorded and ends on every exit of `draw()` (a skipped draw is flagged, still ended). | `vk_draw.cpp:6521-6545`, `8205` |
| Report = window | A report is `{addr, lo = serial at reset, hi = serial now, submission value}` pushed on a pending list. Its count = sum of slots `[lo, hi)`, unbegun slots counted as "unrecorded". | `vk_draw.cpp:6675-6713`, `6568-6645` |
| When read | `occ_poll` at each present for submissions done; `vkGetQueryPoolResults` (64-bit + availability). Mode 4 writes the real count over the fixed value in guest memory **only if it is still the latest report at that address**. | `vk_draw.cpp:6568-6650`, `6714-6760`; `vk_replay.cpp:5851` |
| Value if not ready | Fixed 4096 ("everything visible") in modes 0/1/4 until resolved; modes 2/3 the last count of the same address; mode 5 the count of the same *key* (program + window + rank in frame) from <=3 presents ago. | `vk_draw.cpp:6675-6713` |
| Render condition | CPU at replay: reads 64-bit words at `addr` and `addr+16`; mode 0 = equal, 1 = differ. Dry pass evaluates and appends to `par_conds`; workers only read `par_conds[cond_at]` (default **true** if not yet published). | `vk_replay.cpp:2278-2327` |
| Stop rule (mode 4) | A condition whose `addr`/`addr+16` has a pending (unread) report: the **dry pass stops there** (`occ_stop_at`), the segments up to it are recorded and submitted, the CPU **waits** for the GPU (`occ_sync` / `occ_go_on`), every pending count is written, replay continues. If no submit is possible: give up = fixed value read. | `vk_replay.cpp:2286-2294`, `2631-2650`, `3784-3788`; `vk_draw.cpp:6714-6730` |
| Switch | `RX_VK_OCC_REAL=0` default (off). 1 log only, 2 zero-or-fixed by address, 3 last count by address, 4 same frame synchronous, 5 one frame late by key. `RX_VK_OCC_LOG=<dir>` logs every report/condition/draw with pipeline statistics. | `vk_draw.cpp:6447-6470`; `docs/SWITCHES.md:3250-3256`, `11212-11236` |

Measured (OoT Market, RTX 5070; `docs/research/occlusion.md:126-188`, `docs/tickets/TICK-033.md:9-14`):
- ~1,840 samples-passed reports a frame, windows of 1.1-1.2 draws; ~900 proxy tests first, then ~920 conditions, **all mode 1
  with y = 0** ("render if count != 0"), each gating ~1 draw and followed by its own report (for the next frame).
- **Every condition reads a report of the same frame and the same stream, >= 557 draws (median ~1,690) later.**
- **Report addresses never repeat between consecutive frames** (0 of 1,825 at lags 1, 2, 4). Any late value by address is
  another object's: modes 2/3 = washed-out picture, missing buildings, white characters (1,110 of 1,850 reports written 0).
- Mode 5 (late by key, our occkey's model): not faster (68.6 fps), and "a 1% key mismatch is enough to delete a building for a
  frame" (house fronts missing at load-in).
- Mode 4 (same frame, CPU wait): **correct picture** standing and walking, sun/haze/shadows right, ~350 of ~920 conditions off;
  cost one GPU wait a frame: 70.2 -> 61.7 fps standing, GPU saved only 8.5 -> 8.1 ms.
- Decision: fixed value stays default; "next fix": `VK_EXT_conditional_rendering` with per-report predicate words (no wait,
  correct). Never built.
- `RX_VK_OCC_FIXED` history: a per-draw estimate (16 a draw) made the sun follow the count -> one-frame haze flicker
  (`SWITCHES.md:3230-3232`). F2/`RX_VK_OCC_PROXY_SKIP`: with fixed values the ~900 proxies are dead work (`docs/benchmarks/2026-11-12-tick061.md:54,63-65`;
  `docs/research/suspects/F-gpu.md:28,50-53`).
- Parallel replay (TICK-006/033): serials span command buffers and segments; counts agree with serial replay (H3 not a cause).
  Mode 4 + `RX_VK_SEG_EARLY` never verified (`TICK-056.md:48`).

### 1.2 NX2navi / Longshot
- NX2navi: real queries, a ring of `OQ_N` (16384 now; AUDIT lists 4096: "when full, draws go uncounted"), one query per pass span
  (begin at first counted draw, end at pass end/report/reset), slots reset by the completion thread; report written as a **CPU
  write after the submission's fence**; "a result that is not ready counts as visible"; a full ring gives "at least 1" ->
  "object all but hidden", patched by `NX2NAVI_OQ_HOLD` (= last complete count of the record, i.e. a prior-frame value)
  (`runtime/vk_exec_xfer.c:377-557`, `docs/maxwell.md:140-155`, `docs/architecture.md:369`). Our r50 design is this one.
- Longshot: "Render conditions read reports from earlier in the same stream, so they cannot be resolved at record time"
  (`docs/dev/reference/architecture.md:418`).

### 1.3 Eden query cache (`source/src/video_core`, Vulkan backend)
- Allocation: banks of 256 host occlusion queries (`renderer_vulkan/vk_query_cache.cpp:51`). One query per *counting span*:
  `StartCounter` begins a slot, `PauseCounter` ends it at pass ends / reports, `ResetCounter` abandons the current query
  (`vk_query_cache.cpp:184-235`); rasterizer pauses/resumes around passes (`vk_rasterizer.cpp:628, 795, 976, 1581`).
- Report: `WriteCounter` creates a query object referring to the slot range since the reset (`vk_query_cache.cpp:361-385`).
- Resolution **on the GPU, in stream order**: at a wait-for-idle point (`NotifyWFI`, `query_cache/query_cache.h:485-500`)
  `SyncWrites` records `vkCmdCopyQueryPoolResults(WAIT_BIT|64_BIT)` into a resolve buffer, runs a prefix-scan compute pass to
  accumulate multi-slot windows, and copies the sums into the guest buffer (`vk_query_cache.cpp:237-358`). Separately, after the
  fence, the CPU path sums bank results and writes guest memory (`vk_query_cache.cpp:386-420`, `query_cache.h:297-330`; an
  unsynced value logs "Query report value not synchronized").
- Conditions: **host conditional rendering** (`VK_EXT_conditional_rendering`): a resolve pass computes a predicate from the
  24 bytes (x vs y) on the GPU, then `vkCmdBeginConditionalRenderingEXT` (inverted bit per mode), paused/resumed around
  render-pass changes (`vk_query_cache.cpp:1399-1490`, `1740-1820`). Without the extension: CPU read **after waiting** for
  the pending results (`WaitForConditionRead`, `vk_query_cache.cpp:~1497-1515`), not "unknown -> draw".
- NXN1X's NVN HLE replay does **not** use this cache; it is reference only.

Summary of the references: everyone who gets a correct picture with real counts either **waits** (HeyListen mode 4, Eden
without the extension) or **evaluates on the GPU** (Eden HCR). Late values (by address or by key) are wrong in OoT.

---------------------------------------------------------------------------------------------------------------------------------

## 2. Ours, and what explains each symptom

### 2.1 Current mechanism (default build)
| Piece | Ours | Where |
|---|---|---|
| Pool | `OQ = 4096` queries **per submission slot** (`NSLOT*OQ`), created only when keyed or `DM_VK_OCC_FIXED=-1`; reset at each submission start for the prefix used | `gfx_vk_core.c:2500-2505`, `3528-3530`, `4325-4328` |
| Allocation | one query per **pass span** while counting: begun at the first draw (`vk_oq_draw`), ended at pass end, report, reset (`oq_end`), one kind-4 post per query. `g_oq_counting` is set by the first reset and **never cleared**, so every pass span after the first reset of the run takes a slot. Workers take slots by atomic add (not stream order). | `gfx_vk_core.c:4339-4355`; `gfx_vk_exec.c:2089, 2334` |
| Overflow | `q >= OQ`: the epoch is marked lost, `[RED] occlusion queries: more than 4096 passes counted in one submission` | `gfx_vk_core.c:4351` |
| Report value (default) | occkey: at replay, the report is written with the **count of its key from <= 3 presents ago** (key = VS/FS keys + draws since reset + rank in the present; occaddr: address key first), else **4096**. The real sum (completion thread, `oq_sum` per epoch) only goes into the table (`occk_put`); mode 7 = not written. occland (`DM_VK_OCC_LAND=1`, off) would write the real count after the fence. | `gfx_vk_core.c:4413-4455`, `4463-4480`, `5179-5185`; key `gfx_vk_exec.c:4821-4831` |
| Report write timing | queued (replay2) and flushed per submission before `vkQueueSubmit` and before any `vk_ubd_read` | `gfx_vk_core.c:4376-4405` |
| Render condition | CPU at replay/dry pass. r52 rule: a samples-passed report word is "real" only once the submission that reported there completed; `rc_note` stamps it with the **open** submission's fence, so in the same submission it is **unknown -> draw**; in flight also unknown. The word in memory (the occkey value) is **never** used by a condition. | `gfx_vk_exec.c:847-851`, `876-879`, `883-895`, `902-907`, `953-963` |
| Dry pass | `rc_word_class`: RCW_UNK/ZERO decided in the dry pass, RCW_FENCE/MEM may stop (`SW_RCSTOP`); reports/resets set `WR_SPREP`/`WR_SPRESET` = segment stays on the replay thread; workers `seg_wk_abort(WA_OP)` on them | `gfx_vk_exec.c:5640-5666`, `5645, 5648`, `5203-5210`, `5759-5766` |

### 2.2 Symptoms
**(S1) Conditions never use a real value** (census "0 real, ~94% this submission", NPARK table 92.7-95.2%). Explained by
design, not a bug in the census: r52 can only call a value real after the reporting submission's fence, and HeyListen shows
every OoT condition reads a report of the **same frame and stream** (`occlusion.md:146-147`), i.e. ours is always "this
submission" (or "in flight" across a split). So every proxy-gated draw is drawn: the same as HeyListen's fixed 4096 for
conditions. The ~2.3-3.6% false are CPU-written flags / timestamps (HeyListen: ~62 a frame read clear flags). Evidence: strong.

**(S2) 4096 overflow** (`gfx_vk_core.c:4351`). Explained: slots are per pass span per submission, and counting never stops
after the first reset, so a submission's slot count = (reports + resets + pass ends with draws) in it, unbounded. OoT has
~2,000 reports + passes per frame in 4-6 submissions; a submission that holds most of a frame, a long load or a merged part
passes 4,096. Fixed-size per-submission pools cannot be right; the ring by serial removes the bound. Evidence: code; which
submission overflows is not logged (add it, step 0).

**(S3) Terrain flashing, worse at low / uneven fps** (NPARK table: PAR2 49.8 fps, 17.4 frames > 33 ms / 10 s: hardly;
ASYNC1 46.7 / 13.2: sometimes; PAR4 39.6 / 72.9: a lot). Conditions are not the cause (they are always "draw" whatever the
timing, S1). What *does* depend on GPU timing is the **value occkey writes into the report words**, which the game reads
(CPU, or GPU through uniform shadows taken at replay, `gfx_vk_core.c:4380`):
- the table entry is put by the completion thread (`occk_put`, `gfx_vk_core.c:5184`), so whether the report gets the count
  of 1, 2 or 3 presents ago, or the fixed 4096 (no entry yet), depends on how far the GPU/completion is behind the replay;
  uneven frames move that age from frame to frame;
- ~48-49% of substituted counts are 0: a stale or mismatched 0 culls a visible object for a frame (HeyListen mode 5:
  "a 1% key mismatch is enough to delete a building for a frame"); a stale non-zero shows a hidden one;
- PAR4 log (`logs_poolcopy/par4_182322.txt`, 600-present windows): ~773 K keyed reports, ~388 K of them 0, 12-22 K fixed,
  ~131-137 K zero/non-zero flips of a key (17%).
Caveat (stated as NPARK asked): **flip rate does not track flashing monotonically** (PAR2 13% flips but the least flashing,
ASYNC1 8.2% flips but "sometimes"); flashing correlates better with **frame-time unevenness** (> 33 ms frames), i.e. with
the *age* of the substituted value, which no metric measures today. Also `gfx_vk_core.c:2484`: with 12 submission slots
"the GPU results reached the game a frame later and OoT flashed white": direct evidence that result latency becomes
flashing in OoT. Conclusion: likely, not proven. Proof needs the age metric of step 0 (frame the result was measured vs
frame the guest used it). Guess, lower ranked: the fixed 4096 fallback (no key yet) also flips haze/sun locally.

**(S4) occland trembling** (real count written after the fence over the keyed value). Two writes per report, the second at
completion time: a read between replay and completion sees last frame's (keyed) count, a read after sees this frame's; which
one depends on timing -> tremble. Same mechanism as the earlier `DM_VK_OCC_FIXED=-1` tremble ("written after the fence, a
read before that found the memory's old value", `gfx_vk_core.c:4356-4360`). Evidence: the code comments and NPARK's
observation; the reader's timing itself is unmeasured.

**(S5) Pokemon Z-A dialogue flicker with occkey**: rank keys pair a label's report with another label's when labels come and
go (`gfx_vk_core.c:4468-4471`, NPARK's own analysis); occaddr reduces it. Under (d) the cause disappears: no key at all.
Pokemon's addresses are stable and it reads after waiting (per occland's note), so a same-frame real count is the
console behaviour. Evidence: medium.

**(S6, for R4) Segments with occlusion cannot go to workers**: ~70% of OoT segments contain a samples-passed report or reset
(NPARK note), and both are replay-thread-only (`gfx_vk_exec.c:5203-5210`, `5645-5648`) because slots and post ops are taken
at replay in an order the dry pass cannot name (`design/PARALLEL_REPLAY_DESIGN.md:145, 360-380, 845`).

---------------------------------------------------------------------------------------------------------------------------------

## 3. Decision

**Real counts, measured per draw by stream serial, resolved on the GPU in stream order, and render conditions evaluated on
the GPU with `VK_EXT_conditional_rendering`.** No CPU wait in the common case (HeyListen mode 4's cost avoided), no value from
an earlier frame anywhere (HeyListen modes 2/3/5 and our occkey/occaddr/occland/fixed removed from the new path), no
per-submission slot bound. This is HeyListen's documented "next fix" plus Eden's HCR shape, adapted to our dry pass.

One switch: **`DM_VK_OCC2`** (default 1 when adopted; `0` = the old path exactly as today: occkey, occaddr, occland,
`DM_VK_OCC_FIXED`, r52 rule). During development the same switch selects how far the new path goes (`DM_VK_OCC2=s<k>`:
steps up to k; logs only for k=1), collapsing to 0/1 at adoption. Per NPARK's rule the default-on flip happens only after the
in-game A/B of section 5 passes in the same build.

Why not the alternatives:
- Mode 4 style CPU wait: correct but -8.5 fps on a 5070 at 70 fps; on N1X at 40-50 fps a mid-frame GPU drain is worse. Kept
  only as the rare fallback (3.4).
- Any late value (by address, key, "hold"): proven wrong in OoT (addresses never repeat; key mismatch deletes objects) and it
  is exactly what (d) forbids.
- "Unknown -> draw" (today for conditions): no flashing source, but no real occlusion and the reports' memory still needs a
  value; with (d) that value must be the real one.

---------------------------------------------------------------------------------------------------------------------------------

## 4. Steps (each landable, each with its counters)

### Step 0 - instruments on the current path (log only, `DM_VK_OCCDIAG=1`, no behaviour change)
Answers the open questions before behaviour changes. Every 600 presents:
1. **Report landing age**: per samples-passed report, present index at replay, present index when its GPU result was
   available (fence done), present index when its occkey value was taken from (`occk_get` hit: entry's frame). Histogram of
   "substituted value age" (0 / 1 / 2 / 3 presents, fixed) per window. Correlate with > 33 ms frames (S3).
2. **Guest-read-before-landed**: who reads report words, and when.
   - GPU readers: count `vk_ubd_read` / uniform shadows / copies whose range overlaps a samples-passed report word, with the
     distance (draws, submissions, presents) from the report. If OoT binds report memory as a uniform buffer, the read is a
     CPU copy at replay and a GPU-landed value would arrive too late (see risk R1).
   - CPU readers (sampled, diagnostic only): protect the pages of pending report words (guard pages through the fastmem
     mapping, as pcverify does for stale GPU data) from replay until the real result lands; a fault = "read before landed",
     logged with the report's age in presents and the guest PC. Keep to N reports a window to bound cost.
3. **Condition source census** (extends `g_rc_n`): per condition, word class (samples report this submission / in flight /
   landed with age / CPU flag / timestamp), mode, whether y is 0, report-to-condition distance in draws and in **pass ends**,
   same submission or not. Confirms HeyListen's "same stream, mode 1, y = 0, >= 557 draws".
4. **Overflow context**: on the RED line, the submission's draws, passes, reports, resets, and what was in it (load, menu).
5. Baseline the three NPARK runs (PAR2, ASYNC1, PAR4) with these lines.

### Step 1 - R4: one query slot per draw by stream serial (`DM_VK_OCC2=s1`; values unchanged, logs compare)
- **Serial**: every draw-type record (DRAW, DRAW_IDX, MULTI_DRAW element group as one, DRAW_TEX, mesh) advances one stream
  serial, recorded or not (condition false, discard, skipped pipeline), as HeyListen (`vk_replay.cpp:2450-2455`). The dry
  pass owns the count and hands each segment its start serial (like the timestamp slot naming of S1d); the replay thread and
  workers derive the same numbers. No atomics.
- **Ring**: one `VkQueryPool` of 64K occlusion queries, slot = serial mod 64K, chunks of 4K reset from the host
  (`vkResetQueryPool`, Vulkan 1.2 host query reset: check the feature on N1X) before first use, after waiting for the timeline
  value of the chunk's previous lap (in practice never waits: 64K draws = 15-30 frames). Replaces `NSLOT*OQ` and the
  per-submission reset: **the 4096 bound is gone**; the `occlusion[OQ]` stack array of the completion thread goes.
- **Query per draw**: begin/end around the recorded draw, inside the pass, on every exit path (a draw skipped after the begin
  still ends its query, flagged). Draws not recorded leave their slot reset -> read with the availability bit, counted 0 and
  as "unrecorded". Only draws inside a counting window need a query: the dry pass marks windows (from a samples reset to the
  last report before the next reset); draws outside any window take a serial but no query.
- **Reports** become `{addr, lo, hi}` (serial range since the last reset), named in stream order by the dry pass. No kind-4
  posts, no epochs, no `g_oq_counting`.
- Verify (log `occ2-s1`): queries a frame, slots used per submission (max), ring waits (expect 0), and **per report: the R4
  sum vs the old span sum** for the same report on the same frames (must be equal with the precise bit; log the mismatch
  count and the largest difference). Picture unchanged (values still the old path's). RED overflow line count must be 0.
  Cost: replay ms and GPU ms with/without (per-draw queries on N1X are unmeasured).

### Step 2 - GPU resolve and stream-order landing; old substitutes removed (`DM_VK_OCC2=s2`)
- **Resolve point**: at the first pass end after one or more reports (and at each submission end), the recording thread that
  records that pass end (replay or worker; the dry pass names it) records: `vkCmdCopyQueryPoolResults(64_BIT|WITH_AVAILABILITY)`
  of the slot span `[min lo, max hi)` into a device-local result ring, a barrier, and one small compute dispatch that, per
  pending report `{lo, hi, out}`, sums the available slots and writes the 64-bit count **into the report's guest word** (the
  guest buffer the GPU uses, with the same GPU-write tracking as a dispatch's storage write: `vk_gpu_pending`, write-back for
  copied pools) and into a **predicate ring** (step 3). One dispatch per resolve point, not per report.
- **Replay-time write**: only the timestamp half (`addr+8`) and clear-flag reports keep the replay-time CPU write
  (`report_now`); the samples-passed counter word is written by the GPU only, as on the console. Completion thread: no
  samples sums, no writes.
- **Removed on this path**: occkey table and `occ_key_make`, occaddr, occland, post modes 7/8, `DM_VK_OCC_FIXED` for
  samples-passed, the `1 << 20` and "1 if lost" fallbacks. (Kept untouched on `DM_VK_OCC2=0`.)
- Conditions still CPU (r52 rule), so conditions are still "draw" for same-stream reports; what changes is what readers of
  the report words see: this frame's real count, landed in GPU stream order.
- **Landing bound (only if step 0 item 2 shows guest reads before landing)**: the console's GPU is at most about one frame
  behind; ours can be several submissions behind (NSLOT 6). Emulate the bound, not a value: before the guest passes the point
  where it was seen to read (likely its next present or a frame later), wait for the submissions holding those reports.
  Counters: waits a window, ms waited, reads still before landing (must be 0). This is a timing guarantee with real data,
  allowed by (d).
- **Uniform shadows over report words** (if step 0 finds them): a shadow taken at replay must not read the word before the
  GPU wrote it: for those ranges bind the guest buffer directly or copy on the GPU after the resolve point. Counter: shadows
  overlapping unlanded report words (must be 0).
- Verify in game: Kokiri sunrise spot (haze/over-exposure: must match the 0 / keyed look, not the 4096 look), terrain
  flashing at low fps (PAR4 settings), counters above, `[RED] ootu32` lines 0.

### Step 3 - render conditions on the GPU (`DM_VK_OCC2=s3`)
- **Classification in the dry pass** (replaces the r52 rule on this path): a condition whose word(s) are samples-passed reports
  still in the predicate ring (any age, this submission or in flight or landed: uniform, so no fence-timing dependence) is a
  **GPU condition** with a named predicate index. Words that are CPU-written (clear flags, `NVN_COUNTER_WRITTEN`), timestamps,
  or never reported stay CPU-evaluated exactly as today (bak88/MKW and TotK-style clear gating unaffected).
- **Predicate**: for OoT's shape (mode 1, y word = 0 or a CPU-known constant at replay), the resolve compute writes
  `count != 0` (32-bit) per report; mode 0 uses the inverted bit; a known non-zero y is compared in the same compute (the
  dry pass passes y). Both words reports: compute compares both sums. Anything else: fallback (below).
- **Recording**: the gated draws are recorded under `vkCmdBeginConditionalRenderingEXT(pred ring, index*4, inverted?)`,
  scoped inside one render pass instance (ended at pass end, condition change or `SetRenderEnable`, re-begun in the next
  pass). Clears under a GPU condition must be `vkCmdClearAttachments` (affected by the extension: `gfx_vk_exec.c:2376, 2401`),
  not folded into a load op and not `vkCmdClearColorImage` (not affected; `gfx_vk_exec.c:1410, 2274, 4699`). Dispatches are
  outside passes, so never inside a conditional block (today dispatches ignore `g_rc_skip` too, `gfx_vk_exec.c:5265`).
- **Ordering**: the predicate is written by the resolve dispatch before the pass that reads it; the barrier into the pass must
  include `CONDITIONAL_RENDERING_READ` at `CONDITIONAL_RENDERING` stage (r89 narrowed our full-barrier masks to the device's
  own access kinds: add this one).
- **Report not resolved yet** (report in the still-open pass): the dry pass inserts a pass end + resolve point before the
  condition (a GPU-side split, no CPU wait). HeyListen's >= 557 draws means this should be near 0; counter
  `forced resolves`. Extension missing or a shape the compute does not handle: HeyListen's stop rule as the last resort (submit,
  wait, resolve on CPU, evaluate), counter `cpu waits` (must be ~0 in OoT).
- Verify: condition census becomes `gpu (real, same frame) / cpu flag / cpu timestamp / fallback`; HeyListen expects ~350 of
  ~920 OoT conditions off a frame; GPU time with/without; Market/Kokiri buildings and NPCs present at load-in and while turning
  (HeyListen's mode-5 failure).

### Step 4 - workers get occlusion segments (R4 complete)
- With slots named by serial, reports as named `{addr, lo, hi, pred}` entries and resolve points named by the dry pass,
  REPORT_COUNTER (samples), RESET_COUNTER and GPU conditions no longer set `WR_SPREP` / `WR_SPRESET` / `SW_RCSTOP`, and workers
  no longer `seg_wk_abort(WA_OP)` on them (`gfx_vk_exec.c:5203-5210`, `5645-5648`). The replay-time timestamp write stays a
  dry-pass job in stream order.
- Verify: segments handed to workers / total (target: the ~70% now held by occlusion become eligible; with par_s1e2's flag fix),
  segments refused by reason (WR_SPREP/WR_SPRESET must be 0), and R4 sums equal between serial replay and workers on the same
  frames (`DM_VK_PARALLEL=0` vs `=4`).

### Step 5 - adoption
`DM_VK_OCC2=1` default after section 5 passes; `DM_VK_OCC2=0` keeps the old path whole. Old-path code stays until NPARK
retires it.

---------------------------------------------------------------------------------------------------------------------------------

## 5. Risks and tests

Risks:
- **R1 Late readers.** The real count lands at GPU time. A guest CPU read (or a uniform shadow taken at replay) before that sees
  the old word. This is what made real-after-fence tremble (S4). Step 0 item 2 measures it; step 2's landing bound / direct
  binding is the answer, not a substitute value. Highest risk; do not skip step 0.
- **R2 Sun/haze magnitude** (HeyListen H4): needs precise queries (log says "(precise)" on N1X: ok) and real sample counts at
  the game's resolution; if the render scale differs from native, counts scale. Watch Kokiri sunrise spot.
- **R3 Cost** on N1X: per-draw query begin/end (~2-4 K draws a frame), one copy + dispatch per resolve point (Hyrule Warriors:
  450-513 passes a frame, so resolve only where reports are pending), forced pass splits. Measure replay ms, GPU ms, fps, > 33 ms
  frames.
- **R4 Proxy draws must really draw**: H1 did not hold in HeyListen's OoT, but our mesh/indirect translation differs: zero-count
  reports with "unrecorded"/skipped draws in their window are a translation bug signal (log them).
- **R5 Conditional rendering scope**: begin/end inside one pass instance; clears folded into load ops would escape the
  condition; secondary command buffers (if workers use them) need `conditionalRenderingEnable` inheritance. Feature check of
  `VK_EXT_conditional_rendering` on the N1X driver at start; fallback counted.
- **R6 Report reused within a frame at one address**: the latest report in stream order at that address owns the predicate
  index and the guest word (HeyListen `occ_latest`); the dry pass's map must follow stream order across segments.
- **R7 Clear-flag reports** (`NVN_COUNTER_WRITTEN`) and other counter types stay replay-time CPU writes (HeyListen
  `vk_replay.cpp:1944-1949`: a GPU-timeline write there caused extra clears / black frames).

Tests (each A/B `DM_VK_OCC2=0` vs `1`, same build, same save):
- **OoT (target)**: Kokiri sunrise spot (former 4096 over-exposure/haze): same look as today's keyed result, steady over 60 s;
  terrain flashing standing and walking at PAR4-like load (37-40 fps, many > 33 ms frames) and at PAR2: flashing must not
  correlate with frame-time spikes any more; Market: buildings, fountain, NPCs at load-in and while turning; no white
  characters / washed-out frames; counters: RED overflow 0, `[RED] ootu32` 0, cpu waits ~0, forced resolves small, reads before
  landing 0, R4 sum mismatches 0; fps / 1% low / > 33 ms frames within spread or better (GPU skips hidden objects); worker share
  after step 4.
- **Pokemon Legends Z-A**: the dialogue / label box over a moving character steady (former occkey flicker); report addresses
  stable, so also check the per-address latest-report rule.
- **Hyrule Warriors**: frame time (many passes), no missing units/effects, resolve points per frame; any conditions on reports
  (unknown usage today: census first).
- Regression spot checks: Donkey Kong (r50: skin SSS magnitude), MKW / TotK-style clears gated on CPU flags (must stay CPU).

---------------------------------------------------------------------------------------------------------------------------------

## 6. Implementation status (2026-11-21: steps 0 and 1 landed, syntax-checked only, not run; steps 2 / 3 in 6.9, their cost work occ-s3fast in 6.10)

Code tagged `occ-s0` / `occ-s1`: `host\dm\windows\gfx_vk_core.c` (section "occ-s0 / occ-s1" before `completion_main`, plus hooks in
`vk_init`, `slots_init`, `post_op`, `oq_end`, `vk_oq_draw`, `vk_oq_reset`, `occk_get`, `sub_counter`, `vk_ubd_read`, `mtl_sub_copy`,
`vk_par_wk_begin/end`, `mtl_sub_begin`, `vk_pass_frame`, the completion thread), `gfx_vk_exec.c` (`g_oser`, `rc_occ_t`, `rc_note_diag`,
`occd_cond` / `vk_occd_exec_log`, `x_read`, `vk_end_pass`, the draw paths, `exec_ops`, the dry pass, `seg_report` / `seg_oser_chk`,
`seg_par_handout`, `seg_job_main`, `vk_seg_part_join`, `seg_list`), `gfx_vk.h`. `gfx_vk_eden.inc`, `gfx_vk_tex.c`, `hle_nvn.c`
unchanged. Backups: `_backup\occ_s01\`. No descriptor cache or cache key added (`g_vk_storage_want` untouched).

### 6.1 Switches
- `DM_VK_OCCDIAG=1` (live through `x_read`, i.e. `dm_env_gen` / DM_CTL `env` lines): step 0's instruments, log only, every 600 presents.
- `DM_VK_OCC2=s1` (read once at device init, default unset = 0): step 1. Any other value except `0` logs `[RED] occ-s1: ... not known`
  and keeps the old path. s1 is taken only when the old path has real queries (`g_oq`: occkey on, or `DM_VK_OCC_FIXED=-1`); otherwise a
  line says so.
- Unset / 0: the old path. What it gains: one `g_oser++` per draw-type record (TLS), one `g_vk_pends++` per pass end, one global test
  per draw (`if (g_occ2)`), per report / reset / span end, a 24-byte larger `post_op`, a 32-byte larger `rc_occ_t`, the occkey hit's
  age computed (two subtractions). Values, posts, epochs, conditions: unchanged.

### 6.2 Stream serial (both steps)
- `g_oser` (`_Thread_local`, gfx_vk_exec.c): every DRAW, DRAW_IDX, MULTI_DRAW (one per record) and DRAW_TEX record advances it,
  recorded or not; a draw being recorded is serial `g_oser - 1`. No atomics.
- Dry pass: `g_sd.oser` starts at the replay thread's `g_oser` at the list's start and counts the same records; each segment carries
  `seg_desc.oser0` (its first record's serial), each REPORT/RESET decision `seg_dec.oser`. The replay thread checks itself against them
  (`seg_oser_chk` at reports and resets, the hand-out, the list's end) and takes the dry pass's value on a difference (`[RED] occ-s1:
  ... the dry pass's taken`; counter "stream serials off the dry pass's", must be 0). A worker starts at `oser0`
  (`seg_job_main`), its end serial is checked at the part's join; the helper context saves/restores `g_oser` with the rest of the
  segment TLS. The replay thread advances by `ndraw` at a hand-out (as `g_occ_draws`).

### 6.3 Step 1 (`DM_VK_OCC2=s1`)
- Ring: one `VkQueryPool` of 65,536 occlusion queries, slot = serial mod 64K, 16 chunks of 4,096. A chunk is reset for a lap before its
  first use in it (`occ2_prep`): host `vkResetQueryPool` when `hostQueryReset` is enabled (it is chained with the queried 1.2 features;
  `g_vk_hqr`), else `vkCmdResetQueryPool` in the part's upload command buffer (logged at start). Before the reset it waits for the
  last submission that recorded into or read the chunk's previous lap (`g_occ2_ck[].fence`, set by draws, report windows, spans and
  hand-outs; wait through `mtl_wait` = completion thread done, 2 s bound). A previous lap still in the open submission (> 60K draws in
  one submission) or a wait timeout is `[RED]` and the draw is left without a query.
- Per draw: `vk_occ2_begin(cb, g_oser - 1)` right after `vk_oq_draw` in `draw_mesh_tasks`, `draw_body`, `draw_idx_body` (inside the
  pass, while counting = `g_oq_counting` / the worker view's), `vk_occ2_end` after `draw_body` (in `draw`) and after `draw_idx_body`
  (wrapper `draw_idx`): every exit path; also at every pass end, report and reset (through `oq_end`). Precise bit when the device has
  `occlusionQueryPrecise`.
- Vulkan allows only one active occlusion query per command buffer (VUID-vkCmdBeginQuery-queryPool-01922), so s1 does not record the
  old span queries; the old path's spans become serial ranges (`post_op` kind 4 mode 2, `[addr, key)`) summed by the completion thread
  from the same per-draw queries (`occ2_sum`, 64-bit with availability; unavailable = no query since the reset = 0). occkey's table,
  occland, `DM_VK_OCC_FIXED` and every value written to guest memory are the old path's as configured; with the precise bit the sums
  equal the old per-span sums. The old 4096-per-submission pool is not begun on this path, so its overflow (and its "1 if lost") cannot
  happen there (a behaviour difference only where the old path overflowed).
- Reports: `occ_sp_report` gives each samples-passed report (while counting) its window `[serial at the last reset, serial now)`
  (`post_op.s1lo/s1hi`); at completion `occ_sp_done` compares the window's ring sum with the old epoch sum of the same report.
- Workers: segments with samples-passed reports / resets still stay on the replay thread (`WR_SPREP`, `WR_SPRESET`, unchanged). A worker
  segment inside a counting window records its draws' queries at `oser0 + k` (the hand-out resets their chunks in stream order,
  `vk_occ2_handout`); the worker only checks the chunk's lap. Its spans go into its view's posts as before.
- Not done in step 1: the dry pass does not mark counting windows (every draw after the first reset takes a query, as the old
  `g_oq_counting`, never cleared); the report windows are not handed over by the dry pass (the replay thread forms them from its own
  serial, which the dry pass's checks); kind-4 posts and epochs are still there (needed by the old path).

### 6.4 Step 0 (`DM_VK_OCCDIAG=1`)
- Item 1, landing age: per samples-passed report the present at replay, the class of the value written (occkey hit: present now minus
  the present its count was measured at, `occk_get`'s entry frame, 0..3; fixed value; the real count after the fence with
  `DM_VK_OCC_FIXED=-1`), folded per present and split by that present's frame time (<= 33 ms / > 33 ms, from `vk_pass_frame`); the
  completion thread's present when the GPU result was available (fence done) minus the replay present.
- Item 2, GPU-side readers of report words before landing: a table of samples-passed report words whose submission is not done
  (address, fence, present, serial, submission number; rebuilt when half full); checked by uniform shadows (`vk_ubd_read`, replay
  thread), buffer copy sources (`mtl_sub_copy`), UBO and SSBO binds (at the bind record: the draws read them later). Distances in
  submissions, presents and draws. Not checked: worker segments' reads, texture copies' sources, vertex / index buffers. The CPU
  guard-page part (fastmem protection of pending words) is NOT implemented (not trivial here: it needs the pcverify-style protection
  and fault attribution): CPU reads are not instrumented.
- Item 3, condition census (`occd_cond`, once per evaluated condition: the dry pass's decision or the replay's): x and y word class
  (samples report of this submission / in flight / landed, not measured, other counter = CPU-written flag, timestamp, never reported,
  reported before the diagnostic was on), mode, y word 0 in memory, the samples report's age once landed, same submission or not,
  distance report -> condition in draws (serials) and in pass ends (replay thread only; "not known" for dry-pass decisions). Needs the
  `render_cond` profile key (reports are noted only then, as `rc_note`).
- Item 4, overflow context: at the first `[RED] occlusion queries: more than 4096 passes` of a submission the counts so far; at the next
  submission's start one `[RED] occ-s0: occlusion query overflow context` line with the submission's draws, passes (replay thread),
  samples-passed reports, resets (up to 32 lines).

### 6.5 Log lines (every 600 presents, from the first submission start after the 600th present)
- `occ-s0: report landing, last N presents: samples-passed reports R (r a frame); value written, by age in presents
  0/1/2/3/fixed/real-after-fence: in frames <= 33 ms a/b/c/d/e/f, in frames > 33 ms ..., frame time not known ...; frames F, over
  33 ms S; GPU result available (fence done) 0/1/2/3/4+ presents after the replay: ...`. The S3 question: does the age (1/2/3 and fixed)
  shift in the > 33 ms frames.
- `occ-s0: guest reads of samples-passed report words before their GPU result landed ...: <reader>: reads n, over an unlanded report
  word h (words w; submissions after the report 0/1/2+; presents 0/1/2+; draws after it min / avg)` per reader kind; `pending-word table
  full` (must be 0). R1: h > 0 with presents 0 means a GPU-landed value (step 2) arrives after that reader unless step 2 binds or
  copies on the GPU.
- `occ-s0: render conditions, last N presents: C (c a frame; decided by the dry pass D); mode 0 / mode 1; y word 0 in memory; word x:
  <class n, ...>; word y: ...; the samples report read: landed 0/1/2/3+ presents before ..., same submission s, an earlier one e;
  report to condition in draws 0/1-15/16-127/128-556/557-2047/2048+ ..., in pass ends 0/1/2-7/8-63/64+/not known ...`.
- `[RED] occ-s0: occlusion query overflow context: ...` (only when the old pool overflows).
- `occ-s1: last N presents: per-draw queries q a frame (by workers w), most in one submission m; chunk resets r, ring waits n (x ms;
  expected 0); reports compared c (window slots s a report), the window sum differs from the old span sum k (largest difference d); not
  compared: old spans lost, chunk gone, window longer than the ring; unrecorded slots in windows u, zero-count reports with unrecorded
  draws z; ring overflows and faults (RED, must be 0), draws whose chunk a hand-out had not reset, query ended in another command buffer,
  stream serials off the dry pass's; replay X ms a frame, GPU busy Y ms a frame` (`[RED]` prefix when any fault or mismatch). Cost: compare
  X / Y with a run without s1 (the same numbers as present.c's "replay per frame (last 600)" line). Up to 8 `[RED] occ-s1: report ...
  window ... sums ..., the old span sum ...` lines name mismatches.
- `occ-s0: replay X ms a frame, GPU busy Y ms a frame` when only the diagnostic is on.

### 6.6 Expected (OoT, Kokiri / Market)
- Census: ~900+ conditions a frame, mode 1, y word "never reported" with memory 0, x word "samples report of this submission" ~94%
  (in flight the rest), same submission, distance >= 557 draws (HeyListen median ~1,690), pass ends many (> 8).
- Landing: ~1,800-2,000 reports a frame; value age mostly 1 (keyed from the frame before), fixed ~1-2%; more 2/3 and fixed in > 33 ms
  frames if S3's hypothesis holds; GPU result available 0-1 presents after replay (more 1-2 at PAR4-like load).
- Reads before landing: unknown (the open question of R1).
- s1: per-draw queries ~ the frame's draws after the first reset (OoT ~1,900-2,200 a frame), most in one submission well under 64K, ring
  waits 0, mismatches 0, ring overflows 0, windows ~1.1-1.2 slots a report, zero-count reports with unrecorded draws small (R4 signal
  otherwise). Picture identical to s1 unset. Cost: + per-draw begin/end on N1X (unmeasured; the replay / GPU ms in the line).

### 6.7 Known limits
- Not run, not built with ninja: syntax-only compiles of core, exec (with eden.inc), tex, shader, present, win_glue (known warnings only).
- A report window longer than 61,440 draws (a game that resets once and reports for a long time) is not compared (counted, `[RED]`);
  step 2 needs a running base per window for that, not a longer ring.
- Step 0 item 2 checks only replay-thread readers; item 3's pass-end distance is not known for dry-pass decisions; census entries of
  reports noted before `DM_VK_OCCDIAG` turned on are counted as such.
- Report words are assumed 16-byte aligned (NVN counters are).

### 6.8 What step 2 needs next
- A resolve point per pass end after reports (and per submission end), named by the dry pass: `vkCmdCopyQueryPoolResults(64_BIT |
  WITH_AVAILABILITY)` of `[min lo, max hi)` into a device-local result ring, a barrier, one compute dispatch summing each pending
  `{lo, hi, out}` and writing the count into the report's guest word (GPU write tracking as a dispatch's storage write:
  `vk_gpu_pending`, copied-pool write-back) and into the predicate ring of step 3. The report windows `{addr, lo, hi}` then come from
  the dry pass (`seg_dec.oser` of the reset and the report are already there).
- The replay-time write keeps only the timestamp half (`addr + 8`) and the other counter types; the samples word written by the GPU
  only. Completion thread: no samples sums, no writes; the old kind-4 posts, epochs and `g_oq_counting` dropped on that path.
- On the new path remove occkey / `occ_key_make`, occaddr, occland, post modes 7/8, `DM_VK_OCC_FIXED` for samples, the `1 << 20` and
  "1 if lost" fallbacks (all kept for `DM_VK_OCC2=0`).
- From step 0's numbers: if item 2 shows uniform shadows / binds over unlanded words, bind those ranges directly or copy them on the GPU
  after the resolve point (counter: shadows over unlanded words, must be 0); if item 1 / CPU readers show reads before landing, the
  landing bound (wait at the present the guest reads at).
- A running base for windows longer than the ring; counting windows marked by the dry pass (draws outside any window take no query).
- Step 4 then: REPORT_COUNTER (samples) / RESET_COUNTER no longer set `WR_SPREP` / `WR_SPRESET`, workers no longer
  `seg_wk_abort(WA_OP)` on them in `exec_ops`, the report's commit (post, rptfix queue entry) carried in stream order to the hand-out,
  the window start taken from the reset's `seg_dec.oser`; the serial naming, chunk resets at the hand-out and the worker's own queries
  are already in place.

### 6.9 Steps 2 and 3 (2026-11-21: `DM_VK_OCC2=s2` / `s3`, syntax-checked only, not run)

In-game result of steps 0/1 that this builds on (OoT 0400486024DEC000): 2.36M reports compared, 0 mismatches against the old span sums, 0 ring waits / faults, no 4096
overflow; GPU results ready 0-1 presents after replay; GPU-side reads over unlanded words 0 of ~4.7M; ~600-700 conditions a frame, all mode 1, y word never reported and 0
in memory, x = a samples report of the same submission ~95%, distance mostly 557-2047 draws.

Code tagged `occ-s2` / `occ-s3`: `gfx_vk_core.c` (section "occ-s2 / occ-s3" before `occ_log`, hooks in `occ2_init`, `oq_end`, `vk_oq_reset`, `report_now` /
`vk_rpt_flush_` (RQ_HALF), `sub_counter`, `vk_occ2_begin`, `occ2_prep`, `vk_occ2_handout`, `par_view_take`, `vk_par_wk_begin`, `vk_cmd_split`, `mtl_sub_commit`, the
completion thread (post mode 9), `vk_dev_access`, `vk_pass_frame`, `occ_sub_begin`, `occ_log`, `vk_occd_read`), `gfx_vk_exec.c` (`rc_occ_t.rs`, `rc_note`, `rcg_*`,
`render_cond(_e)`, `seg_cond`, `seg_report`, `vk_end_pass`, `draw` / `draw_idx` and the three draw bodies, `clear_draw`, `clear_color`, `clear_ds`, `mtl_queue_activate`,
the dry pass's REPORT_COUNTER / RENDER_COND / draws / clears, `seg_dry`, `vk_seg_part_join`, `seg_job_main`, `WR_GPUCOND`), `gfx_vk_present.c` (the read watch,
`vk_hwr_*`), `gfx_vk.h`. New: `host\dm\windows\shaders\occ2_resolve.comp` and `vkgen\occ2_resolve_comp.spv.h` (generated with `glslangValidator -V --target-env vulkan1.1`
from `C:\msys64\clangarm64\bin`, checked with `spirv-val --target-env vulkan1.1`, words written as the other `.spv.h` headers are). Backups: `_backup\occ_s23\`.
`gfx_vk_eden.inc`, `gfx_vk_tex.c`, `gfx_vk_shader.c`, `hle_nvn.c` unchanged. No descriptor cache and no cache key added (one set per resolve from `vk_alloc_set`;
`g_vk_storage_want` untouched).

#### 6.9.1 Switches
- `DM_VK_OCC2` (device init): unset / `0` the old path exactly (occkey, occaddr, occland, `DM_VK_OCC_FIXED`, posts, epochs, r52); `s1` step 1 as before; `s2` step 2;
  `s3` steps 2 and 3 (the full new path; default stays off in this build). Anything else: `[RED] occ-s2: DM_VK_OCC2=<v> is not known`, old path. `s2` / `s3` do not need the
  old pool (`g_oq`). `s3` without `VK_EXT_conditional_rendering` enabled on NXN1X's device (Eden enables it when the driver has it; checked in its extension list) runs as
  `s2` with the line `occ-s3: VK_EXT_conditional_rendering is not enabled on the device: DM_VK_OCC2=s3 runs as s2`.
- `DM_VK_OCC_RDWATCH=1` (live, `s2` / `s3` only; diagnostic, off by default): guest CPU reads of report words (6.9.4).

#### 6.9.2 Step 2: GPU resolve, stream-order landing
- **Report** (`occ2_report`, replay thread, in stream order): `{address, timestamp (gpu_ticks at replay), report sequence, window [cut, serial now), base slot}`. The
  report sequence counts samples-passed reports in stream order; the dry pass names the same numbers (`g_sd.orseq`, `seg_dec.rseq`, checked by `vk_occ2_rseq_chk`,
  "report sequences off the dry pass's" must be 0). Only the timestamp half (bytes 8..15) is written at replay (rptfix's queue entry with `RQ_HALF`; left to the
  completion thread as post mode 9, bytes 8..15 only); the counter word is written by the GPU only.
- **Resolve point** (`occ2_resolve`): the first pass end after reports (`vk_end_pass` -> `vk_occ2_pass_end`, replay thread only), each part's end (`vk_cmd_split`, after
  the join), the submission's end (`mtl_sub_commit`, after the join, before the readback), a condition's forced resolve (s3), a part's join (deferred ones). Stream order by
  construction (the replay thread records reports and resolves; workers record neither until step 4); the dry pass does not name the points yet (step 4 needs it once
  reports go to workers). Records, outside a render pass: an empty `vkCmdBeginQuery/EndQuery` for each slot of the span no draw began in its lap (bit per slot,
  `g_o2_bits`; draws not recorded: a false condition, a draw `prepare_draw` dropped, a failed worker chunk), so the waited copy never waits on a query no earlier command
  ends; a full barrier; `vkCmdCopyQueryPoolResults(64_BIT | WITH_AVAILABILITY | WAIT)` of `[min lo, max hi)` (two copies when it wraps) into RES of one device-local buffer;
  a TRANSFER -> COMPUTE barrier; **one** dispatch of `occ2_resolve.comp` (one invocation per entry: report windows -> `REC[record] = {count, timestamp}`,
  `PRED[record] = count != 0`; running bases; s3's own predicates); a COMPUTE -> TRANSFER | COMPUTE | CONDITIONAL_RENDERING barrier (SHADER_WRITE -> TRANSFER_READ |
  SHADER_READ | CONDITIONAL_RENDERING_READ); `vkCmdCopyBuffer` of each report's 16 bytes from REC into the buffer `vk_buf_at` gives for its address (imported pages, or a
  copied pool's device-local copy), one call per destination buffer; a full barrier; `vk_gpu_wrote` per run of adjacent words (the GPU-write tracking: `vk_gpu_pending`
  for CPU readers, `sub_gpu_wrote` -> copy-back of copied pools, the combined1120 mask for later late reports). The latest report at one address in a resolve owns its
  word (R6). **No atomics anywhere** (NPARK's rule: none on imported memory; none needed: each entry writes its own words of a device-local buffer); 64-bit sums as two
  32-bit words (no shaderInt64).
- **Device-local buffer** (`g_o2r.buf`, 2.3 MB, zero-filled once): RES 64K x 16 B (query results by ring slot), REC 64K x 16 B (by record = report sequence mod 64K), BASE
  4K x 8 B, PRED (64K report predicates + 4K condition predicates) x 4 B; usage STORAGE | TRANSFER_SRC | TRANSFER_DST (| CONDITIONAL_RENDERING with s3).
- **Windows**: a reset opens a window at the serial (`occ2_reset`); a report's window is `[cut, serial)`; a window open for more than a quarter of the ring gets a
  running base at the next resolve point (kind-1 entry: `BASE[new] = BASE[old] + sum[cut, now)`, cut moves to now), so a window longer than the ring still counts every
  draw. Counting windows are not marked by the dry pass: as in s1, every draw after the first reset takes a query (limit).
- **Workers**: segments with samples-passed reports / resets stay on the replay thread (unchanged); a resolve whose span meets a segment handed out in the open part is
  deferred to the part's join (`vk_occ2_join_end`), where the job's begun-query bits come in (`vk_occ2_job_join`; a chunk left out keeps its slots unbegun: they get empty
  queries). `occ2_prep`'s `vkCmdResetQueryPool` into the upload buffer now takes `vk_up_lock` (workers record into that buffer; s1 had the same unguarded call).
- **Never a waited copy that cannot end**: the span is clamped to the last 60K serials before now (no slot of it in a chunk reset for a later lap); a slot of a chunk
  `occ2_prep` could not reset is never copied (that resolve's windows count empty, `[RED]`); a refused submission's chunks are reset again before their next use and its
  reports' predicates are not taken (`occ2_dropped`).
- **Landing bound** (`occ2_bound`, at each present, `vk_pass_frame`): the submissions holding the previous present's samples-passed reports must be done before the guest
  goes on (the console's GPU is at most about one frame behind; ours may be several submissions behind). Waits counted; "not reachable" = that submission still open.
  On in s2 / s3 (no switch); the read watch measures whether it is needed and enough.
- **Removed on this path** (kept untouched for `0` / `s1`): `occ_key_make` / occkey, occaddr, occland, post modes 7 / 8 for samples, `DM_VK_OCC_FIXED` for samples, the
  `1 << 20` and "1 if lost" values, kind-4 posts, epochs, `g_oq_counting` (the window state `g_o2w` instead), the 4096 pool's spans.
- Conditions stay CPU in s2 (r52): a report of this submission or in flight is unknown -> draw; a landed one now reads the real count in memory (`vk_oq_real` is 1 on s2).
- GPU-side readers: the step-0 table is kept on in s2 (`occd_word_add` per report; uniform shadows, copy sources, UBO / SSBO binds checked); a GPU reader recorded after a
  word's resolve point reads the GPU's word and is not counted; a uniform shadow (a CPU copy at replay) over a word whose submission is not done is (must be 0). Not
  bound directly / copied on the GPU (step 0 measured 0 such reads).

#### 6.9.3 Step 3: render conditions on the GPU (s3)
- **Classification** (`rcg_class`, the dry pass when it runs, else the replay; `rc_occ_t.rs` = 1 + the report sequence of the samples-passed report at an address): mode 0 / 1
  with word x (or y) a samples-passed report still in the ring (<= 64,000 reports old: any age, this submission, in flight or landed) and the other word never reported (its
  64-bit value read from guest memory at the condition, as the CPU path reads it) or a samples-passed report too. y = 0 (OoT): the report's own predicate; another constant or
  two reports: a predicate of its own (kind-2 entry, forced resolve). Mode 0 uses `VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT`. Other counters (CPU flags), timestamps,
  never-reported words on both sides: CPU as before. In the dry pass, a GPU condition whose constant word a GPU write may change is left to the replay (stop rule).
- **Recording**: `rec_xs_t.rcg_on/off/inv` (per queue state too). Each gated draw (`draw_body`, `draw_idx_body`, mesh draws) and clear gets its own
  `vkCmdBeginConditionalRenderingEXT` ... `End` inside its pass (`rcg_begin` after the draw's query begin, `rcg_end` in the `draw` / `draw_idx` wrappers and at every pass end):
  never across a pass end, a condition change or SetRenderEnable. Clears under a GPU condition: never folded into a load op, never `clear_whole`'s transfer clear; always
  `vkCmdClearAttachments` (masked clears: the clear draw, gated). Dispatches are never gated (as before).
- **Ordering**: the resolve's barrier C names CONDITIONAL_RENDERING_READ at the CONDITIONAL_RENDERING stage; `vk_dev_access` adds CONDITIONAL_RENDERING_READ to every full
  barrier on s3 (ALL_COMMANDS holds that stage), so the pass's own begin barrier has it too.
- **Report still pending** (its resolve not recorded yet): `o3_force` ends the open pass (a GPU-side split) and resolves there, counted "forced resolves" (expected ~0: 557+
  draws and many pass ends between). Not taken (resolve deferred to a part join, report out of the ring): the CPU fallback; a samples-passed report in flight is waited for
  first (`vk_occ3_wait`, "cpu waits", expected ~0), then the CPU rule as before.
- Workers: a GPU condition, the draws and clears it gates and the record that ends its gating keep their segment on the replay thread (`WR_GPUCOND`, "a GPU render condition
  or draws gated by one"); a worker starts with no GPU condition.

#### 6.9.4 CPU reads of report words (`DM_VK_OCC_RDWATCH=1`)
The step-0 guard-page idea is **not** used: the pages' protection belongs to nvn_dirty and the core (memory note: MapRegion / ProtectRegion make watched pages writable
silently), and a fault in the JIT's code would send it to its slow memory path. Instead: once a second one report word is watched with an ARM64 hardware watchpoint (loads of
its 8 bytes, EL0) set on every thread of the process by a helper thread (suspend, set the debug registers, resume; `gfx_vk_present.c`, modelled on r49's store watch, which is
not built in NXN1X). The first load from code in no module (the JIT: guest code) records the host PC, the present and whether the word's submission had completed; loads from
modules (our threads, drivers) are counted apart. Limits: one word at a time, missed if the guest reads it before the watch is armed (a few ms), or through another mapping
of the same memory; the PC is the JIT's, not a guest PC; no age split by frame time.

#### 6.9.5 Log lines (every 600 presents, from the first submission start after the 600th present)
- `occ-s2: last N presents: samples-passed reports R (r a frame), written into the guest's words by the GPU W (a later report at the same address in one resolve d); resolve
  points P (pass ends, part ends, submission ends, forced by a condition, part joins; deferred to a part join), compute dispatches D, query slots copied Q (empty queries for
  unrecorded draws E), guest copies C (runs of adjacent words X); per-draw queries a frame (by workers), most in one submission, chunk resets, ring waits (ms); reports before
  any reset, running bases; RED (must be 0): windows longer than the ring, reports outside the memory pools, over 32K reports at one resolve point, ring faults, draws whose
  chunk a hand-out had not reset, query ended in another command buffer, report sequences off the dry pass's, GPU-side reads before the GPU's write (words, table full);
  landing bound: waits (ms), not reachable; CPU reads of report words (...); replay X ms a frame, GPU busy Y ms a frame` (`occ-s3:` with s3; `[RED] ` prefix when any RED
  count is non-zero). The CPU-read part: `words watched, a guest read seen G: before its result landed B (presents after the replay 0/1/2+), after A (0/1/2+), none within
  8 presents, our own threads' reads; first host PCs of guest reads` or `not watched (DM_VK_OCC_RDWATCH=1)`.
- s3: `occ-s3: render conditions c a frame: on the GPU g (forced resolves, not taken: resolve deferred to a part join, report out of the predicate ring), CPU flag, CPU
  timestamp, CPU memory (never reported), fallback (cpu waits, ms; should be ~0); conditional rendering blocks a frame (gated draws, gated clears)`.
- Init: `occ-s2: DM_VK_OCC2=s2: ...` / `occ-s3: DM_VK_OCC2=s3: ... render conditions on such reports evaluated by the GPU (VK_EXT_conditional_rendering)`; `[RED] occ-s2:`
  lines (first 16) for faults; `[RED] occ-s2: the replay thread's report sequence ... is not the dry pass's`.
- With `DM_VK_OCCDIAG=1` the step-0 lines stay (their GPU-reader counts are then the s2 line's and reset by it; the landing-age line's value classes are the old path's).

#### 6.9.6 Expected (OoT, Kokiri / Market)
- s2: reports ~1,800-2,000 a frame, all written by the GPU (W = R minus duplicates; outside pools 0); resolve points mostly pass ends, ~a few hundred a frame at most
  (one per pass that holds reports), dispatches = resolve points; slots copied ~ the frame's draws after the first reset; empty queries small (draws dropped by CPU-flag
  conditions); RED counts 0; GPU-side reads before the GPU's write 0; landing bound waits ~0 at 60 fps, some in > 33 ms frames (that is the bound doing its job); cost: +1 copy
  + 1 dispatch + a few copies per resolve point; replay ms ~ s1's (no occkey hash per report), GPU ms + small.
- s3: ~600-700 conditions a frame, nearly all on the GPU (mode 1, y = 0: the report's own predicate), forced resolves ~0, fallback ~0, cpu waits 0, CPU flag / timestamp as
  the step-0 census (~3-5%); HeyListen expects ~350 of ~920 OoT conditions off a frame (draws skipped on the GPU: GPU ms may drop).
- Picture: s2 = this frame's real counts in the report words (the sun / haze from real visibility, steady across frame-time spikes); s3 = hidden proxies' objects not drawn.

#### 6.9.7 Known limits
- Not run, not built with ninja: syntax-only compiles of core, exec (with eden.inc), tex, shader, present, win_glue (known warnings only: core `_GNU_SOURCE`, exec dangling-else).
- The dry pass does not name resolve points or mark counting windows (every draw after the first reset takes a query); fine while workers record no reports (step 4).
- A uniform shadow over an unlanded report word is counted, not fixed (no direct bind / GPU copy after the resolve; step 0 measured none).
- The landing bound is one present deep and has no switch; whether guest reads come before it is what the read watch measures.
- A dropped (refused) submission's reports never land (their words keep what was there); RED for the submission as before.
- Conditions whose constant word is a CPU-written flag or timestamp, or whose report left the 64K ring, stay on the CPU (counted); a condition whose report is in the open
  submission and the GPU cannot take it (deferred resolve) draws (unknown), as before.
- The read watch samples one word a second and sees only loads through the address the replay uses.

#### 6.9.8 Test plan (same build, same save)
1. `DM_VK_OCC2=s2`, OoT Kokiri: the `occ-s2` line: RED counts 0, W ~ R, GPU-side reads before the GPU's write 0, `[RED] ootu32: no storage view` absent. Kokiri sunrise
   over-exposure spot: steady, no 4096 haze; light no longer trembles; terrain flashing standing and walking at PAR4-like load: not tied to frame-time spikes.
2. Same with `DM_VK_OCC_RDWATCH=1` (DM_CTL env line, live): CPU reads before their result landed B: if B > 0 with presents 1+, the landing bound is not deep enough; note the
   ages and PCs.
3. `DM_VK_OCC2=s3`: the `occ-s3` line: on the GPU ~ all, forced resolves ~0, fallback ~0, cpu waits 0, blocks ~ gated draws + clears; Market buildings, fountain, NPCs at
   load-in and while turning (HeyListen's mode-5 failure absent), no white characters / washed-out frames; GPU ms vs s2.
4. A/B `DM_VK_OCC2=0` vs `s3`: fps, 1% low, > 33 ms frames (present.c lines), replay / GPU ms (the occ line and "replay per frame"), the picture at the spots above.
5. With workers (`DM_VK_PARALLEL=2`): "deferred to a part join" small, RED 0, S1e "kept ... by a GPU render condition" counted.
6. Regression spot checks with `0`: unchanged by construction (every change is under `g_occ2 >= 2`, except `rc_note`'s extra argument, 0 there).

### 6.10 occ-s3fast (2026-11-21: s2 / s3 cost, syntax-checked only, not run)

Why: s3 gave the right picture (OoT Kokiri) but cost ~31 resolve points a frame (each: a full barrier, waited query copy, barrier, dispatch, barrier, ~50 tiny guest
copies, full barrier), ~700-800 conditional rendering blocks a frame (one per gated draw), and NPARK's A/B (logs_poolcopy\occs0_210159 / occs2_210639 / occs3_205719)
shows the cost is the frames over 33 ms (24.8 -> 52-54 per 10 s) and GPU ms per draw (+0.5-1 us), not replay ms per draw (~10 us in all three).
Code tagged `occ-s3fast`: `gfx_vk_core.c`, `gfx_vk_exec.c`, `gfx_vk_present.c`, `gfx_vk.h`, `shaders\occ2_resolve.comp`, `vkgen\occ2_resolve_comp.spv.h` (rebuilt:
glslangValidator -V --target-env vulkan1.1, spirv-val --target-env vulkan1.1 clean; 2174 words). Backups: `_backup\occ_s3fast\`. No descriptor cache or cache key added.

#### 6.10.1 Resolve points only where something reads (the plan)
- A walk of each top-level list before its replay (`gfx_vk_exec.c o3p_list / o3p_walk`, s2 / s3; same flow as `ind_prescan` / the dry pass, DM_VK_SKIP_OPS honoured) names,
  by stream serial (draws before the record, as `g_oser`): readers of samples-passed reports (s3: render conditions whose x or y word is such a report, by the list's own
  reports or the report table; s2 / s3: UBO / SSBO binds and copy sources over a report word, by page, or over a word pending at the list's start, `vk_occ2_unres_in`)
  with the newest report sequence each may read; and opportunities (SET_RT to other targets, copies, fills, texture copies / clears, dispatches).
- `gfx_vk_core.c occ2_opp` at every replay-thread pass end, opportunity record with no pass open (exec_ops), part end (`vk_cmd_split`, no longer an unconditional
  resolve), hand-out and list start: resolve now when `vk_o3p_want` says a reader of a pending report comes before the next opportunity the plan knows of (strictly: a
  draw between), or a bound needs it (`occ2_must`: a condition's own predicate, a running base due, 8K pending, the oldest pending window half a ring old); else keep
  pending. One resolve point then batches every pending report. Unchanged / still there: the submission end resolves the rest (guest-visible: fences, the landing
  bound, CPU reads); `o3_force` (a GPU condition whose report is still pending ends the pass and resolves, counted); new: a copy reading a pending report word resolves
  first (`O2R_READ`). No plan for a list (table full): every opportunity, as before. Worker hand-outs: the plan's opportunities inside the handed-out segment are skipped.
- Expected OoT: resolve points ~1-4 a frame (one before the first pass holding GPU conditions on the frame's reports, plus submission ends), dispatches = resolve
  points, forced resolves ~0, "opportunities passed over" ~30 a frame.

#### 6.10.2 The resolve itself (Eden's mechanism)
Reused from Eden (no code or shader copied; the C host keeps its own): `vk_query_cache.cpp` SamplesStreamer::PresyncWrites (250-359: resolve only at sync points;
`ApplyBanksWideOp` 445-465 / 272-301: one vkCmdCopyQueryPoolResults per contiguous slot range), QueryCacheRuntime::SyncValues (1871-1963: guest words copied with one
vkCmdCopyBuffer per destination buffer, regions per word), Barriers (1843-1868, called from `query_cache.h` NotifyWFI 487-499: one barrier before and one after the
guest copies, none before the query copy). Here: the windows' slots as sorted runs merged within 256 slots (never across a worker segment of the open part), one waited
copy per run (not one span [min lo, max hi): a batch spans the frame); the dispatch also writes each unique guest word into a staging area STG (new, words 602112..,
32K x 16 B) in guest address order, so runs of adjacent words are one copy region; barriers: B (transfer -> compute) and two full ones (C before the guest copies: also
PRED's CONDITIONAL_RENDERING_READ; D after them); the old A before the query copy is gone (previous D orders RES / REC / STG reuse, the WAIT bit the queries).
Eden's `queries_prefix_scan_sum.comp` is not used: it computes running sums for Eden's single-accumulator model (one reset checkpoint per batch); our windows are
[cut, serial) per report with running bases, ~1.1 slots a report in OoT, so the per-entry sum is cheaper than a scan plus differences. No atomics anywhere (NPARK).
`vk_gpu_wrote` per run of adjacent words as before.

#### 6.10.3 Conditional rendering blocks merged (Eden's hcr_setup)
As Eden (`vk_query_cache.cpp` HostConditionalRenderingCompareValueImpl 1430-1459: same buffer and offset keeps the running block; Pause / Resume 1407-1428 at
render pass end `vk_scheduler.cpp` 918 and draws `vk_rasterizer.cpp` 628 / 795 / 976): `rcg_begin(cb, clear)` keeps the open block when its predicate (offset,
inversion) is the current one; `rcg_settle` after every RENDER_COND ends it unless the new condition gates by the same predicate; every pass end ends it. Invariant: a
block is open only while `rcg_on`, so every draw / clear inside one is a gated one. The draw wrappers and clears no longer end it. Expected OoT: blocks ~ conditions
(~700 a frame; each condition gates ~1 draw), fewer where consecutive conditions share a report.

#### 6.10.4 S1e "a worker segment's place among the post ops is gone ... its 0 post ops"
Root cause: `vk_par_merge_posts` replaces a placeholder by the segment's n post ops, moving every later placeholder by n - 1; under s0 / s1 n >= 1 (kind-4 occlusion
spans), under s2 / s3 n = 0 (no span posts), so later placeholders moved one back, before the `post_at` their forward search began at. Fix: the views handed out and not
merged are listed (`par_out_*`); a merge and a late report's insertion (`vk_rpt_flush_`) move their `post_at` with the list.

#### 6.10.5 Landing bound
`DM_VK_OCC_BOUND=0` (live, default 1 = as before) leaves the per-present wait out for NPARK's A/B (counted "would-wait"); not made the default because no CPU-read
data exists yet (the RDWATCH result was cut from the old line). Decide with the test plan below.

#### 6.10.6 Log lines (every 600 presents; each under ~400 characters)
- `occ-s2|3 (1/5)` reports, words written, resolve points by reason (plan / pass ends, part ends, submission ends, forced, joins, copies reading one); `(2/5)`
  opportunities passed over, taken by a bound, deferred, dispatches, slots copied and query copies, guest vkCmdCopyBuffer calls / regions / runs; `(3/5)` per-draw
  queries, chunk resets, ring waits, landing bound waits (ms), not reachable, would-wait; `(4/5)` RED counts (must be 0); `(5/5)` cost: replay / GPU ms, resolve CPU ms
  (us a point), barriers recorded a frame, forced pass splits, resolve GPU ms (with DM_VK_PASS_TIMES=1; r138 also lists "occlusion resolve points").
- `occ-s3fast plan:` lists, readers named, opportunities, asked now / keep / no plan, walks off the replay's serials.
- `occ-s3 (1/2)` / `(2/2)` the condition census and blocks; `occ-s1 (1/3..3/3)`, `occ-s0` landing (1/2, 2/2), GPU-side readers (one line per kind), conditions (1/3..3/3).
- `occ-s2|3 rdwatch` (DM_VK_OCC_RDWATCH=1, live): watched words, guest reads before / after landing by age; and a line split by frame-time spike (calm / spike).
- `occ-s3fast frames:` frames, over 33 ms, spikes (> 33 ms or > 1.5x the previous), CPU ts / flag conditions and flips a frame calm / spike, landing waits a frame and
  ms calm / spike, frames with a wait calm / spike. `occ-s3fast flips by address (n/m)`: the 8 most-flipping CPU condition addresses (kind, evaluated, flips calm / spike).
- `occ-s3fast spike P` (+ `cont.`): 3 presents after a spike (at most one a second, 64 in all), the last 12 frames: ms, ts / flag conditions evaluated / drawn /
  dropped, flips, landing waits / ms (w = would-wait).
- DM_VK_OCC_CONDDIAG=1 (live, diagnostic: a barrier and a copy per CPU condition): `occ-s3fast terrain draws` (FS c72b6ea746c00433 / VS 1776c0973ce4bbd1: under no
  condition, GPU-gated, CPU-drawn, CPU-dropped); `occ-s3fast gate <addr> <class>` (6 CPU, 4 GPU addresses: draws / clears drawn / gated / dropped, ops on a terrain
  render target that frame, last gated draw VS / FS / TE, last target size / format / address); `occ-s3fast condread` (per address: conditions, CPU decision != the
  words the GPU saw at the command's place, GPU-written at the decision (in flight / this submission) and at the copy, last words).
- DM_VK_FLASHDIAG=1 (live, diagnostic: a 128x72 blit + readback per present): `occ-s3fast flash:` a one-frame jump in the 16x9 luma grid (cells, brighter /
  darker, box, largest jump, present intervals) and `occ-s3fast flash ctx frame F-1 / F / F+1` (ms, spike, CPU conditions and flipped addresses, GPU conditions, resolve
  points, landing waits).

#### 6.10.7 Risks
- The plan is advisory; correctness rests on o3_force, the copy-read resolve and the submission end. A plan off the replay (CALL decisions changed between walk and
  replay) shows as "walk off the replay's serials" and costs only extra / forced points. Forced resolves > 0 means the plan missed (pass split).
- GPU readers binding report words through vertex / index / indirect buffers or texture copies are not in the plan (as before: counted by "GPU-side reads before the
  GPU's write" only for UBO / SSBO / copy); with lazier resolves such a reader would now see an older value where s3 resolved at the pass end. Step 0 measured 0.
- Removing barrier A relies on the previous resolve's D and the WAIT bit; merged blocks rely on the invariant above (no other draw / clear recorded inside a pass while
  rcg_on: all are guest draw implementations).
- Diagnostics: CONDDIAG adds commands (off by default); FLASHDIAG samples every 15th pixel (small flashes may be missed); terrain relation covers render targets only.

#### 6.10.8 Test plan (same build, same save; NPARK runs RUN_OCC_S0 as the baseline)
1. `DM_VK_OCC2=s3`, OoT Kokiri: picture as now (sunrise, light, terrain); `occ-s3 (1/5)` resolve points ~1-4 a frame, forced (1/5 and occ-s3 1/2) ~0, `(4/5)` RED 0,
   `[RED] ootu32: no storage view` absent, no `[RED] S1e ... post ops is gone` with DM_VK_PARALLEL=2; `(5/5)` resolve CPU and barriers a frame small.
2. A/B s0 vs s3: fps, frames over 33 ms per 10 s, replay / GPU ms (present.c lines, `(5/5)`).
3. s3 with `DM_VK_OCC_RDWATCH=1`, then `DM_VK_OCC_BOUND=0` (both live): reads before landing (by age, calm / spike); if 0 with the bound off and the picture holds,
   the bound can go (default flip is a separate change).
4. `DM_VK_PASS_TIMES=1`: `(5/5)` resolve GPU ms; r138's "occlusion resolve points" group.
5. Trembling / flashing: `occ-s3fast frames` / `spike` lines, then `DM_VK_OCC_CONDDIAG=1` and `DM_VK_FLASHDIAG=1` at the spots (condread mismatches, gate lines, flash ctx).

#### 6.10.9 Eden and conditions that are not query results (measurement for later; nothing implemented)
`maxwell_3d.cpp` ProcessQueryCondition (1036-1104) first asks `vk_rasterizer.cpp` AccelerateConditionalRendering (1514: `query_cache.AccelerateHostConditionalRendering`,
`query_cache.h` 511-585). That looks up host queries at the two words; `vk_query_cache.cpp` HostConditionalRenderingCompareValues (1740-1820): neither word a query nor
GPU-modified in the buffer cache -> EndHostConditionalRendering, false -> Maxwell reads the words from guest memory on the CPU (1078-1094, in stream order on Eden's GPU
thread); a word GPU-modified in the buffer cache (or a pending query) -> on the GPU: one word zero in memory -> the other word's buffer used directly as the predicate
(CompareValueImpl 1430-1459), else `ConditionalRenderingResolvePass` (`vk_compute_pass.cpp` 389-431, `host_shaders\resolve_conditional_render.comp`): a compute dispatch
reads the 8 or 24 bytes from the buffer cache's buffer in stream order (after a barrier), writes a 32-bit predicate, barrier to CONDITIONAL_RENDERING_READ, then
VK_EXT_conditional_rendering. Mobile drivers (Qualcomm, ARM, Turnip) and low GPU accuracy end host conditional rendering. So CPU-written flags stay CPU-evaluated in
Eden too, but read at the command's place in the GPU thread's stream; GPU-written words go through the GPU.

## Appendix A. Live measurements, 11-21 (NPARK)

### A.1 Run comparison (Kokiri, build par_s1e; logs in logs_poolcopy\)

| run | fps mean | >33 ms frames / 10 s | terrain flashing (seen) | samples-passed reports filled from a key (prior-frame count) | of them 0 samples | fixed value (no key) | key zero/non-zero flips / reports | render conditions: real value / this submission (unknown -> draw) / false | 4096 overflow RED | segments to workers |
|---|---|---|---|---|---|---|---|---|---|---|
| RUN_PAR2 (2 idle workers) | 49.8 | 17.4 | hardly | 3.05M (7 windows) | 47.8% | 1.73% | 13.0% | 0% / 92.7% / 3.6% | 0 | 0% |
| RUN_AB_IMPORT_ASYNC1 | 46.7 | 13.2 | sometimes | 12.56M (16 windows) | 48.0% | 0.73% | 8.2% | 0% / 95.2% / 2.3% | 1 | n/a |
| RUN_PAR4 (4 idle workers) | 39.6 | 72.9 | a lot | 2.93M (5 windows) | 49.3% | 1.91% | 17.7% | 0% / 94.4% / 2.7% | 1 | 0% |

- No real count is ever written to a report address (occland off): every report gets the key's count from <= 3 presents ago, or the fixed value. There is no
  metric yet for "written after the guest read it / how many frames late" (Step 0 adds it).
- The flip rate does not track the flashing (PAR2 13% flips, least flashing); frame-time unevenness tracks it better.
- Workers 0%: a diagnostic check read NXN1X_VBYTES' unset -1 as on (fixed, par_s1e2), OoT's title default turns pass timestamps on (RUN_PAR*.cmd now set
  DM_VK_PASS_TIMES=0), and ~70% of segments hold samples-passed reports/resets.

### A.2 Render conditions while the Deku Tree floor holes were on screen (PAR2 run, ~11 min in)

- `render_cond diag`: dropped so far: **draws 0**, clears 304,701. Render conditions skipped no draw at all in that run; only clears.
- The 10 most-flipping condition addresses (0x3114b6xxxx) read words 0 / 1 and flip almost every evaluation (e.g. evaluated 1682, drawing 981, dropped 701,
  flips 585); "read while a GPU write to it was unfinished" 0.
- `DM_VK_RC_OCC=0` set live (every occlusion value unknown -> draw): "false" kept growing (+225 in 0.25 s), so these conditions read CPU-written flags in guest
  memory, not occlusion results.
- Conclusion: the terrain holes are not draws skipped by render conditions or occlusion. They are terrain tiles with missing data (see the gpuown / gwrange trace).
