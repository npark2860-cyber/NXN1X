# DLSS for Ocarina of Time (NXN1X), design

2026-11-21. Asked by NPARK: "이제 DLSS 붙이자". Title 0400486024DEC000, NVN2, build e0f3ccfa.

## 1. What the game does now

- The game issues no NVN DLSS command: `OP_DLSS` is 0 in every OoT log (replay_dlss never runs).
- It renders the 3D at the dynamic resolution (1138x640, 953x536, 854x480; a 16:9 R11G11B10F scene target with a D32F depth).
- Its own upscale network does the 536→1440 upscale:
  1. compute a335c1db6270c355 packs the scene into an E5B9G9R9 storage image;
  2. pass d99fbeeb53d38847 samples that image into the 2560x1440 R11G11B10F layer (nvn 0x3f);
  3. the final pass blends the UI over the layer into the RGBA8 output.

  This is the ootu32 / ootbilerp path in gfx_vk_tex.c and gfx_vk_exec.c.
- Frame generation: with `Renderer.frame_gen_dlss=true`, the presenter runs **optical-flow interpolation** (FSR3-style, fsr3_frame_gen).
  - It runs because the DLSS-G discovery never finds the game's inputs: "FG discovery: camera=0 motion_candidates=0".
  - It interpolates the composited picture, HUD included.
  - It is the prime suspect for the ghosting (잔상) NPARK saw.
  - It is gated by that setting alone (nvn_streamline.cpp `OpticalTarget`); NXN1X_DLSS_FG does not turn it off.

## 2. What we reuse

The reference is refer\NX2navi-main\runtime\vk_dlaa.c and shaders\dlaa_mv.comp (MIT). It is a static recompilation of this same game (Switch 2, 1.1.1) with DLAA/DLSS SR, FSR 3.1 and DLSS-G. Facts measured there that carry over, because they come from the game's data:

| Fact | NX2navi |
|---|---|
| Scene camera: the opaque scene draws bind a 0x3500-byte scene context at c3 (vertex stage). Rows 0-2 are the view (right, up, back), rows 7-10 the projection (row-major, GL clip depth). View-projection = P x V (`ocam_vp`). Full-screen passes bind a 0x200 block at c3 instead | `dlaa_draw_locked`, `ocam_vp` |
| Several scene contexts a frame: take the one most light-target draws use | `D.views[]` |
| Frame shape: D32F scene depth; a pass at the scene size with exactly one colour target (R11G11B10F / RGBA16F) and that depth arms the frame; bloom and composite write back into it; the game's upscaler then writes an output-sized R11G11B10F target with no depth; the HUD is drawn over that | `vk_dlaa_pass`, `vk_dlaa_upscaler` |
| DLSS runs where the game's upscaler runs: input = armed light target, output = the upscaler's R11G11B10F target, same shape (±3 %) and larger; the game's scaling draw is skipped; later draws larger than the scene sample DLSS's output | `vk_dlaa_upscaler`, `vk_dlaa_skip_draw`, `vk_dlaa_swap_img` |
| Jitter: Halton(2,3) as a viewport offset on every draw that binds the 0x3500 context into a D32F depth of the scene size, **whichever depth image**. The game shades against a depth copy with an EQUAL test, so jittering only one image turned the ground white. Full-screen passes are not jittered. 8·(out/in)² phases, 16 to 72. Jitter only while DLSS runs | `jitter_draw`, `next_jitter` |
| Motion vectors: reproject each pixel's depth through this and last frame's view-projection (camera motion). Pixels whose depth the camera does not explain (Link moving with the camera) get zero motion and a bias mask (`zfit`). RG16F in render pixels, jitter removed | `mv_pass`, dlaa_mv.comp |
| Quality from the ratio: ≥0.66 Quality, ≥0.57 Balanced, ≥0.49 Performance, else Ultra Performance; one feature per (in, out) size, at most 5 kept | `perf_quality`, `size_for` |
| History reset on a camera cut, a scene-size change or another target | `cam_cut`, `target_changed` |

From our side:
- `nvn_sr_evaluate` (nvn_streamline.cpp): Streamline DLSS SR. On the N1X it was checked with an offscreen readback (docs\DLSS_ARM64_CAPABILITIES.md).
- The DLSS-G path and its tagging (`nvn_fg_tag`).
- `set_dynamic` (gfx_vk_exec.c) is where the viewport is set: the place for the jitter.
- Draws see the bound uniform buffers in `S.ubo[stage][index]`.

NX2navi reads the camera from a submit-time uniform snapshot (vk_usnap). We read the guest memory of the bound c3 block on the replay thread at the draw.

## 3. Steps

Each step is a build that NPARK runs. Each new behaviour has an off switch.

### D1a — survey (log only, no picture change). DM_VK_OOTDLSS_SURVEY=1

Goal: check that our replay sees the frame as NX2navi describes it, before anything changes. Every 300 presents it logs:
- **Scene depth draws:** D32F depth ≥ 640 wide; how many bind a 0x3500 block at vertex c3; distinct addresses a frame; how many are not into a scene colour target.
- **Arming:** the light target found (size, nvn format); passes a frame drawn into it.
- **Upscale pass:** the first pass after arming with one colour target, no depth, R11G11B10F, larger and the same shape. Logs its size and the dispatch and draw shader keys between arming and it; a335c1db and d99fbeeb are expected.
- **Camera:** rows 0-2 orthonormal (error); projection rows 7-10 (fov, aspect against the scene, near/far); this frame's view-projection against last frame's (whether it changed).

### D1b — DLSS SR without jitter. DM_VK_OOT_DLSS=1 to try, default off until checked

At the upscale pass:
1. Run the MV compute (dlaa_mv.comp ported, camera-only plus zfit) on the scene depth.
2. Run `nvn_sr_evaluate` with color = light target, depth = scene depth, mv, output = a work RGBA16F image at the layer's size.
3. Blit the result into the layer.
4. Skip the network's writes into the layer that frame (d99fbeeb's draw, and a335c1db's dispatch when its only reader is skipped).

Expected: a temporal-AA'd, DLSS-scaled picture, softer than with jitter but stable. Check: no shaking, no black frames, HUD sharp, DRS steps (history reset).

### D1c — jitter, LOD bias, history rules

- Viewport jitter in `set_dynamic` for draws with c3 = 0x3500 into a scene-size D32F depth (any image).
- DLSS gets the jitter; texture LOD bias log2(in/out) − 1.
- Camera-cut and size-change resets.

After NPARK's A/B, DLSS SR becomes the default for OoT with DM_VK_OOT_DLSS=0 as the off switch (verified-fix rule).

### D2 — DLSS-G from D1's inputs

- Tag D1's depth, MV and camera, and the HUD-less picture (DLSS output before the UI), for `nvn_fg_tag` and the existing Streamline DLSS-G present.
- Where no inputs exist (menus, movies), generate nothing instead of falling back to optical flow for this title.
- Check against the optical-flow ghosting A/B (D0).

## 4. Open points

- **Parallel recording (S1e):** jittered draws can be recorded on worker threads. The jitter value for a frame must be fixed before the frame's segments are handed out; NX2navi uses `vk_exec_hook_turn` for the same reason. The DLSS evaluation and the MV dispatch are serial ops (like OP_DLSS: `seg_serial`).
- **Occlusion s3:** the scene draws' queries are unaffected; DLSS runs after the occlusion-tested draws.
- **Reading the camera:** the c3 block lives in guest memory. Is it still this frame's at replay time, with async replay one present behind? The D1a survey compares the view-projection the draws read with the one at evaluation.
- **Cost:** the network costs about 3.3 ms (dispatch 4f163e0d 80x180, oot_upsprobe). DLSS SR at 536→1440 on the N1X is to be measured.
