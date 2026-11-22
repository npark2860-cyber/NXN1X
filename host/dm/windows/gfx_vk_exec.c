// Android port, Vulkan backend: the command replay (mtl_exec, see gfx_vk.h and nvn_obj.h's record format).
// State records update a shadow of the NVN state; a draw or dispatch turns it into a pipeline (cached by everything that shapes it), descriptor sets (one per
// stage, filled from the shader's reflection: uniform slots, the shader's c1, the driver's c0 / c2 segments, textures and samplers by handle, images, storage
// buffers) and the draw. Render passes open lazily at the first draw or clear after SetRenderTargets and close before anything that is not drawing.
// Version 1 orders everything with whole-pipeline barriers around passes, copies and dispatches.
#include "gfx_vk.h"
#if defined(DM_NXN1X)
#include "nvn_fg_input.h"
#endif
#include "nvn_hmap.h"
#include "../gfx/tex/nvn_deswizzle.h"
#include <math.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>
int dm_nvn_jump_ok(const void *t);   // hle_nvn.c (jumpchk): a JUMP record's target is a record list

VkPipelineLayout vk_pipeline_layout(VkDescriptorSetLayout sets[5], int nsets);
void vk_shader_c1(vkshader_t *s, VkBuffer *buf, VkDeviceSize *off, VkDeviceSize *len);
VkBufferView vk_tbuf_view(const tex_t *t);
void vk_sub_update(uint64_t dst, const void *data, uint64_t n);
int prog_key(const prog_t *p, unsigned st, uint8_t key[8]);

// ---------------- NVN state shadow ----------------
typedef struct {   // R1 (parallel recording): S's type (S itself is in struct rec_ctx below)
    const prog_t *prog; vkshader_t *sh[6]; int prog_ok;
    p_bind_buf ubo[6][16], ssbo[6][16];
    uint64_t tex[6][256], samp[6][128], img[6][32];
    blend_st blend[8]; color_st color; chmask_st chmask; ms_st ms; poly_st poly; ds_st ds;
    uint32_t vattr[16]; vs_st vstream[16]; NVNbufferRange vbuf[16];
    float vp[16][4]; float dnear, dfar; p_scissor sc; int sc_set; uint8_t vp_sw[16][4];   // vp_sw: viewport swizzles XOR the identity (p_vp_swizzle)
    int sref[2], smask[2], swmask[2];
    p_poly_offset pofs; float blendc[4], alpha_ref; int depth_clamp, rast_discard, prim_restart;
    int patch_n, db_on; float db_lo, db_hi;   // r69: SetPatchSize, SetDepthBounds
    int tex_base, samp_base;   // r90: where the bound texture / sampler pool's IDs start in the host table (OP_SET_TEX_POOL / OP_SET_SAMP_POOL), for handles read from uniform buffers
    uint8_t push[2][DMSH_PUSH_SIZE];   // NVN2 push constants (OP_PUSH_CONST): the graphics and the compute block, read by the shaders as c[DMSH_PUSH_BANK]
    tex_t rt_col[8], rt_dep; view_t rt_cview[8], rt_dview; uint8_t rt_has_col[8], rt_has_cview[8], rt_has_dep, rt_has_dview; int rt_n; uint32_t rt_bad[8];   // rt_bad: the first word of a colour slot that is not a texture object (traces)
    int64_t row_stride, image_stride;
    // the open render pass
    int pass_open; uint32_t pass_w, pass_h; VkFormat cfmt[8], dfmt; VkImageAspectFlags daspect; uint32_t samples; int ncol;
    VkPipeline bound_pipe; int rp_pass;   // rp_pass: the open pass began with vkCmdBeginRenderPass
} rec_state_t;
// R1 (parallel recording): the recording state a draw reads and writes, grouped so that a worker thread can later own its own copy. One static instance
// (the replay thread's); every thread reaches it through g_rec, which defaults to it, so with one recording thread nothing changes. The old names stay
// as macros. Moved here: S, the texture view cache g_tvc, the descriptor set and driver segment caches g_setc / g_drvc, the bind cache g_bc and the
// s45 generations g_s45_kgen / g_s45_list (the Eden-side caches are struct rec_eden, gfx_vk_eden.inc)
enum { TVC = 4096 };   // (was before g_tvc)
enum { SETC = 8192, DRVC = 1024 };   // (was before g_setc)
typedef struct { uint64_t gen; VkViewport vp; VkRect2D sc; float bc[4]; int sm[2], sw[2], sr[2], line; p_poly_offset po; VkPipelineLayout lay; uint32_t nsets, ndofs;
             VkDescriptorSet sets[5]; uint32_t dofs[3 * 96]; VkBuffer vb[17]; VkDeviceSize vo[17]; VkBuffer ib; VkDeviceSize io; VkIndexType it; } rec_bc_t;   // R1b (parallel recording): g_bc's type (was anonymous in struct rec_ctx)
// S1c (parallel recording): the replay counters a recording bumps, per recording context. The replay thread's context counts into the globals the logs read
// (g_vkst, extern in gfx_vk_core.c / gfx_vk_present.c, g_set_hits / g_set_new, g_r131_all); a worker's context into its own copies, which vk_rec_ctx_fold adds into
// the replay thread's once its chunks are joined (gfx_vk_core.c vk_chunk_wait_part). The old names are macros for the thread's (after each definition below)
typedef struct { uint64_t draws, skip_prog, skip_rt, skip_pipe, passes, folded, dispatches, pipes_built, rt_kept, ns_exec, ns_desc, ns_pipe, max_pass_draws; } vkst_t;   // (was anonymous at g_vkst)
typedef struct { uint64_t idx, dir, ind, quad, gs, tes, rd, inst, stor, isum, ib[5], pipe, dset, dsets, dset_ofs, vb, ibind, vp, sc, dup, dupx; } r131_t;   // (was anonymous at g_r131)
extern vkst_t g_vkst; extern uint64_t g_set_hits, g_set_new; static r131_t g_r131_all;
typedef struct { int pw, ph, vw, vh; uint64_t n; } r80_vp_t;   // S1c: r80's viewports drawn into each target size (were g_r80_vp / g_r80_nvp / g_r80_vp_lost, set_dynamic: per draw)
// S1c (parallel recording): the pass and barrier counters (s64a, s64, s65, cbar; were statics where named below), counted at every pass end and OP_BARRIER
typedef struct { uint64_t s64_end[64], s64_passes, s64_bar, s64_bar_open, s64_bit[32], s64_dpp[7], s64_ab_n, s64_ab_fb, s64_npend, s64_kept, s64_fbend, s64_opend,
                 s65_rdp, s65_rddraws, s65_rdsplit, s65_cut, repeated_barriers, cbar_skip, cbar_kept, cbk[2][5]; } rec_pcnt_t;
static rec_pcnt_t g_pcnt_all;   // the replay thread's (the s61_log / cbar lines read it)
// S1c (parallel recording): the cross-record state of a recording that is not NVN state (S): what a draw or pass carries to the next one. Grouped so that a worker owns
// one (its context's) and a segment's snapshot can carry the fields that are order-dependent. Per field: [S] the stream's value at a segment's start must come from the
// dry pass (snapshot), [O] the recording thread's own (a segment starts with no pass open, so it starts from its context's value and the stream rules make that exact).
// The replay thread's context holds the one instance serial mode uses; the old names are macros for the thread's (same behaviour with one thread)
typedef struct {
    uint64_t mdi_cnt, mdi_stride; int mdi_n;   // [O] the indirect-count multi-draw run being drawn (mdi_scan, OP_MULTI_DRAW): set by a run's first record, read by its followers (never a segment's first)
    int s64_pend, s64_pwrite;                  // [O] s64: a barrier pending in the open pass, a storage-writing draw in it (cleared at every pass end; no pass is open at a segment's start)
    int rd_prev, rd_nwa; uint64_t rd_wa[16];   // [S] the last drawn draw had rasterizer discard (prepare_draw; the dry pass keeps such draws and the one after them serial, so 0 at a
                                               // segment's start); [O] the run's storage buffer addresses read back at its end (rd_read; was prepare_draw's static wa / nwa)
    int deq_dir;                               // [S] r57: the direction of the last depth-writing test (prepare_draw, eden_draw; read only with DM_VK_DEPTH_EQ > 0, which keeps workers off)
    p_tf_bind tf[4]; uint64_t tf_off[4]; int tf_on, tf_in_pass; uint32_t tf_verts;   // transform feedback: [S] tf_on (the dry pass's g_sd.tf_on; a key input of every draw);
                                               // [O] the buffers, bytes written, a capture in the open pass, vertices since Begin (only capturing draws use them: serial)
    int rc_skip;                               // [S] render condition false: the draws and clears after it are dropped (render_cond; to come from the dry pass's condition results)
    int s64_afterbar, s64_fb, s65_rdpass;      // [O] s64a: the pass after a barrier-ended one, its attachments sampled; s65: the open pass holds rasterizer-discard draws only
    uint32_t lop_kept; int32_t df_kept;        // [S] B7 (eden_fill_state, were its statics lop_kept / df_kept): the Maxwell logic op register keeps the last op turned on, the
                                               // zeta register the last depth target's format; both follow the stream of key builds
    int rcg_on, rcg_inv; uint32_t rcg_off;     // [S] occ-s3: the draws and clears after a GPU render condition are gated by its predicate (byte rcg_off of the resolve buffer, inverted
                                               // for mode 0); segments under one stay on the replay thread (WR_GPUCOND), so a worker starts with it off
    VkCommandBuffer rcg_cb;                    // [O] occ-s3: the command buffer a conditional rendering block is open in (inside its pass; occ-s3fast: kept open over the gated draws and
                                               // clears that follow with the same predicate, rcg_boff / rcg_binv, until a pass end, a new condition or a condition that gates nothing)
    int rcg_binv; uint32_t rcg_boff;           // [O] occ-s3fast: the open block's predicate (byte offset, inverted)
} rec_xs_t;
struct rec_ctx {
    rec_state_t S;
    struct { uint64_t epoch, gen; int tid, dim, arr, storage, num, want; VkImageView v, scaled_view; vkimg_t *im; } g_tvc[TVC];
    struct { uint64_t h, epoch; VkDescriptorSetLayout l; VkDescriptorSet set; } g_setc[SETC];
    struct { uint64_t h, epoch; VkBuffer b; VkDeviceSize o; } g_drvc[DRVC];
    rec_bc_t g_bc;   // R1b (parallel recording): its type named (rec_bc_t, above) for the memos below
    uint64_t g_s45_kgen, g_s45_list;
    // R1b (parallel recording): the open pass (begin_pass, pass_commit, vk_end_pass; were statics near them): its attachment infos, rendering info (which points
    // at P_ca / P_da / P_sa of the same context), whether a clear went into the load ops, the attachments found, the draws in it, layered or not, the scissors' box
    VkRenderingAttachmentInfo P_ca[8], P_da, P_sa; VkRenderingInfo P_ri; int P_cleared;
    struct { vkimg_t *im; VkImageView v; uint32_t fmt, layers, w, h, image_w, image_h; } P_att[9];
    uint64_t g_pass_draws; int g_pass_layered; int32_t g_pass_bb[4];
    // R1b (parallel recording): the draw-run memos keyed by the command buffer and g_cmd_gen (were statics before r142_merge, r152_note, r151_draw)
    struct { VkCommandBuffer cb; uint64_t gen; int pass, di; uint32_t n; VkPipeline pipe; VkDescriptorSet sets[5]; VkBuffer vb[17]; VkDeviceSize vo[17]; VkBuffer ib; VkDeviceSize io; VkIndexType it;
             int32_t count, bv; uint32_t bi; VkDrawIndexedIndirectCommand *a; } g_mg;   // r142: the run being merged
    struct { VkCommandBuffer cb; uint64_t gen, key; int pass, di; VkPipeline pipe; int32_t count, bv; uint8_t bc[sizeof(rec_bc_t)]; } g_an;   // r152: the previous draw auto looked at
    struct { VkCommandBuffer cb; uint64_t gen; int pass, di; uint32_t n; VkPipeline pipe; int32_t count, bv; uint8_t *st; VkDrawIndexedIndirectCommand *a; uint64_t key; uint8_t bc[sizeof(rec_bc_t)]; } g_ir;   // r151: the run being drawn
    rec_xs_t X;   // S1c (parallel recording): the cross-record state (rec_xs_t above)
    vkst_t *vkst; r131_t *r131; uint64_t *set_hits, *set_new;   // S1c (parallel recording): where this context counts (the globals for the replay thread's, own_* for a worker's)
    vkst_t own_vkst; r131_t own_r131; uint64_t own_set[2];
    r80_vp_t r80_vp[16]; int r80_nvp; uint64_t r80_vp_lost;   // S1c (parallel recording): r80's viewport table, per thread (vk_rec_ctx_fold merges a worker's into the replay thread's, which vk_r80_log reads)
    rec_pcnt_t *pc; rec_pcnt_t own_pc;   // S1c (parallel recording): where its pass and barrier counters go (g_pcnt_all for the replay thread's)
    // S1e (parallel recording): the memo generations of a stage's bindings (were statics before stage_set: g_bind_gen / g_ubo_gen, bumped by this context's state records and
    // compared by its own memos only; a worker's bumps raced the replay thread's, and a lost increment could make a memo match changed bindings), and the open pass's scale
    // state (were statics before materialize_scale: pass_scale, scale_writes, scale_dirty; set at each pass_commit and read in that pass: per recording context)
    uint64_t bind_gen[6], ubo_gen[6];
    int pscale; VkImageAspectFlags sc_writes[9]; struct { int left, top, right, bottom; } sc_dirty;   // (pass_scale, scale_writes, scale_dirty: macros below)
};
static struct rec_ctx g_rec_main = {.vkst = &g_vkst, .r131 = &g_r131_all, .set_hits = &g_set_hits, .set_new = &g_set_new, .pc = &g_pcnt_all, .pscale = 1};   // R1 (parallel recording): the replay thread's (S1c: X zero, as its statics were)
static _Thread_local struct rec_ctx *g_rec = &g_rec_main;   // R1 (parallel recording): this thread's recording state
static inline struct rec_ctx *rec_cur(void) { return g_rec; }   // R1: for a function that keeps g_rec in a local of the same name
#define S (g_rec->S)
#define g_tvc (g_rec->g_tvc)
#define g_setc (g_rec->g_setc)
#define g_drvc (g_rec->g_drvc)
#define g_bc (g_rec->g_bc)
#define g_s45_kgen (g_rec->g_s45_kgen)
#define g_s45_list (g_rec->g_s45_list)
#define P_ca (g_rec->P_ca)   // R1b (parallel recording): the open pass and the run memos, in struct rec_ctx
#define P_da (g_rec->P_da)
#define P_sa (g_rec->P_sa)
#define P_ri (g_rec->P_ri)
#define P_cleared (g_rec->P_cleared)
#define P_att (g_rec->P_att)
#define g_pass_draws (g_rec->g_pass_draws)
#define g_pass_layered (g_rec->g_pass_layered)
#define g_pass_bb (g_rec->g_pass_bb)
#define g_mg (g_rec->g_mg)
#define g_an (g_rec->g_an)
#define g_ir (g_rec->g_ir)
#define g_bind_gen (g_rec->bind_gen)   // S1e (parallel recording): per recording context (struct rec_ctx)
#define g_ubo_gen (g_rec->ubo_gen)
#define pass_scale (g_rec->pscale)
#define scale_writes (g_rec->sc_writes)
#define scale_dirty (g_rec->sc_dirty)
// transform feedback: buffers, bytes written into each, capture on, a capture in the open pass (xfb_begin), vertices captured since Begin (S1c (parallel recording): in rec_xs_t, were statics here)
#define g_tf (g_rec->X.tf)
#define g_tf_off (g_rec->X.tf_off)
#define g_tf_on (g_rec->X.tf_on)
#define g_tf_in_pass (g_rec->X.tf_in_pass)
#define g_tf_verts (g_rec->X.tf_verts)
#define g_rc_skip (g_rec->X.rc_skip)   // S1c (parallel recording): render_cond's result (was a static below)
#define g_s64_pend (g_rec->X.s64_pend)   // S1c: s64's pass state (were statics with g_s64_on)
#define g_s64_pwrite (g_rec->X.s64_pwrite)
#define g_deq_dir (g_rec->X.deq_dir)     // S1c: r57's direction (was a static with g_deq)
#define g_rd_prev (g_rec->X.rd_prev)     // S1c: prepare_draw's rasterizer-discard run (was a static before prepare_draw)
#define g_mdi_cnt (g_rec->X.mdi_cnt)     // S1c: the indirect-count run (were statics before mdi_scan)
#define g_mdi_n (g_rec->X.mdi_n)
#define g_mdi_stride (g_rec->X.mdi_stride)
static int g_depth_zero = 1, g_lower_left;
void mtl_set_depth_zero(int on) { g_depth_zero = on; }
void mtl_set_origin_lower_left(int on) { g_lower_left = on; }
static int g_queue_ll;   // the queue's SET_WINDOW_ORIGIN raster flip (hle_nvn.c, per submission): it reverses the winding only, as nvn_mtl.m front_ccw
void mtl_set_queue_origin_lower_left(int on) { g_queue_ll = on != 0; }
// NVN's upper-left mode has clip space y up (as Metal and D3D): it is drawn with a flipped viewport over the same top-left rectangle, lower-left mode without
// one; the scissor is then never flipped and the winding follows nvn_mtl.m's. Drawn the other way, every blit flipped the picture (NEO docked: upside down)
// and the winding had to be inverted. DM_VK_FLIPY=0 brings the old way back
#include <dlfcn.h>
static int g_flipy;
// Lower-left mode then stores the window image bottom-up (GL's rows), so the present flips it as nvn_mtl.m does (Celeste's MonoGame keeps NVN's default lower-left origin)
int vk_window_bottom_up(void) { return g_lower_left && g_flipy; }
static void state_reset(void) {
    memset(&S, 0, sizeof S);
    S.dnear = 0; S.dfar = 1;
    S.chmask.mask = 0xffffffffu;
    S.color.logic_op = 3;
    S.poly.front = 1; S.poly.mode = 2;
    S.ds.dw = 1; S.ds.df = 2; S.ds.sf_f = S.ds.sf_b = 8; S.ds.fail_f = S.ds.zfail_f = S.ds.zpass_f = S.ds.fail_b = S.ds.zfail_b = S.ds.zpass_b = 1;
    S.smask[0] = S.smask[1] = S.swmask[0] = S.swmask[1] = 0xff;
    for (int i = 0; i < 8; i++) { S.blend[i].target = i; S.blend[i].src_rgb = S.blend[i].src_a = 2; S.blend[i].dst_rgb = S.blend[i].dst_a = 1; S.blend[i].eq_rgb = S.blend[i].eq_a = 1; }
}
// r72 (Switch 1 thread's audit, 11-07): the replay's NVN state per queue, as nvn_mtl.m mtl_queue_activate. mtl_queue_activate / mtl_queue_forget were empty here and S
// carried one queue's bindings into the next queue's submission (on the Mac, Fire Emblem: Three Houses and FFX-2 draw with two queues). hle_nvn.c calls both under the
// submission lock, before the submission's replay (vk_exec_reset then ends the open pass of the restored state). Transform feedback bindings go with it
// S1c (parallel recording): g_rc_skip (render condition false) in rec_xs_t (a macro above)
typedef struct vq_state { int rc_skip, rcg_on, rcg_inv; uint32_t rcg_off; uint64_t serial; typeof(S) s; p_tf_bind tf[4]; uint64_t tf_off[4]; int tf_on; struct vq_state *next; } vq_state;   // (occ-s3: the GPU condition with it)
static vq_state *g_vq, *g_vq_cur;
void mtl_queue_activate(uint64_t serial) {
    if (g_vq_cur && g_vq_cur->serial == serial) return;
    if (g_vq_cur) { g_vq_cur->rc_skip = g_rc_skip; g_vq_cur->rcg_on = g_rec->X.rcg_on; g_vq_cur->rcg_inv = g_rec->X.rcg_inv; g_vq_cur->rcg_off = g_rec->X.rcg_off; g_vq_cur->s = S; memcpy(g_vq_cur->tf, g_tf, sizeof g_tf); memcpy(g_vq_cur->tf_off, g_tf_off, sizeof g_tf_off); g_vq_cur->tf_on = g_tf_on; }
    vq_state *q = g_vq; while (q && q->serial != serial) q = q->next;
    if (q) { S = q->s; memcpy(g_tf, q->tf, sizeof g_tf); memcpy(g_tf_off, q->tf_off, sizeof g_tf_off); g_tf_on = q->tf_on; }
    else {
        q = calloc(1, sizeof *q); if (!q) { VKRED_ONCE(110, "queue state allocation failed: the queues share one state"); return; }
        q->serial = serial; q->next = g_vq;
        if (g_vq) { state_reset(); memset(g_tf, 0, sizeof g_tf); memset(g_tf_off, 0, sizeof g_tf_off); g_tf_on = 0; }   // (the first queue keeps the state as it is)
        g_vq = q;
        { static int n; if (++n == 2) VKLOG("r72: a second NVN queue submits: each queue keeps its own bindings and state"); }
    }
    S.pass_open = 0; S.bound_pipe = VK_NULL_HANDLE;
    g_rc_skip = q->rc_skip; g_rec->X.rcg_on = q->rcg_on; g_rec->X.rcg_inv = q->rcg_inv; g_rec->X.rcg_off = q->rcg_off;
    g_vq_cur = q;
}
void mtl_queue_forget(uint64_t serial) {
    vq_state **pp = &g_vq; while (*pp && (*pp)->serial != serial) pp = &(*pp)->next;
    if (!*pp) return;
    vq_state *q = *pp; *pp = q->next;
    if (g_vq_cur == q) { g_rc_skip = 0; g_rec->X.rcg_on = 0; g_vq_cur = NULL; state_reset(); memset(g_tf, 0, sizeof g_tf); memset(g_tf_off, 0, sizeof g_tf_off); g_tf_on = 0; }
    free(q);
}
// progforget (nxnative 1440a73, Fire Emblem Fortune's Weave: keys_hash crash at the title's hand-over to the opening demo): NVN reads a program only when it is
// bound, so a game may finalize a program a queue still has bound and reuse its memory. The pointer stayed in S and in the saved queue states (r72), and the next
// prog_key on it (pipeline keys, draw records) hashed the reused memory. hle_nvn.c's nvnProgramFinalize calls this under the submission lock; it was empty here.
// The host shaders already taken (sh[]) are host objects and stay until the next bind. DM_VK_PROG_FORGET=0: as before
void mtl_prog_forget(const prog_t *p) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_PROG_FORGET"); on = !(e && *e == '0'); }
    if (!on || !p) return;
    int active = 0, saved = 0;
    if (S.prog == p) { S.prog = NULL; active = 1; }
    for (vq_state *q = g_vq; q; q = q->next) if (q != g_vq_cur && q->s.prog == p) { q->s.prog = NULL; saved++; }
    if (active || saved) { static unsigned n; if (!n++ || !(n & (n - 1))) VKLOG("progforget: finalized program %p dropped from queue bindings (active queue %d, saved queues %d; %u times so far)", (const void *)p, active, saved, n); }
}
static void prewarm_start(void);
static uint64_t g_set_epoch = 1;   // the submission the descriptor set cache belongs to (stage_set)
// R1b (parallel recording): g_pass_bb (bounding box of the scissors the open pass's draws used, x0 y0 x1 y1) in struct rec_ctx
static int g_cb_draws;   // draws in the command buffer being recorded (vk_cmd_split after DM_VK_CB_DRAWS draws, 0 = never)
// s61 (DM_VK_S61_MAXCUT, live): the cap on ordinary cuts (was 15); counters of begin_pass's cuts, logged every 2 s from vk_exec_reset (s61_log)
static int g_s61_maxcut = 15; static uint64_t g_s61_subs, g_s61_cuts, g_s61_cutdraws, g_s61_cap, g_s61_t0;
// s64a (apply_s64a.py): pass ends by op, OP_BARRIER flags, draws per pass, attachment sampling in the pass after a barrier-ended one (logged in s61_log)
#define g_s64_end (g_rec->pc->s64_end)   // S1c (parallel recording): the thread's pass and barrier counters (rec_pcnt_t) and pass state (rec_xs_t); were statics here
#define g_s64_passes (g_rec->pc->s64_passes)
#define g_s64_bar (g_rec->pc->s64_bar)
#define g_s64_bar_open (g_rec->pc->s64_bar_open)
#define g_s64_bit (g_rec->pc->s64_bit)
#define g_s64_dpp (g_rec->pc->s64_dpp)
#define g_s64_ab_n (g_rec->pc->s64_ab_n)
#define g_s64_ab_fb (g_rec->pc->s64_ab_fb)
#define g_s64_afterbar (g_rec->X.s64_afterbar)
#define g_s64_fb (g_rec->X.s64_fb)
extern const char *const g_op_names[64];
void vk_og_op(unsigned op);   // gfx_vk_core.c ootgap
// s64 (apply_s64.py): an OP_BARRIER with a pass open waits (pending) for the next draw, which keeps the pass unless it reads one of its attachments
static int g_s65_on;   // s65 (apply_s65.py): rd-only passes, passes ended by an rd toggle, areas cut (S1c (parallel recording): g_s65_rdpass in rec_xs_t, the counters in rec_pcnt_t)
#define g_s65_rdpass (g_rec->X.s65_rdpass)
#define g_s65_rdp (g_rec->pc->s65_rdp)
#define g_s65_rddraws (g_rec->pc->s65_rddraws)
#define g_s65_rdsplit (g_rec->pc->s65_rdsplit)
#define g_s65_cut (g_rec->pc->s65_cut)
static int g_s64_on;   /* S1c (parallel recording): its counters in rec_pcnt_t */
#define g_s64_npend (g_rec->pc->s64_npend)
#define g_s64_kept (g_rec->pc->s64_kept)
#define g_s64_fbend (g_rec->pc->s64_fbend)
#define g_s64_opend (g_rec->pc->s64_opend)
#define g_repeated_barriers (g_rec->pc->repeated_barriers)   // S1c (parallel recording): rec_pcnt_t (was a static)
static int s64_draw_feedback(int prim, int sampled_only);   // gfx_vk_eden.inc
extern uint64_t g_s61_sent, g_s61_sub_ns;   // gfx_vk_core.c vk_cmd_split: parts submitted early, time in their vkQueueSubmit
static int g_psplit, g_ord_parts, g_setsplit;   // DM_VK_PASS_SPLIT (prepare_draw); g_psplit_now: the pass being begun goes on with the one just ended (begin_pass cuts before it);
                                                                    // g_ord_parts: ordinary DM_VK_CB_DRAWS cuts in this submission, kept to the old 15 (MAIN_CBS was 16): the extra parts are for pass splits
static _Thread_local int g_psplit_now;   // S1e (parallel recording): set and taken within one draw by the thread recording it (was with g_psplit)
// S1d (parallel recording): set while the list being replayed takes its part cuts from the dry pass (DM_VK_PARALLEL_DRY / SERIALSEG; seg_cut_take): begin_pass's DM_VK_CB_DRAWS
// cut and prepare_draw's DM_VK_PASS_SPLIT pass end are then not made (the dry pass placed them so that no segment spans a cut). S1e: per thread (a worker's is set for its segment)
static _Thread_local int g_seg_cuts;
// S1e (parallel recording): this thread records a worker segment (gfx_vk_exec.c seg_job_main; the helper context on the replay thread too): the order-dependent bookkeeping
// that is the replay thread's (g_occ_draws, g_cb_draws, g_sub_pass, the render-condition census, ootbilerp's kept targets, r150's fresh copies, the attachment marks) is skipped
// here and done by the replay thread at the hand-out; g_wk_abort: the segment cannot be recorded here (it is recorded again on the replay thread, seg_par_join)
static _Thread_local int g_wk, g_wk_abort, g_wk_rerun;
// S1e: why a worker could not record its segment (g_wk_abort; counted, the segment then recorded on the replay thread): an image lookup vk_tex refused (its g_vk_tex_refused:
// a new image, an archived one, a storage upgrade, sparse tiles), a pass on targets the hand-out did not resolve, a decision or deferred-handle read off the dry pass's, a record
// the hand-out rules keep off the workers (should not happen), a clear of a whole target larger than the pass (r76: a transfer clear and its marks), the walk stopping at
// another record than the dry pass's end, the chunk's begin or end failing ([RED])
enum { WA_NONE, WA_IMG_NEW, WA_IMG_ARCH, WA_IMG_UPG, WA_IMG_S2T, WA_TARGET, WA_DEC, WA_OP, WA_CLEAR, WA_STOP, WA_BEGIN, WA_END, WA_TEST, WA_N };   // parck: WA_TEST
static int parck_test(void);   // parck (below seg_job_main)
static inline void seg_wk_abort(int why) { if (!g_wk_abort) g_wk_abort = why; }
static _Thread_local int g_var_now;   /* S1c (parallel recording): per-draw scratch, per thread */   // the bisection's change for the draw being recorded (see g_bis_var)
static _Thread_local int g_tracing;   /* S1c (parallel recording): per thread */   // the draw being recorded is the one the bisection found (its descriptors are logged)
// replay counters, printed by gfx_vk_present.c every 600 presents (draws done and skipped by reason, passes, clears folded into load ops, dispatches)
vkst_t g_vkst;   /* R1b (parallel recording): g_pass_draws in struct rec_ctx */   // ns_*: time in the replay, its descriptor sets, its pipeline keys and lookups
#define g_vkst (*g_rec->vkst)   // S1c (parallel recording): the thread's counters (struct rec_ctx; the replay thread's are the global above)
static inline uint64_t tnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
// r80 (diagnostic): pipelines made where a draw or dispatch waits for them (the replay thread), for the long-frame line (hle_nvn.c reads mtl_create_stats /
// mtl_create_ns; they were 0 stubs in gfx_vk_win_glue.c); one taking 100 ms or more is logged with its shaders. Also: the prewarm's end, and the viewports drawn into
// each target size (to see a dynamic resolution that draws part of a fixed target)
static uint64_t g_r80_pso, g_r80_pso_ns, g_r80_pw_have, g_r80_pw_fail, g_r80_pw_t0; static int g_r80_pw_live;
void mtl_create_stats(uint64_t *l, uint64_t *p) { *l = 0; *p = __atomic_load_n(&g_r80_pso, __ATOMIC_RELAXED); }
void mtl_create_ns(uint64_t *l, uint64_t *p) { *l = 0; *p = __atomic_load_n(&g_r80_pso_ns, __ATOMIC_RELAXED); }
struct pkey; struct vkshader; static void r80_made(uint64_t t0, const struct pkey *k, const struct vkshader *cs);
#define g_r80_vp (g_rec->r80_vp)   // S1c (parallel recording): the thread's (struct rec_ctx; were statics here, "replay thread")
#define g_r80_nvp (g_rec->r80_nvp)
#define g_r80_vp_lost (g_rec->r80_vp_lost)
static inline void r80_vp(int pw, int ph, float w, float h) {
    if (pw < 960) return; int vw = (int)(w + 0.5f), vh = (int)(h + 0.5f);
    for (int i = 0; i < g_r80_nvp; i++) if (g_r80_vp[i].pw == pw && g_r80_vp[i].ph == ph && g_r80_vp[i].vw == vw && g_r80_vp[i].vh == vh) { g_r80_vp[i].n++; return; }
    if (g_r80_nvp < 16) g_r80_vp[g_r80_nvp++] = (typeof(g_r80_vp[0])){pw, ph, vw, vh, 1}; else g_r80_vp_lost++;
}
// r131 (diagnostic): what a frame's draws are and how much state is set between them, counted (no per-draw log lines: DM_VK_TRACE_FRAME on a 3,400 draw submission
// froze the game for 20 s). Two live switches split the GPU's time per draw: DM_VK_X_SC1=1 gives every draw a one pixel scissor (what is left is state loading and
// vertex work), DM_VK_X_DUP=N (1..8) issues each plain draw N more times with no state set in between (what is added is the draw alone). Draws with a geometry or
// tessellation shader, rasterizer discard, transform feedback or a storage binding in the vertex or fragment stage are not issued again (they write memory), and
// indirect draws and quads are not either. DM_VK_X_CENSUS=1 prints the two lines without a switch
static int g_x_sc1, g_x_dup, g_x_census; static _Thread_local int g_r131_dupok;   // (S1c (parallel recording): the draw being recorded's, per thread)
// r135 (diagnostic): what a vertex / index buffer bind costs per draw. DM_VK_X_VB=1 (live): before each plain draw of the r131 DM_VK_X_DUP class, the first bound vertex
// buffer is bound again as it is (Turnip does not filter it: a new vertex buffer draw state group per draw). DM_VK_X_IB=1: the index buffer likewise before each such indexed
// draw (Turnip keeps its address for the draw packet, which carries it anyway). The picture stays the same
static int g_x_vb, g_x_ib; static struct { uint64_t vb, ib; } g_r135;
static void r135_force(VkCommandBuffer cb, int indexed);
// r142 (diagnostic): DM_VK_X_MERGE=<vertex shader key hex> (live). A plain indexed draw of that vertex shader (the r131 DM_VK_X_DUP class, one instance) that follows the
// previous one directly (same command buffer and pass) with the same pipeline, descriptor sets, vertex buffers, index buffer and draw parameters, only other dynamic offsets,
// is not drawn: the instance count of the first draw of the run (an indirect draw with its arguments in scratch memory) goes up by one. Every instance reads the first draw's
// offsets, so the picture is wrong on purpose: this measures what one draw per run would save
static uint64_t g_x_merge;
static struct { uint64_t runs, single, merged, brk_seq, brk_state, brk_mesh, notok; } g_r142;
// r151 (diagnostic): DM_VK_X_IRUN=<vertex shader key hex> (live). The r133 windows for that vertex shader and its fragment shader, with the in-window offsets of each draw
// given per instance (vertex binding 17, locations 28-31, uvec4; the vertex shader passes the fragment words on, flat, at locations 28-29) in place of the push constants.
// A plain indexed draw that follows the previous one directly with the same state (pipeline, sets and their window bases, buffers, dynamic state) and mesh becomes one more
// instance of that run's indirect draw, its offsets the next entry of the run's instance stream. The picture stays the same
static uint64_t g_x_irun; static _Thread_local int g_irun_on;   // (S1c (parallel recording): per thread)
// r152 (diagnostic): DM_VK_X_IRUN=<key>[,<key>...] (up to 16) or =auto (live). auto: a vertex shader is taken once, in one frame, DM_VK_X_IRUN_MIN (default 32) of
// its draws would have continued a run (r151_draw's test with the dynamic offsets left out, r152_note); it stays taken. Counters per vertex shader in the r152 lines
static uint64_t g_irun_keys[16]; static int g_nirun, g_irun_auto, g_irun_min = 32; static _Thread_local int g_r152_cand;   // (S1c (parallel recording): per thread)
typedef struct { uint64_t key, frame; uint32_t cnt; int qual; uint64_t draws, runs, brk_win, brk_state, p[4]; } r152_e;
static r152_e g_r152[256];
static r152_e *r152_get(uint64_t key, int add) { uint32_t h = (uint32_t)(key * 0x9E3779B97F4A7C15ull >> 56);
    for (int i = 0; i < 256; i++, h = (h + 1) & 255) { if (g_r152[h].key == key) return &g_r152[h]; if (!g_r152[h].key) { if (!add) return NULL; g_r152[h].key = key; return &g_r152[h]; } }
    return NULL; }
static int irun_want(uint64_t key) { for (int i = 0; i < g_nirun; i++) if (g_irun_keys[i] == key) return 1; if (!g_irun_auto) return 0; r152_e *e = r152_get(key, 0); return e && e->qual; }
static int g_r153_cap, g_r153_n; static struct { const void *c, *z; uint64_t vs, fs; } g_r153_seen[512];
static void r152_read(void) { const char *e = getenv("DM_VK_X_IRUN"); g_nirun = 0; g_irun_auto = 0;
    { const char *q = getenv("DM_VK_LOG_UPLOADS"); g_r153_cap = q && atoi(q) ? (atoi(q) > 1 ? atoi(q) : 60) : 0; }   // r153
    if (e && !strncmp(e, "auto", 4)) g_irun_auto = 1;
    else for (const char *q = e; q && *q && g_nirun < 16;) { char *x; uint64_t v = strtoull(q, &x, 16); if (x == q) break; if (v) g_irun_keys[g_nirun++] = v; if (*x != ',') break; q = x + 1; }
    g_x_irun = (uint64_t)(g_irun_auto || g_nirun); e = getenv("DM_VK_X_IRUN_MIN"); g_irun_min = e && atoi(e) > 0 ? atoi(e) : 32; }
static struct { uint64_t runs, draws, inst, brk_seq, brk_state, brk_mesh, brk_full, notok, novar; } g_r151;
#define IRUN_CAP 128
// r143 (diagnostic): DM_VK_X_VSTAB=1 (live): draws per frame by vertex + fragment shader pair, the 24 most drawn every 600 presents, with how many of them have a
// geometry shader or a storage binding: which groups to leave out with DM_VK_X_SKIPVS to measure their GPU time
static int g_x_vstab;
typedef struct { uint64_t vs, fs, n, gs, st; } r143_e;
enum { R143N = 2048 };
static r143_e g_r143[R143N]; static uint64_t g_r143_full;
static void r143_note(int st) {
    uint64_t vs = S.sh[0] ? S.sh[0]->key : 0, fs = S.sh[1] ? S.sh[1]->key : 0; uint32_t h = (uint32_t)((vs ^ (fs * 0x9e3779b97f4a7c15ull)) >> 20) & (R143N - 1);
    for (int i = 0; i < R143N; i++, h = (h + 1) & (R143N - 1)) { r143_e *e = &g_r143[h];
        if (!e->n) { e->vs = vs; e->fs = fs; }
        else if (e->vs != vs || e->fs != fs) continue;
        e->n++; e->gs += !!S.sh[2]; e->st += (uint64_t)!!st; return; }
    g_r143_full++;
}
static void r143_report(void) {
    uint64_t tot = 0; int used = 0; for (int i = 0; i < R143N; i++) if (g_r143[i].n) { tot += g_r143[i].n; used++; }
    VKLOG("r143: draws per frame (last 600 presents) by vertex + fragment shader: %.0f draws in %d pairs%s; the most drawn:", (double)tot / 600, used, g_r143_full ? " (table full: some not counted)" : "");
    static char done[R143N]; memset(done, 0, sizeof done);
    for (int k = 0; k < 24; k++) { int b = -1; for (int i = 0; i < R143N; i++) if (g_r143[i].n && !done[i] && (b < 0 || g_r143[i].n > g_r143[b].n)) b = i;
        if (b < 0) break; done[b] = 1; r143_e *e = &g_r143[b];
        VKLOG("r143:  vs %016llx fs %016llx draws %.1f (geometry shader %.1f, storage binding %.1f)", (unsigned long long)e->vs, (unsigned long long)e->fs, (double)e->n / 600, (double)e->gs / 600, (double)e->st / 600); }
    memset(g_r143, 0, sizeof g_r143); g_r143_full = 0;
}
#define g_r131 (*g_rec->r131)   // S1c (parallel recording): the thread's (r131_t; the replay thread's is g_r131_all)
static inline int r131_draw(int indexed, uint32_t count, uint32_t inst, int xf) {   // counts a plain draw; how many more times it is issued
    if (indexed) g_r131.idx++; else g_r131.dir++; g_r131.inst += inst > 1; g_r131.isum += count; g_r131.ib[count <= 32 ? 0 : count <= 128 ? 1 : count <= 512 ? 2 : count <= 2048 ? 3 : 4]++;
    if (!g_x_dup) return 0; if (!g_r131_dupok || xf) { g_r131.dupx++; return 0; } g_r131.dup += (uint64_t)g_x_dup; return g_x_dup;
}
// r132 (diagnostic): what another per-draw channel costs on the GPU, beside the dynamic uniform buffer offsets. DM_VK_X_PUSH=1 (live): pipelines whose fragment shader
// has a colour 0 (and no alpha test) get a variant that adds the first fragment push constant word, read as a float, to colour 0's alpha (gfx_vk_shader.c spv_push_read);
// every such draw sets that word (0, so the picture stays the same). DM_VK_X_INST=1 (live): plain draws with one instance and no base instance, whose vertex bindings are
// all per vertex and whose vertex shader reads no instance index, get a different first instance each (the driver segment's base instance stays 0)
static int g_x_push, g_x_inst; static _Thread_local int g_r132_instok;   // (S1c (parallel recording): per thread)
static struct { uint64_t push, pushx, inst, instx; } g_r132;
static int r132_vs_inst(const vkshader_t *s) {   // 1 when the vertex shader reads gl_InstanceIndex, gl_BaseInstance or InstanceId (or its code is not known)
    static struct { const vkshader_t *s; int r; } t[1024];
    uint32_t h = (uint32_t)(((uintptr_t)s >> 4) * 2654435761u) & 1023;
    for (int i = 0; i < 1024; i++, h = (h + 1) & 1023) { if (t[h].s == s) return t[h].r; if (!t[h].s) break; }
    int r = 1;
    if (s->spv && s->nspv > 5) { r = 0; for (uint32_t i = 5; i < s->nspv;) { uint32_t n = s->spv[i] >> 16, op = s->spv[i] & 0xffff; if (!n || i + n > s->nspv) { r = 1; break; }
        if (op == 71 && n >= 4 && s->spv[i + 2] == 11 && (s->spv[i + 3] == 43 || s->spv[i + 3] == 6 || s->spv[i + 3] == 4425)) { r = 1; break; }
        if (op == 54) break;
        i += n; } }
    if (!t[h].s) { t[h].s = s; t[h].r = r; }
    return r;
}
static inline uint32_t r132_binst(uint32_t bi, uint32_t inst) {   // the first instance a plain draw is issued with
    if (!g_x_inst) return bi; if (!g_r132_instok || inst != 1 || bi) { g_r132.instx++; return bi; } g_r132.inst++; return 1u + (uint32_t)(g_r132.inst & 0xfff);
}
// r133 (diagnostic): one shader's constant buffers read at a pushed offset instead of a new dynamic offset per draw. DM_VK_X_POFS=<vertex shader key, hex> (live; 0 off):
// pipelines with that vertex shader take variants of it and of their fragment shader (gfx_vk_shader.c spv_pofs) that add a push constant word to the vec4 index of every
// read of their game constant buffers (dynamic, the first 8 by binding). Such a buffer is bound with a 64 KB range from a window base, given as its dynamic offset, that
// moves only when a draw's block leaves the window; the draw pushes its offset in the window (in vec4s): vertex words 4-11, fragment words 12-19. The picture stays the same
VkDeviceSize vk_buf_size(VkBuffer b); VkPipelineLayout vk_pipeline_layout_pofs(VkDescriptorSetLayout sets[5], int nsets);
static uint64_t g_x_pofs; static _Thread_local int g_pofs_st[2]; static _Thread_local uint32_t g_pofs_words[20];   // (S1c (parallel recording): the draw being recorded's, per thread)
// r134 (diagnostic controls for r133): DM_VK_X_POFS_MODE=1: every converted draw reads the block of the first converted draw since its window was placed (the window
// moves only when the buffer or the set epoch changes; the picture of those objects is wrong, reads stay in the bound range). DM_VK_X_POFS_VS / DM_VK_X_POFS_FS=<hex
// mask of slots, the dynamic buffers in binding order>: only those slots are converted (0: all, as r133); the shader variant and pipeline differ per mask
static int g_pofs_mode; static uint32_t g_pofs_m[2];
#define POFS_TAG(st) (g_pofs_st[st] ? 1 + (int)g_pofs_m[st] * 2 : 0)
typedef struct { VkBuffer buf; uint64_t N, R, Sz; int conv; } pofs_slot_t;   // R1 (parallel recording): Sz was S (now a macro)
static pofs_slot_t g_pofs_slot[2][8]; static struct { VkBuffer buf; uint64_t base, epoch, o0; } g_pofs_win[2][8];
static struct { uint64_t draws, vs, fs, conv, moves; } g_r133;
static int r133_ok(vkshader_t *s, uint32_t m, int irun) {   // the shader's variant could be made (cached per shader and slot mask, r134, and r151 instance-stream form)
    static struct { const vkshader_t *s; uint32_t m; int r; } t[256]; uint32_t mk = m | (irun ? 0x100u : 0);
    uint32_t h = (uint32_t)((((uintptr_t)s >> 4) ^ mk) * 2654435761u) & 255;
    for (int i = 0; i < 256; i++, h = (h + 1) & 255) { if (t[h].s == s && t[h].m == mk) return t[h].r; if (!t[h].s) break; }
    int r = vk_shader_module(s, 0, 0, 0, 8 | (irun ? 16 : 0) | (int)(m << 8), 0, 0) != s->mod;
    if (!t[h].s) { t[h].s = s; t[h].m = mk; t[h].r = r; }
    VKLOG("r133: shader %016llx stage %d slots %#x: %s variant %s", (unsigned long long)s->key, s->stage, m, irun ? "r151 instance-stream" : "pushed-offset", r ? "made" : "NOT made (left as it is)");
    return r;
}
static void vk_r131_log(void) {   // every 600 presents, from vk_r80_log
    static typeof(g_r131) p; static uint64_t pd; typeof(g_r131) c = g_r131; uint64_t d = g_vkst.draws;
    static typeof(g_r132) q; typeof(g_r132) e = g_r132;   // r132
    static typeof(g_r133) q3; typeof(g_r133) e3 = g_r133;   // r133
    if (g_x_census || g_x_pofs)
        VKLOG("r133: per frame (last 600 presents): DM_VK_X_POFS=%016llx: draws converted %.0f (vertex stage %.0f, fragment stage %.0f), buffers read at a pushed offset %.0f, window moves %.0f; r134 mode %d slots %#x / %#x",
              (unsigned long long)g_x_pofs, (double)(e3.draws - q3.draws) / 600, (double)(e3.vs - q3.vs) / 600, (double)(e3.fs - q3.fs) / 600, (double)(e3.conv - q3.conv) / 600, (double)(e3.moves - q3.moves) / 600, g_pofs_mode, g_pofs_m[0], g_pofs_m[1]);
    q3 = e3;
    if (g_x_census || g_x_push || g_x_inst)
        VKLOG("r132: per frame (last 600 presents): DM_VK_X_PUSH=%d: draws that set the push constant %.0f, draws that do not %.0f; DM_VK_X_INST=%d: plain draws given a first instance %.0f, plain draws not %.0f",
              g_x_push, (double)(e.push - q.push) / 600, (double)(e.pushx - q.pushx) / 600, g_x_inst, (double)(e.inst - q.inst) / 600, (double)(e.instx - q.instx) / 600);
    q = e;
    if (g_x_census || g_x_sc1 || g_x_dup) {
#define R131(f) ((double)(c.f - p.f) / 600)
        VKLOG("r131: draws per frame (last 600 presents): %.0f; plain indexed %.0f, plain not indexed %.0f, indirect %.0f, quads %.0f; with a geometry shader %.0f, tessellated %.0f, rasterizer discard %.0f, instanced %.0f, with a storage binding %.0f; indices per plain draw %.0f; plain draws by index count: up to 32 %.0f, up to 128 %.0f, up to 512 %.0f, up to 2048 %.0f, more %.0f",
              (double)(d - pd) / 600, R131(idx), R131(dir), R131(ind), R131(quad), R131(gs), R131(tes), R131(rd), R131(inst), R131(stor),
              c.idx + c.dir > p.idx + p.dir ? (double)(c.isum - p.isum) / (double)(c.idx + c.dir - p.idx - p.dir) : 0.0, R131(ib[0]), R131(ib[1]), R131(ib[2]), R131(ib[3]), R131(ib[4]));
        VKLOG("r131: state calls per frame (last 600 presents): pipeline binds %.0f, descriptor binds %.0f (%.0f sets; %.0f of the binds with the same sets, only offsets differ), vertex buffer binds %.0f, index buffer binds %.0f, viewports %.0f, scissors %.0f; DM_VK_X_SC1=%d DM_VK_X_DUP=%d: draws issued again %.0f, plain draws not issued again %.0f",
              R131(pipe), R131(dset), R131(dsets), R131(dset_ofs), R131(vb), R131(ibind), R131(vp), R131(sc), g_x_sc1, g_x_dup, R131(dup), R131(dupx));
#undef R131
    }
    { static typeof(g_r135) q5; typeof(g_r135) e5 = g_r135;   // r135
      if (g_x_census || g_x_vb || g_x_ib) VKLOG("r135: per frame (last 600 presents): DM_VK_X_VB=%d vertex buffer binds repeated %.0f; DM_VK_X_IB=%d index buffer binds repeated %.0f",
                                                g_x_vb, (double)(e5.vb - q5.vb) / 600, g_x_ib, (double)(e5.ib - q5.ib) / 600);
      q5 = e5; }
    { static typeof(g_r142) q6; typeof(g_r142) e6 = g_r142;   // r142
      if (g_x_merge) VKLOG("r142: per frame (last 600 presents): DM_VK_X_MERGE=%016llx runs %.0f (of one draw %.0f), draws merged into them %.0f; runs ended by another draw, pass or command buffer %.0f, other state %.0f, other mesh or draw parameters %.0f; draws of that shader outside the class %.0f",
                           (unsigned long long)g_x_merge, (double)(e6.runs - q6.runs) / 600, (double)(e6.single - q6.single) / 600, (double)(e6.merged - q6.merged) / 600,
                           (double)(e6.brk_seq - q6.brk_seq) / 600, (double)(e6.brk_state - q6.brk_state) / 600, (double)(e6.brk_mesh - q6.brk_mesh) / 600, (double)(e6.notok - q6.notok) / 600);
      q6 = e6; }
    { static typeof(g_r151) q7; typeof(g_r151) e7 = g_r151;   // r151
      if (g_x_irun) VKLOG("r151: per frame (last 600 presents): DM_VK_X_IRUN %d vertex shaders listed, auto %d: runs %.0f, draws in them %.0f (draws with the game's own instances %.0f); runs ended by another draw, pass or command buffer %.0f, other state %.0f, other mesh or draw parameters %.0f, full %.0f; draws of that shader not taken %.0f (no shader variant %.0f)",
                          g_nirun, g_irun_auto, (double)(e7.runs - q7.runs) / 600, (double)(e7.draws - q7.draws) / 600, (double)(e7.inst - q7.inst) / 600, (double)(e7.brk_seq - q7.brk_seq) / 600,
                          (double)(e7.brk_state - q7.brk_state) / 600, (double)(e7.brk_mesh - q7.brk_mesh) / 600, (double)(e7.brk_full - q7.brk_full) / 600, (double)(e7.notok - q7.notok) / 600, (double)(e7.novar - q7.novar) / 600);
      if (g_x_irun) for (int i = 0; i < 256; i++) { r152_e *x = &g_r152[i]; if (!x->key || x->draws == x->p[0]) continue;   // r152
          VKLOG("r152: per frame (last 600 presents): vertex shader %016llx%s: draws %.0f, runs %.0f; runs ended by a window move %.0f, by other state %.0f", (unsigned long long)x->key, x->qual ? " (auto)" : "",
                (double)(x->draws - x->p[0]) / 600, (double)(x->runs - x->p[1]) / 600, (double)(x->brk_win - x->p[2]) / 600, (double)(x->brk_state - x->p[3]) / 600);
          x->p[0] = x->draws; x->p[1] = x->runs; x->p[2] = x->brk_win; x->p[3] = x->brk_state; }
      q7 = e7; }
    if (g_x_vstab) r143_report(); else if (g_r143_full || g_r143[0].n) { memset(g_r143, 0, sizeof g_r143); g_r143_full = 0; }   // r143
    p = c; pd = d;
}
void vk_r80_log(void) {   // every 600 presents (gfx_vk_present.c): the most used viewports in targets 960 wide or more, then the table starts again
    vk_r131_log();   // r131
    if (!g_r80_nvp) return; char b[900]; int o = 0;
    for (int k = 0; k < 6 && k < g_r80_nvp; k++) { int m = k; for (int i = k + 1; i < g_r80_nvp; i++) if (g_r80_vp[i].n > g_r80_vp[m].n) m = i;
        typeof(g_r80_vp[0]) t = g_r80_vp[k]; g_r80_vp[k] = g_r80_vp[m]; g_r80_vp[m] = t;
        o += snprintf(b + o, sizeof b - (size_t)o, "%s%dx%d in %dx%d: %llu", k ? ", " : "", g_r80_vp[k].vw, g_r80_vp[k].vh, g_r80_vp[k].pw, g_r80_vp[k].ph, (unsigned long long)g_r80_vp[k].n); }
    VKLOG("r80: viewports (last 600 presents, draws per guest viewport in physical render target)%s %s%s", "", b, g_r80_vp_lost ? " (more sizes not counted)" : "");
    g_r80_nvp = 0; g_r80_vp_lost = 0;
}
// The per-draw timers (about 10 a draw, ~3,000 draws a frame): the arm64 counter register read directly, a few ns against clock_gettime's ~20-40, in ns
static inline uint64_t tick(void) {
#if defined(__aarch64__)
    static uint64_t q; if (!q) { uint64_t f; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f)); q = (1000000000ull << 32) / f; }
    uint64_t c; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(c)); return (uint64_t)(((unsigned __int128)c * q) >> 32);
#else
    return tnow();
#endif
}
// replaydiag (DM_VK_REPLAYDIAG=1, log only, off by default; OOT_BARRIER_DESIGN.md 17): where the replay thread's time goes outside the draws already timed. Slots of
// g_rd_ns (ns, counts in g_rd_n): 0-9 mtl_sub_begin stages, 10-16 mtl_sub_commit stages (gfx_vk_core.c), 20-25 a dispatch's parts (dispatch below), 30-33 a draw's
// payload by binding kind, every 8th draw (eden_payload). The record ops' times (s15's ss_mark, here on the counter register) and these are logged per frame with
// the replay line (gfx_vk_present.c rd_log). Off: one load of g_rdiag per site
int g_rdiag = -1;
int rd_on(void) { if (g_rdiag < 0) { const char *e = getenv("DM_VK_REPLAYDIAG"); g_rdiag = e && *e == '1'; } return g_rdiag; }
uint64_t g_rd_ns[64], g_rd_n[64];
static uint64_t g_rd_t;   // the last mark (one replay thread)
static inline void rd_mark(int k) { uint64_t t = tick(); g_rd_ns[k] += t - g_rd_t; g_rd_n[k]++; g_rd_t = t; }
static inline void rd_start(void) { g_rd_t = tick(); }
void vk_rd_start(void) { rd_start(); }   // (gfx_vk_core.c)
void vk_rd_mark(int k) { rd_mark(k); }
// Measurement switches, read again at every submission so DM_CTL "env" lines change them while the game runs (never for normal play: they draw wrong):
// DM_VK_X_NOBAR=1 no barrier before a pass, DM_VK_X_DEPTH=1 depth neither loaded nor stored, DM_VK_X_COLOR=1 colour not loaded unless cleared
static int g_x_passbc;   // DM_VK_PASS_BC_RESET (x_read): -1 unset, 1 the bind cache emptied at every pass, 0 never
static int g_x_nobar, g_x_depth, g_x_color, g_x_cbdraws, g_x_tracepass = -1, g_sub_pass;   // DM_VK_TRACE_PASS=n: log every draw of the submission's pass n (once)
static struct { uint64_t key; int pass; unsigned hits; } g_skip[8]; static int g_nskip;
// r81 DM_VK_OPLOG=WxH (live, diagnostic): for the next 8 SET_RT whose first target (colour 0, else depth) has that size, the record ops replayed until the next SET_RT are counted
// and logged ("oplog:"), with the draws a false render condition dropped: to see what a game does in a pass that has no draws (Bayonetta 3's 800x448 effect target)
static char g_oplog_sz[32]; static int g_oplog_n, g_oplog_seg; static unsigned g_oplog_cnt[OP_COUNT_], g_oplog_rcskip;
static int g_x_listpasses;
uint64_t vk_replay_ns(void) { return g_vkst.ns_exec; }   // the replay time so far (gfx_vk_present.c: a frame's cost for the adaptive rate)
// s45 (TotK run 61, 11-13: a frame's replay 13.5 ms on the game's thread, ~4,900 indexed draws at 1.95 us): replay fast paths in Eden mode, DM_VK_S45=<bits>
// (live, default 15): 1 the pipeline key kept while no op that can change its inputs ran (g_s45_kgen), 2 Eden's dynamic state not recorded again, 4 the depth
// bounds likewise, 8 driver segments reused while their inputs' generations hold (gfx_vk_eden.inc). DM_VK_S45_VERIFY=1 (live): computed as before and compared.
// The record ops that cannot change an input of the pipeline key (eden_fill_state reads S's states, formats, shaders); every other op counts in g_s45_kgen
#define S45_B(o) (1ull << (o))
#define S45_KSAFE (S45_B(OP_END) | S45_B(OP_JUMP) | S45_B(OP_BEGIN) | S45_B(OP_CALL) | S45_B(OP_SIGNAL_EVENT) | S45_B(OP_REPORT_COUNTER) | S45_B(OP_RESET_COUNTER) | \
    S45_B(OP_FENCE_SYNC) | S45_B(OP_UPDATE_UB) | S45_B(OP_VIEWPORTS) | S45_B(OP_SCISSOR) | S45_B(OP_DEPTH_RANGE) | S45_B(OP_BIND_VBUF) | S45_B(OP_BIND_VBUF1) | \
    S45_B(OP_BIND_UBO) | S45_B(OP_BIND_UBO_SMALL) | S45_B(OP_BIND_SSBO) | S45_B(OP_BIND_TEX) | S45_B(OP_BIND_SAMP) | S45_B(OP_BIND_IMG) | S45_B(OP_BIND_TEX1) | \
    S45_B(OP_STENCIL) | S45_B(OP_POLY_OFFSET) | S45_B(OP_BLEND_COLOR) | S45_B(OP_PUSH_CONST) | S45_B(OP_COPY_STRIDE) | S45_B(OP_TF_BIND) | \
    S45_B(OP_DRAW) | S45_B(OP_DRAW_IDX) | S45_B(OP_MULTI_DRAW))
static int g_s45_on = 15, g_s45_verify, g_s45_logged;   // R1 (parallel recording): g_s45_kgen, g_s45_list in struct rec_ctx
// R1b (parallel recording): g_s45_nk, g_s45_ns, g_s45_nd, g_s45_bad (kept / made: key, Eden state, driver segments; mismatches, verify) in struct rec_eden (gfx_vk_eden.inc)
// flicker experiments (DM_CTL env lines, read every submission): DM_VK_X_NOSETC no descriptor set reuse, DM_VK_X_NODRVC no driver segment reuse, DM_VK_X_LANDCUT deferred
// uniform updates land at each cut, DM_VK_X_NOKEEP no pass kept for the same targets, DM_VK_X_NOSHC no shadow reuse
static int g_fdz = 1;  // DM_VK_FDZ (live): fragment-depth variants for draws on depth written by fragment shaders (pipeline key fdz)
static int g_deq = 0;   // (S1c (parallel recording): g_deq_dir in rec_xs_t) r57 DM_VK_DEPTH_EQ=<units> (r61: 0 = off by default; the bias let surfaces behind pass, Bayonetta 3 hair tips flickered with draw order): depth EQUAL without depth writes tested as LEQUAL / GEQUAL with that much bias towards the camera (prepare_draw deq)
static int g_dither64;   // r58 DM_VK_DITHER64=1 (test): 64x64 fragment textures read as zeros (tex_desc_for)
static int g_atest = 1;  // r56 DM_VK_ALPHA=0: no alpha test (the colour state's function and nvnCommandBufferSetAlphaRef ignored, as before)
int vk_spv_alt_has(const vkshader_t *s);   // r146 (gfx_vk_shader.c)
static int g_x_nosetc, g_x_nodrvc, g_x_landcut, g_x_nokeep, g_x_nobc, g_xfb_on = 1, g_x_nogs, g_x_nodb, g_spv_alt; int g_x_noshc;   // g_xfb_on: DM_VK_XFB=0 captures no transform feedback (A/B)
static int g_x_nosetmemo, g_x_setverify;   // DM_VK_X_NOSETMEMO=1 (live, A/B): no per-stage set memo; DM_VK_X_SETVERIFY=1 (live): one memo hit in 64 checked (see stage_set)
static int g_cmp_fix; static uint64_t g_cmp_n;   // r46 (cmp_check)
static int g_x_allmask; static uint64_t g_x_texfs, g_x_texfs_mask; static int g_rd_touch = 1, g_x_rdkeep, g_x_nskipvs; static uint64_t g_x_skipvs[8];   // DM_VK_RD_TOUCH, DM_VK_X_RDKEEP, DM_VK_X_SKIPVS (see prepare_draw)
static char g_x_tracesize[40], g_x_traceframe[40]; static int g_x_traceframe_on;
// DM_VK_WATCH=<pool offset, hex>[#n] (live): log the next n (20) uses of images at that offset of their pool (attachment, sampled, copy), with the pass and present
// numbers, to find what fills a texture a pass reads
static long long g_watch_off = -1; static int g_watch_n; extern uint64_t g_tex_frame; static int g_in_dispatch;   // xwatch
// vbytes (NXN1X_VBYTES=1, log only): the bytes the draws' fixed-function vertex and index fetch reads from the memory pools, per present. On N1X that fetch is
// 100-250x slower from imported (zero-copy) pages than from device-local memory (scratchpad membench, 11-20), so this sizes a device-local copy of those
// ranges: as bound (index count x size, vertex stream declared size), distinct per present, and distinct per pass (summed, and the largest pass)
typedef struct { uint64_t a, n, gen; } vb_rng;
#define VB_H 16384
static vb_rng g_vb_frame[VB_H], g_vb_pass[VB_H];
static uint64_t g_vb_present = ~0ull, g_vb_psub = ~0ull, g_vb_fgen = 1, g_vb_pgen = 1; static int g_vb_ppass = -1, g_vb_on = -1;
static struct { uint64_t idraws, ind, ibytes, vbytes, uidx, uvtx, passes, passsum, pass_max, pass_cur, vmax, frames, ovf; } g_vbs;
// idxmir (NXN1X_IDX_MIRROR=1, live A/B, default off): an indexed draw's indices are copied by the CPU at record time into a host-coherent buffer the driver
// allocated (heap 1, the same system memory), bound instead of the imported pool range. membench (11-20): index fetch from imported pages is ~80x slower than
// from such a buffer. A/B only: indices a GPU write earlier in the same submission produced would be read stale (not handled yet); the 64 MB ring is reused
// after it wraps without waiting (about 3 MB a present in HWAoI: 20 presents before reuse)
// vtxmir (NXN1X_VTX_MIRROR=1, live A/B, default off): the same for the vertex streams Eden's pipelines bind (gfx_vk_eden.inc): HWAoI's grass vertex shader
// loads 11 attributes through fixed-function fetch from the imported pools (membench: vertex fetch from imported pages ~200x slower). A stream is copied once
// per submission (same address and size: the copy made earlier in the submission is bound again); streams over 16 MB stay in the pools. The ring is 256 MB
static int g_imir, g_vmir; static VkBuffer g_imir_b; static VkDeviceMemory g_imir_m; static uint8_t *g_imir_p; static VkDeviceSize g_imir_at; static uint64_t g_imir_bytes, g_imir_draws;
#define IMIR_SIZE (256ull << 20)
typedef struct { uint64_t a, n, sub; VkDeviceSize o; } vmir_e;
static vmir_e g_vmir_t[4096]; static uint64_t g_vmir_copied, g_vmir_reused, g_vmir_big, g_vmir_logged;
static int imir_put(const void *src, VkDeviceSize n, VkBuffer *b, VkDeviceSize *o) {
    if (!g_imir_b) { static int failed; if (failed) return 0;
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, IMIR_SIZE, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VkMemoryRequirements rq; uint32_t t = UINT32_MAX;
        if (vkCreateBuffer(V.dev, &bi, NULL, &g_imir_b)) { g_imir_b = VK_NULL_HANDLE; failed = 1; return 0; }
        vkGetBufferMemoryRequirements(V.dev, g_imir_b, &rq); t = vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, t}; void *p = NULL;
        if (t == UINT32_MAX || vkAllocateMemory(V.dev, &ai, NULL, &g_imir_m) || vkBindBufferMemory(V.dev, g_imir_b, g_imir_m, 0) || vkMapMemory(V.dev, g_imir_m, 0, VK_WHOLE_SIZE, 0, &p)) {
            VKLOG("[RED] idxmir: no ring buffer (memory type %u): index buffers stay in the pools", t); failed = 1; return 0; }
        g_imir_p = p; VKLOG("idxmir/vtxmir: 256 MB ring in memory type %u", t); }
    if (n > IMIR_SIZE / 4) return 0;
    if (g_imir_at + n > IMIR_SIZE) g_imir_at = 0;
    memcpy(g_imir_p + g_imir_at, src, (size_t)n); *b = g_imir_b; *o = g_imir_at;
    g_imir_at = (g_imir_at + n + 255) & ~(VkDeviceSize)255; g_imir_bytes += n; g_imir_draws++;
    return 1;
}
static int vmir_put(uint64_t a, uint64_t n, VkBuffer *b, VkDeviceSize *o) {   // vtxmir: a vertex stream, copied once per submission
    extern uint64_t g_nsubs;
    if (n > (16ull << 20)) { g_vmir_big++; return 0; }
    vmir_e *e = &g_vmir_t[((a >> 4) * 0x9E3779B97F4A7C15ull ^ n) >> 52];
    if (e->sub == g_nsubs + 1 && e->a == a && e->n == n) { *b = g_imir_b; *o = e->o; g_vmir_reused++; return 1; }
    if (!imir_put((const void *)(uintptr_t)a, n, b, o)) return 0;
    e->a = a; e->n = n; e->sub = g_nsubs + 1; e->o = *o; g_vmir_copied += n;
    if (g_tex_frame >= g_vmir_logged + 600) { g_vmir_logged = g_tex_frame; VKLOG("vtxmir: present %llu: %.1f MB copied, %llu binds reused within their submission, %llu streams over 16 MB left in the pools (since the last line)",
        (unsigned long long)g_tex_frame, g_vmir_copied / 1048576.0, (unsigned long long)g_vmir_reused, (unsigned long long)g_vmir_big); g_vmir_copied = g_vmir_reused = g_vmir_big = 0; }
    return 1;
}
// vbmir (default on, NXN1X_VB_MIRROR=0 off, live): fixed-function vertex and index fetch reads device-local mirrors of the zero-copy pools, not the imported
// pages. Cause (11-20): on N1X (NVIDIA, Windows on ARM) that fetch from VK_EXT_external_memory_host memory is ~100-250x slower than from device-local or driver-
// allocated memory (scratchpad membench; shader loads, texel buffers, uniform buffers and copies from the same pages are full speed). HWAoI's grass vertex shader
// loads 11 attributes so: 16 fps; its streams copied into driver memory (vtxmir A/B), 30 fps, as the Adreno build. A mirror is one device-local buffer per pool
// chunk (vkpool_t, by serial) that a vertex or index binding falls in, made at its first such binding; its 64 KB blocks are copied from the chunk's import with
// vkCmdCopyBuffer in the submission's upload command buffer (before its main buffers; a barrier from all earlier work first, so neither an earlier submission
// still reading a block nor its GPU writes to the source race the copy, and one to vertex input after) when a binding reads a block that is not current:
// - never copied, or written by the GPU since (vk_vbm_gpu_wrote from sub_gpu_wrote: copies, fills, storage writes, transform feedback; a raw global memory log:
//   every block), or
// - written by the CPU since its copy (nvn_dirty: the block's pages are write-protected at the copy, a guest write faults and stamps them; nvn_dirty_since).
// A binding over a range the open submission's GPU work wrote before it (vk_sub_gw_overlaps) reads the import as before: a copy in the upload buffer would
// run before that write. A block is checked once per submission (the copy runs at the submission's start, so later CPU writes in it are the game's own race).
// Chunks that are not zero-copy (copied pools: their shadow is driver memory already) or bridges keep their buffers. Mirrors of chunks gone are freed once
// their last submission completed
typedef struct { uint64_t serial, lo, hi, last_fence, last_present, vsub, vlo, vhi; VkBuffer b; VkDeviceMemory m; const void *span; uint32_t nblk; uint64_t *stamp, *csub; uint8_t *valid, *hits, *vol; } vbm_t;
// vbvol (11-20): blocks are pages, so only the pages a vertex or index binding covers are write-protected (64 KB blocks covered data the game rewrites
// each frame: 2,620 write faults a second and 10.5 M host pre-write unprotects, and two fast-fail 0xd crashes in the fault/resume path). A page found
// CPU-written at 3 checks in a row becomes volatile: no longer protected, copied again at every submission that binds it (a few MB a frame by GPU copy)
#define VBM_BLK 4096ull
#define VBM_VOL_HITS 3
static vbm_t *g_vbm; static int g_nvbm, g_capvbm, g_vbm_on = 1, g_vbm_last = -1; static uint64_t g_vbm_cpu, g_vbm_vol, g_vbm_flt0, g_vbm_prep0, g_vbm_t0; static uint64_t g_vbm_bytes, g_vbm_copied, g_vbm_copies, g_vbm_import, g_vbm_binds, g_vbm_logp, g_vbm_upgen = ~0ull; static int g_vbm_open;
void nvn_dirty_protect(const void *addr, size_t len); uint64_t nvn_dirty_now(void); const void *nvn_dirty_span(const void *addr, size_t len); int nvn_dirty_since_span(const void *span, const void *addr, size_t len, uint64_t stamp);
int vk_sub_gw_overlaps(uint64_t a, uint64_t n); uint64_t vk_fence_next(void); VkCommandBuffer vk_upload_cmd(void); extern uint64_t g_up_gen;
// vtxpull (design\VERTEX_PULLING_DESIGN.md phase B): NXN1X_VTX_PULL=1 (default 0, live, applied at the next present) lets Eden pipelines' vertex stages read their
// streams through a fetch table of device addresses into the zero-copy pools instead of fixed-function vertex input (gfx_vk_eden.inc vpull_decide); g_vpull_direct:
// the draw being prepared is a direct one (draw / draw_idx: no indirect arguments, no multi-draw, no quads), the only kind pulled
static int g_vpull_cfg; static _Thread_local int g_vpull_direct;   // (S1c (parallel recording): g_vpull_direct is the draw being recorded's: per thread)
static void vbm_free(vbm_t *m) { vkDestroyBuffer(V.dev, m->b, NULL); vkFreeMemory(V.dev, m->m, NULL); free(m->stamp); free(m->csub); free(m->valid); free(m->hits); free(m->vol); g_vbm_bytes -= m->hi - m->lo; *m = g_vbm[--g_nvbm]; g_vbm_last = -1; }
uint64_t g_vbm_stale;   // hybrid1 diagnostic: valid mirror blocks made stale by GPU writes (gfx_vk_core.c hy_report attributes them by binding kind)
void vk_vbm_gpu_wrote(uint64_t a, uint64_t n) {   // (gfx_vk_core.c sub_gpu_wrote, vk_gml_note) the blocks over [a, a+n) are copied again before their next use
    vk_par_lock();   // R6/R7/R8 (parallel recording), R8: the mirror table (as vbm_bind)
    for (int i = 0; i < g_nvbm; i++) { vbm_t *m = &g_vbm[i];
        if (n == ~0ull) { for (uint32_t k = 0; k < m->nblk; k++) g_vbm_stale += m->valid[k] != 0; memset(m->valid, 0, m->nblk); m->vsub = 0; continue; }
        if (a >= m->hi || a + n <= m->lo) continue;
        uint64_t o0 = a > m->lo ? a - m->lo : 0, o1 = (a + n < m->hi ? a + n : m->hi) - m->lo;
        for (uint64_t k = o0 / VBM_BLK; k <= (o1 - 1) / VBM_BLK; k++) g_vbm_stale += m->valid[k] != 0;
        memset(m->valid + o0 / VBM_BLK, 0, (size_t)((o1 - 1) / VBM_BLK - o0 / VBM_BLK + 1)); m->vsub = 0; }
    vk_par_unlock();
}
void vk_vbm_up_close(VkCommandBuffer cb) {   // (gfx_vk_core.c, at the end of an upload command buffer) the copies made visible to vertex input
    vk_up_lock();   // R6/R7/R8 (parallel recording), R8: g_vbm_open and the upload command buffer (leaf)
    if (!g_vbm_open) { vk_up_unlock(); return; } g_vbm_open = 0;
    VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1, &b, 0, NULL, 0, NULL);
    vk_up_unlock();
}
static vbm_t *vbm_get(vkpool_t *p) {
    if (g_vbm_last >= 0 && g_vbm_last < g_nvbm && g_vbm[g_vbm_last].serial == p->serial) return &g_vbm[g_vbm_last];
    for (int i = 0; i < g_nvbm; i++) if (g_vbm[i].serial == p->serial) { g_vbm_last = i; return &g_vbm[i]; }
    static uint64_t failed_serial; if (failed_serial == p->serial) return NULL;
    const uint64_t size = p->hi - p->lo; const void *span = nvn_dirty_span((const void *)(uintptr_t)p->lo, (size_t)size);
    VkBuffer b = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; const char *why = !span ? "its pool is not write-tracked" : NULL;
    if (!why) { VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
        if (vkCreateBuffer(V.dev, &bi, NULL, &b)) { b = VK_NULL_HANDLE; why = "vkCreateBuffer"; } }
    if (!why) { VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, b, &rq); uint32_t t = vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, t};
        if (t == UINT32_MAX || !(V.mem.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) || vkAllocateMemory(V.dev, &ai, NULL, &mem)) { mem = VK_NULL_HANDLE; why = "device-local memory"; }
        else if (vkBindBufferMemory(V.dev, b, mem, 0)) why = "vkBindBufferMemory"; }
    if (why) { failed_serial = p->serial; if (b) vkDestroyBuffer(V.dev, b, NULL); if (mem) vkFreeMemory(V.dev, mem, NULL);
        static int n; if (n++ < 8) VKLOG("vbmir: pool chunk %#llx +%#llx keeps its import for vertex/index fetch (%s)", (unsigned long long)p->lo, (unsigned long long)size, why); return NULL; }
    if (g_nvbm == g_capvbm) { g_capvbm = g_capvbm ? 2 * g_capvbm : 64; g_vbm = realloc(g_vbm, sizeof *g_vbm * (size_t)g_capvbm); }
    vbm_t *m = &g_vbm[g_nvbm]; memset(m, 0, sizeof *m); m->serial = p->serial; m->lo = p->lo; m->hi = p->hi; m->b = b; m->m = mem; m->span = span;
    m->nblk = (uint32_t)((size + VBM_BLK - 1) / VBM_BLK); m->stamp = calloc(m->nblk, sizeof *m->stamp); m->csub = calloc(m->nblk, sizeof *m->csub); m->valid = calloc(m->nblk, 1); m->hits = calloc(m->nblk, 1); m->vol = calloc(m->nblk, 1);
    g_vbm_bytes += size; g_vbm_last = g_nvbm++;
    { static int n; if (n++ < 4) VKLOG("vbmir: vertex/index fetch from device-local mirrors of the zero-copy pools (first: chunk %#llx +%#llx; NXN1X_VB_MIRROR=0 off)", (unsigned long long)p->lo, (unsigned long long)size); }
    return m;
}
static void vbm_tick(void) {   // every 600 presents: the log line; mirrors whose chunk is gone, freed once their last submission completed
    extern uint64_t g_tex_frame; if (g_tex_frame < g_vbm_logp + 600) return;
    const double f = (double)(g_tex_frame - g_vbm_logp); g_vbm_logp = g_tex_frame;
    uint64_t done = vk_fence_done(); int freed = 0;
    for (int i = g_nvbm - 1; i >= 0; i--) { vbm_t *m = &g_vbm[i]; vkpool_t *p = vk_pool_of(m->lo, 1);
        if ((!p || p->serial != m->serial) && m->last_fence <= done) { vbm_free(m); freed++; } }
    VKLOG("vbmir: per present %.2f MB copied in %.1f copies, %.0f bindings from mirrors, %.1f bound in the pool (written by the GPU earlier in their submission); mirrors %d, %.1f MB device-local%s",
          g_vbm_copied / f / 1048576.0, g_vbm_copies / f, g_vbm_binds / f, g_vbm_import / f, g_nvbm, g_vbm_bytes / 1048576.0, freed ? " (some freed: their chunks are gone)" : "");
    { void nvn_dirty_stats(uint64_t *faults, uint64_t *preps); uint64_t flt, prep; nvn_dirty_stats(&flt, &prep); const uint64_t t = tick(); const double s = g_vbm_t0 ? (t - g_vbm_t0) / 1e9 : 0;   // vbvol
      if (s > 0) VKLOG("vbvol: write-protection faults %.0f/s, host pre-write unprotects %.0f/s (all of nvn_dirty); mirror pages CPU-written at their check %.1f per present, volatile pages %llu (%.1f MB)",
                       (flt - g_vbm_flt0) / s, (prep - g_vbm_prep0) / s, g_vbm_cpu / f, (unsigned long long)g_vbm_vol, g_vbm_vol * VBM_BLK / 1048576.0);
      g_vbm_flt0 = flt; g_vbm_prep0 = prep; g_vbm_t0 = t; g_vbm_cpu = 0; }
    g_vbm_copied = g_vbm_copies = g_vbm_binds = g_vbm_import = 0;
}
// a vertex or index binding of [a, a+n) (as vk_buf_at found it): 1 = bound from the mirror instead (*buf, *off replaced)
static int vbm_bind_0(uint64_t a, uint64_t n, VkBuffer *buf, VkDeviceSize *off);
// R6/R7/R8 (parallel recording), R8: the mirror table (g_vbm and its blocks, g_vbm_last, the counters) under the outer recording lock (gfx_vk.h); its copies take the
// leaf vk_up_lock inside. vk_pool_of / vk_buf_at (the pool lock) run under it: the order is vk_par_lock -> the pool lock -> vk_up_lock
static int vbm_bind(uint64_t a, uint64_t n, VkBuffer *buf, VkDeviceSize *off) {
    if (!g_vbm_on || !n) return 0;
    vk_par_lock(); const int r = vbm_bind_0(a, n, buf, off); vk_par_unlock();
    return r;
}
static int vbm_bind_0(uint64_t a, uint64_t n, VkBuffer *buf, VkDeviceSize *off) {
    extern uint64_t g_nsubs;
    if (!g_vbm_on || !n) return 0;
    vbm_tick();
    vkpool_t *p = vk_pool_of(a, n); if (!p || !p->zero_copy || p->bridge) return 0;
    if (vk_sub_gw_overlaps(a, n)) { g_vbm_import++; return 0; }
    vbm_t *m = vbm_get(p); if (!m) return 0;
    const uint64_t o = a - m->lo, e = o + n, sub = g_nsubs + 1;
    if (!(m->vsub == sub && o >= m->vlo && e <= m->vhi)) {
        uint64_t run_lo = 0, run_hi = 0; int ok = 1;
        for (uint64_t blk = o / VBM_BLK; blk <= (e - 1) / VBM_BLK && ok; blk++) {
            const uint64_t blo = blk * VBM_BLK, bn = m->hi - m->lo - blo < VBM_BLK ? m->hi - m->lo - blo : VBM_BLK;
            int need;
            if (m->valid[blk] && m->csub[blk] == sub) need = 0;   // checked in this submission
            else if (m->vol[blk]) need = 1;   // vbvol: volatile, copied at each submission that binds it
            else { const int cpu = m->valid[blk] && nvn_dirty_since_span(m->span, (const void *)(uintptr_t)(m->lo + blo), (size_t)bn, m->stamp[blk]);
                need = !m->valid[blk] || cpu;
                if (cpu) { g_vbm_cpu++; if (++m->hits[blk] >= VBM_VOL_HITS) { m->vol[blk] = 1; g_vbm_vol++; } }
                else if (m->valid[blk]) m->hits[blk] = 0; }
            m->csub[blk] = sub;
            if (need) { if (!m->vol[blk]) { m->stamp[blk] = nvn_dirty_now(); nvn_dirty_protect((const void *)(uintptr_t)(m->lo + blo), (size_t)bn); } m->valid[blk] = 1;
                if (run_hi == blo) run_hi = blo + bn; else { run_lo = blo; run_hi = blo + bn; } }
            const int last = blk == (e - 1) / VBM_BLK;
            if (run_hi > run_lo && (last || run_hi != blo + bn)) {   // a run ends: copy it
                VkBuffer sb; VkDeviceSize so;
                if (!vk_buf_at(m->lo + run_lo, run_hi - run_lo, &sb, &so)) { memset(m->valid + run_lo / VBM_BLK, 0, (size_t)((run_hi - run_lo + VBM_BLK - 1) / VBM_BLK)); ok = 0; break; }
                vk_up_lock();   // R6/R7/R8 (parallel recording), R8: the upload command buffer and g_vbm_open / g_vbm_upgen (leaf: vk_buf_at above, outside it)
                VkCommandBuffer cb = vk_upload_cmd();
                if (!g_vbm_open || g_vbm_upgen != g_up_gen) { g_vbm_open = 1; g_vbm_upgen = g_up_gen;
                    VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
                    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, NULL, 0, NULL); }
                VkBufferCopy r = {so, run_lo, run_hi - run_lo}; vkCmdCopyBuffer(cb, sb, m->b, 1, &r); vk_up_unlock();
                g_vbm_copied += run_hi - run_lo; g_vbm_copies++; run_lo = run_hi = 0; }
        }
        if (!ok) return 0;
        if (m->vsub == sub && o <= m->vhi && e >= m->vlo) { if (o < m->vlo) m->vlo = o; if (e > m->vhi) m->vhi = e; }
        else if (m->vsub != sub || e - o > m->vhi - m->vlo) { m->vsub = sub; m->vlo = o; m->vhi = e; }
    }
    m->last_fence = vk_fence_next(); g_vbm_binds++;
    *buf = m->b; *off = o; return 1;
}
static int vbytes_on(void) { if (g_vb_on < 0) { const char *e = getenv("NXN1X_VBYTES"); g_vb_on = e && atoi(e) != 0; } return g_vb_on; }
static int vb_insert(vb_rng *t, uint64_t a, uint64_t n, uint64_t gen) {
    uint64_t h = (a * 0x9E3779B97F4A7C15ull ^ n * 0xC2B2AE3D27D4EB4Full) >> 50;
    for (int i = 0; i < 64; i++) { vb_rng *e = &t[(h + (uint64_t)i) & (VB_H - 1)]; if (e->gen != gen) { e->a = a; e->n = n; e->gen = gen; return 1; } if (e->a == a && e->n == n) return 0; }
    g_vbs.ovf++; return 0; }
static void vb_pass_close(void) { if (g_vbs.pass_cur > g_vbs.pass_max) g_vbs.pass_max = g_vbs.pass_cur; g_vbs.passsum += g_vbs.pass_cur; g_vbs.pass_cur = 0; }
static void vb_note(int idx, uint64_t a, uint64_t n) {
    extern uint64_t g_nsubs;
    if (!vbytes_on() || !a || !n) return;
    if (g_tex_frame != g_vb_present) {
        if (g_vb_present != ~0ull) { vb_pass_close(); g_vbs.frames++;
            if (g_vbs.frames >= 60) { const double f = (double)g_vbs.frames, M = 1048576.0;
                VKLOG("vbytes: per present over %llu presents: %.0f indexed draws (%.0f indirect), %.0f passes with fetch; as bound: index %.2f MB, vertex %.2f MB; distinct in the present: index %.2f MB, vertex %.2f MB; "
                      "distinct per pass, summed: %.2f MB (largest pass %.2f MB over the window); largest vertex range %.2f MB; table overflows %llu",
                      (unsigned long long)g_vbs.frames, g_vbs.idraws / f, g_vbs.ind / f, g_vbs.passes / f, g_vbs.ibytes / f / M, g_vbs.vbytes / f / M, g_vbs.uidx / f / M, g_vbs.uvtx / f / M,
                      g_vbs.passsum / f / M, g_vbs.pass_max / M, g_vbs.vmax / M, (unsigned long long)g_vbs.ovf);
                memset(&g_vbs, 0, sizeof g_vbs); } }
        g_vb_present = g_tex_frame; g_vb_fgen++; }
    if (g_nsubs != g_vb_psub || g_sub_pass != g_vb_ppass) { vb_pass_close(); g_vb_psub = g_nsubs; g_vb_ppass = g_sub_pass; g_vb_pgen++; g_vbs.passes++; }
    if (idx) { g_vbs.ibytes += n; g_vbs.idraws++; } else { g_vbs.vbytes += n; if (n > g_vbs.vmax) g_vbs.vmax = n; }
    if (vb_insert(g_vb_pass, a, n, g_vb_pgen)) g_vbs.pass_cur += n;
    if (vb_insert(g_vb_frame, a, n, g_vb_fgen)) { if (idx) g_vbs.uidx += n; else g_vbs.uvtx += n; }
}
static const void *g_ndsr_im; static int g_ndsr_wn;   // r27: the NDSR output image after a blit into it, its next uses logged (what overwrites it)
// DM_VK_UBO_LOG=<shader key, hex>[#n] (live): log the next n (40) uniform buffers that shader reads (present, stage, register, address, a hash and the first 16 words)
static uint64_t g_ubolog_key; static int g_ubolog_n;
// DM_VK_UBO_LOG_FROM=<byte offset> (live): the words logged start there (512 bytes from it), and words whose memory differs from what the draw reads
// (deferred updates, in-flight copies) are listed after them as +offset mem=value. DM_VK_UBD_WATCH=<address hex>[#n] (live, gfx_vk_core.c): log the next n (200)
// inline uniform updates and GPU writes that cover that address
// DM_VK_HWWATCH=<address hex>[,<address hex>][#n] (live, r49, hwwatch_win.c): log the next n (16) CPU stores to those addresses (hardware watchpoints): pc, lr, value, backtrace
static int g_ubolog_from; extern uint64_t g_ubd_watch;
static void ubo_log(const vkshader_t *s, int st, int reg, uint64_t a, uint64_t n) {
    if (!g_ubolog_key || s->key != g_ubolog_key || g_ubolog_n <= 0) return;
    g_ubolog_n--; uint8_t buf[4096]; if (n > sizeof buf) n = sizeof buf; void vk_ubd_read(uint64_t a, uint64_t n, uint8_t *dst); vk_ubd_read(a, n, buf);
    uint64_t h = 1469598103934665603ull; for (uint64_t i = 0; i < n; i++) h = (h ^ buf[i]) * 1099511628211ull; float f[16] = {0}; memcpy(f, buf, n < 64 ? n : 64);
    char t[4096]; int k = 0; uint32_t w0 = (uint32_t)g_ubolog_from / 4, nw = (uint32_t)(n < (uint64_t)g_ubolog_from + 512 ? n : (uint64_t)g_ubolog_from + 512) / 4; (void)f;   // 512 bytes from DM_VK_UBO_LOG_FROM, as floats
    if (w0) k += snprintf(t + k, sizeof t - k, "@+%#x ", g_ubolog_from);
    for (uint32_t i = w0; i < nw && k < (int)sizeof t - 24; i++) { float v; memcpy(&v, buf + 4 * i, 4); k += snprintf(t + k, sizeof t - k, "%s%g", i > w0 ? (i % 4 ? " " : " | ") : "", v); }
    { const uint8_t *mem = (const uint8_t *)(uintptr_t)a; int nd = 0;   // memory itself where it differs from what the draw reads
      for (uint32_t i = w0; i < nw && k < (int)sizeof t - 40 && nd < 32; i++) if (memcmp(mem + 4 * i, buf + 4 * i, 4)) { float v; memcpy(&v, mem + 4 * i, 4); k += snprintf(t + k, sizeof t - k, "%s+%#x mem=%g", nd++ ? " " : " || ", 4 * i, v); } }
    if (st == 0 && reg == 0 && S.vbuf[0].address) {   // the vertex stream 0 the draw reads, hashed the same way (CPU view at replay)
        uint64_t vn = S.vbuf[0].size > sizeof buf ? sizeof buf : S.vbuf[0].size, vh = 1469598103934665603ull; memcpy(buf, (const void *)(uintptr_t)S.vbuf[0].address, vn);
        for (uint64_t i = 0; i < vn; i++) vh = (vh ^ buf[i]) * 1099511628211ull;
        float v[6] = {0}; memcpy(v, buf, vn < 24 ? vn : 24);
        if (S.rt_has_dep) VKLOG("ubo: present %llu pass %d depth target pool %#llx+%#llx serial %llu; depth test %d func %d", (unsigned long long)g_tex_frame, g_sub_pass - 1,
                                (unsigned long long)(uintptr_t)S.rt_dep.pool, (unsigned long long)S.rt_dep.off, (unsigned long long)S.rt_dep.serial, S.ds.dt, S.ds.df);
        VKLOG("ubo: present %llu vertex stream 0 %#llx +%llu hash %016llx: %g %g %g %g %g %g", (unsigned long long)g_tex_frame, (unsigned long long)S.vbuf[0].address, (unsigned long long)S.vbuf[0].size,
              (unsigned long long)vh, v[0], v[1], v[2], v[3], v[4], v[5]);
        void vk_ubd_read(uint64_t a, uint64_t n, uint8_t *dst); vk_ubd_read(a, n, buf); }
    VKLOG("ubo: present %llu pass %d stage %d c%d %#llx +%llu hash %016llx: %s", (unsigned long long)g_tex_frame, g_sub_pass - 1, st, reg, (unsigned long long)a, (unsigned long long)n, (unsigned long long)h, t);
}
// DM_VK_TEXDUMP=<pool hex>+<offset hex> (read when a pass dump starts; the trace prints both): during the dump, the first sampled image there is blitted (level 0, up to 6 layers)
// into RGBA16F images and dumped with the passes, so a compressed texture (BC6H cube maps...) can be looked at decoded by the GPU
static long long g_td_off = -1; static unsigned long long g_td_pool; static vkimg_t *g_td_im; static int g_td_done;
void vk_texdump_note(vkimg_t *im) { if (g_td_off >= 0 && !g_td_done && !g_td_im && im && im->off == g_td_off && im->pool == g_td_pool) g_td_im = im; }
void vk_watch(const vkimg_t *im, const char *what) {
    if (im && im == g_ndsr_im && g_ndsr_wn > 0) { g_ndsr_wn--; VKLOG("NDSR watch: present %llu pass %d (%ux%u): %s", (unsigned long long)g_tex_frame, g_sub_pass - 1, S.pass_w, S.pass_h, what); }
    // xwatch (11-22, XCDE title): any image whose memory covers the watched offset (another description over the same storage reads it too), not only one starting there
    if (g_watch_off < 0 || !im || g_watch_off < im->off || g_watch_off >= im->off + im->size || g_watch_n <= 0) return;
    g_watch_n--; VKLOG("watch: present %llu pass %d (%ux%u%s): %s %ux%u fmt %d pool %#llx+%#llx (image %p, nvn flags %#x, %d overlapping)", (unsigned long long)g_tex_frame, g_sub_pass - 1, S.pass_w, S.pass_h, g_in_dispatch ? ", in a dispatch" : "", what, im->w, im->h, im->fmt,
                       (unsigned long long)im->pool, (unsigned long long)im->off, (const void *)im, im->flags, im->novl);
}
static int g_rc_real = 1;   // r52 DM_VK_RC_OCC (conditional rendering, see rc_value)
// occ-s1 (design\OCCLUSION_REDESIGN.md step 1): the stream serial. Every draw-type record (DRAW, DRAW_IDX, MULTI_DRAW, DRAW_TEX) advances the recording thread's g_oser by one,
// recorded or not; a draw being recorded is serial g_oser - 1. The replay thread's is the stream's; the dry pass counts the same records from it (g_sd.oser) and names each
// segment's first (seg_desc.oser0), which a worker counts from and the replay thread's count is checked against (and follows) at a hand-out. Plain counters, no atomics
static _Thread_local uint64_t g_oser;
uint64_t vk_occ2_serial(void) { return g_oser; }
uint64_t g_vk_pends;   // occ-s0: the replay thread's pass ends (vk_end_pass)
static uint32_t g_occd_gen;   // occ-s0: bumped when DM_VK_OCCDIAG turns on (report-table entries noted before carry an older one: their diagnostic fields are not known)
static uint64_t g_occ2_ser_off;   // occ-s1: the replay thread's serial was not the dry pass's (at a segment, at a list's end; must stay 0)
uint64_t vk_occ2_ser_off(void) { return __atomic_load_n(&g_occ2_ser_off, __ATOMIC_RELAXED); }
extern volatile long dm_env_gen;   // r53 compat/posix_win.c
// r136: DM_VK_TRACE_PASS=-2 (list the passes) and DM_VK_TRACE_FRAME end only after a submission of at least DM_VK_TRACE_MIN passes (default 64, live): Hyrule Warriors:
// Age of Imprisonment submits a 2x1 pass on its own before the frame, which used to take the listing; the -2 listing also logs one line per draw (r136_draw)
static int r136_min(void) { const char *e = getenv("DM_VK_TRACE_MIN"); return e && *e ? atoi(e) : 64; }
static void r136_small(int np) { static int n; if (n++ < 8) VKLOG("r136: that submission had %d passes (under DM_VK_TRACE_MIN=%d): the next one is listed or traced too", np, r136_min()); }
// s30a: the DM_CTL file (env DM_CTL) polled by the replay itself: NXN1X builds no host HID service, so dm/host/hle_hid.c ctl_main never runs there.
// At most every 100 ms, at a submission's start (x_read), its new complete lines are read; "env K=V" is applied with setenv here, on the replay thread
// (nvnh_posix.c bumps dm_env_gen, so x_read reads the switches again at once). Blank lines and # comments are skipped; other commands (keys, shot, dump)
// are logged as not supported. Read from the file's start; DM_CTL_FROM_END=1: from its size at the first poll
#include <stdio.h>
// ctlfix (11-21 23:48 run, NPARK: a line added to ctl.txt did not apply, and at 282 s an empty "env DM_VK_OOT_DLSS=" did): the reader kept a byte offset, so a
// file an editor saved again as a whole (line ends, encoding, or lines moved) put the offset in the middle of other lines. It now counts complete lines:
// the lines past the count already handled are applied, whatever the file's bytes before them became; a file that got shorter sets the count to its own
static void ctl_poll(void) {
    static int state; static char path[512]; static long long done_lines; static uint64_t last;   // state: 0 not looked yet, 1 polling, -1 no DM_CTL
    if (state < 0) return;
    const int first = !state;
    if (!state) {
        const char *p = getenv("DM_CTL");
        if (!p || !*p || strlen(p) >= sizeof path) { state = -1; return; }
        strcpy(path, p); state = 1;
    }
    uint64_t t = tnow(); if (!first && last && t - last < 100000000ull) return; last = t;
    FILE *f = fopen(path, "rb"); if (!f) return;
    static char buf[1 << 18]; size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    long long lines = 0; for (size_t i = 0; i < n; i++) lines += buf[i] == '\n';
    if (first) {
        const char *fe = getenv("DM_CTL_FROM_END");
        done_lines = fe && atoi(fe) ? lines : 0;
        VKLOG("s30a: DM_CTL: env lines read from %s at submissions (lines after line %lld; other commands are not supported in NXN1X)", path, done_lines);
        if (done_lines) return;
    }
    if (lines < done_lines) { VKLOG("s30a: DM_CTL: the file has %lld lines, fewer than the %lld handled: lines after %lld are read from now on", lines, done_lines, lines); done_lines = lines; return; }
    if (lines == done_lines) return;
    long long ln = 0; char *s = buf;
    for (;;) {
        char *nl = memchr(s, '\n', (size_t)(buf + n - s));
        if (!nl) break;
        *nl = 0; char *line = s; s = nl + 1;
        if (ln++ < done_lines) continue;
        size_t L = strlen(line); while (L && (line[L - 1] == '\r' || line[L - 1] == ' ' || line[L - 1] == '\t')) line[--L] = 0;
        while (*line == ' ' || *line == '\t' || (unsigned char)*line == 0xEF || (unsigned char)*line == 0xBB || (unsigned char)*line == 0xBF) line++;   // (blanks, a BOM)
        if (!*line || *line == '#') continue;
        if (!strncmp(line, "env ", 4)) {
            char *k = line + 4; while (*k == ' ') k++;
            char *eq = strchr(k, '=');
            if (!eq || eq == k) { VKLOG("s30a: DM_CTL: env line without K=V: %s", line); continue; }
            *eq = 0; setenv(k, eq + 1, 1);
            VKLOG("s30a: DM_CTL: env %s=%s (line %lld)", k, eq + 1, ln);
        } else { static int said; if (said++ < 8) VKLOG("s30a: DM_CTL: \"%s\" is not supported in NXN1X (env lines only)", line); }
    }
    done_lines = lines;
}
static void x_read(void) { ctl_poll(); int r136_np = g_sub_pass; g_sub_pass = 0;   // r53: the switches are read again only after an environment change (was 37 getenv calls a submission)
    if (g_x_listpasses) { g_x_listpasses = 0; if (r136_np >= r136_min()) setenv("DM_VK_TRACE_PASS", "-1", 1); else r136_small(r136_np); }   // r136: a small submission does not end the listing
    if (g_x_traceframe_on) { g_x_traceframe_on = 0; if (r136_np >= r136_min()) setenv("DM_VK_TRACE_FRAME", "", 1); else r136_small(r136_np); }   // r136   // one submission traced
    int dm_title_default_key(int key);
    const int barrier_default = dm_title_default_key(5);
    { static long seen = -1; static int last_default = -1; long gen = dm_env_gen;
      if (gen == seen && barrier_default == last_default) return;
      seen = gen; last_default = barrier_default; }
    { const char *e = getenv("NXN1X_IDX_MIRROR"); int on = e && atoi(e) != 0; if (on != g_imir) { g_imir = on; VKLOG("idxmir: index buffers %s (NXN1X_IDX_MIRROR, live A/B)", on ? "copied by the CPU into a driver-allocated host-coherent ring and bound from there" : "bound in the memory pools (imported)"); } }   // idxmir A/B
    { const char *e = getenv("NXN1X_VB_MIRROR"); int on = !(e && *e && atoi(e) == 0); if (on != g_vbm_on) { g_vbm_on = on; VKLOG("vbmir: vertex/index fetch %s (NXN1X_VB_MIRROR, live)", on ? "from device-local mirrors of the zero-copy pools" : "from the zero-copy pools (imported)"); } }   // vbmir
    { const char *e = getenv("NXN1X_VTX_PULL"); int on = e && atoi(e) != 0; if (on != g_vpull_cfg) { g_vpull_cfg = on; VKLOG("vtxpull: vertex pulling requested %s (NXN1X_VTX_PULL, live: applied at the next present)", on ? "on" : "off"); } }   // vtxpull
    { const char *e = getenv("NXN1X_VTX_MIRROR"); int on = e && atoi(e) != 0; if (on != g_vmir) { g_vmir = on; VKLOG("vtxmir: Eden pipelines' vertex streams %s (NXN1X_VTX_MIRROR, live A/B)", on ? "copied by the CPU into the driver-allocated ring, once per submission, and bound from there" : "bound in the memory pools (imported)"); } }   // vtxmir A/B
    { const char *e45 = getenv("DM_VK_S45"), *v45 = getenv("DM_VK_S45_VERIFY"); int on = e45 && *e45 ? (int)strtol(e45, NULL, 0) : 15, vf = v45 && atoi(v45) != 0;   // s45 (live)
      if (on != g_s45_on || vf != g_s45_verify || !g_s45_logged) { g_s45_logged = 1;
          VKLOG("s45: replay fast paths 0x%x (1 pipeline key, 2 Eden state, 4 depth bounds, 8 driver segments; DM_VK_S45, live)%s", on, vf ? ", each checked against the full path (DM_VK_S45_VERIFY=1)" : ""); }
      g_s45_on = on; g_s45_verify = vf; g_s45_kgen++; }
    const char *c = getenv("DM_VK_CB_DRAWS"); int tcb = dm_title_default_key(6); g_x_cbdraws = c && *c ? atoi(c) : tcb >= 0 ? tcb : 128;
    { const char *m = getenv("DM_VK_S61_MAXCUT"); int v = m && *m ? atoi(m) : 15; v = v < 0 ? 0 : v > 30 ? 30 : v;   // s61 (live)
      if (v != g_s61_maxcut) VKLOG("s61: at most %d ordinary cuts per submission (DM_VK_S61_MAXCUT, live; 15 before s61)", v); g_s61_maxcut = v; }
    { const char *pb = getenv("DM_VK_PASS_BC_RESET"); g_x_passbc = pb && *pb ? atoi(pb) != 0 : -1; static int was = -2; int m = g_x_passbc > 0;   // r129
      if (m != was) { was = m; VKLOG("r129: bind cache emptied at every pass: %s (%s; DM_VK_PASS_BC_RESET=1 every pass, 0 never, live)", m ? "yes" : "no", g_x_passbc >= 0 ? "DM_VK_PASS_BC_RESET" : "N1X default"); } }
    const char *fy = getenv("DM_VK_FLIPY"); g_flipy = !(fy && !atoi(fy));   // on unless DM_VK_FLIPY=0
    { const char *ro = getenv("DM_VK_RC_OCC"); g_rc_real = !(ro && !atoi(ro)); }   // r52: on unless DM_VK_RC_OCC=0 (see rc_value)
    { const char *od = getenv("DM_VK_OCCDIAG"); const int v = od && atoi(od) != 0;   // occ-s0 (live): the occlusion instruments, every 600 presents
      if (v != g_occdiag) { if (v) g_occd_gen++; VKLOG("occ-s0: occlusion instruments %s (DM_VK_OCCDIAG, live; log only)", v ? "on" : "off"); g_occdiag = v; } }
    const char *tp = getenv("DM_VK_TRACE_PASS"); g_x_tracepass = tp ? atoi(tp) : -1;
    { const char *e = getenv("DM_VK_SKIP_FS"); g_nskip = 0;   // r60 (live): draws whose shaders include one of these keys are not drawn: <key>[@pass][,...]
      for (const char *q = e; q && *q && g_nskip < 8;) { char *n; uint64_t k = strtoull(q, &n, 16); if (n == q) break; int ps = -1; if (*n == '@') ps = (int)strtol(n + 1, &n, 10);
          if (k) { g_skip[g_nskip].key = k; g_skip[g_nskip].pass = ps; g_skip[g_nskip].hits = 0; g_nskip++; } while (*n && *n != ',') n++; q = *n ? n + 1 : n; }
      if (e && *e) VKLOG("DM_VK_SKIP_FS: %d shader keys", g_nskip); }
    { const char *e = getenv("DM_VK_OPLOG"); snprintf(g_oplog_sz, sizeof g_oplog_sz, "%s", e && strcmp(e, "0") ? e : ""); g_oplog_n = g_oplog_sz[0] ? 8 : 0; g_oplog_seg = 0; }   // r81 (live)
    { const char *w = getenv("DM_VK_WATCH"); static char last[40]; if (!w || !*w || !strcmp(w, "0")) g_watch_off = -1;
      else if (strcmp(w, last)) { g_watch_off = strtoll(w, NULL, 16); const char *h = strchr(w, '#'); g_watch_n = h ? atoi(h + 1) : 20; }
      snprintf(last, sizeof last, "%s", w ? w : ""); }
    { const char *w = getenv("DM_VK_UBO_LOG"); static char last[40]; if (!w || !*w || !strcmp(w, "0")) g_ubolog_key = 0;
      else if (strcmp(w, last)) { g_ubolog_key = strtoull(w, NULL, 16); const char *h = strchr(w, '#'); g_ubolog_n = h ? atoi(h + 1) : 40; }
      snprintf(last, sizeof last, "%s", w ? w : ""); }
    { const char *w = getenv("DM_VK_UBO_LOG_FROM"); long v = w ? strtol(w, NULL, 0) : 0; g_ubolog_from = v > 0 && v < 4096 ? (int)(v & ~3L) : 0; }
    { extern uint64_t g_ubd_watch; extern int g_ubd_wn; const char *w = getenv("DM_VK_UBD_WATCH"); static char last[40]; if (!w || !*w || !strcmp(w, "0")) g_ubd_watch = 0;
      else if (strcmp(w, last)) { g_ubd_watch = strtoull(w, NULL, 16); const char *h = strchr(w, '#'); g_ubd_wn = h ? atoi(h + 1) : 200;
        if (!vk_pool_of(g_ubd_watch, 4)) { VKLOG("[RED] DM_VK_UBD_WATCH %#llx is in no memory pool: off", (unsigned long long)g_ubd_watch); g_ubd_watch = 0; } }
      snprintf(last, sizeof last, "%s", w ? w : ""); }
    { const char *w = getenv("DM_VK_HWWATCH"); static char last[64]; void dm_hwwatch_set(uint64_t a0, uint64_t a1, int n);   // r49: hwwatch_win.c
      if (strcmp(w ? w : "", last)) {
          if (!w || !*w || !strcmp(w, "0")) dm_hwwatch_set(0, 0, 0);
          else { char *e; uint64_t a0 = strtoull(w, &e, 16), a1 = *e == ',' ? strtoull(e + 1, &e, 16) : 0; const char *h = strchr(w, '#'); dm_hwwatch_set(a0, a1, h ? atoi(h + 1) : 16); } }
      snprintf(last, sizeof last, "%s", w ? w : ""); }
    { const char *tf = getenv("DM_VK_TRACE_FRAME"); snprintf(g_x_traceframe, sizeof g_x_traceframe, "%s", tf && strcmp(tf, "0") ? tf : ""); }
    { const char *tz = getenv("DM_VK_TRACE_SIZE"); snprintf(g_x_tracesize, sizeof g_x_tracesize, "%s", tz ? tz : ""); } const char *e; g_x_nobar = (e = getenv("DM_VK_X_NOBAR")) && atoi(e); g_x_depth = (e = getenv("DM_VK_X_DEPTH")) && atoi(e); g_x_color = (e = getenv("DM_VK_X_COLOR")) && atoi(e);
    g_x_nosetc = (e = getenv("DM_VK_X_NOSETC")) && atoi(e); g_x_nodrvc = (e = getenv("DM_VK_X_NODRVC")) && atoi(e); g_x_landcut = (e = getenv("DM_VK_X_LANDCUT")) && atoi(e);
    g_x_nokeep = (e = getenv("DM_VK_X_NOKEEP")) && atoi(e); g_x_nobc = (e = getenv("DM_VK_X_NOBC")) && atoi(e); g_x_noshc = (e = getenv("DM_VK_X_NOSHC")) && atoi(e);
    { int vk_set_barrier_mode(int mode); int v = (e = getenv("DM_VK_S63")) && *e ? atoi(e) & 3 : 0;   /* s64a: default 0 (run 93) */
      { int s65 = (e = getenv("DM_VK_S65")) && *e && atoi(e) != 0; if (s65 != g_s65_on) VKLOG("s65: rasterizer-discard passes %s (DM_VK_S65=%d, live)", s65 ? "get a 32x32 render area" : "cover the whole target, as before", s65); g_s65_on = s65; }
      e = getenv("DM_VK_S64");
      int s64 = e && *e ? (atoi(e) == 2 ? 2 : atoi(e) != 0) : (barrier_default == 2 ? 2 : 0);
      if (s64 != g_s64_on) VKLOG("NVN pass barrier policy %d (2: defer texture invalidation across disjoint sampled draws)", s64);
      g_s64_on = s64;
      if (v != vk_set_barrier_mode(v)) VKLOG("s63: full barriers: %s, %s (DM_VK_S63=%d, live)", v & 1 ? "left out with no work since the last" : "all kept", v & 2 ? "write-only source access" : "source access as before", v);
    } g_x_nogs = (e = getenv("DM_VK_X_NOGS")) && atoi(e); g_x_sc1 = (e = getenv("DM_VK_X_SC1")) && atoi(e); g_x_vb = (e = getenv("DM_VK_X_VB")) && atoi(e); g_x_ib = (e = getenv("DM_VK_X_IB")) && atoi(e);   /* r135 */ { int v = (e = getenv("DM_VK_X_DUP")) ? atoi(e) : 0; g_x_dup = v < 0 ? 0 : v > 8 ? 8 : v; } g_x_census = (e = getenv("DM_VK_X_CENSUS")) && atoi(e); g_x_push = (e = getenv("DM_VK_X_PUSH")) && atoi(e); g_x_inst = (e = getenv("DM_VK_X_INST")) && atoi(e);   /* r132 */ g_x_pofs = (e = getenv("DM_VK_X_POFS")) ? strtoull(e, NULL, 16) : 0; g_pofs_mode = (e = getenv("DM_VK_X_POFS_MODE")) ? atoi(e) : 0; /* r142 */ g_x_merge = (e = getenv("DM_VK_X_MERGE")) ? strtoull(e, NULL, 16) : 0; /* r151, r152 */ r152_read(); /* r143 */ g_x_vstab = (e = getenv("DM_VK_X_VSTAB")) && atoi(e);
    g_pofs_m[0] = (e = getenv("DM_VK_X_POFS_VS")) ? (uint32_t)strtoul(e, NULL, 16) & 0xff : 0; g_pofs_m[1] = (e = getenv("DM_VK_X_POFS_FS")) ? (uint32_t)strtoul(e, NULL, 16) & 0xff : 0;   /* r134 */   /* r133 */   /* r131 */ g_x_nodb = (e = getenv("DM_VK_NODB")) && atoi(e); g_spv_alt = (e = getenv("DM_VK_SPV_ALT")) && atoi(e); g_rd_touch = !((e = getenv("DM_VK_RD_TOUCH")) && !atoi(e));
    g_fdz = !((e = getenv("DM_VK_FDZ")) && !atoi(e)); g_atest = !((e = getenv("DM_VK_ALPHA")) && *e == '0'); if ((e = getenv("DM_VK_DEPTH_EQ"))) g_deq = atoi(e); g_dither64 = (e = getenv("DM_VK_DITHER64")) && atoi(e); g_xfb_on = !((e = getenv("DM_VK_XFB")) && !atoi(e));
    g_psplit = (e = getenv("DM_VK_PASS_SPLIT")) ? atoi(e) : 0; g_setsplit = (e = getenv("DM_VK_SETSPLIT")) && atoi(e);
    g_x_allmask = (e = getenv("DM_VK_X_ALLMASK")) && atoi(e);
    { int f = (e = getenv("DM_VK_CMP_FIX")) && atoi(e); if (f != g_cmp_fix) VKLOG("DM_VK_CMP_FIX=%d (mismatches so far %llu)", f, (unsigned long long)g_cmp_n); g_cmp_fix = f; }   // r46
    { const char *q = getenv("DM_VK_X_TEXFS"); char *n = NULL; g_x_texfs = q ? strtoull(q, &n, 16) : 0; g_x_texfs_mask = n && *n == ':' ? strtoull(n + 1, NULL, 16) : ~0ull; }   // DM_VK_X_TEXFS=<fragment key>[:<hex mask of binding indices>] (live, A/B): those textures of draws with that fragment shader are dummy views
    g_x_rdkeep = (e = getenv("DM_VK_X_RDKEEP")) && atoi(e); g_x_nosetmemo = (e = getenv("DM_VK_X_NOSETMEMO")) && atoi(e); g_x_setverify = (e = getenv("DM_VK_X_SETVERIFY")) && atoi(e);
    g_x_nskipvs = 0; for (const char *q = getenv("DM_VK_X_SKIPVS"); q && *q && g_x_nskipvs < 8;) { char *n; uint64_t v = strtoull(q, &n, 16); if (n == q) break; if (v) g_x_skipvs[g_x_nskipvs++] = v; q = *n ? n + 1 : n; } }
// ---------------- conditional rendering (profile render_cond) ----------------
// As nvn_mtl.m (OP_RENDER_COND): SetRenderEnableConditional draws while the 64-bit counters of the reports at addr and addr + 16 are equal / differ, until SetRenderEnable(true).
// Evaluated on the CPU at replay from the records in memory: a report replayed earlier in this submission is not written yet (that happens when the GPU completes), so a timestamp
// one counts with its known counter 0 (vk_sub_complete writes 0 there under the key) and any other one is unknown, which keeps drawing. Inside a block evaluated false, draws and
// clears are dropped. Mario vs. Donkey Kong (gsys, as Tears of the Kingdom and Animal Crossing) clears its HDR scene target under such a condition after lighting it: drawn
// anyway, its levels were black under the HUD

// r52: a condition on a samples-passed report (r50's real occlusion queries) is evaluated from its value once the submission that last reported there has completed:
// the completion thread writes the counter before it advances vk_fence_done, so memory then holds what the GPU would read. A report in this submission or in
// one still in flight stays unknown and keeps drawing; other counter types are written as 0 (not measured) and stay unknown. DM_VK_RC_OCC=0 (live): every
// condition on such an address unknown, as before r52 (and always with DM_VK_OQ=0, whose fixed counter is not a measurement)
typedef struct { uint64_t a, f; int timestamp; int kind; uint32_t dg; uint64_t pf, ser, pe; uint64_t rs; } rc_occ_t;   // f: the fence of the submission that reported at a last; ~0: never a real value there
// (occ-s3: rs = 1 + the report sequence of the samples-passed report there (its record and predicate index, gfx_vk_core.c vk_occ2_rseq), 0 none or another counter)
// (occ-s0, only while DM_VK_OCCDIAG: kind as rc_note's, dg = g_occd_gen when noted (else not known), pf the present, ser the stream serial, pe the replay thread's pass ends: ~0 in the dry pass)
static rc_occ_t *g_rc_occ; static uint32_t g_rc_ocap, g_rc_on;   // addresses ever written by a non-timestamp report (open addressing, a power of two)
static uint64_t g_rc_n[7];   // census: conditions, evaluated false, occlusion values real / this submission / in flight / not measured, draws and clears dropped
static rc_occ_t *rc_occ_find(uint64_t a) {
    if (!g_rc_ocap) return NULL;
    for (uint32_t i = (uint32_t)(a >> 4) & (g_rc_ocap - 1);; i = (i + 1) & (g_rc_ocap - 1)) { if (g_rc_occ[i].a == a) return &g_rc_occ[i]; if (!g_rc_occ[i].a) return NULL; }
}
static void rc_occ_set(uint64_t a, uint64_t f, int timestamp, uint64_t rs) {
    rc_occ_t *e = rc_occ_find(a); if (e) { e->f = f; e->timestamp = timestamp; e->rs = rs; return; }
    if ((g_rc_on + 1) * 2 > g_rc_ocap) {
        uint32_t oc = g_rc_ocap, nc = oc ? oc * 2 : 4096; rc_occ_t *old = g_rc_occ, *t = calloc(nc, sizeof *t); if (!t) die("[vk] no memory for the report address table");
        for (uint32_t i = 0; i < oc; i++) if (old[i].a) { uint32_t j = (uint32_t)(old[i].a >> 4) & (nc - 1); while (t[j].a) j = (j + 1) & (nc - 1); t[j] = old[i]; }
        g_rc_occ = t; g_rc_ocap = nc; free(old);
    }
    uint32_t i = (uint32_t)(a >> 4) & (g_rc_ocap - 1); while (g_rc_occ[i].a) i = (i + 1) & (g_rc_ocap - 1);
    g_rc_occ[i] = (rc_occ_t){a, f, timestamp}; g_rc_occ[i].rs = rs; g_rc_on++;
}
static void rc_note(uint64_t a, int kind, uint64_t rs) {   // kind 1 = timestamp (counter 0), 2 = samples passed, 3 = another counter (written as 0); occ-s3: rs (rc_occ_t)
    uint64_t fence = kind == 1 || (kind == 2 && vk_oq_real()) ? mtl_last_fence() + 1 : UINT64_MAX;
    rc_occ_set(a, fence, kind == 1, kind == 2 ? rs : 0);
}
static void rc_note_diag(uint64_t a, int kind, uint64_t ser, uint64_t pe) {   // occ-s0 (DM_VK_OCCDIAG): after rc_note, where the report is in the stream (ser: the walk's serial)
    rc_occ_t *e = rc_occ_find(a); if (!e) return;
    e->kind = kind; e->dg = g_occd_gen; e->pf = g_tex_frame; e->ser = ser; e->pe = pe;
}
// r50: every NVN counter type the game resets or reports, counted (NPARK 11-05: only timestamps and SAMPLES_PASSED are measured; any other type is written as 0,
// so a type seen here is the next suspect). Each type's first call is logged, and every 16384 calls the totals (Z = reset, R = report, key = type, 11 = other)
static void cnt_note(int rep, uint64_t t, uint64_t ad) {
    static uint64_t n[2][18], tot; static uint32_t seen[2]; int k = t <= 0x10 ? (int)t : 17; n[rep][k]++;
    if (!(seen[rep] >> k & 1)) { seen[rep] |= 1u << k; VKLOG("counter %s type %#llx first seen%s (addr %#llx)", rep ? "report" : "reset", (unsigned long long)t, k == 0 || k == 1 || k == 16 ? "" : " [not measured: reports write 0]", (unsigned long long)ad); }
    if (!(++tot & 0x3fff)) { char b[640]; int m = 0; for (int r = 0; r < 2; r++) for (int i = 0; i < 18; i++) if (n[r][i] && m < (int)sizeof b - 40) m += snprintf(b + m, sizeof b - m, " %c%x=%llu", r ? 'R' : 'Z', i, (unsigned long long)n[r][i]); VKLOG("counter calls:%s", b); }
}
// S1d (parallel recording): rc_value on a given report-table entry (e: NULL none), fence done (RC_FD_LIVE: vk_fence_done() read at each test, as before) and counting (cnt),
// so the dry pass can evaluate a condition in stream order with the table as it is at the record and one fence value for the list, and SERIALSEG can evaluate it again
#define RC_FD_LIVE (~0ull)
static int rc_value_e(uint64_t a, const rc_occ_t *e, uint64_t fd, int cnt, uint64_t *v) {   // 0 = unknown at replay time
    if (e) {
        uint64_t f = e->f;
        if (e->timestamp) {
            *v = f > (fd == RC_FD_LIVE ? vk_fence_done() : fd) ? 0 : *(const volatile uint64_t *)(uintptr_t)a;
            return 1;
        }
        if (f == ~0ull || !g_rc_real) { if (cnt) g_rc_n[5]++; return 0; }
        if (f > (fd == RC_FD_LIVE ? vk_fence_done() : fd)) { if (cnt) g_rc_n[f == mtl_last_fence() + 1 ? 3 : 4]++; return 0; }
        if (cnt) g_rc_n[2]++; *v = *(const volatile uint64_t *)(uintptr_t)a; return 1;
    }
    *v = *(const volatile uint64_t *)(uintptr_t)a; return 1;
}
// S1d2 (parallel recording): what rc_value_e's answer for one word depends on at the replay of the open submission, so the dry pass can tell a condition whose outcome is
// fixed already from one that reads memory or depends on the fence the replay sees. fd: vk_fence_done() at the list's start; cur: the open submission's fence
// (mtl_last_fence() + 1: g_fence moves only at the commit, so no fence the replay can see reaches cur, and fd < cur). RCW_UNK: unknown, whatever the fence and memory
// (a report never measured, DM_VK_RC_OCC=0, a samples-passed report of this submission); RCW_ZERO: 0 (a timestamp of this submission); RCW_FENCE: a report of an earlier
// submission not done at fd: unknown / 0 or memory, as the live fence says; RCW_MEM: read from guest memory (no entry, or its submission done at fd)
enum { RCW_UNK = 0, RCW_ZERO, RCW_FENCE, RCW_MEM };
static int rc_word_class(const rc_occ_t *e, uint64_t fd, uint64_t cur) {
    if (!e) return RCW_MEM;
    if (e->timestamp) return e->f >= cur ? RCW_ZERO : e->f > fd ? RCW_FENCE : RCW_MEM;
    if (e->f == ~0ull || !g_rc_real || e->f >= cur) return RCW_UNK;
    return e->f > fd ? RCW_FENCE : RCW_MEM;
}
// bak88 (11-16, NPARK: MKW's terrain and whole structures flicker; bak83: 41722 of 94208 conditions false, every one read from guest memory): per condition address
// (up to 512), how often it was evaluated, enabled and dropped, unknown, how often its outcome flipped from the previous evaluation, and how often a GPU write
// (a dispatch's storage buffers or a copy, vk_gpu_pending) covering its words was still unfinished when it was read; and the dropped operations split into draws
// and clears. Every 30 s the 10 addresses with the most flips. Logging only; NVN2 boots; DM_VK_RC_DIAG=0 off
typedef struct { uint64_t a; uint32_t ev, en, dr, unk, flips, pend; uint8_t last; uint64_t lx, ly; } rc_diag_t;
enum { RCD_N = 1024 };
static rc_diag_t g_rcd[RCD_N]; static uint32_t g_rcd_n; static uint64_t g_rcd_drop_draw, g_rcd_drop_clear, g_rcd_t, g_rcd_full;
static int rcd_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_RC_DIAG"); v = e && *e ? atoi(e) != 0 : dm_sdk_ounce() != 0; } return v; }
int vk_gpu_pending(uint64_t a, uint64_t n);
static void rcd_note(uint64_t a, int known, int skip, uint64_t x, uint64_t y) {
    uint32_t i = (uint32_t)((a >> 3) * 0x9E3779B1u) & (RCD_N - 1);
    while (g_rcd[i].a && g_rcd[i].a != a) i = (i + 1) & (RCD_N - 1);
    rc_diag_t *d = &g_rcd[i];
    if (!d->a) { if (g_rcd_n >= RCD_N / 2) { g_rcd_full++; return; } d->a = a; g_rcd_n++; d->last = 0xff; }
    const uint8_t out = known ? (uint8_t)skip : 2;
    d->ev++; if (!known) d->unk++; else if (skip) d->dr++; else d->en++;
    if (d->last != 0xff && d->last != out) d->flips++;
    d->last = out; d->lx = x; d->ly = y;
    if (vk_gpu_pending(a, 32)) d->pend++;
}
static void rcd_report(void) {
    const uint64_t t = tnow(); if (t - g_rcd_t < 30000000000ull) return; g_rcd_t = t;
    VKLOG("render_cond diag: %u condition addresses (%llu not tracked: table full); dropped so far: draws %llu, clears %llu; top 10 by outcome flips:",
          g_rcd_n, (unsigned long long)g_rcd_full, (unsigned long long)g_rcd_drop_draw, (unsigned long long)g_rcd_drop_clear);
    uint32_t top[10]; int nt = 0;
    for (uint32_t i = 0; i < RCD_N; i++) { if (!g_rcd[i].a || !g_rcd[i].ev) continue;
        if (nt == 10 && g_rcd[top[9]].flips >= g_rcd[i].flips) continue;
        int p = nt < 10 ? nt++ : 9;
        top[p] = i; while (p > 0 && g_rcd[top[p - 1]].flips < g_rcd[top[p]].flips) { uint32_t s = top[p]; top[p] = top[p - 1]; top[p - 1] = s; p--; } }
    for (int j = 0; j < nt; j++) { const rc_diag_t *d = &g_rcd[top[j]];
        VKLOG("render_cond diag: %#llx: evaluated %u, drawing %u, dropped %u, unknown %u, flips %u, read while a GPU write to it was unfinished %u; last values %#llx / %#llx",
              (unsigned long long)d->a, d->ev, d->en, d->dr, d->unk, d->flips, d->pend, (unsigned long long)d->lx, (unsigned long long)d->ly); }
}
// S1d (parallel recording): a condition (mode < 2) evaluated on the given entries of addr and addr + 16 (rc_value_e); 1 = false (draws dropped)
static int rc_eval(const p_render_cond *rc, const rc_occ_t *ex, const rc_occ_t *ey, uint64_t fd, int cnt, uint64_t *x, uint64_t *y, int *known) {
    *x = 0; *y = 0; *known = rc_value_e(rc->addr, ex, fd, cnt, x) & rc_value_e(rc->addr + 16, ey, fd, cnt, y);
    return *known && (rc->mode ? *x == *y : *x != *y);
}
static void render_cond_note(const p_render_cond *rc, int known, int skip, uint64_t x, uint64_t y);
// occ-s0 (DM_VK_OCCDIAG, design\OCCLUSION_REDESIGN.md step 0 item 3): the render-condition source census, once per condition evaluated (by the dry pass, or at replay): each word's
// class, the mode, y == 0 (the word at addr + 16 in memory now), and for the samples-passed report it reads (x first): its age once landed, the same submission or not, the
// distance from the report to the condition in draws (stream serials) and in pass ends (the replay thread's; not known for a condition the dry pass decided). Log only
enum { OCW_THIS, OCW_FLIGHT, OCW_LANDED, OCW_SPNOQ, OCW_FLAG, OCW_TS, OCW_NEVER, OCW_UNK, OCW_N };
static const char *const g_ocw_names[OCW_N] = {"samples report of this submission", "samples report in flight", "samples report landed", "samples report not measured (DM_VK_OQ=0)",
    "other counter (written as 0 at replay: a CPU flag)", "timestamp", "never reported (guest memory)", "reported before the diagnostic was on"};
static struct { uint64_t n, dry, mode[2], y0, cls[2][OCW_N], age[4], same, other, dd[6], dp[6]; } g_occc;
static int occd_wclass(const rc_occ_t *e, uint64_t cur, uint64_t done) {
    if (!e) return OCW_NEVER;
    if (e->timestamp) return OCW_TS;
    if (!e->dg || e->dg != g_occd_gen) return OCW_UNK;
    if (e->kind == 3) return OCW_FLAG;
    if (e->f == ~0ull) return OCW_SPNOQ;
    return e->f >= cur ? OCW_THIS : e->f > done ? OCW_FLIGHT : OCW_LANDED;
}
static void occd_cond(const p_render_cond *rc, const rc_occ_t *ex, const rc_occ_t *ey, uint64_t fd, uint64_t ser, int dry) {
    if (rc->mode >= 2) return;
    const uint64_t cur = mtl_last_fence() + 1, done = fd == RC_FD_LIVE ? vk_fence_done() : fd;
    const int cx = occd_wclass(ex, cur, done), cy = occd_wclass(ey, cur, done);
    g_occc.n++; g_occc.dry += (uint64_t)dry; g_occc.mode[rc->mode & 1]++; g_occc.cls[0][cx]++; g_occc.cls[1][cy]++;
    if (!*(const volatile uint64_t *)(uintptr_t)(rc->addr + 16)) g_occc.y0++;
    const rc_occ_t *r = cx <= OCW_LANDED ? ex : cy <= OCW_LANDED ? ey : NULL; const int cr = r == ex ? cx : cy;   // the samples-passed report it reads
    if (!r) return;
    if (cr == OCW_LANDED) { const uint64_t d = g_tex_frame - r->pf; g_occc.age[d > 3 ? 3 : d]++; }
    if (r->f >= cur) g_occc.same++; else g_occc.other++;
    const uint64_t dd = ser - r->ser; g_occc.dd[dd == 0 ? 0 : dd < 16 ? 1 : dd < 128 ? 2 : dd < 557 ? 3 : dd < 2048 ? 4 : 5]++;
    if (dry || r->pe == ~0ull) g_occc.dp[5]++;
    else { const uint64_t dp = g_vk_pends - r->pe; g_occc.dp[dp == 0 ? 0 : dp == 1 ? 1 : dp < 8 ? 2 : dp < 64 ? 3 : 4]++; }
}
void vk_occd_exec_log(double f) {   // gfx_vk_core.c occ_log, every 600 presents
    char cb[2][900]; for (int w = 0; w < 2; w++) { size_t k = 0; cb[w][0] = 0;
        for (int i = 0; i < OCW_N; i++) if (g_occc.cls[w][i]) k += (size_t)snprintf(cb[w] + k, sizeof cb[w] - k, "%s%s %llu", k ? ", " : "", g_ocw_names[i], (unsigned long long)g_occc.cls[w][i]);
        if (!k) snprintf(cb[w], sizeof cb[w], "none"); }
    VKLOG("occ-s0: render conditions (1/3), last %.0f presents: %llu (%.0f a frame; decided by the dry pass %llu); mode 0 (equal) %llu, mode 1 (differ) %llu; y word 0 in memory %llu",
          f, (unsigned long long)g_occc.n, g_occc.n / f, (unsigned long long)g_occc.dry, (unsigned long long)g_occc.mode[0], (unsigned long long)g_occc.mode[1], (unsigned long long)g_occc.y0);
    VKLOG("occ-s0: render conditions (2/3): word x: %.180s; word y: %.180s", cb[0], cb[1]);   // (occ-s3fast: three lines, each under ~400 characters; the class lists cut at 180)
    VKLOG("occ-s0: render conditions (3/3): the samples report read: landed 0/1/2/3+ presents before %llu/%llu/%llu/%llu, same submission %llu, an earlier one %llu; report to condition in draws "
          "0/1-15/16-127/128-556/557-2047/2048+: %llu/%llu/%llu/%llu/%llu/%llu, in pass ends 0/1/2-7/8-63/64+/not known (dry pass): %llu/%llu/%llu/%llu/%llu/%llu",
          (unsigned long long)g_occc.age[0], (unsigned long long)g_occc.age[1], (unsigned long long)g_occc.age[2], (unsigned long long)g_occc.age[3],
          (unsigned long long)g_occc.same, (unsigned long long)g_occc.other, (unsigned long long)g_occc.dd[0], (unsigned long long)g_occc.dd[1], (unsigned long long)g_occc.dd[2],
          (unsigned long long)g_occc.dd[3], (unsigned long long)g_occc.dd[4], (unsigned long long)g_occc.dd[5], (unsigned long long)g_occc.dp[0], (unsigned long long)g_occc.dp[1],
          (unsigned long long)g_occc.dp[2], (unsigned long long)g_occc.dp[3], (unsigned long long)g_occc.dp[4], (unsigned long long)g_occc.dp[5]);
    if (g_occ2_ser_off) VKLOG("[RED] occ-s1: the replay thread's stream serial differed from the dry pass's %llu times since start", (unsigned long long)g_occ2_ser_off);
    memset(&g_occc, 0, sizeof g_occc);
}
// occ-s3 (design\OCCLUSION_REDESIGN.md step 3; gfx_vk_core.c vk_occ3_*): a render condition (mode 0 draw while equal, 1 while different) whose word x (or y) is a samples-passed
// report still in the predicate ring (any age: this submission, in flight or landed; no fence timing) is evaluated by the GPU: the report's resolve (or a resolve forced before the
// condition while its report is still pending: the pass ends there, no CPU wait) writes its predicate (count != y; inverted for mode 0), and the draws and clears after the
// condition are recorded under vkCmdBeginConditionalRenderingEXT, one block per gated draw or clear inside its pass (rcg_begin / rcg_end; ended at any pass end too). The other
// word: never reported (read from guest memory at the condition, as the CPU path reads it: OoT's y = 0, the report's own predicate; another constant: a predicate of its own),
// or a samples-passed report too (both counts compared on the GPU). CPU-written flags (other counter types), timestamps and never-reported words stay CPU-evaluated as before;
// a samples-passed report the GPU does not take (out of the ring, a resolve deferred to a part's join, another shape) is the fallback: its submission waited for when it is in
// flight (vk_occ3_wait, counted "cpu waits"), then the CPU rule as before
typedef struct { uint64_t rx, ry, yc; uint8_t yrep, ymem, inv; } rcg_t;
static struct { uint64_t cond, gpu, flag, ts, mem, fb, gd, gc; } g_o3c;   // occ-s3 census (replay thread; the dry pass's CPU decisions counted where it decides)
static int rcg_rs_ok(const rc_occ_t *e, uint64_t rnow) { return e && !e->timestamp && e->rs && e->rs - 1 < rnow && rnow - (e->rs - 1) <= 64000; }
static int rcg_class(const p_render_cond *rc, const rc_occ_t *ex, const rc_occ_t *ey, uint64_t rnow, rcg_t *g) {   // 1: a GPU condition (g filled; y read from memory now when ymem)
    if (g_occ2 < 3 || rc->mode >= 2) return 0;
    const int xr = rcg_rs_ok(ex, rnow), yr = rcg_rs_ok(ey, rnow);
    memset(g, 0, sizeof *g); g->inv = rc->mode == 0;
    if (xr && yr) { g->rx = ex->rs - 1; g->ry = ey->rs - 1; g->yrep = 1; return 1; }
    if (xr && !ey) { g->rx = ex->rs - 1; g->yc = *(const volatile uint64_t *)(uintptr_t)(rc->addr + 16); g->ymem = 1; return 1; }
    if (yr && !ex) { g->rx = ey->rs - 1; g->yc = *(const volatile uint64_t *)(uintptr_t)rc->addr; g->ymem = 1; return 1; }   // (x != y is symmetric)
    return 0;
}
static void rcg_census(const rc_occ_t *ex, const rc_occ_t *ey) {   // a condition (mode < 2) evaluated on the CPU, by what its words are
    if ((ex && ex->rs) || (ey && ey->rs)) g_o3c.fb++;
    else if ((ex && ex->timestamp) || (ey && ey->timestamp)) g_o3c.ts++;
    else if (ex || ey) g_o3c.flag++;
    else g_o3c.mem++;
}
// occ-s3fast diagnostic (DM_VK_OCC_CONDDIAG=1, live through DM_CTL env lines; off: one test per condition and draw-type record): per render condition address (and class:
// gpu = evaluated by the GPU, ts / flag / mem / samples = evaluated on the CPU by what its words are) the ops under it: draws and clears drawn / gated by the GPU / dropped by
// the CPU, the last gated draw's VS / FS / TE keys, the last target (size, NVN format, address), how many of its ops this frame hit a target that a terrain draw (FS
// c72b6ea746c00433 or VS 1776c0973ce4bbd1, OoT's tessellated terrain) renders into in the same frame (render targets only: sampled textures are not followed); terrain
// draws by the condition over them; the CPU conditions' words read back by the GPU (gfx_vk_core.c vk_o3r_*). Log only
enum { O3G_NA = 64 };
static struct {
    int on; long gen; uint64_t a; int cls;   // the condition the ops are under now (a 0: none)
    struct { uint64_t a, op[2][3], terr, vs, fs, te, tgt; uint32_t tw, th, tf; uint8_t cls; } at[O3G_NA]; uint32_t nat, full;
    uint64_t frame; struct { uint64_t t; int ai; } fl[512]; uint32_t nfl; uint64_t tt[64]; uint32_t ntt;   // this frame: ops' targets under a condition; terrain draws' targets
    uint64_t terr[4];   // terrain draws: under no condition, gated by the GPU, drawn by a CPU decision, dropped by one
} g_o3g;
static int o3g_on(void) {
    if (g_o3g.gen != dm_env_gen) { g_o3g.gen = dm_env_gen; const char *e = getenv("DM_VK_OCC_CONDDIAG"); const int on = g_occ2 >= 2 && e && atoi(e) != 0;
        if (on != g_o3g.on) VKLOG("occ-s3fast: render condition census and readback %s (DM_VK_OCC_CONDDIAG, live; diagnostic)", on ? "on" : "off");
        g_o3g.on = on; if (!on) g_o3g.a = 0; }
    return g_o3g.on;
}
static int o3g_slot(uint64_t a, int cls) {
    uint32_t h = (uint32_t)((a >> 4) * 0x9E3779B1u) & (O3G_NA - 1);
    for (uint32_t k = 0; k < O3G_NA; k++, h = (h + 1) & (O3G_NA - 1)) { if (g_o3g.at[h].a == a) return (int)h;
        if (!g_o3g.at[h].a) { if (g_o3g.nat * 4 >= O3G_NA * 3) break; memset(&g_o3g.at[h], 0, sizeof g_o3g.at[h]); g_o3g.at[h].a = a; g_o3g.at[h].cls = (uint8_t)cls; g_o3g.nat++; return (int)h; } }
    g_o3g.full++; return -1;
}
static void o3g_frame(void) {   // a new frame: last frame's ops on terrain targets counted
    extern uint64_t g_tex_frame;
    for (uint32_t i = 0; i < g_o3g.nfl; i++) for (uint32_t j = 0; j < g_o3g.ntt; j++) if (g_o3g.tt[j] == g_o3g.fl[i].t) { g_o3g.at[g_o3g.fl[i].ai].terr++; break; }
    g_o3g.nfl = g_o3g.ntt = 0; g_o3g.frame = g_tex_frame;
}
static uint64_t o3g_taddr(const tex_t *t) { return t->mem ? (uint64_t)(uintptr_t)t->mem : (uint64_t)(uintptr_t)t->pool + (uint64_t)t->off; }
static void o3g_cond(const p_render_cond *rc) {   // exec_ops, a render condition evaluated (replay thread)
    if (!o3g_on()) return;
    if (rc->mode >= 2) { g_o3g.a = 0; return; }
    const rc_occ_t *ex = rc_occ_find(rc->addr), *ey = rc_occ_find(rc->addr + 16);
    const int cls = g_rec->X.rcg_on ? 0 : (ex && ex->timestamp) || (ey && ey->timestamp) ? 1 : (ex && ex->kind == 3) || (ey && ey->kind == 3) ? 2 : (ex && ex->kind == 2) || (ey && ey->kind == 2) ? 4 : 3;
    g_o3g.a = rc->addr; g_o3g.cls = cls;
    if (cls) { void vk_o3r_cond(uint64_t, int, int, int), vk_o3r_flush(void); vk_o3r_cond(rc->addr, rc->mode, g_rc_skip, cls); if (S.pass_open != 1) vk_o3r_flush(); }
}
static void o3g_op(int op, const uint8_t *a) {   // exec_ops, before a draw-type record or a clear (replay thread)
    extern uint64_t g_tex_frame;
    if (g_o3g.frame != g_tex_frame) o3g_frame();
    const int clear = op == OP_CLEAR_COLOR || op == OP_CLEAR_DS, outc = g_rc_skip ? 2 : g_rec->X.rcg_on ? 1 : 0;   // drawn, gated by the GPU, dropped
    const tex_t *t = NULL;
    if (op == OP_CLEAR_COLOR) { const int i = ((const p_clear_color *)a)->index; if (i >= 0 && i < 8 && S.rt_has_col[i]) t = &S.rt_col[i]; }
    else if (op == OP_CLEAR_DS) { if (S.rt_has_dep) t = &S.rt_dep; }
    else if (S.rt_has_col[0]) t = &S.rt_col[0]; else if (S.rt_has_dep) t = &S.rt_dep;
    const uint64_t vs = !clear && S.sh[0] ? S.sh[0]->key : 0, fs = !clear && S.sh[1] ? S.sh[1]->key : 0;
    if (fs == 0xc72b6ea746c00433ull || vs == 0x1776c0973ce4bbd1ull) {
        g_o3g.terr[!g_o3g.a ? 0 : outc == 1 ? 1 : outc == 2 ? 3 : 2]++;
        for (int i = 0; i < S.rt_n && i < 8; i++) if (S.rt_has_col[i] && g_o3g.ntt < 64) g_o3g.tt[g_o3g.ntt++] = o3g_taddr(&S.rt_col[i]);
        if (S.rt_has_dep && g_o3g.ntt < 64) g_o3g.tt[g_o3g.ntt++] = o3g_taddr(&S.rt_dep); }
    if (!g_o3g.a) return;
    const int k = o3g_slot(g_o3g.a, g_o3g.cls); if (k < 0) return;
    g_o3g.at[k].op[clear][outc]++;
    if (!clear && outc) { g_o3g.at[k].vs = vs; g_o3g.at[k].fs = fs; g_o3g.at[k].te = S.sh[4] ? S.sh[4]->key : 0; }
    if (t) { const uint64_t ta = o3g_taddr(t); g_o3g.at[k].tgt = ta; g_o3g.at[k].tw = t->d.width; g_o3g.at[k].th = t->d.height; g_o3g.at[k].tf = (uint32_t)t->d.format;
        if (g_o3g.nfl < 512) { g_o3g.fl[g_o3g.nfl].t = ta; g_o3g.fl[g_o3g.nfl].ai = k; g_o3g.nfl++; } }
}
static void o3g_log(double f) {   // vk_o3p_log, every 600 presents: terrain, then the 6 busiest CPU-condition addresses and the 4 busiest GPU ones, a line each
    static const char *const cn[] = {"gpu", "ts", "flag", "mem", "samples"};
    if (!g_o3g.on) return;
    VKLOG("occ-s3fast terrain draws (FS c72b6ea746c00433 / VS 1776c0973ce4bbd1) a frame: under no condition %.1f, gated by the GPU %.2f, drawn by a CPU decision %.2f, dropped by one %.2f; table full %u",
          g_o3g.terr[0] / f, g_o3g.terr[1] / f, g_o3g.terr[2] / f, g_o3g.terr[3] / f, g_o3g.full);
    int used[10], nu = 0;
    for (int pass = 0; pass < 2; pass++) for (int t = 0; t < (pass ? 4 : 6); t++) { int b = -1; uint64_t bn = 0;
        for (int i = 0; i < O3G_NA; i++) { if (!g_o3g.at[i].a || (pass ? g_o3g.at[i].cls != 0 : g_o3g.at[i].cls == 0)) continue; int u = 0; for (int j = 0; j < nu; j++) u |= used[j] == i; if (u) continue;
            uint64_t n = 0; for (int c = 0; c < 2; c++) for (int o = 0; o < 3; o++) n += g_o3g.at[i].op[c][o]; if (n > bn) { bn = n; b = i; } }
        if (b < 0) break; used[nu++] = b; const __typeof__(g_o3g.at[0]) *s = &g_o3g.at[b];
        VKLOG("occ-s3fast gate %#llx %s: draws drawn/gated/dropped %llu/%llu/%llu, clears %llu/%llu/%llu, ops on a terrain target %llu; gated draw VS %016llx FS %016llx TE %016llx; target %ux%u fmt %u @%#llx",
              (unsigned long long)s->a, cn[s->cls < 5 ? s->cls : 4], (unsigned long long)s->op[0][0], (unsigned long long)s->op[0][1], (unsigned long long)s->op[0][2], (unsigned long long)s->op[1][0],
              (unsigned long long)s->op[1][1], (unsigned long long)s->op[1][2], (unsigned long long)s->terr, (unsigned long long)s->vs, (unsigned long long)s->fs, (unsigned long long)s->te,
              s->tw, s->th, s->tf, (unsigned long long)s->tgt); }
    memset(g_o3g.at, 0, sizeof g_o3g.at); g_o3g.nat = 0; g_o3g.full = 0; memset(g_o3g.terr, 0, sizeof g_o3g.terr);
}
// occ-s3fast (diagnostic, log only): a CPU-evaluated condition on a timestamp (1) or another counter's flag word (2), its outcome (gfx_vk_core.c vk_o3f_cond_note: flips by spikes)
static void o3f_note(const p_render_cond *rc, const rc_occ_t *ex, const rc_occ_t *ey, int skip) {
    if (g_occ2 < 2 || g_wk || rc->mode >= 2) return;
    const int k = (ex && ex->timestamp) || (ey && ey->timestamp) ? 1 : (ex && ex->kind == 3) || (ey && ey->kind == 3) ? 2 : 0;
    if (k) vk_o3f_cond_note(rc->addr, k, skip);
}
static int rcg_apply(const rcg_t *g) {   // replay thread: 1 = the draws after it gated by its GPU predicate (g_rc_skip 0); 0 = not taken
    vk_occ3_q q = {g->rx, g->ry, g->yc, g->yrep}; uint32_t off = 0;
    if (g_wk || !vk_occ3_pred(&q, &off)) return 0;
    g_rec->X.rcg_on = 1; g_rec->X.rcg_off = off; g_rec->X.rcg_inv = g->inv; g_rc_skip = 0; g_o3c.gpu++;
    return 1;
}
static int rcg_try(const p_render_cond *rc, const rc_occ_t *ex, const rc_occ_t *ey) {   // occ-s3, a condition evaluated at the replay: 1 = on the GPU
    rcg_t g; g_o3c.cond++;
    if (rcg_class(rc, ex, ey, vk_occ2_rseq(), &g) && rcg_apply(&g)) return 1;
    rcg_census(ex, ey);
    const rc_occ_t *r = ex && ex->rs ? ex : ey && ey->rs ? ey : NULL;
    if (r) (void)vk_occ3_wait(r->rs - 1);   // the CPU's last resort: the report's submission waited for when it is in flight (the CPU rule then reads its landed word)
    return 0;
}
static void rcg_end(void) { if (!g_rec->X.rcg_cb) return; vk_occ3_end(g_rec->X.rcg_cb); g_rec->X.rcg_cb = NULL; }   // (at every pass end: never across one)
// occ-s3fast: right before a gated draw (clear 0) or clear (1), inside its pass. The open block is kept when its predicate is this one (as Eden keeps hcr_setup running while
// buffer and offset stay the same, vk_query_cache.cpp HostConditionalRenderingCompareValueImpl, and pauses it at a render pass end, vk_scheduler.cpp NotifySegment(false));
// a block is open only while rcg_on (rcg_settle ends it when a condition leaves the draws ungated), so whatever draw or clear is recorded inside one is a gated one
static void rcg_begin(VkCommandBuffer cb, int clear) {
    if (!g_rec->X.rcg_on || S.pass_open != 1) return;
    if (clear) g_o3c.gc++; else g_o3c.gd++;
    if (g_rec->X.rcg_cb) { if (g_rec->X.rcg_cb == cb && g_rec->X.rcg_boff == g_rec->X.rcg_off && g_rec->X.rcg_binv == g_rec->X.rcg_inv) return; rcg_end(); }
    vk_occ3_begin(cb, g_rec->X.rcg_off, g_rec->X.rcg_inv); g_rec->X.rcg_cb = cb; g_rec->X.rcg_boff = g_rec->X.rcg_off; g_rec->X.rcg_binv = g_rec->X.rcg_inv;
}
// occ-s3fast: after a render condition (every path: replay, the dry pass's decision, the queue switch): a block whose predicate no longer gates the draws ends here, inside
// the pass, before the next draw (an ungated draw is never inside a block)
static void rcg_settle(void) { if (g_rec->X.rcg_cb && (!g_rec->X.rcg_on || g_rec->X.rcg_boff != g_rec->X.rcg_off || g_rec->X.rcg_binv != g_rec->X.rcg_inv)) rcg_end(); }
void vk_occ3_exec_log(double f, const uint64_t *c) {   // gfx_vk_core.c occ2_log, every 600 presents (c: forced resolves, deferred, out of the ring, cpu waits, their ns, blocks begun)
    VKLOG("occ-s3 (1/2) render conditions %.1f a frame: on the GPU %.1f (forced resolves %.2f, not taken: resolve deferred to a part join %.2f, report out of the predicate ring %.2f), "
          "CPU flag (another counter) %.1f, CPU timestamp %.1f, CPU memory (never reported) %.1f",
          g_o3c.cond / f, g_o3c.gpu / f, c[0] / f, c[1] / f, c[2] / f, g_o3c.flag / f, g_o3c.ts / f, g_o3c.mem / f);   // (occ-s3fast: two lines, each under ~400 characters)
    VKLOG("occ-s3 (2/2) fallback (a samples-passed report the GPU did not take) %.2f a frame (cpu waits %.2f, %.2f ms; should be ~0); conditional rendering blocks %.1f a frame "
          "(occ-s3fast: one per run of gated draws / clears with one predicate in one pass) for gated draws %.0f, gated clears %.1f",
          g_o3c.fb / f, c[3] / f, c[4] / 1e6 / f, c[5] / f, g_o3c.gd / f, g_o3c.gc / f);
    memset(&g_o3c, 0, sizeof g_o3c);
}
// S1d: with the report table's entries taken where the record is in the stream (the dry pass's copy, for a condition it left to the replay) instead of the table's now
static void render_cond_e(const p_render_cond *rc, const rc_occ_t *ex, const rc_occ_t *ey) {
    g_rec->X.rcg_on = 0;   // occ-s3: a new condition ends the GPU one's gating
    if (rc->mode >= 2) { g_rc_skip = rc->mode == 3; return; }
    if (g_occdiag) occd_cond(rc, ex, ey, RC_FD_LIVE, g_oser, 0);   // occ-s0
    if (g_occ2 >= 3 && !g_wk && rcg_try(rc, ex, ey)) { render_cond_note(rc, 0, 0, 0, 0); return; }   // occ-s3: on the GPU
    uint64_t x, y; int known; g_rc_skip = rc_eval(rc, ex, ey, RC_FD_LIVE, 1, &x, &y, &known);
    render_cond_note(rc, known, g_rc_skip, x, y); o3f_note(rc, ex, ey, g_rc_skip);   // (occ-s3fast diagnostic)
}
static void render_cond(const p_render_cond *rc) {
    g_rec->X.rcg_on = 0;   // occ-s3: a new condition ends the GPU one's gating
    if (rc->mode >= 2) { g_rc_skip = rc->mode == 3; return; }
    if (g_occdiag && !g_wk) occd_cond(rc, rc_occ_find(rc->addr), rc_occ_find(rc->addr + 16), RC_FD_LIVE, g_oser, 0);   // occ-s0
    if (g_occ2 >= 3 && !g_wk && rcg_try(rc, rc_occ_find(rc->addr), rc_occ_find(rc->addr + 16))) { render_cond_note(rc, 0, 0, 0, 0); return; }   // occ-s3: on the GPU
    const rc_occ_t *ex = rc_occ_find(rc->addr), *ey = rc_occ_find(rc->addr + 16);
    uint64_t x, y; int known; g_rc_skip = rc_eval(rc, ex, ey, RC_FD_LIVE, 1, &x, &y, &known);
    render_cond_note(rc, known, g_rc_skip, x, y); o3f_note(rc, ex, ey, g_rc_skip);   // (occ-s3fast diagnostic)
}
static void render_cond_note(const p_render_cond *rc, int known, int skip, uint64_t x, uint64_t y) {   // the diagnostics and census of an evaluated condition (were render_cond's)
    if (rcd_on()) { rcd_note(rc->addr, known, skip, x, y); rcd_report(); void vk_pool_kind_note(uint64_t, int); vk_pool_kind_note(rc->addr, 1); }   // bak88, bak89
    g_rc_n[1] += (uint64_t)skip;
    if (!(++g_rc_n[0] & 0xfff)) VKLOG("render_cond: %llu conditions, %llu false; occlusion values %llu real, %llu this submission, %llu in flight, %llu not measured%s; %llu draws and clears dropped",
        (unsigned long long)g_rc_n[0], (unsigned long long)g_rc_n[1], (unsigned long long)g_rc_n[2], (unsigned long long)g_rc_n[3], (unsigned long long)g_rc_n[4],
        (unsigned long long)g_rc_n[5], g_rc_real ? "" : " (DM_VK_RC_OCC=0)", (unsigned long long)g_rc_n[6]);
}

static void s61_log(void) {   // s61: every 2 s, at a submission's start
    uint64_t t = tnow(); g_s61_subs++; if (!g_s61_t0) { g_s61_t0 = t; return; } if (t - g_s61_t0 < 2000000000ull) return;
    static uint64_t sent0, ns0; double n = g_s61_subs ? (double)g_s61_subs : 1;
    VKLOG("s61: last %.1f s: %llu submissions; parts sent early %llu (%.2f a submission, their submits %.3f ms a submission); ordinary cuts %llu (%.0f draws before each), "
          "past the cap of %d %llu (DM_VK_CB_DRAWS %d)", (t - g_s61_t0) / 1e9, (unsigned long long)g_s61_subs,
          (unsigned long long)(g_s61_sent - sent0), (g_s61_sent - sent0) / n, (g_s61_sub_ns - ns0) / 1e6 / n, (unsigned long long)g_s61_cuts,
          g_s61_cuts ? (double)g_s61_cutdraws / (double)g_s61_cuts : 0.0, g_s61_maxcut, (unsigned long long)g_s61_cap, g_x_cbdraws);
    { char b[600]; int o = 0; uint64_t c[64]; memcpy(c, g_s64_end, sizeof c);   // s64a: the ops that ended passes, most first
      for (int k = 0; k < 10 && o < (int)sizeof b - 40; k++) { int m = -1; for (int i = 0; i < 64; i++) if (c[i] && (m < 0 || c[i] > c[m])) m = i; if (m < 0) break;
          o += snprintf(b + o, sizeof b - (size_t)o, "%s%s %.1f", k ? ", " : "", m == 63 ? "end" : g_op_names[m] ? g_op_names[m] : "?", c[m] / n); c[m] = 0; }
      char fb[400]; int p = 0; for (int i = 0; i < 32 && p < (int)sizeof fb - 24; i++) if (g_s64_bit[i]) p += snprintf(fb + p, sizeof fb - (size_t)p, "%s0x%x %.1f", p ? ", " : "", 1u << i, g_s64_bit[i] / n);
      VKLOG("s64a: per submission: passes %.1f, ended by %s; OP_BARRIER %.1f (%.1f with a pass open), flag bits %s; draws per pass 0:%.1f 1:%.1f 2-3:%.1f 4-7:%.1f 8-15:%.1f 16-63:%.1f 64+:%.1f; "
            "passes right after a barrier-ended pass %.1f, of them sampling one of their own attachments %.1f",
            g_s64_passes / n, o ? b : "-", g_s64_bar / n, g_s64_bar_open / n, p ? fb : "-", g_s64_dpp[0] / n, g_s64_dpp[1] / n, g_s64_dpp[2] / n, g_s64_dpp[3] / n, g_s64_dpp[4] / n,
            g_s64_dpp[5] / n, g_s64_dpp[6] / n, g_s64_ab_n / n, g_s64_ab_fb / n);
      VKLOG("s65: per submission: rd-only passes %.1f with %.1f draws (render area cut %.1f), passes ended where rasterizer discard toggles %.1f (DM_VK_S65=%d, live)", g_s65_rdp / n, g_s65_rddraws / n, g_s65_cut / n, g_s65_rdsplit / n, g_s65_on);
      g_s65_rdp = g_s65_rddraws = g_s65_cut = g_s65_rdsplit = 0;
      VKLOG("NVN redundant full barriers avoided (total): %llu", (unsigned long long)g_repeated_barriers);
      VKLOG("s64: per submission: barriers kept pending %.1f, passes kept %.1f, ended by feedback %.1f, by another op %.1f (DM_VK_S64=%d, live)", g_s64_npend / n, g_s64_kept / n, g_s64_fbend / n, g_s64_opend / n, g_s64_on);
      g_s64_npend = g_s64_kept = g_s64_fbend = g_s64_opend = 0;
      memset(g_s64_end, 0, sizeof g_s64_end); memset(g_s64_bit, 0, sizeof g_s64_bit); memset(g_s64_dpp, 0, sizeof g_s64_dpp); g_s64_passes = g_s64_bar = g_s64_bar_open = g_s64_ab_n = g_s64_ab_fb = 0; }
    sent0 = g_s61_sent; ns0 = g_s61_sub_ns; g_s61_subs = g_s61_cuts = g_s61_cutdraws = g_s61_cap = 0; g_s61_t0 = t;
}
// Observe the first native consumer of each downsample in a submission. A scaled
// sampler does not consume the native image. Never use these observations to
// discard a future command: descriptor bindings and aliasing can change.
enum wb_use { WB_SAMPLE, WB_STORAGE, WB_ATTACHMENT, WB_COPY, WB_OTHER, WB_COVERED, WB_PARTIAL, WB_BOUNDARY, WB_KINDS };
static int wb_trace;
static struct wb_record {
    uint64_t id;
    VkRect2D rect;
    VkImageAspectFlags aspects;
    VkCommandBuffer cb;
} wb_live[128];
static unsigned wb_count;
static uint64_t scale_deferred, scale_elided;
static uint64_t wb_uses[WB_KINDS], wb_overflow, wb_scaled, wb_seen, wb_same_cb;
static int wb_index(const vkimg_t *im) {
    if (im) for (unsigned i = 0; i < wb_count; ++i)
        if (wb_live[i].id == im->residency_id) return (int)i;
    return -1;
}
static void wb_retire(int i, enum wb_use why) {
    if (i < 0) return;
    ++wb_uses[why];
    wb_live[i] = wb_live[--wb_count];
}
static void wb_use(const vkimg_t *im, enum wb_use why) {
    if (wb_trace && wb_count) wb_retire(wb_index(im), why);
}
static void materialize_scale(vkimg_t *overwritten);
void vk_before_native_readback(const vkimg_t *im) { materialize_scale(NULL); wb_use(im, WB_COPY); }
static void wb_descriptor(const vkimg_t *im, int scaled, int storage) {
    if (!wb_trace || !wb_count) return;
    int i = wb_index(im);
    if (i < 0) return;
    if (scaled) ++wb_scaled;
    else wb_retire(i, storage ? WB_STORAGE : WB_SAMPLE);
}
static void wb_blit(vkimg_t *im, VkImageAspectFlags aspects, VkRect2D r) {
    if (!wb_trace) return;
    int i = wb_index(im);
    if (i >= 0) {
        const struct wb_record *old = &wb_live[i];
        VkRect2D a = old->rect;
        int covered = !(old->aspects & ~aspects) && r.offset.x <= a.offset.x && r.offset.y <= a.offset.y &&
            r.offset.x + (int)r.extent.width >= a.offset.x + (int)a.extent.width &&
            r.offset.y + (int)r.extent.height >= a.offset.y + (int)a.extent.height;
        if (covered && old->cb == vk_cmd()) ++wb_same_cb;
        wb_retire(i, covered ? WB_COVERED : WB_PARTIAL);
    }
    ++wb_seen;
    if (wb_count == sizeof wb_live / sizeof wb_live[0]) { ++wb_overflow; return; }
    wb_live[wb_count++] = (struct wb_record){im->residency_id, r, aspects, vk_cmd()};
}
static void wb_submission(void) {
    static int initialized;
    if (!initialized) {
        const char *e = getenv("DM_VK_TRACE_WRITEBACK");
        wb_trace = e && atoi(e);
        initialized = 1;
    }
    if (!wb_trace) return;
    wb_uses[WB_BOUNDARY] += wb_count;
    wb_count = 0;
    static uint64_t last;
    if (g_tex_frame - last < 600) return;
    VKLOG("2X lazy writebacks: deferred %llu, replaced by full overwrite %llu",
          (unsigned long long)scale_deferred, (unsigned long long)scale_elided);
    scale_deferred = scale_elided = 0;
    extern uint64_t g_nvn_discards[2];
    VKLOG("NVN discard calls since launch: color %llu depth/stencil %llu",
          (unsigned long long)__atomic_load_n(&g_nvn_discards[0], __ATOMIC_RELAXED),
          (unsigned long long)__atomic_load_n(&g_nvn_discards[1], __ATOMIC_RELAXED));
    VKLOG("native 2X first use frames %llu..%llu: blits %llu, sampled %llu, storage %llu, native attachment %llu, copied %llu, other/unknown %llu, fully overwritten %llu (same command buffer %llu), partial overwrite %llu, submission boundary %llu, overflow %llu, scaled bindings %llu",
          (unsigned long long)last, (unsigned long long)g_tex_frame, (unsigned long long)wb_seen,
          (unsigned long long)wb_uses[WB_SAMPLE], (unsigned long long)wb_uses[WB_STORAGE],
          (unsigned long long)wb_uses[WB_ATTACHMENT], (unsigned long long)wb_uses[WB_COPY], (unsigned long long)wb_uses[WB_OTHER],
          (unsigned long long)wb_uses[WB_COVERED], (unsigned long long)wb_same_cb, (unsigned long long)wb_uses[WB_PARTIAL],
          (unsigned long long)wb_uses[WB_BOUNDARY], (unsigned long long)wb_overflow, (unsigned long long)wb_scaled);
    memset(wb_uses, 0, sizeof wb_uses);
    wb_seen = wb_same_cb = wb_overflow = wb_scaled = 0;
    last = g_tex_frame;
}
void vk_exec_reset(void) {   // a new submission: NVN state does not carry over between command buffers, the replay starts from the defaults
    wb_submission();
    s61_log();   // s61 (the counters of the submission before)
    g_s64_pend = 0; g_s64_pwrite = 0;   // s64: a new submission has no pass
    static int inited; if (!inited) { inited = 1; state_reset(); prewarm_start(); }
    // Conditional rendering is guest queue state across physical submissions.
    S.pass_open = 0; S.bound_pipe = VK_NULL_HANDLE; g_cb_draws = 0; g_ord_parts = 0; g_psplit_now = 0; g_set_epoch++; x_read();
}

// ---------------- render passes ----------------
// A pass is begun lazily: begin_pass() only sets it up (pass_open 2), and the first command that must run inside it begins it (pass_commit, pass_open 1).
// Until then a clear of a whole attachment becomes that attachment's load op, so a tiler clears in tile memory instead of loading the
// attachment from memory and clearing it with a draw; a pass that is set up and never used is dropped.
// R1b (parallel recording): P_ca, P_da, P_sa, P_ri, P_cleared in struct rec_ctx (top of the file)
// Render passes. Adreno's driver spent about 40 µs in each vkCmdBeginRendering (a quarter of the replay thread in Xenoblade Chronicles 3's battles, 128 passes a
// frame); a VkRenderPass is made once per set of formats and load operations, and an imageless framebuffer once per set of attachment descriptions, and passes
// then begin with vkCmdBeginRenderPass (the battle's replay went from about 20 to 17.5 ms a frame). Pipelines are made against a compatible render pass (the same
// formats, every load operation LOAD). DM_VK_RP=0 (start-up): dynamic rendering
typedef struct { uint32_t ncol, cfmt[8], cload[8], dfmt, dload, sload, dstore, samples; } rpkey;
typedef struct { VkRenderPass rp; uint32_t w, h, layers, n; struct { uint32_t usage, flags, w, h, layers, fmt, fmt2; } a[9]; } fbkey;   // fmt2: a format-listed image's second format (an imageless framebuffer names the image's whole list)
static nvn_hmap g_rps = {.klen = sizeof(rpkey)}, g_fbs = {.klen = sizeof(fbkey)}; static pthread_mutex_t g_rp_mu = PTHREAD_MUTEX_INITIALIZER;
// Without dynamic rendering (a Vulkan 1.1 / 1.2 device, V.has_dynrender) passes must be render pass objects, whatever DM_VK_RP says (vk_init refuses a device with neither)
static int rp_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_RP"); on = V.has_imageless && (!V.has_dynrender || !(e && !atoi(e))); VKLOG("passes: %s", on ? "render pass objects with imageless framebuffers" : "dynamic rendering"); } return on; }
static int fmt_has_depth(uint32_t f) { return f && f != VK_FORMAT_S8_UINT; }
static int fmt_has_stencil(uint32_t f) { return f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_S8_UINT || f == VK_FORMAT_D16_UNORM_S8_UINT; }
static VkRenderPass rp_get(const rpkey *k) {
    pthread_mutex_lock(&g_rp_mu); VkRenderPass rp = nvn_hm_get(&g_rps, k);
    if (!rp) {
        VkAttachmentDescription2 ad[9]; VkAttachmentReference2 cr[8], dr; uint32_t na = 0;
        for (uint32_t i = 0; i < k->ncol; i++) {
            if (!k->cfmt[i]) { cr[i] = (VkAttachmentReference2){VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, NULL, VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT}; continue; }
            ad[na] = (VkAttachmentDescription2){VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2, NULL, 0, (VkFormat)k->cfmt[i], (VkSampleCountFlagBits)k->samples, (VkAttachmentLoadOp)k->cload[i], VK_ATTACHMENT_STORE_OP_STORE,
                VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL};
            cr[i] = (VkAttachmentReference2){VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, NULL, na++, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT};
        }
        if (k->dfmt) {
            int hd = fmt_has_depth(k->dfmt), hs = fmt_has_stencil(k->dfmt);
            ad[na] = (VkAttachmentDescription2){VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2, NULL, 0, (VkFormat)k->dfmt, (VkSampleCountFlagBits)k->samples,
                hd ? (VkAttachmentLoadOp)k->dload : VK_ATTACHMENT_LOAD_OP_DONT_CARE, hd ? (VkAttachmentStoreOp)k->dstore : VK_ATTACHMENT_STORE_OP_DONT_CARE,
                hs ? (VkAttachmentLoadOp)k->sload : VK_ATTACHMENT_LOAD_OP_DONT_CARE, hs ? (VkAttachmentStoreOp)k->dstore : VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL};
            dr = (VkAttachmentReference2){VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, NULL, na++, VK_IMAGE_LAYOUT_GENERAL, (hd ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (hs ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)};
        }
        VkSubpassDescription2 sd = {VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2, NULL, 0, VK_PIPELINE_BIND_POINT_GRAPHICS, 0, 0, NULL, k->ncol, cr, NULL, k->dfmt ? &dr : NULL, 0, NULL};
        // Match the self-dependency used by Eden's RenderPassCache.
        VkSubpassDependency2 dep = {VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2, NULL, 0, 0,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0};
        VkRenderPassCreateInfo2 ci = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2, NULL, 0, na, ad, 1, &sd, 1, &dep, 0, NULL};
        if (vkCreateRenderPass2(V.dev, &ci, NULL, &rp)) { VKRED_ONCE(70, "vkCreateRenderPass2 failed"); rp = VK_NULL_HANDLE; } else nvn_hm_put(&g_rps, k, rp);
    }
    pthread_mutex_unlock(&g_rp_mu); return rp;
}
static VkRenderPass rp_compat(uint32_t ncol, const uint32_t *cfmt, uint32_t dfmt, uint32_t samples) {   // for pipelines: every load operation LOAD
    rpkey k; memset(&k, 0, sizeof k); k.ncol = ncol; for (uint32_t i = 0; i < ncol && i < 8; i++) { k.cfmt[i] = cfmt[i]; k.cload[i] = VK_ATTACHMENT_LOAD_OP_LOAD; }
    k.dfmt = dfmt; k.dload = k.sload = VK_ATTACHMENT_LOAD_OP_LOAD; k.dstore = VK_ATTACHMENT_STORE_OP_STORE; k.samples = samples ? samples : 1; return rp_get(&k);
}
static VkFramebuffer fb_get(const fbkey *k) {
    pthread_mutex_lock(&g_rp_mu); VkFramebuffer fb = nvn_hm_get(&g_fbs, k);
    if (!fb) {
        VkFramebufferAttachmentImageInfo ai[9]; VkFormat vf[9][2];
        for (uint32_t i = 0; i < k->n; i++) { vf[i][0] = (VkFormat)k->a[i].fmt; vf[i][1] = (VkFormat)k->a[i].fmt2;
            ai[i] = (VkFramebufferAttachmentImageInfo){VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENT_IMAGE_INFO, NULL, k->a[i].flags, k->a[i].usage, k->a[i].w, k->a[i].h, k->a[i].layers, k->a[i].fmt2 ? 2u : 1u, vf[i]}; }
        VkFramebufferAttachmentsCreateInfo aci = {VK_STRUCTURE_TYPE_FRAMEBUFFER_ATTACHMENTS_CREATE_INFO, NULL, k->n, ai};
        VkFramebufferCreateInfo ci = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, &aci, VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT, k->rp, k->n, NULL, k->w, k->h, k->layers};
        if (vkCreateFramebuffer(V.dev, &ci, NULL, &fb)) { VKRED_ONCE(71, "vkCreateFramebuffer failed"); fb = VK_NULL_HANDLE; } else nvn_hm_put(&g_fbs, k, fb);
    }
    pthread_mutex_unlock(&g_rp_mu); return fb;
}
// the attachments begin_pass found, for the render pass begin: each colour slot's (and the depth's) image, view, view format and layers, size at the view's level
// R1b (parallel recording): P_att[9] in struct rec_ctx (top of the file)

// r89: a pass begun by a draw whose later draws keep its viewport 0 and scissor (the records looked at up to where the pass ends) gets the area they can touch as its render
// area instead of the whole target: Hyrule Warriors: Age of Imprisonment draws its dynamic resolution (784x441 to 1120x630) inside 1600x900 targets, and its one-draw
// passes took 0.3-1.3 ms of GPU each (DM_VK_PASS_TIMES); a tiled pass loads and stores its whole area. Not for a pass that clears in its load ops (NVN clears the whole
// target) or has a clear, DrawTexture, swizzle change or other viewport or scissor later. The viewport is widened by 32 pixels for points and wide lines
// (the scissor still bounds it). DM_VK_PASS_AREA=0: the whole target, as before
static _Thread_local const uint8_t *g_xo_next; static _Thread_local uint32_t g_xo_op;   // (S1c (parallel recording): per thread: each recording thread's exec_ops) the record after the one replayed now, and its op (exec_ops)
static uint64_t g_parea_n, g_parea_kept;
static int rt_same(const p_set_rt *r); static VkRect2D scissor_rect(void);
static int parea_scan(void) {   // 1: no record up to the pass's end changes viewport 0 or the scissor
    const uint8_t *q = g_xo_next; if (!q) return 0;
    for (int i = 0; i < 4096; i++) {
        op_hdr h; memcpy(&h, q, sizeof h);
        if (h.op >= OP_COUNT_ || h.size < 8 || (h.size & 7) || h.size > (1u << 20)) return 0;
        const uint8_t *a = q + 8;
        switch (h.op) {
        case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); if (!dm_nvn_jump_ok(t)) return 0; q = t; continue; }   // jumpchk
        case OP_END: case OP_CALL: case OP_CLEAR_COLOR: case OP_CLEAR_DS: case OP_DRAW_TEX: case OP_VP_SWIZZLE: return 0;
        case OP_VIEWPORTS: { const p_viewports *v = (const p_viewports *)a; if (v->first <= 0 && v->first + v->n > 0 && memcmp(v->v[-v->first], S.vp[0], 16)) return 0; break; }
        case OP_SCISSOR: if (!S.sc_set || memcmp(a, &S.sc, sizeof S.sc)) return 0; break;
        case OP_SET_RT: if (!rt_same((const p_set_rt *)a)) return 1; break;
        case OP_BARRIER: case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T: case OP_CLEAR_TEX: return 1;   // (these always end the pass)
        default: break;
        }
        q += h.size;
    }
    return 0;
}
// s65: 1 when nothing up to the end of the rasterizer-discard pass being begun would draw into its attachments (a clear or DrawTexture); the scan
// stops, safe, at a record that ends the pass (rasterizer discard set 0, SET_RT, BARRIER, copies, dispatch, the list's end) and unsafe at anything it cannot follow
static int s65_scan(void) {
    const uint8_t *q = g_xo_next; if (!q) return 0;
    for (int i = 0; i < 4096; i++) {
        op_hdr h; memcpy(&h, q, sizeof h);
        if (h.op >= OP_COUNT_ || h.size < 8 || (h.size & 7) || h.size > (1u << 20)) return 0;
        const uint8_t *a = q + 8;
        switch (h.op) {
        case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); if (!dm_nvn_jump_ok(t)) return 0; q = t; continue; }   // jumpchk
        case OP_CALL: case OP_CLEAR_COLOR: case OP_CLEAR_DS: case OP_DRAW_TEX: return 0;
        case OP_END: case OP_SET_RT: case OP_BARRIER: case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T: case OP_CLEAR_TEX: case OP_CLEAR_BUF: case OP_COPY_B2B: case OP_DISPATCH: return 1;
        case OP_RAST_DISCARD: if (!((const p_rast_discard *)a)->on) return 1; break;
        default: break;
        }
        q += h.size;
    }
    return 0;
}
static void pass_area(void) {
    if (S.pass_open == 2 && S.rast_discard && !g_x_rdkeep && (g_xo_op == OP_DRAW || g_xo_op == OP_DRAW_IDX || g_xo_op == OP_MULTI_DRAW)) {   // s65
        g_s65_rdpass = 1;
        int clr = 0; for (uint32_t i = 0; i < P_ri.colorAttachmentCount; i++) clr |= P_ca[i].imageView && P_ca[i].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;
        clr |= (P_ri.pDepthAttachment && P_da.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) || (P_ri.pStencilAttachment && P_sa.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
        if (g_s65_on && !clr && s65_scan()) { P_ri.renderArea = (VkRect2D){{0, 0}, {S.pass_w < 32 ? S.pass_w : 32, S.pass_h < 32 ? S.pass_h : 32}}; g_s65_cut++; return; }   // (rasterizer-discard passes: never a worker's)
    }
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_PASS_AREA"); on = !e || atoi(e) != 0; }
    if (!on || S.pass_open != 2 || (g_xo_op != OP_DRAW && g_xo_op != OP_DRAW_IDX && g_xo_op != OP_MULTI_DRAW)) return;
    for (uint32_t i = 0; i < P_ri.colorAttachmentCount; i++) if (P_ca[i].imageView && P_ca[i].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) return;
    if ((P_ri.pDepthAttachment && P_da.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) || (P_ri.pStencilAttachment && P_sa.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)) return;
    double x = S.vp[0][0], y = S.vp[0][1], w = S.vp[0][2], h = S.vp[0][3];
    if (w == 0 && h == 0) return;
    if (g_lower_left && !g_flipy) { y = (double)S.pass_h - y; h = -h; } else if (!g_lower_left && g_flipy) { y = y + h; h = -h; }   // as set_dynamic
    { uint32_t sw; memcpy(&sw, S.vp_sw[0], 4); if (sw == 0x100) { y += h; h = -h; } else if (sw) return; }
    double x0 = floor(fmin(x, x + w)) - 32, x1 = ceil(fmax(x, x + w)) + 32, y0 = floor(fmin(y, y + h)) - 32, y1 = ceil(fmax(y, y + h)) + 32;
    VkRect2D sc = scissor_rect();
    x0 = fmax(x0, (double)sc.offset.x); y0 = fmax(y0, (double)sc.offset.y); x1 = fmin(x1, (double)sc.offset.x + sc.extent.width); y1 = fmin(y1, (double)sc.offset.y + sc.extent.height);
    x0 = fmax(x0, 0); y0 = fmax(y0, 0); x1 = fmin(x1, (double)S.pass_w); y1 = fmin(y1, (double)S.pass_h);
    if (x1 <= x0 || y1 <= y0) return;
    if ((x1 - x0) * (y1 - y0) * 10 > (double)S.pass_w * S.pass_h * 9) return;   // most of the target: as before
    if (!parea_scan()) { __atomic_fetch_add(&g_parea_kept, 1, __ATOMIC_RELAXED); return; }   // (S1e (parallel recording): counted by every recording thread)
    P_ri.renderArea = (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
    uint64_t n = __atomic_add_fetch(&g_parea_n, 1, __ATOMIC_RELAXED); if (n <= 4 || !(n & (n - 1))) VKLOG("r89: pass %ux%u drawn in %ux%u at %d,%d: render area cut to it (%llu so far, %llu kept whole for a later viewport, scissor or clear; DM_VK_PASS_AREA=0 off)",
        S.pass_w, S.pass_h, P_ri.renderArea.extent.width, P_ri.renderArea.extent.height, P_ri.renderArea.offset.x, P_ri.renderArea.offset.y, (unsigned long long)n, (unsigned long long)g_parea_kept);
}
// 0: no pass could be begun (no render pass or framebuffer, and no dynamic rendering to fall back on): the caller records nothing that needs one
// NVN commands and crops stay in guest texels. A pass uses one physical scale
// for every attachment. Keep the native backing coherent for CPU and storage access.
// (S1e (parallel recording): pass_scale, scale_writes and scale_dirty are the recording context's, struct rec_ctx)
// Carry a downsample through state-only records or one verified draw using
// private scaled attachments. Memory consumers and list boundaries flush it.
static struct pending_scale_write {
    vkimg_t *im;
    VkImageAspectFlags aspects;
    VkRect2D rect;
} pending_scale[9];
static unsigned pending_scale_count;
static int collect_scale, lazy_scale, pending_guest_barrier;
static int carry_one_draw;
static int draw_uses_only_uniforms(int prim);
static int scale_time_mode = -1;  // 1 includes writeback in pass time, 2 times writebacks separately
// scale_writes (struct rec_ctx): excludes loadOp clears, handled separately
// All Eden viewports share a scissor. Collect draws and explicit clears in
// native texels, clipped to the physical render area and rounded outwards (scale_dirty, struct rec_ctx).
static inline int scale_min(int a, int b) { return a < b ? a : b; }
static inline int scale_max(int a, int b) { return a > b ? a : b; }
static void scale_note_scissor(VkRect2D sc) {
    if (pass_scale != 2 || !sc.extent.width || !sc.extent.height) return;
    VkRect2D area = P_ri.renderArea;
    int left = scale_max(sc.offset.x, area.offset.x);
    int top = scale_max(sc.offset.y, area.offset.y);
    int right = scale_min(sc.offset.x + (int)sc.extent.width, area.offset.x + (int)area.extent.width);
    int bottom = scale_min(sc.offset.y + (int)sc.extent.height, area.offset.y + (int)area.extent.height);
    if (right <= left || bottom <= top) return;
    scale_dirty.left = scale_min(scale_dirty.left, left / 2);
    scale_dirty.top = scale_min(scale_dirty.top, top / 2);
    scale_dirty.right = scale_max(scale_dirty.right, (right + 1) / 2);
    scale_dirty.bottom = scale_max(scale_dirty.bottom, (bottom + 1) / 2);
}
static VkRect2D physical_rect(VkRect2D r) {
    r.offset.x *= pass_scale; r.offset.y *= pass_scale;
    r.extent.width *= pass_scale; r.extent.height *= pass_scale;
    return r;
}
// The scale copies run on the graphics queue. Only transfer commands need to
// wait at the first dependency; unrelated graphics work can stay in flight.
static void before_scale_transfer(VkCommandBuffer cb) {
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT,
                               VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
}
static void after_scale_transfer(VkCommandBuffer cb) {
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT,
                               VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
}
static void materialize_scale(vkimg_t *overwritten) {
    if (!pending_scale_count) return;
    VkCommandBuffer cb = vk_cmd();
    int copied = 0;
    for (unsigned i = 0; i < pending_scale_count; ++i) {
        struct pending_scale_write *p = &pending_scale[i];
        if (p->im == overwritten) { ++scale_elided; continue; }
        if (!copied) before_scale_transfer(cb);
        copied = 1;
        wb_blit(p->im, p->aspects, p->rect);
        vk_rescale_blit(cb, p->im, 0, p->aspects, p->rect);
    }
    if (pending_guest_barrier) vk_img_barrier_all(cb);
    else if (copied) after_scale_transfer(cb);
    pending_guest_barrier = 0;
    pending_scale_count = 0;
}
static void choose_pass_scale(void) {
    pass_scale = 1;
    memset(scale_writes, 0, sizeof scale_writes);
#ifdef DM_NXN1X
    if (!vk_eden_mode() || !nvn_eden_resolution_x2()) return;
    int ready = 0;
    for (int i = 0; i < 9; ++i) {
        vkimg_t *im = P_att[i].im;
        if (!im) continue;
        if (!vk_rescale_eligible(im)) {
            if (S.pass_w >= 960 && S.pass_h >= 540) {
                static struct { uint32_t w,h,fmt,levels,layers,aliases,slot; } seen[48];
                static size_t count;
                typeof(seen[0]) value = {S.pass_w,S.pass_h,im->fmt,im->levels,im->layers,im->novl,i};
                size_t j = 0;
                while (j < count && memcmp(&value, &seen[j], sizeof value)) ++j;
                if (j == count && count < 48) {
                    seen[count++] = value;
                    VKLOG("NVN 2X fallback %ux%u: slot %d image %ux%u format %u view %u levels %u layers %u samples %u overlaps %d flags %#x caps %d",
                          S.pass_w,S.pass_h,i,im->w,im->h,im->fmt,im->vtype,im->levels,im->layers,im->samples,im->novl,im->flags,im->scale_caps);
                }
            }
            return;
        }
        if (im->scale_frame != g_tex_frame + 1) {
            im->scale_frame = g_tex_frame + 1;
            if (im->scale_rating < 2) ++im->scale_rating;
        }
        ready |= i == 8 || im->scaled_valid || im->scale_rating >= 2;
    }
    if (!ready) return;
    VkImageView views[9] = {0};
    for (int i = 0; i < 9; ++i) {
        if (!P_att[i].im) continue;
        const tex_t *tex = i == 8 ? &S.rt_dep : &S.rt_col[i];
        const view_t *view = i == 8 ? (S.rt_has_dview ? &S.rt_dview : NULL)
                                    : (S.rt_has_cview[i] ? &S.rt_cview[i] : NULL);
        views[i] = vk_rescaled_view(P_att[i].im, tex, view, 0xff, 0, 1);
        if (!views[i]) return;
    }
    VkCommandBuffer cb = vk_cmd();
    int copied = 0;
    for (int i = 0; i < 9; ++i) {
        vkimg_t *im = P_att[i].im;
        if (!im) continue;
        int load = i < 8 ? P_ca[i].loadOp == VK_ATTACHMENT_LOAD_OP_LOAD :
                   P_da.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD ||
                   (P_sa.imageView && P_sa.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
        if (!im->scaled_valid && (load || im->w != S.pass_w || im->h != S.pass_h ||
            P_ri.renderArea.offset.x || P_ri.renderArea.offset.y ||
            P_ri.renderArea.extent.width != S.pass_w || P_ri.renderArea.extent.height != S.pass_h)) {
            materialize_scale(NULL);
            if (!copied) before_scale_transfer(cb);
            copied = 1;
            wb_use(im, WB_COPY);
            vk_rescale_blit(cb, im, 1, im->aspect, (VkRect2D){{0, 0}, {im->w, im->h}});
        } else {
            vk_tex_prepare(cb, im->scaled);
        }
        im->scaled_valid = 0;
        if (i < 8) P_ca[i].imageView = views[i];
        else {
            P_da.imageView = views[i];
            if (P_sa.imageView) P_sa.imageView = views[i];
        }
    }
    if (copied) after_scale_transfer(cb);
    pass_scale = 2;
    scale_dirty = (typeof(scale_dirty)){INT_MAX, INT_MAX, 0, 0};
    static int reported;
    if (!reported) {
        VKLOG("NVN 2X render target: %ux%u -> %ux%u", S.pass_w, S.pass_h, S.pass_w * 2, S.pass_h * 2);
        reported = 1;
    }
#endif
}
static void finish_pass_scale(void) {
    if (pass_scale != 2) return;
    VkCommandBuffer cb = vk_cmd();
    // The one-draw pass loaded the same scaled attachments and had no image
    // descriptors. Its forthcoming writeback can cover an older pending one.
    if (carry_one_draw && pending_scale_count) {
        for (unsigned n = pending_scale_count; n-- > 0;) {
            const struct pending_scale_write *p = &pending_scale[n];
            for (int i = 0; i < 9; ++i) {
                if (P_att[i].im != p->im || (p->aspects & ~scale_writes[i])) continue;
                if (scale_dirty.left <= p->rect.offset.x && scale_dirty.top <= p->rect.offset.y &&
                    scale_dirty.right >= p->rect.offset.x + (int)p->rect.extent.width &&
                    scale_dirty.bottom >= p->rect.offset.y + (int)p->rect.extent.height) {
                    pending_scale[n] = pending_scale[--pending_scale_count];
                    ++scale_elided;
                }
                break;
            }
        }
        materialize_scale(NULL);
    }
    int transferring = 0;
    for (int i = 0; i < 9; ++i) {
        vkimg_t *im = P_att[i].im;
        if (!im) continue;
        const VkRect2D area = {{P_ri.renderArea.offset.x / 2, P_ri.renderArea.offset.y / 2},
                               {P_ri.renderArea.extent.width / 2, P_ri.renderArea.extent.height / 2}};
        VkImageAspectFlags changed = scale_writes[i];
        VkImageAspectFlags clears = 0;
        if (i < 8 && P_ca[i].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) clears |= VK_IMAGE_ASPECT_COLOR_BIT;
        if (i == 8 && P_da.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) clears |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if (i == 8 && P_sa.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) clears |= VK_IMAGE_ASPECT_STENCIL_BIT;
        clears &= im->aspect;
        if (scale_dirty.right <= scale_dirty.left) changed = 0;
        if (!(changed | clears)) {
            im->scaled_valid = 1;
            continue;
        }
        int defer = collect_scale && !clears && !im->external_use && !im->novl &&
                    !im->packaged && pending_scale_count < 9;
        if (!defer && !transferring) {
            if (scale_time_mode == 2) vk_pass_ts(0, 0, 0, VK_TS_SCALE_WRITEBACK, 0);
            before_scale_transfer(cb);
        }
        if (!defer) transferring = 1;
        // A whole-image load clear with no subsequent writes is a native clear as well.
        if (!area.offset.x && !area.offset.y && area.extent.width == im->w && area.extent.height == im->h) {
            VkImageAspectFlags untouched = clears & ~changed;
            VkImageSubresourceRange range = {untouched, 0, 1, 0, 1};
            if (untouched && i < 8 && P_att[i].fmt == im->fmt)
                vkCmdClearColorImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &P_ca[i].clearValue.color, 1, &range);
            else if (untouched && i == 8) {
                VkClearDepthStencilValue value = {P_da.clearValue.depthStencil.depth, P_sa.clearValue.depthStencil.stencil};
                vkCmdClearDepthStencilImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
            } else untouched = 0;
            if (untouched) wb_use(im, WB_OTHER);
            clears &= ~untouched;
        }
        changed = (changed | clears) & im->aspect;
        if (changed) {
            VkRect2D update = area;
            if (!clears) {
                update.offset = (VkOffset2D){scale_dirty.left, scale_dirty.top};
                update.extent = (VkExtent2D){scale_dirty.right - scale_dirty.left, scale_dirty.bottom - scale_dirty.top};
            }
            if (defer) {
                pending_scale[pending_scale_count++] = (struct pending_scale_write){im, changed, update};
                ++scale_deferred;
            } else {
                wb_blit(im, changed, update);
                vk_rescale_blit(cb, im, 0, changed, update);
            }
        }
        im->scaled_valid = 1;
    }
    if (transferring) {
        after_scale_transfer(cb);
        if (scale_time_mode == 2)
            vk_pass_ts(1, S.pass_w, S.pass_h, VK_TS_SCALE_WRITEBACK, g_xo_op == OP_BARRIER);
    }
    pass_scale = 1;
}
// ootdlss D1a (11-21, design OOT_DLSS_DESIGN.md, log only): DM_VK_OOTDLSS_SURVEY=1 checks that the replay sees Ocarina of Time's frame as NX2navi (refer\NX2navi-main
// runtime\vk_dlaa.c, the same game recompiled) describes it, before DLSS is put in: draws into a D32F scene depth binding a 0x3500-byte scene context at vertex c3 (UBO 0),
// whose rows 0-2 are the view and 7-10 the projection; a pass with one R11G11B10F / RGBA16F target and that depth "arms" the frame (the light target); the first later pass
// with one R11G11B10F target, no depth, larger and of the scene's shape is where the game's upscaler writes (DLSS would run there). Every 300 presents a line. Keeps every
// segment on the replay thread (seg_worker_refuse): its counts follow the stream order
// gpuck (11-21, NPARK: three device losses, one in an area change, and no VK_EXT_device_fault here): NVIDIA's diagnostic checkpoints. Each pass begin, dispatch and
// DLSS evaluation records a checkpoint whose marker is an index into a ring of what it was (submission, present, pass, shader keys, size); at a device loss
// (vk_f0_result, gfx_vk_core.c) vk_ck_report reads the queue's last checkpoints back and logs each with its ring entry: the last work the GPU reached.
// The extension is enabled by vulkan_device.cpp unless DM_VK_CHECKPOINTS=0; without it vkGetDeviceProcAddr gives NULL and nothing is marked
typedef struct { uint64_t sub, frame, k0, k1; uint32_t pass, w, h; uint16_t kind, wk; } ck_ent;   // kind 1 pass (k0 vs, k1 fs), 2 dispatch (k0 cs), 3 DLSS
static ck_ent g_ckr[1 << 16]; static _Atomic uint32_t g_ckn; static int g_ck = -1;
static PFN_vkCmdSetCheckpointNV g_ck_set; static PFN_vkGetQueueCheckpointDataNV g_ck_get;
static void ck_mark(VkCommandBuffer cb, int kind, uint64_t k0, uint64_t k1, uint32_t w, uint32_t h) {
    if (g_ck < 0) { const char *e = getenv("DM_VK_CHECKPOINTS"); g_ck = 0;
        if (!(e && *e == '0')) { g_ck_set = (PFN_vkCmdSetCheckpointNV)vkGetDeviceProcAddr(V.dev, "vkCmdSetCheckpointNV"); g_ck_get = (PFN_vkGetQueueCheckpointDataNV)vkGetDeviceProcAddr(V.dev, "vkGetQueueCheckpointDataNV"); g_ck = g_ck_set && g_ck_get; }
        VKLOG("gpuck: checkpoints %s (DM_VK_CHECKPOINTS %s)", g_ck ? "marked at each pass, dispatch and DLSS evaluation" : "off (extension not enabled or switched off)", e && *e ? e : "unset"); }
    if (!g_ck) return;
    const uint32_t i = atomic_fetch_add_explicit(&g_ckn, 1, memory_order_relaxed);
    ck_ent *x = &g_ckr[i & 0xffff]; x->sub = mtl_last_fence() + 1; x->frame = g_tex_frame; x->k0 = k0; x->k1 = k1; x->pass = (uint32_t)g_sub_pass; x->w = w; x->h = h; x->kind = (uint16_t)kind; x->wk = (uint16_t)(g_wk ? (g_wk_rerun ? 2 : 1) : 0);   // parck: 2 = the replay thread's own recording of a worker segment
    g_ck_set(cb, (const void *)(uintptr_t)(i + 1));
}
void vk_ck_report(void) {   // gfx_vk_core.c vk_f0_result at the first VK_ERROR_DEVICE_LOST
    static int done; if (done++ || g_ck <= 0) return;
    uint32_t n = 0; g_ck_get(V.q, &n, NULL); if (n > 64) n = 64;
    VkCheckpointDataNV d[64]; for (uint32_t i = 0; i < n; i++) d[i] = (VkCheckpointDataNV){VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV, NULL, 0, NULL};
    if (n) g_ck_get(V.q, &n, d);
    const uint32_t last = atomic_load(&g_ckn);
    VKLOG("gpuck: device lost: %u checkpoints read back from the queue (%u marked so far, present %llu)", n, last, (unsigned long long)g_tex_frame);
    static const char *const kn[5] = {"?", "pass", "dispatch", "DLSS", "draw"};   // parck: 4 a draw recorded in a segment chunk (w: its index in the pass, h: 1 indexed, 2 vertex pull, prim << 4)
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t m = (uint32_t)(uintptr_t)d[i].pCheckpointMarker;
        if (!m || last - (m - 1) > 0xffff) { VKLOG("gpuck:   stage %#x marker %u (outside the ring)", (unsigned)d[i].stage, m); continue; }
        const ck_ent *x = &g_ckr[(m - 1) & 0xffff];
        VKLOG("gpuck:   stage %#x marker %u (%u before the last): %s, submission %llu, present %llu, pass %u, %ux%u, keys %016llx %016llx%s", (unsigned)d[i].stage, m, last - m,
              kn[x->kind < 5 ? x->kind : 0], (unsigned long long)x->sub, (unsigned long long)x->frame, x->pass, x->w, x->h, (unsigned long long)x->k0, (unsigned long long)x->k1, x->wk == 2 ? " (segment re-recorded on the replay thread)" : x->wk ? " (worker)" : "");
    }
}
void vk_ubd_read(uint64_t a, uint64_t n, uint8_t *dst);
static int g_ootd = -1;
static int ootd_on(void) { if (g_ootd < 0) { const char *e = getenv("DM_VK_OOTDLSS_SURVEY"); g_ootd = e && *e && atoi(e) != 0; } return g_ootd; }
#define OOTD_CAM 0x3500u
#define OOTD_ROWS 0xb0u
static struct {
    uint64_t frame; int armed, done, gbuf_after; uint32_t aw, ah, afmt; const void *aim;   // this frame (g_tex_frame)
    uint32_t zdraws, cdraws, cother, czoff; uint32_t naddr; uint64_t addr[8]; uint32_t an[8]; float cam[8][OOTD_ROWS / 4]; int got[8];   // draws into the scene depth: all / with the context / with another c3 / with the context, not into a scene colour target
    uint32_t arm_passes, ndk; uint64_t dk[8]; uint32_t dn[8]; uint64_t up_fs; uint32_t up_w, up_h, up_draws; int in_up;   // dispatches between arming and the upscale pass; the upscale pass
    uint32_t one_w, one_h, one_fmt, one_n;   // after arming: passes with one colour target and no depth (the last's)
    // accumulated over the 300-present window
    uint64_t frames, f_armed, f_up, f_cam, zd, cd, co, cz, gb; uint32_t c3sz[4]; uint64_t c3n[4];
    double prev_vp[16]; int prev_ok; uint64_t vp_moved, vp_same; uint64_t dsum[8]; uint64_t dkey[8]; uint32_t ndsum;
    char last[640];
} g_od;
static int ootd_scene_fmt(uint32_t f) { return f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || f == VK_FORMAT_R16G16B16A16_SFLOAT; }
static void ootd_vp(const float *v, double out[16]) {   // NX2navi ocam_vp: P (rows 7-10) x V (rows 0-2, then 0 0 0 1)
    const float *P = v + 7 * 4; double V[16] = {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], 0, 0, 0, 1};
    for (int r = 0; r < 4; r++) for (int k = 0; k < 4; k++) { double acc = 0; for (int j = 0; j < 4; j++) acc += (double)P[r * 4 + j] * V[j * 4 + k]; out[r * 4 + k] = acc; }
}
static void ootd_frame_end(void) {   // the frame g_od.frame is over: its camera checked, the window counts taken
    g_od.frames++; g_od.f_armed += g_od.armed; g_od.f_up += g_od.done; g_od.zd += g_od.zdraws; g_od.cd += g_od.cdraws; g_od.co += g_od.cother; g_od.cz += g_od.czoff; g_od.gb += g_od.gbuf_after;
    for (uint32_t i = 0; i < g_od.ndk; i++) { uint32_t k = 0; while (k < g_od.ndsum && g_od.dkey[k] != g_od.dk[i]) k++;
        if (k == g_od.ndsum && k < 8) { g_od.dkey[k] = g_od.dk[i]; g_od.dsum[k] = 0; g_od.ndsum++; } if (k < 8) g_od.dsum[k] += g_od.dn[i]; }
    int best = -1; for (uint32_t i = 0; i < g_od.naddr; i++) if (g_od.got[i] && (best < 0 || g_od.an[i] > g_od.an[best])) best = (int)i;
    if (best >= 0) {
        g_od.f_cam++; const float *v = g_od.cam[best]; double vp[16]; ootd_vp(v, vp);
        double orth = 0; for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) { double d = 0; for (int k = 0; k < 3; k++) d += (double)v[a * 4 + k] * v[b * 4 + k]; d -= a == b; if (fabs(d) > orth) orth = fabs(d); }
        const float *P = v + 28; double moved = 0; const int had = g_od.prev_ok; if (had) for (int i = 0; i < 16; i++) { double d = fabs(vp[i] - g_od.prev_vp[i]); if (d > moved) moved = d; }
        if (had) { if (moved > 1e-6) g_od.vp_moved++; else g_od.vp_same++; }
        memcpy(g_od.prev_vp, vp, sizeof vp); g_od.prev_ok = 1;
        snprintf(g_od.last, sizeof g_od.last, "context %#llx (%u of %u context draws, %u contexts): view rows orthonormal to %.2e, position %.2f %.2f %.2f; projection rows 7-10: [0][0] %.4f [1][1] %.4f (aspect %.4f, scene %.4f) [2][2] %.5f [2][3] %.5f [3][2] %.5f [3][3] %.5f; view-projection change since last frame %.3e",
            (unsigned long long)g_od.addr[best], g_od.an[best], g_od.cdraws, g_od.naddr, orth, v[3], v[7], v[11], P[0], P[5], P[0] != 0 ? fabs(P[5] / P[0]) : 0.0, g_od.ah ? (double)g_od.aw / g_od.ah : 0.0,
            P[10], P[11], P[14], P[15], had ? moved : -1.0);
    } else g_od.prev_ok = 0;
    if (g_od.frames >= 300) {
        char d[400]; int k = 0; for (uint32_t i = 0; i < g_od.ndsum && k < (int)sizeof d - 40; i++) k += snprintf(d + k, sizeof d - (size_t)k, " %016llx x%.2f", (unsigned long long)g_od.dkey[i], (double)g_od.dsum[i] / g_od.frames);
        char c3[160]; int q = 0; for (int i = 0; i < 4; i++) if (g_od.c3n[i]) q += snprintf(c3 + q, sizeof c3 - (size_t)q, " %#x x%.1f", g_od.c3sz[i], (double)g_od.c3n[i] / g_od.frames); if (!q) c3[0] = 0;
        VKLOG("ootdlss survey (%llu presents): scene-depth draws %.1f a frame, with the 0x3500 context at vertex UBO 0 %.1f (not into a scene colour target %.1f), with another UBO 0 %.1f (sizes:%s); frames armed %llu, with the upscale pass %llu, with a camera %llu; G-buffer passes after arming %.1f a frame",
              (unsigned long long)g_od.frames, (double)g_od.zd / g_od.frames, (double)g_od.cd / g_od.frames, (double)g_od.cz / g_od.frames, (double)g_od.co / g_od.frames, q ? c3 : " none",
              (unsigned long long)g_od.f_armed, (unsigned long long)g_od.f_up, (unsigned long long)g_od.f_cam, (double)g_od.gb / g_od.frames);
        VKLOG("ootdlss survey: last frame: light target %ux%u vk format %u (%u passes into it), upscale pass %ux%u (fs %016llx, %u draws); passes with one colour target and no depth after arming %u (last %ux%u vk format %u); dispatches between arming and the upscale pass, a frame:%s",
              g_od.aw, g_od.ah, g_od.afmt, g_od.arm_passes, g_od.up_w, g_od.up_h, (unsigned long long)g_od.up_fs, g_od.up_draws, g_od.one_n, g_od.one_w, g_od.one_h, g_od.one_fmt, k ? d : " none");
        VKLOG("ootdlss survey: camera: %s; view-projection moved in %llu frames, unchanged in %llu", g_od.last[0] ? g_od.last : "none seen", (unsigned long long)g_od.vp_moved, (unsigned long long)g_od.vp_same);
        g_od.frames = g_od.f_armed = g_od.f_up = g_od.f_cam = g_od.zd = g_od.cd = g_od.co = g_od.cz = g_od.gb = 0; memset(g_od.c3n, 0, sizeof g_od.c3n); g_od.ndsum = 0; g_od.vp_moved = g_od.vp_same = 0; g_od.last[0] = 0;
    }
    g_od.armed = g_od.done = g_od.gbuf_after = 0; g_od.aim = NULL; g_od.aw = g_od.ah = g_od.afmt = 0; g_od.zdraws = g_od.cdraws = g_od.cother = g_od.czoff = 0; g_od.naddr = 0;
    g_od.arm_passes = g_od.ndk = 0; g_od.up_fs = 0; g_od.up_w = g_od.up_h = g_od.up_draws = 0; g_od.in_up = 0; g_od.one_w = g_od.one_h = g_od.one_fmt = g_od.one_n = 0;
}
static void ootd_tick(void) { if (g_od.frame != g_tex_frame) { if (g_od.frame) ootd_frame_end(); g_od.frame = g_tex_frame; } }
static void ootd_pass(void) {   // pass_commit, after the pass began (replay thread: ootd_on keeps the workers out)
    if (!ootd_on() || g_wk) return; ootd_tick();
    g_od.in_up = 0;
    const int scene = P_att[8].im && P_att[8].fmt == VK_FORMAT_D32_SFLOAT && P_att[8].w >= 640;
    if (S.ncol >= 3 && scene) { if (g_od.armed) g_od.gbuf_after++; return; }
    if (S.ncol == 1 && P_att[0].im && scene && ootd_scene_fmt(P_att[0].fmt) && P_att[0].w == P_att[8].w && P_att[0].h == P_att[8].h &&
        (!g_od.armed || (P_att[0].w >= g_od.aw && P_att[0].h >= g_od.ah))) {
        if (g_od.aim != (const void *)P_att[0].im) g_od.arm_passes = 0;
        g_od.armed = 1; g_od.aim = P_att[0].im; g_od.aw = P_att[0].w; g_od.ah = P_att[0].h; g_od.afmt = P_att[0].fmt; g_od.arm_passes++; return;
    }
    if (!g_od.armed || g_od.done) return;
    if (S.ncol == 1 && P_att[0].im && !P_att[8].im && (const void *)P_att[0].im != g_od.aim) {
        g_od.one_n++; g_od.one_w = P_att[0].w; g_od.one_h = P_att[0].h; g_od.one_fmt = P_att[0].fmt;
        const double w = P_att[0].w, h = P_att[0].h;
        if (P_att[0].fmt == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && P_att[0].w > g_od.aw && P_att[0].h > g_od.ah && w * g_od.ah <= 1.03 * h * g_od.aw && h * g_od.aw <= 1.03 * w * g_od.ah) {
            g_od.done = 1; g_od.in_up = 1; g_od.up_w = P_att[0].w; g_od.up_h = P_att[0].h; g_od.up_draws = 0; g_od.up_fs = 0;
        }
    }
}
static void ootd_dispatch(const vkshader_t *cs) {
    if (!ootd_on() || g_wk) return; ootd_tick();
    if (!g_od.armed || g_od.done || !cs) return;
    uint32_t i = 0; while (i < g_od.ndk && g_od.dk[i] != cs->key) i++;
    if (i == g_od.ndk && i < 8) { g_od.dk[i] = cs->key; g_od.dn[i] = 0; g_od.ndk++; } if (i < 8) g_od.dn[i]++;
}
static void ootd_draw(void) {   // fg_observe_draw: every draw of both paths
    if (!ootd_on() || g_wk) return; ootd_tick();
    if (g_od.in_up) { g_od.up_draws++; if (!g_od.up_fs && S.sh[1]) g_od.up_fs = S.sh[1]->key; }
    if (!P_att[8].im || P_att[8].fmt != VK_FORMAT_D32_SFLOAT || P_att[8].w < 640) return;
    g_od.zdraws++;
    const p_bind_buf *u = &S.ubo[0][0];
    if (!(u->addr && u->size == OOTD_CAM)) {
        g_od.cother++; uint32_t sz = u->addr ? (uint32_t)u->size : 0; int i = 0; while (i < 4 && g_od.c3n[i] && g_od.c3sz[i] != sz) i++;
        if (i < 4) { g_od.c3sz[i] = sz; g_od.c3n[i]++; } return;
    }
    g_od.cdraws++;
    const int into_scene = S.ncol >= 1 && P_att[0].im && ootd_scene_fmt(P_att[0].fmt);
    if (!into_scene) { g_od.czoff++; return; }
    uint32_t i = 0; while (i < g_od.naddr && g_od.addr[i] != u->addr) i++;
    if (i == g_od.naddr) { if (i >= 8) return; g_od.addr[i] = u->addr; g_od.an[i] = 0; g_od.got[i] = 0; g_od.naddr++; }
    g_od.an[i]++;
    if (!g_od.got[i] && vk_pool_of(u->addr, OOTD_ROWS)) { vk_ubd_read(u->addr, OOTD_ROWS, (uint8_t *)g_od.cam[i]); g_od.got[i] = 1; }
}
static int pass_commit(void) {
    struct rec_ctx *const g_rec = rec_cur();   // R1b (parallel recording): the thread's context read once (S, P_*, g_pass_* below)
    if (S.pass_open != 2) return S.pass_open == 1;
    if (!g_wk) { void vk_tbuf_fresh(VkCommandBuffer cb); vk_tbuf_fresh(vk_cmd()); }   // r150 (gfx_vk_tex.c): texture buffer copies the GPU wrote since, made again before the pass (S1e (parallel recording): a worker
    // segment's by the replay thread at its hand-out: no GPU write inside the segment marks one, and a mark from a later serial record must wait for the replay thread's next pass)
    vk_pass_ts(0, 0, 0, 0, 0);
    if (scale_time_mode < 0) {
        const char *lazy = getenv("DM_VK_LAZY_WRITEBACK");
        lazy_scale = lazy && atoi(lazy);
        const char *mode = getenv("DM_VK_TIME_PASS_WRITEBACK");
        extern int vk_pass_times_active(void);
        int requested = vk_pass_times_active() && mode ? atoi(mode) : 0;
        scale_time_mode = requested == 2 ? 2 : requested != 0;
    }
    pass_area();   // r89
    choose_pass_scale();
    if (pass_scale == 1) {
        materialize_scale(NULL);
        if (!g_wk) for (int i = 0; i < 9; ++i) if (P_att[i].im) { wb_use(P_att[i].im, WB_ATTACHMENT); P_att[i].im->scaled_valid = 0; }   // (S1e (parallel recording): a worker's targets marked at its hand-out)
    }
    P_ri.renderArea = physical_rect(P_ri.renderArea);
    if (rp_on()) {
        rpkey k; memset(&k, 0, sizeof k); k.ncol = P_ri.colorAttachmentCount; k.samples = S.samples ? S.samples : 1;
        fbkey fk; memset(&fk, 0, sizeof fk); VkImageView views[9]; VkClearValue cv[9]; uint32_t na = 0;
        for (uint32_t i = 0; i < k.ncol; i++) if (P_ca[i].imageView) {
            k.cfmt[i] = P_att[i].fmt; k.cload[i] = P_ca[i].loadOp; cv[na] = P_ca[i].clearValue; views[na] = P_ca[i].imageView;
            fk.a[na].usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; fk.a[na].flags = (pass_scale == 2 ? P_att[i].im->scaled : P_att[i].im)->cflags; fk.a[na].w = P_att[i].image_w * pass_scale; fk.a[na].h = P_att[i].image_h * pass_scale; fk.a[na].layers = P_att[i].layers;
            fk.a[na].fmt = P_att[i].fmt; na++; }
        if (P_ri.pDepthAttachment || P_ri.pStencilAttachment) {
            k.dfmt = P_att[8].fmt; k.dload = P_da.loadOp; k.sload = P_sa.imageView ? P_sa.loadOp : VK_ATTACHMENT_LOAD_OP_DONT_CARE; k.dstore = P_da.storeOp;
            cv[na] = P_da.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ? P_da.clearValue : P_sa.clearValue; if (P_sa.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) cv[na].depthStencil.stencil = P_sa.clearValue.depthStencil.stencil;
            views[na] = P_da.imageView;
            fk.a[na].usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT; fk.a[na].flags = (pass_scale == 2 ? P_att[8].im->scaled : P_att[8].im)->cflags; fk.a[na].w = P_att[8].w * pass_scale; fk.a[na].h = P_att[8].h * pass_scale; fk.a[na].layers = P_att[8].layers; fk.a[na].fmt = P_att[8].fmt; na++; }
        VkRenderPass rp = rp_get(&k);
        fk.rp = rp_compat(k.ncol, k.cfmt, k.dfmt, k.samples); fk.w = S.pass_w * pass_scale; fk.h = S.pass_h * pass_scale;   /* r89: the target's size (the render area may be cut, pass_area) */ fk.layers = P_ri.layerCount; fk.n = na;
        VkFramebuffer fb = rp && fk.rp ? fb_get(&fk) : VK_NULL_HANDLE;
        if (fb) { VkRenderPassAttachmentBeginInfo abi = {VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO, NULL, na, views};
            VkRenderPassBeginInfo rbi = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, &abi, rp, fb, P_ri.renderArea, na, cv};
            vkCmdBeginRenderPass(vk_cmd(), &rbi, VK_SUBPASS_CONTENTS_INLINE); S.rp_pass = 1; }
        else if (V.has_dynrender) { vkCmdBeginRendering(vk_cmd(), &P_ri); S.rp_pass = 0; }
        else { VKRED_ONCE(76, "render pass or framebuffer failed without dynamic rendering: pass skipped"); return 0; }   // (still pass_open 2: vk_end_pass drops it)
    } else { vkCmdBeginRendering(vk_cmd(), &P_ri); S.rp_pass = 0; }
    { void vk_clrdiag_done(vkimg_t *); for (int i = 0; i < 9; i++) if (P_att[i].im && P_att[i].im->clr_pend) vk_clrdiag_done(P_att[i].im); }   // clrorder: the parked clears run here
    if (P_att[8].im) { void vk_res_note_scene(uint32_t, uint32_t); vk_res_note_scene(P_att[8].w, P_att[8].h); }   // resoverlay2: the 3D resolution (the largest non-square depth target of the frame)
    ootd_pass();   // ootdlss D1a survey (DM_VK_OOTDLSS_SURVEY=1, log only)
    ck_mark(vk_cmd(), 1, S.sh[0] ? S.sh[0]->key : 0, S.sh[1] ? S.sh[1]->key : 0, S.pass_w, S.pass_h);   // gpuck
    S.pass_open = 1; g_vkst.passes++; if (!g_wk) g_sub_pass++;   // (S1e (parallel recording): the submission's pass numbers are the replay thread's; a worker's passes are not numbered)
    // the bind cache starts empty in each pass on the Qualcomm driver: it did not keep the state a pass's draws set for the next pass (XC3 drew its lighting black). r129: kept on Turnip,
    // which keeps the bound state for the whole command buffer as Vulkan says (the host's own binds inside a submission, the masked clear and vk_cmd_split, move g_cmd_gen themselves)
    { if (g_x_passbc > 0) g_cmd_gen++; }   // R2 (parallel recording): g_cmd_gen is the thread's recording target's (gfx_vk.h)
    if (g_x_tracepass == -2) { g_x_listpasses = 1; VKLOG("--- pass %d: %ux%u, colour formats %d %d, depth format %d", g_sub_pass - 1, S.pass_w, S.pass_h, S.cfmt[0], S.cfmt[1], S.dfmt); }
    if (g_x_traceframe[0]) {   // DM_VK_TRACE_FRAME=WxH[:fmt[:fmt]] (live): trace every pass of the next submission whose targets match (a whole effect chain in one log)
        char b[48]; snprintf(b, sizeof b, "%ux%u:%d:%d", S.pass_w, S.pass_h, S.cfmt[0], S.cfmt[1]);
        g_x_tracepass = !strncmp(b, g_x_traceframe, strlen(g_x_traceframe)) ? g_sub_pass - 1 : -1; g_x_traceframe_on = 1; }
    if (g_x_tracesize[0] && g_x_tracepass < 0) {   // DM_VK_TRACE_SIZE=WxH[:format[:format]] (live): trace the next pass whose first colour target is that (passes are numbered anew each frame)
        char b[48]; snprintf(b, sizeof b, "%ux%u:%d:%d", S.pass_w, S.pass_h, S.cfmt[0], S.cfmt[1]);
        const char *h = strchr(g_x_tracesize, '#'); size_t l = h ? (size_t)(h - g_x_tracesize) : strlen(g_x_tracesize);   // #n: the (n+1)th such pass
        if (!strncmp(b, g_x_tracesize, l)) { if (h && atoi(h + 1) > 0) { snprintf(g_x_tracesize + l, sizeof g_x_tracesize - l, "#%d", atoi(h + 1) - 1); setenv("DM_VK_TRACE_SIZE", g_x_tracesize, 1); }
            else { g_x_tracepass = g_sub_pass - 1; g_x_tracesize[0] = 0; setenv("DM_VK_TRACE_SIZE", "", 1); } } }
    if (g_x_tracepass >= 0 && g_sub_pass - 1 == g_x_tracepass) VKLOG("=== traced pass %d: %ux%u, colour formats %d %d, depth format %d", g_x_tracepass, S.pass_w, S.pass_h, S.cfmt[0], S.cfmt[1], S.dfmt); g_pass_draws = 0; g_pass_bb[0] = g_pass_bb[1] = INT32_MAX; g_pass_bb[2] = g_pass_bb[3] = INT32_MIN;
    return 1;
}
void vk_ubd_land(VkCommandBuffer cb); int vk_ubd_shadow(uint64_t a, uint64_t n, VkBuffer *buf, VkDeviceSize *off); void vk_ubd_read(uint64_t a, uint64_t n, uint8_t *dst);
// DM_VK_PASSDUMP=1 (a DM_CTL "env" line; =N waits N - 1 presents first): the next submissions' passes each leave their first colour target in a buffer (up to 48 passes), written as
// DM_ROOT/pass_NN.ppm when the GPU is done (RGBA8 as is, RGBA16F clamped to 0..1, with its NaN / infinite pixel count logged)
// DM_VK_PASSDUMP_SKIP=n (read when a dump starts): leave out each submission's first n passes, to reach a frame's later passes (Xenoblade Chronicles 3's title frame has about 200);
// =WxH: start at the first pass of that size (pass numbers move by tens from frame to frame)
// DM_VK_PASSDUMP_SIZE=WxH[,WxH...] (read when a dump starts): only passes whose first colour target has one of these sizes, over every submission of the frame (no depth),
// to follow one effect's targets when the pass numbers move from frame to frame (Xenoblade Chronicles 3's 384x216 clouds)
// DM_VK_PASSDUMP_FRAMES=N (read when a dump starts): keep dumping for N presents (up to 96 dumps in all), to catch an effect that is wrong in single frames
typedef struct { VkBuffer b; VkDeviceMemory m; void *p; uint32_t w, h; VkFormat f; uint64_t fence; int pass; uint64_t present; int rt; uint64_t k0, k1; int kind, nf; } pdump_t;   // k0 k1 kind: NVN2 diagnostic (the pass's program's first two shader keys; kind 1 the NDSR input)   // rt: r45 colour target number
enum { PD_MAX = 96 };
static pdump_t g_pd[PD_MAX]; static int g_npd, g_pd_state, g_pd_skip, g_pd_frames; static char g_pd_size[128], g_pd_from[32];   // state 0 off, 1 dumping this submission's passes, 2 waiting to write
static uint32_t g_pd_ndsr_w, g_pd_ndsr_h; static int g_pd_ndsr_tick = -1;
static int g_pd_stream, g_pd_serial; static vkimg_t *g_pd_ndsr_out;   // NVN2 diagnostic: DM_VK_PASSDUMP_SIZE=ndsr* streams statistics (vk_pass_dump_poll); the NDSR output image (ndsr_blit)   // NVN2 diagnostic: DM_VK_PASSDUMP_SIZE=ndsr follows the last NDSR input's size (ndsr_blit); the dump present its input was last taken at
static int g_pd_every, g_pd_perframe, g_pd_tick, g_pd_inframe;   // NVN2 diagnostic: DM_VK_PASSDUMP_EVERY=n (only every n-th present of the dump), DM_VK_PASSDUMP_PERFRAME=n (at most n passes a present)
#if defined(DM_NXN1X)
#include "gfx_vk_fg_capture.inc"
#include "gfx_vk_fg_auto.inc"
static int fg_motion_draw;
static uint64_t fg_pass_shader;
// A single-level, single-layer image may have an explicit base-subresource view.
static int fg_base_attachment(const view_t *v) {
    return !v || ((!v->set_levels || !v->base_level) && (!v->set_layers || !v->min_layer) &&
                  !v->set_target && !v->set_format && !v->set_swizzle && !v->set_dsmode);
}
static void fg_track_draw(const nvn_fg_profile *p) {
    if (S.sh[1] && S.sh[1]->key == p->fragment) fg_motion_draw = 1;
    if (S.sh[0] && S.sh[0]->key == p->vertex) {
        const p_bind_buf *u = &S.ubo[0][p->camera_ubo];
        float world[16], projection[16];
        const uint64_t bytes = (uint64_t)(p->camera_to_world > p->view_to_clip ? p->camera_to_world : p->view_to_clip) + sizeof world;
        extern int vk_gpu_pending(uint64_t, uint64_t);
        if (u->addr && bytes <= u->size && u->addr <= UINT64_MAX - bytes &&
            vk_pool_of(u->addr, bytes) && !vk_gpu_pending(u->addr, bytes)) {
            vk_ubd_read(u->addr + p->camera_to_world, sizeof world, (uint8_t *)world);
            vk_ubd_read(u->addr + p->view_to_clip, sizeof projection, (uint8_t *)projection);
            nvn_fg_camera(world, projection);
        }
    }
}
#if defined(DM_NXN1X)
static void ootdl_draw(void);   // ootdlss (below dispatch's helpers): this frame's scene camera
static int ootdl_fg_owner(void);   // ootdlss D2: DLSS runs and hands frame generation its camera and inputs (fg_auto stands aside)
#endif
static void fg_observe_draw(void) {
    ootd_draw();   // ootdlss D1a survey (DM_VK_OOTDLSS_SURVEY=1, log only)
#if defined(DM_NXN1X)
    ootdl_draw();
    if (g_wk) return;   // fgpar: a worker records only while ootdlss owns frame generation's inputs (seg_worker_refuse): fg_auto's tables stay the replay thread's
#endif
    if (fg_frames_left) fg_capture_uniforms(S.sh[0]?S.sh[0]->key:0,S.sh[1]?S.sh[1]->key:0);
    if (!nvn_fg_collecting()) return;
    const nvn_fg_profile *p=nvn_fg_get_profile();
    if (p) { fg_track_draw(p); return; }
#if defined(DM_NXN1X)
    if (ootdl_fg_owner()) return;   // ootdlss D2: a second camera in the same frame would make it "conflicting-cameras"
#endif
    fg_auto_draw();
    if (!fg_pass_shader && fg_auto.pass_camera && S.sh[1] &&
        S.sh[0] && S.sh[0]->key == fg_auto.shader)
        fg_pass_shader = S.sh[1]->key;
}
static void fg_submit_pair(const nvn_fg_profile *p, uint64_t shader, unsigned slot,
                           vkimg_t **actual, nvn_fg_image *pair, float factor) {
    float scale[2], jitter[2]={0,0};
    if (p) {scale[0]=p->mv_scale_x*factor; scale[1]=p->mv_scale_y*factor;}
    else {
        if (actual[0]->fmt!=VK_FORMAT_D16_UNORM && actual[0]->fmt!=VK_FORMAT_D32_SFLOAT &&
            actual[0]->fmt!=VK_FORMAT_D24_UNORM_S8_UINT && actual[0]->fmt!=VK_FORMAT_D32_SFLOAT_S8_UINT) return;
        ++fg_counts.usable_pairs;
        if (fg_auto.motion_shader!=shader || fg_auto.slot!=slot || fg_auto.width!=pair[0].width ||
            fg_auto.height!=pair[0].height || fg_auto.format!=actual[1]->fmt) {
            if (fg_auto.fence) return;
            fg_auto.motion_shader=shader; fg_auto.slot=slot; fg_auto.width=pair[0].width;
            fg_auto.height=pair[0].height; fg_auto.format=actual[1]->fmt; fg_auto.votes=0;
        }
        if (fg_auto.votes<3 || !(g_tex_frame%30)) fg_auto_sample(actual[0],actual[1]);
        if (fg_auto.votes<3 || !fg_auto.history) return;
        fg_motion_units(fg_auto.convention,fg_auto.width,fg_auto.height,&scale[0],&scale[1]);
        scale[0]*=fg_auto.width; scale[1]*=fg_auto.height;
    }
    VkCommandBuffer cb = vk_cmd();
    vk_img_barrier_all(cb);
    ++fg_counts.tags;
    nvn_fg_tag(cb, &pair[0], &pair[1], pair[0].width, pair[0].height, scale, jitter, 0, 0);
    vk_img_barrier_all(cb);
}
static int fg_velocity_format(VkFormat f) {
    return f == VK_FORMAT_R16G16_SFLOAT || f == VK_FORMAT_R16G16_SNORM || f == VK_FORMAT_R32G32_SFLOAT;
}
static void fg_profile_pass(void) {
    int matched = fg_motion_draw;
    fg_motion_draw = 0;
    // The UBO producer may be an auxiliary pass. Validate the depth-motion
    // pair against the camera numerically instead of requiring its allocation.
    int auto_pass=fg_auto.camera && fg_auto.frame == g_tex_frame + 1 && P_da.imageView &&
                  S.pass_w >= 320 && S.pass_w > S.pass_h &&
                  fabsf((float)S.pass_w / S.pass_h - fabsf(fg_auto.projection.v[5] /
                        fg_auto.projection.v[0])) < .12f;
    uint64_t shader=fg_pass_shader ? fg_pass_shader : (auto_pass && S.sh[1] ? S.sh[1]->key : 0);
    fg_auto.pass_camera=0; fg_pass_shader=0;
    if (!nvn_fg_collecting()) return;
#if defined(DM_NXN1X)
    if (g_wk || ootdl_fg_owner()) return;   // fgpar: fg_auto's poll and tags are the replay thread's, and stand aside while ootdlss hands DLSS-G its inputs
#endif
    fg_auto_poll();
    const nvn_fg_profile *p = nvn_fg_get_profile();
    if (!g_pass_draws || (p ? (!matched || !p->fragment) : (!auto_pass || !shader))) return;
    unsigned slot=8;
    if (p) slot=p->motion_attachment;
    else for (int i=0;i<S.ncol;++i) {
        vkimg_t *target=P_att[i].im;
        if (!target || !P_ca[i].imageView || !fg_velocity_format(target->fmt)) continue;
        if(slot!=8) return;
        slot=(unsigned)i;
    }
    if (!p && slot < 8) ++fg_counts.motion_passes;
    if (!p && slot == 8) {
        // Log distinct pass signatures, so loading screens cannot consume the whole budget.
        static struct { uint32_t w,h,n; VkFormat formats[9]; } seen[48];
        static unsigned count;
        if (count < 48) {
            typeof(seen[0]) entry = {S.pass_w,S.pass_h,S.ncol,{0}};
            for (int i=0; i<S.ncol; ++i) entry.formats[i]=S.cfmt[i];
            entry.formats[8]=S.dfmt;
            unsigned k=0;
            while (k<count && memcmp(&entry,&seen[k],sizeof entry)) ++k;
            if (k<count) return;
            seen[count++]=entry;
            VKLOG("FG camera-depth pass no velocity: %ux%u colors=%d VkFormats=%d,%d,%d,%d depth=%d",
                  S.pass_w, S.pass_h, S.ncol, S.cfmt[0], S.cfmt[1], S.cfmt[2], S.cfmt[3], S.dfmt);
        }
    }
    if (slot >= (unsigned)S.ncol ||
        P_ca[slot].storeOp != VK_ATTACHMENT_STORE_OP_STORE || P_da.storeOp != VK_ATTACHMENT_STORE_OP_STORE ||
        !P_ca[slot].imageView || !S.rt_has_col[slot] || !P_da.imageView || !S.rt_has_dep ||
        !fg_base_attachment(S.rt_has_dview ? &S.rt_dview : NULL) ||
        !fg_base_attachment(S.rt_has_cview[slot] ? &S.rt_cview[slot] : NULL) || P_ri.layerCount != 1) return;
    const tex_t *textures[2] = {&S.rt_dep, &S.rt_col[slot]};
    nvn_fg_image pair[2];
    vkimg_t *actual[2];
    for (unsigned i = 0; i < 2; ++i) {
        vkimg_t *im = P_att[i ? slot : 8].im;
        if (!im) return;
        VkImageView v;
        if (pass_scale == 2) {
            v = vk_rescaled_view(im, textures[i], NULL, 1, 0, 0);
            im = im->scaled;
        } else {
            vkimg_t *resolved = NULL;
            v = vk_tex_view(textures[i], NULL, 1, 0, VK_VIEW_SAMPLED, &resolved);
            if (resolved != im) return;
        }
        if (!im || !v || im->samples != 1 || im->levels != 1 || im->layers != 1 ||
            im->layout != VK_IMAGE_LAYOUT_GENERAL || !(im->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ||
            im->w != P_ri.renderArea.extent.width || im->h != P_ri.renderArea.extent.height ||
            P_ri.renderArea.offset.x || P_ri.renderArea.offset.y) return;
        pair[i] = (nvn_fg_image){im->img, im->mem, v, im->w, im->h, im->fmt, im->layout, im->usage, im->cflags, 0, 0};
        actual[i]=im;
    }
    fg_submit_pair(p, shader, slot, actual, pair, (float)pass_scale);
}
#endif
static void pass_dump_depth(void);
static void texdump_flush(void);
static void pass_dump_rt(int ci);
static int g_pkm7_on; static _Thread_local int g_pkm7_tr;   /* S1c (parallel recording): g_pkm7_tr per thread */   // pkm7 (diagnostic)
static void pass_dump(void) {
    texdump_flush();
    if (g_pd_state == 1 && g_sub_pass - 1 < g_pd_skip) return;   // (a submission with fewer passes dumps nothing, and the next one is tried)
    if (g_pd_state == 1 && g_pd_from[0]) { char wh[32]; snprintf(wh, sizeof wh, "%ux%u", S.pass_w, S.pass_h); if (strcmp(wh, g_pd_from)) return; g_pd_from[0] = 0; }
    if (!g_pd_size[0]) pass_dump_depth();
    if (g_pd_state != 1 || g_npd >= PD_MAX || !S.ncol || !S.rt_has_col[0]) return;
    vkimg_t *im = vk_tex(&S.rt_col[0]);
    if (im && g_pd_stream) {   // NVN2 diagnostic: =ndsr: targets of the current NDSR input's size (any format), any of 64x64 or smaller (an exposure / luminance chain, unverified) and the NDSR
        // output; =ndsrout: the NDSR output only (passes after the NDSR that draw over it)
        const int o = g_pd_ndsr_out && im == g_pd_ndsr_out;
        if (!(o || (!strcmp(g_pd_size, "ndsr") && ((im->w == g_pd_ndsr_w && im->h == g_pd_ndsr_h) || (im->w <= 64 && im->h <= 64))))) return;
    } else if (im && g_pd_size[0]) { char wh[32]; snprintf(wh, sizeof wh, "%ux%u", im->w, im->h); const char *f = strstr(g_pd_size, wh); size_t l = strlen(wh);
        if (!f || (f != g_pd_size && f[-1] != ',') || (f[l] && f[l] != ',')) return;
        const char *dd = getenv("DM_VK_PASSDUMP_DEPTH"); if (dd && atoi(dd)) pass_dump_depth(); }   // DM_VK_PASSDUMP_DEPTH=1: the size-filtered passes' depth too
    if (g_pd_every > 1 && g_pd_tick % g_pd_every) return;   // (read when a dump starts; the 96 dumps spread over the presents)
    if (g_pd_perframe > 0 && g_pd_inframe >= g_pd_perframe) return;
    g_pd_inframe++;
    pass_dump_rt(0);
    if (g_pd_stream && g_pd_ndsr_out && im == g_pd_ndsr_out && g_npd && g_pd[g_npd - 1].kind == 0) g_pd[g_npd - 1].kind = 4;   // NVN2 diagnostic: a pass that draws over the NDSR output
    { const char *a = getenv("DM_VK_PASSDUMP_ALL"); if (a && atoi(a)) for (int ci = 1; ci < S.ncol && ci < 8; ci++) pass_dump_rt(ci); }   // r45 DM_VK_PASSDUMP_ALL=1: every colour target of the pass
}
static void pass_dump_rt(int ci) {   // r45: colour target ci of the pass (pass_dump)
    if (g_npd >= PD_MAX || !S.rt_has_col[ci]) return;
    vkimg_t *im = vk_tex(&S.rt_col[ci]);
    if (!im || (im->fmt != VK_FORMAT_R8G8B8A8_UNORM && im->fmt != VK_FORMAT_R8G8B8A8_SRGB && im->fmt != VK_FORMAT_R16G16B16A16_SFLOAT && im->fmt != VK_FORMAT_B8G8R8A8_UNORM &&
                                                    im->fmt != VK_FORMAT_B10G11R11_UFLOAT_PACK32 && im->fmt != VK_FORMAT_R16_SFLOAT && im->fmt != VK_FORMAT_R8_UNORM && im->fmt != VK_FORMAT_A2B10G10R10_UNORM_PACK32 && im->fmt != VK_FORMAT_R32G32B32A32_SFLOAT)) return;   // pkm1: RGBA32F too
    const view_t *cv = S.rt_has_cview[ci] ? &S.rt_cview[ci] : NULL;   // the view's level and layer (a cube face of an array, a mip)
    uint32_t lv = cv && cv->set_levels ? cv->base_level : 0, ly = cv && cv->set_layers ? cv->min_layer : 0, w = im->w >> lv ? im->w >> lv : 1, h = im->h >> lv ? im->h >> lv : 1;
    if (lv >= im->levels || ly >= im->layers) lv = ly = 0, w = im->w, h = im->h;
    uint32_t bpp = im->fmt == VK_FORMAT_R32G32B32A32_SFLOAT ? 16 : im->fmt == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : im->fmt == VK_FORMAT_R16_SFLOAT ? 2 : im->fmt == VK_FORMAT_R8_UNORM ? 1 : 4; VkDeviceSize n = (VkDeviceSize)w * h * bpp;
    if (g_pkm7_on) VKLOG("pkm7: pass dump %02d = pass %d colour %d image %p (%ux%u fmt %d)", g_npd, g_sub_pass - 1, ci, (void *)im, w, h, im->fmt);
    pdump_t *d = &g_pd[g_npd++]; d->rt = 0; d->w = w; d->h = h; d->f = im->fmt; d->pass = g_sub_pass - 1; d->present = g_tex_frame; d->rt = ci;
    { uint8_t k[8]; d->k0 = d->k1 = 0; d->kind = 0; d->nf = im->nvn_format; d->fence = 0;   // NVN2 diagnostic: the shader keys of the program bound last in the pass
      for (unsigned s = 0; s < 2 && S.prog; s++) if (prog_key(S.prog, s, k)) { uint64_t v = 0; for (int q = 0; q < 8; q++) v = v << 8 | k[q]; if (s) d->k1 = v; else d->k0 = v; } }
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &d->b)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, d->b, &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0)};
    VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &d->m)); VKCHK(vkBindBufferMemory(V.dev, d->b, d->m, 0)); VKCHK(vkMapMemory(V.dev, d->m, 0, VK_WHOLE_SIZE, 0, &d->p));
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, lv, ly, 1}, {0, 0, 0}, {w, h, 1}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, d->b, 1, &c); vk_img_barrier_all(cb);
}
static int pd_capture_region(vkimg_t *im, int kind, pdump_t *d, int x, int y, uint32_t w, uint32_t h) {   // NVN2 diagnostic: level 0, layer 0 of an image into the dump (kind 1: the NDSR input at its dispatch); its statistics are logged when written
    if (!im || (im->fmt != VK_FORMAT_B10G11R11_UFLOAT_PACK32 && im->fmt != VK_FORMAT_R16G16B16A16_SFLOAT && im->fmt != VK_FORMAT_R8G8B8A8_UNORM && im->fmt != VK_FORMAT_R32G32B32A32_SFLOAT &&
                         im->fmt != VK_FORMAT_R8G8B8A8_SRGB && im->fmt != VK_FORMAT_B8G8R8A8_UNORM && im->fmt != VK_FORMAT_B8G8R8A8_SRGB && im->fmt != VK_FORMAT_A2B10G10R10_UNORM_PACK32)) return 0;
    const uint32_t bpp = im->fmt == VK_FORMAT_R32G32B32A32_SFLOAT ? 16 : im->fmt == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : 4; const VkDeviceSize n = (kind == 5 || kind == 6 ? 64 : (VkDeviceSize)w * h) * bpp;
    memset(d, 0, sizeof *d); d->w = w; d->h = h; d->f = im->fmt; d->pass = g_sub_pass - 1; d->present = g_tex_frame; d->kind = kind; d->nf = im->nvn_format;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(V.dev, &bi, NULL, &d->b) != VK_SUCCESS) return 0;
    VkMemoryRequirements rq;
    vkGetBufferMemoryRequirements(V.dev, d->b, &rq);
    uint32_t type = vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, type};
    if (type == UINT32_MAX || vkAllocateMemory(V.dev, &ai, NULL, &d->m) != VK_SUCCESS ||
        vkBindBufferMemory(V.dev, d->b, d->m, 0) != VK_SUCCESS ||
        vkMapMemory(V.dev, d->m, 0, VK_WHOLE_SIZE, 0, &d->p) != VK_SUCCESS) {
        vkDestroyBuffer(V.dev, d->b, NULL);
        if (d->m) vkFreeMemory(V.dev, d->m, NULL);
        memset(d, 0, sizeof *d);
        return 0;
    }
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, h, 1}};
    if (kind == 5 || kind == 6) {
        VkBufferImageCopy grid[64];
        for (unsigned i = 0; i < 64; ++i) {
            grid[i] = c;
            grid[i].bufferOffset = i * bpp;
            grid[i].imageOffset.x = x + (2 * (i % 8) + 1) * (uint64_t)w / 16;
            grid[i].imageOffset.y = y + (2 * (i / 8) + 1) * (uint64_t)h / 16;
            grid[i].imageExtent = (VkExtent3D){1, 1, 1};
        }
        vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, d->b, 64, grid);
        d->w = d->h = 8;
        d->fence = mtl_last_fence() + 1;
    } else vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, d->b, 1, &c);
    VkMemoryBarrier host = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0, NULL);
    vk_img_barrier_all(cb);
    return 1;
}
static pdump_t sr_probes[2];
static void pd_capture_img(vkimg_t *im, int kind) {
    if (im && g_npd < PD_MAX && pd_capture_region(im, kind, &g_pd[g_npd], 0, 0, im->w, im->h)) ++g_npd;
}
void vk_pd_present(vkimg_t *im) {   // NVN2 diagnostic (gfx_vk_present.c mtl_present): at each sampled present of a DM_VK_PASSDUMP_SIZE=ndsr* dump, the image handed to the screen (kind 3)
#if defined(DM_NXN1X)
    if (fg_frames_left && im) {
        void mtl_sub_begin(void); uint64_t mtl_sub_commit(void);
        const int own = !vk_sub_open();
        if (own) mtl_sub_begin();
        fg_capture_image(im->scaled_valid && im->scaled ? im->scaled : im, 0, 0, "present", 0, 0);
        if (own) mtl_sub_commit();
    }
#endif
    if (g_pd_state != 1 || !g_pd_stream || !im || (g_pd_every > 1 && g_pd_tick % g_pd_every)) return;
    void mtl_sub_begin(void); uint64_t mtl_sub_commit(void); const int op = !vk_sub_open(); if (op) mtl_sub_begin();
    pd_capture_img(im, 3); if (op) mtl_sub_commit();
}
static void texraw(const vkimg_t *im, uint32_t lv, uint32_t ly) {   // r44 DM_VK_TEXDUMP of an ASTC texture: the guest's blocks of that level and layer, deswizzled as the upload does, as a .astc file for a CPU decoder
    static const uint8_t bd[14][2] = {{4, 4}, {5, 4}, {5, 5}, {6, 5}, {6, 6}, {8, 5}, {8, 6}, {8, 8}, {10, 5}, {10, 6}, {10, 8}, {10, 10}, {12, 10}, {12, 12}};
    if (!im->pkg_mem || im->fmt < VK_FORMAT_ASTC_4x4_UNORM_BLOCK || im->fmt > VK_FORMAT_ASTC_12x12_SRGB_BLOCK) { VKLOG("texraw: no guest blocks (packaged %d, format %d)", im->packaged, im->fmt); return; }
    int k = (im->fmt - VK_FORMAT_ASTC_4x4_UNORM_BLOCK) / 2; uint32_t w = im->w >> lv ? im->w >> lv : 1, h = im->h >> lv ? im->h >> lv : 1, bx = (w + bd[k][0] - 1) / bd[k][0], by = (h + bd[k][1] - 1) / bd[k][1];
    size_t pitch = (size_t)bx * 16; uint8_t *buf = malloc(pitch * by); if (!buf) return;
    if (nvn_deswizzle(&im->pkg_d, lv, ly, im->pkg_mem, buf, pitch)) { VKLOG("texraw: deswizzle refused"); free(buf); return; }
    const char *root = getenv("DM_ROOT"); char path[1024]; snprintf(path, sizeof path, "%s/texraw_L%02u_M%u.astc", root ? root : ".", ly, lv);
    uint8_t hd[16] = {0x13, 0xab, 0xa1, 0x5c, bd[k][0], bd[k][1], 1, (uint8_t)w, (uint8_t)(w >> 8), (uint8_t)(w >> 16), (uint8_t)h, (uint8_t)(h >> 8), (uint8_t)(h >> 16), 1, 0, 0};
    FILE *f = fopen(path, "wb"); if (f) { fwrite(hd, 1, 16, f); fwrite(buf, 1, pitch * by, f); fclose(f); }
    VKLOG("texraw: layer %u level %u %ux%u, %ux%u blocks (%s) → %s", ly, lv, w, h, bd[k][0], bd[k][1], im->fmt & 1 ? "UNORM" : "sRGB", path); free(buf);
}
static void texdump_flush(void) {   // DM_VK_TEXDUMP: outside a render pass (pass_dump runs after its end)
    vkimg_t *im = g_td_im; if (!im || g_pd_state != 1) return; g_td_im = NULL; g_td_done = 1;
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, im->fmt, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT)) { VKLOG("texdump: format %d cannot be blitted", im->fmt); return; }
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    const char *le = getenv("DM_VK_TEXDUMP_LAYER"), *ve = getenv("DM_VK_TEXDUMP_LEVEL"); uint32_t l0 = le ? (uint32_t)atoi(le) : 0, lv = ve ? (uint32_t)atoi(ve) : 0;   // r44: first layer, level
    if (lv >= im->levels) lv = im->levels - 1; uint32_t tw = im->w >> lv ? im->w >> lv : 1, th = im->h >> lv ? im->h >> lv : 1;
    for (uint32_t ly = l0; ly < im->layers && ly < l0 + 6 && g_npd < PD_MAX; ly++) {
        texraw(im, lv, ly);
        VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT, {tw, th, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT,
            VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        VkImage ti; VKCHK(vkCreateImage(V.dev, &ic, NULL, &ti)); VkMemoryRequirements rq; vkGetImageMemoryRequirements(V.dev, ti, &rq);
        VkMemoryAllocateInfo ma = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
        VkDeviceMemory tm; VKCHK(vkAllocateMemory(V.dev, &ma, NULL, &tm)); VKCHK(vkBindImageMemory(V.dev, ti, tm, 0));
        VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, ti, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
        VkImageBlit r = {{VK_IMAGE_ASPECT_COLOR_BIT, lv, ly, 1}, {{0, 0, 0}, {(int32_t)tw, (int32_t)th, 1}}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {(int32_t)tw, (int32_t)th, 1}}};
        vkCmdBlitImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, ti, VK_IMAGE_LAYOUT_GENERAL, 1, &r, VK_FILTER_NEAREST); vk_img_barrier_all(cb);
        pdump_t *d = &g_pd[g_npd++]; d->rt = 0; d->w = tw; d->h = th; d->f = VK_FORMAT_R16G16B16A16_SFLOAT; d->pass = -100 - (int)(ly - l0); d->present = g_tex_frame;
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, (VkDeviceSize)tw * th * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &d->b)); vkGetBufferMemoryRequirements(V.dev, d->b, &rq);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0)};
        VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &d->m)); VKCHK(vkBindBufferMemory(V.dev, d->b, d->m, 0)); VKCHK(vkMapMemory(V.dev, d->m, 0, VK_WHOLE_SIZE, 0, &d->p));
        VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {tw, th, 1}};
        vkCmdCopyImageToBuffer(cb, ti, VK_IMAGE_LAYOUT_GENERAL, d->b, 1, &c); vk_img_barrier_all(cb);
        vk_defer_free_image(ti, tm);
    }
    VKLOG("texdump: %ux%u fmt %d layers %u levels %u at pool %#llx+%#llx, level %u layers from %u: dumps with pass -100 - (layer - first)", im->w, im->h, im->fmt, im->layers, im->levels, (unsigned long long)im->pool, (unsigned long long)im->off, lv, l0);
}
static void pass_dump_depth(void) {   // the pass's depth (D24S8 / D32 / D16) as a pdump_t with f = the depth format; written as a grayscale stretched between its min and max
    if (g_pd_state != 1 || g_npd >= PD_MAX || !S.rt_has_dep) return;
    vkimg_t *im = vk_tex(&S.rt_dep); if (!im || !(im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) return;
    uint32_t bpp = im->fmt == VK_FORMAT_D16_UNORM ? 2 : 4; VkDeviceSize n = (VkDeviceSize)im->w * im->h * bpp;
    pdump_t *d = &g_pd[g_npd++]; d->rt = 0; d->w = im->w; d->h = im->h; d->f = im->fmt; d->pass = g_sub_pass - 1; d->present = g_tex_frame;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &d->b)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, d->b, &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0)};
    VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &d->m)); VKCHK(vkBindBufferMemory(V.dev, d->b, d->m, 0)); VKCHK(vkMapMemory(V.dev, d->m, 0, VK_WHOLE_SIZE, 0, &d->p));
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1}, {0, 0, 0}, {im->w, im->h, 1}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, d->b, 1, &c); vk_img_barrier_all(cb);
}
static inline float half2f(uint16_t h) { uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023, f;
    if (!e) { if (!m) f = s; else { e = 1; while (!(m & 1024)) { m <<= 1; e--; } m &= 1023; f = s | ((e + 112) << 23) | (m << 13); } } else if (e == 31) f = s | 0x7f800000 | (m << 13); else f = s | ((e + 112) << 23) | (m << 13);
    float r; memcpy(&r, &f, 4); return r; }
// r37 DM_VK_EXPO_LOG=N (live, 0 = off): every N presents, the first colour target of each 1x1 pass of the next frame (up to 8: the auto-exposure results) is copied
// and logged 10 presents later as "expo:" lines (values as floats)
typedef struct { VkBuffer b; VkDeviceMemory m; void *p; VkFormat f; int pass; uint32_t lv; uint64_t present; } expo_t;
static expo_t g_ex[8]; static int g_nex, g_ex_state; static uint64_t g_ex_at;
static void expo_grab(void) {
    if (g_ex_state != 1 || g_nex >= 8 || S.pass_w != 1 || S.pass_h != 1 || !S.ncol || !S.rt_has_col[0]) return;
    vkimg_t *im = vk_tex(&S.rt_col[0]); if (!im) return;
    uint32_t bpp = im->fmt == VK_FORMAT_R32G32B32A32_SFLOAT ? 16 : im->fmt == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : im->fmt == VK_FORMAT_R16_SFLOAT ? 2 :
                   im->fmt == VK_FORMAT_R32_SFLOAT || im->fmt == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || im->fmt == VK_FORMAT_R8G8B8A8_UNORM || im->fmt == VK_FORMAT_R16G16_SFLOAT ? 4 : 0;
    if (!bpp) { VKLOG("expo: present %llu pass %d: format %d not decoded", (unsigned long long)g_tex_frame, g_sub_pass - 1, im->fmt); return; }
    const view_t *cv = S.rt_has_cview[0] ? &S.rt_cview[0] : NULL;
    uint32_t lv = cv && cv->set_levels ? cv->base_level : 0, ly = cv && cv->set_layers ? cv->min_layer : 0;
    if (lv >= im->levels || ly >= im->layers) lv = ly = 0;
    expo_t *e = &g_ex[g_nex++]; e->f = im->fmt; e->pass = g_sub_pass - 1; e->lv = lv; e->present = g_tex_frame;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &e->b)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, e->b, &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0)};
    VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &e->m)); VKCHK(vkBindBufferMemory(V.dev, e->b, e->m, 0)); VKCHK(vkMapMemory(V.dev, e->m, 0, VK_WHOLE_SIZE, 0, &e->p)); memset(e->p, 0, 16);
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, lv, ly, 1}, {0, 0, 0}, {1, 1, 1}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, e->b, 1, &c); vk_img_barrier_all(cb);
}
static void expo_poll(void) {
    const char *ev = getenv("DM_VK_EXPO_LOG"); int every = ev ? atoi(ev) : 0;
    if (g_ex_state == 0) { if (every > 0 && g_tex_frame % (uint64_t)every == 0) g_ex_state = 1, g_nex = 0; return; }
    if (g_ex_state == 1) { g_ex_state = 2; g_ex_at = g_tex_frame + 10; return; }
    if (g_tex_frame < g_ex_at) return;
    VKLOG("expo: present %llu: %d 1x1 passes", (unsigned long long)(g_nex ? g_ex[0].present : g_tex_frame), g_nex);
    for (int i = 0; i < g_nex; i++) {
        expo_t *e = &g_ex[i]; const uint8_t *b = e->p; float v[4] = {0}; int nv = 4;
        if (e->f == VK_FORMAT_R32G32B32A32_SFLOAT) memcpy(v, b, 16);
        else if (e->f == VK_FORMAT_R16G16B16A16_SFLOAT) { for (int c = 0; c < 4; c++) { uint16_t h; memcpy(&h, b + 2 * c, 2); v[c] = half2f(h); } }
        else if (e->f == VK_FORMAT_R16G16_SFLOAT) { for (int c = 0; c < 2; c++) { uint16_t h; memcpy(&h, b + 2 * c, 2); v[c] = half2f(h); } nv = 2; }
        else if (e->f == VK_FORMAT_R16_SFLOAT) { uint16_t h; memcpy(&h, b, 2); v[0] = half2f(h); nv = 1; }
        else if (e->f == VK_FORMAT_R32_SFLOAT) { memcpy(v, b, 4); nv = 1; }
        else if (e->f == VK_FORMAT_B10G11R11_UFLOAT_PACK32) { uint32_t w; memcpy(&w, b, 4); v[0] = half2f((uint16_t)((w & 0x7ff) << 4)); v[1] = half2f((uint16_t)((w >> 11 & 0x7ff) << 4)); v[2] = half2f((uint16_t)((w >> 22 & 0x3ff) << 5)); nv = 3; }
        else { for (int c = 0; c < 4; c++) v[c] = b[c] / 255.f; }
        VKLOG("expo:   pass %d level %u format %d: %.6g %.6g %.6g %.6g (%d values)", e->pass, e->lv, e->f, v[0], v[1], v[2], v[3], nv);
        vkDestroyBuffer(V.dev, e->b, NULL); vkFreeMemory(V.dev, e->m, NULL);
    }
    g_nex = 0; g_ex_state = 0;
}
// r41 DM_VK_IND_LOG=N (live, 0 = off): every N presents, the indexed indirect draws of the next frame (up to 16; DM_VK_IND_KEY=<vertex key> only those) have their
// arguments and index data copied at the end of their pass and logged 10 presents later as "ind:" lines. Donkey Kong Bananza: does the triangle list the terrain
// geometry shader 5d537f76 appends (read by the colour pass d6614139) hold each triangle once, three times (Turnip runs a geometry shader once per output vertex), or garbage?
typedef struct { VkBuffer ab, ib; VkDeviceSize ao, io; uint32_t isz, win, pass; uint64_t key, gind, gib; VkBuffer b; VkDeviceMemory m; void *p; } ind_t;
static ind_t g_in[256]; static int g_nin, g_in_copied, g_in_state; static uint64_t g_in_at, g_in_present;
static void ind_note(const p_draw_idx *d, uint32_t isz) {
    uint64_t vkey = S.sh[0] ? S.sh[0]->key : 0;   // (NXN1X Eden mode: the program's first key when the vertex stage has no vkshader)
    if (!vkey && S.prog) { uint8_t k8[8]; if (prog_key(S.prog, 0, k8)) for (int i = 0; i < 8; i++) vkey = vkey << 8 | k8[i]; }
    if (g_in_state != 1 || g_nin >= 256 || !vkey) return;
    const char *k = getenv("DM_VK_IND_KEY"); uint64_t want = k ? strtoull(k, NULL, 16) : 0; if (want && vkey != want) return;
    ind_t *e = &g_in[g_nin]; memset(e, 0, sizeof *e);
    if (!vk_buf_at(d->ind, 20, &e->ab, &e->ao)) return;
    static const uint32_t wins[] = {65536, 16384, 4096, 1024}; for (int i = 0; i < 4 && !e->win && g_nin < 16; i++) if (vk_buf_at(d->ibuf, wins[i], &e->ib, &e->io)) e->win = wins[i];
    if (!e->win && g_nin < 16) return;
    e->isz = isz; e->pass = (uint32_t)(g_sub_pass - 1); e->key = vkey; e->gind = d->ind; e->gib = d->ibuf; g_nin++;
}
// r144 (diagnostic): with DM_VK_IND_LOG the indirect quads (r62 ixc) are noted too (isz 0): the game's record {count, instances, first, base instance} and the
// converted VkDrawIndexedIndirectCommand are copied at the pass end like the indexed ones and logged as "r144:" lines (Hyrule Warriors: Age of Imprisonment's lamp
// glows: their indirect quad passes change no pixel; record count 0 = the effect compute, nonzero = the conversion or the draw)
static void r144_note(uint64_t ind, VkBuffer ob, VkDeviceSize oo, int ok) {
    if (g_in_state != 1 || g_nin >= 256 || !S.sh[0]) return;
    const char *k = getenv("DM_VK_IND_KEY"); uint64_t want = k ? strtoull(k, NULL, 16) : 0; if (want && S.sh[0]->key != want) return;
    ind_t *e = &g_in[g_nin]; memset(e, 0, sizeof *e);
    if (!vk_buf_at(ind, 20, &e->ab, &e->ao) && !vk_buf_at(ind, 16, &e->ab, &e->ao)) return;
    e->ib = ob; e->io = oo; e->win = 20; e->isz = 0; e->pass = (uint32_t)(g_sub_pass - 1); e->key = S.sh[0]->key; e->gind = ind; e->gib = (uint64_t)ok; g_nin++;
}
static void ind_grab(void) {   // vk_end_pass: copy what this pass's noted draws read (outside the pass, after a full barrier)
    if (g_in_state != 1 || g_in_copied >= g_nin) return;
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    for (; g_in_copied < g_nin; g_in_copied++) { ind_t *e = &g_in[g_in_copied];
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 32 + e->win, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &e->b)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, e->b, &rq);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0)};
        VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &e->m)); VKCHK(vkBindBufferMemory(V.dev, e->b, e->m, 0)); VKCHK(vkMapMemory(V.dev, e->m, 0, VK_WHOLE_SIZE, 0, &e->p)); memset(e->p, 0xee, 32 + e->win);
        VkBufferCopy c0 = {e->ao, 0, 20}, c1 = {e->io, 32, e->win};
        vkCmdCopyBuffer(cb, e->ab, e->b, 1, &c0); if (e->win) vkCmdCopyBuffer(cb, e->ib, e->b, 1, &c1); }
    vk_img_barrier_all(cb);
}
static int ind_cmp(const void *a, const void *b) { const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static void ind_poll(void) {
    const char *ev = getenv("DM_VK_IND_LOG"); int every = ev ? atoi(ev) : 0;
    if (g_in_state == 0) { if (every > 0 && g_tex_frame % (uint64_t)every == 0) g_in_state = 1, g_nin = g_in_copied = 0, g_in_present = g_tex_frame; return; }
    if (g_in_state == 1) { g_in_state = 2; g_in_at = g_tex_frame + 10; return; }
    if (g_tex_frame < g_in_at) return;
    VKLOG("ind: present %llu: %d indexed indirect draws (%d copied)", (unsigned long long)g_in_present, g_nin, g_in_copied);
    uint32_t cmax = 0, cbig = 0; char cl[700]; int cn = 0;
    for (int i = 0; i < g_in_copied; i++) {
        ind_t *e = &g_in[i]; const uint32_t *a = e->p; const uint8_t *ix = (const uint8_t *)e->p + 32;
        if (a[0] > cmax) cmax = a[0]; if (a[0] > 7520) cbig++;
        cn += snprintf(cl + cn, sizeof cl - cn, " %u", a[0]); if (cn > 600 || i + 1 == g_in_copied) { VKLOG("ind: counts to draw %d:%s", i, cl); cn = 0; }
        if (!e->isz) { const uint32_t *o = (const uint32_t *)((const uint8_t *)e->p + 32);   // r144: an indirect quad
            VKLOG("r144: pass %u vertex %016llx quad record at %#llx: count %u instances %u first %u base instance %u -> indexed count %u instances %u first index %u vertex offset %d first instance %u%s",
                  e->pass, (unsigned long long)e->key, (unsigned long long)e->gind, a[0], a[1], a[2], a[3], o[0], o[1], o[2], (int32_t)o[3], o[4], e->gib ? "" : " (draw not allowed)");
            vkDestroyBuffer(V.dev, e->b, NULL); vkFreeMemory(V.dev, e->m, NULL); continue; }
        if (!e->win) { vkDestroyBuffer(V.dev, e->b, NULL); vkFreeMemory(V.dev, e->m, NULL); continue; }
        uint32_t cnt = a[0], inst = a[1], first = a[2], fin = a[4]; int32_t vo = (int32_t)a[3];
        VKLOG("ind:   pass %u vertex %016llx args at %#llx: count %u instances %u first %u vertex offset %d first instance %u; index buffer %#llx (%u-byte indices)",
              e->pass, (unsigned long long)e->key, (unsigned long long)e->gind, cnt, inst, first, vo, fin, (unsigned long long)e->gib, e->isz);
        uint64_t avail = e->win / e->isz; uint64_t n = cnt; if ((uint64_t)first >= avail) n = 0; else if (first + n > avail) n = avail - first;
        uint32_t ntri = (uint32_t)(n / 3), mx = 0, zero = 0, ee = 0; uint64_t *t = ntri ? malloc(sizeof *t * ntri) : NULL; char s[600]; int k = 0;
        for (uint32_t q = 0; q < ntri; q++) { uint32_t v[3];
            for (int c = 0; c < 3; c++) { uint64_t j = first + (uint64_t)q * 3 + c; v[c] = e->isz == 1 ? ix[j] : e->isz == 2 ? ((const uint16_t *)ix)[j] : ((const uint32_t *)ix)[j]; if (v[c] > mx) mx = v[c]; }
            if (!v[0] && !v[1] && !v[2]) zero++; if (v[0] == 0xeeee || v[0] == 0xeeeeeeee) ee++;
            uint32_t a0 = v[0] < v[1] ? v[0] : v[1], a1 = v[0] < v[1] ? v[1] : v[0], a2 = v[2]; if (a2 < a0) { uint32_t x = a0; a0 = a2; a2 = a1; a1 = x; } else if (a2 < a1) { uint32_t x = a1; a1 = a2; a2 = x; }
            t[q] = (uint64_t)a0 << 42 | (uint64_t)a1 << 21 | a2;
            if (q < 12) k += snprintf(s + k, sizeof s - k, " %u,%u,%u", v[0], v[1], v[2]); }
        uint32_t uniq = 0, x1 = 0, x2 = 0, x3 = 0, xm = 0;
        if (t) { qsort(t, ntri, sizeof *t, ind_cmp); for (uint32_t q = 0; q < ntri;) { uint32_t r = q; while (r < ntri && t[r] == t[q]) r++; uniq++; uint32_t m = r - q; if (m == 1) x1++; else if (m == 2) x2++; else if (m == 3) x3++; else xm++; q = r; } free(t); }
        VKLOG("ind:     %u of %u indices in the window; %u triangles, %u distinct (once %u, twice %u, three times %u, more %u), all-zero %u, unwritten %u, max index %u; first:%s",
              (unsigned)n, cnt, ntri, uniq, x1, x2, x3, xm, zero, ee, mx, s);
        vkDestroyBuffer(V.dev, e->b, NULL); vkFreeMemory(V.dev, e->m, NULL);
    }
    VKLOG("ind: largest count %u, %u draws above 7520 indices", cmax, cbig);
    g_nin = g_in_copied = 0; g_in_state = 0;
}
static void pd_stats(int i, const pdump_t *d) {   // NVN2 diagnostic: per channel min / max / mean (finite values) and NaN / Inf counts of a colour dump, into the log
    const size_t np = (size_t)d->w * d->h; int nc = 4; double mn[4], mx[4], sum[4]; uint64_t nan = 0, inf = 0, cnt[4] = {0, 0, 0, 0};
    for (int c = 0; c < 4; c++) { mn[c] = 1e30; mx[c] = -1e30; sum[c] = 0; }
    if (d->f == VK_FORMAT_B10G11R11_UFLOAT_PACK32) nc = 3; else if (d->f == VK_FORMAT_R16_SFLOAT || d->f == VK_FORMAT_R8_UNORM) nc = 1;
    for (size_t k = 0; k < np; k++) { float v[4] = {0, 0, 0, 0};
        switch (d->f) {
        case VK_FORMAT_R32G32B32A32_SFLOAT: memcpy(v, (const float *)d->p + 4 * k, 16); break;
        case VK_FORMAT_R16G16B16A16_SFLOAT: { const uint16_t *p = (const uint16_t *)d->p + 4 * k; for (int c = 0; c < 4; c++) v[c] = half2f(p[c]); } break;
        case VK_FORMAT_R16_SFLOAT: v[0] = half2f(((const uint16_t *)d->p)[k]); break;
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: { const uint32_t w = ((const uint32_t *)d->p)[k];   // 11-bit and 10-bit floats as halves (an all-ones exponent stays Inf / NaN)
            v[0] = half2f((uint16_t)((w & 0x7ff) << 4)); v[1] = half2f((uint16_t)((w >> 11 & 0x7ff) << 4)); v[2] = half2f((uint16_t)((w >> 22 & 0x3ff) << 5)); } break;
        case VK_FORMAT_R8_UNORM: v[0] = ((const uint8_t *)d->p)[k] / 255.f; break;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: { const uint32_t w = ((const uint32_t *)d->p)[k]; for (int c = 0; c < 3; c++) v[c] = (float)(w >> (10 * c) & 0x3ff) / 1023.f; v[3] = (float)(w >> 30) / 3.f; } break;
        default: { const uint8_t *p = (const uint8_t *)d->p + 4 * k; for (int c = 0; c < 4; c++) v[c] = p[c] / 255.f; } break;   // RGBA8 / BGRA8, channels as stored
        }
        for (int c = 0; c < nc; c++) { const float x = v[c]; if (x != x) { nan++; continue; } if (isinf(x)) { inf++; continue; } if (x < mn[c]) mn[c] = x; if (x > mx[c]) mx[c] = x; sum[c] += x; cnt[c]++; } }
    char s[256]; int p = 0; s[0] = 0;
    for (int c = 0; c < nc; c++) p += snprintf(s + p, sizeof s - (size_t)p, " ch%d %.4g/%.4g/%.4g", c, cnt[c] ? mn[c] : 0, cnt[c] ? mx[c] : 0, cnt[c] ? sum[c] / (double)cnt[c] : 0);
    VKLOG("pass dump %02d stats: %s, present %llu pass %d colour %d, %ux%u vk format %d nvn %#x, shaders %016llx %016llx; min/max/mean%s; NaN %llu, Inf %llu", i, d->kind == 5 ? "OP_DLSS input color (8x8 grid)" : d->kind == 6 ? "OP_DLSS output (8x8 grid)" : d->kind == 1 ? "NDSR input" : d->kind == 2 ? "NDSR output after the blit / GSR1" : d->kind == 3 ? "present source" : d->kind == 4 ? "pass over the NDSR output" : "pass target",
          (unsigned long long)d->present, d->pass, d->rt, d->w, d->h, d->f, (unsigned)d->nf, (unsigned long long)d->k0, (unsigned long long)d->k1, s, (unsigned long long)nan, (unsigned long long)inf);
}
void vk_pass_dump_poll(uint64_t done_fence) {   // gfx_vk_present.c, every present: start on the switch, write once the GPU finished
#if defined(DM_NXN1X)
    for (unsigned i = 0; i < 2; ++i) {
        pdump_t *d = &sr_probes[i];
        if (d->b && d->fence <= done_fence) {
            pd_stats(i, d);
            vkUnmapMemory(V.dev, d->m);
            vkDestroyBuffer(V.dev, d->b, NULL);
            vkFreeMemory(V.dev, d->m, NULL);
            memset(d, 0, sizeof *d);
        }
    }
    fg_capture_poll(done_fence);
    if (nvn_fg_collecting() && !nvn_fg_get_profile()) {
        static unsigned interval;
        if (++interval == 60) {
            VKLOG("FG discovery (60 presents): draws=%u depth_draws=%u UBOs=%u camera=%u motion_candidates=%u usable_pairs=%u readbacks=%u useful_texels=%u calibration=%u/3 tag_attempts=%u",
                  fg_counts.draws, fg_counts.depth_draws, fg_counts.buffers, fg_counts.cameras,
                  fg_counts.motion_passes, fg_counts.usable_pairs, fg_counts.samples, fg_counts.useful,
                  fg_auto.votes, fg_counts.tags);
            memset(&fg_counts, 0, sizeof fg_counts);
            interval = 0;
        }
    }
#endif
    expo_poll(); ind_poll();
    const char *e = getenv("DM_VK_PASSDUMP");
    if (g_pd_state == 0 && e && atoi(e) > 1) { char n[16]; snprintf(n, sizeof n, "%d", atoi(e) - 1); setenv("DM_VK_PASSDUMP", n, 1); return; }   // =N: the frame after N - 1 more presents
    if (g_pd_state == 0 && e && atoi(e)) { const char *sk = getenv("DM_VK_PASSDUMP_SKIP"); g_pd_skip = sk && !strchr(sk, 'x') ? atoi(sk) : 0;
        snprintf(g_pd_from, sizeof g_pd_from, "%s", sk && strchr(sk, 'x') ? sk : "");
        const char *sz = getenv("DM_VK_PASSDUMP_SIZE"); snprintf(g_pd_size, sizeof g_pd_size, "%s", sz && strcmp(sz, "0") ? sz : "");
        { const char *td = getenv("DM_VK_TEXDUMP"); const char *pl = td ? strchr(td, '+') : NULL; g_td_pool = pl ? strtoull(td, NULL, 16) : 0; g_td_off = pl ? strtoll(pl + 1, NULL, 16) : -1; g_td_im = NULL; g_td_done = 0; }
        const char *fr = getenv("DM_VK_PASSDUMP_FRAMES"); g_pd_frames = fr && atoi(fr) > 1 ? atoi(fr) : 1;
        { const char *ev = getenv("DM_VK_PASSDUMP_EVERY"), *pf = getenv("DM_VK_PASSDUMP_PERFRAME"); g_pd_every = ev ? atoi(ev) : 0; g_pd_perframe = pf ? atoi(pf) : 0; g_pd_tick = g_pd_inframe = 0;
          g_pd_stream = !strncmp(g_pd_size, "ndsr", 4); g_pd_serial = 0; }
        g_pd_state = 1; g_npd = 0; setenv("DM_VK_PASSDUMP", "0", 1); VKLOG("pass dump: the next submission's passes from pass %d", g_pd_skip); return; }
    if (g_pd_state == 1) { g_pd_tick++; g_pd_inframe = 0; }   // NVN2 diagnostic (DM_VK_PASSDUMP_EVERY / _PERFRAME)
    if ((g_pd_state == 1 || g_pd_state == 3) && g_pd_stream) {   // NVN2 diagnostic (DM_VK_PASSDUMP_SIZE=ndsr*): each dump's statistics as soon as its GPU work is done, its buffer freed: no ppm,
        // no 96-dump window (Donkey Kong Bananza 11-16: the window ended before the black and the end-of-window writing stalled a frame 1.8 s); FRAMES presents or until the close
        int k = 0; const uint64_t lf = mtl_last_fence() + 1;
        for (int i = 0; i < g_npd; i++) { pdump_t *d = &g_pd[i];
            if (d->fence && done_fence >= d->fence) { pd_stats(g_pd_serial++, d); vkDestroyBuffer(V.dev, d->b, NULL); vkFreeMemory(V.dev, d->m, NULL); continue; }
            if (!d->fence) d->fence = lf;
            if (k != i) g_pd[k] = *d;
            k++; }
        g_npd = k;
        if (g_pd_state == 1 && --g_pd_frames <= 0) { g_pd_state = 3; VKLOG("pass dump: the window ends (%d dumps so far)", g_pd_serial); }
        if (g_pd_state == 3 && !g_npd) { g_pd_state = 0; VKLOG("pass dump: done, %d dumps", g_pd_serial); }
        return;
    }
    if (g_pd_state == 1 && g_npd && --g_pd_frames > 0 && g_npd < PD_MAX - 8) return;   // DM_VK_PASSDUMP_FRAMES=N: N presents in a row (with DM_VK_PASSDUMP_SIZE)
    if (g_pd_state == 1 && g_npd) { g_pd_state = 2; for (int i = 0; i < g_npd; i++) g_pd[i].fence = mtl_last_fence() + 1; return; }
    if (g_pd_state != 2 || !g_npd || done_fence < g_pd[g_npd - 1].fence) return;
    const char *root = getenv("DM_ROOT");
    for (int i = 0; i < g_npd; i++) {
        pdump_t *d = &g_pd[i]; char path[1024]; snprintf(path, sizeof path, "%s/pass_%02d.ppm", root ? root : ".", i);
        VKLOG("pass dump %02d is the submission's pass %d colour target %d (present %llu)", i, d->pass, d->rt, (unsigned long long)d->present);
        FILE *f = fopen(path, "wb"); if (!f) continue; fprintf(f, "P6\n%u %u\n255\n", d->w, d->h);
        uint8_t *row = malloc((size_t)d->w * 3); uint64_t bad = 0; float mx = 0, mn = 0;
        if (d->f == VK_FORMAT_D24_UNORM_S8_UINT || d->f == VK_FORMAT_D32_SFLOAT || d->f == VK_FORMAT_D32_SFLOAT_S8_UINT || d->f == VK_FORMAT_D16_UNORM || d->f == VK_FORMAT_X8_D24_UNORM_PACK32) {
            size_t np = (size_t)d->w * d->h; float lo = 1e30f, hi = -1e30f; float *v = malloc(np * sizeof *v);
            for (size_t k = 0; k < np; k++) { float x;
                if (d->f == VK_FORMAT_D16_UNORM) x = ((const uint16_t *)d->p)[k] / 65535.f;
                else if (d->f == VK_FORMAT_D32_SFLOAT || d->f == VK_FORMAT_D32_SFLOAT_S8_UINT) memcpy(&x, (const uint8_t *)d->p + 4 * k, 4);
                else x = (((const uint32_t *)d->p)[k] & 0xffffff) / 16777215.f;
                v[k] = x; if (x < lo) lo = x; if (x > hi) hi = x; }
            for (uint32_t y = 0; y < d->h; y++) { for (uint32_t x = 0; x < d->w; x++) { float t = hi > lo ? (v[(size_t)y * d->w + x] - lo) / (hi - lo) : 0; row[3 * x] = row[3 * x + 1] = row[3 * x + 2] = (uint8_t)(t * 255); } fwrite(row, 1, (size_t)d->w * 3, f); }
            { char rp[1100]; snprintf(rp, sizeof rp, "%s.f32", path); FILE *rf = fopen(rp, "wb"); if (rf) { fwrite(v, sizeof *v, np, rf); fclose(rf); } }   // the values themselves
            free(v); free(row); fclose(f); VKLOG("pass dump %02d: depth %ux%u format %d, values %.6f .. %.9g → %s", i, d->w, d->h, d->f, lo, hi, path);
            vkDestroyBuffer(V.dev, d->b, NULL); vkFreeMemory(V.dev, d->m, NULL); continue;
        }
        pd_stats(i, d);   // NVN2 diagnostic
        for (uint32_t y = 0; y < d->h; y++) { for (uint32_t x = 0; x < d->w; x++) {
            if (d->f == VK_FORMAT_R32G32B32A32_SFLOAT) { const float *p = (const float *)d->p + ((size_t)y * d->w + x) * 4;   // pkm1
                for (int c = 0; c < 3; c++) { float v = p[c]; if (v != v || v > 3e38f || v < -3e38f) { bad++; v = 1; } if (v > mx) mx = v; if (v < mn) mn = v; row[3 * x + c] = (uint8_t)(v <= 0 ? 0 : v >= 1 ? 255 : v * 255 + .5f); } }
            else if (d->f == VK_FORMAT_R16G16B16A16_SFLOAT || d->f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || d->f == VK_FORMAT_R16_SFLOAT) { float v3[3];
                if (d->f == VK_FORMAT_R16G16B16A16_SFLOAT) { const uint16_t *p = (const uint16_t *)d->p + ((size_t)y * d->w + x) * 4; for (int c = 0; c < 3; c++) v3[c] = half2f(p[c]); }
                else if (d->f == VK_FORMAT_R16_SFLOAT) v3[0] = v3[1] = v3[2] = half2f(((const uint16_t *)d->p)[(size_t)y * d->w + x]);
                else { uint32_t w = ((const uint32_t *)d->p)[(size_t)y * d->w + x];   // 11-bit and 10-bit floats: a half's exponent and top mantissa bits
                    v3[0] = half2f((uint16_t)((w & 0x7ff) << 4)); v3[1] = half2f((uint16_t)((w >> 11 & 0x7ff) << 4)); v3[2] = half2f((uint16_t)((w >> 22 & 0x3ff) << 5)); }
                for (int c = 0; c < 3; c++) { float v = v3[c]; if (v != v || v > 65000.f || v < -65000.f) { bad++; v = 1; } if (v > mx) mx = v; if (v < mn) mn = v; row[3 * x + c] = (uint8_t)(v <= 0 ? 0 : v >= 1 ? 255 : v * 255 + .5f); } }
            else if (d->f == VK_FORMAT_R8_UNORM) row[3 * x] = row[3 * x + 1] = row[3 * x + 2] = ((const uint8_t *)d->p)[(size_t)y * d->w + x];
            else if (d->f == VK_FORMAT_A2B10G10R10_UNORM_PACK32) { uint32_t w = ((const uint32_t *)d->p)[(size_t)y * d->w + x]; for (int c = 0; c < 3; c++) row[3 * x + c] = (uint8_t)((w >> (10 * c) & 0x3ff) >> 2); }
            else { const uint8_t *p = (const uint8_t *)d->p + ((size_t)y * d->w + x) * 4; int bgr = d->f == VK_FORMAT_B8G8R8A8_UNORM; row[3 * x] = p[bgr ? 2 : 0]; row[3 * x + 1] = p[1]; row[3 * x + 2] = p[bgr ? 0 : 2]; } }
            fwrite(row, 1, (size_t)d->w * 3, f); }
        free(row); fclose(f);
        { const char *rw = getenv("DM_VK_PASSDUMP_RAW"); if (rw && atoi(rw)) {   // r45 DM_VK_PASSDUMP_RAW=1: the texels as copied (.raw next to the .ppm), for values the 8-bit image clamps
            size_t bpp = d->f == VK_FORMAT_R32G32B32A32_SFLOAT ? 16 : d->f == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : d->f == VK_FORMAT_R16_SFLOAT ? 2 : d->f == VK_FORMAT_R8_UNORM ? 1 : 4;
            char rp[1100]; snprintf(rp, sizeof rp, "%s.raw", path); FILE *rf = fopen(rp, "wb"); if (rf) { fwrite(d->p, 1, (size_t)d->w * d->h * bpp, rf); fclose(rf); } } }
        VKLOG("pass dump %02d: %ux%u format %d → %s%s", i, d->w, d->h, d->f, path, d->f == VK_FORMAT_R16G16B16A16_SFLOAT ? "" : "");
        if (d->f == VK_FORMAT_R32G32B32A32_SFLOAT && (size_t)d->w * d->h <= 16) for (uint32_t k = 0; k < d->w * d->h; k++) { const float *p = (const float *)d->p + 4 * (size_t)k;   // pkm1
            VKLOG("pkm1: pixel %u of pass dump %02d (pass %d): %.9g %.9g %.9g %.9g", k, i, d->pass, p[0], p[1], p[2], p[3]); }
        if (d->f == VK_FORMAT_R32G32B32A32_SFLOAT || d->f == VK_FORMAT_R16G16B16A16_SFLOAT || d->f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || d->f == VK_FORMAT_R16_SFLOAT) VKLOG("   NaN/inf channels %llu, finite values %.3f .. %.3f", (unsigned long long)bad, mn, mx);
        vkDestroyBuffer(V.dev, d->b, NULL); vkFreeMemory(V.dev, d->m, NULL);
    }
    g_npd = 0; g_pd_state = 0;
}
// DM_VK_CRUMBS=1 (start-up), for a device lost (a GPU hang): every pass end and dispatch writes {submission, pass} into host-visible memory after a full barrier, and the host
// keeps the shaders each pass drew with; when the device is lost (vk_crumbs_report), the last work the GPU finished and the passes after it are logged
static int g_crumbs = -1; static VkBuffer g_crumb_buf; static volatile uint32_t *g_crumb;
typedef struct { uint32_t sub; int pass, n; uint64_t k[8][3]; } crumb_t; static crumb_t g_cring[512]; static unsigned g_cn;
static int crumbs_on(void) {
    if (g_crumbs < 0) { const char *e = getenv("DM_VK_CRUMBS"); g_crumbs = e && atoi(e);
        if (g_crumbs) { VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 256, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
            VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &g_crumb_buf)); VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, g_crumb_buf, &rq);
            VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
            VkDeviceMemory m; VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, g_crumb_buf, m, 0)); void *p; VKCHK(vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, &p)); g_crumb = p; g_crumb[0] = 0; } }
    return g_crumbs;
}
static crumb_t *crumb_cur(void) { crumb_t *c = &g_cring[(g_cn ? g_cn - 1 : 0) % 512]; uint32_t sub = (uint32_t)mtl_last_fence() + 1;
    if (!g_cn || c->sub != sub || c->pass != g_sub_pass - 1) { c = &g_cring[g_cn++ % 512]; memset(c, 0, sizeof *c); c->sub = sub; c->pass = g_sub_pass - 1; } return c; }
static void crumb_draw(uint64_t a, uint64_t b, uint64_t c3) { if (!crumbs_on()) return; crumb_t *c = crumb_cur();
    for (int i = 0; i < c->n; i++) if (c->k[i][0] == a && c->k[i][1] == b && c->k[i][2] == c3) return; if (c->n < 8) { c->k[c->n][0] = a; c->k[c->n][1] = b; c->k[c->n][2] = c3; c->n++; } }
static void crumb_mark(int dispatch) { if (!crumbs_on()) return; VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    vkCmdFillBuffer(cb, g_crumb_buf, 0, 4, ((uint32_t)(mtl_last_fence() + 1) & 0xffff) << 16 | (dispatch ? 0x8000u : 0) | ((uint32_t)(g_sub_pass - 1) & 0x7fff)); }
void vk_crumbs_report(void) {
    if (g_crumbs <= 0 || !g_crumb) return;
    uint32_t v = g_crumb[0], sub = v >> 16, pass = v & 0x7fff;
    VKLOG("[crumbs] the GPU last finished submission %u (low 16 bits) %s %u", sub, v & 0x8000 ? "a dispatch after pass" : "pass", pass);
    int shown = 0;
    for (unsigned i = g_cn > 512 ? g_cn - 512 : 0; i < g_cn && shown < 12; i++) { crumb_t *c = &g_cring[i % 512];
        if ((c->sub & 0xffff) < sub || ((c->sub & 0xffff) == sub && c->pass <= (int)pass)) continue;
        shown++; VKLOG("[crumbs] then submission %u pass %d:", c->sub, c->pass);
        for (int k = 0; k < c->n; k++) VKLOG("[crumbs]   shaders %016llx %016llx %016llx", (unsigned long long)c->k[k][0], (unsigned long long)c->k[k][1], (unsigned long long)c->k[k][2]); }
}
void vk_oq_draw(VkCommandBuffer cb); void vk_oq_pass_end(void); void vk_oq_reset(uint32_t type);   // gfx_vk_core.c (r50 occlusion queries)
// s56 (gfx_vk_core.c vk_s56_mirror, profile key 3 = DM_VK_S56): the pass has ended (vkCmdEnd*): each colour attachment it had, at the level and layers it drew
// (the attachment view's base level and first layer; a layered pass's layerCount layers), is offered to the write-back, which checks the texture and copies it out
void vk_s56_mirror(const tex_t *t, vkimg_t *im, unsigned level, unsigned layer0, unsigned n);
static void s56_pass_end(void) {
    for (int i = 0; i < S.ncol && i < 8; i++) {
        if (!P_ca[i].imageView || !P_att[i].im || !S.rt_has_col[i]) continue;
        const view_t *cv = S.rt_has_cview[i] ? &S.rt_cview[i] : NULL;
        unsigned lv = cv && cv->set_levels ? cv->base_level : 0, ly = cv && cv->set_layers ? cv->min_layer : 0, n = P_ri.layerCount > 1 ? P_ri.layerCount : 1;
        vk_s56_mirror(&S.rt_col[i], P_att[i].im, lv, ly, n);
    }
    void vk_z3_depth(const tex_t *t);   // z3: diagnostic list of depth targets in CPU-accessible pools
    if (S.rt_has_dep) vk_z3_depth(&S.rt_dep);
}
void vk_end_pass(void) {
    struct rec_ctx *const g_rec = rec_cur();   // R1b (parallel recording): the thread's context read once (S, P_*, g_pass_* below)
    const int s64p = g_s64_pend && S.pass_open == 1; g_s64_pend = 0;   // s64: the pending barrier goes right after this pass end
    if (g_x_listpasses && S.pass_open == 1) { Dl_info di; void *ra = __builtin_return_address(0); uintptr_t base = dladdr((void *)vk_end_pass, &di) ? (uintptr_t)di.dli_fbase : 0; VKLOG("    ended from %#llx (%llu draws)", (unsigned long long)((uintptr_t)ra - base), (unsigned long long)g_pass_draws); }   // (DM_VK_TRACE_PASS=-2: who ends each pass)
    if (S.pass_open == 2 && P_cleared) (void)pass_commit();   // only clears: the pass still runs, for its load ops
    const int o2end = S.pass_open == 1 && g_occ2 >= 2 && !g_wk;   // occ-s2: a resolve point after it when reports are pending (the replay thread's)
    if (S.pass_open == 1) {
        if (!g_wk) g_vk_pends++;   // occ-s0: the replay thread's pass ends
        rcg_end();   // occ-s3: a conditional rendering block never spans a pass end
        vk_oq_pass_end();
        if (S.rp_pass) vkCmdEndRenderPass(vk_cmd());
        else vkCmdEndRendering(vk_cmd());

#if defined(DM_NXN1X)
        if (!g_wk) fg_profile_pass();   // (S1e (parallel recording): frame generation's per-pass tables are the replay thread's; a worker records only while it collects nothing, seg_worker_refuse)
        if (fg_frames_left) {
            for (unsigned i = 0; i < 9; ++i) {
                vkimg_t *im = P_att[i].im;
                if (!im || !(i == 8 ? P_da.imageView != VK_NULL_HANDLE : i < S.ncol && P_ca[i].imageView)) continue;
                const view_t *v = i == 8 ? (S.rt_has_dview ? &S.rt_dview : NULL) :
                                           (S.rt_has_cview[i] ? &S.rt_cview[i] : NULL);
                fg_capture_image(pass_scale == 2 ? im->scaled : im, v && v->set_levels ? v->base_level : 0,
                    v && v->set_layers ? v->min_layer : 0, "attachment", i, S.sh[1] ? S.sh[1]->key : 0);
            }
        }
#endif
        if (scale_time_mode == 1) finish_pass_scale();
        vk_pass_ts(1, S.pass_w, S.pass_h, (uint32_t)S.cfmt[0], (uint32_t)g_pass_draws | (g_pass_draws ? (uint32_t)((g_pass_bb[2] - g_pass_bb[0]) & 0xffff) << 16 : 0));
        if (scale_time_mode != 1) finish_pass_scale();
        pass_dump(); expo_grab(); ind_grab(); if (g_x_tracepass >= 0 && g_sub_pass - 1 == g_x_tracepass) { setenv("DM_VK_TRACE_PASS", "-1", 1); g_x_tracepass = -1; } }
    if (S.pass_open == 1) {   // s64a: a pass ends: by which op, how many draws, and the pass after a barrier-ended one
        uint64_t d = g_pass_draws; g_s64_passes++; g_s64_end[g_xo_op < 64 ? g_xo_op : 63]++;
        g_s64_dpp[d == 0 ? 0 : d == 1 ? 1 : d < 4 ? 2 : d < 8 ? 3 : d < 16 ? 4 : d < 64 ? 5 : 6]++;
        if (g_s64_afterbar) { g_s64_ab_n++; g_s64_ab_fb += g_s64_fb != 0; }
        g_s64_afterbar = g_xo_op == OP_BARRIER; g_s64_fb = 0; g_s65_rdp += g_s65_rdpass; if (g_s65_rdpass) g_s65_rddraws += g_pass_draws; g_s65_rdpass = 0; }
    if (S.pass_open == 1) crumb_mark(0);
    int vk_z4_on(void);
    if (S.pass_open == 1 && (dm_profile_key(3) || vk_z4_on())) s56_pass_end();   // s56 (off: nothing); z4: linear targets for every title
    S.pass_open = 0; S.bound_pipe = VK_NULL_HANDLE; g_tf_in_pass = 0;
    g_s64_pwrite = 0; if (s64p) vk_img_barrier_all(vk_cmd());   // s64
    if (g_o3g.on && !g_wk) { void vk_o3r_flush(void); vk_o3r_flush(); }   // occ-s3fast (DM_VK_OCC_CONDDIAG): the conditions of the pass read back here, before its resolve point
    if (o2end) vk_occ2_pass_end();   // occ-s2: the first pass end after reports resolves them (copy, one dispatch, the guest's words), outside the pass, in stream order
}
// a pass end before something that reads or writes pool memory other than through a uniform binding (copies, fills, dispatches, buffer ↔ texture
// copies, the submission's end): the deferred uniform updates land first (draws read them through their shadows, so ordinary pass ends keep them)
void vk_end_pass_land(void) { vk_end_pass(); if (vk_sub_open()) vk_ubd_land(vk_cmd()); }
static int pass_whole(VkRect2D r) { return !r.offset.x && !r.offset.y && r.extent.width >= S.pass_w && r.extent.height >= S.pass_h; }
// r76: layered rendering (nvn_mtl.m rp_begin_ex): a draw whose vertex, geometry or tessellation shader writes gl_Layer picks the attachment layer per primitive only
// among the pass's layers (layerCount), which were 1: every layer went to the first. Pokemon Legends: Z-A builds its 32x32x32 colour grading LUT one depth plane per
// instance this way, and on the Mac without it only plane 0 was written and the whole picture came out black; cube shadow maps and cascades drawn in one pass alike.
// Such a pass gets the layers of its attachments' views (the fewest of them). A pass is layered or not as its draws: a begun pass ends where that changes, a pass not
// begun yet (clears only) takes the draw's. DM_VK_LAYERED=0 turns it off
static int g_layered_on = -1; static uint64_t g_layered_n;   // R1b (parallel recording): g_pass_layered in struct rec_ctx
static int draw_layered(void) {
    if (g_layered_on < 0) { const char *e = getenv("DM_VK_LAYERED"); g_layered_on = !e || atoi(e) != 0; }
    // S1c (parallel recording): layer_out is set once at draw time by any recording thread (eden_draw, pkm2): read atomically
    return g_layered_on && ((S.sh[0] && __atomic_load_n(&S.sh[0]->layer_out, __ATOMIC_RELAXED)) || (S.sh[2] && __atomic_load_n(&S.sh[2]->layer_out, __ATOMIC_RELAXED)) || (S.sh[4] && __atomic_load_n(&S.sh[4]->layer_out, __ATOMIC_RELAXED)));
}
static void pass_set_layered(int lay) {
    uint32_t nl = UINT32_MAX;
    if (lay) { for (int i = 0; i < S.ncol; i++) if (P_ca[i].imageView && P_att[i].layers < nl) nl = P_att[i].layers;
               if (P_da.imageView && P_att[8].layers < nl) nl = P_att[8].layers; }
    g_pass_layered = lay; P_ri.layerCount = nl != UINT32_MAX && nl > 1 ? nl : 1;
    if (P_ri.layerCount > 1) { uint64_t c = ++g_layered_n; if (c <= 2 || !(c & (c - 1))) VKLOG("r76: layered pass %ux%u: %u layers for a shader that writes gl_Layer (%llu so far; DM_VK_LAYERED=0 off)", S.pass_w, S.pass_h, P_ri.layerCount, (unsigned long long)c); }
}
// R6/R7/R8 (parallel recording), R6: the target resolution of a pass, split from its recording (begin_pass). In stream order, for each colour target that is set and then
// the depth target: its attachment view (vk_tex_view role 1: the image found or made, uploaded or filled at first use, and marked GPU-written: note_image_write, wseq,
// wpres, gpu_w, vk_tex_gpu_mark), then its first layout transition into cb (vk_tex_prepare). The side effects and their order are begin_pass's before the split (its loops
// did the same calls in the same order; what they did between them only set S.cfmt and the attachment infos, which the record step below now does after). A dry pass
// can call this at a pass's first draw or clear after a target change, and the workers' lookups then find the views made and the layouts GENERAL (design R6)
typedef struct { VkImageView v; vkimg_t *im; VkFormat fmt; uint32_t layers; } pass_tgt;   // a resolved target: its view, image, and the view's format and layers (vk_view_fmt, vk_view_layers)
// S1e (parallel recording): on a given state's targets (st: S, or at a worker segment's hand-out the targets the segment will draw to, seg_par_handout)
static void pass_resolve_targets_on(const rec_state_t *st, VkCommandBuffer cb, pass_tgt col[8], pass_tgt *dep) {
    for (int i = 0; i < st->rt_n; i++) {
        col[i] = (pass_tgt){0};
        if (!st->rt_has_col[i]) continue;
        vkimg_t *im; VkImageView v = vk_tex_view(&st->rt_col[i], st->rt_has_cview[i] ? &st->rt_cview[i] : NULL, 0xff, 0, 1, &im);
        if (!v) continue;
        vk_tex_prepare(cb, im);
        col[i] = (pass_tgt){v, im, vk_view_fmt, vk_view_layers};
    }
    *dep = (pass_tgt){0};
    if (st->rt_has_dep) {
        vkimg_t *im; VkImageView v = vk_tex_view(&st->rt_dep, st->rt_has_dview ? &st->rt_dview : NULL, 0xff, 0, 1, &im);
        if (v) { vk_tex_prepare(cb, im); *dep = (pass_tgt){v, im, vk_view_fmt, vk_view_layers}; }
    }
}
static void pass_resolve_targets(VkCommandBuffer cb, pass_tgt col[8], pass_tgt *dep) { struct rec_ctx *const g_rec = rec_cur(); pass_resolve_targets_on(&S, cb, col, dep); }
static void seg_wk_targets(pass_tgt col[8], pass_tgt *dep);   // S1e (parallel recording): a worker's pass: the targets the replay thread resolved at the hand-out (after exec_ops)
// a part cut (vk_cmd_split): what is recorded goes to the GPU now. psplit: a DM_VK_PASS_SPLIT cut (not capped), else an ordinary DM_VK_CB_DRAWS one, at most g_s61_maxcut a
// submission. No pass is open here; deferred uniform updates stay deferred (draws read their shadows). (S1d (parallel recording): was inline in begin_pass; the dry pass's
// cuts take it too, seg_cut_take)
static void part_cut(int psplit) {
    materialize_scale(NULL);
    void vk_cmd_split(void); if (g_x_landcut) vk_ubd_land(vk_cmd());
    if (psplit) vk_cmd_split(); else if (g_ord_parts < g_s61_maxcut) { vk_cmd_split(); g_ord_parts++; g_s61_cuts++; g_s61_cutdraws += g_cb_draws; } else g_s61_cap++;   /* s61: the cap was 15 */
    g_cb_draws = 0;
}
static int begin_pass(void) {
    struct rec_ctx *const g_rec = rec_cur();   // R1b (parallel recording): the thread's context read once (S, P_*, g_pass_* below)
    if (S.pass_open) return 1;
    int psplit = g_psplit_now; g_psplit_now = 0;   // a pass split (prepare_draw): taken before any early return, so the flag never forces a later pass's cut
    VkCommandBuffer cb = vk_cmd();
    pass_tgt rcol[8], rdep; if (g_wk) seg_wk_targets(rcol, &rdep); else pass_resolve_targets(cb, rcol, &rdep);   // R6 (parallel recording): the resolve step; the record step follows (S1e: a worker's done by the replay thread at the hand-out, marks and first transitions included)
    VkRenderingAttachmentInfo *ca = P_ca; memset(ca, 0, sizeof P_ca);
    for (int i = 0; i < 9; i++) P_att[i].im = NULL;
    VkRenderingAttachmentInfo da = {0}, sa = {0};
    uint32_t w = UINT32_MAX, h = UINT32_MAX, layers = 1, samples = 1; int ncol = 0, have = 0;
    for (int i = 0; i < S.rt_n; i++) {
        S.cfmt[i] = VK_FORMAT_UNDEFINED;
        ca[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO; ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL; ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        if (!S.rt_has_col[i]) continue;
        vkimg_t *im = rcol[i].im; VkImageView v = rcol[i].v; const VkFormat vk_view_fmt = rcol[i].fmt; const uint32_t vk_view_layers = rcol[i].layers;   // R6: as resolved (the names shadow the TLS ones)
        if (!v) continue;
        uint32_t lv = S.rt_has_cview[i] && S.rt_cview[i].set_levels ? S.rt_cview[i].base_level : 0;
        uint32_t iw = im->w >> lv ? im->w >> lv : 1, ih = im->h >> lv ? im->h >> lv : 1;
        // Imageless attachment metadata uses the image mip extent. Rendering
        // through a raw BC4 view addresses one output pixel per encoded block.
        P_att[i].image_w = iw; P_att[i].image_h = ih;
        if (im->fmt == VK_FORMAT_BC4_UNORM_BLOCK && vk_view_fmt == VK_FORMAT_R16G16B16A16_UINT) {
            iw = (iw + 3) / 4;
            ih = (ih + 3) / 4;
        }
        if (iw < w) w = iw; if (ih < h) h = ih; samples = im->samples; have = 1;
        { static int dup_on = -1; if (dup_on < 0) { const char *e = getenv("DM_VK_DUP_ATT"); dup_on = !e || atoi(e) != 0; } int dup = -1;   // r76: the same level and layers in a second slot
          for (int j = 0; dup_on && j < i; j++) if (ca[j].imageView == v) { dup = j; break; }   // (views are made once per subresource: the same view is the same level and layers)
          if (dup >= 0) { static int n; if (n < 4) { n++; VKLOG("r76: colour slot %d is the same target view as slot %d: attached once (nvn_mtl.m: Breath of the Wild's G-buffer was black; DM_VK_DUP_ATT=0 off)", i, dup); } continue; } }
        ca[i].imageView = v; S.cfmt[i] = vk_view_fmt; ncol = i + 1; P_att[i].im = im;   // (the view's format: a render pass's attachment and its pipelines take it)
        P_att[i].fmt = vk_view_fmt; P_att[i].layers = vk_view_layers ? vk_view_layers : 1; P_att[i].w = iw; P_att[i].h = ih;
    }
    S.dfmt = VK_FORMAT_UNDEFINED; S.daspect = 0;
    if (S.rt_has_dep) {
        vkimg_t *im = rdep.im; VkImageView v = rdep.v; const VkFormat vk_view_fmt = rdep.fmt; const uint32_t vk_view_layers = rdep.layers;   // R6: as resolved
        if (v) {
            uint32_t lv = S.rt_has_dview && S.rt_dview.set_levels ? S.rt_dview.base_level : 0;
            uint32_t iw = im->w >> lv ? im->w >> lv : 1, ih = im->h >> lv ? im->h >> lv : 1;
            if (iw < w) w = iw; if (ih < h) h = ih; samples = im->samples; have = 1;
            da = (VkRenderingAttachmentInfo){VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, v, VK_IMAGE_LAYOUT_GENERAL, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE};
            S.dfmt = im->fmt; S.daspect = im->aspect; P_att[8].im = im; P_att[8].fmt = vk_view_fmt; P_att[8].layers = vk_view_layers ? vk_view_layers : 1; P_att[8].w = iw; P_att[8].h = ih;
            if (im->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) sa = da;
        }
    }
    // No target set at all (nvn_mtl.m rp_begin_ex_ / bare_dims: an attachment-less pass covering the viewport, else the scissor; Tears of the Kingdom's agl shader that only
    // stores to a storage buffer, Bayonetta 3's ~3 draws a frame): rendering with no attachments. A target that is set but cannot be attached still skips. DM_VK_BARE=0: skipped as before
    if (!have) { static int bare_on = -1; if (bare_on < 0) { const char *e = getenv("DM_VK_BARE"); bare_on = !(e && *e == '0'); }
        int bare = bare_on && !S.rt_has_dep; for (int i = 0; i < S.rt_n && bare; i++) if (S.rt_has_col[i]) bare = 0;
        if (bare) { double x1 = S.vp[0][2] ? fmax(S.vp[0][0], S.vp[0][0] + S.vp[0][2]) : S.sc_set ? (double)S.sc.x + S.sc.w : 1, y1 = S.vp[0][3] ? fmax(S.vp[0][1], S.vp[0][1] + S.vp[0][3]) : S.sc_set ? (double)S.sc.y + S.sc.h : 1;
            w = (uint32_t)fmin(fmax(ceil(x1), 1.0), 16384.0); h = (uint32_t)fmin(fmax(ceil(y1), 1.0), 16384.0); samples = 1; have = 1;
            static int n; if (n < 4) { n++; VKLOG("attachment-less pass: the game draws with no render target set, pass opened at the viewport size %ux%u", w, h); } } }
    if (!have) { VKRED_ONCE(30, "draw with no usable render target: skipped"); return 0; }
    if (!g_seg_cuts && (psplit || (g_x_cbdraws > 0 && g_cb_draws >= g_x_cbdraws))) {   // (S1d (parallel recording): not while the dry pass places the cuts)
        part_cut(psplit); cb = vk_cmd();
    }
    { static int n; uint32_t cs[9] = {0}; int any = 0, mis = 0; uint32_t s0 = 0;   // multisampled passes: every attachment's sample count (they must agree)
      for (int i = 0; i < ncol; i++) if (ca[i].imageView) { cs[i] = P_att[i].im->samples; if (cs[i] > 1) any = 1; if (!s0) s0 = cs[i]; else if (cs[i] != s0) mis = 1; }
      if (da.imageView) { cs[8] = P_att[8].im->samples; if (cs[8] > 1) any = 1; if (s0 && cs[8] != s0) mis = 1; }
      if ((any || mis) && n < 10) { n++; VKLOG("pass %ux%u samples: colour %u %u %u %u, depth %u%s", w, h, cs[0], cs[1], cs[2], cs[3], cs[8], mis ? " MISMATCH" : ""); } }
    if (!g_x_nobar) vk_img_barrier_all(cb);
    if (g_x_depth) { da.loadOp = sa.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; da.storeOp = sa.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; }
    if (g_x_color) for (int i = 0; i < ncol; i++) ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    P_da = da; P_sa = sa; P_cleared = 0;
    P_ri = (VkRenderingInfo){VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {w, h}}, layers, 0, (uint32_t)ncol, P_ca,
                             (S.daspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? &P_da : NULL, (S.daspect & VK_IMAGE_ASPECT_STENCIL_BIT) ? &P_sa : NULL};
    S.pass_open = 2; S.pass_w = w; S.pass_h = h; S.samples = samples; S.ncol = ncol; S.bound_pipe = VK_NULL_HANDLE;
    pass_set_layered(draw_layered());   // r76
    return 1;
}
static VkRect2D scissor_rect(void) {
    int32_t x = 0, y = 0; int64_t x1 = S.pass_w, y1 = S.pass_h;
    if (S.sc_set) {
        int64_t sx = S.sc.x, sy = S.sc.y, sx1 = (int64_t)S.sc.x + S.sc.w, sy1 = (int64_t)S.sc.y + S.sc.h;
        if (g_lower_left && !g_flipy) { int64_t t = (int64_t)S.pass_h - sy1; sy1 = (int64_t)S.pass_h - sy; sy = t; }
        if (sx > x) x = (int32_t)sx; if (sy > y) y = (int32_t)sy; if (sx1 < x1) x1 = sx1; if (sy1 < y1) y1 = sy1;
    }
    if (x1 < x) x1 = x; if (y1 < y) y1 = y;
    return (VkRect2D){{x, y}, {(uint32_t)(x1 - x), (uint32_t)(y1 - y)}};
}
// r76: a clear folded into the load ops covers the pass's area and layers, the fewest of its attachments', while NVN clears the whole target view (nvn_mtl.m rp_begin_ex).
// Bayonetta 2 clears its 1280x720 depth beside a 32x32 colour target and only that corner was cleared (the scene failed the depth test elsewhere and only the narration
// showed); Little Nightmares clears its point lights' 6-face shadow cube and only face 0 was cleared. An attachment larger than the pass, or with more layers, whose clear
// covers it (no scissor inside it), is cleared whole with a transfer clear before the pass begins. DM_VK_CLEAR_WHOLE=0 turns it off
static uint64_t g_clrw_n;
static int clear_whole(int i, VkClearValue v, VkImageAspectFlags as) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_CLEAR_WHOLE"); on = !e || atoi(e) != 0; }
    vkimg_t *im = P_att[i].im; if (!on || !im || S.pass_open != 2) return 0;
    uint32_t aw = P_att[i].w, ah = P_att[i].h, al = P_att[i].layers ? P_att[i].layers : 1, pl = P_ri.layerCount ? P_ri.layerCount : 1;
    if (aw <= S.pass_w && ah <= S.pass_h && al <= pl) return 0;
    if (S.sc_set && (S.sc.x > 0 || S.sc.y > 0 || (int64_t)S.sc.x + S.sc.w < (int64_t)aw || (int64_t)S.sc.y + S.sc.h < (int64_t)ah)) return 0;   // a scissor inside the target: the pass's area, as before
    if (g_wk && !g_wk_rerun) { seg_wk_abort(WA_CLEAR); return 1; }   // S1e (parallel recording): its marks (vk_tex_content_changed) are the replay thread's: the segment is recorded there
    const view_t *vw = i < 8 ? (S.rt_has_cview[i] ? &S.rt_cview[i] : NULL) : (S.rt_has_dview ? &S.rt_dview : NULL);
    uint32_t lv = vw && vw->set_levels ? vw->base_level : 0, ly0 = vw && vw->set_layers ? vw->min_layer : 0;
    if (im->vtype == VK_IMAGE_VIEW_TYPE_3D) { ly0 = 0; al = 1; }   // (a 3D level is one layer: all its depth planes)
    VkImageSubresourceRange rg = {as, lv, 1, ly0, al};
    VkCommandBuffer cb = vk_cmd();
    wb_use(im, WB_OTHER);
    vk_tex_content_changed(im);
    if (as & VK_IMAGE_ASPECT_COLOR_BIT) vkCmdClearColorImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &v.color, 1, &rg);
    else vkCmdClearDepthStencilImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &v.depthStencil, 1, &rg);
    vk_img_barrier_all(cb);   // before the pass, which begins after this (pass_open 2)
    uint64_t c = __atomic_add_fetch(&g_clrw_n, 1, __ATOMIC_RELAXED); if (c <= 4 || !(c & (c - 1))) VKLOG("r76: clear of a %ux%u target with %u layers in a %ux%u pass of %u layers: the whole target cleared (%s, %llu so far; DM_VK_CLEAR_WHOLE=0 off)",
        aw, ah, al, S.pass_w, S.pass_h, pl, i < 8 ? "colour" : "depth / stencil", (unsigned long long)c);
    return 1;
}
// r76: clears with a partial channel mask or a partial stencil mask: vkCmdClearAttachments writes every channel and every stencil bit, NVN only the masked ones
// (nvn_mtl.m 11998, 12242: Donkey Kong Bananza clears its stencil with mask 1, PR #108; Tears of the Kingdom clears single channels). Drawn instead: a triangle over the
// scissor with the channel mask as the colour write mask (the other targets masked off) or the stencil mask as the stencil write mask (REPLACE, test ALWAYS; depth written
// with it when cleared too). A query that counts samples is ended first (the next draw begins another, its count added). DM_VK_CLEAR_DRAW=0: cleared in full, as before
#include "clear_vert.spv.h"
#include "clear_f_frag.spv.h"
#include "clear_u_frag.spv.h"
#include "clear_i_frag.spv.h"
static struct { int state, n; VkPipelineLayout pl; VkShaderModule vs, fs[3]; struct { uint32_t k[16]; VkPipeline p; } c[64]; uint64_t draws; } g_cd; static pthread_mutex_t g_cd_mu = PTHREAD_MUTEX_INITIALIZER;
static int clear_draw(int idx, uint32_t cmask, const uint32_t *bits, int ft, int dwrite, float z, uint32_t smask, uint32_t sref) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_CLEAR_DRAW"); on = !e || atoi(e) != 0; }
    if (!on || g_cd.state < 0 || ft < 0 || ft > 2) return 0;
    pthread_mutex_lock(&g_cd_mu);   // S1e (parallel recording): the pipelines' cache and its first set-up, shared by the recording threads (masked clears are rare)
    if (!g_cd.state) {
        g_cd.state = -1;
        VkPushConstantRange pr = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 20};
        VkPipelineLayoutCreateInfo li = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 0, NULL, 1, &pr};
        if (vkCreatePipelineLayout(V.dev, &li, NULL, &g_cd.pl)) { VKRED_ONCE(170, "r76: clear pipeline layout failed: masked clears cleared in full"); pthread_mutex_unlock(&g_cd_mu); return 0; }
        const uint32_t *code[4] = {clear_vert_spv, clear_f_frag_spv, clear_u_frag_spv, clear_i_frag_spv}; size_t sz[4] = {sizeof clear_vert_spv, sizeof clear_f_frag_spv, sizeof clear_u_frag_spv, sizeof clear_i_frag_spv};
        for (int i = 0; i < 4; i++) { VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sz[i], code[i]};
            if (vkCreateShaderModule(V.dev, &mi, NULL, i ? &g_cd.fs[i - 1] : &g_cd.vs)) { VKRED_ONCE(170, "r76: clear shader module failed: masked clears cleared in full"); pthread_mutex_unlock(&g_cd_mu); return 0; } }
        g_cd.state = 1;
    }
    uint32_t key[16] = {0}; key[0] = (uint32_t)S.ncol; for (int i = 0; i < S.ncol && i < 8; i++) key[1 + i] = (uint32_t)S.cfmt[i];
    key[9] = (uint32_t)S.dfmt; key[10] = S.samples ? S.samples : 1; key[11] = (uint32_t)(idx + 1); key[12] = cmask; key[13] = (uint32_t)ft; key[14] = (uint32_t)dwrite; key[15] = smask;
    VkPipeline p = VK_NULL_HANDLE;
    for (int i = 0; i < g_cd.n; i++) if (!memcmp(g_cd.c[i].k, key, sizeof key)) { p = g_cd.c[i].p; break; }
    if (!p) {
        VkPipelineShaderStageCreateInfo ss[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, g_cd.vs, "main", NULL},
                                                 {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, g_cd.fs[ft], "main", NULL}};
        VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0};
        VkPipelineViewportStateCreateInfo vs = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
        VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, 0, 0, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0, 0, 0, 0, 1.0f};
        VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, (VkSampleCountFlagBits)key[10], 0, 0, NULL, 0, 0};
        VkStencilOpState so = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS, 0xff, smask, 0};
        VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, NULL, 0, (VkBool32)dwrite, (VkBool32)dwrite, VK_COMPARE_OP_ALWAYS, 0, smask != 0, so, so, 0, 1};
        VkPipelineColorBlendAttachmentState cba[8]; memset(cba, 0, sizeof cba); if (idx >= 0 && idx < 8) cba[idx].colorWriteMask = cmask;
        VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, 0, VK_LOGIC_OP_COPY, (uint32_t)S.ncol, cba, {0, 0, 0, 0}};
        VkDynamicState dyn[3] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_REFERENCE};
        VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 3, dyn};
        VkFormat cf[8]; for (int i = 0; i < S.ncol && i < 8; i++) cf[i] = S.cfmt[i];
        VkFormat df = S.dfmt; int has_stencil = df == VK_FORMAT_D24_UNORM_S8_UINT || df == VK_FORMAT_D32_SFLOAT_S8_UINT || df == VK_FORMAT_S8_UINT;   // (as the draws' pipelines)
        VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, (uint32_t)S.ncol, cf, df == VK_FORMAT_S8_UINT ? VK_FORMAT_UNDEFINED : df, has_stencil ? df : VK_FORMAT_UNDEFINED};
        VkRenderPass prp = rp_on() ? rp_compat(key[0], key + 1, key[9], key[10]) : VK_NULL_HANDLE;
        if (!prp && !V.has_dynrender) { pthread_mutex_unlock(&g_cd_mu); return 0; }
        VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, prp ? NULL : &rci, 0, 2, ss, &vi, &ia, NULL, &vs, &rs, &ms, &ds, &cbs, &dy, g_cd.pl, prp, 0, VK_NULL_HANDLE, -1};
        VkResult r = vkCreateGraphicsPipelines(V.dev, V.pcache, 1, &ci, NULL, &p);
        if (r) { VKRED_ONCE(171, "r76: clear pipeline failed (%d): masked clears cleared in full", r); p = (VkPipeline)1; }
        if (g_cd.n < 64) { memcpy(g_cd.c[g_cd.n].k, key, sizeof key); g_cd.c[g_cd.n++].p = p; }
    }
    pthread_mutex_unlock(&g_cd_mu);
    if (p == (VkPipeline)1 || !pass_commit()) return 0;
    vk_oq_pass_end();   // (r50: the clear's samples stay out of the game's occlusion query)
    VkCommandBuffer cb = vk_cmd();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    VkViewport vp = {0, 0, (float)(S.pass_w*pass_scale), (float)(S.pass_h*pass_scale), 0.0f, 1.0f}; vkCmdSetViewport(cb, 0, 1, &vp);
    VkRect2D sc = physical_rect(scissor_rect()); vkCmdSetScissor(cb, 0, 1, &sc);
    if (smask) vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, sref & 0xff);
    struct { uint32_t c[4]; float z; } pc = {{0, 0, 0, 0}, z < 0 ? 0 : z > 1 ? 1 : z}; if (bits) memcpy(pc.c, bits, 16);
    vkCmdPushConstants(cb, g_cd.pl, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 20, &pc);
    if (sc.extent.width && sc.extent.height) {
        scale_note_scissor(sc);
        if (idx >= 0 && cmask) scale_writes[idx] |= VK_IMAGE_ASPECT_COLOR_BIT;
        if (dwrite) scale_writes[8] |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if (smask) scale_writes[8] |= VK_IMAGE_ASPECT_STENCIL_BIT;
        rcg_begin(cb, 1);   // occ-s3: a masked clear under a GPU condition is a gated draw (occ-s3fast: in the open block when the predicate is the same)
        vkCmdDraw(cb, 3, 1, 0, 0);
    }
    g_cmd_gen++; S.bound_pipe = VK_NULL_HANDLE;   // R2 (parallel recording): the thread's recording target's generation (gfx_vk.h)     // the draws bind their pipeline, state and sets again
    uint64_t c = __atomic_add_fetch(&g_cd.draws, 1, __ATOMIC_RELAXED); if (c <= 4 || !(c & (c - 1)))
        VKLOG("r76: masked clear drawn: %s mask %#x (%llu so far; DM_VK_CLEAR_DRAW=0 off)", idx >= 0 ? "colour target channel" : "stencil", idx >= 0 ? cmask : smask, (unsigned long long)c);
    return 1;
}
static void clear_color(const p_clear_color *c) {
    const int rd = g_rdiag > 0; if (rd) rd_start();   // replaydiag: 55 begin_pass, 56 folded into the load op / whole, 57 masked draw, 58 pass_commit, 59 vkCmdClearAttachments
    if (!begin_pass()) { if (rd) rd_mark(55); return; }
    if (rd) rd_mark(55);
    if (c->index < 0 || c->index >= S.ncol || S.cfmt[c->index] == VK_FORMAT_UNDEFINED) return;
    if (!c->mask) return;
    VkClearAttachment a = {VK_IMAGE_ASPECT_COLOR_BIT, (uint32_t)c->index, {{{c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3]}}}};
    const vkfmt_t *f = vk_fmt(S.rt_col[c->index].d.format);
    if (f && f->is_int) for (int i = 0; i < 4; i++) { if (f->is_sint) a.clearValue.color.int32[i] = (int32_t)c->rgba[i]; else a.clearValue.color.uint32[i] = c->rgba[i] > 0 ? (uint32_t)c->rgba[i] : 0; }
    VkClearRect r = {scissor_rect(), 0, P_ri.layerCount ? P_ri.layerCount : 1};   // r76: every layer of a layered pass
    if ((c->mask & 0xf) != 0xf) {   // r76: a partial channel mask, drawn
        if (clear_draw(c->index, c->mask & 0xf, a.clearValue.color.uint32, f && f->is_int ? (f->is_sint ? 2 : 1) : 0, 0, 0.0f, 0, 0)) { if (rd) rd_mark(57); return; }
        VKRED_ONCE(31, "partial channel clear mask %#x cleared in full", c->mask); }
    const int gated = g_rec->X.rcg_on;   // occ-s3: under a GPU condition: vkCmdClearAttachments in a conditional block (a load op or a transfer clear would escape the condition)
    if (!gated && S.pass_open == 2 && (c->mask & 0xf) == 0xf && pass_whole(r.rect)) {   // the whole attachment before any draw: its load op
        if (clear_whole(c->index, (VkClearValue){.color = a.clearValue.color}, VK_IMAGE_ASPECT_COLOR_BIT)) { if (rd) rd_mark(56); return; }   // r76: a larger target or more layers than the pass
        P_ca[c->index].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; P_ca[c->index].clearValue.color = a.clearValue.color; P_cleared = 1; g_vkst.folded++; { void vk_clrdiag_fold(vkimg_t *); vk_clrdiag_fold(P_att[c->index].im); }   /* clrorder */ if (rd) rd_mark(56); return;
    }
    if (!pass_commit()) return;
    if (rd) rd_mark(58);
    scale_writes[c->index] |= VK_IMAGE_ASPECT_COLOR_BIT;
    r.rect = physical_rect(r.rect);
    scale_note_scissor(r.rect);
    if (r.rect.extent.width && r.rect.extent.height) { if (gated) rcg_begin(vk_cmd(), 1); vkCmdClearAttachments(vk_cmd(), 1, &a, 1, &r); }   // (occ-s3: gated; occ-s3fast: the block stays open for what follows under the same predicate)
    { void vk_clrdiag_done(vkimg_t *); vk_clrdiag_done(P_att[a.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT ? a.colorAttachment : 8].im); }   // clrorder
    if (rd) rd_mark(59);
}
static void clear_ds(const p_clear_ds *c) {
    const int rd = g_rdiag > 0; if (rd) rd_start();   // replaydiag: 50 begin_pass, 51 folded into the load ops / whole, 52 masked draw, 53 pass_commit, 54 vkCmdClearAttachments
    if (!begin_pass() || !S.daspect) { if (rd) rd_mark(50); return; }
    if (rd) rd_mark(50);
    VkImageAspectFlags as = (c->depth_mask ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (c->stencil_mask && (S.daspect & VK_IMAGE_ASPECT_STENCIL_BIT) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
    as &= S.daspect; if (!as) return;
    VkClearAttachment a = {as, 0, {.depthStencil = {c->depth, (uint32_t)c->stencil}}};
    VkClearRect r = {scissor_rect(), 0, P_ri.layerCount ? P_ri.layerCount : 1};   // r76: every layer of a layered pass
    if ((as & VK_IMAGE_ASPECT_STENCIL_BIT) && (c->stencil_mask & 0xff) != 0xff) {   // r76: a partial stencil mask, drawn (with the depth when it is cleared too)
        if (clear_draw(-1, 0, NULL, 0, (as & VK_IMAGE_ASPECT_DEPTH_BIT) != 0, c->depth, c->stencil_mask & 0xff, (uint32_t)c->stencil)) { if (rd) rd_mark(52); return; }
        VKRED_ONCE(169, "r76: stencil clear with mask %#x cleared in full", c->stencil_mask); }
    const int gated = g_rec->X.rcg_on;   // occ-s3: under a GPU condition: vkCmdClearAttachments in a conditional block (not a load op, not a transfer clear)
    if (!gated && S.pass_open == 2 && pass_whole(r.rect) && (!(as & VK_IMAGE_ASPECT_STENCIL_BIT) || (c->stencil_mask & 0xff) == 0xff)) {   // into the load ops
        if (clear_whole(8, (VkClearValue){.depthStencil = a.clearValue.depthStencil}, as)) { if (rd) rd_mark(51); return; }   // r76: a larger target or more layers than the pass
        if (as & VK_IMAGE_ASPECT_DEPTH_BIT) { P_da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; P_da.clearValue.depthStencil = a.clearValue.depthStencil; }
        if (as & VK_IMAGE_ASPECT_STENCIL_BIT) { P_sa.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; P_sa.clearValue.depthStencil = a.clearValue.depthStencil; }
        P_cleared = 1; g_vkst.folded++; { void vk_clrdiag_fold(vkimg_t *); vk_clrdiag_fold(P_att[8].im); }   /* clrorder */ if (rd) rd_mark(51); return;
    }
    if (!pass_commit()) return;
    if (rd) rd_mark(53);
    scale_writes[8] |= as;
    r.rect = physical_rect(r.rect);
    scale_note_scissor(r.rect);
    if (r.rect.extent.width && r.rect.extent.height) { if (gated) rcg_begin(vk_cmd(), 1); vkCmdClearAttachments(vk_cmd(), 1, &a, 1, &r); }   // (occ-s3: gated; occ-s3fast: the block stays open for what follows under the same predicate)
    { void vk_clrdiag_done(vkimg_t *); vk_clrdiag_done(P_att[a.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT ? a.colorAttachment : 8].im); }   // clrorder
    if (rd) rd_mark(54);
}

// ---------------- formats and enums ----------------
static VkFormat vertex_format(int f, int want_int, int want_sint) {
    static const uint16_t rgb[] = {VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_R8G8B8_SNORM, VK_FORMAT_R8G8B8_UINT, VK_FORMAT_R8G8B8_SINT, VK_FORMAT_R16G16B16_SFLOAT, VK_FORMAT_R16G16B16_UNORM,
        VK_FORMAT_R16G16B16_SNORM, VK_FORMAT_R16G16B16_UINT, VK_FORMAT_R16G16B16_SINT, VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT};
    static const uint16_t rgb_scaled[] = {VK_FORMAT_R8G8B8_USCALED, VK_FORMAT_R8G8B8_SSCALED, VK_FORMAT_R16G16B16_USCALED, VK_FORMAT_R16G16B16_SSCALED, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT};
    VkFormat v;
    if (f >= 0x19 && f <= 0x24) v = (VkFormat)rgb[f - 0x19];
    else if (f >= 0x5D && f <= 0x62) v = (VkFormat)rgb_scaled[f - 0x5D];
    else { const vkfmt_t *e = vk_fmt(f); v = e ? e->vk : VK_FORMAT_UNDEFINED; }
    // r75: an integer input fed a float-type format (normalized, half, packed, scaled) takes the float32 bits of the converted value, as Maxwell's fetch gives them and the
    // shaders take back as floats (nvn_mtl.m vconv_*). Reading the raw bytes as integers (UNORM -> UINT here) gave denormals: Koei Tecmo's skinning and UV inputs
    // (Age of Calamity's and Three Houses' soldiers flat, their UVs on one texel on the Mac). Adreno's fetch converts by the attribute format alone and the shader reads the
    // 32-bit register, so the float format is kept (Vulkan calls the type mismatch undefined; 32-bit floats give the same bits either way). DM_VK_VFMT_BITS=0: as before
    static int fbits = -1; if (fbits < 0) { const char *e = getenv("DM_VK_VFMT_BITS"); fbits = !e || atoi(e) != 0; }
    if (want_int && fbits && (v == VK_FORMAT_R8G8B8A8_UNORM || v == VK_FORMAT_R8G8B8A8_USCALED || v == VK_FORMAT_R16G16_UNORM || v == VK_FORMAT_R16G16_USCALED || v == VK_FORMAT_R16G16B16A16_UNORM || v == VK_FORMAT_R16G16B16A16_USCALED)) {
        static int n; if (n < 4) { n++; VKLOG("r75: vertex format %#x (VkFormat %d) into an integer input: fetched as its float type, the shader gets the float bits", f, v); }
        return v; }
    if (want_int) {   // the shader declared an integer input: the attribute must be an integer format of the same layout
        switch (v) {
        case VK_FORMAT_R32_SFLOAT: return want_sint ? VK_FORMAT_R32_SINT : VK_FORMAT_R32_UINT;
        case VK_FORMAT_R32G32_SFLOAT: return want_sint ? VK_FORMAT_R32G32_SINT : VK_FORMAT_R32G32_UINT;
        case VK_FORMAT_R32G32B32_SFLOAT: return want_sint ? VK_FORMAT_R32G32B32_SINT : VK_FORMAT_R32G32B32_UINT;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return want_sint ? VK_FORMAT_R32G32B32A32_SINT : VK_FORMAT_R32G32B32A32_UINT;
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_USCALED: return want_sint ? VK_FORMAT_R8G8B8A8_SINT : VK_FORMAT_R8G8B8A8_UINT;
        case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16_USCALED: return want_sint ? VK_FORMAT_R16G16_SINT : VK_FORMAT_R16G16_UINT;
        case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_USCALED: return want_sint ? VK_FORMAT_R16G16B16A16_SINT : VK_FORMAT_R16G16B16A16_UINT;
        default: return v;
        }
    }
    return v;
}
static const VkBlendFactor BF[0x65] = {
    [1] = VK_BLEND_FACTOR_ZERO, [2] = VK_BLEND_FACTOR_ONE, [3] = VK_BLEND_FACTOR_SRC_COLOR, [4] = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR, [5] = VK_BLEND_FACTOR_SRC_ALPHA,
    [6] = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, [7] = VK_BLEND_FACTOR_DST_ALPHA, [8] = VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA, [9] = VK_BLEND_FACTOR_DST_COLOR,
    [10] = VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR, [11] = VK_BLEND_FACTOR_SRC_ALPHA_SATURATE, [0x10] = VK_BLEND_FACTOR_SRC1_COLOR, [0x11] = VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
    [0x12] = VK_BLEND_FACTOR_SRC1_ALPHA, [0x13] = VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA, [0x61] = VK_BLEND_FACTOR_CONSTANT_COLOR, [0x62] = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
    [0x63] = VK_BLEND_FACTOR_CONSTANT_ALPHA, [0x64] = VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA};
static VkBlendFactor bfac(unsigned f) { return f < 0x65 && (f == 1 || BF[f]) ? BF[f] : VK_BLEND_FACTOR_ONE; }
static VkBlendOp bop(unsigned e) { return e >= 1 && e <= 5 ? (VkBlendOp)(e - 1) : VK_BLEND_OP_ADD; }
static VkCompareOp cmpop(unsigned f) { return f >= 1 && f <= 8 ? (VkCompareOp)(f - 1) : VK_COMPARE_OP_ALWAYS; }
static VkStencilOp sop(unsigned o) { return o >= 1 && o <= 8 ? (VkStencilOp)(o - 1) : VK_STENCIL_OP_KEEP; }
static VkPrimitiveTopology topo(int p) {
    switch (p) { case 0: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST; case 1: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST; case 2: case 3: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
                 case 5: case 8: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; case 6: case 9: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
                 case 0xA: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY; case 0xB: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
                 case 0xC: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY; case 0xD: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY; case 0xE: return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
                 default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; }
}

// ---------------- programs and pipelines ----------------
static uint64_t key64(const uint8_t k[8]) { uint64_t v = 0; for (int i = 0; i < 8; i++) v = v << 8 | k[i]; return v; }
// DM_NVN_SHDUMP (debugging; hle_nvn.c then keeps each program's control sections): a shader missing from the bundle is written to <files>/shader_<key>.bin as its 0x880-byte
// control section followed by its code area (control +0x704 bytes), so it can be looked for in the game files (Xenoblade Chronicles 3's title had one such program)
static void shader_dump(const prog_t *p, unsigned i, uint64_t kk) {
    static int on = -1; if (on < 0) on = getenv("DM_NVN_SHDUMP") != NULL;
    if (!on || i >= p->nstage || !p->sd[i].control || !p->sd[i].data) return;
    uint32_t ncode; memcpy(&ncode, (const uint8_t *)p->sd[i].control + 0x704, 4);
    const char *dir = getenv("DM_ROOT"); char path[1024]; snprintf(path, sizeof path, "%s/shader_%016llx.bin", dir ? dir : ".", (unsigned long long)kk);
    FILE *f = fopen(path, "wb"); if (!f) return;
    fwrite(p->sd[i].control, 1, 0x880, f); if (ncode < (1u << 24)) fwrite((const void *)(uintptr_t)p->sd[i].data, 1, ncode, f); fclose(f);
    VKLOG("shader %016llx written to %s", (unsigned long long)kk, path);
}
static void bind_program_on(rec_state_t *s, const p_bind_program *bp) {   // S1b (parallel recording): on a given state (S, or the dry pass's copy)
    // NVN binds by stage mask (vertex 1, fragment 2, geometry 4, tessellation 8 / 0x10, compute 0x20): only the masked stages change, so a game that keeps
    // its vertex and fragment shaders in programs of their own (NEO, AI: THE SOMNIUM FILES) binds them one after the other and draws with both.
    // prog_key's index is the position in the program's shader array (as the game passed them to SetShaders), not the stage: the stage is the bundle's.
    int vk_shader_stage(uint64_t key);
    uint32_t mask = bp->stages ? (uint32_t)bp->stages & 0xff : 0xff;
    for (int st = 0; st < 6; st++) if (mask >> st & 1) s->sh[st] = NULL;
    if (mask & (1u << 6)) s->sh[3] = NULL;
    if (mask & (1u << 7)) s->sh[4] = NULL;
    s->prog = bp->prog;
    if (s->prog) for (unsigned i = 0; i < 6; i++) {
        uint8_t k[8]; if (!prog_key(s->prog, i, k)) break;
        uint64_t kk = key64(k); int st = kk ? vk_shader_stage(kk) : -1;
        if (kk && st < 0) {   // the draw or dispatch is skipped. Logged once per shader (the first 16 binds were 4 shaders of Xenoblade Chronicles 3 over and over, which hid others)
            static uint64_t seen[256]; static int nseen; int known = 0; for (int j = 0; j < nseen; j++) if (seen[j] == kk) { known = 1; break; }
            if (!known && nseen < 256) { seen[nseen++] = kk; void vk_shader_missing(uint64_t key); vk_shader_missing(kk); shader_dump(s->prog, i, kk); } }
        if (st < 0 || st > 7 || !(mask >> st & 1)) continue;
        s->sh[st >= 6 ? st - 3 : st] = vk_shader(kk, st);
    }
    s->prog_ok = s->sh[0] || s->sh[5] || (s->sh[4] && s->sh[4]->stage == 7);   // a vertex-only program draws depth only (prepare_draw)
}
static void bind_program(const p_bind_program *bp) { bind_program_on(&S, bp); }
typedef struct {
    uint64_t key[5]; uint32_t topo, restart, raster_discard, cull, front, mode, depth_clamp, bias_en;
    uint32_t dt, dw, df, st, sf[2], sfail[2], szfail[2], szpass[2];
    uint32_t nattr; struct { uint32_t loc, binding, fmt, off; } attr[16];
    uint32_t nbind; struct { uint32_t binding, stride, rate, divisor; } bind[17];
    uint32_t ncol, cfmt[8], dfmt, samples, a2c, blend_en, logic_en, logic_op, wmask[8];
    struct { uint32_t src, dst, srca, dsta, eq, eqa; } bl[8];
    uint32_t clip_neg1, alt, fdz, atest;   // fdz: the fragment shader's variant that writes gl_FragDepth = gl_FragCoord.z (fdz below); atest (r56, was padding): its alpha test variant
    uint64_t um[5], im[5];   // each stage's sampled bindings given integer views (vk_shader_module)
    uint32_t patch, dbt;   // r69: control points per patch (tessellation stages 3 / 4 in key), depth bounds test
} pkey;
static nvn_hmap g_pipes = {.klen = sizeof(pkey)}; static pthread_mutex_t g_pipe_mu = PTHREAD_MUTEX_INITIALIZER;   // the replay thread and the prewarm threads
static VkPipeline build_pipeline(const pkey *k, VkPipelineLayout layout, vkshader_t *const stg[5]);
static void pipe_record(const pkey *k);
// The last key and its pipeline (the replay thread's only; pipelines are never dropped): runs of draws with one pipeline skip the 992-byte hash and the lock
// S1c (parallel recording): kept the replay thread's (a per-thread copy would put 992 bytes of TLS into every thread of the process): another recording thread's
// context skips the memo and reads the map under g_pipe_mu (the bundle path; parallel recording is Eden mode only, seg_refuse)
static pkey g_last_k; static VkPipeline g_last_p;
static VkPipeline graphics_pipeline(const pkey *k, VkPipelineLayout layout) {
    const int memo = g_rec == &g_rec_main;
    if (memo && g_last_p && !memcmp(k, &g_last_k, sizeof *k)) return g_last_p == (void *)1 ? VK_NULL_HANDLE : g_last_p;
    pthread_mutex_lock(&g_pipe_mu); VkPipeline p = nvn_hm_get(&g_pipes, k); pthread_mutex_unlock(&g_pipe_mu);
    if (p) { if (memo) { g_last_k = *k; g_last_p = p; } return p == (void *)1 ? VK_NULL_HANDLE : p; }
    vkshader_t *stg[5] = {S.sh[0], S.sh[1], S.sh[2], S.sh[3], S.sh[4]};
    { uint64_t t80 = tnow(); p = build_pipeline(k, layout, stg); r80_made(t80, (const struct pkey *)k, NULL); }   // r80
    pthread_mutex_lock(&g_pipe_mu);
    VkPipeline had = nvn_hm_get(&g_pipes, k);
    if (had) { if (p != (void *)1) vkDestroyPipeline(V.dev, p, NULL); p = had; }   // a prewarm thread made it meanwhile
    else { nvn_hm_put(&g_pipes, k, p); if (p != (void *)1) pipe_record(k); }
    pthread_mutex_unlock(&g_pipe_mu);
    return p == (void *)1 ? VK_NULL_HANDLE : p;
}
// ---- pipeline prewarm: every pipeline the game needed is recorded (DM_ROOT/vk_pipelines.bin) and made again at the next boot on background
// threads, before the game reaches it, so a scene seen once never stalls a frame on a pipeline compile again (the driver's compile, not the
// pipeline cache's lookup, is the cost: tens of milliseconds per pipeline on Adreno). DM_VK_NOPREWARM=1 turns it off.
#define PIPE_MAGIC 0x31455050u   // "PPE1"; the header also holds sizeof(pkey), so a changed key layout starts a new file
static FILE *g_prec;
static void pipe_record(const pkey *k) {   // under g_pipe_mu
    if (!g_prec || k->alt) return;   // DM_VK_SPV_ALT's pipelines are a debugging run's own
    fwrite(k, sizeof *k, 1, g_prec); fflush(g_prec);
}
static pkey *g_pw; static size_t g_pw_n; static volatile size_t g_pw_next, g_pw_done;
static void *prewarm_thread(void *arg) {
    (void)arg; pthread_setname_np(pthread_self(), "vk prewarm");
    setpriority(PRIO_PROCESS, 0, 10);   // below the game's threads: prewarm uses idle cores
    for (;;) {
        size_t i = __atomic_fetch_add(&g_pw_next, 1, __ATOMIC_RELAXED); if (i >= g_pw_n) break;
        const pkey *k = &g_pw[i];
        pthread_mutex_lock(&g_pipe_mu); int have = nvn_hm_get(&g_pipes, k) != NULL; pthread_mutex_unlock(&g_pipe_mu);
        if (have) { __atomic_add_fetch(&g_r80_pw_have, 1, __ATOMIC_RELAXED); continue; }   // r80: counted
        vkshader_t *stg[5] = {0}; int ok = 1;
        for (int s = 0; s < 5; s++) if (k->key[s] && !(s == 1 && k->raster_discard)) { stg[s] = vk_shader(k->key[s], s); if (!stg[s]) ok = 0; }
        if (!ok || !stg[0]) { __atomic_add_fetch(&g_r80_pw_fail, 1, __ATOMIC_RELAXED); continue; }   // r80: counted
        VkDescriptorSetLayout dsl[5] = {0}; int nsets = 0;
        for (int s = 0; s < 5; s++) if (stg[s]) { dsl[stg[s]->set] = stg[s]->dsl; if (stg[s]->set + 1 > nsets) nsets = stg[s]->set + 1; }
        VkPipeline p = build_pipeline(k, vk_pipeline_layout(dsl, nsets), stg);
        pthread_mutex_lock(&g_pipe_mu);
        if (!nvn_hm_get(&g_pipes, k)) nvn_hm_put(&g_pipes, k, p); else if (p != (void *)1) vkDestroyPipeline(V.dev, p, NULL);
        pthread_mutex_unlock(&g_pipe_mu);
        size_t d = __atomic_add_fetch(&g_pw_done, 1, __ATOMIC_RELAXED);
        if (d == g_pw_n || d % 500 == 0) VKLOG("prewarm: %zu of %zu pipelines", d, g_pw_n);
    }
    if (__atomic_sub_fetch(&g_r80_pw_live, 1, __ATOMIC_ACQ_REL) == 0)   // r80: the last prewarm thread
        VKLOG("r80: prewarm finished in %.1f s: %zu of %zu pipelines made, %llu made already by the game, %llu with a shader missing", (tnow() - g_r80_pw_t0) / 1e9, (size_t)__atomic_load_n(&g_pw_done, __ATOMIC_RELAXED), g_pw_n,
              (unsigned long long)__atomic_load_n(&g_r80_pw_have, __ATOMIC_RELAXED), (unsigned long long)__atomic_load_n(&g_r80_pw_fail, __ATOMIC_RELAXED));
    return NULL;
}
static void r80_made(uint64_t t0, const struct pkey *kk, const struct vkshader *cs) {   // r80 (diagnostic): a pipeline made on the replay thread
    const pkey *k = (const pkey *)kk; uint64_t ns = tnow() - t0;
    __atomic_add_fetch(&g_r80_pso, 1, __ATOMIC_RELAXED); __atomic_add_fetch(&g_r80_pso_ns, ns, __ATOMIC_RELAXED);
    if (ns < 100000000ull) return; static int n; if (n++ >= 64) return;
    size_t nx = __atomic_load_n(&g_pw_next, __ATOMIC_RELAXED); if (nx > g_pw_n) nx = g_pw_n;
    if (k) { int in = 0; for (size_t i = 0; i < g_pw_n; i++) if (!memcmp(&g_pw[i], k, sizeof *k)) { in = 1; break; }
        VKLOG("r80: graphics pipeline made in %.0f ms on the replay thread (shaders %016llx %016llx %016llx); in the prewarm list: %s; prewarm taken %zu of %zu", ns / 1e6,
              (unsigned long long)k->key[0], (unsigned long long)k->key[1], (unsigned long long)k->key[2], in ? "yes" : "no", nx, g_pw_n); }
    else VKLOG("r80: compute pipeline made in %.0f ms on the replay thread (shader %016llx)", ns / 1e6, (unsigned long long)((const vkshader_t *)cs)->key);
}
static void prewarm_start(void) {
    // NXN1X lane B: in Eden mode the prewarm is Eden's own (PipelineCache::LoadDiskResources builds vulkan.bin at boot, NPARK 2026-11-12); the host's
    // vk_pipelines.bin keys name bundle shaders, so it is neither read nor rewritten (the file stays for DM_VK_EDEN=0)
    if (vk_eden_mode()) { VKLOG("prewarm: Eden mode, pipelines come from Eden's vulkan.bin (vk_pipelines.bin not used)"); return; }
    const char *root = getenv("DM_ROOT"); if (!root) return;
    char path[1024]; snprintf(path, sizeof path, "%s/vk_pipelines.bin", root);
    uint32_t hdr[2] = {PIPE_MAGIC, (uint32_t)sizeof(pkey)}, fh[2] = {0};
    FILE *f = fopen(path, "rb");
    if (f && fread(fh, sizeof fh, 1, f) == 1 && fh[0] == hdr[0] && fh[1] == hdr[1]) {
        fseek(f, 0, SEEK_END); long n = (ftell(f) - (long)sizeof fh) / (long)sizeof(pkey); fseek(f, sizeof fh, SEEK_SET);
        if (n > 0) { g_pw = malloc((size_t)n * sizeof(pkey)); g_pw_n = fread(g_pw, sizeof(pkey), (size_t)n, f); }
    }
    if (f) fclose(f);
    // rewrite the file with the records read (dropping a torn last record), then append to it
    g_prec = fopen(path, "wb"); if (g_prec) { fwrite(hdr, sizeof hdr, 1, g_prec); if (g_pw_n) fwrite(g_pw, sizeof(pkey), g_pw_n, g_prec); fflush(g_prec); }
    if (!g_pw_n || getenv("DM_VK_NOPREWARM")) { VKLOG("prewarm: %zu pipelines recorded%s", g_pw_n, g_pw_n ? ", off (DM_VK_NOPREWARM)" : ""); return; }
    long nc = sysconf(_SC_NPROCESSORS_ONLN); int nt = nc >= 8 ? 3 : nc >= 4 ? 2 : 1;
    VKLOG("prewarm: %zu recorded pipelines on %d threads", g_pw_n, nt); g_r80_pw_t0 = tnow(); g_r80_pw_live = nt;   /* r80 */
    for (int i = 0; i < nt; i++) { pthread_t t; pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED); pthread_create(&t, &a, prewarm_thread, NULL); pthread_attr_destroy(&a); }
}
static int gs_pass_mode(void) {   // DM_VK_GS_PASS (build_pipeline)
    static int m = -1; if (m < 0) { const char *e = getenv("DM_VK_GS_PASS"); m = e ? atoi(e) : 1;   /* r53: 1 is the default (r21: the picture is right with it) */ if (m) VKLOG("DM_VK_GS_PASS=%d: passthrough geometry shaders pass on the vertex shader's outputs the fragment shader reads%s", m, m == 2 ? " (through locations below 16)" : m == 3 ? " (diagnostic: zeros, no inputs read)" : m == 4 ? " (diagnostic: far constant, no inputs read)" : ""); }
    return m;
}
static VkPipeline build_pipeline(const pkey *k, VkPipelineLayout layout, vkshader_t *const stg[5]) {
    VkPipeline p;
    VkPipelineShaderStageCreateInfo ss[5]; uint32_t ns = 0;
    VkShaderStageFlagBits fl[5] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_GEOMETRY_BIT, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT};
    // the outputs each stage must declare for the next (vk_shader_module): vertex → geometry or fragment, geometry → fragment
    const vkshader_t *fs = stg[1] && !k->raster_discard ? stg[1] : NULL; uint32_t xo[5] = {0};
    // r69 tessellation (Bayonetta 3: 11 hull / domain shaders, clouds of erasure and Homunculi; Witch-Time STATUS "Tessellation"): vertex -> control -> evaluation -> (geometry) -> fragment.
    // The control stage's outputs are arrays per vertex, which spv_add_outputs does not declare: none added there
    const int tess = stg[3] && stg[4];
    if (stg[0]) xo[0] = (tess ? stg[3]->in_locs : stg[2] ? stg[2]->in_locs : fs ? fs->in_locs : 0) & ~stg[0]->out_locs;
    if (tess) xo[4] = (stg[2] ? stg[2]->in_locs : fs ? fs->in_locs : 0) & ~stg[4]->out_locs;
    if (stg[2]) xo[2] = (fs ? fs->in_locs : 0) & ~stg[2]->out_locs;
    // of those, the ones a passthrough geometry shader passes on from the vertex shader (gfx_vk_shader.c spv_gs_pass), by DM_VK_GS_PASS: 0 (default) not (left undefined),
    // 1 read at their own locations, 2 copied by the vertex shader to free locations below 16 (xr) and read there (spv_vs_dup), 3 / 4 (diagnostics) zeros / far constants written, nothing read. Donkey Kong Bananza 11-04: 1 turned the whole
    // scene to garbage triangles (its passthrough geometry shader builds the index list of the visible triangles from location 0, read wrong once location 19 was an input)
    uint32_t xp = 0, xr = 0; int gpm = gs_pass_mode();
    if (gpm && stg[2] && stg[0] && !tess) {
        xp = xo[2] & stg[0]->out_locs;
        if (xp && gpm == 3) xr = ~0u;   // diagnostic: the geometry shader writes zeros to those outputs and reads no more inputs
        if (xp && gpm == 4) xr = ~1u;   // diagnostic: (1e6, 1e6, 1e6, 1) (Donkey Kong Bananza 7f77 reads location 19 as a distance: far, no fade discard)
        if (xp && gpm == 2) { uint32_t fr = 0xffffu & ~stg[0]->out_locs & ~stg[2]->in_locs;
            for (int l = 0; l < 16 && __builtin_popcount(xr) < __builtin_popcount(xp); l++) if (fr >> l & 1) xr |= 1u << l;
            if (__builtin_popcount(xr) < __builtin_popcount(xp)) { VKRED_ONCE(97, "DM_VK_GS_PASS=2: no free location below 16 for outputs %#x: not passed on", xp); xp = xr = 0; } }
    }
    for (int i = 0; i < 5; i++) if (stg[i] && !(i == 1 && k->raster_discard) && (i < 3 || tess)) ss[ns++] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, fl[i], vk_shader_module(stg[i], k->um[i], k->im[i], k->alt && !xo[i] && ((!k->um[i] && !k->im[i]) || vk_spv_alt_has(stg[i])) ? 1u << 31 : xo[i], i == 1 ? (k->fdz ? 1 : 0) | ((k->atest & 3) == 2 ? 4 : (k->atest & 3) ? 2 : 0) | (k->atest & 8 ? 8 | (int)(k->atest >> 8 & 0xff00) | (int)(k->atest >> 20 & 16) : 0) : i == 0 && (k->atest & 4) ? 8 | (int)(k->atest & 0xff00) | (int)(k->atest >> 20 & 16) : 0, i == 2 || (i == 0 && xr) ? xp : 0, i == 1 || i > 2 ? 0 : xr), "main", NULL};
    VkVertexInputAttributeDescription ad[16]; VkVertexInputBindingDescription bd[17]; VkVertexInputBindingDivisorDescriptionEXT dd[17]; uint32_t ndd = 0;
    for (uint32_t i = 0; i < k->nattr; i++) ad[i] = (VkVertexInputAttributeDescription){k->attr[i].loc, k->attr[i].binding, (VkFormat)k->attr[i].fmt, k->attr[i].off};
    for (uint32_t i = 0; i < k->nbind; i++) { bd[i] = (VkVertexInputBindingDescription){k->bind[i].binding, k->bind[i].stride, (VkVertexInputRate)k->bind[i].rate};
        if (k->bind[i].rate && k->bind[i].divisor > 1) dd[ndd++] = (VkVertexInputBindingDivisorDescriptionEXT){k->bind[i].binding, k->bind[i].divisor}; }
    VkPipelineVertexInputDivisorStateCreateInfoEXT dvs = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT, NULL, ndd, dd};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, ndd ? &dvs : NULL, 0, k->nbind, bd, k->nattr, ad};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, (VkPrimitiveTopology)k->topo, k->restart};
    VkPipelineViewportDepthClipControlCreateInfoEXT dcc = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT, NULL, k->clip_neg1};
    VkPipelineViewportStateCreateInfo vs = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, k->clip_neg1 && V.has_depth_clip_ctl ? &dcc : NULL, 0, 1, NULL, 1, NULL};
    VkPipelineRasterizationDepthClipStateCreateInfoEXT dce = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT, NULL, 0, VK_TRUE};   // r99c DM_VK_DEPTH_CLIP=1 (gfx_vk_core.c)
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, V.has_depth_clip_en && k->depth_clamp ? &dce : NULL, 0, k->depth_clamp, k->raster_discard,
        k->mode == 0 ? VK_POLYGON_MODE_POINT : k->mode == 1 ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL, k->cull, k->front ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE,
        k->bias_en, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, (VkSampleCountFlagBits)k->samples, 0, 0, NULL, k->a2c, 0};
    VkStencilOpState sf = {sop(k->sfail[0]), sop(k->szpass[0]), sop(k->szfail[0]), cmpop(k->sf[0]), 0xff, 0xff, 0}, sb = {sop(k->sfail[1]), sop(k->szpass[1]), sop(k->szfail[1]), cmpop(k->sf[1]), 0xff, 0xff, 0};
    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, NULL, 0, k->dt, k->dt && k->dw, cmpop(k->df), k->dbt, k->st, sf, sb, 0, 1};   // r69: depth bounds (dynamic values)
    // r69: NVN's tessellator is OpenGL's: the domain's origin lower left (as Ryujinx and Witch-Time set it). DM_VK_TESS_ORIGIN=0 (at start, A/B): upper left
    // r82: upper left by default. Our translator's evaluation stages already say VertexOrderCw for the game's GL_CCW (Bayonetta 3 TES 152e86c9 / de621a43), so lower left
    // turned every patch around and back-face culling left fragments of the clouds (glowing-flower fog, city smoke). DM_VK_TESS_ORIGIN=1: lower left (r69)
    static int tess_ul = -1; if (tess_ul < 0) { const char *e = getenv("DM_VK_TESS_ORIGIN"); tess_ul = !(e && *e == '1'); }
    VkPipelineTessellationDomainOriginStateCreateInfo tdo = {VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO, NULL, tess_ul ? VK_TESSELLATION_DOMAIN_ORIGIN_UPPER_LEFT : VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT};
    VkPipelineTessellationStateCreateInfo tsi = {VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO, &tdo, 0, k->patch ? k->patch : 3};
    VkPipelineColorBlendAttachmentState cba[8];
    for (uint32_t i = 0; i < k->ncol; i++) cba[i] = (VkPipelineColorBlendAttachmentState){(k->blend_en >> i) & 1, bfac(k->bl[i].src), bfac(k->bl[i].dst), bop(k->bl[i].eq),
        bfac(k->bl[i].srca), bfac(k->bl[i].dsta), bop(k->bl[i].eqa), k->wmask[i]};
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, k->logic_en, (VkLogicOp)k->logic_op, k->ncol, cba, {0, 0, 0, 0}};
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
                            VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_LINE_WIDTH, VK_DYNAMIC_STATE_DEPTH_BOUNDS};
    VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, sizeof dyn / sizeof *dyn, dyn};
    VkFormat cf[8]; for (uint32_t i = 0; i < k->ncol; i++) cf[i] = (VkFormat)k->cfmt[i];
    VkFormat df = (VkFormat)k->dfmt;
    int has_stencil = df == VK_FORMAT_D24_UNORM_S8_UINT || df == VK_FORMAT_D32_SFLOAT_S8_UINT || df == VK_FORMAT_S8_UINT;
    VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, k->ncol, cf, df == VK_FORMAT_S8_UINT ? VK_FORMAT_UNDEFINED : df, has_stencil ? df : VK_FORMAT_UNDEFINED};
    VkRenderPass prp = rp_on() ? rp_compat(k->ncol, k->cfmt, k->dfmt, k->samples) : VK_NULL_HANDLE;   // with render passes, pipelines are made against a compatible one
    if (!prp && !V.has_dynrender) return (VkPipeline)1;   // no compatible render pass and no dynamic rendering: no valid pipeline (pass_commit skips such passes)
    VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, prp ? NULL : &rci, 0, ns, ss, &vi, &ia, tess ? &tsi : NULL, &vs, &rs, &ms, &ds, &cb, &dy, layout, prp, 0, VK_NULL_HANDLE, -1};
    VkResult r = vkCreateGraphicsPipelines(V.dev, V.pcache, 1, &ci, NULL, &p);
    static uint64_t n; uint64_t c = __atomic_add_fetch(&n, 1, __ATOMIC_RELAXED);
    if (r) { LOG("[RED] [vk] graphics pipeline failed (%d): shaders %016llx %016llx %016llx", r, (unsigned long long)k->key[0], (unsigned long long)k->key[1], (unsigned long long)k->key[2]); if (tess) LOG("[RED] [vk]   tessellation %016llx %016llx, patch %u", (unsigned long long)k->key[3], (unsigned long long)k->key[4], k->patch); p = (VkPipeline)1; }
    else if (c % 200 == 0) VKLOG("%llu graphics pipelines", (unsigned long long)c);
    __atomic_add_fetch(&g_vkst.pipes_built, 1, __ATOMIC_RELAXED);
    return p;   // (VkPipeline)1 = failed
}

// ---------------- descriptors ----------------
static uint64_t c2_handle(int st, int off) {   // the driver's handle table c2, as dmsh_build.py lays it out
    if (off >= 0x20 && off < 0x120) return S.tex[st][(off - 0x20) / 8];
    if (off >= 0x120 && off < 0x160) return S.img[st][(off - 0x120) / 8];
    if (off >= 0x168 && off < 0x568) return S.tex[st][128 + (off - 0x168) / 8];
    if (off >= 0x568 && off < 0x968) return S.samp[st][(off - 0x568) / 8];
    return 0;
}
static uint32_t ubo_word(int st, int reg, int off) {
    const p_bind_buf *u = reg >= 0 && reg < 16 ? &S.ubo[st][reg] : NULL; uint32_t v = 0;
    if (u && u->addr && off >= 0 && (uint64_t)off + 4 <= (u->size ? u->size : 4) && vk_pool_of(u->addr + off, 4)) vk_ubd_read(u->addr + off, 4, (uint8_t *)&v);   // with the deferred updates
    return v;
}
// r90: a handle read from a uniform buffer names IDs of the pools bound when the draw runs (nvn_mtl.m ubo_tex_id / ubo_samp_id, state_op OP_SET_TEX_POOL). Bound handles are
// moved into the host numbering at recording (hle_nvn.c bind_handles); these were looked up in the first pool's numbering. Hyrule Warriors: Age of Imprisonment has a second
// texture pool (IDs from 33024): its reward icons showed a character portrait. DM_VK_UBO_POOL=0: as before
static int ubo_pool_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_UBO_POOL"); v = !e || atoi(e) != 0; } return v; }
static uint64_t ubo_handle(int st, int reg, int off) {
    uint32_t h = ubo_word(st, reg, off); if (!ubo_pool_on()) return h;
    uint64_t t = h & 0xfffff, s = h >> 20;
    if (t) t += (uint64_t)S.tex_base; if (s) s += (uint64_t)S.samp_base;
    return s << 20 | t;
}
static uint32_t g_dispatch[3]; static _Thread_local uint32_t g_draw_binst; /* S1c (parallel recording): the draw being recorded's base instance, per thread */ static int g_dispatch_ind; static uint64_t g_dispatch_indaddr;   // g_dispatch_ind (hybrid2): the dispatch's group counts are read by the GPU
#define SB_ALIGN 256   // storage buffers are bound from this boundary below their address (dm/android/tools/shader_vk.py SB_ALIGN, which the shaders were translated for)
static void nvn2_cs_c0(uint8_t *d);
static int storage_table_offset(int st) {
    static const int offsets[] = {0x110, 0x510, 0x410, 0x210, 0x310, 0x310};
    if ((st == 3 || st == 4) && S.sh[st] && S.sh[st]->stage >= 6)
        return st == 3 ? 0x670 : 0x770;
    if ((st == 3 || st == 4) && !vk_eden_mode()) return -1;
    return st >= 0 && st < 6 ? offsets[st] : -1;
}
static void drv_segment(int st, int bank, uint8_t *d) {
    memset(d, 0, 0x1000);
    if (bank == DMSH_PUSH_BANK) { memcpy(d, S.push[st == 5], DMSH_PUSH_SIZE); return; }
    if (bank == 0) {
        int base = storage_table_offset(st);
        if (base >= 0) for (int r = 0; r < 16; r++) { uint64_t a = S.ssbo[st][r].addr; uint32_t n = (uint32_t)S.ssbo[st][r].size; memcpy(d + base + 16 * r, &a, 8); memcpy(d + base + 16 * r + 8, &n, 4); }
        if (st != 5) { const float mp = 2047.0f; memcpy(d + 0x610, &mp, 4); memcpy(d + 0x644, &g_draw_binst, 4); }
        else { memcpy(d + 0x200, g_dispatch, 12); if (dm_sdk_ounce()) memcpy(d + DMSH_C0_DISPATCH_NVN2, g_dispatch, 12); for (int r = 0; r < 16; r++) { uint64_t a = S.ubo[5][r].addr; uint32_t n = (uint32_t)S.ubo[5][r].size; memcpy(d + 0x210 + 16 * r, &a, 8); memcpy(d + 0x210 + 16 * r + 8, &n, 4); } }
        if (st == 5 && vk_eden_mode() && dm_sdk_ounce()) nvn2_cs_c0(d);   // NVN2 compute: its own c[0] layout
    } else for (int off = 0x20; off < 0x968; off += 8) { uint64_t h = c2_handle(st, off); memcpy(d + off, &h, 8); }
}
// NVN2 compute (Eden mode, Switch 2 boots): the game's Ampere compute shaders read their driver constants from a c[0] laid out per shader (Ryujinx InstEmitAmpere.cs:260-312
// AMapC0, host_nx2 dmsh_build.py:590-593): from 0x10 the push constants (control +0xa9 words), then 0x10 bytes per uniform buffer slot (+0xb1) and per storage buffer slot
// (+0xb2), then 8-byte handles per texture slot (+0xb3), separate texture (+0xb4), separate sampler (+0xb5) and image slot (+0xb6), the whole +0xba bytes. The bundle's
// translator moved those to NVN's fixed places (c[24], c[0] + 0x210 / 0x310, c[2] + 0x20 / 0x168 / 0x568 / 0x120); Eden translates the lowered code as it is, so here
// c[0] is laid out that way from what drv_segment writes at those places. Offsets past the layout keep the fixed segment's bytes (as AMapC0 leaves them)
static void nvn2_cs_c0(uint8_t *d) {
    const vkshader_t *s = S.sh[5]; uint32_t L[8];
    if (!s || !s->eden || !nvn_eden_shader_nvn2_c0(s->eden, L)) return;
    uint32_t push = (L[0] + 0xf) & ~0xfu, ub = 0x10 + push, sb = ub + 0x10 * L[1], hb = sb + 0x10 * L[2], stx = hb + 8 * L[3], ssm = stx + 8 * L[4], ib = ssm + 8 * L[5], end = ib + 8 * L[6];
    if (L[0] > DMSH_PUSH_SIZE || L[1] > 16 || L[2] > 16 || end > 0x648 || 0x20 + 8 * L[3] > 0x168 || 0x168 + 8 * L[4] > 0x568 || 0x568 + 8 * L[5] > 0x968 || 0x120 + 8 * L[6] > 0x168) {
        VKRED_ONCE(172, "NVN2 compute c0 layout of a form not handled (push %u B, %u uniform, %u storage, %u textures, %u separate textures, %u separate samplers, %u images, end %#x): fixed tables kept",
            L[0], L[1], L[2], L[3], L[4], L[5], L[6], end);
        return; }
    uint8_t f[0x1000], c2[0x1000]; memcpy(f, d, sizeof f); drv_segment(5, 2, c2);
    memset(d + 0x10, 0, end - 0x10);
    memcpy(d + 0x10, S.push[1], L[0]);
    for (uint32_t r = 0; r < L[1]; r++) memcpy(d + ub + 16 * r, f + 0x210 + 16 * r, 16);
    for (uint32_t r = 0; r < L[2]; r++) memcpy(d + sb + 16 * r, f + 0x310 + 16 * r, 16);
    for (uint32_t r = 0; r < L[3]; r++) memcpy(d + hb + 8 * r, c2 + 0x20 + 8 * r, 8);
    for (uint32_t r = 0; r < L[4]; r++) memcpy(d + stx + 8 * r, c2 + 0x168 + 8 * r, 8);
    for (uint32_t r = 0; r < L[5]; r++) memcpy(d + ssm + 8 * r, c2 + 0x568 + 8 * r, 8);
    for (uint32_t r = 0; r < L[6]; r++) memcpy(d + ib + 8 * r, c2 + 0x120 + 8 * r, 8);
    static int said; if (!said++) VKLOG("NVN2 compute c0: per-shader layout built for Eden (first: push %u B, %u uniform, %u storage, %u textures, %u separate textures, %u separate samplers, %u images; %#x bytes, control size %#x)",
        L[0], L[1], L[2], L[3], L[4], L[5], L[6], end, L[7]);
}
// vk_samp_by_id without the registry lock while no sampler ID was registered with other parameters (g_samp_gen): a 32-slot texture array took the lock
// per slot, a lock the game thread takes for every texture it registers (DM_NVN_ASYNC). VkSamplers are never destroyed, so a kept handle stays valid
extern uint64_t g_samp_gen;
static VkSampler samp_cached(int sid, int *cmp) {   // r46: and the sampler's compare mode
    enum { SC = 1024 }; static struct { uint64_t gen; int sid, cmp; VkSampler s; } c[SC]; unsigned k = (unsigned)sid % SC;
    uint64_t g = __atomic_load_n(&g_samp_gen, __ATOMIC_ACQUIRE);   // read before the lookup: a registration racing it leaves the entry outdated, never wrong and current
    if (c[k].gen == g && c[k].sid == sid) { *cmp = c[k].cmp; return c[k].s; }
    VkSampler s = vk_samp_by_id(sid); c[k].gen = g; c[k].sid = sid; c[k].s = s; c[k].cmp = *cmp = vk_samp_cmp(sid); return s;
}
// r46: a sampled binding whose sampler's compare differs from the shader's Dref (vkshader_t.dref): logged once per shader, binding and direction;
// DM_VK_CMP_FIX=1 (live) gives it a sampler with the compare the shader reads with (Eglantine HWAoI sampler comparison fix)
static VkSampler cmp_check(int st, const dmsv_bind *b, int sid, int cmp, VkSampler s) {
    vkshader_t *sh = S.sh[st]; if (!sh || b->desc != RD_SAMPLED || b->binding >= 128) return s;
    int want = (int)(sh->dref[b->binding >> 6] >> (b->binding & 63) & 1); if (want == (cmp == 1)) return s;
    g_cmp_n++;
    { enum { SEEN = 512 }; static uint64_t seen[SEEN]; static int nseen; uint64_t k = sh->key ^ ((uint64_t)b->binding << 1 | (uint64_t)want) * 0x9e3779b97f4a7c15ull; int f = 0;
      for (int i = 0; i < nseen; i++) if (seen[i] == k) { f = 1; break; }
      if (!f && nseen < SEEN) { seen[nseen++] = k; VKLOG("cmp mismatch %d: shader %016llx stage %d binding %d: sampler %d compare %d, shader %s (fix %s, %llu so far)", nseen, (unsigned long long)sh->key, st, b->binding, sid, cmp,
                                                       want ? "reads with Dref" : "reads without Dref", g_cmp_fix ? "on" : "off", (unsigned long long)g_cmp_n); } }
    if (!g_cmp_fix) return s;
    enum { VC = 256 }; static struct { uint64_t gen; int sid, want; VkSampler s; } vc[VC]; unsigned k = ((unsigned)sid * 2u + (unsigned)want) % VC;
    uint64_t g = __atomic_load_n(&g_samp_gen, __ATOMIC_ACQUIRE);
    if (!(vc[k].gen == g && vc[k].sid == sid && vc[k].want == want)) { vc[k].gen = g; vc[k].sid = sid; vc[k].want = want; vc[k].s = vk_samp_cmp_variant(sid, want); }
    return vc[k].s ? vc[k].s : s;
}
static VkSampler samp_for(int st, const dmsv_bind *b, uint64_t texh) {
    int sid = 0;
    switch (b->samp) {
    case RS_TEX: sid = (int)(texh >> 20); break;
    case RS_SLOT: sid = b->samp_arg >= 0 && b->samp_arg < 128 ? (int)(S.samp[st][b->samp_arg] >> 20) : 0; break;
    case RS_ID: sid = b->samp_arg; break;
    case RS_UBO: sid = (int)(ubo_handle(st, b->samp_arg & 0xff, b->samp_arg >> 8) >> 20); break;   // r90
    default: break;
    }
    int cmp = 0; VkSampler s = sid ? samp_cached(sid, &cmp) : VK_NULL_HANDLE;
    if (s) s = cmp_check(st, b, sid, cmp, s);
    if (g_tracing) { char d[256]; void vk_samp_describe(int sid, char *out, size_t n); vk_samp_describe(sid, d, sizeof d); VKLOG("  binding %d: %s", b->binding, d); }
    return s ? s : vk_dummy_sampler();
}
// texture views by ID within a submission: a draw's texture arrays (32 slots) looked every slot up again, each through the registry's and the view table's
// locks (NEO: about 2 ms a frame). An entry holds while no texture ID is registered again and no view is dropped (g_tex_gen)
// R1 (parallel recording): TVC and g_tvc in struct rec_ctx (top of the file)
extern _Thread_local int g_vk_storage_want;   // R1 (parallel recording): per thread   // ootu32 (gfx_vk_tex.c): a storage view's declared format is part of the entry
// what a view's texels are, as gfx_vk_shader.c's integer-texel variants need it: 0 float, 1 uint (unsigned integer formats, a stencil), 2 int
static int view_num(const tex_t *t, const view_t *v) {
    const vkfmt_t *f = vk_fmt(v && v->set_format ? v->format : t->d.format); if (!f) return 0;
    if (f->depth || f->stencil) return f->stencil && (!f->depth || t->stencil || (v && v->set_dsmode && v->dsmode == 1));   // vk_tex_view's aspect
    return f->is_int ? (f->is_sint ? 2 : 1) : 0;
}
static uint64_t g_num_um, g_num_im;   // stage_set: the sampled bindings of the stage being built that got uint / int views (bit = binding index)
extern uint64_t g_tex_gen;
static inline void s64_fb_note(const vkimg_t *im) {   // s64a: a descriptor of a draw in the open pass names one of the pass's attachments
    if (!im || S.pass_open != 1) return;
    for (int i = 0; i < 9; i++) if (P_att[i].im == im) { g_s64_fb = 1; return; }
}
// bak94 (BananaNative PR #276 parity): an Eden storage image the shader only reads (its descriptor's is_written 0, gfx_vk_eden.inc eden_image) does not mark its
// texture GPU-written (gpu_w); MKW's sparse terrain arrays were taken as GPU-written. NVN2 only; DM_VK_STORAGE_RO=0: every storage binding marks it, as before
static _Thread_local int g_storage_ro; static uint64_t g_storage_ro_n;   // R1 (parallel recording): g_storage_ro per thread (set and cleared around one lookup, eden_image)
static int tex_desc_for(int st, const dmsv_bind *b, uint64_t h, VkDescriptorImageInfo *ii, int storage, VkImageView *scaled) {   // the view's view_num
    int tid = (int)(h & 0xfffff); if (g_x_texfs && st == 1 && S.sh[1] && S.sh[1]->key == g_x_texfs && (g_x_texfs_mask >> (b->binding & 63) & 1)) tid = 0; const view_t *v = NULL; const tex_t *t = NULL;
    ii->imageLayout = VK_IMAGE_LAYOUT_GENERAL; ii->sampler = storage ? VK_NULL_HANDLE : samp_for(st, b, h); ii->imageView = VK_NULL_HANDLE;
    unsigned ck = ((unsigned)tid * 2654435761u ^ (unsigned)b->dim * 97u ^ (unsigned)storage * 31u ^ (unsigned)(b->flags & 1)) % TVC;
    if (tid && g_var_now != 3 && !g_tracing && g_tvc[ck].epoch == g_set_epoch && g_tvc[ck].gen == g_tex_gen && g_tvc[ck].tid == tid && g_tvc[ck].storage == storage && g_tvc[ck].dim == b->dim && g_tvc[ck].arr == (b->flags & 1) && g_tvc[ck].want == (storage ? g_vk_storage_want : 0) &&
        (!storage || g_storage_ro || (g_tvc[ck].im && g_tvc[ck].im->gpu_w))) {
        if (g_rdiag > 0) g_rd_n[36]++;   // replaydiag: the view cache hit
        // A sampled lookup can populate this view cache before a writable
        // storage lookup in the same submission. It still invalidates backing.
        if (storage && !g_storage_ro) { vk_tex_content_changed(g_tvc[ck].im);   // gwrange2: a storage write through a cached view also refreshes its range's GPU-write time (OoT a335c1db writes the E5 image through an R32_UINT view every frame)
            { void vk_gwr_add(const void *, uint64_t); const vkimg_t *wi = g_tvc[ck].im; if (wi && wi->gaddr && wi->size > 0) vk_gwr_add((const void *)(uintptr_t)wi->gaddr, (uint64_t)wi->size); } }
        if (scaled && g_tvc[ck].im && g_tvc[ck].im->scaled_valid) {
            if (!g_tvc[ck].scaled_view && vk_tex_by_id(tid, &v, &t) && t)
                g_tvc[ck].scaled_view = vk_rescaled_view(g_tvc[ck].im, t, v, b->dim, b->flags & 1, 0);
            *scaled = g_tvc[ck].scaled_view;
        }
        wb_descriptor(g_tvc[ck].im, scaled && *scaled, storage);
        ii->imageView = g_tvc[ck].v; if (g_tvc[ck].im) g_tvc[ck].im->last_use = mtl_last_fence() + 1; vk_texdump_note(g_tvc[ck].im); s64_fb_note(g_tvc[ck].im); return g_tvc[ck].num;
    }
    if (g_rdiag > 0) { g_rd_n[37]++;   // replaydiag: a miss; slot 38: misses only because the cache is per submission (same texture generation, same entry otherwise)
        if (tid && g_tvc[ck].epoch != g_set_epoch && g_tvc[ck].gen == g_tex_gen && g_tvc[ck].tid == tid && g_tvc[ck].storage == storage && g_tvc[ck].dim == b->dim && g_tvc[ck].arr == (b->flags & 1)) g_rd_n[38]++; }
    int num = 0;
    if (tid && g_var_now != 3 && vk_tex_by_id(tid, &v, &t) && t) {
        if (t->d.target == NVN_TEXTURE_TARGET_BUFFER && !storage && b->dim == 1 && !(b->flags & 1)) {   // r74: a texture buffer in a 2D slot, the translator's wrapped 1D fetch (vk_tbuf_rows_view); dmsv dim 1 = 2D (vk_tex_view: 0 1D, 1 2D, 2 3D, 3 cube)
            ii->imageView = vk_tbuf_rows_view(t); if (ii->imageView) num = view_num(t, NULL);
            if (g_tracing) VKLOG("  texture handle %#llx id %d: texture buffer of %u elements, nvn format %#x, pool %p+%#llx, read as 2D rows → view %p", (unsigned long long)h, tid, t->d.width, t->d.format,
                                 (const void *)t->pool, (unsigned long long)t->off, (void *)ii->imageView);
        } else {
        // r58 DM_VK_DITHER64=1 (test): Bayonetta 3's screen-door fade (B3-express diagnostics/dither-20261104: 429 pixel programs discard where the object's alpha
        // is below a texel of a 64x64 pattern read at FragCoord & 63). Every 64x64 texture of a fragment stage reads as zeros (a null descriptor), so no fade discards
        if (g_dither64 && st == 1 && !storage && V.has_robust2 && t->d.width == 64 && t->d.height == 64) {
            enum { DS = 32 }; static int seen[DS], ns; int f = 0; for (int i = 0; i < ns; i++) if (seen[i] == tid) { f = 1; break; }
            if (!f && ns < DS) { seen[ns++] = tid; VKLOG("dither64 %d: texture %d (64x64, nvn format %#x, target %d, levels %u) read as zeros by shader %016llx binding %d", ns, tid, t->d.format, t->d.target, t->d.levels,
                                                        S.sh[1] ? (unsigned long long)S.sh[1]->key : 0ull, b->binding); }
            ii->imageView = VK_NULL_HANDLE; return 0; }
        vkimg_t *im = NULL; ii->imageView = vk_tex_view(t, v, b->dim, b->flags & 1, storage ? VK_VIEW_STORAGE : VK_VIEW_SAMPLED, &im); if (storage && ii->imageView && im && g_storage_ro) { uint64_t c = ++g_storage_ro_n; if (c <= 16 || !(c & (c - 1))) VKLOG("bak94: storage image read only by its shader, not marked GPU-written: %ux%ux%u nvn format %#x (%llu so far)", im->w, im->h, im->layers, im->nvn_format, (unsigned long long)c); }
        else if (storage && ii->imageView && im) { vk_tex_content_changed(im); im->gpu_w = 1; vk_tex_gpu_mark(t); }   // r72 mark: mtl_tex_gpu_owned
        if (g_tracing) VKLOG("  texture handle %#llx id %d: %ux%u nvn format %#x target %d serial %llu pool %p+%#llx size %#llx → image %p (%ux%u fmt %d layers %u levels %u) view %p", (unsigned long long)h, tid, t->d.width, t->d.height, t->d.format, t->d.target,
                             (unsigned long long)t->serial, (const void *)t->pool, (unsigned long long)t->off, (unsigned long long)t->L.size, (void *)im, im ? im->w : 0, im ? im->h : 0, im ? im->fmt : 0, im ? im->layers : 0, im ? im->levels : 0, (void *)ii->imageView);
        if (im) im->last_use = mtl_last_fence() + 1;
        s64_fb_note(im);   // s64a
        if (ii->imageView && scaled && im && im->scaled_valid)
            *scaled = vk_rescaled_view(im, t, v, b->dim, b->flags & 1, 0);
        if (ii->imageView) wb_descriptor(im, scaled && *scaled, storage);
        if (ii->imageView) num = view_num(t, v);
        if (ii->imageView) { g_tvc[ck].epoch = g_set_epoch; g_tvc[ck].gen = g_tex_gen; g_tvc[ck].tid = tid; g_tvc[ck].storage = storage; g_tvc[ck].dim = b->dim; g_tvc[ck].arr = b->flags & 1; g_tvc[ck].v = ii->imageView; g_tvc[ck].scaled_view = scaled ? *scaled : VK_NULL_HANDLE; g_tvc[ck].im = im; g_tvc[ck].num = num; g_tvc[ck].want = storage ? g_vk_storage_want : 0; }
        }   // r74
    }
    if (!ii->imageView) {
        if (g_tracing) VKLOG("  texture handle %#llx id %d: no view: a dummy %s view is bound (texture %p)", (unsigned long long)h, tid, storage ? "storage" : "sampled", (const void *)t);   // dtrace
        const vkfmt_t *f = t ? vk_fmt(t->d.format) : NULL;
        ii->imageView = vk_dummy_view(b->dim == 0xff ? 1 : b->dim, b->flags & 1, (b->flags & 2) != 0, storage, f && f->is_int); num = f && f->is_int ? (f->is_sint ? 2 : 1) : 0;
    }
    return num;
}
// Reuse within a submission (the cache is emptied when it begins, vk_exec_reset): descriptor sets by their whole content (layout, and every buffer range,
// view, sampler and texel buffer view written), and driver segments by their bytes. Unity draws many times with the same bindings, and every new set
// is an allocation and a vkUpdateDescriptorSets; the driver segment's copy is shared so that a set using it can match.
// R1 (parallel recording): SETC, DRVC, g_setc and g_drvc in struct rec_ctx (top of the file)
uint64_t g_set_hits, g_set_new, g_ns_set_build, g_ns_set_hash, g_ns_set_new;   // descriptor sets: reused, new; time collecting their descriptors, hashing them, allocating and writing new ones
#define g_set_hits (*g_rec->set_hits)   // S1c (parallel recording): the thread's (struct rec_ctx; the replay thread's are these globals)
#define g_set_new (*g_rec->set_new)
// content hash of descriptor sets and driver segments: 8 bytes a step (a set with 32-slot texture arrays is several KB, which byte-wise FNV took milliseconds a frame over)
static inline uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p; size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, b + i, 8); h = (h ^ w) * 0x9E3779B97F4A7C15ull; h ^= h >> 29; }
    for (; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; }
    return h;
}
// r75: an indirect dispatch's group counts are written by the GPU, and the shader reads them from c0 + 0x200 (and c0 + 4 on NVN2): the driver segment held the CPU's
// values (0 for an indirect dispatch), so GPU-driven compute (culling, light tiles) ran with zero groups in its arithmetic. As nvn_mtl.m, the counts are copied there on
// the GPU before the dispatch, into a segment of its own (g_drv_fresh: not one shared by content with other dispatches). DM_VK_IND_C0=0 turns it off
static int g_drv_fresh, g_ind_c0 = -1; static VkBuffer g_drv_lb; static VkDeviceSize g_drv_lo; static uint64_t g_ind_c0_n;
static size_t drv_segment_size(int st, int bank) {
    if (bank == DMSH_PUSH_BANK) return DMSH_PUSH_SIZE;
    if (bank != 0) return 0x968;
    return storage_table_offset(st) > 0x648 ? storage_table_offset(st) + 0x100 : 0x648;
}
static void drv_shared(int st, int bank, VkBuffer *buf, VkDeviceSize *off) {   // the driver segment: built, then one copy per distinct content in the submission
    uint8_t d[0x1000]; drv_segment(st, bank, d);
    if (g_drv_fresh && st == 5 && bank == 0) { uint8_t *p = vk_scratch(0x1000, V.props.limits.minUniformBufferOffsetAlignment, buf, off); memcpy(p, d, sizeof d); g_drv_lb = *buf; g_drv_lo = *off; return; }   // r75
    // Hash every byte the stage can address, including the NVN2 task and mesh storage tables.
    uint64_t h = fnv(0xcbf29ce484222325ull, d, drv_segment_size(st, bank)); unsigned k = (unsigned)(h % DRVC);
    if (g_drvc[k].epoch == g_set_epoch && g_drvc[k].h == h && !g_x_nodrvc) { *buf = g_drvc[k].b; *off = g_drvc[k].o; return; }
    uint8_t *p = vk_scratch(0x1000, V.props.limits.minUniformBufferOffsetAlignment, buf, off); memcpy(p, d, sizeof d);
    g_drvc[k].h = h; g_drvc[k].epoch = g_set_epoch; g_drvc[k].b = *buf; g_drvc[k].o = *off;
}
// a constant buffer binding's descriptor (stage_set, set_memo_get): its pool's buffer from the alignment boundary below it, or the shadow of the deferred updates over it
// DM_VK_UBD_WATCH: a constant buffer binding that covers the watched address, with what the draw reads there and the host buffer it gets
static void cbuf_watch(int st, const vkshader_t *s, int reg, uint64_t a, uint64_t n, const VkDescriptorBufferInfo *bi, int shadow) {
    extern uint64_t g_ubd_watch; extern int g_ubd_wn;   /* S1d (parallel recording): g_ubd_gen is gfx_vk.h's macro */ void vk_watch_poll(const char *where);
    if (!g_ubd_watch || g_ubd_wn <= 0 || g_ubd_watch < a || g_ubd_watch + 4 > a + n) return;
    vk_watch_poll("draw"); uint8_t w[4]; float v; vk_ubd_read(g_ubd_watch, 4, w); memcpy(&v, w, 4); g_ubd_wn--;
    VKLOG("ubd watch: draw %llu present %llu pass %d stage %d shader %016llx c%d guest %#llx +%#llx (size %#llx) reads %g; host buffer %#llx offset %#llx range %#llx%s (dynamic offset added at bind), update gen %llu",
          (unsigned long long)g_vkst.draws, (unsigned long long)g_tex_frame, g_sub_pass - 1, st, (unsigned long long)s->key, reg, (unsigned long long)a, (unsigned long long)(g_ubd_watch - a),
          (unsigned long long)n, v, (unsigned long long)(uintptr_t)bi->buffer, (unsigned long long)bi->offset, (unsigned long long)bi->range, shadow ? " shadow" : "", (unsigned long long)g_ubd_gen);
}
static void cbuf_desc(int st, const vkshader_t *s, const dmsv_bind *b, VkDescriptorBufferInfo *bi) {
    const p_bind_buf *u = b->reg >= 0 && b->reg < 16 ? &S.ubo[st][b->reg] : NULL;
    uint64_t n = u && u->size ? (u->size > 65536 ? 65536 : u->size) : 65536; VkBuffer buf; VkDeviceSize off;
    vkpool_t *p = g_var_now != 9 && u && u->addr ? (vk_pool_of(u->addr, n) ?: vk_pool_of(u->addr, 1)) : NULL;   // r78: as far as it reads
    if (!p) { *bi = (VkDescriptorBufferInfo){vk_zero_buffer(), 0, 65536}; return; }
    buf = p->buf; off = u->addr - p->lo; if (off + n > p->hi - p->lo) n = p->hi - p->lo - off;
    ubo_log(s, st, b->reg, u->addr, n);
    if (vk_ubd_shadow(u->addr, n, &buf, &off)) { *bi = (VkDescriptorBufferInfo){buf, off, n}; cbuf_watch(st, s, b->reg, u->addr, n, bi, 1); return; }   // deferred updates in it: their shadow
    VkDeviceSize al = V.props.limits.minUniformBufferOffsetAlignment; if (off % al) VKRED_ONCE(40, "uniform buffer at an offset not aligned to %llu", (unsigned long long)al);
    *bi = (VkDescriptorBufferInfo){buf, off - off % al, n + off % al};
    cbuf_watch(st, s, b->reg, u->addr, n, bi, 0);
}
// The set a stage's last draw got, and what it was built from. Between two draws most stages change no texture, sampler, image or storage buffer, only
// the offsets of their constant buffers, which go to the bind (dynamic uniform buffers). Then nothing is collected or hashed: the constant buffers alone are
// looked at again, and their buffer and range must be the same (the offset too for one that is not dynamic). It holds within one submission (the sets and
// the driver segments' copies are its own) while the stage's shader and bindings (g_bind_gen, g_ubo_gen; only real changes count), the texture IDs (g_tex_gen),
// sampler IDs (g_samp_gen), memory pools (vk_pool_gen) and the base instance (the driver segment's c0 + 0x644) are unchanged. Images keep the last_use the
// first build gave them: the fence it names is the same for the whole submission. Not for compute (its driver segment holds the dispatch size and its storage
// buffers are marked written), nor for a shader that reads handles or addresses from uniform buffer memory (that memory changes under the same bindings).
// DM_VK_X_NOSETMEMO=1 (live): always collected. DM_VK_X_SETVERIFY=1 (live): one hit in 64 is collected anyway and compared, [RED] on a difference
enum { MEMO_CB = 16 };
typedef struct { vkshader_t *s; uint64_t epoch, bind, ubo, ubd, tex, samp, h, um, im; uint32_t pool, binst, dofs[96]; int ncb; uint8_t ci[MEMO_CB], cpos[MEMO_CB]; VkDescriptorBufferInfo cd[MEMO_CB]; VkDescriptorSet set; int pofs; pofs_slot_t ps[8]; } set_memo_t;   // r152b ps: the r133 slots its build found
static set_memo_t g_smemo[5];   // r69: was [3] (tessellation stages 3, 4)
// g_bind_gen[6], g_ubo_gen[6]: a stage's handles or storage buffers changed (bind_handles_rec, OP_BIND_SSBO), its uniform buffers (OP_BIND_UBO) (S1e (parallel recording): struct rec_ctx)
uint64_t g_set_memo, g_set_memo_bad;   // memo hits, and differences the verify mode found
uint32_t vk_pool_gen(void);   // (S1d (parallel recording): g_ubd_gen, the thread's deferred update list's generation, is a macro in gfx_vk.h)
static int set_memo_ok(vkshader_t *s) {
    if (!s->setmemo) { int ok = 1, ncb = 0;
        for (int i = 0; i < s->nbind; i++) { const dmsv_bind *b = &s->b[i]; ncb += b->kind == RK_CBUF;
            ok &= b->kind != RK_SBUF_UBO && b->kind != RK_UBO_PTR && b->kind != RK_TEX_UBO && b->kind != RK_IMG_UBO && b->kind != RK_TBUF_UBO && b->samp != RS_UBO && i < 256; }
        s->setmemo = ok && ncb <= MEMO_CB ? 1 : 2; }
    return s->setmemo == 1;
}
static int set_memo_get(int st, vkshader_t *s, uint32_t *dofs) {
    set_memo_t *m = &g_smemo[st];
    if (m->s != s || (st < 2 && m->pofs != POFS_TAG(st)) || m->epoch != g_set_epoch || m->bind != g_bind_gen[st] || m->tex != g_tex_gen || m->samp != __atomic_load_n(&g_samp_gen, __ATOMIC_ACQUIRE) || m->pool != vk_pool_gen() || m->binst != g_draw_binst) return 0;
    if (m->ubo != g_ubo_gen[st] || m->ubd != g_ubd_gen) {   // the constant buffers again: most draws move only their offsets
        for (int c = 0; c < m->ncb; c++) { VkDescriptorBufferInfo d; cbuf_desc(st, s, &s->b[m->ci[c]], &d);
            if (d.buffer != m->cd[c].buffer || d.range != m->cd[c].range || (m->cpos[c] == 0xff && d.offset != m->cd[c].offset)) return 0;
            if (m->cpos[c] != 0xff) m->dofs[m->cpos[c]] = (uint32_t)d.offset; }
        m->ubo = g_ubo_gen[st]; m->ubd = g_ubd_gen; }
    memcpy(dofs, m->dofs, sizeof *dofs * (size_t)s->ndyn); g_num_um = m->um; g_num_im = m->im;
    if (st < 2 && g_pofs_st[st]) memcpy(g_pofs_slot[st], m->ps, sizeof m->ps);   // r152b: not the slots of whatever shader was built last (one without a memo)
    return 1;
}
// dofs (at least s->ndyn entries): the dynamic offsets of the set's dynamic uniform buffers, in binding order as vkCmdBindDescriptorSets takes them
static VkDescriptorSet stage_set(int st, vkshader_t *s, uint32_t *dofs) {
    int memo = st < 5 && !g_x_nosetmemo && !g_tracing && !g_var_now && !g_x_texfs && !g_x_nosetc && !g_x_nodrvc && !g_ubolog_key && !g_ubd_watch && g_td_off < 0 && set_memo_ok(s), verify = 0;
    uint32_t vd[96]; uint64_t vum = 0, vim = 0;   // verify: what the memo gave
    if (memo && set_memo_get(st, s, dofs)) { static unsigned nv; if (!g_x_setverify || ++nv % 64) { g_set_memo++; return g_smemo[st].set; }
        verify = 1; memcpy(vd, dofs, sizeof *dofs * (size_t)s->ndyn); vum = g_num_um; vim = g_num_im; }
    uint64_t gtex = g_tex_gen, gsamp = __atomic_load_n(&g_samp_gen, __ATOMIC_ACQUIRE); uint32_t gpool = vk_pool_gen();   // before collecting: a change racing it (game thread) leaves the memo outdated
    VkDescriptorSet set = VK_NULL_HANDLE; struct { uint32_t b, o; int i; } dp[96]; int nd = 0; uint64_t tb0 = tick();
    int pw_[96]; uint8_t pc_[96]; memset(pc_, 0, sizeof pc_);   // r133: a converted constant buffer's write
    VkWriteDescriptorSet w[96]; VkDescriptorBufferInfo bi[96]; VkDescriptorImageInfo ii[96 * 4]; VkBufferView bv[96]; uint32_t nw = 0, nii = 0;
    int ncb = 0; uint8_t mci[MEMO_CB]; VkDescriptorBufferInfo mcd[MEMO_CB];   // the constant buffers, for the memo
    g_num_um = g_num_im = 0;
    for (int i = 0; i < s->nbind && nw < 96; i++) {
        const dmsv_bind *b = &s->b[i];
        VkWriteDescriptorSet *wr = &w[nw]; memset(wr, 0, sizeof *wr);
        wr->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr->dstSet = set; wr->dstBinding = b->binding; wr->descriptorCount = 1;
        VkBuffer buf; VkDeviceSize off, len;
        switch (b->kind) {
        case RK_SUPPORT: bi[nw] = (VkDescriptorBufferInfo){vk_support_buffer(), 0, 65536}; break;
        // raw global memory (dmsh_abi.h DMSH_GMEM_*): the pool table (gfx_vk_core.c vk_gmem_map, the pools' device addresses); the shader's write log gets a cleared buffer
        // of its own, never the shared zero buffer, which the shader writes (read after the submission: gfx_vk_core.c gml_copy_back)
        case RK_GMEM_MAP: { VkDeviceSize n; if (vk_gmem_map(&buf, &n)) bi[nw] = (VkDescriptorBufferInfo){buf, 0, n};
            else { VKRED_ONCE(93, "a shader reads global memory through the pool table, and the device has no buffer device address: those reads give 0 and stores are dropped"); bi[nw] = (VkDescriptorBufferInfo){vk_zero_buffer(), 0, 65536}; }
            break; }
        case RK_GMEM_LOG: { const VkDeviceSize n = ((VkDeviceSize)1 << DMSH_GMEM_LOG_LOG2) * 4 + 4; uint8_t *z = vk_scratch(n, V.props.limits.minStorageBufferOffsetAlignment, &buf, &off); memset(z, 0, n); vk_gml_note((const uint32_t *)z); bi[nw] = (VkDescriptorBufferInfo){buf, off, n}; break; }
        case RK_C1: vk_shader_c1(s, &buf, &off, &len); bi[nw] = (VkDescriptorBufferInfo){buf, off, len}; break;
        case RK_CBUF: cbuf_desc(st, s, b, &bi[nw]); if (ncb < MEMO_CB) { mci[ncb] = (uint8_t)i; mcd[ncb++] = bi[nw]; } break;
        case RK_DRV: { VkBuffer sb; VkDeviceSize so; drv_shared(st, b->reg, &sb, &so); bi[nw] = (VkDescriptorBufferInfo){sb, so, 0x1000}; break; }
        case RK_SBUF: case RK_UBO_PTR: case RK_SBUF_UBO: {
            uint64_t a = 0, n = 0;
            if (b->kind == RK_SBUF && b->reg >= 0 && b->reg < 16) { a = S.ssbo[st][b->reg].addr; n = S.ssbo[st][b->reg].size; }
            else if (b->kind == RK_UBO_PTR && b->reg >= 0 && b->reg < 16) { a = S.ubo[st][b->reg].addr; n = S.ubo[st][b->reg].size; }
            else if (b->kind == RK_SBUF_UBO) { a = (uint64_t)ubo_word(st, b->reg, b->off) | (uint64_t)ubo_word(st, b->reg, b->off + 4) << 32; }
            vkpool_t *p = a ? ((n ? vk_pool_of(a, n) : NULL) ?: vk_pool_of(a, 1)) : NULL;   // r78: as far as it reads
            if (p) { buf = p->buf; off = a - p->lo; if (!n || off + n > p->hi - p->lo) n = p->hi - p->lo - off;
                { const VkDeviceSize mx = V.props.limits.maxStorageBufferRange; if (mx > SB_ALIGN && n + SB_ALIGN > mx) n = mx - SB_ALIGN; }   // s12: the rest of a chunk one mapped view long can pass the device's range
                // bound from the 256-byte boundary below it, the boundary the translator offsets every access from (shader_vk.py SB_ALIGN; a multiple of any device's minimum alignment)
                const VkDeviceSize al = SB_ALIGN; if (V.props.limits.minStorageBufferOffsetAlignment > al) VKRED_ONCE(41, "storage buffer alignment %llu above %llu", (unsigned long long)V.props.limits.minStorageBufferOffsetAlignment, (unsigned long long)al);
                bi[nw] = (VkDescriptorBufferInfo){buf, off - off % al, n + off % al};
                if (st == 5 && b->kind != RK_UBO_PTR) { void vk_gpu_wrote(uint64_t a, uint64_t n); void vk_hy_bind(int, uint64_t); void vk_hy_bind_end(void);   // a dispatch's storage buffers: taken as written
                    vk_hy_bind(b->flags & 4 ? 2 : 1, n); vk_gpu_wrote(a, n); vk_hy_bind_end(); }   // (hybrid1 diagnostic: by kind; the bundle has no atomic fact)
                else if (b->kind == RK_SBUF && (b->flags & 4)) { void vk_sub_wrote(uint64_t a, uint64_t n); vk_sub_wrote(a, n); }   // a draw's, when its shader writes it: copied back (copied pools)
                else if (b->kind == RK_SBUF_UBO && (b->flags & 4)) { extern uint64_t g_sbufu_w; g_sbufu_w++;   // one named in uniform memory: counted (gfx_vk_core.c's copied pools line)
                    void vk_sub_wrote(uint64_t a, uint64_t n); vk_sub_wrote(a, n); }   // poolcopy (review P1-2): its size is unknown, so all it may reach, to its chunk's end, is taken as written
            } else bi[nw] = (VkDescriptorBufferInfo){(st == 5 && b->kind != RK_UBO_PTR) || (b->flags & 4) ? vk_wsink_buffer() : vk_zero_buffer(), 0, 65536};   // rptfix: written stand-ins to the sink
            break; }
        case RK_TEX: case RK_TEX_UBO: case RK_TEX_C2A: {
            int cnt = b->count ? b->count : 32; if (nii + (uint32_t)cnt > sizeof ii / sizeof *ii) cnt = 1;
            for (int k = 0; k < cnt; k++) {
                uint64_t h = b->kind == RK_TEX ? (b->reg >= 0 && b->reg < 256 ? S.tex[st][b->reg] : 0) :
                             b->kind == RK_TEX_UBO ? ubo_handle(st, b->reg, b->off + 8 * k) : (b->off >= 0 ? c2_handle(st, b->off + 8 * k) : 0);
                int num = tex_desc_for(st, b, h, &ii[nii + k], 0, NULL);
                if (num && i < 64) { if (num == 1) g_num_um |= 1ull << i; else g_num_im |= 1ull << i; }
            }
            wr->pImageInfo = &ii[nii]; nii += (uint32_t)cnt; wr->descriptorCount = (uint32_t)cnt; wr->descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; nw++;
            continue; }
        case RK_IMG: case RK_IMG_UBO: {
            uint64_t h = b->kind == RK_IMG ? (b->reg >= 0 && b->reg < 32 ? S.img[st][b->reg] : 0) : ubo_handle(st, b->reg, b->off);
            tex_desc_for(st, b, h, &ii[nii], 1, NULL); wr->pImageInfo = &ii[nii++]; wr->descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; nw++;
            continue; }
        case RK_TBUF: case RK_TBUF_UBO: case RK_IMG_TBUF: {
            uint64_t h = b->kind == RK_TBUF ? (b->reg >= 0 && b->reg < 256 ? S.tex[st][b->reg] : 0) : b->kind == RK_IMG_TBUF ? (b->reg >= 0 && b->reg < 32 ? S.img[st][b->reg] : 0) : ubo_handle(st, b->reg, b->off);
            const view_t *v; const tex_t *t = NULL; int tid = (int)(h & 0xfffff);
            bv[nw] = tid && vk_tex_by_id(tid, &v, &t) && t && t->d.target == NVN_TEXTURE_TARGET_BUFFER ? vk_tbuf_view(t) : VK_NULL_HANDLE;
            if (!bv[nw] && t && b->kind != RK_IMG_TBUF && t->d.target == NVN_TEXTURE_TARGET_BUFFER) bv[nw] = vk_tbuf_copy_view(t);   // r75: below the texel-buffer alignment
            if (!bv[nw]) bv[nw] = vk_dummy_tbuf(b->kind == RK_IMG_TBUF);
            else if (b->kind != RK_IMG_TBUF && i < 64) { int num = view_num(t, NULL); if (num == 1) g_num_um |= 1ull << i; else if (num == 2) g_num_im |= 1ull << i; }   // texel buffers read their words alike
            else if (b->kind == RK_IMG_TBUF) { void vk_sub_wrote(uint64_t a, uint64_t n); vk_sub_wrote((uint64_t)(uintptr_t)t->mem, t->L.size); }   // a storage texel buffer: copied back (copied pools)
            wr->pTexelBufferView = &bv[nw]; wr->descriptorType = b->kind == RK_IMG_TBUF ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER; nw++;
            continue; }
        default: {   // unknown: the descriptor type it was declared with, filled with what reads zero
            if (b->desc == RD_UBO || b->desc == RD_SSBO) bi[nw] = (VkDescriptorBufferInfo){vk_zero_buffer(), 0, 65536};
            else if (b->desc == RD_SAMPLED || b->desc == RD_STORAGE) {
                ii[nii] = (VkDescriptorImageInfo){b->desc == RD_SAMPLED ? vk_dummy_sampler() : VK_NULL_HANDLE, vk_dummy_view(b->dim == 0xff ? 1 : b->dim, b->flags & 1, (b->flags & 2) != 0, b->desc == RD_STORAGE, 0), VK_IMAGE_LAYOUT_GENERAL};
                int cnt = b->count ? b->count : 32; for (int k = 1; k < cnt && nii + k < sizeof ii / sizeof *ii; k++) ii[nii + k] = ii[nii];
                wr->pImageInfo = &ii[nii]; nii += (uint32_t)cnt; wr->descriptorCount = (uint32_t)cnt;
                wr->descriptorType = b->desc == RD_SAMPLED ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; nw++; continue;
            } else if (b->desc == RD_TBUF || b->desc == RD_STORAGE_TBUF) { bv[nw] = vk_dummy_tbuf(b->desc == RD_STORAGE_TBUF); wr->pTexelBufferView = &bv[nw];
                wr->descriptorType = b->desc == RD_TBUF ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER; nw++; continue; }
            else continue;
        }
        }
        wr->pBufferInfo = &bi[nw]; wr->descriptorType = b->desc == RD_SSBO ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        if (s->dyn[i]) {   // the offset goes to the bind, so the set's content (and its reuse) does not depend on it
            if (bi[nw].offset > UINT32_MAX) VKRED_ONCE(42, "dynamic uniform buffer offset past 4 GB");
            int j = nd++; while (j > 0 && dp[j - 1].b > b->binding) { dp[j] = dp[j - 1]; j--; }
            dp[j].b = b->binding; dp[j].o = (uint32_t)bi[nw].offset; dp[j].i = i; bi[nw].offset = 0; wr->descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            if (st < 2 && g_pofs_st[st] && b->kind == RK_CBUF && i < 96) { pc_[i] = 1; pw_[i] = (int)nw; }   // r133
        }
        nw++;
    }
    for (int j = 0; j < nd; j++) dofs[j] = dp[j].o;
    if (st < 2 && g_pofs_st[st]) for (int j = 0; j < 8; j++) {   // r133: the converted buffers get a 64 KB range (as far as their buffer goes)
        pofs_slot_t *q = &g_pofs_slot[st][j]; memset(q, 0, sizeof *q);
        if (j >= nd || dp[j].i >= 96 || !pc_[dp[j].i] || (g_pofs_m[st] && !(g_pofs_m[st] >> j & 1))) continue;   // r134: slot mask
        VkDescriptorBufferInfo *x = &bi[pw_[dp[j].i]]; VkDeviceSize sz = vk_buf_size(x->buffer), R = sz < 65536 ? sz : 65536;
        if (!sz || x->range > R || dp[j].o + x->range > sz) continue;
        q->buf = x->buffer; q->N = x->range; q->R = R; q->Sz = sz; q->conv = 1; x->range = R; }
    if (g_tracing) for (int j = 0; j < nd; j++) VKLOG("  stage %d binding %u: dynamic offset %u", st, dp[j].b, dp[j].o);
    if (g_tracing) for (uint32_t i = 0; i < nw; i++) {   // the refused draw (bisection): every descriptor it is given
        const VkWriteDescriptorSet *x = &w[i];
        if (x->pBufferInfo) VKLOG("  stage %d binding %u type %d: buffer %p +%llu size %llu", st, x->dstBinding, x->descriptorType, (void *)x->pBufferInfo->buffer, (unsigned long long)x->pBufferInfo->offset, (unsigned long long)x->pBufferInfo->range);
        else if (x->pImageInfo) for (uint32_t k = 0; k < x->descriptorCount && k < 4; k++) VKLOG("  stage %d binding %u[%u] type %d: view %p sampler %p", st, x->dstBinding, k, x->descriptorType, (void *)x->pImageInfo[k].imageView, (void *)x->pImageInfo[k].sampler);
        else if (x->pTexelBufferView) VKLOG("  stage %d binding %u type %d: texel buffer view %p", st, x->dstBinding, x->descriptorType, (void *)x->pTexelBufferView[0]);
    }
    // the set's content, then an earlier identical set of this submission or a new one
    uint64_t tb1 = tick(); g_ns_set_build += tb1 - tb0;
    uint64_t h = fnv(0xcbf29ce484222325ull, &s->dsl, sizeof s->dsl);
    for (uint32_t i = 0; i < nw; i++) {
        const VkWriteDescriptorSet *x = &w[i]; h = fnv(h, &x->dstBinding, 4); h = fnv(h, &x->descriptorType, 4); h = fnv(h, &x->descriptorCount, 4);
        if (x->pBufferInfo) h = fnv(h, x->pBufferInfo, sizeof *x->pBufferInfo);
        else if (x->pImageInfo) h = fnv(h, x->pImageInfo, sizeof *x->pImageInfo * x->descriptorCount);
        else if (x->pTexelBufferView) h = fnv(h, x->pTexelBufferView, sizeof *x->pTexelBufferView);
    }
    unsigned k = (unsigned)(h % SETC); uint64_t tb2 = tick(); g_ns_set_hash += tb2 - tb1;
    if (g_setc[k].epoch == g_set_epoch && g_setc[k].h == h && g_setc[k].l == s->dsl && !g_tracing && !g_x_nosetc) { g_set_hits++; set = g_setc[k].set; }
    else { set = vk_alloc_set(s->dsl); g_set_new++;
        for (uint32_t i = 0; i < nw; i++) w[i].dstSet = set;
        vkUpdateDescriptorSets(V.dev, nw, w, 0, NULL);
        g_setc[k].h = h; g_setc[k].epoch = g_set_epoch; g_setc[k].l = s->dsl; g_setc[k].set = set; g_ns_set_new += tick() - tb2; }
    // verify: the same content (hash), integer-texel bindings and dynamic offsets as the memo gave. A lone difference in dofs alone can be a driver segment
    // copied again after its g_drvc entry was taken by another (the memo's copy is as good); anything repeated is a dependency the memo misses
    if (verify && (h != g_smemo[st].h || g_num_um != vum || g_num_im != vim || memcmp(vd, dofs, sizeof *dofs * (size_t)s->ndyn))) { g_set_memo_bad++; static int n;
        if (n++ < 20) VKLOG("[RED] descriptor set memo: stage %d shader %016llx: a fresh set differs from the memo's (hash %016llx / %016llx, um %#llx / %#llx, im %#llx / %#llx, dynamic offsets %s)", st, (unsigned long long)s->key,
                            (unsigned long long)h, (unsigned long long)g_smemo[st].h, (unsigned long long)g_num_um, (unsigned long long)vum, (unsigned long long)g_num_im, (unsigned long long)vim, memcmp(vd, dofs, sizeof *dofs * (size_t)s->ndyn) ? "differ" : "same"); }
    if (memo) { set_memo_t *m = &g_smemo[st];   // (set_memo_ok: at most MEMO_CB constant buffers)
        m->s = s; m->epoch = g_set_epoch; m->bind = g_bind_gen[st]; m->ubo = g_ubo_gen[st]; m->ubd = g_ubd_gen; m->tex = gtex; m->samp = gsamp; m->pool = gpool; m->binst = g_draw_binst;
        m->h = h; m->um = g_num_um; m->im = g_num_im; m->set = set; m->pofs = st < 2 ? POFS_TAG(st) : 0; if (st < 2) memcpy(m->ps, g_pofs_slot[st], sizeof m->ps); memcpy(m->dofs, dofs, sizeof *dofs * (size_t)s->ndyn); m->ncb = ncb; memcpy(m->ci, mci, (size_t)ncb); memcpy(m->cd, mcd, sizeof *mcd * (size_t)ncb);
        for (int c = 0; c < ncb; c++) { m->cpos[c] = 0xff; for (int j = 0; j < nd; j++) if (dp[j].i == mci[c]) m->cpos[c] = (uint8_t)j; } }
    return set;
}

// ---------------- draws ----------------
// What the draws of the command buffer being recorded set last: a draw that would set the same again records nothing. Vulkan keeps dynamic state and
// bindings for the whole command buffer, across render passes and across pipelines that declare the same dynamic state (every one of ours, build_pipeline);
// a new command buffer (g_cmd_gen, gfx_vk_core.c) starts empty. Xenoblade Chronicles 3 docked: ~1,900 draws a frame, each with 10 dynamic-state calls and
// its binds, most the same as the draw before; the driver's time grows with what is recorded (vkEndCommandBuffer too). DM_VK_X_NOBC=1: everything every draw
// R2 (parallel recording): g_cmd_gen is a macro for the thread's recording target's generation (gfx_vk.h; was extern uint64_t here)
// R1 (parallel recording): g_bc in struct rec_ctx (top of the file)

static int bc_fresh(void) { if (g_bc.gen != g_cmd_gen || g_x_nobc) { memset(&g_bc, 0, sizeof g_bc); g_bc.gen = g_cmd_gen; g_bc.it = (VkIndexType)-1; return 1; } return 0; }
#define BC_SAME(field, val) (!fresh && !memcmp(&g_bc.field, &(val), sizeof g_bc.field))
static void bc_index(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkIndexType t) {
    int fresh = bc_fresh(); if (!fresh && g_bc.ib == b && g_bc.io == o && g_bc.it == t) return;
    vkCmdBindIndexBuffer(cb, b, o, t); g_bc.ib = b; g_bc.io = o; g_bc.it = t; g_r131.ibind++;
}
static void r135_force(VkCommandBuffer cb, int indexed) {   // r135 DM_VK_X_VB / DM_VK_X_IB: the binds made again as they are (g_bc holds them)
    if (g_x_vb) for (uint32_t b = 0; b < 17; b++) if (g_bc.vb[b]) { vkCmdBindVertexBuffers(cb, b, 1, &g_bc.vb[b], &g_bc.vo[b]); g_r135.vb++; break; }
    if (g_x_ib && indexed && g_bc.ib) { vkCmdBindIndexBuffer(cb, g_bc.ib, g_bc.io, g_bc.it); g_r135.ib++; }
}
static void set_dynamic(VkCommandBuffer cb) {
    int fresh = bc_fresh();
    float x = S.vp[0][0], y = S.vp[0][1], w = S.vp[0][2], h = S.vp[0][3];
    if (w == 0 && h == 0) { w = (float)S.pass_w; h = (float)S.pass_h; }
    r80_vp((int)S.pass_w * pass_scale, (int)S.pass_h * pass_scale, w, h);   // r80 (diagnostic)
    VkViewport vp = {x, y, w, h, S.dnear, S.dfar};
    if (g_lower_left && !g_flipy) { vp.y = (float)S.pass_h - y; vp.height = -h; }   // lower-left viewports count y from the bottom
    else if (!g_lower_left && g_flipy) { vp.y = y + h; vp.height = -h; }        // clip space y up over the same top-left rectangle
    if (!g_depth_zero && !V.has_depth_clip_ctl) { /* GL depth convention without clip control: left as is (rare) */ }
    // Viewport swizzle (Absolum's FNA3D sets NEGATIVE_Y every frame; Advance Wars 1+2's Unity sets it too): NEGATIVE_Y negates clip y, the same as the viewport flipped about
    // its centre; facing is decided after the viewport transform in Vulkan as on NVN, so the winding stays. Other swizzles are drawn unswizzled (nvn_mtl.m does the same)
    { uint32_t sw; memcpy(&sw, S.vp_sw[0], 4);
      if (sw == 0x100) { vp.y += vp.height; vp.height = -vp.height; }
      else if (sw) VKRED_ONCE(81, "viewport swizzle %d,%d,%d,%d not implemented (drawn unswizzled)", S.vp_sw[0][0], S.vp_sw[0][1] ^ 2, S.vp_sw[0][2] ^ 4, S.vp_sw[0][3] ^ 6); }
    if (vp.minDepth < 0) vp.minDepth = 0; if (vp.maxDepth > 1) vp.maxDepth = 1;
    vp.x *= pass_scale; vp.y *= pass_scale; vp.width *= pass_scale; vp.height *= pass_scale;
    if (!BC_SAME(vp, vp)) { vkCmdSetViewport(cb, 0, 1, &vp); g_bc.vp = vp; g_r131.vp++; }
    VkRect2D sc = physical_rect(scissor_rect()), ss = sc;   // r131: set below
    { int32_t vx0 = (int32_t)fmaxf(vp.x, 0), vy0 = (int32_t)fmaxf(fminf(vp.y, vp.y + vp.height), 0), vx1 = (int32_t)(vp.x + vp.width), vy1 = (int32_t)fmaxf(vp.y, vp.y + vp.height);   // what this draw can touch
      int32_t x0 = sc.offset.x > vx0 ? sc.offset.x : vx0, y0 = sc.offset.y > vy0 ? sc.offset.y : vy0, x1 = sc.offset.x + (int32_t)sc.extent.width < vx1 ? sc.offset.x + (int32_t)sc.extent.width : vx1, y1 = sc.offset.y + (int32_t)sc.extent.height < vy1 ? sc.offset.y + (int32_t)sc.extent.height : vy1;
      if (x0 < g_pass_bb[0]) g_pass_bb[0] = x0; if (y0 < g_pass_bb[1]) g_pass_bb[1] = y0; if (x1 > g_pass_bb[2]) g_pass_bb[2] = x1; if (y1 > g_pass_bb[3]) g_pass_bb[3] = y1;
      if (g_x_sc1 && x1 > x0 && y1 > y0) ss = (VkRect2D){{x0, y0}, {1, 1}}; }   // r131 DM_VK_X_SC1=1 (live, diagnostic): the first pixel of what the draw can touch (inside the pass's render area); the bounding box keeps the real scissor
    if (!BC_SAME(sc, ss)) { vkCmdSetScissor(cb, 0, 1, &ss); g_bc.sc = ss; g_r131.sc++; }
    if (!BC_SAME(bc, S.blendc)) { vkCmdSetBlendConstants(cb, S.blendc); memcpy(g_bc.bc, S.blendc, sizeof g_bc.bc); }
    if (!BC_SAME(sm, S.smask)) { vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_BIT, (uint32_t)S.smask[0]); vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_BACK_BIT, (uint32_t)S.smask[1]); memcpy(g_bc.sm, S.smask, sizeof g_bc.sm); }
    if (!BC_SAME(sw, S.swmask)) { vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_BIT, (uint32_t)S.swmask[0]); vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_BACK_BIT, (uint32_t)S.swmask[1]); memcpy(g_bc.sw, S.swmask, sizeof g_bc.sw); }
    if (!BC_SAME(sr, S.sref)) { vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_BIT, (uint32_t)S.sref[0]); vkCmdSetStencilReference(cb, VK_STENCIL_FACE_BACK_BIT, (uint32_t)S.sref[1]); memcpy(g_bc.sr, S.sref, sizeof g_bc.sr); }
    if (!BC_SAME(po, S.pofs)) { vkCmdSetDepthBias(cb, S.pofs.units, S.pofs.clamp, S.pofs.factor); g_bc.po = S.pofs; }
    if (fresh || !g_bc.line) { vkCmdSetLineWidth(cb, 1.0f); g_bc.line = 1; }
}
static int g_no_cull = -1;
static int skip_fs(void) {   // r60 DM_VK_SKIP_FS: a draw left out when any of its stages' shaders has a listed key (in the submission's pass n with @n)
    for (int i = 0; i < g_nskip; i++) { if (g_skip[i].pass >= 0 && g_skip[i].pass != g_sub_pass - 1) continue;
        for (int st = 0; st < 6; st++) if (S.sh[st] && S.sh[st]->key == g_skip[i].key) { if (!g_skip[i].hits++) VKLOG("DM_VK_SKIP_FS: first draw left out with shader %016llx (pass %d)", (unsigned long long)g_skip[i].key, g_sub_pass - 1); return 1; } }
    return 0;
}
// When the driver refuses a submission (vkEndCommandBuffer), the next ones find the draw it refuses: only draws below a limit are recorded, the limit
// bisected between a submission that passed and one that failed; the draw found is logged with its whole state (vk_bisect_result, gfx_vk_core.c)
int g_bis_on, g_bis_lo, g_bis_hi, g_bis_trace = -1, g_bis_var; static _Thread_local int g_draw_i;   /* S1e (parallel recording): per thread (counted per recording thread, reset by the commit's thread) */   // g_bis_var: the change tried on the found draw (1 depth bias values 0, 2 no depth bias, 3 dummy textures, 4 vertex offset 0, 5 zero vertex buffer, 6 a new pass)
static int draw_allowed(void) { int i = g_draw_i++; if (i == g_bis_trace) return g_bis_var == 7 ? 0 : 2;
    return !g_bis_on || i < (g_bis_lo + g_bis_hi) / 2; }
void vk_bisect_reset_count(void) { g_draw_i = 0; }
static _Thread_local int g_r142_ok;   /* S1c (parallel recording): per thread */   // r149: the draw may be merged (r142), storage bindings allowed
// R1b (parallel recording): g_mg (r142: the run being merged) in struct rec_ctx (top of the file)
static int r142_merge(VkCommandBuffer cb, int32_t count, uint32_t inst, int32_t bv, uint32_t bi, int xf) {   // 1: drawn here (or merged)
    if (!g_x_merge || !S.sh[0] || S.sh[0]->key != g_x_merge) return 0;
    if (!g_r142_ok || xf || inst != 1) { g_r142.notok++; if (g_mg.a && g_mg.n == 1) g_r142.single++; g_mg.a = NULL; return 0; }
    int di = g_draw_i;   // (draw_allowed counted this draw)
    if (g_mg.a) {
        int seq = g_mg.cb == cb && g_mg.gen == g_cmd_gen && g_mg.pass == g_sub_pass && g_mg.di + 1 == di;
        int st = seq && g_mg.pipe == S.bound_pipe && !memcmp(g_mg.sets, g_bc.sets, sizeof g_mg.sets) && !memcmp(g_mg.vb, g_bc.vb, sizeof g_mg.vb) && !memcmp(g_mg.vo, g_bc.vo, sizeof g_mg.vo);
        int me = st && g_mg.ib == g_bc.ib && g_mg.io == g_bc.io && g_mg.it == g_bc.it && g_mg.count == count && g_mg.bv == bv && g_mg.bi == bi;
        if (me) { g_mg.a->instanceCount++; g_mg.n++; g_mg.di = di; g_r142.merged++; return 1; }
        if (!seq) g_r142.brk_seq++; else if (!st) g_r142.brk_state++; else g_r142.brk_mesh++;
        if (g_mg.n == 1) g_r142.single++;
    }
    VkBuffer b; VkDeviceSize o; VkDrawIndexedIndirectCommand *a = vk_scratch(sizeof *a, 4, &b, &o);
    a->indexCount = (uint32_t)count; a->instanceCount = 1; a->firstIndex = 0; a->vertexOffset = bv; a->firstInstance = bi;
    vkCmdDrawIndexedIndirect(cb, b, o, 1, sizeof *a);
    g_mg.cb = cb; g_mg.gen = g_cmd_gen; g_mg.pass = g_sub_pass; g_mg.di = di; g_mg.n = 1; g_mg.pipe = S.bound_pipe;
    memcpy(g_mg.sets, g_bc.sets, sizeof g_mg.sets); memcpy(g_mg.vb, g_bc.vb, sizeof g_mg.vb); memcpy(g_mg.vo, g_bc.vo, sizeof g_mg.vo);
    g_mg.ib = g_bc.ib; g_mg.io = g_bc.io; g_mg.it = g_bc.it; g_mg.count = count; g_mg.bv = bv; g_mg.bi = bi; g_mg.a = a; g_r142.runs++;
    return 1;
}
static int bc_same_but_dofs(const uint8_t *b) {   // r152: g_bc as saved in b, the dynamic offsets left out
    const size_t o = offsetof(typeof(g_bc), dofs), n = sizeof g_bc.dofs; return !memcmp(b, &g_bc, o) && !memcmp(b + o + n, (const uint8_t *)&g_bc + o + n, sizeof g_bc - o - n); }
// R1b (parallel recording): g_an (r152: the previous draw auto looked at) in struct rec_ctx (top of the file)
static void r152_note(VkCommandBuffer cb, int32_t count, uint32_t inst, int32_t bv, uint32_t bi, int xf) {   // r152 auto: would this draw have continued a run
    const size_t o = offsetof(typeof(g_bc), dofs), n = sizeof g_bc.dofs; int di = g_draw_i; uint64_t key = S.sh[0]->key;
    if (g_an.cb == cb && g_an.gen == g_cmd_gen && g_an.pass == g_sub_pass && g_an.di + 1 == di && g_an.key == key && g_an.pipe == S.bound_pipe && g_an.count == count && g_an.bv == bv && inst == 1 && !bi && !xf && bc_same_but_dofs(g_an.bc)) {
        r152_e *e = r152_get(key, 1);
        if (e && !e->qual) { if (e->frame != g_tex_frame) { e->frame = g_tex_frame; e->cnt = 0; }
            if (++e->cnt >= (uint32_t)g_irun_min) { e->qual = 1; VKLOG("r152: auto: vertex shader %016llx taken (%u of its draws in one frame would have continued a run)", (unsigned long long)key, e->cnt); } } }
    g_an.cb = cb; g_an.gen = g_cmd_gen; g_an.pass = g_sub_pass; g_an.di = di; g_an.key = key; g_an.pipe = S.bound_pipe; g_an.count = count; g_an.bv = bv;
    memcpy(g_an.bc, &g_bc, o); memcpy(g_an.bc + o + n, (const uint8_t *)&g_bc + o + n, sizeof g_bc - o - n);
}
// R1b (parallel recording): g_ir (r151: the run being drawn) in struct rec_ctx (top of the file)
static int r151_draw(VkCommandBuffer cb, int32_t count, uint32_t inst, int32_t bv, uint32_t bi, int xf) {   // 1: drawn here (the pipeline reads the offsets per instance)
    if (!g_irun_on) { if (g_r152_cand) r152_note(cb, count, inst, bv, bi, xf); return 0; }
    int di = g_draw_i;   // (draw_allowed counted this draw)
    r152_e *ke = r152_get(S.sh[0]->key, 1);   // r152
    if (g_ir.a) {
        int seq = g_ir.cb == cb && g_ir.gen == g_cmd_gen && g_ir.pass == g_sub_pass && g_ir.di + 1 == di;
        int st = seq && g_ir.pipe == S.bound_pipe && !memcmp(g_ir.bc, &g_bc, sizeof g_bc);
        int me = st && g_ir.count == count && g_ir.bv == bv && inst == 1 && !bi && !xf;
        if (me && g_ir.n < IRUN_CAP) { memcpy(g_ir.st + 64 * g_ir.n, &g_pofs_words[4], 64); g_ir.a->instanceCount = ++g_ir.n; g_ir.di = di; g_r151.draws++; if (ke) ke->draws++; return 1; }
        if (!seq) g_r151.brk_seq++; else if (!st) { g_r151.brk_state++; r152_e *ko = r152_get(g_ir.key, 1); if (ko) { if (g_ir.pipe == S.bound_pipe && bc_same_but_dofs(g_ir.bc)) ko->brk_win++; else ko->brk_state++; } } else if (!me) g_r151.brk_mesh++; else g_r151.brk_full++;
        g_ir.a = NULL;
    }
    if (inst > IRUN_CAP) inst = IRUN_CAP;
    VkBuffer sb, ab; VkDeviceSize so, ao; uint8_t *p = vk_scratch(64 * IRUN_CAP, 64, &sb, &so);
    for (uint32_t i = 0; i < inst; i++) memcpy(p + 64 * i, &g_pofs_words[4], 64);
    vkCmdBindVertexBuffers(cb, 17, 1, &sb, &so);
    VkDrawIndexedIndirectCommand *a = vk_scratch(sizeof *a, 4, &ab, &ao);
    a->indexCount = (uint32_t)count; a->instanceCount = inst; a->firstIndex = 0; a->vertexOffset = bv; a->firstInstance = bi;
    vkCmdDrawIndexedIndirect(cb, ab, ao, 1, sizeof *a);
    g_r151.runs++; g_r151.draws++; g_r151.inst += inst > 1; if (ke) { ke->runs++; ke->draws++; }
    if (inst == 1 && !bi && !xf) { g_ir.cb = cb; g_ir.gen = g_cmd_gen; g_ir.pass = g_sub_pass; g_ir.di = di; g_ir.n = 1; g_ir.pipe = S.bound_pipe; g_ir.count = count; g_ir.bv = bv; g_ir.st = p; g_ir.a = a; g_ir.key = S.sh[0]->key; memcpy(g_ir.bc, &g_bc, sizeof g_bc); }
    return 1;
}
static int dbg_draw(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_DBG_DRAW"); v = e ? atoi(e) : 0; } return v; }   // debugging: 1 no draw calls, 2 no vertex buffers, 4 no descriptor sets
// a transfer read of 4 bytes of each buffer the addresses are in (outside a render pass, between barriers): see prepare_draw's runs of rasterizer discard
static void rd_read(const uint64_t *a, int n) {
    VkCommandBuffer cb = vk_cmd(); VkBuffer seen[16]; int ns = 0;
    for (int i = 0; i < n; i++) { VkBuffer b, db; VkDeviceSize o, dof; if (!vk_buf_at(a[i], 4, &b, &o)) continue;
        int dup = 0; for (int j = 0; j < ns; j++) dup |= seen[j] == b; if (dup) continue; seen[ns++] = b;
        vk_scratch(4, 4, &db, &dof); VkBufferCopy r = {o, dof, 4}; vk_img_barrier_all(cb); vkCmdCopyBuffer(cb, b, db, 1, &r); vk_img_barrier_all(cb); }
}
static int eden_draw(VkCommandBuffer cb, int prim, int indexed, int var, uint64_t t0);   // NXN1X lane B: gfx_vk_eden.inc
// S1c (parallel recording): g_rd_prev (the last drawn draw had rasterizer discard: prepare_draw's run rule; S1b: the dry pass starts from it) and the run's addresses in rec_xs_t
// everything a draw needs bound; 0 when the draw cannot be done (logged once per reason)
static int prepare_draw(int prim, int indexed) {
    struct rec_ctx *const g_rec = rec_cur();   // R1b (parallel recording): the thread's context read once (S, P_*, g_pass_* below)
    // DM_VK_CB_DRAWS (read every submission, default 128, 0 = never): the replay is cut every that many draws and what is recorded goes to the GPU at once
    // (vk_cmd_split), so the GPU works on the frame while the rest is replayed instead of after all of it: NEO's Scramble Crossing 36.7 → 15.7 ms a frame
    if (!g_wk) g_cb_draws++;   // the cut itself happens between passes (begin_pass); inside one only with DM_VK_PASS_SPLIT (below): a pass cut in two stores and reloads its attachments (S1e (parallel recording): a worker segment's draws counted by the replay thread at its hand-out)
    g_tracing = g_draw_i == g_bis_trace && g_bis_trace >= 0;
    // the change being tried applies to the found draw and every later one (the later draws are refused too: changing one draw alone proves nothing)
    int var = g_bis_trace >= 0 && g_draw_i >= g_bis_trace ? g_bis_var : 0; if (g_bis_var) g_tracing = 0;   // full logs only the first time
    g_var_now = 0;
    if (var == 6) vk_end_pass();   // a new pass for this draw
    // A program may have no fragment stage (a depth-only draw: NVN binds by stage mask, and Metroid Dread draws about nine a frame so in its intro; nvn_mtl.m makes them vertex-only pipelines):
    // drawn with no fragment shader and no colour written (Vulkan leaves colour outputs undefined without one), as before only rasterizer-discard draws were
    const int mesh = prim == DM_PRIM_MESH_TASKS;
    if (mesh ? (!vk_eden_mode() || !S.sh[4] || S.sh[4]->stage != 7) : !S.sh[0]) { g_vkst.skip_prog++; VKRED_ONCE(50, "draw with a program whose shaders are not in the bundle: skipped"); return 0; }
    // r69: a patch draw needs both tessellation stages and a draw with them needs patches (anything else is no valid Vulkan pipeline)
    if (!mesh) { int tes = S.sh[3] || S.sh[4], patch = topo(prim) == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
      if (tes || patch) {
        if (!S.sh[3] || !S.sh[4] || !patch || !V.has_tess) { g_vkst.skip_prog++; static int n; if (n < 8) { n++; VKLOG("[RED] tessellation draw skipped %d: control %016llx evaluation %016llx, primitive %d%s", n,
              S.sh[3] ? (unsigned long long)S.sh[3]->key : 0ull, S.sh[4] ? (unsigned long long)S.sh[4]->key : 0ull, prim, V.has_tess ? "" : ", the device has no tessellation"); } return 0; }
        static int n; if (n < 4) { n++; VKLOG("tessellated draw %d: vertex %016llx control %016llx evaluation %016llx fragment %016llx, patch size %d", n, (unsigned long long)S.sh[0]->key,
              (unsigned long long)S.sh[3]->key, (unsigned long long)S.sh[4]->key, S.sh[1] ? (unsigned long long)S.sh[1]->key : 0ull, S.patch_n); } } }
    if (g_x_nogs && S.sh[2]) return 0;   // DM_VK_X_NOGS=1 (live, A/B): draws with a geometry shader are left out
    for (int i = 0; i < g_x_nskipvs; i++) if (S.sh[0] && S.sh[0]->key == g_x_skipvs[i]) return 0;   // DM_VK_X_SKIPVS=<key>[,<key>...] (live, up to 8): draws with these vertex shaders are left out
    // A run of draws with rasterizer discard gets a pass of its own: the pass ends where such a run begins and where it ends (and begin_pass puts a barrier
    // before the next). They are vertex shaders that write memory the later draws read: Xenoblade Chronicles 3 transforms its meshes with point draws
    // (724ddce6 then 37d04447, cf66cef4), which share passes with its G-buffer draws. NVN's GPU runs draws in order; Adreno, a tiled GPU, runs the vertex
    // shaders of a pass's draws in its binning pass, and draws read vertices not yet written: thin black slivers over the cliffs and grass in 17-20 of
    // 361 frames, 0 with this (0 also with the draws of 37d04447 and cf66cef4 left out). A pass for each such draw instead, splitting the runs, lost
    // Noah's hair (724ddce6's draws come in pairs in one pass); the run kept together draws it. About +1 ms of replay, +0.1 ms of GPU a frame, 33.2 ms
    // frames held. DM_VK_X_RDKEEP=1 (live, A/B): they stay in the pass
    // Where a run ends, the memory its shaders wrote is read by a transfer (rd_read): with barriers alone, the draws of the next passes still met vertices the
    // run had not written yet. Xenoblade Chronicles 3 sways its grass with point draws (5f6491f9 for its depth pre-pass, c0882609 for its G-buffer) whose records
    // the grass cards draw from, the G-buffer's with depth EQUAL against the pre-pass: whole blades failed the test and stayed black, a different few each frame
    // (the grass flicker, stable on the Switch). Grass blades the G-buffer never wrote: 900-2,300 a frame, 0 with this (8 dumps); flicker on screen 4.2-7.7
    // spiking blocks a frame, 0.0-0.5. Everything the two passes read matched (records over 60 frames, instances, index data, shader code: the G-buffer's
    // vertex shader drawing the pre-pass too changed nothing), and a full GPU wait after each submission, explicit barrier bits or a barrier to vertex input
    // did not help; drawing from a copy of the records did, as did a command buffer cut or a transfer read of the written buffer at either end of the run.
    // About +0.45 ms of GPU a frame, 33.2 ms frames held. DM_VK_RD_TOUCH=0 (live, A/B): no read
    { uint64_t *const wa = g_rec->X.rd_wa; int *const nwa = &g_rec->X.rd_nwa;   // S1c (parallel recording): were statics wa / nwa here (rec_xs_t)
      int rd = S.rast_discard; if (!g_x_rdkeep && rd != g_rd_prev && S.pass_open) { if (S.pass_open == 1) g_s65_rdsplit++; vk_end_pass(); }   /* s65: counted */
      if (rd) { if (!g_rd_prev) *nwa = 0; for (int r = 0; r < 16 && *nwa < 16; r++) { uint64_t x = S.ssbo[0][r].addr; int have = 0; for (int i = 0; i < *nwa; i++) have |= wa[i] == x; if (x && !have) wa[(*nwa)++] = x; } }
      else if (g_rd_prev && g_rd_touch && !S.pass_open) rd_read(wa, *nwa);
      g_rd_prev = rd; }
    // Split long passes at draw boundaries while command buffers remain available.
    if (g_psplit > 0 && g_x_cbdraws > 0 && !g_seg_cuts && S.pass_open == 1 && g_pass_draws >= (uint64_t)g_psplit && !S.rast_discard) {   // (S1d (parallel recording): the dry pass's own split while it places the cuts)
        int vk_cmd_parts_left(void); if (vk_cmd_parts_left() > 0) { { static int n; if (n++ < 3) VKLOG("pass split: pass %d (%ux%u) after %llu draws, depth %d", g_sub_pass - 1, S.pass_w, S.pass_h, (unsigned long long)g_pass_draws, S.dfmt); }
            vk_end_pass(); g_psplit_now = 1; } }
    { int lay = draw_layered(); if (S.pass_open == 2 && g_pass_layered != lay) pass_set_layered(lay); else if (S.pass_open == 1 && g_pass_layered != lay) vk_end_pass(); }   // r76: layered or not as the draw
    if (g_s64_pend && S.pass_open == 1) {   // s64: the draw after a pending barrier: the pass goes on unless the draw reads one of its attachments
        if (g_s64_pend == 2) {
            // Keep the dependency pending until a texture consumer or a pass boundary.
            if (!s64_draw_feedback(prim, 1)) ++g_s64_kept;
            else { ++g_s64_fbend; vk_end_pass(); }
        } else if (s64_draw_feedback(prim, 0)) { g_s64_fbend++; vk_end_pass(); } else { g_s64_pend = 0; g_s64_kept++; } }
    if (!begin_pass()) { g_vkst.skip_rt++; return 0; }
    if (carry_one_draw && pending_scale_count && !draw_uses_only_uniforms(prim)) materialize_scale(NULL);
    if (!pass_commit()) { g_vkst.skip_rt++; return 0; }
    if (g_x_tracepass >= 0 && g_sub_pass - 1 == g_x_tracepass) g_tracing = 1;
    g_pkm7_tr = 0;
    if (S.sh[0] && S.sh[0]->key == 0x7fa0c6373aca1086ull && getenv("DM_VK_PKM7")) { static int armed, traced;   // pkm7 (diagnostic, only with DM_VK_PKM7 since pkm8 fixed it): Pokemon Sword's black draw
        if (!armed) { armed = 1; g_pkm7_on = 1; setenv("DM_VK_PASSDUMP_SIZE", "1920x1080", 1); setenv("DM_VK_PASSDUMP_DEPTH", "1", 1); setenv("DM_VK_PASSDUMP", "1", 1);
            VKLOG("pkm7: first draw of vertex shader 7fa0c6373aca1086 (pass %d): the next submission's 1920x1080 passes are dumped and this draw is traced there", g_sub_pass - 1); }
        else if (g_pd_state == 1 && !traced) { traced = 1; g_tracing = 1; g_pkm7_tr = 1; VKLOG("pkm7: === the black draw, pass %d of the dumped submission (present %llu)", g_sub_pass - 1, (unsigned long long)g_tex_frame); } }
    uint64_t t0 = tick();
    VkCommandBuffer cb = vk_cmd();
    if (g_no_cull < 0) g_no_cull = getenv("DM_VK_NOCULL") != NULL;
    if (vk_eden_mode()) { g_var_now = var; if (g_rdiag <= 0) return eden_draw(cb, prim, indexed, var, t0); const uint64_t pe0 = tick(); const int r = eden_draw(cb, prim, indexed, var, t0); g_rd_ns[29] += tick() - pe0; g_rd_n[29]++; return r; }   // par0: slot 29   // NXN1X lane B: Eden's pipeline, set 0 and dynamic state (gfx_vk_eden.inc)
    pkey k; memset(&k, 0, sizeof k);
    for (int i = 0; i < 5; i++) k.key[i] = S.sh[i] ? S.sh[i]->key : 0;
    if (S.sh[3]) k.patch = S.patch_n > 0 ? (uint32_t)S.patch_n : 3;   // r69
    k.topo = topo(prim); k.restart = S.prim_restart && (k.topo == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP || k.topo == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP || k.topo == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN) && indexed;
    k.raster_discard = S.rast_discard; k.alt = (uint32_t)g_spv_alt; k.cull = g_no_cull ? 0 : S.poly.cull;
    // counter-clockwise front faces in Vulkan's framebuffer: Vulkan's clip space has y down, Metal's up (nvn_mtl.m: CCW when (front == CCW) != lower left), and the
    // lower-left origin mode flips the viewport again, so CCW when (front == CCW) == lower left. Taken as given, upper-left games (NEO) culled their ground and walls
    k.front = ((S.poly.front == NVN_FRONT_FACE_CCW) != (g_flipy == 0)) ^ g_queue_ll;   // pkm8: the device origin does not reverse it again (apply_pkm8.py)
    k.mode = S.poly.mode; k.depth_clamp = S.depth_clamp;
    k.bias_en = (S.poly.offset_en & (S.poly.mode == 0 ? 1 : S.poly.mode == 1 ? 2 : 4)) != 0;
    if (var == 2) k.bias_en = 0;
    if (var == 8) k.raster_discard = 1;   // no fragment shader in the pipeline
    int has_depth = (S.daspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0, has_stencil = (S.daspect & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
    k.dt = has_depth && S.ds.dt; k.dw = S.ds.dw; k.df = S.ds.df; k.st = has_stencil && S.ds.st;
    k.dbt = has_depth && S.db_on && V.has_dbounds && !g_x_nodb;   // r69
    // A depth buffer that a fragment shader wrote through gl_FragDepth: Adreno's early culling (LRZ) then rejects whole triangles of later draws that test it, in some frames
    // and not others, with the same draws, state and depth contents. Hades II downsamples its scene depth into a 640x360 D24S8 with such a shader (Blit2DDepth), then
    // draws its lights into it with the depth test on and no depth write: the lights went out in turn every 24-48 frames (whole halves of light quads missing in pass dumps),
    // and Xenoblade Chronicles 3's cloud dome (its depth downsampled the same way) lost whole triangles. The light shader rewritten to write gl_FragDepth = gl_FragCoord.z
    // (the same depth, tested after the shader) brought the pulses to 0 in 5 recordings, against 8-21; neither writing the depth image back by transfer (one texel or the
    // whole image) nor waiting for the GPU after each submission changed them. So draws that test such a buffer (written in the last two presents) use that variant.
    // Not with a depth bias (gl_FragCoord.z need not carry it). DM_VK_FDZ=0 (live): never
    { vkimg_t *dim = has_depth ? P_att[8].im : NULL; vkshader_t *fsh = k.raster_discard ? NULL : S.sh[1];
      if (dim && fsh && fsh->depth_repl && k.dt && k.dw) dim->fdz_frame = g_tex_frame + 1;
      k.fdz = g_fdz && dim && fsh && !fsh->depth_repl && k.dt && !k.bias_en && dim->fdz_frame && g_tex_frame + 1 - dim->fdz_frame <= 2;
      // r56: the alpha test (gfx_vk_shader.c spv_alpha_test): functions NEVER (1) to GEQUAL (7) on a shader with a colour 0; 0 and ALWAYS (8) test nothing
      k.atest = g_atest && fsh && (fsh->out_locs & 1) && S.color.alpha_func >= 1 && S.color.alpha_func <= 7;
      if (!k.atest && g_x_push && g_atest && fsh && (fsh->out_locs & 1)) k.atest = 2;   // r132 DM_VK_X_PUSH: the variant that reads the push constant
      if (S.color.alpha_func >= 1 && S.color.alpha_func <= 7) { static int n; if (n < 8) { n++; VKLOG("alpha test used %d: shader %016llx function %u reference %g%s", n, fsh ? (unsigned long long)fsh->key : 0ull,
          (unsigned)S.color.alpha_func, (double)S.alpha_ref, k.atest ? "" : g_atest ? " (no colour 0: not tested)" : " (DM_VK_ALPHA=0: not tested)"); } } }
    // r57: Bayonetta 3 draws its characters' colour after a depth pass with the depth test EQUAL and no depth write (Witch-Time issue 92). On Turnip the two vertex
    // stages' depths differ in the last bits even with invariant positions (r55), so a part of the colour pass fails the test: purple dots on faces in cutscenes, as
    // in the emulators (NXN1X 10-30: Turnip only, Qualcomm's driver clean). Such a draw tests LEQUAL (GEQUAL in a reversed-depth game, by the direction of the
    // last depth-writing test) with g_deq units of bias towards the camera. Not with the game's own bias or a fragment-depth variant (its depth does not carry a bias)
    int deq = 0;
    if (k.dt && k.dw) { if (k.df == 2 || k.df == 4) g_deq_dir = 1; else if (k.df == 5 || k.df == 7) g_deq_dir = 2; }
    if (g_deq > 0 && g_deq_dir && k.dt && !k.dw && k.df == 3 && !k.bias_en && !k.fdz) {
        deq = g_deq_dir; k.df = deq == 1 ? 4 : 7; k.bias_en = 1;
        static int n; if (n < 4) { n++; VKLOG("depth EQUAL relaxed %d: %s with %d units of bias towards the camera (DM_VK_DEPTH_EQ=0: off)", n, deq == 1 ? "LEQUAL" : "GEQUAL", g_deq); } }
    k.sf[0] = S.ds.sf_f; k.sf[1] = S.ds.sf_b; k.sfail[0] = S.ds.fail_f; k.szfail[0] = S.ds.zfail_f; k.szpass[0] = S.ds.zpass_f; k.sfail[1] = S.ds.fail_b; k.szfail[1] = S.ds.zfail_b; k.szpass[1] = S.ds.zpass_b;
    // vertex input: every location the vertex shader reads
    vkshader_t *vsh = S.sh[0]; uint32_t in_mask = vsh ? vsh->local[0] : 0, int_mask = vsh ? vsh->local[1] | vsh->local[2] : 0, sint_mask = vsh ? vsh->local[1] : 0;
    uint32_t used_streams = 0; int need_zero = 0;
    for (int loc = 0; loc < 16; loc++) {
        if (!(in_mask >> loc & 1)) continue;
        va_st a; memcpy(&a, &S.vattr[loc], 4);
        VkFormat f = a.fmt ? vertex_format(a.fmt, int_mask >> loc & 1, sint_mask >> loc & 1) : VK_FORMAT_UNDEFINED;
        if (f == VK_FORMAT_UNDEFINED || !S.vbuf[a.stream].address) { k.attr[k.nattr++] = (typeof(k.attr[0])){(uint32_t)loc, 16, int_mask >> loc & 1 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32B32A32_SFLOAT, 0}; need_zero = 1; continue; }
        k.attr[k.nattr++] = (typeof(k.attr[0])){(uint32_t)loc, a.stream, f, a.offset};
        used_streams |= 1u << a.stream;
    }
    for (int s = 0; s < 16; s++) if (used_streams >> s & 1) k.bind[k.nbind++] = (typeof(k.bind[0])){(uint32_t)s, S.vstream[s].stride, S.vstream[s].divisor ? 1u : 0u, S.vstream[s].divisor};
    if (need_zero) k.bind[k.nbind++] = (typeof(k.bind[0])){16, 0, 0, 0};
    // attachments and blending
    k.ncol = (uint32_t)S.ncol; k.dfmt = S.dfmt; k.samples = S.samples ? S.samples : 1; k.a2c = S.ms.a2c;
    k.logic_en = S.color.logic_op != 3 && S.color.logic_op != 0 && 0;   // logic ops: not on mobile render targets in version 1
    k.logic_op = S.color.logic_op;
    for (int i = 0; i < S.ncol; i++) {
        k.cfmt[i] = S.cfmt[i];
        k.wmask[i] = S.cfmt[i] && S.sh[1] && (g_x_allmask || (S.sh[1]->out_locs >> i & 1)) ? (S.chmask.mask >> (4 * i)) & 0xf : 0;   // no fragment shader: no colour written
        // nor to a target the fragment shader has no output for: Vulkan leaves such a target undefined, and Adreno wrote garbage there (Donkey Kong Bananza: black ceilings
        // with white halos, 5b6f44a4 writes only location 0 of three targets; bright flickering points, c8637178 writes 0 and 1 of three). DM_VK_X_ALLMASK=1 (live, A/B): as before
        const vkfmt_t *f = S.rt_has_col[i] ? vk_fmt(S.rt_col[i].d.format) : NULL;
        if (((S.color.blend_en >> i) & 1) && f && !f->is_int) {
            k.blend_en |= 1u << i;
            k.bl[i].src = S.blend[i].src_rgb; k.bl[i].dst = S.blend[i].dst_rgb; k.bl[i].srca = S.blend[i].src_a; k.bl[i].dsta = S.blend[i].dst_a; k.bl[i].eq = S.blend[i].eq_rgb; k.bl[i].eqa = S.blend[i].eq_a;
        }
    }
    k.clip_neg1 = !g_depth_zero;
    // layout and pipeline
    VkDescriptorSetLayout dsl[5] = {0}; int nsets = 0;
    for (int st = 0; st < 5; st++) if (S.sh[st] && !(st == 1 && k.raster_discard)) { dsl[S.sh[st]->set] = S.sh[st]->dsl; if (S.sh[st]->set + 1 > nsets) nsets = S.sh[st]->set + 1; }
    // the last draw's set layouts and what vk_pipeline_layout made of them (it fills absent stages with the empty layout): its lock and hash took ~10% of the
    // replay thread on Unicorn Overlord's battle map. Pipeline layouts are never destroyed
    static VkDescriptorSetLayout l_in[5], l_out[5]; static int l_n = -1; static VkPipelineLayout l_lay; VkPipelineLayout layout;
    if (nsets == l_n && !memcmp(dsl, l_in, sizeof l_in)) { memcpy(dsl, l_out, sizeof l_out); layout = l_lay; }
    else { memcpy(l_in, dsl, sizeof l_in); l_n = nsets; layout = l_lay = vk_pipeline_layout(dsl, nsets); memcpy(l_out, dsl, sizeof l_out); }
    g_pofs_st[0] = g_pofs_st[1] = 0; g_irun_on = 0;   // r133 DM_VK_X_POFS
    int irun = g_x_irun && indexed == 2 && S.sh[0] && !S.sh[2] && !S.sh[3] && !S.sh[4] && !k.raster_discard && !k.dbt && !(k.atest & 3) && k.nattr + 4 <= 16 && k.nbind < 17 &&
               !r132_vs_inst(S.sh[0]) && !(S.sh[0]->in_locs >> 28) && !(S.sh[0]->out_locs >> 28 & 3) && !(S.sh[1] && S.sh[1]->in_locs >> 28 & 3);   // r151 DM_VK_X_IRUN
    for (uint32_t i = 0; irun && i < k.nbind; i++) if (k.bind[i].rate) irun = 0;
    int want = g_x_irun && S.sh[0] && irun_want(S.sh[0]->key);   // r152: listed, or taken by auto
    g_r152_cand = irun && g_irun_auto && !want; if (!want) irun = 0;   // r152: a draw auto may take later (r152_note)
    if (want && !irun) g_r151.notok++;
    if (irun || (g_x_pofs && S.sh[0] && S.sh[0]->key == g_x_pofs && !S.sh[2] && !S.sh[3] && !S.sh[4])) {
        vkshader_t *f = k.raster_discard ? NULL : S.sh[1];
        if ((k.atest & 3) == 2) k.atest &= ~3u;
        g_pofs_st[0] = r133_ok(S.sh[0], g_pofs_m[0], irun); g_pofs_st[1] = f && !(k.atest & 3) && (!irun || g_pofs_st[0]) && r133_ok(f, g_pofs_m[1], irun);
        if (irun && !g_pofs_st[0]) { g_pofs_st[1] = 0; g_r151.novar++; g_r151.notok++; }
        else if (irun) { k.atest |= 1u << 24 | 4u | g_pofs_m[0] << 8 | (g_pofs_st[1] ? 8u | g_pofs_m[1] << 16 : 0); g_irun_on = 1;   // r151: the offsets per instance (r151_draw), no push constants
            for (uint32_t q = 0; q < 4; q++) k.attr[k.nattr++] = (typeof(k.attr[0])){28 + q, 17, VK_FORMAT_R32G32B32A32_UINT, 16 * q};
            k.bind[k.nbind++] = (typeof(k.bind[0])){17, 64, 1, 0}; }
        else if (g_pofs_st[0] || g_pofs_st[1]) { k.atest |= (g_pofs_st[0] ? 4u | g_pofs_m[0] << 8 : 0) | (g_pofs_st[1] ? 8u | g_pofs_m[1] << 16 : 0); layout = vk_pipeline_layout_pofs(dsl, nsets); } }   // r134: masks in the key
    // descriptor sets first: the views they are given decide which shader variants the pipeline takes (integer texels, vk_shader_module)
    g_var_now = var;
    VkDescriptorSet sets[5] = {0};
    uint32_t dofs[5][96], ndofs = 0, dall[5 * 96];
    uint64_t t1d = tick();
    for (int st = 0; st < 5; st++) if (S.sh[st] && !(st == 1 && k.raster_discard)) { sets[S.sh[st]->set] = stage_set(st, S.sh[st], dofs[S.sh[st]->set]); k.um[st] = g_num_um; k.im[st] = g_num_im; }
    uint64_t tdesc = tick() - t1d; g_vkst.ns_desc += tdesc;
    if (g_pofs_st[0] || g_pofs_st[1]) { g_r133.draws++; g_r133.vs += (uint64_t)g_pofs_st[0]; g_r133.fs += (uint64_t)g_pofs_st[1];   // r133: window bases to the bind, offsets in them to the push
        const uint64_t al = V.props.limits.minUniformBufferOffsetAlignment ? V.props.limits.minUniformBufferOffsetAlignment : 16;
        for (int st = 0; st < 2; st++) if (g_pofs_st[st] && S.sh[st]) { int set = S.sh[st]->set;
            for (int j = 0; j < 8; j++) { pofs_slot_t *q = &g_pofs_slot[st][j]; uint32_t *wd = &g_pofs_words[(st ? 12 : 4) + j]; *wd = 0;
                if (j >= S.sh[st]->ndyn || !q->conv) continue;
                uint64_t o = dofs[set][j]; typeof(g_pofs_win[0][0]) *p = &g_pofs_win[st][j];
                if (p->buf != q->buf || p->epoch != g_set_epoch || (g_pofs_mode != 1 && (o < p->base || o - p->base + q->N > q->R))) {   // r134 mode 1: kept
                    uint64_t lo = o + q->N > q->R ? o + q->N - q->R : 0, hi = o < q->Sz - q->R ? o : q->Sz - q->R, b = o & ~32767ull;
                    hi -= hi % al; if (b < lo) b = lo; if (b > hi) b = hi; b -= b % al; if (b < lo) b = hi;
                    p->buf = q->buf; p->base = b; p->epoch = g_set_epoch; p->o0 = o; g_r133.moves++; }
                dofs[set][j] = (uint32_t)p->base; *wd = (uint32_t)(((g_pofs_mode == 1 ? p->o0 : o) - p->base) >> 4); g_r133.conv++; } } }
    if (g_r153_cap && (g_pofs_st[0] || g_pofs_st[1]) && g_r153_n < g_r153_cap) {   // r153: where draws with converted buffers render (DM_VK_LOG_UPLOADS)
        const void *c = S.rt_n > 0 && S.rt_has_col[0] ? S.rt_col[0].mem : NULL, *z = S.rt_has_dep ? S.rt_dep.mem : NULL; uint64_t vk = S.sh[0] ? S.sh[0]->key : 0, fk = S.sh[1] ? S.sh[1]->key : 0; int i = 0;
        for (; i < 512 && (g_r153_seen[i].c || g_r153_seen[i].z || g_r153_seen[i].vs); i++) if (g_r153_seen[i].c == c && g_r153_seen[i].z == z && g_r153_seen[i].vs == vk && g_r153_seen[i].fs == fk) break;
        if (i < 512 && g_r153_seen[i].c == c && g_r153_seen[i].z == z && g_r153_seen[i].vs == vk && g_r153_seen[i].fs == fk && (c || z || vk)) {}
        else { if (i < 512) { g_r153_seen[i].c = c; g_r153_seen[i].z = z; g_r153_seen[i].vs = vk; g_r153_seen[i].fs = fk; } g_r153_n++;
            VKLOG("r153: present %llu converted draw: vertex shader %016llx fragment %016llx (stages %d%d) into colour 0 %p %ux%u nvn format %#x (%d colour targets), depth %p %ux%u",
                  (unsigned long long)g_tex_frame, (unsigned long long)vk, (unsigned long long)fk, g_pofs_st[0], g_pofs_st[1], c, c ? S.rt_col[0].d.width : 0, c ? S.rt_col[0].d.height : 0, c ? S.rt_col[0].d.format : 0, S.rt_n,
                  z, z ? S.rt_dep.d.width : 0, z ? S.rt_dep.d.height : 0); } }
    if (g_tracing) {
        VKLOG("  pipeline: topology %u restart %u discard %u cull %u front %u mode %u clamp %u bias %u; depth test %u write %u func %u stencil %u; attributes %u bindings %u; colour %u (format %u) samples %u a2c %u; blend enable %#x logic %u/%u; fragment depth variant %u (depth written by a shader at present %llu, now %llu); alpha test %u (function %u reference %g)",
              k.topo, k.restart, k.raster_discard, k.cull, k.front, k.mode, k.depth_clamp, k.bias_en, k.dt, k.dw, k.df, k.st, k.nattr, k.nbind, k.ncol, k.cfmt[0], k.samples, k.a2c, k.blend_en, k.logic_en, k.logic_op,
              k.fdz, has_depth && P_att[8].im ? (unsigned long long)P_att[8].im->fdz_frame : 0ull, (unsigned long long)g_tex_frame + 1, k.atest, (unsigned)S.color.alpha_func, (double)S.alpha_ref);
        if (k.st) VKLOG("  stencil: func %u/%u fail %u/%u zfail %u/%u pass %u/%u; compare mask %#x/%#x write mask %#x/%#x reference %#x/%#x", k.sf[0], k.sf[1], k.sfail[0], k.sfail[1], k.szfail[0], k.szfail[1], k.szpass[0], k.szpass[1],
                        (unsigned)S.smask[0], (unsigned)S.smask[1], (unsigned)S.swmask[0], (unsigned)S.swmask[1], (unsigned)S.sref[0], (unsigned)S.sref[1]);
        for (uint32_t i = 0; i < k.ncol; i++) VKLOG("  blend %u: src %#x dst %#x eq %u, alpha src %#x dst %#x eq %u, write mask %#x", i, k.bl[i].src, k.bl[i].dst, k.bl[i].eq, k.bl[i].srca, k.bl[i].dsta, k.bl[i].eqa, k.wmask[i]);
        for (uint32_t i = 0; i < k.nattr; i++) VKLOG("  attribute location %u binding %u format %u offset %u", k.attr[i].loc, k.attr[i].binding, k.attr[i].fmt, k.attr[i].off);
        for (uint32_t i = 0; i < k.nbind; i++) VKLOG("  binding %u stride %u rate %u divisor %u", k.bind[i].binding, k.bind[i].stride, k.bind[i].rate, k.bind[i].divisor);
    }
    crumb_draw(k.key[0], k.key[1], k.key[2]);
    VkPipeline p = graphics_pipeline(&k, layout);
    uint64_t t1 = tick(); g_vkst.ns_pipe += t1 - t0 - tdesc;
    if (!p) { g_vkst.skip_pipe++; return 0; }
    if (g_ubolog_key && g_ubolog_n > 0 && S.sh[0] && S.sh[0]->key == g_ubolog_key) {   // DM_VK_UBO_LOG: the pipeline the draw got, and its key's hash
        uint64_t kh = fnv(1469598103934665603ull, &k, sizeof k); VKLOG("ubo: present %llu pipeline %p key %016llx um %llx/%llx im %llx/%llx", (unsigned long long)g_tex_frame, (void *)p, (unsigned long long)kh,
              (unsigned long long)k.um[0], (unsigned long long)k.um[1], (unsigned long long)k.im[0], (unsigned long long)k.im[1]); }
    g_vkst.draws++; if (++g_pass_draws > g_vkst.max_pass_draws) g_vkst.max_pass_draws = g_pass_draws;
#if defined(DM_NXN1X)
    fg_observe_draw();
#endif
    if (p != S.bound_pipe) { vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p); S.bound_pipe = p; g_r131.pipe++; }
    if ((k.atest & 3) == 1) { uint32_t pc[2] = {S.color.alpha_func - 1u, 0}; memcpy(&pc[1], &S.alpha_ref, 4); vkCmdPushConstants(cb, layout, (k.atest & 12) ? VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pc, pc); }   // r56: pass mask, reference
    else if ((k.atest & 3) == 2 && nsets >= 2) { static const uint32_t z[2]; vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof z, z); g_r132.push++; }   // r132 DM_VK_X_PUSH
    else if (g_x_push) g_r132.pushx++;
    if ((k.atest & 12) && !(k.atest >> 24 & 1)) vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 16, 64, &g_pofs_words[4]);   // r133 (r151: not with the instance stream)
    set_dynamic(cb);
    if (k.dbt) { float lo = S.db_lo < 0 ? 0 : S.db_lo > 1 ? 1 : S.db_lo, hi = S.db_hi < 0 ? 0 : S.db_hi > 1 ? 1 : S.db_hi; vkCmdSetDepthBounds(cb, lo, hi);   // r69 (no VK_EXT_depth_range_unrestricted: 0..1)
        static int n; if (n < 2) { n++; VKLOG("depth bounds test %d: %g .. %g (DM_VK_NODB=1, live: off)", n, (double)S.db_lo, (double)S.db_hi); } }
    if (var == 1) { vkCmdSetDepthBias(cb, 0, 0, 0); memset(&g_bc.po, 0xff, sizeof g_bc.po); }
    if (deq) { vkCmdSetDepthBias(cb, deq == 1 ? (float)-g_deq : (float)g_deq, 0, 0); memset(&g_bc.po, 0xff, sizeof g_bc.po); }   // r57: the next draw sets its own again
    uint32_t at[6] = {0};   // where each set's dynamic offsets start in dall
    for (int i = 0; i < nsets; i++) { at[i] = ndofs; for (int st = 0; st < 5; st++)   // the dynamic offsets: by set, then binding
        if (S.sh[st] && !(st == 1 && k.raster_discard) && S.sh[st]->set == i) { memcpy(dall + ndofs, dofs[i], sizeof(uint32_t) * (size_t)S.sh[st]->ndyn); ndofs += (uint32_t)S.sh[st]->ndyn; } }
    at[nsets] = ndofs;
    for (int i = 0; i < nsets; i++) if (!sets[i]) sets[i] = vk_alloc_set(dsl[i]);   // an absent stage's set: from the empty layout vk_pipeline_layout filled in
    // DM_VK_SETSPLIT=1 (live): under the same layout, only the run of sets whose handle or dynamic offsets changed is bound again. Adreno's vkCmdBindDescriptorSets
    // took ~22% of the replay thread on Unicorn Overlord's battle map (one load in it), and a draw that moves only its vertex stage's offsets sent the fragment set too
    if (!(dbg_draw() & 4) && g_setsplit && !bc_fresh() && g_bc.lay == layout && g_bc.nsets == (uint32_t)nsets && g_bc.ndofs == ndofs) {
        int lo = -1, hi = -1;
        for (int i = 0; i < nsets; i++) if (g_bc.sets[i] != sets[i] || memcmp(g_bc.dofs + at[i], dall + at[i], sizeof *dall * (at[i + 1] - at[i]))) { if (lo < 0) lo = i; hi = i; }
        if (lo >= 0) { g_r131.dset++; g_r131.dsets += (uint64_t)(hi - lo + 1); vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, (uint32_t)lo, (uint32_t)(hi - lo + 1), sets + lo, at[hi + 1] - at[lo], dall + at[lo]);
            memcpy(g_bc.sets + lo, sets + lo, sizeof *sets * (size_t)(hi - lo + 1)); memcpy(g_bc.dofs + at[lo], dall + at[lo], sizeof *dall * (at[hi + 1] - at[lo])); } }
    else if (!(dbg_draw() & 4)) { int fresh = bc_fresh();
        if (fresh || g_bc.lay != layout || g_bc.nsets != (uint32_t)nsets || g_bc.ndofs != ndofs || memcmp(g_bc.sets, sets, sizeof *sets * (size_t)nsets) || memcmp(g_bc.dofs, dall, sizeof *dall * ndofs)) {
            g_r131.dset++; g_r131.dsets += (uint64_t)nsets; g_r131.dset_ofs += !fresh && g_bc.lay == layout && g_bc.nsets == (uint32_t)nsets && !memcmp(g_bc.sets, sets, sizeof *sets * (size_t)nsets);   // r131
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, (uint32_t)nsets, sets, ndofs, dall);
            g_bc.lay = layout; g_bc.nsets = (uint32_t)nsets; g_bc.ndofs = ndofs; memcpy(g_bc.sets, sets, sizeof *sets * (size_t)nsets); memcpy(g_bc.dofs, dall, sizeof *dall * ndofs); } }
    // vertex buffers
    for (uint32_t i = 0; i < k.nbind; i++) {
        uint32_t b = k.bind[i].binding; VkBuffer buf; VkDeviceSize off; if (b == 17) continue;   // r151: the instance stream (r151_draw)
        if (b == 16) { buf = vk_zero_buffer(); off = 0; }
        else if (!((S.vbuf[b].size && vk_buf_at(S.vbuf[b].address, S.vbuf[b].size, &buf, &off)) || vk_buf_at(S.vbuf[b].address, 1, &buf, &off))) {   /* r78: as far as it reads */ buf = vk_zero_buffer(); off = 0; VKRED_ONCE(51, "vertex buffer outside the memory pools"); }
        else { if (g_vb_on) vb_note(0, S.vbuf[b].address, S.vbuf[b].size);   // vbytes (NXN1X_VBYTES, log only)
            if (S.vbuf[b].size) vbm_bind(S.vbuf[b].address, S.vbuf[b].size, &buf, &off); }   // vbmir
        if (g_var_now == 5) { buf = vk_zero_buffer(); off = 0; }
        if (!(dbg_draw() & 2) && b < 17) { int fresh = bc_fresh(); if (fresh || g_bc.vb[b] != buf || g_bc.vo[b] != off) { vkCmdBindVertexBuffers(cb, b, 1, &buf, &off); g_bc.vb[b] = buf; g_bc.vo[b] = off; g_r131.vb++; } }
    }
    (void)indexed;
    g_r131.gs += !!S.sh[2]; g_r131.tes += !!S.sh[3]; g_r131.rd += !!k.raster_discard; g_r131_dupok = 0; g_r142_ok = 0;   // r131
    if (g_x_dup || g_x_census || g_x_vb || g_x_ib || g_x_merge || g_x_vstab) { int st = 0;   // r135: the same class
        for (int s = 0; s < 2 && !st; s++) for (int i = 0; S.sh[s] && i < S.sh[s]->nbind; i++) { int dk = S.sh[s]->b[i].desc; if (dk == RD_SSBO || dk == RD_STORAGE || dk == RD_STORAGE_TBUF) { st = 1; break; } }
        g_r131.stor += (uint64_t)st; if (g_x_vstab) r143_note(st); g_r131_dupok = !st && !S.sh[2] && !S.sh[3] && !S.sh[4] && !k.raster_discard; g_r142_ok = !S.sh[2] && !S.sh[3] && !S.sh[4] && !k.raster_discard; }   /* r149: merging takes draws with storage bindings too */
    g_r132_instok = 0; if (g_x_inst) { int ok = S.sh[0] && !r132_vs_inst(S.sh[0]); for (uint32_t i = 0; ok && i < k.nbind; i++) if (k.bind[i].rate) ok = 0; g_r132_instok = ok; }   // r132
    return 1;
}
// quads without an index buffer: one buffer of 0 1 2 0 2 3, 4 5 6 4 6 7, ... for up to QUAD_MAX quads (more are drawn as the first QUAD_MAX)
#define QUAD_MAX 65536u
static VkBuffer quad_index_buffer(uint32_t *nq) {
    static VkBuffer b; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    if (*nq > QUAD_MAX) { VKRED_ONCE(58, "quad draw of %u quads: the first %u drawn", *nq, QUAD_MAX); *nq = QUAD_MAX; }
    pthread_mutex_lock(&mu);
    if (!b) {
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, QUAD_MAX * 6 * 4, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &b)); VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, b, &req);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
        VkDeviceMemory m; VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, b, m, 0));
        uint32_t *p; VKCHK(vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, (void **)&p));
        static const uint32_t pat[6] = {0, 1, 2, 0, 2, 3};
        for (uint32_t q = 0; q < QUAD_MAX; q++) for (int k = 0; k < 6; k++) p[q * 6 + k] = q * 4 + pat[k];
    }
    pthread_mutex_unlock(&mu);
    return b;
}
// r62: indirect quads (DrawArraysIndirect, primitive 7). The record {count, instances, first, base instance} is usually written by the GPU earlier in the frame
// (Hyrule Warriors: Age of Imprisonment's effects), so the CPU cannot expand it. A one-thread compute pass turns it into a VkDrawIndexedIndirectCommand
// {count / 4 * 6, instances, 0, first, base instance} over the shared quad index buffer: the upstream Metal backend's ixc_run / ixc_arrays (nvn_mtl.m, dm/gfx/msl/ixc.metal).
// A compute pass cannot run inside a render pass: an open one is ended here (it begins again at the draw). Before r62 the record was drawn as a triangle list of
// its vertices (wrong triangles; HWAoI lost the device right after the first ones, 11-07). DM_VK_IXC=0: the old way.
#include "ixc_quads_comp.spv.h"
static struct { int state; VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline pipe; uint64_t runs; } g_ixq;
static int ixq_ready(void) {
    if (g_ixq.state) return g_ixq.state > 0;
    g_ixq.state = -1;
    { const char *e = getenv("DM_VK_IXC"); if (e && !atoi(e)) { VKLOG("indirect quads drawn as triangle lists (DM_VK_IXC=0)"); return 0; } }
    VkDescriptorSetLayoutBinding b[2] = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 2, b};
    if (vkCreateDescriptorSetLayout(V.dev, &li, NULL, &g_ixq.dsl)) { VKRED_ONCE(102, "indirect quads: descriptor set layout failed (drawn as triangle lists)"); return 0; }
    void vk_dsl_register(VkDescriptorSetLayout l, const VkDescriptorSetLayoutBinding *b, uint32_t n); vk_dsl_register(g_ixq.dsl, b, 2);
    VkPipelineLayoutCreateInfo pi = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_ixq.dsl, 0, NULL};   // no push constants: they would touch the draws' (r56 alpha test)
    if (vkCreatePipelineLayout(V.dev, &pi, NULL, &g_ixq.pl)) { VKRED_ONCE(102, "indirect quads: pipeline layout failed (drawn as triangle lists)"); return 0; }
    VkShaderModule m; VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof ixc_quads_comp_spv, ixc_quads_comp_spv};
    if (vkCreateShaderModule(V.dev, &mi, NULL, &m)) { VKRED_ONCE(102, "indirect quads: shader module failed (drawn as triangle lists)"); return 0; }
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", NULL}, g_ixq.pl, VK_NULL_HANDLE, -1};
    VkResult r = vkCreateComputePipelines(V.dev, V.pcache, 1, &ci, NULL, &g_ixq.pipe); vkDestroyShaderModule(V.dev, m, NULL);
    if (r) { VKRED_ONCE(102, "indirect quads: compute pipeline failed (%d, drawn as triangle lists)", r); return 0; }
    g_ixq.state = 1; return 1;
}
// the converted arguments of the indirect quad draw whose record is at ind (20 bytes at *ab + *ao); 0: not converted (drawn the old way)
static int ixq_args(uint64_t ind, VkBuffer *ab, VkDeviceSize *ao) {
    if (!ixq_ready()) return 0;
    VkBuffer rb; VkDeviceSize ro;
    if (!vk_buf_at(ind, 16, &rb, &ro)) return 0;
    if (ro & 3) { VKRED_ONCE(103, "indirect quads: a record not 4-byte aligned (drawn as a triangle list)"); return 0; }
    VkDeviceSize al = V.props.limits.minStorageBufferOffsetAlignment ? V.props.limits.minStorageBufferOffsetAlignment : 16, r0 = ro / al * al;
    VkBuffer ob; VkDeviceSize oo; uint32_t *z = vk_scratch(32, al, &ob, &oo);
    memset(z, 0, 32); z[5] = (uint32_t)((ro - r0) / 4); z[6] = QUAD_MAX * 6;   // zeros: no draw if the pass never ran
    vk_end_pass_land();   // (as dispatch: the deferred uniform updates land first)
    VkCommandBuffer cb = vk_cmd();
    VkMemoryBarrier m1 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};   // the record's writers before the read
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &m1, 0, NULL, 0, NULL);
    VkDescriptorSet set = vk_alloc_set(g_ixq.dsl);
    VkDescriptorBufferInfo bi[2] = {{rb, r0, ro - r0 + 16}, {ob, oo, 32}};
    VkWriteDescriptorSet wr[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi[0], NULL},
                                  {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi[1], NULL}};
    vkUpdateDescriptorSets(V.dev, 2, wr, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ixq.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ixq.pl, 0, 1, &set, 0, NULL);
    vkCmdDispatch(cb, 1, 1, 1);
    VkMemoryBarrier m2 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &m2, 0, NULL, 0, NULL);
    if (!g_ixq.runs++) VKLOG("indirect quads: converted on the GPU into indexed indirect draws over the quad index buffer (first one: record %#llx; DM_VK_IXC=0 = the old triangle lists)", (unsigned long long)ind);
    else if (!(g_ixq.runs & (g_ixq.runs - 1)) && g_ixq.runs >= 1024) VKLOG("indirect quads: %llu converted so far (each ends the open render pass)", (unsigned long long)g_ixq.runs);
    *ab = ob; *ao = oo; return 1;
}
// quads: a triangle list of indices made on the CPU (from the game's index buffer for indexed draws)
static int quad_indices(uint32_t count, const uint8_t *src, int itype, int32_t first, VkBuffer *buf, VkDeviceSize *off) {
    uint32_t nq = count / 4; if (!nq) return 0;
    uint32_t *d = vk_scratch((VkDeviceSize)nq * 6 * 4, 4, buf, off);
    for (uint32_t q = 0; q < nq; q++) for (int k = 0; k < 6; k++) {
        static const int pat[6] = {0, 1, 2, 0, 2, 3}; uint32_t i = q * 4 + (uint32_t)pat[k];
        d[q * 6 + k] = !src ? (uint32_t)first + i : itype == 0 ? src[i] : itype == 1 ? ((const uint16_t *)src)[i] : ((const uint32_t *)src)[i];
    }
    return (int)(nq * 6);
}
// ---------------- transform feedback ----------------
// NVN's capture state (OP_TF_BIND / OP_TF_STATE). Each capturing draw (feedback on, a vertex shader translated with Xfb: shader_vk.py xfb_layout) is a Vulkan capture of its own:
// the buffers bound at the bytes written so far, Begin, the draw, End. So no pipeline is bound and no pass ends while a capture is active, and no counter buffer is read back:
// the host counts the bytes, as nvn_mtl.m tf_draw does (vertices the primitives emit × the buffer's stride). Xenoblade Chronicles 3 steps its GPU particles this way (8 shaders,
// up to 9 outputs interleaved in one buffer); before, those draws wrote nothing and the particles kept stale or zero state
static uint32_t xfb_verts(int prim, uint32_t n) {   // vertices a capture writes for n vertices of this primitive (strips and fans are written as lists)
    switch (topo(prim)) {
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: return n & ~1u;
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: return n >= 2 ? (n - 1) * 2 : 0;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: return n / 3 * 3;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN: return n >= 3 ? (n - 2) * 3 : 0;
    default: return n;
    }
}
static int xfb_begin(VkCommandBuffer cb) {   // inside the pass, after the pipeline is bound; 1 when a capture began
    vkshader_t *vs = S.sh[0];
    if (!g_tf_on || !V.has_xfb || !vs || !vs->xfb || !g_xfb_on) return 0;
    VkBuffer b[4]; VkDeviceSize o[4], n[4];
    for (int k = 0; k < 4; k++) {
        vkpool_t *p = g_tf[k].addr ? vk_pool_of(g_tf[k].addr + g_tf_off[k], 4) : NULL;
        if (vs->xfb_stride[k] && p && g_tf_off[k] < g_tf[k].size && vk_buf_at(g_tf[k].addr + g_tf_off[k], 4, &b[k], &o[k])) {
            n[k] = g_tf[k].size - g_tf_off[k]; if (o[k] + n[k] > p->hi - p->lo) n[k] = p->hi - p->lo - o[k];
        } else { vk_scratch(64, 64, &b[k], &o[k]); n[k] = 64; if (vs->xfb_stride[k]) VKRED_ONCE(83, "transform feedback buffer %d unbound, full or outside the memory pools: its capture is dropped", k); }
    }
    vkCmdBindTransformFeedbackBuffersEXT(cb, 0, 4, b, o, n);
    vkCmdBeginTransformFeedbackEXT(cb, 0, 0, NULL, NULL);
    return 1;
}
static void xfb_end(VkCommandBuffer cb, int prim, uint32_t count, uint32_t inst) {   // after the draw: the capture ends and the bytes it wrote are counted
    vkCmdEndTransformFeedbackEXT(cb, 0, 0, NULL, NULL);
    uint32_t v = xfb_verts(prim, count) * (inst ? inst : 1); g_tf_in_pass = 1; g_tf_verts += v;
    for (int k = 0; k < 4; k++) {
        uint64_t m = (uint64_t)v * S.sh[0]->xfb_stride[k]; if (!m || !g_tf[k].addr) continue;
        if (g_tf_off[k] + m > g_tf[k].size) m = g_tf[k].size > g_tf_off[k] ? g_tf[k].size - g_tf_off[k] : 0;
        void vk_gpu_wrote(uint64_t a, uint64_t n); if (m) vk_gpu_wrote(g_tf[k].addr + g_tf_off[k], m);
        g_tf_off[k] += m;
    }
}
// DrawTransformFeedback draws the vertices the captures ended into one control buffer wrote: counted since Begin (g_tf_verts) and kept by control buffer address at End / Pause
// (a table by address: Xenoblade Chronicles 3 gives every particle emitter a control buffer of its own, hundreds of them, 32 bytes apart)
enum { TFC = 4096 }; static struct { uint64_t ctl; uint32_t verts; } g_tf_ctl[TFC];
static int tf_ctl_slot(uint64_t ctl, int add) {   // the entry of a control buffer; with add, a new one (the oldest in its probe range is replaced when they are all taken)
    unsigned h = (unsigned)((ctl >> 5) * 2654435761u) % TFC;
    for (int k = 0; k < 16; k++) { unsigned i = (h + (unsigned)k) % TFC; if (g_tf_ctl[i].ctl == ctl) return (int)i; if (!g_tf_ctl[i].ctl) return add ? (int)i : -1; }
    return add ? (int)h : -1;
}
static void tf_state(const p_tf_state *t) {
    switch (t->what & 3) {
    case 0: memset(g_tf_off, 0, sizeof g_tf_off); g_tf_verts = 0; g_tf_on = 1; break;   // Begin: written from the start of each buffer
    case 3: g_tf_on = 1; break;                                          // Resume: on from the bytes written
    default: g_tf_on = 0;                                                // End / Pause: what was captured is read by later draws (as vertices), after the barrier the next pass begins with
        if (g_tf_in_pass) { vk_end_pass(); g_tf_in_pass = 0; }
        int i = tf_ctl_slot(t->ctl, 1); g_tf_ctl[i].ctl = t->ctl; g_tf_ctl[i].verts = g_tf_verts;
    }
}
// drawprof (NXN1X_DRAWPROF=<vertex shader key in hex>, log only): the first 32 indexed draws of that vertex shader each get three timestamps: TOP_OF_PIPE before the draw
// (the GPU reaches the command), BOTTOM_OF_PIPE before it (everything recorded earlier has finished) and BOTTOM_OF_PIPE after it (the draw has finished). wait = the
// first two apart (earlier work still running when the draw is reached), own = the last two (this draw alone, at most). Read back 3 presents later, logged once.
// Hyrule Warriors' soldier draws (VS 1df3a0c17a3e9e80): 8-14 ms passes that write nothing, the same with a stub vertex shader: the draw itself, or inherited work?
enum { DP_N = 32 };
static VkQueryPool g_dp_pool, g_dp_stat; static int g_dp_n, g_dp_done; static uint64_t g_dp_frame; static uint32_t g_dp_cnt[DP_N], g_dp_inst[DP_N]; static int g_dp_pass[DP_N];
// pstat (log only, with drawprof): a pipeline statistics query around each timed draw: input-assembly vertices and primitives, vertex shader invocations, clipping
// invocations and the primitives clipping put out, fragment shader invocations. Where does a draw's time go when it writes nothing?
enum { DP_STATS = 6 };
static const VkQueryPipelineStatisticFlags DP_STAT_FLAGS = VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT | VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
    VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT | VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT | VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
    VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
static int drawprof_begin(VkCommandBuffer cb, uint32_t count, uint32_t inst) {
    static int64_t want = -1;
    if (want < 0) { const char *e = getenv("NXN1X_DRAWPROF"); want = e && *e ? (int64_t)strtoull(e, NULL, 16) : 0; }
    if (!want || g_dp_done || g_dp_n >= DP_N || !S.sh[0] || S.sh[0]->key != (uint64_t)want) return -1;
    if (!g_dp_pool) {
        PFN_vkResetQueryPool reset = (PFN_vkResetQueryPool)vkGetDeviceProcAddr(V.dev, "vkResetQueryPool");
        VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, DP_N * 3};
        if (!reset || vkCreateQueryPool(V.dev, &qi, NULL, &g_dp_pool) != VK_SUCCESS) { g_dp_done = 1; VKLOG("drawprof: off (no host query reset or query pool)"); return -1; }
        reset(V.dev, g_dp_pool, 0, DP_N * 3);   // host reset (hostQueryReset, which Eden enables): the draws are inside render passes, where vkCmdResetQueryPool is not allowed
        VkQueryPoolCreateInfo si = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_PIPELINE_STATISTICS, DP_N, DP_STAT_FLAGS};   // pstat
        if (vkCreateQueryPool(V.dev, &si, NULL, &g_dp_stat) != VK_SUCCESS) g_dp_stat = VK_NULL_HANDLE; else reset(V.dev, g_dp_stat, 0, DP_N);
        VKLOG("drawprof: timing the first %d indexed draws of vertex shader %016llx; pipeline statistics %s", DP_N, (unsigned long long)want, g_dp_stat ? "on" : "unavailable");
    }
    const int k = g_dp_n++;
    g_dp_cnt[k] = count; g_dp_inst[k] = inst; g_dp_pass[k] = g_sub_pass - 1;
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_dp_pool, (uint32_t)k * 3);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_dp_pool, (uint32_t)k * 3 + 1);
    if (g_dp_stat) vkCmdBeginQuery(cb, g_dp_stat, (uint32_t)k, 0);   // pstat
    if (g_dp_n == DP_N) { extern uint64_t g_tex_frame; g_dp_frame = g_tex_frame; }
    return k;
}
static void drawprof_end(VkCommandBuffer cb, int k) {
    if (k < 0) return;
    if (g_dp_stat) vkCmdEndQuery(cb, g_dp_stat, (uint32_t)k);   // pstat
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_dp_pool, (uint32_t)k * 3 + 2);
}
static void drawprof_poll(void) {   // at a later draw: once, 3 presents after the last timed draw
    extern uint64_t g_tex_frame;
    if (!g_dp_pool || g_dp_done || g_dp_n < DP_N || g_tex_frame < g_dp_frame + 3) return;
    g_dp_done = 1;
    uint64_t t[DP_N * 3];
    if (vkGetQueryPoolResults(V.dev, g_dp_pool, 0, DP_N * 3, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) { VKLOG("drawprof: results unavailable"); return; }
    const double ns = V.props.limits.timestampPeriod;
    double wait_sum = 0, own_sum = 0;
    uint64_t st[DP_N * DP_STATS]; int have_st = g_dp_stat &&
        vkGetQueryPoolResults(V.dev, g_dp_stat, 0, DP_N, sizeof st, st, DP_STATS * 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS;   // pstat
    for (int k = 0; k < DP_N; k++) {
        const double wait = (double)(t[k * 3 + 1] - t[k * 3]) * ns / 1e6, own = (double)(t[k * 3 + 2] - t[k * 3 + 1]) * ns / 1e6;
        wait_sum += wait; own_sum += own;
        VKLOG("drawprof: draw %2d pass %3d: %u indices x %u instances (%llu vertices): earlier work still running %.3f ms, the draw itself %.3f ms", k, g_dp_pass[k], g_dp_cnt[k],
              g_dp_inst[k], (unsigned long long)g_dp_cnt[k] * g_dp_inst[k], wait, own);
        if (have_st) { const uint64_t *s = st + k * DP_STATS;   // the order of the flag bits: IA vertices, IA primitives, VS invocations, clipping invocations, clipping primitives, FS invocations
            VKLOG("drawprof:   statistics: input vertices %llu primitives %llu, vertex shader runs %llu, clipping in %llu out %llu, fragment shader runs %llu",
                  (unsigned long long)s[0], (unsigned long long)s[1], (unsigned long long)s[2], (unsigned long long)s[3], (unsigned long long)s[4], (unsigned long long)s[5]); }
    }
    VKLOG("drawprof: %d draws: earlier work %.2f ms, the draws themselves %.2f ms in all", DP_N, wait_sum, own_sum);
}
static void draw(const p_draw *d);
static void r136_draw(const char *kind, int prim, uint32_t count, uint32_t inst, uint64_t ind, int ok) {   // r136: the -2 listing's line per draw
    VKLOG("    %s%s prim %d count %u instances %u%s%#llx; shaders %016llx %016llx %016llx %016llx %016llx; targets %d (formats %d %d) depth format %d; viewport %.0f %.0f %.0f %.0f", kind, ok ? "" : " (not drawn)", prim, count, inst,
          ind ? " indirect " : " direct ", (unsigned long long)ind, (unsigned long long)(S.sh[0] ? S.sh[0]->key : 0), (unsigned long long)(S.sh[1] ? S.sh[1]->key : 0), (unsigned long long)(S.sh[2] ? S.sh[2]->key : 0),
          (unsigned long long)(S.sh[3] ? S.sh[3]->key : 0), (unsigned long long)(S.sh[4] ? S.sh[4]->key : 0), S.ncol, S.cfmt[0], S.cfmt[1], S.dfmt, S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3]);
    // svcprobe (log only): the Eden hashes of the first two stages, the names of their SPIR-V dumps (gpu_log_shader_dumps: <title>_<Eden hash>.spv)
    VKLOG("      Eden hashes %016llx %016llx", (unsigned long long)(S.sh[0] && S.sh[0]->eden ? nvn_eden_shader_hash(S.sh[0]->eden) : 0),
          (unsigned long long)(S.sh[1] && S.sh[1]->eden ? nvn_eden_shader_hash(S.sh[1]->eden) : 0));
}
static void trace_draw(const char *kind, int prim, uint32_t count, uint32_t inst, int itype) {
    VKLOG("draw: %s prim %d count %u instances %u itype %d; shaders %016llx %016llx %016llx; targets %d (formats %d %d %d %d) depth format %d samples %u, pass %ux%u; viewport %.0f %.0f %.0f %.0f; streams:",
          kind, prim, count, inst, itype, (unsigned long long)(S.sh[0] ? S.sh[0]->key : 0), (unsigned long long)(S.sh[1] ? S.sh[1]->key : 0), (unsigned long long)(S.sh[2] ? S.sh[2]->key : 0),
          S.ncol, S.cfmt[0], S.cfmt[1], S.cfmt[2], S.cfmt[3], S.dfmt, S.samples, S.pass_w, S.pass_h, S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3]);
    for (int i = 0; i < S.rt_n; i++) if (S.rt_bad[i]) VKLOG("  colour target %d: not a texture object (first word %#x): left out", i, S.rt_bad[i]);
    for (int i = 0; i < S.rt_n; i++) if (S.rt_has_col[i]) { vkimg_t *im = vk_tex(&S.rt_col[i]); const tex_t *t = &S.rt_col[i];
        VKLOG("  colour target %d: serial %llu → image %p: %ux%ux%u layers %u levels %u samples %u fmt %d vtype %d; nvn target %d format %#x flags %#x stride %u size %llu pool %p+%#llx",
              i, (unsigned long long)t->serial, (void *)im, im ? im->w : 0, im ? im->h : 0, im ? im->d : 0, im ? im->layers : 0, im ? im->levels : 0, im ? im->samples : 0, im ? im->fmt : 0, im ? im->vtype : 0,
              t->d.target, t->d.format, t->d.flags, t->d.stride, (unsigned long long)t->L.size, (const void *)t->pool, (unsigned long long)t->off);
        if (S.rt_has_cview[i]) { const view_t *v = &S.rt_cview[i]; VKLOG("  its view: levels %d (%u+%u) layers %d (%u+%u) target %d (%u) format %d (%#x)",
              v->set_levels, v->base_level, v->nlevels, v->set_layers, v->min_layer, v->nlayers, v->set_target, v->target, v->set_format, v->format); } }
    if (S.rt_has_dep) { vkimg_t *im = vk_tex(&S.rt_dep); const tex_t *t = &S.rt_dep;
        VKLOG("  depth target: serial %llu → image %p: %ux%u fmt %d; nvn format %#x flags %#x size %llu pool %p+%#llx", (unsigned long long)t->serial, (void *)im, im ? im->w : 0, im ? im->h : 0,
              im ? im->fmt : 0, t->d.format, t->d.flags, (unsigned long long)t->L.size, (const void *)t->pool, (unsigned long long)t->off); }
    VKLOG("  scissor set %d: %d %d %d %d; depth range %.3f %.3f; viewport count first %.1f %.1f %.1f %.1f", S.sc_set, S.sc.x, S.sc.y, S.sc.w, S.sc.h, S.dnear, S.dfar, S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3]);
    vkshader_t *vsh = S.sh[0]; uint32_t in_mask = vsh ? vsh->local[0] : 0;
    for (int l = 0; l < 16; l++) if (in_mask >> l & 1) { uint32_t a = S.vattr[l]; VKLOG("  location %d: attrib state %#x (stream %u) stride %d divisor %d buffer %#llx +%llu", l, a, (a >> 16) & 0x1f,
        S.vstream[(a >> 16) & 0xf].stride, S.vstream[(a >> 16) & 0xf].divisor, (unsigned long long)S.vbuf[(a >> 16) & 0xf].address, (unsigned long long)S.vbuf[(a >> 16) & 0xf].size); }
}
// Indirect-count multi-draws (hle_nvn.c: a call of up to MDI_INLINE draws is recorded as max indirect draws carrying the count's address and their index). Each used to be drawn,
// past the count too, from stale records. Now the first of a run draws the whole run with vkCmdDraw(Indexed)IndirectCount, the GPU reading the count, and the rest are skipped.
// exec_ops scans the run at its first record (mdi_scan); DM_VK_MDI_COUNT=0 = the old way
// S1c (parallel recording): g_mdi_cnt, g_mdi_n, g_mdi_stride (the run being drawn) in rec_xs_t
static int mdi_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_MDI_COUNT"); on = !(e && *e == '0') && vkCmdDrawIndirectCount && vkCmdDrawIndexedIndirectCount;
    VKLOG("indirect-count multi-draws: %s", on ? "vkCmdDraw(Indexed)IndirectCount" : "each record drawn (DM_VK_MDI_COUNT=0 or no device function)"); } return on; }
static void mdi_scan(const uint8_t *q, uint32_t op, uint64_t ind0, uint64_t cnt) {   // q: the record after a run's first
    int n = 1; uint64_t stride = 0;
    for (int guard = 0; guard < 1 << 16; guard++) {
        op_hdr h; memcpy(&h, q, sizeof h);
        if (h.op == OP_JUMP) { memcpy(&q, q + 8, 8); continue; }
        if (h.op != op || h.size < 8) break;
        uint64_t ind, c; int32_t i;
        if (op == OP_DRAW) { p_draw t; const p_draw *d = draw_rec(q + 8, h.size, &t); ind = d->ind; c = d->mdi_cnt; i = d->mdi_i; }
        else { p_draw_idx t; const p_draw_idx *d = draw_idx_rec(q + 8, h.size, &t); ind = d->ind; c = d->mdi_cnt; i = d->mdi_i; }
        if (c != cnt || i != n || ind <= ind0) break;
        if (n == 1) stride = ind - ind0; else if (ind - ind0 != stride * (uint64_t)n) break;
        n++; q += h.size;
    }
    g_mdi_cnt = cnt; g_mdi_n = n; g_mdi_stride = stride;
}
static int mdi_follower(uint64_t cnt, int i) { return cnt && i > 0 && cnt == g_mdi_cnt && i < g_mdi_n && mdi_on(); }   // drawn by its run's first record
// the run's first record: draw them all with the count read by the GPU; 0 when that cannot be done (the caller draws the record as before)
static int mdi_draw(VkCommandBuffer cb, uint64_t ind, uint64_t cnt, int i, uint32_t rec, int indexed, int ok) {
    if (!cnt || i || cnt != g_mdi_cnt || !mdi_on()) return 0;
    uint64_t st = g_mdi_n > 1 ? g_mdi_stride : rec; VkBuffer b, cbf; VkDeviceSize o, co;
    if (st < rec || st % 4 || st > 0xffffffffu || !vk_buf_at(ind, st * (uint64_t)(g_mdi_n - 1) + rec, &b, &o) || o % 4 || !vk_buf_at(cnt, 4, &cbf, &co) || co % 4) { VKRED_ONCE(101, "an indirect-count multi-draw not drawable with the GPU's count (stride %llu): its records drawn each", (unsigned long long)st); return 0; }
    { static int n; if (n < 8) { n++; VKLOG("indirect-count multi-draw: %d records, stride %llu, count at %#llx", g_mdi_n, (unsigned long long)st, (unsigned long long)cnt); } }
    if (ok) { if (indexed) vkCmdDrawIndexedIndirectCount(cb, b, o, cbf, co, (uint32_t)g_mdi_n, (uint32_t)st); else vkCmdDrawIndirectCount(cb, b, o, cbf, co, (uint32_t)g_mdi_n, (uint32_t)st); }
    return 1;
}
// s18 (with the s15 probe, DM_VK_SLOWSUB): a non-indexed draw's time by part, reported with each slow submission (run 19: 35 DRAWs
// took 45-80 ms of wall time a frame with little CPU). 0 quad conversion, 1 prepare_draw, 2 command buffer and queries, 3 indirect, 4 quads, 5 plain
extern int g_ss_on; uint64_t g_ss_dr[8], g_ss_drn[8]; static uint64_t g_dr_t; static int g_dr_k = -1;
static inline void dr_mark(int k) { uint64_t t = tnow(); if (g_dr_k >= 0) g_ss_dr[g_dr_k] += t - g_dr_t; g_dr_t = t; g_dr_k = k; if (k >= 0) g_ss_drn[k]++; }
#define DR(k) do { if (g_ss_on > 0) dr_mark(k); } while (0)
static void draw_body(const p_draw *d);
static void draw(const p_draw *d) { if (mdi_follower(d->mdi_cnt, d->mdi_i)) return; DR(0); draw_body(d); if (g_occ2) vk_occ2_end(); DR(-1); }   // (occ-s1: the draw's query ends on every exit of draw_body; occ-s3fast: its conditional block stays open for the next gated draw)
static _Thread_local VkDescriptorBufferInfo mesh_draw_args;   // (S1c (parallel recording): the mesh draw being recorded's, per thread)

static void draw_mesh_tasks(const p_draw *d) {
    static PFN_vkCmdDrawMeshTasksIndirectEXT draw_indirect;
    if (!draw_indirect)
        draw_indirect = (PFN_vkCmdDrawMeshTasksIndirectEXT)vkGetDeviceProcAddr(V.dev, "vkCmdDrawMeshTasksIndirectEXT");
    if (!draw_indirect) { VKRED_ONCE(198, "NVN mesh draw requires VK_EXT_mesh_shader"); return; }
    VkBuffer src = VK_NULL_HANDLE;
    VkDeviceSize src_offset = 0;
    if (d->ind && !vk_buf_at(d->ind, 8, &src, &src_offset)) {
        VKRED_ONCE(199, "NVN mesh indirect arguments outside memory pools");
        return;
    }
    VkDeviceSize alignment = V.props.limits.minStorageBufferOffsetAlignment;
    uint32_t *args = vk_scratch(16, alignment < 4 ? 4 : alignment, &mesh_draw_args.buffer, &mesh_draw_args.offset);
    mesh_draw_args.range = 16;
    args[0] = (uint32_t)d->count;
    args[1] = args[2] = 1;
    args[3] = (uint32_t)d->first;
    if (src) {
        vk_end_pass();
        VkCommandBuffer cb = vk_cmd();
        VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
        VkDeviceSize dst = mesh_draw_args.offset;
        VkBufferCopy copies[2] = {{src_offset, dst, 4}, {src_offset + 4, dst + 12, 4}};
        vkCmdCopyBuffer(cb, src, mesh_draw_args.buffer, 2, copies);
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
            VK_PIPELINE_STAGE_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT, 0, 1, &barrier, 0, NULL, 0, NULL);
    }
    g_vpull_direct = 0;   // vtxpull: no vertex input
    if (!prepare_draw(DM_PRIM_MESH_TASKS, 0) || !draw_allowed() || (dbg_draw() & 1)) return;
    VkCommandBuffer cb = vk_cmd();
    vk_oq_draw(cb); if (g_occ2) vk_occ2_begin(cb, g_oser - 1); if (g_occ2 >= 3) rcg_begin(cb, 0);   // (occ-s1: the draw's own query, DM_VK_OCC2=s1; occ-s3: a GPU condition's block; occ-s3fast: the open one when its predicate is the same)
    { void vk_bar_flush(VkCommandBuffer); vk_bar_flush(cb); }   // cbar2: barriers held for cb recorded before this draw (not through the s63 wrappers)
    draw_indirect(cb, mesh_draw_args.buffer, mesh_draw_args.offset, 1, sizeof(VkDrawMeshTasksIndirectCommandEXT));
    static int logged;
    if (!logged) { logged = 1; VKLOG("NVN mesh draw submitted (EXT, GPU firstTask, indirect=%d)", !!src); }
}
static void draw_body(const p_draw *d) {
    if (d->prim == DM_PRIM_MESH_TASKS) { g_draw_binst = 0; draw_mesh_tasks(d); return; }
    if (d->mdi_cnt && !mdi_on()) VKRED_ONCE(52, "multi-draw indirect with a count: drawn once");
    int prim = d->prim;
    // before prepare_draw, whose driver segment carries it (c0 + 0x644): set after it, each draw's segment had the draw before's. An indirect draw's is in GPU memory: 0
    g_draw_binst = d->ind ? 0 : (uint32_t)d->base_instance;
    VkBuffer ixb = VK_NULL_HANDLE; VkDeviceSize ixo = 0;   // r62: indirect quads converted first (ixq_args), before prepare_draw: its compute pass cannot run inside the render pass
    if (d->ind && prim == 7 && !d->mdi_cnt && !ixq_args(d->ind, &ixb, &ixo)) ixb = VK_NULL_HANDLE;
    DR(1);   // s18
    g_vpull_direct = !d->ind && !d->mdi_cnt && prim != 7;   // vtxpull
    if (!prepare_draw(prim == 7 ? 4 : prim, prim == 7)) return;
    DR(2);   // s18
    VkCommandBuffer cb = vk_cmd(); vk_oq_draw(cb); if (g_occ2) vk_occ2_begin(cb, g_oser - 1); if (g_occ2 >= 3) rcg_begin(cb, 0);   // (occ-s1: the draw's own query, ended by draw / draw_idx; occ-s3: a GPU condition's block; occ-s3fast: kept open across draws with the same predicate)
    int g_ok = draw_allowed(); if (g_x_listpasses) r136_draw("draw", d->prim, (uint32_t)d->count, (uint32_t)d->instances, d->ind, g_ok); if (g_ok == 2 || (g_tracing && (g_x_tracepass >= 0 || g_pkm7_tr))) trace_draw("draw", d->prim, (uint32_t)d->count, (uint32_t)d->instances, -1);
    if (g_nskip && skip_fs()) g_ok = 0;
    if (g_var_now == 10 && prim == 7) g_ok = 0; if (g_var_now == 11 && prim != 7) g_ok = 0;   // bisection: quads from the found draw on, or the rest
    { static int nq, nn; if (g_var_now) { if (prim == 7) nq++; else nn++; if ((nq + nn) % 50 == 1) VKLOG("draws from the found one on: %d quads, %d others", nq, nn); } }
    uint32_t inst = d->instances ? (uint32_t)d->instances : 1;
    { static int n; if (n < 12 && (d->ind || (uint32_t)d->count > (1u << 20) || inst > (1u << 16))) { n++;
        VKLOG("unusual draw: count %u instances %u indirect %#llx prim %d first %d", (unsigned)d->count, inst, (unsigned long long)d->ind, d->prim, d->first); } }
    if (d->ind) { g_r131.ind++; DR(3);   // s18
        VkBuffer b; VkDeviceSize o;
        if (g_tf_on && S.sh[0] && S.sh[0]->xfb) VKRED_ONCE(84, "indirect draw under transform feedback: not captured");
        if (ixb) { r144_note(d->ind, ixb, ixo, g_ok); uint32_t nq = QUAD_MAX; VkBuffer qb = quad_index_buffer(&nq); bc_index(cb, qb, 0, VK_INDEX_TYPE_UINT32); if (!(dbg_draw() & 1) && g_ok) vkCmdDrawIndexedIndirect(cb, ixb, ixo, 1, 20); return; }
        if (prim == 7) VKRED_ONCE(104, "indirect quads with a draw count or not converted: drawn as triangle lists");
        if (mdi_draw(cb, d->ind, d->mdi_cnt, d->mdi_i, 16, 0, !(dbg_draw() & 1) && g_ok)) return;
        if (vk_buf_at(d->ind, 16, &b, &o)) if (!(dbg_draw() & 1) && g_ok) vkCmdDrawIndirect(cb, b, o, 1, 16); else VKRED_ONCE(53, "indirect draw arguments outside the memory pools");
        return;
    }
    if (prim == 7) { g_r131.quad++; DR(4);   // quads: the triangle list indices of one static buffer (built once), the first vertex as the vertex offset (s18: DR)
        uint32_t nq = (uint32_t)d->count / 4; if (!nq) return;
        if (g_tf_on && S.sh[0] && S.sh[0]->xfb) VKRED_ONCE(85, "quads under transform feedback: not captured");
        VkBuffer qb = quad_index_buffer(&nq);
        bc_index(cb, qb, 0, VK_INDEX_TYPE_UINT32); if (!(dbg_draw() & 1) && g_ok) vkCmdDrawIndexed(cb, nq * 6, inst, 0, g_var_now == 4 ? 0 : d->first, (uint32_t)d->base_instance); return; }
    if (prim == 2) VKRED_ONCE(54, "line loops drawn as line strips");
    DR(5);   // s18
    if (!(dbg_draw() & 1) && g_ok) { int xf = xfb_begin(cb); uint32_t bi = r132_binst((uint32_t)d->base_instance, inst); if ((g_x_vb || g_x_ib) && g_r131_dupok && !xf) r135_force(cb, 0); vkCmdDraw(cb, (uint32_t)d->count, inst, (uint32_t)d->first, bi);
        for (int r = r131_draw(0, (uint32_t)d->count, inst, xf); r > 0; r--) vkCmdDraw(cb, (uint32_t)d->count, inst, (uint32_t)d->first, bi);   // r131 DM_VK_X_DUP (live, diagnostic)
        if (xf) xfb_end(cb, prim, (uint32_t)d->count, inst); }
}
static void draw_idx_body(const p_draw_idx *d);
static void draw_idx(const p_draw_idx *d) { draw_idx_body(d); if (g_occ2) vk_occ2_end(); }   // occ-s1: the draw's query (vk_occ2_begin) ends on every exit of the body (occ-s3fast: its conditional block stays open)
static void draw_idx_body(const p_draw_idx *d) {
    drawprof_poll();   // drawprof (log only)
    if (mdi_follower(d->mdi_cnt, d->mdi_i)) return;
    if (d->mdi_cnt && !mdi_on()) VKRED_ONCE(55, "multi-draw indirect (indexed) with a count: drawn once");
    int prim = d->prim;
    g_draw_binst = d->ind ? 0 : (uint32_t)d->base_instance;   // before prepare_draw (see draw)
    g_vpull_direct = !d->ind && !d->mdi_cnt && prim != 7;   // vtxpull
    if (!prepare_draw(prim == 7 ? 4 : prim, 1 + (g_x_irun && !d->ind && prim != 7 && !d->base_instance && !g_x_inst && (d->instances ? (uint32_t)d->instances : 1) <= IRUN_CAP))) return;   // r151: 2 = may take the instance stream
    VkCommandBuffer cb = vk_cmd(); vk_oq_draw(cb); if (g_occ2) vk_occ2_begin(cb, g_oser - 1); if (g_occ2 >= 3) rcg_begin(cb, 0);   // (occ-s1: the draw's own query, ended by draw / draw_idx; occ-s3: a GPU condition's block; occ-s3fast: kept open across draws with the same predicate)
    int g_ok = draw_allowed(); if (g_x_listpasses) r136_draw("indexed draw", d->prim, (uint32_t)d->count, (uint32_t)d->instances, d->ind, g_ok); if (g_ok == 2 || (g_tracing && (g_x_tracepass >= 0 || g_pkm7_tr))) trace_draw("indexed draw", d->prim, (uint32_t)d->count, (uint32_t)d->instances, d->itype);
    if (g_nskip && skip_fs()) g_ok = 0;
    if (g_var_now == 11) g_ok = 0;
    VkIndexType it = d->itype == 0 ? VK_INDEX_TYPE_UINT8_EXT : d->itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
    uint32_t isz = d->itype == 0 ? 1 : d->itype == 1 ? 2 : 4;
    VkBuffer ib; VkDeviceSize io;
    uint64_t ilen = d->ind || !d->count ? 1 : (uint64_t)(uint32_t)d->count * isz;   /* (ibuf is where the draw's indices start) */   // r78: as far as it reads
    if (!(vk_buf_at(d->ibuf, ilen, &ib, &io) || (ilen > 1 && vk_buf_at(d->ibuf, 1, &ib, &io)))) { VKRED_ONCE(56, "index buffer outside the memory pools"); return; }
    if (g_vb_on) { if (d->ind) g_vbs.ind++; else vb_note(1, d->ibuf, ilen); }   // vbytes (NXN1X_VBYTES, log only)
    if (g_imir && !d->ind && prim != 7 && ilen > 1) { VkBuffer mb; VkDeviceSize mo; if (imir_put((const void *)(uintptr_t)d->ibuf, ilen, &mb, &mo)) { ib = mb; io = mo; } }   // idxmir A/B
    else if (!d->ind && prim != 7 && ilen > 1) vbm_bind(d->ibuf, ilen, &ib, &io);   // vbmir (an indirect draw's count is unknown here: its indices stay in the pool)
    if (g_ubolog_key && S.sh[0] && S.sh[0]->key == g_ubolog_key && g_ubolog_n > 0) {   // DM_VK_UBO_LOG: the index data the draw reads (CPU view at replay)
        const uint8_t *ix = (const uint8_t *)(uintptr_t)d->ibuf; uint64_t h = 1469598103934665603ull; uint32_t mn = ~0u, mx = 0;
        for (uint32_t i = 0; i < (uint32_t)d->count; i++) { uint32_t v = isz == 1 ? ix[i] : isz == 2 ? ((const uint16_t *)ix)[i] : ((const uint32_t *)ix)[i]; h = (h ^ v) * 1099511628211ull; if (v < mn) mn = v; if (v > mx) mx = v; }
        VKLOG("ubo: present %llu index buffer %#llx count %d base vertex %d indirect %#llx hash %016llx range %u..%u", (unsigned long long)g_tex_frame, (unsigned long long)d->ibuf, d->count, d->base_vertex,
              (unsigned long long)d->ind, (unsigned long long)h, mn, mx);
        if (S.vbuf[0].address && mn != ~0u) { uint32_t st = S.vstream[0].stride ? S.vstream[0].stride : 12; uint64_t v0 = (uint64_t)(int64_t)d->base_vertex + mn;   // the first vertex this draw reads, every word
            if ((v0 + 1) * st <= S.vbuf[0].size) { const float *vf = (const float *)(uintptr_t)(S.vbuf[0].address + v0 * st); char t[512]; int k = 0;
                for (uint32_t i = 0; i < st / 4 && i < 24; i++) k += snprintf(t + k, sizeof t - k, " %g", vf[i]);
                VKLOG("ubo: present %llu first vertex read (%llu):%s", (unsigned long long)g_tex_frame, (unsigned long long)v0, t); } }
        static uint64_t dumped; if (dumped != h && S.vbuf[0].address && d->count <= 4096) { dumped = h; char t[1024]; int k = 0; uint32_t i0 = 0;   // once per index data: every index,
            for (uint32_t i = 0; i < (uint32_t)d->count; i++) { uint32_t v = isz == 1 ? ix[i] : isz == 2 ? ((const uint16_t *)ix)[i] : ((const uint32_t *)ix)[i];   // then every word of stream 0's vertices
                k += snprintf(t + k, sizeof t - k, " %u", v); if (k > 900 || i + 1 == (uint32_t)d->count) { VKLOG("ubo: indices from %u:%s", i0, t); k = 0; i0 = i + 1; } }
            uint32_t st = S.vstream[0].stride ? S.vstream[0].stride : 12, nv = (uint32_t)(S.vbuf[0].size / st);
            for (uint32_t i = 0; i < nv && i <= mx; i++) { const float *vf = (const float *)(uintptr_t)(S.vbuf[0].address + (uint64_t)i * st);   // every word of stream 0's vertex (up to 8)
                for (uint32_t w = 0; w < st / 4 && w < 8; w++) k += snprintf(t + k, sizeof t - k, "%s%g", w ? "," : " ", vf[w]);
                if (k > 800 || i + 1 == nv || i == mx) { VKLOG("ubo: vertices to %u:%s", i, t); k = 0; } } } }
    uint32_t inst = d->instances ? (uint32_t)d->instances : 1;
    { static int n; if (n < 12 && (d->itype == 0 || d->ind || (uint32_t)d->count > (1u << 20) || inst > (1u << 16))) { n++;
        VKLOG("unusual indexed draw: itype %d count %u instances %u indirect %#llx prim %d base vertex %d", d->itype, (unsigned)d->count, inst, (unsigned long long)d->ind, d->prim, d->base_vertex); } }
    if (d->ind && prim == 7) VKRED_ONCE(105, "indexed indirect quads: drawn as triangle lists (not converted yet)");
    if (d->ind) { g_r131.ind++; VkBuffer b; VkDeviceSize o; ind_note(d, isz); bc_index(cb, ib, io, it); if (mdi_draw(cb, d->ind, d->mdi_cnt, d->mdi_i, 20, 1, !(dbg_draw() & 1) && g_ok)) return; if (vk_buf_at(d->ind, 20, &b, &o)) if (!(dbg_draw() & 1) && g_ok) vkCmdDrawIndexedIndirect(cb, b, o, 1, 20); return; }
    if (prim == 7) { g_r131.quad++; VkBuffer b; VkDeviceSize o; int n = quad_indices((uint32_t)d->count, (const uint8_t *)(uintptr_t)d->ibuf, d->itype, 0, &b, &o); if (!n) return;
        bc_index(cb, b, o, VK_INDEX_TYPE_UINT32); if (!(dbg_draw() & 1) && g_ok) vkCmdDrawIndexed(cb, (uint32_t)n, inst, 0, d->base_vertex, (uint32_t)d->base_instance); return; }
    if (io % isz) { VKRED_ONCE(57, "index buffer offset not aligned to its index size"); return; }
    bc_index(cb, ib, io, it);
    const int dp = !(dbg_draw() & 1) && g_ok ? drawprof_begin(cb, (uint32_t)d->count, inst) : -1;   // drawprof (NXN1X_DRAWPROF, log only)
    if (!(dbg_draw() & 1) && g_ok) { int xf = xfb_begin(cb); uint32_t bi = r132_binst((uint32_t)d->base_instance, inst); if ((g_x_vb || g_x_ib) && g_r131_dupok && !xf) r135_force(cb, 1); if (!r151_draw(cb, d->count, inst, d->base_vertex, bi, xf) && !r142_merge(cb, d->count, inst, d->base_vertex, bi, xf)) vkCmdDrawIndexed(cb, (uint32_t)d->count, inst, 0, d->base_vertex, bi);
        drawprof_end(cb, dp);
        for (int r = r131_draw(1, (uint32_t)d->count, inst, xf); r > 0; r--) vkCmdDrawIndexed(cb, (uint32_t)d->count, inst, 0, d->base_vertex, bi);   // r131 DM_VK_X_DUP (live, diagnostic)
        if (xf) xfb_end(cb, prim, (uint32_t)d->count, inst); }
}
static nvn_hmap g_cpipes = {.klen = 24};   // {shader, its integer-texel bindings}
// Donkey Kong Bananza's NDSR upscaler (agl_ndsr.pck, 24 compute variants, 11-04 r23): a neural network on the Switch 2's tensor cores, translated with HMMA emulated
// in shared memory (dmsh_hmma16816_f16), up to 37 ms a dispatch on Turnip. Replaced by a bilinear blit of its sampled input onto its storage image output, as Eglantine
// performance-4 does in kepler_compute.cpp UpscaleInsteadOfNativeProgram. DM_VK_NDSR=1 runs the network. Cost when off: a 24-key scan per dispatch (~70 dispatches a frame)
static const uint64_t k_ndsr[] = {
    0x131d7dcc054f92b9ull, 0xc56db3d5a0f1b5afull, 0x7629d857a0a93833ull, 0xeb0af4c68c522092ull,
    0x75e2ecd80c1e9fb8ull, 0x5e0110401b4a3d8full, 0x0e351bd8ef324466ull, 0x7a8dcb28fa38987dull,
    0x3b716012fcd10575ull, 0x446761e211ec30d8ull, 0x5c5876712b1ebd78ull, 0x9e5597e2fcfc3c2full,
    0x8df4d4ba8fa94292ull, 0x06946f57dd8833c3ull, 0xa9fed0568668bf97ull, 0x1c144a902af468deull,
    0x7ae23d53ffb1c2ffull, 0x14a8aaf3e0a9edbdull, 0xc8910045dc5b59afull, 0x1a4c427f99c58569ull,
    0xf8a6a2e42fe156a6ull, 0xd19ea940b265456full, 0xf585ba048398f697ull, 0xf13b72178b0d5e9full};
static void copy_t2t(const p_copy_t2t *c);
static void clear_tex(const p_clear_tex *c);
// r29 (11-04): Snapdragon GSR 1 in place of the NDSR network (shaders/gsr1.comp, BSD-3-Clause): one compute pass, input rectangle -> output rectangle
#include "gsr1_comp.spv.h"
static struct { VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline pipe; VkSampler samp; int state;   // state 0 not made, 1 ready, -1 failed
                VkImage timg; VkDeviceMemory tmem; VkImageView tview; uint32_t tw, th; int tnew; } g_gsr;   // r29b: GSR1 writes this RGBA16F image, copied to the output (r29: Turnip gave the R11G11B10F output no storage usage)
static int gsr_tmp(uint32_t w, uint32_t h) {   // the RGBA16F image at the output's size (made again when that changes)
    if (g_gsr.timg && g_gsr.tw == w && g_gsr.th == h) return 1;
    if (g_gsr.timg) { vkDeviceWaitIdle(V.dev); vkDestroyImageView(V.dev, g_gsr.tview, NULL); vkDestroyImage(V.dev, g_gsr.timg, NULL); vkFreeMemory(V.dev, g_gsr.tmem, NULL); g_gsr.timg = VK_NULL_HANDLE; }
    VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT, {w, h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    if (vkCreateImage(V.dev, &ic, NULL, &g_gsr.timg)) { g_gsr.timg = VK_NULL_HANDLE; VKRED_ONCE(98, "GSR1: its %ux%u image failed: blit instead", w, h); return 0; }
    VkMemoryRequirements rq; vkGetImageMemoryRequirements(V.dev, g_gsr.timg, &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, g_gsr.timg, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkAllocateMemory(V.dev, &ai, NULL, &g_gsr.tmem) || vkBindImageMemory(V.dev, g_gsr.timg, g_gsr.tmem, 0) || (vi.image = g_gsr.timg, vkCreateImageView(V.dev, &vi, NULL, &g_gsr.tview))) {
        vkDestroyImage(V.dev, g_gsr.timg, NULL); g_gsr.timg = VK_NULL_HANDLE; VKRED_ONCE(98, "GSR1: its %ux%u image failed: blit instead", w, h); return 0; }
    g_gsr.tw = w; g_gsr.th = h; g_gsr.tnew = 1; VKLOG("GSR1: %ux%u RGBA16F image made", w, h); return 1;
}
typedef struct { float vp[4], src[4]; int32_t dst[4]; float tune[4], tune2[4]; } gsr_pc;   // tune: edge threshold, edge sharpness (r31)
static int gsr_ready(void) {
    if (g_gsr.state) return g_gsr.state > 0;
    g_gsr.state = -1;
    VkDescriptorSetLayoutBinding b[2] = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 2, b};
    if (vkCreateDescriptorSetLayout(V.dev, &li, NULL, &g_gsr.dsl)) { VKRED_ONCE(96, "GSR1: descriptor set layout failed: blit instead"); return 0; }
    void vk_dsl_register(VkDescriptorSetLayout l, const VkDescriptorSetLayoutBinding *b, uint32_t n); vk_dsl_register(g_gsr.dsl, b, 2);
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gsr_pc)};
    VkPipelineLayoutCreateInfo pi = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_gsr.dsl, 1, &pr};
    if (vkCreatePipelineLayout(V.dev, &pi, NULL, &g_gsr.pl)) { VKRED_ONCE(96, "GSR1: pipeline layout failed: blit instead"); return 0; }
    VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, NULL, 0, VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    if (vkCreateSampler(V.dev, &si, NULL, &g_gsr.samp)) { VKRED_ONCE(96, "GSR1: sampler failed: blit instead"); return 0; }
    VkShaderModule m; VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof gsr1_comp_spv, gsr1_comp_spv};
    if (vkCreateShaderModule(V.dev, &mi, NULL, &m)) { VKRED_ONCE(96, "GSR1: shader module failed: blit instead"); return 0; }
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", NULL}, g_gsr.pl, VK_NULL_HANDLE, -1};
    VkResult r = vkCreateComputePipelines(V.dev, V.pcache, 1, &ci, NULL, &g_gsr.pipe); vkDestroyShaderModule(V.dev, m, NULL);
    if (r) { VKRED_ONCE(96, "GSR1: compute pipeline failed (%d): blit instead", r); return 0; }
    g_gsr.state = 1; VKLOG("GSR1 ready (Snapdragon Game Super Resolution 1 replaces agl's NDSR network; DM_VK_NDSR=1 network, =3 bilinear blit)"); return 1;
}
static int gsr_run(const tex_t *in, const tex_t *out, const uint32_t *pw) {   // 0: not possible here, the caller blits
    if (!gsr_ready()) return 0;
    vk_end_pass();   // (as copy_t2t: before the lookups, which may record uploads)
    vkimg_t *si = NULL, *di = vk_tex(out); if (di) vk_tex_gpu_mark(out);   // r72
    VkImageView sv = vk_tex_view(in, NULL, 1, 0, 0, &si), dv;
    VkFormatProperties dp = {0}; if (di) vkGetPhysicalDeviceFormatProperties(V.pd, di->fmt, &dp);
    if (!sv || !si || !di || !(di->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) || !(dp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) || di->samples > 1) {
        VKRED_ONCE(97, "GSR1: the output cannot take a copy (format %d, usage %#x): blit instead", di ? di->fmt : 0, di ? di->usage : 0); return 0; }
    if (!gsr_tmp(di->w, di->h)) return 0;
    dv = g_gsr.tview;
    VkCommandBuffer cb = vk_cmd(); wb_use(si, WB_COPY); wb_use(di, WB_OTHER); vk_tex_prepare(cb, si); vk_tex_prepare(cb, di); vk_tex_content_changed(di); di->gpu_w = 1;
    if (g_gsr.tnew) { g_gsr.tnew = 0;
        VkImageMemoryBarrier ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                   VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, g_gsr.timg, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &ib); }
    vk_img_barrier_all(cb);
    VkDescriptorSet set = vk_alloc_set(g_gsr.dsl);
    VkDescriptorImageInfo ii[2] = {{g_gsr.samp, sv, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, dv, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet wr[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii[0], NULL, NULL},
                                  {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ii[1], NULL, NULL}};
    vkUpdateDescriptorSets(V.dev, 2, wr, 0, NULL);
    gsr_pc pc = {{1.0f / (float)in->d.width, 1.0f / (float)in->d.height, (float)in->d.width, (float)in->d.height}, {(float)pw[0], (float)pw[1], (float)(pw[2] - pw[0] + 1), (float)(pw[3] - pw[1] + 1)},
                 {(int32_t)pw[4], (int32_t)pw[5], (int32_t)(pw[6] - pw[4] + 1), (int32_t)(pw[7] - pw[5] + 1)}, {0}};
    static float thr = -1, shp = -1, bri = 1, sat = 1, con = 1, srgb = -1;   // r31: DM_GSR_THRESHOLD (0-255, default 8: lower treats more of the picture as edges) and DM_GSR_SHARPNESS (default 2.0: higher, harder edges)
    static long gen = -1;   // r54: read again after an environment change (live through ctl.txt "env" lines); DM_GSR_BRIGHTNESS (default 1.0, lower = darker middle tones), DM_GSR_SATURATION (default 1.0, lower = paler), DM_GSR_CONTRAST (default 1.0, lower = flatter)
    // 11-16: DM_GSR_SRGB_OUT=1 brings back r51's sRGB-to-linear step on the output (tune2.y; default 0: the value as read, as DM_VK_NDSR=3's copy, which NPARK saw as right)
    if (gen != dm_env_gen) { gen = dm_env_gen; const char *e = getenv("DM_GSR_THRESHOLD"), *f = getenv("DM_GSR_SHARPNESS"), *b = getenv("DM_GSR_BRIGHTNESS"), *c = getenv("DM_GSR_SATURATION"), *k = getenv("DM_GSR_CONTRAST"), *g = getenv("DM_GSR_SRGB_OUT");
        float t = e && atof(e) >= 0 ? (float)atof(e) : 8.0f, h = f && atof(f) > 0 ? (float)atof(f) : 2.0f, br = b && atof(b) > 0 ? (float)atof(b) : 1.0f, sa = c && atof(c) >= 0 ? (float)atof(c) : 1.0f, co = k && atof(k) > 0 ? (float)atof(k) : 1.0f, sr = g && atoi(g) ? 1.0f : 0.0f;
        if (t != thr || h != shp || br != bri || sa != sat || co != con || sr != srgb) VKLOG("GSR1 edge threshold %.1f / 255, edge sharpness %.2f, brightness %.2f, saturation %.2f, contrast %.2f, sRGB-to-linear output %s (DM_GSR_SRGB_OUT=1 turns it on)", t, h, br, sa, co, sr > 0.5f ? "on" : "off");
        thr = t; shp = h; bri = br; sat = sa; con = co; srgb = sr; }
    pc.tune[0] = thr / 255.0f; pc.tune[1] = shp; pc.tune[2] = 1.0f / bri; pc.tune[3] = sat; pc.tune2[0] = con; pc.tune2[1] = srgb;
    uint32_t gx = ((uint32_t)pc.dst[2] + 7) / 8, gy = ((uint32_t)pc.dst[3] + 7) / 8;
    void vk_pass_ts(int end, uint32_t w, uint32_t h, uint32_t fmt, uint32_t draws);
    vk_pass_ts(0, 0, 0, 0, 0);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_gsr.pipe);   // (dispatch binds its own pipeline and set every time)
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_gsr.pl, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(cb, g_gsr.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cb, gx, gy, 1);
    vk_img_barrier_all(cb);
    VkImageBlit r = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{pc.dst[0], pc.dst[1], 0}, {pc.dst[0] + pc.dst[2], pc.dst[1] + pc.dst[3], 1}},
                     {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{pc.dst[0], pc.dst[1], 0}, {pc.dst[0] + pc.dst[2], pc.dst[1] + pc.dst[3], 1}}};
    vkCmdBlitImage(cb, g_gsr.timg, VK_IMAGE_LAYOUT_GENERAL, di->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r, VK_FILTER_NEAREST);   // same size: a format conversion
    vk_pass_ts(1, gx, gy, 0xC0FFEEu, 0x65520001u);   // the pass report shows it (pass and copy) as dispatch ..65520001
    vk_img_barrier_all(cb);
    return 1;
}
#if defined(DM_NXN1X)
static void fg_tag_guest_inputs(const p_dlss *p) {
    if (!nvn_fg_collecting() || p->iw <= 0 || p->ih <= 0 ||
        (p->view_info & ~0x3fu) != DLSS_VIEW_INFO ||
        (p->view_info & ((1u << DLSS_DEPTH) | (1u << DLSS_MOTION)))) return;
    nvn_fg_image images[2];
    vk_end_pass_land();
    materialize_scale(NULL);
    VkCommandBuffer cb = vk_cmd();
    for (int i = 0; i < 2; ++i) {
        int k = i ? DLSS_MOTION : DLSS_DEPTH;
        const tex_t *tex = p->tex[k];
        if (!tex) return;
        vkimg_t *im = NULL;
        VkImageView view = vk_tex_view(tex, NULL, 1, 0, VK_VIEW_SAMPLED, &im);
        if (!view || !im || im->samples != 1 || im->layers != 1 || im->levels != 1 ||
            im->vtype != VK_IMAGE_VIEW_TYPE_2D || !(im->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ||
            p->base[k][0] < 0 || p->base[k][1] < 0 ||
            (uint64_t)p->base[k][0] + p->iw > im->w ||
            (uint64_t)p->base[k][1] + p->ih > im->h) return;
        vk_tex_prepare(cb, im);
        im->last_use = mtl_last_fence() + 1;
        images[i] = (nvn_fg_image){im->img, im->mem, view, im->w, im->h, im->fmt,
            im->layout, im->usage, im->cflags, (uint32_t)p->base[k][0], (uint32_t)p->base[k][1]};
    }
    vk_img_barrier_all(cb);
    nvn_fg_tag(cb, &images[0], &images[1], (uint32_t)p->iw, (uint32_t)p->ih, p->mv, p->jitter, p->reset, 1);
    vk_img_barrier_all(cb);
}
static void replay_dlss(const p_dlss *record) {
    p_dlss params = *record;
    const p_dlss *p = &params;
    if (p->tex[DLSS_COLOR]) {
        if (!p->iw) params.iw = (int)p->tex[DLSS_COLOR]->d.width - p->base[DLSS_COLOR][0];
        if (!p->ih) params.ih = (int)p->tex[DLSS_COLOR]->d.height - p->base[DLSS_COLOR][1];
    }
    static uint64_t calls;
    const uint64_t call = ++calls;
    const char *audit = getenv("NXN1X_DLSS_AUDIT_FRAME");
    const int report = call <= 4 || call % 300 == 0 || (audit && strtoull(audit, NULL, 10) == g_tex_frame);
    vk_end_pass_land();
    materialize_scale(NULL);
    fg_tag_guest_inputs(p);
    nvn_fg_image resources[DLSS_TEXTURES] = {0};
    vkimg_t *images[DLSS_TEXTURES] = {0};
    VkCommandBuffer cb = vk_cmd();
    int eligible = 1;
    for (unsigned i = 0; i < DLSS_TEXTURES; ++i) {
        if (!p->tex[i]) { if (i <= DLSS_OUTPUT) eligible = 0; continue; }
        const int output = i == DLSS_OUTPUT;
        int w = output ? p->ow : i == DLSS_EXPOSURE ? 1 : p->iw;
        int h = output ? p->oh : i == DLSS_EXPOSURE ? 1 : p->ih;
        if (p->view_info & (1u << i)) { eligible = 0; continue; }
        vkimg_t *im = NULL;
        VkImageView view = vk_tex_view(p->tex[i], NULL, 1, 0, output ? VK_VIEW_STORAGE : VK_VIEW_SAMPLED, &im);
        if (!im || im->samples != 1 || im->vtype != VK_IMAGE_VIEW_TYPE_2D ||
            im->levels != 1 || im->layers != 1 || w <= 0 || h <= 0 ||
            p->base[i][0] < 0 || p->base[i][1] < 0 ||
            (uint64_t)p->base[i][0] + w > im->w || (uint64_t)p->base[i][1] + h > im->h ||
            ((output || i == DLSS_COLOR) && im->aspect != VK_IMAGE_ASPECT_COLOR_BIT)) {
            eligible = 0; continue;
        }
        images[i] = im;
        wb_use(im, output ? WB_OTHER : WB_COPY);
        vk_tex_prepare(cb, im);
        im->last_use = mtl_last_fence() + 1;
        if (!view || !(im->usage & (output ? VK_IMAGE_USAGE_STORAGE_BIT : VK_IMAGE_USAGE_SAMPLED_BIT))) eligible = 0;
        resources[i] = (nvn_fg_image){im->img, im->mem, view, im->w, im->h, im->fmt,
            im->layout, im->usage, im->cflags, p->base[i][0], p->base[i][1]};
    }
    vkimg_t *in = images[DLSS_COLOR], *out = images[DLSS_OUTPUT];
    if (!in || !out) {
        if (report) VKLOG("OP_DLSS frame %llu unsupported color/output: input=%p output=%p view=%#x rect=%dx%d -> %dx%d",
            (unsigned long long)g_tex_frame, (void*)in, (void*)out, p->view_info, p->iw, p->ih, p->ow, p->oh);
        return;
    }
    for (unsigned i = 0; i < DLSS_TEXTURES; ++i)
        if (i != DLSS_OUTPUT && images[i] && images[i]->img == out->img) eligible = 0;
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL,
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
    const int sample = report && !sr_probes[0].b && !sr_probes[1].b;
    int a = sample && (in->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) && pd_capture_region(in, 5, &sr_probes[0],
        p->base[DLSS_COLOR][0], p->base[DLSS_COLOR][1], p->iw, p->ih);
    int result = eligible ? nvn_sr_evaluate(cb, resources, p->iw, p->ih, p->ow, p->oh,
        p->mv, p->jitter, p->pre_exposure, p->flags, p->reset, p->net) : -3;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
    if (result == 0) {
        vk_tex_content_changed(out);
        out->gpu_w = 1;
        vk_tex_gpu_mark(p->tex[DLSS_OUTPUT]);
    } else {
        p_copy_t2t copy = {.src = p->tex[DLSS_COLOR], .dst = p->tex[DLSS_OUTPUT], .flags = 1};
        copy.sr = (NVNcopyRegion){p->base[DLSS_COLOR][0], p->base[DLSS_COLOR][1], 0, p->iw, p->ih, 1};
        copy.dr = (NVNcopyRegion){p->base[DLSS_OUTPUT][0], p->base[DLSS_OUTPUT][1], 0, p->ow, p->oh, 1};
        copy_t2t(&copy);
        out = vk_tex(copy.dst);
    }
    int b = sample && out && (out->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) && pd_capture_region(out, 6, &sr_probes[1],
        p->base[DLSS_OUTPUT][0], p->base[DLSS_OUTPUT][1], p->ow, p->oh);
    if (report) VKLOG("OP_DLSS frame %llu call %llu %dx%d -> %dx%d path=%s sr_result=%d probes=%d/%d",
        (unsigned long long)g_tex_frame, (unsigned long long)call, p->iw, p->ih, p->ow, p->oh,
        result ? "spatial-copy" : "NVIDIA-DLSS-SR", result, a, b);
}

#endif
static unsigned char ndsr_trace_state[sizeof k_ndsr / sizeof *k_ndsr];
static int ndsr_trace_slot = -1;
static void ndsr_trace_dispatch(unsigned char state, const char *stage) {
    if (ndsr_trace_slot < 0 || (ndsr_trace_state[ndsr_trace_slot] & state)) return;
    ndsr_trace_state[ndsr_trace_slot] |= state;
    VKLOG("NDSR original: shader=%016llx stage=%s frame=%llu groups=%ux%ux%u",
          (unsigned long long)k_ndsr[ndsr_trace_slot], stage, (unsigned long long)g_tex_frame,
          g_dispatch[0], g_dispatch[1], g_dispatch[2]);
}
static int ndsr_blit(const vkshader_t *cs) {
    static int mode = -1;
    static int trace_native;
    if (mode < 0) {
        const char *override = getenv("DM_VK_NDSR");
        mode = override ? atoi(override) : 1;
        trace_native = mode == 1 && getenv("NXN1X_DLSS_AUDIT_FRAME") != NULL;
        VKLOG("NDSR policy: %s (%s)", mode == 1 ? "original guest compute" :
              mode == 0 ? "GSR1" : "diagnostic override", override ? "override" : "device default");
    }
    ndsr_trace_slot = -1;
    if (mode == 1 && !trace_native
#if defined(DM_NXN1X)
        && !fg_frames_left
#endif
    ) return 0;
    size_t slot = 0;
    while (slot < sizeof k_ndsr / sizeof *k_ndsr && k_ndsr[slot] != cs->key) ++slot;
    if (slot == sizeof k_ndsr / sizeof *k_ndsr) return 0;
    if (mode == 1) {
        ndsr_trace_slot = (int)slot;
        ndsr_trace_dispatch(1, "selected");
        return 0;
    }
    // NVN2 diagnostic: DM_VK_PASSDUMP_AT_NDSR=N starts the pass dump (DM_VK_PASSDUMP=N: after N - 1 presents) at the first NDSR dispatch: Donkey Kong Bananza's play scene
    // is right for about 2 s from there, then black
    { static int at = -1; if (at < 0) { const char *e = getenv("DM_VK_PASSDUMP_AT_NDSR"); at = e ? atoi(e) : 0; }
      if (at > 0) { char n[16]; snprintf(n, sizeof n, "%d", at); setenv("DM_VK_PASSDUMP", n, 1); VKLOG("NDSR %016llx: first dispatch, the pass dump starts (DM_VK_PASSDUMP=%d)", (unsigned long long)cs->key, at); at = 0; } }
    const tex_t *in = NULL, *out = NULL; const view_t *v; uint64_t hin = 0, hout = 0;   // the first sampled texture and the first storage image, their handles as stage_set takes them
    tex_t tin, tout;   // r28: vk_tex_by_id returns its thread's one copy, which the next lookup overwrites (r26/r27 blitted the output onto itself: purple)
    for (int i = 0; i < cs->nbind; i++) { const dmsv_bind *b = &cs->b[i]; const tex_t *t = NULL; uint64_t h = 0;
        if (!in && (b->kind == RK_TEX || b->kind == RK_TEX_UBO || b->kind == RK_TEX_C2A)) {
            h = b->kind == RK_TEX ? (b->reg >= 0 && b->reg < 256 ? S.tex[5][b->reg] : 0) : b->kind == RK_TEX_UBO ? ubo_handle(5, b->reg, b->off) : (b->off >= 0 ? c2_handle(5, b->off) : 0);
            if ((h & 0xfffff) && vk_tex_by_id((int)(h & 0xfffff), &v, &t) && t) { tin = *t; in = &tin; hin = h; } }
        else if (!out && (b->kind == RK_IMG || b->kind == RK_IMG_UBO)) {
            h = b->kind == RK_IMG ? (b->reg >= 0 && b->reg < 32 ? S.img[5][b->reg] : 0) : ubo_handle(5, b->reg, b->off);
            if ((h & 0xfffff) && vk_tex_by_id((int)(h & 0xfffff), &v, &t) && t) { tout = *t; out = &tout; hout = h; } } }
    // NVN2 (Switch 2 boots) in Eden mode: the shader has no binding list of its own (vk_shader: bindings come with the Eden pipeline), so the scan above finds nothing and the
    // network ran instead (Donkey Kong Bananza in NXN1X: a purple screen, the output's clear colour, as on the host before r28). Without asking Eden for the pipeline (it
    // would build the 24 HMMA pipelines the replacement exists to avoid): the shader's slots from its control section (nvn_eden_shader_nvn2_c0: texture and image slot
    // counts) and their bound handles where nvn2_cs_c0 takes them for its c[0] (c2 + 0x20 per texture slot, c2 + 0x120 per image slot). The input is the first texture slot
    // that names a texture; the output the first image slot that names one the output rectangle fits (r24: DK's NDSR reads image slot 1). Copied as r28 copies
    if (!in && !out && cs->eden && dm_sdk_ounce()) {
        uint32_t L[8]; const uint32_t *pr = (const uint32_t *)(const void *)S.push[1];
        if (nvn_eden_shader_nvn2_c0(cs->eden, L)) {
            for (uint32_t r = 0; r < L[3] && r < 32 && !in; r++) { const tex_t *t = NULL; uint64_t h = c2_handle(5, (int)(0x20 + 8 * r));
                if ((h & 0xfffff) && vk_tex_by_id((int)(h & 0xfffff), &v, &t) && t) { tin = *t; in = &tin; hin = h; } }
            for (uint32_t r = 0; r < L[6] && r < 8 && !out; r++) { const tex_t *t = NULL; uint64_t h = c2_handle(5, (int)(0x120 + 8 * r));
                if ((h & 0xfffff) && vk_tex_by_id((int)(h & 0xfffff), &v, &t) && t && pr[6] < t->d.width && pr[7] < t->d.height) { tout = *t; out = &tout; hout = h; } }
            // NVN2 diagnostic (DK black 3D): the first 3 dispatches of each key log every slot that names a texture, with its host image (gpu_w, the present it was last
            // rendered at as an attachment) and which ones were picked: is the first texture slot the image the scene was rendered into?
            static unsigned char sl[sizeof k_ndsr / sizeof *k_ndsr]; size_t kj = 0; while (k_ndsr[kj] != cs->key) kj++;
            if (sl[kj] < 3) { sl[kj]++; extern uint64_t g_tex_frame;
                for (int im = 0; im < 2; im++) for (uint32_t r = 0; r < (im ? L[6] : L[3]) && r < (im ? 8u : 32u); r++) {
                    const tex_t *t = NULL; const uint64_t h = c2_handle(5, (int)((im ? 0x120 : 0x20) + 8 * r));
                    if (!(h & 0xfffff) || !vk_tex_by_id((int)(h & 0xfffff), &v, &t) || !t) continue;
                    const tex_t tc = *t; const vkimg_t *hi = vk_tex_peek(&tc);
                    VKLOG("NDSR %016llx dispatch %u (present %llu): %s slot %u handle %#llx %ux%u levels %u fmt %#x pool %p%+lld; host image %s, gpu_w %d, rendered at present %lld%s",
                          (unsigned long long)cs->key, (unsigned)sl[kj], (unsigned long long)g_tex_frame, im ? "image" : "texture", r, (unsigned long long)h, tc.d.width, tc.d.height,
                          (unsigned)tc.d.levels, (unsigned)tc.d.format, (const void *)tc.pool, (long long)tc.off, hi ? "yes" : "none", hi ? hi->gpu_w : 0, hi ? (long long)hi->wpres - 1 : -1LL,
                          !im && in && h == hin ? " <- picked as input" : im && out && h == hout ? " <- picked as output" : "");
                }
            }
        }
    }
    // r26 (11-04): the push block (c24) words 0-7 are the input and output rectangles, min and max inclusive (r25 log: 0,0,1279,719 -> 0,0,1919,1079 in 1920x1080 textures)
    const uint32_t *pw = (const uint32_t *)(const void *)S.push[1];
    int ok = in && out && pw[2] >= pw[0] && pw[3] >= pw[1] && pw[6] >= pw[4] && pw[7] >= pw[5] && pw[2] < in->d.width && pw[3] < in->d.height && pw[6] < out->d.width && pw[7] < out->d.height
             && pw[6] - pw[4] >= pw[2] - pw[0] && pw[7] - pw[5] >= pw[3] - pw[1];
    static unsigned char logged[sizeof k_ndsr / sizeof *k_ndsr]; size_t ki = 0; while (k_ndsr[ki] != cs->key) ki++;
    if (logged[ki] < 2 || (!ok && logged[ki] < 4)) { logged[ki]++;
        VKLOG("NDSR %016llx: %s; input %#llx %ux%u format %#x rect %u,%u-%u,%u, output %#llx %ux%u format %#x rect %u,%u-%u,%u; dispatch %ux%ux%u", (unsigned long long)cs->key,
              ok ? (mode == 3 ? "bilinear blit instead of the network" : "GSR1 instead of the network") : "rectangles not usable, the network runs", (unsigned long long)hin, in ? in->d.width : 0, in ? in->d.height : 0, in ? (unsigned)in->d.format : 0, pw[0], pw[1], pw[2], pw[3],
              (unsigned long long)hout, out ? out->d.width : 0, out ? out->d.height : 0, out ? (unsigned)out->d.format : 0, pw[4], pw[5], pw[6], pw[7], g_dispatch[0], g_dispatch[1], g_dispatch[2]); }
    if (!ok) return 0;
    // NVN2 diagnostic (DM_VK_PASSDUMP_SIZE=ndsr): the input's size for the pass filter; at each sampled present of a dump, the input itself (once a present, before the blit or GSR1)
    g_pd_ndsr_w = in->d.width; g_pd_ndsr_h = in->d.height; if (g_pd_stream) g_pd_ndsr_out = vk_tex(out);
    { static uint64_t lk; static int nk; extern uint64_t g_tex_frame;   // NVN2 diagnostic: each NDSR key change (the dynamic resolution), to match the black onset against
      if (cs->key != lk && nk < 64) { nk++; VKLOG("NDSR key %016llx -> %016llx at present %llu (input %ux%u, output %ux%u)", (unsigned long long)lk, (unsigned long long)cs->key,
                                                 (unsigned long long)g_tex_frame, in->d.width, in->d.height, out->d.width, out->d.height); }
      lk = cs->key; }
    const int pd_take = g_pd_state == 1 && g_pd_stream && !(g_pd_every > 1 && g_pd_tick % g_pd_every) && g_pd_ndsr_tick != g_pd_tick;   // this present's input and output (kinds 1 and 2)
    if (pd_take) { g_pd_ndsr_tick = g_pd_tick; pd_capture_img(vk_tex(in), 1); }
    if (mode == 2) { p_clear_tex k; memset(&k, 0, sizeof k); k.tex = out; k.c.f[1] = 1; k.c.f[3] = 1; k.r = (NVNcopyRegion){0, 0, 0, (int)out->d.width, (int)out->d.height, 1}; k.mask = 0xf; clear_tex(&k); return 1; }   // DM_VK_NDSR=2: green, to see whether a write to the output reaches the screen
    if (mode != 3 && gsr_run(in, out, pw)) { if (pd_take) pd_capture_img(vk_tex(out), 2); return 1; }
    p_copy_t2t c; memset(&c, 0, sizeof c); c.src = in; c.dst = out; c.flags = 1;
    c.sr = (NVNcopyRegion){(int)pw[0], (int)pw[1], 0, (int)(pw[2] - pw[0] + 1), (int)(pw[3] - pw[1] + 1), 1}; c.dr = (NVNcopyRegion){(int)pw[4], (int)pw[5], 0, (int)(pw[6] - pw[4] + 1), (int)(pw[7] - pw[5] + 1), 1};
    copy_t2t(&c); if (pd_take) pd_capture_img(vk_tex(out), 2); return 1;
}
void vk_pass_ts(int end, uint32_t w, uint32_t h, uint32_t fmt, uint32_t draws);
static uint64_t g_disp_seq = ~0ull, g_disp_cbar; static VkCommandBuffer g_disp_cb;   // r89: g_bar_seq after the last dispatch's barrier, in command buffer g_disp_cb
// cbar1 (11-21, NPARK 06:54): a guest barrier (nvnCommandBufferBarrier) between two dispatches. Cause: OoT records DISPATCH, BARRIER, (binds), DISPATCH about 25 times a
// submission (ORDER_PRIMITIVES / ORDER_FRAGMENTS / INVALIDATE_TEXTURE ...). OP_BARRIER recorded a full barrier (vk_img_barrier_all), which also bumped g_bar_seq, so r89's
// compute-only barrier before the next dispatch did not apply and it got a second full one: 2 full barriers in 78.8 gaps a frame, 5.36 ms of GPU gap a frame
// (ab_import_002257 ootgap). Between two dispatches the guest barrier orders nothing those two barriers do not: the first dispatch's own barrier after it (r89:
// COMPUTE_SHADER/SHADER_WRITE -> ALL_COMMANDS with every access kind) already makes its writes visible to all later work and holds later work behind it, and the next
// dispatch's barrier before it (COMPUTE|TRANSFER -> COMPUTE|TRANSFER, plus DRAW_INDIRECT/INDIRECT_COMMAND_READ when indirect) orders what was recorded between. So the
// guest barrier is left out when (1) the last GPU work in this command buffer is a dispatch with its r89 barrier and nothing since (no full barrier: g_disp_seq ==
// g_bar_seq), and (2) the next work record is a dispatch: the lookahead below, over records that record no GPU work. Anything else (a draw, a pass, a copy, a fill, a
// clear, a counter report, an uniform update, a call, an unknown record, the list's end) keeps the full barrier. DM_VK_CBAR=0: off (A/B)
#define g_cbar_skip (g_rec->pc->cbar_skip)   /* S1c (parallel recording): rec_pcnt_t (were statics) */
#define g_cbar_kept (g_rec->pc->cbar_kept)
extern _Thread_local uint64_t g_bar_seq; int vk_bar_mode(void); int vk_pass_times_active(void); uint64_t vk_bmerge(int k);
static int cbar_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_CBAR"); v = !(e && *e == '0');
    VKLOG("cbar1: a guest barrier between two dispatches %s (DM_VK_CBAR=%s)", v ? "is left to r89's barriers after and before them" : "is a full barrier (as before)", e && *e ? e : "1 (default)"); } return v; }
// cbar2 counter: the guest barriers kept full, by the work before them (0 a dispatch with nothing since, 1 a pass or anything else) and the next work record
// (0 dispatch, 1 draw / pass / clear, 2 copy / fill, 3 counter report / uniform update, 4 call / end / other)
#define g_cbk (g_rec->pc->cbk)   // S1c (parallel recording): rec_pcnt_t (was a static)
static int cbk_next(const uint8_t *q) {
    for (int i = 0; q && i < 4096; i++) {
        op_hdr h; memcpy(&h, q, sizeof h);
        if (h.op >= OP_COUNT_ || h.size < 8 || (h.size & 7) || h.size > (1u << 20)) return 4;
        switch (h.op) {
        case OP_JUMP: { const uint8_t *t; memcpy(&t, q + 8, 8); if (!dm_nvn_jump_ok(t)) return 4; q = t; continue; }   // jumpchk
        case OP_DISPATCH: return 0;
        case OP_DRAW: case OP_DRAW_IDX: case OP_MULTI_DRAW: case OP_DRAW_TEX: case OP_SET_RT: case OP_CLEAR_COLOR: case OP_CLEAR_DS: case OP_CLEAR_TEX: return 1;
        case OP_COPY_B2B: case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T: case OP_CLEAR_BUF: return 2;
        case OP_REPORT_COUNTER: case OP_RESET_COUNTER: case OP_UPDATE_UB: return 3;
        case OP_BARRIER: case OP_END: case OP_CALL: return 4;
        default: break;   // state
        }
        q += h.size;
    }
    return 4;
}
// 1 = the next record that records GPU work, from q on, is a dispatch (records that set state only are passed; OP_JUMP followed)
int vk_cbar_next_dispatch(const uint8_t *q) {
    for (int i = 0; q && i < 4096; i++) {
        op_hdr h; memcpy(&h, q, sizeof h);
        if (h.op >= OP_COUNT_ || h.size < 8 || (h.size & 7) || h.size > (1u << 20)) return 0;
        switch (h.op) {
        case OP_JUMP: { const uint8_t *t; memcpy(&t, q + 8, 8); if (!dm_nvn_jump_ok(t)) return 0; q = t; continue; }   // jumpchk
        case OP_DISPATCH: return 1;
        case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF: case OP_BIND_VBUF1: case OP_BIND_UBO: case OP_BIND_UBO_SMALL:
        case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_TEX1: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: case OP_PUSH_CONST:
        case OP_SIGNAL_EVENT:   // (set by the completion thread when the whole submission has completed: no GPU order)
            break;
        default: return 0;
        }
        q += h.size;
    }
    return 0;
}
// cbar1 self-test (gfx_vk_core.c vk_poolcopy_selftest, cases 170-177): the lookahead over record lists made here. 1 pass, 0 fail, -1 no such case
int vk_cbar_case(int k, const char **what) {
    static uint64_t L[64], J[8]; uint8_t *q = (uint8_t *)L; memset(L, 0, sizeof L); memset(J, 0, sizeof J);
#define REC(op_, sz_) do { op_hdr h_ = {(uint32_t)(op_), (uint32_t)(sz_)}; memcpy(q, &h_, 8); q += (sz_); } while (0)
    int want;
    switch (k) {
    case 0: *what = "binds then a dispatch: left out"; REC(OP_BIND_PROGRAM, 16); REC(OP_BIND_SSBO, 24); REC(OP_BIND_UBO_SMALL, 24); REC(OP_DISPATCH, 40); want = 1; break;
    case 1: *what = "a draw next: kept"; REC(OP_BIND_SSBO, 24); REC(OP_DRAW, 32); want = 0; break;
    case 2: *what = "a buffer copy before the dispatch: kept"; REC(OP_COPY_B2B, 32); REC(OP_DISPATCH, 40); want = 0; break;
    case 3: *what = "a counter report before the dispatch: kept"; REC(OP_REPORT_COUNTER, 24); REC(OP_DISPATCH, 40); want = 0; break;
    case 4: *what = "an uniform update before the dispatch: kept"; REC(OP_UPDATE_UB, 24); REC(OP_DISPATCH, 40); want = 0; break;
    case 5: *what = "the list's end: kept"; REC(OP_BIND_SSBO, 24); REC(OP_END, 8); want = 0; break;
    case 6: *what = "a jump to a dispatch: left out"; { op_hdr h = {OP_JUMP, 16}; memcpy(q, &h, 8); uint8_t *t = (uint8_t *)J; memcpy(q + 8, &t, 8); q += 16;
        op_hdr d = {OP_DISPATCH, 40}; memcpy(t, &d, 8); } want = 1; break;
    case 7: *what = "a call (another list) before the dispatch: kept"; REC(OP_CALL, 16); REC(OP_DISPATCH, 40); want = 0; break;
    default: return -1;
    }
#undef REC
    return vk_cbar_next_dispatch((const uint8_t *)L) == want;
}
// r140 (diagnostic): DM_VK_TRACE_CS=<hex key>[,...] (live; a key matches by its full value or its low 32 bits): each such dispatch logs its group counts, the compute
// push constant block (c24, words 0-15; NVN 2 root table 0: Hyrule Warriors: Age of Imprisonment's upscaler takes its input and output rectangles there, which change
// with the dynamic resolution), its first constant buffer and the graphics viewport at the time. The first 24 of each key, then every 120th
__declspec(dllimport) int IsBadReadPtr(const void *p, unsigned long long n);   // r140 needs it (no windows.h here; latest declares it for r123): added with r141, build failed without it (-Werror=implicit-function-declaration)
static void r140_cs(const vkshader_t *cs, const p_dispatch *d) {
    static long gen = -1; static uint64_t k[8]; static int nk; static unsigned n[8];
    if (gen != dm_env_gen) { gen = dm_env_gen; nk = 0; const char *e = getenv("DM_VK_TRACE_CS");
        for (const char *q = e; q && *q && nk < 8;) { char *x; uint64_t v = strtoull(q, &x, 16); if (x == q) break; if (v) { k[nk] = v; n[nk++] = 0; } q = *x ? x + 1 : x; } }
    for (int i = 0; i < nk; i++) if (cs->key == k[i] || (k[i] >> 32 == 0 && (uint32_t)cs->key == (uint32_t)k[i])) {
        unsigned c = n[i]++; if (c >= 24 && c % 120) return;
        const uint32_t *p = (const uint32_t *)(const void *)S.push[1];
        VKLOG("r140: dispatch %016llx #%u %ux%ux%u%s; push c24: %d %d %d %d | %d %d %d %d | %#x %#x %#x %#x %#x %#x %#x %#x; c0 %#llx size %u; viewport %.0f %.0f %.0f %.0f",
              (unsigned long long)cs->key, c, (unsigned)d->x, (unsigned)d->y, (unsigned)d->z, d->ind ? " (indirect)" : "", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
              p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], (unsigned long long)S.ubo[5][0].addr, (unsigned)S.ubo[5][0].size, S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3]);
        { const uint64_t a = S.ubo[5][0].addr; uint32_t sz = (uint32_t)S.ubo[5][0].size; if (a && sz >= 64 && !IsBadReadPtr((const void *)(uintptr_t)a, 64)) { const uint32_t *u = (const uint32_t *)(uintptr_t)a;
            VKLOG("r140:   its c0 words 0-15 now: %#x %#x %#x %#x | %#x %#x %#x %#x | %#x %#x %#x %#x | %#x %#x %#x %#x", u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11], u[12], u[13], u[14], u[15]); } }
        return; }
}
#include "gfx_vk_eden.inc"   // NXN1X lane B
// S1a (parallel recording): a worker's recording contexts (gfx_vk.h), made as the replay thread's static ones start (g_rec_main zero; g_rec_e_main's initial values,
// with an Eden ring pair of its own: chunks made on first use by eden_ring_alloc as for the shared rings, same usage), and the calling thread's g_rec / g_rec_e set
void *vk_rec_ctx_new(void) { struct rec_ctx *c = calloc(1, sizeof *c); if (!c) die("[vk] parallel recording: no memory for a recording context");
    c->vkst = &c->own_vkst; c->r131 = &c->own_r131; c->set_hits = &c->own_set[0]; c->set_new = &c->own_set[1]; c->pc = &c->own_pc;   // S1c (parallel recording): it counts into its own (vk_rec_ctx_fold)
    c->pscale = 1;   // S1e (parallel recording): as the replay thread's
    return c; }
// S1c (parallel recording): a worker's counters added into the replay thread's (the globals the logs read) and zeroed. Called on the replay thread once the worker's chunks of
// the part are joined (gfx_vk_core.c vk_chunk_wait_part: a chunk job ends with vk_chunk_finish, after its last count), so neither side counts meanwhile. Counted: struct
// rec_ctx's vkst (max_pass_draws as a maximum), r131, set hits / new; struct rec_eden's s45 / s35 / vertex-lookup / vtxpull counters and eden_prof's sums
static void fold_u64(uint64_t *d, uint64_t *s, size_t n) { for (size_t i = 0; i < n; i++) { d[i] += s[i]; s[i] = 0; } }
#pragma push_macro("g_s45_nk")
#undef g_s45_nk
_Static_assert(sizeof(vkst_t) % 8 == 0 && sizeof(r131_t) % 8 == 0 && sizeof(rec_pcnt_t) % 8 == 0, "S1c: the counter structs are uint64_t only");
_Static_assert(offsetof(struct rec_eden, c1buf) - offsetof(struct rec_eden, g_s45_nk) == 8 * (2 + 2 + 2 + 3 + 1 + 1 + 3 + 1 + 1 + 1 + VPF_N + 1 + 4),
               "S1c: struct rec_eden's counters (g_s45_nk .. prof_s) are contiguous uint64_t, nothing else between");
static void rec_e_fold(struct rec_eden *d, struct rec_eden *s) {
    const size_t o = offsetof(struct rec_eden, g_s45_nk), n = (offsetof(struct rec_eden, c1buf) - o) / 8;
    fold_u64((uint64_t *)((uint8_t *)d + o), (uint64_t *)((uint8_t *)s + o), n);
}
#pragma pop_macro("g_s45_nk")
static void rec_fold_into(struct rec_ctx *d, struct rec_eden *de, struct rec_ctx *s, struct rec_eden *se) {
    if (s && s != d) { const uint64_t mx = d->vkst->max_pass_draws > s->vkst->max_pass_draws ? d->vkst->max_pass_draws : s->vkst->max_pass_draws;
        fold_u64((uint64_t *)d->vkst, (uint64_t *)s->vkst, sizeof(vkst_t) / 8); d->vkst->max_pass_draws = mx;
        fold_u64((uint64_t *)d->r131, (uint64_t *)s->r131, sizeof(r131_t) / 8);
        fold_u64((uint64_t *)d->pc, (uint64_t *)s->pc, sizeof(rec_pcnt_t) / 8);
        *d->set_hits += *s->set_hits; *s->set_hits = 0; *d->set_new += *s->set_new; *s->set_new = 0;
        for (int i = 0; i < s->r80_nvp; i++) { const r80_vp_t *v = &s->r80_vp[i]; int j = 0;   // r80's viewports: merged by size
            while (j < d->r80_nvp && (d->r80_vp[j].pw != v->pw || d->r80_vp[j].ph != v->ph || d->r80_vp[j].vw != v->vw || d->r80_vp[j].vh != v->vh)) j++;
            if (j < d->r80_nvp) d->r80_vp[j].n += v->n; else if (d->r80_nvp < 16) d->r80_vp[d->r80_nvp++] = *v; else d->r80_vp_lost++; }
        d->r80_vp_lost += s->r80_vp_lost; s->r80_nvp = 0; s->r80_vp_lost = 0; }
    if (se && se != de) rec_e_fold(de, se);
}
void vk_rec_ctx_fold(void *rec, void *rec_e) { rec_fold_into(&g_rec_main, &g_rec_e_main, rec, rec_e); }
// ---- S1c (parallel recording): the stress self-test (DM_VK_PARALLEL_SELFTEST=2, gfx_vk_core.c par_selftest), no draw recorded ----
// Every worker at once (a start barrier per round, then the same work in another order per worker), ST_R rounds, through the shared-state paths a recording thread
// takes, with synthetic inputs: (a) eden_pipe_put's get-or-insert on fresh replay keys (pairs of keys reaching one Eden pipeline, one key with none); (b)
// eden_pipe_publish's once-publication of an info (compute, no set layout: nothing registered, g_eden_kdyn untouched); (c) eden_pipe_sw_once + eden_sw_note,
// vpull_inputs_publish, layer_out_mark on fresh shader objects; (d) counting into the worker's own contexts; (e) the per-thread scratch (TLS) staying the thread's.
// The replay thread then checks every result, removes what (a) put into the maps (its shaders and pipelines are addresses in g_st's tags, never a real object's),
// restores the s21 counters (and kept their 16 diff lines from being used: g_eden_kdup held at 16 meanwhile), and folds (d) into a scratch context pair
enum { ST_K = 16, ST_R = 32, ST_W = 6 };   // keys a round (the last with no pipeline), rounds, workers at most (gfx_vk_core.c PAR_MAX)
static struct {
    int nw, arrived, tmo;
    uint8_t ktag[ST_R][ST_K], ptag[ST_R][ST_K / 2];
    eden_pipe *got[ST_W][ST_R][ST_K]; int made[ST_R][ST_K];
    eden_pipe pub[ST_R], swp[ST_R]; int pub_won[ST_R], pub_bad, sw_first[ST_R], vp_first[ST_R], lo_first[ST_R], tls_bad;
    vkshader_t sh[ST_R][5]; nvn_eden_bind swb;
    uint64_t kmiss0, kdup0, main_draws0;
} g_st;
static eden_gkey st_key(int r, int k) { eden_gkey key; memset(&key, 0, sizeof key); key.sh[0] = (nvn_eden_shader *)(void *)&g_st.ktag[r][k]; return key; }
static nvn_eden_pipeline *st_pipe(int r, int k) { return k == ST_K - 1 ? NULL : (nvn_eden_pipeline *)(void *)&g_st.ptag[r][k / 2]; }
void vk_rec_stress_begin(int nw) {
    memset(&g_st, 0, sizeof g_st); g_st.nw = nw < ST_W ? nw : ST_W;
    g_st.swb.kind = NVN_EB_SSBO; g_st.swb.is_written = 1; g_st.swb.nvn_stage = 1; g_st.swb.count = 1;   // the fragment stage writes a storage buffer
    for (int r = 0; r < ST_R; r++) { g_st.swp[r].pi.num_binds = 1; g_st.swp[r].pi.binds = &g_st.swb; }
    pthread_mutex_lock(&g_eden_mu); g_st.kmiss0 = g_eden_kmiss; g_st.kdup0 = g_eden_kdup; if (g_eden_kdup < 16) g_eden_kdup = 16; pthread_mutex_unlock(&g_eden_mu);
    g_st.main_draws0 = g_rec_main.vkst->draws;
}
static void st_barrier(int r) {   // all workers into round r together (2 s at most: then the round runs anyway and the test fails)
    __atomic_add_fetch(&g_st.arrived, 1, __ATOMIC_ACQ_REL); const uint64_t t0 = tnow();
    while (__atomic_load_n(&g_st.arrived, __ATOMIC_ACQUIRE) < g_st.nw * (r + 1)) { if (tnow() - t0 > 2000000000ull) { __atomic_store_n(&g_st.tmo, 1, __ATOMIC_RELAXED); break; } sched_yield(); }
}
void vk_rec_stress_run(int w) {
    if (w < 0 || w >= g_st.nw) return;
    for (int r = 0; r < ST_R; r++) {
        st_barrier(r);
        g_vpull_want = w + 1; g_eden_dbuf_bad = w + 1; g_pd_draw = w + 1;   // (e)
        for (int i = 0; i < ST_K; i++) { const int k = (i + 5 * w) % ST_K; const eden_gkey key = st_key(r, k); int made = 0;   // (a)
            g_st.got[w][r][k] = eden_pipe_put(&key, st_pipe(r, k), &made); if (made) __atomic_add_fetch(&g_st.made[r][k], 1, __ATOMIC_RELAXED); }
        nvn_eden_pipeline_info pi; memset(&pi, 0, sizeof pi); pi.built = 1; pi.compute = 1; pi.num_entries = (uint32_t)r + 1; pi.dyn = (uint32_t)w; pi.frag_outputs = 3u * (uint32_t)w + 1u;   // (b)
        if (eden_pipe_publish(&g_st.pub[r], &pi)) __atomic_add_fetch(&g_st.pub_won[r], 1, __ATOMIC_RELAXED);
        { const eden_pipe *e = &g_st.pub[r];   // what any thread then reads: ready, and one publisher's whole info
          if (!__atomic_load_n(&e->ready, __ATOMIC_ACQUIRE) || e->pi.num_entries != (uint32_t)r + 1 || e->pi.frag_outputs != 3u * e->pi.dyn + 1u) __atomic_add_fetch(&g_st.pub_bad, 1, __ATOMIC_RELAXED); }
        int first; (void)eden_pipe_sw_once(&g_st.swp[r], &first); if (first) __atomic_add_fetch(&g_st.sw_first[r], 1, __ATOMIC_RELAXED);   // (c)
        { vkshader_t *const sh[6] = {&g_st.sh[r][0], &g_st.sh[r][1], &g_st.sh[r][2], &g_st.sh[r][3], &g_st.sh[r][4], NULL}; eden_sw_note(&g_st.swp[r].pi, sh); }   // (all: the atomics raced)
        if (vpull_inputs_publish(&g_st.sh[r][0], 0x100u + (uint32_t)w)) __atomic_add_fetch(&g_st.vp_first[r], 1, __ATOMIC_RELAXED);
        if (layer_out_mark(&g_st.sh[r][2])) __atomic_add_fetch(&g_st.lo_first[r], 1, __ATOMIC_RELAXED);
        g_vkst.draws++; g_r131.vb++; g_set_hits++; g_s35_push++; g_vlk_hit++;   // (d) the thread's own contexts
        if (g_vpull_want != w + 1 || g_eden_dbuf_bad != w + 1 || g_pd_draw != w + 1 || g_rec == &g_rec_main || g_rec_e == &g_rec_e_main) __atomic_add_fetch(&g_st.tls_bad, 1, __ATOMIC_RELAXED);
        g_vpull_want = 0; g_eden_dbuf_bad = 0; g_pd_draw = 0;
    }
}
int vk_rec_stress_check(void *const *rec, void *const *rec_e, char *out, size_t n) {
    const int nw = g_st.nw; int put_bad = 0, map_bad = 0, sw_bad = 0, vp_bad = 0, lo_bad = 0, pub_bad = g_st.pub_bad, cnt_bad = 0;
    pthread_mutex_lock(&g_eden_mu);
    for (int r = 0; r < ST_R; r++) for (int k = 0; k < ST_K; k++) {   // (a): one put per key, every worker the same entry, keys of one pipeline one entry, no pipeline the marker
        const eden_pipe *e0 = g_st.got[0][r][k]; const eden_gkey key = st_key(r, k); nvn_eden_pipeline *const p = st_pipe(r, k);
        put_bad += g_st.made[r][k] != 1; for (int w = 1; w < nw; w++) put_bad += g_st.got[w][r][k] != e0;
        if (!p) put_bad += e0 != (void *)1;
        else { put_bad += e0 == (void *)1 || !e0 || e0->p != p; if (k + 1 < ST_K - 1 && !(k & 1)) put_bad += g_st.got[0][r][k + 1] != e0;
               map_bad += nvn_hm_get(&g_eden_pe, &p) != e0; }
        map_bad += nvn_hm_get(&g_eden_gp, &key) != e0;
    }
    for (int r = 0; r < ST_R; r++) {   // undo (a): the test's keys and pipelines out of the maps (each test pipeline's one eden_pipe freed once)
        for (int k = 0; k < ST_K; k++) { const eden_gkey key = st_key(r, k); (void)nvn_hm_del(&g_eden_gp, &key); }
        for (int j = 0; j < ST_K / 2; j++) { nvn_eden_pipeline *const p = (nvn_eden_pipeline *)(void *)&g_st.ptag[r][j]; eden_pipe *e = nvn_hm_del(&g_eden_pe, &p); free(e); }
    }
    g_eden_kmiss = g_st.kmiss0; g_eden_kdup = g_st.kdup0;
    pthread_mutex_unlock(&g_eden_mu);
    for (int r = 0; r < ST_R; r++) {   // (b), (c)
        pub_bad += g_st.pub_won[r] != 1 || !g_st.pub[r].ready;
        sw_bad += g_st.sw_first[r] != 1 || g_st.swp[r].s64w != 2 || g_st.sh[r][1].eden_sw != 2;
        for (int st = 0; st < 5; st++) if (st != 1) sw_bad += g_st.sh[r][st].eden_sw != 1;
        vp_bad += g_st.vp_first[r] != 1 || g_st.sh[r][0].vpull_known != 1 || (g_st.sh[r][0].vpull_inputs & ~0xffu) != 0x100u || (int)(g_st.sh[r][0].vpull_inputs & 0xffu) >= nw;
        lo_bad += g_st.lo_first[r] != 1 || g_st.sh[r][2].layer_out != 1;
    }
    struct rec_ctx *const sc = vk_rec_ctx_new(); struct rec_eden *const se = calloc(1, sizeof *se);   // (d): each worker's counts its own, folded into a scratch pair
    if (!se) die("[vk] parallel recording: no memory for the self-test");
    for (int w = 0; w < nw; w++) { struct rec_ctx *const c = rec[w]; struct rec_eden *const ce = rec_e[w];
        cnt_bad += c->vkst->draws != ST_R || c->r131->vb != ST_R || *c->set_hits != ST_R;
#pragma push_macro("g_s35_push")
#pragma push_macro("g_vlk_hit")
#undef g_s35_push
#undef g_vlk_hit
        cnt_bad += ce->g_s35_push != ST_R || ce->g_vlk_hit != ST_R;
        rec_fold_into(sc, se, c, ce);
        cnt_bad += c->vkst->draws != 0 || ce->g_s35_push != 0; }
    cnt_bad += sc->vkst->draws != (uint64_t)nw * ST_R || sc->r131->vb != (uint64_t)nw * ST_R || *sc->set_hits != (uint64_t)nw * ST_R || se->g_s35_push != (uint64_t)nw * ST_R || se->g_vlk_hit != (uint64_t)nw * ST_R;
#pragma pop_macro("g_vlk_hit")
#pragma pop_macro("g_s35_push")
    cnt_bad += g_rec_main.vkst->draws != g_st.main_draws0;   // nothing went into the replay thread's
    free(sc); free(se);
    const int ok = !g_st.tmo && !put_bad && !map_bad && !pub_bad && !sw_bad && !vp_bad && !lo_bad && !cnt_bad && !g_st.tls_bad;
    snprintf(out, n, "%d workers x %d rounds: get-or-insert %d bad (maps %d), once-publication %d, storage-write marks %d, vertex inputs %d, layer marks %d, counters %d, "
             "per-thread scratch %d%s", nw, ST_R, put_bad, map_bad, pub_bad, sw_bad, vp_bad, lo_bad, cnt_bad, g_st.tls_bad, g_st.tmo ? ", start barrier timed out" : "");
    return ok;
}
void *vk_rec_eden_new(void) {
    struct rec_eden *e = calloc(1, sizeof *e); eden_ring *r = calloc(2, sizeof *r);
    if (!e || !r) die("[vk] parallel recording: no memory for an Eden recording context");
    r[0].usage = g_eden_dring.usage; r[1].usage = g_eden_uring.usage;
    e->dring = &r[0]; e->uring = &r[1];
#pragma push_macro("g_s45_kseen")
#pragma push_macro("g_s45_db")
#undef g_s45_kseen
#undef g_s45_db
    e->g_s45_kseen = ~0ull; e->g_s45_db.gen = ~0ull;   // (g_rec_e_main's initialisers; the field names, not the macros of the thread's instance)
#pragma pop_macro("g_s45_db")
#pragma pop_macro("g_s45_kseen")
    return e;
}
void vk_rec_ctx_bind(void *rec, void *rec_e) { g_rec = rec ? (struct rec_ctx *)rec : &g_rec_main; g_rec_e = rec_e ? (struct rec_eden *)rec_e : &g_rec_e_main; }
const void *vk_rec_ctx_cur(void) { return g_rec; }
void *vk_rec_eden_cur(void) { return g_rec_e; }   // S1e (parallel recording): gfx_vk_core.c vk_par_help_run restores the replay thread's pair
// ootcs (11-20, diagnostic, DM_VK_TRACE_CSIMG=1 live; 2 = every key): Ocarina of Time's upscale network runs (no lowering or pipeline failures) but its output never
// reaches the 2560x1440 layer the final pass draws. For each NVN2 compute key whose image slots name a texture at least 1280 wide (=2: every key), its first 3 dispatches
// log every texture and image slot with its host image, so the network's output image can be matched with the layer's (DM_VK_TRACE_SIZE prints the layer's host image)
static void ootcs_slots(const vkshader_t *cs, const p_dispatch *d) {
    static long gen = -1; static int on; if (gen != dm_env_gen) { gen = dm_env_gen; const char *e = getenv("DM_VK_TRACE_CSIMG"); on = e && *e ? atoi(e) : 0; }
    if (!on || !cs->eden || !dm_sdk_ounce()) return;
    enum { NK = 256 }; static uint64_t key[NK]; static unsigned char n[NK]; static int nk; int k = 0;
    while (k < nk && key[k] != cs->key) k++;
    if (k == nk) { if (nk == NK) return; key[nk++] = cs->key; n[k] = 0; }
    if (n[k] >= 3) return;
    uint32_t L[8]; if (!nvn_eden_shader_nvn2_c0(cs->eden, L)) return;
    const view_t *v; int wide = on >= 2;
    for (uint32_t r = 0; r < L[6] && r < 8 && !wide; r++) { const tex_t *t = NULL; const uint64_t h = c2_handle(5, (int)(0x120 + 8 * r));
        if ((h & 0xfffff) && vk_tex_by_id((int)(h & 0xfffff), &v, &t) && t && t->d.width >= 1280) wide = 1; }
    if (!wide) return;
    n[k]++; extern uint64_t g_tex_frame;
    VKLOG("ootcs: dispatch %016llx #%u %ux%ux%u%s at present %llu after pass %d: %u textures, %u separate textures, %u images, push %u B", (unsigned long long)cs->key, (unsigned)n[k],
          (unsigned)d->x, (unsigned)d->y, (unsigned)d->z, d->ind ? " (indirect)" : "", (unsigned long long)g_tex_frame, g_sub_pass - 1, L[3], L[4], L[6], L[0]);
    for (int im = 0; im < 2; im++) for (uint32_t r = 0; r < (im ? L[6] : L[3]) && r < (im ? 8u : 32u); r++) {
        const tex_t *t = NULL; const uint64_t h = c2_handle(5, (int)((im ? 0x120 : 0x20) + 8 * r));
        if (!(h & 0xfffff) || !vk_tex_by_id((int)(h & 0xfffff), &v, &t) || !t) { VKLOG("ootcs:   %s slot %u handle %#llx: no texture", im ? "image" : "texture", r, (unsigned long long)h); continue; }
        const tex_t tc = *t; const vkimg_t *hi = vk_tex_peek(&tc);
        VKLOG("ootcs:   %s slot %u handle %#llx: %ux%u levels %u nvn format %#x pool %p%+lld -> host image %p (%ux%u vk fmt %d), gpu_w %d, rendered at present %lld",
              im ? "image" : "texture", r, (unsigned long long)h, tc.d.width, tc.d.height, (unsigned)tc.d.levels, (unsigned)tc.d.format, (const void *)tc.pool, (long long)tc.off,
              (const void *)hi, hi ? hi->w : 0, hi ? hi->h : 0, hi ? (int)hi->fmt : 0, hi ? hi->gpu_w : 0, hi ? (long long)hi->wpres - 1 : -1LL); }
}
// ootcs: dispatches dropped before the GPU sees them (Eden pipeline or descriptors not ready, indirect arguments not found), by key, logged for every new key and
// every 4096th drop; they were silent outside the NDSR keys
static void ootcs_dropped(const vkshader_t *cs, const char *why) {
    enum { NK = 64 }; static uint64_t key[NK]; static int nk; static uint64_t total; total++; int k = 0;
    while (k < nk && key[k] != cs->key) k++;
    const int fresh = k == nk; if (fresh && nk < NK) key[nk++] = cs->key;
    if (fresh || !(total & 4095)) VKLOG("[RED] dispatch %016llx dropped: %s (%llu dispatches dropped so far)", (unsigned long long)cs->key, why, (unsigned long long)total);
}
#if defined(DM_NXN1X)
// ootdlss (11-21, NPARK, design OOT_DLSS_DESIGN.md): NVIDIA DLSS Super Resolution (Streamline, nvn_sr_evaluate) in place of Ocarina of Time's gameplay upscale
// network, the compute dispatch d99fbeeb53d38847 that samples the square root of the tonemapped scene (an E5B9G9R9 image at the render size, 854x480 to 1138x640,
// written by a335c1db6270c355) into the 2560x1440 B10G11R11 picture before the HUD (oot_upsprobe 11-21). Reused, with their sources named:
//  - Longshot (refer\Longshot, GPL-3.0): gpu/vk/vk_temporal.cpp temporal_note_scene_draw (the scene camera: vertex uniform buffer 0 of the draws into the scene's
//    colour target, >= 0x230 bytes, the target's size at 0x120, the view-projection at 0x30, the projection at 0x70 with its w row 0 0 -1 0 at 0xa0, its jitter
//    -P[0][2], -P[1][2]; the largest such target of the frame is the scene), shaders/dlss_prep.comp and dlss_out.comp (the colour as the network samples it,
//    LDR, and the picture squared back), dlss_math.h (Halton jitter, its phases, the reprojection and unjitter), nx2/oot_jitter.py (the jitter sites, nso.cpp)
//  - NX2navi (refer\NX2navi-main, MIT): runtime/shaders/dlaa_mv.comp's depth check (zfit) and its bias mask for what moves by itself
//  - ours: nvn_sr_evaluate (nvn_streamline.cpp) as replay_dlss calls it, GSR1's pipeline and image pattern (gsr_ready, gsr_tmp)
// Anything unexpected (no camera this frame, other sizes or formats, Streamline failing) leaves the network to run, the reason in the log when it changes.
// DM_VK_OOT_DLSS: unset = on for Ocarina of Time on an NVIDIA GPU, 0 off (the network, no jitter sites: nso.cpp), 1 on. Read again on an environment change
#include "ootdl_prep_comp.spv.h"
#include "ootdl_out_comp.spv.h"
extern uint8_t *dm_fps_state;   // hle_nvn.c: the frame step state page; nso.cpp's jitter sites read +0xd0..+0xd8 (+0xe4 'OJIT' when installed)
static long g_odl_gen = -1; static int g_odl_on;
static void ootdl_jitter(float x, float y, int on);
static int ootdl_on(void) {
    // dlsstoggle (NPARK 11-21): DM_VK_OOT_DLSS set wins; else the setting "NVIDIA DLSS Super Resolution" (graphics advanced, hotkey "Toggle DLSS"), read
    // again at each present (live), for Ocarina of Time on an NVIDIA GPU
    static uint64_t frame_seen = ~0ull; static int setting = -1;
    if (g_odl_gen != dm_env_gen || frame_seen != g_tex_frame) {
        const int envchg = g_odl_gen != dm_env_gen; g_odl_gen = dm_env_gen; frame_seen = g_tex_frame;
        const char *e = getenv("DM_VK_OOT_DLSS"); int nvnh_oot_upscale_standin(void);   // (it is the Ocarina of Time title test)
        const int s = nvn_dlss_sr_wanted();
        if (envchg || s != setting) { setting = s;
            g_odl_on = e && *e ? atoi(e) != 0 : (s && nvnh_oot_upscale_standin() && V.props.vendorID == 0x10DE);
            static int said = -1; if (said != g_odl_on) { said = g_odl_on; VKLOG("ootdlss: DLSS in place of the upscale network %s (DM_VK_OOT_DLSS %s, setting %s; vendor %#x)", g_odl_on ? "on" : "off", e && *e ? e : "unset", s ? "on" : "off", V.props.vendorID); }
            if (!g_odl_on) ootdl_jitter(0, 0, 0); } }
    return g_odl_on;
}
#define ODL_CAM 0x130u   // the camera block's bytes read: up to the target's size at 0x120
#define ODL_SLOTS 16u
static struct { pthread_mutex_t mu; uint64_t frame; int valid; uint32_t w, h; float c[ODL_CAM / 4]; tex_t dep; view_t dview; int has_dview;
                const char *reject; uint64_t reject_frame, rejects; } g_odc = {PTHREAD_MUTEX_INITIALIZER};
static void ootdl_draw(void) {   // fg_observe_draw: every draw (replay thread and workers): this frame's scene camera
    if (!ootdl_on() || !P_att[0].im || !P_att[8].im || P_att[0].fmt != VK_FORMAT_B10G11R11_UFLOAT_PACK32 || pass_scale != 1 || !S.rt_has_dep) return;
    const p_bind_buf *u = &S.ubo[0][0]; const uint32_t w = P_att[0].w, h = P_att[0].h;
    if (!u->addr || u->size < 0x230 || w < 320 || !vk_pool_of(u->addr, ODL_CAM)) return;
    const uint64_t now = g_tex_frame;
    pthread_mutex_lock(&g_odc.mu);
    const int skip = g_odc.valid && g_odc.frame == now && (uint64_t)w * h <= (uint64_t)g_odc.w * g_odc.h;   // the largest target of the frame wins
    pthread_mutex_unlock(&g_odc.mu);
    if (skip) return;
    float c[ODL_CAM / 4]; vk_ubd_read(u->addr, ODL_CAM, (uint8_t *)c);
    if (c[0x120 / 4] != (float)w || c[0x124 / 4] != (float)h) return;   // (not this target's block: other passes' uniform buffers)
    // NPARK 11-21: DLSS only on a camera that passes these checks while the game runs; the first failing check is kept for the frame's line (ootdl_dispatch)
    const char *bad = NULL;
    for (int i = 0; i < 0xb0 / 4 && !bad; i++) if (!isfinite(c[i])) bad = "a camera value is not finite";
    if (!bad && (c[0xa0 / 4] != 0.0f || c[0xa4 / 4] != 0.0f || c[0xa8 / 4] != -1.0f || c[0xac / 4] != 0.0f)) bad = "the projection's w row is not 0 0 -1 0 (not perspective)";
    if (!bad) { double e = 0; for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) { double d = 0; for (int k = 0; k < 3; k++) d += (double)c[a * 4 + k] * c[b * 4 + k]; d -= a == b; if (fabs(d) > e) e = fabs(d); }
        if (e > 2e-3) bad = "the view's rows are not orthonormal"; }
    const float *P = c + 0x70 / 4;
    if (!bad && !(P[0] > 0 && P[5] > 0)) bad = "the projection's scale is not positive";
    if (!bad && fabs((double)P[5] / P[0] - (double)w / h) > 0.03 * ((double)w / h)) bad = "the projection's aspect is not the target's";
    if (!bad) { double e = 0, m = 0;   // the block's view-projection (0x30) is the projection times the view (rows 0-2, then 0 0 0 1)
        for (int r = 0; r < 4; r++) for (int k = 0; k < 4; k++) { double s = 0; for (int j = 0; j < 4; j++) s += (double)P[r * 4 + j] * (j < 3 ? c[j * 4 + k] : k == 3 ? 1.0 : 0.0);
            const double vpv = c[0x30 / 4 + r * 4 + k]; if (fabs(s - vpv) > e) e = fabs(s - vpv); if (fabs(vpv) > m) m = fabs(vpv); }
        if (e > 1e-3 * (m > 1 ? m : 1)) bad = "the view-projection is not the projection times the view"; }
    if (bad) { pthread_mutex_lock(&g_odc.mu); g_odc.reject = bad; g_odc.reject_frame = now; g_odc.rejects++; pthread_mutex_unlock(&g_odc.mu); return; }
    pthread_mutex_lock(&g_odc.mu);
    if (!(g_odc.valid && g_odc.frame == now && (uint64_t)w * h <= (uint64_t)g_odc.w * g_odc.h)) {
        g_odc.frame = now; g_odc.valid = 1; g_odc.w = w; g_odc.h = h; memcpy(g_odc.c, c, sizeof c); g_odc.dep = S.rt_dep; g_odc.has_dview = S.rt_has_dview; if (S.rt_has_dview) g_odc.dview = S.rt_dview; }
    pthread_mutex_unlock(&g_odc.mu);
}
enum { ODL_COLOR, ODL_DEPTH, ODL_MV, ODL_MASK, ODL_DIST0, ODL_DIST1, ODL_OUT, ODL_NIMG };
typedef struct { uint32_t iw, ih, ow, oh; VkImage img[ODL_NIMG]; VkDeviceMemory mem[ODL_NIMG]; VkImageView view[ODL_NIMG]; int fresh, cur, dist_ok; uint32_t dw, dh; uint64_t used; } odl_size;   // dw, dh: the render size dist_ok holds for
static struct {
    int state; VkDescriptorSetLayout dsl[2]; VkPipelineLayout pl[2]; VkPipeline pipe[2]; VkSampler nearest, linear;   // 0 prep, 1 out; state 0 not made, 1 ready, -1 failed
    odl_size sz[6]; uint64_t evals, falls, jit_n, fg_tags; uint64_t last_eval; int jit_live;
    float prev_vp[16], prev_jit[2]; uint64_t prev_frame; uint32_t prev_w, prev_h; int prev_ok;
    char why[160];
    VkBuffer stat, stat_rb; VkDeviceMemory stat_mem, stat_rb_mem; uint32_t *stat_map;   // NaN guard (NPARK 11-21)
    uint64_t slot_fence[ODL_SLOTS], runs, bad[4], blacks; int force_reset;
    unsigned info[4];   // dlsstoggle overlay: render width, height, DLSS input width, output height
    uint32_t slot_w[ODL_SLOTS], slot_h[ODL_SLOTS], slot_rw[ODL_SLOTS], slot_ow[ODL_SLOTS], slot_oh[ODL_SLOTS], slot_reset[ODL_SLOTS], last_in_sig, last_out_sig; uint64_t stale, stale_run; int have_sig, off_run;   // srfit diag: output standing still
} g_odl;
typedef struct { float m[16]; float jit[2]; int32_t size[2]; float dz[4]; int32_t flags; float zmask; } odl_prep_pc;
// dlsstoggle: for the performance overlay (weak in performance_overlay.cpp): on, running (an evaluation in the last 2 presents), render size, DLSS input size, output
void nvnh_dlss_info(unsigned out[6]) { out[0] = (unsigned)(g_odl_on && !g_odl.off_run); out[1] = g_odl.evals && g_tex_frame <= g_odl.last_eval + 2; out[2] = g_odl.info[0]; out[3] = g_odl.info[1]; out[4] = g_odl.info[2]; out[5] = g_odl.info[3]; }
static int ootdl_fg_owner(void) { return ootdl_on() && g_odl.evals && g_tex_frame <= g_odl.last_eval + 2; }
// resfix (NPARK 11-21, "lines across the picture like a notebook"): while DLSS runs the scene is a fixed share of the picture (nso.cpp, NX2navi res_fixed): state +0xe8
// on, +0xec the share (1.0 = the game's own fixed path), +0xf0 'ORES' when the sites are in. On after 120 evaluations in a row (the controller has filled the fields the
// fixed path keeps: NX2navi RES_FIXED_AFTER), off at once when DLSS goes off (F7, the setting, off for the run; not a pause: resfix2 below). Checked while it runs:
// the scene must reach the share's size (output x share, +-2) within 240 presents and stay there; otherwise off for the run, the game's dynamic resolution and
// srfit, with the reason. DM_VK_OOT_RESFIX: unset 0.5 (1280x720 of 2560x1440: DLSS Performance, 2x with no resampling), 0 off, else the share (0.34 to 0.75:
// above 0.75 the game's render-target heap was not made for it, NX2navi PLAIN_SHARE)
static struct { int on, dead, said; float share; uint64_t on_at, run_evals, last_seen_eval, bad; long gen; } g_rfx = {.gen = -1};
static void ootdl_resfix_set(int on, const char *why) {
    uint8_t *st = dm_fps_state; if (!st || memcmp(st + 0xf0, "ORES", 4)) return;
    const float s = on ? g_rfx.share : 1.0f; uint32_t b; memcpy(&b, &s, 4);
    __atomic_store_n((uint32_t *)(st + 0xec), b, __ATOMIC_RELAXED); __atomic_store_n((uint32_t *)(st + 0xe8), (uint32_t)(on != 0), __ATOMIC_RELEASE);
    if (on != g_rfx.on) VKLOG("resfix: the scene %s at present %llu: %s", on ? "a fixed share of the picture" : "back to the game's dynamic resolution", (unsigned long long)g_tex_frame, why);
    g_rfx.on = on; if (on) g_rfx.on_at = g_tex_frame; g_rfx.bad = 0;
}
static void ootdl_resfix_tick(void) {   // once a present (ootdl_dispatch, the replay thread)
    if (g_rfx.gen != dm_env_gen) { g_rfx.gen = dm_env_gen; const char *e = getenv("DM_VK_OOT_RESFIX"); float v = e && *e ? (float)atof(e) : 0.5f;
        if (e && *e == '0' && v == 0) g_rfx.dead = 2; else if (g_rfx.dead == 2) g_rfx.dead = 0;
        if (v < 0.34f || v > 0.75f) v = 0.5f; if (g_rfx.share != v && g_rfx.on) { g_rfx.share = v; ootdl_resfix_set(1, "the share changed (DM_VK_OOT_RESFIX)"); } g_rfx.share = v;
        if (g_rfx.dead == 2 && g_rfx.on) ootdl_resfix_set(0, "DM_VK_OOT_RESFIX=0"); }
    if (!dm_fps_state || memcmp(dm_fps_state + 0xf0, "ORES", 4)) return;   // (not Ocarina of Time, or the sites not in: nso.cpp says why)
    if (!g_rfx.said) { g_rfx.said = 1; VKLOG("resfix: fixed scene share sites in (share %.3f; on once DLSS has run 120 evaluations)", g_rfx.share); }
    // resfix2 (NPARK 11-21 15:22, "lines come back after an area change"; 154.7 s: the loading's 30 presents without an evaluation turned it off, the game's
    // dynamic resolution went back to 854x480 and srfit stretched it again for 4 s until 120 more evaluations): a pause in the evaluations (loading, a
    // camera the check refuses) keeps the fixed share; only DLSS itself going off (F7, the setting, off for the run) or DM_VK_OOT_RESFIX=0 turns it off.
    // The 120 evaluations are the start's only (the controller's fields filled once); once on, DLSS coming back on turns it on at once
    static int ever;
    const int dlss = g_odl_on && !g_odl.off_run;
    if (g_odl.evals != g_rfx.last_seen_eval) { g_rfx.last_seen_eval = g_odl.evals; g_rfx.run_evals++; }
    if (!dlss) { g_rfx.run_evals = 0; if (g_rfx.on) ootdl_resfix_set(0, !g_odl_on ? "DLSS off" : "DLSS off for this run"); return; }
    if (!g_rfx.on && !g_rfx.dead && (ever || g_rfx.run_evals >= 120)) { ootdl_resfix_set(1, ever ? "DLSS on again" : "DLSS ran 120 evaluations"); ever = 1; }
}
static void ootdl_resfix_check(uint32_t sw, uint32_t sh, uint32_t ow, uint32_t oh) {   // each evaluation: the scene reached the share's size
    if (!g_rfx.on) return;
    const int ew = (int)(ow * g_rfx.share + 0.5f), eh = (int)(oh * g_rfx.share + 0.5f);
    const int ok = abs((int)sw - ew) <= 2 && abs((int)sh - eh) <= 2;
    static uint64_t said_ok; if (ok && said_ok != g_rfx.on_at) { said_ok = g_rfx.on_at; VKLOG("resfix: scene %ux%u reached %llu presents after the fixed share went on (DLSS %ux%u -> %ux%u, no resampling)", sw, sh, (unsigned long long)(g_tex_frame - g_rfx.on_at), sw, sh, ow, oh); }
    if (ok) { g_rfx.bad = 0; return; }
    if (g_tex_frame < g_rfx.on_at + 240) return;
    if (++g_rfx.bad >= 60) { char w[160]; snprintf(w, sizeof w, "the scene stayed %ux%u, not the share's %dx%d, for 60 evaluations: off for this run (srfit as before)", sw, sh, ew, eh); g_rfx.dead = 1; ootdl_resfix_set(0, w); }
}
static const VkFormat k_odl_fmt[ODL_NIMG] = {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R8_UNORM, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT};
static void ootdl_jitter(float x, float y, int on);
static void ootdl_why(const char *w) {   // a reason the network runs this time, logged when it changes; the jitter goes off at once (NPARK 11-21 (4): only while DLSS runs)
    ootdl_jitter(0, 0, 0);
    if (strcmp(g_odl.why, w)) { snprintf(g_odl.why, sizeof g_odl.why, "%s", w); VKLOG("ootdlss: the network runs at present %llu: %s (%llu DLSS evaluations so far)", (unsigned long long)g_tex_frame, w, (unsigned long long)g_odl.evals); }
    g_odl.falls++;
}
static void ootdl_jitter(float x, float y, int on) {   // the jitter the game's next views take (nso.cpp sites), clip units
    uint8_t *st = dm_fps_state; uint32_t tag; if (!st) return; memcpy(&tag, st + 0xe4, 4); if (tag != 0x54494A4Fu) return;
    uint32_t bx, by; memcpy(&bx, &x, 4); memcpy(&by, &y, 4);
    __atomic_store_n((uint32_t *)(st + 0xd0), bx, __ATOMIC_RELAXED); __atomic_store_n((uint32_t *)(st + 0xd4), by, __ATOMIC_RELAXED); __atomic_store_n((uint32_t *)(st + 0xd8), (uint32_t)(on != 0), __ATOMIC_RELEASE);
    g_odl.jit_live = on;
}
static int ootdl_ready(void) {
    if (g_odl.state) return g_odl.state > 0;
    g_odl.state = -1;
    static const VkDescriptorType t0[9] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                           VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
    static const VkDescriptorType t1[3] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};   // (+ the NaN guard's stats)
    const VkDescriptorType *ts[2] = {t0, t1}; const uint32_t nb[2] = {9, 3}; const uint32_t pcs[2] = {sizeof(odl_prep_pc), 12};
    const uint32_t *code[2] = {ootdl_prep_comp_spv, ootdl_out_comp_spv}; const size_t bytes[2] = {sizeof ootdl_prep_comp_spv, sizeof ootdl_out_comp_spv};
    void vk_dsl_register(VkDescriptorSetLayout l, const VkDescriptorSetLayoutBinding *b, uint32_t n);
    for (int p = 0; p < 2; p++) {
        VkDescriptorSetLayoutBinding b[9]; for (uint32_t i = 0; i < nb[p]; i++) b[i] = (VkDescriptorSetLayoutBinding){i, ts[p][i], 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, nb[p], b};
        if (vkCreateDescriptorSetLayout(V.dev, &li, NULL, &g_odl.dsl[p])) { VKRED_ONCE(190, "ootdlss: descriptor set layout failed: the network runs"); return 0; }
        vk_dsl_register(g_odl.dsl[p], b, nb[p]);
        VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, pcs[p]};
        VkPipelineLayoutCreateInfo pi = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_odl.dsl[p], 1, &pr};
        if (vkCreatePipelineLayout(V.dev, &pi, NULL, &g_odl.pl[p])) { VKRED_ONCE(190, "ootdlss: pipeline layout failed: the network runs"); return 0; }
        VkShaderModule m; VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, bytes[p], code[p]};
        if (vkCreateShaderModule(V.dev, &mi, NULL, &m)) { VKRED_ONCE(190, "ootdlss: shader module failed: the network runs"); return 0; }
        VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", NULL}, g_odl.pl[p], VK_NULL_HANDLE, -1};
        VkResult r = vkCreateComputePipelines(V.dev, V.pcache, 1, &ci, NULL, &g_odl.pipe[p]); vkDestroyShaderModule(V.dev, m, NULL);
        if (r) { VKRED_ONCE(190, "ootdlss: compute pipeline failed (%d): the network runs", r); return 0; }
    }
    for (int k = 0; k < 2; k++) {
        VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, NULL, 0, k ? VK_FILTER_LINEAR : VK_FILTER_NEAREST, k ? VK_FILTER_LINEAR : VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
        if (vkCreateSampler(V.dev, &si, NULL, k ? &g_odl.linear : &g_odl.nearest)) { VKRED_ONCE(190, "ootdlss: sampler failed: the network runs"); return 0; }
    }
    // NaN guard's stats: ODL_SLOTS slots of 8 words, device-local (the passes' atomics, never on host or imported memory) and a host-visible copy read back
    for (int k = 0; k < 2; k++) {
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, ODL_SLOTS * 32, k ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT), VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
        VkBuffer *bp = k ? &g_odl.stat_rb : &g_odl.stat; VkDeviceMemory *mp = k ? &g_odl.stat_rb_mem : &g_odl.stat_mem;
        if (vkCreateBuffer(V.dev, &bi, NULL, bp)) { VKRED_ONCE(190, "ootdlss: stats buffer failed: the network runs"); return 0; }
        VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, *bp, &rq);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, k ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
        if (vkAllocateMemory(V.dev, &ai, NULL, mp) || vkBindBufferMemory(V.dev, *bp, *mp, 0)) { VKRED_ONCE(190, "ootdlss: stats memory failed: the network runs"); return 0; }
    }
    if (vkMapMemory(V.dev, g_odl.stat_rb_mem, 0, VK_WHOLE_SIZE, 0, (void **)&g_odl.stat_map)) { VKRED_ONCE(190, "ootdlss: stats map failed: the network runs"); return 0; }
    g_odl.state = 1; VKLOG("ootdlss: ready (DLSS inputs pass and output pass made)"); return 1;
}
static int ootdl_img(uint32_t w, uint32_t h, VkFormat f, VkImage *img, VkDeviceMemory *mem, VkImageView *view) {
    VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, f, {w, h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    if (vkCreateImage(V.dev, &ic, NULL, img)) { *img = VK_NULL_HANDLE; return 0; }
    VkMemoryRequirements rq; vkGetImageMemoryRequirements(V.dev, *img, &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, *img, VK_IMAGE_VIEW_TYPE_2D, f, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkAllocateMemory(V.dev, &ai, NULL, mem) || vkBindImageMemory(V.dev, *img, *mem, 0) || vkCreateImageView(V.dev, &vi, NULL, view)) { vkDestroyImage(V.dev, *img, NULL); *img = VK_NULL_HANDLE; return 0; }
    return 1;
}
static void ootdl_free(odl_size *e) {   // the device is idle (the caller waited)
    for (int i = 0; i < ODL_NIMG; i++) if (e->img[i]) { vkDestroyImageView(V.dev, e->view[i], NULL); vkDestroyImage(V.dev, e->img[i], NULL); vkFreeMemory(V.dev, e->mem[i], NULL); }
    memset(e, 0, sizeof *e);
}
// NaN guard (NPARK 11-21: non-finite DLSS inputs replaced by 0 and counted; a whole black output frame resets DLSS's history): each evaluation's counts (ootdl_prep.comp
// and ootdl_out.comp Stats) are copied to a host-visible slot and read once the GPU is past them; the totals logged when they grow, a black frame logged and the
// next evaluation started as a new history
uint64_t g_flashes;   // flashdiag: one-frame flashes so far (gfx_vk_core.c framebound logs them with its waits)
static void ootdl_stats(void) {
    const uint64_t done = vk_fence_done();
    for (int s = 0; s < (int)ODL_SLOTS; s++) {
        if (!g_odl.slot_fence[s] || g_odl.slot_fence[s] > done) continue;
        g_odl.slot_fence[s] = 0; const uint32_t *w = g_odl.stat_map + s * 8;
        float in_max, out_max; memcpy(&in_max, &w[3], 4); memcpy(&out_max, &w[4], 4);
        if (w[0] | w[1] | w[2] | w[5]) { g_odl.bad[0] += w[0]; g_odl.bad[1] += w[1]; g_odl.bad[2] += w[2]; g_odl.bad[3] += w[5]; static uint64_t said; const uint64_t tot = g_odl.bad[0] + g_odl.bad[1] + g_odl.bad[2] + g_odl.bad[3];
            if (!said || tot >= said * 2) { said = tot ? tot : 1; VKLOG("ootdlss: NaN guard: non-finite texels replaced by 0 so far: colour %llu, depth %llu, motion %llu, DLSS output %llu (this frame %u %u %u %u)", (unsigned long long)g_odl.bad[0],
                (unsigned long long)g_odl.bad[1], (unsigned long long)g_odl.bad[2], (unsigned long long)g_odl.bad[3], w[0], w[1], w[2], w[5]); } }
        // srfit diag (NPARK 11-21: "the picture stops and goes again"): an output whose signature is the frame before's while the input's changed is a DLSS
        // picture that stood still; logged with the render size and the size DLSS was given (slots are read in completion order, close enough for a signature)
        if (g_odl.have_sig && w[7] != g_odl.last_in_sig && w[6] == g_odl.last_out_sig) { const uint64_t n = ++g_odl.stale; g_odl.stale_run++;
            if (n <= 16 || !(n & (n - 1))) VKLOG("ootdlss: DLSS output unchanged while its input changed (render %u wide, DLSS input %ux%u) at present %llu (%llu so far, %llu in a row)",
                g_odl.slot_rw[s], g_odl.slot_w[s], g_odl.slot_h[s], (unsigned long long)g_tex_frame, (unsigned long long)n, (unsigned long long)g_odl.stale_run); }
        else g_odl.stale_run = 0;
        // (NPARK 11-21: once the picture stops it does not come back until a restart): 3 standing frames in a row start DLSS's history again; 60 in a row
        // leave DLSS for the run (the network, the jitter off), logged
        if (g_odl.stale_run == 3) { g_odl.force_reset = 1; VKLOG("ootdlss: DLSS output stood still 3 frames in a row: its history starts again"); }
        if (g_odl.stale_run >= 60 && !g_odl.off_run) { g_odl.off_run = 1; ootdl_jitter(0, 0, 0); VKLOG("ootdlss: DLSS output stood still %llu frames in a row after a new history: DLSS off for this run, the game's network runs", (unsigned long long)g_odl.stale_run); }
        {   // flashdiag (NPARK 11-21: the whole picture flashes white): the input's brightness signature (prep: the sum of min(r+g+b, 3) x 64 over the
            // picture) against its mean over the frames before; a frame 1.8x brighter than that is logged with the largest input value, and the frame
            // after says whether it came back (a one-frame flash, upstream of DLSS: the scene or its bloom, nanguard's case) or stayed (a real light change)
            static double mean; static int warm, pending;
            const double px = (double)g_odl.slot_w[s] * g_odl.slot_h[s]; const double v = px > 0 ? (double)w[7] / px : 0;   // (per pixel: the DLSS input size changes)
            if (pending) { pending = 0; VKLOG("flashdiag: the frame after: %s (signature %.2f a pixel, mean %.2f)", v < mean * 1.3 ? "back to the mean: a one-frame flash" : "still bright: a lasting change", v, mean); }
            if (warm >= 16 && mean > 0 && v > mean * 1.8) { const uint64_t n = ++g_flashes;
                if (n <= 32 || !(n & (n - 1))) { VKLOG("flashdiag: DLSS input %.2fx brighter than the frames before at present %llu (largest component %g; render %u wide; %llu so far; nanguard %s)", v / mean, (unsigned long long)g_tex_frame,
                    (double)in_max, g_odl.slot_rw[s], (unsigned long long)n, getenv("DM_VK_NAN_GUARD") && atoi(getenv("DM_VK_NAN_GUARD")) == 0 ? "off" : "on"); pending = 1; } }
            // (the mean follows every frame, the flagged ones too: a lasting change, from a dark place into the daylight 13x brighter, left it behind and every
            // frame after counted as a flash, 600 a window in the 01:22 run)
            mean = warm ? mean * 0.9 + v * 0.1 : v; if (warm < 16) warm++;
        }
        {   // flashdiag2 (NPARK 11-22: a flash seen that the input's check did not catch, "another kind"): the same check on DLSS's output (the out pass: the
            // sum of min(r+g+b, 3) x 64 over the picture, per pixel), with whether the input jumped too and whether DLSS's history was reset for that frame: an
            // output flash with a steady input is DLSS's (a history reset, its auto exposure), one with both is the scene's
            static double om, im2; static int ow_, pend2; static uint64_t oflashes;
            const double opx = (double)g_odl.slot_ow[s] * g_odl.slot_oh[s], ipx = (double)g_odl.slot_w[s] * g_odl.slot_h[s];
            const double ov = opx > 0 ? (double)w[6] / opx : 0, iv = ipx > 0 ? (double)w[7] / ipx : 0;
            if (pend2) { pend2 = 0; VKLOG("flashdiag2: the frame after: output %s (%.2f a pixel, mean %.2f)", ov < om * 1.3 ? "back to the mean: a one-frame output flash" : "still bright: a lasting change", ov, om); }
            if (ow_ >= 16 && om > 0 && ov > om * 1.8) { const uint64_t n = ++oflashes;
                if (n <= 32 || !(n & (n - 1))) { VKLOG("flashdiag2: DLSS output %.2fx brighter than the frames before at present %llu, its input %.2fx (%s), DLSS history %s this frame (largest output %g; %llu so far)",
                    ov / om, (unsigned long long)g_tex_frame, im2 > 0 ? iv / im2 : 0.0, im2 > 0 && iv > im2 * 1.8 ? "the scene jumped too" : "the input steady: DLSS's own", g_odl.slot_reset[s] ? "reset" : "kept", (double)out_max, (unsigned long long)n); pend2 = 1; } }
            om = ow_ ? om * 0.9 + ov * 0.1 : ov; im2 = ow_ ? im2 * 0.9 + iv * 0.1 : iv; if (ow_ < 16) ow_++;   // (every frame, as the input's)
        }
        g_odl.last_in_sig = w[7]; g_odl.last_out_sig = w[6]; g_odl.have_sig = 1;
        if (in_max > 0.0f && !(out_max > 0.0f)) { g_odl.force_reset = 1; const uint64_t n = ++g_odl.blacks;
            if (n <= 16 || !(n & (n - 1))) VKLOG("ootdlss: DLSS output black (largest component %g, input's %g) at present %llu: its history starts again (%llu so far)", (double)out_max, (double)in_max, (unsigned long long)g_tex_frame, (unsigned long long)n); }
    }
}
static odl_size *ootdl_size(uint32_t iw, uint32_t ih, uint32_t ow, uint32_t oh) {   // an output size's images
    // srmode (11-21: OoT's dynamic resolution steps through 10 and more render sizes; with one image set per render size, 6 kept, every step made images
    // again behind a vkDeviceWaitIdle): the render-size images are made once at the output's size and each render size uses their top left part
    // (the passes take the render size, Streamline its extent); only another output size makes a new set
    iw = ow; ih = oh;
    odl_size *e = NULL, *lru = &g_odl.sz[0];
    for (int i = 0; i < 6; i++) { odl_size *s = &g_odl.sz[i]; if (s->img[0] && s->iw == iw && s->ih == ih && s->ow == ow && s->oh == oh) { s->used = g_tex_frame; return s; }
        if (!s->img[0] && !e) e = s; if (s->used < lru->used) lru = s; }
    if (!e) { vkDeviceWaitIdle(V.dev); ootdl_free(lru); e = lru; }
    for (int i = 0; i < ODL_NIMG; i++) { const int out = i == ODL_OUT;
        if (!ootdl_img(out ? ow : iw, out ? oh : ih, k_odl_fmt[i], &e->img[i], &e->mem[i], &e->view[i])) { VKRED_ONCE(191, "ootdlss: its %ux%u images failed: the network runs", iw, ih); vkDeviceWaitIdle(V.dev); ootdl_free(e); return NULL; } }
    e->iw = iw; e->ih = ih; e->ow = ow; e->oh = oh; e->fresh = 1; e->cur = 0; e->dist_ok = 0; e->used = g_tex_frame;
    VKLOG("ootdlss: images for %ux%u -> %ux%u made", iw, ih, ow, oh); return e;
}
static float odl_halton(int i, int b) { float f = 1, r = 0; for (; i > 0; i /= b) { f /= (float)b; r += f * (float)(i % b); } return r; }
static int odl_reprojection(const float *cur, const float *prev, float out[16]) {   // dlss_math.h reprojection: prev * inverse(cur), row major
    double a[4][8]; for (int i = 0; i < 16; i++) if (!isfinite(cur[i]) || !isfinite(prev[i])) return 0;
    for (int r = 0; r < 4; r++) for (int c = 0; c < 8; c++) a[r][c] = c < 4 ? cur[r * 4 + c] : (c - 4 == r ? 1.0 : 0.0);
    for (int c = 0; c < 4; c++) { int p = c; for (int r = c + 1; r < 4; r++) if (fabs(a[r][c]) > fabs(a[p][c])) p = r;
        if (fabs(a[p][c]) < 1e-12) return 0;
        if (p != c) for (int k = 0; k < 8; k++) { double t = a[p][k]; a[p][k] = a[c][k]; a[c][k] = t; }
        const double inv = 1.0 / a[c][c]; for (int k = 0; k < 8; k++) a[c][k] *= inv;
        for (int r = 0; r < 4; r++) { if (r == c || a[r][c] == 0.0) continue; const double f = a[r][c]; for (int k = 0; k < 8; k++) a[r][k] -= f * a[c][k]; } }
    for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) { double v = 0; for (int k = 0; k < 4; k++) v += prev[r * 4 + k] * a[k][c + 4];
        if (!isfinite(v) || fabs(v) > 3.0e38) return 0; out[r * 4 + c] = (float)v; }
    return 1;
}
static int ootdl_dispatch(const vkshader_t *cs) {   // 1: DLSS wrote the network's output, the dispatch is not made
    if (g_wk) return 0;
    { static uint64_t rf = ~0ull; if (rf != g_tex_frame) { rf = g_tex_frame; ootdl_resfix_tick(); } }   // resfix: once a present, replay thread
    if (!ootdl_on() || g_odl.off_run) return 0;
    if (g_odl.jit_live && g_tex_frame > g_odl.last_eval + 2) ootdl_jitter(0, 0, 0);   // (no evaluation lately: the game's own projection again)
    // ootdlss2c (23:00, 23:13 and 23:23 runs: no DLSS evaluation): the game has an upscale network for each render share (Longshot vk_draw.cpp temporal_stand_in,
    // "by their shape"), and a network can sample more than one texture, so the network is a dispatch whose only storage image is a B10G11R11 picture (the
    // 2560x1440 0x3f layer) and one of whose sampled textures is smaller than it with its shape: the frame's scene size when the camera gives one.
    // ootdlss2d (NPARK 11-21): the formats are the textures' own (an image the shader declares R32_UINT, ootu32, or a texture view in R32_UINT over it, is
    // still its E5B9G9R9 or B10G11R11 texture), and a texture made R32_UINT itself counts as a candidate (logged; DLSS takes only a float input). The
    // 11-21 upsprobe (present 3601): d99fbeeb53d38847 samples tex2.0 854x480 nvn 0x40 (E5B9G9R9) and writes img3.0 2560x1440 nvn 0x3f: one image, B10G11R11,
    // >= 640 wide, its texture E5 and smaller with the same shape (854 / 480 = 1.779, 2560 / 1440 = 1.778): it passes, and a335c1db6270c355 (its image E5) does not.
    // Diagnostics, once per shader key and input: every dispatch that writes an E5B9G9R9 image or a B10G11R11 image 1920 or wider, with every texture it samples
    eden_pipe *ep = cs->eden ? eden_compute((vkshader_t *)cs) : NULL; if (!ep) return 0;
    const nvn_eden_bind *bi = NULL; int ni = 0;
    for (uint32_t i = 0; i < ep->pi.num_binds; i++) { const nvn_eden_bind *b = &ep->pi.binds[i]; if (b->kind == NVN_EB_IMG) { ni += (int)b->count; bi = b; } }
    if (!ni) return 0;
    // ootdlss2e (23:30 run: the network found with its image read as its texture): vk_tex_by_id hands out pointers into one per-thread slot (t_treg), which the
    // next lookup overwrites, so every texture and view is copied at once (as ups_probe does). Before this, the image found first was the texture read last
    tex_t tout_c; view_t vout_c; int vout_has = 0;
    { const tex_t *t0 = NULL; const view_t *v0 = NULL; if (!vk_tex_by_id((int)(eden_handle(5, bi, 0) & 0xfffff), &v0, &t0) || !t0) return 0; tout_c = *t0; if (v0) { vout_c = *v0; vout_has = 1; } }
    const tex_t *tout = &tout_c; const view_t *vout = vout_has ? &vout_c : NULL;
    const vkfmt_t *fo = vk_fmt(tout->d.format); const VkFormat ofmt = fo ? fo->vk : VK_FORMAT_UNDEFINED;
    const int layer_like = (ofmt == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || ofmt == VK_FORMAT_R32_UINT) && tout->d.width >= 640;
    const int diag = ofmt == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 || (layer_like && tout->d.width >= 1920);
    if (!layer_like && !diag) return 0;
    const uint32_t ow = tout->d.width, oh = tout->d.height;
    float c[ODL_CAM / 4]; uint32_t sw = 0, sh = 0; tex_t dep; view_t dview; int has_dview = 0;
    pthread_mutex_lock(&g_odc.mu); const int have = g_odc.valid && g_odc.frame == g_tex_frame;
    if (have) { memcpy(c, g_odc.c, sizeof c); sw = g_odc.w; sh = g_odc.h; dep = g_odc.dep; has_dview = g_odc.has_dview; dview = g_odc.dview; }
    const char *rej = g_odc.reject_frame == g_tex_frame ? g_odc.reject : NULL;
    pthread_mutex_unlock(&g_odc.mu);
    tex_t tin_c; view_t vin_c; int vin_has = 0; const tex_t *tin = NULL; const view_t *vin = NULL; char tl[480]; int tn = 0; tl[0] = 0; uint32_t sig = 0;
    for (uint32_t i = 0; i < ep->pi.num_binds; i++) { const nvn_eden_bind *b = &ep->pi.binds[i]; if (b->kind != NVN_EB_TEX) continue;
        for (uint32_t k = 0; k < b->count && k < 4; k++) { const tex_t *t = NULL; const view_t *v = NULL;
            if (!vk_tex_by_id((int)(eden_handle(5, b, k) & 0xfffff), &v, &t) || !t) continue;
            const int vf = v && v->set_format ? v->format : -1;
            if (tn < (int)sizeof tl - 64) tn += snprintf(tl + tn, sizeof tl - (size_t)tn, " %ux%u nvn %#x%s", t->d.width, t->d.height, (unsigned)t->d.format, vf >= 0 ? (vk_fmt(vf) && vk_fmt(vf)->vk == VK_FORMAT_R32_UINT ? " (R32_UINT view)" : " (format view)") : "");
            sig = sig * 31u + t->d.width;
            const vkfmt_t *f = vk_fmt(t->d.format); const double iw = t->d.width, ih = t->d.height;
            if (!layer_like || !f || (f->vk != VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 && f->vk != VK_FORMAT_B10G11R11_UFLOAT_PACK32 && f->vk != VK_FORMAT_R32_UINT) || ow <= t->d.width || oh <= t->d.height ||
                (double)ow * ih > 1.03 * (double)oh * iw || (double)oh * iw > 1.03 * (double)ow * ih) continue;
            if (!tin || (have && t->d.width == sw && t->d.height == sh)) { tin_c = *t; tin = &tin_c; vin_has = v != NULL; if (v) vin_c = *v; vin = vin_has ? &vin_c : NULL; } } }
    { static uint64_t cand[96]; static uint32_t cs2[96]; static int ncand; int q = 0; while (q < ncand && !(cand[q] == cs->key && cs2[q] == sig)) q++;
      if (q == ncand && ncand < 96) { cand[ncand] = cs->key; cs2[ncand++] = sig;
          VKLOG("ootdlss diag: dispatch %016llx (%u images) writes %ux%u nvn %#x%s, samples:%s -> %s (present %llu, scene %ux%u, #%d)", (unsigned long long)cs->key, ni, ow, oh, (unsigned)tout->d.format,
                vout && vout->set_format ? " (through a format view)" : "", tn ? tl : " nothing", tin ? (ni == 1 ? "an upscale network" : "network-shaped but more than one image") : "not a network",
                (unsigned long long)g_tex_frame, sw, sh, ncand); } }
    if (!tin || ni != 1) return 0;
    if (!have) { char w[200]; snprintf(w, sizeof w, "no scene camera this frame%s%s", rej ? ": the camera check failed, " : "", rej ? rej : ""); ootdl_why(w); return 0; }
    if (tin->d.width != sw || tin->d.height != sh) { ootdl_why("the network's input is not this frame's scene size"); return 0; }
    { const vkfmt_t *f = vk_fmt(tin->d.format); if (!f || f->vk == VK_FORMAT_R32_UINT || ofmt != VK_FORMAT_B10G11R11_UFLOAT_PACK32) { ootdl_why("the network's input or output texture is R32_UINT itself (DLSS takes and writes float images only)"); return 0; } }
    if (!ootdl_ready()) { ootdl_why("its passes could not be made"); return 0; }
    vk_end_pass_land();
    // views in the textures' own formats: a view_t that names another format (an R32_UINT view of the E5 or B10G11R11 image) is taken without it
    view_t vin0, vout0; const view_t *vi = vin, *vo = vout;
    if (vin && vin->set_format) { vin0 = *vin; vin0.set_format = 0; vi = &vin0; }
    if (vout && vout->set_format) { vout0 = *vout; vout0.set_format = 0; vo = &vout0; }
    vkimg_t *iim = NULL, *dim = NULL, *oim = NULL;
    const VkImageView iv = vk_tex_view(tin, vi, 1, 0, VK_VIEW_SAMPLED, &iim);
    const VkImageView dv = vk_tex_view(&dep, has_dview ? &dview : NULL, 1, 0, VK_VIEW_SAMPLED, &dim);
    const int want0 = g_vk_storage_want; g_vk_storage_want = 0;
    const VkImageView ov = vk_tex_view(tout, vo, 1, 0, VK_VIEW_STORAGE, &oim);
    g_vk_storage_want = want0;
    if (!iv || !dv || !ov || !iim || !dim || !oim || !(dim->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || dim->w != sw || dim->h != sh || oim->fmt != VK_FORMAT_B10G11R11_UFLOAT_PACK32 ||
        oim->samples > 1 || dim->samples > 1) { ootdl_why("a view of the input, the depth or the output is not as expected"); return 0; }
    odl_size *e = ootdl_size(sw, sh, ow, oh); if (!e) { ootdl_why("its images could not be made"); return 0; }
    // srfit (nvn_streamline.cpp nvn_sr_fit): the size DLSS is given, the render size raised to its mode's smallest when under it; the prep pass resamples
    // to it, and the jitter, the motion and DLSS's extents are in its pixels. Logged when it changes
    uint32_t fit[2]; nvn_sr_fit(sw, sh, ow, oh, fit); const uint32_t dw = fit[0], dh = fit[1];
    { static uint32_t lw, lh; if (lw != dw || lh != dh) { if (dw != sw || dh != sh) VKLOG("ootdlss: render %ux%u given to DLSS as %ux%u (its mode's smallest size)", sw, sh, dw, dh); lw = dw; lh = dh; } }
    VkCommandBuffer cb = vk_cmd();
    ck_mark(cb, 3, cs->key, 0, ow, oh);   // gpuck
    wb_use(iim, WB_COPY); wb_use(dim, WB_COPY); wb_use(oim, WB_OTHER);
    vk_tex_prepare(cb, iim); vk_tex_prepare(cb, dim); vk_tex_prepare(cb, oim);
    const uint64_t lu = mtl_last_fence() + 1; iim->last_use = lu; dim->last_use = lu; oim->last_use = lu;
    if (e->fresh) { e->fresh = 0; VkImageMemoryBarrier ib[ODL_NIMG];
        for (int i = 0; i < ODL_NIMG; i++) ib[i] = (VkImageMemoryBarrier){VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                                     VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, e->img[i], {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, ODL_NIMG, ib); }
    vk_img_barrier_all(cb);
    // the camera (Longshot temporal_note_scene_draw / temporal_upscale): this frame's jittered view-projection, its jitter, the frame before's without its jitter
    const float *vp = c + 0x30 / 4; const float jit[2] = {-c[0x78 / 4], -c[0x88 / 4]};
    odl_prep_pc pc; memset(&pc, 0, sizeof pc);
    const int motion = g_odl.prev_ok && g_odl.prev_w == sw && g_odl.prev_h == sh && g_tex_frame - g_odl.prev_frame <= 2;
    int have_m = 0;
    if (motion) { float prev[16]; memcpy(prev, g_odl.prev_vp, sizeof prev);
        for (int k = 0; k < 4; k++) { prev[k] -= g_odl.prev_jit[0] * prev[12 + k]; prev[4 + k] -= g_odl.prev_jit[1] * prev[12 + k]; }   // dlss_math unjitter
        have_m = odl_reprojection(vp, prev, pc.m); }
    pc.jit[0] = jit[0]; pc.jit[1] = jit[1]; pc.size[0] = (int32_t)dw; pc.size[1] = (int32_t)dh;
    pc.dz[0] = c[0x98 / 4]; pc.dz[1] = c[0x9c / 4]; pc.dz[2] = c[0xa8 / 4]; pc.dz[3] = c[0xac / 4];   // the projection's [2][2] [2][3] [3][2] [3][3]
    // NPARK 11-21 (1): HDR flag and exposure. The network's input holds the square root of the scene colour, which goes past 1 in the highlights (E5B9G9R9): DLSS
    // gets it squared back to linear light, unclamped, as a feature made for HDR colour with its own exposure (nvn_sr_evaluate flags 1 colorBuffersHDR, 64
    // useAutoExposure; Longshot's HDR engine path), and the out pass writes its linear picture as it is
    const int hdr = 1;
    if (e->dw != dw || e->dh != dh) { e->dist_ok = 0; e->dw = dw; e->dh = dh; }   // (the frame before's distances are of another render size)
    pc.flags = (have_m ? 1 : 0) | 2 | (have_m && e->dist_ok ? 4 : 0) | (hdr ? 8 : 0); pc.zmask = 0.65f;   // (NX2navi's default bias, NX2NAVI_DLSS_CHAR_MASK 65)
    const int dprev = e->cur ? ODL_DIST1 : ODL_DIST0, dcur = e->cur ? ODL_DIST0 : ODL_DIST1;
    const uint32_t slot = (uint32_t)(g_odl.runs++ % ODL_SLOTS);   // NaN guard: this evaluation's stats slot, zeroed first
    g_odl.slot_fence[slot] = 0; vkCmdFillBuffer(cb, g_odl.stat, (VkDeviceSize)slot * 32, 32, 0); vk_img_barrier_all(cb);
    const VkDescriptorBufferInfo sbi = {g_odl.stat, (VkDeviceSize)slot * 32, 32};
    { VkDescriptorSet set = vk_alloc_set(g_odl.dsl[0]);
      VkDescriptorImageInfo ii[8] = {{g_odl.linear, iv, VK_IMAGE_LAYOUT_GENERAL}, {g_odl.nearest, dv, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, e->view[ODL_COLOR], VK_IMAGE_LAYOUT_GENERAL},
          {VK_NULL_HANDLE, e->view[ODL_DEPTH], VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, e->view[ODL_MV], VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, e->view[ODL_MASK], VK_IMAGE_LAYOUT_GENERAL},
          {VK_NULL_HANDLE, e->view[dprev], VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, e->view[dcur], VK_IMAGE_LAYOUT_GENERAL}};
      VkWriteDescriptorSet wr[9]; for (uint32_t i = 0; i < 8; i++) wr[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, i, 0, 1, i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ii[i], NULL, NULL};
      wr[8] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &sbi, NULL};
      vkUpdateDescriptorSets(V.dev, 9, wr, 0, NULL);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_odl.pipe[0]); vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_odl.pl[0], 0, 1, &set, 0, NULL);
      vkCmdPushConstants(cb, g_odl.pl[0], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc); vkCmdDispatch(cb, (dw + 7) / 8, (dh + 7) / 8, 1); }
    vk_img_barrier_all(cb);
    nvn_fg_image res[6]; memset(res, 0, sizeof res);
    const int roles[5] = {ODL_COLOR, ODL_DEPTH, ODL_MV, ODL_OUT, ODL_MASK};   // nvn_sr_evaluate: colour, depth, motion, output, bias mask (no exposure)
    for (int i = 0; i < 5; i++) { const int k = roles[i]; const int out = k == ODL_OUT;
        res[i] = (nvn_fg_image){e->img[k], e->mem[k], e->view[k], out ? ow : e->iw, out ? oh : e->ih, (uint32_t)k_odl_fmt[k], VK_IMAGE_LAYOUT_GENERAL,   // (the image's size: the render size is the extent)
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0, 0, 0}; }
    const float mvs[2] = {1.0f, 1.0f};   // (motion in render pixels: nvn_sr_evaluate divides by the input size)
    const float jpx[2] = {jit[0] * 0.5f * (float)dw, -jit[1] * 0.5f * (float)dh};   // dlss_math jitter_to_pixels: render pixels, x right, y down
    ootdl_stats();
    const int reset = !have_m || g_odl.force_reset;
    if (reset && g_odl.evals > 2) { static uint64_t nr; const uint64_t n = ++nr; if (n <= 32 || !(n & (n - 1))) VKLOG("flashdiag2: DLSS history reset at present %llu (%s; %llu so far)", (unsigned long long)g_tex_frame, g_odl.force_reset ? "asked: a black or standing output" : !g_odl.prev_ok ? "no frame before" : g_odl.prev_w != sw || g_odl.prev_h != sh ? "the render size changed" : g_tex_frame - g_odl.prev_frame > 2 ? "no evaluation in the last presents" : "the camera reprojection failed", (unsigned long long)n); }
    g_odl.force_reset = 0;
    // D2 (NPARK 11-21: DLSS-G from D1's inputs, in place of optical-flow interpolation): the frame's camera and its depth and motion go to Streamline's DLSS-G
    // (nvn_streamline.cpp nvn_fg_camera / nvn_fg_tag; sealed at the present, gfx_vk_present.c nvn_fg_seal; BeforePresent then turns DLSS-G on for it and
    // OpticalTarget leaves the frame alone). Its matrices are row-vector ones (gfx_vk_fg_math.h): the projection transposed with this frame's jitter taken off
    // (P[0][2], P[1][2] = 0, Longshot's "projection without this frame's jitter"), the camera-to-world the inverse of the view (rows 0-2 of the block). Before
    // the SR evaluation, so the tag's frame token is the one SR evaluates with (nvn_sr_evaluate takes pending.token). Only while frame generation is on
    // (nvn_fg_collecting); the tags are copied when made (eOnlyValidNow)
    if (nvn_fg_collecting()) {
        fg_matrix V = {{c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], 0, 0, 0, 1}}, Vt, P;
        for (int i = 0; i < 4; i++) for (int k = 0; k < 4; k++) { Vt.v[i * 4 + k] = V.v[k * 4 + i]; P.v[i * 4 + k] = c[0x70 / 4 + k * 4 + i]; }
        P.v[8] = 0; P.v[9] = 0;   // (the transposed P[0][2], P[1][2]: the jitter)
        const fg_matrix W = fg_inverse_rigid(Vt);
        nvn_fg_camera(W.v, P.v);
        const nvn_fg_image fd = res[1], fm = res[2];
        const float one[2] = {1.0f, 1.0f};   // (motion in render pixels: nvn_fg_tag divides by the size)
        nvn_fg_tag(cb, &fd, &fm, dw, dh, one, jpx, reset, 1);
        vk_img_barrier_all(cb);
        g_odl.fg_tags++;
    }
    const int r = nvn_sr_evaluate(cb, res, dw, dh, ow, oh, mvs, jpx, 1.0f, hdr ? (1u | 64u) : 0u, reset, cs->key);
    vk_img_barrier_all(cb);
    g_cmd_gen++;   // (Streamline recorded into this command buffer: the bind caches start again)
    // the camera kept for the next frame (with its jitter), whatever DLSS did
    memcpy(g_odl.prev_vp, vp, sizeof g_odl.prev_vp); g_odl.prev_jit[0] = jit[0]; g_odl.prev_jit[1] = jit[1]; g_odl.prev_frame = g_tex_frame; g_odl.prev_w = sw; g_odl.prev_h = sh; g_odl.prev_ok = 1;
    e->cur ^= 1; e->dist_ok = 1;
    if (r != 0) { ootdl_jitter(0, 0, 0); char w[96]; snprintf(w, sizeof w, "Streamline's DLSS evaluation returned %d", r); ootdl_why(w); return 0; }
    { VkDescriptorSet set = vk_alloc_set(g_odl.dsl[1]);
      VkDescriptorImageInfo ii[2] = {{g_odl.linear, e->view[ODL_OUT], VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, ov, VK_IMAGE_LAYOUT_GENERAL}};
      VkWriteDescriptorSet wr[3] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii[0], NULL, NULL},
                                    {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ii[1], NULL, NULL},
                                    {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &sbi, NULL}};
      vkUpdateDescriptorSets(V.dev, 3, wr, 0, NULL);
      const int32_t sz[3] = {(int32_t)ow, (int32_t)oh, hdr ? 0 : 1};   // size, square (0: DLSS's picture is linear already)
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_odl.pipe[1]); vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_odl.pl[1], 0, 1, &set, 0, NULL);
      vkCmdPushConstants(cb, g_odl.pl[1], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof sz, sz); vkCmdDispatch(cb, (ow + 7) / 8, (oh + 7) / 8, 1); }
    vk_img_barrier_all(cb);
    { VkBufferCopy bc = {(VkDeviceSize)slot * 32, (VkDeviceSize)slot * 32, 32}; vkCmdCopyBuffer(cb, g_odl.stat, g_odl.stat_rb, 1, &bc);   // NaN guard: the stats to the host (ootdl_stats)
      VkMemoryBarrier hb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
      g_odl.slot_fence[slot] = mtl_last_fence() + 1; g_odl.slot_w[slot] = dw; g_odl.slot_h[slot] = dh; g_odl.slot_rw[slot] = sw; g_odl.slot_ow[slot] = ow; g_odl.slot_oh[slot] = oh; g_odl.slot_reset[slot] = (uint32_t)reset; }
    vk_tex_content_changed(oim); oim->gpu_w = 1; vk_tex_gpu_mark(tout);
    g_odl.evals++; g_odl.last_eval = g_tex_frame; g_odl.why[0] = 0;
    ootdl_resfix_check(sw, sh, ow, oh);   // resfix: the scene at the fixed share
    g_odl.info[0] = sw; g_odl.info[1] = sh; g_odl.info[2] = dw; g_odl.info[3] = oh;
    // the next jitter (dlss_math dlss_jitter_pixels / dlss_jitter_clip): Halton (2, 3) over 8 x (output / render) area phases, 8 to 64, in clip units of the render size
    { const double area = (double)ow * oh / ((double)sw * sh); int ph = (int)(8.0 * area + 0.5); ph = ph < 8 ? 8 : ph > 64 ? 64 : ph;
      const int i = (int)(g_odl.jit_n++ % (uint64_t)ph) + 1; ootdl_jitter(2.0f * (odl_halton(i, 2) - 0.5f) / (float)sw, 2.0f * (odl_halton(i, 3) - 0.5f) / (float)sh, 1); }
    if (g_odl.evals <= 2 || !(g_odl.evals % 600))
        VKLOG("ootdlss: DLSS %ux%u -> %ux%u at present %llu (%llu evaluations, the network ran %llu times), jitter %+.3f %+.3f px%s, motion %s, jitter sites %s; DLSS-G inputs tagged %llu (frame generation %s)",
              sw, sh, ow, oh, (unsigned long long)g_tex_frame, (unsigned long long)g_odl.evals, (unsigned long long)g_odl.falls, jpx[0], jpx[1], reset ? " (new history)" : "",
              have_m ? "reprojected" : "none", dm_fps_state && !memcmp(dm_fps_state + 0xe4, "OJIT", 4) ? "in" : "not in", (unsigned long long)g_odl.fg_tags, nvn_fg_collecting() ? "on" : "off");
    return 1;
}
#endif
static void dispatch_(const p_dispatch *d);
// dtrace (11-22, XCDE title, diagnostic): DM_VK_TRACE_DISPATCH=<compute shader key, hex>[#n] (live): the next n (1) dispatches of that shader are traced as a traced
// draw is (every texture and image it is given, with the image each resolves to)
static void dispatch(const p_dispatch *d) {
    static char last[48]; static uint64_t key; static int left; static long gen = -1;
    if (gen != dm_env_gen) {   // read again only when an env line changed (DM_CTL): no getenv per dispatch on the default path
        gen = dm_env_gen; const char *e = getenv("DM_VK_TRACE_DISPATCH");
        if (e && *e && strcmp(e, last)) { snprintf(last, sizeof last, "%s", e); key = strtoull(e, NULL, 16); const char *h = strchr(e, '#'); left = h ? atoi(h + 1) : 1; } }
    const int tr = left > 0 && S.sh[5] && S.sh[5]->key == key;
    if (tr) { left--; g_tracing = 1; VKLOG("=== traced dispatch %016llx %dx%dx%d after pass %d (present %llu)", (unsigned long long)key, d->x, d->y, d->z, g_sub_pass - 1, (unsigned long long)g_tex_frame); }
    g_in_dispatch = 1; dispatch_(d); g_in_dispatch = 0;   // xwatch
    if (tr) g_tracing = 0;
}
static void dispatch_(const p_dispatch *d) {
    vkshader_t *cs = S.sh[5];
    if (!cs) { VKRED_ONCE(60, "dispatch with no compute shader in the bundle: skipped"); return; }
    g_vkst.dispatches++;
    g_dispatch[0] = (uint32_t)d->x; g_dispatch[1] = (uint32_t)d->y; g_dispatch[2] = (uint32_t)d->z; g_dispatch_ind = d->ind != 0; g_dispatch_indaddr = d->ind;   // r25: before ndsr_blit, which logs it
    r140_cs(cs, d);   // r140
    ootcs_slots(cs, d);   // ootcs
    ootd_dispatch(cs);   // ootdlss D1a survey (DM_VK_OOTDLSS_SURVEY=1, log only)
    if (ndsr_blit(cs)) return;
    const int rd = g_rdiag > 0; if (rd) rd_start();   // replaydiag: 20 pass end, 21 prepare (Eden payload, staging, first stores), 22 barrier before, 23 bind, 24 dispatch, 25 after
    vk_end_pass_land();
    if (rd) rd_mark(20);
    VkCommandBuffer cb = vk_cmd();
    if (g_ind_c0 < 0) { const char *e = getenv("DM_VK_IND_C0"); g_ind_c0 = !e || atoi(e) != 0; }
    // NXN1X lane B: Eden's compute pipeline and set 0 (gfx_vk_eden.inc), prepared before the barrier as stage_set is
    int eden = vk_eden_mode(); eden_cs_t ec; VkPipelineLayout layout = VK_NULL_HANDLE; VkDescriptorSet set = VK_NULL_HANDLE; VkPipeline p = VK_NULL_HANDLE; uint32_t dofs[96];
    if (eden) { g_drv_fresh = d->ind && g_ind_c0; g_drv_lb = VK_NULL_HANDLE; int ok = eden_dispatch_prep(cs, &ec); g_drv_fresh = 0; if (!ok) { ndsr_trace_dispatch(2, "pipeline or descriptors unavailable"); ootcs_dropped(cs, "Eden pipeline or descriptors unavailable"); return; }
#if defined(DM_NXN1X)
        // ootdlss: after the payload (23:13 run: before it, eden_handle's evaluated handles read a stale query state and every one-texture one-image dispatch
        // showed its texture as its 2560x1440 image, so the network was never found); a DLSS that wrote the output returns as a failed preparation does
        if (ootdl_dispatch(cs)) return;
#endif
        ck_mark(cb, 2, cs->key, 0, g_dispatch[0], g_dispatch[1]);   // gpuck
        vk_hy3_pre(cb);   // stage1/2s: the window closed if this dispatch cannot use it, its new staged ranges copied in (the barrier below orders the copies before it)
        eden_first_stores(cb, &ec.ep->pi, cs->key); }   // bak86: before the barrier below, which orders them before the dispatch (into staging where it is staged: vk_hy3_redirect)
    else {
    VkDescriptorSetLayout dsl[5] = {cs->dsl}; layout = vk_pipeline_layout(dsl, 1);
    g_drv_fresh = d->ind && g_ind_c0; g_drv_lb = VK_NULL_HANDLE; set = stage_set(5, cs, dofs); g_drv_fresh = 0;   // r75
    uint64_t ck[3] = {cs->key, g_num_um, g_num_im};
    p = nvn_hm_get(&g_cpipes, ck);
    if (!p) { uint64_t t80 = tnow();   // r80
        VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, vk_shader_module(cs, ck[1], ck[2], 0, 0, 0, 0), "main", NULL}, layout, VK_NULL_HANDLE, -1};
        if (vkCreateComputePipelines(V.dev, V.pcache, 1, &ci, NULL, &p)) { LOG("[RED] [vk] compute pipeline failed: %016llx", (unsigned long long)cs->key); p = (VkPipeline)1; }
        nvn_hm_put(&g_cpipes, ck, p); r80_made(t80, NULL, (const struct vkshader *)cs);   // r80
    }
    if (p == (void *)1) return;
    }
    if (rd) rd_mark(21);
    // r89 (DM_VK_BAR bit 2): a dispatch right after the previous one (no full barrier recorded since its own, so no pass and no copy between) is ordered after it and after
    // transfers (uploads stage_set may have recorded) by a compute-only barrier; the previous dispatch's barrier already made its writes visible to everything. Otherwise
    // the full barrier as before. nvn_mtl.m dispatch_enc: Age of Calamity's ~1,060 small dispatches a frame had the Mac GPU at 16 ms instead of 6.7 with a full drain each
    extern _Thread_local uint64_t g_bar_seq; int vk_bar_mode(void); VkAccessFlags vk_dev_access(void);
    if ((vk_bar_mode() & 2) && g_disp_seq == g_bar_seq && g_disp_cb == cb) {
        VkMemoryBarrier m = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                             VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | (d->ind ? VK_ACCESS_INDIRECT_COMMAND_READ_BIT : 0)};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | (d->ind ? VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT : 0), 0, 1, &m, 0, NULL, 0, NULL);
        uint64_t n = ++g_disp_cbar; if (n <= 2 || !(n & (n - 1))) VKLOG("r89: dispatch %016llx right after a dispatch: compute-only barrier before it (%llu so far; DM_VK_BAR=1 full)", (unsigned long long)cs->key, (unsigned long long)n);
    } else vk_img_barrier_all(cb);
    if (rd) rd_mark(22);
    vk_pass_ts(0, 0, 0, 0, 0);   // DM_VK_PASS_TIMES: dispatches are timed as passes (fmt 0xC0FFEE, gfx_vk_core.c vk_pass_report)
    if (eden) { if (!eden_dispatch_bind(cb, &ec)) { ndsr_trace_dispatch(4, "descriptor bind failed"); ootcs_dropped(cs, "descriptor bind failed"); vk_pass_ts(1, (uint32_t)d->x, (uint32_t)d->y, 0xC0FFEEu, (uint32_t)cs->key); vk_hy3_post(cb); return; } }
    else {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, (uint32_t)cs->ndyn, dofs);
    }
    if (rd) rd_mark(23);   // (with the start timestamp above)
    if (d->ind) { void vk_pool_id_args(uint64_t, int, int); vk_pool_id_args(d->ind, 12, 1); }   // bak91c
    if (d->ind) { VkBuffer b; VkDeviceSize o; if (vk_buf_at(d->ind, 12, &b, &o)) {
        if (g_drv_lb) {   // r75: the GPU-written counts into the segment's c0 + 0x200 (and the NVN2 place)
            VkBufferCopy c[2] = {{o, g_drv_lo + 0x200, 12}, {o, g_drv_lo + DMSH_C0_DISPATCH_NVN2, 12}};
            vk_img_barrier_all(cb); vkCmdCopyBuffer(cb, b, g_drv_lb, dm_sdk_ounce() ? 2u : 1u, c); vk_img_barrier_all(cb);
            uint64_t n = ++g_ind_c0_n; if (n <= 2 || !(n & (n - 1))) VKLOG("r75: indirect dispatch %016llx: its group counts copied into c0 + 0x200 on the GPU (%llu so far; DM_VK_IND_C0=0 off)", (unsigned long long)cs->key, (unsigned long long)n); }
        vkCmdDispatchIndirect(cb, b, o); ndsr_trace_dispatch(8, "vkCmdDispatchIndirect"); }
        else { ndsr_trace_dispatch(16, "indirect arguments unavailable"); ootcs_dropped(cs, "indirect arguments not found"); } }
    else {
        vkCmdDispatch(cb, (uint32_t)d->x, (uint32_t)d->y, (uint32_t)d->z);
        ndsr_trace_dispatch(32, "vkCmdDispatch");
    }
    if (rd) rd_mark(24);
    { static int tsfix = -1; if (tsfix < 0) { const char *e = getenv("DM_VK_TS_FIX"); tsfix = e && *e == '1';   // tsdiag (diagnostic, default off): the end timestamp after the dispatch has finished
        if (tsfix) VKLOG("tsdiag: an execution barrier (compute -> all) before each dispatch's end timestamp: dispatch times include their whole work, gaps are real idle (DM_VK_TS_FIX=1)"); }
      if (tsfix && vk_pass_times_active()) vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 0, NULL); }
    vk_pass_ts(1, (uint32_t)d->x, (uint32_t)d->y, 0xC0FFEEu, (uint32_t)cs->key);
    if (vk_bar_mode() & 2) {   // r89: its shader writes made visible to everything after it (earlier work was ordered by the barrier before it)
        VkAccessFlags da = vk_dev_access(); VkMemoryBarrier m = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, da};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &m, 0, NULL, 0, NULL);
        g_disp_seq = g_bar_seq; g_disp_cb = cb;
    } else vk_img_barrier_all(cb);
    if (eden) vk_hy3_post(cb);   // stage1/2s: the window may be closed again by what comes next (DM_VK_HY3=1: merged here, after the barrier above)
    if (rd) rd_mark(25);   // end timestamp, barrier after, staging after
    if (crumbs_on()) { crumb_t *c = crumb_cur(); if (c->n < 8) { c->k[c->n][0] = 0; c->k[c->n][1] = 0; c->k[c->n][2] = cs->key; c->n++; } crumb_mark(1); }
    if (g_x_tracepass == -2) VKLOG("--- dispatch %016llx %dx%dx%d%s after pass %d (Eden hash %016llx)", (unsigned long long)cs->key, d->x, d->y, d->z, d->ind ? " (indirect)" : "", g_sub_pass - 1,
                                   (unsigned long long)(cs->eden ? nvn_eden_shader_hash(cs->eden) : 0));   // svcprobe (log only): Eden hash = the SPIR-V dump's name   // (DM_VK_TRACE_PASS=-2 lists the dispatches too)
}

// ---------------- copies and clears ----------------
static VkImageSubresourceLayers sub_of(vkimg_t *im, const view_t *v, int has_v, int z) {
    uint32_t lv = has_v && v->set_levels ? v->base_level : 0, ly = has_v && v->set_layers ? v->min_layer : 0;
    int layered = im->layers > 1;
    return (VkImageSubresourceLayers){im->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : (im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_STENCIL_BIT), lv, ly + (layered ? (uint32_t)z : 0), 1};
}
static void copy_b2t(const p_copy_b2t *c, int to_texture) {
    vk_end_pass_land();
    vkimg_t *im = vk_tex(c->tex); const vkfmt_t *f = vk_fmt(c->tex->d.format);
    if (!im || !f) return;
    VkBuffer b; VkDeviceSize o;
    unsigned bw = f->bw ? f->bw : 1, bh = f->bh ? f->bh : 1;
    int layered = im->layers > 1, depth = c->r.depth ? c->r.depth : 1;
    VkImageSubresourceLayers sub = sub_of(im, &c->view, c->has_view, c->r.zoffset);
    if (sub.mipLevel >= im->levels || sub.mipLevel >= 32 || im->samples != 1 ||
        c->r.width <= 0 || c->r.height < 0 || depth < 1 ||
        c->r.xoffset < 0 || c->r.yoffset < 0 || c->r.zoffset < 0) return;
    uint32_t w = c->r.width, h = c->r.height ? c->r.height : 1;
    uint32_t iw = im->w >> sub.mipLevel, ih = im->h >> sub.mipLevel, id = im->d >> sub.mipLevel;
    if (!iw) iw = 1;
    if (!ih) ih = 1;
    if (!id) id = 1;
    if ((uint64_t)c->r.xoffset + w > iw || (uint64_t)c->r.yoffset + h > ih ||
        (uint64_t)sub.baseArrayLayer + (layered ? depth : 1) > im->layers ||
        (!layered && (uint64_t)c->r.zoffset + depth > id) || c->r.xoffset % bw || c->r.yoffset % bh ||
        (w % bw && c->r.xoffset + w != iw) || (h % bh && c->r.yoffset + h != ih)) return;
    unsigned bytes = sub.aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT ? 1 :
        sub.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT ? (im->fmt == VK_FORMAT_D16_UNORM ? 2 : 4) : f->bpb;
    if (!bytes) return;
    uint64_t blocks = ((uint64_t)w + bw - 1) / bw, rows = ((uint64_t)h + bh - 1) / bh;
    uint64_t pitch = S.row_stride > 0 ? (uint64_t)S.row_stride : blocks * bytes;
    if (pitch < blocks * bytes || pitch % bytes || pitch > INT32_MAX ||
        pitch / bytes > UINT32_MAX / bw || rows > UINT64_MAX / pitch) return;
    uint64_t slice = S.image_stride > 0 ? (uint64_t)S.image_stride : rows * pitch;
    if (depth > 1 && (slice % pitch || slice / pitch < rows || slice / pitch > UINT32_MAX / bh)) return;
    uint64_t span = (rows - 1) * pitch + blocks * bytes;
    if (depth > 1 && slice > (UINT64_MAX - span) / (depth - 1)) return;
    span += (depth - 1) * slice;
    if (span > UINT64_MAX - c->src || !vk_buf_at(c->src, span, &b, &o) ||
        o % (sub.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT ? bytes : 4)) {
        VKRED_ONCE(70, "buffer/texture copy exceeds its buffer or has a misaligned offset");
        return;
    }
    uint32_t rowlen = S.row_stride > 0 ? (uint32_t)(pitch / bytes * bw) : 0;
    uint32_t imgh = S.image_stride > 0 && depth > 1 ? (uint32_t)(slice / pitch * bh) : 0;
    VkCommandBuffer cb = vk_cmd(); vk_tex_prepare(cb, im); vk_img_barrier_all(cb);
    VkBufferImageCopy r = {o, rowlen, imgh, sub, {c->r.xoffset, c->r.yoffset, layered ? 0 : c->r.zoffset},
                           {(uint32_t)c->r.width, (uint32_t)(c->r.height ? c->r.height : 1), layered ? 1 : (uint32_t)depth}};
    if (layered) r.imageSubresource.layerCount = (uint32_t)depth;
    if (g_watch_off >= 0 || g_ndsr_im) { char w[96]; snprintf(w, sizeof w, "buffer copy %s (buffer %#llx, %dx%d)", to_texture ? "destination" : "source", (unsigned long long)c->src, c->r.width, c->r.height); vk_watch(im, w); }
    wb_use(im, to_texture ? WB_OTHER : WB_COPY);
    if (to_texture) { vk_tex_content_changed(im); vkCmdCopyBufferToImage(cb, b, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r); im->gpu_w = 1; vk_tex_gpu_mark(c->tex); }   // r72 mark
    else { vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &r);
        void vk_gpu_wrote(uint64_t a, uint64_t n);
        vk_gpu_wrote(c->src, span); }
    vk_img_barrier_all(cb);
}
// Snapshot the source mip before an overlapping copy. Keeping the full mip
// extent also preserves the block-compressed edge and blit sampling coordinates.
static VkImage snapshot_copy_source(VkCommandBuffer cb, vkimg_t *im, VkImageSubresourceLayers sub,
                                    VkExtent3D extent) {
    VkImageType type = im->target == 2 ? VK_IMAGE_TYPE_3D :
                       im->target == 0 || im->target == 3 ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D;
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, type, im->fmt, extent,
        1, 1, (VkSampleCountFlagBits)im->samples, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VkImage tmp;
    if (vkCreateImage(V.dev, &ci, NULL, &tmp) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(V.dev, tmp, &req);
    VkMemoryDedicatedAllocateInfo dedicated = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .image = tmp};
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &dedicated, req.size,
        vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
    if (ai.memoryTypeIndex == UINT32_MAX) ai.memoryTypeIndex = vk_mem_type(req.memoryTypeBits, 0, 0);
    if (ai.memoryTypeIndex == UINT32_MAX) { vkDestroyImage(V.dev, tmp, NULL); return VK_NULL_HANDLE; }
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (vkAllocateMemory(V.dev, &ai, NULL, &mem) != VK_SUCCESS ||
        vkBindImageMemory(V.dev, tmp, mem, 0) != VK_SUCCESS) {
        vkDestroyImage(V.dev, tmp, NULL);
        if (mem) vkFreeMemory(V.dev, mem, NULL);
        return VK_NULL_HANDLE;
    }
    VkImageMemoryBarrier init = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, tmp, {im->aspect, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 0, NULL, 1, &init);
    VkImageCopy copy = {.srcSubresource = sub, .dstSubresource = {sub.aspectMask, 0, 0, 1}, .extent = extent};
    vkCmdCopyImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, tmp, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    vk_img_barrier_all(cb);
    vk_defer_free_image(tmp, mem);
    return tmp;
}
static void copy_t2t(const p_copy_t2t *c) {
    vk_end_pass();
    vkimg_t *s = vk_tex(c->src), *d = vk_tex(c->dst); if (d) vk_tex_gpu_mark(c->dst);   // r72
    if (!s || !d) return;
    vk_watch(s, "copy source"); vk_watch(d, "copy destination");
    VkCommandBuffer cb = vk_cmd(); vk_tex_prepare(cb, s); vk_tex_prepare(cb, d); vk_img_barrier_all(cb);
    VkImageSubresourceLayers ss = sub_of(s, &c->sv, c->has_sv, c->sr.zoffset), ds = sub_of(d, &c->dv, c->has_dv, c->dr.zoffset);
    if (c->sr.zoffset < 0 || c->dr.zoffset < 0 || c->sr.depth < 0 || c->dr.depth < 0 ||
        ss.mipLevel >= s->levels || ds.mipLevel >= d->levels || ss.mipLevel >= 32 || ds.mipLevel >= 32 ||
        ss.baseArrayLayer >= s->layers || ds.baseArrayLayer >= d->layers) return;
    // Vulkan refuses (vkEndCommandBuffer fails for the whole submission) a region outside either image or a blit the format cannot do: regions are clipped
    // to both images' levels, a copy with nothing left is skipped
    int32_t sw = (int32_t)(s->w >> ss.mipLevel ? s->w >> ss.mipLevel : 1), sh = (int32_t)(s->h >> ss.mipLevel ? s->h >> ss.mipLevel : 1);
    int32_t dw = (int32_t)(d->w >> ds.mipLevel ? d->w >> ds.mipLevel : 1), dh = (int32_t)(d->h >> ds.mipLevel ? d->h >> ds.mipLevel : 1);
    p_copy_t2t k = *c; c = &k;
    if ((k.flags & DM_COPY_RESOLVE) && s->samples == d->samples && s->fmt == d->fmt &&
        k.sr.width == k.dr.width && k.sr.height == k.dr.height) k.flags &= ~DM_COPY_RESOLVE;
    if ((k.flags & DM_COPY_RESOLVE) && (s->samples == 1 || d->samples != 1 || s->fmt != d->fmt ||
        ss.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT || ds.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT)) {
        VKRED_ONCE(75, "unsupported NVN resolve format or sample counts");
        return;
    }
    if (k.sr.height <= 0) k.sr.height = 1; if (k.dr.height <= 0) k.dr.height = 1;
    if (ss.mipLevel >= s->levels || ds.mipLevel >= d->levels || k.sr.xoffset < 0 || k.sr.yoffset < 0 || k.dr.xoffset < 0 || k.dr.yoffset < 0 || k.sr.xoffset >= sw || k.sr.yoffset >= sh || k.dr.xoffset >= dw || k.dr.yoffset >= dh) {
        VKRED_ONCE(72, "texture copy outside its images (src %dx%d @%d,%d of %dx%d level %u, dst %dx%d @%d,%d of %dx%d level %u): skipped", c->sr.width, c->sr.height, c->sr.xoffset, c->sr.yoffset, sw, sh, ss.mipLevel,
                   c->dr.width, c->dr.height, c->dr.xoffset, c->dr.yoffset, dw, dh, ds.mipLevel); vk_img_barrier_all(cb); return; }
    int same = k.sr.width == k.dr.width && k.sr.height == k.dr.height;
    if (k.sr.width > sw - k.sr.xoffset) { VKRED_ONCE(73, "texture copy source region clipped to its image"); k.sr.width = sw - k.sr.xoffset; if (same) k.dr.width = k.sr.width; }
    if (k.sr.height > sh - k.sr.yoffset) { VKRED_ONCE(73, "texture copy source region clipped to its image"); k.sr.height = sh - k.sr.yoffset; if (same) k.dr.height = k.sr.height; }
    if (k.dr.width > dw - k.dr.xoffset) { VKRED_ONCE(74, "texture copy destination region clipped to its image"); k.dr.width = dw - k.dr.xoffset; if (same) k.sr.width = k.dr.width; }
    if (k.dr.height > dh - k.dr.yoffset) { VKRED_ONCE(74, "texture copy destination region clipped to its image"); k.dr.height = dh - k.dr.yoffset; if (same) k.sr.height = k.dr.height; }
    if (k.sr.width <= 0 || k.sr.height <= 0 || k.dr.width <= 0 || k.dr.height <= 0) { vk_img_barrier_all(cb); return; }
    if (!(c->flags & DM_COPY_RESOLVE) && !(k.sr.width == k.dr.width && k.sr.height == k.dr.height && s->fmt == d->fmt)) {   // a blit: both formats must allow it (and linear filtering)
        VkFormatProperties sp, dp; vkGetPhysicalDeviceFormatProperties(V.pd, s->fmt, &sp); vkGetPhysicalDeviceFormatProperties(V.pd, d->fmt, &dp);
        const vkfmt_t *sf = vk_fmt(s->nvn_format), *df = vk_fmt(d->nvn_format);
        if (!(sp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) || !(dp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) ||
            s->samples != 1 || d->samples != 1 || s->aspect != d->aspect ||
            (s->aspect != VK_IMAGE_ASPECT_COLOR_BIT && s->fmt != d->fmt) ||
            !sf || !df || sf->is_int != df->is_int || sf->is_sint != df->is_sint) {
            VKRED_ONCE(75, "texture blit between formats %d → %d that cannot be blitted: skipped", s->fmt, d->fmt); vk_img_barrier_all(cb); return; }
        if (!(sp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) || (s->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) k.flags &= ~1u;   // nearest where linear is not allowed
    }
    if (s == d && !memcmp(&ss, &ds, sizeof ss) && k.sr.xoffset == k.dr.xoffset && k.sr.yoffset == k.dr.yoffset && k.sr.width == k.dr.width && k.sr.height == k.dr.height && k.sr.zoffset == k.dr.zoffset) {
        vk_img_barrier_all(cb); return; }   // a copy of a region onto itself (two textures over the same storage): nothing to do, and Vulkan forbids overlapping copies
    VkImage source = s->img;
    // copymirror (FE Fortune's Weave 11-22, a render target's picture upside down where the game copied it with MIRROR_Y; nvn_mtl.m 11596 walks a mirrored
    // source from its far edge): NVN_COPY_FLAGS_MIRROR_X / _Y (0x4 / 0x8) were dropped here, so the copy kept the source's orientation. A mirrored copy goes
    // through vkCmdBlitImage with the source region's edges swapped. NXN1X_COPY_MIRROR=0: as before
    static int mirror_on = -1; if (mirror_on < 0) { const char *ev = getenv("NXN1X_COPY_MIRROR"); mirror_on = !(ev && *ev == '0'); }
    const uint32_t mirror = mirror_on && !(k.flags & DM_COPY_RESOLVE) ? (uint32_t)k.flags & (NVN_COPY_FLAGS_MIRROR_X_BIT | NVN_COPY_FLAGS_MIRROR_Y_BIT) : 0;
    int blit = k.sr.width != k.dr.width || k.sr.height != k.dr.height || s->fmt != d->fmt || mirror;
    if (mirror) { static uint64_t n; if (++n <= 4 || !(n & (n - 1))) VKLOG("copymirror: texture copy %ux%u -> %ux%u mirrored%s%s (%llu so far; NXN1X_COPY_MIRROR=0 off)", k.sr.width, k.sr.height, k.dr.width, k.dr.height,
        (mirror & NVN_COPY_FLAGS_MIRROR_X_BIT) ? " X" : "", (mirror & NVN_COPY_FLAGS_MIRROR_Y_BIT) ? " Y" : "", (unsigned long long)n); }
    uint32_t sd = s->d >> ss.mipLevel, dd = d->d >> ds.mipLevel;
    if (!sd) sd = 1;
    if (!dd) dd = 1;
    uint32_t nz = s->layers > 1 || !k.sr.depth ? 1 : (uint32_t)k.sr.depth;
    if (!blit && !(k.flags & DM_COPY_RESOLVE) &&
        ((uint64_t)(s->layers > 1 ? 0 : k.sr.zoffset) + nz > sd ||
         (uint64_t)(d->layers > 1 ? 0 : k.dr.zoffset) + nz > dd)) return;
    if (!(k.flags & DM_COPY_RESOLVE) && !blit && s->samples != d->samples) return;
    if ((k.flags & DM_COPY_RESOLVE) && (k.sr.width > dw - k.dr.xoffset || k.sr.height > dh - k.dr.yoffset)) return;
    if (source == d->img && ss.mipLevel == ds.mipLevel && ss.baseArrayLayer == ds.baseArrayLayer &&
        (blit || ((int64_t)k.sr.xoffset + k.sr.width > k.dr.xoffset &&
                  (int64_t)k.dr.xoffset + k.dr.width > k.sr.xoffset &&
                  (int64_t)k.sr.yoffset + k.sr.height > k.dr.yoffset &&
                  (int64_t)k.dr.yoffset + k.dr.height > k.sr.yoffset))) {
        uint32_t depth = s->d >> ss.mipLevel;
        source = snapshot_copy_source(cb, s, ss, (VkExtent3D){sw, sh, depth ? depth : 1});
        if (!source) { VKRED_ONCE(75, "could not stage an overlapping image copy"); return; }
        ss.mipLevel = ss.baseArrayLayer = 0;
    }
    wb_use(s, WB_COPY); wb_use(d, WB_OTHER);
    vk_tex_content_changed(d); d->gpu_w = 1;
    if (c->flags & DM_COPY_RESOLVE) {
        VkImageResolve r = {ss, {c->sr.xoffset, c->sr.yoffset, 0}, ds, {c->dr.xoffset, c->dr.yoffset, 0}, {(uint32_t)c->sr.width, (uint32_t)c->sr.height, 1}};
        vkCmdResolveImage(cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    } else if (!mirror && c->sr.width == c->dr.width && c->sr.height == c->dr.height && s->fmt == d->fmt) {
        VkImageCopy r = {ss, {c->sr.xoffset, c->sr.yoffset, s->layers > 1 ? 0 : c->sr.zoffset}, ds, {c->dr.xoffset, c->dr.yoffset, d->layers > 1 ? 0 : c->dr.zoffset},
                         {(uint32_t)c->sr.width, (uint32_t)(c->sr.height ? c->sr.height : 1), s->layers > 1 ? 1 : (uint32_t)(c->sr.depth ? c->sr.depth : 1)}};
        vkCmdCopyImage(cb, source, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
        // Games can copy their final color target to a separate display texture.
        if (s->scaled_valid && s != d && d->aspect == VK_IMAGE_ASPECT_COLOR_BIT &&
            !r.dstOffset.x && !r.dstOffset.y && r.extent.width == d->w && r.extent.height == d->h &&
            vk_rescale_eligible(d) && vk_rescaled_view(d, c->dst, NULL, 0xff, 0, 1)) {
            vk_tex_prepare(cb, d->scaled);
            r.srcOffset.x *= 2; r.srcOffset.y *= 2;
            r.extent.width *= 2; r.extent.height *= 2;
            vkCmdCopyImage(cb, s->scaled->img, VK_IMAGE_LAYOUT_GENERAL,
                           d->scaled->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
            d->scaled_valid = 1;
        }
    } else {
        VkImageBlit r = {ss, {{c->sr.xoffset, c->sr.yoffset, 0}, {c->sr.xoffset + c->sr.width, c->sr.yoffset + (c->sr.height ? c->sr.height : 1), 1}},
                         ds, {{c->dr.xoffset, c->dr.yoffset, 0}, {c->dr.xoffset + c->dr.width, c->dr.yoffset + (c->dr.height ? c->dr.height : 1), 1}}};
        if (mirror & NVN_COPY_FLAGS_MIRROR_X_BIT) { const int32_t t = r.srcOffsets[0].x; r.srcOffsets[0].x = r.srcOffsets[1].x; r.srcOffsets[1].x = t; }   // copymirror
        if (mirror & NVN_COPY_FLAGS_MIRROR_Y_BIT) { const int32_t t = r.srcOffsets[0].y; r.srcOffsets[0].y = r.srcOffsets[1].y; r.srcOffsets[1].y = t; }
        vkCmdBlitImage(cb, source, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r, (c->flags & 1) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    }
    vk_img_barrier_all(cb);
}
static void end_pass_for_scale(void) {
    collect_scale = lazy_scale && pass_scale == 2 && !scale_time_mode && !g_pd_state &&
                    !g_ex_state && !g_in_state;
    vk_end_pass();
    collect_scale = 0;
}
static void clear_tex(const p_clear_tex *c) {
    end_pass_for_scale();
    vkimg_t *im = NULL;
    if (pending_scale_count && c->tex && !c->r.xoffset && !c->r.yoffset && !c->r.zoffset &&
        c->r.depth == 1 && (!c->has_view || ((!c->view.set_levels || !c->view.base_level) &&
        (!c->view.set_layers || !c->view.min_layer)))) {
        vkimg_t *resident = vk_tex_peek(c->tex);
        for (unsigned i = 0; i < pending_scale_count; ++i) {
            if (resident == pending_scale[i].im && c->r.width == (int)resident->w &&
                c->r.height == (int)resident->h && c->mask == 0xf &&
                resident->aspect == VK_IMAGE_ASPECT_COLOR_BIT &&
                !(c->tex->d.flags & NVN_TEXTURE_FLAGS_IMAGE)) {
                // Already resident and used by this very submission; a full clear
                // needs no load of the former native texels.
                im = resident;
                break;
            }
        }
    }
    materialize_scale(im);
    if (!im) im = vk_tex(c->tex);
    if (!im) return;
    vk_tex_gpu_mark(c->tex);
    vk_watch(im, "clear");   // r27
    VkCommandBuffer cb = vk_cmd(); vk_tex_prepare(cb, im); vk_img_barrier_all(cb); wb_use(im, WB_OTHER); vk_tex_content_changed(im); im->gpu_w = 1;
    VkImageSubresourceLayers l = sub_of(im, &c->view, c->has_view, c->r.zoffset);
    VkImageSubresourceRange rg = {l.aspectMask, l.mipLevel, 1, l.baseArrayLayer, im->layers > 1 ? (uint32_t)(c->r.depth ? c->r.depth : 1) : 1};
    if (c->r.xoffset || c->r.yoffset || (uint32_t)c->r.width < (im->w >> l.mipLevel) || (uint32_t)c->r.height < (im->h >> l.mipLevel)) VKRED_ONCE(71, "partial texture clear: the whole level is cleared");
    if (im->aspect & VK_IMAGE_ASPECT_COLOR_BIT) { VkClearColorValue v; memcpy(&v, c->c.u, 16); vkCmdClearColorImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &rg); }
    else { VkClearDepthStencilValue v = {c->c.f[0], c->c.u[1]}; rg.aspectMask = im->aspect; vkCmdClearDepthStencilImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &rg); }
    vk_img_barrier_all(cb);
}

// ---------------- the record walker ----------------
static int rt_same_on(const rec_state_t *s, const p_set_rt *r) {   // the targets the open pass draws into already (S1b (parallel recording): on a given state)
    int n = r->n > 8 ? 8 : r->n; if (n != s->rt_n) return 0;
    for (int i = 0; i < n; i++) {
        int has = r->color[i] != NULL && r->color[i]->magic == MAGIC('T', 'E', 'X', 0);
        if (has != s->rt_has_col[i] || (has && memcmp(r->color[i], &s->rt_col[i], sizeof s->rt_col[i]))) return 0;
        if (r->has_cview[i] != s->rt_has_cview[i] || (r->has_cview[i] && memcmp(&r->cview[i], &s->rt_cview[i], sizeof s->rt_cview[i]))) return 0;
    }
    int hd = r->depth != NULL && r->depth->magic == MAGIC('T', 'E', 'X', 0);
    if (hd != s->rt_has_dep || (hd && memcmp(r->depth, &s->rt_dep, sizeof s->rt_dep))) return 0;
    if (r->has_dview != s->rt_has_dview || (r->has_dview && memcmp(&r->dview, &s->rt_dview, sizeof s->rt_dview))) return 0;
    return 1;
}
static int rt_same(const p_set_rt *r) { return rt_same_on(&S, r); }
static void rt_assign_on(rec_state_t *s, const p_set_rt *r) {   // S1b (parallel recording): set_rt's state change (was inline there), on a given state
    s->rt_n = r->n > 8 ? 8 : r->n;
    for (int i = 0; i < s->rt_n; i++) {
        s->rt_has_col[i] = r->color[i] != NULL && r->color[i]->magic == MAGIC('T', 'E', 'X', 0); if (s->rt_has_col[i]) s->rt_col[i] = *r->color[i];
        s->rt_bad[i] = r->color[i] && !s->rt_has_col[i] ? r->color[i]->magic | 1u : 0;
        s->rt_has_cview[i] = r->has_cview[i]; if (r->has_cview[i]) s->rt_cview[i] = r->cview[i];
    }
    s->rt_has_dep = r->depth != NULL && r->depth->magic == MAGIC('T', 'E', 'X', 0); if (s->rt_has_dep) s->rt_dep = *r->depth;
    s->rt_has_dview = r->has_dview; if (r->has_dview) s->rt_dview = r->dview;
}
static const char *const g_opname[] = {"END", "JUMP", "BEGIN", "CALL", "SIGNAL_EVENT", "REPORT_COUNTER", "COPY_B2B", "UPDATE_UB", "SET_RT", "CLEAR_COLOR", "CLEAR_DS", "VIEWPORTS",
    "SCISSOR", "DEPTH_RANGE", "BIND_PROGRAM", "BIND_STATE", "BIND_VATTR", "BIND_VSTREAM", "BIND_VBUF", "BIND_UBO", "BIND_SSBO", "BIND_TEX", "BIND_SAMP", "STENCIL", "POLY_OFFSET", "DRAW",
    "DRAW_IDX", "DISPATCH", "BARRIER", "SET_TEX_POOL", "SET_SAMP_POOL", "COPY_B2T", "COPY_STRIDE", "BLEND_COLOR", "COPY_T2B", "BIND_IMG", "COPY_T2T", "RESET_COUNTER", "CLEAR_TEX", "CLEAR_BUF",
    "DEPTH_CLAMP", "RENDER_COND", "FENCE_SYNC", "TF_BIND", "TF_STATE", "RAST_DISCARD", "VP_SWIZZLE", "BIND_TEX1", "PRIM_RESTART", "BIND_UBO_SMALL", "BIND_VBUF1", "DRAW_TEX", "ALPHA_REF",
    "PUSH_CONST", "DLSS", "MULTI_DRAW", "PATCH_SIZE", "DEPTH_BOUNDS"};   // r81: nvn_obj.h's op order (oplog)
static void oplog_flush(void) {   // r81 DM_VK_OPLOG: the ops counted since the matching SET_RT
    if (!g_oplog_seg) return;
    g_oplog_seg = 0; char t[1600]; int k = 0;
    for (int i = 0; i < OP_COUNT_ && k < (int)sizeof t - 48; i++) if (g_oplog_cnt[i])
        k += snprintf(t + k, sizeof t - (size_t)k, " %s x%u", i < (int)(sizeof g_opname / sizeof *g_opname) ? g_opname[i] : "?", g_oplog_cnt[i]);
    VKLOG("oplog: ops until the next SET_RT:%s; draws dropped by a false render condition %u", k ? t : " none", g_oplog_rcskip);
}
// ootbilerp (11-20; on for Ocarina of Time by default since ootbilerp3, nvnh_rt.c nvnh_oot_upscale_standin; DM_VK_OOT_BILERP=0/1 overrides, live): Zelda Ocarina of Time (NXN1X) draws its 3D correctly up to the tone map (a 16:9 R11G11B10F target at the dynamic
// resolution: 1138x640, 953x536, 854x480), but nothing writes it into the 2560x1440 RGBA8 output the final pass blends the UI onto (the game's upscale network does not land),
// so the 3D is black. The last 16:9 R11G11B10F colour target narrower than the output is kept; when a pass then targets a wider RGBA8 / sRGB 16:9 image, it is blitted there
// bilinearly first (copy_t2t, flags 1), once per scene target. A stand-in for the network until the shader translator runs it (Eglantine PR #614)
// S1b (parallel recording): the switch and the stand-down split from oot_bilerp_rt (were its statics), so that the dry pass can tell a SET_RT that may copy (a GPU
// write: such a SET_RT stays on the replay thread). Same order as before: the switch read when the environment changed, then the stand-down, then the stand-in
static long g_oob_gen = -1; static int g_oob_on, g_oob_forced, g_oob_have;   // g_oob_have: a scene target is kept (was oot_bilerp_rt's have)
static int oot_bilerp_live(void) {   // 1: the stand-in may act at a SET_RT
    if (g_oob_gen != dm_env_gen) { g_oob_gen = dm_env_gen; const char *e = getenv("DM_VK_OOT_BILERP"); int nvnh_oot_upscale_standin(void); g_oob_forced = e && *e && atoi(e) != 0; g_oob_on = e && *e ? g_oob_forced : nvnh_oot_upscale_standin(); if (!g_oob_on) g_oob_have = 0; }
    // ootu32: once the network's E5B9G9R9 intermediate has a storage view (gfx_vk_tex.c), the network writes the layer itself; the stand-in only runs when forced (=1)
    extern int g_e5_storage_used; if (g_oob_on && !g_oob_forced && g_e5_storage_used) { static int said; if (!said++) VKLOG("ootbilerp: the upscale network's own path is live (ootu32: declared-format storage view), the bilinear stand-in stands down (DM_VK_OOT_BILERP=1 forces it)"); return 0; }
    return g_oob_on;
}
static int oot_bilerp_wide(const p_set_rt *r, uint32_t *w, uint32_t *h, uint32_t *f) {   // colour target 0 is a 16:9 texture of at least 640x360
    if (r->n < 1 || !r->color[0] || r->color[0]->magic != MAGIC('T', 'E', 'X', 0)) return 0;
    const tex_t *t = r->color[0]; *w = t->d.width; *h = t->d.height; *f = (uint32_t)t->d.format;
    return *w >= 640 && *h >= 360 && (*w * 9 > *h * 16 ? *w * 9 - *h * 16 : *h * 16 - *w * 9) <= 16;
}
static int oot_bilerp_may_copy(const p_set_rt *r) {   // S1b dry pass: a superset of the SET_RTs at which oot_bilerp_rt copies (whatever its kept targets)
    uint32_t w, h, f; if (!oot_bilerp_live() || !oot_bilerp_wide(r, &w, &h, &f)) return 0;
    return f == 0x3f ? w >= 2560 : f == 0x25 || f == 0x38;
}
static void oot_bilerp_rt(const p_set_rt *r) {
    static tex_t scene; static unsigned nlog;
    if (!oot_bilerp_live()) return;
    uint32_t w, h, f; if (!oot_bilerp_wide(r, &w, &h, &f)) return;
    const tex_t *t = r->color[0];
    if (f == 0x3f && w < 2560) { scene = *t; g_oob_have = 1; return; }
    if (!g_oob_have || w <= scene.d.width * 3 / 2) return;
    // 21:37 pass dump: the final pass draws the 2560x1440 R11G11B10F layer (serial 4335, cleared in the frame's first pass) with blending off over the whole RGBA8 output, so
    // the scene goes into that layer: at its first pass after the tone map (subtitles drawn on top keep), else just before the final pass samples it
    static tex_t layer; static int have_layer;
    if (f == 0x3f) { layer = *t; have_layer = 1; }
    else if ((f != 0x25 && f != 0x38) || !have_layer || layer.d.width != w || layer.d.height != h) return;
    t = &layer;
#if defined(DM_NXN1X)
    // ootdlss (NPARK 11-21 (3)): DLSS wrote this layer this frame (ootdl_dispatch): the stand-in does not copy the scene over it, whatever g_e5_storage_used says
    if (g_odl.evals && g_odl.last_eval == g_tex_frame) { static int said; if (!said++) VKLOG("ootbilerp: DLSS wrote the layer this frame: the bilinear stand-in leaves it (and every such frame)"); g_oob_have = 0; return; }
#endif
    p_copy_t2t c; memset(&c, 0, sizeof c); c.src = &scene; c.dst = t; c.flags = 1;
    c.sr = (NVNcopyRegion){0, 0, 0, (int)scene.d.width, (int)scene.d.height, 1}; c.dr = (NVNcopyRegion){0, 0, 0, (int)w, (int)h, 1};
    if (nlog < 8 || !(nlog % 600)) VKLOG("ootbilerp: %ux%u scene (nvn format %#x) -> %ux%u layer, at a pass on nvn format %#x, present %llu (#%u)", scene.d.width, scene.d.height,
                                         (unsigned)scene.d.format, w, h, f, (unsigned long long)g_tex_frame, nlog);
    nlog++; g_oob_have = 0; copy_t2t(&c);
}
static void set_rt(const p_set_rt *r) {
    if (g_oplog_n || g_oplog_seg) { oplog_flush();   // r81 DM_VK_OPLOG
        const tex_t *t0 = r->n > 0 && r->color[0] ? r->color[0] : r->depth; char wh[32] = "";
        if (t0 && t0->magic == MAGIC('T', 'E', 'X', 0)) snprintf(wh, sizeof wh, "%ux%u", t0->d.width, t0->d.height);
        if (g_oplog_n && wh[0] && !strcmp(wh, g_oplog_sz)) { g_oplog_n--; g_oplog_seg = 1; g_oplog_rcskip = 0; memset(g_oplog_cnt, 0, sizeof g_oplog_cnt);
            VKLOG("oplog: SET_RT %s (%d colour targets, colour 0 format %#x, depth %s) at present %llu, submission pass %d", wh, r->n,
                  r->n > 0 && r->color[0] && r->color[0]->magic == MAGIC('T', 'E', 'X', 0) ? (unsigned)r->color[0]->d.format : 0u, r->depth ? "yes" : "no", (unsigned long long)g_tex_frame, g_sub_pass); } }
    if (S.pass_open && !g_x_nokeep && rt_same(r)) { g_vkst.rt_kept++; return; }   // the same targets again (Unity sets them before many draws): the pass goes on
    vk_end_pass();
    rt_assign_on(&S, r);
    if (!g_wk) oot_bilerp_rt(r);   // (S1e (parallel recording): its kept targets follow the stream: a worker segment's SET_RTs are given to it by the replay thread at the hand-out)
}
// S1c (parallel recording): a deferred bind (BindTexturesDeferred, hle_nvn.c: defer, src) takes the handles the game keeps at src when the record is executed. They were
// copied into the record (its h, the recording-time fallback) and read from there: a write into command memory at every replay, not idempotent across threads (two
// recorders of one record write it at once) nor in time (src is game memory the game rewrites between submissions; a worker would read it later than the stream
// order). Now the record is not written: the handles are read into a local, from src (the default path; S's values are the same as before), or, while the dry pass
// walked the list (DM_VK_PARALLEL_DRY / SERIALSEG), from the dry pass's copy made in stream order (seg_bh_take), which a worker's segment would take its range of
static int seg_bh_take(const p_bind_handles *b, uint64_t *out);   // (with the S1b globals, after exec_ops)
// S1d (parallel recording): on a given state with the record's handles h (bind_handles_rec: S, live; the dry pass: its copy, no memo generation bumped)
static void bind_handles_on(rec_state_t *s, uint32_t op, const p_bind_handles *b, const uint64_t *h, int live) {
    int st = b->stage; if (st < 0 || st > 5) return;
    for (int i = 0; i < b->n; i++) {
        int slot = b->first + i;
        uint64_t *d = NULL;
        if (op == OP_BIND_TEX || op == OP_BIND_TEX1) { if (slot >= 0 && slot < 256) d = &s->tex[st][slot]; }
        else if (op == OP_BIND_SAMP) { if (slot >= 0 && slot < 128) d = &s->samp[st][slot]; }
        else if (op == OP_BIND_IMG) { if (slot >= 0 && slot < 32) d = &s->img[st][slot]; }
        if (d && *d != h[i]) { *d = h[i]; if (live) g_bind_gen[st]++; }   // only a change counts: games bind the same handles again before many draws (stage_set's memo)
    }
}
static void bind_handles_rec(uint32_t op, const void *a) {
    p_bind_handles tmp; const p_bind_handles *b = bind_rec(op, a, &tmp);
    uint64_t fresh[32]; const uint64_t *h = b->h;
    if (b->defer && b->src && b->n > 0 && b->n <= 32) { if (!seg_bh_take(b, fresh)) memcpy(fresh, (const void *)(uintptr_t)b->src, 8 * (size_t)b->n); h = fresh; }
    bind_handles_on(&S, op, b, h, 1);
}
uint32_t g_sub_ops[64];   // record ops replayed in the open submission, by op (logged when the driver refuses it)
// occkey (gfx_vk_core.c vk_occ_keyed_on): a samples-passed report's key, as HeyListen's RX_VK_OCC_REAL=5: the program bound at it (vertex and fragment shader keys),
// the draws since the last samples-passed reset, and its rank among this present's reports with those two (OoT's report addresses change every frame; the k-th
// report of a frame tests what the k-th of the frame before did)
extern uint64_t g_occ_key, g_occ_key2; static uint32_t g_occ_draws;
static void occ_key_make(uint64_t addr) {
    enum { RN = 4096 }; static struct { uint64_t k; uint32_t n, gen; } rk[RN]; static uint32_t gen = 1; static uint64_t fr = ~0ull;
    extern uint64_t g_tex_frame; if (fr != g_tex_frame) { fr = g_tex_frame; gen++; }
    const uint64_t vs = S.sh[0] ? S.sh[0]->key : 0, fs = S.sh[1] ? S.sh[1]->key : 0;
    uint64_t k0 = (vs * 0x9E3779B97F4A7C15ull) ^ (fs * 0xC2B2AE3D27D4EB4Full) ^ ((uint64_t)g_occ_draws * 0x165667B19E3779F9ull); if (!k0) k0 = 1;
    uint32_t rank = 0;
    for (uint32_t k = 0, i = (uint32_t)(k0 >> 52) & (RN - 1); k < 32; k++, i = (i + 1) & (RN - 1)) {
        if (rk[i].gen != gen) { rk[i].gen = gen; rk[i].k = k0; rk[i].n = 1; rank = 0; break; }
        if (rk[i].k == k0) { rank = rk[i].n++; break; } }
    uint64_t key = (k0 ^ ((uint64_t)rank * 0xFF51AFD7ED558CCDull)) * 0x9E3779B97F4A7C15ull; g_occ_key = key ? key : 1;
    uint64_t k2 = (k0 ^ (addr * 0xD6E8FEB86659FD93ull) ^ 0x5bd1e995ull) * 0xC2B2AE3D27D4EB4Full; g_occ_key2 = k2 ? k2 : 2;   // occaddr: the same report address with the same program and draws
}
// par0 (11-21, design\PARALLEL_REPLAY_DESIGN.md stage 0.1/0.2; DM_VK_REPLAYDIAG=1, log only): the stream's shape for parallel recording. Draws (by call depth),
// the ops that would stay on the replay thread (dispatches, copies, fills, texture clears; storage-writing draws: gfx_vk_eden.inc) and the runs of draws between
// them by length (a run of 48+ draws is a segment a worker could take), plus the time inside eden_draw (slot 29: the worker's share of the replay)
static _Thread_local int g_par_depth; static uint64_t g_par_draws, g_par_deep, g_par_serial, g_par_swd, g_par_run, g_par_hist[5], g_par_elig;   // hist: 1-15, 16-47, 48-127, 128-511, 512+
static void par0_break(void) {
    const uint64_t r = g_par_run; if (!r) return;
    g_par_hist[r < 16 ? 0 : r < 48 ? 1 : r < 128 ? 2 : r < 512 ? 3 : 4]++; if (r >= 48) g_par_elig += r; g_par_run = 0;
}
void vk_par0_storage_draw(void) { if (g_rdiag > 0) { g_par_swd++; par0_break(); } }   // gfx_vk_eden.inc: a draw whose pipeline writes storage
static void par0_op(uint32_t op) {
    switch (op) {
    case OP_DRAW: case OP_DRAW_IDX: case OP_MULTI_DRAW: g_par_draws++; g_par_deep += g_par_depth > 0; g_par_run++; break;
    case OP_DISPATCH: case OP_COPY_B2B: case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T: case OP_CLEAR_TEX: case OP_CLEAR_BUF: case OP_DRAW_TEX: g_par_serial++; par0_break(); break;
    case OP_END: if (!g_par_depth) par0_break(); break;
    default: break;
    }
}
void vk_par0_log(double frames, double eden_ms) {   // vk_rd_log
    par0_break();
    VKLOG("replaydiag: par0 per frame: draws %.0f (%.0f inside a CALL), serial ops %.0f, storage-writing draws %.0f; runs of draws 1-15 %.1f, 16-47 %.1f, 48-127 %.1f, 128-511 %.1f, 512+ %.1f; "
          "draws in runs of 48+ %.0f (%.0f%%); eden_draw %.2f ms of the replay",
          g_par_draws / frames, g_par_deep / frames, g_par_serial / frames, g_par_swd / frames, g_par_hist[0] / frames, g_par_hist[1] / frames, g_par_hist[2] / frames, g_par_hist[3] / frames,
          g_par_hist[4] / frames, g_par_elig / frames, g_par_draws ? 100.0 * g_par_elig / g_par_draws : 0.0, eden_ms);
    g_par_draws = g_par_deep = g_par_serial = g_par_swd = g_par_elig = 0; memset(g_par_hist, 0, sizeof g_par_hist);
}
static void exec_ops(const uint8_t *p);
// DM_VK_PROF=1 (start-up): a 1 ms CPU-time timer on the replay thread samples where it is; every 20 s the most frequent places are logged: offsets in this library,
// time in other libraries charged to the first caller of ours ("via:"), or with =2 the place itself (libc, the driver). Resolve offsets with llvm-symbolizer
// on the debug build's libdmhost.so. Simpleperf is refused on retail devices; this found the command pool resets and render pass begins that took half the replay
#ifndef DM_WINDOWS   // Windows: no SIGPROF or per-thread CPU-time timers; the sampler is left out
#include <signal.h>
#include <sys/syscall.h>
static _Atomic uint64_t g_prof_pc[16384], g_prof_leaf[16384], g_prof_lr[16384], g_prof_st[16384][6]; static _Atomic uint32_t g_prof_n; static uintptr_t g_prof_lo, g_prof_hi;   // this library's text, for the first frame of ours
static void prof_sig(int sig, siginfo_t *si, void *ucv) { (void)sig; (void)si; ucontext_t *u = ucv; uint64_t pc = u->uc_mcontext.pc;
    if (pc < g_prof_lo || pc >= g_prof_hi) {   // outside this library: charge it to our first caller on the frame chain (marked with bit 63)
        uint64_t lr = u->uc_mcontext.regs[30] & 0x7fffffffffull, fp = u->uc_mcontext.regs[29], sp = u->uc_mcontext.sp, c = 0;
        if (lr >= g_prof_lo && lr < g_prof_hi) c = lr;
        for (int k = 0; k < 12 && !c && fp && !(fp & 7) && fp >= sp && fp - sp < 0x400000; k++) { uint64_t r = ((const uint64_t *)(uintptr_t)fp)[1] & 0x7fffffffffull; if (r >= g_prof_lo && r < g_prof_hi) c = r; fp = ((const uint64_t *)(uintptr_t)fp)[0]; }
        if (c) pc = c | (1ull << 63); }
    uint32_t i = atomic_fetch_add(&g_prof_n, 1); if (i < 16384) { atomic_store(&g_prof_pc[i], pc); atomic_store(&g_prof_leaf[i], u->uc_mcontext.pc); atomic_store(&g_prof_lr[i], u->uc_mcontext.regs[30]);
        uint64_t fp = u->uc_mcontext.regs[29], sp = u->uc_mcontext.sp;   // the frame chain's return addresses (DM_VK_PROF_OUT)
        for (int k = 0; k < 6; k++) { uint64_t r = 0; if (fp && !(fp & 7) && fp >= sp && fp - sp < 0x400000) { r = ((const uint64_t *)(uintptr_t)fp)[1] & 0x7fffffffffull; fp = ((const uint64_t *)(uintptr_t)fp)[0]; } else fp = 0; atomic_store(&g_prof_st[i][k], r); } } }
static void prof_tick(void) {
    // armed by the first DM_VK_PROF=1 it sees, read every 64 frames: a ctl.txt env line can start it after loading (Xenoblade Chronicles 3 crashed loading a save with it on from start-up)
    static int on; static unsigned chk; static uint64_t last; if (!on && !(chk++ & 63)) { const char *e = getenv("DM_VK_PROF"); on = e && atoi(e);
        if (on) { Dl_info d0; if (dladdr((void *)prof_tick, &d0)) { g_prof_lo = (uintptr_t)d0.dli_fbase; g_prof_hi = g_prof_lo + 0x4000000; }
            struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_sigaction = prof_sig; sa.sa_flags = SA_SIGINFO | SA_RESTART; sigaction(SIGPROF, &sa, NULL);
            // DM_VK_PROF_TID=<tid>: sample that thread of the process instead (its CPU-time clock; e.g. the game's main thread), and DM_VK_PROF_OUT=<file>: append the raw samples (leaf pc, lr) there
            const char *tt = getenv("DM_VK_PROF_TID"); pid_t tid = tt && atoi(tt) > 0 ? (pid_t)atoi(tt) : (pid_t)syscall(SYS_gettid);
            clockid_t clk = tt && atoi(tt) > 0 ? (clockid_t)((~(unsigned)tid << 3) | 6) : CLOCK_THREAD_CPUTIME_ID;   // Linux MAKE_THREAD_CPUCLOCK(tid, CPUCLOCK_SCHED)
            struct sigevent ev; memset(&ev, 0, sizeof ev); ev.sigev_notify = SIGEV_THREAD_ID; ev.sigev_signo = SIGPROF; ev._sigev_un._tid = tid;
            timer_t tm; if (!timer_create(clk, &ev, &tm)) { struct itimerspec it = {{0, 1000000}, {0, 1000000}}; timer_settime(tm, 0, &it, NULL); VKLOG("[prof] sampling thread %d every 1 ms of its CPU time", (int)tid); } else VKLOG("[RED] [prof] timer_create for thread %d failed", (int)tid); } }
    if (!on) return; uint64_t now = tnow(); if (!last) last = now; if (now - last < 20000000000ull) return; last = now;
    uint32_t n = atomic_exchange(&g_prof_n, 0); if (n > 16384) n = 16384;
    { const char *po = getenv("DM_VK_PROF_OUT"); FILE *pf = po ? fopen(po, "ab") : NULL;
      if (pf) { for (uint32_t i = 0; i < n; i++) { uint64_t v[8] = {atomic_load(&g_prof_leaf[i]), atomic_load(&g_prof_lr[i])}; for (int k = 0; k < 6; k++) v[2 + k] = atomic_load(&g_prof_st[i][k]); fwrite(v, sizeof v, 1, pf); } fclose(pf); } }
    Dl_info di; uintptr_t base = dladdr((void *)prof_tick, &di) ? (uintptr_t)di.dli_fbase : 0;
    static uint64_t pcs[16384]; static uint32_t cnt[16384]; int np = 0;
    static int leaves = -1; if (leaves < 0) { const char *e = getenv("DM_VK_PROF"); leaves = e && atoi(e) == 2; }   // DM_VK_PROF=2: where the time is, not who called
    for (uint32_t i = 0; i < n; i++) { uint64_t pc = atomic_load(leaves ? &g_prof_leaf[i] : &g_prof_pc[i]); int k = 0; for (; k < np; k++) if (pcs[k] == pc) break; if (k == np) { pcs[np] = pc; cnt[np++] = 0; } cnt[k]++; }
    char line[4000]; int o = snprintf(line, sizeof line, "[prof] %u samples, top pcs (library offset:count):", n);
    for (int t = 0; t < 40 && o < (int)sizeof line - 40; t++) { int b = -1; for (int k = 0; k < np; k++) if (cnt[k] && (b < 0 || cnt[k] > cnt[b])) b = k; if (b < 0) break;
        uint64_t pc = pcs[b]; Dl_info d2;
        if (pc >> 63) o += snprintf(line + o, sizeof line - o, " via:%llx:%u", (unsigned long long)((pc & ~(1ull << 63)) - base), cnt[b]);
        else if (pc - base < 0x10000000) o += snprintf(line + o, sizeof line - o, " %llx:%u", (unsigned long long)(pc - base), cnt[b]);
        else if (dladdr((void *)(uintptr_t)pc, &d2) && d2.dli_fname) o += snprintf(line + o, sizeof line - o, " %s(%s)+%llx:%u", strrchr(d2.dli_fname, '/') ? strrchr(d2.dli_fname, '/') + 1 : d2.dli_fname, d2.dli_sname ? d2.dli_sname : "?", (unsigned long long)(pc - (uintptr_t)(d2.dli_saddr ? d2.dli_saddr : d2.dli_fbase)), cnt[b]);
        else o += snprintf(line + o, sizeof line - o, " ?%llx:%u", (unsigned long long)pc, cnt[b]);
        cnt[b] = 0; }
    VKLOG("%s", line);
}
#else
static void prof_tick(void) {}
#endif
// r127 (DK 11-10 device lost): raw global memory (gfx_vk_core.c vk_gmem_map) reaches only the pool chunks made so far, and r77/r78 make a chunk at its first lookup.
// DK's gsys vertex shader resets its terrain's indirect draw records that way each frame before the draws read them. In memory new to it (an area entered, a
// transformation) the chunk was made only at the indirect draw, after the reset had been recorded with a pool table without it: the reset was dropped, the records
// kept the heap's old bytes and a count near 1e9 hung the GPU. Each list is now scanned before it is replayed and the chunks its indirect draws read are made first.
// Making every chunk once raw global memory was in use instead (r126) stalled DK 56 s on 2.6 GB. DM_VK_IND_PRESCAN=0: off
int dm_nvn_call_check(const void *h);   // hle_nvn.c
static uint64_t g_prescan_n;
extern uint64_t g_nvn_fs_ticket; extern int g_nvn_fs_skip;   // s46 (hle_nvn.c): the submit's ticket for FenceSync; set when the submit marked the syncs already
// NVN2 diagnostic (DK black terrain): indirect draw arguments as the replay finds them (what the GPU left there from earlier frames: the pools are zero-copy), sampled
// for one submission every 10 s. Counts draws whose record says instance count 0, count 0, a count over 1e7, and normal ones; the first few of each zero kind with
// their address, words and the program's first two shader keys. One line per 10 s when any draw was sampled. Switch 2 boots only, a 16-byte read per indirect draw
// of the sampled submission
static uint64_t g_ia_t, g_ia_n[4], g_ia_seen, g_ia_unread, g_ia_mdi; static int g_ia_on, g_ia_ex[2];
static void ind_args_sample(uint64_t ind, int indexed) {
    if (!g_ia_on || !ind) return;
    g_ia_seen++;
    if (!vk_pool_of(ind, indexed ? 20 : 16)) { g_ia_unread++; return; }
    uint32_t w[5] = {0}; memcpy(w, (const void *)(uintptr_t)ind, indexed ? 20 : 16);
    const uint32_t count = w[0], inst = w[1];
    const int kind = inst == 0 ? 0 : count == 0 ? 1 : count > 10000000u ? 2 : 3;
    g_ia_n[kind]++;
    if (kind < 2 && g_ia_ex[kind] < 4) { g_ia_ex[kind]++; uint64_t k0 = 0, k1 = 0; uint8_t k[8];
        if (S.prog && prog_key(S.prog, 0, k)) k0 = key64(k); if (S.prog && prog_key(S.prog, 1, k)) k1 = key64(k);
        VKLOG("NVN2 indirect args: %s draw at %#llx: %u %u %u %u%s%u (program %016llx %016llx)", kind ? "count 0" : "instance count 0", (unsigned long long)ind, w[0], w[1], w[2], w[3],
              indexed ? " " : "", indexed ? w[4] : 0u, (unsigned long long)k0, (unsigned long long)k1); }
}
static uint64_t g_ia_frame, g_ia_lists;
static void ind_args_window(void) {   // at a top-level list's start: every 10 s, every list until the next present (one frame)
    if (!dm_sdk_ounce()) return;
    extern uint64_t g_tex_frame;
    const uint64_t now = tnow();
    if (g_ia_on && g_tex_frame != g_ia_frame) {   // the sampled frame is over: report
        g_ia_on = 0;
        VKLOG("NVN2 indirect args (one frame sampled, %llu lists): %llu indirect draws (%llu outside the pools, %llu multi-draw records); instance count 0: %llu, count 0: %llu, count over 1e7: %llu, other: %llu",
              (unsigned long long)g_ia_lists, (unsigned long long)g_ia_seen, (unsigned long long)g_ia_unread, (unsigned long long)g_ia_mdi, (unsigned long long)g_ia_n[0],
              (unsigned long long)g_ia_n[1], (unsigned long long)g_ia_n[2], (unsigned long long)g_ia_n[3]);
        memset(g_ia_n, 0, sizeof g_ia_n); g_ia_seen = g_ia_unread = g_ia_mdi = 0;
    }
    if (g_ia_on) { g_ia_lists++; return; }
    if (now - g_ia_t >= 10000000000ull) { g_ia_t = now; g_ia_on = 1; g_ia_frame = g_tex_frame; g_ia_lists = 1; }
}
static void ind_prescan(const uint8_t *p, int depth) {
    for (int guard = 0; guard < 1 << 22; guard++) {
        op_hdr h; memcpy(&h, p, sizeof h);
        if (h.op >= OP_COUNT_ || h.size < 8 || (h.size & 7) || h.size > (1u << 20)) return;   // (exec_ops reports it)
        const uint8_t *a = p + 8;
        switch (h.op) {
        case OP_END: return;
        case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); if (!dm_nvn_jump_ok(t)) return; p = t; continue; }   // jumpchk
        case OP_CALL: { uint64_t t; memcpy(&t, a, 8); if (depth < 8 && dm_nvn_call_check((const void *)(uintptr_t)t)) ind_prescan((const uint8_t *)(uintptr_t)t, depth + 1); break; }
        case OP_DRAW: { p_draw t; const p_draw *d = draw_rec(a, h.size, &t); if (d->ind) { (void)vk_pool_of(d->ind, d->prim == DM_PRIM_MESH_TASKS ? 8 : 16); g_prescan_n++; } break; }
        case OP_DRAW_IDX: { p_draw_idx t; const p_draw_idx *d = draw_idx_rec(a, h.size, &t); if (d->ind) { (void)vk_pool_of(d->ind, 20); g_prescan_n++; } break; }
        default: break;
        }
        p += h.size;
    }
}
// s15 (diagnostic, DM_VK_SLOWSUB=<ms>, off by default): a submission's replay time by record op, for gfx_vk_core.c's slow-submission line. An op's time runs from
// its start to the next op's start (an OP_CALL's list counts op by op; the tail of a top-level list goes to its last op); 63 is a list's start (the r127 prescan)
int g_ss_on = -1; uint64_t g_ss_ns[64], g_ss_n[64];
static uint64_t g_ss_t; static int g_ss_op = -1;
static inline void ss_mark(int op) { uint64_t t = tick(); /* replaydiag: the counter register (tnow's clock_gettime cost ~30 ns an op) */ if (g_ss_op >= 0) g_ss_ns[g_ss_op] += t - g_ss_t; g_ss_t = t; g_ss_op = op; if (op >= 0) g_ss_n[op]++; }
_Static_assert(OP_COUNT_ == 58, "s15: the op names below follow nvn_obj.h");
const char *const g_op_names[64] = {"END", "JUMP", "BEGIN", "CALL", "SIGNAL_EVENT", "REPORT_COUNTER", "COPY_B2B", "UPDATE_UB", "SET_RT", "CLEAR_COLOR", "CLEAR_DS",
    "VIEWPORTS", "SCISSOR", "DEPTH_RANGE", "BIND_PROGRAM", "BIND_STATE", "BIND_VATTR", "BIND_VSTREAM", "BIND_VBUF", "BIND_UBO", "BIND_SSBO", "BIND_TEX", "BIND_SAMP",
    "STENCIL", "POLY_OFFSET", "DRAW", "DRAW_IDX", "DISPATCH", "BARRIER", "SET_TEX_POOL", "SET_SAMP_POOL", "COPY_B2T", "COPY_STRIDE", "BLEND_COLOR", "COPY_T2B",
    "BIND_IMG", "COPY_T2T", "RESET_COUNTER", "CLEAR_TEX", "CLEAR_BUF", "DEPTH_CLAMP", "RENDER_COND", "FENCE_SYNC", "TF_BIND", "TF_STATE", "RAST_DISCARD",
    "VP_SWIZZLE", "BIND_TEX1", "PRIM_RESTART", "BIND_UBO_SMALL", "BIND_VBUF1", "DRAW_TEX", "ALPHA_REF", "PUSH_CONST", "DLSS", "MULTI_DRAW", "PATCH_SIZE",
    "DEPTH_BOUNDS", [63] = "list-start"};
// replaydiag: the per-frame lines, with the replay line every 600 presents (gfx_vk_present.c)
int vk_nslot_in_use(void);   // gfx_vk_core.c (slot12)
void vk_rd_log(double frames) {
    if (g_rdiag <= 0 || frames <= 0) return;
    static uint64_t pn[64], pc[64], po[64], poc[64]; double d[64], c[64], op[64], opc[64];
    for (int i = 0; i < 64; i++) { d[i] = (double)(g_rd_ns[i] - pn[i]) / 1e6 / frames; c[i] = (double)(g_rd_n[i] - pc[i]); pn[i] = g_rd_ns[i]; pc[i] = g_rd_n[i];
        op[i] = (double)(g_ss_ns[i] - po[i]) / 1e6 / frames; opc[i] = (double)(g_ss_n[i] - poc[i]) / frames; po[i] = g_ss_ns[i]; poc[i] = g_ss_n[i]; }
    double sb = d[17], cm = 0, ds = 0; for (int i = 0; i <= 8; i++) sb += d[i]; for (int i = 10; i <= 16; i++) cm += d[i]; for (int i = 20; i <= 25; i++) ds += d[i];
    // slot12: the slot wait split, the content observer drain, the completion thread, the view cache, the clears, the new images
    VKLOG("replaydiag: slot wait %.2f ms a frame for the GPU fence + %.2f for the completion thread after it (%.1f waits a frame, %d slots in use); content observer drain %.2f ms "
          "on the replay thread, %.2f ms on its own thread (replay2, %.1f passes) (%.0f events a frame); completion thread after the fence %.2f ms a frame (copy-backs + mirrors %.2f, query results + pass times %.2f, %.0f post ops + the rest %.2f)",
          d[40], d[41], c[40] / frames, vk_nslot_in_use(), d[17], d[49], c[49] / frames, c[18] / frames, d[43] + d[44] + d[46], d[43], d[44], c[47] / frames, d[46]);
    VKLOG("replaydiag: texture view cache %.0f hits, %.0f misses a frame, %.0f of the misses only because the cache is per submission; tex_desc_for %.2f us per sampled draw "
          "(of its images and samplers time); clears: depth %.0f a frame (begin_pass %.2f ms, into load ops %.2f (%.0f), masked draws %.2f (%.0f), pass_commit %.2f, vkCmdClearAttachments %.2f (%.0f)), "
          "colour %.0f (begin_pass %.2f, into load ops %.2f (%.0f), masked draws %.2f (%.0f), pass_commit %.2f, vkCmdClearAttachments %.2f (%.0f))",
          c[36] / frames, c[37] / frames, c[38] / frames, c[34] ? d[35] * frames * 1e3 / c[34] : 0.0,
          c[50] / frames, d[50], d[51], c[51] / frames, d[52], c[52] / frames, d[53], d[54], c[54] / frames,
          c[55] / frames, d[55], d[56], c[56] / frames, d[57], c[57] / frames, d[58], d[59], c[59] / frames);
    vk_par0_log(frames, d[29]);   // par0
    VKLOG("replaydiag: gwsuffix eden_handle %.2f us per sampled draw (%.0f handles a timed draw: evaluated %.0f%%, read from a uniform buffer %.0f%%)",
          c[34] ? d[26] * frames * 1e3 / c[34] : 0.0, c[34] ? c[26] / c[34] : 0.0, c[26] ? 100.0 * c[27] / c[26] : 0.0, c[26] ? 100.0 * c[28] / c[26] : 0.0);
    { extern uint64_t g_rpt_now, g_rpt_late, g_rq_flushes, g_rq_runs; static uint64_t pn0, pl0, pf0, pr0;   // rptfix (replay2: the batches, slot 48 their time)
      VKLOG("replaydiag: counter reports written at replay %.0f a frame, left to the completion thread (a GPU write over them in flight) %.0f a frame; replay2 batches %.1f a frame "
            "(%.0f page runs), %.3f ms a frame",
            (double)(g_rpt_now - pn0) / frames, (double)(g_rpt_late - pl0) / frames, (double)(g_rq_flushes - pf0) / frames, (double)(g_rq_runs - pr0) / frames, d[48]);
      pn0 = g_rpt_now; pl0 = g_rpt_late; pf0 = g_rq_flushes; pr0 = g_rq_runs; }
    VKLOG("replaydiag: replay2 new images %.1f a frame: img_new %.3f ms a frame (vkCreateImage, memory, bind), evict_overlaps before it %.3f ms", c[60] / frames, d[60], d[61]);
    { void vk_rdi_log(void); vk_rdi_log(); }
    VKLOG("replaydiag: ms per frame (%.1f submissions a frame): submission start %.2f (slot reclaim + observer %.2f, textures + memory pressure %.2f, lock + slot_recycle %.2f, "
          "command buffer begin %.2f, hy_apply %.2f, pool scan %.2f over %.0f pools, copied-pool wait + sync %.2f, pool_budget + hy_report %.2f, query resets + rest %.2f); "
          "commit %.2f (pass end %.2f, pool report + host barrier + readback %.2f, vkEndCommandBuffer %.2f, refusal + upload end %.2f, guards %.2f, vkQueueSubmit %.2f, rest %.2f)",
          c[0] / frames, sb, d[0], d[1], d[2], d[3], d[4], d[5], c[0] ? c[9] / c[0] : 0.0, d[6], d[7], d[8], cm, d[10], d[11], d[12], d[13], d[14], d[15], d[16]);
    VKLOG("replaydiag: dispatches %.1f a frame, %.2f ms: pass end %.2f, prepare (payload, staging, first stores) %.2f, barrier before %.2f, bind + start timestamp %.2f, dispatch %.2f, "
          "end timestamp + barrier after + staging %.2f", c[25] / frames, ds, d[20], d[21], d[22], d[23], d[24], d[25]);
    const double pdn = c[34] ? c[34] : 1;
    VKLOG("replaydiag: draw payload, us per draw (%.0f draws timed, every 8th): uniform buffers %.2f (%.1f entries), storage buffers %.2f (%.1f), images and samplers %.2f (%.1f), texel buffers %.2f (%.1f)",
          c[34], (double)(d[30] * frames * 1e3) / pdn, c[30] / pdn, (double)(d[31] * frames * 1e3) / pdn, c[31] / pdn, (double)(d[32] * frames * 1e3) / pdn, c[32] / pdn, (double)(d[33] * frames * 1e3) / pdn, c[33] / pdn);
    char line[1200]; int o = 0; uint64_t taken = 0; double all = 0; for (int i = 0; i < 64; i++) all += op[i];
    for (int k = 0; k < 16; k++) { int b = -1; for (int i = 0; i < 64; i++) if (!(taken >> i & 1) && op[i] > 0 && (b < 0 || op[i] > op[b])) b = i;
        if (b < 0) break; taken |= 1ull << b;
        o += snprintf(line + o, sizeof line - (size_t)o, " %s %.2f (%.0f);", g_op_names[b] ? g_op_names[b] : "?", op[b], opc[b]); if (o >= (int)sizeof line - 64) break; }
    VKLOG("replaydiag: record ops, ms per frame (count a frame), %.2f ms in all:%s", all, line);
}
static int prescan_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_IND_PRESCAN"); on = !(e && *e == '0'); VKLOG("r127: indirect draw arguments' pool chunks made before each list is replayed: %s", on ? "on (DM_VK_IND_PRESCAN=0 off)" : "off"); } return on; }
static void seg_list(const uint8_t *p);   // S1b (parallel recording), after exec_ops: a top-level list with the dry pass on
static int seg_mode(void);                // S1b: 0 off, 1 DM_VK_PARALLEL_DRY, 2 DM_VK_PARALLEL_SERIALSEG
static void o3p_list(const uint8_t *p); static void o3p_end(void);   // occ-s3fast: the list's resolve plan (below, before exec_ops)
void mtl_exec(const uint8_t *p) {   // timed at the top level only (OP_CALL nests)
    static int depth; uint64_t t0 = depth ? 0 : tnow(); depth++; if (depth == 1) { g_s45_kgen++; g_s45_list++; prof_tick(); if (g_ss_on > 0 || g_rdiag > 0) ss_mark(63); if (prescan_on()) ind_prescan(p, 0); ind_args_window(); }   // r127
    if (depth == 1 && g_occ2 >= 2) o3p_list(p);   // occ-s3fast: resolve points named before the replay
    if (depth == 1 && seg_mode()) seg_list(p); else exec_ops(p);   // S1b (parallel recording): DM_VK_PARALLEL_DRY / DM_VK_PARALLEL_SERIALSEG (off: as before)
    if (depth == 1 && g_occ2 >= 2) o3p_end();
    if (!--depth) { g_xo_op = 63; /* s64a: what ends a pass from here on is the submission's end */ g_vkst.ns_exec += tnow() - t0; if (g_ss_on > 0 || g_rdiag > 0) ss_mark(-1); }   // s15
}
// A record that cannot be one (size under the header, not 8-aligned or over 1 MiB, or an op past OP_COUNT_): the list is abandoned, as stepping by its size would
// spin in place or walk off into memory (size 0 kept the render thread on one record forever, 11-04). Logged with the record before it and the bytes around it
int dm_nvn_call_check(const void *h);   // hle_nvn.c
static _Thread_local const uint8_t *g_rec_prev; static _Thread_local op_hdr g_rec_prev_h; static _Thread_local const uint8_t *g_rec_jump_from, *g_rec_jump_to;   // (S1c (parallel recording): the thread's walk)
static void bad_record(const uint8_t *p, op_hdr h) {
    static int n; if (n++ >= 8 && n % 1000) return;
    char hx[3 * 64 + 1]; const uint8_t *q = p - 32 >= g_rec_prev && g_rec_prev ? p - 32 : p; int o = 0;
    for (int i = 0; i < 64 && o < (int)sizeof hx - 3; i++) o += snprintf(hx + o, sizeof hx - (size_t)o, "%02x%s", q[i], q + i + 1 == p ? "|" : " ");
    VKLOG("[RED] bad record at %p (op %u size %u, %d so far): list abandoned; record before at %p op %u size %u; last jump %p -> %p; bytes from %p: %s",
          (const void *)p, h.op, h.size, n, (const void *)g_rec_prev, g_rec_prev_h.op, g_rec_prev_h.size, (const void *)g_rec_jump_from, (const void *)g_rec_jump_to, (const void *)q, hx);
}
#define S64_ALLOW (S45_B(OP_DRAW) | S45_B(OP_DRAW_IDX) | S45_B(OP_MULTI_DRAW) | S45_B(OP_SET_RT) | S45_B(OP_BARRIER) | S45_B(OP_BEGIN) | S45_B(OP_JUMP) | S45_B(OP_CALL) | S45_B(OP_UPDATE_UB) | S45_B(OP_VIEWPORTS) | S45_B(OP_SCISSOR) | S45_B(OP_DEPTH_RANGE) | S45_B(OP_BIND_PROGRAM) | S45_B(OP_BIND_STATE) | S45_B(OP_BIND_VATTR) | S45_B(OP_BIND_VSTREAM) | S45_B(OP_BIND_VBUF) | S45_B(OP_BIND_VBUF1) | S45_B(OP_BIND_UBO) | S45_B(OP_BIND_UBO_SMALL) | S45_B(OP_BIND_SSBO) | S45_B(OP_BIND_TEX) | S45_B(OP_BIND_TEX1) | S45_B(OP_BIND_SAMP) | S45_B(OP_BIND_IMG) | S45_B(OP_STENCIL) | S45_B(OP_POLY_OFFSET) | S45_B(OP_BLEND_COLOR) | S45_B(OP_ALPHA_REF) | S45_B(OP_PUSH_CONST) | S45_B(OP_DEPTH_CLAMP) | S45_B(OP_RAST_DISCARD) | S45_B(OP_PRIM_RESTART) | S45_B(OP_PATCH_SIZE) | S45_B(OP_DEPTH_BOUNDS) | S45_B(OP_VP_SWIZZLE) | S45_B(OP_COPY_STRIDE) | S45_B(OP_SET_TEX_POOL) | S45_B(OP_SET_SAMP_POOL))   // s64: records that leave a pending pass pending
// The records that leave exec_ops' full_barrier_current as it is (only CPU-side bindings between two full barriers; was a switch in exec_ops, S1b (parallel recording):
// a mask, so that the dry pass cuts only at records that clear it, where a list resumed with full_barrier_current 0 records exactly what the uncut walk does)
#define FB_KEEP (S45_B(OP_BARRIER) | S45_B(OP_BIND_STATE) | S45_B(OP_BIND_VATTR) | S45_B(OP_BIND_VSTREAM) | S45_B(OP_BIND_VBUF) | S45_B(OP_BIND_VBUF1) | S45_B(OP_BIND_UBO) | \
    S45_B(OP_BIND_UBO_SMALL) | S45_B(OP_BIND_SSBO) | S45_B(OP_SCISSOR) | S45_B(OP_VIEWPORTS) | S45_B(OP_DEPTH_RANGE) | S45_B(OP_BIND_TEX) | S45_B(OP_BIND_TEX1) | \
    S45_B(OP_BIND_SAMP) | S45_B(OP_BIND_IMG) | S45_B(OP_STENCIL) | S45_B(OP_POLY_OFFSET) | S45_B(OP_BLEND_COLOR) | S45_B(OP_ALPHA_REF) | S45_B(OP_PUSH_CONST) | \
    S45_B(OP_DEPTH_CLAMP) | S45_B(OP_RAST_DISCARD) | S45_B(OP_PRIM_RESTART) | S45_B(OP_PATCH_SIZE) | S45_B(OP_DEPTH_BOUNDS) | S45_B(OP_VP_SWIZZLE) | S45_B(OP_COPY_STRIDE))
// S1b (parallel recording): running a list in pieces (design\PARALLEL_REPLAY_DESIGN.md Stage 1, "Resuming inside nested lists"; seg_* after exec_ops). Per thread, all
// zero / NULL unless DM_VK_PARALLEL_DRY or DM_VK_PARALLEL_SERIALSEG is on. g_seg_on: a piece of known length is being run, g_seg_left its records still to run;
// at the next one g_seg_done is set (g_seg_at = that record) and every exec_ops frame returns. g_seg_flow: the dry pass's CALL / JUMP decisions in stream order
// (bit 0 taken, bit 1 a JUMP), read instead of asking hle_nvn.c again (g_seg_fi the next one), so the walk follows the path the dry pass cut
static _Thread_local int g_seg_on, g_seg_done; static _Thread_local uint64_t g_seg_left; static _Thread_local const uint8_t *g_seg_at;
static _Thread_local const uint8_t *g_seg_flow; static _Thread_local uint32_t g_seg_nflow, g_seg_fi;
static uint64_t g_seg_flow_bad;   // decisions that did not match the record (a JUMP's taken for a CALL, or none left): the walk left the dry pass's path
// S1c (parallel recording): the dry pass's reads of deferred-bound handles (bind_handles_rec), in stream order: g_seg_bh[g_seg_bhi ..] the next record's, each entry
// {n, the record's src, n handles}; NULL when the dry pass did not walk the list (or the walk left its path: then read from src, as without it)
static _Thread_local const uint64_t *g_seg_bh; static _Thread_local uint32_t g_seg_nbh, g_seg_bhi;
static uint64_t g_seg_bh_bad;   // a deferred bind whose entry did not match (another record, or none left)
static int seg_stop(const uint8_t *p) { if (g_seg_left) { g_seg_left--; return 0; } g_seg_done = 1; g_seg_at = p; return 1; }
static void seg_handover_drop(void);   // S1d (parallel recording): the dry pass's decisions and part cuts no longer taken for the rest of the list
static int seg_flow_take(int jump, const void *t) {
    if (g_seg_fi < g_seg_nflow) { const uint8_t f = g_seg_flow[g_seg_fi++]; if ((f >> 1) == (uint8_t)jump) return f & 1; }
    if (g_wk) { seg_wk_abort(WA_DEC); return 0; }   // S1e (parallel recording): a worker never asks hle_nvn.c: its segment is recorded on the replay thread
    if (!g_seg_flow_bad++) VKLOG("[RED] S1b: the replay's %s at decision %u is not the dry pass's (%u decisions): it is asked again (the walk left the dry pass's path)", jump ? "JUMP" : "CALL", g_seg_fi, g_seg_nflow);
    g_seg_flow = NULL;   // asked from here on, for the rest of the list
    g_seg_bh = NULL;     // S1c: and the deferred handles read again (the records met from here on may not be the dry pass's)
    seg_handover_drop();   // S1d: and the reports, conditions, updates, fence syncs and part cuts done as without the dry pass
    return jump ? dm_nvn_jump_ok(t) : dm_nvn_call_check(t);
}
static int seg_bh_take(const p_bind_handles *b, uint64_t *out) {   // 1: out holds the dry pass's copy of this record's deferred handles
    if (!g_seg_bh) return 0;
    if (g_seg_bhi + 2 + (uint32_t)b->n <= g_seg_nbh && g_seg_bh[g_seg_bhi] == (uint64_t)b->n && g_seg_bh[g_seg_bhi + 1] == b->src) {
        memcpy(out, &g_seg_bh[g_seg_bhi + 2], 8 * (size_t)b->n); g_seg_bhi += 2 + (uint32_t)b->n; return 1; }
    if (g_wk) { seg_wk_abort(WA_DEC); return 0; }   // S1e (parallel recording)
    if (!g_seg_bh_bad++) VKLOG("[RED] S1c: the replay's deferred bind at entry %u is not the dry pass's (%u words): its handles are read again (the walk left the dry pass's path)", g_seg_bhi, g_seg_nbh);
    g_seg_bh = NULL; return 0;
}
static uint64_t skip_ops(void) {   // DM_VK_SKIP_OPS=<op>[,<op>...] (debugging, read once): record ops not replayed (was a static in exec_ops)
    static uint64_t skip = ~0ull; if (skip == ~0ull) { skip = 0; const char *e = getenv("DM_VK_SKIP_OPS"); for (const char *q = e; q && *q;) { char *n; long v = strtol(q, &n, 10); if (n == q) break; if (v > 0 && v < 64) skip |= 1ull << v; q = *n ? n + 1 : n; } }
    return skip;
}
// ---- occ-s3fast (design\OCCLUSION_REDESIGN.md 6.10): the resolve plan of a top-level list (replay thread, s2 / s3). A walk of the list before it is replayed (the records
// in the order exec_ops takes them: CALLs and JUMPs as hle_nvn.c answers, DM_VK_SKIP_OPS passed over, as ind_prescan and the dry pass walk) names, by stream serial (the
// draws before the record, exactly as g_oser counts them), (a) the readers of samples-passed reports: render conditions whose word x or y is such a report (s3: the GPU
// evaluates them; s2 reads landed words only, so none there) and GPU readers of report words (uniform / storage buffer binds, buffer copy sources), each with the newest
// report sequence it may read; (b) the opportunities: records where the replay thread has no pass open (a SET_RT to other targets, copies, fills, texture copies and
// clears, dispatches). gfx_vk_core.c occ2_opp asks vk_o3p_want at every pass end and opportunity: resolve now when a reader of a pending report comes before the next
// opportunity, else leave the batch pending. Correctness does not hang on the plan: a GPU condition whose report is still pending forces its resolve (o3_force, counted),
// a copy reading a pending word resolves first (occ2_unres_in), the submission's end resolves the rest; a wrong plan costs only extra or forced resolve points
#define O3P_OPP (S45_B(OP_SET_RT) | S45_B(OP_COPY_B2B) | S45_B(OP_CLEAR_BUF) | S45_B(OP_COPY_B2T) | S45_B(OP_COPY_T2B) | S45_B(OP_COPY_T2T) | S45_B(OP_CLEAR_TEX) | S45_B(OP_DISPATCH))
enum { O3P_NH = 8192 };
typedef struct { uint64_t ser, need; } o3p_need;
static struct {
    o3p_need *n; uint32_t nn, capn, ni; uint64_t *o; uint32_t no, capo, oi;   // the readers (serial, newest report sequence read), the opportunities (serials), cursors
    int on; uint64_t ser, rs0, rs, skip; const uint8_t *prt; uint32_t prt_n;  // the walk: serial, first report sequence, next one; hand-out skip; the last SET_RT's payload
    struct { uint64_t a, rs; uint32_t gen; } h[O3P_NH]; struct { uint64_t pg, rs; uint32_t gen; } pgh[O3P_NH]; uint32_t gen, nh, npg;   // this list's reports by word and by page
    uint64_t lists, off, needs, opps, mism, want, keep, nolist;   // since the last line
} g_o3p;
static int o3p_grow(void **p, uint32_t *cap, uint32_t n, size_t el) { if (n < *cap) return 1; const uint32_t c = *cap ? 2 * *cap : 1024; void *q = realloc(*p, (size_t)c * el); if (!q) return 0; *p = q; *cap = c; return 1; }
static void o3p_report(uint64_t a) {   // a samples-passed report of this list at address a: sequence g_o3p.rs
    const uint64_t rs = g_o3p.rs++;
    uint32_t i = (uint32_t)((a >> 4) * 0x9E3779B97F4A7C15ull >> 51) & (O3P_NH - 1);
    for (uint32_t k = 0; k < O3P_NH; k++, i = (i + 1) & (O3P_NH - 1)) { if (g_o3p.h[i].gen != g_o3p.gen) { if (g_o3p.nh * 2 >= O3P_NH) { g_o3p.on = 0; return; } g_o3p.h[i].gen = g_o3p.gen; g_o3p.h[i].a = a; g_o3p.nh++; break; } if (g_o3p.h[i].a == a) break; }
    g_o3p.h[i].rs = rs;
    const uint64_t pg = a >> 12; uint32_t j = (uint32_t)(pg * 0x9E3779B97F4A7C15ull >> 51) & (O3P_NH - 1);
    for (uint32_t k = 0; k < O3P_NH; k++, j = (j + 1) & (O3P_NH - 1)) { if (g_o3p.pgh[j].gen != g_o3p.gen) { if (g_o3p.npg * 2 >= O3P_NH) { g_o3p.on = 0; return; } g_o3p.pgh[j].gen = g_o3p.gen; g_o3p.pgh[j].pg = pg; g_o3p.npg++; break; } if (g_o3p.pgh[j].pg == pg) break; }
    g_o3p.pgh[j].rs = rs;
}
static int o3p_word(uint64_t a, uint64_t *rs) {   // the newest report sequence at word a: this list's, else the report table's (reports before the list)
    uint32_t i = (uint32_t)((a >> 4) * 0x9E3779B97F4A7C15ull >> 51) & (O3P_NH - 1);
    for (uint32_t k = 0; k < O3P_NH; k++, i = (i + 1) & (O3P_NH - 1)) { if (g_o3p.h[i].gen != g_o3p.gen) break; if (g_o3p.h[i].a == a) { *rs = g_o3p.h[i].rs; return 1; } }
    const rc_occ_t *e = rc_occ_find(a);
    if (e && !e->timestamp && e->rs) { *rs = e->rs - 1; return 1; }
    return 0;
}
static void o3p_need_at(uint64_t need) { if (!o3p_grow((void **)&g_o3p.n, &g_o3p.capn, g_o3p.nn, sizeof *g_o3p.n)) { g_o3p.on = 0; return; } g_o3p.n[g_o3p.nn++] = (o3p_need){g_o3p.ser, need}; g_o3p.needs++; }
static void o3p_reader(uint64_t a, uint64_t n) {   // a GPU reader of [a, a + n): the newest report on its pages (this list's), or every report pending at the list's start
    if (!n) return;
    uint64_t best = 0; int hit = 0; const uint64_t p0 = a >> 12, p1 = (a + n - 1) >> 12;
    for (uint64_t pg = p0; pg <= p1 && pg - p0 < 64; pg++) { uint32_t j = (uint32_t)(pg * 0x9E3779B97F4A7C15ull >> 51) & (O3P_NH - 1);
        for (uint32_t k = 0; k < O3P_NH; k++, j = (j + 1) & (O3P_NH - 1)) { if (g_o3p.pgh[j].gen != g_o3p.gen) break; if (g_o3p.pgh[j].pg == pg) { if (!hit || g_o3p.pgh[j].rs > best) best = g_o3p.pgh[j].rs; hit = 1; break; } } }
    if (!hit && g_o3p.rs0 && vk_occ2_unres_in(a, n)) { best = g_o3p.rs0 - 1; hit = 1; }
    if (hit) o3p_need_at(best);
}
static void o3p_walk(const uint8_t *p, int depth, uint64_t skip) {
    for (int guard = 0; guard < 1 << 22 && g_o3p.on; guard++) {
        op_hdr h; memcpy(&h, p, sizeof h);
        if (h.op >= OP_COUNT_ || h.size < 8 || (h.size & 7) || h.size > (1u << 20)) return;   // (exec_ops reports it)
        if (h.op < 64 && h.op > 3 && (skip >> h.op & 1)) { p += h.size; continue; }
        const uint8_t *a = p + 8;
        if (h.op < 64 && (O3P_OPP >> h.op & 1)) {   // an opportunity, unless a SET_RT repeats the last one's targets (the pass goes on)
            int opp = 1;
            if (h.op == OP_SET_RT) { opp = !(g_o3p.prt && g_o3p.prt_n == h.size - 8 && !memcmp(g_o3p.prt, a, h.size - 8)); g_o3p.prt = a; g_o3p.prt_n = h.size - 8; }
            if (opp && (!g_o3p.no || g_o3p.o[g_o3p.no - 1] != g_o3p.ser)) { if (!o3p_grow((void **)&g_o3p.o, &g_o3p.capo, g_o3p.no, sizeof *g_o3p.o)) { g_o3p.on = 0; return; } g_o3p.o[g_o3p.no++] = g_o3p.ser; g_o3p.opps++; } }
        switch (h.op) {
        case OP_END: return;
        case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); if (!dm_nvn_jump_ok(t)) return; p = t; continue; }   // jumpchk
        case OP_CALL: { uint64_t t; memcpy(&t, a, 8); if (depth < 8 && dm_nvn_call_check((const void *)(uintptr_t)t)) o3p_walk((const uint8_t *)(uintptr_t)t, depth + 1, skip); break; }
        case OP_DRAW: case OP_DRAW_IDX: case OP_MULTI_DRAW: case OP_DRAW_TEX: g_o3p.ser++; break;   // (as exec_ops' g_oser: every draw-type record)
        case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, a + 8, 8); if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) o3p_report(ad); break; }
        case OP_RENDER_COND: { const p_render_cond *rc = (const p_render_cond *)a; uint64_t rx = 0, ry = 0;
            if (g_occ2 < 3 || rc->mode >= 2) break;
            const int hx = o3p_word(rc->addr, &rx), hy = o3p_word(rc->addr + 16, &ry);
            if (hx || hy) o3p_need_at(hx && hy ? (rx > ry ? rx : ry) : hx ? rx : ry);
            break; }
        case OP_COPY_B2B: { uint64_t s_, n; memcpy(&s_, a, 8); memcpy(&n, a + 16, 8); o3p_reader(s_, n); break; }
        case OP_BIND_UBO: case OP_BIND_UBO_SMALL: case OP_BIND_SSBO: { p_bind_buf tb; const p_bind_buf *b = h.op == OP_BIND_SSBO ? (const p_bind_buf *)a : bind_buf_rec(h.op, a, &tb); o3p_reader(b->addr, b->size); break; }
        default: break;
        }
        p += h.size;
    }
}
static void o3p_list(const uint8_t *p) {   // mtl_exec, a top-level list, before its replay
    g_o3p.lists++;
    if (!++g_o3p.gen) { memset(g_o3p.h, 0, sizeof g_o3p.h); memset(g_o3p.pgh, 0, sizeof g_o3p.pgh); g_o3p.gen = 1; }
    g_o3p.nh = g_o3p.npg = 0; g_o3p.nn = g_o3p.no = g_o3p.ni = g_o3p.oi = 0; g_o3p.prt = NULL; g_o3p.prt_n = 0; g_o3p.skip = 0;
    g_o3p.ser = g_oser; g_o3p.rs0 = g_o3p.rs = vk_occ2_rseq(); g_o3p.on = 1; (void)o3g_on();
    o3p_walk(p, 0, skip_ops());
    if (!g_o3p.on) { g_o3p.off++; return; }
    g_o3p.on = 2;   // (2: a plan for the list being replayed; checked against the replay at its end)
    if (!S.pass_open) vk_occ2_opp();   // the list's start is an opportunity when no pass is open (a reader early in its first pass)
}
static void o3p_end(void) {   // mtl_exec, the list replayed: the walk's serial and report count against the replay's (a mismatch only costs resolve points)
    if (g_o3p.on == 2 && (g_oser != g_o3p.ser || vk_occ2_rseq() != g_o3p.rs)) g_o3p.mism++;
}
// gfx_vk_core.c occ2_opp: at stream serial ser with reports [plo, phi) pending: 1 = resolve now
int vk_o3p_want(uint64_t ser, uint64_t plo, uint64_t phi) {
    if (g_o3p.on != 2) { g_o3p.nolist++; return 1; }   // no plan: every opportunity (the s2 / s3 rule)
    while (g_o3p.ni < g_o3p.nn && g_o3p.n[g_o3p.ni].ser < ser) g_o3p.ni++;   // (readers before this point are past: they forced or found theirs)
    uint32_t j = g_o3p.ni; while (j < g_o3p.nn && !(g_o3p.n[j].need >= plo && g_o3p.n[j].need < phi)) j++;
    if (j == g_o3p.nn) { g_o3p.keep++; return 0; }   // nothing later in this list reads them: the submission's end (or a later list's plan)
    const uint64_t lim = ser > g_o3p.skip ? ser : g_o3p.skip;   // (opportunities inside a segment handed to a worker never come on this thread)
    while (g_o3p.oi < g_o3p.no && g_o3p.o[g_o3p.oi] <= lim) g_o3p.oi++;
    if (g_o3p.oi < g_o3p.no && g_o3p.o[g_o3p.oi] < g_o3p.n[j].ser) { g_o3p.keep++; return 0; }   // a later opportunity still comes before that reader (strictly: a draw between)
    g_o3p.want++; return 1;
}
void vk_o3p_log(double f) {   // gfx_vk_core.c occ2_log
    VKLOG("occ-s3fast plan: lists %.1f a frame (no plan: table full or no memory %llu, walk off the replay's serials %llu), readers named %.1f a frame, opportunities %.1f a frame; "
          "asked: resolve now %.1f, keep pending %.1f, no plan %.1f a frame", g_o3p.lists / f, (unsigned long long)g_o3p.off, (unsigned long long)g_o3p.mism, g_o3p.needs / f, g_o3p.opps / f,
          g_o3p.want / f, g_o3p.keep / f, g_o3p.nolist / f);
    o3g_log(f);
    g_o3p.lists = g_o3p.off = g_o3p.mism = g_o3p.needs = g_o3p.opps = g_o3p.want = g_o3p.keep = g_o3p.nolist = 0;
}
static int private_scale_draw(uint32_t op, const uint8_t *next) {
    if ((op != OP_DRAW && op != OP_DRAW_IDX) || S.pass_open || g_rc_skip ||
        !pending_guest_barrier || g_s64_on == 1 || !vk_eden_mode()) return 0;
    op_hdr end;
    memcpy(&end, next, sizeof end);
    if (end.op != OP_BARRIER) return 0;
    // No target changes are allowed in the carry window. Exclude images that
    // might be re-uploaded from guest memory or observed outside this queue.
    for (unsigned n = 0; n < pending_scale_count; ++n) {
        vkimg_t *im = pending_scale[n].im;
        int private_target = 0;
        for (int i = 0; i < 9; ++i) {
            if (P_att[i].im != im) continue;
            const tex_t *t = i == 8 ? &S.rt_dep : &S.rt_col[i];
            private_target = im->scaled_valid && t->pool &&
                (t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) &&
                !(t->d.flags & NVN_TEXTURE_FLAGS_DISPLAY) && vk_tex_peek(t) == im;
            break;
        }
        if (!private_target) return 0;
    }
    return 1;
}
// S1d (parallel recording): the pure NVN state records, applied to a given state: S (live: the memo generations bumped and the logs written, as exec_ops did inline) or the
// dry pass's copy (live 0), so the dry pass keeps the whole stream state (rec_state_t up to the open pass) and each segment gets a full snapshot. 0: not such a record
static inline __attribute__((always_inline)) int state_op_on(rec_state_t *st, uint32_t op, const uint8_t *a, int live) {
    switch (op) {
    case OP_VIEWPORTS: { const p_viewports *v = (const p_viewports *)a; for (int i = 0; i < v->n && v->first + i < 16; i++) memcpy(st->vp[v->first + i], v->v[i], 16); break; }
    case OP_SCISSOR: st->sc = *(const p_scissor *)a; st->sc_set = 1; break;
    case OP_DEPTH_RANGE: { const p_depth_range *d = (const p_depth_range *)a; st->dnear = d->n; st->dfar = d->f; break; }
    case OP_BIND_STATE: { const p_bind_state *s = (const p_bind_state *)a;
        switch (s->kind) {
        case 0: { blend_st b; memcpy(&b, s->bytes, sizeof b); if (b.target < 8) st->blend[b.target] = b; break; }
        case 1: memcpy(&st->color, s->bytes, sizeof st->color); break;
        case 2: memcpy(&st->chmask, s->bytes, sizeof st->chmask); break;
        case 3: memcpy(&st->ms, s->bytes, sizeof st->ms); break;
        case 4: memcpy(&st->poly, s->bytes, sizeof st->poly); break;
        case 5: memcpy(&st->ds, s->bytes, sizeof st->ds); break;
        }
        break; }
    case OP_BIND_VATTR: { const p_bind_vattr *v = (const p_bind_vattr *)a; memset(st->vattr, 0, sizeof st->vattr); for (int i = 0; i < v->n && i < 16; i++) st->vattr[v->first + i] = v->st[i]; break; }
    case OP_BIND_VSTREAM: { const p_bind_vstream *v = (const p_bind_vstream *)a; for (int i = 0; i < v->n && v->first + i < 16; i++) st->vstream[v->first + i] = v->st[i]; break; }
    case OP_BIND_VBUF: case OP_BIND_VBUF1: { p_bind_vbuf tv; const p_bind_vbuf *v = vbuf_rec(op, a, &tv); for (int i = 0; i < v->n && v->first + i < 16; i++) st->vbuf[v->first + i] = v->r[i]; break; }
    case OP_BIND_UBO: case OP_BIND_UBO_SMALL: { p_bind_buf tb; const p_bind_buf *b = bind_buf_rec(op, a, &tb); if (live && (g_occdiag || g_occ2 >= 2) && !g_wk) vk_occd_read(b->addr, b->size, 2);   /* occ-s0 */ if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 16 && memcmp(&st->ubo[b->stage][b->index], b, sizeof *b)) { st->ubo[b->stage][b->index] = *b; if (live) g_ubo_gen[b->stage]++; } break; }
    case OP_BIND_SSBO: { const p_bind_buf *b = (const p_bind_buf *)a; if (live && (g_occdiag || g_occ2 >= 2) && !g_wk) vk_occd_read(b->addr, b->size, 3);   /* occ-s0 */ if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 16 && memcmp(&st->ssbo[b->stage][b->index], b, sizeof *b)) { st->ssbo[b->stage][b->index] = *b; if (live) g_bind_gen[b->stage]++; } break; }
    case OP_STENCIL: { const p_stencil *s = (const p_stencil *)a;
        for (int f = 0; f < 2; f++) if (s->face & (1 << f)) { if (s->which == 0) st->smask[f] = s->value_mask; else if (s->which == 1) st->swmask[f] = s->mask; else st->sref[f] = s->ref; }
        break; }
    case OP_POLY_OFFSET: st->pofs = *(const p_poly_offset *)a; break;
    case OP_BLEND_COLOR: memcpy(st->blendc, ((const p_blend_color *)a)->rgba, 16); break;
    case OP_ALPHA_REF: st->alpha_ref = ((const p_alpha_ref *)a)->ref; break;   // r56: the alpha test's reference (prepare_draw atest)
    case OP_PUSH_CONST: { const p_push_const *b = (const p_push_const *)a; int c = b->compute != 0;
        if ((uint32_t)b->offset + b->size <= DMSH_PUSH_SIZE && memcmp(st->push[c] + b->offset, b->data, b->size)) { memcpy(st->push[c] + b->offset, b->data, b->size); if (live) for (int s5 = 0; s5 < 6; s5++) if ((s5 == 5) == c) g_bind_gen[s5]++; }   // bind, not ubo: the set memo refreshes only constant buffers on a ubo change, and the push block is a driver segment copy in the set
        else if (live && (uint32_t)b->offset + b->size > DMSH_PUSH_SIZE) VKRED_ONCE(90, "push constants past the block: offset %u size %u", b->offset, b->size);
        break; }
    case OP_DEPTH_CLAMP: st->depth_clamp = ((const p_depth_clamp *)a)->on; break;
    case OP_RAST_DISCARD: st->rast_discard = ((const p_rast_discard *)a)->on; break;
    case OP_PRIM_RESTART: st->prim_restart = ((const p_prim_restart *)a)->on; break;
    case OP_PATCH_SIZE: st->patch_n = ((const p_patch_size *)a)->n; break;   // r69
    case OP_DEPTH_BOUNDS: { const p_depth_bounds *b = (const p_depth_bounds *)a; st->db_on = b->on; st->db_lo = b->lo; st->db_hi = b->hi; break; }   // r69
    case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: {   // bound handles were rebased when recorded (hle_nvn.c bind_handles); r90: the base kept for handles read from uniform buffers (ubo_handle)
        int nvn_dpool_base(const void *pool, int tex); int tex = op == OP_SET_TEX_POOL, b = nvn_dpool_base(((const p_set_pool *)a)->pool, tex);
        if (live && b != (tex ? st->tex_base : st->samp_base)) { static int n; if (n < 6) { n++; VKLOG("r90: %s pool bound with IDs from %d: handles read from uniform buffers take that base (DM_VK_UBO_POOL=0 off)", tex ? "texture" : "sampler", b); } }
        if (tex) st->tex_base = b; else st->samp_base = b;
        break; }
    case OP_COPY_STRIDE: { const p_copy_stride *c = (const p_copy_stride *)a; if (c->image) st->image_stride = c->stride; else st->row_stride = c->stride; break; }
    case OP_VP_SWIZZLE: { const p_vp_swizzle *b = (const p_vp_swizzle *)a; memcpy(st->vp_sw[b->first], b->sw, 4 * (size_t)b->n); break; }
    default: return 0;
    }
    return 1;
}
// S1d (parallel recording): the dry pass's handed-over decisions (seg_dec, with the S1b/S1c globals after exec_ops), taken by the replay in stream order. g_seg_hot: one
// test per record on the default path (was g_seg_on's), set while a piece is counted or the list's records are numbered (g_seg_ri: the dry pass's record index of the next
// record) for its decisions and part cuts
typedef struct seg_dec seg_dec;
static _Thread_local int g_seg_hot;
static int seg_hot(const uint8_t *p);
static seg_dec *seg_dec_take(int kind);
enum { SD_REPORT = 1, SD_RESET, SD_FENCE, SD_COND, SD_UB };
static void seg_report(seg_dec *d, uint64_t t, uint64_t ad);
static void seg_oser_chk(const seg_dec *d);   // occ-s1: the replay thread's stream serial at a report or reset against the dry pass's
static void seg_cond(seg_dec *d, const p_render_cond *rc);
static void seg_ub(seg_dec *d);
static void exec_ops(const uint8_t *p) {
    // Local to one command list. Only CPU-side bindings can separate two
    // equivalent full dependencies; calls and all GPU work invalidate the proof.
    int full_barrier_current = 0;
    for (;;) {
        if (__builtin_expect(g_seg_hot, 0) && seg_hot(p)) return;   // S1b (parallel recording): the piece being run ends before this record (seg_piece); S1d: its index, a part cut before it
        op_hdr h; memcpy(&h, p, sizeof h);
        if (h.op >= OP_COUNT_ || h.size < 8 || (h.size & 7) || h.size > (1u << 20)) { materialize_scale(NULL); bad_record(p, h); return; }
        g_rec_prev = p; g_rec_prev_h = h;
        if (g_ss_on > 0 || g_rdiag > 0) ss_mark(h.op);   // s15
        if (g_oplog_seg) { g_oplog_cnt[h.op]++; if (g_rc_skip && (h.op == OP_DRAW || h.op == OP_DRAW_IDX || h.op == OP_MULTI_DRAW)) g_oplog_rcskip++; }   // r81 DM_VK_OPLOG
        if (!g_wk) { if (h.op < 64) g_sub_ops[h.op]++; vk_og_op(h.op); }   // ootgap (gfx_vk_core.c): the record ops replayed between timed passes (S1e (parallel recording): the replay thread's)
        const uint64_t skip = skip_ops();   // debugging: record ops not replayed (S1b: read in skip_ops, the dry pass reads it too)
        if (h.op < 64 && h.op > 3 && (skip >> h.op & 1)) { p += h.size; continue; }
        const uint8_t *a = p + 8;
        if (full_barrier_current && !(FB_KEEP >> h.op & 1)) full_barrier_current = 0;   // (S1b: FB_KEEP, was a switch here)
        g_xo_next = p + h.size; g_xo_op = h.op; if (!(S45_KSAFE >> h.op & 1)) g_s45_kgen++;   /* s45 */   // r89 pass_area
        if (pending_scale_count) {
            switch (h.op) {
            case OP_CLEAR_TEX: case OP_SCISSOR: case OP_DEPTH_RANGE: case OP_VIEWPORTS:
            case OP_COPY_STRIDE: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM:
            case OP_BIND_VBUF: case OP_BIND_VBUF1: case OP_BIND_UBO: case OP_BIND_UBO_SMALL:
            case OP_BIND_SSBO: break;
            default:
                if (!skip && private_scale_draw(h.op, g_xo_next)) {
                    vk_img_barrier_all(vk_cmd());
                    pending_guest_barrier = 0;
                    carry_one_draw = 1;
                } else materialize_scale(NULL);
                break;
            }
        }
        const uint64_t cache_allow = S64_ALLOW & ~(S45_B(OP_SET_RT) | S45_B(OP_UPDATE_UB) | S45_B(OP_CALL) | S45_B(OP_JUMP));
        if (g_rdiag > 0) par0_op(h.op);   // par0 (log only): the stream's shape for parallel recording
        if (g_s64_pend && !((g_s64_pend == 2 ? cache_allow : S64_ALLOW) >> h.op & 1)) { g_s64_opend++; vk_end_pass(); }   // s64: an op that is not a draw or state ends the pending pass
        if (g_occ2 >= 2 && !g_wk && h.op < 64 && (O3P_OPP >> h.op & 1) && S.pass_open == 0) vk_occ2_opp();
        if (__builtin_expect(g_o3g.on, 0) && !g_wk && (h.op == OP_DRAW || h.op == OP_DRAW_IDX || h.op == OP_MULTI_DRAW || h.op == OP_CLEAR_COLOR || h.op == OP_CLEAR_DS)) o3g_op(h.op, a);   // occ-s3fast (DM_VK_OCC_CONDDIAG)   // occ-s3fast: an opportunity the plan counts (no pass open here; one that is open ends inside the op, vk_end_pass asks then)
        switch (h.op) {
        case OP_END: return;
        case OP_BEGIN: break;
        case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); if (!(g_seg_flow ? seg_flow_take(1, t) : dm_nvn_jump_ok(t))) return; g_rec_jump_from = p; g_rec_jump_to = t; p = t; continue; }   // jumpchk: a target that is no record list ends it
        case OP_CALL: { uint64_t t; memcpy(&t, a, 8); if (g_seg_flow ? seg_flow_take(0, (const void *)(uintptr_t)t) : dm_nvn_call_check((const void *)(uintptr_t)t)) { g_par_depth++; exec_ops((const uint8_t *)(uintptr_t)t); g_par_depth--; if (g_seg_done) return; } break; }   // hle_nvn.c: skipped when the handle's bytes changed since it was recorded (S1b: the dry pass's decision when it walked the list; a piece's end unwinds)
        case OP_SIGNAL_EVENT: { if (g_wk) { seg_wk_abort(WA_OP); break; }   // S1e (parallel recording): a post op (its place among the posts): never in a worker's segment (seg_dry_list)
            volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, a + 8, 4); memcpy(&v, a + 12, 4); mtl_sub_event(val, m, v); break; }
        case OP_REPORT_COUNTER: { if (g_wk) { seg_wk_abort(WA_OP); break; }   // S1e (parallel recording): the report's commit (queue, posts, occlusion) is the replay thread's: never in a worker's segment
            uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, a + 8, 8); seg_dec *dd = g_seg_hot ? seg_dec_take(SD_REPORT) : NULL;
            if (dd) { seg_report(dd, t, ad); break; }   // S1d (parallel recording): cnt_note / rc_note done by the dry pass, a timestamp's slot named by it
            cnt_note(1, t, ad); if (DM_GAME_RENDER_COND) rc_note(ad, t == 0 || t == 0x10 ? 1 : t == NVN_COUNTER_TYPE_SAMPLES_PASSED ? 2 : 3, g_occ2 >= 3 ? 1 + vk_occ2_rseq() : 0);   // (occ-s3: the sequence this report takes)
            if (DM_GAME_RENDER_COND && g_occdiag) rc_note_diag(ad, t == 0 || t == 0x10 ? 1 : t == NVN_COUNTER_TYPE_SAMPLES_PASSED ? 2 : 3, g_oser, g_vk_pends);   // occ-s0
            if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED && g_occ2 < 2) occ_key_make(ad); mtl_sub_counter(ad, (uint32_t)t); g_occ_key = g_occ_key2 = 0; break; }   // (occ-s2: no occkey)
        case OP_RESET_COUNTER: { uint64_t t; memcpy(&t, a, 8); if (g_wk && t == NVN_COUNTER_TYPE_SAMPLES_PASSED) { seg_wk_abort(WA_OP); break; }   // S1e (parallel recording): occlusion (R4): never in a worker's segment
            seg_dec *dr = g_seg_hot ? seg_dec_take(SD_RESET) : NULL; if (!dr) cnt_note(0, t, 0); else seg_oser_chk(dr);   // (S1d: cnt_note done by the dry pass; occ-s1: its serial checked)
            if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) g_occ_draws = 0; vk_oq_reset((uint32_t)t); break; }
        case OP_FENCE_SYNC: { if (g_seg_hot && seg_dec_take(SD_FENCE)) break;   // S1d (parallel recording): written by the dry pass
            sync_t *s; memcpy(&s, a, 8); if (!g_nvn_fs_skip) { s->fence = g_nvn_fs_ticket; s->fenced = 1; } break; }   // s46: the submit's ticket, not a Vulkan fence
        case OP_COPY_B2B: { uint64_t s_, d, n; memcpy(&s_, a, 8); memcpy(&d, a + 8, 8); memcpy(&n, a + 16, 8); mtl_sub_copy(s_, d, n); break; }
        case OP_CLEAR_BUF: { const p_clear_buf *c = (const p_clear_buf *)a; mtl_sub_fill(c->addr, c->size, c->value); break; }
        case OP_UPDATE_UB: { seg_dec *dd = g_seg_hot ? seg_dec_take(SD_UB) : NULL; if (dd) { seg_ub(dd); break; }   // S1d (parallel recording): the dry pass's copy, the list operation here
            uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, a + 8, 8); vk_sub_update(d, a + 16, n); break; }
        case OP_SET_RT: set_rt((const p_set_rt *)a); break;
        case OP_CLEAR_COLOR: if (!g_rc_skip) clear_color((const p_clear_color *)a); else if (!g_wk) { g_rc_n[6]++; g_rcd_drop_clear++; } break;   // (S1e (parallel recording): the census is the replay thread's)
        case OP_CLEAR_DS: if (!g_rc_skip) clear_ds((const p_clear_ds *)a); else if (!g_wk) { g_rc_n[6]++; g_rcd_drop_clear++; } break;
        // S1d (parallel recording): the pure state records through state_op_on (the dry pass applies the same function to its copy); were inline here, same effects
        case OP_VIEWPORTS: case OP_SCISSOR: case OP_DEPTH_RANGE: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF: case OP_BIND_VBUF1:
        case OP_BIND_UBO: case OP_BIND_UBO_SMALL: case OP_BIND_SSBO: case OP_STENCIL: case OP_POLY_OFFSET: case OP_BLEND_COLOR: case OP_ALPHA_REF: case OP_PUSH_CONST:
        case OP_DEPTH_CLAMP: case OP_RAST_DISCARD: case OP_PRIM_RESTART: case OP_PATCH_SIZE: case OP_DEPTH_BOUNDS: case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL:
        case OP_COPY_STRIDE: case OP_VP_SWIZZLE:
            (void)state_op_on(&S, h.op, a, 1); break;
        case OP_BIND_PROGRAM: bind_program((const p_bind_program *)a); break;
        case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_BIND_TEX1: bind_handles_rec(h.op, a); break;
        case OP_DRAW_TEX: g_oser++; VKRED_ONCE(91, "DrawTexture not replayed yet"); break;   // (occ-s1: a draw-type record: the stream serial)
        case OP_DLSS:
#if defined(DM_NXN1X)
            if (h.size >= sizeof(op_hdr) + sizeof(p_dlss)) replay_dlss((const p_dlss *)a);
            else VKRED_ONCE(92, "OP_DLSS truncated parameters");
#else
            VKRED_ONCE(92, "OP_DLSS unavailable on this backend");
#endif
            break;
        case OP_DRAW: { g_oser++; if (!g_wk) g_occ_draws++; if (g_rc_skip) { if (!g_wk) { g_rc_n[6]++; g_rcd_drop_draw++; } break; }   /* S1e (parallel recording): a worker segment's draws counted by the replay thread at its hand-out */ p_draw t; const p_draw *d = draw_rec(a, h.size, &t); if (d->ind && rcd_on()) { void vk_pool_kind_note(uint64_t, int); vk_pool_kind_note(d->ind, 0); } { void vk_pool_id_args(uint64_t, int, int); if (d->ind) vk_pool_id_args(d->ind, d->prim == DM_PRIM_MESH_TASKS ? 8 : 16, 0); }   /* bak91c */ if (g_ia_on && d->ind && d->prim != DM_PRIM_MESH_TASKS) ind_args_sample(d->ind, 0); /* NVN2 diagnostic */ if (d->mdi_cnt && !d->mdi_i) mdi_scan(p + h.size, OP_DRAW, d->ind, d->mdi_cnt);
            if (!d->ind && !d->mdi_cnt) { n1x_atomic_begin(); fst_begin(); }   // fstage: a learned fragment shader's written storage buffers staged
            draw(d);
            n1x_atomic_end(); fst_end();
            break;
        }
        case OP_DRAW_IDX: { g_oser++; if (!g_wk) g_occ_draws++; if (g_rc_skip) { if (!g_wk) { g_rc_n[6]++; g_rcd_drop_draw++; } break; }   /* (S1e) */ p_draw_idx t; const p_draw_idx *d = draw_idx_rec(a, h.size, &t); if (d->ind && rcd_on()) { void vk_pool_kind_note(uint64_t, int); vk_pool_kind_note(d->ind, 0); } { void vk_pool_id_args(uint64_t, int, int); if (d->ind) vk_pool_id_args(d->ind, 20, 0); }   /* bak91c */ if (g_ia_on && d->ind) ind_args_sample(d->ind, 1); /* NVN2 diagnostic */ if (d->mdi_cnt && !d->mdi_i) mdi_scan(p + h.size, OP_DRAW_IDX, d->ind, d->mdi_cnt);
            if (!d->ind && !d->mdi_cnt) { n1x_atomic_begin(); fst_begin(); }   // fstage
            draw_idx(d);
            n1x_atomic_end(); fst_end();
            break;
        }
        // r72 (Switch 1 thread's audit): a multi-draw of more than MDI_INLINE (hle_nvn.c) is one record; it had no case here, so it fell to "unknown record op" and was
        // never drawn (on the Mac, Monster Hunter Rise's GPU-driven scene). Drawn as a run of max records with vkCmdDraw(Indexed)IndirectCount (mdi_draw), the GPU reading the count
        case OP_MULTI_DRAW: { if (g_wk) { seg_wk_abort(WA_OP); break; }   // S1e (parallel recording): indirect: never in a worker's segment
            g_oser++; g_occ_draws++; if (g_rc_skip) { g_rc_n[6]++; g_rcd_drop_draw++; break; }   // (occ-s1: the stream serial, as every draw-type record)
            p_multi_draw m; memset(&m, 0, sizeof m); memcpy(&m, a, h.size - 8 < sizeof m ? h.size - 8 : sizeof m);
            if (m.max <= 0 || !m.cnt || !m.ind) break;
            if (g_ia_on) {   // NVN2 diagnostic (ind_args_sample): the records the count buffer says are drawn, at most 64
                uint32_t c = 0; if (vk_pool_of(m.cnt, 4)) { memcpy(&c, (const void *)(uintptr_t)m.cnt, 4); if (c > (uint32_t)m.max) c = (uint32_t)m.max; }
                const uint64_t st = m.stride > 0 ? (uint64_t)m.stride : m.itype < 0 ? 16 : 20; g_ia_mdi++;
                for (uint32_t i = 0; i < c && i < 64; i++) ind_args_sample(m.ind + st * i, m.itype >= 0);
            }
            { static int n; if (n < 4) { n++; VKLOG("r72: multi-draw record: up to %d draws, stride %lld, %s", m.max, (long long)m.stride, m.itype < 0 ? "arrays" : "indexed"); } }
            g_mdi_cnt = m.cnt; g_mdi_n = m.max; g_mdi_stride = m.stride > 0 ? (uint64_t)m.stride : m.itype < 0 ? 16 : 20;
            if (m.itype < 0) { p_draw d = {m.prim, 0, 0, 1, 0, m.ind, m.cnt, 0, 0}; draw(&d); }
            else { p_draw_idx d = {m.prim, m.itype, 0, 0, 0, 1, m.ibuf, m.ind, m.cnt, 0, 0}; draw_idx(&d); }
            break; }
        case OP_DISPATCH: dispatch((const p_dispatch *)a); break;
        case OP_BARRIER:   // the pass ends at every barrier: kept across XC3's fragment- and primitive-order barriers (0x1 0x2 0x4) its characters drew torn in some frames
            if (g_x_listpasses) VKLOG("    barrier %#x", *(const int32_t *)a);
            { int32_t fl; memcpy(&fl, a, 4); g_s64_bar++; g_s64_bar_open += S.pass_open == 1; for (int i = 0; i < 32; i++) if ((uint32_t)fl >> i & 1) g_s64_bit[i]++; }   // s64a
            if (g_s64_on && vk_eden_mode() && S.pass_open == 1 && !g_s64_pwrite && !g_tf_in_pass) {
                int32_t flags;
                memcpy(&flags, a, sizeof flags);
                if (g_s64_on == 1 || flags == NVN_BARRIER_INVALIDATE_TEXTURE_BIT) {
                    g_s64_pend = g_s64_on;
                    ++g_s64_npend;
                    break;
                }
            }
            end_pass_for_scale();
            if (pending_scale_count) pending_guest_barrier = 1;
            else if (cbar_on() && (vk_bar_mode() & 2) && S.pass_open == 0 && g_disp_seq == g_bar_seq && g_disp_cb == vk_cmd() && vk_cbar_next_dispatch(g_xo_next)) {
                uint64_t n = ++g_cbar_skip;   // cbar1: between two dispatches: r89's barriers order it (above)
                if (n <= 2 || !(n & (n - 1))) VKLOG("cbar1: guest barrier %#x between two dispatches left to their compute barriers (%llu so far, %llu kept full)", *(const int32_t *)a, (unsigned long long)n, (unsigned long long)g_cbar_kept);
            } else if (!full_barrier_current) {
                if (cbar_on()) { uint64_t kk = ++g_cbar_kept;   // cbar2 counter: kept full, by what is before and after
                    g_cbk[g_disp_seq == g_bar_seq && g_disp_cb == vk_cmd() ? 0 : 1][cbk_next(g_xo_next)]++;
                    if (kk >= 1024 && !(kk & (kk - 1))) VKLOG("cbar2: guest barriers kept full so far %llu, after a dispatch -> dispatch %llu, draw/pass %llu, copy/fill %llu, counter/uniform %llu, other %llu; "
                        "after other work -> dispatch %llu, draw/pass %llu, copy/fill %llu, counter/uniform %llu, other %llu; barriers held %llu, recorded from held %llu, merged into a held one %llu",
                        (unsigned long long)kk, (unsigned long long)g_cbk[0][0], (unsigned long long)g_cbk[0][1], (unsigned long long)g_cbk[0][2], (unsigned long long)g_cbk[0][3], (unsigned long long)g_cbk[0][4],
                        (unsigned long long)g_cbk[1][0], (unsigned long long)g_cbk[1][1], (unsigned long long)g_cbk[1][2], (unsigned long long)g_cbk[1][3], (unsigned long long)g_cbk[1][4],
                        (unsigned long long)vk_bmerge(0), (unsigned long long)vk_bmerge(1), (unsigned long long)vk_bmerge(2)); }
                vk_img_barrier_all(vk_cmd());
                full_barrier_current = 1;
            } else {
                ++g_repeated_barriers;
            }
            break;
        case OP_COPY_B2T: copy_b2t((const p_copy_b2t *)a, 1); break;
        case OP_COPY_T2B: copy_b2t((const p_copy_b2t *)a, 0); break;
        case OP_COPY_T2T: copy_t2t((const p_copy_t2t *)a); break;
        case OP_CLEAR_TEX: clear_tex((const p_clear_tex *)a); break;
        case OP_RENDER_COND: { seg_dec *dd = g_seg_hot ? seg_dec_take(SD_COND) : NULL; if (dd) seg_cond(dd, (const p_render_cond *)a); else render_cond((const p_render_cond *)a);
            if (g_occ2 >= 3) rcg_settle(); if (g_occ2 >= 2 && !g_wk) o3g_cond((const p_render_cond *)a); break; }   // occ-s3fast: the open conditional block ends here unless the new condition gates the draws by the same predicate   // recorded only with the profile key render_cond (hle_nvn.c); S1d (parallel recording): the dry pass's result
        case OP_TF_BIND: { const p_tf_bind *b = (const p_tf_bind *)a; if (b->index >= 0 && b->index < 4) g_tf[b->index] = *b; break; }
        case OP_TF_STATE: if (!V.has_xfb) VKRED_ONCE(80, "transform feedback: the device has none, its draws write nothing"); tf_state((const p_tf_state *)a); break;
        default: VKRED_ONCE(82, "unknown record op %u", h.op); break;
        }
        if (carry_one_draw) {
            // The very next guest record ends this pass. Close it here while
            // the proven shader and attachment set still describe the draw.
            g_xo_op = OP_BARRIER;
            vk_end_pass();
            materialize_scale(NULL);
            carry_one_draw = 0;
        }
        p += h.size;
    }
}

// ---------------- S1b (parallel recording): the dry pass and the same-thread segment harness ----------------
// design\PARALLEL_REPLAY_DESIGN.md Stage 1 ("Boundaries", "Worker-safe ops", "Resuming inside nested lists", "Dry pass") and its S1b section. Replay thread only; nothing
// here runs unless DM_VK_PARALLEL_DRY=1 or DM_VK_PARALLEL_SERIALSEG=1 (both live, read when the environment changes; SERIALSEG includes the dry pass).
// - The dry pass (seg_dry, before each top-level list of a submission) walks the list in stream order as exec_ops will: the same record checks, CALLs descended when
//   dm_nvn_call_check says so, JUMPs followed when dm_nvn_jump_ok does, a bad record or a refused JUMP ending that list, DM_VK_SKIP_OPS records passed over. It asks
//   hle_nvn.c once per CALL / JUMP and keeps the answers (g_sd.flow), which the replay of the list then reads instead of asking again (g_seg_flow), so the replay walks
//   exactly the path that was cut and nothing is asked twice. Apart from that it is discovery only (S1b; S1d moved CPU work here, below): no report, event, fence, render condition, uniform update, texture
//   lookup or command is made; it keeps a copy of the stream state it needs (the program bound, rasterizer discard, the render targets, transform feedback on, the
//   rasterizer-discard run) and cuts the list into segments: runs of worker-safe records that begin at a legal boundary (a draw that may begin a pass of its own, a
//   SET_RT that changes targets), hold at least SEG_MIN draws, are cut again at the next boundary after SEG_TARGET draws, and end before the first record that is not
//   worker-safe. Boundaries may sit inside CALLs (up to SEG_DMAX deep): a segment carries the call-stack position {d, pos[0..d]} (return points, then its first record).
// - DM_VK_PARALLEL_DRY=1: the list is then replayed as today (exec_ops from its start); the segments only feed the statistics (seg_log, every 300 presents).
// - DM_VK_PARALLEL_SERIALSEG=1: the list is replayed in pieces on this thread, in stream order: the records before a segment, the segment resumed from its position
//   (seg_resume: inner lists first, then each enclosing list from its return point), and so on; each piece runs a counted number of records and stops before the next
//   (g_seg_on / g_seg_left in exec_ops; every frame unwinds on g_seg_done). Nothing is ended or begun at a cut (the same pass goes on), so the commands recorded are
//   those of the uncut replay: the harness for CALL-interior resumption. Checked at every cut: the record the piece stopped at is the one the dry pass named, the
//   decisions used are the dry pass's, the replay's stream state (program, rasterizer discard, targets) at a segment's start equals the dry pass's copy
// S1d (parallel recording): the dry pass now also does the order-dependent CPU work of the records it walks, in stream order, and hands its results to the replay of the same
// records (g_sd.dec, taken in order by record index like the CALL/JUMP decisions; a record whose entry does not match is [RED]-logged and the rest of the list is done as
// without the dry pass), keeps the whole stream state (every pure state record applied to its copy through state_op_on / bind_handles_on) and gives each segment a full
// snapshot (seg_snap: rec_state_t up to the open pass, and rec_xs_t's [S] fields), and places the part cuts (g_sd.cut) so that no segment spans one. Per record:
//   REPORT_COUNTER: cnt_note and rc_note (the render-condition report table) moved here; a timestamp's query slot named here (the next of the open submission's), so
//     timestamps no longer end segments; the replay keeps the commit (rptfix's queued write, the post op: its place among the posts) and the vkCmdWriteTimestamp at the
//     named slot. Samples-passed reports: occ_key_make, the query end and the post stay the replay's, as before (occlusion, R4: left as is).
//   RESET_COUNTER: cnt_note moved here; g_occ_draws = 0 and vk_oq_reset stay the replay's (occlusion).
//   FENCE_SYNC: done here (the submit's ticket into the guest sync object); the replay skips it.
//   SIGNAL_EVENT: not moved: it is a post op only (the event value written after the submission's fence), whose place among the posts the replay keeps.
//   RENDER_COND: evaluated here, with the report table as it is at the record (rc_note's in stream order) and one fence value for the list (vk_fence_done at its start);
//     the replay sets g_rc_skip from it. Stop rule (HeyListen's, for a value the dry pass cannot know yet): a condition whose words a GPU write may still change (a write
//     in flight, vk_gpu_pending, or any GPU-writing record before it in this list) is left to the replay, evaluated as before with the table's entries as at the record;
//     it ends the segment, and until the next decided condition the draws are serial (their render-condition state is not known here). S1d2: only a condition whose
//     outcome is not fixed yet (rc_word_class): one word unknown whatever the fence and memory (a samples-passed report of this submission, a report never measured), or two
//     timestamps of this submission, fix the outcome, so such a condition is decided here after any GPU write too (OoT: ~94% of its conditions read reports of this submission)
//   UPDATE_UB: the range check and the copy of the bytes into scratch memory done here (vk_sub_update_pre); the list operation (supersede, append) stays the replay's,
//     in stream order: outside segments on the main list; for a segment (SERIALSEG) the segment's updates are applied to the main list where the segment starts (their
//     superseded indices logged), and the segment is replayed on a copy of the main list made just before (the per-segment snapshot) with the log applied by index
//     (vk_ubl_log_apply, checked against a search); at its end the copy must equal the main list (R5).
//   Part cuts (DM_VK_CB_DRAWS, DM_VK_PASS_SPLIT): planned here at records where the pass ends anyway (SET_RT to other targets, copies, fills, texture clears, dispatches,
//     a rasterizer-discard toggle) once the draws counted since the last cut reach DM_VK_CB_DRAWS (at most g_s61_maxcut ordinary cuts a submission, as before), and for
//     DM_VK_PASS_SPLIT at a draw once the dry pass's pass holds that many draws; the replay cuts there (seg_cut_take: the pass ended, part_cut) and makes none of its own.
//     Today's cut is at the first pass begin after the count (the dry pass cannot know where passes begin), so the positions move to the pass end before that begin
//     (same draws per part give or take one pass).
enum { SEG_MIN = 48, SEG_TARGET = 128, SEG_DMAX = 8 };
// why a segment ended or a run was refused: < 64 the op of the record that is not worker-safe (g_op_names), else a rule
enum { SW_TARGET = 64, SW_STORAGE, SW_SWUNK, SW_RD, SW_RDEND, SW_XFB, SW_ATOMIC, SW_TS, SW_UBLAND, SW_OOTCOPY, SW_DEPTH, SW_LISTEND, SW_BAD,
       SW_IXQ, SW_RCSTOP, SW_RCUNK, SW_PARTCUT, SW_N };
static const char *const g_sw_names[SW_N - SW_TARGET] = {"segment target (128 draws)", "program writes storage", "program's storage writes not known yet",
    "rasterizer discard", "draw after a rasterizer-discard run", "transform feedback capture", "n1x_atomic program", "timestamp report (DM_VK_TS_REPLAY)", "uniform update may land",
    "SET_RT the ootbilerp stand-in may copy at", "CALL deeper than 8", "end of the list", "bad record",
    "indirect quad draw (compute conversion)", "render condition left to the replay", "draw under a render condition left to the replay", "part cut"};
static const char *seg_why_name(int w) { return w < 64 ? (g_op_names[w] ? g_op_names[w] : "?") : w < SW_N ? g_sw_names[w - SW_TARGET] : "?"; }
// records worker-safe as such (draws, SET_RT, UPDATE_UB, REPORT_COUNTER, RENDER_COND are decided per record below): S64_ALLOW's state ops, clears, the CPU-side counter, event,
// fence and condition ops (S1d: their CPU work the dry pass's, see above)
#define SEG_SAFE (S64_ALLOW | S45_B(OP_END) | S45_B(OP_CLEAR_COLOR) | S45_B(OP_CLEAR_DS) | S45_B(OP_RESET_COUNTER) | S45_B(OP_SIGNAL_EVENT) | S45_B(OP_FENCE_SYNC) | \
    S45_B(OP_RENDER_COND) | S45_B(OP_REPORT_COUNTER))
_Static_assert((FB_KEEP & ~SEG_SAFE) == 0, "S1b: the dry pass never cuts at a record that keeps full_barrier_current (seg_resume starts each list with it 0)");
// S1d: the serial records that end the open pass (vk_end_pass_land): a part cut may go before them at no cost
#define SEG_PASS_END (S45_B(OP_COPY_B2B) | S45_B(OP_CLEAR_BUF) | S45_B(OP_COPY_B2T) | S45_B(OP_COPY_T2B) | S45_B(OP_COPY_T2T) | S45_B(OP_CLEAR_TEX) | S45_B(OP_DISPATCH))
_Static_assert((FB_KEEP & SEG_PASS_END) == 0, "S1d: a part cut is never at a record that keeps full_barrier_current");
typedef struct { int d; const uint8_t *pos[SEG_DMAX + 1]; } seg_pos;   // pos[0..d-1]: the record after each open CALL (lists 0..d-1), pos[d]: the record in list d
// S1d (parallel recording): one handed-over decision (exec_ops takes them in order: seg_dec_take)
struct seg_dec {
    uint64_t ri;                   // the record's index in the walk
    uint8_t kind, hx, hy;          // SD_*; RENDER_COND: the report table had an entry for addr (hx) / addr + 16 (hy) at the record
    int32_t v;                     // REPORT_COUNTER: a timestamp's slot (-1 none, INT32_MIN the replay allocates); RENDER_COND: 1 false, 0 true, -1 left to the replay (stop rule)
    int32_t sup;                   // UPDATE_UB: the superseded index the list operation found (set by the replay)
    vk_ubd_pre u;                  // UPDATE_UB: the range and the copy made by the dry pass
    rc_occ_t ex, ey;               // RENDER_COND: those entries as they were at the record
    uint64_t oser;                 // occ-s1: REPORT_COUNTER / RESET_COUNTER: the stream serial where the dry pass walked it (the replay thread checks its own: seg_oser_chk)
    uint64_t rseq;                 // occ-s2: a samples-passed REPORT_COUNTER: its report sequence as the dry pass named it (its record and predicate index; vk_occ2_rseq_chk)
    uint8_t gpu; rcg_t g;          // occ-s3: RENDER_COND: a GPU condition (g: its reports / constant) the replay records its draws gated for
};
typedef struct { uint64_t ri; int psplit; } seg_cut;   // S1d: a part cut before record ri (psplit: a DM_VK_PASS_SPLIT one)
// S1d: everything a segment starts from that the stream decides: the stream part of rec_state_t (up to the open pass) and rec_xs_t's [S] fields
typedef struct { rec_state_t s; int rc_skip, tf_on, rd_prev, deq_dir; uint32_t lop_kept; int32_t df_kept; } seg_snap;
#define SEG_SNAP_BYTES offsetof(rec_state_t, pass_open)
typedef struct {
    seg_pos at, endp;              // its first record's position; the first record not in it (whose depth endp.d may be lower or higher than at.d)
    uint64_t rec0, rec1;           // the two as record indices in the list's walk (stream order, every record of every list walked counted once)
    uint32_t flow0, flow1;         // CALL / JUMP decisions taken before each (g_sd.flow)
    uint32_t dec0, dec1;           // S1d: its handed-over decisions (g_sd.dec: reports, resets, fence syncs, conditions, uniform updates)
    uint32_t bh0, bh1;             // S1c (parallel recording): its deferred binds' handles as the dry pass read them (g_sd.bh words)
    uint64_t draw0; uint32_t ndraw;    // its first draw's serial in the submission, its draws
    int why;                       // what ended it (SW_* or the op)
    uint32_t wk;                   // S1e (parallel recording): why it stays on the replay thread when workers record (WR_* bits; 0: a worker may record it)
    uint32_t tg0, tg1;             // S1e: its target states (g_sd.tg: the one at its start, then each SET_RT to other targets), for the hand-out's resolution and marks
    uint32_t ncb;                  // S1e: its draws prepare_draw counts (g_cb_draws: not dropped by a false condition, not an indirect-count follower)
    uint64_t oser0;                // occ-s1: the stream serial at its first record (a worker counts its draws from it; the hand-out checks the replay thread against it)
} seg_desc;
// S1e (parallel recording): why a segment stays on the replay thread when workers record (recorded there by the same-thread resume, SERIALSEG's path): bits of seg_desc.wk
// (the dry pass, while it gathers the segment), then the list's and the hand-out's rules (seg_par_why)
enum { WR_SPREP = 1, WR_REP = 2, WR_SPRESET = 4, WR_EVENT = 8, WR_IND = 16, WR_TGT = 32, WR_Z4 = 64, WR_STATE = 128, WR_DIAG = 256, WR_GPUCOND = 512, WR_NWK = 10 };   // (occ-s3: WR_GPUCOND)
static const char *const g_wr_names[WR_NWK] = {"samples-passed report (occlusion, R4)", "counter report (its post op's place)", "samples-passed reset (occlusion)", "event (a post op)",
    "indirect draw", "more than 16 target states", "linear render target (z4 write-back)", "state at its end not known (a depth target not made yet)", "diagnostics or DM_VK_DEPTH_EQ on",
    "a GPU render condition or draws gated by one (occ-s3; workers get them in step 4)"};
typedef struct { const p_set_rt *r; uint8_t used; } seg_tg;   // S1e: a target state of a segment (r NULL: the state at its start), used: a draw or clear begins a pass on it
enum { SEG_TG_MAX = 16 };
enum { UBR = 4096 };   // the update range set (a bound of the deferred update list, seg_ub_may_land)
enum { XU_LOP = 1, XU_DF = 2, XU_DEQ = 4 };   // S1d: rec_xs_t fields the dry pass does not know at this point (g_sd.xunk)
static struct {   // the dry pass (replay thread)
    rec_state_t s; int tf_on, rd_prev;   // the stream state as the walk goes: S at the list's start, then the records' changes to it (S1d: every pure state record)
    int rc, rc_unk, deq, xunk; uint32_t lop; int32_t df;   // S1d: rec_xs_t's [S] fields as the walk goes (rc_unk: a condition left to the replay; xunk: XU_* not known)
    tex_t pk_t; const vkimg_t *pk_im; int pk_ok;   // S1d: the last depth target looked up (vk_tex_peek) and its image
    const uint8_t *ret[SEG_DMAX];        // the return points of the open CALLs
    uint64_t rec, epoch, dserial, skip, fdone;   // records walked; the submission (g_set_epoch) whose draws dserial counts; DM_VK_SKIP_OPS; S1d: vk_fence_done() at the list's start
    int cand, ok, gw_dirty; seg_desc cur;   // a segment is being gathered (cur); ok: no allocation failed in this list; S1d gw_dirty: a GPU-writing record walked in this list
    seg_desc *seg; seg_snap *snap; uint32_t nseg, capseg, capsnap;   // S1d: snap[k] is seg[k]'s (snap[nseg]: the candidate's)
    uint32_t r_flow, r_dec, r_bh;   // S1d2: nflow / ndec / nbh before the record being walked (a segment's bounds: what the replay has taken when it reaches that record)
    uint8_t *flow; uint32_t nflow, capflow;
    seg_dec *dec; uint32_t ndec, capdec;   // S1d
    seg_cut *cut; uint32_t ncut, capcut;   // S1d
    seg_tg *tg; uint32_t ntg, captg;       // S1e (parallel recording): the segments' target states (seg_desc tg0..tg1)
    seg_snap *esnap; uint32_t capesnap;    // S1e: esnap[k] the stream state at the end of seg[k] (workers on: the replay thread goes on from it after handing seg[k] out)
    int mode;                              // S1e: seg_mode() of the list (2: start snapshots for SERIALSEG's checks; 3: end snapshots for the workers; 1: neither)
    uint64_t *bh; uint32_t nbh, capbh;   // S1c (parallel recording): deferred binds' handles read in stream order ({n, src, n handles} per record; seg_bh_take)
    struct { uint64_t lo, hi; uint32_t gen; } ubr[UBR]; uint32_t ubr_gen, ubr_n; int ubd_max;
    int ts_base, ts_cap, ts_k;   // S1d: the open submission's next timestamp slot at the list's start (INT_MIN: DM_VK_TS_REPLAY, none named), one past the last, named so far
    int cbd, pd, ord, parts;     // S1d part cuts: draws counted since the last cut (g_cb_draws), draws in the pass as the dry pass sees it, ordinary cuts so far, parts left
    uint64_t oser;               // occ-s1: the stream serial as the walk goes (from the replay thread's at the list's start, every draw-type record one)
    uint64_t orseq; int rcg;     // occ-s2: the samples-passed report sequence as the walk goes (from the replay thread's, vk_occ2_rseq); occ-s3: a GPU condition gates the draws now
} g_sd;
static struct {   // statistics since the last seg_log line
    uint64_t f0, lists, refl[6], draws, seg_draws, segs, seg_max, depth[SEG_DMAX + 1], ends[SW_N], refused[SW_N], ref_runs, ref_draws, dry_ns, flow, oom;
    uint64_t pieces, segs_run, stop_bad, early, flow_bad, state_bad;   // SERIALSEG (stop_bad, early, flow_bad, state_bad: since start, not reset)
    uint64_t rep, ts_named, reset, fence, cond, cond_stop, ub, ub_seg, cuts, psplits, capped, cuts_done, sp_in_seg, open_unk;   // S1d, per frame
    uint64_t cond_const, cond_fixed, cond_mem, cond_stop_mem, cond_stop_fence, tail_cut;   // S1d2: render conditions by class (RENDER_COND in seg_dry_list); short runs after a part cut not merged
    uint64_t dec_bad, cond_off, ubl_off, ublog_off, ub_land, cut_bad, snap_bad[7];   // S1d, since start (must stay 0)
} g_sst;
// S1d: the replay's side (exec_ops' thread): the decisions and cuts of the list being replayed, the record index (g_seg_ri: the next record's; g_seg_ri_cur: this record's)
static _Thread_local seg_dec *g_seg_dec; static _Thread_local uint32_t g_seg_ndec, g_seg_di;
static _Thread_local const seg_cut *g_seg_cutv; static _Thread_local uint32_t g_seg_ncut, g_seg_ci; static _Thread_local uint64_t g_seg_cut_next = UINT64_MAX;
static _Thread_local uint64_t g_seg_ri, g_seg_ri_cur;
static _Thread_local int g_seg_inseg, g_seg_ubl_on; static _Thread_local uint64_t g_seg_inseg_rec0;   // SERIALSEG: a segment's piece is being run (from record g_seg_inseg_rec0); its update list in use
static int g_seg_chk;   // SERIALSEG: decisions checked again at replay
static struct vk_ubl *g_seg_ubl;   // SERIALSEG: the segment's uniform update list (gfx_vk_core.c)
static int seg_mode(void) {
    static long gen = -1; static int m;
    if (gen != dm_env_gen) { gen = dm_env_gen; const char *d = getenv("DM_VK_PARALLEL_DRY"), *s = getenv("DM_VK_PARALLEL_SERIALSEG");
        // S1e (parallel recording): with workers (DM_VK_PARALLEL=<n> > 0, read at device init) the dry pass is on and segments go to the workers (3); SERIALSEG=1 still wins (the harness)
        const int nm = s && atoi(s) == 1 ? 2 : vk_par_workers() > 0 ? 3 : d && atoi(d) == 1 ? 1 : 0;
        if (nm != m && nm == 3) VKLOG("S1e (parallel recording): DM_VK_PARALLEL=%d: each list is cut by the dry pass; segments a worker may record go to the %d workers (recorded "
                                      "into their own command buffers, submitted in stream order with the replay thread's chunks), the rest are recorded on the replay thread as SERIALSEG does", vk_par_workers(), vk_par_workers());
        else if (nm != m) VKLOG("S1b (parallel recording): %s", nm == 2 ? "DM_VK_PARALLEL_SERIALSEG=1: each list is cut by the dry pass and replayed in pieces on the replay thread (resumed inside CALLs; same commands); "
                                "S1d: the dry pass does the reports' and conditions' CPU work, fence syncs, uniform update copies and part cuts, segments get full snapshots (checked) and their own update lists" :
                           nm == 1 ? "DM_VK_PARALLEL_DRY=1: each list is cut by the dry pass and replayed as before, with the dry pass's decisions (S1d: reports, conditions, fence syncs, uniform update copies, part cuts)" : "dry pass off");
        m = nm; }
    return m;
}
static int seg_grow(void **pp, uint32_t *cap, uint32_t n, size_t el) {   // room for element n
    if (n < *cap) return 1;
    const uint32_t c = *cap ? *cap * 2 : 256; void *q = realloc(*pp, (size_t)c * el);
    if (!q) { if (!g_sst.oom++) VKLOG("[RED] S1b: no memory for the dry pass's lists: the list is replayed as before"); g_sd.ok = 0; return 0; }
    *pp = q; *cap = c; return 1;
}
static void seg_flow_push(int jump, int ok) { if (seg_grow((void **)&g_sd.flow, &g_sd.capflow, g_sd.nflow, 1)) g_sd.flow[g_sd.nflow++] = (uint8_t)((jump << 1) | (ok != 0)); }
int vk_cmd_parts_left(void);   // gfx_vk_core.c (S1d: the dry pass's pass splits, as prepare_draw's)
static seg_dec *seg_dec_push(uint64_t ri, int kind) {   // S1d: the next decision (valid until the next push)
    if (!seg_grow((void **)&g_sd.dec, &g_sd.capdec, g_sd.ndec, sizeof *g_sd.dec)) return NULL;
    seg_dec *e = &g_sd.dec[g_sd.ndec++]; memset(e, 0, sizeof *e); e->ri = ri; e->kind = (uint8_t)kind; e->v = INT32_MIN; e->sup = -1; return e;
}
// The deferred update list (gfx_vk_core.c vk_sub_update) lands when it is full and a new range comes. The dry pass keeps a set M of ranges that holds every range of
// the real list (seeded from it at the list's start, every UPDATE_UB's range added; an update of a range already there supersedes it in both): the real list is a
// subset, so it can land only at an UPDATE_UB whose range is new to M while M holds UBD_MAX or more. Lands elsewhere (copies, dispatches) only shrink the real list
static int seg_ub_may_land(uint64_t dst, uint64_t n) {
    const uint64_t lo = dst, hi = dst + n; const uint32_t h = (uint32_t)(((lo * 0x9E3779B97F4A7C15ull) ^ (hi * 0xC2B2AE3D27D4EB4Full)) >> 52) & (UBR - 1);
    if (g_sd.ubr_n >= UBR * 3 / 4) return 1;   // the set is full: every update counts as one that may land
    for (uint32_t k = 0; k < UBR; k++) { const uint32_t i = (h + k) & (UBR - 1);
        if (g_sd.ubr[i].gen != g_sd.ubr_gen) { const int land = (int)g_sd.ubr_n >= g_sd.ubd_max; g_sd.ubr[i].lo = lo; g_sd.ubr[i].hi = hi; g_sd.ubr[i].gen = g_sd.ubr_gen; g_sd.ubr_n++; return land; }
        if (g_sd.ubr[i].lo == lo && g_sd.ubr[i].hi == hi) return (int)g_sd.ubr_n > g_sd.ubd_max; }   // in M: in the real list (superseded: no land) or not (then that holds at most |M| - 1)
    return 1;
}
static void seg_here(seg_pos *ps, const uint8_t *p, int depth) { ps->d = depth; for (int i = 0; i < depth; i++) ps->pos[i] = g_sd.ret[i]; ps->pos[depth] = p; }
static void seg_open(const uint8_t *p, int depth, uint64_t ri) {   // a segment begins at record p
    if (g_sd.rc_unk || g_sd.xunk) { g_sst.open_unk++; return; }   // S1d: its snapshot would not be known (a condition left to the replay, a B7 / r57 register not known)
    // S1e (parallel recording): the full start snapshot only for SERIALSEG's check (mode 2): it was ~40 KB copied at every candidate, refused ones included (most of them). The
    // workers start from the replay thread's own state at the hand-out (the same state, SERIALSEG's check), and the replay thread goes on from the end snapshot (seg_close)
    if (g_sd.mode == 2 && !seg_grow((void **)&g_sd.snap, &g_sd.capsnap, g_sd.nseg, sizeof *g_sd.snap)) return;
    if (!seg_grow((void **)&g_sd.tg, &g_sd.captg, g_sd.ntg, sizeof *g_sd.tg)) return;   // S1e: its first target state
    seg_desc *c = &g_sd.cur; memset(c, 0, sizeof *c);
    seg_here(&c->at, p, depth); c->rec0 = ri; c->flow0 = g_sd.r_flow; c->dec0 = g_sd.r_dec; c->bh0 = g_sd.r_bh; c->draw0 = g_sd.dserial; c->oser0 = g_sd.oser;   // (S1d2: as before record p; occ-s1: its first draw serial)
    c->tg0 = g_sd.ntg; g_sd.tg[g_sd.ntg++] = (seg_tg){NULL, 0};   // S1e: the targets as they are at its start
    if (g_sd.mode == 2) { seg_snap *n = &g_sd.snap[g_sd.nseg];   // S1d: the full snapshot (the candidate's slot; kept if it becomes segment nseg)
        memcpy(&n->s, &g_sd.s, SEG_SNAP_BYTES); n->rc_skip = g_sd.rc; n->tf_on = g_sd.tf_on; n->rd_prev = g_sd.rd_prev; n->deq_dir = g_sd.deq; n->lop_kept = g_sd.lop; n->df_kept = g_sd.df; }
    g_sd.cand = 1;
}
// S1e (parallel recording): the stream state where segment k ends (the replay thread's when it goes on after handing k out): the dry state before the record that ends it. rd_prev
// is 0 there (the segment's draws have no rasterizer discard, and seg_draw_why already set it for a draw that ends it); an unknown B7 / r57 field keeps the segment off the workers
static void seg_end_snap(uint32_t k, seg_desc *d) {
    if (g_sd.mode != 3) return;
    if (!seg_grow((void **)&g_sd.esnap, &g_sd.capesnap, k, sizeof *g_sd.esnap)) { d->wk |= WR_STATE; return; }
    seg_snap *e = &g_sd.esnap[k];
    memcpy(&e->s, &g_sd.s, SEG_SNAP_BYTES); e->rc_skip = g_sd.rc; e->tf_on = g_sd.tf_on; e->rd_prev = 0; e->deq_dir = g_sd.deq; e->lop_kept = g_sd.lop; e->df_kept = g_sd.df;
    if (g_sd.xunk) d->wk |= WR_STATE;
}
// S1d2 (parallel recording): the ranges end where record p begins (g_sd.r_*), not at the counts now: a record that pushes its decision before it ends the segment (RENDER_COND
// left to the replay, a timestamp under DM_VK_TS_REPLAY) had its own decision counted into the segment, so the piece after it was told to begin one decision later than the
// replay was (the segment's piece stops before p, having taken none of p's): SERIALSEG's "a piece begins at decision n, the dry pass cut it at n + 1", then p's take failed
static void seg_end_at(seg_desc *d, const uint8_t *p, int depth, uint64_t ri, int why) {
    seg_here(&d->endp, p, depth); d->rec1 = ri; d->flow1 = g_sd.r_flow; d->dec1 = g_sd.r_dec; d->bh1 = g_sd.r_bh; d->why = why;
}
static void seg_close(const uint8_t *p, int depth, uint64_t ri, int why) {   // record p is not in the segment being gathered
    if (!g_sd.cand) return;
    g_sd.cand = 0; seg_desc *c = &g_sd.cur;
    if (c->ndraw >= SEG_MIN) { if (!seg_grow((void **)&g_sd.seg, &g_sd.capseg, g_sd.nseg, sizeof *g_sd.seg)) return; seg_end_at(c, p, depth, ri, why); c->tg1 = g_sd.ntg;
        seg_end_snap(g_sd.nseg, c); g_sd.seg[g_sd.nseg++] = *c; return; }   // (S1e: and its end snapshot, its target states)
    seg_desc *l = g_sd.nseg ? &g_sd.seg[g_sd.nseg - 1] : NULL;
    // the short tail of a run cut at SEG_TARGET: the segment before takes it. S1d2 (parallel recording): only when SEG_TARGET ended it (seg_boundary); a part cut also closes a
    // segment and opens the next candidate at the same record (seg_cut_at, then seg_boundary there), and merging that short run into the segment before made it span the
    // cut (SERIALSEG: "a part cut at record n is inside a segment")
    if (l && l->rec1 == c->rec0 && l->why == SW_TARGET) { l->ndraw += c->ndraw; l->ncb += c->ncb; l->wk |= c->wk; l->tg1 = g_sd.ntg; seg_end_at(l, p, depth, ri, why); seg_end_snap(g_sd.nseg - 1, l); return; }   // (S1e: its counts, refusals, target states and end with it)
    if (l && l->rec1 == c->rec0 && l->why == SW_PARTCUT && c->ndraw) g_sst.tail_cut++;
    if (c->ndraw) { g_sst.ref_runs++; g_sst.ref_draws += c->ndraw; g_sst.refused[why]++; }
}
static void seg_serial(const uint8_t *p, int depth, uint64_t ri, int why) { seg_close(p, depth, ri, why); }   // record p stays on the replay thread
static void seg_boundary(const uint8_t *p, int depth, uint64_t ri) {   // a segment may begin at record p
    if (depth > SEG_DMAX) return;
    if (g_sd.cand && g_sd.cur.ndraw < SEG_TARGET) return;
    if (g_sd.cand) seg_close(p, depth, ri, SW_TARGET);
    seg_open(p, depth, ri);
}
// S1d: part cuts. A cut before record p: the segment being gathered ends there (no segment spans a cut)
static void seg_cut_at(const uint8_t *p, int depth, uint64_t ri, int psplit) {
    if (!seg_grow((void **)&g_sd.cut, &g_sd.capcut, g_sd.ncut, sizeof *g_sd.cut)) return;
    seg_close(p, depth, ri, SW_PARTCUT);
    g_sd.cut[g_sd.ncut++] = (seg_cut){ri, psplit}; g_sd.parts--; if (!psplit) g_sd.ord++;
    g_sd.cbd = 0; g_sd.pd = 0; if (psplit) g_sst.psplits++; else g_sst.cuts++;
}
// record p ends the open pass anyway (a SET_RT to other targets, SEG_PASS_END, a rasterizer-discard toggle): the ordinary cut goes here once DM_VK_CB_DRAWS draws were counted
static void seg_pass_end_at(const uint8_t *p, int depth, uint64_t ri) {
    if (g_x_cbdraws > 0 && g_sd.cbd >= g_x_cbdraws) {
        if (g_sd.ord < g_s61_maxcut) seg_cut_at(p, depth, ri, 0);
        else { g_s61_cap++; g_sst.capped++; g_sd.cbd = 0; }   // (begin_pass's cap: counted, the count starts again)
    }
    g_sd.pd = 0;
}
static int seg_draw_why(int direct) {   // 0: the draw is worker-safe, else why not
    const rec_state_t *s = &g_sd.s; int why = 0;
    if (s->rast_discard) why = SW_RD;   // prepare_draw's runs: a pass of their own, rd_read after them (memory written by the vertex shaders)
    else if (g_sd.rd_prev) why = SW_RDEND;
    else { const int sw = eden_prog_sw((vkshader_t *const *)s->sh);   // R7: known per shader once a pipeline of it was drawn with
        if (sw == 2) why = SW_STORAGE; else if (!sw) why = SW_SWUNK;
        else if (g_sd.tf_on && s->sh[0] && (s->sh[0]->xfb || s->sh[0]->xfb_layout)) why = SW_XFB;
        else if (direct && (n1x_atomic_prog(s->sh[1]) || fst_prog(s->sh[1]))) why = SW_ATOMIC; }   // (fstage: its staged draws stay on the replay thread too)
    g_sd.rd_prev = s->rast_discard;   // (as prepare_draw's, but for every draw: a draw it drops early, or one under a false render condition, counts here)
    return why;
}
static int seg_gw_why(int why) { return why < 64 || why == SW_STORAGE || why == SW_SWUNK || why == SW_RD || why == SW_RDEND || why == SW_XFB || why == SW_ATOMIC || why == SW_IXQ || why == SW_UBLAND || why == SW_OOTCOPY; }
// S1d: rec_xs_t's [S] fields across a draw the replay will key (eden_fill_state: lop_kept, df_kept; eden_draw: deq_dir), by the same rules on the dry pass's state. maybe: the
// draw may be dropped (a condition left to the replay): a field it would change becomes unknown. Not modelled: draws prepare_draw drops after its program check (no usable
// target, pass_commit), which do not reach the key (SERIALSEG's snapshot check counts any difference)
static void seg_xs_draw(int maybe) {
    const rec_state_t *s = &g_sd.s;
    if (!(s->sh[0] || (s->sh[4] && s->sh[4]->stage == 7))) return;   // prepare_draw's skip_prog: no key
    if (s->color.logic_op != 3) { if (maybe) { if ((uint32_t)s->color.logic_op != g_sd.lop || (g_sd.xunk & XU_LOP)) g_sd.xunk |= XU_LOP; } else { g_sd.lop = (uint32_t)s->color.logic_op; g_sd.xunk &= ~XU_LOP; } }
    if (!s->rt_has_dep) return;   // no depth attachment: S.dfmt 0, no depth test (neither field changes)
    if (!g_sd.pk_ok || memcmp(&g_sd.pk_t, &s->rt_dep, sizeof g_sd.pk_t)) { g_sd.pk_t = s->rt_dep; g_sd.pk_im = vk_tex_peek(&s->rt_dep); g_sd.pk_ok = 1; }
    const vkimg_t *im = g_sd.pk_im;
    if (!im) { g_sd.xunk |= XU_DF | XU_DEQ; return; }   // the image is made when the replay begins its pass: its format not known yet
    if (im->fmt) { const int32_t f = (int32_t)im->fmt; if (maybe) { if (f != g_sd.df || (g_sd.xunk & XU_DF)) g_sd.xunk |= XU_DF; } else { g_sd.df = f; g_sd.xunk &= ~XU_DF; } }
    const int dt = (im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && s->ds.dt, df = s->ds.df;
    if (dt && s->ds.dw && (df == 2 || df == 4 || df == 5 || df == 7)) { const int nd = df == 2 || df == 4 ? 1 : 2;
        if (maybe) { if (nd != g_sd.deq || (g_sd.xunk & XU_DEQ)) g_sd.xunk |= XU_DEQ; } else { g_sd.deq = nd; g_sd.xunk &= ~XU_DEQ; } }
}
// S1e (parallel recording): the target states of the segment being gathered (g_sd.cur.tg0 .. g_sd.ntg): one per SET_RT to other targets; used when a draw or clear will begin
// a pass on it (prepare_draw / clear_*: begin_pass), which the hand-out resolves and marks in stream order on the replay thread (R6). A pass on a linear target ends with z4's
// write-back (s56_pass_end: a GPU -> guest copy and its landing, the replay thread's)
static void seg_tg_push(const p_set_rt *r) {
    if (g_sd.ntg - g_sd.cur.tg0 >= SEG_TG_MAX) { g_sd.cur.wk |= WR_TGT; return; }
    if (seg_grow((void **)&g_sd.tg, &g_sd.captg, g_sd.ntg, sizeof *g_sd.tg)) g_sd.tg[g_sd.ntg++] = (seg_tg){r, 0}; else g_sd.cur.wk |= WR_TGT;
}
int vk_z4_on(void);   // gfx_vk_core.c
static void seg_tg_use(void) {
    if (g_sd.cur.wk & WR_TGT || g_sd.ntg <= g_sd.cur.tg0) return;
    seg_tg *e = &g_sd.tg[g_sd.ntg - 1]; if (e->used) return;
    e->used = 1;
    if (vk_z4_on()) for (int i = 0; i < g_sd.s.rt_n && i < 8; i++)
        if (g_sd.s.rt_has_col[i] && (g_sd.s.rt_col[i].d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET))) g_sd.cur.wk |= WR_Z4;
}
static void seg_dry_list(const uint8_t *p, int depth) {   // exec_ops' walk: the same records in the same order
    for (;;) {
        const uint64_t ri = g_sd.rec++;
        g_sd.r_flow = g_sd.nflow; g_sd.r_dec = g_sd.ndec; g_sd.r_bh = g_sd.nbh;   // S1d2: what the replay has taken when it reaches this record (seg_open, seg_end_at)
        op_hdr h; memcpy(&h, p, sizeof h);
        if (h.op >= OP_COUNT_ || h.size < 8 || (h.size & 7) || h.size > (1u << 20)) { seg_serial(p, depth, ri, SW_BAD); return; }   // exec_ops abandons the list here
        if (h.op > 3 && (g_sd.skip >> h.op & 1)) { p += h.size; continue; }   // DM_VK_SKIP_OPS: not replayed, and no cut (it leaves full_barrier_current as it is)
        const uint8_t *a = p + 8;
        switch (h.op) {
        case OP_END: if (!depth) seg_serial(p, depth, ri, SW_LISTEND); return;
        case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); const int ok = dm_nvn_jump_ok(t);
            if (!ok && !depth) seg_serial(p, depth, ri, SW_LISTEND);   // (the cut before the decision: the piece after it takes it)
            seg_flow_push(1, ok); if (!ok) return; p = t; continue; }
        case OP_CALL: { uint64_t t; memcpy(&t, a, 8); const int ok = dm_nvn_call_check((const void *)(uintptr_t)t);
            if (ok && depth >= SEG_DMAX) seg_serial(p, depth, ri, SW_DEPTH);   // a list deeper than a segment's position can say: replayed serially
            seg_flow_push(0, ok);
            if (ok) { if (depth < SEG_DMAX) g_sd.ret[depth] = p + h.size; seg_dry_list((const uint8_t *)(uintptr_t)t, depth + 1); }
            break; }
        case OP_SET_RT: { const p_set_rt *r = (const p_set_rt *)a; const int same = rt_same_on(&g_sd.s, r);
            if (!same) seg_pass_end_at(p, depth, ri);   // S1d: set_rt ends the pass (a part cut may go here)
            if (oot_bilerp_may_copy(r)) { g_sd.gw_dirty = 1; seg_serial(p, depth, ri, SW_OOTCOPY); }   // copy_t2t from set_rt (a GPU write)
            else if (!same) seg_boundary(p, depth, ri);       // set_rt ends the pass anyway
            if (!same && g_sd.cand) seg_tg_push(r);           // S1e (parallel recording): a target state of the segment (after its boundary: in the one it opened)
            rt_assign_on(&g_sd.s, r); break; }
        case OP_BIND_PROGRAM: bind_program_on(&g_sd.s, (const p_bind_program *)a); break;
        case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_BIND_TEX1: { p_bind_handles tmp; const p_bind_handles *b = bind_rec(h.op, a, &tmp);   // S1d: applied to the dry state
            uint64_t hv[32]; const uint64_t *hp = b->h;
            if (b->defer && b->src && b->n > 0 && b->n <= 32) { memcpy(hv, (const void *)(uintptr_t)b->src, 8 * (size_t)b->n); hp = hv;   // read here, once, in stream order (S1c: bind_handles_rec)
                if (h.op == OP_BIND_TEX && seg_grow((void **)&g_sd.bh, &g_sd.capbh, g_sd.nbh + 1 + (uint32_t)b->n, sizeof *g_sd.bh)) {
                    g_sd.bh[g_sd.nbh] = (uint64_t)b->n; g_sd.bh[g_sd.nbh + 1] = b->src; memcpy(&g_sd.bh[g_sd.nbh + 2], hv, 8 * (size_t)b->n); g_sd.nbh += 2 + (uint32_t)b->n; } }
            bind_handles_on(&g_sd.s, h.op, b, hp, 0); break; }
        case OP_DRAW: case OP_DRAW_IDX: case OP_MULTI_DRAW: { int direct = 0, follower = 0, ixq = 0;
            if (h.op == OP_DRAW) { p_draw t; const p_draw *d = draw_rec(a, h.size, &t); direct = !d->ind && !d->mdi_cnt; follower = d->mdi_cnt && d->mdi_i > 0; ixq = d->ind && d->prim == 7 && !d->mdi_cnt; }
            else if (h.op == OP_DRAW_IDX) { p_draw_idx t; const p_draw_idx *d = draw_idx_rec(a, h.size, &t); direct = !d->ind && !d->mdi_cnt; follower = d->mdi_cnt && d->mdi_i > 0; }
            const int skip = !g_sd.rc_unk && g_sd.rc, keyed = !skip && !follower;   // S1d: dropped under a false condition; a draw that reaches prepare_draw
            if (keyed && g_sd.s.rast_discard != g_sd.rd_prev) seg_pass_end_at(p, depth, ri);   // S1d: prepare_draw ends the pass where rasterizer discard toggles
            else if (keyed && g_psplit > 0 && g_x_cbdraws > 0 && !g_sd.s.rast_discard && g_sd.pd >= g_psplit && g_sd.parts > 0) seg_cut_at(p, depth, ri, 1);   // S1d: DM_VK_PASS_SPLIT
            int why = seg_draw_why(direct);
            if (!why && g_sd.rc_unk) why = SW_RCUNK;   // S1d: dropped or not is the replay's to know
            if (!why && ixq) why = SW_IXQ;              // S1d: ixq_args lands the deferred updates and records a dispatch (a GPU write)
            if (why) { if (seg_gw_why(why)) g_sd.gw_dirty = 1; seg_serial(p, depth, ri, why); }
            else if (!follower) seg_boundary(p, depth, ri);   // (an indirect-count run's later records are drawn by its first: never a segment's first record)
            if (g_sd.cand) { if (!direct) g_sd.cur.wk |= WR_IND;   // S1e (parallel recording): indirect draws stay on the replay thread
                if (g_sd.rcg) g_sd.cur.wk |= WR_GPUCOND;   // occ-s3: gated by a GPU condition (the replay thread records them, step 4 hands them out)
                if (keyed) { g_sd.cur.ncb++; if (g_sd.s.sh[0] || (g_sd.s.sh[4] && g_sd.s.sh[4]->stage == 7)) seg_tg_use(); } }   // (a draw prepare_draw begins a pass for: its targets resolved at the hand-out)
            if (keyed) { seg_xs_draw(g_sd.rc_unk); g_sd.cbd++; g_sd.pd++; }   // S1d: after the boundary (the snapshot is the state before the draw)
            g_sst.draws++; g_sd.dserial++; g_sd.oser++; if (g_sd.cand) g_sd.cur.ndraw++;   // (occ-s1: the stream serial, as exec_ops)
            break; }
        case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, a + 8, 8);
            if (n && seg_ub_may_land(d, n)) { g_sd.gw_dirty = 1; seg_serial(p, depth, ri, SW_UBLAND); }
            seg_dec *e = seg_dec_push(ri, SD_UB); if (e) vk_sub_update_pre(d, a + 16, n, &e->u);   // S1d: the range check and the copy, here
            g_sst.ub++; break; }
        case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, a + 8, 8); seg_dec *e = seg_dec_push(ri, SD_REPORT);
            const int sp2 = t == NVN_COUNTER_TYPE_SAMPLES_PASSED && g_occ2 >= 2;   // occ-s2: its report sequence named here (its record; occ-s3: its predicate)
            cnt_note(1, t, ad); if (DM_GAME_RENDER_COND) rc_note(ad, t == 0 || t == 0x10 ? 1 : t == NVN_COUNTER_TYPE_SAMPLES_PASSED ? 2 : 3, sp2 && g_occ2 >= 3 ? 1 + g_sd.orseq : 0);   // S1d: in stream order, here
            if (e) { e->oser = g_sd.oser; e->rseq = sp2 ? g_sd.orseq : 0; } if (sp2) g_sd.orseq++; if (DM_GAME_RENDER_COND && g_occdiag) rc_note_diag(ad, t == 0 || t == 0x10 ? 1 : t == NVN_COUNTER_TYPE_SAMPLES_PASSED ? 2 : 3, g_sd.oser, ~0ull);   // occ-s1: its serial; occ-s0 (pass ends not known here)
            if (t == 0 || t == 0x10) {   // a timestamp: its query slot named here (g_cq, mtl_sub_counter), or with DM_VK_TS_REPLAY none (its slot depends on GPU writes at replay: serial)
                if (g_sd.ts_base == INT_MIN) seg_serial(p, depth, ri, SW_TS);
                else { if (e) e->v = g_sd.ts_base < 0 || g_sd.ts_base + g_sd.ts_k >= g_sd.ts_cap ? -1 : g_sd.ts_base + g_sd.ts_k; g_sd.ts_k++; g_sst.ts_named++; } }
            if (g_sd.cand) g_sd.cur.wk |= t == NVN_COUNTER_TYPE_SAMPLES_PASSED ? WR_SPREP : WR_REP;   // S1e (parallel recording): the report's commit stays the replay thread's
            g_sst.rep++; break; }
        case OP_RESET_COUNTER: { uint64_t t; memcpy(&t, a, 8); seg_dec *e = seg_dec_push(ri, SD_RESET); if (e) e->oser = g_sd.oser; cnt_note(0, t, 0); g_sst.reset++;   // S1d: cnt_note here (occ-s1: its serial)
            if (g_sd.cand && t == NVN_COUNTER_TYPE_SAMPLES_PASSED) g_sd.cur.wk |= WR_SPRESET;   // S1e: the occlusion epoch moves at it (replay thread)
            break; }
        case OP_SIGNAL_EVENT: if (g_sd.cand) g_sd.cur.wk |= WR_EVENT; break;   // S1e (parallel recording): a post op: its place among the posts is the replay thread's
        case OP_CLEAR_COLOR: case OP_CLEAR_DS: if (g_sd.cand && !(!g_sd.rc_unk && g_sd.rc)) seg_tg_use(); if (g_sd.cand && g_sd.rcg) g_sd.cur.wk |= WR_GPUCOND; break;   // S1e: a clear begins a pass (begin_pass) unless a false condition drops it (occ-s3: gated)
        case OP_FENCE_SYNC: { (void)seg_dec_push(ri, SD_FENCE); sync_t *s; memcpy(&s, a, 8); if (!g_nvn_fs_skip) { s->fence = g_nvn_fs_ticket; s->fenced = 1; } g_sst.fence++; break; }   // S1d: here
        case OP_RENDER_COND: { const p_render_cond *rc = (const p_render_cond *)a; seg_dec *e = seg_dec_push(ri, SD_COND); g_sst.cond++;   // S1d: evaluated here
            if (g_sd.cand && g_sd.rcg) g_sd.cur.wk |= WR_GPUCOND;   // occ-s3: it ends a GPU condition's gating (the replay thread's state after the segment)
            g_sd.rcg = 0;
            if (rc->mode >= 2) { const int v = rc->mode == 3; if (e) e->v = v; g_sd.rc = v; g_sd.rc_unk = 0; g_sst.cond_const++; break; }
            const rc_occ_t *ex = rc_occ_find(rc->addr), *ey = rc_occ_find(rc->addr + 16);
            if (e) { if (ex) { e->ex = *ex; e->hx = 1; } if (ey) { e->ey = *ey; e->hy = 1; } }
            if (g_occ2 >= 3) { rcg_t g;   // occ-s3: a condition on samples-passed reports in the predicate ring: the GPU's (named by its report sequences); its constant word read
                if (rcg_class(rc, ex, ey, g_sd.orseq, &g)) {   // from memory here unless a GPU write may still change it (then the replay classifies it again, the stop rule below)
                    const uint64_t ya = rc->addr + (ey ? 0 : 16);
                    if (!(g.ymem && (g_sd.gw_dirty || vk_gpu_pending(ya, 8)))) {
                        if (e) { e->gpu = 1; e->g = g; e->v = 0; } g_sd.rc = 0; g_sd.rc_unk = 0; g_sd.rcg = 1; if (g_sd.cand) g_sd.cur.wk |= WR_GPUCOND;
                        if (g_occdiag) occd_cond(rc, ex, ey, g_sd.fdone, g_sd.oser, 1);   // occ-s0
                        g_sst.cond_fixed++; break; }
                    if (e) e->v = -1; g_sd.rc_unk = 1; g_sst.cond_stop++; g_sst.cond_stop_mem++; seg_serial(p, depth, ri, SW_RCSTOP); break; } }   // (its constant word may change: the replay's)
            // S1d2 (parallel recording): the stop rule only for a condition whose replay outcome is not fixed yet. rc_eval is "known" only when both words are, so one word
            // unknown whatever the fence and memory (a report of this submission, one never measured) fixes "keep drawing", and two timestamps of this submission fix 0 == 0;
            // such a condition reads no memory that matters and no fence the replay could see differently, so the dry pass decides it as the replay would (the same rc_eval,
            // same result) after any GPU write. Left to the replay: a word read from memory a GPU write may still change, or a report in flight whose completion the replay may see
            const uint64_t cur = mtl_last_fence() + 1; const int cx = rc_word_class(ex, g_sd.fdone, cur), cy = rc_word_class(ey, g_sd.fdone, cur);
            if (cx == RCW_UNK || cy == RCW_UNK || (cx == RCW_ZERO && cy == RCW_ZERO)) g_sst.cond_fixed++;
            else if (g_sd.gw_dirty || vk_gpu_pending(rc->addr, 32)) { if (e) e->v = -1; g_sd.rc_unk = 1; g_sst.cond_stop++; if (cx == RCW_MEM || cy == RCW_MEM) g_sst.cond_stop_mem++; else g_sst.cond_stop_fence++;
                seg_serial(p, depth, ri, SW_RCSTOP); break; }   // the stop rule
            else g_sst.cond_mem++;
            if (g_occdiag) occd_cond(rc, ex, ey, g_sd.fdone, g_sd.oser, 1);   // occ-s0: the census of a condition the dry pass decides
            uint64_t x, y; int known; const int v = rc_eval(rc, ex, ey, g_sd.fdone, 1, &x, &y, &known); render_cond_note(rc, known, v, x, y);
            if (g_occ2 >= 3) { g_o3c.cond++; rcg_census(ex, ey); }   // occ-s3: a CPU condition the dry pass decides (counted once, here)
            if (e) e->v = v; g_sd.rc = v; g_sd.rc_unk = 0; break; }
        case OP_TF_STATE: seg_serial(p, depth, ri, h.op); g_sd.tf_on = (((const p_tf_state *)a)->what & 3) == 0 || (((const p_tf_state *)a)->what & 3) == 3; break;   // as tf_state
        default:
            if (h.op == OP_DRAW_TEX) g_sd.oser++;   // occ-s1: a draw-type record (not replayed): the stream serial, as exec_ops
            if (!(SEG_SAFE >> h.op & 1)) { if (h.op < 64 && (SEG_PASS_END >> h.op & 1)) seg_pass_end_at(p, depth, ri);   // S1d: a part cut may go before it
                g_sd.gw_dirty = 1; seg_serial(p, depth, ri, h.op); }   // copies, fills, texture clears, dispatches, DRAW_TEX, DLSS, TF_BIND, unknown ops
            else (void)state_op_on(&g_sd.s, h.op, a, 0);   // S1d: the pure state records on the dry state (BARRIER, clears, events: nothing)
            break;
        }
        p += h.size;
    }
}
static int seg_refuse(void) {   // why a list is not cut (Stage 1 "Threads": parallel recording off when ...), 0 none
    if (!vk_eden_mode()) return 1;
    // NVN2 (dm_sdk_ounce) is NOT refused (11-21): Ocarina of Time, the target, is an NVN2 title (0400... id) and every list was refused. The dry pass and SERIALSEG run on the
    // replay thread in stream order, so the NVN2 paths (s2tile_refresh, residency, image_version_advance, F0 drop accounting) see the same calls; the worker step must make them
    // per segment or keep their draws serial (design doc "NVN2 specifics"). Slot 2 of the refusal counts stays, always 0
    if (nvn_eden_resolution_x2()) return 3;   // resolution scale (pass_scale, materialize_scale cross passes)
    if (fg_frames_left) return 4;   // frame-gen capture
    if (g_bis_on || g_bis_trace >= 0 || g_oplog_n || g_oplog_seg) return 5;   // diagnostics that follow the replay order (item 33; the rest with the workers)
    return 0;
}
// S1c (parallel recording): what must also keep a list's segments off the workers once they record (not the dry pass: same thread, same order): diagnostics whose tables,
// samples or logs are the replay thread's and are not made per thread (paydiag g_pd_*, atomdiag g_atom, wtrace g_wt_* / vsres, replaydiag g_rd_* and par0, s15 g_ss_*,
// vbytes, the vtxmir A/B), and r57 DM_VK_DEPTH_EQ (g_deq_dir: S1d tracks it per segment, but its rule follows the depth attachment's image, see seg_xs_draw). Nothing calls
// it yet: the worker step does
// S1e (parallel recording): used by the hand-out (seg_par_pieces), per list. Added: diagnostics that record into the replay thread's command buffer or tables at pass ends or
// draws (pass timestamps and ootgap, crumbs, pass / exposure / indirect-argument dumps, pass traces and listings, DM_VK_SKIP_FS, drawprof, the idxmir A/B, writeback traces,
// DM_VK_UBO_LOG, the ubd / texture / NDSR watches, pkm7), and s56's block-linear write-back (vk_s56_mirror at every pass end; z4's linear targets are refused per segment, WR_Z4)
static int seg_worker_refuse(void) {
    static int vsres = -1; if (vsres < 0) { const char *e = getenv("NXN1X_VSRES_DIAG"); vsres = e && *e && strtoull(e, NULL, 16) != 0; }   // (vsres_diag reads it once too)
    static int dprof = -1; if (dprof < 0) { const char *e = getenv("NXN1X_DRAWPROF"); dprof = e && *e && strtoull(e, NULL, 16) != 0; }   // (drawprof_begin reads it once too)
#if defined(DM_NXN1X)
    // frame generation's per-draw collection (fg_observe_draw: fg_auto's camera and motion tables are the replay thread's). fgpar (NPARK 11-21: frame generation
    // on must not turn parallel recording off): while ootdlss hands DLSS-G its inputs (ootdl_fg_owner), fg_auto stands aside, and what remains per draw or pass
    // (ootdl_draw: its own lock; fg_observe_draw and fg_profile_pass return on a worker) is safe there; a title profile or a capture keeps the old rule
    if (nvn_fg_collecting() && (!ootdl_fg_owner() || nvn_fg_get_profile() || fg_frames_left)) return 1;
#endif
    if (ootd_on()) return 1;   // ootdlss D1a survey: its counts follow the stream order on the replay thread
    extern uint64_t g_ubd_watch; int vk_pass_times_active(void);
    // S1e2 (11-21: OoT with DM_VK_PARALLEL=2/4 kept every segment off the workers as "diagnostics or DM_VK_DEPTH_EQ on" with no diagnostic set): the reason is
    // named (once per change of the mask), and g_vb_on, -1 until vbytes_on first runs, is read through vbytes_on() (it counted as on before)
    const uint32_t m = (uint32_t)(paydiag_on() != 0) | (uint32_t)(g_atom_on > 0) << 1 | (uint32_t)(g_nwt != 0) << 2 | (uint32_t)(vsres != 0) << 3 | (uint32_t)(g_rdiag > 0) << 4 |
        (uint32_t)(g_ss_on > 0) << 5 | (uint32_t)(g_deq > 0) << 6 | (uint32_t)(vbytes_on() != 0) << 7 | (uint32_t)(g_vmir != 0) << 8 | (uint32_t)(vk_pass_times_active() != 0) << 9 |
        (uint32_t)(crumbs_on() != 0) << 10 | (uint32_t)(g_pd_state != 0) << 11 | (uint32_t)(g_ex_state != 0) << 12 | (uint32_t)(g_in_state != 0) << 13 |
        (uint32_t)(g_x_tracepass >= 0 || g_x_listpasses || g_x_traceframe[0] || g_x_tracesize[0]) << 14 | (uint32_t)(g_nskip != 0) << 15 | (uint32_t)(dprof != 0) << 16 |
        (uint32_t)(g_imir != 0) << 17 | (uint32_t)(wb_trace != 0) << 18 | (uint32_t)(g_ubolog_key != 0) << 19 | (uint32_t)(g_ubd_watch != 0) << 20 | (uint32_t)(g_watch_n > 0) << 21 |
        (uint32_t)(g_ndsr_wn > 0) << 22 | (uint32_t)(g_pkm7_on != 0) << 23 | (uint32_t)(dm_profile_key(3) != 0) << 24;
    static uint32_t said = ~0u; if (m != said) { said = m; static const char *const nm[25] = {"DM_VK_PAYDIAG", "NXN1X_ATOMDIAG", "wtrace", "NXN1X_VSRES_DIAG", "DM_VK_REPLAYDIAG", "DM_VK_SLOWSUB",
        "DM_VK_DEPTH_EQ", "NXN1X_VBYTES", "NXN1X_VTX_MIRROR", "pass timestamps (DM_VK_PASS_TIMES / ootgap)", "crumbs", "pass dump", "exposure dump", "indirect dump", "pass trace / listing", "DM_VK_SKIP_FS",
        "NXN1X_DRAWPROF", "NXN1X_IDX_MIRROR", "writeback trace", "DM_VK_UBO_LOG", "DM_VK_UBD_WATCH", "watch", "NDSR watch", "pkm7", "DM_VK_S56"};
        char b[600]; size_t k = 0; b[0] = 0; for (int i = 0; i < 25; i++) if (m >> i & 1) k += (size_t)snprintf(b + k, sizeof b - k, " %s", nm[i]);
        VKLOG("S1e2: segments %s for workers%s%s", m ? "refused" : "allowed", m ? " by:" : "", m ? b : " (no diagnostic on)"); }
    return m != 0;
}
static void seg_dry(const uint8_t *p) {
    if (g_sd.epoch != g_set_epoch) { g_sd.epoch = g_set_epoch; g_sd.dserial = 0; }   // (vk_exec_reset bumps g_set_epoch per submission)
    memcpy(&g_sd.s, &S, sizeof g_sd.s); g_sd.tf_on = g_tf_on; g_sd.rd_prev = g_rd_prev; g_sd.skip = skip_ops();
    g_sd.rc = g_rc_skip; g_sd.rc_unk = 0; g_sd.deq = g_deq_dir; g_sd.lop = g_rec->X.lop_kept; g_sd.df = g_rec->X.df_kept; g_sd.xunk = 0; g_sd.pk_ok = 0;   // S1d
    g_sd.fdone = vk_fence_done(); g_sd.gw_dirty = 0;   // S1d
    g_sd.oser = g_oser;   // occ-s1: the stream serial from the replay thread's at the list's start (the dry pass names the list's serials from here)
    g_sd.orseq = vk_occ2_rseq(); g_sd.rcg = g_rec->X.rcg_on;   // occ-s2: the report sequence likewise; occ-s3: the GPU condition the replay thread has on
    g_sd.ts_base = vk_ts_replay_on() ? INT_MIN : vk_sub_ts_next(&g_sd.ts_cap); g_sd.ts_k = 0;   // S1d
    g_sd.cbd = g_cb_draws; g_sd.pd = S.pass_open ? (int)g_pass_draws : 0; g_sd.ord = g_ord_parts; g_sd.parts = vk_cmd_parts_left();   // S1d: as begin_pass would count
    g_sd.rec = 0; g_sd.cand = 0; g_sd.ok = 1; g_sd.nseg = g_sd.nflow = g_sd.ndec = g_sd.ncut = g_sd.nbh = 0; g_sd.ntg = 0;   // (S1e: and the target states)
    { static uint64_t r[2 * 1024]; int cap = 0; const int n = vk_ubd_ranges(r, 1024, &cap); g_sd.ubd_max = cap; g_sd.ubr_n = 0;
      if (!++g_sd.ubr_gen) { memset(g_sd.ubr, 0, sizeof g_sd.ubr); g_sd.ubr_gen = 1; }
      for (int i = 0; i < n; i++) (void)seg_ub_may_land(r[2 * i], r[2 * i + 1] - r[2 * i]); }
    seg_dry_list(p, 0);
    if (g_sd.cand) { g_sd.cand = 0; if (g_sd.cur.ndraw) { g_sst.ref_runs++; g_sst.ref_draws += g_sd.cur.ndraw; g_sst.refused[SW_LISTEND]++; } }   // (every way out at depth 0 cuts; kept for safety)
    g_sst.flow += g_sd.nflow;
    for (uint32_t k = 0; k < g_sd.nseg; k++) { const seg_desc *s = &g_sd.seg[k];
        g_sst.segs++; g_sst.seg_draws += s->ndraw; if (s->ndraw > g_sst.seg_max) g_sst.seg_max = s->ndraw; g_sst.depth[s->at.d]++; g_sst.ends[s->why]++; }
}
// ---- S1d (parallel recording): the replay's side of the handed-over decisions and cuts
static void seg_handover_drop(void) { g_seg_dec = NULL; g_seg_cutv = NULL; g_seg_cut_next = UINT64_MAX; g_seg_cuts = 0; }
static void seg_cut_take(void) {   // the dry pass's next part cut is before the record g_seg_ri_cur
    const seg_cut *c = &g_seg_cutv[g_seg_ci++]; g_seg_cut_next = g_seg_ci < g_seg_ncut ? g_seg_cutv[g_seg_ci].ri : UINT64_MAX;
    if (g_seg_inseg && g_seg_ri_cur != g_seg_inseg_rec0) { if (!g_sst.cut_bad++) VKLOG("[RED] S1d: a part cut at record %llu is inside a segment (from %llu)", (unsigned long long)g_seg_ri_cur, (unsigned long long)g_seg_inseg_rec0); }
    vk_end_pass(); part_cut(c->psplit); g_sst.cuts_done++;
}
static int seg_hot(const uint8_t *p) {   // exec_ops, before each record while g_seg_hot: 1 = the piece ends before it
    if (g_wk) {   // S1e (parallel recording): a worker stops at the first record after anything that keeps its segment off it (recorded again on the replay thread)
        if (g_vk_tex_refused) seg_wk_abort(WA_IMG_NEW + g_vk_tex_refused - 1);
        if (g_wk_abort) { g_seg_done = 1; g_seg_at = NULL; return 1; } }
    if (g_seg_on && seg_stop(p)) return 1;
    g_seg_ri_cur = g_seg_ri++;
    while (g_seg_cutv && g_seg_ri_cur == g_seg_cut_next) seg_cut_take();
    return 0;
}
static seg_dec *seg_dec_take(int kind) {
    if (!g_seg_dec) return NULL;
    if (g_seg_di < g_seg_ndec && g_seg_dec[g_seg_di].ri == g_seg_ri_cur && g_seg_dec[g_seg_di].kind == kind) return &g_seg_dec[g_seg_di++];
    if (g_wk) { static _Thread_local seg_dec none; memset(&none, 0, sizeof none); none.v = 0; none.sup = -1; seg_wk_abort(WA_DEC); return &none; }   // S1e (parallel recording): a worker does none of the
    // record's work itself (an empty decision: no copy, no condition); its segment is recorded on the replay thread
    if (!g_sst.dec_bad++) VKLOG("[RED] S1d: the replay's record %llu (decision kind %d) is not the dry pass's decision %u of %u: the rest of the list is done as without the dry pass",
                                (unsigned long long)g_seg_ri_cur, kind, g_seg_di, g_seg_ndec);
    seg_handover_drop(); return NULL;
}
static void seg_oser_chk(const seg_dec *d) {   // occ-s1: the dry pass owns the serials: the replay thread's must be the one it walked the record at (and follows it)
    if (g_oser == d->oser || (g_wk && g_wk_abort)) return;   // (a worker's empty decision after an abort: its segment is recorded again from its start serial)
    if (!__atomic_fetch_add(&g_occ2_ser_off, 1, __ATOMIC_RELAXED)) VKLOG("[RED] occ-s1: the replay thread's stream serial %llu at record %llu is not the dry pass's %llu: the dry pass's taken", (unsigned long long)g_oser, (unsigned long long)g_seg_ri_cur, (unsigned long long)d->oser);
    g_oser = d->oser;
}
static void seg_report(seg_dec *d, uint64_t t, uint64_t ad) {   // REPORT_COUNTER the dry pass walked (cnt_note, rc_note done there): the commit and the recording, here
    seg_oser_chk(d);   // occ-s1
    if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED && g_occ2 >= 2) { vk_occ2_rseq_chk(d->rseq); if (g_seg_inseg) g_sst.sp_in_seg++; }   // occ-s2: the dry pass's report sequence (no occkey on this path)
    else if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) { occ_key_make(ad); if (g_seg_inseg) g_sst.sp_in_seg++; }   // occlusion: as before (R4 left for later)
    if ((t == 0 || t == 0x10) && d->v != INT32_MIN) vk_sub_counter_q(ad, (uint32_t)t, d->v); else mtl_sub_counter(ad, (uint32_t)t);   // a timestamp at the dry pass's slot
    g_occ_key = g_occ_key2 = 0;
}
static void seg_cond(seg_dec *d, const p_render_cond *rc) {   // RENDER_COND the dry pass evaluated (or left to the replay: v < 0)
    if (d->v < 0) { render_cond_e(rc, d->hx ? &d->ex : NULL, d->hy ? &d->ey : NULL); return; }   // the stop rule: as before, with the table as at the record
    g_rec->X.rcg_on = 0;   // occ-s3: a new condition ends the GPU one's gating
    if (d->gpu) {   // occ-s3: a GPU condition the dry pass named: its predicate taken here (a resolve forced first while its report is pending), else the CPU's fallback
        const rc_occ_t *ex = d->hx ? &d->ex : NULL, *ey = d->hy ? &d->ey : NULL; g_o3c.cond++;
        if (g_wk) { seg_wk_abort(WA_DEC); g_rc_skip = 0; return; }   // (never: such a segment stays on the replay thread, WR_GPUCOND)
        if (rcg_apply(&d->g)) { render_cond_note(rc, 0, 0, 0, 0); return; }
        rcg_census(ex, ey); (void)vk_occ3_wait(d->g.rx);
        uint64_t x, y; int known; g_rc_skip = rc_eval(rc, ex, ey, RC_FD_LIVE, 1, &x, &y, &known); render_cond_note(rc, known, g_rc_skip, x, y); return; }
    g_rc_skip = d->v;
    o3f_note(rc, d->hx ? &d->ex : NULL, d->hy ? &d->ey : NULL, d->v);   // (occ-s3fast diagnostic: a CPU condition the dry pass decided)
    if (g_seg_chk && rc->mode < 2) { uint64_t x, y; int known;   // SERIALSEG: evaluated again (same entries, same fence value): only guest memory may have changed since
        if (rc_eval(rc, d->hx ? &d->ex : NULL, d->hy ? &d->ey : NULL, g_sd.fdone, 0, &x, &y, &known) != d->v && !g_sst.cond_off++)
            VKLOG("[RED] S1d: a render condition at %#llx evaluated at replay differs from the dry pass's (%s)", (unsigned long long)rc->addr, d->v ? "dropped" : "drawn"); }
}
static void seg_ub(seg_dec *d) {   // UPDATE_UB the dry pass copied: the list operation, in stream order
    if (g_seg_ubl_on) { const int ok = vk_ubl_log_apply(&d->u, d->sup, g_seg_chk);   // a segment's own list: the logged supersede (SERIALSEG; S1e: a worker's segment)
        if (!ok && g_wk) { seg_wk_abort(WA_DEC); return; }   // S1e (parallel recording): a worker's list off the log: recorded again on the replay thread
        if (!ok && !g_sst.ublog_off++)
            VKLOG("[RED] S1d: a segment's logged uniform update %#llx +%llu (superseding %d) is not what its list holds", (unsigned long long)d->u.dst, (unsigned long long)d->u.n, d->sup);
        return; }
    int landed; d->sup = vk_sub_update_apply(&d->u, &landed);
}
static void seg_ub_open(const seg_desc *s) {   // SERIALSEG, where segment s starts: its updates applied to the main list (logged), and the segment replayed on a copy made before
    if (!g_seg_dec) return;
    if (!g_seg_ubl) g_seg_ubl = vk_ubl_new();
    vk_ubl_snapshot(g_seg_ubl);
    for (uint32_t i = s->dec0; i < s->dec1 && i < g_sd.ndec; i++) if (g_sd.dec[i].kind == SD_UB) { int landed;
        g_sd.dec[i].sup = vk_sub_update_apply(&g_sd.dec[i].u, &landed); g_sst.ub_seg++;
        if (landed && !g_sst.ub_land++) VKLOG("[RED] S1d: a segment's uniform update landed the list (the dry pass's bound said it cannot)"); }
    vk_ubl_use(g_seg_ubl); g_seg_ubl_on = 1;
}
static void seg_ub_close(void) {
    if (!g_seg_ubl_on) return;
    vk_ubl_use(NULL); g_seg_ubl_on = 0;
    if (!vk_ubl_same(g_seg_ubl) && !g_sst.ubl_off++) VKLOG("[RED] S1d: a segment's uniform update list at its end is not the main list");
}
// SERIALSEG: the replay's state where segment k starts equals its snapshot (stream state up to the open pass, rec_xs_t's [S] fields)
static void seg_check_snap(uint32_t k) {
    const seg_snap *n = &g_sd.snap[k]; const seg_desc *s = &g_sd.seg[k]; unsigned bad = 0;
    if (memcmp(&S, &n->s, SEG_SNAP_BYTES)) bad |= 1;
    if (g_rc_skip != n->rc_skip) bad |= 2;
    if (g_tf_on != n->tf_on) bad |= 4;
    if (g_rd_prev != n->rd_prev) bad |= 8;
    if (g_deq_dir != n->deq_dir) bad |= 16;
    if (g_rec->X.lop_kept != n->lop_kept) bad |= 32;
    if (g_rec->X.df_kept != n->df_kept) bad |= 64;
    if (!bad) return;
    for (int i = 0; i < 7; i++) if (bad >> i & 1) g_sst.snap_bad[i]++;
    if (g_sst.state_bad++ < 8) {
        const char *f = "-";
        if (bad & 1) { static const struct { size_t o; const char *n; } fl[] = {{offsetof(rec_state_t, prog), "program"}, {offsetof(rec_state_t, ubo), "uniform buffers"},
              {offsetof(rec_state_t, ssbo), "storage buffers"}, {offsetof(rec_state_t, tex), "textures"}, {offsetof(rec_state_t, samp), "samplers"}, {offsetof(rec_state_t, img), "images"},
              {offsetof(rec_state_t, blend), "blend / colour / depth-stencil state"}, {offsetof(rec_state_t, vattr), "vertex state"}, {offsetof(rec_state_t, vp), "viewports / scissor"},
              {offsetof(rec_state_t, sref), "stencil / fixed state"}, {offsetof(rec_state_t, tex_base), "pool bases"}, {offsetof(rec_state_t, push), "push constants"},
              {offsetof(rec_state_t, rt_col), "render targets"}, {offsetof(rec_state_t, row_stride), "copy strides"}};
          const uint8_t *x = (const uint8_t *)&S, *y = (const uint8_t *)&n->s; size_t o = 0; while (o < SEG_SNAP_BYTES && x[o] == y[o]) o++;
          for (size_t i = 0; i < sizeof fl / sizeof fl[0]; i++) if (o >= fl[i].o) f = fl[i].n; }
        VKLOG("[RED] S1d: the state at a segment's start (depth %d, record %llu) is not its snapshot: stream state %s (first: %s), render condition %s, transform feedback %s, "
              "rasterizer-discard run %s, depth direction %s, logic op %s, depth format %s", s->at.d, (unsigned long long)s->rec0, bad & 1 ? "differs" : "same", f,
              bad & 2 ? "differs" : "same", bad & 4 ? "differs" : "same", bad & 8 ? "differs" : "same", bad & 16 ? "differs" : "same", bad & 32 ? "differs" : "same", bad & 64 ? "differs" : "same");
    }
}
static void seg_resume(const seg_pos *at, int i) {   // design "Resuming inside nested lists": list i+1 first, then list i from its position
    if (i < at->d) { seg_resume(at, i + 1); if (g_seg_done) return; }
    g_par_depth = i; exec_ops(at->pos[i]);
}
// one piece: records [rec0, rec1) from position at (rec1 0: to the list's end); want: the record it must stop at. 0 when the list ended before that
static int seg_piece(const seg_pos *at, uint64_t rec0, uint64_t rec1, uint32_t flow0, uint32_t bh0, uint32_t dec0, const uint8_t *want) {
    if (g_seg_flow && g_seg_fi != flow0) { if (!g_sst.flow_bad++) VKLOG("[RED] S1b: a piece begins at CALL/JUMP decision %u, the dry pass cut it at %u", g_seg_fi, flow0); g_seg_fi = flow0; }
    if (g_seg_bh && g_seg_bhi != bh0) { if (!g_seg_bh_bad++) VKLOG("[RED] S1c: a piece begins at deferred-handle word %u, the dry pass cut it at %u", g_seg_bhi, bh0); g_seg_bhi = bh0; }   // S1c (parallel recording)
    if (g_seg_dec && g_seg_di != dec0) { if (!g_sst.dec_bad++) VKLOG("[RED] S1d: a piece begins at decision %u, the dry pass cut it at %u", g_seg_di, dec0); g_seg_di = dec0; }   // S1d
    g_seg_on = rec1 != 0; g_seg_left = rec1 ? rec1 - rec0 : 0; g_seg_done = 0; g_seg_at = NULL; g_sst.pieces++; g_seg_ri = rec0;   // (S1d: the records numbered from rec0)
    seg_resume(at, 0);
    const int done = g_seg_done; const uint8_t *stop = g_seg_at; g_seg_on = 0; g_seg_done = 0; g_seg_left = 0; g_par_depth = 0;
    if (!rec1) return 1;
    if (!done) { if (!g_sst.early++) VKLOG("[RED] S1b: the list ended before record %llu, where the dry pass cut it (the rest of its pieces are not run)", (unsigned long long)rec1); return 0; }
    if (stop != want) { if (g_sst.stop_bad++ < 8) VKLOG("[RED] S1b: a piece stopped at record %llu = %p, the dry pass cut at %p", (unsigned long long)rec1, (const void *)stop, (const void *)want); }
    return 1;
}
static void seg_pieces(const uint8_t *p) {   // SERIALSEG: the list in pieces, in stream order
    seg_pos at0; memset(&at0, 0, sizeof at0); at0.pos[0] = p;
    const seg_pos *from = &at0; uint64_t r = 0; uint32_t fl = 0, bh = 0, dc = 0;
    for (uint32_t k = 0; k < g_sd.nseg; k++) { const seg_desc *s = &g_sd.seg[k];
        if (s->rec0 > r && !seg_piece(from, r, s->rec0, fl, bh, dc, s->at.pos[s->at.d])) return;   // the serial records before it
        g_seg_ri_cur = s->rec0; while (g_seg_cutv && g_seg_cut_next == s->rec0) seg_cut_take();   // S1d: a part cut where it starts comes before it
        seg_check_snap(k);
        seg_ub_open(s); g_seg_inseg = 1; g_seg_inseg_rec0 = s->rec0;   // S1d: its own update list
        const int ok = seg_piece(&s->at, s->rec0, s->rec1, s->flow0, s->bh0, s->dec0, s->endp.pos[s->endp.d]);
        g_seg_inseg = 0; seg_ub_close();
        if (!ok) return;
        g_sst.segs_run++; from = &s->endp; r = s->rec1; fl = s->flow1; bh = s->bh1; dc = s->dec1; }
    (void)seg_piece(from, r, 0, fl, bh, dc, NULL);   // the rest of the list
}
// ---------------- S1e (parallel recording): segments recorded on the workers ----------------
// design\PARALLEL_REPLAY_DESIGN.md S1e. With DM_VK_PARALLEL=<n> > 0 (seg_mode 3) each top-level list is cut by the dry pass and replayed as SERIALSEG does (serial pieces and
// same-thread segments on the replay thread), except that a segment a worker may record (seg_desc.wk 0 and the list's rules, seg_par_why) is handed out (seg_par_handout):
// - in stream order on the replay thread, the work of its records that a worker skips: its pass ended; R6's target resolution of every target state it draws to (made, first
//   use, marks, first transitions into the replay thread's chunk) and pass_commit's marks; ootbilerp's kept targets at its SET_RTs; r150's fresh copies; the queued reports flushed
//   (its uniform reads see them); g_occ_draws and g_cb_draws counted; its updates applied to the main list (logged) after a copy of the list for it (as SERIALSEG);
// - a job (seg_job) with copies of what it starts from: the replay thread's state now (S up to the open pass, rec_xs_t's [S] fields: equal to the dry pass's, SERIALSEG's check),
//   the update list, its CALL/JUMP decisions, deferred handles and decisions, the resolved targets, a frozen GPU-write view (gfx_vk_core.c vk_par_view);
// - gfx_vk_core.c vk_par_handout: its place among the post ops, the replay thread's R chunk ended and listed, the worker's W chunk listed, a new R chunk begun; then the replay
//   thread goes on after it from the dry pass's end snapshot (the state the segment's records leave), its memo generations bumped.
// The worker (seg_job_main) begins its chunk with a full barrier, records the segment's records from its position (seg_resume) on its own contexts, ends its pass, a full barrier,
// ends the chunk. At the part's end (vk_cmd_split / mtl_sub_commit: vk_seg_part_join) the replay thread runs queued jobs itself (helper context), waits for the rest, records any
// that failed again itself (no image creation refused there; [RED] for a chunk that could not begin or end), puts each segment's post ops at its place, and the part is submitted
// with all its chunks in stream order in one vkQueueSubmit
enum { SEG_RT_OFF = offsetof(rec_state_t, rt_col), SEG_RT_LEN = offsetof(rec_state_t, row_stride) - offsetof(rec_state_t, rt_col) };
typedef struct { uint8_t rt[SEG_RT_LEN]; pass_tgt col[8], dep; } seg_wtg;   // a target state the hand-out resolved: its bytes of rec_state_t, its views and images
typedef struct seg_job {
    seg_desc d; seg_snap snap; const uint8_t *want;   // the segment, the state it starts from (the replay thread's at the hand-out), the record it must stop at
    uint8_t *flow; uint32_t nflow, capflow; seg_dec *dec; uint32_t ndec, capdec; uint64_t *bh; uint32_t nbh, capbh;   // copies of its ranges of g_sd's lists
    struct vk_ubl *ubl; vk_par_view *view; seg_wtg tg[SEG_TG_MAX]; int ntg;
    int slot, ci, worker, rerun, why; uint64_t ns, oser_end;   // (occ-s1: oser_end the stream serial its recording ended at, checked at the join) its slot, its W chunk, the worker given it; recorded again on the replay thread; why it could not be recorded (WA_*), time
} seg_job;
static seg_job **g_jobs; static int g_njobs, g_pj;   // jobs made so far (reused), those handed out in the open part
static _Thread_local seg_job *g_wk_job;   // the job this thread records (g_wk_rerun: it is the replay thread's own recording of a failed one)
static _Thread_local struct vk_ubl *tl_wubl;   // a worker's own copy of its segment's update list
static struct {   // since the last S1e log line (handed, kept by reason, record time, the join's wait and help, chunks, recorded again by reason); since start: failures, lost
    uint64_t handed, hdraws, kept, kdraws, kwhy[WR_NWK + 2], wk_ns, wait_ns, helped, parts, chunks, rerun, rwhy[WA_N], handout_ns;
    uint64_t red, lost;
} g_pst;
static const char *const g_wa_names[WA_N] = {"-", "image not made yet", "archived image", "storage upgrade", "sparse tiles", "targets not resolved at the hand-out", "decisions off the dry pass's",
    "a record kept off the workers", "whole-target clear", "stopped at another record", "chunk begin failed", "chunk end failed", "parck test (DM_VK_PARALLEL_WKTEST)"};
static int seg_grow_n(void **pp, uint32_t *cap, uint32_t n, size_t el) {   // room for n elements (a job's copies)
    if (n <= *cap) return 1;
    uint32_t c = *cap ? *cap : 64; while (c < n) c *= 2; void *q = realloc(*pp, (size_t)c * el);
    if (!q) return 0; *pp = q; *cap = c; return 1;
}
static seg_job *seg_job_get(void) {
    if (g_pj == g_njobs) { seg_job **a = realloc(g_jobs, sizeof *a * (size_t)(g_njobs + 1)); if (!a) return NULL; g_jobs = a;
        seg_job *j = calloc(1, sizeof *j); if (!j) return NULL; j->ubl = vk_ubl_new(); j->view = vk_par_view_new(); g_jobs[g_njobs++] = j; }
    return g_jobs[g_pj];   // (taken: g_pj++ once handed out)
}
static void seg_wk_targets(pass_tgt col[8], pass_tgt *dep) {   // a worker's begin_pass: its targets as the hand-out resolved them (no lookup, no mark, no transition here)
    const seg_job *const j = g_wk_job; const uint8_t *const cur = (const uint8_t *)&S + SEG_RT_OFF;
    for (int i = 0; j && i < j->ntg; i++) if (!memcmp(j->tg[i].rt, cur, SEG_RT_LEN)) { memcpy(col, j->tg[i].col, sizeof j->tg[i].col); *dep = j->tg[i].dep; return; }
    if (g_wk_rerun) { pass_resolve_targets(vk_cmd(), col, dep); return; }   // the replay thread's own recording of a failed segment: resolved here (marks later than in stream order)
    memset(col, 0, 8 * sizeof *col); *dep = (pass_tgt){0}; seg_wk_abort(WA_TARGET);
}
// why segment k stays on the replay thread (WR_* bits: the dry pass's, then the list's, then the hand-over's state)
static uint32_t seg_par_why(const seg_desc *s, uint32_t lwhy) {
    uint32_t w = s->wk | lwhy;
    if (!g_seg_dec || !g_seg_flow) w |= 1u << WR_NWK;   // the replay left the dry pass's path in this list (its hand-over dropped)
    return w;
}
// the thread's segment-replay state (exec_ops' TLS), kept aside while it records a job (the replay thread as the helper, at a part's join: inside its own list's walk)
typedef struct { int on, done, hot, inseg, ubl_on, cuts, depth, wk, wk_abort, wk_rerun, tex_wk, tex_refused; uint64_t left, ri, ri_cur, inseg_rec0, cut_next; const uint8_t *at, *flow, *xo_next;
                 uint32_t nflow, fi, nbh, bhi, ndec, di, ncut, ci, xo_op; const uint64_t *bh; seg_dec *dec; const seg_cut *cutv; seg_job *job; uint64_t oser; } seg_tls_t;   // (occ-s1: the stream serial)
static void seg_tls_save(seg_tls_t *t) {
    *t = (seg_tls_t){g_seg_on, g_seg_done, g_seg_hot, g_seg_inseg, g_seg_ubl_on, g_seg_cuts, g_par_depth, g_wk, g_wk_abort, g_wk_rerun, g_vk_tex_wk, g_vk_tex_refused, g_seg_left, g_seg_ri,
                     g_seg_ri_cur, g_seg_inseg_rec0, g_seg_cut_next, g_seg_at, g_seg_flow, g_xo_next, g_seg_nflow, g_seg_fi, g_seg_nbh, g_seg_bhi, g_seg_ndec, g_seg_di, g_seg_ncut, g_seg_ci,
                     g_xo_op, g_seg_bh, g_seg_dec, g_seg_cutv, g_wk_job, g_oser};
}
static void seg_tls_restore(const seg_tls_t *t) {
    g_seg_on = t->on; g_seg_done = t->done; g_seg_hot = t->hot; g_seg_inseg = t->inseg; g_seg_ubl_on = t->ubl_on; g_seg_cuts = t->cuts; g_par_depth = t->depth; g_wk = t->wk; g_wk_abort = t->wk_abort;
    g_wk_rerun = t->wk_rerun; g_vk_tex_wk = t->tex_wk; g_vk_tex_refused = t->tex_refused; g_seg_left = t->left; g_seg_ri = t->ri; g_seg_ri_cur = t->ri_cur; g_seg_inseg_rec0 = t->inseg_rec0;
    g_seg_cut_next = t->cut_next; g_seg_at = t->at; g_seg_flow = t->flow; g_xo_next = t->xo_next; g_seg_nflow = t->nflow; g_seg_fi = t->fi; g_seg_nbh = t->nbh; g_seg_bhi = t->bhi;
    g_seg_ndec = t->ndec; g_seg_di = t->di; g_seg_ncut = t->ncut; g_seg_ci = t->ci; g_xo_op = t->xo_op; g_seg_bh = t->bh; g_seg_dec = t->dec; g_seg_cutv = t->cutv; g_wk_job = t->job; g_oser = t->oser;
}
// the worker's (or the helper's) recording of one segment into its own chunk
static void seg_job_main(void *arg) {
    seg_job *const j = arg; const uint64_t t0 = tnow();
    seg_tls_t sv; seg_tls_save(&sv);
    g_wk = 1; g_wk_abort = 0; g_wk_rerun = j->rerun; g_vk_tex_wk = !j->rerun; g_vk_tex_refused = 0; g_wk_job = j;
    g_oser = j->d.oser0;   // occ-s1: its draws' serials as the dry pass named them
    const VkCommandBuffer cb = vk_par_chunk_begin(j->slot);
    if (!cb) { seg_wk_abort(WA_BEGIN); goto out; }
    vk_par_wk_begin(j->view);
    memcpy(&S, &j->snap.s, SEG_SNAP_BYTES); S.pass_open = 0; S.bound_pipe = VK_NULL_HANDLE; S.rp_pass = 0;   // the stream state; the open pass is this thread's (none)
    { rec_xs_t *const x = &g_rec->X; x->rc_skip = j->snap.rc_skip; x->tf_on = j->snap.tf_on; x->rd_prev = j->snap.rd_prev; x->deq_dir = j->snap.deq_dir; x->lop_kept = j->snap.lop_kept; x->df_kept = j->snap.df_kept;
      x->mdi_cnt = 0; x->mdi_n = 0; x->mdi_stride = 0; x->s64_pend = 0; x->s64_pwrite = 0; x->rd_nwa = 0; x->tf_in_pass = 0; x->tf_verts = 0; x->s64_afterbar = 0; x->s64_fb = 0; x->s65_rdpass = 0;
      x->rcg_on = 0; x->rcg_cb = VK_NULL_HANDLE; }   // ([O]: a fresh start; occ-s3: no GPU condition in a worker's segment, WR_GPUCOND)
    for (int st = 0; st < 6; st++) { g_bind_gen[st]++; g_ubo_gen[st]++; } g_s45_kgen++; g_s45_list++;   // S was copied under this context's memos: none of them may match
    if (!tl_wubl) tl_wubl = vk_ubl_new();
    vk_ubl_copy(tl_wubl, j->ubl); vk_ubl_use(tl_wubl); g_seg_ubl_on = 1;   // its own copy (a failed run leaves the job's list as it was for the replay thread's recording)
    g_seg_flow = j->flow; g_seg_nflow = j->nflow; g_seg_fi = 0; g_seg_bh = j->bh; g_seg_nbh = j->nbh; g_seg_bhi = 0;
    g_seg_dec = j->dec; g_seg_ndec = j->ndec; g_seg_di = 0; g_seg_cutv = NULL; g_seg_ncut = 0; g_seg_ci = 0; g_seg_cut_next = UINT64_MAX; g_seg_cuts = 1; g_seg_inseg = 0;
    g_seg_on = 1; g_seg_left = j->d.rec1 - j->d.rec0; g_seg_done = 0; g_seg_at = NULL; g_seg_ri = j->d.rec0; g_seg_hot = 1;
    vk_img_barrier_all(cb);   // the boundary's dependency (everything before it in queue order)
    seg_resume(&j->d.at, 0);
    { const int done = g_seg_done; const uint8_t *const stop = g_seg_at; g_seg_on = 0; g_seg_done = 0; g_seg_hot = 0; g_seg_left = 0;
      if (g_vk_tex_refused) seg_wk_abort(WA_IMG_NEW + g_vk_tex_refused - 1);
      if (!g_wk_abort && (!done || stop != j->want || g_seg_fi != g_seg_nflow || g_seg_bhi != g_seg_nbh || g_seg_di != g_seg_ndec)) seg_wk_abort(WA_STOP);
      vk_end_pass(); vk_img_barrier_all(vk_cmd()); }   // the segment's pass ends in its chunk; everything after it waits for it
    vk_ubl_use(NULL); g_seg_ubl_on = 0;
    vk_par_wk_end();
    if (vk_par_chunk_end() != VK_SUCCESS) seg_wk_abort(WA_END);
    if (!j->rerun && parck_test()) seg_wk_abort(WA_TEST);   // parck: recorded, then thrown away and recorded again on the replay thread (its own chunk)
out:;
    const int ok = !g_wk_abort; j->why = g_wk_abort; j->ns = tnow() - t0; j->oser_end = g_oser;
    seg_tls_restore(&sv);
    vk_chunk_finish(j->slot, j->ci, ok ? cb : VK_NULL_HANDLE, ok);   // (last: the replay thread may reuse the job once it sees this)
}
// segment k handed to a worker (the replay thread at its first record, part cuts taken, no pass open after this): 0 when it could not be
// parck (DK Bananza 11-22: with DM_VK_PARALLEL=4 the GPU died in the first worker-recorded pass of play, a 2048x2048 shadow pass): DM_VK_PARALLEL_WKTEST=N, for the first N
// presents from the first hand-out, records every worker segment on its worker as usual (all its CPU-side work: uploads, rings, sets), then throws the chunk away and
// records the segment again on the replay thread into a chunk of its own (the existing re-record path). No device loss then and one after: a race between the threads;
// one during: the chunk's own state or the segment's side work. Log lines "parck:"; gpuck names each segment draw (kind draw)
static uint64_t g_parck_p0; static int g_parck_n = -1, g_parck_said;
static int parck_test(void) { const uint64_t p0 = __atomic_load_n(&g_parck_p0, __ATOMIC_RELAXED); return g_parck_n > 0 && p0 && g_tex_frame < p0 + (uint64_t)g_parck_n; }
static void parck_note(void) {
    if (g_parck_n < 0) { const char *e = getenv("DM_VK_PARALLEL_WKTEST"); g_parck_n = e && *e ? atoi(e) : 0; }
    if (!g_parck_p0) { __atomic_store_n(&g_parck_p0, g_tex_frame ? g_tex_frame : 1, __ATOMIC_RELAXED);
        if (g_parck_n > 0) VKLOG("parck: first worker segment at present %llu: for %d presents every worker segment is recorded again on the replay thread (DM_VK_PARALLEL_WKTEST)", (unsigned long long)g_tex_frame, g_parck_n); }
    else if (g_parck_n > 0 && !g_parck_said && !parck_test()) { g_parck_said = 1; VKLOG("parck: present %llu: worker chunks are used from here on", (unsigned long long)g_tex_frame); }
}
static int seg_par_handout(uint32_t k) {
    parck_note();
    const seg_desc *const s = &g_sd.seg[k]; const seg_snap *const e = &g_sd.esnap[k];
    seg_job *const j = seg_job_get(); if (!j) return 0;
    if (!seg_grow_n((void **)&j->flow, &j->capflow, s->flow1 - s->flow0, 1) || !seg_grow_n((void **)&j->dec, &j->capdec, s->dec1 - s->dec0, sizeof *j->dec) ||
        !seg_grow_n((void **)&j->bh, &j->capbh, s->bh1 - s->bh0, sizeof *j->bh)) return 0;
    const uint64_t t0 = tnow();
    vk_tex_par_hold();   // image objects dropped from here to the part's join are kept (the resolution below may evict one an earlier target state of this segment resolved to)
    if (g_occ2 >= 2 && s->oser0 + s->ndraw > g_o3p.skip) g_o3p.skip = s->oser0 + s->ndraw;   // occ-s3fast: the plan's opportunities inside the segment never come on this thread
    vk_end_pass();   // no pass spans two command buffers
    if (g_occ2 >= 2) vk_occ2_opp();   // occ-s3fast: the hand-out is an opportunity (no pass open; before the segment in stream order)
    if (g_oser != s->oser0 && !__atomic_fetch_add(&g_occ2_ser_off, 1, __ATOMIC_RELAXED)) VKLOG("[RED] occ-s1: the replay thread's stream serial %llu at a hand-out is not the dry pass's %llu: the dry pass's taken", (unsigned long long)g_oser, (unsigned long long)s->oser0);
    vk_occ2_handout(s->oser0, s->ndraw); g_oser = s->oser0 + s->ndraw;   // occ-s1: its draws' ring chunks reset here, in stream order; the replay thread goes on after them
    // the stream-order work of its records that the worker skips, here, before its chunk in queue order
    g_occ_draws += s->ndraw; g_cb_draws += (int)s->ncb;   // (exec_ops' and prepare_draw's counts of its draws)
    { static rec_state_t ts; memcpy((uint8_t *)&ts + SEG_RT_OFF, (const uint8_t *)&S + SEG_RT_OFF, SEG_RT_LEN); j->ntg = 0;
      for (uint32_t i = s->tg0; i < s->tg1; i++) { const seg_tg *const g = &g_sd.tg[i];
          if (g->r) { rt_assign_on(&ts, g->r); oot_bilerp_rt(g->r); }   // (its SET_RTs: none the stand-in may copy at, seg_dry_list; a repeat of the same targets is idempotent there)
          if (!g->used || j->ntg >= SEG_TG_MAX) continue;
          seg_wtg *const w = &j->tg[j->ntg++]; memset(w->col, 0, sizeof w->col);
          memcpy(w->rt, (const uint8_t *)&ts + SEG_RT_OFF, SEG_RT_LEN);
          pass_resolve_targets_on(&ts, vk_cmd(), w->col, &w->dep);   // R6: made, uploaded or filled at first use, marked, first transition: in stream order, into this R chunk
          for (int c = 0; c < 9; c++) { vkimg_t *const im = c < 8 ? (c < ts.rt_n ? w->col[c].im : NULL) : w->dep.im; if (im) { wb_use(im, WB_ATTACHMENT); im->scaled_valid = 0; } } } }   // pass_commit's marks (no scale: seg_refuse)
    { void vk_tbuf_fresh(VkCommandBuffer cb); vk_tbuf_fresh(vk_cmd()); }   // r150: copies the GPU writes before the segment made stale, before its passes
    { void vk_rpt_flush(void); vk_rpt_flush(); }   // replay2: the reports recorded before it in memory (the worker's uniform reads stand for GPU reads after them)
    // what it starts from
    memcpy(&j->snap.s, &S, SEG_SNAP_BYTES); j->snap.rc_skip = g_rc_skip; j->snap.tf_on = g_tf_on; j->snap.rd_prev = g_rd_prev; j->snap.deq_dir = g_deq_dir; j->snap.lop_kept = g_rec->X.lop_kept; j->snap.df_kept = g_rec->X.df_kept;
    j->d = *s; j->want = s->endp.pos[s->endp.d];
    vk_ubl_snapshot(j->ubl);   // R5: the main list as at its start; its updates then applied to the main list in stream order (the superseded indices logged), as SERIALSEG's seg_ub_open
    for (uint32_t i = s->dec0; i < s->dec1 && i < g_sd.ndec; i++) if (g_sd.dec[i].kind == SD_UB) { int landed;
        g_sd.dec[i].sup = vk_sub_update_apply(&g_sd.dec[i].u, &landed); g_sst.ub_seg++;
        if (landed && !g_sst.ub_land++) VKLOG("[RED] S1e: a worker segment's uniform update landed the list (the dry pass's bound said it cannot)"); }
    j->nflow = s->flow1 - s->flow0; if (j->nflow) memcpy(j->flow, g_sd.flow + s->flow0, j->nflow);
    j->ndec = s->dec1 - s->dec0; if (j->ndec) memcpy(j->dec, g_sd.dec + s->dec0, sizeof *j->dec * (size_t)j->ndec);
    j->nbh = s->bh1 - s->bh0; if (j->nbh) memcpy(j->bh, g_sd.bh + s->bh0, sizeof *j->bh * (size_t)j->nbh);
    j->slot = vk_sub_slot(); j->rerun = 0; j->why = 0; j->ns = 0; j->worker = vk_par_pick();
    j->ci = vk_par_handout(j->worker, j->view);   // its place among the posts, the R chunk ended, its W chunk, a new R chunk
    g_pj++;
    vk_par_run(j->worker, seg_job_main, j);
    // the replay thread goes on after it: the state its records leave (the dry pass's end snapshot), the memos made stale
    memcpy(&S, &e->s, SEG_SNAP_BYTES); g_rc_skip = e->rc_skip; g_tf_on = e->tf_on; g_rd_prev = e->rd_prev; g_deq_dir = e->deq_dir; g_rec->X.lop_kept = e->lop_kept; g_rec->X.df_kept = e->df_kept;
    for (int st = 0; st < 6; st++) { g_bind_gen[st]++; g_ubo_gen[st]++; } g_s45_kgen++;
    g_pst.handed++; g_pst.hdraws += s->ndraw; g_pst.handout_ns += tnow() - t0;
    return 1;
}
// gfx_vk_core.c vk_cmd_split / mtl_sub_commit: the open part's worker segments recorded before it is submitted (no pass of the replay thread open)
void vk_seg_part_join(void) {
    if (!g_pj) return;
    const uint64_t t0 = tnow();
    while (vk_par_help_one()) g_pst.helped++;   // queued jobs recorded here (the helper context) instead of waiting for them
    (void)vk_chunk_wait_all();
    g_pst.wait_ns += tnow() - t0;
    for (int i = 0; i < g_pj; i++) { seg_job *const j = g_jobs[i]; vk_chunk c; vk_chunk_get(j->ci, &c);
        g_pst.wk_ns += j->ns;
        if (c.done != 1) {   // recorded again here, in the same chunk's place, by the replay thread (images may be made; targets resolved here if the hand-out did not)
            g_pst.rwhy[j->why < WA_N ? j->why : 0]++; g_pst.rerun++;
            if (j->why == WA_BEGIN || j->why == WA_END) { g_pst.red++; if (g_pst.red <= 8) VKLOG("[RED] S1e: worker %d could not record its segment (records %llu..%llu): %s; recorded on the replay thread", j->worker,
                (unsigned long long)j->d.rec0, (unsigned long long)j->d.rec1, g_wa_names[j->why]); }
            j->rerun = 1; vk_par_help_run(seg_job_main, j);
            vk_chunk_get(j->ci, &c);
            if (c.done != 1) { g_pst.lost++; g_pst.red++; if (g_pst.lost <= 8) VKLOG("[RED] S1e: a segment (records %llu..%llu, %u draws) could not be recorded on the replay thread either (%s): its draws are left out of this submission",
                (unsigned long long)j->d.rec0, (unsigned long long)j->d.rec1, j->d.ndraw, g_wa_names[j->why < WA_N ? j->why : 0]); } }
        if (c.done == 1 && j->oser_end != j->d.oser0 + j->d.ndraw && !__atomic_fetch_add(&g_occ2_ser_off, 1, __ATOMIC_RELAXED))   // occ-s1: the worker counted the segment's draws as the dry pass did
            VKLOG("[RED] occ-s1: a worker segment (records %llu..%llu) ended at stream serial %llu, the dry pass's %llu", (unsigned long long)j->d.rec0, (unsigned long long)j->d.rec1,
                  (unsigned long long)j->oser_end, (unsigned long long)(j->d.oser0 + j->d.ndraw));
        vk_occ2_job_join(j->view, c.done == 1);   // occ-s2: its begun queries (a chunk left out: its slots get empty queries at the next resolve)
        vk_par_merge_posts(j->view); }
    (void)vk_chunk_wait_all();   // (the helper's counters folded again after the recordings here)
    g_pst.parts++; g_pst.chunks += 2 * (uint64_t)g_pj + 1;
    g_pj = 0;
    vk_tex_par_release();
    vk_occ2_join_end();   // occ-s2: the resolves deferred over the part's segments, now (no pass is open here)
}
// SERIALSEG's walk (seg_pieces) with the hand-out: segments a worker may record go to the workers, the others are recorded here as SERIALSEG does
static void seg_par_pieces(const uint8_t *p) {
    seg_pos at0; memset(&at0, 0, sizeof at0); at0.pos[0] = p;
    const seg_pos *from = &at0; uint64_t r = 0; uint32_t fl = 0, bh = 0, dc = 0;
    const uint32_t lwhy = seg_worker_refuse() ? WR_DIAG : 0;
    for (uint32_t k = 0; k < g_sd.nseg; k++) { const seg_desc *s = &g_sd.seg[k];
        if (s->rec0 > r && !seg_piece(from, r, s->rec0, fl, bh, dc, s->at.pos[s->at.d])) return;   // the serial records before it
        g_seg_ri_cur = s->rec0; while (g_seg_cutv && g_seg_cut_next == s->rec0) seg_cut_take();   // a part cut where it starts comes before it
        const uint32_t why = seg_par_why(s, lwhy);
        if (!why && seg_par_handout(k)) { from = &s->endp; r = s->rec1; fl = s->flow1; bh = s->bh1; dc = s->dec1; g_seg_fi = fl; g_seg_bhi = bh; g_seg_di = dc; continue; }   // (its decisions are the worker's)
        g_pst.kept++; g_pst.kdraws += s->ndraw; for (int b = 0; b < WR_NWK + 1; b++) if (why >> b & 1) g_pst.kwhy[b]++; if (!why) g_pst.kwhy[WR_NWK + 1]++;   // (no job memory: the last slot)
        seg_ub_open(s); g_seg_inseg = 1; g_seg_inseg_rec0 = s->rec0;   // recorded here: its own update list, as SERIALSEG
        const int ok = seg_piece(&s->at, s->rec0, s->rec1, s->flow0, s->bh0, s->dec0, s->endp.pos[s->endp.d]);
        g_seg_inseg = 0; seg_ub_close();
        if (!ok) return;
        g_sst.segs_run++; from = &s->endp; r = s->rec1; fl = s->flow1; bh = s->bh1; dc = s->dec1; }
    (void)seg_piece(from, r, 0, fl, bh, dc, NULL);   // the rest of the list
}
static void seg_par_log(double f) {   // every 300 presents, with the S1b / S1d lines
    char kw[900]; size_t k = 0; kw[0] = 0;
    for (int b = 0; b < WR_NWK + 2; b++) if (g_pst.kwhy[b]) k += (size_t)snprintf(kw + k, sizeof kw - k, "%s %s %.1f", k ? "," : "", b < WR_NWK ? g_wr_names[b] : b == WR_NWK ? "the list's hand-over dropped" : "no memory for a job", g_pst.kwhy[b] / f);
    char rw[600]; size_t q = 0; rw[0] = 0;
    for (int b = 1; b < WA_N; b++) if (g_pst.rwhy[b]) q += (size_t)snprintf(rw + q, sizeof rw - q, "%s %s %.2f", q ? "," : "", g_wa_names[b], g_pst.rwhy[b] / f);
    VKLOG("S1e: parallel recording a frame (%d workers): segments handed to workers %.1f (%.0f draws), kept on the replay thread %.1f (%.0f draws) by:%s; worker record %.2f ms (%.3f ms a segment), "
          "hand-out %.2f ms, the replay thread waited %.2f ms at part ends (ran %.1f queued jobs itself); parts with worker chunks %.1f, chunks per such part %.1f; recorded again on the replay "
          "thread %.2f%s%s; since start: worker failures %llu, segments lost %llu%s",
          vk_par_workers(), g_pst.handed / f, g_pst.hdraws / f, g_pst.kept / f, g_pst.kdraws / f, k ? kw : " none", g_pst.wk_ns / 1e6 / f, g_pst.handed ? g_pst.wk_ns / 1e6 / (double)g_pst.handed : 0.0,
          g_pst.handout_ns / 1e6 / f, g_pst.wait_ns / 1e6 / f, g_pst.helped / f, g_pst.parts / f, g_pst.parts ? (double)g_pst.chunks / (double)g_pst.parts : 0.0, g_pst.rerun / f, q ? " by:" : "", q ? rw : "",
          (unsigned long long)g_pst.red, (unsigned long long)g_pst.lost, g_pst.red ? " [RED]" : "");
    const uint64_t red = g_pst.red, lost = g_pst.lost; memset(&g_pst, 0, sizeof g_pst); g_pst.red = red; g_pst.lost = lost;
}
static void seg_top(char *o, size_t n, const uint64_t *c, double frames) {   // the six largest counts, by name
    size_t k = 0; uint64_t taken[SW_N / 64 + 1] = {0}; o[0] = 0;
    for (int t = 0; t < 6; t++) { int b = -1; for (int i = 0; i < SW_N; i++) if (c[i] && !(taken[i / 64] >> (i % 64) & 1) && (b < 0 || c[i] > c[b])) b = i;
        if (b < 0) break; taken[b / 64] |= 1ull << (b % 64);
        k += (size_t)snprintf(o + k, n - k, "%s %s %.1f", t ? "," : "", seg_why_name(b), c[b] / frames); if (k >= n - 64) break; }
    if (!k) snprintf(o, n, " none");
}
static void seg_log(int mode) {   // every 300 presents
    extern uint64_t g_tex_frame;
    if (!g_sst.f0) { g_sst.f0 = g_tex_frame ? g_tex_frame : 1; return; }
    if (g_tex_frame < g_sst.f0 + 300) return;
    const double f = (double)(g_tex_frame - g_sst.f0); g_sst.f0 = g_tex_frame;
    uint64_t refl = 0; for (int i = 1; i < 6; i++) refl += g_sst.refl[i];
    char dh[160]; size_t k = 0; for (int i = 0; i <= SEG_DMAX; i++) k += (size_t)snprintf(dh + k, sizeof dh - k, "%s%.1f", i ? " " : "", g_sst.depth[i] / f);
    VKLOG("S1b: dry pass per frame (%.0f presents): lists %.1f (refused %.1f: not Eden %.1f, NVN2 %.1f, resolution scale %.1f, frame-gen capture %.1f, diagnostics %.1f); "
          "draws %.0f, in segments %.0f (%.0f%%); segments %.1f, draws per segment mean %.1f max %llu; segment starts by call depth 0..%d: %s; CALL/JUMP decisions %.0f; dry pass %.3f ms",
          f, g_sst.lists / f, refl / f, g_sst.refl[1] / f, g_sst.refl[2] / f, g_sst.refl[3] / f, g_sst.refl[4] / f, g_sst.refl[5] / f,
          g_sst.draws / f, g_sst.seg_draws / f, g_sst.draws ? 100.0 * g_sst.seg_draws / g_sst.draws : 0.0, g_sst.segs / f, g_sst.segs ? (double)g_sst.seg_draws / g_sst.segs : 0.0,
          (unsigned long long)g_sst.seg_max, SEG_DMAX, dh, g_sst.flow / f, g_sst.dry_ns / 1e6 / f);
    char e[700], r[700]; seg_top(e, sizeof e, g_sst.ends, f); seg_top(r, sizeof r, g_sst.refused, f);
    VKLOG("S1b: segments ended a frame by:%s; runs refused (under %d draws) %.1f a frame holding %.0f draws, by:%s", e, SEG_MIN, g_sst.ref_runs / f, g_sst.ref_draws / f, r);
    VKLOG("S1d: handed over a frame: counter reports %.0f (timestamps with a slot named %.1f), counter resets %.1f, fence syncs %.1f, render conditions %.1f (left to the replay %.1f), "
          "uniform updates %.0f (in segments %.0f); part cuts %.1f (pass splits %.1f, over the cap %.1f, taken %.1f); samples-passed reports inside segments %.1f (left to the replay: occlusion, R4); "
          "segment starts refused (state not known) %.1f",
          g_sst.rep / f, g_sst.ts_named / f, g_sst.reset / f, g_sst.fence / f, g_sst.cond / f, g_sst.cond_stop / f, g_sst.ub / f, g_sst.ub_seg / f, g_sst.cuts / f, g_sst.psplits / f,
          g_sst.capped / f, g_sst.cuts_done / f, g_sst.sp_in_seg / f, g_sst.open_unk / f);
    VKLOG("S1d2: render conditions a frame: %.1f; decided by the dry pass: enable/disable %.1f, outcome fixed without memory (a report of this submission or never measured) %.1f, "
          "from memory or the list's fence with no GPU write before %.1f; left to the replay %.1f: memory a GPU write may change %.1f, a report in flight %.1f; "
          "short runs after a part cut (not merged across it) %.1f",
          g_sst.cond / f, g_sst.cond_const / f, g_sst.cond_fixed / f, g_sst.cond_mem / f, g_sst.cond_stop / f, g_sst.cond_stop_mem / f, g_sst.cond_stop_fence / f, g_sst.tail_cut / f);
    if (mode == 3) seg_par_log(f);   // S1e (parallel recording)
    if (mode == 2) VKLOG("S1b: serialseg: %.1f segments resumed a frame (%.1f pieces); since start: stopped at another record %llu, list ended early %llu, CALL/JUMP decisions off the dry pass's %llu, stream state at a segment's start differs %llu, deferred-handle reads off the dry pass's %llu%s",
                         g_sst.segs_run / f, g_sst.pieces / f, (unsigned long long)g_sst.stop_bad, (unsigned long long)g_sst.early, (unsigned long long)g_sst.flow_bad + g_seg_flow_bad,
                         (unsigned long long)g_sst.state_bad, (unsigned long long)g_seg_bh_bad, g_sst.stop_bad + g_sst.early + g_sst.flow_bad + g_seg_flow_bad + g_sst.state_bad + g_seg_bh_bad ? " [RED]" : "");
    { const uint64_t bad = g_sst.dec_bad + vk_ts_slot_off() + g_sst.cond_off + g_sst.ubl_off + g_sst.ublog_off + g_sst.ub_land + g_ubl_seg_lands + g_sst.cut_bad;
      uint64_t sb = 0; for (int i = 0; i < 7; i++) sb += g_sst.snap_bad[i];
      VKLOG("S1d: %s since start: decisions off the dry pass's %llu, timestamp slots off %llu, render conditions evaluated differently at replay %llu, segment update lists off the main list %llu, "
            "segment update log entries off %llu, segment updates that landed %llu, segment lists landed %llu, part cuts not taken or inside a segment %llu; "
            "snapshot fields differing at a segment's start: stream state %llu, render condition %llu, transform feedback %llu, rasterizer-discard run %llu, depth direction %llu, logic op %llu, depth format %llu%s",
            mode == 2 ? "serialseg" : "dry", (unsigned long long)g_sst.dec_bad, (unsigned long long)vk_ts_slot_off(), (unsigned long long)g_sst.cond_off, (unsigned long long)g_sst.ubl_off,
            (unsigned long long)g_sst.ublog_off, (unsigned long long)g_sst.ub_land, (unsigned long long)g_ubl_seg_lands, (unsigned long long)g_sst.cut_bad,
            (unsigned long long)g_sst.snap_bad[0], (unsigned long long)g_sst.snap_bad[1], (unsigned long long)g_sst.snap_bad[2], (unsigned long long)g_sst.snap_bad[3],
            (unsigned long long)g_sst.snap_bad[4], (unsigned long long)g_sst.snap_bad[5], (unsigned long long)g_sst.snap_bad[6], bad + sb ? " [RED]" : ""); }
    const uint64_t keep[] = {g_sst.stop_bad, g_sst.early, g_sst.flow_bad, g_sst.state_bad, g_sst.oom, g_sst.dec_bad, g_sst.cond_off, g_sst.ubl_off, g_sst.ublog_off, g_sst.ub_land, g_sst.cut_bad};
    uint64_t sbk[7]; memcpy(sbk, g_sst.snap_bad, sizeof sbk);
    memset(&g_sst, 0, sizeof g_sst); g_sst.f0 = g_tex_frame;
    g_sst.stop_bad = keep[0]; g_sst.early = keep[1]; g_sst.flow_bad = keep[2]; g_sst.state_bad = keep[3]; g_sst.oom = keep[4];
    g_sst.dec_bad = keep[5]; g_sst.cond_off = keep[6]; g_sst.ubl_off = keep[7]; g_sst.ublog_off = keep[8]; g_sst.ub_land = keep[9]; g_sst.cut_bad = keep[10]; memcpy(g_sst.snap_bad, sbk, sizeof sbk);
}
static void seg_list(const uint8_t *p) {   // mtl_exec, a top-level list, DM_VK_PARALLEL_DRY or DM_VK_PARALLEL_SERIALSEG on
    const int mode = seg_mode(), why = seg_refuse();
    g_sst.lists++;
    if (why) { g_sst.refl[why]++; exec_ops(p); seg_log(mode); return; }   // as before: nothing walked, nothing asked twice
    g_sd.mode = mode;   // S1e (parallel recording): which snapshots the dry pass keeps (seg_open, seg_end_snap)
    const uint64_t t0 = tnow(); seg_dry(p); g_sst.dry_ns += tnow() - t0;
    if (g_sd.ok) { g_seg_flow = g_sd.flow; g_seg_nflow = g_sd.nflow; g_seg_fi = 0; g_seg_bh = g_sd.bh; g_seg_nbh = g_sd.nbh; g_seg_bhi = 0;   // (an allocation failure: hle_nvn.c is asked again, as without the dry pass; S1c: and the deferred handles read again)
        g_seg_dec = g_sd.dec; g_seg_ndec = g_sd.ndec; g_seg_di = 0; g_seg_cutv = g_sd.cut; g_seg_ncut = g_sd.ncut; g_seg_ci = 0;   // S1d: the decisions and part cuts
        g_seg_cut_next = g_sd.ncut ? g_sd.cut[0].ri : UINT64_MAX; g_seg_cuts = 1; g_seg_ri = 0; g_seg_hot = 1; g_seg_chk = mode == 2; }
    // (S1d: with an allocation failure the dry pass's CPU work done so far stays done: counter census, report table, fence syncs, scratch copies; the replay does it again)
    if (mode == 2 && g_sd.ok && g_sd.nseg) seg_pieces(p);
    else if (mode == 3 && g_sd.ok && g_sd.nseg) seg_par_pieces(p);   // S1e (parallel recording): segments to the workers
    else exec_ops(p);
    if (g_seg_flow && g_seg_fi != g_seg_nflow) { if (!g_sst.flow_bad++) VKLOG("[RED] S1b: the replay took %u of the dry pass's %u CALL/JUMP decisions", g_seg_fi, g_seg_nflow); }
    if (g_seg_bh && g_seg_bhi != g_seg_nbh) { if (!g_seg_bh_bad++) VKLOG("[RED] S1c: the replay took %u of the dry pass's %u deferred-handle words", g_seg_bhi, g_seg_nbh); }   // S1c (parallel recording)
    if (g_seg_dec && g_seg_di != g_seg_ndec) { if (!g_sst.dec_bad++) VKLOG("[RED] S1d: the replay took %u of the dry pass's %u decisions", g_seg_di, g_seg_ndec); }   // S1d
    if (g_seg_cutv && g_seg_ci != g_seg_ncut) { if (!g_sst.cut_bad++) VKLOG("[RED] S1d: the replay took %u of the dry pass's %u part cuts", g_seg_ci, g_seg_ncut); }   // S1d
    if (g_oser != g_sd.oser) { if (!__atomic_fetch_add(&g_occ2_ser_off, 1, __ATOMIC_RELAXED)) VKLOG("[RED] occ-s1: the list's replay ended at stream serial %llu, the dry pass's at %llu: the dry pass's taken", (unsigned long long)g_oser, (unsigned long long)g_sd.oser);
        g_oser = g_sd.oser; }   // occ-s1: the dry pass owns the count
    g_seg_flow = NULL; g_seg_nflow = g_seg_fi = 0;
    g_seg_bh = NULL; g_seg_nbh = g_seg_bhi = 0;
    seg_handover_drop(); g_seg_ndec = g_seg_di = g_seg_ncut = g_seg_ci = 0; g_seg_hot = 0; g_seg_chk = 0;   // S1d
    seg_log(mode);
}
