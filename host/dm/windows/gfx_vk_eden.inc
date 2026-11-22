// NXN1X lane B (NXN1X docs/lane_b/SHADER_EDEN_B.md B4b): draws and dispatches with Eden's pipelines.
// Included by gfx_vk_exec.c before dispatch(): it reads the NVN state shadow S and the replay's helpers (c2_handle, ubo_word, drv_segment,
// drv_shared, cbuf_desc, tex_desc_for, set_dynamic, the bind cache g_bc). In Eden mode (vk_eden_mode, the default) a draw gets the Eden
// GraphicsPipeline the Maxwell path would make for the same shaders and state (nvn_eden_graphics_pipeline): its VkPipeline, its set 0 filled in
// Eden's order (per stage: uniform buffers, storage buffers, texel buffers, storage texel buffers, textures, images; pipeline_helper.h
// DescriptorLayoutBuilder) through its update template, its push constants and its dynamic state. The replay's own conventions stay where the
// picture depends on them: the viewport flips (set_dynamic), the winding that follows from them, Bayonetta 3's depth EQUAL relaxation (r57).
// Not carried over from the bundle path: the shader variants (integer views, gl_FragDepth, alpha test: Eden's SPIR-V does what the Maxwell
// path does) and the diagnostic paths that read the bundle's reflection (r131-r153 classes, the set memo, DM_VK_X_TEXFS).

void vk_gpu_wrote(uint64_t a, uint64_t n); void vk_sub_wrote(uint64_t a, uint64_t n);
void vk_dsl_register(VkDescriptorSetLayout l, const VkDescriptorSetLayoutBinding *b, uint32_t n);
int mtl_wait(uint64_t fence, uint64_t timeout_ns);

enum { EDEN_MAXE = 4096 };   // payload entries of one pipeline's set 0 (texture arrays counted element by element). maxe4k (FE Fortune's Weave 11-22: a 1035-entry pipeline had its draws skipped at 1024)
typedef union { VkDescriptorImageInfo image; VkDescriptorBufferInfo buffer; VkBufferView texel; struct { VkDeviceAddress address; VkDeviceSize range; VkFormat format; } addr; } eden_entry;
_Static_assert(sizeof(eden_entry) == 24, "Vulkan::DescriptorUpdateEntry is 24 bytes");

// ---- what a translation asks (nvn_eden_query) and where binds read their handles and addresses ----
// The driver's c0 storage buffer table of a stage (drv_segment; tessellation control 0x210 and evaluation 0x310 in Eden mode)
static uint32_t eden_c0_word(int st, uint32_t off) {
    int base = storage_table_offset(st);
    if (st == 5 && dm_sdk_ounce()) base = -1;   // NVN2 compute: c[0] is laid out per shader (drv_segment nvn2_cs_c0), not NVN's table at 0x310
    if (base >= 0 && off >= (uint32_t)base && off < (uint32_t)base + 256) {   // the table, without building the segment
        uint32_t r = (off - (uint32_t)base) / 16, w = (off - (uint32_t)base) % 16; uint64_t a = S.ssbo[st][r].addr;
        return w == 0 ? (uint32_t)a : w == 4 ? (uint32_t)(a >> 32) : w == 8 ? (uint32_t)S.ssbo[st][r].size : 0; }
    if (off + 4 > 0x1000) return 0;
    uint8_t d[0x1000]; drv_segment(st, 0, d); uint32_t v; memcpy(&v, d + off, 4); return v;
}
static uint32_t eden_cbuf_word(int st, uint32_t bank, uint32_t off) {
    off &= ~3u;
    if (bank == 0) return eden_c0_word(st, off);
    if (bank == 1) { const vkshader_t *s = S.sh[st]; uint32_t v = 0; if (s && s->c1 && off + 4 <= s->c1_len) memcpy(&v, s->c1 + off, 4); return v; }
    if (bank == 2) { uint64_t h = c2_handle(st, (int)(off & ~7u)); return (uint32_t)(off & 4 ? h >> 32 : h); }
    if (bank == DMSH_PUSH_BANK && dm_sdk_ounce()) { uint32_t v = 0; if (off + 4 <= DMSH_PUSH_SIZE) memcpy(&v, S.push[st == 5] + off, 4); return v; }   // NVN2 graphics push constants (c[24], AMapC0)
    return ubo_word(st, (int)bank - 3, (int)off);   // uniform buffer slot bank - 3
}
static uint32_t eden_q_cbuf(void *u, int st, uint32_t bank, uint32_t off) { (void)u; return st >= 0 && st < 6 ? eden_cbuf_word(st, bank, off) : 0; }
static int eden_q_texture(void *u, uint32_t h, int *target, int *vkf) {
    (void)u; const view_t *v = NULL; const tex_t *t = NULL; int tid = (int)(h & 0xfffff);
    if (!tid || !vk_tex_by_id(tid, &v, &t) || !t) return 0;
    const vkfmt_t *f = vk_fmt(v && v->set_format ? v->format : t->d.format);
    *target = (int)t->d.target; *vkf = f ? (int)f->vk : 0; return 1;
}
static const nvn_eden_query g_eden_q = {NULL, eden_q_cbuf, eden_q_texture};
// A texture handle as Eden's read_handle takes it (vk_graphics_pipeline.cpp): the word at c[cbuf][offset + (index << size_shift)], shifted and
// ORed with the secondary's. c2 holds the host-numbered handles (64-bit, bind_handles); a uniform buffer the game's, moved into the host
// numbering as ubo_handle does (r90)
static uint64_t eden_handle_part(int st, uint32_t bank, uint32_t off, int *raw) {
    if (bank == 2) return c2_handle(st, (int)(off & ~7u)) >> (off & 4 ? 32 : 0);
    if (bank >= 3) *raw = 1;
    return eden_cbuf_word(st, bank, off);
}
static uint32_t eden_pool_word(uint64_t addr) {
    uint32_t word = 0;
    if (addr <= UINTPTR_MAX - sizeof word && vk_pool_of(addr, sizeof word))
        memcpy(&word, (const void*)(uintptr_t)addr, sizeof word);
    return word;
}
static uint64_t eden_handle(int st, const nvn_eden_bind *b, uint32_t k) {
    uint32_t io = k << b->size_shift; int raw = 0;
    uint64_t h;
    if (b->handle_info) {
        h = nvn_eden_eval_handle(b, k, &g_eden_q, V.props.limits.minStorageBufferOffsetAlignment, eden_pool_word);
        raw = 1;
    } else {
        if (b->handle_is_pointer) return 0;
        h = eden_handle_part(st, b->cbuf, b->offset + io, &raw);
        if (b->has_secondary) h = h << b->shift_left | eden_handle_part(st, b->secondary_cbuf, b->secondary_offset + io, &raw) << b->secondary_shift_left;
    }
    if (raw && ubo_pool_on()) { uint64_t t = h & 0xfffff, s = h >> 20 & 0xfff; if (t) t += (uint64_t)S.tex_base; if (s) s += (uint64_t)S.samp_base; h = s << 20 | t; }
    return h;
}

// ---- vtxpull (design\VERTEX_PULLING_DESIGN.md phase B): which draws read their vertex streams through the fetch table ----
// Cause: on N1X the fixed-function vertex fetch from imported (VK_EXT_external_memory_host) memory is ~100-250x slower; shader loads from the same pages are
// full speed (membench4: the decoder bit-exact against the fixed-function fetch but NaN payloads, the grass draw 13.8 us pulled vs 12.1 mirrored vs 202.8 imported).
// A draw is pulled only when every stream its vertex shader reads lies whole in one zero-copy pool chunk (the shader checks each load against the stream's size)
// and the open submission's GPU work did not write it before the draw (vk_sub_gw_overlaps: no barrier to the vertex stage is recorded for that, the draw keeps
// the mirror path as before). Everything else keeps the vertex buffers (vbmir). Indices stay in the mirrors. The decision is part of Eden's key
// (nvn_eden_draw_state::vertex_pull), so pulling and fixed-function variants are separate pipelines; the switch changes only at a present
enum { VPF_DRAW, VPF_VS, VPF_ADDR, VPF_POOL, VPF_GPUW, VPF_PIPE, VPF_RING, VPF_N };
static const char *const g_vpf_name[VPF_N] = {"indirect / multi-draw / quads / mesh", "vertex shader's inputs not known yet (its first pipeline)", "stream without address or size",
                                               "stream not whole in one zero-copy pool chunk", "stream written by the GPU earlier in the submission",
                                               "pipeline does not pull (attribute format, indexed attribute load, device)", "no fetch table memory"};
// S1c (parallel recording): the switch g_vpull_on is the replay thread's (vpull_frame, at a present) and read by every recording thread (atomic loads: a draw takes one value,
// which goes into its pipeline key); the decision g_vpull_want / g_vpull_why / g_vpull_tab is per draw, so per thread; the per-present counters g_vps_pulled / g_vps_fb are
// struct rec_eden's (folded into the replay thread's, vk_rec_ctx_fold). vpull_frame and vpull_decide follow struct rec_eden
static int g_vpull_on; static uint64_t g_vpull_fr = ~0ull, g_vps_logp;
static _Thread_local int g_vpull_want, g_vpull_why;   // the last decision (vpull_decide) and, when 0 with the switch on, why
typedef struct { uint64_t addr; uint32_t stride, size; } vpull_ent;
static _Thread_local vpull_ent g_vpull_tab[16];   // the decided draw's fetch table (stream s at 16 s, as EmitPulledAttribute reads it)

// ---- the draw state in NVN values (nvn_eden_draw_state) ----
static void eden_fill_state(nvn_eden_draw_state *d, int prim) {
    memset(d, 0, sizeof *d);
    d->primitive = (uint32_t)prim; d->patch_size = S.patch_n > 0 ? (uint32_t)S.patch_n : 0;   // 0: never set (the bridge gives the Maxwell path's 32)
    d->polygon_mode = S.poly.mode; d->cull_face = S.poly.cull; d->front_face = S.poly.front == NVN_FRONT_FACE_CCW; d->depth_clamp = (uint32_t)S.depth_clamp;
    d->polygon_offset = S.poly.offset_en; d->rasterizer_discard = (uint32_t)S.rast_discard; d->primitive_restart = (uint32_t)S.prim_restart;
    d->depth_test = S.ds.dt; d->depth_write = S.ds.dw; d->depth_func = S.ds.df;
    d->depth_bounds_test = (uint32_t)S.db_on; d->depth_bounds[0] = S.db_lo; d->depth_bounds[1] = S.db_hi;
    d->stencil_test = S.ds.st;
    d->stencil_func[0] = S.ds.sf_f; d->stencil_fail[0] = S.ds.fail_f; d->stencil_zfail[0] = S.ds.zfail_f; d->stencil_zpass[0] = S.ds.zpass_f;
    d->stencil_func[1] = S.ds.sf_b; d->stencil_fail[1] = S.ds.fail_b; d->stencil_zfail[1] = S.ds.zfail_b; d->stencil_zpass[1] = S.ds.zpass_b;
    d->alpha_func = S.color.alpha_func; d->alpha_ref = S.alpha_ref;
    d->samples = S.samples ? S.samples : 1; d->alpha_to_coverage = S.ms.a2c;
    // B7: COPY is logic op off; the Maxwell register then keeps the last op the game turned on (CLEAR at first), which FixedPipelineState reads
    // S1c (parallel recording): the kept op is stream state (rec_xs_t, was a static here): a segment's snapshot carries it
    if (S.color.logic_op != 3) g_rec->X.lop_kept = (uint32_t)S.color.logic_op;
    d->logic_op_enable = S.color.logic_op != 3; d->logic_op = g_rec->X.lop_kept;
    d->blend_enable = S.color.blend_en;
    for (int i = 0; i < 8; i++) {
        d->blend[i].func_src = (uint32_t)S.blend[i].src_rgb; d->blend[i].func_dst = (uint32_t)S.blend[i].dst_rgb; d->blend[i].func_src_a = (uint32_t)S.blend[i].src_a;
        d->blend[i].func_dst_a = (uint32_t)S.blend[i].dst_a; d->blend[i].eq = (uint32_t)S.blend[i].eq_rgb; d->blend[i].eq_a = (uint32_t)S.blend[i].eq_a;
        d->channel_mask[i] = S.chmask.mask >> (4 * i) & 0xf;
    }
    d->num_colors = (uint32_t)S.ncol; for (int i = 0; i < S.ncol && i < 8; i++) d->color_format[i] = (int32_t)S.cfmt[i];
    d->depth_format = (int32_t)S.dfmt;
    if (S.dfmt) g_rec->X.df_kept = (int32_t)S.dfmt; d->depth_format_kept = g_rec->X.df_kept;   // B7: the zeta register keeps the last depth target's format (S1c: rec_xs_t, was a static)
    d->provoking_last = 1;   // B7: the Maxwell path has the last vertex for every NVN pipeline (NVN's convention)
    d->lower_left_origin = (uint32_t)g_lower_left; d->depth_mode_minus_one_to_one = !g_depth_zero; d->point_size = 1.0f;
    for (int v = 0; v < 16; v++) { uint32_t r = 0; for (int c = 0; c < 4; c++) r |= (uint32_t)((S.vp_sw[v][c] ^ (2 * c)) & 7) << (4 * c); d->viewport_swizzle[v] = r; }
    for (int loc = 0; loc < 16; loc++) {   // every attribute the game set: the pipeline takes those its vertex shader loads
        va_st a; memcpy(&a, &S.vattr[loc], 4); if (!a.fmt) continue;
        VkFormat f = vertex_format(a.fmt, 0, 0); if (f == VK_FORMAT_UNDEFINED) continue;
        d->attrib[d->num_attribs].location = (uint32_t)loc; d->attrib[d->num_attribs].stream = a.stream; d->attrib[d->num_attribs].offset = a.offset; d->attrib[d->num_attribs].vk_format = (int32_t)f; d->num_attribs++;
    }
    d->num_streams = 16; for (int s = 0; s < 16; s++) { d->stream[s].stride = S.vstream[s].stride; d->stream[s].divisor = S.vstream[s].divisor; }
    // s30b: transform feedback while the game's capture is on (OP_TF_STATE Begin/Resume .. End/Pause) and the vertex shader declares one (its control section,
    // gfx_vk_shader.c vk_xfb_note): the layout goes to Eden's key, so the pipeline's vertex stage writes the captured outputs; draw() / draw_idx() bind the
    // buffers and begin the capture (xfb_begin, the same condition; DM_VK_XFB=0 turns both off, also live by DM_CTL). Before, no Eden pipeline captured
    // and the game's GPU particles kept stale state
    d->xfb = g_tf_on && g_xfb_on && V.has_xfb && S.sh[0] && S.sh[0]->xfb_layout ? S.sh[0]->xfb_layout : NULL;
    if (d->xfb) { static uint64_t n_all; const uint64_t n = __atomic_add_fetch(&n_all, 1, __ATOMIC_RELAXED); if (n <= 2 || !(n & (n - 1))) VKLOG(   /* S1c (parallel recording): atomic */"s30b: draws capturing transform feedback through Eden's pipelines: %llu (vertex shader %016llx, buffer 0 stride %u)",
                                                                     (unsigned long long)n, (unsigned long long)S.sh[0]->key, d->xfb->stride[0]); }
    d->vertex_pull = (uint32_t)g_vpull_want;   // vtxpull: eden_graphics decided it just before (vpull_decide; S1c: the thread's decision)
}

// ---- pipelines: the replay's cache in front of the bridge (whose lookup builds environments and hashes code) ----
typedef struct { nvn_eden_shader *sh[5]; nvn_eden_draw_state d; } eden_gkey;
typedef struct { nvn_eden_pipeline *p; nvn_eden_pipeline_info pi; int ready; eden_gkey k0; int s64w; } eden_pipe;   // s64w: 0 not looked at, 1 no storage writes, 2 writes storage   // s21 k0: the replay key that made it (diagnostic)
// S1c (parallel recording): ready and s64w are published once with release / read with acquire (eden_ready_w, eden_draw, eden_dispatch_prep); pi is written once, before
// ready, under g_eden_mu, and read only by those that saw ready; p and k0 are set before the entry is put into the maps (eden_pipe_put, under g_eden_mu)
static nvn_hmap g_eden_gp = {.klen = sizeof(eden_gkey)}, g_eden_cp = {.klen = sizeof(void *)}; static pthread_mutex_t g_eden_mu = PTHREAD_MUTEX_INITIALIZER;
// R1 (parallel recording): the Eden side's per-thread recording caches, grouped as struct rec_ctx is (gfx_vk_exec.c): one static instance, the replay
// thread's, reached through g_rec_e, which defaults to it in every thread; the old names stay as macros. The types below were defined further down
typedef struct { const void *sh[5], *xfb; int32_t prim, ncol, cfmt[8], dfmt, samples, ll, dz, tf, xon, kdyn, vpull; } s45_ctx;   // vpull: vtxpull's decision (not from record ops)
typedef struct { int ok, hv, tb, sb, storage_base; uint32_t binst; uint64_t epoch, bind, ubo, ubd, tex, list, h; VkBuffer b; VkDeviceSize o; } s45_dm_t;
// R3 (parallel recording): the host-visible rings with device addresses (descriptor buffer pipelines, below: eden_ring_alloc); the rings a thread allocates
// from are its struct rec_eden's (dring, uring), the replay thread's being these two
enum { EDEN_CHUNK = 4 << 20, EDEN_NCHUNK = 64 };
typedef struct { VkBuffer b; VkDeviceMemory m; uint8_t *p; VkDeviceAddress a; uint64_t tag; } eden_chunk;
typedef struct { VkBufferUsageFlags usage; eden_chunk c[EDEN_NCHUNK]; int n, cur; VkDeviceSize used; } eden_ring;
static eden_ring g_eden_dring = {VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT},
                 g_eden_uring = {VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
struct rec_eden {
    eden_ring *dring, *uring;   // R3 (parallel recording): this thread's descriptor-buffer and uniform rings (eden_bind_set, eden_addr_copy, the vertex pull table)
    eden_gkey g_eden_lk; eden_pipe *g_eden_lp;   // the last draw's
    s45_ctx g_s45_lctx; nvn_eden_draw_state g_s45_ld; uint64_t g_s45_kseen, g_s45_dver;   // g_s45_dver: the draw state's content version
    s45_dm_t g_s45_dm[5][2];
    int g_s35_pok; uint64_t g_s35_pgen; const void *g_s35_play; uint32_t g_s35_pn; eden_entry g_s35_ppay[EDEN_MAXE];
    VkPipeline g_s35_lastbound;
    struct { uint64_t gen; uint32_t n; VkViewport vp; VkRect2D sc; } g_eden_vpm;
    struct { int ok; uint64_t gen, dver; nvn_eden_pipeline *p; nvn_eden_dyn o; nvn_eden_draw_state d; } g_s45_st;
    struct { uint64_t gen; float lo, hi; } g_s45_db;
    struct { uint64_t gen; VkBuffer b; VkDeviceSize o; uint32_t stride; } g_eden_vbm[16];
    struct { uint64_t gen, a, n, wver, stale; uint32_t pgen; int var; VkBuffer b; VkDeviceSize o; } vlk[16];   // eden_draw's vertex stream lookups (was static there)
    eden_entry pay_draw[EDEN_MAXE], pay_disp[EDEN_MAXE];   // eden_draw's and eden_dispatch_prep's payloads (were static there)
    uint64_t dbuf_gen; VkDeviceAddress dbuf_bound;   // R1b (parallel recording): eden_bind_set's descriptor buffer memo (were static gen, bound there)
    // R1b (parallel recording): the per-thread statistics (were statics; read and reset by the replay thread's logs, which see the replay thread's instance)
    uint64_t g_s45_nk[2], g_s45_ns[2], g_s45_nd[2], g_s45_bad[3];   // kept / made: key, Eden state, driver segments; mismatches (verify)
    uint64_t g_s35_skip, g_s35_push, g_s35_ns[3];
    uint64_t g_vlk_hit, g_vlk_miss;   // replay2: vertex stream lookups kept / made (eden_draw)
    // S1c (parallel recording): more per-thread state of a recording (were file or function statics); the counters are added into the replay thread's by vk_rec_ctx_fold
    uint64_t g_vps_pulled, g_vps_fb[VPF_N];   // vtxpull: draws pulled, fallbacks by reason (vpull_frame logs and resets the replay thread's)
    uint64_t prof_n, prof_s[4];               // eden_prof: draws timed since its last line, and their parts (were its statics n, s[4])
    uint8_t c1buf[65536], ubobuf[65536 + 256];   // eden_ubo's copies of a shader's c1 and of a uniform buffer's deferred-update shadow (descriptor buffer pipelines; were statics there)
};
static struct rec_eden g_rec_e_main = {.dring = &g_eden_dring, .uring = &g_eden_uring, .g_s45_kseen = ~0ull, .g_s45_db = {~0ull}};   // R1 (parallel recording): the replay thread's (initial values as before); R3: with the shared rings
static _Thread_local struct rec_eden *g_rec_e = &g_rec_e_main;   // R1 (parallel recording): this thread's
static inline struct rec_eden *rec_e_cur(void) { return g_rec_e; }   // R1: for a function that keeps g_rec_e in a local of the same name
static inline vk_rec_target *rec_rt_cur(void) { return g_vk_rt; }   // R2 (parallel recording): for a function that keeps g_vk_rt in a local of the same name
#define g_eden_lk (g_rec_e->g_eden_lk)
#define g_eden_lp (g_rec_e->g_eden_lp)
#define g_s45_lctx (g_rec_e->g_s45_lctx)
#define g_s45_ld (g_rec_e->g_s45_ld)
#define g_s45_kseen (g_rec_e->g_s45_kseen)
#define g_s45_dver (g_rec_e->g_s45_dver)
#define g_s45_dm (g_rec_e->g_s45_dm)
#define g_s35_pok (g_rec_e->g_s35_pok)
#define g_s35_pgen (g_rec_e->g_s35_pgen)
#define g_s35_play (g_rec_e->g_s35_play)
#define g_s35_pn (g_rec_e->g_s35_pn)
#define g_s35_ppay (g_rec_e->g_s35_ppay)
#define g_s35_lastbound (g_rec_e->g_s35_lastbound)
#define g_eden_vpm (g_rec_e->g_eden_vpm)
#define g_s45_st (g_rec_e->g_s45_st)
#define g_s45_db (g_rec_e->g_s45_db)
#define g_eden_vbm (g_rec_e->g_eden_vbm)
#define g_s45_nk (g_rec_e->g_s45_nk)   // R1b (parallel recording)
#define g_s45_ns (g_rec_e->g_s45_ns)
#define g_s45_nd (g_rec_e->g_s45_nd)
#define g_s45_bad (g_rec_e->g_s45_bad)
#define g_s35_skip (g_rec_e->g_s35_skip)
#define g_s35_push (g_rec_e->g_s35_push)
#define g_s35_ns (g_rec_e->g_s35_ns)
#define g_vlk_hit (g_rec_e->g_vlk_hit)
#define g_vlk_miss (g_rec_e->g_vlk_miss)
#define g_vps_pulled (g_rec_e->g_vps_pulled)   // S1c (parallel recording)
#define g_vps_fb (g_rec_e->g_vps_fb)
// vtxpull (S1c (parallel recording): moved below struct rec_eden, which now holds its counters)
static void vpull_frame(void) {   // at the first decision of each present: the switch applied, every ~600 presents (~10 s) the summary
    extern uint64_t g_tex_frame;
    if (g_rec_e != &g_rec_e_main) return;   // S1c: the replay thread's (the switch and the log); a worker's draws read the switch as it is
    if (g_tex_frame == g_vpull_fr) return; g_vpull_fr = g_tex_frame;
    if (__atomic_load_n(&g_vpull_on, __ATOMIC_RELAXED) != g_vpull_cfg) { __atomic_store_n(&g_vpull_on, g_vpull_cfg, __ATOMIC_RELAXED); g_vps_pulled = 0; memset(g_vps_fb, 0, sizeof g_vps_fb); g_vps_logp = g_tex_frame;
        VKLOG("vtxpull: vertex pulling %s from present %llu", g_vpull_cfg ? "on" : "off", (unsigned long long)g_tex_frame); }
    if (!g_vpull_cfg || g_tex_frame < g_vps_logp + 600) return;
    const double f = (double)(g_tex_frame - g_vps_logp); char s[512]; size_t n = 0; s[0] = 0;
    for (int i = 0; i < VPF_N; i++) if (g_vps_fb[i] && n + 96 < sizeof s) n += (size_t)snprintf(s + n, sizeof s - n, "; %s %.1f", g_vpf_name[i], g_vps_fb[i] / f);
    VKLOG("vtxpull: per present over %.0f presents: %.1f draws pulled, fallbacks to vertex buffers%s (vbmir line: mirror copies)", f, g_vps_pulled / f, s[0] ? s : ": none");
    g_vps_pulled = 0; memset(g_vps_fb, 0, sizeof g_vps_fb); g_vps_logp = g_tex_frame;
}
static int vpull_decide(int prim) {   // 1: ask for the pulling pipeline; g_vpull_tab holds the streams
    vpull_frame(); g_vpull_want = 0; g_vpull_why = -1;
    if (!__atomic_load_n(&g_vpull_on, __ATOMIC_RELAXED)) return 0;
    if (!g_vpull_direct || prim == DM_PRIM_MESH_TASKS) { g_vpull_why = VPF_DRAW; return 0; }
    // S1c (parallel recording): vpull_known is published after vpull_inputs (eden_draw: 1 = known, release); 2 = being written
    if (!S.sh[0] || __atomic_load_n(&S.sh[0]->vpull_known, __ATOMIC_ACQUIRE) != 1) { g_vpull_why = VPF_VS; return 0; }
    const uint32_t inputs = __atomic_load_n(&S.sh[0]->vpull_inputs, __ATOMIC_RELAXED);
    uint32_t streams = 0;
    for (int loc = 0; loc < 16; loc++) { va_st a; memcpy(&a, &S.vattr[loc], 4); if (a.fmt && (inputs >> loc & 1)) streams |= 1u << a.stream; }
    memset(g_vpull_tab, 0, sizeof g_vpull_tab);
    for (uint32_t s = 0; s < 16; s++) if (streams >> s & 1) {
        const uint64_t a = S.vbuf[s].address, n = S.vbuf[s].size;
        if (!a || !n || n > 0xffffffffull) { g_vpull_why = VPF_ADDR; return 0; }
        vkpool_t *p = vk_pool_of(a, n);
        if (!p || !p->buf || !p->zero_copy || p->bridge || !p->gpu || a < p->lo || a + n > p->hi) { g_vpull_why = VPF_POOL; return 0; }
        if (vk_sub_gw_overlaps(a, n)) { g_vpull_why = VPF_GPUW; return 0; }
        g_vpull_tab[s].addr = p->gpu + (a - p->lo); g_vpull_tab[s].stride = S.vstream[s].stride; g_vpull_tab[s].size = (uint32_t)n;
    }
    return g_vpull_want = 1;
}
static uint64_t g_eden_skip_build, g_eden_wait_ns;
// s21: the replay key keeps only what Eden's key keeps (FillNvnGraphicsKey), so draws that differ only in state the Eden pipeline takes
// dynamically find the same entry. Run 23 (Bayonetta 3 gameplay): with extended dynamic state 1 Eden's key has no vertex strides, but the
// replay key had all 16 streams' strides (also of streams no attribute reads), so ~1,000 new keys a second came while Eden made almost no
// new pipelines (GPU_OBJECTS 1887 -> 1973, replay keys 50 -> 132,551 in 4 min); each built environments and hashed code again, and the map
// never shrinks (private memory 6.0 -> 7.0 GB, then the working set was trimmed and submits stalled 50-80 ms). The depth bounds are kept
// as Eden keeps them, truncated to integers (the values are dynamic state). The draw keeps the real values: nvn_eden_cmd_state, the vertex
// binds and the first build of a key get them. DM_VK_EDEN_KEYNORM=0: the raw key, as before s21 (the dedup and its diagnostic stay).
static int g_eden_kdyn = -1;   // 1: Eden's key has no strides (its pipelines have NVN_ED_EDS1); -1 until the first graphics pipeline says
static nvn_hmap g_eden_pe = {.klen = sizeof(void *)};   // Eden pipeline -> its one eden_pipe, whichever replay keys reach it
static uint64_t g_eden_kmiss, g_eden_kdup;
static int eden_keynorm(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_EDEN_KEYNORM"); v = !e || atoi(e) != 0; } return v; }
static void eden_key_norm(nvn_eden_draw_state *d) {
    if (!eden_keynorm()) return;
    if (__atomic_load_n(&g_eden_kdyn, __ATOMIC_RELAXED) > 0) for (int s = 0; s < 16; s++) d->stream[s].stride = 0;   // (S1c (parallel recording): published by eden_ready_w)
    for (int i = 0; i < 2; i++) {   // as static_cast<u32>(float) on ARM64 (fcvtzu): NaN and negatives 0, too large the maximum; the bits go in the float's place
        float f = d->depth_bounds[i]; uint32_t u = f > 0.0f ? (f >= 4294967296.0f ? 0xffffffffu : (uint32_t)f) : 0u; memcpy(&d->depth_bounds[i], &u, 4);
    }
}
static void eden_key_diff(const eden_gkey *a, const eden_gkey *b) {   // which fields told apart two replay keys that reached one Eden pipeline
#define EKF(x) {#x, (uint32_t)__builtin_offsetof(nvn_eden_draw_state, x), (uint32_t)sizeof(((nvn_eden_draw_state *)0)->x)}
    static const struct { const char *n; uint32_t o, len; } f[] = {
        EKF(primitive), EKF(patch_size), EKF(polygon_mode), EKF(cull_face), EKF(front_face), EKF(depth_clamp), EKF(polygon_offset),
        EKF(rasterizer_discard), EKF(primitive_restart), EKF(depth_test), EKF(depth_write), EKF(depth_func), EKF(depth_bounds_test),
        EKF(depth_bounds), EKF(stencil_test), EKF(stencil_func), EKF(stencil_fail), EKF(stencil_zfail), EKF(stencil_zpass), EKF(alpha_func),
        EKF(alpha_ref), EKF(samples), EKF(alpha_to_coverage), EKF(alpha_to_one), EKF(logic_op_enable), EKF(logic_op), EKF(blend_enable),
        EKF(blend), EKF(channel_mask), EKF(num_colors), EKF(color_format), EKF(depth_format), EKF(lower_left_origin),
        EKF(depth_mode_minus_one_to_one), EKF(provoking_last), EKF(conservative_raster), EKF(line_smooth), EKF(line_stipple),
        EKF(line_stipple_factor), EKF(line_stipple_pattern), EKF(point_size), EKF(early_fragment), EKF(viewport_swizzle), EKF(num_attribs),
        EKF(attrib), EKF(num_streams), EKF(depth_format_kept), EKF(xfb), EKF(vertex_pull)};
#undef EKF
    char s[480]; size_t n = 0; s[0] = 0;
    if (memcmp(a->sh, b->sh, sizeof a->sh)) n += (size_t)snprintf(s + n, sizeof s - n, " shaders");
    for (size_t i = 0; i < sizeof f / sizeof f[0] && n + 48 < sizeof s; i++)
        if (memcmp((const char *)&a->d + f[i].o, (const char *)&b->d + f[i].o, f[i].len)) n += (size_t)snprintf(s + n, sizeof s - n, " %s", f[i].n);
    for (int k = 0; k < 16 && n + 48 < sizeof s; k++) {
        if (a->d.stream[k].stride != b->d.stream[k].stride) n += (size_t)snprintf(s + n, sizeof s - n, " stream[%d].stride %u/%u", k, a->d.stream[k].stride, b->d.stream[k].stride);
        if (a->d.stream[k].divisor != b->d.stream[k].divisor) n += (size_t)snprintf(s + n, sizeof s - n, " stream[%d].divisor %u/%u", k, a->d.stream[k].divisor, b->d.stream[k].divisor);
    }
    VKLOG("s21: replay key %llu reached an Eden pipeline an earlier key made; the keys differ in:%s", (unsigned long long)g_eden_kmiss, s[0] ? s : " (nothing)");
}
static int eden_async(void) {   // s69: Eden's use_asynchronous_shaders (the gate hands it over, nvnh_set_eden_async); DM_VK_EDEN_ASYNC (0/1) overrides
    static int v = -1;
    if (v < 0) { extern int nvnh_eden_async_setting(void); const char *e = getenv("DM_VK_EDEN_ASYNC"); const int s = nvnh_eden_async_setting();
        v = e && *e ? atoi(e) != 0 : s > 0;
        char ov[32]; snprintf(ov, sizeof ov, "%s", e && *e ? e : "none");
        VKLOG("s69: Eden pipeline skip-until-built %s (setting use_asynchronous_shaders=%s, env override %s)", v ? "on" : "off", s < 0 ? "unknown" : s ? "1" : "0", ov); }
    return v;
}   // 1: a draw whose pipeline is still being built is skipped (Eden waits)
// glyphwait (11-20): with async shaders a draw whose pipeline is still being built was skipped, every one. A draw whose result the game keeps
// (a glyph drawn once into its font atlas, a LUT, a render-to-texture written once) then stays missing for the whole run: letters left blank in menus.
// Eglantine PR #785 (7a940ca19d, PipelineCache::BuiltPipeline) / Lichii PR #1296: native draws without a depth test wait for their pipeline. Here:
// `wait` = 1 for a draw with no depth test (or no depth target) and for every compute dispatch (its writes always persist); the skip stays only for
// depth-tested draws (scene geometry, redrawn every frame)
static uint64_t g_eden_wait_kept;   // draws and dispatches that waited for their pipeline instead of being skipped
// S1c (parallel recording): once-publication. The info is asked into a local (the bridge's wait for a build may take seconds: no lock is held across it); the first
// thread with a built pipeline publishes it under g_eden_mu: pi, the set layout's registration, the s21 key rule (g_eden_kdyn), then ready with release. A thread that
// sees ready (acquire) reads the published pi; two threads that both asked publish once (the second finds ready set under the lock). The counters are atomic
static int eden_pipe_publish(eden_pipe *e, const nvn_eden_pipeline_info *pi);   // S1c (parallel recording): below
static int eden_ready_w(eden_pipe *e, int wait) {
    if (__atomic_load_n(&e->ready, __ATOMIC_ACQUIRE)) return 1;
    uint64_t t0 = tnow();
    int async = eden_async() && !wait;
    if (eden_async() && wait) __atomic_add_fetch(&g_eden_wait_kept, 1, __ATOMIC_RELAXED);
    nvn_eden_pipeline_info pi; memset(&pi, 0, sizeof pi);
    if (!nvn_eden_pipeline_get(e->p, &pi, !async)) return 0;
    if (!pi.built) {
        if (async) { uint64_t n = __atomic_add_fetch(&g_eden_skip_build, 1, __ATOMIC_RELAXED);
            if (n <= 4 || !(n & (n - 1))) VKLOG("glyphwait: depth-tested draws skipped while their pipeline builds: %llu; draws and dispatches that waited for theirs instead: %llu",
                                                (unsigned long long)n, (unsigned long long)__atomic_load_n(&g_eden_wait_kept, __ATOMIC_RELAXED)); }
        else VKRED_ONCE(160, "an Eden pipeline was not built in 10 s: its draws are skipped");
        return 0;
    }
    __atomic_add_fetch(&g_eden_wait_ns, tnow() - t0, __ATOMIC_RELAXED);
    if (!pi.pipeline) { VKRED_ONCE(161, "an Eden pipeline failed to build: its draws are skipped"); return 0; }
    if (pi.num_entries > EDEN_MAXE) { VKRED_ONCE(162, "an Eden pipeline has %u descriptors (more than %d): its draws are skipped", pi.num_entries, EDEN_MAXE); return 0; }
    if (pi.entry_size != sizeof(eden_entry)) { VKRED_ONCE(163, "Eden's descriptor entry is %u bytes, not %zu: its draws are skipped", pi.entry_size, sizeof(eden_entry)); return 0; }
    (void)eden_pipe_publish(e, &pi);
    return 1;
}
// S1c (parallel recording): e's info published once (eden_ready_w); 1: this call published it, 0: another thread had
static int eden_pipe_publish(eden_pipe *e, const nvn_eden_pipeline_info *pi) {
    int published = 0;
    pthread_mutex_lock(&g_eden_mu);
    if (!__atomic_load_n(&e->ready, __ATOMIC_RELAXED)) {
        e->pi = *pi; published = 1;
        if (e->pi.set_layout && !e->pi.push_descriptor && !e->pi.descriptor_buffer) {   // vk_alloc_set counts what a set takes by its layout
            static const VkDescriptorType ty[6] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
                                                   VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE};
            VkDescriptorSetLayoutBinding lb[256]; uint32_t n = 0;
            for (uint32_t i = 0; i < e->pi.num_binds && n < 256; i++) { memset(&lb[n], 0, sizeof lb[n]); lb[n].binding = n; lb[n].descriptorType = ty[e->pi.binds[i].kind < 6 ? e->pi.binds[i].kind : 0]; lb[n].descriptorCount = e->pi.binds[i].count; n++; }
            vk_dsl_register((VkDescriptorSetLayout)e->pi.set_layout, lb, n);
        }
        if (__atomic_load_n(&g_eden_kdyn, __ATOMIC_RELAXED) < 0 && !e->pi.compute) {   // s21
            const int kd = (e->pi.dyn & NVN_ED_EDS1) != 0; __atomic_store_n(&g_eden_kdyn, kd, __ATOMIC_RELAXED);
            VKLOG("s21: Eden's graphics pipelines have%s extended dynamic state 1 (dyn 0x%x): the replay key %s the vertex strides%s", kd ? "" : " no",
                  e->pi.dyn, kd && eden_keynorm() ? "drops" : "keeps", eden_keynorm() ? "" : " (DM_VK_EDEN_KEYNORM=0)");
        }
        __atomic_store_n(&e->ready, 1, __ATOMIC_RELEASE);
    }
    pthread_mutex_unlock(&g_eden_mu);
    return published;
}
static int eden_ready(eden_pipe *e) { return eden_ready_w(e, 0); }   // (lookups that only ask)
static int eden_draw_must_wait(void) {   // glyphwait: no depth test, or no depth target: its colour output is the result the game keeps
    return !((S.daspect & VK_IMAGE_ASPECT_DEPTH_BIT) && S.ds.dt);
}
// s45 (DM_VK_S45 & 1): what the key reads that no record op carries; with it and g_s45_kgen unchanged the last key and draw state stand
// R1 (parallel recording): s45_ctx (defined above), g_s45_lctx, g_s45_ld, g_s45_kseen, g_s45_dver in struct rec_eden
static void s45_ctx_fill(s45_ctx *c, int prim) {
    memset(c, 0, sizeof *c);
    for (int st = 0; st < 5; st++) c->sh[st] = S.sh[st] ? (const void *)S.sh[st]->eden : NULL;
    c->xfb = S.sh[0] ? (const void *)S.sh[0]->xfb_layout : NULL;
    c->prim = prim; c->ncol = (int32_t)S.ncol; for (int i = 0; i < S.ncol && i < 8; i++) c->cfmt[i] = (int32_t)S.cfmt[i];
    c->dfmt = (int32_t)S.dfmt; c->samples = (int32_t)S.samples; c->ll = (int32_t)g_lower_left; c->dz = (int32_t)g_depth_zero;
    c->tf = (int32_t)g_tf_on; c->xon = (int32_t)g_xfb_on; c->kdyn = (int32_t)__atomic_load_n(&g_eden_kdyn, __ATOMIC_RELAXED); c->vpull = g_vpull_want;
}
static void s45_kbad(const eden_gkey *k, const nvn_eden_draw_state *d) {
    if (++g_s45_bad[0] > 8) return;
    const uint8_t *a = (const uint8_t *)d, *b = (const uint8_t *)&g_s45_ld; size_t i = 0; while (i < sizeof *d && a[i] == b[i]) i++;
    VKLOG("[RED] s45: pipeline key mismatch %llu: no op that changes its inputs ran, yet the key %s (draw state first differs at byte %zu of %zu); the s21 line next names the fields",
          (unsigned long long)g_s45_bad[0], memcmp(k, &g_eden_lk, sizeof *k) ? "changed" : "is the same and the draw state is not", i, sizeof *d);
    eden_key_diff(&g_eden_lk, k);
}
// S1c (parallel recording): a replay key's entry after a miss, get-or-insert in one hold of g_eden_mu (was a get and three puts in separate holds, so two threads that
// missed one key could each calloc an eden_pipe for one Eden pipeline). Both threads asked the bridge (outside the lock: a miss may build for long; Eden makes the
// pipeline once and gives both the same one); the first to get here puts the key, and its Eden pipeline's one eden_pipe (s21: found by pipeline, or made with p and
// k0 set before it is put); the second takes what the first put. *made: this call put the key (counted: g_eden_kmiss, pipes built). (void *)1: no pipeline
static eden_pipe *eden_pipe_put(const eden_gkey *k, nvn_eden_pipeline *p, int *made) {
    pthread_mutex_lock(&g_eden_mu);
    eden_pipe *e = nvn_hm_get(&g_eden_gp, k);
    if (!e) {
        *made = 1; g_eden_kmiss++;
        if (p) {   // s21: an Eden pipeline that an earlier key reached keeps its one eden_pipe (no second calloc, eden_ready, set layout registration)
            e = nvn_hm_get(&g_eden_pe, &p);
            if (e) { if (++g_eden_kdup <= 16) eden_key_diff(&e->k0, k); }
            else { e = calloc(1, sizeof *e); if (e) { e->p = p; e->k0 = *k; nvn_hm_put(&g_eden_pe, &p, e); } }
        }
        if ((g_eden_kmiss >= 256 && !(g_eden_kmiss & (g_eden_kmiss - 1))) || g_eden_kmiss % 4096 == 0)
            VKLOG("s21: replay pipeline keys: %llu made, %llu of them reached an Eden pipeline an earlier key made", (unsigned long long)g_eden_kmiss, (unsigned long long)g_eden_kdup);
        if (!e) e = (void *)1;
        nvn_hm_put(&g_eden_gp, k, e);
    }
    pthread_mutex_unlock(&g_eden_mu);
    return e;
}
static eden_pipe *eden_graphics(int prim, nvn_eden_draw_state *d) {
    vpull_decide(prim);   // vtxpull: before the key (s45_ctx_fill, eden_fill_state read it)
    s45_ctx c45; s45_ctx_fill(&c45, prim); int khit = 0;
    int fast = (g_s45_on & 1) && g_eden_lp && g_s45_kseen == g_s45_kgen && !memcmp(&c45, &g_s45_lctx, sizeof c45);
    if (fast && !g_s45_verify) { *d = g_s45_ld; g_s45_nk[0]++; return g_eden_lp == (void *)1 ? NULL : g_eden_lp; }
    eden_gkey k; memset(&k, 0, sizeof k);
    for (int st = 0; st < 5; st++) k.sh[st] = S.sh[st] ? S.sh[st]->eden : NULL;
    if (prim == DM_PRIM_MESH_TASKS) {
        const int has_task = S.sh[3] && S.sh[3]->stage == 6;
        k.sh[0] = has_task ? k.sh[3] : k.sh[4];
        k.sh[4] = has_task ? k.sh[4] : NULL;
        k.sh[2] = k.sh[3] = NULL;
    }
    eden_fill_state(&k.d, prim == DM_PRIM_MESH_TASKS ? 4 : prim); *d = k.d; eden_key_norm(&k.d);   // s21: the key as Eden's; d keeps the real values
    if (fast) { g_s45_nk[0]++; if (memcmp(&k, &g_eden_lk, sizeof k) || memcmp(d, &g_s45_ld, sizeof *d)) s45_kbad(&k, d); else khit = 1; } else g_s45_nk[1]++;
    g_s45_kseen = g_s45_kgen; g_s45_lctx = c45;
    if (!khit && memcmp(d, &g_s45_ld, sizeof *d)) { g_s45_ld = *d; g_s45_dver++; }
    if (g_eden_lp && !memcmp(&k, &g_eden_lk, sizeof k)) return g_eden_lp == (void *)1 ? NULL : g_eden_lp;
    pthread_mutex_lock(&g_eden_mu); eden_pipe *e = nvn_hm_get(&g_eden_gp, &k); pthread_mutex_unlock(&g_eden_mu);
    if (!e) {
        uint64_t t80 = tnow(); nvn_eden_pipeline *p = nvn_eden_graphics_pipeline(k.sh, d, &g_eden_q);   // s21: the real draw state
        __atomic_add_fetch(&g_r80_pso, 1, __ATOMIC_RELAXED); __atomic_add_fetch(&g_r80_pso_ns, tnow() - t80, __ATOMIC_RELAXED);
        int made = 0; e = eden_pipe_put(&k, p, &made);   // S1c (parallel recording): get-or-insert (the entry another thread put meanwhile, if any)
        if (made && e == (void *)1) { static int n_all; if (__atomic_fetch_add(&n_all, 1, __ATOMIC_RELAXED) < 16) VKLOG("[RED] Eden graphics pipeline not made: vertex %016llx fragment %016llx (primitive %d)", S.sh[0] ? (unsigned long long)S.sh[0]->key : 0ull, S.sh[1] ? (unsigned long long)S.sh[1]->key : 0ull, prim); }
        if (made) __atomic_add_fetch(&g_vkst.pipes_built, 1, __ATOMIC_RELAXED);
    }
    g_eden_lk = k; g_eden_lp = e;
    return e == (void *)1 ? NULL : e;
}
static eden_pipe *eden_compute(vkshader_t *cs) {
    pthread_mutex_lock(&g_eden_mu); eden_pipe *e = nvn_hm_get(&g_eden_cp, &cs->eden); pthread_mutex_unlock(&g_eden_mu);
    if (!e) {
        // bak85 (11-16, MKW scenery / cactus flicker): the IR of the shaders whose first invocation resets a list the others append to (nxnative MKW note "Vanishing
        // scenery"; 649e36162cad9e79 the one dmsh_build.py's first_store refuses), written once each into the log folder (nvn_eden_dump_ir). DM_NVN2_IR_DUMP=1 enables this diagnostic
        if (dm_sdk_ounce() && cs->eden) {
            static int on = -1; if (on < 0) { const char *ev = getenv("DM_NVN2_IR_DUMP"); on = ev && atoi(ev) != 0; }
            // stage1: OoT's heaviest dispatches too, bdf51d78 (f8965bffbdf51d78, Eden a107e12b6771be48) and f4f0bc23 (d17eefaff4f0bc23, Eden d9ee412f00ce399a)
            static const uint64_t want[6] = {0x2ce9071c2f22d74full, 0x5fea9b7ddb30e8a8ull, 0x0823b08c2027a38dull, 0x649e36162cad9e79ull, 0xf8965bffbdf51d78ull, 0xd17eefaff4f0bc23ull};
            for (int i = 0; on && i < 6; i++) if (cs->key == want[i]) {
                const int ok = nvn_eden_dump_ir(cs->eden, &g_eden_q, cs->key);
                LOG("[nvn] [vk] first_store probe: compute shader %016llx (Eden hash %016llx): IR %s", (unsigned long long)cs->key, (unsigned long long)nvn_eden_shader_hash(cs->eden),
                    ok ? "written to the log folder's nvn2_ir_<key>.txt" : "NOT written");
            } }
        uint64_t t80 = tnow(); nvn_eden_pipeline *p = nvn_eden_compute_pipeline(cs->eden, &g_eden_q);
        __atomic_add_fetch(&g_r80_pso, 1, __ATOMIC_RELAXED); __atomic_add_fetch(&g_r80_pso_ns, tnow() - t80, __ATOMIC_RELAXED);
        // S1c (parallel recording): get-or-insert in one hold (as eden_pipe_put): a thread that missed the same shader meanwhile put its entry first, this one takes it
        pthread_mutex_lock(&g_eden_mu);
        e = nvn_hm_get(&g_eden_cp, &cs->eden);
        if (!e) {
            if (p) { e = calloc(1, sizeof *e); if (e) e->p = p; }
            if (!e) { LOG("[RED] [vk] Eden compute pipeline not made: %016llx", (unsigned long long)cs->key); e = (void *)1; }
            nvn_hm_put(&g_eden_cp, &cs->eden, e);
        }
        pthread_mutex_unlock(&g_eden_mu);
    }
    return e == (void *)1 ? NULL : e;
}

// ---- host-visible rings with device addresses (descriptor buffer pipelines): a chunk is used again once the GPU passed every submission that used it ----
// R3 (parallel recording): EDEN_CHUNK, eden_chunk, eden_ring and the two rings are defined before struct rec_eden (which points at them)
static VkDeviceAddress eden_bda(VkBuffer b) { VkBufferDeviceAddressInfo i = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, b}; return vkGetBufferDeviceAddress(V.dev, &i); }
static int eden_chunk_new(eden_ring *r) {
    if (r->n == EDEN_NCHUNK) return 0;
    eden_chunk *c = &r->c[r->n]; memset(c, 0, sizeof *c);
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, EDEN_CHUNK, r->usage, VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(V.dev, &bi, NULL, &c->b)) return 0;
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, c->b, &req);
    VkMemoryAllocateFlagsInfo af = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &af, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
    if (vkAllocateMemory(V.dev, &ai, NULL, &c->m)) { vkDestroyBuffer(V.dev, c->b, NULL); return 0; }
    void *p; if (vkBindBufferMemory(V.dev, c->b, c->m, 0) || vkMapMemory(V.dev, c->m, 0, VK_WHOLE_SIZE, 0, &p)) { vkDestroyBuffer(V.dev, c->b, NULL); vkFreeMemory(V.dev, c->m, NULL); return 0; }
    c->p = p; c->a = eden_bda(c->b); r->n++; return 1;
}
// n bytes aligned to al (a power of two) in memory the open submission may use: host pointer, chunk buffer, offset in it, chunk address
static uint8_t *eden_ring_alloc(eden_ring *r, VkDeviceSize n, VkDeviceSize al, VkBuffer *buf, VkDeviceSize *off, VkDeviceAddress *base) {
    if (!al) al = 16; if (n + al > EDEN_CHUNK) return NULL;
    for (int tries = 0; tries < 2; tries++) {
        if (r->n) { eden_chunk *c = &r->c[r->cur]; VkDeviceSize a0 = c->a + r->used, o = ((a0 + al - 1) & ~(al - 1)) - c->a;
            if (o + n <= EDEN_CHUNK) { r->used = o + n; c->tag = mtl_last_fence() + 1; *buf = c->b; *off = o; *base = c->a; return c->p + o; } }
        uint64_t done = vk_fence_done(); int next = -1;
        for (int i = 1; i <= r->n && next < 0; i++) { int j = (r->cur + i) % r->n; if (r->n && j != r->cur && r->c[j].tag <= done) next = j; }
        if (next < 0 && eden_chunk_new(r)) next = r->n - 1;
        if (next < 0 && !r->n) return NULL;
        if (next < 0) { int j = (r->cur + 1) % r->n; VKRED_ONCE(164, "Eden descriptor memory: all %d chunks in use, waiting for the GPU", r->n); mtl_wait(r->c[j].tag, 5000000000ull); if (r->c[j].tag > vk_fence_done()) return NULL; next = j; }
        r->cur = next; r->used = 0;
    }
    return NULL;
}

// ---- set 0: one payload entry per descriptor element ----
// dbuf: the pipeline reads a descriptor buffer, whose buffers are device addresses: memory the host made without one (driver segments, c1,
// uniform shadows of deferred updates) is copied into g_eden_uring
static _Thread_local int g_eden_dbuf_bad;   // S1c (parallel recording): per thread (eden_payload's draw or dispatch)
static void eden_addr_copy(eden_entry *e, const void *src, VkDeviceSize n, VkDeviceSize al) {
    VkBuffer b; VkDeviceSize o; VkDeviceAddress a; eden_ring *const ur = g_rec_e->uring; uint8_t *p = eden_ring_alloc(ur, n, al, &b, &o, &a);   // R3 (parallel recording): the thread's uniform ring
    if (!p) { g_eden_dbuf_bad = 1; e->addr.address = 0; e->addr.range = VK_WHOLE_SIZE; e->addr.format = VK_FORMAT_UNDEFINED; return; }
    memcpy(p, src, n); e->addr.address = a + o; e->addr.range = n; e->addr.format = VK_FORMAT_UNDEFINED;
}
static void eden_addr_of(eden_entry *e, const VkDescriptorBufferInfo *bi, VkFormat fmt) {   // a pool buffer range (pools have device addresses with V.has_bda)
    if (!bi->buffer || bi->buffer == vk_zero_buffer() || bi->buffer == vk_wsink_buffer()) { e->addr.address = 0; e->addr.range = VK_WHOLE_SIZE; e->addr.format = fmt; return; }
    VkDeviceAddress a = eden_bda(bi->buffer); e->addr.address = a ? a + bi->offset : 0; e->addr.range = a ? bi->range : VK_WHOLE_SIZE; e->addr.format = fmt;
}
// s45 (DM_VK_S45 & 8): a graphics stage's driver segment (c0, c2) as drv_shared made it last, while what drv_segment reads holds: the stage's bindings
// (storage buffers, handles, push constants: g_bind_gen), the base instance, the set epoch (the scratch memory's life); c2 also its uniform buffers and data,
// the texture generation, the pool bases and the top-level list. Saves drv_segment's 4 KB fill and the hash of up to 0x968 bytes a stage and draw.
// R1 (parallel recording): s45_dm_t (defined above) and g_s45_dm in struct rec_eden
static uint64_t s45_drv_hash(int st, int bank) { uint8_t d[0x1000]; drv_segment(st, bank, d); return fnv(0xcbf29ce484222325ull, d, drv_segment_size(st, bank)); }
static void s45_drv(int st, int bank, VkBuffer *buf, VkDeviceSize *off) {
    if (!(g_s45_on & 8) || st < 0 || st > 4 || (bank != 0 && bank != 2) || g_x_nodrvc) { drv_shared(st, bank, buf, off); return; }
    s45_dm_t *m = &g_s45_dm[st][bank >> 1];
    int hit = m->ok && m->storage_base == storage_table_offset(st) && m->epoch == g_set_epoch && m->bind == g_bind_gen[st] && m->binst == g_draw_binst &&
              (bank == 0 || (m->ubo == g_ubo_gen[st] && m->ubd == g_ubd_gen && m->tex == g_tex_gen && m->tb == S.tex_base && m->sb == S.samp_base && m->list == g_s45_list));
    if (hit && !g_s45_verify) { *buf = m->b; *off = m->o; g_s45_nd[0]++; return; }
    drv_shared(st, bank, buf, off);
    uint64_t h = g_s45_verify ? s45_drv_hash(st, bank) : 0;
    if (hit) { g_s45_nd[0]++; if (m->hv && h != m->h && ++g_s45_bad[2] <= 8) VKLOG("[RED] s45: driver segment mismatch %llu: stage %d c%d changed with its generations the same", (unsigned long long)g_s45_bad[2], st, bank); }
    else g_s45_nd[1]++;
    m->ok = 1; m->storage_base = storage_table_offset(st); m->epoch = g_set_epoch; m->bind = g_bind_gen[st]; m->binst = g_draw_binst; m->ubo = g_ubo_gen[st]; m->ubd = g_ubd_gen; m->tex = g_tex_gen;
    m->tb = S.tex_base; m->sb = S.samp_base; m->list = g_s45_list; m->b = *buf; m->o = *off; m->h = h; m->hv = g_s45_verify;
}
static void eden_ubo(int st, uint32_t bank, eden_entry *e, int dbuf) {
    const VkDeviceSize ual = V.props.limits.minUniformBufferOffsetAlignment ? V.props.limits.minUniformBufferOffsetAlignment : 256;
    if (bank == 0 || bank == 2) {   // the driver's c0 / c2 segment
        if (!dbuf) { VkBuffer b; VkDeviceSize o; s45_drv(st, (int)bank, &b, &o); e->buffer = (VkDescriptorBufferInfo){b, o, 0x1000}; return; }
        uint8_t d[0x1000]; drv_segment(st, (int)bank, d); eden_addr_copy(e, d, sizeof d, ual);
        if (g_drv_fresh && st == 5 && bank == 0 && e->addr.address) {   // r75: the indirect counts go into this copy
            const eden_ring *const ur = g_rec_e->uring; g_drv_lb = ur->c[ur->cur].b; g_drv_lo = e->addr.address - ur->c[ur->cur].a; }   // R3 (parallel recording): the ring eden_addr_copy took it from
        return;
    }
    if (bank == DMSH_PUSH_BANK && dm_sdk_ounce()) {   // NVN2: the graphics stages read the game's push constants at c[24] (InstEmitAmpere.cs:264); drv_segment's bank 24 is S.push
        if (!dbuf) { VkBuffer b; VkDeviceSize o; drv_shared(st, (int)bank, &b, &o); e->buffer = (VkDescriptorBufferInfo){b, o, 0x1000}; }
        else { uint8_t d[0x1000]; drv_segment(st, (int)bank, d); eden_addr_copy(e, d, sizeof d, ual); }
        static int said; if (!__atomic_exchange_n(&said, 1, __ATOMIC_RELAXED)) VKLOG(   /* S1c (parallel recording): once, atomically */"NVN2: push constants bound as c[24] for Eden (stage %d)", st);
        return;
    }
    if (bank == 1) { VkBuffer b; VkDeviceSize o, n; vk_shader_c1(S.sh[st], &b, &o, &n);
        if (!dbuf) { e->buffer = (VkDescriptorBufferInfo){b, o, n}; return; }
        const vkshader_t *s = S.sh[st]; if (s && s->c1 && s->c1_len) { uint8_t *const c = g_rec_e->c1buf; uint32_t len = s->c1_len > sizeof g_rec_e->c1buf ? (uint32_t)sizeof g_rec_e->c1buf : s->c1_len, al = (len + 255) & ~255u;   // S1c (parallel recording): the thread's buffer (struct rec_eden; was a static)
            memset(c, 0, al); memcpy(c, s->c1, len); eden_addr_copy(e, c, al, ual); }
        else { e->addr.address = 0; e->addr.range = VK_WHOLE_SIZE; e->addr.format = VK_FORMAT_UNDEFINED; }
        return; }
    dmsv_bind tb; memset(&tb, 0, sizeof tb); tb.kind = RK_CBUF; tb.desc = RD_UBO; tb.reg = (int16_t)(bank - 3);
    VkDescriptorBufferInfo bi; cbuf_desc(st, S.sh[st], &tb, &bi);
    if (!dbuf) { e->buffer = bi; return; }
    const p_bind_buf *u = bank - 3 < 16 ? &S.ubo[st][bank - 3] : NULL; vkpool_t *p = u && u->addr ? vk_pool_of(u->addr, 1) : NULL;
    if (p && bi.buffer == p->buf) { eden_addr_of(e, &bi, VK_FORMAT_UNDEFINED); return; }
    if (p && u->addr) { uint64_t n = bi.range; uint8_t *const c = g_rec_e->ubobuf; if (n > sizeof g_rec_e->ubobuf) n = sizeof g_rec_e->ubobuf; vk_ubd_read(u->addr, n, c); eden_addr_copy(e, c, n, ual); return; }   // the shadow of deferred updates (S1c (parallel recording): into the thread's buffer, struct rec_eden; was a static)
    eden_addr_of(e, &bi, VK_FORMAT_UNDEFINED);
}
// NVN2 diagnostic: storage buffers a translated shader binds that get the zero stand-in (address 0, or outside every pool) or a binding cut at its pool chunk's end
// (a size larger than the rest of the chunk). Counted per NVN stage; the first case of each (bind, slot) logged with what it asked for; the counts every 30 s when they
// changed. Switch 2 boots only, no per-draw lines
static uint64_t g_sd_zero[6], g_sd_cut[6], g_sd_logged_sum, g_sd_t;
static pthread_mutex_t g_sd_mu = PTHREAD_MUTEX_INITIALIZER;   // S1c (parallel recording): eden_ssbo_diag's state (draws of every recording thread reach it)
static void eden_ssbo_diag(int st, const nvn_eden_bind *b, uint32_t k, uint32_t off, uint64_t a, uint64_t req, uint64_t bound, int zero) {
    if (st < 0 || st > 5) return;
    pthread_mutex_lock(&g_sd_mu);   // S1c (parallel recording): the counts, the seen table and the 30 s line under one lock (a bind that is zero or cut: rare, uncontended)
    if (zero) g_sd_zero[st]++; else g_sd_cut[st]++;
    static uint64_t seen[2048]; static int nseen;
    const uint64_t id = ((uint64_t)(uintptr_t)b * 0x9E3779B97F4A7C15ull) ^ ((uint64_t)k << 1) ^ (uint64_t)zero;
    int known = 0; for (int i = 0; i < nseen && !known; i++) known = seen[i] == id;
    if (!known && nseen < 2048) { seen[nseen++] = id;
        const vkshader_t *sh = S.sh[st];
        VKLOG("NVN2 SSBO: stage %d shader %016llx slot %u (c%u+%#x, bind %p)%s: address %#llx, asked %#llx bytes, bound %#llx (%s)", st, (unsigned long long)(sh ? sh->key : 0),
              (unsigned)(b->first_entry + k), b->cbuf, off, (const void *)b, b->is_written ? " written" : "", (unsigned long long)a, (unsigned long long)req,
              (unsigned long long)bound, zero ? (a ? "outside every pool: zero stand-in" : "address 0: zero stand-in") : "cut at the pool chunk's end"); }
    const uint64_t now = tnow(); uint64_t sum = 0; for (int i = 0; i < 6; i++) sum += g_sd_zero[i] + g_sd_cut[i];
    if (sum != g_sd_logged_sum && now - g_sd_t >= 30000000000ull) { g_sd_t = now; g_sd_logged_sum = sum;
        VKLOG("NVN2 SSBO binds so far, zero stand-in / cut per stage (vertex, fragment, geometry, tess control, tess eval, compute): %llu/%llu %llu/%llu %llu/%llu %llu/%llu %llu/%llu %llu/%llu",
              (unsigned long long)g_sd_zero[0], (unsigned long long)g_sd_cut[0], (unsigned long long)g_sd_zero[1], (unsigned long long)g_sd_cut[1], (unsigned long long)g_sd_zero[2],
              (unsigned long long)g_sd_cut[2], (unsigned long long)g_sd_zero[3], (unsigned long long)g_sd_cut[3], (unsigned long long)g_sd_zero[4], (unsigned long long)g_sd_cut[4],
              (unsigned long long)g_sd_zero[5], (unsigned long long)g_sd_cut[5]); }
    pthread_mutex_unlock(&g_sd_mu);
}
static _Thread_local uint64_t g_eden_cs_key;   // (S1c (parallel recording): per thread) bak91c: the NVN key of the dispatch whose bindings eden_payload is making (eden_dispatch_prep; 0 for draws)
// These two measured fullscreen fragment programs use only SSBO slot 0. On N1X,
// atomics against imported host pages are expensive. Stage the exact descriptor
// range for one direct draw, then publish its result before the next guest command.
// S1c (parallel recording): per thread: the staging is begun and ended by the thread recording that one direct draw (a worker never does: the dry pass keeps these
// programs' direct draws serial), so another thread's eden_ssbo never sees it and redirects its own binding
static _Thread_local struct {
    VkBuffer guest, local;
    VkDeviceSize offset, size;
    int bound;
} n1x_atomic;

static void atomic_local_barrier(VkCommandBuffer cb, VkBuffer local, VkDeviceSize size,
                                 VkPipelineStageFlags src, VkAccessFlags writes,
                                 VkPipelineStageFlags dst, VkAccessFlags access) {
    VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, NULL, writes, access,
                                     VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, local, 0, size};
    vkCmdPipelineBarrier(cb, src, dst, 0, 0, NULL, 1, &barrier, 0, NULL);
}

// S1b (parallel recording): the two fragment programs n1x_atomic_begin stages for (was inline there; the dry pass keeps their direct draws on the replay thread)
static int n1x_atomic_prog(const vkshader_t *fs) { return vk_is_n1x() && vk_eden_mode() && fs && (fs->key == 0x24e19d5959e8769dull || fs->key == 0x9a5e3335514e4cbbull); }
static void n1x_atomic_begin(void) {
    if (!n1x_atomic_prog(S.sh[1])) return;
    const p_bind_buf *slot = &S.ssbo[1][0];
    uint64_t a = slot->addr, n = slot->size;
    if (!a || !n || n > 256 * 1024 || (n & 3) || a % V.props.limits.minStorageBufferOffsetAlignment) return;
    vkpool_t *p = vk_pool_of(a, n);
    if (!p || !p->zero_copy || ((a - p->lo) % V.props.limits.minStorageBufferOffsetAlignment)) return;
    VkBuffer buf = vk_atomic_scratch(n);
    if (!buf) return;
    if (S.pass_open == 1) vk_end_pass();
    VkCommandBuffer cb = vk_cmd();
    vk_ubd_land(cb);
    // The imported pages can have other buffer views. Order all earlier uses
    // before transfer, without stalling unrelated later graphics work.
    VkMemoryBarrier prepare = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT,
                               VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &prepare, 0, NULL, 0, NULL);
    VkBufferCopy copy = {a - p->lo, 0, n};
    vkCmdCopyBuffer(cb, p->buf, buf, 1, &copy);
    atomic_local_barrier(cb, buf, n, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    n1x_atomic = (typeof(n1x_atomic)){p->buf, buf, copy.srcOffset, n, 0};
}
static void n1x_atomic_end(void) {
    if (!n1x_atomic.local) return;
    if (n1x_atomic.bound) {
        vk_end_pass();
        VkCommandBuffer cb = vk_cmd();
        atomic_local_barrier(cb, n1x_atomic.local, n1x_atomic.size,
                             VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferCopy copy = {0, n1x_atomic.offset, n1x_atomic.size};
        vkCmdCopyBuffer(cb, n1x_atomic.local, n1x_atomic.guest, 1, &copy);
        VkMemoryBarrier publish = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT,
                                   VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 1, &publish, 0, NULL, 0, NULL);
    }
    memset(&n1x_atomic, 0, sizeof n1x_atomic);
}
// wtrace (NXN1X_VSRES_DIAG, log only): the storage of the watched vertex shader's texel buffers (vsres_diag records them), and every written storage buffer or storage texel
// buffer of a later draw or dispatch that overlaps one (the first 64): which GPU work, if any, fills Hyrule Warriors' empty per-instance buffer (texbuf 0x28)?
static uint64_t g_wt_lo[4], g_wt_hi[4]; static int g_nwt, g_wt_logged;
static void wt_note(uint64_t lo, uint64_t n) {
    for (int i = 0; i < g_nwt; i++) if (g_wt_lo[i] == lo) return;
    if (g_nwt < 4 && lo && n) { g_wt_lo[g_nwt] = lo; g_wt_hi[g_nwt] = lo + n; g_nwt++; VKLOG("wtrace: watching writes to %#llx +%#llx", (unsigned long long)lo, (unsigned long long)n); }
}
static void wt_check(const char *what, int st, uint64_t a, uint64_t n) {
    if (!g_nwt || !a || g_wt_logged >= 64) return;
    if (!n) n = 1;
    for (int i = 0; i < g_nwt; i++) if (a < g_wt_hi[i] && a + n > g_wt_lo[i]) {
        extern uint64_t g_tex_frame; g_wt_logged++;
        VKLOG("wtrace: %s at stage %d writes %#llx +%#llx over watched %#llx (present %llu, %s %016llx, fragment %016llx)", what, st, (unsigned long long)a, (unsigned long long)n,
              (unsigned long long)g_wt_lo[i], (unsigned long long)g_tex_frame, g_eden_cs_key ? "dispatch" : "vertex", (unsigned long long)(g_eden_cs_key ? g_eden_cs_key : S.sh[0] ? S.sh[0]->key : 0),
              (unsigned long long)(!g_eden_cs_key && S.sh[1] ? S.sh[1]->key : 0));
        return;
    }
}
// atomdiag (NXN1X_ATOMDIAG=1, log only, default off): written storage buffers bound from the memory pools, by shader. On N1X atomics into imported
// pool memory are ~25x slower than into device-local memory (membench5, 11-20) and the shader dumps show compute shaders with storage atomics in
// HWAoI, Bayonetta 3 and others; this counts, per present, the binds of written storage buffers by the shader that binds them, whether the pool
// chunk is imported (zero-copy) or a driver copy, and the bytes bound; every 600 presents the 12 busiest shaders are logged. A written storage buffer
// is where a shader's atomics can land: a shader in the list with atomics in its dump (scratchpad atomscan.py) is a candidate for device-local storage
typedef struct { uint64_t key; int st; uint64_t binds, imported, bytes; } atom_e;
static atom_e g_atom[512]; static int g_natom, g_atom_on = -1; static uint64_t g_atom_p0;
static void atomdiag_note(int st, uint64_t key, const vkpool_t *p, uint64_t n) {
    if (g_atom_on < 0) { const char *e = getenv("NXN1X_ATOMDIAG"); g_atom_on = e && atoi(e) != 0; }
    if (!g_atom_on) return;
    extern uint64_t g_tex_frame;
    if (!g_atom_p0) g_atom_p0 = g_tex_frame + 1;
    atom_e *a = NULL; for (int i = 0; i < g_natom; i++) if (g_atom[i].key == key && g_atom[i].st == st) { a = &g_atom[i]; break; }
    if (!a && g_natom < 512) { a = &g_atom[g_natom++]; memset(a, 0, sizeof *a); a->key = key; a->st = st; }
    if (a) { a->binds++; a->imported += p && p->zero_copy; a->bytes += n; }
    if (g_tex_frame >= g_atom_p0 + 600) {
        const double f = (double)(g_tex_frame - g_atom_p0 + 1); g_atom_p0 = g_tex_frame + 1;
        for (int i = 0; i < g_natom; i++) for (int j = i + 1; j < g_natom; j++) if (g_atom[j].binds > g_atom[i].binds) { atom_e t = g_atom[i]; g_atom[i] = g_atom[j]; g_atom[j] = t; }
        VKLOG("atomdiag: written storage buffers bound from the pools, per present over %.0f presents, %d shaders (stage: 0 vertex, 1 fragment, 2 geometry, 3/4 tessellation, 5 compute):", f, g_natom);
        for (int i = 0; i < g_natom && i < 12; i++) VKLOG("atomdiag:   stage %d shader %016llx: %.1f binds, %.0f%% from imported chunks, %.2f MB bound", g_atom[i].st, (unsigned long long)g_atom[i].key,
                                                           g_atom[i].binds / f, g_atom[i].binds ? 100.0 * g_atom[i].imported / g_atom[i].binds : 0.0, g_atom[i].bytes / f / 1048576.0);
        g_natom = 0; }
}
// ---- stage1 / stage2s (hybrid3, design\ATOMIC_STAGING_DESIGN.md 2-3 and 8, NPARK 07:54 / 08:12): atomic storage buffers of dispatches staged in device-local memory ----
// Cause: on N1X atomics on imported game pages are 17-97x slower than on device-local memory (membench7, also with the hardware float add); OoT's heaviest
// dispatch bdf51d78 ran atomics on a storage buffer in an imported chunk: 3.1 us a group, 0.4 us staged (stage1, 08:09). Converting chunks to device-local copies
// (hybrid1/2) cost the CPU write tracking and merges. Here nothing changes on the CPU side: each written storage buffer a dispatch's shader runs an atomic on
// (is_atomic), in an imported chunk, up to DM_VK_HY3_MB (16), is copied on the GPU into staging S and a snapshot P; the dispatch runs on S, and a merge dispatch
// (shaders/hy3_merge.comp) writes back into the game's pages only the bytes where S != P (a byte the CPU wrote meanwhile stays).
// stage2s: a window. The staged ranges stay in S across the dispatches that follow in the same command buffer (OoT: about 23 staged dispatches a frame, each with
// its copy barrier, merge and merge barrier, 46 more serialization points): a later dispatch's storage buffer inside a staged range is bound to S with no copy,
// a new atomic one is copied in and joins the window. The window is merged and closed (hy3_close) before any other GPU work in that command buffer: every s63
// wrapper of a command that does GPU work (copies, fills, updates, clears, blits, resolves, render pass / rendering begin, draws, queries begun, transform feedback,
// barriers with buffer or image barriers, a dispatch that is not the prepared one, the command buffer's end; the mesh draw at vk_bar_flush) calls vk_hy3_close_for
// first; timestamps, query resets and memory barriers do not close it (they touch no game memory). A dispatch that cannot use the window (a binding across a staged
// range's edge, the raw pool table, a texel buffer over a staged range, indirect arguments in one) closes it before it and binds everything to the pages.
// Counter reports and events are seen by the CPU at the submission's completion, after the window's close at the command buffer's end. DM_VK_HY3: 0 off,
// 1 a window per dispatch (stage1), 2 windows across dispatches (default); DM_VK_HY3_MB: the largest binding staged (default 16)
#include "hy3_merge_comp.spv.h"
VkBuffer vk_hy3_mem(VkDeviceSize n, VkDeviceSize al, VkDeviceSize *off);
enum { HY3_MAXS = 32, HY3_MAXB = 128, HY3_MAXR = 128 };
static const VkDeviceSize HY3_WINDOW = 64ull << 20;   // S + P of one window at most (a binding that would pass it stays on the pages)
typedef struct { uint64_t lo, hi; VkBuffer ib; VkDeviceSize io, len; VkBuffer sb, pb; VkDeviceSize so, po; } hy3_st;
static struct { VkCommandBuffer cb; int ns; VkDeviceSize bytes; hy3_st s[HY3_MAXS]; } g_hy3w;   // the open window (replay thread)
static struct {   // the dispatch being prepared
    int nstart, abort, need_close, nb, nr, held;
    struct { uint64_t lo, hi; } b[HY3_MAXB];                                       // bound to the pages (a later atomic binding over them stays there)
    struct { eden_entry *e; VkDescriptorBufferInfo bi; int dbuf; } r[HY3_MAXR];    // entries bound to S (back to the pages if the dispatch cannot use the window)
} g_hy3;
static int g_hy3_hold;   // > 0: the wrappers do not close the window (the staging's own commands, a staged dispatch's own recording)
static uint64_t g_hy3_disp, g_hy3_bufs, g_hy3_bytes, g_hy3_big, g_hy3_alias, g_hy3_cancel, g_hy3_nomem, g_hy3_redir, g_hy3_windows, g_hy3_merges, g_hy3_why[4];   // why: 0 dispatch, 1 other GPU work, 2 phase 1, 3 test
static struct { int state; VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline pipe; } g_hy3m;
static int hy3_mode(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_HY3"); v = e && *e ? atoi(e) : 2; if (v < 0 || v > 2) v = 2;
    VKLOG("stage2s: atomic storage buffers of dispatches (imported, up to the limit) %s (DM_VK_HY3=%d)", v == 2 ? "staged in device-local memory, in windows across the dispatches in a row" :
          v == 1 ? "staged in device-local memory, merged after each dispatch (stage1)" : "run on the imported pages (as before)", v); } return v; }
static int hy3_on(void) { return hy3_mode() != 0; }
static VkDeviceSize hy3_limit(void) { static VkDeviceSize v; if (!v) { const char *e = getenv("DM_VK_HY3_MB"); int mb = e && *e ? atoi(e) : 16; if (mb < 1) mb = 1; if (mb > 32) mb = 32; v = (VkDeviceSize)mb << 20; } return v; }
static int hy3_ready(void) {   // the merge pipeline (as ixq_ready)
    if (g_hy3m.state) return g_hy3m.state > 0;
    g_hy3m.state = -1;
    VkDescriptorSetLayoutBinding b[3]; for (int i = 0; i < 3; i++) b[i] = (VkDescriptorSetLayoutBinding){(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 3, b};
    if (vkCreateDescriptorSetLayout(V.dev, &li, NULL, &g_hy3m.dsl)) return 0;
    void vk_dsl_register(VkDescriptorSetLayout l, const VkDescriptorSetLayoutBinding *b, uint32_t n); vk_dsl_register(g_hy3m.dsl, b, 3);
    VkPipelineLayoutCreateInfo pi = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_hy3m.dsl, 0, NULL};
    if (vkCreatePipelineLayout(V.dev, &pi, NULL, &g_hy3m.pl)) return 0;
    VkShaderModule m; VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof hy3_merge_comp_spv, hy3_merge_comp_spv};
    if (vkCreateShaderModule(V.dev, &mi, NULL, &m)) return 0;
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", NULL}, g_hy3m.pl, VK_NULL_HANDLE, -1};
    VkResult r = vkCreateComputePipelines(V.dev, V.pcache, 1, &ci, NULL, &g_hy3m.pipe); vkDestroyShaderModule(V.dev, m, NULL);
    if (r) { VKRED_ONCE(170, "stage1: merge pipeline not made (%d): atomic storage buffers stay on the imported pages", r); return 0; }
    g_hy3m.state = 1; return 1;
}
static int hy3_overlap(uint64_t lo, uint64_t hi) { for (int i = 0; i < g_hy3w.ns; i++) if (lo < g_hy3w.s[i].hi && g_hy3w.s[i].lo < hi) return 1; return 0; }
// eden_dispatch_prep: a dispatch's bindings begin (ind: its indirect arguments' address, 0 for none: in the window, the window closes before it)
static void hy3_dispatch_begin(uint64_t ind) {
    if (g_hy3w.ns && g_hy3w.cb && g_hy3w.cb != vk_cmd()) {   // a window of another command buffer still open: its end closes it (the s63 wrapper), so this is not expected
        VKRED_ONCE(171, "stage2s: an atomic staging window of another command buffer still open at a dispatch (its merge was not recorded): dropped");
        g_hy3w.ns = 0; g_hy3w.bytes = 0; g_hy3w.cb = VK_NULL_HANDLE; }
    g_hy3.nstart = g_hy3w.ns; g_hy3.abort = g_hy3.need_close = 0; g_hy3.nb = g_hy3.nr = 0;
    if (ind && hy3_overlap(ind, ind + 12)) g_hy3.abort = g_hy3.need_close = 1;
}
static void hy3_cancel(void) {   // this dispatch cannot use the window: its entries on S back to the pages, its new ranges dropped, the window closed before it
    for (int i = 0; i < g_hy3.nr; i++) { eden_entry *e = g_hy3.r[i].e; if (!g_hy3.r[i].dbuf) e->buffer = g_hy3.r[i].bi; else eden_addr_of(e, &g_hy3.r[i].bi, VK_FORMAT_UNDEFINED); }
    if (g_hy3.nr || g_hy3w.ns > g_hy3.nstart) g_hy3_cancel++;
    for (int i = g_hy3.nstart; i < g_hy3w.ns; i++) g_hy3w.bytes -= 2 * g_hy3w.s[i].len;
    g_hy3w.ns = g_hy3.nstart; g_hy3.nr = 0; g_hy3.abort = 1; if (g_hy3w.ns) g_hy3.need_close = 1;
}
static void hy3_note(uint64_t lo, uint64_t hi) { if (g_hy3.nb < HY3_MAXB) { g_hy3.b[g_hy3.nb].lo = lo; g_hy3.b[g_hy3.nb].hi = hi; g_hy3.nb++; } else if (!g_hy3.abort) hy3_cancel(); }
// a binding over guest [lo, hi) the staging cannot rebind (a texel buffer; the raw pool table: lo = 0, hi = ~0): over a staged range, the dispatch cannot use the window
static void hy3_other(uint64_t lo, uint64_t hi) {
    if (!hy3_on()) return;
    if (hy3_overlap(lo, hi)) { hy3_cancel(); return; }
    hy3_note(lo, hi);
}
// eden_ssbo, a compute storage buffer bound from chunk p as *bi (offset relative to the chunk's buffer): 1 = bound to staging (*bi changed; the entry e recorded)
static int hy3_bind(const nvn_eden_bind *b, const vkpool_t *p, VkDescriptorBufferInfo *bi, eden_entry *e, int dbuf) {
    if (!hy3_on() || g_hy3.abort || !p || !bi->range) return 0;
    const uint64_t lo = p->lo + bi->offset, hi = lo + bi->range;
    for (int i = 0; i < g_hy3w.ns; i++) { const hy3_st *s = &g_hy3w.s[i];
        if (lo >= s->lo && hi <= s->hi) {   // inside a staged range: the same bytes in S
            if (g_hy3.nr == HY3_MAXR) { hy3_cancel(); return 0; }
            g_hy3.r[g_hy3.nr].e = e; g_hy3.r[g_hy3.nr].bi = *bi; g_hy3.r[g_hy3.nr].dbuf = dbuf; g_hy3.nr++;
            bi->buffer = s->sb; bi->offset = s->so + (lo - s->lo); g_hy3_redir++; return 1; }
        if (lo < s->hi && s->lo < hi) { hy3_cancel(); return 0; } }   // across a staged range's edge
    if (!(b->is_written && b->is_atomic) || !p->zero_copy) { hy3_note(lo, hi); return 0; }
    if (bi->range > hy3_limit()) { g_hy3_big++; hy3_note(lo, hi); return 0; }
    for (int i = 0; i < g_hy3.nb; i++) if (lo < g_hy3.b[i].hi && g_hy3.b[i].lo < hi) { g_hy3_alias++; hy3_note(lo, hi); return 0; }   // bound to the pages already
    const VkDeviceSize len = (bi->range + 3) & ~(VkDeviceSize)3;
    if (bi->offset + len > p->hi - p->lo || g_hy3w.ns == HY3_MAXS || g_hy3.nr == HY3_MAXR || g_hy3w.bytes + 2 * len > HY3_WINDOW || !hy3_ready()) { hy3_note(lo, hi); return 0; }
    const VkDeviceSize al = V.props.limits.minStorageBufferOffsetAlignment ? V.props.limits.minStorageBufferOffsetAlignment : 256; VkDeviceSize so, po;
    VkBuffer sb = vk_hy3_mem(len, al, &so), pb = sb ? vk_hy3_mem(len, al, &po) : VK_NULL_HANDLE;
    if (!sb || !pb) { g_hy3_nomem++; hy3_note(lo, hi); return 0; }
    g_hy3w.s[g_hy3w.ns++] = (hy3_st){lo, lo + len, bi->buffer, bi->offset, len, sb, pb, so, po}; g_hy3w.bytes += 2 * len;
    g_hy3.r[g_hy3.nr].e = e; g_hy3.r[g_hy3.nr].bi = *bi; g_hy3.r[g_hy3.nr].dbuf = dbuf; g_hy3.nr++;
    bi->buffer = sb; bi->offset = so; g_hy3_bufs++; g_hy3_bytes += len; return 1;
}
static void hy3_close(VkCommandBuffer cb, int why) {   // the window's changed bytes back into the game's pages, then a barrier before everything after
    if (!g_hy3w.ns) return;
    VkDescriptorSet vk_alloc_set(VkDescriptorSetLayout l); VkAccessFlags vk_dev_access(void);
    g_hy3_hold++;
    VkMemoryBarrier m0 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};   // S's writers before the merge
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &m0, 0, NULL, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_hy3m.pipe);
    for (int i = 0; i < g_hy3w.ns; i++) { const hy3_st *s = &g_hy3w.s[i];
        VkDescriptorSet set = vk_alloc_set(g_hy3m.dsl);
        VkDescriptorBufferInfo bi[3] = {{s->sb, s->so, s->len}, {s->pb, s->po, s->len}, {s->ib, s->io, s->len}}; VkWriteDescriptorSet w[3];
        for (int k = 0; k < 3; k++) w[k] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, (uint32_t)k, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi[k], NULL};
        vkUpdateDescriptorSets(V.dev, 3, w, 0, NULL);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_hy3m.pl, 0, 1, &set, 0, NULL);
        vkCmdDispatch(cb, (uint32_t)((s->len / 4 + 255) / 256), 1, 1); }
    VkMemoryBarrier m = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, vk_dev_access()};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &m, 0, NULL, 0, NULL);
    g_hy3_hold--;
    g_hy3w.ns = 0; g_hy3w.bytes = 0; g_hy3w.cb = VK_NULL_HANDLE; g_hy3_merges++; if (why >= 0 && why < 4) g_hy3_why[why]++;
    uint64_t n = g_hy3_merges; if (n <= 2 || !(n & (n - 1))) VKLOG("stage2s: %llu windows closed (%llu opened; closed before a dispatch that could not use them %llu, before other GPU work %llu, after each dispatch (DM_VK_HY3=1) %llu); "
        "%llu dispatches staged, %llu buffers (%.1f MB), %llu bindings on S beside them; not staged: over %llu MB %llu, over a binding bound before %llu, cancelled %llu, no memory %llu",
        (unsigned long long)n, (unsigned long long)g_hy3_windows, (unsigned long long)g_hy3_why[0], (unsigned long long)g_hy3_why[1], (unsigned long long)g_hy3_why[2], (unsigned long long)g_hy3_disp,
        (unsigned long long)g_hy3_bufs, g_hy3_bytes / 1048576.0, (unsigned long long)g_hy3_redir, (unsigned long long)(hy3_limit() >> 20), (unsigned long long)g_hy3_big, (unsigned long long)g_hy3_alias,
        (unsigned long long)g_hy3_cancel, (unsigned long long)g_hy3_nomem);
}
// gfx_vk_core.c's s63 wrappers, before a command that does GPU work in cb (and vk_bar_flush): the open window closed first
// S1c (parallel recording): the window is the replay thread's (dispatches stay serial): another recording thread's commands never close it, nor read it
void vk_hy3_close_for(VkCommandBuffer cb) { if (g_rec_e != &g_rec_e_main) return; if (!g_hy3_hold && g_hy3w.ns && cb == g_hy3w.cb) hy3_close(cb, 1); }
// dispatch (gfx_vk_exec.c), right after its preparation: the window closed if this dispatch cannot use it, its new ranges copied in, and the wrappers held off
// until vk_hy3_post (the dispatch's own commands: first stores, r75's argument copy, its barriers, the dispatch)
void vk_hy3_pre(VkCommandBuffer cb) {
    if (!hy3_on()) return;
    // (stage2s1, 08:21 crash: this line closed "a window left in another command buffer" when the window held only this dispatch's new ranges, whose window has no
    // command buffer yet: hy3_close recorded into a null command buffer. A window of another command buffer is closed when that buffer ends, and checked at
    // hy3_dispatch_begin)
    if (g_hy3.need_close) { const int keep = g_hy3w.ns; hy3_close(cb, 0); (void)keep; }
    if (g_hy3w.ns > g_hy3.nstart) {   // new ranges: copied from the game's pages into S and P
        g_hy3_hold++;
        VkMemoryBarrier m = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &m, 0, NULL, 0, NULL);
        for (int i = g_hy3.nstart; i < g_hy3w.ns; i++) { const hy3_st *s = &g_hy3w.s[i];
            VkBufferCopy c = {s->io, s->so, s->len}; vkCmdCopyBuffer(cb, s->ib, s->sb, 1, &c); c.dstOffset = s->po; vkCmdCopyBuffer(cb, s->ib, s->pb, 1, &c); }
        g_hy3_hold--;
        if (!g_hy3.nstart) { g_hy3_windows++; g_hy3w.cb = cb; } }
    if (g_hy3w.ns) { g_hy3_hold++; g_hy3.held = 1; g_hy3_disp++; }
}
// dispatch, after its barrier after it (and on its early returns after vk_hy3_pre): the wrappers may close the window again; DM_VK_HY3=1 closes it here
void vk_hy3_post(VkCommandBuffer cb) {
    if (g_hy3.held) { g_hy3_hold--; g_hy3.held = 0; }
    if (hy3_mode() == 1) hy3_close(cb, 2);
}
// eden_first_stores: a store into a staged range goes into S (it is part of the dispatch's work on it)
int vk_hy3_redirect(uint64_t a, uint64_t n, VkBuffer *b, VkDeviceSize *o) {
    for (int i = 0; i < g_hy3w.ns; i++) if (a >= g_hy3w.s[i].lo && a + n <= g_hy3w.s[i].hi) { *b = g_hy3w.s[i].sb; *o = g_hy3w.s[i].so + (a - g_hy3w.s[i].lo); return 1; }
    return 0;
}
// ---- fstage (DK Bananza 11-22): a fragment shader's written storage buffers in imported pool chunks staged in device-local memory for its draw ----
// Cause (runs 11-22 01:46 / 02:13): DK's title runs 20 fps on 82 draws because one full-screen draw into a 1440x810 R32F target that binds a storage buffer takes
// 38-40 ms of GPU; with every pool a device-local copy (NXN1X_POOL_MODE=copy) it took 0.16-0.26 ms and the title ran 60 fps (the same draw costs 6-23 ms a frame in
// play). Bayonetta 3's depth pass (fragment stores into imported memory, 40-57 ms -> 0.6 ms in copy mode) is the same. Copy mode itself costs the CPU too much in
// play (DK 8 fps: ~500 MB read back and compared per submission), and n1x_atomic above stages only two TotK programs by key. Here, for any fragment shader:
//  - learning (eden_ssbo, replay thread): a written storage buffer of stage 1 bound from an imported chunk adds its Eden bind (cbuf, offset, index, indirect offset)
//    to the shader object (up to 4);
//  - each direct draw of such a shader (the replay thread, outside any pass): each learned buffer's guest range is copied on the GPU into S and a snapshot P
//    (vk_hy3_mem), the draw's binding of exactly that range goes to S (eden_ssbo), and after the draw hy3's merge (shaders/hy3_merge.comp) writes into the game's
//    pages only the bytes where S != P (a byte the CPU wrote meanwhile stays), as stage1 does for dispatches.
// Checks while the game runs, each falling back to the imported pages with the reason counted and logged: not imported, over DM_VK_FSTAGE_MB (16), alignment, out of
// the chunk, overlapping learned ranges, no staging memory, merge pipeline missing; a shader whose staged draws pass 16 in one present (a pass broken per draw)
// or whose staged range was not bound by its draw 8 times is turned off for the run (logged). DM_VK_FSTAGE=0: off (as before). Log lines "fstage:"
static int fst_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_FSTAGE"); v = !(e && *e == '0') && vk_is_n1x();
    VKLOG("fstage: written storage buffers of fragment shaders in imported pool chunks %s (DM_VK_FSTAGE=%s)", v ? "staged in device-local memory for each direct draw" : "used on the imported pages (as before)", e && *e ? e : "unset"); } return v; }
static VkDeviceSize fst_limit(void) { static VkDeviceSize v; if (!v) { const char *e = getenv("DM_VK_FSTAGE_MB"); int mb = e && *e ? atoi(e) : 16; if (mb < 1) mb = 1; if (mb > 64) mb = 64; v = (VkDeviceSize)mb << 20; } return v; }
static int fst_prog(const vkshader_t *fs) { return fs && vk_eden_mode() && __atomic_load_n(&fs->fst_state, __ATOMIC_ACQUIRE) == 1 && fst_on(); }
typedef struct { VkBuffer gb; VkDeviceSize go, grange, len; VkBuffer sb, pb; VkDeviceSize so, po; int bound; } fst_e;
static _Thread_local struct { int n; vkshader_t *fs; uint64_t draws0; fst_e e[4]; } g_fst;   // draws0: g_vkst.draws at the staging (a skipped draw binds nothing and is not held against the shader)
enum { FSR_NOTIMP, FSR_BIG, FSR_ALIGN, FSR_CHUNK, FSR_OVERLAP, FSR_NOMEM, FSR_NOPIPE, FSR_NOADDR, FSR_N };
static uint64_t g_fst_draws, g_fst_bytes, g_fst_merges, g_fst_unused, g_fst_why[FSR_N], g_fst_p0, g_fst_learned, g_fst_off;
static void fst_report(void) {
    extern uint64_t g_tex_frame;
    if (!g_fst_p0) { g_fst_p0 = g_tex_frame + 1; return; }
    if (g_tex_frame < g_fst_p0 + 600) return;
    const double f = (double)(g_tex_frame - g_fst_p0 + 1); g_fst_p0 = g_tex_frame + 1;
    VKLOG("fstage: last %.0f presents, per present: staged draws %.2f, %.2f MB copied in (S and P), merges %.2f, staged but not bound by the draw %.2f; refused (on the pages): not imported %.2f, "
          "over %llu MB %.2f, alignment %.2f, past the chunk %.2f, overlapping %.2f, no memory %.2f, no merge pipeline %.2f, no address %.2f; shaders learned %llu, turned off %llu",
          f, g_fst_draws / f, g_fst_bytes / f / 1048576.0, g_fst_merges / f, g_fst_unused / f, g_fst_why[FSR_NOTIMP] / f, (unsigned long long)(fst_limit() >> 20), g_fst_why[FSR_BIG] / f,
          g_fst_why[FSR_ALIGN] / f, g_fst_why[FSR_CHUNK] / f, g_fst_why[FSR_OVERLAP] / f, g_fst_why[FSR_NOMEM] / f, g_fst_why[FSR_NOPIPE] / f, g_fst_why[FSR_NOADDR] / f,
          (unsigned long long)g_fst_learned, (unsigned long long)g_fst_off);
    g_fst_draws = g_fst_bytes = g_fst_merges = g_fst_unused = 0; memset(g_fst_why, 0, sizeof g_fst_why);
}
static void fst_turn_off(vkshader_t *fs, const char *why) {
    if (__atomic_exchange_n(&fs->fst_state, 2, __ATOMIC_ACQ_REL) != 2) { g_fst_off++; VKLOG("fstage: fs %016llx turned off for the run: %s", (unsigned long long)fs->key, why); }
}
// eden_ssbo (stage 1, a written binding from an imported chunk, the replay thread): the bind joins the shader's learned set
static void fst_learn(vkshader_t *fs, const nvn_eden_bind *b, uint32_t k, uint64_t a, uint64_t n) {
    if (!fs || !fst_on() || b->cbuf >= 0xFE) return;
    const int st = __atomic_load_n(&fs->fst_state, __ATOMIC_ACQUIRE); if (st == 2) return;
    for (int i = 0; i < fs->fst_n; i++) if (fs->fst_b[i].cbuf == b->cbuf && fs->fst_b[i].off == b->offset && fs->fst_b[i].k == k && fs->fst_b[i].ind == b->indirect_offset) return;
    if (fs->fst_n == 4) { fst_turn_off(fs, "more than 4 written storage buffers in imported chunks"); return; }
    fs->fst_b[fs->fst_n].cbuf = b->cbuf; fs->fst_b[fs->fst_n].off = b->offset; fs->fst_b[fs->fst_n].k = k; fs->fst_b[fs->fst_n].ind = b->indirect_offset; fs->fst_b[fs->fst_n].atomic = b->is_atomic;
    fs->fst_n++;
    if (!st) g_fst_learned++;
    VKLOG("fstage: fs %016llx writes storage buffer %d (cbuf %u +%#x [%u]%s%s) in an imported chunk: %#llx +%#llx; staged from its next direct draw", (unsigned long long)fs->key, fs->fst_n - 1,
          b->cbuf, b->offset, k, b->indirect_offset != ~0u ? ", address read from memory" : "", b->is_atomic ? ", atomic" : "", (unsigned long long)a, (unsigned long long)n);
    __atomic_store_n(&fs->fst_state, 1, __ATOMIC_RELEASE);
}
// a learned bind's guest range, as eden_ssbo finds it (its address words, an indirect pointer)
static int fst_addr(const vkshader_t *fs, int i, uint64_t *a, uint64_t *n) {
    const uint32_t off = fs->fst_b[i].off + 16 * fs->fst_b[i].k, cb = fs->fst_b[i].cbuf;
    *a = (uint64_t)eden_cbuf_word(1, cb, off) | (uint64_t)eden_cbuf_word(1, cb, off + 4) << 32; *n = eden_cbuf_word(1, cb, off + 8);
    if (fs->fst_b[i].ind != ~0u && dm_sdk_ounce()) {
        const uint64_t table = *a; *a = 0; *n = 8ull << 20;   // (a written pointed-to buffer: 8 MiB, as eden_ssbo)
        if (table && vk_pool_of(table + fs->fst_b[i].ind, 8)) *a = *(const volatile uint64_t *)(uintptr_t)(table + fs->fst_b[i].ind);
    }
    return *a != 0;
}
// a direct draw (gfx_vk_exec.c OP_DRAW / OP_DRAW_IDX, before it): the learned ranges copied into S and P, outside any pass
static void fst_begin(void) {
    if (g_wk || n1x_atomic_prog(S.sh[1]) || !fst_prog(S.sh[1])) return;
    vkshader_t *fs = S.sh[1]; extern uint64_t g_tex_frame;
    fst_report();
    if (fs->fst_frame != g_tex_frame) { fs->fst_frame = g_tex_frame; fs->fst_in_frame = 0; }
    if (++fs->fst_in_frame > 16) { fst_turn_off(fs, "more than 16 staged draws in one present (each would end its pass)"); return; }
    if (!hy3_ready()) { g_fst_why[FSR_NOPIPE]++; return; }
    const VkDeviceSize al = V.props.limits.minStorageBufferOffsetAlignment ? V.props.limits.minStorageBufferOffsetAlignment : 16;
    fst_e e[4]; int ne = 0;
    for (int i = 0; i < fs->fst_n && i < 4; i++) {
        uint64_t a, n; if (!fst_addr(fs, i, &a, &n)) { g_fst_why[FSR_NOADDR]++; continue; }
        vkpool_t *p = (n ? vk_pool_of(a, n) : NULL) ?: vk_pool_of(a, 1);
        if (!p) { g_fst_why[FSR_NOADDR]++; continue; }
        if (!p->zero_copy) { g_fst_why[FSR_NOTIMP]++; continue; }
        const VkDeviceSize o = a - p->lo, cs = p->hi - p->lo; if (!n || o + n > cs) n = cs - o;
        const VkDeviceSize go = o - o % al, grange = n + o % al, len = (grange + 3) & ~(VkDeviceSize)3;
        if (grange > fst_limit()) { g_fst_why[FSR_BIG]++; continue; }
        if (go % al) { g_fst_why[FSR_ALIGN]++; continue; }
        if (go + len > cs) { g_fst_why[FSR_CHUNK]++; continue; }
        int ov = 0; for (int j = 0; j < ne; j++) if (e[j].gb == p->buf && go < e[j].go + e[j].len && e[j].go < go + len) ov = 1;
        if (ov) { g_fst_why[FSR_OVERLAP]++; ne = 0; break; }   // (two learned bindings over the same bytes: the draw on the pages)
        e[ne] = (fst_e){p->buf, go, grange, len, VK_NULL_HANDLE, VK_NULL_HANDLE, 0, 0, 0}; ne++;
    }
    if (!ne) return;
    for (int i = 0; i < ne; i++) {
        e[i].sb = vk_hy3_mem(e[i].len, al, &e[i].so); e[i].pb = e[i].sb ? vk_hy3_mem(e[i].len, al, &e[i].po) : VK_NULL_HANDLE;
        if (!e[i].sb || !e[i].pb) { g_fst_why[FSR_NOMEM]++; return; } }
    if (S.pass_open == 1) vk_end_pass();
    VkCommandBuffer cb = vk_cmd();
    vk_ubd_land(cb);
    vk_hy3_close_for(cb);   // (an open dispatch window closed first: its merge binds its own pipeline and sets)
    g_hy3_hold++;
    VkMemoryBarrier m = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &m, 0, NULL, 0, NULL);
    for (int i = 0; i < ne; i++) { VkBufferCopy c = {e[i].go, e[i].so, e[i].len}; vkCmdCopyBuffer(cb, e[i].gb, e[i].sb, 1, &c); c.dstOffset = e[i].po; vkCmdCopyBuffer(cb, e[i].gb, e[i].pb, 1, &c);
        g_fst_bytes += 2 * e[i].len; }
    VkMemoryBarrier m2 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &m2, 0, NULL, 0, NULL);
    g_hy3_hold--;
    g_fst.n = ne; g_fst.fs = fs; g_fst.draws0 = g_vkst.draws; memcpy(g_fst.e, e, sizeof e[0] * (size_t)ne); g_fst_draws++;
}
// eden_ssbo (stage 1, written): the staged range's binding goes to S
static int fst_redirect(VkDescriptorBufferInfo *bi) {
    for (int i = 0; i < g_fst.n; i++) { fst_e *x = &g_fst.e[i];
        if (bi->buffer == x->gb && bi->offset == x->go && bi->range == x->grange) { bi->buffer = x->sb; bi->offset = x->so; x->bound = 1; return 1; } }
    return 0;
}
// after the draw: the changed bytes back into the game's pages (hy3's merge), outside the pass
static void fst_end(void) {
    if (!g_fst.n) return;
    int nb = 0; for (int i = 0; i < g_fst.n; i++) nb += g_fst.e[i].bound;
    if (nb) {
        vk_end_pass();
        VkCommandBuffer cb = vk_cmd(); VkDescriptorSet vk_alloc_set(VkDescriptorSetLayout l); VkAccessFlags vk_dev_access(void);
        vk_hy3_close_for(cb);
        g_hy3_hold++;
        VkMemoryBarrier m0 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &m0, 0, NULL, 0, NULL);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_hy3m.pipe);
        for (int i = 0; i < g_fst.n; i++) { const fst_e *x = &g_fst.e[i]; if (!x->bound) continue;
            VkDescriptorSet set = vk_alloc_set(g_hy3m.dsl);
            VkDescriptorBufferInfo bi[3] = {{x->sb, x->so, x->len}, {x->pb, x->po, x->len}, {x->gb, x->go, x->len}}; VkWriteDescriptorSet w[3];
            for (int k = 0; k < 3; k++) w[k] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, (uint32_t)k, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi[k], NULL};
            vkUpdateDescriptorSets(V.dev, 3, w, 0, NULL);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_hy3m.pl, 0, 1, &set, 0, NULL);
            vkCmdDispatch(cb, (uint32_t)((x->len / 4 + 255) / 256), 1, 1); g_fst_merges++; }
        VkMemoryBarrier m = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT, vk_dev_access()};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &m, 0, NULL, 0, NULL);
        g_hy3_hold--;
    }
    if (nb < g_fst.n && g_vkst.draws != g_fst.draws0) { g_fst_unused += (uint64_t)(g_fst.n - nb);   // staged, the draw recorded, but it bound another range (or none): nothing to merge for it
        vkshader_t *fs = g_fst.fs; if (fs && ++fs->fst_unused >= 8) fst_turn_off(fs, "its staged range was not the one its draw bound, 8 times"); }
    g_fst.n = 0; g_fst.fs = NULL;
}
// ---- self-test (gfx_vk_core.c): the binding rules (cases 190-), the GPU path (198-199) and the windows (200-) ----
static void hy3_test_clear(void) { g_hy3w.ns = 0; g_hy3w.bytes = 0; g_hy3w.cb = VK_NULL_HANDLE; memset(&g_hy3, 0, sizeof g_hy3); }
int vk_hy3_case(int k, const char **what, uint64_t chunk_lo, uint64_t chunk_n, int zero_copy) {
    if (!hy3_on()) { *what = "not run (DM_VK_HY3=0)"; return k == 0 ? 1 : -1; }
    vkpool_t p; memset(&p, 0, sizeof p); p.lo = chunk_lo; p.hi = chunk_lo + chunk_n; p.zero_copy = zero_copy; p.buf = VK_NULL_HANDLE;
    nvn_eden_bind ba; memset(&ba, 0, sizeof ba); ba.is_written = 1; ba.is_atomic = 1; nvn_eden_bind bs = ba; bs.is_atomic = 0; nvn_eden_bind br = bs; br.is_written = 0;
    static eden_entry e[4]; memset(e, 0, sizeof e); VkDescriptorBufferInfo x, y; int ok = 0; hy3_test_clear(); hy3_dispatch_begin(0);
    switch (k) {
    case 0: *what = "an atomic storage buffer in an imported chunk is bound to staging"; x = (VkDescriptorBufferInfo){p.buf, 0x1000, 0x2000};
        ok = hy3_bind(&ba, &p, &x, &e[0], 0) == 1 && x.buffer != VK_NULL_HANDLE && g_hy3w.ns == 1 && g_hy3w.s[0].lo == chunk_lo + 0x1000 && g_hy3w.s[0].len == 0x2000; break;
    case 1: *what = "a storage buffer inside a staged range follows it to staging, at the same offset"; x = (VkDescriptorBufferInfo){p.buf, 0x1000, 0x2000}; y = (VkDescriptorBufferInfo){p.buf, 0x1100, 0x100};
        ok = hy3_bind(&ba, &p, &x, &e[0], 0) && hy3_bind(&br, &p, &y, &e[1], 0) && y.buffer == x.buffer && y.offset == x.offset + 0x100; break;
    case 2: *what = "a storage buffer across a staged range's edge: the dispatch cannot use the window, every binding back on the pages"; x = (VkDescriptorBufferInfo){p.buf, 0x1000, 0x2000}; y = (VkDescriptorBufferInfo){p.buf, 0x2800, 0x1000};
        { VkDescriptorBufferInfo x0 = x; hy3_bind(&ba, &p, &x, &e[0], 0); e[0].buffer = x; hy3_bind(&bs, &p, &y, &e[1], 0);
          ok = g_hy3w.ns == 0 && g_hy3.abort && e[0].buffer.buffer == x0.buffer && e[0].buffer.offset == x0.offset; } break;
    case 3: *what = "an atomic storage buffer over one bound before stays on the pages"; y = (VkDescriptorBufferInfo){p.buf, 0x1100, 0x100}; x = (VkDescriptorBufferInfo){p.buf, 0x1000, 0x2000};
        ok = hy3_bind(&bs, &p, &y, &e[1], 0) == 0 && hy3_bind(&ba, &p, &x, &e[0], 0) == 0 && g_hy3w.ns == 0; break;
    case 4: *what = "the raw pool table bound beside a staged range: the dispatch cannot use the window"; x = (VkDescriptorBufferInfo){p.buf, 0x1000, 0x2000};
        hy3_bind(&ba, &p, &x, &e[0], 0); hy3_other(0, ~0ull); ok = g_hy3w.ns == 0 && g_hy3.abort; break;
    case 5: *what = "over the limit (16 MB): not staged; at the limit: staged"; x = (VkDescriptorBufferInfo){p.buf, 0, hy3_limit() + 0x1000}; y = (VkDescriptorBufferInfo){p.buf, 0, hy3_limit()};
        { const int over = hy3_bind(&ba, &p, &x, &e[0], 0) == 0; hy3_test_clear(); hy3_dispatch_begin(0);   // (each in a dispatch of its own: the first stays on the pages, and would keep the second there)
          ok = chunk_n >= hy3_limit() + 0x1000 && over && hy3_bind(&ba, &p, &y, &e[1], 0) == 1; } break;
    case 6: *what = "a device-local copy's chunk (copy or hybrid-converted): not staged"; p.zero_copy = 0; x = (VkDescriptorBufferInfo){p.buf, 0x1000, 0x2000};
        ok = hy3_bind(&ba, &p, &x, &e[0], 0) == 0; break;
    case 7: *what = "a written storage buffer with no atomic: not staged"; x = (VkDescriptorBufferInfo){p.buf, 0x1000, 0x2000}; ok = hy3_bind(&bs, &p, &x, &e[0], 0) == 0; break;
    case 8: *what = "the next dispatch: a binding inside the window's range goes to S with no new range"; x = (VkDescriptorBufferInfo){p.buf, 0x1000, 0x2000};
        hy3_bind(&ba, &p, &x, &e[0], 0); hy3_dispatch_begin(0); y = (VkDescriptorBufferInfo){p.buf, 0x1000, 0x2000};
        ok = hy3_bind(&ba, &p, &y, &e[1], 0) == 1 && g_hy3w.ns == 1 && y.buffer == x.buffer && y.offset == x.offset; break;
    case 9: *what = "indirect arguments in the window's range: the next dispatch closes it and stays on the pages"; x = (VkDescriptorBufferInfo){p.buf, 0x1000, 0x2000};
        hy3_bind(&ba, &p, &x, &e[0], 0); hy3_dispatch_begin(chunk_lo + 0x1100); y = (VkDescriptorBufferInfo){p.buf, 0x1000, 0x2000};
        ok = g_hy3.need_close && hy3_bind(&ba, &p, &y, &e[1], 0) == 0; break;
    case 10: *what = "a texel buffer over the window's range in the next dispatch: it closes the window, its new ranges dropped"; x = (VkDescriptorBufferInfo){p.buf, 0x1000, 0x2000};
        hy3_bind(&ba, &p, &x, &e[0], 0); hy3_dispatch_begin(0); y = (VkDescriptorBufferInfo){p.buf, 0x8000, 0x1000}; hy3_bind(&ba, &p, &y, &e[1], 0);
        hy3_other(chunk_lo + 0x1800, chunk_lo + 0x1900); ok = g_hy3.need_close && g_hy3w.ns == 1 && g_hy3.abort; break;
    default: hy3_test_clear(); return -1;
    }
    hy3_test_clear(); return ok;
}
// a "dispatch" for the GPU cases: [a, a + n) of an imported chunk bound as an atomic storage buffer (joins the open window, or opens one), then 16 bytes of v written
// at a + at into what it was bound to, as the dispatch would; the window left open (closed by what comes next, as in a game)
void vk_hy3_test_stage(VkCommandBuffer cb, uint64_t a, uint64_t n, uint64_t at, uint8_t v) {
    vkpool_t *p = vk_pool_of(a, n); if (!p) return;
    static eden_entry e0; nvn_eden_bind ba; memset(&ba, 0, sizeof ba); ba.is_written = 1; ba.is_atomic = 1;
    hy3_dispatch_begin(0); VkDescriptorBufferInfo x = {p->buf, a - p->lo, n}; hy3_bind(&ba, p, &x, &e0, 0); vk_hy3_pre(cb);
    VkMemoryBarrier m = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT};
    g_hy3_hold++;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &m, 0, NULL, 0, NULL);
    vkCmdFillBuffer(cb, x.buffer, x.offset + at, 16, 0x01010101u * v);
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &m, 0, NULL, 0, NULL);
    g_hy3_hold--;
    vk_hy3_post(cb);
}
// case 198 / 199 (the merge): byte_case 0: 16 bytes changed in S, the rest stay; 1: S changes byte 0 of a word, a write into the page (as by the CPU meanwhile)
// changes byte 1: both stay. Records into the open submission and closes the window
int vk_hy3_gpu_case(uint64_t a, uint64_t n, int byte_case) {
    if (!hy3_on() || !hy3_ready()) return -1;
    vkpool_t *p = vk_pool_of(a, n); if (!p || !p->zero_copy) return -1;
    hy3_test_clear(); vk_end_pass_land(); VkCommandBuffer cb = vk_cmd();
    if (!byte_case) vk_hy3_test_stage(cb, a, n, 0, 0x5a);
    else { vk_hy3_test_stage(cb, a, n, 0, 0x00);   // (opens the window; S's bytes 0-15 rewritten 0 below)
        if (!g_hy3w.ns) return 0;
        const hy3_st *s = &g_hy3w.s[0]; g_hy3_hold++;
        uint32_t w = 0x000000c3u; vkCmdUpdateBuffer(cb, s->sb, s->so, 4, &w);   // S: byte 0 = 0xc3
        uint32_t d = 0x0000d400u; vkCmdUpdateBuffer(cb, s->ib, s->io, 4, &d);   // the page: byte 1 = 0xd4, written after the copy-in
        VkMemoryBarrier m = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &m, 0, NULL, 0, NULL); g_hy3_hold--; }
    if (!g_hy3w.ns) return 0;
    hy3_close(cb, 3); return 1;
}
// cases 200-: windows across "dispatches" and their closes, on an imported chunk D and a destination C (import / hybrid self-test). 1 pass, 0 fail, -1 none
int vk_hy3_win_case(int k, const char **what, uint8_t *D, uint8_t *C) {
    enum { PG = 4096 }; const uint64_t a = (uint64_t)(uintptr_t)D; int ok = 0;
    void mtl_sub_begin(void); uint64_t mtl_sub_commit(void); int mtl_wait(uint64_t, uint64_t); void mtl_sub_copy(uint64_t, uint64_t, uint64_t); void mtl_sub_fill(uint64_t, uint64_t, uint32_t);
    void mtl_sub_counter(uint64_t, uint32_t); void dm_guest_write_prepare(void *, size_t);
    if (hy3_mode() != 2) { *what = "not run (DM_VK_HY3 is not 2)"; return k == 0 ? 1 : -1; }
    if (k > 5) return -1;
    hy3_test_clear(); dm_guest_write_prepare(D, 2 * PG); memset(D, 0, 2 * PG);
    const uint64_t w0 = g_hy3_windows, m0 = g_hy3_merges; uint64_t f;
    mtl_sub_begin(); VkCommandBuffer cb; vk_end_pass_land(); cb = vk_cmd();
    switch (k) {
    case 0: *what = "S3 two dispatches in a row on one range: one window, one merge, both writes in the game's page";
        vk_hy3_test_stage(cb, a, PG, 0, 0x31); vk_hy3_test_stage(cb, a, PG, 64, 0x32); f = mtl_sub_commit(); mtl_wait(f, 10000000000ull);
        ok = g_hy3_windows - w0 == 1 && g_hy3_merges - m0 == 1 && D[0] == 0x31 && D[15] == 0x31 && D[64] == 0x32 && D[79] == 0x32 && D[16] == 0; break;
    case 1: *what = "S4 a copy from the range after the window: closed before it, the copy reads the GPU's bytes";
        vk_hy3_test_stage(cb, a, PG, 0, 0x41); mtl_sub_copy(a, (uint64_t)(uintptr_t)C, 16); f = mtl_sub_commit(); mtl_wait(f, 10000000000ull);
        ok = g_hy3_merges - m0 == 1 && C[0] == 0x41 && C[15] == 0x41; break;
    case 2: *what = "S5 arguments written in the window read after it (a copy, as an indirect read of them): closed before, the merged bytes read";
        vk_hy3_test_stage(cb, a, PG, 128, 0x05); mtl_sub_copy(a + 128, (uint64_t)(uintptr_t)C + 64, 12); f = mtl_sub_commit(); mtl_wait(f, 10000000000ull);
        ok = g_hy3_merges - m0 == 1 && C[64] == 0x05 && C[75] == 0x05; break;
    case 3: *what = "S12 the window still open at the submission's end: closed before the command buffer ends";
        vk_hy3_test_stage(cb, a, PG, 256, 0x7c); f = mtl_sub_commit(); mtl_wait(f, 10000000000ull);
        ok = g_hy3_merges - m0 == 1 && D[256] == 0x7c && D[271] == 0x7c; break;
    case 4: *what = "S13 a fill after the window: closed before it, the fill's bytes after the merge's";
        vk_hy3_test_stage(cb, a, PG, 0, 0x13); mtl_sub_fill(a + 8, 4, 0x99999999u); f = mtl_sub_commit(); mtl_wait(f, 10000000000ull);
        ok = g_hy3_merges - m0 == 1 && D[0] == 0x13 && D[8] == 0x99 && D[11] == 0x99 && D[12] == 0x13; break;
    case 5: *what = "S15 a counter report after the window: the range right at completion, the report written";
        vk_hy3_test_stage(cb, a, PG, 0, 0x15); mtl_sub_counter(a + PG + 64, 0); f = mtl_sub_commit(); mtl_wait(f, 10000000000ull);
        ok = D[0] == 0x15 && (D[PG + 64] | D[PG + 65] | D[PG + 66] | D[PG + 67] | D[PG + 68] | D[PG + 69] | D[PG + 70] | D[PG + 71]) != 0; break;
    }
    hy3_test_clear(); return ok;
}
// csdiag (diagnostic, DM_VK_CSDIAG=1, default off): per compute shader, its storage buffer bindings by kind (read-only, written, atomic) and memory (imported chunk,
// device-local copy, outside every pool), bytes bound, every ~600 presents the 16 with the most written + atomic bytes into imported chunks. Which dispatches write
// or run atomics on imported game memory (OoT's heavy dispatches f4f0bc23, 39e60874, 4ef1f343: is their time the import's?)
enum { CSD_N = 512 };
static struct { uint64_t key, n, ro, wr, at, zero, imp_wr, imp_at, dl_wr; } g_csd[CSD_N]; static int g_ncsd; static uint64_t g_csd_fr;
static int csd_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_CSDIAG"); v = e && *e == '1'; if (v) VKLOG("csdiag: compute storage buffer bindings counted by shader (DM_VK_CSDIAG=1)"); } return v; }
static void csd_note(uint64_t key, const nvn_eden_bind *b, const vkpool_t *p, uint64_t n) {
    int i = 0; while (i < g_ncsd && g_csd[i].key != key) i++;
    if (i == g_ncsd) { if (g_ncsd == CSD_N) return; memset(&g_csd[i], 0, sizeof g_csd[i]); g_csd[i].key = key; g_ncsd++; }
    g_csd[i].n++;
    if (!p) { g_csd[i].zero++; return; }
    const int at = b->is_written && b->is_atomic, wr = b->is_written && !at;
    if (at) g_csd[i].at += n; else if (wr) g_csd[i].wr += n; else g_csd[i].ro += n;
    if (p->zero_copy) { if (at) g_csd[i].imp_at += n; else if (wr) g_csd[i].imp_wr += n; } else if (at || wr) g_csd[i].dl_wr += n;
}
static void csd_report(void) {
    extern uint64_t g_tex_frame; if (!csd_on() || g_tex_frame < g_csd_fr + 600) return;
    const double f = g_csd_fr ? (double)(g_tex_frame - g_csd_fr) : 600.0; g_csd_fr = g_tex_frame;
    for (int i = 0; i < g_ncsd; i++) for (int j = i + 1; j < g_ncsd; j++) if (g_csd[j].imp_wr + g_csd[j].imp_at > g_csd[i].imp_wr + g_csd[i].imp_at) { __typeof__(g_csd[0]) t = g_csd[i]; g_csd[i] = g_csd[j]; g_csd[j] = t; }
    VKLOG("csdiag: compute shaders by bytes written or run atomics on in imported chunks, per present over %.0f presents (%d shaders):", f, g_ncsd);
    for (int i = 0; i < g_ncsd && i < 16; i++) VKLOG("csdiag:   %016llx: %.1f bindings, read-only %.2f MB, written %.2f MB, atomic %.2f MB; into imported chunks written %.2f MB, atomic %.2f MB; into copies %.2f MB; outside every pool %.1f",
        (unsigned long long)g_csd[i].key, g_csd[i].n / f, g_csd[i].ro / f / 1048576.0, g_csd[i].wr / f / 1048576.0, g_csd[i].at / f / 1048576.0, g_csd[i].imp_wr / f / 1048576.0, g_csd[i].imp_at / f / 1048576.0,
        g_csd[i].dl_wr / f / 1048576.0, g_csd[i].zero / f);
    g_ncsd = 0;
}
static void eden_ssbo(int st, const nvn_eden_bind *b, uint32_t k, eden_entry *e, int dbuf) {
    if (b->cbuf == 0xFE) {
        if (dbuf) eden_addr_of(e, &mesh_draw_args, VK_FORMAT_UNDEFINED);
        else e->buffer = mesh_draw_args;
        return;
    }
    if (b->cbuf == 0xFF) {   // NVN2 raw global memory: the translator's reserved pool table (shader_info.h GMEM_MAP_CBUF_INDEX; gfx_vk_core.c vk_gmem_map). Without one
        // (no buffer device address) the zero buffer: every entry starts and ends at 0, so the lookup finds no pool and the access is dropped, as before
        VkBuffer buf; VkDeviceSize n; VkDescriptorBufferInfo bi = {vk_zero_buffer(), 0, 65536};
        if (b->is_atomic) { void vk_hy_raw_unlocated(uint64_t); vk_hy_raw_unlocated(st == 5 ? g_eden_cs_key : S.sh[st] ? S.sh[st]->key : 0); }   // hybrid1: counted (no write log in Eden mode: rule H11)
        if (st == 5) hy3_other(0, ~0ull);   // stage1: accesses anywhere: no staging in this dispatch
        if (vk_gmem_map(&buf, &n)) bi = (VkDescriptorBufferInfo){buf, 0, n};
        if (!dbuf) e->buffer = bi; else eden_addr_of(e, &bi, VK_FORMAT_UNDEFINED);
        return;
    }
    uint32_t off = b->offset + 16 * k;   // [B7: arrays of storage buffers]
    uint64_t a = (uint64_t)eden_cbuf_word(st, b->cbuf, off) | (uint64_t)eden_cbuf_word(st, b->cbuf, off + 4) << 32, n = eden_cbuf_word(st, b->cbuf, off + 8);
    if (b->indirect_offset != ~0u && dm_sdk_ounce()) {
        // NVN2: a storage buffer whose address the shader loads from memory (Eden's indirect_offset): c[cbuf][offset] holds a table's address, the buffer's is the
        // word at table + indirect_offset, read here from the pool memory that holds it, as Eden's buffer cache binds it (Lichii buffer_cache.h BindGraphicsStorageBuffer,
        // StorageBufferBinding). Bound before as the table itself, Donkey Kong Bananza's gsys vertex shader reset its terrain's indirect draw records into the table, and the
        // terrain drew black (its README, Rendering 7). Size as StorageBufferBinding takes it for a table pointer: a written one 8 MiB, a read-only one to the end of its
        // mapping (here the end of its pool chunk, below). A table or pointer that is 0 or outside the pools binds what a null address binds
        static uint64_t ind_n, ind_null; static int ind_logged;
        const uint64_t table = a; a = 0; n = b->is_written ? 8ull << 20 : 0;
        if (table && vk_pool_of(table + b->indirect_offset, 8)) a = *(const volatile uint64_t *)(uintptr_t)(table + b->indirect_offset);
        const uint64_t in = __atomic_add_fetch(&ind_n, 1, __ATOMIC_RELAXED), nn = a ? __atomic_load_n(&ind_null, __ATOMIC_RELAXED) : __atomic_add_fetch(&ind_null, 1, __ATOMIC_RELAXED);   // S1c (parallel recording): atomic
        if (!__atomic_exchange_n(&ind_logged, 1, __ATOMIC_RELAXED) || (!a && nn == 1)) {
            VKLOG("NVN2: indirect storage buffer bound (pointer read from memory): stage %d, table %#llx + %#x -> %#llx%s (%llu so far, %llu null)", st, (unsigned long long)table,
                  b->indirect_offset, (unsigned long long)a, b->is_written ? ", written" : "", (unsigned long long)in, (unsigned long long)nn); }
    }
    vkpool_t *p = a ? ((n ? vk_pool_of(a, n) : NULL) ?: vk_pool_of(a, 1)) : NULL;
    VkDescriptorBufferInfo bi = {b->is_written ? vk_wsink_buffer() : vk_zero_buffer(), 0, 65536};   // rptfix: a written stand-in's stores go to the sink, not the zero buffer others read
    const uint64_t req = n;   // NVN2 diagnostic (eden_ssbo_diag)
    if (st == 5 && b->is_written && g_eden_cs_key) { void vk_pool_id_ssbo(uint64_t, uint64_t, uint64_t); vk_pool_id_ssbo(g_eden_cs_key, a, n); }   // bak91c (gfx_vk_core.c)
    if (b->is_written) wt_check("storage buffer", st, a, n);   // wtrace (log only)
    if (!p && st == 5 && csd_on()) csd_note(g_eden_cs_key, b, NULL, 0);   // csdiag
    if (!p && dm_sdk_ounce()) eden_ssbo_diag(st, b, k, off, a, req, 0, 1);
    if (p) {   // bound from the storage alignment below the address, as Eden's buffer cache binds and its global memory pass offsets
        VkDeviceSize o = a - p->lo, al = V.props.limits.minStorageBufferOffsetAlignment ? V.props.limits.minStorageBufferOffsetAlignment : 16;
        if (req && o + req > p->hi - p->lo && dm_sdk_ounce()) eden_ssbo_diag(st, b, k, off, a, req, p->hi - p->lo - o, 0);
        if (!n || o + n > p->hi - p->lo) n = p->hi - p->lo - o;
        bi = (VkDescriptorBufferInfo){p->buf, o - o % al, n + o % al};
        if (st == 5 && csd_on()) csd_note(g_eden_cs_key, b, p, n);   // csdiag
        const int hy3 = st == 5 && hy3_bind(b, p, &bi, e, dbuf);   // stage1: an atomic one (or one inside a staged range) bound to device-local staging
        if (b->is_written && g_atom_on) atomdiag_note(st, st == 5 ? g_eden_cs_key : S.sh[st] ? S.sh[st]->key : 0, p, n);   // atomdiag (log only)
        if (st == 1 && b->is_written && n1x_atomic.local && bi.buffer == n1x_atomic.guest &&
            bi.offset == n1x_atomic.offset && bi.range == n1x_atomic.size) {
            bi.buffer = n1x_atomic.local;
            bi.offset = 0;
            n1x_atomic.bound = 1;
            static unsigned noted;
            unsigned bit = S.sh[1]->key == 0x24e19d5959e8769dull ? 1 : 2;
            if (!(noted & bit)) {
                noted |= bit;
                VKLOG("N1X device-local fragment atomics: fs %016llx, %llu bytes copied per direction",
                      (unsigned long long)S.sh[1]->key, (unsigned long long)n);
            }
        }
        // fstage: a written fragment storage buffer in an imported chunk is learned (the replay thread), and while its draw is staged its binding goes to S
        if (st == 1 && b->is_written && !n1x_atomic.bound) {
            if (g_fst.n && fst_redirect(&bi)) {}
            else if (!g_wk && p->zero_copy && S.sh[1] && !n1x_atomic_prog(S.sh[1])) fst_learn(S.sh[1], b, k, a, n);
        }
        // hybrid1: the kind of binding, for the diagnostic line (hy_report) while its GPU-write bookkeeping runs, and an atomic's request (its unit a device-local copy
        // from the next submission on; NXN1X_POOL_MODE=hybrid only)
        { void vk_hy_bind(int, uint64_t); void vk_hy_ssbo(const vkpool_t *, int, int, uint64_t);
          const int atom = b->is_written && b->is_atomic; vk_hy_bind(atom ? 3 : st == 5 ? (b->is_written ? 2 : 1) : b->is_written ? 4 : 0, n);
          vk_hy_ssbo(p, b->is_written, b->is_atomic && !hy3, st == 5 ? g_eden_cs_key : S.sh[st] ? S.sh[st]->key : 0); }   // (stage1: a staged one needs no chunk conversion)
        // hybrid2: only a binding the shader writes is a GPU write (a dispatch's read-only storage buffers were taken as written: OoT 00:25, 65% of 4.9 GB of pre and
        // post copies per 120 submissions), and a dispatch's only as far as its stores reach (Eden write_extent). NXN1X_SSBO_WRITES=all: as before
        { void vk_ssbo_bound(int, uint64_t, uint64_t, int, int, int, uint64_t); uint64_t reach = 0;
          const int hr = st == 5 && b->is_written && nvn_eden_ssbo_reach(b, g_dispatch_ind ? NULL : g_dispatch, &reach);
          vk_ssbo_bound(st, a, n, b->is_written, b->indirect_offset != ~0u, hr, reach); }
        { void vk_hy_bind_end(void); vk_hy_bind_end(); }
    }
    if (!dbuf) e->buffer = bi; else eden_addr_of(e, &bi, VK_FORMAT_UNDEFINED);
}
static void eden_tbuf(int st, const nvn_eden_bind *b, uint32_t k, eden_entry *e, int dbuf) {
    uint64_t h = eden_handle(st, b, k); const view_t *v; const tex_t *t = NULL; int tid = (int)(h & 0xfffff), img = b->kind == NVN_EB_IMG_TBUF;
    int ok = tid && vk_tex_by_id(tid, &v, &t) && t && t->d.target == NVN_TEXTURE_TARGET_BUFFER;
    if (ok && t->mem && st == 5) hy3_other((uint64_t)(uintptr_t)t->mem, (uint64_t)(uintptr_t)t->mem + (t->L.size ? t->L.size : 1));   // stage1: a texel buffer over a staged range cancels it
    if (dbuf) {
        const vkfmt_t *f = ok ? vk_fmt(t->d.format) : NULL; VkBuffer buf; VkDeviceSize o; VkDescriptorBufferInfo bi = {VK_NULL_HANDLE, 0, 0};
        if (f && vk_buf_at((uint64_t)(uintptr_t)t->mem, t->L.size, &buf, &o)) {
            if (o % (V.props.limits.minTexelBufferOffsetAlignment ? V.props.limits.minTexelBufferOffsetAlignment : 1)) VKRED_ONCE(165, "a texel buffer below the texel buffer alignment in a descriptor buffer pipeline: read as null");
            else bi = (VkDescriptorBufferInfo){buf, o, t->L.size}; }
        eden_addr_of(e, &bi, f ? f->vk : VK_FORMAT_UNDEFINED);
    } else {
        VkBufferView bv = ok ? vk_tbuf_view(t) : VK_NULL_HANDLE;
        if (!bv && ok && !img) bv = vk_tbuf_copy_view(t);   // r75
        e->texel = bv ? bv : vk_dummy_tbuf(img);
    }
    if (ok && img) vk_sub_wrote((uint64_t)(uintptr_t)t->mem, t->L.size);   // a storage texel buffer: copied back (copied pools)
    if (ok && img) wt_check("storage texel buffer", st, (uint64_t)(uintptr_t)t->mem, t->L.size);   // wtrace (log only)
}
static _Thread_local int g_pd_tsample;   // (S1c (parallel recording): per thread) slot12 replaydiag: this payload is one of the timed every-8th draws (eden_payload)
// expoprobe (FE Fortune's Weave 11-22, the tutorial square darker than the console; log only): nxnative found FE's dark map in its auto exposure (a 16x16
// RGBA16F luminance, a 256-bin histogram, a 3x1 R16F exposure). Every 2 s, a sampled 3x1 R16_SFLOAT or 16x16 R16G16B16A16_SFLOAT image is copied to the host
// and its values logged when its submission has finished: the exposure's 3 values, the luminance's mean / min / max of the first channel and its log2.
// Their console-matching values: luminance 134.4 (log2 7.07) on the first map, exposure settling at 5.555. NXN1X_EXPOPROBE=0 off; NVN2 boots only
typedef struct { VkBuffer b; VkDeviceMemory m; void *p; uint64_t fence; uint32_t w, h; VkFormat f; uint64_t present; } xprobe_t;
static xprobe_t g_xprobe[2]; static uint64_t g_xprobe_t[2];
static void xprobe_poll(void) {
    for (int i = 0; i < 2; i++) { xprobe_t *d = &g_xprobe[i]; if (!d->b || vk_fence_done() < d->fence) continue;
        const uint16_t *hp = (const uint16_t *)d->p;
        if (d->f == VK_FORMAT_R16_SFLOAT) VKLOG("expoprobe: 3x1 R16F (exposure) at present %llu: %.4f %.4f %.4f", (unsigned long long)d->present, half2f(hp[0]), half2f(hp[1]), half2f(hp[2]));
        else { double sum = 0, mn = 1e30, mx = -1e30; int n = 0, bad = 0; for (uint32_t j = 0; j < d->w * d->h; j++) { const float v = half2f(hp[4 * j]); if (v != v || v > 1e30f || v < -1e30f) { bad++; continue; } sum += v; if (v < mn) mn = v; if (v > mx) mx = v; n++; }
            const double mean = n ? sum / n : 0; VKLOG("expoprobe: 16x16 RGBA16F (luminance) at present %llu: R mean %.4f (log2 %.3f) min %.4f max %.4f, non-finite %d; first texel %.4f %.4f %.4f %.4f",
                (unsigned long long)d->present, mean, mean > 0 ? log2(mean) : -99.0, n ? mn : 0, n ? mx : 0, bad, half2f(hp[0]), half2f(hp[1]), half2f(hp[2]), half2f(hp[3])); }
        vkDestroyBuffer(V.dev, d->b, NULL); vkFreeMemory(V.dev, d->m, NULL); memset(d, 0, sizeof *d); }
}
static void xprobe_take(vkimg_t *im) {
    static int on = -1; if (on < 0) { const char *ev = getenv("NXN1X_EXPOPROBE"); on = ev && *ev == '1'; }   // release: off unless NXN1X_EXPOPROBE=1
    if (!on || !im || !dm_sdk_ounce() || !(im->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) return;
    const int which = im->w == 3 && im->h == 1 && im->fmt == VK_FORMAT_R16_SFLOAT ? 0 : im->w == 16 && im->h == 16 && im->fmt == VK_FORMAT_R16G16B16A16_SFLOAT ? 1 : -1;
    if (which < 0) return;
    xprobe_poll();
    const uint64_t now = tnow(); if (g_xprobe[which].b || now - g_xprobe_t[which] < 2000000000ull) return; g_xprobe_t[which] = now;
    xprobe_t *d = &g_xprobe[which]; const VkDeviceSize n = (VkDeviceSize)im->w * im->h * (which ? 8 : 2);
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(V.dev, &bci, NULL, &d->b) != VK_SUCCESS) { d->b = VK_NULL_HANDLE; return; }
    VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, d->b, &rq);
    const uint32_t type = vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, type};
    if (type == UINT32_MAX || vkAllocateMemory(V.dev, &ai, NULL, &d->m) != VK_SUCCESS || vkBindBufferMemory(V.dev, d->b, d->m, 0) != VK_SUCCESS ||
        vkMapMemory(V.dev, d->m, 0, VK_WHOLE_SIZE, 0, &d->p) != VK_SUCCESS) {
        vkDestroyBuffer(V.dev, d->b, NULL); if (d->m) vkFreeMemory(V.dev, d->m, NULL); memset(d, 0, sizeof *d); return; }
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {im->w, im->h, 1}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, d->b, 1, &c);
    VkMemoryBarrier host = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0, NULL); vk_img_barrier_all(cb);
    extern uint64_t g_tex_frame; d->fence = mtl_last_fence() + 1; d->w = im->w; d->h = im->h; d->f = im->fmt; d->present = g_tex_frame;
}
static void eden_image(int st, const nvn_eden_bind *b, uint32_t k, uint32_t bi, eden_entry *e, VkImageView *scaled) {
    dmsv_bind tb; memset(&tb, 0, sizeof tb); int storage = b->kind == NVN_EB_IMG;
    tb.kind = storage ? RK_IMG : RK_TEX; tb.desc = 0xff;   // (not RD_SAMPLED: cmp_check reads the bundle's Dref bits)
    tb.dim = b->dim; tb.flags = (uint8_t)((b->arrayed ? 1 : 0) | (b->is_ms ? 2 : 0)); tb.binding = (uint16_t)bi; tb.samp = RS_TEX;
    { static int ro_on = -1; if (ro_on < 0) { const char *ev = getenv("DM_VK_STORAGE_RO"); ro_on = !(ev && *ev == '0'); }
      g_storage_ro = storage && !b->is_written && ro_on && dm_sdk_ounce(); }   // bak94
    extern _Thread_local int g_vk_storage_want; g_vk_storage_want = storage ? b->img_format : 0;   // ootu32 (gfx_vk_tex.c)
    if (g_pd_tsample) { const uint64_t th = tick(), h = eden_handle(st, b, k), t0 = tick(); g_rd_ns[26] += t0 - th; g_rd_n[26]++; g_rd_n[27] += b->handle_info ? 1 : 0; g_rd_n[28] += !b->handle_info && b->cbuf >= 3 ? 1 : 0;   // gwsuffix replaydiag: eden_handle (26), of them evaluated (27) and read from a uniform buffer (28)
        tex_desc_for(st, &tb, h, &e->image, storage, scaled); g_rd_ns[35] += tick() - t0; g_rd_n[35]++; }   // replaydiag: tex_desc_for's part of eden_image (sampled draws)
    else tex_desc_for(st, &tb, eden_handle(st, b, k), &e->image, storage, scaled);
    g_storage_ro = 0; g_vk_storage_want = 0;
    if (!storage && dm_sdk_ounce() && !g_wk && S.pass_open == 0) {   // expoprobe (replay thread, outside a render pass: a copy is recorded into the main command buffer; the exposure shaders are compute)
        const uint64_t h = eden_handle(st, b, k); const view_t *v = NULL; const tex_t *t = NULL;
        if ((h & 0xfffff) && vk_tex_by_id((int)(h & 0xfffff), &v, &t) && t) xprobe_take(vk_tex_peek(t));
    }
    // fmtdiag (FE Fortune's Weave 11-22, a UI portrait card wrong since the first run, found to be a MIRROR_Y copy: copymirror; log only): each new combination a fragment shader
    // samples of NVN format, texture and view swizzles and the Vulkan format taken, once, up to 128 lines. NXN1X_FMTDIAG=0 off; NVN2 boots only
    if (st == 1 && !storage && dm_sdk_ounce()) {
        static int on = -1; if (on < 0) { const char *ev = getenv("NXN1X_FMTDIAG"); on = ev && *ev == '1'; }   // release: off unless NXN1X_FMTDIAG=1
        static uint64_t seen[128]; static int nseen;
        if (on && nseen < 128) {
            const uint64_t h = eden_handle(st, b, k); const view_t *v = NULL; const tex_t *t = NULL;
            if ((h & 0xfffff) && vk_tex_by_id((int)(h & 0xfffff), &v, &t) && t) {
                const int fmt = v && v->set_format ? v->format : (int)t->d.format;
                const uint32_t tswz = t->set_swz ? t->swz : 0, vswz = v && v->set_swizzle ? (uint32_t)(v->swz[0] | v->swz[1] << 8 | v->swz[2] << 16 | (uint32_t)v->swz[3] << 24) : 0;
                const vkfmt_t *f = vk_fmt(fmt);
                const uint64_t key = (uint64_t)fmt | (uint64_t)tswz << 16 | (uint64_t)vswz << 32 ^ (uint64_t)(f ? f->vk : 0) << 48;
                int known = 0; for (int j = 0; j < nseen; j++) if (seen[j] == key) { known = 1; break; }
                if (!known) { seen[nseen++] = key; extern uint64_t g_tex_frame;
                    VKLOG("fmtdiag: fragment sampled nvn format %#x %ux%u target %d, texture swizzle %#x (set %d), view swizzle %#x (set %d), Vulkan format %d swizzle %d %d %d %d, present %llu",
                          fmt, t->d.width, t->d.height, t->d.target, tswz, t->set_swz, vswz, v ? v->set_swizzle : 0, f ? (int)f->vk : -1,
                          f ? (int)f->swz.r : -1, f ? (int)f->swz.g : -1, f ? (int)f->swz.b : -1, f ? (int)f->swz.a : -1, (unsigned long long)g_tex_frame); }
            }
        }
    }
    // bak92 (11-16, MKW cacti / trees flicker; logging only): what the scenery culling dispatches sample (0823b08c2027a38d's IR has 8 ImageSampleExplicitLod and an
    // ImageGather; nxnative's README:71 says e04783bda5a52269 reads none). Each binding's image is logged when it first appears and whenever it changes (another image:
    // a new size, a re-made image), up to 256 lines: the texture, the image's storage, whether the GPU drew it and at which present, and the current present.
    // DM_VK_CULL_TEX_DIAG=0 off; NVN2 boots only
    if (g_eden_cs_key && dm_sdk_ounce()) {
        static int on = -1; if (on < 0) { const char *ev = getenv("DM_VK_CULL_TEX_DIAG"); on = !(ev && *ev == '0'); }
        static const uint64_t ks[4] = {0x0823b08c2027a38dull, 0xe04783bda5a52269ull, 0x2ce9071c2f22d74full, 0xfb3fdaea2e0f1779ull};
        int s = -1; for (int j = 0; j < 4; j++) if (ks[j] == g_eden_cs_key) s = j;
        static const void *last[4][32]; static int nl;
        if (on && s >= 0 && bi < 32 && nl < 256) {
            const uint64_t h = eden_handle(st, b, k); const view_t *v = NULL; const tex_t *t = NULL; vkimg_t *im = NULL;
            if ((h & 0xfffff) && vk_tex_by_id((int)(h & 0xfffff), &v, &t) && t) im = vk_tex_peek(t);
            if ((const void *)im != last[s][bi]) { last[s][bi] = im; nl++;
                extern uint64_t g_tex_frame;
                VKLOG("bak92 cull tex: dispatch %016llx binding %u (%s, dim %d%s) handle %#llx: %s%ux%ux%u levels %u nvn format %#x target %d samples %u, pool %p +%#llx; image %p %ux%ux%u, GPU-written %d, rendered at present %lld, made at present %llu (now %llu)%s",
                    (unsigned long long)g_eden_cs_key, bi, storage ? "image" : "sampled", b->dim, b->arrayed ? ", arrayed" : "", (unsigned long long)h, t ? "" : "no texture ",
                    t ? t->d.width : 0, t ? t->d.height : 0, t ? t->d.depth : 0, t ? t->d.levels : 0, t ? t->d.format : 0, t ? t->d.target : -1, t ? t->d.samples : 0,
                    t ? (const void *)t->pool : NULL, t ? (unsigned long long)t->off : 0ull, (void *)im, im ? im->w : 0, im ? im->h : 0, im ? im->layers : 0, im ? im->gpu_w : 0,
                    im ? (long long)im->wpres - 1 : -2LL, im ? (unsigned long long)im->made : 0ull, (unsigned long long)g_tex_frame, im && im->wpres == 0 ? " [never rendered as this image]" : "");
            }
        }
    }
}
// vsres (NXN1X_VSRES_DIAG=<vertex shader key in hex>, log only): for the first 4 draws of that vertex shader, every binding of the Eden pipeline (both stages) and the
// vertex attribute state: texel buffers with their guest storage, size, format and backing, a content summary (first 4 words, zero / NaN / Inf counts over the first
// 64 KiB read as floats), textures with their handle, size, format, serial and storage, and each attribute's state, stream stride, divisor and buffer. Hyrule Warriors'
// soldiers (VS 1df3a0c17a3e9e80) cost 20-25 ms of GPU a frame and write no colour or depth (11-20 passdump): are their vertex inputs valid data?
static void vsres_summary(const uint8_t *p, uint64_t n, char *out, size_t cap) {
    if (!p || !n) { snprintf(out, cap, "no CPU storage"); return; }
    // vsres3: the whole storage (up to 16 MiB), not its first 64 KiB: a ring of per-frame data can sit anywhere in it. The first and last non-zero word and the 64 KiB
    // blocks holding any, so a draw's slice can be told from unused space
    uint64_t m = n < (16u << 20) ? n : (16u << 20), words = m / 4, zero = 0, nan = 0, inf = 0, first_nz = ~0ull, last_nz = 0, blocks = 0, cur_block = ~0ull;
    uint32_t w4[4] = {0, 0, 0, 0};
    for (uint64_t i = 0; i < words; i++) { uint32_t u; memcpy(&u, p + i * 4, 4); if (i < 4) w4[i] = u;
        if (!u) { zero++; continue; }
        if (first_nz == ~0ull) first_nz = i * 4; last_nz = i * 4; if ((i * 4) >> 16 != cur_block) { cur_block = (i * 4) >> 16; blocks++; }
        if ((u & 0x7f800000u) == 0x7f800000u) { if (u & 0x7fffffu) nan++; else inf++; } }
    snprintf(out, cap, "first %08x %08x %08x %08x; of %llu words: zero %llu NaN %llu Inf %llu; non-zero from %#llx to %#llx in %llu of %llu 64 KiB blocks", w4[0], w4[1],
             w4[2], w4[3], (unsigned long long)words, (unsigned long long)zero, (unsigned long long)nan, (unsigned long long)inf,
             (unsigned long long)(first_nz == ~0ull ? 0 : first_nz), (unsigned long long)last_nz, (unsigned long long)blocks, (unsigned long long)((m + 65535) >> 16));
}
void vk_treg_describe(int tid, char *out, size_t cap);   // vsres2 (gfx_vk_tex.c)
static void vsres_diag(const nvn_eden_pipeline_info *pi) {
    // vsres2: the first 2 draws of the vertex shader, and the first 4 whose fragment shader is NXN1X_VSRES_FS (the heavy G-buffer draws), each binding's registrations too
    static int64_t want = -1, want_fs = 0; static int done, done_fs;
    if (want < 0) { const char *e = getenv("NXN1X_VSRES_DIAG"); want = e && *e ? (int64_t)strtoull(e, NULL, 16) : 0;
        const char *f = getenv("NXN1X_VSRES_FS"); want_fs = f && *f ? (int64_t)strtoull(f, NULL, 16) : 0; }
    if (!want || !S.sh[0] || S.sh[0]->key != (uint64_t)want) return;
    const int fs_match = want_fs && S.sh[1] && S.sh[1]->key == (uint64_t)want_fs;
    if (fs_match ? done_fs >= 4 : done >= 2) return;
    if (fs_match) done_fs++; else done++;
    extern uint64_t g_tex_frame;
    VKLOG("vsres: draw %d of vertex shader %016llx (fragment %016llx), present %llu, %u bindings", fs_match ? 100 + done_fs : done, (unsigned long long)S.sh[0]->key,
          (unsigned long long)(S.sh[1] ? S.sh[1]->key : 0), (unsigned long long)g_tex_frame, pi->num_binds);
    for (uint32_t i = 0; i < pi->num_binds; i++) {
        const nvn_eden_bind *b = &pi->binds[i]; int st = b->nvn_stage;
        for (uint32_t k = 0; k < b->count && k < 4; k++) {
            if (b->kind == NVN_EB_UBO) { VKLOG("vsres:   bind %u stage %d uniform buffer c%u", i, st, (unsigned)(b->cbuf + k)); continue; }
            if (b->kind == NVN_EB_SSBO) { VKLOG("vsres:   bind %u stage %d storage buffer #%u", i, st, k); continue; }
            const uint64_t h = eden_handle(st, b, k); const view_t *v = NULL; const tex_t *t = NULL; const int tid = (int)(h & 0xfffff);
            const int ok = tid && vk_tex_by_id(tid, &v, &t) && t;
            char sum[200]; sum[0] = 0;
            if (ok && t->d.target == NVN_TEXTURE_TARGET_BUFFER) {
                vsres_summary((const uint8_t *)t->mem, t->L.size, sum, sizeof sum);
                if (st == 0) wt_note((uint64_t)(uintptr_t)t->mem, t->L.size);   // wtrace
                VkBuffer buf = VK_NULL_HANDLE; VkDeviceSize o = 0; const int inpool = vk_buf_at((uint64_t)(uintptr_t)t->mem, t->L.size, &buf, &o);
                VKLOG("vsres:   bind %u stage %d texel buffer handle %#llx id %d: storage %p size %#llx nvn format %#x, %s (offset %#llx), view %s; %s", i, st,
                      (unsigned long long)h, tid, (void *)t->mem, (unsigned long long)t->L.size, t->d.format, inpool ? "in a pool buffer" : "in no pool buffer (host copy)",
                      (unsigned long long)o, vk_tbuf_view(t) ? "direct" : "copy", sum);
            } else if (ok) {
                vkimg_t *im = vk_tex_peek(t);
                VKLOG("vsres:   bind %u stage %d %s (dim %d) handle %#llx id %d: %ux%ux%u levels %u nvn format %#x target %d serial %llu pool %p+%#llx size %#llx; image %p %ux%u",
                      i, st, b->kind == NVN_EB_IMG ? "image" : "texture", b->dim, (unsigned long long)h, tid, t->d.width, t->d.height, t->d.depth, t->d.levels, t->d.format,
                      t->d.target, (unsigned long long)t->serial, (const void *)t->pool, (unsigned long long)t->off, (unsigned long long)t->L.size, (void *)im, im ? im->w : 0,
                      im ? im->h : 0);
            } else {
                VKLOG("vsres:   bind %u stage %d kind %d handle %#llx id %d: no texture object", i, st, b->kind, (unsigned long long)h, tid);
            }
            if (tid) { char reg[640]; vk_treg_describe(tid, reg, sizeof reg); VKLOG("vsres:     registrations: %s", reg); }   // vsres2
        }
    }
    for (int l = 0; l < 16; l++) { uint32_t a = S.vattr[l]; if (!a) continue; const int s = (a >> 16) & 0xf;
        VKLOG("vsres:   attribute %d: state %#x stream %d stride %d divisor %d buffer %#llx size %llu", l, a, s, S.vstream[s].stride, S.vstream[s].divisor,
              (unsigned long long)S.vbuf[s].address, (unsigned long long)S.vbuf[s].size); }
}
// csres (NXN1X_VSRES_CS=<compute shader key in hex>, log only): the first 2 dispatches of that compute shader: the head of its c0 and c3 (c3 word 0 is the item count Hyrule
// Warriors' 44d0679e6309c2f3 loops over), every texel buffer it reads (resolved to the game's buffer texture or the 64 KiB zero dummy, storage, format, the whole content)
// and its storage buffers. Its output is the grass VS's per-instance data, all zero: nothing to place, or inputs that read as zero?
static void csres_diag(const nvn_eden_pipeline_info *pi, uint64_t key) {
    static int64_t want = -1; static int done;
    if (want < 0) { const char *e = getenv("NXN1X_VSRES_CS"); want = e && *e ? (int64_t)strtoull(e, NULL, 16) : 0; }
    if (!want || done >= 2 || key != (uint64_t)want) return;
    done++;
    extern uint64_t g_tex_frame;
    VKLOG("csres: dispatch %d of compute shader %016llx, present %llu, %u bindings; c0 %08x %08x %08x %08x %08x %08x %08x %08x; c3 %08x %08x %08x %08x %08x %08x %08x %08x", done,
          (unsigned long long)key, (unsigned long long)g_tex_frame, pi->num_binds, eden_cbuf_word(5, 0, 0), eden_cbuf_word(5, 0, 4), eden_cbuf_word(5, 0, 8), eden_cbuf_word(5, 0, 12),
          eden_cbuf_word(5, 0, 16), eden_cbuf_word(5, 0, 20), eden_cbuf_word(5, 0, 24), eden_cbuf_word(5, 0, 28), eden_cbuf_word(5, 3, 0), eden_cbuf_word(5, 3, 4),
          eden_cbuf_word(5, 3, 8), eden_cbuf_word(5, 3, 12), eden_cbuf_word(5, 3, 16), eden_cbuf_word(5, 3, 20), eden_cbuf_word(5, 3, 24), eden_cbuf_word(5, 3, 28));
    for (uint32_t i = 0; i < pi->num_binds; i++) {
        const nvn_eden_bind *b = &pi->binds[i];
        for (uint32_t k = 0; k < b->count && k < 4; k++) {
            if (b->kind == NVN_EB_UBO) { VKLOG("csres:   bind %u uniform buffer c%u", i, (unsigned)(b->cbuf + k)); continue; }
            if (b->kind == NVN_EB_SSBO) {
                const uint32_t off = b->offset + 16 * k;
                const uint64_t a = (uint64_t)eden_cbuf_word(5, b->cbuf, off) | (uint64_t)eden_cbuf_word(5, b->cbuf, off + 4) << 32;
                VKLOG("csres:   bind %u storage buffer c%u+%#x: %#llx size %#x%s%s", i, (unsigned)b->cbuf, off, (unsigned long long)a, eden_cbuf_word(5, b->cbuf, off + 8),
                      b->is_written ? ", written" : "", b->indirect_offset != ~0u ? ", indirect (address read from memory)" : "");
                continue;
            }
            const uint64_t h = eden_handle(5, b, k); const view_t *v = NULL; const tex_t *t = NULL; const int tid = (int)(h & 0xfffff);
            const int ok = tid && vk_tex_by_id(tid, &v, &t) && t;
            char sum[260] = "", reg[640] = "";
            if (tid) vk_treg_describe(tid, reg, sizeof reg);
            if (ok && t->d.target == NVN_TEXTURE_TARGET_BUFFER) {
                vsres_summary((const uint8_t *)t->mem, t->L.size, sum, sizeof sum);
                const int direct = vk_tbuf_view(t) != VK_NULL_HANDLE;
                VKLOG("csres:   bind %u texel buffer handle %#llx id %d: storage %p size %#llx nvn format %#x width %u, view %s; %s", i, (unsigned long long)h, tid, (void *)t->mem,
                      (unsigned long long)t->L.size, t->d.format, t->d.width, direct ? "direct" : "copy or ZERO DUMMY", sum);
            } else {
                VKLOG("csres:   bind %u kind %d handle %#llx id %d: %s -> reads the ZERO DUMMY", i, b->kind, (unsigned long long)h, tid,
                      ok ? "a registered texture that is not a buffer texture" : "no registered texture");
            }
            if (tid) VKLOG("csres:     registrations: %s", reg);
        }
    }
}
// 0: a descriptor could not be written (descriptor buffer memory), the draw is skipped
// cpudiag (DM_VK_PAYDIAG=1, log only; OOT_BARRIER_DESIGN.md 13-14): a draw's eden_payload ("descriptor sets" in the replay line, ~1.8 us a draw in OoT) split by
// binding kind, and per graphics stage whether its inputs (shader, uniform buffer addresses, storage buffer words, texture / image / texel handles; the driver
// segments' bound offsets) and its entries equal the last draw's: the reach of reusing a stage's entries when its inputs did not change. "inputs same, entries
// differ" counts what such a key would miss (a texture re-made or rescaled under the same handle). Logged with the Eden draw timing lines (every 2^17 draws)
static int paydiag_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_PAYDIAG"); v = e && *e == '1'; } return v; }
static _Thread_local int g_pd_draw;   // (S1c (parallel recording): per thread) eden_payload called for a draw (eden_draw), not a dispatch
static uint64_t g_pd_ns[4], g_pd_n[4];   // uniform buffers, storage buffers, images and samplers, texel buffers: ns, entries
static uint64_t g_pd_draws, g_pd_all_in, g_pd_all_out, g_pd_st[5][4];   // per stage: draws with bindings, inputs same, entries same, inputs same but entries differ
static uint64_t g_pd_last_in[5], g_pd_last_out[5]; static int g_pd_have[5];
static _Thread_local uint64_t g_pd_ov;   // (S1c (parallel recording): per thread: eden_draw subtracts its own draw's share) s46pc: ns the hashing above took (subtracted from the replay line's "descriptor sets")
static inline uint64_t pd_mix(uint64_t h, uint64_t v) { return (h ^ v) * 0x100000001b3ull; }
static uint64_t pd_entry_in(int st, const nvn_eden_bind *b, uint32_t k, const eden_entry *e) {
    switch (b->kind) {
    case NVN_EB_UBO: { uint32_t bank = b->cbuf + k;
        if (bank >= 3 && bank - 3 < 16) { const p_bind_buf *u = &S.ubo[st][bank - 3]; return pd_mix(pd_mix(bank, u->addr), u->size); }
        if (bank == 1) return 1;   // the shader's own c1 (the shader is in the stage's key)
        return pd_mix(bank, fnv(0xcbf29ce484222325ull, e, sizeof *e)); }   // c0 / c2 / push constants: the segment the driver bound
    case NVN_EB_SSBO: { if (b->cbuf == 0xFE || b->cbuf == 0xFF) return b->cbuf; uint32_t off = b->offset + 16 * k;
        return pd_mix(pd_mix(eden_cbuf_word(st, b->cbuf, off), eden_cbuf_word(st, b->cbuf, off + 4)), eden_cbuf_word(st, b->cbuf, off + 8)); }
    default: return pd_mix(b->kind, eden_handle(st, b, k));
    }
}
static void pd_draw_end(const uint64_t in[5], const uint64_t out[5], unsigned present) {
    g_pd_draws++; int all_in = 1, all_out = 1;
    for (int st = 0; st < 5; st++) {
        if (!(present >> st & 1)) { g_pd_have[st] = 0; continue; }
        const int si = g_pd_have[st] && g_pd_last_in[st] == in[st], so = g_pd_have[st] && g_pd_last_out[st] == out[st];
        g_pd_st[st][0]++; g_pd_st[st][1] += si; g_pd_st[st][2] += so; g_pd_st[st][3] += si && !so; all_in &= si; all_out &= so;
        g_pd_last_in[st] = in[st]; g_pd_last_out[st] = out[st]; g_pd_have[st] = 1;
    }
    g_pd_all_in += all_in; g_pd_all_out += all_out;
}
static void pd_log(void) {
    if (!paydiag_on() || !g_pd_draws) return;
    const double d = (double)g_pd_draws;
    VKLOG("paydiag: of %llu draws, us per draw: uniform buffers %.2f (%.1f entries), storage buffers %.2f (%.1f), images and samplers %.2f (%.1f), texel buffers %.2f (%.1f); "
          "every stage's inputs same as the last draw %.1f%%, entries same %.1f%%", (unsigned long long)g_pd_draws, g_pd_ns[0] / d / 1e3, g_pd_n[0] / d, g_pd_ns[1] / d / 1e3,
          g_pd_n[1] / d, g_pd_ns[2] / d / 1e3, g_pd_n[2] / d, g_pd_ns[3] / d / 1e3, g_pd_n[3] / d, 100.0 * g_pd_all_in / d, 100.0 * g_pd_all_out / d);
    static const char *const nm[5] = {"vertex", "fragment", "geometry", "tess control", "tess eval"};
    char line[600]; size_t o = 0;
    for (int st = 0; st < 5; st++) { const uint64_t *c = g_pd_st[st]; if (!c[0]) continue;
        o += snprintf(line + o, sizeof line - o, " %s: %llu draws, inputs same %.1f%%, entries same %.1f%%, inputs same but entries differ %llu;", nm[st], (unsigned long long)c[0],
                      100.0 * c[1] / c[0], 100.0 * c[2] / c[0], (unsigned long long)c[3]); }
    VKLOG("paydiag: per stage:%s", line);
    memset(g_pd_ns, 0, sizeof g_pd_ns); memset(g_pd_n, 0, sizeof g_pd_n); memset(g_pd_st, 0, sizeof g_pd_st); g_pd_draws = g_pd_all_in = g_pd_all_out = 0;
}
static int eden_payload(const nvn_eden_pipeline_info *pi, eden_entry *pay, uint32_t sampled_scale_bits[4]) {
    int dbuf = pi->descriptor_buffer; g_eden_dbuf_bad = 0;
    const int pd = g_pd_draw && paydiag_on(); uint64_t pd_in[5], pd_out[5]; unsigned pd_present = 0;   // cpudiag
    // replaydiag (DM_VK_REPLAYDIAG=1): the time by binding kind only (slots 30-33, slot 34 the draws timed), every 8th draw so it costs ~1/8 of paydiag's timers and none of its hashing
    static _Thread_local uint64_t pd_seq; const int pdt = pd || (g_pd_draw && g_rdiag > 0 && !(++pd_seq & 7));
    if (pdt && !pd) g_rd_n[34]++;
    g_pd_tsample = pdt && !pd;   // slot12 replaydiag: eden_image times its tex_desc_for on these draws
    if (pd) for (int st = 0; st < 5; st++) { pd_in[st] = pd_mix(0xcbf29ce484222325ull, (uint64_t)(uintptr_t)S.sh[st]); pd_out[st] = 0xcbf29ce484222325ull; }
    unsigned texture_index = 0;
    memset(sampled_scale_bits, 0, 4 * sizeof(uint32_t));
    const int use_scaled = nvn_eden_resolution_x2();
    if (dbuf && !V.has_bda) { VKRED_ONCE(166, "an Eden pipeline takes a descriptor buffer and the pools have no device addresses: its draws are skipped"); return 0; }
    for (uint32_t i = 0; i < pi->num_binds; i++) {
        const nvn_eden_bind *b = &pi->binds[i]; int st = b->nvn_stage;
        const int pk = b->kind == NVN_EB_UBO ? 0 : b->kind == NVN_EB_SSBO ? 1 : b->kind == NVN_EB_TBUF || b->kind == NVN_EB_IMG_TBUF ? 3 : 2;   // cpudiag
        const uint64_t pt0 = pdt ? tick() : 0;
        for (uint32_t k = 0; k < b->count && b->first_entry + k < EDEN_MAXE; k++) {
            eden_entry *e = &pay[b->first_entry + k]; memset(e, 0, sizeof *e);
            if (pd) g_pd_n[pk]++;
            switch (b->kind) {
            case NVN_EB_UBO: eden_ubo(st, b->cbuf + k, e, dbuf); break;
            case NVN_EB_SSBO: eden_ssbo(st, b, k, e, dbuf); break;
            case NVN_EB_TBUF: case NVN_EB_IMG_TBUF: eden_tbuf(st, b, k, e, dbuf); break;
            default: {
                VkImageView scaled = VK_NULL_HANDLE;
                int can_scale = use_scaled && b->kind == NVN_EB_TEX && !b->is_depth &&
                                b->count == 1 && texture_index < 128;
                eden_image(st, b, k, i, e, can_scale ? &scaled : NULL);
#if defined(DM_NXN1X)
                if (fg_frames_left && st == 5 && ndsr_trace_slot >= 0 &&
                    (b->kind == NVN_EB_TEX || b->kind == NVN_EB_IMG)) {
                    const view_t *view = NULL;
                    const tex_t *texture = NULL;
                    uint64_t handle = eden_handle(st, b, k);
                    if (vk_tex_by_id((int)(handle & 0xfffff), &view, &texture) && texture) {
                        vkimg_t *im = vk_tex_peek(texture);
                        fg_capture_image(scaled && im ? im->scaled : im,
                            view && view->set_levels ? view->base_level : 0,
                            view && view->set_layers ? view->min_layer : 0,
                            b->kind == NVN_EB_TEX ? "sampled" : "storage-before",
                            b->first_entry + k, g_eden_cs_key);
                    }
                }
#endif
                if (scaled) {
                    e->image.imageView = scaled;
                    sampled_scale_bits[texture_index / 32] |= 1u << (texture_index % 32);
                }
                break;
            }
            }
        }
        if (pdt && !pd) { g_rd_ns[30 + pk] += tick() - pt0; g_rd_n[30 + pk] += b->count; }   // replaydiag: timing only, no hashing
        if (pd) { const uint64_t ph = tick(); g_pd_ns[pk] += ph - pt0;   // cpudiag: the stage's inputs and entries hashed outside the timed part
            if (st >= 0 && st < 5) { pd_present |= 1u << st;
                for (uint32_t k = 0; k < b->count && b->first_entry + k < EDEN_MAXE; k++) {
                    const eden_entry *e = &pay[b->first_entry + k];
                    pd_in[st] = pd_mix(pd_mix(pd_in[st], b->first_entry + k), pd_entry_in(st, b, k, e)); pd_out[st] = fnv(pd_mix(pd_out[st], b->first_entry + k), e, sizeof *e); } }
            g_pd_ov += tick() - ph; }   // s46pc: kept out of the replay line's "descriptor sets" (it read ~1.8 us a draw high with paydiag on)
        if (b->kind == NVN_EB_TEX) ++texture_index;
    }
    if (pd) { const uint64_t ph = tick(); pd_draw_end(pd_in, pd_out, pd_present); g_pd_ov += tick() - ph; }
    return !g_eden_dbuf_bad;
}
// s35 (TotK run 52, 11-13: binds and state 0.80 us of a draw's ~4): set 0 pushed again with the same bytes, the same pipeline layout, in the same command buffer
// part (g_cmd_gen) and with nothing else bound in between (another set 0 bound here, or a pipeline the Eden draws did not bind: g_s35_pok cleared) is skipped,
// as totk-pc-recomp skips identical pushes by a byte compare. Graphics only (a dispatch's pushes are on the compute bind point). The timing log splits the rest.
// DM_VK_S35=0: no skip
static int s35_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_S35"); v = !e || !*e || atoi(e) != 0; } return v; }
// R1 (parallel recording): g_s35_pok, g_s35_pgen, g_s35_play, g_s35_pn, g_s35_ppay, g_s35_lastbound in struct rec_eden
// R1b (parallel recording): the counters g_s35_skip, g_s35_push, g_s35_ns too
// payload -> set 0 of the pipeline, recorded in cb
static int eden_bind_set(VkCommandBuffer cb, eden_pipe *ep, const eden_entry *pay, VkPipelineBindPoint bp) {
    const nvn_eden_pipeline_info *pi = &ep->pi;
    if (!pi->set_layout) return 1;
    if (pi->push_descriptor) {
        if (bp != VK_PIPELINE_BIND_POINT_GRAPHICS) { nvn_eden_cmd_push_descriptors(cb, ep->p, pay); return 1; }
        size_t nb = sizeof *pay * (pi->num_entries < EDEN_MAXE ? pi->num_entries : EDEN_MAXE);
        if (s35_on() && g_s35_pok && g_s35_pgen == g_cmd_gen && g_s35_play == pi->layout && g_s35_pn == pi->num_entries && !memcmp(g_s35_ppay, pay, nb)) { g_s35_skip++; return 1; }   // s35
        nvn_eden_cmd_push_descriptors(cb, ep->p, pay); g_s35_push++;
        g_s35_pok = 1; g_s35_pgen = g_cmd_gen; g_s35_play = pi->layout; g_s35_pn = pi->num_entries; memcpy(g_s35_ppay, pay, nb);
        return 1;
    }
    if (bp == VK_PIPELINE_BIND_POINT_GRAPHICS) g_s35_pok = 0;   // s35: set 0 bound another way
    if (pi->descriptor_buffer) {
        uint64_t *const gen = &g_rec_e->dbuf_gen; VkDeviceAddress *const bound = &g_rec_e->dbuf_bound;   // R1b (parallel recording): were static here
        uint64_t al = nvn_eden_dbuf_alignment(); VkBuffer b; VkDeviceSize o; VkDeviceAddress a;
        uint8_t *p = al ? eden_ring_alloc(g_rec_e->dring, pi->dbuf_size, al, &b, &o, &a) : NULL;
        if (!p) { VKRED_ONCE(167, "Eden descriptor buffer memory not available: draw skipped"); return 0; }
        nvn_eden_write_dbuf(ep->p, pay, p);
        int rebind = *gen != g_cmd_gen || *bound != a; *gen = g_cmd_gen; *bound = a;   // R1b (parallel recording)
        nvn_eden_cmd_dbuf(cb, ep->p, a, o, rebind); return 1;
    }
    uint64_t h = fnv(fnv(0xcbf29ce484222325ull, &pi->set_layout, sizeof pi->set_layout), pay, sizeof *pay * pi->num_entries); unsigned k = (unsigned)(h % SETC);
    VkDescriptorSet set;
    if (g_setc[k].epoch == g_set_epoch && g_setc[k].h == h && g_setc[k].l == (VkDescriptorSetLayout)pi->set_layout && !g_x_nosetc) { g_set_hits++; set = g_setc[k].set; }
    else { set = vk_alloc_set((VkDescriptorSetLayout)pi->set_layout); g_set_new++; nvn_eden_update_set(ep->p, set, pay);
           g_setc[k].h = h; g_setc[k].epoch = g_set_epoch; g_setc[k].l = (VkDescriptorSetLayout)pi->set_layout; g_setc[k].set = set; }
    vkCmdBindDescriptorSets(cb, bp, (VkPipelineLayout)pi->layout, 0, 1, &set, 0, NULL);
    return 1;
}

// ---- a draw (prepare_draw in Eden mode, after the pass is open); 1: everything bound, the caller draws ----
// R1 (parallel recording): g_eden_vpm in struct rec_eden
// s45 (DM_VK_S45 & 2, & 4): Eden's dynamic state and the depth bounds as the last Eden draw recorded them
// R1 (parallel recording): g_s45_st, g_s45_db in struct rec_eden
static void s45_dbounds(VkCommandBuffer cb, float lo, float hi, int chain) {
    if ((g_s45_on & 4) && !g_s45_verify && chain && g_s45_db.gen == g_cmd_gen && g_s45_db.lo == lo && g_s45_db.hi == hi) return;
    vkCmdSetDepthBounds(cb, lo, hi); g_s45_db.gen = g_cmd_gen; g_s45_db.lo = lo; g_s45_db.hi = hi;
}
// R1 (parallel recording): g_eden_vbm in struct rec_eden
// R1b (parallel recording): g_vlk_hit, g_vlk_miss (replay2: vertex stream lookups kept / made, eden_draw) in struct rec_eden
static VkCullModeFlags eden_cull(uint32_t nvn) { return nvn == 1 ? VK_CULL_MODE_FRONT_BIT : nvn == 2 ? VK_CULL_MODE_BACK_BIT : nvn == 3 ? VK_CULL_MODE_FRONT_AND_BACK : VK_CULL_MODE_NONE; }
// Measurement (perf principle: measure, keep only wins): where a draw's replay time goes in Eden mode, logged every 2^17 draws in microseconds per draw:
// pre = prepare_draw before eden_draw, key = eden_fill_state + pipeline lookup, state = binds + set_dynamic + viewports + nvn_eden_cmd_state, vb = vertex buffers
static void eden_prof(uint64_t pre, uint64_t key, uint64_t state, uint64_t vb) {
    // S1c (parallel recording): the thread's (struct rec_eden prof_n / prof_s, were statics here); a worker's are folded into the replay thread's (vk_rec_ctx_fold), which
    // logs once 2^17 draws are in (n can then pass it: the averages are over n)
    uint64_t *const s = g_rec_e->prof_s;
    s[0] += pre; s[1] += key; s[2] += state; s[3] += vb;
    if (++g_rec_e->prof_n >= (1u << 17) && g_rec_e == &g_rec_e_main) {
        const uint64_t n = g_rec_e->prof_n; const double k = 1e3 * (double)n; g_rec_e->prof_n = 0;
        VKLOG("Eden draw timing (last %llu draws, us per draw): before %.2f, key %.2f, binds and state %.2f, vertex buffers %.2f", (unsigned long long)n, s[0] / k, s[1] / k, s[2] / k, s[3] / k);
        VKLOG("s35: of those, us per draw: set 0 bind %.2f, set_dynamic %.2f, Eden state %.2f; set 0 pushes %llu, identical ones skipped %llu%s", g_s35_ns[0] / k, g_s35_ns[1] / k, g_s35_ns[2] / k,
              (unsigned long long)g_s35_push, (unsigned long long)g_s35_skip, s35_on() ? "" : " (DM_VK_S35=0)");
        memset(g_s35_ns, 0, sizeof g_s35_ns); g_s35_push = g_s35_skip = 0;
        VKLOG("replay2: of those, vertex stream lookups kept %llu, made %llu", (unsigned long long)g_vlk_hit, (unsigned long long)g_vlk_miss); g_vlk_hit = g_vlk_miss = 0;
        VKLOG("s45: of those, pipeline key kept %llu, made %llu; Eden state kept %llu, recorded %llu; driver segments kept %llu, made %llu; mismatches %llu %llu %llu (fast paths 0x%x%s)",
              (unsigned long long)g_s45_nk[0], (unsigned long long)g_s45_nk[1], (unsigned long long)g_s45_ns[0], (unsigned long long)g_s45_ns[1], (unsigned long long)g_s45_nd[0],
              (unsigned long long)g_s45_nd[1], (unsigned long long)g_s45_bad[0], (unsigned long long)g_s45_bad[1], (unsigned long long)g_s45_bad[2], g_s45_on, g_s45_verify ? ", verified" : "");
        memset(g_s45_nk, 0, sizeof g_s45_nk); memset(g_s45_ns, 0, sizeof g_s45_ns); memset(g_s45_nd, 0, sizeof g_s45_nd);
        pd_log();   // cpudiag (DM_VK_PAYDIAG=1)
        memset(s, 0, sizeof g_rec_e->prof_s);
    }
}
// R6/R7/R8 (parallel recording), R7: whether a program writes memory through storage (a storage image, a storage texel buffer, a storage buffer its shader writes), known
// before recording. The pipeline's own answer (eden_pipe.s64w, from its binds) needs the pipeline, whose key is the whole draw state (eden_graphics); so the answer is also
// kept per shader object (vkshader_t.eden_sw, from the binds of each stage, nvn_stage), which a dry pass reads from the program bound (S.sh) without any pipeline lookup:
// eden_prog_sw / eden_cs_sw. Filled at the first draw / dispatch of each pipeline (where s64w was computed): 0 = not known yet (a dry pass must treat the draw as one
// that writes). Transform feedback (S.sh[0]->xfb_layout with the capture on) is not a bind and is checked apart. Nothing reads eden_sw yet: no behaviour change
static int eden_pipe_sw(const nvn_eden_pipeline_info *pi) {   // 2 = a bind writes storage, 1 = none (s64w's rule, was inline in eden_draw)
    int w = 0; for (uint32_t i = 0; i < pi->num_binds; i++) { const nvn_eden_bind *b = &pi->binds[i];
        w |= b->kind == NVN_EB_IMG || b->kind == NVN_EB_IMG_TBUF || (b->kind == NVN_EB_SSBO && b->is_written); }
    return w ? 2 : 1;
}
static void eden_sw_set(vkshader_t *s, int v);   // S1c (parallel recording): below
static void eden_sw_note(const nvn_eden_pipeline_info *pi, vkshader_t *const sh[6]) {   // graphics: the five NVN stages' shaders of the program the pipeline was made from
    int lost = 0;
    for (uint32_t i = 0; i < pi->num_binds; i++) { const nvn_eden_bind *b = &pi->binds[i];
        if (!(b->kind == NVN_EB_IMG || b->kind == NVN_EB_IMG_TBUF || (b->kind == NVN_EB_SSBO && b->is_written))) continue;
        if (b->nvn_stage < 5 && sh[b->nvn_stage]) eden_sw_set(sh[b->nvn_stage], 2); else lost = 1; }
    for (int st = 0; st < 5; st++) if (sh[st]) eden_sw_set(sh[st], lost ? 2 : 1);   // lost: a writing bind of no stage present here: every shader of the program counted as writing (conservative)
}
// S1c (parallel recording): eden_sw is shared by every thread that draws with the shader: sticky at 2 (an atomic store of 2 is idempotent), 1 only over 0 (compare and
// swap: a 2 stored meanwhile stays), read atomically (the dry pass reads it while workers may write it)
static void eden_sw_set(vkshader_t *s, int v) {
    if (v == 2) { __atomic_store_n(&s->eden_sw, 2, __ATOMIC_RELAXED); return; }
    int z = 0; (void)__atomic_compare_exchange_n(&s->eden_sw, &z, v, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
}
__attribute__((unused)) static int eden_prog_sw(vkshader_t *const sh[6]) {   // 2 the program writes storage, 1 it does not, 0 not known yet
    int unk = 0; for (int st = 0; st < 5; st++) if (sh[st]) { const int w = __atomic_load_n(&sh[st]->eden_sw, __ATOMIC_RELAXED); if (w == 2) return 2; if (!w) unk = 1; }
    return unk ? 0 : 1;
}
__attribute__((unused)) static int eden_cs_sw(const vkshader_t *cs) { return cs ? __atomic_load_n(&cs->eden_sw, __ATOMIC_RELAXED) : 0; }   // a compute shader's (eden_dispatch_prep fills it)
// S1c (parallel recording): a pipeline's s64w, computed once: the thread whose compare and swap stores it (0 -> 1 / 2) is the one that does the first use's work
// (eden_sw_note / the compute shader's eden_sw), as the first draw or dispatch did; the others read the value (acquire). Returns s64w; *first: this call stored it
static int eden_pipe_sw_once(eden_pipe *ep, int *first) {
    int w = __atomic_load_n(&ep->s64w, __ATOMIC_ACQUIRE); *first = 0;
    if (w) return w;
    const int v = eden_pipe_sw(&ep->pi); int z = 0;
    if (__atomic_compare_exchange_n(&ep->s64w, &z, v, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) { *first = 1; return v; }
    return z;
}
// S1c (parallel recording): the vertex shader's inputs (vtxpull) published once: the thread whose compare and swap claims vpull_known (0 -> 2) writes vpull_inputs, then
// vpull_known = 1 with release (vpull_decide acquires it and only then reads the inputs); 1: this call published them
static int vpull_inputs_publish(vkshader_t *s, uint32_t inputs) {
    int z = 0;
    if (__atomic_load_n(&s->vpull_known, __ATOMIC_ACQUIRE) || !__atomic_compare_exchange_n(&s->vpull_known, &z, 2, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return 0;
    __atomic_store_n(&s->vpull_inputs, inputs, __ATOMIC_RELAXED); __atomic_store_n(&s->vpull_known, 1, __ATOMIC_RELEASE); return 1;
}
// S1c (parallel recording): pkm2's mark that a shader writes gl_Layer, set once (compare and swap; draw_layered reads it atomically); 1: this call set it
static int layer_out_mark(vkshader_t *s) {
    int z = 0; return !__atomic_load_n(&s->layer_out, __ATOMIC_RELAXED) && __atomic_compare_exchange_n(&s->layer_out, &z, 1, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
}
static int eden_draw(VkCommandBuffer cb, int prim, int indexed, int var, uint64_t t0) {
    struct rec_ctx *const g_rec = rec_cur(); struct rec_eden *const g_rec_e = rec_e_cur();   // R1 (parallel recording): the thread's contexts read once (S, g_bc, ... below)
    vk_rec_target *const g_vk_rt = rec_rt_cur();   // R2 (parallel recording): and its recording target (g_cmd_gen below)
    if (!S.rp_pass) { g_vkst.skip_rt++; VKRED_ONCE(168, "Eden mode: a pass begun with dynamic rendering (Eden's pipelines are made against render pass objects): its draws are skipped"); return 0; }
    uint64_t tk0 = tick();   // per-part timing (eden_prof), logged every 2^17 draws
    nvn_eden_draw_state d; eden_pipe *ep = eden_graphics(prim, &d);
    if (!ep || !eden_ready_w(ep, eden_draw_must_wait())) { g_vkst.skip_pipe++; return 0; }   // glyphwait
    int sw_first; const int s64w = eden_pipe_sw_once(ep, &sw_first);   // s64: does it write storage; S1c (parallel recording): computed once (eden_pipe_sw_once)
    if (sw_first) eden_sw_note(&ep->pi, S.sh);   // R7 (parallel recording): and which of its shaders do (at its first draw)
    if (s64w == 2 && g_wk) { seg_wk_abort(WA_OP); g_vkst.skip_pipe++; return 0; }   // S1e (parallel recording): a storage-writing draw (its GPU-write bookkeeping) is the replay thread's (the dry pass keeps them serial by R7's flag)
    if (s64w == 2) { g_s64_pwrite = 1; void vk_par0_storage_draw(void); vk_par0_storage_draw(); }   // (par0: a storage-writing draw stays on the replay thread)
    const nvn_eden_pipeline_info *pi = &ep->pi;
    // vtxpull: the vertex shader's inputs, for the next draws' decision; this draw's outcome counted
    if (prim != DM_PRIM_MESH_TASKS && S.sh[0]) vpull_inputs_publish(S.sh[0], pi->vs_inputs);   // S1c (parallel recording): once, published
    if (__atomic_load_n(&g_vpull_on, __ATOMIC_RELAXED)) {   /* S1c (parallel recording): the switch read atomically */ if (g_vpull_want && !pi->vertex_pull) g_vps_fb[VPF_PIPE]++; else if (!g_vpull_want && g_vpull_why >= 0) g_vps_fb[g_vpull_why]++; }
    if (pi->layer_stages) {   // pkm2: an Eden shader that writes gl_Layer: mark its shader object for r76 (draw_layered), Eden stage -> NVN stage 0 VS, 3 TCS, 4 TES, 2 GS
        static const int nvn_of[4] = {0, 3, 4, 2}; int fresh = 0;
        for (int i = 0; i < 4; i++) if (pi->layer_stages >> i & 1) { vkshader_t *s = S.sh[nvn_of[i]]; if (s && layer_out_mark(s)) fresh = 1; }   // S1c (parallel recording): set once
        if (fresh) { static int n; if (__atomic_fetch_add(&n, 1, __ATOMIC_RELAXED) < 16) VKLOG("pkm2: shader %016llx writes gl_Layer (Eden stages %#x): its passes are layered from the next one (r76)",
            (unsigned long long)(S.sh[0] ? S.sh[0]->key : 0), pi->layer_stages); } }
    eden_entry *const pay = g_rec_e->pay_draw;   // R1 (parallel recording): was static here
    uint64_t t1d = tick(), ns_desc0 = g_vkst.ns_desc;
    uint32_t sampled_scale_bits[4];
    const uint64_t pd_ov0 = g_pd_ov;   // s46pc
    g_pd_draw = 1; const int pay_ok = eden_payload(pi, pay, sampled_scale_bits); g_pd_draw = 0;   // cpudiag: a draw's payload (paydiag)
    if (!pay_ok) { g_vkst.skip_pipe++; return 0; }
    vsres_diag(pi);   // vsres (NXN1X_VSRES_DIAG, log only)
    g_vkst.ns_desc += tick() - t1d - (g_pd_ov - pd_ov0);
    g_vkst.draws++; if (++g_pass_draws > g_vkst.max_pass_draws) g_vkst.max_pass_draws = g_pass_draws;
    VkPipeline p = (VkPipeline)pi->pipeline;
    int s45_chain = S.bound_pipe == g_s35_lastbound, s45_same = s45_chain && p == S.bound_pipe;   // s45: no pipeline bound since the last Eden draw but by it; the same one
    if (p != S.bound_pipe) { if (S.bound_pipe != g_s35_lastbound) g_s35_pok = 0;   /* s35: a pipeline these draws did not bind came between */
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p); S.bound_pipe = p; g_r131.pipe++; }
    g_s35_lastbound = p;
    uint64_t s35a = tick();
    if (!eden_bind_set(cb, ep, pay, VK_PIPELINE_BIND_POINT_GRAPHICS)) { g_vkst.draws--; return 0; }
    fg_observe_draw();
#if defined(DM_NXN1X)
    // ootdlss2d diag (NPARK 11-21): every draw into a B10G11R11 target 1920 or wider (the 2560x1440 0x3f layer and its like), once per fragment shader and
    // what it samples: the target, the shaders and every texture of every stage with its size and format (which pass writes the layer, from what)
    if (!g_wk && P_att[0].im && P_att[0].fmt == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && P_att[0].w >= 1920) {
        static uint64_t seen[128]; static uint32_t sig_seen[128]; static int ns; char tl[480]; int tn = 0; uint32_t sig = 0, maxw = 0; tl[0] = 0;   // (draws sampling nothing 320 or wider: the HUD, left out)
        for (uint32_t i = 0; i < pi->num_binds; i++) { const nvn_eden_bind *b = &pi->binds[i]; if (b->kind != NVN_EB_TEX && b->kind != NVN_EB_IMG) continue;
            for (uint32_t k = 0; k < b->count && k < 4; k++) { const tex_t *t = NULL; const view_t *v = NULL;
                if (!vk_tex_by_id((int)(eden_handle(b->nvn_stage, b, k) & 0xfffff), &v, &t) || !t) continue;
                sig = sig * 31u + t->d.width * 7u + t->d.format; if (t->d.width > maxw) maxw = t->d.width;
                if (tn < (int)sizeof tl - 64) tn += snprintf(tl + tn, sizeof tl - (size_t)tn, " %s%d %ux%u nvn %#x%s", b->kind == NVN_EB_IMG ? "img" : "tex", b->nvn_stage, t->d.width, t->d.height,
                                                          (unsigned)t->d.format, v && v->set_format ? " (format view)" : ""); } }
        const uint64_t fs = S.sh[1] ? S.sh[1]->key : 0; int q = 0; while (q < ns && !(seen[q] == fs && sig_seen[q] == sig)) q++;
        if (q == ns && ns < 128 && maxw >= 320) { seen[ns] = fs; sig_seen[ns++] = sig;
            VKLOG("ootdlss diag: draw into %ux%u B10G11R11 (%d colour targets%s), vs %016llx fs %016llx, samples:%s (present %llu, #%d)", P_att[0].w, P_att[0].h, S.ncol, P_att[8].im ? ", depth" : "",
                  (unsigned long long)(S.sh[0] ? S.sh[0]->key : 0), (unsigned long long)fs, tn ? tl : " nothing", (unsigned long long)g_tex_frame, ns); } }
#endif
    g_s35_ns[0] += tick() - s35a;   // s35
    g_bc.lay = VK_NULL_HANDLE;   // the bundle path's set bind cache: its sets are not bound now
    // the base dynamic state as the bundle path sets it (viewport flips, scissor, stencil, bias, blend constants), every viewport the pipeline has, depth bounds
    { uint64_t s35c = tick(); set_dynamic(cb); g_s35_ns[1] += tick() - s35c; }   // s35: timed
    if (pi->num_viewports > 1 && (g_eden_vpm.gen != g_cmd_gen || g_eden_vpm.n != pi->num_viewports || memcmp(&g_eden_vpm.vp, &g_bc.vp, sizeof g_bc.vp) || memcmp(&g_eden_vpm.sc, &g_bc.sc, sizeof g_bc.sc))) {
        VkViewport vps[16]; VkRect2D scs[16]; uint32_t n = pi->num_viewports > 16 ? 16 : pi->num_viewports;
        for (uint32_t i = 0; i < n; i++) { vps[i] = g_bc.vp; scs[i] = g_bc.sc; }   // [B7: the game's other viewports (S.vp[1..]) for layered / multi-viewport draws]
        vkCmdSetViewport(cb, 1, n - 1, vps + 1); vkCmdSetScissor(cb, 1, n - 1, scs + 1);
        g_eden_vpm.gen = g_cmd_gen; g_eden_vpm.n = pi->num_viewports; g_eden_vpm.vp = g_bc.vp; g_eden_vpm.sc = g_bc.sc; }
    { float lo = S.db_lo < 0 ? 0 : S.db_lo > 1 ? 1 : S.db_lo, hi = S.db_hi < 0 ? 0 : S.db_hi > 1 ? 1 : S.db_hi; s45_dbounds(cb, lo, hi, s45_chain); }
    // the extended state: Eden's, with the replay's winding and depth handling (prepare_draw's pkey rules)
    int has_depth = (S.daspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0, has_stencil = (S.daspect & VK_IMAGE_ASPECT_STENCIL_BIT) != 0;
    int dt = has_depth && S.ds.dt, dw = S.ds.dw, df = S.ds.df, bias_en = (S.poly.offset_en & (S.poly.mode == 0 ? 1 : S.poly.mode == 1 ? 2 : 4)) != 0, deq = 0;
    if (var == 2) bias_en = 0;
    if (dt && dw) { if (df == 2 || df == 4) g_deq_dir = 1; else if (df == 5 || df == 7) g_deq_dir = 2; }
    if (g_deq > 0 && g_deq_dir && dt && !dw && df == 3 && !bias_en) { deq = g_deq_dir; df = deq == 1 ? 4 : 7; bias_en = 1; }   // r57
    VkPrimitiveTopology tp = topo(prim);
    nvn_eden_dyn o; memset(&o, 0, sizeof o);
    o.cull_mode = (int32_t)(g_no_cull ? VK_CULL_MODE_NONE : eden_cull(S.poly.cull));
    o.front_face = ((S.poly.front == NVN_FRONT_FACE_CCW) != (g_flipy == 0)) ^ g_queue_ll ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;   // pkm8 (gfx_vk_exec.c k.front)
    o.depth_test = dt; o.depth_write = dw; o.depth_compare = df >= 1 && df <= 8 ? df - 1 : VK_COMPARE_OP_ALWAYS;   // NVN NEVER..ALWAYS = 1..8
    o.stencil_test = has_stencil && S.ds.st; o.depth_bounds_test = has_depth && S.db_on && V.has_dbounds && !g_x_nodb;
    o.depth_bias_enable = bias_en;
    o.primitive_restart = S.prim_restart && (tp == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP || tp == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP || tp == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN) && indexed;
    o.rasterizer_discard = (var == 8) | S.rast_discard; o.mask_unwritten = !g_x_allmask;
    if (pass_scale == 2 && !o.rasterizer_discard) {
        scale_note_scissor(g_bc.sc);
        for (int i = 0; i < S.ncol; ++i)
            if (d.channel_mask[i] && (!o.mask_unwritten || !(pi->dyn & (NVN_ED_EDS3_BLEND | NVN_ED_COLOR_WRITE_ENABLE)) ||
                                      (pi->frag_outputs & (1u << i))))
                scale_writes[i] |= VK_IMAGE_ASPECT_COLOR_BIT;
        if (dt && dw) scale_writes[8] |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if (o.stencil_test &&
            ((S.swmask[0] && (S.ds.fail_f != NVN_STENCIL_OP_KEEP || S.ds.zpass_f != NVN_STENCIL_OP_KEEP || S.ds.zfail_f != NVN_STENCIL_OP_KEEP)) ||
             (S.swmask[1] && (S.ds.fail_b != NVN_STENCIL_OP_KEEP || S.ds.zpass_b != NVN_STENCIL_OP_KEEP || S.ds.zfail_b != NVN_STENCIL_OP_KEEP))))
            scale_writes[8] |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    memcpy(o.scaled_textures, sampled_scale_bits, sizeof sampled_scale_bits);
    o.scaled_target = pass_scale == 2;
    o.render_area[0] = (float)S.pass_w; o.render_area[1] = (float)S.pass_h;
    if (g_tracing || (S.sh[0] && S.sh[0]->key == 0x7fa0c6373aca1086ull)) { static int n; if (g_tracing || __atomic_fetch_add(&n, 1, __ATOMIC_RELAXED) < 6) {   // pkm6 (diagnostic) (pkm7: every traced one) (S1c (parallel recording): atomic)
        VKLOG("pkm6: Eden draw vs %016llx fs %016llx pass %ux%u: depth test %d write %d func %d (NVN %d, r57 %d) has depth %d stencil %d; stencil test %u func %u/%u ops %u %u %u ref %#x/%#x masks %#x/%#x write %#x/%#x; cull %d front %d (NVN cull %u front %u, lower-left %d flipy %d)",
              (unsigned long long)(S.sh[0] ? S.sh[0]->key : 0), (unsigned long long)(S.sh[1] ? S.sh[1]->key : 0), S.pass_w, S.pass_h, o.depth_test, o.depth_write, o.depth_compare, S.ds.df, deq, has_depth, has_stencil,
              d.stencil_test, d.stencil_func[0], d.stencil_func[1], d.stencil_fail[0], d.stencil_zfail[0], d.stencil_zpass[0], (unsigned)S.sref[0], (unsigned)S.sref[1], (unsigned)S.smask[0], (unsigned)S.smask[1], (unsigned)S.swmask[0], (unsigned)S.swmask[1],
              o.cull_mode, o.front_face, S.poly.cull, S.poly.front, g_lower_left, g_flipy);
        VKLOG("pkm6:   blend %#x (0: %u %u %u / %u %u %u) masks %x %x %x %x, colours %u (%d %d %d), depth fmt %d, alpha func %u ref %g, a2c %u, logic %u/%u, discard %d, depth clamp %u, depth range %g..%g, depth -1..1 %u, frag outputs %#x, dyn %#x, var %d",
              d.blend_enable, d.blend[0].func_src, d.blend[0].func_dst, d.blend[0].eq, d.blend[0].func_src_a, d.blend[0].func_dst_a, d.blend[0].eq_a, d.channel_mask[0], d.channel_mask[1], d.channel_mask[2], d.channel_mask[3],
              d.num_colors, d.color_format[0], d.color_format[1], d.color_format[2], d.depth_format, d.alpha_func, (double)d.alpha_ref, d.alpha_to_coverage, d.logic_op_enable, d.logic_op, o.rasterizer_discard, d.depth_clamp,
              (double)S.dnear, (double)S.dfar, d.depth_mode_minus_one_to_one, pi->frag_outputs, pi->dyn, var);
        VKLOG("pkm7:   write masks as set: %x %x %x (outputs the fragment stage does not write masked: %d), scissor %d %d %d %d, viewport %.1f %.1f %.1f %.1f", d.channel_mask[0] & (o.mask_unwritten && !(pi->frag_outputs & 1) ? 0 : 0xf),
              d.channel_mask[1] & (o.mask_unwritten && !(pi->frag_outputs & 2) ? 0 : 0xf), d.channel_mask[2] & (o.mask_unwritten && !(pi->frag_outputs & 4) ? 0 : 0xf), o.mask_unwritten,
              S.sc.x, S.sc.y, S.sc.w, S.sc.h, S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3]); } }
    { uint64_t s35d = tick();
      // s45 (DM_VK_S45 & 2): the same Eden pipeline as the Eden draw before, nothing else bound since, the same command buffer part, draw state and overrides
      int s45 = (g_s45_on & 2) && s45_same && g_s45_st.ok && g_s45_st.gen == g_cmd_gen && g_s45_st.p == ep->p && g_s45_st.dver == g_s45_dver && !memcmp(&g_s45_st.o, &o, sizeof o);
      if (s45 && !g_s45_verify) g_s45_ns[0]++;
      else {
          if (s45) { g_s45_ns[0]++; if (memcmp(&g_s45_st.d, &d, sizeof d) && ++g_s45_bad[1] <= 8) VKLOG("[RED] s45: Eden state mismatch %llu: the draw state changed under the same version", (unsigned long long)g_s45_bad[1]); }
          else g_s45_ns[1]++;
          nvn_eden_cmd_state(cb, ep->p, &d, &o);
          g_s45_st.ok = 1; g_s45_st.gen = g_cmd_gen; g_s45_st.p = ep->p; g_s45_st.dver = g_s45_dver; g_s45_st.o = o; g_s45_st.d = d;
      }
      g_s35_ns[2] += tick() - s35d; }   // s35: timed
    if (var == 1) { vkCmdSetDepthBias(cb, 0, 0, 0); memset(&g_bc.po, 0xff, sizeof g_bc.po); }
    if (deq) { vkCmdSetDepthBias(cb, deq == 1 ? (float)-g_deq : (float)g_deq, 0, 0); memset(&g_bc.po, 0xff, sizeof g_bc.po); }   // r57: the next draw sets its own again
    uint64_t tk2 = tick();
    if (pi->vertex_pull) {   // vtxpull: no vertex buffers; the decided streams' fetch table, in host-visible memory with a device address (as descriptor buffers' copies)
        VkBuffer tb; VkDeviceSize to; VkDeviceAddress ta; uint8_t *t = eden_ring_alloc(g_rec_e->uring, sizeof g_vpull_tab, 16, &tb, &to, &ta);
        if (!t) { g_vps_fb[VPF_RING]++; VKRED_ONCE(190, "vtxpull: no memory for a fetch table: the pulled draw is skipped"); g_vkst.draws--; return 0; }
        memcpy(t, g_vpull_tab, sizeof g_vpull_tab); nvn_eden_cmd_vertex_table(cb, ep->p, ta + to); g_vps_pulled++;
        if (g_vb_on) for (uint32_t s = 0; s < 16; s++) if (g_vpull_tab[s].addr) vb_note(0, S.vbuf[s].address, S.vbuf[s].size);   // vbytes (log only)
        goto vpull_done;
    }
    // vertex buffers: the streams of the attributes the vertex shader loads
    uint32_t streams = 0;
    for (uint32_t a = 0; a < d.num_attribs; a++) if (pi->vs_inputs >> d.attrib[a].location & 1) streams |= 1u << d.attrib[a].stream;
    if (!streams && (pi->dyn & NVN_ED_STRIDE)) streams = 1; // establish dynamic state even for procedural vertices
    // replay2 (DM_VK_DESC_DELTA, default on): a stream's buffer lookup (vk_buf_at, and vbm_bind's mirror) kept per stream while its address and size, the command
    // buffer part (so the submission), the submission's GPU writes so far (wseq, raw logs: vbm_bind's "written earlier in this submission" import and the mirrors'
    // stale blocks), the pools (g_pool_gen) and the mirrors' stale count are unchanged: vbm_bind looks at a block's CPU writes once per submission (csub), so a
    // second lookup in the same submission finds what the first found. OoT 11-21 (rptfix): vertex buffers 0.47 us a draw at ~3,100 draws a frame, the bind
    // itself already skipped when unchanged (g_eden_vbm below). Not with the vbytes / vtxmir diagnostics
    typeof(g_rec_e->vlk[0]) *const vlk = g_rec_e->vlk;   // R1 (parallel recording): was static here
    static int dd = -1; if (dd < 0) { const char *e = getenv("DM_VK_DESC_DELTA"); dd = !(e && *e == '0'); VKLOG("replay2: vertex stream lookups %s (DM_VK_DESC_DELTA=%d)", dd ? "kept while nothing they depend on changed" : "made at every draw", dd); }
    uint64_t vk_sub_wver(void); uint32_t vk_pool_gen(void); extern uint64_t g_vbm_stale;
    const int vmemo = dd && !g_vb_on && !g_vmir; const uint64_t wver = vmemo ? vk_sub_wver() : 0; const uint32_t pgen = vmemo ? vk_pool_gen() : 0;
    for (uint32_t s = 0; s < 16; s++) if (streams >> s & 1) {
        VkBuffer buf; VkDeviceSize off;
        if (vmemo && vlk[s].gen == g_cmd_gen && vlk[s].a == S.vbuf[s].address && vlk[s].n == S.vbuf[s].size && vlk[s].wver == wver && vlk[s].pgen == pgen &&
            vlk[s].stale == g_vbm_stale && vlk[s].var == g_var_now) { buf = vlk[s].b; off = vlk[s].o; g_vlk_hit++; }
        else {
        if (!((S.vbuf[s].size && vk_buf_at(S.vbuf[s].address, S.vbuf[s].size, &buf, &off)) || vk_buf_at(S.vbuf[s].address, 1, &buf, &off))) { buf = vk_zero_buffer(); off = 0; if (S.vbuf[s].address) VKRED_ONCE(51, "vertex buffer outside the memory pools"); }
        else {
            if (g_vb_on) vb_note(0, S.vbuf[s].address, S.vbuf[s].size);   // vbytes (NXN1X_VBYTES, log only)
            if (g_vmir && S.vbuf[s].size) { VkBuffer mb; VkDeviceSize mo; if (vmir_put(S.vbuf[s].address, S.vbuf[s].size, &mb, &mo)) { buf = mb; off = mo; } }   // vtxmir A/B
            else if (S.vbuf[s].size) vbm_bind(S.vbuf[s].address, S.vbuf[s].size, &buf, &off); }   // vbmir
        if (g_var_now == 5) { buf = vk_zero_buffer(); off = 0; }
        if (vmemo) { vlk[s].gen = g_cmd_gen; vlk[s].a = S.vbuf[s].address; vlk[s].n = S.vbuf[s].size; vlk[s].wver = vk_sub_wver(); vlk[s].pgen = vk_pool_gen();
            vlk[s].stale = g_vbm_stale; vlk[s].var = g_var_now; vlk[s].b = buf; vlk[s].o = off; g_vlk_miss++; }
        }
        if (dbg_draw() & 2) continue;
        if (g_eden_vbm[s].gen == g_cmd_gen && g_eden_vbm[s].b == buf && g_eden_vbm[s].o == off && g_eden_vbm[s].stride == S.vstream[s].stride) continue;
        void *bp = (void *)buf; uint64_t o64 = off; nvn_eden_cmd_vertex_buffers(cb, ep->p, &d, s, 1, &bp, &o64); g_r131.vb++;
        g_eden_vbm[s].gen = g_cmd_gen; g_eden_vbm[s].b = buf; g_eden_vbm[s].o = off; g_eden_vbm[s].stride = S.vstream[s].stride;
        g_bc.vb[s] = buf; g_bc.vo[s] = off;
    }
vpull_done:
    g_r131_dupok = 0; g_r142_ok = 0; g_r132_instok = 0; g_pofs_st[0] = g_pofs_st[1] = 0; g_irun_on = 0; g_r152_cand = 0;   // the diagnostic classes read the bundle's reflection: off
    uint64_t tk3 = tick(), tdesc = g_vkst.ns_desc - ns_desc0; g_vkst.ns_pipe += tk3 - t0 - tdesc - (g_pd_ov - pd_ov0);   /* s46pc: paydiag's hashing in neither */   // as the bundle path: without the descriptor sets' time
    eden_prof(tk0 - t0, t1d - tk0, tk2 - t1d - tdesc, tk3 - tk2);
    if (g_wk) ck_mark(cb, 4, S.sh[0] ? S.sh[0]->key : 0, S.sh[1] ? S.sh[1]->key : 0, (uint32_t)g_pass_draws, (uint32_t)indexed | (uint32_t)(pi->vertex_pull != 0) << 1 | (uint32_t)prim << 4);   // parck: a segment's draw (gpuck names the one the GPU died in)
    return 1;
}

// s64: 1 when the draw about to be made reads (through any descriptor of its Eden pipeline) memory of the open pass's attachments, or when that cannot
// be told (the pipeline not ready, a handle naming no texture). Compared by guest memory range, so other views, levels, layers and aliases count too.
static int s64_hits_att(uint64_t a, uint64_t n) {
    if (!a) return 0; if (!n) n = 1;
    for (int i = 0; i < 8; i++) if (S.rt_has_col[i]) { uint64_t m = (uint64_t)(uintptr_t)S.rt_col[i].mem, z = S.rt_col[i].L.size; if (m && a < m + (z ? z : 1) && m < a + n) return 1; }
    if (S.rt_has_dep) { uint64_t m = (uint64_t)(uintptr_t)S.rt_dep.mem, z = S.rt_dep.L.size; if (m && a < m + (z ? z : 1) && m < a + n) return 1; }
    return 0;
}
static int draw_uses_only_uniforms(int prim) {
    nvn_eden_draw_state state;
    eden_pipe *ep = eden_graphics(prim, &state);
    if (!ep || !eden_ready(ep)) return 0;
    for (uint32_t i = 0; i < ep->pi.num_binds; ++i)
        if (ep->pi.binds[i].kind != NVN_EB_UBO) return 0;
    return 1;
}
static int flat_texture_range(const tex_t *t) {
    return t && t->mem && t->L.size && t->L.size <= UINTPTR_MAX - (uintptr_t)t->mem &&
           t->pool && !(t->pool->flags & (NVN_MEMORY_POOL_FLAGS_PHYSICAL | NVN_MEMORY_POOL_FLAGS_VIRTUAL)) &&
           !(t->d.flags & NVN_TEXTURE_FLAGS_SPARSE);
}

static int s64_draw_feedback(int prim, int sampled_only) {
    if (sampled_only) {
        if (g_tf_on || g_s64_pwrite) return 1;
        for (int i = 0; i < 8; ++i)
            if (S.rt_has_col[i] && !flat_texture_range(&S.rt_col[i])) return 1;
        if (S.rt_has_dep && !flat_texture_range(&S.rt_dep)) return 1;
    }
    nvn_eden_draw_state d; eden_pipe *ep = eden_graphics(prim, &d);
    if (!ep || !eden_ready(ep)) return 1;
    const nvn_eden_pipeline_info *pi = &ep->pi;
    for (uint32_t i = 0; i < pi->num_binds; i++) {
        const nvn_eden_bind *b = &pi->binds[i]; int st = b->nvn_stage; if (st < 0 || st > 5) return 1;
        if (sampled_only && b->kind != NVN_EB_UBO && b->kind != NVN_EB_TEX) return 1;
        for (uint32_t k = 0; k < b->count; k++) {
            if (b->kind == NVN_EB_UBO) break;   // constant buffers are not attachments
            if (b->kind == NVN_EB_SSBO) {
                if (b->cbuf == 0xFE) continue;
                if (b->cbuf == 0xFF) return 1;
                uint32_t off = b->offset + 16 * k;
                uint64_t a = (uint64_t)eden_cbuf_word(st, b->cbuf, off) | (uint64_t)eden_cbuf_word(st, b->cbuf, off + 4) << 32, n = eden_cbuf_word(st, b->cbuf, off + 8);
                if (s64_hits_att(a, n)) return 1;
                continue; }
            uint64_t h = eden_handle(st, b, k); int tid = (int)(h & 0xfffff); if (!tid) continue;   // nothing bound: the replay binds a stand-in
            const view_t *v = NULL; const tex_t *t = NULL;
            if (!vk_tex_by_id(tid, &v, &t) || !t) return 1;
            if (sampled_only) {
                if (!flat_texture_range(t) || t->d.target == NVN_TEXTURE_TARGET_BUFFER) return 1;
                vkimg_t *resident = vk_tex_peek(t);
                if (!resident) return 1;
                for (unsigned n = 0; n < 9; ++n)
                    if (P_att[n].im == resident) return 1;
            }
            if (s64_hits_att((uint64_t)(uintptr_t)t->mem, t->L.size)) return 1;
        }
    }
    return 0;
}
// bak86: a dispatch's first-invocation stores, taken out of its shader (vk_pipeline_cache.cpp FirstStorePass), made here in order before it: each 4-byte value at
// the 64-bit address the dispatch's constant buffer holds, plus its byte offset, after every earlier use of that memory. An address in no buffer of the host's is
// logged and skipped (the shader no longer makes the store: the list's reset is lost for that dispatch)
uint64_t g_fs_pre, g_fs_miss;   // host pre-stores made, and not made (no buffer) (gfx_vk_core.c 30 s line)
static void eden_first_stores(VkCommandBuffer cb, const nvn_eden_pipeline_info *pi, uint64_t key) {
    if (!pi->nfirst) return;
    VkMemoryBarrier m = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &m, 0, NULL, 0, NULL);
    for (uint32_t i = 0; i < pi->nfirst && i < 4; i++) {
        const uint32_t *f = pi->first[i];
        uint64_t a = (uint64_t)eden_cbuf_word(5, f[0], f[1]) | (uint64_t)eden_cbuf_word(5, f[0], f[1] + 4) << 32; VkBuffer b; VkDeviceSize o;
        if (!a || !vk_buf_at(a + f[2], 4, &b, &o) || (o & 3)) { g_fs_miss++;
            static int n; if (n++ < 8) VKLOG("[RED] first_store: dispatch %016llx: its list at %#llx (+%u) is in no buffer of the host's: the store of %u not made", (unsigned long long)key, (unsigned long long)a, f[2], f[3]);
            continue; }
        { int vk_hy3_redirect(uint64_t, uint64_t, VkBuffer *, VkDeviceSize *); vk_hy3_redirect(a + f[2], 4, &b, &o); }   // stage2s: into staging where the range is staged
        vkCmdUpdateBuffer(cb, b, o, 4, &f[3]); g_fs_pre++;
    }
    static uint64_t said[8]; static int ns; int known = 0; for (int i = 0; i < ns && !known; i++) known = said[i] == key;
    if (!known && ns < 8) { said[ns++] = key; VKLOG("first_store: compute shader %016llx: %u first-invocation stores made by the host before each dispatch (DM_NVN2_FIRST_STORE=0 off)", (unsigned long long)key, pi->nfirst); }
}
// ---- a dispatch (dispatch() in Eden mode): prepare before its barrier, bind after ----
typedef struct { eden_pipe *ep; eden_entry *pay; nvn_eden_dyn dyn; } eden_cs_t;
// Temporal compute passes can read both scene inputs without ever attaching them
// together. Inspect only resolved, unambiguous full-image descriptors on this dispatch.
static void fg_compute_inputs(uint64_t shader, const eden_cs_t *c) {
    if (!nvn_fg_collecting() || nvn_fg_get_profile() || !fg_auto.camera ||
        fg_auto.frame != g_tex_frame + 1 || S.pass_open == 1) return;
    const nvn_eden_pipeline_info *pi = &c->ep->pi;
    vkimg_t *actual[2] = {0};
    nvn_fg_image pair[2];
    unsigned velocity_slot = 0;
    for (uint32_t i = 0; i < pi->num_binds; ++i) {
        const nvn_eden_bind *b = &pi->binds[i];
        if (b->kind != NVN_EB_TEX || b->count != 1 || b->first_entry >= EDEN_MAXE ||
            b->dim != 1 || b->arrayed || b->is_ms) continue;
        const tex_t *tex = NULL;
        const view_t *view = NULL;
        int id = (int)(eden_handle(5, b, 0) & 0xfffff);
        if (!id || !vk_tex_by_id(id, &view, &tex) || !tex || !fg_base_attachment(view)) continue;
        vkimg_t *im = vk_tex_peek(tex);
        if (!im || (!(im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && !fg_velocity_format(im->fmt))) continue;
        const VkImageView bound = c->pay[b->first_entry].image.imageView;
        vkimg_t *base = NULL;
        VkImageView native = vk_tex_view(tex, view, 1, 0, VK_VIEW_SAMPLED, &base);
        if (!bound || base != im) continue;
        if (bound != native) {
            if (!im->scaled_valid || !im->scaled ||
                vk_rescaled_view(im, tex, view, 1, 0, 0) != bound) continue;
            im = im->scaled;
        }
        if (im->levels != 1 || im->layers != 1 || im->samples != 1 || im->w < 320 ||
            im->layout != VK_IMAGE_LAYOUT_GENERAL || !(im->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) continue;
        unsigned role = (im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? 0 : 1;
        if (actual[role] && actual[role] != im) return;
        actual[role] = im;
        pair[role] = (nvn_fg_image){im->img, im->mem, bound, im->w, im->h, im->fmt,
                                    im->layout, im->usage, im->cflags, 0, 0};
        if (role) velocity_slot = b->first_entry + 8;
    }
    if (!actual[0] || !actual[1] || actual[0]->w != actual[1]->w || actual[0]->h != actual[1]->h) return;
    ++fg_counts.motion_passes;
    fg_submit_pair(NULL, shader, velocity_slot, actual, pair, 1.f);
}
// upsprobe (11-21, NPARK: route OoT's upscale to NVIDIA DLSS SR; log only, DM_VK_UPS_PROBE=<present>, off by default): every compute dispatch of the 2 presents from
// that one that binds a sampled or storage 2D texture of 1280 px or wider, or 16:9 at 480 px or more, logs each texture and image binding (id, size, nvn format,
// depth or colour, written by the GPU, the present it was last an attachment), the first 16 words of its c0 constants (jitter, rectangles, exposure) and its
// groups: the inputs the game's network reads (colour, depth, motion vectors, exposure, jitter) and the image it writes, so DLSS gets them from evidence
static void ups_probe(const vkshader_t *cs, const eden_cs_t *c) {
    static long gen = -1; static uint64_t at, win = ~0ull; static int nl, nwin;   // every <at> presents, 2 presents logged, up to 8 windows (so one falls in gameplay)
    if (gen != dm_env_gen) { gen = dm_env_gen; const char *e = getenv("DM_VK_UPS_PROBE"); at = e && *e ? strtoull(e, NULL, 10) : 0; nl = 0; nwin = 0; }
    if (!at || g_tex_frame < at || g_tex_frame % at >= 2 || nwin > 8) return;
    if (win != g_tex_frame / at) { win = g_tex_frame / at; nl = 0; nwin++; if (nwin > 8) return; }
    if (nl >= 150) return;
    const nvn_eden_pipeline_info *pi = &c->ep->pi; char t[1400]; int o = 0, big = 0;
    for (uint32_t i = 0; i < pi->num_binds && o < (int)sizeof t - 200; i++) {
        const nvn_eden_bind *b = &pi->binds[i]; if (b->kind != NVN_EB_TEX && b->kind != NVN_EB_IMG) continue;
        for (uint32_t k = 0; k < b->count && k < 4 && o < (int)sizeof t - 200; k++) {
            const tex_t *tex = NULL; const view_t *view = NULL; const int id = (int)(eden_handle(5, b, k) & 0xfffff);
            if (!id || !vk_tex_by_id(id, &view, &tex) || !tex) { o += snprintf(t + o, sizeof t - (size_t)o, " [%s%u.%u none]", b->kind == NVN_EB_TEX ? "tex" : "img", i, k); continue; }
            const tex_t tc = *tex; const vkimg_t *im = vk_tex_peek(&tc);
            const uint32_t w = tc.d.width, h = tc.d.height;
            if (w >= 1280 || (h >= 480 && (w * 9 > h * 16 ? w * 9 - h * 16 : h * 16 - w * 9) <= 16)) big = 1;
            o += snprintf(t + o, sizeof t - (size_t)o, " [%s%u.%u id %d %ux%u nvn %#x%s%s gpu_w %d rt@%lld]", b->kind == NVN_EB_TEX ? "tex" : "img", i, k, id, w, h, (unsigned)tc.d.format,
                          im && (im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? " depth" : "", b->kind == NVN_EB_TEX && b->is_depth ? " cmp" : "", im ? im->gpu_w : -1, im ? (long long)im->wpres - 1 : -1LL);
        }
    }
    if (!big) return;
    nl++;
    char cw[220]; int q = 0; for (uint32_t w = 0; w < 16; w++) { const uint32_t v = eden_cbuf_word(5, 0, 4 * w); float fv; memcpy(&fv, &v, 4); q += snprintf(cw + q, sizeof cw - (size_t)q, " %g", (double)fv); }
    VKLOG("upsprobe: present %llu dispatch %016llx groups %ux%ux%u:%s; c0 floats:%s", (unsigned long long)g_tex_frame, (unsigned long long)cs->key, g_dispatch[0], g_dispatch[1], g_dispatch[2], t, cw);
}
// before the barrier (with g_drv_fresh as dispatch() sets it: the c0 segment the indirect counts are copied into); 0: skipped
static int eden_dispatch_prep(vkshader_t *cs, eden_cs_t *c) {
    eden_entry *const pay = g_rec_e->pay_disp;   // R1 (parallel recording): was static here
    csd_report();   // csdiag
    hy3_dispatch_begin(g_dispatch_ind ? g_dispatch_indaddr : 0);   // stage1/2s: this dispatch's bindings (its indirect arguments in the window: it closes it)
    c->pay = pay; c->ep = cs->eden ? eden_compute(cs) : NULL;
    if (!c->ep || !eden_ready_w(c->ep, 1)) return 0;   // glyphwait: a dispatch's writes persist: never skipped for its pipeline
    { int first; const int w = eden_pipe_sw_once(c->ep, &first); if (first) eden_sw_set(cs, w); }   // R7 (parallel recording): a compute pipeline's (only eden_cs_sw reads it); S1c: once, atomically
    g_eden_cs_key = cs->key; const int r = eden_payload(&c->ep->pi, pay, c->dyn.scaled_textures); if (r) csres_diag(&c->ep->pi, cs->key); g_eden_cs_key = 0;   // bak91c; csres (log only)
    if (r) fg_compute_inputs(cs->key, c);
    if (r) ups_probe(cs, c);   // upsprobe (DM_VK_UPS_PROBE, log only)
    return r;
}
static int eden_dispatch_bind(VkCommandBuffer cb, eden_cs_t *c) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)c->ep->pi.pipeline);
    if (!eden_bind_set(cb, c->ep, c->pay, VK_PIPELINE_BIND_POINT_COMPUTE)) return 0;
    nvn_eden_cmd_state(cb, c->ep->p, NULL, &c->dyn);   // its push constants
    return 1;
}
