// Android port, Vulkan backend: the game's shaders (see gfx_vk.h).
// The bundle (dm/android/tools/shader_vk.py) is mapped read-only: entries sorted by key, each the SPIR-V of one Switch shader and its binary reflection.
// The translator numbers every stage's bindings from 0 in set 0; at module creation each stage's bindings move to a set of its own (vertex 0, fragment 1,
// geometry 2, tessellation 3 / 4; compute 0) by rewriting the DescriptorSet decorations, so a program's stages never collide.
#include "gfx_vk.h"
#include <sys/clonefile.h>
#include "nvn_hmap.h"
#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <time.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct __attribute__((packed)) { uint64_t key; uint32_t stage, spv_off, spv_len, refl_off, refl_len, hi; } bent;   // hi: bits 32-47 of spv_off (low half) and refl_off (high half), 0 under 4 GB
static inline uint64_t bent_spv(const bent *e) { return e->spv_off | (uint64_t)(e->hi & 0xffff) << 32; }
static inline uint64_t bent_refl(const bent *e) { return e->refl_off | (uint64_t)(e->hi >> 16) << 32; }
static const uint8_t *g_bundle; static uint32_t g_nent; static const bent *g_ent;
void vk_shaders_open(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC); struct stat st;
    if (fd < 0 || fstat(fd, &st)) { VKLOG("[RED] no shader bundle at %s: nothing will be drawn", path); if (fd >= 0) close(fd); return; }
    g_bundle = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0); close(fd);
    if (g_bundle == MAP_FAILED || memcmp(g_bundle, "DMSV", 4)) { VKLOG("[RED] %s is not a shader bundle", path); g_bundle = NULL; return; }
    memcpy(&g_nent, g_bundle + 8, 4); g_ent = (const bent *)(g_bundle + 12);
    VKLOG("shader bundle %s: %u shaders, %.1f MB", path, g_nent, st.st_size / 1048576.0);
}
static void bundle_open_once(void) {
    const char *p = getenv("DM_VK_SHADERS"); char buf[1024];
    if (!p) { const char *r = getenv("DM_ROOT"); snprintf(buf, sizeof buf, "%s/shaders.dmsv", r ? r : "."); p = buf; }
    vk_shaders_open(p);
}
// once, and every caller waits until it is open: the replay and the prewarm threads both look shaders up at start, and a lookup that ran before the bundle
// was open cached its shader as missing for good (every draw skipped as "no program" in some runs)
static void bundle_auto(void) { static pthread_once_t once = PTHREAD_ONCE_INIT; pthread_once(&once, bundle_open_once); }
static const bent *find(uint64_t key) {
    uint32_t lo = 0, hi = g_nent;
    while (lo < hi) { uint32_t m = (lo + hi) / 2; uint64_t k; memcpy(&k, &g_ent[m].key, 8); if (k < key) lo = m + 1; else hi = m; }
    if (lo < g_nent) { uint64_t k; memcpy(&k, &g_ent[lo].key, 8); if (k == key) return &g_ent[lo]; }
    return NULL;
}
struct nvn_eden_shader *vk_shader_eden(uint64_t key);
int vk_eden_mode(void);
int vk_shader_stage(uint64_t key) {   // the NVN stage the bundle has this shader as (Eden mode: found in its code area)
    if (vk_eden_mode()) return nvn_eden_shader_stage(vk_shader_eden(key));
    bundle_auto(); const bent *e = g_bundle ? find(key) : NULL; return e ? (int)e->stage : -1; }

// Run-time capture: a shader a draw needs that the bundle does not have is saved as DM_ROOT/captured_shaders/<key>.bin ("DMSC", u32 control length, u32 code length, the control
// section as its key hashed it, the code area), and the porter translates the folder into the bundle at the next port (porter/pipeline.py, shader_vk.py --capture).
// Xenoblade Chronicles 3 sets hundreds of programs whose code is in no game file (GLSLC outputs of its Firehawk effects, and vertex / pixel shaders of models: a machine near
// the camera in Everlight Plain drew as a black silhouette, its depth pre-pass drawn and its G-buffer draw skipped); 51 of them in the opening and the first field.
// The game may free a control section once SetShaders returns, so SetShaders keeps a copy of each one the bundle lacks (shader_set); the file is written only when a draw misses
// the key (vk_shader_missing) and the code area then still hashes to it
typedef struct { uint64_t key, data; uint32_t nctl, ncode; uint8_t *ctl; } pend_t;
enum { NPEND = 4096 };
static pend_t g_pend[NPEND]; static unsigned g_npend; static pthread_mutex_t g_pend_mu = PTHREAD_MUTEX_INITIALIZER;
// NXN1X (lane B, NXN1X docs/lane_b/SHADER_EDEN_B.md): every shader a program sets is kept for Eden's recompiler under its host key (the sha256 key stays
// the host's handle for a shader; Eden hashes the code itself, as the Maxwell path does, and that hash keys vulkan.bin). The control section is copied here, while
// it is valid; the code area too when it is mapped (shader_set is called only then, hle_nvn.c keys_hash mapped = 1). DM_VK_EDEN=0: the shaders.dmsv bundle as before
int vk_eden_mode(void) { static int m = -1; if (m < 0) { const char *e = getenv("DM_VK_EDEN"); m = !(e && *e == '0'); } return m; }
static nvn_hmap g_eden = {.klen = 8}; static pthread_mutex_t g_eden_mu = PTHREAD_MUTEX_INITIALIZER;   // key → nvn_eden_shader*
struct nvn_eden_shader *vk_shader_eden(uint64_t key) {
    pthread_mutex_lock(&g_eden_mu); struct nvn_eden_shader *e = nvn_hm_get(&g_eden, &key); pthread_mutex_unlock(&g_eden_mu); return e;
}
static void eden_set(uint64_t key, const void *control, uint64_t data, uint32_t ncode) {
    pthread_mutex_lock(&g_eden_mu);
    if (!nvn_hm_get(&g_eden, &key)) {
        uint32_t nctl, n2; void nvn_ctl_lens(const void *control, uint32_t *nctl, uint32_t *ncode); nvn_ctl_lens(control, &nctl, &n2);
        struct nvn_eden_shader *e = nvn_eden_shader_new(control, nctl, (const void *)(uintptr_t)data, ncode, -1);
        if (e && nvn_eden_shader_stage(e) >= 0) nvn_hm_put(&g_eden, &key, e);
        else { if (e) nvn_eden_shader_free(e); static int n; if (n++ < 16) VKLOG("[RED] shader %016llx: no NVN stage found in its code area (not used)", (unsigned long long)key); }
    }
    pthread_mutex_unlock(&g_eden_mu);
}
static void shader_set(const uint8_t k[8], const void *control, uint64_t data, uint32_t ncode) {
    uint64_t key = 0; for (int i = 0; i < 8; i++) key = key << 8 | k[i];
    if (vk_eden_mode()) { if (key && control && data && ncode && ncode < (1u << 24)) eden_set(key, control, data, ncode); return; }
    if (!key || !control || !data || !ncode || ncode >= (1u << 24) || vk_shader_stage(key) >= 0) return;
    uint32_t nctl = 0x880, own; memcpy(&own, (const uint8_t *)control + 0x20, 4); if (own >= 0x760 && own < 0x880) nctl = own;   // as hle_nvn.c keys_hash
    uint8_t *c = malloc(nctl); if (!c) return; memcpy(c, control, nctl);
    pthread_mutex_lock(&g_pend_mu);
    pend_t *p = &g_pend[g_npend++ % NPEND]; free(p->ctl); *p = (pend_t){key, data, nctl, ncode, c};   // the oldest goes: a draw needs a program soon after it is set
    pthread_mutex_unlock(&g_pend_mu);
}
#ifndef NVN_XFB_LAYOUT_DEFINED   // s30b: also in host/dm/windows/gfx_vk_shader.c, whichever comes first
#define NVN_XFB_LAYOUT_DEFINED
// s30b: a vertex shader's transform feedback layout as its control section declares it (host gfx_vk_shader.c vk_xfb_note): per buffer the stride
// in bytes, the 4-byte words per vertex and Maxwell's stream-out attribute word at each position (0x20 + 4 n + component for generic output n)
typedef struct nvn_xfb_layout { uint32_t stride[4], count[4]; uint8_t words[4][128]; } nvn_xfb_layout;
#endif
// s30b: transform feedback layouts by shader key, read from the control sections nvnProgramSetShaders passes (valid only during that call: mtl_prog_prefetch
// below), where the NVN driver programs the capture from (tools/shader_vk.py xfb_layout): +0x6dc u16 stride of each buffer, +0x6e8 u8 number of outputs it
// captures, +0xdc + 0x80 b those outputs (0x20 + n = generic output n), +0x4dc + 0x80 b their component counts; the outputs follow each other in the buffer.
// words: Maxwell's stream-out attribute word at each 4-byte position (0x20 + 4 n + component; Xenoblade Chronicles 3's 5b44efea / cf93362a on the Maxwell
// path: 32..35, stride 16). Eden mode only (bundle shaders take theirs from their SPIR-V in vk_shader). A form not known is logged and captures nothing
static nvn_hmap g_xfb = {.klen = sizeof(uint64_t)}; static pthread_mutex_t g_xfb_mu = PTHREAD_MUTEX_INITIALIZER;
static int xfb_parse(const uint8_t *c, uint32_t nctl, nvn_xfb_layout *x) {   // 1 a layout, 0 none, -1 a form not known
    memset(x, 0, sizeof *x);
    if (!c || nctl < 0x6ec) return 0;
    uint16_t st[4]; memcpy(st, c + 0x6dc, sizeof st);
    if (!(st[0] | st[1] | st[2] | st[3])) return 0;
    for (int b = 0; b < 4; b++) {
        if (!st[b]) continue;
        int n = c[0x6e8 + b], nw = 0; const uint8_t *o = c + 0xdc + 0x80 * b, *k = c + 0x4dc + 0x80 * b;
        if (!n || n > 0x80) return -1;
        for (int i = 0; i < n; i++) {
            if (o[i] < 0x20 || o[i] >= 0x40 || k[i] < 1 || k[i] > 4) return -1;
            for (int j = 0; j < k[i]; j++) { if (nw >= 128) return -1; x->words[b][nw++] = (uint8_t)(0x20 + 4 * (o[i] - 0x20) + j); }
        }
        if (4 * nw != st[b]) return -1;
        x->stride[b] = st[b]; x->count[b] = (uint32_t)nw;
    }
    return 1;
}
static void vk_xfb_note(uint64_t key, const uint8_t *ctl, uint32_t nctl) {
    nvn_xfb_layout x; int r = xfb_parse(ctl, nctl, &x);
    if (!r) return;
    pthread_mutex_lock(&g_xfb_mu);
    if (!nvn_hm_get(&g_xfb, &key)) {
        nvn_xfb_layout *p = r > 0 ? malloc(sizeof *p) : NULL; if (p) *p = x;
        nvn_hm_put(&g_xfb, &key, p ? (void *)p : (void *)1);
        static int said; if (said++ < 24) {
            uint16_t st[4]; memcpy(st, ctl + 0x6dc, sizeof st);
            if (p) VKLOG("s30b: shader %016llx captures transform feedback: strides %u %u %u %u, words %u %u %u %u, buffer 0 from %u %u %u %u", (unsigned long long)key,
                         p->stride[0], p->stride[1], p->stride[2], p->stride[3], p->count[0], p->count[1], p->count[2], p->count[3],
                         p->words[0][0], p->words[0][1], p->words[0][2], p->words[0][3]);
            else VKLOG("[RED] s30b: shader %016llx: transform feedback in a form not known (strides %u %u %u %u, buffer 0: %u outputs, first %#x with %u components): nothing captured",
                       (unsigned long long)key, st[0], st[1], st[2], st[3], ctl[0x6e8], ctl[0xdc], ctl[0x4dc]);
        }
    }
    pthread_mutex_unlock(&g_xfb_mu);
}
static const nvn_xfb_layout *vk_xfb_get(uint64_t key) {
    pthread_mutex_lock(&g_xfb_mu); void *p = nvn_hm_get(&g_xfb, &key); pthread_mutex_unlock(&g_xfb_mu);
    return p && p != (void *)1 ? (const nvn_xfb_layout *)p : NULL;
}
static void vk_shader_xfb(vkshader_t *s, uint64_t key, int stage) {   // an Eden-mode shader object takes its capture layout (xfb_begin / xfb_end, eden_fill_state)
    const nvn_xfb_layout *x = vk_xfb_get(key);
    if (!x) return;
    if (stage != 0) { static int n; if (n++ < 4) VKLOG("[RED] s30b: shader %016llx (stage %d) declares transform feedback: only a vertex shader's capture is made here", (unsigned long long)key, stage); return; }
    s->xfb = 1; s->xfb_layout = x; for (int b = 0; b < 4; b++) s->xfb_stride[b] = x->stride[b];
}
// nvnProgramSetShaders calls this before it returns (hle_nvn.c), while the program's control sections are valid. Eden mode: each shader whose code area is
// mapped now is kept for Eden (eden_set). A code area not yet mapped (Mario) is not read here [구현 때 확인: its control section is gone by the first draw]
void mtl_prog_prefetch(const prog_t *p) {
    if (!vk_eden_mode() || !p) return;
    for (unsigned i = 0; i < p->nstage && i < 6; i++) {
        const NVNshaderData *sd = &p->sd[i]; uint8_t k[8]; uint32_t nctl, ncode; void *mb; uint64_t mo;
        if (!sd->control || !sd->data || !prog_key(p, i, k)) continue;
        nvn_ctl_lens(sd->control, &nctl, &ncode);
        { uint64_t xk = 0; for (int j = 0; j < 8; j++) xk = xk << 8 | k[j]; if (xk) vk_xfb_note(xk, (const uint8_t *)sd->control, nctl); }   // s30b: the control section is valid only now
        if (!ncode || ncode >= (1u << 24) || !mtl_addr(sd->data, ncode, &mb, &mo)) continue;
        uint64_t key = 0; for (int j = 0; j < 8; j++) key = key << 8 | k[j];
        if (key) eden_set(key, sd->control, sd->data, ncode);
    }
}
__attribute__((constructor)) static void shader_set_on(void) { extern void (*g_nvn_on_shader)(const uint8_t *, const void *, uint64_t, uint32_t); g_nvn_on_shader = shader_set; }
void vk_shader_missing(uint64_t key) {
    pend_t p = {0};
    pthread_mutex_lock(&g_pend_mu);
    for (unsigned i = 0; i < NPEND && i < g_npend; i++) if (g_pend[i].key == key && g_pend[i].ctl) { p = g_pend[i]; p.ctl = malloc(p.nctl); if (p.ctl) memcpy(p.ctl, g_pend[i].ctl, p.nctl); break; }
    pthread_mutex_unlock(&g_pend_mu);
    if (!p.ctl) { VKLOG("[RED] shader %016llx is not in the bundle (and was not set while the capture could see it)", (unsigned long long)key); return; }
    uint8_t md[CC_SHA256_DIGEST_LENGTH]; CC_SHA256_CTX hc; CC_SHA256_Init(&hc); CC_SHA256_Update(&hc, p.ctl, p.nctl); CC_SHA256_Update(&hc, (const void *)(uintptr_t)p.data, p.ncode); CC_SHA256_Final(md, &hc);
    uint64_t now = 0; for (int i = 0; i < 8; i++) now = now << 8 | md[i];
    if (now != key) { VKLOG("[RED] shader %016llx is not in the bundle; its code area changed since SetShaders, not saved", (unsigned long long)key); free(p.ctl); return; }
    const char *r = getenv("DM_ROOT"); char dir[1024], path[1100], tmp[1110]; snprintf(dir, sizeof dir, "%s/captured_shaders", r ? r : "."); mkdir(dir, 0755);
    snprintf(path, sizeof path, "%s/%016llx.bin", dir, (unsigned long long)key); snprintf(tmp, sizeof tmp, "%s.part", path);
    struct stat st; if (!stat(path, &st)) { VKLOG("[RED] shader %016llx is not in the bundle: captured in %s by an earlier run, the next port translates it", (unsigned long long)key, path); free(p.ctl); return; }
    FILE *f = fopen(tmp, "wb"); uint32_t hdr[3] = {0x43534d44, p.nctl, p.ncode};   // "DMSC"
    int ok = f && fwrite(hdr, 1, 12, f) == 12 && fwrite(p.ctl, 1, p.nctl, f) == p.nctl && fwrite((const void *)(uintptr_t)p.data, 1, p.ncode, f) == p.ncode;
    if (f && fclose(f)) ok = 0;
    if (!ok || renamex_np(tmp, path, 0)) {   // r53: replaces (the CRT rename does not)
        VKLOG("[RED] shader %016llx is not in the bundle, and %s could not be written", (unsigned long long)key, path); unlink(tmp); }
    else VKLOG("[RED] shader %016llx is not in the bundle: saved to %s, the next port translates it", (unsigned long long)key, path);
    free(p.ctl);
}
static VkShaderStageFlags stage_flag(int st) {
    switch (st) { case 0: return VK_SHADER_STAGE_VERTEX_BIT; case 1: return VK_SHADER_STAGE_FRAGMENT_BIT; case 2: return VK_SHADER_STAGE_GEOMETRY_BIT;
                  case 3: return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT; case 4: return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT; default: return VK_SHADER_STAGE_COMPUTE_BIT; }
}
static int set_of(int st) { return st == 5 ? 0 : st == 0 ? 0 : st == 1 ? 1 : st == 2 ? 2 : st; }
static VkDescriptorType desc_type(int d) {
    switch (d) { case RD_UBO: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; case RD_SSBO: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; case RD_SAMPLED: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                 case RD_STORAGE: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; case RD_TBUF: return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER; case RD_STORAGE_TBUF: return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
                 case RD_IMAGE: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; default: return VK_DESCRIPTOR_TYPE_SAMPLER; }
}
// Stage interfaces. Vulkan requires every input location of a stage to be declared as an output of the stage before it (VUID-RuntimeSpirv-OpEntryPoint-08743);
// the translator writes each stage apart, so a stage may read locations the one before never declares. Xenoblade Chronicles 3's passthrough geometry shaders
// (written out as ordinary ones, forwarding all 16 attributes) follow vertex shaders with two outputs, and Adreno refused those pipelines (VK_ERROR_UNKNOWN).
// The earlier stage then gets a variant that declares the missing vec4 outputs and leaves them unwritten: the values are undefined, as on the Switch.
// Except a passthrough geometry shader's outputs the vertex shader writes: the Switch passes those on (spv_gs_pass).
static int spv_is_vec4(const uint32_t *w, uint32_t fn, uint32_t t) {   // a vec4 of 32-bit floats, or an array of them (per-vertex inputs)
    for (int depth = 0; depth < 3; depth++) { uint32_t i = 5, found = 0;
        while (i < fn) { uint32_t n = w[i] >> 16; if (!n) return 0; if (w[i + 1] == t && ((w[i] & 0xffff) == 23 || (w[i] & 0xffff) == 28 || (w[i] & 0xffff) == 32)) { found = i; break; } i += n; }
        if (!found) return 0; uint32_t op = w[found] & 0xffff;
        if (op == 32) { t = w[found + 3]; continue; }   // OpTypePointer
        if (op == 28) { t = w[found + 2]; continue; }   // OpTypeArray
        if (w[found + 3] != 4) return 0; for (i = 5; i < fn;) { if ((w[i] & 0xffff) == 22 && w[i + 1] == w[found + 2]) return w[i + 2] == 32; i += w[i] >> 16; } return 0; }
    return 0;
}
// r46: the bindings (0-127) of variables whose image type has Depth 1: the translator declares a shadow sampler that way, and only those are read with Dref
static void spv_dref_bindings(const uint32_t *w, uint32_t nw, uint64_t dref[2]) {
    dref[0] = dref[1] = 0; if (nw < 5 || !w[3] || w[3] > (1u << 22)) return; uint32_t bound = w[3];
    uint32_t *link = calloc(bound, 4), *bind = calloc(bound, 4); uint8_t *dep = calloc(bound, 1);
    if (link && bind && dep) {
        for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) break; const uint32_t *a = &w[i];
            if (op == 25 && n >= 9 && a[1] < bound) dep[a[1]] = a[4] == 1 ? 2 : 1;   // OpTypeImage: Depth operand
            else if ((op == 27 || op == 28 || op == 29) && n >= 3 && a[1] < bound) link[a[1]] = a[2];   // OpTypeSampledImage, OpTypeArray, OpTypeRuntimeArray
            else if (op == 32 && n >= 4 && a[1] < bound) link[a[1]] = a[3];   // OpTypePointer
            else if (op == 71 && n >= 4 && a[2] == 33 && a[1] < bound) bind[a[1]] = a[3] + 1;   // OpDecorate Binding
            i += n; }
        for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) break;
            if (op == 59 && n >= 4 && w[i + 2] < bound && bind[w[i + 2]] && bind[w[i + 2]] <= 128) { uint32_t t = w[i + 1];   // OpVariable
                for (int k = 0; k < 8 && t && t < bound && !dep[t]; k++) t = link[t];
                if (t && t < bound && dep[t] == 2) { uint32_t b = bind[w[i + 2]] - 1; dref[b >> 6] |= 1ull << (b & 63); } }
            i += n; }
    }
    free(link); free(bind); free(dep);
}
static void spv_io_locs(const uint32_t *w, uint32_t nw, uint32_t *in, uint32_t *out) {
    *in = *out = 0; uint32_t fn = nw;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16; if (!n) return; if ((w[i] & 0xffff) == 54) { fn = i; break; } i += n; }
    for (uint32_t i = 5; i < fn;) { uint32_t n = w[i] >> 16;
        if ((w[i] & 0xffff) == 71 && n == 4 && w[i + 2] == 30 && w[i + 3] < 32) {   // OpDecorate Location
            uint32_t v = w[i + 1];
            for (uint32_t j = 5; j < fn;) { uint32_t m = w[j] >> 16; if ((w[j] & 0xffff) == 59 && w[j + 2] == v) {
                    if (spv_is_vec4(w, fn, w[j + 1])) { if (w[j + 3] == 1) *in |= 1u << w[i + 3]; else if (w[j + 3] == 3) *out |= 1u << w[i + 3]; }
                    else if (w[j + 3] == 3) *out |= 1u << w[i + 3];   // an output of another type still takes its location
                    break; }
                j += m; } }
        i += n; }
}
static uint32_t *spv_add_outputs(const uint32_t *w, uint32_t nw, uint32_t mask, uint32_t *onw) {
    uint32_t fn = nw, ep = 0, lastdec = 0, bound = w[3], f32 = 0, v4 = 0, pv4 = 0;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n) return NULL; if (op == 54) { fn = i; break; }
        if (op == 15 && !ep) ep = i; if (op == 71 || op == 72) lastdec = i + n;
        if (op == 22 && w[i + 2] == 32) f32 = w[i + 1];
        i += n; }
    if (!ep || !lastdec || !f32) return NULL;
    for (uint32_t i = 5; i < fn;) { if ((w[i] & 0xffff) == 23 && w[i + 2] == f32 && w[i + 3] == 4) v4 = w[i + 1]; i += w[i] >> 16; }
    if (v4) for (uint32_t i = 5; i < fn;) { if ((w[i] & 0xffff) == 32 && w[i + 2] == 3 && w[i + 3] == v4) pv4 = w[i + 1]; i += w[i] >> 16; }
    int k = __builtin_popcount(mask); uint32_t ids[32], nid = 0;
    uint32_t *o = malloc(((size_t)nw + 20 * k + 40) * 4), no = 5; memcpy(o, w, 20);
    for (int l = 0; l < 32; l++) if (mask >> l & 1) ids[nid++] = bound++;
    uint32_t tv4 = v4 ? v4 : bound++, tpv4 = pv4 ? pv4 : bound++;
    for (uint32_t i = 5; i < fn;) { uint32_t n = w[i] >> 16;
        if (i == ep) { o[no++] = (n + nid) << 16 | 15; memcpy(&o[no], &w[i + 1], (n - 1) * 4); no += n - 1; memcpy(&o[no], ids, nid * 4); no += nid; }   // the entry point lists them
        else { memcpy(&o[no], &w[i], n * 4); no += n; }
        i += n;
        if (i == lastdec) { uint32_t q = 0; for (int l = 0; l < 32; l++) if (mask >> l & 1) { o[no++] = 4u << 16 | 71; o[no++] = ids[q++]; o[no++] = 30; o[no++] = (uint32_t)l; } }   // Location
    }
    if (!v4) { o[no++] = 4u << 16 | 23; o[no++] = tv4; o[no++] = f32; o[no++] = 4; }
    if (!pv4) { o[no++] = 4u << 16 | 32; o[no++] = tpv4; o[no++] = 3; o[no++] = tv4; }
    for (uint32_t q = 0; q < nid; q++) { o[no++] = 4u << 16 | 59; o[no++] = tpv4; o[no++] = ids[q]; o[no++] = 3; }   // OpVariable Output
    // and each starts as (0, 0, 0, 1) at the entry point's first block: never written, Vulkan leaves it undefined, and Adreno handed the fragment shader garbage (Donkey Kong
    // Bananza: c8637178 reads location 24, which a812966e does not write: black ceilings with white halos, bright flickering points). DM_VK_XO_INIT=0 (at start): not stored
    static int xinit = -1; if (xinit < 0) { const char *e = getenv("DM_VK_XO_INIT"); xinit = !e || atoi(e); }
    uint32_t c0 = 0, c1 = 0, cv = 0, efn = w[ep + 2];
    if (xinit) { c0 = bound++; c1 = bound++; cv = bound++;
        o[no++] = 4u << 16 | 43; o[no++] = f32; o[no++] = c0; o[no++] = 0; o[no++] = 4u << 16 | 43; o[no++] = f32; o[no++] = c1; o[no++] = 0x3f800000u;
        o[no++] = 7u << 16 | 44; o[no++] = tv4; o[no++] = cv; o[no++] = c0; o[no++] = c0; o[no++] = c0; o[no++] = c1; }   // OpConstant, OpConstantComposite
    for (uint32_t i = fn, st = 0; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { memcpy(&o[no], &w[i], (nw - i) * 4); no += nw - i; break; }
        if (xinit && st == 2 && op != 59 && op != 8 && op != 317) { for (uint32_t q = 0; q < nid; q++) { o[no++] = 3u << 16 | 62; o[no++] = ids[q]; o[no++] = cv; } st = 3; }   // OpStore before the first non-variable
        memcpy(&o[no], &w[i], n * 4); no += n;
        if (st == 0 && op == 54 && w[i + 2] == efn) st = 1; else if (st == 1 && op == 248) st = 2;   // the entry function, then its first OpLabel
        i += n; }
    o[3] = bound; *onw = no; return o;
}
// A passthrough geometry shader (SPH word 0 bit 24) cannot write the vertices it passes on: the hardware hands the next stage each vertex's outputs from the stage
// before. The translator's emulation (Ryujinx EmitterContext, PassthroughAttributes) copies only the attributes it is told the next stage reads, and the module is
// translated without its fragment shader, so it copies none, and an output spv_add_outputs declares for the fragment shader is never written. Donkey Kong Bananza
// draws its terrain through one (5d537f7630ec3671, location 19 read by the pixel shader's discard): black, flickering triangles. Here each location in cp, which the
// vertex shader writes, gets an input array at that location, and vertex j's value is stored to the output just before the j-th OpEmitVertex of each primitive,
// the order the emulation emits them in. Only for point, line or triangle input whose every primitive has that many vertices (the emulation's shape); NULL otherwise
static uint32_t *spv_gs_pass(const uint32_t *w, uint32_t nw, uint32_t cp, uint32_t xr, uint32_t *onw) {   // xr: the locations it reads them at (0: the same; spv_vs_dup); ~0: writes zeros, ~1: writes (1e6, 1e6, 1e6, 1), reads nothing (DM_VK_GS_PASS=3 / 4, diagnostics)
    uint32_t fn = nw, ep = 0, lastdec = 0, bound = w[3], f32 = 0, u32 = 0, v4 = 0, pv4 = 0, nv = 0, out[32] = {0}, emits = 0, k = 0, groups = 0;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) return NULL; if (op == 54) { fn = i; break; }
        if (op == 15 && !ep) ep = i; if (op == 71 || op == 72) lastdec = i + n;
        if (op == 16 && n >= 3) { uint32_t m = w[i + 2]; if (m == 19) nv = 1; else if (m == 20) nv = 2; else if (m == 22) nv = 3; else if (m == 21 || m == 23) return NULL; }   // InputPoints / InputLines / Triangles; adjacency
        if (op == 22 && w[i + 2] == 32) f32 = w[i + 1];
        if (op == 21 && w[i + 2] == 32 && w[i + 3] == 0) u32 = w[i + 1];
        i += n; }
    if (!ep || !lastdec || !f32 || !u32 || !nv) return NULL;
    for (uint32_t i = 5; i < fn;) { if ((w[i] & 0xffff) == 23 && w[i + 2] == f32 && w[i + 3] == 4) v4 = w[i + 1]; i += w[i] >> 16; }
    if (!v4) return NULL;
    for (uint32_t i = 5; i < fn;) { if ((w[i] & 0xffff) == 32 && w[i + 2] == 1 && w[i + 3] == v4) pv4 = w[i + 1]; i += w[i] >> 16; }   // OpTypePointer Input vec4
    for (uint32_t i = 5; i < fn;) { uint32_t n = w[i] >> 16;   // the output variable at each location (OpDecorate Location on an OpVariable Output of a vec4)
        if ((w[i] & 0xffff) == 71 && n == 4 && w[i + 2] == 30 && w[i + 3] < 32 && (cp >> w[i + 3] & 1))
            for (uint32_t j = 5; j < fn;) { uint32_t m = w[j] >> 16; if ((w[j] & 0xffff) == 59 && w[j + 2] == w[i + 1]) { if (w[j + 3] == 3 && spv_is_vec4(w, fn, w[j + 1])) out[w[i + 3]] = w[i + 1]; break; } j += m; }
        i += n; }
    for (int l = 0; l < 32; l++) if ((cp >> l & 1) && !out[l]) return NULL;
    for (uint32_t i = fn; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) return NULL;
        if (op == 220 || op == 221) return NULL;   // OpEmitStreamVertex / OpEndStreamPrimitive
        if (op == 218) { if (++k > nv) return NULL; emits++; }
        if (op == 219) { if (k != nv) return NULL; k = 0; groups++; }
        i += n; }
    if (k || !groups) return NULL;
    int konst = xr == ~0u ? 1 : xr == ~1u ? 2 : 0; if (konst) xr = 0;
    if (xr && __builtin_popcount(xr) != __builtin_popcount(cp)) return NULL;
    int ncp = __builtin_popcount(cp), nin = konst ? 0 : ncp; uint32_t in[32] = {0}, cj[3], cn = bound++, arr = bound++, parr = bound++, tpv4 = pv4 ? pv4 : bound++, cz = bound++, cf = bound++, c1 = bound++;
    for (uint32_t j = 0; j < nv; j++) cj[j] = bound++;
    if (!konst) for (int l = 0; l < 32; l++) if (cp >> l & 1) in[l] = bound++;
    uint32_t *o = malloc(((size_t)nw + 80 + (size_t)ncp * (12 + 12 * emits)) * 4), no = 5; memcpy(o, w, 20);
    for (uint32_t i = 5; i < fn;) { uint32_t n = w[i] >> 16;
        if (i == ep) { o[no++] = (n + nin) << 16 | 15; memcpy(&o[no], &w[i + 1], (n - 1) * 4); no += n - 1; for (int l = 0; l < 32; l++) if (in[l]) o[no++] = in[l]; }   // the entry point lists them
        else { memcpy(&o[no], &w[i], n * 4); no += n; }
        i += n;
        if (i == lastdec) { uint32_t rl[32]; int q = 0; for (int l = 0; l < 32; l++) if (xr >> l & 1) rl[q++] = (uint32_t)l;
            q = 0; for (int l = 0; l < 32; l++) if (in[l]) { o[no++] = 4u << 16 | 71; o[no++] = in[l]; o[no++] = 30; o[no++] = xr ? rl[q++] : (uint32_t)l; } }   // Location
    }
    o[no++] = 4u << 16 | 43; o[no++] = u32; o[no++] = cn; o[no++] = nv;   // OpConstant uint nv
    for (uint32_t j = 0; j < nv; j++) { o[no++] = 4u << 16 | 43; o[no++] = u32; o[no++] = cj[j]; o[no++] = j; }
    o[no++] = 4u << 16 | 28; o[no++] = arr; o[no++] = v4; o[no++] = cn;      // OpTypeArray vec4[nv]
    o[no++] = 4u << 16 | 32; o[no++] = parr; o[no++] = 1; o[no++] = arr;     // OpTypePointer Input
    if (!pv4) { o[no++] = 4u << 16 | 32; o[no++] = tpv4; o[no++] = 1; o[no++] = v4; }
    for (int l = 0; l < 32; l++) if (in[l]) { o[no++] = 4u << 16 | 59; o[no++] = parr; o[no++] = in[l]; o[no++] = 1; }   // OpVariable Input
    if (konst == 2) { o[no++] = 4u << 16 | 43; o[no++] = f32; o[no++] = cf; o[no++] = 0x49742400u;   // float 1e6
                      o[no++] = 4u << 16 | 43; o[no++] = f32; o[no++] = c1; o[no++] = 0x3f800000u;   // float 1
                      o[no++] = 7u << 16 | 44; o[no++] = v4; o[no++] = cz; o[no++] = cf; o[no++] = cf; o[no++] = cf; o[no++] = c1; }   // OpConstantComposite
    else { o[no++] = 3u << 16 | 46; o[no++] = v4; o[no++] = cz; }   // OpConstantNull vec4
    k = 0;
    for (uint32_t i = fn; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 218) { for (int l = 0; l < 32; l++) if (in[l]) { uint32_t a = bound++, v = bound++;
                o[no++] = 5u << 16 | 65; o[no++] = tpv4; o[no++] = a; o[no++] = in[l]; o[no++] = cj[k];   // OpAccessChain in[l][k]
                o[no++] = 4u << 16 | 61; o[no++] = v4; o[no++] = v; o[no++] = a;                         // OpLoad
                o[no++] = 3u << 16 | 62; o[no++] = out[l]; o[no++] = v; }                                // OpStore out[l]
            if (konst) for (int l = 0; l < 32; l++) if (cp >> l & 1) { o[no++] = 3u << 16 | 62; o[no++] = out[l]; o[no++] = cz; }
            k++; }
        if (op == 219) k = 0;
        memcpy(&o[no], &w[i], n * 4); no += n; i += n; }
    o[3] = bound; *onw = no; return o;
}
// DM_VK_GS_PASS=2 (gfx_vk_exec.c gs_pass_mode): the vertex shader of a passthrough geometry pipeline also writes each output in xp to a free location below 16 (the bits of xr, in
// the same order), so the geometry shader reads it there and not above Turnip's 16 geometry input locations (maxGeometryInputComponents 64). The values are copied before
// each OpReturn of the entry point (the translator writes outputs component by component, also from other functions). NULL when the module is not as expected
static uint32_t *spv_vs_dup(const uint32_t *w, uint32_t nw, uint32_t xp, uint32_t xr, uint32_t *onw) {
    uint32_t fn = nw, ep = 0, lastdec = 0, bound = w[3], efn = 0, src[32] = {0}, ptr[32] = {0}, pt[32] = {0}, dst[32] = {0}, rets = 0;
    if (__builtin_popcount(xp) != __builtin_popcount(xr)) return NULL;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) return NULL; if (op == 54) { fn = i; break; }
        if (op == 15 && !ep) { ep = i; efn = w[i + 2]; } if (op == 71 || op == 72) lastdec = i + n;
        i += n; }
    if (!ep || !lastdec) return NULL;
    for (uint32_t i = 5; i < fn;) { uint32_t n = w[i] >> 16;   // the output variable at each location of xp, its pointer type and the vec4 type it points to
        if ((w[i] & 0xffff) == 71 && n == 4 && w[i + 2] == 30 && w[i + 3] < 32 && (xp >> w[i + 3] & 1))
            for (uint32_t j = 5; j < fn;) { uint32_t m = w[j] >> 16; if ((w[j] & 0xffff) == 59 && w[j + 2] == w[i + 1]) {
                    if (w[j + 3] == 3 && spv_is_vec4(w, fn, w[j + 1])) { src[w[i + 3]] = w[i + 1]; ptr[w[i + 3]] = w[j + 1]; } break; } j += m; }
        i += n; }
    for (int l = 0; l < 32; l++) if (xp >> l & 1) {
        if (!src[l]) return NULL;
        for (uint32_t i = 5; i < fn;) { if ((w[i] & 0xffff) == 32 && w[i + 1] == ptr[l]) { pt[l] = w[i + 3]; break; } i += w[i] >> 16; }
        if (!pt[l]) return NULL; }
    for (uint32_t i = fn, in_ep = 0; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) return NULL;
        if (op == 54) in_ep = w[i + 2] == efn; else if (op == 56) in_ep = 0; else if (in_ep && (op == 253 || op == 254)) rets++;   // OpReturn / OpReturnValue
        i += n; }
    if (!rets) return NULL;
    int ncp = __builtin_popcount(xp); uint32_t rl[32]; { int q = 0; for (int l = 0; l < 32; l++) if (xr >> l & 1) rl[q++] = (uint32_t)l; }
    for (int l = 0; l < 32; l++) if (xp >> l & 1) dst[l] = bound++;
    uint32_t *o = malloc(((size_t)nw + 16 + (size_t)ncp * (8 + 7 * rets)) * 4), no = 5; memcpy(o, w, 20);
    for (uint32_t i = 5; i < fn;) { uint32_t n = w[i] >> 16;
        if (i == ep) { o[no++] = (n + ncp) << 16 | 15; memcpy(&o[no], &w[i + 1], (n - 1) * 4); no += n - 1; for (int l = 0; l < 32; l++) if (dst[l]) o[no++] = dst[l]; }
        else { memcpy(&o[no], &w[i], n * 4); no += n; }
        i += n;
        if (i == lastdec) { int q = 0; for (int l = 0; l < 32; l++) if (dst[l]) { o[no++] = 4u << 16 | 71; o[no++] = dst[l]; o[no++] = 30; o[no++] = rl[q++]; } }   // Location
    }
    for (int l = 0; l < 32; l++) if (dst[l]) { o[no++] = 4u << 16 | 59; o[no++] = ptr[l]; o[no++] = dst[l]; o[no++] = 3; }   // OpVariable Output
    for (uint32_t i = fn, in_ep = 0; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 54) in_ep = w[i + 2] == efn; else if (op == 56) in_ep = 0;
        else if (in_ep && (op == 253 || op == 254)) for (int l = 0; l < 32; l++) if (dst[l]) { uint32_t v = bound++;
                o[no++] = 4u << 16 | 61; o[no++] = pt[l]; o[no++] = v; o[no++] = src[l];   // OpLoad the output
                o[no++] = 3u << 16 | 62; o[no++] = dst[l]; o[no++] = v; }                  // OpStore its copy
        memcpy(&o[no], &w[i], n * 4); no += n; i += n; }
    o[3] = bound; *onw = no; return o;
}
static nvn_hmap g_shaders = {.klen = 16};   // {key, stage} → vkshader_t*
static pthread_mutex_t g_sh_mu = PTHREAD_MUTEX_INITIALIZER;
// Shared layouts keep pipeline layouts and descriptor bindings reusable across shaders; g_sh_mu protects this cache too.
static VkDescriptorSetLayout shader_layout(const VkDescriptorSetLayoutBinding *b, uint32_t n) {
    struct { uint32_t n, b[96][4]; } k = {.n = n};
    static nvn_hmap cache = {.klen = sizeof k};
    for (uint32_t i = 0; i < n; i++) {
        uint32_t j = i;
        while (j && k.b[j - 1][0] > b[i].binding) { memcpy(k.b[j], k.b[j - 1], sizeof k.b[j]); j--; }
        k.b[j][0] = b[i].binding; k.b[j][1] = b[i].descriptorType; k.b[j][2] = b[i].descriptorCount; k.b[j][3] = b[i].stageFlags;
    }
    VkDescriptorSetLayout layout = nvn_hm_get(&cache, &k);
    if (!layout) {
        VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, n, b};
        VKCHK(vkCreateDescriptorSetLayout(V.dev, &li, NULL, &layout));
        void vk_dsl_register(VkDescriptorSetLayout l, const VkDescriptorSetLayoutBinding *b, uint32_t n); vk_dsl_register(layout, b, n);
        nvn_hm_put(&cache, &k, layout);
    }
    return layout;
}
// r85: the game's TXD (texture with given gradients) on a 3D texture is emulated by its compiler with SHFL from lane 3 of each group of four plus FSWZADD, in
// every stage. Our translator keeps that as a helper function with OpGroupNonUniformShuffle, called with the SHFL c operand 7171 (clamp 3, segment mask 0x1c:
// the four lanes of a quad). Outside the fragment shader there are no quads, and lanes past the end of a partly filled subgroup are inactive, so a shuffle
// from them is undefined: Bayonetta 3's clouds of erasure (TES 152e86c9 / de621a43) took a garbage 3D texel and moved those vertices away (thin dark "wires"
// through the fog; A/B with the shuffle replaced 11-08: gone). Eden (NXN1X) turns the pattern into textureGrad. In these stages our translator already
// samples at LOD 0, so each lane's own coordinate gives the same result: in a vertex, tessellation or geometry shader, the shuffle in a helper function whose
// every call passes the constant 7171 as c becomes its own value (OpCopyObject). Other shuffles stay: Donkey Kong Bananza's terrain geometry shader 5d537f76
// hands one lane's atomic result to the others with c = 31, which is a real cross-lane read. DM_VK_VTX_SHFL=0: off, =2: every shuffle in these stages (A/B)
static uint32_t *spv_vtx_shfl(uint32_t *w, uint32_t *pnw, uint64_t key, int all) {
    uint32_t nw = *pnw, bound = w[3], model = ~0u, nsh = 0, cur = 0;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) return w;
        if (op == 15 && n >= 3 && model == ~0u) model = w[i + 1];   // OpEntryPoint
        if (op == 345 && n == 6) nsh++;   // OpGroupNonUniformShuffle
        i += n; }
    if (!nsh || model > 3 || bound < 2) return w;   // Vertex 0, TessellationControl 1, TessellationEvaluation 2, Geometry 3
    uint32_t *cval = calloc(bound, 4), *last = calloc(bound, 4); uint8_t *isc = calloc(bound, 1), *has = calloc(bound, 1), *bad = calloc(bound, 1), *called = calloc(bound, 1);
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 43 && n == 4 && w[i + 2] < bound) { isc[w[i + 2]] = 1; cval[w[i + 2]] = w[i + 3]; }   // OpConstant (32-bit)
        if (op == 54 && n >= 5) cur = w[i + 2]; else if (op == 56) cur = 0;   // OpFunction / OpFunctionEnd
        if (op == 345 && n == 6 && cur < bound) has[cur] = 1;
        if (op == 62 && n >= 3 && w[i + 1] < bound) last[w[i + 1]] = w[i + 2];   // OpStore ptr value: the value a by-pointer argument holds at the call
        if (op == 57 && n >= 4 && w[i + 3] < bound) { uint32_t f = w[i + 3]; called[f] = 1;   // OpFunctionCall type result function args
            uint32_t a = n >= 7 ? w[i + 6] : 0, v = a < bound ? last[a] : 0; if (!a || a >= bound || !((v < bound && isc[v] && cval[v] == 7171) || (isc[a] && cval[a] == 7171))) bad[f] = 1; }
        i += n; }
    uint32_t *o = malloc((size_t)nw * 4), no = 0, nrep = 0; cur = 0;
    for (uint32_t i = 0; i < 5; i++) o[no++] = w[i];
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 54 && n >= 5) cur = w[i + 2]; else if (op == 56) cur = 0;
        if (op == 345 && n == 6 && cur && cur < bound && (all || (has[cur] && called[cur] && !bad[cur]))) { o[no++] = 4u << 16 | 83; o[no++] = w[i + 1]; o[no++] = w[i + 2]; o[no++] = w[i + 4]; nrep++; }   // OpCopyObject type result value
        else for (uint32_t k = 0; k < n; k++) o[no++] = w[i + k];
        i += n; }
    free(cval); free(last); free(isc); free(has); free(bad); free(called);
    static int nlog; if (nlog++ < 16) VKLOG("r85: shader %016llx (model %u): %u of %u shuffles -> own value (%s)", (unsigned long long)key, model, nrep, nsh, all ? "all, DM_VK_VTX_SHFL=2" : "TXD pattern, c = 7171");
    if (!nrep) { free(o); return w; }
    free(w); *pnw = no; return o;
}
// r84: the Maxwell FADD / FMUL / FFMA results are exact, and the game's shaders rely on it: Bayonetta 3's clouds of erasure (TES 152e86c9 / de621a43) put the
// shared edge of two patches at the same point only if no add is fused into a multiply, else the edges part (thin dark "wires" through the fog). Our translator
// gives SPIR-V without NoContraction; Eden (NXN1X) marks every FADD / FMUL / FFMA no_contraction (emit_spirv_floating_point.cpp), and its fog is whole on the
// same Turnip. Every OpFAdd / OpFSub / OpFMul / GLSL.std.450 Fma result gets NoContraction. Off by default since r85 (no effect on the wires; the cause was the TES shuffle): DM_VK_NOCONTRACT=1: on, =2: tessellation stages only
static uint32_t *spv_nocontract(uint32_t *w, uint32_t *pnw, uint64_t key) {
    uint32_t nw = *pnw, bound = w[3], glsl = 0, ins = 0, n42 = 0;
    if (bound < 2) return w;
    uint8_t *mark = calloc(bound, 1);
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { free(mark); return w; }
        if (op == 11 && n >= 3 && !memcmp(&w[i + 2], "GLSL.std.450", 12)) glsl = w[i + 1];   // OpExtInstImport
        int pre = op == 17 || op == 10 || op == 11 || op == 14 || op == 15 || op == 16 || op == 3 || op == 4 || op == 5 || op == 6 || op == 7 || op == 8 || op == 330 ||
                  (op >= 71 && op <= 75) || op == 332 || op == 5632 || op == 5633;   // preamble, debug and annotations
        if (!ins && !pre) ins = i;   // the first type / constant / variable: decorations go before it
        if (op == 71 && n >= 3 && w[i + 2] == 42 && w[i + 1] < bound) mark[w[i + 1]] = 2;   // NoContraction already
        if ((op == 129 || op == 131 || op == 133) && n >= 5 && w[i + 2] < bound && !mark[w[i + 2]]) { mark[w[i + 2]] = 1; n42++; }
        if (op == 12 && n >= 5 && glsl && w[i + 3] == glsl && w[i + 4] == 50 && w[i + 2] < bound && !mark[w[i + 2]]) { mark[w[i + 2]] = 1; n42++; }   // Fma
        i += n; }
    if (!n42 || !ins) { free(mark); return w; }
    uint32_t *o = malloc(((size_t)nw + 3 * (size_t)n42) * 4), no = ins; memcpy(o, w, (size_t)ins * 4);
    for (uint32_t id = 1; id < bound; id++) if (mark[id] == 1) { o[no++] = 3u << 16 | 71; o[no++] = id; o[no++] = 42; }   // OpDecorate id NoContraction
    memcpy(&o[no], &w[ins], (size_t)(nw - ins) * 4); no += nw - ins;
    free(mark);
    static int nlog; if (nlog++ < 4) VKLOG("r84: shader %016llx: NoContraction on %u results", (unsigned long long)key, n42);
    free(w); *pnw = no; return o;
}
// r83 (A/B for the cause): a fragment shader that shuffles across lanes and discards. OpKill ends the invocation, so a lane killed before the shuffles
// (Bayonetta 3's cloud shaders c0ad5d4f / 8a28afe0: scene-depth test, then quad-neighbour noise derivatives by SHFL) leaves its neighbours reading an
// inactive lane: holes in the clouds of erasure (the "wires" in the fog). Each OpKill that is the whole body of an if becomes OpDemoteToHelperInvocation
// and a branch to that if's merge block, so the lane runs on without writing. A shader with any other OpKill is left as it is. Off by default since r85 (no effect): DM_VK_DEMOTE=1: on
static uint32_t *spv_demote(uint32_t *w, uint32_t *pnw, uint64_t key) {
    uint32_t nw = *pnw, bound = w[3], nsh = 0, nkill = 0, lastcap = 0, pend = 0, cur = 0, ok = 1;
    if (w[1] < 0x10300 || bound < 2) return w;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) return w;
        if (op == 17) { if (n == 2 && w[i + 1] == 5379) return w; lastcap = i + n; }   // OpCapability (DemoteToHelperInvocation: declared already)
        if (op == 15 && (n < 3 || w[i + 1] != 4)) return w;   // fragment only
        if (op >= 345 && op <= 348) nsh++;
        if (op == 252) nkill++;
        i += n; }
    if (!nsh || !nkill || !lastcap) return w;
    uint32_t *merge = calloc(bound, 4);   // a block's label -> the merge block of the selection that branches to it
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 247 && n >= 3) pend = w[i + 1];   // OpSelectionMerge
        else if (op == 250 && n >= 4) { if (pend) { for (int k = 2; k <= 3; k++) if (w[i + k] < bound && w[i + k] != pend) merge[w[i + k]] = pend; } pend = 0; }   // OpBranchConditional
        else if (op == 248 || op == 249 || op == 251) pend = 0;
        i += n; }
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 248) cur = w[i + 1]; else if (op == 252 && !(cur < bound && merge[cur])) ok = 0;
        i += n; }
    if (!ok) { free(merge); static int n0; if (n0++ < 8) VKLOG("r83: shader %016llx: an OpKill outside an if, left as it is", (unsigned long long)key); return w; }
    int ext = w[1] < 0x10600;
    uint32_t *o = malloc(((size_t)nw + 12 + 2 * (size_t)nkill) * 4), no = lastcap; memcpy(o, w, (size_t)lastcap * 4);
    o[no++] = 2u << 16 | 17; o[no++] = 5379;   // OpCapability DemoteToHelperInvocation
    if (ext) { o[no++] = 10u << 16 | 10; memset(&o[no], 0, 36); memcpy(&o[no], "SPV_EXT_demote_to_helper_invocation", 35); no += 9; }   // OpExtension
    for (uint32_t i = lastcap; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 248) cur = w[i + 1];
        if (op == 252) { o[no++] = 1u << 16 | 5380; o[no++] = 2u << 16 | 249; o[no++] = merge[cur]; i += n; continue; }   // OpDemoteToHelperInvocation, OpBranch merge
        memcpy(&o[no], &w[i], (size_t)n * 4); no += n; i += n; }
    free(merge);
    static int nlog; if (nlog++ < 8) VKLOG("r83: shader %016llx: %u OpKill -> demote (%u shuffles)", (unsigned long long)key, nkill, nsh);
    free(w); *pnw = no; return o;
}
// r59: Bayonetta 3's fragment shaders read their quad neighbours with SHFL (subgroup shuffles here). Vulkan launches helper invocations for derivatives
// and quad operations only, and Turnip follows it (nir_gather_info: a subgroup operation asks for whole quads only under RequireFullQuadsKHR), so at a
// triangle's edge the uncovered pixels of a quad are not there and a shuffle reads garbage: purple dots along the edges. The module declares
// RequireFullQuadsKHR (VK_KHR_shader_quad_control) when it shuffles. Witch-Time reads the quad through fine derivatives instead (mx_translate.c mx_quad3)
static uint32_t *spv_full_quads(uint32_t *w, uint32_t *pnw, uint64_t key) {
    uint32_t nw = *pnw, nep = 0, epfn = 0, lastcap = 0, lastep = 0, nsh = 0;
    if (w[1] < 0x10300) return w;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) return w;
        if (op == 17) { if (n == 2 && w[i + 1] == 5087) return w; lastcap = i + n; }   // OpCapability (QuadControlKHR: declared already)
        if (op == 15) { if (n < 3 || w[i + 1] != 4) return w; epfn = w[i + 2]; lastep = i + n; nep++; }   // OpEntryPoint Fragment
        if (op >= 345 && op <= 348) nsh++;   // OpGroupNonUniformShuffle / Xor / Up / Down
        i += n; }
    if (!nsh || nep != 1 || !lastcap || lastep <= lastcap) return w;
    uint32_t *o = malloc(((size_t)nw + 12) * 4), no = lastcap; memcpy(o, w, (size_t)lastcap * 4);
    o[no++] = 2u << 16 | 17; o[no++] = 5087;   // OpCapability QuadControlKHR
    o[no++] = 7u << 16 | 10; memcpy(&o[no], "SPV_KHR_quad_control\0\0\0", 24); no += 6;   // OpExtension
    memcpy(&o[no], &w[lastcap], (size_t)(lastep - lastcap) * 4); no += lastep - lastcap;
    o[no++] = 3u << 16 | 16; o[no++] = epfn; o[no++] = 5089;   // OpExecutionMode RequireFullQuadsKHR
    memcpy(&o[no], &w[lastep], (size_t)(nw - lastep) * 4); no += nw - lastep;
    static int nlog; if (nlog++ < 8) VKLOG("full quads: shader %016llx shuffles %u times", (unsigned long long)key, nsh);
    free(w); *pnw = no; return o;
}
static inline uint64_t r80_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static void r80_sh(uint64_t key, int stage, uint64_t t0, uint64_t t1, int loaded) {   // r80 (diagnostic): a shader lookup that took 100 ms or more (the lock, or the load from the bundle)
    uint64_t t2 = r80_ns(); if (t2 - t0 < 100000000ull) return; static int n; if (n++ >= 64) return;
    VKLOG("r80: shader %016llx (stage %d) took %.0f ms: %.0f ms waiting for the shader lock%s", (unsigned long long)key, stage, (t2 - t0) / 1e6, (t1 - t0) / 1e6, loaded ? ", the rest loading it from the bundle" : "");
}
vkshader_t *vk_shader(uint64_t key, int stage) {
    struct { uint64_t key; uint64_t stage; } hk = {key, (uint64_t)stage};
    if (vk_eden_mode()) {   // NXN1X (lane B): the module, layout and bindings come with the Eden pipeline (nvn_eden_graphics_pipeline); this keeps the shader's identity
        pthread_mutex_lock(&g_sh_mu);
        vkshader_t *s = nvn_hm_get(&g_shaders, &hk);
        if (!s) { struct nvn_eden_shader *e = vk_shader_eden(key);
            if (e && nvn_eden_shader_stage(e) == stage) { s = calloc(1, sizeof *s); s->key = key; s->stage = stage; s->set = set_of(stage); s->eden = e; s->setmemo = 2;
                s->c1 = nvn_eden_shader_c1(e, &s->c1_len); vk_shader_xfb(s, key, stage); }   // vk_shader_c1 makes its uniform buffer as before
            nvn_hm_put(&g_shaders, &hk, s ? (void *)s : (void *)1);
            if (!s) { static int n; if (n++ < 16) VKLOG("[RED] shader %016llx (stage %d): no code from SetShaders for Eden", (unsigned long long)key, stage); } }
        pthread_mutex_unlock(&g_sh_mu);
        return s == (void *)1 ? NULL : s;
    }
    bundle_auto();
    uint64_t t80 = r80_ns();   // r80
    pthread_mutex_lock(&g_sh_mu);
    uint64_t t81 = r80_ns();
    vkshader_t *s = nvn_hm_get(&g_shaders, &hk);
    if (s) { pthread_mutex_unlock(&g_sh_mu); r80_sh(key, stage, t80, t81, 0); return s == (void *)1 ? NULL : s; }
    const bent *e = g_bundle ? find(key) : NULL;
    if (!e) { nvn_hm_put(&g_shaders, &hk, (void *)1); pthread_mutex_unlock(&g_sh_mu);
              static int n; if (n++ < 16) VKLOG("[RED] shader %016llx (stage %d) is not in the bundle", (unsigned long long)key, stage); return NULL; }
    s = calloc(1, sizeof *s); s->key = key; s->stage = stage; s->set = set_of(stage);
    // reflection
    const uint8_t *r = g_bundle + bent_refl(e); uint32_t hdr[5]; memcpy(hdr, r, 20);
    s->nbind = (int)hdr[0]; s->c1_len = hdr[1]; memcpy(s->local, hdr + 2, 12);
    s->b = (const dmsv_bind *)(r + 20); s->c1 = r + 20 + 20 * s->nbind;
    // SPIR-V with the stage's set
    uint32_t nw = e->spv_len / 4; uint32_t *w = malloc(e->spv_len); memcpy(w, g_bundle + bent_spv(e), e->spv_len);
    int f64_used = 0; uint32_t xb = 4;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n) break; if (op == 71 && n >= 4 && w[i + 2] == 34) w[i + 3] = (uint32_t)s->set;   // OpDecorate DescriptorSet
        if (op == 22 && n == 3 && w[i + 2] == 64) f64_used = 1;   // OpTypeFloat 64
        if (op == 16 && n >= 3 && w[i + 2] == 11) s->xfb = 1;     // OpExecutionMode Xfb: a vertex shader that captures transform feedback (shader_vk.py xfb_layout)
        if (op == 16 && n >= 3 && (w[i + 2] == 12 || w[i + 2] == 14 || w[i + 2] == 15)) s->depth_repl = 1;   // OpExecutionMode DepthReplacing / DepthGreater / DepthLess
        if (((op == 71 && n >= 4 && w[i + 2] == 11 && w[i + 3] == 9) || (op == 72 && n >= 5 && w[i + 3] == 11 && w[i + 4] == 9)) && (s->stage == 0 || s->stage == 2 || s->stage == 4)) s->layer_out = 1;   // r76: BuiltIn Layer
        // each buffer's stride: the translator decorates an output XfbBuffer b, then XfbStride s (OpDecorate, or OpMemberDecorate for gl_Position's block)
        { uint32_t d = op == 71 && n >= 4 ? w[i + 2] : op == 72 && n >= 5 ? w[i + 3] : 0, v = op == 71 ? w[i + 3] : op == 72 && n >= 5 ? w[i + 4] : 0;
          if (d == 36) xb = v; else if (d == 37 && xb < 4) s->xfb_stride[xb] = v; }
        i += n; }
    // without VK_EXT_transform_feedback the capture declarations would make the module invalid: they are dropped (the capture's draws then write nothing, as before)
    if (s->xfb && !V.has_xfb) { uint32_t o = 5;
        for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { while (i < nw) w[o++] = w[i++]; break; }
            int drop = (op == 17 && n == 2 && w[i + 1] == 53) || (op == 16 && n >= 3 && w[i + 2] == 11) ||   // OpCapability TransformFeedback, OpExecutionMode Xfb
                       (op == 71 && n >= 3 && (w[i + 2] == 36 || w[i + 2] == 37)) || (op == 72 && n >= 4 && (w[i + 3] == 36 || w[i + 3] == 37));   // XfbBuffer / XfbStride
            if (drop) i += n; else for (uint32_t k = 0; k < n; k++) w[o++] = w[i++]; }
        nw = o; s->xfb = 0; VKRED_ONCE(90, "the device has no transform feedback: shaders that capture it run without"); }
    // the translator declares Float64 in every shader; the device has shaderFloat64 off (Adreno has none), which makes the declaration invalid: when the
    // shader has no 64-bit float type it becomes a second Shader declaration
    if (!f64_used) for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || op != 17) break; if (n == 2 && w[i + 1] == 10) w[i + 1] = 1; i += n; }
    // the translator declares fragment shader interlock in every pixel shader; the device has no VK_EXT_fragment_shader_interlock: when the shader uses no
    // interlock instruction, its OpCapability and OpExtension are dropped
    { int used = 0; for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n) break; if (op == 5364 || op == 5365) used = 1; i += n; }   // OpBegin/EndInvocationInterlockEXT
      if (!used) { uint32_t o = 5;
        for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { while (i < nw) w[o++] = w[i++]; break; }
            int drop = (op == 17 && n == 2 && (w[i + 1] == 5378 || w[i + 1] == 5363 || w[i + 1] == 5372)) ||   // FragmentShader{Pixel,Sample,ShadingRate}InterlockEXT
                       (op == 10 && !memcmp(&w[i + 1], "SPV_EXT_fragment_shader_interlock", 33));
            if (drop) i += n; else for (uint32_t k = 0; k < n; k++) w[o++] = w[i++]; }
        nw = o; } }
    { static int nc = -1; if (nc < 0) { const char *e = getenv("DM_VK_NOCONTRACT"); nc = e && *e ? atoi(e) : 0; } if (nc == 1 || (nc == 2 && (stage == 3 || stage == 4))) w = spv_nocontract(w, &nw, key); }   // r84
    { static int dem = -1; if (dem < 0) { const char *e = getenv("DM_VK_DEMOTE"); dem = e && *e == '1'; } if (stage == 1 && dem && V.has_demote) w = spv_demote(w, &nw, key); }   // r83
    { static int vs = -1; if (vs < 0) { const char *e = getenv("DM_VK_VTX_SHFL"); vs = e && *e ? atoi(e) : 1; } if (vs && stage != 1) w = spv_vtx_shfl(w, &nw, key, vs == 2); }   // r85 (model checked inside: no compute)
    if (stage == 1 && V.has_quad) w = spv_full_quads(w, &nw, key);
    spv_io_locs(w, nw, &s->in_locs, &s->out_locs);
    spv_dref_bindings(w, nw, s->dref);
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, (size_t)nw * 4, w};
    VkResult vr = vkCreateShaderModule(V.dev, &ci, NULL, &s->mod); s->spv = w; s->nspv = nw;
    if (vr) { VKLOG("[RED] shader %016llx: vkCreateShaderModule %d", (unsigned long long)key, vr); free(w); free(s); nvn_hm_put(&g_shaders, &hk, (void *)1); pthread_mutex_unlock(&g_sh_mu); return NULL; }
    // descriptor set layout: one entry per binding
    // The game's constant buffers and the driver segment are dynamic uniform buffers: their offset is given when the set is bound, so a set matches any draw's
    // with the same buffers (Unity updates a constant buffer before nearly every draw, and each update is a shadow at a new offset: NEO made a new set for
    // 94% of its draws, 5 ms a frame). Up to half the device's limit per stage, as the vertex and fragment sets share a pipeline layout. DM_VK_DYN_UBO=0: none.
    static int dyn_on = -1; if (dyn_on < 0) { const char *e = getenv("DM_VK_DYN_UBO"); dyn_on = !(e && !atoi(e)); }
    uint32_t dynmax = dyn_on && stage != 3 && stage != 4 ? V.props.limits.maxDescriptorSetUniformBuffersDynamic / 2 : 0;   // r69: tessellation stages none (the vertex and fragment stages take the layout's whole limit)
    VkDescriptorSetLayoutBinding lb[96]; uint32_t nl = 0;
    for (int i = 0; i < s->nbind && nl < 96; i++) {
        const dmsv_bind *b = &s->b[i];
        int dyn = b->desc == RD_UBO && (b->kind == RK_CBUF || b->kind == RK_DRV) && b->count == 1 && (uint32_t)s->ndyn < dynmax;
        s->dyn[i] = (uint8_t)dyn; s->ndyn += dyn;
        lb[nl++] = (VkDescriptorSetLayoutBinding){b->binding, dyn ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : desc_type(b->desc), b->count ? b->count : 32, stage_flag(stage), NULL};
    }
    s->dsl = shader_layout(lb, nl);
    nvn_hm_put(&g_shaders, &hk, s);
    pthread_mutex_unlock(&g_sh_mu);
    r80_sh(key, stage, t80, t81, 1);   // r80
    return s;
}
// Integer texels. Maxwell's texture instructions return the texel's 32-bit words as they are, and the shader decides what they mean; the translator declares
// every sampled image as float, which matches only float views. A view of an integer format, or the stencil of a depth-stencil texture (DepthStencilMode
// STENCIL), must be read through a uint / int image in Vulkan, and Adreno converts the value otherwise. Xenoblade Chronicles 3 tags each pixel's material
// for its lighting from the stencil's bit 3 (shader 094f4afb9c876459 tests the raw word & 8): read as float, every tag was 0 and the lit scene was black.
// A variant of the module retypes those bindings' images to uint / int and bitcasts each texel read back to float, which gives the shader the raw words.
typedef struct { const uint32_t *w; uint32_t fn, bound, *at, *nt, nnt; struct { uint32_t old, kind, id; } map[64]; int nmap; } spvi;
static uint32_t spvi_find(const uint32_t *a, uint32_t lo, uint32_t hi, uint32_t op, const uint32_t *ops, uint32_t nops) {   // a declaration op with these operands (after the result id)
    for (uint32_t j = lo; j < hi;) { uint32_t m = a[j] >> 16; if (!m) break; if ((a[j] & 0xffff) == op && m == nops + 2 && !memcmp(&a[j + 2], ops, nops * 4)) return a[j + 1]; j += m; }
    return 0;
}
static uint32_t spvi_decl(spvi *c, uint32_t op, const uint32_t *ops, uint32_t nops, int reuse) {   // an existing declaration, or a new one
    uint32_t id = reuse ? spvi_find(c->w, 5, c->fn, op, ops, nops) : 0; if (!id && reuse) id = spvi_find(c->nt, 0, c->nnt, op, ops, nops);
    if (id || c->nnt + nops + 2 > 4096) return id;
    id = c->bound++; c->nt[c->nnt++] = (nops + 2) << 16 | op; c->nt[c->nnt++] = id; memcpy(&c->nt[c->nnt], ops, nops * 4); c->nnt += nops; return id;
}
static uint32_t spvi_scalar(spvi *c, uint32_t k) { uint32_t o[2] = {32, k - 1}; return spvi_decl(c, 21, o, 2, 1); }   // OpTypeInt 32, unsigned (kind 1) / signed (2)
static uint32_t spvi_vec4(spvi *c, uint32_t k) { uint32_t o[2] = {spvi_scalar(c, k), 4}; return spvi_decl(c, 23, o, 2, 1); }
// the type for old with its image's sampled type made uint / int: images, sampled images, arrays of them, pointers to those
static uint32_t spvi_type(spvi *c, uint32_t old, uint32_t k) {
    for (int j = 0; j < c->nmap; j++) if (c->map[j].old == old && c->map[j].kind == k) return c->map[j].id;
    uint32_t i = old < c->bound ? c->at[old] : 0; if (!i) return 0;
    uint32_t n = c->w[i] >> 16, op = c->w[i] & 0xffff, d[12]; if (n > 12 || n < 3) return 0; memcpy(d, &c->w[i + 2], (n - 2) * 4);
    if (op == 25) d[0] = spvi_scalar(c, k);                                                                          // OpTypeImage: the sampled type
    else if (op == 27 || op == 28 || op == 29) { if (!(d[0] = spvi_type(c, d[0], k))) return 0; }                  // OpTypeSampledImage, OpTypeArray, OpTypeRuntimeArray
    else if (op == 32 && n == 4) { if (!(d[1] = spvi_type(c, d[1], k))) return 0; }                                // OpTypePointer
    else return 0;
    uint32_t id = spvi_decl(c, op, d, n - 2, op != 28 && op != 29); if (!id) return 0;
    if (c->nmap < 64) c->map[c->nmap++] = (typeof(c->map[0])){old, k, id};
    return id;
}
static uint32_t *spv_int_images(const uint32_t *w, uint32_t nw, const uint16_t *bind, const uint8_t *kind, int nb, uint32_t *onw) {
    spvi c = {.w = w, .fn = nw, .bound = w[3]};
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16; if (!n) return NULL; if ((w[i] & 0xffff) == 54) { c.fn = i; break; } i += n; }   // the first OpFunction
    uint32_t base = c.bound; c.at = calloc(base, sizeof *c.at); c.nt = malloc(4096 * 4);
    uint8_t *taint = calloc(base, 1);   // ids that are (pointers to) the retyped images: their kind
    for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op >= 19 && op <= 39 && w[i + 1] < base) c.at[w[i + 1]] = i;   // OpType*
        if (op == 71 && n >= 4 && w[i + 2] == 33) for (int k = 0; k < nb; k++) if (w[i + 3] == bind[k] && w[i + 1] < base) taint[w[i + 1]] = kind[k];   // OpDecorate Binding
        i += n; }
    // the functions, rewritten apart: every new type they need is known before the declarations are written
    uint32_t *fb = malloc(((size_t)(nw - c.fn) * 2 + 16) * 4), nf = 0, *mv = malloc(1024 * 4), nmv = 0; int ok = 1;
    for (uint32_t i = 5; i < c.fn && ok;) { uint32_t n = w[i] >> 16;   // the retyped variables, with their new pointer types
        if ((w[i] & 0xffff) == 59 && w[i + 2] < base && taint[w[i + 2]]) { uint32_t t = spvi_type(&c, w[i + 1], taint[w[i + 2]]); if (!t || nmv + n > 1024) ok = 0; else { memcpy(&mv[nmv], &w[i], n * 4); mv[nmv + 1] = t; nmv += n; } }
        i += n; }
    for (uint32_t i = c.fn; i < nw && ok;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n) { ok = 0; break; }
        uint32_t *x = &fb[nf]; memcpy(x, &w[i], n * 4); nf += n;
        uint32_t src = n >= 4 ? w[i + 3] : 0, k = src && src < base ? taint[src] : 0;
        if (k && (op == 61 || op == 65 || op == 66 || op == 83 || op == 100 || op == 86)) {   // Load, AccessChain, InBoundsAccessChain, CopyObject, Image, SampledImage
            uint32_t t = spvi_type(&c, x[1], k); if (!t) { ok = 0; break; } x[1] = t; if (x[2] < base) taint[x[2]] = (uint8_t)k;
        } else if (k && (op == 87 || op == 88 || op == 91 || op == 92 || op == 95 || op == 96)) {   // Sample{Implicit,Explicit}Lod, SampleProj*, Fetch, Gather: a vec4 of the integer type, bitcast back
            uint32_t ft = x[1], r = x[2], r2 = c.bound++; x[1] = spvi_vec4(&c, k); x[2] = r2;
            fb[nf++] = 4u << 16 | 124; fb[nf++] = ft; fb[nf++] = r; fb[nf++] = r2;   // OpBitcast
        }
        i += n; }
    uint32_t *o = NULL, no = 5;
    if (ok && c.nnt < 4096) {
        o = malloc(((size_t)c.fn + c.nnt + nmv + nf + 8) * 4); memcpy(o, w, 20); o[3] = c.bound;
        for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16; if (!((w[i] & 0xffff) == 59 && w[i + 2] < base && taint[w[i + 2]])) { memcpy(&o[no], &w[i], n * 4); no += n; } i += n; }
        memcpy(&o[no], c.nt, c.nnt * 4); no += c.nnt; memcpy(&o[no], mv, nmv * 4); no += nmv; memcpy(&o[no], fb, nf * 4); no += nf;
        *onw = no;
    }
    free(c.at); free(c.nt); free(taint); free(fb); free(mv);
    return o;
}
// A variant of a fragment shader that writes gl_FragDepth = gl_FragCoord.z before each return of its entry point (DepthReplacing): the same depth, tested after the
// shader instead of by Adreno's early culling (gfx_vk_exec.c fdz says when). NULL when the shader writes its depth already or the module is not as expected
static uint32_t *spv_frag_depth(const uint32_t *w, uint32_t nw, uint32_t *onw) {
    spvi c = {.w = w, .fn = nw, .bound = w[3]};
    uint32_t ep = 0, epfn = 0, firsttype = 0, coord = 0, c2 = 0;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) return NULL;
        if (op == 15) { if (ep || n < 3 || w[i + 1] != 4) return NULL; ep = i; epfn = w[i + 2]; }   // one entry point, a Fragment one
        if (op == 71 && n >= 4 && w[i + 2] == 11 && w[i + 3] == 22) return NULL;   // BuiltIn FragDepth: written already
        if (op == 71 && n >= 4 && w[i + 2] == 11 && w[i + 3] == 15) coord = w[i + 1];   // BuiltIn FragCoord
        if (!firsttype && ((op >= 19 && op <= 39) || (op >= 41 && op <= 52) || op == 59)) firsttype = i;   // the first type, constant or variable: the annotations end here
        if (op == 54) { c.fn = i; break; }   // the first OpFunction
        i += n; }
    if (!ep || !firsttype || c.fn == nw) return NULL;
    c.nt = malloc(4096 * 4);
    uint32_t o32[1] = {32}; uint32_t f32 = spvi_decl(&c, 22, o32, 1, 1), i32 = spvi_scalar(&c, 2);
    uint32_t ov4[2] = {f32, 4}; uint32_t v4 = spvi_decl(&c, 23, ov4, 2, 1);
    uint32_t op1[2] = {1, v4}, op2[2] = {1, f32}, op3[2] = {3, f32}; uint32_t pin4 = spvi_decl(&c, 32, op1, 2, 1), pinf = spvi_decl(&c, 32, op2, 2, 1), poutf = spvi_decl(&c, 32, op3, 2, 1);
    if (!f32 || !i32 || !v4 || !pin4 || !pinf || !poutf || c.nnt + 16 > 4096) { free(c.nt); return NULL; }
    for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16; if ((w[i] & 0xffff) == 43 && n == 4 && w[i + 1] == i32 && w[i + 3] == 2) c2 = w[i + 2]; i += n; }   // OpConstant int 2
    if (!c2) { c2 = c.bound++; c.nt[c.nnt++] = 4u << 16 | 43; c.nt[c.nnt++] = i32; c.nt[c.nnt++] = c2; c.nt[c.nnt++] = 2; }
    int newcoord = !coord; if (newcoord) { coord = c.bound++; c.nt[c.nnt++] = 4u << 16 | 59; c.nt[c.nnt++] = pin4; c.nt[c.nnt++] = coord; c.nt[c.nnt++] = 1; }   // OpVariable Input
    uint32_t dv = c.bound++; c.nt[c.nnt++] = 4u << 16 | 59; c.nt[c.nnt++] = poutf; c.nt[c.nnt++] = dv; c.nt[c.nnt++] = 3;   // OpVariable Output
    uint32_t nret = 0; for (uint32_t i = c.fn; i < nw;) { uint32_t n = w[i] >> 16; if (!n) { free(c.nt); return NULL; } nret += (w[i] & 0xffff) == 253; i += n; }
    uint32_t *o = malloc(((size_t)nw + c.nnt + 16 + 12 * nret) * 4), no = 5; memcpy(o, w, 20);
    for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16;
        if (i == firsttype) { uint32_t d[8] = {4u << 16 | 71, dv, 11, 22, 4u << 16 | 71, coord, 11, 15}; memcpy(&o[no], d, (newcoord ? 8 : 4) * 4); no += newcoord ? 8 : 4; }   // BuiltIn FragDepth (FragCoord)
        if (i == ep) { o[no++] = (n + 1 + newcoord) << 16 | 15; memcpy(&o[no], &w[i + 1], (n - 1) * 4); no += n - 1; o[no++] = dv; if (newcoord) o[no++] = coord;   // the entry point lists them
                       o[no++] = 3u << 16 | 16; o[no++] = epfn; o[no++] = 12; }   // OpExecutionMode DepthReplacing
        else { memcpy(&o[no], &w[i], n * 4); no += n; }
        i += n; }
    memcpy(&o[no], c.nt, c.nnt * 4); no += c.nnt;
    int in_ep = 0;
    for (uint32_t i = c.fn; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 54) in_ep = w[i + 2] == epfn; else if (op == 56) in_ep = 0;   // OpFunction / OpFunctionEnd
        if (op == 253 && in_ep) { uint32_t a = c.bound++, b = c.bound++, x[12] = {5u << 16 | 65, pinf, a, coord, c2, 4u << 16 | 61, f32, b, a, 3u << 16 | 62, dv, b};   // AccessChain .z, Load, Store
                                  memcpy(&o[no], x, sizeof x); no += 12; }
        memcpy(&o[no], &w[i], n * 4); no += n; i += n; }
    o[3] = c.bound; *onw = no; free(c.nt); return o;
}
// r56: NVN's alpha test (nvnColorStateSetAlphaTest, nvnCommandBufferSetAlphaRef), which Vulkan has no state for. Bayonetta 3's particles and cut-out leaves draw their
// transparent texels as squares without it. A variant of a fragment shader compares its colour 0 alpha with the reference before each return of its entry point and
// discards the fragment when the test fails; fragment push constants hold the pass mask (NVN function - 1: bit 0 less, bit 1 equal, bit 2 greater, as GL's functions)
// at byte 0 and the reference at byte 4 (gfx_vk_exec.c atest). Only draws with a test take it: a shader that can discard loses Adreno's early depth culling.
// Witch-Time tests at the shader's exit from push constants the same way (runtime/nvn_vk_draw.c). NULL when colour 0 is not a float vec4 or the module is not as expected
static uint32_t *spv_alpha_test(const uint32_t *w, uint32_t nw, uint32_t *onw) {
    spvi c = {.w = w, .fn = nw, .bound = w[3]};
    uint32_t ep = 0, epfn = 0, firsttype = 0, out0 = 0, v4 = 0, f32 = 0;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) return NULL;
        if (op == 15) { if (ep || n < 3 || w[i + 1] != 4) return NULL; ep = i; epfn = w[i + 2]; }   // one entry point, a Fragment one
        if (!firsttype && ((op >= 19 && op <= 39) || (op >= 41 && op <= 52) || op == 59)) firsttype = i;
        if (op == 59 && n >= 4 && w[i + 3] == 9) return NULL;   // a push constant block of its own: one per entry point
        if (op == 54) { c.fn = i; break; }
        i += n; }
    if (!ep || !firsttype || c.fn == nw) return NULL;
    for (uint32_t i = 5; i < c.fn && !out0;) { uint32_t n = w[i] >> 16;   // colour 0: the Output variable at Location 0
        if ((w[i] & 0xffff) == 71 && n == 4 && w[i + 2] == 30 && w[i + 3] == 0)
            for (uint32_t j = 5; j < c.fn;) { uint32_t m = w[j] >> 16; if ((w[j] & 0xffff) == 59 && w[j + 2] == w[i + 1]) { if (w[j + 3] == 3 && spv_is_vec4(w, c.fn, w[j + 1])) {
                        for (uint32_t k = 5; k < c.fn;) { uint32_t q = w[k] >> 16; if ((w[k] & 0xffff) == 32 && w[k + 1] == w[j + 1]) { v4 = w[k + 3]; break; } k += q; }
                        out0 = w[i + 1]; } break; }
                j += m; }
        i += n; }
    if (out0 && v4) for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16; if ((w[i] & 0xffff) == 23 && w[i + 1] == v4) { f32 = w[i + 2]; break; } i += n; }
    if (!out0 || !v4 || !f32) return NULL;
    c.nt = malloc(4096 * 4); uint32_t none[1] = {0};
    uint32_t u32 = spvi_scalar(&c, 1), bl = spvi_decl(&c, 20, none, 0, 1);   // OpTypeInt 32 0, OpTypeBool
    if (!u32 || !bl) { free(c.nt); return NULL; }
    uint32_t st = c.bound++; c.nt[c.nnt++] = 4u << 16 | 30; c.nt[c.nnt++] = st; c.nt[c.nnt++] = u32; c.nt[c.nnt++] = f32;   // OpTypeStruct {uint mask; float ref}: its own, for its Block decoration
    uint32_t o1[2] = {9, st}, o2[2] = {9, u32}, o3[2] = {9, f32};
    uint32_t ppc = spvi_decl(&c, 32, o1, 2, 1), ppu = spvi_decl(&c, 32, o2, 2, 1), ppf = spvi_decl(&c, 32, o3, 2, 1);   // PushConstant pointers
    if (!ppc || !ppu || !ppf || c.nnt + 24 > 4096) { free(c.nt); return NULL; }
    uint32_t k[4], kv[4] = {0, 1, 2, 4};
    for (int j = 0; j < 4; j++) { k[j] = c.bound++; c.nt[c.nnt++] = 4u << 16 | 43; c.nt[c.nnt++] = u32; c.nt[c.nnt++] = k[j]; c.nt[c.nnt++] = kv[j]; }   // OpConstant uint
    uint32_t pcv = c.bound++; c.nt[c.nnt++] = 4u << 16 | 59; c.nt[c.nnt++] = ppc; c.nt[c.nnt++] = pcv; c.nt[c.nnt++] = 9;   // OpVariable PushConstant
    int iface = w[1] >= 0x10400;   // SPIR-V 1.4 lists every global the entry point uses; 1.3 only its inputs and outputs
    uint32_t nret = 0; for (uint32_t i = c.fn; i < nw;) { uint32_t n = w[i] >> 16; if (!n) { free(c.nt); return NULL; } nret += (w[i] & 0xffff) == 253; i += n; }
    uint32_t *o = malloc(((size_t)nw + c.nnt + 16 + 96 * nret) * 4), no = 5; memcpy(o, w, 20);
    for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16;
        if (i == firsttype) { uint32_t d[13] = {3u << 16 | 71, st, 2, 5u << 16 | 72, st, 0, 35, 0, 5u << 16 | 72, st, 1, 35, 4}; memcpy(&o[no], d, sizeof d); no += 13; }   // Block, member Offsets 0 and 4
        if (i == ep && iface) { o[no++] = (n + 1) << 16 | 15; memcpy(&o[no], &w[i + 1], (n - 1) * 4); no += n - 1; o[no++] = pcv; }
        else { memcpy(&o[no], &w[i], n * 4); no += n; }
        i += n; }
    memcpy(&o[no], c.nt, c.nnt * 4); no += c.nnt;
    int in_ep = 0;
    for (uint32_t i = c.fn; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 54) in_ep = w[i + 2] == epfn; else if (op == 56) in_ep = 0;
        if (op == 253 && in_ep) {
            uint32_t col = c.bound++, a = c.bound++, pm = c.bound++, m = c.bound++, pr = c.bound++, r = c.bound++, lt = c.bound++, eq = c.bound++, gt = c.bound++;
            uint32_t s1 = c.bound++, s2 = c.bound++, s4 = c.bound++, or1 = c.bound++, or2 = c.bound++, an = c.bound++, fail = c.bound++, lk = c.bound++, lok = c.bound++;
            uint32_t x[] = {4u << 16 | 61, v4, col, out0,   5u << 16 | 81, f32, a, col, 3,                  // the colour as written, its alpha
                            5u << 16 | 65, ppu, pm, pcv, k[0],   4u << 16 | 61, u32, m, pm,   5u << 16 | 65, ppf, pr, pcv, k[1],   4u << 16 | 61, f32, r, pr,
                            5u << 16 | 184, bl, lt, a, r,   5u << 16 | 180, bl, eq, a, r,   5u << 16 | 186, bl, gt, a, r,   // FOrdLessThan, FOrdEqual, FOrdGreaterThan
                            6u << 16 | 169, u32, s1, lt, k[1], k[0],   6u << 16 | 169, u32, s2, eq, k[2], k[0],   6u << 16 | 169, u32, s4, gt, k[3], k[0],
                            5u << 16 | 197, u32, or1, s1, s2,   5u << 16 | 197, u32, or2, or1, s4,   5u << 16 | 199, u32, an, or2, m,   5u << 16 | 170, bl, fail, an, k[0],
                            3u << 16 | 247, lok, 0,   4u << 16 | 250, fail, lk, lok,   2u << 16 | 248, lk,   1u << 16 | 252,   2u << 16 | 248, lok};   // if (fail) kill
            memcpy(&o[no], x, sizeof x); no += sizeof x / 4; }
        memcpy(&o[no], &w[i], n * 4); no += n; i += n; }
    o[3] = c.bound; *onw = no; free(c.nt); return o;
}
// r132 (diagnostic, gfx_vk_exec.c DM_VK_X_PUSH): a variant of a fragment shader that adds the first fragment push constant word, read as a float, to colour 0's alpha
// before each return of its entry point. The host sets that word to 0, so the colour stays the same; the shader reads it, so the driver loads it for every draw that
// sets it. NULL when colour 0 is not a float vec4, the shader has a push constant block of its own, or the module is not as expected
static uint32_t *spv_push_read(const uint32_t *w, uint32_t nw, uint32_t *onw) {
    spvi c = {.w = w, .fn = nw, .bound = w[3]};
    uint32_t ep = 0, epfn = 0, firsttype = 0, out0 = 0, v4 = 0, f32 = 0;
    for (uint32_t i = 5; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) return NULL;
        if (op == 15) { if (ep || n < 3 || w[i + 1] != 4) return NULL; ep = i; epfn = w[i + 2]; }
        if (!firsttype && ((op >= 19 && op <= 39) || (op >= 41 && op <= 52) || op == 59)) firsttype = i;
        if (op == 59 && n >= 4 && w[i + 3] == 9) return NULL;
        if (op == 54) { c.fn = i; break; }
        i += n; }
    if (!ep || !firsttype || c.fn == nw) return NULL;
    for (uint32_t i = 5; i < c.fn && !out0;) { uint32_t n = w[i] >> 16;
        if ((w[i] & 0xffff) == 71 && n == 4 && w[i + 2] == 30 && w[i + 3] == 0)
            for (uint32_t j = 5; j < c.fn;) { uint32_t m = w[j] >> 16; if ((w[j] & 0xffff) == 59 && w[j + 2] == w[i + 1]) { if (w[j + 3] == 3 && spv_is_vec4(w, c.fn, w[j + 1])) {
                        for (uint32_t k = 5; k < c.fn;) { uint32_t q = w[k] >> 16; if ((w[k] & 0xffff) == 32 && w[k + 1] == w[j + 1]) { v4 = w[k + 3]; break; } k += q; }
                        out0 = w[i + 1]; } break; }
                j += m; }
        i += n; }
    if (out0 && v4) for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16; if ((w[i] & 0xffff) == 23 && w[i + 1] == v4) { f32 = w[i + 2]; break; } i += n; }
    if (!out0 || !v4 || !f32) return NULL;
    c.nt = malloc(4096 * 4);
    uint32_t u32 = spvi_scalar(&c, 1);
    if (!u32) { free(c.nt); return NULL; }
    uint32_t st = c.bound++; c.nt[c.nnt++] = 3u << 16 | 30; c.nt[c.nnt++] = st; c.nt[c.nnt++] = u32;   // OpTypeStruct {uint}: its own, for its Block decoration
    uint32_t o1[2] = {9, st}, o2[2] = {9, u32};
    uint32_t ppc = spvi_decl(&c, 32, o1, 2, 1), ppu = spvi_decl(&c, 32, o2, 2, 1);
    if (!ppc || !ppu || c.nnt + 16 > 4096) { free(c.nt); return NULL; }
    uint32_t k0 = c.bound++; c.nt[c.nnt++] = 4u << 16 | 43; c.nt[c.nnt++] = u32; c.nt[c.nnt++] = k0; c.nt[c.nnt++] = 0;
    uint32_t pcv = c.bound++; c.nt[c.nnt++] = 4u << 16 | 59; c.nt[c.nnt++] = ppc; c.nt[c.nnt++] = pcv; c.nt[c.nnt++] = 9;
    int iface = w[1] >= 0x10400;
    uint32_t nret = 0; for (uint32_t i = c.fn; i < nw;) { uint32_t n = w[i] >> 16; if (!n) { free(c.nt); return NULL; } nret += (w[i] & 0xffff) == 253; i += n; }
    uint32_t *o = malloc(((size_t)nw + c.nnt + 16 + 40 * nret) * 4), no = 5; memcpy(o, w, 20);
    for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16;
        if (i == firsttype) { uint32_t d[8] = {3u << 16 | 71, st, 2, 5u << 16 | 72, st, 0, 35, 0}; memcpy(&o[no], d, sizeof d); no += 8; }   // Block, member Offset 0
        if (i == ep && iface) { o[no++] = (n + 1) << 16 | 15; memcpy(&o[no], &w[i + 1], (n - 1) * 4); no += n - 1; o[no++] = pcv; }
        else { memcpy(&o[no], &w[i], n * 4); no += n; }
        i += n; }
    memcpy(&o[no], c.nt, c.nnt * 4); no += c.nnt;
    int in_ep = 0;
    for (uint32_t i = c.fn; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 54) in_ep = w[i + 2] == epfn; else if (op == 56) in_ep = 0;
        if (op == 253 && in_ep) {
            uint32_t col = c.bound++, a = c.bound++, pm = c.bound++, m = c.bound++, mf = c.bound++, a2 = c.bound++, col2 = c.bound++;
            uint32_t x[] = {4u << 16 | 61, v4, col, out0,   5u << 16 | 81, f32, a, col, 3,   5u << 16 | 65, ppu, pm, pcv, k0,   4u << 16 | 61, u32, m, pm,
                            4u << 16 | 124, f32, mf, m,   5u << 16 | 129, f32, a2, a, mf,   6u << 16 | 82, v4, col2, a2, col, 3,   3u << 16 | 62, out0, col2};   // alpha += bitcast(word 0)
            memcpy(&o[no], x, sizeof x); no += sizeof x / 4; }
        memcpy(&o[no], &w[i], n * 4); no += n; i += n; }
    o[3] = c.bound; *onw = no; free(c.nt); return o;
}
// r133 (diagnostic, gfx_vk_exec.c DM_VK_X_POFS): a variant that reads the shader's game constant buffers (dynamic uniform buffers of kind cbuf, the first 8 by binding)
// at an offset given in push constants: each access chain into such a block gets push word (vertex 4 + slot, fragment 12 + slot) added to its vec4 index. The push block
// is uint[20] at offset 0. NULL when the shader has a push constant block of its own, uses such a block other than through an access chain with an element index, or the
// index is not a 32-bit integer
// r151 irun: the words are read from instance inputs in place of the push block: vertex words 4-11 and fragment words 12-19 at vertex inputs 28-31 (uvec4), the fragment
// ones passed on by the vertex shader at outputs 28-29 and read flat there by the fragment shader. A vertex shader takes the variant without buffers of its own to convert
static uint32_t *spv_pofs(const uint32_t *w, uint32_t nw, const vkshader_t *s, uint32_t msk, int irun, uint32_t *onw) {   // r134 msk: the slots converted (0: all)
    uint32_t bnd[8], wd[8]; int nb = 0;
    { int ord[96], nd = 0; for (int i = 0; i < s->nbind && i < 96; i++) if (s->dyn[i]) { int j = nd++; while (j > 0 && s->b[ord[j - 1]].binding > s->b[i].binding) { ord[j] = ord[j - 1]; j--; } ord[j] = i; }
      for (int j = 0; j < nd && j < 8; j++) if (s->b[ord[j]].kind == RK_CBUF && (!msk || (msk >> j & 1))) { bnd[nb] = s->b[ord[j]].binding; wd[nb++] = (s->stage == 1 ? 12u : 4u) + (uint32_t)j; } }
    int vs = s->stage == 0; if (irun && s->stage > 1) return NULL;
    if ((!nb && !(irun && vs)) || nw < 6) return NULL;
    uint32_t bound = w[3]; spvi c = {.w = w, .fn = nw, .bound = bound};
    uint32_t *bindv = calloc(bound, 4), *setv = calloc(bound, 4), *tgt = calloc(bound, 4), *tyof = calloc(bound, 4); uint8_t *isint = calloc(bound, 1);
    uint32_t ep = 0, firsttype = 0, ntgt = 0, nhit = 0, *o = NULL; int bad = !bindv || !setv || !tgt || !tyof || !isint;
    for (uint32_t i = 5; i < nw && !bad;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { bad = 1; break; }
        if (op == 15) { if (ep) bad = 1; ep = i; }
        if (!firsttype && ((op >= 19 && op <= 39) || (op >= 41 && op <= 52) || op == 59)) firsttype = i;
        if (op == 71 && n >= 4 && w[i + 1] < bound) { if (w[i + 2] == 33) bindv[w[i + 1]] = w[i + 3] + 1; else if (w[i + 2] == 34) setv[w[i + 1]] = w[i + 3] + 1; }
        if (op == 21 && n == 4 && w[i + 1] < bound && w[i + 2] == 32) isint[w[i + 1]] = w[i + 3] ? 2 : 1;
        if ((op == 43 || op == 50) && n >= 4 && w[i + 2] < bound) tyof[w[i + 2]] = w[i + 1];
        if (op == 59 && n >= 4 && w[i + 3] == 9) bad = 1;
        if (op == 54) { c.fn = i; break; }
        i += n; }
    if (!bad && (!ep || !firsttype || c.fn == nw)) bad = 1;
    for (uint32_t i = 5; i < c.fn && !bad;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        if (op == 59 && n >= 4 && w[i + 3] == 2 && w[i + 2] < bound && bindv[w[i + 2]]) for (int k = 0; k < nb; k++)
            if (bindv[w[i + 2]] - 1 == bnd[k] && (!setv[w[i + 2]] || setv[w[i + 2]] - 1 == (uint32_t)s->set)) { tgt[w[i + 2]] = wd[k] + 1; ntgt++; }
        i += n; }
    if (!bad && !ntgt && !(irun && vs)) bad = 1;
    for (uint32_t i = c.fn; i < nw && !bad;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff; if (!n || i + n > nw) { bad = 1; break; }
        int res = n >= 3 && !(op == 8 || op == 317 || op == 248 || op == 249 || op == 250 || op == 251 || op == 252 || op == 253 || op == 254 || op == 255 || op == 246 || op == 247 ||
                              op == 62 || op == 63 || op == 56 || op == 218 || op == 219 || op == 224 || op == 225 || op == 4416);
        if (res && w[i + 2] < bound) tyof[w[i + 2]] = w[i + 1];
        i += n; }
    for (uint32_t i = c.fn; i < nw && !bad;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
        for (uint32_t k = 1; k < n; k++) if (w[i + k] < bound && tgt[w[i + k]]) {
            if (op == 12 && k == 4) continue;   // r147: OpExtInst instruction number (a literal, not an id)
            if (!((op == 65 || op == 66) && k == 3 && n >= 6)) { bad = 1; break; }
            uint32_t t = w[i + 5] < bound ? tyof[w[i + 5]] : 0; if (!t || t >= bound || !isint[t]) { bad = 1; break; }
            nhit++; }
        i += n; }
    if (!bad && !nhit && !(irun && vs)) bad = 1;
    if (!bad) {
        c.nt = malloc(4096 * 4);
        uint32_t u32 = spvi_scalar(&c, 1);
        if (!u32 || c.nnt + 64 > 4096) bad = 1;
        else if (irun) {   // r151
            uint32_t uv4 = spvi_vec4(&c, 1), pin[2] = {1, uv4}, pout[2] = {3, uv4}, ppi = uv4 ? spvi_decl(&c, 32, pin, 2, 1) : 0, ppo = uv4 && vs ? spvi_decl(&c, 32, pout, 2, 1) : 1;
            uint32_t ni = vs ? 4 : 2, nv = ni + (vs ? 2 : 0), iv[4] = {0}, ov[2] = {0}, entry = w[ep + 2]; int fwd = 0;
            if (!uv4 || !ppi || !ppo || c.nnt + 32 > 4096) bad = 1;
            else {
                for (uint32_t q = 0; q < ni; q++) { iv[q] = c.bound++; c.nt[c.nnt++] = 4u << 16 | 59; c.nt[c.nnt++] = ppi; c.nt[c.nnt++] = iv[q]; c.nt[c.nnt++] = 1; }
                for (uint32_t q = 0; vs && q < 2; q++) { ov[q] = c.bound++; c.nt[c.nnt++] = 4u << 16 | 59; c.nt[c.nnt++] = ppo; c.nt[c.nnt++] = ov[q]; c.nt[c.nnt++] = 3; }
                o = malloc(((size_t)nw + c.nnt + 96 + 24 * (size_t)nhit) * 4); uint32_t no = 5; memcpy(o, w, 20);
                for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16;
                    if (i == firsttype) {   // Location (and Flat for the fragment inputs)
                        for (uint32_t q = 0; q < ni; q++) { o[no++] = 4u << 16 | 71; o[no++] = iv[q]; o[no++] = 30; o[no++] = 28 + q; if (!vs) { o[no++] = 3u << 16 | 71; o[no++] = iv[q]; o[no++] = 14; } }
                        for (uint32_t q = 0; vs && q < 2; q++) { o[no++] = 4u << 16 | 71; o[no++] = ov[q]; o[no++] = 30; o[no++] = 28 + q; } }
                    if (i == ep) { o[no++] = (n + nv) << 16 | 15; memcpy(&o[no], &w[i + 1], (n - 1) * 4); no += n - 1; for (uint32_t q = 0; q < ni; q++) o[no++] = iv[q]; for (uint32_t q = 0; vs && q < 2; q++) o[no++] = ov[q]; }
                    else { memcpy(&o[no], &w[i], n * 4); no += n; }
                    i += n; }
                memcpy(&o[no], c.nt, c.nnt * 4); no += c.nnt;
                int inent = 0, pend = 0;
                for (uint32_t i = c.fn; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
                    if (op == 54) inent = w[i + 2] == entry;
                    if (pend && op != 59 && op != 8 && op != 317) { pend = 0; fwd = 1;   // the vertex shader: the fragment words passed on, first thing in its entry point
                        for (uint32_t q = 0; q < 2; q++) { uint32_t t = c.bound++; o[no++] = 4u << 16 | 61; o[no++] = uv4; o[no++] = t; o[no++] = iv[2 + q]; o[no++] = 3u << 16 | 62; o[no++] = ov[q]; o[no++] = t; } }
                    if ((op == 65 || op == 66) && n >= 6 && w[i + 3] < bound && tgt[w[i + 3]]) {
                        uint32_t W = tgt[w[i + 3]] - 1, e = W - (vs ? 4u : 12u), T = tyof[w[i + 5]], t = c.bound++, pv = c.bound++, sum = c.bound++, add = pv;
                        o[no++] = 4u << 16 | 61; o[no++] = uv4; o[no++] = t; o[no++] = iv[e / 4 & 1]; o[no++] = 5u << 16 | 81; o[no++] = u32; o[no++] = pv; o[no++] = t; o[no++] = e % 4;   // the word: Load, CompositeExtract
                        if (T != u32) { add = c.bound++; o[no++] = 4u << 16 | 124; o[no++] = T; o[no++] = add; o[no++] = pv; }   // Bitcast to the index type
                        o[no++] = 5u << 16 | 128; o[no++] = T; o[no++] = sum; o[no++] = w[i + 5]; o[no++] = add;   // IAdd
                        memcpy(&o[no], &w[i], n * 4); o[no + 5] = sum; no += n; }
                    else { memcpy(&o[no], &w[i], n * 4); no += n; }
                    if (op == 248 && inent && vs && !fwd) pend = 1;
                    i += n; }
                if (vs && !fwd) bad = 1;
                o[3] = c.bound; *onw = no; } }
        else {
            uint32_t klen = c.bound++; c.nt[c.nnt++] = 4u << 16 | 43; c.nt[c.nnt++] = u32; c.nt[c.nnt++] = klen; c.nt[c.nnt++] = 20;
            uint32_t k0 = c.bound++; c.nt[c.nnt++] = 4u << 16 | 43; c.nt[c.nnt++] = u32; c.nt[c.nnt++] = k0; c.nt[c.nnt++] = 0;
            uint32_t kw[20] = {0}; for (int k = 0; k < nb; k++) if (!kw[wd[k]]) { kw[wd[k]] = c.bound++; c.nt[c.nnt++] = 4u << 16 | 43; c.nt[c.nnt++] = u32; c.nt[c.nnt++] = kw[wd[k]]; c.nt[c.nnt++] = wd[k]; }
            uint32_t arr = c.bound++; c.nt[c.nnt++] = 4u << 16 | 28; c.nt[c.nnt++] = arr; c.nt[c.nnt++] = u32; c.nt[c.nnt++] = klen;   // OpTypeArray uint[20], its own (ArrayStride)
            uint32_t st = c.bound++; c.nt[c.nnt++] = 3u << 16 | 30; c.nt[c.nnt++] = st; c.nt[c.nnt++] = arr;   // OpTypeStruct, its own (Block)
            uint32_t o1[2] = {9, st}, o2[2] = {9, u32}; uint32_t ppc = spvi_decl(&c, 32, o1, 2, 1), ppu = spvi_decl(&c, 32, o2, 2, 1);
            if (!ppc || !ppu || c.nnt + 8 > 4096) bad = 1;
            else {
                uint32_t pcv = c.bound++; c.nt[c.nnt++] = 4u << 16 | 59; c.nt[c.nnt++] = ppc; c.nt[c.nnt++] = pcv; c.nt[c.nnt++] = 9;
                int iface = w[1] >= 0x10400;
                o = malloc(((size_t)nw + c.nnt + 32 + 20 * (size_t)nhit) * 4); uint32_t no = 5; memcpy(o, w, 20);
                for (uint32_t i = 5; i < c.fn;) { uint32_t n = w[i] >> 16;
                    if (i == firsttype) { uint32_t d[12] = {4u << 16 | 71, arr, 6, 4, 3u << 16 | 71, st, 2, 5u << 16 | 72, st, 0, 35, 0}; memcpy(&o[no], d, sizeof d); no += 12; }   // ArrayStride 4, Block, Offset 0
                    if (i == ep && iface) { o[no++] = (n + 1) << 16 | 15; memcpy(&o[no], &w[i + 1], (n - 1) * 4); no += n - 1; o[no++] = pcv; }
                    else { memcpy(&o[no], &w[i], n * 4); no += n; }
                    i += n; }
                memcpy(&o[no], c.nt, c.nnt * 4); no += c.nnt;
                for (uint32_t i = c.fn; i < nw;) { uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
                    if ((op == 65 || op == 66) && n >= 6 && w[i + 3] < bound && tgt[w[i + 3]]) {
                        uint32_t W = tgt[w[i + 3]] - 1, T = tyof[w[i + 5]], pp = c.bound++, pv = c.bound++, sum = c.bound++, add = pv;
                        uint32_t x[10] = {5u << 16 | 65, ppu, pp, pcv, k0, kw[W],   4u << 16 | 61, u32, pv, pp}; x[0] = 6u << 16 | 65; memcpy(&o[no], x, sizeof x); no += 10;   // the word: AccessChain, Load
                        if (T != u32) { add = c.bound++; o[no++] = 4u << 16 | 124; o[no++] = T; o[no++] = add; o[no++] = pv; }   // Bitcast to the index type
                        o[no++] = 5u << 16 | 128; o[no++] = T; o[no++] = sum; o[no++] = w[i + 5]; o[no++] = add;   // IAdd
                        memcpy(&o[no], &w[i], n * 4); o[no + 5] = sum; no += n; }
                    else { memcpy(&o[no], &w[i], n * 4); no += n; }
                    i += n; }
                o[3] = c.bound; *onw = no; } }
        free(c.nt); }
    free(bindv); free(setv); free(tgt); free(tyof); free(isint);
    if (bad) { free(o); return NULL; }
    return o;
}
// DM_VK_SPV_DIR=<dir> with DM_VK_SPV_ALT=1 (live, debugging): pipelines made while it is on use <dir>/<key>.spv in place of the bundle's code where the file
// exists (same bindings), so two compilations of a shader can be compared in one run: the translator's module through spirv-opt, its GLSL through glslc...
static VkShaderModule alt_module(vkshader_t *s) {
    const char *dir = getenv("DM_VK_SPV_DIR"); char path[512]; if (!dir) return s->mod;
    snprintf(path, sizeof path, "%s/%016llx.spv", dir, (unsigned long long)s->key); FILE *f = fopen(path, "rb"); if (!f) return s->mod;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); uint32_t *w = n > 20 && !(n & 3) ? malloc((size_t)n) : NULL; VkShaderModule m = s->mod;
    if (w && fread(w, 1, (size_t)n, f) == (size_t)n) { VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, (size_t)n, w};
        for (uint32_t i = 5, nw = (uint32_t)(n / 4); i < nw;) { uint32_t k = w[i] >> 16; if (!k) break; if ((w[i] & 0xffff) == 71 && k >= 4 && w[i + 2] == 34) w[i + 3] = (uint32_t)s->set; i += k; }   // the stage's set, as at load
        if (vkCreateShaderModule(V.dev, &ci, NULL, &m)) m = s->mod; else VKLOG("shader %016llx: alternative code from %s", (unsigned long long)s->key, path); }
    free(w); fclose(f); return m;
}
// r146: whether DM_VK_SPV_DIR holds <key>.spv for this shader: DM_VK_SPV_ALT then replaces it even where it has integer-texel variants (gfx_vk_exec.c),
// so such a file must carry the integer texel types itself. Before, the switch skipped those shaders (Hyrule Warriors: Age of Imprisonment lighting 3f1082fe)
int vk_spv_alt_has(const vkshader_t *s) {
    const char *dir = getenv("DM_VK_SPV_DIR"); char path[512]; if (!dir || !s) return 0;
    snprintf(path, sizeof path, "%s/%016llx.spv", dir, (unsigned long long)s->key); FILE *f = fopen(path, "rb"); if (!f) return 0; fclose(f); return 1;
}
VkShaderModule vk_shader_module(vkshader_t *s, uint64_t um, uint64_t im, uint32_t xo, int fd, uint32_t xp, uint32_t xr) {
    if (s->stage != 1) fd &= 0xff18; else if (s->depth_repl) fd &= ~1;   // r133: bit 3 (any stage) the pushed-offset variant (spv_pofs)   // fd: bit 0 fragment depth written (spv_frag_depth), bit 1 the alpha test (spv_alpha_test)
    // xp: a geometry shader's added outputs the vertex shader writes, passed on per vertex (spv_gs_pass), read at the locations xr when that is set; a vertex shader's
    // outputs it also writes at the locations xr (spv_vs_dup)
    if (s->stage == 2) xp &= xo & ~(1u << 31); else if (s->stage != 0 || !xr) xp = 0;
    if (s->stage == 0 && xr >= ~1u) xp = 0;   // DM_VK_GS_PASS=3 / 4: the vertex shader is left as it is
    if (!xp || (xr < ~1u && __builtin_popcount(xr) != __builtin_popcount(xp))) xr = 0;
    if (!(um | im | xo | (uint32_t)fd | xp) || !s->spv) return s->mod;
    if (xo == 1u << 31) {   // DM_VK_SPV_ALT
        pthread_mutex_lock(&g_sh_mu); for (int i = 0; i < s->nvar; i++) if (s->var[i].um == 0 && s->var[i].im == 0 && s->var[i].xo == xo) { VkShaderModule m = s->var[i].mod; pthread_mutex_unlock(&g_sh_mu); return m; }
        VkShaderModule m = alt_module(s); if (s->nvar < 8) s->var[s->nvar++] = (typeof(s->var[0])){0, 0, xo, 0, m}; pthread_mutex_unlock(&g_sh_mu); return m; }
    xo &= ~(1u << 31);
    pthread_mutex_lock(&g_sh_mu);
    for (int i = 0; i < s->nvar; i++) if (s->var[i].um == um && s->var[i].im == im && s->var[i].xo == xo && s->var[i].fd == fd && s->var[i].xp == xp && s->var[i].xr == xr) { VkShaderModule m = s->var[i].mod; pthread_mutex_unlock(&g_sh_mu); return m; }
    uint16_t bind[64]; uint8_t kind[64]; int nb = 0;
    for (int i = 0; i < s->nbind && i < 64; i++) if ((um | im) >> i & 1) { bind[nb] = s->b[i].binding; kind[nb++] = um >> i & 1 ? 1 : 2; }
    uint32_t nw = s->nspv, *w = nb ? spv_int_images(s->spv, s->nspv, bind, kind, nb, &nw) : NULL; VkShaderModule m = s->mod;
    if (xo && (w || !nb)) { uint32_t n2 = 0, *w2 = spv_add_outputs(w ? w : s->spv, nw, xo, &n2); free(w); w = w2; nw = n2; }
    uint32_t xpk = xp;   // the variant's key: what was asked for
    if (xp && s->stage == 0 && (w || !nb)) { uint32_t n2 = 0, *w2 = spv_vs_dup(w ? w : s->spv, nw, xp, xr, &n2);
        if (w2) { free(w); w = w2; nw = n2; } else { VKLOG("[RED] shader %016llx: outputs %#x not copied to %#x", (unsigned long long)s->key, xp, xr); xp = 0; } }
    else if (xp && w) { uint32_t n2 = 0, *w2 = spv_gs_pass(w, nw, xp, xr, &n2);
        if (w2) { free(w); w = w2; nw = n2; } else { VKLOG("[RED] shader %016llx: outputs %#x not passed on from the vertex shader (not a passthrough shape): left undefined", (unsigned long long)s->key, xp); xp = 0; } }
    if ((fd & 1) && (w || !(nb || xo || xp))) { uint32_t n2 = 0, *w2 = spv_frag_depth(w ? w : s->spv, nw, &n2); free(w); w = w2; nw = n2; }
    if ((fd & 2) && (w || !(nb || xo || xp || (fd & 1)))) { uint32_t n2 = 0, *w2 = spv_alpha_test(w ? w : s->spv, nw, &n2); free(w); w = w2; nw = n2; }
    if ((fd & 4) && (w || !(nb || xo || xp || (fd & 3)))) { uint32_t n2 = 0, *w2 = spv_push_read(w ? w : s->spv, nw, &n2); free(w); w = w2; nw = n2; }   // r132
    if ((fd & 8) && (w || !(nb || xo || xp || (fd & 7)))) { uint32_t n2 = 0, *w2 = spv_pofs(w ? w : s->spv, nw, s, (uint32_t)fd >> 8 & 0xff, fd >> 4 & 1, &n2); free(w); w = w2; nw = n2; }   // r133
    if (w) { VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, (size_t)nw * 4, w};
        if (vkCreateShaderModule(V.dev, &ci, NULL, &m)) { m = s->mod; VKLOG("[RED] shader %016llx: its variant was refused (bindings %#llx / %#llx, outputs %#x, fragment depth / alpha test %d)", (unsigned long long)s->key, (unsigned long long)um, (unsigned long long)im, xo, fd); }
        else VKLOG("shader %016llx: variant for bindings %#llx (uint texels) %#llx (int texels), outputs added %#x (passed on from the vertex shader %#x, by locations %#x), fragment depth written / alpha test (bits 0 / 1) %d", (unsigned long long)s->key, (unsigned long long)um, (unsigned long long)im, xo, xp, xr, fd);
        free(w); }
    else VKLOG("[RED] shader %016llx: no variant could be made (bindings %#llx / %#llx, outputs %#x, fragment depth / alpha test %d)", (unsigned long long)s->key, (unsigned long long)um, (unsigned long long)im, xo, fd);
    if (s->nvar < 8) s->var[s->nvar++] = (typeof(s->var[0])){um, im, xo, fd, m, xpk, xr};
    pthread_mutex_unlock(&g_sh_mu);
    return m;
}
// static buffers: Ryujinx's support buffer (render scales 1.0, nothing else set: as nvn_mtl.m support_buf) and 64 KB of zeros (unbound uniform slots, driver segments nobody reads)
static VkBuffer static_buf(int which) {
    static VkBuffer b[3]; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;   // rptfix: 2 = the write sink (vk_wsink_buffer)
    { VkBuffer r = __atomic_load_n(&b[which], __ATOMIC_ACQUIRE); if (r) return r; }   // made already: no lock (every draw binds them)
    pthread_mutex_lock(&mu);
    if (!b[which]) {
        VkBuffer nb; VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 65536, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &nb)); VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, nb, &req);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, which == 2 ? vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)
                                                                                                    : vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
        VkDeviceMemory m; VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, nb, m, 0));
        if (which != 2) { void *p; VKCHK(vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, &p)); memset(p, 0, 65536);
        if (which == 0) for (int i = 0; i < 73; i++) *(float *)((uint8_t *)p + 192 + 16 * i) = 1.0f; }   // render_scale[i].x (SupportBuffer.cs layout); the sink is never read, so not filled
        __atomic_store_n(&b[which], nb, __ATOMIC_RELEASE);   // published once filled
    }
    pthread_mutex_unlock(&mu);
    return b[which];
}
VkBuffer vk_support_buffer(void) { return static_buf(0); }
VkBuffer vk_zero_buffer(void) { return static_buf(1); }
// rptfix: where a binding the shader WRITES has no memory to go to (address outside every pool: the zero stand-in). Its stores went into vk_zero_buffer, which other
// bindings read as zeros (unresolved storage/uniform buffers, missing vertex buffers, shaders without c1): those reads then got whatever the last dropped store left
// (OoT: 39e60874 / 4ef1f343 store through c1-addressed pointers, ~2,150 compute stand-ins a second). Written stand-ins now go into a device-local sink of their own,
// so the zero buffer stays zero. DM_VK_WSINK=0: the zero buffer as before
VkBuffer vk_wsink_buffer(void) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_WSINK"); on = !(e && *e == '0'); VKLOG("rptfix: written stand-ins go to %s (DM_VK_WSINK=0: the zero buffer)", on ? "a write sink of their own" : "the shared zero buffer"); }
    return on ? static_buf(2) : static_buf(1);
}
// a shader's c1 constants: copied once into a buffer of their own (the bundle bytes do not change)
uint64_t g_c1_bufs;   // c1 buffers made (one per shader with c1 constants; gfx_vk_core.c NVN2 memory report)
uint64_t g_c1_arena_bytes;   // NVN2: bytes of the c1 blocks made (c1_arena)
// NVN2 (Switch 2 boots): c1 constants are placed in shared 4 MB host-visible blocks (kept for the run, as the per-shader buffers are), one allocation per block instead
// of one per shader. HWAoI 11-16 died at 587 s on this allocation ("FATAL vkAllocateMemory ... -2", gfx_vk_shader.c:1010, the driver's heap at 96% of its budget).
// A block that cannot be made is not fatal: the shader reads zero constants (the zero buffer) and the next use tries again. Returns 0 then
int dm_sdk_ounce(void);
static int c1_arena(vkshader_t *s, VkDeviceSize n) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static VkBuffer blk; static uint8_t *bp; static VkDeviceSize bused, bsize;
    VkDeviceSize al = V.props.limits.minUniformBufferOffsetAlignment > 256 ? V.props.limits.minUniformBufferOffsetAlignment : 256;
    pthread_mutex_lock(&mu);
    VkDeviceSize o = (bused + al - 1) & ~(al - 1);
    if (!blk || o + n > bsize) {
        VkDeviceSize sz = n > (4u << 20) ? (n + 0xffff) & ~(VkDeviceSize)0xffff : (4u << 20);
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, sz, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VkBuffer b = VK_NULL_HANDLE; VkDeviceMemory m = VK_NULL_HANDLE; void *p = NULL; VkResult r = vkCreateBuffer(V.dev, &bi, NULL, &b);
        if (!r) { VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, b, &req);
            VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
            r = vkAllocateMemory(V.dev, &ai, NULL, &m);
            if (!r) r = vkBindBufferMemory(V.dev, b, m, 0);
            if (!r) r = vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, &p); }
        if (r) {
            if (m) vkFreeMemory(V.dev, m, NULL); if (b) vkDestroyBuffer(V.dev, b, NULL);
            pthread_mutex_unlock(&mu);
            static int nr; if (nr++ < 8) VKLOG("[RED] NVN2 c1: no memory for a %llu-byte constants block (%d): shader %016llx reads zero constants until a later use makes it", (unsigned long long)sz, r, (unsigned long long)s->key);
            return 0; }
        blk = b; bp = p; bsize = sz; bused = 0; o = 0; g_c1_arena_bytes += sz;   // (the full block stays: shaders placed in it keep their constants)
        static int nb; if (nb++ < 4) VKLOG("NVN2 c1: constants block %d of %llu bytes", nb, (unsigned long long)sz);
    }
    memset(bp + o, 0, (size_t)n); memcpy(bp + o, s->c1, s->c1_len);
    s->c1_off = o; bused = o + n; __atomic_store_n(&s->c1_buf, blk, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&mu);
    __atomic_add_fetch(&g_c1_bufs, 1, __ATOMIC_RELAXED);
    return 1;
}
void vk_shader_c1(vkshader_t *s, VkBuffer *buf, VkDeviceSize *off, VkDeviceSize *len) {
    if (!s->c1_len) { *buf = vk_zero_buffer(); *off = 0; *len = 65536; return; }
    if (!s->c1_buf && dm_sdk_ounce()) {
        if (!c1_arena(s, (s->c1_len + 255) & ~255u)) { *buf = vk_zero_buffer(); *off = 0; *len = 65536; return; }
    } else if (!s->c1_buf) {
        s->c1_off = 0;
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, (s->c1_len + 255) & ~255u, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &s->c1_buf)); VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, s->c1_buf, &req);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
        VkDeviceMemory m; VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, s->c1_buf, m, 0)); __atomic_add_fetch(&g_c1_bufs, 1, __ATOMIC_RELAXED);
        void *p; VKCHK(vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, &p)); memset(p, 0, bi.size); memcpy(p, s->c1, s->c1_len);
    }
    *buf = s->c1_buf; *off = s->c1_off; *len = (s->c1_len + 255) & ~255u;
}
// pipeline layouts by the set layouts of their stages (an empty layout fills the sets of absent stages)
VkPipelineLayout vk_pipeline_layout(VkDescriptorSetLayout sets[5], int nsets) {
    static VkDescriptorSetLayout empty; static nvn_hmap cache = {.klen = 5 * sizeof(VkDescriptorSetLayout)}; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu);
    if (!empty) { VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; VKCHK(vkCreateDescriptorSetLayout(V.dev, &li, NULL, &empty)); }
    VkDescriptorSetLayout k[5]; for (int i = 0; i < 5; i++) { k[i] = i < nsets && sets[i] ? sets[i] : (i < nsets ? empty : VK_NULL_HANDLE); if (i < nsets) sets[i] = k[i]; }   // absent stages get the empty layout (the caller allocates their sets from it)
    VkPipelineLayout pl = nvn_hm_get(&cache, k);
    if (!pl) {
        static const VkPushConstantRange pr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 8};   // r56: graphics layouts carry the alpha test's push constants (spv_alpha_test); DM_VK_ALPHA=0: none
        static int at = -1; if (at < 0) { const char *e = getenv("DM_VK_ALPHA"); at = !(e && *e == '0'); }
        VkPipelineLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, (uint32_t)nsets, k, at && nsets >= 2, &pr};
        VKCHK(vkCreatePipelineLayout(V.dev, &ci, NULL, &pl)); nvn_hm_put(&cache, k, pl);
    }
    pthread_mutex_unlock(&mu);
    return pl;
}
// r133 (gfx_vk_exec.c DM_VK_X_POFS): the same layouts with one push constant range for both stages, 80 bytes (the alpha test's words 0-1, the pushed offsets' 4-19)
VkPipelineLayout vk_pipeline_layout_pofs(VkDescriptorSetLayout sets[5], int nsets) {
    static nvn_hmap cache = {.klen = 5 * sizeof(VkDescriptorSetLayout)}; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    (void)vk_pipeline_layout(sets, nsets);   // the empty layout in the sets of absent stages, as there
    pthread_mutex_lock(&mu);
    VkDescriptorSetLayout k[5]; for (int i = 0; i < 5; i++) k[i] = i < nsets ? sets[i] : VK_NULL_HANDLE;
    VkPipelineLayout pl = nvn_hm_get(&cache, k);
    if (!pl) {
        static const VkPushConstantRange pr = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 80};
        VkPipelineLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, (uint32_t)nsets, k, 1, &pr};
        VKCHK(vkCreatePipelineLayout(V.dev, &ci, NULL, &pl)); nvn_hm_put(&cache, k, pl);
    }
    pthread_mutex_unlock(&mu);
    return pl;
}
