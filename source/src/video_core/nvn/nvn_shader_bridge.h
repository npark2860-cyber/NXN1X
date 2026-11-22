// SPDX-License-Identifier: GPL-3.0-or-later
// NVN shaders through Eden's shader recompiler (NXN1X docs/lane_b/SHADER_EDEN_B.md).
// C interface: the native host (host/dm/windows/gfx_vk_shader.c, gfx_vk_exec.c) is C and calls only these.
// One NVN shader = one Eden environment, hashed exactly as the Maxwell path hashes the same code
// (GenericEnvironment::Analyze, CityHash64 from the SPH to the self branch), so both paths find the
// same entries in vulkan.bin.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// What a translation asks the replay about the draw it happens in. Called only during
// nvn_eden_translate, on the calling thread.
typedef struct nvn_eden_query {
    void* user;
    // a 32-bit word of constant buffer `bank` as this stage's shader reads it: 0 = the c0 driver
    // segment image, 2 = the c2 handle tables, 3 + r = uniform buffer slot r. 0 outside it.
    uint32_t (*cbuf)(void* user, int nvn_stage, uint32_t bank, uint32_t offset);
    // the texture registered under the handle's texture ID (handle & 0xfffff): its NVN target
    // (NVNtextureTarget) and the VkFormat the replay views it in. Returns 0 if none is registered.
    int (*texture)(void* user, uint32_t handle, int* nvn_target, int* vk_format);
} nvn_eden_query;

// The fixed-function state of one NVN draw, as the NVN driver would program it, in NVN values
// (nvn.h enums equal Maxwell's D3D-style register values; see nvn_pipeline_key.cpp). Formats are
// the VkFormats the replay uses (mapped back to Maxwell formats through Eden's own tables).
#ifndef NVN_XFB_LAYOUT_DEFINED   // s30b: also in host/dm/windows/gfx_vk_shader.c, whichever comes first
#define NVN_XFB_LAYOUT_DEFINED
// s30b: a vertex shader's transform feedback layout as its control section declares it (host gfx_vk_shader.c vk_xfb_note): per buffer the stride
// in bytes, the 4-byte words per vertex and Maxwell's stream-out attribute word at each position (0x20 + 4 n + component for generic output n)
typedef struct nvn_xfb_layout { uint32_t stride[4], count[4]; uint8_t words[4][128]; } nvn_xfb_layout;
#endif
typedef struct nvn_eden_draw_state {
    uint32_t primitive;          // NVNdrawPrimitive (= Maxwell PrimitiveTopology)
    uint32_t patch_size;         // control points per patch
    uint32_t polygon_mode;       // NVNpolygonMode (front)
    uint32_t cull_face;          // NVNface
    uint32_t front_face;         // NVNfrontFace
    uint32_t depth_clamp;        // 1 = depth clamp on (NVN polygon state)
    uint32_t polygon_offset;     // NVNpolygonOffsetEnable bits: 1 point, 2 line, 4 fill
    uint32_t rasterizer_discard;
    uint32_t primitive_restart;
    uint32_t depth_test, depth_write, depth_func;   // depth_func: NVNdepthFunc
    uint32_t depth_bounds_test; float depth_bounds[2];
    uint32_t stencil_test;
    uint32_t stencil_func[2], stencil_fail[2], stencil_zfail[2], stencil_zpass[2];   // front, back: NVN enums
    uint32_t alpha_func; float alpha_ref;              // NVNalphaFunc, ALWAYS when no alpha test
    uint32_t samples;            // 0 or 1 = single sample
    uint32_t alpha_to_coverage, alpha_to_one;
    uint32_t logic_op_enable, logic_op;                // NVNlogicOp
    uint32_t blend_enable;       // bit per colour target
    struct { uint32_t func_src, func_dst, func_src_a, func_dst_a, eq, eq_a; } blend[8];   // NVNblendFunc / NVNblendEquation
    uint32_t channel_mask[8];    // bits 0 r, 1 g, 2 b, 3 a
    uint32_t num_colors; int32_t color_format[8];      // VkFormat, 0 = no target
    int32_t depth_format;        // VkFormat, 0 = no depth target
    uint32_t lower_left_origin;  // nvnWindowSetOrigin / nvnDeviceSetWindowOriginMode: lower left
    uint32_t depth_mode_minus_one_to_one;              // NVN_DEPTH_MODE_NEAR_IS_MINUS_W
    uint32_t provoking_last;
    uint32_t conservative_raster, line_smooth, line_stipple; uint32_t line_stipple_factor, line_stipple_pattern;
    float point_size;
    uint32_t early_fragment;     // the fragment shader's early fragment tests (control +0x718)
    uint32_t viewport_swizzle[16];                     // Maxwell swizzle raw per viewport (identity 0x6420)
    uint32_t num_attribs; struct { uint32_t location, stream, offset; int32_t vk_format; } attrib[16];
    uint32_t num_streams; struct { uint32_t stride, divisor; } stream[16];   // divisor 0 = per vertex
    // B7: the Maxwell registers keep their last value when the NVN driver does not write them, and
    // FixedPipelineState reads them anyway: the depth format of the last depth target (0 = none yet)
    int32_t depth_format_kept;
    // s30b: transform feedback for this draw (the game's capture on and its vertex shader declares one): the layout FillNvnRegs gives Eden's key; NULL = off
    const struct nvn_xfb_layout *xfb;
    // vtxpull: 1 = ask for the pipeline whose vertex stage reads its inputs through the fetch table (FixedPipelineState::vertex_pulling, so a key
    // and disk-cache entry of its own); the pipeline says whether it does (nvn_eden_pipeline_info::vertex_pull)
    uint32_t vertex_pull;
} nvn_eden_draw_state;

typedef struct nvn_eden_shader nvn_eden_shader;   // one NVN shader: its code copy and Eden environment

// At nvnProgramSetShaders (the control section is valid only then): copies the control section and,
// if `code` is readable now, the code area. nvn_stage: 0 vertex 1 fragment 2 geometry 3 tess control
// 4 tess eval 5 compute (nvn.h). Returns NULL for a section it cannot read (logged).
// nvn_stage -1: found from the code area as the host's shader_split.py glslc_stage does (graphics
// header 0x12345678 + SPH shader type, compute header 0x12345679 or a headerless area with the
// control's thread group set).
nvn_eden_shader* nvn_eden_shader_new(const void* control, uint32_t control_size, const void* code,
                                     uint32_t code_size, int nvn_stage);
// Before first use when the code area was not mapped at SetShaders (Mario): copy it now.
int nvn_eden_shader_set_code(nvn_eden_shader* s, const void* code, uint32_t code_size);
void nvn_eden_shader_free(nvn_eden_shader* s);

// The NVN stage (as above), -1 while the code area is not copied or not recognised
int nvn_eden_shader_stage(const nvn_eden_shader* s);

// The Eden shader hash (the value the Maxwell path puts in GraphicsPipelineCacheKey::unique_hashes
// for the same code). 0 when the code could not be sized (no self branch within 1 MiB).
uint64_t nvn_eden_shader_hash(nvn_eden_shader* s);
// The CityHash64 of its c1 segment (diagnostic: same code with another c1, SHADER_EDEN_B.md §2.1)
uint64_t nvn_eden_shader_c1_hash(const nvn_eden_shader* s);

// ---- pipelines (SHADER_EDEN_B.md B4, B4b) ----
// An Eden GraphicsPipeline / ComputePipeline of the renderer's PipelineCache: the same object the
// Maxwell path would make for the same key, so the replay binds Eden's VkPipeline and fills
// Eden's descriptor set 0.
typedef struct nvn_eden_pipeline nvn_eden_pipeline;

// One descriptor of the pipeline's set 0, in Eden's order (per stage: uniform buffers, storage
// buffers, texture buffers, image buffers, textures, images). The replay reads the resource from
// the place the shader reads its handle or address and writes `count` entries of the payload
// from `first_entry` on.
enum {
    NVN_EB_UBO = 0,        // constant buffer `cbuf` (0 c0, 1 c1, 2 c2, 3 + r uniform slot r)
    NVN_EB_SSBO = 1,       // storage buffer: address and size at c[cbuf][offset] (c0: the stage's table)
    NVN_EB_TBUF = 2,       // texel buffer: handle at c[cbuf][offset]
    NVN_EB_IMG_TBUF = 3,   // storage texel buffer: handle at c[cbuf][offset]
    NVN_EB_TEX = 4,        // combined image sampler: handle at c[cbuf][offset], OR secondary
    NVN_EB_IMG = 5,        // storage image: handle at c[cbuf][offset]
};
typedef struct nvn_eden_bind {
    uint8_t nvn_stage, kind;
    uint8_t dim, arrayed;            // TEX / IMG: 0 1D, 1 2D, 2 3D, 3 cube (gfx_vk_exec.c's dim); rect is 2D
    uint8_t is_depth, is_ms, is_written, is_integer;
    uint32_t count;                  // array elements (handles 8 bytes << size_shift apart)
    uint32_t cbuf, offset, shift_left, size_shift;
    uint32_t has_secondary, secondary_cbuf, secondary_offset, secondary_shift_left;
    uint32_t first_entry;            // its first entry in the payload
    uint32_t indirect_offset;        // SSBO: ~0u = its address is at c[cbuf][offset]; else c[cbuf][offset] holds a table's address and the buffer's address is the
                                     // 64-bit word at table + indirect_offset (Eden StorageBufferDescriptor::indirect_offset: a pointer the shader loads from memory)
    const void* handle_info;        // Immutable pipeline stage metadata, while the pipeline lives.
    uint32_t handle_program;
    uint8_t handle_is_pointer;
    uint8_t img_format;              // IMG: the declared format (Shader::ImageFormat: 0 typeless, 5 R32_UINT ...), for its storage view (ootu32)
    uint8_t is_atomic;               // SSBO: the program runs an atomic on it (Eden StorageBufferDescriptor::is_atomic; cbuf 0xFF: an atomic stayed global). hybrid1
    const void* ssbo_desc;           // SSBO: its Shader::StorageBufferDescriptor (immutable while the pipeline lives), for nvn_eden_ssbo_reach. hybrid2
    uint32_t wg[3];                  // compute SSBO: the program's workgroup size (0: not a compute pipeline). hybrid2
} nvn_eden_bind;

// hybrid2: how far a compute dispatch's stores and atomics into storage buffer `bind` reach past its pointer (Eden StorageBufferDescriptor::write_extent,
// ResolveStorageWriteExtent), for a dispatch of grid[0..2] groups (indirect: grid unknown, NULL). 1 and *bytes when bounded, 0 when not known
int nvn_eden_ssbo_reach(const nvn_eden_bind* bind, const uint32_t* grid, uint64_t* bytes);

uint32_t nvn_eden_eval_handle(const nvn_eden_bind* bind, uint32_t element, const nvn_eden_query* query,
                              uint64_t alignment, uint32_t (*read_word)(uint64_t));

typedef struct nvn_eden_pipeline_info {
    void* pipeline;                  // VkPipeline; NULL until built (built == 0)
    void* layout;                    // VkPipelineLayout
    void* set_layout;                // VkDescriptorSetLayout (NULL: no descriptors)
    void* update_template;           // VkDescriptorUpdateTemplate (NULL with a descriptor buffer)
    int push_descriptor;             // 1: vkCmdPushDescriptorSetWithTemplateKHR, else a set + vkUpdateDescriptorSetWithTemplate
    int descriptor_buffer;           // 1: VK_EXT_descriptor_buffer (nvn_eden_write_dbuf, nvn_eden_cmd_dbuf)
    int built;
    uint32_t num_entries;            // payload entries
    uint32_t entry_size;             // bytes of one payload entry (DescriptorUpdateEntry)
    uint32_t num_binds;
    const nvn_eden_bind* binds;      // valid while the pipeline lives (pipelines are never dropped)
    uint64_t unique_hashes[6];       // the key's shader hashes (VertexA, VertexB, TCS, TES, GS, FS)
    // B4b: what the replay needs besides the binds
    int compute;                     // 1: a compute pipeline (bind point COMPUTE)
    uint32_t dyn;                    // NVN_ED_* bits: the dynamic state groups it was made with
    uint32_t vs_inputs;              // generic inputs (attribute locations) the vertex stage loads
    uint32_t frag_outputs;           // colour targets the fragment stage writes (bit per target)
    uint64_t dbuf_size;              // descriptor buffer bytes of its set 0 (descriptor_buffer)
    uint32_t num_viewports;          // viewports and scissors the pipeline has (all must be set)
    uint32_t layer_stages;           // pkm2: Eden stages (0 VertexB, 1 TessCtrl, 2 TessEval, 3 Geometry) that store gl_Layer
    // bak86: a compute pipeline's first-invocation stores, made by the host before each dispatch (vk_pipeline_cache.cpp FirstStorePass):
    // {constant buffer, its offset of the 64-bit address, byte offset from that address, value}
    uint32_t nfirst;
    uint32_t first[4][4];
    // vtxpull: 1 = the vertex stage reads its inputs through the fetch table: no vertex buffers; the table's address goes in with
    // nvn_eden_cmd_vertex_table before each draw (per stream {u64 address, u32 stride, u32 size}, 16 bytes, stream s at 16 s)
    int vertex_pull;
} nvn_eden_pipeline_info;

enum {   // nvn_eden_pipeline_info.dyn (vk_graphics_pipeline.cpp MakePipeline's dynamic states)
    NVN_ED_EDS1 = 1,                 // cull, front face, depth/stencil enables and ops, topology
    NVN_ED_STRIDE = 2,               // vertex binding strides (EDS1 without dynamic vertex input)
    NVN_ED_VERTEX_INPUT = 4,         // VK_EXT_vertex_input_dynamic_state
    NVN_ED_EDS2 = 8,                 // depth bias enable, primitive restart, rasterizer discard
    NVN_ED_EDS2_LOGIC_OP = 16,
    NVN_ED_EDS3_BLEND = 32,          // blend enables, equations, write masks
    NVN_ED_COLOR_WRITE_ENABLE = 64,  // VK_EXT_color_write_enable without EDS3 blending
    NVN_ED_EDS3_ENABLES = 128,       // depth clamp, logic op enable, line mode, conservative, stipple, a2c, a2one
};

// shaders: by NVN stage 0 vertex 1 fragment 2 geometry 3 tess control 4 tess eval, NULL = absent.
// The query answers the translation's questions from the current draw (only if not cached).
nvn_eden_pipeline* nvn_eden_graphics_pipeline(nvn_eden_shader* const shaders[5],
                                              const nvn_eden_draw_state* draw,
                                              const nvn_eden_query* query);
nvn_eden_pipeline* nvn_eden_compute_pipeline(nvn_eden_shader* shader, const nvn_eden_query* query);
// bak85 (diagnostic): the IR Eden makes of a compute shader, its storage buffer descriptors and hash, into the log folder's nvn2_ir_<host_key>.txt; 1 when written
int nvn_eden_dump_ir(nvn_eden_shader* shader, const nvn_eden_query* query, uint64_t host_key);
// NVN2 memory report (host gfx_vk_core.c): graphics and compute pipelines the bridge has recorded so far (they are kept for the run)
void nvn_eden_pipeline_counts(uint64_t out[2]);
// fills info; returns 0 for a NULL pipeline
int nvn_eden_pipeline_get(nvn_eden_pipeline* p, nvn_eden_pipeline_info* info, int wait);

// ---- recording a draw or dispatch of an Eden pipeline (SHADER_EDEN_B.md B4b) ----
// cmdbuf is the replay's VkCommandBuffer; the calls go through Eden's dispatch table (the device
// is Eden's, lane A), so extension entry points the host does not load are reachable.

// The replay's own values where its conventions differ from the draw state's (Vulkan values;
// -1 = from the draw state, as Eden's RasterizerVulkan::UpdateDynamicStates takes it).
int nvn_eden_resolution_x2(void);

typedef struct nvn_eden_dyn {
    uint32_t scaled_target;
    uint32_t scaled_textures[4];
    int32_t cull_mode;               // VkCullModeFlags
    int32_t front_face;              // VkFrontFace (the replay's viewport flips decide the winding)
    int32_t depth_test, depth_write, depth_compare;   // depth_compare: VkCompareOp
    int32_t stencil_test, depth_bounds_test;
    int32_t depth_bias_enable, primitive_restart, rasterizer_discard;
    uint32_t mask_unwritten;         // 1: colour targets the fragment stage does not write get mask 0
    float render_area[2];            // the render target's size (Eden: regs.surface_clip)
} nvn_eden_dyn;

// Push sampled-image scale bits, framebuffer coordinate scale and render area,
// then apply the extended dynamic state used by this pipeline.
// The base dynamic states (viewport, scissor, depth bias, blend constants, depth bounds, stencil
// masks and reference, line width) stay the replay's (gfx_vk_exec.c set_dynamic).
void nvn_eden_cmd_state(void* cmdbuf, nvn_eden_pipeline* p, const nvn_eden_draw_state* draw,
                        const nvn_eden_dyn* dyn);
// Vertex buffers, with their strides (from the draw state's streams) when the pipeline takes them
// dynamically. buffers: VkBuffer each.
void nvn_eden_cmd_vertex_buffers(void* cmdbuf, nvn_eden_pipeline* p, const nvn_eden_draw_state* draw,
                                 uint32_t first, uint32_t count, void* const* buffers,
                                 const uint64_t* offsets);
// vtxpull: the fetch table's device address for a vertex_pull pipeline (push constant at VERTEX_PULL_TABLE_OFFSET; the other push constants are
// nvn_eden_cmd_state's and stay as they are)
void nvn_eden_cmd_vertex_table(void* cmdbuf, nvn_eden_pipeline* p, uint64_t address);
// Set 0 from a payload of num_entries entries (entry_size bytes each, in bind order):
// push descriptor pipelines push it, the others update a set the replay allocated from set_layout
// (then bound by the replay with vkCmdBindDescriptorSets).
void nvn_eden_cmd_push_descriptors(void* cmdbuf, nvn_eden_pipeline* p, const void* payload);
void nvn_eden_update_set(nvn_eden_pipeline* p, void* set, const void* payload);
// Descriptor buffer pipelines: the payload's buffers and texel buffers are device addresses
// ({address, range, format} entries). write_dbuf writes dbuf_size bytes at host;
// cmd_dbuf binds the buffer at address (bind_buffer: vkCmdBindDescriptorBuffersEXT first) and sets
// set 0 at offset in it.
uint64_t nvn_eden_dbuf_alignment(void);
void nvn_eden_write_dbuf(nvn_eden_pipeline* p, const void* payload, void* host);
void nvn_eden_cmd_dbuf(void* cmdbuf, nvn_eden_pipeline* p, uint64_t address, uint64_t offset,
                       int bind_buffer);

// The c1 segment the shader was given at SetShaders (the replay's c1 uniform buffer)
const void* nvn_eden_shader_c1(const nvn_eden_shader* s, uint32_t* size);

// NVN2 compute: the per-shader c[0] layout of its control section (nvn_shader_env.h Control2): out[0] push constant bytes, out[1..6] uniform buffer, storage
// buffer, texture, separate texture, separate sampler and image slots, out[7] the size. 0 for any other shader
int nvn_eden_shader_nvn2_c0(const nvn_eden_shader* s, uint32_t out[8]);

#ifdef __cplusplus
}
#endif
