/* SPDX-License-Identifier: MIT
 *
 * nvrm G0b: shaders through NAK, vertex fetch, viewport/scissor, raster,
 * blend and depth-stencil state, constant buffers and draws.
 *
 * Hardware programming follows NVK at the pinned Mesa (26.2.5 ddaf85e3 + our patch
 * stack); every block names the NVK function it mirrors. G0b emits the whole
 * 3D state before every draw (no dirty tracking yet) and refuses anything it
 * does not implement with a named error instead of drawing something else.
 */
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "nvrm_priv.h"

#include "nak.h"
#include "nil.h"
#include "nv_push.h"
/* nv_push.h names no subchannel for PASCAL_B (NVK only reaches NVC197
 * methods through MMEs); it is the 3D class on subchannel 0. */
#define SUBC_NVC197 0
#include "nv_push_cl9097.h"
#include "nv_push_cla097.h"
#include "nv_push_clb197.h"
#include "nv_push_clc197.h"
#include "nv_push_clc397.h"
#include "nv_push_clc597.h"
#include "nv_push_clc797.h"
#include "clc397.h"
#include "clc597.h"
#include "clc797.h"

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"
#include "util/log.h"
#include "util/u_debug.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_upload_mgr.h"
#include "util/ralloc.h"
#include "util/bitset.h"
#include "nir/tgsi_to_nir.h"

/* NAK (Mesa >= 26) takes the driver's cbuf layout for fragment sample
 * locations/masks and the printf buffer from these driver-defined symbols
 * (NVK: nvk_shader.c, root table / HW root table). nvrm's layout is the one
 * its fragment shaders had through the zeroed nak_fs_key on Mesa 25.2: cbuf 0,
 * offset 0 for both tables (sample locations/masks in CB0 are divergence D12
 * of the shared-emit study, not implemented); printf is compiled out of
 * release NAK (NAK_CAN_PRINTF). */
const struct nak_constant_offset_info nak_const_offsets_base = { 0 };
const struct nak_constant_offset_info nak_const_offsets_turing_graphics = { 0 };

#define GF100_SHADER_HEADER_SIZE (20 * 4) /* nvk_shader.h:29 */
#define TU102_SHADER_HEADER_SIZE (32 * 4) /* nvk_shader.h:30 */

/* ---------------------------------------------------------------- shaders */

/* load_ubo(index, offset) -> ldc_nv, as NVK's lower_load_intrinsic
 * (nvk_shader.c:226-256) for the 1-component (bound slot) index form. GL UBO
 * block n (block 0 = default uniform block from nir_lower_uniforms_to_ubo)
 * is constant buffer slot n of the stage's bind group. */
static bool
nvrm_lower_load_ubo(nir_builder *b, nir_intrinsic_instr *load, void *data)
{
   if (load->intrinsic != nir_intrinsic_load_ubo)
      return false;
   b->cursor = nir_before_instr(&load->instr);
   nir_def *val = nir_ldc_nv(b, load->num_components, load->def.bit_size,
                             load->src[0].ssa, load->src[1].ssa,
                             .access = nir_intrinsic_access(load),
                             .align_mul = nir_intrinsic_align_mul(load),
                             .align_offset = nir_intrinsic_align_offset(load));
   nir_def_rewrite_uses(&load->def, val);
   nir_instr_remove(&load->instr);
   return true;
}

/* Constant buffer slots read by load_ubo (after nvrm_lower_buffers moved
 * slot-less blocks to the descriptor table): NVK keeps the same per-shader
 * map (nvk_cbuf_map) and binds only those slots (nvk_cmd_draw.c:3553-3557).
 * A non-constant block index may reach any hardware slot. */
static uint32_t
nvrm_cbuf_mask(nir_shader *nir)
{
   uint32_t mask = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_ubo)
               continue;
            if (nir_src_is_const(intr->src[0]) &&
                nir_src_as_uint(intr->src[0]) < NVRM_DESC_CBUF_SLOT)
               mask |= 1u << nir_src_as_uint(intr->src[0]);
            else
               mask |= BITFIELD_MASK(NVRM_DESC_CBUF_SLOT);
         }
      }
   }
   return mask;
}

/* Log multi-line text one mesa_log call per line, long lines in 1000-byte
 * pieces: an Android log entry is truncated past ~4 KiB, and
 * _mesa_log_multiline drops a last line without '\n' (util/log.c:461-472). */
static void
nvrm_log_text(enum mesa_log_level level, const char *text)
{
   const size_t piece = 1000;
   while (*text) {
      const size_t len = strcspn(text, "\n");
      size_t at = 0;
      do {
         const int n = (int)MIN2(piece, len - at);
         mesa_log(level, MESA_LOG_TAG, "%.*s", n, text + at);
         at += piece;
      } while (at < len);
      text += len;
      if (*text == '\n')
         text++;
   }
}

static void *
nvrm_create_shader_state(struct pipe_context *pctx,
                         const struct pipe_shader_state *state)
{
   nir_shader *nir;
   if (state->type == PIPE_SHADER_IR_NIR) {
      nir = state->ir.nir;
   } else if (state->type == PIPE_SHADER_IR_TGSI) {
      /* u_blitter / u_simple_shaders build TGSI; convert as asahi and
       * other NIR-only gallium drivers do. */
      nir = tgsi_to_nir(state->tokens, pctx->screen, false);
   } else {
      nvrm_error("shader IR type %d is neither NIR nor TGSI\n", state->type);
      abort();
   }
   return nvrm_compile_shader(pctx, nir);
}

/* Lowers and compiles (NAK) one shader of any stage; takes ownership of
 * `nir`. Returns a new nvrm_shader_bin (one reference), or NULL with the
 * reason logged and named in *failure. Uses no context state: the result is
 * a function of the NIR, the screen's NAK compiler and this library's build,
 * so it runs on a compile-queue worker (nvrm_shader_compile_job). */
static struct nvrm_shader_bin *
nvrm_nak_compile(struct nvrm_screen *screen, nir_shader *nir, bool debug,
                 const char **failure)
{
   const mesa_shader_stage stage = nir->info.stage;

   if (stage == MESA_SHADER_VERTEX) {
      /* st/mesa numbers vertex elements by driver_location (compacted);
       * NAK addresses generic attribute (location - GENERIC0). Make the
       * hardware attribute index equal the gallium vertex element index. */
      nir_foreach_shader_in_variable(var, nir)
         var->data.location = VERT_ATTRIB_GENERIC0 + var->data.driver_location;
   } else if (stage == MESA_SHADER_GEOMETRY) {
      /* GLSL allows max_vertices = 0 (dEQP geometry_shading.emit.*_emit_0_*);
       * a 0 max output vertex count in the program header is an Xid 13
       * "Shader Program Header 7 Error" (10-15 12:55, 9 cases on the 5090).
       * Vulkan forbids 0 (OutputVertices > 0), so NVK never sees it; nvc0
       * clamps the header field to [1, 1024] (nvc0_program.c:417). A GS
       * that may emit no vertex is unchanged by a limit of 1. */
      if (nir->info.gs.vertices_out == 0)
         nir->info.gs.vertices_out = 1;
   } else if (stage == MESA_SHADER_FRAGMENT) {
      /* gl_FragColor (FRAG_RESULT_COLOR) is not accepted by NAK
       * (nak_nir.c nak_fs_out_addr: Vulkan always has explicit locations);
       * GL broadcasts it to every bound draw buffer. */
      NIR_PASS(_, nir, nir_lower_fragcolor, NVRM_MAX_RTS);
   }

   if (debug) {
      mesa_logi("nvrm: st/mesa NIR for %s:", _mesa_shader_stage_to_string(stage));
      nir_log_shaderi(nir);
   }

   const unsigned tex_units = BITSET_LAST_BIT(nir->info.textures_used);
   if (tex_units > NVRM_MAX_TEX_UNITS) {
      nvrm_error("%s shader uses %u texture units (max %u)\n",
                 _mesa_shader_stage_to_string(stage), tex_units, NVRM_MAX_TEX_UNITS);
      *failure = "too many texture units";
      ralloc_free(nir);
      return NULL;
   }
   const unsigned image_units = BITSET_LAST_BIT(nir->info.images_used);
   if (image_units > NVRM_MAX_IMAGES) {
      nvrm_error("%s shader uses %u image units (max %u)\n",
                 _mesa_shader_stage_to_string(stage), image_units, NVRM_MAX_IMAGES);
      *failure = "too many image units";
      ralloc_free(nir);
      return NULL;
   }
   /* Mesa 26 NAK no longer sets lower_cs_local_index_to_id (nak/api.rs
    * nir_options; 25.2 had it) and panics on load_local_invocation_index in
    * compute shaders: the driver lowers it before NAK, as NVK 26.2 does
    * (nvk_shader.c:388-394, csv_options.lower_local_invocation_index for
    * compute). st/mesa already lowered the other compute system values
    * (st_glsl_to_nir.cpp:557-560). */
   if (stage == MESA_SHADER_COMPUTE) {
      const nir_lower_compute_system_values_options csv_options = {
         .lower_local_invocation_index = true,
      };
      NIR_PASS(_, nir, nir_lower_compute_system_values, &csv_options);
   }
   nvrm_lower_textures(nir);
   nvrm_lower_images(nir);
   /* SSBOs (and compute UBOs past block 0, num_workgroups) -> bounded
    * global access through the descriptor cbuf (nvrm_compute.c). */
   const bool uses_desc = nvrm_lower_buffers(nir);
   const uint32_t cbuf_mask = nvrm_cbuf_mask(nir);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, nvrm_lower_load_ubo,
            nir_metadata_control_flow, NULL);

   if (nir->constant_data_size > 0) {
      nvrm_error("%s shader has %u B of constant data: not implemented in G0b\n",
                 _mesa_shader_stage_to_string(stage), nir->constant_data_size);
      *failure = "shader constant data is not implemented";
      ralloc_free(nir);
      return NULL;
   }

   /* NAK lowers the shader in place, so keep the exact NIR it is handed for
    * the failure report below. */
   nir_shader *handed = nir_shader_clone(NULL, nir);
   /* robust2_modes 0 and nvrm_nak_fs_key: both are inputs of the program
    * cache key (nvrm_shader_cache_key). */
   struct nak_shader_bin *bin =
      nak_compile_shader(nir, debug, screen->nak, 0,
                         stage == MESA_SHADER_FRAGMENT ? &nvrm_nak_fs_key : NULL,
                         false /* has_task_shader: no mesh/task in GL */);
   ralloc_free(nir);
   if (!bin) {
      /* NAK panicked (it catches its own panics and returns NULL). Log the
       * panic message (thread-local: read on the thread that compiled) and
       * the NIR it was given; the first use of the shader stops the process
       * (nvrm_shader_wait): there is no fallback compiler. */
      const char *panic = nak_compile_shader_panic_message();
      nvrm_error("NAK failed to compile the %s shader; NAK panic: %s\n",
                 _mesa_shader_stage_to_string(stage), panic ? "" : "(none recorded)");
      if (panic)
         nvrm_log_text(MESA_LOG_ERROR, panic);
      mesa_loge("nvrm: NIR handed to NAK:");
      nir_log_shadere(handed);
      ralloc_free(handed);
      *failure = "NAK failed to compile it (panic logged above)";
      return NULL;
   }
   ralloc_free(handed);
   if (debug && bin->asm_str) {
      mesa_logi("nvrm: NAK %s asm:", _mesa_shader_stage_to_string(stage));
      nvrm_log_text(MESA_LOG_INFO, bin->asm_str);
   }

   struct nvrm_shader_bin *sbin = CALLOC_STRUCT(nvrm_shader_bin);
   void *code = malloc(MAX2(bin->code_size, 1));
   if (!sbin || !code) {
      nvrm_error("out of memory for a %u B %s shader binary\n", bin->code_size,
                 _mesa_shader_stage_to_string(stage));
      abort();
   }
   sbin->refcnt = 1;
   sbin->info = bin->info;
   sbin->code_size = bin->code_size;
   memcpy(code, bin->code, bin->code_size);
   sbin->code = code;
   sbin->tex_units = tex_units;
   sbin->image_units = image_units;
   sbin->cbuf_mask = cbuf_mask;
   sbin->uses_desc = uses_desc;
   nak_shader_bin_destroy(bin);
   return sbin;
}

/* Upload layout as nvk_shader_upload (nvk_shader.c:527-600): header then
 * code; Kepler..Volta put the header 0x30 B before a 0x80-aligned code
 * start, Turing+ have a 128 B header and 0x80-aligned start. Compute shaders
 * have no shader program header (nvk_shader.c:531-537): the QMD points at the
 * first instruction. */
static void
nvrm_shader_upload(struct nvrm_screen *screen, struct nvrm_shader *shader,
                   struct nvrm_shader_bin *bin)
{
   const mesa_shader_stage stage = shader->stage;
   const uint32_t hdr_size = stage == MESA_SHADER_COMPUTE ? 0 :
                             screen->info.cls_eng3d >= TURING_A ?
                             TU102_SHADER_HEADER_SIZE : GF100_SHADER_HEADER_SIZE;
   const uint32_t alignment = 0x80;
   uint32_t hdr_offset = 0;
   if (screen->info.cls_eng3d < TURING_A && hdr_size > 0)
      hdr_offset = alignment - hdr_size;
   const uint32_t code_offset = hdr_offset + hdr_size;
   assert(code_offset % alignment == 0);
   const uint32_t total = code_offset + bin->code_size;

   const uint64_t heap_size = align64(total, alignment);
   /* nvk_heap_alloc_locked: grow until the allocation fits. */
   simple_mtx_lock(&screen->shader_mutex);
   uint64_t heap_addr;
   while (!(heap_addr = util_vma_heap_alloc(&screen->shader_vma, heap_size, alignment))) {
      if (!nvrm_shader_heap_grow(screen)) {
         nvrm_error("shader heap exhausted (%" PRIu64 " B committed, %u memories, "
                    "%" PRIu64 " B requested)\n", screen->shader_committed,
                    screen->shader_mem_count, heap_size);
         abort();
      }
   }
   if (screen->bar_full) {
      /* CPU copy into the BAR1-mapped arena (reads shader_mem[], which a
       * concurrent grow appends to: under the mutex). */
      if (hdr_size)
         nvrm_shader_heap_write(screen, heap_addr + hdr_offset, bin->info.hdr, hdr_size);
      nvrm_shader_heap_write(screen, heap_addr + code_offset, bin->code, bin->code_size);
   } else {
      /* Small BAR: the copy-engine upload waits for the GPU. The range is
       * ours alone once allocated, so the copy runs outside the mutex and
       * another thread's allocation or a context flush (which takes the
       * mutex to recycle freed ranges) never waits for this GPU copy. */
      simple_mtx_unlock(&screen->shader_mutex);
      if (hdr_size)
         nvrm_shader_heap_write(screen, heap_addr + hdr_offset, bin->info.hdr, hdr_size);
      nvrm_shader_heap_write(screen, heap_addr + code_offset, bin->code, bin->code_size);
      simple_mtx_lock(&screen->shader_mutex);
   }
   /* Every context's next draw invalidates the instruction cache once
    * (nvrm_emit_invalidates; nvc0 after each upload, nvc0_program.c:944).
    * Bumped after the code is in place and before the shader's fence
    * signals, so the first draw using the shader (after nvrm_shader_wait)
    * sees the new generation. */
   p_atomic_inc(&screen->shader_gen);
   simple_mtx_unlock(&screen->shader_mutex);
   const uint64_t off = heap_addr - screen->shader_base;

   shader->bin = bin;
   shader->tex_units = bin->tex_units;
   shader->image_units = bin->image_units;
   shader->uses_desc = bin->uses_desc;
   shader->cbuf_mask = bin->cbuf_mask;
   shader->heap_addr = heap_addr;
   shader->heap_size = heap_size;
   shader->hdr_addr = screen->shader_base + off + hdr_offset;
   /* nvk_shader_upload (nvk_shader.c:590-596): pre-Volta program addresses
    * are relative to SET_PROGRAM_REGION (= the shader heap base). */
   if (screen->info.cls_eng3d < VOLTA_A) {
      shader->hdr_addr -= screen->shader_base;
      assert(shader->hdr_addr < UINT32_MAX);
   }
}

/* The compile job (a screen compile-queue worker, or inline under
 * NVRM_DEBUG_SHADERS): program cache lookup, NAK on a miss, upload. It reads
 * and writes only `shader` (not yet visible to any draw: every user waits on
 * shader->ready, nvrm_shader_wait) and screen state behind its own locks
 * (program cache, shader arena). A failure is logged here and named in
 * shader->failure; the first use aborts with it. */
static void
nvrm_shader_compile_job(void *job, UNUSED void *gdata, UNUSED int thread_index)
{
   struct nvrm_shader *shader = job;
   struct nvrm_screen *screen = shader->screen;
   nir_shader *nir = shader->nir;
   shader->nir = NULL;

   struct nvrm_shader_bin *bin;
   if (shader->debug) {
      bin = nvrm_nak_compile(screen, nir, true, &shader->failure);
   } else {
      cache_key key;
      nvrm_shader_cache_key(screen, nir, key);
      bin = nvrm_shader_cache_find(screen, key, shader->stage);
      if (bin) {
         ralloc_free(nir);
      } else {
         bin = nvrm_nak_compile(screen, nir, false, &shader->failure);
         if (bin)
            bin = nvrm_shader_cache_add(screen, key, bin);
      }
   }
   if (!bin)
      return;
   if (shader->stage == MESA_SHADER_COMPUTE &&
       bin->info.cs.smem_size > NVRM_MAX_SHARED_SIZE) {
      nvrm_error("compute shader needs %u B shared memory (max %u)\n",
                 bin->info.cs.smem_size, NVRM_MAX_SHARED_SIZE);
      shader->failure = "too much shared memory";
      nvrm_shader_bin_unref(bin);
      return;
   }
   nvrm_shader_upload(screen, shader, bin);
}

/* Creates the shader object and queues its compile (cache lookup, NAK,
 * upload) on the screen's compile queue; takes ownership of `nir`. The
 * object is usable at once: binding stores the pointer, and every reader of
 * the compiled result waits first (nvrm_shader_wait: draw, launch, compute
 * state info). radeonsi and freedreno defer the same way (radeonsi
 * si_schedule_initial_compile + si_shader_select's wait at draw; ir3
 * ir3_shader_state_create + ir3_get_shader's wait). Compute shaders come
 * from nvrm_compute.c. Shader objects are shareable across the contexts of
 * a share group (pipe caps shareable_shaders): any context may wait on or
 * delete them. */
struct nvrm_shader *
nvrm_compile_shader(struct pipe_context *pctx, nir_shader *nir)
{
   struct nvrm_screen *screen = nvrm_screen(pctx->screen);
   /* NVRM_DEBUG_SHADERS=1 on the host. On Android Mesa's os_get_option falls
    * back to the property system (util/os_misc.c os_get_android_option), so
    * `setprop debug.mesa.nvrm.debug.shaders 1` turns it on for every process
    * (SurfaceFlinger included) without an environment; read per shader, so
    * it applies to shaders created after the setprop. Output goes through
    * mesa_log line by line (logcat tag MESA) instead of stderr, which Android
    * services send to /dev/null. It bypasses the program cache: the dump is
    * produced by compiling (NVK refuses to cache a shader with asm,
    * nvk_shader.c nvk_shader_serialize), and it compiles on the calling
    * thread so the dump of each shader stays in one piece and in order
    * (radeonsi and ir3 also compile synchronously for shader dumps). */
   const bool debug = debug_get_bool_option("NVRM_DEBUG_SHADERS", false);

   /* st/mesa built the NIR with nvrm's st-facing options (no doubles
    * lowering, nvrm_screen_create); from here on it is NAK's NIR, as in
    * NVK (nvk_get_nir_options). Options are not serialized, so this does
    * not enter the cache key. */
   nir->options = nak_nir_options(screen->nak);

   struct nvrm_shader *shader = CALLOC_STRUCT(nvrm_shader);
   if (!shader) {
      nvrm_error("out of memory for a shader object\n");
      abort();
   }
   shader->screen = screen;
   shader->stage = nir->info.stage;
   shader->nir = nir;
   shader->debug = debug;
   util_queue_fence_init(&shader->ready); /* signalled */
   if (debug)
      nvrm_shader_compile_job(shader, NULL, -1);
   else
      util_queue_add_job(&screen->compile_queue, shader, &shader->ready,
                         nvrm_shader_compile_job, NULL, 0);
   return shader;
}

/* Every reader of a shader's compiled result calls this first. */
void
nvrm_shader_wait(struct nvrm_shader *shader)
{
   util_queue_fence_wait(&shader->ready);
   if (unlikely(shader->failure)) {
      nvrm_error("%s shader used, but its compile failed: %s\n",
                 _mesa_shader_stage_to_string(shader->stage), shader->failure);
      abort();
   }
}

static void
nvrm_delete_shader_state(struct pipe_context *pctx, void *s)
{
   nvrm_free_shader(nvrm_context(pctx), s);
}

void
nvrm_free_shader(struct nvrm_context *ctx, struct nvrm_shader *shader)
{
   /* A queued compile is dropped, a running one is waited for
    * (util_queue_drop_job; radeonsi si_delete_shader_selector, ir3
    * ir3_shader_state_delete). Afterwards the job either ran to its end or
    * never started, and nothing else references the object. */
   util_queue_drop_job(&shader->screen->compile_queue, &shader->ready);
   if (shader->nir)
      ralloc_free(shader->nir); /* dropped before it ran */
   if (shader->heap_size) {
      util_dynarray_append_typed(&ctx->shader_frees, uint64_t, shader->heap_addr);
      util_dynarray_append_typed(&ctx->shader_frees, uint64_t, shader->heap_size);
   }
   if (shader->bin)
      nvrm_shader_bin_unref(shader->bin);
   util_queue_fence_destroy(&shader->ready);
   FREE(shader);
}

static void
nvrm_bind_vs_state(struct pipe_context *pctx, void *s)
{
   nvrm_context(pctx)->vs = s;
   nvrm_context(pctx)->dirty |= NVRM_DIRTY_SHADERS | NVRM_DIRTY_TEXTURES;
}

static void
nvrm_bind_gs_state(struct pipe_context *pctx, void *s)
{
   nvrm_context(pctx)->gs = s;
   nvrm_context(pctx)->dirty |= NVRM_DIRTY_SHADERS | NVRM_DIRTY_TEXTURES;
}

static void
nvrm_bind_fs_state(struct pipe_context *pctx, void *s)
{
   nvrm_context(pctx)->fs = s;
   nvrm_context(pctx)->dirty |= NVRM_DIRTY_SHADERS | NVRM_DIRTY_TEXTURES;
}

/* ---------------------------------------------------------------- CSOs */

#define NVRM_CSO(name, type, field, bits)                                         \
   static void *nvrm_create_##name(struct pipe_context *p, const type *s)     \
   {                                                                          \
      type *copy = MALLOC(sizeof(type));                                      \
      if (copy) memcpy(copy, s, sizeof(*copy));                               \
      return copy;                                                            \
   }                                                                          \
   static void nvrm_bind_##name(struct pipe_context *p, void *s)              \
   {                                                                          \
      nvrm_context(p)->field = s;                                             \
      nvrm_context(p)->dirty |= (bits);                                       \
   }                                                                          \
   static void nvrm_delete_##name(struct pipe_context *p, void *s)            \
   {                                                                          \
      FREE(s);                                                                \
   }
NVRM_CSO(blend, struct pipe_blend_state, blend, NVRM_DIRTY_DSA_BLEND)
NVRM_CSO(rasterizer, struct pipe_rasterizer_state, rast,
         NVRM_DIRTY_RAST | NVRM_DIRTY_VIEWPORT)
NVRM_CSO(dsa, struct pipe_depth_stencil_alpha_state, dsa, NVRM_DIRTY_DSA_BLEND)

static void *
nvrm_create_vertex_elements(struct pipe_context *p, unsigned count,
                            const struct pipe_vertex_element *elements)
{
   struct nvrm_velems *ve = CALLOC_STRUCT(nvrm_velems);
   if (!ve)
      return NULL;
   ve->count = MIN2(count, PIPE_MAX_ATTRIBS);
   memcpy(ve->el, elements, ve->count * sizeof(*elements));
   return ve;
}

static void
nvrm_bind_vertex_elements(struct pipe_context *p, void *s)
{
   nvrm_context(p)->velems = s;
   nvrm_context(p)->dirty |= NVRM_DIRTY_VERTEX;
}

static void
nvrm_delete_vertex_elements(struct pipe_context *p, void *s)
{
   FREE(s);
}

/* ---------------------------------------------------------------- set_* */

static void
nvrm_set_viewport_states(struct pipe_context *p, unsigned start, unsigned n,
                         const struct pipe_viewport_state *v)
{
   if (start == 0 && n > 0)
      nvrm_context(p)->viewport = v[0];
   nvrm_context(p)->dirty |= NVRM_DIRTY_VIEWPORT;
}

static void
nvrm_set_scissor_states(struct pipe_context *p, unsigned start, unsigned n,
                        const struct pipe_scissor_state *s)
{
   if (start == 0 && n > 0)
      nvrm_context(p)->scissor = s[0];
   nvrm_context(p)->dirty |= NVRM_DIRTY_VIEWPORT;
}

static void
nvrm_set_blend_color(struct pipe_context *p, const struct pipe_blend_color *c)
{
   nvrm_context(p)->blend_color = *c;
   nvrm_context(p)->dirty |= NVRM_DIRTY_DSA_BLEND;
}

static void
nvrm_set_stencil_ref(struct pipe_context *p, const struct pipe_stencil_ref r)
{
   nvrm_context(p)->stencil_ref = r;
   nvrm_context(p)->dirty |= NVRM_DIRTY_DSA_BLEND;
}

static void
nvrm_set_sample_mask(struct pipe_context *p, unsigned m)
{
   nvrm_context(p)->sample_mask = m;
   nvrm_context(p)->dirty |= NVRM_DIRTY_RAST;
}

static void
nvrm_set_constant_buffer(struct pipe_context *pctx, mesa_shader_stage shader,
                         uint index, const struct pipe_constant_buffer *cb)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   if (shader != MESA_SHADER_VERTEX && shader != MESA_SHADER_GEOMETRY &&
       shader != MESA_SHADER_FRAGMENT && shader != MESA_SHADER_COMPUTE) {
      return;
   }
   if (index >= NVRM_MAX_CBUFS) {
      nvrm_error("constant buffer slot %u is beyond G0b's %u\n", index, NVRM_MAX_CBUFS);
      abort();
   }
   if (index >= NVRM_MAX_UBOS) {
      nvrm_error("constant buffer %u beyond %u\n", index, NVRM_MAX_UBOS);
      abort();
   }
   struct pipe_constant_buffer *slot = &ctx->cb[shader][index];
   const int gfx = nvrm_gfx_stage(shader);
   const bool was_user = ctx->cb_user[shader] && index == 0;
   pipe_resource_reference(&slot->buffer, NULL);
   memset(slot, 0, sizeof(*slot));
   if (index == 0)
      ctx->cb_user[shader] = false;
   if (cb && cb->user_buffer && gfx >= 0) {
      if (index != 0 || cb->buffer_size > NVRM_DRV_CB_UNIFORM_B) {
         nvrm_error("user constants of %u B in slot %u (driver cbuf: slot 0, %u B)\n",
                    cb->buffer_size, index, NVRM_DRV_CB_UNIFORM_B);
         abort();
      }
      /* Default uniform block: loaded now, in stream order, into this
       * stage's driver cbuf area (only the dwords that differ from what is
       * there); the binding itself only changes with the size. nvc0's user
       * constants (nvc0_state_validate.c:532-549); the stock GL driver
       * likewise loads only the written constant-buffer range. */
      ctx->cb_user[shader] = true;
      slot->buffer_size = cb->buffer_size;
      /* Kept as nvc0 keeps it (nvc0_state.c nvc0_set_constant_buffer
       * u.data): u_blitter saves this slot and restores it through
       * set_constant_buffer (u_blitter.h:538-545, u_blitter.c:758-764),
       * which then reloads whatever differs from its own constants. */
      slot->user_buffer = cb->user_buffer;
      if (!was_user)
         ctx->cb_dirty[shader] |= 1u;
      const uint32_t size_B = cb->buffer_size & ~3u;
      if (!size_B)
         return;
      struct nv_push push = nvrm_push_begin(ctx, nvrm_drv_cb_load_dw(size_B));
      nvrm_drv_cb_load(ctx, &push, gfx, 0, ctx->drv_cb_uniforms[gfx],
                       &ctx->drv_cb_uniforms_valid_B[gfx], cb->user_buffer, size_B);
      nvrm_push_end(ctx, &push);
      return;
   }
   /* A buffer binding: the descriptor table (UBO blocks without a slot)
    * and the hardware slot change. */
   ctx->dirty |= NVRM_DIRTY_CBUFS;
   ctx->cb_dirty[shader] |= 1u << index;
   if (!cb)
      return;
   if (cb->user_buffer) {
      /* Compute: QMD constant buffers point at memory (nvrm_compute.c);
       * copied into the const uploader, which never rewrites live memory. */
      u_upload_data_ref(pctx->const_uploader, 0, cb->buffer_size,
                    pctx->screen->caps.constant_buffer_offset_alignment,
                    cb->user_buffer, &slot->buffer_offset, &slot->buffer);
      slot->buffer_size = cb->buffer_size;
      return;
   }
   /* Mesa 26: the caller keeps its reference (no ownership transfer). */
   pipe_resource_reference(&slot->buffer, cb->buffer);
   slot->buffer_offset = cb->buffer_offset;
   slot->buffer_size = cb->buffer_size;
}

static void
nvrm_set_vertex_buffers(struct pipe_context *pctx, unsigned count,
                        const struct pipe_vertex_buffer *vbs)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   if (count > PIPE_MAX_ATTRIBS) {
      nvrm_error("%u vertex buffers exceed %u\n", count, PIPE_MAX_ATTRIBS);
      abort();
   }
   /* Mesa 26: the caller keeps its references; the driver takes its own
    * (util_set_vertex_buffers_mask, u_helpers.c:47-75, as nvc0 does). */
   for (unsigned i = 0; i < count; i++) {
      if (vbs[i].is_user_buffer) {
         nvrm_error("user vertex buffers are not advertised (caps)\n");
         abort();
      }
      pipe_vertex_buffer_reference(&ctx->vb[i], &vbs[i]);
   }
   for (unsigned i = count; i < ctx->num_vb; i++)
      pipe_vertex_buffer_unreference(&ctx->vb[i]);
   ctx->num_vb = count;
   ctx->dirty |= NVRM_DIRTY_VERTEX;
}

/* ---------------------------------------------------------------- formats */

/* pipe_format -> SET_VERTEX_ATTRIBUTE_A {component_bit_widths, numerical_type}
 * for array formats with equal channel sizes, as NVK's nvk_vf_formats table
 * (nvk_format.c:51-170) lists them; anything else is refused. */
bool
nvrm_get_va_format(enum pipe_format format, struct nvrm_va_format *out)
{
   /* Packed 2_10_10_10 (GLES 3.0 vertex types): NVK's nvk_vf_formats
    * A2B10G10R10 rows (nvk_format.c:107-117); Vulkan A2B10G10R10_PACK32 is
    * gallium R10G10B10A2, A2R10G10B10 (swap_r_and_b) is B10G10R10A2. */
#define T(x) NV9097_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_##x
   static const struct { enum pipe_format f; uint32_t type; bool swap; } packed[] = {
      { PIPE_FORMAT_R10G10B10A2_UNORM, T(UNORM), false },
      { PIPE_FORMAT_R10G10B10A2_SNORM, T(SNORM), false },
      { PIPE_FORMAT_R10G10B10A2_USCALED, T(USCALED), false },
      { PIPE_FORMAT_R10G10B10A2_SSCALED, T(SSCALED), false },
      { PIPE_FORMAT_R10G10B10A2_UINT, T(UINT), false },
      { PIPE_FORMAT_R10G10B10A2_SINT, T(SINT), false },
      { PIPE_FORMAT_B10G10R10A2_UNORM, T(UNORM), true },
      { PIPE_FORMAT_B10G10R10A2_SNORM, T(SNORM), true },
      { PIPE_FORMAT_B10G10R10A2_USCALED, T(USCALED), true },
      { PIPE_FORMAT_B10G10R10A2_SSCALED, T(SSCALED), true },
      { PIPE_FORMAT_B10G10R10A2_UINT, T(UINT), true },
      { PIPE_FORMAT_B10G10R10A2_SINT, T(SINT), true },
   };
#undef T
   for (unsigned i = 0; i < ARRAY_SIZE(packed); i++) {
      if (packed[i].f == format) {
         out->widths = NV9097_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_A2B10G10R10;
         out->type = packed[i].type;
         out->swap_rb = packed[i].swap;
         return true;
      }
   }
   out->swap_rb = false;
   const struct util_format_description *d = util_format_description(format);
   if (d->layout != UTIL_FORMAT_LAYOUT_PLAIN || !d->is_array)
      return false;
   const struct util_format_channel_description *c = &d->channel[0];
   const unsigned n = d->nr_channels, size = c->size;
   for (unsigned i = 0; i < 4; i++) {
      if (d->swizzle[i] <= PIPE_SWIZZLE_W && d->swizzle[i] != i)
         return false; /* BGRA etc.: swap_r_and_b not wired in G0b */
   }
#define W(x) NV9097_SET_VERTEX_ATTRIBUTE_A_COMPONENT_BIT_WIDTHS_##x
   static const uint32_t w32[4] = { W(R32), W(R32_G32), W(R32_G32_B32), W(R32_G32_B32_A32) };
   static const uint32_t w16[4] = { W(R16), W(R16_G16), W(R16_G16_B16), W(R16_G16_B16_A16) };
   static const uint32_t w8[4] = { W(R8), W(R8_G8), W(R8_G8_B8), W(R8_G8_B8_A8) };
#undef W
   switch (size) {
   case 32: out->widths = w32[n - 1]; break;
   case 16: out->widths = w16[n - 1]; break;
   case 8:  out->widths = w8[n - 1]; break;
   default: return false;
   }
#define T(x) NV9097_SET_VERTEX_ATTRIBUTE_A_NUMERICAL_TYPE_NUM_##x
   switch (c->type) {
   case UTIL_FORMAT_TYPE_FLOAT: out->type = T(FLOAT); break;
   case UTIL_FORMAT_TYPE_UNSIGNED:
      out->type = c->normalized ? T(UNORM) : c->pure_integer ? T(UINT) : T(USCALED);
      break;
   case UTIL_FORMAT_TYPE_SIGNED:
      out->type = c->normalized ? T(SNORM) : c->pure_integer ? T(SINT) : T(SSCALED);
      break;
   default: return false;
   }
#undef T
   return true;
}

/* PIPE_BLENDFACTOR_* -> the OGL coefficient encoding (GL enum | 0x4000),
 * NV9097_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_* in cl9097.h. */
static uint32_t
nvrm_blend_factor(unsigned f)
{
   switch (f) {
   case PIPE_BLENDFACTOR_ONE:                return 0x4001;
   case PIPE_BLENDFACTOR_SRC_COLOR:          return 0x4300;
   case PIPE_BLENDFACTOR_SRC_ALPHA:          return 0x4302;
   case PIPE_BLENDFACTOR_DST_ALPHA:          return 0x4304;
   case PIPE_BLENDFACTOR_DST_COLOR:          return 0x4306;
   case PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE: return 0x4308;
   case PIPE_BLENDFACTOR_CONST_COLOR:        return 0xc001;
   case PIPE_BLENDFACTOR_CONST_ALPHA:        return 0xc003;
   case PIPE_BLENDFACTOR_ZERO:               return 0x4000;
   case PIPE_BLENDFACTOR_INV_SRC_COLOR:      return 0x4301;
   case PIPE_BLENDFACTOR_INV_SRC_ALPHA:      return 0x4303;
   case PIPE_BLENDFACTOR_INV_DST_ALPHA:      return 0x4305;
   case PIPE_BLENDFACTOR_INV_DST_COLOR:      return 0x4307;
   case PIPE_BLENDFACTOR_INV_CONST_COLOR:    return 0xc002;
   case PIPE_BLENDFACTOR_INV_CONST_ALPHA:    return 0xc004;
   default:
      nvrm_error("blend factor %u is not implemented in G0b\n", f);
      abort();
   }
}

static uint32_t
nvrm_blend_op(unsigned op)
{
   switch (op) {
   case PIPE_BLEND_ADD:              return 0x8006;
   case PIPE_BLEND_SUBTRACT:         return 0x800a;
   case PIPE_BLEND_REVERSE_SUBTRACT: return 0x800b;
   case PIPE_BLEND_MIN:              return 0x8007;
   case PIPE_BLEND_MAX:              return 0x8008;
   default:
      nvrm_error("blend op %u is not implemented in G0b\n", op);
      abort();
   }
}

/* PIPE_STENCIL_OP_* -> the OGL stencil op encoding (GL enums). */
static uint32_t
nvrm_stencil_op(unsigned op)
{
   switch (op) {
   case PIPE_STENCIL_OP_KEEP:      return NV9097_SET_STENCIL_OP_FAIL_V_OGL_KEEP;
   case PIPE_STENCIL_OP_ZERO:      return NV9097_SET_STENCIL_OP_FAIL_V_OGL_ZERO;
   case PIPE_STENCIL_OP_REPLACE:   return NV9097_SET_STENCIL_OP_FAIL_V_OGL_REPLACE;
   case PIPE_STENCIL_OP_INCR:      return NV9097_SET_STENCIL_OP_FAIL_V_OGL_INCRSAT;
   case PIPE_STENCIL_OP_DECR:      return NV9097_SET_STENCIL_OP_FAIL_V_OGL_DECRSAT;
   case PIPE_STENCIL_OP_INCR_WRAP: return NV9097_SET_STENCIL_OP_FAIL_V_OGL_INCR;
   case PIPE_STENCIL_OP_DECR_WRAP: return NV9097_SET_STENCIL_OP_FAIL_V_OGL_DECR;
   case PIPE_STENCIL_OP_INVERT:    return NV9097_SET_STENCIL_OP_FAIL_V_OGL_INVERT;
   default:
      nvrm_error("stencil op %u unknown\n", op);
      abort();
   }
}

/* ---------------------------------------------------------------- emit */

/* Colour target (nvrm_emit_color_target) plus the single-sample layout. A
 * framebuffer change also orders earlier rendering before later texturing
 * from it (render to texture): NVK's colour-attachment-write -> shader-read
 * barrier is a 3D WAIT_FOR_IDLE plus a texture data cache invalidate
 * (emitted with the texture state). */
static void
nvrm_emit_framebuffer(struct nvrm_context *ctx, struct nv_push *p)
{
   P_IMMD(p, NV9097, WAIT_FOR_IDLE, 0);
   nvrm_emit_color_target(ctx, p, &ctx->fb);
   nvrm_emit_zs_target(ctx, p, &ctx->fb);
   nvrm_emit_sample_layout(ctx, p, &ctx->fb);
}

/* The bound targets' sample layout. Emitted with the targets by draws AND
 * clears: NVK sets it at vkCmdBeginRendering, before the render pass's
 * clears (nvk_cmd_draw.c:1399-1406). CLEAR_SURFACE takes its rect in
 * pixels and scales it by SET_ANTI_ALIAS, so a clear of a multisampled
 * target under a stale 1X1 mode cleared only the top-left pixel quarter of
 * a 2x2 layout (dEQP-GLES3 multisample.fbo_*.stencil: background alpha 0
 * outside it). */
void
nvrm_emit_sample_layout(struct nvrm_context *ctx, struct nv_push *p,
                        const struct pipe_framebuffer_state *fb)
{
   /* nvk_cmd_set_sample_layout (nvk_cmd_draw.c:841-923) for the bound
    * attachments' layout, with the NVK_MME_SET_ANTI_ALIAS result for no
    * sample shading (GLES 3.0 has none): one pass, centroid per fragment.
    * NVK's per-pass sample-mask scratch only feeds sample shading. */
   enum nil_sample_layout layout = NIL_SAMPLE_LAYOUT_1X1;
   for (unsigned i = 0; i < fb->nr_cbufs; i++) {
      if (fb->cbufs[i].texture && nvrm_resource(fb->cbufs[i].texture)->tiled) {
         layout = nvrm_resource(fb->cbufs[i].texture)->image.sample_layout;
         break;
      }
   }
   if (layout == NIL_SAMPLE_LAYOUT_1X1 && fb->zsbuf.texture)
      layout = nvrm_resource(fb->zsbuf.texture)->image.sample_layout;
   /* No attachments (GLES 3.1 framebuffer_no_attachment): the layout of
    * the default sample count, as NVK with render->samples == 0
    * (nvk_cmd_draw.c:2952-2958). */
   bool any_target = fb->zsbuf.texture != NULL;
   for (unsigned i = 0; i < fb->nr_cbufs; i++)
      any_target |= fb->cbufs[i].texture != NULL;
   if (!any_target && fb->samples > 1)
      layout = nil_choose_sample_layout(fb->samples);
   static const uint16_t modes[] = {
      [NIL_SAMPLE_LAYOUT_1X1] = NV9097_SET_ANTI_ALIAS_SAMPLES_MODE_1X1,
      [NIL_SAMPLE_LAYOUT_2X1] = NV9097_SET_ANTI_ALIAS_SAMPLES_MODE_2X1,
      [NIL_SAMPLE_LAYOUT_2X1_D3D] = NV9097_SET_ANTI_ALIAS_SAMPLES_MODE_2X1_D3D,
      [NIL_SAMPLE_LAYOUT_2X2] = NV9097_SET_ANTI_ALIAS_SAMPLES_MODE_2X2,
      [NIL_SAMPLE_LAYOUT_4X2] = NV9097_SET_ANTI_ALIAS_SAMPLES_MODE_4X2,
      [NIL_SAMPLE_LAYOUT_4X2_D3D] = NV9097_SET_ANTI_ALIAS_SAMPLES_MODE_4X2_D3D,
      [NIL_SAMPLE_LAYOUT_4X4] = NV9097_SET_ANTI_ALIAS_SAMPLES_MODE_4X4,
   };
   if (layout >= ARRAY_SIZE(modes) ||
       (layout != NIL_SAMPLE_LAYOUT_1X1 && modes[layout] == 0)) {
      nvrm_error("sample layout %u has no SET_ANTI_ALIAS mode\n", layout);
      abort();
   }
   P_IMMD(p, NV9097, SET_ANTI_ALIAS, modes[layout]);
   P_IMMD(p, NV9097, SET_HYBRID_ANTI_ALIAS_CONTROL, {
      .passes = 1,
      .centroid = CENTROID_PER_FRAGMENT,
   });

   /* nvk_flush_ms_state (nvk_cmd_draw.c:2976-3022): the sample positions
    * are programmable on MAXWELL_B+ and NVK always programs them (the
    * standard locations when the app gives none); nothing sets them at
    * channel init, so without this every sample sat at the method's
    * reset position (dEQP-GLES3 multisample.fbo_*.num_samples_*,
    * rasterization.fbo.rbo_multisample_*). Positions are
    * vk_standard_sample_locations.c's tables in 1/16 pixel
    * (vk_to_nak_sample_location: ufixed 0.4, clamped), laid out as NVK's
    * 2x2-pixel table: entry n = sample n % samples of pixel n / samples. */
   static const uint8_t std_pos[4][8][2] = {
      { { 8, 8 } },
      { { 12, 12 }, { 4, 4 } },
      { { 6, 2 }, { 14, 6 }, { 2, 10 }, { 10, 14 } },
      { { 9, 5 }, { 7, 11 }, { 13, 9 }, { 5, 3 },
        { 3, 13 }, { 1, 7 }, { 11, 15 }, { 15, 1 } },
   };
   const unsigned samples = nil_sample_layout_samples(layout);
   const unsigned log2_samples = util_logbase2(samples);
   if (log2_samples >= ARRAY_SIZE(std_pos)) {
      nvrm_error("%u samples have no standard sample positions\n", samples);
      abort();
   }
   if (nvrm_screen(ctx->base.screen)->info.cls_eng3d >= MAXWELL_B) {
      uint8_t loc[16][2];
      for (unsigned n = 0; n < 16; n++) {
         loc[n][0] = std_pos[log2_samples][n % samples][0];
         loc[n][1] = std_pos[log2_samples][n % samples][1];
      }
      P_MTHD(p, NVB197, SET_ANTI_ALIAS_SAMPLE_POSITIONS(0));
      for (unsigned i = 0; i < 4; i++) {
         P_NVB197_SET_ANTI_ALIAS_SAMPLE_POSITIONS(p, i, {
            .x0 = loc[i * 4 + 0][0], .y0 = loc[i * 4 + 0][1],
            .x1 = loc[i * 4 + 1][0], .y1 = loc[i * 4 + 1][1],
            .x2 = loc[i * 4 + 2][0], .y2 = loc[i * 4 + 2][1],
            .x3 = loc[i * 4 + 3][0], .y3 = loc[i * 4 + 3][1],
         });
      }
   }
}

/* nvk_flush_vp_state (nvk_cmd_draw.c:1975-2090) from gallium's
 * scale/translate form. */
static void
nvrm_emit_viewport(struct nvrm_context *ctx, struct nv_push *p)
{
   const struct nvrm_screen *screen = nvrm_screen(ctx->base.screen);
   const struct pipe_viewport_state *vp = &ctx->viewport;
   const struct pipe_rasterizer_state *rs = ctx->rast;

   P_MTHD(p, NV9097, SET_VIEWPORT_SCALE_X(0));
   P_NV9097_SET_VIEWPORT_SCALE_X(p, 0, fui(vp->scale[0]));
   P_NV9097_SET_VIEWPORT_SCALE_Y(p, 0, fui(vp->scale[1]));
   P_NV9097_SET_VIEWPORT_SCALE_Z(p, 0, fui(vp->scale[2]));
   P_NV9097_SET_VIEWPORT_OFFSET_X(p, 0, fui(vp->translate[0]));
   P_NV9097_SET_VIEWPORT_OFFSET_Y(p, 0, fui(vp->translate[1]));
   P_NV9097_SET_VIEWPORT_OFFSET_Z(p, 0, fui(vp->translate[2]));

   float xmin = vp->translate[0] - fabsf(vp->scale[0]);
   float xmax = vp->translate[0] + fabsf(vp->scale[0]);
   float ymin = vp->translate[1] - fabsf(vp->scale[1]);
   float ymax = vp->translate[1] + fabsf(vp->scale[1]);
   const float zlo = rs->clip_halfz ? vp->translate[2] : vp->translate[2] - vp->scale[2];
   const float zhi = vp->translate[2] + vp->scale[2];
   float zmin = MIN2(zlo, zhi), zmax = MAX2(zlo, zhi);
   const float max_dim = (float)0xffff;
   xmin = CLAMP(xmin, 0, max_dim);
   xmax = CLAMP(xmax, 0, max_dim);
   ymin = CLAMP(ymin, 0, max_dim);
   ymax = CLAMP(ymax, 0, max_dim);
   zmin = CLAMP(zmin, 0, 1);
   zmax = CLAMP(zmax, 0, 1);

   P_MTHD(p, NV9097, SET_VIEWPORT_CLIP_HORIZONTAL(0));
   P_NV9097_SET_VIEWPORT_CLIP_HORIZONTAL(p, 0, {
      .x0 = xmin, .width = xmax - xmin,
   });
   P_NV9097_SET_VIEWPORT_CLIP_VERTICAL(p, 0, {
      .y0 = ymin, .height = ymax - ymin,
   });
   /* Pre-Volta has no SET_VIEWPORT_CLIP_CONTROL.z_clip_range: NVK emulates
    * it by writing VIEWPORT_CLIP_MIN/MAX_Z = viewport range when Z clamping
    * is on and -inf/+inf when it is off (NVK_MME_SET_VIEWPORT_MIN_MAX_Z +
    * NVK_MME_SET_Z_CLAMP, nvk_cmd_draw.c:2026-2034, 2170-2210). nvrm knows
    * the rasterizer here (rasterizer binds re-dirty the viewport), so it
    * writes the macro's result directly. */
   if (screen->info.cls_eng3d < VOLTA_A &&
       !(!rs->depth_clip_near || !rs->depth_clip_far)) {
      zmin = -INFINITY;
      zmax = INFINITY;
   }
   P_NV9097_SET_VIEWPORT_CLIP_MIN_Z(p, 0, fui(zmin));
   P_NV9097_SET_VIEWPORT_CLIP_MAX_Z(p, 0, fui(zmax));
   if (screen->info.cls_eng3d >= MAXWELL_B) { /* nvk_cmd_draw.c:2036-2043 */
      P_IMMD(p, NVB197, SET_VIEWPORT_COORDINATE_SWIZZLE(0), {
         .x = X_POS_X, .y = Y_POS_Y, .z = Z_POS_Z, .w = W_POS_W,
      });
   }
   P_IMMD(p, NV9097, SET_VIEWPORT_Z_CLIP,
          rs->clip_halfz ? RANGE_ZERO_TO_POSITIVE_W : RANGE_NEGATIVE_W_TO_POSITIVE_W);

   if (rs->scissor) {
      const struct pipe_scissor_state *s = &ctx->scissor;
      P_MTHD(p, NV9097, SET_SCISSOR_ENABLE(0));
      P_NV9097_SET_SCISSOR_ENABLE(p, 0, V_TRUE);
      P_NV9097_SET_SCISSOR_HORIZONTAL(p, 0, { .xmin = s->minx, .xmax = s->maxx });
      P_NV9097_SET_SCISSOR_VERTICAL(p, 0, { .ymin = s->miny, .ymax = s->maxy });
   } else {
      P_IMMD(p, NV9097, SET_SCISSOR_ENABLE(0), V_FALSE);
   }
}

/* nvk_flush_rs_state (nvk_cmd_draw.c:2224-2350) for the GLES subset. */
static void
nvrm_emit_rasterizer(struct nvrm_context *ctx, struct nv_push *p)
{
   const struct nvrm_screen *screen = nvrm_screen(ctx->base.screen);
   const struct pipe_rasterizer_state *rs = ctx->rast;
   if (rs->fill_front == PIPE_POLYGON_MODE_FILL_RECTANGLE ||
       rs->fill_back == PIPE_POLYGON_MODE_FILL_RECTANGLE ||
       rs->poly_stipple_enable || rs->line_stipple_enable ||
       rs->sprite_coord_enable ||
       rs->clip_plane_enable) {
      nvrm_error("rasterizer state outside G0b's subset (fill %u/%u, offset %u, "
                 "stipple %u/%u, ms %u, flat %u, discard %u)\n",
                 rs->fill_front, rs->fill_back, rs->offset_tri, rs->poly_stipple_enable,
                 rs->line_stipple_enable, rs->multisample, rs->flatshade,
                 rs->rasterizer_discard);
      abort();
   }
   /* GL_MULTISAMPLE (always on in GLES), nvc0's MULTISAMPLE_ENABLE. */
   P_IMMD(p, NV9097, SET_ANTI_ALIAS_ENABLE, rs->multisample);
   /* GL_RASTERIZER_DISCARD, as nvk_flush_rs_state's SET_RASTER_ENABLE. */
   P_IMMD(p, NV9097, SET_RASTER_ENABLE, !rs->rasterizer_discard);

   /* nvk_flush_rs_state's SET_VIEWPORT_CLIP_CONTROL (nvk_cmd_draw.c:
    * 2226-2293); the 3D init only sets the WZERO_CLIP default, so without
    * this nothing is clipped against near/far (dEQP clipping.*). GL depth
    * ranges are [0,1] (z_clamp_zero_one). Pre-Volta: same word without
    * z_clip_range (field absent in NV9097; NVK computes it as
    * USE_FIELD_MIN_Z_ZERO_MAX_Z_ONE = 0 there), the clamp lives in
    * VIEWPORT_CLIP_MIN/MAX_Z (nvrm_emit_viewport). */
   const bool z_clamp = !rs->depth_clip_near || !rs->depth_clip_far;
   const bool z_clip = rs->depth_clip_near && rs->depth_clip_far;
   if (screen->info.cls_eng3d < VOLTA_A) {
      P_IMMD(p, NV9097, SET_VIEWPORT_CLIP_CONTROL, {
         .min_z_zero_max_z_one = !z_clamp ? MIN_Z_ZERO_MAX_Z_ONE_TRUE
                                          : MIN_Z_ZERO_MAX_Z_ONE_FALSE,
         .pixel_min_z = PIXEL_MIN_Z_CLAMP,
         .pixel_max_z = PIXEL_MAX_Z_CLAMP,
         .geometry_guardband = GEOMETRY_GUARDBAND_SCALE_256,
         .line_point_cull_guardband = LINE_POINT_CULL_GUARDBAND_SCALE_256,
         .geometry_clip = z_clip ? GEOMETRY_CLIP_FRUSTUM_XYZ_CLIP
                                 : GEOMETRY_CLIP_FRUSTUM_XY_CLIP,
         .geometry_guardband_z = z_clip ? GEOMETRY_GUARDBAND_Z_SCALE_1
                                        : GEOMETRY_GUARDBAND_Z_SCALE_256,
      });
   } else {
      P_IMMD(p, NVC397, SET_VIEWPORT_CLIP_CONTROL, {
         .min_z_zero_max_z_one = !z_clamp ? MIN_Z_ZERO_MAX_Z_ONE_TRUE
                                          : MIN_Z_ZERO_MAX_Z_ONE_FALSE,
         .z_clip_range = z_clamp ? Z_CLIP_RANGE_MIN_Z_MAX_Z
                                 : Z_CLIP_RANGE_USE_FIELD_MIN_Z_ZERO_MAX_Z_ONE,
         .pixel_min_z = PIXEL_MIN_Z_CLAMP,
         .pixel_max_z = PIXEL_MAX_Z_CLAMP,
         .geometry_guardband = GEOMETRY_GUARDBAND_SCALE_256,
         .line_point_cull_guardband = LINE_POINT_CULL_GUARDBAND_SCALE_256,
         .geometry_clip = z_clip ? GEOMETRY_CLIP_FRUSTUM_XYZ_CLIP
                                 : GEOMETRY_CLIP_FRUSTUM_XY_CLIP,
         .geometry_guardband_z = z_clip ? GEOMETRY_GUARDBAND_Z_SCALE_1
                                        : GEOMETRY_GUARDBAND_Z_SCALE_256,
      });
   }
   /* nvc0_rasterizer_state_create: SHADE_MODEL flat/smooth from flatshade
    * (u_blitter sets it; GLES never does). */
   P_IMMD(p, NV9097, SET_SHADE_MODE, rs->flatshade ? V_OGL_FLAT : V_OGL_SMOOTH);
   /* GL_NV_polygon_mode (glPolygonModeNV, FRONT_AND_BACK only on ES):
    * nvk_flush_rs_state's SET_FRONT/BACK_POLYGON_MODE with
    * vk_to_nv9097_polygon_mode's encoding; PIPE_POLYGON_MODE_FILL/LINE/POINT
    * = VK_POLYGON_MODE_FILL/LINE/POINT = 0/1/2 -> 0x1b02/0x1b01/0x1b00
    * (nvc0's nvgl_polygon_mode maps GL_POINT/LINE/FILL to the same words). */
   STATIC_ASSERT(NV9097_SET_FRONT_POLYGON_MODE_V_FILL == (0x1b00 | (2 - PIPE_POLYGON_MODE_FILL)));
   STATIC_ASSERT(NV9097_SET_FRONT_POLYGON_MODE_V_LINE == (0x1b00 | (2 - PIPE_POLYGON_MODE_LINE)));
   STATIC_ASSERT(NV9097_SET_FRONT_POLYGON_MODE_V_POINT == (0x1b00 | (2 - PIPE_POLYGON_MODE_POINT)));
   P_MTHD(p, NV9097, SET_FRONT_POLYGON_MODE);
   P_NV9097_SET_FRONT_POLYGON_MODE(p, 0x1b00 | (2 - rs->fill_front));
   P_NV9097_SET_BACK_POLYGON_MODE(p, 0x1b00 | (2 - rs->fill_back));
   P_IMMD(p, NV9097, OGL_SET_CULL, rs->cull_face != PIPE_FACE_NONE);
   if (rs->cull_face != PIPE_FACE_NONE) {
      P_IMMD(p, NV9097, OGL_SET_CULL_FACE,
             rs->cull_face == PIPE_FACE_FRONT ? V_FRONT :
             rs->cull_face == PIPE_FACE_BACK ? V_BACK : V_FRONT_AND_BACK);
   }
   P_IMMD(p, NV9097, OGL_SET_FRONT_FACE, rs->front_ccw ? V_CCW : V_CW);

   /* Polygon offset as nvc0_rasterizer_state_create: factor as given,
    * units doubled unless unscaled (matches stock GL's 2 x units on D24S8),
    * clamp. Point/line offset enables come
    * from GL_NV_polygon_mode (POLYGON_OFFSET_POINT/LINE_NV). */
   P_MTHD(p, NV9097, SET_POLY_OFFSET_POINT);
   P_NV9097_SET_POLY_OFFSET_POINT(p, rs->offset_point);
   P_NV9097_SET_POLY_OFFSET_LINE(p, rs->offset_line);
   P_NV9097_SET_POLY_OFFSET_FILL(p, rs->offset_tri);
   if (rs->offset_tri || rs->offset_line || rs->offset_point) {
      /* GL's r is the format's minimum resolvable difference (NVK's
       * LEAST_REPRESENTABLE_VALUE_FORMAT branch). NVK notes "the blob
       * multiplies by 2"; nvc0 and stock GL do. */
      P_IMMD(p, NV9097, SET_DEPTH_BIAS_CONTROL, DEPTH_FORMAT_DEPENDENT_TRUE);
      P_IMMD(p, NV9097, SET_SLOPE_SCALE_DEPTH_BIAS, fui(rs->offset_scale));
      P_IMMD(p, NV9097, SET_DEPTH_BIAS,
             fui(rs->offset_units_unscaled ? rs->offset_units : rs->offset_units * 2.0f));
      P_IMMD(p, NV9097, SET_DEPTH_BIAS_CLAMP, fui(rs->offset_clamp));
   }
   P_IMMD(p, NV9097, SET_PROVOKING_VERTEX, rs->flatshade_first ? V_FIRST : V_LAST);
   /* Line mode as nvk_flush_rs_state's LINE_MODE switch: GL aliased lines
    * (line_rectangular false) are the Bresenham mode, line_smooth the
    * rectangular-smooth mode. */
   const bool line_override = rs->line_smooth || !rs->line_rectangular;
   P_IMMD(p, NV9097, SET_LINE_MULTISAMPLE_OVERRIDE, line_override);
   P_IMMD(p, NV9097, SET_ANTI_ALIASED_LINE, rs->line_smooth);
   P_IMMD(p, NV9097, SET_LINE_WIDTH_FLOAT, fui(rs->line_width));
   P_IMMD(p, NV9097, SET_ALIASED_LINE_WIDTH_FLOAT, fui(rs->line_width));
   P_IMMD(p, NV9097, SET_POINT_SIZE, fui(rs->point_size));

   P_MTHD(p, NV9097, SET_SAMPLE_MASK_X0_Y0);
   P_NV9097_SET_SAMPLE_MASK_X0_Y0(p, ctx->sample_mask & 0xffff);
   P_NV9097_SET_SAMPLE_MASK_X1_Y0(p, ctx->sample_mask & 0xffff);
   P_NV9097_SET_SAMPLE_MASK_X0_Y1(p, ctx->sample_mask & 0xffff);
   P_NV9097_SET_SAMPLE_MASK_X1_Y1(p, ctx->sample_mask & 0xffff);
}

static void
nvrm_emit_dsa_blend(struct nvrm_context *ctx, struct nv_push *p)
{
   const struct pipe_depth_stencil_alpha_state *dsa = ctx->dsa;
   if (dsa->alpha_enabled || dsa->depth_bounds_test) {
      nvrm_error("alpha test / depth bounds test are not implemented (G1)\n");
      abort();
   }
   /* nvk_flush_ds_state (nvk_cmd_draw.c) with GL encodings: compare funcs
    * are OGL 0x200 + PIPE_FUNC_*, stencil ops the OGL enums. Two-sided
    * stencil is enabled at init (SET_TWO_SIDED_STENCIL_TEST). */
   P_IMMD(p, NV9097, SET_DEPTH_TEST, dsa->depth_enabled);
   P_IMMD(p, NV9097, SET_DEPTH_WRITE, dsa->depth_enabled && dsa->depth_writemask);
   if (dsa->depth_enabled)
      P_IMMD(p, NV9097, SET_DEPTH_FUNC, 0x200 + dsa->depth_func);
   P_IMMD(p, NV9097, SET_DEPTH_BOUNDS_TEST, false);
   const struct pipe_stencil_state *front = &dsa->stencil[0];
   const struct pipe_stencil_state *back = dsa->stencil[1].enabled ? &dsa->stencil[1]
                                                                   : &dsa->stencil[0];
   P_IMMD(p, NV9097, SET_STENCIL_TEST, front->enabled);
   if (front->enabled) {
      P_MTHD(p, NV9097, SET_STENCIL_OP_FAIL);
      P_NV9097_SET_STENCIL_OP_FAIL(p, nvrm_stencil_op(front->fail_op));
      P_NV9097_SET_STENCIL_OP_ZFAIL(p, nvrm_stencil_op(front->zfail_op));
      P_NV9097_SET_STENCIL_OP_ZPASS(p, nvrm_stencil_op(front->zpass_op));
      P_NV9097_SET_STENCIL_FUNC(p, 0x200 + front->func);
      P_NV9097_SET_STENCIL_FUNC_REF(p, ctx->stencil_ref.ref_value[0]);
      P_NV9097_SET_STENCIL_FUNC_MASK(p, front->valuemask);
      P_NV9097_SET_STENCIL_MASK(p, front->writemask);
      P_MTHD(p, NV9097, SET_BACK_STENCIL_OP_FAIL);
      P_NV9097_SET_BACK_STENCIL_OP_FAIL(p, nvrm_stencil_op(back->fail_op));
      P_NV9097_SET_BACK_STENCIL_OP_ZFAIL(p, nvrm_stencil_op(back->zfail_op));
      P_NV9097_SET_BACK_STENCIL_OP_ZPASS(p, nvrm_stencil_op(back->zpass_op));
      P_NV9097_SET_BACK_STENCIL_FUNC(p, 0x200 + back->func);
      P_MTHD(p, NV9097, SET_BACK_STENCIL_FUNC_REF);
      P_NV9097_SET_BACK_STENCIL_FUNC_REF(p, ctx->stencil_ref.ref_value[dsa->stencil[1].enabled ? 1 : 0]);
      P_NV9097_SET_BACK_STENCIL_MASK(p, back->writemask);
      P_NV9097_SET_BACK_STENCIL_FUNC_MASK(p, back->valuemask);
   }

   /* nvk_flush_cb_state (nvk_cmd_draw.c:3240-3330) per colour target;
    * gallium's rt[0] applies to all unless independent_blend_enable. GL
    * ignores blending on integer targets (nvc0 nvc0_blend_state_create
    * leaves that to the format; NVK never sees it): disabled here. */
   const struct pipe_blend_state *bl = ctx->blend;
   if (bl->logicop_enable) {
      nvrm_error("logic op is not implemented (G1)\n");
      abort();
   }
   /* nvk_flush_ms_state's SET_ANTI_ALIAS_ALPHA_CONTROL. */
   P_IMMD(p, NV9097, SET_ANTI_ALIAS_ALPHA_CONTROL, {
      .alpha_to_coverage = bl->alpha_to_coverage,
      .alpha_to_one = bl->alpha_to_one,
   });
   /* NVK_MME_SET_WRITE_MASK (nvk_cmd_draw.c:3229-3281): per-target
    * SET_CT_WRITE, and SET_SINGLE_CT_WRITE_CONTROL true only while every
    * target's mask equals target 0's. With it true the hardware applies
    * CT_WRITE(0) to every target (the channel init sets it true), so
    * glColorMaski masks were ignored (dEQP-GLES3 draw_buffers_indexed). */
   bool common_mask = true;
   for (unsigned i = 1; i < NVRM_MAX_RTS; i++) {
      if (bl->rt[bl->independent_blend_enable ? i : 0].colormask != bl->rt[0].colormask)
         common_mask = false;
   }
   P_IMMD(p, NV9097, SET_SINGLE_CT_WRITE_CONTROL, common_mask);
   for (unsigned i = 0; i < NVRM_MAX_RTS; i++) {
      const struct pipe_rt_blend_state *rt = &bl->rt[bl->independent_blend_enable ? i : 0];
      const bool int_target = i < ctx->fb.nr_cbufs && ctx->fb.cbufs[i].texture &&
                              util_format_is_pure_integer(ctx->fb.cbufs[i].format);
      const bool enable = rt->blend_enable && !int_target;
      P_IMMD(p, NV9097, SET_BLEND(i), enable);
      if (enable) {
         P_MTHD(p, NV9097, SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA(i));
         P_NV9097_SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA(p, i, ENABLE_TRUE);
         P_NV9097_SET_BLEND_PER_TARGET_COLOR_OP(p, i, nvrm_blend_op(rt->rgb_func));
         P_NV9097_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF(p, i, nvrm_blend_factor(rt->rgb_src_factor));
         P_NV9097_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF(p, i, nvrm_blend_factor(rt->rgb_dst_factor));
         P_NV9097_SET_BLEND_PER_TARGET_ALPHA_OP(p, i, nvrm_blend_op(rt->alpha_func));
         P_NV9097_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF(p, i, nvrm_blend_factor(rt->alpha_src_factor));
         P_NV9097_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF(p, i, nvrm_blend_factor(rt->alpha_dst_factor));
      }
      P_IMMD(p, NV9097, SET_CT_WRITE(i), {
         .r_enable = !!(rt->colormask & PIPE_MASK_R),
         .g_enable = !!(rt->colormask & PIPE_MASK_G),
         .b_enable = !!(rt->colormask & PIPE_MASK_B),
         .a_enable = !!(rt->colormask & PIPE_MASK_A),
      });
   }
   P_MTHD(p, NV9097, SET_BLEND_CONST_RED);
   P_NV9097_SET_BLEND_CONST_RED(p, fui(ctx->blend_color.color[0]));
   P_NV9097_SET_BLEND_CONST_GREEN(p, fui(ctx->blend_color.color[1]));
   P_NV9097_SET_BLEND_CONST_BLUE(p, fui(ctx->blend_color.color[2]));
   P_NV9097_SET_BLEND_CONST_ALPHA(p, fui(ctx->blend_color.color[3]));
}

/* nvk_shader_fill_push (nvk_shader.c:648-840) minus the NVK MME calls:
 * TESS params (no tessellation), SET_SHADING_RATE_CONTROL (left at the
 * channel default) and SET_ANTI_ALIAS (emitted in nvrm_emit_framebuffer).
 * The "last VTG" part (RT layer select, clip/cull, stream out) is emitted
 * for the last vertex-processing stage only, as nvk_flush_shaders replays
 * only last_vtgm's vtgm push range (nvk_cmd_draw.c:1747-1765). Geometry
 * shader topology, max output vertices and invocations live in the shader
 * program header NAK writes (no separate methods in NVK either). */
static void
nvrm_emit_shader(struct nvrm_screen *screen, struct nv_push *p,
                 const struct nvrm_shader *shader, bool last_vtg)
{
   const struct nak_shader_info *info = &shader->bin->info;
   /* mesa_to_nv9097_shader_type (nvk_shader.c:606-617); index == type. */
   uint32_t type;
   switch (shader->stage) {
   case MESA_SHADER_VERTEX:   type = NV9097_SET_PIPELINE_SHADER_TYPE_VERTEX; break;
   case MESA_SHADER_GEOMETRY: type = NV9097_SET_PIPELINE_SHADER_TYPE_GEOMETRY; break;
   case MESA_SHADER_FRAGMENT: type = NV9097_SET_PIPELINE_SHADER_TYPE_PIXEL; break;
   default:
      nvrm_error("%s shader bound to a graphics slot\n",
                 _mesa_shader_stage_to_string(shader->stage));
      abort();
   }
   const uint32_t idx = type;

   P_IMMD(p, NV9097, SET_PIPELINE_SHADER(idx), { .enable = ENABLE_TRUE, .type = type });
   /* nvk_shader.c:672-681 */
   if (screen->info.cls_eng3d >= VOLTA_A) {
      P_MTHD(p, NVC397, SET_PIPELINE_PROGRAM_ADDRESS_A(idx));
      P_NVC397_SET_PIPELINE_PROGRAM_ADDRESS_A(p, idx, shader->hdr_addr >> 32);
      P_NVC397_SET_PIPELINE_PROGRAM_ADDRESS_B(p, idx, shader->hdr_addr);
   } else {
      P_IMMD(p, NV9097, SET_PIPELINE_PROGRAM(idx), shader->hdr_addr);
   }
   P_MTHD(p, NVC397, SET_PIPELINE_REGISTER_COUNT(idx));
   P_NVC397_SET_PIPELINE_REGISTER_COUNT(p, idx, info->num_gprs);
   P_NVC397_SET_PIPELINE_BINDING(p, idx, shader->stage /* nvk_pipeline_bind_group */);

   if (shader->stage == MESA_SHADER_FRAGMENT) {
      P_MTHD(p, NVC397, SET_SUBTILING_PERF_KNOB_A);
      P_NV9097_SET_SUBTILING_PERF_KNOB_A(p, {
         .fraction_of_spm_register_file_per_subtile         = 0x10,
         .fraction_of_spm_pixel_output_buffer_per_subtile   = 0x40,
         .fraction_of_spm_triangle_ram_per_subtile          = 0x16,
         .fraction_of_max_quads_per_subtile                 = 0x20,
      });
      P_NV9097_SET_SUBTILING_PERF_KNOB_B(p, 0x20);
      P_IMMD(p, NV9097, SET_API_MANDATED_EARLY_Z, info->fs.early_fragment_tests);
      if (screen->info.cls_eng3d >= MAXWELL_B) /* nvk_shader.c:711-716 */
         P_IMMD(p, NVB197, SET_POST_Z_PS_IMASK, info->fs.post_depth_coverage);
      else
         assert(!info->fs.post_depth_coverage);
      P_IMMD(p, NV9097, SET_ZCULL_BOUNDS, {
         .z_min_unbounded_enable = info->fs.writes_depth,
         .z_max_unbounded_enable = info->fs.writes_depth,
      });
      return;
   }
   if (!last_vtg)
      return;

   /* Last VTG stage part. */
   P_IMMD(p, NV9097, SET_RT_LAYER, {
      .v = 0,
      .control = info->vtg.writes_layer ? CONTROL_GEOMETRY_SHADER_SELECTS_LAYER
                                        : CONTROL_V_SELECTS_LAYER,
   });
   if (screen->info.cls_eng3d >= AMPERE_B) {
      P_IMMD(p, NVC797, SET_VARIABLE_PIXEL_RATE_SHADING_TABLE_SELECT, {
         .source = info->vtg.writes_vprs_table_index ? SOURCE_FROM_VPRS_TABLE_INDEX
                                                     : SOURCE_FROM_CONSTANT,
         .source_constant_value = 0,
      });
   }
   const uint8_t clip = info->vtg.clip_enable, cull = info->vtg.cull_enable;
   P_IMMD(p, NV9097, SET_USER_CLIP_ENABLE, {
      .plane0 = ((clip | cull) >> 0) & 1, .plane1 = ((clip | cull) >> 1) & 1,
      .plane2 = ((clip | cull) >> 2) & 1, .plane3 = ((clip | cull) >> 3) & 1,
      .plane4 = ((clip | cull) >> 4) & 1, .plane5 = ((clip | cull) >> 5) & 1,
      .plane6 = ((clip | cull) >> 6) & 1, .plane7 = ((clip | cull) >> 7) & 1,
   });
   P_IMMD(p, NV9097, SET_USER_CLIP_OP, {
      .plane0 = (cull >> 0) & 1, .plane1 = (cull >> 1) & 1,
      .plane2 = (cull >> 2) & 1, .plane3 = (cull >> 3) & 1,
      .plane4 = (cull >> 4) & 1, .plane5 = (cull >> 5) & 1,
      .plane6 = (cull >> 6) & 1, .plane7 = (cull >> 7) & 1,
   });
   const struct nak_xfb_info *xfb = &info->vtg.xfb;
   for (uint8_t b = 0; b < 4; b++) {
      P_MTHD(p, NV9097, SET_STREAM_OUT_CONTROL_STREAM(b));
      P_NV9097_SET_STREAM_OUT_CONTROL_STREAM(p, b, xfb->stream[b]);
      P_NV9097_SET_STREAM_OUT_CONTROL_COMPONENT_COUNT(p, b, xfb->attr_count[b]);
      P_NV9097_SET_STREAM_OUT_CONTROL_STRIDE(p, b, xfb->stride[b]);
      if (xfb->attr_count[b] > 0) {
         /* Packed varying indices, as nvk_shader_fill_push. */
         P_MTHD(p, NV9097, SET_STREAM_OUT_LAYOUT_SELECT(b, 0));
         P_INLINE_ARRAY(p, (const uint32_t *)xfb->attr_index[b],
                        DIV_ROUND_UP(xfb->attr_count[b], 4));
      }
   }
}

static void
nvrm_emit_shaders(struct nvrm_context *ctx, struct nv_push *p)
{
   struct nvrm_screen *screen = nvrm_screen(ctx->base.screen);
   if (!ctx->vs || !ctx->fs) {
      nvrm_error("draw without a vertex and a fragment shader bound\n");
      abort();
   }
   /* Unused stages off, as nvk_flush_shaders (nvk_cmd_draw.c:1735-1742). */
   static const uint32_t off_types[] = {
      NV9097_SET_PIPELINE_SHADER_TYPE_VERTEX_CULL_BEFORE_FETCH,
      NV9097_SET_PIPELINE_SHADER_TYPE_TESSELLATION_INIT,
      NV9097_SET_PIPELINE_SHADER_TYPE_TESSELLATION,
      NV9097_SET_PIPELINE_SHADER_TYPE_GEOMETRY,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(off_types); i++) {
      if (off_types[i] == NV9097_SET_PIPELINE_SHADER_TYPE_GEOMETRY && ctx->gs)
         continue;
      P_IMMD(p, NV9097, SET_PIPELINE_SHADER(off_types[i]), {
         .enable = ENABLE_FALSE, .type = off_types[i],
      });
   }
   nvrm_emit_shader(screen, p, ctx->vs, ctx->gs == NULL);
   if (ctx->gs)
      nvrm_emit_shader(screen, p, ctx->gs, true);
   nvrm_emit_shader(screen, p, ctx->fs, false);
}

/* Driver cbuf loads: LOAD_CONSTANT_BUFFER (NVK's root table loads,
 * nvk_cmd_draw.c:681-694; nvc0_cb_bo_push, nvc0_transfer.c:576-605) of the
 * dwords that differ from the shadow, runs closer than 3 dwords merged,
 * at most NV_PUSH_MAX_COUNT - 1 data dwords per method. */
uint32_t
nvrm_drv_cb_load_dw(uint32_t size_B)
{
   /* selector 4; worst case one 1-dword run per 3 dwords (2 + 1 each) */
   return 4 + 2 + size_B / 4 * 2 + 2 * (size_B / 4 / (NV_PUSH_MAX_COUNT - 1) + 1);
}

void
nvrm_drv_cb_load(struct nvrm_context *ctx, struct nv_push *p, int gfx_stage,
                 uint32_t base_B, uint32_t *shadow, uint32_t *valid_B,
                 const uint32_t *data, uint32_t size_B)
{
   const uint32_t n = size_B / 4;
   /* Dwords below *valid_B hold exactly the shadow. */
   const uint32_t known = MIN2(*valid_B, size_B) / 4;
   bool selected = false;
   uint32_t i = 0;
   while (i < n) {
      if (i < known && shadow[i] == data[i]) {
         i++;
         continue;
      }
      /* Run [i, end): extend over differing dwords and gaps < 3. */
      uint32_t end = i + 1, same = 0;
      for (uint32_t j = i + 1; j < n; j++) {
         if (j < known && shadow[j] == data[j]) {
            if (++same >= 3)
               break;
         } else {
            same = 0;
            end = j + 1;
         }
      }
      if (!selected) {
         const uint32_t window = base_B >= NVRM_DRV_CB_UNIFORM_B ? NVRM_DRV_CB_HANDLES_B
                                                                : NVRM_DRV_CB_UNIFORM_B;
         const uint64_t addr = ctx->drv_cb.va + (uint64_t)gfx_stage * NVRM_DRV_CB_STAGE_B +
                               (base_B >= NVRM_DRV_CB_UNIFORM_B ? NVRM_DRV_CB_UNIFORM_B : 0);
         P_MTHD(p, NV9097, SET_CONSTANT_BUFFER_SELECTOR_A);
         P_NV9097_SET_CONSTANT_BUFFER_SELECTOR_A(p, window);
         P_NV9097_SET_CONSTANT_BUFFER_SELECTOR_B(p, addr >> 32);
         P_NV9097_SET_CONSTANT_BUFFER_SELECTOR_C(p, addr);
         selected = true;
      }
      for (uint32_t at = i; at < end;) {
         const uint32_t cnt = MIN2(end - at, NV_PUSH_MAX_COUNT - 1);
         const uint32_t off_B = (base_B >= NVRM_DRV_CB_UNIFORM_B ? base_B - NVRM_DRV_CB_UNIFORM_B
                                                                 : base_B) + at * 4;
         P_1INC(p, NV9097, LOAD_CONSTANT_BUFFER_OFFSET);
         P_NV9097_LOAD_CONSTANT_BUFFER_OFFSET(p, off_B);
         P_INLINE_ARRAY(p, &data[at], cnt);
         at += cnt;
      }
      memcpy(&shadow[i], &data[i], (end - i) * 4);
      i = end;
   }
   *valid_B = MAX2(*valid_B, size_B);
}

/* Constant buffer slots of each graphics stage whose binding is dirty and
 * that the bound shader reads, as nvk_cmd_flush_gfx_cbufs
 * (nvk_cmd_draw.c:3524-3610): SELECTOR then BIND_GROUP_CONSTANT_BUFFER per
 * slot; slots the shader does not read stay dirty until one does. Slot 0
 * with user constants is the stage's driver cbuf area. */
static void
nvrm_emit_cbufs(struct nvrm_context *ctx, struct nv_push *p)
{
   static const struct { mesa_shader_stage pipe; uint32_t group; } stages[] = {
      { MESA_SHADER_VERTEX, MESA_SHADER_VERTEX },
      { MESA_SHADER_GEOMETRY, MESA_SHADER_GEOMETRY },
      { MESA_SHADER_FRAGMENT, MESA_SHADER_FRAGMENT },
   };
   const struct nvrm_shader *shaders[] = { ctx->vs, ctx->gs, ctx->fs };
   struct nvrm_screen *screen = nvrm_screen(ctx->base.screen);
   for (unsigned s = 0; s < ARRAY_SIZE(stages); s++) {
      if (!shaders[s])
         continue;
      const mesa_shader_stage ps = stages[s].pipe;
      const uint32_t g = stages[s].group;
      const uint32_t rebind = ctx->cb_dirty[ps] & shaders[s]->cbuf_mask &
                              BITFIELD_MASK(NVRM_DESC_CBUF_SLOT);
      u_foreach_bit(i, rebind) {
         const struct pipe_constant_buffer *cb = &ctx->cb[ps][i];
         uint64_t addr;
         uint32_t size;
         if (i == 0 && ctx->cb_user[ps]) {
            /* Bound with exactly the selector window the loads use, as nvc0
             * binds and pushes its uniform area with the same
             * NVC0_MAX_CONSTBUF_SIZE (nvc0_state_validate.c:541-549).
             * Measured 10-15 (RTX 5090, tests/gl-perf-mc.c): binding
             * align(size, 64) while loading through a 64 KiB selector at the
             * same address left draws reading stale uniforms (checksum
             * 24db0a28 instead of 2a846dc0). */
            addr = ctx->drv_cb.va + (uint64_t)nvrm_gfx_stage(ps) * NVRM_DRV_CB_STAGE_B;
            size = NVRM_DRV_CB_UNIFORM_B;
         } else if (cb->buffer) {
            nvrm_keep(ctx, cb->buffer);
            addr = nvrm_resource(cb->buffer)->bo.va + cb->buffer_offset;
            size = MIN2(nvrm_cbuf_size(screen, cb->buffer_size), 65536);
         } else {
            P_IMMD(p, NV9097, BIND_GROUP_CONSTANT_BUFFER(g), {
               .valid = VALID_FALSE, .shader_slot = i,
            });
            continue;
         }
         P_MTHD(p, NV9097, SET_CONSTANT_BUFFER_SELECTOR_A);
         P_NV9097_SET_CONSTANT_BUFFER_SELECTOR_A(p, size);
         P_NV9097_SET_CONSTANT_BUFFER_SELECTOR_B(p, addr >> 32);
         P_NV9097_SET_CONSTANT_BUFFER_SELECTOR_C(p, addr);
         P_IMMD(p, NV9097, BIND_GROUP_CONSTANT_BUFFER(g), {
            .valid = VALID_TRUE, .shader_slot = i,
         });
      }
      ctx->cb_dirty[ps] &= ~rebind;
   }
}

/* Pending cache invalidates (NVRM_INVAL_*, nvrm_priv.h), each emitted once
 * for the event that set it. */
void
nvrm_emit_invalidates(struct nvrm_context *ctx, struct nv_push *p)
{
   struct nvrm_screen *screen = nvrm_screen(ctx->base.screen);
   const uint64_t gen = p_atomic_read(&screen->shader_gen);
   if (gen != ctx->shader_gen_seen) {
      ctx->inval |= NVRM_INVAL_INSTR;
      ctx->shader_gen_seen = gen;
   }
   const uint32_t inval = ctx->inval;
   if (!inval)
      return;
   if (inval & (NVRM_INVAL_BATCH | NVRM_INVAL_HEADER)) {
      /* nvk_cmd_buffer_begin_graphics (nvk_cmd_draw.c:700-708). */
      P_IMMD(p, NV9097, INVALIDATE_SAMPLER_CACHE_NO_WFI, { .lines = LINES_ALL });
      P_IMMD(p, NV9097, INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI, { .lines = LINES_ALL });
   }
   if (inval & (NVRM_INVAL_BATCH | NVRM_INVAL_TEX_DATA))
      P_IMMD(p, NVA097, INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI, { .lines = LINES_ALL });
   if (inval & (NVRM_INVAL_BATCH | NVRM_INVAL_CONSTANT | NVRM_INVAL_GLOBAL |
                NVRM_INVAL_INSTR)) {
      /* nvk_cmd_buffer.c:624-630 field by field. */
      P_IMMD(p, NVA097, INVALIDATE_SHADER_CACHES_NO_WFI, {
         .instruction = (inval & NVRM_INVAL_INSTR) != 0,
         .global_data = (inval & (NVRM_INVAL_BATCH | NVRM_INVAL_GLOBAL)) != 0,
         .constant = (inval & (NVRM_INVAL_BATCH | NVRM_INVAL_CONSTANT)) != 0,
      });
   }
   ctx->inval = 0;
}

/* Vertex attributes and streams: nvk_flush_vi_state (nvk_cmd_draw.c:
 * 1830-1860) and the VB bind macro (:3740-3780, Turing+ size form). */
static void
nvrm_emit_vertex(struct nvrm_context *ctx, struct nv_push *p)
{
   struct nvrm_screen *screen = nvrm_screen(ctx->base.screen);
   const struct nvrm_velems *ve = ctx->velems;
   uint32_t stride[PIPE_MAX_ATTRIBS] = { 0 };
   uint32_t divisor[PIPE_MAX_ATTRIBS] = { 0 };
   uint32_t used = 0;

   for (unsigned a = 0; a < ve->count; a++) {
      const struct pipe_vertex_element *el = &ve->el[a];
      struct nvrm_va_format fmt;
      if (!nvrm_get_va_format(el->src_format, &fmt)) {
         nvrm_error("vertex format %s is not implemented in G0b\n",
                    util_format_short_name(el->src_format));
         abort();
      }
      P_IMMD(p, NV9097, SET_VERTEX_ATTRIBUTE_A(a), {
         .stream = el->vertex_buffer_index,
         .source = SOURCE_ACTIVE,
         .offset = el->src_offset,
         .component_bit_widths = fmt.widths,
         .numerical_type = fmt.type,
         .swap_r_and_b = fmt.swap_rb,
      });
      used |= 1u << el->vertex_buffer_index;
      stride[el->vertex_buffer_index] = el->src_stride;
      divisor[el->vertex_buffer_index] = el->instance_divisor;
   }

   const uint32_t streams = MAX2(ctx->num_vb, ctx->emitted_vb_count);
   for (unsigned b = 0; b < streams; b++) {
      const bool enable = b < ctx->num_vb && (used & (1u << b)) &&
                          ctx->vb[b].buffer.resource;
      if (!enable) {
         P_IMMD(p, NV9097, SET_VERTEX_STREAM_A_FORMAT(b), { .enable = false });
         continue;
      }
      const struct pipe_vertex_buffer *vb = &ctx->vb[b];
      struct nvrm_resource *res = nvrm_resource(vb->buffer.resource);
      nvrm_keep(ctx, &res->base);
      uint64_t addr = res->bo.va + vb->buffer_offset;
      const uint32_t size = res->base.width0 > vb->buffer_offset ?
                            res->base.width0 - vb->buffer_offset : 0;
      /* Pre-Turing has no size method: an empty stream points at the zero
       * page and its limit becomes zero_page - 1, as NVK's nvk_mme_bind_vb
       * (nvk_cmd_draw.c:3750-3755, 3770-3777); addr + 0 - 1 never ends
       * below an arbitrary buffer address. */
      if (size == 0 && screen->info.cls_eng3d < TURING_A)
         addr = screen->zero_page.va;
      P_MTHD(p, NV9097, SET_VERTEX_STREAM_A_FORMAT(b));
      P_NV9097_SET_VERTEX_STREAM_A_FORMAT(p, b, { .stride = stride[b], .enable = true });
      P_NV9097_SET_VERTEX_STREAM_A_LOCATION_A(p, b, addr >> 32);
      P_NV9097_SET_VERTEX_STREAM_A_LOCATION_B(p, b, addr);
      if (screen->info.cls_eng3d >= TURING_A) {
         P_MTHD(p, NVC597, SET_VERTEX_STREAM_SIZE_A(b));
         P_NVC597_SET_VERTEX_STREAM_SIZE_A(p, b, 0);
         P_NVC597_SET_VERTEX_STREAM_SIZE_B(p, b, size);
      } else {
         const uint64_t end = addr + size - 1;
         P_MTHD(p, NV9097, SET_VERTEX_STREAM_LIMIT_A_A(b));
         P_NV9097_SET_VERTEX_STREAM_LIMIT_A_A(p, b, end >> 32);
         P_NV9097_SET_VERTEX_STREAM_LIMIT_A_B(p, b, end);
      }
      P_IMMD(p, NV9097, SET_VERTEX_STREAM_INSTANCE_A(b), divisor[b] != 0);
      P_IMMD(p, NV9097, SET_VERTEX_STREAM_A_FREQUENCY(b), divisor[b]);
   }
   ctx->emitted_vb_count = ctx->num_vb;
}

/* ---------------------------------------------------------------- SLM */

/* nvk_slm_area_ensure (nvk_device.c:56-120) sizing, per context. The old
 * area is released after a synchronous flush, so nothing pending uses it. */
void
nvrm_slm_ensure(struct nvrm_context *ctx, uint32_t bytes_per_lane, uint32_t crs_per_warp)
{
   if (bytes_per_lane <= ctx->slm_bytes_per_lane && crs_per_warp <= ctx->slm_crs_per_warp)
      return;
   struct nvrm_screen *screen = nvrm_screen(ctx->base.screen);
   const struct nv_device_info *info = &screen->info;
   bytes_per_lane = MAX2(bytes_per_lane, ctx->slm_bytes_per_lane);
   crs_per_warp = MAX2(crs_per_warp, ctx->slm_crs_per_warp);
   const uint64_t per_warp = align64((uint64_t)bytes_per_lane * 32 + crs_per_warp, 0x200);
   const uint64_t per_tpc = align64(per_warp * info->max_warps_per_mp * info->mp_per_tpc,
                                    0x8000);
   const uint64_t size = align64(per_tpc * info->tpc_count, 0x20000);
   ctx->base.flush(&ctx->base, NULL, 0);
   nvrm_bo_free(screen, &ctx->slm);
   if (!nvrm_bo_alloc(screen, size, NVRM_DOMAIN_VRAM, &ctx->slm)) {
      nvrm_error("local memory area of %" PRIu64 " B could not be allocated\n", size);
      abort();
   }
   ctx->slm_bytes_per_lane = bytes_per_lane;
   ctx->slm_crs_per_warp = crs_per_warp;
   ctx->slm_per_warp = per_warp;
   ctx->slm_per_tpc = per_tpc;
   ctx->dirty |= NVRM_DIRTY_POOLS;
}

/* 3D part of nvk_queue_state_update's SLM block (nvk_queue.c:157-164). */
static void
nvrm_emit_slm(struct nvrm_context *ctx, struct nv_push *p)
{
   if (!ctx->slm.mem)
      return;
   P_MTHD(p, NV9097, SET_SHADER_LOCAL_MEMORY_A);
   P_NV9097_SET_SHADER_LOCAL_MEMORY_A(p, ctx->slm.va >> 32);
   P_NV9097_SET_SHADER_LOCAL_MEMORY_B(p, ctx->slm.va);
   P_NV9097_SET_SHADER_LOCAL_MEMORY_C(p, ctx->slm.size >> 32);
   P_NV9097_SET_SHADER_LOCAL_MEMORY_D(p, ctx->slm.size);
   P_NV9097_SET_SHADER_LOCAL_MEMORY_E(p, ctx->slm_per_warp);
}

/* ---------------------------------------------------------------- draw */

static uint32_t
nvrm_prim_op(enum mesa_prim mode)
{
   /* MESA_PRIM_POINTS..POLYGON have NV9097_BEGIN_OP_* values 0..9 and
    * LINES_ADJACENCY..TRIANGLE_STRIP_ADJACENCY 0xA..0xD (cl9097.h
    * BEGIN_OP_*_ADJCY; NVK vk_to_nv9097_primitive_topology,
    * nvk_cmd_draw.c:1870-1898). Patches need tessellation (not exposed). */
   STATIC_ASSERT(MESA_PRIM_TRIANGLE_STRIP_ADJACENCY ==
                 NV9097_BEGIN_OP_TRIANGLESTRIP_ADJCY);
   STATIC_ASSERT(MESA_PRIM_LINES_ADJACENCY == NV9097_BEGIN_OP_LINELIST_ADJCY);
   if (mode > MESA_PRIM_TRIANGLE_STRIP_ADJACENCY) {
      nvrm_error("primitive %u is not implemented in G0b\n", mode);
      abort();
   }
   return mode;
}

static void
nvrm_draw_vbo(struct pipe_context *pctx, const struct pipe_draw_info *info,
              unsigned drawid_offset, const struct pipe_draw_indirect_info *indirect,
              const struct pipe_draw_start_count_bias *draws, unsigned num_draws)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   struct nvrm_screen *screen = nvrm_screen(pctx->screen);
   if (indirect) {
      if (indirect->count_from_stream_output || indirect->indirect_draw_count) {
         nvrm_error("draw-count / transform-feedback indirect draws are not supported\n");
         abort();
      }
   }
   if (!ctx->rast || !ctx->blend || !ctx->dsa || !ctx->velems) {
      nvrm_error("draw without rasterizer/blend/dsa/vertex-elements state\n");
      abort();
   }

   /* Index data first: an upload can create a buffer, never a flush. */
   struct pipe_resource *ib = NULL;
   unsigned ib_offset = 0;
   if (info->index_size) {
      if (info->has_user_indices) {
         const unsigned start_B = draws[0].start * info->index_size;
         unsigned span = 0;
         for (unsigned d = 0; d < num_draws; d++)
            span = MAX2(span, (draws[d].start + draws[d].count) * info->index_size);
         u_upload_data_ref(pctx->stream_uploader, start_B, span - start_B, 4,
                       (const uint8_t *)info->index.user + start_B, &ib_offset, &ib);
         ib_offset -= start_B;
      } else {
         pipe_resource_reference(&ib, info->index.resource);
      }
   }

   if (!ctx->vs || !ctx->fs) {
      nvrm_error("draw without a vertex and a fragment shader bound\n");
      abort();
   }
   /* The bound shaders' compiles (nvrm_compile_shader) finish before
    * anything below reads them. */
   nvrm_shader_wait(ctx->vs);
   nvrm_shader_wait(ctx->fs);
   if (ctx->gs)
      nvrm_shader_wait(ctx->gs);
   /* Local memory before any push is opened: growing it flushes. */
   const struct nvrm_shader *gs = ctx->gs;
   nvrm_slm_ensure(ctx, MAX3(ctx->vs->bin->info.slm_size, ctx->fs->bin->info.slm_size,
                             gs ? gs->bin->info.slm_size : 0),
                   MAX3(ctx->vs->bin->info.crs_size, ctx->fs->bin->info.crs_size,
                        gs ? gs->bin->info.crs_size : 0));
   /* Pool room first: a full ring flushes, which must not happen inside an
    * open push. */
   if (ctx->dirty & NVRM_DIRTY_TEXTURES)
      nvrm_texture_reserve(ctx, ctx->vs->tex_units + ctx->vs->image_units +
                                ctx->fs->tex_units + ctx->fs->image_units +
                                (gs ? gs->tex_units + gs->image_units : 0));
   /* Append offsets of resumed stream output come from the counters (may
    * flush). */
   nvrm_so_prepare(ctx);

   /* Pre-PASCAL_B GPUs have no hardware instance iteration: NVK's draw
    * macros loop BEGIN..END once per instance (nvk_cmd_draw.c:3981-4003,
    * 4126-4148), 8 dwords per (draw, instance) here. */
   const bool hw_instancing = screen->info.cls_eng3d >= PASCAL_B;
   /* Instanced draws there go through NVK's direct draw macros
    * (NVRM_MME_DRAW[_INDEXED] = nvk_mme_draw[_indexed]), which loop the
    * instances in the MME: 8 dwords per draw however many instances
    * (10000-instance dEQP grids overflowed the push buffer unrolled). */
   const bool mme_instancing = !hw_instancing && info->instance_count > 1;
   uint64_t draw_dw = 8ull * num_draws *
                      (hw_instancing || mme_instancing ? 1 : MAX2(info->instance_count, 1));
   /* Pre-Turing indirect draws (nvk_CmdDraw(Indexed)Indirect, < TURING_A
    * branch, nvk_cmd_draw.c:4348-4369, 4448-4469): one macro call per
    * chunk of at most (NV_PUSH_MAX_COUNT - 3) * 4 / stride bytes of
    * commands, 3 dwords each plus the DRAW_BEGIN scratch write; 2 GPFIFO
    * entries per chunk. */
   const uint32_t indirect_cmd_B = info->index_size ? 20 : 16;
   const uint32_t indirect_stride = indirect && indirect->draw_count > 1 ?
                                    indirect->stride : indirect_cmd_B;
   uint32_t indirect_chunks = 0;
   if (indirect && !nvrm_mme_dma(screen)) {
      if (indirect_stride < indirect_cmd_B || indirect_stride % 4) {
         nvrm_error("indirect draw stride %u (command %u B)\n", indirect_stride,
                    indirect_cmd_B);
         abort();
      }
      const uint32_t per_chunk = ((NV_PUSH_MAX_COUNT - 3) * 4) / indirect_stride;
      indirect_chunks = DIV_ROUND_UP(indirect->draw_count, per_chunk);
      if (2 * indirect_chunks + 2 > NVRM_EXEC_HEADROOM) {
         nvrm_error("indirect draw of %u commands needs %u push-indirect chunks; "
                    "one command may use %u\n", indirect->draw_count, indirect_chunks,
                    (NVRM_EXEC_HEADROOM - 2) / 2);
         abort();
      }
   }
   draw_dw += 4 + 4 * (uint64_t)indirect_chunks;
   if (2048 + 3 * nvrm_drv_cb_load_dw(NVRM_DRV_CB_HANDLES_B) + draw_dw > NVRM_PUSH_SIZE_B / 4) {
      nvrm_error("draw of %u x %u instances needs %" PRIu64 " push dwords on 3D "
                 "class 0x%04x (no hardware instance iteration); the push "
                 "buffer holds %u\n", num_draws, info->instance_count,
                 2048 + draw_dw, screen->info.cls_eng3d, NVRM_PUSH_SIZE_B / 4);
      abort();
   }
   /* Linear colour buffers that cannot render linear: shadows copied in
    * (flushes, so after every other flushing preparation). */
   nvrm_fb_shadow_begin(ctx);
   /* + texture handle loads (nvrm_emit_textures) of three stages. */
   struct nv_push push = nvrm_push_begin(ctx, 2048 + 3 * nvrm_drv_cb_load_dw(NVRM_DRV_CB_HANDLES_B) +
                                              (uint32_t)draw_dw);
   struct nv_push *p = &push;

   /* Read after push_begin: a flush there marks everything dirty. */
   const uint32_t dirty = ctx->dirty;
   if (dirty & NVRM_DIRTY_BATCH) {
      /* New batch: channel bindings unknown, CPU writes since the last one. */
      for (unsigned s = 0; s < MESA_SHADER_STAGES; s++)
         ctx->cb_dirty[s] = ~0u;
      ctx->inval |= NVRM_INVAL_BATCH;
   }
   /* The previous render targets may be sampled from here on. */
   if (dirty & NVRM_DIRTY_FB)
      ctx->inval |= NVRM_INVAL_TEX_DATA;
   if (dirty & NVRM_DIRTY_POOLS) {
      nvrm_emit_tex_pools(ctx, p);
      nvrm_emit_slm(ctx, p);
   }
   if (dirty & NVRM_DIRTY_FB)
      nvrm_emit_framebuffer(ctx, p);
   if (dirty & NVRM_DIRTY_VIEWPORT)
      nvrm_emit_viewport(ctx, p);
   if (dirty & NVRM_DIRTY_RAST)
      nvrm_emit_rasterizer(ctx, p);
   if (dirty & NVRM_DIRTY_DSA_BLEND)
      nvrm_emit_dsa_blend(ctx, p);
   if (dirty & NVRM_DIRTY_SHADERS)
      nvrm_emit_shaders(ctx, p);
   nvrm_emit_cbufs(ctx, p);
   /* Storage buffer descriptors (slot NVRM_DESC_CBUF_SLOT), nvrm_compute.c. */
   if (dirty & (NVRM_DIRTY_SSBO | NVRM_DIRTY_SHADERS | NVRM_DIRTY_CBUFS)) {
      if (ctx->vs->uses_desc)
         nvrm_emit_ssbo_desc(ctx, p, MESA_SHADER_VERTEX, MESA_SHADER_VERTEX);
      if (gs && gs->uses_desc)
         nvrm_emit_ssbo_desc(ctx, p, MESA_SHADER_GEOMETRY, MESA_SHADER_GEOMETRY);
      if (ctx->fs->uses_desc)
         nvrm_emit_ssbo_desc(ctx, p, MESA_SHADER_FRAGMENT, MESA_SHADER_FRAGMENT);
   }
   if (dirty & NVRM_DIRTY_VERTEX)
      nvrm_emit_vertex(ctx, p);
   if (dirty & NVRM_DIRTY_COND)
      nvrm_emit_render_enable(ctx, p);
   if (dirty & NVRM_DIRTY_SO)
      nvrm_emit_so(ctx, p);
   if (dirty & NVRM_DIRTY_TEXTURES) {
      nvrm_emit_textures(ctx, p, MESA_SHADER_VERTEX, MESA_SHADER_VERTEX, ctx->vs);
      if (gs)
         nvrm_emit_textures(ctx, p, MESA_SHADER_GEOMETRY, MESA_SHADER_GEOMETRY, gs);
      nvrm_emit_textures(ctx, p, MESA_SHADER_FRAGMENT, MESA_SHADER_FRAGMENT, ctx->fs);
   }
   /* After the texture tables: they may have written pool entries. */
   nvrm_emit_invalidates(ctx, p);
   ctx->dirty = 0;

   uint32_t begin_op = nvrm_prim_op(info->mode);

   if (info->index_size) {
      struct nvrm_resource *res = nvrm_resource(ib);
      uint64_t addr = res->bo.va + ib_offset;
      /* An offset at or past the end is an empty index buffer (never a
       * wrapped uint32 size); pre-Turing binds the zero page for it, as
       * NVK's nvk_mme_bind_ib (nvk_cmd_draw.c:3641-3650, 3661-3668). */
      const uint32_t size = res->base.width0 > ib_offset ?
                            res->base.width0 - ib_offset : 0;
      if (size == 0 && screen->info.cls_eng3d < TURING_A)
         addr = screen->zero_page.va;
      P_MTHD(p, NV9097, SET_INDEX_BUFFER_A);
      P_NV9097_SET_INDEX_BUFFER_A(p, addr >> 32);
      P_NV9097_SET_INDEX_BUFFER_B(p, addr);
      if (screen->info.cls_eng3d >= TURING_A) {
         P_MTHD(p, NVC597, SET_INDEX_BUFFER_SIZE_A);
         P_NVC597_SET_INDEX_BUFFER_SIZE_A(p, 0);
         P_NVC597_SET_INDEX_BUFFER_SIZE_B(p, size);
      } else {
         const uint64_t end = addr + size - 1;
         P_NV9097_SET_INDEX_BUFFER_C(p, end >> 32);
         P_NV9097_SET_INDEX_BUFFER_D(p, end);
      }
      P_IMMD(p, NV9097, SET_INDEX_BUFFER_E,
             info->index_size == 1 ? INDEX_SIZE_ONE_BYTE :
             info->index_size == 2 ? INDEX_SIZE_TWO_BYTES : INDEX_SIZE_FOUR_BYTES);
      P_IMMD(p, NV9097, SET_DA_PRIMITIVE_RESTART, info->primitive_restart);
      if (info->primitive_restart)
         P_IMMD(p, NV9097, SET_DA_PRIMITIVE_RESTART_INDEX, info->restart_index);
   } else {
      P_IMMD(p, NV9097, SET_DA_PRIMITIVE_RESTART, false);
   }
   P_IMMD(p, NV9097, SET_GLOBAL_BASE_INSTANCE_INDEX, info->start_instance);

   if (indirect) {
      /* NVK's indirect draw (nvk_CmdDrawIndirect /
       * nvk_CmdDrawIndexedIndirect, nvk_cmd_draw.c:4312-4370, 4412-4470).
       * The BEGIN word is packed here: PASCAL_B+ iterate instances in
       * hardware (NVK sets INSTANCE_ITERATE_ENABLE in the macro), older
       * classes loop them in the macro with NV9097's BEGIN. */
      uint32_t begin_dw[2];
      struct nv_push bp;
      nv_push_init(&bp, begin_dw, 2, SUBC_MASK_ALL);
      if (hw_instancing) {
         P_MTHD(&bp, NVC197, BEGIN);
         P_NVC197_BEGIN(&bp, {
            .op = begin_op,
            .primitive_id = PRIMITIVE_ID_FIRST,
            .instance_id = INSTANCE_ID_FIRST,
            .split_mode = SPLIT_MODE_NORMAL_BEGIN_NORMAL_END,
            .instance_iterate_enable = INSTANCE_ITERATE_ENABLE_TRUE,
         });
      } else {
         P_MTHD(&bp, NV9097, BEGIN);
         P_NV9097_BEGIN(&bp, {
            .op = begin_op,
            .primitive_id = PRIMITIVE_ID_FIRST,
            .instance_id = INSTANCE_ID_FIRST,
            .split_mode = SPLIT_MODE_NORMAL_BEGIN_NORMAL_END,
         });
      }
      nvrm_keep(ctx, indirect->buffer);
      uint64_t addr = nvrm_resource(indirect->buffer)->bo.va + indirect->offset;
      const uint32_t mme = info->index_size ? NVRM_MME_DRAW_INDEXED_INDIRECT
                                            : NVRM_MME_DRAW_INDIRECT;
      if (nvrm_mme_dma(screen)) {
         /* Turing+: the macro DMA-reads each command (nvrm_mme.c). */
         P_1INC(p, NV9097, CALL_MME_MACRO(mme));
         P_INLINE_DATA(p, begin_dw[1]);
         P_INLINE_DATA(p, addr >> 32);
         P_INLINE_DATA(p, addr);
         P_INLINE_DATA(p, indirect->draw_count);
         P_INLINE_DATA(p, indirect_stride);
      } else {
         /* Pre-Turing: BEGIN via MME scratch (NVK writes DRAW_BEGIN from
          * the CPU, nvk_cmd_draw.c:1920), then per chunk {count, pad_dw}
          * with the commands appended to the macro call as a
          * NO_PREFETCH GPFIFO entry. */
         __push_mthd(p, SUBC_NV9097, nvrm_mme_scratch_draw_begin_mthd());
         P_INLINE_DATA(p, begin_dw[1]);
         const uint32_t max_draws_per_push =
            ((NV_PUSH_MAX_COUNT - 3) * 4) / indirect_stride;
         uint32_t draw_count = indirect->draw_count;
         while (draw_count) {
            const uint32_t count = MIN2(draw_count, max_draws_per_push);
            P_1INC(p, NV9097, CALL_MME_MACRO(mme));
            P_INLINE_DATA(p, count);
            P_INLINE_DATA(p, (indirect_stride - indirect_cmd_B) / 4);
            const uint64_t range = count * (uint64_t)indirect_stride;
            nv_push_update_count(p, range / 4);
            nvrm_push_indirect(ctx, p, addr, range);
            addr += range;
            draw_count -= count;
         }
      }
      num_draws = 0;
   }
   if (mme_instancing && num_draws) {
      /* DRAW_BEGIN scratch as NVK writes it (nvk_cmd_draw.c:1920). */
      uint32_t begin_dw[2];
      struct nv_push bp;
      nv_push_init(&bp, begin_dw, 2, SUBC_MASK_ALL);
      P_MTHD(&bp, NV9097, BEGIN);
      P_NV9097_BEGIN(&bp, {
         .op = begin_op,
         .primitive_id = PRIMITIVE_ID_FIRST,
         .instance_id = INSTANCE_ID_FIRST,
         .split_mode = SPLIT_MODE_NORMAL_BEGIN_NORMAL_END,
      });
      __push_mthd(p, SUBC_NV9097, nvrm_mme_scratch_draw_begin_mthd());
      P_INLINE_DATA(p, begin_dw[1]);
   }
   for (unsigned d = 0; d < num_draws; d++) {
      if (!draws[d].count)
         continue;
      /* nvk_mme_build_set_draw_params: both base-vertex methods, 0 for
       * array draws (the channel state is shared with the indirect MME
       * macros and other contexts). */
      const int32_t base_vertex = !info->index_size ? 0 :
         info->index_bias_varies ? draws[d].index_bias : draws[0].index_bias;
      if (mme_instancing) {
         /* nvk_CmdDraw / nvk_CmdDrawIndexed (nvk_cmd_draw.c:4085-4090,
          * 4231-4237): {draw index, Vk*IndirectCommand words}. */
         P_1INC(p, NV9097, CALL_MME_MACRO(info->index_size ? NVRM_MME_DRAW_INDEXED
                                                           : NVRM_MME_DRAW));
         P_INLINE_DATA(p, drawid_offset + d);
         P_INLINE_DATA(p, draws[d].count);
         P_INLINE_DATA(p, info->instance_count);
         P_INLINE_DATA(p, draws[d].start);
         if (info->index_size)
            P_INLINE_DATA(p, base_vertex);
         P_INLINE_DATA(p, info->start_instance);
         continue;
      }
      P_IMMD(p, NV9097, SET_GLOBAL_BASE_VERTEX_INDEX, base_vertex);
      P_IMMD(p, NV9097, SET_VERTEX_ID_BASE, base_vertex);
      /* nvk_mme_build_draw_loop / _indexed_loop (nvk_cmd_draw.c:3975-4003,
       * 4119-4148): PASCAL_B+ iterate instances in hardware; older classes
       * repeat BEGIN..END per instance, INSTANCE_ID FIRST then SUBSEQUENT. */
      const uint32_t iters = hw_instancing ? 1 : info->instance_count;
      if (hw_instancing)
         P_IMMD(p, NVC197, SET_INSTANCE_COUNT, info->instance_count);
      for (uint32_t i = 0; i < iters; i++) {
         if (hw_instancing) {
            P_IMMD(p, NVC197, BEGIN, {
               .op = begin_op,
               .primitive_id = PRIMITIVE_ID_FIRST,
               .instance_id = INSTANCE_ID_FIRST,
               .split_mode = SPLIT_MODE_NORMAL_BEGIN_NORMAL_END,
               .instance_iterate_enable = INSTANCE_ITERATE_ENABLE_TRUE,
            });
         } else {
            P_IMMD(p, NV9097, BEGIN, {
               .op = begin_op,
               .primitive_id = PRIMITIVE_ID_FIRST,
               .instance_id = i == 0 ? INSTANCE_ID_FIRST : INSTANCE_ID_SUBSEQUENT,
               .split_mode = SPLIT_MODE_NORMAL_BEGIN_NORMAL_END,
            });
         }
         if (info->index_size) {
            P_MTHD(p, NV9097, SET_INDEX_BUFFER_F);
            P_NV9097_SET_INDEX_BUFFER_F(p, draws[d].start);
            P_NV9097_DRAW_INDEX_BUFFER(p, draws[d].count);
         } else {
            P_MTHD(p, NV9097, SET_VERTEX_ARRAY_START);
            P_NV9097_SET_VERTEX_ARRAY_START(p, draws[d].start);
            P_NV9097_DRAW_VERTEX_ARRAY(p, draws[d].count);
         }
         P_IMMD(p, NV9097, END, 0);
      }
   }
   nvrm_push_end(ctx, p);

   /* Pending pushes reference ib until the next flush. */
   if (ib)
      nvrm_keep(ctx, ib);
   pipe_resource_reference(&ib, NULL);
}

void
nvrm_init_draw_functions(struct pipe_context *p)
{
   struct nvrm_context *ctx = nvrm_context(p);
   ctx->sample_mask = 0xffff;

   p->draw_vbo = nvrm_draw_vbo;
   p->create_blend_state = (void *)nvrm_create_blend;
   p->bind_blend_state = nvrm_bind_blend;
   p->delete_blend_state = nvrm_delete_blend;
   p->create_rasterizer_state = (void *)nvrm_create_rasterizer;
   p->bind_rasterizer_state = nvrm_bind_rasterizer;
   p->delete_rasterizer_state = nvrm_delete_rasterizer;
   p->create_depth_stencil_alpha_state = (void *)nvrm_create_dsa;
   p->bind_depth_stencil_alpha_state = nvrm_bind_dsa;
   p->delete_depth_stencil_alpha_state = nvrm_delete_dsa;
   p->create_vertex_elements_state = nvrm_create_vertex_elements;
   p->bind_vertex_elements_state = nvrm_bind_vertex_elements;
   p->delete_vertex_elements_state = nvrm_delete_vertex_elements;
   p->create_vs_state = nvrm_create_shader_state;
   p->bind_vs_state = nvrm_bind_vs_state;
   p->delete_vs_state = nvrm_delete_shader_state;
   p->create_gs_state = nvrm_create_shader_state;
   p->bind_gs_state = nvrm_bind_gs_state;
   p->delete_gs_state = nvrm_delete_shader_state;
   p->create_fs_state = nvrm_create_shader_state;
   p->bind_fs_state = nvrm_bind_fs_state;
   p->delete_fs_state = nvrm_delete_shader_state;
   p->set_viewport_states = nvrm_set_viewport_states;
   p->set_scissor_states = nvrm_set_scissor_states;
   p->set_blend_color = nvrm_set_blend_color;
   p->set_stencil_ref = nvrm_set_stencil_ref;
   p->set_sample_mask = nvrm_set_sample_mask;
   p->set_constant_buffer = nvrm_set_constant_buffer;
   p->set_vertex_buffers = nvrm_set_vertex_buffers;
}

void
nvrm_draw_context_destroy(struct nvrm_context *ctx)
{
   for (unsigned s = 0; s < MESA_SHADER_STAGES; s++)
      for (unsigned i = 0; i < NVRM_MAX_CBUFS; i++)
         pipe_resource_reference(&ctx->cb[s][i].buffer, NULL);
   for (unsigned i = 0; i < ctx->num_vb; i++)
      pipe_vertex_buffer_unreference(&ctx->vb[i]);
   nvrm_bo_free(nvrm_screen(ctx->base.screen), &ctx->slm);
}
