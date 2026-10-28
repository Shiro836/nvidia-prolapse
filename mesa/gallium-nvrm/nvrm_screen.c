/* SPDX-License-Identifier: MIT
 *
 * nvrm screen: device open over nv_rm.h, caps, formats, NAK compiler options,
 * fences, and the per-screen channel with NVK's 3D state init.
 */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <xf86drm.h>

#include "nvrm_priv.h"
#include "nvrm_public.h"

#include "nak.h"
#include "nil.h"
#include "nv_push.h"
#include "nv_push_cl9097.h"
#include "nv_push_clb097.h"
#include "nv_push_clb197.h"
#include "nv_push_clc597.h"
#include "nv_push_clcb97.h"
#include "clc097.h"
#include "clc197.h"
#include "clc397.h"
#include "clc597.h"
#include "clcb97.h"
#include "clcd97.h"
#include "clce97.h"

#include "util/cache_ops.h"
#include "util/log.h"
#include "util/os_misc.h"
#include "util/u_debug.h"
#include "util/u_atomic.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/u_screen.h"
#include "util/os_file.h"
#include "util/disk_cache.h"
#include "util/u_cpu_detect.h"
#include "util/u_queue.h"
#include "compiler/glsl_types.h"
#include "util/hex.h"
#include "util/mesa-blake3.h"
#include <inttypes.h>
#include <errno.h>
#include <unistd.h>
#include "util/libsync.h"

/* ---------------------------------------------------------------- memory */

/* fixed_va != 0: bind at that VA, which the caller owns (a reservation);
 * on failure it is not released. */
static bool
nvrm_bo_alloc_va(struct nvrm_screen *screen, uint64_t size, enum nvrm_domain domain,
                 uint64_t fixed_va, struct nvrm_bo *bo)
{
   memset(bo, 0, sizeof(*bo));
   size = align64(MAX2(size, 1), 4096);
   /* Plain VRAM (never compressible: buffers/arenas are pitch, kind 0,
    * NVK's non-tiled nvkmd_mem; nvkmd_rm.c:740-750 asks compression only
    * for 64 KiB-multiple images). */
   const uint32_t placement = domain == NVRM_DOMAIN_GART ? NV_RM_PLACEMENT_GART
                                                         : NV_RM_PLACEMENT_VRAM;
   struct NvRmAllocation alloc;
   int32_t status = nv_rm_alloc(screen->rm, size, 4096, placement, 0, 0, &alloc);
   if (status) {
      nvrm_error("nv_rm_alloc(%" PRIu64 " B, placement %u) failed: %d\n",
                 size, placement, status);
      return false;
   }
   uint64_t va = fixed_va;
   if (!va) {
      simple_mtx_lock(&screen->va_mutex);
      va = util_vma_heap_alloc(&screen->heap, size, 4096);
      simple_mtx_unlock(&screen->va_mutex);
   }
   if (!va) {
      nvrm_error("VA heap exhausted for %" PRIu64 " B\n", size);
      nv_rm_free(screen->rm, alloc.id);
      return false;
   }
   /* Pitch (kind 0) mapping: G0a resources are linear, as NVK's
    * non-tiled buffers (nvkmd_rm.c rm_mem_alloc_kind + rm_va_bind kind 0). */
   uint64_t mapping;
   status = nv_rm_bind(screen->rm, alloc.id, 0, size, va, 0, &mapping);
   if (status) {
      nvrm_error("nv_rm_bind(va 0x%" PRIx64 ") failed: %d\n", va, status);
      if (!fixed_va) {
         simple_mtx_lock(&screen->va_mutex);
         util_vma_heap_free(&screen->heap, va, size);
         simple_mtx_unlock(&screen->va_mutex);
      }
      nv_rm_free(screen->rm, alloc.id);
      return false;
   }
   uintptr_t cpu = 0;
   if (domain != NVRM_DOMAIN_VRAM) {
      /* GART: system pages; VRAM_MAPPED: BAR1 (the backend maps VRAM
       * allocations through BAR1, as nvkmd_rm.c rm_mem_map). */
      status = nv_rm_map(screen->rm, alloc.id, &cpu);
      if (status || !cpu) {
         nvrm_error("nv_rm_map of %" PRIu64 " B %s failed: %d\n", size,
                    domain == NVRM_DOMAIN_GART ? "GART" : "VRAM (BAR1)", status);
         bo->mem = alloc.id;
         bo->mapping = mapping;
         bo->va = va;
         bo->size = size;
         nvrm_bo_free(screen, bo);
         return false;
      }
   }
   *bo = (struct nvrm_bo){
      .mem = alloc.id, .mapping = mapping, .va = va, .size = size,
      .map = (void *)cpu, .vram = domain != NVRM_DOMAIN_GART,
   };
   return true;
}

bool
nvrm_bo_alloc(struct nvrm_screen *screen, uint64_t size, enum nvrm_domain domain,
              struct nvrm_bo *bo)
{
   return nvrm_bo_alloc_va(screen, size, domain, 0, bo);
}

/* nvk_heap_grow_locked + nvk_mem_arena_add_mem for a contiguous arena: back
 * the next memory at base + committed and hand its range to shader_vma. The
 * vma covers [base, committed - pad): the newly added range starts where the
 * previous pad began, so free space merges across memories and the last
 * NVRM_SHADER_PREFETCH_PAD_B bytes of backing are never allocated (I-cache
 * prefetch, nvk_device.c "overalloc"). Caller holds shader_mutex. */
bool
nvrm_shader_heap_grow(struct nvrm_screen *screen)
{
   const unsigned i = screen->shader_mem_count;
   if (i >= NVRM_SHADER_ARENA_MAX_MEMS)
      return false;
   const uint64_t min = 1ull << NVRM_SHADER_ARENA_MIN_SIZE_LOG2;
   const uint64_t size = i == 0 ? min : (min >> 1) << i;
   if (!nvrm_bo_alloc_va(screen, size, NVRM_DOMAIN_VRAM,
                         screen->shader_base + screen->shader_committed,
                         &screen->shader_mem[i]))
      return false;
   const uint64_t start = screen->shader_base + screen->shader_committed;
   if (i == 0)
      util_vma_heap_free(&screen->shader_vma, start, size - NVRM_SHADER_PREFETCH_PAD_B);
   else
      util_vma_heap_free(&screen->shader_vma, start - NVRM_SHADER_PREFETCH_PAD_B, size);
   screen->shader_committed += size;
   screen->shader_mem_count = i + 1;
   return true;
}

/* The shader arena is video memory that is never CPU-mapped, on every BAR
 * size: program bytes reach it only through the GPU. The proprietary driver
 * does the same (libnvidia-eglcore.so.615.78.08: the Vulkan shader heap is
 * never CPU-mapped and every program is copied from system-memory staging
 * by the copy engine; the GL program code is video memory without a CPU
 * mapping, written by inline-to-memory from the pushbuffer). NVK's upload
 * queue is the model (nvk_heap_upload -> nvk_upload_queue_upload,
 * nvk_heap.c:125-150: a GART staging copy + NV90B5 copy on a device-wide
 * queue, ordered before any later submission). nvrm copies on the screen
 * channel and waits, so every context batch submitted afterwards sees the
 * code; the range was free (freed ranges return to shader_vma only after
 * the work using them retired). The arena is one contiguous VA range, so a
 * copy may straddle two memories. */
static void
nvrm_shader_heap_upload(struct nvrm_screen *screen, uint64_t va, const void *data,
                        uint64_t size)
{
   const uint32_t push_dw = nvrm_copy_buffer_dw(size);
   const uint64_t data_off = align64((uint64_t)push_dw * 4, 4096);
   struct nvrm_bo staging;
   if (!nvrm_bo_alloc(screen, data_off + size, NVRM_DOMAIN_GART, &staging)) {
      nvrm_error("shader upload: %" PRIu64 " B GART staging allocation failed\n", size);
      abort(); /* no silent loss of shader code */
   }
   memcpy((uint8_t *)staging.map + data_off, data, size);
   struct nv_push push;
   nv_push_init(&push, staging.map, push_dw, SUBC_MASK_ALL);
   nvrm_emit_copy_buffer(&push, va, staging.va + data_off, size);
   if (!nvrm_submit_wait(screen, staging.va, nv_push_dw_count(&push) * 4, NULL)) {
      nvrm_error("shader upload: copy-engine submission failed\n");
      abort();
   }
   nvrm_bo_free(screen, &staging);
}

void
nvrm_shader_heap_write(struct nvrm_screen *screen, uint64_t va, const void *data,
                       uint64_t size)
{
   nvrm_shader_heap_upload(screen, va, data, size);
}

/* G1 tiled image: VRAM, not CPU-mapped, VA bound with the PTE kind nil
 * chose (NVK: memory allocated kindless, nvk_image binds the plane VA with
 * nil.pte_kind at max(nil.align_B, bind_align_B); nvkmd_rm.c rm_va_bind). */
bool
nvrm_bo_alloc_image(struct nvrm_screen *screen, const struct nil_image *image,
                    struct nvrm_bo *bo)
{
   memset(bo, 0, sizeof(*bo));
   const uint64_t align = MAX2(image->align_B, 4096);
   const uint64_t size = align64(MAX2(image->size_B, 1), align);
   struct NvRmAllocation alloc;
   int32_t status = nv_rm_alloc(screen->rm, size, align, NV_RM_PLACEMENT_VRAM, 0, 0, &alloc);
   if (status) {
      nvrm_error("nv_rm_alloc(image %" PRIu64 " B, align %" PRIu64 ") failed: %d\n",
                 size, align, status);
      return false;
   }
   simple_mtx_lock(&screen->va_mutex);
   uint64_t va = util_vma_heap_alloc(&screen->heap, size, align);
   simple_mtx_unlock(&screen->va_mutex);
   if (!va) {
      nvrm_error("VA heap exhausted for image %" PRIu64 " B\n", size);
      nv_rm_free(screen->rm, alloc.id);
      return false;
   }
   uint64_t mapping;
   status = nv_rm_bind(screen->rm, alloc.id, 0, size, va, image->pte_kind, &mapping);
   if (status) {
      nvrm_error("nv_rm_bind(image va 0x%" PRIx64 ", kind 0x%x) failed: %d\n",
                 va, image->pte_kind, status);
      simple_mtx_lock(&screen->va_mutex);
      util_vma_heap_free(&screen->heap, va, size);
      simple_mtx_unlock(&screen->va_mutex);
      nv_rm_free(screen->rm, alloc.id);
      return false;
   }
   *bo = (struct nvrm_bo){ .mem = alloc.id, .mapping = mapping, .va = va, .size = size,
                           .vram = true };
   return true;
}

static void
nvrm_va_release(void *data, uint64_t address, uint64_t size)
{
   struct nvrm_screen *screen = data;
   simple_mtx_lock(&screen->va_mutex);
   util_vma_heap_free(&screen->heap, address, size);
   simple_mtx_unlock(&screen->va_mutex);
}

void
nvrm_bo_free(struct nvrm_screen *screen, struct nvrm_bo *bo)
{
   if (!bo->mem)
      return;
   if (bo->map)
      nv_rm_unmap(screen->rm, bo->mem);
   /* Same order as nvkmd_rm.c: VA (and its mappings) first, then memory;
    * the backend defers both past work already submitted. */
   int32_t status = nv_rm_va_free(screen->rm, &bo->mapping, 1, bo->va, bo->size,
                                  nvrm_va_release, screen);
   if (status)
      nvrm_error("nv_rm_va_free(0x%" PRIx64 ") failed: %d (range stays reserved)\n",
                 bo->va, status);
   status = nv_rm_free(screen->rm, bo->mem);
   if (status)
      nvrm_error("nv_rm_free failed: %d\n", status);
   memset(bo, 0, sizeof(*bo));
}

bool
nvrm_submit_wait(struct nvrm_screen *screen, uint64_t va, uint32_t size_B,
                 uint64_t *token_out)
{
   return nvrm_submit_wait_fds(screen, va, size_B, NULL, 0, token_out);
}

bool
nvrm_submit_wait_fds(struct nvrm_screen *screen, uint64_t va, uint32_t size_B,
                     const int *wait_fds, uint32_t wait_count, uint64_t *token_out)
{
   struct NvRmExec exec = { .address = va, .size_B = size_B, .flags = 0 };
   return nvrm_submit_execs_wait_fds(screen, &exec, size_B ? 1 : 0, wait_fds, wait_count,
                                     token_out);
}

/* One batch of GPFIFO entries in order, as NVK's queue submits a command
 * buffer's pushes (nvkmd_rm.c rm_ctx_exec: NO_PREFETCH / INCOMPLETE flags
 * per entry). Ring pressure (VK_NOT_READY = 1, nothing queued, *token = the
 * oldest batch): wait for that batch and retry, as nvkmd_rm.c's submit. */
bool
nvrm_submit_async(struct nvrm_screen *screen, const struct NvRmExec *execs,
                  uint32_t exec_count, const int *wait_fds, uint32_t wait_count,
                  uint64_t *token_out)
{
   for (;;) {
      uint64_t token = 0;
      simple_mtx_lock(&screen->submit_mutex);
      /* Waits: borrowed sync_files, retained by the backend as GPU-side
       * gates (nv_rm_submit_waits), as NVK's imported SYNC_FD waits. */
      int32_t status = nv_rm_submit_waits(screen->rm, screen->channel, execs, exec_count,
                                          screen->fd, wait_fds, wait_count, &token);
      if (status == 0)
         p_atomic_set(&screen->last_token, token);
      simple_mtx_unlock(&screen->submit_mutex);
      if (status == 1) {
         if (!nvrm_wait_token(screen, token))
            return false;
         continue;
      }
      if (status) {
         nvrm_error("nv_rm_submit(%u entries) failed: %d\n", exec_count, status);
         return false;
      }
      if (screen->debug_stats)
         p_atomic_inc(&screen->stat_submits);
      *token_out = token;
      return true;
   }
}

bool
nvrm_token_done(struct nvrm_screen *screen, uint64_t token)
{
   if (!token)
      return true;
   uint32_t complete = 0;
   int32_t status = nv_rm_poll(screen->rm, screen->channel, token, &complete);
   if (status) {
      nvrm_error("nv_rm_poll(token %" PRIu64 ") failed: %d\n", token, status);
      abort(); /* device lost: no silent continuation */
   }
   return complete != 0;
}

bool
nvrm_wait_token(struct nvrm_screen *screen, uint64_t token)
{
   if (nvrm_token_done(screen, token))
      return true;
   if (screen->debug_stats)
      p_atomic_inc(&screen->stat_cpu_waits);
   int32_t status = nv_rm_wait(screen->rm, screen->channel, token, UINT64_MAX);
   if (status) {
      nvrm_error("nv_rm_wait(token %" PRIu64 ") failed: %d\n", token, status);
      return false;
   }
   if (nv_rm_check_status(screen->rm)) {
      nvrm_error("channel error after submission (device lost)\n");
      return false;
   }
   return true;
}

/* Synchronous submission (screen init, shader uploads). */
bool
nvrm_submit_execs_wait_fds(struct nvrm_screen *screen, const struct NvRmExec *execs,
                           uint32_t exec_count, const int *wait_fds,
                           uint32_t wait_count, uint64_t *token_out)
{
   uint64_t token = 0;
   if (!nvrm_submit_async(screen, execs, exec_count, wait_fds, wait_count, &token) ||
       !nvrm_wait_token(screen, token))
      return false;
   if (token_out)
      *token_out = token;
   return true;
}

/* ---------------------------------------------------------------- 3D init */

/* Port of nvk_push_draw_state_init (src/nouveau/vulkan/nvk_cmd_draw.c:140-650
 * at Mesa 26.2.5 (ddaf85e3) + our NVK patch stack) for VOLTA_A+ 3D classes, without
 * the NVK-specific parts:
 *  - MME upload + SET_MME_SHADOW_SCRATCH state (NVK's MME macros; GL has its
 *    own, G0b+), including the SET_PRIV_REG macro calls that clear
 *    gr_gpcs_tpcs_sm_disp_ctrl bit 3 and warp ESR report-mask bit 14.
 *    G0b decision: left out, as stock GL does: the measured stock GL 3D init
 *    (captured from the stock driver's push stream, 158 dwords / 111
 *    methods) contains no
 *    SET_FALCON04, CALL_MME_MACRO or MME shadow write. Whether stock sets the
 *    two registers some other way (RM control) is UNTRACED; NVK needs them
 *    for helper-invocation loads and for unmatched-output OOR exceptions,
 *    both G1+ territory (dEQP will show it).
 *    GLES 3.1 (10-15): helper-invocation SSBO loads measurably need bit 3,
 *    so nvrm now uploads its own macros (nvrm_mme.c: indirect draw/dispatch
 *    + NVK's SET_PRIV_REG) and makes both NVK SET_PRIV_REG calls on
 *    MAXWELL_B+ (nvrm_push_mme_init).
 *  - CB0 select/bind/zero (NVK root descriptor table; GL uniforms are G0b).
 * Maxwell/Pascal (MAXWELL_A..PASCAL_B) take NVK's pre-Volta branches of the
 * same function: SET_ALPHA_FRACTION, SET_PRIM_CIRCULAR_BUFFER_THROTTLE,
 * SET_PROGRAM_REGION (shader addresses become heap-relative, see
 * nvrm_emit_shader), no PS watermarks, the MAXWELL_B-only methods gated as
 * NVK gates them, and SET_SELECT_MAXWELL_TEXTURE_HEADERS on MAXWELL_A.
 * Kepler and older (VAB, SET_SHADER_SCHEDULING, image address lowering, the
 * TXF sampler) are refused at screen creation. */
static void
nvrm_push_3d_init(const struct nvrm_screen *screen, struct nv_push *p)
{
   const struct nv_device_info *info = &screen->info;

   P_MTHD(p, NV9097, SET_OBJECT);
   P_NV9097_SET_OBJECT(p, {
      .class_id = info->cls_eng3d,
      .engine_id = 0,
   });

   /* G1: copy engine on subchannel 4 for texture transfers (NVK's
    * SUBC_NV90B5). Bound by SET_OBJECT as stock does (method 0 on
    * subchannel 4 with the copy class, as rm/src/gpu.rs's copy check does); the
    * copy headers name no SET_OBJECT. */
   __push_mthd(p, SUBC_NV90B5, 0x0000);
   P_INLINE_DATA(p, info->cls_copy);

   /* GLES 3.1: compute class on subchannel 1 (nvrm_compute.c) and the
    * indirect draw/dispatch MME macros (nvrm_mme.c). */
   nvrm_push_compute_init(screen, p);
   nvrm_push_mme_init(screen, p);

   if (info->cls_eng3d >= TURING_A && info->cls_eng3d < BLACKWELL_A)
      P_IMMD(p, NVC597, SET_MME_DATA_FIFO_CONFIG, FIFO_SIZE_SIZE_4KB);

   P_IMMD(p, NV9097, SET_TESSELLATION_PARAMETERS, {});
   P_IMMD(p, NV9097, SET_RENDER_ENABLE_C, MODE_TRUE);

   P_IMMD(p, NV9097, SET_Z_COMPRESSION, true);
   P_MTHD(p, NV9097, SET_COLOR_COMPRESSION(0));
   for (unsigned i = 0; i < 8; i++)
      P_NV9097_SET_COLOR_COMPRESSION(p, i, true);

   P_IMMD(p, NV9097, SET_CT_SELECT, { .target_count = 1 });
   P_IMMD(p, NV9097, SET_ALIASED_LINE_WIDTH_ENABLE, V_TRUE);
   P_IMMD(p, NV9097, SET_DA_PRIMITIVE_RESTART_VERTEX_ARRAY, ENABLE_FALSE);
   P_IMMD(p, NV9097, SET_BLEND_SEPARATE_FOR_ALPHA, ENABLE_TRUE);
   P_IMMD(p, NV9097, SET_SINGLE_CT_WRITE_CONTROL, ENABLE_TRUE);
   P_IMMD(p, NV9097, SET_SINGLE_ROP_CONTROL, ENABLE_FALSE);
   P_IMMD(p, NV9097, SET_TWO_SIDED_STENCIL_TEST, ENABLE_TRUE);
   P_IMMD(p, NV9097, SET_SHADE_MODE, V_OGL_SMOOTH);
   P_IMMD(p, NV9097, SET_API_VISIBLE_CALL_LIMIT, V__128);
   P_IMMD(p, NV9097, SET_ZCULL_STATS, ENABLE_TRUE);
   P_IMMD(p, NV9097, SET_L1_CONFIGURATION, DIRECTLY_ADDRESSABLE_MEMORY_SIZE_48KB);

   P_IMMD(p, NV9097, SET_REDUCE_COLOR_THRESHOLDS_ENABLE, V_FALSE);
   P_IMMD(p, NV9097, SET_REDUCE_COLOR_THRESHOLDS_UNORM8, {
      .all_covered_all_hit_once = 0xff,
   });
   P_MTHD(p, NV9097, SET_REDUCE_COLOR_THRESHOLDS_UNORM10);
   P_NV9097_SET_REDUCE_COLOR_THRESHOLDS_UNORM10(p, { .all_covered_all_hit_once = 0xff });
   P_NV9097_SET_REDUCE_COLOR_THRESHOLDS_UNORM16(p, { .all_covered_all_hit_once = 0xff });
   P_NV9097_SET_REDUCE_COLOR_THRESHOLDS_FP11(p, { .all_covered_all_hit_once = 0x3f });
   P_NV9097_SET_REDUCE_COLOR_THRESHOLDS_FP16(p, { .all_covered_all_hit_once = 0xff });
   P_NV9097_SET_REDUCE_COLOR_THRESHOLDS_SRGB8(p, { .all_covered_all_hit_once = 0xff });

   if (info->cls_eng3d < VOLTA_A) /* nvk_cmd_draw.c:318-319 */
      P_IMMD(p, NV9097, SET_ALPHA_FRACTION, 0x3f);

   P_IMMD(p, NV9097, CHECK_SPH_VERSION, { .current = 3, .oldest_supported = 3 });
   P_IMMD(p, NV9097, CHECK_AAM_VERSION, { .current = 2, .oldest_supported = 2 });

   P_IMMD(p, NV9097, SET_L2_CACHE_CONTROL_FOR_ROP_PREFETCH_READ_REQUESTS,
          POLICY_EVICT_NORMAL);
   P_IMMD(p, NV9097, SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_READ_REQUESTS,
          POLICY_EVICT_NORMAL);
   P_IMMD(p, NV9097, SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_READ_REQUESTS,
          POLICY_EVICT_NORMAL);
   P_IMMD(p, NV9097, SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_WRITE_REQUESTS,
          POLICY_EVICT_NORMAL);
   P_IMMD(p, NV9097, SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_WRITE_REQUESTS,
          POLICY_EVICT_NORMAL);

   P_IMMD(p, NV9097, SET_BLEND_PER_FORMAT_ENABLE, SNORM8_UNORM16_SNORM16_TRUE);

   P_IMMD(p, NV9097, SET_ATTRIBUTE_DEFAULT, {
      .color_front_diffuse    = COLOR_FRONT_DIFFUSE_VECTOR_0001,
      .color_front_specular   = COLOR_FRONT_SPECULAR_VECTOR_0001,
      .generic_vector         = GENERIC_VECTOR_VECTOR_0001,
      .fixed_fnc_texture      = FIXED_FNC_TEXTURE_VECTOR_0001,
      .dx9_color0             = DX9_COLOR0_VECTOR_0001,
      .dx9_color1_to_color15  = DX9_COLOR1_TO_COLOR15_VECTOR_0000,
   });

   P_IMMD(p, NV9097, SET_DA_OUTPUT, VERTEX_ID_USES_ARRAY_START_TRUE);
   P_IMMD(p, NV9097, SET_RENDER_ENABLE_CONTROL,
          CONDITIONAL_LOAD_CONSTANT_BUFFER_FALSE);
   P_IMMD(p, NV9097, SET_PS_OUTPUT_SAMPLE_MASK_USAGE, {
      .enable                       = ENABLE_TRUE,
      .qualify_by_anti_alias_enable = QUALIFY_BY_ANTI_ALIAS_ENABLE_ENABLE,
   });

   if (info->cls_eng3d < VOLTA_A) /* nvk_cmd_draw.c:365-366 */
      P_IMMD(p, NV9097, SET_PRIM_CIRCULAR_BUFFER_THROTTLE, 0x3fffff);

   P_IMMD(p, NV9097, SET_BLEND_OPT_CONTROL, ALLOW_FLOAT_PIXEL_KILLS_TRUE);
   P_IMMD(p, NV9097, SET_BLEND_FLOAT_OPTION, ZERO_TIMES_ANYTHING_IS_ZERO_TRUE);
   P_IMMD(p, NV9097, SET_BLEND_STATE_PER_TARGET, ENABLE_TRUE);

   P_IMMD(p, NV9097, SET_ALPHA_TEST, ENABLE_FALSE);
   P_IMMD(p, NV9097, SET_TWO_SIDED_LIGHT, ENABLE_FALSE);
   P_IMMD(p, NV9097, SET_COLOR_CLAMP, ENABLE_TRUE);
   P_IMMD(p, NV9097, SET_PS_SATURATE, {
      .output0 = OUTPUT0_FALSE, .output1 = OUTPUT1_FALSE,
      .output2 = OUTPUT2_FALSE, .output3 = OUTPUT3_FALSE,
      .output4 = OUTPUT4_FALSE, .output5 = OUTPUT5_FALSE,
      .output6 = OUTPUT6_FALSE, .output7 = OUTPUT7_FALSE,
   });

   P_IMMD(p, NV9097, SET_POINT_SIZE, fui(1.0));
   P_IMMD(p, NV9097, SET_ATTRIBUTE_POINT_SIZE, { .enable = ENABLE_TRUE });
   P_IMMD(p, NV9097, SET_POINT_SPRITE, ENABLE_TRUE);
   P_IMMD(p, NV9097, SET_POINT_SPRITE_SELECT, {
      .rmode = RMODE_ZERO, .origin = ORIGIN_TOP,
      .texture0 = TEXTURE0_PASSTHROUGH, .texture1 = TEXTURE1_PASSTHROUGH,
      .texture2 = TEXTURE2_PASSTHROUGH, .texture3 = TEXTURE3_PASSTHROUGH,
      .texture4 = TEXTURE4_PASSTHROUGH, .texture5 = TEXTURE5_PASSTHROUGH,
      .texture6 = TEXTURE6_PASSTHROUGH, .texture7 = TEXTURE7_PASSTHROUGH,
      .texture8 = TEXTURE8_PASSTHROUGH, .texture9 = TEXTURE9_PASSTHROUGH,
   });
   P_IMMD(p, NV9097, SET_ANTI_ALIASED_POINT, ENABLE_FALSE);
   if (info->cls_eng3d >= MAXWELL_B) /* nvk_cmd_draw.c:442-443 */
      P_IMMD(p, NVB197, SET_FILL_VIA_TRIANGLE, MODE_DISABLED);
   P_IMMD(p, NV9097, SET_POLY_SMOOTH, ENABLE_FALSE);
   P_IMMD(p, NV9097, SET_VIEWPORT_PIXEL, CENTER_AT_HALF_INTEGERS);
   P_IMMD(p, NV9097, SET_ANTI_ALIAS_ENABLE, V_TRUE);
   if (info->cls_eng3d >= MAXWELL_B) { /* nvk_cmd_draw.c:459-463 */
      P_IMMD(p, NVB197, SET_POST_PS_INITIAL_COVERAGE, true);
      P_IMMD(p, NVB197, SET_OFFSET_RENDER_TARGET_INDEX, BY_VIEWPORT_INDEX_FALSE);
   }

   P_IMMD(p, NV9097, SET_WINDOW_ORIGIN, {
      .mode = MODE_UPPER_LEFT, .flip_y = FLIP_Y_FALSE,
   });
   P_MTHD(p, NV9097, SET_WINDOW_OFFSET_X);
   P_NV9097_SET_WINDOW_OFFSET_X(p, 0);
   P_NV9097_SET_WINDOW_OFFSET_Y(p, 0);

   P_IMMD(p, NV9097, SET_ACTIVE_ZCULL_REGION, 0x3f);
   P_IMMD(p, NV9097, SET_WINDOW_CLIP_ENABLE, V_FALSE);
   P_IMMD(p, NV9097, SET_CLIP_ID_TEST, ENABLE_FALSE);
   P_IMMD(p, NV9097, SET_VIEWPORT_SCALE_OFFSET, ENABLE_TRUE);
   P_IMMD(p, NV9097, SET_VIEWPORT_CLIP_CONTROL, {
      .min_z_zero_max_z_one      = MIN_Z_ZERO_MAX_Z_ONE_FALSE,
      .pixel_min_z               = PIXEL_MIN_Z_CLAMP,
      .pixel_max_z               = PIXEL_MAX_Z_CLAMP,
      .geometry_guardband        = GEOMETRY_GUARDBAND_SCALE_256,
      .line_point_cull_guardband = LINE_POINT_CULL_GUARDBAND_SCALE_256,
      .geometry_clip             = GEOMETRY_CLIP_WZERO_CLIP,
      .geometry_guardband_z      = GEOMETRY_GUARDBAND_Z_SAME_AS_XY_GUARDBAND,
   });
   for (unsigned i = 0; i < 16; i++)
      P_IMMD(p, NV9097, SET_SCISSOR_ENABLE(i), V_FALSE);
   P_IMMD(p, NV9097, SET_CT_MRT_ENABLE, V_TRUE);

   if (info->cls_eng3d >= TURING_A) {
      /* Same values as NVK (copied there from the blob). */
      P_MTHD(p, NVC597, SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(0));
      P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 0, 0xa23eb139);
      P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 1, 0xfb72ea61);
      P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 2, 0xd950c843);
      P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 3, 0x88fac4e5);
      P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 4, 0x1ab3e1b6);
      P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 5, 0xa98fedc2);
      P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 6, 0x2107654b);
      P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 7, 0xe0539773);
      P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 8, 0x698badcf);
      P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 9, 0x71032547);
      P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 10, 0xdef05397);
      P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 11, 0x56789abc);
      P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 12, 0x1234);
   }

   /* nvk_cmd_draw.c:523-530: pre-Volta shader program addresses are 32-bit
    * offsets from SET_PROGRAM_REGION, NVK's contiguous shader heap base;
    * ours is the shader arena's reservation base (reserved before the channel). */
   if (info->cls_eng3d < VOLTA_A) {
      const uint64_t shader_base_addr = screen->shader_base;
      P_MTHD(p, NV9097, SET_PROGRAM_REGION_A);
      P_NV9097_SET_PROGRAM_REGION_A(p, shader_base_addr >> 32);
      P_NV9097_SET_PROGRAM_REGION_B(p, shader_base_addr);
   }

   P_IMMD(p, NV9097, SET_SHADER_LOCAL_MEMORY_WINDOW, 0xff << 24);

   for (uint32_t group = 0; group < 5; group++) {
      for (uint32_t slot = 0; slot < 16; slot++) {
         P_IMMD(p, NV9097, BIND_GROUP_CONSTANT_BUFFER(group), {
            .valid = VALID_FALSE,
            .shader_slot = slot,
         });
      }
   }

   P_IMMD(p, NV9097, SET_RT_LAYER, { .v = 0, .control = CONTROL_V_SELECTS_LAYER });
   P_IMMD(p, NV9097, SET_POINT_CENTER_MODE, V_OGL);
   P_IMMD(p, NV9097, SET_EDGE_FLAG, V_TRUE);
   /* Stock GL differs from stock Vulkan (and NVK) only here: GL binds
    * samplers via the texture header (stock's init emitter is otherwise
    * shared with Vulkan). */
   P_IMMD(p, NV9097, SET_SAMPLER_BINDING, V_VIA_HEADER_BINDING);

   const uint64_t zero_addr = screen->zero_page.va;
   P_MTHD(p, NV9097, SET_VERTEX_STREAM_SUBSTITUTE_A);
   P_NV9097_SET_VERTEX_STREAM_SUBSTITUTE_A(p, zero_addr >> 32);
   P_NV9097_SET_VERTEX_STREAM_SUBSTITUTE_B(p, zero_addr);

   if (info->cls_eng3d >= VOLTA_A) { /* nvk_cmd_draw.c:571-585 */
      P_IMMD(p, NV9097, SET_PS_WARP_WATERMARKS, {
         .low = 0x8,
         .high = info->max_warps_per_mp * info->mp_per_tpc,
      });
      P_IMMD(p, NV9097, SET_PS_REGISTER_WATERMARKS, { .low = 0x80, .high = 0x1000 });
   }

   for (uint32_t b = 0; b < 32; b++)
      P_IMMD(p, NV9097, SET_VERTEX_STREAM_A_FORMAT(b), { .enable = false });

   /* VAB (nvk_cmd_draw.c:594-603) is FERMI_A..<MAXWELL_A only: refused. */
   if (info->cls_eng3d == MAXWELL_A) /* nvk_cmd_draw.c:605-606 */
      P_IMMD(p, NVB097, SET_SELECT_MAXWELL_TEXTURE_HEADERS, V_TRUE);

   if (info->cls_eng3d >= HOPPER_A)
      P_IMMD(p, NVCB97, SET_TEXTURE_HEADER_VERSION, 1);

   P_IMMD(p, NV9097, SET_GLOBAL_BASE_INSTANCE_INDEX, 0);
}

/* ---------------------------------------------------------------- caps */

static void
nvrm_init_shader_caps(struct nvrm_screen *screen)
{
   for (unsigned i = 0; i <= MESA_SHADER_COMPUTE; i++) {
      struct pipe_shader_caps *caps =
         (struct pipe_shader_caps *)&screen->base.shader_caps[i];
      if (i != MESA_SHADER_VERTEX && i != MESA_SHADER_GEOMETRY &&
          i != MESA_SHADER_FRAGMENT && i != MESA_SHADER_COMPUTE)
         continue; /* no tessellation */
      caps->max_instructions = caps->max_alu_instructions =
      caps->max_tex_instructions = caps->max_tex_indirections = 16384;
      caps->max_control_flow_depth = 16384;
      /* 32 vec4 = NVK maxGeometryInput/OutputComponents 128
       * (nvk_physical_device.c:799-800) for the geometry stage too. */
      caps->max_inputs = i == MESA_SHADER_FRAGMENT ? 32 : 32;
      caps->max_outputs = 32;
      caps->max_const_buffer0_size = 65536;
      /* Mesa's ceilings (nvrm_priv.h NVRM_MAX_SSBOS/UBOS): 16 UBO blocks
       * + default, 16 SSBOs + 16 atomic counter buffers per stage; blocks
       * without a hardware slot go through the descriptor table. */
      caps->max_const_buffers = NVRM_MAX_UBOS;
      caps->max_shader_buffers = NVRM_MAX_SSBOS;
      caps->max_temps = 256;
      caps->cont_supported = true;
      caps->indirect_temp_addr = true;
      caps->indirect_const_addr = true;
      caps->integers = true;
      caps->max_texture_samplers = 16;
      caps->max_sampler_views = 16;
      /* GLES 3.1 part B: image units per stage (bindless storage headers,
       * nvrm_image.c); above the ES 3.1 minimums (4 compute, 0 fragment)
       * and stock GL's 8, as NVK's per-stage limit is far larger. */
      caps->max_shader_images = NVRM_MAX_IMAGES;
      caps->supported_irs = 1 << PIPE_SHADER_IR_NIR;
   }
}

static void
nvrm_init_screen_caps(struct nvrm_screen *screen)
{
   struct pipe_caps *caps = (struct pipe_caps *)&screen->base.caps;
   u_init_pipe_screen_caps(&screen->base, 1);

   caps->npot_textures = true;
   /* G1 GLES 3.0: what nvrm implements (MRT, xfb, queries, conditional
    * render, instancing, arrays/3D targets, UBOs). Not advertised: MSAA
    * (GLES 3.0 wants MAX_SAMPLES >= 4; nvrm has no multisample images yet),
    * timer queries, compute/images/SSBOs (G3). */
   caps->max_render_targets = NVRM_MAX_RTS;
   /* The sampler's border colour is returned without the texture header's
    * swizzle (NVK applies the view swizzle itself, nvk_sampler.c:250-283,
    * VK_EXT_border_color_swizzle): st pre-swizzles it per unit with the
    * texture's swizzle and base format, as for nvc0 (nvc0_screen.c:228).
    * nvrm pairs header slot i with sampler slot i per unit, so the unit's
    * view is the one st swizzles for. */
   caps->texture_border_color_quirk = PIPE_QUIRK_TEXTURE_BORDER_COLOR_SWIZZLE_NV50;
   caps->indep_blend_enable = true;
   caps->indep_blend_func = true;
   caps->mixed_framebuffer_sizes = true;
   caps->mixed_color_depth_bits = true;
   caps->mixed_colorbuffer_formats = true;
   caps->vs_instanceid = true;
   caps->vertex_element_instance_divisor = true;
   caps->fragment_shader_texture_lod = true;
   caps->fragment_shader_derivatives = true;
   caps->texture_float_linear = true;
   caps->texture_half_float_linear = true;
   caps->primitive_restart = true;
   caps->primitive_restart_fixed_index = true;
   caps->max_texture_array_layers = 2048;
   caps->max_stream_output_buffers = NVRM_MAX_SO;
   caps->max_stream_output_separate_components = 128;
   caps->max_stream_output_interleaved_components = 128;
   caps->stream_output_pause_resume = true;
   caps->stream_output_interleave_buffers = true;
   caps->occlusion_query = true;
   caps->conditional_render = true;
   caps->seamless_cube_map = true;
   caps->texture_swizzle = true;
   caps->texture_shadow_map = true;
   caps->dest_surface_srgb_control = true;
   caps->max_constant_buffer_size = 65536;
   /* u_blitter resolves sample a multisampled view with txf_ms. No
    * shader_stencil_export: NAK refuses FRAG_RESULT_STENCIL
    * (nak_nir.c nak_fs_out_addr), stencil blits take u_blitter's
    * per-bit stencil fallback. */
   caps->texture_multisample = true;
   caps->max_texture_2d_size = 16384;
   caps->max_texture_3d_levels = 12;
   caps->max_texture_cube_levels = 15;
   caps->blend_equation_separate = true;
   caps->clear_scissored = true; /* nvrm_clear honours the scissor rect */
   /* NAK takes gl_FragCoord / gl_PointCoord / gl_FrontFacing only as system
    * values (load_frag_coord, load_point_coord, load_front_face), as NVK
    * hands them; as FS input varyings they hit nak_varying_attr_addr
    * ("Invalid varying slot") or, for FACE, an invalid shader program
    * header (Xid 13 "Shader Program Header 9 Error" on dEQP
    * shaders.builtin_variable.frontfacing, 10-15 09:12 and 09:15). */
   caps->fs_position_is_sysval = true;
   caps->fs_point_is_sysval = true;
   caps->fs_face_is_integer_sysval = true;
   caps->fs_coord_origin_upper_left = true;
   caps->fs_coord_pixel_center_half_integer = true;
   /* ---- GLES 3.1 part B (images, gather, cube arrays, no-attachment FBOs);
    * values as NVK's limits (nvk_physical_device.c). */
   caps->image_store_formatted = true;  /* shaderStorageImageWriteWithoutFormat */
   caps->image_load_formatted = true;   /* ...ReadWithoutFormat: MAXWELL_A+ = every nvrm class */
   caps->nir_images_as_deref = false;   /* st/mesa hands image_* on unit indices */
   caps->max_texture_gather_components = 4;
   caps->min_texture_gather_offset = -32; /* minTexelGatherOffset */
   caps->max_texture_gather_offset = 31;
   caps->texture_gather_offsets = true; /* NAK TLD4 per-pixel offsets (nak_nir_lower_tex.c:166-175) */
   caps->cube_map_array = true;         /* imageCubeArray; nil NIL_VIEW_TYPE_CUBE_ARRAY */
   /* GL_NV_polygon_mode on ES (glPolygonMode{,NV}, POINT/LINE fill modes,
    * point/line offset): nvrm_emit_rasterizer as nvk_flush_rs_state. */
   caps->polygon_mode_es = true;
   caps->framebuffer_no_attachment = true; /* framebufferNoAttachmentsSampleCounts */
   /* ARB/EXT_clip_control and ARB/EXT_depth_clamp (st_extensions.c:1030,
    * 1036; extensions_table.h:230,242 alias them on ES2+). NVK exposes
    * depthClamp, depthClipEnable and EXT_depth_clip_control on every class
    * (nvk_physical_device.c:221,328,560,563); nvrm emits the same words from
    * the rasterizer: SET_VIEWPORT_Z_CLIP from clip_halfz (nvk_flush_vp_state),
    * SET_VIEWPORT_CLIP_CONTROL from depth_clip_near/far and, pre-Volta, the
    * VIEWPORT_CLIP_MIN/MAX_Z emulation of NVK_MME_SET_Z_CLAMP
    * (nvrm_emit_viewport / nvrm_emit_rasterizer). GL clamps the depth range
    * and the fragment depth to [0,1] for float depth too (dEQP-GLES3
    * fbo.depth.depth_write_clamp.depth_component32f), i.e. NVK's
    * z_clamp_zero_one (depthClampZeroOne without depth_range_unrestricted)
    * always holds. */
   caps->clip_halfz = true;
   caps->depth_clip_disable = true;
   /* OES/EXT_copy_image (st_extensions.c:1034): resource_copy_region is
    * NVK's nouveau_copy_rect on the copy engine (nvrm_copy.c) in element
    * units, so compressed <-> plain copies of one block size are exact. */
   caps->copy_between_compressed_and_plain_formats = true;
   /* ---- end GLES 3.1 part B */
   /* Geometry shaders (OES/EXT_geometry_shader, which st/mesa also requires
    * for OES/EXT_texture_cube_map_array, st_extensions.c:1758-1761):
    * NVK's limits (nvk_physical_device.c:798-802). */
   caps->max_geometry_output_vertices = 1024;
   caps->max_geometry_total_output_components = 1024;
   caps->max_gs_invocations = 32;
   caps->glsl_feature_level = 130;
   caps->glsl_feature_level_compatibility = 120;
   /* ESSL 3.10: the context reaches ES 3.1 on every nvrm class (indirect
    * draws/dispatches through NVK's MME macros: Turing+ DMA reads, older
    * classes push-indirect data, nvrm_mme.c). st/mesa gates
    * OES_geometry_shader and the cube-map-array/viewport-array ES
    * extensions on it (st_extensions.c:1340, 1758-1766) and forces
    * ARB_draw_indirect on ES with it (:1351). */
   caps->essl_feature_level = 310;
   caps->constant_buffer_offset_alignment = 256;
   caps->min_map_buffer_alignment = 64;
   caps->max_vertex_attrib_stride = 2048;
   caps->max_varyings = 32;
   caps->endianness = PIPE_ENDIAN_LITTLE;
   caps->texture_transfer_modes = 0;
   caps->vendor_id = screen->rm_info.vendor_id;
   caps->device_id = screen->rm_info.device_id;
   caps->video_memory = screen->rm_info.vram_size >> 20;
   caps->uma = false;
   caps->pci_group = screen->rm_info.pci_domain;
   caps->pci_bus = screen->rm_info.pci_bus;
   caps->pci_device = screen->rm_info.pci_device;
   caps->pci_function = screen->rm_info.pci_function;
   caps->min_line_width = caps->min_line_width_aa = 1;
   caps->min_point_size = caps->min_point_size_aa = 1;
   caps->point_size_granularity = caps->line_width_granularity = 0.1;
   caps->max_line_width = caps->max_line_width_aa = 10.0;
   /* The hardware clamps program point sizes itself; NVK reports that
    * range (nvk_physical_device.c:854 pointSizeRange 2047.94). GL requires
    * sizes beyond the advertised maximum to clamp to it (dEQP
    * rasterization.limits.points failed with 63). */
   caps->max_point_size = caps->max_point_size_aa = 2047.94f;
   caps->max_texture_anisotropy = 16.0;
   caps->max_texture_lod_bias = 15.0;
   /* G2: sync_file fences (EGL_ANDROID_native_fence_sync); caps->dmabuf
    * stays DRM_CAP_PRIME from u_init_pipe_screen_caps: import/export are
    * implemented (nvrm_handle.c). */
   caps->native_fence_fd = true;

   /* GLES 3.1 part A: compute + storage buffers (nvrm_compute.c), limits as
    * NVK's (nvk_physical_device.c: maxComputeWorkGroupCount/Size/
    * Invocations, maxComputeSharedMemorySize = NVK_MAX_SHARED_SIZE,
    * minStorageBufferOffsetAlignment 16). */
   caps->compute = true;
   /* Indirect draws through NVK's MME macros on every class (nvrm_mme.c;
    * NVK: multiDrawIndirect on all classes, nvk_physical_device.c:326). */
   caps->draw_indirect = true;
   caps->multi_draw_indirect = true;
   caps->shader_buffer_offset_alignment = 16;
   caps->max_shader_buffer_size = 1u << 30;
   struct pipe_compute_caps *cc = (struct pipe_compute_caps *)&screen->base.compute_caps;
   cc->address_bits = 64;
   cc->grid_dimension = 3;
   cc->max_grid_size[0] = 0x7fffffff;
   cc->max_grid_size[1] = 65535;
   cc->max_grid_size[2] = 65535;
   cc->max_block_size[0] = 1024;
   cc->max_block_size[1] = 1024;
   cc->max_block_size[2] = 64;
   cc->max_threads_per_block = 1024;
   cc->max_local_size = NVRM_MAX_SHARED_SIZE;
   cc->max_variable_threads_per_block = 0;
   cc->subgroup_sizes = 32;
   cc->max_subgroups = 1024 / 32;
   cc->max_compute_units = screen->info.tpc_count * screen->info.mp_per_tpc;
   cc->max_global_size = screen->rm_info.vram_size;
   cc->max_mem_alloc_size = screen->rm_info.vram_size;
}

static const char *
nvrm_get_name(struct pipe_screen *pscreen)
{
   return nvrm_screen(pscreen)->info.device_name;
}

static const char *
nvrm_get_vendor(struct pipe_screen *pscreen)
{
   return "nvidia-prolapse (nvrm)";
}

static const char *
nvrm_get_device_vendor(struct pipe_screen *pscreen)
{
   return "NVIDIA";
}

static int
nvrm_get_screen_fd(struct pipe_screen *pscreen)
{
   return nvrm_screen(pscreen)->fd;
}

static bool
nvrm_is_format_supported(struct pipe_screen *pscreen, enum pipe_format format,
                         enum pipe_texture_target target, unsigned sample_count,
                         unsigned storage_sample_count, unsigned bindings)
{
   struct nvrm_screen *screen = nvrm_screen(pscreen);
   /* MSAA as NVK's sample counts (1/2/4/8) for 2D colour and depth/stencil
    * images; multisampled textures are only sampled by resolves. */
   if (sample_count > 1 || storage_sample_count > 1) {
      if (sample_count != MAX2(storage_sample_count, 1) && storage_sample_count)
         return false;
      if (!util_is_power_of_two_nonzero(sample_count) || sample_count > 8)
         return false;
      if (target != PIPE_TEXTURE_2D && target != PIPE_TEXTURE_2D_ARRAY)
         return false;
      if (bindings & ~(PIPE_BIND_RENDER_TARGET | PIPE_BIND_DEPTH_STENCIL |
                       PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_BLENDABLE |
                       PIPE_BIND_DISPLAY_TARGET))
         return false;
   }
   /* framebuffer_no_attachment: st/mesa asks for the attachment-less
    * sample counts with PIPE_FORMAT_NONE (st_extensions.c void_formats). */
   if (format == PIPE_FORMAT_NONE)
      return !(bindings & ~PIPE_BIND_RENDER_TARGET) && target != PIPE_BUFFER;
   if (target == PIPE_BUFFER) {
      /* Vertex formats: exactly what the draw path can program, so st's
       * u_vbuf translates the rest (GL_FIXED, packed, BGRA...). */
      struct nvrm_va_format va;
      if ((bindings & PIPE_BIND_VERTEX_BUFFER) && !nvrm_get_va_format(format, &va))
         return false;
      return !(bindings & ~(PIPE_BIND_VERTEX_BUFFER | PIPE_BIND_INDEX_BUFFER |
                            PIPE_BIND_CONSTANT_BUFFER | PIPE_BIND_SHADER_BUFFER)) ||
             nil_format_supports_buffer(&screen->info, format);
   }
   /* 96-bit RGB32 textures: nil samples R32G32B32_{FLOAT,SINT,UINT} but
    * cannot render them (nil_formats.csv "TB"), while st/mesa asks GL_RGB32F
    * for a render target first (st_ChooseTextureFormat) and so stores it as
    * R32G32B32X32_FLOAT, but GL_RGB32I/UI without one as the 96-bit format.
    * Those are one copy_image view class (dEQP-GLES31 copy_image
    * viewclass_96_bits rgb32f_rgb32i: st swizzled_copy asserts on the block
    * size mismatch). Textures therefore always take the 128-bit X32 layout
    * (nil "CT"/"CTA", same sampling); RGB32 stays for buffers (texel buffers,
    * vertex fetch), where nil supports it. */
   if ((format == PIPE_FORMAT_R32G32B32_FLOAT || format == PIPE_FORMAT_R32G32B32_SINT ||
        format == PIPE_FORMAT_R32G32B32_UINT))
      return false;
   if ((bindings & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_DISPLAY_TARGET |
                    PIPE_BIND_SCANOUT | PIPE_BIND_SHARED)) &&
       !nil_format_supports_color_targets(&screen->info, format))
      return false;
   if ((bindings & PIPE_BIND_BLENDABLE) &&
       !nil_format_supports_blending(&screen->info, format))
      return false;
   if ((bindings & PIPE_BIND_SAMPLER_VIEW) &&
       !nil_format_supports_texturing(&screen->info, format))
      return false;
   if ((bindings & PIPE_BIND_DEPTH_STENCIL) &&
       !nil_format_supports_depth_stencil(&screen->info, format))
      return false;
   /* Stencil-only S8_UINT targets on Maxwell and GP100: nil's PTE kind 0x2a is
    * refused by RM's VA mapping (GTX TITAN X GM200, 580.173.02 closed KM,
    * kindOverride path: "map GPU VA: RM status 0x1f", rental
    * 20261015-nvrm-prevolta), so S8 is not offered there and st/mesa picks
    * a combined Z24S8 format for GL_STENCIL_INDEX8 instead. GP100
    * (PASCAL_A, Tesla P100 580.178.04) fails the same 181 fbo cases with
    * S8 offered; PASCAL_B GP102/GP104 map the kind (GTX 1080 Ti 580.173.02,
    * Quadro P4000 580.178.04: fbo identical to the 5090). */
   if ((bindings & PIPE_BIND_DEPTH_STENCIL) && format == PIPE_FORMAT_S8_UINT &&
       screen->info.cls_eng3d < PASCAL_B)
      return false;
   /* GLES 3.1 part B: storage images as nil allows (single-sample only). */
   if ((bindings & PIPE_BIND_SHADER_IMAGE) &&
       (sample_count > 1 || !nil_format_supports_storage(&screen->info, format)))
      return false;
   if (bindings & PIPE_BIND_SHADER_BUFFER)
      return false;
   return true;
}

/* ---------------------------------------------------------------- fences */

static void
nvrm_fence_reference(struct pipe_screen *pscreen, struct pipe_fence_handle **ptr,
                     struct pipe_fence_handle *fence)
{
   struct nvrm_fence *old = (struct nvrm_fence *)*ptr;
   struct nvrm_fence *f = (struct nvrm_fence *)fence;
   if (pipe_reference(old ? &old->reference : NULL, f ? &f->reference : NULL)) {
      if (old->fd >= 0)
         close(old->fd);
      FREE(old);
   }
   *ptr = fence;
}

static bool
nvrm_fence_finish(struct pipe_screen *pscreen, struct pipe_context *ctx,
                  struct pipe_fence_handle *fence, uint64_t timeout)
{
   struct nvrm_screen *screen = nvrm_screen(pscreen);
   struct nvrm_fence *f = (struct nvrm_fence *)fence;
   if (f->fd >= 0) {
      /* Imported sync_file (EGL_ANDROID_native_fence_sync, client wait). */
      const int ms = timeout == OS_TIMEOUT_INFINITE ? -1 :
                     (int)MIN2(DIV_ROUND_UP(timeout, 1000000), INT32_MAX);
      if (sync_wait(f->fd, ms) == 0)
         return true;
      if (errno != ETIME)
         nvrm_error("fence wait on sync_file failed: errno %d\n", errno);
      return false;
   }
   if (!f->token)
      return true;
   if (screen->debug_stats && !nvrm_token_done(screen, f->token))
      p_atomic_inc(&screen->stat_cpu_waits);
   int32_t status = nv_rm_wait(screen->rm, screen->channel, f->token, timeout);
   if (status < 0)
      nvrm_error("fence wait failed: %d\n", status);
   return status == 0;
}

/* EGL_ANDROID_native_fence_sync export: a sync_file for the fence's
 * submission token (nv_rm_context_fence = the semaphore-surface sync_file
 * NVK's SYNC_FD export uses; stock GL exports the same kind, gl-sync RE).
 * Imported fences hand back a duplicate of their own sync_file. */
static int
nvrm_fence_get_fd(struct pipe_screen *pscreen, struct pipe_fence_handle *fence)
{
   struct nvrm_screen *screen = nvrm_screen(pscreen);
   struct nvrm_fence *f = (struct nvrm_fence *)fence;
   if (f->fd >= 0)
      return os_dupfd_cloexec(f->fd);
   if (!f->token) {
      nvrm_error("fence_get_fd: fence has no submission (deferred flush?)\n");
      return -1;
   }
   int fd = -1;
   int32_t status = nv_rm_context_fence(screen->rm, screen->channel, f->token, screen->fd, &fd);
   if (status || fd < 0) {
      nvrm_error("fence_get_fd: nv_rm_context_fence(token %" PRIu64 ") failed: %d\n",
                 f->token, status);
      return -1;
   }
   return fd;
}

void
nvrm_init_fence_functions(struct pipe_screen *pscreen)
{
   pscreen->fence_get_fd = nvrm_fence_get_fd;
}

/* ---------------------------------------------------------------- screen */

static struct disk_cache *
nvrm_get_disk_shader_cache(struct pipe_screen *pscreen)
{
   return nvrm_screen(pscreen)->disk_cache;
}

/* KHR_parallel_shader_compile (main/hint.c:136-145): the worker count of
 * the compile queue, as iris_set_max_shader_compiler_threads
 * (iris_program.c:3961-3968). */
static void
nvrm_set_max_shader_compiler_threads(struct pipe_screen *pscreen, unsigned max_threads)
{
   util_queue_adjust_num_threads(&nvrm_screen(pscreen)->compile_queue, max_threads, false);
}

/* GL_COMPLETION_STATUS (main/shaderapi.c): non-blocking. */
static bool
nvrm_is_parallel_shader_compilation_finished(struct pipe_screen *pscreen, void *shader,
                                             mesa_shader_stage stage)
{
   return util_queue_fence_is_signalled(&((struct nvrm_shader *)shader)->ready);
}

static void
nvrm_screen_destroy(struct pipe_screen *pscreen)
{
   struct nvrm_screen *screen = nvrm_screen(pscreen);
   /* Compile workers first: a job may still be using the channel (shader
    * upload), the shader arena, the program cache and the NAK compiler
    * (iris_screen_destroy order, iris_screen.c:524-530). Contexts deleted
    * their shaders (dropping their jobs) before the screen goes. */
   if (util_queue_is_initialized(&screen->compile_queue))
      util_queue_destroy(&screen->compile_queue);
   /* Recycled buffer memory goes back while the device is open. */
   nvrm_bo_cache_fini(screen);
   /* Pending cache writes finish here (disk_cache_destroy drains its
    * queue); every shader is gone, so the in-memory table goes too. */
   disk_cache_destroy(screen->disk_cache);
   nvrm_shader_cache_fini(screen);
   if (screen->channel)
      nv_rm_context_destroy(screen->rm, screen->channel);
   nvrm_bo_free(screen, &screen->zero_page);
   if (screen->shader_base) {
      util_vma_heap_finish(&screen->shader_vma);
      util_dynarray_fini(&screen->shader_pending);
      /* Each memory's VA goes back through nvrm_va_release; the unbacked
       * rest of the reservation directly. */
      for (unsigned i = 0; i < screen->shader_mem_count; i++)
         nvrm_bo_free(screen, &screen->shader_mem[i]);
      const uint64_t max = 1ull << NVRM_SHADER_ARENA_MAX_SIZE_LOG2;
      simple_mtx_lock(&screen->va_mutex);
      util_vma_heap_free(&screen->heap, screen->shader_base + screen->shader_committed,
                         max - screen->shader_committed);
      simple_mtx_unlock(&screen->va_mutex);
   }
   if (screen->nak)
      nak_compiler_destroy(screen->nak);
   if (screen->rm)
      nv_rm_close(screen->rm);
   simple_mtx_destroy(&screen->bo_cache_mutex);
   util_vma_heap_finish(&screen->heap);
   simple_mtx_destroy(&screen->va_mutex);
   simple_mtx_destroy(&screen->submit_mutex);
   simple_mtx_destroy(&screen->shader_mutex);
   close(screen->fd);
   if (screen->glsl_types_ref)
      glsl_type_singleton_decref();
   FREE(screen);
}

static bool
nvrm_screen_init_channel(struct nvrm_screen *screen)
{
   /* Engines as NVK's graphics queue family (nvk_queue.c:437-448: 3D,
    * COMPUTE, and COPY for its TRANSFER bit). */
   int32_t status = nv_rm_context_create(screen->rm,
                                         NV_RM_ENGINE_3D | NV_RM_ENGINE_COMPUTE |
                                         NV_RM_ENGINE_COPY,
                                         &screen->channel);
   if (status) {
      nvrm_error("nv_rm_context_create(3D|COMPUTE|COPY) failed: %d\n", status);
      screen->channel = 0;
      return false;
   }
   /* Zero page in VRAM, as NVK's dev->zero_page (nvk_device.c:192-200,
    * NVKMD_MEM_LOCAL); zeroed once through BAR1 (4 KiB of the window). */
   if (!nvrm_bo_alloc(screen, 4096, NVRM_DOMAIN_VRAM_MAPPED, &screen->zero_page))
      return false;
   memset(screen->zero_page.map, 0, 4096);

   struct nvrm_bo push_bo;
   if (!nvrm_bo_alloc(screen, 64 * 1024, NVRM_DOMAIN_GART, &push_bo))
      return false;
   struct nv_push push;
   nv_push_init(&push, push_bo.map, 64 * 1024 / 4, SUBC_MASK_ALL);
   nvrm_push_3d_init(screen, &push);
   const uint32_t size_B = nv_push_dw_count(&push) * 4;
   bool ok = nvrm_submit_wait(screen, push_bo.va, size_B, NULL);
   nvrm_bo_free(screen, &push_bo);
   if (!ok)
      nvrm_error("3D state init submission failed\n");
   return ok;
}

struct pipe_screen *
nvrm_screen_create(int fd, const struct pipe_screen_config *config)
{
   drmVersionPtr version = drmGetVersion(fd);
   bool native = version && version->name && strcmp(version->name, "nvidia-drm") == 0;
   if (!native) {
      nvrm_error("fd's kernel driver is '%s', not nvidia-drm: refusing\n",
                 version && version->name ? version->name : "(unknown)");
      if (version)
         drmFreeVersion(version);
      return NULL;
   }
   drmFreeVersion(version);

   drmDevicePtr drm = NULL;
   if (drmGetDevice2(fd, 0, &drm) || !drm || drm->bustype != DRM_BUS_PCI) {
      nvrm_error("drmGetDevice2 failed or not a PCI device\n");
      if (drm)
         drmFreeDevice(&drm);
      return NULL;
   }

   struct nvrm_screen *screen = CALLOC_STRUCT(nvrm_screen);
   if (!screen) {
      drmFreeDevice(&drm);
      return NULL;
   }
   screen->fd = os_dupfd_cloexec(fd);
   simple_mtx_init(&screen->va_mutex, mtx_plain);
   simple_mtx_init(&screen->submit_mutex, mtx_plain);
   simple_mtx_init(&screen->shader_mutex, mtx_plain);
   simple_mtx_init(&screen->bo_cache_mutex, mtx_plain);
   util_dynarray_init(&screen->bo_cache, NULL);
   util_vma_heap_init(&screen->heap, NV_RM_VA_START, NV_RM_VA_END - NV_RM_VA_START);

   int32_t status = nv_rm_open(drm->businfo.pci->domain, drm->businfo.pci->bus,
                               drm->businfo.pci->dev, drm->businfo.pci->func,
                               &screen->rm, &screen->rm_info);
   drmFreeDevice(&drm);
   if (status) {
      nvrm_error("nv_rm_open failed: %d\n", status);
      goto fail;
   }
   const struct NvRmInfo *ri = &screen->rm_info;
   /* NVK's full-BAR test (nvk_device.c:256-258); NVRM_FORCE_SMALL_BAR=1
    * takes the small-BAR (copy-engine upload) paths on a full-BAR GPU, for
    * testing them on the host. */
   screen->bar_full = ri->bar_size >= ri->vram_size &&
                      !debug_get_bool_option("NVRM_FORCE_SMALL_BAR", false);
   screen->debug_stats = debug_get_bool_option("NVRM_DEBUG_STATS", false);
   /* Same nv_device_info as nvkmd_rm_try_create_pdev (nvkmd_rm.c). */
   screen->info = (struct nv_device_info){
      .type = NV_DEVICE_TYPE_DIS,
      .device_id = ri->device_id,
      .chipset = ri->chipset,
      .sm = ri->sm,
      .pci = { .domain = ri->pci_domain, .bus = ri->pci_bus,
               .dev = ri->pci_device, .func = ri->pci_function },
      .gpc_count = ri->gpc_count,
      .tpc_count = ri->tpc_count,
      .mp_per_tpc = ri->mp_per_tpc,
      .max_warps_per_mp = ri->max_warps_per_mp,
      .cls_copy = ri->cls_copy,
      .cls_eng2d = ri->cls_eng2d,
      .cls_eng3d = ri->cls_eng3d,
      .cls_m2mf = ri->cls_m2mf,
      .cls_compute = ri->cls_compute,
      .vram_size_B = ri->vram_size,
      .bar_size_B = ri->bar_size,
   };
   snprintf(screen->info.device_name, sizeof(screen->info.device_name), "%s",
            ri->name);
   snprintf(screen->info.chipset_name, sizeof(screen->info.chipset_name),
            "NV%03X", ri->chipset);
   /* As nvkmd_rm.c / the nouveau winsys (nouveau_device.c): shared memory
    * configurations from the SM version, non-coherent atom = cache line. */
   nv_device_info_init_shared_mem_sizes(&screen->info);
   screen->info.nc_atom_size_B = util_cache_granularity();

   /* The 3D class comes from RM exactly as NVK's (NvRmInfo.cls_eng3d).
    * nvrm ports NVK's paths for MAXWELL_A (0xb097) through BLACKWELL_B.
    * Kepler (and older) would need NVK's VAB memory, SET_SHADER_SCHEDULING,
    * SET_MAX_TI_WARPS_PER_BATCH, Kepler texture-instruction ordering, image
    * address lowering and the reserved TXF sampler: not ported. */
   if (ri->cls_eng3d < MAXWELL_A || ri->cls_eng3d > BLACKWELL_B) {
      nvrm_error("3D class 0x%04x (%s, chipset 0x%x) is not supported: nvrm "
                 "implements MAXWELL_A (0xb097) through BLACKWELL_B (0xce97)\n",
                 ri->cls_eng3d, ri->name, ri->chipset);
      goto fail;
   }

   screen->nak = nak_compiler_create(&screen->info);
   if (!screen->nak) {
      nvrm_error("nak_compiler_create failed for SM%u\n", ri->sm);
      goto fail;
   }

   screen->nir_options = *nak_nir_options(screen->nak);
   screen->nir_options.lower_uniforms_to_ubo = true;
   /* st/mesa sees no doubles lowering; the NIR handed to NAK carries NAK's
    * own options again (nvrm_compile_shader), which is NVK's contract
    * (nvk_get_nir_options = nak_nir_options, nvk_shader.c:86-94) and lets
    * NAK lower doubles and int64 itself (nak_nir.c:1766-1767). With
    * lower_doubles_options set, st_glsl_to_nir.cpp:290-347 scalarizes
    * every phi (nir_lower_phis_to_scalar is not limited to 64-bit ones),
    * re-vectorizes and runs a whole gl_nir_opts loop on every program
    * although nvrm exposes no fp64 or int64 (no doubles/int64 caps):
    * measured -3.2 ms per Minecraft program on the host. */
   screen->nir_options.lower_doubles_options = 0;

   /* Shader heap: NVK's contiguous shader arena (nvk_device.c:257-260,
    * nvk_mem_arena.c): reserve the whole 4 GiB VA range now (its base is the
    * pre-Volta SET_PROGRAM_REGION, programmed by the channel init), back it
    * on demand with video memory that is never CPU-mapped (uploads:
    * nvrm_shader_heap_upload). The first memory is allocated here so a
    * broken RM fails at screen creation. */
   {
      const uint64_t max = 1ull << NVRM_SHADER_ARENA_MAX_SIZE_LOG2;
      simple_mtx_lock(&screen->va_mutex);
      screen->shader_base = util_vma_heap_alloc(&screen->heap, max, 1ull << 16);
      simple_mtx_unlock(&screen->va_mutex);
      if (!screen->shader_base) {
         nvrm_error("VA heap exhausted for the 4 GiB shader arena reservation\n");
         goto fail;
      }
      util_vma_heap_init(&screen->shader_vma, screen->shader_base, 0);
      util_dynarray_init(&screen->shader_pending, NULL);
      if (!nvrm_shader_heap_grow(screen))
         goto fail;
   }

   if (!nvrm_screen_init_channel(screen))
      goto fail;

   /* Compile queue (nvrm_compile_shader). Workers use NIR, so they hold
    * their own reference on the GLSL type singleton: libGL's reference is
    * dropped at exit before u_queue's atexit handler stops the workers
    * (iris_screen.c:814, si_pipe.c). Thread count as iris
    * (iris_screen.c:820-831): 3/4 of the CPUs from 12 up; the queue starts
    * with one worker and adds them on demand. Normal priority: draws wait
    * for these jobs. Default stack: the compile path needs < 80 KiB
    * (measured); bionic's default thread stack is 1 MiB - 32 KiB. */
   glsl_type_singleton_init_or_ref();
   screen->glsl_types_ref = true;
   {
      const unsigned cpus = util_get_cpu_caps()->nr_cpus;
      unsigned threads = 1;
      if (cpus >= 12)
         threads = cpus * 3 / 4;
      else if (cpus >= 6)
         threads = cpus - 2;
      else if (cpus >= 2)
         threads = cpus - 1;
      if (!util_queue_init(&screen->compile_queue, "nvrmsh", 64, threads,
                           UTIL_QUEUE_INIT_RESIZE_IF_FULL |
                           UTIL_QUEUE_INIT_SET_FULL_THREAD_AFFINITY, NULL)) {
         nvrm_error("could not create the shader compile queue (%u threads)\n", threads);
         goto fail;
      }
   }

   struct pipe_screen *ps = &screen->base;
   ps->destroy = nvrm_screen_destroy;
   ps->get_name = nvrm_get_name;
   ps->get_vendor = nvrm_get_vendor;
   ps->get_device_vendor = nvrm_get_device_vendor;
   ps->get_screen_fd = nvrm_get_screen_fd;
   ps->get_timestamp = u_default_get_timestamp;
   ps->is_format_supported = nvrm_is_format_supported;
   /* Mesa 26: per-stage compiler options live in pipe_screen::nir_options
    * (get_compiler_options was removed); one NAK option set for every stage. */
   for (unsigned i = 0; i < MESA_SHADER_MESH_STAGES; i++)
      ps->nir_options[i] = &screen->nir_options;
   ps->get_disk_shader_cache = nvrm_get_disk_shader_cache;
   ps->set_max_shader_compiler_threads = nvrm_set_max_shader_compiler_threads;
   ps->is_parallel_shader_compilation_finished = nvrm_is_parallel_shader_compilation_finished;
   ps->context_create = nvrm_context_create;
   ps->fence_reference = nvrm_fence_reference;
   ps->fence_finish = nvrm_fence_finish;
   nvrm_init_resource_functions(ps);
   nvrm_init_handle_functions(ps);
   nvrm_init_fence_functions(ps);
   nvrm_init_shader_caps(screen);
   nvrm_init_screen_caps(screen);
   nvrm_shader_cache_init(screen);

   mesa_logi("nvrm: %s (chipset 0x%x, 3D class 0x%04x, SM%u) on channel %" PRIu64
             ", VRAM %" PRIu64 " MiB, BAR1 %" PRIu64 " MiB (%s BAR)\n",
             ri->name, ri->chipset, ri->cls_eng3d, ri->sm, screen->channel,
             ri->vram_size >> 20, ri->bar_size >> 20, screen->bar_full ? "full" : "small");
   return ps;

fail:
   nvrm_screen_destroy(&screen->base);
   return NULL;
}
