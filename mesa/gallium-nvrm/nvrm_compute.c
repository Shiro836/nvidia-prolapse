/* SPDX-License-Identifier: MIT
 *
 * nvrm GLES 3.1 part A: compute launches, shader storage buffers (incl.
 * st/mesa's atomic counters lowered to SSBOs, max_hw_atomic_counters = 0)
 * and memory barriers.
 *
 * Hardware programming follows NVK at the pinned Mesa:
 *  - compute class init: nvk_push_dispatch_state_init (nvk_cmd_dispatch.c:
 *    31-88), compute SLM: nvk_queue_state_update (nvk_queue.c:131-155);
 *  - QMD: nvk_cmd_upload_qmd (nvk_cmd_dispatch.c:139-241) through NAK's
 *    nak_fill_qmd, which picks the QMD version of the compute class
 *    (pre-Volta included); launch: nvk_cmd_dispatch_shader
 *    (nvk_cmd_dispatch.c:333-377) SEND_PCAS_A + SEND_SIGNALING_PCAS(2)_B;
 *  - barriers: nvk_cmd_flush_wait_dep / nvk_cmd_invalidate_deps
 *    (nvk_cmd_buffer.c:515-640);
 *  - buffer access: NVK keeps SSBO addresses/sizes in its root descriptor
 *    cbuf and lowers SSBO access to bounded 64-bit global access
 *    (nvk_shader.c:425-430, nir_address_format_64bit_bounded_global): out of
 *    bounds loads return 0, stores and atomics are dropped. st/mesa hands
 *    gallium drivers index/offset intrinsics (load_ssbo, ...), lowered here
 *    with the same semantics.
 * Compute texture/image handles: part B's table (nvrm_upload_tex_handles)
 * as QMD cbuf NVRM_CS_TEX_CBUF; the pools are the context's, bound on the
 * compute class per dispatch. Indirect dispatch: NVRM_MME_DISPATCH_INDIRECT
 * (nvrm_mme.c, Turing+). Still refused loudly: images (GLES 3.1 part B),
 * pre-Turing indirect dispatch.
 */
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "nvrm_priv.h"

#include "nak.h"
#include "nv_push.h"
#include "nv_push_cl9097.h"
#include "nv_push_cla097.h"
#include "nv_push_cl90c0.h"
#include "nv_push_cla0c0.h"
#include "nv_push_clb0c0.h"
#include "nv_push_clb1c0.h"
#include "nv_push_clc3c0.h"
#include "nv_push_clc6c0.h"
#include "cl906f.h"
#include "nv_push_clc597.h"
#include "nv_push_clc86f.h"
#include "cla0c0.h"
#include "clb0c0.h"
#include "clb1c0.h"
#include "clc3c0.h"
#include "clc5c0.h"
#include "clc6c0.h"
#include "clcbc0.h"
#include "clc597.h"
#include "clcb97.h"
#include "clcdc0.h"

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_upload_mgr.h"

/* ---------------------------------------------------------------- init */

/* nvk_push_dispatch_state_init (nvk_cmd_dispatch.c:31-88): compute class on
 * subchannel 1 (SUBC_NV90C0), the Maxwell texture-header select, the
 * pre-Volta program region (our shader heap, as the 3D class) and the
 * shared/local memory windows. Part of the screen channel's init push. */
void
nvrm_push_compute_init(const struct nvrm_screen *screen, struct nv_push *p)
{
   const struct nv_device_info *info = &screen->info;

   P_MTHD(p, NV90C0, SET_OBJECT);
   P_NV90C0_SET_OBJECT(p, {
      .class_id = info->cls_compute,
      .engine_id = 0,
   });

   if (info->cls_compute == MAXWELL_COMPUTE_A)
      P_IMMD(p, NVB0C0, SET_SELECT_MAXWELL_TEXTURE_HEADERS, V_TRUE);

   if (info->cls_compute < VOLTA_COMPUTE_A) {
      const uint64_t shader_base_addr = screen->shader_base;
      P_MTHD(p, NVA0C0, SET_PROGRAM_REGION_A);
      P_NVA0C0_SET_PROGRAM_REGION_A(p, shader_base_addr >> 32);
      P_NVA0C0_SET_PROGRAM_REGION_B(p, shader_base_addr);
   }

   if (info->cls_compute >= VOLTA_COMPUTE_A) {
      /* nvk_cmd_dispatch.c:57-80: shared window 4 GiB aligned on Hopper+. */
      uint64_t temp = info->cls_compute >= HOPPER_COMPUTE_A ? 1ULL << 32 : 0xfeULL << 24;
      P_MTHD(p, NVC3C0, SET_SHADER_SHARED_MEMORY_WINDOW_A);
      P_NVC3C0_SET_SHADER_SHARED_MEMORY_WINDOW_A(p, temp >> 32);
      P_NVC3C0_SET_SHADER_SHARED_MEMORY_WINDOW_B(p, temp & 0xffffffff);

      temp = 0xffULL << 24;
      P_MTHD(p, NVC3C0, SET_SHADER_LOCAL_MEMORY_WINDOW_A);
      P_NVC3C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A(p, temp >> 32);
      P_NVC3C0_SET_SHADER_LOCAL_MEMORY_WINDOW_B(p, temp & 0xffffffff);
   } else {
      P_MTHD(p, NVA0C0, SET_SHADER_LOCAL_MEMORY_WINDOW);
      P_NVA0C0_SET_SHADER_LOCAL_MEMORY_WINDOW(p, 0xff << 24);
      P_MTHD(p, NVA0C0, SET_SHADER_SHARED_MEMORY_WINDOW);
      P_NVA0C0_SET_SHADER_SHARED_MEMORY_WINDOW(p, 0xfe << 24);
   }
}

/* ---------------------------------------------------------------- NIR */

struct nvrm_lower_buffers_state {
   bool compute;
};

/* {addr lo, addr hi, size, 0} of entry `index` of the table at `base`. */
static nir_def *
nvrm_load_desc(nir_builder *b, const struct nvrm_lower_buffers_state *st,
               uint32_t base, nir_def *index)
{
   nir_def *cbuf = nir_imm_int(b, st->compute ? NVRM_CS_DESC_CBUF : NVRM_DESC_CBUF_SLOT);
   nir_def *off = nir_iadd_imm(b, nir_imul_imm(b, index, 16), base);
   return nir_ldc_nv(b, 4, 32, cbuf, off, .access = ACCESS_CAN_REORDER,
                     .align_mul = 16, .align_offset = 0);
}

static nir_def *
nvrm_desc_addr(nir_builder *b, nir_def *desc, nir_def *offset)
{
   nir_def *base = nir_pack_64_2x32_split(b, nir_channel(b, desc, 0), nir_channel(b, desc, 1));
   return nir_iadd(b, base, nir_u2u64(b, offset));
}

/* offset + bytes <= size, without overflow (nir_address_format_64bit_bounded_global). */
static nir_def *
nvrm_desc_in_bounds(nir_builder *b, nir_def *desc, nir_def *offset, unsigned bytes)
{
   nir_def *size = nir_channel(b, desc, 2);
   return nir_iand(b, nir_ult(b, offset, size),
                   nir_uge(b, nir_isub(b, size, offset), nir_imm_int(b, bytes)));
}

/* Bounded global load of `load`'s shape at desc + offset. */
static nir_def *
nvrm_bounded_load(nir_builder *b, nir_intrinsic_instr *load, nir_def *desc,
                  nir_def *offset, enum gl_access_qualifier access)
{
   const unsigned nc = load->def.num_components, bs = load->def.bit_size;
   nir_push_if(b, nvrm_desc_in_bounds(b, desc, offset, nc * bs / 8));
   nir_def *val = nir_build_load_global(b, nc, bs, nvrm_desc_addr(b, desc, offset),
                                  .access = access,
                                  .align_mul = nir_intrinsic_align_mul(load),
                                  .align_offset = nir_intrinsic_align_offset(load));
   nir_push_else(b, NULL);
   nir_def *zero = nir_imm_zero(b, nc, bs);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, val, zero);
}

static bool
nvrm_lower_buffers_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct nvrm_lower_buffers_state *st = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *res = NULL;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_ssbo: {
      nir_def *desc = nvrm_load_desc(b, st, NVRM_DESC_SSBO_OFFSET, intr->src[0].ssa);
      res = nvrm_bounded_load(b, intr, desc, intr->src[1].ssa, nir_intrinsic_access(intr));
      break;
   }
   case nir_intrinsic_store_ssbo: {
      nir_def *val = intr->src[0].ssa;
      nir_def *desc = nvrm_load_desc(b, st, NVRM_DESC_SSBO_OFFSET, intr->src[1].ssa);
      nir_def *offset = intr->src[2].ssa;
      nir_push_if(b, nvrm_desc_in_bounds(b, desc, offset,
                                         val->num_components * val->bit_size / 8));
      nir_build_store_global(b, val, nvrm_desc_addr(b, desc, offset),
                       .write_mask = nir_intrinsic_write_mask(intr),
                       .access = nir_intrinsic_access(intr),
                       .align_mul = nir_intrinsic_align_mul(intr),
                       .align_offset = nir_intrinsic_align_offset(intr));
      nir_pop_if(b, NULL);
      break;
   }
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap: {
      const unsigned bs = intr->def.bit_size;
      nir_def *desc = nvrm_load_desc(b, st, NVRM_DESC_SSBO_OFFSET, intr->src[0].ssa);
      nir_def *offset = intr->src[1].ssa;
      nir_push_if(b, nvrm_desc_in_bounds(b, desc, offset, bs / 8));
      nir_def *addr = nvrm_desc_addr(b, desc, offset);
      nir_def *val;
      if (intr->intrinsic == nir_intrinsic_ssbo_atomic)
         val = nir_global_atomic(b, bs, addr, intr->src[2].ssa,
                                 .atomic_op = nir_intrinsic_atomic_op(intr));
      else
         val = nir_global_atomic_swap(b, bs, addr, intr->src[2].ssa, intr->src[3].ssa,
                                      .atomic_op = nir_intrinsic_atomic_op(intr));
      nir_push_else(b, NULL);
      nir_def *zero = nir_imm_zero(b, 1, bs);
      nir_pop_if(b, NULL);
      res = nir_if_phi(b, val, zero);
      break;
   }
   case nir_intrinsic_get_ssbo_size: {
      nir_def *desc = nvrm_load_desc(b, st, NVRM_DESC_SSBO_OFFSET, intr->src[0].ssa);
      res = nir_channel(b, desc, 2);
      break;
   }
   case nir_intrinsic_load_ubo: {
      /* Blocks with a hardware constant buffer stay load_ubo (-> ldc_nv in
       * nvrm_draw.c): graphics slots 0..NVRM_DESC_CBUF_SLOT-1, compute
       * only block 0 (the QMD has 8 cbufs, nak_qmd_info.cbufs). Others,
       * and dynamically indexed blocks, are bounded global loads through
       * the descriptor table, as NVK's UBOs outside its cbuf map
       * (nvk_ubo_addr_format). */
      if (nir_src_is_const(intr->src[0]) &&
          nir_src_as_uint(intr->src[0]) < (st->compute ? 1 : NVRM_DESC_CBUF_SLOT))
         return false;
      nir_def *desc = nvrm_load_desc(b, st, NVRM_DESC_UBO_OFFSET, intr->src[0].ssa);
      res = nvrm_bounded_load(b, intr, desc, intr->src[1].ssa,
                              ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
      break;
   }
   case nir_intrinsic_load_num_workgroups: {
      /* NVK: lower_sysval_to_root_table(cs.group_count). */
      if (!st->compute)
         return false;
      nir_def *cbuf = nir_imm_int(b, NVRM_CS_DESC_CBUF);
      res = nir_ldc_nv(b, 3, 32, cbuf, nir_imm_int(b, NVRM_DESC_NUM_WG_OFFSET),
                       .access = ACCESS_CAN_REORDER, .align_mul = 16, .align_offset = 0);
      if (intr->def.bit_size != 32)
         res = nir_u2uN(b, res, intr->def.bit_size);
      break;
   }
   default:
      return false;
   }

   if (res)
      nir_def_rewrite_uses(&intr->def, res);
   nir_instr_remove(&intr->instr);
   return true;
}

/* Returns whether the shader now reads the descriptor cbuf. */
bool
nvrm_lower_buffers(nir_shader *nir)
{
   struct nvrm_lower_buffers_state st = {
      .compute = nir->info.stage == MESA_SHADER_COMPUTE,
   };
   return nir_shader_intrinsics_pass(nir, nvrm_lower_buffers_instr, nir_metadata_none, &st);
}

/* ---------------------------------------------------------------- state */

static void
nvrm_set_shader_buffers(struct pipe_context *pctx, mesa_shader_stage shader,
                        unsigned start, unsigned count,
                        const struct pipe_shader_buffer *buffers,
                        unsigned writable_bitmask)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   if (start + count > NVRM_MAX_SSBOS) {
      nvrm_error("shader buffers %u..%u exceed %u\n", start, start + count, NVRM_MAX_SSBOS);
      abort();
   }
   for (unsigned i = 0; i < count; i++) {
      struct pipe_shader_buffer *slot = &ctx->ssbo[shader][start + i];
      pipe_resource_reference(&slot->buffer, buffers ? buffers[i].buffer : NULL);
      slot->buffer_offset = buffers ? buffers[i].buffer_offset : 0;
      slot->buffer_size = buffers ? buffers[i].buffer_size : 0;
   }
   const uint32_t mask = BITFIELD_RANGE(start, count);
   ctx->ssbo_writable[shader] = (ctx->ssbo_writable[shader] & ~mask) |
                                ((writable_bitmask << start) & mask);
   ctx->dirty |= NVRM_DIRTY_SSBO;
}

static void
nvrm_desc_entry(struct nvrm_context *ctx, uint32_t *dw, struct pipe_resource *res,
                unsigned offset, unsigned size)
{
   if (!res)
      return; /* size 0: every access out of bounds */
   nvrm_keep(ctx, res);
   const uint64_t addr = nvrm_resource(res)->bo.va + offset;
   dw[0] = addr;
   dw[1] = addr >> 32;
   dw[2] = MIN2(size, res->width0 > offset ? res->width0 - offset : 0);
   dw[3] = 0;
}

/* Fills the stage's descriptor table and uploads it; returns its address. */
static uint64_t
nvrm_upload_desc(struct nvrm_context *ctx, mesa_shader_stage stage,
                 const uint32_t num_wg[3])
{
   uint32_t desc[NVRM_DESC_SIZE / 4] = { 0 };
   for (unsigned i = 0; i < NVRM_MAX_SSBOS; i++) {
      const struct pipe_shader_buffer *sb = &ctx->ssbo[stage][i];
      nvrm_desc_entry(ctx, &desc[(NVRM_DESC_SSBO_OFFSET + i * 16) / 4], sb->buffer,
                      sb->buffer_offset, sb->buffer_size);
   }
   for (unsigned i = 0; i < NVRM_MAX_UBOS; i++) {
      const struct pipe_constant_buffer *cb = &ctx->cb[stage][i];
      nvrm_desc_entry(ctx, &desc[(NVRM_DESC_UBO_OFFSET + i * 16) / 4], cb->buffer,
                      cb->buffer_offset, cb->buffer_size);
   }
   if (num_wg)
      memcpy(&desc[NVRM_DESC_NUM_WG_OFFSET / 4], num_wg, 3 * sizeof(uint32_t));
   struct pipe_resource *buf = NULL;
   unsigned offset = 0;
   u_upload_data_ref(ctx->base.const_uploader, 0, sizeof(desc),
                 ctx->base.screen->caps.constant_buffer_offset_alignment,
                 desc, &offset, &buf);
   if (!buf) {
      nvrm_error("descriptor table upload failed\n");
      abort();
   }
   nvrm_keep(ctx, buf);
   const uint64_t addr = nvrm_resource(buf)->bo.va + offset;
   pipe_resource_reference(&buf, NULL);
   return addr;
}

/* Graphics: the stage's table as constant buffer NVRM_DESC_CBUF_SLOT, the
 * way nvk_cmd_flush_gfx_cbufs binds its root table (nvk_cmd_draw.c). */
void
nvrm_emit_ssbo_desc(struct nvrm_context *ctx, struct nv_push *p,
                    mesa_shader_stage stage, uint32_t group)
{
   const uint64_t addr = nvrm_upload_desc(ctx, stage, NULL);
   P_MTHD(p, NV9097, SET_CONSTANT_BUFFER_SELECTOR_A);
   P_NV9097_SET_CONSTANT_BUFFER_SELECTOR_A(p, NVRM_DESC_SIZE);
   P_NV9097_SET_CONSTANT_BUFFER_SELECTOR_B(p, addr >> 32);
   P_NV9097_SET_CONSTANT_BUFFER_SELECTOR_C(p, addr);
   P_IMMD(p, NV9097, BIND_GROUP_CONSTANT_BUFFER(group), {
      .valid = VALID_TRUE, .shader_slot = NVRM_DESC_CBUF_SLOT,
   });
}

static void *
nvrm_create_compute_state(struct pipe_context *pctx, const struct pipe_compute_state *cso)
{
   if (cso->ir_type != PIPE_SHADER_IR_NIR) {
      nvrm_error("compute shader IR type %d (only NIR)\n", cso->ir_type);
      abort();
   }
   /* static_shared_mem is st/mesa's copy of nir->info.shared_size, which
    * NAK sizes the QMD shared memory from (cs.smem_size). */
   nir_shader *nir = (nir_shader *)cso->prog;
   if (cso->static_shared_mem > nir->info.shared_size) {
      nvrm_error("compute state asks %u B static shared memory beyond the "
                 "shader's %u B\n", cso->static_shared_mem, nir->info.shared_size);
      abort();
   }
   if (nir->info.workgroup_size_variable) {
      nvrm_error("variable workgroup size is not advertised\n");
      abort();
   }
   /* The shared memory limit is checked by the compile job
    * (nvrm_shader_compile_job) on NAK's result. */
   return nvrm_compile_shader(pctx, nir);
}

static void
nvrm_bind_compute_state(struct pipe_context *pctx, void *s)
{
   nvrm_context(pctx)->cs = s;
}

static void
nvrm_delete_compute_state(struct pipe_context *pctx, void *s)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   if (ctx->cs == s)
      ctx->cs = NULL;
   nvrm_free_shader(ctx, s);
}

static void
nvrm_get_compute_state_info(struct pipe_context *pctx, void *s,
                            struct pipe_compute_state_object_info *info)
{
   struct nvrm_shader *shader = s;
   nvrm_shader_wait(shader);
   info->max_threads = 1024;
   info->preferred_simd_size = 32;
   info->simd_sizes = 32;
   info->private_memory = shader->bin->info.slm_size;
}

/* ---------------------------------------------------------------- barriers */

/* Source side of NVK's shader-write barrier (nvk_cmd_flush_wait_dep,
 * NVK_BARRIER_FLUSH_SHADER_DATA with RENDER_WFI and COMPUTE_WFI): the
 * waiting INVALIDATE_SHADER_CACHES with flush_data on both classes. */
static void
nvrm_emit_flush_shader_data(struct nv_push *p)
{
   P_IMMD(p, NVA097, INVALIDATE_SHADER_CACHES, {
      .data = DATA_TRUE,
      .flush_data = FLUSH_DATA_TRUE,
   });
   P_IMMD(p, NVA0C0, INVALIDATE_SHADER_CACHES, {
      .data = DATA_TRUE,
      .flush_data = FLUSH_DATA_TRUE,
   });
}

/* Destination side (nvk_cmd_invalidate_deps): texture data, shader global
 * data + constants, and the SKED (QMD) cache. */
static void
nvrm_emit_invalidate_shader_data(const struct nvrm_screen *screen, struct nv_push *p)
{
   P_IMMD(p, NVA097, INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI, { .lines = LINES_ALL });
   P_IMMD(p, NVA097, INVALIDATE_SHADER_CACHES_NO_WFI, {
      .instruction = true, .global_data = true, .constant = true,
   });
   if (screen->info.cls_compute >= MAXWELL_COMPUTE_B)
      P_IMMD(p, NVB1C0, INVALIDATE_SKED_CACHES, 0);
}

/* NVK_BARRIER_INVALIDATE_MME_DATA (nvk_cmd_buffer.c:615-630): shader
 * writes visible to the MME's DMA reads of indirect commands. */
static void
nvrm_emit_invalidate_mme_data(const struct nvrm_screen *screen, struct nv_push *p)
{
   if (screen->info.cls_eng3d >= HOPPER_A) {
      P_IMMD(p, NVC86F, WFI, 0);
      P_MTHD(p, NVC86F, MEM_OP_A);
      P_NVC86F_MEM_OP_A(p, {});
      P_NVC86F_MEM_OP_B(p, 0);
      P_NVC86F_MEM_OP_C(p, { .membar_type = 0 });
      P_NVC86F_MEM_OP_D(p, { .operation = OPERATION_MEMBAR });
   } else {
      __push_immd(p, SUBC_NV9097, NV906F_SET_REFERENCE, 0);
      if (screen->info.cls_eng3d >= TURING_A)
         P_IMMD(p, NVC597, MME_DMA_SYSMEMBAR, 0);
   }
}

/* glMemoryBarrier: every GL barrier bit maps to NVK's full shader-data
 * flush + invalidate (there is no finer split worth the risk here). CPU
 * visibility of mapped buffers comes from fence / map waits on the
 * submission tokens. */
static void
nvrm_memory_barrier(struct pipe_context *pctx, unsigned flags)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   const struct nvrm_screen *screen = nvrm_screen(pctx->screen);
   if (!(flags & ~PIPE_BARRIER_UPDATE))
      return;
   struct nv_push push = nvrm_push_begin(ctx, 32);
   P_IMMD(&push, NV9097, WAIT_FOR_IDLE, 0);
   nvrm_emit_flush_shader_data(&push);
   nvrm_emit_invalidate_shader_data(screen, &push);
   if (flags & PIPE_BARRIER_INDIRECT_BUFFER)
      nvrm_emit_invalidate_mme_data(screen, &push);
   nvrm_push_end(ctx, &push);
}

/* ---------------------------------------------------------------- launch */

static void
nvrm_launch_grid(struct pipe_context *pctx, const struct pipe_grid_info *info)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   struct nvrm_screen *screen = nvrm_screen(pctx->screen);
   struct nvrm_shader *cs = ctx->cs;
   if (!cs) {
      nvrm_error("launch_grid without a compute shader bound\n");
      abort();
   }
   nvrm_shader_wait(cs);
   if (info->variable_shared_mem || info->work_dim > 3 ||
       info->grid_base[0] || info->grid_base[1] || info->grid_base[2]) {
      nvrm_error("launch_grid with variable shared memory / base groups "
                 "(OpenCL-style) is not supported\n");
      abort();
   }
   const struct nak_shader_info *si = &cs->bin->info;
   for (unsigned i = 0; i < 3; i++) {
      if (info->block[i] != si->cs.local_size[i]) {
         nvrm_error("launch_grid block %ux%ux%u differs from the shader's %ux%ux%u\n",
                    info->block[0], info->block[1], info->block[2],
                    si->cs.local_size[0], si->cs.local_size[1], si->cs.local_size[2]);
         abort();
      }
   }
   if (!info->indirect && (!info->grid[0] || !info->grid[1] || !info->grid[2]))
      return;
   /* Indirect: group counts are written by NVRM_MME_DISPATCH_INDIRECT. */
   static const uint32_t zero_grid[3] = { 0, 0, 0 };
   const uint32_t *grid = info->indirect ? zero_grid : info->grid;

   /* May flush: before any push is opened. */
   nvrm_slm_ensure(ctx, si->slm_size, si->crs_size);
   /* The handle table below writes this batch slot's texture pool: the
    * launch push must not flush (rotate the slot) after it. */
   const uint32_t tex_write_dw = nvrm_tex_pool_write_dw(cs->tex_units + cs->image_units);
   nvrm_push_reserve(ctx, 96 + tex_write_dw);
   nvrm_texture_reserve(ctx, cs->tex_units + cs->image_units);

   /* Descriptor table (SSBOs, UBOs past block 0, gl_NumWorkGroups) and the
    * QMD, as nvk_cmd_upload_qmd: cbuf 0 = default uniform block (the
    * root table in NVK), cbuf NVRM_CS_DESC_CBUF = descriptors. */
   const uint64_t desc_addr = nvrm_upload_desc(ctx, MESA_SHADER_COMPUTE, grid);
   /* Texture/image handle table (part B, nvrm_texture.c) as QMD cbuf 2. */
   uint32_t tex_size_B = 0;
   const uint64_t tex_addr = nvrm_upload_tex_handles(ctx, MESA_SHADER_COMPUTE, cs,
                                                     &tex_size_B);
   struct nak_qmd_info qi = {
      .addr = cs->hdr_addr,
      /* NAK picks the shared memory configuration from the device's
       * sm_smem_sizes_kB (nv_device_info_init_shared_mem_sizes), as for NVK
       * (nvk_cmd_dispatch.c nvk_cmd_upload_qmd). */
      .smem_size = si->cs.smem_size,
      .global_size = { grid[0], grid[1], grid[2] },
   };
   const struct pipe_constant_buffer *cb0 = &ctx->cb[MESA_SHADER_COMPUTE][0];
   if (cb0->buffer) {
      nvrm_keep(ctx, cb0->buffer);
      qi.cbufs[qi.num_cbufs++] = (struct nak_qmd_cbuf) {
         .index = 0,
         .addr = nvrm_resource(cb0->buffer)->bo.va + cb0->buffer_offset,
         .size = MIN2(nvrm_cbuf_size(screen, cb0->buffer_size), 65536),
      };
   }
   qi.cbufs[qi.num_cbufs++] = (struct nak_qmd_cbuf) {
      .index = NVRM_CS_DESC_CBUF,
      .addr = desc_addr,
      .size = NVRM_DESC_SIZE,
   };
   if (tex_addr) {
      qi.cbufs[qi.num_cbufs++] = (struct nak_qmd_cbuf) {
         .index = NVRM_CS_TEX_CBUF,
         .addr = tex_addr,
         .size = tex_size_B,
      };
   }
   /* QMD (nak_qmd_size_B: 256 B up to Hopper, larger for Blackwell's QMD
    * V5, as nvk_cmd_dispatch.c nvk_cmd_upload_qmd) + 256 B of zero padding:
    * SKED reads past the QMD. Measured
    * 10-15 (5090, dEQP-GLES31 image_load_store after ~3.9k dispatches):
    * Xid 31 HUBCLIENT_SKED read fault at exactly qmd + 0x100 when an
    * earlier QMD sat in the last 256 B of an upload buffer (qmd ffff72ff00,
    * fault ff_ff730000 = the unmapped page after the buffer). NVK's
    * upload stream (nvk_cmd_buffer_alloc_qmd) never ends a mapping right
    * after a QMD in practice; nvrm's uploader can. */
   const uint32_t qmd_size_B = nak_qmd_size_B(&screen->info);
   uint32_t qmd[(NAK_MAX_QMD_SIZE_B + 256) / 4];
   memset(qmd, 0, sizeof(qmd));
   nak_fill_qmd(&screen->info, si, &qi, qmd, qmd_size_B);

   struct pipe_resource *qbuf = NULL;
   unsigned qoff = 0;
   u_upload_data_ref(pctx->stream_uploader, 0, qmd_size_B + 256, NAK_QMD_ALIGN_B, qmd,
                 &qoff, &qbuf);
   if (!qbuf) {
      nvrm_error("QMD upload failed\n");
      abort();
   }
   nvrm_keep(ctx, qbuf);
   const uint64_t qmd_addr = nvrm_resource(qbuf)->bo.va + qoff;
   pipe_resource_reference(&qbuf, NULL);

   struct nv_push push = nvrm_push_begin(ctx, 96 + tex_write_dw);
   struct nv_push *p = &push;

   /* GL orders a dispatch after earlier rendering (render to texture,
    * transform feedback, ...) without an application barrier: wait for the
    * 3D pipe and invalidate what compute reads, as an NVK
    * colour-attachment-write -> compute-read barrier. Shader code and the
    * descriptor table were written by the CPU. */
   P_IMMD(p, NV9097, WAIT_FOR_IDLE, 0);
   nvrm_emit_invalidate_shader_data(screen, p);
   /* Pool entries the handle table added, before the header/sampler cache
    * invalidates below. */
   nvrm_tex_pool_write(ctx, p, true);

   /* Compute texture header/sampler pools: the context's pools, as
    * nvk_queue_state_update binds the same pools on both classes
    * (nvk_queue.c:85-117). */
   if (tex_addr) {
      /* Pools once per batch (they change only with the batch slot; the
       * channel is shared, so each batch sets its own), as 3D's
       * NVRM_DIRTY_POOLS; header/sampler caches invalidated per dispatch,
       * since the table may have added pool entries. */
      if (ctx->cs_pools_seq != ctx->flush_seq + 1) {
         const uint64_t tic = ctx->tex_pool.va;
         const uint64_t tsc = ctx->tex_pool.va + NVRM_TEX_POOL_SLOTS * 32u;
         P_MTHD(p, NVA0C0, SET_TEX_HEADER_POOL_A);
         P_NVA0C0_SET_TEX_HEADER_POOL_A(p, tic >> 32);
         P_NVA0C0_SET_TEX_HEADER_POOL_B(p, tic);
         P_NVA0C0_SET_TEX_HEADER_POOL_C(p, NVRM_TEX_POOL_SLOTS - 1);
         P_MTHD(p, NVA0C0, SET_TEX_SAMPLER_POOL_A);
         P_NVA0C0_SET_TEX_SAMPLER_POOL_A(p, tsc >> 32);
         P_NVA0C0_SET_TEX_SAMPLER_POOL_B(p, tsc);
         P_NVA0C0_SET_TEX_SAMPLER_POOL_C(p, NVRM_TEX_POOL_SLOTS - 1);
         ctx->cs_pools_seq = ctx->flush_seq + 1;
      }
      P_IMMD(p, NVA0C0, INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI, { .lines = LINES_ALL });
      P_IMMD(p, NVA0C0, INVALIDATE_SAMPLER_CACHE_NO_WFI, { .lines = LINES_ALL });
   }

   /* Compute part of nvk_queue_state_update's SLM block (nvk_queue.c:139-155). */
   if (ctx->slm.mem) {
      P_MTHD(p, NVA0C0, SET_SHADER_LOCAL_MEMORY_A);
      P_NVA0C0_SET_SHADER_LOCAL_MEMORY_A(p, ctx->slm.va >> 32);
      P_NVA0C0_SET_SHADER_LOCAL_MEMORY_B(p, ctx->slm.va);
      P_MTHD(p, NVA0C0, SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A);
      P_NVA0C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A(p, ctx->slm_per_tpc >> 32);
      P_NVA0C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B(p, ctx->slm_per_tpc);
      P_NVA0C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C(p, 0xff);
      if (screen->info.cls_compute < VOLTA_COMPUTE_A) {
         P_MTHD(p, NVA0C0, SET_SHADER_LOCAL_MEMORY_THROTTLED_A);
         P_NVA0C0_SET_SHADER_LOCAL_MEMORY_THROTTLED_A(p, ctx->slm_per_tpc >> 32);
         P_NVA0C0_SET_SHADER_LOCAL_MEMORY_THROTTLED_B(p, ctx->slm_per_tpc);
         P_NVA0C0_SET_SHADER_LOCAL_MEMORY_THROTTLED_C(p, 0xff);
      }
   }

   /* GLES has no conditional rendering; GL's st/mesa checks it on the CPU
    * before launch_grid. The 3D render-enable state of a pending GL
    * conditional render must not skip the dispatch: as NVK's internal
    * dispatches (nvk_cmd_dispatch_shader). */
   if (info->indirect && nvrm_mme_dma(screen)) {
      /* nvk_CmdDispatchIndirect, Turing+ branch (nvk_cmd_dispatch.c:560-571). */
      nvrm_keep(ctx, info->indirect);
      const uint64_t dispatch_addr = nvrm_resource(info->indirect)->bo.va +
                                     info->indirect_offset;
      if (screen->info.cls_compute < BLACKWELL_COMPUTE_A)
         P_IMMD(p, NVC597, SET_MME_DATA_FIFO_CONFIG, FIFO_SIZE_SIZE_4KB);
      P_1INC(p, NV9097, CALL_MME_MACRO(NVRM_MME_DISPATCH_INDIRECT));
      P_INLINE_DATA(p, dispatch_addr >> 32);
      P_INLINE_DATA(p, dispatch_addr);
      P_INLINE_DATA(p, desc_addr >> 32);
      P_INLINE_DATA(p, desc_addr);
      P_INLINE_DATA(p, qmd_addr >> 32);
      P_INLINE_DATA(p, qmd_addr);
   } else if (info->indirect) {
      /* nvk_CmdDispatchIndirect, pre-Turing branch (nvk_cmd_dispatch.c:
       * 572-588): stall the command streamer (the counts may come from
       * earlier GPU work), then the macro call whose first three data
       * words are the VkDispatchIndirectCommand-layout counts as a
       * NO_PREFETCH GPFIFO entry, then the descriptor-table and QMD
       * addresses inline. */
      nvrm_keep(ctx, info->indirect);
      const uint64_t dispatch_addr = nvrm_resource(info->indirect)->bo.va +
                                     info->indirect_offset;
      __push_immd(p, SUBC_NV9097, NV906F_SET_REFERENCE, 0);
      P_1INC(p, NV9097, CALL_MME_MACRO(NVRM_MME_DISPATCH_INDIRECT));
      nv_push_update_count(p, 3);
      nvrm_push_indirect(ctx, p, dispatch_addr, 3 * 4);
      P_INLINE_DATA(p, desc_addr >> 32);
      P_INLINE_DATA(p, desc_addr);
      P_INLINE_DATA(p, qmd_addr >> 32);
      P_INLINE_DATA(p, qmd_addr);
   }
   P_IMMD(p, NVA0C0, SET_RENDER_ENABLE_OVERRIDE, MODE_ALWAYS_RENDER);
   P_MTHD(p, NVA0C0, SEND_PCAS_A);
   P_NVA0C0_SEND_PCAS_A(p, qmd_addr >> 8);
   if (screen->info.cls_compute <= TURING_COMPUTE_A) {
      P_IMMD(p, NVA0C0, SEND_SIGNALING_PCAS_B, {
         .invalidate = INVALIDATE_TRUE,
         .schedule = SCHEDULE_TRUE,
      });
   } else {
      P_IMMD(p, NVC6C0, SEND_SIGNALING_PCAS2_B, PCAS_ACTION_INVALIDATE_COPY_SCHEDULE);
   }
   P_IMMD(p, NVA0C0, SET_RENDER_ENABLE_OVERRIDE, MODE_USE_RENDER_ENABLE);

   /* Later GL commands may overwrite what this dispatch still reads (WAR)
    * without a barrier: wait for compute and flush its writes (NVK's
    * COMPUTE_WFI shader-data flush). */
   nvrm_emit_flush_shader_data(p);
   nvrm_push_end(ctx, p);
   /* Next draw re-invalidates texture/shader caches (what the per-draw
    * invalidate used to give the 3D side after a dispatch). */
   ctx->inval |= NVRM_INVAL_TEX_DATA | NVRM_INVAL_GLOBAL | NVRM_INVAL_CONSTANT;
}

void
nvrm_init_compute_functions(struct pipe_context *p)
{
   p->create_compute_state = nvrm_create_compute_state;
   p->bind_compute_state = nvrm_bind_compute_state;
   p->delete_compute_state = nvrm_delete_compute_state;
   p->get_compute_state_info = nvrm_get_compute_state_info;
   p->launch_grid = nvrm_launch_grid;
   p->set_shader_buffers = nvrm_set_shader_buffers;
   p->memory_barrier = nvrm_memory_barrier;
}

void
nvrm_compute_context_destroy(struct nvrm_context *ctx)
{
   for (unsigned s = 0; s < MESA_SHADER_STAGES; s++)
      for (unsigned i = 0; i < NVRM_MAX_SSBOS; i++)
         pipe_resource_reference(&ctx->ssbo[s][i].buffer, NULL);
}
