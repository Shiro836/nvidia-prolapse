/* SPDX-License-Identifier: MIT
 *
 * nvrm MME macros for GPU-sourced (indirect) draws and dispatches, built with
 * Mesa's mme_builder and uploaded into the 3D class's macro RAM at channel init
 * the way nvk_push_draw_state_init does (nvk_cmd_draw.c:152-172).
 *
 * Each macro is NVK's Turing+ macro (MME DMA read through the data FIFO,
 * mme_tu104_read_fifoed) minus what NVK needs only for Vulkan's root
 * descriptor table / multiview / statistics:
 *  - NVRM_MME_DRAW_INDIRECT: nvk_mme_draw_indirect + nvk_mme_build_draw +
 *    nvk_mme_build_draw_loop (nvk_cmd_draw.c:3974-4060, 4274-4310);
 *  - NVRM_MME_DRAW_INDEXED_INDIRECT: nvk_mme_draw_indexed_indirect +
 *    nvk_mme_build_draw_indexed(_loop) (nvk_cmd_draw.c:4119-4200, 4374-4410);
 *  - NVRM_MME_DISPATCH_INDIRECT: nvk_mme_dispatch_indirect
 *    (nvk_cmd_dispatch.c:459-500) writing the group counts into nvrm's
 *    descriptor table (gl_NumWorkGroups) instead of NVK's root table, and into
 *    the QMD at NAK's dispatch-size layout.
 * Turing+: the begin word (BEGIN op + INSTANCE_ITERATE_ENABLE) comes inline
 * from the CPU. Pre-Turing (Fermi-style MME, no DMA, 7 registers): NVK's
 * pre-Turing branches of the same functions: the commands arrive as push
 * data (nvk_cmd_buffer_push_indirect: a NO_PREFETCH GPFIFO entry continuing
 * the CALL_MME_MACRO method, nvrm_push_indirect), BEGIN and spilled loop
 * registers live in MME shadow scratch, and below PASCAL_B instances are
 * looped in the macro.
 */
#include <stdlib.h>

#include "nvrm_priv.h"

#include "drf.h"
#include "mme_builder.h"
#include "nak.h"
#include "nv_push.h"
#include "nv_push_cl9097.h"
#include "nv_push_clc597.h"
#include "cl9097.h"
#include "clc197.h"
#include "clc597.h"
#include "clb197.h"
#include "clc397.h"
#include "clcb97.h"

/* nvk_cmd_dispatch.c:381-389 */
static void
nvrm_mme_store_global(struct mme_builder *b, struct mme_value64 addr, struct mme_value v)
{
   mme_mthd(b, NV9097_SET_REPORT_SEMAPHORE_A);
   mme_emit_addr64(b, addr);
   mme_emit(b, v);
   mme_emit(b, mme_imm(0x10000000));
}

/* MME shadow scratch slots of nvrm's macros (SET_MME_SHADOW_SCRATCH(i) =
 * 0x3400 + 4 i; 0..2 are the FALCON slots of SET_PRIV_REG below). The
 * pre-Turing draw macros keep the BEGIN word and spilled loop registers
 * there, as NVK's DRAW_BEGIN / DRAW_COUNT / DRAW_PAD_DW / DRAW_IDX
 * (nvk_mme.h:54-57): the Fermi-style MME has 7 registers. */
enum {
   NVRM_MME_SCRATCH_DRAW_BEGIN = 3,
   NVRM_MME_SCRATCH_DRAW_COUNT,
   NVRM_MME_SCRATCH_DRAW_PAD_DW,
   NVRM_MME_SCRATCH_DRAW_IDX,
};
#define NVRM_MME_SCRATCH_MTHD(i) (0x3400 + (i) * 4)

uint32_t
nvrm_mme_scratch_draw_begin_mthd(void)
{
   return NVRM_MME_SCRATCH_MTHD(NVRM_MME_SCRATCH_DRAW_BEGIN);
}

/* nvk_mme.h:105-203 helpers. */
static struct mme_value
nvrm_mme_load_scratch(struct mme_builder *b, uint32_t scratch)
{
   struct mme_value val = mme_alloc_reg(b);
   mme_state_to(b, val, NVRM_MME_SCRATCH_MTHD(scratch));
   return val;
}

static void
nvrm_mme_store_scratch(struct mme_builder *b, uint32_t scratch, struct mme_value data)
{
   mme_mthd(b, NVRM_MME_SCRATCH_MTHD(scratch));
   mme_emit(b, data);
}

static void
nvrm_mme_load_to_scratch(struct mme_builder *b, uint32_t scratch)
{
   struct mme_value val = mme_load(b);
   nvrm_mme_store_scratch(b, scratch, val);
   mme_free_reg(b, val);
}

static void
nvrm_mme_spill(struct mme_builder *b, uint32_t scratch, struct mme_value val)
{
   if (val.type == MME_VALUE_TYPE_REG) {
      nvrm_mme_store_scratch(b, scratch, val);
      mme_free_reg(b, val);
   }
}

static void
nvrm_mme_unspill(struct mme_builder *b, uint32_t scratch, struct mme_value val)
{
   if (val.type == MME_VALUE_TYPE_REG) {
      mme_realloc_reg(b, val);
      mme_state_to(b, val, NVRM_MME_SCRATCH_MTHD(scratch));
   }
}

/* nvk_mme_build_draw_loop / nvk_mme_build_draw_indexed_loop
 * (nvk_cmd_draw.c:3974-4003, 4119-4148). Turing+ callers pass the BEGIN
 * word (INSTANCE_ITERATE_ENABLE already set by the CPU, PASCAL_B+);
 * pre-Turing ones load it from DRAW_BEGIN scratch like NVK. Below PASCAL_B
 * there is no SET_INSTANCE_COUNT: the macro repeats BEGIN..END per
 * instance, INSTANCE_ID FIRST then SUBSEQUENT. */
static void
nvrm_mme_build_draw_loop(struct mme_builder *b, struct mme_value begin_in,
                         struct mme_value instance_count, uint32_t start_mthd,
                         struct mme_value first, struct mme_value count)
{
   struct mme_value begin = begin_in;
   if (b->devinfo->cls_eng3d < TURING_A)
      begin = nvrm_mme_load_scratch(b, NVRM_MME_SCRATCH_DRAW_BEGIN);

   if (b->devinfo->cls_eng3d < PASCAL_B) {
      mme_start_loop(b, instance_count);
   } else {
      mme_mthd(b, NVC197_SET_INSTANCE_COUNT);
      mme_emit(b, instance_count);
   }

   mme_mthd(b, NV9097_BEGIN);
   mme_emit(b, begin);

   mme_mthd(b, start_mthd);
   mme_emit(b, first);
   mme_emit(b, count);

   mme_mthd(b, NV9097_END);
   mme_emit(b, mme_zero());

   if (b->devinfo->cls_eng3d < PASCAL_B) {
      mme_set_field_enum(b, begin, NV9097_BEGIN_INSTANCE_ID, SUBSEQUENT);
      mme_end_loop(b);
   }

   if (b->devinfo->cls_eng3d < TURING_A)
      mme_free_reg(b, begin);
}

/* One DrawArraysIndirectCommand from the macro's data (DMA FIFO on
 * Turing+, push data before): nvk_mme_build_draw (nvk_cmd_draw.c:4005-4060)
 * without NVK's root-table draw parameters and multiview. */
static void
nvrm_mme_build_draw(struct mme_builder *b, struct mme_value begin, struct mme_value draw)
{
   struct mme_value vertex_count = mme_load(b);
   struct mme_value instance_count = mme_load(b);
   struct mme_value first_vertex = mme_load(b);
   struct mme_value first_instance = mme_load(b);

   /* nvk_mme_build_set_draw_params: base vertex 0 for array draws. */
   mme_mthd(b, NV9097_SET_GLOBAL_BASE_INSTANCE_INDEX);
   mme_emit(b, first_instance);
   mme_free_reg(b, first_instance);
   mme_mthd(b, NV9097_SET_GLOBAL_BASE_VERTEX_INDEX);
   mme_emit(b, mme_zero());
   mme_mthd(b, NV9097_SET_VERTEX_ID_BASE);
   mme_emit(b, mme_zero());

   if (b->devinfo->cls_eng3d < TURING_A)
      nvrm_mme_spill(b, NVRM_MME_SCRATCH_DRAW_IDX, draw);

   nvrm_mme_build_draw_loop(b, begin, instance_count, NV9097_SET_VERTEX_ARRAY_START,
                            first_vertex, vertex_count);

   mme_free_reg(b, instance_count);
   mme_free_reg(b, first_vertex);
   mme_free_reg(b, vertex_count);

   if (b->devinfo->cls_eng3d < TURING_A)
      nvrm_mme_unspill(b, NVRM_MME_SCRATCH_DRAW_IDX, draw);
}

/* One DrawElementsIndirectCommand: nvk_mme_build_draw_indexed
 * (nvk_cmd_draw.c:4150-4210). */
static void
nvrm_mme_build_draw_indexed(struct mme_builder *b, struct mme_value begin,
                            struct mme_value draw)
{
   struct mme_value index_count = mme_load(b);
   struct mme_value instance_count = mme_load(b);
   struct mme_value first_index = mme_load(b);
   struct mme_value vertex_offset = mme_load(b);
   struct mme_value first_instance = mme_load(b);

   mme_mthd(b, NV9097_SET_GLOBAL_BASE_VERTEX_INDEX);
   mme_emit(b, vertex_offset);
   mme_mthd(b, NV9097_SET_VERTEX_ID_BASE);
   mme_emit(b, vertex_offset);
   mme_free_reg(b, vertex_offset);
   mme_mthd(b, NV9097_SET_GLOBAL_BASE_INSTANCE_INDEX);
   mme_emit(b, first_instance);
   mme_free_reg(b, first_instance);

   if (b->devinfo->cls_eng3d < TURING_A)
      nvrm_mme_spill(b, NVRM_MME_SCRATCH_DRAW_IDX, draw);

   nvrm_mme_build_draw_loop(b, begin, instance_count, NV9097_SET_INDEX_BUFFER_F,
                            first_index, index_count);

   mme_free_reg(b, instance_count);
   mme_free_reg(b, first_index);
   mme_free_reg(b, index_count);

   if (b->devinfo->cls_eng3d < TURING_A)
      nvrm_mme_unspill(b, NVRM_MME_SCRATCH_DRAW_IDX, draw);
}

/* nvk_mme_draw_indirect / nvk_mme_draw_indexed_indirect
 * (nvk_cmd_draw.c:4273-4310, 4373-4410). Turing+: {begin, addr, count,
 * stride}, commands DMA-read. Pre-Turing: {count, pad_dw} followed by the
 * commands as push data (count * stride bytes, nvrm_push_indirect); pad_dw
 * dwords after each command are dropped. */
static void
nvrm_mme_draw_indirect_common(struct mme_builder *b, bool indexed)
{
   const uint32_t cmd_dw = indexed ? 5 : 4;
   if (b->devinfo->cls_eng3d >= TURING_A) {
      struct mme_value begin = mme_load(b);
      struct mme_value64 draw_addr = mme_load_addr64(b);
      struct mme_value draw_count = mme_load(b);
      struct mme_value stride = mme_load(b);

      struct mme_value draw = mme_mov(b, mme_zero());
      mme_while(b, ult, draw, draw_count) {
         mme_tu104_read_fifoed(b, draw_addr, mme_imm(cmd_dw));

         if (indexed)
            nvrm_mme_build_draw_indexed(b, begin, draw);
         else
            nvrm_mme_build_draw(b, begin, draw);

         mme_add_to(b, draw, draw, mme_imm(1));
         mme_add64_to(b, draw_addr, draw_addr, mme_value64(stride, mme_zero()));
      }
   } else {
      struct mme_value draw_count = mme_load(b);
      nvrm_mme_load_to_scratch(b, NVRM_MME_SCRATCH_DRAW_PAD_DW);

      struct mme_value draw = mme_mov(b, mme_zero());
      mme_while(b, ine, draw, draw_count) {
         nvrm_mme_spill(b, NVRM_MME_SCRATCH_DRAW_COUNT, draw_count);

         if (indexed)
            nvrm_mme_build_draw_indexed(b, mme_zero(), draw);
         else
            nvrm_mme_build_draw(b, mme_zero(), draw);
         mme_add_to(b, draw, draw, mme_imm(1));

         struct mme_value pad_dw = nvrm_mme_load_scratch(b, NVRM_MME_SCRATCH_DRAW_PAD_DW);
         mme_loop(b, pad_dw) {
            mme_free_reg(b, mme_load(b));
         }
         mme_free_reg(b, pad_dw);

         nvrm_mme_unspill(b, NVRM_MME_SCRATCH_DRAW_COUNT, draw_count);
      }
   }
}

static void
nvrm_mme_draw_indirect(struct mme_builder *b)
{
   nvrm_mme_draw_indirect_common(b, false);
}

static void
nvrm_mme_draw_indexed_indirect(struct mme_builder *b)
{
   nvrm_mme_draw_indirect_common(b, true);
}

/* nvk_mme_draw / nvk_mme_draw_indexed (nvk_cmd_draw.c:4066-4070,
 * 4212-4216): one direct draw {draw index, the draw's command words}
 * through the same loop, so classes without hardware instance iteration
 * repeat BEGIN..END in the MME instead of the push buffer. nvrm calls them
 * below PASCAL_B for instanced draws; Turing+ take a leading BEGIN word
 * like the indirect macros. */
static void
nvrm_mme_draw_direct(struct mme_builder *b, bool indexed)
{
   struct mme_value begin = b->devinfo->cls_eng3d >= TURING_A ? mme_load(b) : mme_zero();
   struct mme_value draw_index = mme_load(b);
   if (indexed)
      nvrm_mme_build_draw_indexed(b, begin, draw_index);
   else
      nvrm_mme_build_draw(b, begin, draw_index);
}

static void
nvrm_mme_draw(struct mme_builder *b)
{
   nvrm_mme_draw_direct(b, false);
}

static void
nvrm_mme_draw_indexed(struct mme_builder *b)
{
   nvrm_mme_draw_direct(b, true);
}

/* mme_store_global_vec3_free_addr (nvk_cmd_dispatch.c:391-409). */
static void
nvrm_mme_store_vec3_free_addr(struct mme_builder *b, struct mme_value64 addr,
                              uint32_t offset, struct mme_value x,
                              struct mme_value y, struct mme_value z)
{
   if (offset > 0)
      mme_add64_to(b, addr, addr, mme_imm64(offset));
   nvrm_mme_store_global(b, addr, x);
   mme_add64_to(b, addr, addr, mme_imm64(4));
   nvrm_mme_store_global(b, addr, y);
   mme_add64_to(b, addr, addr, mme_imm64(4));
   nvrm_mme_store_global(b, addr, z);
   mme_free_reg64(b, addr);
}

/* mme_store_qmd_dispatch_size (nvk_cmd_dispatch.c:426-457). */
static void
nvrm_mme_store_qmd_dispatch_size(struct mme_builder *b, struct mme_value64 qmd_addr,
                                 struct mme_value x, struct mme_value y,
                                 struct mme_value z)
{
   const struct nak_qmd_dispatch_size_layout l = nak_get_qmd_dispatch_size_layout(b->devinfo);
   assert(l.y_start == l.x_start + 32);
   if (l.z_start == l.y_start + 32) {
      nvrm_mme_store_vec3_free_addr(b, qmd_addr, l.x_start / 8, x, y, z);
   } else {
      mme_add64_to(b, qmd_addr, qmd_addr, mme_imm64(l.x_start / 8));
      nvrm_mme_store_global(b, qmd_addr, x);
      assert(l.z_start == l.y_start + 16);
      struct mme_value yz = mme_merge(b, y, z, 16, 16, 0);
      mme_add64_to(b, qmd_addr, qmd_addr, mme_imm64(4));
      nvrm_mme_store_global(b, qmd_addr, yz);
      mme_free_reg(b, yz);
      mme_free_reg64(b, qmd_addr);
   }
}

/* nvk_mme_dispatch_indirect (nvk_cmd_dispatch.c:459-525) writing the
 * group counts into nvrm's descriptor table (gl_NumWorkGroups) instead of
 * NVK's root table, and into the QMD; no invocation statistics. Turing+:
 * {dispatch addr, desc addr, qmd addr}, counts DMA-read. Pre-Turing: the
 * three counts as push data first, then {desc addr, qmd addr}. */
static void
nvrm_mme_dispatch_indirect(struct mme_builder *b)
{
   if (b->devinfo->cls_eng3d >= TURING_A) {
      /* Load everything before switching to the indirect read. */
      struct mme_value64 dispatch_addr = mme_load_addr64(b);
      struct mme_value64 desc_addr = mme_load_addr64(b);
      struct mme_value64 qmd_addr = mme_load_addr64(b);

      mme_tu104_read_fifoed(b, dispatch_addr, mme_imm(3));
      mme_free_reg64(b, dispatch_addr);
      struct mme_value x = mme_load(b);
      struct mme_value y = mme_load(b);
      struct mme_value z = mme_load(b);

      nvrm_mme_store_vec3_free_addr(b, desc_addr, NVRM_DESC_NUM_WG_OFFSET, x, y, z);
      nvrm_mme_store_qmd_dispatch_size(b, qmd_addr, x, y, z);
      mme_free_reg(b, x);
      mme_free_reg(b, y);
      mme_free_reg(b, z);
   } else {
      struct mme_value x = mme_load(b);
      struct mme_value y = mme_load(b);
      struct mme_value z = mme_load(b);

      struct mme_value64 desc_addr = mme_load_addr64(b);
      nvrm_mme_store_vec3_free_addr(b, desc_addr, NVRM_DESC_NUM_WG_OFFSET, x, y, z);

      struct mme_value64 qmd_addr = mme_load_addr64(b);
      nvrm_mme_store_qmd_dispatch_size(b, qmd_addr, x, y, z);
      mme_free_reg(b, x);
      mme_free_reg(b, y);
      mme_free_reg(b, z);
   }
}

/* mme_set_priv_reg (nvk_cmd_draw.c:74-97): the FECS firmware method
 * SET_FALCON04 takes {0, value, mask} from MME shadow scratch 0..2
 * (NVK_MME_SCRATCH_FALCON_*, nvk_mme.h:47-50; SET_MME_SHADOW_SCRATCH(i) =
 * 0x3400 + 4 i) and clears scratch 0 when done. */
#define NVRM_SET_MME_SCRATCH_FALCON_0 0x3400

static void
nvrm_mme_set_priv_reg_raw(struct mme_builder *b, struct mme_value value,
                          struct mme_value mask, struct mme_value reg)
{
   mme_mthd(b, NV9097_WAIT_FOR_IDLE);
   mme_emit(b, mme_zero());

   mme_mthd(b, NVRM_SET_MME_SCRATCH_FALCON_0);
   mme_emit(b, mme_zero());
   mme_emit(b, value);
   mme_emit(b, mask);

   mme_mthd(b, NV9097_SET_FALCON04);
   mme_emit(b, reg);

   struct mme_value loop_cond = mme_mov(b, mme_zero());
   mme_while(b, ine, loop_cond, mme_imm(1)) {
      mme_state_to(b, loop_cond, NVRM_SET_MME_SCRATCH_FALCON_0);
      mme_mthd(b, NV9097_NO_OPERATION);
      mme_emit(b, mme_zero());
   };
}

/* nvk_mme_set_priv_reg (nvk_cmd_draw.c:100-107). */
static void
nvrm_mme_set_priv_reg(struct mme_builder *b)
{
   struct mme_value value = mme_load(b);
   struct mme_value mask = mme_load(b);
   struct mme_value reg = mme_load(b);
   nvrm_mme_set_priv_reg_raw(b, value, mask, reg);
}

static void (*const nvrm_mme_builders[NVRM_MME_COUNT])(struct mme_builder *) = {
   [NVRM_MME_DRAW_INDIRECT] = nvrm_mme_draw_indirect,
   [NVRM_MME_DRAW_INDEXED_INDIRECT] = nvrm_mme_draw_indexed_indirect,
   [NVRM_MME_DISPATCH_INDIRECT] = nvrm_mme_dispatch_indirect,
   [NVRM_MME_SET_PRIV_REG] = nvrm_mme_set_priv_reg,
   [NVRM_MME_DRAW] = nvrm_mme_draw,
   [NVRM_MME_DRAW_INDEXED] = nvrm_mme_draw_indexed,
};

/* Registers as NVK's get_sm_disp_ctrl_reg /
 * get_sms_hww_warp_esp_report_mask_reg (nvk_cmd_draw.c:46-64). */
static uint32_t
nvrm_sm_disp_ctrl_reg(const struct nv_device_info *info)
{
   if (info->cls_eng3d >= HOPPER_A)
      return 0x4243a4;
   if (info->cls_eng3d >= VOLTA_A)
      return 0x419ba4;
   return 0x419f78;
}

static uint32_t
nvrm_warp_esr_report_mask_reg(const struct nv_device_info *info)
{
   if (info->cls_eng3d >= HOPPER_A)
      return 0x4246a8;
   if (info->cls_eng3d >= VOLTA_A)
      return 0x419ea8;
   return 0x419e44;
}

/* MME DMA reads (mme_tu104_read_fifoed) exist from TURING_A; older classes
 * take indirect data as push data. */
bool
nvrm_mme_dma(const struct nvrm_screen *screen)
{
   return screen->info.cls_eng3d >= TURING_A;
}

/* nvk_push_draw_state_init's upload loop (nvk_cmd_draw.c:152-172). */
void
nvrm_push_mme_init(const struct nvrm_screen *screen, struct nv_push *p)
{
   /* Every class gets every macro (NVK uploads its set unconditionally);
    * the builder picks the Fermi-style or TU104 MME ISA by class. */
   for (uint32_t mme = 0, mme_pos = 0; mme < NVRM_MME_COUNT; mme++) {
      struct mme_builder b;
      mme_builder_init(&b, &screen->info);
      nvrm_mme_builders[mme](&b);
      size_t size;
      uint32_t *dw = mme_builder_finish(&b, &size);
      if (!dw) {
         nvrm_error("MME macro %u build failed\n", mme);
         abort();
      }
      const uint32_t num_dw = size / sizeof(uint32_t);
      P_MTHD(p, NV9097, LOAD_MME_START_ADDRESS_RAM_POINTER);
      P_NV9097_LOAD_MME_START_ADDRESS_RAM_POINTER(p, mme);
      P_NV9097_LOAD_MME_START_ADDRESS_RAM(p, mme_pos);
      P_1INC(p, NV9097, LOAD_MME_INSTRUCTION_RAM_POINTER);
      P_NV9097_LOAD_MME_INSTRUCTION_RAM_POINTER(p, mme_pos);
      P_INLINE_ARRAY(p, dw, num_dw);
      mme_pos += num_dw;
      free(dw);
   }

   /* nvk_push_draw_state_init (nvk_cmd_draw.c:177-252), MAXWELL_B+:
    * clear gr_gpcs_tpcs_sm_disp_ctrl bit 3 (memory loads in fragment
    * helper invocations: without it derivatives of SSBO loads are wrong on
    * partially covered quads, measured 10-15: 32 edge pixels of a
    * dFdx/dFdy(ssbo value) triangle differ from stock 615 and llvmpipe,
    * which agree) and gr_gpcs_tpcs_sms_hww_warp_esr_report_mask bit 14
    * (out-of-range output writes are ignored instead of an exception that
    * kills the context). Stock GL's captured 3D init has no such method;
    * how stock sets them is not known. */
   if (screen->info.cls_eng3d < MAXWELL_B)
      return;
   P_1INC(p, NV9097, CALL_MME_MACRO(NVRM_MME_SET_PRIV_REG));
   P_INLINE_DATA(p, 0);
   P_INLINE_DATA(p, BITFIELD_BIT(3));
   P_INLINE_DATA(p, nvrm_sm_disp_ctrl_reg(&screen->info));
   P_1INC(p, NV9097, CALL_MME_MACRO(NVRM_MME_SET_PRIV_REG));
   P_INLINE_DATA(p, 0);
   P_INLINE_DATA(p, BITFIELD_BIT(14));
   P_INLINE_DATA(p, nvrm_warp_esr_report_mask_reg(&screen->info));
}
