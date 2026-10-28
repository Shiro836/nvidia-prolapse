/* SPDX-License-Identifier: MIT
 *
 * nvrm G1 textures:
 *  - tiled (block-linear) VRAM images laid out by nil (nil_image_init), with
 *    the PTE kind nil chose, as NVK binds image planes (nvk_image.c +
 *    nvkmd_rm.c rm_va_bind);
 *  - copy-engine transfers between such an image and a linear GART staging
 *    buffer, following NVK's nouveau_copy_rect (nvk_cmd_copy.c:186-370);
 *  - texture headers (TIC) from nil_image_view_descriptor and sampler
 *    headers (TSC) packed as nvk_sampler_get_header (nvk_sampler.c:140-290)
 *    from gallium sampler state;
 *  - a driver-owned header pool + sampler pool per context bound with
 *    SET_TEX_HEADER_POOL / SET_TEX_SAMPLER_POOL (NVK nvk_queue.c:88-130; stock
 *    GL does the same at init); a (TIC, TSC) pair is written once per batch
 *    at first use and the header/sampler caches invalidated once after such
 *    writes (as stock does);
 *  - NIR lowering of GL texture/sampler units to NAK texture handles: the
 *    handle (header index | sampler index << 20) is read from the driver
 *    constant buffer (slot NVRM_TEX_CBUF_SLOT, changed handles loaded with
 *    LOAD_CONSTANT_BUFFER), the same ldc_nv form NVK's
 *    descriptor lowering produces for bound descriptors, which NAK folds
 *    into a cbuf texture reference (nak_nir_lower_tex.c tex_handle_as_cbuf).
 * Header slot i always pairs with sampler slot i, so the binding is correct
 * whether the hardware takes the sampler index from the handle or via the
 * header binding (SET_SAMPLER_BINDING VIA_HEADER_BINDING, as stock GL).
 */
#include <inttypes.h>
#include <string.h>

#include "nvrm_priv.h"

#include "nil.h"
#include "nv_push.h"
#include "nv_push_cl9097.h"
#include "nv_push_cla097.h"
#include "nv_push_cl90b5.h"
#include "nv_push_clc1b5.h"
#include "nv_push_clcab5.h"
#include "cl9097tex.h"
#include "cla097tex.h"
#include "clb197tex.h"
#include "clc1b5.h"
#include "clcab5.h"
#include "clb197.h"
#include "drf.h"

#include "compiler/nir/nir_builder.h"
#include "util/bitpack_helpers.h"
#include "util/format/u_format.h"
#include "util/format/format_utils.h"
#include "util/format_srgb.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_sample_positions.h"
#include "util/u_upload_mgr.h"

/* ---------------------------------------------------------------- copies */

static uint32_t
nvrm_cab5_gob_kind(enum nil_gob_type gob_type)
{
   /* nil_to_nvcab5_gob_type (Mesa 26.2 nvk_cmd_copy.c:175-187) */
   switch (gob_type) {
   case NIL_GOB_TYPE_TURING_COLOR2_D:  return NVCAB5_SET_SRC_BLOCK_SIZE_KIND_BPP_BL_32;
   case NIL_GOB_TYPE_BLACKWELL8_BIT:   return NVCAB5_SET_SRC_BLOCK_SIZE_KIND_BPP_BL_8;
   case NIL_GOB_TYPE_BLACKWELL16_BIT:  return NVCAB5_SET_SRC_BLOCK_SIZE_KIND_BPP_BL_16;
   case NIL_GOB_TYPE_BLACKWELL_Z24:    return NVCAB5_SET_SRC_BLOCK_SIZE_KIND_BPP_BL_24;
   default:
      nvrm_error("GOB type %u has no Blackwell copy-engine kind\n", gob_type);
      abort();
   }
}

/* One copy-engine rectangle per layer/slice between level `level` of a
 * tiled image and a linear buffer (box in pixels, linear side packed with
 * the given strides). Single-plane, single-sample, no component remap:
 * NVK's nouveau_copy_rect with remap.comp_size == 0, where X units are
 * bytes. Emitted into the context push; the caller makes it synchronous. */
void
nvrm_copy_image_linear(struct nvrm_context *ctx, struct nvrm_resource *res,
                       unsigned level, const struct pipe_box *box,
                       uint64_t linear_va, uint32_t linear_stride,
                       uint64_t linear_layer_stride, bool to_image)
{
   struct nvrm_screen *screen = nvrm_screen(ctx->base.screen);
   const struct nil_image *img = &res->image;
   const struct nil_image_level *lvl = &img->levels[level];
   const enum pipe_format format = res->base.format;
   const uint32_t bpp = util_format_get_blocksize(format);
   const struct nil_Extent4D_Pixels lext = nil_image_level_extent_px(img, level);
   const uint32_t lext_h_el = util_format_get_nblocksy(format, lext.height);
   const uint32_t x_el = box->x / util_format_get_blockwidth(format);
   const uint32_t y_el = box->y / util_format_get_blockheight(format);
   const uint32_t w_el = util_format_get_nblocksx(format, box->width);
   const uint32_t h_el = util_format_get_nblocksy(format, box->height);
   const bool is_3d = img->dim == NIL_IMAGE_DIM_3D;
   const uint16_t cls_copy = screen->info.cls_copy;

   if (lvl->tiling.gob_type == NIL_GOB_TYPE_LINEAR) {
      nvrm_error("copy: linear-tiled image level reached the tiled copy path\n");
      abort();
   }

   for (int z = 0; z < box->depth; z++) {
      uint64_t img_addr = res->bo.va + lvl->offset_B;
      if (!is_3d)
         img_addr += (uint64_t)(box->z + z) * img->array_stride_B;
      const uint64_t lin_addr = linear_va + (uint64_t)z * linear_layer_stride;

      struct nv_push push = nvrm_push_begin(ctx, 40);
      struct nv_push *p = &push;

      P_MTHD(p, NV90B5, OFFSET_IN_UPPER);
      P_NV90B5_OFFSET_IN_UPPER(p, (to_image ? lin_addr : img_addr) >> 32);
      P_NV90B5_OFFSET_IN_LOWER(p, (to_image ? lin_addr : img_addr) & 0xffffffff);
      P_NV90B5_OFFSET_OUT_UPPER(p, (to_image ? img_addr : lin_addr) >> 32);
      P_NV90B5_OFFSET_OUT_LOWER(p, (to_image ? img_addr : lin_addr) & 0xffffffff);
      P_NV90B5_PITCH_IN(p, to_image ? linear_stride : lvl->row_stride_B);
      P_NV90B5_PITCH_OUT(p, to_image ? lvl->row_stride_B : linear_stride);
      P_NV90B5_LINE_LENGTH_IN(p, w_el * bpp);
      P_NV90B5_LINE_COUNT(p, h_el);

      /* Tiled side (source when downloading, destination when uploading). */
      if (to_image) {
         P_MTHD(p, NV90B5, SET_DST_BLOCK_SIZE);
         if (cls_copy >= BLACKWELL_DMA_COPY_B) { /* nvk_cmd_copy.c:265 */
            P_NVCAB5_SET_DST_BLOCK_SIZE(p, {
               .width = 0,
               .height = lvl->tiling.y_log2,
               .depth = lvl->tiling.z_log2,
               .gob_height = GOB_HEIGHT_GOB_HEIGHT_FERMI_8,
               .kind_bpp = nvrm_cab5_gob_kind(lvl->tiling.gob_type),
            });
         } else {
            P_NV90B5_SET_DST_BLOCK_SIZE(p, {
               .width = 0,
               .height = lvl->tiling.y_log2,
               .depth = lvl->tiling.z_log2,
               .gob_height = GOB_HEIGHT_GOB_HEIGHT_FERMI_8,
            });
         }
         P_NV90B5_SET_DST_WIDTH(p, lvl->row_stride_B);
         P_NV90B5_SET_DST_HEIGHT(p, lext_h_el);
         P_NV90B5_SET_DST_DEPTH(p, is_3d ? lext.depth : 1);
         P_NV90B5_SET_DST_LAYER(p, is_3d ? box->z + z : 0);
         if (cls_copy >= PASCAL_DMA_COPY_B) {
            P_MTHD(p, NVC1B5, DST_ORIGIN_X);
            P_NVC1B5_DST_ORIGIN_X(p, x_el * bpp);
            P_NVC1B5_DST_ORIGIN_Y(p, y_el);
         } else {
            P_MTHD(p, NV90B5, SET_DST_ORIGIN);
            P_NV90B5_SET_DST_ORIGIN(p, { .x = x_el * bpp, .y = y_el });
         }
      } else {
         P_MTHD(p, NV90B5, SET_SRC_BLOCK_SIZE);
         if (cls_copy >= BLACKWELL_DMA_COPY_B) { /* nvk_cmd_copy.c:265 */
            P_NVCAB5_SET_SRC_BLOCK_SIZE(p, {
               .width = 0,
               .height = lvl->tiling.y_log2,
               .depth = lvl->tiling.z_log2,
               .gob_height = GOB_HEIGHT_GOB_HEIGHT_FERMI_8,
               .kind_bpp = nvrm_cab5_gob_kind(lvl->tiling.gob_type),
            });
         } else {
            P_NV90B5_SET_SRC_BLOCK_SIZE(p, {
               .width = 0,
               .height = lvl->tiling.y_log2,
               .depth = lvl->tiling.z_log2,
               .gob_height = GOB_HEIGHT_GOB_HEIGHT_FERMI_8,
            });
         }
         P_NV90B5_SET_SRC_WIDTH(p, lvl->row_stride_B);
         P_NV90B5_SET_SRC_HEIGHT(p, lext_h_el);
         P_NV90B5_SET_SRC_DEPTH(p, is_3d ? lext.depth : 1);
         P_NV90B5_SET_SRC_LAYER(p, is_3d ? box->z + z : 0);
         if (cls_copy >= PASCAL_DMA_COPY_B) {
            P_MTHD(p, NVC1B5, SRC_ORIGIN_X);
            P_NVC1B5_SRC_ORIGIN_X(p, x_el * bpp);
            P_NVC1B5_SRC_ORIGIN_Y(p, y_el);
         } else {
            P_MTHD(p, NV90B5, SET_SRC_ORIGIN);
            P_NV90B5_SET_SRC_ORIGIN(p, { .x = x_el * bpp, .y = y_el });
         }
      }

      P_IMMD(p, NV90B5, LAUNCH_DMA, {
         .data_transfer_type = DATA_TRANSFER_TYPE_NON_PIPELINED,
         .multi_line_enable = MULTI_LINE_ENABLE_TRUE,
         .flush_enable = FLUSH_ENABLE_TRUE,
         .src_memory_layout = to_image ? SRC_MEMORY_LAYOUT_PITCH
                                       : SRC_MEMORY_LAYOUT_BLOCKLINEAR,
         .dst_memory_layout = to_image ? DST_MEMORY_LAYOUT_BLOCKLINEAR
                                       : DST_MEMORY_LAYOUT_PITCH,
      });
      nvrm_push_end(ctx, p);
   }
}

void
nvrm_emit_copy_buffer(struct nv_push *p, uint64_t dst, uint64_t src, uint64_t size_B)
{
   while (size_B) {
      const uint32_t bytes = (uint32_t)MIN2(size_B, 1u << 17);
      P_MTHD(p, NV90B5, OFFSET_IN_UPPER);
      P_NV90B5_OFFSET_IN_UPPER(p, src >> 32);
      P_NV90B5_OFFSET_IN_LOWER(p, src & 0xffffffff);
      P_NV90B5_OFFSET_OUT_UPPER(p, dst >> 32);
      P_NV90B5_OFFSET_OUT_LOWER(p, dst & 0xffffffff);
      P_MTHD(p, NV90B5, LINE_LENGTH_IN);
      P_NV90B5_LINE_LENGTH_IN(p, bytes);
      P_NV90B5_LINE_COUNT(p, 1);
      P_IMMD(p, NV90B5, LAUNCH_DMA, {
         .data_transfer_type = DATA_TRANSFER_TYPE_NON_PIPELINED,
         .multi_line_enable = MULTI_LINE_ENABLE_TRUE,
         .flush_enable = FLUSH_ENABLE_TRUE,
         .src_memory_layout = SRC_MEMORY_LAYOUT_PITCH,
         .dst_memory_layout = DST_MEMORY_LAYOUT_PITCH,
      });
      src += bytes;
      dst += bytes;
      size_B -= bytes;
   }
}

void
nvrm_push_copy_buffer(struct nvrm_context *ctx, uint64_t dst, uint64_t src, uint64_t size_B)
{
   /* One push per 1 MiB so a large copy never exceeds the push buffer. */
   while (size_B) {
      const uint64_t n = MIN2(size_B, 8u << 17);
      struct nv_push push = nvrm_push_begin(ctx, nvrm_copy_buffer_dw(n));
      nvrm_emit_copy_buffer(&push, dst, src, n);
      nvrm_push_end(ctx, &push);
      dst += n;
      src += n;
      size_B -= n;
   }
}

/* NVK's RENDER_WFI before a copy-engine write (nvk_barrier_flushes_waits,
 * nvk_cmd_buffer.c:440-475: colour/depth attachment and transform-feedback
 * writes -> 3D WAIT_FOR_IDLE): the 3D pipe drains its pending ROP / stream
 * output writes before the copy that follows in the channel. */
void
nvrm_push_render_wfi(struct nvrm_context *ctx)
{
   struct nv_push push = nvrm_push_begin(ctx, 2);
   P_IMMD(&push, NV9097, WAIT_FOR_IDLE, 0);
   nvrm_push_end(ctx, &push);
}

/* Staging -> image upload recorded into the batch, no flush or wait (the
 * texture side of nvc0's/NVK's staged transfers): earlier rendering into
 * the image drains first (render WFI), the copy runs in channel order
 * before later draws, which invalidate the texture data cache first
 * (NVRM_INVAL_TEX_DATA; compute launches always invalidate). The batch keeps both resources alive until it is
 * submitted; frees after that are deferred by the backend until it retires. */
void
nvrm_copy_image_upload(struct nvrm_context *ctx, struct nvrm_resource *res,
                       struct pipe_resource *keep_res, unsigned level,
                       const struct pipe_box *box, struct pipe_resource *staging,
                       uint32_t linear_stride, uint64_t linear_layer_stride)
{
   nvrm_keep(ctx, keep_res);
   nvrm_keep(ctx, staging);
   nvrm_push_render_wfi(ctx);
   nvrm_copy_image_linear(ctx, res, level, box, nvrm_resource(staging)->bo.va, linear_stride,
                          linear_layer_stride, true);
   ctx->inval |= NVRM_INVAL_TEX_DATA;
}

/* Download (and D1 shadow) copies: the copy runs in its own batch after
 * everything the context queued before, and the CPU waits for it. */
void
nvrm_copy_image_linear_sync(struct nvrm_context *ctx, struct nvrm_resource *res,
                            unsigned level, const struct pipe_box *box,
                            uint64_t linear_va, uint32_t linear_stride,
                            uint64_t linear_layer_stride, bool to_image)
{
   ctx->base.flush(&ctx->base, NULL, 0);
   /* Earlier rendering into the image (previous batches may still be in
    * the 3D pipe) drains before the copy engine reads or writes it. */
   nvrm_push_render_wfi(ctx);
   nvrm_copy_image_linear(ctx, res, level, box, linear_va, linear_stride,
                          linear_layer_stride, to_image);
   ctx->base.flush(&ctx->base, NULL, 0);
   if (!nvrm_wait_token(nvrm_screen(ctx->base.screen), ctx->last_token)) {
      nvrm_error("copy-engine image transfer did not complete\n");
      abort();
   }
}

/* ---------------------------------------------------------------- TSC */

#define SAMP_SET_U(o, NV, i, FIELD, val) \
   (o)[i] |= util_bitpack_uint((val), DRF_LO(NV##_TEXSAMP##i##_##FIELD) % 32, \
                               DRF_HI(NV##_TEXSAMP##i##_##FIELD) % 32)
#define SAMP_SET_UF(o, NV, i, FIELD, val) \
   (o)[i] |= util_bitpack_ufixed_clamp((val), DRF_LO(NV##_TEXSAMP##i##_##FIELD) % 32, \
                                       DRF_HI(NV##_TEXSAMP##i##_##FIELD) % 32, 8)
#define SAMP_SET_SF(o, NV, i, FIELD, val) \
   (o)[i] |= util_bitpack_sfixed_clamp((val), DRF_LO(NV##_TEXSAMP##i##_##FIELD) % 32, \
                                       DRF_HI(NV##_TEXSAMP##i##_##FIELD) % 32, 8)
#define SAMP_SET_E(o, NV, i, FIELD, E) \
   SAMP_SET_U(o, NV, i, FIELD, NV##_TEXSAMP##i##_##FIELD##_##E)

static uint32_t
nvrm_wrap(unsigned wrap)
{
   switch (wrap) {
   case PIPE_TEX_WRAP_REPEAT:                 return NV9097_TEXSAMP0_ADDRESS_U_WRAP;
   case PIPE_TEX_WRAP_MIRROR_REPEAT:          return NV9097_TEXSAMP0_ADDRESS_U_MIRROR;
   case PIPE_TEX_WRAP_CLAMP_TO_EDGE:          return NV9097_TEXSAMP0_ADDRESS_U_CLAMP_TO_EDGE;
   case PIPE_TEX_WRAP_CLAMP_TO_BORDER:        return NV9097_TEXSAMP0_ADDRESS_U_BORDER;
   case PIPE_TEX_WRAP_CLAMP:                  return NV9097_TEXSAMP0_ADDRESS_U_CLAMP_OGL;
   case PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE:   return NV9097_TEXSAMP0_ADDRESS_U_MIRROR_ONCE_CLAMP_TO_EDGE;
   case PIPE_TEX_WRAP_MIRROR_CLAMP_TO_BORDER: return NV9097_TEXSAMP0_ADDRESS_U_MIRROR_ONCE_BORDER;
   case PIPE_TEX_WRAP_MIRROR_CLAMP:           return NV9097_TEXSAMP0_ADDRESS_U_MIRROR_ONCE_CLAMP_OGL;
   default:
      nvrm_error("texture wrap mode %u unknown\n", wrap);
      abort();
   }
}

/* nvk_sampler_get_header (nvk_sampler.c:140-290) over pipe_sampler_state. */
static void
nvrm_pack_tsc(const struct nvrm_screen *screen, const struct pipe_sampler_state *s,
              uint32_t out[8])
{
   memset(out, 0, 8 * sizeof(uint32_t));
   SAMP_SET_U(out, NV9097, 0, ADDRESS_U, nvrm_wrap(s->wrap_s));
   SAMP_SET_U(out, NV9097, 0, ADDRESS_V, nvrm_wrap(s->wrap_t));
   SAMP_SET_U(out, NV9097, 0, ADDRESS_P, nvrm_wrap(s->wrap_r));
   if (s->compare_mode == PIPE_TEX_COMPARE_R_TO_TEXTURE) {
      SAMP_SET_U(out, NV9097, 0, DEPTH_COMPARE, 1);
      /* PIPE_FUNC_* has the ZC_* order (NEVER..ALWAYS = 0..7). */
      SAMP_SET_U(out, NV9097, 0, DEPTH_COMPARE_FUNC, s->compare_func);
   }
   SAMP_SET_U(out, NV9097, 0, S_R_G_B_CONVERSION, 1);
   SAMP_SET_E(out, NV9097, 0, FONT_FILTER_WIDTH, SIZE_2);
   SAMP_SET_E(out, NV9097, 0, FONT_FILTER_HEIGHT, SIZE_2);

   const bool aniso = s->max_anisotropy > 1;
   if (aniso) {
      uint32_t a = s->max_anisotropy >= 16 ? NV9097_TEXSAMP0_MAX_ANISOTROPY_ANISO_16_TO_1 :
                   s->max_anisotropy >= 12 ? NV9097_TEXSAMP0_MAX_ANISOTROPY_ANISO_12_TO_1 :
                   s->max_anisotropy >> 1;
      SAMP_SET_U(out, NV9097, 0, MAX_ANISOTROPY, a);
   }

   if (s->mag_img_filter == PIPE_TEX_FILTER_LINEAR)
      SAMP_SET_E(out, NV9097, 1, MAG_FILTER, MAG_LINEAR);
   else
      SAMP_SET_E(out, NV9097, 1, MAG_FILTER, MAG_POINT);

   if (s->min_img_filter == PIPE_TEX_FILTER_LINEAR) {
      if (aniso)
         SAMP_SET_E(out, NV9097, 1, MIN_FILTER, MIN_ANISO);
      else
         SAMP_SET_E(out, NV9097, 1, MIN_FILTER, MIN_LINEAR);
   } else {
      SAMP_SET_E(out, NV9097, 1, MIN_FILTER, MIN_POINT);
   }

   switch (s->min_mip_filter) {
   case PIPE_TEX_MIPFILTER_NEAREST: SAMP_SET_E(out, NV9097, 1, MIP_FILTER, MIP_POINT); break;
   case PIPE_TEX_MIPFILTER_LINEAR:  SAMP_SET_E(out, NV9097, 1, MIP_FILTER, MIP_LINEAR); break;
   default:                         SAMP_SET_E(out, NV9097, 1, MIP_FILTER, MIP_NONE); break;
   }

   if (s->seamless_cube_map)
      SAMP_SET_E(out, NVA097, 1, CUBEMAP_INTERFACE_FILTERING, AUTO_SPAN_SEAM);
   else
      SAMP_SET_E(out, NVA097, 1, CUBEMAP_INTERFACE_FILTERING, USE_WRAP);

   if (screen->info.cls_eng3d >= MAXWELL_B) {
      switch (s->reduction_mode) {
      case PIPE_TEX_REDUCTION_MIN: SAMP_SET_E(out, NVB197, 1, REDUCTION_FILTER, RED_MINIMUM); break;
      case PIPE_TEX_REDUCTION_MAX: SAMP_SET_E(out, NVB197, 1, REDUCTION_FILTER, RED_MAXIMUM); break;
      default:                     SAMP_SET_E(out, NVB197, 1, REDUCTION_FILTER, RED_NONE); break;
      }
   }

   SAMP_SET_SF(out, NV9097, 1, MIP_LOD_BIAS, s->lod_bias);
   if (s->unnormalized_coords)
      SAMP_SET_E(out, NVA097, 1, FLOAT_COORD_NORMALIZATION, FORCE_UNNORMALIZED_COORDS);
   else
      SAMP_SET_E(out, NVA097, 1, FLOAT_COORD_NORMALIZATION, USE_HEADER_SETTING);
   const uint32_t trilin = s->max_anisotropy >= 12 ? 0 : s->max_anisotropy >= 4 ? 6 :
                           s->max_anisotropy >= 2 ? 4 : 0;
   SAMP_SET_U(out, NV9097, 1, TRILIN_OPT, trilin);

   /* LOD clamps as given, also with MIP_NONE (which samples the base level
    * by itself): the computed LOD still selects min vs mag filtering, so
    * clamping it to 0 would magnify everything (nv50_sampler_state_create
    * keeps the clamps the same way). */
   SAMP_SET_UF(out, NV9097, 2, MIN_LOD_CLAMP, s->min_lod);
   SAMP_SET_UF(out, NV9097, 2, MAX_LOD_CLAMP, s->max_lod);

   /* Border colour: st hands it already swizzled by the unit's texture
    * swizzle (PIPE_QUIRK_TEXTURE_BORDER_COLOR_SWIZZLE_NV50, nvrm_screen.c);
    * the sRGB copy is the linear->sRGB encoding of the swizzled value, as
    * NVK's non-sRGB-view branch (nvk_sampler.c:263-275). */
   uint8_t srgb[3];
   for (unsigned i = 0; i < 3; i++)
      srgb[i] = util_format_linear_float_to_srgb_8unorm(s->border_color.f[i]);
   SAMP_SET_U(out, NV9097, 2, S_R_G_B_BORDER_COLOR_R, srgb[0]);
   SAMP_SET_U(out, NV9097, 3, S_R_G_B_BORDER_COLOR_G, srgb[1]);
   SAMP_SET_U(out, NV9097, 3, S_R_G_B_BORDER_COLOR_B, srgb[2]);
   SAMP_SET_U(out, NV9097, 4, BORDER_COLOR_R, s->border_color.ui[0]);
   SAMP_SET_U(out, NV9097, 5, BORDER_COLOR_G, s->border_color.ui[1]);
   SAMP_SET_U(out, NV9097, 6, BORDER_COLOR_B, s->border_color.ui[2]);
   SAMP_SET_U(out, NV9097, 7, BORDER_COLOR_A, s->border_color.ui[3]);
}

/* NVK's sRGB-view border (nvk_sampler.c:254-270): R/G/B are sRGB-encoded
 * BEFORE the swizzle, so an output channel fed from alpha carries alpha
 * unencoded in the sRGB border field. st already swizzled the border
 * (texture_border_color_quirk); for an sRGB view whose swizzle reads W
 * into channel c, that channel's sRGB field is unorm8(border[c]) instead of
 * the encoding. tsc_srgb_alpha[c] = tsc with channel c's field rewritten
 * that way; the fields are independent, so emission merges per channel. */
#define SAMP_CLEAR(o, NV, i, FIELD) \
   (o)[i] &= ~util_bitpack_ones(DRF_LO(NV##_TEXSAMP##i##_##FIELD) % 32, \
                                DRF_HI(NV##_TEXSAMP##i##_##FIELD) % 32)

static void *
nvrm_create_sampler_state(struct pipe_context *pctx, const struct pipe_sampler_state *s)
{
   struct nvrm_sampler *samp = CALLOC_STRUCT(nvrm_sampler);
   if (!samp)
      return NULL;
   nvrm_pack_tsc(nvrm_screen(pctx->screen), s, samp->tsc);
   for (unsigned c = 0; c < 3; c++) {
      uint32_t *t = samp->tsc_srgb_alpha[c];
      memcpy(t, samp->tsc, sizeof(samp->tsc));
      const uint8_t raw = _mesa_float_to_unorm(s->border_color.f[c], 8);
      if (c == 0) {
         SAMP_CLEAR(t, NV9097, 2, S_R_G_B_BORDER_COLOR_R);
         SAMP_SET_U(t, NV9097, 2, S_R_G_B_BORDER_COLOR_R, raw);
      } else if (c == 1) {
         SAMP_CLEAR(t, NV9097, 3, S_R_G_B_BORDER_COLOR_G);
         SAMP_SET_U(t, NV9097, 3, S_R_G_B_BORDER_COLOR_G, raw);
      } else {
         SAMP_CLEAR(t, NV9097, 3, S_R_G_B_BORDER_COLOR_B);
         SAMP_SET_U(t, NV9097, 3, S_R_G_B_BORDER_COLOR_B, raw);
      }
   }
   return samp;
}

/* The sampler words for unit pairing (header slot i <-> sampler slot i):
 * the plain TSC unless the view is sRGB and its swizzle feeds alpha into
 * R/G/B, then those channels' sRGB border fields as NVK writes them. */
static void
nvrm_tsc_for_view(const struct nvrm_sampler *samp, const struct pipe_sampler_view *v,
                  uint32_t out[8])
{
   memcpy(out, samp->tsc, 32);
   if (!v || !util_format_is_srgb(v->format))
      return;
   const unsigned char swz[3] = { v->swizzle_r, v->swizzle_g, v->swizzle_b };
   for (unsigned c = 0; c < 3; c++) {
      if (swz[c] != PIPE_SWIZZLE_W)
         continue;
      /* Word 2 holds R, word 3 holds G and B (the other fields of those
       * words are identical in every variant). */
      if (c == 0) {
         out[2] = samp->tsc_srgb_alpha[0][2];
      } else {
         const uint32_t mask = c == 1 ?
            util_bitpack_ones(DRF_LO(NV9097_TEXSAMP3_S_R_G_B_BORDER_COLOR_G) % 32,
                              DRF_HI(NV9097_TEXSAMP3_S_R_G_B_BORDER_COLOR_G) % 32) :
            util_bitpack_ones(DRF_LO(NV9097_TEXSAMP3_S_R_G_B_BORDER_COLOR_B) % 32,
                              DRF_HI(NV9097_TEXSAMP3_S_R_G_B_BORDER_COLOR_B) % 32);
         out[3] = (out[3] & ~mask) | (samp->tsc_srgb_alpha[c][3] & mask);
      }
   }
}

static void
nvrm_bind_sampler_states(struct pipe_context *pctx, mesa_shader_stage shader,
                         unsigned start, unsigned count, void **states)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   if (start + count > NVRM_MAX_TEX_UNITS) {
      nvrm_error("sampler units %u..%u exceed %u\n", start, start + count, NVRM_MAX_TEX_UNITS);
      abort();
   }
   for (unsigned i = 0; i < count; i++)
      ctx->samplers[shader][start + i] = states ? states[i] : NULL;
   ctx->dirty |= NVRM_DIRTY_TEXTURES;
}

/* A deleted sampler state may still sit in a unit the frontend did not
 * rebind (cso_cache eviction protects only the frontend's pending set,
 * cso_context.c sanitize_hash), and nvrm_tex_handles reads every unit up to
 * the shader's last one, holes included: forget it everywhere first, as
 * nvc0_sampler_state_delete (nvc0_state.c:445-457) does, so no draw packs a
 * TSC from freed memory (NULL = default sampler). */
static void
nvrm_delete_sampler_state(struct pipe_context *pctx, void *s)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   for (unsigned st = 0; st < MESA_SHADER_STAGES; st++)
      for (unsigned i = 0; i < NVRM_MAX_TEX_UNITS; i++)
         if (ctx->samplers[st][i] == s)
            ctx->samplers[st][i] = NULL;
   FREE(s);
}

/* ---------------------------------------------------------------- TIC */

static nil_view_type
nvrm_view_type(enum pipe_texture_target target)
{
   switch (target) {
   case PIPE_TEXTURE_1D:         return NIL_VIEW_TYPE_1D;
   case PIPE_TEXTURE_1D_ARRAY:   return NIL_VIEW_TYPE_1D_ARRAY;
   case PIPE_TEXTURE_2D:
   case PIPE_TEXTURE_RECT:       return NIL_VIEW_TYPE_2D;
   case PIPE_TEXTURE_2D_ARRAY:   return NIL_VIEW_TYPE_2D_ARRAY;
   case PIPE_TEXTURE_3D:         return NIL_VIEW_TYPE_3D;
   case PIPE_TEXTURE_CUBE:       return NIL_VIEW_TYPE_CUBE;
   case PIPE_TEXTURE_CUBE_ARRAY: return NIL_VIEW_TYPE_CUBE_ARRAY;
   default:
      nvrm_error("sampler view target %u is not implemented (G1 textures)\n", target);
      abort();
   }
}

static struct pipe_sampler_view *
nvrm_create_sampler_view(struct pipe_context *pctx, struct pipe_resource *pres,
                         const struct pipe_sampler_view *templ)
{
   struct nvrm_screen *screen = nvrm_screen(pctx->screen);
   struct nvrm_resource *res = nvrm_resource(pres);
   if (pres->target == PIPE_BUFFER || (!res->tiled && !res->linear_image)) {
      nvrm_error("sampler view of a %s resource is not implemented (G1 textures: "
                 "tiled images only)\n", pres->target == PIPE_BUFFER ? "buffer" : "linear");
      abort();
   }
   if (!nil_format_supports_texturing(&screen->info, templ->format)) {
      nvrm_error("sampler view format %s is not texturable (nil)\n",
                 util_format_short_name(templ->format));
      abort();
   }
   struct nvrm_sampler_view *view = CALLOC_STRUCT(nvrm_sampler_view);
   if (!view)
      return NULL;
   view->base = *templ;
   view->base.reference.count = 1;
   view->base.texture = NULL;
   pipe_resource_reference(&view->base.texture, pres);
   view->base.context = pctx;

   const enum pipe_texture_target target = templ->target;
   struct nil_view nv = {
      .view_type = nvrm_view_type(target),
      .format = nil_format(templ->format),
      .base_level = templ->u.tex.first_level,
      .num_levels = templ->u.tex.last_level - templ->u.tex.first_level + 1,
      .base_array_layer = target == PIPE_TEXTURE_3D ? 0 : templ->u.tex.first_layer,
      .array_len = target == PIPE_TEXTURE_3D ? 1 :
                   templ->u.tex.last_layer - templ->u.tex.first_layer + 1,
      .swizzle = { templ->swizzle_r, templ->swizzle_g, templ->swizzle_b, templ->swizzle_a },
      .min_lod_clamp = 0.0f,
   };
   /* Stencil views of a Blackwell separate Z/S resource read the S8 plane
    * (NVK picks the stencil plane for VK_IMAGE_ASPECT_STENCIL_BIT views). */
   const struct util_format_description *vd = util_format_description(templ->format);
   const bool stencil_view = res->separate_zs && util_format_has_stencil(vd) &&
                             !util_format_has_depth(vd);
   if (stencil_view)
      nv.format = nil_format(PIPE_FORMAT_S8_UINT);
   else if (res->separate_zs && util_format_has_depth(vd))
      /* Depth views of the separate plane use its depth-only format
       * (Z32_FLOAT for Z32_FLOAT_S8X24_UINT; nil asserts equal sizes). */
      nv.format = nil_format(util_format_get_depth_only(templ->format));
   struct nil_descriptor d =
      nil_image_view_descriptor(&screen->info, stencil_view ? &res->stencil_image : &res->image,
                                &nv, stencil_view ? res->stencil_bo.va :
                                res->bo.va + res->plane_offset_B);
   memcpy(view->tic, d.bits, sizeof(view->tic));
   return &view->base;
}

static void
nvrm_sampler_view_destroy(struct pipe_context *pctx, struct pipe_sampler_view *view)
{
   pipe_resource_reference(&view->texture, NULL);
   FREE(view);
}

static void
nvrm_set_sampler_views(struct pipe_context *pctx, mesa_shader_stage shader,
                       unsigned start, unsigned num, unsigned unbind_trailing,
                       struct pipe_sampler_view **views)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   if (start + num + unbind_trailing > NVRM_MAX_TEX_UNITS) {
      nvrm_error("sampler view units %u..%u exceed %u\n", start,
                 start + num + unbind_trailing, NVRM_MAX_TEX_UNITS);
      abort();
   }
   for (unsigned i = 0; i < num; i++)
      pipe_sampler_view_reference(&ctx->views[shader][start + i], views ? views[i] : NULL);
   for (unsigned i = 0; i < unbind_trailing; i++)
      pipe_sampler_view_reference(&ctx->views[shader][start + num + i], NULL);
   ctx->dirty |= NVRM_DIRTY_TEXTURES;
}

/* ---------------------------------------------------------------- pools */

bool
nvrm_texture_context_init(struct nvrm_context *ctx)
{
   struct nvrm_screen *screen = nvrm_screen(ctx->base.screen);
   /* TIC pool then TSC pool, NVRM_TEX_POOL_SLOTS x 32 B each, in VRAM and
    * CPU-written through BAR1 on every BAR size, as NVK's descriptor tables
    * (nvk_descriptor_table.c:73, NVKMD_MEM_LOCAL + NVKMD_MEM_MAP_WR). */
   for (unsigned r = 0; r < NVRM_BATCH_RING; r++) {
      if (!nvrm_bo_alloc(screen, 2u * NVRM_TEX_POOL_SLOTS * 32u, NVRM_DOMAIN_VRAM_MAPPED,
                         &ctx->ring[r].tex_pool))
         return false;
      memset(ctx->ring[r].tex_pool.map, 0, 2u * NVRM_TEX_POOL_SLOTS * 32u);
   }
   ctx->tex_pool = ctx->ring[ctx->ring_cur].tex_pool;
   /* Driver constant buffer (nvrm_priv.h NVRM_DRV_CB_*): GPU-written only. */
   if (!nvrm_bo_alloc(screen, (uint64_t)NVRM_GFX_STAGES * NVRM_DRV_CB_STAGE_B, NVRM_DOMAIN_VRAM,
                      &ctx->drv_cb))
      return false;
   for (unsigned s = 0; s < NVRM_GFX_STAGES; s++) {
      ctx->drv_cb_uniforms[s] = malloc(NVRM_DRV_CB_UNIFORM_B);
      if (!ctx->drv_cb_uniforms[s])
         return false;
   }
   ctx->tex_slots = calloc(NVRM_TEX_SLOT_HASH, sizeof(*ctx->tex_slots));
   if (!ctx->tex_slots)
      return false;
   ctx->tex_slot_gen = 1;
   struct pipe_sampler_state def = {
      .wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE, .wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
      .wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
      .min_img_filter = PIPE_TEX_FILTER_NEAREST, .mag_img_filter = PIPE_TEX_FILTER_NEAREST,
      .min_mip_filter = PIPE_TEX_MIPFILTER_NONE, .max_lod = 0,
   };
   nvrm_pack_tsc(screen, &def, ctx->default_tsc);
   struct nil_descriptor null = nil_null_descriptor(&screen->info, screen->zero_page.va);
   memcpy(ctx->null_tic, null.bits, sizeof(ctx->null_tic));
   for (unsigned s = 0; s < MESA_SHADER_STAGES; s++)
      for (unsigned i = 0; i < NVRM_MAX_IMAGES; i++)
         memcpy(ctx->image_tic[s][i], ctx->null_tic, sizeof(ctx->null_tic));
   return true;
}

void
nvrm_texture_context_destroy(struct nvrm_context *ctx)
{
   for (unsigned s = 0; s < MESA_SHADER_STAGES; s++)
      for (unsigned i = 0; i < NVRM_MAX_TEX_UNITS; i++)
         pipe_sampler_view_reference(&ctx->views[s][i], NULL);
   nvrm_image_context_destroy(ctx);
   for (unsigned r = 0; r < NVRM_BATCH_RING; r++)
      nvrm_bo_free(nvrm_screen(ctx->base.screen), &ctx->ring[r].tex_pool);
   memset(&ctx->tex_pool, 0, sizeof(ctx->tex_pool));
   nvrm_bo_free(nvrm_screen(ctx->base.screen), &ctx->drv_cb);
   for (unsigned s = 0; s < NVRM_GFX_STAGES; s++) {
      free(ctx->drv_cb_uniforms[s]);
      ctx->drv_cb_uniforms[s] = NULL;
   }
   free(ctx->tex_slots);
   ctx->tex_slots = NULL;
}

/* Pool binding, as NVK's nvk_queue_state_update for the 3D class. Emitted
 * once per submission batch (every flush marks all state dirty, because the
 * screen channel is shared by all contexts). */
void
nvrm_emit_tex_pools(struct nvrm_context *ctx, struct nv_push *p)
{
   const uint64_t tic = ctx->tex_pool.va;
   const uint64_t tsc = ctx->tex_pool.va + NVRM_TEX_POOL_SLOTS * 32u;
   P_MTHD(p, NV9097, SET_TEX_HEADER_POOL_A);
   P_NV9097_SET_TEX_HEADER_POOL_A(p, tic >> 32);
   P_NV9097_SET_TEX_HEADER_POOL_B(p, tic);
   P_NV9097_SET_TEX_HEADER_POOL_C(p, NVRM_TEX_POOL_SLOTS - 1);
   P_IMMD(p, NV9097, INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI, { .lines = LINES_ALL });
   P_MTHD(p, NV9097, SET_TEX_SAMPLER_POOL_A);
   P_NV9097_SET_TEX_SAMPLER_POOL_A(p, tsc >> 32);
   P_NV9097_SET_TEX_SAMPLER_POOL_B(p, tsc);
   P_NV9097_SET_TEX_SAMPLER_POOL_C(p, NVRM_TEX_POOL_SLOTS - 1);
   P_IMMD(p, NV9097, INVALIDATE_SAMPLER_CACHE_NO_WFI, { .lines = LINES_ALL });
}

/* Called before nvrm_push_begin of a draw: makes room in the pool (a full
 * pool means a flush, which moves to the next batch slot's empty pool). */
void
nvrm_texture_reserve(struct nvrm_context *ctx, unsigned slots)
{
   if (ctx->tex_pool_next + slots > NVRM_TEX_POOL_SLOTS)
      ctx->base.flush(&ctx->base, NULL, 0);
}

/* Header/sampler pool slot holding (tic, tsc) in the current batch's pool:
 * an entry already written for this batch is reused, otherwise the pair
 * goes into the next free slot and the header/sampler caches are
 * invalidated before the next draw (stock writes TIC/TSC only on first use
 * and invalidates once after the write;
 * NVK's descriptor tables are likewise written once per view/sampler).
 * The pool and this table live for one batch (ctx->ring slot), so an entry
 * is never rewritten while a submitted batch can read it. */
static uint32_t
nvrm_tex_slot(struct nvrm_context *ctx, const uint32_t tic[8], const uint32_t tsc[8])
{
   if (ctx->tex_pool_next == 0)
      ctx->tex_slot_gen++; /* the pool rotated (flush): forget its entries */
   uint32_t words[16];
   memcpy(words, tic, 32);
   memcpy(words + 8, tsc, 32);
   uint32_t h = 2166136261u;
   for (unsigned i = 0; i < 16; i++)
      h = (h ^ words[i]) * 16777619u;
   struct nvrm_tex_slot *free_entry = NULL;
   for (unsigned probe = 0; probe < 32; probe++) {
      struct nvrm_tex_slot *e = &ctx->tex_slots[(h + probe) & (NVRM_TEX_SLOT_HASH - 1)];
      if (e->gen != ctx->tex_slot_gen) {
         free_entry = e;
         break;
      }
      if (!memcmp(e->words, words, sizeof(words)))
         return e->slot;
   }
   const uint32_t slot = ctx->tex_pool_next++;
   assert(slot < NVRM_TEX_POOL_SLOTS);
   uint8_t *tic_pool = ctx->tex_pool.map;
   uint8_t *tsc_pool = tic_pool + NVRM_TEX_POOL_SLOTS * 32u;
   memcpy(tic_pool + slot * 32u, tic, 32);
   memcpy(tsc_pool + slot * 32u, tsc, 32);
   ctx->inval |= NVRM_INVAL_HEADER;
   if (free_entry) {
      memcpy(free_entry->words, words, sizeof(words));
      free_entry->slot = slot;
      free_entry->gen = ctx->tex_slot_gen;
   }
   return slot;
}

/* Handle table of `shader`'s texture units (header index | sampler index
 * << 20 at unit * 4) and image units (at (NVRM_MAX_TEX_UNITS + unit) * 4,
 * storage headers from nvrm_image.c with the default sampler, unused by
 * surface instructions). Returns its size in dwords (0 = none used). */
unsigned
nvrm_tex_handles(struct nvrm_context *ctx, mesa_shader_stage stage,
                 const struct nvrm_shader *shader, uint32_t *handles)
{
   const unsigned units = shader ? shader->tex_units : 0;
   const unsigned images = shader ? shader->image_units : 0;
   if (!units && !images)
      return 0;
   const unsigned count = images ? NVRM_MAX_TEX_UNITS + images : units;
   memset(handles, 0, count * 4);
   for (unsigned u = 0; u < units; u++) {
      struct nvrm_sampler_view *view = (struct nvrm_sampler_view *)ctx->views[stage][u];
      struct nvrm_sampler *samp = ctx->samplers[stage][u];
      uint32_t tsc[8];
      if (samp)
         nvrm_tsc_for_view(samp, view ? &view->base : NULL, tsc);
      else
         memcpy(tsc, ctx->default_tsc, 32);
      const uint32_t slot = nvrm_tex_slot(ctx, view ? view->tic : ctx->null_tic, tsc);
      if (view)
         nvrm_keep(ctx, view->base.texture);
      handles[u] = slot | (slot << 20);
   }
   for (unsigned i = 0; i < images; i++) {
      handles[NVRM_MAX_TEX_UNITS + i] = nvrm_tex_slot(ctx, ctx->image_tic[stage][i],
                                                      ctx->default_tsc);
      if (ctx->images[stage][i].resource)
         nvrm_keep(ctx, ctx->images[stage][i].resource);
   }
   return count;
}

/* Compute: the table in upload memory for the QMD's cbuf slot. Returns its
 * GPU address (0 = the shader uses none) and bound size in *size_B. */
uint64_t
nvrm_upload_tex_handles(struct nvrm_context *ctx, mesa_shader_stage stage,
                        const struct nvrm_shader *shader, uint32_t *size_B)
{
   uint32_t handles[NVRM_MAX_TEX_UNITS + NVRM_MAX_IMAGES];
   const unsigned count = nvrm_tex_handles(ctx, stage, shader, handles);
   *size_B = 0;
   if (!count)
      return 0;
   struct pipe_resource *buf = NULL;
   unsigned offset = 0;
   u_upload_data_ref(ctx->base.const_uploader, 0, count * 4,
                 ctx->base.screen->caps.constant_buffer_offset_alignment,
                 handles, &offset, &buf);
   nvrm_keep(ctx, buf);
   const uint64_t addr = nvrm_resource(buf)->bo.va + offset;
   pipe_resource_reference(&buf, NULL);
   *size_B = nvrm_cbuf_size(nvrm_screen(ctx->base.screen), count * 4);
   return addr;
}

/* 3D stages: the table lives in the stage's driver cbuf area, bound once
 * per batch as constant buffer NVRM_TEX_CBUF_SLOT; a bind change loads only
 * the handles that changed (stock: one 4-byte CB load per texture bind). */
void
nvrm_emit_textures(struct nvrm_context *ctx, struct nv_push *p,
                   mesa_shader_stage stage, uint32_t group,
                   const struct nvrm_shader *shader)
{
   const int gfx = nvrm_gfx_stage(stage);
   uint32_t handles[NVRM_MAX_TEX_UNITS + NVRM_MAX_IMAGES];
   const unsigned count = nvrm_tex_handles(ctx, stage, shader, handles);
   if (!count)
      return;
   STATIC_ASSERT(sizeof(handles) <= NVRM_DRV_CB_HANDLES_B);
   nvrm_drv_cb_load(ctx, p, gfx, NVRM_DRV_CB_UNIFORM_B, ctx->drv_cb_handles[gfx],
                    &ctx->drv_cb_handles_valid_B[gfx], handles, count * 4);
   if (ctx->cb_dirty[stage] & BITFIELD_BIT(NVRM_TEX_CBUF_SLOT)) {
      const uint64_t addr = ctx->drv_cb.va + (uint64_t)gfx * NVRM_DRV_CB_STAGE_B +
                            NVRM_DRV_CB_UNIFORM_B;
      P_MTHD(p, NV9097, SET_CONSTANT_BUFFER_SELECTOR_A);
      P_NV9097_SET_CONSTANT_BUFFER_SELECTOR_A(p, NVRM_DRV_CB_HANDLES_B);
      P_NV9097_SET_CONSTANT_BUFFER_SELECTOR_B(p, addr >> 32);
      P_NV9097_SET_CONSTANT_BUFFER_SELECTOR_C(p, addr);
      P_IMMD(p, NV9097, BIND_GROUP_CONSTANT_BUFFER(group), {
         .valid = VALID_TRUE, .shader_slot = NVRM_TEX_CBUF_SLOT,
      });
      ctx->cb_dirty[stage] &= ~BITFIELD_BIT(NVRM_TEX_CBUF_SLOT);
   }
}

/* ---------------------------------------------------------------- NIR */

/* GL texture unit n -> handle read from the driver constant buffer at
 * n * 4 (+ 4 * dynamic offset). Same unit for texture and sampler (gallium
 * binds sampler state and sampler view per unit). */
static bool
nvrm_lower_tex_instr(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   b->cursor = nir_before_instr(instr);

   /* samplerExternalOES (OES_EGL_image_external, always exposed by Mesa:
    * main/extensions.c) reaches the driver as GLSL_SAMPLER_DIM_EXTERNAL,
    * which NAK rejects (from_nir.rs parse_tex: "Unsupported texture
    * dimension: 6"; Vulkan never produces it). st/mesa binds the image as a
    * PIPE_TEXTURE_2D view, and YUV images are split into per-plane samples
    * by st/mesa's variant key before this point (st_program.c:1155-1170,
    * nir_lower_tex *_external), so it is a plain 2D sample, as
    * nv50_ir_from_nir.cpp (TEX_TARGET_2D) and zink (SpvDim2D) treat it. */
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_EXTERNAL)
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;

   if (nir_tex_instr_src_index(tex, nir_tex_src_texture_deref) >= 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref) >= 0) {
      nvrm_error("texture derefs reached the driver (st/mesa lowers samplers)\n");
      abort();
   }
   if (nir_tex_instr_need_sampler(tex) && tex->sampler_index != tex->texture_index) {
      nvrm_error("texture unit %u sampled with sampler unit %u: not implemented\n",
                 tex->texture_index, tex->sampler_index);
      abort();
   }
   nir_def *offset = nir_imm_int(b, tex->texture_index * 4);
   const int toff = nir_tex_instr_src_index(tex, nir_tex_src_texture_offset);
   if (toff >= 0)
      offset = nir_iadd(b, offset, nir_imul_imm(b, tex->src[toff].src.ssa, 4));
   nir_def *handle = nir_ldc_nv(b, 1, 32,
                                nir_imm_int(b, NVRM_TEX_CBUF_FOR(b->shader->info.stage)),
                                offset,
                                .align_mul = 4, .align_offset = 0);
   int soff = nir_tex_instr_src_index(tex, nir_tex_src_sampler_offset);
   if (soff >= 0)
      nir_tex_instr_remove_src(tex, soff);
   const int toff2 = nir_tex_instr_src_index(tex, nir_tex_src_texture_offset);
   if (toff2 >= 0)
      nir_tex_instr_remove_src(tex, toff2);
   nir_tex_instr_add_src(tex, nir_tex_src_texture_handle, handle);
   tex->texture_index = 0;
   tex->sampler_index = 0;
   return true;
}

void
nvrm_lower_textures(nir_shader *nir)
{
   /* The texture part of nak_preprocess_nir (nak_nir.c:339-348), which our
    * st/mesa path does not run. */
   const nir_lower_tex_options tex_options = {
      .lower_txd_3d = true,
      .lower_txd_cube_map = true,
      .lower_txd_clamp = true,
      .lower_txd_shadow = true,
      .lower_txp = ~0,
      .lower_invalid_implicit_lod = true,
   };
   NIR_PASS(_, nir, nir_lower_tex, &tex_options);
   NIR_PASS(_, nir, nir_normalize_cubemap_coords);
   NIR_PASS(_, nir, nir_shader_instructions_pass, nvrm_lower_tex_instr,
            nir_metadata_control_flow, NULL);
}

void
nvrm_init_texture_functions(struct pipe_context *p)
{
   p->create_sampler_state = nvrm_create_sampler_state;
   p->bind_sampler_states = nvrm_bind_sampler_states;
   p->delete_sampler_state = nvrm_delete_sampler_state;
   p->create_sampler_view = nvrm_create_sampler_view;
   p->sampler_view_destroy = nvrm_sampler_view_destroy;
   p->sampler_view_release = u_default_sampler_view_release;
   p->set_sampler_views = nvrm_set_sampler_views;
   /* Standard sample positions = what nvrm_emit_sample_layout programs
    * (vk_standard_sample_locations in 1/16 px; u_sample_positions.c has the
    * same table for 1/2/4/8 samples). */
   p->get_sample_position = u_default_get_sample_position;
   nvrm_init_image_functions(p);
}
