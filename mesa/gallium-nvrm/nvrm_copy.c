/* SPDX-License-Identifier: MIT
 *
 * nvrm resource_copy_region on the copy engine (OES/EXT_copy_image,
 * glCopyBufferSubData, st/u_blitter copies):
 *  - buffers: nvk_CmdCopyBuffer2 (nvk_cmd_copy.c:404-445), pitch lines of at
 *    most 1 << 17 bytes;
 *  - textures: nvk_CmdCopyImage2 (nvk_cmd_copy.c:691-822) over
 *    nouveau_copy_rect (:186-370): element offsets/extents from nil (so
 *    compressed <-> plain copies of one block size and multisampled images
 *    work in elements), block-linear or pitch on either side, one launch per
 *    array layer or 3D slice, component remap as NVK picks it, Blackwell's
 *    separate depth/stencil planes copied plane by plane.
 * GL orders a copy after all earlier rendering, compute and copies and
 * before all later ones without application barriers, so each copy is
 * bracketed by NVK's barrier for the widest GL dependency:
 *  - before: nvk_cmd_flush_wait_dep (nvk_cmd_buffer.c:516-569) for source
 *    access SHADER_STORAGE_WRITE | COLOR/DEPTH_STENCIL_ATTACHMENT_WRITE |
 *    TRANSFORM_FEEDBACK_WRITE from graphics and compute stages =
 *    FLUSH_SHADER_DATA + RENDER_WFI + COMPUTE_WFI -> the waiting
 *    INVALIDATE_SHADER_CACHES (data + flush_data) on 3D and compute;
 *  - after: nvk_cmd_invalidate_deps (:571-640) for every destination read:
 *    texture data, shader global data + constants, and for buffers also the
 *    MME (indirect) and SKED (QMD) caches.
 * The copy itself relies on in-channel ordering of the 3D and copy
 * subchannels exactly as NVK's graphics queue does (copy class on subchannel
 * 4 of the same channel, LAUNCH_DMA non-pipelined with flush_enable).
 */
#include <inttypes.h>
#include <string.h>

#include "nvrm_priv.h"

#include "nil.h"
#include "nv_push.h"
#include "nv_push_cl9097.h"
#include "nv_push_cla097.h"
#include "nv_push_cla0c0.h"
#include "nv_push_clb1c0.h"
#include "nv_push_cl90b5.h"
#include "nv_push_clc1b5.h"
#include "nv_push_clcab5.h"
#include "nv_push_clc597.h"
#include "nv_push_clc86f.h"
#include "cl906f.h"
#include "clc1b5.h"
#include "clcab5.h"
#include "clb1c0.h"
#include "clc597.h"
#include "clcb97.h"

#include "util/format/u_format.h"
#include "util/u_inlines.h"
#include "util/u_math.h"

/* struct nouveau_copy_buffer (nvk_cmd_copy.c:31-40); is_3d stands for
 * image_type == VK_IMAGE_TYPE_3D. */
struct nvrm_copy_buffer {
   uint64_t base_addr;
   bool is_3d;
   struct nil_Offset4D_Elements offset_el;
   struct nil_Extent4D_Elements extent_el;
   uint32_t bpp;
   uint32_t row_stride;
   uint64_t array_stride;
   struct nil_tiling tiling;
};

/* struct nouveau_copy (nvk_cmd_copy.c:42-50). */
struct nvrm_copy {
   struct nvrm_copy_buffer src;
   struct nvrm_copy_buffer dst;
   struct {
      uint8_t comp_size;
      uint8_t dst[4];
   } remap;
   struct nil_Extent4D_Elements extent_el;
};

/* Gallium boxes carry the first array layer (or 3D slice) in z; NVK's
 * vk_image_sanitize_offset + vk_to_nil_offset split that into z (3D) or
 * a (layers). */
static struct nil_Offset4D_Pixels
nvrm_copy_offset_px(bool is_3d, unsigned x, unsigned y, unsigned z)
{
   return (struct nil_Offset4D_Pixels) {
      .x = x, .y = y, .z = is_3d ? z : 0, .a = is_3d ? 0 : z,
   };
}

/* nouveau_copy_rect_image (nvk_cmd_copy.c:88-120) for one nil plane. */
static struct nvrm_copy_buffer
nvrm_copy_rect_image(const struct nil_image *img, uint64_t va, unsigned level,
                     unsigned x, unsigned y, unsigned z)
{
   const bool is_3d = img->dim == NIL_IMAGE_DIM_3D;
   const struct nil_Extent4D_Pixels lvl_extent4d_px =
      nil_image_level_extent_px(img, level);
   return (struct nvrm_copy_buffer) {
      .base_addr = va + img->levels[level].offset_B,
      .is_3d = is_3d,
      .offset_el = nil_offset4d_px_to_el(nvrm_copy_offset_px(is_3d, x, y, z),
                                         img->format, img->sample_layout),
      .extent_el = nil_extent4d_px_to_el(lvl_extent4d_px, img->format,
                                         img->sample_layout),
      .bpp = util_format_get_blocksize(img->format.p_format),
      .row_stride = img->levels[level].row_stride_B,
      .array_stride = img->array_stride_B,
      .tiling = img->levels[level].tiling,
   };
}

/* A linear (PIPE_BIND_LINEAR / LINEAR-modifier / imported pitch) 2D texture:
 * single level, single layer, single sample (nvrm_resource_create_modifier,
 * nvrm_handle.c), the pitch side of nouveau_copy_rect. */
static struct nvrm_copy_buffer
nvrm_copy_rect_linear(const struct nvrm_resource *res, unsigned x, unsigned y)
{
   const enum pipe_format f = res->base.format;
   return (struct nvrm_copy_buffer) {
      .base_addr = res->bo.va + res->plane_offset_B,
      .offset_el = {
         .x = x / util_format_get_blockwidth(f),
         .y = y / util_format_get_blockheight(f),
      },
      .extent_el = {
         .width = util_format_get_nblocksx(f, res->base.width0),
         .height = util_format_get_nblocksy(f, res->base.height0),
         .depth = 1, .array_len = 1,
      },
      .bpp = util_format_get_blocksize(f),
      .row_stride = res->stride,
      .tiling = { .gob_type = NIL_GOB_TYPE_LINEAR },
   };
}

/* nouveau_copy_remap_format (nvk_cmd_copy.c:122-143). */
static void
nvrm_copy_remap_format(struct nvrm_copy *copy, enum pipe_format p_format)
{
   unsigned comp_size = util_format_get_blocksize(p_format);
   if (comp_size % 3 == 0) {
      comp_size /= 3;
      assert(util_is_power_of_two_nonzero(comp_size) && comp_size <= 4);
   } else {
      assert(util_is_power_of_two_nonzero(comp_size) && comp_size <= 16);
      comp_size = MIN2(comp_size, 4);
   }
   copy->remap.comp_size = comp_size;
   for (unsigned i = 0; i < 4; i++)
      copy->remap.dst[i] = i;
}

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
      nvrm_error("copy: GOB type %u has no Blackwell copy-engine kind\n", gob_type);
      abort();
   }
}

/* nouveau_copy_rect (nvk_cmd_copy.c:186-370), statement for statement. */
static void
nvrm_copy_rect(struct nvrm_context *ctx, const struct nvrm_copy *copy)
{
   const uint16_t cls_copy = nvrm_screen(ctx->base.screen)->info.cls_copy;
   uint32_t src_bw, dst_bw;
   if (copy->remap.comp_size > 0) {
      if (copy->src.bpp % copy->remap.comp_size || copy->dst.bpp % copy->remap.comp_size) {
         nvrm_error("copy: %u/%u B elements do not split into %u B components\n",
                    copy->src.bpp, copy->dst.bpp, copy->remap.comp_size);
         abort();
      }
      const uint32_t num_src_comps = copy->src.bpp / copy->remap.comp_size;
      const uint32_t num_dst_comps = copy->dst.bpp / copy->remap.comp_size;
      /* With component remapping most X/Y dimensions are in blocks. */
      src_bw = dst_bw = 1;
      struct nv_push push = nvrm_push_begin(ctx, 2);
      /* NVK passes to_90b5_remap_comp_size(num_*_comps) for the component
       * counts; both encodings are value - 1 (asserted equal there). */
      P_IMMD(&push, NV90B5, SET_REMAP_COMPONENTS, {
         .dst_x = copy->remap.dst[0],
         .dst_y = copy->remap.dst[1],
         .dst_z = copy->remap.dst[2],
         .dst_w = copy->remap.dst[3],
         .component_size = copy->remap.comp_size - 1,
         .num_src_components = num_src_comps - 1,
         .num_dst_components = num_dst_comps - 1,
      });
      nvrm_push_end(ctx, &push);
   } else {
      /* Without remapping dimensions are in bytes. */
      if (copy->src.bpp != copy->dst.bpp) {
         nvrm_error("copy: byte copy between %u B and %u B elements\n",
                    copy->src.bpp, copy->dst.bpp);
         abort();
      }
      src_bw = copy->src.bpp;
      dst_bw = copy->dst.bpp;
   }

   assert(copy->extent_el.depth == 1 || copy->extent_el.array_len == 1);
   const uint32_t layers = MAX2(copy->extent_el.depth, copy->extent_el.array_len);
   for (unsigned z = 0; z < layers; z++) {
      uint64_t src_addr = copy->src.base_addr;
      uint64_t dst_addr = copy->dst.base_addr;

      if (!copy->src.is_3d)
         src_addr += (z + copy->src.offset_el.a) * copy->src.array_stride;
      if (!copy->dst.is_3d)
         dst_addr += (z + copy->dst.offset_el.a) * copy->dst.array_stride;

      if (copy->src.tiling.gob_type == NIL_GOB_TYPE_LINEAR) {
         src_addr += copy->src.offset_el.x * copy->src.bpp +
                     copy->src.offset_el.y * copy->src.row_stride;
      }
      if (copy->dst.tiling.gob_type == NIL_GOB_TYPE_LINEAR) {
         dst_addr += copy->dst.offset_el.x * copy->dst.bpp +
                     copy->dst.offset_el.y * copy->dst.row_stride;
      }

      struct nv_push push = nvrm_push_begin(ctx, 31);
      struct nv_push *p = &push;

      P_MTHD(p, NV90B5, OFFSET_IN_UPPER);
      P_NV90B5_OFFSET_IN_UPPER(p, src_addr >> 32);
      P_NV90B5_OFFSET_IN_LOWER(p, src_addr & 0xffffffff);
      P_NV90B5_OFFSET_OUT_UPPER(p, dst_addr >> 32);
      P_NV90B5_OFFSET_OUT_LOWER(p, dst_addr & 0xffffffff);
      P_NV90B5_PITCH_IN(p, copy->src.row_stride);
      P_NV90B5_PITCH_OUT(p, copy->dst.row_stride);
      P_NV90B5_LINE_LENGTH_IN(p, copy->extent_el.width * src_bw);
      P_NV90B5_LINE_COUNT(p, copy->extent_el.height);

      uint32_t src_layout, dst_layout;
      if (copy->src.tiling.gob_type != NIL_GOB_TYPE_LINEAR) {
         assert(nil_gob_type_height(copy->src.tiling.gob_type) == 8);
         P_MTHD(p, NV90B5, SET_SRC_BLOCK_SIZE);
         if (cls_copy >= BLACKWELL_DMA_COPY_B) { /* nvk_cmd_copy.c:265 */
            P_NVCAB5_SET_SRC_BLOCK_SIZE(p, {
               .width = 0, /* tiles are always 1 GOB wide */
               .height = copy->src.tiling.y_log2,
               .depth = copy->src.tiling.z_log2,
               .gob_height = GOB_HEIGHT_GOB_HEIGHT_FERMI_8,
               .kind_bpp = nvrm_cab5_gob_kind(copy->src.tiling.gob_type),
            });
         } else {
            P_NV90B5_SET_SRC_BLOCK_SIZE(p, {
               .width = 0,
               .height = copy->src.tiling.y_log2,
               .depth = copy->src.tiling.z_log2,
               .gob_height = GOB_HEIGHT_GOB_HEIGHT_FERMI_8,
            });
         }
         /* The copy engine has no tile width: width = stride / bpp. */
         const uint32_t src_stride_el = copy->src.row_stride / copy->src.bpp;
         P_NV90B5_SET_SRC_WIDTH(p, src_stride_el * src_bw);
         P_NV90B5_SET_SRC_HEIGHT(p, copy->src.extent_el.height);
         P_NV90B5_SET_SRC_DEPTH(p, copy->src.extent_el.depth);
         P_NV90B5_SET_SRC_LAYER(p, copy->src.is_3d ? z + copy->src.offset_el.z : 0);
         if (cls_copy >= PASCAL_DMA_COPY_B) {
            P_MTHD(p, NVC1B5, SRC_ORIGIN_X);
            P_NVC1B5_SRC_ORIGIN_X(p, copy->src.offset_el.x * src_bw);
            P_NVC1B5_SRC_ORIGIN_Y(p, copy->src.offset_el.y);
         } else {
            P_MTHD(p, NV90B5, SET_SRC_ORIGIN);
            P_NV90B5_SET_SRC_ORIGIN(p, {
               .x = copy->src.offset_el.x * src_bw,
               .y = copy->src.offset_el.y,
            });
         }
         src_layout = NV90B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_BLOCKLINEAR;
      } else {
         src_layout = NV90B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH;
      }

      if (copy->dst.tiling.gob_type != NIL_GOB_TYPE_LINEAR) {
         assert(nil_gob_type_height(copy->dst.tiling.gob_type) == 8);
         P_MTHD(p, NV90B5, SET_DST_BLOCK_SIZE);
         if (cls_copy >= BLACKWELL_DMA_COPY_B) { /* nvk_cmd_copy.c:265 */
            P_NVCAB5_SET_DST_BLOCK_SIZE(p, {
               .width = 0,
               .height = copy->dst.tiling.y_log2,
               .depth = copy->dst.tiling.z_log2,
               .gob_height = GOB_HEIGHT_GOB_HEIGHT_FERMI_8,
               .kind_bpp = nvrm_cab5_gob_kind(copy->dst.tiling.gob_type),
            });
         } else {
            P_NV90B5_SET_DST_BLOCK_SIZE(p, {
               .width = 0,
               .height = copy->dst.tiling.y_log2,
               .depth = copy->dst.tiling.z_log2,
               .gob_height = GOB_HEIGHT_GOB_HEIGHT_FERMI_8,
            });
         }
         const uint32_t dst_stride_el = copy->dst.row_stride / copy->dst.bpp;
         P_NV90B5_SET_DST_WIDTH(p, dst_stride_el * dst_bw);
         P_NV90B5_SET_DST_HEIGHT(p, copy->dst.extent_el.height);
         P_NV90B5_SET_DST_DEPTH(p, copy->dst.extent_el.depth);
         P_NV90B5_SET_DST_LAYER(p, copy->dst.is_3d ? z + copy->dst.offset_el.z : 0);
         if (cls_copy >= PASCAL_DMA_COPY_B) {
            P_MTHD(p, NVC1B5, DST_ORIGIN_X);
            P_NVC1B5_DST_ORIGIN_X(p, copy->dst.offset_el.x * dst_bw);
            P_NVC1B5_DST_ORIGIN_Y(p, copy->dst.offset_el.y);
         } else {
            P_MTHD(p, NV90B5, SET_DST_ORIGIN);
            P_NV90B5_SET_DST_ORIGIN(p, {
               .x = copy->dst.offset_el.x * dst_bw,
               .y = copy->dst.offset_el.y,
            });
         }
         dst_layout = NV90B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_BLOCKLINEAR;
      } else {
         dst_layout = NV90B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH;
      }

      P_IMMD(p, NV90B5, LAUNCH_DMA, {
         .data_transfer_type = DATA_TRANSFER_TYPE_NON_PIPELINED,
         .multi_line_enable = MULTI_LINE_ENABLE_TRUE,
         .flush_enable = FLUSH_ENABLE_TRUE,
         .src_memory_layout = src_layout,
         .dst_memory_layout = dst_layout,
         .remap_enable = copy->remap.comp_size > 0,
      });
      nvrm_push_end(ctx, p);
   }
}

/* ---------------------------------------------------------------- barriers */

static void
nvrm_copy_barrier_before(struct nvrm_context *ctx)
{
   struct nv_push push = nvrm_push_begin(ctx, 4);
   P_IMMD(&push, NVA097, INVALIDATE_SHADER_CACHES, {
      .data = DATA_TRUE,
      .flush_data = FLUSH_DATA_TRUE,
   });
   P_IMMD(&push, NVA0C0, INVALIDATE_SHADER_CACHES, {
      .data = DATA_TRUE,
      .flush_data = FLUSH_DATA_TRUE,
   });
   nvrm_push_end(ctx, &push);
}

static void
nvrm_copy_barrier_after(struct nvrm_context *ctx, bool buffer)
{
   const struct nvrm_screen *screen = nvrm_screen(ctx->base.screen);
   struct nv_push push = nvrm_push_begin(ctx, 16);
   struct nv_push *p = &push;
   /* INVALIDATE_TEX_DATA: every nvrm class is MAXWELL_A+ (NO_WFI form). */
   P_IMMD(p, NVA097, INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI, { .lines = LINES_ALL });
   /* INVALIDATE_SHADER_DATA | INVALIDATE_CONSTANT */
   P_IMMD(p, NVA097, INVALIDATE_SHADER_CACHES_NO_WFI, {
      .global_data = true,
      .constant = true,
   });
   if (buffer) {
      /* INVALIDATE_MME_DATA (indirect draw/dispatch arguments). */
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
      /* INVALIDATE_QMD_DATA */
      if (screen->info.cls_compute >= MAXWELL_COMPUTE_B)
         P_IMMD(p, NVB1C0, INVALIDATE_SKED_CACHES, 0);
   }
   nvrm_push_end(ctx, p);
}

/* ---------------------------------------------------------------- entry */

static void
nvrm_copy_buffer(struct nvrm_context *ctx, struct nvrm_resource *dst, unsigned dstx,
                 struct nvrm_resource *src, const struct pipe_box *src_box)
{
   /* nvk_CmdCopyBuffer2 (nvk_cmd_copy.c:404-445) */
   uint64_t src_addr = src->bo.va + src_box->x;
   uint64_t dst_addr = dst->bo.va + dstx;
   uint64_t size = src_box->width;
   while (size) {
      struct nv_push push = nvrm_push_begin(ctx, 10);
      struct nv_push *p = &push;
      P_MTHD(p, NV90B5, OFFSET_IN_UPPER);
      P_NV90B5_OFFSET_IN_UPPER(p, src_addr >> 32);
      P_NV90B5_OFFSET_IN_LOWER(p, src_addr & 0xffffffff);
      P_NV90B5_OFFSET_OUT_UPPER(p, dst_addr >> 32);
      P_NV90B5_OFFSET_OUT_LOWER(p, dst_addr & 0xffffffff);
      const unsigned bytes = MIN2(size, 1 << 17);
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
      nvrm_push_end(ctx, p);
      src_addr += bytes;
      dst_addr += bytes;
      size -= bytes;
   }
}

/* One texture side of a copy: a tiled nil plane or the linear layout. */
static struct nvrm_copy_buffer
nvrm_copy_side(const struct nvrm_resource *res, bool stencil, unsigned level,
               unsigned x, unsigned y, unsigned z)
{
   if (!res->tiled)
      return nvrm_copy_rect_linear(res, x, y);
   if (stencil)
      return nvrm_copy_rect_image(&res->stencil_image, res->stencil_bo.va, level, x, y, z);
   return nvrm_copy_rect_image(&res->image, res->bo.va, level, x, y, z);
}

static void
nvrm_copy_texture(struct nvrm_context *ctx, struct nvrm_resource *dst, unsigned dst_level,
                  unsigned dstx, unsigned dsty, unsigned dstz,
                  struct nvrm_resource *src, unsigned src_level,
                  const struct pipe_box *box)
{
   const enum pipe_format src_pf = src->tiled ? src->image.format.p_format : src->base.format;
   const bool src_3d = src->tiled && src->image.dim == NIL_IMAGE_DIM_3D;
   /* "When copying between compressed and uncompressed formats the extent
    * members represent the texel dimensions of the source image" (NVK
    * nvk_CmdCopyImage2): extent in source pixels, converted with the source
    * format and sample layout. */
   const struct nil_Extent4D_Pixels extent4d_px = {
      .width = box->width, .height = box->height,
      .depth = src_3d ? box->depth : 1,
      .array_len = src_3d ? 1 : box->depth,
   };
   struct nil_Extent4D_Elements extent_el;
   if (src->tiled) {
      extent_el = nil_extent4d_px_to_el(extent4d_px, src->image.format,
                                        src->image.sample_layout);
   } else {
      extent_el = (struct nil_Extent4D_Elements) {
         .width = util_format_get_nblocksx(src_pf, box->width),
         .height = util_format_get_nblocksy(src_pf, box->height),
         .depth = 1, .array_len = box->depth,
      };
   }

   /* nvk_CmdCopyImage2's separate_zs branch: both planes, no remap. */
   if (src->separate_zs || dst->separate_zs) {
      if (!src->separate_zs || !dst->separate_zs || src->base.format != dst->base.format) {
         nvrm_error("copy: separate depth/stencil %s -> %s needs one combined format "
                    "on both sides\n", util_format_short_name(src->base.format),
                    util_format_short_name(dst->base.format));
         abort();
      }
      for (unsigned plane = 0; plane < 2; plane++) {
         const struct nil_image *pimg = plane ? &src->stencil_image : &src->image;
         const struct nvrm_copy copy = {
            .src = nvrm_copy_side(src, plane, src_level, box->x, box->y, box->z),
            .dst = nvrm_copy_side(dst, plane, dst_level, dstx, dsty, dstz),
            .extent_el = nil_extent4d_px_to_el(extent4d_px, pimg->format,
                                               pimg->sample_layout),
         };
         nvrm_copy_rect(ctx, &copy);
      }
      return;
   }

   struct nvrm_copy copy = {
      .src = nvrm_copy_side(src, false, src_level, box->x, box->y, box->z),
      .dst = nvrm_copy_side(dst, false, dst_level, dstx, dsty, dstz),
      .extent_el = extent_el,
   };
   /* nvk_CmdCopyImage2's format switch; GL copies every aspect of a
    * combined format, where NVK does a byte copy. */
   switch (src_pf) {
   case PIPE_FORMAT_Z24_UNORM_S8_UINT:
   case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT:
      break;
   case PIPE_FORMAT_Z16_UNORM_S8_UINT:
   case PIPE_FORMAT_S8_UINT_Z24_UNORM:
      nvrm_error("copy: packed depth/stencil %s is unsupported (as in NVK)\n",
                 util_format_short_name(src_pf));
      abort();
   default:
      nvrm_copy_remap_format(&copy, src_pf);
      break;
   }
   nvrm_copy_rect(ctx, &copy);
}

static void
nvrm_resource_copy_region(struct pipe_context *pctx,
                          struct pipe_resource *pdst, unsigned dst_level,
                          unsigned dstx, unsigned dsty, unsigned dstz,
                          struct pipe_resource *psrc, unsigned src_level,
                          const struct pipe_box *src_box)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   struct nvrm_resource *dst = nvrm_resource(pdst);
   struct nvrm_resource *src = nvrm_resource(psrc);
   const bool buffer = psrc->target == PIPE_BUFFER;

   if (buffer != (pdst->target == PIPE_BUFFER)) {
      nvrm_error("resource_copy_region between a buffer and a texture\n");
      abort();
   }
   if (!buffer && (psrc->nr_samples > 1) != (pdst->nr_samples > 1)) {
      nvrm_error("resource_copy_region between %u and %u samples\n",
                 psrc->nr_samples, pdst->nr_samples);
      abort();
   }
   /* A YUV plane import may be a source (its plane offset is applied),
    * never a destination: those imports are for sampling only
    * (nvrm_fb_linear_cbuf). */
   if (!buffer && !dst->tiled && dst->plane_offset_B) {
      nvrm_error("resource_copy_region into a YUV plane import\n");
      abort();
   }

   /* A linear colour buffer may live in its block-linear render shadow
    * during a batch (D1, nvrm_fb_shadow_begin/end): end that "render pass"
    * first so the linear memory is current (inline copy-out; the copy
    * engine's non-pipelined launches run in order), and make a later draw
    * reload the shadow from what this copy wrote. */
   if (!buffer && (!src->tiled || !dst->tiled) &&
       (ctx->shadow_written || ctx->shadow_loaded)) {
      nvrm_fb_shadow_end(ctx);
      ctx->shadow_loaded = false;
   }

   nvrm_keep(ctx, psrc);
   nvrm_keep(ctx, pdst);
   nvrm_copy_barrier_before(ctx);
   if (buffer)
      nvrm_copy_buffer(ctx, dst, dstx, src, src_box);
   else
      nvrm_copy_texture(ctx, dst, dst_level, dstx, dsty, dstz, src, src_level, src_box);
   nvrm_copy_barrier_after(ctx, buffer);
}

void
nvrm_init_copy_functions(struct pipe_context *pctx)
{
   pctx->resource_copy_region = nvrm_resource_copy_region;
}
