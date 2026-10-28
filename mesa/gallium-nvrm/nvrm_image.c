/* SPDX-License-Identifier: MIT
 *
 * nvrm GLES 3.1 shader images (part B):
 *  - set_shader_images builds one storage texture header per bound image
 *    as NVK's storage image view does (nvk_image_view.c:216-264): cube and
 *    cube-array views become 2D arrays, a layered 3D level is cut out
 *    with nil_image_for_level and a single slice (non-layered binding) is
 *    a 2D view of the level as a 2D array (nil_image_3d_level_as_2d_array);
 *    the header comes from nil_image_view_descriptor (Maxwell+ bindless
 *    path, cls_eng3d >= MAXWELL_A, the only classes nvrm accepts);
 *  - the headers are written per draw into fresh header-pool slots after the
 *    stage's texture units (nvrm_emit_textures) and their pool indices are
 *    the image handles in the driver constant buffer, at
 *    (NVRM_MAX_TEX_UNITS + unit) * 4: NVK's non-MSAA storage descriptor is
 *    exactly the header index (nvk_storage_image_descriptor.image_index,
 *    sw/sh_log2 = 0), which lower_image_intrin hands to NAK unchanged
 *    (nvk_nir_lower_descriptors.c:1104-1119);
 *  - NIR: st/mesa's image_* intrinsics on a unit index (gl_nir_lower_images,
 *    nir_images_as_deref = false) become bindless_image_* on that handle
 *    (nir_rewrite_image_intrinsic, as NVK), after NAK's own image
 *    preprocessing (nak_nir.c:350-353 lower_cube_size), which our st/mesa
 *    path does not run;
 *  - memory_barrier as NVK's barrier pair for shader storage writes read by
 *    later 3D work (nvk_cmd_buffer.c:540-560 flush, 606-630 invalidate).
 * Multisampled images are not exposed (max_image_samples 0; GLES 3.1 has no
 * image2DMS) and are refused loudly if they arrive.
 */
#include <string.h>

#include "nvrm_priv.h"

#include "nil.h"
#include "nv_push.h"
#include "nv_push_cl9097.h"
#include "nv_push_cla097.h"

#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"

static nil_view_type
nvrm_image_view_type(enum pipe_texture_target target, bool layered)
{
   switch (target) {
   case PIPE_TEXTURE_1D:         return NIL_VIEW_TYPE_1D;
   case PIPE_TEXTURE_1D_ARRAY:   return layered ? NIL_VIEW_TYPE_1D_ARRAY : NIL_VIEW_TYPE_1D;
   case PIPE_TEXTURE_2D:
   case PIPE_TEXTURE_RECT:       return NIL_VIEW_TYPE_2D;
   /* nvk_image_view.c:217-220: storage views have no cubes. */
   case PIPE_TEXTURE_2D_ARRAY:
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_CUBE_ARRAY: return layered ? NIL_VIEW_TYPE_2D_ARRAY : NIL_VIEW_TYPE_2D;
   case PIPE_TEXTURE_3D:         return NIL_VIEW_TYPE_3D;
   default:
      nvrm_error("shader image of texture target %u is not implemented\n", target);
      abort();
   }
}

/* Storage header for one pipe_image_view (resource non-NULL). */
static void
nvrm_image_tic(struct nvrm_screen *screen, const struct pipe_image_view *iv,
               uint32_t tic[8])
{
   struct nvrm_resource *res = nvrm_resource(iv->resource);
   if (iv->resource->target == PIPE_BUFFER) {
      nvrm_error("buffer images (imageBuffer, OES_texture_buffer) are not implemented\n");
      abort();
   }
   if (!res->tiled) {
      nvrm_error("shader image of a linear resource is not implemented\n");
      abort();
   }
   if (iv->resource->nr_samples > 1) {
      nvrm_error("multisampled shader images are not exposed (max_image_samples 0)\n");
      abort();
   }
   if (res->separate_zs || util_format_is_depth_or_stencil(iv->format)) {
      nvrm_error("depth/stencil shader image %s is not supported\n",
                 util_format_short_name(iv->format));
      abort();
   }
   if (!nil_format_supports_storage(&screen->info, iv->format)) {
      nvrm_error("shader image format %s has no storage support (nil)\n",
                 util_format_short_name(iv->format));
      abort();
   }

   const enum pipe_texture_target target = iv->resource->target;
   const unsigned first = iv->u.tex.first_layer, last = iv->u.tex.last_layer;
   /* st/mesa (st_convert_image) binds a layered image as layers
    * 0..max_layer and a non-layered one as first == last == the layer. A
    * one-layer array resource is ambiguous; it keeps its array view. */
   const bool layered = last > first ||
                        (target != PIPE_TEXTURE_3D && iv->resource->array_size == 1);
   struct nil_image image = res->image;
   uint64_t base_addr = res->bo.va;
   struct nil_view nv = {
      .view_type = nvrm_image_view_type(target, layered),
      .format = nil_format(iv->format),
      .base_level = iv->u.tex.level,
      .num_levels = 1,
      .base_array_layer = target == PIPE_TEXTURE_3D ? 0 : first,
      .array_len = target == PIPE_TEXTURE_3D ? 1 : last - first + 1,
      .swizzle = { PIPE_SWIZZLE_X, PIPE_SWIZZLE_Y, PIPE_SWIZZLE_Z, PIPE_SWIZZLE_W },
      .min_lod_clamp = 0.0f,
   };
   if (target == PIPE_TEXTURE_3D && iv->u.tex.is_2d_view_of_3d) {
      /* Non-layered binding of a 3D level: the shader declares an image2D
       * of slice `first` (st_atom_image.c sets is_2d_view_of_3d). A 3D or
       * 3D_SLICED header does not match the 2D coordinates NAK emits (the
       * 26 image_load_store.3d.*_single_layer failures). As NVK's 2D view
       * of a 3D image (VK_EXT_image_2d_view_of_3d, which Zink uses for
       * this binding, zink_context.c:1931-1940): image_3d_view_as_2d_array
       * (nvk_image_view.c:75-87, 183-185) with the slice as the array
       * layer. Needs the resource's NIL_IMAGE_USAGE_2D_VIEW_BIT
       * (nvrm_resource.c). */
      assert(first == last);
      uint64_t offset_B;
      image = nil_image_3d_level_as_2d_array(&image, nv.base_level, &offset_B);
      base_addr += offset_B;
      nv.view_type = NIL_VIEW_TYPE_2D;
      nv.base_level = 0;
      nv.base_array_layer = first;
      nv.array_len = 1;
   } else if (target == PIPE_TEXTURE_3D) {
      /* Layered: the whole level as 3D (nvk_image_view.c:222-238, one
       * level via image_single_level_view). */
      uint64_t offset_B;
      image = nil_image_for_level(&image, nv.base_level, &offset_B);
      base_addr += offset_B;
      nv.base_level = 0;
      nv.base_array_layer = 0;
      nv.array_len = 1;
   }
   struct nil_descriptor d = nil_image_view_descriptor(&screen->info, &image, &nv, base_addr);
   memcpy(tic, d.bits, 8 * sizeof(uint32_t));
}

static void
nvrm_set_shader_images(struct pipe_context *pctx, mesa_shader_stage shader,
                       unsigned start, unsigned count, unsigned unbind_trailing,
                       const struct pipe_image_view *images)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   struct nvrm_screen *screen = nvrm_screen(pctx->screen);
   if (start + count + unbind_trailing > NVRM_MAX_IMAGES) {
      nvrm_error("image units %u..%u exceed %u\n", start, start + count + unbind_trailing,
                 NVRM_MAX_IMAGES);
      abort();
   }
   for (unsigned i = 0; i < count + unbind_trailing; i++) {
      struct pipe_image_view *dst = &ctx->images[shader][start + i];
      const struct pipe_image_view *src = (images && i < count) ? &images[i] : NULL;
      if (src && src->resource) {
         pipe_resource_reference(&dst->resource, src->resource);
         dst->format = src->format;
         dst->access = src->access;
         dst->shader_access = src->shader_access;
         dst->u = src->u;
         nvrm_image_tic(screen, src, ctx->image_tic[shader][start + i]);
      } else {
         pipe_resource_reference(&dst->resource, NULL);
         memcpy(ctx->image_tic[shader][start + i], ctx->null_tic, sizeof(ctx->null_tic));
      }
   }
   ctx->dirty |= NVRM_DIRTY_TEXTURES;
}

void
nvrm_image_context_destroy(struct nvrm_context *ctx)
{
   for (unsigned s = 0; s < MESA_SHADER_STAGES; s++)
      for (unsigned i = 0; i < NVRM_MAX_IMAGES; i++)
         pipe_resource_reference(&ctx->images[s][i].resource, NULL);
}

/* glMemoryBarrier: shader stores of earlier 3D work must be visible to
 * every later read (image/texture/constant/vertex/index fetch or CE copy).
 * Flush side = NVK FLUSH_SHADER_DATA with RENDER_WFI: INVALIDATE_SHADER_CACHES
 * (the WFI form) with data + flush_data; invalidate side = texture data
 * cache + global/constant shader caches (nvk_cmd_invalidate_deps). The
 * compute-side flush (NVA0C0 INVALIDATE_SHADER_CACHES) belongs with compute
 * dispatch. CPU maps flush and wait for the resource's submissions. */
/* glMemoryBarrier is nvrm_compute.c's nvrm_memory_barrier: the same NVK
 * flush/invalidate pair on the 3D and compute classes, plus SKED and MME
 * data invalidation. */

/* glTextureBarrier / framebuffer fetch-style feedback: earlier render
 * target writes must be visible to texture reads (NVK RENDER_WFI +
 * INVALIDATE_TEX_DATA). */
static void
nvrm_texture_barrier(struct pipe_context *pctx, unsigned flags)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   struct nv_push push = nvrm_push_begin(ctx, 8);
   struct nv_push *p = &push;
   P_IMMD(p, NVA097, WAIT_FOR_IDLE, 0);
   P_IMMD(p, NVA097, INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI, { .lines = LINES_ALL });
   nvrm_push_end(ctx, p);
}

/* ---------------------------------------------------------------- NIR */

static bool
nvrm_lower_image_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_image_load:
   case nir_intrinsic_image_sparse_load:
   case nir_intrinsic_image_store:
   case nir_intrinsic_image_atomic:
   case nir_intrinsic_image_atomic_swap:
   case nir_intrinsic_image_size:
   case nir_intrinsic_image_samples:
      break;
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
      nvrm_error("image derefs reached the driver (nir_images_as_deref is off)\n");
      abort();
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_bindless_image_store:
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_bindless_image_atomic_swap:
   case nir_intrinsic_bindless_image_size:
      nvrm_error("bindless images (ARB_bindless_texture) are not implemented\n");
      abort();
   default:
      return false;
   }
   if (nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_MS ||
       nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_SUBPASS_MS) {
      nvrm_error("multisampled shader images are not exposed\n");
      abort();
   }
   if (nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_BUF) {
      nvrm_error("buffer images (imageBuffer) are not implemented\n");
      abort();
   }
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *unit = intr->src[0].ssa;
   if (nir_intrinsic_has_range_base(intr))
      unit = nir_iadd_imm(b, unit, nir_intrinsic_range_base(intr));
   nir_def *offset = nir_imul_imm(b, nir_iadd_imm(b, unit, NVRM_MAX_TEX_UNITS), 4);
   nir_def *handle = nir_ldc_nv(b, 1, 32, nir_imm_int(b, NVRM_TEX_CBUF_FOR(b->shader->info.stage)),
                                offset,
                                .align_mul = 4, .align_offset = 0);
   nir_rewrite_image_intrinsic(intr, handle, nir_image_intrinsic_type_bindless);
   return true;
}

void
nvrm_lower_images(nir_shader *nir)
{
   const nir_lower_image_options image_options = {
      .lower_cube_size = true,
   };
   NIR_PASS(_, nir, nir_lower_image, &image_options);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, nvrm_lower_image_instr,
            nir_metadata_control_flow, NULL);
}

void
nvrm_init_image_functions(struct pipe_context *pctx)
{
   pctx->set_shader_images = nvrm_set_shader_images;
   pctx->texture_barrier = nvrm_texture_barrier;
}
