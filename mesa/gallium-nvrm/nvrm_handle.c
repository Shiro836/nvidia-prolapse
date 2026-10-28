/* SPDX-License-Identifier: MIT
 *
 * nvrm G2: dma-buf import/export and DRM format modifiers.
 *
 * Memory follows NVK on the same backend (nvkmd_rm.c): an imported dma-buf
 * is nv_rm_import'ed as kindless memory and the image VA carries the PTE
 * kind nil derives from the modifier (rm_mem_import + rm_va_bind); an
 * export is nv_rm_export of the resource's allocation (rm_mem_export).
 * Layouts are nil's DRM-modifier layouts (nil_image_init with .modifier,
 * as nvk_image_init for VK_EXT_image_drm_format_modifier), modifiers come
 * from nil_drm_format_mods_for_format (NVK's modifier list): uncompressed
 * block-linear 2D for the format's GOB type plus LINEAR. Exports are
 * uncompressed (nvrm never allocates compressible memory).
 *
 * YUV (NV12, P010): nvrm has no YUV texture format, so dri2 imports
 * each plane as its own resource of the plane's format (R8 + R8G8, R16 +
 * R16G16; dri2.c dri_create_image_from_winsys use_lowered) and st/mesa
 * samples the planes and converts (st_program.c lower_nv12 / lower_p010,
 * colour space and range from EGL_YUV_COLOR_SPACE_HINT_EXT /
 * EGL_SAMPLE_RANGE_HINT_EXT). A plane import follows NVK's rules for a
 * plane of a LINEAR DRM-modifier image (nvk_image.c explicit plane
 * offsets, nil Image::new_linear): LINEAR only (NVK exposes no
 * block-linear modifier for multi-planar formats), pitch a 32 B multiple
 * >= the plane's row, offset a multiple of the plane's nil alignment (32 B,
 * 256 B on Kepler), the plane inside the dma-buf. The plane is bound as the
 * whole dma-buf (kindless import, kind 0 VA) and addressed at VA + offset,
 * like an NVK image whose planes sit at their explicit offsets in one
 * memory object. Sampler-only: never a colour target, never exported.
 *
 * Refused loudly: other handle types (KMS/SHARED), buffers, YUV formats
 * other than NV12/P010, block-linear YUV, offset != 0 for non-YUV imports,
 * mipmapped/array/MSAA/depth exports, implicit (INVALID) modifiers on
 * import, block-linear strides that differ from nil's layout.
 */
#include <inttypes.h>
#include <unistd.h>

#include "nvrm_priv.h"

#include "frontend/winsys_handle.h"
#include "util/format/u_format.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "drm-uapi/drm_fourcc.h"

/* The plane formats dri2 imports for a YUV fourcc we accept (dri_helpers.c
 * mapping: NV12 = R8 + GR88, P010 = R16 + GR1616), or NONE. Only the two
 * decoder-output formats our gralloc allocates (minigbm 0004) are accepted;
 * any other YUV format is refused loudly at import. */
static enum pipe_format
nvrm_yuv_plane_format(enum pipe_format yuv, unsigned plane)
{
   switch (yuv) {
   case PIPE_FORMAT_NV12:
      return plane == 0 ? PIPE_FORMAT_R8_UNORM :
             plane == 1 ? PIPE_FORMAT_R8G8_UNORM : PIPE_FORMAT_NONE;
   case PIPE_FORMAT_P010:
      return plane == 0 ? PIPE_FORMAT_R16_UNORM :
             plane == 1 ? PIPE_FORMAT_R16G16_UNORM : PIPE_FORMAT_NONE;
   default:
      return PIPE_FORMAT_NONE;
   }
}

/* Sampling a YUV format = sampling each of its plane formats (st lowering). */
static bool
nvrm_yuv_supported(struct nvrm_screen *screen, enum pipe_format yuv)
{
   for (unsigned p = 0; p < util_format_get_num_planes(yuv); p++) {
      const enum pipe_format pf = nvrm_yuv_plane_format(yuv, p);
      if (pf == PIPE_FORMAT_NONE || !nil_format_supports_texturing(&screen->info, pf))
         return false;
   }
   return true;
}

static bool
nvrm_modifier_supported(struct nvrm_screen *screen, enum pipe_format format,
                        uint64_t modifier)
{
   if (modifier == DRM_FORMAT_MOD_INVALID)
      return false;
   /* YUV: LINEAR only (nil has no YUV format; NVK's multi-planar modifier
    * list is LINEAR only). Never hand a YUV format to nil_format(). */
   if (util_format_is_yuv(format))
      return modifier == DRM_FORMAT_MOD_LINEAR && nvrm_yuv_supported(screen, format);
   size_t count = NIL_MAX_DRM_FORMAT_MODS;
   uint64_t mods[NIL_MAX_DRM_FORMAT_MODS];
   nil_drm_format_mods_for_format(&screen->info, nil_format(format), &count, &mods);
   for (size_t i = 0; i < count; i++) {
      if (mods[i] == modifier)
         return true;
   }
   return false;
}

static void
nvrm_query_dmabuf_modifiers(struct pipe_screen *pscreen, enum pipe_format format,
                            int max, uint64_t *modifiers, unsigned int *external_only,
                            int *count)
{
   struct nvrm_screen *screen = nvrm_screen(pscreen);
   size_t n = NIL_MAX_DRM_FORMAT_MODS;
   uint64_t mods[NIL_MAX_DRM_FORMAT_MODS];
   if (util_format_is_yuv(format)) {
      /* Per-plane import + st colour conversion: LINEAR, external-only
       * (samplerExternalOES), dri2 marks it external-only too. */
      if (!nvrm_yuv_supported(screen, format)) {
         *count = 0;
         return;
      }
      if (max > 0) {
         modifiers[0] = DRM_FORMAT_MOD_LINEAR;
         if (external_only)
            external_only[0] = true;
      }
      *count = 1;
      return;
   }
   if (!nil_format_supports_color_targets(&screen->info, format) ||
       util_format_is_depth_or_stencil(format)) {
      *count = 0;
      return;
   }
   nil_drm_format_mods_for_format(&screen->info, nil_format(format), &n, &mods);
   if (max == 0) {
      *count = n;
      return;
   }
   int out = 0;
   for (size_t i = 0; i < n && out < max; i++, out++) {
      modifiers[out] = mods[i];
      /* LINEAR is sampled through a pitch TIC and rendered through the
       * LAYOUT_PITCH colour target (128 B pitch), so not external-only. */
      if (external_only)
         external_only[out] = false;
   }
   *count = out;
}

static bool
nvrm_is_dmabuf_modifier_supported(struct pipe_screen *pscreen, uint64_t modifier,
                                  enum pipe_format format, bool *external_only)
{
   if (external_only)
      *external_only = util_format_is_yuv(format);
   return nvrm_modifier_supported(nvrm_screen(pscreen), format, modifier);
}

static unsigned
nvrm_get_dmabuf_modifier_planes(struct pipe_screen *pscreen, uint64_t modifier,
                                enum pipe_format format)
{
   return util_format_get_num_planes(format);
}

static struct pipe_resource *
nvrm_resource_create_with_modifiers(struct pipe_screen *pscreen,
                                    const struct pipe_resource *templ,
                                    const uint64_t *modifiers, int count)
{
   struct nvrm_screen *screen = nvrm_screen(pscreen);
   const uint64_t modifier = nil_select_best_drm_format_mod(
      &screen->info, nil_format(templ->format), count, modifiers);
   if (modifier == DRM_FORMAT_MOD_INVALID) {
      nvrm_error("resource_create_with_modifiers: none of %d modifiers is supported "
                 "for %s\n", count, util_format_short_name(templ->format));
      return NULL;
   }
   return nvrm_resource_create_modifier(pscreen, templ, modifier);
}

/* The modifier naming this resource's layout, or INVALID (named error). A
 * resource created without modifiers is exportable when nil's layout for
 * one of the format's modifiers is identical to its own: same level-0
 * tiling, row pitch and PTE kind (nil picks the same tiling for the same
 * extent; checked here, never assumed). */
static uint64_t
nvrm_resource_modifier(struct nvrm_screen *screen, struct nvrm_resource *res)
{
   const struct pipe_resource *p = &res->base;
   if (res->has_modifier)
      return res->modifier;
   if (p->target == PIPE_BUFFER || (p->target != PIPE_TEXTURE_2D &&
                                    p->target != PIPE_TEXTURE_RECT) ||
       p->last_level != 0 || p->array_size != 1 || p->nr_samples > 1 ||
       util_format_is_depth_or_stencil(p->format)) {
      nvrm_error("export: only single-level 2D colour resources have a DRM modifier "
                 "(target %u, %u levels, %u layers, %u samples, %s)\n", p->target,
                 p->last_level + 1, p->array_size, p->nr_samples,
                 util_format_short_name(p->format));
      return DRM_FORMAT_MOD_INVALID;
   }
   if (!res->tiled)
      return DRM_FORMAT_MOD_LINEAR; /* 2D single-level PIPE_BIND_LINEAR */
   size_t n = NIL_MAX_DRM_FORMAT_MODS;
   uint64_t mods[NIL_MAX_DRM_FORMAT_MODS];
   nil_drm_format_mods_for_format(&screen->info, nil_format(p->format), &n, &mods);
   for (size_t i = 0; i < n; i++) {
      if (mods[i] == DRM_FORMAT_MOD_LINEAR)
         continue;
      struct nil_image_init_info info = {
         .dim = NIL_IMAGE_DIM_2D,
         .format = nil_format(p->format),
         .extent_px = { .width = p->width0, .height = p->height0, .depth = 1,
                        .array_len = 1 },
         .levels = 1,
         .samples = 1,
         .usage = NIL_IMAGE_USAGE_UNCOMPRESSED_BIT,
         .modifier = mods[i],
      };
      struct nil_image img;
      if (!nil_image_init(&screen->info, &img, &info))
         continue;
      if (img.pte_kind == res->image.pte_kind &&
          img.levels[0].row_stride_B == res->image.levels[0].row_stride_B &&
          !memcmp(&img.levels[0].tiling, &res->image.levels[0].tiling,
                  sizeof(img.levels[0].tiling)) &&
          img.size_B <= res->bo.size)
         return mods[i];
   }
   nvrm_error("export: %s %ux%u has no modifier with an identical layout (tiling y_log2 "
              "%u, pitch %u, kind 0x%x)\n", util_format_short_name(p->format), p->width0,
              p->height0, res->image.levels[0].tiling.y_log2,
              res->image.levels[0].row_stride_B, res->image.pte_kind);
   return DRM_FORMAT_MOD_INVALID;
}

static uint32_t
nvrm_resource_pitch(const struct nvrm_resource *res)
{
   return res->tiled ? res->image.levels[0].row_stride_B : res->stride;
}

static bool
nvrm_resource_get_handle(struct pipe_screen *pscreen, struct pipe_context *pctx,
                         struct pipe_resource *pres, struct winsys_handle *whandle,
                         unsigned usage)
{
   struct nvrm_screen *screen = nvrm_screen(pscreen);
   struct nvrm_resource *res = nvrm_resource(pres);
   if (whandle->type != WINSYS_HANDLE_TYPE_FD) {
      nvrm_error("resource_get_handle: handle type %u is not supported (dma-buf FD "
                 "only; nvidia-drm has no GEM names for RM memory)\n", whandle->type);
      return false;
   }
   if (res->plane_offset_B) {
      nvrm_error("resource_get_handle: a YUV plane import (offset %u B) is not "
                 "exported\n", res->plane_offset_B);
      return false;
   }
   if (whandle->plane != 0 || res->separate_zs) {
      nvrm_error("resource_get_handle: plane %u / separate Z/S export\n", whandle->plane);
      return false;
   }
   const uint64_t modifier = nvrm_resource_modifier(screen, res);
   if (modifier == DRM_FORMAT_MOD_INVALID)
      return false;
   int fd = -1;
   int32_t status = nv_rm_export(screen->rm, res->bo.mem, screen->fd, &fd);
   if (status || fd < 0) {
      nvrm_error("resource_get_handle: nv_rm_export failed: %d\n", status);
      return false;
   }
   /* Another process may hold the memory now: never back into the
    * buffer recycler (nvrm_resource.c). */
   res->recycle = false;
   whandle->handle = fd;
   whandle->stride = nvrm_resource_pitch(res);
   whandle->offset = 0;
   whandle->modifier = modifier;
   whandle->format = pres->format;
   whandle->size = res->bo.size;
   return true;
}

static bool
nvrm_resource_get_param(struct pipe_screen *pscreen, struct pipe_context *pctx,
                        struct pipe_resource *pres, unsigned plane, unsigned layer,
                        unsigned level, enum pipe_resource_param param,
                        unsigned handle_usage, uint64_t *value)
{
   struct nvrm_screen *screen = nvrm_screen(pscreen);
   struct nvrm_resource *res = nvrm_resource(pres);
   switch (param) {
   case PIPE_RESOURCE_PARAM_NPLANES:
      *value = 1;
      return true;
   case PIPE_RESOURCE_PARAM_STRIDE:
      *value = nvrm_resource_pitch(res);
      return true;
   case PIPE_RESOURCE_PARAM_OFFSET:
      *value = res->plane_offset_B;
      return true;
   case PIPE_RESOURCE_PARAM_LAYER_STRIDE:
      *value = res->tiled ? res->image.array_stride_B :
               (uint64_t)res->stride * pres->height0;
      return true;
   case PIPE_RESOURCE_PARAM_MODIFIER: {
      const uint64_t m = nvrm_resource_modifier(screen, res);
      if (m == DRM_FORMAT_MOD_INVALID)
         return false;
      *value = m;
      return true;
   }
   case PIPE_RESOURCE_PARAM_HANDLE_TYPE_FD: {
      struct winsys_handle wh = { .type = WINSYS_HANDLE_TYPE_FD, .plane = plane };
      if (!nvrm_resource_get_handle(pscreen, pctx, pres, &wh, handle_usage))
         return false;
      *value = wh.handle;
      return true;
   }
   default:
      /* KMS handles / flink names do not exist for RM memory. */
      return false;
   }
}

static struct pipe_resource *
nvrm_resource_from_handle(struct pipe_screen *pscreen, const struct pipe_resource *templ,
                          struct winsys_handle *whandle, unsigned usage)
{
   struct nvrm_screen *screen = nvrm_screen(pscreen);
   const uint64_t modifier = whandle->modifier;
   if (whandle->type != WINSYS_HANDLE_TYPE_FD) {
      nvrm_error("resource_from_handle: handle type %u is not supported (dma-buf FD "
                 "only)\n", whandle->type);
      return NULL;
   }
   if ((templ->target != PIPE_TEXTURE_2D && templ->target != PIPE_TEXTURE_RECT) ||
       templ->last_level != 0 || templ->array_size != 1 || templ->depth0 != 1 ||
       templ->nr_samples > 1 || util_format_is_depth_or_stencil(templ->format) ||
       util_format_get_num_planes(templ->format) != 1) {
      nvrm_error("resource_from_handle: only single-plane single-level 2D colour images "
                 "(target %u, %s, %u levels)\n", templ->target,
                 util_format_short_name(templ->format), templ->last_level + 1);
      return NULL;
   }
   if (modifier == DRM_FORMAT_MOD_INVALID) {
      nvrm_error("resource_from_handle: no DRM modifier given; NVIDIA memory has no "
                 "implicit layout (pass EGL_DMA_BUF_PLANE0_MODIFIER_*_EXT)\n");
      return NULL;
   }
   /* dri2 passes the dma-buf's own format in whandle->format; a YUV one
    * means templ is one plane of a per-plane (lowered) import. */
   const bool yuv_plane = util_format_is_yuv(whandle->format);
   if (yuv_plane) {
      const enum pipe_format pf = nvrm_yuv_plane_format(whandle->format, whandle->plane);
      if (pf == PIPE_FORMAT_NONE || !nvrm_yuv_supported(screen, whandle->format)) {
         nvrm_error("resource_from_handle: YUV format %s is not supported (NV12 and P010 "
                    "only, imported per plane)\n", util_format_short_name(whandle->format));
         return NULL;
      }
      if (templ->format != pf) {
         nvrm_error("resource_from_handle: %s plane %u imported as %s, expected %s\n",
                    util_format_short_name(whandle->format), whandle->plane,
                    util_format_short_name(templ->format), util_format_short_name(pf));
         return NULL;
      }
      if (modifier != DRM_FORMAT_MOD_LINEAR) {
         nvrm_error("resource_from_handle: %s modifier 0x%016" PRIx64 ": YUV dma-bufs are "
                    "imported LINEAR only (NVK's multi-planar modifier list)\n",
                    util_format_short_name(whandle->format), modifier);
         return NULL;
      }
      if (templ->bind & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_DISPLAY_TARGET |
                         PIPE_BIND_SCANOUT | PIPE_BIND_SHARED)) {
         nvrm_error("resource_from_handle: %s plane %u with bind 0x%x (YUV planes are "
                    "sampled only)\n", util_format_short_name(whandle->format),
                    whandle->plane, templ->bind);
         return NULL;
      }
   } else if (whandle->plane != 0 || whandle->offset != 0) {
      nvrm_error("resource_from_handle: %s plane %u offset %u (non-YUV dma-bufs: only "
                 "plane 0 at offset 0)\n", util_format_short_name(templ->format),
                 whandle->plane, whandle->offset);
      return NULL;
   }
   if (!nvrm_modifier_supported(screen, templ->format, modifier)) {
      nvrm_error("resource_from_handle: modifier 0x%016" PRIx64 " is not supported for "
                 "%s on this GPU (nil_drm_format_mods_for_format)\n", modifier,
                 util_format_short_name(templ->format));
      return NULL;
   }
   const bool linear = modifier == DRM_FORMAT_MOD_LINEAR;
   if (linear && (whandle->stride % 32 ||
                  whandle->stride < util_format_get_stride(templ->format, templ->width0))) {
      nvrm_error("resource_from_handle: LINEAR pitch %u B (needs a 32 B multiple >= %u)\n",
                 whandle->stride, util_format_get_stride(templ->format, templ->width0));
      return NULL;
   }

   struct nvrm_resource *res = CALLOC_STRUCT(nvrm_resource);
   if (!res)
      return NULL;
   res->base = *templ;
   res->base.screen = pscreen;
   /* templ->next: the planes after this one (dri2 chains a per-plane
    * import's resources through next and st samples them from there); the
    * chain's reference moves into this resource like in every gallium
    * driver's from_handle, pipe_resource_reference releases it with us. */
   pipe_reference_init(&res->base.reference, 1);

   struct nil_image_init_info info = {
      .dim = NIL_IMAGE_DIM_2D,
      .format = nil_format(templ->format),
      .extent_px = { .width = templ->width0, .height = templ->height0, .depth = 1,
                     .array_len = 1 },
      .levels = 1,
      .samples = 1,
      .usage = NIL_IMAGE_USAGE_UNCOMPRESSED_BIT,
      .modifier = modifier,
      .explicit_row_stride_B = linear ? whandle->stride : 0,
   };
   if (!nil_image_init(&screen->info, &res->image, &info)) {
      nvrm_error("resource_from_handle: nil_image_init refused %s %ux%u modifier 0x%"
                 PRIx64 "\n", util_format_short_name(templ->format), templ->width0,
                 templ->height0, modifier);
      FREE(res);
      return NULL;
   }
   if (!linear && whandle->stride != res->image.levels[0].row_stride_B) {
      nvrm_error("resource_from_handle: block-linear pitch %u B differs from the "
                 "modifier layout's %u B (%s %ux%u)\n", whandle->stride,
                 res->image.levels[0].row_stride_B, util_format_short_name(templ->format),
                 templ->width0, templ->height0);
      FREE(res);
      return NULL;
   }

   if (whandle->offset % res->image.align_B) {
      nvrm_error("resource_from_handle: %s plane %u offset %u B is not a multiple of the "
                 "plane's %u B alignment (nil linear image; NVK's plane-offset rule)\n",
                 util_format_short_name(templ->format), whandle->plane, whandle->offset,
                 res->image.align_B);
      FREE(res);
      return NULL;
   }

   struct NvRmAllocation alloc;
   int32_t status = nv_rm_import(screen->rm, screen->fd, whandle->handle, &alloc);
   if (status) {
      nvrm_error("resource_from_handle: nv_rm_import failed: %d\n", status);
      FREE(res);
      return NULL;
   }
   if (alloc.size < (uint64_t)whandle->offset + res->image.size_B) {
      nvrm_error("resource_from_handle: dma-buf has %" PRIu64 " B, %s plane %u at offset "
                 "%u needs %" PRIu64 " B\n", alloc.size, util_format_short_name(templ->format),
                 whandle->plane, whandle->offset,
                 (uint64_t)whandle->offset + res->image.size_B);
      nv_rm_free(screen->rm, alloc.id);
      FREE(res);
      return NULL;
   }
   const uint64_t align = MAX2(res->image.align_B, 4096);
   const uint64_t size = align64(alloc.size, 4096);
   simple_mtx_lock(&screen->va_mutex);
   const uint64_t va = util_vma_heap_alloc(&screen->heap, size, align);
   simple_mtx_unlock(&screen->va_mutex);
   if (!va) {
      nvrm_error("resource_from_handle: VA heap exhausted for %" PRIu64 " B\n", size);
      nv_rm_free(screen->rm, alloc.id);
      FREE(res);
      return NULL;
   }
   /* Kindless imported memory, VA with the modifier's PTE kind (LINEAR:
    * kind 0), as NVK binds an imported image (nvkmd_rm.c rm_va_bind). */
   uint64_t mapping;
   status = nv_rm_bind(screen->rm, alloc.id, 0, size, va, res->image.pte_kind, &mapping);
   if (status) {
      nvrm_error("resource_from_handle: nv_rm_bind(kind 0x%x) failed: %d\n",
                 res->image.pte_kind, status);
      simple_mtx_lock(&screen->va_mutex);
      util_vma_heap_free(&screen->heap, va, size);
      simple_mtx_unlock(&screen->va_mutex);
      nv_rm_free(screen->rm, alloc.id);
      FREE(res);
      return NULL;
   }
   res->bo = (struct nvrm_bo){ .mem = alloc.id, .mapping = mapping, .va = va,
                               .size = size };
   res->imported = true;
   res->has_modifier = true;
   res->modifier = modifier;
   res->plane_offset_B = whandle->offset;
   if (linear) {
      res->stride = whandle->stride;
      res->linear_image = true;
   } else {
      res->tiled = true;
   }
   return &res->base;
}

void
nvrm_init_handle_functions(struct pipe_screen *pscreen)
{
   pscreen->resource_from_handle = nvrm_resource_from_handle;
   pscreen->resource_get_handle = nvrm_resource_get_handle;
   pscreen->resource_get_param = nvrm_resource_get_param;
   pscreen->resource_create_with_modifiers = nvrm_resource_create_with_modifiers;
   pscreen->query_dmabuf_modifiers = nvrm_query_dmabuf_modifiers;
   pscreen->is_dmabuf_modifier_supported = nvrm_is_dmabuf_modifier_supported;
   pscreen->get_dmabuf_modifier_planes = nvrm_get_dmabuf_modifier_planes;
}
