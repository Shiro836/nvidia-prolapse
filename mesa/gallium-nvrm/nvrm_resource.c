/* SPDX-License-Identifier: MIT
 *
 * nvrm resources: buffers and PIPE_BIND_LINEAR 2D textures are linear,
 * GART-backed and persistently mapped (G0a); every other texture is a tiled
 * (block-linear) VRAM image laid out by nil (G1), mapped through a linear
 * staging buffer and the copy engine. MSAA fails loudly.
 */
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "nvrm_priv.h"

#include "clcd97.h"

#include "util/format/u_format.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/u_transfer.h"
#include "util/u_upload_mgr.h"
#include "drm-uapi/drm_fourcc.h"

/* Recycled buffer memory (nvrm_bo_cache_get): smallest transient size
 * rounded to a power of two, and the total kept. */
#define NVRM_BO_CACHE_MIN_B (64u * 1024u)
#define NVRM_BO_CACHE_MAX_B (128ull * 1024u * 1024u)

static nil_image_dim
nvrm_image_dim(enum pipe_texture_target target)
{
   switch (target) {
   case PIPE_TEXTURE_1D:
   case PIPE_TEXTURE_1D_ARRAY:   return NIL_IMAGE_DIM_1D;
   case PIPE_TEXTURE_3D:         return NIL_IMAGE_DIM_3D;
   default:                      return NIL_IMAGE_DIM_2D;
   }
}

/* Buffer placement. Full BAR: NVK's DEVICE_LOCAL | HOST_VISIBLE type
 * (BAR1-mapped VRAM) for every GPU-read buffer, the type ANGLE-on-NVK
 * buffers get on a resizable BAR (nvk_physical_device.c:1472-1478); system
 * memory only for CPU readback staging, nvc0's PIPE_USAGE_STAGING. Small
 * BAR: nvc0's domain policy (nouveau_buffer.c:686-716): persistent/coherent
 * maps and STREAM/STAGING usage in GART, DEFAULT/IMMUTABLE/DYNAMIC in VRAM
 * written through GART staging + copy engine, so BAR1 never holds buffers.
 * Shared (exportable) buffers stay in GART (dma-buf export contract
 * unchanged). */
static enum nvrm_domain
nvrm_buffer_domain(const struct nvrm_screen *screen, const struct pipe_resource *templ)
{
   if (templ->usage == PIPE_USAGE_STAGING || (templ->bind & PIPE_BIND_SHARED))
      return NVRM_DOMAIN_GART;
   if (screen->bar_full)
      return NVRM_DOMAIN_VRAM_MAPPED;
   if (templ->flags & (PIPE_RESOURCE_FLAG_MAP_PERSISTENT | PIPE_RESOURCE_FLAG_MAP_COHERENT))
      return NVRM_DOMAIN_GART;
   return templ->usage == PIPE_USAGE_STREAM ? NVRM_DOMAIN_GART : NVRM_DOMAIN_VRAM;
}

/* modifier: DRM_FORMAT_MOD_INVALID = the driver picks the layout (nil);
 * LINEAR = the PIPE_BIND_LINEAR path; a block-linear modifier = nil's
 * modifier layout (as nvk_image_init with an explicit DRM modifier), so the
 * layout is exactly what an importer of that modifier computes. Callers
 * validated the modifier (nvrm_handle.c). */
struct pipe_resource *
nvrm_resource_create_modifier(struct pipe_screen *pscreen,
                              const struct pipe_resource *templ, uint64_t modifier)
{
   struct nvrm_screen *screen = nvrm_screen(pscreen);

   struct nvrm_resource *res = CALLOC_STRUCT(nvrm_resource);
   if (!res)
      return NULL;
   res->base = *templ;
   res->base.screen = pscreen;
   pipe_reference_init(&res->base.reference, 1);

   if (templ->nr_samples > 1 &&
       (templ->target == PIPE_BUFFER || (templ->bind & PIPE_BIND_LINEAR) ||
        templ->last_level != 0)) {
      nvrm_error("resource_create: %u samples on a buffer/linear/mipmapped resource\n",
                 templ->nr_samples);
      FREE(res);
      return NULL;
   }

   if (templ->target == PIPE_BUFFER) {
      res->domain = nvrm_buffer_domain(screen, templ);
      uint64_t size = templ->width0;
      /* Buffer memory is recycled (nvrm_bo_cache_get): u_upload's
       * stream/const buffers, transfer staging, and the new storage
       * st/mesa creates for every glBufferData (orphaning). Transient sizes
       * of NVRM_BO_CACHE_MIN_B and more are rounded to a power of two so
       * released memory fits the next request; other sizes match exactly
       * (4 KiB pages, nvrm_bo_alloc). Exported buffers are never reused. */
      res->recycle = !(templ->bind & PIPE_BIND_SHARED);
      if (res->recycle) {
         if ((templ->usage == PIPE_USAGE_STREAM || templ->usage == PIPE_USAGE_STAGING) &&
             size >= NVRM_BO_CACHE_MIN_B)
            size = util_next_power_of_two64(size);
         size = align64(MAX2(size, 1), 4096);
         if (nvrm_bo_cache_get(screen, size, res->domain, &res->bo))
            return &res->base;
      }
      if (!nvrm_bo_alloc(screen, size, res->domain, &res->bo))
         goto fail;
      return &res->base;
   }

   if ((templ->bind & PIPE_BIND_LINEAR) || modifier == DRM_FORMAT_MOD_LINEAR) {
      if (templ->target != PIPE_TEXTURE_2D && templ->target != PIPE_TEXTURE_RECT) {
         nvrm_error("resource_create: linear %u-target texture not implemented\n",
                    templ->target);
         goto fail;
      }
      if (templ->last_level != 0 || templ->array_size != 1 ||
          util_format_is_depth_or_stencil(templ->format)) {
         nvrm_error("resource_create: linear textures are single-level 2D colour\n");
         goto fail;
      }
      /* NVK's linear render-target rule: 128 B aligned pitch and base
       * (nvk_cmd_draw.c LAYOUT_PITCH branch). */
      res->stride = align(util_format_get_stride(templ->format, templ->width0), 128);
      const uint64_t size = (uint64_t)res->stride *
                            util_format_get_nblocksy(templ->format, templ->height0);
      if (!nvrm_bo_alloc(screen, size, NVRM_DOMAIN_GART, &res->bo))
         goto fail;
      if (modifier == DRM_FORMAT_MOD_LINEAR) {
         res->has_modifier = true;
         res->modifier = DRM_FORMAT_MOD_LINEAR;
      }
      return &res->base;
   }

   /* Tiled image laid out by nil, as nvk_image_init for an optimal-tiling
    * single-plane image. */
   if (!nil_format_supports_texturing(&screen->info, templ->format) &&
       !nil_format_supports_color_targets(&screen->info, templ->format) &&
       !nil_format_supports_depth_stencil(&screen->info, templ->format)) {
      nvrm_error("resource_create: format %s unknown to nil\n",
                 util_format_short_name(templ->format));
      goto fail;
   }
   /* Blackwell keeps combined Z/S in two planes (nvk_use_separate_zs):
    * depth-only format + S8_UINT, as nvk_image_init. */
   res->separate_zs = util_format_is_depth_and_stencil(templ->format) &&
                      screen->info.cls_eng3d >= BLACKWELL_A;
   const enum pipe_format plane0_format = res->separate_zs ?
      util_format_get_depth_only(templ->format) : templ->format;
   const bool cube = templ->target == PIPE_TEXTURE_CUBE ||
                     templ->target == PIPE_TEXTURE_CUBE_ARRAY;
   struct nil_image_init_info info = {
      .dim = nvrm_image_dim(templ->target),
      .format = nil_format(plane0_format),
      .extent_px = {
         .width = templ->width0,
         .height = templ->height0,
         .depth = templ->depth0,
         .array_len = templ->array_size,
      },
      .levels = templ->last_level + 1,
      /* Multisampled: nil picks the sample layout (nvk_image_init). */
      .samples = MAX2(templ->nr_samples, 1),
      /* 3D: one Z-slice per GOB so a single slice can be viewed as a 2D
       * image (GLES 3.1 non-layered glBindImageTexture of a 3D level,
       * st_atom_image.c is_2d_view_of_3d). As GL on NVK does: Zink creates
       * every non-sparse 3D image 2D_VIEW_COMPATIBLE (zink_resource.c:
       * 771-779), which NVK turns into NIL_IMAGE_USAGE_2D_VIEW_BIT
       * (nvk_image.c:810-813); nil_image_3d_level_as_2d_array requires
       * z_log2 == 0 (nil/image.rs:735). */
      /* UNCOMPRESSED: nvrm images never use a compressible kind (NVK passes
       * it whenever an image cannot be compressed, nvk_image.c
       * nvk_image_init). */
      .usage = NIL_IMAGE_USAGE_UNCOMPRESSED_BIT |
               (templ->target == PIPE_TEXTURE_3D ? NIL_IMAGE_USAGE_2D_VIEW_BIT : 0),
      .modifier = modifier,
   };
   if (modifier != DRM_FORMAT_MOD_INVALID &&
       (info.dim != NIL_IMAGE_DIM_2D || info.levels != 1 || info.samples != 1 ||
        info.extent_px.array_len != 1 || util_format_is_depth_or_stencil(templ->format))) {
      nvrm_error("resource_create: modifier 0x%" PRIx64 " needs a single-level 2D colour "
                 "image (got %s, %u levels, %u samples, %u layers)\n", modifier,
                 util_format_short_name(templ->format), info.levels, info.samples,
                 info.extent_px.array_len);
      goto fail;
   }
   (void)cube; /* cube faces are array layers (array_size is 6 * cubes) */
   if (!nil_image_init(&screen->info, &res->image, &info)) {
      nvrm_error("resource_create: nil_image_init refused %s %ux%ux%u x%u levels %u\n",
                 util_format_short_name(templ->format), templ->width0, templ->height0,
                 templ->depth0, templ->array_size, templ->last_level + 1);
      goto fail;
   }
   if (!nvrm_bo_alloc_image(screen, &res->image, &res->bo))
      goto fail;
   if (res->separate_zs) {
      info.format = nil_format(PIPE_FORMAT_S8_UINT);
      if (!nil_image_init(&screen->info, &res->stencil_image, &info) ||
          !nvrm_bo_alloc_image(screen, &res->stencil_image, &res->stencil_bo)) {
         nvrm_error("resource_create: stencil plane of %s failed\n",
                    util_format_short_name(templ->format));
         nvrm_bo_free(screen, &res->bo);
         goto fail;
      }
   }
   res->tiled = true;
   if (modifier != DRM_FORMAT_MOD_INVALID) {
      res->has_modifier = true;
      res->modifier = modifier;
   }
   return &res->base;

fail:
   FREE(res);
   return NULL;
}

/* ---------------------------------------------------------------- recycling */

/* Released transient buffer memory, reused once its last submission
 * completed: steady frames then allocate, bind and CPU-map nothing (NVK
 * recycles upload memory the same way, nvk_cmd_pool.c free_mem lists;
 * stock streams through persistent sysmem/VRAM rings). Bounded: past NVRM_BO_CACHE_MAX_B the oldest
 * entries are freed (the backend defers a free past submitted work). */

bool
nvrm_bo_cache_get(struct nvrm_screen *screen, uint64_t size, enum nvrm_domain domain,
                  struct nvrm_bo *bo)
{
   bool found = false;
   simple_mtx_lock(&screen->bo_cache_mutex);
   struct nvrm_bo_cache_entry *e = screen->bo_cache.data;
   const unsigned n = util_dynarray_num_elements(&screen->bo_cache, struct nvrm_bo_cache_entry);
   for (unsigned i = 0; i < n; i++) {
      if (e[i].bo.size != size || e[i].domain != domain || !nvrm_token_done(screen, e[i].token))
         continue;
      *bo = e[i].bo;
      screen->bo_cache_bytes -= size;
      /* Keep age order (oldest first) for the eviction below. */
      memmove(&e[i], &e[i + 1], (n - i - 1) * sizeof(*e));
      screen->bo_cache.size -= sizeof(*e);
      found = true;
      break;
   }
   simple_mtx_unlock(&screen->bo_cache_mutex);
   return found;
}

static void
nvrm_bo_cache_put(struct nvrm_screen *screen, struct nvrm_resource *res)
{
   const struct nvrm_bo_cache_entry entry = {
      .bo = res->bo, .domain = res->domain, .token = p_atomic_read(&res->last_token),
   };
   simple_mtx_lock(&screen->bo_cache_mutex);
   util_dynarray_append_typed(&screen->bo_cache, struct nvrm_bo_cache_entry, entry);
   screen->bo_cache_bytes += entry.bo.size;
   while (screen->bo_cache_bytes > NVRM_BO_CACHE_MAX_B) {
      struct nvrm_bo_cache_entry *e = screen->bo_cache.data;
      screen->bo_cache_bytes -= e->bo.size;
      nvrm_bo_free(screen, &e->bo);
      memmove(e, e + 1, screen->bo_cache.size - sizeof(*e));
      screen->bo_cache.size -= sizeof(*e);
   }
   simple_mtx_unlock(&screen->bo_cache_mutex);
}

void
nvrm_bo_cache_fini(struct nvrm_screen *screen)
{
   util_dynarray_foreach(&screen->bo_cache, struct nvrm_bo_cache_entry, e)
      nvrm_bo_free(screen, &e->bo);
   util_dynarray_fini(&screen->bo_cache);
   screen->bo_cache_bytes = 0;
}

static void
nvrm_resource_destroy(struct pipe_screen *pscreen, struct pipe_resource *pres)
{
   struct nvrm_resource *res = nvrm_resource(pres);
   pipe_resource_reference(&res->render_shadow, NULL);
   if (res->recycle && res->bo.mem) {
      nvrm_bo_cache_put(nvrm_screen(pscreen), res);
      FREE(res);
      return;
   }
   nvrm_bo_free(nvrm_screen(pscreen), &res->bo);
   nvrm_bo_free(nvrm_screen(pscreen), &res->stencil_bo);
   FREE(res);
}

static struct pipe_resource *
nvrm_resource_create(struct pipe_screen *pscreen, const struct pipe_resource *templ)
{
   return nvrm_resource_create_modifier(pscreen, templ, DRM_FORMAT_MOD_INVALID);
}

void
nvrm_init_resource_functions(struct pipe_screen *pscreen)
{
   pscreen->resource_create = nvrm_resource_create;
   pscreen->resource_destroy = nvrm_resource_destroy;
}

/* ---------------------------------------------------------------- transfers */

struct nvrm_transfer {
   struct pipe_transfer base;
   struct pipe_resource *staging; /* tiled images: linear GART copy of box;
                                   * VRAM buffers (small BAR): GART copy of
                                   * [box.x, box.x + box.width) at
                                   * staging_offset */
   unsigned staging_offset;
   /* Blackwell separate Z/S: per-plane staging, CPU (de)interleaved into
    * the packed gallium format (as u_transfer_helper's interleaved Z/S
    * path does for drivers with separate stencil). */
   struct pipe_resource *staging_z, *staging_s;
   uint32_t stride_z, stride_s;
   void *packed;
};

/* One plane of a separate Z/S resource as a resource the copy path
 * understands (image, BO, plane format). */
static struct nvrm_resource
nvrm_zs_plane(const struct nvrm_resource *res, bool stencil)
{
   struct nvrm_resource plane = *res;
   plane.separate_zs = false;
   if (stencil) {
      plane.image = res->stencil_image;
      plane.bo = res->stencil_bo;
      plane.base.format = PIPE_FORMAT_S8_UINT;
   } else {
      plane.base.format = util_format_get_depth_only(res->base.format);
   }
   return plane;
}

static void
nvrm_zs_copy(struct nvrm_context *ctx, struct nvrm_transfer *x, bool to_image)
{
   struct nvrm_resource *res = nvrm_resource(x->base.resource);
   const struct pipe_box *box = &x->base.box;
   const uint64_t lz = (uint64_t)x->stride_z * box->height;
   const uint64_t ls = (uint64_t)x->stride_s * box->height;
   struct nvrm_resource z = nvrm_zs_plane(res, false), s = nvrm_zs_plane(res, true);
   if (to_image) {
      nvrm_copy_image_upload(ctx, &z, &res->base, x->base.level, box, x->staging_z,
                             x->stride_z, lz);
      nvrm_copy_image_upload(ctx, &s, &res->base, x->base.level, box, x->staging_s,
                             x->stride_s, ls);
      return;
   }
   nvrm_copy_image_linear_sync(ctx, &z, x->base.level, box,
                               nvrm_resource(x->staging_z)->bo.va, x->stride_z, lz, to_image);
   nvrm_copy_image_linear_sync(ctx, &s, x->base.level, box,
                               nvrm_resource(x->staging_s)->bo.va, x->stride_s, ls, to_image);
}

/* pack (planes -> packed) or unpack (packed -> planes). */
static void
nvrm_zs_interleave(struct nvrm_transfer *x, bool pack)
{
   const struct pipe_box *box = &x->base.box;
   const enum pipe_format f = x->base.resource->format;
   uint8_t *zp = nvrm_resource(x->staging_z)->bo.map;
   uint8_t *sp = nvrm_resource(x->staging_s)->bo.map;
   for (int l = 0; l < box->depth; l++) {
      for (int y = 0; y < box->height; y++) {
         const uint64_t row = (uint64_t)l * box->height + y;
         uint8_t *prow = (uint8_t *)x->packed + l * x->base.layer_stride + y * x->base.stride;
         uint32_t *zrow = (uint32_t *)(zp + row * x->stride_z);
         uint8_t *srow = sp + row * x->stride_s;
         for (int i = 0; i < box->width; i++) {
            if (f == PIPE_FORMAT_Z24_UNORM_S8_UINT) {
               uint32_t *v = (uint32_t *)prow + i;
               if (pack)
                  *v = (zrow[i] & 0xffffff) | ((uint32_t)srow[i] << 24);
               else {
                  zrow[i] = *v & 0xffffff;
                  srow[i] = *v >> 24;
               }
            } else { /* Z32_FLOAT_S8X24_UINT */
               uint32_t *v = (uint32_t *)prow + 2 * i;
               if (pack) {
                  v[0] = zrow[i];
                  v[1] = srow[i];
               } else {
                  zrow[i] = v[0];
                  srow[i] = v[1] & 0xff;
               }
            }
         }
      }
   }
}

/* GART window onto a buffer. Small BAR: every map of a VRAM buffer, as
 * nvc0's VRAM-domain transfers (nouveau_buffer.c:424-452); any BAR: busy
 * discarding writes (see nvrm_resource_map). Discarding maps get an empty
 * staging area, other maps download the range first (flush, copy, flush:
 * the download waits for earlier GPU writes and completes before the CPU
 * reads); writes go back on unmap through nvrm_buffer_unmap_staging. */
static void *
nvrm_buffer_map_staging(struct nvrm_context *ctx, struct nvrm_resource *res,
                        unsigned usage, const struct pipe_box *box,
                        struct nvrm_transfer *xfer)
{
   struct pipe_context *pctx = &ctx->base;
   if (usage & PIPE_MAP_PERSISTENT) {
      nvrm_error("persistent map of a VRAM buffer without MAP_PERSISTENT flags\n");
      abort();
   }
   if (usage & (PIPE_MAP_DISCARD_RANGE | PIPE_MAP_DISCARD_WHOLE_RESOURCE)) {
      /* Write-only window: suballocated from the stream uploader (nvc0's
       * nouveau_transfer_staging takes small windows from its GART
       * mm_GART suballocator, nouveau_buffer.c:115-140; u_upload never
       * reuses a range and the batch keeps the buffer, so no wait). */
      void *ptr = NULL;
      u_upload_alloc_ref(pctx->stream_uploader, 0, box->width, 256, &xfer->staging_offset,
                         &xfer->staging, &ptr);
      if (!xfer->staging || !ptr) {
         nvrm_error("buffer map: %d B upload window allocation failed\n", box->width);
         pipe_resource_reference(&xfer->staging, NULL);
         pipe_resource_reference(&xfer->base.resource, NULL);
         FREE(xfer);
         return NULL;
      }
      return ptr;
   }
   xfer->staging = pipe_buffer_create(pctx->screen, 0, PIPE_USAGE_STAGING, box->width);
   if (!xfer->staging) {
      nvrm_error("buffer map: %d B GART staging allocation failed\n", box->width);
      pipe_resource_reference(&xfer->base.resource, NULL);
      FREE(xfer);
      return NULL;
   }
   xfer->staging_offset = 0;
   struct nvrm_resource *st = nvrm_resource(xfer->staging);
   {
      if (!(usage & PIPE_MAP_UNSYNCHRONIZED) || nvrm_kept(ctx, &res->base))
         pctx->flush(pctx, NULL, 0);
      nvrm_keep(ctx, &res->base);
      nvrm_keep(ctx, xfer->staging);
      /* Transform-feedback / shader writes of earlier batches drain first. */
      nvrm_push_render_wfi(ctx);
      nvrm_push_copy_buffer(ctx, st->bo.va, res->bo.va + box->x, box->width);
      pctx->flush(pctx, NULL, 0);
      if (!nvrm_wait_token(nvrm_screen(pctx->screen), ctx->last_token)) {
         nvrm_error("buffer download did not complete\n");
         abort();
      }
   }
   return st->bo.map;
}

/* The copy back is recorded into the batch: the channel runs it after the
 * work queued before (reads of the old contents) and before later draws,
 * the in-channel copy-engine ordering NVK relies on for vkCmdCopyBuffer
 * with transfer barriers (nvk_cmd_buffer.c:440-475 emits no WFI for copy
 * writes). Readers then need NVK's TRANSFER_WRITE -> UNIFORM/STORAGE_READ
 * invalidates (nvk_cmd_buffer.c:477-511): global + constant shader caches,
 * emitted by the next draw (NVRM_INVAL_CONSTANT | NVRM_INVAL_GLOBAL) and
 * every compute launch. */
static void
nvrm_buffer_unmap_staging(struct nvrm_context *ctx, struct nvrm_transfer *xfer)
{
   struct nvrm_resource *res = nvrm_resource(xfer->base.resource);
   struct nvrm_resource *st = nvrm_resource(xfer->staging);
   nvrm_keep(ctx, &res->base);
   nvrm_keep(ctx, xfer->staging);
   /* Transform-feedback writes drain first (NVK RENDER_WFI, as images). */
   if (res->base.bind & PIPE_BIND_STREAM_OUTPUT)
      nvrm_push_render_wfi(ctx);
   nvrm_push_copy_buffer(ctx, res->bo.va + xfer->base.box.x,
                         st->bo.va + xfer->staging_offset, xfer->base.box.width);
   ctx->inval |= NVRM_INVAL_CONSTANT | NVRM_INVAL_GLOBAL;
}

static void *
nvrm_resource_map(struct pipe_context *pctx, struct pipe_resource *pres,
                  unsigned level, unsigned usage, const struct pipe_box *box,
                  struct pipe_transfer **out)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   struct nvrm_resource *res = nvrm_resource(pres);
   struct nvrm_transfer *xfer = CALLOC_STRUCT(nvrm_transfer);
   if (!xfer)
      return NULL;
   pipe_resource_reference(&xfer->base.resource, pres);
   xfer->base.level = level;
   xfer->base.usage = usage;
   xfer->base.box = *box;
   *out = &xfer->base;

   if (pres->target == PIPE_BUFFER && !res->bo.map && res->bo.vram)
      return nvrm_buffer_map_staging(ctx, res, usage, box, xfer);
   /* A discarding write to a buffer the pending batch still uses: nvc0's
    * staging transfer for busy buffers with DISCARD_RANGE
    * (nouveau_buffer.c:500-506) instead of flushing and waiting; the copy
    * back on unmap is ordered after the batch's earlier reads and before
    * its later ones. Persistent maps keep the direct pointer. */
   if (pres->target == PIPE_BUFFER && !res->imported &&
       (usage & PIPE_MAP_WRITE) && !(usage & (PIPE_MAP_READ | PIPE_MAP_UNSYNCHRONIZED |
                                             PIPE_MAP_PERSISTENT)) &&
       (usage & (PIPE_MAP_DISCARD_RANGE | PIPE_MAP_DISCARD_WHOLE_RESOURCE)) &&
       nvrm_resource_busy(ctx, res))
      return nvrm_buffer_map_staging(ctx, res, usage, box, xfer);

   if (!res->tiled) {
      /* Busy tracking, as nvc0's per-buffer fences (nouveau_buffer_busy /
       * nouveau_buffer_sync, nouveau_buffer.c:493-506): flush the buffer
       * out of this context's batch if it is there, then wait only for
       * its last submitted use. Linear textures may also be written by
       * this context's render shadows or dma-buf peers: flush + wait for
       * everything this context submitted as well. */
      if (!(usage & PIPE_MAP_UNSYNCHRONIZED)) {
         if (pres->target != PIPE_BUFFER) {
            pctx->flush(pctx, NULL, 0);
            if (!nvrm_wait_token(nvrm_screen(pctx->screen), ctx->last_token)) {
               nvrm_error("map: wait for the context's GPU work failed\n");
               abort();
            }
         }
         nvrm_resource_wait_idle(ctx, res);
      }
      if (!res->bo.map) {
         /* Imported linear dma-buf: CPU map on first use (nv_rm_map, as
          * NVK's rm_mem_map for imported memory). */
         uintptr_t cpu = 0;
         int32_t status = nv_rm_map(nvrm_screen(pctx->screen)->rm, res->bo.mem, &cpu);
         if (status || !cpu) {
            nvrm_error("map of imported linear resource: nv_rm_map failed: %d\n", status);
            pipe_resource_reference(&xfer->base.resource, NULL);
            FREE(xfer);
            *out = NULL;
            return NULL;
         }
         res->bo.map = (void *)cpu;
      }
      xfer->base.stride = res->stride;
      xfer->base.layer_stride = (uintptr_t)res->stride * pres->height0;
      uint8_t *map = (uint8_t *)res->bo.map + res->plane_offset_B;
      if (pres->target == PIPE_BUFFER)
         return map + box->x;
      return map + (uint64_t)box->y / util_format_get_blockheight(pres->format) *
                      res->stride +
             box->x / util_format_get_blockwidth(pres->format) *
                util_format_get_blocksize(pres->format);
   }

   if (res->separate_zs) {
      if (pres->format != PIPE_FORMAT_Z24_UNORM_S8_UINT &&
          pres->format != PIPE_FORMAT_Z32_FLOAT_S8X24_UINT) {
         nvrm_error("map of separate-plane depth/stencil %s is not implemented\n",
                    util_format_short_name(pres->format));
         abort();
      }
      xfer->stride_z = align(box->width * 4, 64);
      xfer->stride_s = align(box->width, 64);
      xfer->base.stride = util_format_get_stride(pres->format, box->width);
      xfer->base.layer_stride = (uint64_t)xfer->base.stride * box->height;
      xfer->staging_z = pipe_buffer_create(pctx->screen, 0, PIPE_USAGE_STAGING,
                                           (uint64_t)xfer->stride_z * box->height * box->depth);
      xfer->staging_s = pipe_buffer_create(pctx->screen, 0, PIPE_USAGE_STAGING,
                                           (uint64_t)xfer->stride_s * box->height * box->depth);
      xfer->packed = calloc(1, xfer->base.layer_stride * box->depth);
      if (!xfer->staging_z || !xfer->staging_s || !xfer->packed) {
         nvrm_error("separate Z/S map staging allocation failed\n");
         abort();
      }
      if (!(usage & (PIPE_MAP_DISCARD_RANGE | PIPE_MAP_DISCARD_WHOLE_RESOURCE))) {
         nvrm_zs_copy(ctx, xfer, false);
         nvrm_zs_interleave(xfer, true);
      }
      return xfer->packed;
   }
   /* Tiled VRAM image: the box goes through a linear staging buffer and the
    * copy engine (download on map unless the whole box is discarded,
    * upload on unmap for writes). */
   const enum pipe_format format = pres->format;
   const uint32_t stride = align(util_format_get_stride(format, box->width), 64);
   const uint64_t layer_stride = (uint64_t)stride * util_format_get_nblocksy(format, box->height);
   const uint64_t size = layer_stride * box->depth;
   xfer->staging = pipe_buffer_create(pctx->screen, 0, PIPE_USAGE_STAGING, size);
   if (!xfer->staging) {
      pipe_resource_reference(&xfer->base.resource, NULL);
      FREE(xfer);
      return NULL;
   }
   xfer->base.stride = stride;
   xfer->base.layer_stride = layer_stride;
   struct nvrm_resource *st = nvrm_resource(xfer->staging);
   if (!(usage & (PIPE_MAP_DISCARD_RANGE | PIPE_MAP_DISCARD_WHOLE_RESOURCE)))
      nvrm_copy_image_linear_sync(ctx, res, level, box, st->bo.va, stride, layer_stride,
                                  false);
   return st->bo.map;
}

static void
nvrm_resource_unmap(struct pipe_context *pctx, struct pipe_transfer *ptrans)
{
   struct nvrm_transfer *xfer = (struct nvrm_transfer *)ptrans;
   if (xfer->packed) {
      if (ptrans->usage & PIPE_MAP_WRITE) {
         nvrm_zs_interleave(xfer, false);
         nvrm_zs_copy(nvrm_context(pctx), xfer, true);
      }
      pipe_resource_reference(&xfer->staging_z, NULL);
      pipe_resource_reference(&xfer->staging_s, NULL);
      free(xfer->packed);
   }
   if (xfer->staging && ptrans->resource->target == PIPE_BUFFER) {
      if (ptrans->usage & PIPE_MAP_WRITE)
         nvrm_buffer_unmap_staging(nvrm_context(pctx), xfer);
      pipe_resource_reference(&xfer->staging, NULL);
   } else if (xfer->staging) {
      if (ptrans->usage & PIPE_MAP_WRITE)
         nvrm_copy_image_upload(nvrm_context(pctx), nvrm_resource(ptrans->resource),
                                ptrans->resource, ptrans->level, &ptrans->box, xfer->staging,
                                ptrans->stride, ptrans->layer_stride);
      pipe_resource_reference(&xfer->staging, NULL);
   }
   pipe_resource_reference(&ptrans->resource, NULL);
   FREE(xfer);
}

void
nvrm_init_transfer_functions(struct pipe_context *pctx)
{
   pctx->buffer_map = nvrm_resource_map;
   pctx->texture_map = nvrm_resource_map;
   pctx->buffer_unmap = nvrm_resource_unmap;
   pctx->texture_unmap = nvrm_resource_unmap;
   pctx->transfer_flush_region = u_default_transfer_flush_region;
   pctx->buffer_subdata = u_default_buffer_subdata;
   pctx->texture_subdata = u_default_texture_subdata;
}
