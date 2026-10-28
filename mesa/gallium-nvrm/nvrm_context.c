/* SPDX-License-Identifier: MIT
 *
 * nvrm context: framebuffer state, CLEAR_SURFACE colour clears, flush with
 * fences, transfers. Draw state and draws are in nvrm_draw.c (G0b); compute
 * launches and blits fail loudly.
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nvrm_priv.h"

#include "nil.h"
#include "nv_push.h"
#include "nv_push_cl9097.h"
#include "nv_push_clc597.h"
#include "nv_push_clcd97.h"
#include "clb197.h"
#include "clc097.h"
#include "clcd97.h"
#include "nvk_pascal_a.h" /* PASCAL_A clear bracket, shared with NVK */

#include "util/format/u_format.h"
#include "util/u_framebuffer.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_surface.h"
#include "util/ralloc.h"
#include "util/bitscan.h"
#include "util/u_upload_mgr.h"
#include "util/u_blitter.h"
#include "util/u_atomic.h"
#include "util/os_file.h"
#include <unistd.h>


static void nvrm_close_segment(struct nvrm_context *ctx, bool incomplete);

static uint32_t *
nvrm_push_limit(const struct nvrm_context *ctx)
{
   return (uint32_t *)((uint8_t *)ctx->push_bo.map + NVRM_PUSH_SIZE_B);
}

static void
nvrm_push_use_chunk(struct nvrm_context *ctx, struct nvrm_push_chunk *c)
{
   ctx->push_chunk = c;
   ctx->push_bo = c->bo;
   ctx->push_start = ctx->push_cur = c->bo.map;
}

/* Chunks one open batch may fill before it is submitted at the next command
 * boundary (nvrm_push_reserve): half the budget, so the other half is either
 * not allocated yet or in flight, and a chunk can always be had without a
 * submission in the middle of a command. One such kick per 8 MiB of commands
 * is what the stock GL driver does when a push segment fills up (an implicit
 * kick, observed in its push stream), at a larger size. */
#define NVRM_PUSH_BATCH_MAX_CHUNKS (NVRM_PUSH_MEM_MAX_B / NVRM_PUSH_SIZE_B / 2)

/* An idle or new chunk for the open batch, as NVK's nvk_cmd_buffer_alloc_mem
 * from the command pool (nvk_cmd_pool.c:20-60), bounded by
 * NVRM_PUSH_MEM_MAX_B: below the budget a busy oldest chunk means a new one;
 * at the budget the oldest chunk's submission is waited for. */
static struct nvrm_push_chunk *
nvrm_push_get_chunk(struct nvrm_context *ctx)
{
   struct nvrm_screen *screen = nvrm_screen(ctx->base.screen);
   const unsigned free_count =
      util_dynarray_num_elements(&ctx->push_free_chunks, struct nvrm_push_chunk *);
   struct nvrm_push_chunk *oldest = free_count ?
      *util_dynarray_element(&ctx->push_free_chunks, struct nvrm_push_chunk *, 0) : NULL;
   const bool at_budget =
      (uint64_t)(ctx->push_chunk_count + 1) * NVRM_PUSH_SIZE_B > NVRM_PUSH_MEM_MAX_B;
   if (!oldest || (!at_budget && !nvrm_token_done(screen, oldest->token))) {
      if (at_budget) {
         /* Unreachable by construction (NVRM_PUSH_BATCH_MAX_CHUNKS). */
         nvrm_error("push chunk budget of %u B exhausted by unsubmitted commands\n",
                    NVRM_PUSH_MEM_MAX_B);
         abort();
      }
      struct nvrm_push_chunk *c = CALLOC_STRUCT(nvrm_push_chunk);
      if (!c || !nvrm_bo_alloc(screen, NVRM_PUSH_SIZE_B, NVRM_DOMAIN_GART, &c->bo)) {
         nvrm_error("push chunk of %u B could not be allocated (%u chunks owned)\n",
                    NVRM_PUSH_SIZE_B, ctx->push_chunk_count);
         abort(); /* no silent loss of commands */
      }
      ctx->push_chunk_count++;
      return c;
   }
   if (!nvrm_wait_token(screen, oldest->token)) {
      nvrm_error("push chunk recycle wait failed\n");
      abort();
   }
   memmove(util_dynarray_begin(&ctx->push_free_chunks),
           (struct nvrm_push_chunk **)util_dynarray_begin(&ctx->push_free_chunks) + 1,
           (free_count - 1) * sizeof(struct nvrm_push_chunk *));
   ctx->push_free_chunks.size -= sizeof(struct nvrm_push_chunk *);
   oldest->token = 0;
   return oldest;
}

/* The current chunk is full: it becomes a GPFIFO entry of the open batch and
 * recording continues in another chunk (nvk_cmd_buffer_new_push). Commands
 * never straddle chunks (callers reserve whole commands). */
static void
nvrm_push_next_chunk(struct nvrm_context *ctx)
{
   nvrm_close_segment(ctx, false);
   /* The full chunk belongs to the open batch from here on. */
   util_dynarray_append_typed(&ctx->push_batch_chunks, struct nvrm_push_chunk *, ctx->push_chunk);
   nvrm_push_use_chunk(ctx, nvrm_push_get_chunk(ctx));
}

/* Makes room now for a later nvrm_push_begin(ctx, dw), so CPU writes into
 * the open batch's texture pool made in between land in the pool the
 * command binds: the only flush left on this path is the GPFIFO entry limit
 * of one submission. */
void
nvrm_push_reserve(struct nvrm_context *ctx, uint32_t dw)
{
   const uint32_t execs = util_dynarray_num_elements(&ctx->execs, struct NvRmExec);
   const unsigned chunks =
      util_dynarray_num_elements(&ctx->push_batch_chunks, struct nvrm_push_chunk *);
   if (execs + NVRM_EXEC_HEADROOM > NVRM_MAX_EXECS || chunks >= NVRM_PUSH_BATCH_MAX_CHUNKS)
      ctx->base.flush(&ctx->base, NULL, 0);
   if (ctx->push_cur + dw > nvrm_push_limit(ctx))
      nvrm_push_next_chunk(ctx);
}

struct nv_push
nvrm_push_begin(struct nvrm_context *ctx, uint32_t dw)
{
   if (dw > NVRM_PUSH_SIZE_B / 4) {
      nvrm_error("push of %u dwords is larger than a push chunk\n", dw);
      abort(); /* a single command larger than a chunk is a bug */
   }
   nvrm_push_reserve(ctx, dw);
   struct nv_push p;
   nv_push_init(&p, ctx->push_cur, dw, SUBC_MASK_ALL);
   return p;
}

void
nvrm_push_end(struct nvrm_context *ctx, struct nv_push *p)
{
   ctx->push_cur = p->end;
}

/* Context teardown after its last flush: the backend defers the frees past
 * submitted work, as for every other buffer. */
static void
nvrm_push_chunks_free(struct nvrm_context *ctx)
{
   struct nvrm_screen *screen = nvrm_screen(ctx->base.screen);
   util_dynarray_foreach(&ctx->push_batch_chunks, struct nvrm_push_chunk *, c) {
      nvrm_bo_free(screen, &(*c)->bo);
      FREE(*c);
   }
   util_dynarray_foreach(&ctx->push_free_chunks, struct nvrm_push_chunk *, c) {
      nvrm_bo_free(screen, &(*c)->bo);
      FREE(*c);
   }
   util_dynarray_fini(&ctx->push_batch_chunks);
   util_dynarray_fini(&ctx->push_free_chunks);
   if (ctx->push_chunk) {
      nvrm_bo_free(screen, &ctx->push_chunk->bo);
      FREE(ctx->push_chunk);
      ctx->push_chunk = NULL;
   }
}

bool
nvrm_has_pending(const struct nvrm_context *ctx)
{
   return ctx->push_cur != ctx->push_start || ctx->execs.size;
}

/* Closes [push_start, push_cur) as one GPFIFO entry (nvk_cmd_buffer_flush_push,
 * nvk_cmd_buffer.c:150-166); empty segments are not recorded (the backend
 * refuses zero-size entries). */
static void
nvrm_close_segment(struct nvrm_context *ctx, bool incomplete)
{
   if (ctx->push_cur == ctx->push_start)
      return;
   const uint64_t offset = (uint8_t *)ctx->push_start - (uint8_t *)ctx->push_bo.map;
   const struct NvRmExec e = {
      .address = ctx->push_bo.va + offset,
      .size_B = (uint32_t)((uint8_t *)ctx->push_cur - (uint8_t *)ctx->push_start),
      .flags = incomplete ? NV_RM_EXEC_INCOMPLETE : 0,
   };
   util_dynarray_append_typed(&ctx->execs, struct NvRmExec, e);
   ctx->push_start = ctx->push_cur;
}

void
nvrm_push_indirect(struct nvrm_context *ctx, struct nv_push *p,
                   uint64_t addr, uint32_t range)
{
   if (!range || (addr & 3) || (range & 3)) {
      nvrm_error("push-indirect range 0x%" PRIx64 "+%u is not a dword range\n", addr, range);
      abort();
   }
   if (util_dynarray_num_elements(&ctx->execs, struct NvRmExec) + 2 > NVRM_MAX_EXECS) {
      nvrm_error("push-indirect: more than %u GPFIFO entries in one command\n",
                 NVRM_MAX_EXECS);
      abort();
   }
   ctx->push_cur = p->end;
   nvrm_close_segment(ctx, true);
   const struct NvRmExec e = {
      .address = addr,
      .size_B = range,
      .flags = NV_RM_EXEC_NO_PREFETCH,
   };
   util_dynarray_append_typed(&ctx->execs, struct NvRmExec, e);
}

void
nvrm_keep(struct nvrm_context *ctx, struct pipe_resource *res)
{
   bool found;
   _mesa_set_search_or_add(ctx->keep, res, &found);
   if (!found) {
      struct pipe_resource *ref = NULL;
      pipe_resource_reference(&ref, res);
   }
}

bool
nvrm_kept(const struct nvrm_context *ctx, const struct pipe_resource *res)
{
   return _mesa_set_search(ctx->keep, res) != NULL;
}

static void
nvrm_unref_kept(struct set_entry *entry)
{
   struct pipe_resource *res = (struct pipe_resource *)entry->key;
   pipe_resource_reference(&res, NULL);
}

static void
nvrm_release_kept(struct nvrm_context *ctx)
{
   _mesa_set_clear(ctx->keep, nvrm_unref_kept);
}

/* Tokens of the screen channel grow with submission order. */
static void
nvrm_resource_mark_used(struct nvrm_resource *res, uint64_t token)
{
   uint64_t old = p_atomic_read(&res->last_token);
   while (old < token) {
      const uint64_t prev = p_atomic_cmpxchg(&res->last_token, old, token);
      if (prev == old)
         break;
      old = prev;
   }
}

bool
nvrm_resource_busy(struct nvrm_context *ctx, struct nvrm_resource *res)
{
   return nvrm_kept(ctx, &res->base) ||
          !nvrm_token_done(nvrm_screen(ctx->base.screen), p_atomic_read(&res->last_token));
}

void
nvrm_resource_wait_idle(struct nvrm_context *ctx, struct nvrm_resource *res)
{
   if (nvrm_kept(ctx, &res->base))
      ctx->base.flush(&ctx->base, NULL, 0);
   if (!nvrm_wait_token(nvrm_screen(ctx->base.screen), p_atomic_read(&res->last_token))) {
      nvrm_error("wait for a resource's GPU work failed\n");
      abort();
   }
}

uint64_t
nvrm_seq_token(const struct nvrm_context *ctx, uint64_t seq)
{
   /* Every flush records its seq (with the previous token when it submitted
    * nothing), so the ring holds [flush_seq - NVRM_SEQ_RING, flush_seq). */
   assert(seq < ctx->flush_seq);
   if (seq + NVRM_SEQ_RING < ctx->flush_seq)
      seq = ctx->flush_seq - NVRM_SEQ_RING; /* evicted: a newer batch */
   assert(ctx->seq_token[seq % NVRM_SEQ_RING].seq == seq);
   return ctx->seq_token[seq % NVRM_SEQ_RING].token;
}

/* Shader ranges whose last possible user completed go back to the heap. */
static void
nvrm_reclaim_shader_ranges(struct nvrm_screen *screen)
{
   simple_mtx_lock(&screen->shader_mutex);
   uint64_t *r = screen->shader_pending.data;
   unsigned n = screen->shader_pending.size / sizeof(uint64_t), kept = 0;
   for (unsigned i = 0; i + 3 <= n; i += 3) {
      if (nvrm_token_done(screen, r[i + 2])) {
         util_vma_heap_free(&screen->shader_vma, r[i], r[i + 1]);
      } else {
         r[kept++] = r[i];
         r[kept++] = r[i + 1];
         r[kept++] = r[i + 2];
      }
   }
   screen->shader_pending.size = kept * sizeof(uint64_t);
   simple_mtx_unlock(&screen->shader_mutex);
}

/* ---------------------------------------------------------------- linear render shadows (D1) */

/* nvk_rendering_linear (nvk_cmd_draw.c:935-963): depth/stencil is never
 * linear, and linear colour targets need a 128 B aligned base and pitch;
 * any block-linear colour target makes the whole pass non-linear. */
static bool
nvrm_fb_renders_linear(const struct pipe_framebuffer_state *fb)
{
   if (fb->zsbuf.texture)
      return false;
   for (unsigned i = 0; i < fb->nr_cbufs; i++) {
      if (!fb->cbufs[i].texture)
         continue;
      const struct nvrm_resource *res = nvrm_resource(fb->cbufs[i].texture);
      if (res->tiled)
         return false;
      const uint64_t addr = res->bo.va + res->plane_offset_B;
      if (addr % 128 || res->stride % 128)
         return false;
   }
   return true;
}

/* The linear colour buffer bound at slot i, checked to be renderable at
 * all (single level/layer, not a YUV plane). */
static struct nvrm_resource *
nvrm_fb_linear_cbuf(const struct pipe_surface *surf)
{
   struct nvrm_resource *res = nvrm_resource(surf->texture);
   if (res->plane_offset_B) {
      nvrm_error("colour target on a YUV plane import (offset %u B): YUV "
                 "dma-bufs are imported for sampling only\n", res->plane_offset_B);
      abort();
   }
   if (surf->level != 0 || surf->first_layer != 0 || surf->last_layer != 0) {
      nvrm_error("linear colour target level %u layers %u-%u\n",
                 surf->level, surf->first_layer, surf->last_layer);
      abort();
   }
   return res;
}

/* nvk_linear_render_copy (nvk_cmd_copy.c:642-688): the whole level-0 image
 * between a linear buffer and its block-linear shadow on the copy engine,
 * recorded inline in the open batch like NVK's copies at the render pass
 * boundaries (nvk_CmdBeginRendering nvk_cmd_draw.c:1420-1427, copy-in for
 * LOAD_OP_LOAD; nvk_CmdEndRendering :1497-1503, copy-out for STORE_OP_STORE),
 * no submission of its own. */
static void
nvrm_fb_shadow_copy(struct nvrm_context *ctx, bool to_shadow)
{
   /* Rendering into / sampling of the shadows drains before the copy
    * engine touches them (NVK RENDER_WFI for attachment writes). */
   nvrm_push_render_wfi(ctx);
   u_foreach_bit(i, ctx->shadow_mask) {
      struct nvrm_resource *res = nvrm_fb_linear_cbuf(&ctx->fb.cbufs[i]);
      const struct pipe_box box = {
         .width = res->base.width0, .height = res->base.height0, .depth = 1,
      };
      nvrm_copy_image_linear(ctx, nvrm_resource(res->render_shadow), 0, &box,
                             res->bo.va, res->stride, 0, to_shadow);
   }
}

/* Every linear colour buffer of the framebuffer gets its block-linear shadow
 * (NVK allocates linear_tiled_shadow at image creation, nvk_image.c:1250-1262;
 * gallium learns about the use only here). */
static void
nvrm_fb_shadow_alloc(struct nvrm_context *ctx)
{
   struct pipe_screen *pscreen = ctx->base.screen;
   u_foreach_bit(i, ctx->shadow_mask) {
      struct nvrm_resource *res = nvrm_fb_linear_cbuf(&ctx->fb.cbufs[i]);
      if (p_atomic_read(&res->render_shadow))
         continue;
      const struct pipe_resource templ = {
         .target = PIPE_TEXTURE_2D,
         .format = res->base.format,
         .width0 = res->base.width0,
         .height0 = res->base.height0,
         .depth0 = 1,
         .array_size = 1,
         .bind = PIPE_BIND_RENDER_TARGET,
      };
      struct pipe_resource *shadow = pscreen->resource_create(pscreen, &templ);
      if (!shadow || !nvrm_resource(shadow)->tiled) {
         nvrm_error("block-linear render shadow for a linear %s %ux%u colour "
                    "buffer could not be created\n",
                    util_format_short_name(res->base.format),
                    res->base.width0, res->base.height0);
         abort();
      }
      /* Shared resources: the first context to install a shadow wins. */
      if (p_atomic_cmpxchg(&res->render_shadow, NULL, shadow) != NULL)
         pipe_resource_reference(&shadow, NULL);
   }
}

/* Before a draw or clear on a framebuffer that cannot render linear: the
 * shadows get the linear contents once per batch unless they already hold
 * them (GL preserves colour contents: NVK's LOAD_OP_LOAD copy,
 * nvk_cmd_draw.c:1420-1427), recorded inline before the rendering. */
void
nvrm_fb_shadow_begin(struct nvrm_context *ctx)
{
   if (!ctx->shadow_mask || ctx->shadow_loaded)
      return;
   nvrm_fb_shadow_alloc(ctx);
   nvrm_fb_shadow_copy(ctx, true);
   ctx->shadow_loaded = true;
}

/* End of the shadows' "render pass": what was rendered goes back to the
 * linear buffers, inline in the open batch (NVK's STORE_OP_STORE copy in
 * nvk_CmdEndRendering, nvk_cmd_draw.c:1497-1503; render WFI first in
 * nvrm_fb_shadow_copy). The shadows still equal the linear contents
 * afterwards, so a later draw in the same batch needs no copy-in. */
void
nvrm_fb_shadow_end(struct nvrm_context *ctx)
{
   if (!ctx->shadow_written)
      return;
   ctx->shadow_written = false;
   nvrm_fb_shadow_copy(ctx, false);
}

/* The pass starts with contents that do not depend on the old ones: a clear
 * of every shadowed buffer over its whole extent (or a discard), so no
 * copy-in; NVK copies in only for LOAD_OP_LOAD (nvk_cmd_draw.c:1423-1426),
 * a full-area clear is LOAD_OP_CLEAR and a discard LOAD_OP_DONT_CARE. */
static void
nvrm_fb_shadow_begin_overwritten(struct nvrm_context *ctx)
{
   if (!ctx->shadow_mask || ctx->shadow_loaded)
      return;
   nvrm_fb_shadow_alloc(ctx);
   ctx->shadow_loaded = true;
}

/* ---------------------------------------------------------------- flush */

static void
nvrm_flush(struct pipe_context *pctx, struct pipe_fence_handle **fence,
           unsigned flags)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   struct nvrm_screen *screen = nvrm_screen(pctx->screen);
   uint64_t token = 0;
   /* The linear buffers are current when this batch completes. */
   nvrm_fb_shadow_end(ctx);
   /* Stream output state is channel state shared by every context: pause
    * it into the counters; the next draw resumes from them. */
   nvrm_so_pause(ctx);
   const uint32_t wait_count = util_dynarray_num_elements(&ctx->wait_fds, int);
   /* A requested fence needs a real completion token for its sync_file
    * (nv_rm_context_fence refuses 0): with no work, an empty batch. Pending
    * fence_server_sync fds gate the next submitted batch (G2). */
   if (nvrm_has_pending(ctx) || fence) {
      nvrm_close_segment(ctx, false);
      const uint32_t exec_count = util_dynarray_num_elements(&ctx->execs, struct NvRmExec);
      if (exec_count && (util_dynarray_top_ptr(&ctx->execs, struct NvRmExec)->flags &
                         NV_RM_EXEC_INCOMPLETE)) {
         nvrm_error("flush: batch ends inside a method (incomplete push)\n");
         abort();
      }
      /* Asynchronous: no CPU wait here (glFinish / fences / CPU maps wait
       * for the tokens they need). */
      if (!nvrm_submit_async(screen, util_dynarray_begin(&ctx->execs), exec_count,
                             util_dynarray_begin(&ctx->wait_fds), wait_count, &token)) {
         nvrm_error("flush: submission of %u entries (%u waits) failed\n", exec_count,
                    wait_count);
         abort(); /* no silent loss of rendering */
      }
      util_dynarray_clear(&ctx->execs);
      util_dynarray_foreach(&ctx->wait_fds, int, fd)
         close(*fd);
      util_dynarray_clear(&ctx->wait_fds);
      ctx->last_token = token;
      ctx->ring[ctx->ring_cur].token = token;
      set_foreach(ctx->keep, e)
         nvrm_resource_mark_used(nvrm_resource((struct pipe_resource *)e->key), token);
      /* The batch's full chunks join the recycle list (submission order);
       * recording continues after push_cur in the current chunk, whose
       * submitted part is protected by the same token. */
      util_dynarray_foreach(&ctx->push_batch_chunks, struct nvrm_push_chunk *, c) {
         (*c)->token = token;
         util_dynarray_append_typed(&ctx->push_free_chunks, struct nvrm_push_chunk *, *c);
      }
      util_dynarray_clear(&ctx->push_batch_chunks);
      ctx->push_chunk->token = token;
      /* Next batch slot: its texture pool is reused once the batch that
       * last used it completed. */
      ctx->ring_cur = (ctx->ring_cur + 1) % NVRM_BATCH_RING;
      struct nvrm_batch_slot *slot = &ctx->ring[ctx->ring_cur];
      if (!nvrm_wait_token(screen, slot->token)) {
         nvrm_error("flush: batch slot recycle wait failed\n");
         abort();
      }
      ctx->tex_pool = slot->tex_pool;
      ctx->tex_pool_next = 0;
   }
   nvrm_release_kept(ctx);
   /* Deleted shader ranges: free once every submission made so far on the
    * channel (any context may have used them) completed. */
   if (ctx->shader_frees.size) {
      const uint64_t last = p_atomic_read(&screen->last_token);
      simple_mtx_lock(&screen->shader_mutex);
      const uint64_t *r = ctx->shader_frees.data;
      for (unsigned i = 0; i + 1 < ctx->shader_frees.size / sizeof(uint64_t); i += 2) {
         const uint64_t e[3] = { r[i], r[i + 1], last };
         util_dynarray_append_array(&screen->shader_pending, uint64_t, e, 3);
      }
      simple_mtx_unlock(&screen->shader_mutex);
      util_dynarray_clear(&ctx->shader_frees);
   }
   nvrm_reclaim_shader_ranges(screen);
   /* Next batch starts from unknown channel state (shared by all contexts). */
   ctx->dirty = NVRM_DIRTY_ALL;
   ctx->seq_token[ctx->flush_seq % NVRM_SEQ_RING].seq = ctx->flush_seq;
   ctx->seq_token[ctx->flush_seq % NVRM_SEQ_RING].token = ctx->last_token;
   ctx->flush_seq++;
   /* The linear buffers may change outside this context between batches
    * (CPU maps flush first, other contexts, dma-buf producers). */
   ctx->shadow_loaded = false;
   if (fence) {
      struct nvrm_fence *f = CALLOC_STRUCT(nvrm_fence);
      pipe_reference_init(&f->reference, 1);
      f->token = token;
      f->fd = -1;
      pctx->screen->fence_reference(pctx->screen, fence, NULL);
      *fence = (struct pipe_fence_handle *)f;
   }
}

/* ---------------------------------------------------------------- clears */

/* Colour targets from the framebuffer: NVK's nvk_CmdBeginRendering
 * (nvk_cmd_draw.c:1060-1210) per attachment, block-linear branch for nil
 * images and the LAYOUT_PITCH branch for linear ones; an unbound slot is
 * NVK's disabled target. CT_SELECT maps colour output i to target i (MRT,
 * up to 8 as NVK's NVK_MAX_RTS). Also programs the surface clip. */
static void
nvrm_emit_one_color_target(struct nvrm_context *ctx, struct nv_push *p, unsigned i,
                           const struct pipe_surface *surf)
{
   if (!surf || !surf->texture) {
      P_MTHD(p, NV9097, SET_COLOR_TARGET_A(i));
      P_NV9097_SET_COLOR_TARGET_A(p, i, 0);
      P_NV9097_SET_COLOR_TARGET_B(p, i, 0);
      P_NV9097_SET_COLOR_TARGET_WIDTH(p, i, 64);
      P_NV9097_SET_COLOR_TARGET_HEIGHT(p, i, 0);
      P_NV9097_SET_COLOR_TARGET_FORMAT(p, i, V_DISABLED);
      P_NV9097_SET_COLOR_TARGET_MEMORY(p, i, { .layout = LAYOUT_BLOCKLINEAR });
      P_NV9097_SET_COLOR_TARGET_THIRD_DIMENSION(p, i, 1);
      P_NV9097_SET_COLOR_TARGET_ARRAY_PITCH(p, i, 0);
      P_NV9097_SET_COLOR_TARGET_LAYER(p, i, 0);
      return;
   }
   struct nvrm_resource *res = nvrm_resource(surf->texture);
   if (!res->tiled && !ctx->render_linear) {
      /* NVK never programs LAYOUT_PITCH unless the whole pass renders
       * linear; the target is the block-linear shadow instead
       * (nvk_CmdBeginRendering, nvk_cmd_draw.c:1093-1095). */
      struct nvrm_resource *lin = nvrm_fb_linear_cbuf(surf);
      if (!(ctx->shadow_mask & (1u << i)) || !lin->render_shadow) {
         nvrm_error("linear colour target %u emitted without its render shadow\n", i);
         abort();
      }
      res = nvrm_resource(lin->render_shadow);
      ctx->shadow_written = true;
   }
   nvrm_keep(ctx, &res->base);
   const uint32_t ct_format = nil_format_to_color_target(surf->format);

   if (res->tiled) {
      const struct nil_image *img = &res->image;
      const struct nil_image_level *lvl = &img->levels[surf->level];
      /* Sizes in samples, as NVK (level_extent_sa). */
      const struct nil_Extent4D_Samples ext = nil_image_level_extent_sa(img, surf->level);
      const uint32_t layers = surf->last_layer - surf->first_layer + 1;
      uint64_t addr = res->bo.va + lvl->offset_B;
      if (img->dim == NIL_IMAGE_DIM_3D)
         addr += nil_image_level_z_offset_B(img, surf->level, surf->first_layer);
      else
         addr += (uint64_t)surf->first_layer * img->array_stride_B;

      P_MTHD(p, NV9097, SET_COLOR_TARGET_A(i));
      P_NV9097_SET_COLOR_TARGET_A(p, i, addr >> 32);
      P_NV9097_SET_COLOR_TARGET_B(p, i, addr);
      P_NV9097_SET_COLOR_TARGET_WIDTH(p, i, lvl->row_stride_B /
                                            util_format_get_blocksize(surf->format));
      P_NV9097_SET_COLOR_TARGET_HEIGHT(p, i, ext.height);
      P_NV9097_SET_COLOR_TARGET_FORMAT(p, i, ct_format);
      P_NV9097_SET_COLOR_TARGET_MEMORY(p, i, {
         .block_width   = BLOCK_WIDTH_ONE_GOB,
         .block_height  = lvl->tiling.y_log2,
         .block_depth   = lvl->tiling.z_log2,
         .layout        = LAYOUT_BLOCKLINEAR,
         .third_dimension_control = img->dim == NIL_IMAGE_DIM_3D ?
            THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_DEPTH_SIZE :
            THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_ARRAY_SIZE,
      });
      P_NV9097_SET_COLOR_TARGET_THIRD_DIMENSION(p, i, layers);
      P_NV9097_SET_COLOR_TARGET_ARRAY_PITCH(p, i, img->array_stride_B >> 2);
      P_NV9097_SET_COLOR_TARGET_LAYER(p, i, 0);
      /* nvrm images are never compressed (nil usage UNCOMPRESSED_BIT,
       * nvrm_resource.c): NVK emits image->is_compressed, false for them. */
      P_IMMD(p, NV9097, SET_COLOR_COMPRESSION(i), false);
   } else {
      /* LAYOUT_PITCH only when the whole pass renders linear: no Z/S,
       * base and pitch 128 B aligned (nvrm_fb_renders_linear). */
      nvrm_fb_linear_cbuf(surf);
      P_MTHD(p, NV9097, SET_COLOR_TARGET_A(i));
      P_NV9097_SET_COLOR_TARGET_A(p, i, res->bo.va >> 32);
      P_NV9097_SET_COLOR_TARGET_B(p, i, res->bo.va);
      /* LAYOUT_PITCH: WIDTH takes the row pitch in bytes. */
      P_NV9097_SET_COLOR_TARGET_WIDTH(p, i, res->stride);
      P_NV9097_SET_COLOR_TARGET_HEIGHT(p, i, surf->texture->height0);
      P_NV9097_SET_COLOR_TARGET_FORMAT(p, i, ct_format);
      P_NV9097_SET_COLOR_TARGET_MEMORY(p, i, {
         .layout = LAYOUT_PITCH,
         .third_dimension_control =
            THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_ARRAY_SIZE,
      });
      P_NV9097_SET_COLOR_TARGET_THIRD_DIMENSION(p, i, 1);
      P_NV9097_SET_COLOR_TARGET_ARRAY_PITCH(p, i, 0);
      P_NV9097_SET_COLOR_TARGET_LAYER(p, i, 0);
      P_IMMD(p, NV9097, SET_COLOR_COMPRESSION(i), false);
   }
}

void
nvrm_emit_color_target(struct nvrm_context *ctx, struct nv_push *p,
                       const struct pipe_framebuffer_state *fb)
{
   if (fb->nr_cbufs > NVRM_MAX_RTS) {
      nvrm_error("framebuffer with %u colour targets (max %u)\n", fb->nr_cbufs, NVRM_MAX_RTS);
      abort();
   }
   /* gallium's framebuffer size is the intersection of all attachments. */
   P_MTHD(p, NV9097, SET_SURFACE_CLIP_HORIZONTAL);
   P_NV9097_SET_SURFACE_CLIP_HORIZONTAL(p, { .x = 0, .width = fb->width });
   P_NV9097_SET_SURFACE_CLIP_VERTICAL(p, { .y = 0, .height = fb->height });
   const unsigned count = MAX2(fb->nr_cbufs, 1);
   for (unsigned i = 0; i < count; i++)
      nvrm_emit_one_color_target(ctx, p, i, i < fb->nr_cbufs ? &fb->cbufs[i] : NULL);
   P_IMMD(p, NV9097, SET_CT_SELECT, {
      .target_count = count,
      .target0 = 0, .target1 = 1, .target2 = 2, .target3 = 3,
      .target4 = 4, .target5 = 5, .target6 = 6, .target7 = 7,
   });
}

/* Depth/stencil target: NVK nvk_CmdBeginRendering (nvk_cmd_draw.c:
 * 1208-1320) incl. the Blackwell separate stencil target (SET_ST_*). */
void
nvrm_emit_zs_target(struct nvrm_context *ctx, struct nv_push *p,
                    const struct pipe_framebuffer_state *fb)
{
   struct nvrm_screen *screen = nvrm_screen(ctx->base.screen);
   const struct pipe_surface *zs = &fb->zsbuf;
   if (!zs->texture) {
      P_IMMD(p, NV9097, SET_ZT_SELECT, 0 /* target_count */);
      return;
   }
   struct nvrm_resource *res = nvrm_resource(zs->texture);
   if (!res->tiled) {
      nvrm_error("linear depth/stencil target\n");
      abort();
   }
   nvrm_keep(ctx, &res->base);
   const uint32_t layers = zs->last_layer - zs->first_layer + 1;
   const struct nil_image *img = &res->image;
   if (img->dim == NIL_IMAGE_DIM_3D) {
      nvrm_error("3D depth/stencil target is not implemented\n");
      abort();
   }
   const struct nil_image_level *lvl = &img->levels[zs->level];
   const struct nil_Extent4D_Samples ext = nil_image_level_extent_sa(img, zs->level);
   const uint64_t addr = res->bo.va + lvl->offset_B;

   P_MTHD(p, NV9097, SET_ZT_A);
   P_NV9097_SET_ZT_A(p, addr >> 32);
   P_NV9097_SET_ZT_B(p, addr);
   const uint8_t zs_format = nil_format_to_depth_stencil(zs->texture->format);
   if (screen->info.cls_eng3d >= BLACKWELL_A) {
      P_NVCD97_SET_ZT_FORMAT(p, {
         .v = zs_format,
         .stencil_is_separate = res->separate_zs,
      });
   } else {
      P_NV9097_SET_ZT_FORMAT(p, zs_format);
   }
   P_NV9097_SET_ZT_BLOCK_SIZE(p, {
      .width = WIDTH_ONE_GOB,
      .height = lvl->tiling.y_log2,
      .depth = DEPTH_ONE_GOB,
   });
   P_NV9097_SET_ZT_ARRAY_PITCH(p, img->array_stride_B >> 2);
   P_IMMD(p, NV9097, SET_ZT_SELECT, 1 /* target_count */);

   const uint32_t row_stride_el =
      lvl->row_stride_B / util_format_get_blocksize(img->format.p_format);
   P_MTHD(p, NV9097, SET_ZT_SIZE_A);
   P_NV9097_SET_ZT_SIZE_A(p, row_stride_el);
   P_NV9097_SET_ZT_SIZE_B(p, ext.height);
   P_NV9097_SET_ZT_SIZE_C(p, {
      .third_dimension  = zs->first_layer + layers,
      .control          = CONTROL_THIRD_DIMENSION_DEFINES_ARRAY_SIZE,
   });
   P_IMMD(p, NV9097, SET_ZT_LAYER, zs->first_layer);
   P_IMMD(p, NV9097, SET_Z_COMPRESSION, false); /* see SET_COLOR_COMPRESSION */
   if (screen->info.cls_eng3d >= MAXWELL_B)
      P_IMMD(p, NVC597, SET_ZT_SPARSE, { .enable = ENABLE_FALSE });

   /* Blackwell: the stencil target is programmed for separate Z/S and
    * for stencil-only S8_UINT (plane 0 then), nvk_cmd_draw.c:1293-1294. */
   const bool s8_only = zs->texture->format == PIPE_FORMAT_S8_UINT;
   if (screen->info.cls_eng3d >= BLACKWELL_A && (res->separate_zs || s8_only)) {
      const struct nil_image *simg = res->separate_zs ? &res->stencil_image : &res->image;
      const struct nvrm_bo *sbo = res->separate_zs ? &res->stencil_bo : &res->bo;
      const struct nil_image_level *slvl = &simg->levels[zs->level];
      const struct nil_Extent4D_Samples sext = nil_image_level_extent_sa(simg, zs->level);
      const uint64_t saddr = sbo->va + slvl->offset_B;
      P_MTHD(p, NVCD97, SET_ST_A);
      P_NVCD97_SET_ST_A(p, saddr >> 32);
      P_NVCD97_SET_ST_B(p, saddr);
      P_MTHD(p, NVCD97, SET_ST_BLOCK_SIZE);
      P_NVCD97_SET_ST_BLOCK_SIZE(p, {
         .width = WIDTH_ONE_GOB,
         .height = slvl->tiling.y_log2,
         .depth = DEPTH_ONE_GOB,
      });
      P_NVCD97_SET_ST_ARRAY_PITCH(p, simg->array_stride_B >> 2);
      P_MTHD(p, NVCD97, SET_ST_SIZE_A);
      P_NVCD97_SET_ST_SIZE_A(p, slvl->row_stride_B /
                                util_format_get_blocksize(PIPE_FORMAT_S8_UINT));
      P_NVCD97_SET_ST_SIZE_B(p, sext.height);
   }
}

/* The clear follows nvk_CmdClearAttachments + emit_clear_rects
 * (nvk_cmd_clear.c:89-185), with the NVK_MME_CLEAR macro's CLEAR_SURFACE
 * written directly, once per bound layer. */
static void
nvrm_clear(struct pipe_context *pctx, unsigned buffers,
           uint32_t color_clear_mask, uint8_t stencil_clear_mask,
           const struct pipe_scissor_state *scissor,
           const union pipe_color_union *color, double depth, unsigned stencil)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   const struct nvrm_screen *screen = nvrm_screen(pctx->screen);
   const struct pipe_framebuffer_state *fb = &ctx->fb;
   /* caps.clear_masked is false: st/mesa clears with a draw whenever a
    * colour or stencil write mask is partial (st_cb_clear.c:442, 467), so
    * every clear reaching here writes all components. */
   (void)color_clear_mask;
   (void)stencil_clear_mask;

   uint32_t color_mask = 0;
   unsigned color_layers = 0;
   for (unsigned i = 0; i < fb->nr_cbufs; i++) {
      if ((buffers & (PIPE_CLEAR_COLOR0 << i)) && fb->cbufs[i].texture) {
         color_mask |= 1u << i;
         color_layers += fb->cbufs[i].last_layer - fb->cbufs[i].first_layer + 1;
      }
   }
   const bool clear_z = (buffers & PIPE_CLEAR_DEPTH) && fb->zsbuf.texture &&
                        util_format_has_depth(util_format_description(fb->zsbuf.format));
   const bool clear_s = (buffers & PIPE_CLEAR_STENCIL) && fb->zsbuf.texture &&
                        util_format_has_stencil(util_format_description(fb->zsbuf.format));
   if (!color_mask && !clear_z && !clear_s)
      return;
   const uint32_t zs_layers = (clear_z || clear_s) ?
      fb->zsbuf.last_layer - fb->zsbuf.first_layer + 1 : 0;

   /* A clear of every shadowed buffer over its whole extent, not subject
    * to conditional rendering, replaces the contents: no copy-in. */
   bool overwrites_shadows = (color_mask & ctx->shadow_mask) == ctx->shadow_mask &&
                             !ctx->cond_query;
   u_foreach_bit(i, ctx->shadow_mask) {
      const struct pipe_resource *t = fb->cbufs[i].texture;
      if (fb->width < t->width0 || fb->height < t->height0 ||
          (scissor && (scissor->minx || scissor->miny ||
                       scissor->maxx < t->width0 || scissor->maxy < t->height0)))
         overwrites_shadows = false;
   }
   /* A clear that writes no shadowed colour buffer (depth/stencil only, or
    * other colour targets) neither reads nor writes the shadows: they only
    * need to exist for the target emission (NVK: the copies belong to the
    * attachment's own load/store ops, nvk_cmd_draw.c:1420-1427/1497-1503). */
   const bool writes_shadows = (color_mask & ctx->shadow_mask) != 0;
   const bool shadow_written = ctx->shadow_written;
   const uint64_t flush_seq = ctx->flush_seq;
   if (!writes_shadows)
      nvrm_fb_shadow_alloc(ctx);
   else if (overwrites_shadows)
      nvrm_fb_shadow_begin_overwritten(ctx);
   else
      nvrm_fb_shadow_begin(ctx);
   const uint16_t cls_eng3d = screen->info.cls_eng3d;
   struct nv_push push = nvrm_push_begin(ctx, 400 + 2 * (color_layers + zs_layers) +
                                         nvk_pascal_a_clear_begin_dw(cls_eng3d) +
                                         nvk_pascal_a_clear_end_dw(cls_eng3d));
   struct nv_push *p = &push;
   /* Earlier draws may still sample the surface being cleared (dEQP
    * fbo.render.repeated_clear: clear A, draw from A, clear A): the same
    * 3D WAIT_FOR_IDLE barrier the draw path emits on a framebuffer change. */
   P_IMMD(p, NV9097, WAIT_FOR_IDLE, 0);
   nvrm_emit_color_target(ctx, p, fb);
   nvrm_emit_zs_target(ctx, p, fb);
   nvrm_emit_sample_layout(ctx, p, fb);
   nvrm_emit_render_enable(ctx, p); /* clears obey conditional rendering */
   if (clear_z)
      P_IMMD(p, NV9097, SET_Z_CLEAR_VALUE, fui(depth));
   if (clear_s)
      P_IMMD(p, NV9097, SET_STENCIL_CLEAR_VALUE, stencil & 0xff);

   uint32_t w = fb->width, h = fb->height;
   uint32_t xmin = 0, ymin = 0, xmax = w, ymax = h;
   if (scissor) {
      xmin = scissor->minx;
      ymin = scissor->miny;
      xmax = MIN2(scissor->maxx, w);
      ymax = MIN2(scissor->maxy, h);
   }

   /* PASCAL_A (GP100) clear bracket: NVK's own helper from our NVK patch
    * (mesa/pascal-a-clear.patch, nvk_pascal_a.h, used by
    * nvk_CmdClearAttachments), one definition for both drivers. */
   nvk_push_pascal_a_clear_begin(p, cls_eng3d);

   P_IMMD(p, NV9097, SET_CLEAR_SURFACE_CONTROL, {
      .respect_stencil_mask = RESPECT_STENCIL_MASK_FALSE,
      .use_clear_rect       = USE_CLEAR_RECT_TRUE,
      .use_scissor0         = USE_SCISSOR0_FALSE,
      .use_viewport_clip0   = USE_VIEWPORT_CLIP0_FALSE,
   });

   P_MTHD(p, NV9097, SET_COLOR_CLEAR_VALUE(0));
   P_NV9097_SET_COLOR_CLEAR_VALUE(p, 0, color->ui[0]);
   P_NV9097_SET_COLOR_CLEAR_VALUE(p, 1, color->ui[1]);
   P_NV9097_SET_COLOR_CLEAR_VALUE(p, 2, color->ui[2]);
   P_NV9097_SET_COLOR_CLEAR_VALUE(p, 3, color->ui[3]);

   P_MTHD(p, NV9097, SET_CLEAR_RECT_HORIZONTAL);
   P_NV9097_SET_CLEAR_RECT_HORIZONTAL(p, { .xmin = xmin, .xmax = xmax });
   P_NV9097_SET_CLEAR_RECT_VERTICAL(p, { .ymin = ymin, .ymax = ymax });

   /* One CLEAR_SURFACE per target and layer (NVK emit_clear_rects clears
    * each attachment with its own MRT select). */
   u_foreach_bit(i, color_mask) {
      const uint32_t layers = fb->cbufs[i].last_layer - fb->cbufs[i].first_layer + 1;
      for (uint32_t l = 0; l < layers; l++) {
         P_IMMD(p, NV9097, CLEAR_SURFACE, {
            .r_enable = true, .g_enable = true, .b_enable = true, .a_enable = true,
            .mrt_select = i,
            .rt_array_index = l,
         });
      }
   }
   for (uint32_t l = 0; l < zs_layers; l++) {
      P_IMMD(p, NV9097, CLEAR_SURFACE, {
         .z_enable = clear_z, .stencil_enable = clear_s,
         .mrt_select = 0,
         .rt_array_index = l,
      });
   }
   nvk_push_pascal_a_clear_end(p, cls_eng3d);
   nvrm_push_end(ctx, p);
   /* The target emission above marks the shadows written; only a colour
    * clear of them does (the next draw re-emits the targets: DIRTY_FB). */
   if (!writes_shadows)
      ctx->shadow_written = shadow_written && ctx->flush_seq == flush_seq;
   ctx->dirty |= NVRM_DIRTY_FB;
}

static void
nvrm_set_framebuffer_state(struct pipe_context *pctx,
                           const struct pipe_framebuffer_state *fb)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   /* The bound shadows go back to their linear buffers before the
    * framebuffer that names them is replaced (inline, no submission). */
   nvrm_fb_shadow_end(ctx);
   util_copy_framebuffer_state(&ctx->fb, fb);
   ctx->render_linear = nvrm_fb_renders_linear(fb);
   ctx->shadow_mask = 0;
   ctx->shadow_loaded = false;
   ctx->shadow_discarded = 0;
   for (unsigned i = 0; i < fb->nr_cbufs && !ctx->render_linear; i++) {
      if (fb->cbufs[i].texture && !nvrm_resource(fb->cbufs[i].texture)->tiled)
         ctx->shadow_mask |= 1u << i;
   }
   /* Blend enables depend on the target formats (integer targets). */
   ctx->dirty |= NVRM_DIRTY_FB | NVRM_DIRTY_DSA_BLEND;
}

/* ---------------------------------------------------------------- blits */

/* Blits (glGenerateMipmap, CopyTex[Sub]Image, glBlitFramebuffer) through
 * u_blitter: textured-quad draws on this driver's own 3D path, the way
 * every gallium driver without a 2D engine path does it. */
static void
nvrm_blitter_save(struct nvrm_context *ctx)
{
   struct blitter_context *b = ctx->blitter;
   util_blitter_save_vertex_buffers(b, ctx->vb, ctx->num_vb);
   util_blitter_save_vertex_elements(b, ctx->velems);
   util_blitter_save_vertex_shader(b, ctx->vs);
   util_blitter_save_geometry_shader(b, ctx->gs);
   util_blitter_save_rasterizer(b, ctx->rast);
   util_blitter_save_viewport(b, &ctx->viewport);
   util_blitter_save_scissor(b, &ctx->scissor);
   util_blitter_save_fragment_shader(b, ctx->fs);
   util_blitter_save_blend(b, ctx->blend);
   util_blitter_save_depth_stencil_alpha(b, ctx->dsa);
   util_blitter_save_stencil_ref(b, &ctx->stencil_ref);
   util_blitter_save_sample_mask(b, ctx->sample_mask, 0);
   util_blitter_save_framebuffer(b, &ctx->fb);
   util_blitter_save_so_targets(b, ctx->num_so, ctx->so, MESA_PRIM_UNKNOWN);
   util_blitter_save_render_condition(b, ctx->cond_query, ctx->cond_cond, ctx->cond_mode);
   util_blitter_save_fragment_sampler_states(b, NVRM_MAX_TEX_UNITS,
                                             (void **)ctx->samplers[MESA_SHADER_FRAGMENT]);
   util_blitter_save_fragment_sampler_views(b, NVRM_MAX_TEX_UNITS,
                                            ctx->views[MESA_SHADER_FRAGMENT]);
   util_blitter_save_fragment_constant_buffer_slot(b, ctx->cb[MESA_SHADER_FRAGMENT]);
}

static void
nvrm_blit(struct pipe_context *pctx, const struct pipe_blit_info *info)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   /* Stencil without shader stencil export: depth through the normal
    * path, stencil cleared then rebuilt bit by bit (u_blitter's
    * util_blitter_stencil_fallback, as d3d12 blit_replicate_stencil). */
   if ((info->mask & PIPE_MASK_S) && !info->num_window_rectangles &&
       !util_blitter_is_blit_supported(ctx->blitter, info)) {
      if (info->mask & PIPE_MASK_Z) {
         struct pipe_blit_info z = *info;
         z.mask = PIPE_MASK_Z;
         nvrm_blit(pctx, &z);
      }
      struct pipe_surface dst_templ;
      util_blitter_default_dst_texture(&dst_templ, info->dst.resource,
                                       info->dst.level, info->dst.box.z);
      nvrm_blitter_save(ctx);
      util_blitter_clear_depth_stencil(ctx->blitter, &dst_templ, PIPE_CLEAR_STENCIL,
                                       0, 0, info->dst.box.x, info->dst.box.y,
                                       info->dst.box.width, info->dst.box.height);
      nvrm_blitter_save(ctx);
      util_blitter_stencil_fallback(ctx->blitter, info->dst.resource, info->dst.level,
                                    &info->dst.box, info->src.resource, info->src.level,
                                    &info->src.box,
                                    info->scissor_enable ? &info->scissor : NULL);
      return;
   }
   /* u_blitter honours render_condition_enable (it suspends the saved
    * condition otherwise) and resolves multisampled sources with txf_ms. */
   if (info->window_rectangle_include || info->num_window_rectangles ||
       !util_blitter_is_blit_supported(ctx->blitter, info)) {
      nvrm_error("blit %s -> %s (mask 0x%x, filter %u) is not supported by the "
                 "u_blitter path (G1)\n",
                 util_format_short_name(info->src.format),
                 util_format_short_name(info->dst.format), info->mask, info->filter);
      abort();
   }
   nvrm_blitter_save(ctx);
   util_blitter_blit(ctx->blitter, info, NULL);
}

/* glInvalidateFramebuffer / glDiscardFramebufferEXT of a whole attachment
 * (fbobject.c do_discard_framebuffer): a bound shadowed colour buffer's
 * contents become undefined, so the pass needs no copy-in (NVK's
 * LOAD_OP_DONT_CARE) and what was rendered need not be copied out (NVK's
 * STORE_OP_DONT_CARE, nvk_cmd_draw.c:1500-1502). Only when every shadowed
 * buffer is discarded, since the copies are per framebuffer. Anything else
 * is a hint and keeps its contents. */
static void
nvrm_invalidate_resource(struct pipe_context *pctx, struct pipe_resource *res)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   if (!ctx->shadow_mask || res->target == PIPE_BUFFER)
      return;
   uint8_t discarded = 0;
   u_foreach_bit(i, ctx->shadow_mask) {
      if (ctx->fb.cbufs[i].texture == res)
         discarded |= 1u << i;
   }
   if (!discarded)
      return;
   ctx->shadow_discarded |= discarded;
   if (ctx->shadow_discarded != ctx->shadow_mask)
      return;
   ctx->shadow_discarded = 0;
   ctx->shadow_written = false;
   nvrm_fb_shadow_begin_overwritten(ctx);
   /* Later rendering re-emits the targets, which marks the shadows written. */
   ctx->dirty |= NVRM_DIRTY_FB;
}

static void nvrm_set_min_samples(struct pipe_context *p, unsigned m) {}
static void nvrm_set_clip_state(struct pipe_context *p, const struct pipe_clip_state *c) {}
static void nvrm_set_polygon_stipple(struct pipe_context *p, const struct pipe_poly_stipple *s) {}

/* No queries exist yet (occlusion queries are G1 work), so there is no
 * query state to suspend while u_blitter runs. */


/* ---------------------------------------------------------------- G2 fences */

/* EGL_ANDROID_native_fence_sync import: the fence owns a duplicate of the
 * sync_file (dri_create_fence_fd keeps ownership of its fd). */
static void
nvrm_create_fence_fd(struct pipe_context *pctx, struct pipe_fence_handle **fence,
                     int fd, enum pipe_fd_type type)
{
   *fence = NULL;
   if (type != PIPE_FD_TYPE_NATIVE_SYNC) {
      nvrm_error("create_fence_fd: fd type %u (only native sync_file)\n", type);
      return;
   }
   int dup = os_dupfd_cloexec(fd);
   if (dup < 0) {
      nvrm_error("create_fence_fd: dup(%d) failed\n", fd);
      return;
   }
   struct nvrm_fence *f = CALLOC_STRUCT(nvrm_fence);
   if (!f) {
      close(dup);
      return;
   }
   pipe_reference_init(&f->reference, 1);
   f->fd = dup;
   *fence = (struct pipe_fence_handle *)f;
}

/* eglWaitSyncKHR: a GPU-side wait before the next submission (stock: the
 * wait happens at the next flush). Own fences are on the
 * same in-order channel and need nothing. */
static void
nvrm_fence_server_sync(struct pipe_context *pctx, struct pipe_fence_handle *fence,
                       uint64_t value)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   struct nvrm_fence *f = (struct nvrm_fence *)fence;
   if (f->fd < 0)
      return;
   int dup = os_dupfd_cloexec(f->fd);
   if (dup < 0) {
      nvrm_error("fence_server_sync: dup failed; CPU-waiting the sync_file instead "
                 "is not done silently: aborting\n");
      abort();
   }
   /* No submission here: stock stores the imported fd and waits for it at
    * the next flush, without a kick (observed for
    * EGL_SYNC_NATIVE_FENCE_ANDROID import + eglWaitSync). The whole next batch is gated, including commands
    * recorded before this call: over-waiting only, and never a deadlock,
    * since a sync_file can depend only on work already submitted. Android
    * makes this call for every dequeued window buffer
    * (dri2_allocate_textures -> dri_image_fence_sync), so a submission here
    * would be a kick per frame. */
   util_dynarray_append_typed(&ctx->wait_fds, int, dup);
}

/* Before a resource is shared (EGLImage export, dri_helpers.c): nvrm has
 * no compression or fast-clear metadata to resolve (every nvrm image is
 * uncompressed, nil usage UNCOMPRESSED_BIT), and the frontend's flush that follows
 * submits pending writes. */
static void
nvrm_flush_resource(struct pipe_context *pctx, struct pipe_resource *res)
{
}

void
nvrm_init_context_fence_functions(struct pipe_context *pctx)
{
   pctx->flush_resource = nvrm_flush_resource;
   pctx->create_fence_fd = nvrm_create_fence_fd;
   pctx->fence_server_sync = nvrm_fence_server_sync;
}

static void
nvrm_context_destroy(struct pipe_context *pctx)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   if (ctx->blitter)
      util_blitter_destroy(ctx->blitter);
   nvrm_flush(pctx, NULL, 0);
   nvrm_draw_context_destroy(ctx);
   nvrm_compute_context_destroy(ctx);
   nvrm_query_context_destroy(ctx);
   nvrm_texture_context_destroy(ctx);
   if (pctx->stream_uploader)
      u_upload_destroy(pctx->stream_uploader);
   nvrm_flush(pctx, NULL, 0);
   if (ctx->keep)
      _mesa_set_destroy(ctx->keep, nvrm_unref_kept);
   struct nvrm_screen *screen = nvrm_screen(pctx->screen);
   if (screen->debug_stats)
      fprintf(stderr, "nvrm stats at context destruction: %" PRIu64 " submissions, %" PRIu64
              " CPU waits (screen totals)\n", screen->stat_submits, screen->stat_cpu_waits);
   util_dynarray_fini(&ctx->execs);
   util_dynarray_fini(&ctx->shader_frees);
   util_dynarray_foreach(&ctx->wait_fds, int, fd)
      close(*fd);
   util_dynarray_fini(&ctx->wait_fds);
   util_unreference_framebuffer_state(&ctx->fb);
   nvrm_push_chunks_free(ctx);
   FREE(ctx);
}

struct pipe_context *
nvrm_context_create(struct pipe_screen *pscreen, void *priv, unsigned flags)
{
   struct nvrm_screen *screen = nvrm_screen(pscreen);
   struct nvrm_context *ctx = CALLOC_STRUCT(nvrm_context);
   if (!ctx)
      return NULL;
   ctx->base.screen = pscreen;
   util_dynarray_init(&ctx->push_batch_chunks, NULL);
   util_dynarray_init(&ctx->push_free_chunks, NULL);
   struct nvrm_push_chunk *first = CALLOC_STRUCT(nvrm_push_chunk);
   if (!first || !nvrm_bo_alloc(screen, NVRM_PUSH_SIZE_B, NVRM_DOMAIN_GART, &first->bo)) {
      FREE(first);
      FREE(ctx);
      return NULL;
   }
   ctx->push_chunk_count = 1;
   nvrm_push_use_chunk(ctx, first);
   util_dynarray_init(&ctx->execs, NULL);
   ctx->keep = _mesa_pointer_set_create(NULL);
   if (!ctx->keep) {
      nvrm_push_chunks_free(ctx);
      FREE(ctx);
      return NULL;
   }

   struct pipe_context *p = &ctx->base;
   p->screen = pscreen;
   p->priv = priv;
   p->destroy = nvrm_context_destroy;
   p->flush = nvrm_flush;
   p->clear = nvrm_clear;
   /* Mesa 26: the frontend hands released resources / sampler views back to
    * the driver; nvrm keeps no deferred-release state, so these drop the
    * reference as nvc0 does (nvc0_state.c:1428-1429). */
   p->resource_release = u_default_resource_release;
   p->sampler_view_release = u_default_sampler_view_release;
   p->set_framebuffer_state = nvrm_set_framebuffer_state;
   p->blit = nvrm_blit;
   p->invalidate_resource = nvrm_invalidate_resource;
   nvrm_init_copy_functions(p); /* resource_copy_region: copy engine (nvrm_copy.c) */


   p->set_min_samples = nvrm_set_min_samples;
   p->set_clip_state = nvrm_set_clip_state;
   p->set_polygon_stipple = nvrm_set_polygon_stipple;

   nvrm_init_transfer_functions(p);
   nvrm_init_draw_functions(p);
   nvrm_init_texture_functions(p);
   nvrm_init_query_functions(p);
   nvrm_init_compute_functions(p);
   ctx->dirty = NVRM_DIRTY_ALL;
   util_dynarray_init(&ctx->shader_frees, NULL);
   util_dynarray_init(&ctx->wait_fds, NULL);
   nvrm_init_context_fence_functions(p);
   p->stream_uploader = u_upload_create_default(p);
   if (!p->stream_uploader) {
      nvrm_context_destroy(p);
      return NULL;
   }
   p->const_uploader = p->stream_uploader;
   if (!nvrm_texture_context_init(ctx)) {
      nvrm_context_destroy(p);
      return NULL;
   }
   ctx->blitter = util_blitter_create(p);
   if (!ctx->blitter) {
      nvrm_context_destroy(p);
      return NULL;
   }
   return p;
}
