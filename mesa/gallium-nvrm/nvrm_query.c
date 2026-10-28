/* SPDX-License-Identifier: MIT
 *
 * nvrm G1 (GLES 3.0): queries, conditional rendering and transform feedback.
 *
 * Hardware programming follows NVK (nvk_query_pool.c nvk_cmd_begin_end_query,
 * nvk_cmd_draw.c nvk_CmdBegin/EndTransformFeedbackEXT); GL semantics follow
 * nvc0 (nvc0_query_hw.c: begin report at +16, end report at +0, conditional
 * render compares the two). Stock GL (observed push stream) uses the same
 * report types and locations:
 * PRIMITIVES_GENERATED = VTG_PRIMITIVES_OUT at STREAMING_OUTPUT begin+end,
 * xfb begin programs STREAM_OUT buffers, pause = STREAM_OUTPUT 0 + WFI,
 * conditional render through SET_RENDER_ENABLE_C (RENDER_IF_NOT_EQUAL when
 * waiting, TRUE for NO_WAIT). Stock's occlusion query is end-only with a
 * counter reset we have not traced; nvrm uses NVK's
 * begin+end ZPASS_PIXEL_CNT64 difference instead.
 *
 * A result is available once the batch that ended the query completed
 * (its token, nvrm_seq_token).
 */
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "nvrm_priv.h"

#include "nv_push.h"
#include "nv_push_cl9097.h"
#include "nv_push_cla097.h"

#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_blitter.h"

struct nvrm_query {
   unsigned type, index;
   struct nvrm_bo bo;   /* [0,16) end report, [16,32) begin report */
   uint64_t flush_seq;  /* ctx->flush_seq when the end report was pushed */
   bool active;
};

struct nvrm_so_target {
   struct pipe_stream_output_target base;
   struct nvrm_bo counter; /* STREAMING_BYTE_COUNT report (one word) */
   uint64_t flush_seq;     /* flush that holds the last counter report */
   bool counter_written;
};

static inline struct nvrm_so_target *
nvrm_so_target(struct pipe_stream_output_target *t)
{
   return (struct nvrm_so_target *)t;
}

static void
nvrm_report(struct nv_push *p, uint64_t addr, uint32_t location, uint32_t report,
            uint32_t sub_report, bool one_word)
{
   P_MTHD(p, NV9097, SET_REPORT_SEMAPHORE_A);
   P_NV9097_SET_REPORT_SEMAPHORE_A(p, addr >> 32);
   P_NV9097_SET_REPORT_SEMAPHORE_B(p, addr);
   P_NV9097_SET_REPORT_SEMAPHORE_C(p, 0);
   P_NV9097_SET_REPORT_SEMAPHORE_D(p, {
      .operation = OPERATION_REPORT_ONLY,
      .pipeline_location = location,
      .report = report,
      .sub_report = sub_report,
      .structure_size = one_word ? STRUCTURE_SIZE_ONE_WORD : STRUCTURE_SIZE_FOUR_WORDS,
      .flush_disable = true,
   });
}

/* ---------------------------------------------------------------- queries */

static struct pipe_query *
nvrm_create_query(struct pipe_context *pctx, unsigned type, unsigned index)
{
   switch (type) {
   case PIPE_QUERY_OCCLUSION_COUNTER:
   case PIPE_QUERY_OCCLUSION_PREDICATE:
   case PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE:
   case PIPE_QUERY_PRIMITIVES_GENERATED:
   case PIPE_QUERY_PRIMITIVES_EMITTED:
      break;
   default:
      nvrm_error("query type %u is not implemented (G1 implements occlusion, "
                 "primitives generated/emitted)\n", type);
      return NULL;
   }
   struct nvrm_query *q = CALLOC_STRUCT(nvrm_query);
   if (!q)
      return NULL;
   q->type = type;
   q->index = index;
   if (!nvrm_bo_alloc(nvrm_screen(pctx->screen), 64, NVRM_DOMAIN_GART, &q->bo)) {
      nvrm_error("query buffer could not be allocated\n");
      FREE(q);
      return NULL;
   }
   memset(q->bo.map, 0, 64);
   return (struct pipe_query *)q;
}

static void
nvrm_destroy_query(struct pipe_context *pctx, struct pipe_query *pq)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   struct nvrm_query *q = (struct nvrm_query *)pq;
   /* A pending push (report or render enable) may still address it. */
   if (nvrm_has_pending(ctx))
      pctx->flush(pctx, NULL, 0);
   if (ctx->cond_query == pq)
      ctx->cond_query = NULL;
   nvrm_bo_free(nvrm_screen(pctx->screen), &q->bo);
   FREE(q);
}

/* Begin (end = false) or end report of one query, as
 * nvk_cmd_begin_end_query's per-type blocks. */
static void
nvrm_query_report(struct nvrm_context *ctx, struct nvrm_query *q, bool end)
{
   const uint64_t addr = q->bo.va + (end ? 0 : 16);
   struct nv_push push = nvrm_push_begin(ctx, 16);
   struct nv_push *p = &push;
   switch (q->type) {
   case PIPE_QUERY_OCCLUSION_COUNTER:
   case PIPE_QUERY_OCCLUSION_PREDICATE:
   case PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE:
      P_IMMD(p, NV9097, SET_ZPASS_PIXEL_COUNT, !end && ctx->queries_enabled);
      nvrm_report(p, addr, NV9097_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_ALL,
                  NV9097_SET_REPORT_SEMAPHORE_D_REPORT_ZPASS_PIXEL_CNT64, 0, false);
      break;
   case PIPE_QUERY_PRIMITIVES_GENERATED:
      nvrm_report(p, addr, NV9097_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_STREAMING_OUTPUT,
                  NV9097_SET_REPORT_SEMAPHORE_D_REPORT_VTG_PRIMITIVES_OUT, 0, false);
      break;
   case PIPE_QUERY_PRIMITIVES_EMITTED:
      nvrm_report(p, addr, NV9097_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_STREAMING_OUTPUT,
                  NV9097_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_PRIMITIVES_SUCCEEDED,
                  q->index, false);
      break;
   }
   nvrm_push_end(ctx, p);
}

static bool
nvrm_is_occlusion(unsigned type)
{
   return type == PIPE_QUERY_OCCLUSION_COUNTER || type == PIPE_QUERY_OCCLUSION_PREDICATE ||
          type == PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE;
}

static bool
nvrm_begin_query(struct pipe_context *pctx, struct pipe_query *pq)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   struct nvrm_query *q = (struct nvrm_query *)pq;
   if (nvrm_is_occlusion(q->type)) {
      if (ctx->occlusion_query) {
         nvrm_error("two occlusion queries active at once\n");
         abort();
      }
      ctx->occlusion_query = q;
   }
   q->active = true;
   nvrm_query_report(ctx, q, false);
   return true;
}

static bool
nvrm_end_query(struct pipe_context *pctx, struct pipe_query *pq)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   struct nvrm_query *q = (struct nvrm_query *)pq;
   nvrm_query_report(ctx, q, true);
   if (ctx->occlusion_query == q)
      ctx->occlusion_query = NULL;
   q->active = false;
   q->flush_seq = ctx->flush_seq;
   return true;
}

static bool
nvrm_get_query_result(struct pipe_context *pctx, struct pipe_query *pq, bool wait,
                      union pipe_query_result *result)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   struct nvrm_query *q = (struct nvrm_query *)pq;
   if (q->flush_seq == ctx->flush_seq) {
      /* The end report is still in an unsubmitted push. */
      pctx->flush(pctx, NULL, 0);
   }
   /* Final once the batch that wrote the end report completed. */
   struct nvrm_screen *screen = nvrm_screen(pctx->screen);
   const uint64_t token = nvrm_seq_token(ctx, q->flush_seq);
   if (!wait && !nvrm_token_done(screen, token))
      return false;
   if (!nvrm_wait_token(screen, token)) {
      nvrm_error("query result wait failed\n");
      return false;
   }
   const uint64_t *r = q->bo.map;
   const uint64_t diff = r[0] - r[2];
   switch (q->type) {
   case PIPE_QUERY_OCCLUSION_PREDICATE:
   case PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE:
      result->b = diff != 0;
      break;
   default:
      result->u64 = diff;
      break;
   }
   return true;
}

static void
nvrm_set_active_query_state(struct pipe_context *pctx, bool enable)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   ctx->queries_enabled = enable;
   if (!ctx->occlusion_query)
      return;
   /* u_blitter pauses counting around its own draws (nvc0 does the same
    * with SAMPLECOUNT_ENABLE). */
   struct nv_push push = nvrm_push_begin(ctx, 4);
   P_IMMD(&push, NV9097, SET_ZPASS_PIXEL_COUNT, enable);
   nvrm_push_end(ctx, &push);
}

/* ---------------------------------------------------- conditional render */

static void
nvrm_render_condition(struct pipe_context *pctx, struct pipe_query *pq,
                      bool condition, enum pipe_render_cond_flag mode)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   ctx->cond_query = pq;
   ctx->cond_cond = condition;
   ctx->cond_mode = mode;
   ctx->dirty |= NVRM_DIRTY_COND;
}

/* SET_RENDER_ENABLE as stock GL: compare the end and begin reports
 * (RENDER_IF_NOT_EQUAL = samples passed) after waiting for the query;
 * NO_WAIT renders unconditionally, as stock does for it. */
void
nvrm_emit_render_enable(struct nvrm_context *ctx, struct nv_push *p)
{
   struct nvrm_query *q = (struct nvrm_query *)ctx->cond_query;
   const bool wait = ctx->cond_mode == PIPE_RENDER_COND_WAIT ||
                     ctx->cond_mode == PIPE_RENDER_COND_BY_REGION_WAIT;
   if (!q || !wait) {
      P_IMMD(p, NV9097, SET_RENDER_ENABLE_C, MODE_TRUE);
      return;
   }
   if (!nvrm_is_occlusion(q->type) && q->type != PIPE_QUERY_PRIMITIVES_GENERATED &&
       q->type != PIPE_QUERY_PRIMITIVES_EMITTED) {
      nvrm_error("conditional render on query type %u\n", q->type);
      abort();
   }
   /* The reports land when the preceding work drains (nvc0 acquires the
    * query sequence; a 3D WFI orders the same). */
   P_IMMD(p, NV9097, WAIT_FOR_IDLE, 0);
   P_MTHD(p, NV9097, SET_RENDER_ENABLE_A);
   P_NV9097_SET_RENDER_ENABLE_A(p, q->bo.va >> 32);
   P_NV9097_SET_RENDER_ENABLE_B(p, q->bo.va);
   /* gallium: skip rendering when the result equals `condition`. */
   P_NV9097_SET_RENDER_ENABLE_C(p, ctx->cond_cond ? NV9097_SET_RENDER_ENABLE_C_MODE_RENDER_IF_EQUAL
                                                  : NV9097_SET_RENDER_ENABLE_C_MODE_RENDER_IF_NOT_EQUAL);
}

/* ------------------------------------------------- transform feedback */

static struct pipe_stream_output_target *
nvrm_create_so_target(struct pipe_context *pctx, struct pipe_resource *res,
                      unsigned buffer_offset, unsigned buffer_size)
{
   struct nvrm_so_target *t = CALLOC_STRUCT(nvrm_so_target);
   if (!t)
      return NULL;
   if (!nvrm_bo_alloc(nvrm_screen(pctx->screen), 64, NVRM_DOMAIN_GART, &t->counter)) {
      nvrm_error("transform feedback counter could not be allocated\n");
      FREE(t);
      return NULL;
   }
   memset(t->counter.map, 0, 64);
   pipe_reference_init(&t->base.reference, 1);
   pipe_resource_reference(&t->base.buffer, res);
   t->base.context = pctx;
   t->base.buffer_offset = buffer_offset;
   t->base.buffer_size = buffer_size;
   return &t->base;
}

static void
nvrm_so_target_destroy(struct pipe_context *pctx, struct pipe_stream_output_target *pt)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   struct nvrm_so_target *t = nvrm_so_target(pt);
   if (nvrm_has_pending(ctx))
      pctx->flush(pctx, NULL, 0);
   pipe_resource_reference(&t->base.buffer, NULL);
   nvrm_bo_free(nvrm_screen(pctx->screen), &t->counter);
   FREE(t);
}

/* Pause: stock's SET_STREAM_OUTPUT 0 + WAIT_FOR_IDLE, plus NVK's end
 * (STREAMING_BYTE_COUNT per buffer into the target's counter) so a resume
 * can reload the write pointer. */
void
nvrm_so_pause(struct nvrm_context *ctx)
{
   if (!ctx->so_active)
      return;
   ctx->so_active = false; /* before push_begin: it may flush */
   struct nv_push push = nvrm_push_begin(ctx, 8 + 6 * NVRM_MAX_SO);
   struct nv_push *p = &push;
   P_IMMD(p, NV9097, SET_STREAM_OUTPUT, ENABLE_FALSE);
   for (unsigned i = 0; i < ctx->num_so; i++) {
      struct nvrm_so_target *t = nvrm_so_target(ctx->so[i]);
      if (!t)
         continue;
      nvrm_report(p, t->counter.va, NV9097_SET_REPORT_SEMAPHORE_D_PIPELINE_LOCATION_STREAMING_OUTPUT,
                  NV9097_SET_REPORT_SEMAPHORE_D_REPORT_STREAMING_BYTE_COUNT, i, true);
      t->counter_written = true;
      t->flush_seq = ctx->flush_seq;
      ctx->so_load[i] = UINT32_MAX; /* resume appends */
   }
   P_IMMD(p, NV9097, WAIT_FOR_IDLE, 0);
   /* Captured data is read next as vertices/uniforms/by the CPU. */
   P_IMMD(p, NVA097, INVALIDATE_SHADER_CACHES_NO_WFI, {
      .global_data = true, .constant = true,
   });
   nvrm_push_end(ctx, p);
}

static void
nvrm_set_stream_output_targets(struct pipe_context *pctx, unsigned num,
                               struct pipe_stream_output_target **targets,
                               const unsigned *offsets, enum mesa_prim output_prim)
{
   struct nvrm_context *ctx = nvrm_context(pctx);
   if (num > NVRM_MAX_SO) {
      nvrm_error("%u stream output targets (max %u)\n", num, NVRM_MAX_SO);
      abort();
   }
   nvrm_so_pause(ctx);
   for (unsigned i = 0; i < NVRM_MAX_SO; i++) {
      struct pipe_stream_output_target *t = i < num ? targets[i] : NULL;
      pipe_so_target_reference(&ctx->so[i], t);
      ctx->so_load[i] = (t && offsets) ? offsets[i] : 0;
   }
   ctx->num_so = num;
   ctx->dirty |= NVRM_DIRTY_SO;
}

static void nvrm_so_counter_wait(struct nvrm_context *ctx, const struct nvrm_so_target *t);

/* Resolve append offsets (UINT32_MAX) from the counters; may flush, so it
 * runs before a draw opens its push. */
void
nvrm_so_prepare(struct nvrm_context *ctx)
{
   for (unsigned i = 0; i < ctx->num_so; i++) {
      struct nvrm_so_target *t = ctx->so[i] ? nvrm_so_target(ctx->so[i]) : NULL;
      if (!t || ctx->so_load[i] != UINT32_MAX)
         continue;
      if (t->counter_written && t->flush_seq == ctx->flush_seq)
         ctx->base.flush(&ctx->base, NULL, 0);
      if (t->counter_written)
         nvrm_so_counter_wait(ctx, t);
      ctx->so_load[i] = t->counter_written ? *(volatile uint32_t *)t->counter.map : 0;
   }
}

/* Stream output counters are read by the CPU on resume: wait for the batch
 * whose pause wrote them (the counter flush_seq). */
static void
nvrm_so_counter_wait(struct nvrm_context *ctx, const struct nvrm_so_target *t)
{
   if (!nvrm_wait_token(nvrm_screen(ctx->base.screen), nvrm_seq_token(ctx, t->flush_seq))) {
      nvrm_error("stream output counter wait failed\n");
      abort();
   }
}

/* Begin as stock GL / nvk_CmdBindTransformFeedbackBuffersEXT +
 * nvk_CmdBeginTransformFeedbackEXT: buffers, write pointers, enable. */
void
nvrm_emit_so(struct nvrm_context *ctx, struct nv_push *p)
{
   if (!ctx->num_so) {
      P_IMMD(p, NV9097, SET_STREAM_OUTPUT, ENABLE_FALSE);
      return;
   }
   for (unsigned i = 0; i < NVRM_MAX_SO; i++) {
      struct pipe_stream_output_target *t = i < ctx->num_so ? ctx->so[i] : NULL;
      if (!t) {
         P_IMMD(p, NV9097, SET_STREAM_OUT_BUFFER_ENABLE(i), V_FALSE);
         continue;
      }
      if (ctx->so_load[i] == UINT32_MAX) {
         /* Paused by a flush inside this draw's push_begin: wait for
          * that batch, then the counter is final. */
         struct nvrm_so_target *st = nvrm_so_target(t);
         if (st->flush_seq == ctx->flush_seq) {
            nvrm_error("stream output resume before its counter landed\n");
            abort();
         }
         nvrm_so_counter_wait(ctx, st);
         ctx->so_load[i] = *(volatile uint32_t *)st->counter.map;
      }
      nvrm_keep(ctx, t->buffer);
      const uint64_t addr = nvrm_resource(t->buffer)->bo.va + t->buffer_offset;
      P_MTHD(p, NV9097, SET_STREAM_OUT_BUFFER_ENABLE(i));
      P_NV9097_SET_STREAM_OUT_BUFFER_ENABLE(p, i, V_TRUE);
      P_NV9097_SET_STREAM_OUT_BUFFER_ADDRESS_A(p, i, addr >> 32);
      P_NV9097_SET_STREAM_OUT_BUFFER_ADDRESS_B(p, i, addr);
      P_NV9097_SET_STREAM_OUT_BUFFER_SIZE(p, i, t->buffer_size);
      P_NV9097_SET_STREAM_OUT_BUFFER_LOAD_WRITE_POINTER(p, i, ctx->so_load[i]);
   }
   P_IMMD(p, NV9097, SET_STREAM_OUTPUT, ENABLE_TRUE);
   ctx->so_active = true;
}

void
nvrm_init_query_functions(struct pipe_context *p)
{
   struct nvrm_context *ctx = nvrm_context(p);
   ctx->queries_enabled = true;
   p->create_query = nvrm_create_query;
   p->destroy_query = nvrm_destroy_query;
   p->begin_query = nvrm_begin_query;
   p->end_query = nvrm_end_query;
   p->get_query_result = nvrm_get_query_result;
   p->set_active_query_state = nvrm_set_active_query_state;
   p->render_condition = nvrm_render_condition;
   p->create_stream_output_target = nvrm_create_so_target;
   p->stream_output_target_destroy = nvrm_so_target_destroy;
   p->set_stream_output_targets = nvrm_set_stream_output_targets;
}

void
nvrm_query_context_destroy(struct nvrm_context *ctx)
{
   for (unsigned i = 0; i < NVRM_MAX_SO; i++)
      pipe_so_target_reference(&ctx->so[i], NULL);
}
