/* Real C adapter tests. The RM mock consumes exact call expectations. */
static void
fusion_prepare(struct nvkmd_rm_ctx *ctx, uint32_t count)
{
   fusion = (struct fusion_state){ .active = true };
   util_dynarray_clear(&ctx->pending);
   util_dynarray_clear(&ctx->waits);
   ctx->last_token = 0;
   struct NvRmSyncPoint wait = { 77, 8 };
   util_dynarray_append_typed(&ctx->waits, struct NvRmSyncPoint, wait);
   for (uint32_t i = 0; i < count; i++) {
      struct nvkmd_ctx_exec exec = { .addr = 0x1000 + 4 * i, .size_B = 4 };
      assert(rm_ctx_exec(&ctx->base, NULL, 1, &exec) == VK_SUCCESS);
   }
}
static void
fusion_expect(uint32_t pushes, uint32_t waits, uint32_t signals,
              uint64_t address, int32_t result)
{
   assert(fusion.count < ARRAY_SIZE(fusion.calls));
   fusion.calls[fusion.count++] = (struct fusion_call) {
      pushes, waits, signals, address, result,
   };
}
static void
test_submit_fusion(struct vk_device *vk, const struct vk_sync_type *type)
{
   sync_api = (struct sync_api_state){ .active = true };
   struct nvkmd_rm_ctx ctx = {
      .base = { .ops = &nvkmd_rm_ctx_ops, .dev = &rm_sync_device(vk)->base },
      .vk = vk, .channel = 33,
   };
   util_dynarray_init(&ctx.pending, NULL);
   util_dynarray_init(&ctx.waits, NULL);
   struct nvkmd_rm_sync a = { .base = { .type = type, .flags = VK_SYNC_IS_TIMELINE }, .id = 77 };
   struct nvkmd_rm_sync b = { .base = { .type = type }, .id = 78 };
   struct vk_sync_signal signals[] = {
      { .sync = &a.base, .signal_value = 9 }, { .sync = &b.base },
   };

   /* Empty, ordinary, exactly-full, and split submissions: only the final
    * accepted chunk carries signals, only the first carries waits. */
   const unsigned sizes[] = { 0, 1, 1000, 1001, 2000 };
   for (unsigned c = 0; c < ARRAY_SIZE(sizes); c++) {
      unsigned total = sizes[c];
      fusion_prepare(&ctx, total);
      if (total <= 1000) {
         fusion_expect(total, 1, 2, 0x1000, VK_SUCCESS);
      } else {
         fusion_expect(1000, 1, 0, 0x1000, VK_SUCCESS);
         fusion_expect(total - 1000, 0, 2, 0x1fa0, VK_SUCCESS);
      }
      assert(rm_ctx_signal(&ctx.base, NULL, 2, signals) == VK_SUCCESS);
      assert(fusion.next == fusion.count && !fusion.ring_waits);
      assert(!ctx.pending.size && !ctx.waits.size);
      assert(ctx.last_token == 99 + fusion.count);
   }
   /* A continuation crossing entry 1000 moves the split back to 999. */
   fusion_prepare(&ctx, 1001);
   ((struct NvRmExec *)ctx.pending.data)[999].flags = NV_RM_EXEC_INCOMPLETE;
   fusion_expect(999, 1, 0, 0x1000, VK_SUCCESS);
   fusion_expect(2, 0, 2, 0x1f9c, VK_SUCCESS);
   assert(rm_ctx_signal(&ctx.base, NULL, 2, signals) == VK_SUCCESS);
   assert(fusion.next == 2 && !ctx.pending.size);

   fusion_prepare(&ctx, 0);
   util_dynarray_clear(&ctx.waits);
   fusion_expect(0, 0, 2, 0, VK_SUCCESS);
   assert(rm_ctx_signal(&ctx.base, NULL, 2, signals) == VK_SUCCESS);
   assert(fusion.next == 1); /* Signal-only still needs exactly one GPU batch. */
   fusion_prepare(&ctx, 0);
   fusion_expect(0, 1, 0, 0, VK_SUCCESS);
   assert(rm_ctx_flush(&ctx.base, NULL) == VK_SUCCESS);
   assert(fusion.next == 1 && !ctx.waits.size);
   fusion_prepare(&ctx, 1);
   fusion_expect(1, 1, 2, 0x1000, VK_NOT_READY);
   fusion_expect(1, 1, 2, 0x1000, VK_SUCCESS);
   assert(rm_ctx_signal(&ctx.base, NULL, 2, signals) == VK_SUCCESS);
   assert(fusion.next == 2 && fusion.ring_waits == 1);

   /* Ring pressure accepts nothing. Both first-chunk waits and final-chunk
    * signals must survive their own retry, without an extra signal batch. */
   fusion_prepare(&ctx, 1001);
   fusion_expect(1000, 1, 0, 0x1000, VK_NOT_READY);
   fusion_expect(1000, 1, 0, 0x1000, VK_SUCCESS);
   fusion_expect(1, 0, 2, 0x1fa0, VK_NOT_READY);
   fusion_expect(1, 0, 2, 0x1fa0, VK_SUCCESS);
   assert(rm_ctx_signal(&ctx.base, NULL, 2, signals) == VK_SUCCESS);
   assert(fusion.next == 4 && fusion.ring_waits == 2 && ctx.last_token == 103);

   /* Failed ring waits and rejected batches retain all unaccepted work. */
   for (unsigned ring = 0; ring < 2; ring++) {
      fusion_prepare(&ctx, 1);
      fusion.wait_result = VK_ERROR_DEVICE_LOST;
      fusion_expect(1, 1, 2, 0x1000, ring ? VK_NOT_READY : VK_ERROR_DEVICE_LOST);
      assert(rm_ctx_signal(&ctx.base, NULL, 2, signals) == VK_ERROR_DEVICE_LOST);
      assert(fusion.next == 1 && fusion.ring_waits == ring);
      assert(ctx.pending.size == sizeof(struct NvRmExec));
      assert(ctx.waits.size == sizeof(struct NvRmSyncPoint) && !ctx.last_token);
   }
   /* A later failure must never replay the accepted prefix or its waits. */
   fusion_prepare(&ctx, 1001);
   fusion_expect(1000, 1, 0, 0x1000, VK_SUCCESS);
   fusion_expect(1, 0, 2, 0x1fa0, VK_ERROR_DEVICE_LOST);
   fusion_expect(1, 0, 2, 0x1fa0, VK_SUCCESS);
   assert(rm_ctx_signal(&ctx.base, NULL, 2, signals) == VK_ERROR_DEVICE_LOST);
   assert(ctx.pending.size == sizeof(struct NvRmExec) && !ctx.waits.size);
   assert(ctx.last_token == 100);
   assert(rm_ctx_signal(&ctx.base, NULL, 2, signals) == VK_SUCCESS);
   assert(fusion.next == 3 && !ctx.pending.size && ctx.last_token == 102);

   /* Invalid final continuation chain is rejected before even a valid first
    * chunk. Likewise validate every signal before submitting queued pushes. */
   fusion_prepare(&ctx, 2001);
   for (unsigned i = 1000; i < 2000; i++)
      ((struct NvRmExec *)ctx.pending.data)[i].flags = NV_RM_EXEC_INCOMPLETE;
   assert(rm_ctx_signal(&ctx.base, NULL, 2, signals) != VK_SUCCESS);
   assert(!fusion.next && ctx.pending.size == 2001 * sizeof(struct NvRmExec));
   assert(ctx.waits.size == sizeof(struct NvRmSyncPoint));
   fusion_prepare(&ctx, 1);
   struct vk_sync_type foreign_type = { 0 };
   b.base.type = &foreign_type;
   assert(rm_ctx_signal(&ctx.base, NULL, 2, signals) != VK_SUCCESS);
   assert(!fusion.next && ctx.pending.size == sizeof(struct NvRmExec));
   b.base.type = type;
   fake.calloc_fail = fake.calloc_calls + 1;
   assert(rm_ctx_signal(&ctx.base, NULL, 2, signals) == VK_ERROR_OUT_OF_HOST_MEMORY);
   fake.calloc_fail = 0;
   assert(!fusion.next && ctx.waits.size == sizeof(struct NvRmSyncPoint));

   /* A zero-signal call still flushes work, but empty flush has no batch. */
   fusion_prepare(&ctx, 1);
   fusion_expect(1, 1, 0, 0x1000, VK_SUCCESS);
   assert(rm_ctx_signal(&ctx.base, NULL, 0, NULL) == VK_SUCCESS);
   assert(rm_ctx_signal(&ctx.base, NULL, 0, NULL) == VK_SUCCESS);
   assert(fusion.next == 1);
   /* Bind-only CPU contexts must not acquire a GPU channel just to signal. */
   ctx.channel = 0;
   assert(rm_ctx_signal(&ctx.base, NULL, 1, signals) == VK_SUCCESS);
   assert(fusion.next == 1 && sync_api.signals == 1);
   util_dynarray_fini(&ctx.pending);
   util_dynarray_fini(&ctx.waits);
   fusion.active = false;
   sync_api.active = false;
   puts("PASS: fused submit/wait/signal, 1000-entry boundaries, ring retries, transactional failure and CPU bind signal");
}
