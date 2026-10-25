/* Exercise the production C adapter, not a second implementation of it.
 * Rust payload ownership is covered by Rust tests; these mocks assert the
 * exact bridge arguments and fail if ordinary work requests an export. */
static struct sync_api_state {
   bool active;
   unsigned signals, resets, moves, reserves, imports, exports, submits, batches, waits, destroys;
   uint32_t shared, opaque;
   uint64_t value;
} sync_api;
static struct fusion_state {
   bool active;
   struct fusion_call {
      uint32_t pushes, waits, signals;
      uint64_t address;
      int32_t result;
   } calls[16];
   unsigned count, next, ring_waits;
   int32_t wait_result;
} fusion;

static void check_sync_api(struct NvRmDevice *dev)
{
   assert(sync_api.active && dev == (struct NvRmDevice *)(uintptr_t)42);
}
int32_t nv_rm_sync_reset(struct NvRmDevice *dev, uint64_t id)
{ check_sync_api(dev); assert(id == 77); sync_api.resets++; return VK_SUCCESS; }
int32_t nv_rm_sync_move(struct NvRmDevice *dev, uint64_t dst, uint64_t src, uint32_t shared)
{
   check_sync_api(dev); assert(dst == 77 && src == 78);
   sync_api.moves++; sync_api.shared = shared; return VK_SUCCESS;
}
int32_t nv_rm_sync_reserve(struct NvRmDevice *dev, uint64_t dst, uint64_t src, uint32_t *reserved)
{
   check_sync_api(dev); assert(dst == 77 && src == 78);
   sync_api.reserves++; *reserved = 1; return VK_SUCCESS;
}
int32_t nv_rm_sync_signal(struct NvRmDevice *dev, uint64_t id, uint64_t value)
{
   check_sync_api(dev); assert(id == 77);
   sync_api.signals++; sync_api.value = value; return VK_SUCCESS;
}
int32_t nv_rm_sync_get_value(struct NvRmDevice *dev, uint64_t id, uint64_t *value)
{ check_sync_api(dev); assert(id == 77); *value = 8; return VK_SUCCESS; }
int32_t nv_rm_sync_import_fd(struct NvRmDevice *dev, uint64_t id, int fd, uint32_t opaque)
{
   check_sync_api(dev); assert(id == 77 && fd == 66);
   sync_api.imports++; sync_api.opaque = opaque; return VK_SUCCESS;
}
int32_t nv_rm_sync_export_fd(struct NvRmDevice *dev, uint64_t id, uint32_t opaque,
                            int drm_fd, int *fd)
{
   check_sync_api(dev); assert(id == 77 && drm_fd == 99);
   sync_api.exports++; sync_api.opaque = opaque; *fd = 88; return VK_SUCCESS;
}
int32_t nv_rm_submit_sync(struct NvRmDevice *dev, uint64_t context,
                         const struct NvRmExec *pushes, uint32_t count,
                         const struct NvRmSyncPoint *waits, uint32_t nwaits,
                         uint64_t *token)
{
   check_sync_api(dev); assert(context == 33 && count == 1 && nwaits == 1);
   assert(pushes[0].address == 0x1000 && pushes[0].size_B == 4);
   assert(waits[0].sync == 77 && waits[0].value == 8);
   *token = ++sync_api.submits == 1 ? 9 : 10;
   return sync_api.submits == 1 ? VK_NOT_READY : VK_SUCCESS;
}
int32_t nv_rm_context_signal(struct NvRmDevice *dev, uint64_t context,
                            const struct NvRmSyncPoint *signals, uint32_t count,
                            uint64_t *token)
{
   check_sync_api(dev); assert(context == 33 && count == 2);
   assert(signals[0].sync == 77 && signals[0].value == 9);
   assert(signals[1].sync == 78 && signals[1].value == 0);
   *token = ++sync_api.batches == 1 ? 9 : 11;
   return sync_api.batches == 1 ? VK_NOT_READY : VK_SUCCESS;
}
int32_t nv_rm_wait(struct NvRmDevice *dev, uint64_t context, uint64_t token,
                  uint64_t timeout)
{
   if (fusion.active) {
      check_sync_api(dev); assert(context == 33 && token == 9);
      assert(timeout == UINT64_MAX); fusion.ring_waits++;
      return fusion.wait_result;
   }
   check_sync_api(dev); assert(context == 33 && (token == 9 || token == 11));
   assert(timeout == UINT64_MAX); sync_api.waits++; return VK_SUCCESS;
}
int32_t nv_rm_submit_batch(struct NvRmDevice *dev, uint64_t context,
                          const struct NvRmExec *pushes, uint32_t count,
                          const struct NvRmSyncPoint *waits, uint32_t nwaits,
                          const struct NvRmSyncPoint *signals, uint32_t nsignals,
                          uint64_t *token)
{
   if (!fusion.active) {
      if (nsignals) {
         assert(!count && !nwaits);
         return nv_rm_context_signal(dev, context, signals, nsignals, token);
      }
      return nv_rm_submit_sync(dev, context, pushes, count, waits, nwaits, token);
   }
   check_sync_api(dev); assert(context == 33 && fusion.next < fusion.count);
   unsigned i = fusion.next++;
   assert(count == fusion.calls[i].pushes && nwaits == fusion.calls[i].waits);
   assert(nsignals == fusion.calls[i].signals);
   for (unsigned p = 0; p < count; p++) {
      assert(pushes[p].address == fusion.calls[i].address + 4 * p);
      assert(pushes[p].size_B == 4);
   }
   if (nwaits) assert(nwaits == 1 && waits[0].sync == 77 && waits[0].value == 8);
   if (nsignals) {
      assert(nsignals == 2 && signals[0].sync == 77 && signals[0].value == 9);
      assert(signals[1].sync == 78 && signals[1].value == 0);
   }
   *token = fusion.calls[i].result == VK_NOT_READY ? 9 : 100 + i;
   return fusion.calls[i].result;
}
int32_t nv_rm_context_destroy(struct NvRmDevice *dev, uint64_t context)
{ check_sync_api(dev); assert(context == 33); sync_api.destroys++; return VK_SUCCESS; }

static void test_sync_api(struct vk_device *vk, const struct vk_sync_type *type,
                          const struct vk_sync_type *file_type)
{
   sync_api = (struct sync_api_state){ .active = true };
   struct nvkmd_rm_sync a = { .base = { .type = type }, .id = 77 };
   struct nvkmd_rm_sync f = { .base = { .type = file_type }, .id = 77 };
   struct nvkmd_rm_sync b = { .base = { .type = type }, .id = 78 };
   assert(vk_sync_reset(vk, &a.base) == VK_SUCCESS && sync_api.resets == 1);
   assert(vk_sync_move(vk, &a.base, &b.base) == VK_SUCCESS && sync_api.shared == 0);
   b.base.flags = VK_SYNC_IS_SHAREABLE | VK_SYNC_IS_SHARED;
   assert(vk_sync_move(vk, &a.base, &b.base) == VK_SUCCESS && sync_api.shared == 1);
   bool reserved = false;
   assert(type->reserve_signal && file_type->reserve_signal);
   assert(type->reserve_signal(vk, &a.base, &b.base, &reserved) == VK_SUCCESS &&
          reserved && sync_api.reserves == 1);
   assert(vk_sync_import_sync_file(vk, &f.base, -1) == VK_SUCCESS);
   assert(sync_api.signals == 1 && sync_api.value == 0 && sync_api.imports == 0);
   assert(vk_sync_import_sync_file(vk, &f.base, 66) == VK_SUCCESS);
   assert(sync_api.imports == 1 && sync_api.opaque == 0);
   int fd;
   assert(vk_sync_export_sync_file(vk, &f.base, &fd) == VK_SUCCESS && fd == 88);
   assert(sync_api.opaque == 0);
   assert(vk_sync_import_opaque_fd(vk, &a.base, 66) == VK_SUCCESS);
   assert(sync_api.opaque == 1 && (a.base.flags & VK_SYNC_IS_SHARED));
   assert(vk_sync_export_opaque_fd(vk, &a.base, &fd) == VK_SUCCESS && fd == 88);
   assert(sync_api.opaque == 1);
   a.base.flags = VK_SYNC_IS_TIMELINE;
   uint64_t value;
   assert(vk_sync_get_value(vk, &a.base, &value) == VK_SUCCESS && value == 8);
   assert(vk_sync_signal(vk, &a.base, 9) == VK_SUCCESS && sync_api.value == 9);

   struct nvkmd_rm_ctx *ctx = calloc(1, sizeof(*ctx));
   assert(ctx);
   ctx->base = (struct nvkmd_ctx){ .ops = &nvkmd_rm_ctx_ops,
                                  .dev = &rm_sync_device(vk)->base };
   ctx->vk = vk; ctx->channel = 33;
   util_dynarray_init(&ctx->pending, NULL); util_dynarray_init(&ctx->waits, NULL);
   struct vk_sync_wait wait = { .sync = &a.base, .wait_value = 8 };
   assert(rm_ctx_wait(&ctx->base, NULL, 1, &wait) == VK_SUCCESS);
   struct nvkmd_ctx_exec exec = { .addr = 0x1000, .size_B = 4 };
   assert(rm_ctx_exec(&ctx->base, NULL, 1, &exec) == VK_SUCCESS);
   assert(rm_ctx_flush(&ctx->base, NULL) == VK_SUCCESS);
   assert(sync_api.submits == 2 && sync_api.waits == 1 && ctx->last_token == 10);
   assert(!util_dynarray_num_elements(&ctx->waits, struct NvRmSyncPoint));
   struct vk_sync_signal signals[] = {
      { .sync = &a.base, .signal_value = 9 }, { .sync = &b.base },
   };
   assert(rm_ctx_signal(&ctx->base, NULL, 2, signals) == VK_SUCCESS);
   assert(sync_api.batches == 2 && sync_api.waits == 2 && ctx->last_token == 11);
   assert(rm_ctx_sync(&ctx->base, NULL) == VK_SUCCESS && sync_api.waits == 3);
   assert(sync_api.exports == 2); /* Only the two explicit exports above. */
   assert(rm_ctx_exec(&ctx->base, NULL, 1, &exec) == VK_SUCCESS);
   rm_ctx_destroy(&ctx->base);
   assert(sync_api.destroys == 1 && sync_api.waits == 3 && sync_api.submits == 2);
   /* The actual void finish callback disposes the payload on the caller,
    * including after loss. A backend failure must publish Mesa/Rust loss
    * through the safe hook. */
   fake.loss_device = vk;
   fake.loss_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
   assert(fake.loss_fd >= 0);
   vk->notify_lost = rm_notify_lost;
   vk->notify_lost_data = (void *)(uintptr_t)42;
   p_atomic_set(&vk->_lost.lost, 0);
   fake.rm_lost = 0;
   int destroys = fake.sync_destroys, losses = fake.loss_calls;
   a.base.type->finish(vk, &a.base);
   assert(fake.sync_destroys == destroys + 1 && fake.loss_calls == losses);
   fake.deferred_error = 1;
   unsigned diagnostics = fake.stderr_count;
   a.base.type->finish(vk, &a.base);
   assert(fake.sync_destroys == destroys + 2 && fake.stderr_count == diagnostics + 1);
   assert(strstr(fake.stderr_diagnostic, "sync disposal failed"));
   assert(vk_device_is_lost_no_report(vk) && fake.rm_lost);
   assert(fake.loss_calls == losses + 1);
   uint64_t notification;
   assert(read(fake.loss_fd, &notification, sizeof(notification)) == sizeof(notification));
   fake.deferred_error = 0;
   a.base.type->finish(vk, &a.base);
   assert(fake.sync_destroys == destroys + 3 && fake.loss_calls == losses + 1);
   vk->notify_lost = NULL;
   vk->notify_lost_data = NULL;
   fake.loss_device = NULL;
   assert(__real_close(fake.loss_fd) == 0);
   p_atomic_set(&vk->_lost.lost, 0);
   fake.rm_lost = 0;
   sync_api.active = false;
   puts("PASS: sync flags/reset/move/FD/-1 import and ordinary submit/signal/ring retry/destroy contracts");
}
