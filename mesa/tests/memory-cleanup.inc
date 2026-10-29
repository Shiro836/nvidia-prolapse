/* Actual production mapping callbacks, actual Mesa loss publication/callback;
 * failures are injected solely at the Rust ABI boundary. No GPU or device IO. */
static void
memory_reset_case(struct vk_device *vk)
{
   memory_api = (struct memory_api_state) { .active = true, .address = 0x12345000 };
   p_atomic_set(&vk->_lost.lost, 0);
   p_atomic_set(&fake.rm_lost, 0);
   fake.loss_calls = 0;
   fake.diagnostic[0] = fake.stderr_diagnostic[0] = 0;
}
static void
memory_expect_lost(struct vk_device *vk, const char *operation)
{
   assert(vk_device_is_lost_no_report(vk));
   assert(fake.rm_lost && fake.loss_calls == 1);
   assert(strstr(fake.stderr_diagnostic, operation));
   expect_loss_wake(1);
}
static void
test_memory_cleanup(void)
{
   reset();
   struct nvkmd_pdev *pdev = create_pdev(VK_SUCCESS, NULL);
   struct vk_device vk = {0};
   list_inithead(&vk.queues);
   vk.base.device = &vk;
   struct nvkmd_dev *base = NULL;
   assert(rm_dev_create(pdev, &vk.base, &base) == VK_SUCCESS);
   fake.loss_device = &vk;
   fake.loss_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
   assert(fake.loss_fd >= 0);
   struct nvkmd_rm_mem mem = {
      .base = { .dev = base, .ops = &nvkmd_rm_mem_ops },
      .allocation = { .id = 55 },
   };
   void *address;

   memory_reset_case(&vk);
   address = (void *)(uintptr_t)1;
   assert(rm_mem_map(&mem.base, &vk.base, 0, NULL, &address) == VK_SUCCESS);
   assert(address == (void *)memory_api.address && memory_api.maps == 1);
   rm_mem_unmap(&mem.base, 0, address);
   assert(memory_api.unmaps == 1 && !vk_device_is_lost_no_report(&vk));

   const int32_t errors[] = { VK_ERROR_OUT_OF_HOST_MEMORY,
      VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_INITIALIZATION_FAILED };
   for (unsigned i = 0; i < ARRAY_SIZE(errors); i++) {
      memory_reset_case(&vk);
      memory_api.map_status = errors[i];
      address = (void *)(uintptr_t)1;
      VkResult expected = i < 2 ? errors[i] : VK_ERROR_MEMORY_MAP_FAILED;
      assert(rm_mem_map(&mem.base, &vk.base, 0, NULL, &address) == expected);
      assert(address == NULL && memory_api.maps == 1 && !memory_api.unmaps);
      assert(!vk_device_is_lost_no_report(&vk) && !fake.loss_calls && !fake.rm_lost);
      assert(strstr(fake.diagnostic, "CPU map failed"));
   }
   memory_reset_case(&vk);
   address = (void *)(uintptr_t)1;
   assert(rm_mem_map(&mem.base, &vk.base, NVKMD_MEM_MAP_FIXED, NULL, &address) == VK_ERROR_FEATURE_NOT_PRESENT);
   assert(address == NULL && !memory_api.maps && !fake.loss_calls);

   for (unsigned original = 0; original < 2; original++) {
      memory_reset_case(&vk);
      memory_api.map_status = original ? VK_ERROR_INITIALIZATION_FAILED : VK_ERROR_DEVICE_LOST;
      memory_api.publish_loss = original;
      /* vkMapMemory has no VK_ERROR_DEVICE_LOST; the loss is published. */
      assert(rm_mem_map(&mem.base, &vk.base, 0, NULL, &address) == VK_ERROR_MEMORY_MAP_FAILED);
      assert(address == NULL);
      memory_expect_lost(&vk, "CPU map");
   }
   memory_reset_case(&vk);
   memory_api.address = 0;
   assert(rm_mem_map(&mem.base, &vk.base, 0, NULL, &address) == VK_ERROR_MEMORY_MAP_FAILED);
   assert(address == NULL && memory_api.unmaps == 1 && !fake.loss_calls);
   assert(strstr(fake.diagnostic, "null address"));
   memory_reset_case(&vk);
   memory_api.address = 0;
   memory_api.unmap_status = VK_ERROR_INITIALIZATION_FAILED;
   assert(rm_mem_map(&mem.base, &vk.base, 0, NULL, &address) == VK_ERROR_MEMORY_MAP_FAILED);
   assert(address == NULL && memory_api.unmaps == 1);
   memory_expect_lost(&vk, "null CPU map rollback");

   /* A void callback has already discarded its caller's logical ownership:
    * even an OOM-shaped cleanup error is irreversible, unlike map OOM above. */
   memory_reset_case(&vk);
   memory_api.unmap_status = VK_ERROR_OUT_OF_HOST_MEMORY;
   rm_mem_unmap(&mem.base, 0, (void *)(uintptr_t)0x12345000);
   memory_expect_lost(&vk, "CPU unmap");

   for (unsigned failed = 0; failed < 2; failed++) {
      memory_reset_case(&vk);
      memory_api.free_status = failed ? VK_ERROR_INITIALIZATION_FAILED : VK_SUCCESS;
      struct nvkmd_rm_mem *owned_mem = calloc(1, sizeof(*owned_mem));
      assert(owned_mem);
      nvkmd_mem_init(base, &owned_mem->base, &nvkmd_rm_mem_ops, NVKMD_MEM_CAN_MAP, 4096, 4096);
      owned_mem->allocation.id = 55;
      assert(rm_va_alloc(base, &vk.base, 0, 0, 4096, 4096, 0, &owned_mem->base.va) == VK_SUCCESS);
      rm_mem_free(&owned_mem->base);
      assert(memory_api.frees == 1);
      if (failed) memory_expect_lost(&vk, "memory free");
      else assert(!fake.loss_calls && !vk_device_is_lost_no_report(&vk));
   }

   for (unsigned imported = 0; imported < 2; imported++) {
      for (unsigned cleanup_failed = 0; cleanup_failed < 2; cleanup_failed++) {
         memory_reset_case(&vk);
         memory_api.free_status = cleanup_failed ? VK_ERROR_INITIALIZATION_FAILED : VK_SUCCESS;
         /* mem allocation succeeds; the subsequent real rm_va_alloc calloc fails. */
         fake.calloc_fail = fake.calloc_calls + 2;
         struct nvkmd_mem *out = NULL;
         VkResult result = imported ? rm_mem_import(base, &vk.base, 22, &out) :
            rm_mem_alloc(base, &vk.base, 4096, 4096, NVKMD_MEM_CAN_MAP, &out);
         fake.calloc_fail = 0;
         /* vkAllocateMemory has no VK_ERROR_DEVICE_LOST: a failed rollback
          * publishes the loss and keeps the allocation's own failure. */
         assert(result == VK_ERROR_OUT_OF_HOST_MEMORY);
         assert(out == NULL && memory_api.frees == 1);
         if (cleanup_failed)
            memory_expect_lost(&vk, imported ? "import rollback free" : "allocation rollback free");
         else assert(!fake.loss_calls && !vk_device_is_lost_no_report(&vk));
      }
   }
   /* VA teardown hands every mapping to one nv_rm_va_free and never unbinds
    * synchronously. A failed call cannot return this address to Mesa's
    * allocator, even after its C wrapper disappears, and publishes loss. */
   for (unsigned failed = 0; failed < 2; failed++) {
      memory_reset_case(&vk);
      memory_api.va_free_status = failed ? VK_ERROR_INITIALIZATION_FAILED : VK_SUCCESS;
      struct nvkmd_va *va = NULL;
      mem.base.size_B = 4096;
      /* A capture/replay VA, so the fixed re-allocation below probes it. */
      assert(rm_va_alloc(base, &vk.base, NVKMD_VA_REPLAY, 0, 4096, 4096, 0, &va) == VK_SUCCESS);
      uint64_t reserved = va->addr;
      assert(rm_va_bind(va, &vk.base, 0, &mem.base, 0, 4096) == VK_SUCCESS);
      rm_va_free(va);
      assert(memory_api.binds == 1 && memory_api.unbinds == 0 && memory_api.va_frees == 1);
      assert(memory_api.va_count == 1 && memory_api.va_ids[0] == 66);
      assert(memory_api.va_address == reserved && memory_api.va_size == 4096);
      if (failed)
         memory_expect_lost(&vk, "VA free; reservation retained");
      else
         assert(!fake.loss_calls && !vk_device_is_lost_no_report(&vk));
      va = NULL;
      VkResult result = rm_va_alloc(base, &vk.base, NVKMD_VA_REPLAY | NVKMD_VA_ALLOC_FIXED, 0,
                                    4096, 4096, reserved, &va);
      assert(result == (failed ? VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS : VK_SUCCESS));
      if (va) {
         memory_api.va_free_status = VK_SUCCESS;
         rm_va_free(va);
      }
   }
   /* Deferred release: the range stays allocated (fixed re-allocation fails)
    * until the backend's release runs, then becomes reusable. No loss. */
   {
      memory_reset_case(&vk);
      memory_api.defer_release = true;
      struct nvkmd_va *va = NULL;
      assert(rm_va_alloc(base, &vk.base, NVKMD_VA_REPLAY, 0, 4096, 4096, 0, &va) == VK_SUCCESS);
      uint64_t reserved = va->addr;
      assert(rm_va_bind(va, &vk.base, 0, &mem.base, 0, 4096) == VK_SUCCESS);
      rm_va_free(va);
      assert(memory_api.va_frees == 1 && memory_api.release);
      struct nvkmd_va *again = NULL;
      assert(rm_va_alloc(base, &vk.base, NVKMD_VA_REPLAY | NVKMD_VA_ALLOC_FIXED, 0, 4096, 4096,
                         reserved, &again) == VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
      memory_run_deferred_release();
      assert(rm_va_alloc(base, &vk.base, NVKMD_VA_REPLAY | NVKMD_VA_ALLOC_FIXED, 0, 4096, 4096,
                         reserved, &again) == VK_SUCCESS);
      memory_api.defer_release = false;
      rm_va_free(again); /* no bindings: immediate release */
      assert(memory_api.va_frees == 2 && memory_api.va_count == 0 && !memory_api.release);
      assert(!fake.loss_calls && !vk_device_is_lost_no_report(&vk));
   }
   /* Capture/replay VAs come from their own range: an ordinary VA never takes
    * a replayable address, a replayed address must lie in that range, and a
    * fixed VA is only a replayed capture address. */
   {
      memory_reset_case(&vk);
      struct nvkmd_va *plain = NULL, *capture = NULL, *fixed = NULL;
      assert(rm_va_alloc(base, &vk.base, 0, 0, 4096, 4096, 0, &plain) == VK_SUCCESS);
      assert(rm_va_alloc(base, &vk.base, NVKMD_VA_REPLAY, 0, 4096, 4096, 0, &capture) == VK_SUCCESS);
      assert(plain->addr >= NV_RM_VA_START && plain->addr < NVKMD_RM_REPLAY_HEAP_START);
      assert(capture->addr >= NVKMD_RM_REPLAY_HEAP_START && capture->addr < NV_RM_VA_END);
      assert(rm_va_alloc(base, &vk.base, NVKMD_VA_REPLAY | NVKMD_VA_ALLOC_FIXED, 0, 4096, 4096,
                         plain->addr, &fixed) == VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
      assert(rm_va_alloc(base, &vk.base, NVKMD_VA_ALLOC_FIXED, 0, 4096, 4096,
                         capture->addr + 4096, &fixed) == VK_ERROR_FEATURE_NOT_PRESENT);
      assert(fixed == NULL);
      rm_va_free(plain);
      rm_va_free(capture);
      assert(!fake.loss_calls && !vk_device_is_lost_no_report(&vk));
   }
   /* Application-ordered rebind over an existing binding unbinds immediately
    * (no backend wait) before binding the same address again. */
   {
      memory_reset_case(&vk);
      struct nvkmd_va *va = NULL;
      assert(rm_va_alloc(base, &vk.base, 0, 0, 4096, 4096, 0, &va) == VK_SUCCESS);
      assert(rm_va_bind(va, &vk.base, 0, &mem.base, 0, 4096) == VK_SUCCESS);
      assert(rm_va_bind(va, &vk.base, 0, &mem.base, 0, 4096) == VK_SUCCESS);
      assert(memory_api.binds == 2 && memory_api.unbinds == 1);
      rm_va_free(va);
      assert(memory_api.va_frees == 1 && memory_api.va_count == 1);
      assert(!fake.loss_calls && !vk_device_is_lost_no_report(&vk));
   }
   rm_dev_destroy(base);
   assert(__real_close(fake.loss_fd) == 0);
   fake.loss_device = NULL;
   rm_pdev_destroy(pdev);
   assert(!fake.allocations && fake.opens == fake.closes && fake.rm_opens == fake.rm_closes);

   /* Enumeration/physical probing has no logical vk_device, but must still
    * wake the Rust owner after irreversible cleanup failure. */
   reset();
   pdev = create_pdev(VK_SUCCESS, NULL);
   assert(rm_dev_create(pdev, NULL, &base) == VK_SUCCESS);
   fake.allow_no_vk_loss = 1;
   fake.loss_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
   assert(fake.loss_fd >= 0);
   memory_api = (struct memory_api_state){ .active = true, .unmap_status = VK_ERROR_INITIALIZATION_FAILED };
   mem.base.dev = base;
   rm_mem_unmap(&mem.base, 0, (void *)(uintptr_t)1);
   assert(fake.loss_calls == 1 && fake.rm_lost);
   expect_loss_wake(1);
   rm_dev_destroy(base);
   rm_pdev_destroy(pdev);
   assert(__real_close(fake.loss_fd) == 0);
   memory_api.active = false;
   assert(!fake.allocations && fake.opens == fake.closes && fake.rm_opens == fake.rm_closes);
   puts("PASS: map OOM remains recoverable; device loss preserved; void unmap/free and allocation/import rollback failures publish immediate loss/wake; null map lease rolled back; failed VA free publishes loss and retains reservation; deferred VA release keeps the range until the backend returns it; rebind unbinds without waiting");
}
