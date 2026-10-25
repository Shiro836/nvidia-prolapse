int __real_close(int);
/* Real Mesa wait/provider/queue paths, scripted RM outcomes and one real
 * submit thread. Deadlines use a deterministic clock except the threaded case. */
static struct wait_case_data {
   bool active, threaded, drain;
   struct vk_device *device;
   uint64_t now, deadline;
   unsigned calls, lose_at, rust_lose_at, finish_at, flags, count;
   VkResult finish_result;
   bool timeline;
   uint64_t value;
   int entered;
} wait_case;

int64_t __real_os_time_get_nano(void);
int64_t __wrap_os_time_get_nano(void)
{
   return wait_case.active && !wait_case.threaded ? wait_case.now :
          __real_os_time_get_nano();
}

int __real_nanosleep(const struct timespec *, struct timespec *);
int __wrap_nanosleep(const struct timespec *delay, struct timespec *remain)
{
   assert(delay->tv_sec == 0 && delay->tv_nsec > 0 && delay->tv_nsec <= 1000000);
   if (wait_case.active && !wait_case.threaded) {
      wait_case.now += delay->tv_nsec;
      assert(wait_case.now <= wait_case.deadline);
      return 0;
   }
   return __real_nanosleep(delay, remain);
}

int32_t nv_rm_sync_status_many(struct NvRmDevice *device,
                               const struct NvRmSyncPoint *points, uint32_t count,
                               uint32_t flags, uint32_t *ready)
{
   assert(!"C wait must call the blocking boundary exactly once, not poll status");
   abort();
}

int32_t nv_rm_sync_wait_many(struct NvRmDevice *device,
                             const struct NvRmSyncPoint *points, uint32_t count,
                             uint32_t flags, uint64_t timeout_ns)
{
   assert(device == (struct NvRmDevice *)(uintptr_t)42 && wait_case.active);
   assert(count == wait_case.count && flags == wait_case.flags);
   assert(timeout_ns == (wait_case.deadline == UINT64_MAX ? UINT64_MAX :
      wait_case.deadline > wait_case.now ? wait_case.deadline - wait_case.now : 0));
   for (unsigned i = 0; i < count; i++) {
      assert(points[i].sync == 77);
      assert(points[i].value == wait_case.value);
   }
   wait_case.calls++;
   assert(wait_case.calls == 1);
   p_atomic_set(&wait_case.entered, 1);
   if (wait_case.threaded) {
      /* Simulate the backend waiter wake boundary with a real FD. No timer
       * polling is accepted: the submit callback or maintenance must wake it. */
      struct pollfd fd = { .fd = fake.loss_fd, .events = POLLIN };
      assert(poll(&fd, 1, 1000) == 1 && fd.revents == POLLIN);
      uint64_t value;
      assert(read(fd.fd, &value, sizeof(value)) == sizeof(value) && value);
      assert(p_atomic_read(&fake.rm_lost));
      return VK_ERROR_DEVICE_LOST;
   }
   if (wait_case.lose_at)
      vk_device_set_lost(wait_case.device, "scripted software loss during wait");
   if (wait_case.rust_lose_at)
      p_atomic_set(&fake.rm_lost, 1);
   if (wait_case.finish_at) {
      wait_case.now += (uint64_t)(wait_case.finish_at - 1) * 1000000;
      /* Success racing loss deliberately remains success at this boundary;
       * the adapter must recheck the loss snapshot after waking. */
      return wait_case.finish_result;
   }
   if (wait_case.lose_at || wait_case.rust_lose_at)
      return VK_ERROR_DEVICE_LOST;
   assert(timeout_ns != UINT64_MAX);
   wait_case.now += timeout_ns;
   return 2;
}

int __wrap_drmSyncobjTimelineWait(int fd, uint32_t *handles, uint64_t *values,
                                 unsigned count, int64_t deadline,
                                 unsigned flags, uint32_t *first)
{
   assert(!"ordinary RM wait must not use DRM timeline waits");
   abort();
}

static VkResult
test_failed_submit(struct vk_queue *queue, struct vk_queue_submit *submit)
{
   (void)queue;
   assert(submit->signal_count == (wait_case.drain ? 0 : 1));
   if (wait_case.threaded)
      while (!p_atomic_read(&wait_case.entered)) thrd_yield();
   if (wait_case.drain) {
      struct timespec delay = { .tv_nsec = 20000000 };
      thrd_sleep(&delay, NULL);
   }
   return VK_ERROR_DEVICE_LOST;
}

static int
test_publish_rust_loss(void *data)
{
   struct vk_device *device = data;
   while (!p_atomic_read(&wait_case.entered)) thrd_yield();
   assert(!vk_device_is_lost_no_report(device));
   p_atomic_set(&fake.rm_lost, 1);
   uint64_t one = 1;
   assert(write(fake.loss_fd, &one, sizeof(one)) == sizeof(one));
   return 0;
}

static void
test_wait_loss(struct vk_device *device, const struct vk_sync_type *type)
{
   fake.loss_device = device;
   fake.loss_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
   assert(fake.loss_fd >= 0);
   device->notify_lost_data = (void *)(uintptr_t)42;
   device->notify_lost = rm_notify_lost;
   struct nvkmd_rm_sync objects[2] = {
      { .base = { .type = type }, .id = 77 },
      { .base = { .type = type }, .id = 77 },
   };
   struct vk_sync_wait waits[2] = {
      { .sync = &objects[0].base }, { .sync = &objects[1].base },
   };
   /* Complete binary, timeline and pending waits; ALL and ANY keep their
    * flags and values. Lost, finite expiry, late success and RM status error. */
   for (unsigned mode = 0; mode < 4; mode++) {
      for (unsigned any = 0; any < 2; any++) {
         for (unsigned outcome = 0; outcome < 7; outcome++) {
            p_atomic_set(&device->_lost.lost, 0);
            fake.rm_lost = 0;
            for (unsigned i = 0; i < 2; i++) {
               objects[i].base.flags = mode == 1 || mode == 2 ? VK_SYNC_IS_TIMELINE : 0;
               waits[i].wait_value = mode == 1 || mode == 2 ? 3 : 0;
            }
            wait_case = (struct wait_case_data) {
               .active = true, .device = device, .now = 1000000000,
               .deadline = outcome == 1 ? UINT64_C(1002000000) : UINT64_MAX,
               .count = 2, .timeline = mode != 0, .value = mode == 1 || mode == 2 ? 3 : 0,
               .flags = (any ? NV_RM_SYNC_WAIT_ANY : 0) |
                        (mode >= 2 ? NV_RM_SYNC_WAIT_PENDING : 0),
               .lose_at = outcome == 0 || outcome == 4 ? 2 : 0,
               .rust_lose_at = outcome >= 5 ? 2 : 0,
               .finish_at = outcome >= 2 && outcome != 5 ? 2 : 0,
               .finish_result = outcome == 3 ? VK_ERROR_UNKNOWN : VK_SUCCESS,
            };
            VkResult expected = outcome == 0 || outcome >= 4 ? VK_ERROR_DEVICE_LOST :
               outcome == 1 ? VK_TIMEOUT : outcome == 2 ? VK_SUCCESS : VK_ERROR_UNKNOWN;
            enum vk_sync_wait_flags flags = (any ? VK_SYNC_WAIT_ANY : 0) |
               (mode >= 2 ? VK_SYNC_WAIT_PENDING : VK_SYNC_WAIT_COMPLETE);
            assert(vk_sync_wait_many(device, 2, waits, flags, wait_case.deadline) == expected);
            assert(wait_case.calls == 1);
            /* Drain scripted notifications before the next independent case. */
            uint64_t event_value;
            while (read(fake.loss_fd, &event_value, sizeof(event_value)) == sizeof(event_value));
            assert(errno == EAGAIN);
            if (outcome >= 5) {
               assert(vk_device_is_lost_no_report(device));
               assert(strstr(fake.diagnostic, "RM maintenance reported"));
            }
         }
      }
   }
   p_atomic_set(&device->_lost.lost, 0);
   fake.rm_lost = 0;
   wait_case = (struct wait_case_data) {
      .active = true, .device = device, .now = 1000000000, .deadline = 0,
      .count = 1, .flags = 0,
   };
   objects[0].base.flags = 0;
   assert(vk_sync_wait(device, &objects[0].base, 0, VK_SYNC_WAIT_COMPLETE, 0) == VK_TIMEOUT);
   assert(wait_case.calls == 1);
   p_atomic_inc(&device->_lost.lost);
   assert(vk_sync_wait(device, &objects[0].base, 0, VK_SYNC_WAIT_COMPLETE, UINT64_MAX) == VK_ERROR_DEVICE_LOST);
   assert(wait_case.calls == 1); /* Already lost: never query the backend. */
   p_atomic_set(&device->_lost.lost, 0);
   fake.rm_lost = 1;
   assert(vk_sync_wait(device, &objects[0].base, 0, VK_SYNC_WAIT_COMPLETE, UINT64_MAX) == VK_ERROR_DEVICE_LOST);
   assert(wait_case.calls == 1 && vk_device_is_lost_no_report(device));
   fake.rm_lost = 0;
   wait_case.active = false;

   uint64_t event_value;
   while (read(fake.loss_fd, &event_value, sizeof(event_value)) == sizeof(event_value));
   assert(errno == EAGAIN);

   /* Rust maintenance can discover missing completion while no Vulkan job is
    * running. Only the shared snapshot changes on this helper thread. */
   p_atomic_set(&device->_lost.lost, 0);
   wait_case = (struct wait_case_data) {
      .active = true, .threaded = true, .device = device,
      .deadline = UINT64_MAX, .count = 1,
      .flags = 0,
   };
   thrd_t maintenance;
   assert(thrd_create(&maintenance, test_publish_rust_loss, device) == thrd_success);
   uint64_t rust_before = __real_os_time_get_nano();
   assert(vk_sync_wait(device, &objects[0].base, 0, VK_SYNC_WAIT_COMPLETE, UINT64_MAX) == VK_ERROR_DEVICE_LOST);
   assert((uint64_t)__real_os_time_get_nano() - rust_before < UINT64_C(1000000000));
   thrd_join(maintenance, NULL);
   assert(wait_case.calls >= 1 && vk_device_is_lost_no_report(device));
   assert(strstr(fake.diagnostic, "RM maintenance reported"));
   fake.rm_lost = 0;
   wait_case.active = false;

   while (read(fake.loss_fd, &event_value, sizeof(event_value)) == sizeof(event_value));
   assert(errno == EAGAIN);

   /* A legitimate dependency held beyond the old five-second age must stay
    * pending and eventually complete. Deterministic time avoids slow tests. */
   p_atomic_set(&device->_lost.lost, 0);
   wait_case = (struct wait_case_data) {
      .active = true, .device = device, .now = 1000000000,
      .deadline = UINT64_MAX, .count = 1, .finish_at = 6252,
      .finish_result = VK_SUCCESS,
   };
   assert(vk_sync_wait(device, &objects[0].base, 0, VK_SYNC_WAIT_COMPLETE,
                       UINT64_MAX) == VK_SUCCESS);
   assert(wait_case.now == UINT64_C(7251000000));
   assert(!vk_device_is_lost_no_report(device));
   wait_case.active = false;

   /* The actual QueueWaitIdle -> submit thread -> driver_submit failure path.
    * Its sync lifecycle and waits use the real Mesa implementation. */
   for (unsigned threaded = 0; threaded < 2; threaded++) {
      p_atomic_set(&device->_lost.lost, 0);
      p_atomic_set(&fake.rm_lost, 0);
      while (read(fake.loss_fd, &event_value, sizeof(event_value)) == sizeof(event_value));
      assert(errno == EAGAIN);
      device->_lost.reported = true;
      list_inithead(&device->queues);
      device->physical->supported_sync_types = (const struct vk_sync_type *[]) { type, NULL };
      device->submit_mode = threaded ? VK_QUEUE_SUBMIT_MODE_THREADED : VK_QUEUE_SUBMIT_MODE_IMMEDIATE;
      device->timeline_mode = VK_DEVICE_TIMELINE_MODE_ASSISTED;
      struct vk_queue queue;
      VkDeviceQueueCreateInfo info = { .queueCount = 1 };
      assert(vk_queue_init(&queue, device, &info, 0) == VK_SUCCESS);
      queue.driver_submit = test_failed_submit;
      wait_case = (struct wait_case_data) {
         .active = true, .threaded = threaded, .device = device,
         .deadline = UINT64_MAX, .count = 1,
         .flags = 0,
      };
      int destroys_before = fake.sync_destroys;
      uint64_t before = __real_os_time_get_nano();
      assert(vk_common_QueueWaitIdle(vk_queue_to_handle(&queue)) == VK_ERROR_DEVICE_LOST);
      assert((uint64_t)__real_os_time_get_nano() - before < UINT64_C(1000000000));
      assert(threaded ? wait_case.calls >= 1 : wait_case.calls == 0);
      assert(fake.sync_destroys == destroys_before + 1);
      vk_queue_finish(&queue); /* Failed threaded submit remains queued. */
      wait_case.active = false;
   }
   /* A queue destroy can already be sleeping in drain when submit fails.
    * The submit-thread failure must broadcast pop even with no next job. */
   p_atomic_set(&device->_lost.lost, 0);
   wait_case = (struct wait_case_data) {
      .active = true, .threaded = true, .drain = true, .device = device,
   };
   struct vk_queue queue;
   VkDeviceQueueCreateInfo info = { .queueCount = 1 };
   assert(vk_queue_init(&queue, device, &info, 0) == VK_SUCCESS);
   queue.driver_submit = test_failed_submit;
   VkSubmitInfo2 submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
   assert(vk_common_QueueSubmit2(vk_queue_to_handle(&queue), 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS);
   p_atomic_set(&wait_case.entered, 1);
   uint64_t before = __real_os_time_get_nano();
   vk_queue_finish(&queue);
   assert((uint64_t)__real_os_time_get_nano() - before < UINT64_C(1000000000));
   assert(vk_device_is_lost_no_report(device));
   wait_case.active = false;
   p_atomic_set(&device->_lost.lost, 0);
   device->notify_lost = NULL;
   device->notify_lost_data = NULL;
   fake.loss_device = NULL;
   assert(__real_close(fake.loss_fd) == 0);
   puts("PASS: real Mesa RM waits preserve deadlines/ANY/ALL/timeline/pending and >6s completion; actual threaded QueueWaitIdle loss and teardown");
}
