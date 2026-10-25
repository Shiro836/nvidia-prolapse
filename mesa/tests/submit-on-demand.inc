/* THREADED_ON_DEMAND routing through the actual patched vk_queue.c.
 *
 * Invariant under test: queue::driver_submit never receives a wait whose
 * signal has not been submitted (or host-signalled), so a ring-full CPU wait
 * inside the driver can never depend on work only the submitting thread could
 * still submit. Also: once the submit thread is drained, ordinary submissions
 * (all waits pending) run on the caller again, in order.
 *
 * A minimal timeline vk_sync stands in for the RM payload: `pending` is the
 * largest submitted/host value (what VK_SYNC_WAIT_PENDING observes), and the
 * fake driver completes a signal at submission. */
struct od_sync {
   struct vk_sync base;
   uint64_t pending, completed;
   struct od_sync *target; /* reserved temporary: signals land on target */
};
static struct {
   mtx_t mutex;
   cnd_t cond;
   unsigned calls, caller_calls;
   thrd_t caller;
   uint64_t order[64];
} od;

static struct od_sync *od_cast(struct vk_sync *sync)
{
   return container_of(sync, struct od_sync, base);
}
static VkResult od_init(struct vk_device *d, struct vk_sync *s, uint64_t v)
{
   od_cast(s)->pending = od_cast(s)->completed = v;
   return VK_SUCCESS;
}
static void od_finish(struct vk_device *d, struct vk_sync *s) {}
static void od_publish(struct vk_sync *s, uint64_t v)
{
   if (!(s->flags & VK_SYNC_IS_TIMELINE))
      v = 1; /* binary: signalled */
   mtx_lock(&od.mutex);
   struct od_sync *sync = od_cast(s);
   if (sync->target)
      sync = sync->target;
   sync->pending = MAX2(sync->pending, v);
   sync->completed = MAX2(sync->completed, v);
   cnd_broadcast(&od.cond);
   mtx_unlock(&od.mutex);
}
static VkResult od_signal(struct vk_device *d, struct vk_sync *s, uint64_t v)
{
   od_publish(s, v);
   return VK_SUCCESS;
}
static VkResult od_get_value(struct vk_device *d, struct vk_sync *s, uint64_t *v)
{
   mtx_lock(&od.mutex);
   *v = od_cast(s)->completed;
   mtx_unlock(&od.mutex);
   return VK_SUCCESS;
}
static bool od_ready(const struct vk_sync_wait *w, enum vk_sync_wait_flags f)
{
   struct od_sync *s = od_cast(w->sync);
   uint64_t target = (w->sync->flags & VK_SYNC_IS_TIMELINE) ? w->wait_value : 1;
   return s->completed >= target ||
          ((f & VK_SYNC_WAIT_PENDING) && s->pending >= target);
}
static VkResult od_wait_many(struct vk_device *d, uint32_t count,
                             const struct vk_sync_wait *waits,
                             enum vk_sync_wait_flags flags, uint64_t deadline)
{
   assert(!(flags & VK_SYNC_WAIT_ANY));
   mtx_lock(&od.mutex);
   for (;;) {
      bool ready = true;
      for (uint32_t i = 0; i < count; i++)
         ready &= od_ready(&waits[i], flags);
      if (ready) {
         mtx_unlock(&od.mutex);
         return VK_SUCCESS;
      }
      if (deadline != UINT64_MAX) {
         /* Only zero-timeout checks reach here in this test. */
         mtx_unlock(&od.mutex);
         return VK_TIMEOUT;
      }
      cnd_wait(&od.cond, &od.mutex);
   }
}
static VkResult od_reset(struct vk_device *d, struct vk_sync *s)
{
   mtx_lock(&od.mutex);
   od_cast(s)->pending = od_cast(s)->completed = 0;
   mtx_unlock(&od.mutex);
   return VK_SUCCESS;
}
static VkResult od_move(struct vk_device *d, struct vk_sync *dst, struct vk_sync *src)
{
   mtx_lock(&od.mutex);
   od_cast(dst)->pending = od_cast(src)->pending;
   od_cast(dst)->completed = od_cast(src)->completed;
   od_cast(src)->pending = od_cast(src)->completed = 0;
   mtx_unlock(&od.mutex);
   return VK_SUCCESS;
}
static unsigned od_reserves;
static VkResult od_reserve(struct vk_device *d, struct vk_sync *dst,
                           struct vk_sync *src, bool *reserved)
{
   mtx_lock(&od.mutex);
   /* Model: the reserved signal is pending at once; the temporary forwards. */
   od_cast(dst)->target = od_cast(src);
   od_cast(src)->pending = 1;
   od_reserves++;
   cnd_broadcast(&od.cond);
   mtx_unlock(&od.mutex);
   *reserved = true;
   return VK_SUCCESS;
}
static const struct vk_sync_type od_type = {
   .size = sizeof(struct od_sync),
   .features = VK_SYNC_FEATURE_BINARY | VK_SYNC_FEATURE_CPU_RESET | VK_SYNC_FEATURE_TIMELINE | VK_SYNC_FEATURE_GPU_WAIT |
      VK_SYNC_FEATURE_CPU_WAIT | VK_SYNC_FEATURE_CPU_SIGNAL |
      VK_SYNC_FEATURE_WAIT_PENDING | VK_SYNC_FEATURE_WAIT_BEFORE_SIGNAL,
   .init = od_init, .finish = od_finish, .signal = od_signal,
   .get_value = od_get_value, .wait_many = od_wait_many,
   .reset = od_reset, .move = od_move, .reserve_signal = od_reserve,
};

static VkResult od_driver_submit(struct vk_queue *queue, struct vk_queue_submit *submit)
{
   mtx_lock(&od.mutex);
   for (uint32_t i = 0; i < submit->wait_count; i++) {
      /* The invariant: every received wait is already pending. */
      assert(od_ready(&submit->waits[i], VK_SYNC_WAIT_PENDING));
   }
   if (thrd_equal(thrd_current(), od.caller))
      od.caller_calls++;
   assert(od.calls < ARRAY_SIZE(od.order));
   od.order[od.calls++] = submit->signal_count ? submit->signals[0].signal_value : 0;
   mtx_unlock(&od.mutex);
   for (uint32_t i = 0; i < submit->signal_count; i++)
      od_publish(submit->signals[i].sync, submit->signals[i].signal_value);
   return VK_SUCCESS;
}

static VkSemaphore od_semaphore(struct vk_device *device)
{
   VkSemaphoreTypeCreateInfo type = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
   };
   VkSemaphoreCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &type,
   };
   VkSemaphore s;
   assert(vk_common_CreateSemaphore(vk_device_to_handle(device), &info, NULL, &s) == VK_SUCCESS);
   return s;
}
static void od_submit(struct vk_queue *queue, VkSemaphore wait, uint64_t wait_value,
                      VkSemaphore signal, uint64_t signal_value)
{
   VkSemaphoreSubmitInfo w = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = wait, .value = wait_value, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT };
   VkSemaphoreSubmitInfo s = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = signal, .value = signal_value, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT };
   VkSubmitInfo2 info = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = wait ? 1 : 0, .pWaitSemaphoreInfos = &w,
      .signalSemaphoreInfoCount = signal ? 1 : 0, .pSignalSemaphoreInfos = &s };
   assert(vk_common_QueueSubmit2(vk_queue_to_handle(queue), 1, &info, VK_NULL_HANDLE) == VK_SUCCESS);
}
static unsigned od_calls(void)
{
   mtx_lock(&od.mutex);
   unsigned calls = od.calls;
   mtx_unlock(&od.mutex);
   return calls;
}
static void od_host_signal(struct vk_device *device, VkSemaphore s, uint64_t v)
{
   VkSemaphoreSignalInfo info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
      .semaphore = s, .value = v };
   assert(vk_common_SignalSemaphore(vk_device_to_handle(device), &info) == VK_SUCCESS);
}
static void od_wait(struct vk_device *device, VkSemaphore s, uint64_t v)
{
   VkSemaphoreWaitInfo info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 1, .pSemaphores = &s, .pValues = &v };
   assert(vk_common_WaitSemaphores(vk_device_to_handle(device), &info, UINT64_MAX) == VK_SUCCESS);
}

static void test_submit_on_demand(struct vk_device *device)
{
   const struct vk_sync_type *const *saved_types = device->physical->supported_sync_types;
   enum vk_queue_submit_mode saved_submit = device->submit_mode;
   enum vk_device_timeline_mode saved_timeline = device->timeline_mode;
   const struct vk_sync_type *types[] = { &od_type, NULL };
   /* The harness device is a bare struct: give it its handle type so the
    * real vk_common_* entrypoints accept it. */
   VkObjectType saved_type = device->base.type;
   device->base.type = VK_OBJECT_TYPE_DEVICE;
   device->physical->supported_sync_types = types;
   device->timeline_mode = VK_DEVICE_TIMELINE_MODE_NATIVE;
   device->submit_mode = VK_QUEUE_SUBMIT_MODE_THREADED_ON_DEMAND;
   p_atomic_set(&device->_lost.lost, 0);
   memset(&od, 0, sizeof(od));
   od_reserves = 0;
   assert(mtx_init(&od.mutex, mtx_plain) == thrd_success);
   assert(cnd_init(&od.cond) == thrd_success);
   od.caller = thrd_current();

   struct vk_queue queue;
   VkDeviceQueueCreateInfo info = { .queueCount = 1 };
   assert(vk_queue_init(&queue, device, &info, 0) == VK_SUCCESS);
   assert(queue.submit.mode == VK_QUEUE_SUBMIT_MODE_IMMEDIATE);
   queue.driver_submit = od_driver_submit;
   VkSemaphore gate = od_semaphore(device), done = od_semaphore(device);

   /* Ordinary submits on a fresh queue: caller, no thread. */
   od_submit(&queue, VK_NULL_HANDLE, 0, done, 1);
   od_submit(&queue, done, 1, done, 2);
   assert(od_calls() == 2 && od.caller_calls == 2);
   assert(queue.submit.mode == VK_QUEUE_SUBMIT_MODE_IMMEDIATE);

   /* Wait-before-signal on the same thread, then work behind it: none of it
    * reaches the driver until the signal exists (the former ring deadlock). */
   od_submit(&queue, gate, 1, VK_NULL_HANDLE, 0);
   assert(queue.submit.mode == VK_QUEUE_SUBMIT_MODE_THREADED);
   for (uint64_t v = 3; v < 40; v++)
      od_submit(&queue, VK_NULL_HANDLE, 0, done, v);
   /* A pending wait queued behind the thread keeps submission order too. */
   od_submit(&queue, done, 2, done, 40);
   assert(od_calls() == 2);
   od_host_signal(device, gate, 1);
   od_wait(device, done, 40);
   assert(od_calls() == 2 + 1 + 38 && od.caller_calls == 2);
   assert(od.order[0] == 1 && od.order[1] == 2 && od.order[2] == 0);
   for (unsigned i = 3; i < od.calls; i++)
      assert(od.order[i] == i); /* submission order kept through the thread */

   /* Drained thread: ordinary submissions run on the caller again. */
   od_submit(&queue, gate, 1, done, 41);
   od_submit(&queue, VK_NULL_HANDLE, 0, done, 42);
   assert(od_calls() == 43 && od.caller_calls == 4);
   assert(vk_common_QueueWaitIdle(vk_queue_to_handle(&queue)) == VK_SUCCESS);
   assert(od.caller_calls == 5); /* idle-queue signal on the caller as well */

   /* A second wait-before-signal episode on the same queue. */
   od_submit(&queue, gate, 2, done, 43);
   od_submit(&queue, VK_NULL_HANDLE, 0, done, 44);
   assert(od_calls() == 44);
   od_host_signal(device, gate, 2);
   od_wait(device, done, 44);
   assert(od_calls() == 46 && od.caller_calls == 5);
   for (unsigned i = 1; i < od.calls; i++)
      assert(od.order[i] == 0 || od.order[i] > od.order[i - 1]);

   /* A deferred submission's binary signal is reserved at vkQueueSubmit: a
    * binary wait submitted behind it (THREADED branch: pending wait + move)
    * must not block until the thread runs. Without reserve_signal this call
    * would wait for gate 3, which only this thread signals afterwards. */
   VkSemaphoreCreateInfo binary_info = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
   VkSemaphore binary;
   assert(vk_common_CreateSemaphore(vk_device_to_handle(device), &binary_info, NULL, &binary) == VK_SUCCESS);
   VkSemaphoreSubmitInfo gate_wait = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = gate, .value = 3, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT };
   VkSemaphoreSubmitInfo binary_op = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = binary, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT };
   VkSubmitInfo2 producer = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = 1, .pWaitSemaphoreInfos = &gate_wait,
      .signalSemaphoreInfoCount = 1, .pSignalSemaphoreInfos = &binary_op };
   assert(vk_common_QueueSubmit2(vk_queue_to_handle(&queue), 1, &producer, VK_NULL_HANDLE) == VK_SUCCESS);
   assert(od_reserves == 1);
   VkSubmitInfo2 consumer = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = 1, .pWaitSemaphoreInfos = &binary_op };
   od_submit(&queue, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, 0); /* keep list order */
   assert(vk_common_QueueSubmit2(vk_queue_to_handle(&queue), 1, &consumer, VK_NULL_HANDLE) == VK_SUCCESS);
   od_submit(&queue, VK_NULL_HANDLE, 0, done, 45);
   assert(od_calls() == 46); /* nothing reached the driver before the gate */
   od_host_signal(device, gate, 3);
   od_wait(device, done, 45);
   assert(od_calls() == 50 && od.caller_calls == 5);
   vk_common_DestroySemaphore(vk_device_to_handle(device), binary, NULL);

   vk_common_DestroySemaphore(vk_device_to_handle(device), gate, NULL);
   vk_common_DestroySemaphore(vk_device_to_handle(device), done, NULL);
   vk_queue_finish(&queue);
   cnd_destroy(&od.cond);
   mtx_destroy(&od.mutex);
   device->physical->supported_sync_types = saved_types;
   device->submit_mode = saved_submit;
   device->timeline_mode = saved_timeline;
   device->base.type = saved_type;
   puts("PASS: THREADED_ON_DEMAND never hands the driver an unsubmitted wait; same-thread wait-before-signal queues behind the thread; drained queues submit on the caller in order; deferred binary signals are reserved so a binary wait behind them never blocks vkQueueSubmit");
}
