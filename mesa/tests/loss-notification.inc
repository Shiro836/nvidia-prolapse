/* Actual Mesa setters and canonical adapter hook; no GPU/RM operation.
 * An eventfd models only the nonblocking Rust wake boundary. */
int __real_close(int);

static void
expect_loss_wake(unsigned count)
{
   struct pollfd fd = { .fd = fake.loss_fd, .events = POLLIN };
   assert(poll(&fd, 1, 0) == 1 && fd.revents == POLLIN);
   uint64_t value = 0;
   assert(read(fake.loss_fd, &value, sizeof(value)) == sizeof(value));
   assert(value == count);
   assert(poll(&fd, 1, 0) == 0);
}

static void
test_loss_notification(void)
{
   reset();
   struct nvkmd_pdev *pdev = create_pdev(VK_SUCCESS, NULL);
   struct vk_device vk = {0};
   list_inithead(&vk.queues);
   vk.base.device = &vk;
   struct nvkmd_dev *dev = NULL;
   assert(rm_dev_create(pdev, &vk.base, &dev) == VK_SUCCESS);
   assert(vk.notify_lost == rm_notify_lost && vk.notify_lost_data);
   fake.loss_device = &vk;
   fake.loss_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
   assert(fake.loss_fd >= 0);

   /* Direct software device loss publishes before notification, and a repeated
    * setter cannot leave a second unread notification for this same loss. */
   assert(vk_device_set_lost(&vk, "offline device failure") == VK_ERROR_DEVICE_LOST);
   assert(fake.loss_calls == 1 && fake.rm_lost);
   expect_loss_wake(1);
   assert(vk_device_set_lost(&vk, "repeat") == VK_ERROR_DEVICE_LOST);
   assert(fake.loss_calls == 1);

   /* Queue loss directly increments device loss; it must also notify while its
    * submit mutex is held. Two independently failed queues may each notify. */
   p_atomic_set(&vk._lost.lost, 0);
   struct vk_queue queues[2] = {0};
   for (unsigned i = 0; i < 2; i++) {
      queues[i].base.device = &vk;
      assert(mtx_init(&queues[i].submit.mutex, mtx_plain) == thrd_success);
      mtx_lock(&queues[i].submit.mutex);
      assert(vk_queue_set_lost(&queues[i], "offline submit failure") == VK_ERROR_DEVICE_LOST);
      assert(vk_queue_set_lost(&queues[i], "repeat") == VK_ERROR_DEVICE_LOST);
      mtx_unlock(&queues[i].submit.mutex);
      mtx_destroy(&queues[i].submit.mutex);
   }
   assert(fake.loss_calls == 3 && vk._lost.lost == 2);
   expect_loss_wake(2);

   /* Fake nv_rm_close asserts callback/data/status are cleared before freeing
    * the owner. Further software loss cannot access that destroyed backend. */
   rm_dev_destroy(dev);
   p_atomic_set(&vk._lost.lost, 0);
   assert(vk_device_set_lost(&vk, "after backend detach") == VK_ERROR_DEVICE_LOST);
   assert(fake.loss_calls == 3);
   assert(__real_close(fake.loss_fd) == 0);
   fake.loss_device = NULL;
   rm_pdev_destroy(pdev);
   assert(fake.allocations == 0 && fake.opens == fake.closes && fake.rm_opens == fake.rm_closes);
   puts("PASS: actual device/queue loss wakes after publication, repeated loss, queue-lock callback and detach-before-close");
}
