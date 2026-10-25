/* SPDX-License-Identifier: MIT
 * Executes the production bridge with native sync payloads against fake DRM/RM.
 * No device file or GPU operation is permitted by any wrapper. */
#include "../nvkmd_rm.c"
#include <assert.h>
#include <stdarg.h>
#include <errno.h>
#include "nvk_mem_stream.h"
#include "vk_alloc.h"
#include "vk_queue.h"
#include "vk_common_entrypoints.h"
#include <time.h>
#include <sys/eventfd.h>
#include <poll.h>


static struct {
   int open_error, version_mode, cap_error;
   uint64_t binary, timeline;
   int create_error, wait_error, stat_error, rm_error, rm_partial, rm_lost;
   int calloc_fail, calloc_calls, allocations;
   int opens, failed_opens, closes, rm_opens, rm_closes, creates, destroys;
   int sync_creates, sync_destroys, deferred_error, cap_queries;
   char diagnostic[512], stderr_diagnostic[512];
   unsigned stderr_count;
   struct vk_device *loss_device;
   int loss_fd, loss_calls, allow_no_vk_loss;
} fake;
#include "wait-loss.inc"

int __wrap_fprintf(FILE *stream, const char *format, ...)
{
   assert(stream == stderr);
   va_list ap; va_start(ap, format);
   int result = vsnprintf(fake.stderr_diagnostic, sizeof(fake.stderr_diagnostic), format, ap);
   va_end(ap);
   fake.stderr_count++;
   return result;
}

static void *owned[128];
void *__real_calloc(size_t, size_t);
void __real_free(void *);
void *__wrap_calloc(size_t n, size_t size)
{
   if (++fake.calloc_calls == fake.calloc_fail)
      return NULL;
   void *p = __real_calloc(n, size);
   if (p) {
      unsigned i;
      for (i = 0; i < ARRAY_SIZE(owned) && owned[i]; i++);
      assert(i < ARRAY_SIZE(owned));
      owned[i] = p;
      fake.allocations++;
   }
   return p;
}
void __wrap_free(void *p)
{
   if (p) {
      for (unsigned i = 0; i < ARRAY_SIZE(owned); i++) {
         if (owned[i] == p) {
            owned[i] = NULL;
            fake.allocations--;
            break;
         }
      }
   }
   __real_free(p);
}
/* Mesa 26's util_cache_granularity() (nvkmd_rm.c nc_atom_size_B) runs the
 * CPU detection once, which reads sysfs (u_cpu_detect.c get_cpu_topology ->
 * os_read_file). Those read-only sysfs opens go to the real libc; every
 * other open must be the fake render node. */
int __real_open64(const char *path, int flags, ...);
int __real_close(int fd);
static int sysfs_fds[64];
static unsigned sysfs_fd_count;
int __wrap_open64(const char *path, int flags, ...)
{
   if (strncmp(path, "/sys/", 5) == 0) {
      assert((flags & O_ACCMODE) == O_RDONLY);
      int fd = __real_open64(path, flags);
      if (fd >= 0) {
         assert(sysfs_fd_count < ARRAY_SIZE(sysfs_fds));
         sysfs_fds[sysfs_fd_count++] = fd;
      }
      return fd;
   }
   assert(strcmp(path, "/fake/render") == 0);
   assert((flags & (O_RDWR | O_CLOEXEC)) == (O_RDWR | O_CLOEXEC));
   fake.opens++;
   if (fake.open_error) { fake.failed_opens++; errno = EACCES; return -1; }
   return 99;
}
int __wrap_close(int fd)
{
   for (unsigned i = 0; i < sysfs_fd_count; i++) {
      if (sysfs_fds[i] == fd) {
         sysfs_fds[i] = sysfs_fds[--sysfs_fd_count];
         return __real_close(fd);
      }
   }
   assert(fd == 99); fake.closes++; return 0;
}
int __real_fstat64(int fd, struct stat64 *st);
int __wrap_fstat64(int fd, struct stat64 *st)
{
   for (unsigned i = 0; i < sysfs_fd_count; i++) {
      if (sysfs_fds[i] == fd)
         return __real_fstat64(fd, st);
   }
   assert(fd == 99);
   if (fake.stat_error) { errno = EIO; return -1; }
   memset(st, 0, sizeof(*st)); st->st_rdev = 0x1234; return 0;
}
drmVersionPtr __wrap_drmGetVersion(int fd)
{
   assert(fd == 99);
   static drmVersion version;
   if (fake.version_mode == 1) return NULL;
   version.name = fake.version_mode == 2 ? "nouveau" :
                  fake.version_mode == 3 ? NULL : "nvidia-drm";
   return &version;
}
void __wrap_drmFreeVersion(drmVersionPtr version) { assert(version); }
int __wrap_drmGetCap(int fd, uint64_t capability, uint64_t *value)
{
   assert(fd == 99);
   fake.cap_queries++;
   assert(capability == DRM_CAP_SYNCOBJ || capability == DRM_CAP_SYNCOBJ_TIMELINE);
   if (fake.cap_error == (int)capability) return -EINVAL;
   *value = capability == DRM_CAP_SYNCOBJ ? fake.binary : fake.timeline;
   return 0;
}
int __wrap_drmSyncobjCreate(int fd, uint32_t flags, uint32_t *handle)
{ assert(!"native sync must never create DRM syncobjs"); abort(); }
int __wrap_drmSyncobjWait(int fd, uint32_t *handles, unsigned count,
                         int64_t timeout, unsigned flags, uint32_t *first)
{ assert(!"native sync must never wait on DRM syncobjs"); abort(); }
int __wrap_drmSyncobjDestroy(int fd, uint32_t handle)
{ assert(!"native sync must never destroy DRM syncobjs"); abort(); }
VkResult __wrap___vk_errorf(const void *object, VkResult result,
                            const char *file, int line, const char *format, ...)
{
   (void)object; (void)file; (void)line;
   if (format) {
      va_list ap; va_start(ap, format);
      vsnprintf(fake.diagnostic, sizeof(fake.diagnostic), format, ap);
      va_end(ap);
   }
   return result;
}
int32_t nv_rm_open(uint32_t domain, uint32_t bus, uint32_t device, uint32_t function,
                   struct NvRmDevice **out, struct NvRmInfo *info)
{
   assert(domain == 1 && bus == 2 && device == 3 && function == 4);
   fake.rm_opens++;
   if (!fake.rm_error || fake.rm_partial) *out = (struct NvRmDevice *)(uintptr_t)42;
   *info = (struct NvRmInfo) { .pci_domain=1, .pci_bus=2, .pci_device=3,
      .pci_function=4, .generic_kind=6, .sm=86, .cls_eng3d=0xc797 };
   return fake.rm_error ? VK_ERROR_INITIALIZATION_FAILED : VK_SUCCESS;
}
void nv_rm_close(struct NvRmDevice *device)
{
   assert(device == (struct NvRmDevice *)(uintptr_t)42);
   if (fake.loss_device) {
      assert(!fake.loss_device->notify_lost);
      assert(!fake.loss_device->notify_lost_data);
      assert(!fake.loss_device->check_status);
   }
   fake.rm_closes++;
}
void nv_rm_mark_lost(const struct NvRmDevice *device)
{
   assert(device == (const struct NvRmDevice *)(uintptr_t)42);
   assert(fake.loss_device ? vk_device_is_lost_no_report(fake.loss_device) : fake.allow_no_vk_loss);
   fake.loss_calls++;
   p_atomic_set(&fake.rm_lost, 1);
   uint64_t one = 1;
   assert(write(fake.loss_fd, &one, sizeof(one)) == sizeof(one));
}
int32_t nv_rm_is_lost(const struct NvRmDevice *device)
{ assert(device == (const struct NvRmDevice *)(uintptr_t)42); return p_atomic_read(&fake.rm_lost); }
VkResult __wrap___vk_errorv(const void *object, VkResult result,
                            const char *file, int line, const char *format, va_list ap)
{
   (void)object; (void)file; (void)line;
   if (format) vsnprintf(fake.diagnostic, sizeof(fake.diagnostic), format, ap);
   return result;
}
void __wrap___vk_log_impl(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                          VkDebugUtilsMessageTypeFlagsEXT types, int object_count,
                          const void **objects, const char *file, int line,
                          const char *format, ...)
{
   (void)severity; (void)types; (void)object_count; (void)objects;
   (void)file; (void)line; (void)format;
}
int32_t nv_rm_sync_create(struct NvRmDevice *device, uint32_t flags,
                          uint64_t initial, uint64_t *out)
{
   assert(device == (struct NvRmDevice *)(uintptr_t)42);
   assert((flags & ~7u) == 0);
   fake.sync_creates++;
   if (fake.create_error) return VK_ERROR_OUT_OF_HOST_MEMORY;
   *out = 77;
   return VK_SUCCESS;
}
int32_t nv_rm_sync_destroy(struct NvRmDevice *device, uint64_t id)
{
   assert(!"void Mesa finish must not use synchronous sync destruction");
   abort();
}
int32_t nv_rm_sync_destroy_deferred(struct NvRmDevice *device, uint64_t id)
{
   assert(device == (struct NvRmDevice *)(uintptr_t)42 && id == 77);
   fake.sync_destroys++;
   return fake.deferred_error ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
}
/* Every unused Rust bridge entrypoint is a tripwire, not a no-op success. */
#include "sync-api.inc"
#include "submit-fusion.inc"
#include "memory-api.inc"
#include "pdev-unused-rm.inc"
#include "sync-properties.inc"
#include "submit-on-demand.inc"

static void reset(void)
{
   assert(fake.allocations == 0);
   memset(&fake, 0, sizeof(fake));
   fake.binary = fake.timeline = 1;
}
static drmDevice drm_device(void)
{
   static char *nodes[DRM_NODE_MAX] = { [DRM_NODE_RENDER] = "/fake/render" };
   static drmPciDeviceInfo info = { .vendor_id = NVIDIA_VENDOR_ID };
   static drmPciBusInfo bus = { .domain=1, .bus=2, .dev=3, .func=4 };
   return (drmDevice) { .nodes=nodes, .available_nodes=1<<DRM_NODE_RENDER,
      .bustype=DRM_BUS_PCI, .deviceinfo.pci=&info, .businfo.pci=&bus };
}
static struct nvkmd_pdev *create_pdev(VkResult expected, const char *message)
{
   drmDevice drm = drm_device();
   struct nvkmd_pdev *out = (void *)(uintptr_t)1;
   VkResult result = nvkmd_rm_try_create_pdev(&drm, NULL, 0, &out);
   assert(result == expected);
   if (result != VK_SUCCESS) {
      assert(out == NULL && fake.allocations == 0);
      assert(fake.closes == (fake.open_error ? 0 : fake.opens));
      if (message) assert(strstr(fake.diagnostic, message));

   } else {
      assert(out && out->sync_types && out->sync_types[0] && out->sync_types[1] && !out->sync_types[2]);
      assert(out->sync_types[0]->features & VK_SYNC_FEATURE_TIMELINE);
      assert(out->sync_types[0]->features & VK_SYNC_FEATURE_WAIT_BEFORE_SIGNAL);
      assert(out->sync_types[0]->init == rm_sync_init);
      assert(fake.sync_creates == 0); /* Enumeration never initializes RM syncs. */
   }
   return out;
}
#include "loss-notification.inc"
#include "memory-cleanup.inc"

int main(void)
{
   test_loss_notification();
   test_memory_cleanup();
   reset(); fake.open_error=1; create_pdev(VK_ERROR_INCOMPATIBLE_DRIVER, NULL);
   for (int i=1; i<=3; i++) {
      reset(); fake.version_mode=i; create_pdev(VK_ERROR_INCOMPATIBLE_DRIVER, NULL);
   }
   reset(); fake.calloc_fail=1; create_pdev(VK_ERROR_OUT_OF_HOST_MEMORY, NULL);
   /* 550-like absent capabilities (and failed legacy capability queries)
    * no longer gate the native sync implementation. No syncobj operation is
    * issued: only the selected NVIDIA node and Rust RM ABI gates apply. */
   for (unsigned mode = 0; mode < 3; mode++) {
      reset(); fake.binary = fake.timeline = 0;
      fake.cap_error = mode == 1 ? DRM_CAP_SYNCOBJ : mode == 2 ? DRM_CAP_SYNCOBJ_TIMELINE : 0;
      struct nvkmd_pdev *accepted = create_pdev(VK_SUCCESS, NULL);
      struct nvkmd_dev *logical = NULL;
      assert(rm_dev_create(accepted, NULL, &logical) == VK_SUCCESS && logical);
      assert(fake.cap_queries == 0 && fake.creates == 0 && fake.destroys == 0);
      rm_dev_destroy(logical); rm_pdev_destroy(accepted);
      assert(fake.allocations == 0 && fake.rm_opens == fake.rm_closes && fake.opens == fake.closes);
   }
   reset(); fake.stat_error=1; create_pdev(VK_ERROR_INITIALIZATION_FAILED, "stat failed"); assert(fake.rm_opens==0);
   reset(); fake.rm_error=1; create_pdev(VK_ERROR_INITIALIZATION_FAILED, "query device"); assert(fake.rm_closes==0);
   reset(); fake.rm_error=1; fake.rm_partial=1; create_pdev(VK_ERROR_INITIALIZATION_FAILED, "query device"); assert(fake.rm_closes==1);
   reset(); struct nvkmd_pdev *pdev=create_pdev(VK_SUCCESS, NULL);
   assert(fake.rm_opens==1 && fake.rm_closes==1 && fake.creates==0 && fake.destroys==0 && fake.cap_queries==0);
   struct nvkmd_dev *dev=(void *)(uintptr_t)1;
   fake.calloc_fail=fake.calloc_calls+1;
   assert(rm_dev_create(pdev, NULL, &dev)==VK_ERROR_OUT_OF_HOST_MEMORY && dev==NULL);
   fake.calloc_fail=0; fake.open_error=1; dev=(void *)(uintptr_t)1;
   assert(rm_dev_create(pdev, NULL, &dev)==VK_ERROR_INITIALIZATION_FAILED && dev==NULL);
   fake.open_error=0;
   fake.binary=0; fake.timeline=0;
   fake.rm_error=1; fake.rm_partial=1; dev=(void *)(uintptr_t)1;
   assert(rm_dev_create(pdev, NULL, &dev)==VK_ERROR_INITIALIZATION_FAILED && dev==NULL);
   fake.rm_error=0; fake.rm_partial=0;
   assert(rm_dev_create(pdev, NULL, &dev)==VK_SUCCESS && dev);
   /* The logical Vulkan device's separate sync-provider allocation can fail
    * after RM device creation. The checked NULL result selects fail_nvkmd. */
   struct vk_device vk = {0};
   fake.calloc_fail=fake.calloc_calls+1;
   vk_device_set_drm_fd(&vk, 99);
   assert(vk.sync==NULL);
   fake.calloc_fail=0;
   vk_device_set_drm_fd(&vk, 99);
   assert(vk.sync && !vk.sync->timeline_wait); /* No DRM timeline capability needed. */
   /* Execute the actual NVK call that formerly consumed the zero sync type.
    * It must fail before vk_sync_create can dereference its NULL init method. */
   struct nvk_physical_device physical = { .nvkmd = pdev };
   struct nvk_device logical = {0};
   logical.vk.physical = &physical.vk;
   logical.nvkmd = dev;
   logical.vk.alloc = *vk_default_allocator();
   logical.vk.sync = vk.sync;
   struct nvk_mem_stream stream = {0};
   const struct vk_sync_type *const *saved_types = pdev->sync_types;
   pdev->sync_types = NULL;
   assert(nvk_mem_stream_init(&logical, &stream)==VK_ERROR_INITIALIZATION_FAILED && stream.sync==NULL);
   struct vk_sync_type zero_type = {0};
   const struct vk_sync_type *zero_types[] = { &zero_type, NULL };
   pdev->sync_types = zero_types;
   assert(nvk_mem_stream_init(&logical, &stream)==VK_ERROR_INITIALIZATION_FAILED && stream.sync==NULL);
   zero_type = *saved_types[0]; zero_type.features &= ~VK_SYNC_FEATURE_TIMELINE;
   assert(nvk_mem_stream_init(&logical, &stream)==VK_ERROR_INITIALIZATION_FAILED && stream.sync==NULL);
   pdev->sync_types = saved_types;
   fake.create_error=1;
   assert(nvk_mem_stream_init(&logical, &stream)!=VK_SUCCESS && stream.sync==NULL);
   fake.create_error=0;
   assert(nvk_mem_stream_init(&logical, &stream)==VK_SUCCESS && stream.sync);
   nvk_mem_stream_finish(&logical, &stream);
   /* As nvk_create_drm_physical_device (supported_sync_types =
    * nvkmd->sync_types): Mesa 26's vk_sync_wait_many reads them
    * (vk_sync.c can_wait_many). */
   physical.vk.supported_sync_types = saved_types;
   test_wait_loss(&logical.vk, saved_types[0]);
   test_sync_api(&logical.vk, saved_types[0], saved_types[1]);
   test_submit_fusion(&logical.vk, saved_types[0]);
   test_submit_on_demand(&logical.vk);
   test_sync_properties(&physical.vk, saved_types);
   assert(fake.creates == 0 && fake.destroys == 0);
   vk.sync->finalize(vk.sync);
   rm_dev_destroy(dev); rm_pdev_destroy(pdev);
   assert(fake.allocations==0 && fake.opens-fake.failed_opens==fake.closes && fake.rm_opens==fake.rm_closes);
   puts("PASS: native sync accepts absent DRM syncobj caps; selected-driver/RM gates and pdev/dev/provider cleanup; no syncobj ioctls or device calls");
   return 0;
}
