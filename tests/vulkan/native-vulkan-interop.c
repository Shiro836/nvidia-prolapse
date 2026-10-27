/* Independent Vulkan external dma-buf and sync-file interoperability test.
 * cc -std=c11 -Wall -Wextra -Werror tests/vulkan/native-vulkan-interop.c -ldl -o PROBE
 * PROBE --driver /absolute/ICD.so [--allow-stock-driver]
 * Two independent VkDevices share a real GPU buffer, with explicit queue-family
 * ownership and sync-fd semaphore/fence transfer. No private driver ABI. */
#define _GNU_SOURCE
#define VK_NO_PROTOTYPES
#include "native-driver-identity.h"
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>
#include "native-vulkan-runtime.h"

#define DEVICE_FUNCTIONS(X) \
  X(DestroyDevice) \
  X(GetDeviceQueue) \
  X(CreateBuffer) \
  X(DestroyBuffer) \
  X(GetBufferMemoryRequirements) \
  X(AllocateMemory) \
  X(FreeMemory) \
  X(BindBufferMemory) \
  X(MapMemory) \
  X(UnmapMemory) \
  X(GetMemoryFdKHR) \
  X(GetMemoryFdPropertiesKHR) \
  X(CreateCommandPool) \
  X(DestroyCommandPool) \
  X(AllocateCommandBuffers) \
  X(ResetCommandBuffer) \
  X(BeginCommandBuffer) \
  X(EndCommandBuffer) \
  X(CmdCopyBuffer) \
  X(CmdPipelineBarrier) \
  X(CreateFence) \
  X(DestroyFence) \
  X(WaitForFences) \
  X(ResetFences) \
  X(GetFenceStatus) \
  X(GetFenceFdKHR) \
  X(ImportFenceFdKHR) \
  X(CreateSemaphore) \
  X(DestroySemaphore) \
  X(GetSemaphoreFdKHR) \
  X(ImportSemaphoreFdKHR) \
  X(QueueSubmit) \
  X(DeviceWaitIdle)
struct Device {
  VkDevice handle;
  VkQueue queue;
  VkCommandPool pool;
  VkCommandBuffer command;
#define DECLARE(name) PFN_vk##name name;
  DEVICE_FUNCTIONS(DECLARE)
#undef DECLARE
};
struct Buffer {
  VkBuffer buffer;
  VkDeviceMemory memory;
  VkDeviceSize allocation_size;
  uint32_t *mapped;
};
static VkPhysicalDeviceMemoryProperties memory_properties;
static uint32_t queue_family;
static void require(int good, const char *operation) {
  if (!good) {
    fprintf(stderr, "FAIL: %s (errno %d: %s)\n", operation, errno,
            strerror(errno));
    exit(1);
  }
}
static void result(VkResult value, const char *operation) {
  if (value != VK_SUCCESS) {
    fprintf(stderr, "FAIL: %s returned %d\n", operation, value);
    exit(1);
  }
}
#define CHECK(call) result((call), #call)
static int fd_count(void) {
  DIR *d = opendir("/proc/self/fd");
  require(d != NULL, "count fd");
  int n = 0;
  struct dirent *e;
  while ((e = readdir(d)))
    if (e->d_name[0] != '.')
      n++;
  closedir(d);
  return n;
}
static void begin(struct Device *d);
static void init_device(struct Device *d, PFN_vkCreateDevice create,
                        PFN_vkGetDeviceProcAddr resolve,
                        VkPhysicalDevice physical) {
  float priority = 1;
  VkDeviceQueueCreateInfo q = {.sType =
                                   VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                               .queueFamilyIndex = queue_family,
                               .queueCount = 1,
                               .pQueuePriorities = &priority};
  const char *extensions[] = {
      "VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf",
      "VK_KHR_external_fence_fd", "VK_KHR_external_semaphore_fd"};
  VkDeviceCreateInfo info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &q,
                             .enabledExtensionCount = 4,
                             .ppEnabledExtensionNames = extensions};
  CHECK(create(physical, &info, NULL, &d->handle));
#define LOAD(name)                                                             \
  d->name = (PFN_vk##name)resolve(d->handle, "vk" #name);                      \
  require(d->name != NULL, "vk" #name);
  DEVICE_FUNCTIONS(LOAD)
#undef LOAD
  d->GetDeviceQueue(d->handle, queue_family, 0, &d->queue);
  require(d->queue != VK_NULL_HANDLE, "compute queue");
  VkCommandPoolCreateInfo pool = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = queue_family};
  CHECK(d->CreateCommandPool(d->handle, &pool, NULL, &d->pool));
  VkCommandBufferAllocateInfo command = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = d->pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
  CHECK(d->AllocateCommandBuffers(d->handle, &command, &d->command));
  /* Device creation leaves GPU work outstanding (the queue's initial state, internal uploads that NVK
   * flushes with the first command-buffer submission). The driver frees an object right away when no
   * submitted work is outstanding and otherwise defers the free until that work retires, so the dma-buf
   * import's property query below would free early or late depending on GPU timing, and two recordings of
   * one build (capture.py) would differ in ioctl order and in the GPU VAs handed out after it. */
  begin(d);
  CHECK(d->EndCommandBuffer(d->command));
  VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                         .commandBufferCount = 1,
                         .pCommandBuffers = &d->command};
  CHECK(d->QueueSubmit(d->queue, 1, &submit, VK_NULL_HANDLE));
  CHECK(d->DeviceWaitIdle(d->handle));
}
static struct Buffer create_buffer(struct Device *d, int shared, int import_fd,
                                   int host) {
  struct Buffer b = {0};
  VkExternalMemoryBufferCreateInfo external = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
  VkBufferCreateInfo info = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                             .pNext = shared ? &external : NULL,
                             .size = 65536,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
  CHECK(d->CreateBuffer(d->handle, &info, NULL, &b.buffer));
  VkMemoryRequirements req;
  d->GetBufferMemoryRequirements(d->handle, b.buffer, &req);
  uint32_t bits = req.memoryTypeBits;
  if (import_fd >= 0) {
    VkMemoryFdPropertiesKHR fdprops = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    CHECK(d->GetMemoryFdPropertiesKHR(
        d->handle, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, import_fd,
        &fdprops));
    bits &= fdprops.memoryTypeBits;
  }
  uint32_t type = UINT32_MAX;
  VkMemoryPropertyFlags needed = host ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                                      : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
    if ((bits & (1u << i)) &&
        (memory_properties.memoryTypes[i].propertyFlags & needed) == needed) {
      type = i;
      break;
    }
  require(type != UINT32_MAX, "compatible external/host memory type");
  VkExportMemoryAllocateInfo export = {
      .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
  VkImportMemoryFdInfoKHR import = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
      .fd = import_fd};
  VkMemoryDedicatedAllocateInfo dedicated = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
      .buffer = b.buffer,
      .pNext = import_fd >= 0 ? (const void *)&import
               : shared       ? (const void *)&export
                              : NULL};
  VkMemoryAllocateInfo allocate = {.sType =
                                       VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                   .pNext = &dedicated,
                                   .allocationSize = req.size,
                                   .memoryTypeIndex = type};
  CHECK(d->AllocateMemory(d->handle, &allocate, NULL, &b.memory));
  b.allocation_size = req.size;
  CHECK(d->BindBufferMemory(d->handle, b.buffer, b.memory, 0));
  if (host) {
    void *p = NULL;
    CHECK(d->MapMemory(d->handle, b.memory, 0, req.size, 0, &p));
    require(p != NULL, "mapped host buffer");
    b.mapped = p;
  }
  return b;
}
static void destroy_buffer(struct Device *d, struct Buffer *b) {
  if (b->mapped)
    d->UnmapMemory(d->handle, b->memory);
  d->DestroyBuffer(d->handle, b->buffer, NULL);
  d->FreeMemory(d->handle, b->memory, NULL);
}
static void begin(struct Device *d) {
  CHECK(d->ResetCommandBuffer(d->command, 0));
  VkCommandBufferBeginInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
  CHECK(d->BeginCommandBuffer(d->command, &info));
}
static void barrier(struct Device *d, struct Buffer *b, VkAccessFlags src,
                    VkAccessFlags dst, uint32_t src_family, uint32_t dst_family,
                    VkPipelineStageFlags src_stage,
                    VkPipelineStageFlags dst_stage) {
  VkBufferMemoryBarrier info = {.sType =
                                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                .srcAccessMask = src,
                                .dstAccessMask = dst,
                                .srcQueueFamilyIndex = src_family,
                                .dstQueueFamilyIndex = dst_family,
                                .buffer = b->buffer,
                                .offset = 0,
                                .size = VK_WHOLE_SIZE};
  d->CmdPipelineBarrier(d->command, src_stage, dst_stage, 0, 0, NULL, 1, &info,
                        0, NULL);
}
static VkFence fence(struct Device *d, int external) {
  VkExportFenceCreateInfo exp = {
      .sType = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT};
  VkFenceCreateInfo info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                            .pNext = external ? &exp : NULL};
  VkFence f = VK_NULL_HANDLE;
  CHECK(d->CreateFence(d->handle, &info, NULL, &f));
  return f;
}
static VkSemaphore semaphore(struct Device *d, int external) {
  VkExportSemaphoreCreateInfo exp = {
      .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
  VkSemaphoreCreateInfo info = {.sType =
                                    VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                .pNext = external ? &exp : NULL};
  VkSemaphore s = VK_NULL_HANDLE;
  CHECK(d->CreateSemaphore(d->handle, &info, NULL, &s));
  return s;
}
static void maps_check(int allow_stock, enum nv_expect_driver expect) {
  char why[4200];
  if (!allow_stock && nv_expect_driver_check(expect, why, sizeof(why))) {
    fprintf(stderr, "FAIL: %s", why);
    exit(1);
  }
}
int main(int argc, char **argv) {
  /* The fd checks count the whole process. Mesa's shader disk cache writes each entry on its own nice-19
   * "disk$" threads (src/util/disk_cache.c:90-93, src/util/u_queue.c:258) and holds a <hash>.tmp descriptor
   * from open to close (src/util/disk_cache_os.c:883-972, disk_cache_write_item_to_disk), unordered with
   * those counts. The explicit disable also covers a cache at an explicit path (disk_cache_os.c:1182).
   * Lines: Mesa 26.2.5 with this repository's Mesa patches applied. Set before any Mesa library loads. */
  if (setenv("MESA_SHADER_CACHE_DISABLE", "true", 1) != 0) {
    perror("setenv MESA_SHADER_CACHE_DISABLE");
    return 2;
  }
  const char *driver = NULL;
  int allow_stock = 0;
  enum nv_expect_driver expect = NV_EXPECT_PROLAPSE;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--driver") && i + 1 < argc)
      driver = argv[++i];
    else if (!strcmp(argv[i], "--allow-stock-driver"))
      allow_stock = 1;
    else if (!strcmp(argv[i], "--expect-driver") && i + 1 < argc)
      require(!nv_expect_driver_parse(argv[++i], &expect),
              "--expect-driver is nvidia-prolapse or glibc-shim");
    else
      require(0, "usage: PROBE --driver /absolute/ICD.so [--expect-driver "
                 "nvidia-prolapse|glibc-shim] [--allow-stock-driver]");
  }
  require(driver && driver[0] == '/', "absolute driver path");
  setvbuf(stdout, NULL, _IONBF, 0);
  nv_probe_android_runtime_init();
  int before = fd_count();
  void *library = dlopen(driver, RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    fprintf(stderr, "dlopen: %s\n", dlerror());
    return 1;
  }
  PFN_vkGetInstanceProcAddr gipa =
      (PFN_vkGetInstanceProcAddr)dlsym(library, "vkGetInstanceProcAddr");
  if (!gipa)
    gipa = (PFN_vkGetInstanceProcAddr)dlsym(library, "vk_icdGetInstanceProcAddr");
  require(gipa != NULL, "vkGetInstanceProcAddr");
  PFN_vkCreateInstance create =
      (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
  require(create != NULL, "vkCreateInstance");
  VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                           .pApplicationName =
                               "Independent external memory and sync probe",
                           .apiVersion = VK_API_VERSION_1_1};
  VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                             .pApplicationInfo = &app};
  VkInstance instance = VK_NULL_HANDLE;
  CHECK(create(&ci, NULL, &instance));
#define INSTANCE(name)                                                         \
  PFN_vk##name name = (PFN_vk##name)gipa(instance, "vk" #name);                \
  require(name != NULL, "vk" #name)
  INSTANCE(DestroyInstance);
  INSTANCE(EnumeratePhysicalDevices);
  INSTANCE(GetPhysicalDeviceProperties);
  INSTANCE(GetPhysicalDeviceMemoryProperties);
  INSTANCE(GetPhysicalDeviceQueueFamilyProperties);
  INSTANCE(GetPhysicalDeviceExternalBufferProperties);
  INSTANCE(GetPhysicalDeviceExternalFenceProperties);
  INSTANCE(GetPhysicalDeviceExternalSemaphoreProperties);
  INSTANCE(CreateDevice);
  INSTANCE(GetDeviceProcAddr);
#undef INSTANCE
  uint32_t n = 0;
  CHECK(EnumeratePhysicalDevices(instance, &n, NULL));
  require(n && n <= 64, "physical device count");
  VkPhysicalDevice devices[64], physical = VK_NULL_HANDLE;
  CHECK(EnumeratePhysicalDevices(instance, &n, devices));
  VkPhysicalDeviceProperties props;
  for (uint32_t i = 0; i < n; i++) {
    GetPhysicalDeviceProperties(devices[i], &props);
    if (props.vendorID == 0x10de) {
      physical = devices[i];
      break;
    }
  }
  require(physical != VK_NULL_HANDLE, "NVIDIA vendor device");
  printf("device: %.256s vendor=%04x device=%04x\n", props.deviceName,
         props.vendorID, props.deviceID);
  GetPhysicalDeviceMemoryProperties(physical, &memory_properties);
  require(memory_properties.memoryTypeCount <= 32, "memory type count");
  n = 0;
  GetPhysicalDeviceQueueFamilyProperties(physical, &n, NULL);
  require(n && n <= 64, "queue families");
  VkQueueFamilyProperties queues[64];
  GetPhysicalDeviceQueueFamilyProperties(physical, &n, queues);
  queue_family = UINT32_MAX;
  for (uint32_t i = 0; i < n; i++)
    if (queues[i].queueCount && (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
      queue_family = i;
      break;
    }
  require(queue_family != UINT32_MAX, "compute queue family");
  VkPhysicalDeviceExternalBufferInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
      .usage =
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
  VkExternalBufferProperties buffer_props = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
  GetPhysicalDeviceExternalBufferProperties(physical, &buffer_info,
                                            &buffer_props);
  require((buffer_props.externalMemoryProperties.externalMemoryFeatures &
           (VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT |
            VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT)) ==
              (VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT |
               VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT),
          "DMA-BUF import/export support");
  VkPhysicalDeviceExternalFenceInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO,
      .handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT};
  VkExternalFenceProperties fence_props = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES};
  GetPhysicalDeviceExternalFenceProperties(physical, &fence_info, &fence_props);
  require((fence_props.externalFenceFeatures &
           (VK_EXTERNAL_FENCE_FEATURE_IMPORTABLE_BIT |
            VK_EXTERNAL_FENCE_FEATURE_EXPORTABLE_BIT)) ==
              (VK_EXTERNAL_FENCE_FEATURE_IMPORTABLE_BIT |
               VK_EXTERNAL_FENCE_FEATURE_EXPORTABLE_BIT),
          "fence sync-fd import/export support");
  VkPhysicalDeviceExternalSemaphoreInfo semaphore_info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
      .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
  VkExternalSemaphoreProperties semaphore_props = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
  GetPhysicalDeviceExternalSemaphoreProperties(physical, &semaphore_info,
                                               &semaphore_props);
  require((semaphore_props.externalSemaphoreFeatures &
           (VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT |
            VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT)) ==
              (VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT |
               VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT),
          "semaphore sync-fd import/export support");
  printf("capabilities: DMA-BUF=0x%x fenceSYNC_FD=0x%x semaphoreSYNC_FD=0x%x\n",
         buffer_props.externalMemoryProperties.externalMemoryFeatures,
         fence_props.externalFenceFeatures,
         semaphore_props.externalSemaphoreFeatures);
  struct Device producer = {0}, consumer = {0};
  init_device(&producer, CreateDevice, GetDeviceProcAddr, physical);
  init_device(&consumer, CreateDevice, GetDeviceProcAddr, physical);
  struct Buffer upload = create_buffer(&producer, 0, -1, 1),
                shared = create_buffer(&producer, 1, -1, 0),
                readback = create_buffer(&consumer, 0, -1, 1);
  VkMemoryGetFdInfoKHR get_fd = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
      .memory = shared.memory,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
  int fd = -1;
  CHECK(producer.GetMemoryFdKHR(producer.handle, &get_fd, &fd));
  require(fd >= 0, "DMA-BUF exported fd");
  struct Buffer imported = create_buffer(&consumer, 1, fd, 0);
  fd = -1;
  require(imported.allocation_size <= shared.allocation_size,
          "import fits exporter allocation");
  VkFence exported_fence = fence(&producer, 1),
          imported_fence = fence(&consumer, 0), done = fence(&consumer, 0);
  VkSemaphore exported_sem = semaphore(&producer, 1),
              imported_sem = semaphore(&consumer, 0);
  const uint32_t counts[] = {13, 4096, 127};
  uint32_t total = 0;
  for (uint32_t round = 0; round < 3; round++) {
    uint32_t count = counts[round];
    for (uint32_t i = 0; i < upload.allocation_size / 4; i++)
      upload.mapped[i] = 0xf0f00f0f;
    for (uint32_t i = 0; i < readback.allocation_size / 4; i++)
      readback.mapped[i] = 0xa55aa55a;
    for (uint32_t i = 0; i < count; i++)
      upload.mapped[64 + i] =
          (i * 0x12345u + 0xabcde000u) ^ (round * 0x87654321u);
    begin(&producer);
    if (round)
      barrier(&producer, &shared, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
              VK_QUEUE_FAMILY_EXTERNAL, queue_family,
              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
              VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferCopy copy = {
        .srcOffset = 256, .dstOffset = 512, .size = (VkDeviceSize)count * 4};
    producer.CmdCopyBuffer(producer.command, upload.buffer, shared.buffer, 1,
                           &copy);
    barrier(&producer, &shared, VK_ACCESS_TRANSFER_WRITE_BIT, 0, queue_family,
            VK_QUEUE_FAMILY_EXTERNAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    CHECK(producer.EndCommandBuffer(producer.command));
    VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                           .commandBufferCount = 1,
                           .pCommandBuffers = &producer.command,
                           .signalSemaphoreCount = 1,
                           .pSignalSemaphores = &exported_sem};
    CHECK(producer.QueueSubmit(producer.queue, 1, &submit, exported_fence));
    VkSemaphoreGetFdInfoKHR get_sem = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
        .semaphore = exported_sem,
        .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
    int sem_fd = -2;
    CHECK(producer.GetSemaphoreFdKHR(producer.handle, &get_sem, &sem_fd));
    require(sem_fd >= -1, "semaphore fd is valid or already signalled");
    VkFenceGetFdInfoKHR get_fence = {
        .sType = VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR,
        .fence = exported_fence,
        .handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT};
    int fence_fd = -2;
    CHECK(producer.GetFenceFdKHR(producer.handle, &get_fence, &fence_fd));
    require(fence_fd >= -1, "fence fd is valid or already signalled");
    printf("round %u exported semaphore-fd=%d fence-fd=%d\n", round + 1, sem_fd,
           fence_fd);
    VkImportSemaphoreFdInfoKHR import_sem = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
        .semaphore = imported_sem,
        .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
        .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
        .fd = sem_fd};
    CHECK(consumer.ImportSemaphoreFdKHR(consumer.handle, &import_sem));
    sem_fd = -1;
    VkImportFenceFdInfoKHR import_fence = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_FENCE_FD_INFO_KHR,
        .fence = imported_fence,
        .flags = VK_FENCE_IMPORT_TEMPORARY_BIT,
        .handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT,
        .fd = fence_fd};
    CHECK(consumer.ImportFenceFdKHR(consumer.handle, &import_fence));
    fence_fd = -1;
    begin(&consumer);
    barrier(&consumer, &imported, 0, VK_ACCESS_TRANSFER_READ_BIT,
            VK_QUEUE_FAMILY_EXTERNAL, queue_family,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    copy = (VkBufferCopy){
        .srcOffset = 512, .dstOffset = 1024, .size = (VkDeviceSize)count * 4};
    consumer.CmdCopyBuffer(consumer.command, imported.buffer, readback.buffer,
                           1, &copy);
    barrier(&consumer, &imported, VK_ACCESS_TRANSFER_READ_BIT, 0, queue_family,
            VK_QUEUE_FAMILY_EXTERNAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    barrier(&consumer, &readback, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_HOST_READ_BIT, VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT);
    CHECK(consumer.EndCommandBuffer(consumer.command));
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    submit = (VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                            .waitSemaphoreCount = 1,
                            .pWaitSemaphores = &imported_sem,
                            .pWaitDstStageMask = &wait_stage,
                            .commandBufferCount = 1,
                            .pCommandBuffers = &consumer.command};
    CHECK(consumer.QueueSubmit(consumer.queue, 1, &submit, done));
    CHECK(consumer.WaitForFences(consumer.handle, 1, &done, VK_TRUE,
                                 UINT64_C(10000000000)));
    CHECK(consumer.WaitForFences(consumer.handle, 1, &imported_fence, VK_TRUE,
                                 UINT64_C(10000000000)));
    CHECK(consumer.GetFenceStatus(consumer.handle, imported_fence));
    for (uint32_t i = 0; i < readback.allocation_size / 4; i++) {
      uint32_t expected =
          i >= 256 && i < 256 + count
              ? (((i - 256) * 0x12345u + 0xabcde000u) ^ (round * 0x87654321u))
              : 0xa55aa55a;
      require(readback.mapped[i] == expected,
              "cross-device GPU-copy bytes and guards");
    }
    CHECK(consumer.ResetFences(consumer.handle, 1, &done));
    CHECK(consumer.ResetFences(consumer.handle, 1, &imported_fence));
    CHECK(producer.ResetFences(producer.handle, 1, &exported_fence));
    total += count;
    printf("round %u: %u words transferred through DMA-BUF; semaphore/fence "
           "SYNC_FD imported; guards intact\n",
           round + 1, count);
  }
  CHECK(producer.DeviceWaitIdle(producer.handle));
  CHECK(consumer.DeviceWaitIdle(consumer.handle));
  maps_check(allow_stock, expect);
  producer.DestroyFence(producer.handle, exported_fence, NULL);
  consumer.DestroyFence(consumer.handle, imported_fence, NULL);
  consumer.DestroyFence(consumer.handle, done, NULL);
  producer.DestroySemaphore(producer.handle, exported_sem, NULL);
  consumer.DestroySemaphore(consumer.handle, imported_sem, NULL);
  destroy_buffer(&producer, &upload);
  destroy_buffer(&producer, &shared);
  destroy_buffer(&consumer, &readback);
  destroy_buffer(&consumer, &imported);
  producer.DestroyCommandPool(producer.handle, producer.pool, NULL);
  consumer.DestroyCommandPool(consumer.handle, consumer.pool, NULL);
  producer.DestroyDevice(producer.handle, NULL);
  consumer.DestroyDevice(consumer.handle, NULL);
  DestroyInstance(instance, NULL);
  nv_probe_android_logger_close();
  require(dlclose(library) == 0, "dlclose");
  int after = fd_count();
  printf("file descriptors %d -> %d\n", before, after);
  if (before != after) nv_probe_dump_fds();
  if (allow_stock || nv_expect_driver_fds_checked(expect))
    require(before == after, "fd leak");
  else if (before != after)
    printf("file descriptors held by the driver, which stays loaded (glibc-shim): listed above\n");
  printf("PASS: %u cross-device GPU-copy words, DMA-BUF ownership and SYNC_FD "
         "semaphore/fence round trips; no image presentation tested\n",
         total);
  return 0;
}
