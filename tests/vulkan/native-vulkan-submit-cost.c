/* CPU cost of the per-submit path: vkQueueSubmit of one small command buffer with one already-signalled
 * timeline wait and one timeline signal, then vkWaitSemaphores on a reached value and vkMapMemory/vkUnmapMemory.
 * Reports CPU time of the calling thread (CLOCK_THREAD_CPUTIME_ID) and of the whole process (getrusage: includes
 * Mesa's submit thread and the RM's threads) per operation, and wall time. No driver-private ABI.
 *
 *   submit-cost-<target> --driver /abs/libvulkan_nouveau.so [--count N]
 *
 * Numbers are only comparable between runs on the same idle GPU (no other GPU clients). Built by
 * tests/vulkan/build.py.
 */
#define _POSIX_C_SOURCE 200809L
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#define INSTANCE_FUNCTIONS(X) \
 X(DestroyInstance) X(EnumeratePhysicalDevices) X(GetPhysicalDeviceProperties) \
 X(GetPhysicalDeviceQueueFamilyProperties) X(GetPhysicalDeviceMemoryProperties) \
 X(CreateDevice) X(GetDeviceProcAddr)
#define DEVICE_FUNCTIONS(X) \
 X(DestroyDevice) X(GetDeviceQueue) X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) \
 X(AllocateMemory) X(FreeMemory) X(BindBufferMemory) X(MapMemory) X(UnmapMemory) \
 X(CreateCommandPool) X(DestroyCommandPool) X(AllocateCommandBuffers) X(BeginCommandBuffer) \
 X(EndCommandBuffer) X(CmdFillBuffer) X(CreateSemaphore) X(DestroySemaphore) X(SignalSemaphore) \
 X(WaitSemaphores) X(QueueSubmit) X(DeviceWaitIdle)
#define DECLARE(name) static PFN_vk##name p##name;
INSTANCE_FUNCTIONS(DECLARE)
DEVICE_FUNCTIONS(DECLARE)
#undef DECLARE

_Noreturn static void die(const char *message) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
static void check_result(VkResult result, const char *operation, int line) {
    if (result != VK_SUCCESS) {
        fprintf(stderr, "FAIL: line %d %s returned %d\n", line, operation, result);
        exit(1);
    }
}
#define VK_CHECK(call) check_result((call), #call, __LINE__)

static VkDevice device;

/* Thread CPU, process CPU and wall time, in ns. */
struct Clock { uint64_t thread, process, wall; };
static uint64_t ns(struct timespec t) { return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec; }
static uint64_t tv_ns(struct timeval t) { return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_usec * 1000u; }
static struct Clock now(void) {
    struct timespec thread, wall;
    struct rusage usage;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &thread) || clock_gettime(CLOCK_MONOTONIC, &wall) ||
        getrusage(RUSAGE_SELF, &usage))
        die("clock");
    return (struct Clock){ns(thread), tv_ns(usage.ru_utime) + tv_ns(usage.ru_stime), ns(wall)};
}
static void report(const char *what, struct Clock start, uint64_t count) {
    struct Clock end = now();
    printf("%-12s count=%" PRIu64 " thread_cpu_ns_per_op=%.1f process_cpu_ns_per_op=%.1f wall_ns_per_op=%.1f\n",
           what, count, (double)(end.thread - start.thread) / (double)count,
           (double)(end.process - start.process) / (double)count, (double)(end.wall - start.wall) / (double)count);
}

static VkSemaphore timeline(uint64_t initial) {
    VkSemaphoreTypeCreateInfo type = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = initial};
    VkSemaphoreCreateInfo info = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &type};
    VkSemaphore semaphore;
    VK_CHECK(pCreateSemaphore(device, &info, NULL, &semaphore));
    return semaphore;
}
static void wait_value(VkSemaphore semaphore, uint64_t value) {
    VkSemaphoreWaitInfo info = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1,
                                .pSemaphores = &semaphore, .pValues = &value};
    VK_CHECK(pWaitSemaphores(device, &info, UINT64_MAX));
}

int main(int argc, char **argv) {
    const char *driver = NULL;
    uint64_t count = 200000;
    for (int i = 1; i < argc; i++) {
        if (i + 1 < argc && !strcmp(argv[i], "--driver")) driver = argv[++i];
        else if (i + 1 < argc && !strcmp(argv[i], "--count")) count = strtoull(argv[++i], NULL, 10);
        else die("usage: submit-cost --driver /absolute/libvulkan_nouveau.so [--count N]");
    }
    if (!driver || driver[0] != '/' || count == 0) die("provide an absolute --driver path and a positive --count");
    void *library = dlopen(driver, RTLD_NOW | RTLD_LOCAL);
    if (!library) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    PFN_vkGetInstanceProcAddr get_instance =
        (PFN_vkGetInstanceProcAddr)dlsym(library, "vk_icdGetInstanceProcAddr");
    if (!get_instance) die("driver lacks vk_icdGetInstanceProcAddr");
    PFN_vkCreateInstance create_instance = (PFN_vkCreateInstance)get_instance(VK_NULL_HANDLE, "vkCreateInstance");
    if (!create_instance) die("cannot resolve vkCreateInstance");
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "submit cost",
                             .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo instance_info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance instance;
    VK_CHECK(create_instance(&instance_info, NULL, &instance));
#define LOAD(name) p##name = (PFN_vk##name)get_instance(instance, "vk" #name); if (!p##name) die("resolve vk" #name);
    INSTANCE_FUNCTIONS(LOAD)
#undef LOAD
    uint32_t physical_count = 8;
    VkPhysicalDevice physicals[8];
    VkResult enumerated = pEnumeratePhysicalDevices(instance, &physical_count, physicals);
    if (enumerated != VK_SUCCESS && enumerated != VK_INCOMPLETE) die("vkEnumeratePhysicalDevices");
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties;
    for (uint32_t i = 0; i < physical_count && !physical; i++) {
        pGetPhysicalDeviceProperties(physicals[i], &properties);
        if (properties.vendorID == 0x10de) physical = physicals[i];
    }
    if (!physical) die("no NVIDIA physical device");
    printf("device: %s (%04x:%04x)\n", properties.deviceName, properties.vendorID, properties.deviceID);
    uint32_t family_count = 16;
    VkQueueFamilyProperties families[16];
    pGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families);
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < family_count && family == UINT32_MAX; i++)
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) family = i;
    if (family == UINT32_MAX) die("no graphics queue family");
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                          .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority};
    VkPhysicalDeviceVulkan12Features features12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                                    .timelineSemaphore = VK_TRUE};
    VkDeviceCreateInfo device_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &features12,
                                      .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info};
    VK_CHECK(pCreateDevice(physical, &device_info, NULL, &device));
#define LOAD(name) p##name = (PFN_vk##name)pGetDeviceProcAddr(device, "vk" #name); if (!p##name) die("resolve vk" #name);
    DEVICE_FUNCTIONS(LOAD)
#undef LOAD
    VkQueue queue;
    pGetDeviceQueue(device, family, 0, &queue);

    /* A 4 KiB host-visible buffer: the fill target and the map/unmap object. */
    VkBufferCreateInfo buffer_info = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4096,
                                      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer buffer;
    VK_CHECK(pCreateBuffer(device, &buffer_info, NULL, &buffer));
    VkMemoryRequirements requirements;
    pGetBufferMemoryRequirements(device, buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory;
    pGetPhysicalDeviceMemoryProperties(physical, &memory);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < memory.memoryTypeCount && type == UINT32_MAX; i++)
        if ((requirements.memoryTypeBits & (1u << i)) &&
            (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
            type = i;
    if (type == UINT32_MAX) die("no host-visible memory type");
    VkMemoryAllocateInfo allocation = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                       .allocationSize = requirements.size, .memoryTypeIndex = type};
    VkDeviceMemory buffer_memory;
    VK_CHECK(pAllocateMemory(device, &allocation, NULL, &buffer_memory));
    VK_CHECK(pBindBufferMemory(device, buffer, buffer_memory, 0));

    VkCommandPoolCreateInfo pool_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                         .queueFamilyIndex = family};
    VkCommandPool pool;
    VK_CHECK(pCreateCommandPool(device, &pool_info, NULL, &pool));
    VkCommandBufferAllocateInfo command_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                                .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                .commandBufferCount = 1};
    VkCommandBuffer command;
    VK_CHECK(pAllocateCommandBuffers(device, &command_info, &command));
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                      .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT};
    VK_CHECK(pBeginCommandBuffer(command, &begin));
    pCmdFillBuffer(command, buffer, 0, 4, 0x5eed);
    VK_CHECK(pEndCommandBuffer(command));

    VkSemaphore ready = timeline(1), done = timeline(0);
    uint64_t wait_value_one = 1, signal_value = 0;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkTimelineSemaphoreSubmitInfo values = {.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                                            .waitSemaphoreValueCount = 1, .pWaitSemaphoreValues = &wait_value_one,
                                            .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &signal_value};
    VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &values, .waitSemaphoreCount = 1,
                           .pWaitSemaphores = &ready, .pWaitDstStageMask = &stage, .commandBufferCount = 1,
                           .pCommandBuffers = &command, .signalSemaphoreCount = 1, .pSignalSemaphores = &done};
    /* The queue is drained every this many submissions, so the measurement never depends on ring depth. */
    const uint64_t window = 256;
    for (int phase = 0; phase < 2; phase++) {
        uint64_t total = phase ? count : 1000; /* phase 0 warms caches and the backend's lists */
        struct Clock start = now();
        for (uint64_t i = 0; i < total; i++) {
            signal_value++;
            VK_CHECK(pQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
            if (signal_value % window == 0) wait_value(done, signal_value);
        }
        wait_value(done, signal_value);
        if (phase) report("submit", start, total);
    }
    struct Clock start = now();
    for (uint64_t i = 0; i < count; i++) wait_value(done, signal_value);
    report("wait-reached", start, count);
    start = now();
    for (uint64_t i = 0; i < count; i++) {
        void *mapped = NULL;
        VK_CHECK(pMapMemory(device, buffer_memory, 0, VK_WHOLE_SIZE, 0, &mapped));
        if (!mapped) die("vkMapMemory returned NULL");
        pUnmapMemory(device, buffer_memory);
    }
    report("map-unmap", start, count);

    VK_CHECK(pDeviceWaitIdle(device));
    pDestroySemaphore(device, ready, NULL);
    pDestroySemaphore(device, done, NULL);
    pDestroyCommandPool(device, pool, NULL);
    pDestroyBuffer(device, buffer, NULL);
    pFreeMemory(device, buffer_memory, NULL);
    pDestroyDevice(device, NULL);
    pDestroyInstance(instance, NULL);
    if (dlclose(library)) die("dlclose");
    puts("submit cost: PASS");
    return 0;
}
