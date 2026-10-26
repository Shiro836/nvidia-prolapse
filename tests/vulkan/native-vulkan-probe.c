/* Independent Vulkan 1.0 compute/copy diagnostic. No driver-private ABI.
 * Built and run by tests/vulkan/build.py and run.py (README.md there).
 */
#define _POSIX_C_SOURCE 200809L
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "native-vulkan-runtime.h"
#include "native-driver-identity.h"
#include <dlfcn.h>
#include <dirent.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INSTANCE_FUNCTIONS(X) \
 X(DestroyInstance) X(EnumeratePhysicalDevices) X(GetPhysicalDeviceProperties) \
 X(GetPhysicalDeviceQueueFamilyProperties) X(GetPhysicalDeviceMemoryProperties) \
 X(CreateDevice) X(GetDeviceProcAddr)
#define DEVICE_FUNCTIONS(X) \
 X(DestroyDevice) X(GetDeviceQueue) X(CreateBuffer) X(DestroyBuffer) \
 X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory) X(BindBufferMemory) \
 X(MapMemory) X(UnmapMemory) X(CreateShaderModule) X(DestroyShaderModule) \
 X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) \
 X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateComputePipelines) X(DestroyPipeline) \
 X(CreateDescriptorPool) X(DestroyDescriptorPool) X(AllocateDescriptorSets) X(UpdateDescriptorSets) \
 X(CreateCommandPool) X(DestroyCommandPool) X(AllocateCommandBuffers) X(ResetCommandBuffer) \
 X(BeginCommandBuffer) X(EndCommandBuffer) X(CmdBindPipeline) X(CmdBindDescriptorSets) \
 X(CmdPushConstants) X(CmdDispatch) X(CmdPipelineBarrier) X(CmdCopyBuffer) \
 X(CreateFence) X(DestroyFence) X(GetFenceStatus) X(WaitForFences) X(ResetFences) \
 X(QueueSubmit) X(QueueWaitIdle) X(DeviceWaitIdle)
#define DECLARE(name) static PFN_vk##name p##name;
INSTANCE_FUNCTIONS(DECLARE)
DEVICE_FUNCTIONS(DECLARE)
#undef DECLARE

static void die(const char *message) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
static int open_fd_count(void) {
    DIR *directory = opendir("/proc/self/fd");
    if (!directory) die("cannot count file descriptors");
    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(directory))) if (entry->d_name[0] != '.') count++;
    if (closedir(directory)) die("cannot close file descriptor directory");
    return count;
}
static void check_result(VkResult result, const char *operation, int line) {
    if (result != VK_SUCCESS) {
        fprintf(stderr, "FAIL: line %d %s returned %d\n", line, operation, result);
        exit(1);
    }
}
#define VK_CHECK(call) check_result((call), #call, __LINE__)
static VkDevice device;
static VkPhysicalDeviceMemoryProperties memory_properties;
static VkPhysicalDeviceProperties properties;

struct Buffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize allocation_size, binding_offset, descriptor_offset;
    uint32_t *mapped;
    uint32_t canary;
};
struct Parameters { uint32_t count, multiplier, addend, xor_mask; };
static VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize alignment) {
    if (!alignment || alignment > (1u << 20)) die("invalid or excessive reported alignment");
    return (value + alignment - 1) / alignment * alignment;
}
static struct Buffer create_buffer(uint32_t canary) {
    struct Buffer buffer = {.canary = canary};
    buffer.descriptor_offset = align_up(64, properties.limits.minStorageBufferOffsetAlignment);
    VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = buffer.descriptor_offset + 1024 * sizeof(uint32_t) + 256,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VK_CHECK(pCreateBuffer(device, &info, NULL, &buffer.buffer));
    VkMemoryRequirements requirements;
    pGetBufferMemoryRequirements(device, buffer.buffer, &requirements);
    if (requirements.size < info.size) die("buffer memory requirement is smaller than buffer");
    buffer.binding_offset = align_up(4096, requirements.alignment);
    buffer.allocation_size = align_up(buffer.binding_offset + requirements.size + 256, 4096);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        VkMemoryPropertyFlags flags = memory_properties.memoryTypes[i].propertyFlags;
        VkMemoryPropertyFlags needed = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((requirements.memoryTypeBits & (1u << i)) && (flags & needed) == needed) { type = i; break; }
    }
    if (type == UINT32_MAX) die("probe requires a HOST_VISIBLE | HOST_COHERENT buffer memory type");
    VkMemoryAllocateInfo allocation = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = buffer.allocation_size, .memoryTypeIndex = type,
    };
    VK_CHECK(pAllocateMemory(device, &allocation, NULL, &buffer.memory));
    VK_CHECK(pBindBufferMemory(device, buffer.buffer, buffer.memory, buffer.binding_offset));
    void *mapped = NULL;
    VK_CHECK(pMapMemory(device, buffer.memory, 0, buffer.allocation_size, 0, &mapped));
    if (!mapped) die("vkMapMemory returned a NULL address");
    buffer.mapped = mapped;
    printf("buffer: allocation=%" PRIu64 " bind-offset=%" PRIu64 " descriptor-offset=%" PRIu64 "\n",
           (uint64_t)buffer.allocation_size, (uint64_t)buffer.binding_offset, (uint64_t)buffer.descriptor_offset);
    return buffer;
}
static void initialize_buffer(struct Buffer *buffer) {
    for (VkDeviceSize i = 0; i < buffer->allocation_size / 4; i++) buffer->mapped[i] = buffer->canary;
}
static void destroy_buffer(struct Buffer *buffer) {
    pUnmapMemory(device, buffer->memory);
    pDestroyBuffer(device, buffer->buffer, NULL);
    pFreeMemory(device, buffer->memory, NULL);
}
static void verify_buffer(const struct Buffer *buffer, const uint32_t *expected, uint32_t count, const char *name) {
    VkDeviceSize first = (buffer->binding_offset + buffer->descriptor_offset) / 4;
    for (VkDeviceSize i = 0; i < buffer->allocation_size / 4; i++) {
        uint32_t wanted = i >= first && i < first + count ? expected[i - first] : buffer->canary;
        if (buffer->mapped[i] != wanted) {
            fprintf(stderr, "FAIL: %s allocation word %" PRIu64 ": %08x != %08x (%s)\n",
                    name, (uint64_t)i, buffer->mapped[i], wanted,
                    i >= first && i < first + count ? "payload" : "canary");
            exit(1);
        }
    }
}
static VkShaderModule load_shader(const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) { perror(path); exit(1); }
    if (fseek(file, 0, SEEK_END) != 0) die("seek shader");
    long length = ftell(file);
    if (length < 20 || length > 16 * 1024 * 1024 || length % 4) die("invalid SPIR-V file length");
    rewind(file);
    uint32_t *code = malloc((size_t)length);
    if (!code) die("allocate shader input");
    if (fread(code, 1, (size_t)length, file) != (size_t)length) die("read shader");
    if (fclose(file)) die("close shader");
    if (code[0] != 0x07230203) die("shader is not little-endian SPIR-V");
    VkShaderModuleCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = (size_t)length, .pCode = code,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    VK_CHECK(pCreateShaderModule(device, &info, NULL, &shader));
    free(code);
    return shader;
}
static VkPipeline create_pipeline(VkShaderModule shader, VkPipelineLayout layout) {
    VkComputePipelineCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main"},
        .layout = layout,
    };
    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(pCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, NULL, &pipeline));
    return pipeline;
}
static void barrier(VkCommandBuffer command, struct Buffer *buffer, uint32_t count,
                    VkAccessFlags source_access, VkAccessFlags destination_access,
                    VkPipelineStageFlags source_stage, VkPipelineStageFlags destination_stage) {
    VkBufferMemoryBarrier memory = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = source_access, .dstAccessMask = destination_access,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buffer->buffer, .offset = buffer->descriptor_offset, .size = (VkDeviceSize)count * 4,
    };
    pCmdPipelineBarrier(command, source_stage, destination_stage, 0, 0, NULL, 1, &memory, 0, NULL);
}
static void verify_runtime(int allow_stock_driver, enum nv_expect_driver expect) {
    char why[4200];
    if (!allow_stock_driver && nv_expect_driver_check(expect, why, sizeof(why))) {
        fprintf(stderr, "%s", why);
        exit(1);
    }
    FILE *file = fopen("/proc/self/maps", "r");
    if (!file) die("cannot inspect runtime library mappings");
    char line[4096];
    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "libc.so") || strstr(line, "libnv_bionic_native") || strstr(line, "libvulkan.so"))
            printf("runtime-map: %s", line);
    }
    if (ferror(file)) die("error reading runtime mappings");
    fclose(file);
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
    const char *driver = NULL, *generator_path = NULL, *transform_path = NULL;
    int loader = 0, allow_stock_driver = 0, negative = 0, icd = 0;
    enum nv_expect_driver expect = NV_EXPECT_PROLAPSE;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--loader")) loader = 1;
        else if (!strcmp(argv[i], "--icd")) icd = 1;
        else if (!strcmp(argv[i], "--negative")) negative = 1;
        else if (!strcmp(argv[i], "--allow-stock-driver")) allow_stock_driver = 1;
        else if (i + 1 < argc && !strcmp(argv[i], "--expect-driver")) {
            if (nv_expect_driver_parse(argv[++i], &expect)) die("--expect-driver is nvidia-prolapse or glibc-shim");
        }
        else if (i + 1 < argc && !strcmp(argv[i], "--driver")) driver = argv[++i];
        else if (i + 1 < argc && !strcmp(argv[i], "--shader")) generator_path = argv[++i];
        else if (i + 1 < argc && !strcmp(argv[i], "--transform")) transform_path = argv[++i];
        else die("usage: PROBE --driver /absolute/library.so --shader generate.spv --transform transform.spv [--loader|--icd] [--expect-driver nvidia-prolapse|glibc-shim] [--allow-stock-driver] [--negative]");
    }
    if (!driver || driver[0] != '/' || !generator_path || !transform_path)
        die("provide an absolute --driver path plus --shader and --transform SPIR-V paths");
    if (loader && !getenv("VK_DRIVER_FILES")) die("--loader requires explicit VK_DRIVER_FILES ICD manifest");
    if (loader && icd) die("select either --loader or --icd");
    nv_probe_android_runtime_init();
    const int descriptors_before = open_fd_count();
    void *library = dlopen(driver, RTLD_NOW | RTLD_LOCAL);
    if (!library) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    const char *entry = icd ? "vk_icdGetInstanceProcAddr" : "vkGetInstanceProcAddr";
    PFN_vkGetInstanceProcAddr get_instance = (PFN_vkGetInstanceProcAddr)dlsym(library, entry);
    if (!get_instance) { fprintf(stderr, "library lacks %s\n", entry); return 1; }
    PFN_vkCreateInstance create_instance = (PFN_vkCreateInstance)get_instance(VK_NULL_HANDLE, "vkCreateInstance");
    if (!create_instance) die("cannot resolve vkCreateInstance");
    VkApplicationInfo application = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Independent source-built compute probe", .apiVersion = VK_API_VERSION_1_0};
    VkInstanceCreateInfo instance_info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &application};
    VkInstance instance = VK_NULL_HANDLE;
    VK_CHECK(create_instance(&instance_info, NULL, &instance));
#define LOAD_INSTANCE(name) p##name = (PFN_vk##name)get_instance(instance, "vk" #name); if (!p##name) die("missing vk" #name);
    INSTANCE_FUNCTIONS(LOAD_INSTANCE)
#undef LOAD_INSTANCE
    uint32_t device_count = 0;
    VK_CHECK(pEnumeratePhysicalDevices(instance, &device_count, NULL));
    if (!device_count || device_count > 64) die("invalid physical device count");
    VkPhysicalDevice physical_devices[64], physical = VK_NULL_HANDLE;
    VK_CHECK(pEnumeratePhysicalDevices(instance, &device_count, physical_devices));
    for (uint32_t i = 0; i < device_count; i++) {
        pGetPhysicalDeviceProperties(physical_devices[i], &properties);
        if (properties.vendorID == 0x10de) { physical = physical_devices[i]; break; }
    }
    if (physical == VK_NULL_HANDLE) die("no NVIDIA physical device (vendor 0x10de)");
    printf("device: %.256s vendor=%04x device=%04x API=%u.%u.%u\n", properties.deviceName,
           properties.vendorID, properties.deviceID, VK_VERSION_MAJOR(properties.apiVersion),
           VK_VERSION_MINOR(properties.apiVersion), VK_VERSION_PATCH(properties.apiVersion));
    pGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    if (memory_properties.memoryTypeCount > VK_MAX_MEMORY_TYPES) die("invalid memory type count");
    uint32_t queue_count = 0;
    pGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, NULL);
    if (!queue_count || queue_count > 64) die("invalid queue family count");
    VkQueueFamilyProperties queues[64];
    pGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, queues);
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < queue_count; i++)
        if (queues[i].queueCount && (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) { family = i; break; }
    if (family == UINT32_MAX) die("no compute queue family");
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority};
    VkDeviceCreateInfo device_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info};
    if (negative) {
        const char *missing_extension = "VK_NV_native_probe_intentionally_missing_extension";
        VkDeviceCreateInfo rejected_info = device_info;
        rejected_info.enabledExtensionCount = 1;
        rejected_info.ppEnabledExtensionNames = &missing_extension;
        VkDevice rejected = VK_NULL_HANDLE;
        VkResult result = pCreateDevice(physical, &rejected_info, NULL, &rejected);
        if (result != VK_ERROR_EXTENSION_NOT_PRESENT) {
            fprintf(stderr, "FAIL: unknown device extension returned %d instead of VK_ERROR_EXTENSION_NOT_PRESENT\n", result);
            return 1;
        }
        puts("negative: unknown device extension rejected with VK_ERROR_EXTENSION_NOT_PRESENT");
        PFN_vkGetPhysicalDeviceFeatures get_features =
            (PFN_vkGetPhysicalDeviceFeatures)get_instance(instance, "vkGetPhysicalDeviceFeatures");
        if (!get_features) die("--negative requires vkGetPhysicalDeviceFeatures");
        VkPhysicalDeviceFeatures supported = {0}, requested = {0};
        get_features(physical, &supported);
        /* VkPhysicalDeviceFeatures consists exclusively of consecutive VkBool32
         * members. memcpy avoids aliasing a struct as an array. Request exactly
         * one member the driver itself reported unsupported. */
        size_t unsupported_offset = sizeof(supported);
        for (size_t offset = 0; offset < sizeof(supported); offset += sizeof(VkBool32)) {
            VkBool32 value;
            memcpy(&value, (const unsigned char *)&supported + offset, sizeof(value));
            if (value == VK_FALSE) { unsupported_offset = offset; break; }
        }
        if (unsupported_offset < sizeof(supported)) {
            VkBool32 enabled = VK_TRUE;
            memcpy((unsigned char *)&requested + unsupported_offset, &enabled, sizeof(enabled));
            rejected_info = device_info;
            rejected_info.pEnabledFeatures = &requested;
            result = pCreateDevice(physical, &rejected_info, NULL, &rejected);
            if (result != VK_ERROR_FEATURE_NOT_PRESENT) {
                fprintf(stderr, "FAIL: unsupported feature at offset %zu returned %d instead of VK_ERROR_FEATURE_NOT_PRESENT\n",
                        unsupported_offset, result);
                return 1;
            }
            printf("negative: unsupported core feature at byte offset %zu rejected with VK_ERROR_FEATURE_NOT_PRESENT\n", unsupported_offset);
        } else puts("negative: all core features reported supported; unsupported-feature check not applicable");
    }
    VK_CHECK(pCreateDevice(physical, &device_info, NULL, &device));
#define LOAD_DEVICE(name) p##name = (PFN_vk##name)pGetDeviceProcAddr(device, "vk" #name); if (!p##name) die("missing vk" #name);
    DEVICE_FUNCTIONS(LOAD_DEVICE)
#undef LOAD_DEVICE
    VkQueue queue = VK_NULL_HANDLE;
    pGetDeviceQueue(device, family, 0, &queue);
    if (queue == VK_NULL_HANDLE) die("vkGetDeviceQueue returned NULL");
    struct Buffer buffers[3] = {create_buffer(0xa55af00d), create_buffer(0xb66b1337), create_buffer(0xc77c9876)};
    VkDescriptorSetLayoutBinding bindings[2];
    memset(bindings, 0, sizeof(bindings));
    for (uint32_t i = 0; i < 2; i++) bindings[i] = (VkDescriptorSetLayoutBinding){
        .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo set_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2, .pBindings = bindings};
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VK_CHECK(pCreateDescriptorSetLayout(device, &set_info, NULL, &set_layout));
    VkPushConstantRange push_range = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof(struct Parameters)};
    VkPipelineLayoutCreateInfo layout_info = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &push_range};
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VK_CHECK(pCreatePipelineLayout(device, &layout_info, NULL, &layout));
    VkShaderModule generator = load_shader(generator_path), transform = load_shader(transform_path);
    VkPipeline pipelines[2] = {create_pipeline(generator, layout), create_pipeline(transform, layout)};
    VkDescriptorPoolSize pool_size = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2};
    VkDescriptorPoolCreateInfo pool_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size};
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VK_CHECK(pCreateDescriptorPool(device, &pool_info, NULL, &descriptor_pool));
    VkDescriptorSetAllocateInfo set_allocate = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool, .descriptorSetCount = 1, .pSetLayouts = &set_layout};
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VK_CHECK(pAllocateDescriptorSets(device, &set_allocate, &descriptor_set));
    VkCommandPoolCreateInfo command_pool_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = family};
    VkCommandPool command_pool = VK_NULL_HANDLE;
    VK_CHECK(pCreateCommandPool(device, &command_pool_info, NULL, &command_pool));
    VkCommandBufferAllocateInfo command_allocate = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer command = VK_NULL_HANDLE;
    VK_CHECK(pAllocateCommandBuffers(device, &command_allocate, &command));
    VkFenceCreateInfo fence_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(pCreateFence(device, &fence_info, NULL, &fence));
    const uint32_t counts[] = {1, 63, 64, 65, 257, 1024};
    uint32_t total = 0;
    for (uint32_t round = 0; round < sizeof(counts) / sizeof(counts[0]); round++) {
        uint32_t count = counts[round];
        struct Buffer *source = &buffers[round % 3];
        struct Buffer *destination = &buffers[(round + 1) % 3];
        struct Buffer *copy = &buffers[(round + 2) % 3];
        for (uint32_t i = 0; i < 3; i++) initialize_buffer(&buffers[i]);
        if (pGetFenceStatus(device, fence) != VK_NOT_READY) die("new/reset fence must be unsignalled");
        if (pWaitForFences(device, 1, &fence, VK_TRUE, 0) != VK_TIMEOUT) die("zero timeout on unsignalled fence must return VK_TIMEOUT");
        VkDescriptorBufferInfo buffer_info[2] = {
            {.buffer = source->buffer, .offset = source->descriptor_offset, .range = (VkDeviceSize)count * 4},
            {.buffer = destination->buffer, .offset = destination->descriptor_offset, .range = (VkDeviceSize)count * 4},
        };
        VkWriteDescriptorSet writes[2];
        for (uint32_t i = 0; i < 2; i++) writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set, .dstBinding = i,
            .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buffer_info[i]};
        pUpdateDescriptorSets(device, 2, writes, 0, NULL);
        struct Parameters generate = {count, 0x12345u + round * 14u, 0xfedcba98u + round * 331u, 0};
        struct Parameters modify = {count, 97u + round * 2u, 0x76543210u - round * 277u, 0xdeadbeefu ^ round * 123u};
        uint32_t source_expected[1024], result_expected[1024];
        for (uint32_t i = 0; i < count; i++) {
            source_expected[i] = i * generate.multiplier + generate.addend;
            result_expected[i] = (source_expected[i] ^ modify.xor_mask) + i * modify.multiplier + modify.addend;
        }
        VK_CHECK(pResetCommandBuffer(command, 0));
        VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        VK_CHECK(pBeginCommandBuffer(command, &begin));
        pCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &descriptor_set, 0, NULL);
        pCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[0]);
        pCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(generate), &generate);
        pCmdDispatch(command, (count + 63) / 64, 1, 1);
        barrier(command, source, count, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        pCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[1]);
        pCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(modify), &modify);
        pCmdDispatch(command, (count + 63) / 64, 1, 1);
        barrier(command, destination, count, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferCopy region = {.srcOffset = destination->descriptor_offset, .dstOffset = copy->descriptor_offset,
            .size = (VkDeviceSize)count * 4};
        pCmdCopyBuffer(command, destination->buffer, copy->buffer, 1, &region);
        barrier(command, source, count, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
        barrier(command, destination, count, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
        barrier(command, copy, count, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
        VK_CHECK(pEndCommandBuffer(command));
        VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &command};
        VK_CHECK(pQueueSubmit(queue, 1, &submit, fence));
        VK_CHECK(pWaitForFences(device, 1, &fence, VK_TRUE, UINT64_C(10000000000)));
        VK_CHECK(pGetFenceStatus(device, fence));
        verify_buffer(source, source_expected, count, "generator");
        verify_buffer(destination, result_expected, count, "transform");
        verify_buffer(copy, result_expected, count, "GPU copy");
        VK_CHECK(pResetFences(device, 1, &fence));
        printf("round %u: count=%u groups=%u generate/transform/copy correct; guards intact; fence reset\n",
               round + 1, count, (count + 63) / 64);
        total += count;
    }
    if (pGetFenceStatus(device, fence) != VK_NOT_READY ||
        pWaitForFences(device, 1, &fence, VK_TRUE, 0) != VK_TIMEOUT) die("final fence reset failed");
    VK_CHECK(pQueueWaitIdle(queue));
    VK_CHECK(pDeviceWaitIdle(device));
    verify_runtime(allow_stock_driver, expect);
    pDestroyFence(device, fence, NULL);
    pDestroyCommandPool(device, command_pool, NULL);
    pDestroyDescriptorPool(device, descriptor_pool, NULL);
    for (uint32_t i = 0; i < 2; i++) pDestroyPipeline(device, pipelines[i], NULL);
    pDestroyShaderModule(device, generator, NULL);
    pDestroyShaderModule(device, transform, NULL);
    pDestroyPipelineLayout(device, layout, NULL);
    pDestroyDescriptorSetLayout(device, set_layout, NULL);
    for (uint32_t i = 0; i < 3; i++) destroy_buffer(&buffers[i]);
    pDestroyDevice(device, NULL);
    pDestroyInstance(instance, NULL);
    nv_probe_android_logger_close();
    if (dlclose(library)) die("dlclose failed");
    const int descriptors_after = open_fd_count();
    printf("file descriptors before/after: %d/%d\n", descriptors_before, descriptors_after);
    if (descriptors_after != descriptors_before) nv_probe_dump_fds();
    if (!nv_expect_driver_fds_checked(expect) && !allow_stock_driver)
        printf("file descriptors held by the driver, which stays loaded (glibc-shim): listed above\n");
    else if (descriptors_after != descriptors_before) die("file descriptor count changed after full Vulkan resource cleanup");
    printf("PASS: Vulkan compute subset, two shaders, six sizes, %u output words checked across three buffers; not Vulkan conformance\n", total * 3);
    return 0;
}
