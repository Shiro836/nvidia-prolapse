/* Independent VK_KHR_copy_memory_indirect diagnostic on a compute-only queue family: each iteration records
 * vkCmdCopyMemoryIndirectKHR (two commands, source -> destination), a pipeline barrier (transfer write ->
 * shader read) and a compute dispatch that reads the copied words, then checks every output word. The
 * dispatch consumes the destination on the GPU right after the barrier, without a host readback in
 * between. No driver-private ABI. Built and run by tests/vulkan/build.py and run.py (README.md there).
 *   copy-indirect --driver /absolute/libvulkan.so --transform transform.spv [--iterations N]
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

#define INSTANCE_FUNCTIONS(X) \
 X(DestroyInstance) X(EnumeratePhysicalDevices) X(GetPhysicalDeviceProperties) \
 X(GetPhysicalDeviceQueueFamilyProperties) X(GetPhysicalDeviceMemoryProperties) \
 X(GetPhysicalDeviceFeatures2) X(CreateDevice) X(GetDeviceProcAddr)
#define DEVICE_FUNCTIONS(X) \
 X(DestroyDevice) X(GetDeviceQueue) X(CreateBuffer) X(DestroyBuffer) \
 X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory) X(BindBufferMemory) \
 X(MapMemory) X(GetBufferDeviceAddress) X(CreateShaderModule) X(DestroyShaderModule) \
 X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) \
 X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateComputePipelines) X(DestroyPipeline) \
 X(CreateDescriptorPool) X(DestroyDescriptorPool) X(AllocateDescriptorSets) X(UpdateDescriptorSets) \
 X(CreateCommandPool) X(DestroyCommandPool) X(AllocateCommandBuffers) X(ResetCommandBuffer) \
 X(BeginCommandBuffer) X(EndCommandBuffer) X(CmdBindPipeline) X(CmdBindDescriptorSets) \
 X(CmdPushConstants) X(CmdDispatch) X(CmdPipelineBarrier) X(CmdCopyMemoryIndirectKHR) \
 X(CreateFence) X(DestroyFence) X(WaitForFences) X(ResetFences) X(QueueSubmit) X(DeviceWaitIdle)
#define DECLARE(name) static PFN_vk##name p##name;
INSTANCE_FUNCTIONS(DECLARE)
DEVICE_FUNCTIONS(DECLARE)
#undef DECLARE

#define WORDS 4096u
static void die(const char *message) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
static void check_result(VkResult result, const char *operation, int line) {
    if (result != VK_SUCCESS) {
        fprintf(stderr, "FAIL: line %d %s returned %d\n", line, operation, result);
        exit(1);
    }
}
#define VK_CHECK(call) check_result((call), #call, __LINE__)
static VkDevice device;
static VkPhysicalDeviceMemoryProperties memory_properties;

struct Buffer { VkBuffer buffer; VkDeviceMemory memory; uint32_t *mapped; VkDeviceAddress address; };
static struct Buffer create_buffer(VkDeviceSize size) {
    struct Buffer b = {0};
    VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VK_CHECK(pCreateBuffer(device, &info, NULL, &b.buffer));
    VkMemoryRequirements requirements;
    pGetBufferMemoryRequirements(device, b.buffer, &requirements);
    uint32_t type = UINT32_MAX;
    const VkMemoryPropertyFlags needed = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
        if ((requirements.memoryTypeBits & (1u << i)) &&
            (memory_properties.memoryTypes[i].propertyFlags & needed) == needed) { type = i; break; }
    if (type == UINT32_MAX) die("no HOST_VISIBLE | HOST_COHERENT memory type");
    VkMemoryAllocateFlagsInfo flags = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                       .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo allocation = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &flags,
                                       .allocationSize = requirements.size, .memoryTypeIndex = type};
    VK_CHECK(pAllocateMemory(device, &allocation, NULL, &b.memory));
    VK_CHECK(pBindBufferMemory(device, b.buffer, b.memory, 0));
    void *mapped = NULL;
    VK_CHECK(pMapMemory(device, b.memory, 0, size, 0, &mapped));
    b.mapped = mapped;
    VkBufferDeviceAddressInfo address = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b.buffer};
    b.address = pGetBufferDeviceAddress(device, &address);
    if (!b.address) die("vkGetBufferDeviceAddress returned 0");
    return b;
}
static void destroy_buffer(struct Buffer *b) {
    pDestroyBuffer(device, b->buffer, NULL);
    pFreeMemory(device, b->memory, NULL);
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
    VkShaderModuleCreateInfo info = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                     .codeSize = (size_t)length, .pCode = code};
    VkShaderModule shader = VK_NULL_HANDLE;
    VK_CHECK(pCreateShaderModule(device, &info, NULL, &shader));
    free(code);
    return shader;
}
struct Parameters { uint32_t count, multiplier, addend, xor_mask; };

int main(int argc, char **argv) {
    const char *driver = NULL, *transform_path = NULL;
    uint32_t iterations = 1000;
    for (int i = 1; i < argc; i++) {
        if (i + 1 < argc && !strcmp(argv[i], "--driver")) driver = argv[++i];
        else if (i + 1 < argc && !strcmp(argv[i], "--transform")) transform_path = argv[++i];
        else if (i + 1 < argc && !strcmp(argv[i], "--iterations")) iterations = (uint32_t)strtoul(argv[++i], NULL, 0);
        else die("usage: copy-indirect --driver /absolute/library.so --transform transform.spv [--iterations N]");
    }
    if (!driver || driver[0] != '/' || !transform_path || !iterations) die("provide an absolute --driver and --transform");
    void *library = dlopen(driver, RTLD_NOW | RTLD_LOCAL);
    if (!library) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    PFN_vkGetInstanceProcAddr get_instance = (PFN_vkGetInstanceProcAddr)dlsym(library, "vk_icdGetInstanceProcAddr");
    if (!get_instance) die("library lacks vk_icdGetInstanceProcAddr");
    PFN_vkCreateInstance create_instance = (PFN_vkCreateInstance)get_instance(VK_NULL_HANDLE, "vkCreateInstance");
    if (!create_instance) die("cannot resolve vkCreateInstance");
    VkApplicationInfo application = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "copy-memory-indirect consumer probe", .apiVersion = VK_API_VERSION_1_3};
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
    VkPhysicalDeviceProperties properties;
    for (uint32_t i = 0; i < device_count; i++) {
        pGetPhysicalDeviceProperties(physical_devices[i], &properties);
        if (properties.vendorID == 0x10de) { physical = physical_devices[i]; break; }
    }
    if (physical == VK_NULL_HANDLE) die("no NVIDIA physical device (vendor 0x10de)");
    printf("device: %.256s\n", properties.deviceName);
    pGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    uint32_t family_count = 0;
    pGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, NULL);
    if (!family_count || family_count > 64) die("invalid queue family count");
    VkQueueFamilyProperties families[64];
    pGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families);
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < family_count; i++)
        if (families[i].queueCount && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { family = i; break; }
    if (family == UINT32_MAX) {
        puts("SKIP: no compute-only queue family");
        return 77;
    }
    VkPhysicalDeviceCopyMemoryIndirectFeaturesKHR copy_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_FEATURES_KHR};
    VkPhysicalDeviceVulkan12Features features12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                                   .pNext = &copy_features};
    VkPhysicalDeviceFeatures2 features = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &features12};
    pGetPhysicalDeviceFeatures2(physical, &features);
    if (!copy_features.indirectMemoryCopy || !features12.bufferDeviceAddress)
        die("indirectMemoryCopy or bufferDeviceAddress unsupported");
    VkPhysicalDeviceCopyMemoryIndirectFeaturesKHR enable_copy = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COPY_MEMORY_INDIRECT_FEATURES_KHR, .indirectMemoryCopy = VK_TRUE};
    VkPhysicalDeviceVulkan12Features enable12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                                 .pNext = &enable_copy, .bufferDeviceAddress = VK_TRUE};
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority};
    const char *extension = "VK_KHR_copy_memory_indirect";
    VkDeviceCreateInfo device_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &enable12,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info,
        .enabledExtensionCount = 1, .ppEnabledExtensionNames = &extension};
    VK_CHECK(pCreateDevice(physical, &device_info, NULL, &device));
#define LOAD_DEVICE(name) p##name = (PFN_vk##name)pGetDeviceProcAddr(device, "vk" #name); if (!p##name) die("missing vk" #name);
    DEVICE_FUNCTIONS(LOAD_DEVICE)
#undef LOAD_DEVICE
    VkQueue queue;
    pGetDeviceQueue(device, family, 0, &queue);

    struct Buffer source = create_buffer(WORDS * 4), copied = create_buffer(WORDS * 4),
                  output = create_buffer(WORDS * 4), commands = create_buffer(4096);
    VkShaderModule shader = load_shader(transform_path);
    VkDescriptorSetLayoutBinding bindings[2] = {
        {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}};
    VkDescriptorSetLayoutCreateInfo set_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                .bindingCount = 2, .pBindings = bindings};
    VkDescriptorSetLayout set_layout;
    VK_CHECK(pCreateDescriptorSetLayout(device, &set_info, NULL, &set_layout));
    VkPushConstantRange range = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof(struct Parameters)};
    VkPipelineLayoutCreateInfo layout_info = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &range};
    VkPipelineLayout layout;
    VK_CHECK(pCreatePipelineLayout(device, &layout_info, NULL, &layout));
    VkComputePipelineCreateInfo pipeline_info = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main"},
        .layout = layout};
    VkPipeline pipeline;
    VK_CHECK(pCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline));
    VkDescriptorPoolSize pool_size = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2};
    VkDescriptorPoolCreateInfo pool_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                            .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size};
    VkDescriptorPool descriptor_pool;
    VK_CHECK(pCreateDescriptorPool(device, &pool_info, NULL, &descriptor_pool));
    VkDescriptorSetAllocateInfo set_alloc = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool, .descriptorSetCount = 1, .pSetLayouts = &set_layout};
    VkDescriptorSet set;
    VK_CHECK(pAllocateDescriptorSets(device, &set_alloc, &set));
    VkDescriptorBufferInfo infos[2] = {{copied.buffer, 0, WORDS * 4}, {output.buffer, 0, WORDS * 4}};
    VkWriteDescriptorSet writes[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &infos[1]}};
    pUpdateDescriptorSets(device, 2, writes, 0, NULL);

    /* Two commands: the lower and the upper half of the buffer */
    VkCopyMemoryIndirectCommandKHR *c = (VkCopyMemoryIndirectCommandKHR *)commands.mapped;
    for (uint32_t i = 0; i < 2; i++) {
        c[i].srcAddress = source.address + i * (WORDS / 2) * 4;
        c[i].dstAddress = copied.address + i * (WORDS / 2) * 4;
        c[i].size = (WORDS / 2) * 4;
    }
    VkCommandPoolCreateInfo command_pool_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = family};
    VkCommandPool command_pool;
    VK_CHECK(pCreateCommandPool(device, &command_pool_info, NULL, &command_pool));
    VkCommandBufferAllocateInfo command_alloc = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer command;
    VK_CHECK(pAllocateCommandBuffers(device, &command_alloc, &command));
    VkFenceCreateInfo fence_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    VK_CHECK(pCreateFence(device, &fence_info, NULL, &fence));

    for (uint32_t it = 0; it < iterations; it++) {
        for (uint32_t i = 0; i < WORDS; i++) {
            source.mapped[i] = i * 2654435761u + it;
            copied.mapped[i] = 0xdeadbeef;
            output.mapped[i] = 0xcafef00d;
        }
        const struct Parameters parameters = {WORDS, 3 + it % 7, it, 0x5a5a0000u ^ it};
        VK_CHECK(pResetCommandBuffer(command, 0));
        VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                          .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        VK_CHECK(pBeginCommandBuffer(command, &begin));
        VkCopyMemoryIndirectInfoKHR copy = {.sType = VK_STRUCTURE_TYPE_COPY_MEMORY_INDIRECT_INFO_KHR,
            .copyCount = 2,
            .copyAddressRange = {commands.address, 2 * sizeof(*c), sizeof(*c)}};
        pCmdCopyMemoryIndirectKHR(command, &copy);
        VkMemoryBarrier memory_barrier = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
        pCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                            1, &memory_barrier, 0, NULL, 0, NULL);
        pCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        pCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
        pCmdPushConstants(command, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(parameters), &parameters);
        pCmdDispatch(command, WORDS / 64, 1, 1);
        VkMemoryBarrier host_barrier = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
        pCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                            1, &host_barrier, 0, NULL, 0, NULL);
        VK_CHECK(pEndCommandBuffer(command));
        VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                               .pCommandBuffers = &command};
        VK_CHECK(pQueueSubmit(queue, 1, &submit, fence));
        VK_CHECK(pWaitForFences(device, 1, &fence, VK_TRUE, UINT64_C(10000000000)));
        VK_CHECK(pResetFences(device, 1, &fence));
        for (uint32_t i = 0; i < WORDS; i++) {
            const uint32_t wanted = (source.mapped[i] ^ parameters.xor_mask) + i * parameters.multiplier + parameters.addend;
            if (output.mapped[i] != wanted || copied.mapped[i] != source.mapped[i]) {
                fprintf(stderr, "FAIL: iteration %" PRIu32 " word %" PRIu32 ": output %08x != %08x, copied %08x != %08x\n",
                        it, i, output.mapped[i], wanted, copied.mapped[i], source.mapped[i]);
                return 1;
            }
        }
    }
    printf("copy-indirect: %" PRIu32 " iterations of copy (2 commands) -> barrier -> dispatch reading the copy: all %u words correct\n",
           iterations, WORDS);
    VK_CHECK(pDeviceWaitIdle(device));
    pDestroyFence(device, fence, NULL);
    pDestroyCommandPool(device, command_pool, NULL);
    pDestroyDescriptorPool(device, descriptor_pool, NULL);
    pDestroyPipeline(device, pipeline, NULL);
    pDestroyPipelineLayout(device, layout, NULL);
    pDestroyDescriptorSetLayout(device, set_layout, NULL);
    pDestroyShaderModule(device, shader, NULL);
    destroy_buffer(&source);
    destroy_buffer(&copied);
    destroy_buffer(&output);
    destroy_buffer(&commands);
    pDestroyDevice(device, NULL);
    pDestroyInstance(instance, NULL);
    return 0;
}
