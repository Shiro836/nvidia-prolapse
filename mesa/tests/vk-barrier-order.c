/* Pipeline-barrier ordering under a tight loop: a slow producer, a pipeline barrier, a fast consumer.
 *
 * Three dependencies, each ROUNDS times per command buffer with a new value per round:
 *   colour: draw (spinning fragment shader) into colour image C, barrier COLOR_ATTACHMENT_OUTPUT /
 *           COLOR_ATTACHMENT_WRITE -> FRAGMENT_SHADER / SHADER_SAMPLED_READ, draw sampling C into C2;
 *           rounds cycle through an image barrier (COLOR_ATTACHMENT_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL),
 *           a memory barrier (C in GENERAL) and an image barrier with VK_DEPENDENCY_BY_REGION_BIT;
 *   compute: dispatch (spinning) writes buffer A, barrier COMPUTE / SHADER_STORAGE_WRITE ->
 *           COMPUTE / SHADER_STORAGE_READ, dispatch reads A and writes A + 1 into buffer O;
 *   cs-gfx: dispatch (spinning) writes A, barrier COMPUTE / SHADER_STORAGE_WRITE -> FRAGMENT_SHADER /
 *           SHADER_STORAGE_READ, draw reading A per pixel into C2.
 * Each consumer result is copied to a host buffer and compared with the round's value. A consumer that
 * is not ordered behind its producer reads the previous round's data. PASS = 0 mismatches.
 *
 * Usage: vk-barrier-order [ITERATIONS [SPIN [unordered]]]   (default 50, 2000). "unordered" is the
 * negative control: the three producer -> consumer barriers become TOP_OF_PIPE -> BOTTOM_OF_PIPE
 * without accesses (layouts still change), so mismatches are expected. GPU picked by vendor 0x10de;
 * driver chosen with VK_DRIVER_FILES. Build/run: tests/README.md "vk-barrier-order".
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "vk-barrier-order.quad.h"
#include "vk-barrier-order.fill.h"
#include "vk-barrier-order.sample.h"
#include "vk-barrier-order.ssbo.h"
#include "vk-barrier-order.produce.h"
#include "vk-barrier-order.consume.h"

#define VKCHECK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { \
   fprintf(stderr, "FATAL %s:%d %s -> %d\n", __FILE__, __LINE__, #call, r_); exit(2); } } while (0)

#define W 256
#define H 256
#define N (W * H)
#define ROUNDS 12

struct pc { uint32_t value, spin; };

static VkDevice dev;
static VkPhysicalDevice pd;

static uint32_t
mem_type(uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   fprintf(stderr, "no memory type 0x%x for bits 0x%x\n", want, bits);
   exit(2);
}

static VkImage
make_image(VkImageUsageFlags usage)
{
   VkImageCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R32_UINT, .extent = { W, H, 1 }, .mipLevels = 1,
      .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   VkImage img;
   VKCHECK(vkCreateImage(dev, &ici, NULL, &img));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(dev, img, &mr);
   VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
      .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) };
   VkDeviceMemory mem;
   VKCHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
   VKCHECK(vkBindImageMemory(dev, img, mem, 0));
   return img;
}

static VkImageView
make_view(VkImage img)
{
   VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R32_UINT,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   VkImageView v;
   VKCHECK(vkCreateImageView(dev, &vci, NULL, &v));
   return v;
}

static VkBuffer
make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, void **map)
{
   VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage };
   VkBuffer b;
   VKCHECK(vkCreateBuffer(dev, &bci, NULL, &b));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, b, &mr);
   VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
      .memoryTypeIndex = mem_type(mr.memoryTypeBits, props) };
   VkDeviceMemory mem;
   VKCHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
   VKCHECK(vkBindBufferMemory(dev, b, mem, 0));
   if (map)
      VKCHECK(vkMapMemory(dev, mem, 0, size, 0, map));
   return b;
}

static VkShaderModule
module(const uint32_t *code, size_t size)
{
   VkShaderModuleCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size, .pCode = code };
   VkShaderModule m;
   VKCHECK(vkCreateShaderModule(dev, &ci, NULL, &m));
   return m;
}

static void
img_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 ss,
            VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da, VkDependencyFlags flags)
{
   VkImageMemoryBarrier2 b = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = ss, .srcAccessMask = sa, .dstStageMask = ds, .dstAccessMask = da,
      .oldLayout = from, .newLayout = to, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   VkDependencyInfo di = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .dependencyFlags = flags,
      .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b };
   vkCmdPipelineBarrier2(cb, &di);
}

static void
mem_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds,
            VkAccessFlags2 da, VkDependencyFlags flags)
{
   VkMemoryBarrier2 b = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = ss, .srcAccessMask = sa, .dstStageMask = ds, .dstAccessMask = da };
   VkDependencyInfo di = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .dependencyFlags = flags,
      .memoryBarrierCount = 1, .pMemoryBarriers = &b };
   vkCmdPipelineBarrier2(cb, &di);
}

static VkPipelineLayout
make_layout(VkDescriptorSetLayout *dsl, const VkDescriptorSetLayoutBinding *b, uint32_t n)
{
   VkDescriptorSetLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = n, .pBindings = b };
   VKCHECK(vkCreateDescriptorSetLayout(dev, &lci, NULL, dsl));
   VkPushConstantRange pcr = { VK_SHADER_STAGE_ALL, 0, sizeof(struct pc) };
   VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
   VkPipelineLayout l;
   VKCHECK(vkCreatePipelineLayout(dev, &plci, NULL, &l));
   return l;
}

static VkPipeline
gfx_pipeline(VkPipelineLayout layout, VkShaderModule vs, VkShaderModule fs)
{
   VkPipelineShaderStageCreateInfo st[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fs, .pName = "main" } };
   VkPipelineVertexInputStateCreateInfo vi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
   VkPipelineInputAssemblyStateCreateInfo ia = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP };
   VkViewport vp = { 0, 0, W, H, 0.0f, 1.0f };
   VkRect2D sc = { { 0, 0 }, { W, H } };
   VkPipelineViewportStateCreateInfo vps = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc };
   VkPipelineRasterizationStateCreateInfo rs = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1.0f };
   VkPipelineMultisampleStateCreateInfo ms = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
   VkPipelineColorBlendStateCreateInfo cbs = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba };
   VkFormat f = VK_FORMAT_R32_UINT;
   VkPipelineRenderingCreateInfo prci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &f };
   VkGraphicsPipelineCreateInfo gpci = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &prci, .stageCount = 2, .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs, .pMultisampleState = &ms, .pColorBlendState = &cbs,
      .layout = layout };
   VkPipeline p;
   VKCHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &p));
   return p;
}

static VkPipeline
cs_pipeline(VkPipelineLayout layout, VkShaderModule cs)
{
   VkComputePipelineCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                 .module = cs, .pName = "main" }, .layout = layout };
   VkPipeline p;
   VKCHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, &p));
   return p;
}

static void
render(VkCommandBuffer cb, VkImageView view, VkImageLayout layout)
{
   VkRenderingAttachmentInfo ca = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = view, .imageLayout = layout, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE };
   VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = { { 0, 0 }, { W, H } },
      .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca };
   vkCmdBeginRendering(cb, &ri);
   vkCmdDraw(cb, 4, 1, 0, 0);
   vkCmdEndRendering(cb);
}

int
main(int argc, char **argv)
{
   int iterations = argc > 1 ? atoi(argv[1]) : 50;
   uint32_t spin = argc > 2 ? (uint32_t)atoi(argv[2]) : 2000;
   int unordered = argc > 3 && !strcmp(argv[3], "unordered");
   if (iterations <= 0) { fprintf(stderr, "usage: %s [ITERATIONS [SPIN]]\n", argv[0]); return 2; }

   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vk-barrier-order", .apiVersion = VK_API_VERSION_1_3 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
   VkInstance inst;
   VKCHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 0;
   VKCHECK(vkEnumeratePhysicalDevices(inst, &n, NULL));
   VkPhysicalDevice pds[16];
   if (n > 16) n = 16;
   VKCHECK(vkEnumeratePhysicalDevices(inst, &n, pds));
   VkPhysicalDeviceDriverProperties drv = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
   for (uint32_t i = 0; i < n && !pd; i++) {
      VkPhysicalDeviceProperties2 p = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &drv };
      vkGetPhysicalDeviceProperties2(pds[i], &p);
      if (p.properties.vendorID == 0x10de)
         pd = pds[i];
   }
   if (!pd) { fprintf(stderr, "no NVIDIA (0x10de) Vulkan device\n"); return 2; }
   printf("driver: %s (%s)\n", drv.driverName, drv.driverInfo);

   uint32_t qn = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, NULL);
   VkQueueFamilyProperties qp[16];
   if (qn > 16) qn = 16;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < qn && qf == UINT32_MAX; i++)
      if ((qp[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
          (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
         qf = i;
   if (qf == UINT32_MAX) { fprintf(stderr, "no graphics+compute queue\n"); return 2; }
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio };
   VkPhysicalDeviceVulkan13Features f13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .dynamicRendering = VK_TRUE, .synchronization2 = VK_TRUE };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f13,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
   VKCHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);

   VkImage c = make_image(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
   VkImage c2 = make_image(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
   VkImageView cv = make_view(c), c2v = make_view(c2);
   const VkDeviceSize bsize = (VkDeviceSize)N * 4, slots = 3 * ROUNDS;
   VkBuffer a = make_buffer(bsize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, NULL);
   VkBuffer o = make_buffer(bsize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, NULL);
   uint32_t *rb_map;
   VkBuffer rb = make_buffer(slots * bsize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                             (void **)&rb_map);

   VkSamplerCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE };
   VkSampler smp;
   VKCHECK(vkCreateSampler(dev, &sci, NULL, &smp));

   VkDescriptorSetLayout dsl[4];
   VkDescriptorSetLayoutBinding b_tex = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL };
   VkDescriptorSetLayoutBinding b_one = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, NULL };
   VkDescriptorSetLayoutBinding b_two[2] = { { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, NULL },
                                             { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, NULL } };
   VkPipelineLayout l_fill = make_layout(&dsl[0], NULL, 0);
   VkPipelineLayout l_tex = make_layout(&dsl[1], &b_tex, 1);
   VkPipelineLayout l_one = make_layout(&dsl[2], &b_one, 1);
   VkPipelineLayout l_two = make_layout(&dsl[3], b_two, 2);

   VkShaderModule m_quad = module(vk_barrier_order_quad, sizeof(vk_barrier_order_quad));
   VkPipeline p_fill = gfx_pipeline(l_fill, m_quad, module(vk_barrier_order_fill, sizeof(vk_barrier_order_fill)));
   VkPipeline p_sample = gfx_pipeline(l_tex, m_quad, module(vk_barrier_order_sample, sizeof(vk_barrier_order_sample)));
   VkPipeline p_ssbo = gfx_pipeline(l_one, m_quad, module(vk_barrier_order_ssbo, sizeof(vk_barrier_order_ssbo)));
   VkPipeline p_produce = cs_pipeline(l_one, module(vk_barrier_order_produce, sizeof(vk_barrier_order_produce)));
   VkPipeline p_consume = cs_pipeline(l_two, module(vk_barrier_order_consume, sizeof(vk_barrier_order_consume)));

   VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 },
                                  { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 } };
   VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 4,
      .poolSizeCount = 2, .pPoolSizes = ps };
   VkDescriptorPool dp;
   VKCHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
   /* [0] C as SHADER_READ_ONLY, [1] C as GENERAL, [2] A, [3] A + O */
   VkDescriptorSetLayout want[4] = { dsl[1], dsl[1], dsl[2], dsl[3] };
   VkDescriptorSet set[4];
   VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 4, .pSetLayouts = want };
   VKCHECK(vkAllocateDescriptorSets(dev, &dsai, set));
   VkDescriptorImageInfo ii[2] = { { smp, cv, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
                                   { smp, cv, VK_IMAGE_LAYOUT_GENERAL } };
   VkDescriptorBufferInfo bi[2] = { { a, 0, VK_WHOLE_SIZE }, { o, 0, VK_WHOLE_SIZE } };
   VkWriteDescriptorSet wr[5] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set[0], .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii[0] },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set[1], .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii[1] },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set[2], .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[0] },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set[3], .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[0] },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set[3], .dstBinding = 1, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[1] } };
   vkUpdateDescriptorSets(dev, 5, wr, 0, NULL);

   VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = qf };
   VkCommandPool pool;
   VKCHECK(vkCreateCommandPool(dev, &cpci, NULL, &pool));
   VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
   VkCommandBuffer cb;
   VKCHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
   VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   VkFence fence;
   VKCHECK(vkCreateFence(dev, &fci, NULL, &fence));

   const VkPipelineStageFlags2 CAO = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      FS = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, CS = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      COPY = VK_PIPELINE_STAGE_2_COPY_BIT, TOP = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
      BOTTOM = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
   const VkAccessFlags2 CAW = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, SMP = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
      SW = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, SR = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
      TR = VK_ACCESS_2_TRANSFER_READ_BIT, TW = VK_ACCESS_2_TRANSFER_WRITE_BIT;
   const VkImageLayout CA = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, RO = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      GEN = VK_IMAGE_LAYOUT_GENERAL, TS = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   const char *names[3] = { "colour", "compute", "cs-gfx" };
   long bad[3] = { 0, 0, 0 }, bad_rounds[3] = { 0, 0, 0 };
   uint32_t base = 1000;

   for (int it = 0; it < iterations; it++) {
      VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
      VKCHECK(vkBeginCommandBuffer(cb, &cbi));
      img_barrier(cb, c, VK_IMAGE_LAYOUT_UNDEFINED, CA, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, CAO, CAW, 0);
      img_barrier(cb, c2, VK_IMAGE_LAYOUT_UNDEFINED, CA, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, CAO, CAW, 0);
      for (int r = 0; r < ROUNDS; r++) {
         struct pc pc = { base + (uint32_t)r, spin };
         VkDeviceSize slot = (VkDeviceSize)r * 3;

         /* colour -> sampled */
         int variant = r % 3;
         VkImageLayout cl = variant == 1 ? GEN : CA;
         if (variant == 1)
            img_barrier(cb, c, CA, GEN, CAO, CAW, CAO, CAW, 0);
         vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_fill);
         vkCmdPushConstants(cb, l_fill, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
         render(cb, cv, cl);
         if (unordered) {
            if (variant != 1)
               img_barrier(cb, c, CA, RO, TOP, 0, BOTTOM, 0, 0);
         } else if (variant == 1) {
            mem_barrier(cb, CAO, CAW, FS, SMP, 0);
         } else {
            img_barrier(cb, c, CA, RO, CAO, CAW, FS, SMP, variant == 2 ? VK_DEPENDENCY_BY_REGION_BIT : 0);
         }
         vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_sample);
         vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, l_tex, 0, 1, &set[variant == 1 ? 1 : 0], 0, NULL);
         render(cb, c2v, CA);
         img_barrier(cb, c, variant == 1 ? GEN : RO, CA, FS, SMP, CAO, CAW, 0);
         img_barrier(cb, c2, CA, TS, CAO, CAW, COPY, TR, 0);
         VkBufferImageCopy cc = { .bufferOffset = slot * bsize, .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .imageExtent = { W, H, 1 } };
         vkCmdCopyImageToBuffer(cb, c2, TS, rb, 1, &cc);
         img_barrier(cb, c2, TS, CA, COPY, TR, CAO, CAW, 0);

         /* compute -> compute */
         vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_produce);
         vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, l_one, 0, 1, &set[2], 0, NULL);
         vkCmdPushConstants(cb, l_one, VK_SHADER_STAGE_ALL, 0, sizeof(pc), &pc);
         vkCmdDispatch(cb, N / 64, 1, 1);
         if (!unordered)
            mem_barrier(cb, CS, SW, CS, SR, 0);
         vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_consume);
         vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, l_two, 0, 1, &set[3], 0, NULL);
         vkCmdDispatch(cb, N / 64, 1, 1);
         mem_barrier(cb, CS, SW | SR, COPY, TR, 0);
         VkBufferCopy bc = { 0, (slot + 1) * bsize, bsize };
         vkCmdCopyBuffer(cb, o, rb, 1, &bc);

         /* compute -> fragment storage read; producer value differs from the compute case */
         struct pc pc2 = { pc.value + 500000u, spin };
         mem_barrier(cb, CS, SR, CS, SW, 0);
         vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p_produce);
         vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, l_one, 0, 1, &set[2], 0, NULL);
         vkCmdPushConstants(cb, l_one, VK_SHADER_STAGE_ALL, 0, sizeof(pc2), &pc2);
         vkCmdDispatch(cb, N / 64, 1, 1);
         if (!unordered)
            mem_barrier(cb, CS, SW, FS, SR, 0);
         vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p_ssbo);
         vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, l_one, 0, 1, &set[2], 0, NULL);
         render(cb, c2v, CA);
         img_barrier(cb, c2, CA, TS, CAO, CAW, COPY, TR, 0);
         VkBufferImageCopy cc2 = { .bufferOffset = (slot + 2) * bsize,
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { W, H, 1 } };
         vkCmdCopyImageToBuffer(cb, c2, TS, rb, 1, &cc2);
         img_barrier(cb, c2, TS, CA, COPY, TR, CAO, CAW, 0);
         /* the next round's producers overwrite A after this round's fragment reads */
         mem_barrier(cb, FS, SR, CS, SW, 0);
      }
      mem_barrier(cb, COPY, TW, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT, 0);
      VKCHECK(vkEndCommandBuffer(cb));
      memset(rb_map, 0xcd, slots * bsize);
      VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
      VKCHECK(vkQueueSubmit(queue, 1, &si, fence));
      VKCHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 20ull * 1000 * 1000 * 1000));
      VKCHECK(vkResetFences(dev, 1, &fence));
      VKCHECK(vkResetCommandBuffer(cb, 0));

      for (int r = 0; r < ROUNDS; r++) {
         uint32_t v = base + (uint32_t)r;
         for (int k = 0; k < 3; k++) {
            const uint32_t *got = rb_map + ((size_t)r * 3 + k) * N;
            long wrong = 0;
            uint32_t first = 0;
            int first_i = -1;
            for (int i = 0; i < N; i++) {
               uint32_t want = k == 0 ? v : k == 1 ? v + (uint32_t)i + 1 : v + 500000u + (uint32_t)i;
               if (got[i] != want) {
                  if (first_i < 0) { first_i = i; first = got[i]; }
                  wrong++;
               }
            }
            if (wrong) {
               if (bad_rounds[k] < 5)
                  printf("MISMATCH %s it %d round %d: %ld wrong, first [%d] = %u\n", names[k], it, r, wrong,
                         first_i, first);
               bad[k] += wrong;
               bad_rounds[k]++;
            }
         }
      }
      base += ROUNDS;
   }
   long total = bad[0] + bad[1] + bad[2];
   for (int k = 0; k < 3; k++)
      printf("%-8s %d rounds, %ld bad rounds, %ld wrong values\n", names[k], iterations * ROUNDS, bad_rounds[k], bad[k]);
   printf("%s\n", total ? "FAIL" : "PASS");
   return total ? 1 : 0;
}
