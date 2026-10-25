/* Depth-tested rendering with ZCULL active (NVK zcull path on the RM backend).
 *
 * NVK uses ZCULL for a render pass whenever the KMD reports ZCULL geometry
 * (nvk_cmd_draw.c:1494-1590 of Mesa 26.2.5: SET_ZCULL_REGION_* / STORAGE / SUBREGION, CLEAR_ZCULL_REGION
 * for loadOp CLEAR, LOAD_ZCULL for loadOp LOAD, STORE_ZCULL at the end for storeOp STORE). A depth
 * image whose usage is only depth attachment + read-only usages gets a ZCULL save plane
 * (nvk_image.c:1142-1155); any other usage (TRANSFER_DST here) renders with ZCULL but without a plane.
 *
 * Each frame, for one of the two depth images (plane / no plane), 300x200 (not a multiple of the
 * ZCULL alignment):
 *   pass 1: clear depth 1.0, LESS, three rectangles (one fully hidden behind an earlier one in part);
 *   pass 2: LOAD (LOAD_ZCULL with a plane), LESS, a full-screen quad partly visible, a full-screen
 *           quad that is hidden everywhere (the case ZCULL rejects), a nearer rectangle;
 *   readback colour + depth;
 *   pass 3: clear depth 0.0, GREATER (opposite direction to pass 1/2), three quads;
 *   pass 4: LOAD, GREATER, one full-screen quad hidden in the centre;
 *   readback colour + depth.
 * Every pixel and every depth value is compared exactly against a CPU model of the same draws.
 * Rectangle edges lie on pixel edges, so coverage is unambiguous.
 *
 * Usage: vk-zcull-depth [FRAMES]   (default 40). PASS = 0 mismatches. GPU picked by vendor 0x10de;
 * the driver is chosen with VK_DRIVER_FILES (NVK on the RM backend or the proprietary ICD).
 * Build/run: tests/README.md "vk-zcull-depth".
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "vk-zcull-depth.vert.h"
#include "vk-zcull-depth.frag.h"

#define VKCHECK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { \
   fprintf(stderr, "FATAL %s:%d %s -> %d\n", __FILE__, __LINE__, #call, r_); exit(2); } } while (0)

#define W 300
#define H 200
#define PIXELS (W * H)

struct pc { float rect[4]; float color[4]; float z; };

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
make_image(VkFormat format, VkImageUsageFlags usage, VkDeviceMemory *mem, VkDeviceSize *size)
{
   VkImageCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = format, .extent = { W, H, 1 }, .mipLevels = 1,
      .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   VkImage img;
   VKCHECK(vkCreateImage(dev, &ici, NULL, &img));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(dev, img, &mr);
   VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size,
      .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) };
   VKCHECK(vkAllocateMemory(dev, &mai, NULL, mem));
   VKCHECK(vkBindImageMemory(dev, img, *mem, 0));
   *size = mr.size;
   return img;
}

static VkImageView
make_view(VkImage img, VkFormat format, VkImageAspectFlags aspect)
{
   VkImageViewCreateInfo vci = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
      .subresourceRange = { aspect, 0, 1, 0, 1 } };
   VkImageView v;
   VKCHECK(vkCreateImageView(dev, &vci, NULL, &v));
   return v;
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
barrier(VkCommandBuffer cb, VkImage img, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to,
        VkPipelineStageFlags2 src_stage, VkAccessFlags2 src, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst)
{
   VkImageMemoryBarrier2 b = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = src_stage, .srcAccessMask = src, .dstStageMask = dst_stage, .dstAccessMask = dst,
      .oldLayout = from, .newLayout = to, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
      .subresourceRange = { aspect, 0, 1, 0, 1 } };
   VkDependencyInfo di = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &b };
   vkCmdPipelineBarrier2(cb, &di);
}

/* CPU model of the same draws (colour channels are 0, 0.2 or 1: exact in UNORM8). */
static uint32_t model_color[PIXELS];
static float model_depth[PIXELS];

static uint32_t
pack(const float c[4])
{
   uint32_t v = 0;
   for (int i = 0; i < 4; i++)
      v |= (uint32_t)(c[i] * 255.0f + 0.5f) << (8 * i);
   return v;
}

static VkPipelineLayout layout;

static void
draw(VkCommandBuffer cb, int x0, int y0, int x1, int y1, const float color[4], float z, VkCompareOp op)
{
   struct pc p = { { -1.0f + 2.0f * x0 / W, -1.0f + 2.0f * y0 / H, -1.0f + 2.0f * x1 / W,
                     -1.0f + 2.0f * y1 / H },
                   { color[0], color[1], color[2], color[3] }, z };
   vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                      sizeof(p), &p);
   vkCmdDraw(cb, 4, 1, 0, 0);
   uint32_t packed = pack(color);
   for (int y = y0; y < y1; y++)
      for (int x = x0; x < x1; x++) {
         float *d = &model_depth[y * W + x];
         int pass = op == VK_COMPARE_OP_LESS ? z < *d : z > *d;
         if (pass) {
            *d = z;
            model_color[y * W + x] = packed;
         }
      }
}

static void
begin_pass(VkCommandBuffer cb, VkImageView color_view, VkImageView depth_view, int clear, float clear_depth,
           VkCompareOp op)
{
   VkRenderingAttachmentInfo ca = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = color_view, .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE };
   VkRenderingAttachmentInfo da = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = depth_view, .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
      .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.depthStencil = { clear_depth, 0 } };
   VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = { { 0, 0 }, { W, H } }, .layerCount = 1, .colorAttachmentCount = 1,
      .pColorAttachments = &ca, .pDepthAttachment = &da };
   vkCmdBeginRendering(cb, &ri);
   vkCmdSetDepthCompareOp(cb, op);
   if (clear) {
      for (int i = 0; i < PIXELS; i++) {
         model_depth[i] = clear_depth;
         model_color[i] = 0;
      }
   }
}

int
main(int argc, char **argv)
{
   int frames = argc > 1 ? atoi(argv[1]) : 40;
   if (frames <= 0) { fprintf(stderr, "usage: %s [FRAMES]\n", argv[0]); return 2; }

   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vk-zcull-depth", .apiVersion = VK_API_VERSION_1_3 };
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
      if (qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
         qf = i;
   if (qf == UINT32_MAX) { fprintf(stderr, "no graphics queue\n"); return 2; }

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

   const VkFormat cfmt = VK_FORMAT_R8G8B8A8_UNORM, dfmt = VK_FORMAT_D32_SFLOAT;
   VkDeviceMemory cmem, dmem[2];
   VkDeviceSize csize, dsize[2];
   VkImage color = make_image(cfmt, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                              &cmem, &csize);
   /* [0]: depth attachment + read-only usage = ZCULL save plane; [1]: + TRANSFER_DST = no plane. */
   VkImage depth[2] = {
      make_image(dfmt, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                 &dmem[0], &dsize[0]),
      make_image(dfmt, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                 VK_IMAGE_USAGE_TRANSFER_DST_BIT, &dmem[1], &dsize[1]),
   };
   printf("depth image sizes: plane %llu B, no plane %llu B\n", (unsigned long long)dsize[0],
          (unsigned long long)dsize[1]);
   VkImageView cview = make_view(color, cfmt, VK_IMAGE_ASPECT_COLOR_BIT);
   VkImageView dview[2] = { make_view(depth[0], dfmt, VK_IMAGE_ASPECT_DEPTH_BIT),
                            make_view(depth[1], dfmt, VK_IMAGE_ASPECT_DEPTH_BIT) };

   /* Readback: two snapshots (after pass 2, after pass 4) of colour + depth. */
   const VkDeviceSize snap = (VkDeviceSize)PIXELS * 8, total = 2 * snap;
   VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = total,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT };
   VkBuffer rb;
   VKCHECK(vkCreateBuffer(dev, &bci, NULL, &rb));
   VkMemoryRequirements bmr;
   vkGetBufferMemoryRequirements(dev, rb, &bmr);
   VkMemoryAllocateInfo bmai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = bmr.size,
      .memoryTypeIndex = mem_type(bmr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
   VkDeviceMemory bmem;
   VKCHECK(vkAllocateMemory(dev, &bmai, NULL, &bmem));
   VKCHECK(vkBindBufferMemory(dev, rb, bmem, 0));
   uint8_t *map;
   VKCHECK(vkMapMemory(dev, bmem, 0, total, 0, (void **)&map));

   VkPushConstantRange pcr = { VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(struct pc) };
   VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
   VKCHECK(vkCreatePipelineLayout(dev, &plci, NULL, &layout));
   VkShaderModule vs = module(vk_zcull_depth_vert, sizeof(vk_zcull_depth_vert));
   VkShaderModule fs = module(vk_zcull_depth_frag, sizeof(vk_zcull_depth_frag));
   VkPipelineShaderStageCreateInfo stages[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fs, .pName = "main" },
   };
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
   VkPipelineDepthStencilStateCreateInfo ds = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE, .depthCompareOp = VK_COMPARE_OP_LESS };
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
   VkPipelineColorBlendStateCreateInfo cbs = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba };
   VkDynamicState dyn[] = { VK_DYNAMIC_STATE_DEPTH_COMPARE_OP };
   VkPipelineDynamicStateCreateInfo dys = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 1, .pDynamicStates = dyn };
   VkPipelineRenderingCreateInfo prci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &cfmt, .depthAttachmentFormat = dfmt };
   VkGraphicsPipelineCreateInfo gpci = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &prci, .stageCount = 2, .pStages = stages, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds,
      .pColorBlendState = &cbs, .pDynamicState = &dys, .layout = layout };
   VkPipeline pipe;
   VKCHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe));

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

   static const float red[4] = { 1, 0, 0, 1 }, green[4] = { 0, 1, 0, 1 }, blue[4] = { 0, 0, 1, 1 },
      yellow[4] = { 1, 1, 0, 1 }, white[4] = { 1, 1, 1, 1 }, orange[4] = { 1, 0.2f, 0, 1 },
      magenta[4] = { 1, 0, 1, 1 }, cyan[4] = { 0, 1, 1, 1 }, grey[4] = { 0.2f, 0.2f, 0.2f, 1 };
   static uint32_t expect_color[2][PIXELS];
   static float expect_depth[2][PIXELS];
   const VkPipelineStageFlags2 frag_tests = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
   const VkAccessFlags2 depth_rw = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
   long bad_total = 0;
   for (int frame = 0; frame < frames; frame++) {
      int k = frame & 1;
      VkImage dimg = depth[k];
      VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
      VKCHECK(vkBeginCommandBuffer(cb, &bi));
      barrier(cb, color, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
              VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      /* UNDEFINED -> attachment: NVK zero-fills the ZCULL plane here (nvk_cmd_buffer.c:839-849). */
      barrier(cb, dimg, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
              VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
              frag_tests, depth_rw);
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);

      for (int half = 0; half < 2; half++) {
         VkCompareOp op = half == 0 ? VK_COMPARE_OP_LESS : VK_COMPARE_OP_GREATER;
         if (half == 0) {
            begin_pass(cb, cview, dview[k], 1, 1.0f, op);
            draw(cb, 0, 0, 150, H, red, 0.5f, op);
            draw(cb, 0, 0, W, H, green, 0.7f, op);
            draw(cb, 0, 0, W, 100, blue, 0.3f, op);
            vkCmdEndRendering(cb);
            barrier(cb, dimg, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, frag_tests, depth_rw, frag_tests, depth_rw);
            begin_pass(cb, cview, dview[k], 0, 0.0f, op);
            draw(cb, 0, 0, W, H, yellow, 0.6f, op);
            draw(cb, 0, 0, W, H, white, 0.9f, op);   /* hidden everywhere */
            draw(cb, 50, 50, 250, 150, orange, 0.25f, op);
            vkCmdEndRendering(cb);
         } else {
            begin_pass(cb, cview, dview[k], 1, 0.0f, op);
            draw(cb, 0, 0, W, H, cyan, 0.2f, op);
            draw(cb, 100, 40, 200, 160, magenta, 0.4f, op);
            draw(cb, 0, 0, W, H, grey, 0.3f, op);
            vkCmdEndRendering(cb);
            barrier(cb, dimg, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, frag_tests, depth_rw, frag_tests, depth_rw);
            begin_pass(cb, cview, dview[k], 0, 0.0f, op);
            draw(cb, 0, 0, W, H, white, 0.35f, op);  /* hidden in the centre */
            draw(cb, 0, 0, W, H, red, 0.1f, op);     /* hidden everywhere */
            vkCmdEndRendering(cb);
         }
         memcpy(expect_color[half], model_color, sizeof(model_color));
         memcpy(expect_depth[half], model_depth, sizeof(model_depth));

         barrier(cb, color, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
         barrier(cb, dimg, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, frag_tests, depth_rw, VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT);
         VkBufferImageCopy cc = { .bufferOffset = half * snap,
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { W, H, 1 } };
         VkBufferImageCopy dc = { .bufferOffset = half * snap + (VkDeviceSize)PIXELS * 4,
            .imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 }, .imageExtent = { W, H, 1 } };
         vkCmdCopyImageToBuffer(cb, color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &cc);
         vkCmdCopyImageToBuffer(cb, dimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &dc);
         if (half == 0) {
            barrier(cb, color, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                    VK_ACCESS_2_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
            barrier(cb, dimg, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                    VK_ACCESS_2_TRANSFER_READ_BIT, frag_tests, depth_rw);
         }
      }
      VkBufferMemoryBarrier2 hb = { .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .buffer = rb, .size = VK_WHOLE_SIZE };
      VkDependencyInfo hdi = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .bufferMemoryBarrierCount = 1,
         .pBufferMemoryBarriers = &hb };
      vkCmdPipelineBarrier2(cb, &hdi);
      VKCHECK(vkEndCommandBuffer(cb));
      memset(map, 0xcd, total);
      VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
      VKCHECK(vkQueueSubmit(queue, 1, &si, fence));
      VKCHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 10ull * 1000 * 1000 * 1000));
      VKCHECK(vkResetFences(dev, 1, &fence));

      for (int half = 0; half < 2; half++) {
         const uint32_t *gc = (const uint32_t *)(map + half * snap);
         const float *gd = (const float *)(map + half * snap + (size_t)PIXELS * 4);
         long bad_c = 0, bad_d = 0;
         int first = -1;
         for (int i = 0; i < PIXELS; i++) {
            int bc = gc[i] != expect_color[half][i];
            int bd = memcmp(&gd[i], &expect_depth[half][i], 4) != 0;
            bad_c += bc;
            bad_d += bd;
            if ((bc || bd) && first < 0)
               first = i;
         }
         if (bad_c || bad_d) {
            printf("frame %d %s %s: %ld colour / %ld depth mismatches; first (%d,%d) colour %08x want %08x "
                   "depth %.9g want %.9g\n", frame, k ? "no-plane" : "plane", half ? "GREATER" : "LESS",
                   bad_c, bad_d, first % W, first / W, gc[first], expect_color[half][first], gd[first],
                   expect_depth[half][first]);
         }
         bad_total += bad_c + bad_d;
      }
   }
   printf("%s: %d frames x 2 snapshots, %ld mismatches\n", bad_total ? "FAIL" : "PASS", frames, bad_total);
   vkDeviceWaitIdle(dev);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   return bad_total ? 1 : 0;
}
