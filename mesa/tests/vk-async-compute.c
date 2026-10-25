/*
 * A graphics queue and a compute-only-family queue of one device.
 *
 *   overlap   a busy draw on the graphics queue and a busy dispatch on the
 *             compute queue, alone and submitted together; prints the wall
 *             times and the overlap factor (alone sum / together). Results of
 *             both are checked.
 *   solo      a device with only a compute-family queue (its context in a
 *             group of its own): one busy dispatch, result checked.
 *   teardown  a busy draw and a busy dispatch submitted on both queues, then
 *             vkDestroyDevice at once: the device must drain both queues
 *             before freeing their channels (no fault, bounded time).
 *   pingpong  STEPS dependent dispatches alternating between the two queues
 *             (the compute queue's dispatched indirectly), each v[k+1] = v[k] + 1: with a timeline semaphore submitted in
 *             step order, with the timeline submitted compute-queue steps
 *             first (GPU-side wait before signal), and with binary
 *             semaphores. Every chain must end at STEPS.
 *
 * The device's queue create infos list the compute family first: queue
 * creation must not depend on that order.
 *
 * Build: glslangValidator -V --target-env vulkan1.3 -S comp -DSTAGE_SPIN  --vn spin_spv  -o spin.h  vk-async-compute.glsl
 *        (and -DSTAGE_CHAIN chain_spv, -S vert -DSTAGE_VERT vert_spv, -S frag -DSTAGE_FRAG frag_spv)
 *        cc -O2 -o vk-async-compute vk-async-compute.c -I<dir with the four headers> -lvulkan
 * usage: vk-async-compute overlap|pingpong [TARGET_MS]
 * Exit 0 pass, 1 fail, 77 no compute-only family.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#include "spin.h"
#include "chain.h"
#include "vert.h"
#include "frag.h"

#define VKCHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
   fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #x, r_); exit(2); } } while (0)

#define SPIN_GROUPS 16
#define SPIN_N (SPIN_GROUPS * 64)
#define RT 64
#define STEPS 16

static VkDevice dev;
static VkPhysicalDevice pd;
static uint32_t gfx_family, comp_family;
static VkQueue gq, cq;
static VkCommandPool gpool, cpool;

static double now_ms(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   fprintf(stderr, "no memory type\n");
   exit(2);
}

struct buf { VkBuffer b; VkDeviceMemory m; uint32_t *map; };

static struct buf make_buffer(VkDeviceSize size, VkBufferUsageFlags usage)
{
   struct buf r;
   uint32_t fams[2] = { gfx_family, comp_family };
   VkBufferCreateInfo bi = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
      .usage = usage, .sharingMode = VK_SHARING_MODE_CONCURRENT,
      .queueFamilyIndexCount = 2, .pQueueFamilyIndices = fams };
   VKCHECK(vkCreateBuffer(dev, &bi, NULL, &r.b));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, r.b, &mr);
   VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
      .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
   VKCHECK(vkAllocateMemory(dev, &ai, NULL, &r.m));
   VKCHECK(vkBindBufferMemory(dev, r.b, r.m, 0));
   VKCHECK(vkMapMemory(dev, r.m, 0, size, 0, (void **)&r.map));
   memset(r.map, 0, size);
   return r;
}

static VkShaderModule module(const uint32_t *code, size_t size)
{
   VkShaderModuleCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size, .pCode = code };
   VkShaderModule m;
   VKCHECK(vkCreateShaderModule(dev, &ci, NULL, &m));
   return m;
}

static VkDescriptorSetLayout dsl;
static VkPipelineLayout cpl, gpl;
static VkPipeline spin_pipe, chain_pipe, draw_pipe;
static VkDescriptorPool dpool;

static VkDescriptorSet buffer_set(VkBuffer b)
{
   VkDescriptorSetAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl };
   VkDescriptorSet s;
   VKCHECK(vkAllocateDescriptorSets(dev, &ai, &s));
   VkDescriptorBufferInfo bi = { b, 0, VK_WHOLE_SIZE };
   VkWriteDescriptorSet w = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = s,
      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi };
   vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
   return s;
}

static void make_pipelines(void)
{
   VkDescriptorSetLayoutBinding b = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT };
   VkDescriptorSetLayoutCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &b };
   VKCHECK(vkCreateDescriptorSetLayout(dev, &dci, NULL, &dsl));
   VkPushConstantRange cr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 };
   VkPipelineLayoutCreateInfo lci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &cr };
   VKCHECK(vkCreatePipelineLayout(dev, &lci, NULL, &cpl));
   VkPushConstantRange gr = { VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4 };
   VkPipelineLayoutCreateInfo glci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &gr };
   VKCHECK(vkCreatePipelineLayout(dev, &glci, NULL, &gpl));

   VkComputePipelineCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT, .pName = "main",
                 .module = module(spin_spv, sizeof(spin_spv)) }, .layout = cpl };
   VKCHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &spin_pipe));
   cpi.stage.module = module(chain_spv, sizeof(chain_spv));
   VKCHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &chain_pipe));

   VkPipelineShaderStageCreateInfo st[2] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = module(vert_spv, sizeof(vert_spv)), .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = module(frag_spv, sizeof(frag_spv)), .pName = "main" } };
   VkPipelineVertexInputStateCreateInfo vi = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
   VkPipelineInputAssemblyStateCreateInfo ia = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
   VkViewport vp = { 0, 0, RT, RT, 0, 1 };
   VkRect2D sc = { { 0, 0 }, { RT, RT } };
   VkPipelineViewportStateCreateInfo vs = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc };
   VkPipelineRasterizationStateCreateInfo rs = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1 };
   VkPipelineMultisampleStateCreateInfo ms = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
   VkPipelineColorBlendAttachmentState cba = { .colorWriteMask = 0xf };
   VkPipelineColorBlendStateCreateInfo cb = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &cba };
   VkFormat fmt = VK_FORMAT_R32G32B32A32_UINT;
   VkPipelineRenderingCreateInfo ri = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt };
   VkGraphicsPipelineCreateInfo gpi = { .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
      .stageCount = 2, .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vs, .pRasterizationState = &rs, .pMultisampleState = &ms,
      .pColorBlendState = &cb, .layout = gpl };
   VKCHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpi, NULL, &draw_pipe));
}

static VkCommandBuffer begin(VkCommandPool pool)
{
   VkCommandBufferAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
   VkCommandBuffer c;
   VKCHECK(vkAllocateCommandBuffers(dev, &ai, &c));
   VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   VKCHECK(vkBeginCommandBuffer(c, &bi));
   return c;
}

/* x after `iters` LCG steps, as one affine map. */
static uint32_t lcg(uint32_t x, uint32_t iters)
{
   uint32_t a = 1, c = 0, sa = 1664525u, sc = 1013904223u;
   while (iters) {
      if (iters & 1) { c = sa * c + sc; a = sa * a; }
      sc = sa * sc + sc;
      sa = sa * sa;
      iters >>= 1;
   }
   return a * x + c;
}

/* ---------------- overlap ---------------- */
static struct buf spin_out, rt_read;
static VkImage rt;
static VkImageView rt_view;
static VkDescriptorSet spin_set;

static void make_rt(void)
{
   VkImageCreateInfo ii = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R32G32B32A32_UINT, .extent = { RT, RT, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   VKCHECK(vkCreateImage(dev, &ii, NULL, &rt));
   VkMemoryRequirements mr;
   vkGetImageMemoryRequirements(dev, rt, &mr);
   VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
      .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) };
   VkDeviceMemory m;
   VKCHECK(vkAllocateMemory(dev, &ai, NULL, &m));
   VKCHECK(vkBindImageMemory(dev, rt, m, 0));
   VkImageViewCreateInfo vi = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = rt,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = ii.format,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   VKCHECK(vkCreateImageView(dev, &vi, NULL, &rt_view));
}

static VkCommandBuffer record_draw(uint32_t iters)
{
   VkCommandBuffer c = begin(gpool);
   VkImageMemoryBarrier2 to_att = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = rt,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   VkDependencyInfo d = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &to_att };
   vkCmdPipelineBarrier2(c, &d);
   VkRenderingAttachmentInfo att = { .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = rt_view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE };
   VkRenderingInfo ri = { .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = { { 0, 0 }, { RT, RT } },
      .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &att };
   vkCmdBeginRendering(c, &ri);
   vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, draw_pipe);
   vkCmdPushConstants(c, gpl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4, &iters);
   vkCmdDraw(c, 3, 1, 0, 0);
   vkCmdEndRendering(c);
   VkImageMemoryBarrier2 to_src = to_att;
   to_src.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
   to_src.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
   to_src.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
   to_src.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
   to_src.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   to_src.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   d.pImageMemoryBarriers = &to_src;
   vkCmdPipelineBarrier2(c, &d);
   VkBufferImageCopy bic = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .imageExtent = { RT, RT, 1 } };
   vkCmdCopyImageToBuffer(c, rt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rt_read.b, 1, &bic);
   VKCHECK(vkEndCommandBuffer(c));
   return c;
}

static VkCommandBuffer record_spin(uint32_t iters, uint32_t seed)
{
   VkCommandBuffer c = begin(cpool);
   vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, spin_pipe);
   vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, cpl, 0, 1, &spin_set, 0, NULL);
   uint32_t pc[2] = { iters, seed };
   vkCmdPushConstants(c, cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
   vkCmdDispatch(c, SPIN_GROUPS, 1, 1);
   VKCHECK(vkEndCommandBuffer(c));
   return c;
}

static double run(VkCommandBuffer g, VkCommandBuffer c)
{
   VkFence f[2];
   VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   VKCHECK(vkCreateFence(dev, &fi, NULL, &f[0]));
   VKCHECK(vkCreateFence(dev, &fi, NULL, &f[1]));
   double t0 = now_ms();
   uint32_t n = 0;
   if (g) {
      VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &g };
      VKCHECK(vkQueueSubmit(gq, 1, &si, f[n++]));
   }
   if (c) {
      VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &c };
      VKCHECK(vkQueueSubmit(cq, 1, &si, f[n++]));
   }
   VKCHECK(vkWaitForFences(dev, n, f, VK_TRUE, 10ull * 1000 * 1000 * 1000));
   double t = now_ms() - t0;
   vkDestroyFence(dev, f[0], NULL);
   vkDestroyFence(dev, f[1], NULL);
   return t;
}

static int cmp_d(const void *a, const void *b)
{
   double x = *(const double *)a, y = *(const double *)b;
   return x < y ? -1 : x > y;
}

static int check_spin(uint32_t iters, uint32_t seed)
{
   for (uint32_t i = 0; i < SPIN_N; i++)
      if (spin_out.map[i] != lcg(seed + i, iters)) {
         fprintf(stderr, "spin[%u] = %#x, want %#x\n", i, spin_out.map[i], lcg(seed + i, iters));
         return 1;
      }
   return 0;
}

static int check_draw(uint32_t iters)
{
   /* Pixel (x, y) of the R32G32B32A32 target, row-major, tightly packed. */
   for (uint32_t y = 0; y < RT; y += 21)
      for (uint32_t x = 0; x < RT; x += 13) {
         uint32_t got = rt_read.map[(y * RT + x) * 4];
         uint32_t want = lcg(x * 4096u + y, iters);
         if (got != want) {
            fprintf(stderr, "pixel (%u,%u) = %#x, want %#x\n", x, y, got, want);
            return 1;
         }
      }
   return 0;
}

static int overlap(double target)
{
   spin_out = make_buffer(SPIN_N * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
   rt_read = make_buffer(RT * RT * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   spin_set = buffer_set(spin_out.b);
   make_rt();
   /* Calibrate each side to `target` ms alone. */
   uint32_t ic = 1u << 20, ig = 1u << 16;
   double tc = run(VK_NULL_HANDLE, record_spin(ic, 1));
   tc = run(VK_NULL_HANDLE, record_spin(ic, 1));
   double tg = run(record_draw(ig), VK_NULL_HANDLE);
   tg = run(record_draw(ig), VK_NULL_HANDLE);
   ic = (uint32_t)(ic * target / tc);
   ig = (uint32_t)(ig * target / tg);
   printf("calibrated: compute %u iterations, fragment %u iterations\n", ic, ig);
   VkCommandBuffer c = record_spin(ic, 7), g = record_draw(ig);
   double a[5], b[5], ab[5];
   for (int r = 0; r < 5; r++) {
      a[r] = run(g, VK_NULL_HANDLE);
      if (check_draw(ig)) return 1;
      b[r] = run(VK_NULL_HANDLE, c);
      if (check_spin(ic, 7)) return 1;
      memset(spin_out.map, 0, SPIN_N * 4);
      memset(rt_read.map, 0, RT * RT * 16);
      ab[r] = run(g, c);
      if (check_draw(ig) || check_spin(ic, 7)) return 1;
   }
   qsort(a, 5, sizeof(double), cmp_d);
   qsort(b, 5, sizeof(double), cmp_d);
   qsort(ab, 5, sizeof(double), cmp_d);
   printf("graphics queue draw alone %.2f ms, compute queue dispatch alone %.2f ms, together %.2f ms: overlap %.2fx\n",
          a[2], b[2], ab[2], (a[2] + b[2]) / ab[2]);
   return 0;
}

/* ---------------- pingpong ---------------- */
static struct buf chain, indirect;
static VkDescriptorSet chain_set;

static VkCommandBuffer record_step(uint32_t k, uint32_t iters)
{
   VkCommandBuffer c = begin(k & 1 ? cpool : gpool);
   vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, chain_pipe);
   vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, cpl, 0, 1, &chain_set, 0, NULL);
   uint32_t pc[3] = { k, k + 1, iters };
   vkCmdPushConstants(c, cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, pc);
   /* Compute-queue steps dispatch indirectly: that runs an MME macro
    * through the compute class on the compute queue's subcontext.
    */
   if (k & 1)
      vkCmdDispatchIndirect(c, indirect.b, 0);
   else
      vkCmdDispatch(c, 1, 1, 1);
   VKCHECK(vkEndCommandBuffer(c));
   return c;
}

static int check_chain(const char *what)
{
   for (uint32_t k = 0; k <= STEPS; k++)
      if (chain.map[k] != k) {
         fprintf(stderr, "%s: v[%u] = %u, want %u\n", what, k, chain.map[k], k);
         return 1;
      }
   printf("%s: chain of %u steps across both queues correct\n", what, STEPS);
   return 0;
}

static int pingpong(void)
{
   chain = make_buffer(4096 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
   indirect = make_buffer(16, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
   indirect.map[0] = indirect.map[1] = indirect.map[2] = 1;
   chain_set = buffer_set(chain.b);
   const uint32_t iters = 1u << 18;
   VkCommandBuffer steps[STEPS];
   for (uint32_t k = 0; k < STEPS; k++)
      steps[k] = record_step(k, iters);

   /* Timeline, in step order and then with the compute steps first. */
   for (int reverse = 0; reverse < 2; reverse++) {
      memset(chain.map, 0, 4096 * 4);
      VkSemaphoreTypeCreateInfo ti = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
         .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE };
      VkSemaphoreCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &ti };
      VkSemaphore tl;
      VKCHECK(vkCreateSemaphore(dev, &sci, NULL, &tl));
      for (int pass = 0; pass < 2; pass++) {
         for (uint32_t k = 0; k < STEPS; k++) {
            /* reverse: compute-queue (odd) steps in pass 0, graphics in pass 1. */
            int odd = k & 1;
            if (reverse ? (pass == 0) != odd : pass == 1)
               continue;
            uint64_t wait = k, signal = k + 1;
            VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            VkTimelineSemaphoreSubmitInfo tsi = { .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
               .waitSemaphoreValueCount = k ? 1 : 0, .pWaitSemaphoreValues = &wait,
               .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &signal };
            VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &tsi,
               .waitSemaphoreCount = k ? 1 : 0, .pWaitSemaphores = &tl, .pWaitDstStageMask = &stage,
               .commandBufferCount = 1, .pCommandBuffers = &steps[k],
               .signalSemaphoreCount = 1, .pSignalSemaphores = &tl };
            VKCHECK(vkQueueSubmit(odd ? cq : gq, 1, &si, VK_NULL_HANDLE));
         }
      }
      uint64_t want = STEPS;
      VkSemaphoreWaitInfo wi = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1,
         .pSemaphores = &tl, .pValues = &want };
      VKCHECK(vkWaitSemaphores(dev, &wi, 10ull * 1000 * 1000 * 1000));
      if (check_chain(reverse ? "timeline, compute steps submitted first" : "timeline, step order"))
         return 1;
      vkDestroySemaphore(dev, tl, NULL);
   }

   /* Binary semaphores in step order. */
   memset(chain.map, 0, 4096 * 4);
   VkSemaphore bs[STEPS];
   VkSemaphoreCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
   for (uint32_t k = 0; k < STEPS; k++)
      VKCHECK(vkCreateSemaphore(dev, &bci, NULL, &bs[k]));
   VkFence done;
   VkFenceCreateInfo fi = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   VKCHECK(vkCreateFence(dev, &fi, NULL, &done));
   for (uint32_t k = 0; k < STEPS; k++) {
      VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .waitSemaphoreCount = k ? 1 : 0, .pWaitSemaphores = k ? &bs[k - 1] : NULL, .pWaitDstStageMask = &stage,
         .commandBufferCount = 1, .pCommandBuffers = &steps[k],
         .signalSemaphoreCount = k + 1 < STEPS ? 1 : 0, .pSignalSemaphores = &bs[k] };
      VKCHECK(vkQueueSubmit(k & 1 ? cq : gq, 1, &si, k + 1 == STEPS ? done : VK_NULL_HANDLE));
   }
   VKCHECK(vkWaitForFences(dev, 1, &done, VK_TRUE, 10ull * 1000 * 1000 * 1000));
   return check_chain("binary semaphores");
}

int main(int argc, char **argv)
{
   setvbuf(stdout, NULL, _IONBF, 0);
   const char *mode = argc > 1 ? argv[1] : "overlap";
   double target = argc > 2 ? atof(argv[2]) : 100.0;
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
   VkInstance inst;
   VKCHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1;
   VkResult r = vkEnumeratePhysicalDevices(inst, &n, &pd);
   if (r != VK_SUCCESS && r != VK_INCOMPLETE) return 2;
   uint32_t fc = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &fc, NULL);
   VkQueueFamilyProperties qf[16];
   if (fc > 16) fc = 16;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &fc, qf);
   gfx_family = comp_family = UINT32_MAX;
   for (uint32_t i = 0; i < fc; i++) {
      if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && gfx_family == UINT32_MAX) gfx_family = i;
      if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
          comp_family == UINT32_MAX) comp_family = i;
   }
   if (comp_family == UINT32_MAX) {
      printf("no compute-only queue family\n");
      return 77;
   }
   printf("graphics family %u, compute family %u (x%u)\n", gfx_family, comp_family, qf[comp_family].queueCount);
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci[2] = {
      { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = comp_family, .queueCount = 1, .pQueuePriorities = &prio },
      { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = gfx_family, .queueCount = 1, .pQueuePriorities = &prio } };
   VkPhysicalDeviceVulkan13Features f13 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .dynamicRendering = VK_TRUE, .synchronization2 = VK_TRUE };
   VkPhysicalDeviceVulkan12Features f12 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &f13, .timelineSemaphore = VK_TRUE };
   const int solo = !strcmp(mode, "solo");
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f12,
      .queueCreateInfoCount = solo ? 1 : 2, .pQueueCreateInfos = qci };
   VKCHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   if (!solo)
      vkGetDeviceQueue(dev, gfx_family, 0, &gq);
   vkGetDeviceQueue(dev, comp_family, 0, &cq);
   VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = gfx_family };
   if (!solo)
      VKCHECK(vkCreateCommandPool(dev, &pci, NULL, &gpool));
   pci.queueFamilyIndex = comp_family;
   VKCHECK(vkCreateCommandPool(dev, &pci, NULL, &cpool));
   VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 };
   VkDescriptorPoolCreateInfo dpi = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 8,
      .poolSizeCount = 1, .pPoolSizes = &ps };
   VKCHECK(vkCreateDescriptorPool(dev, &dpi, NULL, &dpool));
   make_pipelines();
   int ret;
   if (!strcmp(mode, "teardown")) {
      spin_out = make_buffer(SPIN_N * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      rt_read = make_buffer(RT * RT * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
      spin_set = buffer_set(spin_out.b);
      make_rt();
      VkCommandBuffer g = record_draw(4u << 20), c = record_spin(4u << 20, 5);
      VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &g };
      VKCHECK(vkQueueSubmit(gq, 1, &si, VK_NULL_HANDLE));
      si.pCommandBuffers = &c;
      VKCHECK(vkQueueSubmit(cq, 1, &si, VK_NULL_HANDLE));
      double t0 = now_ms();
      vkDestroyPipeline(dev, spin_pipe, NULL);
      vkDestroyPipeline(dev, chain_pipe, NULL);
      vkDestroyPipeline(dev, draw_pipe, NULL);
      vkDestroyDevice(dev, NULL);
      printf("device destroyed with both queues busy in %.2f ms\n", now_ms() - t0);
      vkDestroyInstance(inst, NULL);
      printf("PASS\n");
      return 0;
   }
   if (solo) {
      spin_out = make_buffer(SPIN_N * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      spin_set = buffer_set(spin_out.b);
      double t = run(VK_NULL_HANDLE, record_spin(1u << 16, 3));
      ret = check_spin(1u << 16, 3);
      printf("solo compute queue dispatch %.2f ms\n", t);
   } else {
      ret = !strcmp(mode, "pingpong") ? pingpong() : overlap(target);
   }
   /* Work is idle here: destroying the device tears down both queues. */
   VKCHECK(vkDeviceWaitIdle(dev));
   vkDestroyPipeline(dev, spin_pipe, NULL);
   vkDestroyPipeline(dev, chain_pipe, NULL);
   vkDestroyPipeline(dev, draw_pipe, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   printf("%s\n", ret ? "FAIL" : "PASS");
   return ret;
}
