/*
 * Compute -> compute barriers on a command buffer that is executed more than once.
 *
 * The command buffer runs EPOCHS epochs; each epoch dispatches SLICES dispatches, dispatch j
 * incrementing every element of slice j of buffer A once (read, spin, write x + 1), then a
 * COMPUTE / SHADER_STORAGE_WRITE|READ -> COMPUTE / SHADER_STORAGE_READ|WRITE memory barrier.
 * Each slice is therefore incremented once per epoch and every increment must be ordered behind
 * the previous epoch's: after one execution every element is EPOCHS. Several dispatches per
 * epoch exercise more than one dispatch waiting on the same barrier; enough epochs reach the
 * driver's dependency-chain limit.
 *
 * Modes (argv[3]):
 *   reuse   (default) record once (no ONE_TIME_SUBMIT), submit ITERATIONS times, one at a time,
 *           A reset to 0 by the host before each; every execution must give EPOCHS.
 *   double  SIMULTANEOUS_USE, the same command buffer twice in one vkQueueSubmit, ITERATIONS
 *           times: 2 * EPOCHS (the trailing barrier orders the second execution behind the first).
 *   clear   as reuse, plus a vkCmdClearColorImage of an unrelated image between the first and
 *           second dispatch of every epoch (3D work while the barrier's dependency is open).
 *   split   epoch 0 in a first command buffer, then a second command buffer (same submit) that
 *           starts with a one-workgroup dispatch on an extra slice and a barrier, then epochs
 *           1..: the barrier must also order the first command buffer's dispatches.
 *   unordered  as reuse without the barriers: must FAIL (negative control).
 *
 * usage: vk-cs-join-reuse [ITERATIONS [SPIN [MODE [EPOCHS [SLICES]]]]]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "vk-cs-join-reuse.inc.h"

#define VKCHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
   fprintf(stderr, "%s:%d: %s = %d\n", __FILE__, __LINE__, #x, r_); exit(2); } } while (0)

#define N 65536

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

static void
cs_barrier(VkCommandBuffer cb)
{
   VkMemoryBarrier2 mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT };
   VkDependencyInfo di = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1,
      .pMemoryBarriers = &mb };
   vkCmdPipelineBarrier2(cb, &di);
}

int
main(int argc, char **argv)
{
   int iterations = argc > 1 ? atoi(argv[1]) : 20;
   uint32_t spin = argc > 2 ? (uint32_t)atoi(argv[2]) : 2000;
   const char *mode = argc > 3 ? argv[3] : "reuse";
   int epochs = argc > 4 ? atoi(argv[4]) : 16;
   int slices = argc > 5 ? atoi(argv[5]) : 3;
   int dbl = !strcmp(mode, "double"), clear = !strcmp(mode, "clear"), unordered = !strcmp(mode, "unordered");
   int split = !strcmp(mode, "split");
   if (iterations <= 0 || epochs <= 0 || slices <= 0 || slices > 16 ||
       (!dbl && !clear && !unordered && !split && strcmp(mode, "reuse"))) {
      fprintf(stderr, "usage: %s [ITERATIONS [SPIN [reuse|double|clear|unordered [EPOCHS [SLICES]]]]]\n", argv[0]);
      return 2;
   }

   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vk-cs-join-reuse", .apiVersion = VK_API_VERSION_1_3 };
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
   printf("driver: %s (%s), mode %s, %d epochs x %d slices, spin %u\n", drv.driverName, drv.driverInfo,
          mode, epochs, slices, spin);

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
      .synchronization2 = VK_TRUE };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f13,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
   VKCHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qf, 0, &queue);

   /* A: host-visible so the host resets and checks it directly */
   const VkDeviceSize asize = (VkDeviceSize)N * 4 * (VkDeviceSize)(slices + 1);
   VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = asize,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
   VkBuffer a;
   VKCHECK(vkCreateBuffer(dev, &bci, NULL, &a));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, a, &mr);
   VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
      .memoryTypeIndex = mem_type(mr.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
   VkDeviceMemory amem;
   VKCHECK(vkAllocateMemory(dev, &mai, NULL, &amem));
   VKCHECK(vkBindBufferMemory(dev, a, amem, 0));
   uint32_t *amap;
   VKCHECK(vkMapMemory(dev, amem, 0, asize, 0, (void **)&amap));

   /* unrelated image for the clear mode */
   VkImageCreateInfo imci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = { 64, 64, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   VkImage img;
   VKCHECK(vkCreateImage(dev, &imci, NULL, &img));
   vkGetImageMemoryRequirements(dev, img, &mr);
   VkMemoryAllocateInfo imai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
      .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) };
   VkDeviceMemory imem;
   VKCHECK(vkAllocateMemory(dev, &imai, NULL, &imem));
   VKCHECK(vkBindImageMemory(dev, img, imem, 0));

   VkDescriptorSetLayoutBinding b = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
   VkDescriptorSetLayoutCreateInfo dlci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &b };
   VkDescriptorSetLayout dsl;
   VKCHECK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl));
   VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 12 };
   VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr };
   VkPipelineLayout pl;
   VKCHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
   VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(vk_cs_join_reuse_inc), .pCode = vk_cs_join_reuse_inc };
   VkShaderModule sm;
   VKCHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
   VkComputePipelineCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" },
      .layout = pl };
   VkPipeline pipe;
   VKCHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

   VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
   VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
      .poolSizeCount = 1, .pPoolSizes = &ps };
   VkDescriptorPool dp;
   VKCHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl };
   VkDescriptorSet set;
   VKCHECK(vkAllocateDescriptorSets(dev, &dsai, &set));
   VkDescriptorBufferInfo dbi = { a, 0, VK_WHOLE_SIZE };
   VkWriteDescriptorSet w = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi };
   vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);

   VkCommandPoolCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qf };
   VkCommandPool pool;
   VKCHECK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
   VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 2 };
   VkCommandBuffer cbv[2];
   VKCHECK(vkAllocateCommandBuffers(dev, &cai, cbv));
   VkCommandBuffer cb = cbv[0];
   VkCommandBufferBeginInfo cbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = dbl ? VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT : 0 };
   VKCHECK(vkBeginCommandBuffer(cb, &cbi));
   if (clear) {
      VkImageMemoryBarrier2 ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_NONE, .dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT,
         .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_GENERAL, .image = img,
         .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
      VkDependencyInfo di = { .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
         .pImageMemoryBarriers = &ib };
      vkCmdPipelineBarrier2(cb, &di);
   }
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
   for (int e = 0; e < epochs; e++) {
      if (split && e == 1) {
         /* second command buffer: a dispatch on the extra slice, then the barrier */
         VKCHECK(vkEndCommandBuffer(cb));
         cb = cbv[1];
         VKCHECK(vkBeginCommandBuffer(cb, &cbi));
         vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
         vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
         uint32_t pc[3] = { (uint32_t)slices, spin, N };
         vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
         vkCmdDispatch(cb, 1, 1, 1); /* short: done long before epoch 0 */
         cs_barrier(cb);
      }
      for (int j = 0; j < slices; j++) {
         uint32_t pc[3] = { (uint32_t)j, spin, N };
         vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
         vkCmdDispatch(cb, N / 64, 1, 1);
         if (clear && j == 0) {
            VkClearColorValue cv = { .float32 = { (float)e / epochs, 0, 0, 1 } };
            VkImageSubresourceRange r = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &r);
         }
      }
      if (!unordered && !(split && e == 0))
         cs_barrier(cb);
   }
   VKCHECK(vkEndCommandBuffer(cb));

   VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   VkFence fence;
   VKCHECK(vkCreateFence(dev, &fci, NULL, &fence));
   VkCommandBuffer cbs[2] = { cb, cb };
   if (split)
      cbs[0] = cbv[0], cbs[1] = cbv[1];
   const uint32_t want = (uint32_t)epochs * (dbl ? 2u : 1u);
   long bad_execs = 0, bad_values = 0;
   for (int it = 0; it < iterations; it++) {
      memset(amap, 0, asize);
      VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = dbl || split ? 2 : 1,
         .pCommandBuffers = cbs };
      VKCHECK(vkQueueSubmit(queue, 1, &si, fence));
      VKCHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 20ull * 1000 * 1000 * 1000));
      VKCHECK(vkResetFences(dev, 1, &fence));
      long wrong = 0;
      int first_i = -1;
      uint32_t first = 0;
      for (size_t i = 0; i < (size_t)N * (size_t)(slices + 1); i++)
         if (amap[i] != (i < (size_t)N * (size_t)slices ? want :
                         split && i < (size_t)N * (size_t)slices + 64 ? 1u : 0u)) {
            if (first_i < 0) { first_i = (int)i; first = amap[i]; }
            wrong++;
         }
      if (wrong) {
         if (bad_execs < 5)
            printf("MISMATCH execution %d: %ld wrong, first [%d] = %u (want %u)\n", it, wrong, first_i, first, want);
         bad_execs++;
         bad_values += wrong;
      }
   }
   printf("%d executions, %ld bad executions, %ld wrong values\n", iterations, bad_execs, bad_values);
   printf("%s\n", bad_values ? "FAIL" : "PASS");
   return bad_values ? 1 : 0;
}
