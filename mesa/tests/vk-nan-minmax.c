/* NaN through GLSL.std.450 FMax/FMin/FClamp for ANGLE (GTA San Andreas black car paint).
 *
 * GTA SA (Android 2.11) computes its vehicle specular term as
 * max(pow(dot(R, L), 10.0), 0.0); for dot < 0 the pow is NaN on NVIDIA (exp2(10 * log2(x))). ANGLE
 * translates GLSL ES max/min/clamp to GLSL.std.450 FMax/FMin/FClamp without float-controls
 * execution modes, so vtn (vtn_glsl450.c:127-150, Mesa 26.2.5) emits them without
 * nir_fp_preserve_nan and Mesa 26 nir_opt_algebraic (:1012-1015) folds
 * fmax(nsz)(a(is_a_number_not_negative), b(is_not_positive)) -> fcanonicalize(a): the NaN reaches
 * the fragment shader and the colour target stores black. Mesa's own GLSL frontend always marks
 * float min/max NaN-preserving (glsl_to_nir.cpp:2435-2453), so the same GLES shader is correct on
 * nvrm, and NVIDIA FMNMX returns the non-NaN operand.
 *
 * NVK fix (nvk-angle-nan-preserve-min-max.patch): driconf nvk_force_nan_preserve_min_max (RADV's
 * radv_force_nan_preserve_min_max, radv_drirc_gen.py:90-92) set for engine "ANGLE" in
 * 00-nvk-defaults.conf -> spirv workarounds.force_nan_preserve_min_max.
 *
 * Usage: vk-nan-minmax ENGINE_NAME EXPECT
 *   EXPECT = maxnum : every lane must equal the IEEE maxNum/minNum result (NaN operand ignored)
 *            report : print only (upstream semantics for non-ANGLE engines are undefined by SPIR-V)
 * The GPU is picked by vendor ID 0x10de (any NVIDIA driver: NVK or the proprietary ICD), never by
 * a node path; the driver is chosen with VK_DRIVER_FILES.
 *
 * Build/run (repository root; see mesa/tests/README.md "vk-nan-minmax"):
 *   glslangValidator -V mesa/tests/vk-nan-minmax.comp \
 *     --vn vk_nan_minmax_spv -o build/mesa-tests/vk-nan-minmax.spv.h
 *   cc -std=c11 -Wall -Wextra -Werror -Ibuild/mesa-tests \
 *     mesa/tests/vk-nan-minmax.c -lvulkan -lm -o build/mesa-tests/vk-nan-minmax
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "vk-nan-minmax.spv.h"

#define VKCHECK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { \
   fprintf(stderr, "FATAL %s:%d %s -> %d\n", __FILE__, __LINE__, #call, r_); exit(2); } } while (0)

#define LANES 4
struct data { float d[LANES], o_max[LANES], o_clamp[LANES], o_min[LANES]; };

int
main(int argc, char **argv)
{
   if (argc != 3 || (strcmp(argv[2], "maxnum") && strcmp(argv[2], "report"))) {
      fprintf(stderr, "usage: %s ENGINE_NAME maxnum|report\n", argv[0]);
      return 2;
   }
   const char *engine = argv[1];
   int strict = !strcmp(argv[2], "maxnum");

   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "vk-nan-minmax", .pEngineName = engine, .apiVersion = VK_API_VERSION_1_1 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
   VkInstance inst;
   VKCHECK(vkCreateInstance(&ici, NULL, &inst));

   uint32_t n = 0;
   VKCHECK(vkEnumeratePhysicalDevices(inst, &n, NULL));
   VkPhysicalDevice pds[16];
   if (n > 16) n = 16;
   VKCHECK(vkEnumeratePhysicalDevices(inst, &n, pds));
   VkPhysicalDevice pd = VK_NULL_HANDLE;
   VkPhysicalDeviceDriverProperties drv = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
   for (uint32_t i = 0; i < n; i++) {
      VkPhysicalDeviceProperties2 p = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &drv };
      vkGetPhysicalDeviceProperties2(pds[i], &p);
      if (p.properties.vendorID == 0x10de) { pd = pds[i]; break; }
   }
   if (pd == VK_NULL_HANDLE) { fprintf(stderr, "no NVIDIA (0x10de) Vulkan device\n"); return 2; }
   printf("driver: %s (%s), engine \"%s\"\n", drv.driverName, drv.driverInfo, engine);

   uint32_t qn = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, NULL);
   VkQueueFamilyProperties qp[16];
   if (qn > 16) qn = 16;
   vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qp);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < qn; i++)
      if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
   if (qf == UINT32_MAX) { fprintf(stderr, "no compute queue\n"); return 2; }

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf, .queueCount = 1, .pQueuePriorities = &prio };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
   VkDevice dev;
   VKCHECK(vkCreateDevice(pd, &dci, NULL, &dev));
   VkQueue q;
   vkGetDeviceQueue(dev, qf, 0, &q);

   VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sizeof(struct data),
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
   VkBuffer buf;
   VKCHECK(vkCreateBuffer(dev, &bci, NULL, &buf));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, buf, &mr);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   uint32_t mt = UINT32_MAX;
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) { mt = i; break; }
   if (mt == UINT32_MAX) { fprintf(stderr, "no host-visible coherent memory type\n"); return 2; }
   VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size, .memoryTypeIndex = mt };
   VkDeviceMemory mem;
   VKCHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
   VKCHECK(vkBindBufferMemory(dev, buf, mem, 0));
   struct data *map;
   VKCHECK(vkMapMemory(dev, mem, 0, sizeof(struct data), 0, (void **)&map));
   static const float in[LANES] = { -0.5f, 0.5f, -2.0f, 0.0f };
   memset(map, 0x55, sizeof(*map));
   memcpy(map->d, in, sizeof(in));

   VkDescriptorSetLayoutBinding b0 = { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
   VkDescriptorSetLayoutCreateInfo dlci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &b0 };
   VkDescriptorSetLayout dsl;
   VKCHECK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl));
   VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl };
   VkPipelineLayout pl;
   VKCHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
   VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(vk_nan_minmax_spv), .pCode = vk_nan_minmax_spv };
   VkShaderModule sm;
   VKCHECK(vkCreateShaderModule(dev, &smci, NULL, &sm));
   VkComputePipelineCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" },
      .layout = pl };
   VkPipeline pipe;
   VKCHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

   VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
   VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
   VkDescriptorPool dp;
   VKCHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl };
   VkDescriptorSet ds;
   VKCHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
   VkDescriptorBufferInfo dbi = { buf, 0, VK_WHOLE_SIZE };
   VkWriteDescriptorSet w = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds,
      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi };
   vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);

   VkCommandPoolCreateInfo cpi = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = qf };
   VkCommandPool cp;
   VKCHECK(vkCreateCommandPool(dev, &cpi, NULL, &cp));
   VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
   VkCommandBuffer cb;
   VKCHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
   VkCommandBufferBeginInfo cbbi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
   VKCHECK(vkBeginCommandBuffer(cb, &cbbi));
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
   vkCmdDispatch(cb, 1, 1, 1);
   VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                        1, &mb, 0, NULL, 0, NULL);
   VKCHECK(vkEndCommandBuffer(cb));
   VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
   VkFence fence;
   VKCHECK(vkCreateFence(dev, &fci, NULL, &fence));
   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
   VKCHECK(vkQueueSubmit(q, 1, &si, fence));
   VKCHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 10ull * 1000 * 1000 * 1000));

   int failures = 0;
   for (int i = 0; i < LANES; i++) {
      /* maxNum/minNum reference: pow(x < 0, 10) is NaN on NVIDIA, NaN operands are dropped. */
      const float p = in[i] < 0.0f ? NAN : powf(in[i], 10.0f);
      const float want_max = (isnan(p) ? 0.0f : fmaxf(p, 0.0f)) * 2.0f;
      const float want_clamp = isnan(p) ? 0.0f : fminf(fmaxf(p, 0.0f), 1.0f);
      const float want_min = isnan(p) ? 0.0f : fminf(-p, 0.0f);
      const float got[3] = { map->o_max[i], map->o_clamp[i], map->o_min[i] };
      const float want[3] = { want_max, want_clamp, want_min };
      static const char *names[3] = { "max(pow(d,10),0)*2", "clamp(pow(d,10),0,1)", "min(-pow(d,10),0)" };
      for (int k = 0; k < 3; k++) {
         /* 1 ulp-ish tolerance for the exp2/log2 pow on d = 0.5 */
         const int ok = !isnan(got[k]) && fabsf(got[k] - want[k]) <= 1e-6f + 1e-4f * fabsf(want[k]);
         printf("  d=%5.2f %-22s = %-12g (maxNum %g)%s\n", in[i], names[k], got[k], want[k],
                ok ? "" : (strict ? "  FAIL" : "  differs"));
         if (!ok && strict)
            failures++;
      }
   }
   printf("%s\n", strict ? (failures ? "FAIL" : "PASS") : "REPORT");

   vkDestroyFence(dev, fence, NULL);
   vkDestroyCommandPool(dev, cp, NULL);
   vkDestroyDescriptorPool(dev, dp, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyShaderModule(dev, sm, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkUnmapMemory(dev, mem);
   vkFreeMemory(dev, mem, NULL);
   vkDestroyBuffer(dev, buf, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   return failures ? 1 : 0;
}
