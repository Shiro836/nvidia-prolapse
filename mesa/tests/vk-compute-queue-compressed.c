/* vk-compute-queue-compressed: compressed array images with mip levels through a COMPUTE-without-GRAPHICS queue.
 * A BC1 (8-byte blocks) and a BC7 (16-byte blocks) 2D image, 64x64, 4 levels, 3 layers each: every level/layer is
 * written with random blocks by vkCmdCopyBufferToImage, copied to a second image by vkCmdCopyImage, read back by
 * vkCmdCopyImageToBuffer (per level, all layers in one region) and compared block by block. All transfers run on
 * the compute-only queue. Exit 0 = PASS, 1 = mismatch/error, 77 = no compute-only family or format unsupported.
 * Build: cc -std=c11 -O2 -Wall -Wextra -Werror -o vk-compute-queue-compressed vk-compute-queue-compressed.c -lvulkan
 * Run:   VK_DRIVER_FILES=<NVK ICD json> ./vk-compute-queue-compressed */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define C(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s:%d %s = %d\n", __FILE__, __LINE__, #x, r_); exit(1); } } while (0)

static VkDevice dev; static VkPhysicalDevice pd; static VkQueue q; static VkCommandPool pool; static uint32_t qf;

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
   VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
   fprintf(stderr, "no memory type\n"); exit(1);
}
struct buf { VkBuffer b; VkDeviceMemory m; void *map; };
static struct buf mkbuf(VkDeviceSize size) {
   struct buf r; VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 0, 0, size,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, 0 };
   C(vkCreateBuffer(dev, &bi, 0, &r.b)); VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, r.b, &mr);
   VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, mr.size,
      mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
   C(vkAllocateMemory(dev, &ai, 0, &r.m)); C(vkBindBufferMemory(dev, r.b, r.m, 0)); C(vkMapMemory(dev, r.m, 0, size, 0, &r.map));
   return r;
}
static void freebuf(struct buf *b) { vkUnmapMemory(dev, b->m); vkDestroyBuffer(dev, b->b, 0); vkFreeMemory(dev, b->m, 0); }
struct img { VkImage i; VkDeviceMemory m; };
static struct img mkimg(VkFormat f, uint32_t levels, uint32_t layers) {
   struct img r; VkImageCreateInfo ii = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, 0, 0, VK_IMAGE_TYPE_2D, f, {64, 64, 1}, levels, layers,
      VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      VK_SHARING_MODE_EXCLUSIVE, 0, 0, VK_IMAGE_LAYOUT_UNDEFINED };
   C(vkCreateImage(dev, &ii, 0, &r.i)); VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, r.i, &mr);
   VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, mr.size, mem_type(mr.memoryTypeBits, 0) };
   C(vkAllocateMemory(dev, &ai, 0, &r.m)); C(vkBindImageMemory(dev, r.i, r.m, 0)); return r;
}
static VkCommandBuffer begin(void) {
   VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
   VkCommandBuffer cb; C(vkAllocateCommandBuffers(dev, &ai, &cb));
   VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, 0 };
   C(vkBeginCommandBuffer(cb, &bi)); return cb;
}
static void barrier(VkCommandBuffer cb) {
   VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, 0, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT,
      VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, 0, 0, 0);
}
static void to_general(VkCommandBuffer cb, VkImage i) {
   VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, 0, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, i,
      { VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS } };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, 0, 0, 0, 1, &b);
}
static void submit(VkCommandBuffer cb) {
   C(vkEndCommandBuffer(cb));
   VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &cb, 0, 0 };
   C(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)); C(vkQueueWaitIdle(q)); vkFreeCommandBuffers(dev, pool, 1, &cb);
}

/* 0 = pass, 1 = fail, 77 = unsupported */
static int run(VkFormat f, uint32_t block_B, const char *name) {
   VkImageFormatProperties ifp;
   if (vkGetPhysicalDeviceImageFormatProperties(pd, f, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0, &ifp) != VK_SUCCESS) {
      printf("%s: unsupported\n", name); return 77;
   }
   const uint32_t levels = 4, layers = 3;
   VkDeviceSize off[4], total = 0;
   for (uint32_t l = 0; l < levels; l++) { uint32_t bw = ((64 >> l) + 3) / 4; off[l] = total; total += (VkDeviceSize)bw * bw * block_B * layers; }
   struct buf up = mkbuf(total), down = mkbuf(total);
   srand(1234u + block_B);
   for (VkDeviceSize i = 0; i < total; i++) ((unsigned char *)up.map)[i] = (unsigned char)rand();
   memset(down.map, 0, total);
   struct img a = mkimg(f, levels, layers), b = mkimg(f, levels, layers);
   VkBufferImageCopy rg[4]; VkImageCopy ic[4];
   for (uint32_t l = 0; l < levels; l++) {
      uint32_t w = 64 >> l;
      rg[l] = (VkBufferImageCopy){ off[l], 0, 0, { VK_IMAGE_ASPECT_COLOR_BIT, l, 0, layers }, {0, 0, 0}, {w, w, 1} };
      ic[l] = (VkImageCopy){ { VK_IMAGE_ASPECT_COLOR_BIT, l, 0, layers }, {0, 0, 0}, { VK_IMAGE_ASPECT_COLOR_BIT, l, 0, layers }, {0, 0, 0}, {w, w, 1} };
   }
   VkCommandBuffer cb = begin(); to_general(cb, a.i); to_general(cb, b.i);
   vkCmdCopyBufferToImage(cb, up.b, a.i, VK_IMAGE_LAYOUT_GENERAL, levels, rg); barrier(cb);
   vkCmdCopyImage(cb, a.i, VK_IMAGE_LAYOUT_GENERAL, b.i, VK_IMAGE_LAYOUT_GENERAL, levels, ic); barrier(cb);
   vkCmdCopyImageToBuffer(cb, b.i, VK_IMAGE_LAYOUT_GENERAL, down.b, levels, rg); barrier(cb);
   submit(cb);
   int bad = 0;
   for (uint32_t l = 0; l < levels; l++) {
      uint32_t bw = ((64 >> l) + 3) / 4; VkDeviceSize n = (VkDeviceSize)bw * bw * block_B;
      for (uint32_t y = 0; y < layers; y++)
         if (memcmp((char *)up.map + off[l] + y * n, (char *)down.map + off[l] + y * n, n)) { printf("%s: level %u layer %u differs\n", name, l, y); bad = 1; }
   }
   vkDestroyImage(dev, a.i, 0); vkFreeMemory(dev, a.m, 0); vkDestroyImage(dev, b.i, 0); vkFreeMemory(dev, b.m, 0);
   freebuf(&up); freebuf(&down);
   printf("%s: %s\n", name, bad ? "FAIL" : "ok");
   return bad;
}

int main(void) {
   VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "vk-compute-queue-compressed", 1, 0, 0, VK_API_VERSION_1_3 };
   VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &app, 0, 0, 0, 0 };
   VkInstance inst; C(vkCreateInstance(&ici, 0, &inst));
   uint32_t n = 8; VkPhysicalDevice pds[8]; C(vkEnumeratePhysicalDevices(inst, &n, pds));
   pd = VK_NULL_HANDLE;
   for (uint32_t i = 0; i < n; i++) { VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i], &p); if (p.vendorID == 0x10de) { pd = pds[i]; break; } }
   if (!pd) { printf("no NVIDIA device\n"); return 77; }
   uint32_t nf = 16; VkQueueFamilyProperties qfp[16]; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nf, qfp);
   qf = UINT32_MAX;
   for (uint32_t i = 0; i < nf; i++) if ((qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { qf = i; break; }
   if (qf == UINT32_MAX) { printf("no compute-only queue family\n"); return 77; }
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0, 0, qf, 1, &prio };
   VkPhysicalDeviceFeatures feat = { 0 }; feat.textureCompressionBC = VK_TRUE;
   VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, 0, 0, 1, &qci, 0, 0, 0, 0, &feat };
   C(vkCreateDevice(pd, &dci, 0, &dev)); vkGetDeviceQueue(dev, qf, 0, &q);
   VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, 0, 0, qf }; C(vkCreateCommandPool(dev, &pci, 0, &pool));
   int r1 = run(VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 8, "bc1 64x64 4 levels 3 layers");
   int r2 = run(VK_FORMAT_BC7_UNORM_BLOCK, 16, "bc7 64x64 4 levels 3 layers");
   vkDestroyCommandPool(dev, pool, 0); vkDestroyDevice(dev, 0); vkDestroyInstance(inst, 0);
   int r = (r1 == 1 || r2 == 1) ? 1 : (r1 == 77 && r2 == 77) ? 77 : 0;
   printf("%s\n", r == 0 ? "PASS" : r == 77 ? "UNSUPPORTED" : "FAIL");
   return r;
}
