/* SPDX-License-Identifier: MIT
 *
 * G2: dma-buf import/export and native fences on the nvrm gallium driver,
 * with Vulkan (our NVK on the same RM backend) as the second party.
 *
 *   gl-dmabuf <vendor-substring> [--vk]
 *
 * 1. eglQueryDmaBufModifiersEXT for the window-buffer fourccs.
 * 2. GL texture -> eglExportDMABUFImageMESA -> EGL_LINUX_DMA_BUF_EXT import
 *    into a second texture: readback, sampling (texelFetch draw) and a write
 *    through the import that the exporter must see. Pixel-exact.
 * 3. EGL_ANDROID_native_fence_sync export: fd after rendering, poll() until
 *    signalled (time reported), eglClientWaitSync on it.
 * With --vk (VK_DRIVER_FILES = our NVK ICD):
 * 4. GL export -> VkImage import (explicit modifier + pitch) -> copy to a
 *    buffer: same pixels.
 * 5. VkImage (modifier list = GL's, then LINEAR only) exported -> EGLImage:
 *    Vulkan clears it after a long chain of large clears and signals a
 *    SYNC_FD semaphore; GL imports the sync_file, eglWaitSyncKHR, reads the
 *    image: must be the final Vulkan colour (the fd is checked pending at
 *    the time of the wait, so the ordering is really exercised).
 * Exit 0 only if every check passed.
 */
#define _GNU_SOURCE
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>
#include <drm_fourcc.h>
#include <poll.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifndef NVRM_NO_VULKAN
#include <vulkan/vulkan.h>
#endif

#define W 64
#define H 64

static int failures;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } \
                           else { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)

static PFNEGLCREATEIMAGEKHRPROC create_image;
static PFNEGLDESTROYIMAGEKHRPROC destroy_image;
static PFNGLEGLIMAGETARGETTEXTURE2DOESPROC image_target;
static EGLDisplay dpy;

static double
now_ms(void)
{
   struct timespec t;
   clock_gettime(CLOCK_MONOTONIC, &t);
   return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static const uint8_t quad_colors[4][4] = {
   { 255, 0, 0, 255 }, { 0, 255, 0, 255 }, { 0, 0, 255, 255 }, { 16, 32, 48, 64 },
};

static void
expected_pattern(uint8_t *px, bool patched)
{
   for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
         int q = (y >= H / 2) * 2 + (x >= W / 2);
         const uint8_t *c = quad_colors[q];
         if (patched && x >= 8 && x < 24 && y >= 40 && y < 56) {
            static const uint8_t p[4] = { 200, 100, 50, 255 };
            c = p;
         }
         memcpy(px + 4 * (y * W + x), c, 4);
      }
}

static int
diff(const uint8_t *a, const uint8_t *b, int n)
{
   int d = 0;
   for (int i = 0; i < n; i++)
      d += memcmp(a + 4 * i, b + 4 * i, 4) != 0;
   return d;
}

static GLuint
fbo_for(GLuint tex)
{
   GLuint fbo;
   glGenFramebuffers(1, &fbo);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
   return fbo;
}

static void
read_tex(GLuint tex, uint8_t *px)
{
   GLuint fbo = fbo_for(tex);
   if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      printf("note: framebuffer for readback incomplete\n");
   glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glDeleteFramebuffers(1, &fbo);
}

/* Draw a full-screen triangle that texelFetches tex into a new RGBA8
 * texture and read that back: exercises the sampler path of an import. */
static void
sample_tex(GLuint tex, uint8_t *px)
{
   static GLuint prog;
   if (!prog) {
      const char *vs = "#version 300 es\nvoid main(){vec2 p=vec2(gl_VertexID==1?3.0:-1.0,"
                       "gl_VertexID==2?3.0:-1.0);gl_Position=vec4(p,0.0,1.0);}\n";
      const char *fs = "#version 300 es\nprecision highp float;uniform highp sampler2D t;"
                       "out vec4 o;void main(){o=texelFetch(t,ivec2(gl_FragCoord.xy),0);}\n";
      GLuint s[2] = { glCreateShader(GL_VERTEX_SHADER), glCreateShader(GL_FRAGMENT_SHADER) };
      glShaderSource(s[0], 1, &vs, NULL);
      glShaderSource(s[1], 1, &fs, NULL);
      prog = glCreateProgram();
      for (int i = 0; i < 2; i++) {
         glCompileShader(s[i]);
         glAttachShader(prog, s[i]);
      }
      glLinkProgram(prog);
   }
   GLuint dst;
   glGenTextures(1, &dst);
   glBindTexture(GL_TEXTURE_2D, dst);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, H);
   GLuint fbo = fbo_for(dst);
   glViewport(0, 0, W, H);
   glUseProgram(prog);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, tex);
   /* single level: a mipmapping min filter would make it incomplete */
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   glUniform1i(glGetUniformLocation(prog, "t"), 0);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glDeleteFramebuffers(1, &fbo);
   glDeleteTextures(1, &dst);
}

static EGLImageKHR
import_dmabuf(int fd, uint32_t fourcc, uint32_t stride, uint32_t offset, uint64_t mod)
{
   const EGLint attrs[] = {
      EGL_WIDTH, W, EGL_HEIGHT, H, EGL_LINUX_DRM_FOURCC_EXT, (EGLint)fourcc,
      EGL_DMA_BUF_PLANE0_FD_EXT, fd, EGL_DMA_BUF_PLANE0_OFFSET_EXT, (EGLint)offset,
      EGL_DMA_BUF_PLANE0_PITCH_EXT, (EGLint)stride,
      EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, (EGLint)(mod & 0xffffffff),
      EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, (EGLint)(mod >> 32), EGL_NONE,
   };
   return create_image(dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, attrs);
}

static GLuint
tex_from_image(EGLImageKHR img)
{
   GLuint t;
   glGenTextures(1, &t);
   glBindTexture(GL_TEXTURE_2D, t);
   image_target(GL_TEXTURE_2D, img);
   return t;
}

/* ------------------------------------------------------------------ Vulkan */
#ifndef NVRM_NO_VULKAN

static struct {
   VkInstance inst;
   VkPhysicalDevice pd;
   VkDevice dev;
   VkQueue q;
   uint32_t qf;
   VkCommandPool pool;
   PFN_vkGetMemoryFdKHR get_memory_fd;
   PFN_vkGetSemaphoreFdKHR get_semaphore_fd;
   PFN_vkGetImageDrmFormatModifierPropertiesEXT get_mod_props;
} vk;

#define VKCHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
   printf("FAIL: %s = %d\n", #x, r_); failures++; return false; } } while (0)

static bool
vk_init(void)
{
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_2 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
   VKCHECK(vkCreateInstance(&ici, NULL, &vk.inst));
   uint32_t n = 8;
   VkPhysicalDevice pds[8];
   vkEnumeratePhysicalDevices(vk.inst, &n, pds);
   for (uint32_t i = 0; i < n && !vk.pd; i++) {
      VkPhysicalDeviceProperties p;
      vkGetPhysicalDeviceProperties(pds[i], &p);
      printf("vk device %u: %s vendor 0x%x driver 0x%x\n", i, p.deviceName, p.vendorID,
             p.driverVersion);
      if (p.vendorID == 0x10de)
         vk.pd = pds[i];
   }
   if (!vk.pd) {
      printf("FAIL: no NVIDIA Vulkan device\n");
      failures++;
      return false;
   }
   VkPhysicalDeviceDriverProperties drv = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
   VkPhysicalDeviceProperties2 p2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                      .pNext = &drv };
   vkGetPhysicalDeviceProperties2(vk.pd, &p2);
   printf("vk driver: %s (%s)\n", drv.driverName, drv.driverInfo);
   const float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                   .queueFamilyIndex = 0, .queueCount = 1,
                                   .pQueuePriorities = &prio };
   const char *exts[] = { "VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf",
                          "VK_EXT_image_drm_format_modifier", "VK_KHR_external_semaphore_fd" };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
                              .enabledExtensionCount = 4, .ppEnabledExtensionNames = exts };
   VKCHECK(vkCreateDevice(vk.pd, &dci, NULL, &vk.dev));
   vkGetDeviceQueue(vk.dev, 0, 0, &vk.q);
   VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                   .queueFamilyIndex = 0 };
   VKCHECK(vkCreateCommandPool(vk.dev, &pci, NULL, &vk.pool));
   vk.get_memory_fd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(vk.dev, "vkGetMemoryFdKHR");
   vk.get_semaphore_fd = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(vk.dev, "vkGetSemaphoreFdKHR");
   vk.get_mod_props = (PFN_vkGetImageDrmFormatModifierPropertiesEXT)
      vkGetDeviceProcAddr(vk.dev, "vkGetImageDrmFormatModifierPropertiesEXT");
   return true;
}

static uint32_t
mem_type(uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(vk.pd, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   return UINT32_MAX;
}

static bool
vk_buffer(VkDeviceSize size, VkBuffer *buf, VkDeviceMemory *mem, void **map)
{
   VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                              .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT };
   VKCHECK(vkCreateBuffer(vk.dev, &bci, NULL, buf));
   VkMemoryRequirements r;
   vkGetBufferMemoryRequirements(vk.dev, *buf, &r);
   VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = r.size,
      .memoryTypeIndex = mem_type(r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
   VKCHECK(vkAllocateMemory(vk.dev, &ai, NULL, mem));
   VKCHECK(vkBindBufferMemory(vk.dev, *buf, *mem, 0));
   VKCHECK(vkMapMemory(vk.dev, *mem, 0, size, 0, map));
   return true;
}

static VkCommandBuffer
vk_cmd(void)
{
   VkCommandBuffer cb;
   VkCommandBufferAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                      .commandPool = vk.pool, .commandBufferCount = 1 };
   vkAllocateCommandBuffers(vk.dev, &ai, &cb);
   VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   vkBeginCommandBuffer(cb, &bi);
   return cb;
}

static void
barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to)
{
   VkImageMemoryBarrier b = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
      .oldLayout = from, .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
}

/* 4: import GL's dma-buf as a VkImage and copy it out. */
static bool
vk_import_and_read(int fd, uint32_t stride, uint64_t mod, uint8_t *px)
{
   VkSubresourceLayout plane = { .offset = 0, .rowPitch = stride };
   VkImageDrmFormatModifierExplicitCreateInfoEXT emi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
      .drmFormatModifier = mod, .drmFormatModifierPlaneCount = 1, .pPlaneLayouts = &plane };
   VkExternalMemoryImageCreateInfo emci = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, .pNext = &emi,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
   VkImageCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &emci,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = { W, H, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   VkImage img;
   VKCHECK(vkCreateImage(vk.dev, &ici, NULL, &img));
   VkMemoryRequirements r;
   vkGetImageMemoryRequirements(vk.dev, img, &r);
   VkMemoryDedicatedAllocateInfo ded = { .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                                         .image = img };
   VkImportMemoryFdInfoKHR imp = { .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
      .pNext = &ded, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
      .fd = dup(fd) };
   VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &imp,
      .allocationSize = r.size, .memoryTypeIndex = mem_type(r.memoryTypeBits, 0) };
   VkDeviceMemory mem;
   VKCHECK(vkAllocateMemory(vk.dev, &ai, NULL, &mem));
   VKCHECK(vkBindImageMemory(vk.dev, img, mem, 0));
   VkBuffer buf;
   VkDeviceMemory bmem;
   void *map;
   if (!vk_buffer(W * H * 4, &buf, &bmem, &map))
      return false;
   VkCommandBuffer cb = vk_cmd();
   barrier(cb, img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
   VkBufferImageCopy c = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                           .imageExtent = { W, H, 1 } };
   vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &c);
   vkEndCommandBuffer(cb);
   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                       .pCommandBuffers = &cb };
   VKCHECK(vkQueueSubmit(vk.q, 1, &si, VK_NULL_HANDLE));
   VKCHECK(vkQueueWaitIdle(vk.q));
   memcpy(px, map, W * H * 4);
   vkDestroyBuffer(vk.dev, buf, NULL);
   vkFreeMemory(vk.dev, bmem, NULL);
   vkDestroyImage(vk.dev, img, NULL);
   vkFreeMemory(vk.dev, mem, NULL);
   return true;
}

/* 5: Vulkan producer. Exports the image (fd, pitch, modifier) and a SYNC_FD
 * for the final clear, which runs behind `chain` 4096x4096 clears. */
static bool
vk_produce(const uint64_t *mods, uint32_t mod_count, const VkClearColorValue *color,
           int chain, int *mem_fd, uint32_t *pitch, uint64_t *mod, int *sync_fd)
{
   VkImageDrmFormatModifierListCreateInfoEXT ml = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
      .drmFormatModifierCount = mod_count, .pDrmFormatModifiers = mods };
   VkExternalMemoryImageCreateInfo emci = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, .pNext = &ml,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
   VkImageCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &emci,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = { W, H, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
      .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   VkImage img;
   VKCHECK(vkCreateImage(vk.dev, &ici, NULL, &img));
   VkImageDrmFormatModifierPropertiesEXT mp = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT };
   VKCHECK(vk.get_mod_props(vk.dev, img, &mp));
   *mod = mp.drmFormatModifier;
   VkImageSubresource sub = { VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT, 0, 0 };
   VkSubresourceLayout lay;
   vkGetImageSubresourceLayout(vk.dev, img, &sub, &lay);
   *pitch = lay.rowPitch;
   VkMemoryRequirements r;
   vkGetImageMemoryRequirements(vk.dev, img, &r);
   VkMemoryDedicatedAllocateInfo ded = { .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                                         .image = img };
   VkExportMemoryAllocateInfo exp = { .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
      .pNext = &ded, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
   VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &exp,
      .allocationSize = r.size,
      .memoryTypeIndex = mem_type(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) };
   VkDeviceMemory mem;
   VKCHECK(vkAllocateMemory(vk.dev, &ai, NULL, &mem));
   VKCHECK(vkBindImageMemory(vk.dev, img, mem, 0));
   VkMemoryGetFdInfoKHR gfi = { .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = mem,
                                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
   VKCHECK(vk.get_memory_fd(vk.dev, &gfi, mem_fd));

   /* Busy work: a large scratch image cleared `chain` times. */
   VkImageCreateInfo big = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R32G32B32A32_SFLOAT,
      .extent = { 4096, 4096, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   VkImage scratch;
   VKCHECK(vkCreateImage(vk.dev, &big, NULL, &scratch));
   vkGetImageMemoryRequirements(vk.dev, scratch, &r);
   VkMemoryAllocateInfo sai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = r.size,
      .memoryTypeIndex = mem_type(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) };
   VkDeviceMemory smem;
   VKCHECK(vkAllocateMemory(vk.dev, &sai, NULL, &smem));
   VKCHECK(vkBindImageMemory(vk.dev, scratch, smem, 0));

   VkCommandBuffer cb = vk_cmd();
   barrier(cb, scratch, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
   VkImageSubresourceRange rng = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
   for (int i = 0; i < chain; i++) {
      VkClearColorValue c = { .float32 = { i, 0, 0, 1 } };
      vkCmdClearColorImage(cb, scratch, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c, 1, &rng);
      barrier(cb, scratch, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
   }
   barrier(cb, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
   vkCmdClearColorImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, color, 1, &rng);
   barrier(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
   vkEndCommandBuffer(cb);

   VkExportSemaphoreCreateInfo esci = { .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT };
   VkSemaphoreCreateInfo sci = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &esci };
   VkSemaphore sem;
   VKCHECK(vkCreateSemaphore(vk.dev, &sci, NULL, &sem));
   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                       .pCommandBuffers = &cb, .signalSemaphoreCount = 1,
                       .pSignalSemaphores = &sem };
   VKCHECK(vkQueueSubmit(vk.q, 1, &si, VK_NULL_HANDLE));
   VkSemaphoreGetFdInfoKHR sgi = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
      .semaphore = sem, .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT };
   VKCHECK(vk.get_semaphore_fd(vk.dev, &sgi, sync_fd));
   /* image/memory/scratch intentionally kept alive until exit */
   return true;
}


/* 6: window-buffer formats. Vulkan allocates an exportable image of `fmt`
 * from GL's modifier list for `fourcc` (as our minigbm backend allocates
 * through NVK), GL renders into it (FBO clear), Vulkan reads it back and
 * compares with its own clear of the same colour into an optimal image. */
static bool
vk_copy_out(VkImage img, VkImageLayout layout, VkDeviceSize size, void *out)
{
   VkBuffer buf;
   VkDeviceMemory bmem;
   void *map;
   if (!vk_buffer(size, &buf, &bmem, &map))
      return false;
   VkCommandBuffer cb = vk_cmd();
   barrier(cb, img, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
   VkBufferImageCopy c = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
                           .imageExtent = { W, H, 1 } };
   vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &c);
   vkEndCommandBuffer(cb);
   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                       .pCommandBuffers = &cb };
   VKCHECK(vkQueueSubmit(vk.q, 1, &si, VK_NULL_HANDLE));
   VKCHECK(vkQueueWaitIdle(vk.q));
   memcpy(out, map, size);
   vkDestroyBuffer(vk.dev, buf, NULL);
   vkFreeMemory(vk.dev, bmem, NULL);
   return true;
}

static bool
vk_window_format(uint32_t fourcc, VkFormat fmt, unsigned bpp, const char *name,
                 PFNEGLQUERYDMABUFMODIFIERSEXTPROC query_mods)
{
   uint64_t mods[16];
   EGLBoolean ext[16];
   EGLint n = 0;
   query_mods(dpy, fourcc, 16, mods, ext, &n);
   VkImageDrmFormatModifierListCreateInfoEXT ml = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
      .drmFormatModifierCount = n, .pDrmFormatModifiers = mods };
   VkExternalMemoryImageCreateInfo emci = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, .pNext = &ml,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
   VkImageCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &emci,
      .imageType = VK_IMAGE_TYPE_2D, .format = fmt, .extent = { W, H, 1 }, .mipLevels = 1,
      .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   VkImage img, ref;
   VKCHECK(vkCreateImage(vk.dev, &ici, NULL, &img));
   VkImageDrmFormatModifierPropertiesEXT mp = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT };
   VKCHECK(vk.get_mod_props(vk.dev, img, &mp));
   VkImageSubresource sub = { VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT, 0, 0 };
   VkSubresourceLayout lay;
   vkGetImageSubresourceLayout(vk.dev, img, &sub, &lay);
   VkMemoryRequirements r;
   vkGetImageMemoryRequirements(vk.dev, img, &r);
   VkMemoryDedicatedAllocateInfo ded = { .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                                         .image = img };
   VkExportMemoryAllocateInfo exp = { .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
      .pNext = &ded, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
   VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &exp,
      .allocationSize = r.size,
      .memoryTypeIndex = mem_type(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) };
   VkDeviceMemory mem;
   VKCHECK(vkAllocateMemory(vk.dev, &ai, NULL, &mem));
   VKCHECK(vkBindImageMemory(vk.dev, img, mem, 0));
   int fd = -1;
   VkMemoryGetFdInfoKHR gfi = { .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = mem,
                                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
   VKCHECK(vk.get_memory_fd(vk.dev, &gfi, &fd));

   EGLImageKHR eimg = import_dmabuf(fd, fourcc, lay.rowPitch, 0, mp.drmFormatModifier);
   close(fd);
   if (eimg == EGL_NO_IMAGE_KHR) {
      printf("FAIL: %s modifier 0x%016llx pitch %llu: EGL import 0x%x\n", name,
             (unsigned long long)mp.drmFormatModifier, (unsigned long long)lay.rowPitch,
             eglGetError());
      failures++;
      return false;
   }
   GLuint tex = tex_from_image(eimg);
   GLuint fbo = fbo_for(tex);
   const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
   glViewport(0, 0, W, H);
   glClearColor(1.0f, 0.0f, 1.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glEnable(GL_SCISSOR_TEST);
   glScissor(4, 4, 8, 8);
   glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glDisable(GL_SCISSOR_TEST);
   glFinish();
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glDeleteFramebuffers(1, &fbo);

   ici.pNext = NULL;
   ici.tiling = VK_IMAGE_TILING_OPTIMAL;
   VKCHECK(vkCreateImage(vk.dev, &ici, NULL, &ref));
   vkGetImageMemoryRequirements(vk.dev, ref, &r);
   VkMemoryAllocateInfo rai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = r.size,
      .memoryTypeIndex = mem_type(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) };
   VkDeviceMemory rmem;
   VKCHECK(vkAllocateMemory(vk.dev, &rai, NULL, &rmem));
   VKCHECK(vkBindImageMemory(vk.dev, ref, rmem, 0));
   VkCommandBuffer cb = vk_cmd();
   VkImageSubresourceRange rng = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
   barrier(cb, ref, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
   VkClearColorValue magenta = { .float32 = { 1, 0, 1, 1 } }, green = { .float32 = { 0, 1, 0, 1 } };
   vkCmdClearColorImage(cb, ref, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &magenta, 1, &rng);
   barrier(cb, ref, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
   VkClearAttachment unused; (void)unused;
   vkEndCommandBuffer(cb);
   VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                       .pCommandBuffers = &cb };
   VKCHECK(vkQueueSubmit(vk.q, 1, &si, VK_NULL_HANDLE));
   VKCHECK(vkQueueWaitIdle(vk.q));
   static uint8_t got[W * H * 8], want[W * H * 8], g[8];
   if (!vk_copy_out(ref, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, W * H * bpp, want))
      return false;
   /* green texel reference: clear a second time (whole image) */
   cb = vk_cmd();
   vkCmdClearColorImage(cb, ref, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, &green, 1, &rng);
   vkEndCommandBuffer(cb);
   si.pCommandBuffers = &cb;
   VKCHECK(vkQueueSubmit(vk.q, 1, &si, VK_NULL_HANDLE));
   VKCHECK(vkQueueWaitIdle(vk.q));
   static uint8_t greenimg[W * H * 8];
   if (!vk_copy_out(ref, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, W * H * bpp, greenimg))
      return false;
   memcpy(g, greenimg, bpp);
   for (int y = 4; y < 12; y++)
      for (int x = 4; x < 12; x++)
         memcpy(want + (y * W + x) * bpp, g, bpp); /* GL y=4..11 = rows 4..11 bottom-up */
   if (!vk_copy_out(img, VK_IMAGE_LAYOUT_GENERAL, W * H * bpp, got))
      return false;
   int bad = 0;
   for (int i = 0; i < W * H; i++) {
      if (fourcc == DRM_FORMAT_XBGR8888) /* X channel undefined */
         bad += memcmp(got + 4 * i, want + 4 * i, 3) != 0;
      else
         bad += memcmp(got + bpp * i, want + bpp * i, bpp) != 0;
   }
   CHECK(status == GL_FRAMEBUFFER_COMPLETE && !bad,
         "window format %s: modifier 0x%016llx pitch %llu, GL renders, Vulkan reads: FBO 0x%x, "
         "%d px differ", name, (unsigned long long)mp.drmFormatModifier,
         (unsigned long long)lay.rowPitch, status, bad);
   return true;
}
#endif /* NVRM_NO_VULKAN */

static bool
fd_signaled(int fd)
{
   struct pollfd p = { .fd = fd, .events = POLLIN };
   return poll(&p, 1, 0) == 1;
}

int
main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "usage: %s <vendor-substring> [--vk]\n", argv[0]);
      return 2;
   }
   const bool use_vk = argc > 2 && !strcmp(argv[2], "--vk");
   PFNEGLGETPLATFORMDISPLAYEXTPROC get_display =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
   dpy = get_display(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
   if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, NULL, NULL)) {
      printf("FAIL: eglInitialize 0x%x\n", eglGetError());
      return 1;
   }
   const char *exts = eglQueryString(dpy, EGL_EXTENSIONS);
   printf("EGL_EXTENSIONS: %s\n", exts);
   static const char *need[] = { "EGL_EXT_image_dma_buf_import ",
      "EGL_EXT_image_dma_buf_import_modifiers", "EGL_MESA_image_dma_buf_export",
      "EGL_ANDROID_native_fence_sync", "EGL_KHR_wait_sync", "EGL_KHR_gl_texture_2D_image" };
   for (unsigned i = 0; i < sizeof(need) / sizeof(need[0]); i++)
      CHECK(strstr(exts, need[i]), "extension %s", need[i]);
   eglBindAPI(EGL_OPENGL_ES_API);
   const EGLint ca[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
   EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ca);
   if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
      printf("FAIL: make current\n");
      return 1;
   }
   const char *renderer = (const char *)glGetString(GL_RENDERER);
   const char *vendor = (const char *)glGetString(GL_VENDOR);
   printf("GL_VENDOR '%s' GL_RENDERER '%s'\n", vendor, renderer);
   if (!strstr(vendor, argv[1]) && !strstr(renderer, argv[1])) {
      printf("FAIL: not on %s\n", argv[1]);
      return 1;
   }
   create_image = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
   destroy_image = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
   image_target = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
   PFNEGLQUERYDMABUFMODIFIERSEXTPROC query_mods =
      (PFNEGLQUERYDMABUFMODIFIERSEXTPROC)eglGetProcAddress("eglQueryDmaBufModifiersEXT");
   PFNEGLEXPORTDMABUFIMAGEQUERYMESAPROC export_query =
      (PFNEGLEXPORTDMABUFIMAGEQUERYMESAPROC)eglGetProcAddress("eglExportDMABUFImageQueryMESA");
   PFNEGLEXPORTDMABUFIMAGEMESAPROC export_image =
      (PFNEGLEXPORTDMABUFIMAGEMESAPROC)eglGetProcAddress("eglExportDMABUFImageMESA");
   PFNEGLCREATESYNCKHRPROC create_sync = (PFNEGLCREATESYNCKHRPROC)eglGetProcAddress("eglCreateSyncKHR");
   PFNEGLWAITSYNCKHRPROC wait_sync = (PFNEGLWAITSYNCKHRPROC)eglGetProcAddress("eglWaitSyncKHR");
   PFNEGLCLIENTWAITSYNCKHRPROC client_wait =
      (PFNEGLCLIENTWAITSYNCKHRPROC)eglGetProcAddress("eglClientWaitSyncKHR");
   PFNEGLDUPNATIVEFENCEFDANDROIDPROC dup_fence =
      (PFNEGLDUPNATIVEFENCEFDANDROIDPROC)eglGetProcAddress("eglDupNativeFenceFDANDROID");

   /* 1. modifiers */
   uint64_t gl_mods[16];
   EGLint gl_mod_count = 0;
   static const struct { uint32_t fourcc; const char *name; } fmts[] = {
      { DRM_FORMAT_ABGR8888, "ABGR8888 (RGBA_8888)" }, { DRM_FORMAT_XBGR8888, "XBGR8888 (RGBX_8888)" },
      { DRM_FORMAT_RGB565, "RGB565" }, { DRM_FORMAT_ARGB8888, "ARGB8888 (BGRA_8888)" },
   };
   for (unsigned f = 0; f < 4; f++) {
      uint64_t m[16];
      EGLBoolean ext[16];
      EGLint n = 0;
      query_mods(dpy, fmts[f].fourcc, 16, m, ext, &n);
      printf("modifiers %s: %d:", fmts[f].name, n);
      for (int i = 0; i < n; i++)
         printf(" 0x%016llx%s", (unsigned long long)m[i], ext[i] ? "(ext)" : "");
      printf("\n");
      /* nil's list (= NVK's): 6 block-linear heights + LINEAR for every one
       * of these formats. Since Mesa 26 nil also encodes Blackwell's 8/16-bit
       * GOB types (sector layouts Blackwell8Bpp/16Bpp, nil/modifiers.rs
       * GOB_TYPE_MODIFIER_INFOS), so RGB565 is no longer LINEAR-only on
       * Blackwell. */
      CHECK(n == 7 && m[6] == DRM_FORMAT_MOD_LINEAR,
            "%s: 6 block-linear + LINEAR modifiers", fmts[f].name);
      if (f == 0) {
         memcpy(gl_mods, m, sizeof(m));
         gl_mod_count = n;
      }
   }

   /* 2. GL export -> GL import */
   uint8_t want[W * H * 4], got[W * H * 4];
   GLuint texA;
   glGenTextures(1, &texA);
   glBindTexture(GL_TEXTURE_2D, texA);
   glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, W, H);
   GLuint fboA = fbo_for(texA);
   glEnable(GL_SCISSOR_TEST);
   for (int q = 0; q < 4; q++) {
      glScissor((q & 1) * W / 2, (q >> 1) * H / 2, W / 2, H / 2);
      glClearColor(quad_colors[q][0] / 255.0f, quad_colors[q][1] / 255.0f,
                   quad_colors[q][2] / 255.0f, quad_colors[q][3] / 255.0f);
      glClear(GL_COLOR_BUFFER_BIT);
   }
   glDisable(GL_SCISSOR_TEST);
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glFinish();
   expected_pattern(want, false);
   read_tex(texA, got);
   CHECK(!diff(want, got, W * H), "exporter pattern (%d px differ)", diff(want, got, W * H));

   EGLImageKHR imgA = create_image(dpy, ctx, EGL_GL_TEXTURE_2D_KHR,
                                   (EGLClientBuffer)(uintptr_t)texA, NULL);
   CHECK(imgA != EGL_NO_IMAGE_KHR, "EGLImage from texture (0x%x)", eglGetError());
   int fourcc = 0, planes = 0;
   EGLuint64KHR modifier = 0;
   EGLBoolean ok = export_query(dpy, imgA, &fourcc, &planes, &modifier);
   CHECK(ok && planes == 1, "export query: fourcc %.4s planes %d modifier 0x%016llx",
         (char *)&fourcc, planes, (unsigned long long)modifier);
   int fd = -1;
   EGLint stride = 0, offset = 0;
   ok = export_image(dpy, imgA, &fd, &stride, &offset);
   CHECK(ok && fd >= 0, "export: fd %d stride %d offset %d", fd, stride, offset);
   if (fd < 0)
      return 1;

   EGLImageKHR imgB = import_dmabuf(fd, fourcc, stride, offset, modifier);
   CHECK(imgB != EGL_NO_IMAGE_KHR, "import of own export (0x%x)", eglGetError());
   GLuint texB = tex_from_image(imgB);
   memset(got, 0, sizeof(got));
   read_tex(texB, got);
   CHECK(!diff(want, got, W * H), "GL->GL readback through import (%d px differ)",
         diff(want, got, W * H));
   memset(got, 0, sizeof(got));
   sample_tex(texB, got);
   CHECK(!diff(want, got, W * H), "GL->GL sampled through import (%d px differ)",
         diff(want, got, W * H));
   /* write through the import, read through the exporter */
   GLuint fboB = fbo_for(texB);
   glEnable(GL_SCISSOR_TEST);
   glScissor(8, 40, 16, 16);
   glClearColor(200 / 255.0f, 100 / 255.0f, 50 / 255.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glDisable(GL_SCISSOR_TEST);
   glBindFramebuffer(GL_FRAMEBUFFER, 0);
   glFinish();
   expected_pattern(want, true);
   read_tex(texA, got);
   CHECK(!diff(want, got, W * H), "write through import seen by exporter (%d px differ)",
         diff(want, got, W * H));
   (void)fboA;
   (void)fboB;

   /* 3. native fence export */
   {
      GLuint big;
      glGenTextures(1, &big);
      glBindTexture(GL_TEXTURE_2D, big);
      glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 4096, 4096);
      fbo_for(big);
      for (int i = 0; i < 50; i++) {
         glClearColor(i / 50.0f, 0, 0, 1);
         glClear(GL_COLOR_BUFFER_BIT);
      }
      double t0 = now_ms();
      EGLSyncKHR s = create_sync(dpy, EGL_SYNC_NATIVE_FENCE_ANDROID, NULL);
      CHECK(s != EGL_NO_SYNC_KHR, "eglCreateSyncKHR(NATIVE_FENCE) (0x%x)", eglGetError());
      glFlush();
      int ffd = dup_fence(dpy, s);
      double t1 = now_ms();
      CHECK(ffd >= 0, "eglDupNativeFenceFDANDROID fd %d (create+flush+dup %.3f ms)", ffd, t1 - t0);
      struct pollfd p = { .fd = ffd, .events = POLLIN };
      int pr = poll(&p, 1, 5000);
      double t2 = now_ms();
      CHECK(pr == 1, "fence fd signalled after %.3f ms (poll %d)", t2 - t1, pr);
      EGLint r = client_wait(dpy, s, 0, EGL_FOREVER_KHR);
      CHECK(r == EGL_CONDITION_SATISFIED_KHR, "eglClientWaitSync on native fence");
      /* re-import the exported fd as a GL fence and wait on it */
      const EGLint sa[] = { EGL_SYNC_NATIVE_FENCE_FD_ANDROID, dup(ffd), EGL_NONE };
      EGLSyncKHR s2 = create_sync(dpy, EGL_SYNC_NATIVE_FENCE_ANDROID, sa);
      CHECK(s2 != EGL_NO_SYNC_KHR && wait_sync(dpy, s2, 0) == EGL_TRUE,
            "import own fence fd + eglWaitSyncKHR");
      glClear(GL_COLOR_BUFFER_BIT);
      glFinish();
      close(ffd);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
   }

#ifdef NVRM_NO_VULKAN
   (void)fd_signaled;
   (void)gl_mod_count;
   if (use_vk) {
      printf("FAIL: built without Vulkan (NVRM_NO_VULKAN)\n");
      failures++;
   }
#else
   if (use_vk && vk_init()) {
      /* 4. GL -> Vulkan */
      memset(got, 0, sizeof(got));
      if (vk_import_and_read(fd, stride, modifier, got))
         CHECK(!diff(want, got, W * H), "GL->Vulkan import, modifier 0x%016llx (%d px differ)",
               (unsigned long long)modifier, diff(want, got, W * H));
      /* 5. Vulkan -> GL with a SYNC_FD and eglWaitSyncKHR */
      for (int pass = 0; pass < 2; pass++) {
         const uint64_t linear = DRM_FORMAT_MOD_LINEAR;
         uint64_t bl[16];
         uint32_t bl_count = 0;
         for (int i = 0; i < gl_mod_count; i++)
            if (gl_mods[i] != DRM_FORMAT_MOD_LINEAR)
               bl[bl_count++] = gl_mods[i];
         const VkClearColorValue color = { .float32 = { pass ? 0.2f : 0.8f, 0.4f,
                                                        pass ? 0.6f : 0.0f, 1.0f } };
         const uint8_t expect[4] = { pass ? 51 : 204, 102, pass ? 153 : 0, 255 };
         int mfd = -1, sfd = -1;
         uint32_t pitch = 0;
         uint64_t mod = 0;
         double t0 = now_ms();
         if (!vk_produce(pass ? &linear : bl, pass ? 1 : bl_count, &color, 400, &mfd, &pitch,
                         &mod, &sfd))
            continue;
         printf("vk producer: modifier 0x%016llx pitch %u, submit %.3f ms\n",
                (unsigned long long)mod, pitch, now_ms() - t0);
         EGLImageKHR vimg = import_dmabuf(mfd, DRM_FORMAT_ABGR8888, pitch, 0, mod);
         CHECK(vimg != EGL_NO_IMAGE_KHR, "Vulkan image (modifier 0x%016llx) as EGLImage (0x%x)",
               (unsigned long long)mod, eglGetError());
         if (vimg == EGL_NO_IMAGE_KHR)
            continue;
         GLuint vtex = tex_from_image(vimg);
         const bool pending = !fd_signaled(sfd);
         const EGLint sa[] = { EGL_SYNC_NATIVE_FENCE_FD_ANDROID, sfd, EGL_NONE };
         EGLSyncKHR s = create_sync(dpy, EGL_SYNC_NATIVE_FENCE_ANDROID, sa);
         double t1 = now_ms();
         CHECK(s != EGL_NO_SYNC_KHR && wait_sync(dpy, s, 0) == EGL_TRUE,
               "import Vulkan SYNC_FD + eglWaitSyncKHR (fd pending at wait: %s)",
               pending ? "yes" : "NO");
         memset(got, 0, sizeof(got));
         if (pass)
            sample_tex(vtex, got); /* LINEAR: sampler (pitch TIC) path */
         else
            read_tex(vtex, got);
         double t2 = now_ms();
         int bad = 0;
         for (int i = 0; i < W * H; i++)
            bad += memcmp(got + 4 * i, expect, 4) != 0;
         CHECK(!bad && pending, "Vulkan->GL %s after eglWaitSync: %d px differ from "
               "(%u,%u,%u,%u), wait+%s %.3f ms", pass ? "LINEAR sampled" : "block-linear read",
               bad, expect[0], expect[1], expect[2], expect[3], pass ? "sample" : "read", t2 - t1);
         if (bad)
            printf("   first px: %u %u %u %u\n", got[0], got[1], got[2], got[3]);
      }
      /* 6. window-buffer formats (minigbm whitelist) */
      static const struct { uint32_t fourcc; VkFormat fmt; unsigned bpp; const char *name; } wf[] = {
         { DRM_FORMAT_ABGR8888, VK_FORMAT_R8G8B8A8_UNORM, 4, "RGBA_8888" },
         { DRM_FORMAT_XBGR8888, VK_FORMAT_R8G8B8A8_UNORM, 4, "RGBX_8888" },
         { DRM_FORMAT_ARGB8888, VK_FORMAT_B8G8R8A8_UNORM, 4, "BGRA_8888" },
         { DRM_FORMAT_RGB565, VK_FORMAT_R5G6B5_UNORM_PACK16, 2, "RGB_565" },
         { DRM_FORMAT_ABGR16161616F, VK_FORMAT_R16G16B16A16_SFLOAT, 8, "RGBA_FP16" },
         { DRM_FORMAT_ABGR2101010, VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4, "RGBA_1010102" },
      };
      for (unsigned i = 0; i < sizeof(wf) / sizeof(wf[0]); i++)
         vk_window_format(wf[i].fourcc, wf[i].fmt, wf[i].bpp, wf[i].name, query_mods);
   }
#endif
   printf(failures ? "GL DMABUF FAIL (%d)\n" : "GL DMABUF PASS\n", failures);
   return failures != 0;
}
