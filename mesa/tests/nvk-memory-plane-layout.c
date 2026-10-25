/* MEMORY_PLANE_i aspects on multi-planar LINEAR-modifier images (NV12/P010).
 *
 * NVK (upstream since Mesa 26.2, !40473) creates 2-plane YCbCr images with
 * DRM_FORMAT_MOD_LINEAR and an explicit per-plane layout; the Vulkan spec
 * (VK_EXT_image_drm_format_modifier) addresses those planes with
 * VK_IMAGE_ASPECT_MEMORY_PLANE_i_BIT_EXT in vkGetImageSubresourceLayout,
 * VkImagePlaneMemoryRequirementsInfo and VkBindImagePlaneMemoryInfo. Before
 * nvk-upstream-bfb90f05-memory-plane-aspects.patch MEMORY_PLANE_1 hit
 * assert(aspectMask == MEMORY_PLANE_0) in nvk_image_memory_aspects_to_plane
 * and, with NDEBUG, silently answered with plane 0. A DISJOINT plane's
 * explicit offset is reported as given (nvk-explicit-plane-layouts.patch;
 * Mesa 26.2 reported 0 while binding at the offset).
 *
 * For NV12 and P010 at two sizes, packed and DISJOINT: create the image with a
 * known explicit layout (pitches 256-B aligned, plane 1 placed one 64 KiB
 * page past where NVK would pack it, so a packed answer is distinguishable),
 * then check that both memory planes report exactly that offset and pitch,
 * plus per-plane requirements and binds through MEMORY_PLANE_i on DISJOINT.
 *
 * Real GPU, host NVK; the device is selected by driverID (MESA_NVK), never by
 * a node path. Build/run (repository root):
 *   cc -std=c11 -Wall -Wextra -Werror -I<mesa>/include \
 *     nvidia-prolapse/driver-source/native/mesa/tests/nvk-memory-plane-layout.c -ldl \
 *     -o nvk-memory-plane-layout
 *   LD_LIBRARY_PATH=<rust lib dir> ./nvk-memory-plane-layout <nvk-host>/src/nouveau/vulkan/libvulkan_nouveau.so
 */
#include <dlfcn.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#ifndef DRM_FORMAT_MOD_LINEAR
#define DRM_FORMAT_MOD_LINEAR 0ull
#endif

static int failures;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s: ", __FILE__, __LINE__, #c); \
   fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++; } } while (0)
#define VKCHECK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { \
   fprintf(stderr, "FATAL %s:%d %s -> %d\n", __FILE__, __LINE__, #call, r_); exit(2); } } while (0)

static PFN_vkGetInstanceProcAddr gipa;
static VkInstance inst;
static VkDevice dev;
static uint32_t mem_type_bits_all;
#define F(name) static PFN_vk##name p_##name;
F(CreateImage) F(DestroyImage) F(GetImageSubresourceLayout) F(GetImageMemoryRequirements2)
F(AllocateMemory) F(FreeMemory) F(BindImageMemory2) F(GetImageDrmFormatModifierPropertiesEXT)
#undef F

static uint64_t align64(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

struct fmt { VkFormat vk; const char *name; uint32_t bpp; /* bytes per luma sample */ };

static void
one(struct fmt f, uint32_t w, uint32_t h, int disjoint)
{
   const uint64_t page = 64 * 1024;
   VkSubresourceLayout planes[2] = {0};
   planes[0].rowPitch = align64((uint64_t)w * f.bpp, 256);
   planes[1].rowPitch = align64((uint64_t)w * f.bpp, 256); /* w/2 CbCr pairs */
   /* Packed: plane 1 one page beyond the page-aligned end of plane 0.
    * DISJOINT: each plane in its own binding; plane 1 at a non-zero offset
    * too so offset 0 (plane 0's answer) is wrong for it. */
   planes[0].offset = 0;
   planes[1].offset = disjoint ? page : align64(planes[0].rowPitch * h, page) + page;

   VkImageDrmFormatModifierExplicitCreateInfoEXT eci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
      .drmFormatModifier = DRM_FORMAT_MOD_LINEAR,
      .drmFormatModifierPlaneCount = 2,
      .pPlaneLayouts = planes,
   };
   VkImageCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &eci,
      .flags = disjoint ? VK_IMAGE_CREATE_DISJOINT_BIT : 0,
      .imageType = VK_IMAGE_TYPE_2D, .format = f.vk, .extent = {w, h, 1},
      .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage img;
   VKCHECK(p_CreateImage(dev, &ci, NULL, &img));

   VkImageDrmFormatModifierPropertiesEXT mp = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT };
   VKCHECK(p_GetImageDrmFormatModifierPropertiesEXT(dev, img, &mp));
   CHECK(mp.drmFormatModifier == DRM_FORMAT_MOD_LINEAR, "modifier %#" PRIx64, mp.drmFormatModifier);

   static const VkImageAspectFlagBits mem_plane[2] = {
      VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT, VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT };
   VkSubresourceLayout got[2];
   for (int p = 0; p < 2; p++) {
      VkImageSubresource sub = { .aspectMask = mem_plane[p] };
      p_GetImageSubresourceLayout(dev, img, &sub, &got[p]);
      printf("  %s %ux%u %s MEMORY_PLANE_%d: offset %" PRIu64 " (want %" PRIu64 ") rowPitch %" PRIu64
             " (want %" PRIu64 ") size %" PRIu64 "\n", f.name, w, h, disjoint ? "disjoint" : "packed", p,
             got[p].offset, planes[p].offset, got[p].rowPitch, planes[p].rowPitch, got[p].size);
      CHECK(got[p].offset == planes[p].offset, "%s plane %d offset %" PRIu64 " != %" PRIu64, f.name, p,
            got[p].offset, planes[p].offset);
      CHECK(got[p].rowPitch == planes[p].rowPitch, "%s plane %d rowPitch %" PRIu64 " != %" PRIu64, f.name, p,
            got[p].rowPitch, planes[p].rowPitch);
      /* luma h rows, CbCr h/2 rows (4:2:0) */
      const uint64_t rows = p ? (h + 1) / 2 : h;
      CHECK(got[p].size >= planes[p].rowPitch * (rows - 1) + (uint64_t)w * f.bpp,
            "%s plane %d size %" PRIu64 " too small", f.name, p, got[p].size);
   }

   VkDeviceMemory mem[2] = {0};
   if (!disjoint) {
      VkImageMemoryRequirementsInfo2 mi = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
                                            .image = img };
      VkMemoryRequirements2 mr = { .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
      p_GetImageMemoryRequirements2(dev, &mi, &mr);
      CHECK(mr.memoryRequirements.size >= got[1].offset + got[1].size,
            "%s packed size %" PRIu64 " < plane 1 end %" PRIu64, f.name, mr.memoryRequirements.size,
            got[1].offset + got[1].size);
      VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                  .allocationSize = mr.memoryRequirements.size,
                                  .memoryTypeIndex = __builtin_ctz(mr.memoryRequirements.memoryTypeBits) };
      VKCHECK(p_AllocateMemory(dev, &ai, NULL, &mem[0]));
      VkBindImageMemoryInfo bi = { .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
                                   .image = img, .memory = mem[0] };
      VKCHECK(p_BindImageMemory2(dev, 1, &bi));
   } else {
      /* Spec (VkImagePlaneMemoryRequirementsInfo / VkBindImagePlaneMemoryInfo
       * valid usage): DRM-modifier tiling addresses DISJOINT planes as
       * MEMORY_PLANE_i. The per-plane
       * requirement must match the PLANE_i answer our HAL/gralloc read. */
      static const VkImageAspectFlagBits fmt_plane[2] = {
         VK_IMAGE_ASPECT_PLANE_0_BIT, VK_IMAGE_ASPECT_PLANE_1_BIT };
      VkBindImagePlaneMemoryInfo bpi[2];
      VkBindImageMemoryInfo bi[2];
      for (int p = 0; p < 2; p++) {
         VkMemoryRequirements req[2];
         for (int k = 0; k < 2; k++) {
            VkImagePlaneMemoryRequirementsInfo pi = {
               .sType = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO,
               .planeAspect = k ? fmt_plane[p] : mem_plane[p] };
            VkImageMemoryRequirementsInfo2 mi = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
                                                  .pNext = &pi, .image = img };
            VkMemoryRequirements2 mr = { .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
            p_GetImageMemoryRequirements2(dev, &mi, &mr);
            req[k] = mr.memoryRequirements;
         }
         printf("  %s %ux%u disjoint plane %d req: MEMORY_PLANE size %" PRIu64 " align %" PRIu64
                " | PLANE size %" PRIu64 " align %" PRIu64 "\n", f.name, w, h, p, req[0].size,
                req[0].alignment, req[1].size, req[1].alignment);
         CHECK(req[0].size == req[1].size && req[0].alignment == req[1].alignment,
               "%s plane %d MEMORY_PLANE vs PLANE requirements differ", f.name, p);
         CHECK(req[0].size >= got[p].offset + got[p].size, "%s plane %d req size %" PRIu64
               " < layout end %" PRIu64, f.name, p, req[0].size, got[p].offset + got[p].size);
         VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                     .allocationSize = req[0].size,
                                     .memoryTypeIndex = __builtin_ctz(req[0].memoryTypeBits) };
         VKCHECK(p_AllocateMemory(dev, &ai, NULL, &mem[p]));
         bpi[p] = (VkBindImagePlaneMemoryInfo){ .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO,
                                                .planeAspect = mem_plane[p] };
         bi[p] = (VkBindImageMemoryInfo){ .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
                                          .pNext = &bpi[p], .image = img, .memory = mem[p] };
      }
      VKCHECK(p_BindImageMemory2(dev, 2, bi));
   }
   p_DestroyImage(dev, img, NULL);
   for (int p = 0; p < 2; p++)
      if (mem[p])
         p_FreeMemory(dev, mem[p], NULL);
}

int
main(int argc, char **argv)
{
   if (argc != 2) {
      fprintf(stderr, "usage: %s path/to/libvulkan_nouveau.so\n", argv[0]);
      return 2;
   }
   void *icd = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
   if (!icd) {
      fprintf(stderr, "FATAL dlopen %s: %s\n", argv[1], dlerror());
      return 2;
   }
   gipa = (PFN_vkGetInstanceProcAddr)dlsym(icd, "vk_icdGetInstanceProcAddr");
   if (!gipa) {
      fprintf(stderr, "FATAL %s has no vk_icdGetInstanceProcAddr\n", argv[1]);
      return 2;
   }
   PFN_vkCreateInstance create_instance = (PFN_vkCreateInstance)gipa(NULL, "vkCreateInstance");
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
   VKCHECK(create_instance(&ici, NULL, &inst));
#define I(name) PFN_vk##name i_##name = (PFN_vk##name)gipa(inst, "vk" #name)
   I(EnumeratePhysicalDevices); I(GetPhysicalDeviceProperties2); I(CreateDevice);
   I(GetDeviceProcAddr); I(DestroyDevice); I(DestroyInstance);
#undef I
   uint32_t n = 8;
   VkPhysicalDevice pdevs[8];
   VKCHECK(i_EnumeratePhysicalDevices(inst, &n, pdevs));
   VkPhysicalDevice pd = VK_NULL_HANDLE;
   for (uint32_t i = 0; i < n && !pd; i++) {
      VkPhysicalDeviceDriverProperties drv = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
      VkPhysicalDeviceProperties2 pp = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &drv };
      i_GetPhysicalDeviceProperties2(pdevs[i], &pp);
      if (pp.properties.vendorID == 0x10de && drv.driverID == VK_DRIVER_ID_MESA_NVK) {
         pd = pdevs[i];
         printf("device: %s (%s %s)\n", pp.properties.deviceName, drv.driverName, drv.driverInfo);
      }
   }
   if (!pd) {
      fprintf(stderr, "FATAL no NVIDIA device with driverID MESA_NVK\n");
      return 2;
   }
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
                                   .pQueuePriorities = &prio };
   const char *exts[] = { "VK_EXT_image_drm_format_modifier" };
   VkPhysicalDeviceVulkan11Features f11 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                                            .samplerYcbcrConversion = VK_TRUE };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f11,
                              .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
                              .enabledExtensionCount = 1, .ppEnabledExtensionNames = exts };
   VKCHECK(i_CreateDevice(pd, &dci, NULL, &dev));
#define D(name) p_##name = (PFN_vk##name)i_GetDeviceProcAddr(dev, "vk" #name); \
   if (!p_##name) { fprintf(stderr, "FATAL no vk" #name "\n"); return 2; }
   D(CreateImage) D(DestroyImage) D(GetImageSubresourceLayout) D(GetImageMemoryRequirements2)
   D(AllocateMemory) D(FreeMemory) D(BindImageMemory2) D(GetImageDrmFormatModifierPropertiesEXT)
#undef D
   (void)mem_type_bits_all;

   static const struct fmt fmts[] = {
      { VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, "NV12", 1 },
      { VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16, "P010", 2 },
   };
   static const uint32_t sizes[][2] = { {1920, 1080}, {1280, 720} };
   for (unsigned i = 0; i < 2; i++)
      for (unsigned s = 0; s < 2; s++)
         for (int disjoint = 0; disjoint < 2; disjoint++)
            one(fmts[i], sizes[s][0], sizes[s][1], disjoint);

   i_DestroyDevice(dev, NULL);
   i_DestroyInstance(inst, NULL);
   if (failures) {
      fprintf(stderr, "%d failures\n", failures);
      return 1;
   }
   printf("PASS nvk-memory-plane-layout (8 images: NV12/P010 x 2 sizes x packed/disjoint)\n");
   return 0;
}
