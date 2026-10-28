/* SPDX-License-Identifier: MIT
 *
 * YUV (NV12 / P010) EGLImage import + samplerExternalOES sampling on nvrm.
 *
 *   NVRM_YUV_VK_ICD=<libvulkan_nouveau.so> gl-yuv <vendor-substring>
 *
 * The dma-bufs come from our NVK (the ICD is dlopen'ed and driven through
 * vk_icdGetInstanceProcAddr, so the same source runs on glibc and on bionic
 * without a Vulkan loader), laid out exactly as our minigbm gralloc lays
 * out decoder output (minigbm 0004 + NVK's plane_layout: LINEAR, pitch
 * ALIGN(row bytes, 256), CbCr at pitch * even(luma rows); NVK's per-plane
 * rule leaves it packed):
 *   raw   - exportable host-visible VkDeviceMemory of the layout's size (the
 *           gralloc "R8 carrier"), filled through vkMapMemory;
 *   image - a VK multi-planar LINEAR image (G8_B8R8_2PLANE_420 /
 *           G10X6_B10X6R10X6_2PLANE_420) created with the explicit gralloc
 *           plane layouts, its dedicated memory exported.
 * Sizes 854x480, 1920x1080 (1088 luma rows allocated) and 640x362; each
 * import with BT601/BT709/BT2020 x limited/full hints, sampled NEAREST
 * through samplerExternalOES into RGBA8 and compared to a CPU reference
 * (ITU-R conversion; <= 2 LSB), plus one GL_LINEAR (bilinear chroma) run
 * per buffer. Same fd for both planes (gralloc) and,
 * once per size, a dup'ed fd for plane 1.
 * Negative cases (must fail with EGL_NO_IMAGE, no crash): NV12 with a
 * block-linear modifier, NV12 without a modifier, YUV420 (3-plane, not
 * accepted), CbCr offset off the 32 B plane alignment, CbCr plane past the
 * end of the dma-buf, pitch not a 32 B multiple.
 * Exit 0 only if every check passed ("GL YUV PASS").
 */
#define _GNU_SOURCE
#define VK_NO_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <dlfcn.h>
#include <drm_fourcc.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

static int failures;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); failures++; } \
                           else { printf("ok: " __VA_ARGS__); printf("\n"); } } while (0)

/* ---------------------------------------------------------------- Vulkan */
static struct {
   PFN_vkGetInstanceProcAddr gipa;
   VkInstance inst;
   VkPhysicalDevice pd;
   VkDevice dev;
   PFN_vkCreateInstance CreateInstance;
   PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices;
   PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties;
   PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
   PFN_vkCreateDevice CreateDevice;
   PFN_vkGetDeviceProcAddr GetDeviceProcAddr;
   PFN_vkAllocateMemory AllocateMemory;
   PFN_vkFreeMemory FreeMemory;
   PFN_vkMapMemory MapMemory;
   PFN_vkUnmapMemory UnmapMemory;
   PFN_vkCreateImage CreateImage;
   PFN_vkDestroyImage DestroyImage;
   PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
   PFN_vkGetImageSubresourceLayout GetImageSubresourceLayout;
   PFN_vkBindImageMemory BindImageMemory;
   PFN_vkGetMemoryFdKHR GetMemoryFdKHR;
} vk;

#define VKCHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
   printf("FAIL: %s = %d\n", #x, r_); failures++; return false; } } while (0)

static bool
vk_init(void)
{
   const char *icd = getenv("NVRM_YUV_VK_ICD");
   if (!icd) {
      printf("FAIL: NVRM_YUV_VK_ICD (path of our NVK libvulkan_nouveau.so) not set\n");
      failures++;
      return false;
   }
   void *lib = dlopen(icd, RTLD_NOW | RTLD_LOCAL);
   if (!lib) {
      printf("FAIL: dlopen %s: %s\n", icd, dlerror());
      failures++;
      return false;
   }
   VkResult (*neg)(uint32_t *) =
      (VkResult (*)(uint32_t *))dlsym(lib, "vk_icdNegotiateLoaderICDInterfaceVersion");
   vk.gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vk_icdGetInstanceProcAddr");
   if (!vk.gipa) {
      printf("FAIL: %s has no vk_icdGetInstanceProcAddr\n", icd);
      failures++;
      return false;
   }
   if (neg) {
      uint32_t v = 5;
      neg(&v);
   }
#define GI(n) vk.n = (PFN_vk##n)vk.gipa(vk.inst, "vk" #n)
   GI(CreateInstance);
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .apiVersion = VK_API_VERSION_1_2 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app };
   VKCHECK(vk.CreateInstance(&ici, NULL, &vk.inst));
   GI(EnumeratePhysicalDevices);
   GI(GetPhysicalDeviceProperties);
   GI(GetPhysicalDeviceMemoryProperties);
   GI(CreateDevice);
   GI(GetDeviceProcAddr);
   uint32_t n = 8;
   VkPhysicalDevice pds[8];
   vk.EnumeratePhysicalDevices(vk.inst, &n, pds);
   for (uint32_t i = 0; i < n && !vk.pd; i++) {
      VkPhysicalDeviceProperties p;
      vk.GetPhysicalDeviceProperties(pds[i], &p);
      printf("vk device %u: %s vendor 0x%x\n", i, p.deviceName, p.vendorID);
      if (p.vendorID == 0x10de)
         vk.pd = pds[i];
   }
   if (!vk.pd) {
      printf("FAIL: no NVIDIA Vulkan device\n");
      failures++;
      return false;
   }
   const float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                   .queueFamilyIndex = 0, .queueCount = 1,
                                   .pQueuePriorities = &prio };
   const char *exts[] = { "VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf",
                          "VK_EXT_image_drm_format_modifier" };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
                              .enabledExtensionCount = 3, .ppEnabledExtensionNames = exts };
   VKCHECK(vk.CreateDevice(vk.pd, &dci, NULL, &vk.dev));
#define GD(n) vk.n = (PFN_vk##n)vk.GetDeviceProcAddr(vk.dev, "vk" #n)
   GD(AllocateMemory);
   GD(FreeMemory);
   GD(MapMemory);
   GD(UnmapMemory);
   GD(CreateImage);
   GD(DestroyImage);
   GD(GetImageMemoryRequirements);
   GD(GetImageSubresourceLayout);
   GD(BindImageMemory);
   GD(GetMemoryFdKHR);
   return vk.GetMemoryFdKHR && vk.BindImageMemory;
}

static uint32_t
mem_type(uint32_t bits, VkMemoryPropertyFlags want)
{
   VkPhysicalDeviceMemoryProperties mp;
   vk.GetPhysicalDeviceMemoryProperties(vk.pd, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
         return i;
   return UINT32_MAX;
}

/* -------------------------------------------------------------- layout */
struct yuv_buf {
   uint32_t fourcc, w, h, rows;   /* rows = allocated luma rows (>= h) */
   uint32_t bpc;                  /* bytes per component: 1 NV12, 2 P010 */
   uint32_t pitch, offset1;
   uint64_t size;
   int fd;
   VkDeviceMemory mem;
   VkImage img;
   uint8_t *map;
};

/* minigbm 0004: pitch ALIGN(row bytes, 256), CbCr at pitch * even(rows),
 * CbCr rows = ceil(rows / 2); NVK's plane_layout keeps it packed. */
static void
gralloc_layout(struct yuv_buf *b)
{
   b->bpc = b->fourcc == DRM_FORMAT_P010 ? 2 : 1;
   b->pitch = (b->w * b->bpc + 255) & ~255u;
   const uint32_t even = (b->rows + 1) & ~1u;
   b->offset1 = b->pitch * even;
   b->size = (uint64_t)b->offset1 + (uint64_t)b->pitch * ((b->rows + 1) / 2);
}

/* Deterministic test content: 8-bit codes (P010: 10-bit codes in the MSBs).
 * Full code range incl. sub-black / super-white for limited range. */
static uint32_t
y_code(uint32_t x, uint32_t y, uint32_t bits)
{
   uint32_t v = (x * 7 + y * 13 + ((x * y) >> 3)) & 0x3ff;
   return bits == 8 ? (v & 0xff) : v;
}

static void
uv_code(uint32_t cx, uint32_t cy, uint32_t bits, uint32_t *u, uint32_t *v)
{
   uint32_t a = (cx * 11 + cy * 5) & 0x3ff, c = (cx * 3 + cy * 17 + 300) & 0x3ff;
   *u = bits == 8 ? (a & 0xff) : a;
   *v = bits == 8 ? (c & 0xff) : c;
}

static void
fill(struct yuv_buf *b)
{
   const uint32_t bits = b->bpc == 1 ? 8 : 10;
   memset(b->map, 0x5a, b->size);
   for (uint32_t y = 0; y < b->h; y++)
      for (uint32_t x = 0; x < b->w; x++) {
         uint32_t c = y_code(x, y, bits);
         if (b->bpc == 1)
            b->map[(uint64_t)y * b->pitch + x] = c;
         else
            *(uint16_t *)(b->map + (uint64_t)y * b->pitch + 2 * x) = c << 6;
      }
   for (uint32_t cy = 0; cy < (b->h + 1) / 2; cy++)
      for (uint32_t cx = 0; cx < (b->w + 1) / 2; cx++) {
         uint32_t u, v;
         uv_code(cx, cy, bits, &u, &v);
         uint8_t *row = b->map + b->offset1 + (uint64_t)cy * b->pitch;
         if (b->bpc == 1) {
            row[2 * cx] = u;
            row[2 * cx + 1] = v;
         } else {
            ((uint16_t *)row)[2 * cx] = u << 6;
            ((uint16_t *)row)[2 * cx + 1] = v << 6;
         }
      }
}

static bool
export_mem(struct yuv_buf *b, VkMemoryRequirements r, VkImage dedicated)
{
   VkMemoryDedicatedAllocateInfo ded = { .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                                         .image = dedicated };
   VkExportMemoryAllocateInfo exp = { .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
      .pNext = dedicated ? &ded : NULL,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
   uint32_t t = mem_type(r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (t == UINT32_MAX) {
      printf("FAIL: no host-visible memory type in 0x%x\n", r.memoryTypeBits);
      failures++;
      return false;
   }
   VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &exp,
                               .allocationSize = r.size, .memoryTypeIndex = t };
   VKCHECK(vk.AllocateMemory(vk.dev, &ai, NULL, &b->mem));
   if (dedicated)
      VKCHECK(vk.BindImageMemory(vk.dev, dedicated, b->mem, 0));
   void *map;
   VKCHECK(vk.MapMemory(vk.dev, b->mem, 0, VK_WHOLE_SIZE, 0, &map));
   b->map = map;
   VkMemoryGetFdInfoKHR gi = { .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
      .memory = b->mem, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
   VKCHECK(vk.GetMemoryFdKHR(vk.dev, &gi, &b->fd));
   return true;
}

/* gralloc's R8 carrier: plain exportable memory of the layout's size. */
static bool
make_raw(struct yuv_buf *b)
{
   VkMemoryRequirements r = { .size = (b->size + 4095) & ~4095ull, .alignment = 4096,
                              .memoryTypeBits = ~0u };
   if (!export_mem(b, r, VK_NULL_HANDLE))
      return false;
   fill(b);
   return true;
}

/* A VK multi-planar LINEAR image with the gralloc plane layouts. */
static bool
make_image(struct yuv_buf *b)
{
   VkSubresourceLayout planes[2] = {
      { .offset = 0, .rowPitch = b->pitch },
      { .offset = b->offset1, .rowPitch = b->pitch },
   };
   VkImageDrmFormatModifierExplicitCreateInfoEXT emi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
      .drmFormatModifier = DRM_FORMAT_MOD_LINEAR, .drmFormatModifierPlaneCount = 2,
      .pPlaneLayouts = planes };
   VkExternalMemoryImageCreateInfo emci = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, .pNext = &emi,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
   VkImageCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &emci,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = b->bpc == 1 ? VK_FORMAT_G8_B8R8_2PLANE_420_UNORM :
                              VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
      .extent = { b->w, b->rows, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };
   VKCHECK(vk.CreateImage(vk.dev, &ici, NULL, &b->img));
   VkMemoryRequirements r;
   vk.GetImageMemoryRequirements(vk.dev, b->img, &r);
   for (uint32_t p = 0; p < 2; p++) {
      /* PLANE_n, not MEMORY_PLANE_n: NVK (nvk_image.h
       * nvk_image_memory_aspects_to_plane) supports memory-plane aspects
       * only for single-plane modifiers and asserts on MEMORY_PLANE_1. */
      VkImageSubresource sub = { .aspectMask = p ? VK_IMAGE_ASPECT_PLANE_1_BIT :
                                                   VK_IMAGE_ASPECT_PLANE_0_BIT };
      VkSubresourceLayout l;
      vk.GetImageSubresourceLayout(vk.dev, b->img, &sub, &l);
      if (l.offset != planes[p].offset || l.rowPitch != planes[p].rowPitch) {
         printf("FAIL: NVK plane %u layout offset %llu pitch %llu, gralloc %llu/%llu\n", p,
                (unsigned long long)l.offset, (unsigned long long)l.rowPitch,
                (unsigned long long)planes[p].offset, (unsigned long long)planes[p].rowPitch);
         failures++;
         return false;
      }
   }
   if (r.size > b->size)
      b->size = r.size; /* the exported dma-buf is what NVK requires */
   if (!export_mem(b, r, b->img))
      return false;
   fill(b);
   return true;
}

static void
free_buf(struct yuv_buf *b)
{
   if (b->fd >= 0)
      close(b->fd);
   if (b->map)
      vk.UnmapMemory(vk.dev, b->mem);
   if (b->img)
      vk.DestroyImage(vk.dev, b->img, NULL);
   if (b->mem)
      vk.FreeMemory(vk.dev, b->mem, NULL);
   memset(b, 0, sizeof(*b));
   b->fd = -1;
}

/* ---------------------------------------------------------- reference */
enum cs { BT601, BT709, BT2020 };
static const char *cs_name[] = { "BT601", "BT709", "BT2020" };
static const EGLint cs_hint[] = { EGL_ITU_REC601_EXT, EGL_ITU_REC709_EXT, EGL_ITU_REC2020_EXT };

/* H.273 / BT.601/709/2020 YCbCr -> R'G'B' on the code values (strict).
 * P010 non-strict = what st/mesa defines for every gallium driver that
 * samples P010 per plane (nir_lower_tex CSC): the plane is sampled as
 * UNORM16, s = (code << 6) / 65535, and the 8-bit-defined ranges are applied
 * to s (limited 16/255, 219/255, 224/255; Mesa's full-range chroma midpoint
 * is 0.5). That differs from strict 10-bit H.273 by up to ~3 LSB at 8 bits
 * (Mesa core, not the driver); the strict difference is reported as info. */
/* Chroma codes at luma pixel (x, y): NEAREST = texel (x/2, y/2); LINEAR =
 * bilinear at the luma centre, chroma coordinate x/2 - 0.25 (weights 3/4,
 * 1/4; GL_CLAMP_TO_EDGE, the external-texture default). */
static void
chroma(const struct yuv_buf *b, bool linear, uint32_t x, uint32_t y, double *u, double *v)
{
   const uint32_t bits = b->bpc == 1 ? 8 : 10, cw = (b->w + 1) / 2, ch = (b->h + 1) / 2;
   if (!linear) {
      uint32_t iu, iv;
      uv_code(x / 2, y / 2, bits, &iu, &iv);
      *u = iu;
      *v = iv;
      return;
   }
   const int x0 = (x & 1) ? (int)x / 2 : (int)x / 2 - 1, y0 = (y & 1) ? (int)y / 2 : (int)y / 2 - 1;
   const double wx1 = (x & 1) ? 0.25 : 0.75, wy1 = (y & 1) ? 0.25 : 0.75;
   *u = *v = 0;
   for (int j = 0; j < 2; j++)
      for (int i = 0; i < 2; i++) {
         int cx = x0 + i, cy = y0 + j;
         cx = cx < 0 ? 0 : cx >= (int)cw ? (int)cw - 1 : cx;
         cy = cy < 0 ? 0 : cy >= (int)ch ? (int)ch - 1 : cy;
         const double w = (i ? wx1 : 1 - wx1) * (j ? wy1 : 1 - wy1);
         uint32_t iu, iv;
         uv_code(cx, cy, bits, &iu, &iv);
         *u += w * iu;
         *v += w * iv;
      }
}

static bool g_linear;

static void
reference(const struct yuv_buf *b, enum cs cs, bool full, bool strict, uint32_t x, uint32_t y,
          uint8_t out[3])
{
   static const double kr[] = { 0.299, 0.2126, 0.2627 }, kb[] = { 0.114, 0.0722, 0.0593 };
   const uint32_t bits = b->bpc == 1 ? 8 : 10;
   double u, v;
   chroma(b, g_linear, x, y, &u, &v);
   const uint32_t yc = y_code(x, y, bits);
   double Y, Cb, Cr;
   if (strict || bits == 8) {
      const double maxc = (1 << bits) - 1, scale = 1 << (bits - 8);
      if (full) {
         Y = yc / maxc;
         Cb = (u - 128 * scale) / maxc;
         Cr = (v - 128 * scale) / maxc;
      } else {
         Y = (yc - 16 * scale) / (219 * scale);
         Cb = (u - 128 * scale) / (224 * scale);
         Cr = (v - 128 * scale) / (224 * scale);
      }
   } else {
      const double sy = (yc << 6) / 65535.0, su = u * 64 / 65535.0, sv = v * 64 / 65535.0;
      if (full) {
         Y = sy;
         Cb = su - 0.5;
         Cr = sv - 0.5;
      } else {
         Y = (sy - 16 / 255.0) * 255 / 219;
         Cb = (su - 128 / 255.0) * 255 / 224;
         Cr = (sv - 128 / 255.0) * 255 / 224;
      }
   }
   const double r = kr[cs], bb = kb[cs], g = 1 - r - bb;
   double rgb[3] = {
      Y + 2 * (1 - r) * Cr,
      Y - 2 * bb * (1 - bb) / g * Cb - 2 * r * (1 - r) / g * Cr,
      Y + 2 * (1 - bb) * Cb,
   };
   for (int i = 0; i < 3; i++) {
      double c = rgb[i] < 0 ? 0 : rgb[i] > 1 ? 1 : rgb[i];
      out[i] = (uint8_t)lround(c * 255);
   }
}

/* ----------------------------------------------------------------- GL */
static PFNEGLCREATEIMAGEKHRPROC create_image;
static PFNEGLDESTROYIMAGEKHRPROC destroy_image;
static PFNGLEGLIMAGETARGETTEXTURE2DOESPROC image_target;
static EGLDisplay dpy;
static GLuint prog, fbo, rb;
static GLint u_size;

static EGLImageKHR
import(const struct yuv_buf *b, int fd1, uint32_t off1, uint32_t pitch, bool with_mod,
       uint64_t mod, uint32_t fourcc, enum cs cs, bool full)
{
   EGLint a[64];
   int n = 0;
#define A(k, v) do { a[n++] = (k); a[n++] = (EGLint)(v); } while (0)
   A(EGL_WIDTH, b->w);
   A(EGL_HEIGHT, b->h);
   A(EGL_LINUX_DRM_FOURCC_EXT, fourcc);
   A(EGL_DMA_BUF_PLANE0_FD_EXT, b->fd);
   A(EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0);
   A(EGL_DMA_BUF_PLANE0_PITCH_EXT, pitch);
   A(EGL_DMA_BUF_PLANE1_FD_EXT, fd1);
   A(EGL_DMA_BUF_PLANE1_OFFSET_EXT, off1);
   A(EGL_DMA_BUF_PLANE1_PITCH_EXT, pitch);
   if (fourcc == DRM_FORMAT_YUV420) {
      A(EGL_DMA_BUF_PLANE2_FD_EXT, fd1);
      A(EGL_DMA_BUF_PLANE2_OFFSET_EXT, off1 + pitch / 2);
      A(EGL_DMA_BUF_PLANE2_PITCH_EXT, pitch);
   }
   if (with_mod) {
      A(EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, mod & 0xffffffff);
      A(EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, mod >> 32);
      A(EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT, mod & 0xffffffff);
      A(EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT, mod >> 32);
      if (fourcc == DRM_FORMAT_YUV420) {
         A(EGL_DMA_BUF_PLANE2_MODIFIER_LO_EXT, mod & 0xffffffff);
         A(EGL_DMA_BUF_PLANE2_MODIFIER_HI_EXT, mod >> 32);
      }
   }
   A(EGL_YUV_COLOR_SPACE_HINT_EXT, cs_hint[cs]);
   A(EGL_SAMPLE_RANGE_HINT_EXT, full ? EGL_YUV_FULL_RANGE_EXT : EGL_YUV_NARROW_RANGE_EXT);
   a[n++] = EGL_NONE;
   return create_image(dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, a);
}

static bool
sample_and_compare(const struct yuv_buf *b, EGLImageKHR img, enum cs cs, bool full,
                   const char *what)
{
   GLuint tex;
   glGenTextures(1, &tex);
   glBindTexture(GL_TEXTURE_EXTERNAL_OES, tex);
   image_target(GL_TEXTURE_EXTERNAL_OES, img);
   const GLint filter = g_linear ? GL_LINEAR : GL_NEAREST;
   glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, filter);
   glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, filter);
   glBindRenderbuffer(GL_RENDERBUFFER, rb);
   glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8_OES, b->w, b->h);
   glBindFramebuffer(GL_FRAMEBUFFER, fbo);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
   glViewport(0, 0, b->w, b->h);
   glUseProgram(prog);
   glUniform2f(u_size, b->w, b->h);
   glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
   uint8_t *px = malloc((size_t)b->w * b->h * 4);
   glReadPixels(0, 0, b->w, b->h, GL_RGBA, GL_UNSIGNED_BYTE, px);
   GLenum err = glGetError();
   int maxd = 0, bad = 0, bx = -1, by = -1, strictd = 0;
   for (uint32_t y = 0; y < b->h; y++)
      for (uint32_t x = 0; x < b->w; x++) {
         uint8_t ref[3], sref[3];
         reference(b, cs, full, false, x, y, ref);
         reference(b, cs, full, true, x, y, sref);
         const uint8_t *p = px + 4 * ((size_t)y * b->w + x);
         for (int c = 0; c < 3; c++) {
            int sd = abs((int)p[c] - (int)sref[c]);
            if (sd > strictd)
               strictd = sd;
            int d = abs((int)p[c] - (int)ref[c]);
            if (d > maxd) {
               maxd = d;
               bx = x;
               by = y;
            }
            if (d > 2)
               bad++;
         }
      }
   if (bad && bx >= 0) {
      uint8_t ref[3];
      reference(b, cs, full, false, bx, by, ref);
      const uint8_t *p = px + 4 * ((size_t)by * b->w + bx);
      printf("   worst (%d,%d): got %u %u %u, ref %u %u %u\n", bx, by, p[0], p[1], p[2],
             ref[0], ref[1], ref[2]);
   }
   free(px);
   glDeleteTextures(1, &tex);
   CHECK(err == GL_NO_ERROR && bad == 0, "%s %s %s %s: max diff %d LSB, %d channels > 2 "
         "(strict H.273: max %d) (gl 0x%x)", what, cs_name[cs], full ? "full" : "limited",
         g_linear ? "LINEAR" : "NEAREST", maxd, bad, strictd, err);
   return bad == 0;
}

static const char *vs_src =
   "attribute vec2 p;\nvoid main() { gl_Position = vec4(p, 0.0, 1.0); }\n";
static const char *fs_src =
   "#extension GL_OES_EGL_image_external : require\n"
   "precision highp float;\n"
   "uniform samplerExternalOES t;\nuniform vec2 size;\n"
   "void main() { gl_FragColor = texture2D(t, gl_FragCoord.xy / size); }\n";

static GLuint
shader(GLenum type, const char *src)
{
   GLuint s = glCreateShader(type);
   glShaderSource(s, 1, &src, NULL);
   glCompileShader(s);
   GLint ok;
   glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[1024];
      glGetShaderInfoLog(s, sizeof(log), NULL, log);
      printf("FAIL: shader: %s\n", log);
      failures++;
   }
   return s;
}

static bool
gl_init(const char *vendor)
{
   PFNEGLGETPLATFORMDISPLAYEXTPROC gpd =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
   dpy = gpd ? gpd(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL) : EGL_NO_DISPLAY;
   if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, NULL, NULL)) {
      printf("FAIL: surfaceless EGL display\n");
      failures++;
      return false;
   }
   eglBindAPI(EGL_OPENGL_ES_API);
   const EGLint cattr[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
   EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, cattr);
   if (ctx == EGL_NO_CONTEXT || !eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
      printf("FAIL: ES3 context\n");
      failures++;
      return false;
   }
   const char *v = (const char *)glGetString(GL_VENDOR);
   const char *r = (const char *)glGetString(GL_RENDERER);
   printf("GL_VENDOR %s, GL_RENDERER %s, %s\n", v, r, glGetString(GL_VERSION));
   if (!v || !strstr(v, vendor)) {
      printf("FAIL: vendor is not '%s'\n", vendor);
      failures++;
      return false;
   }
   create_image = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
   destroy_image = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
   image_target = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
   prog = glCreateProgram();
   glAttachShader(prog, shader(GL_VERTEX_SHADER, vs_src));
   glAttachShader(prog, shader(GL_FRAGMENT_SHADER, fs_src));
   glBindAttribLocation(prog, 0, "p");
   glLinkProgram(prog);
   u_size = glGetUniformLocation(prog, "size");
   glUseProgram(prog);
   glUniform1i(glGetUniformLocation(prog, "t"), 0);
   static const float quad[] = { -1, -1, 1, -1, -1, 1, 1, 1 };
   GLuint vbo;
   glGenBuffers(1, &vbo);
   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
   glEnableVertexAttribArray(0);
   glGenFramebuffers(1, &fbo);
   glGenRenderbuffers(1, &rb);
   return true;
}

static void
query_modifiers(uint32_t fourcc, const char *name)
{
   PFNEGLQUERYDMABUFMODIFIERSEXTPROC qm =
      (PFNEGLQUERYDMABUFMODIFIERSEXTPROC)eglGetProcAddress("eglQueryDmaBufModifiersEXT");
   PFNEGLQUERYDMABUFFORMATSEXTPROC qf =
      (PFNEGLQUERYDMABUFFORMATSEXTPROC)eglGetProcAddress("eglQueryDmaBufFormatsEXT");
   EGLint nf = 0, fmts[128];
   qf(dpy, 128, fmts, &nf);
   bool listed = false;
   for (int i = 0; i < nf; i++)
      listed |= (uint32_t)fmts[i] == fourcc;
   EGLuint64KHR mods[16];
   EGLBoolean ext[16];
   EGLint n = 0;
   qm(dpy, fourcc, 16, mods, ext, &n);
   CHECK(listed && n == 1 && mods[0] == DRM_FORMAT_MOD_LINEAR && ext[0],
         "%s: in eglQueryDmaBufFormatsEXT %d, modifiers %d (first 0x%llx external_only %d)",
         name, listed, n, n ? (unsigned long long)mods[0] : 0ull, n ? (int)ext[0] : -1);
}

static void
expect_refused(EGLImageKHR img, const char *what)
{
   EGLint e = eglGetError();
   CHECK(img == EGL_NO_IMAGE_KHR, "refused: %s (EGL error 0x%x)", what, e);
   if (img != EGL_NO_IMAGE_KHR)
      destroy_image(dpy, img);
}

int
main(int argc, char **argv)
{
   setvbuf(stdout, NULL, _IONBF, 0);
   if (argc < 2) {
      fprintf(stderr, "usage: gl-yuv <vendor-substring>\n");
      return 2;
   }
   if (!gl_init(argv[1]) || !vk_init()) {
      printf("GL YUV FAIL\n");
      return 1;
   }
   const char *exts = eglQueryString(dpy, EGL_EXTENSIONS);
   CHECK(strstr(exts, "EGL_EXT_image_dma_buf_import_modifiers") != NULL,
         "EGL_EXT_image_dma_buf_import_modifiers");
   query_modifiers(DRM_FORMAT_NV12, "NV12");
   query_modifiers(DRM_FORMAT_P010, "P010");

   static const struct { uint32_t w, h, rows; } sizes[] = {
      { 854, 480, 480 }, { 1920, 1080, 1088 }, { 640, 362, 362 },
   };
   static const uint32_t fourccs[] = { DRM_FORMAT_NV12, DRM_FORMAT_P010 };
   for (unsigned f = 0; f < 2; f++)
      for (unsigned s = 0; s < 3; s++)
         for (unsigned src = 0; src < 2; src++) {
            struct yuv_buf b = { .fourcc = fourccs[f], .w = sizes[s].w, .h = sizes[s].h,
                                 .rows = sizes[s].rows, .fd = -1 };
            gralloc_layout(&b);
            if (!(src ? make_image(&b) : make_raw(&b))) {
               free_buf(&b);
               continue;
            }
            char what[96];
            snprintf(what, sizeof(what), "%s %ux%u (rows %u, pitch %u, CbCr @%u, %s)",
                     f ? "P010" : "NV12", b.w, b.h, b.rows, b.pitch, b.offset1,
                     src ? "VK image" : "raw");
            for (unsigned cs = 0; cs < 3; cs++)
               for (unsigned full = 0; full < 2; full++) {
                  /* BT2020 only with P010 (the HDR case) to keep the run short */
                  if (cs == BT2020 && !f)
                     continue;
                  EGLImageKHR img = import(&b, b.fd, b.offset1, b.pitch, true,
                                           DRM_FORMAT_MOD_LINEAR, b.fourcc, cs, full);
                  if (img == EGL_NO_IMAGE_KHR) {
                     CHECK(false, "%s %s: eglCreateImage failed 0x%x", what, cs_name[cs],
                           eglGetError());
                     continue;
                  }
                  sample_and_compare(&b, img, cs, full, what);
                  destroy_image(dpy, img);
               }
            {
               /* GL_LINEAR (what RenderEngine uses): chroma filtered per plane */
               EGLImageKHR img = import(&b, b.fd, b.offset1, b.pitch, true,
                                        DRM_FORMAT_MOD_LINEAR, b.fourcc, BT709, false);
               if (img == EGL_NO_IMAGE_KHR)
                  CHECK(false, "%s LINEAR: eglCreateImage failed 0x%x", what, eglGetError());
               else {
                  g_linear = true;
                  sample_and_compare(&b, img, BT709, false, what);
                  g_linear = false;
                  destroy_image(dpy, img);
               }
            }
            if (s == 0) {
               /* separate fd for plane 1 (same dma-buf, dup'ed) */
               int fd1 = dup(b.fd);
               EGLImageKHR img = import(&b, fd1, b.offset1, b.pitch, true,
                                        DRM_FORMAT_MOD_LINEAR, b.fourcc, BT709, false);
               if (img == EGL_NO_IMAGE_KHR)
                  CHECK(false, "%s separate plane fd: eglCreateImage failed 0x%x", what,
                        eglGetError());
               else {
                  char w2[128];
                  snprintf(w2, sizeof(w2), "%s, plane 1 own fd", what);
                  sample_and_compare(&b, img, BT709, false, w2);
                  destroy_image(dpy, img);
               }
               close(fd1);
            }
            if (s == 0 && f == 0 && src == 0) {
               /* negative cases, each must fail loudly without a crash */
               expect_refused(import(&b, b.fd, b.offset1, b.pitch, true,
                                     0x0300000000606010ull, b.fourcc, BT709, false),
                              "NV12 block-linear modifier 0x0300000000606010");
               expect_refused(import(&b, b.fd, b.offset1, b.pitch, false, 0, b.fourcc,
                                     BT709, false), "NV12 without a modifier");
               expect_refused(import(&b, b.fd, b.offset1, b.pitch, true,
                                     DRM_FORMAT_MOD_LINEAR, DRM_FORMAT_YUV420, BT709, false),
                              "YUV420 (3-plane) is not accepted");
               expect_refused(import(&b, b.fd, b.offset1 + 16, b.pitch, true,
                                     DRM_FORMAT_MOD_LINEAR, b.fourcc, BT709, false),
                              "CbCr offset off the 32 B plane alignment");
               expect_refused(import(&b, b.fd, (uint32_t)b.size, b.pitch, true,
                                     DRM_FORMAT_MOD_LINEAR, b.fourcc, BT709, false),
                              "CbCr plane past the end of the dma-buf");
               expect_refused(import(&b, b.fd, b.offset1, 880, true, DRM_FORMAT_MOD_LINEAR,
                                     b.fourcc, BT709, false),
                              "pitch 880 B (not a 32 B multiple)");
            }
            free_buf(&b);
         }
   printf(failures ? "GL YUV FAIL (%d)\n" : "GL YUV PASS\n", failures);
   return failures ? 1 : 0;
}
