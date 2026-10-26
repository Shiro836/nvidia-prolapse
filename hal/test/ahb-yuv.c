/* ahb-yuv: host test of vulkan.nvidia.so's NV12/P010 AHardwareBuffer import (external formats + Ycbcr
 * conversion), the path a decoded video frame takes into Chrome (AImageReader PRIVATE + GPU_SAMPLED_IMAGE),
 * ANGLE (samplerExternalOES) and HWUI.
 *
 * The fake gralloc mirrors minigbm gbm_mesa with gralloc/minigbm/0004: an NV12/P010 buffer is
 * one LINEAR dma-buf (minigbm's R8 carrier), plane 0 at offset 0, byte pitch ALIGN(width * bytes, 256),
 * CbCr plane at pitch * ALIGN(height, 2) or later, moved by the gralloc wrapper's plane_layout() to the driver's
 * per-plane alignment (DISJOINT-plane memory requirements; NVIDIA 615: 64 KiB, NVK: 32 B) and the buffer sized to the
 * driver's requirement; fds[1] is a dup of fds[0]; an optional reserved region carries CrosGralloc4Metadata (dataspace
 * at offset 1028). Cases can ask for the CbCr plane further (stock's own layout pads luma to 1088 rows), so the explicit
 * plane offset is exercised, not just the packed default. A layout off the driver's per-plane alignment is refused
 * by the HAL (negative cases).
 *
 * Per case: vkGetAndroidHardwareBufferPropertiesANDROID (format UNDEFINED, externalFormat, features,
 * suggested model/range from the dataspace) -> vkCreateImage(format UNDEFINED, VkExternalFormatANDROID,
 * usage SAMPLED|INPUT_ATTACHMENT as ANGLE asks) -> vkAllocateMemory(import AHB, dedicated) ->
 * vkBindImageMemory -> vkCreateSamplerYcbcrConversion(VkExternalFormatANDROID, suggested values) -> view
 * (format UNDEFINED) -> compute shader samples every luma texel centre into a float buffer, compared against
 * a CPU implementation of the Vulkan spec's Ycbcr reconstruction + model conversion (tolerance 2/255).
 *
 * usage: ahb-yuv HAL.so      exit 0 only when every check passed */
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <drm/drm_fourcc.h>
#include <vulkan/vulkan.h>
#include <hardware/hwvulkan.h>
#include <cros_gralloc_handle.h>
#include "nvhal_ahb.h"
#include "ahb_yuv_spv.h"

#define FAKE_MAGIC 0x59555621u
#define TOL (2.0 / 255.0)

static int failures;
static void check(int ok, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void check(int ok, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("  [%s] ", ok ? "PASS" : "FAIL");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
    if (!ok) failures++;
}
#define MUST(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("FATAL: %s -> %d\n", #x, r_); exit(2); } } while (0)

static PFN_vkGetInstanceProcAddr gipa;
static PFN_vkGetDeviceProcAddr gdpa;
static VkInstance inst;
static VkPhysicalDevice pd;
static VkDevice dev;
static uint32_t qfi;
static VkQueue queue;
static VkPhysicalDeviceMemoryProperties memprops;
static void *ip(const char *n) { void *f = (void *)gipa(inst, n); if (!f) { printf("FATAL: no %s\n", n); exit(2); } return f; }
static void *dp(const char *n) { void *f = (void *)gdpa(dev, n); if (!f) { printf("FATAL: no %s\n", n); exit(2); } return f; }

#define FNS(X) X(GetDeviceQueue) X(CreateImage) X(DestroyImage) X(AllocateMemory) X(FreeMemory) X(BindImageMemory) \
    X(GetMemoryFdKHR) X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) X(BindBufferMemory) X(MapMemory) \
    X(CreateCommandPool) X(DestroyCommandPool) X(AllocateCommandBuffers) X(BeginCommandBuffer) X(EndCommandBuffer) \
    X(CmdPipelineBarrier) X(QueueSubmit) X(QueueWaitIdle) X(DestroyDevice) X(CreateImageView) X(DestroyImageView) \
    X(GetAndroidHardwareBufferPropertiesANDROID) X(CreateSamplerYcbcrConversion) X(DestroySamplerYcbcrConversion) \
    X(CreateSampler) X(DestroySampler) X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) \
    X(CreatePipelineLayout) X(DestroyPipelineLayout) X(CreateShaderModule) X(DestroyShaderModule) \
    X(CreateComputePipelines) X(DestroyPipeline) X(CreateDescriptorPool) X(DestroyDescriptorPool) \
    X(AllocateDescriptorSets) X(UpdateDescriptorSets) X(CmdBindPipeline) X(CmdBindDescriptorSets) X(CmdPushConstants) \
    X(CmdDispatch) X(ResetCommandBuffer) X(GetImageMemoryRequirements) X(GetImageMemoryRequirements2)
#define DECL(n) static PFN_vk##n p_##n;
FNS(DECL)
#undef DECL

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

static int count_fds(void) {
    int n = 0;
    DIR *d = opendir("/proc/self/fd");
    for (struct dirent *e; d && (e = readdir(d));) n += e->d_name[0] != '.';
    if (d) closedir(d);
    return n - 1;
}

/* ------------------------------------------------------------------ fake gralloc (gbm_mesa + 0004) */

struct fake_ahb {
    uint32_t magic;
    int refs;
    AHardwareBuffer_Desc desc;
    struct cros_gralloc_handle h;
    VkBuffer buf;
    VkDeviceMemory mem;
    uint8_t *map;
};
static int fake_live;
static struct fake_ahb *F(const AHardwareBuffer *b) {
    struct fake_ahb *f = (struct fake_ahb *)b;
    if (!f || f->magic != FAKE_MAGIC) { printf("FATAL: not a fake AHB %p\n", (const void *)b); exit(2); }
    return f;
}
static void fake_describe(const AHardwareBuffer *b, AHardwareBuffer_Desc *d) { *d = F(b)->desc; }
static const native_handle_t *fake_handle(const AHardwareBuffer *b) { return (const native_handle_t *)(const void *)&F(b)->h; }
static void fake_acquire(AHardwareBuffer *b) { F(b)->refs++; }
static void fake_release(AHardwareBuffer *b) {
    struct fake_ahb *f = F(b);
    if (--f->refs > 0) return;
    p_DestroyBuffer(dev, f->buf, NULL);
    p_FreeMemory(dev, f->mem, NULL);
    for (int i = 0; i < DRV_MAX_FDS; i++) if (f->h.fds[i] >= 0) close(f->h.fds[i]);
    f->magic = 0;
    free(f);
    fake_live--;
}
static int fake_allocate(const AHardwareBuffer_Desc *desc, AHardwareBuffer **out) {
    (void)desc; (void)out;
    return -1; /* YUV export is not implemented by the HAL; nothing here allocates through this entry */
}
__attribute__((visibility("default"))) struct nvhal_ahb_fns nvhal_test_ahb_fns = {
    fake_describe, fake_handle, fake_acquire, fake_release, fake_allocate,
};

#define ALIGN(v, a) (((v) + (a) - 1) / (a) * (a))

/* nvg_plane_layout (gralloc/nvidia_gbm_wrapper.c), which minigbm 0004 applies to its packed layout: each plane moves to
 * the alignment of its DISJOINT-image memory requirement, past the previous plane's required size; the buffer holds the
 * last plane's requirement and the whole image's (same create info: LINEAR, SAMPLED, dma-buf) */
static VkImage yuv_image(uint32_t fourcc, uint32_t w, uint32_t h, const uint32_t *strides, const uint64_t *offsets,
                         bool disjoint) {
    VkSubresourceLayout planes[2] = {{.offset = offsets[0], .rowPitch = strides[0]}, {.offset = offsets[1], .rowPitch = strides[1]}};
    VkImageDrmFormatModifierExplicitCreateInfoEXT ex = {.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
                                                        .drmFormatModifier = DRM_FORMAT_MOD_LINEAR,
                                                        .drmFormatModifierPlaneCount = 2, .pPlaneLayouts = planes};
    VkExternalMemoryImageCreateInfo emi = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, .pNext = &ex,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkImageCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &emi,
                             .flags = disjoint ? VK_IMAGE_CREATE_DISJOINT_BIT : 0, .imageType = VK_IMAGE_TYPE_2D,
                             .format = fourcc == DRM_FORMAT_P010 ? VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
                                                                 : VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
                             .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                             .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
                             .usage = VK_IMAGE_USAGE_SAMPLED_BIT};
    VkImage img;
    MUST(p_CreateImage(dev, &ici, NULL, &img));
    return img;
}
static uint64_t gralloc_plane_layout(uint32_t fourcc, uint32_t w, uint32_t h, const uint32_t *strides, uint32_t *offsets) {
    const uint64_t zero[2] = {0, 0};
    VkImage img = yuv_image(fourcc, w, h, strides, zero, true);
    VkMemoryRequirements preq[2];
    for (uint32_t p = 0; p < 2; p++) {
        VkImagePlaneMemoryRequirementsInfo pi = {.sType = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO,
                                                 .planeAspect = p ? VK_IMAGE_ASPECT_PLANE_1_BIT : VK_IMAGE_ASPECT_PLANE_0_BIT};
        VkImageMemoryRequirementsInfo2 mi = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, .pNext = &pi,
                                             .image = img};
        VkMemoryRequirements2 mr = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
        p_GetImageMemoryRequirements2(dev, &mi, &mr);
        preq[p] = mr.memoryRequirements;
    }
    p_DestroyImage(dev, img, NULL);
    uint64_t off[2], end = 0;
    for (uint32_t p = 0; p < 2; p++) {
        const uint64_t a = preq[p].alignment ? preq[p].alignment : 1;
        off[p] = ALIGN(offsets[p] > end ? offsets[p] : end, a);
        end = off[p] + preq[p].size;
        offsets[p] = (uint32_t)off[p];
    }
    img = yuv_image(fourcc, w, h, strides, off, false);
    VkMemoryRequirements mr;
    p_GetImageMemoryRequirements(dev, img, &mr);
    p_DestroyImage(dev, img, NULL);
    return mr.size > end ? mr.size : end;
}

/* luma_rows: rows reserved for plane 0 (gralloc: ALIGN(h, 2)); dataspace < 0: no metadata region */
static struct fake_ahb *fake_yuv(uint32_t fourcc, uint32_t w, uint32_t h, uint32_t luma_rows, uint32_t extra_offset,
                                 int64_t dataspace) {
    struct fake_ahb *f = calloc(1, sizeof *f);
    f->magic = FAKE_MAGIC;
    f->refs = 1;
    for (int i = 0; i < DRV_MAX_FDS; i++) f->h.fds[i] = -1;
    const uint32_t bpc = fourcc == DRM_FORMAT_P010 ? 2 : 1;
    const uint32_t stride = ALIGN(w * bpc, 256u);
    /* minigbm 0004: packed layout, then the wrapper's plane_layout() moves the planes and sizes the buffer; the
     * carrier is R8 4096 x ceil(total_size / 4096) (gbm_mesa_internals.cpp), i.e. whole 4 KiB rows */
    const uint32_t strides[2] = {stride, stride};
    uint32_t offsets[2] = {0, stride * luma_rows + extra_offset};
    const uint64_t total_size = gralloc_plane_layout(fourcc, w, h, strides, offsets);
    const uint32_t off1 = offsets[1];
    const uint32_t size1 = stride * ((h + 1) / 2);
    const VkDeviceSize carrier = ALIGN(total_size, 4096u);
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = carrier,
                              .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    MUST(p_CreateBuffer(dev, &bci, NULL, &f->buf));
    VkMemoryRequirements mr;
    p_GetBufferMemoryRequirements(dev, f->buf, &mr);
    VkExportMemoryAllocateInfo ex = {.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
                                     .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &ex, .allocationSize = mr.size,
                                .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    MUST(p_AllocateMemory(dev, &mai, NULL, &f->mem));
    MUST(p_BindBufferMemory(dev, f->buf, f->mem, 0));
    MUST(p_MapMemory(dev, f->mem, 0, VK_WHOLE_SIZE, 0, (void **)&f->map));
    int fd = -1;
    VkMemoryGetFdInfoKHR gfi = {.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = f->mem,
                                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    MUST(p_GetMemoryFdKHR(dev, &gfi, &fd));
    int nfds = 2;
    f->h.fds[0] = fd;
    f->h.fds[1] = dup(fd);
    if (dataspace >= 0) {
        /* CrosGralloc4Metadata: name[1024], BlendMode, Dataspace; the region is sizeof(that) and more */
        const size_t rsize = 4096;
        int md = memfd_create("cros-gralloc4-metadata", MFD_CLOEXEC);
        if (md < 0 || ftruncate(md, (off_t)rsize) != 0) { printf("FATAL: memfd\n"); exit(2); }
        uint8_t *m = mmap(NULL, rsize, PROT_READ | PROT_WRITE, MAP_SHARED, md, 0);
        int32_t ds = (int32_t)dataspace;
        memcpy(m + 1028, &ds, sizeof ds);
        munmap(m, rsize);
        f->h.fds[2] = md;
        f->h.reserved_region_size = rsize;
        nfds = 3;
    }
    f->h.base.version = sizeof(native_handle_t);
    f->h.base.numFds = nfds;
    f->h.base.numInts = (int)CROS_GRALLOC_HANDLE_NUM_INTS_TOTAL - nfds;
    f->h.strides[0] = f->h.strides[1] = stride;
    f->h.offsets[0] = 0;
    f->h.offsets[1] = off1;
    f->h.sizes[0] = stride * h;
    f->h.sizes[1] = size1;
    f->h.width = w;
    f->h.height = h;
    f->h.format = fourcc;
    f->h.format_modifier = DRM_FORMAT_MOD_LINEAR;
    f->h.magic = CROS_GRALLOC_MAGIC;
    f->h.num_planes = 2;
    f->h.total_size = total_size;
    f->desc = (AHardwareBuffer_Desc){.width = w, .height = h, .layers = 1,
                                     .format = fourcc == DRM_FORMAT_P010 ? AHARDWAREBUFFER_FORMAT_YCbCr_P010
                                                                         : AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420,
                                     .usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE, .stride = w};
    f->h.droid_format = (int32_t)f->desc.format;
    f->h.usage = (int32_t)f->desc.usage;
    fake_live++;
    return f;
}

/* the test picture: legal-range codes (8-bit, scaled by 4 for 10-bit) with structure in both axes */
static uint32_t pat_y(uint32_t x, uint32_t y) { return 16 + (x * 7 + y * 3 + (x / 16) * 29) % 220; }
static uint32_t pat_u(uint32_t x, uint32_t y) { return 16 + (x * 11 + y * 5 + (y / 8) * 37) % 225; }
static uint32_t pat_v(uint32_t x, uint32_t y) { return 16 + (x * 3 + y * 13 + (x / 8) * 17) % 225; }

static void fill(struct fake_ahb *f) {
    const bool p010 = f->h.format == DRM_FORMAT_P010;
    const uint32_t w = f->h.width, h = f->h.height, s = f->h.strides[0];
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) {
            uint32_t c = pat_y(x, y);
            if (p010) { uint16_t v = (uint16_t)((c * 4 + (x & 3)) << 6); memcpy(f->map + y * s + x * 2, &v, 2); }
            else f->map[y * s + x] = (uint8_t)c;
        }
    uint8_t *uv = f->map + f->h.offsets[1];
    for (uint32_t y = 0; y < (h + 1) / 2; y++)
        for (uint32_t x = 0; x < (w + 1) / 2; x++) {
            uint32_t u = pat_u(x, y), v = pat_v(x, y);
            if (p010) {
                uint16_t uu = (uint16_t)((u * 4 + (y & 3)) << 6), vv = (uint16_t)((v * 4) << 6);
                memcpy(uv + y * f->h.strides[1] + x * 4, &uu, 2);
                memcpy(uv + y * f->h.strides[1] + x * 4 + 2, &vv, 2);
            } else {
                uv[y * f->h.strides[1] + x * 2] = (uint8_t)u;
                uv[y * f->h.strides[1] + x * 2 + 1] = (uint8_t)v;
            }
        }
}

/* codes as stored (8-bit, or the 10-bit value of P010) */
static double code_y(const struct fake_ahb *f, uint32_t x, uint32_t y) {
    return f->h.format == DRM_FORMAT_P010 ? pat_y(x, y) * 4 + (x & 3) : pat_y(x, y);
}
static double code_c(const struct fake_ahb *f, int cx, int cy, int comp) {
    const int cw = (int)(f->h.width + 1) / 2, ch = (int)(f->h.height + 1) / 2;
    cx = cx < 0 ? 0 : cx >= cw ? cw - 1 : cx; /* CLAMP_TO_EDGE */
    cy = cy < 0 ? 0 : cy >= ch ? ch - 1 : cy;
    if (f->h.format == DRM_FORMAT_P010)
        return comp == 0 ? pat_u((uint32_t)cx, (uint32_t)cy) * 4 + (cy & 3) : pat_v((uint32_t)cx, (uint32_t)cy) * 4;
    return comp == 0 ? pat_u((uint32_t)cx, (uint32_t)cy) : pat_v((uint32_t)cx, (uint32_t)cy);
}

/* Vulkan spec "Sampler Y'CBCR Conversion": midpoint chroma, chroma reconstruction (nearest / linear),
 * narrow/full range expansion, model conversion; RGB clamped to [0,1] on both sides of the comparison */
static void reference(const struct fake_ahb *f, uint32_t x, uint32_t y, VkSamplerYcbcrModelConversion model,
                      VkSamplerYcbcrRange range, VkFilter chroma, double rgb[3]) {
    const int n = f->h.format == DRM_FORMAT_P010 ? 10 : 8;
    const double sc = (double)(1 << (n - 8));
    double c[2];
    for (int k = 0; k < 2; k++) {
        if (chroma == VK_FILTER_NEAREST) {
            c[k] = code_c(f, (int)(x / 2), (int)(y / 2), k);
        } else {
            double tx = (x + 0.5) / 2.0 - 0.5, ty = (y + 0.5) / 2.0 - 0.5;
            int x0 = (int)floor(tx), y0 = (int)floor(ty);
            double ax = tx - x0, ay = ty - y0;
            c[k] = (1 - ax) * (1 - ay) * code_c(f, x0, y0, k) + ax * (1 - ay) * code_c(f, x0 + 1, y0, k) +
                   (1 - ax) * ay * code_c(f, x0, y0 + 1, k) + ax * ay * code_c(f, x0 + 1, y0 + 1, k);
        }
    }
    double Y, Cb, Cr;
    if (range == VK_SAMPLER_YCBCR_RANGE_ITU_NARROW) {
        Y = (code_y(f, x, y) - 16 * sc) / (219 * sc);
        Cb = (c[0] - 128 * sc) / (224 * sc);
        Cr = (c[1] - 128 * sc) / (224 * sc);
    } else {
        const double mx = (1 << n) - 1;
        Y = code_y(f, x, y) / mx;
        Cb = (c[0] - 128 * sc) / mx;
        Cr = (c[1] - 128 * sc) / mx;
    }
    double kr, kb;
    switch (model) {
    case VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601: kr = 0.299; kb = 0.114; break;
    case VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020: kr = 0.2627; kb = 0.0593; break;
    default: kr = 0.2126; kb = 0.0722; break;
    }
    const double kg = 1 - kr - kb;
    rgb[0] = Y + (2 - 2 * kr) * Cr;
    rgb[1] = Y - (2 * kb * (1 - kb) / kg) * Cb - (2 * kr * (1 - kr) / kg) * Cr;
    rgb[2] = Y + (2 - 2 * kb) * Cb;
    for (int i = 0; i < 3; i++) rgb[i] = rgb[i] < 0 ? 0 : rgb[i] > 1 ? 1 : rgb[i];
}

static VkCommandPool pool;
static VkCommandBuffer cmd;

struct ycase {
    const char *name;
    uint32_t fourcc, w, h, luma_rows, extra;
    int64_t dataspace;
    VkSamplerYcbcrModelConversion want_model;
    VkSamplerYcbcrRange want_range;
    VkFilter chroma;
};

static void run_case(const struct ycase *c) {
    printf("case %s: %s %ux%u, CbCr at pitch*%u+%u, dataspace %s0x%llx, chroma filter %s\n", c->name,
           c->fourcc == DRM_FORMAT_P010 ? "P010" : "NV12", c->w, c->h, c->luma_rows, c->extra,
           c->dataspace < 0 ? "none " : "", (unsigned long long)(c->dataspace < 0 ? 0 : c->dataspace),
           c->chroma == VK_FILTER_LINEAR ? "LINEAR" : "NEAREST");
    struct fake_ahb *f = fake_yuv(c->fourcc, c->w, c->h, c->luma_rows, c->extra, c->dataspace);
    printf("  gralloc layout: CbCr at %u, pitch %u, %llu B buffer (handle total_size %llu)\n", f->h.offsets[1],
           f->h.strides[0], (unsigned long long)ALIGN(f->h.total_size, 4096u), (unsigned long long)f->h.total_size);
    fill(f);
    AHardwareBuffer *ahb = (AHardwareBuffer *)f;
    const VkFormat want_fmt = c->fourcc == DRM_FORMAT_P010 ? VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
                                                           : VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
    VkAndroidHardwareBufferFormatPropertiesANDROID fp = {.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID};
    VkAndroidHardwareBufferPropertiesANDROID pp = {.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID, .pNext = &fp};
    VkResult r = p_GetAndroidHardwareBufferPropertiesANDROID(dev, ahb, &pp);
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_MIDPOINT_CHROMA_SAMPLES_BIT;
    check(r == VK_SUCCESS && fp.format == VK_FORMAT_UNDEFINED && fp.externalFormat == (uint64_t)want_fmt &&
          (fp.formatFeatures & need) == need && pp.allocationSize >= f->h.total_size && pp.memoryTypeBits,
          "properties: r %d, format %d (UNDEFINED), externalFormat %llu (VkFormat %d), features 0x%x, %llu B, types 0x%x",
          r, fp.format, (unsigned long long)fp.externalFormat, want_fmt, fp.formatFeatures,
          (unsigned long long)pp.allocationSize, pp.memoryTypeBits);
    check(fp.suggestedYcbcrModel == c->want_model && fp.suggestedYcbcrRange == c->want_range &&
          fp.suggestedXChromaOffset == VK_CHROMA_LOCATION_MIDPOINT && fp.suggestedYChromaOffset == VK_CHROMA_LOCATION_MIDPOINT,
          "suggested model %d range %d chroma %d/%d (want model %d range %d, midpoint)", fp.suggestedYcbcrModel,
          fp.suggestedYcbcrRange, fp.suggestedXChromaOffset, fp.suggestedYChromaOffset, c->want_model, c->want_range);
    if (r != VK_SUCCESS) { fake_release(ahb); return; }

    VkExternalFormatANDROID xf = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID, .externalFormat = fp.externalFormat};
    VkExternalMemoryImageCreateInfo emi = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, .pNext = &xf,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID};
    VkImageCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &emi, .imageType = VK_IMAGE_TYPE_2D,
                             .format = VK_FORMAT_UNDEFINED, .extent = {c->w, c->h, 1}, .mipLevels = 1, .arrayLayers = 1,
                             .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                             /* ANGLE's external-format usage (HBS:78-107) */
                             .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
                             .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage img;
    MUST(p_CreateImage(dev, &ici, NULL, &img));
    VkImportAndroidHardwareBufferInfoANDROID imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
                                                    .buffer = ahb};
    VkMemoryDedicatedAllocateInfo ded = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .pNext = &imp, .image = img};
    uint32_t type = 0;
    while (!(pp.memoryTypeBits & (1u << type))) type++;
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &ded,
                                .allocationSize = pp.allocationSize, .memoryTypeIndex = type};
    VkDeviceMemory mem;
    r = p_AllocateMemory(dev, &mai, NULL, &mem);
    check(r == VK_SUCCESS, "vkAllocateMemory(import AHB, dedicated, type %u) -> %d", type, r);
    if (r != VK_SUCCESS) { p_DestroyImage(dev, img, NULL); fake_release(ahb); return; }
    MUST(p_BindImageMemory(dev, img, mem, 0));

    VkSamplerYcbcrConversionCreateInfo yci = {.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO, .pNext = &xf,
        .format = VK_FORMAT_UNDEFINED, .ycbcrModel = fp.suggestedYcbcrModel, .ycbcrRange = fp.suggestedYcbcrRange,
        .components = fp.samplerYcbcrConversionComponents, .xChromaOffset = fp.suggestedXChromaOffset,
        .yChromaOffset = fp.suggestedYChromaOffset, .chromaFilter = c->chroma};
    VkSamplerYcbcrConversion conv;
    MUST(p_CreateSamplerYcbcrConversion(dev, &yci, NULL, &conv));
    VkSamplerYcbcrConversionInfo cvi = {.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO, .conversion = conv};
    VkSamplerCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .pNext = &cvi,
                               .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
                               .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                               .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                               .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    VkSampler smp;
    MUST(p_CreateSampler(dev, &sci, NULL, &smp));
    VkImageViewCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = &cvi, .image = img,
                                 .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_UNDEFINED,
                                 .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkImageView view;
    MUST(p_CreateImageView(dev, &vci, NULL, &view));

    VkDescriptorSetLayoutBinding b[2] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, &smp},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo dl = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = b};
    VkDescriptorSetLayout dsl;
    MUST(p_CreateDescriptorSetLayout(dev, &dl, NULL, &dsl));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                       .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
    VkPipelineLayout pl;
    MUST(p_CreatePipelineLayout(dev, &plci, NULL, &pl));
    VkShaderModuleCreateInfo smci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof ahb_yuv_spv,
                                     .pCode = ahb_yuv_spv};
    VkShaderModule sm;
    MUST(p_CreateShaderModule(dev, &smci, NULL, &sm));
    VkComputePipelineCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = sm, .pName = "main"}, .layout = pl};
    VkPipeline pipe;
    MUST(p_CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
    VkDescriptorPoolCreateInfo dpci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
                                       .poolSizeCount = 2, .pPoolSizes = ps};
    VkDescriptorPool dpool;
    MUST(p_CreateDescriptorPool(dev, &dpci, NULL, &dpool));
    VkDescriptorSetAllocateInfo dsai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dpool,
                                        .descriptorSetCount = 1, .pSetLayouts = &dsl};
    VkDescriptorSet set;
    MUST(p_AllocateDescriptorSets(dev, &dsai, &set));

    const VkDeviceSize osize = (VkDeviceSize)c->w * c->h * 16;
    VkBufferCreateInfo obci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = osize,
                               .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    VkBuffer ob;
    MUST(p_CreateBuffer(dev, &obci, NULL, &ob));
    VkMemoryRequirements omr;
    p_GetBufferMemoryRequirements(dev, ob, &omr);
    VkMemoryAllocateInfo omai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = omr.size,
        .memoryTypeIndex = mem_type(omr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    VkDeviceMemory omem;
    MUST(p_AllocateMemory(dev, &omai, NULL, &omem));
    MUST(p_BindBufferMemory(dev, ob, omem, 0));
    float *out;
    MUST(p_MapMemory(dev, omem, 0, VK_WHOLE_SIZE, 0, (void **)&out));
    memset(out, 0xff, osize); /* NaN: an unwritten pixel cannot pass */

    VkDescriptorImageInfo dii = {VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo dbi = {ob, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wr[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &dii},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi}};
    p_UpdateDescriptorSets(dev, 2, wr, 0, NULL);

    MUST(p_ResetCommandBuffer(cmd, 0));
    VkCommandBufferBeginInfo cbbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                     .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    MUST(p_BeginCommandBuffer(cmd, &cbbi));
    /* acquire from the producer (decoder / foreign queue), as ANGLE does for AHB images */
    VkImageMemoryBarrier ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                               .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT, .dstQueueFamilyIndex = qfi, .image = img,
                               .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    p_CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL,
                         1, &ib);
    p_CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    p_CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
    uint32_t pc[2] = {c->w, c->h};
    p_CmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
    p_CmdDispatch(cmd, (c->w + 7) / 8, (c->h + 7) / 8, 1);
    VkBufferMemoryBarrier bb = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                .dstAccessMask = VK_ACCESS_HOST_READ_BIT, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .buffer = ob, .size = VK_WHOLE_SIZE};
    p_CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1, &bb, 0, NULL);
    MUST(p_EndCommandBuffer(cmd));
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
    MUST(p_QueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    MUST(p_QueueWaitIdle(queue));

    double maxd = 0, sum = 0;
    uint32_t bad = 0, worst_x = 0, worst_y = 0;
    for (uint32_t y = 0; y < c->h; y++)
        for (uint32_t x = 0; x < c->w; x++) {
            double ref[3];
            reference(f, x, y, fp.suggestedYcbcrModel, fp.suggestedYcbcrRange, c->chroma, ref);
            const float *g = out + ((size_t)y * c->w + x) * 4;
            for (int k = 0; k < 3; k++) {
                double v = g[k] != g[k] ? 99 : g[k] < 0 ? 0 : g[k] > 1 ? 1 : g[k];
                double d = fabs(v - ref[k]);
                sum += d;
                if (d > maxd) { maxd = d; worst_x = x; worst_y = y; }
                if (d > TOL) bad++;
            }
            if (g[3] != g[3] || fabs(g[3] - 1.0) > 1e-6) bad++;
        }
    {
        double ref[3];
        reference(f, worst_x, worst_y, fp.suggestedYcbcrModel, fp.suggestedYcbcrRange, c->chroma, ref);
        const float *g = out + ((size_t)worst_y * c->w + worst_x) * 4;
        check(bad == 0, "%u px x RGB vs CPU reference: max |d| %.3f/255 (mean %.4f/255), %u component(s) over %.0f/255; "
              "worst (%u,%u) GPU %.4f %.4f %.4f CPU %.4f %.4f %.4f", c->w * c->h, maxd * 255,
              sum * 255 / (3.0 * c->w * c->h), bad, TOL * 255, worst_x, worst_y, g[0], g[1], g[2], ref[0], ref[1], ref[2]);
    }

    p_DestroyBuffer(dev, ob, NULL);
    p_FreeMemory(dev, omem, NULL);
    p_DestroyDescriptorPool(dev, dpool, NULL);
    p_DestroyPipeline(dev, pipe, NULL);
    p_DestroyShaderModule(dev, sm, NULL);
    p_DestroyPipelineLayout(dev, pl, NULL);
    p_DestroyDescriptorSetLayout(dev, dsl, NULL);
    p_DestroyImageView(dev, view, NULL);
    p_DestroySampler(dev, smp, NULL);
    p_DestroySamplerYcbcrConversion(dev, conv, NULL);
    p_DestroyImage(dev, img, NULL);
    p_FreeMemory(dev, mem, NULL);
    fake_release(ahb);
}

/* layouts the HAL must refuse loudly, not mis-sample */
static void negative(void) {
    printf("negative cases\n");
    struct fake_ahb *f = fake_yuv(DRM_FORMAT_NV12, 256, 128, 128, 0, -1);
    AHardwareBuffer *ahb = (AHardwareBuffer *)f;
    VkAndroidHardwareBufferFormatPropertiesANDROID fp = {.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID};
    VkAndroidHardwareBufferPropertiesANDROID pp = {.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID, .pNext = &fp};
    f->h.format_modifier = 0x0300000000606014ull; /* a block-linear modifier */
    check(p_GetAndroidHardwareBufferPropertiesANDROID(dev, ahb, &pp) == VK_ERROR_INVALID_EXTERNAL_HANDLE,
          "NV12 handle with a block-linear modifier -> VK_ERROR_INVALID_EXTERNAL_HANDLE");
    f->h.format_modifier = DRM_FORMAT_MOD_LINEAR;
    int keep = f->h.fds[1];
    int other = memfd_create("not-the-dmabuf", MFD_CLOEXEC);
    f->h.fds[1] = other;
    check(p_GetAndroidHardwareBufferPropertiesANDROID(dev, ahb, &pp) == VK_ERROR_INVALID_EXTERNAL_HANDLE,
          "NV12 handle whose plane 1 fd is another buffer (disjoint) -> VK_ERROR_INVALID_EXTERNAL_HANDLE");
    f->h.fds[1] = keep;
    close(other);
    uint32_t off = f->h.offsets[1];
    f->h.offsets[1] = off + 16; /* off the driver's per-plane alignment (NVIDIA 615: 64 KiB, NVK: 32 B) */
    MUST(p_GetAndroidHardwareBufferPropertiesANDROID(dev, ahb, &pp));
    VkExternalFormatANDROID xf = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID, .externalFormat = fp.externalFormat};
    VkExternalMemoryImageCreateInfo emi = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, .pNext = &xf,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID};
    VkImageCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &emi, .imageType = VK_IMAGE_TYPE_2D,
                             .format = VK_FORMAT_UNDEFINED, .extent = {256, 128, 1}, .mipLevels = 1, .arrayLayers = 1,
                             .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                             .usage = VK_IMAGE_USAGE_SAMPLED_BIT};
    VkImage img;
    MUST(p_CreateImage(dev, &ici, NULL, &img));
    VkImportAndroidHardwareBufferInfoANDROID imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
                                                    .buffer = ahb};
    VkMemoryDedicatedAllocateInfo ded = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .pNext = &imp, .image = img};
    uint32_t type = 0;
    while (!(pp.memoryTypeBits & (1u << type))) type++;
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &ded,
                                .allocationSize = pp.allocationSize, .memoryTypeIndex = type};
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkResult r = p_AllocateMemory(dev, &mai, NULL, &mem);
    check(r == VK_ERROR_INVALID_EXTERNAL_HANDLE,
          "CbCr plane offset %u (off the per-plane alignment) -> vkAllocateMemory VK_ERROR_INVALID_EXTERNAL_HANDLE (got %d)",
          f->h.offsets[1], r);
    if (r == VK_SUCCESS) p_FreeMemory(dev, mem, NULL);
    p_DestroyImage(dev, img, NULL);
    f->h.offsets[1] = off;
    fake_release(ahb);
}

int main(int argc, char **argv) {
    /* The fd checks count the whole process. Mesa's shader disk cache writes each entry on its own nice-19
     * "disk$" threads (src/util/disk_cache.c:90-93, src/util/u_queue.c:258) and holds a <hash>.tmp descriptor
     * from open to close (src/util/disk_cache_os.c:883-972, disk_cache_write_item_to_disk), unordered with
     * those counts. The explicit disable also covers a cache at an explicit path (disk_cache_os.c:1182).
     * Lines: Mesa 26.2.5 with this repository's Mesa patches applied. Set before any Mesa library loads. */
    if (setenv("MESA_SHADER_CACHE_DISABLE", "true", 1) != 0) {
        perror("setenv MESA_SHADER_CACHE_DISABLE");
        return 2;
    }
    const char *halpath = argc > 1 ? argv[1] : "out/host/vulkan.nvidia.so";
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("ahb-yuv: HAL %s\n", halpath);
    void *so = dlopen(halpath, RTLD_NOW | RTLD_LOCAL);
    if (!so) { printf("FATAL: dlopen %s: %s\n", halpath, dlerror()); return 2; }
    hwvulkan_module_t *hmi = dlsym(so, HAL_MODULE_INFO_SYM_AS_STR);
    hmi->common.dso = so;
    hwvulkan_device_t *hal = NULL;
    if (hmi->common.methods->open(&hmi->common, HWVULKAN_DEVICE_0, (hw_device_t **)&hal) != 0) return 2;
    gipa = hal->GetInstanceProcAddr;
    VkApplicationInfo ai = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "ahb-yuv",
                            .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai};
    MUST(hal->CreateInstance(&ici, NULL, &inst));
    gdpa = ip("vkGetDeviceProcAddr");
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices = ip("vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties = ip("vkGetPhysicalDeviceProperties");
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetQF = ip("vkGetPhysicalDeviceQueueFamilyProperties");
    PFN_vkGetPhysicalDeviceMemoryProperties GetMP = ip("vkGetPhysicalDeviceMemoryProperties");
    PFN_vkGetPhysicalDeviceFormatProperties2 GPDFP2 = ip("vkGetPhysicalDeviceFormatProperties2");
    PFN_vkCreateDevice CreateDevice = ip("vkCreateDevice");
    PFN_vkDestroyInstance DestroyInstance = ip("vkDestroyInstance");
    uint32_t npd = 8;
    VkPhysicalDevice pds[8];
    MUST(EnumeratePhysicalDevices(inst, &npd, pds));
    VkPhysicalDeviceProperties props;
    for (uint32_t i = 0; i < npd && !pd; i++) {
        GetPhysicalDeviceProperties(pds[i], &props);
        if (props.vendorID == 0x10de) pd = pds[i];
    }
    if (!pd) { printf("FATAL: no NVIDIA physical device\n"); return 2; }
    GetMP(pd, &memprops);
    printf("  device: %s\n", props.deviceName);

    /* the driver's LINEAR modifier entry for each YUV format (what the HAL imports with) */
    const VkFormat yf[2] = {VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16};
    for (int i = 0; i < 2; i++) {
        VkDrmFormatModifierProperties2EXT mp[16];
        VkDrmFormatModifierPropertiesList2EXT ml = {.sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT,
                                                    .drmFormatModifierCount = 16, .pDrmFormatModifierProperties = mp};
        VkFormatProperties2 fp = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &ml};
        GPDFP2(pd, yf[i], &fp);
        bool lin2 = false;
        for (uint32_t k = 0; k < ml.drmFormatModifierCount; k++)
            if (mp[k].drmFormatModifier == DRM_FORMAT_MOD_LINEAR && mp[k].drmFormatModifierPlaneCount == 2) {
                lin2 = true;
                printf("  VkFormat %d: LINEAR, 2 planes, features 0x%llx\n", yf[i],
                       (unsigned long long)mp[k].drmFormatModifierTilingFeatures);
            }
        check(lin2, "VkFormat %d lists DRM_FORMAT_MOD_LINEAR with 2 planes (%u modifier(s))", yf[i], ml.drmFormatModifierCount);
    }

    uint32_t nq = 16;
    VkQueueFamilyProperties qf[16];
    GetQF(pd, &nq, qf);
    qfi = UINT32_MAX;
    for (uint32_t i = 0; i < nq && qfi == UINT32_MAX; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) qfi = i;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qfi,
                                   .queueCount = 1, .pQueuePriorities = &prio};
    const char *dexts[] = {VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME, "VK_EXT_queue_family_foreign",
                           "VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf",
                           "VK_EXT_image_drm_format_modifier", "VK_KHR_image_format_list"};
    VkPhysicalDeviceVulkan11Features v11 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                                            .samplerYcbcrConversion = VK_TRUE};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &v11, .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci, .enabledExtensionCount = 6, .ppEnabledExtensionNames = dexts};
    MUST(CreateDevice(pd, &dci, NULL, &dev));
#define D(name) p_##name = dp("vk" #name);
    FNS(D)
#undef D
    p_GetDeviceQueue(dev, qfi, 0, &queue);
    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = qfi};
    MUST(p_CreateCommandPool(dev, &cpci, NULL, &pool));
    VkCommandBufferAllocateInfo cbai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    MUST(p_AllocateCommandBuffers(dev, &cbai, &cmd));

    const int fds0 = count_fds();
    /* Dataspace values (graphics-common Dataspace.aidl): STANDARD << 16, TRANSFER << 22, RANGE << 27 */
    const int64_t BT601_625_LIMITED = (2 << 16) | (3 << 22) | (2 << 27);
    const int64_t BT2020_LIMITED = (6 << 16) | (7 << 22) | (2 << 27);
    const int64_t BT709_FULL = (1 << 16) | (3 << 22) | (1 << 27);
    const struct ycase cases[] = {
        {"nv12-gralloc-nearest", DRM_FORMAT_NV12, 854, 480, 480, 0, -1, VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709,
         VK_SAMPLER_YCBCR_RANGE_ITU_NARROW, VK_FILTER_NEAREST},
        {"nv12-gralloc-linear", DRM_FORMAT_NV12, 854, 480, 480, 0, -1, VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709,
         VK_SAMPLER_YCBCR_RANGE_ITU_NARROW, VK_FILTER_LINEAR},
        {"nv12-1080p-padded-bt601", DRM_FORMAT_NV12, 1920, 1080, 1088, 0, BT601_625_LIMITED,
         VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601, VK_SAMPLER_YCBCR_RANGE_ITU_NARROW, VK_FILTER_LINEAR},
        {"nv12-odd-chroma-rows-gap-709full", DRM_FORMAT_NV12, 640, 362, 362, 8192, BT709_FULL,
         VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709, VK_SAMPLER_YCBCR_RANGE_ITU_FULL, VK_FILTER_NEAREST},
        {"p010-gralloc-bt2020", DRM_FORMAT_P010, 640, 360, 360, 0, BT2020_LIMITED,
         VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020, VK_SAMPLER_YCBCR_RANGE_ITU_NARROW, VK_FILTER_NEAREST},
        {"p010-gralloc-linear", DRM_FORMAT_P010, 1280, 720, 720, 0, -1, VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709,
         VK_SAMPLER_YCBCR_RANGE_ITU_NARROW, VK_FILTER_LINEAR},
    };
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) run_case(&cases[i]);
    negative();
    /* lifetime: 50 import cycles must not leak fds or fake AHB references */
    for (int i = 0; i < 50; i++) {
        struct fake_ahb *f = fake_yuv(DRM_FORMAT_NV12, 320, 240, 240, 0, 0);
        VkAndroidHardwareBufferFormatPropertiesANDROID fp = {.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID};
        VkAndroidHardwareBufferPropertiesANDROID pp = {.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID, .pNext = &fp};
        MUST(p_GetAndroidHardwareBufferPropertiesANDROID(dev, (AHardwareBuffer *)f, &pp));
        VkExternalFormatANDROID xf = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID, .externalFormat = fp.externalFormat};
        VkExternalMemoryImageCreateInfo emi = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, .pNext = &xf,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID};
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &emi, .imageType = VK_IMAGE_TYPE_2D,
                                .extent = {320, 240, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_SAMPLED_BIT};
        VkImage img;
        MUST(p_CreateImage(dev, &ci, NULL, &img));
        VkImportAndroidHardwareBufferInfoANDROID imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
                                                        .buffer = (AHardwareBuffer *)f};
        VkMemoryDedicatedAllocateInfo ded = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .pNext = &imp, .image = img};
        uint32_t type = 0;
        while (!(pp.memoryTypeBits & (1u << type))) type++;
        VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &ded,
                                    .allocationSize = pp.allocationSize, .memoryTypeIndex = type};
        VkDeviceMemory mem;
        MUST(p_AllocateMemory(dev, &mai, NULL, &mem));
        MUST(p_BindImageMemory(dev, img, mem, 0));
        fake_release((AHardwareBuffer *)f); /* the memory holds its own reference */
        if (i & 1) { p_DestroyImage(dev, img, NULL); p_FreeMemory(dev, mem, NULL); }
        else { p_FreeMemory(dev, mem, NULL); p_DestroyImage(dev, img, NULL); }
    }
    const int fds1 = count_fds();
    check(fds1 == fds0 && fake_live == 0, "50 import cycles: fds %d -> %d, live fake AHBs %d", fds0, fds1, fake_live);
    p_DestroyCommandPool(dev, pool, NULL);
    p_DestroyDevice(dev, NULL);
    DestroyInstance(inst, NULL);
    printf("ahb-yuv: %d failure(s) -> %s\n", failures, failures ? "FAIL" : "OK");
    return failures ? 1 : 0;
}
