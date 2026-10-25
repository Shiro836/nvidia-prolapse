/* ahb-roundtrip: host test of vulkan.nvidia.so's VK_ANDROID_external_memory_android_hardware_buffer
 * on the real GPU, driving the sequence ANGLE and Skia/HWUI use (AHB-CONTRACT.md).
 *
 * The host has no libnativewindow: this executable exports `nvhal_test_ahb_fns` (linked -rdynamic),
 * a fake AHardwareBuffer API whose allocator mirrors nvidia-prolapse/gralloc/nvidia_gbm_wrapper.c: an exportable
 * VkImage with DRM format modifiers (block-linear for GPU buffers, LINEAR + host-visible memory for
 * CPU buffers), exported as a dma-buf and described by a synthetic minigbm cros_gralloc_handle.
 * The "original" image of each fake AHB stays alive, so pixels written through the HAL's AHB
 * image can be read back through it and vice versa.
 *
 *   1. extension list, image-format query with the AHB handle type (HWUI's call), external buffer query;
 *   2. ANGLE's import: properties -> vkCreateImage(OPTIMAL, MUTABLE + format list, AHB, external
 *      format 0) -> vkGetImageMemoryRequirements2 before memory (invalid usage, logged) ->
 *      vkAllocateMemory(import, dedicated) -> vkBindImageMemory -> views -> FOREIGN-queue barriers,
 *      clear through the AHB image, read through the original; the reverse direction with
 *      sync2 barriers and copy2 commands; private data set before realisation;
 *   3. Skia's import (vkBindImageMemory2), memory freed before the image;
 *   4. export: VkExportMemoryAllocateInfo{AHB} -> vkGetMemoryAndroidHardwareBufferANDROID -> import
 *      that AHB into a second image -> pixels;
 *   5. BLOB AHB as a VkBuffer (ANGLE's GL_EXT_external_buffer path);
 *   6. negative cases and fd/AHB accounting over 100 cycles.
 * Exit 0 only when every check passed.
 *
 * usage: ahb-roundtrip HAL.so [--linear] [--api 1.0] */
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <drm/drm_fourcc.h>
#include <vulkan/vulkan.h>
#include <hardware/hwvulkan.h>
#include <cros_gralloc_handle.h>
#include "nvhal_ahb.h"

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
#define VKR(call) vkr(#call, (call))
static VkResult vkr(const char *what, VkResult r) {
    printf("  %-74.74s -> %d%s\n", what, r, r == VK_SUCCESS ? "" : " (!)");
    return r;
}
#define MUST(call) do { if (VKR(call) != VK_SUCCESS) { printf("FATAL\n"); exit(2); } } while (0)

static PFN_vkGetInstanceProcAddr gipa;
static PFN_vkGetDeviceProcAddr gdpa;
static VkInstance inst;
static VkPhysicalDevice pd;
static VkDevice dev;
static int api10, linear;
static uint32_t qfi;
static VkPhysicalDeviceMemoryProperties memprops;

static void *ip(const char *name) {
    void *p = (void *)gipa(inst, name);
    if (!p) { printf("FATAL: GIPA(instance, %s) = NULL\n", name); exit(2); }
    return p;
}
static void *dp(const char *name) {
    void *p = (void *)gdpa(dev, name);
    if (!p) { printf("FATAL: GDPA(device, %s) = NULL\n", name); exit(2); }
    return p;
}
static const char *v11(const char *core, const char *khr) { return api10 ? khr : core; }

static int count_fds(void) {
    int n = 0;
    DIR *d = opendir("/proc/self/fd");
    for (struct dirent *e; d && (e = readdir(d));) n += e->d_name[0] != '.';
    if (d) closedir(d);
    return n - 1;
}
static int count_dmabufs(void) {
    int n = 0;
    for (int i = 0; i < 1024; i++) {
        char p[64], t[256];
        snprintf(p, sizeof p, "/proc/self/fd/%d", i);
        ssize_t l = readlink(p, t, sizeof t - 1);
        if (l > 0) { t[l] = 0; n += !strncmp(t, "/dmabuf:", 8); }
    }
    return n;
}

/* ------------------------------------------------------------------ device functions */
#define FNS(X) X(GetDeviceQueue) X(CreateImage) X(DestroyImage) X(AllocateMemory) X(FreeMemory) X(BindImageMemory) \
    X(GetImageMemoryRequirements) X(GetMemoryFdKHR) X(GetImageDrmFormatModifierPropertiesEXT) \
    X(GetImageSubresourceLayout) X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) X(BindBufferMemory) \
    X(MapMemory) X(CreateCommandPool) X(DestroyCommandPool) X(AllocateCommandBuffers) X(BeginCommandBuffer) \
    X(EndCommandBuffer) X(ResetCommandBuffer) X(CmdPipelineBarrier) X(CmdClearColorImage) X(CmdCopyImageToBuffer) \
    X(CmdCopyBufferToImage) X(CmdFillBuffer) X(CmdCopyBuffer) X(QueueSubmit) X(QueueWaitIdle) X(CreateFence) \
    X(DestroyFence) X(WaitForFences) X(DestroyDevice) X(CreateImageView) X(DestroyImageView) \
    X(GetAndroidHardwareBufferPropertiesANDROID) X(GetMemoryAndroidHardwareBufferANDROID)
#define DECL(n) static PFN_vk##n p_##n;
FNS(DECL)
#undef DECL
static PFN_vkBindImageMemory2 p_BindImageMemory2;
static PFN_vkGetImageMemoryRequirements2 p_GetImageMemoryRequirements2;
/* 1.3 (absent with --api 1.0) */
static PFN_vkCmdPipelineBarrier2 p_CmdPipelineBarrier2;
static PFN_vkCmdCopyImageToBuffer2 p_CmdCopyImageToBuffer2;
static PFN_vkCreatePrivateDataSlot p_CreatePrivateDataSlot;
static PFN_vkDestroyPrivateDataSlot p_DestroyPrivateDataSlot;
static PFN_vkSetPrivateData p_SetPrivateData;
static PFN_vkGetPrivateData p_GetPrivateData;

static uint32_t mem_type_want(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}
static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
    uint32_t t = mem_type_want(bits, want);
    if (t == UINT32_MAX) { printf("FATAL: no memory type in 0x%x with flags 0x%x\n", bits, want); exit(2); }
    return t;
}
/* ANGLE's choice (vk_utils.cpp:382-438): DEVICE_LOCAL, else for external memory any allowed type */
static uint32_t angle_mem_type(uint32_t bits) {
    uint32_t t = mem_type_want(bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    for (uint32_t i = 0; i < memprops.memoryTypeCount && t == UINT32_MAX; i++)
        if (bits & (1u << i)) t = i;
    return t;
}

static VkQueue queue;
static VkCommandPool pool;
static VkCommandBuffer cmd;
static void begin(void) {
    MUST(p_ResetCommandBuffer(cmd, 0));
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    MUST(p_BeginCommandBuffer(cmd, &bi));
}
static void barrier_q(VkImage img, VkImageLayout from, VkImageLayout to, uint32_t srcq, uint32_t dstq) {
    VkImageMemoryBarrier b = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .oldLayout = from, .newLayout = to, .srcQueueFamilyIndex = srcq, .dstQueueFamilyIndex = dstq,
        .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    p_CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL,
                         1, &b);
}
static void barrier(VkImage img, VkImageLayout from, VkImageLayout to) {
    barrier_q(img, from, to, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED);
}
static void submit_wait(const char *what) {
    MUST(p_EndCommandBuffer(cmd));
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence f;
    MUST(p_CreateFence(dev, &fci, NULL, &f));
    printf("  %s:\n", what);
    MUST(p_QueueSubmit(queue, 1, &si, f));
    MUST(p_WaitForFences(dev, 1, &f, VK_TRUE, 10ull * 1000000000ull));
    p_DestroyFence(dev, f, NULL);
}

struct hostbuf { VkBuffer buf; VkDeviceMemory mem; uint8_t *ptr; };
static struct hostbuf hostbuf(VkDeviceSize size) {
    struct hostbuf h;
    VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                              .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    MUST(p_CreateBuffer(dev, &bci, NULL, &h.buf));
    VkMemoryRequirements mr;
    p_GetBufferMemoryRequirements(dev, h.buf, &mr);
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
        .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    MUST(p_AllocateMemory(dev, &mai, NULL, &h.mem));
    MUST(p_BindBufferMemory(dev, h.buf, h.mem, 0));
    MUST(p_MapMemory(dev, h.mem, 0, VK_WHOLE_SIZE, 0, (void **)&h.ptr));
    return h;
}
static void hostbuf_free(struct hostbuf *h) {
    p_DestroyBuffer(dev, h->buf, NULL);
    p_FreeMemory(dev, h->mem, NULL);
}

/* ------------------------------------------------------------------ the fake AHardwareBuffer */

#define FAKE_MAGIC 0xfa4eb0ffu
struct fake_ahb {
    uint32_t magic;
    int refs;
    AHardwareBuffer_Desc desc;
    struct cros_gralloc_handle h;
    VkImage img;   /* the "original" (gralloc-side) image; VK_NULL_HANDLE for BLOB */
    VkBuffer buf;  /* BLOB backing */
    VkDeviceMemory mem;
    uint64_t modifier;
};
static int fake_live;
static bool quiet;

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
    if (f->img) p_DestroyImage(dev, f->img, NULL);
    if (f->buf) p_DestroyBuffer(dev, f->buf, NULL);
    p_FreeMemory(dev, f->mem, NULL);
    if (f->h.fds[0] >= 0) close(f->h.fds[0]);
    f->magic = 0;
    free(f);
    fake_live--;
}

static VkFormat ahb_vk(uint32_t f, uint32_t *fourcc) {
    switch (f) {
    case AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM: *fourcc = DRM_FORMAT_ABGR8888; return VK_FORMAT_R8G8B8A8_UNORM;
    case AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM: *fourcc = DRM_FORMAT_XBGR8888; return VK_FORMAT_R8G8B8A8_UNORM;
    case AHARDWAREBUFFER_FORMAT_R5G6B5_UNORM: *fourcc = DRM_FORMAT_RGB565; return VK_FORMAT_R5G6B5_UNORM_PACK16;
    case NVHAL_AHB_FORMAT_B8G8R8A8_UNORM: *fourcc = DRM_FORMAT_ARGB8888; return VK_FORMAT_B8G8R8A8_UNORM;
    default: *fourcc = 0; return VK_FORMAT_UNDEFINED;
    }
}

/* the gralloc backend's choice (nvidia_gbm_wrapper.c nvg_alloc), condensed */
static int fake_allocate(const AHardwareBuffer_Desc *desc, AHardwareBuffer **out) {
    struct fake_ahb *f = calloc(1, sizeof *f);
    f->magic = FAKE_MAGIC;
    f->refs = 1;
    f->desc = *desc;
    for (int i = 0; i < DRV_MAX_FDS; i++) f->h.fds[i] = -1;
    const bool cpu = desc->usage & (AHARDWAREBUFFER_USAGE_CPU_READ_MASK | AHARDWAREBUFFER_USAGE_CPU_WRITE_MASK);
    int fd = -1;
    VkDeviceSize size;
    if (desc->format == AHARDWAREBUFFER_FORMAT_BLOB) {
        VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                               .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
        VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = desc->width,
                                  .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
        if (p_CreateBuffer(dev, &bci, NULL, &f->buf) != VK_SUCCESS) return -1;
        VkMemoryRequirements mr;
        p_GetBufferMemoryRequirements(dev, f->buf, &mr);
        VkExportMemoryAllocateInfo ex = {.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
                                         .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
        VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &ex,
                                    .allocationSize = mr.size,
                                    .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        if (p_AllocateMemory(dev, &mai, NULL, &f->mem) != VK_SUCCESS) return -1;
        p_BindBufferMemory(dev, f->buf, f->mem, 0);
        size = mr.size;
        f->h.format = DRM_FORMAT_R8;
        f->h.strides[0] = desc->width;
        f->modifier = DRM_FORMAT_MOD_LINEAR;
    } else {
        uint32_t fourcc;
        VkFormat vf = ahb_vk(desc->format, &fourcc);
        if (vf == VK_FORMAT_UNDEFINED) return -1;
        PFN_vkGetPhysicalDeviceFormatProperties2 fp2 = ip(v11("vkGetPhysicalDeviceFormatProperties2",
                                                              "vkGetPhysicalDeviceFormatProperties2KHR"));
        VkDrmFormatModifierPropertiesList2EXT ml = {.sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT};
        VkFormatProperties2 fp = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &ml};
        fp2(pd, vf, &fp);
        VkDrmFormatModifierProperties2EXT mp[64];
        ml.pDrmFormatModifierProperties = mp;
        fp2(pd, vf, &fp);
        const VkFormatFeatureFlags2 need = VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT |
                                           VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT |
                                           (cpu ? 0 : VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT);
        uint64_t mods[64];
        uint32_t n = 0;
        bool color = true, storage = true;
        for (uint32_t i = 0; i < ml.drmFormatModifierCount; i++) {
            bool lin = mp[i].drmFormatModifier == DRM_FORMAT_MOD_LINEAR;
            if (lin != cpu || mp[i].drmFormatModifierPlaneCount != 1 ||
                (mp[i].drmFormatModifierTilingFeatures & need) != need)
                continue;
            color &= !!(mp[i].drmFormatModifierTilingFeatures & VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT);
            storage &= !!(mp[i].drmFormatModifierTilingFeatures & VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT);
            mods[n++] = mp[i].drmFormatModifier;
        }
        if (!n) return -1;
        VkFormat srgb = vf == VK_FORMAT_R8G8B8A8_UNORM ? VK_FORMAT_R8G8B8A8_SRGB
                        : vf == VK_FORMAT_B8G8R8A8_UNORM ? VK_FORMAT_B8G8R8A8_SRGB : VK_FORMAT_UNDEFINED;
        VkFormat views[2] = {vf, srgb};
        VkImageFormatListCreateInfo fl = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, .viewFormatCount = 2,
                                          .pViewFormats = views};
        VkImageDrmFormatModifierListCreateInfoEXT mli = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
            .pNext = srgb ? &fl : NULL, .drmFormatModifierCount = n, .pDrmFormatModifiers = mods};
        VkExternalMemoryImageCreateInfo emi = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
                                               .pNext = &mli, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &emi,
                                .flags = srgb ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0,
                                .imageType = VK_IMAGE_TYPE_2D, .format = vf, .extent = {desc->width, desc->height, 1},
                                .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                         VK_IMAGE_USAGE_TRANSFER_DST_BIT | (color ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0) |
                                         (storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0),
                                /* PREINITIALIZED: the first transition of the original keeps bytes
                                 * written through the HAL's image */
                                .initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED};
        if (p_CreateImage(dev, &ci, NULL, &f->img) != VK_SUCCESS) return -1;
        VkMemoryRequirements mr;
        p_GetImageMemoryRequirements(dev, f->img, &mr);
        uint32_t type = cpu ? mem_type_want(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                            : UINT32_MAX;
        if (type == UINT32_MAX) type = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkExportMemoryAllocateInfo ex = {.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
                                         .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
        VkMemoryDedicatedAllocateInfo ded = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .pNext = &ex,
                                             .image = f->img};
        VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &ded,
                                    .allocationSize = mr.size, .memoryTypeIndex = type};
        if (p_AllocateMemory(dev, &mai, NULL, &f->mem) != VK_SUCCESS) return -1;
        p_BindImageMemory(dev, f->img, f->mem, 0);
        VkImageDrmFormatModifierPropertiesEXT mp1 = {.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT};
        p_GetImageDrmFormatModifierPropertiesEXT(dev, f->img, &mp1);
        VkImageSubresource sub = {.aspectMask = VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT};
        VkSubresourceLayout lay;
        p_GetImageSubresourceLayout(dev, f->img, &sub, &lay);
        f->modifier = mp1.drmFormatModifier;
        f->h.format = fourcc;
        f->h.strides[0] = (uint32_t)lay.rowPitch;
        f->h.offsets[0] = (uint32_t)lay.offset;
        f->h.sizes[0] = (uint32_t)lay.size;
        f->h.pixel_stride = (uint32_t)(lay.rowPitch / (vf == VK_FORMAT_R5G6B5_UNORM_PACK16 ? 2 : 4));
        f->desc.stride = f->h.pixel_stride;
        size = mr.size;
        if (!quiet)
            printf("    fake gralloc: %ux%u AHB format 0x%x usage 0x%llx -> %s modifier 0x%016llx, stride %u B, "
                   "offset %u, %llu B, memory type %u\n", desc->width, desc->height, desc->format,
                   (unsigned long long)desc->usage, cpu ? "CPU" : "GPU", (unsigned long long)f->modifier,
                   f->h.strides[0], f->h.offsets[0], (unsigned long long)mr.size, type);
    }
    VkMemoryGetFdInfoKHR gfi = {.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = f->mem,
                                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    if (p_GetMemoryFdKHR(dev, &gfi, &fd) != VK_SUCCESS) return -1;
    f->h.base.version = sizeof(native_handle_t);
    f->h.base.numFds = 1;
    f->h.base.numInts = (int)CROS_GRALLOC_HANDLE_NUM_INTS_TOTAL - 1;
    f->h.fds[0] = fd;
    f->h.width = desc->width;
    f->h.height = desc->height;
    f->h.format_modifier = f->modifier;
    f->h.magic = CROS_GRALLOC_MAGIC;
    f->h.droid_format = (int32_t)desc->format;
    f->h.usage = (int32_t)desc->usage;
    f->h.num_planes = 1;
    f->h.total_size = size;
    fake_live++;
    *out = (AHardwareBuffer *)f;
    return 0;
}

__attribute__((visibility("default"))) struct nvhal_ahb_fns nvhal_test_ahb_fns = {
    fake_describe, fake_handle, fake_acquire, fake_release, fake_allocate,
};

/* ------------------------------------------------------------------ helpers for the AHB sequence */

#define W 256u
#define H 256u
static const uint64_t GPU_USAGE = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER;
static const uint64_t CPU_USAGE = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                                  AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;

static AHardwareBuffer *new_ahb(uint32_t format, uint64_t usage) {
    AHardwareBuffer_Desc d = {.width = W, .height = H, .layers = 1, .format = format, .usage = usage};
    AHardwareBuffer *b = NULL;
    if (fake_allocate(&d, &b) != 0) { printf("FATAL: fake allocate failed\n"); exit(2); }
    return b;
}

/* ANGLE's vkCreateImage for an AHB (HBS:413-572): usage from the AHB usage, OPTIMAL, MUTABLE +
 * {UNORM, SRGB} list, VkExternalMemoryImageCreateInfo{AHB} -> VkExternalFormatANDROID{0} */
static VkFormat angle_views[2] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB};
static VkResult angle_image(uint64_t ahb_usage, VkImage *out) {
    VkExternalFormatANDROID xf = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID, .externalFormat = 0};
    VkExternalMemoryImageCreateInfo emi = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, .pNext = &xf,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID};
    VkImageFormatListCreateInfo fl = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, .pNext = &emi,
                                      .viewFormatCount = 2, .pViewFormats = angle_views};
    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (ahb_usage & AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE) usage |= VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
    if (ahb_usage & AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER) usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &fl,
                            .flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {W, H, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
                            .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    return p_CreateImage(dev, &ci, NULL, out);
}

static VkResult import(AHardwareBuffer *b, VkImage img, VkDeviceMemory *mem, VkAndroidHardwareBufferPropertiesANDROID *pp) {
    VkAndroidHardwareBufferPropertiesANDROID p = {.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
    VkResult r = p_GetAndroidHardwareBufferPropertiesANDROID(dev, b, &p);
    if (r != VK_SUCCESS) return r;
    if (pp) *pp = p;
    VkImportAndroidHardwareBufferInfoANDROID imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
                                                     .buffer = b};
    VkMemoryDedicatedAllocateInfo ded = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .pNext = &imp,
                                         .image = img};
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &ded,
                                .allocationSize = p.allocationSize, .memoryTypeIndex = angle_mem_type(p.memoryTypeBits)};
    return p_AllocateMemory(dev, &mai, NULL, mem);
}

static uint32_t count_color(const uint8_t *p, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    uint32_t good = 0;
    for (uint32_t i = 0; i < W * H; i++) good += p[4 * i] == r && p[4 * i + 1] == g && p[4 * i + 2] == b && p[4 * i + 3] == a;
    return good;
}

/* ------------------------------------------------------------------ main */

int main(int argc, char **argv) {
    const char *halpath = argc > 1 ? argv[1] : "out/host/vulkan.nvidia.so";
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--api") && i + 1 < argc) api10 = !strcmp(argv[++i], "1.0");
        else if (!strcmp(argv[i], "--linear")) linear = 1;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("ahb-roundtrip: HAL %s, instance apiVersion %s, %s AHBs\n", halpath, api10 ? "1.0" : "1.3",
           linear ? "CPU-usage (LINEAR, host-visible)" : "GPU-usage (block-linear)");
    void *so = dlopen(halpath, RTLD_NOW | RTLD_LOCAL);
    if (!so) { printf("FATAL: dlopen %s: %s\n", halpath, dlerror()); return 2; }
    hwvulkan_module_t *hmi = dlsym(so, HAL_MODULE_INFO_SYM_AS_STR);
    hmi->common.dso = so;
    hwvulkan_device_t *hal = NULL;
    if (hmi->common.methods->open(&hmi->common, HWVULKAN_DEVICE_0, (hw_device_t **)&hal) != 0) return 2;
    gipa = hal->GetInstanceProcAddr;

    VkApplicationInfo ai = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "ahb-roundtrip",
                            .apiVersion = api10 ? VK_API_VERSION_1_0 : VK_API_VERSION_1_3};
    const char *iexts[] = {"VK_KHR_get_physical_device_properties2", "VK_KHR_external_memory_capabilities"};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai,
                                .enabledExtensionCount = api10 ? 2 : 0, .ppEnabledExtensionNames = iexts};
    MUST(hal->CreateInstance(&ici, NULL, &inst));
    gdpa = ip("vkGetDeviceProcAddr");
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices = ip("vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties = ip("vkGetPhysicalDeviceProperties");
    PFN_vkGetPhysicalDeviceImageFormatProperties2 GPDIFP2 =
        ip(v11("vkGetPhysicalDeviceImageFormatProperties2", "vkGetPhysicalDeviceImageFormatProperties2KHR"));
    PFN_vkGetPhysicalDeviceExternalBufferProperties GPDEBP =
        ip(v11("vkGetPhysicalDeviceExternalBufferProperties", "vkGetPhysicalDeviceExternalBufferPropertiesKHR"));
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetQF = ip("vkGetPhysicalDeviceQueueFamilyProperties");
    PFN_vkGetPhysicalDeviceMemoryProperties GetMP = ip("vkGetPhysicalDeviceMemoryProperties");
    PFN_vkEnumerateDeviceExtensionProperties EnumDE = ip("vkEnumerateDeviceExtensionProperties");
    PFN_vkCreateDevice CreateDevice = ip("vkCreateDevice");
    PFN_vkDestroyInstance DestroyInstance = ip("vkDestroyInstance");
    uint32_t npd = 0;
    MUST(EnumeratePhysicalDevices(inst, &npd, NULL));
    VkPhysicalDevice pds[8];
    if (npd > 8) npd = 8;
    MUST(EnumeratePhysicalDevices(inst, &npd, pds));
    VkPhysicalDeviceProperties props;
    for (uint32_t i = 0; i < npd && !pd; i++) {
        GetPhysicalDeviceProperties(pds[i], &props);
        if (props.vendorID == 0x10de) pd = pds[i];
    }
    if (!pd) { printf("FATAL: no NVIDIA physical device\n"); return 2; }
    GetMP(pd, &memprops);
    printf("  device: %s; memory types:", props.deviceName);
    for (uint32_t i = 0; i < memprops.memoryTypeCount; i++) printf(" %u=0x%x", i, memprops.memoryTypes[i].propertyFlags);
    printf("\n");

    /* ---- 1. extension list ---- */
    uint32_t n = 0;
    MUST(EnumDE(pd, NULL, &n, NULL));
    VkExtensionProperties *de = calloc(n, sizeof *de);
    MUST(EnumDE(pd, NULL, &n, de));
    int ahb_spec = -1, qff = 0, rt = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (!strcmp(de[i].extensionName, VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME))
            ahb_spec = (int)de[i].specVersion;
        qff |= !strcmp(de[i].extensionName, "VK_EXT_queue_family_foreign");
        rt |= !strcmp(de[i].extensionName, "VK_KHR_ray_query") || !strcmp(de[i].extensionName, "VK_NV_cuda_kernel_launch");
    }
    check(ahb_spec == 5 && qff, "device lists VK_ANDROID_external_memory_android_hardware_buffer spec %d and "
          "VK_EXT_queue_family_foreign (%u extensions)", ahb_spec, n);
    check(!rt, "the missing-library group (ray tracing / CUDA) is hidden");

    /* ---- 1b. image-format query with the AHB handle type, exactly HWUI's (VKS:225-268) ---- */
    {
        VkPhysicalDeviceExternalImageFormatInfo ei = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID};
        VkPhysicalDeviceImageFormatInfo2 fi = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, .pNext = &ei,
            .format = VK_FORMAT_R8G8B8A8_UNORM, .type = VK_IMAGE_TYPE_2D, .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VkAndroidHardwareBufferUsageANDROID hu = {.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_USAGE_ANDROID};
        VkImageFormatProperties2 fp = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2, .pNext = &hu};
        VkResult r = VKR(GPDIFP2(pd, &fi, &fp));
        check(r == VK_SUCCESS && hu.androidHardwareBufferUsage == GPU_USAGE && fp.pNext == &hu && !hu.pNext,
              "HWUI query (OPTIMAL, COLOR|SAMPLED|TRANSFER, only VkAndroidHardwareBufferUsageANDROID chained): "
              "androidHardwareBufferUsage 0x%llx, max %ux%u, mips %u, layers %u", (unsigned long long)hu.androidHardwareBufferUsage,
              fp.imageFormatProperties.maxExtent.width, fp.imageFormatProperties.maxExtent.height,
              fp.imageFormatProperties.maxMipLevels, fp.imageFormatProperties.maxArrayLayers);
        VkExternalImageFormatProperties ep = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES, .pNext = &hu};
        fp.pNext = &ep;
        fi.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
        VkImageFormatListCreateInfo fl = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, .viewFormatCount = 2,
                                          .pViewFormats = angle_views};
        ei.pNext = &fl;
        r = VKR(GPDIFP2(pd, &fi, &fp));
        VkExternalMemoryProperties *m = &ep.externalMemoryProperties;
        check(r == VK_SUCCESS &&
              m->externalMemoryFeatures == (VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT | VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT |
                                            VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) &&
              m->compatibleHandleTypes == VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID,
              "query with MUTABLE + format list and VkExternalImageFormatProperties: features 0x%x compatible 0x%x",
              m->externalMemoryFeatures, m->compatibleHandleTypes);
        fi.format = VK_FORMAT_R8G8B8A8_SRGB;
        fi.flags = 0;
        ei.pNext = NULL;
        check(VKR(GPDIFP2(pd, &fi, &fp)) == VK_ERROR_FORMAT_NOT_SUPPORTED,
              "AHB query for R8G8B8A8_SRGB (no AHB format) -> VK_ERROR_FORMAT_NOT_SUPPORTED");
        fi.format = VK_FORMAT_R8G8B8A8_UNORM;
        fi.tiling = VK_IMAGE_TILING_LINEAR;
        fi.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        fp.pNext = &hu;
        check(VKR(GPDIFP2(pd, &fi, &fp)) == VK_SUCCESS, "AHB query LINEAR SAMPLED|TRANSFER (CPU-buffer modifier)");
        VkPhysicalDeviceExternalBufferInfo bi = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID};
        VkExternalBufferProperties ebp = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
        GPDEBP(pd, &bi, &ebp);
        check(ebp.externalMemoryProperties.externalMemoryFeatures == VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT &&
              ebp.externalMemoryProperties.compatibleHandleTypes == VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID,
              "external buffer query (AHB): features 0x%x (import only) compatible 0x%x",
              ebp.externalMemoryProperties.externalMemoryFeatures, ebp.externalMemoryProperties.compatibleHandleTypes);
    }

    /* ---- device ---- */
    uint32_t nq = 0;
    GetQF(pd, &nq, NULL);
    VkQueueFamilyProperties qf[16];
    if (nq > 16) nq = 16;
    GetQF(pd, &nq, qf);
    qfi = UINT32_MAX;
    for (uint32_t i = 0; i < nq && qfi == UINT32_MAX; i++)
        if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) qfi = i;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qfi,
                                   .queueCount = 1, .pQueuePriorities = &prio};
    const char *dexts[] = {VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME, "VK_EXT_queue_family_foreign",
                           "VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf", "VK_EXT_image_drm_format_modifier",
                           "VK_KHR_image_format_list",
                           /* 1.0 device: what 1.1 made core */
                           "VK_KHR_get_memory_requirements2", "VK_KHR_dedicated_allocation", "VK_KHR_bind_memory2",
                           "VK_KHR_external_memory", "VK_KHR_maintenance1", "VK_KHR_sampler_ycbcr_conversion"};
    VkPhysicalDeviceSynchronization2Features s2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
                                                   .synchronization2 = VK_TRUE};
    VkPhysicalDevicePrivateDataFeatures pdf = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES,
                                               .pNext = &s2, .privateData = VK_TRUE};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = api10 ? NULL : &pdf,
                              .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
                              .enabledExtensionCount = api10 ? 12 : 6, .ppEnabledExtensionNames = dexts};
    MUST(CreateDevice(pd, &dci, NULL, &dev));
#define D(name) p_##name = dp("vk" #name);
    FNS(D)
#undef D
    p_BindImageMemory2 = dp(v11("vkBindImageMemory2", "vkBindImageMemory2KHR"));
    p_GetImageMemoryRequirements2 = dp(v11("vkGetImageMemoryRequirements2", "vkGetImageMemoryRequirements2KHR"));
    if (!api10) {
        p_CmdPipelineBarrier2 = dp("vkCmdPipelineBarrier2");
        p_CmdCopyImageToBuffer2 = dp("vkCmdCopyImageToBuffer2");
        p_CreatePrivateDataSlot = dp("vkCreatePrivateDataSlot");
        p_DestroyPrivateDataSlot = dp("vkDestroyPrivateDataSlot");
        p_SetPrivateData = dp("vkSetPrivateData");
        p_GetPrivateData = dp("vkGetPrivateData");
    }
    p_GetDeviceQueue(dev, qfi, 0, &queue);
    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = qfi};
    MUST(p_CreateCommandPool(dev, &cpci, NULL, &pool));
    VkCommandBufferAllocateInfo cbai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    MUST(p_AllocateCommandBuffers(dev, &cbai, &cmd));
    const uint64_t usage = linear ? CPU_USAGE : GPU_USAGE;
    struct hostbuf rb = hostbuf((VkDeviceSize)W * H * 4);
    VkBufferImageCopy region = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {W, H, 1}};
    VkImageSubresourceRange rng = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const int fds_at_start = count_fds(), dmabufs_at_start = count_dmabufs();

    /* ---- 2. ANGLE's import ---- */
    printf("-- ANGLE import sequence\n");
    AHardwareBuffer *ahbA = new_ahb(AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM, usage);
    struct fake_ahb *fa = F(ahbA);
    VkAndroidHardwareBufferFormatProperties2ANDROID fp2 = {.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_2_ANDROID};
    VkAndroidHardwareBufferFormatPropertiesANDROID fpr = {.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID,
                                                          .pNext = &fp2};
    VkAndroidHardwareBufferPropertiesANDROID ap = {.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID,
                                                   .pNext = &fpr};
    MUST(p_GetAndroidHardwareBufferPropertiesANDROID(dev, ahbA, &ap));
    off_t dsz = lseek(fa->h.fds[0], 0, SEEK_END);
    lseek(fa->h.fds[0], 0, SEEK_SET);
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                      VK_FORMAT_FEATURE_TRANSFER_DST_BIT | (linear ? 0 : VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT);
    check(ap.allocationSize == (VkDeviceSize)dsz && ap.memoryTypeBits && fpr.format == VK_FORMAT_R8G8B8A8_UNORM &&
          fpr.externalFormat != 0 && (fpr.formatFeatures & need) == need && fp2.format == fpr.format &&
          (fp2.formatFeatures & 0x7fffffffu) == fpr.formatFeatures && fpr.suggestedYcbcrModel == VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY,
          "properties: allocationSize %llu (dma-buf %lld), memoryTypeBits 0x%x, format %d, externalFormat %llu, "
          "formatFeatures 0x%x (features2 0x%llx)", (unsigned long long)ap.allocationSize, (long long)dsz,
          ap.memoryTypeBits, fpr.format, (unsigned long long)fpr.externalFormat, fpr.formatFeatures,
          (unsigned long long)fp2.formatFeatures);
    printf("    ANGLE's memory type for it: %u; Skia's (DEVICE_LOCAL required): %s\n", angle_mem_type(ap.memoryTypeBits),
           mem_type_want(ap.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == UINT32_MAX ? "none -> Skia refuses"
                                                                                                 : "found");
    VkImage wa = VK_NULL_HANDLE;
    VkResult cr = VKR(angle_image(usage, &wa));
    check(cr == VK_SUCCESS && wa != fa->img, "vkCreateImage(OPTIMAL, MUTABLE + format list, AHB, "
          "VkExternalFormatANDROID{0}) -> wrapper %p", (void *)(uintptr_t)wa);
    VkPrivateDataSlot slot = VK_NULL_HANDLE;
    if (!api10) {
        VkPrivateDataSlotCreateInfo psci = {.sType = VK_STRUCTURE_TYPE_PRIVATE_DATA_SLOT_CREATE_INFO};
        MUST(p_CreatePrivateDataSlot(dev, &psci, NULL, &slot));
        MUST(p_SetPrivateData(dev, VK_OBJECT_TYPE_IMAGE, (uint64_t)wa, slot, 0x1234abcd));
    }
    /* invalid usage (VUID-VkImageMemoryRequirementsInfo2-image-01897), answered from a probe + logged */
    VkMemoryDedicatedRequirements dr = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 mr2 = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dr};
    VkImageMemoryRequirementsInfo2 mri = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, .image = wa};
    p_GetImageMemoryRequirements2(dev, &mri, &mr2);
    check(mr2.memoryRequirements.size && mr2.memoryRequirements.size <= ap.allocationSize &&
          (mr2.memoryRequirements.memoryTypeBits & ap.memoryTypeBits) && dr.requiresDedicatedAllocation,
          "vkGetImageMemoryRequirements2 before memory: %llu B (<= allocationSize %llu), types 0x%x (overlap with 0x%x), "
          "requiresDedicated %u", (unsigned long long)mr2.memoryRequirements.size, (unsigned long long)ap.allocationSize,
          mr2.memoryRequirements.memoryTypeBits, ap.memoryTypeBits, dr.requiresDedicatedAllocation);
    VkDeviceMemory ma;
    check(VKR(import(ahbA, wa, &ma, NULL)) == VK_SUCCESS, "vkAllocateMemory(VkImportAndroidHardwareBufferInfoANDROID, dedicated wrapper)");
    check(fa->refs == 2, "the memory holds an AHB reference (refs %d)", fa->refs);
    check(VKR(p_BindImageMemory(dev, wa, ma, 0)) == VK_SUCCESS, "vkBindImageMemory (v1, as ANGLE)");
    if (!api10) {
        uint64_t v = 0;
        p_GetPrivateData(dev, VK_OBJECT_TYPE_IMAGE, (uint64_t)wa, slot, &v);
        check(v == 0x1234abcd, "private data set before the image had memory survives realisation (0x%llx)",
              (unsigned long long)v);
    }
    VkImageView views[2];
    for (int i = 0; i < 2; i++) {
        VkImageViewCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = wa,
                                     .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = angle_views[i], .subresourceRange = rng};
        check(VKR(p_CreateImageView(dev, &vci, NULL, &views[i])) == VK_SUCCESS, "vkCreateImageView(wrapper, %s)",
              i ? "R8G8B8A8_SRGB" : "R8G8B8A8_UNORM");
    }
    /* clear through the AHB image (acquired from FOREIGN, released to FOREIGN as ANGLE does), read via the original */
    VkClearColorValue cc = {.float32 = {0.2f, 0.4f, 0.6f, 1.0f}};
    memset(rb.ptr, 0, (size_t)W * H * 4);
    begin();
    barrier_q(wa, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_FOREIGN_EXT, qfi);
    p_CmdClearColorImage(cmd, wa, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cc, 1, &rng);
    barrier_q(wa, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, qfi, VK_QUEUE_FAMILY_FOREIGN_EXT);
    submit_wait("clear the AHB image to {0.2,0.4,0.6,1.0} (FOREIGN acquire/release barriers on the wrapper)");
    begin();
    barrier(fa->img, VK_IMAGE_LAYOUT_PREINITIALIZED, VK_IMAGE_LAYOUT_GENERAL);
    p_CmdCopyImageToBuffer(cmd, fa->img, VK_IMAGE_LAYOUT_GENERAL, rb.buf, 1, &region);
    submit_wait("copy the original (gralloc) image to a host buffer");
    uint32_t good = count_color(rb.ptr, 51, 102, 153, 255);
    check(good == W * H, "clear via AHB image, read via original: %u/%u texels (first {%u,%u,%u,%u}), modifier 0x%016llx",
          good, W * H, rb.ptr[0], rb.ptr[1], rb.ptr[2], rb.ptr[3], (unsigned long long)fa->modifier);
    /* reverse: pattern into the original, read via the AHB image (sync2 + copy2 when available) */
    struct hostbuf up = hostbuf((VkDeviceSize)W * H * 4);
    for (uint32_t y = 0; y < H; y++)
        for (uint32_t x = 0; x < W; x++) {
            uint8_t *p = up.ptr + 4 * (y * W + x);
            p[0] = (uint8_t)x; p[1] = (uint8_t)y; p[2] = (uint8_t)(x ^ (y * 7)); p[3] = (uint8_t)(x + 3 * y);
        }
    memset(rb.ptr, 0, (size_t)W * H * 4);
    begin();
    barrier(fa->img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    p_CmdCopyBufferToImage(cmd, up.buf, fa->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    barrier(fa->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    submit_wait("upload a per-texel pattern into the original image");
    begin();
    if (p_CmdPipelineBarrier2) {
        VkImageMemoryBarrier2 b2 = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT, .dstQueueFamilyIndex = qfi, .image = wa,
            .subresourceRange = rng};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                               .pImageMemoryBarriers = &b2};
        p_CmdPipelineBarrier2(cmd, &di);
        VkBufferImageCopy2 r2 = {.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
                                 .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {W, H, 1}};
        VkCopyImageToBufferInfo2 c2 = {.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2, .srcImage = wa,
                                       .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, .dstBuffer = rb.buf,
                                       .regionCount = 1, .pRegions = &r2};
        p_CmdCopyImageToBuffer2(cmd, &c2);
    } else {
        barrier_q(wa, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_QUEUE_FAMILY_FOREIGN_EXT, qfi);
        p_CmdCopyImageToBuffer(cmd, wa, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buf, 1, &region);
    }
    submit_wait(p_CmdPipelineBarrier2 ? "vkCmdPipelineBarrier2 + vkCmdCopyImageToBuffer2 from the AHB image"
                                      : "vkCmdPipelineBarrier + vkCmdCopyImageToBuffer from the AHB image");
    check(memcmp(rb.ptr, up.ptr, (size_t)W * H * 4) == 0, "pattern written via the original reads back identically "
          "via the AHB image (%u x %u texels, stride %u B)", W, H, fa->h.strides[0]);
    /* the AHB image as copy source and blit destination, and image barriers through events */
    {
        VkImageCreateInfo nci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                                 .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {W, H, 1}, .mipLevels = 1, .arrayLayers = 1,
                                 .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                 .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VkImage nimg;
        MUST(p_CreateImage(dev, &nci, NULL, &nimg));
        VkMemoryRequirements nmr;
        p_GetImageMemoryRequirements(dev, nimg, &nmr);
        VkMemoryAllocateInfo nmai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = nmr.size,
                                     .memoryTypeIndex = mem_type(nmr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        VkDeviceMemory nmem;
        MUST(p_AllocateMemory(dev, &nmai, NULL, &nmem));
        MUST(p_BindImageMemory(dev, nimg, nmem, 0));
        PFN_vkCmdCopyImage CmdCopyImage = dp("vkCmdCopyImage");
        PFN_vkCmdBlitImage CmdBlitImage = dp("vkCmdBlitImage");
        PFN_vkCmdSetEvent CmdSetEvent = dp("vkCmdSetEvent");
        PFN_vkCmdWaitEvents CmdWaitEvents = dp("vkCmdWaitEvents");
        PFN_vkCreateEvent CreateEvent = dp("vkCreateEvent");
        PFN_vkDestroyEvent DestroyEvent = dp("vkDestroyEvent");
        VkEventCreateInfo eci = {.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
        VkEvent ev;
        MUST(CreateEvent(dev, &eci, NULL, &ev));
        begin();
        barrier(nimg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageCopy ic = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .extent = {W, H, 1}};
        CmdCopyImage(cmd, wa, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, nimg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &ic);
        barrier(nimg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        /* wa: TRANSFER_SRC -> TRANSFER_DST through an event */
        CmdSetEvent(cmd, ev, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkImageMemoryBarrier eb = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                                   .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                   .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = wa, .subresourceRange = rng};
        CmdWaitEvents(cmd, 1, &ev, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, NULL, 0, NULL, 1, &eb);
        VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {W, H, 1}},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{W, 0, 0}, {0, H, 1}}};
        CmdBlitImage(cmd, nimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, wa, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl,
                     VK_FILTER_NEAREST);
        barrier_q(wa, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, qfi, VK_QUEUE_FAMILY_FOREIGN_EXT);
        submit_wait("vkCmdCopyImage(AHB image -> image), vkCmdWaitEvents(barrier on the AHB image), "
                    "vkCmdBlitImage(image -> AHB image, mirrored in x)");
        memset(rb.ptr, 0, (size_t)W * H * 4);
        begin();
        barrier(fa->img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
        p_CmdCopyImageToBuffer(cmd, fa->img, VK_IMAGE_LAYOUT_GENERAL, rb.buf, 1, &region);
        submit_wait("copy the original image to a host buffer");
        uint32_t mir = 0;
        for (uint32_t y = 0; y < H; y++)
            for (uint32_t x = 0; x < W; x++)
                mir += !memcmp(rb.ptr + 4 * (y * W + x), up.ptr + 4 * (y * W + (W - 1 - x)), 4);
        check(mir == W * H, "copy out of / blit into the AHB image: %u/%u texels mirrored as expected", mir, W * H);
        DestroyEvent(dev, ev, NULL);
        p_DestroyImage(dev, nimg, NULL);
        p_FreeMemory(dev, nmem, NULL);
        /* leave the original holding the plain pattern again for the Skia step */
        begin();
        barrier(fa->img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        p_CmdCopyBufferToImage(cmd, up.buf, fa->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        barrier(fa->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        submit_wait("re-upload the pattern into the original image");
    }
    for (int i = 0; i < 2; i++) p_DestroyImageView(dev, views[i], NULL);
    /* the wrapper destroyed before its memory */
    p_DestroyImage(dev, wa, NULL);
    p_FreeMemory(dev, ma, NULL);
    check(fa->refs == 1, "image destroyed, then memory freed: AHB reference returned (refs %d)", fa->refs);

    /* ---- 3. Skia's import: vkBindImageMemory2, memory freed before the image ---- */
    printf("-- Skia import sequence\n");
    {
        VkExternalFormatANDROID xf = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID};
        VkExternalMemoryImageCreateInfo emi = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, .pNext = &xf,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID};
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &emi, .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {W, H, 1}, .mipLevels = 1, .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     (linear ? 0 : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT),
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
        VkImage ws;
        MUST(p_CreateImage(dev, &ci, NULL, &ws));
        VkAndroidHardwareBufferPropertiesANDROID sp = {.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
        MUST(p_GetAndroidHardwareBufferPropertiesANDROID(dev, ahbA, &sp));
        uint32_t type = mem_type_want(sp.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (type == UINT32_MAX) {
            printf("    memoryTypeBits 0x%x has no DEVICE_LOCAL type: Skia (GAU:425-450) would abandon this buffer; "
                   "importing with ANGLE's choice instead\n", sp.memoryTypeBits);
            type = angle_mem_type(sp.memoryTypeBits);
        }
        VkImportAndroidHardwareBufferInfoANDROID imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
                                                         .buffer = ahbA};
        VkMemoryDedicatedAllocateInfo ded = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .pNext = &imp,
                                             .image = ws};
        VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &ded,
                                    .allocationSize = sp.allocationSize, .memoryTypeIndex = type};
        VkDeviceMemory ms;
        MUST(p_AllocateMemory(dev, &mai, NULL, &ms));
        VkBindImageMemoryInfo bi = {.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, .image = ws, .memory = ms};
        check(VKR(p_BindImageMemory2(dev, 1, &bi)) == VK_SUCCESS, "vkBindImageMemory2 (as Skia), memory type %u", type);
        memset(rb.ptr, 0, (size_t)W * H * 4);
        begin();
        /* acquire from EXTERNAL in the layout the producer released it in (GENERAL) */
        barrier_q(ws, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_QUEUE_FAMILY_EXTERNAL, qfi);
        p_CmdCopyImageToBuffer(cmd, ws, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buf, 1, &region);
        submit_wait("copy the Skia-imported image (EXTERNAL-queue acquire) to a host buffer");
        check(memcmp(rb.ptr, up.ptr, (size_t)W * H * 4) == 0, "Skia-style import sees the same pixels");
        p_FreeMemory(dev, ms, NULL); /* memory first, image after */
        p_DestroyImage(dev, ws, NULL);
        check(fa->refs == 1, "memory freed, then image destroyed: refs %d", fa->refs);
    }

    /* ---- 4. export ---- */
    printf("-- export\n");
    {
        VkImage we;
        MUST(angle_image(GPU_USAGE, &we));
        VkMemoryRequirements2 er = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
        VkImageMemoryRequirementsInfo2 eri = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, .image = we};
        p_GetImageMemoryRequirements2(dev, &eri, &er);
        VkExportMemoryAllocateInfo ex = {.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
                                         .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID};
        VkMemoryDedicatedAllocateInfo ded = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .pNext = &ex, .image = we};
        VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &ded,
                                    .allocationSize = er.memoryRequirements.size,
                                    .memoryTypeIndex = mem_type(er.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        int live0 = fake_live;
        VkDeviceMemory me;
        VkResult ar = VKR(p_AllocateMemory(dev, &mai, NULL, &me));
        check(ar == VK_SUCCESS && fake_live == live0 + 1,
              "vkAllocateMemory(VkExportMemoryAllocateInfo{AHB}, dedicated) allocated an AHB through "
              "AHardwareBuffer_allocate (live fake AHBs %d -> %d)", live0, fake_live);
        MUST(p_BindImageMemory(dev, we, me, 0));
        AHardwareBuffer *x = NULL;
        VkMemoryGetAndroidHardwareBufferInfoANDROID gi = {.sType = VK_STRUCTURE_TYPE_MEMORY_GET_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
                                                          .memory = me};
        VkResult gr = VKR(p_GetMemoryAndroidHardwareBufferANDROID(dev, &gi, &x));
        check(gr == VK_SUCCESS && x && F(x)->refs == 2,
              "vkGetMemoryAndroidHardwareBufferANDROID -> AHB %p with a new reference (refs %d), %ux%u format 0x%x "
              "usage 0x%llx", (void *)x, x ? F(x)->refs : -1, x ? F(x)->desc.width : 0, x ? F(x)->desc.height : 0,
              x ? F(x)->desc.format : 0, x ? (unsigned long long)F(x)->desc.usage : 0ull);
        VkClearColorValue c2 = {.float32 = {1.0f, 0.0f, 0.2f, 0.6f}};
        begin();
        barrier(we, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        p_CmdClearColorImage(cmd, we, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c2, 1, &rng);
        barrier_q(we, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, qfi, VK_QUEUE_FAMILY_FOREIGN_EXT);
        submit_wait("clear the exporting image to {1.0,0.0,0.2,0.6}");
        VkImage wi;
        VkDeviceMemory mi;
        MUST(angle_image(F(x)->desc.usage, &wi));
        check(VKR(import(x, wi, &mi, NULL)) == VK_SUCCESS, "import the exported AHB into a second image");
        MUST(p_BindImageMemory(dev, wi, mi, 0));
        memset(rb.ptr, 0, (size_t)W * H * 4);
        begin();
        barrier_q(wi, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_QUEUE_FAMILY_FOREIGN_EXT, qfi);
        p_CmdCopyImageToBuffer(cmd, wi, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buf, 1, &region);
        submit_wait("copy the importing image to a host buffer");
        good = count_color(rb.ptr, 255, 0, 51, 153);
        check(good == W * H, "export -> vkGetMemoryAndroidHardwareBufferANDROID -> import: %u/%u texels == {255,0,51,153}",
              good, W * H);
        p_DestroyImage(dev, wi, NULL);
        p_FreeMemory(dev, mi, NULL);
        p_DestroyImage(dev, we, NULL);
        p_FreeMemory(dev, me, NULL);
        fake_release(x);
        check(fake_live == live0, "exported AHB released by the memory and by us: live fake AHBs %d", fake_live);
    }

    /* ---- 5. BLOB AHB as a VkBuffer ---- */
    printf("-- BLOB buffer (GL_EXT_external_buffer)\n");
    {
        AHardwareBuffer_Desc bd = {.width = 65536, .height = 1, .layers = 1, .format = AHARDWAREBUFFER_FORMAT_BLOB,
                                   .usage = AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER};
        AHardwareBuffer *blob;
        MUST(fake_allocate(&bd, &blob) == 0 ? VK_SUCCESS : VK_ERROR_UNKNOWN);
        VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                               .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID};
        VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = 65536,
                                  .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VkBuffer bb;
        check(VKR(p_CreateBuffer(dev, &bci, NULL, &bb)) == VK_SUCCESS, "vkCreateBuffer(VkExternalMemoryBufferCreateInfo{AHB})");
        VkAndroidHardwareBufferPropertiesANDROID bp = {.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
        MUST(p_GetAndroidHardwareBufferPropertiesANDROID(dev, blob, &bp));
        VkImportAndroidHardwareBufferInfoANDROID imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
                                                         .buffer = blob};
        VkMemoryDedicatedAllocateInfo ded = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .pNext = &imp, .buffer = bb};
        VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &ded,
                                    .allocationSize = bp.allocationSize, .memoryTypeIndex = angle_mem_type(bp.memoryTypeBits)};
        VkDeviceMemory bm;
        check(VKR(p_AllocateMemory(dev, &mai, NULL, &bm)) == VK_SUCCESS, "BLOB import with VkMemoryDedicatedAllocateInfo{buffer} "
              "(%llu B, types 0x%x)", (unsigned long long)bp.allocationSize, bp.memoryTypeBits);
        MUST(p_BindBufferMemory(dev, bb, bm, 0));
        begin();
        p_CmdFillBuffer(cmd, bb, 0, 65536, 0xc0ffee11u);
        VkBufferMemoryBarrier bmb = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                     .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                     .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .buffer = F(blob)->buf, .size = VK_WHOLE_SIZE};
        p_CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 1, &bmb, 0, NULL);
        VkBufferCopy bc = {0, 0, 65536};
        p_CmdCopyBuffer(cmd, F(blob)->buf, rb.buf, 1, &bc);
        submit_wait("fill the imported buffer, read through the original buffer");
        uint32_t ok = 0;
        for (uint32_t i = 0; i < 65536 / 4; i++) ok += ((uint32_t *)rb.ptr)[i] == 0xc0ffee11u;
        check(ok == 65536 / 4, "BLOB AHB as VkBuffer shares memory: %u/%u words", ok, 65536 / 4);
        p_DestroyBuffer(dev, bb, NULL);
        p_FreeMemory(dev, bm, NULL);
        fake_release(blob);
    }

    /* ---- 6. negative cases ---- */
    printf("-- negative cases\n");
    {
        /* YUV AHB: NV12, two planes */
        struct fake_ahb *y = calloc(1, sizeof *y);
        *y = *fa;
        y->refs = 1;
        y->desc.format = AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420;
        y->h.format = DRM_FORMAT_NV12;
        y->h.num_planes = 2;
        y->img = VK_NULL_HANDLE;
        VkAndroidHardwareBufferFormatPropertiesANDROID yf = {.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID};
        VkAndroidHardwareBufferPropertiesANDROID yp = {.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID, .pNext = &yf};
        check(VKR(p_GetAndroidHardwareBufferPropertiesANDROID(dev, (AHardwareBuffer *)y, &yp)) == VK_ERROR_INVALID_EXTERNAL_HANDLE,
              "YUV (NV12, 2 planes) AHB -> VK_ERROR_INVALID_EXTERNAL_HANDLE (the spec's only failure code for the query)");
        y->desc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
        y->h.format = DRM_FORMAT_ABGR8888;
        y->h.num_planes = 1;
        y->desc.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN;
        check(VKR(p_GetAndroidHardwareBufferPropertiesANDROID(dev, (AHardwareBuffer *)y, &yp)) == VK_ERROR_INVALID_EXTERNAL_HANDLE,
              "AHB without GPU usage -> VK_ERROR_INVALID_EXTERNAL_HANDLE (VUID-...-buffer-01884)");
        free(y);

        VkImage wn;
        MUST(angle_image(usage, &wn));
        /* plain memory bound to a wrapper */
        VkMemoryAllocateInfo pm = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = 1 << 20,
                                   .memoryTypeIndex = mem_type(~0u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        VkDeviceMemory plain;
        MUST(p_AllocateMemory(dev, &pm, NULL, &plain));
        check(VKR(p_BindImageMemory(dev, wn, plain, 0)) == VK_ERROR_INVALID_EXTERNAL_HANDLE,
              "vkBindImageMemory(wrapper, non-AHB memory) -> VK_ERROR_INVALID_EXTERNAL_HANDLE");
        VkBindImageMemoryInfo bi = {.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, .image = wn, .memory = plain};
        check(VKR(p_BindImageMemory2(dev, 1, &bi)) == VK_ERROR_INVALID_EXTERNAL_HANDLE,
              "vkBindImageMemory2(wrapper, non-AHB memory) -> VK_ERROR_INVALID_EXTERNAL_HANDLE");
        p_FreeMemory(dev, plain, NULL);
        VkMemoryDedicatedAllocateInfo ded = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .image = wn};
        pm.pNext = &ded;
        check(VKR(p_AllocateMemory(dev, &pm, NULL, &plain)) == VK_ERROR_INVALID_EXTERNAL_HANDLE,
              "dedicated non-AHB allocation for a wrapper -> VK_ERROR_INVALID_EXTERNAL_HANDLE");
        /* allocationSize that is not the AHB's */
        VkImportAndroidHardwareBufferInfoANDROID imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
                                                         .buffer = ahbA};
        ded.pNext = &imp;
        pm.allocationSize = ap.allocationSize - 4096;
        pm.memoryTypeIndex = angle_mem_type(ap.memoryTypeBits);
        check(VKR(p_AllocateMemory(dev, &pm, NULL, &plain)) == VK_ERROR_INVALID_EXTERNAL_HANDLE,
              "AHB import with a wrong allocationSize -> VK_ERROR_INVALID_EXTERNAL_HANDLE (VUID-...-allocationSize-02383)");
        /* image-format AHB without a dedicated image */
        pm.pNext = &imp;
        pm.allocationSize = ap.allocationSize;
        check(VKR(p_AllocateMemory(dev, &pm, NULL, &plain)) == VK_ERROR_INVALID_EXTERNAL_HANDLE,
              "image-format AHB imported without a dedicated image -> VK_ERROR_INVALID_EXTERNAL_HANDLE (VUID-...-pNext-02384)");
        p_DestroyImage(dev, wn, NULL);
        check(fa->refs == 1, "failed imports left no AHB reference behind (refs %d)", fa->refs);
    }

    /* ---- fd / AHB accounting: 2 x 100 full cycles ---- */
    {
        quiet = true;
        int fdc[3];
        fdc[0] = count_fds();
        int ok = 1;
        for (int run = 0; run < 2; run++) {
            for (int i = 0; i < 100; i++) {
                AHardwareBuffer *b = new_ahb(AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM, usage);
                VkImage w;
                VkDeviceMemory m;
                VkImageView v;
                if (angle_image(usage, &w) != VK_SUCCESS || import(b, w, &m, NULL) != VK_SUCCESS ||
                    p_BindImageMemory(dev, w, m, 0) != VK_SUCCESS) { ok = 0; break; }
                VkImageViewCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = w,
                                             .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
                                             .subresourceRange = rng};
                if (p_CreateImageView(dev, &vci, NULL, &v) != VK_SUCCESS) { ok = 0; break; }
                p_DestroyImageView(dev, v, NULL);
                if (i & 1) { p_DestroyImage(dev, w, NULL); p_FreeMemory(dev, m, NULL); }
                else { p_FreeMemory(dev, m, NULL); p_DestroyImage(dev, w, NULL); }
                fake_release(b);
            }
            fdc[run + 1] = count_fds();
        }
        quiet = false;
        check(ok && fdc[2] == fdc[1] && fdc[1] == fdc[0], "2 x 100 cycles (allocate AHB, wrapper, import, bind, view, "
              "destroy in both orders, release): fd count %d -> %d -> %d, live fake AHBs %d", fdc[0], fdc[1], fdc[2],
              fake_live);
    }

    if (slot) p_DestroyPrivateDataSlot(dev, slot, NULL);
    hostbuf_free(&up);
    fake_release(ahbA);
    check(fake_live == 0 && count_dmabufs() == dmabufs_at_start,
          "teardown: live fake AHBs %d, dma-buf fds %d (start %d), fds %d (start %d)", fake_live, count_dmabufs(),
          dmabufs_at_start, count_fds(), fds_at_start);
    hostbuf_free(&rb);
    p_DestroyCommandPool(dev, pool, NULL);
    p_DestroyDevice(dev, NULL);
    DestroyInstance(inst, NULL);
    hal->common.close(&hal->common);
    printf("ahb-roundtrip: %d failure(s) -> %s\n", failures, failures ? "FAIL" : "OK");
    return failures ? 1 : 0;
}
