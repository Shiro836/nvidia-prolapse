/* astc-roundtrip: host test of the HAL's ASTC LDR emulation (src/astc.c) on the real GPU, which has no ASTC.
 *
 * Loads vulkan.nvidia.so like AOSP libvulkan and checks, by vendor ID 0x10de:
 *   - queries: textureCompressionASTC_LDR = 1; every 2D LDR ASTC format reports the mandatory optimal-tiling
 *     features (SAMPLED_IMAGE, BLIT_SRC, SAMPLED_IMAGE_FILTER_LINEAR, TRANSFER_SRC/DST) and nothing linear/buffer;
 *     vkGetPhysicalDeviceImageFormatProperties[2] accept 2D OPTIMAL sampled images and refuse LINEAR, 3D, STORAGE
 *     and external memory;
 *   - vkCreateDevice with textureCompressionASTC_LDR enabled succeeds (the HAL keeps the bit from the ICD);
 *   - pixels: per case (block sizes 4x4, 5x4, 6x5, 8x8, 12x12; UNORM and SRGB; NPOT sizes, mip chains, layers)
 *     blocks that Mesa's CPU decoder (src/mesa/main/texcompress_astc.cpp, test/astc-ref.cpp) accepts -- random
 *     valid blocks and void-extent (constant colour) blocks -- are uploaded with vkCmdCopyBufferToImage, read back
 *     raw with vkCmdCopyImageToBuffer (must be bit-exact) and sampled with texelFetch in a compute shader (must
 *     equal the CPU decode, +-1 per channel for UNORM: Mesa's CPU path interpolates UNORM16-replicated endpoints,
 *     the GPU decoder uses the ASTC decode_unorm8 rule; SRGB uses that rule in both and must be exact);
 *   - the app's compute state survives the decode: the sampling pipeline, its descriptor set and push constants
 *     are bound BEFORE the upload and the first dispatch after it does not rebind (markers + texels prove it);
 *   - copies: ASTC -> ASTC (vkCmdCopyImage), ASTC -> R32G32B32A32_UINT and back into a fresh ASTC image (size-
 *     compatible copies; the second one decodes), and vkCmdBlitImage from the ASTC image to RGBA8.
 * --api 1.0 runs everything on an API 1.0 instance/device (the HAL then enables VK_KHR_maintenance2 itself).
 * Exit 0 only when every check passed.
 *
 * usage: astc-roundtrip HAL.so [--api 1.0] */
#include <dlfcn.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include <hardware/hwvulkan.h>

static const uint32_t sample_spv[] = {
#include "astc_sample_spv.h"
};
int astc_ref_decode(const uint8_t *block, unsigned bw, unsigned bh, int srgb, uint8_t *rgba); /* astc-ref.cpp */

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
#define MUST(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { printf("FATAL: %s -> %d\n", #call, r_); exit(2); } } while (0)

static PFN_vkGetInstanceProcAddr gipa;
static PFN_vkGetDeviceProcAddr gdpa;
static VkInstance inst;
static VkPhysicalDevice pd;
static VkDevice dev;
static VkQueue queue;
static uint32_t api = VK_API_VERSION_1_3, qfi;
static VkPhysicalDeviceMemoryProperties memprops;

#define DEV_FNS(X) X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements) X(vkAllocateMemory) \
    X(vkFreeMemory) X(vkBindImageMemory) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) \
    X(vkBindBufferMemory) X(vkMapMemory) X(vkCreateImageView) X(vkDestroyImageView) X(vkCreateSampler) \
    X(vkDestroySampler) X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout) \
    X(vkDestroyPipelineLayout) X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreateComputePipelines) \
    X(vkDestroyPipeline) X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) X(vkAllocateDescriptorSets) \
    X(vkUpdateDescriptorSets) X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkQueueSubmit) X(vkQueueWaitIdle) X(vkGetDeviceQueue) \
    X(vkCmdPipelineBarrier) X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) X(vkCmdCopyImage) X(vkCmdBlitImage) \
    X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdDispatch) X(vkDestroyDevice)
#define DECL(n) static PFN_##n n;
DEV_FNS(DECL)
static PFN_vkBindImageMemory2 vkBindImageMemory2_;
static PFN_vkQueueBindSparse vkQueueBindSparse_;

static void *ip(const char *name) {
    void *p = (void *)gipa(inst, name);
    if (!p) { printf("FATAL: GIPA(instance, %s) = NULL\n", name); exit(2); }
    return p;
}

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & want) == want) return i;
    printf("FATAL: no memory type 0x%x with flags 0x%x\n", bits, want);
    exit(2);
}

struct buf { VkBuffer b; VkDeviceMemory m; unsigned char *p; VkDeviceSize size; };
static struct buf mkbuf(VkDeviceSize size, VkBufferUsageFlags usage) {
    struct buf b = {.size = size};
    VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size, usage, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    MUST(vkCreateBuffer(dev, &ci, NULL, &b.b));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, b.b, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size,
                               mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    MUST(vkAllocateMemory(dev, &ai, NULL, &b.m));
    MUST(vkBindBufferMemory(dev, b.b, b.m, 0));
    MUST(vkMapMemory(dev, b.m, 0, VK_WHOLE_SIZE, 0, (void **)&b.p));
    memset(b.p, 0, size);
    return b;
}
static void rmbuf(struct buf *b) {
    vkDestroyBuffer(dev, b->b, NULL);
    vkFreeMemory(dev, b->m, NULL);
}

/* device-local memory for an image; dedicated (through the HAL's ASTC rewrite) and vkBindImageMemory2 on >= 1.1;
 * a SPARSE_BINDING image is bound with vkQueueBindSparse in two opaque binds split at a sparse-block boundary
 * in the middle (the HAL splits them again over the RGBA8 image and the block images) */
static VkDeviceMemory image_memory(VkImage img, bool dedicated, bool sparse) {
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(dev, img, &mr);
    if (sparse) {
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size,
                                   mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        VkDeviceMemory m;
        MUST(vkAllocateMemory(dev, &ai, NULL, &m));
        VkDeviceSize half = mr.size / 2 / mr.alignment * mr.alignment;
        VkSparseMemoryBind b[2] = {{0, half, m, 0, 0}, {half, mr.size - half, m, half, 0}};
        VkSparseImageOpaqueMemoryBindInfo ob = {img, half ? 2 : 1, half ? b : &b[1]};
        VkBindSparseInfo bsi = {VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, NULL, 0, NULL, 0, NULL, 1, &ob, 0, NULL, 0, NULL};
        MUST(vkQueueBindSparse_(queue, 1, &bsi, VK_NULL_HANDLE));
        MUST(vkQueueWaitIdle(queue));
        return m;
    }
    VkMemoryDedicatedAllocateInfo ded = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, NULL, img, VK_NULL_HANDLE};
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, dedicated && api >= VK_API_VERSION_1_1 ? &ded : NULL,
                               mr.size, mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    VkDeviceMemory m;
    MUST(vkAllocateMemory(dev, &ai, NULL, &m));
    if (api >= VK_API_VERSION_1_1) {
        VkBindImageMemoryInfo bi = {VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, NULL, img, m, 0};
        MUST(vkBindImageMemory2_(dev, 1, &bi));
    } else {
        MUST(vkBindImageMemory(dev, img, m, 0));
    }
    return m;
}

static VkImage mkimage(VkFormat f, uint32_t w, uint32_t h, uint32_t mips, uint32_t layers, VkImageUsageFlags usage,
                       VkImageCreateFlags flags) {
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, flags, VK_IMAGE_TYPE_2D, f, {w, h, 1}, mips, layers,
                            VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, usage, VK_SHARING_MODE_EXCLUSIVE, 0, NULL,
                            VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage img;
    MUST(vkCreateImage(dev, &ci, NULL, &img));
    return img;
}

static void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags sa,
                    VkAccessFlags da, VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, sa, da, from, to, VK_QUEUE_FAMILY_IGNORED,
                              VK_QUEUE_FAMILY_IGNORED, img,
                              {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    vkCmdPipelineBarrier(cb, ss, ds, 0, 0, NULL, 0, NULL, 1, &b);
}
static void host_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier m = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                         VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &m, 0, NULL, 0, NULL);
}

static VkCommandPool cpool;
static VkCommandBuffer begin(void) {
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cpool,
                                      VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb;
    MUST(vkAllocateCommandBuffers(dev, &ai, &cb));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL,
                                   VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
    MUST(vkBeginCommandBuffer(cb, &bi));
    return cb;
}
static void submit(VkCommandBuffer cb) {
    MUST(vkEndCommandBuffer(cb));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL};
    MUST(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    MUST(vkQueueWaitIdle(queue));
}

/* ---- blocks ---- */

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) {
    rng ^= rng >> 12;
    rng ^= rng << 25;
    rng ^= rng >> 27;
    return rng * 0x2545F4914F6CDD1Dull;
}
/* LDR void-extent block (ASTC spec: block mode 0x1FC, LDR, reserved 11b, all-ones extents = constant colour) */
static void void_extent(uint8_t b[16], uint64_t rgba16) {
    const uint64_t lo = 0xFFFFFFFFFFFFFDFCull;
    memcpy(b, &lo, 8);
    memcpy(b + 8, &rgba16, 8);
}
static int is_error_colour(const uint8_t *rgba, unsigned n) {
    for (unsigned i = 0; i < n; i++)
        if (rgba[4 * i] != 255 || rgba[4 * i + 1] != 0 || rgba[4 * i + 2] != 255 || rgba[4 * i + 3] != 255) return 0;
    return 1;
}
static unsigned n_void, n_random;
/* a block the reference decodes without error, and its texels */
static void gen_block(uint8_t b[16], unsigned bw, unsigned bh, int srgb, uint8_t *texels) {
    if (rnd() % 4 == 0) {
        void_extent(b, rnd());
        n_void++;
    } else {
        for (int tries = 0; tries < 4096; tries++) {
            uint64_t x[2] = {rnd(), rnd()};
            /* void-extent with reserved bits 10-11 != 11b: Mesa's GLSL decoder (astc_decoder.glsl void_extent_color)
             * returns the error colour, Mesa's CPU decoder and dEQP's tcuAstcUtil.cpp decodeVoidExtentBlock() ignore the
             * bits; encoders write 11b. Not generated: the two references disagree. */
            if ((x[0] & 0x1ff) == 0x1fc && ((x[0] >> 10) & 3) != 3) continue;
            memcpy(b, x, 16);
            if (astc_ref_decode(b, bw, bh, srgb, texels) == 0 && !is_error_colour(texels, bw * bh)) {
                n_random++;
                return;
            }
        }
        void_extent(b, rnd());
        n_void++;
    }
    if (astc_ref_decode(b, bw, bh, srgb, texels) != 0) {
        printf("FATAL: reference rejects a void-extent block\n");
        exit(2);
    }
}

static const uint8_t dims[14][2] = {{4, 4}, {5, 4}, {5, 5}, {6, 5}, {6, 6}, {8, 5}, {8, 6}, {8, 8}, {10, 5}, {10, 6},
                                    {10, 8}, {10, 10}, {12, 10}, {12, 12}};
static uint32_t mipd(uint32_t d, uint32_t m) { return (d >> m) ? (d >> m) : 1; }
static uint32_t dup(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

struct tcase {
    const char *name;
    VkFormat fmt, view; /* view: the ASTC format the sampling view uses (UNORM: the shader reads raw bytes) */
    uint32_t w, h, mips, layers;
    int extra; /* run the copy/blit checks */
    VkImageCreateFlags sparse; /* SPARSE_BINDING[|SPARSE_ALIASED]: bound with vkQueueBindSparse */
};

/* the app's sampling pipeline */
static VkDescriptorSetLayout dsl;
static VkPipelineLayout pl;
static VkPipeline pipe;
static VkSampler sampler;
struct pc { int32_t mip, layer; uint32_t w, h, base, idx, marker; };

static void bind_set(VkDescriptorSet set, VkImageView view, VkBuffer out) {
    VkDescriptorImageInfo ii = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo bi = {out, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[2] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii, NULL, NULL},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi, NULL}};
    vkUpdateDescriptorSets(dev, 2, w, 0, NULL);
}

/* compare sampled texels (out buffer) of subresources [0, n) of the case against the expected decode */
static void compare(const char *what, const struct tcase *c, const uint8_t *const *expected, const struct buf *out,
                    uint32_t mips, uint32_t marker0, int tol) {
    const uint32_t *o = (const uint32_t *)out->p;
    unsigned bad = 0, mbad = 0, maxd = 0, n = 0, idx = 0, first_bad = 1;
    uint32_t base = 0;
    for (uint32_t m = 0; m < mips; m++)
        for (uint32_t l = 0; l < c->layers; l++, idx++) {
            const uint32_t mw = mipd(c->w, m), mh = mipd(c->h, m);
            if (o[idx] != marker0 + idx) {
                if (first_bad) printf("    first mismatch: marker[%u] = 0x%08x, expected 0x%08x\n", idx, o[idx], marker0 + idx);
                first_bad = 0;
                mbad++;
            }
            const uint8_t *e = expected[m * c->layers + l];
            for (uint32_t i = 0; i < mw * mh; i++, n++) {
                uint32_t g = o[64 + base + i];
                for (int k = 0; k < 4; k++) {
                    int dg = (int)((g >> (8 * k)) & 0xff), de = e[4 * i + k], d = dg > de ? dg - de : de - dg;
                    if ((unsigned)d > maxd) maxd = (unsigned)d;
                    if (d > tol) {
                        if (first_bad)
                            printf("    first mismatch: mip %u layer %u texel (%u,%u) channel %d: GPU %d, reference %d "
                                   "(GPU texel 0x%08x)\n", m, l, i % mw, i / mw, k, dg, de, g);
                        first_bad = 0;
                        bad++;
                    }
                }
            }
            base += mw * mh;
        }
    check(bad == 0 && mbad == 0, "%s %s: %u texels in %u subresource(s) vs the CPU reference: %u channel(s) off (max "
          "|diff| %u, tolerance %d); push-constant markers: %u wrong", c->name, what, n, idx, bad, maxd, tol, mbad);
}

static void run_case(const struct tcase *c, unsigned ci_idx) {
    const unsigned t = (c->fmt - VK_FORMAT_ASTC_4x4_UNORM_BLOCK) / 2;
    const unsigned bw = dims[t][0], bh = dims[t][1];
    const int srgb = (c->fmt - VK_FORMAT_ASTC_4x4_UNORM_BLOCK) & 1;
    printf("case %s: %ux%u, %u mip(s), %u layer(s), view format %d\n", c->name, c->w, c->h, c->mips, c->layers, c->view);
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const VkImageCreateFlags flags = (c->view != c->fmt ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0) | c->sparse;
    const bool sp = c->sparse != 0;
    VkImage img = mkimage(c->fmt, c->w, c->h, c->mips, c->layers, usage, flags);
    VkMemoryRequirements mr, mr_rgba;
    vkGetImageMemoryRequirements(dev, img, &mr);
    VkImage rgba = mkimage(VK_FORMAT_R8G8B8A8_UNORM, c->w, c->h, c->mips, c->layers, usage, 0);
    vkGetImageMemoryRequirements(dev, rgba, &mr_rgba);
    vkDestroyImage(dev, rgba, NULL);
    check(mr.size >= mr_rgba.size && mr.memoryTypeBits && mr.alignment,
          "%s memory requirements: %llu B (an RGBA8 image alone: %llu B), alignment %llu, types 0x%x", c->name,
          (unsigned long long)mr.size, (unsigned long long)mr_rgba.size, (unsigned long long)mr.alignment,
          mr.memoryTypeBits);
    VkDeviceMemory mem = image_memory(img, true, sp);

    /* blocks + expected texels per (mip, layer) */
    uint32_t nsub = c->mips * c->layers;
    uint8_t **expected = calloc(nsub, sizeof *expected);
    VkDeviceSize bytes = 0, texels = 0;
    for (uint32_t m = 0; m < c->mips; m++) {
        bytes += (VkDeviceSize)dup(mipd(c->w, m), bw) * dup(mipd(c->h, m), bh) * 16 * c->layers;
        texels += (VkDeviceSize)mipd(c->w, m) * mipd(c->h, m) * c->layers;
    }
    struct buf staging = mkbuf(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    struct buf raw = mkbuf(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    struct buf out = mkbuf(4 * (64 + texels), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkBufferImageCopy regions[16];
    VkDeviceSize off = 0;
    uint8_t blk_tex[12 * 12 * 4];
    for (uint32_t m = 0; m < c->mips; m++) {
        const uint32_t mw = mipd(c->w, m), mh = mipd(c->h, m), bx = dup(mw, bw), by = dup(mh, bh);
        regions[m] = (VkBufferImageCopy){off, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, c->layers}, {0, 0, 0}, {mw, mh, 1}};
        for (uint32_t l = 0; l < c->layers; l++) {
            uint8_t *e = expected[m * c->layers + l] = calloc(mw * mh, 4);
            for (uint32_t y = 0; y < by; y++)
                for (uint32_t x = 0; x < bx; x++) {
                    gen_block(staging.p + off, bw, bh, srgb, blk_tex);
                    off += 16;
                    for (uint32_t ty = 0; ty < bh; ty++)
                        for (uint32_t tx = 0; tx < bw; tx++) {
                            uint32_t px = x * bw + tx, py = y * bh + ty;
                            if (px < mw && py < mh) memcpy(e + 4 * (py * mw + px), blk_tex + 4 * (ty * bw + tx), 4);
                        }
                }
        }
    }

    VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, img, VK_IMAGE_VIEW_TYPE_2D_ARRAY, c->view,
                                 {0, 0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, c->mips, 0, c->layers}};
    VkImageView view;
    VkResult vr = vkCreateImageView(dev, &vci, NULL, &view);
    check(vr == VK_SUCCESS, "%s: vkCreateImageView(ASTC format %d) -> %d", c->name, c->view, vr);
    if (vr != VK_SUCCESS) exit(2);

    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 3, 2, ps};
    VkDescriptorPool dp;
    MUST(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
    VkDescriptorSetLayout dsls[3] = {dsl, dsl, dsl};
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 3, dsls};
    VkDescriptorSet sets[3];
    MUST(vkAllocateDescriptorSets(dev, &dsai, sets));
    bind_set(sets[0], view, out.b);

    /* the app binds its compute state, THEN uploads (the HAL decodes in between), then dispatches without rebinding */
    const uint32_t marker0 = 0xA57C0000u + (ci_idx << 8);
    VkCommandBuffer cb = begin();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &sets[0], 0, NULL);
    struct pc p0 = {0, 0, c->w, c->h, 0, 0, marker0};
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p0, &p0);
    barrier(cb, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    vkCmdCopyBufferToImage(cb, staging.b, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, c->mips, regions);
    barrier(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
    vkCmdCopyImageToBuffer(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, raw.b, c->mips, regions);
    barrier(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    uint32_t base = 0, idx = 0;
    for (uint32_t m = 0; m < c->mips; m++)
        for (uint32_t l = 0; l < c->layers; l++, idx++) {
            const uint32_t mw = mipd(c->w, m), mh = mipd(c->h, m);
            struct pc p = {(int32_t)m, (int32_t)l, mw, mh, base, idx, marker0 + idx};
            if (idx) vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p, &p);
            vkCmdDispatch(cb, dup(mw, 8), dup(mh, 8), 1);
            base += mw * mh;
        }
    host_barrier(cb);
    submit(cb);
    check(memcmp(raw.p, staging.p, bytes) == 0, "%s: %llu B of ASTC blocks read back bit-exact (vkCmdCopyImageToBuffer)",
          c->name, (unsigned long long)bytes);
    const int tol = srgb ? 0 : 1;
    compare("upload + sample (compute state bound before the upload)", c, (const uint8_t *const *)expected, &out, c->mips,
            marker0, tol);

    if (c->extra) {
        /* ASTC -> ASTC, ASTC -> R32G32B32A32_UINT -> ASTC (decodes), blit ASTC -> RGBA8 */
        const uint32_t bx0 = dup(c->w, bw), by0 = dup(c->h, bh);
        VkImage b2 = mkimage(c->fmt, c->w, c->h, c->mips, c->layers, usage, flags);
        VkImage u = mkimage(VK_FORMAT_R32G32B32A32_UINT, bx0, by0, 1, c->layers, usage, 0);
        VkImage c3 = mkimage(c->fmt, c->w, c->h, 1, c->layers, usage, flags);
        VkFormat rf = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
        VkImage r = mkimage(rf, c->w, c->h, 1, 1, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0);
        VkDeviceMemory m2 = image_memory(b2, false, sp), mu = image_memory(u, false, false),
                       m3 = image_memory(c3, false, sp), mrr = image_memory(r, false, false);
        VkImageView v2, v3;
        vci.image = b2;
        MUST(vkCreateImageView(dev, &vci, NULL, &v2));
        vci.image = c3;
        vci.subresourceRange.levelCount = 1;
        MUST(vkCreateImageView(dev, &vci, NULL, &v3));
        struct buf out2 = mkbuf(out.size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        struct buf out3 = mkbuf(out.size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        struct buf rb = mkbuf((VkDeviceSize)c->w * c->h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        bind_set(sets[1], v2, out2.b);
        bind_set(sets[2], v3, out3.b);
        cb = begin();
        const VkAccessFlags rw = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        const VkPipelineStageFlags tr = VK_PIPELINE_STAGE_TRANSFER_BIT;
        barrier(cb, img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, tr);
        VkImage dsts[4] = {b2, u, c3, r};
        for (int i = 0; i < 4; i++)
            barrier(cb, dsts[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, rw,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, tr);
        VkImageCopy ic[16];
        for (uint32_t m = 0; m < c->mips; m++)
            ic[m] = (VkImageCopy){{VK_IMAGE_ASPECT_COLOR_BIT, m, 0, c->layers}, {0, 0, 0},
                                  {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, c->layers}, {0, 0, 0},
                                  {mipd(c->w, m), mipd(c->h, m), 1}};
        vkCmdCopyImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, b2, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, c->mips, ic);
        /* compressed -> uncompressed: the extent is the source's, in texels */
        VkImageCopy cu = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, c->layers}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, c->layers},
                          {0, 0, 0}, {c->w, c->h, 1}};
        vkCmdCopyImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, u, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cu);
        barrier(cb, u, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rw, rw, tr, tr);
        /* uncompressed -> compressed: the extent is the source's (blocks) */
        VkImageCopy uc = cu;
        uc.extent = (VkExtent3D){bx0, by0, 1};
        vkCmdCopyImage(cb, u, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, c3, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &uc);
        VkImageBlit bl = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {(int32_t)c->w, (int32_t)c->h, 1}},
                          {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {(int32_t)c->w, (int32_t)c->h, 1}}};
        vkCmdBlitImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, r, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl,
                       VK_FILTER_NEAREST);
        barrier(cb, r, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rw, rw, tr, tr);
        VkBufferImageCopy rr = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {c->w, c->h, 1}};
        vkCmdCopyImageToBuffer(cb, r, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.b, 1, &rr);
        barrier(cb, b2, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, rw,
                VK_ACCESS_SHADER_READ_BIT, tr, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        barrier(cb, c3, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, rw,
                VK_ACCESS_SHADER_READ_BIT, tr, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        for (int s = 1; s <= 2; s++) {
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &sets[s], 0, NULL);
            base = idx = 0;
            for (uint32_t m = 0; m < (s == 1 ? c->mips : 1); m++)
                for (uint32_t l = 0; l < c->layers; l++, idx++) {
                    const uint32_t mw = mipd(c->w, m), mh = mipd(c->h, m);
                    struct pc p = {(int32_t)m, (int32_t)l, mw, mh, base, idx, marker0 + idx};
                    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p, &p);
                    vkCmdDispatch(cb, dup(mw, 8), dup(mh, 8), 1);
                    base += mw * mh;
                }
        }
        host_barrier(cb);
        submit(cb);
        compare("vkCmdCopyImage ASTC -> ASTC, sampled", c, (const uint8_t *const *)expected, &out2, c->mips, marker0, tol);
        compare("vkCmdCopyImage ASTC -> R32G32B32A32_UINT -> ASTC (decoded), sampled", c, (const uint8_t *const *)expected,
                &out3, 1, marker0, tol);
        unsigned bad = 0, maxd = 0;
        for (uint32_t i = 0; i < c->w * c->h * 4; i++) {
            int d = (int)rb.p[i] - (int)expected[0][i];
            d = d < 0 ? -d : d;
            if ((unsigned)d > maxd) maxd = (unsigned)d;
            bad += d > tol;
        }
        check(bad == 0, "%s vkCmdBlitImage ASTC -> %s (nearest): %u texels match (max |diff| %u)", c->name,
              srgb ? "R8G8B8A8_SRGB" : "R8G8B8A8_UNORM", c->w * c->h, maxd);
        vkDestroyImageView(dev, v2, NULL);
        vkDestroyImageView(dev, v3, NULL);
        VkImage imgs[4] = {b2, u, c3, r};
        VkDeviceMemory mems[4] = {m2, mu, m3, mrr};
        for (int i = 0; i < 4; i++) {
            vkDestroyImage(dev, imgs[i], NULL);
            vkFreeMemory(dev, mems[i], NULL);
        }
        rmbuf(&out2);
        rmbuf(&out3);
        rmbuf(&rb);
    }
    vkDestroyDescriptorPool(dev, dp, NULL);
    vkDestroyImageView(dev, view, NULL);
    vkDestroyImage(dev, img, NULL);
    vkFreeMemory(dev, mem, NULL);
    rmbuf(&staging);
    rmbuf(&raw);
    rmbuf(&out);
    for (uint32_t i = 0; i < nsub; i++) free(expected[i]);
    free(expected);
}

int main(int argc, char **argv) {
    const char *halpath = "out/host/vulkan.nvidia.so";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--api") && i + 1 < argc && !strcmp(argv[i + 1], "1.0")) {
            api = VK_API_VERSION_1_0;
            i++;
        } else {
            halpath = argv[i];
        }
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("astc-roundtrip: HAL %s, API %u.%u\n", halpath, VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api));
    void *so = dlopen(halpath, RTLD_NOW | RTLD_LOCAL);
    if (!so) { printf("FATAL: dlopen %s: %s\n", halpath, dlerror()); return 2; }
    hwvulkan_module_t *hmi = dlsym(so, HAL_MODULE_INFO_SYM_AS_STR);
    if (!hmi) { printf("FATAL: no HMI\n"); return 2; }
    hmi->common.dso = so;
    hwvulkan_device_t *hal = NULL;
    if (hmi->common.methods->open(&hmi->common, HWVULKAN_DEVICE_0, (hw_device_t **)&hal) || !hal) {
        printf("FATAL: open(vk0)\n");
        return 2;
    }
    gipa = hal->GetInstanceProcAddr;
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "astc-roundtrip", 1, NULL, 0, api};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &ai, 0, NULL, 0, NULL};
    MUST(hal->CreateInstance(&ici, NULL, &inst));
    gdpa = ip("vkGetDeviceProcAddr");
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices = ip("vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties = ip("vkGetPhysicalDeviceProperties");
    PFN_vkGetPhysicalDeviceFeatures GetPhysicalDeviceFeatures = ip("vkGetPhysicalDeviceFeatures");
    PFN_vkGetPhysicalDeviceFormatProperties GetPhysicalDeviceFormatProperties = ip("vkGetPhysicalDeviceFormatProperties");
    PFN_vkGetPhysicalDeviceImageFormatProperties GPDIFP = ip("vkGetPhysicalDeviceImageFormatProperties");
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties = ip("vkGetPhysicalDeviceMemoryProperties");
    PFN_vkCreateDevice CreateDevice = ip("vkCreateDevice");
    PFN_vkDestroyInstance DestroyInstance = ip("vkDestroyInstance");

    uint32_t npd = 0;
    MUST(EnumeratePhysicalDevices(inst, &npd, NULL));
    VkPhysicalDevice *pds = calloc(npd, sizeof *pds);
    MUST(EnumeratePhysicalDevices(inst, &npd, pds));
    VkPhysicalDeviceProperties props;
    for (uint32_t i = 0; i < npd && !pd; i++) {
        GetPhysicalDeviceProperties(pds[i], &props);
        if (props.vendorID == 0x10de) pd = pds[i]; /* by vendor ID, never by node path */
    }
    if (!pd) { printf("FATAL: no NVIDIA (0x10de) physical device\n"); return 2; }
    printf("  device: %s api %u.%u.%u\n", props.deviceName, VK_API_VERSION_MAJOR(props.apiVersion),
           VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion));
    GetPhysicalDeviceMemoryProperties(pd, &memprops);

    /* ---- queries ---- */
    VkPhysicalDeviceFeatures f;
    GetPhysicalDeviceFeatures(pd, &f);
    check(f.textureCompressionASTC_LDR == VK_TRUE, "vkGetPhysicalDeviceFeatures: textureCompressionASTC_LDR = %u",
          f.textureCompressionASTC_LDR);
    if (api >= VK_API_VERSION_1_1) {
        PFN_vkGetPhysicalDeviceFeatures2 GPDF2 = ip("vkGetPhysicalDeviceFeatures2");
        VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, NULL, {0}};
        GPDF2(pd, &f2);
        check(f2.features.textureCompressionASTC_LDR == VK_TRUE, "vkGetPhysicalDeviceFeatures2: textureCompressionASTC_LDR = %u",
              f2.features.textureCompressionASTC_LDR);
    }
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                      VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                      VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    unsigned fmt_bad = 0;
    VkFormatFeatureFlags seen = 0;
    for (VkFormat fm = VK_FORMAT_ASTC_4x4_UNORM_BLOCK; fm <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; fm++) {
        VkFormatProperties fp;
        GetPhysicalDeviceFormatProperties(pd, fm, &fp);
        seen |= fp.optimalTilingFeatures;
        fmt_bad += (fp.optimalTilingFeatures & need) != need || fp.linearTilingFeatures || fp.bufferFeatures;
        if (api >= VK_API_VERSION_1_1) {
            PFN_vkGetPhysicalDeviceFormatProperties2 GPDFP2 = ip("vkGetPhysicalDeviceFormatProperties2");
            VkFormatProperties3 f3 = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3, NULL, 0, 0, 0};
            VkFormatProperties2 fp2 = {VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, api >= VK_API_VERSION_1_3 ? &f3 : NULL, {0}};
            GPDFP2(pd, fm, &fp2);
            fmt_bad += fp2.formatProperties.optimalTilingFeatures != fp.optimalTilingFeatures;
            if (api >= VK_API_VERSION_1_3) fmt_bad += f3.optimalTilingFeatures != fp.optimalTilingFeatures || f3.linearTilingFeatures;
        }
    }
    check(fmt_bad == 0, "28 ASTC LDR formats: optimal features 0x%x include SAMPLED|BLIT_SRC|FILTER_LINEAR|TRANSFER_SRC|"
          "TRANSFER_DST, no linear/buffer features%s", seen, api >= VK_API_VERSION_1_1 ? ", FormatProperties2/3 agree" : "");
    VkImageFormatProperties ifp;
    const VkImageUsageFlags su = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkResult r_ok = GPDIFP(pd, VK_FORMAT_ASTC_4x4_UNORM_BLOCK, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, su,
                           VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, &ifp);
    check(r_ok == VK_SUCCESS && ifp.maxMipLevels >= 15 && ifp.maxArrayLayers >= 6 && ifp.sampleCounts == VK_SAMPLE_COUNT_1_BIT,
          "vkGetPhysicalDeviceImageFormatProperties(ASTC_4x4_UNORM 2D OPTIMAL SAMPLED|TRANSFER_DST, CUBE_COMPATIBLE) -> %d, "
          "max %ux%u, %u mips, %u layers, samples 0x%x", r_ok, ifp.maxExtent.width, ifp.maxExtent.height, ifp.maxMipLevels,
          ifp.maxArrayLayers, ifp.sampleCounts);
    VkResult r_lin = GPDIFP(pd, VK_FORMAT_ASTC_4x4_UNORM_BLOCK, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_LINEAR, su, 0, &ifp);
    VkResult r_3d = GPDIFP(pd, VK_FORMAT_ASTC_4x4_UNORM_BLOCK, VK_IMAGE_TYPE_3D, VK_IMAGE_TILING_OPTIMAL, su, 0, &ifp);
    VkResult r_st = GPDIFP(pd, VK_FORMAT_ASTC_4x4_UNORM_BLOCK, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                           VK_IMAGE_USAGE_STORAGE_BIT, 0, &ifp);
    check(r_lin == VK_ERROR_FORMAT_NOT_SUPPORTED && r_3d == VK_ERROR_FORMAT_NOT_SUPPORTED &&
              r_st == VK_ERROR_FORMAT_NOT_SUPPORTED,
          "vkGetPhysicalDeviceImageFormatProperties refuses LINEAR (%d), 3D (%d), STORAGE (%d)", r_lin, r_3d, r_st);
    if (api >= VK_API_VERSION_1_1) {
        PFN_vkGetPhysicalDeviceImageFormatProperties2 GPDIFP2 = ip("vkGetPhysicalDeviceImageFormatProperties2");
        VkPhysicalDeviceExternalImageFormatInfo ext = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO, NULL, 0};
        VkPhysicalDeviceImageFormatInfo2 q = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, &ext,
                                              VK_FORMAT_ASTC_8x8_SRGB_BLOCK, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, su,
                                              VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT};
        VkExternalImageFormatProperties ep = {VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES, NULL, {0, 0, 0}};
        VkImageFormatProperties2 p2 = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2, &ep, {{0, 0, 0}, 0, 0, 0, 0}};
        VkResult r1 = GPDIFP2(pd, &q, &p2);
        ext.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkResult r2 = GPDIFP2(pd, &q, &p2);
        check(r1 == VK_SUCCESS && r2 == VK_ERROR_FORMAT_NOT_SUPPORTED,
              "vkGetPhysicalDeviceImageFormatProperties2(ASTC_8x8_SRGB, MUTABLE): no handle type -> %d, OPAQUE_FD -> %d",
              r1, r2);
    }

    /* dEQP-VK.api.info.image_format_properties.2d.optimal.astc_*: every usage set the features allow
     * (SAMPLED|TRANSFER_SRC|TRANSFER_DST) x create flags getValidImageCreateFlags() offers (MUTABLE, CUBE, and with
     * SAMPLED SPARSE_BINDING / RESIDENCY / ALIASED per device features); isRequiredImageParameterCombination()
     * requires all but SPARSE_RESIDENCY (vktApiFeatureInfo.cpp) */
    {
        unsigned req = 0, refused = 0;
        VkFormat first_fmt = 0;
        VkImageUsageFlags first_u = 0;
        VkImageCreateFlags first_c = 0;
        for (VkFormat fm = VK_FORMAT_ASTC_4x4_UNORM_BLOCK; fm <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; fm++)
            for (VkImageUsageFlags u = 1; u <= 7; u++) {
                VkImageCreateFlags set = 0;
                if (u & VK_IMAGE_USAGE_SAMPLED_BIT) {
                    set |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
                    if (f.sparseBinding) set |= VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
                    if (f.sparseResidencyAliased) set |= VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_ALIASED_BIT;
                }
                for (VkImageCreateFlags c = 0; c <= set; c++) {
                    if ((c & ~set) || ((c & (VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT | VK_IMAGE_CREATE_SPARSE_ALIASED_BIT)) &&
                                       !(c & VK_IMAGE_CREATE_SPARSE_BINDING_BIT)))
                        continue;
                    if (c & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT) continue; /* optional for compressed formats */
                    req++;
                    if (GPDIFP(pd, fm, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, u, c, &ifp) != VK_SUCCESS && !refused++) {
                        first_fmt = fm;
                        first_u = u;
                        first_c = c;
                    }
                }
            }
        check(refused == 0, "dEQP image_format_properties required combinations (28 formats, 2D OPTIMAL, sparseBinding %u, "
              "sparseResidencyAliased %u): %u queried, %u refused (first: format %d usage 0x%x flags 0x%x)",
              f.sparseBinding, f.sparseResidencyAliased, req, refused, first_fmt, first_u, first_c);
    }

    /* ---- device: the app enables textureCompressionASTC_LDR ---- */
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GQFP = ip("vkGetPhysicalDeviceQueueFamilyProperties");
    uint32_t nq = 0;
    GQFP(pd, &nq, NULL);
    VkQueueFamilyProperties *qf = calloc(nq, sizeof *qf);
    GQFP(pd, &nq, qf);
    while (qfi < nq && (qf[qfi].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) !=
                           (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
        qfi++;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qfi, 1, &prio};
    VkPhysicalDeviceFeatures ef = {0};
    ef.textureCompressionASTC_LDR = VK_TRUE;
    ef.sparseBinding = f.sparseBinding;
    ef.sparseResidencyAliased = f.sparseResidencyAliased;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &qci, 0, NULL, 0, NULL, &ef};
    VkResult rd = CreateDevice(pd, &dci, NULL, &dev);
    check(rd == VK_SUCCESS, "vkCreateDevice with textureCompressionASTC_LDR enabled -> %d", rd);
    if (rd != VK_SUCCESS) return 2;
#define LOAD(n) n = (PFN_##n)gdpa(dev, #n); if (!n) { printf("FATAL: GDPA(%s) = NULL\n", #n); return 2; }
    DEV_FNS(LOAD)
#undef LOAD
    if (api >= VK_API_VERSION_1_1) vkBindImageMemory2_ = (PFN_vkBindImageMemory2)gdpa(dev, "vkBindImageMemory2");
    vkQueueBindSparse_ = (PFN_vkQueueBindSparse)gdpa(dev, "vkQueueBindSparse");
    vkGetDeviceQueue(dev, qfi, 0, &queue);
    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, qfi};
    MUST(vkCreateCommandPool(dev, &cpci, NULL, &cpool));

    VkSamplerCreateInfo sci;
    memset(&sci, 0, sizeof sci);
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
    sci.maxLod = 16.0f;
    MUST(vkCreateSampler(dev, &sci, NULL, &sampler));
    VkDescriptorSetLayoutBinding b[2] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo dslci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 2, b};
    MUST(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(struct pc)};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 1, &pcr};
    MUST(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof sample_spv, sample_spv};
    VkShaderModule sm;
    MUST(vkCreateShaderModule(dev, &smci, NULL, &sm));
    VkComputePipelineCreateInfo cpi = {
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL}, pl,
        VK_NULL_HANDLE, -1};
    MUST(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));

    /* ---- refusals: named errors, never an image the ICD cannot back ---- */
    VkImageCreateInfo lci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
                             {64, 64, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_LINEAR, VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage li = VK_NULL_HANDLE;
    VkResult rl = vkCreateImage(dev, &lci, NULL, &li);
    lci.tiling = VK_IMAGE_TILING_OPTIMAL;
    lci.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    VkResult rs = vkCreateImage(dev, &lci, NULL, &li);
    lci.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    lci.flags = VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
    VkResult rb = api >= VK_API_VERSION_1_1 ? vkCreateImage(dev, &lci, NULL, &li) : VK_ERROR_FORMAT_NOT_SUPPORTED;
    check(rl == VK_ERROR_FORMAT_NOT_SUPPORTED && rs == VK_ERROR_FORMAT_NOT_SUPPORTED && rb == VK_ERROR_FORMAT_NOT_SUPPORTED,
          "vkCreateImage(ASTC) refused with VK_ERROR_FORMAT_NOT_SUPPORTED (and an R8 error line): LINEAR %d, STORAGE %d, "
          "BLOCK_TEXEL_VIEW_COMPATIBLE %d", rl, rs, rb);

    /* ---- pixels ---- */
    const struct tcase cases[] = {
        {"ASTC_4x4_UNORM", VK_FORMAT_ASTC_4x4_UNORM_BLOCK, VK_FORMAT_ASTC_4x4_UNORM_BLOCK, 20, 12, 3, 2, 1, 0},
        {"ASTC_8x8_SRGB (UNORM view)", VK_FORMAT_ASTC_8x8_SRGB_BLOCK, VK_FORMAT_ASTC_8x8_UNORM_BLOCK, 40, 24, 2, 1, 1, 0},
        {"ASTC_12x12_UNORM", VK_FORMAT_ASTC_12x12_UNORM_BLOCK, VK_FORMAT_ASTC_12x12_UNORM_BLOCK, 30, 30, 1, 1, 0, 0},
        {"ASTC_6x5_UNORM", VK_FORMAT_ASTC_6x5_UNORM_BLOCK, VK_FORMAT_ASTC_6x5_UNORM_BLOCK, 64, 64, 7, 1, 0, 0},
        {"ASTC_5x4_SRGB (UNORM view)", VK_FORMAT_ASTC_5x4_SRGB_BLOCK, VK_FORMAT_ASTC_5x4_UNORM_BLOCK, 17, 9, 2, 3, 1, 0},
        {"ASTC_4x4_UNORM 300x260", VK_FORMAT_ASTC_4x4_UNORM_BLOCK, VK_FORMAT_ASTC_4x4_UNORM_BLOCK, 300, 260, 4, 2, 0, 0},
        {"ASTC_4x4_UNORM SPARSE_BINDING|SPARSE_ALIASED", VK_FORMAT_ASTC_4x4_UNORM_BLOCK, VK_FORMAT_ASTC_4x4_UNORM_BLOCK, 300,
         260, 4, 2, 1, VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_ALIASED_BIT},
        {"ASTC_8x8_SRGB SPARSE_BINDING (UNORM view)", VK_FORMAT_ASTC_8x8_SRGB_BLOCK, VK_FORMAT_ASTC_8x8_UNORM_BLOCK, 200, 120,
         3, 1, 0, VK_IMAGE_CREATE_SPARSE_BINDING_BIT},
    };
    const VkQueueFlags qflags = qf[qfi].queueFlags;
    for (unsigned i = 0; i < sizeof cases / sizeof *cases; i++) {
        if (cases[i].sparse && (!f.sparseBinding || !(qflags & VK_QUEUE_SPARSE_BINDING_BIT) ||
                                ((cases[i].sparse & VK_IMAGE_CREATE_SPARSE_ALIASED_BIT) && !f.sparseResidencyAliased))) {
            printf("case %s: skipped, the device has no sparse binding\n", cases[i].name);
            continue;
        }
        run_case(&cases[i], i);
    }
    printf("  blocks: %u random valid (accepted by the CPU reference), %u void-extent\n", n_random, n_void);

    vkDestroyPipeline(dev, pipe, NULL);
    vkDestroyShaderModule(dev, sm, NULL);
    vkDestroyPipelineLayout(dev, pl, NULL);
    vkDestroyDescriptorSetLayout(dev, dsl, NULL);
    vkDestroySampler(dev, sampler, NULL);
    vkDestroyCommandPool(dev, cpool, NULL);
    vkDestroyDevice(dev, NULL);
    DestroyInstance(inst, NULL);
    free(pds);
    free(qf);
    printf("astc-roundtrip: %s (%d failure(s))\n", failures ? "FAILED" : "all checks passed", failures);
    return failures ? 1 : 0;
}
