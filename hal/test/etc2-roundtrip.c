/* etc2-roundtrip: host test of the HAL's ETC2/EAC emulation (src/astc.c, src/etc2_decoder.comp) on the real GPU,
 * which has no ETC2.
 *
 * Loads vulkan.nvidia.so like AOSP libvulkan and checks, by vendor ID 0x10de:
 *   - queries: textureCompressionETC2 = 1; the 10 ETC2/EAC formats report the mandatory optimal-tiling features
 *     (SAMPLED_IMAGE, BLIT_SRC, SAMPLED_IMAGE_FILTER_LINEAR, TRANSFER_SRC/DST), nothing linear/buffer and no host
 *     image transfer; vkGetPhysicalDeviceImageFormatProperties[2] accept 2D OPTIMAL sampled images (every
 *     combination dEQP's image_format_properties requires) and refuse LINEAR, 3D, STORAGE, HOST_TRANSFER and
 *     external memory;
 *   - vkCreateDevice with textureCompressionETC2 enabled succeeds (the HAL keeps the bit from the ICD); vkCreateImage
 *     refuses LINEAR, STORAGE, 3D, BLOCK_TEXEL_VIEW_COMPATIBLE and an EAC UNORM image with an SNORM view format;
 *   - pixels: per case (all 10 formats; NPOT sizes that end in partial blocks, mip chains down to 1x1, layers,
 *     sparse-bound images) random blocks covering every ETC2 mode (individual, differential, T, H, planar, punchthrough
 *     transparent/opaque) and every EAC corner (multiplier 0, signed base -128) are uploaded with
 *     vkCmdCopyBufferToImage, read back raw with vkCmdCopyImageToBuffer (must be bit-exact) and sampled with
 *     texelFetch in a compute shader: every texel must equal Mesa's CPU decoder (src/mesa/main/texcompress_etc.c,
 *     test/etc2-ref.c) bit for bit (RGBA8 bytes; R11/RG11 as 16-bit UNORM/SNORM values);
 *   - the app's compute state survives the decode: the sampling pipeline, its descriptor set and push constants are
 *     bound BEFORE the upload and the first dispatch after it does not rebind (markers + texels prove it);
 *   - copies: ETC2 -> ETC2 (vkCmdCopyImage), ETC2 -> R32G32[B32A32]_UINT and back into a fresh image (size-compatible
 *     copies; the second one decodes), RGBA8 ETC2 blocks -> an RG11 EAC image (a size-compatible compressed copy:
 *     decoded as RG11), and vkCmdBlitImage from the ETC2/EAC image to its decoded format.
 * --api 1.0 runs everything on an API 1.0 instance/device (the HAL then enables VK_KHR_maintenance2 itself).
 * Exit 0 only when every check passed.
 *
 * usage: etc2-roundtrip HAL.so [--api 1.0] */
#include <dlfcn.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include <hardware/hwvulkan.h>

static const uint32_t sample_spv[] = {
#include "etc2_sample_spv.h"
};
int etc2_ref_decode(const uint8_t *block, int vkfmt, uint8_t *out); /* etc2-ref.c */

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

/* device-local memory for an image; dedicated (through the HAL's rewrite) and vkBindImageMemory2 on >= 1.1; a
 * SPARSE_BINDING image is bound with vkQueueBindSparse in two opaque binds split at a sparse-block boundary in the
 * middle (the HAL splits them again over the decoded image and the block images) */
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

/* ---- formats ---- */

#define ETC2_FIRST VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK
#define ETC2_LAST VK_FORMAT_EAC_R11G11_SNORM_BLOCK
static unsigned fidx(VkFormat f) { return (unsigned)(f - ETC2_FIRST); }
static unsigned block_bytes(VkFormat f) { unsigned i = fidx(f); return i == 4 || i == 5 || i >= 8 ? 16 : 8; }
static bool is_eac(VkFormat f) { return fidx(f) >= 6; }
static bool is_snorm(VkFormat f) { return f == VK_FORMAT_EAC_R11_SNORM_BLOCK || f == VK_FORMAT_EAC_R11G11_SNORM_BLOCK; }
static bool is_srgb(VkFormat f) { return !is_eac(f) && (fidx(f) & 1); }
static bool is_rg(VkFormat f) { return fidx(f) >= 8; }
/* the format the HAL decodes into (and the blit destination) */
static VkFormat decoded(VkFormat f) {
    static const VkFormat d[10] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R8G8B8A8_UNORM,
                                   VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB,
                                   VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16G16_UNORM,
                                   VK_FORMAT_R16G16_SNORM};
    return d[fidx(f)];
}
static unsigned decoded_bytes(VkFormat f) { return is_eac(f) && !is_rg(f) ? 2 : 4; }
static const char *const fname[10] = {"ETC2_R8G8B8_UNORM", "ETC2_R8G8B8_SRGB", "ETC2_R8G8B8A1_UNORM",
                                      "ETC2_R8G8B8A1_SRGB", "ETC2_R8G8B8A8_UNORM", "ETC2_R8G8B8A8_SRGB",
                                      "EAC_R11_UNORM", "EAC_R11_SNORM", "EAC_R11G11_UNORM", "EAC_R11G11_SNORM"};

/* ---- blocks ---- */

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) {
    rng ^= rng >> 12;
    rng ^= rng << 25;
    rng ^= rng >> 27;
    return rng * 0x2545F4914F6CDD1Dull;
}

/* the mode of an 8-byte colour block (KDFS "Format RGB ETC2"): 0 individual, 1 differential, 2 T, 3 H, 4 planar;
 * +5 for a punchthrough block with the opaque bit clear */
static unsigned n_mode[10], n_eac_mult0, n_eac_m128;
static unsigned colour_mode(const uint8_t *b, bool punch) {
    static const int d[8] = {0, 1, 2, 3, -4, -3, -2, -1};
    const bool diff = (b[3] & 2) != 0;
    unsigned m;
    if (!punch && !diff) m = 0;
    else if ((b[0] >> 3) + d[b[0] & 7] < 0 || (b[0] >> 3) + d[b[0] & 7] > 31) m = 2;
    else if ((b[1] >> 3) + d[b[1] & 7] < 0 || (b[1] >> 3) + d[b[1] & 7] > 31) m = 3;
    else if ((b[2] >> 3) + d[b[2] & 7] < 0 || (b[2] >> 3) + d[b[2] & 7] > 31) m = 4;
    else m = 1;
    return m + (punch && !diff ? 5 : 0);
}
/* a base-colour byte whose 5-bit value plus 3-bit delta overflows [0, 31] (overflow = 1) or stays inside (0) */
static uint8_t base_byte(int overflow) {
    static const int d[8] = {0, 1, 2, 3, -4, -3, -2, -1};
    for (;;) {
        uint8_t v = (uint8_t)rnd();
        int s = (v >> 3) + d[v & 7];
        if ((s < 0 || s > 31) == (overflow != 0)) return v;
    }
}
/* 8-byte colour block: random bits; of every 5 blocks, 3 are steered into T, H and planar mode (random bits rarely
 * give planar) by choosing which base-colour channel overflows, with the diff bit set for RGB ETC2 (for
 * punchthrough that bit is the opaque bit and stays random) */
static unsigned blk_no;
static void gen_colour(uint8_t *b, bool punch) {
    uint64_t x = rnd();
    memcpy(b, &x, 8);
    const unsigned steer = blk_no % 5;
    if (steer >= 1 && steer <= 3) {
        b[0] = base_byte(steer == 1);
        b[1] = base_byte(steer == 2);
        if (steer == 3) b[2] = base_byte(1);
        if (!punch) b[3] |= 2;
    }
    n_mode[colour_mode(b, punch)]++;
}
static void gen_eac(uint8_t *b) {
    uint64_t x = rnd();
    memcpy(b, &x, 8);
    if (blk_no % 7 == 3) b[1] &= 0x0f, n_eac_mult0++;
    if (blk_no % 11 == 5) b[0] = 0x80, n_eac_m128++;
}
static void gen_block(uint8_t *b, VkFormat f) {
    blk_no++;
    switch (fidx(f)) {
    case 0: case 1: gen_colour(b, false); break;
    case 2: case 3: gen_colour(b, true); break;
    case 4: case 5: gen_eac(b); gen_colour(b + 8, false); break;
    case 6: case 7: gen_eac(b); break;
    default: gen_eac(b); blk_no++; gen_eac(b + 8); break;
    }
}

/* the reference texels of one block as the sampling shader's words (see test/etc2-sample.comp) */
static void ref_words(const uint8_t *b, VkFormat f, uint32_t out[16]) {
    uint8_t t[16 * 4];
    if (etc2_ref_decode(b, (int)f, t)) { printf("FATAL: reference refuses VkFormat %d\n", f); exit(2); }
    for (unsigned i = 0; i < 16; i++) {
        if (decoded_bytes(f) == 2) out[i] = (uint32_t)t[2 * i] | (uint32_t)t[2 * i + 1] << 8;
        else memcpy(&out[i], t + 4 * i, 4);
    }
}

static uint32_t mipd(uint32_t d, uint32_t m) { return (d >> m) ? (d >> m) : 1; }
static uint32_t dup(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

struct tcase {
    VkFormat fmt;
    uint32_t w, h, mips, layers;
    int extra;                 /* run the copy/blit checks */
    VkImageCreateFlags sparse; /* SPARSE_BINDING[|SPARSE_ALIASED]: bound with vkQueueBindSparse */
};

/* the app's sampling pipeline */
static VkDescriptorSetLayout dsl;
static VkPipelineLayout pl;
static VkPipeline pipe;
static VkSampler sampler;
struct pc { int32_t mip, layer; uint32_t w, h, base, idx, marker, mode; };

static void bind_set(VkDescriptorSet set, VkImageView view, VkBuffer out) {
    VkDescriptorImageInfo ii = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo bi = {out, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[2] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii, NULL, NULL},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi, NULL}};
    vkUpdateDescriptorSets(dev, 2, w, 0, NULL);
}
static uint32_t sample_mode(VkFormat f) { return !is_eac(f) ? 0 : is_snorm(f) ? 2 : 1; }

static unsigned total_texels, total_bytes;
/* compare sampled words (out buffer) of subresources [0, mips) x layers against the expected decode */
static void compare(const char *what, const char *name, uint32_t w, uint32_t h, uint32_t layers,
                    uint32_t *const *expected, const struct buf *out, uint32_t mips, uint32_t marker0) {
    const uint32_t *o = (const uint32_t *)out->p;
    unsigned bad = 0, mbad = 0, n = 0, idx = 0, first_bad = 1;
    uint32_t base = 0;
    for (uint32_t m = 0; m < mips; m++)
        for (uint32_t l = 0; l < layers; l++, idx++) {
            const uint32_t mw = mipd(w, m), mh = mipd(h, m);
            if (o[idx] != marker0 + idx) {
                if (first_bad) printf("    first mismatch: marker[%u] = 0x%08x, expected 0x%08x\n", idx, o[idx], marker0 + idx);
                first_bad = 0;
                mbad++;
            }
            const uint32_t *e = expected[m * layers + l];
            for (uint32_t i = 0; i < mw * mh; i++, n++) {
                uint32_t g = o[64 + base + i];
                if (g != e[i]) {
                    if (first_bad)
                        printf("    first mismatch: mip %u layer %u texel (%u,%u): GPU 0x%08x, reference 0x%08x\n", m, l,
                               i % mw, i / mw, g, e[i]);
                    first_bad = 0;
                    bad++;
                }
            }
            base += mw * mh;
        }
    total_texels += n;
    check(bad == 0 && mbad == 0, "%s %s: %u texels in %u subresource(s) vs the CPU reference: %u differ (bit-exact "
          "required); push-constant markers: %u wrong", name, what, n, idx, bad, mbad);
}

/* records dispatches sampling subresources [0, mips) x layers of the bound set into out; base offsets in texels */
static void sample_all(VkCommandBuffer cb, const struct tcase *c, uint32_t mips, uint32_t marker0, bool push_first) {
    uint32_t base = 0, idx = 0;
    for (uint32_t m = 0; m < mips; m++)
        for (uint32_t l = 0; l < c->layers; l++, idx++) {
            const uint32_t mw = mipd(c->w, m), mh = mipd(c->h, m);
            struct pc p = {(int32_t)m, (int32_t)l, mw, mh, base, idx, marker0 + idx, sample_mode(c->fmt)};
            if (idx || push_first) vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p, &p);
            vkCmdDispatch(cb, dup(mw, 8), dup(mh, 8), 1);
            base += mw * mh;
        }
}

static void run_case(const struct tcase *c, unsigned ci_idx) {
    const char *name = fname[fidx(c->fmt)];
    const unsigned bb = block_bytes(c->fmt);
    /* SRGB images are sampled through a UNORM view (MUTABLE_FORMAT): the shader reads the stored bytes */
    const VkFormat view_fmt = is_srgb(c->fmt) ? c->fmt - 1 : c->fmt;
    printf("case %s%s: %ux%u, %u mip(s), %u layer(s)%s\n", name, is_srgb(c->fmt) ? " (UNORM view)" : "", c->w, c->h,
           c->mips, c->layers, c->sparse ? ", sparse binding" : "");
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const VkImageCreateFlags flags = (view_fmt != c->fmt ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0) | c->sparse;
    const bool sp = c->sparse != 0;
    VkImage img = mkimage(c->fmt, c->w, c->h, c->mips, c->layers, usage, flags);
    VkMemoryRequirements mr, mr_dec;
    vkGetImageMemoryRequirements(dev, img, &mr);
    VkImage di = mkimage(decoded(c->fmt), c->w, c->h, c->mips, c->layers, usage, 0);
    vkGetImageMemoryRequirements(dev, di, &mr_dec);
    vkDestroyImage(dev, di, NULL);
    check(mr.size >= mr_dec.size && mr.memoryTypeBits && mr.alignment,
          "%s memory requirements: %llu B (the decoded format alone: %llu B), alignment %llu, types 0x%x", name,
          (unsigned long long)mr.size, (unsigned long long)mr_dec.size, (unsigned long long)mr.alignment,
          mr.memoryTypeBits);
    VkDeviceMemory mem = image_memory(img, true, sp);

    /* blocks + expected words per (mip, layer) */
    const uint32_t nsub = c->mips * c->layers;
    uint32_t **expected = calloc(nsub, sizeof *expected);
    VkDeviceSize bytes = 0, texels = 0;
    for (uint32_t m = 0; m < c->mips; m++) {
        bytes += (VkDeviceSize)dup(mipd(c->w, m), 4) * dup(mipd(c->h, m), 4) * bb * c->layers;
        texels += (VkDeviceSize)mipd(c->w, m) * mipd(c->h, m) * c->layers;
    }
    struct buf staging = mkbuf(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    struct buf raw = mkbuf(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    struct buf out = mkbuf(4 * (64 + texels), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkBufferImageCopy regions[16];
    VkDeviceSize off = 0;
    uint32_t words[16];
    for (uint32_t m = 0; m < c->mips; m++) {
        const uint32_t mw = mipd(c->w, m), mh = mipd(c->h, m), bx = dup(mw, 4), by = dup(mh, 4);
        regions[m] = (VkBufferImageCopy){off, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, c->layers}, {0, 0, 0}, {mw, mh, 1}};
        for (uint32_t l = 0; l < c->layers; l++) {
            uint32_t *e = expected[m * c->layers + l] = calloc(mw * mh, 4);
            for (uint32_t y = 0; y < by; y++)
                for (uint32_t x = 0; x < bx; x++) {
                    gen_block(staging.p + off, c->fmt);
                    ref_words(staging.p + off, c->fmt, words);
                    off += bb;
                    for (uint32_t ty = 0; ty < 4; ty++)
                        for (uint32_t tx = 0; tx < 4; tx++) {
                            uint32_t px = x * 4 + tx, py = y * 4 + ty;
                            if (px < mw && py < mh) e[py * mw + px] = words[ty * 4 + tx];
                        }
                }
        }
    }
    total_bytes += (unsigned)bytes;

    VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, img, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                                 view_fmt, {0, 0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, c->mips, 0, c->layers}};
    VkImageView view;
    VkResult vr = vkCreateImageView(dev, &vci, NULL, &view);
    check(vr == VK_SUCCESS, "%s: vkCreateImageView(VkFormat %d) -> %d", name, view_fmt, vr);
    if (vr != VK_SUCCESS) exit(2);

    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 4, 2, ps};
    VkDescriptorPool dp;
    MUST(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
    VkDescriptorSetLayout dsls[4] = {dsl, dsl, dsl, dsl};
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 4, dsls};
    VkDescriptorSet sets[4];
    MUST(vkAllocateDescriptorSets(dev, &dsai, sets));
    bind_set(sets[0], view, out.b);

    /* the app binds its compute state, THEN uploads (the HAL decodes in between), then dispatches without rebinding */
    const uint32_t marker0 = 0xE7C20000u + (ci_idx << 8);
    VkCommandBuffer cb = begin();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &sets[0], 0, NULL);
    struct pc p0 = {0, 0, c->w, c->h, 0, 0, marker0, sample_mode(c->fmt)};
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
    sample_all(cb, c, c->mips, marker0, false);
    host_barrier(cb);
    submit(cb);
    check(memcmp(raw.p, staging.p, bytes) == 0, "%s: %llu B of blocks read back bit-exact (vkCmdCopyImageToBuffer)", name,
          (unsigned long long)bytes);
    compare("upload + sample (compute state bound before the upload)", name, c->w, c->h, c->layers, expected, &out,
            c->mips, marker0);

    if (c->extra) {
        /* ETC2 -> ETC2; ETC2 -> R32G32[B32A32]_UINT -> ETC2 (decodes); blit to the decoded format; for RGBA8 blocks
         * also -> an RG11 EAC image (size-compatible compressed formats: the blocks decode as RG11) */
        const uint32_t bx0 = dup(c->w, 4), by0 = dup(c->h, 4);
        const VkFormat uf = bb == 8 ? VK_FORMAT_R32G32_UINT : VK_FORMAT_R32G32B32A32_UINT;
        const bool cross = fidx(c->fmt) == 4;
        VkImage b2 = mkimage(c->fmt, c->w, c->h, c->mips, c->layers, usage, flags);
        VkImage u = mkimage(uf, bx0, by0, 1, c->layers, usage, 0);
        VkImage c3 = mkimage(c->fmt, c->w, c->h, 1, c->layers, usage, flags);
        VkImage r = mkimage(decoded(c->fmt), c->w, c->h, 1, 1, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0);
        VkImage x4 = cross ? mkimage(VK_FORMAT_EAC_R11G11_UNORM_BLOCK, c->w, c->h, 1, c->layers, usage, 0) : VK_NULL_HANDLE;
        VkDeviceMemory m2 = image_memory(b2, false, sp), mu = image_memory(u, false, false),
                       m3 = image_memory(c3, false, sp), mrr = image_memory(r, false, false),
                       m4 = cross ? image_memory(x4, false, false) : VK_NULL_HANDLE;
        VkImageView v2, v3, v4 = VK_NULL_HANDLE;
        vci.image = b2;
        MUST(vkCreateImageView(dev, &vci, NULL, &v2));
        vci.image = c3;
        vci.subresourceRange.levelCount = 1;
        MUST(vkCreateImageView(dev, &vci, NULL, &v3));
        if (cross) {
            VkImageViewCreateInfo xv = vci;
            xv.image = x4;
            xv.format = VK_FORMAT_EAC_R11G11_UNORM_BLOCK;
            MUST(vkCreateImageView(dev, &xv, NULL, &v4));
        }
        struct buf out2 = mkbuf(out.size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        struct buf out3 = mkbuf(out.size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        struct buf out4 = mkbuf(out.size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        struct buf rb = mkbuf((VkDeviceSize)c->w * c->h * decoded_bytes(c->fmt), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        bind_set(sets[1], v2, out2.b);
        bind_set(sets[2], v3, out3.b);
        if (cross) bind_set(sets[3], v4, out4.b);
        cb = begin();
        const VkAccessFlags rw = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        const VkPipelineStageFlags tr = VK_PIPELINE_STAGE_TRANSFER_BIT;
        barrier(cb, img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, tr);
        VkImage dsts[5] = {b2, u, c3, r, x4};
        for (int i = 0; i < (cross ? 5 : 4); i++)
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
        if (cross) /* compressed -> compressed of the same block size: texels of the source */
            vkCmdCopyImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, x4, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cu);
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
        VkImage sampled[3] = {b2, c3, x4};
        for (int i = 0; i < (cross ? 3 : 2); i++)
            barrier(cb, sampled[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, rw,
                    VK_ACCESS_SHADER_READ_BIT, tr, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        for (int s = 1; s <= (cross ? 3 : 2); s++) {
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &sets[s], 0, NULL);
            struct tcase cc = *c;
            if (s == 3) cc.fmt = VK_FORMAT_EAC_R11G11_UNORM_BLOCK;
            sample_all(cb, &cc, s == 1 ? c->mips : 1, marker0, true);
        }
        host_barrier(cb);
        submit(cb);
        compare("vkCmdCopyImage -> same format, sampled", name, c->w, c->h, c->layers, expected, &out2, c->mips, marker0);
        char what[128];
        snprintf(what, sizeof what, "vkCmdCopyImage -> %s -> back (decoded), sampled",
                 bb == 8 ? "R32G32_UINT" : "R32G32B32A32_UINT");
        compare(what, name, c->w, c->h, c->layers, expected, &out3, 1, marker0);
        if (cross) {
            /* the reference: the same blocks (mip 0) decoded as RG11 */
            uint32_t **ex4 = calloc(c->layers, sizeof *ex4);
            const uint8_t *blk = staging.p;
            for (uint32_t l = 0; l < c->layers; l++) {
                ex4[l] = calloc(c->w * c->h, 4);
                for (uint32_t y = 0; y < by0; y++)
                    for (uint32_t x = 0; x < bx0; x++, blk += 16) {
                        ref_words(blk, VK_FORMAT_EAC_R11G11_UNORM_BLOCK, words);
                        for (uint32_t t = 0; t < 16; t++) {
                            uint32_t px = x * 4 + t % 4, py = y * 4 + t / 4;
                            if (px < c->w && py < c->h) ex4[l][py * c->w + px] = words[t];
                        }
                    }
            }
            compare("vkCmdCopyImage -> EAC_R11G11_UNORM (decoded as RG11), sampled", name, c->w, c->h, c->layers, ex4,
                    &out4, 1, marker0);
            for (uint32_t l = 0; l < c->layers; l++) free(ex4[l]);
            free(ex4);
        }
        unsigned bad = 0;
        for (uint32_t i = 0; i < c->w * c->h; i++) {
            uint32_t g = 0;
            memcpy(&g, rb.p + (size_t)i * decoded_bytes(c->fmt), decoded_bytes(c->fmt));
            bad += g != expected[0][i];
        }
        check(bad == 0, "%s vkCmdBlitImage -> VkFormat %d (nearest): %u texels, %u differ", name, decoded(c->fmt),
              c->w * c->h, bad);
        vkDestroyImageView(dev, v2, NULL);
        vkDestroyImageView(dev, v3, NULL);
        if (v4) vkDestroyImageView(dev, v4, NULL);
        VkImage imgs[5] = {b2, u, c3, r, x4};
        VkDeviceMemory mems[5] = {m2, mu, m3, mrr, m4};
        for (int i = 0; i < 5; i++) {
            if (imgs[i]) vkDestroyImage(dev, imgs[i], NULL);
            if (mems[i]) vkFreeMemory(dev, mems[i], NULL);
        }
        rmbuf(&out2);
        rmbuf(&out3);
        rmbuf(&out4);
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
    printf("etc2-roundtrip: HAL %s, API %u.%u\n", halpath, VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api));
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
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "etc2-roundtrip", 1, NULL, 0, api};
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
    check(f.textureCompressionETC2 == VK_TRUE, "vkGetPhysicalDeviceFeatures: textureCompressionETC2 = %u",
          f.textureCompressionETC2);
    if (api >= VK_API_VERSION_1_1) {
        PFN_vkGetPhysicalDeviceFeatures2 GPDF2 = ip("vkGetPhysicalDeviceFeatures2");
        VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, NULL, {0}};
        GPDF2(pd, &f2);
        check(f2.features.textureCompressionETC2 == VK_TRUE, "vkGetPhysicalDeviceFeatures2: textureCompressionETC2 = %u",
              f2.features.textureCompressionETC2);
    }
    const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                      VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                      VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    unsigned fmt_bad = 0;
    VkFormatFeatureFlags seen = 0;
    for (VkFormat fm = ETC2_FIRST; fm <= ETC2_LAST; fm++) {
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
            if (api >= VK_API_VERSION_1_3)
                fmt_bad += f3.optimalTilingFeatures != fp.optimalTilingFeatures || f3.linearTilingFeatures ||
                           (f3.optimalTilingFeatures & VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT);
        }
    }
    check(fmt_bad == 0, "10 ETC2/EAC formats: optimal features 0x%x include SAMPLED|BLIT_SRC|FILTER_LINEAR|TRANSFER_SRC|"
          "TRANSFER_DST, no linear/buffer features%s", seen,
          api >= VK_API_VERSION_1_1 ? ", FormatProperties2/3 agree, no HOST_IMAGE_TRANSFER" : "");
    VkImageFormatProperties ifp;
    const VkImageUsageFlags su = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkResult r_ok = GPDIFP(pd, VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, su,
                           VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, &ifp);
    check(r_ok == VK_SUCCESS && ifp.maxMipLevels >= 15 && ifp.maxArrayLayers >= 6 && ifp.sampleCounts == VK_SAMPLE_COUNT_1_BIT,
          "vkGetPhysicalDeviceImageFormatProperties(ETC2_R8G8B8_UNORM 2D OPTIMAL SAMPLED|TRANSFER_DST, CUBE_COMPATIBLE) -> %d, "
          "max %ux%u, %u mips, %u layers, samples 0x%x", r_ok, ifp.maxExtent.width, ifp.maxExtent.height, ifp.maxMipLevels,
          ifp.maxArrayLayers, ifp.sampleCounts);
    VkResult r_lin = GPDIFP(pd, VK_FORMAT_EAC_R11_SNORM_BLOCK, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_LINEAR, su, 0, &ifp);
    VkResult r_3d = GPDIFP(pd, VK_FORMAT_EAC_R11_SNORM_BLOCK, VK_IMAGE_TYPE_3D, VK_IMAGE_TILING_OPTIMAL, su, 0, &ifp);
    VkResult r_st = GPDIFP(pd, VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                           VK_IMAGE_USAGE_STORAGE_BIT, 0, &ifp);
    VkResult r_ht = GPDIFP(pd, VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_HOST_TRANSFER_BIT, 0, &ifp);
    check(r_lin == VK_ERROR_FORMAT_NOT_SUPPORTED && r_3d == VK_ERROR_FORMAT_NOT_SUPPORTED &&
              r_st == VK_ERROR_FORMAT_NOT_SUPPORTED && r_ht == VK_ERROR_FORMAT_NOT_SUPPORTED,
          "vkGetPhysicalDeviceImageFormatProperties refuses LINEAR (%d), 3D (%d), STORAGE (%d), HOST_TRANSFER (%d)", r_lin,
          r_3d, r_st, r_ht);
    if (api >= VK_API_VERSION_1_1) {
        PFN_vkGetPhysicalDeviceImageFormatProperties2 GPDIFP2 = ip("vkGetPhysicalDeviceImageFormatProperties2");
        VkPhysicalDeviceExternalImageFormatInfo ext = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO, NULL, 0};
        VkPhysicalDeviceImageFormatInfo2 q = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, &ext,
                                              VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, su,
                                              VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT};
        VkExternalImageFormatProperties ep = {VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES, NULL, {0, 0, 0}};
        VkImageFormatProperties2 p2 = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2, &ep, {{0, 0, 0}, 0, 0, 0, 0}};
        VkResult r1 = GPDIFP2(pd, &q, &p2);
        ext.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkResult r2 = GPDIFP2(pd, &q, &p2);
        check(r1 == VK_SUCCESS && r2 == VK_ERROR_FORMAT_NOT_SUPPORTED,
              "vkGetPhysicalDeviceImageFormatProperties2(ETC2_R8G8B8A1_SRGB, MUTABLE): no handle type -> %d, OPAQUE_FD -> %d",
              r1, r2);
    }
    /* dEQP-VK.api.info.image_format_properties.2d.optimal.{etc2,eac}_*: every usage set the features allow
     * (SAMPLED|TRANSFER_SRC|TRANSFER_DST) x create flags getValidImageCreateFlags() offers (MUTABLE, CUBE, and with
     * SAMPLED SPARSE_BINDING / RESIDENCY / ALIASED per device features); isRequiredImageParameterCombination()
     * requires all but SPARSE_RESIDENCY (vktApiFeatureInfo.cpp) */
    {
        unsigned req = 0, refused = 0;
        VkFormat first_fmt = 0;
        VkImageUsageFlags first_u = 0;
        VkImageCreateFlags first_c = 0;
        for (VkFormat fm = ETC2_FIRST; fm <= ETC2_LAST; fm++)
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
        check(refused == 0, "dEQP image_format_properties required combinations (10 formats, 2D OPTIMAL, sparseBinding %u, "
              "sparseResidencyAliased %u): %u queried, %u refused (first: format %d usage 0x%x flags 0x%x)",
              f.sparseBinding, f.sparseResidencyAliased, req, refused, first_fmt, first_u, first_c);
    }

    /* ---- device: the app enables textureCompressionETC2 ---- */
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
    ef.textureCompressionETC2 = VK_TRUE;
    ef.sparseBinding = f.sparseBinding;
    ef.sparseResidencyAliased = f.sparseResidencyAliased;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &qci, 0, NULL, 0, NULL, &ef};
    VkResult rd = CreateDevice(pd, &dci, NULL, &dev);
    check(rd == VK_SUCCESS, "vkCreateDevice with textureCompressionETC2 enabled -> %d", rd);
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
    VkImageCreateInfo lci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
                             {64, 64, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_LINEAR, VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage li = VK_NULL_HANDLE;
    VkResult rl = vkCreateImage(dev, &lci, NULL, &li);
    lci.tiling = VK_IMAGE_TILING_OPTIMAL;
    lci.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    VkResult rs = vkCreateImage(dev, &lci, NULL, &li);
    lci.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    lci.imageType = VK_IMAGE_TYPE_3D;
    lci.extent.depth = 4;
    VkResult r3 = vkCreateImage(dev, &lci, NULL, &li);
    lci.imageType = VK_IMAGE_TYPE_2D;
    lci.extent.depth = 1;
    lci.flags = VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
    VkResult rb = api >= VK_API_VERSION_1_1 ? vkCreateImage(dev, &lci, NULL, &li) : VK_ERROR_FORMAT_NOT_SUPPORTED;
    /* an R11 UNORM image with an R11 SNORM view format: the decoded values differ, no reinterpretation */
    VkFormat vf[2] = {VK_FORMAT_EAC_R11_UNORM_BLOCK, VK_FORMAT_EAC_R11_SNORM_BLOCK};
    VkImageFormatListCreateInfo fl = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, NULL, 2, vf};
    lci.pNext = api >= VK_API_VERSION_1_1 ? &fl : NULL;
    lci.format = VK_FORMAT_EAC_R11_UNORM_BLOCK;
    lci.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    VkResult rv = api >= VK_API_VERSION_1_1 ? vkCreateImage(dev, &lci, NULL, &li) : VK_ERROR_FORMAT_NOT_SUPPORTED;
    check(rl == VK_ERROR_FORMAT_NOT_SUPPORTED && rs == VK_ERROR_FORMAT_NOT_SUPPORTED && r3 == VK_ERROR_FORMAT_NOT_SUPPORTED &&
              rb == VK_ERROR_FORMAT_NOT_SUPPORTED && rv == VK_ERROR_FORMAT_NOT_SUPPORTED,
          "vkCreateImage(ETC2/EAC) refused with VK_ERROR_FORMAT_NOT_SUPPORTED (and an error line): LINEAR %d, STORAGE %d, "
          "3D %d, BLOCK_TEXEL_VIEW_COMPATIBLE %d, R11 UNORM with an R11 SNORM view format %d", rl, rs, r3, rb, rv);

    /* ---- pixels ---- */
    const VkImageCreateFlags sb = VK_IMAGE_CREATE_SPARSE_BINDING_BIT, sa = VK_IMAGE_CREATE_SPARSE_ALIASED_BIT;
    const struct tcase cases[] = {
        {VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 20, 12, 3, 2, 1, 0},
        {VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK, 17, 9, 2, 3, 1, 0},
        {VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK, 33, 18, 3, 1, 1, 0},
        {VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK, 64, 64, 7, 1, 0, 0},
        {VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, 21, 13, 3, 2, 1, 0},
        {VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK, 300, 260, 4, 2, 1, 0},
        {VK_FORMAT_EAC_R11_UNORM_BLOCK, 19, 7, 3, 2, 1, 0},
        {VK_FORMAT_EAC_R11_SNORM_BLOCK, 40, 24, 2, 1, 1, 0},
        {VK_FORMAT_EAC_R11G11_UNORM_BLOCK, 5, 3, 2, 1, 1, 0},
        {VK_FORMAT_EAC_R11G11_SNORM_BLOCK, 128, 96, 5, 2, 1, 0},
        {VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 300, 260, 4, 2, 1, sb | sa},
        {VK_FORMAT_EAC_R11G11_SNORM_BLOCK, 200, 120, 3, 1, 0, sb},
    };
    const VkQueueFlags qflags = qf[qfi].queueFlags;
    for (unsigned i = 0; i < sizeof cases / sizeof *cases; i++) {
        if (cases[i].sparse && (!f.sparseBinding || !(qflags & VK_QUEUE_SPARSE_BINDING_BIT) ||
                                ((cases[i].sparse & sa) && !f.sparseResidencyAliased))) {
            printf("case %s: skipped, the device has no sparse binding\n", fname[fidx(cases[i].fmt)]);
            continue;
        }
        run_case(&cases[i], i);
    }
    printf("  blocks: colour modes individual %u, differential %u, T %u, H %u, planar %u; punchthrough transparent "
           "(opaque bit 0): differential %u, T %u, H %u, planar %u; EAC multiplier 0: %u, base 0x80: %u\n",
           n_mode[0], n_mode[1], n_mode[2], n_mode[3], n_mode[4], n_mode[6], n_mode[7], n_mode[8], n_mode[9], n_eac_mult0,
           n_eac_m128);
    check(n_mode[0] && n_mode[1] && n_mode[2] && n_mode[3] && n_mode[4] && n_mode[6] && n_mode[7] && n_mode[8] &&
              n_eac_mult0 && n_eac_m128,
          "every ETC2 mode and EAC corner case was exercised");
    printf("  totals: %u texels compared bit-exact, %u B of blocks round-tripped\n", total_texels, total_bytes);

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
    printf("etc2-roundtrip: %s (%d failure(s))\n", failures ? "FAILED" : "all checks passed", failures);
    return failures ? 1 : 0;
}
