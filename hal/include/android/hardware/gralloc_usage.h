/*
 * EXTRACT (values only) of the gralloc usage constants this HAL needs.
 * Each value is copied verbatim from:
 *   https://android.googlesource.com/platform/hardware/libhardware
 *   branch android13-release @ bbd07bd2e60752a665ebc9b0915241f3c1636516
 *   include/hardware/gralloc1.h  (gralloc1 producer/consumer usage, lines cited)
 *   include/hardware/gralloc.h   (gralloc0 usage, lines cited)
 * The full headers are not copied: gralloc1.h/gralloc.h pull in system/graphics.h and
 * the gralloc device API, none of which a Vulkan HAL needs. Upstream defines them as
 * enumerators; they are macros here so they keep their 64-bit type in C.
 */
#ifndef NVHAL_GRALLOC_USAGE_H
#define NVHAL_GRALLOC_USAGE_H

/* gralloc1.h:66,69 (gralloc1_consumer_usage_t) */
#define GRALLOC1_CONSUMER_USAGE_GPU_TEXTURE       (1ULL << 8)
#define GRALLOC1_CONSUMER_USAGE_HWCOMPOSER        (1ULL << 11)
/* gralloc1.h:189,198 (gralloc1_producer_usage_t) */
#define GRALLOC1_PRODUCER_USAGE_GPU_RENDER_TARGET (1ULL << 9)
#define GRALLOC1_PRODUCER_USAGE_PROTECTED         (1ULL << 14)

/* gralloc.h:88,90,94,107 (gralloc0 usage, used by vkGetSwapchainGrallocUsageANDROID) */
#define GRALLOC_USAGE_HW_TEXTURE   0x00000100U
#define GRALLOC_USAGE_HW_RENDER    0x00000200U
#define GRALLOC_USAGE_HW_COMPOSER  0x00000800U
#define GRALLOC_USAGE_PROTECTED    0x00004000U

#endif
