/*
 * COPY of the header our minigbm patch adds, do not edit here.
 * Source: minigbm/0004-gbm_mesa-NV12-P010-decoder-output-layout.patch
 *         (new file gbm_mesa_driver/gbm_mesa_wrapper_layout.h, on minigbm 16844ba).
 * `make check-header` verifies that everything below this block is byte-identical
 * to the file in the patched tree (out/minigbm-src).
 */
/*
 * Optional extension of the gbm_mesa wrapper ABI (gbm_mesa_wrapper.h): place
 * the planes of a multi-planar LINEAR buffer where the wrapper's driver can
 * bind them.
 *
 * Formats the wrapper does not allocate itself (NV12, P010) are laid out by
 * minigbm and allocated as an R8 carrier of bo->meta.total_size. Importers
 * bind the buffer as a multi-planar image with the explicit plane offsets and
 * pitches of that layout. A driver can bind each plane like a plane of a
 * DISJOINT image, with that plane's own memory requirements: NVIDIA's Vulkan
 * driver (615) maps each plane separately, pitch * rows rounded up to 64 KiB,
 * so a plane offset off its alignment samples zeros, and the image needs more
 * memory than the packed layout. plane_layout() moves the offsets to the
 * driver's per-plane requirements and reports the buffer size that holds them.
 */

#pragma once

#include <stdint.h>

struct gbm_device;

#define GBM_LAYOUT_OPS_VERSION 1u
#define GBM_GET_LAYOUT_OPS_SYMBOL "get_gbm_layout_ops"

struct gbm_layout_ops {
	/* GBM_LAYOUT_OPS_VERSION of the wrapper; the glue refuses another */
	uint32_t version;

	/*
	 * A width x height image of drm_format, LINEAR, num_planes planes in one
	 * buffer with row pitches strides[] (bytes).
	 * In:  offsets[] = the smallest offset of each plane the caller's layout
	 *      allows (e.g. its packed layout).
	 * Out: offsets[] >= the input, each a multiple of the driver's alignment
	 *      for that plane and past the previous plane's required size;
	 *      *total_size = the bytes the buffer needs for the image with those
	 *      offsets (the last plane's required size, and at least the driver's
	 *      memory requirement of the whole image).
	 * 0 on success, else a negative errno (the wrapper logs why); the caller
	 * must not allocate then.
	 */
	int (*plane_layout)(struct gbm_device *gbm, uint32_t drm_format, uint32_t width,
			    uint32_t height, uint32_t num_planes, const uint32_t *strides,
			    uint32_t *offsets, uint64_t *total_size);
};
