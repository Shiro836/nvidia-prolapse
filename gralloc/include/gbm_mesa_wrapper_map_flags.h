/*
 * COPY of the header our minigbm patch adds, do not edit here.
 * Source: minigbm/0002-gbm_mesa-pass-the-lock-access-flags-to-the-wrapper.patch
 *         (new file gbm_mesa_driver/gbm_mesa_wrapper_map_flags.h, on minigbm 16844ba).
 * `make check-header` verifies that everything below this block is byte-identical
 * to the file in the patched tree (out/minigbm-src).
 */
/*
 * Optional extension of the gbm_mesa wrapper ABI (gbm_mesa_wrapper.h): map and
 * unmap with the lock's CPU access flags.
 *
 * gbm_ops.map/unmap carry no access flags, so a wrapper that maps a tiled
 * buffer through a linear copy (a GPU copy in, a GPU copy back) must write the
 * copy back after every lock, including read-only locks, and a read-only lock
 * then writes the buffer. A wrapper that exports GBM_GET_MAP_FLAGS_OPS_SYMBOL
 * gets the flags minigbm derives from the lock usage (BO_MAP_READ/BO_MAP_WRITE,
 * cros_gralloc_convert_map_usage). A wrapper without the symbol keeps the
 * plain gbm_ops.map/unmap ABI unchanged.
 *
 * The unmap flags are those of the whole mapping: a nested lock of the same
 * buffer that adds access (e.g. a write lock while a read lock is held) widens
 * the mapping's flags, so unmap may carry more flags than map did.
 */

#pragma once

#include <stdint.h>

struct gbm_bo;

#define GBM_MAP_FLAGS_OPS_VERSION 1u
#define GBM_GET_MAP_FLAGS_OPS_SYMBOL "get_gbm_map_flags_ops"

/* equal to minigbm's BO_MAP_READ / BO_MAP_WRITE (drv.h) */
#define GBM_MAP_FLAG_READ (1u << 0)
#define GBM_MAP_FLAG_WRITE (1u << 1)

struct gbm_map_flags_ops {
	/* GBM_MAP_FLAGS_OPS_VERSION of the wrapper; the glue refuses another */
	uint32_t version;

	/* as gbm_ops.map, with map_flags != 0 and a subset of READ|WRITE */
	void (*map)(struct gbm_bo *bo, int w, int h, uint32_t map_flags, void **addr,
		    void **map_data);

	/* as gbm_ops.unmap; map_flags = the mapping's flags (>= the map's) */
	void (*unmap)(struct gbm_bo *bo, void *map_data, uint32_t map_flags);
};
