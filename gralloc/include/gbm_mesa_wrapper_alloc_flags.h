/*
 * COPY of the header our minigbm patch adds, do not edit here.
 * Source: minigbm/0006-gbm_mesa-CPU-cached-memory-for-buffers-the-CPU-reads-often.patch
 *         (new file gbm_mesa_driver/gbm_mesa_wrapper_alloc_flags.h, on minigbm 16844ba).
 * `make check-header` verifies that everything below this block is byte-identical
 * to the file in the patched tree (out/minigbm-src).
 */
/*
 * Optional extension of the gbm_mesa wrapper ABI (gbm_mesa_wrapper.h): allocate
 * with placement hints derived from the buffer's usage.
 *
 * struct alloc_args carries the layout (force_linear, needs_map_stride) and
 * scanout, but not how the CPU uses a LINEAR buffer. A buffer the CPU reads
 * often (BO_USE_SW_READ_OFTEN, e.g. hardware decoder output that a Codec2
 * client copies out on the CPU) is unusable in write-combined or uncached
 * memory: uncached CPU reads run at a small fraction of cached ones. minigbm's
 * amdgpu backend makes the same distinction (amdgpu.c amdgpu_create_bo: GTT
 * buffers are USWC unless BO_USE_SW_READ_OFTEN, and always USWC for scanout).
 * A wrapper that exports GBM_GET_ALLOC_FLAGS_OPS_SYMBOL gets these hints with
 * every allocation; a wrapper without the symbol keeps gbm_ops.alloc unchanged.
 */

#pragma once

#include <stdint.h>

struct alloc_args;

#define GBM_ALLOC_FLAGS_OPS_VERSION 1u
#define GBM_GET_ALLOC_FLAGS_OPS_SYMBOL "get_gbm_alloc_flags_ops"

/*
 * The CPU reads the buffer often and it is not scanned out: a LINEAR buffer
 * goes to CPU-cached memory.
 */
#define GBM_ALLOC_FLAG_CPU_READ_OFTEN (1u << 0)

struct gbm_alloc_flags_ops {
	/* GBM_ALLOC_FLAGS_OPS_VERSION of the wrapper; the glue refuses another */
	uint32_t version;

	/* as gbm_ops.alloc, with flags a subset of the GBM_ALLOC_FLAG_* bits */
	int (*alloc)(struct alloc_args *args, uint32_t flags);
};
