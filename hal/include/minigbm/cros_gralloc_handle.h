/*
 * C TRANSLATION of minigbm's buffer handle layout.
 * Source: https://github.com/waydroid/android_external_minigbm
 *         commit 16844babfbd83d7c206ca22256257cbf26ca7de1 (the gralloc the guest runs,
 *         ro.hardware.gralloc=minigbm_gbm_mesa)
 * Paths:  cros_gralloc/cros_gralloc_handle.h  (struct cros_gralloc_handle, DRV_MAX_*)
 *         cros_gralloc/cros_gralloc_helpers.h:35 (cros_gralloc_magic = 0xABCDDCBA)
 *         cros_gralloc/cros_gralloc_driver.cc:273-274 (numInts formula)
 * Upstream is C++ ("struct cros_gralloc_handle : public native_handle_t"); here the
 * native_handle_t base is spelled out as the first member. Field order, types and the
 * packed attribute are unchanged; the static asserts pin the resulting layout.
 * Units: strides[] are minigbm plane strides in BYTES; pixel_stride is in pixels.
 */
#ifndef NVHAL_CROS_GRALLOC_HANDLE_H
#define NVHAL_CROS_GRALLOC_HANDLE_H

#include <stddef.h>
#include <stdint.h>
#include <cutils/native_handle.h>

#define DRV_MAX_PLANES 4
#define DRV_MAX_FDS (DRV_MAX_PLANES + 1)
#define CROS_GRALLOC_MAGIC 0xABCDDCBAu

struct cros_gralloc_handle {
	native_handle_t base;
	int32_t fds[DRV_MAX_FDS];
	uint32_t strides[DRV_MAX_PLANES];
	uint32_t offsets[DRV_MAX_PLANES];
	uint32_t sizes[DRV_MAX_PLANES];
	uint32_t id;
	uint32_t width;
	uint32_t height;
	uint32_t format; /* DRM format */
	uint32_t tiling;
	uint64_t format_modifier;
	uint64_t use_flags; /* Buffer creation flags */
	uint32_t magic;
	uint32_t pixel_stride;
	int32_t droid_format;
	int32_t usage; /* Android usage. */
	uint32_t num_planes;
	uint64_t reserved_region_size;
	uint64_t total_size; /* Total allocation size */
} __attribute__((packed));

/* numFds + numInts of every minigbm handle (cros_gralloc_driver.cc:273-274). */
#define CROS_GRALLOC_HANDLE_NUM_INTS_TOTAL \
	((sizeof(struct cros_gralloc_handle) - sizeof(native_handle_t)) / sizeof(int))

_Static_assert(sizeof(native_handle_t) == 12, "native_handle_t layout");
_Static_assert(offsetof(struct cros_gralloc_handle, fds) == 12, "fds must follow native_handle_t");
_Static_assert(offsetof(struct cros_gralloc_handle, format_modifier) == 100, "cros_gralloc_handle layout");
_Static_assert(offsetof(struct cros_gralloc_handle, magic) == 116, "cros_gralloc_handle layout");
_Static_assert(offsetof(struct cros_gralloc_handle, num_planes) == 132, "cros_gralloc_handle layout");
_Static_assert(sizeof(struct cros_gralloc_handle) == 152, "cros_gralloc_handle size");
_Static_assert(CROS_GRALLOC_HANDLE_NUM_INTS_TOTAL == 35, "cros_gralloc_handle int count");

/* NULL if `h` is a minigbm handle of exactly this layout, else the reason (for a loud error that
 * names the remedy: the guest must run minigbm gralloc). The same checks as nvhal_cros_handle()
 * in hal/src/anb.c; used by the EGL platform (nvidia-prolapse/egl/src/gralloc.c). */
static inline const char *cros_gralloc_handle_mismatch(const native_handle_t *h)
{
	if (h == NULL)
		return "the buffer handle is NULL";
	if (h->version != (int)sizeof(native_handle_t) || h->numFds < 1 || h->numFds > DRV_MAX_FDS ||
	    h->numInts < 0 || (size_t)h->numFds + (size_t)h->numInts != CROS_GRALLOC_HANDLE_NUM_INTS_TOTAL)
		return "not a minigbm cros_gralloc_handle (native_handle version/numFds/numInts differ)";
	if (((const struct cros_gralloc_handle *)h)->magic != CROS_GRALLOC_MAGIC)
		return "magic is not minigbm's 0xABCDDCBA";
	return NULL;
}

#endif
