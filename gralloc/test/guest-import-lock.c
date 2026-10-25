/*
 * guest-import-lock: in the Android guest, CPU-lock a GPU/codec-only NV12/P010 gralloc buffer
 * through a given libminigbm_gralloc_gbm_mesa.so (the live mapper path's cros_gralloc_driver),
 * without touching the vendor mounts.
 *
 *   AHardwareBuffer_allocate(YCbCr_420_888 | YCbCr_P010, usage VIDEO_DECODER only) -> the live
 *   allocator service allocates (another process); the pattern is written through a plain mmap
 *   of the dma-buf (stands in for the decoder's copy engine); then the libminigbm under test,
 *   dlopened by path, does what the IMapper@4 lock in the tombstone did:
 *   cros_gralloc_driver::retain(handle) -> lock(READ) -> check planes -> unlock -> release.
 *
 * usage (from /data/local/tmp, the linker's "unrestricted" section):
 *   guest-import-lock /data/local/tmp/LIB.so
 */
#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <linux/dma-buf.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <android/hardware_buffer.h>

/* cros_gralloc_handle (minigbm 16844ba cros_gralloc/cros_gralloc_handle.h), packed */
struct native_handle_hdr {
	int version, numFds, numInts;
};
struct cros_handle {
	struct native_handle_hdr base;
	int32_t fds[5];
	uint32_t strides[4], offsets[4], sizes[4];
	uint32_t id, width, height, format, tiling;
	uint64_t format_modifier, use_flags;
	uint32_t magic, pixel_stride;
	int32_t droid_format, usage;
	uint32_t num_planes;
	uint64_t reserved_region_size, total_size;
} __attribute__((packed));

const void *AHardwareBuffer_getNativeHandle(const AHardwareBuffer *buffer);

struct rectangle {
	uint32_t x, y, width, height;
};
typedef void *(*get_instance_t)(void);
typedef int32_t (*retain_t)(void *self, const void *handle);
typedef int32_t (*lock_t)(void *self, const void *handle, int32_t fence, bool close_fence,
			  const struct rectangle *rect, uint32_t map_flags, uint8_t *addr[4]);
typedef int32_t (*unlock_t)(void *self, const void *handle, int32_t *release_fence);

static uint8_t pat(uint32_t p, uint32_t r, uint32_t c)
{
	return (uint8_t)(r * 31u + c * 7u + p * 101u + 1u);
}

static uint32_t rows(const struct cros_handle *h, uint32_t p)
{
	return p == 0 ? h->height : (h->height + 1) / 2;
}

static int run(void *lib, uint32_t ahb_format, const char *name, uint32_t w, uint32_t h)
{
	get_instance_t get_instance = (get_instance_t)dlsym(lib, "_ZN19cros_gralloc_driver12get_instanceEv");
	retain_t retain = (retain_t)dlsym(lib, "_ZN19cros_gralloc_driver6retainEPK13native_handle");
	retain_t release = (retain_t)dlsym(lib, "_ZN19cros_gralloc_driver7releaseEPK13native_handle");
	lock_t lock = (lock_t)dlsym(lib, "_ZN19cros_gralloc_driver4lockEPK13native_handleibPK9rectanglejPPh");
	unlock_t unlock = (unlock_t)dlsym(lib, "_ZN19cros_gralloc_driver6unlockEPK13native_handlePi");
	if (!get_instance || !retain || !release || !lock || !unlock) {
		printf("missing cros_gralloc_driver symbols\n");
		return 2;
	}
	AHardwareBuffer_Desc d = { .width = w, .height = h, .layers = 1, .format = ahb_format,
				   .usage = 1ull << 22 /* BufferUsage::VIDEO_DECODER */ };
	AHardwareBuffer *ahb = NULL;
	int e = AHardwareBuffer_allocate(&d, &ahb);
	if (e) {
		printf("%s: AHardwareBuffer_allocate(usage VIDEO_DECODER) = %d\n", name, e);
		return 2;
	}
	const struct cros_handle *hd = AHardwareBuffer_getNativeHandle(ahb);
	printf("== %s %ux%u: handle use_flags 0x%" PRIx64 " usage 0x%x planes %u strides %u/%u "
	       "offsets %u/%u modifier 0x%" PRIx64 " total %" PRIu64 "\n",
	       name, w, h, hd->use_flags, hd->usage, hd->num_planes, hd->strides[0], hd->strides[1],
	       hd->offsets[0], hd->offsets[1], hd->format_modifier, hd->total_size);

	/* the "decoder": pattern through a plain mmap of the dma-buf */
	uint8_t *raw = mmap(NULL, hd->total_size, PROT_READ | PROT_WRITE, MAP_SHARED, hd->fds[0], 0);
	if (raw == MAP_FAILED) {
		printf("mmap: %s\n", strerror(errno));
		return 2;
	}
	struct dma_buf_sync s = { DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE };
	ioctl(hd->fds[0], DMA_BUF_IOCTL_SYNC, &s);
	for (uint32_t p = 0; p < hd->num_planes; p++)
		for (uint32_t r = 0; r < rows(hd, p); r++)
			for (uint32_t c = 0; c < hd->strides[p]; c++)
				raw[hd->offsets[p] + (size_t)r * hd->strides[p] + c] = pat(p, r, c);
	s.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
	ioctl(hd->fds[0], DMA_BUF_IOCTL_SYNC, &s);
	munmap(raw, hd->total_size);

	void *drv = get_instance();
	if (!drv || retain(drv, hd)) {
		printf("retain failed\n");
		return 1;
	}
	struct rectangle rect = { 0, 0, w, h };
	uint8_t *addr[4] = { 0 };
	fflush(stdout);
	int32_t rc = lock(drv, hd, -1, false, &rect, 1 /* BO_MAP_READ */, addr);
	if (rc) {
		printf("  lock(READ) = %d\n  FAIL\n", rc);
		release(drv, hd);
		return 1;
	}
	long bad = 0;
	for (uint32_t p = 0; p < hd->num_planes; p++)
		for (uint32_t r = 0; r < rows(hd, p); r++)
			for (uint32_t c = 0; c < hd->strides[p]; c++)
				bad += addr[p][(size_t)r * hd->strides[p] + c] != pat(p, r, c);
	int32_t fence = -1;
	unlock(drv, hd, &fence);
	release(drv, hd);
	AHardwareBuffer_release(ahb);
	printf("  lock(READ): planes at +%td/+%td, %ld bad bytes\n  %s\n", addr[0] - addr[0],
	       addr[1] - addr[0], bad, bad ? "FAIL" : "PASS");
	return bad ? 1 : 0;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (argc != 2) {
		fprintf(stderr, "usage: %s /path/libminigbm_gralloc_gbm_mesa.so\n", argv[0]);
		return 2;
	}
	void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
	if (!lib) {
		printf("dlopen %s: %s\n", argv[1], dlerror());
		return 2;
	}
	printf("libminigbm under test: %s\n", argv[1]);
	int fails = 0;
	fails += run(lib, AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420, "NV12 (YCbCr_420_888)", 854, 480) != 0;
	fails += run(lib, AHARDWAREBUFFER_FORMAT_YCbCr_P010, "P010", 1280, 720) != 0;
	printf("guest-import-lock: %s\n", fails ? "FAIL" : "PASS");
	return fails ? 1 : 0;
}
