/*
 * import-lock-test: CPU lock of a GPU/codec-only NV12/P010 buffer in another process.
 *
 * Reproduces the Codec2 ByteBuffer-mode abort (gbm_mesa_internals.cpp
 * "assertion priv->gbm_bo != nullptr failed" from cros_gralloc_buffer::lock)
 * at minigbm's driver level, the layer cros_gralloc calls:
 *
 *   allocator process  drv_bo_create(NV12|P010, BO_USE_HW_VIDEO_DECODER)  (no SW usage, like
 *                      a hardware decoder's output block), writes a pattern through its own
 *                      mmap of the dma-buf (stands in for the copy engine), passes the fds and
 *                      the layout over a unix socket (SCM_RIGHTS)
 *   client process     drv_bo_import(same use_flags, as cros_gralloc_driver::retain does from
 *                      the handle) -> drv_bo_map(READ) -> checks both planes at the bo's
 *                      offsets/strides -> drv_bo_map(READ|WRITE) -> writes a second pattern
 *   allocator process  reads the second pattern back through its mmap, then locks the buffer
 *                      itself through drv_bo_map (the allocating process has no gbm_bo either)
 *
 * Built against minigbm's gbm_mesa core compiled for the host (Makefile: import-lock-test) and
 * run with the host libgbm_mesa_wrapper.so on LD_LIBRARY_PATH. The DRM node minigbm opens is the
 * NVIDIA render node, chosen by PCI vendor id 0x10de from sysfs and handed over through the
 * test's property_get stub (gralloc.gbm.device), never by a fixed /dev/dri path.
 *
 * usage: import-lock-test       exit 0 = all cases pass
 *        (the client is the same binary re-executed with --client FD)
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/dma-buf.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <drm_fourcc.h>

#include "drv.h"

/* ---- stubs for the Android libraries the gbm_mesa core links on the guest */
static char g_node[92]; /* PROPERTY_VALUE_MAX */

int property_get(const char *key, char *value, const char *default_value)
{
	if (!strcmp(key, "gralloc.gbm.device") && g_node[0]) {
		snprintf(value, 92, "%.80s", g_node);
		return (int)strlen(value);
	}
	if (default_value) {
		snprintf(value, 92, "%.80s", default_value);
		return (int)strlen(value);
	}
	value[0] = 0;
	return 0;
}

/* ---- NVIDIA render node by PCI vendor id */
static bool find_nvidia_render_node(void)
{
	DIR *d = opendir("/sys/class/drm");
	if (!d)
		return false;
	struct dirent *e;
	while ((e = readdir(d))) {
		if (strncmp(e->d_name, "renderD", 7))
			continue;
		char p[320], buf[32] = { 0 };
		snprintf(p, sizeof(p), "/sys/class/drm/%s/device/vendor", e->d_name);
		FILE *f = fopen(p, "r");
		if (!f)
			continue;
		if (fgets(buf, sizeof(buf), f) && strtoul(buf, NULL, 16) == 0x10de)
			snprintf(g_node, sizeof(g_node), "/dev/dri/%.64s", e->d_name);
		fclose(f);
		if (g_node[0])
			break;
	}
	closedir(d);
	return g_node[0] != 0;
}

/* ---- layout passed between the processes */
struct layout {
	uint32_t width, height, format, num_planes;
	uint32_t strides[DRV_MAX_PLANES], offsets[DRV_MAX_PLANES];
	uint64_t modifier, use_flags, total_size;
};

static int send_bo(int sock, const struct layout *l, const int *fds)
{
	char cbuf[CMSG_SPACE(sizeof(int) * DRV_MAX_PLANES)];
	memset(cbuf, 0, sizeof(cbuf));
	struct iovec iov = { .iov_base = (void *)l, .iov_len = sizeof(*l) };
	struct msghdr msg = { .msg_iov = &iov, .msg_iovlen = 1, .msg_control = cbuf,
			      .msg_controllen = CMSG_SPACE(sizeof(int) * l->num_planes) };
	struct cmsghdr *c = CMSG_FIRSTHDR(&msg);
	c->cmsg_level = SOL_SOCKET;
	c->cmsg_type = SCM_RIGHTS;
	c->cmsg_len = CMSG_LEN(sizeof(int) * l->num_planes);
	memcpy(CMSG_DATA(c), fds, sizeof(int) * l->num_planes);
	return sendmsg(sock, &msg, 0) == (ssize_t)sizeof(*l) ? 0 : -1;
}

static int recv_bo(int sock, struct layout *l, int *fds)
{
	char cbuf[CMSG_SPACE(sizeof(int) * DRV_MAX_PLANES)];
	struct iovec iov = { .iov_base = l, .iov_len = sizeof(*l) };
	struct msghdr msg = { .msg_iov = &iov, .msg_iovlen = 1, .msg_control = cbuf,
			      .msg_controllen = sizeof(cbuf) };
	if (recvmsg(sock, &msg, 0) != (ssize_t)sizeof(*l))
		return -1;
	struct cmsghdr *c = CMSG_FIRSTHDR(&msg);
	if (!c || c->cmsg_type != SCM_RIGHTS)
		return -1;
	memcpy(fds, CMSG_DATA(c), sizeof(int) * l->num_planes);
	return 0;
}

/* ---- pattern: every byte of every plane row, padding included */
static uint32_t plane_rows(const struct layout *l, uint32_t p)
{
	return p == 0 ? l->height : (l->height + 1) / 2;
}

static uint8_t pat(uint32_t p, uint32_t r, uint32_t c, uint32_t seed)
{
	return (uint8_t)(r * 31u + c * 7u + p * 101u + seed);
}

static void fill(uint8_t *base, const struct layout *l, uint32_t seed)
{
	for (uint32_t p = 0; p < l->num_planes; p++)
		for (uint32_t r = 0; r < plane_rows(l, p); r++)
			for (uint32_t c = 0; c < l->strides[p]; c++)
				base[l->offsets[p] + (size_t)r * l->strides[p] + c] = pat(p, r, c, seed);
}

static long check(const uint8_t *base, const struct layout *l, uint32_t seed)
{
	long bad = 0;
	for (uint32_t p = 0; p < l->num_planes; p++)
		for (uint32_t r = 0; r < plane_rows(l, p); r++)
			for (uint32_t c = 0; c < l->strides[p]; c++)
				if (base[l->offsets[p] + (size_t)r * l->strides[p] + c] != pat(p, r, c, seed))
					bad++;
	return bad;
}

static int dmabuf_sync(int fd, uint64_t flags)
{
	struct dma_buf_sync s = { .flags = flags };
	return ioctl(fd, DMA_BUF_IOCTL_SYNC, &s);
}

/* ---- client: import + lock, exactly like a mapper process */
static int client(int sock)
{
	struct layout l;
	int fds[DRV_MAX_PLANES];
	if (recv_bo(sock, &l, fds)) {
		fprintf(stderr, "client: recv failed\n");
		return 2;
	}
	int rfd = open(g_node, O_RDWR | O_CLOEXEC);
	struct driver *drv = drv_create(rfd);
	if (!drv) {
		fprintf(stderr, "client: drv_create failed\n");
		return 2;
	}
	struct drv_import_fd_data data = { 0 };
	for (uint32_t p = 0; p < l.num_planes; p++) {
		data.fds[p] = fds[p];
		data.strides[p] = l.strides[p];
		data.offsets[p] = l.offsets[p];
	}
	data.format_modifier = l.modifier;
	data.width = l.width;
	data.height = l.height;
	data.format = l.format;
	data.use_flags = l.use_flags; /* the handle's (allocation) usage: no SW bit */
	struct bo *bo = drv_bo_import(drv, &data);
	if (!bo) {
		fprintf(stderr, "client: drv_bo_import failed\n");
		return 2;
	}
	int rc = 0;
	struct rectangle rect = { 0, 0, l.width, l.height };
	struct mapping *m = NULL;
	uint8_t *addr = drv_bo_map(bo, &rect, BO_MAP_READ, &m, 0);
	if (addr == MAP_FAILED) {
		fprintf(stderr, "client: lock(READ) failed\n");
		return 1;
	}
	/* drv_bo_map returns plane 0's address; cros_gralloc_buffer::lock adds the plane offsets */
	uint8_t *base = addr - l.offsets[0];
	long bad = check(base, &l, 1);
	printf("  client lock(READ): %ld bad bytes of %" PRIu64 "\n", bad, l.total_size);
	if (bad)
		rc = 1;
	drv_bo_unmap(bo, m);

	m = NULL;
	addr = drv_bo_map(bo, &rect, BO_MAP_READ_WRITE, &m, 0);
	if (addr == MAP_FAILED) {
		fprintf(stderr, "client: lock(READ|WRITE) failed\n");
		return 1;
	}
	fill(addr - l.offsets[0], &l, 2);
	drv_bo_unmap(bo, m); /* gbm_mesa: DMA_BUF_SYNC_END(RW) */
	printf("  client lock(READ|WRITE): wrote pattern 2\n");

	drv_bo_destroy(bo);
	drv_destroy(drv);
	for (uint32_t p = 0; p < l.num_planes; p++)
		close(fds[p]);
	char ok = 1;
	if (write(sock, &ok, 1) != 1)
		rc = 2;
	return rc;
}

static int run_case(struct driver *drv, uint32_t format, const char *name, uint32_t w, uint32_t h)
{
	printf("== %s %ux%u, allocation usage HW_VIDEO_DECODER only\n", name, w, h);
	int sv[2];
	if (socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sv))
		return 2;
	const uint64_t use = BO_USE_HW_VIDEO_DECODER;
	struct bo *bo = drv_bo_create(drv, w, h, format, use);
	if (!bo) {
		fprintf(stderr, "drv_bo_create(%s) failed\n", name);
		return 1;
	}
	struct layout l = { .width = w, .height = h, .format = format, .use_flags = use };
	l.num_planes = (uint32_t)drv_bo_get_num_planes(bo);
	l.modifier = drv_bo_get_format_modifier(bo);
	l.total_size = drv_bo_get_total_size(bo);
	int fds[DRV_MAX_PLANES];
	for (uint32_t p = 0; p < l.num_planes; p++) {
		l.strides[p] = drv_bo_get_plane_stride(bo, p);
		l.offsets[p] = drv_bo_get_plane_offset(bo, p);
		fds[p] = drv_bo_get_plane_fd(bo, p);
	}
	printf("  layout: planes %u, strides %u/%u, offsets %u/%u, modifier 0x%" PRIx64
	       ", total %" PRIu64 "\n",
	       l.num_planes, l.strides[0], l.strides[1], l.offsets[0], l.offsets[1], l.modifier,
	       l.total_size);
	if (l.num_planes != 2 || l.modifier != DRM_FORMAT_MOD_LINEAR) {
		fprintf(stderr, "unexpected layout\n");
		return 1;
	}

	/* the "decoder": write pattern 1 through a plain mmap of the dma-buf */
	uint8_t *raw = mmap(NULL, l.total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fds[0], 0);
	if (raw == MAP_FAILED) {
		perror("mmap");
		return 1;
	}
	dmabuf_sync(fds[0], DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE);
	fill(raw, &l, 1);
	dmabuf_sync(fds[0], DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE);

	fflush(stdout);
	pid_t pid = fork();
	if (pid == 0) {
		/* a fresh process (exec, not just fork): nothing of the allocator's driver state */
		char fdstr[16];
		snprintf(fdstr, sizeof(fdstr), "%d", sv[1]);
		close(sv[0]);
		execl("/proc/self/exe", "import-lock-test", "--client", fdstr, (char *)NULL);
		_exit(127);
	}
	close(sv[1]);
	int rc = send_bo(sv[0], &l, fds) ? 2 : 0;
	char ok = 0;
	if (!rc && read(sv[0], &ok, 1) != 1)
		ok = 0;
	int st = 0;
	waitpid(pid, &st, 0);
	if (WIFSIGNALED(st)) {
		printf("  client process killed by signal %d (%s)\n", WTERMSIG(st),
		       strsignal(WTERMSIG(st)));
		rc = 1;
	} else if (WEXITSTATUS(st)) {
		printf("  client exit %d\n", WEXITSTATUS(st));
		rc = 1;
	}
	close(sv[0]);

	if (!rc) {
		dmabuf_sync(fds[0], DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ);
		long bad = check(raw, &l, 2);
		dmabuf_sync(fds[0], DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ);
		printf("  allocator reads client's pattern 2: %ld bad bytes\n", bad);
		if (bad)
			rc = 1;

		/* lock in the allocating process too (bo_create sets no gbm_bo either) */
		struct rectangle rect = { 0, 0, w, h };
		struct mapping *m = NULL;
		uint8_t *addr = drv_bo_map(bo, &rect, BO_MAP_READ, &m, 0);
		if (addr == MAP_FAILED) {
			printf("  allocator lock(READ) failed\n");
			rc = 1;
		} else {
			bad = check(addr - l.offsets[0], &l, 2);
			printf("  allocator lock(READ): %ld bad bytes\n", bad);
			if (bad)
				rc = 1;
			drv_bo_unmap(bo, m);
		}
	}
	munmap(raw, l.total_size);
	for (uint32_t p = 0; p < l.num_planes; p++)
		close(fds[p]);
	drv_bo_destroy(bo);
	printf("  %s\n", rc ? "FAIL" : "PASS");
	return rc;
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (!find_nvidia_render_node()) {
		fprintf(stderr, "no render node with PCI vendor 0x10de\n");
		return 2;
	}
	if (argc == 3 && !strcmp(argv[1], "--client"))
		return client(atoi(argv[2]));
	printf("NVIDIA render node (vendor 0x10de): %s\n", g_node);
	int rfd = open(g_node, O_RDWR | O_CLOEXEC);
	struct driver *drv = drv_create(rfd);
	if (!drv) {
		fprintf(stderr, "drv_create failed\n");
		return 2;
	}
	int fails = 0;
	fails += run_case(drv, DRM_FORMAT_NV12, "NV12", 854, 480) != 0;
	fails += run_case(drv, DRM_FORMAT_NV12, "NV12", 1920, 1080) != 0;
	fails += run_case(drv, DRM_FORMAT_P010, "P010", 1280, 720) != 0;
	fails += run_case(drv, DRM_FORMAT_P010, "P010", 3840, 2160) != 0;
	drv_destroy(drv);
	printf("import-lock-test: %s (%d failing case%s)\n", fails ? "FAIL" : "PASS", fails,
	       fails == 1 ? "" : "s");
	return fails ? 1 : 0;
}
