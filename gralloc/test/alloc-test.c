/*
 * Host test for libgbm_mesa_wrapper.so (NVIDIA native gralloc backend).
 *
 * Drives the library exactly through the minigbm ABI (get_gbm_ops()) and
 * verifies every buffer on the GPU through a *separate* Vulkan device
 * created via the system Vulkan loader (libvulkan -> the loader's NVIDIA ICD
 * manifest), importing the dma-bufs with VkImageDrmFormatModifierExplicitCreateInfoEXT
 * from the stride/modifier the library reported.
 *
 *   usage: alloc-test [path/to/libgbm_mesa_wrapper.so]
 *   run with DISPLAY= WAYLAND_DISPLAY= (make test does)
 *
 * Exit status 0 only when every check passed.
 */

#define _GNU_SOURCE

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <drm_fourcc.h>
#include <vulkan/vulkan.h>

#include "gbm_mesa_wrapper.h"
#include "gbm_mesa_wrapper_alloc_flags.h"

static int failures;
static char gpu_pci[32]; /* PCI address of the GPU under test, from sysfs */

#define PASS(...)                                                                                  \
	do {                                                                                       \
		printf("  PASS ");                                                                 \
		printf(__VA_ARGS__);                                                               \
		printf("\n");                                                                      \
	} while (0)
#define FAIL(...)                                                                                  \
	do {                                                                                       \
		printf("  FAIL ");                                                                 \
		printf(__VA_ARGS__);                                                               \
		printf("\n");                                                                      \
		failures++;                                                                        \
	} while (0)
#define EXPECT(cond, ...)                                                                          \
	do {                                                                                       \
		if (cond)                                                                          \
			PASS(__VA_ARGS__);                                                         \
		else                                                                               \
			FAIL(__VA_ARGS__);                                                         \
	} while (0)

#define VKCHECK(expr)                                                                              \
	do {                                                                                       \
		VkResult r_ = (expr);                                                              \
		if (r_ != VK_SUCCESS) {                                                            \
			printf("  FATAL %s = %d (%s:%d)\n", #expr, r_, __FILE__, __LINE__);        \
			exit(2);                                                                   \
		}                                                                                  \
	} while (0)

/* ------------------------------------------------------------------------ */
/* NVIDIA render node by sysfs PCI vendor (never by node name)               */

static int open_nvidia_render_node(dev_t *rdev, char *name, size_t name_len)
{
	glob_t g;
	if (glob("/dev/dri/renderD*", 0, NULL, &g))
		return -1;
	int fd = -1;
	for (size_t i = 0; i < g.gl_pathc && fd < 0; i++) {
		const char *base = strrchr(g.gl_pathv[i], '/') + 1;
		char path[PATH_MAX];
		snprintf(path, sizeof(path), "/sys/class/drm/%s/device/vendor", base);
		FILE *f = fopen(path, "r");
		unsigned vendor = 0;
		if (!f)
			continue;
		if (fscanf(f, "%x", &vendor) != 1)
			vendor = 0;
		fclose(f);
		if (vendor != 0x10de)
			continue;
		fd = open(g.gl_pathv[i], O_RDWR | O_CLOEXEC);
		snprintf(path, sizeof(path), "/sys/class/drm/%s/device", base);
		char real[PATH_MAX];
		if (realpath(path, real))
			snprintf(gpu_pci, sizeof(gpu_pci), "%s", strrchr(real, '/') + 1);
		struct stat st;
		if (fd >= 0 && fstat(fd, &st) == 0)
			*rdev = st.st_rdev;
		snprintf(name, name_len, "%s", g.gl_pathv[i]);
	}
	globfree(&g);
	return fd;
}

/* ------------------------------------------------------------------------ */
/* process measurements                                                      */

static int count_fds(void)
{
	DIR *d = opendir("/proc/self/fd");
	int n = 0;
	for (struct dirent *e; (e = readdir(d));)
		if (e->d_name[0] != '.')
			n++;
	closedir(d);
	return n - 1; /* the DIR's own fd */
}

static long vm_rss_kb(void)
{
	FILE *f = fopen("/proc/self/status", "r");
	char line[256];
	long kb = -1;
	while (fgets(line, sizeof(line), f))
		if (sscanf(line, "VmRSS: %ld kB", &kb) == 1)
			break;
	fclose(f);
	return kb;
}

/*
 * Device memory in use on the GPU under test, MiB (-1 on error). This is the
 * whole-GPU figure, matched by PCI address: nvidia-smi's per-process
 * accounting (--query-compute-apps) does not attribute memory that only an
 * exported dma-buf keeps alive (measured: 20 live 1080p buffers, +0 MiB), so
 * it cannot see a leak of exactly the kind this test looks for.
 */
static long gpu_mem_mib(void)
{
	FILE *p = popen("nvidia-smi --query-gpu=pci.bus_id,memory.used "
			"--format=csv,noheader,nounits 2>/dev/null",
			"r");
	if (!p)
		return -1;
	char bus[64];
	long mib, found = -1;
	while (fscanf(p, " %63[^,], %ld", bus, &mib) == 2) {
		/* nvidia-smi: 00000000:01:00.0, sysfs: 0000:01:00.0 */
		const char *a = strchr(bus, ':'), *b = strchr(gpu_pci, ':');
		if (a && b && !strcasecmp(a, b))
			found = mib;
	}
	pclose(p);
	return found;
}

/* system-wide MemAvailable, MiB: CPU (LINEAR) buffers live in system memory,
 * which neither VmRSS (unmapped) nor the GPU's memory.used shows */
static long mem_available_mib(void)
{
	FILE *f = fopen("/proc/meminfo", "r");
	char line[256];
	long kb = -1;
	while (fgets(line, sizeof(line), f))
		if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1)
			break;
	fclose(f);
	return kb < 0 ? -1 : kb / 1024;
}

static bool is_dmabuf_fd(int fd, char *link, size_t link_len)
{
	char path[64];
	snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
	ssize_t n = readlink(path, link, link_len - 1);
	if (n < 0)
		return false;
	link[n] = 0;
	return !strncmp(link, "/dmabuf:", 8);
}

static uint64_t fd_size(int fd)
{
	off_t s = lseek(fd, 0, SEEK_END);
	lseek(fd, 0, SEEK_SET);
	return s < 0 ? 0 : (uint64_t)s;
}

static bool is_nv_block_linear(uint64_t mod)
{
	return (mod >> 56) == DRM_FORMAT_MOD_VENDOR_NVIDIA && (mod & 0x10) &&
	       ((mod >> 23) & 7) == 0;
}

/* ------------------------------------------------------------------------ */
/* the verifying Vulkan device (system loader, independent of the library)   */

struct vkctx {
	VkInstance inst;
	VkPhysicalDevice pd;
	VkDevice dev;
	VkQueue q;
	uint32_t qf;
	VkCommandPool pool;
	VkCommandBuffer cb;
	VkFence fence;
	VkPhysicalDeviceMemoryProperties mp;
	PFN_vkGetMemoryFdPropertiesKHR GetMemoryFdPropertiesKHR;
	char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
};

static void vk_setup(struct vkctx *v, dev_t want_rdev)
{
	const VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
					.pApplicationName = "alloc-test",
					.apiVersion = VK_API_VERSION_1_1 };
	const VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
					   .pApplicationInfo = &app };
	VKCHECK(vkCreateInstance(&ici, NULL, &v->inst));
	uint32_t n = 0;
	VKCHECK(vkEnumeratePhysicalDevices(v->inst, &n, NULL));
	VkPhysicalDevice pds[16];
	if (n > 16)
		n = 16;
	VKCHECK(vkEnumeratePhysicalDevices(v->inst, &n, pds));
	for (uint32_t i = 0; i < n && !v->pd; i++) {
		VkPhysicalDeviceDrmPropertiesEXT drm = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT
		};
		VkPhysicalDeviceProperties2 p2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
						   .pNext = &drm };
		vkGetPhysicalDeviceProperties2(pds[i], &p2);
		if (p2.properties.vendorID != 0x10de)
			continue;
		if (drm.hasRender && ((uint64_t)drm.renderMajor != major(want_rdev) ||
				      (uint64_t)drm.renderMinor != minor(want_rdev)))
			continue;
		v->pd = pds[i];
		snprintf(v->name, sizeof(v->name), "%s", p2.properties.deviceName);
	}
	if (!v->pd) {
		printf("  FATAL loader has no NVIDIA device for render node %u:%u\n",
		       major(want_rdev), minor(want_rdev));
		exit(2);
	}
	vkGetPhysicalDeviceMemoryProperties(v->pd, &v->mp);
	uint32_t qn = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(v->pd, &qn, NULL);
	VkQueueFamilyProperties qfp[16];
	if (qn > 16)
		qn = 16;
	vkGetPhysicalDeviceQueueFamilyProperties(v->pd, &qn, qfp);
	v->qf = UINT32_MAX;
	for (uint32_t i = 0; i < qn && v->qf == UINT32_MAX; i++)
		if (qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
			v->qf = i;
	const float prio = 1.0f;
	const VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
					      .queueFamilyIndex = v->qf,
					      .queueCount = 1,
					      .pQueuePriorities = &prio };
	const char *exts[] = { "VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf",
			       "VK_EXT_image_drm_format_modifier", "VK_KHR_image_format_list" };
	const VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
					 .queueCreateInfoCount = 1,
					 .pQueueCreateInfos = &qci,
					 .enabledExtensionCount = 4,
					 .ppEnabledExtensionNames = exts };
	VKCHECK(vkCreateDevice(v->pd, &dci, NULL, &v->dev));
	vkGetDeviceQueue(v->dev, v->qf, 0, &v->q);
	v->GetMemoryFdPropertiesKHR =
	    (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(v->dev, "vkGetMemoryFdPropertiesKHR");
	const VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
					      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
					      .queueFamilyIndex = v->qf };
	VKCHECK(vkCreateCommandPool(v->dev, &pci, NULL, &v->pool));
	const VkCommandBufferAllocateInfo cai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
						  .commandPool = v->pool,
						  .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
						  .commandBufferCount = 1 };
	VKCHECK(vkAllocateCommandBuffers(v->dev, &cai, &v->cb));
	const VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	VKCHECK(vkCreateFence(v->dev, &fci, NULL, &v->fence));
}

static void vk_teardown(struct vkctx *v)
{
	vkDestroyFence(v->dev, v->fence, NULL);
	vkDestroyCommandPool(v->dev, v->pool, NULL);
	vkDestroyDevice(v->dev, NULL);
	vkDestroyInstance(v->inst, NULL);
}

static uint32_t find_type(const struct vkctx *v, uint32_t bits, VkMemoryPropertyFlags want)
{
	for (uint32_t i = 0; i < v->mp.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (v->mp.memoryTypes[i].propertyFlags & want) == want)
			return i;
	return UINT32_MAX;
}

struct hostbuf {
	VkBuffer buf;
	VkDeviceMemory mem;
	void *ptr;
	VkDeviceSize size;
};

static void hostbuf_create(struct vkctx *v, VkDeviceSize size, struct hostbuf *b)
{
	b->size = size;
	const VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
					 .size = size,
					 .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
						  VK_BUFFER_USAGE_TRANSFER_DST_BIT };
	VKCHECK(vkCreateBuffer(v->dev, &bci, NULL, &b->buf));
	VkMemoryRequirements r;
	vkGetBufferMemoryRequirements(v->dev, b->buf, &r);
	const VkMemoryAllocateInfo mai = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.allocationSize = r.size,
		.memoryTypeIndex = find_type(v, r.memoryTypeBits,
					     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
						 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
	};
	VKCHECK(vkAllocateMemory(v->dev, &mai, NULL, &b->mem));
	VKCHECK(vkBindBufferMemory(v->dev, b->buf, b->mem, 0));
	VKCHECK(vkMapMemory(v->dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->ptr));
}

static void hostbuf_destroy(struct vkctx *v, struct hostbuf *b)
{
	vkDestroyBuffer(v->dev, b->buf, NULL);
	vkFreeMemory(v->dev, b->mem, NULL);
}

struct imported {
	VkImage img;
	VkDeviceMemory mem;
	uint32_t mem_type_bits;
};

/* import the dma-buf as an image with the layout the library reported */
static bool vk_import(struct vkctx *v, int fd, uint32_t w, uint32_t h, VkFormat fmt,
		      uint32_t stride, uint64_t modifier, struct imported *out)
{
	const VkSubresourceLayout plane = { .offset = 0, .size = 0, .rowPitch = stride };
	const VkImageDrmFormatModifierExplicitCreateInfoEXT exp = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
		.drmFormatModifier = modifier,
		.drmFormatModifierPlaneCount = 1,
		.pPlaneLayouts = &plane,
	};
	const VkExternalMemoryImageCreateInfo emi = {
		.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
		.pNext = &exp,
		.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
	};
	const VkImageCreateInfo ici = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.pNext = &emi,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = fmt,
		.extent = { w, h, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
		.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	VkResult r = vkCreateImage(v->dev, &ici, NULL, &out->img);
	if (r != VK_SUCCESS) {
		printf("  import: vkCreateImage = %d\n", r);
		return false;
	}
	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(v->dev, out->img, &req);
	VkMemoryFdPropertiesKHR fdp = { .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR };
	r = v->GetMemoryFdPropertiesKHR(v->dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd,
					&fdp);
	if (r != VK_SUCCESS) {
		printf("  import: vkGetMemoryFdPropertiesKHR = %d\n", r);
		return false;
	}
	out->mem_type_bits = fdp.memoryTypeBits;
	const uint32_t type = find_type(v, req.memoryTypeBits & fdp.memoryTypeBits, 0);
	if (type == UINT32_MAX || req.size > fd_size(fd)) {
		printf("  import: no memory type (image 0x%x, fd 0x%x) or size %" PRIu64
		       " > dma-buf %" PRIu64 "\n",
		       req.memoryTypeBits, fdp.memoryTypeBits, (uint64_t)req.size, fd_size(fd));
		return false;
	}
	const int dupfd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
	const VkImportMemoryFdInfoKHR imp = { .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
					      .handleType =
						  VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
					      .fd = dupfd };
	const VkMemoryDedicatedAllocateInfo ded = { .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
						    .pNext = &imp,
						    .image = out->img };
	const VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
					   .pNext = &ded,
					   .allocationSize = req.size,
					   .memoryTypeIndex = type };
	r = vkAllocateMemory(v->dev, &mai, NULL, &out->mem);
	if (r != VK_SUCCESS) {
		close(dupfd);
		printf("  import: vkAllocateMemory(import) = %d\n", r);
		return false;
	}
	VKCHECK(vkBindImageMemory(v->dev, out->img, out->mem, 0));
	return true;
}

static void vk_unimport(struct vkctx *v, struct imported *im)
{
	vkDestroyImage(v->dev, im->img, NULL);
	vkFreeMemory(v->dev, im->mem, NULL);
}

static void begin(struct vkctx *v)
{
	VKCHECK(vkResetCommandBuffer(v->cb, 0));
	const VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
					      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
	VKCHECK(vkBeginCommandBuffer(v->cb, &bi));
}

static void submit_wait(struct vkctx *v)
{
	VKCHECK(vkEndCommandBuffer(v->cb));
	const VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
				  .commandBufferCount = 1,
				  .pCommandBuffers = &v->cb };
	VKCHECK(vkQueueSubmit(v->q, 1, &si, v->fence));
	VKCHECK(vkWaitForFences(v->dev, 1, &v->fence, VK_TRUE, 5000000000ull));
	VKCHECK(vkResetFences(v->dev, 1, &v->fence));
}

static void barrier(struct vkctx *v, VkImage img, VkImageLayout from, VkImageLayout to,
		    uint32_t src_qf, uint32_t dst_qf, VkAccessFlags src_a, VkAccessFlags dst_a,
		    VkPipelineStageFlags src_s, VkPipelineStageFlags dst_s)
{
	const VkImageMemoryBarrier b = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = src_a,
		.dstAccessMask = dst_a,
		.oldLayout = from,
		.newLayout = to,
		.srcQueueFamilyIndex = src_qf,
		.dstQueueFamilyIndex = dst_qf,
		.image = img,
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
	};
	vkCmdPipelineBarrier(v->cb, src_s, dst_s, 0, 0, NULL, 0, NULL, 1, &b);
}

/* GPU write: clear (src == NULL) or upload a tightly packed host buffer, then
 * release the image to VK_QUEUE_FAMILY_EXTERNAL */
static void gpu_write(struct vkctx *v, VkImage img, uint32_t w, uint32_t h,
		      const VkClearColorValue *clear, struct hostbuf *src)
{
	begin(v);
	barrier(v, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	if (clear) {
		const VkImageSubresourceRange rng = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		vkCmdClearColorImage(v->cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, clear, 1, &rng);
	} else {
		const VkBufferImageCopy c = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
					      .imageExtent = { w, h, 1 } };
		vkCmdCopyBufferToImage(v->cb, src->buf, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
				       &c);
	}
	barrier(v, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, v->qf,
		VK_QUEUE_FAMILY_EXTERNAL, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
	submit_wait(v);
}

/* GPU read: acquire from VK_QUEUE_FAMILY_EXTERNAL (contents preserved), copy
 * to a tightly packed host buffer */
static void gpu_read(struct vkctx *v, VkImage img, uint32_t w, uint32_t h, struct hostbuf *dst)
{
	begin(v);
	barrier(v, img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		VK_QUEUE_FAMILY_EXTERNAL, v->qf, 0, VK_ACCESS_TRANSFER_READ_BIT,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	const VkBufferImageCopy c = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
				      .imageExtent = { w, h, 1 } };
	vkCmdCopyImageToBuffer(v->cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst->buf, 1, &c);
	const VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
				     .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
				     .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
	vkCmdPipelineBarrier(v->cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
			     &mb, 0, NULL, 0, NULL);
	submit_wait(v);
}

/* ------------------------------------------------------------------------ */

static uint32_t pat32(uint32_t x, uint32_t y)
{
	return (x & 0xff) | (y & 0xff) << 8 | ((x * 7 + y * 13) & 0xff) << 16 | 0xa5u << 24;
}
static uint16_t pat16(uint32_t x, uint32_t y)
{
	return (uint16_t)(x * 1021u + y * 263u + 0x1234u);
}

static struct gbm_ops *ops;
static const struct gbm_alloc_flags_ops *aflags_ops;
static uint32_t alloc_flags; /* != 0: allocate through get_gbm_alloc_flags_ops() */
static struct gbm_device *gdev;
static struct vkctx V;

static int alloc_buf(uint32_t w, uint32_t h, uint32_t fmt, bool cpu, bool scanout,
		     struct alloc_args *a)
{
	memset(a, 0, sizeof(*a));
	a->gbm = gdev;
	a->width = w;
	a->height = h;
	a->drm_format = fmt;
	a->use_scanout = scanout;
	a->force_linear = cpu;
	a->needs_map_stride = cpu;
	a->out_fd = -1;
	return alloc_flags ? aflags_ops->alloc(a, alloc_flags) : ops->alloc(a);
}

static void print_alloc(const char *what, const struct alloc_args *a)
{
	char link[128] = "";
	is_dmabuf_fd(a->out_fd, link, sizeof(link));
	printf("  %s: fd %d -> %s, size %" PRIu64 ", modifier 0x%016" PRIx64
	       ", stride %u, map_stride %u\n",
	       what, a->out_fd, link, fd_size(a->out_fd), a->out_modifier, a->out_stride,
	       a->out_map_stride);
}

/* (a) GPU buffer: block-linear, displayable layout, GPU round trip between
 * two independent imports of the same dma-buf */
static int saved_bl_fd = -1;
static struct alloc_args saved_bl;

static void test_a(void)
{
	const uint32_t w = 1280, h = 800;
	struct alloc_args a;
	printf("(a) GPU buffer %ux%u ABGR8888, use_scanout\n", w, h);
	int r = alloc_buf(w, h, DRM_FORMAT_ABGR8888, false, true, &a);
	if (r) {
		FAIL("alloc returned %d", r);
		return;
	}
	print_alloc("alloc", &a);
	char link[128];
	EXPECT(is_dmabuf_fd(a.out_fd, link, sizeof(link)), "fd is a dma-buf (%s)", link);
	int fdfl = fcntl(a.out_fd, F_GETFD);
	EXPECT(fdfl >= 0 && (fdfl & FD_CLOEXEC), "fd is O_CLOEXEC");
	EXPECT(is_nv_block_linear(a.out_modifier),
	       "modifier 0x%016" PRIx64 " is NVIDIA block-linear (vendor 0x03, bit4, c=0), h=%u",
	       a.out_modifier, (unsigned)(a.out_modifier & 0xf));
	EXPECT(a.out_stride >= w * 4, "stride %u >= width*4 = %u", a.out_stride, w * 4);
	EXPECT(a.out_map_stride == 0, "map_stride 0 (= stride; GPU buffer)");

	/* the library already destroyed its VkImage and freed its VkDeviceMemory:
	 * every access below goes through the dma-buf alone */
	struct imported A, B;
	if (!vk_import(&V, a.out_fd, w, h, VK_FORMAT_R8G8B8A8_UNORM, a.out_stride, a.out_modifier,
		       &A) ||
	    !vk_import(&V, a.out_fd, w, h, VK_FORMAT_R8G8B8A8_UNORM, a.out_stride, a.out_modifier,
		       &B)) {
		FAIL("import into the loader's Vulkan device");
		close(a.out_fd);
		return;
	}
	PASS("imported twice into a separate VkDevice (\"%s\", via loader), fd memoryTypeBits 0x%x",
	     V.name, A.mem_type_bits);

	struct hostbuf hb;
	hostbuf_create(&V, (VkDeviceSize)w * h * 4, &hb);

	/* clear through A, read through B */
	const VkClearColorValue cc = { .float32 = { 0x11 / 255.0f, 0x22 / 255.0f, 0x33 / 255.0f,
						    0x44 / 255.0f } };
	gpu_write(&V, A.img, w, h, &cc, NULL);
	memset(hb.ptr, 0, hb.size);
	gpu_read(&V, B.img, w, h, &hb);
	uint64_t bad = 0;
	const uint32_t *px = hb.ptr;
	for (uint64_t i = 0; i < (uint64_t)w * h; i++)
		bad += px[i] != 0x44332211u;
	EXPECT(!bad, "GPU clear via import A read back via import B: %" PRIu64 "/%u pixels wrong "
		     "(px0 0x%08x, want 0x44332211)",
	       bad, w * h, px[0]);

	/* upload a position-dependent pattern through A, read through B: the two
	 * images agree on the tiled layout described by (modifier, stride) */
	uint32_t *up = hb.ptr;
	for (uint32_t y = 0; y < h; y++)
		for (uint32_t x = 0; x < w; x++)
			up[y * w + x] = pat32(x, y);
	gpu_write(&V, A.img, w, h, NULL, &hb);
	memset(hb.ptr, 0, hb.size);
	gpu_read(&V, B.img, w, h, &hb);
	bad = 0;
	uint32_t fx = 0, fy = 0;
	for (uint32_t y = 0; y < h; y++)
		for (uint32_t x = 0; x < w; x++)
			if (px[y * w + x] != pat32(x, y) && !bad++) {
				fx = x;
				fy = y;
			}
	EXPECT(!bad, "pattern upload via A read back via B: %" PRIu64 " pixels wrong%s", bad,
	       bad ? "" : " (tiled layout consistent)");
	if (bad)
		printf("    first bad pixel (%u,%u): 0x%08x want 0x%08x\n", fx, fy, px[fy * w + fx],
		       pat32(fx, fy));

	hostbuf_destroy(&V, &hb);
	vk_unimport(&V, &A);
	vk_unimport(&V, &B);
	saved_bl = a;
	saved_bl_fd = a.out_fd; /* kept for the map-refusal check in (b) */
}

/* (b) CPU buffer: LINEAR, CPU write -> GPU read, GPU write -> CPU read */
static uint32_t patpx(uint32_t bpp, uint32_t x, uint32_t y)
{
	return bpp == 1 ? (x * 7 + y * 13) & 0xffu : bpp == 2 ? pat16(x, y) : pat32(x, y);
}
static uint32_t getpx(const void *row, uint32_t bpp, uint32_t x)
{
	return bpp == 1   ? ((const uint8_t *)row)[x]
	       : bpp == 2 ? ((const uint16_t *)row)[x]
			  : ((const uint32_t *)row)[x];
}
static void setpx(void *row, uint32_t bpp, uint32_t x, uint32_t v)
{
	if (bpp == 1)
		((uint8_t *)row)[x] = (uint8_t)v;
	else if (bpp == 2)
		((uint16_t *)row)[x] = (uint16_t)v;
	else
		((uint32_t *)row)[x] = v;
}

static void test_b(uint32_t w, uint32_t h, uint32_t drm_fmt, const char *fname, VkFormat vkfmt,
		   uint32_t bpp, bool check_refusal)
{
	struct alloc_args a;
	printf("(b) CPU buffer %ux%u %s, force_linear + needs_map_stride, alloc flags 0x%x\n", w, h,
	       fname, alloc_flags);
	int r = alloc_buf(w, h, drm_fmt, true, false, &a);
	if (r) {
		FAIL("alloc returned %d", r);
		return;
	}
	print_alloc("alloc", &a);
	char link[128];
	EXPECT(is_dmabuf_fd(a.out_fd, link, sizeof(link)), "fd is a dma-buf (%s)", link);
	EXPECT(a.out_modifier == DRM_FORMAT_MOD_LINEAR, "modifier is LINEAR");
	EXPECT(a.out_stride >= w * bpp && a.out_stride % bpp == 0,
	       "stride %u >= width*%u = %u and a whole number of pixels (stride %% 128 = %u, "
	       "%% 256 = %u)",
	       a.out_stride, bpp, w * bpp, a.out_stride % 128, a.out_stride % 256);
	const uint32_t map_stride = a.out_map_stride ? a.out_map_stride : a.out_stride;
	printf("  CPU stride used: %u (map_stride %u -> minigbm falls back to stride)\n", map_stride,
	       a.out_map_stride);

	struct gbm_bo *bo =
	    ops->import(gdev, a.out_fd, w, h, a.out_stride, a.out_modifier, drm_fmt);
	EXPECT(bo != NULL, "import (mapper path)");
	if (!bo) {
		close(a.out_fd);
		return;
	}
	void *addr = NULL, *md = NULL;
	ops->map(bo, w, h, &addr, &md);
	EXPECT(addr != MAP_FAILED && md != NULL, "map -> %p", addr);
	if (addr == MAP_FAILED) {
		ops->free(bo);
		close(a.out_fd);
		return;
	}
	for (uint32_t y = 0; y < h; y++)
		for (uint32_t x = 0; x < w; x++)
			setpx((uint8_t *)addr + (size_t)y * map_stride, bpp, x, patpx(bpp, x, y));
	ops->unmap(bo, md);

	struct imported I;
	if (!vk_import(&V, a.out_fd, w, h, vkfmt, a.out_stride,
		       a.out_modifier, &I)) {
		FAIL("import LINEAR into the loader's Vulkan device");
		ops->free(bo);
		close(a.out_fd);
		return;
	}
	PASS("LINEAR imported into the verifying VkDevice, fd memoryTypeBits 0x%x",
	     I.mem_type_bits);
	struct hostbuf hb;
	hostbuf_create(&V, (VkDeviceSize)w * h * bpp, &hb);
	gpu_read(&V, I.img, w, h, &hb);
	uint32_t bad = 0;
	for (uint32_t y = 0; y < h; y++)
		for (uint32_t x = 0; x < w; x++)
			bad += getpx((uint8_t *)hb.ptr + (size_t)y * w * bpp, bpp, x) !=
			       patpx(bpp, x, y);
	EXPECT(!bad, "CPU-written pattern read on the GPU: %u/%u pixels wrong", bad, w * h);

	/* GPU clear -> CPU map: R=10/31 G=40/63 B=20/31 is exact in 565 */
	const uint32_t want = bpp == 1 ? 0x11u : bpp == 2 ? (10u << 11) | (40u << 5) | 20u : 0x44332211u;
	const VkClearColorValue cc =
	    bpp == 2 ? (VkClearColorValue){ .float32 = { 10 / 31.0f, 40 / 63.0f, 20 / 31.0f, 1.0f } }
		     : (VkClearColorValue){ .float32 = { 0x11 / 255.0f, 0x22 / 255.0f,
							 0x33 / 255.0f, 0x44 / 255.0f } };
	gpu_write(&V, I.img, w, h, &cc, NULL);
	ops->map(bo, w, h, &addr, &md);
	bad = 0;
	if (addr != MAP_FAILED) {
		for (uint32_t y = 0; y < h; y++)
			for (uint32_t x = 0; x < w; x++)
				bad += getpx((uint8_t *)addr + (size_t)y * map_stride, bpp, x) != want;
		ops->unmap(bo, md);
	}
	EXPECT(addr != MAP_FAILED && !bad,
	       "GPU clear read through map: %u/%u pixels wrong (want 0x%x)", bad, w * h, want);

	hostbuf_destroy(&V, &hb);
	vk_unimport(&V, &I);
	ops->free(bo);
	close(a.out_fd);

	/* a block-linear buffer maps through the backend's GPU staging copy: the
	 * CPU view is linear with rows of the buffer's stride and shows the
	 * pattern (a) uploaded (lock-test covers the full lock contract) */
	if (check_refusal && saved_bl_fd >= 0) {
		const uint32_t bw = saved_bl.width, bh = saved_bl.height, bs = saved_bl.out_stride;
		struct gbm_bo *bl = ops->import(gdev, saved_bl_fd, bw, bh, bs,
						saved_bl.out_modifier, DRM_FORMAT_ABGR8888);
		void *ba = MAP_FAILED, *bmd = NULL;
		if (bl)
			ops->map(bl, (int)bw, (int)bh, &ba, &bmd);
		uint64_t bbad = 0;
		if (ba != MAP_FAILED) {
			for (uint32_t y = 0; y < bh; y++)
				for (uint32_t x = 0; x < bw; x++)
					bbad += ((const uint32_t *)((const uint8_t *)ba +
								    (size_t)y * bs))[x] != pat32(x, y);
			ops->unmap(bl, bmd);
		}
		EXPECT(bl && ba != MAP_FAILED && !bbad,
		       "map of the block-linear buffer from (a) through the staging copy: %" PRIu64
		       " pixels wrong",
		       bbad);
		if (bl)
			ops->free(bl);
		close(saved_bl_fd);
		saved_bl_fd = -1;
	}
}

/* (c) format whitelist + minigbm's R8-blob path for everything else */
static void test_c(void)
{
	printf("(c) formats\n");
	static const struct {
		uint32_t f;
		const char *n;
		bool ok;
	} t[] = {
		{ DRM_FORMAT_R8, "R8", true },
		{ DRM_FORMAT_RGB565, "RGB565", true },
		{ DRM_FORMAT_XRGB8888, "XRGB8888", true },
		{ DRM_FORMAT_ARGB8888, "ARGB8888", true },
		{ DRM_FORMAT_XBGR8888, "XBGR8888", true },
		{ DRM_FORMAT_ABGR8888, "ABGR8888", true },
		{ DRM_FORMAT_ABGR2101010, "ABGR2101010", true },
		{ DRM_FORMAT_ABGR16161616F, "ABGR16161616F", true },
		{ DRM_FORMAT_NV12, "NV12", false },
		{ DRM_FORMAT_YVU420, "YVU420", false },
		{ DRM_FORMAT_BGR888, "BGR888", false },
		{ DRM_FORMAT_GR88, "GR88", false },
		{ DRM_FORMAT_ARGB2101010, "ARGB2101010", false },
		{ DRM_FORMAT_XBGR16161616F, "XBGR16161616F", false },
		{ DRM_FORMAT_P010, "P010", false },
	};
	for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) {
		const uint32_t g = ops->get_gbm_format(t[i].f);
		EXPECT(t[i].ok ? g == t[i].f : g == 0, "get_gbm_format(%s) = 0x%08x", t[i].n, g);
	}

	/* every whitelisted format allocates both ways */
	for (size_t i = 0; i < sizeof(t) / sizeof(t[0]); i++) {
		if (!t[i].ok)
			continue;
		for (int cpu = 0; cpu < 2; cpu++) {
			struct alloc_args a;
			int r = alloc_buf(256, 256, t[i].f, cpu, false, &a);
			EXPECT(!r && (cpu ? a.out_modifier == DRM_FORMAT_MOD_LINEAR
					  : is_nv_block_linear(a.out_modifier)),
			       "alloc 256x256 %-13s %s: modifier 0x%016" PRIx64 " stride %u size %" PRIu64,
			       t[i].n, cpu ? "CPU" : "GPU", a.out_modifier, a.out_stride,
			       r ? 0 : fd_size(a.out_fd));
			if (!r)
				close(a.out_fd);
		}
	}

	/* what minigbm does with an unsupported format (gbm_mesa_internals.cpp:342-364):
	 * a 4096-wide linear R8 carrier, CPU mapped */
	struct alloc_args a;
	const uint32_t blob = 1920 * 1080 * 3 / 2; /* NV12 1080p */
	const uint32_t bh = (blob + 4095) / 4096;
	int r = alloc_buf(4096, bh, DRM_FORMAT_R8, true, false, &a);
	if (r) {
		FAIL("R8 blob alloc 4096x%u returned %d", bh, r);
		return;
	}
	print_alloc("R8 blob 4096x" , &a);
	EXPECT(a.out_modifier == DRM_FORMAT_MOD_LINEAR && a.out_stride == 4096 &&
		   fd_size(a.out_fd) >= blob,
	       "R8 blob carrier 4096x%u: LINEAR, stride %u, dma-buf %" PRIu64 " >= %u bytes", bh,
	       a.out_stride, fd_size(a.out_fd), blob);
	struct gbm_bo *bo = ops->import(gdev, a.out_fd, 4096, bh, a.out_stride, a.out_modifier,
					DRM_FORMAT_R8);
	void *addr = MAP_FAILED, *md = NULL;
	if (bo)
		ops->map(bo, 4096, (int)bh, &addr, &md);
	if (addr != MAP_FAILED) {
		for (uint32_t i = 0; i < blob; i++)
			((uint8_t *)addr)[i] = (uint8_t)(i * 31 + (i >> 12));
		ops->unmap(bo, md);
		ops->map(bo, 4096, (int)bh, &addr, &md);
	}
	uint32_t bad = 0;
	if (addr != MAP_FAILED) {
		for (uint32_t i = 0; i < blob; i++)
			bad += ((uint8_t *)addr)[i] != (uint8_t)(i * 31 + (i >> 12));
		ops->unmap(bo, md);
	}
	EXPECT(bo && addr != MAP_FAILED && !bad, "R8 blob CPU write/remap/read: %u bytes wrong",
	       bad);
	if (bo)
		ops->free(bo);
	close(a.out_fd);
}

/* (0) the very first allocations of the process race each other: the lazy
 * Vulkan init must happen once and every thread must get a buffer */
static void *race_thread(void *arg)
{
	return (void *)(intptr_t)ops->alloc((struct alloc_args *)arg);
}

static void test_race(void)
{
	enum { N = 8 };
	struct alloc_args a[N];
	pthread_t t[N];
	printf("(0) %d threads make the process's first allocations concurrently\n", N);
	for (int i = 0; i < N; i++) {
		memset(&a[i], 0, sizeof(a[i]));
		a[i].gbm = gdev;
		a[i].width = 640 + 16 * (uint32_t)i;
		a[i].height = 480;
		a[i].drm_format = DRM_FORMAT_ARGB8888;
		a[i].force_linear = a[i].needs_map_stride = i & 1;
		a[i].out_fd = -1;
		pthread_create(&t[i], NULL, race_thread, &a[i]);
	}
	int ok = 0;
	for (int i = 0; i < N; i++) {
		void *rv;
		pthread_join(t[i], &rv);
		const bool good = !rv && a[i].out_fd >= 0 &&
				  ((i & 1) ? a[i].out_modifier == DRM_FORMAT_MOD_LINEAR
					   : is_nv_block_linear(a[i].out_modifier));
		ok += good;
		if (a[i].out_fd >= 0)
			close(a[i].out_fd);
	}
	EXPECT(ok == N, "%d/%d concurrent first allocations succeeded with the right layout", ok,
	       N);
}

/* (d) leak check */
static void test_d(void)
{
	const int cycles = 200, warm = 10;
	const uint32_t w = 1920, h = 1080;
	printf("(d) %d alloc/free cycles of %ux%u (GPU ABGR8888 + CPU ABGR8888 with "
	       "import/map/unmap/free)\n",
	       cycles, w, h);
	int errors = 0;
	for (int pass = 0; pass < 2; pass++) {
		static int fds0;
		static long rss0, gpu0, avail0;
		const int n = pass ? cycles : warm;
		if (pass) {
			fds0 = count_fds();
			rss0 = vm_rss_kb();
			gpu0 = gpu_mem_mib();
			avail0 = mem_available_mib();
		}
		for (int i = 0; i < n; i++) {
			struct alloc_args a;
			if (alloc_buf(w, h, DRM_FORMAT_ABGR8888, false, true, &a) ||
			    !is_nv_block_linear(a.out_modifier)) {
				errors++;
				continue;
			}
			close(a.out_fd);
			if (alloc_buf(w, h, DRM_FORMAT_ABGR8888, true, false, &a)) {
				errors++;
				continue;
			}
			struct gbm_bo *bo = ops->import(gdev, a.out_fd, w, h, a.out_stride,
							a.out_modifier, DRM_FORMAT_ABGR8888);
			void *addr = MAP_FAILED, *md = NULL;
			if (bo)
				ops->map(bo, (int)w, (int)h, &addr, &md);
			if (addr == MAP_FAILED)
				errors++;
			else {
				((volatile uint32_t *)addr)[0] = 0xdeadbeef;
				ops->unmap(bo, md);
			}
			if (bo)
				ops->free(bo);
			close(a.out_fd);
		}
		if (pass) {
			const int fds1 = count_fds();
			const long rss1 = vm_rss_kb();
			const long gpu1 = gpu_mem_mib();
			printf("  open fds   %d -> %d\n", fds0, fds1);
			printf("  VmRSS      %ld -> %ld kB (delta %+ld kB)\n", rss0, rss1, rss1 - rss0);
			printf("  GPU memory %ld -> %ld MiB (nvidia-smi memory.used, GPU %s, whole "
			       "device)\n",
			       gpu0, gpu1, gpu_pci);
			EXPECT(!errors, "%d/%d cycles succeeded", 2 * cycles - errors, 2 * cycles);
			EXPECT(fds1 == fds0, "no fd leak (%+d)", fds1 - fds0);
			/* one leaked 1080p buffer per cycle would be >= 1.6 GB */
			EXPECT(rss1 - rss0 < 8 * 1024, "RSS stable (%+ld kB < 8 MiB)", rss1 - rss0);
			const long avail1 = mem_available_mib();
			printf("  MemAvailable %ld -> %ld MiB (system-wide)\n", avail0, avail1);
			/* sensitivity for system memory: hold 20 CPU 1080p buffers */
			{
				int cheld[20], nc = 0;
				for (; nc < 20; nc++) {
					struct alloc_args a;
					if (alloc_buf(w, h, DRM_FORMAT_ABGR8888, true, false, &a))
						break;
					cheld[nc] = a.out_fd;
				}
				const long av_held = mem_available_mib();
				for (int i = 0; i < nc; i++)
					close(cheld[i]);
				const long av_after = mem_available_mib();
				printf("  sensitivity: %d live 1080p CPU buffers -> MemAvailable %ld MiB, "
				       "closed -> %ld MiB\n",
				       nc, av_held, av_after);
				EXPECT(nc == 20 && avail1 - av_held >= 20 * 7,
				       "MemAvailable sees live CPU dma-bufs (-%ld MiB while held)",
				       avail1 - av_held);
				/* a leak of the 200 CPU buffers would be >= 1580 MiB */
				EXPECT(avail0 - avail1 < 512,
				       "system memory stable over the cycles (%+ld MiB, whole system; "
				       "a leak would be <= -1580 MiB)",
				       avail1 - avail0);
			}
			/* sensitivity: the same measurement must see live buffers */
			int held[20];
			int nheld = 0;
			for (; nheld < 20; nheld++) {
				struct alloc_args a;
				if (alloc_buf(w, h, DRM_FORMAT_ABGR8888, false, true, &a))
					break;
				held[nheld] = a.out_fd;
			}
			const long gpu_held = gpu_mem_mib();
			for (int i = 0; i < nheld; i++)
				close(held[i]);
			const long gpu2 = gpu_mem_mib();
			printf("  sensitivity: %d live 1080p GPU buffers -> %ld MiB, closed -> %ld MiB\n",
			       nheld, gpu_held, gpu2);
			EXPECT(nheld == 20 && gpu_held - gpu1 >= 20 * 8 && gpu2 - gpu1 <= 32,
			       "measurement sees live dma-bufs (+%ld MiB while held, %+ld MiB after "
			       "close)",
			       gpu_held - gpu1, gpu2 - gpu1);
			if (gpu0 < 0 || gpu1 < 0)
				FAIL("nvidia-smi memory.used not readable for GPU %s", gpu_pci);
			else
				EXPECT(gpu1 - gpu0 <= 32, "GPU memory stable (%+ld MiB, other "
							  "processes included; a leak would be "
							  ">= 1580 MiB)",
				       gpu1 - gpu0);
		}
	}
}

int main(int argc, char **argv)
{
	/* The fd checks count the whole process. Mesa's shader disk cache writes each entry on its own nice-19
	 * "disk$" threads (src/util/disk_cache.c:90-93, src/util/u_queue.c:258) and holds a <hash>.tmp descriptor
	 * from open to close (src/util/disk_cache_os.c:883-972, disk_cache_write_item_to_disk), unordered with
	 * those counts. The explicit disable also covers a cache at an explicit path (disk_cache_os.c:1182).
	 * Lines: Mesa 26.2.5 with this repository's Mesa patches applied. Set before any Mesa library loads. */
	if (setenv("MESA_SHADER_CACHE_DISABLE", "true", 1) != 0) {
		perror("setenv MESA_SHADER_CACHE_DISABLE");
		return 2;
	}
	const char *lib = argc > 1 ? argv[1] : "out/host/libgbm_mesa_wrapper.so";
	setvbuf(stdout, NULL, _IOLBF, 0);

	dev_t rdev = 0;
	char node[64] = "";
	int drm_fd = open_nvidia_render_node(&rdev, node, sizeof(node));
	if (drm_fd < 0) {
		printf("FATAL: no render node with sysfs vendor 0x10de\n");
		return 2;
	}
	printf("NVIDIA render node by sysfs vendor 0x10de: %s (%u:%u)\n", node, major(rdev),
	       minor(rdev));

	void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
	if (!h) {
		printf("FATAL: dlopen %s: %s\n", lib, dlerror());
		return 2;
	}
	struct gbm_ops *(*get)(void) = (struct gbm_ops * (*)(void)) dlsym(h, "get_gbm_ops");
	if (!get || !(ops = get())) {
		printf("FATAL: get_gbm_ops\n");
		return 2;
	}
	gdev = ops->dev_create(drm_fd);
	if (!gdev) {
		printf("FATAL: dev_create returned NULL\n");
		return 2;
	}
	vk_setup(&V, rdev);
	printf("verifier: \"%s\" via the system Vulkan loader\n", V.name);

	test_race();
	test_a();
	test_b(64, 64, DRM_FORMAT_RGB565, "RGB565", VK_FORMAT_R5G6B5_UNORM_PACK16, 2, true);
	/* odd width: the CPU stride must come from the driver's rowPitch */
	test_b(1001, 37, DRM_FORMAT_ABGR8888, "ABGR8888", VK_FORMAT_R8G8B8A8_UNORM, 4, false);
	test_b(333, 17, DRM_FORMAT_RGB565, "RGB565", VK_FORMAT_R5G6B5_UNORM_PACK16, 2, false);
	/* (b) again for buffers the CPU reads often: the wrapper places them in cached
	 * memory; the CPU-write -> GPU-read and GPU-clear -> CPU-read checks are the
	 * coherence contract of that placement */
	const struct gbm_alloc_flags_ops *(*get_af)(void) =
	    (const struct gbm_alloc_flags_ops *(*)(void))dlsym(h, GBM_GET_ALLOC_FLAGS_OPS_SYMBOL);
	aflags_ops = get_af ? get_af() : NULL;
	EXPECT(aflags_ops && aflags_ops->version == GBM_ALLOC_FLAGS_OPS_VERSION && aflags_ops->alloc,
	       "%s exported, version %u", GBM_GET_ALLOC_FLAGS_OPS_SYMBOL,
	       aflags_ops ? aflags_ops->version : 0u);
	if (aflags_ops && aflags_ops->alloc) {
		alloc_flags = GBM_ALLOC_FLAG_CPU_READ_OFTEN;
		test_b(1001, 37, DRM_FORMAT_ABGR8888, "ABGR8888", VK_FORMAT_R8G8B8A8_UNORM, 4, false);
		test_b(333, 17, DRM_FORMAT_RGB565, "RGB565", VK_FORMAT_R5G6B5_UNORM_PACK16, 2, false);
		/* the R8 carrier minigbm allocates for a 1920x816 YV12 decoder output block */
		test_b(4096, 574, DRM_FORMAT_R8, "R8", VK_FORMAT_R8_UNORM, 1, false);
		alloc_flags = 1u << 31;
		struct alloc_args a;
		const int r = alloc_buf(64, 64, DRM_FORMAT_R8, true, false, &a);
		EXPECT(r == -EINVAL, "unknown allocation flag refused: %d", r);
		if (!r)
			close(a.out_fd);
		alloc_flags = 0;
	}
	test_c();
	test_d();

	ops->dev_destroy(gdev);
	vk_teardown(&V);
	close(drm_fd);
	printf("%s: %d failure(s)\n", failures ? "FAILED" : "ALL PASSED", failures);
	return failures ? 1 : 0;
}
