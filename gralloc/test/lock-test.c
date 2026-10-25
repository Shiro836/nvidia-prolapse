/*
 * Host test: CPU lock/unlock of CPU-mapped GPU render targets (README "Layout
 * rule", "Lock/unlock of block-linear buffers").
 *
 * A SurfaceFlinger-screenshot-style buffer (usage 0xb33: SW_READ/WRITE_OFTEN |
 * HW_RENDER | HW_TEXTURE | HW_COMPOSER) reaches the backend, with our minigbm
 * patch, as alloc(needs_map_stride=1, force_linear=0, use_scanout=1). For each
 * format/size this test:
 *   1. allocates it: block-linear, map_stride == stride (a valid linear pitch);
 *   2. imports it as minigbm's mapper does and CPU-writes a pattern through
 *      map/unmap; checks it on the GPU through an independent Vulkan import;
 *   3. renders into it with GL (stock EGL/GLES via EGL_EXT_image_dma_buf_import
 *      as GL_TEXTURE_2D + FBO: a scissored clear and a shader draw) and reads it
 *      back through map;
 *   4. renders into it with Vulkan (render pass + vkCmdClearAttachments) and
 *      reads it back through map;
 *   5. maps it in a separate exec'd process that never allocated (lazy ICD
 *      bring-up), checks everything, CPU-writes a rectangle, unmaps; the parent
 *      checks that rectangle on the GPU (Vulkan) and, for RGBA8, through GL;
 * exact pixels everywhere. Then the lock access flags (get_gbm_map_flags_ops,
 * what our minigbm patch 0002 calls): a READ lock never writes the buffer (a
 * GPU write that lands between its map and its unmap survives, and CPU writes
 * into the read view are not copied back), a READ|WRITE lock and the plain
 * gbm_ops.map/unmap do copy back (control: the same GPU write is overwritten,
 * so the test can see a copy-back), a WRITE-only lock keeps the content it
 * does not write, and a mapping widened to WRITE at unmap is copied back.
 * Then: lock/unlock timings at 2560x1412 RGBA8, BAR write bandwidth for
 * comparison, and a leak check.
 *
 *   usage: lock-test [path/to/libgbm_mesa_wrapper.so] [--source-linear|--source-modifiers|--modifiers|--source-stress]
 * --source-stress runs the full suite, including 200-cycle stress, with
 * native NVK format expectations; stock EGL is exercised on block-linear
 * buffers, and R8/RGB565 LINEAR buffers use the native Vulkan consumer.
 * --modifiers runs the same bounded modifier cases with any NVIDIA driver.
 * --source-modifiers runs four bounded RGBA8/FP16 sizes through stock EGL and
 * native NVK imports; it omits timing and stress loops.
 * --source-linear verifies native NVK R8/RGB565 LINEAR rendering, CPU mapping
 * and cross-process sharing. It does not run the proprietary GLES consumer.
 *   (internal) lock-test --child LIB FD FMT W H STRIDE MODHI MODLO
 * Exit 0 only when every check passed.
 */
#define _GNU_SOURCE

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <drm_fourcc.h>
#include <vulkan/vulkan.h>
#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#define GL_GLEXT_PROTOTYPES
#include <GLES2/gl2ext.h>

#include "gbm_mesa_wrapper.h"
#include "gbm_mesa_wrapper_map_flags.h"

static int failures;
static char gpu_pci[32];
static const char *g_lib;
static bool g_source_linear;
static bool g_source_stress;

#define PASS(...) (printf("  PASS "), printf(__VA_ARGS__), printf("\n"))
#define FAIL(...) (printf("  FAIL "), printf(__VA_ARGS__), printf("\n"), failures++)
#define EXPECT(c, ...)                                                                             \
	do {                                                                                       \
		if (c)                                                                             \
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

static double now_ms(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b)
{
	const double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

/* ------------------------------------------------------------------------ */
/* formats under test                                                        */

struct fmt {
	uint32_t drm;
	VkFormat vk;
	uint32_t bpp;
	const char *name;
};
static const struct fmt F_R8 = { DRM_FORMAT_R8, VK_FORMAT_R8_UNORM, 1, "R8" };
static const struct fmt F_RGBA8 = { DRM_FORMAT_ABGR8888, VK_FORMAT_R8G8B8A8_UNORM, 4, "ABGR8888" };
static const struct fmt F_565 = { DRM_FORMAT_RGB565, VK_FORMAT_R5G6B5_UNORM_PACK16, 2, "RGB565" };
static const struct fmt F_FP16 = { DRM_FORMAT_ABGR16161616F, VK_FORMAT_R16G16B16A16_SFLOAT, 8,
				   "ABGR16161616F" };

static const struct fmt *fmt_by_drm(uint32_t drm)
{
	const struct fmt *all[] = { &F_RGBA8, &F_565, &F_FP16, &F_R8 };
	for (unsigned i = 0; i < sizeof(all) / sizeof(all[0]); i++)
		if (all[i]->drm == drm)
			return all[i];
	return NULL;
}

/* pixel values in memory, as uint64 (low bytes first) */
enum colour { RED, BLUE, GREEN };
static uint64_t colour_px(const struct fmt *f, enum colour c)
{
	const float v[3][4] = { { 1, 0, 0, 1 }, { 0, 0, 1, 1 }, { 0, 1, 0, 1 } };
	const float *k = v[c];
	switch (f->bpp) {
	case 1:
		return (uint64_t)(k[0] * 255);
	case 4:
		return (uint64_t)(k[0] * 255) | (uint64_t)(k[1] * 255) << 8 |
		       (uint64_t)(k[2] * 255) << 16 | (uint64_t)(k[3] * 255) << 24;
	case 2:
		return (uint64_t)(k[0] * 31) << 11 | (uint64_t)(k[1] * 63) << 5 | (uint64_t)(k[2] * 31);
	default: /* half floats: 1.0 = 0x3c00, 0 = 0 */
		return (k[0] ? 0x3c00ull : 0) | (k[1] ? 0x3c00ull : 0) << 16 |
		       (k[2] ? 0x3c00ull : 0) << 32 | (k[3] ? 0x3c00ull : 0) << 48;
	}
}
static const float colour_f[3][4] = { { 1, 0, 0, 1 }, { 0, 0, 1, 1 }, { 0, 1, 0, 1 } };

static uint64_t pat_px(const struct fmt *f, uint32_t seed, uint32_t x, uint32_t y)
{
	uint64_t h = (uint64_t)x * 0x9e3779b97f4a7c15ull ^ (uint64_t)y * 0xc2b2ae3d27d4eb4full ^
		     (uint64_t)seed * 0x165667b19e3779f9ull;
	h ^= h >> 29;
	h *= 0xbf58476d1ce4e5b9ull;
	h ^= h >> 32;
	return f->bpp == 8 ? h : h & ((1ull << (8 * f->bpp)) - 1);
}

static uint64_t get_px(const void *row, const struct fmt *f, uint32_t x)
{
	uint64_t v = 0;
	memcpy(&v, (const uint8_t *)row + (size_t)x * f->bpp, f->bpp);
	return v;
}
static void set_px(void *row, const struct fmt *f, uint32_t x, uint64_t v)
{
	memcpy((uint8_t *)row + (size_t)x * f->bpp, &v, f->bpp);
}

/* the painting order of a case: P1 everywhere, then A (GL clear, red),
 * A2 (GL draw, blue), B (Vulkan, green), D (CPU in the child, pattern 3) */
struct rect {
	uint32_t x0, y0, x1, y1;
};
struct scene {
	uint32_t w, h;
	struct rect A, A2, B, D;
	int stage; /* how many paints have happened: 0 = P1 only .. 4 = D */
};
static bool in(const struct rect *r, uint32_t x, uint32_t y)
{
	return x >= r->x0 && x < r->x1 && y >= r->y0 && y < r->y1;
}
static bool empty(const struct rect *r)
{
	return r->x1 <= r->x0 || r->y1 <= r->y0;
}
static void scene_init(struct scene *s, uint32_t w, uint32_t h)
{
	s->w = w;
	s->h = h;
	s->A = (struct rect){ 0, 0, (w + 1) / 2, (h + 2) / 3 };
	s->A2 = (struct rect){ w / 4, h / 2, w, h };
	s->B = (struct rect){ w / 3, h / 5, w / 3 + (w + 2) / 3, h / 5 + (h + 1) / 2 };
	s->D = (struct rect){ w / 2, 0, w, (h + 1) / 2 };
	s->stage = 0;
}
static uint64_t scene_px(const struct scene *s, const struct fmt *f, uint32_t x, uint32_t y)
{
	if (s->stage >= 4 && in(&s->D, x, y))
		return pat_px(f, 3, x, y);
	if (s->stage >= 3 && in(&s->B, x, y))
		return colour_px(f, GREEN);
	if (s->stage >= 2 && in(&s->A2, x, y))
		return colour_px(f, BLUE);
	if (s->stage >= 1 && in(&s->A, x, y))
		return colour_px(f, RED);
	return pat_px(f, 1, x, y);
}
/* count wrong pixels of a linear view with `pitch`-byte rows */
static uint64_t scene_check(const struct scene *s, const struct fmt *f, const void *base,
			    size_t pitch, uint32_t *fx, uint32_t *fy, uint64_t *got)
{
	uint64_t bad = 0;
	for (uint32_t y = 0; y < s->h; y++) {
		const uint8_t *row = (const uint8_t *)base + (size_t)y * pitch;
		for (uint32_t x = 0; x < s->w; x++) {
			const uint64_t v = get_px(row, f, x);
			if (v != scene_px(s, f, x, y) && !bad++) {
				*fx = x;
				*fy = y;
				*got = v;
			}
		}
	}
	return bad;
}

/* ------------------------------------------------------------------------ */
/* render node, process measurements (as alloc-test)                         */

static int open_nvidia_render_node(dev_t *rdev)
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
	}
	globfree(&g);
	return fd;
}

static int count_fds(void)
{
	DIR *d = opendir("/proc/self/fd");
	int n = 0;
	for (struct dirent *e; (e = readdir(d));)
		if (e->d_name[0] != '.')
			n++;
	closedir(d);
	return n - 1;
}

static long status_kb(const char *key)
{
	FILE *f = fopen("/proc/self/status", "r");
	char line[256];
	long kb = -1;
	const size_t kl = strlen(key);
	while (fgets(line, sizeof(line), f))
		if (!strncmp(line, key, kl) && sscanf(line + kl, " %ld", &kb) == 1)
			break;
	fclose(f);
	return kb;
}

static long gpu_mem_mib(void)
{
	FILE *p = popen("nvidia-smi --query-gpu=pci.bus_id,memory.used --format=csv,noheader,nounits "
			"2>/dev/null",
			"r");
	if (!p)
		return -1;
	char bus[64];
	long mib, found = -1;
	while (fscanf(p, " %63[^,], %ld", bus, &mib) == 2) {
		const char *a = strchr(bus, ':'), *b = strchr(gpu_pci, ':');
		if (a && b && !strcasecmp(a, b))
			found = mib;
	}
	pclose(p);
	return found;
}

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

static bool is_nv_block_linear(uint64_t mod)
{
	return (mod >> 56) == DRM_FORMAT_MOD_VENDOR_NVIDIA && (mod & 0x10) && ((mod >> 23) & 7) == 0;
}

/* ------------------------------------------------------------------------ */
/* the library under test                                                    */

static struct gbm_ops *ops;
static const struct gbm_map_flags_ops *mfops;
static struct gbm_device *gdev;

static void load_lib(int drm_fd)
{
	void *h = dlopen(g_lib, RTLD_NOW | RTLD_LOCAL);
	if (!h) {
		printf("FATAL: dlopen %s: %s\n", g_lib, dlerror());
		exit(2);
	}
	struct gbm_ops *(*get)(void) = (struct gbm_ops * (*)(void)) dlsym(h, "get_gbm_ops");
	if (!get || !(ops = get()) || !(gdev = ops->dev_create(drm_fd))) {
		printf("FATAL: get_gbm_ops/dev_create\n");
		exit(2);
	}
	const struct gbm_map_flags_ops *(*getmf)(void) =
	    (const struct gbm_map_flags_ops * (*)(void)) dlsym(h, GBM_GET_MAP_FLAGS_OPS_SYMBOL);
	mfops = getmf ? getmf() : NULL;
	if (!mfops || mfops->version != GBM_MAP_FLAGS_OPS_VERSION || !mfops->map || !mfops->unmap) {
		printf("FATAL: %s missing or not version %u\n", GBM_GET_MAP_FLAGS_OPS_SYMBOL,
		       GBM_MAP_FLAGS_OPS_VERSION);
		exit(2);
	}
}

/* what gbm_mesa_bo_create passes for an 0xb33 buffer with our minigbm patch */
static int alloc_sw_render(uint32_t w, uint32_t h, const struct fmt *f, struct alloc_args *a)
{
	memset(a, 0, sizeof(*a));
	a->gbm = gdev;
	a->width = w;
	a->height = h;
	a->drm_format = f->drm;
	a->use_scanout = true;
	a->force_linear = false;
	a->needs_map_stride = true;
	a->out_fd = -1;
	return ops->alloc(a);
}

/* ------------------------------------------------------------------------ */
/* verifying Vulkan device (system loader)                                   */

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
};
static struct vkctx V;

static void vk_setup(struct vkctx *v, dev_t want_rdev)
{
	const VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
					.pApplicationName = "lock-test",
					.apiVersion = VK_API_VERSION_1_1 };
	const VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
					   .pApplicationInfo = &app };
	VKCHECK(vkCreateInstance(&ici, NULL, &v->inst));
	uint32_t n = 16;
	VkPhysicalDevice pds[16];
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
	}
	if (!v->pd) {
		printf("  FATAL loader has no NVIDIA device for the render node\n");
		exit(2);
	}
	vkGetPhysicalDeviceMemoryProperties(v->pd, &v->mp);
	uint32_t qn = 16;
	VkQueueFamilyProperties qfp[16];
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

static uint32_t find_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid)
{
	for (uint32_t i = 0; i < V.mp.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (V.mp.memoryTypes[i].propertyFlags & want) == want &&
		    !(V.mp.memoryTypes[i].propertyFlags & avoid))
			return i;
	return UINT32_MAX;
}

struct hostbuf {
	VkBuffer buf;
	VkDeviceMemory mem;
	void *ptr;
	VkDeviceSize size;
};

static bool hostbuf_create(VkDeviceSize size, VkMemoryPropertyFlags want,
			   VkMemoryPropertyFlags avoid, struct hostbuf *b)
{
	b->size = size;
	const VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
					 .size = size,
					 .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
						  VK_BUFFER_USAGE_TRANSFER_DST_BIT };
	VKCHECK(vkCreateBuffer(V.dev, &bci, NULL, &b->buf));
	VkMemoryRequirements r;
	vkGetBufferMemoryRequirements(V.dev, b->buf, &r);
	const uint32_t t = find_type(r.memoryTypeBits, want, avoid);
	if (t == UINT32_MAX) {
		vkDestroyBuffer(V.dev, b->buf, NULL);
		return false;
	}
	const VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
					   .allocationSize = r.size,
					   .memoryTypeIndex = t };
	VKCHECK(vkAllocateMemory(V.dev, &mai, NULL, &b->mem));
	VKCHECK(vkBindBufferMemory(V.dev, b->buf, b->mem, 0));
	VKCHECK(vkMapMemory(V.dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->ptr));
	return true;
}

static void hostbuf_destroy(struct hostbuf *b)
{
	vkDestroyBuffer(V.dev, b->buf, NULL);
	vkFreeMemory(V.dev, b->mem, NULL);
}

struct imported {
	VkImage img;
	VkDeviceMemory mem;
	VkImageView view;
};

static bool vk_import(int fd, uint32_t w, uint32_t h, const struct fmt *f, uint32_t stride,
		      uint64_t modifier, bool colour_target, struct imported *out)
{
	memset(out, 0, sizeof(*out));
	const VkSubresourceLayout plane = { .offset = 0, .rowPitch = stride };
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
		.format = f->vk,
		.extent = { w, h, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
		.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
			 (colour_target ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0),
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	if (vkCreateImage(V.dev, &ici, NULL, &out->img) != VK_SUCCESS)
		return false;
	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(V.dev, out->img, &req);
	VkMemoryFdPropertiesKHR fdp = { .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR };
	VKCHECK(V.GetMemoryFdPropertiesKHR(V.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd,
					   &fdp));
	const uint32_t type = find_type(req.memoryTypeBits & fdp.memoryTypeBits, 0, 0);
	const int dupfd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
	const VkImportMemoryFdInfoKHR imp = { .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
					      .handleType =
						  VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
					      .fd = dupfd };
	const VkMemoryDedicatedAllocateInfo ded = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .pNext = &imp, .image = out->img
	};
	const VkMemoryAllocateInfo mai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
					   .pNext = &ded,
					   .allocationSize = req.size,
					   .memoryTypeIndex = type };
	if (type == UINT32_MAX || vkAllocateMemory(V.dev, &mai, NULL, &out->mem) != VK_SUCCESS) {
		close(dupfd);
		vkDestroyImage(V.dev, out->img, NULL);
		return false;
	}
	VKCHECK(vkBindImageMemory(V.dev, out->img, out->mem, 0));
	if (colour_target) {
		const VkImageViewCreateInfo vci = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = out->img,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = f->vk,
			.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
		};
		VKCHECK(vkCreateImageView(V.dev, &vci, NULL, &out->view));
	}
	return true;
}

static void vk_unimport(struct imported *im)
{
	if (im->view)
		vkDestroyImageView(V.dev, im->view, NULL);
	vkDestroyImage(V.dev, im->img, NULL);
	vkFreeMemory(V.dev, im->mem, NULL);
}

static void begin(void)
{
	VKCHECK(vkResetCommandBuffer(V.cb, 0));
	const VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
					      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
	VKCHECK(vkBeginCommandBuffer(V.cb, &bi));
}

static void submit_wait(void)
{
	VKCHECK(vkEndCommandBuffer(V.cb));
	const VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
				  .commandBufferCount = 1,
				  .pCommandBuffers = &V.cb };
	VKCHECK(vkQueueSubmit(V.q, 1, &si, V.fence));
	VKCHECK(vkWaitForFences(V.dev, 1, &V.fence, VK_TRUE, 5000000000ull));
	VKCHECK(vkResetFences(V.dev, 1, &V.fence));
}

static void img_barrier(VkImage img, VkImageLayout from, VkImageLayout to, uint32_t sq,
			uint32_t dq, VkAccessFlags sa, VkAccessFlags da, VkPipelineStageFlags ss,
			VkPipelineStageFlags ds)
{
	const VkImageMemoryBarrier b = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = sa,
		.dstAccessMask = da,
		.oldLayout = from,
		.newLayout = to,
		.srcQueueFamilyIndex = sq,
		.dstQueueFamilyIndex = dq,
		.image = img,
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
	};
	vkCmdPipelineBarrier(V.cb, ss, ds, 0, 0, NULL, 0, NULL, 1, &b);
}

/* tightly packed copy of the image into a host buffer (contents preserved) */
static void gpu_read(VkImage img, uint32_t w, uint32_t h, struct hostbuf *dst)
{
	begin();
	img_barrier(img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		    VK_QUEUE_FAMILY_EXTERNAL, V.qf, 0, VK_ACCESS_TRANSFER_READ_BIT,
		    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
	const VkBufferImageCopy c = { .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
				      .imageExtent = { w, h, 1 } };
	vkCmdCopyImageToBuffer(V.cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst->buf, 1, &c);
	img_barrier(img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, V.qf,
		    VK_QUEUE_FAMILY_EXTERNAL, VK_ACCESS_TRANSFER_READ_BIT, 0,
		    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
	const VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
				     .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
				     .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
	vkCmdPipelineBarrier(V.cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
			     &mb, 0, NULL, 0, NULL);
	submit_wait();
}

/* Vulkan rendering: a render pass on the imported image (load = LOAD, so the
 * rest is preserved) and vkCmdClearAttachments on rect r: colour attachment
 * writes, the way a Vulkan renderer writes the buffer */
static void vk_render_rect(struct imported *im, const struct fmt *f, uint32_t w, uint32_t h,
			   const struct rect *r, enum colour c)
{
	const VkAttachmentDescription ad = {
		.format = f->vk,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
		.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
		.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	};
	const VkAttachmentReference ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
	const VkSubpassDescription sp = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
					  .colorAttachmentCount = 1,
					  .pColorAttachments = &ref };
	const VkRenderPassCreateInfo rpci = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
					      .attachmentCount = 1,
					      .pAttachments = &ad,
					      .subpassCount = 1,
					      .pSubpasses = &sp };
	VkRenderPass rp;
	VKCHECK(vkCreateRenderPass(V.dev, &rpci, NULL, &rp));
	const VkFramebufferCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
					      .renderPass = rp,
					      .attachmentCount = 1,
					      .pAttachments = &im->view,
					      .width = w,
					      .height = h,
					      .layers = 1 };
	VkFramebuffer fb;
	VKCHECK(vkCreateFramebuffer(V.dev, &fci, NULL, &fb));
	begin();
	img_barrier(im->img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		    VK_QUEUE_FAMILY_EXTERNAL, V.qf, 0,
		    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
	const VkRenderPassBeginInfo rbi = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
					    .renderPass = rp,
					    .framebuffer = fb,
					    .renderArea = { { 0, 0 }, { w, h } } };
	vkCmdBeginRenderPass(V.cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);
	if (!empty(r)) {
		VkClearAttachment ca = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = 0 };
		memcpy(ca.clearValue.color.float32, colour_f[c], sizeof(colour_f[c]));
		const VkClearRect cr = { .rect = { { (int32_t)r->x0, (int32_t)r->y0 },
						   { r->x1 - r->x0, r->y1 - r->y0 } },
					 .baseArrayLayer = 0,
					 .layerCount = 1 };
		vkCmdClearAttachments(V.cb, 1, &ca, 1, &cr);
	}
	vkCmdEndRenderPass(V.cb);
	img_barrier(im->img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, V.qf,
		    VK_QUEUE_FAMILY_EXTERNAL, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0,
		    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
	submit_wait();
	vkDestroyFramebuffer(V.dev, fb, NULL);
	vkDestroyRenderPass(V.dev, rp, NULL);
}

/* ------------------------------------------------------------------------ */
/* GL (stock EGL/GLES of the host, EGL_EXT_platform_device)                 */

static EGLDisplay E_dpy = EGL_NO_DISPLAY;
static EGLContext E_ctx = EGL_NO_CONTEXT;
static GLuint E_prog;
static PFNEGLCREATEIMAGEKHRPROC p_eglCreateImageKHR;
static PFNEGLDESTROYIMAGEKHRPROC p_eglDestroyImageKHR;
static PFNGLEGLIMAGETARGETTEXTURE2DOESPROC p_glEGLImageTargetTexture2DOES;
static char E_formats[256];

static GLuint compile(GLenum type, const char *src)
{
	GLuint s = glCreateShader(type);
	glShaderSource(s, 1, &src, NULL);
	glCompileShader(s);
	GLint ok = 0;
	glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		char log[512];
		glGetShaderInfoLog(s, sizeof(log), NULL, log);
		printf("  FATAL shader: %s\n", log);
		exit(2);
	}
	return s;
}

static bool gl_setup(dev_t rdev)
{
	PFNEGLQUERYDEVICESEXTPROC qd = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
	PFNEGLQUERYDEVICESTRINGEXTPROC qs =
	    (PFNEGLQUERYDEVICESTRINGEXTPROC)eglGetProcAddress("eglQueryDeviceStringEXT");
	PFNEGLGETPLATFORMDISPLAYEXTPROC gpd =
	    (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
	if (!qd || !qs || !gpd)
		return false;
	EGLDeviceEXT devs[16];
	EGLint n = 0;
	if (!qd(16, devs, &n))
		return false;
	for (EGLint i = 0; i < n && E_dpy == EGL_NO_DISPLAY; i++) {
		const char *node = qs(devs[i], EGL_DRM_RENDER_NODE_FILE_EXT);
		struct stat st;
		if (!node || stat(node, &st) || st.st_rdev != rdev)
			continue;
		E_dpy = gpd(EGL_PLATFORM_DEVICE_EXT, devs[i], NULL);
	}
	EGLint maj, min;
	if (E_dpy == EGL_NO_DISPLAY || !eglInitialize(E_dpy, &maj, &min))
		return false;
	eglBindAPI(EGL_OPENGL_ES_API);
	const EGLint ca[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE };
	E_ctx = eglCreateContext(E_dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ca);
	if (E_ctx == EGL_NO_CONTEXT || !eglMakeCurrent(E_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, E_ctx))
		return false;
	p_eglCreateImageKHR = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
	p_eglDestroyImageKHR = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
	p_glEGLImageTargetTexture2DOES =
	    (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
	PFNEGLQUERYDMABUFFORMATSEXTPROC qf =
	    (PFNEGLQUERYDMABUFFORMATSEXTPROC)eglGetProcAddress("eglQueryDmaBufFormatsEXT");
	EGLint fm[128], nf = 0;
	if (qf && qf(E_dpy, 128, fm, &nf)) {
		size_t o = 0;
		const uint32_t want[] = { DRM_FORMAT_ABGR8888, DRM_FORMAT_RGB565,
					  DRM_FORMAT_ABGR16161616F };
		for (int k = 0; k < 3; k++) {
			bool has = false;
			for (EGLint i = 0; i < nf; i++)
				has |= (uint32_t)fm[i] == want[k];
			o += (size_t)snprintf(E_formats + o, sizeof(E_formats) - o, "%s%.4s:%s",
					      k ? " " : "", (const char *)&want[k], has ? "yes" : "no");
		}
	}
	static const char *vs = "#version 300 es\n"
				"void main() {\n"
				"  vec2 p = vec2(float((gl_VertexID & 1) << 2) - 1.0,\n"
				"                float((gl_VertexID & 2) << 1) - 1.0);\n"
				"  gl_Position = vec4(p, 0.0, 1.0);\n"
				"}\n";
	static const char *fs = "#version 300 es\n"
				"precision highp float;\n"
				"uniform vec4 col;\n"
				"out vec4 o;\n"
				"void main() { o = col; }\n";
	E_prog = glCreateProgram();
	glAttachShader(E_prog, compile(GL_VERTEX_SHADER, vs));
	glAttachShader(E_prog, compile(GL_FRAGMENT_SHADER, fs));
	glLinkProgram(E_prog);
	GLint ok = 0;
	glGetProgramiv(E_prog, GL_LINK_STATUS, &ok);
	return ok && p_eglCreateImageKHR && p_glEGLImageTargetTexture2DOES;
}

struct glimg {
	EGLImageKHR img;
	GLuint tex, fbo;
};

/* import the dma-buf as an EGLImage -> GL_TEXTURE_2D -> FBO colour attachment */
static bool gl_import(int fd, uint32_t w, uint32_t h, const struct fmt *f, uint32_t stride,
		      uint64_t mod, struct glimg *g, char *why, size_t why_len)
{
	const EGLint at[] = { EGL_WIDTH,
			      (EGLint)w,
			      EGL_HEIGHT,
			      (EGLint)h,
			      EGL_LINUX_DRM_FOURCC_EXT,
			      (EGLint)f->drm,
			      EGL_DMA_BUF_PLANE0_FD_EXT,
			      fd,
			      EGL_DMA_BUF_PLANE0_OFFSET_EXT,
			      0,
			      EGL_DMA_BUF_PLANE0_PITCH_EXT,
			      (EGLint)stride,
			      EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
			      (EGLint)(uint32_t)mod,
			      EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
			      (EGLint)(uint32_t)(mod >> 32),
			      EGL_NONE };
	g->img = p_eglCreateImageKHR(E_dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, at);
	if (g->img == EGL_NO_IMAGE_KHR) {
		snprintf(why, why_len, "eglCreateImageKHR(dma-buf): 0x%x", eglGetError());
		return false;
	}
	while (glGetError() != GL_NO_ERROR)
		;
	glGenTextures(1, &g->tex);
	glBindTexture(GL_TEXTURE_2D, g->tex);
	p_glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, g->img);
	GLenum e = glGetError();
	if (e != GL_NO_ERROR) {
		snprintf(why, why_len, "glEGLImageTargetTexture2DOES(GL_TEXTURE_2D): 0x%x", e);
		glDeleteTextures(1, &g->tex);
		p_eglDestroyImageKHR(E_dpy, g->img);
		return false;
	}
	glGenFramebuffers(1, &g->fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, g->fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g->tex, 0);
	const GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (st != GL_FRAMEBUFFER_COMPLETE) {
		snprintf(why, why_len, "FBO incomplete 0x%x", st);
		return false;
	}
	return true;
}

static void gl_unimport(struct glimg *g)
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glDeleteFramebuffers(1, &g->fbo);
	glDeleteTextures(1, &g->tex);
	p_eglDestroyImageKHR(E_dpy, g->img);
}

/* GL rendering: scissored clear of A (red), shader draw into A2 (blue) */
static GLenum gl_render(struct glimg *g, const struct scene *s)
{
	glBindFramebuffer(GL_FRAMEBUFFER, g->fbo);
	glViewport(0, 0, (GLsizei)s->w, (GLsizei)s->h);
	glEnable(GL_SCISSOR_TEST);
	/* GL's window origin is the bottom-left, the buffer's row 0 is the image's
	 * first row = GL's y = 0 for an FBO texture: no flip */
	if (!empty(&s->A)) {
		glScissor((GLint)s->A.x0, (GLint)s->A.y0, (GLsizei)(s->A.x1 - s->A.x0),
			  (GLsizei)(s->A.y1 - s->A.y0));
		glClearColor(colour_f[RED][0], colour_f[RED][1], colour_f[RED][2], colour_f[RED][3]);
		glClear(GL_COLOR_BUFFER_BIT);
	}
	if (!empty(&s->A2)) {
		glScissor((GLint)s->A2.x0, (GLint)s->A2.y0, (GLsizei)(s->A2.x1 - s->A2.x0),
			  (GLsizei)(s->A2.y1 - s->A2.y0));
		glUseProgram(E_prog);
		glUniform4fv(glGetUniformLocation(E_prog, "col"), 1, colour_f[BLUE]);
		glDrawArrays(GL_TRIANGLES, 0, 3);
	}
	glDisable(GL_SCISSOR_TEST);
	glFinish(); /* the producer's completion; a gralloc lock waits the acquire fence */
	return glGetError();
}

/* ------------------------------------------------------------------------ */
/* lock helpers                                                              */

static double g_first_map_ms;

static uint64_t map_check(struct gbm_bo *bo, const struct scene *s, const struct fmt *f,
			  uint32_t pitch, const char *what)
{
	void *addr = MAP_FAILED, *md = NULL;
	ops->map(bo, (int)s->w, (int)s->h, &addr, &md);
	if (addr == MAP_FAILED) {
		FAIL("%s: map failed (errno %d)", what, errno);
		return UINT64_MAX;
	}
	uint32_t fx = 0, fy = 0;
	uint64_t got = 0;
	const uint64_t bad = scene_check(s, f, addr, pitch, &fx, &fy, &got);
	ops->unmap(bo, md);
	EXPECT(!bad, "%s: read through map: %" PRIu64 "/%u pixels wrong", what, bad, s->w * s->h);
	if (bad)
		printf("    first bad (%u,%u): 0x%" PRIx64 " want 0x%" PRIx64 "\n", fx, fy, got,
		       scene_px(s, f, fx, fy));
	return bad;
}

static uint64_t gpu_check(int fd, const struct alloc_args *a, const struct fmt *f,
			  const struct scene *s, const char *what)
{
	struct imported I;
	if (!vk_import(fd, s->w, s->h, f, a->out_stride, a->out_modifier, false, &I)) {
		FAIL("%s: Vulkan import for the check", what);
		return UINT64_MAX;
	}
	struct hostbuf hb;
	hostbuf_create((VkDeviceSize)s->w * s->h * f->bpp,
		       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0,
		       &hb);
	memset(hb.ptr, 0xcd, hb.size);
	gpu_read(I.img, s->w, s->h, &hb);
	uint32_t fx = 0, fy = 0;
	uint64_t got = 0;
	const uint64_t bad = scene_check(s, f, hb.ptr, (size_t)s->w * f->bpp, &fx, &fy, &got);
	EXPECT(!bad, "%s: read on the GPU (independent Vulkan import): %" PRIu64 "/%u pixels wrong",
	       what, bad, s->w * s->h);
	if (bad)
		printf("    first bad (%u,%u): 0x%" PRIx64 " want 0x%" PRIx64 "\n", fx, fy, got,
		       scene_px(s, f, fx, fy));
	hostbuf_destroy(&hb);
	vk_unimport(&I);
	return bad;
}

/* ------------------------------------------------------------------------ */
/* the child: a process that never allocated maps the buffer                  */

static int child_main(int argc, char **argv)
{
	if (argc != 10)
		return 100;
	g_lib = argv[2];
	const int fd = atoi(argv[3]);
	const struct fmt *f = fmt_by_drm((uint32_t)strtoul(argv[4], NULL, 0));
	const uint32_t w = (uint32_t)atoi(argv[5]), h = (uint32_t)atoi(argv[6]),
		       stride = (uint32_t)atoi(argv[7]);
	const uint64_t mod = strtoull(argv[8], NULL, 0) << 32 | strtoull(argv[9], NULL, 0);
	dev_t rdev = 0;
	const int drm_fd = open_nvidia_render_node(&rdev);
	if (!f || drm_fd < 0)
		return 101;
	load_lib(drm_fd);
	struct gbm_bo *bo = ops->import(gdev, fd, w, h, stride, mod, f->drm);
	if (!bo)
		return 102;
	struct scene s;
	scene_init(&s, w, h);
	s.stage = 3;
	void *addr = MAP_FAILED, *md = NULL;
	const double t0 = now_ms();
	ops->map(bo, (int)w, (int)h, &addr, &md);
	const double t1 = now_ms();
	if (addr == MAP_FAILED)
		return 103;
	uint32_t fx = 0, fy = 0;
	uint64_t got = 0;
	const uint64_t bad = scene_check(&s, f, addr, stride, &fx, &fy, &got);
	if (bad)
		printf("    child: %" PRIu64 " pixels wrong, first (%u,%u) 0x%" PRIx64 " want 0x%" PRIx64
		       "\n",
		       bad, fx, fy, got, scene_px(&s, f, fx, fy));
	for (uint32_t y = s.D.y0; y < s.D.y1; y++)
		for (uint32_t x = s.D.x0; x < s.D.x1; x++)
			set_px((uint8_t *)addr + (size_t)y * stride, f, x, pat_px(f, 3, x, y));
	const double t2 = now_ms();
	ops->unmap(bo, md);
	const double t3 = now_ms();
	ops->free(bo);
	ops->dev_destroy(gdev);
	printf("    child pid %d: first map %.2f ms (lazy ICD bring-up + import + copy), unmap %.2f "
	       "ms; VmRSS %ld kB\n",
	       getpid(), t1 - t0, t3 - t2, status_kb("VmRSS:"));
	return bad ? 1 : 0;
}

static bool run_child(int fd, const struct fmt *f, const struct alloc_args *a)
{
	const int fl = fcntl(fd, F_GETFD);
	fcntl(fd, F_SETFD, fl & ~FD_CLOEXEC);
	const pid_t pid = fork();
	if (pid == 0) {
		char sfd[16], sf[16], sw[16], sh[16], ss[16], mh[24], ml[24];
		snprintf(sfd, sizeof(sfd), "%d", fd);
		snprintf(sf, sizeof(sf), "0x%x", f->drm);
		snprintf(sw, sizeof(sw), "%u", a->width);
		snprintf(sh, sizeof(sh), "%u", a->height);
		snprintf(ss, sizeof(ss), "%u", a->out_stride);
		snprintf(mh, sizeof(mh), "0x%x", (uint32_t)(a->out_modifier >> 32));
		snprintf(ml, sizeof(ml), "0x%x", (uint32_t)a->out_modifier);
		execl("/proc/self/exe", "lock-test", "--child", g_lib, sfd, sf, sw, sh, ss, mh, ml,
		      (char *)NULL);
		_exit(127);
	}
	fcntl(fd, F_SETFD, fl);
	int st = 0;
	waitpid(pid, &st, 0);
	return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}

/* ------------------------------------------------------------------------ */
/* one case                                                                  */

static bool g_have_gl;

static void run_case(const struct fmt *f, uint32_t w, uint32_t h)
{
	const bool expect_linear = g_source_linear ||
		(g_source_stress && (f == &F_R8 || f == &F_565));
	printf("[%s %ux%u] alloc as minigbm passes usage 0xb33 (needs_map_stride, !force_linear, "
	       "scanout)\n",
	       f->name, w, h);
	struct alloc_args a;
	if (alloc_sw_render(w, h, f, &a)) {
		FAIL("alloc failed");
		return;
	}
	EXPECT((expect_linear ? a.out_modifier == DRM_FORMAT_MOD_LINEAR :
		       is_nv_block_linear(a.out_modifier)) &&
		       (a.out_map_stride == a.out_stride || (expect_linear && !a.out_map_stride)) &&
		       a.out_stride % f->bpp == 0 && a.out_stride >= w * f->bpp,
	       "%s 0x%016" PRIx64 ", stride %u, map_stride %u (zero inherits stride, whole pixels, "
	       ">= %u)",
	       expect_linear ? "LINEAR" : "block-linear",
	       a.out_modifier, a.out_stride, a.out_map_stride, w * f->bpp);
	/* minigbm defines zero map_stride as inheriting the image stride. */
	const uint32_t pitch = a.out_map_stride ? a.out_map_stride : a.out_stride;
	struct scene s;
	scene_init(&s, w, h);

	/* mapper: import (minigbm passes strides[0]) and CPU-write P1 */
	struct gbm_bo *bo = ops->import(gdev, a.out_fd, w, h, a.out_stride, a.out_modifier, f->drm);
	if (!bo) {
		FAIL("import");
		close(a.out_fd);
		return;
	}
	void *addr = MAP_FAILED, *md = NULL;
	const double t0 = now_ms();
	ops->map(bo, (int)w, (int)h, &addr, &md);
	const double t1 = now_ms();
	if (addr == MAP_FAILED) {
		FAIL("first map failed (errno %d)", errno);
		ops->free(bo);
		close(a.out_fd);
		return;
	}
	if (!g_first_map_ms)
		g_first_map_ms = t1 - t0;
	for (uint32_t y = 0; y < h; y++)
		for (uint32_t x = 0; x < w; x++)
			set_px((uint8_t *)addr + (size_t)y * pitch, f, x, pat_px(f, 1, x, y));
	ops->unmap(bo, md);
	gpu_check(a.out_fd, &a, f, &s, "CPU write (P1)");

	/* Stock GLES cannot render LINEAR dma-bufs; native Vulkan checks those. */
	if (g_have_gl && !expect_linear) {
		struct glimg g;
		char why[128] = "";
		if (gl_import(a.out_fd, w, h, f, a.out_stride, a.out_modifier, &g, why, sizeof(why))) {
			const GLenum e = gl_render(&g, &s);
			gl_unimport(&g);
			EXPECT(e == GL_NO_ERROR, "GL import as GL_TEXTURE_2D render target + clear + "
						 "draw (GL error 0x%x)",
			       e);
			s.stage = 2;
			map_check(bo, &s, f, pitch, "after GL render");
		} else {
			FAIL("GL import of the block-linear buffer as a render target: %s", why);
		}
	}
	if (s.stage < 2) {
		/* keep the scene consistent for the next stages: paint A/A2 with Vulkan */
		struct imported I;
		if (vk_import(a.out_fd, w, h, f, a.out_stride, a.out_modifier, true, &I)) {
			vk_render_rect(&I, f, w, h, &s.A, RED);
			vk_render_rect(&I, f, w, h, &s.A2, BLUE);
			vk_unimport(&I);
			s.stage = 2;
		}
	}

	/* Vulkan renders B */
	{
		struct imported I;
		if (vk_import(a.out_fd, w, h, f, a.out_stride, a.out_modifier, true, &I)) {
			vk_render_rect(&I, f, w, h, &s.B, GREEN);
			vk_unimport(&I);
			s.stage = 3;
			map_check(bo, &s, f, pitch, "after Vulkan render");
		} else {
			FAIL("Vulkan import as a colour attachment");
		}
	}

	/* a process that never allocated: lazy ICD bring-up, read + CPU write D */
	const bool child_ok = run_child(a.out_fd, f, &a);
	EXPECT(child_ok, "separate process (never allocated): map/check/CPU-write/unmap");
	s.stage = 4;
	gpu_check(a.out_fd, &a, f, &s, "after the child's CPU write");
	map_check(bo, &s, f, pitch, "after the child's CPU write");
	if (g_have_gl && f == &F_RGBA8) {
		struct glimg g;
		char why[128] = "";
		if (gl_import(a.out_fd, w, h, f, a.out_stride, a.out_modifier, &g, why, sizeof(why))) {
			uint8_t *px = malloc((size_t)w * h * 4);
			glPixelStorei(GL_PACK_ALIGNMENT, 1);
			glReadPixels(0, 0, (GLsizei)w, (GLsizei)h, GL_RGBA, GL_UNSIGNED_BYTE, px);
			uint32_t fx = 0, fy = 0;
			uint64_t got = 0;
			const uint64_t bad = scene_check(&s, f, px, (size_t)w * 4, &fx, &fy, &got);
			EXPECT(!bad && glGetError() == GL_NO_ERROR,
			       "GL sees every write (glReadPixels): %" PRIu64 " pixels wrong", bad);
			free(px);
			gl_unimport(&g);
		} else {
			FAIL("GL import for glReadPixels: %s", why);
		}
	}
	ops->free(bo);
	close(a.out_fd);
}

/* ------------------------------------------------------------------------ */
/* lock access flags (minigbm patch 0002, get_gbm_map_flags_ops)              */

/* expected image: one uint64 per pixel */
static void exp_fill(uint64_t *e, const struct fmt *f, uint32_t w, uint32_t h, uint32_t seed)
{
	for (uint32_t y = 0; y < h; y++)
		for (uint32_t x = 0; x < w; x++)
			e[(size_t)y * w + x] = pat_px(f, seed, x, y);
}
static void exp_rect(uint64_t *e, uint32_t w, const struct rect *r, uint64_t v)
{
	for (uint32_t y = r->y0; y < r->y1; y++)
		for (uint32_t x = r->x0; x < r->x1; x++)
			e[(size_t)y * w + x] = v;
}
static uint64_t cmp_view(const uint64_t *e, const struct fmt *f, uint32_t w, uint32_t h,
			 const void *base, size_t pitch, uint32_t *fx, uint32_t *fy, uint64_t *got)
{
	uint64_t bad = 0;
	for (uint32_t y = 0; y < h; y++)
		for (uint32_t x = 0; x < w; x++) {
			const uint64_t v = get_px((const uint8_t *)base + (size_t)y * pitch, f, x);
			if (v != e[(size_t)y * w + x] && !bad++) {
				*fx = x;
				*fy = y;
				*got = v;
			}
		}
	return bad;
}
/* wrong pixels of the buffer on the GPU (independent Vulkan import) against e */
static uint64_t gpu_cmp(int fd, const struct alloc_args *a, const struct fmt *f, const uint64_t *e)
{
	const uint32_t w = a->width, h = a->height;
	struct imported I;
	if (!vk_import(fd, w, h, f, a->out_stride, a->out_modifier, false, &I))
		return UINT64_MAX;
	struct hostbuf hb;
	hostbuf_create((VkDeviceSize)w * h * f->bpp,
		       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0,
		       &hb);
	memset(hb.ptr, 0xcd, hb.size);
	gpu_read(I.img, w, h, &hb);
	uint32_t fx = 0, fy = 0;
	uint64_t got = 0;
	const uint64_t bad = cmp_view(e, f, w, h, hb.ptr, (size_t)w * f->bpp, &fx, &fy, &got);
	if (bad)
		printf("    GPU view: first bad (%u,%u): 0x%" PRIx64 " want 0x%" PRIx64 "\n", fx, fy,
		       got, e[(size_t)fy * w + fx]);
	hostbuf_destroy(&hb);
	vk_unimport(&I);
	return bad;
}
/* a GPU write on an independent Vulkan device: rect r cleared to colour c,
 * complete on return (vkQueueSubmit + vkWaitForFences: explicitly ordered
 * after everything before the call and before everything after it) */
static bool gpu_write(int fd, const struct alloc_args *a, const struct fmt *f, const struct rect *r,
		      enum colour c)
{
	struct imported I;
	if (!vk_import(fd, a->width, a->height, f, a->out_stride, a->out_modifier, true, &I))
		return false;
	vk_render_rect(&I, f, a->width, a->height, r, c);
	vk_unimport(&I);
	return true;
}
/* map through the flags ABI (flags 0 = the plain gbm_ops ABI) */
static void *lock_map(struct gbm_bo *bo, const struct alloc_args *a, uint32_t flags, void **md)
{
	void *addr = MAP_FAILED;
	*md = NULL;
	if (flags)
		mfops->map(bo, (int)a->width, (int)a->height, flags, &addr, md);
	else
		ops->map(bo, (int)a->width, (int)a->height, &addr, md);
	return addr;
}
static void lock_unmap(struct gbm_bo *bo, void *md, uint32_t flags)
{
	if (flags)
		mfops->unmap(bo, md, flags);
	else
		ops->unmap(bo, md);
}

static void access_flags_case(const struct fmt *f, uint32_t w, uint32_t h)
{
	printf("[access flags %s %ux%u] 0xb33-style buffer, locks through get_gbm_map_flags_ops\n",
	       f->name, w, h);
	struct alloc_args a;
	if (alloc_sw_render(w, h, f, &a)) {
		FAIL("alloc failed");
		return;
	}
	const uint32_t pitch = a.out_map_stride;
	struct gbm_bo *bo = ops->import(gdev, a.out_fd, w, h, a.out_stride, a.out_modifier, f->drm);
	uint64_t *e = malloc((size_t)w * h * sizeof(uint64_t));
	if (!bo || !e) {
		FAIL("import");
		free(e);
		close(a.out_fd);
		return;
	}
	const struct rect R = { w / 4, h / 4, w / 4 + (w + 1) / 2, h / 4 + (h + 1) / 2 }; /* GPU */
	const struct rect W = { 0, h / 2, (w + 1) / 2, h }; /* CPU, WRITE-only lock */
	void *md, *p;
	uint32_t fx = 0, fy = 0;
	uint64_t got = 0, bad;

	/* 1. READ|WRITE lock: pattern 1 everywhere */
	exp_fill(e, f, w, h, 1);
	p = lock_map(bo, &a, GBM_MAP_FLAG_READ | GBM_MAP_FLAG_WRITE, &md);
	if (p == MAP_FAILED) {
		FAIL("READ|WRITE map failed (errno %d)", errno);
		goto out;
	}
	for (uint32_t y = 0; y < h; y++)
		for (uint32_t x = 0; x < w; x++)
			set_px((uint8_t *)p + (size_t)y * pitch, f, x, e[(size_t)y * w + x]);
	lock_unmap(bo, md, GBM_MAP_FLAG_READ | GBM_MAP_FLAG_WRITE);
	bad = gpu_cmp(a.out_fd, &a, f, e);
	EXPECT(!bad, "READ|WRITE lock: CPU write reaches the buffer (%" PRIu64 " wrong)", bad);

	/* 2. READ lock: sees the content; a GPU write landing after the map and
	 *    before the unmap survives the unmap; CPU writes into the read view
	 *    (out of contract, to prove there is no copy-back at all) vanish */
	p = lock_map(bo, &a, GBM_MAP_FLAG_READ, &md);
	if (p == MAP_FAILED) {
		FAIL("READ map failed (errno %d)", errno);
		goto out;
	}
	bad = cmp_view(e, f, w, h, p, pitch, &fx, &fy, &got);
	EXPECT(!bad, "READ lock: the CPU view shows the buffer (%" PRIu64 " wrong)", bad);
	const bool gw = gpu_write(a.out_fd, &a, f, &R, GREEN);
	EXPECT(gw, "GPU write (independent Vulkan device, fence-waited) between READ map and unmap");
	memset(p, 0x5a, (size_t)pitch * h);
	lock_unmap(bo, md, GBM_MAP_FLAG_READ);
	exp_rect(e, w, &R, colour_px(f, GREEN));
	bad = gpu_cmp(a.out_fd, &a, f, e);
	EXPECT(!bad, "READ lock does not write the buffer: the GPU write made during the lock "
		     "survives its unlock, the CPU scribble is not copied back (%" PRIu64 " wrong)",
	       bad);

	/* 3. WRITE-only lock: the CPU writes rect W only; the rest (incl. the GPU
	 *    write from 2.) is kept (gralloc.h lock(): content outside the area
	 *    is NOT modified) */
	p = lock_map(bo, &a, GBM_MAP_FLAG_WRITE, &md);
	if (p == MAP_FAILED) {
		FAIL("WRITE map failed (errno %d)", errno);
		goto out;
	}
	for (uint32_t y = W.y0; y < W.y1; y++)
		for (uint32_t x = W.x0; x < W.x1; x++)
			set_px((uint8_t *)p + (size_t)y * pitch, f, x, pat_px(f, 7, x, y));
	lock_unmap(bo, md, GBM_MAP_FLAG_WRITE);
	for (uint32_t y = W.y0; y < W.y1; y++)
		for (uint32_t x = W.x0; x < W.x1; x++)
			e[(size_t)y * w + x] = pat_px(f, 7, x, y);
	bad = gpu_cmp(a.out_fd, &a, f, e);
	EXPECT(!bad, "WRITE-only lock: its rect written back, everything else kept (%" PRIu64
		     " wrong)",
	       bad);

	/* 4. a READ mapping widened to WRITE before unmap (a nested write lock,
	 *    drv_bo_widen_map_flags): copied back */
	p = lock_map(bo, &a, GBM_MAP_FLAG_READ, &md);
	if (p == MAP_FAILED) {
		FAIL("READ map (widen) failed (errno %d)", errno);
		goto out;
	}
	for (uint32_t y = R.y0; y < R.y1; y++)
		for (uint32_t x = R.x0; x < R.x1; x++)
			set_px((uint8_t *)p + (size_t)y * pitch, f, x, pat_px(f, 9, x, y));
	lock_unmap(bo, md, GBM_MAP_FLAG_READ | GBM_MAP_FLAG_WRITE);
	for (uint32_t y = R.y0; y < R.y1; y++)
		for (uint32_t x = R.x0; x < R.x1; x++)
			e[(size_t)y * w + x] = pat_px(f, 9, x, y);
	bad = gpu_cmp(a.out_fd, &a, f, e);
	EXPECT(!bad, "READ map widened to READ|WRITE at unmap: CPU writes copied back (%" PRIu64
		     " wrong)",
	       bad);

	/* 5. control: READ|WRITE through the flags ABI and through the plain
	 *    gbm_ops ABI copy back, so the same GPU write during the lock is
	 *    overwritten by the staging copy (the test can see a copy-back) */
	for (int abi = 0; abi < 2; abi++) {
		const uint32_t fl = abi ? 0 : (GBM_MAP_FLAG_READ | GBM_MAP_FLAG_WRITE);
		p = lock_map(bo, &a, fl, &md);
		if (p == MAP_FAILED) {
			FAIL("control map failed (errno %d)", errno);
			goto out;
		}
		const enum colour c = abi ? RED : BLUE;
		const bool ok = gpu_write(a.out_fd, &a, f, &R, c);
		lock_unmap(bo, md, fl);
		/* expected: unchanged (the copy-back restores the lock-time content) */
		bad = ok ? gpu_cmp(a.out_fd, &a, f, e) : UINT64_MAX;
		EXPECT(!bad, "control, %s: the copy-back overwrites a GPU write made during the lock "
			     "(%" PRIu64 " wrong)",
		       abi ? "plain gbm_ops map/unmap (READ|WRITE)" : "READ|WRITE lock", bad);
	}

	/* 6. invalid flags are refused */
	p = lock_map(bo, &a, 4, &md);
	EXPECT(p == MAP_FAILED && errno == EINVAL && !md, "map with flags 0x4 refused (EINVAL)");
out:
	ops->free(bo);
	free(e);
	close(a.out_fd);
}

/* a LINEAR CPU buffer (force_linear) with READ, WRITE and READ|WRITE locks: direct mmap */
static void access_flags_linear(void)
{
	const struct fmt *f = &F_RGBA8;
	const uint32_t w = 333, h = 17;
	printf("[access flags LINEAR %s %ux%u] force_linear buffer, direct mmap\n", f->name, w, h);
	struct alloc_args a;
	memset(&a, 0, sizeof(a));
	a.gbm = gdev;
	a.width = w;
	a.height = h;
	a.drm_format = f->drm;
	a.force_linear = true;
	a.needs_map_stride = true;
	a.out_fd = -1;
	if (ops->alloc(&a) || a.out_modifier != DRM_FORMAT_MOD_LINEAR) {
		FAIL("LINEAR alloc");
		return;
	}
	struct gbm_bo *bo = ops->import(gdev, a.out_fd, w, h, a.out_stride, a.out_modifier, f->drm);
	uint64_t *e = malloc((size_t)w * h * sizeof(uint64_t));
	exp_fill(e, f, w, h, 4);
	void *md;
	void *p = bo ? lock_map(bo, &a, GBM_MAP_FLAG_WRITE, &md) : MAP_FAILED;
	bool ok = p != MAP_FAILED;
	if (ok) {
		for (uint32_t y = 0; y < h; y++)
			for (uint32_t x = 0; x < w; x++)
				set_px((uint8_t *)p + (size_t)y * a.out_stride, f, x, e[(size_t)y * w + x]);
		lock_unmap(bo, md, GBM_MAP_FLAG_WRITE);
		p = lock_map(bo, &a, GBM_MAP_FLAG_READ, &md);
		ok = p != MAP_FAILED;
	}
	uint64_t bad = UINT64_MAX;
	if (ok) {
		uint32_t fx, fy;
		uint64_t got;
		bad = cmp_view(e, f, w, h, p, a.out_stride, &fx, &fy, &got);
		lock_unmap(bo, md, GBM_MAP_FLAG_READ);
	}
	EXPECT(ok && !bad, "LINEAR: WRITE lock then READ lock see the same pixels (%" PRIu64 " wrong)",
	       bad);
	if (bo)
		ops->free(bo);
	free(e);
	close(a.out_fd);
}

/* ------------------------------------------------------------------------ */
/* timings                                                                   */

static void timings(void)
{
	const uint32_t w = 2560, h = 1412;
	const struct fmt *f = &F_RGBA8;
	enum { N = 30 };
	printf("[timing] %s %ux%u (%.1f MiB), %d lock/unlock cycles, median\n", f->name, w, h,
	       w * h * 4 / 1048576.0, N);
	struct alloc_args a;
	if (alloc_sw_render(w, h, f, &a)) {
		FAIL("alloc");
		return;
	}
	struct gbm_bo *bo = ops->import(gdev, a.out_fd, w, h, a.out_stride, a.out_modifier, f->drm);
	double tm[N], tu[N], tr[N], tw[N];
	volatile uint64_t sink = 0;
	void *addr, *md;
	ops->map(bo, (int)w, (int)h, &addr, &md); /* setup: import + staging */
	ops->unmap(bo, md);
	for (int i = 0; i < N; i++) {
		double t0 = now_ms();
		ops->map(bo, (int)w, (int)h, &addr, &md);
		double t1 = now_ms();
		uint64_t acc = 0;
		for (size_t k = 0; k < (size_t)a.out_stride * h / 8; k++)
			acc += ((const uint64_t *)addr)[k];
		sink += acc;
		double t2 = now_ms();
		memset(addr, i, (size_t)a.out_stride * h);
		double t3 = now_ms();
		ops->unmap(bo, md);
		double t4 = now_ms();
		tm[i] = t1 - t0;
		tr[i] = t2 - t1;
		tw[i] = t3 - t2;
		tu[i] = t4 - t3;
	}
	qsort(tm, N, sizeof(double), cmp_double);
	qsort(tu, N, sizeof(double), cmp_double);
	qsort(tr, N, sizeof(double), cmp_double);
	qsort(tw, N, sizeof(double), cmp_double);
	printf("  lock  (GPU copy tiled -> staging + wait): median %.3f ms (min %.3f, max %.3f)\n",
	       tm[N / 2], tm[0], tm[N - 1]);
	printf("  unlock (GPU copy staging -> tiled + wait): median %.3f ms (min %.3f, max %.3f)\n",
	       tu[N / 2], tu[0], tu[N - 1]);
	printf("  CPU read of the staging copy (cached sysmem): median %.3f ms\n", tr[N / 2]);
	printf("  CPU write of the staging copy (memset): median %.3f ms\n", tw[N / 2]);
	printf("  first lock of the process (ICD already up from alloc; import + staging + copy): "
	       "%.3f ms\n",
	       g_first_map_ms);
	ops->free(bo);
	close(a.out_fd);

	/* comparison for a CPU-detiling-through-BAR write path: CPU write and read
	 * bandwidth of a DEVICE_LOCAL|HOST_VISIBLE (ReBAR) mapping of the same size */
	struct hostbuf bar;
	if (hostbuf_create((VkDeviceSize)a.out_stride * h,
			   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
			   0, &bar)) {
		double bw[5], br[5];
		for (int i = 0; i < 5; i++) {
			double t0 = now_ms();
			memset(bar.ptr, i, bar.size);
			double t1 = now_ms();
			uint64_t acc = 0;
			for (size_t k = 0; k < bar.size / 8; k++)
				acc += ((const volatile uint64_t *)bar.ptr)[k];
			sink += acc;
			double t2 = now_ms();
			bw[i] = t1 - t0;
			br[i] = t2 - t1;
		}
		qsort(bw, 5, sizeof(double), cmp_double);
		qsort(br, 5, sizeof(double), cmp_double);
		printf("  for comparison, ReBAR (DEVICE_LOCAL|HOST_VISIBLE) mapping of %.1f MiB: CPU "
		       "write %.3f ms, CPU read %.3f ms (median of 5)\n",
		       bar.size / 1048576.0, bw[2], br[2]);
		hostbuf_destroy(&bar);
	} else {
		printf("  no DEVICE_LOCAL|HOST_VISIBLE memory type (no ReBAR): BAR comparison skipped\n");
	}
	(void)sink;
}

/* ------------------------------------------------------------------------ */
/* leak check                                                                */

static void leaks(void)
{
	const uint32_t w = 1920, h = 1080;
	const struct fmt *f = &F_RGBA8;
	const int cycles = 200, warm = 10;
	printf("[leaks] %d x (alloc 0xb33-style %ux%u -> import -> map -> CPU write -> unmap -> "
	       "free -> close) + %d x lock/unlock of one buffer\n",
	       cycles, w, h, cycles);
	int errors = 0;
	int fds0 = 0;
	long rss0 = 0, gpu0 = 0, av0 = 0;
	for (int pass = 0; pass < 2; pass++) {
		if (pass) {
			fds0 = count_fds();
			rss0 = status_kb("VmRSS:");
			gpu0 = gpu_mem_mib();
			av0 = mem_available_mib();
		}
		for (int i = 0; i < (pass ? cycles : warm); i++) {
			struct alloc_args a;
			if (alloc_sw_render(w, h, f, &a)) {
				errors++;
				continue;
			}
			struct gbm_bo *bo = ops->import(gdev, a.out_fd, w, h, a.out_stride,
							a.out_modifier, f->drm);
			void *addr = MAP_FAILED, *md = NULL;
			if (bo)
				ops->map(bo, (int)w, (int)h, &addr, &md);
			if (addr == MAP_FAILED)
				errors++;
			else {
				((volatile uint32_t *)addr)[i] = 0xdeadbeef;
				ops->unmap(bo, md);
			}
			if (bo)
				ops->free(bo);
			close(a.out_fd);
		}
	}
	/* repeated locks of one buffer: the staging cache is reused, nothing grows */
	struct alloc_args a;
	struct gbm_bo *bo = NULL;
	if (!alloc_sw_render(w, h, f, &a))
		bo = ops->import(gdev, a.out_fd, w, h, a.out_stride, a.out_modifier, f->drm);
	for (int i = 0; bo && i < cycles; i++) {
		void *addr = MAP_FAILED, *md = NULL;
		ops->map(bo, (int)w, (int)h, &addr, &md);
		if (addr == MAP_FAILED)
			errors++;
		else
			ops->unmap(bo, md);
	}
	const long rss_one = status_kb("VmRSS:");
	if (bo) {
		ops->free(bo);
		close(a.out_fd);
	} else {
		errors++;
	}
	const int fds1 = count_fds();
	const long rss1 = status_kb("VmRSS:");
	const long gpu1 = gpu_mem_mib();
	const long av1 = mem_available_mib();
	printf("  open fds %d -> %d; VmRSS %ld -> %ld kB (%ld with one buffer + staging live); GPU "
	       "%ld -> %ld MiB; MemAvailable %ld -> %ld MiB\n",
	       fds0, fds1, rss0, rss1, rss_one, gpu0, gpu1, av0, av1);
	EXPECT(!errors, "%d errors over the cycles", errors);
	EXPECT(fds1 == fds0, "no fd leak (%+d)", fds1 - fds0);
	/* a leaked staging copy per cycle would be 200 x 7.9 MiB */
	EXPECT(rss1 - rss0 < 8 * 1024, "VmRSS stable (%+ld kB; a staging leak would be >= +1.5 GiB)",
	       rss1 - rss0);
	EXPECT(gpu0 >= 0 && gpu1 >= 0 && gpu1 - gpu0 <= 32,
	       "GPU memory stable (%+ld MiB, whole device; a buffer leak would be >= +1.5 GiB)",
	       gpu1 - gpu0);
	EXPECT(av0 - av1 < 512, "MemAvailable stable (%+ld MiB, whole system)", av1 - av0);
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	if (argc > 1 && !strcmp(argv[1], "--child"))
		return child_main(argc, argv);
	if (argc > 3 || (argc > 2 && strcmp(argv[2], "--source-linear") &&
			strcmp(argv[2], "--source-modifiers") && strcmp(argv[2], "--modifiers") &&
			strcmp(argv[2], "--source-stress"))) {
		fprintf(stderr, "usage: %s [LIB] [--source-linear|--source-modifiers|--modifiers|--source-stress]\n", argv[0]);
		return 2;
	}
	g_lib = argc > 1 ? argv[1] : "out/host/libgbm_mesa_wrapper.so";
	if (g_lib[0] != '/') {
		static char abs[PATH_MAX];
		if (realpath(g_lib, abs))
			g_lib = abs;
	}
	dev_t rdev = 0;
	const int drm_fd = open_nvidia_render_node(&rdev);
	if (drm_fd < 0) {
		printf("FATAL: no render node with sysfs vendor 0x10de\n");
		return 2;
	}
	printf("lock-test (%zu-bit), library %s, GPU %s\n", sizeof(void *) * 8, g_lib, gpu_pci);
	load_lib(drm_fd);
	vk_setup(&V, rdev);
	g_source_linear = argc > 2 && !strcmp(argv[2], "--source-linear");
	g_source_stress = argc > 2 && !strcmp(argv[2], "--source-stress");
	const bool source_modifiers = argc > 2 && !strcmp(argv[2], "--source-modifiers");
	const bool modifiers_only = source_modifiers || (argc > 2 && !strcmp(argv[2], "--modifiers"));
	if (g_source_linear || source_modifiers || g_source_stress) {
		VkPhysicalDeviceDriverProperties driver = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
		};
		VkPhysicalDeviceProperties2 props = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &driver,
		};
		vkGetPhysicalDeviceProperties2(V.pd, &props);
		if (props.properties.vendorID != 0x10de || driver.driverID != VK_DRIVER_ID_MESA_NVK) {
			FAIL("source-native checks require NVIDIA Mesa NVK");
			goto cleanup;
		}
	}
	if (g_source_linear) {
		run_case(&F_R8, 257, 129);
		run_case(&F_R8, 1, 1);
		run_case(&F_565, 333, 17);
		run_case(&F_565, 1280, 720);
		goto cleanup;
	}
	g_have_gl = gl_setup(rdev);
	if (!g_have_gl)
		FAIL("stock EGL/GLES on the NVIDIA device (EGL_EXT_platform_device) unavailable");
	else
		printf("GL: %s / %s; EGL dma-buf formats %s\n", glGetString(GL_RENDERER),
		       glGetString(GL_VERSION), E_formats);

	if (modifiers_only) {
		run_case(&F_RGBA8, 1001, 37);
		run_case(&F_RGBA8, 1, 1);
		run_case(&F_FP16, 64, 64);
		run_case(&F_FP16, 257, 129);
		goto cleanup;
	}

	if (g_source_stress) {
		run_case(&F_R8, 257, 129);
		run_case(&F_R8, 1, 1);
	}
	run_case(&F_RGBA8, 2560, 1412);
	run_case(&F_RGBA8, 1001, 37);
	run_case(&F_RGBA8, 1, 1);
	run_case(&F_565, 1280, 720);
	run_case(&F_565, 333, 17);
	run_case(&F_FP16, 257, 129);
	run_case(&F_FP16, 64, 64);
	access_flags_case(&F_RGBA8, 2560, 1412);
	access_flags_case(&F_RGBA8, 1001, 37);
	/* These checks test staging-copy isolation; LINEAR maps are direct. */
	if (!g_source_stress)
		access_flags_case(&F_565, 333, 17);
	access_flags_case(&F_FP16, 64, 64);
	access_flags_linear();
	timings();
	leaks();

cleanup:
	if (g_have_gl) {
		eglMakeCurrent(E_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
		eglDestroyContext(E_dpy, E_ctx);
		eglTerminate(E_dpy);
	}
	ops->dev_destroy(gdev);
	vk_teardown(&V);
	close(drm_fd);
	printf("%s: %d failure(s)\n", failures ? "FAILED" : "ALL PASSED", failures);
	return failures ? 1 : 0;
}
