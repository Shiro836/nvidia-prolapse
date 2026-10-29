/*
 * libgbm_mesa_wrapper.so for NVIDIA native mode.
 *
 * minigbm's gbm_mesa backend (build/minigbm/gbm_mesa_driver, pinned 16844ba)
 * does not allocate by itself: it dlopens "libgbm_mesa_wrapper.so" and calls
 * get_gbm_ops() (gbm_mesa_internals.cpp:231-245). This library is that
 * backend. Buffers are allocated in the calling process through NVIDIA's own
 * Vulkan ICD (libEGL_nvidia.so.0, dlopened by soname and driven through the
 * Khronos loader<->ICD interface; no libvulkan in between) as exportable
 * VkImages with explicit DRM format modifiers, exported as dma-bufs.
 *
 *  - force_linear (CPU-mapped buffers the GPU does not render into; with our
 *    minigbm patch, minigbm/0001-*.patch): DRM_FORMAT_MOD_LINEAR only, mapped
 *    by mmap() of the dma-buf.
 *  - everything else, including CPU-mapped GPU render targets
 *    (needs_map_stride without force_linear, e.g. SurfaceFlinger's screenshot
 *    buffers): NVIDIA block-linear modifiers only. NVIDIA's EGL cannot render
 *    into LINEAR dma-bufs (it imports them external-only) and the compositor
 *    can only display block-linear ones. A CPU lock of a block-linear buffer
 *    goes through a GPU copy to/from a linear staging buffer in system memory
 *    (the design of minigbm's amdgpu backend, amdgpu.c bo_map/bo_unmap),
 *    made with NVIDIA's Vulkan ICD in the locking process.
 *
 * Nothing falls back: an unsupported request fails with the VkResult, the
 * format, the size and the modifier list in the log.
 *
 * The struct gbm_device / struct gbm_bo pointers are opaque to minigbm; we
 * hand out our own structs.
 */

#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/types.h>
#include <unistd.h>

#include <linux/dma-buf.h>

#ifdef __ANDROID__
#include <android/log.h>
#include <drm/drm_fourcc.h>
#include <sys/system_properties.h>
#else
#include <drm_fourcc.h>
#endif

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <vulkan/vk_icd.h>

#include "gbm_mesa_wrapper.h"
#include "gbm_mesa_wrapper_map_flags.h"
#include "gbm_mesa_wrapper_layout.h"
#include "gbm_mesa_wrapper_alloc_flags.h"

#ifndef NVG_ICD_SONAME
#ifdef NVG_SOURCE_RM
#define NVG_ICD_SONAME "libvulkan_nouveau.so"
#else
#define NVG_ICD_SONAME "libEGL_nvidia.so.0"
#endif
#endif
#define NVG_LOG_TAG "gralloc.nvidia"
#define NVG_VENDOR_NVIDIA 0x10de
#define NVG_ICD_INTERFACE_VERSION 5u

#define NVG_EXPORT __attribute__((visibility("default")))
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

#if defined(NVG_SOURCE_RM)
#define NVG_REMEDY_ICD "install the source-built Mesa NVK ICD and its Rust RM backend in the vendor driver namespace"
#elif defined(__ANDROID__) && defined(__LP64__)
#define NVG_REMEDY_ICD                                                                            \
	"NVIDIA native mode must provide the host driver in /vendor_extra/lib64 "                \
	"(dev/native-deploy + the nvidia_native LXC mounts) together with the libnvglibc shims, " \
	"and the host driver version must match"
#elif defined(__ANDROID__)
#define NVG_REMEDY_ICD                                                                            \
	"NVIDIA native mode must provide the 32-bit host driver (lib32-nvidia-utils) in "        \
	"/vendor_extra/lib (dev/native-deploy + the nvidia_native LXC mounts) together with the " \
	"32-bit libnvglibc shims, and the host driver version must match"
#else
#define NVG_REMEDY_ICD "install the NVIDIA proprietary driver (it ships " NVG_ICD_SONAME ")"
#endif
/* measured (615.71.09): with no /dev/nvidia* nodes the ICD already fails the
 * interface negotiation with VK_ERROR_INITIALIZATION_FAILED */
#define NVG_REMEDY_DEV                                                                            \
	"check that /dev/nvidiactl and /dev/nvidia0 exist and are accessible to this process and " \
	"that the user-space driver version equals the kernel module's "                          \
	"(/proc/driver/nvidia/version)"

/* ------------------------------------------------------------------------ */
/* logging                                                                   */

enum nvg_level { NVG_ERR, NVG_INFO, NVG_TRACE };

static pthread_once_t trace_once = PTHREAD_ONCE_INIT;
static bool trace_on;

static void trace_init(void)
{
	const char *e = getenv("NVGRALLOC_TRACE");
	trace_on = e && !strcmp(e, "1");
#ifdef __ANDROID__
	/* the allocator service is started by init, where an env var is awkward;
	 * the property is the same switch */
	char v[PROP_VALUE_MAX] = "";
	if (__system_property_get("debug.nvgralloc.trace", v) > 0 && !strcmp(v, "1"))
		trace_on = true;
#endif
}

static bool nvg_tracing(void)
{
	pthread_once(&trace_once, trace_init);
	return trace_on;
}

__attribute__((format(printf, 2, 3))) static void nvg_log(enum nvg_level lvl, const char *fmt,
							   ...)
{
	if (lvl == NVG_TRACE && !nvg_tracing())
		return;
	va_list ap;
	va_start(ap, fmt);
#ifdef __ANDROID__
	static const int prio[] = { ANDROID_LOG_ERROR, ANDROID_LOG_INFO, ANDROID_LOG_DEBUG };
	__android_log_vprint(prio[lvl], NVG_LOG_TAG, fmt, ap);
#else
	static const char *const tag[] = { "E", "I", "T" };
	char buf[2048];
	vsnprintf(buf, sizeof(buf), fmt, ap);
	fprintf(stderr, NVG_LOG_TAG ": %s: %s\n", tag[lvl], buf);
#endif
	va_end(ap);
}

#define LOGE(...) nvg_log(NVG_ERR, __VA_ARGS__)
#define LOGI(...) nvg_log(NVG_INFO, __VA_ARGS__)
#define LOGT(...) nvg_log(NVG_TRACE, __VA_ARGS__)

static const char *vk_result_str(VkResult r)
{
	switch (r) {
#define R(x)                                                                                       \
	case x:                                                                                    \
		return #x
		R(VK_SUCCESS);
		R(VK_INCOMPLETE);
		R(VK_ERROR_OUT_OF_HOST_MEMORY);
		R(VK_ERROR_OUT_OF_DEVICE_MEMORY);
		R(VK_ERROR_INITIALIZATION_FAILED);
		R(VK_ERROR_DEVICE_LOST);
		R(VK_ERROR_MEMORY_MAP_FAILED);
		R(VK_ERROR_LAYER_NOT_PRESENT);
		R(VK_ERROR_EXTENSION_NOT_PRESENT);
		R(VK_ERROR_FEATURE_NOT_PRESENT);
		R(VK_ERROR_INCOMPATIBLE_DRIVER);
		R(VK_ERROR_TOO_MANY_OBJECTS);
		R(VK_ERROR_FORMAT_NOT_SUPPORTED);
		R(VK_ERROR_INVALID_EXTERNAL_HANDLE);
		R(VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT);
		R(VK_ERROR_UNKNOWN);
#undef R
	default:
		return "VkResult(?)";
	}
}

/* ------------------------------------------------------------------------ */
/* formats: the same whitelist and mapping as the baseline                   */
/* (waydroid-nvidia src/minigbm-vtest/vtest_wrapper.c vtest_get_gbm_format,  */
/*  src/virglrenderer-vtest/vtest_gpu_alloc.c drm_format_to_vk)              */

struct nvg_format {
	uint32_t drm;
	VkFormat vk;
	VkFormat srgb; /* sRGB view twin (VK_FORMAT_UNDEFINED if none) */
	uint32_t bpp;
	const char *name;
};

static const struct nvg_format nvg_formats[] = {
	{ DRM_FORMAT_R8, VK_FORMAT_R8_UNORM, VK_FORMAT_UNDEFINED, 1, "R8" },
	{ DRM_FORMAT_RGB565, VK_FORMAT_R5G6B5_UNORM_PACK16, VK_FORMAT_UNDEFINED, 2, "RGB565" },
	{ DRM_FORMAT_XRGB8888, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB, 4, "XRGB8888" },
	{ DRM_FORMAT_ARGB8888, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB, 4, "ARGB8888" },
	{ DRM_FORMAT_XBGR8888, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB, 4, "XBGR8888" },
	{ DRM_FORMAT_ABGR8888, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB, 4, "ABGR8888" },
	{ DRM_FORMAT_ABGR2101010, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_UNDEFINED, 4,
	  "ABGR2101010" },
	{ DRM_FORMAT_ABGR16161616F, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_UNDEFINED, 8,
	  "ABGR16161616F" },
};

static const struct nvg_format *find_format(uint32_t drm_format)
{
	for (size_t i = 0; i < ARRAY_SIZE(nvg_formats); i++)
		if (nvg_formats[i].drm == drm_format)
			return &nvg_formats[i];
	return NULL;
}

/* ------------------------------------------------------------------------ */
/* the process-wide Vulkan device                                            */

struct nvg_vk {
	void *icd;
	PFN_vkGetInstanceProcAddr gipa;
	VkInstance instance;
	VkPhysicalDevice pd;
	VkDevice device;
	VkPhysicalDeviceMemoryProperties memprops;
	char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];

	PFN_vkDestroyInstance DestroyInstance;
	PFN_vkGetPhysicalDeviceFormatProperties2 GetPhysicalDeviceFormatProperties2;
	PFN_vkGetPhysicalDeviceImageFormatProperties2 GetPhysicalDeviceImageFormatProperties2;
	PFN_vkCreateImage CreateImage;
	PFN_vkDestroyImage DestroyImage;
	PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
	PFN_vkGetImageMemoryRequirements2 GetImageMemoryRequirements2;
	PFN_vkGetImageSubresourceLayout GetImageSubresourceLayout;
	PFN_vkGetImageDrmFormatModifierPropertiesEXT GetImageDrmFormatModifierPropertiesEXT;
	PFN_vkAllocateMemory AllocateMemory;
	PFN_vkFreeMemory FreeMemory;
	PFN_vkBindImageMemory BindImageMemory;
	PFN_vkGetMemoryFdKHR GetMemoryFdKHR;

	/* CPU lock/unlock of block-linear buffers: a GPU copy through a linear
	 * staging buffer (nvg_map_tiled). The queue, the memory type and the
	 * entry points come with the device; the copy context (pool, command
	 * buffer, fence) is made on the first such lock in the process. */
	uint32_t qf;
	VkQueue queue;
	PFN_vkGetMemoryFdPropertiesKHR GetMemoryFdPropertiesKHR;
	PFN_vkCreateBuffer CreateBuffer;
	PFN_vkDestroyBuffer DestroyBuffer;
	PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
	PFN_vkBindBufferMemory BindBufferMemory;
	PFN_vkMapMemory MapMemory;
	PFN_vkUnmapMemory UnmapMemory;
	PFN_vkFlushMappedMemoryRanges FlushMappedMemoryRanges;
	PFN_vkInvalidateMappedMemoryRanges InvalidateMappedMemoryRanges;
	PFN_vkCreateCommandPool CreateCommandPool;
	PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
	PFN_vkResetCommandBuffer ResetCommandBuffer;
	PFN_vkBeginCommandBuffer BeginCommandBuffer;
	PFN_vkEndCommandBuffer EndCommandBuffer;
	PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
	PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer;
	PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage;
	PFN_vkQueueSubmit QueueSubmit;
	PFN_vkCreateFence CreateFence;
	PFN_vkWaitForFences WaitForFences;
	PFN_vkResetFences ResetFences;
};

static struct nvg_vk g_vk;
static pthread_mutex_t g_vk_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_vk_state; /* 0 = not tried, 1 = ready, -1 = failed (sticky) */
static char g_vk_error[1024];

/* exactly these, nothing else (VK_KHR_format_feature_flags2 is only queried,
 * through a physical-device query, which needs support but no enabling).
 * Deliberately neither VK_KHR_external_semaphore_fd nor
 * VK_KHR_external_fence_fd: this device can neither import a SYNC_FD wait nor
 * export a sync_fd, so NVIDIA's queue-timeline defect D1 cannot reach it
 * (hal/AHB-CONTRACT.md §8.2 "Which devices"; README). */
static const char *const nvg_device_exts[] = {
	VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
	VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
	VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
	VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME,
};

__attribute__((format(printf, 1, 2))) static void vk_fail(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(g_vk_error, sizeof(g_vk_error), fmt, ap);
	va_end(ap);
	LOGE("%s", g_vk_error);
}

static bool has_ext(const VkExtensionProperties *exts, uint32_t n, const char *name)
{
	for (uint32_t i = 0; i < n; i++)
		if (!strcmp(exts[i].extensionName, name))
			return true;
	return false;
}

struct nvg_pd_info {
	VkPhysicalDevice pd;
	VkPhysicalDeviceProperties props;
	VkExtensionProperties *exts;
	uint32_t ext_count;
	bool has_drm;
	VkPhysicalDeviceDrmPropertiesEXT drm;
};

/* Called with g_vk_lock held. rdev is the st_rdev of the DRM fd minigbm gave
 * us (0 if it was not a character device); it is used only to choose between
 * several NVIDIA GPUs, by comparing device numbers, never paths. */
static int vk_init_locked(dev_t rdev)
{
	struct nvg_vk *vk = &g_vk;
	struct nvg_pd_info *infos = NULL;
	VkPhysicalDevice *pds = NULL;
	uint32_t pd_count = 0;
	int ret = -ENODEV;

	vk->icd = dlopen(NVG_ICD_SONAME, RTLD_NOW | RTLD_LOCAL);
	if (!vk->icd) {
		vk_fail("cannot dlopen %s: %s. Remedy: %s", NVG_ICD_SONAME, dlerror(),
			NVG_REMEDY_ICD);
		return -ENODEV;
	}
	PFN_vk_icdNegotiateLoaderICDInterfaceVersion negotiate =
	    (PFN_vk_icdNegotiateLoaderICDInterfaceVersion)dlsym(
		vk->icd, "vk_icdNegotiateLoaderICDInterfaceVersion");
	vk->gipa = (PFN_vkGetInstanceProcAddr)dlsym(vk->icd, "vk_icdGetInstanceProcAddr");
	if (!negotiate || !vk->gipa) {
		vk_fail("%s does not export vk_icdNegotiateLoaderICDInterfaceVersion/"
			"vk_icdGetInstanceProcAddr; it is not the NVIDIA Vulkan ICD this backend "
			"was written against. Remedy: %s",
			NVG_ICD_SONAME, NVG_REMEDY_ICD);
		return -ENODEV;
	}
	uint32_t iface = NVG_ICD_INTERFACE_VERSION;
	VkResult r = negotiate(&iface);
	if (r != VK_SUCCESS || iface < NVG_ICD_INTERFACE_VERSION) {
		vk_fail("ICD interface negotiation: %s (%d), version %u (need %u). Remedy: %s",
			vk_result_str(r), r, iface, NVG_ICD_INTERFACE_VERSION, NVG_REMEDY_DEV);
		return -ENODEV;
	}

	PFN_vkCreateInstance CreateInstance =
	    (PFN_vkCreateInstance)vk->gipa(VK_NULL_HANDLE, "vkCreateInstance");
	if (!CreateInstance) {
		vk_fail("ICD returned no vkCreateInstance");
		return -ENODEV;
	}
	const VkApplicationInfo app = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "gralloc.nvidia",
		.pEngineName = "libgbm_mesa_wrapper",
		.apiVersion = VK_API_VERSION_1_1,
	};
	const VkInstanceCreateInfo ici = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &app,
	};
	r = CreateInstance(&ici, NULL, &vk->instance);
	if (r != VK_SUCCESS) {
		vk->instance = VK_NULL_HANDLE;
		vk_fail("vkCreateInstance: %s (%d). Remedy: %s", vk_result_str(r), r,
			NVG_REMEDY_DEV);
		return -ENODEV;
	}

#define INST_FN(var, name)                                                                         \
	PFN_##name var = (PFN_##name)vk->gipa(vk->instance, #name);                               \
	if (!var) {                                                                                \
		vk_fail("ICD returned no %s", #name);                                              \
		goto out;                                                                          \
	}
	INST_FN(EnumeratePhysicalDevices, vkEnumeratePhysicalDevices);
	INST_FN(GetPhysicalDeviceProperties, vkGetPhysicalDeviceProperties);
	INST_FN(GetPhysicalDeviceProperties2, vkGetPhysicalDeviceProperties2);
	INST_FN(EnumerateDeviceExtensionProperties, vkEnumerateDeviceExtensionProperties);
	INST_FN(GetPhysicalDeviceQueueFamilyProperties, vkGetPhysicalDeviceQueueFamilyProperties);
	INST_FN(GetPhysicalDeviceMemoryProperties, vkGetPhysicalDeviceMemoryProperties);
	INST_FN(CreateDevice, vkCreateDevice);
	INST_FN(GetDeviceProcAddr, vkGetDeviceProcAddr);
	INST_FN(DestroyInstance, vkDestroyInstance);
	INST_FN(GetPhysicalDeviceFormatProperties2, vkGetPhysicalDeviceFormatProperties2);
	INST_FN(GetPhysicalDeviceImageFormatProperties2, vkGetPhysicalDeviceImageFormatProperties2);
#undef INST_FN
	vk->DestroyInstance = DestroyInstance;
	vk->GetPhysicalDeviceFormatProperties2 = GetPhysicalDeviceFormatProperties2;
	vk->GetPhysicalDeviceImageFormatProperties2 = GetPhysicalDeviceImageFormatProperties2;

	r = EnumeratePhysicalDevices(vk->instance, &pd_count, NULL);
	if (r != VK_SUCCESS || !pd_count) {
		vk_fail("vkEnumeratePhysicalDevices: %s (%d), %u devices. Remedy: %s",
			vk_result_str(r), r, pd_count, NVG_REMEDY_DEV);
		goto out;
	}
	pds = calloc(pd_count, sizeof(*pds));
	infos = calloc(pd_count, sizeof(*infos));
	if (!pds || !infos) {
		vk_fail("out of memory");
		ret = -ENOMEM;
		goto out;
	}
	r = EnumeratePhysicalDevices(vk->instance, &pd_count, pds);
	if (r != VK_SUCCESS) {
		vk_fail("vkEnumeratePhysicalDevices: %s (%d)", vk_result_str(r), r);
		goto out;
	}

	uint32_t nv_count = 0;
	struct nvg_pd_info *nv[16];
	for (uint32_t i = 0; i < pd_count; i++) {
		struct nvg_pd_info *pi = &infos[i];
		pi->pd = pds[i];
		GetPhysicalDeviceProperties(pi->pd, &pi->props);
		if (pi->props.vendorID != NVG_VENDOR_NVIDIA)
			continue;
		if (EnumerateDeviceExtensionProperties(pi->pd, NULL, &pi->ext_count, NULL) !=
			VK_SUCCESS ||
		    !(pi->exts = calloc(pi->ext_count ? pi->ext_count : 1, sizeof(*pi->exts))) ||
		    EnumerateDeviceExtensionProperties(pi->pd, NULL, &pi->ext_count, pi->exts) !=
			VK_SUCCESS) {
			vk_fail("vkEnumerateDeviceExtensionProperties failed on \"%s\"",
				pi->props.deviceName);
			goto out;
		}
		if (has_ext(pi->exts, pi->ext_count, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME)) {
			pi->drm.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT;
			VkPhysicalDeviceProperties2 p2 = {
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
				.pNext = &pi->drm,
			};
			GetPhysicalDeviceProperties2(pi->pd, &p2);
			pi->has_drm = true;
		}
		if (nv_count < ARRAY_SIZE(nv))
			nv[nv_count++] = pi;
	}
	if (!nv_count) {
		vk_fail("no Vulkan physical device with vendorID 0x%04x among %u. Remedy: %s",
			NVG_VENDOR_NVIDIA, pd_count, NVG_REMEDY_DEV);
		goto out;
	}

	struct nvg_pd_info *pick = NULL;
	for (uint32_t i = 0; i < nv_count; i++) {
		struct nvg_pd_info *pi = nv[i];
		if (rdev && pi->has_drm && pi->drm.hasRender &&
		    (uint64_t)pi->drm.renderMajor == major(rdev) &&
		    (uint64_t)pi->drm.renderMinor == minor(rdev))
			pick = pi;
	}
	if (!pick) {
		if (nv_count == 1) {
			/* minigbm hands us the first render node it globbed (any vendor,
			 * gbm_mesa_internals.cpp:217-227); with one NVIDIA GPU there is
			 * nothing to choose */
			pick = nv[0];
			LOGT("DRM fd %u:%u is not \"%s\" (render %s %" PRId64 ":%" PRId64
			     "); single NVIDIA GPU, using it",
			     rdev ? major(rdev) : 0, rdev ? minor(rdev) : 0, pick->props.deviceName,
			     pick->has_drm ? "node" : "unknown", pick->drm.renderMajor,
			     pick->drm.renderMinor);
		} else {
			vk_fail("%u NVIDIA GPUs and the DRM fd minigbm opened (%u:%u) is none of "
				"their render nodes. Remedy: set the property gralloc.gbm.device to "
				"the render node of the NVIDIA GPU to allocate on",
				nv_count, rdev ? major(rdev) : 0, rdev ? minor(rdev) : 0);
			goto out;
		}
	}
	vk->pd = pick->pd;
	snprintf(vk->name, sizeof(vk->name), "%s", pick->props.deviceName);

	for (size_t i = 0; i < ARRAY_SIZE(nvg_device_exts); i++) {
		if (!has_ext(pick->exts, pick->ext_count, nvg_device_exts[i])) {
			vk_fail("\"%s\" lacks %s. Remedy: this backend needs NVIDIA driver with "
				"dma-buf + DRM format modifier support (515 or newer)",
				vk->name, nvg_device_exts[i]);
			goto out;
		}
	}
	if (!has_ext(pick->exts, pick->ext_count, VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME)) {
		vk_fail("\"%s\" lacks %s (needed for VkDrmFormatModifierPropertiesList2EXT). "
			"Remedy: newer NVIDIA driver",
			vk->name, VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME);
		goto out;
	}

	uint32_t qf_count = 0;
	GetPhysicalDeviceQueueFamilyProperties(vk->pd, &qf_count, NULL);
	VkQueueFamilyProperties qfs[32];
	if (qf_count > ARRAY_SIZE(qfs))
		qf_count = ARRAY_SIZE(qfs);
	GetPhysicalDeviceQueueFamilyProperties(vk->pd, &qf_count, qfs);
	uint32_t qf = UINT32_MAX;
	for (uint32_t i = 0; i < qf_count && qf == UINT32_MAX; i++)
		if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
			qf = i;
	if (qf == UINT32_MAX) {
		vk_fail("\"%s\" has no graphics queue family", vk->name);
		goto out;
	}
	vk->qf = qf;

	const float prio = 1.0f;
	const VkDeviceQueueCreateInfo qci = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
		.queueFamilyIndex = qf,
		.queueCount = 1,
		.pQueuePriorities = &prio,
	};
	const VkDeviceCreateInfo dci = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &qci,
		.enabledExtensionCount = ARRAY_SIZE(nvg_device_exts),
		.ppEnabledExtensionNames = nvg_device_exts,
	};
	r = CreateDevice(vk->pd, &dci, NULL, &vk->device);
	if (r != VK_SUCCESS) {
		vk->device = VK_NULL_HANDLE;
		vk_fail("vkCreateDevice on \"%s\": %s (%d)", vk->name, vk_result_str(r), r);
		goto out;
	}
	GetPhysicalDeviceMemoryProperties(vk->pd, &vk->memprops);

#define DEV_FN(field, name)                                                                        \
	vk->field = (PFN_##name)GetDeviceProcAddr(vk->device, #name);                             \
	if (!vk->field) {                                                                          \
		vk_fail("ICD returned no %s", #name);                                              \
		goto out;                                                                          \
	}
#define DEV_FN2(var, name)                                                                         \
	var = (PFN_##name)GetDeviceProcAddr(vk->device, #name);                                   \
	if (!var) {                                                                                \
		vk_fail("ICD returned no %s", #name);                                              \
		goto out;                                                                          \
	}
	DEV_FN(CreateImage, vkCreateImage);
	DEV_FN(DestroyImage, vkDestroyImage);
	DEV_FN(GetImageMemoryRequirements, vkGetImageMemoryRequirements);
	DEV_FN(GetImageMemoryRequirements2, vkGetImageMemoryRequirements2);
	DEV_FN(GetImageSubresourceLayout, vkGetImageSubresourceLayout);
	DEV_FN(GetImageDrmFormatModifierPropertiesEXT, vkGetImageDrmFormatModifierPropertiesEXT);
	DEV_FN(AllocateMemory, vkAllocateMemory);
	DEV_FN(FreeMemory, vkFreeMemory);
	DEV_FN(BindImageMemory, vkBindImageMemory);
	DEV_FN(GetMemoryFdKHR, vkGetMemoryFdKHR);
	PFN_vkGetDeviceQueue GetDeviceQueue;
	DEV_FN2(GetDeviceQueue, vkGetDeviceQueue);
	DEV_FN(GetMemoryFdPropertiesKHR, vkGetMemoryFdPropertiesKHR);
	DEV_FN(CreateBuffer, vkCreateBuffer);
	DEV_FN(DestroyBuffer, vkDestroyBuffer);
	DEV_FN(GetBufferMemoryRequirements, vkGetBufferMemoryRequirements);
	DEV_FN(BindBufferMemory, vkBindBufferMemory);
	DEV_FN(MapMemory, vkMapMemory);
	DEV_FN(UnmapMemory, vkUnmapMemory);
	DEV_FN(FlushMappedMemoryRanges, vkFlushMappedMemoryRanges);
	DEV_FN(InvalidateMappedMemoryRanges, vkInvalidateMappedMemoryRanges);
	DEV_FN(CreateCommandPool, vkCreateCommandPool);
	DEV_FN(AllocateCommandBuffers, vkAllocateCommandBuffers);
	DEV_FN(ResetCommandBuffer, vkResetCommandBuffer);
	DEV_FN(BeginCommandBuffer, vkBeginCommandBuffer);
	DEV_FN(EndCommandBuffer, vkEndCommandBuffer);
	DEV_FN(CmdPipelineBarrier, vkCmdPipelineBarrier);
	DEV_FN(CmdCopyImageToBuffer, vkCmdCopyImageToBuffer);
	DEV_FN(CmdCopyBufferToImage, vkCmdCopyBufferToImage);
	DEV_FN(QueueSubmit, vkQueueSubmit);
	DEV_FN(CreateFence, vkCreateFence);
	DEV_FN(WaitForFences, vkWaitForFences);
	DEV_FN(ResetFences, vkResetFences);
#undef DEV_FN
#undef DEV_FN2
	GetDeviceQueue(vk->device, vk->qf, 0, &vk->queue);

	LOGI("Vulkan device (allocation + block-linear locks) on \"%s\" (vendor 0x%04x device 0x%04x, driver %u.%u.%02u, render %" PRId64
	     ":%" PRId64 ") through %s",
	     vk->name, pick->props.vendorID, pick->props.deviceID,
	     /* NVIDIA packs driverVersion as major:10 minor:8 patch:8 build:6 */
	     (pick->props.driverVersion >> 22) & 0x3ff,
	     (pick->props.driverVersion >> 14) & 0xff, (pick->props.driverVersion >> 6) & 0xff,
	     pick->drm.renderMajor, pick->drm.renderMinor, NVG_ICD_SONAME);
	ret = 0;

out:
	if (infos)
		for (uint32_t i = 0; i < pd_count; i++)
			free(infos[i].exts);
	free(infos);
	free(pds);
	if (ret) {
		/* the ICD stays loaded (dlclose of a driver with live threads is not
		 * safe); the failure is sticky and re-reported on every alloc */
		if (vk->device == VK_NULL_HANDLE && vk->instance != VK_NULL_HANDLE &&
		    vk->DestroyInstance) {
			vk->DestroyInstance(vk->instance, NULL);
			vk->instance = VK_NULL_HANDLE;
		}
	}
	return ret;
}

static struct nvg_vk *vk_get(dev_t rdev)
{
	pthread_mutex_lock(&g_vk_lock);
	if (g_vk_state == 0)
		g_vk_state = vk_init_locked(rdev) ? -1 : 1;
	const int state = g_vk_state;
	pthread_mutex_unlock(&g_vk_lock);
	if (state < 0) {
		LOGE("NVIDIA Vulkan allocator unavailable: %s", g_vk_error);
		return NULL;
	}
	return &g_vk;
}

/* ------------------------------------------------------------------------ */
/* gbm_ops                                                                   */

struct nvg_device {
	dev_t rdev;
};

struct nvg_bo {
	int fd;
	uint64_t size;
	uint64_t modifier;
	uint32_t width, height, stride, drm_format;
	dev_t rdev; /* of the importing gbm device: GPU choice for a lazy ICD setup */

	/* block-linear only, made by the first CPU lock in this process and kept
	 * until free (the per-buffer staging cache; guarded by g_xfer_lock) */
	VkImage img;          /* this dma-buf imported with its explicit modifier */
	VkDeviceMemory img_mem;
	VkBuffer stg;         /* linear staging copy, rows of `stride` bytes */
	VkDeviceMemory stg_mem;
	void *stg_ptr;        /* persistently mapped */
	VkDeviceSize stg_size;
	bool stg_coherent;
	bool locked;          /* between map and unmap */
	unsigned linear_maps; /* live mmap()s of a LINEAR dma-buf (atomic) */
};

struct nvg_map {
	void *addr;
	size_t size;
	bool tiled;     /* addr is the staging copy of a block-linear buffer */
	uint32_t flags; /* GBM_MAP_FLAG_READ/WRITE the lock asked for at map */
};

#define NVG_MAP_RW (GBM_MAP_FLAG_READ | GBM_MAP_FLAG_WRITE)

static uint32_t nvg_get_gbm_format(uint32_t drm_format)
{
	/* minigbm contract (gbm_mesa_internals.cpp:326): non-zero = this backend
	 * allocates the format; 0 = minigbm allocates it as a linear R8 blob */
	return find_format(drm_format) ? drm_format : 0;
}

static struct gbm_device *nvg_dev_create(int fd)
{
	/*
	 * minigbm calls dev_create in every process that touches a buffer: the
	 * allocator service, but also each mapper (app) process on import/map of
	 * a CPU buffer (gbm_mesa_internals.cpp:413, 453). The Vulkan device is
	 * created lazily, on the first alloc or the first CPU lock of a
	 * block-linear buffer, not here: a mapper process that only locks LINEAR
	 * buffers never loads the NVIDIA driver on our behalf.
	 */
	struct nvg_device *dev = calloc(1, sizeof(*dev));
	if (!dev) {
		LOGE("dev_create: out of memory");
		return NULL;
	}
	struct stat st;
	if (fd >= 0 && fstat(fd, &st) == 0 && S_ISCHR(st.st_mode))
		dev->rdev = st.st_rdev;
	LOGT("dev_create(fd=%d rdev=%u:%u) -> %p", fd, dev->rdev ? major(dev->rdev) : 0,
	     dev->rdev ? minor(dev->rdev) : 0, (void *)dev);
	return (struct gbm_device *)dev;
}

static void nvg_dev_destroy(struct gbm_device *gbm)
{
	/* The Vulkan device is process-wide and lives until exit; this library is
	 * linked -z nodelete so minigbm's dlclose cannot drop it under us. */
	LOGT("dev_destroy(%p)", (void *)gbm);
	free(gbm);
}

static void fmt_modifiers(char *buf, size_t len, const uint64_t *mods, uint32_t n)
{
	size_t o = 0;
	buf[0] = 0;
	for (uint32_t i = 0; i < n && o + 24 < len; i++)
		o += (size_t)snprintf(buf + o, len - o, "%s0x%016" PRIx64, i ? "," : "", mods[i]);
	if (!n)
		snprintf(buf, len, "(none)");
}

#define NVG_MAX_MODS 64

static bool modifier_fits_height(uint64_t modifier, uint32_t height)
{
	/* NVIDIA Fermi/Turing GOBs have eight rows. NIL clamps the block height
	 * to ceil(log2(ceil(height / 8))) when laying out a small image. Keep the
	 * advertised modifier consistent with that backing: external consumers
	 * may validate its full block footprint before importing the allocation.
	 * LINEAR and other modifier generations retain their existing policy. */
	const uint32_t gob_generation = (uint32_t)((modifier >> 20) & 3);
	if ((modifier >> 56) != 3 || !(modifier & 0x10) ||
	    (gob_generation != 0 && gob_generation != 2))
		return true;
	uint32_t gob_rows = height / 8 + (height % 8 != 0);
	uint32_t max_log2 = 0;
	while (gob_rows > 1) {
		gob_rows = gob_rows / 2 + (gob_rows % 2 != 0);
		max_log2++;
	}
	return (modifier & 0xf) <= max_log2;
}

static int nvg_alloc_impl(struct alloc_args *args, uint32_t alloc_flags)
{
	if (!args || !args->gbm) {
		LOGE("alloc: no args/device");
		return -EINVAL;
	}
	struct nvg_device *dev = (struct nvg_device *)args->gbm;
	const struct nvg_format *f = find_format(args->drm_format);
	/*
	 * Layout rule (README "Layout rule"). minigbm (with our patch,
	 * minigbm/0001-*.patch) sets needs_map_stride for every CPU-mapped buffer
	 * and force_linear only when the GPU does not render into it (and for its
	 * R8 carrier of spoofed formats). LINEAR exactly when force_linear; a
	 * CPU-mapped render target is block-linear and locked through a staging
	 * copy (nvg_map_tiled). Unpatched minigbm sets both flags together, which
	 * gives the old behaviour (every CPU buffer LINEAR).
	 */
	bool linear = args->force_linear;
	const bool mappable = args->force_linear || args->needs_map_stride;
	const char *kind = linear ? "CPU/LINEAR" : mappable ? "GPU+CPU/block-linear" : "GPU/block-linear";
	const uint32_t w = args->width, h = args->height;

	args->out_fd = -1;
	if (!f) {
		LOGE("alloc %ux%u: DRM format 0x%08x is not in the whitelist; minigbm only "
		     "passes formats get_gbm_format accepted",
		     w, h, args->drm_format);
		return -EINVAL;
	}
	if (!w || !h) {
		LOGE("alloc %ux%u %s: empty extent", w, h, f->name);
		return -EINVAL;
	}
	struct nvg_vk *vk = vk_get(dev->rdev);
	if (!vk)
		return -ENODEV;

	/* --- modifier candidates ------------------------------------------- */
	VkDrmFormatModifierPropertiesList2EXT ml = {
		.sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT,
	};
	VkFormatProperties2 fp = { .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &ml };
	vk->GetPhysicalDeviceFormatProperties2(vk->pd, f->vk, &fp);
	VkDrmFormatModifierProperties2EXT mprops[NVG_MAX_MODS];
	if (ml.drmFormatModifierCount > NVG_MAX_MODS)
		ml.drmFormatModifierCount = NVG_MAX_MODS;
	ml.pDrmFormatModifierProperties = mprops;
	vk->GetPhysicalDeviceFormatProperties2(vk->pd, f->vk, &fp);

	/*
	 * GPU buffers must be renderable. NVIDIA's LINEAR modifier does not list
	 * COLOR_ATTACHMENT (615.71.09: it lists LINEAR_COLOR_ATTACHMENT_NV, which
	 * belongs to VK_NV_linear_color_attachment, an importer's business), so
	 * for CPU buffers COLOR_ATTACHMENT is requested only where the modifier
	 * features allow it, like STORAGE. The usage stays within what every
	 * candidate modifier supports; nothing is requested out of spec.
	 */
	const VkFormatFeatureFlags2 need =
	    VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT |
	    VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT |
	    (linear ? 0 : VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT);
	uint64_t all_mods[NVG_MAX_MODS] = { 0 };
	uint64_t mods[NVG_MAX_MODS];
	VkFormatFeatureFlags2 feat[NVG_MAX_MODS];
	uint32_t n = 0;
	bool storage = true, color = true;
	for (uint32_t i = 0; i < ml.drmFormatModifierCount; i++) {
		const VkDrmFormatModifierProperties2EXT *p = &mprops[i];
		all_mods[i] = p->drmFormatModifier;
		const bool is_linear = p->drmFormatModifier == DRM_FORMAT_MOD_LINEAR;
		if (p->drmFormatModifier == DRM_FORMAT_MOD_INVALID || is_linear != linear ||
		    p->drmFormatModifierPlaneCount != 1 ||
		    (p->drmFormatModifierTilingFeatures & need) != need)
			continue;
		if (!modifier_fits_height(p->drmFormatModifier, h)) {
			LOGT("alloc %ux%u %s: modifier 0x%016" PRIx64
			     " exceeds the image's block height", w, h, f->name,
			     p->drmFormatModifier);
			continue;
		}
		feat[n] = p->drmFormatModifierTilingFeatures;
		storage = storage && (feat[n] & VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT);
		color = color && (feat[n] & VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT);
		mods[n++] = p->drmFormatModifier;
	}
#if defined(NVG_SOURCE_RM)
	/* NVK can render LINEAR images when the queried format features say so.
	 * Prefer a supported block-linear layout; formats without a defined DRM
	 * GOB modifier (Blackwell R8/RGB565) use explicit LINEAR only when it
	 * meets the same rendering requirements. Proprietary mode keeps its
	 * strict block-linear rule because its GLES consumer cannot render LINEAR. */
	if (!n && !linear) {
		for (uint32_t i = 0; i < ml.drmFormatModifierCount; i++) {
			const VkDrmFormatModifierProperties2EXT *p = &mprops[i];
			if (p->drmFormatModifier != DRM_FORMAT_MOD_LINEAR ||
			    p->drmFormatModifierPlaneCount != 1 ||
			    (p->drmFormatModifierTilingFeatures & need) != need)
				continue;
			mods[0] = DRM_FORMAT_MOD_LINEAR;
			feat[0] = p->drmFormatModifierTilingFeatures;
			storage = feat[0] & VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT;
			color = true;
			n = 1;
			linear = true;
			kind = "GPU/LINEAR (queried attachment support)";
			break;
		}
	}
#endif
	char modstr[NVG_MAX_MODS * 20 + 8];
	if (!n) {
		fmt_modifiers(modstr, sizeof(modstr), all_mods, ml.drmFormatModifierCount);
		LOGE("alloc %ux%u %s %s: the device lists no single-plane %s modifier with "
		     "%sSAMPLED|TRANSFER_SRC|TRANSFER_DST (device modifiers: %s); "
		     "not substituting another layout",
		     w, h, f->name, kind, linear ? "LINEAR" : "non-LINEAR",
		     linear ? "" : "COLOR_ATTACHMENT|", modstr);
		return -EINVAL;
	}

	const VkImageUsageFlags usage =
	    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
	    VK_IMAGE_USAGE_TRANSFER_DST_BIT | (color ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0) |
	    (storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
	/* importers (ANGLE, the HAL) create sRGB views of RGBA8/BGRA8 buffers:
	 * allocate mutable with the UNORM/sRGB pair so that use is in-spec */
	const VkFormat view_formats[2] = { f->vk, f->srgb };
	const bool mutable_fmt = f->srgb != VK_FORMAT_UNDEFINED;
	const VkImageFormatListCreateInfoKHR fmt_list = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO_KHR,
		.viewFormatCount = 2,
		.pViewFormats = view_formats,
	};
	const VkImageCreateFlags flags = mutable_fmt ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;

	/* keep only modifiers the image query accepts for this exact image:
	 * exportable as dma-buf, extent within limits */
	uint32_t m = 0;
	bool dedicated_only = false;
	for (uint32_t i = 0; i < n; i++) {
		VkPhysicalDeviceExternalImageFormatInfo ext_info = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
			.pNext = mutable_fmt ? &fmt_list : NULL,
			.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
		};
		VkPhysicalDeviceImageDrmFormatModifierInfoEXT mod_info = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
			.pNext = &ext_info,
			.drmFormatModifier = mods[i],
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};
		const VkPhysicalDeviceImageFormatInfo2 ifi = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
			.pNext = &mod_info,
			.format = f->vk,
			.type = VK_IMAGE_TYPE_2D,
			.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
			.usage = usage,
			.flags = flags,
		};
		VkExternalImageFormatProperties ext_props = {
			.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
		};
		VkImageFormatProperties2 ifp = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
			.pNext = &ext_props,
		};
		VkResult r = vk->GetPhysicalDeviceImageFormatProperties2(vk->pd, &ifi, &ifp);
		const VkExternalMemoryFeatureFlags emf =
		    ext_props.externalMemoryProperties.externalMemoryFeatures;
		const VkExtent3D *mx = &ifp.imageFormatProperties.maxExtent;
		if (r != VK_SUCCESS || !(emf & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) ||
		    w > mx->width || h > mx->height) {
			LOGT("alloc %ux%u %s: modifier 0x%016" PRIx64 " rejected by image query: "
			     "%s, ext features 0x%x, max %ux%u",
			     w, h, f->name, mods[i], vk_result_str(r), emf, mx->width, mx->height);
			continue;
		}
		if (emf & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT)
			dedicated_only = true;
		mods[m++] = mods[i];
	}
	fmt_modifiers(modstr, sizeof(modstr), mods, m);
	if (!m) {
		fmt_modifiers(modstr, sizeof(modstr), mods, n);
		LOGE("alloc %ux%u %s %s: no candidate modifier (%s) is exportable as dma-buf at "
		     "this size/usage (usage 0x%x flags 0x%x)",
		     w, h, f->name, kind, modstr, usage, flags);
		return -EINVAL;
	}

	/* --- image ----------------------------------------------------------- */
	VkImage image = VK_NULL_HANDLE;
	VkDeviceMemory mem = VK_NULL_HANDLE;
	int fd = -1;
	int ret = -EINVAL;
	VkResult r;

	const VkExternalMemoryImageCreateInfo emi = {
		.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
		.pNext = mutable_fmt ? &fmt_list : NULL,
		.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
	};
	const VkImageDrmFormatModifierListCreateInfoEXT mli = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
		.pNext = &emi,
		.drmFormatModifierCount = m,
		.pDrmFormatModifiers = mods,
	};
	const VkImageCreateInfo ici = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.pNext = &mli,
		.flags = flags,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = f->vk,
		.extent = { w, h, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	r = vk->CreateImage(vk->device, &ici, NULL, &image);
	if (r != VK_SUCCESS) {
		image = VK_NULL_HANDLE;
		LOGE("alloc %ux%u %s %s: vkCreateImage: %s (%d), modifiers %s", w, h, f->name, kind,
		     vk_result_str(r), r, modstr);
		ret = r == VK_ERROR_OUT_OF_HOST_MEMORY || r == VK_ERROR_OUT_OF_DEVICE_MEMORY
			  ? -ENOMEM
			  : -EINVAL;
		goto out;
	}

	VkMemoryRequirements reqs;
	vk->GetImageMemoryRequirements(vk->device, image, &reqs);

	/* --- memory type ------------------------------------------------------ */
	/*
	 * A LINEAR buffer the CPU reads often (GBM_ALLOC_FLAG_CPU_READ_OFTEN) needs
	 * CPU-cached system memory: the dma-buf mmap keeps the allocation's caching
	 * (open-gpu-kernel-modules nv-dmabuf.c nv_dma_buf_mmap: nv_encode_caching with
	 * the handle's cache_type), and CPU reads of the uncached type run at ~90 MB/s
	 * against several GB/s cached (README "Memory placement"). It must be coherent:
	 * the GPU writes it (decoder copy) and mappers have no cache maintenance call.
	 * No such type is a loud failure, never an uncached or BAR placement.
	 */
	const bool cpu_read_often = linear && (alloc_flags & GBM_ALLOC_FLAG_CPU_READ_OFTEN);
	const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
					   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
					   (cpu_read_often ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0);
	uint32_t type = UINT32_MAX;
	for (int pass = linear ? 0 : 1; pass < (cpu_read_often ? 1 : 2) && type == UINT32_MAX;
	     pass++) {
		const VkMemoryPropertyFlags want = pass == 0 ? host
							      : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		for (uint32_t i = 0; i < vk->memprops.memoryTypeCount; i++) {
			if ((reqs.memoryTypeBits & (1u << i)) &&
			    (vk->memprops.memoryTypes[i].propertyFlags & want) == want) {
				type = i;
				break;
			}
		}
		if (cpu_read_often && type == UINT32_MAX)
			LOGE("alloc %ux%u %s: memoryTypeBits 0x%x has no HOST_VISIBLE|HOST_COHERENT|"
			     "HOST_CACHED type for a buffer the CPU reads often; refusing an uncached "
			     "placement",
			     w, h, f->name, reqs.memoryTypeBits);
		else if (linear && pass == 0 && type == UINT32_MAX)
			LOGT("alloc %ux%u %s: memoryTypeBits 0x%x has no HOST_VISIBLE|HOST_COHERENT "
			     "type; CPU buffer goes to DEVICE_LOCAL (mmap through BAR1)",
			     w, h, f->name, reqs.memoryTypeBits);
	}
	if (type == UINT32_MAX) {
		LOGE("alloc %ux%u %s %s: memoryTypeBits 0x%x has no %s type", w, h, f->name, kind,
		     reqs.memoryTypeBits, linear ? "HOST_VISIBLE|HOST_COHERENT or DEVICE_LOCAL"
						 : "DEVICE_LOCAL");
		goto out;
	}

	/* --- memory: always dedicated (block-linear reports DEDICATED_ONLY) --- */
	const VkExportMemoryAllocateInfo export_info = {
		.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
		.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
	};
	const VkMemoryDedicatedAllocateInfo dedicated = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
		.pNext = &export_info,
		.image = image,
	};
	const VkMemoryAllocateInfo mai = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
		.pNext = &dedicated,
		.allocationSize = reqs.size,
		.memoryTypeIndex = type,
	};
	r = vk->AllocateMemory(vk->device, &mai, NULL, &mem);
	if (r != VK_SUCCESS) {
		mem = VK_NULL_HANDLE;
		LOGE("alloc %ux%u %s %s: vkAllocateMemory(%" PRIu64 " bytes, type %u): %s (%d), "
		     "modifiers %s",
		     w, h, f->name, kind, (uint64_t)reqs.size, type, vk_result_str(r), r, modstr);
		ret = -ENOMEM;
		goto out;
	}
	r = vk->BindImageMemory(vk->device, image, mem, 0);
	if (r != VK_SUCCESS) {
		LOGE("alloc %ux%u %s %s: vkBindImageMemory: %s (%d)", w, h, f->name, kind,
		     vk_result_str(r), r);
		goto out;
	}

	VkImageDrmFormatModifierPropertiesEXT chosen = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT,
	};
	r = vk->GetImageDrmFormatModifierPropertiesEXT(vk->device, image, &chosen);
	if (r != VK_SUCCESS) {
		LOGE("alloc %ux%u %s %s: vkGetImageDrmFormatModifierPropertiesEXT: %s (%d)", w, h,
		     f->name, kind, vk_result_str(r), r);
		goto out;
	}
	const VkImageSubresource sub = { .aspectMask = VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT };
	VkSubresourceLayout layout;
	vk->GetImageSubresourceLayout(vk->device, image, &sub, &layout);

	/* minigbm describes the buffer as one plane at offset 0 with a 32-bit
	 * byte stride (drv_bo_from_format); a CPU stride must be whole pixels
	 * (drv_bo_get_pixel_stride divides by bpp). For a mappable block-linear
	 * buffer the rowPitch is also the pitch of the linear staging copy that
	 * a lock returns, so it must be a valid linear pitch too. */
	if (layout.offset != 0 || layout.rowPitch == 0 || layout.rowPitch > UINT32_MAX ||
	    (mappable && (layout.rowPitch % f->bpp || layout.rowPitch < (uint64_t)w * f->bpp))) {
		LOGE("alloc %ux%u %s %s: unusable plane-0 layout offset %" PRIu64 " rowPitch %" PRIu64
		     " size %" PRIu64 " (modifier 0x%016" PRIx64 ")",
		     w, h, f->name, kind, (uint64_t)layout.offset, (uint64_t)layout.rowPitch,
		     (uint64_t)layout.size, chosen.drmFormatModifier);
		goto out;
	}

	const VkMemoryGetFdInfoKHR gfi = {
		.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
		.memory = mem,
		.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
	};
	r = vk->GetMemoryFdKHR(vk->device, &gfi, &fd);
	if (r != VK_SUCCESS || fd < 0) {
		LOGE("alloc %ux%u %s %s: vkGetMemoryFdKHR: %s (%d) fd %d", w, h, f->name, kind,
		     vk_result_str(r), r, fd);
		fd = -1;
		goto out;
	}
	/* vkGetMemoryFdKHR fds are not guaranteed O_CLOEXEC */
	{
		const int fl = fcntl(fd, F_GETFD);
		if (fl >= 0 && !(fl & FD_CLOEXEC))
			fcntl(fd, F_SETFD, fl | FD_CLOEXEC);
	}

	args->out_fd = fd;
	args->out_stride = (uint32_t)layout.rowPitch;
	args->out_modifier = chosen.drmFormatModifier;
	/*
	 * minigbm consults map_stride only for BO_USE_SW_MASK buffers and falls
	 * back to strides[0] when it is 0 (drv.c:734-738); it becomes the
	 * handle's pixel_stride, the stride every CPU user of a lock assumes.
	 * - LINEAR: 0 = "same as out_stride". Our mmap is direct, so the map
	 *   stride is rowPitch, already strides[0] (gbm_mesa_internals.cpp:379).
	 *   A non-zero value would be wrong on minigbm's R8-blob path, where
	 *   strides[0] is minigbm's own layout of the real format.
	 * - block-linear + mappable: the staging copy a lock returns has rows of
	 *   exactly rowPitch bytes (nvg_map_tiled uses the importer's stride,
	 *   i.e. strides[0] = rowPitch), so map_stride = rowPitch. Mapper
	 *   processes never see map_stride (gbm_mesa_bo_import leaves it 0, so
	 *   pixel_stride, computed once in the allocator, and strides[0] are
	 *   what they use): both equal the staging pitch by construction.
	 */
	args->out_map_stride = (!linear && mappable) ? (uint32_t)layout.rowPitch : 0;
	fd = -1;
	ret = 0;

	LOGT("alloc %ux%u %s %s scanout=%d flags 0x%x -> fd %d modifier 0x%016" PRIx64 " stride %u size %" PRIu64
	     " memtype %u (0x%x) usage 0x%x%s, from [%s]",
	     w, h, f->name, kind, args->use_scanout, alloc_flags, args->out_fd, args->out_modifier,
	     args->out_stride, (uint64_t)reqs.size, type,
	     vk->memprops.memoryTypes[type].propertyFlags, usage,
	     dedicated_only ? " dedicated-only" : "",
	     modstr);

out:
	/* the exported dma-buf holds its own reference to the allocation; the
	 * Vulkan handles are transient */
	if (fd >= 0)
		close(fd);
	if (image != VK_NULL_HANDLE)
		vk->DestroyImage(vk->device, image, NULL);
	if (mem != VK_NULL_HANDLE)
		vk->FreeMemory(vk->device, mem, NULL);
	return ret;
}

static struct gbm_bo *nvg_import(struct gbm_device *gbm, int buf_fd, uint32_t width,
				 uint32_t height, uint32_t stride, uint64_t modifier,
				 uint32_t drm_format)
{
	const off_t size = lseek(buf_fd, 0, SEEK_END);
	if (size <= 0) {
		LOGE("import fd %d (%ux%u fmt 0x%08x): lseek(SEEK_END) = %lld (%s); not a dma-buf?",
		     buf_fd, width, height, drm_format, (long long)size, strerror(errno));
		return NULL;
	}
	if ((uint64_t)size > SIZE_MAX) {
		/* ILP32 mapper (32-bit mediaserver / apps): a buffer this process cannot address */
		LOGE("import fd %d (%ux%u fmt 0x%08x): %lld bytes exceeds this process's address space",
		     buf_fd, width, height, drm_format, (long long)size);
		errno = EFBIG;
		return NULL;
	}
	struct nvg_bo *bo = calloc(1, sizeof(*bo));
	if (!bo) {
		LOGE("import: out of memory");
		return NULL;
	}
	bo->fd = fcntl(buf_fd, F_DUPFD_CLOEXEC, 0);
	if (bo->fd < 0) {
		LOGE("import fd %d: dup: %s", buf_fd, strerror(errno));
		free(bo);
		return NULL;
	}
	bo->size = (uint64_t)size;
	bo->modifier = modifier;
	bo->width = width;
	bo->height = height;
	bo->stride = stride;
	bo->drm_format = drm_format;
	bo->rdev = gbm ? ((struct nvg_device *)gbm)->rdev : 0;
	LOGT("import fd %d -> bo %p (fd %d) %ux%u fmt 0x%08x stride %u modifier 0x%016" PRIx64
	     " size %" PRIu64,
	     buf_fd, (void *)bo, bo->fd, width, height, drm_format, stride, modifier, bo->size);
	return (struct gbm_bo *)bo;
}

static pthread_mutex_t g_xfer_lock;
static void tiled_release_locked(struct nvg_bo *bo);

static void nvg_free(struct gbm_bo *gbm_bo)
{
	struct nvg_bo *bo = (struct nvg_bo *)gbm_bo;
	if (!bo)
		return;
	LOGT("free bo %p (fd %d)", (void *)bo, bo->fd);
	const unsigned live = __atomic_load_n(&bo->linear_maps, __ATOMIC_RELAXED);
	if (live)
		LOGE("free bo %p: %u CPU mapping(s) still live (unmap never called); the dma-buf "
		     "stays allocated until they are unmapped",
		     (void *)bo, live);
	if (bo->img != VK_NULL_HANDLE || bo->stg != VK_NULL_HANDLE) {
		pthread_mutex_lock(&g_xfer_lock);
		if (bo->locked)
			LOGE("free bo %p: still CPU-locked (unmap never called); releasing the "
			     "staging copy anyway, the CPU writes since the lock are lost",
			     (void *)bo);
		tiled_release_locked(bo);
		pthread_mutex_unlock(&g_xfer_lock);
	}
	close(bo->fd);
	free(bo);
}

static uint64_t dma_buf_dir(uint32_t flags)
{
	return (flags & GBM_MAP_FLAG_READ ? DMA_BUF_SYNC_READ : 0) |
	       (flags & GBM_MAP_FLAG_WRITE ? DMA_BUF_SYNC_WRITE : 0);
}

static int dma_buf_sync(int fd, uint64_t flags)
{
	struct dma_buf_sync s = { .flags = flags };
	int r;
	do {
		r = ioctl(fd, DMA_BUF_IOCTL_SYNC, &s);
	} while (r < 0 && (errno == EINTR || errno == EAGAIN));
	return r < 0 ? -errno : 0;
}

/* ------------------------------------------------------------------------ */
/* CPU lock/unlock of block-linear buffers: GPU copy through linear staging   */
/*
 * A block-linear buffer has no linear CPU view. The design is minigbm's
 * amdgpu backend (amdgpu.c amdgpu_map_bo/amdgpu_unmap_bo: SDMA copy to and
 * from a linear GTT buffer), done through NVIDIA's Vulkan ICD in the locking
 * process:
 *   map:   copy the image into a linear staging buffer in system memory
 *          (rows of the importer's stride = strides[0] = the allocator's
 *          rowPitch = map_stride), wait for the copy, return the staging
 *          pointer.
 *   unmap: copy the staging buffer back into the image, wait, return.
 * Synchronisation, all explicit (NVIDIA takes no part in dma-buf implicit
 * sync, nvidia-prolapse/egl/FENCE-CONTRACT.md §2.5):
 *   - before map: minigbm's lock waits for the acquire fence on the CPU
 *     before it calls bo_map (cros_gralloc_driver.cc:446-450,
 *     cros_gralloc_sync_wait), so the producer's GPU work is complete when
 *     our copy is submitted. Nothing here waits on or relies on implicit
 *     sync; a lock without a fence is ordered against nothing, exactly as
 *     the gralloc contract says.
 *   - after unmap: minigbm's unlock always returns release fence -1
 *     (cros_gralloc_driver.cc:493, "may access immediately"), so the
 *     copy-back must be complete before unmap returns: we wait on the CPU.
 *   - the copies use one queue of our own device with a VkFence only: no
 *     semaphore, no sync_fd import or export (the device does not even
 *     enable them), so NVIDIA's queue-timeline defect D1 cannot apply.
 * Access flags (BO_MAP_READ/BO_MAP_WRITE from the lock usage) come through
 * get_gbm_map_flags_ops(), which our minigbm patch 0002 calls:
 *   - every lock copies tiled -> staging at map, READ or WRITE: the copy-back
 *     covers the whole buffer, and the gralloc contract keeps the content a
 *     write lock does not touch ("The content of the buffer outside of the
 *     specified area is NOT modified by this call", hardware/libhardware
 *     gralloc.h lock(); AHardwareBuffer_lock in hardware_buffer.h says the
 *     same; IMapper 4.0 lock() never makes it undefined), so a WRITE-only
 *     lock must start from the buffer's content too;
 *   - unmap copies staging -> tiled only when the mapping had WRITE: a
 *     read-only lock never writes the buffer.
 * The plain gbm_ops.map/unmap (an unpatched glue, which has no flags) are
 * READ|WRITE: copy back after every lock.
 * Staging memory is HOST_VISIBLE|HOST_CACHED system memory: CPU reads of an
 * uncached (BAR/write-combined) mapping are very slow. Never the VRAM itself.
 */

/* every lock/unlock copy and the per-buffer staging cache are serialised by
 * this lock (minigbm already serialises lock/unlock per process: its driver
 * mutex is held across bo_map, cros_gralloc_driver.cc:452-467) */
static pthread_mutex_t g_xfer_lock = PTHREAD_MUTEX_INITIALIZER;
static VkCommandPool g_xfer_pool = VK_NULL_HANDLE;
static VkCommandBuffer g_xfer_cb = VK_NULL_HANDLE;
static VkFence g_xfer_fence = VK_NULL_HANDLE;
static int g_xfer_state; /* 0 = not made, 1 = ready, -1 = failed or a copy hung (sticky) */
static char g_xfer_error[512];

#define NVG_COPY_TIMEOUT_NS 10000000000ull /* a copy of <= 256 MiB; anything longer is a hang */

static bool xfer_ready_locked(struct nvg_vk *vk)
{
	if (g_xfer_state > 0)
		return true;
	if (g_xfer_state < 0) {
		LOGE("CPU lock of a block-linear buffer: copy context unusable: %s", g_xfer_error);
		return false;
	}
	const VkCommandPoolCreateInfo pci = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = vk->qf,
	};
	VkResult r = vk->CreateCommandPool(vk->device, &pci, NULL, &g_xfer_pool);
	if (r == VK_SUCCESS) {
		const VkCommandBufferAllocateInfo cai = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = g_xfer_pool,
			.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
			.commandBufferCount = 1,
		};
		r = vk->AllocateCommandBuffers(vk->device, &cai, &g_xfer_cb);
	}
	if (r == VK_SUCCESS) {
		const VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
		r = vk->CreateFence(vk->device, &fci, NULL, &g_xfer_fence);
	}
	if (r != VK_SUCCESS) {
		/* objects made so far live with the device for the process lifetime */
		snprintf(g_xfer_error, sizeof(g_xfer_error), "creating the copy command pool/"
			 "buffer/fence: %s (%d)", vk_result_str(r), r);
		g_xfer_state = -1;
		LOGE("CPU lock of a block-linear buffer: %s", g_xfer_error);
		return false;
	}
	g_xfer_state = 1;
	return true;
}

static void tiled_release_locked(struct nvg_bo *bo)
{
	struct nvg_vk *vk = &g_vk; /* only set up buffers have handles, so the device exists */
	if (bo->stg != VK_NULL_HANDLE)
		vk->DestroyBuffer(vk->device, bo->stg, NULL);
	/* Unmap before the free: a CPU mapping is never live while its memory
	 * is freed (the NVIDIA driver itself orders free as RM unmap, munmap,
	 * RM free). */
	if (bo->stg_ptr != NULL)
		vk->UnmapMemory(vk->device, bo->stg_mem);
	bo->stg_ptr = NULL;
	if (bo->stg_mem != VK_NULL_HANDLE)
		vk->FreeMemory(vk->device, bo->stg_mem, NULL);
	if (bo->img != VK_NULL_HANDLE)
		vk->DestroyImage(vk->device, bo->img, NULL);
	if (bo->img_mem != VK_NULL_HANDLE)
		vk->FreeMemory(vk->device, bo->img_mem, NULL); /* drops the import's dma-buf ref */
	bo->stg = VK_NULL_HANDLE;
	bo->stg_mem = VK_NULL_HANDLE;
	bo->img = VK_NULL_HANDLE;
	bo->img_mem = VK_NULL_HANDLE;
	bo->locked = false;
}

/* import the dma-buf as a VkImage with its explicit modifier and stride, and
 * create the persistently mapped staging buffer; both cached on the bo */
static bool tiled_setup_locked(struct nvg_vk *vk, struct nvg_bo *bo, const struct nvg_format *f)
{
	VkResult r;
	int dupfd = -1;
	const VkSubresourceLayout plane = { .offset = 0, .rowPitch = bo->stride };
	const VkImageDrmFormatModifierExplicitCreateInfoEXT exp = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
		.drmFormatModifier = bo->modifier,
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
		.extent = { bo->width, bo->height, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
		.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	r = vk->CreateImage(vk->device, &ici, NULL, &bo->img);
	if (r != VK_SUCCESS) {
		bo->img = VK_NULL_HANDLE;
		LOGE("lock bo %p %ux%u %s modifier 0x%016" PRIx64 " stride %u: importing the dma-buf "
		     "as an image: vkCreateImage: %s (%d)",
		     (void *)bo, bo->width, bo->height, f->name, bo->modifier, bo->stride,
		     vk_result_str(r), r);
		goto fail;
	}
	VkMemoryRequirements req;
	vk->GetImageMemoryRequirements(vk->device, bo->img, &req);
	VkMemoryFdPropertiesKHR fdp = { .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR };
	r = vk->GetMemoryFdPropertiesKHR(vk->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
					 bo->fd, &fdp);
	uint32_t type = UINT32_MAX;
	for (uint32_t i = 0; r == VK_SUCCESS && i < vk->memprops.memoryTypeCount; i++)
		if (req.memoryTypeBits & fdp.memoryTypeBits & (1u << i)) {
			type = i;
			break;
		}
	if (r != VK_SUCCESS || type == UINT32_MAX || req.size > bo->size) {
		LOGE("lock bo %p: dma-buf import: vkGetMemoryFdPropertiesKHR %s (%d), image types "
		     "0x%x, fd types 0x%x, image needs %" PRIu64 " bytes, dma-buf has %" PRIu64,
		     (void *)bo, vk_result_str(r), r, req.memoryTypeBits, fdp.memoryTypeBits,
		     (uint64_t)req.size, bo->size);
		goto fail;
	}
	dupfd = fcntl(bo->fd, F_DUPFD_CLOEXEC, 0);
	if (dupfd < 0) {
		LOGE("lock bo %p: dup(fd %d): %s", (void *)bo, bo->fd, strerror(errno));
		goto fail;
	}
	{
		const VkImportMemoryFdInfoKHR imp = {
			.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
			.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
			.fd = dupfd,
		};
		const VkMemoryDedicatedAllocateInfo ded = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
			.pNext = &imp,
			.image = bo->img,
		};
		const VkMemoryAllocateInfo mai = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.pNext = &ded,
			.allocationSize = req.size,
			.memoryTypeIndex = type,
		};
		r = vk->AllocateMemory(vk->device, &mai, NULL, &bo->img_mem);
	}
	if (r != VK_SUCCESS) {
		bo->img_mem = VK_NULL_HANDLE;
		close(dupfd); /* ownership passes to the driver only on success */
		LOGE("lock bo %p: dma-buf import: vkAllocateMemory: %s (%d)", (void *)bo,
		     vk_result_str(r), r);
		goto fail;
	}
	r = vk->BindImageMemory(vk->device, bo->img, bo->img_mem, 0);
	if (r != VK_SUCCESS) {
		LOGE("lock bo %p: vkBindImageMemory: %s (%d)", (void *)bo, vk_result_str(r), r);
		goto fail;
	}

	/* staging: rows of exactly `stride` bytes (= map_stride) */
	bo->stg_size = (VkDeviceSize)bo->stride * bo->height;
	const VkBufferCreateInfo bci = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = bo->stg_size,
		.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	r = vk->CreateBuffer(vk->device, &bci, NULL, &bo->stg);
	if (r != VK_SUCCESS) {
		bo->stg = VK_NULL_HANDLE;
		LOGE("lock bo %p: staging vkCreateBuffer(%" PRIu64 "): %s (%d)", (void *)bo,
		     (uint64_t)bo->stg_size, vk_result_str(r), r);
		goto fail;
	}
	VkMemoryRequirements breq;
	vk->GetBufferMemoryRequirements(vk->device, bo->stg, &breq);
	/* HOST_CACHED is required (CPU reads of uncached memory are very slow);
	 * prefer a coherent one (no flush/invalidate needed) */
	const VkMemoryPropertyFlags cached =
	    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
	type = UINT32_MAX;
	for (int pass = 0; pass < 2 && type == UINT32_MAX; pass++) {
		const VkMemoryPropertyFlags want =
		    cached | (pass == 0 ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : 0);
		for (uint32_t i = 0; i < vk->memprops.memoryTypeCount; i++)
			if ((breq.memoryTypeBits & (1u << i)) &&
			    (vk->memprops.memoryTypes[i].propertyFlags & want) == want &&
			    !(vk->memprops.memoryTypes[i].propertyFlags &
			      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
				type = i;
				break;
			}
	}
	if (type == UINT32_MAX) {
		LOGE("lock bo %p: staging memoryTypeBits 0x%x has no HOST_VISIBLE|HOST_CACHED system "
		     "memory type; not substituting an uncached one",
		     (void *)bo, breq.memoryTypeBits);
		goto fail;
	}
	bo->stg_coherent =
	    vk->memprops.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	{
		const VkMemoryAllocateInfo mai = {
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = breq.size,
			.memoryTypeIndex = type,
		};
		r = vk->AllocateMemory(vk->device, &mai, NULL, &bo->stg_mem);
	}
	if (r != VK_SUCCESS) {
		bo->stg_mem = VK_NULL_HANDLE;
		LOGE("lock bo %p: staging vkAllocateMemory(%" PRIu64 " bytes, type %u): %s (%d)",
		     (void *)bo, (uint64_t)breq.size, type, vk_result_str(r), r);
		goto fail;
	}
	r = vk->BindBufferMemory(vk->device, bo->stg, bo->stg_mem, 0);
	if (r == VK_SUCCESS)
		r = vk->MapMemory(vk->device, bo->stg_mem, 0, VK_WHOLE_SIZE, 0, &bo->stg_ptr);
	if (r != VK_SUCCESS) {
		bo->stg_ptr = NULL; /* nothing to unmap */
		LOGE("lock bo %p: staging bind/map: %s (%d)", (void *)bo, vk_result_str(r), r);
		goto fail;
	}
	LOGT("lock bo %p: block-linear %ux%u %s modifier 0x%016" PRIx64 " imported; staging %" PRIu64
	     " bytes, pitch %u, memtype %u (0x%x)",
	     (void *)bo, bo->width, bo->height, f->name, bo->modifier, (uint64_t)bo->stg_size,
	     bo->stride, type, vk->memprops.memoryTypes[type].propertyFlags);
	return true;

fail:
	tiled_release_locked(bo);
	return false;
}

/* one copy between the image and the staging buffer, waited for on the CPU */
static bool tiled_copy_locked(struct nvg_vk *vk, struct nvg_bo *bo, const struct nvg_format *f,
			      bool to_staging)
{
	VkResult r = vk->ResetCommandBuffer(g_xfer_cb, 0);
	const VkCommandBufferBeginInfo bi = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	if (r == VK_SUCCESS)
		r = vk->BeginCommandBuffer(g_xfer_cb, &bi);
	if (r != VK_SUCCESS) {
		LOGE("lock bo %p: recording the copy: %s (%d)", (void *)bo, vk_result_str(r), r);
		return false;
	}
	const VkImageLayout lay = to_staging ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
					     : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	const VkAccessFlags acc = to_staging ? VK_ACCESS_TRANSFER_READ_BIT
					     : VK_ACCESS_TRANSFER_WRITE_BIT;
	/* the image's contents belong to other users (GL, Vulkan, other
	 * processes): acquire it from VK_QUEUE_FAMILY_EXTERNAL in GENERAL
	 * (contents preserved) and hand it back the same way */
	VkImageMemoryBarrier ib = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = 0,
		.dstAccessMask = acc,
		.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
		.newLayout = lay,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL,
		.dstQueueFamilyIndex = vk->qf,
		.image = bo->img,
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
	};
	/* host writes before the submit are visible to it (vkQueueSubmit's
	 * implicit host-write ordering); flushed above for non-coherent memory */
	vk->CmdPipelineBarrier(g_xfer_cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
	const VkBufferImageCopy c = {
		.bufferOffset = 0,
		.bufferRowLength = bo->stride / f->bpp,
		.bufferImageHeight = bo->height,
		.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
		.imageExtent = { bo->width, bo->height, 1 },
	};
	if (to_staging)
		vk->CmdCopyImageToBuffer(g_xfer_cb, bo->img, lay, bo->stg, 1, &c);
	else
		vk->CmdCopyBufferToImage(g_xfer_cb, bo->stg, bo->img, lay, 1, &c);
	ib.srcAccessMask = acc;
	ib.dstAccessMask = 0;
	ib.oldLayout = lay;
	ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	ib.srcQueueFamilyIndex = vk->qf;
	ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
	vk->CmdPipelineBarrier(g_xfer_cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
			       VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0, NULL, 1, &ib);
	if (to_staging) {
		const VkBufferMemoryBarrier bb = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_HOST_READ_BIT,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.buffer = bo->stg,
			.size = VK_WHOLE_SIZE,
		};
		vk->CmdPipelineBarrier(g_xfer_cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
				       VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1, &bb, 0, NULL);
	}
	r = vk->EndCommandBuffer(g_xfer_cb);
	const VkSubmitInfo si = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &g_xfer_cb,
	};
	if (r == VK_SUCCESS)
		r = vk->QueueSubmit(vk->queue, 1, &si, g_xfer_fence);
	if (r != VK_SUCCESS) {
		LOGE("lock bo %p: submitting the %s copy: %s (%d)", (void *)bo,
		     to_staging ? "read-back" : "write-back", vk_result_str(r), r);
		return false;
	}
	r = vk->WaitForFences(vk->device, 1, &g_xfer_fence, VK_TRUE, NVG_COPY_TIMEOUT_NS);
	if (r != VK_SUCCESS) {
		/* the command buffer may still be in flight: never reuse it */
		snprintf(g_xfer_error, sizeof(g_xfer_error), "the %s copy of bo %p (%ux%u) did not "
			 "complete: vkWaitForFences %s (%d); GPU hang or device loss",
			 to_staging ? "read-back" : "write-back", (void *)bo, bo->width, bo->height,
			 vk_result_str(r), r);
		g_xfer_state = -1;
		LOGE("%s", g_xfer_error);
		return false;
	}
	r = vk->ResetFences(vk->device, 1, &g_xfer_fence);
	if (r != VK_SUCCESS) {
		snprintf(g_xfer_error, sizeof(g_xfer_error), "vkResetFences: %s (%d)",
			 vk_result_str(r), r);
		g_xfer_state = -1;
		LOGE("%s", g_xfer_error);
		return false;
	}
	return true;
}

static void nvg_map_tiled(struct nvg_bo *bo, uint32_t flags, void **addr, void **map_data)
{
	const struct nvg_format *f = find_format(bo->drm_format);
	if (!f || bo->stride % f->bpp || bo->stride < bo->width * f->bpp) {
		/* minigbm only imports whitelisted formats with this backend's
		 * layout; anything else is not a buffer we allocated */
		LOGE("lock bo %p %ux%u fmt 0x%08x modifier 0x%016" PRIx64 " stride %u: not a mappable "
		     "block-linear buffer of this backend (format %s, stride must be whole pixels "
		     ">= width): EINVAL",
		     (void *)bo, bo->width, bo->height, bo->drm_format, bo->modifier, bo->stride,
		     f ? f->name : "not whitelisted");
		errno = EINVAL;
		return;
	}
	struct nvg_vk *vk = vk_get(bo->rdev); /* lazy: may be this process's first ICD use */
	if (!vk) {
		LOGE("lock bo %p (block-linear): needs the NVIDIA Vulkan ICD in this process for "
		     "the staging copy and it is unavailable (above): ENODEV", (void *)bo);
		errno = ENODEV;
		return;
	}
	struct nvg_map *m = malloc(sizeof(*m));
	if (!m) {
		LOGE("map: out of memory");
		errno = ENOMEM;
		return;
	}
	int err = EIO;
	pthread_mutex_lock(&g_xfer_lock);
	if (bo->locked) {
		/* minigbm maps a bo once and refcounts (drv.c:498-523) */
		LOGE("lock bo %p: already CPU-locked in this process (map without unmap): EBUSY",
		     (void *)bo);
		err = EBUSY;
		goto fail;
	}
	if (!xfer_ready_locked(vk))
		goto fail;
	if (bo->img == VK_NULL_HANDLE && !tiled_setup_locked(vk, bo, f))
		goto fail;
	if (!tiled_copy_locked(vk, bo, f, true))
		goto fail;
	if (!bo->stg_coherent) {
		const VkMappedMemoryRange mr = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
						 .memory = bo->stg_mem,
						 .size = VK_WHOLE_SIZE };
		VkResult r = vk->InvalidateMappedMemoryRanges(vk->device, 1, &mr);
		if (r != VK_SUCCESS) {
			LOGE("lock bo %p: vkInvalidateMappedMemoryRanges: %s (%d)", (void *)bo,
			     vk_result_str(r), r);
			goto fail;
		}
	}
	bo->locked = true;
	pthread_mutex_unlock(&g_xfer_lock);
	m->addr = bo->stg_ptr;
	m->size = (size_t)bo->stg_size;
	m->tiled = true;
	m->flags = flags;
	*addr = bo->stg_ptr;
	*map_data = m;
	LOGT("map bo %p (block-linear, %s%s) -> staging %p (%zu bytes, pitch %u)", (void *)bo,
	     flags & GBM_MAP_FLAG_READ ? "R" : "", flags & GBM_MAP_FLAG_WRITE ? "W" : "", m->addr,
	     m->size, bo->stride);
	return;
fail:
	pthread_mutex_unlock(&g_xfer_lock);
	free(m);
	errno = err;
}

static void nvg_unmap_tiled(struct nvg_bo *bo, struct nvg_map *m, uint32_t flags)
{
	const struct nvg_format *f = find_format(bo->drm_format);
	struct nvg_vk *vk = &g_vk; /* a successful map made it */
	/* the mapping's flags: a nested lock may have widened them since map */
	const bool write = (m->flags | flags) & GBM_MAP_FLAG_WRITE;
	pthread_mutex_lock(&g_xfer_lock);
	if (!write) {
		/* read-only lock: the buffer is not written, whatever the CPU did
		 * to the staging copy (the next lock copies in afresh) */
		const bool was_locked = bo->locked;
		bo->locked = false;
		pthread_mutex_unlock(&g_xfer_lock);
		if (!was_locked)
			LOGE("unlock bo %p %ux%u (block-linear, read-only): it was not locked",
			     (void *)bo, bo->width, bo->height);
		LOGT("unmap bo %p (block-linear, read-only, staging %p): no write-back", (void *)bo,
		     m->addr);
		free(m);
		return;
	}
	bool ok = bo->locked && f && g_xfer_state > 0;
	if (ok && !bo->stg_coherent) {
		const VkMappedMemoryRange mr = { .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
						 .memory = bo->stg_mem,
						 .size = VK_WHOLE_SIZE };
		VkResult r = vk->FlushMappedMemoryRanges(vk->device, 1, &mr);
		if (r != VK_SUCCESS) {
			LOGE("unlock bo %p: vkFlushMappedMemoryRanges: %s (%d)", (void *)bo,
			     vk_result_str(r), r);
			ok = false;
		}
	}
	if (ok)
		ok = tiled_copy_locked(vk, bo, f, false);
	if (!ok)
		LOGE("unlock bo %p %ux%u (block-linear): the CPU's changes were NOT written back to "
		     "the buffer (%s)",
		     (void *)bo, bo->width, bo->height,
		     !bo->locked ? "it was not locked" : g_xfer_state < 0 ? g_xfer_error : "see above");
	bo->locked = false;
	pthread_mutex_unlock(&g_xfer_lock);
	LOGT("unmap bo %p (block-linear, staging %p)%s", (void *)bo, m->addr,
	     ok ? ", written back" : "");
	free(m);
}

/* minigbm checks the result against MAP_FAILED (drv.c:534-535) and asserts that
 * map_data is non-NULL on unmap (gbm_mesa_internals.cpp:483) */
static void nvg_map_impl(struct gbm_bo *gbm_bo, int w, int h, uint32_t flags, void **addr,
			 void **map_data)
{
	struct nvg_bo *bo = (struct nvg_bo *)gbm_bo;
	*addr = MAP_FAILED;
	*map_data = NULL;

	if (!bo) {
		/* minigbm maps the gbm_bo its import returned; NULL means import failed */
		LOGE("map %dx%d: buffer was never imported (import failed earlier)", w, h);
		errno = EINVAL;
		return;
	}
	if (bo->modifier != DRM_FORMAT_MOD_LINEAR) {
		/* a direct CPU view of tiled memory is garbage: GPU copy instead */
		nvg_map_tiled(bo, flags, addr, map_data);
		return;
	}
	struct nvg_map *m = malloc(sizeof(*m));
	if (!m) {
		LOGE("map: out of memory");
		errno = ENOMEM;
		return;
	}
	void *p = mmap(NULL, (size_t)bo->size, PROT_READ | PROT_WRITE, MAP_SHARED, bo->fd, 0);
	if (p == MAP_FAILED) {
		const int e = errno;
		LOGE("map bo %p: mmap(fd %d, %" PRIu64 " bytes): %s", (void *)bo, bo->fd, bo->size,
		     strerror(e));
		free(m);
		errno = e;
		return;
	}
	/* minigbm's gbm_mesa backend has no bo_invalidate/bo_flush
	 * (gbm_mesa_driver.cpp:26-38), so CPU access is bracketed here: map is
	 * called per lock and unmap per unlock (drv.c:534, 569) */
	const int sr = dma_buf_sync(bo->fd, DMA_BUF_SYNC_START | dma_buf_dir(flags));
	if (sr) {
		LOGE("map bo %p: DMA_BUF_IOCTL_SYNC(START|0x%x): %s", (void *)bo, flags,
		     strerror(-sr));
		munmap(p, (size_t)bo->size);
		free(m);
		errno = -sr;
		return;
	}
	m->addr = p;
	m->size = (size_t)bo->size;
	m->tiled = false;
	m->flags = flags;
	__atomic_add_fetch(&bo->linear_maps, 1, __ATOMIC_RELAXED);
	*addr = p;
	*map_data = m;
	LOGT("map bo %p -> %p (%zu bytes)", (void *)bo, p, m->size);
}

static void nvg_unmap_impl(struct gbm_bo *gbm_bo, void *map_data, uint32_t flags)
{
	struct nvg_bo *bo = (struct nvg_bo *)gbm_bo;
	struct nvg_map *m = map_data;
	if (!bo || !m) {
		LOGE("unmap: bo %p map_data %p", (void *)bo, map_data);
		return;
	}
	if (m->tiled) {
		nvg_unmap_tiled(bo, m, flags);
		return;
	}
	const int sr = dma_buf_sync(bo->fd, DMA_BUF_SYNC_END | dma_buf_dir(m->flags | flags));
	if (sr)
		LOGE("unmap bo %p: DMA_BUF_IOCTL_SYNC(END|0x%x): %s", (void *)bo, m->flags | flags,
		     strerror(-sr));
	if (munmap(m->addr, m->size))
		LOGE("unmap bo %p: munmap(%p): %s", (void *)bo, m->addr, strerror(errno));
	else
		__atomic_sub_fetch(&bo->linear_maps, 1, __ATOMIC_RELAXED);
	LOGT("unmap bo %p (%p)", (void *)bo, m->addr);
	free(m);
}

/* the plain gbm_ops ABI carries no access flags (an unpatched glue): READ|WRITE */
static void nvg_map(struct gbm_bo *gbm_bo, int w, int h, void **addr, void **map_data)
{
	nvg_map_impl(gbm_bo, w, h, NVG_MAP_RW, addr, map_data);
}

static void nvg_unmap(struct gbm_bo *gbm_bo, void *map_data)
{
	nvg_unmap_impl(gbm_bo, map_data, NVG_MAP_RW);
}

/* gbm_mesa_wrapper_map_flags.h: the lock's BO_MAP_READ/BO_MAP_WRITE */
static void nvg_map_flags(struct gbm_bo *gbm_bo, int w, int h, uint32_t flags, void **addr,
			  void **map_data)
{
	if (!flags || (flags & ~NVG_MAP_RW)) {
		*addr = MAP_FAILED;
		*map_data = NULL;
		LOGE("map %dx%d: access flags 0x%x are not a non-empty subset of READ|WRITE: EINVAL",
		     w, h, flags);
		errno = EINVAL;
		return;
	}
	nvg_map_impl(gbm_bo, w, h, flags, addr, map_data);
}

static void nvg_unmap_flags(struct gbm_bo *gbm_bo, void *map_data, uint32_t flags)
{
	if (flags & ~NVG_MAP_RW)
		LOGE("unmap: unknown access flags 0x%x ignored", flags & ~NVG_MAP_RW);
	nvg_unmap_impl(gbm_bo, map_data, flags & NVG_MAP_RW);
}

static const struct gbm_map_flags_ops nvg_map_flags_ops = {
	.version = GBM_MAP_FLAGS_OPS_VERSION,
	.map = nvg_map_flags,
	.unmap = nvg_unmap_flags,
};

/*
 * gbm_mesa_wrapper_layout.h: where the planes of a NV12/P010 buffer go, which
 * minigbm lays out itself on its R8 carrier. Importers (the HAL's AHB import,
 * ANGLE, NVK) bind it as a two-plane LINEAR image with explicit offsets.
 * NVIDIA's driver (615, measured) binds each plane of that image as a mapping
 * of its own, sized and aligned like the plane of a DISJOINT image (pitch *
 * rows rounded up to 64 KiB, 64 KiB alignment): a plane offset off a page makes
 * RM reject the plane's mapping and the plane samples zeros, a plane range past
 * the buffer samples zeros too. So each plane goes where a DISJOINT image of the
 * same layout could bind it: offset aligned to that plane's requirement, after
 * the previous plane's required size; the buffer holds the last plane's
 * requirement and the whole image's. The driver answers for its own rule
 * (NVK: 32-byte alignment, exact sizes, i.e. minigbm's packed layout).
 * Usage = what importers ask (SAMPLED); dma-buf external memory.
 */
static int nvg_plane_layout(struct gbm_device *gbm, uint32_t drm_format, uint32_t width,
			    uint32_t height, uint32_t num_planes, const uint32_t *strides,
			    uint32_t *offsets, uint64_t *total_size)
{
	VkFormat vf;
	switch (drm_format) {
	case DRM_FORMAT_NV12: vf = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM; break;
	case DRM_FORMAT_P010: vf = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16; break;
	default:
		LOGE("plane_layout %ux%u: DRM format 0x%08x has no multi-planar Vulkan format here "
		     "(NV12 and P010 only)", width, height, drm_format);
		return -EINVAL;
	}
	if (!gbm || !total_size || !strides || !offsets || num_planes != 2 || !width || !height) {
		LOGE("plane_layout %ux%u format 0x%08x: bad arguments (%u planes)", width, height,
		     drm_format, num_planes);
		return -EINVAL;
	}
	struct nvg_vk *vk = vk_get(((struct nvg_device *)gbm)->rdev);
	if (!vk)
		return -ENODEV;

	/* DISJOINT must be a LINEAR feature of the format for the per-plane query */
	VkDrmFormatModifierPropertiesList2EXT ml = {
		.sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT,
	};
	VkFormatProperties2 fp = { .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &ml };
	vk->GetPhysicalDeviceFormatProperties2(vk->pd, vf, &fp);
	VkDrmFormatModifierProperties2EXT mprops[NVG_MAX_MODS];
	if (ml.drmFormatModifierCount > NVG_MAX_MODS)
		ml.drmFormatModifierCount = NVG_MAX_MODS;
	ml.pDrmFormatModifierProperties = mprops;
	vk->GetPhysicalDeviceFormatProperties2(vk->pd, vf, &fp);
	VkFormatFeatureFlags2 feat = 0;
	for (uint32_t i = 0; i < ml.drmFormatModifierCount; i++)
		if (mprops[i].drmFormatModifier == DRM_FORMAT_MOD_LINEAR &&
		    mprops[i].drmFormatModifierPlaneCount == 2)
			feat = mprops[i].drmFormatModifierTilingFeatures;
	if (!(feat & VK_FORMAT_FEATURE_2_DISJOINT_BIT)) {
		LOGE("plane_layout %ux%u format 0x%08x: the driver lists no 2-plane LINEAR modifier with "
		     "DISJOINT (features 0x%" PRIx64 "); its per-plane requirements are unknown",
		     width, height, drm_format, (uint64_t)feat);
		return -EINVAL;
	}

	/* importers sample it (the decoder writes through its own copy engine, not Vulkan); the
	 * pinned NVK refuses TRANSFER usage on explicit multi-planar LINEAR images */
	const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT;
	VkSubresourceLayout planes[2];
	const VkImageDrmFormatModifierExplicitCreateInfoEXT explicit_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
		.drmFormatModifier = DRM_FORMAT_MOD_LINEAR,
		.drmFormatModifierPlaneCount = 2,
		.pPlaneLayouts = planes,
	};
	const VkExternalMemoryImageCreateInfo emi = {
		.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
		.pNext = &explicit_info,
		.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
	};
	VkImageCreateInfo ici = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.pNext = &emi,
		.flags = VK_IMAGE_CREATE_DISJOINT_BIT,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = vf,
		.extent = { width, height, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	/* 1. per-plane requirements: a DISJOINT image, each plane at offset 0 of its binding */
	for (uint32_t p = 0; p < 2; p++)
		planes[p] = (VkSubresourceLayout){ .offset = 0, .rowPitch = strides[p] };
	VkImage image = VK_NULL_HANDLE;
	VkResult r = vk->CreateImage(vk->device, &ici, NULL, &image);
	if (r != VK_SUCCESS) {
		LOGE("plane_layout %ux%u format 0x%08x: vkCreateImage(DISJOINT, LINEAR, pitches %u/%u): "
		     "%s (%d)", width, height, drm_format, strides[0], strides[1], vk_result_str(r), r);
		return r == VK_ERROR_OUT_OF_HOST_MEMORY || r == VK_ERROR_OUT_OF_DEVICE_MEMORY ? -ENOMEM
											       : -EINVAL;
	}
	VkMemoryRequirements preq[2];
	for (uint32_t p = 0; p < 2; p++) {
		VkImagePlaneMemoryRequirementsInfo pi = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO,
			.planeAspect = p ? VK_IMAGE_ASPECT_PLANE_1_BIT : VK_IMAGE_ASPECT_PLANE_0_BIT,
		};
		VkImageMemoryRequirementsInfo2 mi = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
			.pNext = &pi,
			.image = image,
		};
		VkMemoryRequirements2 mr = { .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
		vk->GetImageMemoryRequirements2(vk->device, &mi, &mr);
		preq[p] = mr.memoryRequirements;
	}
	vk->DestroyImage(vk->device, image, NULL);

	/* 2. offsets: aligned to the plane's requirement, past the previous plane's size */
	uint64_t off[2], end = 0;
	for (uint32_t p = 0; p < 2; p++) {
		const uint64_t a = preq[p].alignment ? preq[p].alignment : 1;
		uint64_t o = offsets[p] > end ? offsets[p] : end;
		o = (o + a - 1) / a * a;
		if (o > UINT32_MAX) {
			LOGE("plane_layout %ux%u format 0x%08x: plane %u offset %" PRIu64 " does not fit "
			     "32 bits", width, height, drm_format, p, o);
			return -EINVAL;
		}
		off[p] = o;
		end = o + preq[p].size;
	}

	/* 3. the whole image with those offsets: the driver must accept it; its requirement counts too */
	for (uint32_t p = 0; p < 2; p++)
		planes[p] = (VkSubresourceLayout){ .offset = off[p], .rowPitch = strides[p] };
	ici.flags = 0;
	r = vk->CreateImage(vk->device, &ici, NULL, &image);
	if (r != VK_SUCCESS) {
		LOGE("plane_layout %ux%u format 0x%08x: vkCreateImage(LINEAR, plane offsets %" PRIu64
		     "/%" PRIu64 ", pitches %u/%u): %s (%d); the driver cannot import this layout",
		     width, height, drm_format, off[0], off[1], strides[0], strides[1],
		     vk_result_str(r), r);
		return r == VK_ERROR_OUT_OF_HOST_MEMORY || r == VK_ERROR_OUT_OF_DEVICE_MEMORY ? -ENOMEM
											       : -EINVAL;
	}
	VkMemoryRequirements reqs;
	vk->GetImageMemoryRequirements(vk->device, image, &reqs);
	vk->DestroyImage(vk->device, image, NULL);
	if (reqs.size > end)
		end = reqs.size;

	LOGT("plane_layout %ux%u format 0x%08x pitches %u/%u: plane offsets %u/%u -> %" PRIu64
	     "/%" PRIu64 " (plane requirements %" PRIu64 "/%" PRIu64 " B, alignment %" PRIu64
	     "/%" PRIu64 "), image %" PRIu64 " B, buffer %" PRIu64 " B",
	     width, height, drm_format, strides[0], strides[1], offsets[0], offsets[1], off[0], off[1],
	     (uint64_t)preq[0].size, (uint64_t)preq[1].size, (uint64_t)preq[0].alignment,
	     (uint64_t)preq[1].alignment, (uint64_t)reqs.size, end);
	offsets[0] = (uint32_t)off[0];
	offsets[1] = (uint32_t)off[1];
	*total_size = end;
	return 0;
}

static int nvg_alloc(struct alloc_args *args)
{
	return nvg_alloc_impl(args, 0);
}

static int nvg_alloc_flags(struct alloc_args *args, uint32_t flags)
{
	if (flags & ~GBM_ALLOC_FLAG_CPU_READ_OFTEN) {
		LOGE("alloc: unknown allocation flags 0x%x", flags & ~GBM_ALLOC_FLAG_CPU_READ_OFTEN);
		return -EINVAL;
	}
	return nvg_alloc_impl(args, flags);
}

static const struct gbm_alloc_flags_ops nvg_alloc_flags_ops = {
	.version = GBM_ALLOC_FLAGS_OPS_VERSION,
	.alloc = nvg_alloc_flags,
};

static const struct gbm_layout_ops nvg_layout_ops = {
	.version = GBM_LAYOUT_OPS_VERSION,
	.plane_layout = nvg_plane_layout,
};

static struct gbm_ops nvg_ops = {
	.get_gbm_format = nvg_get_gbm_format,
	.dev_create = nvg_dev_create,
	.dev_destroy = nvg_dev_destroy,
	.alloc = nvg_alloc,
	.import = nvg_import,
	.free = nvg_free,
	.map = nvg_map,
	.unmap = nvg_unmap,
};

NVG_EXPORT struct gbm_ops *get_gbm_ops(void)
{
	return &nvg_ops;
}

NVG_EXPORT const struct gbm_map_flags_ops *get_gbm_map_flags_ops(void)
{
	return &nvg_map_flags_ops;
}

NVG_EXPORT const struct gbm_layout_ops *get_gbm_layout_ops(void)
{
	return &nvg_layout_ops;
}

NVG_EXPORT const struct gbm_alloc_flags_ops *get_gbm_alloc_flags_ops(void)
{
	return &nvg_alloc_flags_ops;
}
