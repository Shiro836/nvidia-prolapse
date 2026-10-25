/* vulkan.nvidia.so: the hwvulkan module, ICD bootstrap, instance-level entry points and the
 * GetInstanceProcAddr / GetDeviceProcAddr dispatch.
 *
 * NVIDIA's handles are returned to the loader unwrapped. Everything not in the intercept
 * tables below is resolved by the ICD itself, so the app calls the ICD directly. */
#include "nvhal.h"

#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#if defined(NVHAL_SOURCE_RM)
/* Source-built Mesa NVK using the Rust RM backend. Packaging selects this mode
 * explicitly; resolving another ICD is never an automatic fallback. */
#define NVHAL_ICD_DEFAULT "libvulkan_nouveau.so"
#define NVHAL_ICD_REMEDY "make source-built libvulkan_nouveau.so, libnv_bionic_native.so and " \
    "their libdrm/Android platform dependencies resolvable in the caller's linker namespace; " \
    "check the source-native mounts and matching host NVIDIA kernel device nodes"
#elif defined(__ANDROID__)
/* Resolved by soname in the caller's linker namespace: libvulkan loads this HAL with
 * android_load_sphal_library(), so that is the sphal namespace, where the ICD, its blobs
 * and the libnvglibc shims live. */
#define NVHAL_ICD_DEFAULT "libEGL_nvidia.so.0"
#define NVHAL_ICD_REMEDY "libEGL_nvidia.so.0, the matching libnvidia-* blobs and the libnvglibc " \
    "shims must be resolvable in the sphal linker namespace (/vendor_extra/lib64 with the host " \
    "driver bind-mounted); redeploy with dev/native-deploy and check the container mounts"
#elif defined(__i386__)
#define NVHAL_ICD_DEFAULT "/usr/lib32/libEGL_nvidia.so.0"
#define NVHAL_ICD_REMEDY "install the 32-bit NVIDIA driver userspace (lib32-nvidia-utils), or set NVHAL_ICD to " \
    "the path of the 32-bit libEGL_nvidia.so.0"
#else
#define NVHAL_ICD_DEFAULT "/usr/lib/libEGL_nvidia.so.0"
#define NVHAL_ICD_REMEDY "install the NVIDIA driver userspace, or set NVHAL_ICD to the path " \
    "of libEGL_nvidia.so.0"
#endif

/* Highest loader<->ICD interface version whose semantics this HAL implements: v5 (the ICD
 * accepts any VkApplicationInfo::apiVersion, which the HAL passes through unchanged). v6 and
 * v7 only add Windows adapter enumeration and direct-driver-loading entry points. */
#define NVHAL_ICD_INTERFACE_MAX 5u
#define NVHAL_ICD_INTERFACE_MIN 3u

typedef VkResult (VKAPI_PTR *PFN_icdNegotiate)(uint32_t *);

PFN_vkGetInstanceProcAddr nvhal_icd_gipa;
static void *icd_dl;
static unsigned open_refs;
static PFN_vkCreateInstance icd_CreateInstance;
static PFN_vkEnumerateInstanceExtensionProperties icd_EnumerateInstanceExtensionProperties;
static PFN_vkEnumerateInstanceVersion icd_EnumerateInstanceVersion;

static struct nvhal_map instances; /* VkInstance -> struct nvhal_instance* */
static struct nvhal_map physdevs;  /* VkPhysicalDevice -> struct nvhal_instance* */

/* ------------------------------------------------------------------ dispatchable word */

void nvhal_dispatch_check(void *obj, const void *parent, const char *what) {
    uintptr_t *word = obj;
    uintptr_t v = *word;
    if (v == (uintptr_t)HWVULKAN_DISPATCH_MAGIC) {
        NVHAL_TRACE("dispatch word: %s %p = 0x%" PRIxPTR " (ICD_LOADER_MAGIC already set by the ICD)",
                    what, obj, v);
        return;
    }
    /* A physical device / queue / command buffer the loader already tagged on an earlier
     * return carries the parent's loader data (driver.cpp SetData uses the instance's data
     * for physical devices, the device's for queues and command buffers). */
    if (parent && v == *(const uintptr_t *)parent) {
        NVHAL_TRACE("dispatch word: %s %p = 0x%" PRIxPTR " (already tagged by the loader)", what, obj, v);
        return;
    }
    if ((v & 0xffffffffu) == (uintptr_t)HWVULKAN_DISPATCH_MAGIC)
        NVHAL_ERR("dispatch word: %s %p = 0x%" PRIxPTR ": the ICD set only the low 32 bits of "
                  "ICD_LOADER_MAGIC; the Android loader compares the whole word, writing 0x%x",
                  what, obj, v, HWVULKAN_DISPATCH_MAGIC);
    else
        NVHAL_ERR("dispatch word: %s %p = 0x%" PRIxPTR " is not ICD_LOADER_MAGIC (the ICD did not "
                  "initialise the word the Khronos ICD contract reserves for the loader); writing 0x%x",
                  what, obj, v, HWVULKAN_DISPATCH_MAGIC);
    *word = (uintptr_t)HWVULKAN_DISPATCH_MAGIC;
}

/* ------------------------------------------------------------------ extension helpers */

bool nvhal_ext_in(const VkExtensionProperties *list, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(list[i].extensionName, name)) return true;
    return false;
}

VkResult nvhal_fill_ext_props(const VkExtensionProperties *src, uint32_t n, uint32_t *pCount,
                              VkExtensionProperties *pProps) {
    if (!pProps) {
        *pCount = n;
        return VK_SUCCESS;
    }
    uint32_t c = *pCount < n ? *pCount : n;
    memcpy(pProps, src, c * sizeof *src);
    *pCount = c;
    return c < n ? VK_INCOMPLETE : VK_SUCCESS;
}

/* Instance extensions that need a surface or a display the ICD cannot have here: the
 * Android loader implements VK_KHR_surface / VK_KHR_android_surface itself. */
static const char *const hidden_instance_exts[] = {
    "VK_KHR_surface", "VK_KHR_xcb_surface", "VK_KHR_xlib_surface", "VK_KHR_wayland_surface",
    "VK_KHR_display", "VK_KHR_get_display_properties2", "VK_EXT_direct_mode_display",
    "VK_EXT_display_surface_counter", "VK_KHR_get_surface_capabilities2",
    "VK_EXT_surface_maintenance1", "VK_KHR_surface_maintenance1", "VK_EXT_swapchain_colorspace",
    "VK_KHR_surface_protected_capabilities", "VK_EXT_headless_surface", "VK_NV_display_stereo",
};

static bool instance_ext_hidden(const char *name) {
    for (size_t i = 0; i < sizeof hidden_instance_exts / sizeof *hidden_instance_exts; i++)
        if (!strcmp(name, hidden_instance_exts[i])) return true;
    /* VK_EXT_acquire_xlib_display, VK_EXT_acquire_drm_display, ... */
    size_t l = strlen(name);
    return !strncmp(name, "VK_EXT_acquire_", 15) && l > 8 && !strcmp(name + l - 8, "_display");
}

/* The ICD's instance extension list (malloc'd), and the subset this HAL exposes. */
static VkResult icd_instance_exts(VkExtensionProperties **all, uint32_t *n_all,
                                  VkExtensionProperties **vis, uint32_t *n_vis) {
    VkResult r;
    VkExtensionProperties *p = NULL;
    uint32_t n = 0;
    do {
        r = icd_EnumerateInstanceExtensionProperties(NULL, &n, NULL);
        if (r != VK_SUCCESS) break;
        VkExtensionProperties *q = realloc(p, (n ? n : 1) * sizeof *p);
        if (!q) { r = VK_ERROR_OUT_OF_HOST_MEMORY; break; }
        p = q;
        r = icd_EnumerateInstanceExtensionProperties(NULL, &n, p);
    } while (r == VK_INCOMPLETE);
    if (r != VK_SUCCESS) {
        NVHAL_ERR("ICD vkEnumerateInstanceExtensionProperties -> %s", NVHAL_VKR(r));
        free(p);
        return r;
    }
    VkExtensionProperties *v = malloc((n ? n : 1) * sizeof *v);
    if (!v) { free(p); return VK_ERROR_OUT_OF_HOST_MEMORY; }
    uint32_t nv = 0;
    for (uint32_t i = 0; i < n; i++)
        if (!instance_ext_hidden(p[i].extensionName)) v[nv++] = p[i];
    *all = p; *n_all = n; *vis = v; *n_vis = nv;
    return VK_SUCCESS;
}

/* ------------------------------------------------------------------ global entry points */

static VkResult VKAPI_CALL hal_EnumerateInstanceExtensionProperties(const char *pLayerName,
                                                                    uint32_t *pCount,
                                                                    VkExtensionProperties *pProps) {
    if (pLayerName) return icd_EnumerateInstanceExtensionProperties(pLayerName, pCount, pProps);
    VkExtensionProperties *all, *vis;
    uint32_t n_all, n_vis;
    VkResult r = icd_instance_exts(&all, &n_all, &vis, &n_vis);
    if (r != VK_SUCCESS) return r;
    r = nvhal_fill_ext_props(vis, n_vis, pCount, pProps);
    free(all);
    free(vis);
    return r;
}

static VkResult VKAPI_CALL hal_EnumerateInstanceLayerProperties(uint32_t *pCount, VkLayerProperties *pProps) {
    (void)pProps;
    *pCount = 0; /* an ICD has no layers; the loader never asks (HAL-CONTRACT 6.3) */
    return VK_SUCCESS;
}

static VkResult VKAPI_CALL hal_EnumerateInstanceVersion(uint32_t *pApiVersion) {
    return icd_EnumerateInstanceVersion(pApiVersion);
}

static int api_minor_ge(uint32_t v, uint32_t want) {
    return VK_API_VERSION_MAJOR(v) > VK_API_VERSION_MAJOR(want) ||
           (VK_API_VERSION_MAJOR(v) == VK_API_VERSION_MAJOR(want) &&
            VK_API_VERSION_MINOR(v) >= VK_API_VERSION_MINOR(want));
}

/* Instance extensions a 1.0 instance needs so that the device can enable what
 * VK_ANDROID_native_buffer is built on (VK_EXT_image_drm_format_modifier and the
 * VK_KHR_external_*_fd family depend on them). Added only if the ICD lists them; a
 * device that then lacks its prerequisites fails loudly in vkCreateDevice. */
static const char *const legacy_instance_exts[] = {
    "VK_KHR_get_physical_device_properties2", "VK_KHR_external_memory_capabilities",
    "VK_KHR_external_semaphore_capabilities", "VK_KHR_external_fence_capabilities",
};

#define GIPA_INST(inst, name) nvhal_icd_gipa((inst), (name))

static VkResult VKAPI_CALL hal_CreateInstance(const VkInstanceCreateInfo *ci,
                                              const VkAllocationCallbacks *alloc,
                                              VkInstance *pInstance) {
    if (ci->enabledLayerCount) {
        NVHAL_ERR("vkCreateInstance: %u layer(s) requested from the driver; a Vulkan HAL has no "
                  "layers (the loader handles them) -> VK_ERROR_LAYER_NOT_PRESENT", ci->enabledLayerCount);
        return VK_ERROR_LAYER_NOT_PRESENT;
    }
    VkExtensionProperties *all, *vis;
    uint32_t n_all, n_vis;
    VkResult r = icd_instance_exts(&all, &n_all, &vis, &n_vis);
    if (r != VK_SUCCESS) return r;

    uint32_t api = (ci->pApplicationInfo && ci->pApplicationInfo->apiVersion)
                       ? ci->pApplicationInfo->apiVersion : VK_API_VERSION_1_0;
    const size_t n_legacy = sizeof legacy_instance_exts / sizeof *legacy_instance_exts;
    const char **names = malloc((ci->enabledExtensionCount + n_legacy + 1) * sizeof *names);
    if (!names) { free(all); free(vis); return VK_ERROR_OUT_OF_HOST_MEMORY; }
    uint32_t nn = 0;
    for (uint32_t i = 0; i < ci->enabledExtensionCount; i++) {
        const char *e = ci->ppEnabledExtensionNames[i];
        if (!nvhal_ext_in(vis, n_vis, e)) {
            NVHAL_ERR("vkCreateInstance: extension %s is not exposed by this driver%s -> "
                      "VK_ERROR_EXTENSION_NOT_PRESENT", e,
                      nvhal_ext_in(all, n_all, e) ? " (the ICD has it, but it needs a native "
                      "window system the Android loader replaces)" : "");
            free(names); free(all); free(vis);
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
        names[nn++] = e;
    }
    if (!api_minor_ge(api, VK_API_VERSION_1_1)) {
        for (size_t k = 0; k < n_legacy; k++) {
            bool have = false;
            for (uint32_t i = 0; i < nn; i++) have |= !strcmp(names[i], legacy_instance_exts[k]);
            if (!have && nvhal_ext_in(all, n_all, legacy_instance_exts[k])) {
                names[nn++] = legacy_instance_exts[k];
                NVHAL_TRACE("vkCreateInstance: apiVersion 1.0 instance, also enabling %s",
                            legacy_instance_exts[k]);
            }
        }
    }
    VkInstanceCreateInfo ici = *ci;
    ici.enabledExtensionCount = nn;
    ici.ppEnabledExtensionNames = names;
    VkInstance inst = VK_NULL_HANDLE;
    r = icd_CreateInstance(&ici, alloc, &inst);
    free(names);
    free(all);
    free(vis);
    if (r != VK_SUCCESS) {
        NVHAL_ERR("ICD vkCreateInstance(apiVersion %u.%u, %u extensions) -> %s",
                  VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api), nn, NVHAL_VKR(r));
        return r;
    }
    nvhal_dispatch_check(inst, NULL, "VkInstance");

    struct nvhal_instance *s = calloc(1, sizeof *s);
    if (!s) goto oom;
    s->handle = inst;
    s->api_version = api;
#define IFN(field, name) s->field = (void *)GIPA_INST(inst, name)
    IFN(DestroyInstance, "vkDestroyInstance");
    IFN(EnumeratePhysicalDevices, "vkEnumeratePhysicalDevices");
    IFN(EnumeratePhysicalDeviceGroups, "vkEnumeratePhysicalDeviceGroups");
    IFN(EnumeratePhysicalDeviceGroupsKHR, "vkEnumeratePhysicalDeviceGroupsKHR");
    IFN(EnumerateDeviceExtensionProperties, "vkEnumerateDeviceExtensionProperties");
    IFN(CreateDevice, "vkCreateDevice");
    IFN(GetPhysicalDeviceProperties, "vkGetPhysicalDeviceProperties");
    IFN(GetPhysicalDeviceProperties2, "vkGetPhysicalDeviceProperties2");
    IFN(GetPhysicalDeviceProperties2KHR, "vkGetPhysicalDeviceProperties2KHR");
    IFN(GetPhysicalDeviceMemoryProperties, "vkGetPhysicalDeviceMemoryProperties");
    IFN(GetDeviceProcAddr, "vkGetDeviceProcAddr");
    if (api_minor_ge(api, VK_API_VERSION_1_1)) {
        IFN(GetPhysicalDeviceImageFormatProperties2, "vkGetPhysicalDeviceImageFormatProperties2");
        IFN(GetPhysicalDeviceFormatProperties2, "vkGetPhysicalDeviceFormatProperties2");
        IFN(GetPhysicalDeviceExternalBufferProperties, "vkGetPhysicalDeviceExternalBufferProperties");
    } else {
        IFN(GetPhysicalDeviceImageFormatProperties2, "vkGetPhysicalDeviceImageFormatProperties2KHR");
        IFN(GetPhysicalDeviceFormatProperties2, "vkGetPhysicalDeviceFormatProperties2KHR");
        IFN(GetPhysicalDeviceExternalBufferProperties, "vkGetPhysicalDeviceExternalBufferPropertiesKHR");
    }
    IFN(GPDIFP2_core, "vkGetPhysicalDeviceImageFormatProperties2");
    IFN(GPDIFP2_khr, "vkGetPhysicalDeviceImageFormatProperties2KHR");
    IFN(GPDEBP_core, "vkGetPhysicalDeviceExternalBufferProperties");
    IFN(GPDEBP_khr, "vkGetPhysicalDeviceExternalBufferPropertiesKHR");
    IFN(GetPhysicalDeviceFeatures, "vkGetPhysicalDeviceFeatures");
    IFN(GPDF2_core, "vkGetPhysicalDeviceFeatures2");
    IFN(GPDF2_khr, "vkGetPhysicalDeviceFeatures2KHR");
    IFN(GetPhysicalDeviceFormatProperties, "vkGetPhysicalDeviceFormatProperties");
    IFN(GPDFP2_core, "vkGetPhysicalDeviceFormatProperties2");
    IFN(GPDFP2_khr, "vkGetPhysicalDeviceFormatProperties2KHR");
    IFN(GetPhysicalDeviceImageFormatProperties, "vkGetPhysicalDeviceImageFormatProperties");
    IFN(GetPhysicalDeviceQueueFamilyProperties, "vkGetPhysicalDeviceQueueFamilyProperties");
#undef IFN
    {
        const char *missing = !s->DestroyInstance ? "vkDestroyInstance"
            : !s->EnumeratePhysicalDevices ? "vkEnumeratePhysicalDevices"
            : !s->EnumerateDeviceExtensionProperties ? "vkEnumerateDeviceExtensionProperties"
            : !s->CreateDevice ? "vkCreateDevice"
            : !s->GetPhysicalDeviceProperties ? "vkGetPhysicalDeviceProperties"
            : !s->GetPhysicalDeviceMemoryProperties ? "vkGetPhysicalDeviceMemoryProperties"
            : !s->GetPhysicalDeviceFeatures ? "vkGetPhysicalDeviceFeatures"
            : !s->GetPhysicalDeviceFormatProperties ? "vkGetPhysicalDeviceFormatProperties"
            : !s->GetPhysicalDeviceImageFormatProperties ? "vkGetPhysicalDeviceImageFormatProperties"
            : !s->GetPhysicalDeviceQueueFamilyProperties ? "vkGetPhysicalDeviceQueueFamilyProperties"
            : !s->GetDeviceProcAddr ? "vkGetDeviceProcAddr" : NULL;
        if (missing) {
            NVHAL_ERR("ICD vk_icdGetInstanceProcAddr(instance, \"%s\") returned NULL; the ICD is "
                      "unusable -> VK_ERROR_INCOMPATIBLE_DRIVER", missing);
            PFN_vkDestroyInstance d = (PFN_vkDestroyInstance)GIPA_INST(inst, "vkDestroyInstance");
            if (d) d(inst, alloc);
            free(s);
            return VK_ERROR_INCOMPATIBLE_DRIVER;
        }
    }
    pthread_mutex_lock(&nvhal_lock);
    bool ok = nvhal_map_put(&instances, inst, s);
    pthread_mutex_unlock(&nvhal_lock);
    if (!ok) { free(s); goto oom; }
    NVHAL_TRACE("vkCreateInstance: VkInstance %p, apiVersion %u.%u, %u extension(s) passed to the ICD",
                (void *)inst, VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api), nn);
    *pInstance = inst;
    return VK_SUCCESS;
oom:
    NVHAL_ERR("vkCreateInstance: out of host memory for the HAL's instance state");
    {
        PFN_vkDestroyInstance d = (PFN_vkDestroyInstance)GIPA_INST(inst, "vkDestroyInstance");
        if (d) d(inst, alloc);
    }
    return VK_ERROR_OUT_OF_HOST_MEMORY;
}

struct nvhal_instance *nvhal_instance_get(VkInstance instance) {
    pthread_mutex_lock(&nvhal_lock);
    struct nvhal_instance *s = nvhal_map_get(&instances, instance);
    pthread_mutex_unlock(&nvhal_lock);
    return s;
}

struct nvhal_instance *nvhal_physdev_instance(VkPhysicalDevice pd) {
    pthread_mutex_lock(&nvhal_lock);
    struct nvhal_instance *s = nvhal_map_get(&physdevs, pd);
    pthread_mutex_unlock(&nvhal_lock);
    return s;
}

static void VKAPI_CALL hal_DestroyInstance(VkInstance instance, const VkAllocationCallbacks *alloc) {
    if (instance == VK_NULL_HANDLE) return;
    pthread_mutex_lock(&nvhal_lock);
    struct nvhal_instance *s = nvhal_map_del(&instances, instance);
    if (s) nvhal_map_del_val(&physdevs, s);
    pthread_mutex_unlock(&nvhal_lock);
    if (!s) {
        NVHAL_ERR("vkDestroyInstance(%p): not an instance created through this HAL; not destroyed",
                  (void *)instance);
        return;
    }
    s->DestroyInstance(instance, alloc);
    nvhal_astc_instance_teardown(s);
    free(s);
}

static void record_physdevs(struct nvhal_instance *s, VkPhysicalDevice *pds, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) nvhal_dispatch_check(pds[i], s->handle, "VkPhysicalDevice");
    pthread_mutex_lock(&nvhal_lock);
    for (uint32_t i = 0; i < n; i++) {
        struct nvhal_instance *prev = nvhal_map_get(&physdevs, pds[i]);
        /* measured on 615.71.09: each instance gets its own VkPhysicalDevice objects */
        if (prev && prev != s)
            NVHAL_ERR("VkPhysicalDevice %p is shared by instances %p and %p; this HAL keys physical "
                      "devices to one instance, so the older instance loses it", (void *)pds[i],
                      (void *)prev->handle, (void *)s->handle);
        if (!nvhal_map_put(&physdevs, pds[i], s))
            NVHAL_ERR("out of host memory recording VkPhysicalDevice %p", (void *)pds[i]);
    }
    pthread_mutex_unlock(&nvhal_lock);
}

static VkResult VKAPI_CALL hal_EnumeratePhysicalDevices(VkInstance instance, uint32_t *pCount,
                                                        VkPhysicalDevice *pDevs) {
    struct nvhal_instance *s = nvhal_instance_get(instance);
    if (!s) {
        NVHAL_ERR("vkEnumeratePhysicalDevices(%p): unknown instance", (void *)instance);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    VkResult r = s->EnumeratePhysicalDevices(instance, pCount, pDevs);
    if ((r == VK_SUCCESS || r == VK_INCOMPLETE) && pDevs) record_physdevs(s, pDevs, *pCount);
    return r;
}

static VkResult enumerate_groups(VkInstance instance, uint32_t *pCount,
                                 VkPhysicalDeviceGroupProperties *pProps, bool khr) {
    struct nvhal_instance *s = nvhal_instance_get(instance);
    if (!s) {
        NVHAL_ERR("vkEnumeratePhysicalDeviceGroups(%p): unknown instance", (void *)instance);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    PFN_vkEnumeratePhysicalDeviceGroups f = khr ? s->EnumeratePhysicalDeviceGroupsKHR
                                                : s->EnumeratePhysicalDeviceGroups;
    VkResult r = f(instance, pCount, pProps);
    if ((r == VK_SUCCESS || r == VK_INCOMPLETE) && pProps)
        for (uint32_t g = 0; g < *pCount; g++)
            record_physdevs(s, pProps[g].physicalDevices, pProps[g].physicalDeviceCount);
    return r;
}
static VkResult VKAPI_CALL hal_EnumeratePhysicalDeviceGroups(VkInstance i, uint32_t *c,
                                                             VkPhysicalDeviceGroupProperties *p) {
    return enumerate_groups(i, c, p, false);
}
static VkResult VKAPI_CALL hal_EnumeratePhysicalDeviceGroupsKHR(VkInstance i, uint32_t *c,
                                                                VkPhysicalDeviceGroupProperties *p) {
    return enumerate_groups(i, c, p, true);
}

/* ------------------------------------------------------------------ proc addr dispatch */

static PFN_vkVoidFunction VKAPI_CALL hal_GetInstanceProcAddr(VkInstance instance, const char *pName);
static PFN_vkVoidFunction VKAPI_CALL hal_GetDeviceProcAddr(VkDevice device, const char *pName);

#define ANY_LEVEL NVHAL_IC_ANY
#define DEVICE_LEVEL NVHAL_IC_DEVICE
#define ANB_CMD NVHAL_IC_ANB
#define AHB_CMD NVHAL_IC_AHB   /* intercepted on devices with the AHB extension; ICD's own elsewhere */
#define AHB_ONLY NVHAL_IC_AHB_ONLY /* the extension's own commands (no ICD counterpart) */
#define WRAP_CMD NVHAL_IC_WRAP /* must see wrapper or emulated ASTC images: devices with AHB, ANB or ASTC emulation */
#define ASTC_CMD NVHAL_IC_ASTC /* the compute-state shadow (cmdstate.c): devices with ASTC emulation */
struct intercept {
    const char *name;
    PFN_vkVoidFunction fn;
    int kind;
    /* ICD entry point that must exist for this one to be offered (NULL: always) */
    const char *needs;
};

#define F(x) ((PFN_vkVoidFunction)(x))
static const struct intercept intercepts[] = {
    /* instance level */
    {"vkGetInstanceProcAddr", F(hal_GetInstanceProcAddr), ANY_LEVEL, NULL},
    {"vkDestroyInstance", F(hal_DestroyInstance), ANY_LEVEL, NULL},
    {"vkEnumeratePhysicalDevices", F(hal_EnumeratePhysicalDevices), ANY_LEVEL, NULL},
    {"vkEnumeratePhysicalDeviceGroups", F(hal_EnumeratePhysicalDeviceGroups), ANY_LEVEL, "vkEnumeratePhysicalDeviceGroups"},
    {"vkEnumeratePhysicalDeviceGroupsKHR", F(hal_EnumeratePhysicalDeviceGroupsKHR), ANY_LEVEL, "vkEnumeratePhysicalDeviceGroupsKHR"},
    {"vkEnumerateDeviceExtensionProperties", F(nvhal_EnumerateDeviceExtensionProperties), ANY_LEVEL, NULL},
    {"vkCreateDevice", F(nvhal_CreateDevice), ANY_LEVEL, NULL},
    {"vkGetPhysicalDeviceProperties2", F(nvhal_GetPhysicalDeviceProperties2), ANY_LEVEL, "vkGetPhysicalDeviceProperties2"},
    {"vkGetPhysicalDeviceProperties2KHR", F(nvhal_GetPhysicalDeviceProperties2KHR), ANY_LEVEL, "vkGetPhysicalDeviceProperties2KHR"},
    {"vkGetPhysicalDeviceImageFormatProperties2", F(nvhal_GetPhysicalDeviceImageFormatProperties2), ANY_LEVEL, "vkGetPhysicalDeviceImageFormatProperties2"},
    {"vkGetPhysicalDeviceImageFormatProperties2KHR", F(nvhal_GetPhysicalDeviceImageFormatProperties2KHR), ANY_LEVEL, "vkGetPhysicalDeviceImageFormatProperties2KHR"},
    {"vkGetPhysicalDeviceExternalBufferProperties", F(nvhal_GetPhysicalDeviceExternalBufferProperties), ANY_LEVEL, "vkGetPhysicalDeviceExternalBufferProperties"},
    {"vkGetPhysicalDeviceExternalBufferPropertiesKHR", F(nvhal_GetPhysicalDeviceExternalBufferPropertiesKHR), ANY_LEVEL, "vkGetPhysicalDeviceExternalBufferPropertiesKHR"},
    /* ASTC LDR emulation (astc.c): the feature bit and the ASTC formats' properties */
    {"vkGetPhysicalDeviceFeatures", F(nvhal_GetPhysicalDeviceFeatures), ANY_LEVEL, NULL},
    {"vkGetPhysicalDeviceFeatures2", F(nvhal_GetPhysicalDeviceFeatures2), ANY_LEVEL, "vkGetPhysicalDeviceFeatures2"},
    {"vkGetPhysicalDeviceFeatures2KHR", F(nvhal_GetPhysicalDeviceFeatures2KHR), ANY_LEVEL, "vkGetPhysicalDeviceFeatures2KHR"},
    {"vkGetPhysicalDeviceFormatProperties", F(nvhal_GetPhysicalDeviceFormatProperties), ANY_LEVEL, NULL},
    {"vkGetPhysicalDeviceFormatProperties2", F(nvhal_GetPhysicalDeviceFormatProperties2), ANY_LEVEL, "vkGetPhysicalDeviceFormatProperties2"},
    {"vkGetPhysicalDeviceFormatProperties2KHR", F(nvhal_GetPhysicalDeviceFormatProperties2KHR), ANY_LEVEL, "vkGetPhysicalDeviceFormatProperties2KHR"},
    {"vkGetPhysicalDeviceImageFormatProperties", F(nvhal_GetPhysicalDeviceImageFormatProperties), ANY_LEVEL, NULL},
    /* device level (also reachable through GIPA(instance, name)) */
    {"vkGetDeviceProcAddr", F(hal_GetDeviceProcAddr), DEVICE_LEVEL, NULL},
    {"vkDestroyDevice", F(nvhal_DestroyDevice), DEVICE_LEVEL, NULL},
    {"vkGetDeviceQueue", F(nvhal_GetDeviceQueue), DEVICE_LEVEL, NULL},
    {"vkGetDeviceQueue2", F(nvhal_GetDeviceQueue2), DEVICE_LEVEL, "vkGetDeviceQueue2"},
    {"vkAllocateCommandBuffers", F(nvhal_AllocateCommandBuffers), DEVICE_LEVEL, NULL},
    {"vkCreateImage", F(nvhal_CreateImage), DEVICE_LEVEL, NULL},
    {"vkDestroyImage", F(nvhal_DestroyImage), DEVICE_LEVEL, NULL},
    {"vkBindImageMemory2", F(nvhal_BindImageMemory2), DEVICE_LEVEL, "vkBindImageMemory2"},
    {"vkBindImageMemory2KHR", F(nvhal_BindImageMemory2KHR), DEVICE_LEVEL, "vkBindImageMemory2KHR"},
    {"vkGetSwapchainGrallocUsageANDROID", F(nvhal_GetSwapchainGrallocUsageANDROID), ANB_CMD, NULL},
    {"vkGetSwapchainGrallocUsage2ANDROID", F(nvhal_GetSwapchainGrallocUsage2ANDROID), ANB_CMD, NULL},
    {"vkAcquireImageANDROID", F(nvhal_AcquireImageANDROID), ANB_CMD, NULL},
    {"vkQueueSignalReleaseImageANDROID", F(nvhal_QueueSignalReleaseImageANDROID), ANB_CMD, NULL},
    /* VK_ANDROID_external_memory_android_hardware_buffer (ahb.c); the VkImage translators are
     * imgxlate.c's nvhal_xl_entries */
    {"vkGetAndroidHardwareBufferPropertiesANDROID", F(nvhal_GetAndroidHardwareBufferPropertiesANDROID), AHB_ONLY, NULL},
    {"vkGetMemoryAndroidHardwareBufferANDROID", F(nvhal_GetMemoryAndroidHardwareBufferANDROID), AHB_ONLY, NULL},
    /* WRAP: a dedicated allocation naming a swapchain-bound wrapper must be refused, not passed on */
    {"vkAllocateMemory", F(nvhal_AllocateMemory), WRAP_CMD, NULL},
    {"vkFreeMemory", F(nvhal_FreeMemory), AHB_CMD, NULL},
    {"vkBindImageMemory", F(nvhal_BindImageMemory), WRAP_CMD, NULL},
    {"vkCreateBuffer", F(nvhal_CreateBuffer), AHB_CMD, NULL},
    {"vkCreateSamplerYcbcrConversion", F(nvhal_CreateSamplerYcbcrConversion), AHB_CMD, "vkCreateSamplerYcbcrConversion"},
    {"vkCreateSamplerYcbcrConversionKHR", F(nvhal_CreateSamplerYcbcrConversionKHR), AHB_CMD, "vkCreateSamplerYcbcrConversionKHR"},
};

static bool find_intercept(const char *name, struct intercept *out) {
    for (size_t i = 0; i < sizeof intercepts / sizeof *intercepts; i++)
        if (!strcmp(intercepts[i].name, name)) { *out = intercepts[i]; return true; }
    for (const struct nvhal_xl_entry *e = nvhal_xl_entries; e->name; e++)
        if (!strcmp(e->name, name)) {
            *out = (struct intercept){e->name, e->fn, e->kind, e->name}; /* the ICD must have it */
            return true;
        }
    for (const struct nvhal_xl_entry *e = nvhal_cs_entries; e->name; e++)
        if (!strcmp(e->name, name)) {
            *out = (struct intercept){e->name, e->fn, e->kind, e->name};
            return true;
        }
    return false;
}

static PFN_vkVoidFunction global_proc(const char *name) {
    if (!strcmp(name, "vkCreateInstance")) return F(hal_CreateInstance);
    if (!strcmp(name, "vkEnumerateInstanceExtensionProperties")) return F(hal_EnumerateInstanceExtensionProperties);
    if (!strcmp(name, "vkEnumerateInstanceLayerProperties")) return F(hal_EnumerateInstanceLayerProperties);
    if (!strcmp(name, "vkEnumerateInstanceVersion"))
        return icd_EnumerateInstanceVersion ? F(hal_EnumerateInstanceVersion) : NULL;
    return NULL;
}

static PFN_vkVoidFunction VKAPI_CALL hal_GetInstanceProcAddr(VkInstance instance, const char *pName) {
    if (!pName) return NULL;
    PFN_vkVoidFunction g = global_proc(pName);
    if (instance == VK_NULL_HANDLE) {
        if (g) return g;
        if (!strcmp(pName, "vkGetInstanceProcAddr")) return F(hal_GetInstanceProcAddr);
        return NULL;
    }
    /* Never hand out the ICD's own global commands: vkCreateInstance must stay ours. */
    if (g) return g;
    struct intercept ic;
    if (find_intercept(pName, &ic)) {
        if (ic.kind == AHB_ONLY) return nvhal_ahb_available ? ic.fn : NULL;
        if (ic.needs && !nvhal_icd_gipa(instance, ic.needs)) return NULL;
        return ic.fn;
    }
    return nvhal_icd_gipa(instance, pName);
}

static PFN_vkVoidFunction device_proc(struct nvhal_device *d, VkDevice device, const char *pName);

static PFN_vkVoidFunction VKAPI_CALL hal_GetDeviceProcAddr(VkDevice device, const char *pName) {
    if (!pName) return NULL;
    if (!strcmp(pName, "vkGetDeviceProcAddr")) return F(hal_GetDeviceProcAddr);
    struct nvhal_device *d = nvhal_device_get(device);
    if (!d) {
        NVHAL_ERR("vkGetDeviceProcAddr(%p, %s): not a device created through this HAL", (void *)device, pName);
        return NULL;
    }
    PFN_vkVoidFunction fn = device_proc(d, device, pName);
    /* opt-in GPU timing (gputime.c): off unless the device has it, then fn is returned unchanged */
    return d->gt ? nvhal_gt_wrap(d, pName, fn) : fn;
}

static PFN_vkVoidFunction device_proc(struct nvhal_device *d, VkDevice device, const char *pName) {
    struct intercept ic;
    bool found = find_intercept(pName, &ic);
    if (found && ic.kind == ANB_CMD) return d->anb_enabled ? ic.fn : NULL;
    if (found && ic.kind == AHB_ONLY) return d->ahb_enabled ? ic.fn : NULL;
    /* without the AHB extension no AHB image / memory can exist on this device, and without it and
     * VK_ANDROID_native_buffer no wrapper image at all: the ICD's own entry point */
    if (found && ic.kind == AHB_CMD && !d->ahb_enabled) return d->GetDeviceProcAddr(device, pName);
    if (found && ic.kind == WRAP_CMD && !d->ahb_enabled && !d->anb_enabled && !d->astc)
        return d->GetDeviceProcAddr(device, pName);
    /* without ASTC emulation nothing records a dispatch into the app's command buffers: nothing to shadow */
    if (found && ic.kind == ASTC_CMD && !d->astc) return d->GetDeviceProcAddr(device, pName);
    if (found && (ic.kind == DEVICE_LEVEL || ic.kind == AHB_CMD || ic.kind == WRAP_CMD || ic.kind == ASTC_CMD)) {
        if (ic.needs && !d->GetDeviceProcAddr(device, ic.needs)) return NULL;
        return ic.fn;
    }
    /* instance-level commands are not device commands: GDPA must return NULL for them */
    if (found || global_proc(pName)) return NULL;
    return d->GetDeviceProcAddr(device, pName);
}
#undef F

/* ------------------------------------------------------------------ hwvulkan module */

static int load_icd(void) {
    const char *path = NVHAL_ICD_DEFAULT;
#ifndef __ANDROID__
    const char *env = getenv("NVHAL_ICD");
    if (env && *env) path = env;
#endif
    dlerror();
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        const char *e = dlerror();
        NVHAL_ERR("cannot load the NVIDIA Vulkan ICD: dlopen(%s): %s. Remedy: %s", path,
                  e ? e : "(no dlerror)", NVHAL_ICD_REMEDY);
        return -ENOENT;
    }
    PFN_icdNegotiate neg = (PFN_icdNegotiate)dlsym(h, "vk_icdNegotiateLoaderICDInterfaceVersion");
    PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)dlsym(h, "vk_icdGetInstanceProcAddr");
    if (!neg || !gipa) {
        NVHAL_ERR("%s does not export %s: not a Vulkan ICD of loader interface >= %u. Remedy: %s",
                  path, !neg ? "vk_icdNegotiateLoaderICDInterfaceVersion" : "vk_icdGetInstanceProcAddr",
                  NVHAL_ICD_INTERFACE_MIN, NVHAL_ICD_REMEDY);
        dlclose(h);
        return -EINVAL;
    }
    uint32_t ver = NVHAL_ICD_INTERFACE_MAX;
    VkResult r = neg(&ver);
    if (r != VK_SUCCESS || ver < NVHAL_ICD_INTERFACE_MIN) {
        NVHAL_ERR("%s: vk_icdNegotiateLoaderICDInterfaceVersion(%u) -> %s, version %u; this HAL "
                  "needs %u..%u. Remedy: use a driver whose ICD supports loader interface >= %u",
                  path, NVHAL_ICD_INTERFACE_MAX, NVHAL_VKR(r), ver, NVHAL_ICD_INTERFACE_MIN,
                  NVHAL_ICD_INTERFACE_MAX, NVHAL_ICD_INTERFACE_MIN);
        dlclose(h);
        return -EINVAL;
    }
    PFN_vkCreateInstance ci = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
    PFN_vkEnumerateInstanceExtensionProperties eie =
        (PFN_vkEnumerateInstanceExtensionProperties)gipa(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties");
    if (!ci || !eie) {
        NVHAL_ERR("%s: vk_icdGetInstanceProcAddr(NULL, \"%s\") returned NULL; the ICD is unusable",
                  path, !ci ? "vkCreateInstance" : "vkEnumerateInstanceExtensionProperties");
        dlclose(h);
        return -EINVAL;
    }
    icd_dl = h;
    nvhal_icd_gipa = gipa;
    icd_CreateInstance = ci;
    icd_EnumerateInstanceExtensionProperties = eie;
    icd_EnumerateInstanceVersion = (PFN_vkEnumerateInstanceVersion)gipa(VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
    Dl_info di;
    NVHAL_TRACE("ICD %s (%s) loaded, loader<->ICD interface version %u",
                path, dladdr((void *)gipa, &di) && di.dli_fname ? di.dli_fname : "?", ver);
    return 0;
}

static int hal_close(struct hw_device_t *dev);
extern hwvulkan_module_t HAL_MODULE_INFO_SYM;

static hwvulkan_device_t hal_device = {
    .common = {
        .tag = HARDWARE_DEVICE_TAG,
        .version = HWVULKAN_DEVICE_API_VERSION_0_1,
        .module = &HAL_MODULE_INFO_SYM.common,
        .close = hal_close,
    },
    .EnumerateInstanceExtensionProperties = hal_EnumerateInstanceExtensionProperties,
    .CreateInstance = hal_CreateInstance,
    .GetInstanceProcAddr = hal_GetInstanceProcAddr,
};

static int hal_open(const struct hw_module_t *module, const char *id, struct hw_device_t **device) {
    (void)module;
    nvhal_log_init();
    if (!id || strcmp(id, HWVULKAN_DEVICE_0) != 0) {
        NVHAL_ERR("open(\"%s\"): unknown device id (only \"" HWVULKAN_DEVICE_0 "\")", id ? id : "(null)");
        return -ENOENT;
    }
    int rc = 0;
    pthread_mutex_lock(&nvhal_lock);
    if (!icd_dl) rc = load_icd();
    if (rc == 0) {
        open_refs++;
        nvhal_ahb_load();
    }
    pthread_mutex_unlock(&nvhal_lock);
    if (rc != 0) return rc;
    *device = &hal_device.common;
    return 0;
}

/* close() never unloads the ICD: once loaded it stays for the life of the process (as the gralloc backend keeps it).
 * NVIDIA's libEGL_nvidia keeps a dlopen() handle of its own (its "libEGL_nvidia.so.0" string; the dlopen and dlclose
 * wrappers at ELF 0xb80e0 / 0xb8140 in 615.78.08) and closes it from the exit-time destructor it registers with
 * __cxa_atexit (ELF 0x316d0 registers 0x9f950; 0x9f950 -> 0x33e70 -> 0x33ea0 -> 0xb8140). With our reference gone,
 * that dlclose drops the last one and bionic unmaps the library while its destructor is still running: the process
 * dies with SIGSEGV at exit, on the return address after the dlclose call (measured in the guest: HAL open, instance,
 * close, exit -> SIGSEGV at libEGL_nvidia+0xb8178; without the close: exit 0). Android's libvulkan closes the device
 * when an updated driver namespace replaces the built-in driver (frameworks/native android13-release
 * vulkan/libvulkan/driver.cpp Hal::UnloadBuiltinDriver), so the path is reachable from apps too. */
static int hal_close(struct hw_device_t *dev) {
    (void)dev;
    pthread_mutex_lock(&nvhal_lock);
    if (open_refs && --open_refs == 0) {
        if (instances.n) {
            NVHAL_ERR("close(): %zu VkInstance(s) still alive", instances.n);
        } else {
            nvhal_map_free(&instances);
            nvhal_map_free(&physdevs);
        }
    }
    pthread_mutex_unlock(&nvhal_lock);
    return 0;
}

static struct hw_module_methods_t hal_module_methods = {.open = hal_open};

/* The loader writes HMI.common.dso (driver.cpp:209), so this must stay writable. */
__attribute__((visibility("default"))) hwvulkan_module_t HAL_MODULE_INFO_SYM = {
    .common = {
        .tag = HARDWARE_MODULE_TAG,
        .module_api_version = HWVULKAN_MODULE_API_VERSION_0_1,
        .hal_api_version = HARDWARE_HAL_API_VERSION,
        .id = HWVULKAN_HARDWARE_MODULE_ID,
        .name = "NVIDIA Vulkan HAL (waydroid-nvidia-prolapse)",
        .author = "waydroid-nvidia-prolapse",
        .methods = &hal_module_methods,
    },
};
