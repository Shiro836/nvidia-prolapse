/* ext-duplicates: host reproducer for dEQP-VK.api.extension_duplicates.* (heap corruption seen in the
 * guest: "Scudo ERROR: corrupted chunk header" at process exit after creating an instance/device whose
 * ppEnabledExtensionNames repeats every name). The spec puts no uniqueness rule on
 * ppEnabledExtensionNames (validusage.json, VUID-*-ppEnabledExtensionNames-parameter only), so
 * duplicates are valid input and the CTS checks them.
 *
 * Loads either the raw ICD (--icd PATH: vk_icdGetInstanceProcAddr) or the HAL (--hal PATH: HMI,
 * open "vk0", GetInstanceProcAddr), enables every exposed instance extension --dup times, creates an
 * instance, then every exposed device extension --dup times, creates a device, destroys both, --loop
 * times. Build it with -fsanitize=address: an overflow in a blob's heap buffer lands in a redzone.
 *
 * usage: ext-duplicates (--icd ICD.so | --hal HAL.so) [--dup N] [--loop K] [--no-instance-dup]
 *        [--no-device-dup] [--skip name,name,...]                                                   */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include <hardware/hwvulkan.h>

static PFN_vkGetInstanceProcAddr gipa;

#ifdef __ANDROID__
/* libvndksupport's public prototype (system/core/libvndksupport/include/vndksupport/linker.h):
 * dlopen NAME by soname in the "sphal" namespace, the call AOSP libvulkan makes for vulkan.<hw>.so. */
typedef void *(*PFN_android_load_sphal_library)(const char *name, int flag);
static void *sphal_dlopen(const char *name) {
    void *vndk = dlopen("libvndksupport.so", RTLD_NOW | RTLD_LOCAL);
    if (!vndk) { printf("FATAL: dlopen libvndksupport.so: %s\n", dlerror()); exit(2); }
    PFN_android_load_sphal_library load = (PFN_android_load_sphal_library)dlsym(vndk, "android_load_sphal_library");
    if (!load) { printf("FATAL: libvndksupport.so has no android_load_sphal_library\n"); exit(2); }
    void *h = load(name, RTLD_NOW | RTLD_LOCAL);
    if (!h) { printf("FATAL: android_load_sphal_library(%s): %s\n", name, dlerror()); exit(2); }
    return h;
}
#endif
static void *load_lib(const char *path, int sphal) {
#ifdef __ANDROID__
    if (sphal) return sphal_dlopen(path);
#else
    (void)sphal;
#endif
    void *so = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!so) { printf("FATAL: dlopen %s: %s\n", path, dlerror()); exit(2); }
    return so;
}

/* Device extensions whose vkCreateDevice needs a library the guest does not ship (and, on the host,
 * libnvidia-rtcore / libcuda may be absent too). The HAL hides them; the raw ICD does not. Same list
 * as src/device.c hidden_device_exts_missing_lib. */
static const char *const default_skip[] = {
    "VK_KHR_acceleration_structure", "VK_KHR_ray_query", "VK_KHR_ray_tracing_pipeline",
    "VK_NV_cuda_kernel_launch", "VK_NV_optical_flow", "VK_NV_ray_tracing", "VK_NVX_binary_import",
    /* needs a VkDeviceQueueGlobalPriorityCreateInfo to matter; enabling alone is fine but keep the
     * raw-ICD run identical to what the HAL passes through */
    NULL,
};

static int skipped(const char *name, char **skip, size_t nskip) {
    for (size_t i = 0; i < nskip; i++)
        if (!strcmp(name, skip[i])) return 1;
    for (size_t i = 0; default_skip[i]; i++)
        if (!strcmp(name, default_skip[i])) return 1;
    return 0;
}

/* names[] gets every kept extension repeated `dup` times by pointer, like the CTS by_pointers case */
static const char **duplicate(const VkExtensionProperties *p, uint32_t n, unsigned dup, char **skip,
                              size_t nskip, uint32_t *out_n, uint32_t *out_unique) {
    const char **names = calloc((size_t)n * dup + 1, sizeof *names);
    if (!names) { fprintf(stderr, "oom\n"); exit(2); }
    uint32_t k = 0, u = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (skipped(p[i].extensionName, skip, nskip)) continue;
        u++;
        for (unsigned d = 0; d < dup; d++) names[k++] = p[i].extensionName;
    }
    *out_n = k;
    *out_unique = u;
    return names;
}

int main(int argc, char **argv) {
    const char *icd = NULL, *hal = NULL, *loader = NULL;
    unsigned dup = 3, loop = 1;
    int idup = 1, ddup = 1, sphal = 0;
    char *skip[64];
    size_t nskip = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--icd") && i + 1 < argc) icd = argv[++i];
        else if (!strcmp(argv[i], "--hal") && i + 1 < argc) hal = argv[++i];
        else if (!strcmp(argv[i], "--loader") && i + 1 < argc) loader = argv[++i];
        else if (!strcmp(argv[i], "--sphal")) sphal = 1; /* Android: --icd/--hal name is a soname, loaded via sphal */
        else if (!strcmp(argv[i], "--dup") && i + 1 < argc) dup = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--loop") && i + 1 < argc) loop = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-instance-dup")) idup = 0;
        else if (!strcmp(argv[i], "--no-device-dup")) ddup = 0;
        else if (!strcmp(argv[i], "--skip") && i + 1 < argc) {
            char *save = NULL;
            for (char *t = strtok_r(argv[++i], ",", &save); t && nskip < 64; t = strtok_r(NULL, ",", &save))
                skip[nskip++] = t;
        } else { fprintf(stderr, "bad arg %s\n", argv[i]); return 2; }
    }
    if ((!icd + !hal + !loader) != 2 || dup == 0) {
        fprintf(stderr, "usage: --icd ICD.so | --hal HAL.so | --loader libvulkan.so [--sphal] [--dup N]\n");
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);

    PFN_vkEnumerateInstanceExtensionProperties enum_iext;
    PFN_vkCreateInstance create_inst;
    if (icd || loader) {
        /* --loader: the platform Vulkan loader (AOSP libvulkan on Android, Khronos loader on the host);
         * it finds the driver itself (ro.hardware.vulkan / ICD manifests). */
        void *so = load_lib(icd ? icd : loader, sphal);
        gipa = (PFN_vkGetInstanceProcAddr)dlsym(so, icd ? "vk_icdGetInstanceProcAddr" : "vkGetInstanceProcAddr");
        if (!gipa) { printf("FATAL: no %s in %s\n", icd ? "vk_icdGetInstanceProcAddr" : "vkGetInstanceProcAddr",
                            icd ? icd : loader); return 2; }
        enum_iext = (PFN_vkEnumerateInstanceExtensionProperties)gipa(NULL, "vkEnumerateInstanceExtensionProperties");
        create_inst = (PFN_vkCreateInstance)gipa(NULL, "vkCreateInstance");
    } else {
        void *so = load_lib(hal, sphal);
        hwvulkan_module_t *hmi = dlsym(so, HAL_MODULE_INFO_SYM_AS_STR);
        if (!hmi || strcmp(hmi->common.id, HWVULKAN_HARDWARE_MODULE_ID)) { printf("FATAL: no HMI\n"); return 2; }
        hmi->common.dso = so;
        hwvulkan_device_t *h = NULL;
        if (hmi->common.methods->open(&hmi->common, HWVULKAN_DEVICE_0, (hw_device_t **)&h) || !h) {
            printf("FATAL: open vk0\n");
            return 2;
        }
        gipa = h->GetInstanceProcAddr;
        enum_iext = h->EnumerateInstanceExtensionProperties;
        create_inst = h->CreateInstance;
    }
    if (!enum_iext || !create_inst) { printf("FATAL: global entry points missing\n"); return 2; }

    for (unsigned it = 0; it < loop; it++) {
        uint32_t n = 0;
        if (enum_iext(NULL, &n, NULL) != VK_SUCCESS) { printf("FATAL: enumerate instance exts\n"); return 2; }
        VkExtensionProperties *ie = calloc(n, sizeof *ie);
        if (enum_iext(NULL, &n, ie) != VK_SUCCESS) { printf("FATAL: enumerate instance exts\n"); return 2; }
        uint32_t ni, ui;
        const char **inames = duplicate(ie, n, idup ? dup : 1, skip, nskip, &ni, &ui);
        VkApplicationInfo ai = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "ext-duplicates",
                                .pEngineName = "ext-duplicates", .apiVersion = VK_API_VERSION_1_3};
        VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai,
                                    .enabledExtensionCount = ni, .ppEnabledExtensionNames = inames};
        VkInstance inst = VK_NULL_HANDLE;
        VkResult r = create_inst(&ici, NULL, &inst);
        printf("[%u] vkCreateInstance(%u names = %u unique x %u) -> %d\n", it, ni, ui, idup ? dup : 1, (int)r);
        if (r != VK_SUCCESS) return 3;
        free(inames);
        free(ie);

        PFN_vkEnumeratePhysicalDevices enum_pd = (void *)gipa(inst, "vkEnumeratePhysicalDevices");
        PFN_vkEnumerateDeviceExtensionProperties enum_dext = (void *)gipa(inst, "vkEnumerateDeviceExtensionProperties");
        PFN_vkCreateDevice create_dev = (void *)gipa(inst, "vkCreateDevice");
        PFN_vkDestroyDevice destroy_dev = (void *)gipa(inst, "vkDestroyDevice");
        PFN_vkDestroyInstance destroy_inst = (void *)gipa(inst, "vkDestroyInstance");
        if (!enum_pd || !enum_dext || !create_dev || !destroy_dev || !destroy_inst) {
            printf("FATAL: instance entry points missing\n");
            return 2;
        }
        uint32_t npd = 1;
        VkPhysicalDevice pd = VK_NULL_HANDLE;
        r = enum_pd(inst, &npd, &pd);
        if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || !pd) { printf("FATAL: no physical device (%d)\n", (int)r); return 2; }

        n = 0;
        if (enum_dext(pd, NULL, &n, NULL) != VK_SUCCESS) { printf("FATAL: enumerate device exts\n"); return 2; }
        VkExtensionProperties *de = calloc(n, sizeof *de);
        if (enum_dext(pd, NULL, &n, de) != VK_SUCCESS) { printf("FATAL: enumerate device exts\n"); return 2; }
        uint32_t nd, ud;
        const char **dnames = duplicate(de, n, ddup ? dup : 1, skip, nskip, &nd, &ud);
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
                                       .pQueuePriorities = &prio};
        VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
                                  .pQueueCreateInfos = &qci, .enabledExtensionCount = nd,
                                  .ppEnabledExtensionNames = dnames};
        VkDevice dev = VK_NULL_HANDLE;
        r = create_dev(pd, &dci, NULL, &dev);
        printf("[%u] vkCreateDevice(%u names = %u unique x %u) -> %d\n", it, nd, ud, ddup ? dup : 1, (int)r);
        if (r != VK_SUCCESS) return 4;
        free(dnames);
        free(de);
        destroy_dev(dev, NULL);
        destroy_inst(inst, NULL);
        printf("[%u] destroyed\n", it);
    }
    printf("OK\n");
    return 0;
}
