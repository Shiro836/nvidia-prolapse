/* loader-pd: host test of private data on objects the Android loader owns (VkSwapchainKHR,
 * VkSurfaceKHR), through vulkan.nvidia.so on the real GPU.
 *
 * dEQP-VK.wsi.android.swapchain.private_data.* (vktWsiSwapchainTests.cpp:748-870) calls
 * vkSet/GetPrivateData[EXT] with VK_OBJECT_TYPE_SWAPCHAIN_KHR and the loader's swapchain handle; the
 * Android 13 loader has no private-data handling, so the call reaches the HAL with a handle the ICD
 * never issued. The HAL must keep those values itself (Mesa vk_object.c does the same on Android).
 *
 * Loads the HAL like AOSP libvulkan, creates a Vulkan 1.3 device with VK_ANDROID_native_buffer and
 * VK_EXT_private_data (privateData enabled), and checks with a fake swapchain handle 0x5C:
 *   - every private-data entry point vkGetDeviceProcAddr hands out lives in the HAL (dladdr);
 *   - never-set reads 0, core and EXT entry points see the same values, swapchain and surface
 *     with the same handle value are distinct, values survive interleaving;
 *   - a destroyed slot's values are gone (a new slot reads 0, whether or not its handle is recycled);
 *   - a real VkImage still round-trips through the ICD;
 *   - informational control: the raw ICD with the same fake handle, in a forked child.
 * Exit 0 only when every check passed. No process in this test ever creates a wrapper image, so the
 * HAL's "no wrappers" fast path is the one exercised.
 *
 * usage: loader-pd HAL.so */
#include <dlfcn.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vulkan/vulkan.h>
#include <hardware/hwvulkan.h>
#include <vulkan/vk_android_native_buffer.h>

static int failures;
static void check(int ok, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void check(int ok, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    printf("  [%s] ", ok ? "PASS" : "FAIL");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
    if (!ok) failures++;
}
#define MUST(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { printf("FATAL: %s -> %d\n", #call, r_); exit(2); } } while (0)

static PFN_vkGetInstanceProcAddr gipa;
static PFN_vkGetDeviceProcAddr gdpa;
static VkInstance inst;
static VkDevice dev;
static const char *halpath;

static void *ip(const char *name) {
    void *p = (void *)gipa(inst, name);
    if (!p) { printf("FATAL: GIPA(instance, %s) = NULL\n", name); exit(2); }
    return p;
}
/* a device entry point that must be the HAL's own */
static void *dp_hal(const char *name) {
    void *p = (void *)gdpa(dev, name);
    if (!p) { printf("FATAL: GDPA(device, %s) = NULL\n", name); exit(2); }
    Dl_info di;
    int in_hal = dladdr(p, &di) && di.dli_fname && strstr(di.dli_fname, "vulkan.nvidia.so");
    check(in_hal, "GDPA(device, %s) = %p in %s", name, p, in_hal ? di.dli_fname : (di.dli_fname ? di.dli_fname : "?"));
    return p;
}

int main(int argc, char **argv) {
    halpath = argc > 1 ? argv[1] : "out/host/vulkan.nvidia.so";
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("loader-pd: HAL %s\n", halpath);
    void *so = dlopen(halpath, RTLD_NOW | RTLD_LOCAL);
    if (!so) { printf("FATAL: dlopen %s: %s\n", halpath, dlerror()); return 2; }
    hwvulkan_module_t *hmi = dlsym(so, HAL_MODULE_INFO_SYM_AS_STR);
    if (!hmi) { printf("FATAL: no HMI\n"); return 2; }
    hmi->common.dso = so;
    hwvulkan_device_t *hal = NULL;
    if (hmi->common.methods->open(&hmi->common, HWVULKAN_DEVICE_0, (hw_device_t **)&hal) || !hal) {
        printf("FATAL: open(vk0)\n");
        return 2;
    }
    gipa = hal->GetInstanceProcAddr;

    VkApplicationInfo ai = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "loader-pd",
                            .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai};
    MUST(hal->CreateInstance(&ici, NULL, &inst));
    gdpa = ip("vkGetDeviceProcAddr");
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices = ip("vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties = ip("vkGetPhysicalDeviceProperties");
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties =
        ip("vkGetPhysicalDeviceQueueFamilyProperties");
    PFN_vkCreateDevice CreateDevice = ip("vkCreateDevice");
    PFN_vkDestroyInstance DestroyInstance = ip("vkDestroyInstance");

    uint32_t npd = 0;
    MUST(EnumeratePhysicalDevices(inst, &npd, NULL));
    VkPhysicalDevice *pds = calloc(npd, sizeof *pds);
    MUST(EnumeratePhysicalDevices(inst, &npd, pds));
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props;
    for (uint32_t i = 0; i < npd && !pd; i++) {
        GetPhysicalDeviceProperties(pds[i], &props);
        if (props.vendorID == 0x10de) pd = pds[i]; /* by vendor ID, never by node path */
    }
    if (!pd) { printf("FATAL: no NVIDIA (0x10de) physical device\n"); return 2; }
    printf("  device: %s api %u.%u.%u\n", props.deviceName, VK_API_VERSION_MAJOR(props.apiVersion),
           VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion));
    if (props.apiVersion < VK_API_VERSION_1_3) { printf("FATAL: needs a Vulkan 1.3 device\n"); return 2; }

    uint32_t nq = 0;
    GetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties *qf = calloc(nq, sizeof *qf);
    GetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
    uint32_t qfi = 0;
    while (qfi < nq && !(qf[qfi].queueFlags & VK_QUEUE_GRAPHICS_BIT)) qfi++;
    float qprio = 1.0f;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qfi,
                                   .queueCount = 1, .pQueuePriorities = &qprio};
    /* as dEQP: VK_EXT_private_data on top of what the loader enables for a swapchain device */
    const char *dexts[] = {VK_ANDROID_NATIVE_BUFFER_EXTENSION_NAME, "VK_EXT_private_data"};
    VkPhysicalDevicePrivateDataFeatures pdf = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES,
                                               .privateData = VK_TRUE};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &pdf, .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci, .enabledExtensionCount = 2, .ppEnabledExtensionNames = dexts};
    MUST(CreateDevice(pd, &dci, NULL, &dev));

    PFN_vkSetPrivateData SetPD = dp_hal("vkSetPrivateData");
    PFN_vkGetPrivateData GetPD = dp_hal("vkGetPrivateData");
    PFN_vkSetPrivateData SetPDEXT = dp_hal("vkSetPrivateDataEXT");
    PFN_vkGetPrivateData GetPDEXT = dp_hal("vkGetPrivateDataEXT");
    PFN_vkDestroyPrivateDataSlot DestroySlot = dp_hal("vkDestroyPrivateDataSlot");
    PFN_vkDestroyPrivateDataSlot DestroySlotEXT = dp_hal("vkDestroyPrivateDataSlotEXT");
    PFN_vkCreatePrivateDataSlot CreateSlot = (PFN_vkCreatePrivateDataSlot)gdpa(dev, "vkCreatePrivateDataSlot");
    PFN_vkCreatePrivateDataSlot CreateSlotEXT = (PFN_vkCreatePrivateDataSlot)gdpa(dev, "vkCreatePrivateDataSlotEXT");
    PFN_vkCreateImage CreateImage = (PFN_vkCreateImage)gdpa(dev, "vkCreateImage");
    PFN_vkDestroyImage DestroyImage = (PFN_vkDestroyImage)gdpa(dev, "vkDestroyImage");
    PFN_vkDestroyDevice DestroyDevice = (PFN_vkDestroyDevice)gdpa(dev, "vkDestroyDevice");
    if (!CreateSlot || !CreateSlotEXT || !CreateImage || !DestroyImage || !DestroyDevice) {
        printf("FATAL: missing device entry point\n");
        return 2;
    }

    VkPrivateDataSlotCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_PRIVATE_DATA_SLOT_CREATE_INFO};
    VkPrivateDataSlot s1, s2, s3;
    MUST(CreateSlot(dev, &sci, NULL, &s1));
    MUST(CreateSlot(dev, &sci, NULL, &s2));
    MUST(CreateSlotEXT(dev, &sci, NULL, &s3));

    const uint64_t sc = 0x5C;
    const VkObjectType SC = VK_OBJECT_TYPE_SWAPCHAIN_KHR, SURF = VK_OBJECT_TYPE_SURFACE_KHR;
    uint64_t v1 = 1234, v2 = 1234, v3 = 1234;
    GetPD(dev, SC, sc, s1, &v1);
    GetPDEXT(dev, SC, sc, s2, &v2);
    GetPD(dev, SURF, sc, s3, &v3);
    check(v1 == 0 && v2 == 0 && v3 == 0, "never-set private data of swapchain 0x5C / surface 0x5C reads 0 "
          "(core %llu, EXT %llu, surface %llu)", (unsigned long long)v1, (unsigned long long)v2, (unsigned long long)v3);

    VkResult r1 = SetPD(dev, SC, sc, s1, 111), r2 = SetPDEXT(dev, SC, sc, s2, 222), r3 = SetPD(dev, SC, sc, s3, 333);
    VkResult r4 = SetPDEXT(dev, SURF, sc, s1, 444);
    check(r1 == VK_SUCCESS && r2 == VK_SUCCESS && r3 == VK_SUCCESS && r4 == VK_SUCCESS,
          "vkSetPrivateData(swapchain 0x5C: s1 core, s2 EXT, s3 core; surface 0x5C: s1 EXT) -> %d %d %d %d", r1, r2,
          r3, r4);
    v1 = v2 = v3 = 0;
    uint64_t v4 = 0;
    GetPDEXT(dev, SC, sc, s1, &v1);
    GetPD(dev, SC, sc, s2, &v2);
    GetPDEXT(dev, SC, sc, s3, &v3);
    GetPD(dev, SURF, sc, s1, &v4);
    check(v1 == 111 && v2 == 222 && v3 == 333 && v4 == 444,
          "read back across core/EXT entry points: swapchain s1 %llu s2 %llu s3 %llu, surface s1 %llu",
          (unsigned long long)v1, (unsigned long long)v2, (unsigned long long)v3, (unsigned long long)v4);
    MUST(SetPD(dev, SC, sc, s1, 0xdeadbeefcafef00dull));
    GetPDEXT(dev, SC, sc, s1, &v1);
    GetPD(dev, SC, sc + 1, s1, &v2);
    check(v1 == 0xdeadbeefcafef00dull && v2 == 0, "overwrite s1 -> 0x%llx; another handle (0x5D) on s1 reads %llu",
          (unsigned long long)v1, (unsigned long long)v2);

    /* destroyed slot: its values go; a fresh slot starts at 0 */
    VkPrivateDataSlot old1 = s1, old3 = s3;
    DestroySlot(dev, s1, NULL);
    DestroySlotEXT(dev, s3, NULL);
    VkPrivateDataSlot s4, s5;
    MUST(CreateSlot(dev, &sci, NULL, &s4));
    MUST(CreateSlotEXT(dev, &sci, NULL, &s5));
    v1 = v3 = 1234;
    GetPD(dev, SC, sc, s4, &v1);
    GetPDEXT(dev, SC, sc, s5, &v3);
    GetPD(dev, SC, sc, s2, &v2);
    check(v1 == 0 && v3 == 0 && v2 == 222, "after destroying s1/s3: new slots read %llu / %llu (handles %s), s2 still "
          "%llu", (unsigned long long)v1, (unsigned long long)v3,
          (s4 == old1 || s4 == old3 || s5 == old1 || s5 == old3) ? "recycled" : "fresh", (unsigned long long)v2);

    /* a real image: the ICD's own private data */
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {16, 16, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT};
    VkImage img;
    MUST(CreateImage(dev, &ci, NULL, &img));
    uint64_t ih = (uint64_t)img, iv = 0, iv0 = 1234;
    GetPD(dev, VK_OBJECT_TYPE_IMAGE, ih, s4, &iv0);
    VkResult ri = SetPD(dev, VK_OBJECT_TYPE_IMAGE, ih, s2, 555);
    GetPDEXT(dev, VK_OBJECT_TYPE_IMAGE, ih, s2, &iv);
    check(ri == VK_SUCCESS && iv == 555 && iv0 == 0, "real VkImage %p: never-set %llu, set s2 = 555 -> %d, read back "
          "%llu", (void *)(uintptr_t)img, (unsigned long long)iv0, ri, (unsigned long long)iv);

    /* control (informational): the raw ICD's vkSetPrivateData with the same fake swapchain handle */
    {
        #if defined(__i386__)
        void *icd = dlopen("/usr/lib32/libEGL_nvidia.so.0", RTLD_NOW | RTLD_NOLOAD);
#else
        void *icd = dlopen("/usr/lib/libEGL_nvidia.so.0", RTLD_NOW | RTLD_NOLOAD);
#endif
        PFN_vkGetInstanceProcAddr icd_gipa = icd ? (PFN_vkGetInstanceProcAddr)dlsym(icd, "vk_icdGetInstanceProcAddr") : NULL;
        PFN_vkGetDeviceProcAddr icd_gdpa = icd_gipa ? (PFN_vkGetDeviceProcAddr)icd_gipa(inst, "vkGetDeviceProcAddr") : NULL;
        PFN_vkSetPrivateData raw = icd_gdpa ? (PFN_vkSetPrivateData)icd_gdpa(dev, "vkSetPrivateData") : NULL;
        if (!raw) {
            printf("  control: raw ICD vkSetPrivateData not reachable (not run)\n");
        } else {
            fflush(stdout);
            pid_t pid = fork();
            if (pid == 0) {
                VkResult rr = raw(dev, SC, sc, s2, 1);
                _exit(rr == VK_SUCCESS ? 0 : 1);
            }
            int st = 0;
            waitpid(pid, &st, 0);
            printf("  control: raw ICD vkSetPrivateData(SWAPCHAIN, 0x5C) in a child: %s %d\n",
                   WIFSIGNALED(st) ? "killed by signal" : "exited", WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st));
        }
        if (icd) dlclose(icd);
    }

    DestroyImage(dev, img, NULL);
    DestroySlot(dev, s2, NULL);
    DestroySlot(dev, s4, NULL);
    DestroySlotEXT(dev, s5, NULL);
    DestroyDevice(dev, NULL);
    DestroyInstance(inst, NULL);
    hal->common.close(&hal->common);
    printf("loader-pd: %d failure(s) -> %s\n", failures, failures ? "FAIL" : "OK");
    return failures ? 1 : 0;
}
