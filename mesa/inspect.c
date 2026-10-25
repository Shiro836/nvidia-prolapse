/* SPDX-License-Identifier: MIT: inspect a source NVK artifact without GPU use. */
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr, "usage: %s libvulkan_nouveau.so\n", argv[0]); return 2; }
    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "%s\n", dlerror()); return 1; }
    PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vk_icdGetInstanceProcAddr");
    if (!gipa) return 1;
    PFN_vkEnumerateInstanceVersion version = (PFN_vkEnumerateInstanceVersion)gipa(VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
    PFN_vkCreateInstance create = (PFN_vkCreateInstance)gipa(VK_NULL_HANDLE, "vkCreateInstance");
    uint32_t api = 0;
    if (!version || !create || version(&api)) return 1;
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "native NVK artifact inspector", .apiVersion = VK_API_VERSION_1_0 };
    VkInstanceCreateInfo info = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
    VkInstance instance = VK_NULL_HANDLE;
    VkResult result = create(&info, NULL, &instance);
    if (result) { fprintf(stderr, "vkCreateInstance: %d\n", result); return 1; }
    PFN_vkDestroyInstance destroy = (PFN_vkDestroyInstance)gipa(instance, "vkDestroyInstance");
    if (!destroy) return 1;
    destroy(instance, NULL);
    printf("NVK load + instance lifecycle PASS: Vulkan %u.%u.%u, Android HAL HMI=%s\n", VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api), VK_API_VERSION_PATCH(api), dlsym(lib, "HMI") ? "present" : "absent");
    dlclose(lib);
    return 0;
}
