/* The AHardwareBuffer entry points the HAL uses, as a table so that the host build (which has no
 * libnativewindow) can run against an injected fake.
 *
 *   Android: filled by dlopen("libnativewindow.so") + dlsym in the caller's (sphal) linker
 *            namespace. libnativewindow is LLNDK; AHardwareBuffer_getNativeHandle is its LLNDK-only
 *            symbol (declared in frameworks/native/libs/nativewindow/include/vndk/hardware_buffer.h,
 *            copy in build/android13/nativewindow-include/vndk/hardware_buffer.h:33), absent from the
 *            NDK stub, hence dlsym rather than -lnativewindow.
 *   host:    the test executable exports a `struct nvhal_ahb_fns` named NVHAL_TEST_AHB_FNS_SYMBOL
 *            (linked -rdynamic); the HAL finds it with dlsym(RTLD_DEFAULT). Without it the host
 *            HAL does not offer VK_ANDROID_external_memory_android_hardware_buffer. */
#ifndef NVHAL_AHB_H
#define NVHAL_AHB_H

#include <android/hardware_buffer.h>
#include <cutils/native_handle.h>

struct nvhal_ahb_fns {
    void (*describe)(const AHardwareBuffer *buffer, AHardwareBuffer_Desc *desc);
    const native_handle_t *(*get_native_handle)(const AHardwareBuffer *buffer);
    void (*acquire)(AHardwareBuffer *buffer);
    void (*release)(AHardwareBuffer *buffer);
    int (*allocate)(const AHardwareBuffer_Desc *desc, AHardwareBuffer **out);
};

#define NVHAL_TEST_AHB_FNS_SYMBOL "nvhal_test_ahb_fns"

/* HAL pixel formats that AHardwareBuffer_Desc.format can carry beyond the NDK enum
 * (system/core/libsystem/include/system/graphics-base-v1.0.h) */
#define NVHAL_AHB_FORMAT_B8G8R8A8_UNORM 5u
#define NVHAL_AHB_FORMAT_IMPLEMENTATION_DEFINED 0x22u

#endif
