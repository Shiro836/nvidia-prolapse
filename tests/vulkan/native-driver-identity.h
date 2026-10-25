/* Which Vulkan driver a gate program must find in its own process, from /proc/self/maps (--expect-driver MODE).
 *   nvidia-prolapse  our source-built stack: no NVIDIA userspace library and no glibc-compat shim may be mapped
 *                    (the default when --expect-driver is absent)
 *   glibc-shim       NVIDIA's own driver through the glibc-compat shims: libEGL_nvidia.so.<version> and
 *                    libnvglibc.so must be mapped, and nothing of our source-built stack (libvulkan_nouveau,
 *                    libnv_bionic_native)
 * Each mode fails loudly, naming the mapping, when it finds the other driver or misses its own.
 *
 * The descriptor check after dlclose (every fd the run opened is closed again) holds for nvidia-prolapse, whose driver
 * unloads. In glibc-shim the shims load every library NVIDIA's driver opens RTLD_NODELETE (libEGL_nvidia otherwise
 * unmaps itself under its own exit-time destructor), so the driver and the descriptors it holds outlive dlclose by
 * design: nv_expect_driver_fds_checked() is false there and the programs list what is still open instead. */
#pragma once
#include <stdio.h>
#include <string.h>

enum nv_expect_driver { NV_EXPECT_PROLAPSE, NV_EXPECT_GLIBC_SHIM };

static int nv_expect_driver_parse(const char *s, enum nv_expect_driver *out) {
    if (!strcmp(s, "nvidia-prolapse")) *out = NV_EXPECT_PROLAPSE;
    else if (!strcmp(s, "glibc-shim")) *out = NV_EXPECT_GLIBC_SHIM;
    else return -1;
    return 0;
}

/* 0 when the process maps the expected driver; else writes the reason to why (prefixed "driver identity:"). */
static int nv_expect_driver_check(enum nv_expect_driver want, char *why, size_t n) {
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) { snprintf(why, n, "driver identity: cannot open /proc/self/maps"); return -1; }
    char line[4096];
    int icd = 0, shim = 0, bad = 0;
    while (fgets(line, sizeof(line), f)) {
        int proprietary = strstr(line, "libnvidia-") || strstr(line, "libEGL_nvidia") || strstr(line, "libnvglibc");
        int ours = strstr(line, "libvulkan_nouveau") || strstr(line, "libnv_bionic_native");
        if (want == NV_EXPECT_PROLAPSE ? proprietary : ours) {
            snprintf(why, n, "driver identity: %s expected, but this is mapped: %s",
                     want == NV_EXPECT_PROLAPSE ? "nvidia-prolapse (no NVIDIA userspace/shim)"
                                                : "glibc-shim (NVIDIA's driver, none of ours)", line);
            bad = 1;
            break;
        }
        icd |= strstr(line, "libEGL_nvidia.so.") != NULL;
        shim |= strstr(line, "libnvglibc.so") != NULL;
    }
    fclose(f);
    if (bad) return -1;
    if (want == NV_EXPECT_GLIBC_SHIM && !(icd && shim)) {
        snprintf(why, n, "driver identity: glibc-shim expected, but %s is not mapped",
                 !icd ? "NVIDIA's libEGL_nvidia.so.<version>" : "the glibc-compat shim libnvglibc.so");
        return -1;
    }
    return 0;
}

/* Whether "every descriptor closed again after dlclose" applies to this driver (see the header comment). */
static int nv_expect_driver_fds_checked(enum nv_expect_driver want) { return want == NV_EXPECT_PROLAPSE; }
