/* Independent wire proof compiled against public matching OGKM headers. */
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <linux/sync_file.h>
#include "class/cl00da.h"
#include "ctrl/ctrl00da.h"
#include "ctrl/ctrl2080/ctrl2080fb.h"
#include "nv_drm_common_ioctl.h"
#include "nvkms-kapi-private.h"
#define SIZE(t,n) _Static_assert(sizeof(t)==(n), #t " size")
SIZE(NV_SEMAPHORE_SURFACE_ALLOC_PARAMETERS,16);
SIZE(NV_SEMAPHORE_SURFACE_CTRL_BIND_CHANNEL_PARAMS,40);
SIZE(NV2080_CTRL_FB_GET_SEMAPHORE_SURFACE_LAYOUT_PARAMS,32);
SIZE(struct NvKmsKapiPrivImportSemaphoreSurfaceParams,16);
SIZE(struct drm_nvidia_semsurf_fence_ctx_create_params,32);
SIZE(struct drm_nvidia_semsurf_fence_create_params,24);
SIZE(struct drm_nvidia_semsurf_fence_wait_params,24);
SIZE(struct sync_file_info,56);
_Static_assert(DRM_IOCTL_NVIDIA_SEMSURF_FENCE_CTX_CREATE==0xc0206454UL,"context ioctl");
_Static_assert(DRM_IOCTL_NVIDIA_SEMSURF_FENCE_CREATE==0xc0186455UL,"fence ioctl");
_Static_assert(DRM_IOCTL_NVIDIA_SEMSURF_FENCE_WAIT==0x40186456UL,"wait ioctl");
_Static_assert(SYNC_IOC_FILE_INFO==0xc0383e04UL,"sync_file ioctl");
_Static_assert(offsetof(struct drm_nvidia_semsurf_fence_create_params,wait_value)==8,"wait offset");
_Static_assert(offsetof(struct drm_nvidia_semsurf_fence_create_params,fd)==16,"fd offset");
_Static_assert(offsetof(struct NvKmsKapiPrivImportSemaphoreSurfaceParams,semaphoreSurfaceSize)==8,"import size offset");
int main(void) { puts("PASS: 8 public C wire sizes, 4 ioctl numbers, 3 field offsets match fences.rs"); return 0; }
