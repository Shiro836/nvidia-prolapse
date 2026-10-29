/* SPDX-License-Identifier: MIT */
/* The Mesa interfaces of the nvkmd backend, for mesa/gen-nvkmd-bindings.py. */
#include <fcntl.h>
#include <stddef.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xf86drm.h>

#include "nvk_device.h"
#include "nvkmd/nvkmd.h"
#include "util/cache_ops.h"
#include "util/os_time.h"
#include "util/vma.h"
#include "c11/threads.h"
#include "vk_common_entrypoints.h"
#include "vk_device.h"
#include "vk_log.h"
#include "vk_object.h"
#include "vk_physical_device.h"
#include "vk_queue.h"
#include "vk_sync.h"

/* The members the backend reaches inside structures it otherwise treats as
 * opaque: the nvkmd device of an NVK device (nvk_device.h), the loss state
 * and hooks of a vk_device (vk_device.h), and the device of a log object
 * (vk_object.h). The offline harness that drives the backend through these
 * interfaces also initialises a vk_device's queue list, which Mesa's loss
 * report walks (vk_device.c _vk_device_report_lost). */
enum nvkmd_sys_layout {
   NVKMD_SYS_NVK_DEVICE_VK = offsetof(struct nvk_device, vk),
   NVKMD_SYS_NVK_DEVICE_NVKMD = offsetof(struct nvk_device, nvkmd),
   NVKMD_SYS_VK_DEVICE_LOST = offsetof(struct vk_device, _lost.lost),
   NVKMD_SYS_VK_DEVICE_NOTIFY_LOST = offsetof(struct vk_device, notify_lost),
   NVKMD_SYS_VK_DEVICE_NOTIFY_LOST_DATA = offsetof(struct vk_device, notify_lost_data),
   NVKMD_SYS_VK_DEVICE_CHECK_STATUS = offsetof(struct vk_device, check_status),
   NVKMD_SYS_VK_DEVICE_QUEUES = offsetof(struct vk_device, queues),
   NVKMD_SYS_VK_OBJECT_BASE_DEVICE = offsetof(struct vk_object_base, device),
   /* The harness's stand-ins for Mesa's runtime queries over the backend's
    * sync types: a physical device listing them (vk_fence.c, vk_semaphore.c)
    * and a queue of a logical device (vk_queue.c _vk_queue_set_lost). */
   NVKMD_SYS_VK_OBJECT_BASE_TYPE = offsetof(struct vk_object_base, type),
   NVKMD_SYS_VK_PHYSICAL_DEVICE_SUPPORTED_SYNC_TYPES =
      offsetof(struct vk_physical_device, supported_sync_types),
   NVKMD_SYS_VK_QUEUE_LOST = offsetof(struct vk_queue, _lost.lost),
   NVKMD_SYS_VK_QUEUE_SUBMIT_MUTEX = offsetof(struct vk_queue, submit.mutex),
};

/* The types of those members. */
typedef __typeof__(((struct vk_device *)0)->_lost.lost) nvkmd_sys_vk_device_lost;
typedef __typeof__(((struct vk_device *)0)->notify_lost) nvkmd_sys_vk_device_notify_lost;
typedef __typeof__(((struct vk_device *)0)->check_status) nvkmd_sys_vk_device_check_status;
