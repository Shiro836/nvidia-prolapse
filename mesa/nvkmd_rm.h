/* SPDX-License-Identifier: MIT */
#ifndef NVKMD_RM_H
#define NVKMD_RM_H
#include "../nvkmd.h"
VkResult
nvkmd_rm_try_create_pdev(struct _drmDevice* drm,
                         struct vk_object_base* log_obj,
                         enum nvk_debug debug,
                         struct nvkmd_pdev** out);
/* Create a context of `dev` whose channel shares the TSG of `share` (a
 * context of the same device) on its own subcontext, so the two contexts'
 * work runs concurrently on the GR engine. Compute engine only. */
VkResult
nvkmd_rm_create_shared_ctx(struct nvkmd_dev* dev,
                           struct vk_object_base* log_obj,
                           enum nvkmd_engines engines,
                           struct nvkmd_ctx* share,
                           struct nvkmd_ctx** out);
#endif
