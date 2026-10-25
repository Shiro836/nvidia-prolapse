/* Compute-state shadow for the ASTC decode.
 *
 * astc.c records its own compute dispatch into the app's command buffer (bind pipeline, bind descriptor set 0,
 * push 20 bytes of compute push constants, dispatch). By the spec that disturbs the app's state:
 *  - the compute pipeline binding (or the compute shader object, VK_EXT_shader_object) is replaced;
 *  - descriptor sets at the COMPUTE bind point: binding set 0 with a layout not compatible with the app's
 *    disturbs set 0 and, the previous set 0 having been bound with an incompatible layout, every higher set
 *    ("Pipeline Layout Compatibility", descriptorsets.adoc);
 *  - push constants: compute-stage bytes [0, 20) now hold the HAL's values, set with the HAL's layout, and "the
 *    bound pipeline's layout must be compatible with the layouts used to set the values of all push constants
 *    in the pipeline layout's push constant ranges" (descriptorsets.adoc, vkCmdPushConstants);
 *  - with VK_EXT_descriptor_heap, recording vkCmdBindDescriptorSets/vkCmdPushConstants invalidates heap state.
 * Mesa's drivers save and restore this around their meta dispatches (radv_meta_save, the baseline's vn shadow);
 * a pass-through HAL has to shadow what the app records. This file keeps, per command buffer, the last compute
 * pipeline/shader binding, the still-effective descriptor-set binding calls of the COMPUTE bind point (deep
 * copies, in recording order), and the push-constant updates since the first one that reached the compute
 * stage, and replays them after the decode (nvhal_cs_restore). Objects passed to recorded commands stay valid
 * while the command buffer records (fundamentals.adoc, Object Lifetime), so the replay is legal.
 * What cannot be replayed (descriptor update templates, heaps, dynamic pipeline layouts, an unknown pNext) is
 * remembered and named in an error should a decode follow it in the same command buffer. */
#include "nvhal.h"

#include <stdlib.h>
#include <string.h>

#define CS_MAX_SETS 32
#define CS_MAX_CALLS CS_MAX_SETS /* every live call owns at least one set index */
#define CS_MAX_PUSH 32

enum cs_kind { CS_BIND, CS_PUSH_DESC, CS_DB_OFFSETS, CS_DB_SAMPLERS };
struct cs_call {
    enum cs_kind kind;
    VkPipelineLayout layout;
    uint32_t first, count; /* set indices [first, first + count) */
    uint32_t refs;         /* how many of them it still owns */
    uint32_t n;            /* CS_BIND: dynamic offsets; CS_PUSH_DESC: writes */
    VkDescriptorSet *sets;
    uint32_t *dyn;
    VkWriteDescriptorSet *writes;
    PFN_vkCmdPushDescriptorSet push_fn;
    uint32_t *indices;
    VkDeviceSize *offsets;
    /* payload follows */
};
struct cs_push {
    VkPipelineLayout layout;
    VkShaderStageFlags stages;
    uint32_t offset, size;
    uint8_t data[];
};
enum { CS_NONE, CS_PIPELINE, CS_SHADER };
struct nvhal_cmd_state {
    int bound;
    VkPipeline pipeline;
    VkShaderEXT shader;
    struct cs_call *owner[CS_MAX_SETS];
    struct cs_call *calls[CS_MAX_CALLS];
    uint32_t ncalls;
    struct cs_push *push[CS_MAX_PUSH];
    uint32_t npush;
    bool push_compute; /* an update reached the compute stage: from then on every update is kept */
    bool conditional;  /* inside vkCmdBeginConditionalRenderingEXT .. End */
    const char *lost;  /* the first thing that cannot be replayed, NULL while everything can */
};

static void cs_reset(struct nvhal_cmd_state *s) {
    for (uint32_t i = 0; i < s->ncalls; i++) free(s->calls[i]);
    for (uint32_t i = 0; i < s->npush; i++) free(s->push[i]);
    memset(s, 0, sizeof *s);
}

void nvhal_cs_free(struct nvhal_cmd_state *s) {
    if (!s) return;
    cs_reset(s);
    free(s);
}

static struct nvhal_cmd_state *cs_of(struct nvhal_cmd *e) {
    if (!e->cs && !(e->cs = calloc(1, sizeof *e->cs)))
        NVHAL_ERR("command buffer %p: out of host memory for the compute-state shadow; an ASTC decode in it cannot "
                  "restore the app's compute state", (void *)e);
    return e->cs;
}

static void lose(struct nvhal_cmd_state *s, const char *why) {
    if (!s->lost) s->lost = why;
}

static void remove_call(struct nvhal_cmd_state *s, struct cs_call *c) {
    for (uint32_t i = 0; i < s->ncalls; i++)
        if (s->calls[i] == c) {
            memmove(&s->calls[i], &s->calls[i + 1], (s->ncalls - i - 1) * sizeof *s->calls);
            s->ncalls--;
            break;
        }
    free(c);
}

/* c binds sets [first, first + count): it now owns them; calls that own nothing any more are dropped */
static void add_call(struct nvhal_cmd_state *s, struct cs_call *c) {
    c->refs = c->count;
    for (uint32_t k = c->first; k < c->first + c->count; k++) {
        struct cs_call *o = s->owner[k];
        s->owner[k] = c;
        if (o && --o->refs == 0) remove_call(s, o);
    }
    s->calls[s->ncalls++] = c; /* <= CS_MAX_CALLS: each live call owns a distinct set index */
}

static struct cs_call *new_call(struct nvhal_cmd_state *s, enum cs_kind kind, VkPipelineLayout layout, uint32_t first,
                                uint32_t count, size_t payload) {
    if (count == 0 || first >= CS_MAX_SETS || count > CS_MAX_SETS - first) {
        lose(s, "a descriptor set index beyond 32");
        return NULL;
    }
    struct cs_call *c = calloc(1, sizeof *c + payload);
    if (!c) {
        lose(s, "out of host memory");
        return NULL;
    }
    c->kind = kind;
    c->layout = layout;
    c->first = first;
    c->count = count;
    return c;
}

static void record_bind(struct nvhal_cmd *e, VkPipelineLayout layout, uint32_t first, uint32_t count,
                        const VkDescriptorSet *sets, uint32_t ndyn, const uint32_t *dyn) {
    struct nvhal_cmd_state *s = cs_of(e);
    if (!s) return;
    if (layout == VK_NULL_HANDLE) {
        lose(s, "a descriptor-set binding with a dynamic pipeline layout (VkPipelineLayoutCreateInfo in pNext)");
        return;
    }
    struct cs_call *c = new_call(s, CS_BIND, layout, first, count, count * sizeof *sets + ndyn * sizeof *dyn);
    if (!c) return;
    c->sets = (VkDescriptorSet *)(c + 1);
    memcpy(c->sets, sets, count * sizeof *sets);
    c->dyn = (uint32_t *)(c->sets + count);
    c->n = ndyn;
    if (ndyn) memcpy(c->dyn, dyn, ndyn * sizeof *dyn);
    add_call(s, c);
}

/* ---- push descriptors: deep copy of the writes ---- */

static int write_kind(VkDescriptorType t) {
    switch (t) {
    case VK_DESCRIPTOR_TYPE_SAMPLER:
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
    case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        return 1;
    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        return 2;
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
        return 3;
    case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
        return 4;
    default:
        return 0;
    }
}

#define ALIGN8(x) (((x) + 7) & ~(size_t)7)
/* bytes of one write's arrays; 0 with *ok = false when it cannot be copied */
static size_t write_payload(const VkWriteDescriptorSet *w, bool *ok) {
    int k = write_kind(w->descriptorType);
    const VkWriteDescriptorSetInlineUniformBlock *ib = w->pNext;
    if (k == 4) {
        if (!ib || ib->sType != VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK || ib->pNext) *ok = false;
        return *ok ? ALIGN8(sizeof *ib) + ALIGN8(ib->dataSize) : 0;
    }
    if (!k || w->pNext) {
        *ok = false;
        return 0;
    }
    size_t e = k == 1 ? sizeof(VkDescriptorImageInfo) : k == 2 ? sizeof(VkBufferView) : sizeof(VkDescriptorBufferInfo);
    return ALIGN8(e * w->descriptorCount);
}

static void record_push_desc(struct nvhal_cmd *e, PFN_vkCmdPushDescriptorSet fn, VkPipelineLayout layout, uint32_t set,
                             uint32_t n, const VkWriteDescriptorSet *w) {
    struct nvhal_cmd_state *s = cs_of(e);
    if (!s) return;
    if (!fn || layout == VK_NULL_HANDLE) {
        lose(s, !fn ? "push descriptors without a vkCmdPushDescriptorSet entry point to replay them"
                    : "push descriptors with a dynamic pipeline layout");
        return;
    }
    bool ok = true;
    size_t bytes = ALIGN8(n * sizeof *w);
    for (uint32_t i = 0; i < n && ok; i++) bytes += write_payload(&w[i], &ok);
    if (!ok) {
        lose(s, "push descriptors of a type or with a pNext structure the shadow does not copy");
        return;
    }
    struct cs_call *c = new_call(s, CS_PUSH_DESC, layout, set, 1, bytes);
    if (!c) return;
    unsigned char *p = (unsigned char *)(c + 1);
    c->writes = (VkWriteDescriptorSet *)p;
    c->n = n;
    c->push_fn = fn;
    memcpy(c->writes, w, n * sizeof *w);
    p += ALIGN8(n * sizeof *w);
    for (uint32_t i = 0; i < n; i++) {
        VkWriteDescriptorSet *cw = &c->writes[i];
        int k = write_kind(w[i].descriptorType);
        if (k == 4) {
            const VkWriteDescriptorSetInlineUniformBlock *ib = w[i].pNext;
            VkWriteDescriptorSetInlineUniformBlock *cb = (void *)p;
            *cb = *ib;
            p += ALIGN8(sizeof *ib);
            memcpy(p, ib->pData, ib->dataSize);
            cb->pData = p;
            p += ALIGN8(ib->dataSize);
            cw->pNext = cb;
            continue;
        }
        size_t sz = k == 1 ? sizeof(VkDescriptorImageInfo) : k == 2 ? sizeof(VkBufferView) : sizeof(VkDescriptorBufferInfo);
        const void *src = k == 1 ? (const void *)w[i].pImageInfo : k == 2 ? (const void *)w[i].pTexelBufferView
                                                                         : (const void *)w[i].pBufferInfo;
        memcpy(p, src, sz * w[i].descriptorCount);
        cw->pImageInfo = k == 1 ? (const void *)p : NULL;
        cw->pTexelBufferView = k == 2 ? (const void *)p : NULL;
        cw->pBufferInfo = k == 3 ? (const void *)p : NULL;
        p += ALIGN8(sz * w[i].descriptorCount);
    }
    add_call(s, c);
}

static void record_db_offsets(struct nvhal_cmd *e, VkPipelineLayout layout, uint32_t first, uint32_t count,
                              const uint32_t *idx, const VkDeviceSize *off) {
    struct nvhal_cmd_state *s = cs_of(e);
    if (!s) return;
    if (layout == VK_NULL_HANDLE) {
        lose(s, "descriptor buffer offsets with a dynamic pipeline layout");
        return;
    }
    struct cs_call *c = new_call(s, CS_DB_OFFSETS, layout, first, count, count * (sizeof *idx + sizeof *off));
    if (!c) return;
    c->offsets = (VkDeviceSize *)(c + 1);
    c->indices = (uint32_t *)(c->offsets + count);
    memcpy(c->offsets, off, count * sizeof *off);
    memcpy(c->indices, idx, count * sizeof *idx);
    add_call(s, c);
}

static void record_db_samplers(struct nvhal_cmd *e, VkPipelineLayout layout, uint32_t set) {
    struct nvhal_cmd_state *s = cs_of(e);
    if (!s) return;
    if (layout == VK_NULL_HANDLE) {
        lose(s, "embedded descriptor buffer samplers with a dynamic pipeline layout");
        return;
    }
    struct cs_call *c = new_call(s, CS_DB_SAMPLERS, layout, set, 1, 0);
    if (c) add_call(s, c);
}

/* ---- push constants ---- */

static void record_push(struct nvhal_cmd *e, VkPipelineLayout layout, VkShaderStageFlags stages, uint32_t offset,
                        uint32_t size, const void *data) {
    const bool compute = (stages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
    if (!compute && (!e->cs || !e->cs->push_compute)) return; /* nothing of the compute stage to protect yet */
    struct nvhal_cmd_state *s = cs_of(e);
    if (!s) return;
    if (layout == VK_NULL_HANDLE) {
        lose(s, "push constants with a dynamic pipeline layout");
        return;
    }
    s->push_compute |= compute;
    struct cs_push *p = NULL;
    for (uint32_t i = 0; i < s->npush; i++)
        if (s->push[i]->layout == layout && s->push[i]->stages == stages && s->push[i]->offset == offset &&
            s->push[i]->size == size) {
            p = s->push[i];
            memmove(&s->push[i], &s->push[i + 1], (s->npush - i - 1) * sizeof *s->push);
            s->npush--;
            break;
        }
    if (!p) {
        if (s->npush == CS_MAX_PUSH) {
            lose(s, "more than 32 distinct push-constant updates");
            return;
        }
        if (!(p = malloc(sizeof *p + size))) {
            lose(s, "out of host memory");
            return;
        }
        *p = (struct cs_push){layout, stages, offset, size};
    }
    memcpy(p->data, data, size);
    s->push[s->npush++] = p; /* most recent last: the replay order */
}

/* ---- restore ---- */

void nvhal_cs_restore(struct nvhal_device *d, VkCommandBuffer cb, const char *who) {
    struct nvhal_cmd *e = nvhal_cmd_get(cb);
    struct nvhal_cmd_state *s = e ? e->cs : NULL;
    if (!s) return; /* the app recorded no compute state: nothing of its own was disturbed */
    const VkPipelineBindPoint bp = VK_PIPELINE_BIND_POINT_COMPUTE;
    if (s->bound == CS_PIPELINE) {
        d->cs.CmdBindPipeline(cb, bp, s->pipeline);
    } else if (s->bound == CS_SHADER) {
        const VkShaderStageFlagBits st = VK_SHADER_STAGE_COMPUTE_BIT;
        d->cs.CmdBindShadersEXT(cb, 1, &st, &s->shader);
    }
    for (uint32_t i = 0; i < s->ncalls; i++) {
        const struct cs_call *c = s->calls[i];
        switch (c->kind) {
        case CS_BIND:
            d->cs.CmdBindDescriptorSets(cb, bp, c->layout, c->first, c->count, c->sets, c->n, c->dyn);
            break;
        case CS_PUSH_DESC:
            c->push_fn(cb, bp, c->layout, c->first, c->n, c->writes);
            break;
        case CS_DB_OFFSETS:
            d->cs.CmdSetDescriptorBufferOffsetsEXT(cb, bp, c->layout, c->first, c->count, c->indices, c->offsets);
            break;
        case CS_DB_SAMPLERS:
            d->cs.CmdBindDescriptorBufferEmbeddedSamplersEXT(cb, bp, c->layout, c->first);
            break;
        }
    }
    uint32_t from = s->npush;
    for (uint32_t i = 0; i < s->npush && from == s->npush; i++)
        if (s->push[i]->stages & VK_SHADER_STAGE_COMPUTE_BIT) from = i;
    for (uint32_t i = from; i < s->npush; i++)
        d->cs.CmdPushConstants(cb, s->push[i]->layout, s->push[i]->stages, s->push[i]->offset, s->push[i]->size,
                               s->push[i]->data);
    if (s->lost)
        NVHAL_ERR("%s: command buffer %p: the app's compute state could not be fully restored after the ASTC decode "
                  "(%s); a later dispatch in this command buffer that relies on it without rebinding reads disturbed "
                  "state (ASTC gap, R8)", who, (void *)cb, s->lost);
}

/* ---- intercepts ---- */

#define CSCMD(fn)                                                                                             \
    struct nvhal_cmd *e = nvhal_cmd_get(cb);                                                                  \
    if (!e) {                                                                                                 \
        NVHAL_ERR("%s(%p): command buffer was not allocated through this HAL; dropped", fn, (void *)cb);      \
        return;                                                                                               \
    }                                                                                                         \
    struct nvhal_device *d = e->dev;

static VKAPI_ATTR VkResult VKAPI_CALL cs_BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo *bi) {
    struct nvhal_cmd *e = nvhal_cmd_get(cb);
    if (!e) {
        NVHAL_ERR("vkBeginCommandBuffer(%p): command buffer was not allocated through this HAL (its allocation ran out "
                  "of host memory) -> VK_ERROR_OUT_OF_HOST_MEMORY", (void *)cb);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    if (e->cs) cs_reset(e->cs); /* begin resets the command buffer, and with it every bound state */
    return e->dev->cs.BeginCommandBuffer(cb, bi);
}

static VKAPI_ATTR void VKAPI_CALL cs_CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p) {
    CSCMD("vkCmdBindPipeline")
    d->cs.CmdBindPipeline(cb, bp, p);
    if (bp != VK_PIPELINE_BIND_POINT_COMPUTE) return;
    struct nvhal_cmd_state *s = cs_of(e);
    if (!s) return;
    s->bound = CS_PIPELINE;
    s->pipeline = p;
}

static VKAPI_ATTR void VKAPI_CALL cs_CmdBindShadersEXT(VkCommandBuffer cb, uint32_t n, const VkShaderStageFlagBits *st,
                                                       const VkShaderEXT *sh) {
    CSCMD("vkCmdBindShadersEXT")
    d->cs.CmdBindShadersEXT(cb, n, st, sh);
    for (uint32_t i = 0; i < n; i++)
        if (st[i] == VK_SHADER_STAGE_COMPUTE_BIT) {
            struct nvhal_cmd_state *s = cs_of(e);
            if (!s) return;
            s->bound = CS_SHADER;
            s->shader = sh ? sh[i] : VK_NULL_HANDLE;
        }
}

static VKAPI_ATTR void VKAPI_CALL cs_CmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp,
                                                          VkPipelineLayout layout, uint32_t first, uint32_t count,
                                                          const VkDescriptorSet *sets, uint32_t ndyn,
                                                          const uint32_t *dyn) {
    CSCMD("vkCmdBindDescriptorSets")
    d->cs.CmdBindDescriptorSets(cb, bp, layout, first, count, sets, ndyn, dyn);
    if (bp == VK_PIPELINE_BIND_POINT_COMPUTE) record_bind(e, layout, first, count, sets, ndyn, dyn);
}

static void bind_sets2(struct nvhal_cmd *e, const VkBindDescriptorSetsInfo *i) {
    if (!(i->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)) return;
    if (i->pNext) {
        struct nvhal_cmd_state *s = cs_of(e);
        if (s) lose(s, "vkCmdBindDescriptorSets2 with a pNext structure");
        return;
    }
    record_bind(e, i->layout, i->firstSet, i->descriptorSetCount, i->pDescriptorSets, i->dynamicOffsetCount,
                i->pDynamicOffsets);
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdBindDescriptorSets2(VkCommandBuffer cb, const VkBindDescriptorSetsInfo *i) {
    CSCMD("vkCmdBindDescriptorSets2")
    d->cs.CmdBindDescriptorSets2(cb, i);
    bind_sets2(e, i);
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdBindDescriptorSets2KHR(VkCommandBuffer cb, const VkBindDescriptorSetsInfo *i) {
    CSCMD("vkCmdBindDescriptorSets2KHR")
    d->cs.CmdBindDescriptorSets2KHR(cb, i);
    bind_sets2(e, i);
}

static VKAPI_ATTR void VKAPI_CALL cs_CmdPushConstants(VkCommandBuffer cb, VkPipelineLayout layout, VkShaderStageFlags st,
                                                     uint32_t offset, uint32_t size, const void *data) {
    CSCMD("vkCmdPushConstants")
    d->cs.CmdPushConstants(cb, layout, st, offset, size, data);
    record_push(e, layout, st, offset, size, data);
}
static void push2(struct nvhal_cmd *e, const VkPushConstantsInfo *i) {
    if (i->pNext) {
        if ((i->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) || (e->cs && e->cs->push_compute)) {
            struct nvhal_cmd_state *s = cs_of(e);
            if (s) lose(s, "vkCmdPushConstants2 with a pNext structure");
        }
        return;
    }
    record_push(e, i->layout, i->stageFlags, i->offset, i->size, i->pValues);
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdPushConstants2(VkCommandBuffer cb, const VkPushConstantsInfo *i) {
    CSCMD("vkCmdPushConstants2")
    d->cs.CmdPushConstants2(cb, i);
    push2(e, i);
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdPushConstants2KHR(VkCommandBuffer cb, const VkPushConstantsInfo *i) {
    CSCMD("vkCmdPushConstants2KHR")
    d->cs.CmdPushConstants2KHR(cb, i);
    push2(e, i);
}

static VKAPI_ATTR void VKAPI_CALL cs_CmdPushDescriptorSet(VkCommandBuffer cb, VkPipelineBindPoint bp,
                                                         VkPipelineLayout layout, uint32_t set, uint32_t n,
                                                         const VkWriteDescriptorSet *w) {
    CSCMD("vkCmdPushDescriptorSet")
    d->cs.CmdPushDescriptorSet(cb, bp, layout, set, n, w);
    if (bp == VK_PIPELINE_BIND_POINT_COMPUTE) record_push_desc(e, d->cs.CmdPushDescriptorSet, layout, set, n, w);
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdPushDescriptorSetKHR(VkCommandBuffer cb, VkPipelineBindPoint bp,
                                                            VkPipelineLayout layout, uint32_t set, uint32_t n,
                                                            const VkWriteDescriptorSet *w) {
    CSCMD("vkCmdPushDescriptorSetKHR")
    d->cs.CmdPushDescriptorSetKHR(cb, bp, layout, set, n, w);
    if (bp == VK_PIPELINE_BIND_POINT_COMPUTE) record_push_desc(e, d->cs.CmdPushDescriptorSetKHR, layout, set, n, w);
}
static void push_desc2(struct nvhal_cmd *e, struct nvhal_device *d, const VkPushDescriptorSetInfo *i) {
    if (!(i->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)) return;
    if (i->pNext) {
        struct nvhal_cmd_state *s = cs_of(e);
        if (s) lose(s, "vkCmdPushDescriptorSet2 with a pNext structure");
        return;
    }
    record_push_desc(e, d->cs.CmdPushDescriptorSet ? d->cs.CmdPushDescriptorSet : d->cs.CmdPushDescriptorSetKHR,
                     i->layout, i->set, i->descriptorWriteCount, i->pDescriptorWrites);
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdPushDescriptorSet2(VkCommandBuffer cb, const VkPushDescriptorSetInfo *i) {
    CSCMD("vkCmdPushDescriptorSet2")
    d->cs.CmdPushDescriptorSet2(cb, i);
    push_desc2(e, d, i);
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdPushDescriptorSet2KHR(VkCommandBuffer cb, const VkPushDescriptorSetInfo *i) {
    CSCMD("vkCmdPushDescriptorSet2KHR")
    d->cs.CmdPushDescriptorSet2KHR(cb, i);
    push_desc2(e, d, i);
}

/* templates: the data layout lives in the template object; not replayed */
static void template_lost(struct nvhal_cmd *e) {
    struct nvhal_cmd_state *s = cs_of(e);
    if (s) lose(s, "push descriptors through a descriptor update template");
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdPushDescriptorSetWithTemplate(VkCommandBuffer cb, VkDescriptorUpdateTemplate t,
                                                                     VkPipelineLayout layout, uint32_t set,
                                                                     const void *data) {
    CSCMD("vkCmdPushDescriptorSetWithTemplate")
    d->cs.CmdPushDescriptorSetWithTemplate(cb, t, layout, set, data);
    template_lost(e);
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdPushDescriptorSetWithTemplateKHR(VkCommandBuffer cb,
                                                                        VkDescriptorUpdateTemplate t,
                                                                        VkPipelineLayout layout, uint32_t set,
                                                                        const void *data) {
    CSCMD("vkCmdPushDescriptorSetWithTemplateKHR")
    d->cs.CmdPushDescriptorSetWithTemplateKHR(cb, t, layout, set, data);
    template_lost(e);
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdPushDescriptorSetWithTemplate2(VkCommandBuffer cb,
                                                                      const VkPushDescriptorSetWithTemplateInfo *i) {
    CSCMD("vkCmdPushDescriptorSetWithTemplate2")
    d->cs.CmdPushDescriptorSetWithTemplate2(cb, i);
    template_lost(e);
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdPushDescriptorSetWithTemplate2KHR(VkCommandBuffer cb,
                                                                         const VkPushDescriptorSetWithTemplateInfo *i) {
    CSCMD("vkCmdPushDescriptorSetWithTemplate2KHR")
    d->cs.CmdPushDescriptorSetWithTemplate2KHR(cb, i);
    template_lost(e);
}

static VKAPI_ATTR void VKAPI_CALL cs_CmdSetDescriptorBufferOffsetsEXT(VkCommandBuffer cb, VkPipelineBindPoint bp,
                                                                     VkPipelineLayout layout, uint32_t first,
                                                                     uint32_t count, const uint32_t *idx,
                                                                     const VkDeviceSize *off) {
    CSCMD("vkCmdSetDescriptorBufferOffsetsEXT")
    d->cs.CmdSetDescriptorBufferOffsetsEXT(cb, bp, layout, first, count, idx, off);
    if (bp == VK_PIPELINE_BIND_POINT_COMPUTE) record_db_offsets(e, layout, first, count, idx, off);
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdSetDescriptorBufferOffsets2EXT(VkCommandBuffer cb,
                                                                      const VkSetDescriptorBufferOffsetsInfoEXT *i) {
    CSCMD("vkCmdSetDescriptorBufferOffsets2EXT")
    d->cs.CmdSetDescriptorBufferOffsets2EXT(cb, i);
    if (!(i->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)) return;
    if (i->pNext) {
        struct nvhal_cmd_state *s = cs_of(e);
        if (s) lose(s, "vkCmdSetDescriptorBufferOffsets2EXT with a pNext structure");
        return;
    }
    record_db_offsets(e, i->layout, i->firstSet, i->setCount, i->pBufferIndices, i->pOffsets);
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdBindDescriptorBufferEmbeddedSamplersEXT(VkCommandBuffer cb,
                                                                               VkPipelineBindPoint bp,
                                                                               VkPipelineLayout layout, uint32_t set) {
    CSCMD("vkCmdBindDescriptorBufferEmbeddedSamplersEXT")
    d->cs.CmdBindDescriptorBufferEmbeddedSamplersEXT(cb, bp, layout, set);
    if (bp == VK_PIPELINE_BIND_POINT_COMPUTE) record_db_samplers(e, layout, set);
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdBindDescriptorBufferEmbeddedSamplers2EXT(
    VkCommandBuffer cb, const VkBindDescriptorBufferEmbeddedSamplersInfoEXT *i) {
    CSCMD("vkCmdBindDescriptorBufferEmbeddedSamplers2EXT")
    d->cs.CmdBindDescriptorBufferEmbeddedSamplers2EXT(cb, i);
    if (!(i->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)) return;
    if (i->pNext) {
        struct nvhal_cmd_state *s = cs_of(e);
        if (s) lose(s, "vkCmdBindDescriptorBufferEmbeddedSamplers2EXT with a pNext structure");
        return;
    }
    record_db_samplers(e, i->layout, i->set);
}

/* vkCmdDispatch is subject to conditional rendering, copies are not: a decode recorded inside such a block would
 * be skipped with the app's predicate while the copy is not (astc.c names it) */
bool nvhal_cs_conditional(VkCommandBuffer cb) {
    struct nvhal_cmd *e = nvhal_cmd_get(cb);
    return e && e->cs && e->cs->conditional;
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdBeginConditionalRenderingEXT(VkCommandBuffer cb,
                                                                     const VkConditionalRenderingBeginInfoEXT *i) {
    CSCMD("vkCmdBeginConditionalRenderingEXT")
    d->cs.CmdBeginConditionalRenderingEXT(cb, i);
    struct nvhal_cmd_state *s = cs_of(e);
    if (s) s->conditional = true;
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdEndConditionalRenderingEXT(VkCommandBuffer cb) {
    CSCMD("vkCmdEndConditionalRenderingEXT")
    d->cs.CmdEndConditionalRenderingEXT(cb);
    if (e->cs) e->cs->conditional = false;
}

#ifdef VK_EXT_descriptor_heap
/* recording the HAL's vkCmdBindDescriptorSets/vkCmdPushConstants invalidates heap state (descriptorheaps.adoc
 * "descriptorheaps-invalidate-sets"); heap bindings are not shadowed */
static void heap_lost(struct nvhal_cmd *e) {
    struct nvhal_cmd_state *s = cs_of(e);
    if (s) lose(s, "descriptor heap state (VK_EXT_descriptor_heap), which the decode's descriptor-set binding invalidates");
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdBindSamplerHeapEXT(VkCommandBuffer cb, const VkBindHeapInfoEXT *i) {
    CSCMD("vkCmdBindSamplerHeapEXT")
    d->cs.CmdBindSamplerHeapEXT(cb, i);
    heap_lost(e);
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdBindResourceHeapEXT(VkCommandBuffer cb, const VkBindHeapInfoEXT *i) {
    CSCMD("vkCmdBindResourceHeapEXT")
    d->cs.CmdBindResourceHeapEXT(cb, i);
    heap_lost(e);
}
static VKAPI_ATTR void VKAPI_CALL cs_CmdPushDataEXT(VkCommandBuffer cb, const VkPushDataInfoEXT *i) {
    CSCMD("vkCmdPushDataEXT")
    d->cs.CmdPushDataEXT(cb, i);
    heap_lost(e);
}
#endif

/* ---- command pools -> queue family ---- */

static struct nvhal_hash pools = NVHAL_HASH_INIT; /* VkCommandPool -> queueFamilyIndex + 1 */

uint32_t nvhal_pool_family(VkCommandPool pool) {
    void *v = pool ? nvhal_hash_get(&pools, NVHAL_ND_U64(pool)) : NULL;
    return v ? (uint32_t)((uintptr_t)v - 1) : UINT32_MAX;
}
void nvhal_pool_forget(VkCommandPool pool) {
    if (pool) nvhal_hash_del(&pools, NVHAL_ND_U64(pool));
}

static VKAPI_ATTR VkResult VKAPI_CALL cs_CreateCommandPool(VkDevice device, const VkCommandPoolCreateInfo *ci,
                                                           const VkAllocationCallbacks *alloc, VkCommandPool *pool) {
    struct nvhal_device *d = nvhal_device_get(device);
    if (!d) {
        NVHAL_ERR("vkCreateCommandPool(%p): not a device created through this HAL", (void *)device);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    VkResult r = d->cs.CreateCommandPool(device, ci, alloc, pool);
    if (r == VK_SUCCESS &&
        !nvhal_hash_put(&pools, NVHAL_ND_U64(*pool), (void *)(uintptr_t)((uintptr_t)ci->queueFamilyIndex + 1)))
        NVHAL_ERR("vkCreateCommandPool: out of host memory recording pool %p's queue family; ASTC uploads recorded from "
                  "it cannot be decoded", NVHAL_LOGH(*pool));
    return r;
}

#define XF(x) ((PFN_vkVoidFunction)(x))
/* handed out by vkGetDeviceProcAddr only on devices that emulate ASTC (hal.c); elsewhere the ICD's own */
const struct nvhal_xl_entry nvhal_cs_entries[] = {
    {"vkBeginCommandBuffer", XF(cs_BeginCommandBuffer), NVHAL_IC_ASTC},
    {"vkCreateCommandPool", XF(cs_CreateCommandPool), NVHAL_IC_ASTC},
    {"vkCmdBindPipeline", XF(cs_CmdBindPipeline), NVHAL_IC_ASTC},
    {"vkCmdBindShadersEXT", XF(cs_CmdBindShadersEXT), NVHAL_IC_ASTC},
    {"vkCmdBindDescriptorSets", XF(cs_CmdBindDescriptorSets), NVHAL_IC_ASTC},
    {"vkCmdBindDescriptorSets2", XF(cs_CmdBindDescriptorSets2), NVHAL_IC_ASTC},
    {"vkCmdBindDescriptorSets2KHR", XF(cs_CmdBindDescriptorSets2KHR), NVHAL_IC_ASTC},
    {"vkCmdPushConstants", XF(cs_CmdPushConstants), NVHAL_IC_ASTC},
    {"vkCmdPushConstants2", XF(cs_CmdPushConstants2), NVHAL_IC_ASTC},
    {"vkCmdPushConstants2KHR", XF(cs_CmdPushConstants2KHR), NVHAL_IC_ASTC},
    {"vkCmdPushDescriptorSet", XF(cs_CmdPushDescriptorSet), NVHAL_IC_ASTC},
    {"vkCmdPushDescriptorSetKHR", XF(cs_CmdPushDescriptorSetKHR), NVHAL_IC_ASTC},
    {"vkCmdPushDescriptorSet2", XF(cs_CmdPushDescriptorSet2), NVHAL_IC_ASTC},
    {"vkCmdPushDescriptorSet2KHR", XF(cs_CmdPushDescriptorSet2KHR), NVHAL_IC_ASTC},
    {"vkCmdPushDescriptorSetWithTemplate", XF(cs_CmdPushDescriptorSetWithTemplate), NVHAL_IC_ASTC},
    {"vkCmdPushDescriptorSetWithTemplateKHR", XF(cs_CmdPushDescriptorSetWithTemplateKHR), NVHAL_IC_ASTC},
    {"vkCmdPushDescriptorSetWithTemplate2", XF(cs_CmdPushDescriptorSetWithTemplate2), NVHAL_IC_ASTC},
    {"vkCmdPushDescriptorSetWithTemplate2KHR", XF(cs_CmdPushDescriptorSetWithTemplate2KHR), NVHAL_IC_ASTC},
    {"vkCmdSetDescriptorBufferOffsetsEXT", XF(cs_CmdSetDescriptorBufferOffsetsEXT), NVHAL_IC_ASTC},
    {"vkCmdSetDescriptorBufferOffsets2EXT", XF(cs_CmdSetDescriptorBufferOffsets2EXT), NVHAL_IC_ASTC},
    {"vkCmdBindDescriptorBufferEmbeddedSamplersEXT", XF(cs_CmdBindDescriptorBufferEmbeddedSamplersEXT), NVHAL_IC_ASTC},
    {"vkCmdBindDescriptorBufferEmbeddedSamplers2EXT", XF(cs_CmdBindDescriptorBufferEmbeddedSamplers2EXT), NVHAL_IC_ASTC},
    {"vkCmdBeginConditionalRenderingEXT", XF(cs_CmdBeginConditionalRenderingEXT), NVHAL_IC_ASTC},
    {"vkCmdEndConditionalRenderingEXT", XF(cs_CmdEndConditionalRenderingEXT), NVHAL_IC_ASTC},
#ifdef VK_EXT_descriptor_heap
    {"vkCmdBindSamplerHeapEXT", XF(cs_CmdBindSamplerHeapEXT), NVHAL_IC_ASTC},
    {"vkCmdBindResourceHeapEXT", XF(cs_CmdBindResourceHeapEXT), NVHAL_IC_ASTC},
    {"vkCmdPushDataEXT", XF(cs_CmdPushDataEXT), NVHAL_IC_ASTC},
#endif
    {NULL, NULL, 0},
};
