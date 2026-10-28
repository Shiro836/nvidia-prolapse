/* SPDX-License-Identifier: MIT
 *
 * nvrm: a gallium driver over the native RM backend (nv_rm.h, the same flat C
 * API NVK's nvkmd_rm.c uses), NAK and nil.
 *
 * G0a: screen + one channel per screen with NVK's 3D init, linear
 * GART-backed resources, CLEAR_SURFACE colour clears, synchronous flush
 * (all three superseded by the perf work below).
 * G0b: VS/FS through NAK, vertex fetch, viewport/scissor, raster/blend
 * state, constant buffers, array and indexed draws.
 * G1 (GLES2): nil tiled VRAM images + copy-engine transfers, TIC/TSC header
 * and sampler pools, NAK texture handles, u_blitter blits, depth/stencil
 * (Blackwell separate Z/S), DSA, Z/S clears, polygon offset, Z clipping,
 * local memory, shader heap reclaim, dirty-state groups.
 * G1 (GLES 3.0): 8 colour targets with per-target blend/write mask, MSAA
 * 2/4/8 render targets (standard sample positions, layout also on clears)
 * + u_blitter resolves, NIR disk cache (program binaries; NAK binaries too since
 * shader-f, nvrm_shader_cache.c), transform feedback with
 * pause/resume, occlusion / primitives generated / xfb written queries,
 * conditional render, 2_10_10_10 vertex formats, separate-Z/S maps,
 * stencil blits via u_blitter's per-bit fallback (NAK has no stencil
 * export).
 * G2: dma-buf import/export with nil DRM modifiers (nvrm_handle.c),
 * sync_file native fences (fence_get_fd / create_fence_fd /
 * fence_server_sync).
 * 3D classes: MAXWELL_A (0xb097) through BLACKWELL_B. Maxwell/Pascal take
 * NVK's pre-Volta paths (program region + heap-relative shader offsets,
 * clip-Z emulation, per-instance draw loop below PASCAL_B, MAXWELL_B-only
 * methods gated, MAXWELL_A texture-header select, PASCAL_A clear bracket)
 * in the GLES 3.0 paths too (MRT clears, MSAA sample locations, xfb,
 * queries).
 * GLES 3.1 part B: shader images (bindless storage headers in the driver
 * header pool, handles after the texture handles in the driver cbuf, 32
 * units per stage, nvrm_image.c), memory/texture barriers, textureGather
 * (4 components, offsets -32..31, per-pixel offsets), cube-map arrays,
 * framebuffer_no_attachment, stencil texturing, sample positions.
 * Perf (gl-perf): NVK memory types (VRAM buffers / shader arena / header
 * pools, BAR1-mapped on a full BAR, copy-engine staging on a small one,
 * nvc0's buffer domain policy), asynchronous flushes with per-submission
 * tokens, a batch-slot ring for push buffers and texture pools,
 * per-resource busy tracking and in-batch staged uploads.
 * Perf P-B (gl-pb): dirty-slot constant buffer binds, default uniforms and
 * texture handles loaded in-stream into a driver cbuf (LOAD_CONSTANT_BUFFER),
 * cache invalidates only for the event that made a cache stale
 * (NVRM_INVAL_*), TIC/TSC entries written once per batch, buffer memory
 * recycled instead of RM alloc/map/free per frame.
 * Still refused loudly: compute, SSBOs, buffer/MS images, timer queries,
 * logic op, alpha test, depth bounds, buffer textures, sample shading,
 * multisampled linear/mipmapped resources, Kepler and older 3D classes
 * (named at screen creation).
 */
#ifndef NVRM_PRIV_H
#define NVRM_PRIV_H

#include <stdbool.h>
#include <stdint.h>

#include "nv_rm.h"
#include "nv_device_info.h"
#include "nil.h"
#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/simple_mtx.h"
#include "util/u_inlines.h"
#include "util/vma.h"
#include "util/log.h"
#include "util/u_dynarray.h"
#include "util/set.h"
#include "compiler/nir/nir.h"
#include "nak.h"
#include "util/disk_cache.h"
#include "util/u_queue.h"

struct nak_compiler;
struct nak_shader_bin;
struct nvrm_query;
struct nv_push;
struct nvrm_context;

#define NVRM_MAX_CBUFS 16
#define NVRM_MAX_RTS 8 /* colour targets, NVK_MAX_RTS */
/* G1 textures: GL units per stage, the driver constant buffer slot holding
 * their handles (GL UBOs use 0..NVRM_TEX_CBUF_SLOT-1), header/sampler pool
 * size in entries. */
#define NVRM_MAX_TEX_UNITS 16
#define NVRM_TEX_CBUF_SLOT 15
#define NVRM_TEX_POOL_SLOTS 4096
/* GLES 3.1 part A (compute + storage buffers). Limits are Mesa's
 * ceilings, not hardware slot counts (stock's low GL limits are not
 * copied): per stage NVRM_MAX_SSBOS = 32 shader buffers, which
 * st/mesa splits into 16 SSBOs (MAX_SHADER_STORAGE_BUFFERS) + 16 atomic
 * counter buffers (max_hw_atomic_counters = 0), and 16 constant buffers
 * (default block + MAX_UNIFORM_BUFFERS 15). Buffer addresses/sizes live in
 * a driver constant buffer ("descriptor table") read by bounded global
 * loads/stores, as NVK's descriptor lowering keeps them in its root
 * descriptor cbuf (nvk_nir_lower_descriptors.c): graphics bind it at
 * NVRM_DESC_CBUF_SLOT, compute as QMD constant buffer NVRM_CS_DESC_CBUF.
 * UBO blocks that have no hardware slot (graphics: index >=
 * NVRM_DESC_CBUF_SLOT or dynamic; compute: every block but 0, the QMD has
 * 8 cbufs) are read through the table too. */
#define NVRM_MAX_SSBOS 32
#define NVRM_MAX_UBOS 16
#define NVRM_DESC_CBUF_SLOT 14
#define NVRM_CS_DESC_CBUF 1
#define NVRM_DESC_SSBO_OFFSET 0     /* 32 x {addr lo, addr hi, size, 0} */
#define NVRM_DESC_UBO_OFFSET 512    /* 16 x {addr lo, addr hi, size, 0} */
#define NVRM_DESC_NUM_WG_OFFSET 768 /* compute: gl_NumWorkGroups xyz */
/* Compute: the texture/image handle table (nvrm_upload_tex_handles) is QMD
 * constant buffer 2; the QMD has 8 cbuf slots (nak_qmd_info.cbufs), so the
 * graphics slot NVRM_TEX_CBUF_SLOT (15) is not addressable there. */
#define NVRM_CS_TEX_CBUF 2
/* Constant buffer holding texture/image handles for a stage. */
#define NVRM_TEX_CBUF_FOR(stage) \
   ((stage) == MESA_SHADER_COMPUTE ? NVRM_CS_TEX_CBUF : NVRM_TEX_CBUF_SLOT)
#define NVRM_DESC_SIZE 1024         /* multiple of the 256 B cbuf granule */
#define NVRM_MAX_SHARED_SIZE (48 * 1024) /* NVK_MAX_SHARED_SIZE, nvk_private.h:35 */
/* GLES 3.1 image units per stage; their handles follow the texture handles
 * in the same driver constant buffer (nvrm_image.c). Mesa's MAX_IMAGE_UNITS. */
#define NVRM_MAX_IMAGES 32

/* Dirty groups (G1): re-emitted only when changed; every flush sets all,
 * since the screen channel is shared by every context. */
#define NVRM_DIRTY_FB        (1u << 0)
#define NVRM_DIRTY_VIEWPORT  (1u << 1)
#define NVRM_DIRTY_RAST      (1u << 2)
#define NVRM_DIRTY_DSA_BLEND (1u << 3)
#define NVRM_DIRTY_SHADERS   (1u << 4)
#define NVRM_DIRTY_CBUFS     (1u << 5)
#define NVRM_DIRTY_VERTEX    (1u << 6)
#define NVRM_DIRTY_TEXTURES  (1u << 7)
#define NVRM_DIRTY_POOLS     (1u << 8)
#define NVRM_DIRTY_SO        (1u << 9)
#define NVRM_DIRTY_COND      (1u << 10)
#define NVRM_DIRTY_SSBO      (1u << 11)
/* Set only through NVRM_DIRTY_ALL (context creation, every flush): the
 * channel's cbuf bindings are unknown and the CPU may have written memory
 * the caches hold (NVRM_INVAL_BATCH). */
#define NVRM_DIRTY_BATCH     (1u << 12)
#define NVRM_DIRTY_ALL       0x1fffu

/* Pending 3D cache invalidates (ctx->inval), each set by the event that
 * makes a cache stale and emitted once before the next draw
 * (nvrm_emit_invalidates), instead of on every dirty draw:
 *  BATCH     a new submission batch: the CPU wrote GPU-visible memory
 *            (uploads, BAR1 maps, recycled buffers) since the previous one;
 *            NVK's command-buffer begin (nvk_cmd_draw.c:700-712: sampler +
 *            header caches, constant cache);
 *  CONSTANT  memory read as constants changed behind the constant cache
 *            within the batch (copy-engine writes into a buffer):
 *            NVK TRANSFER_WRITE -> UNIFORM_READ (nvk_cmd_buffer.c:494-497);
 *  GLOBAL    same for global (SSBO) reads (nvk_cmd_buffer.c:503-504);
 *  INSTR     shader code was uploaded since this context last invalidated
 *            (nvc0_program.c:944-945 invalidates after each upload; stock
 *            uploads before first use);
 *  HEADER    a texture header / sampler pool entry was written (stock: once
 *            after TIC/TSC writes);
 *  TEX_DATA  texel memory changed (copy-engine upload, rendering into a
 *            texture that later draws sample; stock FBO bind = texture data
 *            invalidates, gl-perf #4). */
#define NVRM_INVAL_BATCH     (1u << 0)
#define NVRM_INVAL_CONSTANT  (1u << 1)
#define NVRM_INVAL_GLOBAL    (1u << 2)
#define NVRM_INVAL_INSTR     (1u << 3)
#define NVRM_INVAL_HEADER    (1u << 4)
#define NVRM_INVAL_TEX_DATA  (1u << 5)
#define NVRM_INVAL_ALL       0x3fu

/* Driver constant buffer (per context, VRAM, written only by
 * LOAD_CONSTANT_BUFFER, which the 3D pipe orders with the draws): per
 * graphics stage the default uniform block (bound at slot 0 when st/mesa
 * hands user constants) and the texture/image handle table (slot
 * NVRM_TEX_CBUF_SLOT). nvc0's per-stage uniform_bo areas filled by
 * nvc0_cb_bo_push (nvc0_state_validate.c:532-549, nvc0_transfer.c:576-605);
 * NVK's root descriptor table loads (nvk_cmd_draw.c:681-694); stock's driver
 * CB window + MME 45/60/59 loads. */
#define NVRM_GFX_STAGES 3 /* VS, GS, FS */
#define NVRM_DRV_CB_UNIFORM_B 65536u /* caps.max_const_buffer0_size */
#define NVRM_DRV_CB_HANDLES_B 256u   /* (NVRM_MAX_TEX_UNITS + NVRM_MAX_IMAGES) * 4, cbuf granule */
#define NVRM_DRV_CB_STAGE_B (NVRM_DRV_CB_UNIFORM_B + NVRM_DRV_CB_HANDLES_B)
#define NVRM_MAX_SO 4 /* stream output buffers */
/* Shader heap = NVK's contiguous nvk_mem_arena (nvk_mem_arena.h/.c): one VA
 * reservation of 4 GiB (pre-Volta program addresses are 32-bit offsets from
 * SET_PROGRAM_REGION), backed on demand by memories of 64 KiB, 64 KiB,
 * 128 KiB, ... (size of memory i = i == 0 ? MIN : (MIN >> 1) << i), each
 * bound at the reservation base + the sum of the previous sizes. */
#define NVRM_SHADER_ARENA_MIN_SIZE_LOG2 16
#define NVRM_SHADER_ARENA_MAX_SIZE_LOG2 32
#define NVRM_SHADER_ARENA_MAX_MEMS (NVRM_SHADER_ARENA_MAX_SIZE_LOG2 - \
                                    NVRM_SHADER_ARENA_MIN_SIZE_LOG2 + 1)
/* Unallocated tail of the shader heap for I-cache prefetch (NVK: 2 KiB). */
#define NVRM_SHADER_PREFETCH_PAD_B 2048u

/* Push memory = chunks of NVRM_PUSH_SIZE_B GART, chained into the open batch
 * as further GPFIFO entries when one fills (NVK: nvk_cmd_buffer_new_push,
 * nvk_cmd_buffer.c:169-186, closes the chunk with nvk_cmd_buffer_flush_push
 * :150-166 and all of a command buffer's chunks go out in one submission),
 * so a full chunk is never a kick. A context owns at most
 * NVRM_PUSH_MEM_MAX_B of chunks (recorded + in flight): past that, the
 * oldest chunk's submission is waited for (and, when every chunk belongs to
 * the open batch, that batch is submitted first). */
#define NVRM_PUSH_SIZE_B (256u * 1024u)
#define NVRM_PUSH_MEM_MAX_B (16u * 1024u * 1024u)
/* GPFIFO entries of one flush: the backend refuses batches of more than
 * 1000 pushes (rm/src/gpu.rs submit); nvrm_push_begin flushes before a
 * command once fewer than NVRM_EXEC_HEADROOM entries remain, which covers
 * one command's push-indirect splits (2 entries each). */
#define NVRM_MAX_EXECS 1000u
#define NVRM_EXEC_HEADROOM 64u

/* Memory domains = NVK's memory types (nvk_physical_device.c:1445-1497):
 *  GART        HOST_VISIBLE system memory (push buffers, CPU readback
 *              staging, queries; nvc0's NOUVEAU_BO_GART buffers);
 *  VRAM        DEVICE_LOCAL only, never CPU-mapped (images, SLM, and on a
 *              small BAR the buffers nvc0 keeps in VRAM: written through
 *              GART staging + copy engine, nouveau_buffer.c:424-452);
 *  VRAM_MAPPED DEVICE_LOCAL | HOST_VISIBLE through a BAR1 CPU mapping
 *              (nv_rm_map of VRAM), NVK's second memory type. */
enum nvrm_domain {
   NVRM_DOMAIN_GART,
   NVRM_DOMAIN_VRAM,
   NVRM_DOMAIN_VRAM_MAPPED,
};

struct nvrm_bo {
   uint64_t mem;     /* nv_rm allocation id */
   uint64_t mapping; /* nv_rm_bind mapping id */
   uint64_t va;
   uint64_t size;
   void *map;        /* persistent CPU map (GART, or BAR1 for VRAM_MAPPED) */
   bool vram;        /* VRAM placement (mapped or not) */
};

/* Asynchronous submission: a batch's CPU-written memory (push buffer, texture
 * header/sampler pool) is one of NVRM_BATCH_RING slots, reused only after
 * the GPU completed the batch that last used it (the flush that rotates onto
 * a slot waits for its token, as NVK recycles push memory per completed
 * submission). Tokens of one screen channel complete in submission order. */
#define NVRM_BATCH_RING 4
/* flush_seq -> submission token of the most recent batches (queries and
 * stream-output counters name the batch that wrote them). */
#define NVRM_SEQ_RING 64

struct nvrm_push_chunk {
   struct nvrm_bo bo;
   uint64_t token; /* last submission with pushes in this chunk, 0 = none */
};

struct nvrm_batch_slot {
   struct nvrm_bo tex_pool; /* TIC pool, then TSC pool */
   uint64_t token;          /* last submission using this slot, 0 = none */
};

struct nvrm_screen {
   struct pipe_screen base;
   int fd;
   struct NvRmDevice *rm;
   struct NvRmInfo rm_info;
   struct nv_device_info info;
   struct nak_compiler *nak;
   /* st/mesa's NIR shader cache; its presence is what enables
    * GL_OES_get_program_binary (st_extensions.c NumProgramBinaryFormats). */
   struct disk_cache *disk_cache;

   simple_mtx_t va_mutex;
   struct util_vma_heap heap;

   /* The whole of VRAM is CPU-visible through BAR1 (resizable BAR): NVK's
    * test for CPU shader uploads (nvk_device.c:256-258, bar_size_B >=
    * vram_size_B). Without it, BAR1 is a small window: buffers nvc0 keeps
    * in VRAM and the shader arena are written through the copy engine, so
    * BAR1 holds only the small driver pools (NVK maps its descriptor/event
    * heaps the same way, nvk_device.c:264-275). */
   bool bar_full;

   /* NVRM_DEBUG_STATS=1: channel submissions and CPU waits for GPU
    * completion, printed at context destruction (perf reproducers). */
   bool debug_stats;
   uint64_t stat_submits, stat_cpu_waits;

   /* One channel per screen (G0a): every context submits here, serialised
    * by submit_mutex. Engines as NVK's graphics queue (nvk_queue.c:437-440). */
   simple_mtx_t submit_mutex;
   uint64_t channel;
   uint64_t last_token; /* newest submission on the channel (submit_mutex) */
   struct nvrm_bo zero_page;

   /* Shader heap: contiguous growable arena (NVRM_SHADER_ARENA_*) in VRAM
    * as NVK's dev->shader_heap (nvk_device.c:259-262): BAR1-mapped and
    * CPU-written with a full BAR, else uploaded by the copy engine
    * (nvk_heap_upload, nvk_heap.c:115-150). shader_base = SET_PROGRAM_REGION. */
   simple_mtx_t shader_mutex;
   uint64_t shader_base;      /* VA reservation start */
   uint64_t shader_committed; /* bytes backed by shader_mem[0..count) */
   unsigned shader_mem_count;
   struct nvrm_bo shader_mem[NVRM_SHADER_ARENA_MAX_MEMS];
   struct util_vma_heap shader_vma;
   /* Deleted shader ranges {va, size, token}: back to shader_vma once the
    * submission `token` (the last one that could use them) completed. */
   struct util_dynarray shader_pending;

   /* Bumped after every shader code upload (shader_mutex held): contexts
    * whose last instruction-cache invalidate saw an older value invalidate
    * before their next draw (NVRM_INVAL_INSTR). */
   uint64_t shader_gen;

   /* Released STREAM/STAGING buffer memory (u_upload buffers, transfer
    * staging) kept for reuse once its last submission completed, so a
    * steady frame makes no RM allocation or CPU-mapping calls: the
    * recycling NVK's command pools do for upload memory
    * (nvk_cmd_pool.c nvk_cmd_pool_alloc_mem / free_mem lists).
    * Entries: struct nvrm_bo_cache_entry. */
   simple_mtx_t bo_cache_mutex;
   struct util_dynarray bo_cache;
   uint64_t bo_cache_bytes;

   /* NAK's options plus lower_uniforms_to_ubo, so st/mesa hands us
    * load_ubo(0, offset) for the default uniform block, minus doubles
    * lowering (see nvrm_screen_create). */
   struct nir_shader_compiler_options nir_options;

   /* In-memory program cache (nvrm_shader_cache.c): cache key ->
    * struct nvrm_shader_bin, one table reference per entry, for the
    * screen's lifetime (radeonsi's sscreen->shader_cache). */
   simple_mtx_t shader_bins_mutex;
   struct hash_table *shader_bins;

   /* Shader compiles (nvrm_shader_compile_job): one queue per screen, as
    * every gallium driver with a compile queue (iris_screen.c:820-841,
    * radeonsi si_pipe.c, ir3_gallium.c). Workers hold a reference on the
    * GLSL type singleton through glsl_types_ref. */
   struct util_queue compile_queue;
   bool glsl_types_ref;
};

/* A compiled shader: NAK's output plus what nvrm derived from the NIR,
 * shared (refcounted) between the program cache and every nvrm_shader
 * created from the same key. Immutable once published. */
struct nvrm_shader_bin {
   uint32_t refcnt;
   cache_key key;
   struct nak_shader_info info;
   uint32_t code_size;
   void *code;
   unsigned tex_units, image_units;
   uint32_t cbuf_mask;
   bool uses_desc;
};

struct nvrm_shader {
   /* Compile state (nvrm_compile_shader): `ready` signals when the job has
    * run (or was dropped); until then only the job touches the fields below
    * it. `failure` names why the compile failed (logged by the job); the
    * first nvrm_shader_wait aborts with it. */
   struct util_queue_fence ready;
   struct nvrm_screen *screen;
   nir_shader *nir; /* input, owned until the job takes it */
   bool debug;      /* NVRM_DEBUG_SHADERS at creation: compiled inline */
   const char *failure;

   struct nvrm_shader_bin *bin; /* one reference */
   uint64_t heap_addr, heap_size; /* range in screen->shader_vma */
   unsigned tex_units; /* GL texture units used (BITSET_LAST_BIT) */
   bool uses_desc; /* reads the descriptor cbuf (SSBOs, compute sysvals/UBOs) */
   unsigned image_units; /* GL image units used (BITSET_LAST_BIT) */
   mesa_shader_stage stage;
   uint64_t hdr_addr; /* Volta+: absolute SET_PIPELINE_PROGRAM_ADDRESS;
                       * pre-Volta: offset from SET_PROGRAM_REGION */
   /* Constant buffer slots the shader reads (load_ubo indices; all slots
    * below NVRM_DESC_CBUF_SLOT for a dynamic index), NVK's cbuf_map. */
   uint32_t cbuf_mask;
};

struct nvrm_velems {
   unsigned count;
   struct pipe_vertex_element el[PIPE_MAX_ATTRIBS];
};

struct nvrm_resource {
   struct pipe_resource base;
   struct nvrm_bo bo;
   /* Busy tracking: the newest submission token that referenced this
    * resource (atomic max over all contexts of the screen; 0 = never). */
   uint64_t last_token;
   uint32_t stride; /* linear resources: bytes per row, 0 for buffers */
   bool tiled;      /* G1: block-linear VRAM image laid out by nil */
   struct nil_image image;
   /* Blackwell combined Z/S formats: depth in `image`/`bo`, stencil in a
    * second S8 plane (NVK separate_zs, nvk_image.c:1013-1018). */
   bool separate_zs;
   struct nil_image stencil_image;
   struct nvrm_bo stencil_bo;
   /* G2 dma-buf: an imported LINEAR resource also carries a nil linear
    * image (TIC for sampling); has_modifier = the layout is the one named
    * by `modifier` (imported, or created with modifiers). */
   bool linear_image;
   bool imported;
   bool has_modifier;
   uint64_t modifier;
   /* One plane of an imported YUV dma-buf (dri2 per-plane lowering): the
    * plane starts this many bytes into the bound allocation (EGL plane
    * offset). Sampler-only; 0 for everything else. */
   uint32_t plane_offset_B;
   /* Linear colour buffers: block-linear render shadow (NVK's
    * linear_tiled_shadow, nvk_image.c:975-991), created on first use by a
    * framebuffer that cannot render linear (nvrm_fb_shadow_begin). */
   struct pipe_resource *render_shadow;
   /* STREAM/STAGING buffer memory goes back to screen->bo_cache on destroy
    * (domain = its placement). */
   bool recycle;
   enum nvrm_domain domain;
};

struct nvrm_sampler_view {
   struct pipe_sampler_view base;
   uint32_t tic[8];
};

struct nvrm_sampler {
   uint32_t tsc[8];
   /* tsc with the sRGB border fields for an sRGB view whose swizzle moves
    * alpha into R/G/B (nvrm_tsc_for_view). */
   uint32_t tsc_srgb_alpha[3][8];
};

#define NVRM_TEX_SLOT_HASH 8192 /* power of two, > NVRM_TEX_POOL_SLOTS */
struct nvrm_tex_slot {
   uint32_t words[16]; /* TIC then TSC */
   uint32_t slot;
   uint32_t gen;       /* == ctx->tex_slot_gen: valid in the current pool */
};

struct nvrm_bo_cache_entry {
   struct nvrm_bo bo;
   enum nvrm_domain domain;
   uint64_t token; /* last submission that used the memory */
};

struct nvrm_fence {
   struct pipe_reference reference;
   uint64_t token; /* nv_rm submission token on the screen channel */
   int fd;         /* imported sync_file (create_fence_fd), else -1 */
};

struct nvrm_context {
   struct pipe_context base;
   /* Current push chunk (push_bo = push_chunk->bo). */
   struct nvrm_bo push_bo;
   struct nvrm_push_chunk *push_chunk;
   /* Chunks (struct nvrm_push_chunk *) filled in the open batch, and chunks
    * not in it (idle or in flight) in submission order, oldest first. */
   struct util_dynarray push_batch_chunks, push_free_chunks;
   unsigned push_chunk_count;
   struct nvrm_batch_slot ring[NVRM_BATCH_RING];
   unsigned ring_cur;
   uint64_t last_token; /* newest submission of this context, 0 = none */
   struct { uint64_t seq, token; } seq_token[NVRM_SEQ_RING];
   uint32_t *push_start; /* first dword not yet submitted */
   uint32_t *push_cur;
   /* GPFIFO entries recorded before push_start in this batch (struct
    * NvRmExec): closed push_bo segments and pre-Turing indirect data
    * (nvrm_push_indirect), as NVK's cmd->pushes (nvk_cmd_buffer.c:150-200). */
   struct util_dynarray execs;
   struct pipe_framebuffer_state fb;
   /* D1: NVK's linear rendering rule for `fb` (nvk_rendering_linear,
    * nvk_cmd_draw.c:935-963). When it does not hold, the linear colour
    * buffers in shadow_mask render through their block-linear shadows:
    * shadow_loaded = the shadows hold the linear contents (copied in, fully
    * cleared or invalidated) in this batch, shadow_written = a framebuffer
    * emission targeted them since the last copy-out. */
   bool render_linear;
   uint8_t shadow_mask;
   uint8_t shadow_discarded; /* shadow_mask bits invalidated since binding */
   bool shadow_loaded, shadow_written;

   /* G0b draw state (emitted in full before every draw). */
   struct nvrm_shader *vs, *gs, *fs; /* gs optional (OES_geometry_shader) */
   struct nvrm_shader *cs; /* GLES 3.1 compute (nvrm_compute.c) */
   struct pipe_shader_buffer ssbo[MESA_SHADER_STAGES][NVRM_MAX_SSBOS];
   uint32_t ssbo_writable[MESA_SHADER_STAGES];
   struct nvrm_velems *velems;
   struct pipe_blend_state *blend;
   struct pipe_rasterizer_state *rast;
   struct pipe_depth_stencil_alpha_state *dsa;
   struct pipe_viewport_state viewport;
   struct pipe_scissor_state scissor;
   struct pipe_blend_color blend_color;
   struct pipe_stencil_ref stencil_ref;
   unsigned sample_mask;
   struct pipe_constant_buffer cb[MESA_SHADER_STAGES][NVRM_MAX_CBUFS];
   /* Slots whose hardware binding may differ from cb[][] (set by
    * set_constant_buffer and by every flush), bound by the next draw if
    * its shader reads them: NVK's cbuf_groups[].dirty
    * (nvk_cmd_draw.c:3553-3557, 3609). */
   uint32_t cb_dirty[MESA_SHADER_STAGES];
   /* Graphics stage slot 0 holds user constants in the driver cbuf. */
   bool cb_user[MESA_SHADER_STAGES];
   struct nvrm_bo drv_cb;
   /* CPU copy of what LOAD_CONSTANT_BUFFER put into drv_cb, per graphics
    * stage: only differing dwords are loaded (stock loads only the written
    * range). valid = bytes known. */
   uint32_t *drv_cb_uniforms[NVRM_GFX_STAGES];
   uint32_t drv_cb_uniforms_valid_B[NVRM_GFX_STAGES];
   uint32_t drv_cb_handles[NVRM_GFX_STAGES][NVRM_DRV_CB_HANDLES_B / 4];
   uint32_t drv_cb_handles_valid_B[NVRM_GFX_STAGES];
   uint32_t inval;          /* NVRM_INVAL_* pending */
   uint64_t shader_gen_seen; /* screen->shader_gen at the last INSTR invalidate */
   struct pipe_vertex_buffer vb[PIPE_MAX_ATTRIBS];
   unsigned num_vb;
   unsigned emitted_vb_count;

   /* G1 textures. */
   struct pipe_sampler_view *views[MESA_SHADER_STAGES][NVRM_MAX_TEX_UNITS];
   struct nvrm_sampler *samplers[MESA_SHADER_STAGES][NVRM_MAX_TEX_UNITS];
   struct nvrm_bo tex_pool; /* current slot's TIC pool, then TSC pool */
   uint32_t tex_pool_next;
   /* Entries [tex_pool_written, tex_pool_next) are only in tex_shadow yet:
    * the next nvrm_tex_pool_write puts them into the pool. */
   uint32_t tex_pool_written;
   uint32_t (*tex_shadow)[16]; /* per slot: TIC words, then TSC words */
   /* (TIC, TSC) pairs already written into the current batch's pool ->
    * slot: a texture bind reuses its entry instead of writing a new one
    * (stock writes TIC/TSC only on first use).
    * Emptied when the pool rotates (tex_pool_next back to 0). */
   struct nvrm_tex_slot *tex_slots;
   uint32_t tex_slot_gen; /* entries of older generations are empty */
   uint64_t cs_pools_seq; /* flush_seq + 1 of the batch that set the compute pools */
   uint32_t default_tsc[8];
   uint32_t null_tic[8];
   /* GLES 3.1 shader images: bound views + their storage headers. */
   struct pipe_image_view images[MESA_SHADER_STAGES][NVRM_MAX_IMAGES];
   uint32_t image_tic[MESA_SHADER_STAGES][NVRM_MAX_IMAGES][8];

   uint32_t dirty;

   /* G1 local memory (scratch), as NVK's nvk_slm_area: grows only. */
   struct nvrm_bo slm;
   uint32_t slm_bytes_per_lane, slm_crs_per_warp;
   uint64_t slm_per_warp;
   uint64_t slm_per_tpc; /* compute SET_SHADER_LOCAL_MEMORY_*_THROTTLED */

   /* Shader heap ranges deleted by this context: handed to
    * screen->shader_pending with the token of the next flush. */
   struct util_dynarray shader_frees;

   struct blitter_context *blitter; /* G1: blits through u_blitter draws */

   /* G1 GLES3: queries, conditional render, transform feedback. */
   uint64_t flush_seq; /* incremented by every flush */
   bool queries_enabled;
   struct nvrm_query *occlusion_query;
   struct pipe_query *cond_query;
   bool cond_cond;
   unsigned cond_mode;
   struct pipe_stream_output_target *so[NVRM_MAX_SO];
   unsigned num_so;
   uint32_t so_load[NVRM_MAX_SO]; /* write pointer to load; UINT32_MAX = counter */
   bool so_active; /* SET_STREAM_OUTPUT TRUE emitted and not paused */

   /* G2 fence_server_sync: owned sync_file fds the next submission waits
    * for on the GPU side (nv_rm_submit_waits). */
   struct util_dynarray wait_fds;

   /* Resources referenced by pushes not yet submitted (a pointer set, each
    * holding one reference): stamped with the batch's token at flush and
    * released after submission (the backend defers frees past submitted
    * work). Membership = unsubmitted use by this context
    * (nvrm_resource_busy). */
   struct set *keep;
};

static inline struct nvrm_screen *
nvrm_screen(struct pipe_screen *s)
{
   return (struct nvrm_screen *)s;
}

static inline struct nvrm_resource *
nvrm_resource(struct pipe_resource *r)
{
   return (struct nvrm_resource *)r;
}

static inline struct nvrm_context *
nvrm_context(struct pipe_context *c)
{
   return (struct nvrm_context *)c;
}

/* Constant buffer size granularity, as nvk_min_cbuf_alignment
 * (nvk_physical_device.h:61-64): every SET_CONSTANT_BUFFER_SELECTOR_A size
 * NVK binds is a multiple of this (nvk_cmd_draw.c:3558-3559; nvc0 aligns
 * to 0x100 too, nvc0_state.c:838). Measured on GM200 (GTX TITAN X):
 * with a 16-B-aligned 0x160-B cbuf 0, reads at >= 0x100 returned 0.
 * Bindings start 256-B aligned in 4 KiB-aligned BOs, so the padded
 * range never leaves the BO. */
static inline uint32_t
nvrm_cbuf_size(const struct nvrm_screen *screen, uint32_t size)
{
   const uint32_t granule = screen->info.cls_eng3d >= 0xC597 /* TURING_A */ ? 64 : 256;
   return align(size, granule);
}

/* Loud failure: every unsupported path names itself. */
#define nvrm_error(...) mesa_loge("nvrm: " __VA_ARGS__)

bool nvrm_shader_heap_grow(struct nvrm_screen *screen);
void nvrm_shader_heap_write(struct nvrm_screen *screen, uint64_t va, const void *data,
                            uint64_t size);
bool nvrm_bo_alloc(struct nvrm_screen *screen, uint64_t size, enum nvrm_domain domain,
                   struct nvrm_bo *bo);
/* Buffer copies on the copy engine as nvk_CmdCopyBuffer2 (nvk_cmd_copy.c:
 * 369-410): 128 KiB NON_PIPELINED pitch launches, 10 dwords each.
 * nvrm_push_copy_buffer records into the context's batch. */
static inline uint32_t
nvrm_copy_buffer_dw(uint64_t size_B)
{
   return 10u * (uint32_t)DIV_ROUND_UP(MAX2(size_B, 1), 1u << 17);
}
void nvrm_emit_copy_buffer(struct nv_push *p, uint64_t dst, uint64_t src, uint64_t size_B);
void nvrm_push_copy_buffer(struct nvrm_context *ctx, uint64_t dst, uint64_t src,
                           uint64_t size_B);
void nvrm_bo_free(struct nvrm_screen *screen, struct nvrm_bo *bo);

/* Asynchronous batch submission on the screen channel (nv_rm_submit_waits;
 * ring pressure = wait for the oldest batch the backend names, retry).
 * *token names the batch's GPU completion. */
bool nvrm_submit_async(struct nvrm_screen *screen, const struct NvRmExec *execs,
                       uint32_t exec_count, const int *wait_fds, uint32_t wait_count,
                       uint64_t *token);
/* CPU wait for a submission token (0 = nothing); false + named error on
 * failure or device loss. nvrm_token_done polls. */
bool nvrm_wait_token(struct nvrm_screen *screen, uint64_t token);
bool nvrm_token_done(struct nvrm_screen *screen, uint64_t token);
/* Token of the batch with this flush_seq (a newer one when evicted: its
 * completion implies the older's). The batch must have been flushed. */
uint64_t nvrm_seq_token(const struct nvrm_context *ctx, uint64_t seq);
/* CPU access to a resource: flush it out of this context's batch if needed,
 * then wait for its last GPU use. */
void nvrm_resource_wait_idle(struct nvrm_context *ctx, struct nvrm_resource *res);
bool nvrm_resource_busy(struct nvrm_context *ctx, struct nvrm_resource *res);

/* Submit [start, end) dwords at GPU address va on the screen channel and wait
 * for completion. Returns false (named error) on any backend failure. */
bool nvrm_submit_wait(struct nvrm_screen *screen, uint64_t va, uint32_t size_B,
                      uint64_t *token);
/* Same, with borrowed sync_file fds the GPU waits for first; size_B 0 is an
 * empty batch (a completion token only). */
bool nvrm_submit_wait_fds(struct nvrm_screen *screen, uint64_t va, uint32_t size_B,
                          const int *wait_fds, uint32_t wait_count, uint64_t *token);
/* Same for a batch of GPFIFO entries (count 0 = empty batch). */
struct NvRmExec;
bool nvrm_submit_execs_wait_fds(struct nvrm_screen *screen, const struct NvRmExec *execs,
                                uint32_t exec_count, const int *wait_fds,
                                uint32_t wait_count, uint64_t *token);

struct nv_push;
void nvrm_push_reserve(struct nvrm_context *ctx, uint32_t dw);
struct nv_push nvrm_push_begin(struct nvrm_context *ctx, uint32_t dw);
void nvrm_push_end(struct nvrm_context *ctx, struct nv_push *p);
/* nvk_cmd_buffer_push_indirect (nvk_cmd_buffer.c:187-200): ends the open
 * segment at p->end as an incomplete GPFIFO entry, then queues [addr,
 * addr + range) as a NO_PREFETCH entry; p continues in a new segment.
 * The caller grew the last method's count by range / 4
 * (nv_push_update_count) and reserved p's space up front. */
void nvrm_push_indirect(struct nvrm_context *ctx, struct nv_push *p,
                        uint64_t addr, uint32_t range);
/* Recorded and not yet submitted work. */
bool nvrm_has_pending(const struct nvrm_context *ctx);
void nvrm_keep(struct nvrm_context *ctx, struct pipe_resource *res);
bool nvrm_kept(const struct nvrm_context *ctx, const struct pipe_resource *res);
void nvrm_init_draw_functions(struct pipe_context *pctx);
void nvrm_slm_ensure(struct nvrm_context *ctx, uint32_t bytes_per_lane, uint32_t crs_per_warp);
struct nvrm_shader *nvrm_compile_shader(struct pipe_context *pctx, nir_shader *nir);
void nvrm_free_shader(struct nvrm_context *ctx, struct nvrm_shader *shader);
void nvrm_shader_wait(struct nvrm_shader *shader);

/* nvrm_shader_cache.c */
extern const struct nak_fs_key nvrm_nak_fs_key;
void nvrm_shader_cache_init(struct nvrm_screen *screen);
void nvrm_shader_cache_fini(struct nvrm_screen *screen);
void nvrm_shader_cache_key(struct nvrm_screen *screen, nir_shader *nir, cache_key key);
struct nvrm_shader_bin *nvrm_shader_cache_find(struct nvrm_screen *screen, const cache_key key,
                                               mesa_shader_stage stage);
struct nvrm_shader_bin *nvrm_shader_cache_add(struct nvrm_screen *screen, const cache_key key,
                                              struct nvrm_shader_bin *bin);
void nvrm_shader_bin_unref(struct nvrm_shader_bin *bin);

/* nvrm_mme.c: MME macros for indirect draws/dispatches: Turing+ DMA-read
 * the commands (mme_tu104_read_fifoed), older classes take them as push
 * data (nvrm_push_indirect), as NVK. */
enum nvrm_mme {
   NVRM_MME_DRAW_INDIRECT,
   NVRM_MME_DRAW_INDEXED_INDIRECT,
   NVRM_MME_DISPATCH_INDIRECT,
   NVRM_MME_SET_PRIV_REG,
   NVRM_MME_DRAW,
   NVRM_MME_DRAW_INDEXED,
   NVRM_MME_COUNT,
};
bool nvrm_mme_dma(const struct nvrm_screen *screen);
/* SET_MME_SHADOW_SCRATCH slot of the pre-Turing draw macros' BEGIN word
 * (NVK_MME_SCRATCH_DRAW_BEGIN, written by the CPU, nvk_cmd_draw.c:1920). */
uint32_t nvrm_mme_scratch_draw_begin_mthd(void);
void nvrm_push_mme_init(const struct nvrm_screen *screen, struct nv_push *p);

/* nvrm_compute.c: compute launches, shader buffers, memory barriers. */
void nvrm_init_compute_functions(struct pipe_context *pctx);
void nvrm_compute_context_destroy(struct nvrm_context *ctx);
void nvrm_push_compute_init(const struct nvrm_screen *screen, struct nv_push *p);
bool nvrm_lower_buffers(nir_shader *nir);
void nvrm_emit_ssbo_desc(struct nvrm_context *ctx, struct nv_push *p,
                         mesa_shader_stage stage, uint32_t group);
void nvrm_draw_context_destroy(struct nvrm_context *ctx);

/* Vertex attribute format (SET_VERTEX_ATTRIBUTE_A widths/type). */
struct nvrm_va_format {
   uint32_t widths, type;
   bool swap_rb;
};
bool nvrm_get_va_format(enum pipe_format format, struct nvrm_va_format *out);

void nvrm_emit_color_target(struct nvrm_context *ctx, struct nv_push *p,
                            const struct pipe_framebuffer_state *fb);
void nvrm_emit_sample_layout(struct nvrm_context *ctx, struct nv_push *p,
                             const struct pipe_framebuffer_state *fb);
void nvrm_emit_zs_target(struct nvrm_context *ctx, struct nv_push *p,
                         const struct pipe_framebuffer_state *fb);
void nvrm_fb_shadow_begin(struct nvrm_context *ctx);
void nvrm_fb_shadow_end(struct nvrm_context *ctx);
void nvrm_copy_image_linear(struct nvrm_context *ctx, struct nvrm_resource *res,
                            unsigned level, const struct pipe_box *box,
                            uint64_t linear_va, uint32_t linear_stride,
                            uint64_t linear_layer_stride, bool to_image);

bool nvrm_texture_context_init(struct nvrm_context *ctx);
void nvrm_texture_context_destroy(struct nvrm_context *ctx);
void nvrm_init_texture_functions(struct pipe_context *p);
void nvrm_emit_tex_pools(struct nvrm_context *ctx, struct nv_push *p);
void nvrm_texture_reserve(struct nvrm_context *ctx, unsigned slots);
unsigned nvrm_tex_pool_write_dw(unsigned slots);
void nvrm_tex_pool_write(struct nvrm_context *ctx, struct nv_push *p, bool compute);
struct nvrm_shader;
uint64_t nvrm_upload_tex_handles(struct nvrm_context *ctx, mesa_shader_stage stage,
                                 const struct nvrm_shader *shader, uint32_t *size_B);
void nvrm_emit_textures(struct nvrm_context *ctx, struct nv_push *p,
                        mesa_shader_stage stage, uint32_t group,
                        const struct nvrm_shader *shader);
void nvrm_emit_invalidates(struct nvrm_context *ctx, struct nv_push *p);
/* Driver cbuf (nvrm_draw.c): stage index for VS/GS/FS, -1 otherwise. */
static inline int
nvrm_gfx_stage(mesa_shader_stage s)
{
   return s == MESA_SHADER_VERTEX ? 0 : s == MESA_SHADER_GEOMETRY ? 1 :
          s == MESA_SHADER_FRAGMENT ? 2 : -1;
}
/* LOAD_CONSTANT_BUFFER of the dwords of data[0..size_B) that differ from
 * shadow, into drv_cb at stage area + base_B (base_B >= NVRM_DRV_CB_UNIFORM_B:
 * the handle table); shadow and *valid_B (bytes of the area equal to the
 * shadow) updated. nvrm_drv_cb_load_dw: push dwords it needs at most. */
uint32_t nvrm_drv_cb_load_dw(uint32_t size_B);
void nvrm_drv_cb_load(struct nvrm_context *ctx, struct nv_push *p, int gfx_stage,
                      uint32_t base_B, uint32_t *shadow, uint32_t *valid_B,
                      const uint32_t *data, uint32_t size_B);
/* Fills handles[] for `shader` (writes new TIC/TSC pool entries), returns
 * the table size in dwords (0 = none). */
unsigned nvrm_tex_handles(struct nvrm_context *ctx, mesa_shader_stage stage,
                          const struct nvrm_shader *shader, uint32_t *handles);
/* Buffer memory recycling (nvrm_resource.c). */
bool nvrm_bo_cache_get(struct nvrm_screen *screen, uint64_t size, enum nvrm_domain domain,
                       struct nvrm_bo *bo);
void nvrm_bo_cache_fini(struct nvrm_screen *screen);
void nvrm_lower_textures(nir_shader *nir);
void nvrm_lower_images(nir_shader *nir);
void nvrm_init_image_functions(struct pipe_context *p);
void nvrm_image_context_destroy(struct nvrm_context *ctx);
void nvrm_copy_image_linear_sync(struct nvrm_context *ctx, struct nvrm_resource *res,
                                 unsigned level, const struct pipe_box *box,
                                 uint64_t linear_va, uint32_t linear_stride,
                                 uint64_t linear_layer_stride, bool to_image);
void nvrm_push_render_wfi(struct nvrm_context *ctx);
void nvrm_copy_image_upload(struct nvrm_context *ctx, struct nvrm_resource *res,
                            struct pipe_resource *keep_res, unsigned level,
                            const struct pipe_box *box, struct pipe_resource *staging,
                            uint32_t linear_stride, uint64_t linear_layer_stride);
bool nvrm_bo_alloc_image(struct nvrm_screen *screen, const struct nil_image *image,
                         struct nvrm_bo *bo);

void nvrm_init_query_functions(struct pipe_context *p);
void nvrm_query_context_destroy(struct nvrm_context *ctx);
void nvrm_emit_render_enable(struct nvrm_context *ctx, struct nv_push *p);
void nvrm_so_pause(struct nvrm_context *ctx);
void nvrm_so_prepare(struct nvrm_context *ctx);
void nvrm_emit_so(struct nvrm_context *ctx, struct nv_push *p);

void nvrm_init_resource_functions(struct pipe_screen *pscreen);
struct pipe_resource *nvrm_resource_create_modifier(struct pipe_screen *pscreen,
                                                    const struct pipe_resource *templ,
                                                    uint64_t modifier);
void nvrm_init_handle_functions(struct pipe_screen *pscreen);
void nvrm_init_fence_functions(struct pipe_screen *pscreen);
void nvrm_init_context_fence_functions(struct pipe_context *pctx);
void nvrm_init_transfer_functions(struct pipe_context *pctx);
void nvrm_init_copy_functions(struct pipe_context *pctx);
struct pipe_context *nvrm_context_create(struct pipe_screen *pscreen, void *priv,
                                         unsigned flags);

#endif
