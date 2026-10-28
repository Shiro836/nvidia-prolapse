/* SPDX-License-Identifier: MIT
 * Generated from rm-abi/src/lib.rs by mesa/gen-rm-header.py (cbindgen); do not edit.
 * The contract of every entry is documented there. */

#ifndef NV_RM_H
#define NV_RM_H

#include <stddef.h>
#include <stdint.h>
typedef struct NvRmDevice NvRmDevice;

/**
 * [`nv_rm_sync_create`]: a timeline payload.
 */
#define NV_RM_SYNC_TIMELINE 1

/**
 * [`nv_rm_sync_create`]: the payload may be exported.
 */
#define NV_RM_SYNC_SHAREABLE 2

/**
 * [`nv_rm_sync_create`]: the payload is shared with other processes.
 */
#define NV_RM_SYNC_SHARED 4

/**
 * [`nv_rm_sync_status_many`], [`nv_rm_sync_wait_many`]: a submitted, not yet
 * executed signal counts as ready.
 */
#define NV_RM_SYNC_WAIT_PENDING 1

/**
 * [`nv_rm_sync_status_many`], [`nv_rm_sync_wait_many`]: any point instead of
 * all points.
 */
#define NV_RM_SYNC_WAIT_ANY 2

/**
 * [`NvRmExec::flags`]: no prefetch of the segment.
 */
#define NV_RM_EXEC_NO_PREFETCH 1

/**
 * [`NvRmExec::flags`]: the segment continues in the next one.
 */
#define NV_RM_EXEC_INCOMPLETE 2

/**
 * Engine bit (the same bits as Mesa's `nvkmd_engines`).
 */
#define NV_RM_ENGINE_COPY 1

/**
 * Engine bit.
 */
#define NV_RM_ENGINE_2D 2

/**
 * Engine bit.
 */
#define NV_RM_ENGINE_3D 4

/**
 * Engine bit.
 */
#define NV_RM_ENGINE_M2MF 8

/**
 * Engine bit.
 */
#define NV_RM_ENGINE_COMPUTE 16

/**
 * [`nv_rm_alloc`] placement: device memory.
 */
#define NV_RM_PLACEMENT_VRAM 0

/**
 * [`nv_rm_alloc`] placement: system memory.
 */
#define NV_RM_PLACEMENT_GART 1

/**
 * Device-local, never CPU-mapped or exported: RM is asked for compression and
 * big pages (`NVOS32` `COMPR ANY | PAGE_SIZE BIG`), as the stock driver does. RM
 * decides whether compression is granted; the allocation's own VA keeps the
 * requested (uncompressed) kind. Size must be a 64 KiB multiple; the
 * alignment is raised to 64 KiB.
 */
#define NV_RM_PLACEMENT_VRAM_COMPRESSIBLE 2

/**
 * One GPU-internal pool block: 2 MiB of device memory in one physically
 * contiguous huge page, GPU-cached, uncompressed, never CPU-mapped or
 * exported (the NVOS32 request the proprietary driver uses for the memory
 * holding its command buffers' QMDs and launch counters). Size must be
 * exactly 2 MiB and the kind 0; [`nv_rm_map`] fails on it, and it binds whole
 * at a 2 MiB aligned VA.
 */
#define NV_RM_PLACEMENT_VRAM_POOL_BLOCK 3

/**
 * Start of the VA window the caller owns, `[16 GiB, 1 TiB)`; the backend's
 * internal mappings stay below it.
 */
#define NV_RM_VA_START 17179869184

/**
 * End of the caller's VA window.
 */
#define NV_RM_VA_END 1099511627776

/**
 * `NV2080_CTRL_GR_GET_ZCULL_INFO_PARAMS` field order (ctrl2080gr.h), which is
 * also the order of Mesa's `nv_zcull_device_info`.
 */
struct NvRmZcullInfo {
  uint32_t width_align_pixels;
  uint32_t height_align_pixels;
  uint32_t pixel_squares_by_aliquots;
  uint32_t aliquot_total;
  uint32_t region_byte_multiplier;
  uint32_t region_header_size;
  uint32_t subregion_header_size;
  uint32_t subregion_count;
  uint32_t subregion_width_align_pixels;
  uint32_t subregion_height_align_pixels;
};

/**
 * What [`nv_rm_open`] reports about the GPU.
 */
struct NvRmInfo {
  uint32_t vendor_id;
  uint32_t device_id;
  uint32_t chipset;
  uint32_t sm;
  uint32_t pci_domain;
  uint32_t pci_bus;
  uint32_t pci_device;
  uint32_t pci_function;
  uint32_t gpc_count;
  uint32_t tpc_count;
  uint32_t mp_per_tpc;
  uint32_t max_warps_per_mp;
  uint32_t cls_copy;
  uint32_t cls_eng2d;
  uint32_t cls_eng3d;
  uint32_t cls_m2mf;
  uint32_t cls_compute;
  /**
   * NIL uncompressed color PTE kind: Fermi 0xfe, Turing+ 6.
   */
  uint32_t generic_kind;
  uint64_t vram_size;
  uint64_t bar_size;
  /**
   * The GPU name, NUL-terminated.
   */
  char name[64];
  /**
   * ZCULL region geometry (`GR_GET_ZCULL_INFO`). Meaningful only when
   * `has_zcull_info` is 1.
   */
  struct NvRmZcullInfo zcull;
  /**
   * The `GRAPHICS_ZCULL` context-switch buffer size.
   */
  uint32_t zcull_ctxsw_size;
  /**
   * The `GRAPHICS_ZCULL` context-switch buffer alignment.
   */
  uint32_t zcull_ctxsw_align;
  /**
   * 0 when the RM query failed or returned values outside the supported
   * range; the reason is logged at open.
   */
  uint32_t has_zcull_info;
  /**
   * RM's SM ISA code (`NV2080_CTRL_GR_INFO_INDEX_SM_VERSION`, e.g. 0xa04 on
   * `GB20x`); `sm` is the shader model decimal derived from it.
   */
  uint32_t sm_isa;
  /**
   * `NV2080_CTRL_GR_INFO_INDEX_SHADER_PIPE_COUNT`: an input of the compute
   * contexts' `SET_SCG_CONTROL`.
   */
  uint32_t shader_pipe_count;
  /**
   * `NV2080_CTRL_GR_INFO_INDEX_SHADER_PIPE_SUB_COUNT` (the TPC count).
   */
  uint32_t shader_pipe_sub_count;
  /**
   * Contexts that can join another context's channel group
   * ([`nv_rm_context_create_shared`]): from Volta on RM's subcontext count
   * (GR info `MAX_SUBCONTEXT_COUNT`) less VEID 0; before Volta, where
   * joining channels share the group's single GR context, the group's 32
   * channels less one (gm107 `dev_ram.h` `NV_RAMRL_ENTRY_TSG_LENGTH_MAX`).
   */
  uint32_t max_shared_ctx;
  /**
   * Zero.
   */
  uint32_t reserved;
};

/**
 * An allocation made by [`nv_rm_alloc`] or [`nv_rm_import`].
 */
typedef uint64_t NvRmMemoryId;

/**
 * An allocation made by [`nv_rm_alloc`] or [`nv_rm_import`].
 */
struct NvRmAllocation {
  /**
   * The allocation's handle.
   */
  NvRmMemoryId id;
  /**
   * Its size in bytes.
   */
  uint64_t size;
  /**
   * Reserved zero; CPU maps are acquired separately.
   */
  uintptr_t cpu_address;
};

/**
 * A GPU mapping made by [`nv_rm_bind`].
 */
typedef uint64_t NvRmMappingId;

/**
 * Returns a VA range to the caller's allocator. Called at most once per
 * successful [`nv_rm_va_free`], only after RM unmapped every listed mapping,
 * on whichever thread reclaims it (that call, a later backend call, or
 * [`nv_rm_close`], which runs all outstanding releases before returning).
 * Never called if an unmap failed (device lost; the range stays reserved).
 * The backend may hold its resource lock while calling it: it may take only
 * the caller's VA-heap lock, which must never be held across backend calls.
 */
typedef void (*NvRmVaRelease)(void *data, uint64_t address, uint64_t size);

/**
 * A context (one channel) made by [`nv_rm_context_create`].
 */
typedef uint64_t NvRmContextId;

/**
 * One push buffer segment to execute.
 */
struct NvRmExec {
  /**
   * GPU VA of the segment.
   */
  uint64_t address;
  /**
   * Size in bytes.
   */
  uint32_t size_B;
  /**
   * `NV_RM_EXEC_*`.
   */
  uint32_t flags;
};

/**
 * A context's GPU completion point, returned by submission.
 */
typedef uint64_t NvRmToken;

/**
 * A synchronization payload made by [`nv_rm_sync_create`].
 */
typedef uint64_t NvRmSyncId;

/**
 * One (payload, value) pair of a wait or signal list.
 */
struct NvRmSyncPoint {
  /**
   * The payload.
   */
  NvRmSyncId sync;
  /**
   * The timeline value (ignored for binary payloads).
   */
  uint64_t value;
};

/**
 * Opens the GPU at a PCI address: `*out` receives the device, `*info`
 * what RM reports about it.
 */
extern int32_t nv_rm_open(uint32_t domain,
                          uint32_t bus,
                          uint32_t device,
                          uint32_t function,
                          NvRmDevice **out,
                          struct NvRmInfo *info);

/**
 * Nonblocking snapshot: 0 = live, 1 = lost (also for NULL). One atomic
 * load. The caller keeps the device alive for this call, as with other
 * entries.
 */
extern int32_t nv_rm_is_lost(const NvRmDevice *device);

/**
 * CPU-only software-loss publication: atomics, CPU waiter notification
 * and an eventfd wake. May take the waiters' condition-variable lock
 * only; never waits for GPU progress, takes submission/resource locks or
 * issues RM calls. The caller retains the device throughout the call.
 */
extern void nv_rm_mark_lost(const NvRmDevice *device);

/**
 * Destroys a device made by [`nv_rm_open`]; NULL is ignored. No call on
 * it may be in flight or follow.
 */
extern void nv_rm_close(NvRmDevice *device);

/**
 * `pte_kind` 0: plain memory (RM picks the layout). Non-zero: uncompressed
 * block-linear memory of that kind (the device's generic kind only, VRAM
 * or GART); RM must grant exactly that kind. `tile_mode` must be 0.
 */
extern int32_t nv_rm_alloc(NvRmDevice *device,
                           uint64_t size,
                           uint64_t alignment,
                           uint32_t placement,
                           uint32_t pte_kind,
                           uint32_t tile_mode,
                           struct NvRmAllocation *out);

/**
 * Separate internal/client map references share one lazy mapping.
 */
extern int32_t nv_rm_map(NvRmDevice *device, NvRmMemoryId memory, uintptr_t *address);

/**
 * Drops one map reference.
 */
extern int32_t nv_rm_unmap(NvRmDevice *device, NvRmMemoryId memory);

/**
 * Drops the allocation id now (CPU mapping must be released). The RM
 * object is freed after the work submitted before this call retires.
 */
extern int32_t nv_rm_free(NvRmDevice *device, NvRmMemoryId memory);

/**
 * Maps `[memory_offset, memory_offset + size)` of `memory` at `address`
 * with `pte_kind`; `*mapping` receives the mapping.
 */
extern int32_t nv_rm_bind(NvRmDevice *device,
                          NvRmMemoryId memory,
                          uint64_t memory_offset,
                          uint64_t size,
                          uint64_t address,
                          uint32_t pte_kind,
                          NvRmMappingId *mapping);

/**
 * Immediate RM unmap, no GPU wait: only for application-ordered binds,
 * where Vulkan forbids unbinding/rebinding memory still used by pending
 * work.
 */
extern int32_t nv_rm_unbind(NvRmDevice *device, NvRmMappingId mapping);

/**
 * Frees a VA reservation `[address, address + size)` and all its mappings
 * (ids from [`nv_rm_bind`], each inside the range). On success the backend
 * owns the mappings and calls `release(data, address, size)` once,
 * deferred as described in the crate documentation (immediately if
 * `count` is 0 or nothing is outstanding). On failure nothing was taken:
 * the caller keeps the range reserved.
 */
extern int32_t nv_rm_va_free(NvRmDevice *device,
                             const NvRmMappingId *mappings,
                             uint32_t count,
                             uint64_t address,
                             uint64_t size,
                             NvRmVaRelease release,
                             void *data);

/**
 * Creates a context with its own channel group for the `NV_RM_ENGINE_*`
 * bits in `engines`.
 */
extern int32_t nv_rm_context_create(NvRmDevice *device, uint32_t engines, NvRmContextId *context);

/**
 * Like [`nv_rm_context_create`], but the new context's channel joins the
 * TSG of context `share` on the lowest free subcontext (VEID) instead of
 * getting a TSG of its own, so work on both contexts can run on the GR
 * engine at the same time. Only the compute engine (16) is supported, on
 * Volta and later; anything else fails with a named reason. The TSG
 * lives until the last context in it is destroyed.
 */
extern int32_t nv_rm_context_create_shared(NvRmDevice *device,
                                           uint32_t engines,
                                           NvRmContextId share,
                                           NvRmContextId *context);

/**
 * Destroys a context; bounded even when its work never completes.
 */
extern int32_t nv_rm_context_destroy(NvRmDevice *device, NvRmContextId context);

/**
 * All pushes (including an incomplete push and its following
 * continuation) execute on the persistent channel in order. Success
 * means full GPU completion.
 */
extern int32_t nv_rm_exec(NvRmDevice *device,
                          NvRmContextId context,
                          const struct NvRmExec *pushes,
                          uint32_t count);

/**
 * Async submission: 0 means queued, `*token` names GPU completion; 1
 * (`VK_NOT_READY`) means no work/waits were queued and `*token` names the
 * oldest batch to wait before retrying. Other results are negative
 * errors. Borrowed wait fds remain owned by the caller; accepted waits
 * retain owned descriptor duplicates and cancellable userspace gates, not
 * kernel WAIT callbacks. Empty batches are valid and issue a completion.
 */
extern int32_t nv_rm_submit_waits(NvRmDevice *device,
                                  NvRmContextId context,
                                  const struct NvRmExec *pushes,
                                  uint32_t count,
                                  int drm_fd,
                                  const int *wait_fds,
                                  uint32_t wait_count,
                                  NvRmToken *token);

/**
 * [`nv_rm_submit_waits`] without waits.
 */
extern int32_t nv_rm_submit(NvRmDevice *device,
                            NvRmContextId context,
                            const struct NvRmExec *pushes,
                            uint32_t count,
                            NvRmToken *token);

/**
 * Nonblocking completion query; `*complete` is 0 or 1.
 */
extern int32_t nv_rm_poll(NvRmDevice *device,
                          NvRmContextId context,
                          NvRmToken token,
                          uint32_t *complete);

/**
 * Waits on the caller. `u64::MAX` is infinite, zero polls. Returns 0 when
 * complete, 2 (`VK_TIMEOUT`) at deadline, negative on error.
 */
extern int32_t nv_rm_wait(NvRmDevice *device,
                          NvRmContextId context,
                          NvRmToken token,
                          uint64_t timeout_ns);

/**
 * Exports a real pending GPU fence; the returned fd belongs to the caller.
 */
extern int32_t nv_rm_context_fence(NvRmDevice *device,
                                   NvRmContextId context,
                                   NvRmToken token,
                                   int drm_fd,
                                   int *sync_fd);

/**
 * Polls channel and retained incoming dependency errors after Vulkan
 * waits. Returns `VK_SUCCESS` or `VK_ERROR_DEVICE_LOST`, preserving
 * producer failures.
 */
extern int32_t nv_rm_check_status(NvRmDevice *device);

/**
 * The GPU timestamp.
 */
extern int32_t nv_rm_timestamp(NvRmDevice *device, uint64_t *timestamp);

/**
 * Exports an allocation as a dma-buf; the fd belongs to the caller.
 */
extern int32_t nv_rm_export(NvRmDevice *device, NvRmMemoryId memory, int drm_fd, int *dma_buf_fd);

/**
 * Imports a dma-buf (borrowed) as an allocation.
 */
extern int32_t nv_rm_import(NvRmDevice *device,
                            int drm_fd,
                            int dma_buf_fd,
                            struct NvRmAllocation *out);

/**
 * Creates a payload (`NV_RM_SYNC_*` flags). Sync identities belong to
 * this device. Payloads accepted by submission are retained independently
 * of identity reset/move/destruction. Mutation is transactional: a failed
 * create/reset/move/import leaves live state intact.
 */
extern int32_t nv_rm_sync_create(NvRmDevice *device,
                                 uint32_t flags,
                                 uint64_t initial_value,
                                 NvRmSyncId *sync);

/**
 * Destroys a payload.
 */
extern int32_t nv_rm_sync_destroy(NvRmDevice *device, NvRmSyncId sync);

/**
 * Historical name kept for ABI stability: disposal is immediate on the
 * caller. Accepted submissions keep their own payload slot references
 * until retired.
 */
extern int32_t nv_rm_sync_destroy_deferred(NvRmDevice *device, NvRmSyncId sync);

/**
 * Resets a binary payload.
 */
extern int32_t nv_rm_sync_reset(NvRmDevice *device, NvRmSyncId sync);

/**
 * Moves a binary payload from `src` to `dst`.
 */
extern int32_t nv_rm_sync_move(NvRmDevice *device,
                               NvRmSyncId dst,
                               NvRmSyncId src,
                               uint32_t shared_source);

/**
 * Binds a deferred (submit-thread) binary signal: `dst` becomes the
 * handle the deferred batch signals, on `src`'s current slot; `src`
 * reports it pending. `*reserved` = 0: `src` is not reservable and
 * nothing changed.
 */
extern int32_t nv_rm_sync_reserve(NvRmDevice *device,
                                  NvRmSyncId dst,
                                  NvRmSyncId src,
                                  uint32_t *reserved);

/**
 * Host signal (a binary payload ignores `value`).
 */
extern int32_t nv_rm_sync_signal(NvRmDevice *device, NvRmSyncId sync, uint64_t value);

/**
 * A timeline payload's value.
 */
extern int32_t nv_rm_sync_get_value(NvRmDevice *device, NvRmSyncId sync, uint64_t *value);

/**
 * Nonblocking status query; `*ready` is 0 or 1. `flags` select
 * pending/any.
 */
extern int32_t nv_rm_sync_status_many(NvRmDevice *device,
                                      const struct NvRmSyncPoint *points,
                                      uint32_t count,
                                      uint32_t flags,
                                      uint32_t *ready);

/**
 * Returns 0 on completion, 2 on caller timeout, negative on failure.
 * Infinite waits use `u64::MAX`. Sleeps on the caller: one waiting thread
 * polls the RM event descriptor and wakes the others; the deadline is the
 * caller's only.
 */
extern int32_t nv_rm_sync_wait_many(NvRmDevice *device,
                                    const struct NvRmSyncPoint *points,
                                    uint32_t count,
                                    uint32_t flags,
                                    uint64_t timeout_ns);

/**
 * Import borrows `fd`. `opaque` selects `OPAQUE_FD` instead of `SYNC_FD`.
 */
extern int32_t nv_rm_sync_import_fd(NvRmDevice *device, NvRmSyncId sync, int fd, uint32_t opaque);

/**
 * Export returns an owned fd. Only explicit export needs a kernel export
 * fence.
 */
extern int32_t nv_rm_sync_export_fd(NvRmDevice *device,
                                    NvRmSyncId sync,
                                    uint32_t opaque,
                                    int drm_fd,
                                    int *fd);

/**
 * Returns `VK_NOT_READY` with the oldest token on ring pressure; no
 * waits, pushes or signals are accepted in that case. Success retains all
 * payloads and returns the actual GPU completion token for this batch.
 */
extern int32_t nv_rm_submit_batch(NvRmDevice *device,
                                  NvRmContextId context,
                                  const struct NvRmExec *pushes,
                                  uint32_t count,
                                  const struct NvRmSyncPoint *waits,
                                  uint32_t wait_count,
                                  const struct NvRmSyncPoint *signals,
                                  uint32_t signal_count,
                                  NvRmToken *token);

/**
 * [`nv_rm_submit_batch`] without signals.
 */
extern int32_t nv_rm_submit_sync(NvRmDevice *device,
                                 NvRmContextId context,
                                 const struct NvRmExec *pushes,
                                 uint32_t count,
                                 const struct NvRmSyncPoint *waits,
                                 uint32_t wait_count,
                                 NvRmToken *token);

/**
 * [`nv_rm_submit_batch`] with signals only.
 */
extern int32_t nv_rm_context_signal(NvRmDevice *device,
                                    NvRmContextId context,
                                    const struct NvRmSyncPoint *signals,
                                    uint32_t count,
                                    NvRmToken *token);

#endif  /* NV_RM_H */
